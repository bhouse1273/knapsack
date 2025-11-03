Got it—here are the three artifacts you asked for.

---

# 1) YAML template — items/groups with a soft team-size target

```yaml
# instance.yaml
version: 1

meta:
  name: "Items-to-Field (Jetson)"
  description: "Block-aware multi-constraint knapsack with soft team-size target"
  seed: 42

globals:
  field_coordinates: [ -6.509, -79.771 ]     # lat, lng
  fuel_rate_per_km: 0.18                     # currency / km
  van_count: 2
  target_team_size: 64                       # workers requested overall
  hard_max_trips: 3                          # max trips per group across all blocks (example)

attributes: # Structure-of-Arrays on device; names used in terms below
  - name: workers            # integer units available for pickup
  - name: dist_garage        # km: garage -> pickup
  - name: dist_pickup_field  # km: pickup -> field
  - name: dist_field_garage  # km: field -> garage
  - name: productivity       # 1..4 scalar multiplier (float)
  - name: pick_cost          # optional fixed pickup cost per item (currency)

items: # tiny demo set; normally loaded from CSV/DB
  - id: A1; attrs: {workers: 11, dist_garage: 15.2, dist_pickup_field: 22.4, dist_field_garage: 30.0, productivity: 3.0, pick_cost: 2.0}
  - id: A2; attrs: {workers:  7, dist_garage: 10.0, dist_pickup_field: 19.1, dist_field_garage: 27.0, productivity: 2.0, pick_cost: 2.0}
  - id: B1; attrs: {workers: 18, dist_garage: 21.9, dist_pickup_field: 12.5, dist_field_garage: 29.0, productivity: 4.0, pick_cost: 3.0}
  - id: B2; attrs: {workers: 12, dist_garage:  9.3, dist_pickup_field: 15.7, dist_field_garage: 24.0, productivity: 2.0, pick_cost: 2.0}
  - id: C1; attrs: {workers: 20, dist_garage:  6.8, dist_pickup_field: 14.3, dist_field_garage: 23.0, productivity: 1.0, pick_cost: 1.0}

knapsacks:  # groups; each block can reuse the same groups (trips) subject to context counters
  - id: group-1
    capacities:
      - {name: seats, value: 16}
  - id: group-2
    capacities:
      - {name: seats, value: 16}

# Constraints apply at one of three scopes: global, block, or knapsack.
# LHS is formed by summing item attributes chosen within the scope.
constraints:
  # Hard seat capacity per assignment (per group, per block)
  - name: seat_capacity
    scope: knapsack
    sense: "<="
  lhs: { attr: workers }          # sum workers in that group
    rhs: { expr: "seats" }          # look up from knapsack.capacities
    soft: false

  # Global soft team target across *all blocks/decisions*
  - name: team_target
    scope: global
    sense: ">="                     # want at least target_team_size
    lhs: { attr: workers }
    rhs: { expr: "globals.target_team_size" }
    soft: true
    penalty:
      weight: 10.0                  # cost units per worker short
      power: 2.0                    # quadratic penalty to strongly punish shortfall

  # Optional: max trips per group across blocks (enforced via context)
  - name: trips_per_group
    scope: knapsack
    sense: "<="
    lhs: { context_counter: "trips_used" }   # provided by ContextUpdater
      group_count: 2
    soft: false

# Objective is a weighted sum here. You can switch to lexicographic/pareto if needed.
objectives:
  mode: "weighted-sum"
  terms:
    # Reward workers with productivity bonus
    - name: "productive_workers"
      weight: 1.0
      sum_over: item
      expr: "workers * (1.0 + 0.15 * (productivity - 1.0))"

  # Subtract fuel cost per selected item, routing: garage→pickup→field→garage
    - name: "fuel_cost"
      weight: -1.0
      sum_over: item
  expr: "(dist_garage + dist_pickup_field + dist_field_garage) * globals.fuel_rate_per_km"

    # Optional fixed pick cost
    - name: "pickup_fixed_cost"
      weight: -1.0
      sum_over: item
      expr: "pick_cost"

blocks:  # Partition items into evaluation windows; context flows across blocks
  - id: block-1
    item_ids: [A1, A2]
    local_caps: []          # example: could add per-block budget if desired
    context_bits: 0b000000
  - id: block-2
    item_ids: [B1, B2]
    local_caps: []
    context_bits: 0b000000
  - id: block-3
    item_ids: [C1]
    local_caps: []
    context_bits: 0b000000

context:
  # These counters are updated when a decision (selection/assignment) is committed for a block.
  counters:
  - name: trips_used        # per group; increment when any item assigned to that group in a block forms a trip
    - name: workers_accum     # global; sum of workers chosen so far

  updater:
    # Pseudocode evaluated on host between blocks; kept simple for clarity.
    on_block_commit: |
      if any(group has ≥ 1 assigned item):
          group.trips_used += 1
      globals.workers_accum += sum_assigned_attr("workers")

solver:
  decision_mode: "assign"       # "select" or "assign"
  engine: "beam_lns"            # or "bnb", "dp", …
  beam_width: 128
  candidates_per_block: 4096
  deterministic: true
  checkpoint_seconds: 30
```

---

# 2) CUDA kernel skeleton — evaluate N_candidates × N_items_in_block

> Notes
> • SoA layout for coalesced reads.
> • Each **thread block** evaluates many candidates; each **warp** accumulates partials across items.
> • Supports weighted-sum objective, soft penalties, and per-knapsack seat capacity.
> • Replace placeholders with your actual device arrays / counts.

```cpp
// eval_block_candidates.cu
#include <cuda_runtime.h>
#include <stdint.h>

enum Sense : uint8_t { LE = 0, GE = 1, EQ = 2 };

struct DeviceSoA {
  // len = N_items_total (but we slice per block using item_index_start/len)
  const float* workers;             // stored as float for accumulation
  const float* dist_garage;
  const float* dist_pickup_field;
  const float* dist_field_garage;
  const float* productivity;
  const float* pick_cost;
};

struct ObjTerm {
  // Weighted-sum objective: weight * sum(expr(item))
  float weight;
  // simple tag-based expression for demo; replace with codegen or fn ptr tables as you generalize
  uint8_t expr_id;  // 0: prod_workers, 1: fuel_cost, 2: pickup_cost
};

struct SoftConstraint {
  Sense sense;
  float rhs;         // resolved RHS for this scope (global for the run); for knapsack-scoped use per-K array
  float weight;      // penalty weight
  float power;       // e.g., 1 or 2
  uint8_t lhs_attr;  // 0: workers (global), others extend as needed
};

struct Knapsack {
  float seats;       // seat capacity per trip
};

struct BlockSlice {
  int item_offset;   // start index in SoA arrays for this block
  int item_count;    // items in this block
};

// Candidate encoding
// For "assign" mode with K knapsacks, store 2-bit or 3-bit per item (0=unassigned, 1..K=group idx).
// Here we assume up to 4 knapsacks → 2 bits per item packed into uint32_t lanes.
struct CandidatePack {
  const uint32_t* lanes;  // length = ceil(2 * item_count / 32.0)
  int bits_per_item;      // 2
  int K;                  // knapsack count
};

// Output
struct EvalOut {
  float* obj;             // size = N_candidates
  float* soft_penalty;    // size = N_candidates (aggregate over all soft constraints)
  // (Optional) write per-constraint violations for debugging
};

__device__ inline float expr_eval(uint8_t expr_id,
                                  int idx,
                                  const DeviceSoA& A,
                                  const float fuel_rate_per_km)
{
  switch (expr_id) {
    case 0: { // productive workers
      float w = A.workers[idx];
      float p = A.productivity[idx];
      return w * (1.0f + 0.15f * (p - 1.0f));
    }
    case 1: { // fuel cost (garage→pickup→field→garage)
      float d = A.dist_garage[idx] + A.dist_pickup_field[idx] + A.dist_field_garage[idx];
      return d * fuel_rate_per_km;
    }
    case 2: { // pickup fixed cost
      return A.pick_cost[idx];
    }
    default:
      return 0.0f;
  }
}

__device__ inline int get_assignment(const CandidatePack& P, int item_id) {
  // 2-bit packed per item
  const int bit = item_id * P.bits_per_item;
  const int lane = bit >> 5;          // /32
  const int shift = bit & 31;         // %32
  uint32_t word = P.lanes[lane];
  int a = (word >> shift) & ((1 << P.bits_per_item) - 1);
  return a; // 0..K (0 = not selected)
}

template<int WARPS_PER_BLOCK = 4>
__global__ void eval_block_candidates_kernel(
    DeviceSoA A,
    BlockSlice S,
    CandidatePack cands,
    const ObjTerm* __restrict__ obj_terms,   // length T
    int T,
    const SoftConstraint* __restrict__ soft_cs, // length Ssoft (global-scope soft only in this skeleton)
    int Ssoft,
    const Knapsack* __restrict__ groups,     // K knapsacks (capacities)
    int K,
    float fuel_rate_per_km,
    EvalOut out,
    int N_candidates)
{
  // grid: (ceil(N_candidates / (WARPS_PER_BLOCK*warpSize)), 1, 1)
  // blockDim: WARPS_PER_BLOCK * 32
  const int tid = threadIdx.x;
  const int warp_id = tid >> 5;        // 0..WARPS_PER_BLOCK-1
  const int lane = tid & 31;

  // assign one warp per candidate (could be more elaborate with persistent loops)
  int cand_base = (blockIdx.x * WARPS_PER_BLOCK) + warp_id;
  if (cand_base >= N_candidates) return;

  // Compute base pointer to this candidate's packed lanes
  // (Assume candidates are stored back-to-back; compute stride externally and pass as part of CandidatePack if needed.)
  CandidatePack P = cands;
  P.lanes += cand_base * ((S.item_count * P.bits_per_item + 31) / 32);

  // Per-warp accumulators
  float obj_sum = 0.0f;

  // For knapsack seat capacity feasibility (hard), we compute per-group worker load within this block.
  // Keep in registers and reduce across warp.
  const int MAX_K = 4; // demo
  float group_load_local[MAX_K] = {0,0,0,0};

  // Stride over items of this block
  for (int i = lane; i < S.item_count; i += 32) {
    const int idx = S.item_offset + i;

    // Assignment 0..K
    int a = get_assignment(P, i);

    if (a > 0 && a <= K) {
      // Objective terms
  // (If you need item counted only once regardless of group, this is fine; if group affects cost, pass group data here.)
      for (int t = 0; t < T; ++t) {
        float val = expr_eval(obj_terms[t].expr_id, idx, A, fuel_rate_per_km);
        obj_sum += obj_terms[t].weight * val;
      }
  // Track seat load per assigned group (a-1 is 0-based)
  group_load_local[a - 1] += A.workers[idx];
    }
  }

  // Warp reductions
  #pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    obj_sum += __shfl_down_sync(0xFFFFFFFF, obj_sum, offset);
    for (int v = 0; v < MAX_K; ++v) {
  group_load_local[v] += __shfl_down_sync(0xFFFFFFFF, group_load_local[v], offset);
    }
  }

  // Lane 0 of the warp finalizes
  if (lane == 0) {
  // HARD feasibility: capacity per group in this block
    bool infeasible = false;
    for (int v = 0; v < K; ++v) {
  if (group_load_local[v] > groups[v].seats + 1e-6f) { infeasible = true; break; }
    }
    float penalty_sum = 0.0f;

    // SOFT constraints (global scope example)
    // Here we only support the global team_target within-this-block contribution.
    // In a full implementation, combine with context counters (workers_accum) on host when committing.
    for (int s = 0; s < Ssoft; ++s) {
      const SoftConstraint& C = soft_cs[s];
      float lhs = 0.0f;
      if (C.lhs_attr == 0) { // workers in this block/candidate
        float block_workers = 0.0f;
  for (int v = 0; v < K; ++v) block_workers += group_load_local[v];
        lhs = block_workers; // host adds context.workers_accum before checking final
      }
      float viol = 0.0f;
      if (C.sense == GE) viol = fmaxf(0.0f, C.rhs - lhs);
      else if (C.sense == LE) viol = fmaxf(0.0f, lhs - C.rhs);
      else /*EQ*/ viol = fabsf(lhs - C.rhs);
      if (viol > 0.0f) {
        penalty_sum += C.weight * powf(viol, C.power);
      }
    }

    float final_obj = (infeasible ? -CUDART_INF_F : obj_sum - penalty_sum);
    out.obj[cand_base] = final_obj;
    out.soft_penalty[cand_base] = penalty_sum;
  }
}

// --- Host wrapper sketch (shape only) ---
void eval_block_candidates(const DeviceSoA& A,
                           const BlockSlice& S,
                           const CandidatePack& P_all,
                           const ObjTerm* d_terms, int T,
                           const SoftConstraint* d_soft, int Ssoft,
                           const Knapsack* d_vans, int K,
                           float fuel_rate,
                           EvalOut out,
                           int N_candidates,
                           cudaStream_t stream)
{
  constexpr int WARPS = 4;
  dim3 block(WARPS * 32);
  dim3 grid((N_candidates + WARPS - 1) / WARPS);
  eval_block_candidates_kernel<WARPS><<<grid, block, 0, stream>>>(
      A, S, P_all, d_terms, T, d_soft, Ssoft, d_vans, K, fuel_rate, out, N_candidates);
}
```

**Compile tip**

```bash
nvcc -O3 -arch=sm_87 -Xptxas -O3 -lineinfo eval_block_candidates.cu -c -o eval_block_candidates.o
```

---

# 3) Chariot snippet — emit JSON config + cost composition

Below is a small Chariot program (using your “function-outside-parens” style) that:

1. Loads `entities.csv`
2. Computes per-item attributes
3. Defines objectives/constraints
4. Partitions into 3 blocks
5. Emits the JSON used by your solver

> Assumptions
> • `csv.read`, `map`, `json.write`, and `rand.seed` are standard Chariot built-ins in your environment.
> • Arithmetic is scalar; strings are quoted; lists use `[ ... ]`.

```lisp
; chariot.knapsack.chr

; ---- Parameters ------------------------------------------------------------
let globals (
  field_lat   -6.509
  field_lng  -79.771
  fuel_rate    0.18
  group_seats    16
  target_team    64
  seed           42
)

; ---- Data loading ---------------------------------------------------------
; entities.csv columns: name,workers,dist_garage,dist_pickup_field,dist_field_garage,productivity
let entities (csv.read "entities.csv")

; Ensure types and add pick_cost default
let items
  (map entities (fn (row)
     (obj
       id               (get row "name")
       attrs            (obj
                          workers            (int (get row "workers"))
                          dist_garage        (float (get row "dist_garage"))
                          dist_pickup_field (float (get row "dist_pickup_field"))
                          dist_field_garage  (float (get row "dist_field_garage"))
                          productivity       (float (get row "productivity"))
                          pick_cost          2.0))))

; ---- Knapsacks (two groups) ------------------------------------------------
let groups
  [ (obj id "group-1" capacities [ (obj name "seats" value globals.group_seats) ])
    (obj id "group-2" capacities [ (obj name "seats" value globals.group_seats) ]) ]

; ---- Constraints -----------------------------------------------------------
let constraints
  [ (obj name "seat_capacity" scope "knapsack" sense "<="
         lhs (obj attr "workers")
         rhs (obj expr "seats")
         soft false)

    (obj name "team_target" scope "global" sense ">="
         lhs (obj attr "workers")
         rhs (obj expr "globals.target_team_size")
         soft true
         penalty (obj weight 10.0 power 2.0))

  (obj name "trips_per_group" scope "knapsack" sense "<="
         lhs (obj context_counter "trips_used")
         rhs (obj expr "globals.hard_max_trips")
         soft false)
  ]

; ---- Objectives (weighted-sum) --------------------------------------------
let objectives
  (obj mode "weighted-sum"
       terms [
         (obj name "productive_workers" weight 1.0  sum_over "item" expr "workers*(1.0+0.15*(productivity-1.0))")
         (obj name "fuel_cost"          weight -1.0 sum_over "item" expr "(dist_garage+dist_pickup_field+dist_field_garage)*globals.fuel_rate_per_km")
         (obj name "pickup_fixed_cost"  weight -1.0 sum_over "item" expr "pick_cost")
       ])

; ---- Blocks: simple round-robin partition into 3 blocks -------------------
let block_count 3
let block_items (fn (all i n)
  (filter all (fn (it idx) (= (% idx n) i))))  ; keep every n-th item starting at offset i

let blocks
  [ (obj id "block-1" item_ids (map (block_items items 0 block_count) (fn (it) it.id)) local_caps [] context_bits 0)
    (obj id "block-2" item_ids (map (block_items items 1 block_count) (fn (it) it.id)) local_caps [] context_bits 0)
    (obj id "block-3" item_ids (map (block_items items 2 block_count) (fn (it) it.id)) local_caps [] context_bits 0)
  ]

; ---- Context updater (opaque to device; host applies between blocks) ------
let context
  (obj counters [ (obj name "trips_used") (obj name "workers_accum") ]
       updater  (obj on_block_commit
                   "if any(group has >=1 assigned item): group.trips_used += 1; globals.workers_accum += sum_assigned_attr('workers')"))

; ---- Solver runtime knobs --------------------------------------------------
let solver
  (obj decision_mode "assign" engine "beam_lns" beam_width 128 candidates_per_block 4096 deterministic true checkpoint_seconds 30)

; ---- Assemble instance -----------------------------------------------------
let instance
  (obj version 1
  meta (obj name "Items-to-Field (Jetson)" description "Generated by Chariot")
       globals (obj field_coordinates [globals.field_lat globals.field_lng]
                    fuel_rate_per_km globals.fuel_rate
                    group_count (len groups)
                    target_team_size globals.target_team
                    hard_max_trips 3)
       attributes [ (obj name "workers") (obj name "dist_garage")
                    (obj name "dist_pickup_field") (obj name "dist_field_garage")
                    (obj name "productivity") (obj name "pick_cost") ]
       items items
  knapsacks groups
       constraints constraints
       objectives objectives
       blocks blocks
       context context
       solver solver
       seed globals.seed)

; ---- Emit JSON file --------------------------------------------------------
json.write instance "instance.json"
```

**What you get:** `instance.json` with the same structure as the YAML. Point your Jetson solver at it and go. If you’d like, I can also generate a tiny `entities.csv` starter file to match this program.
