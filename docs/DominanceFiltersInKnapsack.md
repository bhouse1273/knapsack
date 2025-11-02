## what “dominance filters” are

- Goal: Remove items that can never be part of any optimal solution because some other item is strictly better “in every way that matters.”
- Idea: For two items i and j with:
  - objective contribution vectors v(i), v(j) (or a scalar value if weighted-sum)
  - constraint consumption vectors a(i), a(j) (e.g., weights, volumes, time)
- i dominates j if:
  - a(i) ≤ a(j) componentwise, and
  - v(i) ≥ v(j) componentwise (or ≥ for the chosen scalarized objective),
  - with at least one strict inequality.
- If i dominates j, you can drop j and never worsen the optimal solution.

This is the Pareto skylining of items w.r.t. constraints (costs) and objective value(s).

## when it’s safe and useful

- Classic 0/1 knapsack (single capacity):
  - Dominance reduces to: weight(i) ≤ weight(j) and value(i) ≥ value(j), with at least one strict → drop j.
  - Fast O(n log n) via sort-by-weight and tracking running max value.
- Multi-constraint knapsack (select mode):
  - Use vector dominance on all capacity-like attributes.
  - Still safe: removing dominated items cannot remove any optimal solution.
  - Computation is more expensive; see algorithms below.
- Multiple objective terms:
  - If the solver uses a weighted sum with non-negative weights, compute per-item scalar value = Σ w_t * attr_t[i] and use that in dominance.
  - If you truly need multi-objective Pareto optimal sets (lexicographic or non-sum), you can keep a small set per block (phase 2).
- Assign mode (K knapsacks):
  - Item constraint usage is the same regardless of which knapsack it goes to, so item-level dominance is still valid.
- Penalties and soft constraints:
  - Since penalties apply to total violation, dominance defined on raw consumptions still holds for pruning items.

Edge cases to watch:
- Equal vectors: if a(i) == a(j) and v(i) == v(j), keep one (dedupe).
- Negative weights or attributes: dominance assumptions require non-negative consumptions and non-decreasing objective with respect to v; if not true, skip filtering for safety.
- Floating point noise: use epsilon-dominance (<= (1+eps), >= (1−eps)).

## algorithms

Single constraint (fast skyline)
- Sort by weight asc; sweep keeping a running maximum of value.
- If value(current) ≤ max_value_so_far, current is dominated → drop.
- Complexity: O(n log n). Great speedup in practice.

Two constraints (exact skyline)
- Sort by first constraint asc, then by second asc.
- Maintain a data structure over the second constraint tracking the best value so far with non-increasing second consumption; an efficient approach uses a monotone queue or a balanced tree keyed by second consumption with running max value.
- Drop items whose value ≤ best value at a point with second consumption ≤ theirs.
- Complexity: O(n log n). Generalizes partially.

K constraints (general)
- Exact: naive O(n^2) pairwise dominance checks with early breaks; acceptable for moderate n.
- Better: divide-and-conquer skylining (Bentley’s algorithm) or R-trees; complexity around O(n log^{k-1} n) in the average case.

Approximations when K is large
- Surrogate capacity: α = Σ_c (w_c / limit_c) · a_c(i) → single scalar consumption; sort by α and do single-constraint skyline with value.
- Multiple surrogates: do a few different α weightings (e.g., using cons penalty weights) and intersect surviving sets.
- Epsilon dominance to avoid removing near-equals.

## contract for our implementation

- Inputs:
  - Config with objective terms (weights), constraints (capacity-like) and HostSoA attributes.
- Outputs:
  - Filtered HostSoA (same attribute layout but fewer items)
  - A mapping orig_index_of[filtered_idx] to reconstruct or report on originals
  - Stats: dropped count, epsilon used, time
- Success criteria:
  - No loss in optimality for selection/assignment modes under non-negative constraints and weighted-sum objective
  - Reduced N, ideally faster beam seeding/repair and equal-or-better initial objective

## where to wire it in this repo

- New files:
  - include/v2/Preprocess.h: DominanceFilterOptions { epsilon, mode: exact|surrogate, maxKForExact }, API:
    - bool ApplyDominanceFilters(const Config&, const HostSoA&, HostSoA* filtered, std::vector<int>* map, DominanceFilterStats*, std::string* err);
  - src/v2/Preprocess.cpp: implement single-constraint O(n log n), 2-constraint O(n log n), and general fallback O(n^2) + surrogate option.
- Integration points:
  - BeamSearch seeding path:
    - If enabled in options (e.g., SolverOptions.enable_dominance_filter), run filtering after BuildHostSoA and before seeding.
  - CLI and C API:
    - Accept an optional options JSON flag enable_dominance_filter with epsilon and mode to toggle this.
  - Debug logs:
    - Print “dominance: dropped X/Y items, epsilon=.., method=..”

## example filter logic (single-constraint)

- value[i] = Σ_t w_t * attr_t[i]  (only if all w_t ≥ 0)
- weight[i] = attribute referenced by the capacity constraint
- Sort items by weight asc, value desc to keep better ties.
- Sweep:
  - Keep best_value = −inf
  - For item in order:
    - If value ≤ best_value + eps: drop it (dominated)
    - Else keep and best_value = value

## example filter logic (multi-constraint exact, small K)

- For each i:
  - For each j != i, check:
    - all a_c(j) ≤ a_c(i) + eps
    - and value(j) ≥ value(i) − eps
    - with one strict → i dominated → drop.
- Early exit by sorting on one constraint and pruning candidate j indices.
- For speed, first run single-constraint skyline by surrogate α to prune, then exact check on survivors.

## complexity and gains

- Single constraint: near linear after sort; drops a lot when there are many near-duplicate items.
- Multiple constraints: O(n^2) fallback is fine for hundreds to a few thousands; use surrogates beyond that.
- Gains: Fewer items → better greedy seed and fewer repairs; smaller candidate encodings for GPU → less memory traffic and faster evaluation.

## how it helps our beam

- Seed quality: density uses surrogate weights already; removing dominated items reduces noise so the top density items are more meaningful.
- Repair: fewer low-quality items reduces flips that immediately violate constraints.
- GPU eval: fewer columns in attribute arrays (still N), so packing candidates and evaluation improves.

If you want, I can implement ApplyDominanceFilters now (single-constraint O(n log n) + 2-constraint O(n log n) + general fallback + surrogate mode), wire it behind a SolverOptions flag, and add a small sanity tool that prints before/after N and verifies objective monotonicity with the current example.