#include <metal_stdlib>
using namespace metal;

struct Uniforms {
  uint  num_items;
  uint  num_candidates;
  uint  bytes_per_candidate;
  uint  num_vans;
  float penalty_coeff;   // legacy per-van penalty coeff (assign mode)
  float penalty_power;   // legacy per-van penalty power (assign mode)
  uint  num_obj_terms;   // number of objective terms (0 => use legacy item_values)
  uint  num_soft_constraints; // number of global soft capacity constraints
};

kernel void eval_block_candidates(
  device const uchar* candidates   [[buffer(0)]],
  device float*       obj_out      [[buffer(1)]],
  device float*       pen_out      [[buffer(2)]],
  device const float* item_values  [[buffer(3)]],
  device const float* item_weights [[buffer(4)]],
  device const float* van_caps     [[buffer(5)]],
  // New multi-term objective and multi-constraint buffers
  device const float* obj_attrs    [[buffer(6)]], // len = num_obj_terms * num_items
  device const float* obj_weights  [[buffer(7)]], // len = num_obj_terms
  device const float* cons_attrs   [[buffer(8)]], // len = num_soft_constraints * num_items
  device const float* cons_limits  [[buffer(9)]], // len = num_soft_constraints
  device const float* cons_weights [[buffer(10)]],// len = num_soft_constraints
  device const float* cons_powers  [[buffer(11)]],// len = num_soft_constraints
  constant Uniforms&  U            [[buffer(15)]],
  uint tid [[thread_position_in_grid]])
{
  if (tid >= U.num_candidates) return;

  float obj = 0.0f;
  float pen = 0.0f;

  // Accumulate per-van loads (support up to 8 vans; only first U.num_vans used).
  constexpr uint MaxVans = 8;
  float loads[MaxVans];
  for (uint v = 0; v < MaxVans; ++v) loads[v] = 0.0f;

  // Unpack 2-bit lanes and evaluate objective/constraints:
  // lane=0 => unassigned; lane in [1..num_vans] => assigned to (lane-1) van index.
  const uint base = tid * U.bytes_per_candidate;
  // track selection mask for global objective/constraints
  // Note: we stream over items; we don't store a take[] array to keep registers low.
  for (uint i = 0; i < U.num_items; ++i) {
    const uint byteIdx = base + (i >> 2);
    const uint shift = (i & 3u) * 2u;
    const uint lane = (candidates[byteIdx] >> shift) & 0x3u;
    if (lane == 0u) continue;
    // Objective accumulation
    if (U.num_obj_terms > 0u && obj_attrs && obj_weights) {
      // sum over objective terms: w[t] * attr[t * num_items + i]
      for (uint t = 0; t < U.num_obj_terms; ++t) {
        obj += obj_weights[t] * obj_attrs[t * U.num_items + i];
      }
    } else {
      // legacy single-term path
      obj += (item_values ? item_values[i] : 0.0f);
    }
    // Per-van loads for assign mode legacy capacity penalty
    const uint van = lane - 1u;
    if (van < U.num_vans && van < MaxVans) {
      loads[van] += (item_weights ? item_weights[i] : 0.0f);
    }
    // Global soft constraints: accumulate attribute sums
    if (U.num_soft_constraints > 0u && cons_attrs) {
      for (uint c = 0; c < U.num_soft_constraints; ++c) {
        // We can't store per-constraint sums without registers; compute later in a second pass.
        // To keep this single-pass, we accumulate into temporary threadgroup memory would be overkill here.
        // We'll defer penalty computation to a separate loop below that scans items again if needed.
      }
    }
  }

  // Soft capacity penalty per van (legacy path).
  const float alpha = U.penalty_coeff;
  if (alpha != 0.0f && van_caps && U.num_vans > 0u) {
    const uint n = (U.num_vans > MaxVans) ? MaxVans : U.num_vans;
    for (uint v = 0; v < n; ++v) {
      const float over = loads[v] - van_caps[v];
      if (over > 0.0f) {
        // pow(over, power) * coeff
        pen += pow(over, U.penalty_power) * alpha;
      }
    }
  }

  // Global soft capacity constraints (select-mode friendly):
  if (U.num_soft_constraints > 0u && cons_attrs && cons_limits && cons_weights && cons_powers) {
    // Recompute per-constraint sums by scanning items once per constraint.
    for (uint c = 0; c < U.num_soft_constraints; ++c) {
      float sum = 0.0f;
      for (uint i = 0; i < U.num_items; ++i) {
        const uint byteIdx = base + (i >> 2);
        const uint shift = (i & 3u) * 2u;
        const uint lane = (candidates[byteIdx] >> shift) & 0x3u;
        if (lane == 0u) continue;
        sum += cons_attrs[c * U.num_items + i];
      }
      const float over = sum - cons_limits[c];
      if (over > 0.0f) {
        pen += cons_weights[c] * pow(over, cons_powers[c]);
      }
    }
  }

  obj_out[tid] = obj;
  pen_out[tid] = pen;
}
