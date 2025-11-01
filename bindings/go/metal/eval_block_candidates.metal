#include <metal_stdlib>
using namespace metal;

struct Uniforms {
  uint  num_items;
  uint  num_candidates;
  uint  bytes_per_candidate;
  uint  num_vans;
  float penalty_coeff;
};

kernel void eval_block_candidates(
  device const uchar* candidates   [[buffer(0)]],
  device float*       obj_out      [[buffer(1)]],
  device float*       pen_out      [[buffer(2)]],
  device const float* item_values  [[buffer(3)]],
  device const float* item_weights [[buffer(4)]],
  device const float* van_caps     [[buffer(5)]],
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

  // Unpack 2-bit lanes and evaluate simple objective/constraint:
  // lane=0 => unassigned; lane in [1..num_vans] => assigned to (lane-1) van index.
  const uint base = tid * U.bytes_per_candidate;
  for (uint i = 0; i < U.num_items; ++i) {
    const uint byteIdx = base + (i >> 2);
    const uint shift = (i & 3u) * 2u;
    const uint lane = (candidates[byteIdx] >> shift) & 0x3u;
    if (lane == 0u) continue;
    obj += (item_values ? item_values[i] : 0.0f);
    const uint van = lane - 1u;
    if (van < U.num_vans && van < MaxVans) {
      loads[van] += (item_weights ? item_weights[i] : 0.0f);
    }
  }

  // Soft capacity penalty per van.
  const float alpha = U.penalty_coeff;
  if (alpha != 0.0f && van_caps && U.num_vans > 0u) {
    const uint n = (U.num_vans > MaxVans) ? MaxVans : U.num_vans;
    for (uint v = 0; v < n; ++v) {
      const float over = loads[v] - van_caps[v];
      if (over > 0.0f) pen += over * alpha;
    }
  }

  obj_out[tid] = obj;
  pen_out[tid] = pen;
}
