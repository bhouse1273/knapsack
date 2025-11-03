#include "cuda_api.h"
#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>

struct Uniforms {
  unsigned int num_items;
  unsigned int num_candidates;
  unsigned int bytes_per_candidate;
  unsigned int num_groups;
  float penalty_coeff;
  float penalty_power;
  unsigned int num_obj_terms;
  unsigned int num_soft_constraints;
};

__global__ void eval_block_candidates_cuda(
  const unsigned char* __restrict__ candidates,
  float* __restrict__ obj_out,
  float* __restrict__ pen_out,
  const float* __restrict__ item_values,
  const float* __restrict__ item_weights,
  const float* __restrict__ group_caps,
  const float* __restrict__ obj_attrs,
  const float* __restrict__ obj_weights,
  const float* __restrict__ cons_attrs,
  const float* __restrict__ cons_limits,
  const float* __restrict__ cons_weights,
  const float* __restrict__ cons_powers,
  Uniforms U)
{
  unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= U.num_candidates) return;

  float obj = 0.0f;
  float pen = 0.0f;

  const unsigned int MaxGroups = 8u;
  float loads[MaxGroups];
  #pragma unroll
  for (unsigned int g = 0; g < MaxGroups; ++g) loads[g] = 0.0f;

  const unsigned int base = tid * U.bytes_per_candidate;
  for (unsigned int i = 0; i < U.num_items; ++i) {
    const unsigned int byteIdx = base + (i >> 2);
    const unsigned int shift = (i & 3u) * 2u;
    const unsigned int lane = (candidates[byteIdx] >> shift) & 0x3u;
    if (lane == 0u) continue;
    if (U.num_obj_terms > 0u && obj_attrs && obj_weights) {
      for (unsigned int t = 0; t < U.num_obj_terms; ++t) {
        obj += obj_weights[t] * obj_attrs[t * U.num_items + i];
      }
    } else {
      obj += (item_values ? item_values[i] : 0.0f);
    }
    const unsigned int group = lane - 1u;
    if (group < U.num_groups && group < MaxGroups) {
      loads[group] += (item_weights ? item_weights[i] : 0.0f);
    }
  }

  // Per-group soft penalties (legacy assign-mode capacity)
  if (U.penalty_coeff != 0.0f && group_caps && U.num_groups > 0u) {
    const unsigned int n = (U.num_groups > MaxGroups) ? MaxGroups : U.num_groups;
    for (unsigned int g = 0; g < n; ++g) {
      const float over = loads[g] - group_caps[g];
      if (over > 0.0f) pen += powf(over, U.penalty_power) * U.penalty_coeff;
    }
  }

  // Global soft constraints (select-mode friendly)
  if (U.num_soft_constraints > 0u && cons_attrs && cons_limits && cons_weights && cons_powers) {
    for (unsigned int c = 0; c < U.num_soft_constraints; ++c) {
      float sum = 0.0f;
      for (unsigned int i = 0; i < U.num_items; ++i) {
        const unsigned int byteIdx = base + (i >> 2);
        const unsigned int shift = (i & 3u) * 2u;
        const unsigned int lane = (candidates[byteIdx] >> shift) & 0x3u;
        if (lane == 0u) continue;
        sum += cons_attrs[c * U.num_items + i];
      }
      const float over = sum - cons_limits[c];
      if (over > 0.0f) pen += cons_weights[c] * powf(over, cons_powers[c]);
    }
  }

  obj_out[tid] = obj;
  pen_out[tid] = pen;
}

static void setErr(char* buf, int len, const char* msg) {
  if (buf && len > 0 && msg) snprintf(buf, (size_t)len, "%s", msg);
}

int knapsack_cuda_eval(const CudaEvalIn* in, CudaEvalOut* out, char* errbuf, int errlen) {
  if (!in || !out || !in->candidates || !out->obj || !out->soft_penalty) { setErr(errbuf, errlen, "invalid pointers"); return -1; }
  if (in->num_items < 0 || in->num_candidates < 0) { setErr(errbuf, errlen, "negative sizes"); return -2; }
  const unsigned int num_items = (unsigned int)in->num_items;
  const unsigned int num_cands = (unsigned int)in->num_candidates;
  const unsigned int bytes_per_cand = (num_items + 3u) / 4u;
  const size_t candBytes = (size_t)bytes_per_cand * (size_t)num_cands;

  unsigned char* d_cand = nullptr; float* d_obj = nullptr; float* d_pen = nullptr;
  float *d_item_values = nullptr, *d_item_weights = nullptr, *d_group_caps = nullptr;
  float *d_obj_attrs = nullptr, *d_obj_weights = nullptr;
  float *d_cons_attrs = nullptr, *d_cons_limits = nullptr, *d_cons_weights = nullptr, *d_cons_powers = nullptr;

  cudaError_t st;
  if ((st = cudaMalloc(&d_cand, candBytes)) != cudaSuccess) { setErr(errbuf, errlen, "cudaMalloc d_cand failed"); return -3; }
  if ((st = cudaMalloc(&d_obj, sizeof(float)*num_cands)) != cudaSuccess) { setErr(errbuf, errlen, "cudaMalloc d_obj failed"); return -3; }
  if ((st = cudaMalloc(&d_pen, sizeof(float)*num_cands)) != cudaSuccess) { setErr(errbuf, errlen, "cudaMalloc d_pen failed"); return -3; }
  if ((st = cudaMemcpy(d_cand, in->candidates, candBytes, cudaMemcpyHostToDevice)) != cudaSuccess) { setErr(errbuf, errlen, "cudaMemcpy candidates failed"); return -4; }

  if (in->item_values && num_items > 0) { cudaMalloc(&d_item_values, sizeof(float)*num_items); cudaMemcpy(d_item_values, in->item_values, sizeof(float)*num_items, cudaMemcpyHostToDevice); }
  if (in->item_weights && num_items > 0) { cudaMalloc(&d_item_weights, sizeof(float)*num_items); cudaMemcpy(d_item_weights, in->item_weights, sizeof(float)*num_items, cudaMemcpyHostToDevice); }
  if (in->group_capacities && in->num_groups > 0) { cudaMalloc(&d_group_caps, sizeof(float)*in->num_groups); cudaMemcpy(d_group_caps, in->group_capacities, sizeof(float)*in->num_groups, cudaMemcpyHostToDevice); }

  if (in->obj_attrs && in->obj_weights && in->num_obj_terms > 0) {
    size_t objAttrCount = (size_t)in->num_obj_terms * (size_t)num_items;
    cudaMalloc(&d_obj_attrs, sizeof(float)*objAttrCount);
    cudaMemcpy(d_obj_attrs, in->obj_attrs, sizeof(float)*objAttrCount, cudaMemcpyHostToDevice);
    cudaMalloc(&d_obj_weights, sizeof(float)*in->num_obj_terms);
    cudaMemcpy(d_obj_weights, in->obj_weights, sizeof(float)*in->num_obj_terms, cudaMemcpyHostToDevice);
  }

  if (in->cons_attrs && in->cons_limits && in->cons_weights && in->cons_powers && in->num_soft_constraints > 0) {
    size_t consAttrCount = (size_t)in->num_soft_constraints * (size_t)num_items;
    cudaMalloc(&d_cons_attrs, sizeof(float)*consAttrCount);
    cudaMemcpy(d_cons_attrs, in->cons_attrs, sizeof(float)*consAttrCount, cudaMemcpyHostToDevice);
    cudaMalloc(&d_cons_limits, sizeof(float)*in->num_soft_constraints);
    cudaMemcpy(d_cons_limits, in->cons_limits, sizeof(float)*in->num_soft_constraints, cudaMemcpyHostToDevice);
    cudaMalloc(&d_cons_weights, sizeof(float)*in->num_soft_constraints);
    cudaMemcpy(d_cons_weights, in->cons_weights, sizeof(float)*in->num_soft_constraints, cudaMemcpyHostToDevice);
    cudaMalloc(&d_cons_powers, sizeof(float)*in->num_soft_constraints);
    cudaMemcpy(d_cons_powers, in->cons_powers, sizeof(float)*in->num_soft_constraints, cudaMemcpyHostToDevice);
  }

  Uniforms U; U.num_items = num_items; U.num_candidates = num_cands; U.bytes_per_candidate = bytes_per_cand;
  U.num_groups = (unsigned int)in->num_groups; U.penalty_coeff = in->penalty_coeff; U.penalty_power = in->penalty_power;
  U.num_obj_terms = (unsigned int)((in->obj_attrs && in->num_obj_terms>0) ? in->num_obj_terms : 0);
  U.num_soft_constraints = (unsigned int)((in->cons_attrs && in->num_soft_constraints>0) ? in->num_soft_constraints : 0);

  dim3 block(128); dim3 grid((num_cands + block.x - 1) / block.x);
  eval_block_candidates_cuda<<<grid, block>>>(d_cand, d_obj, d_pen, d_item_values, d_item_weights, d_group_caps,
                                              d_obj_attrs, d_obj_weights,
                                              d_cons_attrs, d_cons_limits, d_cons_weights, d_cons_powers, U);
  if ((st = cudaDeviceSynchronize()) != cudaSuccess) { setErr(errbuf, errlen, "kernel failed"); return -5; }

  cudaMemcpy(out->obj, d_obj, sizeof(float)*num_cands, cudaMemcpyDeviceToHost);
  cudaMemcpy(out->soft_penalty, d_pen, sizeof(float)*num_cands, cudaMemcpyDeviceToHost);

  cudaFree(d_cand); cudaFree(d_obj); cudaFree(d_pen);
  if (d_item_values) cudaFree(d_item_values);
  if (d_item_weights) cudaFree(d_item_weights);
  if (d_group_caps) cudaFree(d_group_caps);
  if (d_obj_attrs) cudaFree(d_obj_attrs);
  if (d_obj_weights) cudaFree(d_obj_weights);
  if (d_cons_attrs) cudaFree(d_cons_attrs);
  if (d_cons_limits) cudaFree(d_cons_limits);
  if (d_cons_weights) cudaFree(d_cons_weights);
  if (d_cons_powers) cudaFree(d_cons_powers);

  return 0;
}
