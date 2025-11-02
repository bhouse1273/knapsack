#pragma once
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
  const unsigned char* candidates; // bit-packed (2 bits per item)
  int num_items;
  int num_candidates;
  // Legacy evaluator attributes (single-term + per-van capacity)
  const float* item_values;   // len = num_items
  const float* item_weights;  // len = num_items
  const float* van_capacities;// len = num_vans
  int num_vans;
  float penalty_coeff;        // soft penalty weight
  float penalty_power;        // soft penalty exponent (e.g., 1=linear, 2=quadratic)
  // Multi-term objective
  const float* obj_attrs;     // len = num_obj_terms * num_items (term-major)
  const float* obj_weights;   // len = num_obj_terms
  int num_obj_terms;
  // Global soft constraints
  const float* cons_attrs;    // len = num_soft_constraints * num_items
  const float* cons_limits;   // len = num_soft_constraints
  const float* cons_weights;  // len = num_soft_constraints
  const float* cons_powers;   // len = num_soft_constraints
  int num_soft_constraints;
} CudaEvalIn;

typedef struct {
  float* obj;          // len = num_candidates
  float* soft_penalty; // len = num_candidates
} CudaEvalOut;

// Evaluate all candidates using CUDA. Returns 0 on success.
int knapsack_cuda_eval(const CudaEvalIn* in, CudaEvalOut* out, char* errbuf, int errlen);

#ifdef __cplusplus
}
#endif
