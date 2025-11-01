#pragma once
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
  const unsigned char* candidates; // bit-packed (2 bits per item)
  int num_items;
  int num_candidates;
  // Evaluator attributes
  const float* item_values;   // len = num_items
  const float* item_weights;  // len = num_items
  const float* van_capacities;// len = num_vans
  int num_vans;
  float penalty_coeff;
} MetalEvalIn;

typedef struct {
  float* obj;          // len = num_candidates
  float* soft_penalty; // len = num_candidates
} MetalEvalOut;

// Initialize the Metal pipeline from an in-memory metallib blob.
// Returns 0 on success, non-zero on failure.
int knapsack_metal_init_from_data(const void* data, size_t len, char* errbuf, int errlen);

// Initialize the Metal pipeline from in-memory Metal Shading Language (MSL) source.
// This compiles the shader at runtime using the system Metal framework, avoiding the need
// for the external 'metal' CLI. Returns 0 on success.
int knapsack_metal_init_from_source(const char* src, size_t len, char* errbuf, int errlen);

// Evaluate all candidates using the active Metal pipeline.
// Returns 0 on success, non-zero on failure.
int knapsack_metal_eval(const MetalEvalIn* in, MetalEvalOut* out, char* errbuf, int errlen);

#ifdef __cplusplus
}
#endif
