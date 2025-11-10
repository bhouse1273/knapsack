// Copyright (c) 2025
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "v2/Config.h"
#include "v2/Data.h"

namespace v2 {

struct EvalResult {
  double objective = 0.0;
  double penalty = 0.0;
  double total = 0.0; // objective - penalty
  std::vector<double> constraint_violations; // same order as cfg.constraints
  std::vector<double> knapsack_violations;   // size K (assign mode)
};

// Candidate encodings
struct CandidateSelect {
  // size == N items; 0 or 1
  std::vector<uint8_t> select;
};

struct CandidateAssign {
  // size == N items; -1 means unassigned, otherwise [0..K-1]
  std::vector<int> assign;
};

// Evaluate a candidate in select mode.
bool EvaluateCPU_Select(const Config& cfg, const HostSoA& soa,
                        const CandidateSelect& cand, EvalResult* out, std::string* err);

// Evaluate a candidate in assign mode. Adds per-knapsack capacity penalties using
// cfg.knapsack.capacities and cfg.knapsack.capacity_attr. If a matching soft
// capacity constraint is present in cfg.constraints (kind=="capacity" and attr==capacity_attr),
// its penalty {weight,power} is used; otherwise defaults to {1.0,1.0}.
bool EvaluateCPU_Assign(const Config& cfg, const HostSoA& soa,
                        const CandidateAssign& cand, EvalResult* out, std::string* err);

#ifdef __APPLE__
// Evaluate a candidate in select mode using Metal GPU.
// Requires Metal framework initialization via knapsack_metal_init_from_source().
bool EvaluateMetal_Select(const Config& cfg, const HostSoA& soa,
                          const CandidateSelect& cand, EvalResult* out, std::string* err);

// Evaluate multiple candidates in parallel using Metal GPU.
// This is where GPU acceleration shines - evaluating many candidates simultaneously.
// Expected speedup over CPU: 2-50x for batches of 100-10,000 candidates.
bool EvaluateMetal_Batch(const Config& cfg, const HostSoA& soa,
                         const std::vector<CandidateSelect>& candidates,
                         std::vector<EvalResult>* results, std::string* err);
#endif

} // namespace v2
