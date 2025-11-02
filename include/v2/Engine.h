// Copyright (c) 2025
// SPDX-License-Identifier: MIT

#pragma once

#include <string>
#include <vector>

#include "v2/Config.h"
#include "v2/Data.h"

namespace v2 {

struct SolverOptions {
  int beam_width = 16;
  int iters = 3;
  unsigned int seed = 1234;
  bool debug = false; // enable lightweight debug logs
  // Dominance filters (preprocess)
  bool enable_dominance_filter = false;
  double dom_eps = 1e-9;
  bool dom_use_surrogate = true;
};

struct BeamResult {
  std::vector<uint8_t> best_select; // size N, 0/1
  double objective = 0.0;
  double penalty = 0.0;
  double total = 0.0;
};

// Minimal beam search over select-mode with a single capacity attribute and one knapsack.
// Uses Metal evaluator on Apple; falls back to CPU if Metal init fails.
// Returns false on errors (e.g., missing required attributes).
bool SolveBeamSelect(const Config& cfg, const HostSoA& soa, const SolverOptions& opt,
                     BeamResult* out, std::string* err);

} // namespace v2
