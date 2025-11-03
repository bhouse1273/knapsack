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
  // Scout mode: track active items for exact solver handoff
  bool scout_mode = false;
  double scout_threshold = 0.5; // item must appear in this fraction of top candidates to be "active"
  int scout_top_k = 8;           // how many top candidates to consider for active set
};

struct BeamResult {
  std::vector<uint8_t> best_select; // size N, 0/1
  double objective = 0.0;
  double penalty = 0.0;
  double total = 0.0;
};

struct ScoutResult {
  // Best solution from beam search
  std::vector<uint8_t> best_select; // size N, 0/1
  double objective = 0.0;
  double penalty = 0.0;
  double total = 0.0;
  
  // Active set for exact solver handoff
  std::vector<int> active_items;      // indices of frequently selected items
  std::vector<double> item_frequency; // how often each item appeared in top-K (0.0-1.0)
  int original_item_count = 0;
  int active_item_count = 0;
  
  // Statistics
  int dominated_items_removed = 0;   // from dominance filter
  double filter_time_ms = 0.0;
  double solve_time_ms = 0.0;
  
  // Optional: reduced config as JSON string for exact solver
  std::string reduced_config_json;
};

// Minimal beam search over select-mode with a single capacity attribute and one knapsack.
// Uses Metal evaluator on Apple; falls back to CPU if Metal init fails.
// Returns false on errors (e.g., missing required attributes).
bool SolveBeamSelect(const Config& cfg, const HostSoA& soa, const SolverOptions& opt,
                     BeamResult* out, std::string* err);

// Scout mode: run beam search with active set tracking for exact solver handoff.
// Applies dominance filters, tracks frequently selected items, and generates reduced config.
// Returns false on errors.
bool SolveBeamScout(const Config& cfg, const HostSoA& soa, const SolverOptions& opt,
                    ScoutResult* out, std::string* err);

} // namespace v2
