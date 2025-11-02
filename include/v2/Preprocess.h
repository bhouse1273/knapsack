// Copyright (c) 2025
// SPDX-License-Identifier: MIT

#pragma once

#include <string>
#include <vector>

#include "v2/Config.h"
#include "v2/Data.h"

namespace v2 {

struct DominanceFilterOptions {
  bool enabled = true;
  double epsilon = 1e-9;         // tolerance for comparisons
  bool use_surrogate = true;     // when multiple constraints, use scalar surrogate alpha
};

struct DominanceFilterStats {
  int dropped = 0;
  int kept = 0;
  double epsilon_used = 0.0;
  std::string method; // "single", "surrogate", or "disabled"
};

// Apply item-level dominance filtering.
// - Produces a filtered HostSoA and a mapping filtered_to_orig indices.
// - Returns true on success; on failure, returns false and leaves outputs untouched.
bool ApplyDominanceFilters(const Config& cfg, const HostSoA& in,
                           const DominanceFilterOptions& opt,
                           HostSoA* out, std::vector<int>* filtered_to_orig,
                           DominanceFilterStats* stats, std::string* err);

} // namespace v2
