// Copyright (c) 2025
// SPDX-License-Identifier: MIT

#pragma once

#include <map>
#include <string>
#include <vector>

#include "v2/Config.h"

namespace v2 {

// Host-side Structure-of-Arrays for attributes.
struct HostSoA {
  int count = 0; // number of items
  std::map<std::string, std::vector<double>> attr; // name -> values[count]
};

// A block slice as concrete indices into items [0..count-1].
struct BlockSlice {
  std::string name;
  std::vector<int> indices; // size == number of items in this block
};

// Build HostSoA from Config::items. Validates sizes; returns false on mismatch.
bool BuildHostSoA(const Config& cfg, HostSoA* out, std::string* err);

// Convert block specs to explicit index lists.
std::vector<BlockSlice> BuildBlockSlices(const Config& cfg);

} // namespace v2
