// Copyright (c) 2025
// SPDX-License-Identifier: MIT

#include "v2/Data.h"

#include <algorithm>

namespace v2 {

bool BuildHostSoA(const Config& cfg, HostSoA* out, std::string* err) {
  if (!out) { if (err) *err = "out is null"; return false; }
  HostSoA soa;
  soa.count = cfg.items.count;
  for (const auto& kv : cfg.items.attributes) {
    const auto& name = kv.first;
    const auto& vec = kv.second;
    if ((int)vec.size() != cfg.items.count) {
      if (err) *err = "attribute '" + name + "' size mismatch";
      return false;
    }
    soa.attr[name] = vec; // copy; could move if cfg is non-const
  }
  *out = std::move(soa);
  return true;
}

std::vector<BlockSlice> BuildBlockSlices(const Config& cfg) {
  std::vector<BlockSlice> out;
  out.reserve(cfg.blocks.size());
  for (const auto& b : cfg.blocks) {
    BlockSlice bs;
    bs.name = b.name;
    if (b.start >= 0 && b.count > 0) {
      bs.indices.resize(b.count);
      for (int i = 0; i < b.count; ++i) bs.indices[i] = b.start + i;
    } else if (!b.indices.empty()) {
      bs.indices = b.indices;
    } else {
      // Should not occur if ValidateConfig was called; leave empty
    }
    out.push_back(std::move(bs));
  }
  return out;
}

} // namespace v2
