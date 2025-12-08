// Copyright (c) 2025
// SPDX-License-Identifier: MIT

#pragma once

#include <cstddef>
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

// Incremental builder for HostSoA data, supports columnar and row ingestion.
class HostSoABuilder {
 public:
  HostSoABuilder() = default;

  bool Begin(int expected_count, std::string* err = nullptr);
  bool DeclareAttribute(const std::string& name, std::string* err = nullptr);
  bool DeclareAttributes(const std::vector<std::string>& names, std::string* err = nullptr);
  bool AppendAttributeValues(const std::string& name,
                            const std::vector<double>& values,
                            std::string* err = nullptr);
  bool AppendAttributeValues(const std::string& name,
                            const double* values,
                            std::size_t count,
                            std::string* err = nullptr);
  bool AppendRow(const std::vector<double>& values, std::string* err = nullptr);
  bool Finish(HostSoA* out, std::string* err = nullptr);

 private:
  bool EnsureStarted(std::string* err) const;
  bool EnsureAttributeBuffer(const std::string& name, std::string* err);
  bool CheckRemaining(std::size_t current, std::size_t incoming, std::string* err) const;

  int expected_count_ = 0;
  bool started_ = false;
  bool finished_ = false;
  bool row_mode_ = false;
  std::size_t rows_appended_ = 0;
  std::vector<std::string> row_order_;
  std::map<std::string, std::vector<double>> buffers_;
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
