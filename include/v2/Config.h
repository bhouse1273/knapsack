// Copyright (c) 2025
// SPDX-License-Identifier: MIT

#pragma once

#include <cstddef>
#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace v2 {

enum class AttributeSourceKind {
  kInline,
  kFile,
  kStream,
};

struct AttributeSourceSpec {
  AttributeSourceKind kind = AttributeSourceKind::kInline;
  std::string format = "binary64_le";   // encoding of payload
  std::string path;                      // primary path for file/chunk inputs
  std::vector<std::string> chunks;       // optional chunk files processed sequentially
  std::string channel;                   // named pipe/descriptor for streaming (e.g., "stdin")
  std::size_t offset_bytes = 0;          // byte offset for binary files (applied to first chunk)

  bool is_inline() const { return kind == AttributeSourceKind::kInline; }
  bool is_file() const { return kind == AttributeSourceKind::kFile; }
  bool is_stream() const { return kind == AttributeSourceKind::kStream; }
};

struct PenaltySpec {
  double weight = 0.0;   // penalty coefficient
  double power = 1.0;    // penalty exponent (e.g., 1 = linear, 2 = quadratic)
};

struct ConstraintSpec {
  std::string kind;      // "capacity", "cardinality", ...
  std::string attr;      // attribute name to constrain (e.g., "weight")
  double limit = 0.0;    // capacity/limit
  bool soft = false;     // if true, violation allowed with penalty
  PenaltySpec penalty;   // used when soft == true
};

struct CostTermSpec {
  std::string attr;      // attribute name to score (e.g., "value")
  double weight = 1.0;   // contribution to objective
};

struct BlockSpec {
  std::string name;      // block label
  // Choose one encoding:
  int start = -1;        // contiguous range start (inclusive)
  int count = 0;         // number of items in the range
  std::vector<int> indices; // or explicit indices
};

struct ItemsSpec {
  int count = 0; // total items
  // Inline attribute payloads in Structure-of-Arrays form, all sized [count]
  std::map<std::string, std::vector<double>> attributes;
  // Optional external attribute descriptors (file/stream-backed)
  std::map<std::string, AttributeSourceSpec> sources;

  bool HasAttribute(const std::string& name) const {
    return attributes.find(name) != attributes.end() || sources.find(name) != sources.end();
  }

  const AttributeSourceSpec* FindSource(const std::string& name) const {
    auto it = sources.find(name);
    return it == sources.end() ? nullptr : &it->second;
  }
};

struct KnapsackSpec {
  int K = 1;                          // number of knapsacks (for assignment mode)
  std::vector<double> capacities;     // size K
  std::string capacity_attr;          // which attribute represents capacity consumption (e.g., "weight")
};

struct Config {
  int version = 2;                    // schema version
  std::string mode = "assign";       // "select" | "assign"
  std::uint64_t random_seed = 0;      // deterministic runs

  ItemsSpec items;
  std::vector<BlockSpec> blocks;
  KnapsackSpec knapsack;              // required when mode == "assign"
  std::vector<CostTermSpec> objective;
  std::vector<ConstraintSpec> constraints;
};

// Loaders. On Apple, JSON parsing uses Foundation (NSJSONSerialization).
// On other platforms, this currently returns false with an error (to be implemented).
bool LoadConfigFromJsonString(const std::string& json, Config* out, std::string* err);
bool LoadConfigFromFile(const std::string& path, Config* out, std::string* err);

// Structural validations independent of parsing backend.
bool ValidateConfig(const Config& cfg, std::string* err);

} // namespace v2
