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

enum class AttributeFormatKind {
  kBinary64LE,
  kCSV,
  kArrow,
  kParquet,
  kUnknown,
};

struct AttributeSourceSpec {
  AttributeSourceKind kind = AttributeSourceKind::kInline;
  std::string format = "binary64_le";   // encoding of payload
  AttributeFormatKind format_kind = AttributeFormatKind::kBinary64LE;
  std::string path;                      // primary path for file/chunk inputs
  std::vector<std::string> chunks;       // optional chunk files processed sequentially
  std::string channel;                   // named pipe/descriptor for streaming (e.g., "stdin")
  std::size_t offset_bytes = 0;          // byte offset for binary files (applied to first chunk)
  char csv_delimiter = ',';              // CSV-specific delimiter
  bool csv_has_header = false;           // CSV-specific header toggle
  std::string column_name;               // Column name (CSV/Arrow/Parquet)
  int column_index = -1;                 // Column index fallback
  bool optional = false;                 // Allow missing external data

  bool is_inline() const { return kind == AttributeSourceKind::kInline; }
  bool is_file() const { return kind == AttributeSourceKind::kFile; }
  bool is_stream() const { return kind == AttributeSourceKind::kStream; }
  AttributeFormatKind format_kind_enum() const { return format_kind; }
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
  std::string strategy = "weighted_sum"; // weighted_sum | epsilon | pareto_component
  double epsilon = 0.0;   // epsilon constraint threshold (for strategy == epsilon)
  double target = 0.0;    // optional target value for hybrid strategies
};

struct ParetoArchiveSpec {
  int max_size = 32;                 // maximum number of solutions kept in archive
  double dominance_epsilon = 1e-9;   // tolerance when comparing objectives
  std::string diversity_metric = "crowding"; // "crowding" | "hypervolume"
  bool keep_feasible_only = true;    // drop infeasible members from archive
};

struct AOASolverSpec {
  int population = 128;        // candidate count
  int max_iterations = 500;    // iteration budget
  double exploration_rate = 0.5; // relative time spent exploring
  double exploitation_rate = 0.5; // relative time spent exploiting
  double anneal_start = 1.0;   // initial temperature for annealing schedule
  double anneal_end = 0.01;    // terminal temperature
  double repair_penalty = 1.0; // penalty multiplier applied during constraint repair
};

struct MOAOASolverSpec {
  AOASolverSpec base;          // shared arithmetic parameters
  ParetoArchiveSpec archive;   // pareto archive configuration
  int weight_vectors = 32;     // number of reference weight vectors
  int archive_refresh = 5;     // iterations between archive resort / trimming
};

struct SolverSpec {
  std::string kind = "beam";  // "beam" | "aoa" | "moaoa"
  AOASolverSpec aoa;           // used when kind == "aoa"
  MOAOASolverSpec moaoa;       // used when kind == "moaoa"
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
  SolverSpec solver;                  // solver selection + options

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
