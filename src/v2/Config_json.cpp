// Copyright (c) 2025
// SPDX-License-Identifier: MIT

#include "v2/Config.h"

#include <algorithm>
#include <cctype>
#include <cstddef>
#include <fstream>
#include <sstream>
#include <string>

#include "third_party/picojson/picojson.h"

namespace v2 {

static bool get_string(const picojson::object& o, const char* key, std::string* out) {
  auto it = o.find(key); if (it == o.end()) return false; if (!it->second.is<std::string>()) return false; *out = it->second.get<std::string>(); return true;
}
static bool get_int(const picojson::object& o, const char* key, int* out) {
  auto it = o.find(key); if (it == o.end()) return false; if (!it->second.is<double>()) return false; *out = (int)it->second.get<double>(); return true;
}
static bool get_uint64(const picojson::object& o, const char* key, std::uint64_t* out) {
  auto it = o.find(key); if (it == o.end()) return false; if (!it->second.is<double>()) return false; double v = it->second.get<double>(); if (v < 0) return false; *out = (std::uint64_t)v; return true;
}
static bool get_array(const picojson::object& o, const char* key, picojson::array* out) {
  auto it = o.find(key); if (it == o.end()) return false; if (!it->second.is<picojson::array>()) return false; *out = it->second.get<picojson::array>(); return true;
}
static bool get_object(const picojson::object& o, const char* key, picojson::object* out) {
  auto it = o.find(key); if (it == o.end()) return false; if (!it->second.is<picojson::object>()) return false; *out = it->second.get<picojson::object>(); return true;
}

static AttributeFormatKind parse_format_kind(const std::string& format) {
  std::string lower = format;
  std::transform(lower.begin(), lower.end(), lower.begin(), [](unsigned char c) { return std::tolower(c); });
  if (lower == "binary64_le" || lower == "binary64" || lower == "binary" || lower == "float64") return AttributeFormatKind::kBinary64LE;
  if (lower == "csv") return AttributeFormatKind::kCSV;
  if (lower == "arrow") return AttributeFormatKind::kArrow;
  if (lower == "parquet") return AttributeFormatKind::kParquet;
  return AttributeFormatKind::kUnknown;
}

static bool parse_external_attr(const std::string& name, const picojson::object& obj, ItemsSpec* items, std::string* err) {
  AttributeSourceSpec spec;
  std::string source;
  if (!get_string(obj, "source", &source)) {
    if (err) *err = "attribute '" + name + "' missing source";
    return false;
  }
  if (source == "file") {
    spec.kind = AttributeSourceKind::kFile;
  } else if (source == "stream") {
    spec.kind = AttributeSourceKind::kStream;
  } else {
    if (err) *err = "attribute '" + name + "' has unsupported source '" + source + "'";
    return false;
  }
  std::string format;
  if (get_string(obj, "format", &format)) spec.format = format;
  spec.format_kind = parse_format_kind(spec.format);
  std::string path;
  if (get_string(obj, "path", &path)) spec.path = path;
  std::string channel;
  if (get_string(obj, "channel", &channel)) spec.channel = channel;
  auto it = obj.find("offset_bytes");
  if (it != obj.end() && it->second.is<double>()) {
    double raw = it->second.get<double>();
    if (raw < 0) {
      if (err) *err = "attribute '" + name + "' offset_bytes must be >= 0";
      return false;
    }
    spec.offset_bytes = static_cast<std::size_t>(raw);
  }
  auto chunks_it = obj.find("chunks");
  if (chunks_it != obj.end()) {
    if (!chunks_it->second.is<picojson::array>()) {
      if (err) *err = "attribute '" + name + "' chunks must be an array";
      return false;
    }
    for (const auto& entry : chunks_it->second.get<picojson::array>()) {
      if (!entry.is<std::string>()) {
        if (err) *err = "attribute '" + name + "' chunks must be strings";
        return false;
      }
      spec.chunks.push_back(entry.get<std::string>());
    }
  }
  auto delim_it = obj.find("delimiter");
  if (delim_it != obj.end()) {
    if (delim_it->second.is<std::string>()) {
      const auto& s = delim_it->second.get<std::string>();
      spec.csv_delimiter = s.empty() ? '\0' : s[0];
    } else if (delim_it->second.is<double>()) {
      int v = static_cast<int>(delim_it->second.get<double>());
      spec.csv_delimiter = static_cast<char>(v);
    }
  }
  auto header_it = obj.find("has_header");
  if (header_it != obj.end()) {
    if (header_it->second.is<bool>()) spec.csv_has_header = header_it->second.get<bool>();
    else if (header_it->second.is<double>()) spec.csv_has_header = header_it->second.get<double>() != 0.0;
  }
  auto column_it = obj.find("column");
  if (column_it != obj.end() && column_it->second.is<std::string>()) spec.column_name = column_it->second.get<std::string>();
  auto column_index_it = obj.find("column_index");
  if (column_index_it != obj.end() && column_index_it->second.is<double>()) spec.column_index = static_cast<int>(column_index_it->second.get<double>());
  auto optional_it = obj.find("optional");
  if (optional_it != obj.end()) {
    if (optional_it->second.is<bool>()) spec.optional = optional_it->second.get<bool>();
    else if (optional_it->second.is<double>()) spec.optional = optional_it->second.get<double>() != 0.0;
  }
  if (items->sources.count(name) || items->attributes.count(name)) {
    if (err) *err = "duplicate attribute '" + name + "'";
    return false;
  }
  items->sources[name] = std::move(spec);
  return true;
}

static bool parse_items(const picojson::object& root, ItemsSpec* items, std::string* err) {
  picojson::object itemsObj; if (!get_object(root, "items", &itemsObj)) { if (err) *err = "missing 'items'"; return false; }
  if (!get_int(itemsObj, "count", &items->count)) { if (err) *err = "items.count missing or invalid"; return false; }
  picojson::object attrs; if (!get_object(itemsObj, "attributes", &attrs)) { if (err) *err = "items.attributes missing"; return false; }
  for (const auto& kv : attrs) {
    if (kv.second.is<picojson::array>()) {
      const auto& arr = kv.second.get<picojson::array>();
      std::vector<double> vec; vec.reserve(arr.size());
      for (const auto& e : arr) {
        if (!e.is<double>()) { if (err) *err = "attribute '" + kv.first + "' contains non-numeric"; return false; }
        vec.push_back(e.get<double>());
      }
      if (items->sources.count(kv.first)) { if (err) *err = "duplicate attribute '" + kv.first + "'"; return false; }
      items->attributes[kv.first] = std::move(vec);
    } else if (kv.second.is<picojson::object>()) {
      if (!parse_external_attr(kv.first, kv.second.get<picojson::object>(), items, err)) return false;
    } else {
      if (err) *err = "attribute '" + kv.first + "' must be array or object";
      return false;
    }
  }
  return true;
}

static bool parse_blocks(const picojson::object& root, std::vector<BlockSpec>* blocks, std::string* err) {
  picojson::array arr; if (!get_array(root, "blocks", &arr)) { if (err) *err = "missing 'blocks'"; return false; }
  blocks->clear(); blocks->reserve(arr.size());
  for (const auto& e : arr) {
    if (!e.is<picojson::object>()) { if (err) *err = "block entry not an object"; return false; }
    const auto& bo = e.get<picojson::object>();
    BlockSpec bs; bs.name = "";
    (void)get_string(bo, "name", &bs.name);
    int start = -1, count = 0; (void)get_int(bo, "start", &start); (void)get_int(bo, "count", &count);
    bs.start = start; bs.count = count; bs.indices.clear();
    auto it = bo.find("indices");
    if (it != bo.end() && it->second.is<picojson::array>()) {
      const auto& idx = it->second.get<picojson::array>();
      bs.indices.reserve(idx.size());
      for (const auto& v : idx) { if (!v.is<double>()) { if (err) *err = "block indices contain non-numeric"; return false; } bs.indices.push_back((int)v.get<double>()); }
    }
    blocks->push_back(std::move(bs));
  }
  return true;
}

static bool parse_objective(const picojson::object& root, std::vector<CostTermSpec>* objective, std::string* err) {
  picojson::array arr; if (!get_array(root, "objective", &arr)) { if (err) *err = "missing 'objective'"; return false; }
  objective->clear(); objective->reserve(arr.size());
  for (const auto& e : arr) {
    if (!e.is<picojson::object>()) { if (err) *err = "objective entry not an object"; return false; }
    const auto& oo = e.get<picojson::object>();
    CostTermSpec ct; if (!get_string(oo, "attr", &ct.attr)) { if (err) *err = "objective.attr missing"; return false; }
    auto it = oo.find("weight"); ct.weight = (it != oo.end() && it->second.is<double>()) ? it->second.get<double>() : 1.0;
    auto strat = oo.find("strategy");
    if (strat != oo.end() && strat->second.is<std::string>()) ct.strategy = strat->second.get<std::string>();
    auto eps = oo.find("epsilon");
    if (eps != oo.end() && eps->second.is<double>()) ct.epsilon = eps->second.get<double>();
    auto tgt = oo.find("target");
    if (tgt != oo.end() && tgt->second.is<double>()) ct.target = tgt->second.get<double>();
    objective->push_back(std::move(ct));
  }
  return true;
}

static bool parse_constraints(const picojson::object& root, std::vector<ConstraintSpec>* constraints, std::string* err) {
  constraints->clear();
  auto it = root.find("constraints"); if (it == root.end()) return true; if (!it->second.is<picojson::array>()) return true;
  const auto& arr = it->second.get<picojson::array>();
  constraints->reserve(arr.size());
  for (const auto& e : arr) {
    if (!e.is<picojson::object>()) { if (err) *err = "constraint entry not an object"; return false; }
    const auto& co = e.get<picojson::object>();
    ConstraintSpec cs;
    (void)get_string(co, "kind", &cs.kind);
    (void)get_string(co, "attr", &cs.attr);
    auto il = co.find("limit"); if (il != co.end() && il->second.is<double>()) cs.limit = il->second.get<double>();
    auto is = co.find("soft"); if (is != co.end()) cs.soft = is->second.is<bool>() ? is->second.get<bool>() : (is->second.is<double>() ? (is->second.get<double>() != 0.0) : false);
    auto ip = co.find("penalty");
    if (ip != co.end() && ip->second.is<picojson::object>()) {
      const auto& po = ip->second.get<picojson::object>();
      auto iw = po.find("weight"); if (iw != po.end() && iw->second.is<double>()) cs.penalty.weight = iw->second.get<double>();
      auto ipow = po.find("power"); if (ipow != po.end() && ipow->second.is<double>()) cs.penalty.power = ipow->second.get<double>();
    }
    constraints->push_back(std::move(cs));
  }
  return true;
}

static bool parse_knapsack(const picojson::object& root, KnapsackSpec* ks, std::string* err) {
  picojson::object k; if (!get_object(root, "knapsack", &k)) { if (err) *err = "missing 'knapsack'"; return false; }
  (void)get_int(k, "K", &ks->K);
  auto icaps = k.find("capacities"); if (icaps != k.end() && icaps->second.is<picojson::array>()) {
    const auto& arr = icaps->second.get<picojson::array>();
    ks->capacities.resize(arr.size()); for (size_t i = 0; i < arr.size(); ++i) ks->capacities[i] = arr[i].is<double>() ? arr[i].get<double>() : 0.0;
  }
  (void)get_string(k, "capacity_attr", &ks->capacity_attr);
  return true;
}

static void parse_archive_spec(const picojson::object& obj, ParetoArchiveSpec* archive) {
  auto size_it = obj.find("max_size");
  if (size_it != obj.end() && size_it->second.is<double>()) archive->max_size = static_cast<int>(size_it->second.get<double>());
  auto eps_it = obj.find("dominance_epsilon");
  if (eps_it != obj.end() && eps_it->second.is<double>()) archive->dominance_epsilon = eps_it->second.get<double>();
  auto div_it = obj.find("diversity_metric");
  if (div_it != obj.end() && div_it->second.is<std::string>()) archive->diversity_metric = div_it->second.get<std::string>();
  auto feas_it = obj.find("keep_feasible_only");
  if (feas_it != obj.end()) {
    if (feas_it->second.is<bool>()) archive->keep_feasible_only = feas_it->second.get<bool>();
    else if (feas_it->second.is<double>()) archive->keep_feasible_only = (feas_it->second.get<double>() != 0.0);
  }
}

static void parse_aoa_spec(const picojson::object& obj, AOASolverSpec* aoa) {
  auto pop_it = obj.find("population");
  if (pop_it != obj.end() && pop_it->second.is<double>()) aoa->population = static_cast<int>(pop_it->second.get<double>());
  auto iter_it = obj.find("max_iterations");
  if (iter_it != obj.end() && iter_it->second.is<double>()) aoa->max_iterations = static_cast<int>(iter_it->second.get<double>());
  auto explore_it = obj.find("exploration_rate");
  if (explore_it != obj.end() && explore_it->second.is<double>()) aoa->exploration_rate = explore_it->second.get<double>();
  auto exploit_it = obj.find("exploitation_rate");
  if (exploit_it != obj.end() && exploit_it->second.is<double>()) aoa->exploitation_rate = exploit_it->second.get<double>();
  auto anneal_start_it = obj.find("anneal_start");
  if (anneal_start_it != obj.end() && anneal_start_it->second.is<double>()) aoa->anneal_start = anneal_start_it->second.get<double>();
  auto anneal_end_it = obj.find("anneal_end");
  if (anneal_end_it != obj.end() && anneal_end_it->second.is<double>()) aoa->anneal_end = anneal_end_it->second.get<double>();
  auto repair_it = obj.find("repair_penalty");
  if (repair_it != obj.end() && repair_it->second.is<double>()) aoa->repair_penalty = repair_it->second.get<double>();
}

static void parse_moaoa_spec(const picojson::object& obj, MOAOASolverSpec* moaoa) {
  parse_aoa_spec(obj, &moaoa->base);
  auto weight_it = obj.find("weight_vectors");
  if (weight_it != obj.end() && weight_it->second.is<double>()) moaoa->weight_vectors = static_cast<int>(weight_it->second.get<double>());
  auto refresh_it = obj.find("archive_refresh");
  if (refresh_it != obj.end() && refresh_it->second.is<double>()) moaoa->archive_refresh = static_cast<int>(refresh_it->second.get<double>());
  auto archive_it = obj.find("archive");
  if (archive_it != obj.end() && archive_it->second.is<picojson::object>()) {
    parse_archive_spec(archive_it->second.get<picojson::object>(), &moaoa->archive);
  }
}

static void parse_solver(const picojson::object& root, SolverSpec* solver) {
  solver->kind = "beam";
  picojson::object obj;
  if (!get_object(root, "solver", &obj)) return;
  (void)get_string(obj, "kind", &solver->kind);
  auto aoa_it = obj.find("aoa");
  if (aoa_it != obj.end() && aoa_it->second.is<picojson::object>()) {
    parse_aoa_spec(aoa_it->second.get<picojson::object>(), &solver->aoa);
  }
  auto moaoa_it = obj.find("moaoa");
  if (moaoa_it != obj.end() && moaoa_it->second.is<picojson::object>()) {
    parse_moaoa_spec(moaoa_it->second.get<picojson::object>(), &solver->moaoa);
  }
}

static bool parse_root(const picojson::object& root, Config* out, std::string* err) {
  Config cfg;
  int version = 2; (void)get_int(root, "version", &version); cfg.version = version;
  std::string mode; if (get_string(root, "mode", &mode)) cfg.mode = mode; else cfg.mode = "assign";
  std::uint64_t seed = 0; (void)get_uint64(root, "random_seed", &seed); cfg.random_seed = seed;
  parse_solver(root, &cfg.solver);

  if (!parse_items(root, &cfg.items, err)) return false;
  if (!parse_blocks(root, &cfg.blocks, err)) return false;
  if (!parse_objective(root, &cfg.objective, err)) return false;
  if (!parse_constraints(root, &cfg.constraints, err)) return false;
  if (cfg.mode == "assign") { if (!parse_knapsack(root, &cfg.knapsack, err)) return false; }

  if (!ValidateConfig(cfg, err)) return false;
  *out = std::move(cfg);
  return true;
}

bool LoadConfigFromJsonString(const std::string& json, Config* out, std::string* err) {
  if (!out) { if (err) *err = "out is null"; return false; }
  picojson::value v; std::string perr = picojson::parse(v, json);
  if (!perr.empty()) { if (err) *err = perr; return false; }
  if (!v.is<picojson::object>()) { if (err) *err = "invalid JSON root"; return false; }
  return parse_root(v.get<picojson::object>(), out, err);
}

bool LoadConfigFromFile(const std::string& path, Config* out, std::string* err) {
  if (!out) { if (err) *err = "out is null"; return false; }
  std::ifstream in(path);
  if (!in) { if (err) *err = "failed to read file"; return false; }
  std::ostringstream ss; ss << in.rdbuf();
  return LoadConfigFromJsonString(ss.str(), out, err);
}

} // namespace v2
