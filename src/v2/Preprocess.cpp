// Copyright (c) 2025
// SPDX-License-Identifier: MIT

#include "v2/Preprocess.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace v2 {

static bool any_negative_weights(const Config& cfg) {
  for (const auto& term : cfg.objective) if (term.weight < 0.0) return true;
  return false;
}

static std::vector<double> compute_values(const Config& cfg, const HostSoA& soa) {
  const int N = soa.count; std::vector<double> v(N, 0.0);
  for (const auto& term : cfg.objective) {
    auto it = soa.attr.find(term.attr); if (it == soa.attr.end()) continue;
    const auto& a = it->second; const double w = term.weight;
    for (int i = 0; i < N; ++i) v[i] += w * a[i];
  }
  return v;
}

struct ConsSpec { const std::vector<double>* attr; double limit; double weight; };

static std::vector<ConsSpec> collect_capacity_constraints(const Config& cfg, const HostSoA& soa) {
  std::vector<ConsSpec> out;
  for (const auto& c : cfg.constraints) {
    if (c.kind != std::string("capacity") || c.attr.empty()) continue;
    auto it = soa.attr.find(c.attr); if (it == soa.attr.end()) continue;
    double w = c.soft ? c.penalty.weight : 1.0; if (w <= 0) w = 1.0;
    out.push_back(ConsSpec{ &it->second, c.limit, w });
  }
  return out;
}

static std::vector<double> compute_surrogate_alpha(const std::vector<ConsSpec>& cons, int N) {
  std::vector<double> a(N, 0.0);
  for (const auto& c : cons) {
    const double scale = (c.limit > 0.0 ? (c.weight / c.limit) : c.weight);
    for (int i = 0; i < N; ++i) a[i] += scale * (*(c.attr))[i];
  }
  return a;
}

// Scalar skyline: alpha asc, value desc; drop if value <= running best_value + eps
static std::vector<char> skyline_scalar(const std::vector<double>& alpha, const std::vector<double>& value, double eps) {
  const int N = (int)alpha.size();
  std::vector<int> order(N); for (int i = 0; i < N; ++i) order[i] = i;
  std::stable_sort(order.begin(), order.end(), [&](int i, int j){
    if (alpha[i] != alpha[j]) return alpha[i] < alpha[j];
    return value[i] > value[j];
  });
  double best = -std::numeric_limits<double>::infinity();
  std::vector<char> keep(N, 0);
  for (int idx : order) {
    if (value[idx] <= best + eps) {
      // dominated
    } else {
      keep[idx] = 1; best = std::max(best, value[idx]);
    }
  }
  return keep;
}

bool ApplyDominanceFilters(const Config& cfg, const HostSoA& in,
                           const DominanceFilterOptions& opt,
                           HostSoA* out, std::vector<int>* filtered_to_orig,
                           DominanceFilterStats* stats, std::string* err) {
  if (!out || !filtered_to_orig) { if (err) *err = "null out or mapping"; return false; }
  if (!opt.enabled) { if (stats) { stats->dropped = 0; stats->kept = in.count; stats->epsilon_used = 0.0; stats->method = "disabled"; } *out = in; filtered_to_orig->resize(in.count); for (int i=0;i<in.count;++i)(*filtered_to_orig)[i]=i; return true; }
  if (in.count <= 0) { if (stats) { stats->dropped = 0; stats->kept = 0; stats->epsilon_used = opt.epsilon; stats->method = "disabled"; } out->count = 0; out->attr.clear(); filtered_to_orig->clear(); return true; }

  // Compute scalar values; if objective has negative weights, skip filtering (safety)
  if (cfg.objective.empty() || any_negative_weights(cfg)) {
    if (stats) { stats->dropped = 0; stats->kept = in.count; stats->epsilon_used = 0.0; stats->method = "disabled"; }
    *out = in; filtered_to_orig->resize(in.count); for (int i=0;i<in.count;++i)(*filtered_to_orig)[i]=i; return true;
  }
  std::vector<double> value = compute_values(cfg, in);
  auto cons = collect_capacity_constraints(cfg, in);
  if (cons.empty()) {
    if (stats) { stats->dropped = 0; stats->kept = in.count; stats->epsilon_used = 0.0; stats->method = "disabled"; }
    *out = in; filtered_to_orig->resize(in.count); for (int i=0;i<in.count;++i)(*filtered_to_orig)[i]=i; return true;
  }

  std::vector<double> alpha;
  std::string method;
  if (cons.size() == 1 && !opt.use_surrogate) {
    alpha = *(cons[0].attr); method = "single";
  } else {
    alpha = compute_surrogate_alpha(cons, in.count); method = "surrogate";
  }

  auto keep = skyline_scalar(alpha, value, opt.epsilon);

  // Build filtered SoA in original index order
  filtered_to_orig->clear(); filtered_to_orig->reserve(in.count);
  int kept = 0; for (int i = 0; i < in.count; ++i) if (keep[i]) { ++kept; filtered_to_orig->push_back(i); }
  out->count = kept; out->attr.clear();
  for (const auto& kv : in.attr) {
    std::vector<double> vec; vec.reserve(kept);
    for (int idx : *filtered_to_orig) vec.push_back(kv.second[idx]);
    out->attr[kv.first] = std::move(vec);
  }
  if (stats) { stats->dropped = in.count - kept; stats->kept = kept; stats->epsilon_used = opt.epsilon; stats->method = method; }
  return true;
}

} // namespace v2
