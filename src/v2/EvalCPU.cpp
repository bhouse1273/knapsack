// Copyright (c) 2025
// SPDX-License-Identifier: MIT

#include "v2/Eval.h"

#include <cmath>
#include <unordered_map>

namespace v2 {

static inline double pow_pos(double x, double p) {
  return x <= 0.0 ? 0.0 : std::pow(x, p);
}

static double objective_sum(const Config& cfg, const HostSoA& soa, const std::vector<uint8_t>& take) {
  const int n = soa.count;
  double obj = 0.0;
  for (const auto& term : cfg.objective) {
    auto it = soa.attr.find(term.attr);
    if (it == soa.attr.end()) continue; // validated earlier; be tolerant
    const auto& a = it->second;
    const double w = term.weight;
    for (int i = 0; i < n; ++i) {
      if (take[i]) obj += w * a[i];
    }
  }
  return obj;
}

static void compute_constraint_penalties(const Config& cfg, const HostSoA& soa,
                                         const std::vector<uint8_t>& take,
                                         EvalResult* out) {
  out->constraint_violations.assign(cfg.constraints.size(), 0.0);
  const int n = soa.count;
  for (size_t ci = 0; ci < cfg.constraints.size(); ++ci) {
    const auto& c = cfg.constraints[ci];
    if (c.kind == "capacity" && !c.attr.empty()) {
      auto it = soa.attr.find(c.attr);
      if (it == soa.attr.end()) continue; // validated
      const auto& a = it->second;
      double sum = 0.0;
      for (int i = 0; i < n; ++i) if (take[i]) sum += a[i];
      double viol = std::max(0.0, sum - c.limit);
      out->constraint_violations[ci] = viol;
      if (c.soft) out->penalty += c.penalty.weight * pow_pos(viol, c.penalty.power);
    }
    // future: other kinds
  }
}

bool EvaluateCPU_Select(const Config& cfg, const HostSoA& soa,
                        const CandidateSelect& cand, EvalResult* out, std::string* err) {
  if (!out) { if (err) *err = "out is null"; return false; }
  if ((int)cand.select.size() != soa.count) { if (err) *err = "candidate size mismatch"; return false; }
  out->objective = objective_sum(cfg, soa, cand.select);
  out->penalty = 0.0;
  compute_constraint_penalties(cfg, soa, cand.select, out);
  out->total = out->objective - out->penalty;
  return true;
}

static void build_take_from_assign(const CandidateAssign& cand, std::vector<uint8_t>* take) {
  take->assign(cand.assign.size(), 0u);
  for (size_t i = 0; i < cand.assign.size(); ++i) if (cand.assign[i] >= 0) (*take)[i] = 1u;
}

bool EvaluateCPU_Assign(const Config& cfg, const HostSoA& soa,
                        const CandidateAssign& cand, EvalResult* out, std::string* err) {
  if (!out) { if (err) *err = "out is null"; return false; }
  if ((int)cand.assign.size() != soa.count) { if (err) *err = "candidate size mismatch"; return false; }
  const int n = soa.count;
  const int K = cfg.knapsack.K;
  for (int i = 0; i < n; ++i) {
    int a = cand.assign[i];
    if (a >= K) { if (err) *err = "assignment out of range"; return false; }
  }
  std::vector<uint8_t> take; build_take_from_assign(cand, &take);
  out->objective = objective_sum(cfg, soa, take);
  out->penalty = 0.0;
  compute_constraint_penalties(cfg, soa, take, out);

  // Per-knapsack capacity penalties
  out->knapsack_violations.assign(K, 0.0);
  if (!cfg.knapsack.capacity_attr.empty() && (int)cfg.knapsack.capacities.size() == K) {
    auto it = soa.attr.find(cfg.knapsack.capacity_attr);
    if (it != soa.attr.end()) {
      const auto& capAttr = it->second;
      std::vector<double> sums(K, 0.0);
      for (int i = 0; i < n; ++i) {
        int a = cand.assign[i];
        if (a >= 0) sums[a] += capAttr[i];
      }
      // penalty weight/power selection: try to find matching soft constraint
      double pen_w = 1.0, pen_p = 1.0;
      for (const auto& c : cfg.constraints) {
        if (c.kind == "capacity" && c.attr == cfg.knapsack.capacity_attr && c.soft) { pen_w = c.penalty.weight; pen_p = c.penalty.power; break; }
      }
      for (int k = 0; k < K; ++k) {
        double viol = std::max(0.0, sums[k] - cfg.knapsack.capacities[k]);
        out->knapsack_violations[k] = viol;
        out->penalty += pen_w * pow_pos(viol, pen_p);
      }
    }
  }

  out->total = out->objective - out->penalty;
  return true;
}

} // namespace v2
