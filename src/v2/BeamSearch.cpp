// Copyright (c) 2025
// SPDX-License-Identifier: MIT

#include "v2/Engine.h"
#include "v2/Eval.h"
#include "v2/Preprocess.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <string>

// Metal API (only available on Apple with Metal support enabled)
#if defined(__APPLE__) && !defined(KNAPSACK_CPU_ONLY)
#include "metal_api.h"
#endif

namespace v2 {

static bool read_file_first_of(const std::vector<std::string>& paths, std::string* out) {
  for (const auto& p : paths) {
    std::ifstream in(p, std::ios::binary);
    if (!in) continue;
    out->assign((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    if (!out->empty()) return true;
  }
  return false;
}

static double sum_weight(const std::vector<uint8_t>& sel, const std::vector<double>& w) {
  double s = 0.0; for (size_t i = 0; i < sel.size(); ++i) if (sel[i]) s += w[i]; return s;
}

static std::vector<double> surrogate_weight(const std::vector<std::vector<double>>& consAttrs,
                                            const std::vector<double>& consLimits,
                                            const std::vector<double>& consWeights) {
  const int N = consAttrs.empty() ? 0 : (int)consAttrs[0].size();
  std::vector<double> surr(N, 0.0);
  for (size_t c = 0; c < consAttrs.size(); ++c) {
    const double alpha = (c < consWeights.size() && c < consLimits.size()) ? (consWeights[c] / std::max(1e-9, consLimits[c])) : 1.0;
    for (int i = 0; i < N; ++i) surr[i] += alpha * consAttrs[c][i];
  }
  for (int i = 0; i < N; ++i) surr[i] = std::max(1e-12, surr[i]);
  return surr;
}

static bool feasible_multi(const std::vector<uint8_t>& sel,
                           const std::vector<std::vector<double>>& consAttrs,
                           const std::vector<double>& consLimits) {
  const int N = (int)sel.size();
  for (size_t c = 0; c < consAttrs.size(); ++c) {
    double s = 0.0; for (int i = 0; i < N; ++i) if (sel[i]) s += consAttrs[c][i];
    if (s > consLimits[c] + 1e-9) return false;
  }
  return true;
}

static void clamp_to_constraints(std::vector<uint8_t>& sel,
                                 const std::vector<double>& value,
                                 const std::vector<std::vector<double>>& consAttrs,
                                 const std::vector<double>& consLimits,
                                 const std::vector<double>& consWeights) {
  if (consAttrs.empty()) return;
  const int N = (int)sel.size();
  auto surr = surrogate_weight(consAttrs, consLimits, consWeights);
  // compute density v/surr for selected items and drop lowest until feasible
  std::vector<std::pair<double,int>> dens;
  dens.reserve(N);
  for (int i = 0; i < N; ++i) if (sel[i]) dens.emplace_back(value[i] / surr[i], i);
  std::sort(dens.begin(), dens.end()); // ascending density
  size_t idx = 0;
  while (!feasible_multi(sel, consAttrs, consLimits) && idx < dens.size()) {
    int i = dens[idx++].second; if (!sel[i]) continue; sel[i] = 0;
  }
}

static std::vector<unsigned char> pack2bit(const std::vector<std::vector<uint8_t>>& cand_bits, int num_items) {
  const int bytes_per = (num_items + 3) / 4;
  std::vector<unsigned char> out(bytes_per * cand_bits.size(), 0);
  for (size_t c = 0; c < cand_bits.size(); ++c) {
    for (int i = 0; i < num_items; ++i) {
      const unsigned lane = cand_bits[c][i] ? 1u : 0u;
      const int byteIdx = (int)c * bytes_per + (i >> 2);
      const int shift = (i & 3) * 2;
      out[byteIdx] |= (unsigned char)((lane & 0x3u) << shift);
    }
  }
  return out;
}

static int argmax_score(const std::vector<float>& obj, const std::vector<float>& pen) {
  int best = 0; float bestS = obj.empty() ? -1e30f : (obj[0] - pen[0]);
  for (int i = 1; i < (int)obj.size(); ++i) { float s = obj[i] - pen[i]; if (s > bestS) { bestS = s; best = i; } }
  return best;
}

bool SolveBeamSelect(const Config& cfg, const HostSoA& soa, const SolverOptions& opt,
                     BeamResult* out, std::string* err) {
  if (!out) { if (err) *err = "out is null"; return false; }
  if (cfg.mode != "select") { if (err) *err = "SolveBeamSelect expects cfg.mode == select"; return false; }
  // Optional dominance filtering
  HostSoA soaF = soa; std::vector<int> f2o; DominanceFilterStats dfStats; bool filtered = false;
  if (opt.enable_dominance_filter) {
    DominanceFilterOptions dopt; dopt.enabled = true; dopt.epsilon = opt.dom_eps; dopt.use_surrogate = opt.dom_use_surrogate;
    HostSoA tmp; std::string e;
    if (ApplyDominanceFilters(cfg, soa, dopt, &tmp, &f2o, &dfStats, &e)) {
      if (tmp.count > 0 && tmp.count < soa.count) { soaF = std::move(tmp); filtered = true; }
      if (opt.debug) {
        std::cout << "[dominance] method=" << dfStats.method << " kept=" << dfStats.kept << "/" << soa.count << " dropped=" << dfStats.dropped << " eps=" << dfStats.epsilon_used << "\n";
      }
    } else if (opt.debug) {
      std::cout << "[dominance] skipped: " << e << "\n";
    }
  }

  const int N = soaF.count;
  if (N <= 0) { if (err) *err = "no items"; return false; }
  if (cfg.objective.empty()) { if (err) *err = "objective empty"; return false; }
  // Use first objective term as value
  const auto& valueAttr = cfg.objective[0].attr;
  auto itV = soaF.attr.find(valueAttr); if (itV == soaF.attr.end()) { if (err) *err = "objective attr not found"; return false; }
  std::vector<double> values = itV->second;
  // Capacity from first matching capacity constraint
  double limit = 0.0, penW = 1.0, penP = 1.0;
  std::vector<double> weights;
  for (const auto& c : cfg.constraints) {
    if (c.kind == "capacity" && !c.attr.empty()) {
  auto itW = soaF.attr.find(c.attr); if (itW == soaF.attr.end()) continue;
      weights = itW->second; limit = c.limit; if (c.soft) { penW = c.penalty.weight; penP = c.penalty.power; }
      break;
    }
  }
  if (weights.empty()) { if (err) *err = "capacity constraint with attr not found"; return false; }

  // Prepare Metal pipeline
  bool useMetal = false;
#if defined(__APPLE__) && !defined(KNAPSACK_CPU_ONLY)
  std::string msl;
  if (read_file_first_of({
        "../kernels/metal/shaders/eval_block_candidates.metal",
        "kernels/metal/shaders/eval_block_candidates.metal",
        "../../kernels/metal/shaders/eval_block_candidates.metal"}, &msl)) {
    if (!msl.empty()) {
      if (knapsack_metal_init_from_source(msl.data(), msl.size(), nullptr, 0) == 0) useMetal = true;
    }
  }
#endif

  // Precompute optional multi-term objective and soft constraints for GPU evaluation.
  // Objective terms
  const int T = (int)cfg.objective.size();
  std::vector<float> objW;
  std::vector<float> objAttrFlat;
  if (T > 0) {
    objW.resize(T);
    objAttrFlat.resize((size_t)T * (size_t)N);
    for (int t = 0; t < T; ++t) {
      objW[t] = (float)cfg.objective[t].weight;
  auto itA = soaF.attr.find(cfg.objective[t].attr);
      if (itA != soa.attr.end()) {
        for (int i = 0; i < N; ++i) objAttrFlat[(size_t)t * (size_t)N + (size_t)i] = (float)itA->second[i];
      }
    }
  }
  // Soft constraints (capacity kind)
  std::vector<int> consIdx; consIdx.reserve(cfg.constraints.size());
  for (int i = 0; i < (int)cfg.constraints.size(); ++i) {
    if (cfg.constraints[i].kind == std::string("capacity") && cfg.constraints[i].soft) consIdx.push_back(i);
  }
  const int C = (int)consIdx.size();
  std::vector<float> consAttrFlat; std::vector<float> consLimits; std::vector<float> consW; std::vector<float> consP;
  if (C > 0) {
    consAttrFlat.resize((size_t)C * (size_t)N);
    consLimits.resize(C); consW.resize(C); consP.resize(C);
    for (int j = 0; j < C; ++j) {
      const auto& c = cfg.constraints[consIdx[j]];
      consLimits[j] = (float)c.limit; consW[j] = (float)c.penalty.weight; consP[j] = (float)c.penalty.power;
  auto itCA = soaF.attr.find(c.attr);
      if (itCA != soa.attr.end()) {
        for (int i = 0; i < N; ++i) consAttrFlat[(size_t)j * (size_t)N + (size_t)i] = (float)itCA->second[i];
      }
    }
  }

  // Build constraint arrays for multi-constraint handling (select-mode soft constraints only)
  std::vector<std::vector<double>> consAttrs;
  std::vector<double> consLimitsD, consWeightsD;
  if (C > 0) {
    consAttrs.resize(C, std::vector<double>(N, 0.0));
    consLimitsD.resize(C); consWeightsD.resize(C);
    for (int j = 0; j < C; ++j) {
      consLimitsD[j] = (double)consLimits[j]; consWeightsD[j] = (double)consW[j];
      for (int i = 0; i < N; ++i) consAttrs[j][i] = (double)consAttrFlat[(size_t)j * (size_t)N + (size_t)i];
    }
  }

  // Seed: greedy by value/surrogate-density for multi-constraints; otherwise classic single-constraint density.
  std::vector<uint8_t> seed(N, 0);
  if (C > 0) {
    auto surr = surrogate_weight(consAttrs, consLimitsD, consWeightsD);
    std::vector<std::pair<double,int>> dens; dens.reserve(N);
    for (int i = 0; i < N; ++i) dens.emplace_back(values[i] / std::max(1e-9, surr[i]), i);
    std::sort(dens.begin(), dens.end(), [](auto& a, auto& b){ return a.first > b.first; });
    for (auto& kv : dens) { int i = kv.second; seed[i] = 1; if (!feasible_multi(seed, consAttrs, consLimitsD)) { seed[i] = 0; } }
  } else {
    std::vector<std::pair<double,int>> dens; dens.reserve(N);
    for (int i = 0; i < N; ++i) dens.emplace_back(values[i] / std::max(1e-9, weights[i]), i);
    std::sort(dens.begin(), dens.end(), [](auto& a, auto& b){ return a.first > b.first; });
    std::vector<uint8_t> tmp(N, 0); double wsum = 0.0;
    for (auto& kv : dens) { int i = kv.second; if (wsum + weights[i] <= limit) { tmp[i] = 1; wsum += weights[i]; } }
    seed.swap(tmp);
  }

  // Beam container
  std::vector<std::vector<uint8_t>> beam; beam.push_back(seed);
  std::mt19937 rng(opt.seed);

  for (int it = 0; it < opt.iters; ++it) {
    std::vector<std::vector<uint8_t>> cand;
    // expand
    for (const auto& b : beam) {
      // local flips
      for (int i = 0; i < N; ++i) {
        std::vector<uint8_t> v = b; v[i] = v[i] ? 0u : 1u;
        if (C > 0) clamp_to_constraints(v, values, consAttrs, consLimitsD, consWeightsD);
        else {
          // single-constraint clamp
          double wsum = sum_weight(v, weights);
          if (wsum > limit) {
            // remove lowest density until feasible
            std::vector<std::pair<double,int>> dens; for (int j = 0; j < N; ++j) if (v[j]) dens.emplace_back(values[j] / std::max(1e-9, weights[j]), j);
            std::sort(dens.begin(), dens.end()); for (auto& kv : dens) { if (sum_weight(v, weights) <= limit) break; v[kv.second] = 0; }
          }
        }
        cand.push_back(std::move(v));
        if ((int)cand.size() >= opt.beam_width * 8) break;
      }
      if ((int)cand.size() >= opt.beam_width * 8) break;
    }
    // random candidates
    std::bernoulli_distribution pick(0.4);
    while ((int)cand.size() < opt.beam_width * 8) {
      std::vector<uint8_t> v(N, 0); for (int i = 0; i < N; ++i) v[i] = pick(rng) ? 1u : 0u;
      if (C > 0) clamp_to_constraints(v, values, consAttrs, consLimitsD, consWeightsD);
      else {
        double wsum = sum_weight(v, weights);
        if (wsum > limit) { std::vector<std::pair<double,int>> dens; for (int j = 0; j < N; ++j) if (v[j]) dens.emplace_back(values[j] / std::max(1e-9, weights[j]), j); std::sort(dens.begin(), dens.end()); for (auto& kv : dens) { if (sum_weight(v, weights) <= limit) break; v[kv.second] = 0; } }
      }
      cand.push_back(std::move(v));
    }

    // Evaluate
    const int M = (int)cand.size();
    std::vector<float> obj(M, 0.0f), pen(M, 0.0f);
#if defined(__APPLE__) && !defined(KNAPSACK_CPU_ONLY)
    if (useMetal) {
      auto packed = pack2bit(cand, N);
      const int bytes_per = (N + 3) / 4;
      std::vector<float> fvals(N), fw(N), caps(1);
      for (int i = 0; i < N; ++i) { fvals[i] = (float)values[i]; fw[i] = (float)weights[i]; }
      caps[0] = (float)limit;
      MetalEvalIn in{}; in.candidates = packed.data(); in.num_items = N; in.num_candidates = M;
      // If multi-term objective defined, use it; else legacy single-term value
      if (T > 0) { in.obj_attrs = objAttrFlat.data(); in.obj_weights = objW.data(); in.num_obj_terms = T; }
      else { in.item_values = fvals.data(); }
      // Soft constraints
      if (C > 0) { in.cons_attrs = consAttrFlat.data(); in.cons_limits = consLimits.data(); in.cons_weights = consW.data(); in.cons_powers = consP.data(); in.num_soft_constraints = C; }
  // Legacy per-group capacity penalty for single knapsack selection
  in.item_weights = fw.data(); in.group_capacities = caps.data(); in.num_groups = 1; in.penalty_coeff = (float)penW; in.penalty_power = (float)penP;
      MetalEvalOut out{ obj.data(), pen.data() };
      (void)bytes_per; // silence unused
      if (knapsack_metal_eval(&in, &out, nullptr, 0) != 0) {
        useMetal = false; // fall back to CPU
      }
    }
    if (!useMetal)
#endif
    {
      // CPU fallback: EvaluateCPU_Select
      for (int c = 0; c < M; ++c) {
        CandidateSelect cs; cs.select = cand[c]; EvalResult r; std::string e; if (EvaluateCPU_Select(cfg, soaF, cs, &r, &e)) { obj[c] = (float)r.objective; pen[c] = (float)r.penalty; }
      }
    }

    // pick top-K by obj-pen
    std::vector<int> idx(M); std::iota(idx.begin(), idx.end(), 0);
    std::partial_sort(idx.begin(), idx.begin() + std::min(opt.beam_width, M), idx.end(), [&](int a, int b){ return (obj[a]-pen[a]) > (obj[b]-pen[b]); });
    if (opt.debug && M > 0) {
      int besti = idx[0];
      double bestTotal = (double)obj[besti] - (double)pen[besti];
      // Compute per-constraint slack and penalty parts for the best candidate (debug only)
      std::vector<double> slacks;
      std::vector<double> pen_parts;
      if (C > 0) {
        slacks.resize(C, 0.0);
        pen_parts.resize(C, 0.0);
        const auto& bc = cand[besti];
        for (int j = 0; j < C; ++j) {
          double used = 0.0;
          for (int i = 0; i < N; ++i) if (bc[i]) used += (double)consAttrFlat[(size_t)j * (size_t)N + (size_t)i];
          double slack = (double)consLimits[j] - used;
          slacks[j] = slack;
          double viol = slack < 0.0 ? -slack : 0.0;
          // penalty part = w * viol^p
          pen_parts[j] = (double)consW[j] * (viol > 0.0 ? std::pow(viol, (double)consP[j]) : 0.0);
        }
      }
      std::cout << "[beam] iter=" << it << " best_total=" << bestTotal << " obj=" << obj[besti] << " pen=" << pen[besti];
      if (!slacks.empty()) {
        std::cout << " slacks=[";
        for (int j = 0; j < (int)slacks.size(); ++j) { if (j) std::cout << ", "; std::cout << slacks[j]; }
        std::cout << "] pen_parts=[";
        for (int j = 0; j < (int)pen_parts.size(); ++j) { if (j) std::cout << ", "; std::cout << pen_parts[j]; }
        std::cout << "]";
      }
      std::cout << "\n";
    }
    beam.clear();
    for (int i = 0; i < std::min(opt.beam_width, M); ++i) beam.push_back(std::move(cand[idx[i]]));
  }

  // Final best
  std::vector<float> obj(beam.size(), 0.0f), pen(beam.size(), 0.0f);
  bool useMetalFinal = false;
#if defined(__APPLE__) && !defined(KNAPSACK_CPU_ONLY)
  useMetalFinal = true;
  if (useMetalFinal) {
    auto packed = pack2bit(beam, N);
    std::vector<float> fvals(N), fw(N), caps(1); for (int i = 0; i < N; ++i) { fvals[i] = (float)values[i]; fw[i] = (float)weights[i]; } caps[0] = (float)limit;
    MetalEvalIn in{}; in.candidates = packed.data(); in.num_items = N; in.num_candidates = (int)beam.size();
    if (T > 0) { in.obj_attrs = objAttrFlat.data(); in.obj_weights = objW.data(); in.num_obj_terms = T; } else { in.item_values = fvals.data(); }
    if (C > 0) { in.cons_attrs = consAttrFlat.data(); in.cons_limits = consLimits.data(); in.cons_weights = consW.data(); in.cons_powers = consP.data(); in.num_soft_constraints = C; }
  in.item_weights = fw.data(); in.group_capacities = caps.data(); in.num_groups = 1; in.penalty_coeff = (float)penW; in.penalty_power = (float)penP; MetalEvalOut out{ obj.data(), pen.data() };
    if (knapsack_metal_eval(&in, &out, nullptr, 0) != 0) { useMetalFinal = false; }
  }
  if (!useMetalFinal)
#endif
  {
    for (int c = 0; c < (int)beam.size(); ++c) { CandidateSelect cs; cs.select = beam[c]; EvalResult r; std::string e; if (EvaluateCPU_Select(cfg, soaF, cs, &r, &e)) { obj[c] = (float)r.objective; pen[c] = (float)r.penalty; } }
  }
  int best = argmax_score(obj, pen);
  // Map selection back to original indexing if filtered
  if (filtered) {
    std::vector<uint8_t> full((size_t)soa.count, 0u);
    const auto& bsel = beam[best];
    for (int i = 0; i < (int)bsel.size(); ++i) if (bsel[i]) full[f2o[i]] = 1u;
    out->best_select = std::move(full);
  } else {
    out->best_select = beam[best];
  }
  out->objective = obj[best]; out->penalty = pen[best]; out->total = obj[best] - pen[best];
  return true;
}

// Scout mode: beam search with active set tracking for exact solver handoff
bool SolveBeamScout(const Config& cfg, const HostSoA& soa, const SolverOptions& opt,
                    ScoutResult* out, std::string* err) {
  if (!out) { if (err) *err = "out is null"; return false; }
  
  auto start = std::chrono::high_resolution_clock::now();
  
  // Run standard beam search with dominance filters
  BeamResult beam_result;
  SolverOptions beam_opt = opt;
  beam_opt.enable_dominance_filter = true; // Always enable for scout mode
  
  if (!SolveBeamSelect(cfg, soa, beam_opt, &beam_result, err)) {
    return false;
  }
  
  // Copy beam result to scout result
  out->best_select = beam_result.best_select;
  out->objective = beam_result.objective;
  out->penalty = beam_result.penalty;
  out->total = beam_result.total;
  out->original_item_count = soa.count;
  
  // Track active items across top-K beam candidates
  // For now, we'll track items that appear in the best solution
  // In a full implementation, we'd track across all beam iterations
  const int N = soa.count;
  std::vector<int> item_count(N, 0);
  
  // Count how many times each item was selected in the best solution
  for (int i = 0; i < N; ++i) {
    if (out->best_select[i]) {
      item_count[i] = 1;
    }
  }
  
  // Build active set based on threshold
  out->item_frequency.resize(N);
  for (int i = 0; i < N; ++i) {
    out->item_frequency[i] = (double)item_count[i];
    if (out->item_frequency[i] >= opt.scout_threshold) {
      out->active_items.push_back(i);
    }
  }
  
  out->active_item_count = (int)out->active_items.size();
  
  auto end = std::chrono::high_resolution_clock::now();
  out->solve_time_ms = std::chrono::duration<double, std::milli>(end - start).count();
  
  if (opt.debug) {
    std::cout << "[scout] original=" << out->original_item_count 
              << " active=" << out->active_item_count
              << " reduction=" << std::fixed << std::setprecision(1)
              << (100.0 * (1.0 - (double)out->active_item_count / (double)out->original_item_count)) << "%"
              << " time=" << out->solve_time_ms << "ms\n";
  }
  
  return true;
}

} // namespace v2
