// Copyright (c) 2025
// SPDX-License-Identifier: MIT

#include "v2/Eval.h"
#include "metal_api.h"
#include <cstring>

namespace v2 {

// Evaluate a single candidate in select mode using Metal GPU.
// Converts the candidate to the packed format expected by Metal,
// calls the Metal evaluator, and unpacks the result.
bool EvaluateMetal_Select(const Config& cfg, const HostSoA& soa,
                          const CandidateSelect& cand, EvalResult* out, std::string* err) {
  if (!out) {
    if (err) *err = "null output pointer";
    return false;
  }
  if (cand.select.size() != soa.count) {
    if (err) *err = "candidate size mismatch";
    return false;
  }

  // Pack candidate into 2-bit format (4 items per byte)
  // For select mode: lane 0 = not selected, lane 1 = selected
  const int num_items = static_cast<int>(soa.count);
  const int bytes_per_candidate = (num_items + 3) / 4;
  std::vector<unsigned char> packed(bytes_per_candidate, 0);
  
  for (int i = 0; i < num_items; i++) {
    const uint8_t lane = cand.select[i] ? 1 : 0;
    const int byte_idx = i / 4;
    const int shift = (i % 4) * 2;
    packed[byte_idx] |= (lane << shift);
  }

  // Prepare objective terms
  std::vector<float> obj_attrs_flat;
  std::vector<float> obj_weights_vec;
  const float* obj_attrs_ptr = nullptr;
  const float* obj_weights_ptr = nullptr;
  int num_obj_terms = 0;

  if (!cfg.objective.empty()) {
    num_obj_terms = static_cast<int>(cfg.objective.empty() ? 0 : cfg.objective.size());
    obj_attrs_flat.reserve(num_obj_terms * num_items);
    obj_weights_vec.reserve(num_obj_terms);

    for (const auto& term : cfg.objective) {
      obj_weights_vec.push_back(static_cast<float>(term.weight));
      auto it = soa.attr.find(term.attr);
      if (it != soa.attr.end()) {
        // Convert double to float
        for (double val : it->second) {
          obj_attrs_flat.push_back(static_cast<float>(val));
        }
      } else {
        // Attribute not found, use zeros
        obj_attrs_flat.insert(obj_attrs_flat.end(), num_items, 0.0f);
      }
    }

    obj_attrs_ptr = obj_attrs_flat.data();
    obj_weights_ptr = obj_weights_vec.data();
  }

  // Prepare soft constraints
  std::vector<float> cons_attrs_flat;
  std::vector<float> cons_limits_vec;
  std::vector<float> cons_weights_vec;
  std::vector<float> cons_powers_vec;
  const float* cons_attrs_ptr = nullptr;
  const float* cons_limits_ptr = nullptr;
  const float* cons_weights_ptr = nullptr;
  const float* cons_powers_ptr = nullptr;
  int num_soft_constraints = 0;

  // Count soft constraints
  for (const auto& c : cfg.constraints) {
    if (c.soft) num_soft_constraints++;
  }

  if (num_soft_constraints > 0) {
    cons_attrs_flat.reserve(num_soft_constraints * num_items);
    cons_limits_vec.reserve(num_soft_constraints);
    cons_weights_vec.reserve(num_soft_constraints);
    cons_powers_vec.reserve(num_soft_constraints);

    for (const auto& c : cfg.constraints) {
      if (!c.soft) continue;
      
      cons_limits_vec.push_back(static_cast<float>(c.limit));
      cons_weights_vec.push_back(static_cast<float>(c.penalty.weight));
      cons_powers_vec.push_back(static_cast<float>(c.penalty.power));

      auto it = soa.attr.find(c.attr);
      if (it != soa.attr.end()) {
        // Convert double to float
        for (double val : it->second) {
          cons_attrs_flat.push_back(static_cast<float>(val));
        }
      } else {
        // Attribute not found, use zeros
        cons_attrs_flat.insert(cons_attrs_flat.end(), num_items, 0.0f);
      }
    }

    cons_attrs_ptr = cons_attrs_flat.data();
    cons_limits_ptr = cons_limits_vec.data();
    cons_weights_ptr = cons_weights_vec.data();
    cons_powers_ptr = cons_powers_vec.data();
  }

  // Setup Metal input
  MetalEvalIn in = {};
  in.candidates = packed.data();
  in.num_items = num_items;
  in.num_candidates = 1;
  in.num_groups = 0;  // not used in select mode
  in.penalty_coeff = 0.0f;  // not used (we use soft constraints instead)
  in.penalty_power = 0.0f;
  in.item_values = nullptr;   // not used (we use obj_attrs instead)
  in.item_weights = nullptr;  // not used (we use cons_attrs instead)
  in.group_capacities = nullptr;
  in.obj_attrs = obj_attrs_ptr;
  in.obj_weights = obj_weights_ptr;
  in.num_obj_terms = num_obj_terms;
  in.cons_attrs = cons_attrs_ptr;
  in.cons_limits = cons_limits_ptr;
  in.cons_weights = cons_weights_ptr;
  in.cons_powers = cons_powers_ptr;
  in.num_soft_constraints = num_soft_constraints;

  // Setup output buffers
  std::vector<float> obj_out(1);
  std::vector<float> pen_out(1);
  MetalEvalOut metal_out = {};
  metal_out.obj = obj_out.data();
  metal_out.soft_penalty = pen_out.data();

  // Call Metal evaluator
  char metal_err[512] = {0};
  if (knapsack_metal_eval(&in, &metal_out, metal_err, sizeof(metal_err)) != 0) {
    if (err) *err = std::string("Metal eval failed: ") + metal_err;
    return false;
  }

  // Unpack results
  out->objective = static_cast<double>(obj_out[0]);
  out->penalty = static_cast<double>(pen_out[0]);
  out->total = out->objective - out->penalty;

  // Compute constraint violations for reporting
  out->constraint_violations.clear();
  out->constraint_violations.reserve(cfg.constraints.size());
  
  for (const auto& c : cfg.constraints) {
    auto it = soa.attr.find(c.attr);
    if (it == soa.attr.end()) {
      out->constraint_violations.push_back(0.0);
      continue;
    }
    
    double sum = 0.0;
    for (size_t i = 0; i < cand.select.size(); i++) {
      if (cand.select[i]) {
        sum += it->second[i];
      }
    }
    
    double violation = std::max(0.0, sum - c.limit);
    out->constraint_violations.push_back(violation);
  }

  out->knapsack_violations.clear();  // not used in select mode

  return true;
}

// Evaluate multiple candidates in parallel using Metal GPU.
// This is the primary use case for GPU acceleration.
bool EvaluateMetal_Batch(const Config& cfg, const HostSoA& soa,
                         const std::vector<CandidateSelect>& candidates,
                         std::vector<EvalResult>* results, std::string* err) {
  if (!results) {
    if (err) *err = "null output pointer";
    return false;
  }
  if (candidates.empty()) {
    results->clear();
    return true;
  }

  const int num_items = static_cast<int>(soa.count);
  const int num_candidates = static_cast<int>(candidates.size());
  const int bytes_per_candidate = (num_items + 3) / 4;

  // Validate all candidates
  for (const auto& cand : candidates) {
    if (cand.select.size() != soa.count) {
      if (err) *err = "candidate size mismatch";
      return false;
    }
  }

  // Pack all candidates into one buffer (2 bits per item)
  std::vector<unsigned char> packed(num_candidates * bytes_per_candidate, 0);
  
  for (int c = 0; c < num_candidates; c++) {
    const auto& cand = candidates[c];
    const int base = c * bytes_per_candidate;
    
    for (int i = 0; i < num_items; i++) {
      const uint8_t lane = cand.select[i] ? 1 : 0;
      const int byte_idx = base + (i / 4);
      const int shift = (i % 4) * 2;
      packed[byte_idx] |= (lane << shift);
    }
  }

  // Prepare objective terms (same as single evaluation)
  std::vector<float> obj_attrs_flat;
  std::vector<float> obj_weights_vec;
  const float* obj_attrs_ptr = nullptr;
  const float* obj_weights_ptr = nullptr;
  int num_obj_terms = 0;

  if (!cfg.objective.empty()) {
    num_obj_terms = static_cast<int>(cfg.objective.size());
    obj_attrs_flat.reserve(num_obj_terms * num_items);
    obj_weights_vec.reserve(num_obj_terms);

    for (const auto& term : cfg.objective) {
      obj_weights_vec.push_back(static_cast<float>(term.weight));
      auto it = soa.attr.find(term.attr);
      if (it != soa.attr.end()) {
        for (double val : it->second) {
          obj_attrs_flat.push_back(static_cast<float>(val));
        }
      } else {
        obj_attrs_flat.insert(obj_attrs_flat.end(), num_items, 0.0f);
      }
    }

    obj_attrs_ptr = obj_attrs_flat.data();
    obj_weights_ptr = obj_weights_vec.data();
  }

  // Prepare soft constraints
  std::vector<float> cons_attrs_flat;
  std::vector<float> cons_limits_vec;
  std::vector<float> cons_weights_vec;
  std::vector<float> cons_powers_vec;
  const float* cons_attrs_ptr = nullptr;
  const float* cons_limits_ptr = nullptr;
  const float* cons_weights_ptr = nullptr;
  const float* cons_powers_ptr = nullptr;
  int num_soft_constraints = 0;

  for (const auto& c : cfg.constraints) {
    if (c.soft) num_soft_constraints++;
  }

  if (num_soft_constraints > 0) {
    cons_attrs_flat.reserve(num_soft_constraints * num_items);
    cons_limits_vec.reserve(num_soft_constraints);
    cons_weights_vec.reserve(num_soft_constraints);
    cons_powers_vec.reserve(num_soft_constraints);

    for (const auto& c : cfg.constraints) {
      if (!c.soft) continue;
      
      cons_limits_vec.push_back(static_cast<float>(c.limit));
      cons_weights_vec.push_back(static_cast<float>(c.penalty.weight));
      cons_powers_vec.push_back(static_cast<float>(c.penalty.power));

      auto it = soa.attr.find(c.attr);
      if (it != soa.attr.end()) {
        for (double val : it->second) {
          cons_attrs_flat.push_back(static_cast<float>(val));
        }
      } else {
        cons_attrs_flat.insert(cons_attrs_flat.end(), num_items, 0.0f);
      }
    }

    cons_attrs_ptr = cons_attrs_flat.data();
    cons_limits_ptr = cons_limits_vec.data();
    cons_weights_ptr = cons_weights_vec.data();
    cons_powers_ptr = cons_powers_vec.data();
  }

  // Setup Metal input for batch
  MetalEvalIn in = {};
  in.candidates = packed.data();
  in.num_items = num_items;
  in.num_candidates = num_candidates;  // Multiple candidates!
  in.num_groups = 0;
  in.penalty_coeff = 0.0f;
  in.penalty_power = 0.0f;
  in.item_values = nullptr;
  in.item_weights = nullptr;
  in.group_capacities = nullptr;
  in.obj_attrs = obj_attrs_ptr;
  in.obj_weights = obj_weights_ptr;
  in.num_obj_terms = num_obj_terms;
  in.cons_attrs = cons_attrs_ptr;
  in.cons_limits = cons_limits_ptr;
  in.cons_weights = cons_weights_ptr;
  in.cons_powers = cons_powers_ptr;
  in.num_soft_constraints = num_soft_constraints;

  // Setup output buffers for all candidates
  std::vector<float> obj_out(num_candidates);
  std::vector<float> pen_out(num_candidates);
  MetalEvalOut metal_out = {};
  metal_out.obj = obj_out.data();
  metal_out.soft_penalty = pen_out.data();

  // Call Metal evaluator (evaluates all candidates in parallel!)
  char metal_err[512] = {0};
  if (knapsack_metal_eval(&in, &metal_out, metal_err, sizeof(metal_err)) != 0) {
    if (err) *err = std::string("Metal batch eval failed: ") + metal_err;
    return false;
  }

  // Unpack results for all candidates
  results->clear();
  results->reserve(num_candidates);

  for (int c = 0; c < num_candidates; c++) {
    EvalResult result;
    result.objective = static_cast<double>(obj_out[c]);
    result.penalty = static_cast<double>(pen_out[c]);
    result.total = result.objective - result.penalty;

    // Compute constraint violations
    result.constraint_violations.reserve(cfg.constraints.size());
    
    const auto& cand = candidates[c];
    for (const auto& constraint : cfg.constraints) {
      auto it = soa.attr.find(constraint.attr);
      if (it == soa.attr.end()) {
        result.constraint_violations.push_back(0.0);
        continue;
      }
      
      double sum = 0.0;
      for (size_t i = 0; i < cand.select.size(); i++) {
        if (cand.select[i]) {
          sum += it->second[i];
        }
      }
      
      double violation = std::max(0.0, sum - constraint.limit);
      result.constraint_violations.push_back(violation);
    }

    result.knapsack_violations.clear();
    results->push_back(result);
  }

  return true;
}

} // namespace v2
