// Config_validate.cpp - Validation logic for v2::Config
#include "v2/Config.h"
#include <sstream>

namespace v2 {

bool ValidateConfig(const Config& cfg, std::string* err) {
  // Basic validation checks for knapsack configuration
  
  // Check items exist
  if (cfg.items.count <= 0) {
    if (err) *err = "no items provided (items.count must be > 0)";
    return false;
  }
  
  // Check that all attribute arrays match the item count
  for (const auto& kv : cfg.items.attributes) {
    const std::string& attr_name = kv.first;
    const std::vector<double>& values = kv.second;
    
    if (values.size() != static_cast<size_t>(cfg.items.count)) {
      if (err) {
        std::ostringstream oss;
        oss << "attribute '" << attr_name << "' has " << values.size() 
            << " values but items.count is " << cfg.items.count;
        *err = oss.str();
      }
      return false;
    }
  }

  for (const auto& kv : cfg.items.sources) {
    const auto& name = kv.first;
    const auto& spec = kv.second;
    if (cfg.items.attributes.count(name)) {
      if (err) {
        std::ostringstream oss;
        oss << "attribute '" << name << "' provided both inline and as external source";
        *err = oss.str();
      }
      return false;
    }
    if (spec.kind == AttributeSourceKind::kInline) {
      if (err) {
        std::ostringstream oss;
        oss << "attribute '" << name << "' external spec must declare 'file' or 'stream' source";
        *err = oss.str();
      }
      return false;
    }
    if (spec.is_file() && spec.path.empty() && spec.chunks.empty()) {
      if (err) {
        std::ostringstream oss;
        oss << "file-backed attribute '" << name << "' missing path or chunks";
        *err = oss.str();
      }
      return false;
    }
    if (spec.is_stream() && spec.channel.empty() && spec.chunks.empty()) {
      if (err) {
        std::ostringstream oss;
        oss << "stream-backed attribute '" << name << "' missing channel or chunks";
        *err = oss.str();
      }
      return false;
    }
    if (spec.format != "binary64_le") {
      if (err) {
        std::ostringstream oss;
        oss << "attribute '" << name << "' declares unsupported format '" << spec.format << "'";
        *err = oss.str();
      }
      return false;
    }
  }
  
  // For assign mode, validate knapsack specs
  if (cfg.mode == "assign") {
    if (cfg.knapsack.K <= 0) {
      if (err) *err = "knapsack.K must be positive for assign mode";
      return false;
    }
    
    if (cfg.knapsack.capacities.size() != static_cast<size_t>(cfg.knapsack.K)) {
      if (err) {
        std::ostringstream oss;
        oss << "knapsack.capacities size (" << cfg.knapsack.capacities.size()
            << ") does not match K (" << cfg.knapsack.K << ")";
        *err = oss.str();
      }
      return false;
    }
    
    // Check all capacities are positive
    for (size_t i = 0; i < cfg.knapsack.capacities.size(); ++i) {
      if (cfg.knapsack.capacities[i] <= 0.0) {
        if (err) {
          std::ostringstream oss;
          oss << "knapsack capacity " << i << " must be positive, got " 
              << cfg.knapsack.capacities[i];
          *err = oss.str();
        }
        return false;
      }
    }
    
    // Check that capacity_attr exists in items.attributes
    if (cfg.knapsack.capacity_attr.empty()) {
      if (err) *err = "knapsack.capacity_attr must be specified for assign mode";
      return false;
    }
    
    if (!cfg.items.HasAttribute(cfg.knapsack.capacity_attr)) {
      if (err) {
        std::ostringstream oss;
        oss << "knapsack.capacity_attr '" << cfg.knapsack.capacity_attr 
            << "' not found in items.attributes";
        *err = oss.str();
      }
      return false;
    }
  }
  
  // Validate constraints reference valid attributes
  for (size_t i = 0; i < cfg.constraints.size(); ++i) {
    const auto& constraint = cfg.constraints[i];
    
    if (!constraint.attr.empty()) {
      if (!cfg.items.HasAttribute(constraint.attr)) {
        if (err) {
          std::ostringstream oss;
          oss << "constraint " << i << " references unknown attribute '" 
              << constraint.attr << "'";
          *err = oss.str();
        }
        return false;
      }
    }
    
    if (constraint.limit < 0.0) {
      if (err) {
        std::ostringstream oss;
        oss << "constraint " << i << " has negative limit: " << constraint.limit;
        *err = oss.str();
      }
      return false;
    }
  }
  
  // Validate objective terms reference valid attributes
  for (size_t i = 0; i < cfg.objective.size(); ++i) {
    const auto& term = cfg.objective[i];
    
    if (!cfg.items.HasAttribute(term.attr)) {
      if (err) {
        std::ostringstream oss;
        oss << "objective term " << i << " references unknown attribute '" 
            << term.attr << "'";
        *err = oss.str();
      }
      return false;
    }
  }
  
  // Validation passed
  return true;
}

} // namespace v2
