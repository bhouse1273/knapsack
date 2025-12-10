// Config_validate.cpp - Validation logic for v2::Config
#include "v2/Config.h"
#include <algorithm>
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

  auto validate_source = [&](const std::string& name, const AttributeSourceSpec& spec) -> bool {
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
    switch (spec.format_kind) {
      case AttributeFormatKind::kBinary64LE:
        break;
      case AttributeFormatKind::kCSV:
        if (spec.csv_delimiter == '\0') {
          if (err) {
            std::ostringstream oss;
            oss << "attribute '" << name << "' csv delimiter must be non-zero";
            *err = oss.str();
          }
          return false;
        }
        break;
      case AttributeFormatKind::kArrow:
        if (!spec.is_file()) {
          if (err) {
            std::ostringstream oss;
            oss << "attribute '" << name << "' Arrow format requires 'file' source";
            *err = oss.str();
          }
          return false;
        }
        if (spec.column_name.empty() && spec.column_index < 0) {
          if (err) {
            std::ostringstream oss;
            oss << "attribute '" << name << "' Arrow format requires column or column_index";
            *err = oss.str();
          }
          return false;
        }
        break;
      case AttributeFormatKind::kParquet:
        if (!spec.is_file()) {
          if (err) {
            std::ostringstream oss;
            oss << "attribute '" << name << "' Parquet format supports file sources only";
            *err = oss.str();
          }
          return false;
        }
        if (spec.column_name.empty() && spec.column_index < 0) {
          if (err) {
            std::ostringstream oss;
            oss << "attribute '" << name << "' Parquet format requires column or column_index";
            *err = oss.str();
          }
          return false;
        }
        break;
      case AttributeFormatKind::kUnknown:
        if (err) {
          std::ostringstream oss;
          oss << "attribute '" << name << "' declares unsupported format '" << spec.format << "'";
          *err = oss.str();
        }
        return false;
    }
    return true;
  };

  for (const auto& kv : cfg.items.sources) {
    if (!validate_source(kv.first, kv.second)) return false;
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
  
  if (cfg.objective.empty()) {
    if (err) *err = "objective must have at least one term";
    return false;
  }

  auto validate_strategy = [&](const CostTermSpec& term, size_t index) -> bool {
    static const std::vector<std::string> allowed = {"weighted_sum", "epsilon", "pareto_component"};
    bool ok = std::find(allowed.begin(), allowed.end(), term.strategy) != allowed.end();
    if (!ok) {
      if (err) {
        std::ostringstream oss;
        oss << "objective term " << index << " has unsupported strategy '" << term.strategy << "'";
        *err = oss.str();
      }
      return false;
    }
    if (term.strategy == "epsilon" && term.epsilon <= 0.0) {
      if (err) {
        std::ostringstream oss;
        oss << "objective term " << index << " uses epsilon strategy but epsilon <= 0";
        *err = oss.str();
      }
      return false;
    }
    return true;
  };

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
    if (!validate_strategy(term, i)) return false;
  }

  auto validate_aoa = [&](const AOASolverSpec& spec, const char* label) -> bool {
    if (spec.population <= 0) {
      if (err) *err = std::string(label) + ".population must be > 0";
      return false;
    }
    if (spec.max_iterations <= 0) {
      if (err) *err = std::string(label) + ".max_iterations must be > 0";
      return false;
    }
    if (spec.exploration_rate < 0.0 || spec.exploitation_rate < 0.0) {
      if (err) *err = std::string(label) + ".exploration/exploitation rates must be >= 0";
      return false;
    }
    if (spec.anneal_start <= 0.0 || spec.anneal_end < 0.0) {
      if (err) *err = std::string(label) + ".anneal_start must be > 0 and anneal_end >= 0";
      return false;
    }
    if (spec.repair_penalty < 0.0) {
      if (err) *err = std::string(label) + ".repair_penalty must be >= 0";
      return false;
    }
    return true;
  };

  auto validate_archive = [&](const ParetoArchiveSpec& archive) -> bool {
    if (archive.max_size <= 0) {
      if (err) *err = "pareto archive max_size must be > 0";
      return false;
    }
    if (archive.dominance_epsilon < 0.0) {
      if (err) *err = "pareto archive dominance_epsilon must be >= 0";
      return false;
    }
    if (archive.diversity_metric != "crowding" && archive.diversity_metric != "hypervolume") {
      if (err) *err = "pareto archive diversity_metric must be 'crowding' or 'hypervolume'";
      return false;
    }
    return true;
  };

  auto validate_solver = [&]() -> bool {
    const std::string& kind = cfg.solver.kind;
    if (kind != "beam" && kind != "aoa" && kind != "moaoa") {
      if (err) {
        std::ostringstream oss;
        oss << "solver.kind '" << kind << "' not recognized";
        *err = oss.str();
      }
      return false;
    }
    if (kind == "aoa" || kind == "moaoa") {
      if (!validate_aoa(cfg.solver.aoa, "solver.aoa")) return false;
    }
    if (kind == "moaoa") {
      if (!validate_aoa(cfg.solver.moaoa.base, "solver.moaoa")) return false;
      if (!validate_archive(cfg.solver.moaoa.archive)) return false;
      if (cfg.solver.moaoa.weight_vectors <= 0) {
        if (err) *err = "solver.moaoa.weight_vectors must be > 0";
        return false;
      }
      if (cfg.solver.moaoa.archive_refresh <= 0) {
        if (err) *err = "solver.moaoa.archive_refresh must be > 0";
        return false;
      }
    }
    return true;
  };

  if (!validate_solver()) return false;
  
  // Validation passed
  return true;
}

} // namespace v2
