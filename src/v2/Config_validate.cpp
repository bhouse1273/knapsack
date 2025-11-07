// Config_validate.cpp - Validation logic for v2::Config
#include "v2/Config.h"
#include <sstream>

namespace v2 {

bool ValidateConfig(const Config& cfg, std::string* err) {
  // Basic validation checks for knapsack configuration
  
  // Check capacity is positive
  if (cfg.capacity <= 0.0) {
    if (err) *err = "capacity must be positive";
    return false;
  }
  
  // Check items exist
  if (cfg.items.empty()) {
    if (err) *err = "no items provided";
    return false;
  }
  
  // Check each item has valid weight and value
  for (size_t i = 0; i < cfg.items.size(); ++i) {
    const auto& item = cfg.items[i];
    
    if (item.weight < 0.0) {
      if (err) {
        std::ostringstream oss;
        oss << "item " << i << " has negative weight: " << item.weight;
        *err = oss.str();
      }
      return false;
    }
    
    if (item.value < 0.0) {
      if (err) {
        std::ostringstream oss;
        oss << "item " << i << " has negative value: " << item.value;
        *err = oss.str();
      }
      return false;
    }
  }
  
  // Validation passed
  return true;
}

} // namespace v2
