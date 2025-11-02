#include <iostream>
#include <string>
#include "v2/Config.h"

int main(int argc, char** argv) {
  std::string path = argc > 1 ? argv[1] : std::string("docs/v2/example_villages.json");
  v2::Config cfg; std::string err;
  if (!v2::LoadConfigFromFile(path, &cfg, &err)) {
    std::cerr << "Failed to load config: " << err << "\n";
    return 1;
  }
  std::cout << "Loaded config v" << cfg.version << ", mode=" << cfg.mode
            << ", items=" << cfg.items.count << ", blocks=" << cfg.blocks.size()
            << ", objective_terms=" << cfg.objective.size() << ", constraints=" << cfg.constraints.size() << "\n";
  if (cfg.mode == "assign") {
    std::cout << "K=" << cfg.knapsack.K << ", capacities=[";
    for (size_t i = 0; i < cfg.knapsack.capacities.size(); ++i) {
      if (i) std::cout << ",";
      std::cout << cfg.knapsack.capacities[i];
    }
    std::cout << "] capacity_attr=" << cfg.knapsack.capacity_attr << "\n";
  }
  // Print attributes available
  std::cout << "attributes:";
  for (const auto& kv : cfg.items.attributes) {
    std::cout << " " << kv.first << "(" << kv.second.size() << ")";
  }
  std::cout << "\n";
  return 0;
}
