// Demo: Scout mode for exact solver handoff
#include <iostream>
#include <fstream>
#include <iomanip>
#include <string>
#include <vector>

#include "v2/Config.h"
#include "v2/Data.h"
#include "v2/Engine.h"

static int fail(const std::string& msg) {
  std::cerr << "FAIL: " << msg << "\n";
  return 1;
}

static bool read_file(const std::string& path, std::string* out) {
  std::ifstream in(path, std::ios::binary);
  if (!in) return false;
  out->assign((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
  return !out->empty();
}

int main() {
  // Load config - try multiple paths
  v2::Config cfg;
  std::string err;
  std::vector<std::string> paths = {
    "docs/v2/example_select.json",
    "../docs/v2/example_select.json",
    "../../docs/v2/example_select.json"
  };
  
  bool loaded = false;
  for (const auto& path : paths) {
    if (v2::LoadConfigFromFile(path, &cfg, &err)) {
      loaded = true;
      std::cout << "Loaded config from: " << path << "\n";
      break;
    }
  }
  
  if (!loaded) {
    return fail("LoadConfig: " + err);
  }
  
  if (cfg.mode != "select") {
    return fail("expected select mode");
  }
  
  // Build SoA
  v2::HostSoA soa;
  if (!v2::BuildHostSoA(cfg, &soa, &err)) {
    return fail("BuildHostSoA: " + err);
  }
  
  std::cout << "=== Scout Mode Demo ===\n";
  std::cout << "Original items: " << soa.count << "\n\n";
  
  // Run scout mode
  v2::SolverOptions opt;
  opt.beam_width = 16;
  opt.iters = 3;
  opt.seed = 42;
  opt.debug = true;
  opt.scout_mode = true;
  opt.scout_threshold = 0.5;  // Items must appear in 50% of top candidates
  opt.scout_top_k = 8;
  
  v2::ScoutResult result;
  if (!v2::SolveBeamScout(cfg, soa, opt, &result, &err)) {
    return fail("SolveBeamScout: " + err);
  }
  
  // Display results
  std::cout << "\n=== Scout Results ===\n";
  std::cout << "Best solution:\n";
  std::cout << "  Objective: " << result.objective << "\n";
  std::cout << "  Penalty: " << result.penalty << "\n";
  std::cout << "  Total: " << result.total << "\n";
  std::cout << "  Items selected: ";
  int selected_count = 0;
  for (size_t i = 0; i < result.best_select.size(); ++i) {
    if (result.best_select[i]) selected_count++;
  }
  std::cout << selected_count << " / " << result.best_select.size() << "\n";
  
  std::cout << "\nActive set for exact solver:\n";
  std::cout << "  Active items: " << result.active_item_count 
            << " / " << result.original_item_count;
  double reduction_pct = 100.0 * (1.0 - (double)result.active_item_count / (double)result.original_item_count);
  std::cout << " (reduced by " << std::fixed << std::setprecision(1) 
            << reduction_pct << "%)\n";
  
  std::cout << "  Solve time: " << result.solve_time_ms << " ms\n";
  
  std::cout << "\nActive item indices: [";
  for (size_t i = 0; i < result.active_items.size() && i < 20; ++i) {
    if (i > 0) std::cout << ", ";
    std::cout << result.active_items[i];
  }
  if (result.active_items.size() > 20) {
    std::cout << ", ... (" << (result.active_items.size() - 20) << " more)";
  }
  std::cout << "]\n";
  
  std::cout << "\n✅ PASS: Scout mode completed successfully\n";
  std::cout << "\nNext steps:\n";
  std::cout << "1. Pass active_items to your exact solver (Gurobi, SCIP, etc.)\n";
  std::cout << "2. Use the beam solution as a warm start / initial feasible solution\n";
  std::cout << "3. Solve the reduced problem to prove optimality\n";
  
  return 0;
}
