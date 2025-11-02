#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include "knapsack_c.h"

static void usage(const char* prog) {
  std::cerr << "Usage: " << prog << " <config.json> [options.json]" << std::endl;
}

static bool read_file(const std::string& path, std::string* out) {
  std::ifstream in(path);
  if (!in) return false;
  std::ostringstream ss; ss << in.rdbuf();
  *out = ss.str();
  return true;
}

int main(int argc, char** argv) {
  if (argc < 2) { usage(argv[0]); return 2; }
  std::string cfg, opt;
  if (!read_file(argv[1], &cfg)) { std::cerr << "Failed to read config: " << argv[1] << std::endl; return 3; }
  if (argc >= 3) { if (!read_file(argv[2], &opt)) { std::cerr << "Failed to read options: " << argv[2] << std::endl; return 3; } }

  KnapsackSolutionV2* sol = nullptr;
  int rc = solve_knapsack_v2_from_json(cfg.c_str(), opt.empty() ? nullptr : opt.c_str(), &sol);
  if (rc != 0) { std::cerr << "solve_knapsack_v2_from_json error: " << rc << std::endl; return 4; }

  int selected = 0; for (int i = 0; i < sol->num_items; ++i) if (sol->select[i]) ++selected;
  std::cout << "objective=" << sol->objective << " penalty=" << sol->penalty << " total=" << sol->total << "\n";
  std::cout << "selected_items=" << selected << "/" << sol->num_items << "\n";
  // Print indices (first few)
  std::vector<int> idx; idx.reserve(selected);
  for (int i = 0; i < sol->num_items; ++i) if (sol->select[i]) idx.push_back(i);
  std::cout << "indices:";
  int max_show = 32;
  for (size_t i = 0; i < idx.size() && (int)i < max_show; ++i) std::cout << (i==0?" ":", ") << idx[i];
  if ((int)idx.size() > max_show) std::cout << ", ...";
  std::cout << "\n";

  free_knapsack_solution_v2(sol);
  return 0;
}
