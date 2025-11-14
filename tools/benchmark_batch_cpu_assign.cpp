// Batch benchmark tool: CPU sequential for assign mode (multi-knapsack)
// Generates random assignment candidates and evaluates throughput.

#include <iostream>
#include <chrono>
#include <vector>
#include <random>
#include <iomanip>

#include "v2/Config.h"
#include "v2/Data.h"
#include "v2/Eval.h"

using namespace v2;
using namespace std::chrono;

// Generate random assign-mode candidates.
// -1 means unassigned, otherwise in [0, K-1]. Density roughly 50% assigned.
static std::vector<CandidateAssign> generate_random_assign_candidates(
    int num_candidates, int num_items, int K, unsigned seed = 42) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<> u01(0.0, 1.0);
  std::uniform_int_distribution<int> kdist(0, std::max(0, K - 1));

  std::vector<CandidateAssign> candidates;
  candidates.reserve(num_candidates);
  for (int c = 0; c < num_candidates; ++c) {
    CandidateAssign cand;
    cand.assign.reserve(num_items);
    for (int i = 0; i < num_items; ++i) {
      if (K <= 0) { cand.assign.push_back(-1); continue; }
      bool take = (u01(rng) < 0.5);
      cand.assign.push_back(take ? kdist(rng) : -1);
    }
    candidates.push_back(std::move(cand));
  }
  return candidates;
}

int main(int argc, char** argv) {
  if (argc < 3) {
    std::cerr << "Usage: " << argv[0] << " <assign_config.json> <num_candidates>\n";
    std::cerr << "Example: " << argv[0] << " data/benchmarks/small_mkp_v2_assign.json 10000\n";
    return 1;
  }

  std::string config_file = argv[1];
  int num_candidates = std::atoi(argv[2]);
  if (num_candidates < 1 || num_candidates > 200000) {
    std::cerr << "num_candidates must be between 1 and 200000\n";
    return 1;
  }

  // Load config
  Config cfg; std::string err;
  if (!LoadConfigFromFile(config_file, &cfg, &err)) {
    std::cerr << "Failed to load config: " << err << "\n"; return 1;
  }
  if (cfg.mode != std::string("assign")) {
    std::cerr << "Config mode is not 'assign'\n"; return 1;
  }

  HostSoA soa;
  if (!BuildHostSoA(cfg, &soa, &err)) { std::cerr << "SoA build failed: " << err << "\n"; return 1; }

  const int K = cfg.knapsack.K;
  std::cout << "Generating " << num_candidates << " random assign candidates for "
            << soa.count << " items across K=" << K << " knapsacks...\n";
  auto candidates = generate_random_assign_candidates(num_candidates, soa.count, K);

  // Benchmark CPU
  std::cout << "Benchmarking CPU (sequential, assign)...\n";
  auto cpu_start = high_resolution_clock::now();

  std::vector<EvalResult> cpu_results; cpu_results.reserve(num_candidates);
  for (const auto& cand : candidates) {
    EvalResult r;
    if (!EvaluateCPU_Assign(cfg, soa, cand, &r, &err)) {
      std::cerr << "CPU assign evaluation failed: " << err << "\n"; return 1;
    }
    cpu_results.push_back(std::move(r));
  }

  auto cpu_end = high_resolution_clock::now();
  double cpu_time_ms = duration_cast<microseconds>(cpu_end - cpu_start).count() / 1000.0;

  // Stats
  double cpu_throughput = (num_candidates / cpu_time_ms) * 1000.0; // candidates/sec

  // Pretty print
  std::cout << "\n";
  std::cout << "╔════════════════════════════════════════════════════════════╗\n";
  std::cout << "║      Batch Evaluation (ASSIGN): CPU throughput             ║\n";
  std::cout << "╚════════════════════════════════════════════════════════════╝\n";
  std::cout << "\n";
  std::cout << "Problem Size:      " << soa.count << " items, K=" << K << "\n";
  std::cout << "Batch Size:        " << num_candidates << " candidates\n";
  std::cout << "\n";
  std::cout << std::fixed << std::setprecision(2);
  std::cout << "CPU Time:          " << cpu_time_ms << " ms\n";
  std::cout << std::fixed << std::setprecision(0);
  std::cout << "CPU Throughput:    " << cpu_throughput << " candidates/sec\n";
  std::cout << "\n";
  std::cout << "Note: Metal assign batch evaluation isn't implemented yet.\n";
  std::cout << "      This tool measures CPU-only for assign mode.\n";
  std::cout << "\n";

  // CSV header and line for easy parsing
  std::cout << "# CSV: items,K,candidates,cpu_ms,cpu_throughput\n";
  std::cout << soa.count << "," << K << "," << num_candidates << ","
            << std::fixed << std::setprecision(3) << cpu_time_ms << ","
            << std::setprecision(0) << cpu_throughput << "\n";

  return 0;
}
