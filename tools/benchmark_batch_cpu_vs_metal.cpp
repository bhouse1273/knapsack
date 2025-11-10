// Batch benchmark tool: CPU sequential vs Metal parallel
// This demonstrates where GPU acceleration truly shines!

#include <iostream>
#include <chrono>
#include <vector>
#include <random>
#include <iomanip>

#include "v2/Config.h"
#include "v2/Data.h"
#include "v2/Eval.h"

#ifdef __APPLE__
#include "metal_api.h"
#include <fstream>
#endif

using namespace v2;
using namespace std::chrono;

bool read_file(const std::vector<std::string>& paths, std::string* out) {
    for (const auto& p : paths) {
        std::ifstream in(p, std::ios::binary);
        if (!in) continue;
        out->assign((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
        if (!out->empty()) return true;
    }
    return false;
}

// Generate random candidates for benchmarking
std::vector<CandidateSelect> generate_random_candidates(int num_candidates, int num_items, unsigned seed = 42) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<> dist(0.0, 1.0);
    
    std::vector<CandidateSelect> candidates;
    candidates.reserve(num_candidates);
    
    for (int c = 0; c < num_candidates; c++) {
        CandidateSelect cand;
        cand.select.reserve(num_items);
        
        // Random selection with ~50% density
        for (int i = 0; i < num_items; i++) {
            cand.select.push_back(dist(rng) < 0.5 ? 1 : 0);
        }
        
        candidates.push_back(cand);
    }
    
    return candidates;
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <config.json> <num_candidates>\n";
        std::cerr << "Example: " << argv[0] << " config.json 1000\n";
        return 1;
    }
    
    std::string config_file = argv[1];
    int num_candidates = std::atoi(argv[2]);
    
    if (num_candidates < 1 || num_candidates > 100000) {
        std::cerr << "num_candidates must be between 1 and 100000\n";
        return 1;
    }
    
    // Load config
    Config cfg;
    std::string err;
    if (!LoadConfigFromFile(config_file, &cfg, &err)) {
        std::cerr << "Failed to load config: " << err << "\n";
        return 1;
    }
    
    HostSoA soa;
    if (!BuildHostSoA(cfg, &soa, &err)) {
        std::cerr << "Failed to build SoA: " << err << "\n";
        return 1;
    }
    
    std::cout << "Generating " << num_candidates << " random candidates for " 
              << soa.count << " items...\n";
    
    auto candidates = generate_random_candidates(num_candidates, soa.count);
    
    // Benchmark CPU (sequential evaluation)
    std::cout << "Benchmarking CPU (sequential)...\n";
    auto cpu_start = high_resolution_clock::now();
    
    std::vector<EvalResult> cpu_results;
    cpu_results.reserve(num_candidates);
    
    for (const auto& cand : candidates) {
        EvalResult result;
        if (!EvaluateCPU_Select(cfg, soa, cand, &result, &err)) {
            std::cerr << "CPU evaluation failed: " << err << "\n";
            return 1;
        }
        cpu_results.push_back(result);
    }
    
    auto cpu_end = high_resolution_clock::now();
    double cpu_time_ms = duration_cast<microseconds>(cpu_end - cpu_start).count() / 1000.0;
    
#ifdef __APPLE__
    // Initialize Metal
    std::cout << "Initializing Metal GPU...\n";
    std::string shader;
    if (!read_file({
        "kernels/metal/shaders/eval_block_candidates.metal",
        "../kernels/metal/shaders/eval_block_candidates.metal",
        "../../kernels/metal/shaders/eval_block_candidates.metal"
    }, &shader)) {
        std::cerr << "Error: Metal shader not found\n";
        return 1;
    }
    
    char metal_err[512] = {0};
    if (knapsack_metal_init_from_source(shader.data(), shader.size(),
                                       metal_err, sizeof(metal_err)) != 0) {
        std::cerr << "Metal init failed: " << metal_err << "\n";
        return 1;
    }
    
    // Benchmark Metal (parallel evaluation)
    std::cout << "Benchmarking Metal GPU (parallel)...\n";
    auto metal_start = high_resolution_clock::now();
    
    std::vector<EvalResult> metal_results;
    if (!EvaluateMetal_Batch(cfg, soa, candidates, &metal_results, &err)) {
        std::cerr << "Metal batch evaluation failed: " << err << "\n";
        return 1;
    }
    
    auto metal_end = high_resolution_clock::now();
    double metal_time_ms = duration_cast<microseconds>(metal_end - metal_start).count() / 1000.0;
    
    // Validate correctness
    std::cout << "Validating correctness...\n";
    bool all_match = true;
    double max_diff = 0.0;
    
    for (size_t i = 0; i < cpu_results.size(); i++) {
        double diff = std::abs(cpu_results[i].objective - metal_results[i].objective);
        max_diff = std::max(max_diff, diff);
        if (diff > 1e-3) {
            std::cerr << "Mismatch at candidate " << i << ": CPU=" 
                     << cpu_results[i].objective << " Metal=" 
                     << metal_results[i].objective << "\n";
            all_match = false;
        }
    }
    
    // Calculate statistics
    double cpu_throughput = (num_candidates / cpu_time_ms) * 1000.0;  // candidates/sec
    double metal_throughput = (num_candidates / metal_time_ms) * 1000.0;
    double speedup = cpu_time_ms / metal_time_ms;
    
    // Print results
    std::cout << "\n";
    std::cout << "╔════════════════════════════════════════════════════════════╗\n";
    std::cout << "║        Batch Evaluation: CPU vs Metal GPU                  ║\n";
    std::cout << "╚════════════════════════════════════════════════════════════╝\n";
    std::cout << "\n";
    std::cout << "Problem Size:      " << soa.count << " items\n";
    std::cout << "Batch Size:        " << num_candidates << " candidates\n";
    std::cout << "\n";
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "CPU Time:          " << cpu_time_ms << " ms\n";
    std::cout << "Metal Time:        " << metal_time_ms << " ms\n";
    std::cout << "Speedup:           " << speedup << "x\n";
    std::cout << "\n";
    std::cout << std::fixed << std::setprecision(0);
    std::cout << "CPU Throughput:    " << cpu_throughput << " candidates/sec\n";
    std::cout << "Metal Throughput:  " << metal_throughput << " candidates/sec\n";
    std::cout << "\n";
    std::cout << "Correctness:       " << (all_match ? "✅ PASS" : "❌ FAIL") << "\n";
    std::cout << "Max Difference:    " << std::scientific << std::setprecision(2) << max_diff << "\n";
    std::cout << "\n";
    
    // CSV output for easy parsing
    std::cout << "# CSV: items,candidates,cpu_ms,metal_ms,speedup,cpu_throughput,metal_throughput\n";
    std::cout << soa.count << "," << num_candidates << ","
              << std::fixed << std::setprecision(3)
              << cpu_time_ms << "," << metal_time_ms << "," << speedup << ","
              << std::setprecision(0)
              << cpu_throughput << "," << metal_throughput << "\n";
    
    return all_match ? 0 : 1;
#else
    std::cout << "\nMetal GPU not available on this platform.\n";
    std::cout << "CPU processed " << num_candidates << " candidates in " 
              << cpu_time_ms << " ms\n";
    return 0;
#endif
}
