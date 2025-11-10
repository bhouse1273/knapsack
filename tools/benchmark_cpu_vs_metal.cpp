// Benchmark tool: CPU vs Metal performance comparison
// Measures evaluation performance on various problem sizes

#include <iostream>
#include <chrono>
#include <fstream>
#include <vector>
#include <numeric>
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

struct BenchmarkResult {
    std::string name;
    int items;
    int iterations;
    double cpu_time_ms;
    double metal_time_ms;
    double speedup;
    double cpu_objective;
    double metal_objective;
};

bool read_file(const std::vector<std::string>& paths, std::string* out) {
    for (const auto& p : paths) {
        std::ifstream in(p, std::ios::binary);
        if (!in) continue;
        out->assign((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
        if (!out->empty()) return true;
    }
    return false;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <config.json> [iterations]\n";
        return 1;
    }
    
    std::string config_file = argv[1];
    int iterations = argc > 2 ? std::atoi(argv[2]) : 100;
    
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
    
    // Create a candidate to evaluate (select half the items)
    CandidateSelect cand;
    cand.select.assign(soa.count, 0);
    for (int i = 0; i < soa.count / 2; i++) {
        cand.select[i] = 1;
    }
    
    // Warm up
    EvalResult warmup;
    for (int i = 0; i < 10; i++) {
        if (!EvaluateCPU_Select(cfg, soa, cand, &warmup, &err)) {
            std::cerr << "CPU warmup failed: " << err << "\n";
            return 1;
        }
    }
    
    // Benchmark CPU
    auto cpu_start = high_resolution_clock::now();
    EvalResult cpu_result;
    for (int i = 0; i < iterations; i++) {
        if (!EvaluateCPU_Select(cfg, soa, cand, &cpu_result, &err)) {
            std::cerr << "CPU evaluation failed: " << err << "\n";
            return 1;
        }
    }
    auto cpu_end = high_resolution_clock::now();
    double cpu_time = duration_cast<microseconds>(cpu_end - cpu_start).count() / 1000.0;
    
#ifdef __APPLE__
    // Initialize Metal
    std::string shader;
    if (!read_file({
        "kernels/metal/shaders/eval_block_candidates.metal",
        "../kernels/metal/shaders/eval_block_candidates.metal",
        "../../kernels/metal/shaders/eval_block_candidates.metal"
    }, &shader)) {
        std::cerr << "Warning: Metal shader not found, skipping GPU benchmark\n";
        
        // Output CPU-only results
        std::cout << soa.count << "," << iterations << ","
                  << std::fixed << std::setprecision(3)
                  << cpu_time << ",0.0,0.0,"
                  << cpu_result.objective << ",0.0\n";
        return 0;
    }
    
    char metal_err[512] = {0};
    if (knapsack_metal_init_from_source(shader.data(), shader.size(),
                                       metal_err, sizeof(metal_err)) != 0) {
        std::cerr << "Warning: Metal init failed: " << metal_err << "\n";
        std::cerr << "Skipping Metal benchmark\n";
        
        // Output CPU-only results
        std::cout << soa.count << "," << iterations << ","
                  << std::fixed << std::setprecision(3)
                  << cpu_time << ",0.0,0.0,"
                  << cpu_result.objective << ",0.0\n";
        return 0;
    }
    
    // Benchmark Metal
    auto metal_start = high_resolution_clock::now();
    EvalResult metal_result;
    
    for (int i = 0; i < iterations; i++) {
        if (!EvaluateMetal_Select(cfg, soa, cand, &metal_result, &err)) {
            std::cerr << "Metal evaluation failed: " << err << "\n";
            return 1;
        }
    }
    
    auto metal_end = high_resolution_clock::now();
    double metal_time = duration_cast<microseconds>(metal_end - metal_start).count() / 1000.0;
    
    double speedup = metal_time > 0 ? cpu_time / metal_time : 0.0;
#else
    double metal_time = 0.0;
    double speedup = 0.0;
    EvalResult metal_result = cpu_result;
#endif
    
    // Output: items,iterations,cpu_ms,metal_ms,speedup,cpu_obj,metal_obj
    std::cout << soa.count << "," << iterations << ","
              << std::fixed << std::setprecision(3)
              << cpu_time << "," << metal_time << "," << speedup << ","
              << std::setprecision(2)
              << cpu_result.objective << "," << metal_result.objective << "\n";
    
    return 0;
}
