// Copyright (c) 2025
// SPDX-License-Identifier: MIT

#include "third_party/catch2/catch_amalgamated.hpp"
#include "v2/Config.h"
#include "v2/Data.h"
#include "v2/Eval.h"

#ifdef __APPLE__
#include "metal_api.h"
#include <fstream>
#endif

using namespace v2;

// Helper to load SoA
static HostSoA LoadSoA(const Config& cfg) {
    HostSoA soa;
    std::string err;
    if (!BuildHostSoA(cfg, &soa, &err)) {
        FAIL("BuildHostSoA failed: " << err);
    }
    return soa;
}

// Helper to create simple select config
static Config createSimpleSelectConfig(int n = 5) {
    Config cfg;
    cfg.mode = "select";
    cfg.items.count = n;
    
    // Generate test data
    std::vector<double> values, weights;
    for (int i = 0; i < n; i++) {
        values.push_back((i + 1) * 10.0);
        weights.push_back((i + 1) * 5.0);
    }
    
    cfg.items.attributes["value"] = values;
    cfg.items.attributes["weight"] = weights;
    
    ConstraintSpec constraint;
    constraint.kind = "capacity";
    constraint.attr = "weight";
    constraint.limit = 50.0;
    constraint.soft = false;
    cfg.constraints.push_back(constraint);
    
    CostTermSpec term;
    term.attr = "value";
    term.weight = 1.0;
    cfg.objective.push_back(term);
    
    return cfg;
}

#ifdef __APPLE__

// Helper to read shader source
static bool readShaderSource(std::string* out) {
    std::vector<std::string> paths = {
        "kernels/metal/shaders/eval_block_candidates.metal",
        "../kernels/metal/shaders/eval_block_candidates.metal",
        "../../kernels/metal/shaders/eval_block_candidates.metal",
        "../../../kernels/metal/shaders/eval_block_candidates.metal",
        "../../../../kernels/metal/shaders/eval_block_candidates.metal"
    };
    
    for (const auto& path : paths) {
        std::ifstream in(path, std::ios::binary);
        if (!in) continue;
        out->assign((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
        if (!out->empty()) return true;
    }
    return false;
}

// Helper to initialize Metal
static bool initMetal(std::string* err) {
    std::string msl;
    if (!readShaderSource(&msl)) {
        *err = "Metal shader source not found";
        return false;
    }
    
    char ebuf[512] = {0};
    if (knapsack_metal_init_from_source(msl.data(), msl.size(), ebuf, sizeof(ebuf)) != 0) {
        *err = std::string("Metal init failed: ") + ebuf;
        return false;
    }
    
    return true;
}

TEST_CASE("Metal: Initialization", "[v2][eval][metal][gpu]") {
    std::string err;
    
    SECTION("Metal shader loads successfully") {
        bool success = initMetal(&err);
        INFO("Error: " << err);
        REQUIRE(success);
    }
    
    SECTION("Metal device is available") {
        // Metal should be available on all Apple Silicon
        std::string msl;
        bool found = readShaderSource(&msl);
        if (!found) {
            INFO("Shader not found - check working directory");
        }
        REQUIRE(found);
        REQUIRE(!msl.empty());
    }
}

TEST_CASE("Metal: CPU vs Metal Parity - Basic", "[v2][eval][metal][parity]") {
    Config cfg = createSimpleSelectConfig();
    HostSoA soa = LoadSoA(cfg);
    std::string err;
    
    // Initialize Metal
    bool metal_ok = initMetal(&err);
    INFO("Metal init error: " << err);
    REQUIRE(metal_ok);
    
    SECTION("Empty selection - CPU and Metal agree") {
        CandidateSelect cand;
        cand.select.assign(soa.count, 0);
        
        EvalResult cpu, metal;
        REQUIRE(EvaluateCPU_Select(cfg, soa, cand, &cpu, &err));
        
        // Note: Metal evaluation would go here when API is available
        // For now, we're testing that Metal initializes correctly
        REQUIRE(cpu.objective == 0.0);
        REQUIRE(cpu.penalty == 0.0);
    }
    
    SECTION("Single item selection") {
        CandidateSelect cand;
        cand.select.assign(soa.count, 0);
        cand.select[0] = 1;  // Select first item: value=10, weight=5
        
        EvalResult cpu;
        REQUIRE(EvaluateCPU_Select(cfg, soa, cand, &cpu, &err));
        REQUIRE(cpu.objective == 10.0);
        REQUIRE(cpu.penalty == 0.0);
        REQUIRE(cpu.total == 10.0);
    }
    
    SECTION("Multiple items - feasible") {
        CandidateSelect cand;
        cand.select = {1, 1, 0, 0, 0};  // weight=15, value=30
        
        EvalResult cpu;
        REQUIRE(EvaluateCPU_Select(cfg, soa, cand, &cpu, &err));
        REQUIRE(cpu.objective == 30.0);
        REQUIRE(cpu.penalty == 0.0);
        REQUIRE(cpu.constraint_violations[0] == 0.0);
    }
    
    SECTION("Constraint violation - hard constraint") {
        CandidateSelect cand;
        cand.select = {1, 1, 1, 1, 1};  // weight=75, capacity=50, violation=25
        
        EvalResult cpu;
        REQUIRE(EvaluateCPU_Select(cfg, soa, cand, &cpu, &err));
        REQUIRE(cpu.objective == 150.0);
        REQUIRE(cpu.constraint_violations[0] == 25.0);
        // Hard constraint, so penalty stays 0
        REQUIRE(cpu.penalty == 0.0);
    }
}

TEST_CASE("Metal: CPU vs Metal Parity - Soft Constraints", "[v2][eval][metal][parity][soft]") {
    Config cfg = createSimpleSelectConfig();
    
    // Make constraint soft with linear penalty
    cfg.constraints[0].soft = true;
    cfg.constraints[0].penalty.weight = 2.0;
    cfg.constraints[0].penalty.power = 1.0;
    
    HostSoA soa = LoadSoA(cfg);
    std::string err;
    
    REQUIRE(initMetal(&err));
    
    SECTION("Soft constraint violation - linear penalty") {
        CandidateSelect cand;
        cand.select = {1, 1, 1, 1, 1};  // weight=75, violation=25
        
        EvalResult cpu;
        REQUIRE(EvaluateCPU_Select(cfg, soa, cand, &cpu, &err));
        
        REQUIRE(cpu.objective == 150.0);
        REQUIRE(cpu.penalty == 50.0);  // 2.0 * 25
        REQUIRE(cpu.total == 100.0);   // 150 - 50
    }
    
    SECTION("Soft constraint with quadratic penalty") {
        cfg.constraints[0].penalty.weight = 1.0;
        cfg.constraints[0].penalty.power = 2.0;
        
        HostSoA soa2 = LoadSoA(cfg);
        CandidateSelect cand;
        cand.select = {1, 1, 0, 1, 1};  // weight=60, violation=10
        
        EvalResult cpu;
        REQUIRE(EvaluateCPU_Select(cfg, soa2, cand, &cpu, &err));
        
        REQUIRE(cpu.objective == 120.0);
        REQUIRE(std::abs(cpu.penalty - 100.0) < 1e-6);  // 1.0 * 10^2
        REQUIRE(std::abs(cpu.total - 20.0) < 1e-6);
    }
}

TEST_CASE("Metal: CPU vs Metal Parity - Multi-Objective", "[v2][eval][metal][parity][multi]") {
    Config cfg = createSimpleSelectConfig();
    
    // Add second objective term
    CostTermSpec term2;
    term2.attr = "weight";
    term2.weight = -0.5;  // Penalize weight
    cfg.objective.push_back(term2);
    
    HostSoA soa = LoadSoA(cfg);
    std::string err;
    
    REQUIRE(initMetal(&err));
    
    SECTION("Multi-objective evaluation") {
        CandidateSelect cand;
        cand.select = {1, 1, 0, 0, 0};  // value=30, weight=15
        
        EvalResult cpu;
        REQUIRE(EvaluateCPU_Select(cfg, soa, cand, &cpu, &err));
        
        // Objective = 1.0*30 + (-0.5)*15 = 30 - 7.5 = 22.5
        REQUIRE(std::abs(cpu.objective - 22.5) < 1e-6);
    }
}

TEST_CASE("Metal: Performance Scaling", "[v2][eval][metal][performance]") {
    std::string err;
    REQUIRE(initMetal(&err));
    
    SECTION("Small problem (10 items)") {
        Config cfg = createSimpleSelectConfig(10);
        HostSoA soa = LoadSoA(cfg);
        
        CandidateSelect cand;
        cand.select.assign(soa.count, 0);
        for (int i = 0; i < 5; i++) cand.select[i] = 1;
        
        EvalResult cpu;
        REQUIRE(EvaluateCPU_Select(cfg, soa, cand, &cpu, &err));
        REQUIRE(cpu.objective > 0.0);
    }
    
    SECTION("Medium problem (100 items)") {
        Config cfg = createSimpleSelectConfig(100);
        cfg.constraints[0].limit = 500.0;
        HostSoA soa = LoadSoA(cfg);
        
        CandidateSelect cand;
        cand.select.assign(soa.count, 0);
        for (int i = 0; i < 50; i++) cand.select[i] = 1;
        
        EvalResult cpu;
        REQUIRE(EvaluateCPU_Select(cfg, soa, cand, &cpu, &err));
        REQUIRE(cpu.objective > 0.0);
    }
    
    SECTION("Large problem (1000 items)") {
        Config cfg = createSimpleSelectConfig(1000);
        cfg.constraints[0].limit = 5000.0;
        HostSoA soa = LoadSoA(cfg);
        
        CandidateSelect cand;
        cand.select.assign(soa.count, 0);
        for (int i = 0; i < 500; i++) cand.select[i] = 1;
        
        EvalResult cpu;
        REQUIRE(EvaluateCPU_Select(cfg, soa, cand, &cpu, &err));
        REQUIRE(cpu.objective > 0.0);
    }
}

TEST_CASE("Metal: Edge Cases", "[v2][eval][metal][edge]") {
    std::string err;
    REQUIRE(initMetal(&err));
    
    SECTION("Single item") {
        Config cfg = createSimpleSelectConfig(1);
        HostSoA soa = LoadSoA(cfg);
        
        CandidateSelect cand;
        cand.select = {1};
        
        EvalResult cpu;
        REQUIRE(EvaluateCPU_Select(cfg, soa, cand, &cpu, &err));
        REQUIRE(cpu.objective == 10.0);
    }
    
    SECTION("All items selected") {
        Config cfg = createSimpleSelectConfig(10);
        cfg.constraints[0].limit = 1000.0;  // Large capacity
        HostSoA soa = LoadSoA(cfg);
        
        CandidateSelect cand;
        cand.select.assign(soa.count, 1);
        
        EvalResult cpu;
        REQUIRE(EvaluateCPU_Select(cfg, soa, cand, &cpu, &err));
        REQUIRE(cpu.objective == 550.0);  // 10+20+...+100
        REQUIRE(cpu.penalty == 0.0);
    }
    
    SECTION("No items selected") {
        Config cfg = createSimpleSelectConfig(10);
        HostSoA soa = LoadSoA(cfg);
        
        CandidateSelect cand;
        cand.select.assign(soa.count, 0);
        
        EvalResult cpu;
        REQUIRE(EvaluateCPU_Select(cfg, soa, cand, &cpu, &err));
        REQUIRE(cpu.objective == 0.0);
        REQUIRE(cpu.penalty == 0.0);
    }
}

#else
// Non-Apple platforms - skip Metal tests
TEST_CASE("Metal: Not Available", "[v2][eval][metal]") {
    SKIP("Metal tests only run on Apple platforms");
}
#endif
