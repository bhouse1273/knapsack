// test_eval_cpu.cpp - Comprehensive tests for v2 CPU evaluation logic
#include "third_party/catch2/catch_amalgamated.hpp"
#include "v2/Eval.h"
#include "v2/Data.h"
#include "v2/Config.h"
#include <cmath>

using namespace v2;

// Helper to build HostSoA from Config (wraps BuildHostSoA with error checking)
HostSoA LoadSoA(const Config& cfg) {
    HostSoA soa;
    std::string err;
    if (!BuildHostSoA(cfg, &soa, &err)) {
        FAIL("BuildHostSoA failed: " << err);
    }
    return soa;
}

// Helper to create a simple select mode config
Config createSimpleEvalConfig() {
    Config cfg;
    cfg.mode = "select";
    cfg.items.count = 5;
    cfg.items.attributes["value"] = {10.0, 20.0, 30.0, 40.0, 50.0};
    cfg.items.attributes["weight"] = {5.0, 10.0, 15.0, 20.0, 25.0};
    
    // Capacity constraint
    ConstraintSpec constraint;
    constraint.kind = "capacity";
    constraint.attr = "weight";
    constraint.limit = 50.0;
    constraint.soft = false;
    cfg.constraints.push_back(constraint);
    
    // Objective: maximize value
    CostTermSpec term;
    term.attr = "value";
    term.weight = 1.0;
    cfg.objective.push_back(term);
    
    return cfg;
}

// Helper to create assign mode config
Config createSimpleAssignConfig() {
    Config cfg;
    cfg.mode = "assign";
    cfg.items.count = 6;
    cfg.items.attributes["value"] = {10.0, 20.0, 30.0, 40.0, 50.0, 60.0};
    cfg.items.attributes["weight"] = {5.0, 10.0, 15.0, 20.0, 25.0, 30.0};
    
    cfg.knapsack.K = 2;
    cfg.knapsack.capacities = {50.0, 75.0};
    cfg.knapsack.capacity_attr = "weight";
    
    // Objective: maximize value
    CostTermSpec term;
    term.attr = "value";
    term.weight = 1.0;
    cfg.objective.push_back(term);
    
    return cfg;
}

TEST_CASE("EvaluateCPU_Select: Basic Functionality", "[v2][eval][cpu][select]") {
    Config cfg = createSimpleEvalConfig();
    HostSoA soa = LoadSoA(cfg);
    std::string err;

    SECTION("Empty selection evaluates correctly") {
        CandidateSelect cand;
        cand.select = {0, 0, 0, 0, 0};
        
        EvalResult result;
        bool success = EvaluateCPU_Select(cfg, soa, cand, &result, &err);
        
        REQUIRE(success);
        REQUIRE(err.empty());
        REQUIRE(result.objective == 0.0);
        REQUIRE(result.penalty == 0.0);
        REQUIRE(result.total == 0.0);
    }

    SECTION("Single item selection") {
        CandidateSelect cand;
        cand.select = {1, 0, 0, 0, 0};  // Select first item
        
        EvalResult result;
        bool success = EvaluateCPU_Select(cfg, soa, cand, &result, &err);
        
        REQUIRE(success);
        REQUIRE(result.objective == 10.0);  // value[0] = 10
        REQUIRE(result.total == 10.0);
    }

    SECTION("Multiple items selection") {
        CandidateSelect cand;
        cand.select = {1, 1, 0, 0, 0};  // Select first two items
        
        EvalResult result;
        bool success = EvaluateCPU_Select(cfg, soa, cand, &result, &err);
        
        REQUIRE(success);
        REQUIRE(result.objective == 30.0);  // 10 + 20
        REQUIRE(result.total == 30.0);
    }

    SECTION("All items selection") {
        CandidateSelect cand;
        cand.select = {1, 1, 1, 1, 1};
        
        EvalResult result;
        bool success = EvaluateCPU_Select(cfg, soa, cand, &result, &err);
        
        REQUIRE(success);
        REQUIRE(result.objective == 150.0);  // 10+20+30+40+50
    }
}

TEST_CASE("EvaluateCPU_Select: Constraint Violations", "[v2][eval][cpu][constraints]") {
    Config cfg = createSimpleEvalConfig();
    HostSoA soa = LoadSoA(cfg);
    std::string err;

    SECTION("Feasible solution - no violation") {
        CandidateSelect cand;
        cand.select = {1, 1, 0, 0, 0};  // weight = 5+10 = 15, capacity = 50
        
        EvalResult result;
        bool success = EvaluateCPU_Select(cfg, soa, cand, &result, &err);
        
        REQUIRE(success);
        REQUIRE(result.constraint_violations.size() >= 1);
        REQUIRE(result.constraint_violations[0] == 0.0);  // No violation
        REQUIRE(result.penalty == 0.0);
    }

    SECTION("Infeasible solution - exceeds capacity") {
        CandidateSelect cand;
        cand.select = {1, 1, 1, 1, 1};  // weight = 75, capacity = 50
        
        EvalResult result;
        bool success = EvaluateCPU_Select(cfg, soa, cand, &result, &err);
        
        REQUIRE(success);
        REQUIRE(result.constraint_violations[0] > 0.0);  // Violation = 25
        REQUIRE(std::abs(result.constraint_violations[0] - 25.0) < 1e-6);
    }

    SECTION("Exactly at capacity - no violation") {
        CandidateSelect cand;
        cand.select = {1, 1, 1, 0, 0};  // weight = 5+10+15 = 30, capacity = 50
        
        EvalResult result;
        bool success = EvaluateCPU_Select(cfg, soa, cand, &result, &err);
        
        REQUIRE(success);
        REQUIRE(result.constraint_violations[0] == 0.0);
        REQUIRE(result.penalty == 0.0);
    }
}

TEST_CASE("EvaluateCPU_Select: Soft Constraints", "[v2][eval][cpu][soft]") {
    Config cfg = createSimpleEvalConfig();
    
    // Make constraint soft with penalty
    cfg.constraints[0].soft = true;
    cfg.constraints[0].penalty.weight = 2.0;
    cfg.constraints[0].penalty.power = 1.0;  // Linear penalty
    
    HostSoA soa = LoadSoA(cfg);
    std::string err;

    SECTION("Soft constraint violation incurs penalty") {
        CandidateSelect cand;
        cand.select = {1, 1, 1, 1, 1};  // weight = 75, capacity = 50, violation = 25
        
        EvalResult result;
        bool success = EvaluateCPU_Select(cfg, soa, cand, &result, &err);
        
        REQUIRE(success);
        REQUIRE(result.objective == 150.0);  // Sum of values
        REQUIRE(result.penalty > 0.0);       // Penalty for violation
        REQUIRE(result.total < result.objective);  // Total = obj - penalty
        
        // Linear penalty: weight * violation = 2.0 * 25 = 50
        REQUIRE(std::abs(result.penalty - 50.0) < 1e-6);
    }

    SECTION("Soft constraint with quadratic penalty") {
        cfg.constraints[0].penalty.weight = 1.0;
        cfg.constraints[0].penalty.power = 2.0;  // Quadratic
        
        HostSoA soa2 = LoadSoA(cfg);
        CandidateSelect cand;
        cand.select = {1, 1, 0, 1, 1};  // weight = 5+10+20+25 = 60, violation = 10
        
        EvalResult result;
        bool success = EvaluateCPU_Select(cfg, soa2, cand, &result, &err);
        
        REQUIRE(success);
        REQUIRE(result.objective == 120.0);  // value = 10+20+40+50
        REQUIRE(result.constraint_violations[0] == 10.0);  // violation = 60-50
        // Quadratic penalty: weight * violation^power = 1.0 * 10^2 = 100
        REQUIRE(std::abs(result.penalty - 100.0) < 1e-6);
        REQUIRE(std::abs(result.total - 20.0) < 1e-6);  // total = 120 - 100
    }
}

TEST_CASE("EvaluateCPU_Select: Multi-Objective", "[v2][eval][cpu][objective]") {
    Config cfg = createSimpleEvalConfig();
    
    // Add second objective term
    CostTermSpec term2;
    term2.attr = "weight";
    term2.weight = -0.5;  // Penalize weight
    cfg.objective.push_back(term2);
    
    HostSoA soa = LoadSoA(cfg);
    std::string err;

    SECTION("Multi-objective calculation") {
        CandidateSelect cand;
        cand.select = {1, 1, 0, 0, 0};  // value = 30, weight = 15
        
        EvalResult result;
        bool success = EvaluateCPU_Select(cfg, soa, cand, &result, &err);
        
        REQUIRE(success);
        // Objective = 1.0 * 30 + (-0.5) * 15 = 30 - 7.5 = 22.5
        REQUIRE(std::abs(result.objective - 22.5) < 1e-6);
    }
}

TEST_CASE("EvaluateCPU_Assign: Basic Functionality", "[v2][eval][cpu][assign]") {
    Config cfg = createSimpleAssignConfig();
    HostSoA soa = LoadSoA(cfg);
    std::string err;

    SECTION("Empty assignment evaluates correctly") {
        CandidateAssign cand;
        cand.assign = {-1, -1, -1, -1, -1, -1};  // All unassigned
        
        EvalResult result;
        bool success = EvaluateCPU_Assign(cfg, soa, cand, &result, &err);
        
        REQUIRE(success);
        REQUIRE(result.objective == 0.0);
        REQUIRE(result.penalty == 0.0);
    }

    SECTION("Single item to first knapsack") {
        CandidateAssign cand;
        cand.assign = {0, -1, -1, -1, -1, -1};  // First item to knapsack 0
        
        EvalResult result;
        bool success = EvaluateCPU_Assign(cfg, soa, cand, &result, &err);
        
        REQUIRE(success);
        REQUIRE(result.objective == 10.0);  // value[0]
    }

    SECTION("Items distributed across knapsacks") {
        CandidateAssign cand;
        cand.assign = {0, 0, 1, 1, -1, -1};  // Items 0,1 -> ks 0; items 2,3 -> ks 1
        
        EvalResult result;
        bool success = EvaluateCPU_Assign(cfg, soa, cand, &result, &err);
        
        REQUIRE(success);
        REQUIRE(result.objective == 100.0);  // 10+20+30+40
    }

    SECTION("All items assigned") {
        CandidateAssign cand;
        cand.assign = {0, 0, 1, 1, 0, 1};  // Distribute all items
        
        EvalResult result;
        bool success = EvaluateCPU_Assign(cfg, soa, cand, &result, &err);
        
        REQUIRE(success);
        REQUIRE(result.objective == 210.0);  // Sum of all values
    }
}

TEST_CASE("EvaluateCPU_Assign: Capacity Violations", "[v2][eval][cpu][assign]") {
    Config cfg = createSimpleAssignConfig();
    HostSoA soa = LoadSoA(cfg);
    std::string err;

    SECTION("Feasible assignment") {
        CandidateAssign cand;
        // Knapsack 0 (cap=50): items 0,1,2 -> weight=30
        // Knapsack 1 (cap=75): items 3,4 -> weight=45
        cand.assign = {0, 0, 0, 1, 1, -1};
        
        EvalResult result;
        bool success = EvaluateCPU_Assign(cfg, soa, cand, &result, &err);
        
        REQUIRE(success);
        REQUIRE(result.knapsack_violations.size() == 2);
        REQUIRE(result.knapsack_violations[0] == 0.0);
        REQUIRE(result.knapsack_violations[1] == 0.0);
        REQUIRE(result.penalty == 0.0);
    }

    SECTION("Infeasible - exceeds knapsack 0 capacity") {
        CandidateAssign cand;
        // Knapsack 0 (cap=50): items 0,1,2,3 -> weight=5+10+15+20=50
        // Knapsack 0 (cap=50): adding item 4 -> weight=75, violation=25
        cand.assign = {0, 0, 0, 0, 0, -1};
        
        EvalResult result;
        bool success = EvaluateCPU_Assign(cfg, soa, cand, &result, &err);
        
        REQUIRE(success);
        REQUIRE(result.knapsack_violations[0] > 0.0);
        REQUIRE(result.penalty > 0.0);
    }

    SECTION("Both knapsacks exceed capacity") {
        CandidateAssign cand;
        // Overload both knapsacks
        cand.assign = {0, 0, 0, 0, 1, 1};
        // Knapsack 0: 5+10+15+20 = 50 (at limit)
        // Knapsack 1: 25+30 = 55, capacity=75 (OK)
        
        EvalResult result;
        bool success = EvaluateCPU_Assign(cfg, soa, cand, &result, &err);
        
        REQUIRE(success);
        // Check violations
        REQUIRE(result.knapsack_violations.size() == 2);
    }
}

TEST_CASE("EvaluateCPU: Multiple Constraints", "[v2][eval][cpu][multi]") {
    Config cfg;
    cfg.mode = "select";
    cfg.items.count = 5;
    cfg.items.attributes["value"] = {10, 20, 30, 40, 50};
    cfg.items.attributes["weight"] = {5, 10, 15, 20, 25};
    cfg.items.attributes["volume"] = {3, 6, 9, 12, 15};
    
    // Two constraints
    ConstraintSpec c1, c2;
    c1.kind = "capacity";
    c1.attr = "weight";
    c1.limit = 50.0;
    c2.kind = "capacity";
    c2.attr = "volume";
    c2.limit = 30.0;
    
    cfg.constraints.push_back(c1);
    cfg.constraints.push_back(c2);
    
    CostTermSpec term;
    term.attr = "value";
    term.weight = 1.0;
    cfg.objective.push_back(term);
    
    HostSoA soa = LoadSoA(cfg);
    std::string err;

    SECTION("Both constraints satisfied") {
        CandidateSelect cand;
        cand.select = {1, 1, 0, 0, 0};  // weight=15, volume=9
        
        EvalResult result;
        bool success = EvaluateCPU_Select(cfg, soa, cand, &result, &err);
        
        REQUIRE(success);
        REQUIRE(result.constraint_violations.size() == 2);
        REQUIRE(result.constraint_violations[0] == 0.0);
        REQUIRE(result.constraint_violations[1] == 0.0);
        REQUIRE(result.penalty == 0.0);
    }

    SECTION("First constraint violated") {
        CandidateSelect cand;
        cand.select = {1, 1, 1, 1, 1};  // weight=75 > 50, volume=45 > 30
        
        EvalResult result;
        bool success = EvaluateCPU_Select(cfg, soa, cand, &result, &err);
        
        REQUIRE(success);
        REQUIRE(result.constraint_violations[0] > 0.0);  // Weight violation
        REQUIRE(result.constraint_violations[1] > 0.0);  // Volume violation
    }
}

TEST_CASE("EvaluateCPU: Edge Cases", "[v2][eval][cpu][edge]") {
    std::string err;

    SECTION("Zero-valued items") {
        Config cfg;
        cfg.mode = "select";
        cfg.items.count = 3;
        cfg.items.attributes["value"] = {0, 0, 0};
        cfg.items.attributes["weight"] = {1, 2, 3};
        
        ConstraintSpec constraint;
        constraint.attr = "weight";
        constraint.limit = 5.0;
        cfg.constraints.push_back(constraint);
        
        CostTermSpec term;
        term.attr = "value";
        term.weight = 1.0;
        cfg.objective.push_back(term);
        
        HostSoA soa = LoadSoA(cfg);
        CandidateSelect cand;
        cand.select = {1, 1, 1};
        
        EvalResult result;
        bool success = EvaluateCPU_Select(cfg, soa, cand, &result, &err);
        
        REQUIRE(success);
        REQUIRE(result.objective == 0.0);
    }

    SECTION("Negative objective weights") {
        Config cfg;
        cfg.mode = "select";
        cfg.items.count = 3;
        cfg.items.attributes["cost"] = {10, 20, 30};
        cfg.items.attributes["weight"] = {1, 2, 3};
        
        ConstraintSpec constraint;
        constraint.attr = "weight";
        constraint.limit = 10.0;
        cfg.constraints.push_back(constraint);
        
        CostTermSpec term;
        term.attr = "cost";
        term.weight = -1.0;  // Minimize cost
        cfg.objective.push_back(term);
        
        HostSoA soa = LoadSoA(cfg);
        CandidateSelect cand;
        cand.select = {1, 1, 0};  // cost = 30
        
        EvalResult result;
        bool success = EvaluateCPU_Select(cfg, soa, cand, &result, &err);
        
        REQUIRE(success);
        REQUIRE(result.objective == -30.0);  // -1.0 * 30
    }

    SECTION("Very large values") {
        Config cfg;
        cfg.mode = "select";
        cfg.items.count = 2;
        cfg.items.attributes["value"] = {1e9, 1e12};
        cfg.items.attributes["weight"] = {1, 2};
        
        ConstraintSpec constraint;
        constraint.attr = "weight";
        constraint.limit = 10.0;
        cfg.constraints.push_back(constraint);
        
        CostTermSpec term;
        term.attr = "value";
        term.weight = 1.0;
        cfg.objective.push_back(term);
        
        HostSoA soa = LoadSoA(cfg);
        CandidateSelect cand;
        cand.select = {1, 1};
        
        EvalResult result;
        bool success = EvaluateCPU_Select(cfg, soa, cand, &result, &err);
        
        REQUIRE(success);
        REQUIRE(result.objective > 1e12);
    }
}

TEST_CASE("EvaluateCPU: Error Handling", "[v2][eval][cpu][errors]") {
    std::string err;

    SECTION("Invalid selection size") {
        Config cfg = createSimpleEvalConfig();
        HostSoA soa = LoadSoA(cfg);
        
        CandidateSelect cand;
        cand.select = {1, 1};  // Only 2 items, but cfg has 5
        
        EvalResult result;
        bool success = EvaluateCPU_Select(cfg, soa, cand, &result, &err);
        
        // Should fail with error
        REQUIRE(success == false);
        REQUIRE(!err.empty());
    }

    SECTION("Invalid assignment size") {
        Config cfg = createSimpleAssignConfig();
        HostSoA soa = LoadSoA(cfg);
        
        CandidateAssign cand;
        cand.assign = {0, 1, 0};  // Only 3 items, but cfg has 6
        
        EvalResult result;
        bool success = EvaluateCPU_Assign(cfg, soa, cand, &result, &err);
        
        REQUIRE(success == false);
        REQUIRE(!err.empty());
    }

    SECTION("Invalid knapsack index") {
        Config cfg = createSimpleAssignConfig();
        HostSoA soa = LoadSoA(cfg);
        
        CandidateAssign cand;
        cand.assign = {0, 1, 5, -1, -1, -1};  // Knapsack 5 doesn't exist (only 0-1)
        
        EvalResult result;
        bool success = EvaluateCPU_Assign(cfg, soa, cand, &result, &err);
        
        REQUIRE(success == false);
        REQUIRE(!err.empty());
    }
}
