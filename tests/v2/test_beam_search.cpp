// test_beam_search.cpp - Comprehensive tests for v2 Beam Search engine
#include "third_party/catch2/catch_amalgamated.hpp"
#include "v2/Engine.h"
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

// Helper to create a simple knapsack problem
Config createSimpleSelectConfig(int N = 10) {
    Config cfg;
    cfg.mode = "select";
    cfg.items.count = N;
    
    // Create value and weight attributes
    std::vector<double> values;
    std::vector<double> weights;
    
    for (int i = 0; i < N; i++) {
        values.push_back(static_cast<double>((i + 1) * 10));  // 10, 20, 30, ...
        weights.push_back(static_cast<double>(i + 1));         // 1, 2, 3, ...
    }
    
    cfg.items.attributes["value"] = values;
    cfg.items.attributes["weight"] = weights;
    
    // Single capacity constraint
    ConstraintSpec constraint;
    constraint.kind = "capacity";
    constraint.attr = "weight";
    constraint.limit = N * 0.5;  // Can fit about half the items
    cfg.constraints.push_back(constraint);
    
    // Objective: maximize value
    CostTermSpec term;
    term.attr = "value";
    term.weight = 1.0;
    cfg.objective.push_back(term);
    
    return cfg;
}

// Helper to check if a solution is feasible
bool isFeasible(const std::vector<uint8_t>& selection,
                const std::vector<double>& weights,
                double capacity) {
    double total = 0.0;
    for (size_t i = 0; i < selection.size(); i++) {
        if (selection[i]) {
            total += weights[i];
        }
    }
    return total <= capacity + 1e-6;  // Small epsilon for floating point
}

// Helper to calculate objective value
double calculateObjective(const std::vector<uint8_t>& selection,
                          const std::vector<double>& values) {
    double total = 0.0;
    for (size_t i = 0; i < selection.size(); i++) {
        if (selection[i]) {
            total += values[i];
        }
    }
    return total;
}

TEST_CASE("BeamSearch: Basic Functionality", "[v2][beam][basic]") {
    SolverOptions opt;
    opt.beam_width = 16;
    opt.iters = 3;
    opt.seed = 42;
    opt.debug = false;
    
    std::string err;

    SECTION("Small problem (10 items) solves successfully") {
        Config cfg = createSimpleSelectConfig(10);
        HostSoA soa = LoadSoA(cfg);
        
        BeamResult result;
        bool success = SolveBeamSelect(cfg, soa, opt, &result, &err);
        
        REQUIRE(success == true);
        REQUIRE(err.empty());
        REQUIRE(result.best_select.size() == 10);
        REQUIRE(result.objective > 0.0);
        REQUIRE(result.total > 0.0);
    }

    SECTION("Solution is feasible") {
        Config cfg = createSimpleSelectConfig(10);
        HostSoA soa = LoadSoA(cfg);
        
        BeamResult result;
        bool success = SolveBeamSelect(cfg, soa, opt, &result, &err);
        
        REQUIRE(success);
        REQUIRE(isFeasible(result.best_select,
                          cfg.items.attributes["weight"],
                          cfg.constraints[0].limit));
    }

    SECTION("Objective value is correct") {
        Config cfg = createSimpleSelectConfig(10);
        HostSoA soa = LoadSoA(cfg);
        
        BeamResult result;
        bool success = SolveBeamSelect(cfg, soa, opt, &result, &err);
        
        REQUIRE(success);
        double computed_obj = calculateObjective(result.best_select,
                                                 cfg.items.attributes["value"]);
        REQUIRE(std::abs(result.objective - computed_obj) < 1e-6);
    }
}

TEST_CASE("BeamSearch: Problem Sizes", "[v2][beam][scale]") {
    SolverOptions opt;
    opt.beam_width = 16;
    opt.iters = 2;
    opt.seed = 123;
    
    std::string err;

    SECTION("Single item problem") {
        Config cfg = createSimpleSelectConfig(1);
        HostSoA soa = LoadSoA(cfg);
        
        BeamResult result;
        bool success = SolveBeamSelect(cfg, soa, opt, &result, &err);
        
        REQUIRE(success);
        REQUIRE(result.best_select.size() == 1);
        // Item weighs 1, capacity is 0.5, so shouldn't select it
        REQUIRE(result.best_select[0] == 0);
    }

    SECTION("Medium problem (50 items)") {
        Config cfg = createSimpleSelectConfig(50);
        HostSoA soa = LoadSoA(cfg);
        
        BeamResult result;
        bool success = SolveBeamSelect(cfg, soa, opt, &result, &err);
        
        REQUIRE(success);
        REQUIRE(result.best_select.size() == 50);
        REQUIRE(result.objective > 0.0);
    }

    SECTION("Large problem (200 items)") {
        Config cfg = createSimpleSelectConfig(200);
        HostSoA soa = LoadSoA(cfg);
        
        BeamResult result;
        bool success = SolveBeamSelect(cfg, soa, opt, &result, &err);
        
        REQUIRE(success);
        REQUIRE(result.best_select.size() == 200);
    }
}

TEST_CASE("BeamSearch: Solver Options", "[v2][beam][options]") {
    Config cfg = createSimpleSelectConfig(20);
    HostSoA soa = LoadSoA(cfg);
    std::string err;

    SECTION("Different beam widths") {
        SolverOptions opt_narrow, opt_wide;
        opt_narrow.beam_width = 4;
        opt_narrow.iters = 2;
        opt_wide.beam_width = 64;
        opt_wide.iters = 2;
        
        BeamResult result_narrow, result_wide;
        
        REQUIRE(SolveBeamSelect(cfg, soa, opt_narrow, &result_narrow, &err));
        REQUIRE(SolveBeamSelect(cfg, soa, opt_wide, &result_wide, &err));
        
        // Wider beam should generally find better or equal solutions
        REQUIRE(result_wide.objective >= result_narrow.objective - 1e-6);
    }

    SECTION("Different iteration counts") {
        SolverOptions opt_few, opt_many;
        opt_few.beam_width = 16;
        opt_few.iters = 1;
        opt_many.beam_width = 16;
        opt_many.iters = 5;
        
        BeamResult result_few, result_many;
        
        REQUIRE(SolveBeamSelect(cfg, soa, opt_few, &result_few, &err));
        REQUIRE(SolveBeamSelect(cfg, soa, opt_many, &result_many, &err));
        
        // More iterations should generally improve quality
        REQUIRE(result_many.objective >= result_few.objective - 1e-6);
    }

    SECTION("Deterministic with same seed") {
        SolverOptions opt1, opt2;
        opt1.seed = 42;
        opt2.seed = 42;
        opt1.beam_width = 16;
        opt2.beam_width = 16;
        opt1.iters = 3;
        opt2.iters = 3;
        
        BeamResult result1, result2;
        
        REQUIRE(SolveBeamSelect(cfg, soa, opt1, &result1, &err));
        REQUIRE(SolveBeamSelect(cfg, soa, opt2, &result2, &err));
        
        // Same seed should produce identical results
        REQUIRE(result1.objective == result2.objective);
        REQUIRE(result1.best_select == result2.best_select);
    }

    SECTION("Different results with different seeds") {
        SolverOptions opt1, opt2;
        opt1.seed = 42;
        opt2.seed = 999;
        opt1.beam_width = 16;
        opt2.beam_width = 16;
        opt1.iters = 2;
        opt2.iters = 2;
        
        BeamResult result1, result2;
        
        REQUIRE(SolveBeamSelect(cfg, soa, opt1, &result1, &err));
        REQUIRE(SolveBeamSelect(cfg, soa, opt2, &result2, &err));
        
        // Different seeds may produce different results
        // (not guaranteed, but likely with randomized search)
        INFO("Result1 objective: " << result1.objective);
        INFO("Result2 objective: " << result2.objective);
    }
}

TEST_CASE("BeamSearch: Scout Mode", "[v2][beam][scout]") {
    Config cfg = createSimpleSelectConfig(30);
    HostSoA soa = LoadSoA(cfg);
    std::string err;

    SECTION("Scout mode produces valid results") {
        SolverOptions opt;
        opt.beam_width = 16;
        opt.iters = 3;
        opt.scout_mode = true;
        opt.scout_threshold = 0.5;
        opt.scout_top_k = 8;
        opt.seed = 42;
        
        ScoutResult result;
        bool success = SolveBeamScout(cfg, soa, opt, &result, &err);
        
        REQUIRE(success);
        REQUIRE(err.empty());
        REQUIRE(result.best_select.size() == 30);
        REQUIRE(result.objective > 0.0);
        REQUIRE(result.original_item_count == 30);
    }

    SECTION("Scout mode tracks active items") {
        SolverOptions opt;
        opt.beam_width = 16;
        opt.iters = 3;
        opt.scout_mode = true;
        opt.scout_threshold = 0.5;
        opt.scout_top_k = 8;
        
        ScoutResult result;
        bool success = SolveBeamScout(cfg, soa, opt, &result, &err);
        
        REQUIRE(success);
        REQUIRE(result.item_frequency.size() == 30);
        
        // Active items should be a subset
        REQUIRE(result.active_item_count <= 30);
        REQUIRE(result.active_item_count >= 0);
        
        // Frequencies should be between 0 and 1
        for (double freq : result.item_frequency) {
            REQUIRE(freq >= 0.0);
            REQUIRE(freq <= 1.0);
        }
    }

    SECTION("Scout mode with dominance filter") {
        SolverOptions opt;
        opt.beam_width = 16;
        opt.iters = 3;
        opt.scout_mode = true;
        opt.enable_dominance_filter = true;
        opt.dom_eps = 1e-9;
        
        ScoutResult result;
        bool success = SolveBeamScout(cfg, soa, opt, &result, &err);
        
        REQUIRE(success);
        
        // Dominance filter may remove some items
        INFO("Dominated items removed: " << result.dominated_items_removed);
        REQUIRE(result.dominated_items_removed >= 0);
        REQUIRE(result.dominated_items_removed <= 30);
    }

    SECTION("Scout mode timing information") {
        SolverOptions opt;
        opt.beam_width = 16;
        opt.iters = 2;
        opt.scout_mode = true;
        opt.enable_dominance_filter = true;
        
        ScoutResult result;
        bool success = SolveBeamScout(cfg, soa, opt, &result, &err);
        
        REQUIRE(success);
        
        // Timing should be non-negative
        REQUIRE(result.filter_time_ms >= 0.0);
        REQUIRE(result.solve_time_ms >= 0.0);
        
        INFO("Filter time: " << result.filter_time_ms << "ms");
        INFO("Solve time: " << result.solve_time_ms << "ms");
    }
}

TEST_CASE("BeamSearch: Multiple Constraints", "[v2][beam][constraints]") {
    std::string err;
    
    SECTION("Problem with two capacity constraints") {
        Config cfg;
        cfg.mode = "select";
        cfg.items.count = 20;
        
        // Create items with two attributes
        std::vector<double> values, weights, volumes;
        for (int i = 0; i < 20; i++) {
            values.push_back(static_cast<double>((i + 1) * 10));
            weights.push_back(static_cast<double>(i + 1));
            volumes.push_back(static_cast<double>((i + 1) * 0.5));
        }
        
        cfg.items.attributes["value"] = values;
        cfg.items.attributes["weight"] = weights;
        cfg.items.attributes["volume"] = volumes;
        
        // Two capacity constraints
        ConstraintSpec c1, c2;
        c1.kind = "capacity";
        c1.attr = "weight";
        c1.limit = 50.0;
        c2.kind = "capacity";
        c2.attr = "volume";
        c2.limit = 25.0;
        
        cfg.constraints.push_back(c1);
        cfg.constraints.push_back(c2);
        
        // Objective
        CostTermSpec term;
        term.attr = "value";
        term.weight = 1.0;
        cfg.objective.push_back(term);
        
        HostSoA soa = LoadSoA(cfg);
        SolverOptions opt;
        opt.beam_width = 16;
        opt.iters = 3;
        
        BeamResult result;
        bool success = SolveBeamSelect(cfg, soa, opt, &result, &err);
        
        REQUIRE(success);
        REQUIRE(result.best_select.size() == 20);
        
        // Verify both constraints are satisfied
        double total_weight = 0.0, total_volume = 0.0;
        for (size_t i = 0; i < result.best_select.size(); i++) {
            if (result.best_select[i]) {
                total_weight += weights[i];
                total_volume += volumes[i];
            }
        }
        
        REQUIRE(total_weight <= 50.0 + 1e-6);
        REQUIRE(total_volume <= 25.0 + 1e-6);
    }
}

TEST_CASE("BeamSearch: Multi-Objective", "[v2][beam][objective]") {
    std::string err;
    
    SECTION("Problem with weighted objectives") {
        Config cfg;
        cfg.mode = "select";
        cfg.items.count = 15;
        
        std::vector<double> values, weights, priorities;
        for (int i = 0; i < 15; i++) {
            values.push_back(static_cast<double>((i + 1) * 10));
            weights.push_back(static_cast<double>(i + 1));
            priorities.push_back(static_cast<double>(15 - i));  // Reverse priority
        }
        
        cfg.items.attributes["value"] = values;
        cfg.items.attributes["weight"] = weights;
        cfg.items.attributes["priority"] = priorities;
        
        ConstraintSpec constraint;
        constraint.kind = "capacity";
        constraint.attr = "weight";
        constraint.limit = 30.0;
        cfg.constraints.push_back(constraint);
        
        // Two objective terms
        CostTermSpec t1, t2;
        t1.attr = "value";
        t1.weight = 1.0;
        t2.attr = "priority";
        t2.weight = 0.5;
        
        cfg.objective.push_back(t1);
        cfg.objective.push_back(t2);
        
        HostSoA soa = LoadSoA(cfg);
        SolverOptions opt;
        opt.beam_width = 16;
        opt.iters = 3;
        
        BeamResult result;
        bool success = SolveBeamSelect(cfg, soa, opt, &result, &err);
        
        REQUIRE(success);
        REQUIRE(result.objective > 0.0);
    }
}

TEST_CASE("BeamSearch: Edge Cases", "[v2][beam][edge]") {
    SolverOptions opt;
    opt.beam_width = 8;
    opt.iters = 2;
    std::string err;

    SECTION("All items too heavy - selects nothing") {
        Config cfg;
        cfg.mode = "select";
        cfg.items.count = 5;
        cfg.items.attributes["value"] = {10, 20, 30, 40, 50};
        cfg.items.attributes["weight"] = {100, 200, 300, 400, 500};
        
        ConstraintSpec constraint;
        constraint.kind = "capacity";
        constraint.attr = "weight";
        constraint.limit = 50.0;
        cfg.constraints.push_back(constraint);
        
        CostTermSpec term;
        term.attr = "value";
        term.weight = 1.0;
        cfg.objective.push_back(term);
        
        HostSoA soa = LoadSoA(cfg);
        BeamResult result;
        bool success = SolveBeamSelect(cfg, soa, opt, &result, &err);
        
        REQUIRE(success);
        
        // Should select no items
        int selected = 0;
        for (uint8_t sel : result.best_select) {
            if (sel) selected++;
        }
        REQUIRE(selected == 0);
        REQUIRE(result.objective == 0.0);
    }

    SECTION("All items fit - selects all") {
        Config cfg;
        cfg.mode = "select";
        cfg.items.count = 5;
        cfg.items.attributes["value"] = {10, 20, 30, 40, 50};
        cfg.items.attributes["weight"] = {1, 2, 3, 4, 5};
        
        ConstraintSpec constraint;
        constraint.kind = "capacity";
        constraint.attr = "weight";
        constraint.limit = 1000.0;  // Very large capacity
        cfg.constraints.push_back(constraint);
        
        CostTermSpec term;
        term.attr = "value";
        term.weight = 1.0;
        cfg.objective.push_back(term);
        
        HostSoA soa = LoadSoA(cfg);
        BeamResult result;
        bool success = SolveBeamSelect(cfg, soa, opt, &result, &err);
        
        REQUIRE(success);
        
        // Should select all items
        int selected = 0;
        for (uint8_t sel : result.best_select) {
            if (sel) selected++;
        }
        REQUIRE(selected == 5);
        REQUIRE(result.objective == 150.0);  // Sum of all values
    }

    SECTION("Zero capacity - selects nothing") {
        Config cfg = createSimpleSelectConfig(10);
        cfg.constraints[0].limit = 0.0;
        
        HostSoA soa = LoadSoA(cfg);
        BeamResult result;
        bool success = SolveBeamSelect(cfg, soa, opt, &result, &err);
        
        REQUIRE(success);
        
        int selected = 0;
        for (uint8_t sel : result.best_select) {
            if (sel) selected++;
        }
        REQUIRE(selected == 0);
    }
}

TEST_CASE("BeamSearch: Error Handling", "[v2][beam][errors]") {
    SolverOptions opt;
    opt.beam_width = 16;
    opt.iters = 2;
    std::string err;

    SECTION("Missing required attribute fails gracefully") {
        Config cfg;
        cfg.mode = "select";
        cfg.items.count = 10;
        cfg.items.attributes["value"] = std::vector<double>(10, 1.0);
        // Missing "weight" attribute that constraint requires
        
        ConstraintSpec constraint;
        constraint.kind = "capacity";
        constraint.attr = "weight";  // References non-existent attribute
        constraint.limit = 50.0;
        cfg.constraints.push_back(constraint);
        
        CostTermSpec term;
        term.attr = "value";
        term.weight = 1.0;
        cfg.objective.push_back(term);
        
        HostSoA soa = LoadSoA(cfg);
        BeamResult result;
        bool success = SolveBeamSelect(cfg, soa, opt, &result, &err);
        
        // Should fail with error message
        REQUIRE(success == false);
        REQUIRE(!err.empty());
    }
}
