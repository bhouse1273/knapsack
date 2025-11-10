// test_config_validate.cpp - Comprehensive tests for v2::ValidateConfig
#include "third_party/catch2/catch_amalgamated.hpp"
#include "v2/Config.h"
#include <string>

using namespace v2;

// Helper function to create a minimal valid config
Config createMinimalValidConfig() {
    Config cfg;
    cfg.mode = "select";
    cfg.items.count = 3;
    cfg.items.attributes["value"] = {10.0, 20.0, 30.0};
    cfg.items.attributes["weight"] = {5.0, 10.0, 15.0};
    return cfg;
}

// Helper function to create a minimal valid assign config
Config createMinimalValidAssignConfig() {
    Config cfg;
    cfg.mode = "assign";
    cfg.items.count = 5;
    cfg.items.attributes["value"] = {10.0, 20.0, 30.0, 40.0, 50.0};
    cfg.items.attributes["weight"] = {5.0, 10.0, 15.0, 20.0, 25.0};
    cfg.knapsack.K = 2;
    cfg.knapsack.capacities = {50.0, 75.0};
    cfg.knapsack.capacity_attr = "weight";
    return cfg;
}

TEST_CASE("ValidateConfig: Basic Structure", "[v2][config][validation]") {
    std::string err;

    SECTION("Valid minimal config passes") {
        Config cfg = createMinimalValidConfig();
        REQUIRE(ValidateConfig(cfg, &err) == true);
        REQUIRE(err.empty());
    }

    SECTION("Empty config fails - no items") {
        Config cfg;
        cfg.mode = "select";
        cfg.items.count = 0;
        
        REQUIRE(ValidateConfig(cfg, &err) == false);
        REQUIRE_THAT(err, Catch::Matchers::ContainsSubstring("no items"));
    }

    SECTION("Negative items count fails") {
        Config cfg;
        cfg.mode = "select";
        cfg.items.count = -5;
        
        REQUIRE(ValidateConfig(cfg, &err) == false);
        REQUIRE_THAT(err, Catch::Matchers::ContainsSubstring("no items"));
    }
}

TEST_CASE("ValidateConfig: Attribute Array Sizes", "[v2][config][validation]") {
    std::string err;

    SECTION("Attribute arrays must match item count") {
        Config cfg;
        cfg.mode = "select";
        cfg.items.count = 5;
        cfg.items.attributes["value"] = {10.0, 20.0, 30.0};  // Only 3 values!
        cfg.items.attributes["weight"] = {5.0, 10.0, 15.0, 20.0, 25.0};  // 5 values
        
        REQUIRE(ValidateConfig(cfg, &err) == false);
        REQUIRE_THAT(err, Catch::Matchers::ContainsSubstring("value"));
        REQUIRE_THAT(err, Catch::Matchers::ContainsSubstring("3"));
        REQUIRE_THAT(err, Catch::Matchers::ContainsSubstring("5"));
    }

    SECTION("All attributes with correct size pass") {
        Config cfg;
        cfg.mode = "select";
        cfg.items.count = 4;
        cfg.items.attributes["value"] = {10.0, 20.0, 30.0, 40.0};
        cfg.items.attributes["weight"] = {5.0, 10.0, 15.0, 20.0};
        cfg.items.attributes["priority"] = {1.0, 2.0, 3.0, 4.0};
        
        REQUIRE(ValidateConfig(cfg, &err) == true);
        REQUIRE(err.empty());
    }

    SECTION("Empty attributes map is valid if count > 0") {
        Config cfg;
        cfg.mode = "select";
        cfg.items.count = 10;
        // No attributes defined yet
        
        REQUIRE(ValidateConfig(cfg, &err) == true);
    }
}

TEST_CASE("ValidateConfig: Assign Mode Validation", "[v2][config][validation][assign]") {
    std::string err;

    SECTION("Valid assign config passes") {
        Config cfg = createMinimalValidAssignConfig();
        REQUIRE(ValidateConfig(cfg, &err) == true);
        REQUIRE(err.empty());
    }

    SECTION("Assign mode requires K > 0") {
        Config cfg = createMinimalValidAssignConfig();
        cfg.knapsack.K = 0;
        
        REQUIRE(ValidateConfig(cfg, &err) == false);
        REQUIRE_THAT(err, Catch::Matchers::ContainsSubstring("K"));
        REQUIRE_THAT(err, Catch::Matchers::ContainsSubstring("positive"));
    }

    SECTION("Assign mode requires K to match capacities size") {
        Config cfg = createMinimalValidAssignConfig();
        cfg.knapsack.K = 3;
        cfg.knapsack.capacities = {50.0, 75.0};  // Only 2 capacities!
        
        REQUIRE(ValidateConfig(cfg, &err) == false);
        REQUIRE_THAT(err, Catch::Matchers::ContainsSubstring("capacities"));
        REQUIRE_THAT(err, Catch::Matchers::ContainsSubstring("2"));
        REQUIRE_THAT(err, Catch::Matchers::ContainsSubstring("3"));
    }

    SECTION("All capacities must be positive") {
        Config cfg = createMinimalValidAssignConfig();
        cfg.knapsack.capacities = {50.0, -10.0};  // Negative capacity!
        
        REQUIRE(ValidateConfig(cfg, &err) == false);
        REQUIRE_THAT(err, Catch::Matchers::ContainsSubstring("capacity"));
        REQUIRE_THAT(err, Catch::Matchers::ContainsSubstring("positive"));
    }

    SECTION("Zero capacity fails") {
        Config cfg = createMinimalValidAssignConfig();
        cfg.knapsack.capacities = {50.0, 0.0};
        
        REQUIRE(ValidateConfig(cfg, &err) == false);
        REQUIRE_THAT(err, Catch::Matchers::ContainsSubstring("positive"));
    }

    SECTION("capacity_attr must be specified") {
        Config cfg = createMinimalValidAssignConfig();
        cfg.knapsack.capacity_attr = "";
        
        REQUIRE(ValidateConfig(cfg, &err) == false);
        REQUIRE_THAT(err, Catch::Matchers::ContainsSubstring("capacity_attr"));
        REQUIRE_THAT(err, Catch::Matchers::ContainsSubstring("specified"));
    }

    SECTION("capacity_attr must exist in items.attributes") {
        Config cfg = createMinimalValidAssignConfig();
        cfg.knapsack.capacity_attr = "nonexistent";
        
        REQUIRE(ValidateConfig(cfg, &err) == false);
        REQUIRE_THAT(err, Catch::Matchers::ContainsSubstring("nonexistent"));
        REQUIRE_THAT(err, Catch::Matchers::ContainsSubstring("not found"));
    }

    SECTION("Large K value works") {
        Config cfg = createMinimalValidAssignConfig();
        cfg.knapsack.K = 100;
        cfg.knapsack.capacities = std::vector<double>(100, 50.0);
        
        REQUIRE(ValidateConfig(cfg, &err) == true);
        REQUIRE(err.empty());
    }
}

TEST_CASE("ValidateConfig: Constraint Validation", "[v2][config][validation][constraints]") {
    std::string err;

    SECTION("Constraint with valid attribute passes") {
        Config cfg = createMinimalValidConfig();
        
        ConstraintSpec constraint;
        constraint.kind = "capacity";
        constraint.attr = "weight";
        constraint.limit = 50.0;
        cfg.constraints.push_back(constraint);
        
        REQUIRE(ValidateConfig(cfg, &err) == true);
        REQUIRE(err.empty());
    }

    SECTION("Constraint with unknown attribute fails") {
        Config cfg = createMinimalValidConfig();
        
        ConstraintSpec constraint;
        constraint.kind = "capacity";
        constraint.attr = "unknown_attr";
        constraint.limit = 50.0;
        cfg.constraints.push_back(constraint);
        
        REQUIRE(ValidateConfig(cfg, &err) == false);
        REQUIRE_THAT(err, Catch::Matchers::ContainsSubstring("unknown_attr"));
        REQUIRE_THAT(err, Catch::Matchers::ContainsSubstring("unknown attribute"));
    }

    SECTION("Constraint with negative limit fails") {
        Config cfg = createMinimalValidConfig();
        
        ConstraintSpec constraint;
        constraint.kind = "capacity";
        constraint.attr = "weight";
        constraint.limit = -10.0;
        cfg.constraints.push_back(constraint);
        
        REQUIRE(ValidateConfig(cfg, &err) == false);
        REQUIRE_THAT(err, Catch::Matchers::ContainsSubstring("negative limit"));
    }

    SECTION("Constraint with empty attr is allowed") {
        Config cfg = createMinimalValidConfig();
        
        ConstraintSpec constraint;
        constraint.kind = "cardinality";
        constraint.attr = "";  // Empty attr
        constraint.limit = 5.0;
        cfg.constraints.push_back(constraint);
        
        REQUIRE(ValidateConfig(cfg, &err) == true);
    }

    SECTION("Multiple constraints all valid") {
        Config cfg = createMinimalValidConfig();
        
        ConstraintSpec c1, c2, c3;
        c1.attr = "weight";
        c1.limit = 50.0;
        c2.attr = "value";
        c2.limit = 100.0;
        c3.attr = "weight";
        c3.limit = 25.0;
        
        cfg.constraints.push_back(c1);
        cfg.constraints.push_back(c2);
        cfg.constraints.push_back(c3);
        
        REQUIRE(ValidateConfig(cfg, &err) == true);
    }

    SECTION("Soft constraint with penalty") {
        Config cfg = createMinimalValidConfig();
        
        ConstraintSpec constraint;
        constraint.attr = "weight";
        constraint.limit = 50.0;
        constraint.soft = true;
        constraint.penalty.weight = 1.5;
        constraint.penalty.power = 2.0;
        cfg.constraints.push_back(constraint);
        
        REQUIRE(ValidateConfig(cfg, &err) == true);
    }
}

TEST_CASE("ValidateConfig: Objective Validation", "[v2][config][validation][objective]") {
    std::string err;

    SECTION("Objective with valid attribute passes") {
        Config cfg = createMinimalValidConfig();
        
        CostTermSpec term;
        term.attr = "value";
        term.weight = 1.0;
        cfg.objective.push_back(term);
        
        REQUIRE(ValidateConfig(cfg, &err) == true);
        REQUIRE(err.empty());
    }

    SECTION("Objective with unknown attribute fails") {
        Config cfg = createMinimalValidConfig();
        
        CostTermSpec term;
        term.attr = "profit";  // Doesn't exist
        term.weight = 1.0;
        cfg.objective.push_back(term);
        
        REQUIRE(ValidateConfig(cfg, &err) == false);
        REQUIRE_THAT(err, Catch::Matchers::ContainsSubstring("profit"));
        REQUIRE_THAT(err, Catch::Matchers::ContainsSubstring("unknown attribute"));
    }

    SECTION("Multiple objective terms all valid") {
        Config cfg = createMinimalValidConfig();
        
        CostTermSpec t1, t2;
        t1.attr = "value";
        t1.weight = 1.0;
        t2.attr = "weight";
        t2.weight = -0.5;
        
        cfg.objective.push_back(t1);
        cfg.objective.push_back(t2);
        
        REQUIRE(ValidateConfig(cfg, &err) == true);
    }

    SECTION("Objective term with negative weight is allowed") {
        Config cfg = createMinimalValidConfig();
        
        CostTermSpec term;
        term.attr = "value";
        term.weight = -2.5;
        cfg.objective.push_back(term);
        
        REQUIRE(ValidateConfig(cfg, &err) == true);
    }

    SECTION("Empty objective is valid") {
        Config cfg = createMinimalValidConfig();
        // No objective terms added
        
        REQUIRE(ValidateConfig(cfg, &err) == true);
    }
}

TEST_CASE("ValidateConfig: Complex Scenarios", "[v2][config][validation][integration]") {
    std::string err;

    SECTION("Full assign config with everything") {
        Config cfg;
        cfg.mode = "assign";
        cfg.version = 2;
        cfg.random_seed = 42;
        
        // Items with multiple attributes
        cfg.items.count = 10;
        cfg.items.attributes["value"] = {10, 20, 30, 40, 50, 60, 70, 80, 90, 100};
        cfg.items.attributes["weight"] = {5, 10, 15, 20, 25, 30, 35, 40, 45, 50};
        cfg.items.attributes["priority"] = {1, 2, 1, 3, 2, 1, 2, 3, 1, 2};
        
        // Knapsack specs
        cfg.knapsack.K = 3;
        cfg.knapsack.capacities = {100.0, 150.0, 125.0};
        cfg.knapsack.capacity_attr = "weight";
        
        // Multiple constraints
        ConstraintSpec c1, c2;
        c1.kind = "capacity";
        c1.attr = "weight";
        c1.limit = 100.0;
        c2.kind = "cardinality";
        c2.limit = 5.0;
        cfg.constraints.push_back(c1);
        cfg.constraints.push_back(c2);
        
        // Multiple objective terms
        CostTermSpec t1, t2;
        t1.attr = "value";
        t1.weight = 1.0;
        t2.attr = "priority";
        t2.weight = 0.5;
        cfg.objective.push_back(t1);
        cfg.objective.push_back(t2);
        
        // Blocks
        BlockSpec block;
        block.name = "high_priority";
        block.start = 0;
        block.count = 3;
        cfg.blocks.push_back(block);
        
        REQUIRE(ValidateConfig(cfg, &err) == true);
        REQUIRE(err.empty());
    }

    SECTION("Large dataset with 1000 items") {
        Config cfg;
        cfg.mode = "select";
        cfg.items.count = 1000;
        
        std::vector<double> values(1000);
        std::vector<double> weights(1000);
        for (int i = 0; i < 1000; i++) {
            values[i] = static_cast<double>(i * 10);
            weights[i] = static_cast<double>(i + 1);
        }
        
        cfg.items.attributes["value"] = values;
        cfg.items.attributes["weight"] = weights;
        
        CostTermSpec term;
        term.attr = "value";
        term.weight = 1.0;
        cfg.objective.push_back(term);
        
        REQUIRE(ValidateConfig(cfg, &err) == true);
    }

    SECTION("Multiple validation errors reported correctly") {
        Config cfg;
        cfg.mode = "assign";
        cfg.items.count = 5;
        cfg.items.attributes["value"] = {10, 20, 30};  // Wrong size
        cfg.knapsack.K = 0;  // Invalid K
        cfg.knapsack.capacities = {};  // Empty
        cfg.knapsack.capacity_attr = "";  // Not specified
        
        // Should fail on first error encountered
        REQUIRE(ValidateConfig(cfg, &err) == false);
        REQUIRE(!err.empty());
    }
}

TEST_CASE("ValidateConfig: Edge Cases", "[v2][config][validation][edge]") {
    std::string err;

    SECTION("Single item config") {
        Config cfg;
        cfg.mode = "select";
        cfg.items.count = 1;
        cfg.items.attributes["value"] = {42.0};
        
        REQUIRE(ValidateConfig(cfg, &err) == true);
    }

    SECTION("Very large capacity values") {
        Config cfg = createMinimalValidAssignConfig();
        cfg.knapsack.capacities = {1e9, 1e12};
        
        REQUIRE(ValidateConfig(cfg, &err) == true);
    }

    SECTION("Zero-valued attributes are valid") {
        Config cfg;
        cfg.mode = "select";
        cfg.items.count = 3;
        cfg.items.attributes["value"] = {0.0, 0.0, 0.0};
        cfg.items.attributes["weight"] = {10.0, 20.0, 30.0};
        
        REQUIRE(ValidateConfig(cfg, &err) == true);
    }

    SECTION("Null error pointer doesn't crash") {
        Config cfg = createMinimalValidConfig();
        REQUIRE(ValidateConfig(cfg, nullptr) == true);
        
        cfg.items.count = 0;
        REQUIRE(ValidateConfig(cfg, nullptr) == false);
    }

    SECTION("Config with only blocks defined") {
        Config cfg;
        cfg.mode = "select";
        cfg.items.count = 10;
        cfg.items.attributes["value"] = std::vector<double>(10, 1.0);
        
        BlockSpec b1, b2;
        b1.name = "block1";
        b1.indices = {0, 1, 2};
        b2.name = "block2";
        b2.start = 5;
        b2.count = 3;
        
        cfg.blocks.push_back(b1);
        cfg.blocks.push_back(b2);
        
        REQUIRE(ValidateConfig(cfg, &err) == true);
    }
}
