#include "v2/Config.h"
#include "v2/Data.h"
#include "v2/Eval.h"
#include <iostream>
#include <cmath>

using namespace v2;

int main() {
    // Create simple config
    Config cfg;
    cfg.mode = "select";
    cfg.items.count = 5;
    cfg.items.attributes["value"] = {10.0, 20.0, 30.0, 40.0, 50.0};
    cfg.items.attributes["weight"] = {5.0, 10.0, 15.0, 20.0, 25.0};
    
    // Add soft constraint with quadratic penalty
    ConstraintSpec constraint;
    constraint.kind = "capacity";
    constraint.attr = "weight";
    constraint.limit = 50.0;
    constraint.soft = true;
    constraint.penalty.weight = 1.0;
    constraint.penalty.power = 2.0;  // Quadratic!
    cfg.constraints.push_back(constraint);
    
    // Objective
    CostTermSpec term;
    term.attr = "value";
    term.weight = 1.0;
    cfg.objective.push_back(term);
    
    // Load SoA
    HostSoA soa;
    std::string err;
    if (!BuildHostSoA(cfg, &soa, &err)) {
        std::cerr << "BuildHostSoA failed: " << err << std::endl;
        return 1;
    }
    
    // Print config state
    std::cout << "Config constraint[0]:" << std::endl;
    std::cout << "  soft: " << cfg.constraints[0].soft << std::endl;
    std::cout << "  penalty.weight: " << cfg.constraints[0].penalty.weight << std::endl;
    std::cout << "  penalty.power: " << cfg.constraints[0].penalty.power << std::endl;
    std::cout << "  limit: " << cfg.constraints[0].limit << std::endl;
    
    // Create candidate with violation
    CandidateSelect cand;
    cand.select = {1, 1, 1, 1, 0};  // weight = 5+10+15+20 = 50... wait no, = 60!
    
    // Actually let me calculate: we want violation of 10
    // limit = 50, so we need total weight = 60
    // Items: 5, 10, 15, 20, 25
    // Take: 1,1,1,1,0 = 5+10+15+20 = 50 (no violation)
    // Take: 1,1,1,1,1 = 75 (violation = 25)
    // Take: 0,1,1,1,1 = 70 (violation = 20)
    // Take: 1,0,1,1,1 = 65 (violation = 15)
    // Take: 1,1,0,1,1 = 60 (violation = 10) ✓
    
    cand.select = {1, 1, 0, 1, 1};  // weight = 5+10+20+25 = 60, violation = 10
    
    std::cout << "\nCandidate: ";
    for (auto v : cand.select) std::cout << (int)v << " ";
    std::cout << std::endl;
    
    double total_weight = 0.0;
    for (size_t i = 0; i < cand.select.size(); i++) {
        if (cand.select[i]) {
            total_weight += cfg.items.attributes["weight"][i];
        }
    }
    std::cout << "Total weight: " << total_weight << std::endl;
    std::cout << "Expected violation: " << std::max(0.0, total_weight - 50.0) << std::endl;
    std::cout << "Expected penalty: " << 1.0 * std::pow(10.0, 2.0) << std::endl;
    
    // Evaluate
    EvalResult result;
    if (!EvaluateCPU_Select(cfg, soa, cand, &result, &err)) {
        std::cerr << "Evaluation failed: " << err << std::endl;
        return 1;
    }
    
    std::cout << "\nActual results:" << std::endl;
    std::cout << "  objective: " << result.objective << std::endl;
    std::cout << "  penalty: " << result.penalty << std::endl;
    std::cout << "  total: " << result.total << std::endl;
    std::cout << "  violation: " << result.constraint_violations[0] << std::endl;
    
    // Test pow_pos directly
    auto pow_pos = [](double x, double p) {
        return x <= 0.0 ? 0.0 : std::pow(x, p);
    };
    std::cout << "\nDirect pow_pos test:" << std::endl;
    std::cout << "  pow_pos(10.0, 2.0) = " << pow_pos(10.0, 2.0) << std::endl;
    std::cout << "  std::pow(10.0, 2.0) = " << std::pow(10.0, 2.0) << std::endl;
    
    return 0;
}
