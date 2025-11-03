#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include "v2/Engine.h"
#include "v2/Config.h"
#include "v2/Data.h"

namespace py = pybind11;

// Helper to build a Config and HostSoA from Python arrays
bool build_knapsack_problem(
    const std::vector<double>& values,
    const std::vector<double>& weights,
    double capacity,
    v2::Config* cfg,
    v2::HostSoA* soa,
    std::string* err
) {
    if (values.size() != weights.size()) {
        if (err) *err = "values and weights must have the same length";
        return false;
    }
    
    int n = static_cast<int>(values.size());
    
    // Build Config for a single capacity knapsack in "select" mode
    cfg->mode = "select";
    cfg->random_seed = 1234;
    
    // Set up items spec with attributes
    cfg->items.count = n;
    cfg->items.attributes["value"].resize(n);
    cfg->items.attributes["weight"].resize(n);
    
    for (int i = 0; i < n; ++i) {
        cfg->items.attributes["value"][i] = values[i];
        cfg->items.attributes["weight"][i] = weights[i];
    }
    
    // Set up knapsack spec (even for select mode, we define capacity)
    cfg->knapsack.K = 1;
    cfg->knapsack.capacities = {capacity};
    cfg->knapsack.capacity_attr = "weight";
    
    // Set up constraints (capacity constraint)
    v2::ConstraintSpec constraint;
    constraint.kind = "capacity";
    constraint.attr = "weight";
    constraint.limit = capacity;
    constraint.soft = false;
    cfg->constraints.push_back(constraint);
    
    // Set up objective (maximize value)
    v2::CostTermSpec cost;
    cost.attr = "value";
    cost.weight = 1.0;
    cfg->objective.push_back(cost);
    
    // Build HostSoA
    if (!v2::BuildHostSoA(*cfg, soa, err)) {
        return false;
    }
    
    return true;
}

// Simple result structure matching the API
struct SimpleSolution {
    std::vector<int> selected_indices;
    double best_value;
    double solve_time_ms;
};

// Wrapper for solve() that accepts Python arrays
SimpleSolution solve_wrapper(
    const std::vector<double>& values,
    const std::vector<double>& weights,
    double capacity,
    const py::dict& config_dict = py::dict()
) {
    v2::Config cfg;
    v2::HostSoA soa;
    std::string err;
    
    if (!build_knapsack_problem(values, weights, capacity, &cfg, &soa, &err)) {
        throw std::runtime_error("Failed to build problem: " + err);
    }
    
    // Set solver options from config_dict
    v2::SolverOptions opt;
    opt.beam_width = config_dict.contains("beam_width") ? config_dict["beam_width"].cast<int>() : 16;
    opt.iters = config_dict.contains("iters") ? config_dict["iters"].cast<int>() : 3;
    opt.seed = config_dict.contains("seed") ? config_dict["seed"].cast<unsigned int>() : 1234;
    opt.debug = config_dict.contains("debug") ? config_dict["debug"].cast<bool>() : false;
    opt.enable_dominance_filter = config_dict.contains("enable_dominance") ? 
        config_dict["enable_dominance"].cast<bool>() : false;
    
    // Run solver
    v2::BeamResult result;
    auto start = std::chrono::high_resolution_clock::now();
    
    if (!v2::SolveBeamSelect(cfg, soa, opt, &result, &err)) {
        throw std::runtime_error("Solve failed: " + err);
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    double solve_ms = std::chrono::duration<double, std::milli>(end - start).count();
    
    // Extract selected indices
    SimpleSolution sol;
    sol.best_value = result.objective;
    sol.solve_time_ms = solve_ms;
    for (size_t i = 0; i < result.best_select.size(); ++i) {
        if (result.best_select[i]) {
            sol.selected_indices.push_back(static_cast<int>(i));
        }
    }
    
    return sol;
}

// Wrapper for solve_scout() that accepts Python arrays
v2::ScoutResult solve_scout_wrapper(
    const std::vector<double>& values,
    const std::vector<double>& weights,
    double capacity,
    const py::dict& config_dict = py::dict()
) {
    v2::Config cfg;
    v2::HostSoA soa;
    std::string err;
    
    if (!build_knapsack_problem(values, weights, capacity, &cfg, &soa, &err)) {
        throw std::runtime_error("Failed to build problem: " + err);
    }
    
    // Set solver options from config_dict
    v2::SolverOptions opt;
    opt.beam_width = config_dict.contains("beam_width") ? config_dict["beam_width"].cast<int>() : 16;
    opt.iters = config_dict.contains("iters") ? config_dict["iters"].cast<int>() : 3;
    opt.seed = config_dict.contains("seed") ? config_dict["seed"].cast<unsigned int>() : 1234;
    opt.debug = config_dict.contains("debug") ? config_dict["debug"].cast<bool>() : false;
    opt.enable_dominance_filter = config_dict.contains("enable_dominance") ? 
        config_dict["enable_dominance"].cast<bool>() : false;
    opt.scout_mode = true;  // Always enable scout mode
    opt.scout_threshold = config_dict.contains("scout_threshold") ? 
        config_dict["scout_threshold"].cast<double>() : 0.5;
    opt.scout_top_k = config_dict.contains("scout_top_k") ? 
        config_dict["scout_top_k"].cast<int>() : 8;
    
    // Run scout mode
    v2::ScoutResult result;
    if (!v2::SolveBeamScout(cfg, soa, opt, &result, &err)) {
        throw std::runtime_error("Scout solve failed: " + err);
    }
    
    return result;
}

PYBIND11_MODULE(knapsack, m) {
    m.doc() = "Python bindings for knapsack solver with beam search and scout mode";
    
    // Expose SimpleSolution struct
    py::class_<SimpleSolution>(m, "Solution")
        .def(py::init<>())
        .def_readwrite("best_value", &SimpleSolution::best_value)
        .def_readwrite("selected_indices", &SimpleSolution::selected_indices)
        .def_readwrite("solve_time_ms", &SimpleSolution::solve_time_ms)
        .def("__repr__", [](const SimpleSolution& sol) {
            return "<Solution value=" + std::to_string(sol.best_value) + 
                   " items=" + std::to_string(sol.selected_indices.size()) + ">";
        });
    
    // Expose ScoutResult struct
    py::class_<v2::ScoutResult>(m, "ScoutResult")
        .def(py::init<>())
        .def_readwrite("active_items", &v2::ScoutResult::active_items)
        .def_readwrite("item_frequency", &v2::ScoutResult::item_frequency)
        .def_readwrite("original_item_count", &v2::ScoutResult::original_item_count)
        .def_readwrite("active_item_count", &v2::ScoutResult::active_item_count)
        .def_readwrite("solve_time_ms", &v2::ScoutResult::solve_time_ms)
        .def_readwrite("filter_time_ms", &v2::ScoutResult::filter_time_ms)
        .def_readwrite("objective", &v2::ScoutResult::objective)
        .def_readwrite("penalty", &v2::ScoutResult::penalty)
        .def_readwrite("total", &v2::ScoutResult::total)
        .def_readwrite("best_select", &v2::ScoutResult::best_select)
        .def("__repr__", [](const v2::ScoutResult& result) {
            return "<ScoutResult active=" + std::to_string(result.active_item_count) + 
                   "/" + std::to_string(result.original_item_count) + 
                   " reduction=" + std::to_string(100.0 * (1.0 - (double)result.active_item_count / result.original_item_count)) + "%>";
        });
    
    // Main solve function
    m.def("solve", &solve_wrapper,
          py::arg("values"),
          py::arg("weights"),
          py::arg("capacity"),
          py::arg("config") = py::dict(),
          R"pbdoc(
              Solve a knapsack problem using beam search.
              
              Parameters
              ----------
              values : list of float
                  Value of each item
              weights : list of float
                  Weight of each item
              capacity : float
                  Knapsack capacity constraint
              config : dict, optional
                  Configuration options:
                  - beam_width: int (default 16)
                  - iters: int (default 3)
                  - seed: int (default 1234)
                  - debug: bool (default False)
                  - enable_dominance: bool (default False)
              
              Returns
              -------
              Solution
                  Solution object with best_value, selected_indices, solve_time_ms
              
              Examples
              --------
              >>> import knapsack
              >>> values = [60.0, 100.0, 120.0]
              >>> weights = [10.0, 20.0, 30.0]
              >>> capacity = 50.0
              >>> solution = knapsack.solve(values, weights, capacity)
              >>> print(f"Best value: {solution.best_value}")
          )pbdoc");
    
    // Scout mode function
    m.def("solve_scout", &solve_scout_wrapper,
          py::arg("values"),
          py::arg("weights"),
          py::arg("capacity"),
          py::arg("config") = py::dict(),
          R"pbdoc(
              Use beam search as a "data scout" to identify active items.
              
              This function runs beam search and analyzes which items appear
              frequently in high-quality solutions. It returns a filtered
              "active set" of items that can be passed to an exact solver
              (like Gurobi, CPLEX, or SCIP) for guaranteed optimality.
              
              Parameters
              ----------
              values : list of float
                  Value of each item
              weights : list of float
                  Weight of each item
              capacity : float
                  Knapsack capacity constraint
              config : dict, optional
                  Configuration options (same as solve(), plus):
                  - scout_threshold: float (default 0.5) - Minimum frequency to include item
                  - scout_top_k: int (default 8) - Number of top candidates to analyze
              
              Returns
              -------
              ScoutResult
                  Result object with active_items list, item_frequency, reduction metrics, etc.
              
              Examples
              --------
              >>> result = knapsack.solve_scout(values, weights, capacity, 
              ...                                {"scout_threshold": 0.5, "scout_top_k": 8})
              >>> print(f"Reduced from {result.original_item_count} to {result.active_item_count} items")
              >>> print(f"Active items: {result.active_items}")
          )pbdoc");
    
    // Version info
    m.attr("__version__") = "2.0.0";
}
