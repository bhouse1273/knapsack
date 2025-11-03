#ifndef KNAPSACK_C_H
#define KNAPSACK_C_H

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    int group_id;
    char* item_names;  // comma-separated
    double distance;
    double cost;
    int units;
} GroupTrip;

typedef struct {
    GroupTrip* trips;
    int num_trips;
    int total_units;
    int shortfall;
    double total_cost;
} KnapsackSolution;

// Main solver function
KnapsackSolution* solve_knapsack(const char* csv_path, int target_team_size);

// Cleanup function
void free_knapsack_solution(KnapsackSolution* solution);

// ---------------- V2 C API (config-driven) ----------------

typedef struct {
    int num_items;         // number of items in the problem
    int* select;           // length num_items; 0/1 selection (select mode)
    double objective;      // sum of weighted objective terms
    double penalty;        // total penalty from soft constraints
    double total;          // objective - penalty
} KnapsackSolutionV2;

// Solve from a JSON string according to the V2 schema (see docs/v2/README.md).
// Currently supports mode="select" with a single capacity constraint in the Metal path;
// CPU fallback supports multiple capacity constraints as defined in cfg.
// Returns 0 on success and allocates *out_solution; caller must free with free_knapsack_solution_v2.
int solve_knapsack_v2_from_json(const char* json_config, const char* options_json, KnapsackSolutionV2** out_solution);

void free_knapsack_solution_v2(KnapsackSolutionV2* sol);

// Accessor helpers to avoid cgo keyword conflicts
int* ks_v2_select_ptr(KnapsackSolutionV2* sol);

#ifdef __cplusplus
}
#endif

#endif