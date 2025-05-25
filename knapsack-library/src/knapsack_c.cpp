#ifndef KNAPSACK_C_H
#define KNAPSACK_C_H

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    int van_id;
    char* village_names;  // comma-separated
    double distance;
    double fuel_cost;
    int crew_size;
} VanTripResult;

typedef struct {
    VanTripResult* trips;
    int num_trips;
    int total_crew;
    int shortfall;
    double total_fuel_cost;
} KnapsackSolution;

// Main solver function
KnapsackSolution* solve_knapsack(const char* csv_path, int target_team_size);

// Cleanup function
void free_knapsack_solution(KnapsackSolution* solution);

#ifdef __cplusplus
}
#endif

#endif