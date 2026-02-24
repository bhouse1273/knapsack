#pragma once
#include <vector>

// Shared context for recursive knapsack layers
struct Context {
    std::vector<int> item_indices;
    float* d_lat;
    float* d_lon;
    int* d_weights;
    int N;
    float field_lat;
    float field_lon;
    std::vector<std::vector<int>> warm_population; // optional warm-start bitstrings (length == N)
    std::vector<int> warm_seed;                   // preferred primary warm start
};
