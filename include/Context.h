#pragma once
#include <vector>

// Shared context for recursive knapsack layers
struct Context {
    std::vector<int> village_indices;
    float* d_lat;
    float* d_lon;
    int* d_weights;
    int N;
    float field_lat;
    float field_lon;
};
