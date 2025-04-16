#include "RecursiveSolver.h"
#include "Kernels.h"
#include "Constants.h"
#include <vector>
#include <algorithm>
#include <cstring>
#include <cuda_runtime.h>
#include <ctime>
#include <iostream>
#include <numeric>
#include <cmath>
#include "InputModule.h"
#include "RouteUtils.h" // for shared haversine

std::vector<int> recursive_worker(Context ctx, int depth, int target_team_size, const std::vector<Village>& villages) {
    if (ctx.N == 0 || depth >= MAX_RECURSION_DEPTH) {
#ifdef DEBUG_VAN
        std::cout << "⚠️ Empty or max-depth block encountered at depth " << depth << ". Returning no plan.\n";
#endif
        return {};
    }

#ifdef DEBUG_VAN
    std::cout << "🧠 Classical block: " << ctx.N << " villages in scope\n";
    for (int i = 0; i < ctx.N; ++i) {
        int globalIdx = ctx.village_indices[i];
        const auto& v = villages[globalIdx];
        double d = haversine(GARAGE_LAT, GARAGE_LON, v.latitude, v.longitude);
        int w = std::max(1, v.workers);
        int s = std::max(1, v.productivityScore);
        double cost = std::sqrt(d) / std::pow(w * s, 0.75);
        std::cout << "📍 Village: " << v.name
                  << ", d=" << std::fixed << std::setprecision(2) << d
                  << ", w=" << w
                  << ", s=" << s
                  << ", cost=" << std::setprecision(3) << cost << "\n";
    }
#endif

    float* d_distances;
    int* d_plans;
    float* d_fitness;

    cudaMalloc(&d_distances, ctx.N * sizeof(float));
    cudaMalloc(&d_plans, POPULATION_SIZE * ctx.N * sizeof(int));
    cudaMalloc(&d_fitness, POPULATION_SIZE * sizeof(float));

    // NOTE: Distance calculation used here should be consistent with garage→village and village→field→garage logic in RoutePlanner.
    launch_compute_distances(ctx.d_lat, ctx.d_lon, ctx.field_lat, ctx.field_lon, d_distances, ctx.N);
    launch_generate_random_plans(d_plans, ctx.d_weights, ctx.N, time(NULL));
    launch_evaluate_plans(d_plans, d_distances, ctx.d_weights, d_fitness, ctx.N, target_team_size);

    float* h_fitness = new float[POPULATION_SIZE];
    cudaMemcpy(h_fitness, d_fitness, POPULATION_SIZE * sizeof(float), cudaMemcpyDeviceToHost);

    std::vector<std::pair<float, int>> ranked;
    for (int i = 0; i < POPULATION_SIZE; ++i)
        ranked.emplace_back(h_fitness[i], i);

    std::sort(ranked.begin(), ranked.end());

    int* h_plans = new int[POPULATION_SIZE * ctx.N];
    cudaMemcpy(h_plans, d_plans, POPULATION_SIZE * ctx.N * sizeof(int), cudaMemcpyDeviceToHost);
    for (int i = 0; i < ELITE_COUNT; ++i) {
        int from = ranked[i].second;
        std::memcpy(&h_plans[i * ctx.N], &h_plans[from * ctx.N], ctx.N * sizeof(int));
    }
    cudaMemcpy(d_plans, h_plans, ELITE_COUNT * ctx.N * sizeof(int), cudaMemcpyHostToDevice);

    launch_mutate_population(d_plans, ctx.N, ELITE_COUNT, MUTATION_RATE, time(NULL));

    std::vector<int> best_plan(ctx.N, 0);
    std::vector<int> selectionBits(ctx.N, 0);
    for (int i = 0; i < ctx.N; ++i) {
        best_plan[i] = h_plans[ranked[0].second * ctx.N + i];
        selectionBits[i] = best_plan[i];
    }

#ifdef DEBUG_VAN
    std::cout << "🔎 Best plan: ";
    for (int b : selectionBits)
        std::cout << b;
    std::cout << "\n";
#endif

    delete[] h_plans;
    delete[] h_fitness;
    cudaFree(d_distances);
    cudaFree(d_plans);
    cudaFree(d_fitness);

    return best_plan;
}
