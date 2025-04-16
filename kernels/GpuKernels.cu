#include <curand_kernel.h>
#include <cuda_runtime.h>
#include "Constants.h"

__device__ float haversine(float lat1, float lon1, float lat2, float lon2) {
    const float R = 6371.0f;
    float dLat = (lat2 - lat1) * 3.1415926f / 180.0f;
    float dLon = (lon2 - lon1) * 3.1415926f / 180.0f;
    float a = sinf(dLat / 2) * sinf(dLat / 2) +
              cosf(lat1 * 3.1415926f / 180.0f) * cosf(lat2 * 3.1415926f / 180.0f) *
              sinf(dLon / 2) * sinf(dLon / 2);
    float c = 2 * atan2f(sqrtf(a), sqrtf(1 - a));
    return R * c;
}

__global__ void compute_distances_kernel(
    const float* lat, const float* lon,
    float field_lat, float field_lon,
    float* out_distances, int N
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        out_distances[i] = haversine(lat[i], lon[i], field_lat, field_lon);
    }
}

__global__ void generate_random_plans_kernel(int* plans, int* village_weights, int N, unsigned long seed) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= POPULATION_SIZE) return;

    curandState state;
    curand_init(seed + id, 0, 0, &state);

    int total = 0;
    for (int i = 0; i < N; ++i) {
        int w = village_weights[i];
        int include = (curand(&state) % 2 == 1);
        if (include && (total + w <= MAX_CAPACITY)) {
            plans[id * N + i] = 1;
            total += w;
        } else {
            plans[id * N + i] = 0;
        }
    }
}

__global__ void evaluate_plans_kernel(
    int* plans, float* distances, int* weights, float* fitness, int N, int target_team_size
) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= POPULATION_SIZE) return;

    float total_cost = 0.0f;
    int total_workers = 0;

    for (int i = 0; i < N; ++i) {
        if (plans[id * N + i]) {
            total_cost += distances[i] * 2.0f;
            total_workers += weights[i];
        }
    }

    float penalty = 0.0f;
    if (total_workers > MAX_CAPACITY)
        penalty += 10000.0f + 100.0f * (total_workers - MAX_CAPACITY);

    float diff = abs(total_workers - target_team_size);
    penalty += diff * 100.0f;

    fitness[id] = total_cost + penalty;
}

__global__ void mutate_population_kernel(int* plans, int N, int start_index, float mutation_rate, unsigned long seed) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    int plan_id = id + start_index;
    if (plan_id >= POPULATION_SIZE) return;

    curandState state;
    curand_init(seed + plan_id, 0, 0, &state);

    for (int i = 0; i < N; ++i) {
        float prob = curand_uniform(&state);
        if (prob < mutation_rate) {
            plans[plan_id * N + i] ^= 1;
        }
    }
}
