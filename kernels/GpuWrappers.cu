#include <cuda_runtime.h>
#include "Kernels.h"
#include "Constants.h"

// Kernel declarations
__global__ void compute_distances_kernel(const float*, const float*, float, float, float*, int);
__global__ void generate_random_plans_kernel(int*, int*, int, unsigned long);
__global__ void evaluate_plans_kernel(int*, float*, int*, float*, int, int);
__global__ void mutate_population_kernel(int*, int, int, float, unsigned long);

// Host-side launchers
void launch_compute_distances(const float* lat, const float* lon, float field_lat, float field_lon, float* out_distances, int N) {
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    compute_distances_kernel<<<blocks, threads>>>(lat, lon, field_lat, field_lon, out_distances, N);
}

void launch_generate_random_plans(int* plans, int* weights, int N, unsigned long seed) {
    int threads = 256;
    int blocks = (POPULATION_SIZE + threads - 1) / threads;
    generate_random_plans_kernel<<<blocks, threads>>>(plans, weights, N, seed);
}

void launch_evaluate_plans(int* plans, float* distances, int* weights, float* fitness, int N, int target_team_size) {
    int threads = 256;
    int blocks = (POPULATION_SIZE + threads - 1) / threads;
    evaluate_plans_kernel<<<blocks, threads>>>(plans, distances, weights, fitness, N, target_team_size);
}

void launch_mutate_population(int* plans, int N, int start_index, float mutation_rate, unsigned long seed) {
    int threads = 256;
    int count = POPULATION_SIZE - start_index;
    int blocks = (count + threads - 1) / threads;
    mutate_population_kernel<<<blocks, threads>>>(plans, N, start_index, mutation_rate, seed);
}
