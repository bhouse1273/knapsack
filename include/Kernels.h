#pragma once

void launch_compute_distances(const float* lat, const float* lon, float field_lat, float field_lon, float* out_distances, int N);
void launch_generate_random_plans(int* plans, int* weights, int N, unsigned long seed);
void launch_evaluate_plans(int* plans, float* distances, int* weights, float* fitness, int N, int target_team_size);
void launch_mutate_population(int* plans, int N, int start_index, float mutation_rate, unsigned long seed);
