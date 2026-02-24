#pragma once

#include <cstddef>
#include <random>
#include <utility>
#include <vector>

struct ConstraintQAOAProblem {
    int num_items = 0;
    std::vector<float> weights;
    std::vector<float> values;
    float capacity = 0.f;
    std::vector<std::pair<int, int>> entangler_edges; // widened light-cone edges
    std::vector<std::vector<unsigned char>> warm_starts; // pre-feasible seeds
    std::vector<unsigned char> greedy_seed; // fallback seed
    float analog_bias = 0.f; // density hint for analog drift (0..1)
};

struct ConstraintQAOAConfig {
    int layers = 3;
    int mixer_moves = 4;
    int analog_passes = 2;
    int population_size = 64;
    float beta_min = 0.05f;
    float beta_max = 1.2f;
    float gamma_min = 0.2f;
    float gamma_max = 2.0f;
    float penalty_base = 120.0f;
    float analog_drift = 0.35f;
    float capacity_relaxation = 0.05f;
};

struct HybridAnalogDigitalStats {
    int warm_start_count = 0;
    int analog_cycles = 0;
    int mixer_moves = 0;
    std::vector<float> beta_schedule;
    std::vector<float> gamma_schedule;
};

struct ConstraintQAOAResult {
    std::vector<std::vector<unsigned char>> population;
    HybridAnalogDigitalStats stats;
};

class ConstraintQAOAEngine {
public:
    explicit ConstraintQAOAEngine(const ConstraintQAOAConfig& cfg);
    ConstraintQAOAResult run(const ConstraintQAOAProblem& problem, unsigned seed = 0) const;

private:
    struct LayerParams {
        float beta;
        float gamma;
    };

    ConstraintQAOAConfig cfg_;

    std::vector<LayerParams> build_schedule(float density) const;
    void ensure_seed(const ConstraintQAOAProblem& problem, std::vector<unsigned char>& seed) const;
    void apply_phase_separator(std::vector<unsigned char>& bits,
                               const ConstraintQAOAProblem& problem,
                               float gamma) const;
    void apply_constraint_mixer(std::vector<unsigned char>& bits,
                                const ConstraintQAOAProblem& problem,
                                int moves,
                                std::mt19937& rng) const;
    void analog_project(std::vector<unsigned char>& bits,
                        const ConstraintQAOAProblem& problem,
                        float tighten) const;
    float score_state(const std::vector<unsigned char>& bits,
                      const ConstraintQAOAProblem& problem,
                      float gamma) const;
};
