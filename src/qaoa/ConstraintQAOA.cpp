#include "qaoa/ConstraintQAOA.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>
#include <string>
#include <unordered_set>

namespace {

float sum_weights(const std::vector<float>& weights, const std::vector<unsigned char>& bits) {
    float total = 0.f;
    const int n = static_cast<int>(std::min(weights.size(), bits.size()));
    for (int i = 0; i < n; ++i) if (bits[i]) total += weights[i];
    return total;
}

float sum_weights(const std::vector<float>& weights) {
    float total = 0.f;
    for (float w : weights) total += w;
    return total;
}

} // namespace

ConstraintQAOAEngine::ConstraintQAOAEngine(const ConstraintQAOAConfig& cfg) : cfg_(cfg) {}

ConstraintQAOAResult ConstraintQAOAEngine::run(const ConstraintQAOAProblem& problem, unsigned seed) const {
    ConstraintQAOAResult result;
    if (problem.num_items <= 0) {
        return result;
    }

    const float weight_sum = sum_weights(problem.weights);
    const float density = (weight_sum <= 0.f) ? 0.f : std::min(1.f, problem.capacity / weight_sum);
    result.stats.warm_start_count = static_cast<int>(problem.warm_starts.size());
    auto schedule = build_schedule(density);
    result.stats.beta_schedule.reserve(schedule.size());
    result.stats.gamma_schedule.reserve(schedule.size());

    std::vector<std::vector<unsigned char>> population;
    population.reserve(std::max(cfg_.population_size, 1));
    for (auto seed_bits : problem.warm_starts) {
        ensure_seed(problem, seed_bits);
        analog_project(seed_bits, problem, cfg_.capacity_relaxation);
        population.push_back(std::move(seed_bits));
    }

    std::vector<unsigned char> greedy = problem.greedy_seed;
    ensure_seed(problem, greedy);
    if (!greedy.empty()) {
        analog_project(greedy, problem, cfg_.capacity_relaxation * 0.5f);
        population.push_back(greedy);
    }

    std::mt19937 rng(seed ? seed : static_cast<unsigned>(problem.num_items * 2654435761u + population.size()));

    while (population.size() < static_cast<size_t>(cfg_.population_size)) {
        std::vector<unsigned char> bits(problem.num_items, 0);
        std::bernoulli_distribution pick(std::max(0.05f, std::min(0.9f, density + problem.analog_bias * 0.25f)));
        for (int i = 0; i < problem.num_items; ++i) bits[i] = pick(rng) ? 1u : 0u;
        analog_project(bits, problem, cfg_.capacity_relaxation);
        population.push_back(std::move(bits));
    }

    struct State {
        std::vector<unsigned char> bits;
        float score;
    };

    std::vector<State> states;
    states.reserve(population.size());
    for (auto& cand : population) {
        float s = score_state(cand, problem, cfg_.penalty_base);
        states.push_back({std::move(cand), s});
    }

    for (const auto& layer : schedule) {
        result.stats.beta_schedule.push_back(layer.beta);
        result.stats.gamma_schedule.push_back(layer.gamma);
        std::vector<State> next;
        next.reserve(states.size() * 2);
        std::unordered_set<std::string> fingerprint;
        fingerprint.reserve(states.size() * 2 + 4);

        auto consider = [&](std::vector<unsigned char>&& candidate) {
            if (candidate.size() != static_cast<size_t>(problem.num_items)) candidate.resize(problem.num_items, 0);
            std::string key(candidate.begin(), candidate.end());
            if (!fingerprint.insert(key).second) return;
            float s = score_state(candidate, problem, cfg_.penalty_base * (1.f + layer.gamma));
            next.push_back({std::move(candidate), s});
        };

        for (auto state : states) {
            auto phased = state.bits;
            apply_phase_separator(phased, problem, layer.gamma);
            analog_project(phased, problem, cfg_.capacity_relaxation + cfg_.analog_drift * (1.f - density));
            consider(std::move(phased));

            int moves = std::max(1, static_cast<int>(std::round(layer.beta * cfg_.mixer_moves)));
            auto mixed = state.bits;
            apply_constraint_mixer(mixed, problem, moves, rng);
            analog_project(mixed, problem, cfg_.capacity_relaxation);
            consider(std::move(mixed));
            result.stats.mixer_moves += moves;
        }

        std::sort(next.begin(), next.end(), [](const State& a, const State& b){ return a.score > b.score; });
        if (next.size() > static_cast<size_t>(cfg_.population_size)) next.resize(cfg_.population_size);
        states = std::move(next);
        result.stats.analog_cycles++;
    }

    result.population.reserve(states.size());
    for (auto& st : states) {
        result.population.push_back(std::move(st.bits));
    }
    return result;
}

std::vector<ConstraintQAOAEngine::LayerParams> ConstraintQAOAEngine::build_schedule(float density) const {
    int layer_count = std::max(1, cfg_.layers);
    std::vector<LayerParams> schedule(layer_count);
    float beta_hi = cfg_.beta_min + (1.f - density) * (cfg_.beta_max - cfg_.beta_min);
    float beta_lo = cfg_.beta_min * (0.5f + 0.5f * density);
    float gamma_lo = cfg_.gamma_min * (0.7f + 0.3f * density);
    float gamma_hi = cfg_.gamma_max * (0.6f + 0.4f * density);
    for (int i = 0; i < layer_count; ++i) {
        float t = (layer_count == 1) ? 0.f : static_cast<float>(i) / static_cast<float>(layer_count - 1);
        schedule[i].beta = beta_hi - (beta_hi - beta_lo) * t;
        schedule[i].gamma = gamma_lo + (gamma_hi - gamma_lo) * t;
    }
    return schedule;
}

void ConstraintQAOAEngine::ensure_seed(const ConstraintQAOAProblem& problem, std::vector<unsigned char>& seed) const {
    if (static_cast<int>(seed.size()) != problem.num_items) {
        seed.resize(problem.num_items, 0);
    }
}

void ConstraintQAOAEngine::apply_phase_separator(std::vector<unsigned char>& bits,
                                                 const ConstraintQAOAProblem& problem,
                                                 float gamma) const {
    if (problem.num_items == 0) return;
    ensure_seed(problem, bits);
    float total = sum_weights(problem.weights, bits);

    struct Delta {
        double score;
        int idx;
    };
    std::vector<Delta> drops;
    std::vector<Delta> adds;
    drops.reserve(problem.num_items);
    adds.reserve(problem.num_items);
    for (int i = 0; i < problem.num_items; ++i) {
        double bias = static_cast<double>(problem.values[i]) - gamma * static_cast<double>(problem.weights[i]);
        if (bits[i]) {
            if (bias < 0.0) drops.push_back({bias, i});
        } else {
            if (bias > 0.0) adds.push_back({bias, i});
        }
    }
    std::sort(drops.begin(), drops.end(), [](const Delta& a, const Delta& b){ return a.score < b.score; });
    for (const auto& d : drops) {
        if (total <= problem.capacity) break;
        bits[d.idx] = 0;
        total -= problem.weights[d.idx];
    }

    std::sort(adds.begin(), adds.end(), [](const Delta& a, const Delta& b){ return a.score > b.score; });
    const float relaxed = problem.capacity * (1.f + cfg_.capacity_relaxation);
    for (const auto& d : adds) {
        float w = problem.weights[d.idx];
        if (total + w <= relaxed) {
            bits[d.idx] = 1;
            total += w;
        }
    }
}

void ConstraintQAOAEngine::apply_constraint_mixer(std::vector<unsigned char>& bits,
                                                  const ConstraintQAOAProblem& problem,
                                                  int moves,
                                                  std::mt19937& rng) const {
    if (problem.entangler_edges.empty() || moves <= 0) return;
    ensure_seed(problem, bits);
    float total = sum_weights(problem.weights, bits);
    std::uniform_int_distribution<size_t> edge_dist(0, problem.entangler_edges.size() - 1);
    const float relaxed = problem.capacity * (1.f + cfg_.capacity_relaxation);
    for (int step = 0; step < moves; ++step) {
        const auto& edge = problem.entangler_edges[edge_dist(rng)];
        int a = edge.first;
        int b = edge.second;
        if (a < 0 || b < 0 || a >= problem.num_items || b >= problem.num_items) continue;
        if (bits[a] == bits[b]) {
            if (bits[a] == 0) {
                float add_weight = problem.weights[a] + problem.weights[b];
                if (total + add_weight <= relaxed) {
                    bits[a] = bits[b] = 1u;
                    total += add_weight;
                }
            } else {
                // both 1 -> damp the weaker one
                if (problem.values[a] < problem.values[b]) {
                    bits[a] = 0;
                    total -= problem.weights[a];
                } else {
                    bits[b] = 0;
                    total -= problem.weights[b];
                }
            }
            continue;
        }
        if (bits[a] == 1) {
            float delta = problem.weights[b] - problem.weights[a];
            if (total + delta <= relaxed) {
                bits[a] = 0;
                bits[b] = 1;
                total += delta;
            }
        } else {
            float delta = problem.weights[a] - problem.weights[b];
            if (total + delta <= relaxed) {
                bits[a] = 1;
                bits[b] = 0;
                total += delta;
            }
        }
    }
}

void ConstraintQAOAEngine::analog_project(std::vector<unsigned char>& bits,
                                          const ConstraintQAOAProblem& problem,
                                          float tighten) const {
    if (problem.num_items == 0) return;
    ensure_seed(problem, bits);
    float total = sum_weights(problem.weights, bits);
    const float upper = problem.capacity * (1.f + tighten);
    if (total > upper) {
        std::vector<std::pair<double, int>> density;
        density.reserve(problem.num_items);
        for (int i = 0; i < problem.num_items; ++i) if (bits[i]) {
            double ratio = static_cast<double>(problem.values[i]) / std::max(1e-3, static_cast<double>(problem.weights[i]));
            density.emplace_back(ratio, i);
        }
        std::sort(density.begin(), density.end(), [](const auto& a, const auto& b){ return a.first < b.first; });
        for (const auto& entry : density) {
            if (total <= upper) break;
            bits[entry.second] = 0;
            total -= problem.weights[entry.second];
        }
    } else {
        const float lower = problem.capacity * std::max(0.f, 1.f - tighten - cfg_.analog_drift * 0.5f);
        if (total < lower) {
            std::vector<std::pair<double, int>> density;
            density.reserve(problem.num_items);
            for (int i = 0; i < problem.num_items; ++i) if (!bits[i]) {
                double ratio = static_cast<double>(problem.values[i]) / std::max(1e-3, static_cast<double>(problem.weights[i]));
                density.emplace_back(ratio, i);
            }
            std::sort(density.begin(), density.end(), [](const auto& a, const auto& b){ return a.first > b.first; });
            for (const auto& entry : density) {
                float w = problem.weights[entry.second];
                if (total + w <= upper) {
                    bits[entry.second] = 1;
                    total += w;
                }
                if (total >= lower) break;
            }
        }
    }
}

float ConstraintQAOAEngine::score_state(const std::vector<unsigned char>& bits,
                                        const ConstraintQAOAProblem& problem,
                                        float gamma) const {
    if (problem.num_items == 0) return 0.f;
    float obj = 0.f;
    float weight = 0.f;
    const int n = std::min(static_cast<int>(bits.size()), problem.num_items);
    for (int i = 0; i < n; ++i) {
        if (!bits[i]) continue;
        obj += problem.values[i];
        weight += problem.weights[i];
    }
    float overflow = std::max(0.f, weight - problem.capacity);
    return obj - gamma * overflow * overflow;
}
