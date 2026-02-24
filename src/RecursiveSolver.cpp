#include "RecursiveSolver.h"
#include "Constants.h"
#include "qaoa/ConstraintQAOA.h"
#include <vector>
#include <algorithm>
#include <cstring>
#include <iostream>
#include <limits>
#include <numeric>
#include <cmath>
#include <fstream>
#include <random>
#include "InputModule.h"
#include "RouteUtils.h" // for shared haversine

// Metal API (only available on Apple with Metal enabled)
#if defined(__APPLE__) && defined(KNAPSACK_METAL_SUPPORT)
#include "metal_api.h"
#endif

namespace {
// Initialize Metal evaluator once per process, reading shader source at runtime.
bool g_metal_ready = false;
void ensure_metal_ready() {
#if defined(__APPLE__) && defined(KNAPSACK_METAL_SUPPORT)
    if (g_metal_ready) return;
    const char* candidate_paths[] = {
        "../kernels/metal/shaders/eval_block_candidates.metal",
        "kernels/metal/shaders/eval_block_candidates.metal",
        "../../kernels/metal/shaders/eval_block_candidates.metal"
    };
    std::string src;
    for (const char* p : candidate_paths) {
        std::ifstream in(p, std::ios::binary);
        if (!in) continue;
        src.assign((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
        if (!src.empty()) break;
    }
    char err[512] = {0};
    if (!src.empty()) {
        int rc = knapsack_metal_init_from_source(src.data(), src.size(), err, sizeof(err));
        if (rc == 0) {
            g_metal_ready = true;
        } else {
            std::cerr << "Metal init failed: " << err << "\n";
        }
    } else {
        std::cerr << "Metal shader source not found; continuing with CPU heuristic.\n";
    }
#endif
}

// Pack candidates (2-bit lanes per item). lane: 0=unselected; 1=group0
std::vector<unsigned char> pack2bit(const std::vector<std::vector<unsigned char>>& cand_bits, int num_items) {
    const int bytes_per = (num_items + 3) / 4; // 4 items per byte (2 bits each)
    std::vector<unsigned char> out;
    out.assign(cand_bits.size() * bytes_per, 0);
    for (size_t c = 0; c < cand_bits.size(); ++c) {
        for (int i = 0; i < num_items; ++i) {
            const unsigned char lane = cand_bits[c][i] ? 1u : 0u;
            const int byteIdx = static_cast<int>(c) * bytes_per + (i >> 2);
            const int shift = (i & 3) * 2;
            out[byteIdx] |= static_cast<unsigned char>((lane & 0x3u) << shift);
        }
    }
    return out;
}

// Select best candidate index by maximizing obj - pen
int argmax_score(const std::vector<float>& obj, const std::vector<float>& pen) {
    int best = 0;
    float bestScore = obj.empty() ? -1e30f : (obj[0] - pen[0]);
    for (int i = 1; i < static_cast<int>(obj.size()); ++i) {
        const float s = obj[i] - pen[i];
        if (s > bestScore) { bestScore = s; best = i; }
    }
    return best;
}

std::vector<std::pair<int,int>> build_entangler_edges(const Context& ctx,
                                                      const std::vector<Entity>& entities,
                                                      int neighbors_per_node) {
    std::vector<std::pair<int,int>> edges;
    const int N = ctx.N;
    if (N <= 1) return edges;
    neighbors_per_node = std::max(1, neighbors_per_node);
    std::vector<std::pair<double,int>> dist;
    dist.reserve(std::max(1, N - 1));
    for (int i = 0; i < N; ++i) {
        dist.clear();
        const auto& src = entities[ctx.item_indices[i]];
        for (int j = 0; j < N; ++j) {
            if (i == j) continue;
            const auto& dst = entities[ctx.item_indices[j]];
            double d = haversine(src.latitude, src.longitude, dst.latitude, dst.longitude);
            dist.emplace_back(d, j);
        }
        std::sort(dist.begin(), dist.end(), [](const auto& a, const auto& b){ return a.first < b.first; });
        const int take = std::min(neighbors_per_node, static_cast<int>(dist.size()));
        for (int k = 0; k < take; ++k) {
            edges.emplace_back(i, dist[k].second);
        }
    }
    if (N > 2) {
        std::mt19937 rng(static_cast<unsigned>(N * 211u + neighbors_per_node * 17u));
        std::uniform_int_distribution<int> pick(0, N - 1);
        const int extras = std::max(1, N / 2);
        for (int e = 0; e < extras; ++e) {
            int a = pick(rng);
            int b = pick(rng);
            if (a == b) continue;
            edges.emplace_back(a, b);
        }
    }
    return edges;
}

std::vector<unsigned char> ints_to_bits(const std::vector<int>& src, int length) {
    std::vector<unsigned char> bits(std::max(0, length), 0u);
    const int limit = std::min(length, static_cast<int>(src.size()));
    for (int i = 0; i < limit; ++i) bits[i] = src[i] ? 1u : 0u;
    return bits;
}

std::vector<int> bits_to_plan(const std::vector<unsigned char>& bits) {
    std::vector<int> plan(bits.size(), 0);
    for (size_t i = 0; i < bits.size(); ++i) plan[i] = bits[i] ? 1 : 0;
    return plan;
}

void evaluate_candidates_cpu(const std::vector<std::vector<unsigned char>>& cand_bits,
                             const std::vector<float>& values,
                             const std::vector<float>& weights,
                             float capacity,
                             std::vector<float>* obj,
                             std::vector<float>* pen) {
    const float penalty_coeff = 100.0f;
    const size_t count = cand_bits.size();
    obj->assign(count, 0.f);
    pen->assign(count, 0.f);
    for (size_t c = 0; c < count; ++c) {
        float total_value = 0.f;
        float total_weight = 0.f;
        const auto& bits = cand_bits[c];
        const size_t limit = std::min(bits.size(), values.size());
        for (size_t i = 0; i < limit; ++i) {
            if (!bits[i]) continue;
            total_value += values[i];
            total_weight += weights[i];
        }
        (*obj)[c] = total_value;
        const float overflow = std::max(0.f, total_weight - capacity);
        (*pen)[c] = penalty_coeff * overflow * overflow;
    }
}
}

std::vector<int> recursive_worker(Context ctx, int depth, int target_team_size, const std::vector<Entity>& entities) {
    if (ctx.N == 0 || depth >= MAX_RECURSION_DEPTH) {
#ifdef DEBUG_VAN
        std::cout << "⚠️ Empty or max-depth block encountered at depth " << depth << ". Returning no plan.\n";
#endif
        return {};
    }

#ifdef DEBUG_VAN
    std::cout << "🧠 Classical block: " << ctx.N << " entities in scope\n";
    for (int i = 0; i < ctx.N; ++i) {
        int globalIdx = ctx.item_indices[i];
        const auto& v = entities[globalIdx];
        double d = haversine(GARAGE_LAT, GARAGE_LON, v.latitude, v.longitude);
        int w = std::max(1, v.resourceUnits);
        int s = std::max(1, v.priority);
        double cost = std::sqrt(d) / std::pow(w * s, 0.75);
        std::cout << "📍 Entity: " << v.name
                  << ", d=" << std::fixed << std::setprecision(2) << d
                  << ", w=" << w
                  << ", s=" << s
                  << ", cost=" << std::setprecision(3) << cost << "\n";
    }
#endif

    // Compute local item features (weights, productivity) and a greedy baseline
    struct Item {
        int localIdx;
        int globalIdx;
        int weight;
        int productivity;
        double cost;
    };

    std::vector<Item> items;
    items.reserve(ctx.N);
    for (int i = 0; i < ctx.N; ++i) {
        int globalIdx = ctx.item_indices[i];
        const auto& v = entities[globalIdx];
        int w = std::max(1, v.resourceUnits);
        int s = std::max(1, v.priority);
        double d = haversine(GARAGE_LAT, GARAGE_LON, v.latitude, v.longitude);
        double cost = std::sqrt(d) / std::pow(static_cast<double>(w) * s, 0.75);
        items.push_back({i, globalIdx, w, s, cost});
    }

    std::sort(items.begin(), items.end(), [](const Item& a, const Item& b){
        if (a.cost == b.cost) return a.weight > b.weight; // prefer more workers when tie
        return a.cost < b.cost;
    });

    std::vector<int> greedy_plan(ctx.N, 0);
    int crew = 0;
    for (const auto& it : items) {
        if (crew + it.weight <= std::max(1, target_team_size)) {
            greedy_plan[it.localIdx] = 1;
            crew += it.weight;
            if (crew >= target_team_size) break;
        }
    }

    const int N = ctx.N;
    const int NUM_CAND = std::min(256, std::max(8, N * 8));
    std::vector<float> weights(N, 0.f), values(N, 0.f);
    float block_weight_sum = 0.f;
    for (int i = 0; i < N; ++i) {
        const auto& v = entities[ctx.item_indices[i]];
        weights[i] = static_cast<float>(std::max(1, v.resourceUnits));
        values[i] = weights[i] * static_cast<float>(std::max(1, v.priority));
        block_weight_sum += weights[i];
    }

    ConstraintQAOAProblem qprob;
    qprob.num_items = N;
    qprob.weights = weights;
    qprob.values = values;
    qprob.capacity = static_cast<float>(target_team_size);
    qprob.entangler_edges = build_entangler_edges(ctx, entities, std::max(2, std::min(5, N / 2)));
    qprob.greedy_seed = ints_to_bits(greedy_plan, N);
    qprob.analog_bias = (block_weight_sum <= 1e-3f) ? 0.f : std::min(1.f, static_cast<float>(target_team_size) / block_weight_sum);
    for (const auto& warm : ctx.warm_population) {
        qprob.warm_starts.push_back(ints_to_bits(warm, N));
    }
    if (qprob.warm_starts.empty() && !ctx.warm_seed.empty()) {
        qprob.warm_starts.push_back(ints_to_bits(ctx.warm_seed, N));
    }

    ConstraintQAOAConfig qcfg;
    qcfg.population_size = NUM_CAND;
    qcfg.layers = std::clamp(N / 5, 2, 5);
    qcfg.mixer_moves = std::max(3, N / 2);
    qcfg.analog_passes = 2;
    ConstraintQAOAEngine qengine(qcfg);
    auto qaoa_result = qengine.run(qprob);
    std::vector<std::vector<unsigned char>> cand_bits = qaoa_result.population;
    if (cand_bits.empty()) cand_bits.push_back(qprob.greedy_seed);

#ifdef DEBUG_VAN
    std::cout << "🎛️ Constraint-QAOA layers=" << qcfg.layers
              << " candidates=" << cand_bits.size()
              << " warm=" << ctx.warm_population.size() << "\n";
#endif

#if defined(__APPLE__) && defined(KNAPSACK_METAL_SUPPORT)
    ensure_metal_ready();
    if (g_metal_ready && !cand_bits.empty()) {
        std::vector<float> caps(1, static_cast<float>(target_team_size));
        const auto packed = pack2bit(cand_bits, N);
        const int num_cand = static_cast<int>(cand_bits.size());
        std::vector<float> obj(num_cand, 0.f), pen(num_cand, 0.f);
        MetalEvalIn in{
            /*candidates*/ reinterpret_cast<const unsigned char*>(packed.data()),
            /*num_items*/ N,
            /*num_candidates*/ num_cand,
            /*item_values*/ values.data(),
            /*item_weights*/ weights.data(),
            /*group_capacities*/ caps.data(),
            /*num_groups*/ 1,
            /*penalty_coeff*/ 100.0f,
            /*penalty_power*/ 1.0f
        };
        MetalEvalOut out{ obj.data(), pen.data() };
        char err[256] = {0};
        int rc = knapsack_metal_eval(&in, &out, err, sizeof(err));
        if (rc == 0) {
            int best_idx = argmax_score(obj, pen);
            std::vector<int> best_plan = bits_to_plan(cand_bits[best_idx]);
#ifdef DEBUG_VAN
            std::cout << "🔎 Best plan (Metal eval): ";
            for (int bit : best_plan) std::cout << bit;
            std::cout << "  score=" << (obj[best_idx]-pen[best_idx]) << " obj=" << obj[best_idx] << " pen=" << pen[best_idx] << "\n";
#endif
            return best_plan;
        } else {
            std::cerr << "Metal eval failed: " << err << "; falling back to CPU scoring.\n";
        }
    }
#endif

    if (!cand_bits.empty()) {
        std::vector<float> obj, pen;
        evaluate_candidates_cpu(cand_bits, values, weights, static_cast<float>(target_team_size), &obj, &pen);
        int best_idx = argmax_score(obj, pen);
        std::vector<int> best_plan = bits_to_plan(cand_bits[best_idx]);
#ifdef DEBUG_VAN
        std::cout << "🔎 Best plan (CPU scoring): ";
        for (int bit : best_plan) std::cout << bit;
        std::cout << " score=" << (obj[best_idx]-pen[best_idx]) << "\n";
#endif
        return best_plan;
    }

    return greedy_plan;
}
