#include "RecursiveSolver.h"
#include "Constants.h"
#include <vector>
#include <algorithm>
#include <cstring>
#include <ctime>
#include <iostream>
#include <numeric>
#include <cmath>
#include <fstream>
#include <random>
#include "InputModule.h"
#include "RouteUtils.h" // for shared haversine

// Metal API (only available on Apple)
#ifdef __APPLE__
#include "metal_api.h"
#endif

namespace {
// Initialize Metal evaluator once per process, reading shader source at runtime.
bool g_metal_ready = false;
void ensure_metal_ready() {
#ifdef __APPLE__
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

#ifdef __APPLE__
    ensure_metal_ready();
    if (g_metal_ready) {
        // Build candidates: start with greedy, then random variations
        const int N = ctx.N;
        const int NUM_CAND = std::min(256, std::max(8, N * 8));
        std::vector<std::vector<unsigned char>> cand_bits;
        cand_bits.reserve(NUM_CAND);

        // helper to clamp by capacity (optional)
        auto clamp_capacity = [&](std::vector<unsigned char>& bits){
            int sum = 0;
            for (int i = 0; i < N; ++i) if (bits[i]) sum += std::max(1, entities[ctx.item_indices[i]].resourceUnits);
            if (sum <= target_team_size) return;
            // drop lowest value/weight until within capacity
            std::vector<std::pair<double,int>> score_idx;
            score_idx.reserve(N);
            for (int i = 0; i < N; ++i) if (bits[i]) {
                const auto& v = entities[ctx.item_indices[i]];
                double value = std::max(1, v.resourceUnits) * std::max(1, v.priority);
                double key = value / std::max(1, v.resourceUnits);
                score_idx.emplace_back(key, i);
            }
            std::sort(score_idx.begin(), score_idx.end()); // remove lowest value density first
            for (auto& kv : score_idx) {
                if (sum <= target_team_size) break;
                int i = kv.second;
                bits[i] = 0;
                sum -= std::max(1, entities[ctx.item_indices[i]].resourceUnits);
            }
        };

        // 1) Greedy baseline
        cand_bits.push_back(std::vector<unsigned char>(N, 0));
        for (int i = 0; i < N; ++i) cand_bits[0][i] = greedy_plan[i] ? 1 : 0;

        // 2) Single-bit improvements around greedy
        for (int i = 0; i < N && static_cast<int>(cand_bits.size()) < NUM_CAND/2; ++i) {
            auto v = cand_bits[0];
            v[i] = v[i] ? 0u : 1u; // flip
            clamp_capacity(v);
            cand_bits.push_back(std::move(v));
        }

        // 3) Random candidates
        std::mt19937 rng(static_cast<unsigned>(time(nullptr)));
        std::bernoulli_distribution pick(0.4);
        while (static_cast<int>(cand_bits.size()) < NUM_CAND) {
            std::vector<unsigned char> v(N, 0);
            for (int i = 0; i < N; ++i) v[i] = pick(rng) ? 1u : 0u;
            clamp_capacity(v);
            cand_bits.push_back(std::move(v));
        }

        // Prepare evaluator inputs
    std::vector<float> values(N, 0.f), weights(N, 0.f), caps(1, static_cast<float>(target_team_size));
        for (int i = 0; i < N; ++i) {
            const auto& v = entities[ctx.item_indices[i]];
            weights[i] = static_cast<float>(std::max(1, v.resourceUnits));
            values[i] = weights[i] * static_cast<float>(std::max(1, v.priority));
        }

        const auto packed = pack2bit(cand_bits, N);
        const int bytes_per = (N + 3) / 4;
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
            std::vector<int> best_plan(N, 0);
            for (int i = 0; i < N; ++i) best_plan[i] = cand_bits[best_idx][i] ? 1 : 0;
#ifdef DEBUG_VAN
            std::cout << "🔎 Best plan (Metal eval): ";
            for (int bit : best_plan) std::cout << bit;
            std::cout << "  score=" << (obj[best_idx]-pen[best_idx]) << " obj=" << obj[best_idx] << " pen=" << pen[best_idx] << "\n";
#endif
            return best_plan;
        } else {
            std::cerr << "Metal eval failed: " << err << "; falling back to CPU heuristic.\n";
        }
    }
#endif

    // CPU heuristic fallback (also used on non-Apple platforms)
    std::vector<int> best_plan(ctx.N, 0);
    for (int i = 0; i < ctx.N; ++i) best_plan[i] = greedy_plan[i];

#ifdef DEBUG_VAN
    std::cout << "🔎 Best plan (CPU heuristic): ";
    for (int bit : best_plan) std::cout << bit;
    std::cout << "\n";
#endif

    return best_plan;
}
