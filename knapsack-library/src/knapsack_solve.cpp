// C API
#include "knapsack_c.h"

// CPU helpers
#include "InputModule.h"
#include "RouteUtils.h"
#include "Constants.h"

#include <algorithm>
#include <sstream>
#include <cstring>
#include <vector>
#include <stdexcept>
#include <random>

// Metal C API (only available on Apple with Metal support enabled)
#if defined(__APPLE__) && !defined(KNAPSACK_CPU_ONLY)
#include "metal_api.h"
#endif

extern "C" {

// Minimal CPU fallback: greedy accumulate entities into trips until target is met.
KnapsackSolution* solve_knapsack(const char* csv_path, int target_team_size) {
    try {
        if (!csv_path) return nullptr;
        std::vector<Entity> entities = load_entities_from_csv(csv_path);
        if (entities.empty()) return nullptr;

        // Greedy: visit items in file order, batch into trips until target is met.
    int remaining = std::max(0, target_team_size);
    std::vector<int> picked(entities.size(), 0);

        std::vector<std::vector<int>> trips;
    int cursor = 0;

        // Minimal CPU fallback: greedy accumulate items into groups until target is met.
        std::mt19937 rng(12345);
        std::uniform_int_distribution<int> bit(0, 1);

        while (remaining > 0 && cursor < (int)entities.size()) {
            // Form a block of up to 15 unpicked entities (similar to RoutePlanner)
            std::vector<int> blockIdx;
            for (int i = cursor; i < (int)entities.size() && (int)blockIdx.size() < 15; ++i) {
                if (!picked[i] && entities[i].resourceUnits > 0) blockIdx.push_back(i);
            }
            if (blockIdx.empty()) break;

            const int num_items = (int)blockIdx.size();
            const int num_groups = 1; // one group per trip
            const int num_candidates = 64;
            const int bytes_per_cand = (num_items + 3) / 4; // 2 bits per item

            // Prepare SoA inputs for evaluator
            std::vector<float> values(num_items, 1.0f);
            std::vector<float> weights(num_items, 0.0f);
            for (int i = 0; i < num_items; ++i) {
                const auto& v = entities[blockIdx[i]];
                values[i] = std::max(1, v.priority);
                weights[i] = std::max(0, v.resourceUnits);
            }
            float group_caps_arr[1] = { (float)MAX_UNITS_PER_GROUP };

            // Generate random candidates; lane 1 means assign to group 0, lane 0 unassigned
            std::vector<unsigned char> cand(num_candidates * bytes_per_cand, 0);
            auto set_lane = [&](int c, int item, unsigned lane){
                const int byteIdx = c * bytes_per_cand + (item >> 2);
                const int shift = (item & 3) * 2;
                unsigned char mask = (unsigned char)(0x3u << shift);
                cand[byteIdx] = (cand[byteIdx] & ~mask) | (unsigned char)((lane & 0x3u) << shift);
            };
            for (int c = 0; c < num_candidates; ++c) {
                int approxCrew = 0;
                for (int i = 0; i < num_items; ++i) {
                    unsigned lane = bit(rng) ? 1u : 0u; // 50% choose
                    // Heuristic: avoid obvious overfill when already near cap
                    if (lane == 1u && approxCrew + (int)weights[i] > MAX_UNITS_PER_GROUP) lane = 0u;
                    if (lane == 1u) approxCrew += (int)weights[i];
                    set_lane(c, i, lane);
                }
            }

            // Evaluate with Metal if available, else skip to CPU greedy
            std::vector<float> obj(num_candidates, 0.0f), pen(num_candidates, 0.0f);
            bool used_metal = false;
#if defined(__APPLE__) && !defined(KNAPSACK_CPU_ONLY)
            MetalEvalIn in{};
            in.candidates = cand.data();
            in.num_items = num_items;
            in.num_candidates = num_candidates;
            in.item_values = values.data();
            in.item_weights = weights.data();
            in.group_capacities = group_caps_arr;
            in.num_groups = num_groups;
            in.penalty_coeff = 1.0f;
            in.penalty_power = 1.0f;
            MetalEvalOut out{ obj.data(), pen.data() };

            // If Metal was initialized earlier, this should succeed.
            if (knapsack_metal_eval(&in, &out, nullptr, 0) == 0) {
                used_metal = true;
            }
#endif

            std::vector<int> trip;
            int crew = 0;
            if (used_metal) {
                // Pick best candidate by (obj - pen)
                int best = 0;
                float bestScore = obj[0] - pen[0];
                for (int c = 1; c < num_candidates; ++c) {
                    float s = obj[c] - pen[c];
                    if (s > bestScore) { bestScore = s; best = c; }
                }
                // Decode best candidate to trip indices
                for (int i = 0; i < num_items; ++i) {
                    const int byteIdx = best * bytes_per_cand + (i >> 2);
                    const int shift = (i & 3) * 2;
                    unsigned lane = (cand[byteIdx] >> shift) & 0x3u;
                    if (lane == 1u) {
                        trip.push_back(blockIdx[i]);
                        crew += std::max(0, entities[blockIdx[i]].resourceUnits);
                    }
                }
            } else {
                // CPU fallback: simple greedy until capacity
                for (int i = 0; i < num_items && crew < MAX_UNITS_PER_GROUP; ++i) {
                    trip.push_back(blockIdx[i]);
                    crew += std::max(0, entities[blockIdx[i]].resourceUnits);
                }
            }

            if (trip.empty()) break;
            for (int idx : trip) picked[idx] = 1;
            trips.push_back(trip);
            remaining -= crew;

            // Advance cursor beyond picked
            while (cursor < (int)entities.size() && picked[cursor]) cursor++;
        }

    // Try to initialize Metal on Apple Silicon; if available, we'll use it in later passes.
    // This is a no-op on non-Apple builds.
    extern int knapsack_metal_init_default();
#if defined(__APPLE__) && (defined(__aarch64__) || defined(__arm64__))
    (void)knapsack_metal_init_default();
#endif

    // Convert to C struct
        KnapsackSolution* solution = new KnapsackSolution;
        solution->num_trips = (int)trips.size();
        solution->trips = new GroupTrip[trips.size()];
        solution->total_units = 0;
        solution->total_cost = 0.0;

        for (size_t ti = 0; ti < trips.size(); ++ti) {
            const auto& trip = trips[ti];

            double distance = 0.0;
            int crew = 0;
            double prev_lat = GARAGE_LAT;
            double prev_lon = GARAGE_LON;

            std::stringstream ss;
            for (size_t j = 0; j < trip.size(); ++j) {
                const auto& v = entities[trip[j]];
                if (j > 0) ss << ",";
                ss << v.name;
                distance += haversine(prev_lat, prev_lon, v.latitude, v.longitude);
                prev_lat = v.latitude;
                prev_lon = v.longitude;
                crew += std::max(0, v.resourceUnits);
            }
            distance += haversine(prev_lat, prev_lon, FIELD_LAT, FIELD_LON);
            distance += haversine(FIELD_LAT, FIELD_LON, GARAGE_LAT, GARAGE_LON);

            solution->trips[ti].group_id = (int)ti + 1;
            std::string names = ss.str();
            solution->trips[ti].item_names = new char[names.size() + 1];
            std::strcpy(solution->trips[ti].item_names, names.c_str());
            solution->trips[ti].distance = distance;
            solution->trips[ti].cost = distance * (GAS_PRICE_PER_LITER / KM_PER_LITER);
            solution->trips[ti].units = crew;

            solution->total_units += crew;
            solution->total_cost += solution->trips[ti].cost;
        }

        solution->shortfall = std::max(0, target_team_size - solution->total_units);
        return solution;
    } catch (...) {
        return nullptr;
    }
}

void free_knapsack_solution(KnapsackSolution* solution) {
    if (!solution) return;
    if (solution->trips) {
        for (int i = 0; i < solution->num_trips; ++i) {
            delete[] solution->trips[i].item_names;
        }
        delete[] solution->trips;
    }
    delete solution;
}

}