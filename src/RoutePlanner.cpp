#include "InputModule.h"
#include "Constants.h"
#include "RecursiveSolver.h"
#include <cuda_runtime.h>
#include <fstream>
#include <cmath>
#include <iomanip>
#include <iostream>

struct VanTrip {
    std::vector<int> villageIndices;
    int crewSize = 0;
    double distance = 0.0;
    double fuelCost = 0.0;
};

std::vector<VanTrip> solveVanRoutes(std::vector<Village>& villages, int targetTeamSize) {
    std::vector<VanTrip> trips;
    int totalPickedUp = 0;

    std::vector<int> workersRemaining(villages.size());
    for (size_t i = 0; i < villages.size(); ++i) {
        workersRemaining[i] = villages[i].workers;
        if (villages[i].productivityScore <= 0 || villages[i].productivityScore > 4) {
            villages[i].productivityScore = 1 + (std::rand() % 4);  // Assign if missing
        }
    }

    int vanId = 1;
    int maxVanTrips = MAX_VANS * 5;

    while (totalPickedUp < targetTeamSize && vanId <= maxVanTrips) {
        std::vector<int> blockIndices;
        for (size_t i = 0; i < workersRemaining.size(); ++i) {
            if (workersRemaining[i] > 0)
                blockIndices.push_back(i);
            if (blockIndices.size() >= 15) break;
        }

        if (blockIndices.empty()) break;

        std::vector<float> lat(blockIndices.size()), lon(blockIndices.size());
        std::vector<int> weights(blockIndices.size());
        for (size_t i = 0; i < blockIndices.size(); ++i) {
            lat[i] = villages[blockIndices[i]].latitude;
            lon[i] = villages[blockIndices[i]].longitude;
            weights[i] = workersRemaining[blockIndices[i]];
        }

        float* d_lat;
        float* d_lon;
        int* d_weights;
        cudaMalloc(&d_lat, blockIndices.size() * sizeof(float));
        cudaMalloc(&d_lon, blockIndices.size() * sizeof(float));
        cudaMalloc(&d_weights, blockIndices.size() * sizeof(int));

        cudaMemcpy(d_lat, lat.data(), blockIndices.size() * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_lon, lon.data(), blockIndices.size() * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_weights, weights.data(), blockIndices.size() * sizeof(int), cudaMemcpyHostToDevice);

        Context ctx;
        ctx.village_indices = blockIndices;
        ctx.d_lat = d_lat;
        ctx.d_lon = d_lon;
        ctx.d_weights = d_weights;
        ctx.N = blockIndices.size();
        ctx.field_lat = FIELD_LAT;
        ctx.field_lon = FIELD_LON;

        auto plan = recursive_worker(ctx, 0, MAX_WORKERS_PER_VAN, villages);

        cudaFree(d_lat);
        cudaFree(d_lon);
        cudaFree(d_weights);

        if (plan.empty()) {
            std::cout << "⚠️ No plan returned for van trip " << vanId << "\n";
            break;
        }

        VanTrip trip;
        double distance = 0.0;
        int crew = 0;
        double prev_lat = GARAGE_LAT;
        double prev_lon = GARAGE_LON;

        for (size_t i = 0; i < plan.size(); ++i) {
            if (!plan[i]) continue;
            int villageIdx = blockIndices[i];

            crew += workersRemaining[villageIdx];
            trip.villageIndices.push_back(villageIdx);

            double d = haversine(prev_lat, prev_lon,
                                 villages[villageIdx].latitude, villages[villageIdx].longitude);
            distance += d;
            prev_lat = villages[villageIdx].latitude;
            prev_lon = villages[villageIdx].longitude;

            workersRemaining[villageIdx] = 0;
        }

        if (crew == 0) {
            std::cout << "⚠️ No crew picked up on van trip " << vanId << "\n";
            continue;
        }

        distance += haversine(prev_lat, prev_lon, FIELD_LAT, FIELD_LON);
        distance += haversine(FIELD_LAT, FIELD_LON, GARAGE_LAT, GARAGE_LON);

        trip.crewSize = crew;
        trip.distance = distance;
        trip.fuelCost = distance * (GAS_PRICE_PER_LITER / KM_PER_LITER);

        std::cout << "🚐 Van " << vanId << " trip: " << crew << " workers, "
                  << std::fixed << std::setprecision(2) << trip.distance << " km\n";

        trips.push_back(trip);
        totalPickedUp += crew;
        vanId++;
    }

    return trips;
}
