#include "knapsack_c.h"
#include "InputModule.h"
#include "RecursiveSolver.h"
#include "Context.h"
#include "VanRouteWriter.h"
#include "RoutePlanner.h"
#include <cuda_runtime.h>
#include <iostream>
#include <sstream>
#include <cstring>

extern "C" {

KnapsackSolution* solve_knapsack(const char* csv_path, int target_team_size) {
    try {
        // Load villages (similar to main.cpp)
        auto villages = load_villages_from_csv(csv_path);
        int N = villages.size();

        // Allocate and populate CUDA arrays
        float *h_lat = new float[N];
        float *h_lon = new float[N];
        int *h_weights = new int[N];

        int total_available_workers = 0;
        for (int i = 0; i < N; ++i) {
            h_lat[i] = villages[i].latitude;
            h_lon[i] = villages[i].longitude;
            h_weights[i] = villages[i].workers;
            total_available_workers += villages[i].workers;
        }

        // CUDA memory allocation
        float *d_lat, *d_lon;
        int *d_weights;
        cudaMalloc(&d_lat, N * sizeof(float));
        cudaMalloc(&d_lon, N * sizeof(float));
        cudaMalloc(&d_weights, N * sizeof(int));

        cudaMemcpy(d_lat, h_lat, N * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_lon, h_lon, N * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_weights, h_weights, N * sizeof(int), cudaMemcpyHostToDevice);

        // Call your actual solver
        std::vector<VanTrip> trips = solveVanRoutes(villages, target_team_size);

        // Convert to C struct
        KnapsackSolution* solution = new KnapsackSolution;
        solution->num_trips = trips.size();
        solution->trips = new VanTripResult[trips.size()];
        solution->total_crew = 0;
        solution->total_fuel_cost = 0.0;

        for (size_t i = 0; i < trips.size(); ++i) {
            const auto& trip = trips[i];
            
            solution->trips[i].van_id = i + 1;
            solution->trips[i].distance = trip.distance;
            solution->trips[i].fuel_cost = trip.fuelCost;
            solution->trips[i].crew_size = trip.crewSize;
            
            // Build village names string
            std::stringstream ss;
            for (size_t j = 0; j < trip.villageIndices.size(); ++j) {
                if (j > 0) ss << ",";
                ss << villages[trip.villageIndices[j]].name;
            }
            std::string village_str = ss.str();
            solution->trips[i].village_names = new char[village_str.length() + 1];
            strcpy(solution->trips[i].village_names, village_str.c_str());
            
            solution->total_crew += trip.crewSize;
            solution->total_fuel_cost += trip.fuelCost;
        }

        solution->shortfall = std::max(0, target_team_size - solution->total_crew);

        // Cleanup CUDA
        cudaFree(d_lat);
        cudaFree(d_lon);
        cudaFree(d_weights);
        delete[] h_lat;
        delete[] h_lon;
        delete[] h_weights;

        return solution;
    }
    catch (...) {
        return nullptr;
    }
}

void free_knapsack_solution(KnapsackSolution* solution) {
    if (solution) {
        if (solution->trips) {
            for (int i = 0; i < solution->num_trips; ++i) {
                delete[] solution->trips[i].village_names;
            }
            delete[] solution->trips;
        }
        delete solution;
    }
}

}