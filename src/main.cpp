#include "InputModule.h"
#include "RecursiveSolver.h"
#include "Context.h"
#include "VanRouteWriter.h"
#include "RoutePlanner.h"
#include <cuda_runtime.h>
#include <iostream>
#include <numeric>
#include <cstdlib>
#include <iomanip>
#include <fstream>

int main(int argc, char *argv[])
{
    auto villages = load_villages_from_csv("../data/villages.csv");
    std::cout << "Loaded villages: " << villages.size() << std::endl;
    int N = villages.size();

    float *h_lat = new float[N];
    float *h_lon = new float[N];
    int *h_weights = new int[N];

    int total_available_workers = 0;
    for (int i = 0; i < N; ++i)
    {
        h_lat[i] = villages[i].latitude;
        h_lon[i] = villages[i].longitude;
        h_weights[i] = villages[i].workers;
        total_available_workers += villages[i].workers;
    }

    int target_team_size = (argc > 1) ? std::atoi(argv[1]) : total_available_workers;
    std::cout << "🎯 Target team size: " << target_team_size << std::endl;

    float *d_lat, *d_lon;
    int *d_weights;
    cudaMalloc(&d_lat, N * sizeof(float));
    cudaMalloc(&d_lon, N * sizeof(float));
    cudaMalloc(&d_weights, N * sizeof(int));

    cudaMemcpy(d_lat, h_lat, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_lon, h_lon, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weights, h_weights, N * sizeof(int), cudaMemcpyHostToDevice);

    // Main recursive trip planner
    std::vector<VanTrip> trips = solveVanRoutes(villages, target_team_size);

    // Write van_routes.csv
    std::ofstream out("van_routes.csv");
    out << "Van ID,Villages,Distance (km),Fuel Cost (USD)\n";
    double totalFuel = 0.0;
    int totalCrew = 0;

    for (size_t i = 0; i < trips.size(); ++i)
    {
        const auto &trip = trips[i];
        out << (i + 1) << ",";
        for (int vidx : trip.villageIndices)
            out << villages[vidx].name << " ";
        out << "," << std::fixed << std::setprecision(4) << trip.distance << ",";
        out << std::setprecision(5) << trip.fuelCost << "\n";

        totalFuel += trip.fuelCost;
        totalCrew += trip.crewSize;
    }

    out.close();

    // Summary
    std::cout << "✅ Total workers picked up: " << totalCrew << std::endl;
    std::cout << "📉 Shortfall: " << std::max(0, target_team_size - totalCrew) << std::endl;
    std::cout << "⛽ Total fuel cost: $" << std::fixed << std::setprecision(2) << totalFuel << std::endl;

    // Clean up
    cudaFree(d_lat);
    cudaFree(d_lon);
    cudaFree(d_weights);
    delete[] h_lat;
    delete[] h_lon;
    delete[] h_weights;

    return 0;
}
