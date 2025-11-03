#include "InputModule.h"
#include "RoutePlanner.h"
#include <iostream>
#include <numeric>
#include <cstdlib>
#include <iomanip>
#include <fstream>

int main(int argc, char *argv[])
{
    auto entities = load_entities_from_csv("../data/villages_50.csv");
    std::cout << "Loaded entities: " << entities.size() << std::endl;
    int N = entities.size();

    float *h_lat = new float[N];
    float *h_lon = new float[N];
    int *h_weights = new int[N];

    int total_available_units = 0;
    for (int i = 0; i < N; ++i)
    {
        h_lat[i] = entities[i].latitude;
        h_lon[i] = entities[i].longitude;
        h_weights[i] = entities[i].resourceUnits;
        total_available_units += entities[i].resourceUnits;
    }

    int target_team_size = (argc > 1) ? std::atoi(argv[1]) : total_available_units;
    std::string output_file = (argc > 2) ? argv[2] : "routes.csv";
    std::cout << "🎯 Target team size: " << target_team_size << std::endl;
    std::cout << "📄 Output file: " << output_file << std::endl;

    // Main recursive grouping planner
    std::vector<GroupResult> trips = solveGroups(entities, target_team_size);

    // Write results to specified output file
    std::ofstream out(output_file);
    out << "ID,Items,Distance (km),Cost\n";
    double totalFuel = 0.0;
    int totalCrew = 0;

    for (size_t i = 0; i < trips.size(); ++i)
    {
        const auto &trip = trips[i];
        out << (i + 1) << ",";
        for (int vidx : trip.itemIndices)
            out << entities[vidx].name << " ";
        out << "," << std::fixed << std::setprecision(4) << trip.metric << ",";
        out << std::setprecision(5) << trip.cost << "\n";

    totalFuel += trip.cost;
    totalCrew += trip.resourceUnits;
    }

    out.close();

    // Summary
    std::cout << "✅ Total units selected: " << totalCrew << std::endl;
    std::cout << "📉 Shortfall: " << std::max(0, target_team_size - totalCrew) << std::endl;
    std::cout << "⛽ Total fuel cost: $" << std::fixed << std::setprecision(2) << totalFuel << std::endl;

    // Clean up
    delete[] h_lat;
    delete[] h_lon;
    delete[] h_weights;

    return 0;
}
