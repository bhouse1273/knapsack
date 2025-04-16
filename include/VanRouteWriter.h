#pragma once
#include <vector>
#include <string>
#include "InputModule.h"

struct VanTrip {
    std::vector<int> villageIndices; // indices into villages[]
    int crewSize = 0;
    double distance = 0.0;
    double fuelCost = 0.0;
};

void write_van_routes_csv(
    const std::vector<int>& solution,
    const std::vector<Village>& villages,
    const std::string& filename = "van_routes.csv"
);
