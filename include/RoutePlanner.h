#pragma once
#include <vector>
#include <string>
#include "InputModule.h"
#include "Constants.h"

// Public struct describing a single van trip (exposed to callers)
struct VanTrip {
	std::vector<int> villageIndices; // indices into the "villages" vector
	int crewSize = 0;
	double distance = 0.0;
	double fuelCost = 0.0;
};

// Forward-declared in header and implemented in RoutePlanner.cpp
std::vector<VanTrip> solveVanRoutes(std::vector<Village>& villages, int targetTeamSize);