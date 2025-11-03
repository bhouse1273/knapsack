#pragma once
#include <vector>
#include <string>
#include "InputModule.h"
#include "Constants.h"
#include "ResultWriter.h" // for GroupResult

// Forward-declared in header and implemented in RoutePlanner.cpp
// Returns neutral GroupResult entries instead of domain-specific trips.
std::vector<GroupResult> solveGroups(std::vector<Entity>& entities, int targetGroupSize);