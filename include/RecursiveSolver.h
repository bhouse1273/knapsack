#pragma once
#include "Context.h"
#include "InputModule.h"
#include "RouteUtils.h"
#include <vector>

// Updated to match RecursiveSolver.cpp
std::vector<int> recursive_worker(Context ctx, int depth, int target_team_size, const std::vector<Entity>& entities);

