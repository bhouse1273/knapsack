#pragma once
#include <vector>
#include <string>
#include "InputModule.h"

// Generic summary of a selected group and its aggregates.
struct GroupResult {
    std::vector<int> itemIndices;  // indices into the problem's items
    int resourceUnits = 0;          // generic resource count (e.g., crew, slots)
    double metric = 0.0;            // domain-specific aggregate (e.g., distance, volume)
    double cost = 0.0;              // aggregate cost
};

// Legacy alias removed to avoid domain-specific naming.

// Legacy demo writer: outputs a CSV with neutral column labels.
// The caller must provide the output filename.
void write_results_csv(
    const std::vector<int>& solution,
    const std::vector<Entity>& entities,
    const std::string& filename
);
