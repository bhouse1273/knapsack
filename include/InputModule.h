#pragma once
#include <string>
#include <vector>

// Generic entity record for classical CSV demos (kept minimal and neutral).
struct Entity {
    std::string name;     // identifier/label
    float latitude;       // optional geo latitude (degrees)
    float longitude;      // optional geo longitude (degrees)
    int resourceUnits;    // generic resource count (e.g., workers, slots)
    int priority = 1;     // generic priority/grade (1..N)
};

// Generic CSV loader for entities (preferred going forward).
std::vector<Entity> load_entities_from_csv(const std::string& filename);
