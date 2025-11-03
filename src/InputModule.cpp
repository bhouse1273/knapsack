#include "InputModule.h"
#include <string>
#include <vector>
#include <stdexcept>
#include <fstream>
#include <sstream>
#include <iostream>

std::vector<Entity> load_entities_from_csv(const std::string& filename) {
    std::vector<Entity> entities;
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file: " + filename);
    }

    std::string line;
    std::getline(file, line); // Skip header if present

    while (std::getline(file, line)) {
        if (line.empty()) continue;
        std::istringstream ss(line);
        std::string token;
        Entity e{};

        try {
            // CSV columns (legacy order): name, latitude, longitude, resourceUnits, [priority]
            std::getline(ss, e.name, ',');
            std::getline(ss, token, ','); e.latitude = std::stof(token);
            std::getline(ss, token, ','); e.longitude = std::stof(token);
            std::getline(ss, token, ','); e.resourceUnits = std::stoi(token);
            if (std::getline(ss, token, ',')) {
                if (!token.empty()) e.priority = std::stoi(token);
            }
        } catch (...) {
            std::cerr << "Skipping malformed line: " << line << std::endl;
            continue;
        }

        entities.push_back(std::move(e));
    }

    return entities;
}
