#include "InputModule.h"
#include <string>
#include <vector>
#include <stdexcept>   // for std::runtime_error
#include <fstream>    // for std::ifstream
#include <sstream>   // for std::istringstream
#include <iostream>  // for std::cerr

std::vector<Village> load_villages_from_csv(const std::string& filename) {
    std::vector<Village> villages;
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file: " + filename);
    }

    std::string line;
    std::getline(file, line); // Skip header

    while (std::getline(file, line)) {
        std::istringstream ss(line);
        std::string token;
        Village v;

        try {
            std::getline(ss, v.name, ',');
            std::getline(ss, token, ','); v.latitude = std::stof(token);
            std::getline(ss, token, ','); v.longitude = std::stof(token);
            std::getline(ss, token, ','); v.workers = std::stoi(token);
        } catch (...) {
            std::cerr << "⚠️ Skipping malformed line: " << line << std::endl;
            continue;
        }
        v.productivityScore = 1 + (std::rand() % 4);  // Random score between 1–4

        villages.push_back(v);
    }

    return villages;
}
