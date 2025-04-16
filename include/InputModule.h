#pragma once
#include <string>
#include <vector>

struct Village {
    std::string name;
    float latitude;
    float longitude;
    int workers;
    int productivityScore = 1;  // 1 (entry-level) to 4 (superstar)
};

std::vector<Village> load_villages_from_csv(const std::string& filename);
