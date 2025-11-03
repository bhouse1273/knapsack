#include <fstream>
#include <iomanip>
#include <cmath>
#include "Constants.h"
#include "InputModule.h"
#include "RouteUtils.h"

// NOTE: This writer is a legacy demo utility. To keep the solver generic,
// the output labels are neutral (no domain-specific words) and the file name
// is caller-controlled. Prefer a JSON-based V2 pipeline for production use.
void write_results_csv(
    const std::vector<int>& solution,
    const std::vector<Entity>& entities,
    const std::string& filename /*= "routes.csv"*/
) {
    std::ofstream out(filename);
    // Neutral column labels to avoid hard-wiring a business domain.
    out << "ID,Items,Distance (km),Cost\n";

    int route_id = 1;
    int current_load = 0;
    std::vector<const Entity*> current_group;

    for (size_t i = 0; i < solution.size(); ++i) {
        if (solution[i]) {
            const Entity& rec = entities[i];
            if (current_load + rec.resourceUnits > MAX_UNITS_PER_GROUP) {
                // Emit current group
                std::string names;
                double distance = 0.0;

                if (!current_group.empty()) {
                    distance += haversine(GARAGE_LAT, GARAGE_LON, current_group[0]->latitude, current_group[0]->longitude);
                    for (size_t j = 1; j < current_group.size(); ++j) {
                        distance += haversine(
                            current_group[j - 1]->latitude, current_group[j - 1]->longitude,
                            current_group[j]->latitude, current_group[j]->longitude
                        );
                    }
                    distance += haversine(
                        current_group.back()->latitude, current_group.back()->longitude,
                        FIELD_LAT, FIELD_LON
                    );
                    distance += haversine(FIELD_LAT, FIELD_LON, GARAGE_LAT, GARAGE_LON);
                }

                for (auto* ptr : current_group)
                    names += ptr->name + " ";

                const double cost = distance * (GAS_PRICE_PER_LITER / KM_PER_LITER);
                out << route_id++ << "," << names << "," << std::fixed << std::setprecision(4) << distance << "," << std::setprecision(5) << cost << "\n";
                current_group.clear();
                current_load = 0;
            }

            current_group.push_back(&rec);
            current_load += rec.resourceUnits;
        }
    }

    // Flush final trip
    if (!current_group.empty()) {
        std::string names;
        double distance = 0.0;

        distance += haversine(GARAGE_LAT, GARAGE_LON, current_group[0]->latitude, current_group[0]->longitude);
        for (size_t j = 1; j < current_group.size(); ++j) {
            distance += haversine(
                current_group[j - 1]->latitude, current_group[j - 1]->longitude,
                current_group[j]->latitude, current_group[j]->longitude
            );
        }
        distance += haversine(
            current_group.back()->latitude, current_group.back()->longitude,
            FIELD_LAT, FIELD_LON
        );
        distance += haversine(FIELD_LAT, FIELD_LON, GARAGE_LAT, GARAGE_LON);

        for (auto* ptr : current_group)
            names += ptr->name + " ";

        const double cost = distance * (GAS_PRICE_PER_LITER / KM_PER_LITER);
        out << route_id++ << "," << names << "," << std::fixed << std::setprecision(4) << distance << "," << std::setprecision(5) << cost << "\n";
    }

    out.close();
}
