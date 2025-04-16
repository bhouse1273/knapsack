#include <fstream>
#include <iomanip>
#include <cmath>
#include "Constants.h"
#include "InputModule.h"
#include "RouteUtils.h"

void write_van_routes_csv(
    const std::vector<int>& solution,
    const std::vector<Village>& villages,
    const std::string& filename = "van_routes.csv"
) {
    std::ofstream out(filename);
    out << "Van ID,Villages,Distance (km),Fuel Cost (USD)\n";

    int van_id = 1;
    int current_crew = 0;
    std::vector<const Village*> current_trip;

    for (size_t i = 0; i < solution.size(); ++i) {
        if (solution[i]) {
            const Village& v = villages[i];
            if (current_crew + v.workers > MAX_WORKERS_PER_VAN) {
                // Write out current trip
                std::string names;
                double distance = 0.0;

                if (!current_trip.empty()) {
                    distance += haversine(GARAGE_LAT, GARAGE_LON, current_trip[0]->latitude, current_trip[0]->longitude);
                    for (size_t j = 1; j < current_trip.size(); ++j) {
                        distance += haversine(
                            current_trip[j - 1]->latitude, current_trip[j - 1]->longitude,
                            current_trip[j]->latitude, current_trip[j]->longitude
                        );
                    }
                    distance += haversine(
                        current_trip.back()->latitude, current_trip.back()->longitude,
                        FIELD_LAT, FIELD_LON
                    );
                    distance += haversine(FIELD_LAT, FIELD_LON, GARAGE_LAT, GARAGE_LON);
                }

                for (auto* village : current_trip)
                    names += village->name + " ";

                double fuel_cost = distance * (GAS_PRICE_PER_LITER / KM_PER_LITER);
                out << van_id++ << "," << names << "," << std::fixed << std::setprecision(4) << distance << "," << std::setprecision(5) << fuel_cost << "\n";
                current_trip.clear();
                current_crew = 0;
            }

            current_trip.push_back(&v);
            current_crew += v.workers;
        }
    }

    // Flush final trip
    if (!current_trip.empty()) {
        std::string names;
        double distance = 0.0;

        distance += haversine(GARAGE_LAT, GARAGE_LON, current_trip[0]->latitude, current_trip[0]->longitude);
        for (size_t j = 1; j < current_trip.size(); ++j) {
            distance += haversine(
                current_trip[j - 1]->latitude, current_trip[j - 1]->longitude,
                current_trip[j]->latitude, current_trip[j]->longitude
            );
        }
        distance += haversine(
            current_trip.back()->latitude, current_trip.back()->longitude,
            FIELD_LAT, FIELD_LON
        );
        distance += haversine(FIELD_LAT, FIELD_LON, GARAGE_LAT, GARAGE_LON);

        for (auto* village : current_trip)
            names += village->name + " ";

        double fuel_cost = distance * (GAS_PRICE_PER_LITER / KM_PER_LITER);
        out << van_id++ << "," << names << "," << std::fixed << std::setprecision(4) << distance << "," << std::setprecision(5) << fuel_cost << "\n";
    }

    out.close();
}
