#include "RoutePlanner.h"
#include "InputModule.h"
#include "Constants.h"
#include "RecursiveSolver.h"
#include "RouteUtils.h"
#include <fstream>
#include <cmath>
#include <iomanip>
#include <iostream>

std::vector<GroupResult> solveGroups(std::vector<Entity>& entities, int targetGroupSize) {
    std::vector<GroupResult> groups;
    int totalPickedUp = 0;

    std::vector<int> unitsRemaining(entities.size());
    for (size_t i = 0; i < entities.size(); ++i) {
        unitsRemaining[i] = entities[i].resourceUnits;
        if (entities[i].priority <= 0) {
            entities[i].priority = 1 + (std::rand() % 4);  // Assign if missing
        }
    }

    int groupId = 1;
    int maxGroups = MAX_GROUPS * 5;

    while (totalPickedUp < targetGroupSize && groupId <= maxGroups) {
        std::vector<int> blockIndices;
        for (size_t i = 0; i < unitsRemaining.size(); ++i) {
            if (unitsRemaining[i] > 0)
                blockIndices.push_back(i);
            if (blockIndices.size() >= 15) break;
        }

        if (blockIndices.empty()) break;

        std::vector<float> lat(blockIndices.size()), lon(blockIndices.size());
        std::vector<int> weights(blockIndices.size());
        for (size_t i = 0; i < blockIndices.size(); ++i) {
            lat[i] = entities[blockIndices[i]].latitude;
            lon[i] = entities[blockIndices[i]].longitude;
            weights[i] = unitsRemaining[blockIndices[i]];
        }

        Context ctx;
        ctx.item_indices = blockIndices;
    ctx.d_lat = nullptr;     // CPU/Metal path: not used by recursive_worker
    ctx.d_lon = nullptr;     // CPU/Metal path: not used by recursive_worker
    ctx.d_weights = nullptr; // CPU/Metal path: not used by recursive_worker
        ctx.N = blockIndices.size();
        ctx.field_lat = FIELD_LAT;
        ctx.field_lon = FIELD_LON;

    auto plan = recursive_worker(ctx, 0, MAX_UNITS_PER_GROUP, entities);

        if (plan.empty()) {
            std::cout << "⚠️ No plan returned for group " << groupId << "\n";
            break;
        }

        GroupResult trip;
        double distance = 0.0;
        int units = 0;
        double prev_lat = GARAGE_LAT;
        double prev_lon = GARAGE_LON;

        for (size_t i = 0; i < plan.size(); ++i) {
            if (!plan[i]) continue;
            int itemIdx = blockIndices[i];

            units += unitsRemaining[itemIdx];
            trip.itemIndices.push_back(itemIdx);

            double d = haversine(prev_lat, prev_lon,
                                 entities[itemIdx].latitude, entities[itemIdx].longitude);
            distance += d;
            prev_lat = entities[itemIdx].latitude;
            prev_lon = entities[itemIdx].longitude;

            unitsRemaining[itemIdx] = 0;
        }

        if (units == 0) {
            std::cout << "⚠️ No units selected for group " << groupId << "\n";
            continue;
        }

        distance += haversine(prev_lat, prev_lon, FIELD_LAT, FIELD_LON);
        distance += haversine(FIELD_LAT, FIELD_LON, GARAGE_LAT, GARAGE_LON);

        trip.resourceUnits = units;
        trip.metric = distance;
        trip.cost = distance * (GAS_PRICE_PER_LITER / KM_PER_LITER);

        std::cout << "� Group " << groupId << ": " << units << " units, "
                  << std::fixed << std::setprecision(2) << trip.metric << " km\n";

        groups.push_back(trip);
        totalPickedUp += units;
        groupId++;
    }

    return groups;
}
