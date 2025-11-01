#pragma once
#include <cstdint>
#include <vector>
#include <string>

// Minimal data views used by both backends.
// In a real project, replace with your SoA device views and candidate packing.
struct DeviceSoAHostView {
    // Flat arrays sized to item_count
    const float* workers;
    const float* dist_garage;
    const float* dist_village_field;
    const float* dist_field_garage;
    const float* productivity;
    const float* pick_cost;
    int item_count;
};

struct BlockSlice {
    int item_offset; // start index in the global arrays
    int item_count;  // number of items in this block
};

struct EvalOutHost {
    std::vector<float> obj;         // size = N_candidates
    std::vector<float> soft_penalty;
};

struct ObjTerm {
    float weight;
    uint8_t expr_id;  // 0: prod_workers, 1: fuel_cost, 2: fixed_pick_cost
};

struct SoftConstraint {
    uint8_t sense;    // 0=LE,1=GE,2=EQ
    float rhs;
    float weight;
    float power;
    uint8_t lhs_attr; // 0: workers (demo)
};

struct Knapsack {
    float seats;
};

struct CandidatePackHost {
    // Packed 2-bit assignments back-to-back in lanes.
    std::vector<uint32_t> lanes;
    int bits_per_item{2};
    int K{2};
};

class Backend {
public:
    virtual ~Backend() = default;
    virtual const char* name() const = 0;
    virtual bool eval_block_candidates(const DeviceSoAHostView& A,
                                       const BlockSlice& S,
                                       const CandidatePackHost& P,
                                       const std::vector<ObjTerm>& obj_terms,
                                       const std::vector<SoftConstraint>& soft_cs,
                                       const std::vector<Knapsack>& vans,
                                       float fuel_rate_per_km,
                                       int N_candidates,
                                       EvalOutHost& out,
                                       std::string& log) = 0;
};

// Factory (implemented in backend_factory.cpp)
std::unique_ptr<Backend> make_backend();
