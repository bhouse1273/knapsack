#include <metal_stdlib>
using namespace metal;

struct ObjTerm {
    float weight;
    uchar expr_id; // 0: prod_workers, 1: fuel_cost, 2: pickup_cost
};

struct SoftConstraint {
    uchar sense;    // 0=LE,1=GE,2=EQ
    float rhs;
    float weight;
    float power;
    uchar lhs_attr; // 0: workers
};

struct Uniforms {
    int   item_offset;
    int   item_count;
    int   bits_per_item;
    int   K;                 // knapsacks
    int   words_per_cand;    // uint32 words per candidate
    float fuel_rate;
    float seats0;
    float seats1;
    float seats2;
    float seats3;
    int   T;                 // number of objective terms
    int   Ssoft;             // number of soft constraints
};

inline float expr_eval(uchar expr_id,
                       int idx,
                       device const float* workers,
                       device const float* dg,
                       device const float* dvf,
                       device const float* dfg,
                       device const float* prod,
                       device const float* pick,
                       float fuel_rate) {
    switch (expr_id) {
        case 0: {
            float w = workers[idx];
            float p = prod[idx];
            return w * (1.0f + 0.15f * (p - 1.0f));
        }
        case 1: {
            float d = dg[idx] + dvf[idx] + dfg[idx];
            return d * fuel_rate;
        }
        case 2: {
            return pick[idx];
        }
        default:
            return 0.0f;
    }
}

// Read 2-bit assignment for item i from packed lanes buffer for candidate c
inline int get_assignment(device const uint* lanes,
                          int words_per_cand,
                          int bits_per_item,
                          int cand_index,
                          int i) {
    int bit = i * bits_per_item;
    int lane = bit >> 5;       // /32
    int shift = bit & 31;      // %32
    uint word = lanes[cand_index * words_per_cand + lane];
    int mask = (1 << bits_per_item) - 1;
    return (word >> shift) & mask; // 0..K
}

kernel void eval_block_candidates(
    device const float* workers           [[ buffer(0) ]],
    device const float* dist_garage       [[ buffer(1) ]],
    device const float* dist_village_field[[ buffer(2) ]],
    device const float* dist_field_garage [[ buffer(3) ]],
    device const float* productivity      [[ buffer(4) ]],
    device const float* pick_cost         [[ buffer(5) ]],

    device const uint*  cand_lanes        [[ buffer(6) ]],

    device const ObjTerm* obj_terms       [[ buffer(7) ]],
    device const SoftConstraint* soft_cs  [[ buffer(8) ]],

    device float* out_obj                 [[ buffer(9) ]],
    device float* out_pen                 [[ buffer(10) ]],

    constant Uniforms& U                  [[ buffer(11) ]],

    uint tid [[thread_position_in_grid]])
{
    uint cand = tid;
    // Accumulators
    float obj_sum = 0.0f;
    float van_load[4] = {0.f,0.f,0.f,0.f};

    // Iterate items in this block
    for (int i = 0; i < U.item_count; ++i) {
        int idx = U.item_offset + i;
        int a = get_assignment(cand_lanes, U.words_per_cand, U.bits_per_item, cand, i);
        if (a > 0 && a <= U.K) {
            // Objective terms
            for (int t = 0; t < U.T; ++t) {
                float val = expr_eval(obj_terms[t].expr_id, idx, workers, dist_garage, dist_village_field, dist_field_garage, productivity, pick_cost, U.fuel_rate);
                obj_sum += obj_terms[t].weight * val;
            }
            van_load[a-1] += workers[idx];
        }
    }

    // Hard seat capacity per van (block-local)
    bool infeasible = false;
    float seats[4] = {U.seats0, U.seats1, U.seats2, U.seats3};
    for (int v = 0; v < U.K; ++v) {
        if (van_load[v] > seats[v] + 1e-6f) { infeasible = true; break; }
    }

    // Soft constraints (global example for workers GE target, block-local contribution only)
    float pen = 0.0f;
    for (int s = 0; s < U.Ssoft; ++s) {
        SoftConstraint C = soft_cs[s];
        float lhs = 0.0f;
        if (C.lhs_attr == 0) {
            float sumw = 0.0f;
            for (int v = 0; v < U.K; ++v) sumw += van_load[v];
            lhs = sumw;
        }
        float viol = 0.0f;
        if (C.sense == 1) { // GE
            viol = max(0.0f, C.rhs - lhs);
        } else if (C.sense == 0) { // LE
            viol = max(0.0f, lhs - C.rhs);
        } else { // EQ
            viol = fabs(lhs - C.rhs);
        }
        if (viol > 0.0f) {
            pen += C.weight * pow(viol, C.power);
        }
    }

    float final_obj = infeasible ? -INFINITY : (obj_sum - pen);
    out_obj[cand] = final_obj;
    out_pen[cand] = pen;
}
