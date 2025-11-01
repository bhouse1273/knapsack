#include "backend.hpp"
#include <iostream>
#include <vector>
#include <cstdint>

static void pack_assignment_row(std::vector<uint32_t>& lanes,
                                int words_per_cand,
                                int bits_per_item,
                                const std::vector<int>& assign) {
    // assign.size() = item_count, values 0..K
    for (int i = 0; i < (int)assign.size(); ++i) {
        int bit = i * bits_per_item;
        int lane = bit >> 5;
        int shift = bit & 31;
        uint32_t mask = ((1u << bits_per_item) - 1u) << shift;
        lanes[lane] = (lanes[lane] & ~mask) | ((uint32_t)assign[i] << shift);
    }
}

int main() {
    auto backend = make_backend();
    std::cout << "Backend: " << backend->name() << std::endl;

    // Demo data: 5 items
    std::vector<float> workers   = {11,7,18,12,20};
    std::vector<float> dg        = {15.2f,10.0f,21.9f,9.3f,6.8f};
    std::vector<float> dvf       = {22.4f,19.1f,12.5f,15.7f,14.3f};
    std::vector<float> dfg       = {30.0f,27.0f,29.0f,24.0f,23.0f};
    std::vector<float> prod      = {3,2,4,2,1};
    std::vector<float> pick      = {2,2,3,2,1};

    DeviceSoAHostView A{workers.data(), dg.data(), dvf.data(), dfg.data(), prod.data(), pick.data(), 5};
    BlockSlice S{0, 5};

    // Backends share this candidate pack format:
    CandidatePackHost P;
    P.K = 2;
    P.bits_per_item = 2;
    const int words_per_cand = (S.item_count * P.bits_per_item + 31) / 32;

    // Create 8 candidates; lanes laid out as [cand0 words][cand1 words]...
    int N_candidates = 8;
    P.lanes.assign(words_per_cand * N_candidates, 0);

    // Fill a few example assignment rows (0=not selected, 1=van1, 2=van2)
    // Candidate 0 selects first two on van1
    {
        std::vector<uint32_t> lane(words_per_cand, 0);
        std::vector<int> a = {1,1,0,0,0};
        pack_assignment_row(lane, words_per_cand, P.bits_per_item, a);
        for (int w=0; w<words_per_cand; ++w) P.lanes[w] = lane[w];
    }
    // Candidate 1 selects third on van2
    {
        std::vector<uint32_t> lane(words_per_cand, 0);
        std::vector<int> a = {0,0,2,0,0};
        pack_assignment_row(lane, words_per_cand, P.bits_per_item, a);
        for (int w=0; w<words_per_cand; ++w) P.lanes[words_per_cand*1 + w] = lane[w];
    }
    // The rest remain zeros.

    std::vector<ObjTerm> terms = {
        { 1.0f, 0 },     // productive workers
        { -1.0f, 1 },    // fuel cost
        { -1.0f, 2 }     // pickup fixed cost
    };
    std::vector<SoftConstraint> soft = {
        { 1 /*GE*/, 64.0f /*rhs*/, 10.0f /*w*/, 2.0f /*p*/, 0 /*workers*/ }
    };
    std::vector<Knapsack> vans = {{16.0f}, {16.0f}};

    EvalOutHost out;
    std::string log;

    bool ok = backend->eval_block_candidates(A, S, P, terms, soft, vans, 0.18f, N_candidates, out, log);
    std::cout << (ok ? "OK: " : "ERR: ") << log << std::endl;
    for (int i = 0; i < N_candidates; ++i) {
        std::cout << "cand " << i << ": obj=" << out.obj[i] << " pen=" << out.soft_penalty[i] << std::endl;
    }
    return ok ? 0 : 1;
}
