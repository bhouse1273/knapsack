#include "backend.hpp"
#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include <cmath>

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

    // Candidates: make 8 dummy entries (packed lanes would be built by host).
    CandidatePackHost P;
    P.K = 2;
    P.bits_per_item = 2;
    // Enough space for 5 items * 2 bits = 10 bits -> 1 lane per candidate. We'll just fill zeros.
    P.lanes.assign(1 * 8, 0);

    std::vector<ObjTerm> terms = {{ 1.0f, 0 }, { -1.0f, 1 }, { -1.0f, 2 }};
    std::vector<SoftConstraint> soft; // empty in demo
    std::vector<Knapsack> vans = {{16.0f}, {16.0f}};

    EvalOutHost out;
    std::string log;
    bool ok = backend->eval_block_candidates(A, S, P, terms, soft, vans, 0.18f, 8, out, log);
    std::cout << (ok ? "OK: " : "ERR: ") << log << std::endl;
    std::cout << "obj[0]=" << out.obj[0] << "  pen[0]=" << out.soft_penalty[0] << std::endl;
    return ok ? 0 : 1;
}
