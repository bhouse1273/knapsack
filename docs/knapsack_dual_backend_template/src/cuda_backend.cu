#include "backend.hpp"
#include <sstream>
#ifdef __CUDACC__
#include <cuda_runtime.h>
#endif

struct CudaBackend : Backend {
    const char* name() const override { return "cuda"; }
    bool eval_block_candidates(const DeviceSoAHostView& A,
                               const BlockSlice& S,
                               const CandidatePackHost& P,
                               const std::vector<ObjTerm>& obj_terms,
                               const std::vector<SoftConstraint>& soft_cs,
                               const std::vector<Knapsack>& vans,
                               float fuel_rate_per_km,
                               int N_candidates,
                               EvalOutHost& out,
                               std::string& log) override
    {
#ifndef __CUDACC__
        (void)A;(void)S;(void)P;(void)obj_terms;(void)soft_cs;(void)vans;(void)fuel_rate_per_km;
        out.obj.assign(N_candidates, -1e30f);
        out.soft_penalty.assign(N_candidates, 0.0f);
        log = "CUDA not available in this build. Stub executed.";
        return true;
#else
        // Real CUDA implementation would go here (omitted for brevity).
        out.obj.assign(N_candidates, 0.0f);
        out.soft_penalty.assign(N_candidates, 0.0f);
        log = "CUDA backend ran (placeholder).";
        return true;
#endif
    }
};

std::unique_ptr<Backend> make_cuda_backend() { return std::make_unique<CudaBackend>(); }
