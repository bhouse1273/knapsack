#include "backend.hpp"
#include <memory>
#include <string>

// Compile with -DBACKEND_CUDA or -DBACKEND_METAL to choose implementation.

#if defined(BACKEND_CUDA)
std::unique_ptr<Backend> make_cuda_backend();
#elif defined(BACKEND_METAL)
std::unique_ptr<Backend> make_metal_backend();
#endif

std::unique_ptr<Backend> make_backend() {
#if defined(BACKEND_CUDA)
    return make_cuda_backend();
#elif defined(BACKEND_METAL)
    return make_metal_backend();
#else
    struct NullBackend : Backend {
        const char* name() const override { return "null"; }
        bool eval_block_candidates(const DeviceSoAHostView&, const BlockSlice&, const CandidatePackHost&,
                                   const std::vector<ObjTerm>&, const std::vector<SoftConstraint>&,
                                   const std::vector<Knapsack>&, float, int, EvalOutHost&, std::string& log) override {
            log = "No backend selected. Define BACKEND_CUDA or BACKEND_METAL.";
            return false;
        }
    };
    return std::make_unique<NullBackend>();
#endif
}
