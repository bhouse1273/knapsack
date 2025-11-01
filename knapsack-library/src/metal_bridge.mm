#if defined(__APPLE__)
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#endif

#include <string>
#include <fstream>
#include <sstream>

// Use the existing C API for Metal runtime init/eval
#include "../../kernels/metal/metal_api.h"

extern "C" {

// Attempt to initialize Metal from a shader source file, preferring the copy next to the Go package.
// Returns 1 on success, 0 on failure (or non-Apple builds).
int knapsack_metal_init_default() {
#if defined(__APPLE__)
  const char* candidates[] = {
    // go-embedded copy used by bindings/go/metal
    "../bindings/go/metal/eval_block_candidates.metal",
    // canonical shader path in repo
    "../kernels/metal/shaders/eval_block_candidates.metal",
    // fallback relative to this file location (when build dir differs)
    "../../bindings/go/metal/eval_block_candidates.metal",
    "../../kernels/metal/shaders/eval_block_candidates.metal",
  };
  std::string src;
  for (const char* p : candidates) {
    std::ifstream f(p);
    if (!f.good()) continue;
    std::ostringstream ss; ss << f.rdbuf();
    src = ss.str();
    if (!src.empty()) break;
  }
  if (src.empty()) {
    return 0;
  }
  int rc = knapsack_metal_init_from_source(src.data(), src.size(), nullptr, 0);
  return rc == 0 ? 1 : 0;
#else
  return 0;
#endif
}

}
