#include <metal_stdlib>
using namespace metal;

// Placeholder kernel: does nothing useful yet.
// In a real implementation you'd pass buffers for SoA arrays, candidates, etc.
kernel void eval_block_candidates(uint tid [[thread_position_in_grid]]) {
    // no-op
}
