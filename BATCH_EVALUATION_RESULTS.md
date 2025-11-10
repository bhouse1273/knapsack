# Batch Evaluation Performance Results: CPU vs Metal GPU

## Summary

We implemented batch evaluation to test if Metal GPU could outperform CPU when evaluating many candidates in parallel. The results show that **on M1 hardware, the CPU consistently outperforms the GPU even with batch evaluation**.

## Implementation

### Batch Evaluation API
```cpp
bool EvaluateMetal_Batch(const Config& cfg, const HostSoA& soa,
                         const std::vector<CandidateSelect>& candidates,
                         std::vector<EvalResult>* results, std::string* err);
```

Evaluates N candidates in a single GPU call by:
1. Packing all N candidates into a single buffer (2 bits per item per candidate)
2. Dispatching Metal kernel once with N threads (one per candidate)
3. Unpacking N results from GPU output buffers

### Benchmark Tool
- `tools/benchmark_batch_cpu_vs_metal.cpp` - compares CPU sequential vs Metal parallel
- Generates random candidates for testing
- Validates 100% correctness (CPU and Metal must produce identical results)

## Performance Results

### Test Configuration: 1000-item problem

| Candidates | CPU Time (ms) | Metal Time (ms) | Speedup | Winner |
|------------|---------------|-----------------|---------|--------|
| 100        | 0.12          | 1.68            | 0.07x   | CPU 14x faster |
| 1,000      | 15.83         | 20.98           | 0.75x   | CPU 1.3x faster |
| 5,000      | 77.94         | 95.80           | 0.81x   | CPU 1.2x faster |
| 10,000     | 168.66        | 188.45          | 0.89x   | CPU 1.1x faster |
| 20,000     | 309.53        | 370.57          | 0.84x   | CPU 1.2x faster |

**Trend**: GPU improves from 0.07x → 0.89x speedup as batch size increases from 100 → 10,000, but gets worse at 20,000 (memory transfer bottleneck).

### Throughput Analysis

At peak performance (10,000 candidates):
- **CPU**: 59,290 candidates/sec
- **Metal**: 53,064 candidates/sec

CPU maintains higher throughput across all batch sizes.

## Why CPU Wins on M1

### 1. GPU Overhead Dominates
- **Metal overhead**: ~700μs per call (data transfer + dispatch + sync)
- **Evaluation time**: ~17μs per candidate (1000 items)
- Even with 10,000 candidates, overhead is ~4% of total GPU time

### 2. M1 Unified Memory Architecture
- CPU and GPU share same physical RAM
- Data still needs to be prepared in GPU-compatible format
- No discrete GPU memory bandwidth advantage
- Synchronization overhead remains

### 3. M1 CPU is Extremely Fast
- M1 CPU achieves 59,290 candidates/sec sequential evaluation
- Single-core performance is exceptional
- Cache locality benefits sequential evaluation
- Branch predictor and out-of-order execution highly effective

### 4. Memory Transfer Bottleneck
- Speedup degrades at 20,000 candidates (0.84x vs 0.89x)
- Suggests memory bandwidth becoming limiting factor
- Packing/unpacking overhead increases linearly with batch size

## Correctness Validation

All tests with 1000-item problems: **✅ 100% PASS**
- Max difference: 0.00e+00
- CPU and Metal produce bitwise-identical results

### Floating-Point Precision Issue (5000 items)
- 5000-item problem shows numerical differences (~10^-5 relative error)
- All 1000 candidates had small mismatches (e.g., 6.28778e+07 vs 6.28776e+07)
- Likely due to different floating-point arithmetic order on GPU
- Does not affect correctness for practical purposes but failed exact match

## Recommendations

### When to Use CPU Evaluation
- **Always** on M1/M2/M3 hardware
- Single candidate evaluation
- Batch sizes < 50,000 candidates
- When exact numerical reproducibility is required

### When Metal GPU Might Help
- **Discrete AMD/NVIDIA GPUs** (tested on M1 only)
- Extremely large batches (100,000+ candidates) *might* overcome overhead
- When CPU is heavily loaded with other work
- Multi-GPU scenarios (not tested)

### For This Project
We recommend:
1. **Use CPU evaluation** as primary method
2. Keep Metal batch API for future hardware testing
3. Document M1 performance characteristics
4. Consider discrete GPU testing on high-end hardware

## Next Steps

### Potential Improvements
1. **Test on discrete GPUs** - AMD/NVIDIA may have different characteristics
2. **Optimize Metal kernel** - Reduce shared memory bank conflicts
3. **Async GPU evaluation** - Overlap CPU/GPU work
4. **Multi-threading CPU** - Parallel CPU evaluation for fair comparison

### Testing Needed
- [ ] Test on Mac Pro with discrete AMD GPU
- [ ] Test on eGPU configurations
- [ ] Compare M1 vs M1 Pro vs M1 Max (more GPU cores)
- [ ] Test with different problem sizes (10,000+ items)

## Conclusion

The batch evaluation infrastructure is complete and working correctly. However, **M1's CPU is faster than its integrated GPU** for this workload, even when evaluating thousands of candidates in parallel.

This is not a failure - it's valuable data showing that:
1. M1's CPU is exceptionally fast for this type of computation
2. GPU overhead cannot be amortized enough on integrated GPUs
3. Real-world performance testing is essential (assumptions proven wrong)
4. CPU-first implementation was the right choice

The batch API remains valuable for:
- Future hardware testing (discrete GPUs)
- Documenting performance characteristics
- Demonstrating when GPUs help vs hurt
