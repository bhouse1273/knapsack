# Benchmark Dataset Setup Summary

## Overview

Public benchmark datasets have been downloaded to an external SSD to avoid disk space issues on the M1. The data is accessible from the project via a symlink.

## Storage Location

**External SSD Path**: `/Volumes/mtheoryssd/2025-M-Theory/KnapsackTestData/`

**Project Symlink**: `data/benchmarks` → External SSD location

**Total Size**: ~3.9 MB (small, but allows room for growth)

## Downloaded Datasets

### ✅ OR-Library Multiple Knapsack Problem (MKP)
- **File**: `or-library/mknap1.txt` (4.3 KB, 7 problems)
- **File**: `or-library/mknap2.txt` (68 KB, 30 problems)  
- **File**: `or-library/mknapcb1.txt` (75 KB, larger instances)
- **Format**: Text format with n, m, profits, capacities, weight matrix
- **Use**: Industry-standard benchmarks with known optimal solutions
- **Optimal solutions**: Included in files for verification

### ✅ Sample Test Instances
- **Location**: `samples/`
- **File**: `small_mkp.txt` (10 items, 3 knapsacks)
- **Use**: Quick testing and unit tests

### ⚠️ Pisinger's Hard Instances (Manual Download Required)
- **Location**: `pisinger/` (README only)
- **Source**: http://hjemmesidi.diku.dk/~pisinger/codes.html
- **Types**: Uncorrelated, weakly correlated, strongly correlated, subset sum
- **Use**: Test algorithmic robustness on challenging instances
- **Action Required**: Manual download from Pisinger's website

### ⚠️ Generalized Assignment Problem (GAP) (Partial)
- **Location**: `gap/`
- **Files**: Attempted download (may need manual retry)
- **Source**: http://people.brunel.ac.uk/~mastjjb/jeb/orlib/gapinfo.html
- **Use**: Multi-agent assignment problems (maps to assign mode)

## Directory Structure

```
/Volumes/mtheoryssd/2025-M-Theory/KnapsackTestData/
├── README.md                    # Overview and references
├── or-library/                  # MKP instances ✅
│   ├── mknap1.txt              # 7 problems
│   ├── mknap2.txt              # 30 problems
│   └── mknapcb1.txt            # Large instances
├── pisinger/                    # Hard instances ⚠️
│   └── README.md               # Download instructions
├── gap/                         # GAP instances ⚠️
│   └── README.md               # Format information
├── miplib/                      # Real-world MIP (future)
└── samples/                     # Small test cases ✅
    ├── small_mkp.txt
    └── README.md
```

## Access from Project

The benchmarks are accessible via:

```bash
# Direct access
cd /Volumes/mtheoryssd/2025-M-Theory/KnapsackTestData

# Via project symlink
cd data/benchmarks

# Both point to same location!
ls -l data/benchmarks  # Shows: data/benchmarks -> /Volumes/mtheoryssd/...
```

## Using the Benchmarks

### Example: Convert OR-Library instance to v2 Config

```bash
# Convert first problem from mknap1.txt
python examples/python/convert_orlib_mkp.py \
    data/benchmarks/or-library/mknap1.txt \
    --problem 0 \
    --mode assign \
    --pretty > problem1.json

# Solve with your solver (future)
./build/knapsack_solver problem1.json
```

### Example: OR-Library mknap1.txt Format

```
7                                    # Number of problems in file

6 10 3800                           # Problem 1: 6 items, 10 knapsacks, optimal=3800
100 600 1200 2400 500 2000          # Profits (6 values)
8 12 13 64 22 41                    # Knapsack 1 capacities
8 12 13 75 22 41                    # Knapsack 2 capacities
... (8 more knapsack rows)
80 96 20 36 44 48 10 18 22 24       # Weights per knapsack

10 10 8706.1                        # Problem 2: 10 items, 10 knapsacks, optimal=8706.1
...
```

## Validation Strategy

### Phase 1: Correctness (Small Problems)
✅ **Dataset**: mknap1.txt problems 1-3 (6-10 items)
- Run with beam_width=64, scout mode
- Compare against known optimal solutions
- Verify you can reach 100% of optimal

### Phase 2: Scalability (Medium Problems)
📋 **Dataset**: mknap2.txt (100-500 items)
- Test varying beam widths (8, 16, 32, 64)
- Measure time vs quality trade-offs
- Show linear scaling

### Phase 3: Robustness (Hard Problems)
📋 **Dataset**: Pisinger instances (requires manual download)
- Test on subset sum, strongly correlated
- Show beam search handles difficult structures
- Compare vs greedy heuristics

### Phase 4: Real-World Validation
📋 **Dataset**: MIPLIB instances (future)
- Compare against Gurobi/CPLEX
- Show competitive performance

## Tools Created

### ✅ Download Script
```bash
# Download to custom location (already run)
./scripts/download_benchmarks.sh /Volumes/mtheoryssd/2025-M-Theory/KnapsackTestData

# Re-run to download to default location
./scripts/download_benchmarks.sh
```

### ✅ Format Converter
```bash
# Convert OR-Library MKP to v2 Config JSON
python examples/python/convert_orlib_mkp.py <input> [--problem N] [--mode select|assign]
```

### 📋 Validation Runner (Future)
```bash
# Run systematic validation
make test-benchmarks
```

## Next Steps

### 1. Fix Test Bug First
Before running benchmarks, fix the quadratic penalty test:
```bash
# Edit tests/v2/test_eval_cpu.cpp line 196
# Change: cand.select = {1, 1, 1, 1, 0};  // Wrong: weight=50, no violation
# To:     cand.select = {1, 1, 0, 1, 1};  // Correct: weight=60, violation=10

# Rebuild and verify all tests pass
make build-tests
make test  # Should show 100% pass rate
```

### 2. Test Converter
```bash
# Verify the OR-Library converter works
python examples/python/convert_orlib_mkp.py \
    data/benchmarks/or-library/mknap1.txt \
    --problem 0 --pretty

# Should output valid v2::Config JSON
```

### 3. Create Validation Suite
```python
# benchmarks/validate.py (future)
# - Load OR-Library instances
# - Convert to v2::Config
# - Solve with beam search
# - Compare against known optimal
# - Generate report
```

### 4. Download Additional Datasets
```bash
# Manual download Pisinger instances
# Visit: http://hjemmesidi.diku.dk/~pisinger/codes.html
# Extract to: /Volumes/mtheoryssd/2025-M-Theory/KnapsackTestData/pisinger/
```

### 5. Create Benchmark Results Document
```markdown
# BENCHMARK_RESULTS.md (future)
## OR-Library mknap1.txt Results
| Problem | Items | Knapsacks | Optimal | v2 Found | Gap % | Time |
|---------|-------|-----------|---------|----------|-------|------|
| 1       | 6     | 10        | 3800    | 3800     | 0.0%  | 0.01s|
| 2       | 10    | 10        | 8706.1  | ?        | ?     | ?    |
...
```

## Documentation References

- **Complete Guide**: `docs/BENCHMARK_DATASETS.md`
- **Download Script**: `scripts/download_benchmarks.sh`
- **Converter**: `examples/python/convert_orlib_mkp.py`
- **Dataset README**: `/Volumes/mtheoryssd/2025-M-Theory/KnapsackTestData/README.md`

## Disk Space Status

✅ **Benchmarks on External SSD**: 3.9 MB
✅ **Project uses symlink**: No internal disk usage
✅ **Room for growth**: Pisinger instances ~50 MB when downloaded

This setup ensures we won't run out of disk space on the M1 while having access to comprehensive validation datasets.
