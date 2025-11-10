#!/bin/bash
# Download standard benchmark datasets for knapsack solver validation
# See docs/BENCHMARK_DATASETS.md for details on each dataset
#
# Usage:
#   ./download_benchmarks.sh [target_directory]
#
# If no directory specified, uses PROJECT_ROOT/data/benchmarks
# Example:
#   ./download_benchmarks.sh /Volumes/mtheoryssd/2025-M-Theory/KnapsackTestData

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Use provided path or default to project data directory
if [ -n "$1" ]; then
    BENCHMARK_DIR="$1"
    echo "Using custom benchmark directory: $BENCHMARK_DIR"
else
    BENCHMARK_DIR="$PROJECT_ROOT/data/benchmarks"
    echo "Using default benchmark directory: $BENCHMARK_DIR"
fi

# Verify the directory exists or can be created
if [ ! -d "$BENCHMARK_DIR" ]; then
    echo "Creating benchmark directory: $BENCHMARK_DIR"
    mkdir -p "$BENCHMARK_DIR" || {
        echo "Error: Cannot create directory $BENCHMARK_DIR"
        echo "Please ensure the path exists and you have write permissions."
        exit 1
    }
fi

echo "Creating benchmark directory structure..."
mkdir -p "$BENCHMARK_DIR"/{or-library,pisinger,miplib,gap}

# ============================================================================
# OR-Library: Multiple Knapsack Problem (MKP)
# ============================================================================
echo ""
echo "Downloading OR-Library MKP instances..."
cd "$BENCHMARK_DIR/or-library"

# Main MKP instances
if [ ! -f "mknap1.txt" ]; then
    echo "  - mknap1.txt (30 problems, 6-10 knapsacks)"
    curl -fsSL -o mknap1.txt \
        "http://people.brunel.ac.uk/~mastjjb/jeb/orlib/files/mknap1.txt"
fi

if [ ! -f "mknap2.txt" ]; then
    echo "  - mknap2.txt (30 problems, different seeds)"
    curl -fsSL -o mknap2.txt \
        "http://people.brunel.ac.uk/~mastjjb/jeb/orlib/files/mknap2.txt"
fi

# Large instances
if [ ! -f "mknapcb1.txt" ]; then
    echo "  - mknapcb1.txt (larger instances)"
    curl -fsSL -o mknapcb1.txt \
        "http://people.brunel.ac.uk/~mastjjb/jeb/orlib/files/mknapcb1.txt" || \
        echo "    (Note: mknapcb1.txt may not be available)"
fi

echo "  ✓ OR-Library instances downloaded"

# ============================================================================
# Pisinger's Benchmark Instances
# ============================================================================
echo ""
echo "Downloading Pisinger's hard instances..."
cd "$BENCHMARK_DIR/pisinger"

# Note: Pisinger's website may require manual download
# These URLs are examples and may need updating
echo "  Note: Pisinger instances may require manual download from:"
echo "  http://hjemmesidi.diku.dk/~pisinger/codes.html"
echo ""
echo "  For now, creating placeholder README..."

cat > README.md << 'EOF'
# Pisinger's Benchmark Instances

## Manual Download Required

Visit: http://hjemmesidi.diku.dk/~pisinger/codes.html

Download the following datasets:
- `kplib_uncorrelated.tar` - Random weights and values
- `kplib_weaklycorr.tar` - Weakly correlated instances
- `kplib_stronglycorr.tar` - Strongly correlated instances
- `kplib_subsum.tar` - Subset sum problems (hardest)

Extract them to this directory.

## Instance Types

1. **Uncorrelated**: Random weights/values (easiest)
2. **Weakly correlated**: v_i = w_i + random
3. **Strongly correlated**: v_i = w_i + constant
4. **Subset sum**: v_i = w_i (hardest for many algorithms)

These instances test algorithmic robustness and identify weaknesses.
EOF

echo "  ✓ Created Pisinger README (manual download needed)"

# ============================================================================
# GAP: Generalized Assignment Problem
# ============================================================================
echo ""
echo "Downloading GAP instances..."
cd "$BENCHMARK_DIR/gap"

if [ ! -f "gap1.txt" ]; then
    echo "  - gap1.txt (Type A: 5 agents, 100 jobs)"
    curl -fsSL -o gap1.txt \
        "http://people.brunel.ac.uk/~mastjjb/jeb/orlib/files/gap1.txt" || \
        echo "    (Note: gap1.txt may not be available)"
fi

if [ ! -f "gapa.txt" ]; then
    echo "  - gapa.txt (Type A instances)"
    curl -fsSL -o gapa.txt \
        "http://people.brunel.ac.uk/~mastjjb/jeb/orlib/files/gapa.txt" || \
        echo "    (Note: gapa.txt may not be available)"
fi

cat > README.md << 'EOF'
# Generalized Assignment Problem (GAP) Instances

## Source
OR-Library: http://people.brunel.ac.uk/~mastjjb/jeb/orlib/gapinfo.html

## Instance Types

- **Type A**: 5 agents, 100 jobs
- **Type B**: 10 agents, 100 jobs
- **Type C**: 5 agents, 200 jobs
- **Type D**: 20 agents, 200 jobs
- **Type E**: 20 agents, 900 jobs

Each agent has a capacity constraint, and each job has different costs
and resource requirements depending on which agent it's assigned to.

## Format

```
number_of_agents number_of_jobs
cost_matrix (agents x jobs)
resource_matrix (agents x jobs)
capacity_of_agent_1 capacity_of_agent_2 ...
```

This maps well to knapsack assign mode with multiple knapsacks.
EOF

echo "  ✓ Created GAP README"

# ============================================================================
# Create sample instances for testing
# ============================================================================
echo ""
echo "Creating small test instances..."
cd "$BENCHMARK_DIR"

mkdir -p samples

cat > samples/small_mkp.txt << 'EOF'
# Small Multiple Knapsack Problem
# 10 items, 3 knapsacks
# Format: n m / profits / capacities / weights per knapsack
10 3
50 60 70 80 90 100 110 120 130 140
100 150 120
15 20 25 30 35 40 45 50 55 60
18 22 28 32 38 42 48 52 58 62
12 16 20 24 28 32 36 40 44 48
EOF

cat > samples/README.md << 'EOF'
# Sample Test Instances

Small instances for quick testing and validation.

## small_mkp.txt
- 10 items, 3 knapsacks
- Simple instance for unit testing
- Can solve optimally by hand for verification
EOF

echo "  ✓ Created sample instances"

# ============================================================================
# Summary
# ============================================================================
echo ""
echo "============================================================================"
echo "Benchmark Download Summary"
echo "============================================================================"
echo ""
echo "Downloaded:"
echo "  ✓ OR-Library MKP instances (mknap1.txt, mknap2.txt)"
echo "  ✓ Sample test instances"
echo ""
echo "Manual download required:"
echo "  ⚠ Pisinger instances - visit http://hjemmesidi.diku.dk/~pisinger/codes.html"
echo "  ⚠ GAP instances - may need manual download from OR-Library"
echo ""
echo "Benchmark location: $BENCHMARK_DIR"
echo ""

# Create symlink if using external directory
if [ "$BENCHMARK_DIR" != "$PROJECT_ROOT/data/benchmarks" ]; then
    SYMLINK_PATH="$PROJECT_ROOT/data/benchmarks"
    if [ ! -e "$SYMLINK_PATH" ]; then
        echo "Creating symlink from $SYMLINK_PATH to $BENCHMARK_DIR"
        mkdir -p "$PROJECT_ROOT/data"
        ln -s "$BENCHMARK_DIR" "$SYMLINK_PATH"
        echo "  ✓ Symlink created - benchmarks accessible via data/benchmarks/"
    elif [ -L "$SYMLINK_PATH" ]; then
        echo "  ℹ Symlink already exists at $SYMLINK_PATH"
    elif [ -d "$SYMLINK_PATH" ]; then
        echo "  ⚠ Warning: $SYMLINK_PATH is a directory, not creating symlink"
    fi
    echo ""
fi

echo "Next steps:"
echo "  1. Review docs/BENCHMARK_DATASETS.md"
echo "  2. Create format converters in examples/python/"
echo "  3. Run validation tests with make test-benchmarks"
echo ""
echo "============================================================================"

# Create a quick reference file
cat > "$BENCHMARK_DIR/README.md" << 'EOF'
# Benchmark Datasets

This directory contains standard benchmark instances for validating the knapsack solver.

## Directory Structure

```
benchmarks/
├── or-library/      # Multiple Knapsack Problem (MKP) instances
├── pisinger/        # Hard knapsack instances (various correlations)
├── gap/             # Generalized Assignment Problem instances
├── miplib/          # Real-world MIP instances (future)
└── samples/         # Small test instances
```

## Quick Start

```bash
# Download benchmarks
./scripts/download_benchmarks.sh

# Convert OR-Library instance to v2 format (future)
python examples/python/convert_orlib.py data/benchmarks/or-library/mknap1.txt

# Run validation suite (future)
make test-benchmarks
```

## References

See `docs/BENCHMARK_DATASETS.md` for detailed information about each dataset,
format specifications, and recommended validation strategies.

## Known Optimal Solutions

OR-Library instances include optimal solutions for verification:
- mknap1.txt: 30 instances with known optima
- mknap2.txt: 30 instances with known optima

Compare your results against these to measure solution quality gap.
EOF

echo "Created $BENCHMARK_DIR/README.md"
echo "Done!"
