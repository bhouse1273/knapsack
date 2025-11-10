# Benchmark Datasets for Knapsack Solver Validation

This document outlines public datasets that can be used to validate and demonstrate the effectiveness of the knapsack v2 solver.

## Recommended Datasets

### 1. OR-Library: Multiple Knapsack Problem (MKP)
**Source**: http://people.brunel.ac.uk/~mastjjb/jeb/orlib/mknapinfo.html

**Description**: The gold standard for multiple knapsack benchmarks, created by John Beasley.

**Instance Sets**:
- `mknap1.txt`: 30 problems, 6-10 knapsacks, 100-500 items
- `mknap2.txt`: 30 problems, similar structure with different random seeds
- Known optimal solutions provided for verification

**Why Use It**:
- Industry standard since 1990s
- Widely cited in academic literature (1000+ citations)
- Optimal solutions known, allowing precise performance measurement
- Tests your assign mode with multiple knapsacks

**Format Example**:
```
n m (items, knapsacks)
profit_1 profit_2 ... profit_n
capacity_1 capacity_2 ... capacity_m
resource_consumption_matrix (m x n)
```

**How to Use**:
```python
# Convert OR-Library format to v2::Config
cfg.mode = "assign"
cfg.items.count = n
cfg.items.attributes["value"] = profits
cfg.knapsack.K = m
cfg.knapsack.capacities = capacities
cfg.knapsack.capacity_attr = "weight"
# Add constraint for each knapsack
```

### 2. Pisinger's Hard Instances
**Source**: http://hjemmesidi.diku.dk/~pisinger/codes.html

**Description**: Challenging instances designed by David Pisinger specifically to test algorithm robustness.

**Instance Types**:
- **Uncorrelated**: Random weights and values
- **Weakly correlated**: `v_i = w_i + r` where r is random
- **Strongly correlated**: `v_i = w_i + c` (constant offset)
- **Subset sum**: `v_i = w_i` (hardest for many algorithms)
- **Almost strongly correlated**: Slight perturbation of strongly correlated

**Sizes**: 10 to 10,000 items

**Why Use It**:
- Tests algorithmic weaknesses
- Shows performance degradation patterns
- Validates that beam search handles difficult structures

### 3. MIPLIB 2017
**Source**: https://miplib.zib.de/

**Description**: Real-world mixed integer programming problems from industry.

**Relevant Problems**:
- `binpacking*`: Bin packing variants
- `mkp*`: Multiple knapsack problems
- `assign*`: Assignment problems
- Many contain knapsack-like constraints

**Why Use It**:
- Real-world problems (not synthetic)
- Used to benchmark Gurobi, CPLEX, SCIP
- Direct comparison with commercial solvers
- Tests your solver on practical instances

### 4. Capacitated Vehicle Routing Problem (CVRP) Instances
**Source**: http://vrp.atd-lab.inf.puc-rio.br/index.php/en/

**Description**: If your RoutePlanner is for vehicle routing, these are essential.

**Datasets**:
- **Augerat** (A, B, P sets): 32-80 customers
- **Christofides & Eilon**: 50-199 customers
- **Golden**: Large instances (240-484 customers)

**Why Use It**:
- Known optimal solutions
- Tests capacity constraints + routing
- Validates your beam search on routing problems

### 5. Generalized Assignment Problem (GAP)
**Source**: http://people.brunel.ac.uk/~mastjjb/jeb/orlib/gapinfo.html

**Description**: Assign jobs to machines with capacity constraints and varying costs.

**Instance Sets**:
- Type A: 5 agents, 100 jobs
- Type B: 10 agents, 100 jobs  
- Type C: 5 agents, 200 jobs
- Type D: 20 agents, 200 jobs
- Type E: 20 agents, 900 jobs

**Why Use It**:
- Perfect match for your assign mode
- Multiple constraints (capacity per agent)
- Real-world problem structure
- Optimal solutions known

## Recommended Validation Strategy

### Phase 1: Correctness Validation
1. **Small OR-Library instances** (n=100, m=5)
   - Verify you can reach known optimal solutions
   - Use beam_width=64, scout mode
   - Compare against optimal values

2. **Pisinger uncorrelated instances** (n=100-500)
   - Verify basic functionality on standard problems
   - Measure solution quality gap from optimal

### Phase 2: Scalability Testing
1. **Large OR-Library instances** (n=500, m=10)
   - Test performance with varying beam widths
   - Measure time vs quality trade-offs

2. **Very large instances** (n=5000+)
   - Show linear scaling of beam search
   - Demonstrate advantages over exact methods

### Phase 3: Algorithm Robustness
1. **Pisinger hard instances** (subset sum, strongly correlated)
   - Show beam search handles difficult structures
   - Compare vs greedy heuristics

2. **Multi-constraint problems**
   - Test your multi-constraint features
   - Show handling of conflicting objectives

### Phase 4: Real-World Validation
1. **MIPLIB instances**
   - Compare against Gurobi/CPLEX
   - Show competitive performance on practical problems

2. **Your domain-specific data** (villages_300.csv)
   - Demonstrate on your actual use case
   - Show interpretable results

## Creating Comparison Reports

### Recommended Metrics

**Solution Quality**:
- Gap from optimal: `(optimal - found) / optimal * 100%`
- Best solution found across all runs
- Average solution quality over 10 runs

**Performance**:
- Time to best solution
- Time vs quality curves
- Items/second throughput

**Comparison Table Format**:
```
| Instance | n | m | Optimal | v2 Beam | Gap % | Time (s) | Gurobi Time (s) |
|----------|---|---|---------|---------|-------|----------|-----------------|
| mknap1-1 | 100| 5 | 24381  | 24381   | 0.00% | 0.05     | 0.12            |
| mknap1-2 | 200| 5 | 47793  | 47650   | 0.30% | 0.15     | 0.45            |
```

### Visualization Recommendations

1. **Quality vs Time Curves**: Show anytime behavior
2. **Beam Width Impact**: Quality and time vs beam_width
3. **Scaling Charts**: Time vs problem size (log-log plot)
4. **Hard Instance Performance**: Gap % across Pisinger categories

## Downloading and Using Datasets

### OR-Library Download Script
```bash
#!/bin/bash
# Download OR-Library MKP instances

mkdir -p data/benchmarks/or-library
cd data/benchmarks/or-library

wget http://people.brunel.ac.uk/~mastjjb/jeb/orlib/files/mknap1.txt
wget http://people.brunel.ac.uk/~mastjjb/jeb/orlib/files/mknap2.txt
wget http://people.brunel.ac.uk/~mastjjb/jeb/orlib/files/mknapcb1.txt

echo "Downloaded OR-Library instances"
```

### Pisinger Download Script
```bash
#!/bin/bash
mkdir -p data/benchmarks/pisinger
cd data/benchmarks/pisinger

# Download various problem types
wget http://hjemmesidi.diku.dk/~pisinger/kplib_uncorrelated.tar
wget http://hjemmesidi.diku.dk/~pisinger/kplib_weaklycorr.tar
wget http://hjemmesidi.diku.dk/~pisinger/kplib_stronglycorr.tar

tar -xf kplib_uncorrelated.tar
tar -xf kplib_weaklycorr.tar
tar -xf kplib_stronglycorr.tar
```

## Format Converters

You'll need to create converters from benchmark formats to v2::Config. Example:

```python
# examples/python/convert_orlib.py
import json

def parse_orlib_mkp(filename):
    """Convert OR-Library MKP format to v2 Config JSON"""
    with open(filename) as f:
        lines = [l.strip() for l in f if l.strip()]
    
    # First line: n_items, n_knapsacks
    n, m = map(int, lines[0].split())
    
    # Next n lines: profits
    profits = list(map(float, lines[1:n+1]))
    
    # Next m lines: capacities
    capacities = list(map(float, lines[n+1:n+m+1]))
    
    # Next m*n values: resource consumption matrix
    weights = []
    # Parse matrix...
    
    config = {
        "mode": "assign",
        "items": {
            "count": n,
            "attributes": {
                "value": profits,
                "weight": weights  # You'd extract from matrix
            }
        },
        "knapsack": {
            "K": m,
            "capacities": capacities,
            "capacity_attr": "weight"
        },
        "constraints": [
            {
                "kind": "capacity",
                "attr": "weight",
                "limit": cap,
                "soft": False
            } for cap in capacities
        ],
        "objective": [
            {"attr": "value", "weight": 1.0}
        ]
    }
    
    return config
```

## References

1. Martello, S., & Toth, P. (1990). *Knapsack Problems: Algorithms and Computer Implementations*. Wiley.

2. Pisinger, D. (2005). "Where are the hard knapsack problems?" *Computers & Operations Research*, 32(9), 2271-2284.

3. Chu, P. C., & Beasley, J. E. (1998). "A genetic algorithm for the multidimensional knapsack problem." *Journal of heuristics*, 4(1), 63-86.

4. Koch, T., et al. (2011). "MIPLIB 2010." *Mathematical Programming Computation*, 3(2), 103-163.

## Next Steps

1. Create `scripts/download_benchmarks.sh` to fetch datasets
2. Create `examples/python/convert_benchmarks.py` to convert formats
3. Create `benchmarks/run_validation.py` to run systematic tests
4. Create comparison reports showing v2 performance vs known optima
5. Document results in a paper or technical report

This will provide verifiable evidence of your solver's effectiveness on problems that the optimization community recognizes and trusts.
