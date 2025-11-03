# Knapsack Solver Examples

This directory contains comprehensive examples demonstrating real-world applications of the knapsack solver.

## Directory Structure

```
examples/
├── python/                    # Python binding examples
│   ├── 01_basic_knapsack.py  # Classic knapsack problems
│   ├── 02_debt_portfolio.py  # Debt collection portfolio selection
│   ├── 03_investment_portfolio.py  # Investment/asset selection
│   ├── 04_pandas_integration.py    # Working with CSV data and DataFrames
│   ├── 05_visualization.py   # Plotting and analysis
│   └── 06_scout_mode.py      # Scout mode for exact solver integration
└── data/                      # Sample datasets
    ├── debts.csv
    ├── investments.csv
    └── projects.csv
```

## Prerequisites

### Basic Examples
```bash
pip install knapsack-solver
```

### Advanced Examples (with visualization)
```bash
pip install knapsack-solver pandas matplotlib
```

### Scout Mode Examples (with exact solvers)
```bash
pip install knapsack-solver gurobipy  # or other MIP solver
```

## Running Examples

```bash
# From the examples/python directory
cd examples/python

# Run any example
python3 01_basic_knapsack.py
python3 02_debt_portfolio.py
python3 03_investment_portfolio.py
python3 04_pandas_integration.py
python3 05_visualization.py
python3 06_scout_mode.py
```

## Example Descriptions

### 1. Basic Knapsack (`01_basic_knapsack.py`)
Classic knapsack problems with increasing complexity:
- Simple 0/1 knapsack
- Multiple knapsacks
- Custom configuration
- Performance benchmarking

### 2. Debt Portfolio Selection (`02_debt_portfolio.py`)
Real-world debt collection optimization:
- Expected value calculation (balance × probability × urgency)
- Staff hour constraints
- Legal budget considerations
- ROI analysis
- Age and credit score factors

### 3. Investment Portfolio (`03_investment_portfolio.py`)
Financial portfolio optimization:
- Risk-adjusted returns (Sharpe ratio)
- Sector diversification constraints
- Capital allocation
- ESG scoring integration
- Rebalancing scenarios

### 4. Pandas Integration (`04_pandas_integration.py`)
Working with real CSV data:
- Loading data from CSV files
- DataFrame preprocessing
- Feature engineering
- Result analysis and export
- Integration with existing data pipelines

### 5. Visualization (`05_visualization.py`)
Plotting and analysis:
- Selected vs. rejected items
- Pareto frontiers (value vs. effort)
- Constraint utilization charts
- ROI distributions
- Solution quality over time

### 6. Scout Mode (`06_scout_mode.py`)
Hybrid optimization with exact solvers:
- Active set identification
- Gurobi/CPLEX integration
- Warm start techniques
- Performance comparison (beam vs. exact vs. hybrid)
- Problem reduction metrics

## Sample Datasets

Sample CSV files are provided in `examples/data/`:

- `debts.csv`: 500 synthetic debt records
- `investments.csv`: 100 investment opportunities
- `projects.csv`: 50 project proposals with multiple constraints

## Use Case Scenarios

### Financial Services
- Debt portfolio prioritization
- Investment allocation
- Loan origination selection
- Credit line optimization

### Operations Research
- Resource allocation
- Project selection
- Workforce scheduling
- Inventory optimization

### Logistics
- Route selection
- Warehouse allocation
- Fleet optimization
- Load planning

### Healthcare
- Patient scheduling
- Resource allocation
- Treatment prioritization
- Equipment purchasing

## Performance Tips

1. **Start small**: Test with a subset of data first
2. **Tune beam width**: Typically 16-64 is sufficient
3. **Use scout mode**: For problems with >1000 items
4. **Enable dominance**: When many items are clearly inferior
5. **Iterate**: More iterations improve quality (diminishing returns after 5-10)

## Additional Examples

### 5. Visualization (`05_visualization.py`)

Visual analysis and plotting of knapsack solutions using matplotlib.

**Sub-examples:**
- Basic scatter plots (selected vs. rejected items)
- Value/weight ratio analysis (histograms, box plots)
- Capacity sensitivity analysis (multi-panel plots)
- Pareto frontier visualization (quality vs. speed trade-offs)
- Scout mode frequency heatmaps
- Category breakdown analysis (pie charts, bar charts)

**Dependencies:**
```bash
pip install matplotlib numpy
```

**When to Use:**
- Exploratory data analysis
- Presenting results to stakeholders
- Understanding solution patterns
- Parameter tuning and sensitivity analysis

---

### 6. Exact Solver Integration (`06_exact_solver_integration.py`)

Integration with commercial MIP solvers for optimal solutions and hybrid approaches.

**Sub-examples:**
- Solver comparison (Beam vs. Gurobi vs. PuLP)
- Warm start techniques (using beam solution to accelerate MIP)
- Hybrid approach (scout mode + exact solving)
- Large-scale benchmarking (100-1000+ items)
- Iterative refinement (alternating beam + exact)

**Dependencies:**
```bash
# Option 1: Gurobi (commercial, requires license)
pip install gurobipy

# Option 2: PuLP (free, uses open-source solvers)
pip install pulp
```

**When to Use:**
- Optimality guarantees required
- Critical business decisions
- Benchmarking solution quality
- Research and algorithm comparison
- Large problems needing reduction

---

## Contributing

Have a great example use case? Please contribute!

1. Fork the repository
2. Add your example to `examples/python/`
3. Update this README
4. Submit a pull request

## License

All examples are provided under the MIT license. Feel free to use and adapt for your needs.
