# Python Bindings for Knapsack Solver

Python bindings for the high-performance knapsack solver with beam search and scout mode.

## Features

- **Fast beam search** solver for knapsack problems
- **Scout mode** to identify "active items" for exact solver integration
- Pythonic API with NumPy-compatible arrays
- Metal acceleration on macOS, CPU fallback on other platforms
- Type hints and comprehensive docstrings

## Installation

### From Source

```bash
# Clone the repository with submodules
git clone --recursive https://github.com/yourusername/knapsack.git
cd knapsack

# Build the Python module
mkdir -p build && cd build
cmake .. -DBUILD_PYTHON_BINDINGS=ON
cmake --build . --target knapsack_py

# Install (or add build directory to PYTHONPATH)
# Option 1: Install with pip
cd ..
pip install .

# Option 2: Use directly from build directory
export PYTHONPATH=/path/to/knapsack/build:$PYTHONPATH
```

## Quick Start

```python
import knapsack

# Define your problem
values = [60, 100, 120, 50, 80, 90]
weights = [10, 20, 30, 15, 25, 20]
capacity = 50

# Solve it!
solution = knapsack.solve(values, weights, capacity)

print(f"Best value: {solution.best_value}")
print(f"Selected items: {solution.selected_indices}")
print(f"Solve time: {solution.solve_time_ms:.2f} ms")
```

## API Reference

### `knapsack.solve(values, weights, capacity, config={})`

Solve a knapsack problem using beam search.

**Parameters:**
- `values` (list of float): Value of each item
- `weights` (list of float): Weight of each item
- `capacity` (float): Knapsack capacity constraint
- `config` (dict, optional): Configuration options
  - `beam_width` (int, default 16): Beam search width
  - `iters` (int, default 3): Number of iterations
  - `seed` (int, default 1234): Random seed
  - `debug` (bool, default False): Enable debug logging
  - `enable_dominance` (bool, default False): Enable dominance filters

**Returns:**
- `Solution` object with:
  - `best_value` (float): Optimal objective value
  - `selected_indices` (list[int]): Indices of selected items
  - `solve_time_ms` (float): Solve time in milliseconds

**Example:**
```python
config = {"beam_width": 32, "iters": 5}
solution = knapsack.solve(values, weights, capacity, config)
```

### `knapsack.solve_scout(values, weights, capacity, config={})`

Use beam search as a "data scout" to identify frequently selected items.

This function analyzes which items appear in high-quality solutions and returns
a filtered "active set" that can be passed to an exact solver (Gurobi, CPLEX, SCIP)
for guaranteed optimality on a smaller problem.

**Parameters:**
- `values`, `weights`, `capacity`: Same as `solve()`
- `config` (dict, optional): Same as `solve()`, plus:
  - `scout_threshold` (float, default 0.5): Minimum frequency to include item (0.0-1.0)
  - `scout_top_k` (int, default 8): Number of top candidates to analyze

**Returns:**
- `ScoutResult` object with:
  - `active_items` (list[int]): Indices of frequently selected items
  - `item_frequency` (list[float]): Selection frequency for each item (0.0-1.0)
  - `original_item_count` (int): Number of items in original problem
  - `active_item_count` (int): Number of items in active set
  - `objective` (float): Best objective value found
  - `best_select` (list[int]): Binary selection vector (0/1 for each item)
  - `solve_time_ms` (float): Beam search time
  - `filter_time_ms` (float): Active set filtering time

**Example:**
```python
# Use scout mode to reduce problem size
scout_config = {"scout_threshold": 0.5, "scout_top_k": 8}
result = knapsack.solve_scout(values, weights, capacity, scout_config)

# Extract active items
active_values = [values[i] for i in result.active_items]
active_weights = [weights[i] for i in result.active_items]

# Pass to exact solver (Gurobi, SCIP, etc.)
# ... solve reduced problem optimally ...

# Map solution back to original indices
print(f"Reduced from {result.original_item_count} to {result.active_item_count} items")
print(f"Reduction: {100 * (1 - result.active_item_count / result.original_item_count):.1f}%")
```

## Scout Mode Workflow

Scout mode enables a hybrid approach: use beam search to quickly identify promising items,
then pass a reduced problem to an exact solver for guaranteed optimality.

```python
import knapsack

# 1. Run scout mode
result = knapsack.solve_scout(values, weights, capacity)

# 2. Extract filtered problem
filtered_values = [values[i] for i in result.active_items]
filtered_weights = [weights[i] for i in result.active_items]

# 3. Solve with exact solver (e.g., Gurobi)
# import gurobipy as gp
# model = gp.Model()
# x = model.addVars(len(filtered_values), vtype=gp.GRB.BINARY)
# model.setObjective(sum(filtered_values[i] * x[i] for i in range(len(filtered_values))), gp.GRB.MAXIMIZE)
# model.addConstr(sum(filtered_weights[i] * x[i] for i in range(len(filtered_values))) <= capacity)
# model.optimize()

# 4. Map solution back to original indices
# optimal_indices = [result.active_items[i] for i in range(len(filtered_values)) if x[i].X > 0.5]
```

## Configuration Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `beam_width` | int | 16 | Number of states to keep at each level |
| `iters` | int | 3 | Number of beam search iterations |
| `seed` | int | 1234 | Random seed for reproducibility |
| `debug` | bool | False | Enable debug logging |
| `enable_dominance` | bool | False | Enable dominance pruning |
| `scout_threshold` | float | 0.5 | Min frequency for active set (scout mode only) |
| `scout_top_k` | int | 8 | Top candidates to analyze (scout mode only) |

## Performance Tips

1. **Start with default beam_width (16)** - it's usually sufficient
2. **Increase iters for better quality** - diminishing returns after 5-10
3. **Use scout mode for large problems** - reduces exact solver time by 50-90%
4. **Enable dominance filters** for problems with many dominated items

## Examples

See `bindings/python/example.py` for comprehensive examples including:
- Basic knapsack solving
- Custom configuration
- Scout mode for exact solver handoff
- Filtering and result mapping

Run the demo:
```bash
cd build
PYTHONPATH=$PWD:$PYTHONPATH python3 ../bindings/python/example.py
```

## Requirements

- Python 3.7+
- C++17 compiler
- CMake 3.18+
- macOS: Metal support (automatic)
- Linux: Optional CUDA support

## License

MIT License - see LICENSE file for details

## Citation

If you use this solver in academic work, please cite:

```bibtex
@software{knapsack_solver,
  title = {Knapsack Solver with Beam Search and Scout Mode},
  author = {Your Name},
  year = {2025},
  url = {https://github.com/yourusername/knapsack}
}
```
