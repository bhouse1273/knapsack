#!/usr/bin/env python3
"""
Example usage of the knapsack Python bindings.

This demonstrates both regular solve() and solve_scout() modes.
"""

import knapsack

def main():
    print("=" * 60)
    print("Knapsack Solver - Python Binding Demo")
    print("=" * 60)
    
    # Define a simple knapsack problem
    values = [60, 100, 120, 50, 80, 90]
    weights = [10, 20, 30, 15, 25, 20]
    capacity = 50
    
    print(f"\nProblem:")
    print(f"  Items: {len(values)}")
    print(f"  Values:  {values}")
    print(f"  Weights: {weights}")
    print(f"  Capacity: {capacity}")
    
    # -----------------------------------------------------------------
    # Example 1: Basic solve with default config
    # -----------------------------------------------------------------
    print("\n" + "-" * 60)
    print("Example 1: Basic solve() with default configuration")
    print("-" * 60)
    
    solution = knapsack.solve(values, weights, capacity)
    
    print(f"Solution: {solution}")
    print(f"  Best value: {solution.best_value}")
    print(f"  Selected items: {solution.selected_indices}")
    print(f"  Solve time: {solution.solve_time_ms:.2f} ms")
    
    # Calculate weight from selected items
    total_weight = sum(weights[i] for i in solution.selected_indices)
    print(f"  Total weight: {total_weight} / {capacity}")
    
    # -----------------------------------------------------------------
    # Example 2: Solve with custom config
    # -----------------------------------------------------------------
    print("\n" + "-" * 60)
    print("Example 2: solve() with custom beam_width")
    print("-" * 60)
    
    config = {
        "beam_width": 50,
        "iters": 5,
        "debug": False
    }
    
    solution2 = knapsack.solve(values, weights, capacity, config)
    print(f"Solution: {solution2}")
    print(f"  Best value: {solution2.best_value}")
    print(f"  Solve time: {solution2.solve_time_ms:.2f} ms")
    
    # -----------------------------------------------------------------
    # Example 3: Scout mode - identify active items
    # -----------------------------------------------------------------
    print("\n" + "-" * 60)
    print("Example 3: solve_scout() - Data scout for exact solvers")
    print("-" * 60)
    
    scout_config = {
        "beam_width": 32,
        "iters": 3,
        "scout_threshold": 0.5,  # Items appearing in >50% of top candidates
        "scout_top_k": 8,        # Analyze top 8 candidates
        "debug": False
    }
    
    scout_result = knapsack.solve_scout(values, weights, capacity, scout_config)
    
    print(f"Scout result: {scout_result}")
    print(f"\n  Active items: {scout_result.active_items}")
    print(f"  Item frequencies: {[f'{x:.2f}' for x in scout_result.item_frequency]}")
    print(f"  Original item count: {scout_result.original_item_count}")
    print(f"  Active item count: {scout_result.active_item_count}")
    
    reduction_pct = 100.0 * (1.0 - scout_result.active_item_count / scout_result.original_item_count)
    print(f"  Reduction: {reduction_pct:.1f}%")
    print(f"  Scout solve time: {scout_result.solve_time_ms:.2f} ms")
    print(f"  Filter time: {scout_result.filter_time_ms:.2f} ms")
    
    print(f"\n  Beam search best value: {scout_result.objective}")
    print(f"  Beam search total: {scout_result.total}")
    print(f"  Beam search penalty: {scout_result.penalty}")
    print(f"  Items selected in best: {sum(scout_result.best_select)}")
    
    # -----------------------------------------------------------------
    # Example 4: Using scout mode to feed exact solver (conceptual)
    # -----------------------------------------------------------------
    print("\n" + "-" * 60)
    print("Example 4: Scout mode workflow for exact solvers")
    print("-" * 60)
    
    print("\n  Workflow:")
    print("  1. Run scout mode to identify active items")
    print(f"     → Active items: {scout_result.active_items}")
    print(f"     → Reduction: {reduction_pct:.1f}%")
    print("\n  2. Filter the problem data")
    print("     → Extract values[i], weights[i] for i in active_items")
    print("\n  3. Pass filtered data to exact solver (Gurobi, SCIP, CPLEX)")
    print("     → Solve smaller problem optimally")
    print("\n  4. Map solution back to original indices")
    print("     → Convert active_items[j] back to original item indices")
    
    # Show filtered data
    filtered_values = [values[i] for i in scout_result.active_items]
    filtered_weights = [weights[i] for i in scout_result.active_items]
    
    print(f"\n  Filtered problem:")
    print(f"    Original: {len(values)} items")
    print(f"    Filtered: {len(filtered_values)} items")
    print(f"    Filtered values:  {filtered_values}")
    print(f"    Filtered weights: {filtered_weights}")
    print(f"    Capacity: {capacity} (unchanged)")
    
    print("\n" + "=" * 60)
    print("Demo complete!")
    print("=" * 60)
    print(f"\nPython binding version: {knapsack.__version__}")


if __name__ == "__main__":
    main()
