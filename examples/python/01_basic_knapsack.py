#!/usr/bin/env python3
"""
Example 1: Basic Knapsack Problems

Classic knapsack optimization scenarios demonstrating the Python API.
"""

import sys
import os
import time

# Add build directory to path for development
build_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'build')
if os.path.exists(build_dir):
    sys.path.insert(0, os.path.abspath(build_dir))

import knapsack


def example_1_classic_knapsack():
    """Classic 0/1 knapsack problem"""
    
    print("\n" + "="*80)
    print("EXAMPLE 1: Classic 0/1 Knapsack")
    print("="*80)
    
    # Classic problem: maximize value within weight constraint
    items = [
        {"name": "Laptop", "value": 500, "weight": 5},
        {"name": "Camera", "value": 300, "weight": 3},
        {"name": "Book", "value": 100, "weight": 1},
        {"name": "Phone", "value": 400, "weight": 2},
        {"name": "Tablet", "value": 250, "weight": 2},
        {"name": "Headphones", "value": 150, "weight": 1},
    ]
    
    values = [item["value"] for item in items]
    weights = [item["weight"] for item in items]
    capacity = 8.0
    
    print(f"\nItems available: {len(items)}")
    print(f"Capacity: {capacity} kg")
    print("\nItem list:")
    for item in items:
        print(f"  {item['name']:<15} Value: ${item['value']:<4} Weight: {item['weight']} kg")
    
    # Solve
    solution = knapsack.solve(values, weights, capacity)
    
    # Display results
    selected_items = [items[i] for i in solution.selected_indices]
    total_value = sum(item["value"] for item in selected_items)
    total_weight = sum(item["weight"] for item in selected_items)
    
    print(f"\n{'SOLUTION':-^80}")
    print(f"Selected items: {len(selected_items)}")
    print(f"Total value: ${total_value}")
    print(f"Total weight: {total_weight} / {capacity} kg")
    print(f"Solve time: {solution.solve_time_ms:.2f} ms")
    
    print("\nSelected items:")
    for item in selected_items:
        print(f"  ✓ {item['name']:<15} ${item['value']:<4} {item['weight']} kg")
    
    print("\nNot selected:")
    not_selected = [item for i, item in enumerate(items) if i not in solution.selected_indices]
    for item in not_selected:
        print(f"  ✗ {item['name']:<15} ${item['value']:<4} {item['weight']} kg")


def example_2_larger_problem():
    """Larger problem with custom configuration"""
    
    print("\n" + "="*80)
    print("EXAMPLE 2: Larger Problem with Custom Configuration")
    print("="*80)
    
    # Generate 50 items
    import random
    random.seed(42)
    
    num_items = 50
    values = [random.randint(10, 1000) for _ in range(num_items)]
    weights = [random.randint(1, 50) for _ in range(num_items)]
    capacity = sum(weights) * 0.3  # Can fit ~30% of items
    
    print(f"\nItems: {num_items}")
    print(f"Capacity: {capacity:.1f}")
    print(f"Total weight if all selected: {sum(weights)}")
    print(f"Total value if all selected: {sum(values)}")
    
    # Test different configurations
    configs = [
        {"name": "Fast (beam=16)", "beam_width": 16, "iters": 3},
        {"name": "Balanced (beam=32)", "beam_width": 32, "iters": 5},
        {"name": "Quality (beam=64)", "beam_width": 64, "iters": 10},
    ]
    
    print(f"\n{'Configuration Comparison':-^80}")
    print(f"{'Config':<25} {'Items':<8} {'Value':<12} {'Time (ms)':<12} {'Value/ms'}")
    print("-" * 80)
    
    for config in configs:
        start = time.time()
        solution = knapsack.solve(
            values, weights, capacity,
            {"beam_width": config["beam_width"], "iters": config["iters"]}
        )
        elapsed_ms = (time.time() - start) * 1000
        
        value_per_ms = solution.best_value / elapsed_ms if elapsed_ms > 0 else 0
        
        print(f"{config['name']:<25} "
              f"{len(solution.selected_indices):<8} "
              f"{solution.best_value:<12.0f} "
              f"{elapsed_ms:<12.2f} "
              f"{value_per_ms:.2f}")


def example_3_greedy_comparison():
    """Compare knapsack solver with greedy heuristic"""
    
    print("\n" + "="*80)
    print("EXAMPLE 3: Comparison with Greedy Heuristic")
    print("="*80)
    
    # Problem where greedy fails
    values = [60.0, 100.0, 120.0, 80.0, 90.0, 110.0]
    weights = [10.0, 20.0, 30.0, 15.0, 18.0, 25.0]
    capacity = 50.0
    
    print(f"\nItems: {len(values)}")
    print(f"Capacity: {capacity}")
    
    # Greedy by value/weight ratio
    ratios = [(i, values[i]/weights[i]) for i in range(len(values))]
    ratios.sort(key=lambda x: x[1], reverse=True)
    
    greedy_selected = []
    greedy_weight = 0
    greedy_value = 0
    
    print(f"\n{'Greedy Selection (by value/weight ratio)':-^80}")
    print(f"{'Item':<6} {'Value':<10} {'Weight':<10} {'Ratio':<10} {'Action'}")
    print("-" * 80)
    
    for idx, ratio in ratios:
        if greedy_weight + weights[idx] <= capacity:
            greedy_selected.append(idx)
            greedy_weight += weights[idx]
            greedy_value += values[idx]
            action = "✓ Selected"
        else:
            action = "✗ Skipped (exceeds capacity)"
        
        print(f"{idx:<6} {values[idx]:<10.1f} {weights[idx]:<10.1f} "
              f"{ratio:<10.2f} {action}")
    
    print(f"\nGreedy result: ${greedy_value:.1f} (weight: {greedy_weight:.1f})")
    
    # Beam search solution
    solution = knapsack.solve(values, weights, capacity)
    selected_weight = sum(weights[i] for i in solution.selected_indices)
    
    print(f"\n{'Beam Search Solution':-^80}")
    print(f"Selected items: {sorted(solution.selected_indices)}")
    print(f"Total value: ${solution.best_value:.1f}")
    print(f"Total weight: {selected_weight:.1f} / {capacity}")
    print(f"Solve time: {solution.solve_time_ms:.2f} ms")
    
    improvement = ((solution.best_value - greedy_value) / greedy_value * 100)
    print(f"\nImprovement over greedy: {improvement:+.1f}%")


def example_4_sensitivity_analysis():
    """Analyze sensitivity to capacity constraints"""
    
    print("\n" + "="*80)
    print("EXAMPLE 4: Capacity Sensitivity Analysis")
    print("="*80)
    
    # Fixed items
    values = [60.0, 100.0, 120.0, 80.0, 90.0, 110.0, 70.0, 95.0]
    weights = [10.0, 20.0, 30.0, 15.0, 18.0, 25.0, 12.0, 22.0]
    
    # Vary capacity from 20% to 100% of total weight
    total_weight = sum(weights)
    capacities = [total_weight * pct for pct in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]]
    
    print(f"\nTotal weight available: {total_weight:.1f}")
    print(f"Total value available: {sum(values):.1f}")
    
    print(f"\n{'Capacity Analysis':-^80}")
    print(f"{'Capacity':<12} {'% of Total':<12} {'Items':<8} {'Value':<12} {'Utilization'}")
    print("-" * 80)
    
    for capacity in capacities:
        solution = knapsack.solve(values, weights, capacity)
        selected_weight = sum(weights[i] for i in solution.selected_indices)
        utilization = selected_weight / capacity * 100
        pct_total = capacity / total_weight * 100
        
        print(f"{capacity:<12.1f} {pct_total:<12.1f} "
              f"{len(solution.selected_indices):<8} "
              f"{solution.best_value:<12.1f} "
              f"{utilization:.1f}%")


def example_5_edge_cases():
    """Test edge cases and boundary conditions"""
    
    print("\n" + "="*80)
    print("EXAMPLE 5: Edge Cases and Boundary Conditions")
    print("="*80)
    
    # Test 1: Empty knapsack (zero capacity)
    print("\nTest 1: Zero capacity")
    solution = knapsack.solve([60, 100, 120], [10, 20, 30], 0.0)
    print(f"  Selected: {len(solution.selected_indices)} items")
    print(f"  Value: {solution.best_value}")
    assert len(solution.selected_indices) == 0 or solution.best_value == 0
    print("  ✓ Passed")
    
    # Test 2: All items fit
    print("\nTest 2: All items fit")
    values = [60, 100, 120]
    weights = [10, 20, 30]
    capacity = 100.0
    solution = knapsack.solve(values, weights, capacity)
    print(f"  Selected: {len(solution.selected_indices)} / {len(values)} items")
    print(f"  Value: {solution.best_value} (expected: {sum(values)})")
    assert len(solution.selected_indices) == len(values)
    print("  ✓ Passed")
    
    # Test 3: Single item
    print("\nTest 3: Single item")
    solution = knapsack.solve([100.0], [10.0], 20.0)
    print(f"  Selected: {len(solution.selected_indices)} items")
    print(f"  Value: {solution.best_value}")
    assert len(solution.selected_indices) == 1
    print("  ✓ Passed")
    
    # Test 4: Identical items
    print("\nTest 4: Identical items")
    solution = knapsack.solve([50]*10, [10]*10, 30.0)
    print(f"  Selected: {len(solution.selected_indices)} items")
    print(f"  Value: {solution.best_value}")
    print("  ✓ Passed")
    
    print("\n✅ All edge cases passed!")


def main():
    """Run all basic examples"""
    
    print("╔" + "═"*78 + "╗")
    print("║" + " BASIC KNAPSACK EXAMPLES ".center(78) + "║")
    print("╚" + "═"*78 + "╝")
    print(f"\nKnapsack solver version: {knapsack.__version__}")
    
    example_1_classic_knapsack()
    example_2_larger_problem()
    example_3_greedy_comparison()
    example_4_sensitivity_analysis()
    example_5_edge_cases()
    
    print("\n" + "="*80)
    print("✅ All examples completed successfully!")
    print("="*80)


if __name__ == "__main__":
    main()
