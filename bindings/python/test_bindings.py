#!/usr/bin/env python3
"""
Quick test suite for knapsack Python bindings.
"""

import sys
import os

# Add build directory to path
build_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'build')
sys.path.insert(0, os.path.abspath(build_dir))

import knapsack


def test_basic_solve():
    """Test basic knapsack solving."""
    values = [60.0, 100.0, 120.0]
    weights = [10.0, 20.0, 30.0]
    capacity = 50.0
    
    solution = knapsack.solve(values, weights, capacity)
    
    assert solution.best_value > 0, "Should have positive value"
    assert len(solution.selected_indices) > 0, "Should select some items"
    assert all(i >= 0 and i < len(values) for i in solution.selected_indices), "Indices should be valid"
    
    # Check capacity constraint
    total_weight = sum(weights[i] for i in solution.selected_indices)
    assert total_weight <= capacity, f"Weight {total_weight} exceeds capacity {capacity}"
    
    print("✓ test_basic_solve passed")


def test_empty_knapsack():
    """Test with zero capacity (should select nothing)."""
    values = [60.0, 100.0, 120.0]
    weights = [10.0, 20.0, 30.0]
    capacity = 0.0
    
    solution = knapsack.solve(values, weights, capacity)
    
    # With zero capacity, we expect either no selection or zero value
    total_weight = sum(weights[i] for i in solution.selected_indices)
    assert total_weight == 0, "Should select nothing with zero capacity"
    
    print("✓ test_empty_knapsack passed")


def test_all_items_fit():
    """Test when all items fit in knapsack."""
    values = [60.0, 100.0, 120.0]
    weights = [10.0, 20.0, 30.0]
    capacity = 100.0  # More than enough
    
    solution = knapsack.solve(values, weights, capacity)
    
    # Should select all items
    assert len(solution.selected_indices) == len(values), "Should select all items when they all fit"
    assert abs(solution.best_value - sum(values)) < 1e-6, "Should get total value"
    
    print("✓ test_all_items_fit passed")


def test_scout_mode():
    """Test scout mode for active set identification."""
    values = [60.0, 100.0, 120.0, 50.0, 80.0, 90.0]
    weights = [10.0, 20.0, 30.0, 15.0, 25.0, 20.0]
    capacity = 50.0
    
    result = knapsack.solve_scout(values, weights, capacity, 
                                   {"scout_threshold": 0.5, "scout_top_k": 8})
    
    assert result.original_item_count == len(values), "Should track original count"
    assert result.active_item_count <= result.original_item_count, "Active count should be <= original"
    assert len(result.active_items) == result.active_item_count, "Active items list should match count"
    assert len(result.item_frequency) == result.original_item_count, "Frequency list should match original count"
    assert all(f >= 0.0 and f <= 1.0 for f in result.item_frequency), "Frequencies should be in [0, 1]"
    
    print(f"✓ test_scout_mode passed (reduced from {result.original_item_count} to {result.active_item_count} items)")


def test_custom_config():
    """Test with custom configuration."""
    values = [60.0, 100.0, 120.0, 50.0]
    weights = [10.0, 20.0, 30.0, 15.0]
    capacity = 50.0
    
    config = {
        "beam_width": 32,
        "iters": 5,
        "seed": 42,
        "debug": False
    }
    
    solution = knapsack.solve(values, weights, capacity, config)
    
    assert solution.best_value > 0, "Should have positive value with custom config"
    assert solution.solve_time_ms >= 0, "Solve time should be non-negative"
    
    print("✓ test_custom_config passed")


def test_version():
    """Test that version attribute exists."""
    assert hasattr(knapsack, '__version__'), "Module should have __version__"
    assert isinstance(knapsack.__version__, str), "Version should be a string"
    print(f"✓ test_version passed (version: {knapsack.__version__})")


def main():
    """Run all tests."""
    print("Running knapsack Python binding tests...\n")
    
    try:
        test_version()
        test_basic_solve()
        test_empty_knapsack()
        test_all_items_fit()
        test_scout_mode()
        test_custom_config()
        
        print("\n" + "=" * 60)
        print("All tests passed! ✅")
        print("=" * 60)
        return 0
        
    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
        return 1
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
