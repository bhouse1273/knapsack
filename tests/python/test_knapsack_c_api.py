#!/usr/bin/env python3
"""
Python tests for knapsack C API - Ready for chariot-ecosystem replication

These tests validate the knapsack() C function call with various argument patterns.
Use these as a reference to debug CGO integration issues in go-chariot.

Requirements:
    pip install numpy

Build the library first:
    cd knapsack-library
    mkdir -p build && cd build
    cmake .. -DCMAKE_BUILD_TYPE=Release -DBUILD_CPU_ONLY=ON
    cmake --build . --target knapsack -j8
"""

import ctypes
import json
import os
import sys
from pathlib import Path

import numpy as np

# Find the library
LIB_PATHS = [
    "knapsack-library/build/libknapsack.dylib",  # macOS
    "knapsack-library/build/libknapsack.so",      # Linux
    "build/libknapsack.dylib",
    "build/libknapsack.so",
    "/usr/local/lib/libknapsack_metal.dylib",
    "/usr/local/lib/libknapsack_cpu.dylib",
]

lib_path = None
for path in LIB_PATHS:
    if os.path.exists(path):
        lib_path = path
        break

if not lib_path:
    print("ERROR: Could not find knapsack library. Build it first:")
    print("  cd knapsack-library && mkdir -p build && cd build")
    print("  cmake .. -DCMAKE_BUILD_TYPE=Release -DBUILD_CPU_ONLY=ON")
    print("  cmake --build . --target knapsack -j8")
    sys.exit(1)

print(f"Loading library: {lib_path}")
lib = ctypes.CDLL(lib_path)

# Define the C function signature
# int knapsack(int n, int* weights, int* values, int capacity, int* selection)
lib.knapsack.argtypes = [
    ctypes.c_int,                    # n (number of items)
    ctypes.POINTER(ctypes.c_int),    # weights array
    ctypes.POINTER(ctypes.c_int),    # values array
    ctypes.c_int,                    # capacity
    ctypes.POINTER(ctypes.c_int),    # selection output array
]
lib.knapsack.restype = ctypes.c_int  # returns total value


def test_basic_knapsack():
    """Test 1: Basic knapsack with simple integers"""
    print("\n=== Test 1: Basic Knapsack ===")
    
    n = 5
    weights = np.array([2, 3, 4, 5, 9], dtype=np.int32)
    values = np.array([3, 4, 5, 8, 10], dtype=np.int32)
    capacity = 10
    selection = np.zeros(n, dtype=np.int32)
    
    print(f"Items: {n}")
    print(f"Weights: {weights.tolist()}")
    print(f"Values: {values.tolist()}")
    print(f"Capacity: {capacity}")
    
    # Call the C function
    total_value = lib.knapsack(
        n,
        weights.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        values.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        capacity,
        selection.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
    )
    
    print(f"Result: Total value = {total_value}")
    print(f"Selection: {selection.tolist()}")
    print(f"Selected items: {[i for i in range(n) if selection[i] == 1]}")
    
    selected_weight = sum(weights[i] for i in range(n) if selection[i] == 1)
    selected_value = sum(values[i] for i in range(n) if selection[i] == 1)
    
    print(f"Total weight used: {selected_weight}/{capacity}")
    print(f"Total value: {selected_value}")
    
    assert total_value == selected_value, "Return value should match sum of selected values"
    assert selected_weight <= capacity, "Total weight should not exceed capacity"
    
    print("✅ PASSED")
    return True


def test_edge_case_single_item():
    """Test 2: Single item that fits"""
    print("\n=== Test 2: Single Item ===")
    
    n = 1
    weights = np.array([5], dtype=np.int32)
    values = np.array([10], dtype=np.int32)
    capacity = 10
    selection = np.zeros(n, dtype=np.int32)
    
    print(f"Items: {n}")
    print(f"Weights: {weights.tolist()}")
    print(f"Values: {values.tolist()}")
    print(f"Capacity: {capacity}")
    
    total_value = lib.knapsack(
        n,
        weights.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        values.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        capacity,
        selection.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
    )
    
    print(f"Result: Total value = {total_value}")
    print(f"Selection: {selection.tolist()}")
    
    assert total_value == 10, "Should select the single item"
    assert selection[0] == 1, "Item should be selected"
    
    print("✅ PASSED")
    return True


def test_edge_case_single_item_too_heavy():
    """Test 3: Single item that doesn't fit"""
    print("\n=== Test 3: Single Item Too Heavy ===")
    
    n = 1
    weights = np.array([15], dtype=np.int32)
    values = np.array([10], dtype=np.int32)
    capacity = 10
    selection = np.zeros(n, dtype=np.int32)
    
    print(f"Items: {n}")
    print(f"Weights: {weights.tolist()}")
    print(f"Values: {values.tolist()}")
    print(f"Capacity: {capacity}")
    
    total_value = lib.knapsack(
        n,
        weights.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        values.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        capacity,
        selection.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
    )
    
    print(f"Result: Total value = {total_value}")
    print(f"Selection: {selection.tolist()}")
    
    assert total_value == 0, "Should select nothing (item too heavy)"
    assert selection[0] == 0, "Item should not be selected"
    
    print("✅ PASSED")
    return True


def test_zero_capacity():
    """Test 4: Zero capacity knapsack"""
    print("\n=== Test 4: Zero Capacity ===")
    
    n = 3
    weights = np.array([1, 2, 3], dtype=np.int32)
    values = np.array([10, 20, 30], dtype=np.int32)
    capacity = 0
    selection = np.zeros(n, dtype=np.int32)
    
    print(f"Items: {n}")
    print(f"Weights: {weights.tolist()}")
    print(f"Values: {values.tolist()}")
    print(f"Capacity: {capacity}")
    
    total_value = lib.knapsack(
        n,
        weights.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        values.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        capacity,
        selection.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
    )
    
    print(f"Result: Total value = {total_value}")
    print(f"Selection: {selection.tolist()}")
    
    assert total_value == 0, "Zero capacity should yield zero value"
    assert all(s == 0 for s in selection), "No items should be selected"
    
    print("✅ PASSED")
    return True


def test_large_problem():
    """Test 5: Larger problem (20 items)"""
    print("\n=== Test 5: Large Problem (20 items) ===")
    
    n = 20
    np.random.seed(42)  # Reproducible
    weights = np.random.randint(1, 20, size=n, dtype=np.int32)
    values = np.random.randint(1, 50, size=n, dtype=np.int32)
    capacity = 50
    selection = np.zeros(n, dtype=np.int32)
    
    print(f"Items: {n}")
    print(f"Capacity: {capacity}")
    print(f"Weights: {weights.tolist()}")
    print(f"Values: {values.tolist()}")
    
    total_value = lib.knapsack(
        n,
        weights.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        values.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        capacity,
        selection.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
    )
    
    print(f"Result: Total value = {total_value}")
    print(f"Selection: {selection.tolist()}")
    
    selected_indices = [i for i in range(n) if selection[i] == 1]
    selected_weight = sum(weights[i] for i in selected_indices)
    selected_value = sum(values[i] for i in selected_indices)
    
    print(f"Selected items: {selected_indices}")
    print(f"Total weight used: {selected_weight}/{capacity}")
    print(f"Total value: {selected_value}")
    
    assert total_value == selected_value, "Return value should match sum"
    assert selected_weight <= capacity, "Should respect capacity"
    
    print("✅ PASSED")
    return True


def test_memory_alignment():
    """Test 6: Verify memory alignment doesn't cause issues"""
    print("\n=== Test 6: Memory Alignment Test ===")
    
    # Create arrays with potential alignment issues
    n = 7  # Odd number
    weights = np.array([1, 2, 3, 4, 5, 6, 7], dtype=np.int32)
    values = np.array([7, 6, 5, 4, 3, 2, 1], dtype=np.int32)
    capacity = 15
    selection = np.zeros(n, dtype=np.int32)
    
    print(f"Items: {n}")
    print(f"Weights: {weights.tolist()}")
    print(f"Values: {values.tolist()}")
    print(f"Capacity: {capacity}")
    print(f"Weights address: {hex(weights.ctypes.data)}")
    print(f"Values address: {hex(values.ctypes.data)}")
    print(f"Selection address: {hex(selection.ctypes.data)}")
    
    total_value = lib.knapsack(
        n,
        weights.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        values.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        capacity,
        selection.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
    )
    
    print(f"Result: Total value = {total_value}")
    print(f"Selection: {selection.tolist()}")
    
    selected_weight = sum(weights[i] for i in range(n) if selection[i] == 1)
    selected_value = sum(values[i] for i in range(n) if selection[i] == 1)
    
    print(f"Total weight used: {selected_weight}/{capacity}")
    print(f"Total value: {selected_value}")
    
    assert total_value == selected_value
    assert selected_weight <= capacity
    
    print("✅ PASSED")
    return True


def test_comparison_with_greedy():
    """Test 7: Compare result with greedy approximation"""
    print("\n=== Test 7: Comparison with Greedy ===")
    
    n = 6
    weights = np.array([10, 20, 30, 40, 50, 60], dtype=np.int32)
    values = np.array([60, 100, 120, 160, 200, 240], dtype=np.int32)
    capacity = 100
    selection = np.zeros(n, dtype=np.int32)
    
    print(f"Items: {n}")
    print(f"Weights: {weights.tolist()}")
    print(f"Values: {values.tolist()}")
    print(f"Capacity: {capacity}")
    
    # Greedy by value/weight ratio
    ratios = values.astype(float) / weights.astype(float)
    greedy_indices = sorted(range(n), key=lambda i: ratios[i], reverse=True)
    greedy_weight = 0
    greedy_value = 0
    greedy_selection = [0] * n
    
    for i in greedy_indices:
        if greedy_weight + weights[i] <= capacity:
            greedy_selection[i] = 1
            greedy_weight += weights[i]
            greedy_value += values[i]
    
    print(f"Greedy selection: {greedy_selection}")
    print(f"Greedy value: {greedy_value} (weight: {greedy_weight})")
    
    # Optimal solution
    total_value = lib.knapsack(
        n,
        weights.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        values.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        capacity,
        selection.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
    )
    
    print(f"Optimal selection: {selection.tolist()}")
    print(f"Optimal value: {total_value}")
    
    selected_weight = sum(weights[i] for i in range(n) if selection[i] == 1)
    print(f"Optimal weight: {selected_weight}")
    
    # Optimal should be >= greedy
    assert total_value >= greedy_value, f"Optimal ({total_value}) should be >= greedy ({greedy_value})"
    assert selected_weight <= capacity
    
    print("✅ PASSED")
    return True


def generate_go_test_case(test_name, n, weights, values, capacity):
    """Helper to generate Go/CGO test code"""
    print(f"\n=== Go/CGO Test Case for: {test_name} ===")
    print("```go")
    print(f"// Test case: {test_name}")
    print(f"n := C.int({n})")
    print(f"weights := []C.int{{{', '.join(f'C.int({w})' for w in weights)}}}")
    print(f"values := []C.int{{{', '.join(f'C.int({v})' for v in values)}}}")
    print(f"capacity := C.int({capacity})")
    print(f"selection := make([]C.int, {n})")
    print()
    print("totalValue := C.knapsack(")
    print("    n,")
    print("    (*C.int)(unsafe.Pointer(&weights[0])),")
    print("    (*C.int)(unsafe.Pointer(&values[0])),")
    print("    capacity,")
    print("    (*C.int)(unsafe.Pointer(&selection[0])),")
    print(")")
    print()
    print(f"// Expected behavior:")
    print(f"// - totalValue should be > 0 if any items fit")
    print(f"// - Sum of selected weights <= {capacity}")
    print(f"// - Sum of selected values == totalValue")
    print("```")


def main():
    """Run all tests and generate Go examples"""
    print("=" * 60)
    print("KNAPSACK C API VALIDATION TESTS")
    print("=" * 60)
    print(f"Library: {lib_path}")
    
    tests = [
        ("Basic Knapsack", test_basic_knapsack),
        ("Single Item", test_edge_case_single_item),
        ("Single Item Too Heavy", test_edge_case_single_item_too_heavy),
        ("Zero Capacity", test_zero_capacity),
        ("Large Problem", test_large_problem),
        ("Memory Alignment", test_memory_alignment),
        ("Greedy Comparison", test_comparison_with_greedy),
    ]
    
    passed = 0
    failed = 0
    
    for name, test_func in tests:
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"❌ FAILED: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("\n" + "=" * 60)
    print(f"SUMMARY: {passed} passed, {failed} failed out of {len(tests)} tests")
    print("=" * 60)
    
    # Generate Go test examples
    print("\n" + "=" * 60)
    print("GO/CGO TEST EXAMPLES FOR CHARIOT")
    print("=" * 60)
    
    generate_go_test_case(
        "Basic Test",
        n=5,
        weights=[2, 3, 4, 5, 9],
        values=[3, 4, 5, 8, 10],
        capacity=10
    )
    
    generate_go_test_case(
        "Single Item",
        n=1,
        weights=[5],
        values=[10],
        capacity=10
    )
    
    generate_go_test_case(
        "Zero Capacity",
        n=3,
        weights=[1, 2, 3],
        values=[10, 20, 30],
        capacity=0
    )
    
    print("\n" + "=" * 60)
    print("DEBUGGING TIPS FOR CHARIOT CGO INTEGRATION:")
    print("=" * 60)
    print("""
1. Verify array pointer conversion:
   - Use unsafe.Pointer(&array[0]) for first element
   - Ensure arrays are C.int type (int32)
   - Check array lengths match n parameter

2. Check memory alignment:
   - Arrays should be contiguous in memory
   - No padding between elements for int32
   - Print pointer addresses to verify

3. Validate return value:
   - Should match sum of selected values
   - Should be >= 0
   - Should be <= sum of all values

4. Common CGO issues:
   - Mixing Go int with C.int (different sizes on some platforms)
   - Passing slice vs array pointer
   - Forgetting to allocate selection array
   - Not checking if n matches array lengths

5. Debug with prints:
   fmt.Printf("Before: n=%d, cap=%d, weights=%v, values=%v\\n", n, capacity, weights, values)
   fmt.Printf("After: totalValue=%d, selection=%v\\n", totalValue, selection)
""")
    
    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
