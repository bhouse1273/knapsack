#!/usr/bin/env python3
"""
Python tests for knapsack V2 JSON API - Ready for chariot-ecosystem replication

These tests validate the knapsack_v2() C function call which accepts JSON input.
This is the actual API that go-chariot will use.

The V2 API is simpler for CGO integration:
- Single JSON string input
- Single JSON string output
- No complex pointer marshaling

Build and test:
    cd build
    ./knapsack_solver --mode=select --config='{"items":[...]}' --beam-width=100
"""

import json
import subprocess
import sys
from pathlib import Path

# Find the knapsack_solver executable
SOLVER_PATHS = [
    "build/knapsack_solver",
    "../build/knapsack_solver",
    "../../build/knapsack_solver",
]

solver_path = None
for path in SOLVER_PATHS:
    if Path(path).exists():
        solver_path = path
        break

if not solver_path:
    print("ERROR: Could not find knapsack_solver executable")
    print("Build it first:")
    print("  cd build && cmake .. && make knapsack_solver")
    sys.exit(1)

print(f"Using solver: {solver_path}")


def run_solver(config, mode="select", beam_width=100):
    """Run the knapsack solver with JSON config"""
    cmd = [
        solver_path,
        f"--mode={mode}",
        f"--config={json.dumps(config)}",
        f"--beam-width={beam_width}",
    ]
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if result.returncode != 0:
            print(f"Error running solver: {result.stderr}")
            return None
            
        # Parse JSON output
        return json.loads(result.stdout)
        
    except subprocess.TimeoutExpired:
        print("Solver timed out")
        return None
    except json.JSONDecodeError as e:
        print(f"Failed to parse JSON output: {e}")
        print(f"Output was: {result.stdout}")
        return None
    except Exception as e:
        print(f"Error: {e}")
        return None


def test_basic_select():
    """Test 1: Basic select mode"""
    print("\n=== Test 1: Basic Select Mode ===")
    
    config = {
        "items": [
            {"id": "item1", "value": 10, "weight": 5},
            {"id": "item2", "value": 15, "weight": 7},
            {"id": "item3", "value": 20, "weight": 10},
            {"id": "item4", "value": 25, "weight": 12},
        ],
        "constraints": [
            {"name": "weight", "capacity": 20, "type": "hard"}
        ],
        "objectives": [
            {"name": "value", "weight": 1.0}
        ]
    }
    
    print(f"Config: {json.dumps(config, indent=2)}")
    
    result = run_solver(config, mode="select")
    
    if not result:
        print("❌ FAILED: No result")
        return False
    
    print(f"Result: {json.dumps(result, indent=2)}")
    
    # Validate result
    if "best_select" not in result:
        print("❌ FAILED: No best_select in result")
        return False
    
    selected = result["best_select"]
    print(f"Selected items: {selected}")
    
    # Calculate total weight
    total_weight = sum(
        item["weight"]
        for item in config["items"]
        if item["id"] in selected
    )
    
    total_value = sum(
        item["value"]
        for item in config["items"]
        if item["id"] in selected
    )
    
    print(f"Total weight: {total_weight}/20")
    print(f"Total value: {total_value}")
    
    if total_weight > 20:
        print(f"❌ FAILED: Weight {total_weight} exceeds capacity 20")
        return False
    
    print("✅ PASSED")
    return True


def test_assign_mode():
    """Test 2: Assign mode (multiple knapsacks)"""
    print("\n=== Test 2: Assign Mode ===")
    
    config = {
        "items": [
            {"id": "player1", "value": 85, "weight": 1},
            {"id": "player2", "value": 90, "weight": 1},
            {"id": "player3", "value": 78, "weight": 1},
            {"id": "player4", "value": 88, "weight": 1},
        ],
        "knapsacks": [
            {"id": "team1", "capacity": 2},
            {"id": "team2", "capacity": 2},
        ],
        "objectives": [
            {"name": "value", "weight": 1.0}
        ]
    }
    
    print(f"Config: {json.dumps(config, indent=2)}")
    
    result = run_solver(config, mode="assign")
    
    if not result:
        print("❌ FAILED: No result")
        return False
    
    print(f"Result: {json.dumps(result, indent=2)}")
    
    if "best_assign" not in result:
        print("❌ FAILED: No best_assign in result")
        return False
    
    assignments = result["best_assign"]
    print(f"Assignments: {assignments}")
    
    # Check each knapsack doesn't exceed capacity
    for knapsack in config["knapsacks"]:
        assigned_to_knapsack = [
            item_id for item_id, k_id in assignments.items()
            if k_id == knapsack["id"]
        ]
        count = len(assigned_to_knapsack)
        capacity = knapsack["capacity"]
        
        print(f"  {knapsack['id']}: {count}/{capacity} items - {assigned_to_knapsack}")
        
        if count > capacity:
            print(f"❌ FAILED: Knapsack {knapsack['id']} has {count} items, capacity {capacity}")
            return False
    
    print("✅ PASSED")
    return True


def test_multi_constraint():
    """Test 3: Multiple constraints"""
    print("\n=== Test 3: Multiple Constraints ===")
    
    config = {
        "items": [
            {"id": "item1", "value": 10, "weight": 5, "volume": 2},
            {"id": "item2", "value": 15, "weight": 7, "volume": 3},
            {"id": "item3", "value": 20, "weight": 10, "volume": 5},
        ],
        "constraints": [
            {"name": "weight", "capacity": 15, "type": "hard"},
            {"name": "volume", "capacity": 7, "type": "hard"},
        ],
        "objectives": [
            {"name": "value", "weight": 1.0}
        ]
    }
    
    print(f"Config: {json.dumps(config, indent=2)}")
    
    result = run_solver(config, mode="select")
    
    if not result:
        print("❌ FAILED: No result")
        return False
    
    print(f"Result: {json.dumps(result, indent=2)}")
    
    selected = result.get("best_select", [])
    print(f"Selected items: {selected}")
    
    # Check constraints
    total_weight = sum(
        item["weight"]
        for item in config["items"]
        if item["id"] in selected
    )
    
    total_volume = sum(
        item["volume"]
        for item in config["items"]
        if item["id"] in selected
    )
    
    print(f"Total weight: {total_weight}/15")
    print(f"Total volume: {total_volume}/7")
    
    if total_weight > 15 or total_volume > 7:
        print(f"❌ FAILED: Constraints violated")
        return False
    
    print("✅ PASSED")
    return True


def generate_go_example():
    """Generate Go/CGO example code"""
    print("\n" + "=" * 60)
    print("GO/CGO EXAMPLE FOR CHARIOT")
    print("=" * 60)
    print("""
The knapsack V2 API uses JSON for simplicity:

```go
package main

/*
#cgo CFLAGS: -I/path/to/knapsack/include
#cgo LDFLAGS: -L/path/to/knapsack/lib -lknapsack_cpu -lstdc++ -lm

#include <stdlib.h>
#include <string.h>
#include "knapsack_c.h"

// Helper to call knapsack_v2 with JSON
int solve_knapsack_json(const char* config_json, char* result_json, size_t result_size) {
    // Parse JSON, call knapsack_v2, return JSON result
    // See knapsack_c.h for full API
    return 0;
}
*/
import "C"
import (
    "encoding/json"
    "fmt"
    "unsafe"
)

type KnapsackConfig struct {
    Items       []Item       `json:"items"`
    Constraints []Constraint `json:"constraints"`
    Objectives  []Objective  `json:"objectives"`
}

type Item struct {
    ID     string  `json:"id"`
    Value  float64 `json:"value"`
    Weight float64 `json:"weight"`
}

type Constraint struct {
    Name     string  `json:"name"`
    Capacity float64 `json:"capacity"`
    Type     string  `json:"type"`
}

type Objective struct {
    Name   string  `json:"name"`
    Weight float64 `json:"weight"`
}

func SolveKnapsack(config KnapsackConfig) (map[string]interface{}, error) {
    // Convert config to JSON
    configJSON, err := json.Marshal(config)
    if err != nil {
        return nil, err
    }
    
    // Call C function
    cConfig := C.CString(string(configJSON))
    defer C.free(unsafe.Pointer(cConfig))
    
    resultBuf := make([]byte, 1024*1024) // 1MB buffer
    cResult := (*C.char)(unsafe.Pointer(&resultBuf[0]))
    
    ret := C.solve_knapsack_json(cConfig, cResult, C.size_t(len(resultBuf)))
    
    if ret != 0 {
        return nil, fmt.Errorf("knapsack solver failed: %d", ret)
    }
    
    // Parse result JSON
    var result map[string]interface{}
    if err := json.Unmarshal(resultBuf, &result); err != nil {
        return nil, err
    }
    
    return result, nil
}
```

Key points for chariot team:
1. Use JSON API (simpler than raw pointers)
2. Link against libknapsack_cpu.a or libknapsack_cuda.a
3. Include libstdc++ and libm in LDFLAGS
4. See knapsack_c.h for full C API documentation
""")


def main():
    """Run all tests"""
    print("=" * 60)
    print("KNAPSACK V2 API VALIDATION TESTS")
    print("=" * 60)
    print(f"Solver: {solver_path}")
    
    tests = [
        ("Basic Select", test_basic_select),
        ("Assign Mode", test_assign_mode),
        ("Multi Constraint", test_multi_constraint),
    ]
    
    passed = 0
    failed = 0
    
    for name, test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"❌ FAILED: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("\n" + "=" * 60)
    print(f"SUMMARY: {passed} passed, {failed} failed out of {len(tests)} tests")
    print("=" * 60)
    
    generate_go_example()
    
    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
