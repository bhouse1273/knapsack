# Knapsack V2 JSON API Schema - Complete Documentation

## Overview

This document provides the **complete V2 JSON schema** for `solve_knapsack_v2_from_json()` function calls.

**Status**: ✅ Complete and validated against working examples  
**Date**: November 18, 2025

## TL;DR - Minimal Working Example

```json
{
  "version": 2,
  "mode": "select",
  "items": {
    "count": 3,
    "attributes": {
      "value":  [5.0, 6.0, 7.0],
      "weight": [2.0, 3.0, 4.0]
    }
  },
  "blocks": [
    { "name": "all", "start": 0, "count": 3 }
  ],
  "objective": [
    { "attr": "value", "weight": 1.0 }
  ],
  "constraints": [
    { "kind": "capacity", "attr": "weight", "limit": 10.0 }
  ]
}
```

**This will solve the 3-item knapsack problem that chariot team is testing!**

## Complete Schema Reference

### Root Object

```typescript
{
  "version": number,              // Required: Always 2 for V2 API
  "mode": string,                 // Required: "select" or "assign"
  "random_seed"?: number,         // Optional: Random seed (default: 0)
  "items": ItemsSpec,             // Required: Item definitions
  "blocks": BlockSpec[],          // Required: Item groupings (can be empty)
  "objective": CostTermSpec[],    // Required: Objective function
  "constraints"?: ConstraintSpec[],  // Optional: Constraint definitions
  "knapsack"?: KnapsackSpec      // Required only for mode="assign"
}
```

### ItemsSpec

```typescript
{
  "count": number,                // Required: Number of items
  "attributes": {                 // Required: Map of attribute name -> array
    "attribute_name": number[]   // Each array must have length = count
  }
}
```

**Example:**
```json
{
  "count": 3,
  "attributes": {
    "value":  [10.0, 20.0, 30.0],
    "weight": [5.0, 10.0, 15.0],
    "priority": [1.0, 2.0, 3.0]
  }
}
```

**Rules:**
- `count` must be > 0
- Every attribute array must have exactly `count` elements
- You can have any number of attributes (not just value/weight)

### BlockSpec

```typescript
{
  "name"?: string,               // Optional: Block name
  "start"?: number,              // Optional: Starting index (0-based)
  "count"?: number,              // Optional: Number of items in block
  "indices"?: number[]           // Optional: Explicit item indices
}
```

**Simple approach (single block for all items):**
```json
"blocks": [
  { "name": "all", "start": 0, "count": 3 }
]
```

**Or empty (if you don't need blocks):**
```json
"blocks": []
```

### CostTermSpec (Objective)

```typescript
{
  "attr": string,                // Required: Attribute name to optimize
  "weight"?: number              // Optional: Weight multiplier (default: 1.0)
}
```

**Example (maximize value):**
```json
"objective": [
  { "attr": "value", "weight": 1.0 }
]
```

**Example (multi-objective):**
```json
"objective": [
  { "attr": "value", "weight": 1.0 },
  { "attr": "priority", "weight": 0.5 }
]
```

### ConstraintSpec

```typescript
{
  "kind": string,                // Required: "capacity" (other kinds may exist)
  "attr": string,                // Required: Attribute name for constraint
  "limit": number,               // Required: Constraint limit/capacity
  "soft"?: boolean,              // Optional: Soft constraint (default: false = hard)
  "penalty"?: {                  // Optional: Penalty for soft constraints
    "weight": number,            // Penalty weight multiplier
    "power": number              // Penalty exponent (1.0 = linear, 2.0 = quadratic)
  }
}
```

**Hard constraint (must not exceed):**
```json
{
  "kind": "capacity",
  "attr": "weight",
  "limit": 10.0
}
```

**Soft constraint (penalized if exceeded):**
```json
{
  "kind": "capacity",
  "attr": "weight",
  "limit": 10.0,
  "soft": true,
  "penalty": {
    "weight": 5.0,
    "power": 2.0
  }
}
```

### KnapsackSpec (for mode="assign")

```typescript
{
  "K"?: number,                  // Optional: Number of knapsacks
  "capacities"?: number[],       // Optional: Capacity for each knapsack
  "capacity_attr"?: string       // Optional: Attribute name for capacity
}
```

**Note:** Only needed when `mode="assign"` (not for select mode)

## Working Examples

### Example 1: Simple 3-Item Knapsack (SELECT Mode)

**Problem:** Select items to maximize value, weight limit 10

```json
{
  "version": 2,
  "mode": "select",
  "items": {
    "count": 3,
    "attributes": {
      "value":  [5.0, 6.0, 7.0],
      "weight": [2.0, 3.0, 4.0]
    }
  },
  "blocks": [
    { "name": "all", "start": 0, "count": 3 }
  ],
  "objective": [
    { "attr": "value", "weight": 1.0 }
  ],
  "constraints": [
    { "kind": "capacity", "attr": "weight", "limit": 10.0 }
  ]
}
```

**Expected Output:**
```json
{
  "select": [1, 1, 1],           // All items selected (2+3+4 = 9 <= 10)
  "objective": 18.0,             // 5+6+7 = 18
  "penalty": 0.0,
  "total": 18.0
}
```

### Example 2: 6-Item Problem with Soft Constraint

From `docs/v2/example_select.json`:

```json
{
  "version": 2,
  "mode": "select",
  "random_seed": 1,
  "items": {
    "count": 6,
    "attributes": {
      "value":  [10, 12, 7, 5, 8, 6],
      "weight": [8, 10, 5, 4, 7, 6]
    }
  },
  "blocks": [
    { "name": "all", "start": 0, "count": 6 }
  ],
  "objective": [
    { "attr": "value", "weight": 1.0 }
  ],
  "constraints": [
    {
      "kind": "capacity",
      "attr": "weight",
      "limit": 15,
      "soft": true,
      "penalty": {
        "weight": 10.0,
        "power": 2.0
      }
    }
  ]
}
```

### Example 3: Multiple Constraints

From `docs/v2/example_select_multi.json`:

```json
{
  "version": 2,
  "mode": "select",
  "random_seed": 42,
  "items": {
    "count": 6,
    "attributes": {
      "value":  [10, 12, 7, 5, 8, 6],
      "weight": [8, 10, 5, 4, 7, 6],
      "volume": [1, 1, 1, 1, 1, 1]
    }
  },
  "blocks": [
    { "name": "all", "start": 0, "count": 6 }
  ],
  "objective": [
    { "attr": "value", "weight": 1.0 }
  ],
  "constraints": [
    {
      "kind": "capacity",
      "attr": "weight",
      "limit": 15,
      "soft": true,
      "penalty": { "weight": 10.0, "power": 2.0 }
    },
    {
      "kind": "capacity",
      "attr": "volume",
      "limit": 3,
      "soft": true,
      "penalty": { "weight": 2.0, "power": 2.0 }
    }
  ]
}
```

### Example 4: 10-Item Benchmark Problem

From `data/benchmarks/small_mkp_v2.json`:

```json
{
  "version": 2,
  "mode": "select",
  "random_seed": 42,
  "items": {
    "count": 10,
    "attributes": {
      "value": [50, 60, 70, 80, 90, 100, 110, 120, 130, 140],
      "weight": [15, 20, 25, 30, 35, 40, 45, 50, 55, 60]
    }
  },
  "blocks": [
    { "name": "all", "start": 0, "count": 10 }
  ],
  "objective": [
    { "attr": "value", "weight": 1.0 }
  ],
  "constraints": [
    {
      "kind": "capacity",
      "attr": "weight",
      "limit": 100,
      "soft": true,
      "penalty": { "weight": 10.0, "power": 2.0 }
    }
  ]
}
```

## CGO Integration

### C Function Signature

```c
int solve_knapsack_v2_from_json(
    const char* json_config,      // JSON string (see above)
    const char* options_json,     // Optional solver options (can be NULL)
    KnapsackSolutionV2** out_solution  // Output solution pointer
);
```

**Return Codes:**
- `0`: Success
- `-1`: out_solution is NULL
- `-2`: json_config is NULL
- `-3`: JSON parsing failed (invalid JSON or schema)
- `-4`: Failed to build internal data structures
- `-5`: Unsupported mode (only "select" currently supported via V2 API)
- `-6`: Solver failed
- `-7`: Memory allocation failed

### Solver Options (optional_json)

```json
{
  "beam_width": 100,             // Beam search width (default: 100)
  "iters": 1,                    // Number of iterations (default: 1)
  "seed": 42,                    // Random seed (default: 0)
  "debug": false,                // Debug output (default: false)
  "dom_enable": true,            // Enable dominance filter (default: false)
  "dom_eps": 1e-6,               // Dominance epsilon (default: 1e-6)
  "dom_surrogate": false         // Use surrogate dominance (default: false)
}
```

**Example:**
```json
{
  "beam_width": 200,
  "iters": 5,
  "debug": true
}
```

### Go/CGO Example

```go
package main

/*
#cgo CFLAGS: -I/path/to/knapsack/include
#cgo linux LDFLAGS: -L/path/to/knapsack/lib/linux-cpu -lknapsack_cpu -lstdc++ -lm
#cgo darwin LDFLAGS: -L/path/to/knapsack/lib/macos-metal -lknapsack_metal -lstdc++ -lm

#include <stdlib.h>
#include "knapsack_c.h"
*/
import "C"
import (
	"encoding/json"
	"fmt"
	"unsafe"
)

type V2Config struct {
	Version     int                    `json:"version"`
	Mode        string                 `json:"mode"`
	RandomSeed  int                    `json:"random_seed,omitempty"`
	Items       V2Items                `json:"items"`
	Blocks      []V2Block              `json:"blocks"`
	Objective   []V2CostTerm           `json:"objective"`
	Constraints []V2Constraint         `json:"constraints,omitempty"`
}

type V2Items struct {
	Count      int                       `json:"count"`
	Attributes map[string][]float64      `json:"attributes"`
}

type V2Block struct {
	Name    string `json:"name,omitempty"`
	Start   int    `json:"start,omitempty"`
	Count   int    `json:"count,omitempty"`
	Indices []int  `json:"indices,omitempty"`
}

type V2CostTerm struct {
	Attr   string  `json:"attr"`
	Weight float64 `json:"weight,omitempty"`
}

type V2Constraint struct {
	Kind    string     `json:"kind"`
	Attr    string     `json:"attr"`
	Limit   float64    `json:"limit"`
	Soft    bool       `json:"soft,omitempty"`
	Penalty *V2Penalty `json:"penalty,omitempty"`
}

type V2Penalty struct {
	Weight float64 `json:"weight"`
	Power  float64 `json:"power"`
}

func SolveKnapsackV2(config V2Config) ([]int, float64, error) {
	// Convert config to JSON
	configJSON, err := json.Marshal(config)
	if err != nil {
		return nil, 0, fmt.Errorf("marshal config: %w", err)
	}
	
	cConfig := C.CString(string(configJSON))
	defer C.free(unsafe.Pointer(cConfig))
	
	// Call C function
	var solution *C.KnapsackSolutionV2
	rc := C.solve_knapsack_v2_from_json(cConfig, nil, &solution)
	if rc != 0 {
		return nil, 0, fmt.Errorf("solver failed with code %d", rc)
	}
	defer C.free_knapsack_solution_v2(solution)
	
	// Extract results
	numItems := int(solution.num_items)
	selection := make([]int, numItems)
	
	selectPtr := C.ks_v2_select_ptr(solution)
	for i := 0; i < numItems; i++ {
		selection[i] = int(*(*C.int)(unsafe.Pointer(uintptr(unsafe.Pointer(selectPtr)) + uintptr(i)*unsafe.Sizeof(C.int(0)))))
	}
	
	objective := float64(solution.objective)
	
	return selection, objective, nil
}

func main() {
	// Create config for 3-item problem
	config := V2Config{
		Version: 2,
		Mode:    "select",
		Items: V2Items{
			Count: 3,
			Attributes: map[string][]float64{
				"value":  {5.0, 6.0, 7.0},
				"weight": {2.0, 3.0, 4.0},
			},
		},
		Blocks: []V2Block{
			{Name: "all", Start: 0, Count: 3},
		},
		Objective: []V2CostTerm{
			{Attr: "value", Weight: 1.0},
		},
		Constraints: []V2Constraint{
			{
				Kind:  "capacity",
				Attr:  "weight",
				Limit: 10.0,
			},
		},
	}
	
	selection, objective, err := SolveKnapsackV2(config)
	if err != nil {
		fmt.Printf("Error: %v\n", err)
		return
	}
	
	fmt.Printf("Selection: %v\n", selection)
	fmt.Printf("Objective: %.2f\n", objective)
}
```

## Common Errors and Fixes

### Error: "solve_knapsack_v2_from_json failed with code -3"

**Cause:** JSON schema validation failed

**Common issues:**
1. ❌ Missing required field (`version`, `mode`, `items`, etc.)
2. ❌ Attribute array length doesn't match `items.count`
3. ❌ Empty `objective` array
4. ❌ Invalid JSON syntax

**Fix:** Validate JSON against schema above

### Error: "solve_knapsack_v2_from_json failed with code -5"

**Cause:** Unsupported mode

**Fix:** Use `"mode": "select"` (assign mode not yet supported via C API)

### Error: Wrong results

**Cause:** Usually attribute mismatch

**Example wrong:**
```json
{
  "items": {
    "count": 3,
    "attributes": {
      "objective": [5, 6, 7],     // ❌ Wrong name
      "item_weights": [2, 3, 4]   // ❌ Wrong name
    }
  },
  "objective": [
    { "attr": "value", ... }       // ❌ Refers to non-existent attribute
  ],
  "constraints": [
    { "attr": "weight", ... }      // ❌ Refers to non-existent attribute
  ]
}
```

**Example correct:**
```json
{
  "items": {
    "count": 3,
    "attributes": {
      "value": [5, 6, 7],           // ✅ Match attribute names
      "weight": [2, 3, 4]           // ✅ Used in constraints
    }
  },
  "objective": [
    { "attr": "value", ... }         // ✅ Exists in attributes
  ],
  "constraints": [
    { "attr": "weight", ... }        // ✅ Exists in attributes
  ]
}
```

## Validation Checklist

Before calling `solve_knapsack_v2_from_json`:

- [ ] `version` is 2
- [ ] `mode` is "select" (only mode currently supported)
- [ ] `items.count` > 0
- [ ] Every attribute array has length = `items.count`
- [ ] `objective` array is not empty
- [ ] All `objective[].attr` names exist in `items.attributes`
- [ ] All `constraints[].attr` names exist in `items.attributes`
- [ ] `blocks` array exists (can be empty or single block covering all items)
- [ ] JSON is valid (use `json.Marshal` or validate with JSON parser)

## Summary

**The V2 JSON schema uses a structured attribute-based format, NOT the flat arrays that chariot team was trying.**

Key differences from what chariot tried:
- ❌ `"objective": [5, 6, 7]` (flat array)
- ✅ `"items": { "attributes": { "value": [5, 6, 7] } }` (structured)

- ❌ `"constraints": [{ "item_weights": [...] }]` (weights in constraint)
- ✅ `"items": { "attributes": { "weight": [...] } }` + `"constraints": [{ "attr": "weight" }]` (reference attribute)

**Use the minimal working example above to get started, then expand from there!**

## Additional Resources

- Working JSON examples: `docs/v2/example_*.json`
- Benchmark examples: `data/benchmarks/*.json`
- C++ test examples: `tests/v2/test_*.cpp`
- V2 Config implementation: `src/v2/Config_json.cpp`

**Status**: Ready for chariot-ecosystem integration! 🚀
