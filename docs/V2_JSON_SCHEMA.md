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
  "count": number,                // Required: Number of rows/items
  "attributes": {                 // Required: Map of attribute name -> payload
    "attribute_name": number[] | ExternalAttributeSpec
  }
}
```

Inline arrays (the original format) still work exactly the same:

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

For large datasets you can now stream attributes from disk or pipes by providing an **ExternalAttributeSpec** object instead of an inline array:

```typescript
type ExternalAttributeSpec = {
  "source": "file" | "stream",
  "format"?: "binary64_le" | "csv",  // Encoding of payload (default binary64_le)
  "path"?: string,                    // File that contains this attribute
  "chunks"?: string[],                // Optional chunk files processed sequentially
  "channel"?: string,                 // Stream identifier ("stdin" or file://...)
  "offset_bytes"?: number,            // Byte offset applied to the first file (binary only)
  "delimiter"?: string,               // CSV: column delimiter (default ",")
  "has_header"?: boolean,             // CSV: true if first row is header
  "column"?: string,                  // CSV: column name to read (requires header)
  "column_index"?: number             // CSV: fallback column index (default 0)
}
```

**File-backed example (raw doubles on disk):**

```json
{
  "count": 100000,
  "attributes": {
    "value": {
      "source": "file",
      "format": "binary64_le",
      "path": "data/value.bin"
    },
    "weight": {
      "source": "file",
      "format": "binary64_le",
      "path": "data/weight_part1.bin",
      "chunks": ["data/weight_part2.bin"]
    }
  }
}
```

**CSV-backed example (one value per row):**

```json
{
  "count": 4,
  "attributes": {
    "value": {
      "source": "file",
      "format": "csv",
      "path": "data/value.csv",
      "has_header": true,
      "column": "value"
    }
  }
}
```

**CSV-backed example (multiple columns, one file):**

```json
{
  "count": 4,
  "attributes": {
    "value": {
      "source": "file",
      "format": "csv",
      "path": "data/items.csv",
      "has_header": true,
      "column": "value"
    },
    "weight": {
      "source": "file",
      "format": "csv",
      "path": "data/items.csv",
      "has_header": true,
      "column": "weight"
    },
    "priority": {
      "source": "file",
      "format": "csv",
      "path": "data/items.csv",
      "column_index": 2
    }
  }
}
```

Use the same CSV file for as many attributes as you like—the loader re-reads the file per column so each attribute can reference its own header name or `column_index`.

**Streaming example (pipe/STDIN):**

```json
{
  "count": 500000,
  "attributes": {
    "value": {
      "source": "stream",
      "channel": "stdin",
      "format": "binary64_le"
    },
    "weight": {
      "source": "stream",
      "channel": "file://tmp/weight.pipe",
      "format": "binary64_le"
    }
  }
}
```

**CSV stream example (stdin with delimiter override):**

```json
{
  "count": 1000,
  "attributes": {
    "value": {
      "source": "stream",
      "format": "csv",
      "channel": "stdin",
      "delimiter": "|",
      "has_header": false,
      "column_index": 0
    }
  }
}
```

Pipe newline-delimited rows into stdin (or reference `file://path/to/fifo`) using the requested delimiter. Each attribute creates its own stream subscription, so you can combine CSV streams with binary streams in the same request as long as every attribute produces exactly `items.count` rows.

**Rules:**
- `count` must be > 0
- Inline arrays must have exactly `count` elements
- External specs must provide either `path`/`chunks` (file mode) or `channel` (stream mode)
- Supported encodings today:
  - `binary64_le`: raw IEEE-754 doubles (requires binary files or streams)
  - `csv`: newline-delimited textual values (one file or stream per attribute)
  - `arrow`: Arrow IPC/Feather files, column selected by `column` or `column_index`
  - `parquet`: Parquet columnar files, column selected by `column` or `column_index`
- You can mix inline, file, and stream attributes within the same request

#### How ingestion works under the hood

- All payloads flow through the new `HostSoABuilder` (see `include/v2/Data.h`), which enforces `items.count` while accepting column chunks or streaming rows.
- **File mode** reads raw doubles from `path` followed by any `chunks`, optionally skipping `offset_bytes` bytes in the first file. Use this for `.bin` or `.npy`-style assets sitting next to the config JSON.
- **Stream mode** reads raw doubles from a live channel. Use `"channel": "stdin"` to push data via standard input, or `"channel": "file://..."` to attach an already-open FIFO. If you don't have an actual pipe yet, provide a list of `chunks` and the solver will treat them as a streamed sequence.
- Every attribute remains columnar/SoA. If you need row-wise streaming, emit each column sequentially (value stream, weight stream, …) or use the builder API directly in CGO/C++ to push rows.

### BlockSpec

```typescript
{
  "name"?: string,               // Optional: Block name
  "start"?: number,              // Optional: Starting index (0-based)
```json
{
  "count": 1024,
  "attributes": {
    "value": {
      "source": "file",
      "format": "arrow",
      "path": "data/items.arrow",
      "column": "value"
    }
  }
}
```

Arrow support expects on-disk IPC/Feather files. Provide `column` (preferred) or `column_index` to select the numeric column to ingest. Streaming Arrow pipes are not yet supported.

**Parquet example (re-using a data lake file):**

```json
{
  "count": 25000,
  "attributes": {
    "weight": {
      "source": "file",
      "format": "parquet",
      "path": "/mnt/data/items.parquet",
      "column": "weight_kg"
    }
  }
}
```

Parquet ingestion currently works for local files (single file or `chunks` list). Like Arrow, select the numeric column via `column` or `column_index`.
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
- [ ] Every inline attribute array has length = `items.count`
- [ ] Every external attribute spec (`source: file|stream`) declares a path/channel and uses `format = "binary64_le"`
- [ ] `objective` array is not empty
- [ ] All `objective[].attr` names exist in either `items.attributes` or external specs
- [ ] All `constraints[].attr` names exist in either `items.attributes` or external specs
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
