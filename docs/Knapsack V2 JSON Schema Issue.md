# Knapsack V2 JSON Schema Issue

## Summary

The go-chariot integration with the knapsack C++ library is failing because we don't have the correct V2 JSON schema documentation. The library function `solve_knapsack_v2_from_json()` is rejecting all our JSON formats.

## Current Status

✅ **Library compiled and linked** - No CGO errors
✅ **Config generation works** - `knapsackConfig()` generates valid JSON
❌ **Solver execution fails** - C++ library returns error code (rc != 0)

## What We've Tried

### JSON Format Attempts

1. **Current format** (`item_weights` in constraints):
```json
{
  "mode": "select",
  "num_items": 3,
  "objective": [5.0, 6.0, 7.0],
  "constraints": [{
    "type": "capacity",
    "capacity": 10.0,
    "item_weights": [2.0, 3.0, 4.0]
  }]
}
```
**Result**: ❌ `solve_knapsack_v2_from_json failed`

2. **Previous format** (`weights` in constraints):
```json
{
  "mode": "select",
  "num_items": 3,
  "objective": [5.0, 6.0, 7.0],
  "constraints": [{
    "type": "capacity",
    "capacity": 10.0,
    "weights": [2.0, 3.0, 4.0]
  }]
}
```
**Result**: ❌ `solve_knapsack_v2_from_json failed`

3. **Nested objective format**:
```json
{
  "mode": "select",
  "num_items": 3,
  "objective": {
    "weights": [5.0, 6.0, 7.0]
  },
  "constraints": [{
    "type": "capacity",
    "capacity": 10.0,
    "weights": [2.0, 3.0, 4.0]
  }]
}
```
**Result**: ❌ `solve_knapsack_v2_from_json failed`

## What We Need from Knapsack Project

### 1. V2 JSON Schema Documentation

We need the **complete V2 JSON schema** that `solve_knapsack_v2_from_json()` expects.

**Header comment says:**
```c
// Solve from a JSON string according to the V2 schema (see docs/v2/README.md).
```

But `docs/v2/README.md` **does not exist** in the chariot-ecosystem copy of the library.

### 2. Example JSON Configs

Please provide **working examples** of V2 JSON configs, such as:

- Simple 3-item knapsack problem
- Multi-constraint problem
- Problem with options_json parameter

### 3. Unit Test Examples

If the C++ library has unit tests that call `solve_knapsack_v2_from_json()`, please share:
- The test JSON configs
- Expected outputs
- Any validation/parsing code

## Temporary Workarounds Considered

### Option A: Use Legacy V1 API

The debug guides reference a simpler V1 API:
```c
int knapsack(int n, int* weights, int* values, int capacity, int* selection);
```

However, this function **does not exist** in the macOS library we have:
```bash
$ nm knapsack-library/lib/macos-cpu/libknapsack_macos_cpu.a | grep " T " | grep knapsack
# Only shows: solve_knapsack_v2_from_json, free_knapsack_solution_v2, ks_v2_select_ptr
```

### Option B: Implement Simple Wrapper

We could implement a simple knapsack solver in Go as a fallback, but this defeats the purpose of using the optimized C++ library.

### Option C: Skip Tests for Now

Mark knapsack tests as `// TODO: Pending V2 schema documentation` until we get the schema.

## Recommended Actions

### For Knapsack Project Team

1. **Provide V2 schema documentation** - Either as markdown or JSON schema file
2. **Share working JSON examples** - From C++ unit tests or integration tests  
3. **Consider adding V1 legacy API** - For simpler CGO integration

### For Chariot Team

1. **Wait for schema documentation** - Don't guess at the format
2. **Update tests with correct schema** - Once we have it
3. **Add schema validation** - To `knapsackConfig()` function
4. **Document JSON format** - In Chariot language reference

## Files to Update Once Schema Available

1. **`chariot/knapsack_funcs.go`** - Update `knapsackConfig()` to generate correct JSON
2. **`tests/knapsack_test.go`** - Verify all tests pass with real solver
3. **`docs/KnapsackFunctions.md`** - Document JSON format requirements
4. **`tests/KNAPSACK_TEST_SUMMARY.md`** - Update with real test results

## Contact Points

**Knapsack Project**: Please provide V2 JSON schema and examples

**Questions**:
1. What is the exact JSON format expected by `solve_knapsack_v2_from_json()`?
2. Where can we find the V2 schema documentation mentioned in the header?
3. Do you have example JSON configs from your C++ unit tests?
4. Is there validation code we can reference?

---

**Status**: ⏸️ Blocked pending V2 JSON schema documentation  
**Date**: November 18, 2025  
**Reporter**: Chariot Ecosystem Team
