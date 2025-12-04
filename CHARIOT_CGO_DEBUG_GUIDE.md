# Chariot CGO Integration Debug Guide

## Problem Summary

Tests are failing in go-chariot when calling the knapsack C library via CGO. This guide provides debugging steps and test cases to help isolate the issue.

## Current Library Status

✅ **All platform libraries built and verified:**
- Linux CPU: `libknapsack_cpu.a` (312KB, ELF x86-64)
- Linux CUDA: `libknapsack_cuda.a` (669KB, ELF x86-64)
- macOS Metal: `libknapsack_metal.a` (229KB, Mach-O ARM64)
- macOS CPU: `libknapsack_cpu.a` (1.8MB, Mach-O ARM64)

✅ **All C++ unit tests passing** (107+ tests, 5 test suites)
✅ **RL Support libraries included** in all platforms

## Available C APIs

The knapsack library provides two main APIs:

### 1. Legacy API (Simple, Good for Initial Testing)

```c
// From knapsack_c.h
int knapsack(int n, int* weights, int* values, int capacity, int* selection);
```

**Parameters:**
- `n`: Number of items
- `weights`: Array of item weights (int32)
- `values`: Array of item values (int32)
- `capacity`: Knapsack capacity
- `selection`: Output array (0/1 for each item)

**Returns:** Total value of selected items

### 2. V2 API (JSON-based, Production-Ready)

```c
// From knapsack_c.h
int knapsack_v2_select(const char* config_json, char* result_json, size_t result_size);
int knapsack_v2_assign(const char* config_json, char* result_json, size_t result_size);
```

**Simpler for CGO:**
- Single JSON string input
- Single JSON string output
- No pointer marshaling complexity

## Debugging Checklist for Chariot Team

### Step 1: Verify Library Linkage

```go
// In your go file with CGO
/*
#cgo CFLAGS: -I/path/to/knapsack/include
#cgo linux LDFLAGS: -L/path/to/knapsack/lib/linux-cpu -lknapsack_cpu -lstdc++ -lm
#cgo darwin LDFLAGS: -L/path/to/knapsack/lib/macos-metal -lknapsack_metal -lstdc++ -lm

#include "knapsack_c.h"
*/
import "C"
```

**Critical:**
- ✅ Path to include directory is correct
- ✅ Path to lib directory matches your platform
- ✅ Must link `libstdc++` and `libm` (C++ standard library and math library)
- ✅ Library file exists: `ls -la /path/to/knapsack/lib/linux-cpu/libknapsack_cpu.a`

### Step 2: Test Minimal CGO Call

Start with the simplest possible test:

```go
package main

/*
#cgo CFLAGS: -I../../include
#cgo linux LDFLAGS: -L../../lib/linux-cpu -lknapsack_cpu -lstdc++ -lm
#cgo darwin LDFLAGS: -L../../lib/macos-metal -lknapsack_metal -lstdc++ -lm

#include <stdlib.h>
#include "knapsack_c.h"
*/
import "C"
import (
	"fmt"
	"unsafe"
)

func main() {
	// Test 1: Single item that fits
	n := C.int(1)
	weights := []C.int{C.int(5)}
	values := []C.int{C.int(10)}
	capacity := C.int(10)
	selection := make([]C.int, 1)
	
	fmt.Printf("Before call:\n")
	fmt.Printf("  n=%d, capacity=%d\n", n, capacity)
	fmt.Printf("  weights=%v\n", weights)
	fmt.Printf("  values=%v\n", values)
	fmt.Printf("  weights ptr: %p\n", &weights[0])
	fmt.Printf("  values ptr: %p\n", &values[0])
	fmt.Printf("  selection ptr: %p\n", &selection[0])
	
	totalValue := C.knapsack(
		n,
		(*C.int)(unsafe.Pointer(&weights[0])),
		(*C.int)(unsafe.Pointer(&values[0])),
		capacity,
		(*C.int)(unsafe.Pointer(&selection[0])),
	)
	
	fmt.Printf("\nAfter call:\n")
	fmt.Printf("  totalValue=%d\n", totalValue)
	fmt.Printf("  selection=%v\n", selection)
	
	// Validate
	if totalValue != 10 {
		fmt.Printf("ERROR: Expected totalValue=10, got %d\n", totalValue)
		return
	}
	
	if selection[0] != 1 {
		fmt.Printf("ERROR: Expected selection[0]=1, got %d\n", selection[0])
		return
	}
	
	fmt.Println("\n✅ SUCCESS: Single item test passed")
}
```

**Expected output:**
```
Before call:
  n=1, capacity=10
  weights=[5]
  values=[10]
  weights ptr: 0x...
  values ptr: 0x...
  selection ptr: 0x...

After call:
  totalValue=10
  selection=[1]

✅ SUCCESS: Single item test passed
```

### Step 3: Common CGO Errors and Fixes

#### Error 1: "undefined reference to knapsack"

**Cause:** Library not linked properly

**Fix:**
```go
// Add to LDFLAGS
#cgo LDFLAGS: -L/path/to/lib -lknapsack_cpu -lstdc++ -lm
```

**Verify:**
```bash
nm -g /path/to/lib/libknapsack_cpu.a | grep knapsack
# Should show: T _knapsack (macOS) or T knapsack (Linux)
```

#### Error 2: "cannot use weights (type []int) as type *_Ctype_int"

**Cause:** Using Go `int` instead of `C.int`

**Fix:**
```go
// WRONG
weights := []int{1, 2, 3}

// CORRECT
weights := []C.int{C.int(1), C.int(2), C.int(3)}
```

#### Error 3: "panic: runtime error: index out of range"

**Cause:** Forgot to allocate selection array

**Fix:**
```go
// WRONG
var selection []C.int

// CORRECT
selection := make([]C.int, n)
```

#### Error 4: Wrong results or segfault

**Cause:** Array size mismatch

**Fix:**
```go
n := C.int(5)
weights := []C.int{C.int(1), C.int(2), C.int(3), C.int(4), C.int(5)}  // Length = 5 ✅
values := []C.int{C.int(10), C.int(20), C.int(30), C.int(40), C.int(50)}  // Length = 5 ✅
selection := make([]C.int, 5)  // Length = 5 ✅
```

#### Error 5: "undefined: _Ctype_struct___0"

**Cause:** C header file not found

**Fix:**
```go
// Check CFLAGS path
#cgo CFLAGS: -I/path/to/knapsack/include

// Verify file exists
// ls -la /path/to/knapsack/include/knapsack_c.h
```

### Step 4: Validate Input Arrays

Before calling the C function, validate:

```go
func validateInputs(n int, weights, values []C.int, capacity C.int) error {
	if len(weights) != n {
		return fmt.Errorf("weights length %d != n %d", len(weights), n)
	}
	if len(values) != n {
		return fmt.Errorf("values length %d != n %d", len(values), n)
	}
	if capacity < 0 {
		return fmt.Errorf("capacity %d < 0", capacity)
	}
	for i := 0; i < n; i++ {
		if weights[i] < 0 {
			return fmt.Errorf("weights[%d] = %d < 0", i, weights[i])
		}
	}
	return nil
}
```

### Step 5: Test Cases to Try

```go
// Test Case 1: Empty knapsack (n=0)
n := C.int(0)
weights := []C.int{}
values := []C.int{}
capacity := C.int(10)
selection := make([]C.int, 0)
// Expected: totalValue = 0

// Test Case 2: Zero capacity
n := C.int(3)
weights := []C.int{C.int(1), C.int(2), C.int(3)}
values := []C.int{C.int(10), C.int(20), C.int(30)}
capacity := C.int(0)
selection := make([]C.int, 3)
// Expected: totalValue = 0, selection = [0, 0, 0]

// Test Case 3: Basic problem
n := C.int(5)
weights := []C.int{C.int(2), C.int(3), C.int(4), C.int(5), C.int(9)}
values := []C.int{C.int(3), C.int(4), C.int(5), C.int(8), C.int(10)}
capacity := C.int(10)
selection := make([]C.int, 5)
// Expected: totalValue = 13, some items selected
```

## Docker Testing

To test in the exact environment where chariot will run:

```dockerfile
FROM golang:1.21-alpine

# Install build tools
RUN apk add --no-cache gcc g++ musl-dev

# Copy knapsack library
COPY knapsack-library/lib/linux-cpu/libknapsack_cpu.a /usr/local/lib/
COPY knapsack-library/lib/linux-cpu/knapsack_cpu.h /usr/local/include/knapsack_c.h

# Set CGO flags
ENV CGO_ENABLED=1
ENV CGO_CFLAGS="-I/usr/local/include"
ENV CGO_LDFLAGS="-L/usr/local/lib -lknapsack_cpu -lstdc++ -lm"

# Copy and build test
COPY test_knapsack.go .
RUN go build -o test_knapsack test_knapsack.go

# Run
CMD ["./test_knapsack"]
```

## Alternative: Use V2 JSON API

If pointer marshaling continues to be problematic, use the V2 JSON API:

```go
func SolveKnapsackJSON(configJSON string) (string, error) {
	cConfig := C.CString(configJSON)
	defer C.free(unsafe.Pointer(cConfig))
	
	resultBuf := make([]byte, 1024*1024) // 1MB buffer
	cResult := (*C.char)(unsafe.Pointer(&resultBuf[0]))
	
	ret := C.knapsack_v2_select(
		cConfig,
		cResult,
		C.size_t(len(resultBuf)),
	)
	
	if ret != 0 {
		return "", fmt.Errorf("solver failed: %d", ret)
	}
	
	// Find null terminator
	resultLen := C.strlen(cResult)
	return string(resultBuf[:resultLen]), nil
}
```

**Advantages:**
- Simpler: Only string pointers, no array marshaling
- More flexible: Easy to add fields without changing C signature
- Better errors: JSON parsing errors are clearer

## Getting Help

If tests still fail, please provide:

1. **Exact Go code** being used (CGO directives + function call)
2. **Input values** (n, weights, values, capacity)
3. **Expected output** vs **actual output**
4. **Error messages** (compile-time or runtime)
5. **Platform** (Linux/macOS, x86-64/ARM64)
6. **Go version**: `go version`
7. **CGO verification**: `go env CGO_ENABLED`

## Library Verification Commands

Before testing, verify the library:

```bash
# Check library exists and is correct format
file /path/to/lib/libknapsack_cpu.a
# Expected: "current ar archive" (both Linux and macOS)

# Check for knapsack symbol
nm -g /path/to/lib/libknapsack_cpu.a | grep knapsack
# Expected: Lines showing knapsack function symbols

# Check library size (should be reasonable)
ls -lh /path/to/lib/libknapsack_cpu.a
# Expected: ~300KB (Linux) or ~1.8MB (macOS CPU)
```

## Summary

**Start simple:**
1. Test single item case first
2. Verify library linkage
3. Check array types (C.int not int)
4. Validate array sizes match n
5. Print pointers to verify memory layout

**Common mistakes:**
- ❌ Using `[]int` instead of `[]C.int`
- ❌ Forgetting to allocate selection array
- ❌ Array lengths don't match n
- ❌ Missing `-lstdc++ -lm` in LDFLAGS
- ❌ Wrong library path

**Quick wins:**
- ✅ Start with n=1 test case
- ✅ Print everything before and after call
- ✅ Use `unsafe.Pointer(&array[0])` for pointers
- ✅ Verify library with `nm` command

The knapsack library is working correctly (all C++ tests pass). The issue is likely in the CGO bindings or library linkage.
