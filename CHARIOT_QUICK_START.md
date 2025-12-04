# Quick Reference: Chariot CGO Integration

## TL;DR for Chariot Team

Your knapsack library integration is failing. Here's the quickest path to debug it:

## Step 1: Copy This Test (2 minutes)

```go
package main

/*
#cgo CFLAGS: -I/path/to/knapsack/include
#cgo linux LDFLAGS: -L/path/to/knapsack/lib/linux-cpu -lknapsack_cpu -lstdc++ -lm
#cgo darwin LDFLAGS: -L/path/to/knapsack/lib/macos-metal -lknapsack_metal -lstdc++ -lm

#include "knapsack_c.h"
*/
import "C"
import (
	"fmt"
	"unsafe"
)

func main() {
	n := C.int(1)
	weights := []C.int{C.int(5)}
	values := []C.int{C.int(10)}
	capacity := C.int(10)
	selection := make([]C.int, 1)
	
	totalValue := C.knapsack(
		n,
		(*C.int)(unsafe.Pointer(&weights[0])),
		(*C.int)(unsafe.Pointer(&values[0])),
		capacity,
		(*C.int)(unsafe.Pointer(&selection[0])),
	)
	
	fmt.Printf("Result: totalValue=%d, selection=%v\n", totalValue, selection)
	
	if totalValue == 10 && selection[0] == 1 {
		fmt.Println("✅ SUCCESS")
	} else {
		fmt.Println("❌ FAILED")
	}
}
```

**Expected Output:**
```
Result: totalValue=10, selection=[1]
✅ SUCCESS
```

## Step 2: Fix the Paths (1 minute)

Update the `#cgo` directives to point to your actual library location:

```go
#cgo CFLAGS: -I/actual/path/to/knapsack/include
#cgo linux LDFLAGS: -L/actual/path/to/knapsack/lib/linux-cpu -lknapsack_cpu -lstdc++ -lm
```

## Step 3: Verify Library (1 minute)

```bash
# Check file exists
ls -la /path/to/knapsack/lib/linux-cpu/libknapsack_cpu.a

# Check it has the knapsack symbol
nm -g /path/to/knapsack/lib/linux-cpu/libknapsack_cpu.a | grep knapsack
```

## Common Errors & Quick Fixes

| Error | Fix |
|-------|-----|
| `undefined reference to 'knapsack'` | Add `-lstdc++ -lm` to LDFLAGS |
| `cannot use weights (type []int)` | Use `[]C.int` not `[]int` |
| `panic: index out of range` | Allocate selection: `make([]C.int, n)` |
| `no such file: knapsack_c.h` | Fix CFLAGS path to include directory |

## The 3 Critical Rules

1. **Arrays must be `[]C.int`** - Not `[]int`, not `[]int32`
2. **Must link C++ stdlib** - Add `-lstdc++ -lm` to LDFLAGS
3. **Use `unsafe.Pointer(&array[0])`** - Not just `&array`

## Full Documentation

See **`CHARIOT_CGO_DEBUG_GUIDE.md`** for:
- Complete error reference
- Docker testing approach
- Alternative JSON API
- Input validation helpers
- More test cases

## Library Locations

In the knapsack repo:
```
knapsack-library/lib/
├── linux-cpu/libknapsack_cpu.a        ← Use this for Linux
├── linux-cuda/libknapsack_cuda.a      ← Use this for GPU
└── macos-metal/libknapsack_metal.a    ← Use this for macOS
```

Each directory also contains:
- Header file: `knapsack_cpu.h` → include as `knapsack_c.h`
- RL library: `librl_support.a` (if using RL features)

## Still Not Working?

Share with us:
1. Your exact CGO directives
2. The error message
3. Output of: `ls -la /your/lib/path/`
4. Output of: `go env CGO_ENABLED`

## Success Checklist

- [ ] Test program compiles without errors
- [ ] Returns totalValue=10
- [ ] selection=[1]
- [ ] Prints "✅ SUCCESS"

Once this works, you can move to real data!
