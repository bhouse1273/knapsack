# Cross-Platform Build Documentation

## Overview

The knapsack library now supports cross-platform builds with proper Metal API guards, allowing:
- ✅ **macOS Apple Silicon**: Full Metal GPU acceleration
- ✅ **Linux AMD64**: CPU-only mode (for Azure deployment)
- ✅ **Automatic fallback**: Metal usage automatically disabled on non-Apple platforms

## Changes Made

### 1. CMakeLists.txt Fix for Cross-Platform Builds

The `knapsack-library/CMakeLists.txt` has been updated to ensure all required include directories are available on all platforms:

**Key Changes:**
```cmake
# Set Metal directory path (always needed for headers, even on Linux)
set(KERNELS_METAL_DIR "${CMAKE_CURRENT_LIST_DIR}/../kernels/metal")

# Public headers for consumers + private includes for our sources
# Note: Include PROJ_ROOT (../) not (../third_party) so that
# source files can use #include "third_party/..." paths
target_include_directories(knapsack
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
    ${PROJ_ROOT}                    # Project root for third_party/ includes
    ${KERNELS_METAL_DIR}           # Metal headers (unconditionally)
)
```

**What This Fixes:**
- ✅ `third_party/picojson/picojson.h` is found by including project root
- ✅ `metal_api.h` is found by including `kernels/metal` directory on all platforms
- ✅ Metal API is only *used* on Apple platforms (via `#ifdef __APPLE__` guards)

### 2. Metal Header Include Guards

### 2. Metal Header Include Guards

All Metal API includes are now guarded with `#ifdef __APPLE__`:

**Files Modified:**
- `knapsack-library/src/knapsack_solve.cpp`
- `src/v2/BeamSearch.cpp`
- `src/RecursiveSolver.cpp`
- `tools/v2_metal_parity_sanity.cpp`
- `tools/v2_multi_constraint_parity.cpp`
- `tools/v2_assign_sanity.cpp`

**Pattern:**
```cpp
// Metal API (only available on Apple)
#ifdef __APPLE__
#include "metal_api.h"
#endif
```

### 3. Metal Usage Guards

All code that uses Metal types (`MetalEvalIn`, `MetalEvalOut`) and functions (`knapsack_metal_eval`, etc.) is now wrapped in compile-time guards:

**Pattern:**
```cpp
bool useMetal = false;
#ifdef __APPLE__
useMetal = true;
if (useMetal) {
    MetalEvalIn in{};
    // ... Metal setup ...
    MetalEvalOut out{ obj.data(), pen.data() };
    if (knapsack_metal_eval(&in, &out, nullptr, 0) != 0) {
        useMetal = false; // fallback
    }
}
if (!useMetal)
#endif
{
    // CPU fallback code
}
```

## Building

### macOS (with Metal)
```bash
cd knapsack-library
mkdir -p build && cd build
cmake ..
cmake --build . --target knapsack -j
```

**Result:**
- Compiles with Metal support enabled
- Runtime Metal shader compilation
- Automatic CPU fallback if Metal unavailable

### Linux (CPU-only via Docker)

**Simplified Dockerfile** (no manual CMakeLists.txt patching needed):

```bash
cd knapsack
docker build -f docker/Dockerfile.builder -t knapsack-builder .
```

**Result:**
- Compiles without Metal dependency
- Pure CPU implementation
- No Metal headers/libraries required
- Output: `libknapsack.a` (274 KB) + `knapsack_c.h`

## Verification

### Test macOS Build
```bash
cd knapsack-library/build
ls -lh libknapsack.a
file libknapsack.a
# Expected: Mach-O 64-bit staticlib arm64
```

### Test Linux Build
```bash
docker run --rm knapsack-builder-full \
  ls -lh /usr/local/lib/libknapsack.a /usr/local/include/knapsack_c.h
# Expected: -rw-r--r-- 274K libknapsack.a
```

### Test Cross-Compilation
```bash
docker build --target builder -t knapsack-builder-full \
  -f docker/Dockerfile.builder .
docker run --rm knapsack-builder-full file /usr/local/lib/libknapsack.a
# Expected: ELF 64-bit LSB ... x86-64
```

## Integration with go-chariot

### Option 1: Multi-stage Docker Build

```dockerfile
# Stage 1: Build knapsack library
FROM knapsack-builder AS knapsack-lib

# Stage 2: Build go-chariot
FROM golang:1.21 AS builder

# Copy knapsack library and headers
COPY --from=knapsack-lib /lib/libknapsack.a /usr/local/lib/
COPY --from=knapsack-lib /include/knapsack_c.h /usr/local/include/

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set CGO flags
ENV CGO_ENABLED=1
ENV CGO_CFLAGS="-I/usr/local/include"
ENV CGO_LDFLAGS="-L/usr/local/lib -lknapsack -lstdc++ -lm"

WORKDIR /build
COPY services/go-chariot/ ./

# Build
RUN go build -o go-chariot-linux-amd64 ./cmd/server
```

### Option 2: Go Code with Build Tags

```go
//go:build cgo
// +build cgo

package solver

/*
#cgo CFLAGS: -I/usr/local/include
#cgo LDFLAGS: -L/usr/local/lib -lknapsack -lstdc++ -lm

#include "knapsack_c.h"
#include <stdlib.h>
*/
import "C"

// KnapsackSolver with CGO enabled
type KnapsackSolver struct {
    // Implementation using C library
}
```

## Platform-Specific Behavior

### macOS Apple Silicon
- **Compilation**: Uses Objective-C++ compiler for `.mm` files
- **Runtime**: Attempts Metal initialization, falls back to CPU if failed
- **Dependencies**: Metal framework, Foundation framework
- **Performance**: GPU-accelerated beam search (when Metal available)

### Linux AMD64
- **Compilation**: Pure C++ compilation, no Objective-C++
- **Runtime**: CPU-only implementation
- **Dependencies**: libstdc++, libm (standard libraries)
- **Performance**: CPU-based beam search

## Troubleshooting

### Build Fails with "Metal API not found"
- **Cause**: Trying to use Metal on non-Apple platform
- **Solution**: Ensure all Metal usage is wrapped in `#ifdef __APPLE__`

### Linker Error: "undefined reference to Metal functions"
- **Cause**: Metal functions called outside `#ifdef` guards
- **Solution**: Wrap both type declarations AND function calls

### CGO Compilation Fails
- **Cause**: Missing C++ standard library or math library
- **Solution**: Add `-lstdc++ -lm` to `CGO_LDFLAGS`

## Performance Notes

- **macOS with Metal**: ~10-100x faster for large problems (GPU acceleration)
- **macOS without Metal**: Same as Linux CPU performance
- **Linux CPU-only**: Suitable for medium-sized problems (<10K items)
- **CPU Fallback**: Always available as safety net

## Future Improvements

1. **CUDA Support**: Add Linux GPU support via CUDA (conditional compilation)
2. **OpenCL Backend**: Platform-agnostic GPU acceleration
3. **WASM Build**: Browser-based solver (CPU-only)
4. **ARM Linux**: Support for ARM-based cloud instances

## References

- **Main Documentation**: `README.md`
- **Docker Build**: `docker/Dockerfile.builder`
- **C API**: `knapsack-library/include/knapsack_c.h`
- **Python Bindings**: `bindings/python/README.md`
- **Go Bindings**: `bindings/go/metal/README.md`
