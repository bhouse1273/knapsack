# Knapsack Cross-Platform Build - Change Summary

## Date: November 6, 2024

## Problem Statement

go-chariot Docker builds were failing on Linux AMD64 with:
```
fatal error: third_party/picojson/picojson.h: No such file or directory
fatal error: metal_api.h: No such file or directory
```

Even though the CROSS_PLATFORM_BUILD.md documentation claimed the fixes were implemented, the CMakeLists.txt still had conditional include paths that only worked on macOS.

## Root Cause

1. **CMakeLists.txt include paths were conditional**: The Metal directory was only included on Apple platforms using CMake generator expressions: `$<$<BOOL:${APPLE}>:${KERNELS_METAL_DIR}>`

2. **Wrong include path for third_party**: The path pointed to `../third_party` instead of `..` (project root), causing the compiler to look for `third_party/third_party/picojson/picojson.h`

3. **Duplicate target_include_directories**: Previous edits had created duplicate sections

## Changes Made

### 1. Fixed knapsack-library/CMakeLists.txt

**Location**: Lines 38-50

**Before**:
```cmake
target_include_directories(knapsack
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
    ${CMAKE_CURRENT_SOURCE_DIR}/../third_party
    $<$<BOOL:${APPLE}>:${KERNELS_METAL_DIR}>  # ❌ Only on Apple!
)
```

**After**:
```cmake
# Set Metal directory path (always needed for headers, even on Linux)
set(KERNELS_METAL_DIR "${CMAKE_CURRENT_LIST_DIR}/../kernels/metal")

target_include_directories(knapsack
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
    ${PROJ_ROOT}                    # ✅ Project root (not third_party/)
    ${KERNELS_METAL_DIR}           # ✅ Unconditional Metal headers
)
```

### 2. Simplified docker/Dockerfile.builder

Removed manual CMakeLists.txt patching since the fixes are now committed:

**Removed**:
```dockerfile
# Fix CMakeLists.txt to include kernels/metal directory
RUN cd /build/knapsack/knapsack-library && \
    sed -i '5 a set(KERNELS_METAL_DIR ...)' CMakeLists.txt && \
    sed -i 's|...|...|' CMakeLists.txt
```

**Added**:
```dockerfile
# Build with USE_METAL=OFF (no Metal framework linking on Linux)
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=/usr/local \
    -DUSE_METAL=OFF
```

### 3. Updated .dockerignore

Added `knapsack-library/build/` to prevent local build artifacts from being copied:

```
# Build directories
build/
build-*/
cmake-build-*/
.cmake/
knapsack-library/build/  # ✅ Added
```

## Verification

### macOS Build (with Metal)
```bash
cd knapsack-library && rm -rf build && mkdir -p build && cd build
cmake .. && cmake --build . --target knapsack -j4
```
**Result**: ✅ Built target knapsack (with Metal support)

### Linux Docker Build (CPU-only)
```bash
docker build -f docker/Dockerfile.builder -t knapsack-builder .
```
**Result**: ✅ Successfully built

### Artifacts
```bash
docker run --rm knapsack-builder-full \
  ls -lh /usr/local/lib/libknapsack.a /usr/local/include/knapsack_c.h
```
**Result**:
- ✅ libknapsack.a: 274K
- ✅ knapsack_c.h: 1.6K

## Key Insights

1. **CMake generator expressions are platform-specific**: Using `$<$<BOOL:${APPLE}>:...>` makes includes conditional at configure time, not just compile time

2. **Include paths must point to parent of include statement**: If source says `#include "third_party/picojson.h"`, the include path should be the project root, not `third_party/`

3. **Metal headers can exist on all platforms**: As long as the code using Metal types is wrapped in `#ifdef __APPLE__`, the headers can be present on Linux

4. **Docker build context matters**: `.dockerignore` is critical to prevent local build artifacts from polluting the Docker build

## Documentation Updated

1. ✅ `CROSS_PLATFORM_BUILD.md` - Added CMakeLists.txt fix section
2. ✅ `docs/GO_CHARIOT_INTEGRATION.md` - Created complete integration guide

## Next Steps for go-chariot Integration

1. Create `Dockerfile.build` in chariot-ecosystem repository
2. Implement Go CGO bindings with build tags (`//go:build cgo`)
3. Add CGO flags: `CGO_LDFLAGS="-L/usr/local/lib -lknapsack -lstdc++ -lm"`
4. Test build and deployment to Azure

## Files Modified

- `knapsack-library/CMakeLists.txt` (lines 38-50)
- `docker/Dockerfile.builder` (simplified build steps)
- `.dockerignore` (added knapsack-library/build/)
- `CROSS_PLATFORM_BUILD.md` (documentation update)
- `docs/GO_CHARIOT_INTEGRATION.md` (new file)

## Git Commit Message

```
fix: update CMakeLists.txt for cross-platform builds

- Make Metal header directory unconditionally included
- Fix include path for third_party (use project root)
- Remove duplicate target_include_directories
- Simplify Dockerfile (no manual patching needed)
- Update .dockerignore to exclude library build/

This fixes go-chariot Docker builds on Linux AMD64 by ensuring
all required headers (picojson, metal_api.h) are found during
compilation, while Metal API usage remains guarded by #ifdef __APPLE__.

Verified:
- macOS build with Metal: ✅
- Linux Docker build (CPU-only): ✅
- Artifacts: libknapsack.a (274K), knapsack_c.h (1.6K)
```
