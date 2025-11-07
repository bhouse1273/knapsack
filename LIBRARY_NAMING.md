# Platform-Specific Library Naming Convention

## Overview

All knapsack libraries now use **platform-specific names** to eliminate confusion and prevent linking errors in go-chariot and other projects.

## Library Naming Standard

| Platform | Library Name | Header Name | Purpose |
|----------|-------------|-------------|---------|
| **Linux CPU** | `libknapsack_cpu.a` | `knapsack_cpu.h` | CPU-only, no GPU dependencies |
| **Linux CUDA** | `libknapsack_cuda.a` | `knapsack_cuda.h` | NVIDIA GPU acceleration |
| **macOS Metal** | `libknapsack_metal.a` | `knapsack_c.h` | Apple Metal GPU acceleration |

## Important Changes

### ❌ Old (Legacy) Naming
```
libknapsack.a          # Ambiguous - which platform?
knapsack_c.h           # Generic header
```

**Problem**: A single `libknapsack.a` name was used for all platforms, causing:
- Confusion about which platform variant was being used
- Linking errors when wrong library was picked up
- Difficulty debugging platform-specific issues
- Build cache issues in multi-platform projects

### ✅ New (Platform-Specific) Naming
```
libknapsack_cpu.a      # Clearly CPU-only
libknapsack_cuda.a     # Clearly CUDA GPU
libknapsack_metal.a    # Clearly Metal GPU
```

**Benefits**:
- ✅ No ambiguity - name tells you exactly what you're using
- ✅ Can build multiple variants side-by-side
- ✅ CGO flags explicitly reference correct library
- ✅ Docker images contain only what they need
- ✅ Easier to debug linking issues

## Building Libraries

### Automatic Platform-Specific Names

CMakeLists.txt now automatically sets the output name based on build options:

```cmake
if(BUILD_CPU_ONLY)
  set_target_properties(knapsack PROPERTIES OUTPUT_NAME "knapsack_cpu")
elseif(BUILD_CUDA)
  set_target_properties(knapsack PROPERTIES OUTPUT_NAME "knapsack_cuda")
elseif(USE_METAL)
  set_target_properties(knapsack PROPERTIES OUTPUT_NAME "knapsack_metal")
endif()
```

### Build Commands

```bash
# CPU-only library
make build-cpu
# Output: knapsack-library/build-cpu/libknapsack_cpu.a

# CUDA library (requires CUDA toolkit)
make build-cuda
# Output: knapsack-library/build-cuda/libknapsack_cuda.a

# Metal library (macOS only)
make build-metal
# Output: knapsack-library/build-metal/libknapsack_metal.a
```

## Docker Integration

### CPU-Only Dockerfile (docker/Dockerfile.linux-cpu)

```dockerfile
# Build produces libknapsack_cpu.a directly
RUN cmake .. -DBUILD_CPU_ONLY=ON && \
    cmake --build . --target knapsack -j$(nproc) && \
    cp libknapsack_cpu.a /usr/local/lib/libknapsack_cpu.a
```

**No renaming needed** - CMake outputs `libknapsack_cpu.a` automatically.

### CUDA Dockerfile (docker/Dockerfile.linux-cuda)

```dockerfile
# Build produces libknapsack_cuda.a directly
RUN cmake .. -DBUILD_CUDA=ON && \
    cmake --build . --target knapsack -j$(nproc) && \
    cp libknapsack_cuda.a /usr/local/lib/libknapsack_cuda.a
```

**No renaming needed** - CMake outputs `libknapsack_cuda.a` automatically.

## Go Integration

### CGO Linker Flags

Each platform uses the appropriate library name:

```go
// knapsack_linux_cpu.go
//go:build linux && cgo && !cuda

package solver

/*
#cgo LDFLAGS: -L/usr/local/lib -lknapsack_cpu -lstdc++ -lm
#include "knapsack_cpu.h"
*/
import "C"
```

```go
// knapsack_linux_cuda.go
//go:build linux && cgo && cuda

package solver

/*
#cgo LDFLAGS: -L/usr/local/lib -lknapsack_cuda -lstdc++ -lm -lcudart
#include "knapsack_cuda.h"
*/
import "C"
```

```go
// knapsack_darwin.go
//go:build darwin && cgo

package solver

/*
#cgo LDFLAGS: -L/usr/local/lib -lknapsack_metal -framework Metal -framework Foundation -lstdc++ -lm
#include "knapsack_c.h"
*/
import "C"
```

## Cleaning Up Legacy Libraries

### Finding Legacy Libraries

```bash
# Find any libknapsack.a without platform suffix
find . -name "libknapsack.a" -type f ! -path "*/third_party/*"
```

### Automatic Cleanup

```bash
# Remove all legacy libraries
make clean-legacy
```

This removes any `libknapsack.a` files that don't have the platform-specific suffix.

### Complete Clean

```bash
# Remove all build artifacts and legacy libraries
make clean-all
```

## Verification

### Quick Verification

```bash
# Run verification script
./verify_libraries.sh
```

This checks:
- ✅ No legacy `libknapsack.a` files exist
- ✅ Platform-specific libraries have correct names
- ✅ CPU library has no GPU symbols
- ✅ CUDA library has CUDA symbols
- ✅ Metal library has Metal symbols (macOS)

### Manual Verification

```bash
# Check CPU library has no GPU symbols
nm knapsack-library/build-cpu/libknapsack_cpu.a | grep -i "metal\|cuda"
# Should output nothing

# Check CUDA library has CUDA symbols
nm knapsack-library/build-cuda/libknapsack_cuda.a | grep -i cuda
# Should show CUDA-related symbols

# Check Metal library has Metal symbols
nm knapsack-library/build-metal/libknapsack_metal.a | grep -i metal
# Should show Metal-related symbols
```

## Migration Guide

### For Existing Projects

If you have an existing project using the old `libknapsack.a`:

1. **Identify which variant you need**:
   - CPU-only? → `libknapsack_cpu.a`
   - NVIDIA GPU? → `libknapsack_cuda.a`
   - Apple Metal? → `libknapsack_metal.a`

2. **Update CGO flags**:
   ```go
   // Old
   #cgo LDFLAGS: -lknapsack
   
   // New (CPU)
   #cgo LDFLAGS: -lknapsack_cpu
   ```

3. **Update header includes** (if needed):
   ```c
   // CPU or CUDA builds
   #include "knapsack_cpu.h"    // or knapsack_cuda.h
   
   // Metal builds
   #include "knapsack_c.h"
   ```

4. **Rebuild**:
   ```bash
   make clean-all
   make build-cpu  # or build-cuda, build-metal
   ```

### For Docker Images

Update your Dockerfile to copy the correct library:

```dockerfile
# Old
COPY --from=knapsack-lib /lib/libknapsack.a /usr/local/lib/

# New (CPU)
COPY --from=knapsack-lib /lib/libknapsack_cpu.a /usr/local/lib/

# New (CUDA)
COPY --from=knapsack-lib /lib/libknapsack_cuda.a /usr/local/lib/
```

## Troubleshooting

### Error: "cannot find -lknapsack"

**Cause**: Your code is trying to link against old `libknapsack.a`

**Solution**: Update CGO flags to use platform-specific name:
```go
#cgo LDFLAGS: -lknapsack_cpu  // or _cuda, _metal
```

### Error: Library file not found

**Cause**: Library wasn't built or is in wrong location

**Solution**:
```bash
# Rebuild the specific library you need
make build-cpu    # or build-cuda, build-metal

# Check it was created
ls -l knapsack-library/build-*/libknapsack_*.a
```

### Error: "undefined reference to knapsack_solve_v2"

**Cause**: Wrong library variant or missing library

**Solution**:
1. Verify correct library is in path: `ls /usr/local/lib/libknapsack_*.a`
2. Check CGO flags match the library name
3. Rebuild library: `make build-cpu` (or appropriate variant)

### Mixed Platform Symbols

**Symptoms**: Build succeeds but runtime errors, or unexpected behavior

**Solution**:
```bash
# Clean everything and rebuild
make clean-all
make build-cpu  # Build only what you need

# Verify no legacy libraries
./verify_libraries.sh
```

## Summary

- ✅ All libraries now have platform-specific names
- ✅ CMake automatically outputs correct names
- ✅ No manual renaming in Dockerfiles
- ✅ Clear separation of CPU vs GPU variants
- ✅ Easy to verify and debug
- ✅ No ambiguity or confusion

For more information:
- **Quick Reference**: [QUICK_REFERENCE.md](QUICK_REFERENCE.md)
- **Build Guide**: [CROSS_PLATFORM_BUILD.md](CROSS_PLATFORM_BUILD.md)
- **Go Integration**: [docs/GO_CHARIOT_INTEGRATION.md](docs/GO_CHARIOT_INTEGRATION.md)
