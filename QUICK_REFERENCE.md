# Quick Reference: Platform-Specific Knapsack Libraries

## Build Commands

### Linux (Docker) - CPU Only
```bash
docker build -f docker/Dockerfile.linux-cpu -t knapsack-linux-cpu .
```
**Output**: `libknapsack_cpu.a` (274KB), `knapsack_cpu.h`

### Linux (Docker) - CUDA GPU
```bash
docker build -f docker/Dockerfile.linux-cuda -t knapsack-linux-cuda .
```
**Output**: `libknapsack_cuda.a` (~300KB), `knapsack_cuda.h`  
**Requires**: NVIDIA GPU, nvidia-docker

### macOS - Metal GPU (Default)
```bash
cd knapsack-library && mkdir build-metal && cd build-metal
cmake .. -DUSE_METAL=ON
cmake --build . --target knapsack -j
```
**Output**: `libknapsack_metal.a` (Metal GPU-accelerated)

### macOS - CPU Only
```bash
cd knapsack-library && mkdir build-cpu && cd build-cpu
cmake .. -DBUILD_CPU_ONLY=ON
cmake --build . --target knapsack -j
```
**Output**: `libknapsack_cpu.a` (CPU-only, no GPU)

### Local Build (All Platforms) - Makefile
```bash
# Build CPU-only library
make build-cpu

# Build CUDA library (requires CUDA toolkit)
make build-cuda

# Build Metal library (macOS only)
make build-metal

# Build all supported libraries for current platform
make build-all

# Clean legacy libraries (old libknapsack.a)
make clean-legacy

# Clean all build artifacts
make clean-all
```

## Go Build Tags

```go
//go:build linux && cgo && !cuda   // Uses libknapsack_cpu.a
//go:build linux && cgo && cuda    // Uses libknapsack_cuda.a (NVIDIA)
//go:build darwin && cgo           // Uses libknapsack_metal.a (Metal)
//go:build !cgo                    // Stub implementation
```

## CGO Flags

### Linux CPU-Only
```bash
CGO_ENABLED=1
CGO_LDFLAGS="-L/usr/local/lib -lknapsack_cpu -lstdc++ -lm"
```

### Linux CUDA
```bash
CGO_ENABLED=1
CGO_LDFLAGS="-L/usr/local/lib -lknapsack_cuda -lstdc++ -lm -lcudart"
```

### macOS Metal
```bash
CGO_ENABLED=1
CGO_LDFLAGS="-L/usr/local/lib -lknapsack_metal -framework Metal -framework Foundation -lstdc++ -lm"
```

## Verification

```bash
# Verify platform-specific libraries
./verify_libraries.sh

# Check CPU library (no GPU symbols)
nm knapsack-library/build-cpu/libknapsack_cpu.a | grep -i "metal\|cuda" || echo "✓ No GPU symbols"

# Check CUDA library (has CUDA symbols)
nm knapsack-library/build-cuda/libknapsack_cuda.a | grep -i cuda

# Check Metal library (has Metal symbols, macOS only)
nm knapsack-library/build-metal/libknapsack_metal.a | grep -i metal
```

## Docker Verification

```bash
# Check CPU library in Docker
docker run --rm knapsack-linux-cpu-full \
  nm /usr/local/lib/libknapsack_cpu.a | grep -i metal || echo "✓ No Metal symbols"

# Check CUDA library in Docker
docker run --gpus all --rm knapsack-linux-cuda-full \
  nm /usr/local/lib/libknapsack_cuda.a | grep -i cuda

# List library sizes
docker run --rm knapsack-linux-cpu-full ls -lh /usr/local/lib/libknapsack_cpu.a
docker run --gpus all --rm knapsack-linux-cuda-full ls -lh /usr/local/lib/libknapsack_cuda.a
```

## Key Files

| Platform | Library | Header | Expected Size | Build Command |
|----------|---------|--------|---------------|---------------|
| Linux CPU | `libknapsack_cpu.a` | `knapsack_cpu.h` | 274KB | `make build-cpu` |
| Linux CUDA | `libknapsack_cuda.a` | `knapsack_cuda.h` | ~300KB | `make build-cuda` |
| macOS Metal | `libknapsack_metal.a` | `knapsack_c.h` | ~1.7MB | `make build-metal` |

**Important**: All libraries now have platform-specific names. If you find `libknapsack.a` without a suffix, it's a legacy library. Run `make clean-legacy` to remove it.

## Documentation

- **Full Guide**: [docs/PLATFORM_SPECIFIC_LIBS.md](docs/PLATFORM_SPECIFIC_LIBS.md)
- **Integration**: [docs/GO_CHARIOT_INTEGRATION.md](docs/GO_CHARIOT_INTEGRATION.md)
- **Build Guide**: [CROSS_PLATFORM_BUILD.md](CROSS_PLATFORM_BUILD.md)
- **Summary**: [SUCCESS_SUMMARY.md](SUCCESS_SUMMARY.md)
