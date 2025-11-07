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
cd knapsack-library && mkdir build && cd build
cmake ..
cmake --build . --target knapsack -j
```
**Output**: `libknapsack.a` (1.7MB with Metal)

### macOS - CPU Only
```bash
cd knapsack-library && mkdir build-cpu && cd build-cpu
cmake .. -DBUILD_CPU_ONLY=ON
cmake --build . --target knapsack -j
```
**Output**: `libknapsack.a` (1.7MB without Metal)

## Go Build Tags

```go
//go:build linux && cgo && !cuda   // Uses libknapsack_cpu.a
//go:build linux && cgo && cuda    // Uses libknapsack_cuda.a (NVIDIA)
//go:build darwin && cgo           // Uses libknapsack.a (Metal)
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
CGO_LDFLAGS="-L/usr/local/lib -lknapsack -framework Metal -framework Foundation -lstdc++ -lm"
```

## Docker Integration

```dockerfile
FROM knapsack-linux-cpu AS knapsack-lib
FROM golang:1.21 AS builder

COPY --from=knapsack-lib /lib/libknapsack_cpu.a /usr/local/lib/
COPY --from=knapsack-lib /include/knapsack_cpu.h /usr/local/include/

ENV CGO_ENABLED=1
ENV CGO_LDFLAGS="-L/usr/local/lib -lknapsack_cpu -lstdc++ -lm"

RUN go build -tags cgo -o app ./cmd/server
```

## Verification

```bash
# Check for Metal symbols (should be none on Linux build)
docker run --rm knapsack-linux-cpu-full \
  nm /usr/local/lib/libknapsack_cpu.a | grep -i metal

# List library size
docker run --rm knapsack-linux-cpu-full \
  ls -lh /usr/local/lib/libknapsack_cpu.a
```

## Key Files

| Platform | Library | Header | Size |
|----------|---------|--------|------|
| Linux | `libknapsack_cpu.a` | `knapsack_cpu.h` | 274KB |
| macOS | `libknapsack.a` | `knapsack_c.h` | 1.7MB |

## Documentation

- **Full Guide**: [docs/PLATFORM_SPECIFIC_LIBS.md](docs/PLATFORM_SPECIFIC_LIBS.md)
- **Integration**: [docs/GO_CHARIOT_INTEGRATION.md](docs/GO_CHARIOT_INTEGRATION.md)
- **Build Guide**: [CROSS_PLATFORM_BUILD.md](CROSS_PLATFORM_BUILD.md)
- **Summary**: [SUCCESS_SUMMARY.md](SUCCESS_SUMMARY.md)
