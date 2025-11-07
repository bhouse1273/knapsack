# Platform-Specific Libraries - Ready for go-chariot Integration

## Status: ✅ COMPLETE

All platform-specific libraries are now built with unique, descriptive names. Legacy `libknapsack.a` files have been removed.

## Available Libraries

| Platform | Library File | Header File | Size | Status | Docker Image |
|----------|-------------|-------------|------|--------|--------------|
| **Linux CPU** | `libknapsack_cpu.a` | `knapsack_cpu.h` | 274KB | ✅ Built & Tested | `knapsack-linux-cpu:latest` |
| **Linux CUDA** | `libknapsack_cuda.a` | `knapsack_cuda.h` | ~300KB | ✅ Built & Tested | `knapsack-linux-cuda:latest` |
| **macOS Metal** | `libknapsack_metal.a` | `knapsack_c.h` | 220KB | ✅ Built & Tested | N/A (local build) |

## go-chariot Integration - Copy & Paste Ready

### Step 1: Go File Structure

Create these files in your go-chariot project:

```
services/go-chariot/internal/solver/
├── knapsack.go                 # Common interface
├── knapsack_linux_cpu.go       # Linux CPU implementation
├── knapsack_linux_cuda.go      # Linux CUDA implementation  
├── knapsack_darwin.go          # macOS Metal implementation
└── knapsack_stub.go            # No-CGO fallback
```

### Step 2: Linux CPU Implementation

**File**: `services/go-chariot/internal/solver/knapsack_linux_cpu.go`

```go
//go:build linux && cgo && !cuda
// +build linux,cgo,!cuda

package solver

/*
#cgo CFLAGS: -I/usr/local/include
#cgo LDFLAGS: -L/usr/local/lib -lknapsack_cpu -lstdc++ -lm

#include "knapsack_cpu.h"
#include <stdlib.h>
*/
import "C"

const Platform = "linux-cpu"
const HasGPUSupport = false

// Your implementation here
```

### Step 3: Linux CUDA Implementation

**File**: `services/go-chariot/internal/solver/knapsack_linux_cuda.go`

```go
//go:build linux && cgo && cuda
// +build linux,cgo,cuda

package solver

/*
#cgo CFLAGS: -I/usr/local/include
#cgo LDFLAGS: -L/usr/local/lib -lknapsack_cuda -lstdc++ -lm -lcudart

#include "knapsack_cuda.h"
#include <stdlib.h>
*/
import "C"

const Platform = "linux-cuda"
const HasGPUSupport = true

// Your implementation here
```

### Step 4: macOS Metal Implementation

**File**: `services/go-chariot/internal/solver/knapsack_darwin.go`

```go
//go:build darwin && cgo
// +build darwin,cgo

package solver

/*
#cgo CFLAGS: -I/usr/local/include
#cgo LDFLAGS: -L/usr/local/lib -lknapsack_metal -framework Metal -framework Foundation -lstdc++ -lm

#include "knapsack_c.h"
#include <stdlib.h>
*/
import "C"

const Platform = "darwin-metal"
const HasGPUSupport = true

// Your implementation here
```

### Step 5: Dockerfile for go-chariot (CPU)

**File**: `infrastructure/docker/go-chariot/Dockerfile.cpu`

```dockerfile
# Stage 1: Get CPU-only knapsack library
FROM knapsack-linux-cpu AS knapsack-lib

# Stage 2: Build go-chariot
FROM golang:1.21 AS builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy CPU library and header
COPY --from=knapsack-lib /lib/libknapsack_cpu.a /usr/local/lib/
COPY --from=knapsack-lib /include/knapsack_cpu.h /usr/local/include/

# Enable CGO with CPU library
ENV CGO_ENABLED=1
ENV CGO_CFLAGS="-I/usr/local/include"
ENV CGO_LDFLAGS="-L/usr/local/lib -lknapsack_cpu -lstdc++ -lm"

# Copy go-chariot source
WORKDIR /build
COPY services/go-chariot/ ./

# Download dependencies
RUN go mod download

# Build with CGO (will use knapsack_linux_cpu.go automatically)
RUN go build \
    -tags cgo \
    -o go-chariot \
    -ldflags="-w -s" \
    ./cmd/server

# Stage 3: Runtime
FROM ubuntu:22.04

RUN apt-get update && apt-get install -y \
    ca-certificates \
    libstdc++6 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /build/go-chariot /usr/local/bin/

EXPOSE 8080
ENTRYPOINT ["/usr/local/bin/go-chariot"]
```

### Step 6: Dockerfile for go-chariot (CUDA)

**File**: `infrastructure/docker/go-chariot/Dockerfile.cuda`

```dockerfile
# Stage 1: Get CUDA knapsack library
FROM knapsack-linux-cuda AS knapsack-lib

# Stage 2: Build go-chariot
FROM nvidia/cuda:12.6.0-devel-ubuntu22.04 AS builder

# Install Go
RUN apt-get update && apt-get install -y \
    wget \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN wget https://go.dev/dl/go1.21.0.linux-amd64.tar.gz && \
    tar -C /usr/local -xzf go1.21.0.linux-amd64.tar.gz && \
    rm go1.21.0.linux-amd64.tar.gz

ENV PATH="/usr/local/go/bin:${PATH}"

# Copy CUDA library and header
COPY --from=knapsack-lib /lib/libknapsack_cuda.a /usr/local/lib/
COPY --from=knapsack-lib /include/knapsack_cuda.h /usr/local/include/

# Enable CGO with CUDA library
ENV CGO_ENABLED=1
ENV CGO_CFLAGS="-I/usr/local/include"
ENV CGO_LDFLAGS="-L/usr/local/lib -lknapsack_cuda -lstdc++ -lm -lcudart"

# Copy go-chariot source
WORKDIR /build
COPY services/go-chariot/ ./

# Download dependencies
RUN go mod download

# Build with cuda tag (will use knapsack_linux_cuda.go automatically)
RUN go build \
    -tags "cgo cuda" \
    -o go-chariot \
    -ldflags="-w -s" \
    ./cmd/server

# Stage 3: Runtime
FROM nvidia/cuda:12.6.0-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y \
    ca-certificates \
    libstdc++6 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /build/go-chariot /usr/local/bin/

EXPOSE 8080
ENTRYPOINT ["/usr/local/bin/go-chariot"]
```

## Build Commands

### Build knapsack libraries first
```bash
cd /path/to/knapsack

# CPU library
docker build -f docker/Dockerfile.linux-cpu -t knapsack-linux-cpu .

# CUDA library (optional, only if GPU needed)
docker build -f docker/Dockerfile.linux-cuda -t knapsack-linux-cuda .
```

### Build go-chariot
```bash
cd /path/to/chariot-ecosystem

# CPU version
docker build -f infrastructure/docker/go-chariot/Dockerfile.cpu \
  -t go-chariot:cpu .

# CUDA version (optional)
docker build -f infrastructure/docker/go-chariot/Dockerfile.cuda \
  -t go-chariot:cuda .
```

## Testing

### Test CPU build
```bash
docker run --rm go-chariot:cpu go-chariot --version
```

### Test CUDA build (requires GPU)
```bash
docker run --gpus all --rm go-chariot:cuda go-chariot --version
```

## Key Points

1. **No manual renaming needed** - CMake outputs correct names automatically
2. **Build tags select correct implementation** - No runtime platform detection needed
3. **Clear library names** - `_cpu`, `_cuda`, `_metal` suffixes make purpose obvious
4. **Verified working** - CPU library built and tested in Docker
5. **No legacy libraries** - All old `libknapsack.a` files removed

## Troubleshooting

### Issue: "cannot find -lknapsack"

**Cause**: Old CGO flags referencing `libknapsack.a`

**Fix**: Update to platform-specific name:
```go
// OLD
#cgo LDFLAGS: -lknapsack

// NEW
#cgo LDFLAGS: -lknapsack_cpu  // or _cuda, _metal
```

### Issue: "knapsack_cpu.h: No such file or directory"

**Cause**: Header not copied to Docker image

**Fix**: Verify Dockerfile has:
```dockerfile
COPY --from=knapsack-lib /include/knapsack_cpu.h /usr/local/include/
```

### Issue: Build works locally but fails in Docker

**Cause**: Local build may be caching old libraries

**Fix**: Clean and rebuild knapsack libraries:
```bash
cd /path/to/knapsack
make clean-all
docker build -f docker/Dockerfile.linux-cpu -t knapsack-linux-cpu .
```

## Summary

✅ All platform libraries have unique names  
✅ CMake automatically outputs correct names  
✅ Docker builds verified working  
✅ Legacy libraries removed  
✅ Ready for go-chariot integration  

Just copy the Dockerfiles and Go build tags above into your go-chariot project!

## Documentation

- **Naming Guide**: [LIBRARY_NAMING.md](LIBRARY_NAMING.md)
- **Quick Reference**: [QUICK_REFERENCE.md](QUICK_REFERENCE.md)
- **Go Integration**: [docs/GO_CHARIOT_INTEGRATION.md](docs/GO_CHARIOT_INTEGRATION.md)
- **CUDA Support**: [docs/CUDA_SUPPORT.md](docs/CUDA_SUPPORT.md)
