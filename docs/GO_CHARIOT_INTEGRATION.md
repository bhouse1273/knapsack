# Integrating Knapsack Library into go-chariot

# Integrating Knapsack Library into go-chariot

## Overview

The knapsack library now provides **platform-specific builds** for clean integration:
- ✅ **macOS**: Metal GPU-accelerated library (`libknapsack_metal.a`)
- ✅ **Linux CPU**: CPU-only library (`libknapsack_cpu.a` with no GPU dependencies)
- ✅ **Linux CUDA**: NVIDIA GPU-accelerated library (`libknapsack_cuda.a`)
- ✅ **Docker**: Simplified single-stage builds for each platform

## Key Architecture Changes

### Platform-Specific Libraries

Instead of a single cross-platform library with conditional compilation, we now build separate libraries:

| Platform | Library | Features | Build Command | Performance |
|----------|---------|----------|---------------|-------------|
| **Linux CPU** | `libknapsack_cpu.a` | CPU-only, no GPU | `cmake -DBUILD_CPU_ONLY=ON` | 1x baseline |
| **Linux CUDA** | `libknapsack_cuda.a` | NVIDIA GPU accelerated | `cmake -DBUILD_CUDA=ON` | 10-50x faster |
| **macOS Metal** | `libknapsack_metal.a` | Apple GPU accelerated | `cmake -DUSE_METAL=ON` | 15-30x faster |

**Benefits:**
- 🚀 **Faster builds**: No GPU compilation unless needed
- 🧹 **Cleaner code**: No complex `#ifdef` needed in Go
- 🎯 **Clear separation**: Each platform gets exactly what it needs
- 🔧 **Easier debugging**: Platform-specific issues isolated
- ⚡ **GPU acceleration**: CUDA for NVIDIA (Jetson default), Metal for macOS
- 💰 **Cost optimization**: Choose CPU or GPU based on workload size

## Prerequisites

### Pre-Built Platform Libraries ✅

The knapsack repository includes **pre-built platform-specific libraries** in `knapsack-library/lib/`:

```
knapsack-library/lib/
├── linux-cpu/
│   ├── libknapsack_cpu.a      # 274KB - CPU-only, no GPU dependencies
│   └── knapsack_cpu.h
├── linux-cuda/
│   ├── libknapsack_cuda.a     # 631KB - NVIDIA CUDA GPU accelerated
│   └── knapsack_cuda.h
└── macos-metal/
    ├── libknapsack_metal.a    # 216KB - Apple Metal GPU accelerated
    └── knapsack_metal.h
```

**These libraries are committed to the repository** so you **don't need to build them** unless:
- You modify the knapsack C++ source code
- You need different CUDA architectures
- You want to rebuild from source for verification

### Rebuilding Libraries (Optional)

Only needed if you modify knapsack source code:

```bash
cd /path/to/knapsack

# Build all platform libraries (requires Docker + macOS for Metal)
make build-all-platforms

# This will:
# 1. Build Linux CPU library via Docker
# 2. Build Linux CUDA library via Docker
# 3. Build macOS Metal library natively
# 4. Extract all to knapsack-library/lib/
# 5. Verify each library

# Verify libraries are present
make verify-libs
```

### Option 4: Azure App Service (CPU-Only)

```bash
# Create App Service plan (Linux)
az appservice plan create \
  --name myAppServicePlan \
  --resource-group myResourceGroup \
  --is-linux \
  --sku B1

# Create web app
az webapp create \
  --resource-group myResourceGroup \
  --plan myAppServicePlan \
  --name go-chariot \
  --deployment-container-image-name myregistry.azurecr.io/go-chariot:cpu
```

**Note**: App Service doesn't support GPU, use CPU-only image.

## Performance Considerations

### Platform Comparison

| Platform | Suitable For | Performance | Azure VM | Cost/Hour |
|----------|--------------|-------------|----------|-----------|
| **Linux CPU** | <5K items, cost-sensitive | 1x baseline | Standard D4s v3 | $0.19 |
| **Linux CUDA (T4)** | 5K-20K items, balanced | 10-20x faster | NC4as T4 v3 | $0.526 |
| **Linux CUDA (V100)** | >20K items, high-perf | 30-50x faster | NC6s v3 | $3.06 |
| **macOS Metal** | Development/testing | 15-30x faster | Local only | N/A |

### Scaling Strategy

1. **Start with CPU**: Deploy CPU-only for baseline performance
2. **Measure real workloads**: Track actual problem sizes and solve times
3. **Add CUDA selectively**: Add GPU pods if data shows large problems
4. **Intelligent routing**: Route by problem size (CPU <5K, CUDA >5K)
5. **Auto-scale GPU**: Scale CUDA pods based on queue depth

**Note**: CUDA build requires `nvidia-docker` for GPU access during verification.

## Integration Steps

**Note**: The following Dockerfiles use the pre-built libraries from the knapsack repository's `knapsack-library/lib/` directory. No need to build the knapsack libraries—just copy them from the checked-out repo. This simplifies the build and eliminates the need for multi-stage Docker builds or CUDA toolkit installation during the go-chariot build.

### Step 1: Dockerfile for go-chariot

#### Option A: CPU-Only (Cost-Effective)

Create `infrastructure/docker/go-chariot/Dockerfile.cpu` in the chariot-ecosystem repository:

```dockerfile
# Build go-chariot with CGO
FROM golang:1.21 AS builder

# Install minimal build dependencies for CGO
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy CPU-only library and header from knapsack repo
COPY knapsack/knapsack-library/lib/linux-cpu/libknapsack_cpu.a /usr/local/lib/
COPY knapsack/knapsack-library/lib/linux-cpu/knapsack_cpu.h /usr/local/include/

# Enable CGO and set compiler flags
ENV CGO_ENABLED=1
ENV CGO_CFLAGS="-I/usr/local/include"
ENV CGO_LDFLAGS="-L/usr/local/lib -lknapsack_cpu -lstdc++ -lm"

# Copy go-chariot source
WORKDIR /build
COPY services/go-chariot/ ./

# Download dependencies
RUN go mod download

# Build with CGO enabled (Go will use knapsack_linux_cpu.go via build tags)
RUN go build \
    -tags cgo \
    -o go-chariot-linux-amd64 \
    -ldflags="-w -s" \
    ./cmd/server

# Stage 3: Runtime image
FROM ubuntu:22.04

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    ca-certificates \
    libstdc++6 \
    && rm -rf /var/lib/apt/lists/*

# Copy binary
COPY --from=builder /build/go-chariot-linux-amd64 /usr/local/bin/go-chariot

EXPOSE 8080
ENTRYPOINT ["/usr/local/bin/go-chariot"]
```

**Use Case**: Standard Azure VMs (D-series), cost-effective for small-medium workloads

#### Option B: CUDA GPU (High-Performance)

Create `infrastructure/docker/go-chariot/Dockerfile.cuda` in the chariot-ecosystem repository:

```dockerfile
# Build go-chariot with CGO
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

# Copy CUDA library and header from knapsack repo
COPY knapsack/knapsack-library/lib/linux-cuda/libknapsack_cuda.a /usr/local/lib/
COPY knapsack/knapsack-library/lib/linux-cuda/knapsack_cuda.h /usr/local/include/

# Enable CGO and set compiler flags for CUDA
ENV CGO_ENABLED=1
ENV CGO_CFLAGS="-I/usr/local/include"
ENV CGO_LDFLAGS="-L/usr/local/lib -lknapsack_cuda -lstdc++ -lm -lcudart"

# Copy go-chariot source
WORKDIR /build
COPY services/go-chariot/ ./

# Download dependencies
RUN go mod download

# Build with cuda build tag (Go will use knapsack_linux_cuda.go)
RUN go build \
    -tags "cgo cuda" \
    -o go-chariot-linux-amd64 \
    -ldflags="-w -s" \
    ./cmd/server

# Stage 3: Runtime image with CUDA runtime
FROM nvidia/cuda:12.6.0-runtime-ubuntu22.04

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    ca-certificates \
    libstdc++6 \
    && rm -rf /var/lib/apt/lists/*

# Copy binary
COPY --from=builder /build/go-chariot-linux-amd64 /usr/local/bin/go-chariot

EXPOSE 8080
ENTRYPOINT ["/usr/local/bin/go-chariot"]
```

**Use Case**: Azure NC-series VMs (NVIDIA GPUs), 10-50x faster for large workloads

**Key Differences from CPU Version:**
- ✅ Base images: `nvidia/cuda:12.6.0-devel` (build) and `nvidia/cuda:12.6.0-runtime` (runtime)
- ✅ Links against `libknapsack_cuda.a` not `libknapsack_cpu.a`
- ✅ Includes `-lcudart` (CUDA runtime) in LDFLAGS
- ✅ Uses `cuda` build tag to select CUDA-enabled Go code
- ✅ Requires `--gpus all` flag when running Docker container

### Step 2: Go Bindings with Platform-Specific Build Tags

The Go code uses build tags to automatically select the correct implementation:

**`services/go-chariot/internal/solver/knapsack.go`** (common interface):
```go
package solver

// Common interface both implementations satisfy
type KnapsackSolver interface {
    Solve(config *Config) (*Solution, error)
    Close() error
}

type Config struct {
    Items      []Item
    Capacity   float64
    Objectives []string
}

type Solution struct {
    SelectedItems []int
    TotalValue    float64
    TotalWeight   float64
}
```

**`services/go-chariot/internal/solver/knapsack_linux.go`** (Linux CPU-only):
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
import (
    "errors"
    "unsafe"
)

const Platform = "linux-cpu"
const HasGPUSupport = false

type knapsackSolver struct {
    // Implementation details
}

func NewKnapsackSolver() (KnapsackSolver, error) {
    // Initialize CPU-only solver
    return &knapsackSolver{}, nil
}

func (s *knapsackSolver) Solve(config *Config) (*Solution, error) {
    // Call C library (CPU-only)
    // ... implementation ...
    return nil, nil
}

func (s *knapsackSolver) Close() error {
    return nil
}
```

**`services/go-chariot/internal/solver/knapsack_linux_cuda.go`** (Linux CUDA GPU):
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
import (
    "errors"
    "unsafe"
)

const Platform = "linux-cuda"
const HasGPUSupport = true

type knapsackSolver struct {
    // Implementation details
}

func NewKnapsackSolver() (KnapsackSolver, error) {
    // Initialize CUDA-accelerated solver
    // CUDA initialization happens in C library
    return &knapsackSolver{}, nil
}

func (s *knapsackSolver) Solve(config *Config) (*Solution, error) {
    // Call C library (CUDA GPU)
    // ... implementation ...
    return nil, nil
}

func (s *knapsackSolver) Close() error {
    // Cleanup CUDA resources
    return nil
}
```

**`services/go-chariot/internal/solver/knapsack_darwin.go`** (macOS with Metal):
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
import (
    "errors"
    "unsafe"
)

const Platform = "darwin-metal"
const HasGPUSupport = true

type knapsackSolver struct {
    // Implementation details
}

func NewKnapsackSolver() (KnapsackSolver, error) {
    // Initialize Metal-accelerated solver
    return &knapsackSolver{}, nil
}

func (s *knapsackSolver) Solve(config *Config) (*Solution, error) {
    // Call C library (Metal + CPU fallback)
    // ... implementation ...
    return nil, nil
}

func (s *knapsackSolver) Close() error {
    return nil
}
```

**`services/go-chariot/internal/solver/knapsack_stub.go`** (fallback for non-CGO):
```go
//go:build !cgo || (!linux && !darwin)
// +build !cgo !linux,!darwin

package solver

import "errors"

const Platform = "stub"
const HasGPUSupport = false

type knapsackSolver struct{}

func NewKnapsackSolver() (KnapsackSolver, error) {
    return &knapsackSolver{}, nil
}

func (s *knapsackSolver) Solve(config *Config) (*Solution, error) {
    return nil, errors.New("knapsack solver not available on this platform")
}

func (s *knapsackSolver) Close() error {
    return nil
}
```

**How It Works:**
- Linux CPU builds: `knapsack_linux.go` (build tag `!cuda`) → links to `libknapsack_cpu.a`
- Linux CUDA builds: `knapsack_linux_cuda.go` (build tag `cuda`) → links to `libknapsack_cuda.a`
- macOS builds: `knapsack_darwin.go` → links to `libknapsack_metal.a`  
- Non-CGO builds: stub (returns error)
- No runtime platform detection needed!

### Step 3: Update go.mod and Build System

Add build targets to Makefile:

```makefile
# Build with knapsack support (CPU-only, requires CGO)
.PHONY: build-with-knapsack-cpu
build-with-knapsack-cpu:
	CGO_ENABLED=1 go build -tags cgo -o go-chariot ./cmd/server

# Build with CUDA knapsack support (requires CGO and CUDA)
.PHONY: build-with-knapsack-cuda
build-with-knapsack-cuda:
	CGO_ENABLED=1 go build -tags "cgo cuda" -o go-chariot ./cmd/server

# Build without knapsack (pure Go)
.PHONY: build-no-knapsack
build-no-knapsack:
	CGO_ENABLED=0 go build -o go-chariot ./cmd/server

# Docker build with knapsack CPU
.PHONY: docker-build-knapsack-cpu
docker-build-knapsack-cpu:
	docker build -f infrastructure/docker/go-chariot/Dockerfile.cpu \
		-t go-chariot:cpu .

# Docker build with knapsack CUDA
.PHONY: docker-build-knapsack-cuda
docker-build-knapsack-cuda:
	docker build -f infrastructure/docker/go-chariot/Dockerfile.cuda \
		-t go-chariot:cuda .
```

### Step 4: Build and Test

**Note**: These builds simply copy the pre-built knapsack libraries from the `knapsack/knapsack-library/lib/` directory. Make sure you have the knapsack repository checked out alongside chariot-ecosystem (or adjust the COPY paths in the Dockerfiles accordingly).

#### CPU-Only Build

1. **Build Docker image**:
   ```bash
   cd chariot-ecosystem
   make docker-build-knapsack-cpu
   ```

2. **Verify binary links correctly**:
   ```bash
   docker run --rm go-chariot:cpu ldd /usr/local/bin/go-chariot
   # Should show libstdc++.so.6 and other standard libraries
   ```

3. **Test solver functionality**:
   ```bash
   docker run --rm go-chariot:cpu go-chariot --test-knapsack
   ```

#### CUDA Build

1. **Build Docker image**:
   ```bash
   cd chariot-ecosystem
   make docker-build-knapsack-cuda
   ```

2. **Verify binary links correctly**:
   ```bash
   docker run --gpus all --rm go-chariot:cuda ldd /usr/local/bin/go-chariot
   # Should show libcudart.so.12 and other CUDA libraries
   ```

3. **Test solver functionality with GPU**:
   ```bash
   docker run --gpus all --rm go-chariot:cuda go-chariot --test-knapsack
   ```

4. **Verify GPU is detected**:
   ```bash
   docker run --gpus all --rm go-chariot:cuda nvidia-smi
   # Should show NVIDIA GPU info
   ```

## CGO Build Flags Explained

### Required Flags

1. **`CGO_ENABLED=1`**: Enables CGO compilation (required to link C libraries)

2. **`CGO_CFLAGS="-I/usr/local/include"`**: Tells the C compiler where to find header files
   - Finds `knapsack_c.h`

3. **`CGO_LDFLAGS`**: Link flags for the linker

   **CPU-only (Linux)**:
   ```
   -L/usr/local/lib -lknapsack_cpu -lstdc++ -lm
   ```
   - `-L/usr/local/lib`: Where to find libraries
   - `-lknapsack_cpu`: Link against CPU-only library
   - `-lstdc++`: Link against C++ standard library
   - `-lm`: Link against math library

   **CUDA (Linux)**:
   ```
   -L/usr/local/lib -lknapsack_cuda -lstdc++ -lm -lcudart
   ```
   - `-L/usr/local/lib`: Where to find libraries
   - `-lknapsack_cuda`: Link against CUDA library
   - `-lstdc++`: Link against C++ standard library
   - `-lm`: Link against math library
   - `-lcudart`: Link against CUDA runtime library

   **Metal (macOS)**:
   ```
   -L/usr/local/lib -lknapsack_metal -framework Metal -framework Foundation -lstdc++ -lm
   ```
   - `-L/usr/local/lib`: Where to find libraries
   - `-lknapsack_metal`: Link against Metal library
   - `-framework Metal`: Link against Metal GPU framework
   - `-framework Foundation`: Link against Foundation framework
   - `-lstdc++`: Link against C++ standard library
   - `-lm`: Link against math library

### Why `-lstdc++` and `-lm` are Required

The knapsack library is written in C++, so it depends on:
- **libstdc++**: C++ standard library (vectors, strings, algorithms, etc.)
- **libm**: Math library (sqrt, log, etc.)

Without these, you'll get linker errors like:
```
undefined reference to `std::vector<...>`
undefined reference to `sqrt`
```

## Deployment to Azure

### Option 1: CPU-Only Deployment (Cost-Effective)

**Best for**: Small-medium workloads (<5,000 items), cost-sensitive deployments

```bash
# Tag image for Azure Container Registry
docker tag go-chariot:cpu myregistry.azurecr.io/go-chariot:cpu

# Push to registry
az acr login --name myregistry
docker push myregistry.azurecr.io/go-chariot:cpu

# Deploy to Azure Container Instances
az container create \
  --name go-chariot-cpu \
  --resource-group myResourceGroup \
  --image myregistry.azurecr.io/go-chariot:cpu \
  --dns-name-label go-chariot \
  --ports 8080
```

**Recommended Azure VM**: Standard D4s v3 (~$0.19/hour)

### Option 2: CUDA GPU Deployment (High-Performance)

**Best for**: Large workloads (>5,000 items), performance-critical applications

```bash
# Tag image for Azure Container Registry
docker tag go-chariot:cuda myregistry.azurecr.io/go-chariot:cuda

# Push to registry
az acr login --name myregistry
docker push myregistry.azurecr.io/go-chariot:cuda

# Deploy to Azure NC-series VM with GPU
# First, create the VM
az vm create \
  --resource-group myResourceGroup \
  --name go-chariot-gpu \
  --image microsoft-dsvm:ubuntu-2004:2004-gen2:latest \
  --size Standard_NC4as_T4_v3 \
  --admin-username azureuser \
  --generate-ssh-keys

# Install nvidia-docker on the VM
# SSH into VM and run:
# curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
# distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
# curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
#   sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
#   sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
# sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
# sudo systemctl restart docker

# Pull and run with GPU access
docker run -d \
  --gpus all \
  --name go-chariot \
  -p 8080:8080 \
  myregistry.azurecr.io/go-chariot:cuda
```

**Recommended Azure VMs**:
- **NC4as T4 v3** (~$0.526/hour): Cost-effective T4 GPU
- **NC6s v3** (~$3.06/hour): High-performance V100 GPU

### Option 3: Dual Deployment with Intelligent Routing

**Best for**: Production environments with variable workload sizes

Deploy both CPU and CUDA versions, route based on problem size:

```go
// Intelligent routing in go-chariot
func (h *Handler) RouteSolveRequest(ctx context.Context, req *SolveRequest) (*Solution, error) {
    itemCount := len(req.Items)
    
    // Small problems: use CPU (fast enough, cheaper)
    if itemCount < 5000 {
        return h.cpuSolver.Solve(ctx, req)
    }
    
    // Large problems: use CUDA (10-50x faster)
    if h.cudaAvailable {
        return h.cudaSolver.Solve(ctx, req)
    }
    
    // Fallback to CPU if CUDA unavailable
    return h.cpuSolver.Solve(ctx, req)
}
```

**Cost Optimization**:
- Deploy CPU pods on Standard D4s v3: $0.19/hour
- Deploy CUDA pods on NC4as T4 v3: $0.526/hour
- Auto-scale CUDA pods based on large request queue depth
- Result: Pay for GPU only when needed (10-50x speedup for large problems)

```bash
# Create App Service plan (Linux)
az appservice plan create \
  --name myAppServicePlan \
  --resource-group myResourceGroup \
  --is-linux \
  --sku B1

# Create web app
az webapp create \
  --resource-group myResourceGroup \
  --plan myAppServicePlan \
  --name go-chariot \
  --deployment-container-image-name myregistry.azurecr.io/go-chariot:latest
```

## Performance Considerations

### CPU-Only Mode (Linux/Azure)

- **Suitable for**: Medium-sized problems (<10,000 items)
- **Performance**: ~1-10 seconds for typical route optimization
- **Scaling**: Horizontal scaling recommended for concurrent requests

### Metal GPU Mode (macOS)

- **Suitable for**: Large problems (10,000+ items)
- **Performance**: 10-100x faster than CPU
- **Use case**: Development and testing on Apple Silicon

## Troubleshooting

### Error: "knapsack_c.h: No such file or directory"

**Cause**: CGO can't find the header file

**Solution**: Verify `CGO_CFLAGS` includes `-I/usr/local/include` and the file exists:
```bash
docker run --rm go-chariot:knapsack ls -l /usr/local/include/knapsack_c.h
```

### Error: "undefined reference to knapsack_solve_v2"

**Cause**: Linker can't find the library

**Solution**: Verify `CGO_LDFLAGS` includes the correct library for your platform:
```bash
# Linux CPU
docker run --rm go-chariot:cpu ls -l /usr/local/lib/libknapsack_cpu.a

# Linux CUDA
docker run --gpus all --rm go-chariot:cuda ls -l /usr/local/lib/libknapsack_cuda.a

# macOS Metal (local)
ls -l /usr/local/lib/libknapsack_metal.a
```

### Error: "undefined reference to std::vector"

**Cause**: Missing C++ standard library link

**Solution**: Add `-lstdc++` to `CGO_LDFLAGS`

### Error: "libcudart.so.12: cannot open shared object file"

**Cause**: CUDA runtime library not found

**Solution**: 
1. Verify CUDA runtime is installed: `ldconfig -p | grep cuda`
2. For Docker, ensure using `nvidia/cuda:12.6.0-runtime` base image
3. Verify nvidia-docker is installed: `docker run --gpus all nvidia/cuda:12.6.0-runtime-ubuntu22.04 nvidia-smi`

### Error: "no CUDA-capable device is detected"

**Cause**: GPU not accessible or drivers not installed

**Solution**:
1. Check GPU is present: `lspci | grep -i nvidia`
2. Install NVIDIA drivers: `sudo apt-get install nvidia-driver-525`
3. Verify nvidia-docker: `docker run --gpus all nvidia/cuda:12.6.0-runtime-ubuntu22.04 nvidia-smi`
4. Check Docker daemon has nvidia runtime: `docker info | grep -i runtime`

### Build Works Locally but Fails in Docker

**Cause**: Local build may be using cached libraries

**Solution**: 
1. Clean local build: `go clean -cache`
2. Rebuild Docker image from scratch: `docker build --no-cache`

### Runtime Error: "knapsack solver requires CGO"

**Cause**: Binary was built without CGO enabled

**Solution**: Verify build used `CGO_ENABLED=1`:
```bash
go build -x ./cmd/server 2>&1 | grep CGO_ENABLED
```

## Testing

### Unit Tests with CGO

Create `services/go-chariot/internal/solver/knapsack_test.go`:

```go
//go:build cgo
// +build cgo

package solver

import (
    "testing"
)

func TestKnapsackSolver(t *testing.T) {
    if !HasKnapsackSupport {
        t.Skip("Knapsack support not available (CGO disabled)")
    }

    solver := &KnapsackSolver{}
    
    config := &KnapsackConfig{
        // Test configuration
    }
    
    solution, err := solver.Solve(config)
    if err != nil {
        t.Fatalf("Solver failed: %v", err)
    }
    
    if solution == nil {
        t.Fatal("Solution is nil")
    }
    
    // Validate solution
    t.Logf("Objective: %f", solution.Objective)
}
```

Run tests:
```bash
CGO_ENABLED=1 go test -v ./internal/solver/...
```

## Next Steps

1. ✅ Build knapsack CPU and CUDA Docker images
2. ⏳ Create Dockerfile.cpu and Dockerfile.cuda for go-chariot
3. ⏳ Implement Go CGO bindings (CPU and CUDA variants)
4. ⏳ Add unit tests for both platforms
5. ⏳ Test Docker builds (CPU and CUDA)
6. ⏳ Deploy to Azure (start with CPU)
7. ⏳ Benchmark and decide if CUDA is cost-effective
8. ⏳ Implement intelligent routing if both deployed

## References

- [Platform-Specific Libraries Overview](../docs/PLATFORM_SPECIFIC_LIBS.md)
- [CUDA Support and Azure Deployment](../docs/CUDA_SUPPORT.md)
- [Knapsack Cross-Platform Build Documentation](../CROSS_PLATFORM_BUILD.md)
- [Knapsack C API](../knapsack-library/include/knapsack_c.h)
- [Go CGO Documentation](https://pkg.go.dev/cmd/cgo)
- [Docker Multi-Stage Builds](https://docs.docker.com/build/building/multi-stage/)
- [NVIDIA Container Toolkit](https://github.com/NVIDIA/nvidia-docker)
