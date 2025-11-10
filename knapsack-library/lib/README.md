# Pre-Built Knapsack Libraries

This directory contains pre-built, platform-specific knapsack solver libraries for immediate use in the go-chariot integration. These libraries are **committed to the repository** so users don't need to build them.

## Directory Structure

```
lib/
├── linux-cpu/              # Linux CPU-only (274KB)
│   ├── libknapsack_cpu.a
│   └── knapsack_cpu.h
├── linux-cuda/             # Linux with NVIDIA CUDA (631KB)
│   ├── libknapsack_cuda.a
│   └── knapsack_cuda.h
└── macos-metal/            # macOS with Metal GPU (216KB)
    ├── libknapsack_metal.a
    └── knapsack_metal.h
```

## Platform Details

### Linux CPU (`linux-cpu/`)
- **Library**: `libknapsack_cpu.a` (274KB)
- **Header**: `knapsack_cpu.h`
- **Compiler**: GCC 11+ on Ubuntu 22.04
- **Architecture**: x86_64
- **GPU Support**: None
- **Dependencies**: libstdc++, libm
- **Use Case**: Cost-effective deployments on standard VMs
- **Build Flags**: `-DBUILD_CPU_ONLY=ON`

### Linux CUDA (`linux-cuda/`)
- **Library**: `libknapsack_cuda.a` (631KB)
- **Header**: `knapsack_cuda.h`
- **Compiler**: NVCC (CUDA 12.6.0) on Ubuntu 22.04
- **Architecture**: x86_64
- **GPU Support**: NVIDIA CUDA
- **CUDA Architectures**: SM 7.0, 7.5, 8.0, 8.6, 8.9, 9.0
- **Dependencies**: libstdc++, libm, libcudart
- **Use Case**: High-performance GPU deployments on NVIDIA hardware
- **Build Flags**: `-DBUILD_CUDA=ON`
- **Runtime Requirement**: CUDA Runtime 12.0+

### macOS Metal (`macos-metal/`)
- **Library**: `libknapsack_metal.a` (216KB)
- **Header**: `knapsack_metal.h`
- **Compiler**: Clang (Apple Silicon)
- **Architecture**: arm64 (M1/M2/M3)
- **GPU Support**: Apple Metal
- **Dependencies**: libstdc++, Metal framework
- **Use Case**: Development and testing on Apple Silicon
- **Build Flags**: `-DUSE_METAL=ON`

## Verification

Verify the libraries are correctly built:

```bash
# From the knapsack root directory
make verify-libs
```

This checks:
- All three libraries exist
- File sizes are reasonable
- CPU library has no GPU symbols
- CUDA library contains CUDA symbols
- Metal library contains Metal symbols

## Usage in go-chariot

These libraries are designed for direct integration into go-chariot via Docker COPY commands:

```dockerfile
# CPU-only
COPY knapsack/knapsack-library/lib/linux-cpu/libknapsack_cpu.a /usr/local/lib/
COPY knapsack/knapsack-library/lib/linux-cpu/knapsack_cpu.h /usr/local/include/

# CUDA
COPY knapsack/knapsack-library/lib/linux-cuda/libknapsack_cuda.a /usr/local/lib/
COPY knapsack/knapsack-library/lib/linux-cuda/knapsack_cuda.h /usr/local/include/

# Metal (macOS)
COPY knapsack/knapsack-library/lib/macos-metal/libknapsack_metal.a /usr/local/lib/
COPY knapsack/knapsack-library/lib/macos-metal/knapsack_metal.h /usr/local/include/
```

See [GO_CHARIOT_INTEGRATION.md](../../docs/GO_CHARIOT_INTEGRATION.md) for complete integration instructions.

## Rebuilding Libraries

If you modify the C++ source code, you can rebuild all three libraries:

```bash
# From the knapsack root directory
make build-all-platforms
```

This script:
1. Builds Linux CPU library via Docker
2. Builds Linux CUDA library via Docker (requires nvidia-docker for verification)
3. Builds macOS Metal library natively (on macOS only)
4. Extracts all libraries to this directory
5. Verifies each library's symbols

**Note**: The script runs on M1 Mac via Docker emulation for Linux builds. CUDA builds work without requiring CUDA hardware on the build machine.

## Build System Details

### Automated Build Script
- **Location**: `scripts/build-all-platforms.sh`
- **Docker Images**:
  - CPU: `ubuntu:22.04` → GCC 11
  - CUDA: `nvidia/cuda:12.6.0-devel-ubuntu22.04` → NVCC
- **Native Build**: CMake with Metal flags on macOS
- **Extraction**: Uses `docker create` + `docker cp` to extract artifacts
- **Verification**: Checks symbols with `nm` command

### CMake Configuration
The root `CMakeLists.txt` automatically sets the output library name based on build flags:
- `BUILD_CPU_ONLY=ON` → `libknapsack_cpu.a`
- `BUILD_CUDA=ON` → `libknapsack_cuda.a`
- `USE_METAL=ON` → `libknapsack_metal.a`

## Version Information

These libraries were built from the knapsack v2 solver codebase:
- **Solver Version**: v2
- **Build Date**: January 2025
- **Compiler Versions**:
  - GCC: 11.4.0 (Ubuntu 22.04)
  - NVCC: 12.6.0 (CUDA Toolkit)
  - Clang: Apple LLVM (Xcode)

## License

Same as the knapsack project (see [LICENSE](../../LICENSE) in the root directory).

## Support

For issues or questions about these libraries:
1. Check [GO_CHARIOT_INTEGRATION.md](../../docs/GO_CHARIOT_INTEGRATION.md) for integration help
2. Verify library integrity with `make verify-libs`
3. Rebuild if necessary with `make build-all-platforms`
4. See troubleshooting section in GO_CHARIOT_INTEGRATION.md
