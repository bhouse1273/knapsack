# Pre-Built Knapsack Libraries

This directory contains pre-built, platform-specific knapsack solver libraries for immediate use in the go-chariot integration. These libraries are **committed to the repository** so users don't need to build them.

## Directory Structure

```
lib/
├── linux-cpu/              # Linux CPU-only (274KB + 258KB RL)
│   ├── libknapsack_cpu.a
│   ├── knapsack_cpu.h
│   ├── librl_support.a      # RL Support library (NEW!)
│   └── rl_api.h             # RL API header (NEW!)
├── linux-cuda/             # Linux with NVIDIA CUDA (631KB + 258KB RL)
│   ├── libknapsack_cuda.a
│   ├── knapsack_cuda.h
│   ├── librl_support.a      # RL Support library (NEW!)
│   └── rl_api.h             # RL API header (NEW!)
├── macos-metal/            # macOS with Metal GPU (216KB + 258KB RL)
│   ├── libknapsack_metal.a
│   ├── knapsack_macos_metal.h
│   ├── librl_support.a      # RL Support library (NEW!)
│   ├── librl_support.dylib  # Shared library for Go/Python (NEW!)
│   └── rl_api.h             # RL API header (NEW!)
└── macos-cpu/              # macOS CPU-only (Apple Silicon)
    ├── libknapsack_macos_cpu.a
    ├── knapsack_macos_cpu.h
    ├── librl_support.a      # RL Support library (NEW!)
    ├── librl_support.dylib  # Shared library for Go/Python (NEW!)
    └── rl_api.h             # RL API header (NEW!)
```

## Platform Details

### RL Support Library (All Platforms)

Each platform directory now includes the RL Support library for Next Best Action (NBA) scoring:

- **Static Library**: `librl_support.a` (258KB)
- **Shared Library**: `librl_support.dylib` (macOS) or `librl_support.so` (Linux) - 203KB
- **Header**: `rl_api.h`
- **Build**: Compiled with C++17, includes LinUCB contextual bandit
- **ONNX Support**: The libraries are built WITHOUT ONNX Runtime by default (LinUCB only)
  - To enable ONNX: Rebuild with `-DBUILD_ONNX=ON` and link against ONNX Runtime
  - ONNX allows loading trained ML models for production inference
  - Graceful fallback to LinUCB if ONNX loading fails

**RL Features:**
- LinUCB contextual bandit with alpha exploration parameter
- Feature extraction for select-mode and assign-mode slates
- Online learning with structured feedback (rewards, chosen+decay, events)
- Batch inference (<1ms for NBA decisions)
- Analytics APIs (feature inspection, config retrieval)
- Language bindings ready: Go (cgo), Python (ctypes)

**API Functions:**
- `rl_init_from_json()` - Initialize RL context from JSON config
- `rl_score_batch()` - Score candidates (auto feature extraction)
- `rl_score_batch_with_features()` - Score with pre-computed features
- `rl_learn_batch()` - Update model from feedback
- `rl_prepare_features()` - Extract features from candidates
- `rl_get_feat_dim()`, `rl_get_config_json()`, `rl_get_last_features()` - Introspection
- `rl_close()` - Cleanup

**See**: `../../docs/RL_SUPPORT.md` for complete API reference and usage examples.

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

### macOS CPU-only (`macos-cpu/`)
- **Library**: `libknapsack_macos_cpu.a`
- **Header**: `knapsack_cpu.h`
- **Compiler**: Clang (Apple Silicon)
- **Architecture**: arm64 (M1/M2/M3)
- **GPU Support**: None
- **Dependencies**: libstdc++
- **Use Case**: Fastest option on Apple Silicon for tested datasets
- **Build Flags**: `-DBUILD_CPU_ONLY=ON`, `-DUSE_METAL=OFF`

## Verification

Verify the libraries are correctly built:

```bash
# From the knapsack root directory
make verify-libs
```

This checks:
- All four libraries exist (macOS Metal and macOS CPU are optional on non-macOS)
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
COPY knapsack/knapsack-library/lib/linux-cpu/librl_support.a /usr/local/lib/
COPY knapsack/knapsack-library/lib/linux-cpu/rl_api.h /usr/local/include/

# CUDA
COPY knapsack/knapsack-library/lib/linux-cuda/libknapsack_cuda.a /usr/local/lib/
COPY knapsack/knapsack-library/lib/linux-cuda/knapsack_cuda.h /usr/local/include/
COPY knapsack/knapsack-library/lib/linux-cuda/librl_support.a /usr/local/lib/
COPY knapsack/knapsack-library/lib/linux-cuda/rl_api.h /usr/local/include/

# Metal (macOS)
COPY knapsack/knapsack-library/lib/macos-metal/libknapsack_metal.a /usr/local/lib/
COPY knapsack/knapsack-library/lib/macos-metal/knapsack_macos_metal.h /usr/local/include/
COPY knapsack/knapsack-library/lib/macos-metal/librl_support.a /usr/local/lib/
COPY knapsack/knapsack-library/lib/macos-metal/librl_support.dylib /usr/local/lib/
COPY knapsack/knapsack-library/lib/macos-metal/rl_api.h /usr/local/include/

# macOS CPU-only
COPY knapsack/knapsack-library/lib/macos-cpu/libknapsack_macos_cpu.a /usr/local/lib/
COPY knapsack/knapsack-library/lib/macos-cpu/knapsack_macos_cpu.h /usr/local/include/
COPY knapsack/knapsack-library/lib/macos-cpu/librl_support.a /usr/local/lib/
COPY knapsack/knapsack-library/lib/macos-cpu/librl_support.dylib /usr/local/lib/
COPY knapsack/knapsack-library/lib/macos-cpu/rl_api.h /usr/local/include/
```

See [GO_CHARIOT_INTEGRATION.md](../../docs/GO_CHARIOT_INTEGRATION.md) for complete integration instructions.

## Rebuilding Libraries

If you modify the C++ source code, you can rebuild all platform libraries:

```bash
# From the knapsack root directory
make build-all-platforms
```

This script:
1. Builds Linux CPU library via Docker
2. Builds Linux CUDA library via Docker (requires nvidia-docker for verification)
3. Builds macOS Metal library natively (on macOS only)
4. Builds macOS CPU-only library natively (on macOS only)
5. Extracts all libraries to this directory
6. Verifies each library's symbols

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
- `BUILD_CPU_ONLY=ON` → `libknapsack_cpu.a` (Linux) or `libknapsack_macos_cpu.a` (macOS)
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
