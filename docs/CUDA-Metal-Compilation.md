## CUDA and Metal Compilation (actual project files)

This guide shows how to build and run the knapsack solver using the real source tree:

- macOS Apple Silicon: Metal backend (no external `metal` CLI needed; shader compiles at runtime)
- Jetson/NVIDIA: CUDA backend
- C library install for external consumers (e.g., Chariot)
- Go binding (darwin/arm64)

### Prerequisites

Common:
- CMake 3.18+
- C++17 toolchain

macOS (Apple Silicon):
- Xcode Command Line Tools (Apple Clang, Metal and Foundation frameworks)

Jetson/NVIDIA:
- CUDA Toolkit 11.x+
- Set `CMAKE_CUDA_ARCHITECTURES` if needed (defaults to 87 for Orin in other parts of this repo)

### Build on macOS (Apple Silicon, Metal backend)

The project links an Objective‑C++ Metal bridge and compiles the Metal shader at runtime via the system framework (no external `metal` tool required).

```bash
# From repo root
mkdir -p build && cd build
cmake ..
cmake --build . -j

# Run (optional arg = target team size)
./knapsack_solver 50

# Output CSV will be written next to the binary
ls van_routes.csv
```

Notes:
- Default input CSV is `data/villages.csv` (relative to repo root when run from `build/`).
- If Metal initialization fails, the solver falls back to a CPU heuristic.

### Build on Jetson/NVIDIA (CUDA backend)

```bash
# From repo root
mkdir -p build && cd build
cmake ..
cmake --build . -j
./knapsack_solver 50
```

Tips:
- On non-Apple hosts, the top-level CMake enables CUDA and builds `.cu` kernels.
- If your GPU architecture differs from Jetson Orin, pass `-DCMAKE_CUDA_ARCHITECTURES=<arch>` to `cmake ..`.

### Install the C library (for external apps)

The C API lives under `knapsack-library/` and exposes `solve_knapsack` and `free_knapsack_solution`.

```bash
cd knapsack/knapsack-library
mkdir -p build && cd build
cmake -DCMAKE_INSTALL_PREFIX=/usr/local ..
cmake --build . -j
sudo cmake --install .

# Verify
ls /usr/local/include/knapsack_c.h
ls /usr/local/lib/libknapsack.a
```

External apps (C/C++ or Go via cgo) can then use `-I/usr/local/include` and `-L/usr/local/lib -lknapsack`.

### Go Metal binding (darwin/arm64)

The Go binding evaluates candidates on Apple GPUs with in-process shader compilation.

```bash
cd bindings/go/metal
go generate     # builds static lib and copies shader for embedding
go test -v
```

If your editor reports cgo problems, ensure CGO is enabled and GOOS/GOARCH are `darwin/arm64`. This repo includes `.vscode/settings.json` to help.

### Troubleshooting

- macOS: “Metal init failed”
	- Ensure Xcode Command Line Tools are installed; the shader is read from `kernels/metal/shaders/eval_block_candidates.metal` at runtime.
- macOS: linker errors for C++/frameworks
	- Make sure you link `-framework Metal -framework Foundation` and `-lc++` where relevant (already wired in CMake/Go binding).
- Jetson: incorrect or missing CUDA arch
	- Configure with `-DCMAKE_CUDA_ARCHITECTURES=<arch>`.
- Go binding: editor shows Problems but tests pass
	- Ensure CGO is enabled and environment matches `darwin/arm64`.
