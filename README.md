# Knapsack

A route-planning knapsack solver inspired by classical QAOA, now with dual backends:

- NVIDIA CUDA (Jetson and other CUDA-capable systems)
- Apple Metal (Apple Silicon, in-process via Objective‑C++/cgo)

On macOS Apple Silicon, CUDA is not required; the solver uses a Metal-based evaluator by default with a CPU fallback.

## Overview

This project tackles generalized knapsack-style selection for van trips and worker pickup, balancing distance and capacity. The classical algorithm (inspired by QAOA scoring) evaluates candidate selections and chooses the best plan per trip.

Backends:
- CUDA: original GPU kernels for population generation/evaluation on Jetson/NVIDIA.
- Metal: runtime-compiled shader that scores candidates on Apple GPUs without requiring the external `metal` CLI.
- CPU fallback: deterministic heuristic used when a GPU backend isn’t available.

Key outputs: a `van_routes.csv` describing trips, crew sizes, distances, and fuel cost.

## Features

- Dual GPU backends: CUDA (NVIDIA) and Metal (Apple Silicon)
- Runtime shader compilation on macOS (no external `metal` tool dependency)
- Capacity-aware candidate generation with soft penalties
- Recursive/greedy hybrid classical solver for trip selection
- Distance-based optimization using the haversine formula
- Configurable mutation/population parameters (for CUDA path)

## Requirements

Common:
- C++17 compatible compiler
- CMake 3.18+

CUDA (Jetson/other NVIDIA machines):
- CUDA Toolkit 11.0+
- NVIDIA GPU (set `CMAKE_CUDA_ARCHITECTURES` accordingly; default in this repo is 87 for Jetson Orin)

Apple Metal (Apple Silicon):
- macOS with Apple Clang toolchain (Xcode Command Line Tools)
- Apple Silicon GPU (Metal framework available by default)

## Build and Run

### macOS (Apple Silicon, Metal backend)

CUDA is not required. The build links the Objective‑C++ Metal bridge and compiles the Metal shader at runtime.

```bash
# From repo root
mkdir -p build && cd build
cmake ..
cmake --build . -j

# Run the solver (optional arg = target team size)
./knapsack_solver 50

# Output CSV will be written in the build directory
ls van_routes.csv
```

Notes:
- The default input CSV is `data/villages.csv` relative to the repo root.
- The Metal shader (`kernels/metal/shaders/eval_block_candidates.metal`) is read at runtime and compiled in-process.
- If Metal initialization fails, the solver falls back to a CPU heuristic.

### Jetson / NVIDIA (CUDA backend)

```bash
# From repo root
mkdir -p build && cd build
cmake ..
cmake --build . -j
./knapsack_solver 50
```

Tips:
- The top-level `CMakeLists.txt` detects Apple vs non-Apple. On non-Apple hosts it enables CUDA and builds `.cu` kernels.
- If your GPU architecture differs from Jetson Orin (8.7), set `-DCMAKE_CUDA_ARCHITECTURES=<arch>` when configuring CMake.

## V2 solver (JSON configs)

V2 is a generic, block-aware solver with multi-term objectives and multiple soft capacity constraints. A small CLI is installed as `knapsack_v2_cli`.

Quick start:

```bash
# From repo root
mkdir -p build-lib && cd build-lib
cmake -DCMAKE_INSTALL_PREFIX=/usr/local ..
cmake --build . -j
# Optional: install system-wide (requires sudo if /usr/local)
sudo cmake --install .

# Run against an example config
./knapsack_v2_cli ../docs/v2/example_select.json

# With options (beam and debug)
cat > /tmp/opts.json <<'JSON'
{ "beam_width": 32, "iters": 5, "seed": 42, "debug": true }
JSON
./knapsack_v2_cli ../docs/v2/example_select.json /tmp/opts.json
```

Notes:
- Debug mode prints per-iteration best totals with constraint slacks and per-constraint penalty parts.
- The C API is available via `knapsack-library/include/knapsack_c.h` with `solve_knapsack_v2_from_json`.
- On macOS, the library uses Metal at runtime and falls back to CPU if Metal is unavailable.

## Go Metal Binding (darwin/arm64)

A Go package provides in-process access to the Metal evaluator for Apple Silicon.

Location: `bindings/go/metal` (build-tagged `darwin && arm64`)

Quick start:

```bash
cd bindings/go/metal
go generate     # builds static lib and copies shader for embedding
go test -v      # runs a smoke test using the Metal evaluator
```

API sketch:

- Input (EvalIn):
  - Candidates: pointer to packed 2‑bit lanes (4 items per byte; 0=off, 1=van0)
  - NumItems, NumCandidates
  - ItemValues, ItemWeights, VanCaps (float32 buffers)
  - NumVans (currently 1 used here), PenaltyCoeff
- Output (EvalOut): Obj and SoftPenalty arrays (float32)

Editor note: If VS Code reports cgo Problems here, ensure CGO is enabled and GOOS/GOARCH are darwin/arm64. Tests will still pass. This repo includes `.vscode/settings.json` to help.

## Chariot integration

If you’re integrating this solver into another application (e.g., Chariot), use the C API shipped in `knapsack-library`.

- Header: `knapsack-library/include/knapsack_c.h`
- Library target: `knapsack` (static library built via CMake)
- Behavior:
  - On macOS Apple Silicon, the library initializes the Metal evaluator at runtime and falls back to CPU if Metal is unavailable.
  - On other platforms, it uses the existing CPU path; CUDA remains available in the top-level app for Jetson builds.

### CMake (consumer app)

Option A: add as a subdirectory and link the library:

```cmake
# In your app's CMakeLists.txt
add_subdirectory(${CMAKE_SOURCE_DIR}/knapsack/knapsack-library ${CMAKE_BINARY_DIR}/knapsack-library)

add_executable(chariot_app ...)
# Public headers are included by the target; link against the library
target_link_libraries(chariot_app PRIVATE knapsack)
```

Option B: build the library separately and link by path (not recommended unless you manage install/deploy):

```bash
cd knapsack/knapsack-library
mkdir -p build && cd build
cmake ..
cmake --build . -j
# Produces libknapsack.a in this build tree
```

Option C: install system-wide (so cgo can use -I/usr/local/include and -L/usr/local/lib):

```bash
cd knapsack/knapsack-library
mkdir -p build && cd build
cmake -DCMAKE_INSTALL_PREFIX=/usr/local ..
cmake --build . -j
sudo cmake --install .

# Verifications
ls /usr/local/include/knapsack_c.h
ls /usr/local/lib/libknapsack.a
```

### Minimal C usage

```c
#include "knapsack_c.h"
#include <stdio.h>

int main() {
    const char* csv = "../data/villages.csv"; // adjust path for your runtime
    int target = 50;
    KnapsackSolution* sol = solve_knapsack(csv, target);
    if (!sol) {
        fprintf(stderr, "knapsack solve failed\n");
        return 1;
    }
    printf("Trips: %d, Crew: %d, Shortfall: %d, Fuel: %.2f\n",
           sol->num_trips, sol->total_crew, sol->shortfall, sol->total_fuel_cost);
    // ... iterate sol->trips ...
    free_knapsack_solution(sol);
    return 0;
}
```

### Go (cgo) usage for Chariot

If your Chariot service is in Go, here are ready-to-use cgo bindings that match the current C API.

Darwin/arm64 (Apple Silicon, Metal runtime):

```go
//go:build darwin && arm64

package chariot

/*
#cgo CFLAGS: -I${SRCDIR}/../third_party/knapsack/knapsack-library/include
#cgo darwin LDFLAGS: -L${SRCDIR}/../third_party/knapsack/build/knapsack-library -lknapsack -framework Metal -framework Foundation -lc++
#include "knapsack_c.h"
#include <stdlib.h>
*/
import "C"

import (
    "errors"
    "unsafe"
)

type VanTrip struct {
    VanID       int
    VillageCSV  string
    DistanceKM  float64
    FuelUSD     float64
    CrewSize    int
}

type Solution struct {
    Trips       []VanTrip
    TotalCrew   int
    Shortfall   int
    TotalFuel   float64
}

func SolveKnapsack(csvPath string, target int) (*Solution, error) {
    cpath := C.CString(csvPath)
    defer C.free(unsafe.Pointer(cpath))

    sol := C.solve_knapsack(cpath, C.int(target))
    if sol == nil {
        return nil, errors.New("knapsack solve failed")
    }
    defer C.free_knapsack_solution(sol)

    n := int(sol.num_trips)
    trips := make([]VanTrip, 0, n)
    if n > 0 && sol.trips != nil {
        slice := (*[1 << 30]C.VanTripResult)(unsafe.Pointer(sol.trips))[:n:n]
        for i := 0; i < n; i++ {
            t := slice[i]
            trips = append(trips, VanTrip{
                VanID:      int(t.van_id),
                VillageCSV: C.GoString(t.village_names),
                DistanceKM: float64(t.distance),
                FuelUSD:    float64(t.fuel_cost),
                CrewSize:   int(t.crew_size),
            })
        }
    }

    return &Solution{
        Trips:     trips,
        TotalCrew: int(sol.total_crew),
        Shortfall: int(sol.shortfall),
        TotalFuel: float64(sol.total_fuel_cost),
    }, nil
}
```

Linux/arm64 (Jetson, optional CUDA link):

```go
//go:build linux && arm64

package chariot

/*
#cgo CFLAGS: -I/usr/local/include
#cgo LDFLAGS: -L/usr/local/lib -lknapsack -lstdc++ -lm
// If your app separately depends on CUDA runtime, you may also need:
// #cgo LDFLAGS: -L/usr/local/cuda/lib64 -lcudart -lcuda -lcurand
#include "knapsack_c.h"
#include <stdlib.h>
*/
import "C"

import (
    "errors"
    "unsafe"
)

type VanTrip struct {
    VanID       int
    VillageCSV  string
    DistanceKM  float64
    FuelUSD     float64
    CrewSize    int
}

type Solution struct {
    Trips       []VanTrip
    TotalCrew   int
    Shortfall   int
    TotalFuel   float64
}

func SolveKnapsack(csvPath string, target int) (*Solution, error) {
    cpath := C.CString(csvPath)
    defer C.free(unsafe.Pointer(cpath))

    sol := C.solve_knapsack(cpath, C.int(target))
    if sol == nil {
        return nil, errors.New("knapsack solve failed")
    }
    defer C.free_knapsack_solution(sol)

    n := int(sol.num_trips)
    trips := make([]VanTrip, 0, n)
    if n > 0 && sol.trips != nil {
        slice := (*[1 << 30]C.VanTripResult)(unsafe.Pointer(sol.trips))[:n:n]
        for i := 0; i < n; i++ {
            t := slice[i]
            trips = append(trips, VanTrip{
                VanID:      int(t.van_id),
                VillageCSV: C.GoString(t.village_names),
                DistanceKM: float64(t.distance),
                FuelUSD:    float64(t.fuel_cost),
                CrewSize:   int(t.crew_size),
            })
        }
    }

    return &Solution{
        Trips:     trips,
        TotalCrew: int(sol.total_crew),
        Shortfall: int(sol.shortfall),
        TotalFuel: float64(sol.total_fuel_cost),
    }, nil
}
```

Notes:
- Adjust `-I` and `-L` paths to where you install or vendor the knapsack headers and library (e.g., `/usr/local/include` and `/usr/local/lib`).
- On Jetson, the knapsack library itself does not require CUDA; only link CUDA libs if your application also uses them.
- Provide non-matching build-tag stubs as needed (e.g., return a clear error on unsupported platforms).
- Ensure CGO is enabled and your toolchain targets the correct OS/ARCH in your editor/CI.

## Data

Sample CSVs live under `data/` (e.g., `villages.csv`, `villages_300.csv`). The executable reads `../data/villages.csv` by default when run from `build/`.

## Project layout

- `src/` main C++ sources (RoutePlanner, RecursiveSolver, etc.)
- `kernels/metal/` Metal API (Objective‑C++ bridge and shader)
- `kernels/` (CUDA kernels for non-Apple builds)
- `knapsack-library/` C API wrapper (used by other integrations)
- `bindings/go/metal/` Go cgo package for Metal evaluator

## FAQ

- Q: Do I need the external `metal` CLI on macOS?
  - A: No. The shader is compiled at runtime via Metal APIs.

- Q: Can I still build the CUDA version?
  - A: Yes. On non-Apple hosts, the build enables CUDA automatically.

- Q: Where is the output written?
  - A: `van_routes.csv` in the current working directory (e.g., the `build/` dir when executed there).