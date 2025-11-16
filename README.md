# Knapsack

A generalized, block‑aware knapsack solver with GPU acceleration, reinforcement learning support, and a modern V2 pipeline.

Backends:
- NVIDIA CUDA (Jetson and other CUDA‑capable systems)
- Apple Metal (Apple Silicon, in‑process via Objective‑C++/cgo)
- CPU fallback on all platforms

RL Support:
- LinUCB contextual bandit for NBA (Next-Best Action) scoring
- ONNX Runtime integration for trained ML models
- Online learning with structured feedback
- Sub-millisecond batch inference

On macOS Apple Silicon, CUDA is not required; the library uses a Metal‑based evaluator by default and automatically falls back to CPU if Metal is unavailable.

## Overview

This project tackles generalized knapsack‑style selection for many domains (e.g., logistics, packing, portfolio selection). The V2 pipeline adds a JSON‑driven, multi‑constraint solver with GPU evaluators (CUDA/Metal), a C API, Go bindings, and a small CLI.

The legacy CSV demo is still available and produces a tabular solution CSV; column names and meaning are example‑specific to the demo, not the core solver.

## Features

- V2 general solver: multi‑term objective and multiple soft capacity constraints
- Dominance filtering preprocessor (drop dominated items; optional surrogate capacity)
- Beam search (select mode) with per‑iteration debug metrics (constraint slacks, penalty parts)
- Dual GPU backends: CUDA (NVIDIA) and Metal (Apple Silicon) with automatic CPU fallback
- Runtime shader compilation on macOS (no external `metal` CLI dependency)
- Classical (CSV) solver retained for historical scenarios
- **RL Support Library**: Production-ready reinforcement learning for NBA scoring
  - LinUCB contextual bandit with exploration/exploitation
  - ONNX Runtime integration for trained ML models (XGBoost, TensorFlow, PyTorch, etc.)
  - Feature extraction for select and assign modes
  - Online learning with structured feedback (rewards, chosen+decay, events)
  - Batch inference <1ms for thousands of candidates
  - Graceful fallback to bandit if ONNX model unavailable
  - Go (cgo) and Python (ctypes) bindings

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

Optional (for RL ONNX support):
- ONNX Runtime 1.22+ (`brew install onnxruntime` on macOS, or `apt-get install libonnxruntime-dev` on Linux)
- Python packages for model generation: `pip install onnx numpy`

## Build and Run

### macOS (Apple Silicon, Metal backend)

CUDA is not required. The build links the Objective‑C++ Metal bridge and compiles the Metal shader at runtime.

```bash
# From repo root
mkdir -p build && cd build
cmake ..
cmake --build . -j

# Run the solver (optional args: target team size, output filename)
./knapsack_solver 50
# or specify a custom output file:
./knapsack_solver 50 my_results.csv

# Output CSV will be written in the build directory
ls routes.csv
```

Notes:
- The default input CSV is `data/villages_50.csv` relative to the repo root (classical path).
- The Metal shader (`kernels/metal/shaders/eval_block_candidates.metal`) is read at runtime and compiled in‑process for the V2 evaluator.
- If Metal initialization fails, the library falls back to a CPU evaluator.

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
- Debug mode prints per‑iteration best totals with constraint slacks and per‑constraint penalty parts.
- The C API is available via `knapsack-library/include/knapsack_c.h` exposing `solve_knapsack_v2_from_json` and `free_knapsack_solution_v2`.
- On macOS, the library uses Metal at runtime and falls back to CPU if Metal is unavailable.

V2 at a glance:
- Problem modes: select (current), assign (WIP wiring)
- Objective: weighted sum of terms (e.g., value, profit)
- Constraints: one or more soft capacity constraints with penalties
- Search: beam search with configurable width/iterations/seed
- Preprocess: dominance filters (exact single‑constraint or surrogate) with mapping back to original item indices

Example config (select, tiny):

```json
{
    "mode": "select",
    "items": [
        { "id": 0, "obj": [10.0], "w": [3.0] },
        { "id": 1, "obj": [9.0],  "w": [4.0] },
        { "id": 2, "obj": [5.0],  "w": [2.0] }
    ],
    "constraints": [ { "capacity": 5.0 } ]
}
```

Example options JSON:

```json
{ "beam_width": 32, "iters": 5, "seed": 42, "debug": true,
    "dom_enable": true, "dom_eps": 1e-9, "dom_surrogate": true }
```

### Scout Mode (Hybrid with Exact Solvers)

Scout mode enables beam search to act as a **data scout** for exact solvers (Gurobi, SCIP, CPLEX). It identifies active items, provides warm starts, and reduces problem size before handing off to an exact solver.

**C++ API:**
```cpp
v2::SolverOptions opt;
opt.scout_mode = true;
opt.scout_threshold = 0.5;  // items must appear in 50% of top solutions
opt.scout_top_k = 8;

v2::ScoutResult result;
if (v2::SolveBeamScout(cfg, soa, opt, &result, &err)) {
    // Use result.active_items for exact solver
    // Use result.best_select as warm start
}
```

**Demo:**
```bash
cd build
make v2_scout_demo
./v2_scout_demo
```

See [`docs/BeamAsDataScout.md`](docs/BeamAsDataScout.md) for detailed documentation and integration patterns.

Solution fields (C API): `num_items`, `select[]`, `objective`, `penalty`, `total`.

## RL Support (NBA Scoring)

The RL support library enables **Next-Best Action (NBA) scoring** using reinforcement learning and trained ML models.

### Features

- **LinUCB Contextual Bandit**: Exploration/exploitation with configurable alpha parameter
- **ONNX Runtime Integration**: Load trained models (XGBoost, neural nets, etc.) for production inference
- **Feature Extraction**: Automated slate feature generation for select and assign modes
- **Online Learning**: Update models with structured feedback (rewards, chosen+decay, events)
- **Batch Inference**: Score thousands of candidates in <1ms
- **Graceful Fallback**: Auto-fallback to LinUCB bandit if ONNX model unavailable
- **Language Bindings**: Go (cgo) and Python (ctypes) ready

### Quick Start

**Build with ONNX Support:**
```bash
# macOS
brew install onnxruntime
cmake -B build -DBUILD_ONNX=ON
cmake --build build

# Linux
apt-get install libonnxruntime-dev
cmake -B build -DBUILD_ONNX=ON
cmake --build build
```

**C++ API:**
```cpp
#include "rl/rl_api.h"

// Initialize with ONNX model (optional)
const char* cfg = R"({
    "feat_dim": 12,
    "alpha": 0.3,
    "model_path": "models/nba_scorer.onnx",
    "model_input": "input",
    "model_output": "output"
})";

rl_handle_t rl = rl_init_from_json(cfg, err, sizeof(err));

// Prepare features from candidates
float features[num_candidates * feat_dim];
rl_prepare_features(rl, candidates, num_items, num_candidates, 
                     0 /*select mode*/, features, err, sizeof(err));

// Score batch with ONNX model (or LinUCB fallback)
double scores[num_candidates];
rl_score_batch_with_features(rl, features, feat_dim, num_candidates, 
                              scores, err, sizeof(err));

// Learn from feedback
const char* feedback = R"({"rewards": [1.0, 0.0, 0.5]})";
rl_learn_batch(rl, feedback, err, sizeof(err));

rl_close(rl);
```

**Python API:**
```python
from rl_support import RL

# Initialize RL scorer
scorer = RL({
    "feat_dim": 12,
    "alpha": 0.3,
    "model_path": "models/nba_scorer.onnx"
})

# Score candidates
scores = scorer.score_with_features(features, feat_dim, num_candidates)

# Learn from feedback
scorer.learn({"rewards": [1.0, 0.0, 0.5]})
scorer.close()
```

**Generate Test ONNX Model:**
```bash
pip install onnx numpy
python3 tools/gen_onnx_model.py 12 models/my_model.onnx
```

### Documentation

- **RL Support Guide**: [`docs/RL_SUPPORT.md`](docs/RL_SUPPORT.md) - Complete API reference
- **ONNX Integration**: [`ONNX_INTEGRATION_COMPLETE.md`](ONNX_INTEGRATION_COMPLETE.md) - Implementation details
- **Model Generation**: [`docs/ONNX_MODEL_GEN.md`](docs/ONNX_MODEL_GEN.md) - Creating test models
- **NBA Example**: [`docs/BeamNextBestAction.md`](docs/BeamNextBestAction.md) - Usage patterns

### Go Integration

```go
import "github.com/bhouse1273/knapsack/bindings/go/rl"

// Initialize RL scorer
config := `{"feat_dim": 12, "alpha": 0.3, "model_path": "models/nba_scorer.onnx"}`
scorer, err := rl.InitFromJSON(config)
if err != nil {
    log.Fatal(err)
}
defer scorer.Close()

// Score candidates
scores, err := scorer.ScoreWithFeatures(features, featDim, numCandidates)
if err != nil {
    log.Fatal(err)
}

// Learn from feedback
feedback := `{"rewards": [1.0, 0.0, 0.5]}`
err = scorer.Learn(feedback)
```

See [`bindings/go/rl/rl.go`](bindings/go/rl/rl.go) for complete Go API.

### Running the NBA Example

```bash
# Build the example
cd build
cmake --build . --target nba_beam_example

# Run with default config
./nba_beam_example docs/v2/example_select.json 12 3 5

# Run with RL config
RL_CONFIG_PATH=docs/RL_CONFIG_EXAMPLE.json \
  ./nba_beam_example docs/v2/example_select.json 12 3 5
```

### Tests

```bash
# Run RL tests (9 tests without ONNX, 13 with ONNX)
cd build
./tests/v2/test_rl_api

# With ONNX support enabled
cmake -B build -DBUILD_ONNX=ON
cmake --build build --target test_rl_api
./build/tests/v2/test_rl_api  # 13 tests, 48 assertions
```

## Python Bindings

Python bindings are available via pybind11, providing a Pythonic interface to the V2 solver with full scout mode support.

**Installation:**
```bash
# Build with Python bindings enabled
mkdir -p build && cd build
cmake .. -DBUILD_PYTHON_BINDINGS=ON
cmake --build . --target knapsack_py

# Install with pip
cd ..
pip install .

# Or use directly from build directory
export PYTHONPATH=/path/to/knapsack/build:$PYTHONPATH
```

**Quick Start:**
```python
import knapsack

# Define problem
values = [60, 100, 120, 50, 80, 90]
weights = [10, 20, 30, 15, 25, 20]
capacity = 50

# Solve
solution = knapsack.solve(values, weights, capacity)
print(f"Best value: {solution.best_value}")
print(f"Selected items: {solution.selected_indices}")

# Scout mode for exact solver integration
result = knapsack.solve_scout(values, weights, capacity, 
                               {"scout_threshold": 0.5, "scout_top_k": 8})
print(f"Active items: {result.active_items}")
print(f"Reduction: {100 * (1 - result.active_item_count / result.original_item_count):.1f}%")

# Filter problem for exact solver (Gurobi, SCIP, etc.)
filtered_values = [values[i] for i in result.active_items]
filtered_weights = [weights[i] for i in result.active_items]
```

**Demo:**
```bash
cd build
PYTHONPATH=$PWD:$PYTHONPATH python3 ../bindings/python/example.py
```

See [`bindings/python/README.md`](bindings/python/README.md) for complete API documentation and examples.

**Comprehensive Examples:**

The [`examples/`](examples/) directory contains extensive real-world examples:
- `01_basic_knapsack.py` - Classic problems and edge cases
- `02_debt_portfolio.py` - Debt collection optimization
- `03_investment_portfolio.py` - Financial portfolio allocation
- `04_pandas_integration.py` - CSV data and DataFrame workflows
- `05_visualization.py` - Plotting and analysis
- `06_scout_mode.py` - Exact solver integration

Run any example:
```bash
cd examples/python
PYTHONPATH=../../build:$PYTHONPATH python3 01_basic_knapsack.py
```

## Go bindings (V2)

The V2 C API is wrapped for Go via cgo. Platform wrappers expose a consistent Go entry point:

```go
func SolveKnapsack(configJSON string, optionsJSON string) (*V2Solution, error)

type V2Solution struct {
    NumItems  int
    Select    []int   // length == NumItems; 0/1 per item
    Objective float64 // sum of weighted objective terms
    Penalty   float64 // total penalty from soft constraints
    Total     float64 // Objective - Penalty
}
```

Platform notes:
- Darwin/arm64: linked with `-framework Metal -framework Foundation -lc++`. Uses Metal at runtime with CPU fallback.
- Linux: linked with `-lstdc++` and `-lknapsack`. CUDA linkage is optional and guarded via a build tag (e.g., `-tags=cuda`).

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

### Minimal C usage (V2)

```c
#include "knapsack_c.h"
#include <stdio.h>

int main() {
    const char* cfg = "{\"mode\":\"select\",\"items\":[{\"obj\":[10],\"w\":[3]}],\"constraints\":[{\"capacity\":5}]}";
    const char* opts = "{\"beam_width\":8,\"iters\":3}";
    KnapsackSolutionV2* out = NULL;
    int rc = solve_knapsack_v2_from_json(cfg, opts, &out);
    if (rc != 0 || !out) { fprintf(stderr, "V2 solve failed (%d)\n", rc); return 1; }
    printf("num_items=%d total=%.3f\n", out->num_items, out->total);
    free_knapsack_solution_v2(out);
    return 0;
}
```

### Go (cgo) usage for Chariot

Use the V2 C API via cgo. The platform wrappers expose the same Go signature on macOS and Linux:

```go
func SolveKnapsack(configJSON string, optionsJSON string) (*V2Solution, error)
```

Ensure headers/libs are discoverable (`/usr/local/include/knapsack_c.h`, `/usr/local/lib/libknapsack.a`), or vendor them and point `-I`/`-L` accordingly. On macOS, Metal is used at runtime with CPU fallback; on Linux, CUDA linkage is optional and should be guarded with a build tag (e.g., `-tags=cuda`).

## Data

Sample CSVs live under `data/` (e.g., `entities.csv`, `entities_300.csv` for legacy tests). The executable reads `../data/entities.csv` by default when run from `build/`.

## Project layout

- `src/` main C++ sources (RoutePlanner, RecursiveSolver, etc.)
- `kernels/metal/` Metal API (Objective‑C++ bridge and shader)
- `kernels/` (CUDA kernels for non-Apple builds)
- `knapsack-library/` C API wrapper (used by other integrations)
- `rl/` RL support library (LinUCB + ONNX integration)
  - `rl_api.h` - C API header for RL functions
  - `rl_api.cpp` - Implementation with feature extraction, scoring, learning
- `bindings/go/metal/` Go cgo package for Metal evaluator
- `bindings/go/rl/` Go cgo package for RL support
- `bindings/python/` Python ctypes wrapper for RL support
- `tools/` Utilities including ONNX model generation
- `tests/v2/` Unit tests including RL and ONNX inference tests
- `docs/` Comprehensive documentation
  - `RL_SUPPORT.md` - RL API reference and guide
  - `BeamNextBestAction.md` - NBA usage patterns
  - `ONNX_INTEGRATION_STATUS.md` - ONNX setup and status
  - `ONNX_MODEL_GEN.md` - Model generation guide

## FAQ

- Q: Do I need the external `metal` CLI on macOS?
  - A: No. The shader is compiled at runtime via Metal APIs.

- Q: Can I still build the CUDA version?
  - A: Yes. On non‑Apple hosts, the build enables CUDA as configured by CMake. CUDA is not available on Apple Silicon; Metal is used instead with CPU fallback.

- Q: Where is the output written?
  - A: By default, `routes.csv` in the current working directory. You can specify a custom filename as the second command-line argument: `./knapsack_solver 50 my_output.csv`

- Q: Do I need ONNX Runtime for the RL support library?
  - A: No, it's optional. The RL library works with LinUCB bandit by default. Build with `-DBUILD_ONNX=ON` and install ONNX Runtime (`brew install onnxruntime` or `apt-get install libonnxruntime-dev`) to enable trained model inference.

- Q: How do I use my own trained models with RL support?
  - A: Train your model in Python (XGBoost, TensorFlow, PyTorch, scikit-learn), export to ONNX format, then load via the `model_path` config field. The model must have input shape `[batch, feat_dim]` and output shape `[batch]` (float32). See `docs/ONNX_MODEL_GEN.md` for examples.

- Q: What's the performance difference between LinUCB and ONNX models?
  - A: Both provide sub-millisecond batch scoring. LinUCB is simpler (linear weights) while ONNX models can capture non-linear patterns for better accuracy. The library gracefully falls back to LinUCB if the ONNX model fails to load.

- Q: Can I use RL support without the knapsack solver?
  - A: Yes! The RL library (`librl_support.a`) is independent and can be used standalone for any NBA scoring task. See `docs/RL_SUPPORT.md` for the C API and language bindings.