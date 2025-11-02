Let’s wire Chariot to the new V2 solver via cgo on your Apple M1, with minimal moving parts and a quick way to smoke-test it.

## plan

- Build and install the knapsack static library (includes the V2 C API and a tiny CLI).
- Integrate cgo in Chariot (darwin/arm64) to call the V2 C API.
- Note the Metal requirement for the shader source and a low-friction way to get GPU on macOS; CPU fallback still works.
- Provide a short smoke test path and common troubleshooting.

## 1) Build and install the library (macOS, Apple Silicon)

Requirements:
- Xcode Command Line Tools (Apple clang, Metal frameworks)
- CMake 3.18+ and Go 1.20+ (your repo uses 1.22)

From the knapsack repo root:

```bash
# Configure and build all targets (lib, headers, CLI, tools)
cmake -S . -B build
cmake --build build -j

# Optional: sanity check the V2 CLI on the example config
./build/knapsack-library/knapsack_v2_cli docs/v2/example_select.json

# Install system-wide (headers to /usr/local/include, library to /usr/local/lib, CLI to /usr/local/bin)
sudo cmake --install build
```

What gets installed:
- knapsack_c.h
- /usr/local/lib/libknapsack.a
- /usr/local/bin/knapsack_v2_cli

Verification:

```bash
ls /usr/local/include/knapsack_c.h
ls /usr/local/lib/libknapsack.a
/usr/local/bin/knapsack_v2_cli --help 2>/dev/null || true
```

Notes
- The library uses Metal at runtime. If it can’t find the shader source, it automatically falls back to CPU (safe for functional tests).
- You do not need the external metal CLI; the library compiles MSL at runtime through the Metal framework.

## 2) Integrate cgo in Chariot (darwin/arm64)

You can call the V2 C API directly. Add a small cgo file in your Chariot repo (e.g., chariot/knapsack_v2_darwin.go) with this build tag:

- Build tag: darwin && arm64 (and optionally your own feature tag, e.g., knapsack).

cgo directives for the installed lib + frameworks:

```go
//go:build darwin && arm64

package chariot

/*
#cgo CFLAGS: -I/usr/local/include
#cgo LDFLAGS: -L/usr/local/lib -lknapsack -framework Metal -framework Foundation -lc++
#include "knapsack_c.h"
#include <stdlib.h>
*/
import "C"

import (
  "errors"
  "unsafe"
)

type V2Solution struct {
  NumItems  int
  Select    []int  // length NumItems, 0/1
  Objective float64
  Penalty   float64
  Total     float64
}

func SolveV2FromJSON(configJSON string, optionsJSON string) (*V2Solution, error) {
  if configJSON == "" {
    return nil, errors.New("empty config JSON")
  }
  cCfg := C.CString(configJSON)
  defer C.free(unsafe.Pointer(cCfg))

  var cOpts *C.char
  if optionsJSON != "" {
    cOpts = C.CString(optionsJSON)
    defer C.free(unsafe.Pointer(cOpts))
  }

  var out *C.KnapsackSolutionV2
  rc := C.solve_knapsack_v2_from_json(cCfg, cOpts, &out)
  if rc != 0 || out == nil {
    return nil, errors.New("solve_knapsack_v2_from_json failed")
  }
  defer C.free_knapsack_solution_v2(out)

  n := int(out.num_items)
  sel := make([]int, n)
  if n > 0 && out.select != nil {
    // Copy selection vector from C
    slice := (*[1 << 30]C.int)(unsafe.Pointer(out.select))[:n:n]
    for i := 0; i < n; i++ {
      sel[i] = int(slice[i])
    }
  }

  return &V2Solution{
    NumItems:  n,
    Select:    sel,
    Objective: float64(out.objective),
    Penalty:   float64(out.penalty),
    Total:     float64(out.total),
  }, nil
}
```

- CGO must be enabled (it is by default on macOS).
- The -framework Metal -framework Foundation -lc++ flags are required for the Objective‑C++ bridge and C++ symbols.
- No additional runtime dylibs are needed; libknapsack.a is static.

Options JSON (optional)
- You can pass a flat JSON string to tune the beam engine and dominance filters, e.g.:
  - {"beam_width":32,"iters":5,"seed":42,"debug":true,"dom_enable":true,"dom_eps":1e-9,"dom_surrogate":true}

## 3) Metal shader note (GPU vs CPU on macOS)

The V2 solver compiles the Metal shader from source at runtime. By default, it looks for the file:
- eval_block_candidates.metal (relative to the current working directory; a few nearby variants are probed)

If the shader isn’t found, the library falls back to CPU automatically. To ensure GPU on macOS, pick one:

- Easiest for now
  - At Chariot runtime, make the working directory a folder that contains the kernels/metal/shaders tree from this repo so the relative path resolves, OR
  - Copy eval_block_candidates.metal into a project-relative path the library probes (e.g., keep the same folder structure in your service repository).
- Longer-term (optional)
  - We can add an env var or embed the shader into the library to remove this runtime file dependency. If you want me to do that, I can implement it next.

CPU fallback is correct and deterministic; it just won’t use the GPU.

## 4) Smoke test in Chariot

- Make sure the library is installed as above.
- In Chariot, pick or create a tiny V2 config JSON (you can copy knapsack/docs/v2/example_select.json) and load it as a string.
- Call SolveV2FromJSON(configJSON, optsJSON) where optsJSON is optional.

Minimal runtime options to see debug logs:

```json
{"beam_width": 16, "iters": 3, "seed": 7, "debug": true}
```

Expect logs like:

- [beam] iter=0 best_total=… obj=… pen=… slacks=[…] pen_parts=[…]
- And a final objective/penalty/total with a 0/1 selection vector.

## 5) Troubleshooting

- ld: library not found for -lknapsack
  - Ensure /usr/local/lib/libknapsack.a exists. If not, re-run sudo cmake --install build.
- Undefined symbols for architecture arm64 (Metal/Foundation)
  - Ensure you’re linking -framework Metal -framework Foundation -lc++ in the cgo LDFLAGS.
- cgo header not found knapsack_c.h
  - Ensure knapsack_c.h exists and CFLAGS has -I/usr/local/include.
- Still CPU, not GPU
  - Provide the shader source at a probed relative path as noted above, or run from a directory that has kernels/metal/shaders/eval_block_candidates.metal.
- Workspace noise in VS Code
  - If gopls shows errors from other modules, add a go.work that only includes Chariot’s module, or set directory filters.

## 6) Optional: use the ready-made Go binding

If you prefer to avoid writing cgo in Chariot, you can import the binding we added:

- Module: github.com/bhouse1273/knapsack/bindings/go/v2
- API: v2.SolveJSON(configJSON, optsJSON) (*Solution, error)

Caveat: The binding currently looks for the static lib in a relative build path; for production in Chariot you’d still prefer the direct cgo approach above that links against the installed library. If you want me to add an alternate build tag in that binding to link against /usr/local automatically, I can implement that toggle next.

---

If you like, I can also add a tiny Chariot-side test helper that reads your config JSON from disk and prints the selected indices and totals, so you can validate the full path with one go test on M1.