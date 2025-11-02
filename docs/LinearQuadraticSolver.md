Yes. The pragmatic path is to surface an LP/QP engine in Chariot by wrapping proven open-source solvers with a stable C API and JSON models, then calling it via cgo (like V2). You don’t need to build a solver from scratch.

Recommendation
- Tier 1 (fastest path, permissive licenses):
  - LP + convex QP via OSQP (Apache-2.0). LP is QP with P=0. ADMM-based, approximate but robust and fast.
- Tier 2 (exact LP if needed):
  - LP via HiGHS (MIT) for simplex/interior-point exactness; QP stays with OSQP.
- One C API in this repo, two backends selectable by model type.

What you’ll expose to Chariot
- New Chariot functions (alongside knapsack):
  - linearSolve(modelJSON [, optionsJSON]) → LPSolution
  - quadSolve(modelJSON [, optionsJSON]) → QPSolution
- Keep varargs in closures: accept either direct JSON (power users) or structured args you validate and convert to JSON (friendly UX).
- You can later alias knapsackSolverLinear/Quadratic to these.

JSON Model (sparse, solver-agnostic)
- Common fields:
  - version: "lp1" or "qp1"
  - sense: "min" | "max"
  - n: number of variables
  - m: number of constraints
  - bounds: lb (len n), ub (len n) with +/-inf allowed
  - constraints: l (len m), u (len m) for range constraints (l ≤ Ax ≤ u)
  - A: sparse matrix in CSC or triplet form
    - Prefer triplets for authoring: A_triplet: {i:[], j:[], v:[]}
- LP-only:
  - c: objective linear term (len n), offset optional
- QP-only (convex):
  - c: as above
  - P_triplet: upper-triangular entries of symmetric P (i<=j), values compose 1/2 x^T P x
- Minimal examples:

````json
{
  "version": "lp1",
  "sense": "min",
  "n": 3,
  "m": 2,
  "c": [1, 2, 3],
  "bounds": { "lb":[0,0,0], "ub":[1e20,1e20,1e20] },
  "constraints": { "l":[-1e20, 1], "u":[4, 1] },
  "A_triplet": { "i":[0,0,1], "j":[0,1,2], "v":[1,1,1] }
}
````

````json
{
  "version": "qp1",
  "sense": "min",
  "n": 2,
  "m": 1,
  "c": [0, 0],
  "P_triplet": { "i":[0,1], "j":[0,1], "v":[1,1] },
  "bounds": { "lb":[-1e20,-1e20], "ub":[1e20,1e20] },
  "constraints": { "l":[1], "u":[1] },
  "A_triplet": { "i":[0,0], "j":[0,1], "v":[1,1] }
}
````

C API to add in this repo
- C header (installed to /usr/local/include):
  - int solve_lp_json(const char* model_json, const char* opts_json, LPSolution** out);
  - int solve_qp_json(const char* model_json, const char* opts_json, QPSolution** out);
  - Solutions: status, objective, x (len n), duals optional.
- Options:
  - OSQP: eps_abs, eps_rel, max_iter, polish, rho, time_limit
  - HiGHS: presolve, simplex|ipm, time_limit
- Build options:
  - -DBUILD_LPQP=ON
  - Link vendored OSQP (and HiGHS if Tier 2): static libs, no runtime deps.

Chariot cgo glue (darwin/arm64)
- Two small files, similar to V2:

````go
//go:build darwin && arm64

package chariot

/*
#cgo CFLAGS: -I/usr/local/include
#cgo LDFLAGS: -L/usr/local/lib -lknapsack -framework Foundation -lc++
#include "knapsack_c.h"
#include <stdlib.h>
*/
import "C"
import (
  "errors"
  "unsafe"
)

type LPSolution struct {
  Status    string
  Objective float64
  X         []float64
}

func linearSolve(modelJSON, optsJSON string) (*LPSolution, error) {
  if modelJSON == "" { return nil, errors.New("empty LP model") }
  cModel := C.CString(modelJSON); defer C.free(unsafe.Pointer(cModel))
  var cOpts *C.char
  if optsJSON != "" { cOpts = C.CString(optsJSON); defer C.free(unsafe.Pointer(cOpts)) }

  var out *C.LPSolution
  rc := C.solve_lp_json(cModel, cOpts, &out)
  if rc != 0 || out == nil { return nil, errors.New("LP solve failed") }
  defer C.free_lp_solution(out)

  n := int(out.n)
  x := make([]float64, n)
  if n > 0 && out.x != nil {
    sl := (*[1 << 30]C.double)(unsafe.Pointer(out.x))[:n:n]
    for i := 0; i < n; i++ { x[i] = float64(sl[i]) }
  }
  return &LPSolution{Status: C.GoString(out.status), Objective: float64(out.objective), X: x}, nil
}
````

- Linux/Jetson: same CFLAGS/LDFLAGS but without Foundation; link -lstdc++ -lm. No Metal required.

Why OSQP + HiGHS
- OSQP gives you LP and convex QP quickly via one JSON path (P=0 for LP). Great for large sparse problems, warm starts, and embedded use; returns accurate but approximate solutions with convergence criteria.
- HiGHS gives exact LP solutions (simplex/IPM) with strong presolve and duals; good when you need optimality guarantees for LP.
- Both are permissive licenses and easy to vendor into your static lib.

What’s not included
- Mixed-integer support (MILP/MIQP) like Gurobi. If you need integer variables, plan either:
  - Use HiGHS MIP path (adds complexity; still CPU-only), or
  - Keep Gurobi as an optional backend for those cases.

Rollout plan
- Phase 0: Decide Tier 1 vs Tier 2 (OSQP-only vs OSQP+HiGHS).
- Phase 1: Vendor solver(s) into knapsack repo, add C wrappers + JSON schema, install headers/libs.
- Phase 2: Add Chariot closures:
  - linearSolve(json [, opts]) and quadSolve(json [, opts])
  - Optional helpers to build models from triplets and dense inputs; validate in closures.
- Phase 3: Tests
  - Tiny LP/QP fixtures in Chariot; assert objective matches known optimum (LP via HiGHS) and OSQP status="solved".
- Phase 4: Docs
  - Add examples and options to README in both repos.

Effort estimate
- Tier 1 (OSQP only): 1–2 days to vendor, wrap, install, and add Chariot glue + smoke tests.
- Tier 2 (add HiGHS for LP): +2–3 days for build integration, wrapper, and tests (dual checks, sense, range constraints mapping).

If you want, I can start with Tier 1 (OSQP) and give you a working linearSolve/quadSolve path on M1 using the same install-and-cgo flow you already have for V2.