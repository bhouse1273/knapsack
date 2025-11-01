All set. I’ve fleshed out a functional Metal path with real buffers and a compute kernel that mirrors your CUDA evaluator shape.

* [Download knapsack_dual_backend_template_v2.zip](sandbox:/mnt/data/knapsack_dual_backend_template_v2.zip)

## What changed

* `shaders/eval_block_candidates.metal` — real evaluator:

  * Loops items in the active block, unpacks 2-bit assignments, accumulates objective terms (expr_ids 0–2), checks per-van seat capacity, and applies soft penalties.
* `src/metal_backend.mm` — full buffer/dispatch:

  * Builds MTLBuffers for SoA attributes, candidate lanes, objective/constraint arrays, outputs, and uniforms.
  * Dispatches one thread per candidate and returns `out.obj` and `out.soft_penalty`.
* `src/main.cpp` — packs per-candidate lanes properly and prints per-candidate results.

## Build & run on Apple Silicon

```bash
make metal
./build/knapsack_dual_metal
```

You should see something like:

```
Backend: metal
OK: Metal backend executed with real buffers/dispatch.
cand 0: obj=... pen=...
cand 1: obj=... pen=...
...
```

## Integrating with your solver

* Keep your CPU/CUDA evaluator logic; port the same expressions and constraints into the Metal shader switch blocks (currently supports expr_ids {0,1,2} and workers-GE soft constraint).
* If you need more attributes or constraints, extend:

  * `Uniforms` and kernel buffer list (new attribute buffer index),
  * `expr_eval(...)` switch for new term IDs,
  * Soft constraint handling (new `lhs_attr` or custom flags).

If you want me to port your exact CUDA expressions and add per-constraint debug outputs (slacks, per-van loads) to the Metal path, I can drop that in next.
