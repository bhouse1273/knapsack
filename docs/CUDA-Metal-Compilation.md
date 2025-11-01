Awesome—here’s a ready-to-use dual-backend template (CUDA for Jetson, Metal for Apple Silicon) with a Makefile switch:

* [Download knapsack_dual_backend_template.zip](sandbox:/mnt/data/knapsack_dual_backend_template.zip)

## What’s inside

```
include/backend.hpp                # Backend interface (shared)
src/backend_factory.cpp            # Chooses CUDA or Metal via -D macro
src/cuda_backend.cu                # CUDA backend (stubbed, drop in your kernel)
src/metal_backend.mm               # Metal backend (Objective-C++ bridge, stubbed)
shaders/eval_block_candidates.metal# Metal compute shader (placeholder)
src/main.cpp                       # Tiny driver proving both backends link/run
Makefile                           # 'make cuda' / 'make metal' / 'make all'
build/                              (created by Make)
```

## Build & run

### On Jetson (CUDA)

```bash
make cuda        # produces build/knapsack_dual_cuda
./build/knapsack_dual_cuda
```

* Set `CUDA_ARCH` if needed (defaults to `sm_87` for Orin; e.g., `make cuda CUDA_ARCH=sm_72` for Xavier).

### On Apple Silicon (Metal)

```bash
make metal       # produces build/knapsack_dual_metal + eval_block_candidates.metallib
./build/knapsack_dual_metal
```

* Requires Xcode CLT. The Makefile uses `xcrun -sdk macosx metal/metallib` and links `-framework Metal -framework Foundation`.

### Build both (on a Mac with no CUDA installed, CUDA target will be skipped gracefully)

```bash
make all
```

## How to plug in your real evaluators

* **CUDA**: Replace the placeholder in `src/cuda_backend.cu` with your kernel(s) and memory transfers. The interface accepts your SoA host view, candidate pack, terms/constraints, vans, etc., and fills `EvalOutHost`.
* **Metal**: Port the core evaluation loop into `shaders/eval_block_candidates.metal` and extend `src/metal_backend.mm` to create MTLBuffers for your SoA arrays, candidate lanes, and outputs, then dispatch with a thread grid sized to your `N_candidates`.

## Notes & tips

* Keep the **host interface identical** so the rest of your pipeline (NSQ/Chariot) doesn’t care which backend is used.
* Use **bit-packed candidates** the same way on both backends to avoid format conversion.
* For Apple Silicon, consider MPS (Metal Performance Shaders) if you later need reductions/scan primitives; for now, raw Metal compute is enough.
* Determinism: seed your RNG once on the host and pass seeds into both backends to keep results comparable.

If you want, I can flesh out the Metal buffers/dispatch code to match your current CUDA evaluator (with real inputs/outputs) next.
