# Constraint-Aware QAOA Enhancements

This document summarizes the latest round of improvements to the knapsack QAOA path. The work was informed by the findings in docs/QAOA-p1-2511.17259v1.pdf and focuses on keeping constrained instances feasible at shallow depth while reducing barren regions in the parameter landscape.

## Highlights

- **Warm-start ingestion** now seeds each recursive block with multiple feasible classical bitstrings before any quantum-style passes run.
- **Constraint-preserving mixers** and **widened light cones** provide the non-diagonal, locality-aware mixing the paper recommends.
- **Hybrid analog/digital loop** alternates projected gradient-style adjustments with mixer sweeps, ensuring capacity feasibility without discarding good phases.
- **Schedule priors** restrict $(\beta, \gamma)$ ranges based on per-block density, avoiding the flat bands identified in Section 2.5 of the paper.

## Warm-Start Pipeline

- The shared recursion context gained explicit warm-start buffers (`warm_population`, `warm_seed`) so downstream evaluators can inject curated states without touching global solver state. See [include/Context.h](include/Context.h).
- `RoutePlanner` now generates a structured set of feasible candidates (priority density, geo bias, heavy-load bias, randomized) for each block before calling the recursive worker. These plans are deduplicated and attached to the context; details live in [src/RoutePlanner.cpp](src/RoutePlanner.cpp).

**Why it matters:** Warm-starts shrink the search horizon for sparse feasible regions, reducing the required QAOA depth and letting us exploit the “warm feasible subspace” strategy outlined in Section 3.4 of the referenced paper.

## Constraint QAOA Engine

A new module provides a purpose-built driver for constrained instances:

- API and configuration live in [include/qaoa/ConstraintQAOA.h](include/qaoa/ConstraintQAOA.h).
- Implementation resides in [src/qaoa/ConstraintQAOA.cpp](src/qaoa/ConstraintQAOA.cpp).

Key behaviors:

| Feature | Description |
| --- | --- |
| Layer schedule | Builds $(\beta, \gamma)$ trajectories that narrow as block density increases, preventing exponential flat bands. |
| Mixer | Uses entangler edges plus swap-like moves that conserve (or smoothly trade) capacity, matching the non-diagonal mixer requirement from Theorem 7. |
| Analog projection | After each phase or mixer sweep, states are projected toward the feasible region using density-ordered add/remove moves, emulating the analog drift loop in Section 4. |
| Population control | Unique-state filtering plus warm-start preservation keeps the population diverse without redundant work. |

## Entangling Layout Expansion

`recursive_worker` now constructs per-block entangler graphs using nearest-neighbor distances plus random long-range hops before calling the QAOA engine. This widens each layer’s light cone, countering the Lieb-Robinson-style limits cited in the paper. Implementation lives in [src/RecursiveSolver.cpp](src/RecursiveSolver.cpp).

## Hybrid Analog/Digital Evaluation

- QAOA-generated populations are evaluated via Metal when available, but a shared CPU scorer mimics the same objective/penalty structure when Metal is offline.
- The evaluation path is decoupled from candidate generation, so adding future emulators (e.g., QuEST, CUDA) only requires plugging into the scoring call.

## Operational Notes

1. No build flag changes are required; the new sources compile as part of the default target.
2. Warm-start heuristics live entirely in-classical code, so they run on every platform.
3. The QAOA engine is header-only on the public surface; external callers can reuse it if needed, but it is currently considered experimental.

## Next Steps

1. Benchmark the new stack on multi-constraint beam scenarios to quantify depth reduction vs. the previous greedy+Metal pipeline.
2. Expose telemetry (beta/gamma schedules, analog cycles) through DEBUG_VAN logs or a structured trace for future tuning.
3. Evaluate swapping the current analog projection with exact constraint-preserving BFS mixers on the hardest benchmarks.
