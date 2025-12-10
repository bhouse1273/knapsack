# Using the Arithmetic Optimization Algorithms (AOA & MOAOA)

## 1. Why AOA / MOAOA?

The Arithmetic Optimization Algorithm (AOA) is a population-based meta-heuristic that navigates the search space through alternating exploration/exploitation phases and a lightweight annealing schedule. It is a good drop-in replacement for beam search when:
- the search surface is noisy or highly non-convex,
- you want more stochastic diversity than a deterministic beam,
- you prefer a single solution but need stronger escape from local optima.

Multi-Objective AOA (MOAOA) layers Pareto-archive logic and reference-weight vectors on top of the same arithmetic operators. Use it when:
- you must retain multiple trade-off solutions (e.g., value vs. risk vs. carbon),
- objectives have incompatible units and you do not want to pre-combine them,
- you need epsilon constraints and Pareto components in the same run.

Both solvers plug into the existing V2 JSON pipeline: the data ingestion, knapsack modeling, and bindings stay identical to beam search.

## 2. Prerequisites

1. Build or install the V2 CLI/bindings (identical to the base solver workflow):
   ```bash
   cmake -B build -DCMAKE_BUILD_TYPE=Release
   cmake --build build -j
   # Optional install step if you want knapsack_v2_cli in PATH
   sudo cmake --install build
   ```
2. Prepare a V2 JSON config (see `docs/V2_JSON_SCHEMA.md` for every field).
3. Decide on solver kind (`beam`, `aoa`, or `moaoa`) and describe it under the `solver` key in the JSON payload.

## 3. Describe objectives and strategies

AOA/MOAOA consume the same `objective` array as the beam solver. Each entry can set:
- `strategy: "weighted_sum"` (default) → contributes via `weight` like today.
- `strategy: "epsilon"` plus an `epsilon` bound → ensures the metric stays above a minimum (solver repairs violations with penalties instead of rejecting the candidate outright).
- `strategy: "pareto_component"` plus an optional `target` → exposes the metric as a dimension in the Pareto archive (MOAOA) or logs it in AOA runs.

Example objective block mixing all three strategies:

```json
"objective": [
  { "attr": "value", "weight": 1.0, "strategy": "weighted_sum" },
  { "attr": "risk", "strategy": "epsilon", "epsilon": 0.15 },
  { "attr": "co2", "strategy": "pareto_component", "target": 0.0 }
]
```

MOAOA requires at least one `pareto_component`; AOA can run with weighted sums only.

## 4. Configuring the AOA solver

Minimal solver block:

```json
"solver": {
  "kind": "aoa",
  "aoa": {
    "population": 256,
    "max_iterations": 1000,
    "exploration_rate": 0.55,
    "exploitation_rate": 0.45,
    "anneal_start": 1.0,
    "anneal_end": 0.01,
    "repair_penalty": 2.0
  }
}
```

Parameter notes:
- `population`: candidate count tracked per iteration (bigger = better diversity, higher eval cost).
- `max_iterations`: upper bound on arithmetic updates; affects runtime linearly.
- `exploration_rate` / `exploitation_rate`: relative time slices for global vs. local moves. They should sum to ≈1.0 (the solver normalizes but balanced inputs avoid surprises).
- `anneal_start` / `anneal_end`: temperature schedule for probabilistic acceptance; raise `anneal_start` if the search keeps restarting, lower `anneal_end` for greedier late iterations.
- `repair_penalty`: multiplier applied while repairing infeasible solutions; increase it when constraints are routinely violated.

## 5. Configuring the MOAOA solver

MOAOA nests the base AOA parameters plus Pareto-archive controls:

```json
"solver": {
  "kind": "moaoa",
  "moaoa": {
    "base": {
      "population": 192,
      "max_iterations": 800,
      "exploration_rate": 0.6,
      "exploitation_rate": 0.4,
      "anneal_start": 1.0,
      "anneal_end": 0.02,
      "repair_penalty": 1.5
    },
    "archive": {
      "max_size": 48,
      "dominance_epsilon": 1e-8,
      "diversity_metric": "crowding",
      "keep_feasible_only": true
    },
    "weight_vectors": 40,
    "archive_refresh": 10
  }
}
```

Key fields:
- `base`: identical knobs as single-objective AOA; tune it first.
- `archive.max_size`: total Pareto members retained; larger archives capture more nuance at the cost of extra dominance checks.
- `archive.dominance_epsilon`: tolerance while comparing objective vectors; loosen it (e.g., `1e-6`) if floating-point noise produces near-duplicates.
- `archive.diversity_metric`: `"crowding"` (default) keeps evenly spaced solutions; `"hypervolume"` favors improvements in dominated volume.
- `keep_feasible_only`: discard infeasible points (recommended when epsilon strategies already encode acceptable slack).
- `weight_vectors`: number of reference directions guiding MOAOA’s component-wise search; increase when optimizing ≥3 Pareto components.
- `archive_refresh`: number of iterations between archive resorting/trimming; reducing it gives smoother updates but costs more compute.

Remember to mark every metric you want inside the archive with `strategy: "pareto_component"`. You can still include weighted and epsilon terms; MOAOA will optimize all of them simultaneously.

## 6. Running the solver

Once the JSON config contains the `solver` block:

```bash
# Run directly with the CLI (reads solver.kind automatically)
./build/knapsack_v2_cli path/to/config.json

# Optional runtime overrides (seed, iteration cap, logging, etc.)
cat > /tmp/aoa_opts.json <<'JSON'
{ "seed": 42, "debug": true }
JSON
./build/knapsack_v2_cli path/to/config.json /tmp/aoa_opts.json
```

The same config works with:
- C API: `solve_knapsack_v2_from_json()` already parses the solver spec and returns the solver-specific solution.
- Go bindings: feed the JSON into `knapsack.SolveV2` (or equivalent) and the binding relays the solver selection to the native library.
- Python bindings: pass the JSON string/path to `knapsack.solve_v2_json`.

## 7. Interpreting results

- **AOA**: The solver still returns a single best selection plus metadata (`objective`, `penalty`, constraint slacks). Expect a different stochastic trajectory each run unless you pin `random_seed`.
- **MOAOA**: In addition to the incumbent best solution, the solver maintains a Pareto archive. The CLI prints a short summary per archive member (objective vector + constraint slack). In bindings, read the `pareto_archive` field from the returned solution object to iterate over all non-dominated candidates.

## 8. Practical tips

- Start from a small `population` (128) and `max_iterations` (400) to validate configs, then scale up until the score stabilizes.
- When objectives have wildly different scales, normalize the corresponding attributes ahead of time or use `target` values so the archive metrics remain comparable.
- Use epsilon strategies for metrics that must beat a firm threshold (service level, revenue floor) and Pareto components for metrics where you want to see the trade-off (risk, emissions, churn).
- Keep `weight_vectors` ≥ the number of Pareto components; otherwise, some dimensions will never receive a guiding direction.
- If MOAOA’s archive stops changing early, increase `exploration_rate` or relax `dominance_epsilon`.

With these settings in place you can flip between beam, AOA, and MOAOA by editing a single field in the JSON config while reusing every other part of your ingestion, CLI automation, and bindings.
