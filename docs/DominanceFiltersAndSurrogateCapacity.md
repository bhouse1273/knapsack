# Dominance Filters and Surrogate Capacity

## actions taken

- Added a dominance filter preprocessor
  - New API: `ApplyDominanceFilters` with options and stats
    - Header: Preprocess.h
    - Impl: Preprocess.cpp
  - Behavior:
    - Computes per-item scalar objective value via weighted sum (requires non-negative weights to stay safe; otherwise filter is skipped)
    - Collects capacity constraints; two modes:
      - single: exact skyline using the single capacity attribute
      - surrogate: α[i] = Σ (penalty_weight/limit)·attr_c[i], then skyline on (α asc, value desc)
    - Scalar skyline rule: drop item when value ≤ running_best + epsilon at the same-or-smaller α
    - Outputs a filtered `HostSoA` and mapping `filtered_to_orig`
    - Stats include kept/dropped and method used

- Integrated into the beam solver
  - File: BeamSearch.cpp
  - If enabled, filter items before seeding; use the filtered SoA for evaluation (CPU/Metal)
  - At the end, map the best bitset back to the original item indexing so external APIs still return length-N selections
  - Debug mode prints a dominance summary line when filtering runs

- Extended solver options and C API options parsing
  - File: Engine.h
    - New fields: `enable_dominance_filter`, `dom_eps`, `dom_use_surrogate`
  - File: knapsack_v2.cpp
    - Parses flat keys in options JSON: dom_enable (bool), dom_eps (number), dom_surrogate (bool)

- Build wiring
  - Static library links `Preprocess.cpp`: CMakeLists.txt
  - Beam sanity executables include `Preprocess.cpp`: top-level CMakeLists.txt

- Verified builds and runs
  - Rebuilt all targets: knapsack_solver, parity tools, beam sanities, and knapsack_v2_cli
  - Ran CLI on example config with debug and dominance flags; run produced a valid solution (the tiny example didn’t drop items, as expected)

## how to use

- Enable dominance filtering from the V2 CLI options:
  ```bash
  cat > build-lib/opts_dom.json <<'JSON'
  { "beam_width": 16, "iters": 3, "seed": 7, "debug": true,
    "dom_enable": true, "dom_eps": 1e-9, "dom_surrogate": false }
  JSON
  ./build-lib/knapsack_v2_cli docs/v2/example_select.json build-lib/opts_dom.json
  ```
  - dom_surrogate=false uses the single-constraint exact skyline when only one capacity exists; set true for general cases.

- Programmatic options via C API
  - Pass the same flat keys in options JSON to `solve_knapsack_v2_from_json`.

## notes

- Safety constraints:
  - Filter runs only when all objective term weights are non-negative and capacities are present.
  - If any term has a negative weight, the filter is skipped to avoid removing items that could help a negative-weight objective.

- What you’ll see:
  - In debug mode, a line like:
    - [dominance] method=surrogate kept=123/200 dropped=77 eps=1e-09
  - Then the usual beam iteration logs.

## quality gates

- Build: PASS (all executables and the static lib)
- Runtime: PASS (CLI on example config)
- API stability: The C API continues to return a full-length selection bitset (we map the filtered solution back to original indices).

## next steps

- Property tests
  - Add tests verifying that filtering doesn’t worsen best-known totals on tiny instances, feasibility holds, and penalties are monotonic.
- Broader dominance
  - Add an exact 2-constraint skyline (O(n log n)) and a bounded O(n^2) exact K-constraint check behind a size threshold.
- Metrics in logs
  - Add dominance stats (kept/dropped) to the final summary in the CLI.
- Tuning
  - Try a slightly larger example to observe non-zero drops and quantify beam speed-ups.

