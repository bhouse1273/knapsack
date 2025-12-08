**AOA/MOAOA Implementation Plan (for review)**

1. **Extend Config Schema**  
   - Update V2_JSON_SCHEMA.md, Config.h, and Config_json.cpp to describe new solver modes (`"aoa"`, `"moaoa"`), multi-objective blocks (`objectives[*].strategy`, `pareto_archive` params), and algorithm-specific options (population size, arithmetic parameters, annealing schedule).  
   - Add validation rules in Config_validate.cpp and Objective-C parser.

2. **Shared Evaluation Hooks**  
   - Refactor EvalCPU.cpp into reusable routines for objective sums, constraint slack, and penalty accumulation.  
   - Expose lightweight helpers (e.g., `EvaluateSelection(const HostSoA&, const Config&, const SelectionView&)`) so new solvers can score candidate solutions consistently with Beam Search.

3. **HostSoA Utilities**  
   - Introduce a `SelectionVector` abstraction (bitset/byte array) plus conversion helpers and incremental delta evaluation (reuse `HostSoABuilder` buffers).  
   - Provide warm-start serialization (JSON or binary) for seeding AOA/MOAOA.

4. **Arithmetic Optimization Algorithm (AOA)**  
   - Create `src/v2/solvers/AOA.{h,cpp}` implementing population initialization, arithmetic operators (addition, subtraction, multiplication, division), exploration→exploitation schedule, constraint repair/penalty injection, and stopping criteria.  
   - Support multi-threaded evaluation over candidates using the shared evaluation hooks.  
   - Surface via new entry point `SolveAOA(const Config&, const SolverOptions&, KnapsackSolutionV2*)`.

5. **Multi-Objective AOA (MOAOA)**  
   - Add Pareto-archive management (dominance checks, crowding distance) and weight-vector schedules in `src/v2/solvers/MOAOA.{h,cpp}`.  
   - Allow epsilon-constraint or weighted-sum fallback for single-solution requests; otherwise return top-K Pareto solutions.  
   - Provide serialization of the archive through `knapsack_c.h` and Go/Python bindings.

6. **Solver Dispatcher & CLI**  
   - Update knapsack_v2.cpp and `toosl/cli_v2.cpp` so `mode` or `solver` selects Beam, AOA, or MOAOA at runtime.  
   - Add CLI flags to control population sizes, iteration counts, Pareto archive size, and warm-start files.

7. **Bindings & API**  
   - Extend `knapsack_c.h`, Go CGO wrappers, and Python module (knapsack_py.cpp) with functions to request AOA/MOAOA solves and retrieve multiple solutions (archive).  
   - Ensure ABI compatibility (struct layouts, enums) across platforms.

8. **Testing**  
   - Add Catch2 suites under v2:  
     - `test_solver_aoa.cpp` (feasibility, convergence on small benchmarks, determinism with fixed seeds).  
     - `test_solver_moaoa.cpp` (Pareto dominance checks, archive reproducibility).  
   - Integrate into CMakeLists.txt and CI workflows.

9. **Benchmarking & Validation**  
   - Update `scripts/benchmark_batch_cpu_assign` (or new script) to compare Beam vs AOA/MOAOA on representative datasets; log runtime, objective, Pareto coverage.  
   - Produce summary in PERFORMANCE_STUDY.md.

10. **Documentation & Release Notes**  
    - Document usage, options, and best practices in BeamSearchAlgo.md, LargeDatasetOptions.md (if relevant), and a new `docs/AOA_vs_Beam.md`.  
    - Update README.md, READY_FOR_GO_CHARIOT.md, and CHANGES.md with feature highlights and integration guidance.

Let me know what to adjust before I start implementing.