# V2 Config Schema (preview)

This is a minimal preview of the V2 config used to drive a general, block-aware, multi-constraint solver.

- version: integer schema version (2)
- mode: "select" (single knapsack) or "assign" (K knapsacks)
- random_seed: unsigned integer for deterministic runs
- items:
  - count: number of items
  - attributes: map of attribute name -> array[double] of length `count` (SoA layout)
- blocks: list of blocks, each with either a contiguous range (`start`, `count`) or explicit `indices`
- knapsack (assign mode):
  - K: number of knapsacks
  - capacities: array[double] of length K
  - capacity_attr: attribute used for capacity consumption (e.g. "weight")
- objective: list of terms { attr, weight }
- constraints: list of { kind, attr, limit, soft, penalty { weight, power } }

See `example_villages.json` for a concrete instance.
