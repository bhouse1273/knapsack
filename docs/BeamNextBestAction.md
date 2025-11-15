# Beam Search for Next-Best Action (NBA) in a Business Decision Engine (BDE)

This note shows how to drive a Next-Best Action agent with beam search, how to encode constraints/objectives with the V2 config, and how to blend reinforcement learning (RL) or rules-based scoring into the selection.

- When to use: many candidate actions, multiple constraints (budgets, caps, eligibility, fairness), need fast decisions (10–500 ms) with good quality.
- Why beam: explores the top-K promising slates/sequences without exhaustive search, supports strict and soft constraints, and is GPU-friendly for scoring lots of candidates in parallel.

See also: `docs/BeamSearchAlgo.md` (mechanics) and `docs/BeamAsDataScout.md` (as a scout/stage-1), plus the code in `tools/v2_beam_sanity.cpp` and the batch evaluators.

## Mapping NBA to the V2 config

The V2 schema represents items, attributes, constraints, and an objective. For NBA:

- Items: actions (e.g., offers, messages, UI interventions)
- Attributes: per-action value signals (CTR uplift, revenue, risk score, cost, eligibility flags), contextual multipliers
- Objective: weighted sum of value attributes
- Constraints: budgets, frequency caps, channel limits, fairness/eligibility
- Mode:
  - "select": choose a set/slate of actions now (e.g., pick 1–N items)
  - "assign": assign actions to channels/slots (multi-knapsack)

Example (simplified select-mode):

```json
{
  "version": 2,
  "mode": "select",
  "items": {
    "count": N,
    "attributes": {
      "pred_value": [ ... ],
      "cost": [ ... ],
      "eligible": [0/1 ...]
    }
  },
  "blocks": [{"name":"all","start":0,"count":N}],
  "objective": [
    {"attr":"pred_value","weight":1.0}
  ],
  "constraints": [
    {"kind":"capacity","attr":"cost","limit": budget, "soft": true,
     "penalty": {"weight": lambda_cost, "power": 2.0}},
    {"kind":"capacity","attr":"eligible","limit": N, "soft": false}
  ]
}
```

Use blocks for segment/channel grouping; switch to `mode:"assign"` when assigning items to multiple slots/channels with per-slot capacities.

## Beam search inside an NBA agent

At each decision step, the agent needs the best next action (or slate):

1. Seed K candidates (beam) from heuristics: greedy by predicted value, rules-based picks, prior best, and a few random/diverse samples.
2. Expand: propose small edits to each candidate (flip include/exclude, swap action A→B, replace slot j with action i, etc.).
3. Score in parallel: evaluate objective − penalties for all expansions using the V2 evaluator (CPU or Metal GPU).
4. Rank and prune to top-K; keep a global best-so-far.
5. Stop on time/iteration/plateau; emit the highest-score candidate.

Because evaluation is stateless and parallel, step 3 scales well with GPU batch kernels; see `tools/benchmark_batch_cpu_vs_metal.cpp`.

## Blending RL or Rules into scoring

You often have a learned policy/value model or business rules you want to honor. Combine them with the simulator score:

Let
- S_sim(c) = simulated score (objective − penalties) from V2 evaluator
- S_rl(c) = policy/Q-value or propensity-based score from an RL/bandit model
- S_rule(c) = rules-based score/boosts/penalties (e.g., compliance, segment priorities)
- D(c) = diversity/novelty score (optional)

Composite score for beam ranking:

$$
S(c) = w_{sim}\, S_{sim}(c) + w_{rl}\, S_{rl}(c) + w_{rule}\, S_{rule}(c) + w_{div}\, D(c)
$$

Guidelines:
- Keep hard constraints as hard (use soft only when you want graceful trade-offs).
- Use S_rl as a prior/guide; S_sim enforces constraints and cost/value economics.
- Calibrate weights via offline policy eval or A/B tests.

### Where to inject RL/Rules

- Seeding: bias initial K solutions with the policy’s top suggestions.
- Expansion: prefer moves that increase policy score.
- Ranking: the composite S(c) above.
- Tie-breakers: pick higher S_rl among equals to exploit learned preferences.

## Tiny contract for an NBA Scorer

Inputs: candidate c (select vector or assign vector), context features, V2 SoA
Outputs: scalar score S(c), breakdown {sim, rl, rule, penalty}
Error modes: infeasible (hard constraint), NaNs/overflow, missing attrs

Edge cases to plan for:
- All candidates ineligible → return a safe default/“do nothing”.
- Score ties → deterministic tie-break (ID) to ensure stability.
- Very large action sets → cap beam width, enable diversity filtering.
- Tight latency budgets → limit expansions per iteration and use GPU batch.

## Code sketch (select-mode)

Pseudocode using existing primitives (`v2::EvaluateCPU_Select` / `EvaluateMetal_Batch`):

```cpp
struct NBAScorer {
  double w_sim=1, w_rl=0.2, w_rule=0.1, w_div=0.0;
  double score(const CandidateSelect& c, const Config& cfg, const HostSoA& soa,
               const EvalResult& sim, const Context& ctx) const {
    double s_sim = sim.total;
    double s_rl = rl_model_score(c, ctx);      // your model
    double s_rule = rule_score(c, ctx);        // business rules
    double s_div = diversity_score(c, ctx);    // optional
    return w_sim*s_sim + w_rl*s_rl + w_rule*s_rule + w_div*s_div;
  }
};

BeamState beam; // holds K candidates and their scores
initialize_beam(beam, K, seeds_from_greedy_and_rl());

for (int it = 0; it < max_iters && within_sla(); ++it) {
  auto expansions = mutate_all(beam.candidates, mut_params);
  // Evaluate in batch (GPU if available)
  std::vector<EvalResult> evals;
  #if defined(__APPLE__) && defined(KNAPSACK_METAL_SUPPORT)
    EvaluateMetal_Batch(cfg, soa, expansions, &evals, &err);
  #else
    evals.resize(expansions.size());
    for (size_t i=0;i<expansions.size();++i)
      EvaluateCPU_Select(cfg, soa, expansions[i], &evals[i], &err);
  #endif

  // Rank with blended score
  std::vector<ScoredCand> scored;
  scored.reserve(expansions.size());
  for (size_t i=0;i<expansions.size();++i) {
    double s = scorer.score(expansions[i], cfg, soa, evals[i], ctx);
    scored.push_back({expansions[i], s});
  }
  prune_to_topK(scored, beam, K, diversity_opts);
}
return beam.best();
```

For assign-mode (multi-slot/channel), substitute `CandidateAssign` and `EvaluateCPU_Assign`; a Metal batch path for assign can be added similarly to select.

## Serving pattern in a BDE

- Online loop:
  - Fetch context (user/session), filter ineligible actions, build V2 SoA once.
  - Run beam with tight SLA; return best action/slate + logging metadata (scores, constraints, ablation info).
- Offline loop:
  - Train/update RL policy (Q-value/propensity models) on logged data.
  - Calibrate weights {w_sim,w_rl,w_rule} with offline policy evaluation.
  - Refresh attributes (pred_value, risk, cost) periodically.

## Metal/CUDA acceleration

- Use GPU for the heavy part—candidate evaluation.
- On Apple, `EvaluateMetal_Batch` parallelizes select-mode scoring; see `tools/benchmark_batch_cpu_vs_metal.cpp` for usage and throughput characteristics.
- Profiling shows GPU overtakes CPU beyond a crossover batch size; tune beam width × expansions accordingly.

## Guardrails and compliance

- Hard constraints for legal/compliance/eligibility.
- Penalties for business policy (soft) when trade-offs are acceptable.
- Per-segment fairness: add ratio/capacity-like constraints at block-level.
- Explainability: log score breakdown {sim, rl, rule, penalty} per decision.

## Metrics and experimentation

- Primary: incremental value (lift), regret, SLA hit rate, constraint violations (should be 0 for hard constraints).
- Secondary: diversity, long-term retention metrics.
- Run A/B: (Beam-only) vs (RL-only) vs (Hybrid Beam+RL). Inspect exploration–exploitation trade-offs.

## Minimal checklist to integrate

- [ ] Define items and attributes in V2 (pred_value, cost, eligibility, risk, etc.)
- [ ] Encode business constraints (hard/soft) and objective weights
- [ ] Implement NBAScorer wrapper that blends sim + RL + rules
- [ ] Start with K ∈ [16, 64], 2–5 iterations; cap expansions per item
- [ ] Batch-evaluate on GPU when available; fall back to CPU
- [ ] Log decision traces for audit/learning

---

In short: beam search provides a constraint-aware, GPU-accelerated planner for NBA; RL supplies learned priors/values. Blend them to get fast, high-quality, and controllable recommendations under real-world constraints.