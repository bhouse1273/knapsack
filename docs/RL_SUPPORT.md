# RL Support Integration (Preview)

This document outlines the initial reinforcement learning (RL) and rules-based scoring layer integrated with the V2 beam search pipeline.

## Goals
1. Provide a fast in-process scoring API for Next-Best Action (NBA) decisions.
2. Allow blending simulator (objective − penalties) with learned priors and simple rules.
3. Keep latency low (microseconds per batch) and support batch scoring aligned with beam expansions.
4. Offer a progressive path: heuristic → contextual bandit → full model (ONNX) → online learning.

## Components
| Component | Purpose |
|-----------|---------|
| `rl/rl_api.h` | C API for CGO / external integration |
| `rl/rl_api.cpp` | Minimal LinUCB-style bandit + density penalty rules |
| `nba_beam_example` | Demonstrates blending RL scores + simulator + mock rules |

## C API Summary (Updated)
```c
// Config
rl_handle_t rl_init_from_json(const char* json_cfg, char* err, int errlen);

// Feature preparation (preferred path) – mode 0 select, mode 1 assign (experimental)
int rl_prepare_features(rl_handle_t h,
                        const unsigned char* candidates,
                        int num_items, int num_candidates,
                        int mode,
                        float* out_features,
                        char* err, int errlen);

// Scoring (legacy internal feature + rule path or assign-mode binarization)
int rl_score_batch(rl_handle_t h, const char* context_json,
                   const unsigned char* candidates, int num_items,
                   int num_candidates, int mode, double* out_scores,
                   char* err, int errlen);

// Scoring with caller-provided features (no rule penalty applied)
int rl_score_batch_with_features(rl_handle_t h,
                                 const float* features,
                                 int feat_dim, int num_candidates,
                                 double* out_scores,
                                 char* err, int errlen);

// Learning update – expects JSON like {"rewards":[1.0,0.0,...]}
int rl_learn_batch(rl_handle_t h, const char* feedback_json, char* err, int errlen);

void rl_close(rl_handle_t h);
// Analytics helpers
int rl_get_feat_dim(rl_handle_t h);
int rl_get_last_batch_size(rl_handle_t h);
int rl_get_last_features(rl_handle_t h, float* out, int max);
int rl_get_config_json(rl_handle_t h, char* out, int outlen);
```

Candidate layout:
* Mode 0 (select): flat bytes (0/1) length = num_candidates * num_items.
* Mode 1 (assign, experimental): flat int8 bytes; -1 unassigned, >=0 assigned (binarized internally for density features).

## Configuration

The RL context is initialized with a JSON configuration string. Example fields:

```json
{
  "w_rl": 1.0,
  "alpha": 0.3,
  "feat_dim": 12,
  "model_path": "models/nba_scorer.onnx",
  "model_input": "input",
  "model_output": "output"
}
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `w_rl` | float | 1.0 | Weight for RL contribution in blending |
| `alpha` | float | 0.2 | Exploration parameter for LinUCB bandit fallback |
| `feat_dim` | int | 8 | Feature dimension (must match model if using ONNX) |
| `model_path` | string | "" | Path to ONNX model file (optional; requires BUILD_ONNX=ON) |
| `model_input` | string | "input" | ONNX model input tensor name |
| `model_output` | string | "output" | ONNX model output tensor name |

**ONNX Model Requirements:**
- Input shape: `[batch, feat_dim]` (float32)
- Output shape: `[batch]` (float32)
- IR version: 8 or lower (ONNX Runtime 1.22+ compatible)
- Opset: 13 recommended

**Fallback Behavior:**
If ONNX model loading fails (missing file, invalid format, or BUILD_ONNX=OFF), the system automatically falls back to LinUCB bandit scoring with a warning logged to stderr.

## Scoring Blend
```
TotalScore = SimComposite (objective - penalties + local mock RL/rule) + 0.3 * ExternalRL
```
ExternalRL currently: LinUCB score + density penalty. Replace with production model(s) as needed.

## Evolution Roadmap
| Phase | Enhancement | Status |
|-------|-------------|--------|
| MVP | LinUCB + simple density rule | Done |
| 1 | Feature extraction API (`rl_prepare_features`) | Done (select + assign slate features) |
| 2 | Online updates (`rl_learn_batch` structured feedback) | Done (rewards, chosen+decay, events) |
| 3 | ONNX Runtime inference (`model_path` with real session) | **Done** (BUILD_ONNX=ON, ONNX Runtime 1.22+) |
| 4 | Assign-mode richer features | Done (occupancy variance + per-bin ratios) |
| 5 | Distributed training / evaluation pipeline | Pending |

## Integration Points
1. Beam expansion: generate candidate bitvectors already compatible with RL batch scoring.
2. Evaluation: call `EvaluateMetal_Batch` or CPU fallback for simulator totals; call `rl_score_batch` for RL augment.
3. Ranking: combine scores and prune to beam width.
4. Logging: record (context, candidate, sim_total, rl_score, final_score, chosen_flag).

## Latency Considerations
The RL batch scorer is O(num_candidates * feature_dim). Current feature dim=8; typical beam expansion sets (hundreds) remain negligible versus simulator evaluation cost. Keep feature extraction linear and precompute reusable parts when introducing richer models.

## Next Steps
1. ~~Real ONNX inference~~ ✅ **Done** (BUILD_ONNX=ON enables ONNX Runtime 1.22+ integration)
2. Add context personalization features (user embedding hashing, time-of-day, recency stats).
3. Conflict / diversity metrics for assign mode (cross-bin attribute overlaps).
4. Batch serialization helper (JSON) for (features, scores, rewards).
5. Adjustable rule penalties via config: `rule_density_penalty_threshold`, `rule_density_penalty_scale`.
6. Multi-model ensembles: support `models:[{"path":"m1.onnx"},{"path":"m2.onnx"}]` with averaging.
7. Performance profiling hooks: optional timing output for feature prep vs scoring.

### Bindings
* Go: `bindings/go/rl/rl.go` (cgo). Use `InitFromJSON`, `PrepareFeatures`, `ScoreWithFeatures`, `Learn`.
* Python: `bindings/python/rl_support.py` (ctypes). Use `RL(...).score_select(...)` and `learn(...)`.
Requires shared library target `rl_support_shared` (built automatically now).

## Building with ONNX Support

To enable real ONNX model inference (vs. LinUCB bandit fallback):

### Prerequisites
```bash
# macOS (Homebrew)
brew install onnxruntime

# Linux (from source or package manager)
# See https://onnxruntime.ai/docs/install/
```

### Build Commands
```bash
# Configure with ONNX support
cmake -B build -DBUILD_ONNX=ON

# Build
cmake --build build

# Run tests (includes ONNX inference tests)
./build/tests/v2/test_rl_api
```

### Generating Test Models
```bash
# Install Python dependencies
pip install onnx numpy

# Generate test linear model (8 features)
python3 tools/gen_onnx_model.py 8 models/test_linear_8.onnx
```

## Try the Example
```
RL_CONFIG_PATH=docs/RL_CONFIG_EXAMPLE.json ./build/nba_beam_example docs/v2/example_select.json 12 3 5
```
Observe terminal output for Sim total vs Composite (now includes RL blend).

### Example Config (docs/RL_CONFIG_EXAMPLE.json)
```json
{
    "w_rl": 1.0,
    "alpha": 0.3,
    "feat_dim": 12,
    "model_path": "tests/v2/dummy_model.onnx"
}
```
Fields:
- `w_rl`: weight factor (reserved for future blending inside RL layer)
- `alpha`: exploration parameter for LinUCB (higher ⇒ more optimistic bonus)
- `feat_dim`: number of features used per candidate/context (first feature is bias)
- `model_path`: optional ONNX model file path; stub loader adds a small score bonus based on file size (placeholder for real inference)

---
For more on beam search usage see `BeamNextBestAction.md` and `BeamSearchAlgo.md`.
