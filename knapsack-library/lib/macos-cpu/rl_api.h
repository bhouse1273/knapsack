#pragma once
#ifdef __cplusplus
extern "C" {
#endif

// Opaque handle for RL context
typedef void* rl_handle_t;

// Initialize RL context from JSON config string. Returns nullptr on failure.
// Recognized keys: w_rl (double), alpha (double), feat_dim (int), model_path (string)
// err: optional buffer for error message.
rl_handle_t rl_init_from_json(const char* json_cfg, char* err, int errlen);

// Prepare features for a batch of candidates (SELECT mode only for now).
// candidates: mode=0 -> num_candidates * num_items bytes (0/1 per item)
// out_features: size num_candidates * feat_dim (row-major: candidate i at offset i*feat_dim)
// Returns 0 on success.
int rl_prepare_features(rl_handle_t h,
                        const unsigned char* candidates,
                        int num_items,
                        int num_candidates,
                        int mode,
                        float* out_features,
                        char* err, int errlen);

// Score a batch of candidates using internal feature path (legacy path) or assign-mode.
// mode: 0 select, 1 assign (experimental: candidate entries are int8: -1 for unassigned, >=0 bin index)
int rl_score_batch(rl_handle_t h,
                   const char* context_json,
                   const unsigned char* candidates,
                   int num_items,
                   int num_candidates,
                   int mode,
                   double* out_scores,
                   char* err, int errlen);

// Score a batch with caller-prepared features (preferred new path).
// features: num_candidates * feat_dim
int rl_score_batch_with_features(rl_handle_t h,
                                 const float* features,
                                 int feat_dim,
                                 int num_candidates,
                                 double* out_scores,
                                 char* err, int errlen);

// Learning update from feedback JSON.
// Accepted schemas (first found is used):
// 1) Explicit rewards per candidate:
//    {"rewards":[1.0,0.0,0.5]}
// 2) Structured choice with decay by (optional) position:
//    {"chosen":[1,0,1, ...], "base_reward":1.0, "decay":0.9, "positions":[0,1,2,...]}
//    Effective reward[i] = chosen[i]? base_reward * pow(decay, positions[i or i]) : 0
// 3) Event list:
//    {"events":[{"idx":2, "reward":1.5}, {"idx":0, "reward":0.2}]}
// Applies updates to most recent scored batch features. Returns 0 on success.
int rl_learn_batch(rl_handle_t h, const char* feedback_json, char* err, int errlen);

// Getter APIs for bindings / analytics.
// Returns feature dimension (>=1) or -1 if handle null.
int rl_get_feat_dim(rl_handle_t h);
// Returns size of last scored batch (0 if none or handle null).
int rl_get_last_batch_size(rl_handle_t h);

// Destroy context.
void rl_close(rl_handle_t h);

// Logging / analytics helpers
// Copy last batch features (up to max floats). Returns count copied, 0 if none, -1 on error.
int rl_get_last_features(rl_handle_t h, float* out, int max);
// Copy original config JSON into out (null-terminated). Returns length written (excluding null) or -1 on error.
int rl_get_config_json(rl_handle_t h, char* out, int outlen);

// NOTE: If model_path is provided in config JSON, the library attempts a stub load
// (reads file size) and adds a small bonus term model_factor = log(file_size+1)
// into the RL score before rule penalty blending. This is a placeholder for real ONNX inference.

#ifdef __cplusplus
}
#endif
