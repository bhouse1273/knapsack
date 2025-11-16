#include "rl_api.h"
#include <vector>
#include <string>
#include <cstring>
#include <cmath>
#include <mutex>
#include <random>
#include <cstdio>
#include <fstream>
#include <memory>
#include <iostream>

#ifdef RL_ONNX_ENABLED
#include <onnxruntime/onnxruntime_cxx_api.h>
#endif

struct LinUCB {
  int d = 8;
  double alpha = 0.2;
  std::vector<double> w; // weights
  LinUCB(int dim=8, double a=0.2): d(dim), alpha(a), w(dim, 0.0) {}
  double score(const std::vector<float>& feat) const {
    double mu = 0.0; int lim = std::min(d, (int)feat.size());
    for (int i=0;i<lim;++i) mu += w[i] * feat[i];
    double norm=0.0; for (int i=0;i<lim;++i) norm += feat[i]*feat[i];
    return mu + alpha * std::sqrt(norm);
  }
  void learn(const std::vector<float>& feat, double reward) {
    // Naive SGD update
    int lim = std::min(d, (int)feat.size());
    double pred = 0.0; for (int i=0;i<lim;++i) pred += w[i]*feat[i];
    double err = reward - pred;
    double lr = 0.05;
    for (int i=0;i<lim;++i) w[i] += lr * err * feat[i];
  }
};

struct RLContext {
  double w_rl = 1.0;      // weight for RL contribution (blend performed by caller)
  double alpha = 0.2;     // exploration parameter for LinUCB
  int feat_dim = 8;       // feature dimension
  LinUCB bandit;
  std::mutex mu;
  // Cache of last batch features (for learning updates)
  std::vector<float> last_features; // size = last_num_candidates * feat_dim
  int last_num_candidates = 0;
  std::string config_json; // original config JSON for debugging/logging
  // ONNX model configuration
  std::string model_path;
  std::string model_input = "input";
  std::string model_output = "output";
  bool model_loaded = false;
#ifdef RL_ONNX_ENABLED
  std::unique_ptr<Ort::Session> ort_session;
  Ort::AllocatorWithDefaultOptions allocator;
#endif
  RLContext(): bandit(feat_dim, alpha) {}
};

struct Handle { RLContext ctx; };

static bool parse_cfg(const char* json, RLContext* out) {
  // Minimal JSON key scanning (no external dependency). Accepts numeric values for:
  //   w_rl, alpha, feat_dim and string values for model_path, model_input, model_output
  // Example: {"w_rl":1.0,"alpha":0.3,"feat_dim":12, "model_path":"tests/v2/tiny_linear_8.onnx", "model_input":"input", "model_output":"output"}
  if (!out) return false;
  // Defaults
  double w_rl = 1.0;
  double alpha = 0.2;
  int feat_dim = 8;
  std::string model_path;
  std::string model_input = "input";
  std::string model_output = "output";
  
  if (json) {
    std::string s(json);
    auto find_number = [&](const std::string& key, double* out_num)->bool {
      size_t pos = s.find("\"" + key + "\"");
      if (pos == std::string::npos) return false;
      pos = s.find(':', pos);
      if (pos == std::string::npos) return false;
      // Skip spaces
      ++pos; while (pos < s.size() && std::isspace(static_cast<unsigned char>(s[pos]))) ++pos;
      // Read until non-number char
      size_t end = pos;
      while (end < s.size() && (std::isdigit(static_cast<unsigned char>(s[end])) || s[end]=='+' || s[end]=='-' || s[end]=='.' || s[end]=='e' || s[end]=='E')) ++end;
      if (end == pos) return false;
      try { *out_num = std::stod(s.substr(pos, end-pos)); return true; } catch (...) { return false; }
    };
    auto find_string = [&](const std::string& key)->std::string {
      size_t pos = s.find("\"" + key + "\"");
      if (pos == std::string::npos) return "";
      pos = s.find(':', pos);
      if (pos == std::string::npos) return "";
      ++pos; while (pos < s.size() && std::isspace((unsigned char)s[pos])) ++pos;
      if (pos < s.size() && s[pos]=='"') {
        size_t start = pos+1; size_t end = s.find('"', start);
        if (end != std::string::npos) return s.substr(start, end-start);
      }
      return "";
    };
    
    double tmp;
    if (find_number("w_rl", &tmp)) w_rl = tmp;
    if (find_number("alpha", &tmp)) alpha = tmp;
    if (find_number("feat_dim", &tmp)) feat_dim = (int)tmp;
    if (feat_dim <= 0) feat_dim = 8;
    
    // Extract string fields
    std::string mp = find_string("model_path");
    if (!mp.empty()) model_path = mp;
    std::string mi = find_string("model_input");
    if (!mi.empty()) model_input = mi;
    std::string mo = find_string("model_output");
    if (!mo.empty()) model_output = mo;
  }
  
  out->w_rl = w_rl;
  out->alpha = alpha;
  out->feat_dim = feat_dim;
  out->bandit = LinUCB(out->feat_dim, out->alpha);
  out->model_path = model_path;
  out->model_input = model_input;
  out->model_output = model_output;
  out->model_loaded = false;
  
#ifdef RL_ONNX_ENABLED
  // Real ONNX Runtime session initialization
  if (!model_path.empty()) {
    try {
      // Static Ort::Env (thread-safe, initialized once)
      static Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "RLContext");
      
      // Session options
      Ort::SessionOptions session_options;
      session_options.SetIntraOpNumThreads(1);
      session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_BASIC);
      
      // Create session with model path (macOS arm64 uses UTF-8 paths)
      out->ort_session = std::make_unique<Ort::Session>(env, model_path.c_str(), session_options);
      out->model_loaded = true;
      
      std::cout << "[RL] Loaded ONNX model: " << model_path 
                << " (feat_dim=" << feat_dim << ")" << std::endl;
    } catch (const Ort::Exception& e) {
      std::cerr << "[RL] ONNX session init failed: " << e.what() 
                << " - falling back to LinUCB bandit" << std::endl;
      out->model_loaded = false;
      out->ort_session.reset();
    }
  }
#endif
  
  return true;
}

static double rule_density_penalty(const unsigned char* bits, int n) {
  if (n<=0) return 0.0; int ones=0; for (int i=0;i<n;++i) ones += (bits[i]!=0);
  double density = (double)ones / (double)n;
  return (density>0.6)? -(density-0.6): 0.0; // penalize overly dense slates
}

extern "C" rl_handle_t rl_init_from_json(const char* json_cfg, char* err, int errlen) {
  try {
    auto* h = new Handle();
    if (!parse_cfg(json_cfg?json_cfg:"{}", &h->ctx)) {
      if (err && errlen) std::snprintf(err, errlen, "config parse failed");
      delete h; return nullptr;
    }
    h->ctx.config_json = json_cfg? json_cfg : "{}";
    return (rl_handle_t)h;
  } catch (...) {
    if (err && errlen) std::snprintf(err, errlen, "exception rl_init_from_json");
    return nullptr;
  }
}

// Internal helper: feature extraction for select mode.
static void extract_select_features(const unsigned char* candidates,
                                    int num_items, int num_candidates,
                                    int feat_dim, std::vector<float>& out) {
  out.resize((size_t)num_candidates * (size_t)feat_dim);
  for (int c=0;c<num_candidates;++c) {
    const unsigned char* row = candidates + (size_t)c * (size_t)num_items;
    int ones=0; for (int i=0;i<num_items;++i) ones += (row[i]!=0);
    float density = (num_items>0)? float(ones)/float(num_items) : 0.f;
    float* f = out.data() + (size_t)c * (size_t)feat_dim;
    std::fill(f, f+feat_dim, 0.f);
    if (feat_dim>0) f[0] = 1.f;           // bias
    if (feat_dim>1) f[1] = density;       // density
    if (feat_dim>2) f[2] = std::sqrt(float(ones)+1.f); // sqrt count
    // Simple hashed occupancy buckets for remaining dims
    for (int i=3;i<feat_dim;++i) {
      // Mix ones, i, and num_items
      uint32_t h = (uint32_t)((ones * 1315423911u) ^ (i*2654435761u) ^ (num_items*97531u));
      // Map to pseudo-random stable float in [0,1]
      h ^= (h << 13); h ^= (h >> 17); h ^= (h << 5);
      f[i] = (h & 0xFFFFFF) / float(0xFFFFFF);
    }
  }
}

// Internal helper: feature extraction for assign mode (binarized + per-bin occupancy stats).
// Representation: candidates row bytes are int8: -1 unassigned, >=0 bin index.
// Features layout (if feat_dim >=):
// 0 bias
// 1 density (fraction assigned)
// 2 sqrt(assigned_count+1)
// 3 variance of bin occupancy (population variance)
// 4.. per-bin normalized occupancy ratios (up to feat_dim limit)
// Remaining dims hashed for stability.
static void extract_assign_features(const unsigned char* candidates,
                                    int num_items, int num_candidates,
                                    int feat_dim, std::vector<float>& out) {
  out.resize((size_t)num_candidates * (size_t)feat_dim);
  if (feat_dim <= 0) return;
  // Assume modest K (bins). We infer K as max(bin_index)+1 across row.
  for (int c=0;c<num_candidates;++c) {
    const unsigned char* row = candidates + (size_t)c * (size_t)num_items;
    int max_bin = -1; int assigned=0;
    for (int i=0;i<num_items;++i) {
      signed char v = (signed char)row[i];
      if (v >= 0) { ++assigned; if (v > max_bin) max_bin = v; }
    }
    int K = max_bin + 1; if (K < 1) K = 1; // at least 1 to avoid div zero
    std::vector<int> occ(K,0);
    for (int i=0;i<num_items;++i) { signed char v = (signed char)row[i]; if (v>=0) occ[(int)v]++; }
    float density = (num_items>0)? float(assigned)/float(num_items) : 0.f;
    // Compute variance of occupancy
    float mean = (K>0)? float(assigned)/float(K) : 0.f;
    float var=0.f; for (int b=0;b<K;++b){ float diff = occ[b]-mean; var += diff*diff; }
    if (K>0) var /= float(K);
    float* f = out.data() + (size_t)c * (size_t)feat_dim;
    std::fill(f, f+feat_dim, 0.f);
    f[0] = 1.f;
    if (feat_dim>1) f[1] = density;
    if (feat_dim>2) f[2] = std::sqrt(float(assigned)+1.f);
    if (feat_dim>3) f[3] = var;
    int write_idx = 4;
    for (int b=0; b<K && write_idx < feat_dim; ++b) {
      f[write_idx++] = (assigned>0)? float(occ[b]) / float(assigned) : 0.f;
    }
    // Hash fill remaining
    for (int i=write_idx; i<feat_dim; ++i) {
      uint32_t h = (uint32_t)((assigned * 2654435761u) ^ (K*1315423911u) ^ (i*97531u));
      h ^= (h << 13); h ^= (h >> 17); h ^= (h << 5);
      f[i] = (h & 0xFFFFFF) / float(0xFFFFFF);
    }
  }
}

extern "C" int rl_prepare_features(rl_handle_t handle,
                                    const unsigned char* candidates,
                                    int num_items,
                                    int num_candidates,
                                    int mode,
                                    float* out_features,
                                    char* err, int errlen) {
  if (!handle || !candidates || !out_features || num_items<=0 || num_candidates<=0) {
    if (err && errlen) std::snprintf(err, errlen, "bad args"); return -1;
  }
  auto* h = (Handle*)handle;
  if (mode == 0) {
    std::vector<float> tmp;
    extract_select_features(candidates, num_items, num_candidates, h->ctx.feat_dim, tmp);
    std::memcpy(out_features, tmp.data(), tmp.size()*sizeof(float));
    return 0;
  } else if (mode == 1) {
    std::vector<float> tmp;
    extract_assign_features(candidates, num_items, num_candidates, h->ctx.feat_dim, tmp);
    std::memcpy(out_features, tmp.data(), tmp.size()*sizeof(float));
    return 0;
  } else {
    if (err && errlen) std::snprintf(err, errlen, "mode not supported"); return -2;
  }
  return 0;
}

extern "C" int rl_score_batch(rl_handle_t handle,
                               const char* context_json,
                               const unsigned char* candidates,
                               int num_items,
                               int num_candidates,
                               int mode,
                               double* out_scores,
                               char* err, int errlen) {
  (void)context_json; // reserved for future personalization hashing
  if (!handle || !candidates || !out_scores || num_items<=0 || num_candidates<=0) {
    if (err && errlen) std::snprintf(err, errlen, "bad args"); return -1;
  }
  auto* h = (Handle*)handle;
  std::vector<float> features;
  if (mode == 0) {
    extract_select_features(candidates, num_items, num_candidates, h->ctx.feat_dim, features);
  } else if (mode == 1) {
    extract_assign_features(candidates, num_items, num_candidates, h->ctx.feat_dim, features);
  } else {
    if (err && errlen) std::snprintf(err, errlen, "mode not supported"); return -2;
  }
  // Score each candidate
  h->ctx.last_features = features;
  h->ctx.last_num_candidates = num_candidates;
  for (int c=0;c<num_candidates;++c) {
    const unsigned char* row = candidates + (size_t)c * (size_t)num_items;
    const float* f = features.data() + (size_t)c * (size_t)h->ctx.feat_dim;
    std::vector<float> fvec(f, f + h->ctx.feat_dim);
    double s_rl = h->ctx.bandit.score(fvec);
    // Note: rl_score_batch always uses LinUCB bandit (no ONNX inference)
    // For ONNX scoring, use rl_prepare_features + rl_score_batch_with_features
    double s_rule = rule_density_penalty(row, num_items);
    out_scores[c] = s_rl + s_rule;
  }
  return 0;
}

extern "C" int rl_score_batch_with_features(rl_handle_t handle,
                                             const float* features,
                                             int feat_dim,
                                             int num_candidates,
                                             double* out_scores,
                                             char* err, int errlen) {
  if (!handle || !features || !out_scores || feat_dim<=0 || num_candidates<=0) {
    if (err && errlen) std::snprintf(err, errlen, "bad args"); return -1;
  }
  auto* h = (Handle*)handle;
  if (feat_dim != h->ctx.feat_dim) {
    if (err && errlen) std::snprintf(err, errlen, "feat_dim mismatch (expected %d)", h->ctx.feat_dim); return -2;
  }
  
  // Cache features for potential learning updates
  h->ctx.last_features.assign(features, features + (size_t)num_candidates * (size_t)feat_dim);
  h->ctx.last_num_candidates = num_candidates;
  
#ifdef RL_ONNX_ENABLED
  // Attempt ONNX inference if model loaded
  if (h->ctx.model_loaded && h->ctx.ort_session) {
    try {
      // Prepare input tensor: shape = [num_candidates, feat_dim]
      std::vector<int64_t> input_shape = {static_cast<int64_t>(num_candidates), static_cast<int64_t>(feat_dim)};
      size_t input_tensor_size = num_candidates * feat_dim;
      
      // Create input tensor (copy features to non-const buffer for safety)
      std::vector<float> input_data(features, features + input_tensor_size);
      auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
      Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
          memory_info, input_data.data(), input_tensor_size, 
          input_shape.data(), input_shape.size());
      
      // Input/output names
      const char* input_names[] = {h->ctx.model_input.c_str()};
      const char* output_names[] = {h->ctx.model_output.c_str()};
      
      // Run inference
      auto output_tensors = h->ctx.ort_session->Run(
          Ort::RunOptions{nullptr}, input_names, &input_tensor, 1, output_names, 1);
      
      // Extract output tensor (shape should be [num_candidates])
      float* output_data = output_tensors[0].GetTensorMutableData<float>();
      for (int c = 0; c < num_candidates; ++c) {
        out_scores[c] = static_cast<double>(output_data[c]);
      }
      
      return 0; // Success via ONNX
      
    } catch (const Ort::Exception& e) {
      if (err && errlen) {
        std::snprintf(err, errlen, "ONNX inference failed: %s", e.what());
      }
      std::cerr << "[RL] ONNX inference error: " << e.what() 
                << " - falling back to LinUCB" << std::endl;
      // Fall through to bandit fallback
    }
  }
#endif
  
  // Fallback: LinUCB bandit scoring (when ONNX disabled or failed)
  for (int c = 0; c < num_candidates; ++c) {
    const float* f = features + (size_t)c * (size_t)feat_dim;
    std::vector<float> fvec(f, f + feat_dim);
    double s_rl = h->ctx.bandit.score(fvec);
    // No rule penalty applied here; caller blends separately or use rl_score_batch for built-in rule.
    out_scores[c] = s_rl;
  }
  
  return 0;
}

extern "C" int rl_learn_batch(rl_handle_t handle, const char* feedback_json, char* err, int errlen) {
  if (!handle) { if (err && errlen) std::snprintf(err, errlen, "null handle"); return -1; }
  auto* h = (Handle*)handle;
  if (h->ctx.last_features.empty()) { if (err && errlen) std::snprintf(err, errlen, "no prior batch"); return -2; }
  // Helpers for tiny JSON parsing
  auto find_key = [](const std::string& s, const std::string& key)->size_t {
    return s.find("\"" + key + "\"");
  };
  auto parse_number_after_colon = [](const std::string& s, size_t key_pos, double def)->double {
    if (key_pos == std::string::npos) return def;
    size_t pos = s.find(':', key_pos); if (pos==std::string::npos) return def; ++pos;
    while (pos < s.size() && std::isspace(static_cast<unsigned char>(s[pos]))) ++pos;
    size_t end = pos;
    while (end < s.size() && (std::isdigit(static_cast<unsigned char>(s[end])) || s[end]=='+' || s[end]=='-' || s[end]=='.' || s[end]=='e' || s[end]=='E')) ++end;
    if (end == pos) return def;
    try { return std::stod(s.substr(pos, end-pos)); } catch (...) { return def; }
  };
  auto parse_number_array_for_key = [&](const std::string& s, const std::string& key)->std::vector<double> {
    std::vector<double> arr_out; size_t key_pos = find_key(s, key); if (key_pos==std::string::npos) return arr_out;
    size_t lb = s.find('[', key_pos); size_t rb = (lb==std::string::npos)? std::string::npos : s.find(']', lb);
    if (lb == std::string::npos || rb == std::string::npos || rb<=lb) return arr_out;
    std::string arr = s.substr(lb+1, rb-lb-1);
    size_t pos=0; while (pos < arr.size()) {
      while (pos < arr.size() && std::isspace(static_cast<unsigned char>(arr[pos]))) ++pos;
      size_t start = pos;
      while (pos < arr.size() && (std::isdigit(static_cast<unsigned char>(arr[pos])) || arr[pos]=='+' || arr[pos]=='-' || arr[pos]=='.' || arr[pos]=='e' || arr[pos]=='E')) ++pos;
      if (start!=pos) { try { arr_out.push_back(std::stod(arr.substr(start, pos-start))); } catch (...) {} }
      while (pos < arr.size() && arr[pos] != ',') ++pos; if (pos < arr.size() && arr[pos]==',') ++pos;
    }
    return arr_out;
  };
  auto parse_int_array_for_key = [&](const std::string& s, const std::string& key)->std::vector<int> {
    std::vector<int> out; size_t key_pos = find_key(s, key); if (key_pos==std::string::npos) return out;
    size_t lb = s.find('[', key_pos); size_t rb = (lb==std::string::npos)? std::string::npos : s.find(']', lb);
    if (lb == std::string::npos || rb == std::string::npos || rb<=lb) return out;
    std::string arr = s.substr(lb+1, rb-lb-1);
    size_t pos=0; while (pos < arr.size()) {
      while (pos < arr.size() && std::isspace(static_cast<unsigned char>(arr[pos]))) ++pos;
      size_t start = pos;
      while (pos < arr.size() && (std::isdigit(static_cast<unsigned char>(arr[pos])) || arr[pos]=='+' || arr[pos]=='-')) ++pos;
      if (start!=pos) { try { out.push_back(std::stoi(arr.substr(start, pos-start))); } catch (...) {} }
      while (pos < arr.size() && arr[pos] != ',') ++pos; if (pos < arr.size() && arr[pos]==',') ++pos;
    }
    return out;
  };

  // Parse feedback
  std::vector<double> rewards;
  if (feedback_json && *feedback_json) {
    std::string s(feedback_json);
    // 1) rewards array
    rewards = parse_number_array_for_key(s, "rewards");
    if (rewards.empty()) {
      // 2) chosen + base_reward + decay (+ optional positions)
      auto chosen = parse_int_array_for_key(s, "chosen");
      if (!chosen.empty()) {
        double base_reward = parse_number_after_colon(s, find_key(s, "base_reward"), 1.0);
        double decay = parse_number_after_colon(s, find_key(s, "decay"), 1.0);
        auto positions = parse_int_array_for_key(s, "positions");
        rewards.resize(chosen.size(), 0.0);
        for (size_t i=0;i<chosen.size();++i) {
          if (chosen[i]) {
            int pos = positions.empty() ? (int)i : positions[std::min(i, positions.size()-1)];
            double factor = 1.0; if (decay != 1.0 && pos>0) { factor = std::pow(decay, (double)pos); }
            rewards[i] = base_reward * factor;
          }
        }
      } else {
        // 3) events with idx and reward
        size_t events_pos = find_key(s, "events");
        if (events_pos != std::string::npos) {
          // Very simple parse: scan for pairs of idx and reward underneath events array
          // First estimate batch size for resizing rewards buffer
          rewards.assign(h->ctx.last_num_candidates, 0.0);
          size_t lb = s.find('[', events_pos); size_t rb = (lb==std::string::npos)? std::string::npos : s.find(']', lb);
          if (lb != std::string::npos && rb != std::string::npos && rb>lb) {
            std::string arr = s.substr(lb+1, rb-lb-1);
            size_t pos=0;
            while (pos < arr.size()) {
              size_t idx_key = arr.find("\"idx\"", pos);
              if (idx_key == std::string::npos) break;
              int idx = (int)parse_number_after_colon(arr, idx_key, -1);
              size_t rkey = arr.find("\"reward\"", idx_key);
              double r = parse_number_after_colon(arr, rkey, 0.0);
              if (idx >= 0 && idx < (int)rewards.size()) rewards[(size_t)idx] = r;
              pos = (rkey==std::string::npos? idx_key+5 : rkey+7);
            }
          }
        }
      }
    }
  }
  if (rewards.empty()) { if (err && errlen) std::snprintf(err, errlen, "no rewards parsed"); return -3; }
  int n = std::min(h->ctx.last_num_candidates, (int)rewards.size());
  for (int c=0;c<n;++c) {
    const float* f = h->ctx.last_features.data() + (size_t)c * (size_t)h->ctx.feat_dim;
    std::vector<float> fvec(f, f + h->ctx.feat_dim);
    h->ctx.bandit.learn(fvec, rewards[c]);
  }
  return 0;
}

extern "C" void rl_close(rl_handle_t h) { delete (Handle*)h; }

extern "C" int rl_get_feat_dim(rl_handle_t h) {
  if (!h) return -1; return ((Handle*)h)->ctx.feat_dim;
}

extern "C" int rl_get_last_batch_size(rl_handle_t h) {
  if (!h) return 0; return ((Handle*)h)->ctx.last_num_candidates;
}

extern "C" int rl_get_last_features(rl_handle_t h, float* out, int max) {
  if (!h || !out || max <= 0) return -1;
  auto* handle = (Handle*)h;
  int total = (int)handle->ctx.last_features.size();
  if (total == 0) return 0;
  int n = std::min(max, total);
  std::memcpy(out, handle->ctx.last_features.data(), (size_t)n * sizeof(float));
  return n;
}

extern "C" int rl_get_config_json(rl_handle_t h, char* out, int outlen) {
  if (!h || !out || outlen <= 1) return -1;
  auto* handle = (Handle*)h;
  const std::string& s = handle->ctx.config_json;
  int n = std::min(outlen - 1, (int)s.size());
  std::memcpy(out, s.data(), (size_t)n);
  out[n] = '\0';
  return n;
}
