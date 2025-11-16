#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <iomanip>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "v2/Config.h"
#include "v2/Data.h"
#include "v2/Eval.h"
#include "rl/rl_api.h"

#if defined(__APPLE__) && defined(KNAPSACK_METAL_SUPPORT)
#include "metal_api.h"
#endif

using namespace v2;
using namespace std;

struct Context {
  uint64_t user_id = 42; // placeholder for personalization
};

struct NBAScorer {
  double w_sim = 1.0, w_rl = 0.2, w_rule = 0.1, w_div = 0.0;

  // Simple deterministic per-item RL score stub (could be a real model)
  vector<double> rl_item_score;

  void init_rl_scores(int n, uint64_t seed = 12345) {
    rl_item_score.resize(n);
    std::mt19937 rng(static_cast<uint32_t>(seed));
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    for (int i = 0; i < n; ++i) rl_item_score[i] = dist(rng);
  }

  double rl_model_score(const CandidateSelect& c, const Context&) const {
    double s = 0.0; // sum of selected per-item RL scores
    for (size_t i = 0; i < c.select.size() && i < rl_item_score.size(); ++i)
      if (c.select[i]) s += rl_item_score[i];
    return s;
  }

  double rule_score(const CandidateSelect& c) const {
    // Toy rule: mild penalty if slate is too dense (> 50% selected)
    int ones = 0;
    for (auto b : c.select) ones += (b != 0);
    double density = (c.select.empty() ? 0.0 : (double)ones / (double)c.select.size());
    return -max(0.0, density - 0.5); // negative if density > 0.5
  }

  double diversity_score(const CandidateSelect&) const {
    // Placeholder; could reward change vs last action set
    return 0.0;
  }

  double score(const CandidateSelect& c, const EvalResult& sim, const Context& ctx) const {
    double s_sim = sim.total;
    double s_rl = rl_model_score(c, ctx);
    double s_rule = rule_score(c);
    double s_div = diversity_score(c);
    return w_sim * s_sim + w_rl * s_rl + w_rule * s_rule + w_div * s_div;
  }
};

static CandidateSelect random_candidate(int n, std::mt19937& rng) {
  CandidateSelect c; c.select.resize(n, 0);
  std::bernoulli_distribution take(0.4);
  for (int i = 0; i < n; ++i) c.select[i] = take(rng) ? 1 : 0;
  return c;
}

static vector<CandidateSelect> mutate_all(const vector<CandidateSelect>& beam, int num_mut, std::mt19937& rng) {
  vector<CandidateSelect> out;
  for (const auto& base : beam) {
    std::uniform_int_distribution<int> idx(0, (int)base.select.size() - 1);
    for (int m = 0; m < num_mut; ++m) {
      CandidateSelect c = base;
      if (!c.select.empty()) {
        int j = idx(rng);
        c.select[j] ^= 1; // flip one bit
      }
      out.push_back(std::move(c));
    }
  }
  return out;
}

struct ScoredCand { CandidateSelect c; EvalResult sim; double s = 0.0; };

static bool load_metal_shader(std::string* out) {
#if defined(__APPLE__) && defined(KNAPSACK_METAL_SUPPORT)
  const char* paths[] = {
    "kernels/metal/shaders/eval_block_candidates.metal",
    "../kernels/metal/shaders/eval_block_candidates.metal",
    "../../kernels/metal/shaders/eval_block_candidates.metal"
  };
  for (auto p : paths) {
    std::ifstream in(p, std::ios::binary);
    if (in) { out->assign((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>()); if (!out->empty()) return true; }
  }
  return false;
#else
  (void)out; return false;
#endif
}

int main(int argc, char** argv) {
  std::string cfg_path = (argc >= 2 ? argv[1] : std::string("docs/v2/example_select.json"));
  int K = (argc >= 3 ? std::atoi(argv[2]) : 8);
  int iters = (argc >= 4 ? std::atoi(argv[3]) : 3);
  int muts_per = (argc >= 5 ? std::atoi(argv[4]) : 4);

  std::string err;
  Config cfg;
  if (!LoadConfigFromFile(cfg_path, &cfg, &err)) {
    std::cerr << "Failed to load config: " << err << "\n"; return 1;
  }
  if (cfg.mode != std::string("select")) {
    std::cerr << "Expected select-mode config for this example.\n"; return 1;
  }

  HostSoA soa;
  if (!BuildHostSoA(cfg, &soa, &err)) { std::cerr << "BuildHostSoA failed: " << err << "\n"; return 1; }

  // Optional: initialize Metal
#if defined(__APPLE__) && defined(KNAPSACK_METAL_SUPPORT)
  std::string shader;
  if (!load_metal_shader(&shader)) {
    std::cerr << "Warning: Metal shader not found; falling back to CPU.\n";
  } else {
    char m_err[512] = {0};
    if (knapsack_metal_init_from_source(shader.data(), shader.size(), m_err, sizeof(m_err)) != 0) {
      std::cerr << "Warning: Metal init failed: " << m_err << "; using CPU.\n";
    }
  }
#endif

  // Scorer setup
  Context ctx; NBAScorer scorer; scorer.init_rl_scores(soa.count, 777);
  // Load RL config JSON (example file or env override)
  std::string rl_cfg_path = "docs/RL_CONFIG_EXAMPLE.json";
  const char* env_cfg = std::getenv("RL_CONFIG_PATH");
  if (env_cfg && *env_cfg) rl_cfg_path = env_cfg;
  std::string rl_cfg_json = "{}";
  {
    std::ifstream rlf(rl_cfg_path);
    if (rlf) rl_cfg_json.assign((std::istreambuf_iterator<char>(rlf)), std::istreambuf_iterator<char>());
    else std::cerr << "Warning: RL config not found at " << rl_cfg_path << ", using defaults.\n";
  }
  // Initialize RL handle
  char rl_err[256] = {0};
  rl_handle_t rl = rl_init_from_json(rl_cfg_json.c_str(), rl_err, sizeof(rl_err));
  if (!rl) {
    std::cerr << "RL init failed: " << rl_err << " (continuing without RL)\n";
  } else {
    std::cout << "Loaded RL config from " << rl_cfg_path << "\n";
  }

  // Seed beam
  std::mt19937 rng(1234);
  vector<CandidateSelect> beam; beam.reserve(K);
  for (int i = 0; i < K; ++i) beam.push_back(random_candidate(soa.count, rng));

  ScoredCand global_best;
  global_best.s = -1e300;

  for (int it = 0; it < iters; ++it) {
    auto expansions = mutate_all(beam, muts_per, rng);

    // Evaluate
    vector<EvalResult> evals(expansions.size());
#if defined(__APPLE__) && defined(KNAPSACK_METAL_SUPPORT)
    std::string e2;
    if (!EvaluateMetal_Batch(cfg, soa, expansions, &evals, &e2)) {
      // Fallback to CPU
      for (size_t i = 0; i < expansions.size(); ++i) {
        if (!EvaluateCPU_Select(cfg, soa, expansions[i], &evals[i], &e2)) { std::cerr << e2 << "\n"; return 1; }
      }
    }
#else
    for (size_t i = 0; i < expansions.size(); ++i) {
      if (!EvaluateCPU_Select(cfg, soa, expansions[i], &evals[i], &err)) { std::cerr << err << "\n"; return 1; }
    }
#endif

    // Rank and prune
    vector<ScoredCand> scored; scored.reserve(expansions.size());
    // Prepare raw candidate bit matrix for RL batch scoring (mode=0 select)
    std::vector<unsigned char> cand_bits; cand_bits.reserve(expansions.size() * soa.count);
    for (auto& c : expansions) cand_bits.insert(cand_bits.end(), c.select.begin(), c.select.end());
    std::vector<double> rl_scores(expansions.size(), 0.0);
    if (rl) {
      // Parse feat_dim from rl_cfg_json (fallback 8)
      static int rl_feat_dim = 8;
      if (rl_feat_dim == 8) { // first use: attempt parse
        size_t pos = rl_cfg_json.find("\"feat_dim\"");
        if (pos != std::string::npos) {
          pos = rl_cfg_json.find(':', pos);
            if (pos != std::string::npos) {
              ++pos; while (pos < rl_cfg_json.size() && std::isspace((unsigned char)rl_cfg_json[pos])) ++pos;
              size_t end = pos;
              while (end < rl_cfg_json.size() && (std::isdigit((unsigned char)rl_cfg_json[end]) || rl_cfg_json[end]=='+' || rl_cfg_json[end]=='-' )) ++end;
              if (end > pos) {
                try { rl_feat_dim = std::stoi(rl_cfg_json.substr(pos, end-pos)); } catch (...) { /* ignore */ }
                if (rl_feat_dim <= 0) rl_feat_dim = 8;
              }
            }
        }
      }
      std::vector<float> features(expansions.size() * rl_feat_dim);
      char fe_err[128] = {0};
      if (rl_prepare_features(rl, cand_bits.data(), soa.count, (int)expansions.size(), 0, features.data(), fe_err, sizeof(fe_err)) != 0) {
        std::cerr << "rl_prepare_features failed: " << fe_err << "\n";
      } else {
        char sc_err[128] = {0};
        if (rl_score_batch_with_features(rl, features.data(), rl_feat_dim, (int)expansions.size(), rl_scores.data(), sc_err, sizeof(sc_err)) != 0) {
          std::cerr << "rl_score_batch_with_features failed: " << sc_err << "\n";
          std::fill(rl_scores.begin(), rl_scores.end(), 0.0);
        }
      }
    }
    for (size_t i = 0; i < expansions.size(); ++i) {
      ScoredCand sc{expansions[i], evals[i], 0.0};
      double sim_component = scorer.score(sc.c, sc.sim, ctx); // existing composite (sim+mock rl+rule)
      double rl_component = rl_scores[i];
      // Blend: weight existing composite heavily, add RL support term
      sc.s = sim_component + 0.3 * rl_component;
      scored.push_back(std::move(sc));
    }
    std::partial_sort(scored.begin(), scored.begin() + std::min((size_t)K, scored.size()), scored.end(),
                      [](const ScoredCand& a, const ScoredCand& b){ return a.s > b.s; });

    // Optional: simulate learning feedback using top-K pseudo rewards (higher score => reward 1.0 else 0.0)
    if (rl) {
      std::string feedback = "{\"rewards\":[";
      for (size_t i=0;i<expansions.size();++i) {
        feedback += (scored[i].s >= global_best.s ? "1" : "0");
        if (i+1<expansions.size()) feedback += ",";
      }
      feedback += "]}";
      char lerr[128] = {0};
      rl_learn_batch(rl, feedback.c_str(), lerr, sizeof(lerr));
    }
    beam.clear();
    for (int i = 0; i < K && i < (int)scored.size(); ++i) {
      beam.push_back(scored[i].c);
      if (scored[i].s > global_best.s) global_best = scored[i];
    }
  }

  // Report
  int ones = 0; for (auto b : global_best.c.select) ones += (b != 0);
  std::cout << "Best candidate after " << iters << " iters (K=" << K << ")\n";
  std::cout << "  Sim total:  " << std::fixed << std::setprecision(3) << global_best.sim.total << "\n";
  std::cout << "  Composite:  " << std::fixed << std::setprecision(3) << global_best.s << "\n";
  std::cout << "  Items on:   " << ones << "/" << global_best.c.select.size() << "\n";
  std::cout << "  Vector:     ";
  for (size_t i = 0; i < std::min<size_t>(global_best.c.select.size(), 32); ++i) std::cout << int(global_best.c.select[i]);
  if (global_best.c.select.size() > 32) std::cout << "...";
  std::cout << "\n";
  if (rl) rl_close(rl);
  return 0;
}
