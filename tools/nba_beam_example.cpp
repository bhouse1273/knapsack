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
    for (size_t i = 0; i < expansions.size(); ++i) {
      ScoredCand sc{expansions[i], evals[i], 0.0};
      sc.s = scorer.score(sc.c, sc.sim, ctx);
      scored.push_back(std::move(sc));
    }
    std::partial_sort(scored.begin(), scored.begin() + std::min((size_t)K, scored.size()), scored.end(),
                      [](const ScoredCand& a, const ScoredCand& b){ return a.s > b.s; });

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
  return 0;
}
