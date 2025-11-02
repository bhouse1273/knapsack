#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <cmath>

#include "v2/Config.h"
#include "v2/Data.h"
#include "v2/Eval.h"
#include "kernels/cuda/cuda_api.h"

static int fail(const std::string& msg) { std::cerr << "FAIL: " << msg << "\n"; return 1; }
static bool approx(double a, double b, double eps=1e-3) { return std::fabs(a-b) <= eps; }

static bool read_file_first_of(const std::vector<std::string>& paths, std::string* out) {
  for (const auto& p : paths) {
    std::ifstream in(p, std::ios::binary);
    if (!in) continue;
    out->assign((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    if (!out->empty()) return true;
  }
  return false;
}

int main() {
  v2::Config cfg; std::string err;
  if (!v2::LoadConfigFromFile("docs/v2/example_select_multi.json", &cfg, &err)) return fail(err);
  v2::HostSoA soa; if (!v2::BuildHostSoA(cfg, &soa, &err)) return fail(err);

  // Pick a candidate that violates both constraints
  v2::CandidateSelect cand; cand.select.assign(soa.count, 0u); for (int i = 0; i < std::min(4, soa.count); ++i) cand.select[i] = 1;
  v2::EvalResult cpu; if (!v2::EvaluateCPU_Select(cfg, soa, cand, &cpu, &err)) return fail(err);

  const int N = soa.count;
  const int bytes_per = (N + 3) / 4;
  std::vector<unsigned char> packed(bytes_per, 0);
  for (int i = 0; i < N; ++i) {
    unsigned lane = cand.select[i] ? 1u : 0u;
    const int byteIdx = (i >> 2);
    const int shift = (i & 3) * 2;
    unsigned char mask = (unsigned char)(0x3u << shift);
    packed[byteIdx] = (packed[byteIdx] & ~mask) | (unsigned char)((lane & 0x3u) << shift);
  }

  // Multi-term objective buffers
  const int T = (int)cfg.objective.size();
  std::vector<float> obj_weights(T, 0.0f); std::vector<float> obj_attrs(T * N, 0.0f);
  for (int t = 0; t < T; ++t) { obj_weights[t] = (float)cfg.objective[t].weight; const auto& a = soa.attr.at(cfg.objective[t].attr); for (int i = 0; i < N; ++i) obj_attrs[t * N + i] = (float)a[i]; }

  // Soft constraints buffers (only soft capacity constraints)
  std::vector<int> cons_idx; for (int i = 0; i < (int)cfg.constraints.size(); ++i) if (cfg.constraints[i].kind == "capacity" && cfg.constraints[i].soft) cons_idx.push_back(i);
  const int C = (int)cons_idx.size();
  std::vector<float> cons_attrs(C * N, 0.0f), cons_limits(C, 0.0f), cons_weights(C, 0.0f), cons_powers(C, 1.0f);
  for (int j = 0; j < C; ++j) { const auto& c = cfg.constraints[cons_idx[j]]; cons_limits[j] = (float)c.limit; cons_weights[j] = (float)c.penalty.weight; cons_powers[j] = (float)c.penalty.power; const auto& a = soa.attr.at(c.attr); for (int i = 0; i < N; ++i) cons_attrs[j * N + i] = (float)a[i]; }

  std::vector<float> obj(1, 0.0f), pen(1, 0.0f);
  CudaEvalIn in{}; in.candidates = packed.data(); in.num_items = N; in.num_candidates = 1; in.num_vans = 0; in.penalty_coeff = 0.0f; in.penalty_power = 1.0f; in.obj_attrs = obj_attrs.data(); in.obj_weights = obj_weights.data(); in.num_obj_terms = T; in.cons_attrs = cons_attrs.data(); in.cons_limits = cons_limits.data(); in.cons_weights = cons_weights.data(); in.cons_powers = cons_powers.data(); in.num_soft_constraints = C;
  CudaEvalOut out{ obj.data(), pen.data() };
  char ebuf[512] = {0};
  if (knapsack_cuda_eval(&in, &out, ebuf, sizeof(ebuf)) != 0) return fail(std::string("CUDA eval: ")+ebuf);

  const double total = (double)obj[0] - (double)pen[0];
  if (!approx(obj[0], cpu.objective) || !approx(pen[0], cpu.penalty) || !approx(total, cpu.total)) {
    std::cerr << "CPU: obj=" << cpu.objective << " pen=" << cpu.penalty << " total=" << cpu.total << "\n";
    std::cerr << "GPU(CUDA): obj=" << obj[0] << " pen=" << pen[0] << " total=" << total << "\n";
    return fail("CPU vs CUDA mismatch (multi-constraint)");
  }

  std::cout << "PASS: CUDA multi-constraint parity objective=" << obj[0] << " penalty=" << pen[0] << " total=" << total << "\n";
  return 0;
}
