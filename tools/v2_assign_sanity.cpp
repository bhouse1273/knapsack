#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <cmath>

#include "v2/Config.h"
#include "v2/Data.h"
#include "v2/Eval.h"
#include "metal_api.h"

static int fail(const std::string& msg) { std::cerr << "FAIL: " << msg << "\n"; return 1; }
static bool approx(double a, double b, double eps=1e-4) { return std::fabs(a-b) <= eps; }

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
  // Load assign-mode config
  v2::Config cfg; std::string err;
  if (!v2::LoadConfigFromFile("docs/v2/example_assign.json", &cfg, &err)) return fail(err);

  v2::HostSoA soa; if (!v2::BuildHostSoA(cfg, &soa, &err)) return fail(err);
  if (cfg.mode != "assign" || cfg.knapsack.K != 2) return fail("unexpected config; need K=2 assign mode");

  // Candidate: assign items 0,1 -> group 0; 3,4 -> group 1; item 2 unassigned
  v2::CandidateAssign cand; cand.assign.assign(soa.count, -1);
  cand.assign[0] = 0; cand.assign[1] = 0; // group 0
  cand.assign[3] = 1; cand.assign[4] = 1; // group 1

  v2::EvalResult cpu;
  if (!v2::EvaluateCPU_Assign(cfg, soa, cand, &cpu, &err)) return fail(err);

  // Prepare Metal
  std::string msl;
  if (!read_file_first_of({
        "../kernels/metal/shaders/eval_block_candidates.metal",
        "kernels/metal/shaders/eval_block_candidates.metal",
        "../../kernels/metal/shaders/eval_block_candidates.metal"
      }, &msl)) {
    return fail("shader source not found");
  }
  char ebuf[512] = {0};
  if (knapsack_metal_init_from_source(msl.data(), msl.size(), ebuf, sizeof(ebuf)) != 0) return fail(std::string("Metal init: ")+ebuf);

  const int N = soa.count;
  const int bytes_per = (N + 3) / 4;
  std::vector<unsigned char> packed(bytes_per, 0);
  for (int i = 0; i < N; ++i) {
    unsigned lane = 0u;
  if (cand.assign[i] >= 0) lane = (unsigned)(cand.assign[i] + 1); // lane=group+1
    const int byteIdx = (i >> 2);
    const int shift = (i & 3) * 2;
    unsigned char mask = (unsigned char)(0x3u << shift);
    packed[byteIdx] = (packed[byteIdx] & ~mask) | (unsigned char)((lane & 0x3u) << shift);
  }

  std::vector<float> values(N, 0.0f), weights(N, 0.0f);
  for (int i = 0; i < N; ++i) { values[i] = (float)soa.attr.at("value")[i]; weights[i] = (float)soa.attr.at("weight")[i]; }
  std::vector<float> caps(cfg.knapsack.K, 0.0f);
  for (int k = 0; k < cfg.knapsack.K; ++k) caps[k] = (float)cfg.knapsack.capacities[k];

  // No soft constraint provided; default linear penalties (1.0, 1.0)
  std::vector<float> obj(1, 0.0f), pen(1, 0.0f);
  MetalEvalIn in{};
  in.candidates = packed.data();
  in.num_items = N;
  in.num_candidates = 1;
  in.item_values = values.data();
  in.item_weights = weights.data();
  in.group_capacities = caps.data();
  in.num_groups = cfg.knapsack.K;
  in.penalty_coeff = 1.0f;
  in.penalty_power = 1.0f;
  MetalEvalOut out{ obj.data(), pen.data() };

  if (knapsack_metal_eval(&in, &out, ebuf, sizeof(ebuf)) != 0) return fail(std::string("Metal eval: ")+ebuf);

  const double total = (double)obj[0] - (double)pen[0];
  if (!approx(obj[0], cpu.objective) || !approx(pen[0], cpu.penalty) || !approx(total, cpu.total)) {
    std::cerr << "CPU: obj=" << cpu.objective << " pen=" << cpu.penalty << " total=" << cpu.total << "\n";
    std::cerr << "GPU: obj=" << obj[0] << " pen=" << pen[0] << " total=" << total << "\n";
    return fail("CPU vs Metal mismatch");
  }

  std::cout << "PASS: Assign-mode parity objective=" << obj[0] << " penalty=" << pen[0] << " total=" << total << "\n";
  return 0;
}
