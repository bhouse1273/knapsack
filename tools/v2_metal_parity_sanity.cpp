#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <cmath>

#include "v2/Config.h"
#include "v2/Data.h"
#include "v2/Eval.h"

// Metal API (only available on Apple)
#ifdef __APPLE__
#include "metal_api.h"
#endif

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
  // Load select-mode config
  v2::Config cfg; std::string err;
  if (!v2::LoadConfigFromFile("docs/v2/example_select.json", &cfg, &err)) return fail(err);

  v2::HostSoA soa; if (!v2::BuildHostSoA(cfg, &soa, &err)) return fail(err);

  // Build CPU expected
  v2::CandidateSelect cand; cand.select.assign(soa.count, 0u); cand.select[0] = 1; cand.select[1] = 1;
  v2::EvalResult cpu;
  if (!v2::EvaluateCPU_Select(cfg, soa, cand, &cpu, &err)) return fail(err);

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
  if (knapsack_metal_init_from_source(msl.data(), msl.size(), ebuf, sizeof(ebuf)) != 0) {
    return fail(std::string("Metal init: ")+ebuf);
  }

  const int N = soa.count;
  const int bytes_per = (N + 3) / 4;
  std::vector<unsigned char> packed(bytes_per, 0);
  // pack lane 1 for selected items
  for (int i = 0; i < N; ++i) {
    unsigned lane = cand.select[i] ? 1u : 0u;
    const int byteIdx = (i >> 2);
    const int shift = (i & 3) * 2;
    unsigned char mask = (unsigned char)(0x3u << shift);
    packed[byteIdx] = (packed[byteIdx] & ~mask) | (unsigned char)((lane & 0x3u) << shift);
  }

  // Objective attr = value, capacity attr = weight; num_groups=1 with limit from the first capacity constraint.
  std::vector<float> values(N, 0.0f), weights(N, 0.0f), caps(1, 0.0f);
  for (int i = 0; i < N; ++i) {
    values[i] = (float)soa.attr.at("value")[i];
    weights[i] = (float)soa.attr.at("weight")[i];
  }
  double pen_w = 1.0, pen_p = 1.0;
  double limit = 0.0;
  for (const auto& c : cfg.constraints) {
    if (c.kind == std::string("capacity") && c.attr == std::string("weight")) {
      limit = c.limit; pen_w = c.soft ? c.penalty.weight : pen_w; pen_p = c.soft ? c.penalty.power : pen_p; break;
    }
  }
  if (limit <= 0.0) return fail("capacity limit not found");
  caps[0] = (float)limit;

  std::vector<float> obj(1, 0.0f), pen(1, 0.0f);
  MetalEvalIn in{};
  in.candidates = packed.data();
  in.num_items = N;
  in.num_candidates = 1;
  in.item_values = values.data();
  in.item_weights = weights.data();
  in.group_capacities = caps.data();
  in.num_groups = 1;
  in.penalty_coeff = (float)pen_w;
  in.penalty_power = (float)pen_p;
  MetalEvalOut out{ obj.data(), pen.data() };

  if (knapsack_metal_eval(&in, &out, ebuf, sizeof(ebuf)) != 0) return fail(std::string("Metal eval: ")+ebuf);

  const double total = (double)obj[0] - (double)pen[0];
  if (!approx(obj[0], cpu.objective) || !approx(pen[0], cpu.penalty) || !approx(total, cpu.total)) {
    std::cerr << "CPU: obj=" << cpu.objective << " pen=" << cpu.penalty << " total=" << cpu.total << "\n";
    std::cerr << "GPU: obj=" << obj[0] << " pen=" << pen[0] << " total=" << total << "\n";
    return fail("CPU vs Metal mismatch");
  }

  std::cout << "PASS: CPU vs Metal parity objective=" << obj[0] << " penalty=" << pen[0] << " total=" << total << "\n";
  return 0;
}
