#include <iostream>
#include <vector>
#include <cmath>
#include <string>

#include "v2/Config.h"
#include "v2/Data.h"
#include "v2/Eval.h"

static int fail(const std::string& msg) { std::cerr << "FAIL: " << msg << "\n"; return 1; }
static bool approx(double a, double b, double eps=1e-9) { return std::fabs(a-b) <= eps; }

int main() {
  // Use select-mode config
  v2::Config cfg; std::string err;
  if (!v2::LoadConfigFromFile("docs/v2/example_select.json", &cfg, &err)) return fail(err);

  v2::HostSoA soa; if (!v2::BuildHostSoA(cfg, &soa, &err)) return fail(err);

  // Candidate: select items 0 and 1
  v2::CandidateSelect cand;
  cand.select.assign(soa.count, 0u);
  cand.select[0] = 1; cand.select[1] = 1;

  v2::EvalResult res;
  if (!v2::EvaluateCPU_Select(cfg, soa, cand, &res, &err)) return fail(err);

  // Expected: objective = 10 + 12 = 22
  // weight sum = 8 + 10 = 18, limit 15 -> violation 3, penalty = 10 * 3^2 = 90
  // total = 22 - 90 = -68
  double expected_obj = 22.0;
  double expected_pen = 90.0;
  double expected_total = -68.0;

  if (!approx(res.objective, expected_obj)) return fail("objective mismatch");
  if (!approx(res.penalty, expected_pen)) return fail("penalty mismatch");
  if (!approx(res.total, expected_total)) return fail("total mismatch");
  if (cfg.constraints.size() != res.constraint_violations.size()) return fail("violations size mismatch");
  if (!approx(res.constraint_violations[0], 3.0)) return fail("violation mismatch");

  std::cout << "PASS: EvalCPU_Select objective=" << res.objective << " penalty=" << res.penalty << " total=" << res.total << "\n";
  return 0;
}
