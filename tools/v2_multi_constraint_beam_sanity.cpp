#include <iostream>
#include <string>

#include "v2/Config.h"
#include "v2/Data.h"
#include "v2/Engine.h"
#include "v2/Eval.h"

static int fail(const std::string& msg) { std::cerr << "FAIL: " << msg << "\n"; return 1; }

int main() {
  v2::Config cfg; std::string err;
  if (!v2::LoadConfigFromFile("docs/v2/example_select_multi.json", &cfg, &err)) return fail(err);
  if (cfg.mode != "select") return fail("expected select mode");
  v2::HostSoA soa; if (!v2::BuildHostSoA(cfg, &soa, &err)) return fail(err);

  v2::SolverOptions opt; opt.beam_width = 16; opt.iters = 3; opt.seed = 7;
  v2::BeamResult res; if (!v2::SolveBeamSelect(cfg, soa, opt, &res, &err)) return fail(err);

  // Verify feasibility under all capacity constraints
  v2::CandidateSelect cand{res.best_select}; v2::EvalResult e;
  if (!v2::EvaluateCPU_Select(cfg, soa, cand, &e, &err)) return fail(err);
  bool feasible = true;
  for (double v : e.constraint_violations) if (v > 1e-9) { feasible = false; break; }
  if (!feasible) return fail("beam result violates soft constraints");

  std::cout << "PASS: Multi-constraint beam produced feasible solution total=" << res.total
            << " obj=" << res.objective << " pen=" << res.penalty << "\n";
  return 0;
}
