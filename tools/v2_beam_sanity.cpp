#include <iostream>
#include <string>

#include "v2/Config.h"
#include "v2/Data.h"
#include "v2/Engine.h"

int main() {
  v2::Config cfg; std::string err;
  if (!v2::LoadConfigFromFile("docs/v2/example_select.json", &cfg, &err)) {
    std::cerr << "LoadConfig: " << err << "\n"; return 1;
  }
  v2::HostSoA soa; if (!v2::BuildHostSoA(cfg, &soa, &err)) { std::cerr << err << "\n"; return 1; }

  v2::SolverOptions opt; opt.beam_width = 8; opt.iters = 2; opt.seed = 42;
  v2::BeamResult res;
  if (!v2::SolveBeamSelect(cfg, soa, opt, &res, &err)) {
    std::cerr << "SolveBeamSelect: " << err << "\n"; return 1;
  }
  // Verify feasibility: weight <= limit
  double limit = 0.0; std::string capAttr;
  for (const auto& c : cfg.constraints) { if (c.kind == std::string("capacity")) { limit = c.limit; capAttr = c.attr; break; } }
  if (limit <= 0.0 || capAttr.empty()) { std::cerr << "No capacity constraint found\n"; return 1; }
  auto it = soa.attr.find(capAttr); if (it == soa.attr.end()) { std::cerr << "capacity attr missing\n"; return 1; }
  double wsum = 0.0; for (int i = 0; i < soa.count; ++i) if (res.best_select[i]) wsum += it->second[i];
  if (wsum > limit + 1e-6) { std::cerr << "FAIL: infeasible selection (" << wsum << " > " << limit << ")\n"; return 1; }

  std::cout << "PASS: Beam select produced feasible solution total=" << res.total << " obj=" << res.objective << " pen=" << res.penalty << "\n";
  return 0;
}
