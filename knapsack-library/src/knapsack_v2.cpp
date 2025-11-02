#include "knapsack_c.h"

#include <cstdlib>
#include <cstring>
#include <string>

// Minimal JSON options parsing (flat key-value extraction) to avoid adding dependencies here.

#include "v2/Config.h"
#include "v2/Data.h"
#include "v2/Engine.h"

extern "C" {

static KnapsackSolutionV2* alloc_solution_v2(int n) {
    KnapsackSolutionV2* s = (KnapsackSolutionV2*)std::malloc(sizeof(KnapsackSolutionV2));
    if (!s) return nullptr;
    s->num_items = n;
    s->select = (int*)std::calloc((size_t)n, sizeof(int));
    s->objective = 0.0;
    s->penalty = 0.0;
    s->total = 0.0;
    return s;
}

static bool parse_number_after(const std::string& s, const std::string& key, double* out) {
    size_t pos = s.find(key);
    if (pos == std::string::npos) return false;
    pos = s.find(":", pos);
    if (pos == std::string::npos) return false;
    // advance past colon and whitespace
    ++pos; while (pos < s.size() && isspace(static_cast<unsigned char>(s[pos]))) ++pos;
    char* endp = nullptr;
    const char* start = s.c_str() + pos;
    double v = std::strtod(start, &endp);
    if (endp == start) return false;
    *out = v; return true;
}

static bool parse_bool_after(const std::string& s, const std::string& key, bool* out) {
    size_t pos = s.find(key);
    if (pos == std::string::npos) return false;
    pos = s.find(":", pos);
    if (pos == std::string::npos) return false;
    ++pos; while (pos < s.size() && isspace(static_cast<unsigned char>(s[pos]))) ++pos;
    if (s.compare(pos, 4, "true") == 0) { *out = true; return true; }
    if (s.compare(pos, 5, "false") == 0) { *out = false; return true; }
    return false;
}

int solve_knapsack_v2_from_json(const char* json_config, const char* options_json, KnapsackSolutionV2** out_solution) {
    if (!out_solution) return -1;
    *out_solution = nullptr;
    if (!json_config) return -2;

    v2::Config cfg; std::string err;
    if (!v2::LoadConfigFromJsonString(std::string(json_config), &cfg, &err)) {
        return -3;
    }
    v2::HostSoA soa; if (!v2::BuildHostSoA(cfg, &soa, &err)) {
        return -4;
    }

    // Only select-mode is supported for now via SolveBeamSelect.
    if (cfg.mode != std::string("select")) {
        return -5; // unsupported mode yet
    }

    v2::SolverOptions opt; // defaults
    // Parse optional options_json: { "beam_width": int, "iters": int, "seed": uint, "debug": bool }
    if (options_json && options_json[0] != '\0') {
        std::string s(options_json);
        double v = 0.0; bool b = false;
        if (parse_number_after(s, "\"beam_width\"", &v)) opt.beam_width = (int)v;
        if (parse_number_after(s, "\"iters\"", &v)) opt.iters = (int)v;
        if (parse_number_after(s, "\"seed\"", &v)) opt.seed = (unsigned int)v;
        if (parse_bool_after(s, "\"debug\"", &b)) opt.debug = b;
        // Dominance filter flags (flat keys)
        if (parse_bool_after(s, "\"dom_enable\"", &b)) opt.enable_dominance_filter = b;
        if (parse_number_after(s, "\"dom_eps\"", &v)) opt.dom_eps = v;
        if (parse_bool_after(s, "\"dom_surrogate\"", &b)) opt.dom_use_surrogate = b;
    }
    v2::BeamResult r;
    if (!v2::SolveBeamSelect(cfg, soa, opt, &r, &err)) {
        return -6;
    }

    KnapsackSolutionV2* sol = alloc_solution_v2(soa.count);
    if (!sol) return -7;
    for (int i = 0; i < soa.count; ++i) sol->select[i] = r.best_select[i] ? 1 : 0;
    sol->objective = r.objective;
    sol->penalty = r.penalty;
    sol->total = r.total;

    *out_solution = sol;
    return 0;
}

void free_knapsack_solution_v2(KnapsackSolutionV2* sol) {
    if (!sol) return;
    if (sol->select) std::free(sol->select);
    std::free(sol);
}

int* ks_v2_select_ptr(KnapsackSolutionV2* sol) {
    if (!sol) return nullptr;
    return sol->select;
}

} // extern "C"
