#include <iostream>
#include <string>
#include <cassert>

#include "v2/Config.h"
#include "v2/Data.h"

static int fail(const std::string& msg) { std::cerr << "FAIL: " << msg << "\n"; return 1; }

int main(int argc, char** argv) {
  std::string path = argc > 1 ? argv[1] : std::string("docs/v2/example_villages.json");
  v2::Config cfg; std::string err;
  if (!v2::LoadConfigFromFile(path, &cfg, &err)) {
    return fail("LoadConfigFromFile: " + err);
  }

  v2::HostSoA soa;
  if (!v2::BuildHostSoA(cfg, &soa, &err)) {
    return fail("BuildHostSoA: " + err);
  }
  if (soa.count != cfg.items.count) {
    return fail("soa.count mismatch");
  }
  // Check attributes exist and sizes match
  for (const auto& kv : cfg.items.attributes) {
    auto it = soa.attr.find(kv.first);
    if (it == soa.attr.end()) return fail("missing attr: " + kv.first);
    if (it->second.size() != kv.second.size()) return fail("attr size mismatch: " + kv.first);
    // spot-check a couple values
    if (!it->second.empty() && it->second[0] != kv.second[0]) return fail("attr value[0] mismatch: " + kv.first);
  }

  // Build blocks and validate indices
  auto blocks = v2::BuildBlockSlices(cfg);
  if (blocks.size() != cfg.blocks.size()) return fail("blocks size mismatch");
  for (size_t i = 0; i < blocks.size(); ++i) {
    const auto& bspec = cfg.blocks[i];
    const auto& bs = blocks[i];
    if (bs.name != bspec.name) return fail("block name mismatch");
    if (bspec.start >= 0) {
      if ((int)bs.indices.size() != bspec.count) return fail("block indices size mismatch for " + bs.name);
      for (int j = 0; j < bspec.count; ++j) {
        if (bs.indices[j] != bspec.start + j) return fail("block indices value mismatch for " + bs.name);
      }
    } else {
      if (bs.indices != bspec.indices) return fail("block explicit indices mismatch for " + bs.name);
    }
  }

  std::cout << "PASS: SoA and block slices built successfully for " << soa.count << " items, "
            << blocks.size() << " blocks\n";
  return 0;
}
