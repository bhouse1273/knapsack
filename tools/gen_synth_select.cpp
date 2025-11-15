#include <iostream>
#include <vector>
#include <random>
#include <string>
#include <fstream>
#include <numeric>
#include <cstdlib>

// Simple synthetic select-mode JSON generator
// Usage: gen_synth_select <count> [cap_ratio=0.5] [output_path]
// Writes a v2 Config JSON with 'value' and 'weight' arrays.
// value ~ Uniform[10,100], weight ~ Uniform[1,10]
// capacity = cap_ratio * sum(weight)

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cerr << "Usage: " << argv[0] << " <count> [cap_ratio=0.5] [output_path]\n";
    return 1;
  }
  int count = std::atoi(argv[1]);
  if (count <= 0) {
    std::cerr << "count must be > 0\n"; return 1;
  }
  double cap_ratio = 0.5;
  if (argc >= 3) cap_ratio = std::atof(argv[2]);
  if (cap_ratio <= 0.0 || cap_ratio > 1.0) {
    std::cerr << "cap_ratio must be in (0,1]\n"; return 1;
  }
  std::string out_path;
  if (argc >= 4) {
    out_path = argv[3];
  } else {
    out_path = "data/benchmarks/synth_select_" + std::to_string(count) + ".json";
  }

  std::mt19937 rng(42);
  std::uniform_int_distribution<int> vdist(10, 100);
  std::uniform_int_distribution<int> wdist(1, 10);

  std::vector<int> values(count), weights(count);
  for (int i = 0; i < count; ++i) { values[i] = vdist(rng); weights[i] = wdist(rng); }
  long long wsum = std::accumulate(weights.begin(), weights.end(), 0LL);
  long long cap = static_cast<long long>(cap_ratio * static_cast<double>(wsum));
  if (cap < 1) cap = 1;

  std::ofstream out(out_path);
  if (!out) { std::cerr << "Failed to open output: " << out_path << "\n"; return 1; }
  out << "{\n";
  out << "  \"version\": 2,\n";
  out << "  \"mode\": \"select\",\n";
  out << "  \"random_seed\": 42,\n";
  out << "  \"items\": {\n";
  out << "    \"count\": " << count << ",\n";
  out << "    \"attributes\": {\n";
  out << "      \"value\": [\n        ";
  for (int i = 0; i < count; ++i) {
    out << values[i]; if (i + 1 != count) out << ","; if ((i+1) % 16 == 0) out << "\n        ";
  }
  out << "\n      ],\n";
  out << "      \"weight\": [\n        ";
  for (int i = 0; i < count; ++i) {
    out << weights[i]; if (i + 1 != count) out << ","; if ((i+1) % 16 == 0) out << "\n        ";
  }
  out << "\n      ]\n";
  out << "    }\n";
  out << "  },\n";
  out << "  \"blocks\": [ { \"name\": \"all\", \"start\": 0, \"count\": " << count << " } ],\n";
  out << "  \"objective\": [ { \"attr\": \"value\", \"weight\": 1.0 } ],\n";
  out << "  \"constraints\": [ { \"kind\": \"capacity\", \"attr\": \"weight\", \"limit\": " << cap << ", \"soft\": true, \"penalty\": { \"weight\": 10.0, \"power\": 2.0 } } ]\n";
  out << "}\n";
  out.close();
  std::cout << "Wrote " << out_path << " (count=" << count << ", cap=" << cap << ")\n";
  return 0;
}
