#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <cctype>

// Convert samples/small_mkp.txt (simple MKP format) to V2 JSON
// Usage: convert_small_mkp_to_v2 [input_path] [output_path] [mode]
// Defaults: input=data/benchmarks/samples/small_mkp.txt,
//           output=data/benchmarks/small_mkp_v2.json,
//           mode=select (also supports: assign)

static bool read_numbers(std::istream& in, int needed, std::vector<double>& out) {
  out.clear();
  std::string line;
  while ((int)out.size() < needed && std::getline(in, line)) {
    // Skip comments and blank lines
    std::string trimmed = line;
    while (!trimmed.empty() && (trimmed.back()=='\r' || trimmed.back()=='\n' || std::isspace((unsigned char)trimmed.back()))) trimmed.pop_back();
    size_t pos = 0; while (pos < trimmed.size() && std::isspace((unsigned char)trimmed[pos])) ++pos;
    if (pos >= trimmed.size() || trimmed[pos] == '#') continue;
    std::istringstream ls(trimmed);
    double x;
    while (ls >> x) {
      out.push_back(x);
      if ((int)out.size() == needed) break;
    }
  }
  return (int)out.size() == needed;
}

int main(int argc, char** argv) {
  std::string in_path = "data/benchmarks/samples/small_mkp.txt";
  std::string out_path = "data/benchmarks/small_mkp_v2.json";
  std::string mode = "select";
  if (argc >= 2) in_path = argv[1];
  if (argc >= 3) out_path = argv[2];
  if (argc >= 4) mode = argv[3];

  std::ifstream in(in_path);
  if (!in) {
    std::cerr << "Failed to open input: " << in_path << "\n";
    return 1;
  }

  // Read first non-comment line for n and m
  int n = 0, m = 0;
  std::string line;
  while (std::getline(in, line)) {
    std::string trimmed = line;
    // trim leading spaces
    size_t i = 0; while (i < trimmed.size() && std::isspace((unsigned char)trimmed[i])) ++i;
    if (i >= trimmed.size() || trimmed[i] == '#') continue;
    std::istringstream ls(trimmed);
    if (ls >> n >> m) break;
  }
  if (n <= 0 || m <= 0) {
    std::cerr << "Failed to parse 'n m' header from: " << in_path << "\n";
    return 1;
  }

  // Read profits (n numbers)
  std::vector<double> profits;
  if (!read_numbers(in, n, profits)) {
    std::cerr << "Failed to read " << n << " profit values\n";
    return 1;
  }

  // Read capacities (m numbers)
  std::vector<double> capacities;
  if (!read_numbers(in, m, capacities)) {
    std::cerr << "Failed to read " << m << " capacities\n";
    return 1;
  }

  // Read weight matrix (m rows * n cols)
  std::vector<std::vector<double>> weight_matrix(m, std::vector<double>(n, 0.0));
  for (int k = 0; k < m; ++k) {
    std::vector<double> row;
    if (!read_numbers(in, n, row)) {
      std::cerr << "Failed to read weights row " << k << " (" << n << ")\n";
      return 1;
    }
    weight_matrix[k] = std::move(row);
  }

  std::ofstream out(out_path);
  if (!out) {
    std::cerr << "Failed to open output: " << out_path << "\n";
    return 1;
  }

  auto write_items = [&](std::ostream& os) {
    os << "  \"items\": {\n";
    os << "    \"count\": " << n << ",\n";
    os << "    \"attributes\": {\n";
    os << "      \"value\": [\n        ";
    for (int i = 0; i < n; ++i) { os << profits[i]; if (i+1 != n) os << ","; if ((i+1)%16==0) os << "\n        "; }
    os << "\n      ],\n";
    os << "      \"weight\": [\n        ";
    const std::vector<double>& weights0 = weight_matrix[0];
    for (int i = 0; i < n; ++i) { os << weights0[i]; if (i+1 != n) os << ","; if ((i+1)%16==0) os << "\n        "; }
    os << "\n      ]\n";
    os << "    }\n";
    os << "  },\n";
  };

  out << "{\n";
  out << "  \"version\": 2,\n";
  if (mode == "assign") {
    out << "  \"mode\": \"assign\",\n";
    out << "  \"random_seed\": 42,\n";
    write_items(out);
    out << "  \"blocks\": [ { \"name\": \"all\", \"start\": 0, \"count\": " << n << " } ],\n";
    out << "  \"knapsack\": { \"K\": " << m << ", \"capacities\": [";
    for (int k = 0; k < m; ++k) { out << capacities[k]; if (k+1 != m) out << ","; }
    out << "], \"capacity_attr\": \"weight\" },\n";
    out << "  \"objective\": [ { \"attr\": \"value\", \"weight\": 1.0 } ],\n";
    out << "  \"constraints\": []\n";
  } else {
    const double capacity = capacities[0];
    out << "  \"mode\": \"select\",\n";
    out << "  \"random_seed\": 42,\n";
    write_items(out);
    out << "  \"blocks\": [ { \"name\": \"all\", \"start\": 0, \"count\": " << n << " } ],\n";
    out << "  \"objective\": [ { \"attr\": \"value\", \"weight\": 1.0 } ],\n";
    out << "  \"constraints\": [ { \"kind\": \"capacity\", \"attr\": \"weight\", \"limit\": " << capacity << ", \"soft\": true, \"penalty\": { \"weight\": 10.0, \"power\": 2.0 } } ]\n";
  }
  out << "}\n";
  out.close();

  std::cout << "Wrote " << out_path << " from " << in_path
            << " (n=" << n << ", m=" << m << ", mode=" << mode << ")\n";
  return 0;
}
