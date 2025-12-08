#include <arrow/api.h>
#include <arrow/io/api.h>
#include <arrow/ipc/api.h>
#include <parquet/arrow/writer.h>

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct Instance {
  int items = 0;
  int constraints = 0;
  double best = 0.0;
  std::vector<double> profits;
  std::vector<std::vector<double>> weights;
  std::vector<double> capacities;
};

struct Options {
  std::filesystem::path root;
  std::string dataset = "or-library/mknap1.txt";
  int instance_index = 0;
  std::filesystem::path out_prefix;
  bool emit_arrow = true;
  bool emit_parquet = true;
  bool emit_config = true;
  std::string mode = "select";
};

std::filesystem::path DefaultRoot() {
  if (const char* env = std::getenv("KNAPSACK_TESTDATA_ROOT")) {
    if (*env) {
      return std::filesystem::path(env);
    }
  }
  return std::filesystem::path("data") / "benchmarks";
}

void PrintUsage(const char* argv0) {
  std::cerr << "Usage: " << argv0 << " [--root DIR] [--dataset REL_PATH] [--instance INDEX]\\n"
            << "             [--out PREFIX] [--arrow-only|--parquet-only] [--mode select|assign]\\n"
            << "Example: " << argv0 << " --dataset or-library/mknap1.txt --instance 3 \\\n             --out data/benchmarks/orlib_mknap1_i3" << std::endl;
}

bool ParseArgs(int argc, char** argv, Options* opts) {
  *opts = Options{};
  opts->root = DefaultRoot();
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--root" && i + 1 < argc) {
      opts->root = argv[++i];
    } else if (arg == "--dataset" && i + 1 < argc) {
      opts->dataset = argv[++i];
    } else if (arg == "--instance" && i + 1 < argc) {
      opts->instance_index = std::stoi(argv[++i]);
    } else if (arg == "--out" && i + 1 < argc) {
      opts->out_prefix = argv[++i];
    } else if (arg == "--arrow-only") {
      opts->emit_parquet = false;
    } else if (arg == "--parquet-only") {
      opts->emit_arrow = false;
    } else if (arg == "--no-config") {
      opts->emit_config = false;
    } else if (arg == "--mode" && i + 1 < argc) {
      opts->mode = argv[++i];
    } else if (arg == "--help" || arg == "-h") {
      PrintUsage(argv[0]);
      return false;
    } else {
      std::cerr << "Unknown argument: " << arg << std::endl;
      PrintUsage(argv[0]);
      return false;
    }
  }
  if (opts->instance_index < 0) {
    std::cerr << "instance index must be >= 0" << std::endl;
    return false;
  }
  if (opts->mode != "select" && opts->mode != "assign") {
    std::cerr << "mode must be 'select' or 'assign'" << std::endl;
    return false;
  }
  if (!opts->emit_arrow && !opts->emit_parquet) {
    std::cerr << "At least one format must be enabled" << std::endl;
    return false;
  }
  if (opts->out_prefix.empty()) {
    auto stem = std::filesystem::path(opts->dataset).filename().string();
    opts->out_prefix = std::filesystem::path("data") / "benchmarks" /
        (stem + "_instance" + std::to_string(opts->instance_index));
  }
  return true;
}

bool ReadInstance(const std::filesystem::path& file,
                  int target_index,
                  Instance* out) {
  std::ifstream in(file);
  if (!in) {
    std::cerr << "Failed to open dataset: " << file << std::endl;
    return false;
  }
  int total_instances = 0;
  if (!(in >> total_instances)) {
    std::cerr << "Dataset missing instance count" << std::endl;
    return false;
  }
  if (target_index >= total_instances) {
    std::cerr << "Requested instance " << target_index
              << " but dataset only has " << total_instances << std::endl;
    return false;
  }
  for (int idx = 0; idx < total_instances; ++idx) {
    int n = 0, m = 0;
    double best = 0.0;
    if (!(in >> n >> m >> best)) {
      std::cerr << "Failed to read header for instance " << idx << std::endl;
      return false;
    }
    std::vector<double> profits(n, 0.0);
    for (int i = 0; i < n; ++i) {
      if (!(in >> profits[i])) {
        std::cerr << "Failed to read profit " << i << " for instance " << idx << std::endl;
        return false;
      }
    }
    std::vector<std::vector<double>> weights(m, std::vector<double>(n, 0.0));
    for (int row = 0; row < m; ++row) {
      for (int col = 0; col < n; ++col) {
        if (!(in >> weights[row][col])) {
          std::cerr << "Failed to read weight[" << row << "][" << col
                    << "] for instance " << idx << std::endl;
          return false;
        }
      }
    }
    std::vector<double> capacities(m, 0.0);
    for (int j = 0; j < m; ++j) {
      if (!(in >> capacities[j])) {
        std::cerr << "Failed to read capacity " << j << " for instance " << idx << std::endl;
        return false;
      }
    }
    if (idx == target_index) {
      out->items = n;
      out->constraints = m;
      out->best = best;
      out->profits = std::move(profits);
      out->weights = std::move(weights);
      out->capacities = std::move(capacities);
      return true;
    }
  }
  return false;
}

std::shared_ptr<arrow::Array> BuildArray(const std::vector<double>& values) {
  arrow::DoubleBuilder builder;
  if (!builder.AppendValues(values).ok()) {
    throw std::runtime_error("Failed to append Arrow values");
  }
  std::shared_ptr<arrow::Array> array;
  if (!builder.Finish(&array).ok()) {
    throw std::runtime_error("Failed to finish Arrow array");
  }
  return array;
}

std::shared_ptr<arrow::Table> BuildTable(const Instance& inst) {
  std::vector<std::shared_ptr<arrow::Field>> fields;
  std::vector<std::shared_ptr<arrow::Array>> columns;
  fields.push_back(arrow::field("value", arrow::float64()));
  columns.push_back(BuildArray(inst.profits));
  for (int k = 0; k < inst.constraints; ++k) {
    auto name = std::string("weight_") + std::to_string(k);
    fields.push_back(arrow::field(name, arrow::float64()));
    columns.push_back(BuildArray(inst.weights[static_cast<std::size_t>(k)]));
  }
  auto schema = arrow::schema(fields);
  return arrow::Table::Make(schema, columns);
}

bool WriteArrow(const std::shared_ptr<arrow::Table>& table,
                const std::filesystem::path& path) {
  auto outfile_result = arrow::io::FileOutputStream::Open(path.string());
  if (!outfile_result.ok()) {
    std::cerr << "Failed to open Arrow output: " << path << " => "
              << outfile_result.status().ToString() << std::endl;
    return false;
  }
  auto outfile = *outfile_result;
  auto writer_result = arrow::ipc::MakeFileWriter(outfile, table->schema());
  if (!writer_result.ok()) {
    std::cerr << "Failed to create Arrow writer: "
              << writer_result.status().ToString() << std::endl;
    return false;
  }
  auto writer = *writer_result;
  if (!writer->WriteTable(*table).ok()) {
    std::cerr << "Arrow table write failed" << std::endl;
    return false;
  }
  if (!writer->Close().ok() || !outfile->Close().ok()) {
    std::cerr << "Failed to close Arrow output" << std::endl;
    return false;
  }
  return true;
}

bool WriteParquet(const std::shared_ptr<arrow::Table>& table,
                  const std::filesystem::path& path) {
  auto outfile_result = arrow::io::FileOutputStream::Open(path.string());
  if (!outfile_result.ok()) {
    std::cerr << "Failed to open Parquet output: " << path << " => "
              << outfile_result.status().ToString() << std::endl;
    return false;
  }
  auto outfile = *outfile_result;
  auto status = parquet::arrow::WriteTable(*table, arrow::default_memory_pool(), outfile, table->num_rows());
  if (!status.ok()) {
    std::cerr << "Parquet write failed: " << status.ToString() << std::endl;
    return false;
  }
  if (!outfile->Close().ok()) {
    std::cerr << "Failed to close Parquet output" << std::endl;
    return false;
  }
  return true;
}

bool WriteConfig(const Instance& inst,
                 const Options& opts,
                 const std::filesystem::path& arrow_path,
                 const std::filesystem::path& parquet_path,
                 const std::filesystem::path& json_path) {
  std::ofstream out(json_path);
  if (!out) {
    std::cerr << "Failed to open config output: " << json_path << std::endl;
    return false;
  }
  auto attr_spec = [&](const std::string& column) {
    out << "      \"" << column << "\": {\n"
        << "        \"kind\": \"file\",\n"
        << "        \"format\": \"arrow\",\n"
        << "        \"format_kind\": \"arrow\",\n"
        << "        \"path\": \"" << arrow_path.string() << "\",\n"
        << "        \"column_name\": \"" << column << "\"\n"
        << "      }";
  };
  out << "{\n";
  out << "  \"version\": 2,\n";
  out << "  \"mode\": \"" << opts.mode << "\",\n";
  out << "  \"random_seed\": 42,\n";
  out << "  \"items\": {\n";
  out << "    \"count\": " << inst.items << ",\n";
  out << "    \"sources\": {\n";
  attr_spec("value");
  out << ",\n";
  for (int k = 0; k < inst.constraints; ++k) {
    auto column = std::string("weight_") + std::to_string(k);
    attr_spec(column);
    if (k + 1 != inst.constraints) out << ",\n";
    else out << "\n";
  }
  out << "    }\n";
  out << "  },\n";
  out << "  \"objective\": [ { \"attr\": \"value\", \"weight\": 1.0 } ],\n";
  out << "  \"constraints\": [\n";
  for (int k = 0; k < inst.constraints; ++k) {
    out << "    { \"kind\": \"capacity\", \"attr\": \"weight_" << k
        << "\", \"limit\": " << inst.capacities[static_cast<std::size_t>(k)] << " }";
    if (k + 1 != inst.constraints) out << ",\n";
    else out << "\n";
  }
  out << "  ],\n";
  out << "  \"metadata\": {\n";
  out << "    \"dataset\": \"" << opts.dataset << "\",\n";
  out << "    \"instance\": " << opts.instance_index << ",\n";
  out << "    \"best_known\": " << inst.best << ",\n";
  out << "    \"arrow_path\": \"" << arrow_path.string() << "\",\n";
  out << "    \"parquet_path\": \"" << parquet_path.string() << "\"\n";
  out << "  }\n";
  out << "}\n";
  return true;
}

} // namespace

int main(int argc, char** argv) {
  Options opts;
  if (!ParseArgs(argc, argv, &opts)) {
    return 1;
  }
  auto dataset_path = opts.root / opts.dataset;
  Instance inst;
  if (!ReadInstance(dataset_path, opts.instance_index, &inst)) {
    return 2;
  }
  std::error_code ec;
  auto out_dir = opts.out_prefix.parent_path();
  if (!out_dir.empty()) {
    std::filesystem::create_directories(out_dir, ec);
    if (ec) {
      std::cerr << "Failed to create output directory " << out_dir << ": " << ec.message() << std::endl;
      return 3;
    }
  }
  std::shared_ptr<arrow::Table> table;
  try {
    table = BuildTable(inst);
  } catch (const std::exception& ex) {
    std::cerr << "Arrow table build failed: " << ex.what() << std::endl;
    return 4;
  }
  std::filesystem::path arrow_path = opts.out_prefix;
  arrow_path.replace_extension(".arrow");
  std::filesystem::path parquet_path = opts.out_prefix;
  parquet_path.replace_extension(".parquet");
  if (opts.emit_arrow) {
    if (!WriteArrow(table, arrow_path)) {
      return 5;
    }
    std::cout << "Wrote Arrow file: " << arrow_path << std::endl;
  }
  if (opts.emit_parquet) {
    if (!WriteParquet(table, parquet_path)) {
      return 6;
    }
    std::cout << "Wrote Parquet file: " << parquet_path << std::endl;
  }
  if (opts.emit_config) {
    auto json_path = opts.out_prefix;
    json_path.replace_extension(".json");
    if (!WriteConfig(inst, opts, arrow_path, parquet_path, json_path)) {
      return 7;
    }
    std::cout << "Wrote config: " << json_path << std::endl;
  }
  std::cout << "Instance " << opts.dataset << "[#" << opts.instance_index << "]"
            << " => items=" << inst.items << " constraints=" << inst.constraints << std::endl;
  return 0;
}
