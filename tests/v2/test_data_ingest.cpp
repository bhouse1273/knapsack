#include "third_party/catch2/catch_amalgamated.hpp"

#include "v2/Config.h"
#include "v2/Data.h"

#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <optional>
#include <random>
#include <sstream>
#include <system_error>
#include <string>
#include <vector>

#ifdef KNAPSACK_ARROW_ENABLED
#include <arrow/builder.h>
#include <arrow/api.h>
#include <arrow/io/api.h>
#include <arrow/ipc/api.h>
#include <parquet/arrow/writer.h>
#endif

using namespace v2;

namespace {

bool PathExists(const std::filesystem::path& path) {
    std::error_code ec;
    return std::filesystem::exists(path, ec);
}

std::vector<std::filesystem::path> CandidateTestDataRoots() {
    std::vector<std::filesystem::path> roots;
    if (const char* env = std::getenv("KNAPSACK_TESTDATA_ROOT")) {
        std::filesystem::path env_path(env);
        if (!env_path.empty() && PathExists(env_path)) {
            roots.push_back(env_path);
        }
    }
    const std::vector<std::filesystem::path> defaults = {
        "/Volumes/mtheoryssd/2025-M-Theory/KnapsackTestData",
        std::filesystem::path("KnapsackTestData"),
        std::filesystem::path("tests") / "fixtures" / "KnapsackTestData",
        std::filesystem::path("data") / "external" / "KnapsackTestData"
    };
    for (const auto& candidate : defaults) {
        if (PathExists(candidate)) {
            roots.push_back(candidate);
        }
    }
    return roots;
}

std::optional<std::filesystem::path> FindTestDataFile(const std::vector<std::filesystem::path>& relative_candidates) {
    auto roots = CandidateTestDataRoots();
    for (const auto& root : roots) {
        for (const auto& rel : relative_candidates) {
            auto candidate = root / rel;
            if (PathExists(candidate)) {
                return candidate;
            }
        }
    }
    return std::nullopt;
}

std::string TrimWhitespace(const std::string& input) {
    std::size_t start = 0;
    while (start < input.size() && std::isspace(static_cast<unsigned char>(input[start]))) {
        ++start;
    }
    std::size_t end = input.size();
    while (end > start && std::isspace(static_cast<unsigned char>(input[end - 1]))) {
        --end;
    }
    return input.substr(start, end - start);
}

bool NextDataLine(std::istream& in, std::string* out_line) {
    std::string line;
    while (std::getline(in, line)) {
        auto hash = line.find('#');
        if (hash != std::string::npos) {
            line = line.substr(0, hash);
        }
        line = TrimWhitespace(line);
        if (!line.empty()) {
            *out_line = line;
            return true;
        }
    }
    return false;
}

bool ParseLineDoubles(const std::string& line, int expected_count, std::vector<double>* out) {
    std::istringstream iss(line);
    out->clear();
    double value = 0.0;
    while (iss >> value) {
        out->push_back(value);
    }
    if (expected_count >= 0 && static_cast<int>(out->size()) != expected_count) {
        return false;
    }
    return !out->empty();
}

struct SmallMKPFixture {
    int item_count = 0;
    int knapsack_count = 0;
    std::vector<double> profits;
    std::vector<double> capacities;
    std::vector<std::vector<double>> weights;
    std::filesystem::path source_path;
};

Config BuildConfigForFixture(const SmallMKPFixture& fixture,
                             const std::filesystem::path& file_path,
                             const std::string& format_name,
                             AttributeFormatKind format_kind) {
    Config cfg;
    cfg.items.count = fixture.item_count;
    auto make_spec = [&](const std::string& column) {
        AttributeSourceSpec spec;
        spec.kind = AttributeSourceKind::kFile;
        spec.format = format_name;
        spec.format_kind = format_kind;
        spec.path = file_path.string();
        spec.column_name = column;
        return spec;
    };
    cfg.items.sources["value"] = make_spec("value");
    for (int i = 0; i < fixture.knapsack_count; ++i) {
        auto column_name = std::string("weight_") + std::to_string(i);
        cfg.items.sources[column_name] = make_spec(column_name);
    }
    return cfg;
}

std::optional<SmallMKPFixture> LoadSmallMKPFixture() {
    const std::vector<std::filesystem::path> candidates = {
        std::filesystem::path("samples") / "small_mkp.txt",
        std::filesystem::path("small_mkp.txt")
    };
    auto path = FindTestDataFile(candidates);
    if (!path) {
        return std::nullopt;
    }
    std::ifstream in(*path);
    if (!in) {
        return std::nullopt;
    }
    std::string line;
    SmallMKPFixture fixture;
    fixture.source_path = *path;
    if (!NextDataLine(in, &line)) {
        return std::nullopt;
    }
    {
        std::istringstream header(line);
        if (!(header >> fixture.item_count >> fixture.knapsack_count)) {
            return std::nullopt;
        }
    }
    if (fixture.item_count <= 0 || fixture.knapsack_count <= 0) {
        return std::nullopt;
    }
    if (!NextDataLine(in, &line)) {
        return std::nullopt;
    }
    if (!ParseLineDoubles(line, fixture.item_count, &fixture.profits)) {
        return std::nullopt;
    }
    if (!NextDataLine(in, &line)) {
        return std::nullopt;
    }
    if (!ParseLineDoubles(line, fixture.knapsack_count, &fixture.capacities)) {
        return std::nullopt;
    }
    fixture.weights.resize(fixture.knapsack_count);
    for (int k = 0; k < fixture.knapsack_count; ++k) {
        if (!NextDataLine(in, &line)) {
            return std::nullopt;
        }
        if (!ParseLineDoubles(line, fixture.item_count, &fixture.weights[static_cast<std::size_t>(k)])) {
            return std::nullopt;
        }
    }
    return fixture;
}

std::filesystem::path make_temp_path() {
    auto dir = std::filesystem::temp_directory_path();
    std::random_device rd;
    std::mt19937_64 gen(rd());
    std::uniform_int_distribution<std::uint64_t> dist;
    for (int attempt = 0; attempt < 32; ++attempt) {
        auto candidate = dir / (
            "knapsack_attr_" + std::to_string(dist(gen)) + ".bin");
        if (!std::filesystem::exists(candidate)) {
            return candidate;
        }
    }
    return dir / "knapsack_attr_fallback.bin";
}

std::filesystem::path write_binary(const std::vector<double>& values) {
    auto path = make_temp_path();
    std::ofstream out(path, std::ios::binary);
    REQUIRE(out.good());
    out.write(reinterpret_cast<const char*>(values.data()),
              static_cast<std::streamsize>(values.size() * sizeof(double)));
    REQUIRE(out.good());
    out.close();
    return path;
}

std::filesystem::path write_csv(const std::vector<double>& values,
                                bool with_header = true,
                                const std::string& column = "value") {
    auto path = make_temp_path();
    std::ofstream out(path);
    REQUIRE(out.good());
    if (with_header) {
        out << column << "\n";
    }
    for (const auto& v : values) {
        out << v << "\n";
    }
    REQUIRE(out.good());
    out.close();
    return path;
}

#ifdef KNAPSACK_ARROW_ENABLED
std::shared_ptr<arrow::Array> build_arrow_array(const std::vector<double>& values) {
    arrow::DoubleBuilder builder;
    REQUIRE(builder.AppendValues(values).ok());
    std::shared_ptr<arrow::Array> array;
    REQUIRE(builder.Finish(&array).ok());
    return array;
}

std::shared_ptr<arrow::Table> build_arrow_table(const std::vector<double>& values,
                                                const std::string& column) {
    auto array = build_arrow_array(values);
    auto schema = arrow::schema({arrow::field(column, arrow::float64())});
    return arrow::Table::Make(schema, {array});
}

std::filesystem::path write_arrow_ipc(const std::shared_ptr<arrow::Table>& table);
std::filesystem::path write_parquet(const std::shared_ptr<arrow::Table>& table);

std::shared_ptr<arrow::Table> build_small_mkp_arrow_table(const SmallMKPFixture& fixture) {
    std::vector<std::shared_ptr<arrow::Field>> fields;
    std::vector<std::shared_ptr<arrow::Array>> columns;
    fields.push_back(arrow::field("value", arrow::float64()));
    columns.push_back(build_arrow_array(fixture.profits));
    for (int i = 0; i < fixture.knapsack_count; ++i) {
        auto column_name = std::string("weight_") + std::to_string(i);
        fields.push_back(arrow::field(column_name, arrow::float64()));
        columns.push_back(build_arrow_array(fixture.weights[static_cast<std::size_t>(i)]));
    }
    auto schema = arrow::schema(fields);
    return arrow::Table::Make(schema, columns);
}

std::filesystem::path write_arrow_ipc(const std::vector<double>& values,
                                      const std::string& column = "value") {
    auto table = build_arrow_table(values, column);
    return write_arrow_ipc(table);
}

std::filesystem::path write_arrow_ipc(const std::shared_ptr<arrow::Table>& table) {
    auto path = make_temp_path();
    path.replace_extension(".arrow");
    auto outfile_result = arrow::io::FileOutputStream::Open(path.string());
    REQUIRE(outfile_result.ok());
    auto outfile = *outfile_result;
    auto writer_result = arrow::ipc::MakeFileWriter(outfile, table->schema());
    REQUIRE(writer_result.ok());
    auto writer = std::move(writer_result).ValueOrDie();
    REQUIRE(writer->WriteTable(*table).ok());
    REQUIRE(writer->Close().ok());
    REQUIRE(outfile->Close().ok());
    return path;
}

std::filesystem::path write_parquet(const std::vector<double>& values,
                                    const std::string& column = "value") {
    auto table = build_arrow_table(values, column);
    return write_parquet(table);
}

std::filesystem::path write_parquet(const std::shared_ptr<arrow::Table>& table) {
    auto path = make_temp_path();
    path.replace_extension(".parquet");
    auto outfile_result = arrow::io::FileOutputStream::Open(path.string());
    REQUIRE(outfile_result.ok());
    auto outfile = *outfile_result;
    auto status = parquet::arrow::WriteTable(*table, arrow::default_memory_pool(), outfile, table->num_rows());
    REQUIRE(status.ok());
    REQUIRE(outfile->Close().ok());
    return path;
}
#endif

struct CinRedirect {
    explicit CinRedirect(std::streambuf* new_buf) : old_buf(std::cin.rdbuf(new_buf)) {}
    ~CinRedirect() { std::cin.rdbuf(old_buf); }
    std::streambuf* old_buf;
};

} // namespace

TEST_CASE("HostSoABuilder enforces counts", "[v2][data][builder]") {
    HostSoABuilder builder;
    std::string err;

    REQUIRE(builder.Begin(3, &err));
    REQUIRE(builder.AppendAttributeValues("value", std::vector<double>{1.0, 2.0, 3.0}, &err));

    SECTION("Rejects overflow") {
        REQUIRE_FALSE(builder.AppendAttributeValues("value", std::vector<double>{4.0}, &err));
        REQUIRE_FALSE(err.empty());
    }

    SECTION("Finishes when counts match") {
        REQUIRE(builder.AppendAttributeValues("weight", std::vector<double>{1.0, 1.0, 1.0}, &err));
        HostSoA soa;
        REQUIRE(builder.Finish(&soa, &err));
        REQUIRE(soa.count == 3);
        REQUIRE(soa.attr.at("value").size() == 3);
    }
}

TEST_CASE("BuildHostSoA loads binary file attributes", "[v2][data][file]") {
    Config cfg;
    cfg.items.count = 4;
    cfg.items.attributes["weight"] = {1.0, 2.0, 3.0, 4.0};

    std::vector<double> values = {10.0, 20.0, 30.0, 40.0};
    auto file_path = write_binary(values);

    AttributeSourceSpec spec;
    spec.kind = AttributeSourceKind::kFile;
    spec.path = file_path.string();
    spec.format = "binary64_le";
    cfg.items.sources["value"] = spec;

    HostSoA soa;
    std::string err;
    REQUIRE(BuildHostSoA(cfg, &soa, &err));
    REQUIRE(soa.attr.at("value") == values);

    std::filesystem::remove(file_path);
}

TEST_CASE("BuildHostSoA loads CSV attributes", "[v2][data][csv]") {
    Config cfg;
    cfg.items.count = 3;
    cfg.items.attributes["weight"] = {1.0, 2.0, 3.0};

    std::vector<double> values = {4.0, 5.0, 6.0};
    auto csv_path = write_csv(values, true, "value");

    AttributeSourceSpec spec;
    spec.kind = AttributeSourceKind::kFile;
    spec.format = "csv";
    spec.format_kind = AttributeFormatKind::kCSV;
    spec.path = csv_path.string();
    spec.csv_has_header = true;
    spec.column_name = "value";
    cfg.items.sources["value"] = spec;

    HostSoA soa;
    std::string err;
    REQUIRE(BuildHostSoA(cfg, &soa, &err));
    REQUIRE(soa.attr.at("value") == values);

    std::filesystem::remove(csv_path);
}

TEST_CASE("BuildHostSoA consumes stream channels", "[v2][data][stream]") {
    Config cfg;
    cfg.items.count = 3;

    std::vector<double> inline_attr = {5.0, 6.0, 7.0};
    cfg.items.attributes["value"] = inline_attr;

    std::vector<double> streamed = {9.0, 8.0, 7.0};
    std::string blob(reinterpret_cast<const char*>(streamed.data()),
                     streamed.size() * sizeof(double));
    std::istringstream fake(blob);
    CinRedirect redirect(fake.rdbuf());

    AttributeSourceSpec spec;
    spec.kind = AttributeSourceKind::kStream;
    spec.channel = "stdin";
    spec.format = "binary64_le";
    cfg.items.sources["weight"] = spec;

    HostSoA soa;
    std::string err;
    REQUIRE(BuildHostSoA(cfg, &soa, &err));
    REQUIRE(soa.attr.at("weight") == streamed);
    REQUIRE(soa.attr.at("value") == inline_attr);
}

#ifdef KNAPSACK_ARROW_ENABLED

TEST_CASE("BuildHostSoA loads Arrow IPC attributes", "[v2][data][arrow]") {
    Config cfg;
    cfg.items.count = 3;
    cfg.items.attributes["weight"] = {1.0, 2.0, 3.0};

    std::vector<double> values = {7.0, 8.0, 9.0};
    auto arrow_path = write_arrow_ipc(values, "value");

    AttributeSourceSpec spec;
    spec.kind = AttributeSourceKind::kFile;
    spec.format = "arrow";
    spec.format_kind = AttributeFormatKind::kArrow;
    spec.path = arrow_path.string();
    spec.column_name = "value";
    cfg.items.sources["value"] = spec;

    HostSoA soa;
    std::string err;
    REQUIRE(BuildHostSoA(cfg, &soa, &err));
    REQUIRE(soa.attr.at("value") == values);

    std::filesystem::remove(arrow_path);
}

TEST_CASE("BuildHostSoA loads Parquet attributes", "[v2][data][parquet]") {
    Config cfg;
    cfg.items.count = 4;
    cfg.items.attributes["weight"] = {1.0, 1.0, 1.0, 1.0};

    std::vector<double> values = {2.0, 4.0, 6.0, 8.0};
    auto parquet_path = write_parquet(values, "metric");

    AttributeSourceSpec spec;
    spec.kind = AttributeSourceKind::kFile;
    spec.format = "parquet";
    spec.format_kind = AttributeFormatKind::kParquet;
    spec.path = parquet_path.string();
    spec.column_name = "metric";
    cfg.items.sources["metric"] = spec;

    HostSoA soa;
    std::string err;
    REQUIRE(BuildHostSoA(cfg, &soa, &err));
    REQUIRE(soa.attr.at("metric") == values);

    std::filesystem::remove(parquet_path);
}

TEST_CASE("BuildHostSoA ingests Arrow MKP fixture when available", "[v2][data][arrow][fixture]") {
    auto fixture = LoadSmallMKPFixture();
    if (!fixture) {
        WARN("Small MKP fixture not found; set KNAPSACK_TESTDATA_ROOT to enable Arrow fixture coverage");
        return;
    }
    auto table = build_small_mkp_arrow_table(*fixture);
    auto arrow_path = write_arrow_ipc(table);
    Config cfg = BuildConfigForFixture(*fixture, arrow_path, "arrow", AttributeFormatKind::kArrow);

    HostSoA soa;
    std::string err;
    CAPTURE(fixture->source_path.string());
    REQUIRE(BuildHostSoA(cfg, &soa, &err));
    REQUIRE(soa.count == fixture->item_count);
    REQUIRE(soa.attr.at("value") == fixture->profits);
    for (int i = 0; i < fixture->knapsack_count; ++i) {
        auto column_name = std::string("weight_") + std::to_string(i);
        REQUIRE(soa.attr.at(column_name) == fixture->weights[static_cast<std::size_t>(i)]);
    }

    std::filesystem::remove(arrow_path);
}

TEST_CASE("BuildHostSoA ingests Parquet MKP fixture when available", "[v2][data][parquet][fixture]") {
    auto fixture = LoadSmallMKPFixture();
    if (!fixture) {
        WARN("Small MKP fixture not found; set KNAPSACK_TESTDATA_ROOT to enable Parquet fixture coverage");
        return;
    }
    auto table = build_small_mkp_arrow_table(*fixture);
    auto parquet_path = write_parquet(table);
    Config cfg = BuildConfigForFixture(*fixture, parquet_path, "parquet", AttributeFormatKind::kParquet);

    HostSoA soa;
    std::string err;
    CAPTURE(fixture->source_path.string());
    REQUIRE(BuildHostSoA(cfg, &soa, &err));
    REQUIRE(soa.count == fixture->item_count);
    REQUIRE(soa.attr.at("value") == fixture->profits);
    for (int i = 0; i < fixture->knapsack_count; ++i) {
        auto column_name = std::string("weight_") + std::to_string(i);
        REQUIRE(soa.attr.at(column_name) == fixture->weights[static_cast<std::size_t>(i)]);
    }

    std::filesystem::remove(parquet_path);
}

#endif
