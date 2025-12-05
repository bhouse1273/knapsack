#include "third_party/catch2/catch_amalgamated.hpp"

#include "v2/Config.h"
#include "v2/Data.h"

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>

using namespace v2;

namespace {

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
