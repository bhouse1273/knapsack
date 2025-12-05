// Copyright (c) 2025
// SPDX-License-Identifier: MIT

#include "v2/Data.h"

#include <algorithm>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>

namespace {

bool set_error(std::string* err, const std::string& msg) {
  if (err) *err = msg;
  return false;
}

constexpr std::size_t kBinaryChunk = 4096;

} // namespace

namespace v2 {

bool HostSoABuilder::Begin(int expected_count, std::string* err) {
  if (started_) {
    return set_error(err, "HostSoABuilder already started");
  }
  if (expected_count < 0) {
    return set_error(err, "expected_count must be >= 0");
  }
  expected_count_ = expected_count;
  started_ = true;
  finished_ = false;
  row_mode_ = false;
  rows_appended_ = 0;
  row_order_.clear();
  buffers_.clear();
  return true;
}

bool HostSoABuilder::DeclareAttribute(const std::string& name, std::string* err) {
  if (!EnsureStarted(err)) return false;
  if (finished_) return set_error(err, "builder already finished");
  if (buffers_.count(name)) {
    return set_error(err, "attribute '" + name + "' already declared");
  }
  buffers_[name] = std::vector<double>();
  buffers_[name].reserve(expected_count_);
  row_order_.push_back(name);
  return true;
}

bool HostSoABuilder::DeclareAttributes(const std::vector<std::string>& names, std::string* err) {
  for (const auto& name : names) {
    if (!DeclareAttribute(name, err)) return false;
  }
  return true;
}

bool HostSoABuilder::AppendAttributeValues(const std::string& name,
                                           const std::vector<double>& values,
                                           std::string* err) {
  return AppendAttributeValues(name, values.data(), values.size(), err);
}

bool HostSoABuilder::AppendAttributeValues(const std::string& name,
                                           const double* values,
                                           std::size_t count,
                                           std::string* err) {
  if (!EnsureStarted(err)) return false;
  if (finished_) return set_error(err, "builder already finished");
  if (row_mode_) return set_error(err, "cannot append column chunks after row ingestion started");
  if (count == 0) return true;
  if (!values) return set_error(err, "values pointer is null");
  if (!EnsureAttributeBuffer(name, err)) return false;
  auto& buf = buffers_[name];
  if (!CheckRemaining(buf.size(), count, err)) return false;
  buf.insert(buf.end(), values, values + count);
  return true;
}

bool HostSoABuilder::AppendRow(const std::vector<double>& values, std::string* err) {
  if (!EnsureStarted(err)) return false;
  if (finished_) return set_error(err, "builder already finished");
  if (row_order_.empty()) {
    return set_error(err, "row ingestion requires DeclareAttribute(s) first");
  }
  if (values.size() != row_order_.size()) {
    return set_error(err, "row column count mismatch");
  }
  if (!CheckRemaining(rows_appended_, 1, err)) return false;
  row_mode_ = true;
  for (std::size_t i = 0; i < values.size(); ++i) {
    const auto& attr = row_order_[i];
    if (!EnsureAttributeBuffer(attr, err)) return false;
    buffers_[attr].push_back(values[i]);
  }
  ++rows_appended_;
  return true;
}

bool HostSoABuilder::Finish(HostSoA* out, std::string* err) {
  if (!EnsureStarted(err)) return false;
  if (finished_) return set_error(err, "builder already finished");
  if (!out) return set_error(err, "out is null");
  for (auto& kv : buffers_) {
    if (kv.second.size() != static_cast<std::size_t>(expected_count_)) {
      std::ostringstream oss;
      oss << "attribute '" << kv.first << "' received " << kv.second.size()
          << " values but expected " << expected_count_;
      return set_error(err, oss.str());
    }
  }
  HostSoA soa;
  soa.count = expected_count_;
  soa.attr = buffers_;
  *out = std::move(soa);
  finished_ = true;
  return true;
}

bool HostSoABuilder::EnsureStarted(std::string* err) const {
  if (!started_) {
    return set_error(err, "builder not started");
  }
  return true;
}

bool HostSoABuilder::EnsureAttributeBuffer(const std::string& name, std::string* err) {
  if (!buffers_.count(name)) {
    buffers_[name] = std::vector<double>();
    buffers_[name].reserve(expected_count_);
  }
  return true;
}

bool HostSoABuilder::CheckRemaining(std::size_t current, std::size_t incoming, std::string* err) const {
  if (incoming == 0) return true;
  std::size_t expected = static_cast<std::size_t>(expected_count_);
  if (current > expected || incoming > expected) {
    return set_error(err, "attribute length overflow");
  }
  if (current + incoming > expected) {
    return set_error(err, "attribute received more values than items.count");
  }
  return true;
}

static bool ReadBinaryChunks(std::istream& in,
                             const std::string& attr_name,
                             int expected_count,
                             HostSoABuilder* builder,
                             std::string* err) {
  if (expected_count <= 0) return set_error(err, "items.count must be positive");
  std::vector<double> chunk(kBinaryChunk);
  int remaining = expected_count;
  while (remaining > 0) {
    std::size_t want = std::min<std::size_t>(chunk.size(), static_cast<std::size_t>(remaining));
    in.read(reinterpret_cast<char*>(chunk.data()), want * sizeof(double));
    std::streamsize bytes = in.gcount();
    if (bytes <= 0) {
      return set_error(err, "attribute '" + attr_name + "' stream ended before items.count reached");
    }
    if (bytes % static_cast<std::streamsize>(sizeof(double)) != 0) {
      return set_error(err, "attribute '" + attr_name + "' stream length not aligned to double");
    }
    std::size_t got = static_cast<std::size_t>(bytes) / sizeof(double);
    if (!builder->AppendAttributeValues(attr_name, chunk.data(), got, err)) {
      return false;
    }
    remaining -= static_cast<int>(got);
    if (got < want && remaining > 0) {
      return set_error(err, "attribute '" + attr_name + "' stream exhausted early");
    }
  }
  return true;
}

static std::vector<std::string> BuildFileList(const AttributeSourceSpec& spec) {
  std::vector<std::string> files;
  if (!spec.path.empty()) files.push_back(spec.path);
  files.insert(files.end(), spec.chunks.begin(), spec.chunks.end());
  return files;
}

static bool LoadAttributeFromFiles(const std::string& attr_name,
                                   const AttributeSourceSpec& spec,
                                   int expected_count,
                                   HostSoABuilder* builder,
                                   std::string* err) {
  auto files = BuildFileList(spec);
  if (files.empty()) {
    return set_error(err, "attribute '" + attr_name + "' missing file paths");
  }
  int remaining = expected_count;
  bool applied_offset = false;
  std::vector<double> chunk(kBinaryChunk);
  for (std::size_t idx = 0; idx < files.size() && remaining > 0; ++idx) {
    std::ifstream in(files[idx], std::ios::binary);
    if (!in) {
      return set_error(err, "failed to open attribute file '" + files[idx] + "'");
    }
    if (!applied_offset && spec.offset_bytes > 0) {
      in.seekg(static_cast<std::streamoff>(spec.offset_bytes), std::ios::beg);
      if (!in) {
        return set_error(err, "offset_bytes past end of file for attribute '" + attr_name + "'");
      }
    }
    applied_offset = true;
    while (remaining > 0 && in) {
      std::size_t want = std::min<std::size_t>(chunk.size(), static_cast<std::size_t>(remaining));
      in.read(reinterpret_cast<char*>(chunk.data()), want * sizeof(double));
      std::streamsize bytes = in.gcount();
      if (bytes <= 0) break;
      if (bytes % static_cast<std::streamsize>(sizeof(double)) != 0) {
        return set_error(err, "attribute '" + attr_name + "' file bytes not aligned to double");
      }
      std::size_t got = static_cast<std::size_t>(bytes) / sizeof(double);
      if (!builder->AppendAttributeValues(attr_name, chunk.data(), got, err)) return false;
      remaining -= static_cast<int>(got);
      if (got < want) break; // reached EOF of this file
    }
  }
  if (remaining != 0) {
    return set_error(err, "attribute '" + attr_name + "' file data ended before items.count reached");
  }
  return true;
}

static bool LoadAttributeFromStream(const std::string& attr_name,
                                    const AttributeSourceSpec& spec,
                                    int expected_count,
                                    HostSoABuilder* builder,
                                    std::string* err) {
  if (!spec.channel.empty()) {
    if (spec.channel == "stdin" || spec.channel == "STDIN") {
      return ReadBinaryChunks(std::cin, attr_name, expected_count, builder, err);
    }
    std::string path = spec.channel;
    const std::string file_prefix = "file://";
    if (spec.channel.rfind(file_prefix, 0) == 0 && spec.channel.size() > file_prefix.size()) {
      path = spec.channel.substr(file_prefix.size());
    }
    std::ifstream in(path, std::ios::binary);
    if (!in) {
      return set_error(err, "failed to open stream channel '" + spec.channel + "'");
    }
    return ReadBinaryChunks(in, attr_name, expected_count, builder, err);
  }
  if (!spec.chunks.empty()) {
    int remaining = expected_count;
    std::vector<double> chunk(kBinaryChunk);
    for (const auto& path : spec.chunks) {
      if (remaining == 0) break;
      std::ifstream in(path, std::ios::binary);
      if (!in) return set_error(err, "failed to open stream chunk '" + path + "'");
      while (remaining > 0) {
        std::size_t want = std::min<std::size_t>(chunk.size(), static_cast<std::size_t>(remaining));
        in.read(reinterpret_cast<char*>(chunk.data()), want * sizeof(double));
        std::streamsize bytes = in.gcount();
        if (bytes <= 0) break;
        if (bytes % static_cast<std::streamsize>(sizeof(double)) != 0) {
          return set_error(err, "attribute '" + attr_name + "' stream chunk misaligned");
        }
        std::size_t got = static_cast<std::size_t>(bytes) / sizeof(double);
        if (!builder->AppendAttributeValues(attr_name, chunk.data(), got, err)) return false;
        remaining -= static_cast<int>(got);
        if (got < want) break;
      }
    }
    if (remaining != 0) {
      return set_error(err, "attribute '" + attr_name + "' stream chunks ended early");
    }
    return true;
  }
  return set_error(err, "attribute '" + attr_name + "' stream requires channel or chunks");
}

bool BuildHostSoA(const Config& cfg, HostSoA* out, std::string* err) {
  if (!out) { if (err) *err = "out is null"; return false; }
  HostSoABuilder builder;
  if (!builder.Begin(cfg.items.count, err)) return false;

  for (const auto& kv : cfg.items.attributes) {
    if (!builder.AppendAttributeValues(kv.first, kv.second, err)) return false;
  }
  for (const auto& kv : cfg.items.sources) {
    const auto& name = kv.first;
    const auto& spec = kv.second;
    if (spec.format != "binary64_le") {
      return set_error(err, "attribute '" + name + "' format not supported");
    }
    if (spec.is_file()) {
      if (!LoadAttributeFromFiles(name, spec, cfg.items.count, &builder, err)) return false;
    } else if (spec.is_stream()) {
      if (!LoadAttributeFromStream(name, spec, cfg.items.count, &builder, err)) return false;
    }
  }

  return builder.Finish(out, err);
}

std::vector<BlockSlice> BuildBlockSlices(const Config& cfg) {
  std::vector<BlockSlice> out;
  out.reserve(cfg.blocks.size());
  for (const auto& b : cfg.blocks) {
    BlockSlice bs;
    bs.name = b.name;
    if (b.start >= 0 && b.count > 0) {
      bs.indices.resize(b.count);
      for (int i = 0; i < b.count; ++i) bs.indices[i] = b.start + i;
    } else if (!b.indices.empty()) {
      bs.indices = b.indices;
    } else {
      // Should not occur if ValidateConfig was called; leave empty
    }
    out.push_back(std::move(bs));
  }
  return out;
}

} // namespace v2
