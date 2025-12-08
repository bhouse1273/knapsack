// Copyright (c) 2025
// SPDX-License-Identifier: MIT

#include "v2/Data.h"

#include <algorithm>
#include <limits>
#include <sstream>

#include "v2/DataFormats.h"

namespace {

bool set_error(std::string* err, const std::string& msg) {
  if (err) *err = msg;
  return false;
}

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
    if (!LoadAttributeFromSource(name, spec, cfg.items.count, &builder, err)) return false;
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
