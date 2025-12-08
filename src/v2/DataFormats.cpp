// Copyright (c) 2025
// SPDX-License-Identifier: MIT

#include "v2/DataFormats.h"

#include <algorithm>
#include <cctype>
#include <cstddef>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <type_traits>
#include <vector>

#ifdef KNAPSACK_ARROW_ENABLED
#include <arrow/api.h>
#include <arrow/io/api.h>
#include <arrow/ipc/api.h>
#include <parquet/arrow/reader.h>
#endif

namespace {

bool set_error(std::string* err, const std::string& msg) {
  if (err) *err = msg;
  return false;
}

constexpr std::size_t kBinaryChunk = 4096;

std::vector<std::string> BuildFileList(const v2::AttributeSourceSpec& spec) {
  std::vector<std::string> files;
  if (!spec.path.empty()) files.push_back(spec.path);
  files.insert(files.end(), spec.chunks.begin(), spec.chunks.end());
  return files;
}

bool AppendBinaryChunk(v2::HostSoABuilder* builder,
                       const std::string& attr_name,
                       const double* data,
                       std::size_t count,
                       std::string* err) {
  if (count == 0) return true;
  return builder->AppendAttributeValues(attr_name, data, count, err);
}

bool ReadBinaryStream(std::istream& in,
                      const std::string& attr_name,
                      int expected_count,
                      v2::HostSoABuilder* builder,
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
    if (!AppendBinaryChunk(builder, attr_name, chunk.data(), got, err)) return false;
    remaining -= static_cast<int>(got);
    if (got < want && remaining > 0) {
      return set_error(err, "attribute '" + attr_name + "' stream exhausted early");
    }
  }
  return true;
}

bool LoadBinaryFromFiles(const std::string& attr_name,
                         const v2::AttributeSourceSpec& spec,
                         int expected_count,
                         v2::HostSoABuilder* builder,
                         std::string* err) {
  auto files = BuildFileList(spec);
  if (files.empty()) {
    return set_error(err, "attribute '" + attr_name + "' missing file paths");
  }
  int remaining = expected_count;
  std::vector<double> chunk(kBinaryChunk);
  bool applied_offset = false;
  for (const auto& path : files) {
    if (remaining <= 0) break;
    std::ifstream in(path, std::ios::binary);
    if (!in) {
      return set_error(err, "failed to open attribute file '" + path + "'");
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
      if (!AppendBinaryChunk(builder, attr_name, chunk.data(), got, err)) return false;
      remaining -= static_cast<int>(got);
      if (got < want) break;
    }
  }
  if (remaining != 0) {
    return set_error(err, "attribute '" + attr_name + "' file data ended before items.count reached");
  }
  return true;
}

bool LoadBinaryFromChannel(const std::string& attr_name,
                           const v2::AttributeSourceSpec& spec,
                           int expected_count,
                           v2::HostSoABuilder* builder,
                           std::string* err) {
  if (!spec.channel.empty()) {
    if (spec.channel == "stdin" || spec.channel == "STDIN") {
      return ReadBinaryStream(std::cin, attr_name, expected_count, builder, err);
    }
    const std::string file_prefix = "file://";
    std::string path = spec.channel;
    if (spec.channel.rfind(file_prefix, 0) == 0 && spec.channel.size() > file_prefix.size()) {
      path = spec.channel.substr(file_prefix.size());
    }
    std::ifstream in(path, std::ios::binary);
    if (!in) {
      return set_error(err, "failed to open stream channel '" + spec.channel + "'");
    }
    return ReadBinaryStream(in, attr_name, expected_count, builder, err);
  }
  auto files = BuildFileList(spec);
  if (files.empty()) {
    return set_error(err, "attribute '" + attr_name + "' stream requires channel or chunks");
  }
  int remaining = expected_count;
  std::vector<double> chunk(kBinaryChunk);
  for (const auto& path : files) {
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
      if (!AppendBinaryChunk(builder, attr_name, chunk.data(), got, err)) return false;
      remaining -= static_cast<int>(got);
      if (got < want) break;
    }
  }
  if (remaining != 0) {
    return set_error(err, "attribute '" + attr_name + "' stream chunks ended early");
  }
  return true;
}

void SplitCSV(const std::string& line, char delimiter, std::vector<std::string>* out) {
  out->clear();
  std::string field;
  field.reserve(16);
  bool in_quotes = false;
  for (size_t i = 0; i < line.size(); ++i) {
    char c = line[i];
    if (c == '"') {
      if (in_quotes && i + 1 < line.size() && line[i + 1] == '"') {
        field.push_back('"');
        ++i;
      } else {
        in_quotes = !in_quotes;
      }
    } else if (c == delimiter && !in_quotes) {
      out->push_back(field);
      field.clear();
    } else {
      field.push_back(c);
    }
  }
  out->push_back(field);
}

bool ParseDouble(const std::string& token, double* value) {
  try {
    size_t idx = 0;
    *value = std::stod(token, &idx);
    if (idx != token.size()) {
      auto rest = token.substr(idx);
      if (std::all_of(rest.begin(), rest.end(), [](char c) { return std::isspace(static_cast<unsigned char>(c)); })) {
        return true;
      }
      return false;
    }
    return true;
  } catch (...) {
    return false;
  }
}

bool LoadCSVFromStream(const std::string& attr_name,
                       const v2::AttributeSourceSpec& spec,
                       int expected_count,
                       std::istream& in,
                       v2::HostSoABuilder* builder,
                       std::string* err) {
  if (expected_count <= 0) return set_error(err, "items.count must be positive");
  std::vector<std::string> fields;
  fields.reserve(8);
  bool need_header = spec.csv_has_header || !spec.column_name.empty();
  bool header_processed = !need_header;
  bool skip_header_row = spec.csv_has_header || !spec.column_name.empty();
  int column_index = spec.column_index;
  int values_appended = 0;
  std::string line;
  std::string column_name_lower = spec.column_name;
  std::transform(column_name_lower.begin(), column_name_lower.end(), column_name_lower.begin(), [](unsigned char c) { return std::tolower(c); });
  while (std::getline(in, line)) {
    // Ignore empty lines
    bool only_ws = std::all_of(line.begin(), line.end(), [](char c) { return std::isspace(static_cast<unsigned char>(c)); });
    if (only_ws) continue;
    SplitCSV(line, spec.csv_delimiter, &fields);
    if (!header_processed) {
      if (!fields.empty() && !spec.column_name.empty()) {
        auto it = std::find_if(fields.begin(), fields.end(), [&](const std::string& col) {
          std::string lower = col;
          std::transform(lower.begin(), lower.end(), lower.begin(), [](unsigned char c) { return std::tolower(c); });
          return lower == column_name_lower;
        });
        if (it == fields.end()) {
          return set_error(err, "attribute '" + attr_name + "' column '" + spec.column_name + "' not found in header");
        }
        column_index = static_cast<int>(std::distance(fields.begin(), it));
      }
      if (column_index < 0) column_index = 0;
      header_processed = true;
      if (skip_header_row) continue;
    }
    if (column_index < 0) column_index = 0;
    if (column_index >= static_cast<int>(fields.size())) {
      return set_error(err, "attribute '" + attr_name + "' column index out of range in CSV row");
    }
    double value = 0.0;
    if (!ParseDouble(fields[column_index], &value)) {
      return set_error(err, "attribute '" + attr_name + "' CSV value parse failure");
    }
    if (!builder->AppendAttributeValues(attr_name, &value, 1, err)) {
      return false;
    }
    ++values_appended;
    if (values_appended == expected_count) break;
  }
  if (values_appended != expected_count) {
    return set_error(err, "attribute '" + attr_name + "' CSV rows did not match items.count");
  }
  return true;
}

bool LoadCSVFromFiles(const std::string& attr_name,
                      const v2::AttributeSourceSpec& spec,
                      int expected_count,
                      v2::HostSoABuilder* builder,
                      std::string* err) {
  auto files = BuildFileList(spec);
  if (files.size() != 1) {
    return set_error(err, "attribute '" + attr_name + "' CSV format expects exactly one file");
  }
  std::ifstream in(files[0]);
  if (!in) return set_error(err, "failed to open CSV file '" + files[0] + "'");
  return LoadCSVFromStream(attr_name, spec, expected_count, in, builder, err);
}

bool LoadCSVFromChannel(const std::string& attr_name,
                        const v2::AttributeSourceSpec& spec,
                        int expected_count,
                        v2::HostSoABuilder* builder,
                        std::string* err) {
  if (spec.channel == "stdin" || spec.channel == "STDIN") {
    return LoadCSVFromStream(attr_name, spec, expected_count, std::cin, builder, err);
  }
  const std::string file_prefix = "file://";
  if (!spec.channel.empty()) {
    std::string path = spec.channel;
    if (spec.channel.rfind(file_prefix, 0) == 0 && spec.channel.size() > file_prefix.size()) {
      path = spec.channel.substr(file_prefix.size());
    }
    std::ifstream in(path);
    if (!in) {
      return set_error(err, "failed to open CSV stream '" + spec.channel + "'");
    }
    return LoadCSVFromStream(attr_name, spec, expected_count, in, builder, err);
  }
  return set_error(err, "attribute '" + attr_name + "' CSV stream missing channel");
}

#ifdef KNAPSACK_ARROW_ENABLED

bool ArrowStatusOk(const arrow::Status& status,
                   const std::string& context,
                   std::string* err) {
  if (status.ok()) return true;
  return set_error(err, context + ": " + status.ToString());
}

template <typename T>
bool AssignArrowResult(arrow::Result<T>&& result,
                       T* out,
                       const std::string& context,
                       std::string* err) {
  if (!result.ok()) {
    return set_error(err, context + ": " + result.status().ToString());
  }
  *out = std::move(result).ValueOrDie();
  return true;
}

bool ResolveArrowColumnIndex(const std::string& attr_name,
                             const v2::AttributeSourceSpec& spec,
                             const std::shared_ptr<arrow::Schema>& schema,
                             int* out_index,
                             std::string* err) {
  if (!schema) {
    return set_error(err, "attribute '" + attr_name + "' Arrow schema missing");
  }
  if (!spec.column_name.empty()) {
    int idx = schema->GetFieldIndex(spec.column_name);
    if (idx < 0) {
      return set_error(err, "attribute '" + attr_name + "' column '" + spec.column_name + "' not found in Arrow schema");
    }
    *out_index = idx;
    return true;
  }
  if (spec.column_index >= 0) {
    if (spec.column_index >= schema->num_fields()) {
      return set_error(err, "attribute '" + attr_name + "' column_index out of range for Arrow schema");
    }
    *out_index = spec.column_index;
    return true;
  }
  if (schema->num_fields() <= 0) {
    return set_error(err, "attribute '" + attr_name + "' Arrow table has no columns");
  }
  *out_index = 0;
  return true;
}

template <typename ArrowType>
bool AppendNumericArrowChunk(const arrow::NumericArray<ArrowType>& array,
                             const std::string& attr_name,
                             v2::HostSoABuilder* builder,
                             std::string* err) {
  if (array.null_count() > 0) {
    return set_error(err, "attribute '" + attr_name + "' Arrow column contains null values");
  }
  std::size_t len = static_cast<std::size_t>(array.length());
  if constexpr (std::is_same<ArrowType, arrow::DoubleType>::value) {
    return builder->AppendAttributeValues(attr_name, array.raw_values(), len, err);
  } else {
    std::vector<double> values(len);
    for (int64_t i = 0; i < array.length(); ++i) {
      values[static_cast<std::size_t>(i)] = static_cast<double>(array.Value(i));
    }
    return builder->AppendAttributeValues(attr_name, values.data(), values.size(), err);
  }
}

bool AppendArrowArray(const std::shared_ptr<arrow::Array>& array,
                      const std::string& attr_name,
                      v2::HostSoABuilder* builder,
                      std::string* err) {
  if (!array) return true;
  switch (array->type_id()) {
    case arrow::Type::DOUBLE:
      return AppendNumericArrowChunk(*std::static_pointer_cast<arrow::DoubleArray>(array), attr_name, builder, err);
    case arrow::Type::FLOAT:
      return AppendNumericArrowChunk(*std::static_pointer_cast<arrow::FloatArray>(array), attr_name, builder, err);
    case arrow::Type::INT64:
      return AppendNumericArrowChunk(*std::static_pointer_cast<arrow::Int64Array>(array), attr_name, builder, err);
    case arrow::Type::INT32:
      return AppendNumericArrowChunk(*std::static_pointer_cast<arrow::Int32Array>(array), attr_name, builder, err);
    case arrow::Type::UINT64:
      return AppendNumericArrowChunk(*std::static_pointer_cast<arrow::UInt64Array>(array), attr_name, builder, err);
    case arrow::Type::UINT32:
      return AppendNumericArrowChunk(*std::static_pointer_cast<arrow::UInt32Array>(array), attr_name, builder, err);
    default:
      return set_error(err, "attribute '" + attr_name + "' Arrow column type '" + array->type()->ToString() + "' not supported");
  }
}

bool AppendArrowColumn(const std::shared_ptr<arrow::ChunkedArray>& column,
                       const std::string& attr_name,
                       v2::HostSoABuilder* builder,
                       std::string* err) {
  if (!column) {
    return set_error(err, "attribute '" + attr_name + "' Arrow column missing");
  }
  for (const auto& chunk : column->chunks()) {
    if (!AppendArrowArray(chunk, attr_name, builder, err)) return false;
  }
  return true;
}

bool LoadArrowTableIntoBuilder(const std::string& attr_name,
                               const v2::AttributeSourceSpec& spec,
                               const std::shared_ptr<arrow::Table>& table,
                               v2::HostSoABuilder* builder,
                               std::string* err) {
  if (!table) return true;
  int column_index = 0;
  if (!ResolveArrowColumnIndex(attr_name, spec, table->schema(), &column_index, err)) return false;
  auto column = table->column(column_index);
  if (!column) {
    return set_error(err, "attribute '" + attr_name + "' Arrow column index missing in table");
  }
  return AppendArrowColumn(column, attr_name, builder, err);
}

bool ReadArrowTableFromStream(const std::shared_ptr<arrow::ipc::RecordBatchReader>& reader,
                              std::shared_ptr<arrow::Table>* table,
                              std::string* err) {
  if (!reader) {
    return set_error(err, "Arrow record batch reader is null");
  }
  std::vector<std::shared_ptr<arrow::RecordBatch>> batches;
  while (true) {
    auto next = reader->Next();
    if (!next.ok()) {
      return set_error(err, "Arrow stream read failed: " + next.status().ToString());
    }
    auto batch = next.ValueOrDie();
    if (!batch) break;
    batches.push_back(batch);
  }
  return AssignArrowResult(arrow::Table::FromRecordBatches(reader->schema(), batches), table, "Arrow stream to table", err);
}

bool ReadArrowTableFromFileReader(const std::shared_ptr<arrow::ipc::RecordBatchFileReader>& reader,
                                  std::shared_ptr<arrow::Table>* table,
                                  std::string* err) {
  if (!reader) {
    return set_error(err, "Arrow file reader is null");
  }
  std::vector<std::shared_ptr<arrow::RecordBatch>> batches;
  int num_batches = reader->num_record_batches();
  batches.reserve(std::max(0, num_batches));
  for (int i = 0; i < num_batches; ++i) {
    auto batch_result = reader->ReadRecordBatch(i);
    if (!batch_result.ok()) {
      return set_error(err, "Arrow file read failed: " + batch_result.status().ToString());
    }
    batches.push_back(batch_result.ValueOrDie());
  }
  return AssignArrowResult(arrow::Table::FromRecordBatches(reader->schema(), batches), table, "Arrow file to table", err);
}

bool ReadArrowTableFromFile(const std::string& path,
                            std::shared_ptr<arrow::Table>* table,
                            std::string* err) {
  std::shared_ptr<arrow::io::ReadableFile> input;
  if (!AssignArrowResult(arrow::io::ReadableFile::Open(path), &input, "open Arrow file '" + path + "'", err)) {
    return false;
  }
  auto file_reader_result = arrow::ipc::RecordBatchFileReader::Open(input);
  if (file_reader_result.ok()) {
    auto reader = std::move(file_reader_result).ValueOrDie();
    return ReadArrowTableFromFileReader(reader, table, err);
  }
  if (!ArrowStatusOk(input->Seek(0), "rewind Arrow file '" + path + "'", err)) {
    return false;
  }
  auto stream_reader_result = arrow::ipc::RecordBatchStreamReader::Open(input);
  if (!stream_reader_result.ok()) {
    return set_error(err, "failed to interpret Arrow file '" + path + "': file reader error = " + file_reader_result.status().ToString() + ", stream reader error = " + stream_reader_result.status().ToString());
  }
  auto stream_reader = std::move(stream_reader_result).ValueOrDie();
  return ReadArrowTableFromStream(stream_reader, table, err);
}

bool LoadArrowFromFiles(const std::string& attr_name,
                        const v2::AttributeSourceSpec& spec,
                        v2::HostSoABuilder* builder,
                        std::string* err) {
  auto files = BuildFileList(spec);
  if (files.empty()) {
    return set_error(err, "attribute '" + attr_name + "' Arrow format requires path or chunks");
  }
  for (const auto& path : files) {
    std::shared_ptr<arrow::Table> table;
    if (!ReadArrowTableFromFile(path, &table, err)) return false;
    if (!LoadArrowTableIntoBuilder(attr_name, spec, table, builder, err)) return false;
  }
  return true;
}

bool LoadParquetFromFiles(const std::string& attr_name,
                          const v2::AttributeSourceSpec& spec,
                          v2::HostSoABuilder* builder,
                          std::string* err) {
  auto files = BuildFileList(spec);
  if (files.empty()) {
    return set_error(err, "attribute '" + attr_name + "' Parquet format requires path or chunks");
  }
  for (const auto& path : files) {
    std::shared_ptr<arrow::io::ReadableFile> input;
    if (!AssignArrowResult(arrow::io::ReadableFile::Open(path), &input, "open Parquet file '" + path + "'", err)) {
      return false;
    }
    auto reader_result = parquet::arrow::OpenFile(input, arrow::default_memory_pool());
    if (!reader_result.ok()) {
      return set_error(err, "read Parquet file '" + path + "': " + reader_result.status().ToString());
    }
    std::unique_ptr<parquet::arrow::FileReader> reader = std::move(reader_result).ValueOrDie();
    if (!reader) {
      return set_error(err, "Parquet reader for '" + path + "' is null");
    }
    std::shared_ptr<arrow::Table> table;
    if (!ArrowStatusOk(reader->ReadTable(&table), "convert Parquet file '" + path + "'", err)) {
      return false;
    }
    if (!LoadArrowTableIntoBuilder(attr_name, spec, table, builder, err)) return false;
  }
  return true;
}

#endif // KNAPSACK_ARROW_ENABLED

} // namespace

namespace v2 {

bool LoadAttributeFromSource(const std::string& attr_name,
                             const AttributeSourceSpec& spec,
                             int expected_count,
                             HostSoABuilder* builder,
                             std::string* err) {
  switch (spec.format_kind) {
    case AttributeFormatKind::kBinary64LE:
      if (spec.is_file()) return LoadBinaryFromFiles(attr_name, spec, expected_count, builder, err);
      if (spec.is_stream()) return LoadBinaryFromChannel(attr_name, spec, expected_count, builder, err);
      return set_error(err, "attribute '" + attr_name + "' has unsupported source kind for binary format");
    case AttributeFormatKind::kCSV:
      if (spec.csv_delimiter == '\0') return set_error(err, "attribute '" + attr_name + "' CSV delimiter invalid");
      if (spec.is_file()) return LoadCSVFromFiles(attr_name, spec, expected_count, builder, err);
      if (spec.is_stream()) return LoadCSVFromChannel(attr_name, spec, expected_count, builder, err);
      return set_error(err, "attribute '" + attr_name + "' has unsupported source kind for CSV format");
    case AttributeFormatKind::kArrow:
    #ifdef KNAPSACK_ARROW_ENABLED
      if (!spec.is_file()) {
        return set_error(err, "attribute '" + attr_name + "' Arrow format requires file source");
      }
      return LoadArrowFromFiles(attr_name, spec, builder, err);
    #else
      return set_error(err, "attribute '" + attr_name + "' format '" + spec.format + "' not supported in this build");
    #endif
        case AttributeFormatKind::kParquet:
    #ifdef KNAPSACK_ARROW_ENABLED
      if (!spec.is_file()) {
        return set_error(err, "attribute '" + attr_name + "' Parquet format requires file source");
      }
      return LoadParquetFromFiles(attr_name, spec, builder, err);
    #else
      return set_error(err, "attribute '" + attr_name + "' format '" + spec.format + "' not supported in this build");
    #endif
    case AttributeFormatKind::kUnknown:
      return set_error(err, "attribute '" + attr_name + "' format '" + spec.format + "' not recognized");
  }
  return set_error(err, "attribute '" + attr_name + "' encountered unexpected format");
}

} // namespace v2
