// Copyright (c) 2025
// SPDX-License-Identifier: MIT

#ifdef __APPLE__
#import <Foundation/Foundation.h>
#endif

#include "v2/Config.h"

#include <cstddef>
#include <fstream>
#include <sstream>

namespace v2 {

static bool validateItems(const ItemsSpec& items, std::string* err) {
  for (const auto& kv : items.attributes) {
    const auto& name = kv.first;
    const auto& vec = kv.second;
    if ((int)vec.size() != items.count) {
      if (err) *err = "attribute '" + name + "' size (" + std::to_string(vec.size()) + ") != items.count (" + std::to_string(items.count) + ")";
      return false;
    }
  }
  for (const auto& kv : items.sources) {
    const auto& name = kv.first;
    const auto& spec = kv.second;
    if (items.attributes.count(name)) {
      if (err) *err = "attribute '" + name + "' provided both inline and via external source";
      return false;
    }
    if (spec.kind == AttributeSourceKind::kInline) {
      if (err) *err = "attribute '" + name + "' external spec missing source";
      return false;
    }
    if (spec.is_file() && spec.path.empty() && spec.chunks.empty()) {
      if (err) *err = "file-backed attribute '" + name + "' missing path/chunks";
      return false;
    }
    if (spec.is_stream() && spec.channel.empty() && spec.chunks.empty()) {
      if (err) *err = "stream-backed attribute '" + name + "' missing channel/chunks";
      return false;
    }
    if (spec.format != "binary64_le") {
      if (err) *err = "attribute '" + name + "' format '" + spec.format + "' unsupported";
      return false;
    }
  }
  return true;
}

static bool validateBlocks(const ItemsSpec& items, const std::vector<BlockSpec>& blocks, std::string* err) {
  for (const auto& b : blocks) {
    if (b.start >= 0) {
      if (b.count < 0 || b.start < 0 || b.start + b.count > items.count) {
        if (err) *err = "block '" + b.name + "' range out of bounds";
        return false;
      }
    } else if (!b.indices.empty()) {
      for (int idx : b.indices) {
        if (idx < 0 || idx >= items.count) {
          if (err) *err = "block '" + b.name + "' index out of bounds";
          return false;
        }
      }
    } else {
      if (err) *err = "block '" + b.name + "' missing range or indices";
      return false;
    }
  }
  return true;
}

bool ValidateConfig(const Config& cfg, std::string* err) {
  if (cfg.version != 2) {
    // allow other versions but warn via err if provided
  }
  if (cfg.items.count <= 0) {
    if (err) *err = "items.count must be > 0";
    return false;
  }
  if (!validateItems(cfg.items, err)) return false;
  if (!validateBlocks(cfg.items, cfg.blocks, err)) return false;
  if (cfg.mode == "assign") {
    if (cfg.knapsack.K <= 0) {
      if (err) *err = "knapsack.K must be > 0 in assign mode";
      return false;
    }
    if ((int)cfg.knapsack.capacities.size() != cfg.knapsack.K) {
      if (err) *err = "knapsack.capacities size must equal K";
      return false;
    }
    if (cfg.knapsack.capacity_attr.empty()) {
      if (err) *err = "knapsack.capacity_attr required in assign mode";
      return false;
    }
    if (!cfg.items.HasAttribute(cfg.knapsack.capacity_attr)) {
      if (err) *err = "capacity_attr '" + cfg.knapsack.capacity_attr + "' not found in items.attributes";
      return false;
    }
  }
  if (cfg.objective.empty()) {
    if (err) *err = "objective must have at least one term";
    return false;
  }
  for (const auto& term : cfg.objective) {
    if (!cfg.items.HasAttribute(term.attr)) {
      if (err) *err = "objective attr '" + term.attr + "' not found in items.attributes";
      return false;
    }
  }
  for (const auto& c : cfg.constraints) {
    if (c.attr.size() && !cfg.items.HasAttribute(c.attr)) {
      if (err) *err = "constraint attr '" + c.attr + "' not found in items.attributes";
      return false;
    }
    if (c.soft) {
      if (c.penalty.weight <= 0) {
        if (err) *err = "soft constraint requires positive penalty.weight";
        return false;
      }
      if (c.penalty.power <= 0) {
        if (err) *err = "soft constraint requires positive penalty.power";
        return false;
      }
    }
  }
  return true;
}

#ifdef __APPLE__

static bool getString(NSDictionary* dict, NSString* key, std::string* out) {
  id val = dict[key];
  if (!val) return false;
  if (![val isKindOfClass:[NSString class]]) return false;
  *out = std::string([(NSString*)val UTF8String]);
  return true;
}

static bool getInt(NSDictionary* dict, NSString* key, int* out) {
  id val = dict[key];
  if (!val) return false;
  if ([val respondsToSelector:@selector(intValue)]) { *out = (int)[val intValue]; return true; }
  return false;
}

static bool getUInt64(NSDictionary* dict, NSString* key, std::uint64_t* out) {
  id val = dict[key];
  if (!val) return false;
  if ([val respondsToSelector:@selector(unsignedLongLongValue)]) { *out = (std::uint64_t)[val unsignedLongLongValue]; return true; }
  if ([val respondsToSelector:@selector(longLongValue)]) { long long v = [val longLongValue]; if (v >= 0) { *out = (std::uint64_t)v; return true; } }
  return false;
}

static bool getDoubleArray(NSDictionary* dict, NSString* key, std::vector<double>* out) {
  id val = dict[key];
  if (!val) return false;
  if (![val isKindOfClass:[NSArray class]]) return false;
  NSArray* arr = (NSArray*)val;
  out->clear();
  out->reserve([arr count]);
  for (id e in arr) {
    if (![e respondsToSelector:@selector(doubleValue)]) return false;
    out->push_back([e doubleValue]);
  }
  return true;
}

static bool getIntArray(NSDictionary* dict, NSString* key, std::vector<int>* out) {
  id val = dict[key];
  if (!val) return false;
  if (![val isKindOfClass:[NSArray class]]) return false;
  NSArray* arr = (NSArray*)val;
  out->clear();
  out->reserve([arr count]);
  for (id e in arr) {
    if (![e respondsToSelector:@selector(intValue)]) return false;
    out->push_back((int)[e intValue]);
  }
  return true;
}

static bool parseExternalAttr(NSString* key, NSDictionary* obj, ItemsSpec* items, std::string* err) {
  AttributeSourceSpec spec;
  NSString* source = obj[@"source"];
  if (!source || ![source isKindOfClass:[NSString class]]) {
    if (err) *err = "attribute '" + std::string([key UTF8String]) + "' missing source";
    return false;
  }
  std::string sourceStr([source UTF8String]);
  if (sourceStr == "file") {
    spec.kind = AttributeSourceKind::kFile;
  } else if (sourceStr == "stream") {
    spec.kind = AttributeSourceKind::kStream;
  } else {
    if (err) *err = "attribute '" + std::string([key UTF8String]) + "' has unsupported source";
    return false;
  }
  NSString* fmt = obj[@"format"]; if (fmt) spec.format = std::string([fmt UTF8String]);
  NSString* path = obj[@"path"]; if (path) spec.path = std::string([path UTF8String]);
  NSString* channel = obj[@"channel"]; if (channel) spec.channel = std::string([channel UTF8String]);
  id offset = obj[@"offset_bytes"];
  if (offset) {
    if (![offset respondsToSelector:@selector(unsignedLongLongValue)]) {
      if (err) *err = "attribute '" + std::string([key UTF8String]) + "' offset_bytes invalid";
      return false;
    }
    spec.offset_bytes = (std::size_t)[offset unsignedLongLongValue];
  }
  id chunks = obj[@"chunks"];
  if (chunks) {
    if (![chunks isKindOfClass:[NSArray class]]) {
      if (err) *err = "attribute '" + std::string([key UTF8String]) + "' chunks must be array";
      return false;
    }
    for (id entry in (NSArray*)chunks) {
      if (![entry isKindOfClass:[NSString class]]) {
        if (err) *err = "attribute '" + std::string([key UTF8String]) + "' chunks must be strings";
        return false;
      }
      spec.chunks.push_back(std::string([(NSString*)entry UTF8String]));
    }
  }
  std::string attrName([key UTF8String]);
  if (items->sources.count(attrName) || items->attributes.count(attrName)) {
    if (err) *err = "duplicate attribute '" + attrName + "'";
    return false;
  }
  items->sources[attrName] = std::move(spec);
  return true;
}

static bool parseItems(NSDictionary* root, ItemsSpec* items, std::string* err) {
  NSDictionary* itemsDict = root[@"items"];
  if (!itemsDict || ![itemsDict isKindOfClass:[NSDictionary class]]) { if (err) *err = "missing 'items'"; return false; }
  if (!getInt(itemsDict, @"count", &items->count)) { if (err) *err = "items.count missing or invalid"; return false; }
  NSDictionary* attrs = itemsDict[@"attributes"];
  if (!attrs || ![attrs isKindOfClass:[NSDictionary class]]) { if (err) *err = "items.attributes missing"; return false; }
  for (NSString* key in attrs) {
    id val = attrs[key];
    if ([val isKindOfClass:[NSArray class]]) {
      std::vector<double> vec;
      NSArray* arr = (NSArray*)val;
      vec.reserve([arr count]);
      for (id e in arr) {
        if (![e respondsToSelector:@selector(doubleValue)]) { if (err) *err = "attribute '" + std::string([key UTF8String]) + "' contains non-numeric"; return false; }
        vec.push_back([e doubleValue]);
      }
      std::string attrName([key UTF8String]);
      if (items->sources.count(attrName)) { if (err) *err = "duplicate attribute '" + attrName + "'"; return false; }
      items->attributes[attrName] = std::move(vec);
    } else if ([val isKindOfClass:[NSDictionary class]]) {
      if (!parseExternalAttr(key, (NSDictionary*)val, items, err)) return false;
    } else {
      if (err) *err = "attribute '" + std::string([key UTF8String]) + "' must be array or object";
      return false;
    }
  }
  return true;
}

static bool parseBlocks(NSDictionary* root, std::vector<BlockSpec>* blocks, std::string* err) {
  NSArray* arr = root[@"blocks"];
  if (!arr || ![arr isKindOfClass:[NSArray class]]) { if (err) *err = "missing 'blocks'"; return false; }
  blocks->clear();
  for (id elem in arr) {
    if (![elem isKindOfClass:[NSDictionary class]]) { if (err) *err = "block entry not an object"; return false; }
    NSDictionary* b = (NSDictionary*)elem;
    BlockSpec bs;
    NSString* name = b[@"name"]; if (name) bs.name = std::string([name UTF8String]); else bs.name = "";
    int start = -1, count = 0; std::vector<int> indices;
    getInt(b, @"start", &start);
    getInt(b, @"count", &count);
    getIntArray(b, @"indices", &indices);
    bs.start = start; bs.count = count; bs.indices = std::move(indices);
    blocks->push_back(std::move(bs));
  }
  return true;
}

static bool parseObjective(NSDictionary* root, std::vector<CostTermSpec>* objective, std::string* err) {
  NSArray* arr = root[@"objective"];
  if (!arr || ![arr isKindOfClass:[NSArray class]]) { if (err) *err = "missing 'objective'"; return false; }
  objective->clear();
  for (id elem in arr) {
    if (![elem isKindOfClass:[NSDictionary class]]) { if (err) *err = "objective entry not an object"; return false; }
    NSDictionary* o = (NSDictionary*)elem;
    CostTermSpec ct;
    NSString* attr = o[@"attr"]; if (!attr) { if (err) *err = "objective.attr missing"; return false; }
    ct.attr = std::string([attr UTF8String]);
    id w = o[@"weight"]; ct.weight = w ? [w doubleValue] : 1.0;
    objective->push_back(std::move(ct));
  }
  return true;
}

static bool parseConstraints(NSDictionary* root, std::vector<ConstraintSpec>* constraints, std::string* err) {
  NSArray* arr = root[@"constraints"];
  if (!arr || ![arr isKindOfClass:[NSArray class]]) { constraints->clear(); return true; }
  constraints->clear();
  for (id elem in arr) {
    if (![elem isKindOfClass:[NSDictionary class]]) { if (err) *err = "constraint entry not an object"; return false; }
    NSDictionary* c = (NSDictionary*)elem;
    ConstraintSpec cs;
    NSString* kind = c[@"kind"]; if (kind) cs.kind = std::string([kind UTF8String]);
    NSString* attr = c[@"attr"]; if (attr) cs.attr = std::string([attr UTF8String]);
    id lim = c[@"limit"]; if (lim) cs.limit = [lim doubleValue];
    id soft = c[@"soft"]; if (soft) cs.soft = [soft boolValue];
    NSDictionary* p = c[@"penalty"];
    if (p && [p isKindOfClass:[NSDictionary class]]) {
      id w = p[@"weight"]; if (w) cs.penalty.weight = [w doubleValue];
      id pow = p[@"power"]; if (pow) cs.penalty.power = [pow doubleValue];
    }
    constraints->push_back(std::move(cs));
  }
  return true;
}

static bool parseKnapsack(NSDictionary* root, KnapsackSpec* ks, std::string* err) {
  NSDictionary* k = root[@"knapsack"];
  if (!k || ![k isKindOfClass:[NSDictionary class]]) { if (err) *err = "missing 'knapsack'"; return false; }
  getInt(k, @"K", &ks->K);
  getDoubleArray(k, @"capacities", &ks->capacities);
  NSString* capAttr = k[@"capacity_attr"]; if (capAttr) ks->capacity_attr = std::string([capAttr UTF8String]);
  return true;
}

bool LoadConfigFromJsonString(const std::string& json, Config* out, std::string* err) {
  if (!out) { if (err) *err = "out is null"; return false; }
  @autoreleasepool {
    NSData* data = [NSData dataWithBytes:json.data() length:json.size()];
    NSError* nserr = nil;
    id obj = [NSJSONSerialization JSONObjectWithData:data options:0 error:&nserr];
    if (!obj || ![obj isKindOfClass:[NSDictionary class]]) {
      if (err) *err = nserr ? std::string([[nserr localizedDescription] UTF8String]) : std::string("invalid JSON root");
      return false;
    }
    NSDictionary* root = (NSDictionary*)obj;

    Config cfg;
    int version = 2; getInt(root, @"version", &version); cfg.version = version;
    std::string mode; if (getString(root, @"mode", &mode)) cfg.mode = mode; else cfg.mode = "assign";
    std::uint64_t seed = 0; getUInt64(root, @"random_seed", &seed); cfg.random_seed = seed;

    if (!parseItems(root, &cfg.items, err)) return false;
    if (!parseBlocks(root, &cfg.blocks, err)) return false;
    if (!parseObjective(root, &cfg.objective, err)) return false;
    if (!parseConstraints(root, &cfg.constraints, err)) return false;
    if (cfg.mode == "assign") {
      if (!parseKnapsack(root, &cfg.knapsack, err)) return false;
    }

    if (!ValidateConfig(cfg, err)) return false;
    *out = std::move(cfg);
    return true;
  }
}

bool LoadConfigFromFile(const std::string& path, Config* out, std::string* err) {
  if (!out) { if (err) *err = "out is null"; return false; }
  @autoreleasepool {
    NSString* nsPath = [NSString stringWithUTF8String:path.c_str()];
    NSError* nserr = nil;
    NSData* data = [NSData dataWithContentsOfFile:nsPath options:0 error:&nserr];
    if (!data) {
      if (err) *err = nserr ? std::string([[nserr localizedDescription] UTF8String]) : std::string("failed to read file");
      return false;
    }
    id obj = [NSJSONSerialization JSONObjectWithData:data options:0 error:&nserr];
    if (!obj || ![obj isKindOfClass:[NSDictionary class]]) {
      if (err) *err = nserr ? std::string([[nserr localizedDescription] UTF8String]) : std::string("invalid JSON root");
      return false;
    }
    NSDictionary* root = (NSDictionary*)obj;

    Config cfg;
    int version = 2; getInt(root, @"version", &version); cfg.version = version;
    std::string mode; if (getString(root, @"mode", &mode)) cfg.mode = mode; else cfg.mode = "assign";
    std::uint64_t seed = 0; getUInt64(root, @"random_seed", &seed); cfg.random_seed = seed;

    if (!parseItems(root, &cfg.items, err)) return false;
    if (!parseBlocks(root, &cfg.blocks, err)) return false;
    if (!parseObjective(root, &cfg.objective, err)) return false;
    if (!parseConstraints(root, &cfg.constraints, err)) return false;
    if (cfg.mode == "assign") {
      if (!parseKnapsack(root, &cfg.knapsack, err)) return false;
    }
    if (!ValidateConfig(cfg, err)) return false;
    *out = std::move(cfg);
    return true;
  }
}

#else

bool LoadConfigFromJsonString(const std::string& /*json*/, Config* /*out*/, std::string* err) {
  if (err) *err = "LoadConfigFromJsonString not implemented on this platform yet";
  return false;
}

bool LoadConfigFromFile(const std::string& /*path*/, Config* /*out*/, std::string* err) {
  if (err) *err = "LoadConfigFromFile not implemented on this platform yet";
  return false;
}

#endif

} // namespace v2
