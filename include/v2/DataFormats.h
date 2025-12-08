// Copyright (c) 2025
// SPDX-License-Identifier: MIT

#pragma once

#include <string>

#include "v2/Config.h"
#include "v2/Data.h"

namespace v2 {

// Load attribute data from file/stream sources into HostSoABuilder according to the
// provided AttributeSourceSpec. Returns false and fills err on failure.
bool LoadAttributeFromSource(const std::string& attr_name,
                             const AttributeSourceSpec& spec,
                             int expected_count,
                             HostSoABuilder* builder,
                             std::string* err);

} // namespace v2
