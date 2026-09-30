/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "declarative_stream_registry.h"

#include <limits>
#include <memory>
#include <algorithm>

#include "framework/common/debug/ge_log.h"

namespace ge {
DeclarativeStreamRegistry::DeclarativeStreamRegistry(const uint32_t first_stream_id)
    : next_stream_id_(first_stream_id) {}

uint32_t DeclarativeStreamRegistry::RequestAttachedStream(const AscendString &key) {
  const std::string key_string(key.GetString());
  if (key_string.empty()) {
    GELOGE(PARAM_INVALID, "[DeclarativeStreamRegistry] empty attached stream key.");
    return std::numeric_limits<uint32_t>::max();
  }
  const auto found = key_to_stream_id_.find(key_string);
  if (found != key_to_stream_id_.end()) {
    return found->second;
  }
  if (next_stream_id_ == std::numeric_limits<uint32_t>::max()) {
    GELOGE(INTERNAL_ERROR, "[DeclarativeStreamRegistry] attached stream id overflow, next_id=%u.", next_stream_id_);
    return std::numeric_limits<uint32_t>::max();
  }
  const auto stream_id = next_stream_id_++;
  key_to_stream_id_.emplace(key_string, stream_id);
  allocated_stream_ids_.emplace_back(stream_id);
  return stream_id;
}

bool DeclarativeStreamRegistry::Contains(const uint32_t stream_id) const {
  return std::find(allocated_stream_ids_.cbegin(), allocated_stream_ids_.cend(), stream_id) !=
         allocated_stream_ids_.cend();
}

const std::vector<uint32_t> &DeclarativeStreamRegistry::GetAllocatedStreamIds() const {
  return allocated_stream_ids_;
}
}  // namespace ge
