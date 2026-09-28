/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GE_GRAPH_BUILD_STREAM_DECLARATIVE_STREAM_REGISTRY_H_
#define GE_GRAPH_BUILD_STREAM_DECLARATIVE_STREAM_REGISTRY_H_

#include <cstdint>
#include <memory>
#include <map>
#include <string>
#include <vector>

#include "graph/ascend_string.h"

namespace ge {
constexpr char kDeclarativeStreamRegistryAttr[] = "declarative_stream_registry";
class DeclarativeStreamRegistry {
 public:
  explicit DeclarativeStreamRegistry(uint32_t first_stream_id);
  uint32_t RequestAttachedStream(const AscendString &key);
  bool Contains(uint32_t stream_id) const;
  const std::vector<uint32_t> &GetAllocatedStreamIds() const;

 private:
  std::map<std::string, uint32_t> key_to_stream_id_;
  std::vector<uint32_t> allocated_stream_ids_;
  uint32_t next_stream_id_;
};
using DeclarativeStreamRegistryPtr = std::shared_ptr<DeclarativeStreamRegistry>;
}  // namespace ge

#endif  // GE_GRAPH_BUILD_STREAM_DECLARATIVE_STREAM_REGISTRY_H_
