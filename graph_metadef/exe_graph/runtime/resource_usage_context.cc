/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "exe_graph/runtime/resource_usage_context.h"

#include "resource_usage_collector.h"
#include "common/checker.h"

namespace gert {
ge::graphStatus ResourceUsageCollector::CollectAttachedStreamKeys(const std::vector<ge::AscendString> &keys) {
  for (size_t i = 0U; i < keys.size(); ++i) {
    // 判空必须使用 GetString：stub 库中 AscendString::GetLength 恒返 0，不可依赖
    const char *const key = keys[i].GetString();
    if ((key == nullptr) || (key[0] == '\0')) {
      GELOGE(ge::GRAPH_FAILED, "Attached stream key at index %zu is empty.", i);
      has_error_ = true;
      return ge::GRAPH_FAILED;
    }
    (void)attached_stream_keys_.emplace(key);
  }
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus ResourceUsageContext::ReportAttachedStream(const std::vector<ge::AscendString> &keys) {
  const auto additional_input_start = GetAdditionalInputStartIndex();
  if (additional_input_start < 0) {
    GELOGE(ge::GRAPH_FAILED, "Resource usage context has no compute node info.");
    return ge::GRAPH_FAILED;
  }
  const int64_t collector_index = additional_input_start + static_cast<int64_t>(ResourceUsageInput::kCollector);
  if (collector_index < 0) {
    GELOGE(ge::GRAPH_FAILED, "Resource usage context collector index %ld overflow.", collector_index);
    return ge::GRAPH_FAILED;
  }
  auto *collector = GetInputValue<ResourceUsageCollector *>(static_cast<size_t>(collector_index));
  if (collector == nullptr) {
    GELOGE(ge::GRAPH_FAILED, "Resource usage context collector is null.");
    return ge::GRAPH_FAILED;
  }
  return collector->CollectAttachedStreamKeys(keys);
}
}  // namespace gert
