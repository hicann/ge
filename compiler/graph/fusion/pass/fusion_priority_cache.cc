/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "fusion_priority_cache.h"

namespace ge {
namespace fusion {
FusionPriorityCache &FusionPriorityCache::GetInstance() {
  static FusionPriorityCache instance;
  return instance;
}

void FusionPriorityCache::UpdateGraphFusionPriorityMap(const std::map<std::string, int32_t> &priority_map) {
  std::lock_guard<std::mutex> lock(mutex_);
  graph_fusion_priority_map_ = priority_map;
}

std::map<std::string, int32_t> FusionPriorityCache::GetGraphFusionPriorityMap() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return graph_fusion_priority_map_;
}
}  // namespace fusion
}  // namespace ge
