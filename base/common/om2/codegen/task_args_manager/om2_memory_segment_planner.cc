/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "om2_memory_segment_planner.h"
#include "common/ge_common/debug/ge_log.h"

#include <limits>

namespace ge {
namespace om2 {
uintptr_t MemorySegmentPlanner::Allocate(SegmentType type, const uint64_t size) {
  MemorySegmentInfo segment{};
  segment.type = type;
  segment.base = current_base_;
  segment.size = size;
  memory_segment_infos_.push_back(segment);
  current_base_ += static_cast<uintptr_t>(size);
  GELOGI("[OM2] Success to allocate segment, type[%d], base=0x%lx, size=%lu.", static_cast<int32_t>(type),
         static_cast<unsigned long>(segment.base), static_cast<unsigned long>(size));
  return segment.base;
}
}  // namespace om2
}  // namespace ge
