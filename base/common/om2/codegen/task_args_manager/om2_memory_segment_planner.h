/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_CXX_BASE_COMMON_OM2_CODEGEN_OM2_MEMORY_SEGMENT_PLANNER_H_
#define AIR_CXX_BASE_COMMON_OM2_CODEGEN_OM2_MEMORY_SEGMENT_PLANNER_H_
#include <cstdint>
#include "common/om2/codegen/om2_codegen_types.h"

namespace ge {
namespace om2 {
class MemorySegmentPlanner {
 public:
  uintptr_t Allocate(SegmentType type, uint64_t size);

 private:
  uintptr_t current_base_{kPlanMemSegmentInitialBase};

  std::vector<MemorySegmentInfo> memory_segment_infos_;
};
}  // namespace om2
}  // namespace ge

#endif  // AIR_CXX_BASE_COMMON_OM2_CODEGEN_OM2_MEMORY_SEGMENT_PLANNER_H_
