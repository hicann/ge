/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GE_COMPILER_GRAPH_BUILD_STREAM_ANNOTATED_ARGS_TASK_PLAN_H_
#define GE_COMPILER_GRAPH_BUILD_STREAM_ANNOTATED_ARGS_TASK_PLAN_H_

#include <cstdint>
#include <memory>
#include <vector>

#include "graph/buffer.h"
#include "graph/op_desc.h"
#include "ge/ge_api_error_codes.h"
#include "framework/omg/omg_inner_types.h"
#include "common/opskernel/ops_kernel_info_types.h"
#include "proto/task.pb.h"

namespace ge {
struct AnnotatedArgsLaunchDependency {
  uint32_t predecessor_launch_index;
  uint32_t successor_launch_index;
  uint32_t event_id;
};

struct AnnotatedArgsTaskPlan {
  std::vector<domi::TaskDef> task_templates;
  std::vector<uint32_t> launch_stream_ids;
  std::vector<AnnotatedArgsLaunchDependency> dependencies;
  std::vector<uint32_t> attached_stream_ids;
};

using AnnotatedArgsTaskPlanPtr = std::shared_ptr<const AnnotatedArgsTaskPlan>;

Status SerializeAnnotatedArgsTaskPlan(const AnnotatedArgsTaskPlan &plan, Buffer &encoded);
Status DeserializeAnnotatedArgsTaskPlan(const Buffer &encoded, AnnotatedArgsTaskPlan &plan);
}  // namespace ge

#endif
