/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_CXX_TASK_GENERATOR_UTILS_H
#define AIR_CXX_TASK_GENERATOR_UTILS_H

#include "graph/op_desc.h"
#include "proto/task.pb.h"
namespace ge {
// 标记节点的任务已生成完毕且不可再次生成：拆流阶段会往任务列表插入事件任务并把流号改写为真实流号，
// 二次生成只能从算子侧的缓存计划重物化，会丢掉这些结果。目前仅声明式自定义算子（AnnotatedArgsOp）
// 的带依赖/附着流计划会打标，见 custom_ops_kernel_builder.cc 的 CacheAnnotatedArgsTaskPlan。
constexpr char kKeepGeneratedTasksAttr[] = "_keep_generated_tasks";
bool NoNeedGenTask(const OpDescPtr &op_desc);
bool NeedKeepGeneratedTasks(const OpDescPtr &op_desc);
void RefreshTaskDefStreamId(bool has_attached_stream, int64_t logical_stream_id, int64_t real_stream_id,
                            std::vector<domi::TaskDef> &task_defs);
}  // namespace ge

#endif  // AIR_CXX_TASK_GENERATOR_UTILS_H
