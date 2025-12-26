/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "task_generator/schedule_task_generator.h"
#include "task_generator/concat_schedule_case_generator.h"
#include "task_generator/transpose_schedule_case_generator.h"
#include "task_generator/reduce_schedule_case_generator.h"
#include "task_generator/recompute_case_generator.h"
#include "task_generator/split_schedule_case_generator.h"
#include "task_generator/cube_schedule_case_generator.h"

namespace optimize {
Status ScheduleTaskGenerator::GenerateTasks(::ascir::ImplGraph &optimize_graph, std::vector<ScheduleTask> &tasks,
                                            const OptimizerOptions &options) {                                             
  GE_CHK_STATUS_RET(SplitFusionCaseGenerator().GeneratorTask(optimize_graph, tasks, options),
                    "Failed to generate tasks for split");                                             
  GE_CHK_STATUS_RET(CubeFusionCaseGenerator().GeneratorTask(optimize_graph, tasks, options),
                    "Failed to generate tasks for cube");
  GE_CHK_STATUS_RET(ConcatFusionCaseGenerator().GeneratorTask(optimize_graph, tasks, options),
                    "Failed to generate tasks for concat");
  GE_CHK_STATUS_RET(TransposeFusionCaseGenerator().GeneratorTask(optimize_graph, tasks, options),
                    "Failed to generate tasks for Transpose");
  GE_CHK_STATUS_RET(ReducePartitionCaseGenerator().GeneratorTask(optimize_graph, tasks, options),
                    "Failed to generate tasks for Reduce");
  if (tasks.empty()) {
    GE_CHK_STATUS_RET(RecomputeCaseGenerator().GeneratorTask(optimize_graph, tasks, options),
                      "Failed to generate recomputation tasks for graph[%s].", optimize_graph.GetName().c_str());
  }
  return ge::SUCCESS;
}
}  // namespace optimize
