/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "multi_thread_exe_graph_resource_guard.h"

namespace gert {
TaskScheduler *MultiThreadResourceGuard::ResetTaskScheduler(std::unique_ptr<TaskScheduler> scheduler) {
  task_scheduler_guarder_ = std::move(scheduler);
  return task_scheduler_guarder_.get();
}

const FreeLaunchRelationCsr &MultiThreadResourceGuard::ResetFreeLaunchRelationCsr(std::unique_ptr<uint8_t[]> offsets,
                                                                                  std::unique_ptr<uint8_t[]> launch_ids,
                                                                                  const size_t node_num,
                                                                                  const size_t relation_num) {
  free_launch_offsets_guarder_ = std::move(offsets);
  free_launch_ids_guarder_ = std::move(launch_ids);
  free_launch_relation_csr_ = {reinterpret_cast<const NodeIdentity *>(free_launch_offsets_guarder_.get()),
                               reinterpret_cast<const NodeIdentity *>(free_launch_ids_guarder_.get()), node_num,
                               relation_num};
  return free_launch_relation_csr_;
}
}  // namespace gert
