/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ASCGEN_DEV_SRC_OPTIMIZE_TASK_GENERATOR_SCHEDULE_GROUP_PARTITIONER_H_
#define ASCGEN_DEV_SRC_OPTIMIZE_TASK_GENERATOR_SCHEDULE_GROUP_PARTITIONER_H_

#include "ascir.h"
#include "ascgen_log.h"

namespace optimize {
class ScheduleGroupGraphPartitioner {
 public:
  static Status PartitionByConnectivity(const ::ascir::ImplGraph &optimize_graph,
                                        std::vector<::ascir::ImplGraph> &sub_optimize_graphs,
                                        std::vector<ge::AscNodePtr> node_order = {});

 private:
  static Status AddConnectedNodes(const ge::AscNodePtr &root_node, ::ascir::ImplGraph &sub_graph,
                                  std::set<ge::NodePtr> &all_visited);

  static bool CompareByNodeId(const ge::AscNodePtr &lhs, const ge::AscNodePtr &rhs);
};
}  // namespace optimize

#endif  // ASCGEN_DEV_SRC_OPTIMIZE_TASK_GENERATOR_SCHEDULE_GROUP_PARTITIONER_H_
