/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph/build/dag/dag_stream_allocator.h"
#include "graph/build/dag/dag_log.h"

namespace minidag {
void DagStreamAllocator::ByTopological(DAGGraph &graph, StreamAllocConfig &config) {
  // 空实现：当前拓扑序轮询策略暂时不启用
  // 后续可根据需求实现具体的流分配算法
  auto all_nodes = graph.GetAllNodes();
  MINIDAG_LOG_INFO("ByTopological skipped (empty implementation): graph=%s, nodes=%zu",
                   graph.GetName().c_str(), all_nodes.size());

  // 设置默认返回值
  config.required_streams = 0;
  MINIDAG_LOG_INFO("ByTopological done: required_streams=0 (no allocation)");
}
}  // namespace minidag