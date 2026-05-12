/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph/build/stream/dag_stream_allocator_pass.h"
#include "graph/build/stream/dag_adapter.h"
#include "graph/build/dag/dag_stream_allocator.h"
#include "framework/common/debug/ge_log.h"

namespace minidag {
ge::Status RunMiniDAGStreamPass(
    const ge::ConstGraphPtr &graph,
    ge::StreamPassContext &context) {
  // 1. 从 GE Graph 构建 DAG
  std::shared_ptr<DAGGraph> dag;
  auto ret = DAGAdapter::FromGEGraph(graph, dag);
  if (ret != ge::GRAPH_SUCCESS || dag == nullptr) {
    GELOGE(ge::FAILED, "MiniDAGStreamPass failed: FromGEGraph returned %d", ret);
    return ge::FAILED;
  }

  // 2. 执行拓扑序流分配策略（当前为空实现）
  StreamAllocConfig config{-1, 0};
  DagStreamAllocator::ByTopological(*dag, config);

  // 3. 分配 stream ID（基于策略返回的需求）
  for (int64_t i = 0; i < config.required_streams; ++i) {
    (void)context.AllocateNextStreamId();
  }

  // 4. 将 DAG 的 stream_id 写回 GE Graph
  auto refresh_ret = DAGAdapter::RefreshStreamIdsToGE(*dag, graph, context);
  return (refresh_ret == ge::GRAPH_SUCCESS) ? ge::SUCCESS : ge::FAILED;
}
}  // namespace minidag

// Pass 注册：调用提取的函数
REGISTER_CUSTOM_PASS("MiniDAGStreamPass")
    .CustomAllocateStreamPassFn([](const ge::ConstGraphPtr &graph,
                                    ge::StreamPassContext &context) -> ge::Status {
        return minidag::RunMiniDAGStreamPass(graph, context);
    })
    .Stage(ge::CustomPassStage::kAfterAssignLogicStream);