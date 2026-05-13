/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GE_GRAPH_BUILD_STREAM_DAG_ADAPTER_H_
#define GE_GRAPH_BUILD_STREAM_DAG_ADAPTER_H_

#include <memory>

#include "graph/build/dag/dag_graph.h"
#include "external/register/register_custom_pass.h"

namespace minidag {
class DAGAdapter {
 public:
  static ge::graphStatus ToGEStatus(graphStatus status);
  static ge::graphStatus FromGEGraph(const ge::ConstGraphPtr &ge_graph,
                                      std::shared_ptr<DAGGraph> &dag);
  static ge::graphStatus RefreshStreamIdsToGE(
      const DAGGraph &dag,
      const ge::ConstGraphPtr &ge_graph,
      ge::StreamPassContext &context);
  DAGAdapter() = delete;

 private:
  static graphStatus ConvertNodes(const ge::ConstGraphPtr &ge_graph, DAGGraph &dag);
  static graphStatus ConvertEdges(const ge::ConstGraphPtr &ge_graph, DAGGraph &dag);
  static graphStatus ConvertDataEdgesForNode(
      const ge::GNode &gnode,
      const std::shared_ptr<DAGNode> &src_node,
      DAGGraph &dag,
      int64_t &edge_count);
  static graphStatus ConvertControlEdgesForNode(
      const ge::GNode &gnode,
      const std::shared_ptr<DAGNode> &src_node,
      DAGGraph &dag,
      int64_t &edge_count);
};
}  // namespace minidag
#endif  // GE_GRAPH_BUILD_STREAM_DAG_ADAPTER_H_