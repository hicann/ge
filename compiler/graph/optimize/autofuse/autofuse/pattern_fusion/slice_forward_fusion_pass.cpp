/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "slice_forward_fusion_pass.h"
#include "common/checker.h"
#include "graph/utils/graph_utils.h"
#include "pattern_fusion_utils.h"

namespace ge {

namespace {
using namespace pattern_fusion;
const std::unordered_set<std::string> kSliceOpTypes = {"Slice", "SliceD", "StridedSlice", "StridedSliceD"};

std::vector<NodePtr> CollectElemChainForSlice(const NodePtr &slice_node) {
  std::vector<NodePtr> elem_chain;
  const auto &slice_in_nodes = slice_node->GetInDataNodes();
  if (slice_in_nodes.empty()) {
    return elem_chain;
  }
  auto current = slice_in_nodes.at(0);
  while (current != nullptr && IsElementwise(current) && current->GetInNodesSize() == 1UL &&
         current->GetOutNodesSize() == 1UL) {
    // dtype发生变化时不前提，防止出现slice不支持的dtype或引入精度、性能问题
    const auto input_dtype = current->GetOpDesc()->GetInputDesc(0).GetDataType();
    const auto output_dtype = current->GetOpDesc()->GetOutputDesc(0).GetDataType();
    if (input_dtype != output_dtype) {
      break;
    }
    elem_chain.push_back(current);
    // 检查是否有下一个输入节点
    const auto &current_in_nodes = current->GetInDataNodes();
    if (current_in_nodes.empty()) {
      break;
    }
    current = current_in_nodes.at(0);
  }
  return elem_chain;
}

graphStatus DoSliceForward(const NodePtr &slice_node, const std::vector<NodePtr> &elem_chain) {
  const auto &first_elem = elem_chain.front();  // elem1
  const auto &last_elem = elem_chain.back();    // elemN
  auto last_elem_input = last_elem->GetInDataAnchor(0);
  auto last_elem_input_peer = last_elem_input->GetPeerOutAnchor();  // input的输出

  // 1. 断开 elem1 -> slice, input -> elemN
  GE_CHK_GRAPH_STATUS_RET(GraphUtils::RemoveEdge(first_elem->GetOutDataAnchor(0), slice_node->GetInDataAnchor(0)),
                          "Failed to remove edge from %s to %s", first_elem->GetNamePtr(), slice_node->GetNamePtr());
  GE_CHK_GRAPH_STATUS_RET(GraphUtils::RemoveEdge(last_elem_input_peer, last_elem_input),
                          "Failed to remove edge from input peer to %s", last_elem->GetNamePtr());

  // 2. 替换 slice 的输出边：slice -> output 改为 elem1 -> output
  for (const auto &peer_in : slice_node->GetOutDataAnchor(0)->GetPeerInDataAnchors()) {
    GE_CHK_GRAPH_STATUS_RET(
        GraphUtils::ReplaceEdgeSrc(slice_node->GetOutDataAnchor(0), peer_in, first_elem->GetOutDataAnchor(0)),
        "Failed to replace edge source for slice %s", slice_node->GetNamePtr());
  }

  // 3. 重连slice的输入输出 input -> slice -> elemN
  GE_CHK_GRAPH_STATUS_RET(GraphUtils::AddEdge(last_elem_input_peer, slice_node->GetInDataAnchor(0)),
                          "Failed to add edge to slice %s", slice_node->GetNamePtr());
  GE_CHK_GRAPH_STATUS_RET(GraphUtils::AddEdge(slice_node->GetOutDataAnchor(0), last_elem_input),
                          "Failed to add edge from slice %s to %s", slice_node->GetNamePtr(), last_elem->GetNamePtr());
  return GRAPH_SUCCESS;
}

// 更新 elem 链中所有节点的 shape
graphStatus UpdateElemShapes(const std::vector<NodePtr> &elem_chain, const GeShape &shape,
                             const gert::SymbolShape &symbol_shape) {
  for (const auto &elem : elem_chain) {
    GE_CHK_STATUS_RET(SetNodeShape(elem, shape, shape, symbol_shape), "Failed to update shape for node %s.",
                      elem->GetNamePtr());
  }
  return GRAPH_SUCCESS;
}
}  // namespace

graphStatus SliceForwardFusionPass::Run(const ComputeGraphPtr &graph) {
  GE_CHECK_NOTNULL(graph);
  for (const auto &slice_node : graph->GetDirectNode()) {
    if (kSliceOpTypes.find(slice_node->GetType()) == kSliceOpTypes.end()) {
      continue;
    }

    auto elem_chain = CollectElemChainForSlice(slice_node);
    if (elem_chain.empty()) {
      continue;
    }

    GELOGD("SliceForwardFusionPass: hoist slice[%s] past elem chain(size=%zu)", slice_node->GetNamePtr(),
           elem_chain.size());

    const auto &slice_output_shape = slice_node->GetOpDesc()->GetOutputDesc(0).GetShape();
    const auto &slice_symbol_shape = GetNodeSymbolShape(slice_node);

    GE_CHK_GRAPH_STATUS_RET(DoSliceForward(slice_node, elem_chain), "Failed to do slice forward for node %s",
                            slice_node->GetNamePtr());
    GE_CHK_GRAPH_STATUS_RET(UpdateElemShapes(elem_chain, slice_output_shape, slice_symbol_shape),
                            "Failed to update elem shapes for slice %s", slice_node->GetNamePtr());
  }
  return GRAPH_SUCCESS;
}

}  // namespace ge
