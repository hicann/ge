/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "transpose_with_broadcast_eliminate_pass.h"
#include "common/checker.h"
#include "graph/utils/graph_utils.h"
#include "attribute_group/attr_group_symbolic_desc.h"
#include "pattern_fusion_utils.h"

namespace ge {
using namespace pattern_fusion;
namespace {
const std::unordered_set<std::string> kTransposeOpTypes = {"Transpose", "TransposeD"};
const std::unordered_set<std::string> kScalarBroadcastOpTypes = {"ZerosLike", "OnesLike"};
const std::string kFillOpType = "Fill";

// 更新节点shape和符号化shape
void UpdateNodeShapeAndSymbolShape(const NodePtr &node, const NodePtr &transpose_node) {
  const auto &transpose_output_shape = transpose_node->GetOpDesc()->GetOutputDesc(0U).GetShape();
  auto output_desc = node->GetOpDesc()->MutableOutputDesc(0U);
  output_desc->SetShape(transpose_output_shape);
  output_desc->SetOriginShape(transpose_output_shape);

  const auto &transpose_symbol_shape = GetNodeSymbolShape(transpose_node);
  if (!transpose_symbol_shape.GetDims().empty()) {
    auto sym_attr = output_desc->GetOrCreateAttrsGroup<SymbolicDescAttr>();
    if (sym_attr != nullptr) {
      sym_attr->symbolic_tensor.MutableOriginSymbolShape() = transpose_symbol_shape;
    }
  }
}

// ScalarBroadcast + Transpose, 直接删除transpose，刷新shape、符号信息到brc上
graphStatus EliminateScalarBroadcastWithTranspose(const ComputeGraphPtr &graph, const NodePtr &transpose_node,
                                                  const NodePtr &brc_node, bool &changed) {
  UpdateNodeShapeAndSymbolShape(brc_node, transpose_node);
  GE_ASSERT_SUCCESS(GraphUtils::IsolateNode(transpose_node, {0U}), "Failed to isolate transpose node %s",
                    transpose_node->GetName().c_str());
  GE_ASSERT_GRAPH_SUCCESS(GraphUtils::RemoveJustNode(graph, transpose_node), "Failed to remove transpose node %s",
                          transpose_node->GetName().c_str());
  changed = true;
  return GRAPH_SUCCESS;
}

// 处理 Fill + Transpose 消除（当前仅支持scalar value）
graphStatus EliminateFillWithTranspose(const ComputeGraphPtr &graph, const NodePtr &transpose_node,
                                       const NodePtr &fill_node, bool &changed) {
  // 获取value的shape
  const auto &fill_inputs = fill_node->GetInDataNodes();
  if (fill_inputs.size() < 2UL) {
    return GRAPH_SUCCESS;
  }

  auto value_node = fill_inputs.at(1U);
  const auto &value_shape = value_node->GetOpDesc()->GetOutputDesc(0U).GetShape();
  const auto &value_dims = value_shape.GetDims();

  // 只支持value是标量或者1dtensor的情况
  if (!value_dims.empty() && !(value_dims.size() == 1UL)) {
    // value是张量，暂不支持
    return GRAPH_SUCCESS;
  }

  UpdateNodeShapeAndSymbolShape(fill_node, transpose_node);
  GE_ASSERT_SUCCESS(GraphUtils::IsolateNode(transpose_node, {0U}), "Failed to isolate transpose node %s",
                    transpose_node->GetName().c_str());
  GE_ASSERT_GRAPH_SUCCESS(GraphUtils::RemoveJustNode(graph, transpose_node), "Failed to remove transpose node %s",
                          transpose_node->GetName().c_str());
  changed = true;
  return GRAPH_SUCCESS;
}

}  // namespace

graphStatus TransposeWithBroadcastEliminatePass::Run(const ComputeGraphPtr &graph, bool &changed) {
  GE_CHECK_NOTNULL(graph);

  for (const auto &transpose_node : graph->GetDirectNode()) {
    GE_ASSERT_NOTNULL(transpose_node, "Transpose node is null");

    // 检查是否为Transpose类型
    if (kTransposeOpTypes.find(transpose_node->GetType()) == kTransposeOpTypes.end()) {
      continue;
    }

    auto brc_node = transpose_node->GetInDataNodes().at(0U);
    GE_ASSERT_NOTNULL(brc_node, "Input node of transpose %s is null", transpose_node->GetName().c_str());
    const auto &brc_type = brc_node->GetType();

    // 处理ScalarBroadcast类型 (ZerosLike, OnesLike)
    if (kScalarBroadcastOpTypes.find(brc_type) != kScalarBroadcastOpTypes.end()) {
      GE_ASSERT_SUCCESS(EliminateScalarBroadcastWithTranspose(graph, transpose_node, brc_node, changed),
                        "Failed to eliminate %s with transpose %s", brc_type.c_str(),
                        transpose_node->GetName().c_str());
      GELOGD("TransposeWithBroadcastEliminatePass: eliminated %s + Transpose[%s]", brc_type.c_str(),
             transpose_node->GetName().c_str());
      continue;
    }

    // 处理 Fill 类型（仅支持 scalar value）
    if (brc_type == kFillOpType) {
      bool single_changed = false;
      GE_ASSERT_SUCCESS(EliminateFillWithTranspose(graph, transpose_node, brc_node, single_changed),
                        "Failed to eliminate Fill with transpose %s", transpose_node->GetName().c_str());

      if (single_changed) {
        changed = true;
        GELOGD("TransposeWithBroadcastEliminatePass: eliminated Fill + Transpose[%s]",
               transpose_node->GetName().c_str());
      }
    }
  }

  return GRAPH_SUCCESS;
}

}  // namespace ge
