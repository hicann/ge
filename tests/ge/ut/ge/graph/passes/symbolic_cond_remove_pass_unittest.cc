/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include "graph/passes/symbolic/symbolic_cond_remove_pass.h"
#include "graph/compute_graph.h"
#include "graph/op_desc.h"
#include "graph/utils/graph_utils.h"
#include "graph_builder_utils.h"

namespace ge {
class UtestSymbolicCondRemovePass : public testing::Test {
 protected:
  void SetUp() {}
  void TearDown() {}
};

TEST_F(UtestSymbolicCondRemovePass, run_non_cond_node_success) {
  std::vector<GeTensor> graph_inputs;
  SymbolicCondRemovePass pass(graph_inputs);
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph");
  GeTensorDesc tensor_desc(GeShape({1}), FORMAT_NCHW, DT_FLOAT);
  auto op_desc = std::make_shared<OpDesc>("add", ADD);
  op_desc->AddInputDesc(tensor_desc);
  op_desc->AddOutputDesc(tensor_desc);
  auto node = graph->AddNode(op_desc);
  EXPECT_EQ(pass.Run(node), SUCCESS);
}

TEST_F(UtestSymbolicCondRemovePass, run_null_node_failed) {
  std::vector<GeTensor> graph_inputs;
  SymbolicCondRemovePass pass(graph_inputs);
  NodePtr node = nullptr;
  EXPECT_NE(pass.Run(node), SUCCESS);
}

TEST_F(UtestSymbolicCondRemovePass, run_if_node_cond_not_data_success) {
  std::vector<GeTensor> graph_inputs;
  SymbolicCondRemovePass pass(graph_inputs);
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph");
  GeTensorDesc tensor_desc(GeShape({1}), FORMAT_NCHW, DT_FLOAT);
  auto const_desc = std::make_shared<OpDesc>("const", CONSTANTOP);
  const_desc->AddOutputDesc(tensor_desc);
  auto const_node = graph->AddNode(const_desc);

  auto if_desc = std::make_shared<OpDesc>("if", "If");
  if_desc->AddInputDesc(tensor_desc);
  if_desc->AddOutputDesc(tensor_desc);
  auto if_node = graph->AddNode(if_desc);
  (void)GraphUtils::AddEdge(const_node->GetOutDataAnchor(0), if_node->GetInDataAnchor(0));

  EXPECT_EQ(pass.Run(if_node), SUCCESS);
}

TEST_F(UtestSymbolicCondRemovePass, run_case_node_cond_not_data_success) {
  std::vector<GeTensor> graph_inputs;
  SymbolicCondRemovePass pass(graph_inputs);
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph");
  GeTensorDesc tensor_desc(GeShape({1}), FORMAT_NCHW, DT_FLOAT);
  auto const_desc = std::make_shared<OpDesc>("const", CONSTANTOP);
  const_desc->AddOutputDesc(tensor_desc);
  auto const_node = graph->AddNode(const_desc);

  auto case_desc = std::make_shared<OpDesc>("case", "Case");
  case_desc->AddInputDesc(tensor_desc);
  case_desc->AddOutputDesc(tensor_desc);
  auto case_node = graph->AddNode(case_desc);
  (void)GraphUtils::AddEdge(const_node->GetOutDataAnchor(0), case_node->GetInDataAnchor(0));

  EXPECT_EQ(pass.Run(case_node), SUCCESS);
}

TEST_F(UtestSymbolicCondRemovePass, run_if_node_no_cond_input_failed) {
  std::vector<GeTensor> graph_inputs;
  SymbolicCondRemovePass pass(graph_inputs);
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph");
  GeTensorDesc tensor_desc(GeShape({1}), FORMAT_NCHW, DT_FLOAT);
  auto if_desc = std::make_shared<OpDesc>("if", "If");
  if_desc->AddInputDesc(tensor_desc);
  if_desc->AddOutputDesc(tensor_desc);
  auto if_node = graph->AddNode(if_desc);

  EXPECT_NE(pass.Run(if_node), SUCCESS);
}
}  // namespace ge
