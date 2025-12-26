/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <memory>

#include "eager_style_graph_builder/esb_funcs_cpp.h"
#include "ge/fusion/pattern.h"
#include "graph/utils/graph_utils_ex.h"
#include "graph/utils/node_adapter.h"

namespace ge {
namespace fusion {
class UtestPatternGraph : public testing::Test {
 protected:
  void SetUp() {
  }

  void TearDown() {
  }
};
TEST_F(UtestPatternGraph, CreatePattern) {
  es::Graph graph_builder("pattern");
  auto input0 = graph_builder.CreateInput(0);
  auto input1 = graph_builder.CreateInput(1);
  auto y = input0 + input1;
  graph_builder.SetOutput(y, 0);
  auto graph = graph_builder.Build();

  auto pattern = std::make_shared<Pattern>(std::move(*graph));
  auto pattern_graph = pattern->GetGraph();
  auto pattern_compute_graph = GraphUtilsEx::GetComputeGraph(pattern_graph);
  EXPECT_EQ(pattern_compute_graph->GetDirectNodesSize(), 4);
}

TEST_F(UtestPatternGraph, CaptrueTensor) {
  es::Graph graph_builder("pattern");
  auto input0 = graph_builder.CreateInput(0);
  auto input1 = graph_builder.CreateInput(1);
  auto y = input0 + input1;
  auto x = y - input1;
  graph_builder.SetOutput(x, 0);
  auto graph = graph_builder.Build();
  auto pattern_compute_graph = GraphUtilsEx::GetComputeGraph(*graph);

  auto y_producer = pattern_compute_graph->FindFirstNodeMatchType("Add");
  auto x_producer = pattern_compute_graph->FindFirstNodeMatchType("Sub");

  // capture
  auto pattern = std::make_shared<Pattern>(std::move(*graph));
  pattern->CaptureTensor({NodeAdapter::Node2GNode(y_producer), 0})
      .CaptureTensor({NodeAdapter::Node2GNode(x_producer), 0});

  std::vector<NodeIo> node_outputs;
  EXPECT_EQ(pattern->GetCapturedTensors(node_outputs), GRAPH_SUCCESS);

  EXPECT_EQ(node_outputs.size(), 2);
  EXPECT_EQ(NodeAdapter::GNode2Node(node_outputs[0].node), y_producer);
  EXPECT_EQ(NodeAdapter::GNode2Node(node_outputs[1].node), x_producer);
}
} // namespace fusion
} // namespace ge

