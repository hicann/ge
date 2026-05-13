/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>

#include "graph/build/stream/dag_stream_allocator_pass.h"
#include "graph/build/stream/dag_adapter.h"
#include "framework/common/ge_inner_error_codes.h"
#include "external/graph/graph.h"
#include "external/graph/operator.h"
#include "register/custom_pass_context_impl.h"

namespace minidag {

namespace {
// 测试辅助：构建简单的 Graph（仅包含节点和控制边）
ge::ConstGraphPtr BuildGraphWithNodes() {
  auto graph = std::make_shared<ge::Graph>("test_graph");

  ge::Operator data1_op("data1", "Data");
  ge::Operator add1_op("add1", "Add");
  ge::Operator netoutput_op("NetOutput", "NetOutput");

  (void)graph->AddNodeByOp(data1_op);
  (void)graph->AddNodeByOp(add1_op);
  (void)graph->AddNodeByOp(netoutput_op);

  return graph;
}

ge::ConstGraphPtr BuildGraphWithControlEdge() {
  auto graph = std::make_shared<ge::Graph>("control_edge_graph");

  ge::Operator data1_op("data1", "Data");
  ge::Operator add1_op("add1", "Add");
  ge::Operator relu1_op("relu1", "Relu");
  ge::Operator netoutput_op("NetOutput", "NetOutput");

  auto data1 = graph->AddNodeByOp(data1_op);
  auto add1 = graph->AddNodeByOp(add1_op);
  auto relu1 = graph->AddNodeByOp(relu1_op);
  auto netoutput = graph->AddNodeByOp(netoutput_op);

  graph->AddControlEdge(data1, relu1);
  graph->AddControlEdge(add1, netoutput);

  return graph;
}
}  // namespace

// --------------------
// 场景 A：RunMiniDAGStreamPass 正常执行
// --------------------

/**
 * 场景 A1: 正常图执行返回 SUCCESS
 */
TEST(DagStreamAllocatorPassTest, RunPass_Success) {
  auto graph = BuildGraphWithControlEdge();
  ASSERT_NE(graph, nullptr);

  ge::StreamPassContext context(0);  // current_max_stream_id = 0
  auto ret = RunMiniDAGStreamPass(graph, context);
  EXPECT_EQ(ret, ge::SUCCESS);
}

/**
 * 场景 A2: 空图（nullptr）返回 FAILED
 */
TEST(DagStreamAllocatorPassTest, RunPass_NullGraph) {
  ge::StreamPassContext context(0);  // current_max_stream_id = 0
  auto ret = RunMiniDAGStreamPass(nullptr, context);
  EXPECT_EQ(ret, ge::FAILED);
}

/**
 * 场景 A3: 多次执行均返回 SUCCESS
 */
TEST(DagStreamAllocatorPassTest, RunPass_MultipleExecution) {
  auto graph1 = BuildGraphWithNodes();
  auto graph2 = BuildGraphWithControlEdge();
  ASSERT_NE(graph1, nullptr);
  ASSERT_NE(graph2, nullptr);

  ge::StreamPassContext context1(0), context2(0);

  auto ret1 = RunMiniDAGStreamPass(graph1, context1);
  auto ret2 = RunMiniDAGStreamPass(graph2, context2);

  EXPECT_EQ(ret1, ge::SUCCESS);
  EXPECT_EQ(ret2, ge::SUCCESS);
}

/**
 * 场景 A4: 仅包含 Data/NetOutput 的图返回 SUCCESS（空 DAG）
 */
TEST(DagStreamAllocatorPassTest, RunPass_EmptyDAG) {
  auto graph = BuildGraphWithNodes();
  ASSERT_NE(graph, nullptr);

  ge::StreamPassContext context(0);  // current_max_stream_id = 0
  auto ret = RunMiniDAGStreamPass(graph, context);
  EXPECT_EQ(ret, ge::SUCCESS);
}

}  // namespace minidag