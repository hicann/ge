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

#include <memory>
#include <stdexcept>
#include <vector>

#include "common/om2/codegen/task_args_manager/om2_task_node_map.h"
#include "graph/compute_graph.h"
#include "graph/node.h"
#include "graph/debug/ge_op_types.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/tensor_utils.h"

namespace ge {
namespace om2 {
namespace {

class TaskNodeMapTest : public testing::Test {
 protected:
  struct GraphFixture {
    ComputeGraphPtr graph;
    int64_t first_id;
    int64_t second_id;
  };

  GraphFixture BuildGraph() {
    auto graph = std::make_shared<ComputeGraph>("test_graph");
    GeTensorDesc tensor_desc(GeShape({1}), FORMAT_ND, DT_FLOAT);
    (void)TensorUtils::SetSize(tensor_desc, 4);

    auto first_op = std::make_shared<OpDesc>("first_node", DATA);
    (void)first_op->AddInputDesc(tensor_desc);
    (void)first_op->AddOutputDesc(tensor_desc);
    auto first_node = graph->AddNode(first_op);
    EXPECT_NE(first_node, nullptr);

    auto second_op = std::make_shared<OpDesc>("second_node", NETOUTPUT);
    (void)second_op->AddInputDesc(tensor_desc);
    auto second_node = graph->AddNode(second_op);
    EXPECT_NE(second_node, nullptr);

    if ((first_node == nullptr) || (second_node == nullptr)) {
      return {graph, -1, -1};
    }

    GraphUtils::AddEdge(first_node->GetOutDataAnchor(0), second_node->GetInDataAnchor(0));
    graph->TopologicalSorting();

    return {graph, first_node->GetOpDesc()->GetId(), second_node->GetOpDesc()->GetId()};
  }
};

TEST_F(TaskNodeMapTest, InitAndFindRelationWorks) {
  auto fixture = BuildGraph();
  TaskNodeMap map;

  ASSERT_EQ(map.Init(fixture.graph, 2U), SUCCESS);
  ASSERT_EQ(map.AddRelation(0U, fixture.first_id), SUCCESS);
  ASSERT_EQ(map.AddRelation(1U, fixture.second_id), SUCCESS);

  const auto &first_info = map.FindNodeByTaskIndex(0U);
  EXPECT_EQ(first_info.node_id, fixture.first_id);
  ASSERT_NE(first_info.node, nullptr);
  EXPECT_EQ(first_info.node->GetName(), "first_node");

  const auto &second_info = map.FindNodeByTaskIndex(1U);
  EXPECT_EQ(second_info.node_id, fixture.second_id);
  ASSERT_NE(second_info.node, nullptr);
  EXPECT_EQ(second_info.node->GetName(), "second_node");

  const auto &tasks_for_first = map.FindTasksByNodeId(fixture.first_id);
  ASSERT_EQ(tasks_for_first.size(), 1U);
  EXPECT_EQ(tasks_for_first[0], 0U);

  const auto &tasks_for_second = map.FindTasksByNodeId(fixture.second_id);
  ASSERT_EQ(tasks_for_second.size(), 1U);
  EXPECT_EQ(tasks_for_second[0], 1U);

  const auto &empty_tasks = map.FindTasksByNodeId(999);
  EXPECT_TRUE(empty_tasks.empty());
}

TEST_F(TaskNodeMapTest, AddRelationRejectsInvalidNodeId) {
  auto fixture = BuildGraph();
  TaskNodeMap map;

  ASSERT_EQ(map.Init(fixture.graph, 1U), SUCCESS);
  EXPECT_NE(map.AddRelation(0U, 999), SUCCESS);
}

TEST_F(TaskNodeMapTest, AddRelationRejectsTaskIndexOutOfRange) {
  auto fixture = BuildGraph();
  TaskNodeMap map;

  ASSERT_EQ(map.Init(fixture.graph, 1U), SUCCESS);
  EXPECT_NE(map.AddRelation(1U, fixture.first_id), SUCCESS);
}

TEST_F(TaskNodeMapTest, FindNodeByTaskIndexOutOfRangeThrows) {
  auto fixture = BuildGraph();
  TaskNodeMap map;

  ASSERT_EQ(map.Init(fixture.graph, 1U), SUCCESS);
  ASSERT_EQ(map.AddRelation(0U, fixture.first_id), SUCCESS);

  EXPECT_THROW((void)map.FindNodeByTaskIndex(1U), std::out_of_range);
}

}  // namespace
}  // namespace om2
}  // namespace ge
