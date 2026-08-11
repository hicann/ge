/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "core/executor/sequential/execution_data/sequential_execution_data_builder.h"
#include "core/executor/topological/execution_data/topological_execution_data_builder.h"
#include "core/executor/priority_topological/execution_data/priority_topological_execution_data_builder.h"
#include "core/executor/multi_thread_topological/execution_data/multi_thread_execution_data_builder.h"
#include "core/executor/multi_thread_topological/execution_data/multi_thread_exe_graph_resource_guard.h"
#include "core/executor/multi_thread_topological/execution_data/free_launch_relation.h"
#include "core/executor/multi_thread_topological/execution_data/multi_thread_execution_data.h"
#include <gtest/gtest.h>
#include <memory>

#include "common/bg_test.h"
#include "exe_graph/lowering/value_holder.h"
#include "faker/exe_graph_model_level_data_faker.h"
#include "framework/runtime/executor_option/multi_thread_executor_option.h"

namespace gert {
namespace {
struct RelationGraph {
  ge::ExecuteGraphPtr graph;
  std::vector<ge::FastNode *> nodes;
};

RelationGraph BuildRelationGraph(const std::vector<const char *> &node_types) {
  RelationGraph result;
  for (const auto node_type : node_types) {
    auto holder = bg::ValueHolder::CreateVoid<bg::ValueHolder>(node_type, {});
    result.nodes.emplace_back(holder->GetFastNode());
  }
  auto frame = bg::ValueHolder::PopGraphFrame();
  if (frame != nullptr) {
    result.graph = frame->GetExecuteGraph();
  }
  return result;
}

std::vector<Node> BuildExecutionNodes(const std::vector<NodeIdentity> &node_ids) {
  std::vector<Node> nodes(node_ids.size());
  for (size_t i = 0U; i < node_ids.size(); ++i) {
    nodes[i].node_id = node_ids[i];
  }
  return nodes;
}

ge::graphStatus BuildRelationCsr(const ge::ExecuteGraphPtr &graph,
                                 const std::vector<std::pair<ge::FastNode *, Node *>> &mapping,
                                 MultiThreadResourceGuard &resource_guard) {
  GraphExecutorBuilder executor_builder(ModelLevelData{}, graph, nullptr);
  MultiThreadExecutionDataBuilder builder(executor_builder);
  return builder.BuildFreeLaunchRelationCsr(mapping, resource_guard);
}
}  // namespace

class ExecutionDataBuilderUT : public testing::Test {
 protected:
  void SetUp() override {
    executor_builder = std::make_unique<GraphExecutorBuilder>(ModelLevelData{}, nullptr, nullptr);
  }
  std::unique_ptr<GraphExecutorBuilder> executor_builder;
};

TEST_F(ExecutionDataBuilderUT, SequentialBuilder_GetExecuteFunc_Correct) {
  SequentialExecutionDataBuilder execution_data_builder(*executor_builder);
  auto funcs = execution_data_builder.GetExecuteFunc();
  EXPECT_EQ(funcs.first, SequentialExecute);
  EXPECT_EQ(funcs.second, SequentialExecuteWithCallback);
}
TEST_F(ExecutionDataBuilderUT, TopologicalBuilder_GetExecuteFunc_Correct) {
  TopologicalExecutionDataBuilder execution_data_builder(*executor_builder);
  auto funcs = execution_data_builder.GetExecuteFunc();
  EXPECT_EQ(funcs.first, TopologicalExecute);
  EXPECT_EQ(funcs.second, TopologicalExecuteWithCallback);
}
TEST_F(ExecutionDataBuilderUT, PriorityTopologicalBuilder_GetExecuteFunc_Correct) {
  PriorityTopologicalExecutionDataBuilder execution_data_builder(*executor_builder);
  auto funcs = execution_data_builder.GetExecuteFunc();
  EXPECT_EQ(funcs.first, PriorityTopologicalExecute);
  EXPECT_EQ(funcs.second, PriorityTopologicalExecuteWithCallback);
}

TEST_F(ExecutionDataBuilderUT, MultiThreadBuilder_GetExecuteFunc_Correct) {
  MultiThreadExecutionDataBuilder execution_data_builder(*executor_builder);
  auto funcs = execution_data_builder.GetExecuteFunc();
  EXPECT_EQ(funcs.first, MultiThreadTopologicalExecute);
  EXPECT_EQ(funcs.second, MultiThreadTopologicalExecuteWithCallback);
}

class FreeLaunchRelationCsrUT : public bg::BgTestAutoCreateFrame {};

TEST_F(FreeLaunchRelationCsrUT, BuilderCreatesDeterministicDeduplicatedOneToManyAndManyToOneCsr) {
  auto relation_graph =
      BuildRelationGraph({"FreeMemoryHoldAddr", "ExecuteOpLaunch", "FreeMemHbmHoldAddr", "ExecuteOpLaunch"});
  ASSERT_NE(relation_graph.graph, nullptr);
  auto free0 = relation_graph.nodes[0];
  auto launch0 = relation_graph.nodes[1];
  auto free1 = relation_graph.nodes[2];
  auto launch1 = relation_graph.nodes[3];
  ASSERT_TRUE(relation_graph.graph->SetExtAttr(
      kFreeLaunchRelationsAttr,
      FreeLaunchRelations{{free1, launch0}, {free0, launch1}, {free0, launch0}, {free0, launch1}}));

  auto execution_nodes = BuildExecutionNodes({2U, 0U, 3U, 1U});
  std::vector<std::pair<ge::FastNode *, Node *>> mapping{{free0, &execution_nodes[0]},
                                                         {launch0, &execution_nodes[1]},
                                                         {free1, &execution_nodes[2]},
                                                         {launch1, &execution_nodes[3]}};
  MultiThreadResourceGuard resource_guard;
  ASSERT_EQ(BuildRelationCsr(relation_graph.graph, mapping, resource_guard), ge::GRAPH_SUCCESS);

  const auto &csr = resource_guard.GetFreeLaunchRelationCsr();
  ASSERT_EQ(csr.node_num, 4U);
  ASSERT_EQ(csr.relation_num, 3U);
  ASSERT_NE(csr.offsets, nullptr);
  ASSERT_NE(csr.launch_ids, nullptr);
  EXPECT_EQ(std::vector<NodeIdentity>(csr.offsets, csr.offsets + 5U), std::vector<NodeIdentity>({0U, 0U, 0U, 2U, 3U}));
  EXPECT_EQ(std::vector<NodeIdentity>(csr.launch_ids, csr.launch_ids + 3U), std::vector<NodeIdentity>({0U, 1U, 0U}));

  const auto free0_launches = csr.GetLaunchIds(2U);
  ASSERT_EQ(free0_launches.size, 2U);
  EXPECT_EQ(std::vector<NodeIdentity>(free0_launches.data, free0_launches.data + free0_launches.size),
            std::vector<NodeIdentity>({0U, 1U}));
  const auto free1_launches = csr.GetLaunchIds(3U);
  ASSERT_EQ(free1_launches.size, 1U);
  EXPECT_EQ(free1_launches.data[0], 0U);
}

TEST_F(FreeLaunchRelationCsrUT, EmptyRelationsCreateEmptyRangeForEveryExecutionNode) {
  auto relation_graph = BuildRelationGraph({"FreeMemoryHoldAddr", "ExecuteOpLaunch"});
  ASSERT_NE(relation_graph.graph, nullptr);
  auto execution_nodes = BuildExecutionNodes({0U, 1U});
  std::vector<std::pair<ge::FastNode *, Node *>> mapping{{relation_graph.nodes[0], &execution_nodes[0]},
                                                         {relation_graph.nodes[1], &execution_nodes[1]}};
  MultiThreadResourceGuard resource_guard;
  ASSERT_EQ(BuildRelationCsr(relation_graph.graph, mapping, resource_guard), ge::GRAPH_SUCCESS);

  const auto &csr = resource_guard.GetFreeLaunchRelationCsr();
  ASSERT_EQ(csr.node_num, 2U);
  ASSERT_EQ(csr.relation_num, 0U);
  ASSERT_NE(csr.offsets, nullptr);
  EXPECT_EQ(csr.launch_ids, nullptr);
  EXPECT_EQ(std::vector<NodeIdentity>(csr.offsets, csr.offsets + 3U), std::vector<NodeIdentity>({0U, 0U, 0U}));
  for (NodeIdentity node_id = 0U; node_id < csr.node_num; ++node_id) {
    const auto range = csr.GetLaunchIds(node_id);
    EXPECT_EQ(range.data, nullptr);
    EXPECT_EQ(range.size, 0U);
  }
}

TEST_F(FreeLaunchRelationCsrUT, NullFreeRelationNodeFailsBuild) {
  auto relation_graph = BuildRelationGraph({"FreeMemoryHoldAddr", "ExecuteOpLaunch"});
  ASSERT_NE(relation_graph.graph, nullptr);
  ASSERT_TRUE(relation_graph.graph->SetExtAttr(kFreeLaunchRelationsAttr,
                                               FreeLaunchRelations{{nullptr, relation_graph.nodes[1]}}));
  auto execution_nodes = BuildExecutionNodes({0U, 1U});
  std::vector<std::pair<ge::FastNode *, Node *>> mapping{{relation_graph.nodes[0], &execution_nodes[0]},
                                                         {relation_graph.nodes[1], &execution_nodes[1]}};
  MultiThreadResourceGuard resource_guard;
  EXPECT_NE(BuildRelationCsr(relation_graph.graph, mapping, resource_guard), ge::GRAPH_SUCCESS);
}

TEST_F(FreeLaunchRelationCsrUT, CrossGraphLaunchRelationFailsBuild) {
  auto relation_graph = BuildRelationGraph({"FreeMemoryHoldAddr", "ExecuteOpLaunch"});
  ASSERT_NE(relation_graph.graph, nullptr);
  ASSERT_NE(bg::ValueHolder::PushGraphFrame(), nullptr);
  auto other_graph = BuildRelationGraph({"ExecuteOpLaunch"});
  ASSERT_NE(other_graph.graph, nullptr);
  ASSERT_TRUE(relation_graph.graph->SetExtAttr(kFreeLaunchRelationsAttr,
                                               FreeLaunchRelations{{relation_graph.nodes[0], other_graph.nodes[0]}}));
  auto execution_nodes = BuildExecutionNodes({0U, 1U});
  std::vector<std::pair<ge::FastNode *, Node *>> mapping{{relation_graph.nodes[0], &execution_nodes[0]},
                                                         {other_graph.nodes[0], &execution_nodes[1]}};
  MultiThreadResourceGuard resource_guard;
  EXPECT_NE(BuildRelationCsr(relation_graph.graph, mapping, resource_guard), ge::GRAPH_SUCCESS);
}

TEST_F(FreeLaunchRelationCsrUT, CrossGraphFreeRelationFailsBuild) {
  auto relation_graph = BuildRelationGraph({"FreeMemoryHoldAddr", "ExecuteOpLaunch"});
  ASSERT_NE(relation_graph.graph, nullptr);
  ASSERT_NE(bg::ValueHolder::PushGraphFrame(), nullptr);
  auto other_graph = BuildRelationGraph({"FreeMemoryHoldAddr"});
  ASSERT_NE(other_graph.graph, nullptr);
  ASSERT_TRUE(relation_graph.graph->SetExtAttr(kFreeLaunchRelationsAttr,
                                               FreeLaunchRelations{{other_graph.nodes[0], relation_graph.nodes[1]}}));
  auto execution_nodes = BuildExecutionNodes({0U, 1U});
  std::vector<std::pair<ge::FastNode *, Node *>> mapping{{other_graph.nodes[0], &execution_nodes[0]},
                                                         {relation_graph.nodes[1], &execution_nodes[1]}};
  MultiThreadResourceGuard resource_guard;
  EXPECT_NE(BuildRelationCsr(relation_graph.graph, mapping, resource_guard), ge::GRAPH_SUCCESS);
}

TEST_F(FreeLaunchRelationCsrUT, BuilderCollectsRelationFromOwningSubgraph) {
  auto subgraph_call = bg::ValueHolder::CreateVoid<bg::ValueHolder>("SubgraphCall", {});
  ASSERT_NE(bg::ValueHolder::PushGraphFrame(subgraph_call, "Subgraph"), nullptr);
  auto relation_graph = BuildRelationGraph({"FreeMemoryHoldAddr", "ExecuteOpLaunch"});
  ASSERT_NE(relation_graph.graph, nullptr);
  auto root_frame = bg::ValueHolder::PopGraphFrame();
  ASSERT_NE(root_frame, nullptr);
  auto root_graph = root_frame->GetExecuteGraph();
  ASSERT_NE(root_graph, nullptr);
  ASSERT_TRUE(relation_graph.graph->SetExtAttr(
      kFreeLaunchRelationsAttr, FreeLaunchRelations{{relation_graph.nodes[0], relation_graph.nodes[1]}}));
  auto execution_nodes = BuildExecutionNodes({0U, 1U});
  std::vector<std::pair<ge::FastNode *, Node *>> mapping{{relation_graph.nodes[0], &execution_nodes[0]},
                                                         {relation_graph.nodes[1], &execution_nodes[1]}};
  MultiThreadResourceGuard resource_guard;
  ASSERT_EQ(BuildRelationCsr(root_graph, mapping, resource_guard), ge::GRAPH_SUCCESS);

  const auto &csr = resource_guard.GetFreeLaunchRelationCsr();
  ASSERT_EQ(csr.node_num, 2U);
  ASSERT_EQ(csr.relation_num, 1U);
  EXPECT_EQ(std::vector<NodeIdentity>(csr.offsets, csr.offsets + 3U), std::vector<NodeIdentity>({0U, 1U, 1U}));
  ASSERT_NE(csr.launch_ids, nullptr);
  EXPECT_EQ(csr.launch_ids[0], 1U);
}

TEST_F(FreeLaunchRelationCsrUT, MultiThreadBuildCollectsNestedOwnerFromMainWithoutSiblingRelations) {
  auto init_node = bg::ValueHolder::CreateVoid<bg::ValueHolder>("Init", {});
  ASSERT_NE(bg::ValueHolder::PushGraphFrame(init_node, "Init"), nullptr);
  auto init_graph = BuildRelationGraph({"FreeMemoryHoldAddr", "ExecuteOpLaunch"});
  ASSERT_NE(init_graph.graph, nullptr);
  ASSERT_TRUE(init_graph.graph->SetExtAttr(kFreeLaunchRelationsAttr,
                                           FreeLaunchRelations{{init_graph.nodes[0], init_graph.nodes[1]}}));

  auto deinit_node = bg::ValueHolder::CreateVoid<bg::ValueHolder>("DeInit", {});
  ASSERT_NE(bg::ValueHolder::PushGraphFrame(deinit_node, "DeInit"), nullptr);
  auto deinit_graph = BuildRelationGraph({"FreeMemoryHoldAddr", "ExecuteOpLaunch"});
  ASSERT_NE(deinit_graph.graph, nullptr);
  ASSERT_TRUE(deinit_graph.graph->SetExtAttr(kFreeLaunchRelationsAttr,
                                             FreeLaunchRelations{{deinit_graph.nodes[0], deinit_graph.nodes[1]}}));

  auto main_node = bg::ValueHolder::CreateVoid<bg::ValueHolder>("Main", {});
  ASSERT_NE(bg::ValueHolder::PushGraphFrame(main_node, "Main"), nullptr);
  auto nested_call = bg::ValueHolder::CreateVoid<bg::ValueHolder>("SyncStream", {});
  ASSERT_NE(bg::ValueHolder::PushGraphFrame(nested_call, "NestedOwner"), nullptr);
  auto nested_graph = BuildRelationGraph({"FreeMemoryHoldAddr", "ExecuteOpLaunch"});
  ASSERT_NE(nested_graph.graph, nullptr);
  ASSERT_TRUE(nested_graph.graph->SetExtAttr(kFreeLaunchRelationsAttr,
                                             FreeLaunchRelations{{nested_graph.nodes[0], nested_graph.nodes[1]}}));
  auto main_frame = bg::ValueHolder::PopGraphFrame();
  ASSERT_NE(main_frame, nullptr);
  auto main_graph = main_frame->GetExecuteGraph();
  ASSERT_NE(main_graph, nullptr);
  auto root_frame = bg::ValueHolder::PopGraphFrame();
  ASSERT_NE(root_frame, nullptr);
  auto root_graph = root_frame->GetExecuteGraph();
  ASSERT_NE(root_graph, nullptr);
  ASSERT_EQ(root_graph->TopologicalSorting(), ge::GRAPH_SUCCESS);

  auto model_level_data = ExeGraphModelLevelDataFaker(root_graph).Build();
  GraphExecutorBuilder executor_builder(model_level_data.GetModelLevelData(), main_graph,
                                        &model_level_data.symbols_to_value);
  auto option = MultiThreadExecutorOption(kLeastThreadNumber);
  executor_builder.ExecutorOpt(option);
  MultiThreadExecutionDataBuilder builder(executor_builder);

  auto resource_guard = builder.Build();
  ASSERT_NE(resource_guard, nullptr);
  const auto &csr = static_cast<MultiThreadResourceGuard *>(resource_guard.get())->GetFreeLaunchRelationCsr();
  EXPECT_EQ(csr.relation_num, 1U);
}

TEST_F(FreeLaunchRelationCsrUT, UnmappedLaunchRelationNodeFailsBuild) {
  auto relation_graph = BuildRelationGraph({"FreeMemoryHoldAddr", "ExecuteOpLaunch"});
  ASSERT_NE(relation_graph.graph, nullptr);
  ASSERT_TRUE(relation_graph.graph->SetExtAttr(
      kFreeLaunchRelationsAttr, FreeLaunchRelations{{relation_graph.nodes[0], relation_graph.nodes[1]}}));
  auto execution_nodes = BuildExecutionNodes({0U});
  std::vector<std::pair<ge::FastNode *, Node *>> mapping{{relation_graph.nodes[0], &execution_nodes[0]}};
  MultiThreadResourceGuard resource_guard;
  EXPECT_NE(BuildRelationCsr(relation_graph.graph, mapping, resource_guard), ge::GRAPH_SUCCESS);
}

TEST_F(FreeLaunchRelationCsrUT, UnmappedFreeRelationNodeFailsBuild) {
  auto relation_graph = BuildRelationGraph({"FreeMemoryHoldAddr", "ExecuteOpLaunch"});
  ASSERT_NE(relation_graph.graph, nullptr);
  ASSERT_TRUE(relation_graph.graph->SetExtAttr(
      kFreeLaunchRelationsAttr, FreeLaunchRelations{{relation_graph.nodes[0], relation_graph.nodes[1]}}));
  auto execution_nodes = BuildExecutionNodes({0U});
  std::vector<std::pair<ge::FastNode *, Node *>> mapping{{relation_graph.nodes[1], &execution_nodes[0]}};
  MultiThreadResourceGuard resource_guard;
  EXPECT_NE(BuildRelationCsr(relation_graph.graph, mapping, resource_guard), ge::GRAPH_SUCCESS);
}

TEST_F(FreeLaunchRelationCsrUT, NullMappedExecutionNodeFailsBuild) {
  auto relation_graph = BuildRelationGraph({"FreeMemoryHoldAddr", "ExecuteOpLaunch"});
  ASSERT_NE(relation_graph.graph, nullptr);
  ASSERT_TRUE(relation_graph.graph->SetExtAttr(
      kFreeLaunchRelationsAttr, FreeLaunchRelations{{relation_graph.nodes[0], relation_graph.nodes[1]}}));
  auto execution_nodes = BuildExecutionNodes({0U, 1U});
  std::vector<std::pair<ge::FastNode *, Node *>> mapping{{relation_graph.nodes[0], nullptr},
                                                         {relation_graph.nodes[1], &execution_nodes[1]}};
  MultiThreadResourceGuard resource_guard;
  EXPECT_NE(BuildRelationCsr(relation_graph.graph, mapping, resource_guard), ge::GRAPH_SUCCESS);
}

TEST_F(FreeLaunchRelationCsrUT, NonHoldAddressFreeRelationFailsBuild) {
  auto relation_graph = BuildRelationGraph({"FreeMemory", "ExecuteOpLaunch"});
  ASSERT_NE(relation_graph.graph, nullptr);
  ASSERT_TRUE(relation_graph.graph->SetExtAttr(
      kFreeLaunchRelationsAttr, FreeLaunchRelations{{relation_graph.nodes[0], relation_graph.nodes[1]}}));
  auto execution_nodes = BuildExecutionNodes({0U, 1U});
  std::vector<std::pair<ge::FastNode *, Node *>> mapping{{relation_graph.nodes[0], &execution_nodes[0]},
                                                         {relation_graph.nodes[1], &execution_nodes[1]}};
  MultiThreadResourceGuard resource_guard;
  EXPECT_NE(BuildRelationCsr(relation_graph.graph, mapping, resource_guard), ge::GRAPH_SUCCESS);
}

TEST_F(FreeLaunchRelationCsrUT, NonLaunchTargetRelationFailsBuild) {
  auto relation_graph = BuildRelationGraph({"FreeMemoryHoldAddr", "AllocMemory"});
  ASSERT_NE(relation_graph.graph, nullptr);
  ASSERT_TRUE(relation_graph.graph->SetExtAttr(
      kFreeLaunchRelationsAttr, FreeLaunchRelations{{relation_graph.nodes[0], relation_graph.nodes[1]}}));
  auto execution_nodes = BuildExecutionNodes({0U, 1U});
  std::vector<std::pair<ge::FastNode *, Node *>> mapping{{relation_graph.nodes[0], &execution_nodes[0]},
                                                         {relation_graph.nodes[1], &execution_nodes[1]}};
  MultiThreadResourceGuard resource_guard;
  EXPECT_NE(BuildRelationCsr(relation_graph.graph, mapping, resource_guard), ge::GRAPH_SUCCESS);
}

TEST_F(FreeLaunchRelationCsrUT, OutOfRangeLaunchExecutionIdFailsBuild) {
  auto relation_graph = BuildRelationGraph({"FreeMemoryHoldAddr", "ExecuteOpLaunch"});
  ASSERT_NE(relation_graph.graph, nullptr);
  ASSERT_TRUE(relation_graph.graph->SetExtAttr(
      kFreeLaunchRelationsAttr, FreeLaunchRelations{{relation_graph.nodes[0], relation_graph.nodes[1]}}));
  auto execution_nodes = BuildExecutionNodes({0U, 2U});
  std::vector<std::pair<ge::FastNode *, Node *>> mapping{{relation_graph.nodes[0], &execution_nodes[0]},
                                                         {relation_graph.nodes[1], &execution_nodes[1]}};
  MultiThreadResourceGuard resource_guard;
  EXPECT_NE(BuildRelationCsr(relation_graph.graph, mapping, resource_guard), ge::GRAPH_SUCCESS);
}

TEST_F(FreeLaunchRelationCsrUT, OutOfRangeFreeExecutionIdFailsBuild) {
  auto relation_graph = BuildRelationGraph({"FreeMemoryHoldAddr", "ExecuteOpLaunch"});
  ASSERT_NE(relation_graph.graph, nullptr);
  ASSERT_TRUE(relation_graph.graph->SetExtAttr(
      kFreeLaunchRelationsAttr, FreeLaunchRelations{{relation_graph.nodes[0], relation_graph.nodes[1]}}));
  auto execution_nodes = BuildExecutionNodes({2U, 1U});
  std::vector<std::pair<ge::FastNode *, Node *>> mapping{{relation_graph.nodes[0], &execution_nodes[0]},
                                                         {relation_graph.nodes[1], &execution_nodes[1]}};
  MultiThreadResourceGuard resource_guard;
  EXPECT_NE(BuildRelationCsr(relation_graph.graph, mapping, resource_guard), ge::GRAPH_SUCCESS);
}

TEST_F(FreeLaunchRelationCsrUT, MultiThreadBuildPropagatesNullLaunchRelationFailure) {
  auto relation_graph = BuildRelationGraph({"FreeMemoryHoldAddr", "ExecuteOpLaunch"});
  ASSERT_NE(relation_graph.graph, nullptr);
  ASSERT_TRUE(relation_graph.graph->SetExtAttr(kFreeLaunchRelationsAttr,
                                               FreeLaunchRelations{{relation_graph.nodes[0], nullptr}}));
  auto model_level_data = ExeGraphModelLevelDataFaker(relation_graph.graph).Build();
  GraphExecutorBuilder executor_builder(model_level_data.GetModelLevelData(), relation_graph.graph,
                                        &model_level_data.symbols_to_value);
  auto option = MultiThreadExecutorOption(kLeastThreadNumber);
  executor_builder.ExecutorOpt(option);
  MultiThreadExecutionDataBuilder builder(executor_builder);

  auto resource_guard = builder.Build();
  EXPECT_EQ(resource_guard, nullptr);
}

TEST_F(FreeLaunchRelationCsrUT, MultiThreadBuildPublishesCsrToPreparedScheduler) {
  auto relation_graph = BuildRelationGraph({"FreeMemoryHoldAddr", "ExecuteOpLaunch"});
  ASSERT_NE(relation_graph.graph, nullptr);
  ASSERT_TRUE(relation_graph.graph->SetExtAttr(
      kFreeLaunchRelationsAttr, FreeLaunchRelations{{relation_graph.nodes[0], relation_graph.nodes[1]}}));
  auto model_level_data = ExeGraphModelLevelDataFaker(relation_graph.graph).Build();
  GraphExecutorBuilder executor_builder(model_level_data.GetModelLevelData(), relation_graph.graph,
                                        &model_level_data.symbols_to_value);
  auto option = MultiThreadExecutorOption(kLeastThreadNumber);
  executor_builder.ExecutorOpt(option);
  MultiThreadExecutionDataBuilder builder(executor_builder);

  auto resource_guard = builder.Build();
  ASSERT_NE(resource_guard, nullptr);
  auto multi_thread_guard = static_cast<MultiThreadResourceGuard *>(resource_guard.get());
  const auto &csr = multi_thread_guard->GetFreeLaunchRelationCsr();
  ASSERT_EQ(csr.node_num, 2U);
  ASSERT_EQ(csr.relation_num, 1U);

  auto execution_data = static_cast<const MultiThreadExecutionData *>(resource_guard->GetExecutionData());
  ASSERT_NE(execution_data, nullptr);
  ASSERT_NE(execution_data->scheduler, nullptr);
  bool found_relation = false;
  for (NodeIdentity free_id = 0U; free_id < csr.node_num; ++free_id) {
    const auto expected = csr.GetLaunchIds(free_id);
    const auto actual = execution_data->scheduler->GetLaunchIds(free_id);
    EXPECT_EQ(actual.data, expected.data);
    EXPECT_EQ(actual.size, expected.size);
    found_relation = found_relation || (actual.size > 0U);
  }
  EXPECT_TRUE(found_relation);
}
}  // namespace gert
