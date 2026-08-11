/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "lowering/pass/remove_launch_free_edge.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <utility>
#include <vector>

#include "common/bg_test.h"
#include "common/topo_checker.h"
#include "core/executor/multi_thread_topological/execution_data/free_launch_relation.h"
#include "core/executor/multi_thread_topological/executor/schedule/producer/producers/kernel_tags/critical_section_config.h"
#include "exe_graph/lowering/value_holder.h"
#include "graph/utils/execute_graph_utils.h"
#include "graph/utils/fast_node_utils.h"
#include "register/kernel_registry.h"

namespace gert {
namespace bg {
namespace {
constexpr const char *kRemoveLaunchFreeEdgeAllocAttr = "remove_launch_free_edge_alloc";

UINT32 LaunchResourceProducer(KernelContext *) {
  return ge::GRAPH_SUCCESS;
}

REGISTER_KERNEL(LaunchResourceProducerForRemoveLaunchFreeEdgeUT)
    .RunFunc(LaunchResourceProducer)
    .ConcurrentCriticalSectionKey(kKernelLaunch);

class RemoveLaunchFreeEdgeUT : public BgTestAutoCreateFrame {};

ge::FastNode *FindNodeByType(ge::ExecuteGraph *const graph, const char *const type) {
  auto node = ge::ExecuteGraphUtils::FindFirstNodeMatchType(graph, type);
  EXPECT_NE(node, nullptr) << type;
  return node;
}

size_t CountControlEdges(const ge::FastNode *const src, const ge::FastNode *const dst) {
  return static_cast<size_t>(
      std::count_if(dst->GetAllInControlEdgesRef().begin(), dst->GetAllInControlEdgesRef().end(),
                    [src](const ge::FastEdge *const edge) { return (edge != nullptr) && (edge->src == src); }));
}

bool HasDirectDependency(const ge::FastNode *const src, const ge::FastNode *const dst) {
  const auto &in_nodes = dst->GetAllInNodes();
  return std::find(in_nodes.begin(), in_nodes.end(), src) != in_nodes.end();
}
}  // namespace

TEST_F(RemoveLaunchFreeEdgeUT, LaunchCriticalNodeToFreeEdgeRemoved) {
  auto size = ValueHolder::CreateFeed(0);
  auto allocator = ValueHolder::CreateFeed(1);
  auto alloc = ValueHolder::CreateSingleDataOutput("AllocMemory", {size, allocator});
  auto stream = ValueHolder::CreateFeed(2);
  auto launch = ValueHolder::CreateVoid<ValueHolder>("ExecuteOpLaunch", {stream});
  auto free = ValueHolder::CreateVoidGuarder("FreeMemory", alloc, {});
  ASSERT_TRUE(ValueHolder::AddDependency(launch, free).IsSuccess());

  auto frame = ValueHolder::PopGraphFrame();
  ASSERT_NE(frame, nullptr);
  auto graph = frame->GetExecuteGraph();
  ASSERT_NE(graph, nullptr);

  bool changed = false;
  ASSERT_EQ(RemoveLaunchFreeEdge().Run(graph.get(), changed), ge::GRAPH_SUCCESS);
  ASSERT_TRUE(changed);

  auto alloc_node = FindNodeByType(graph.get(), "AllocMemory");
  auto launch_node = FindNodeByType(graph.get(), "ExecuteOpLaunch");
  auto free_hold_node = FindNodeByType(graph.get(), "FreeMemoryHoldAddr");

  int64_t remove_edge_alloc = 0;
  ASSERT_TRUE(ge::AttrUtils::GetInt(alloc_node->GetOpDescBarePtr(), kRemoveLaunchFreeEdgeAllocAttr, remove_edge_alloc));
  EXPECT_EQ(remove_edge_alloc, 1);
  EXPECT_EQ(FastNodeTopoChecker(free_hold_node).StrictConnectFrom({{"AllocMemory", 0}}), "success");
  EXPECT_EQ(FastNodeTopoChecker(launch_node).StrictConnectTo(-1, {}), "success");
  EXPECT_EQ(CountControlEdges(alloc_node, free_hold_node), 0U);

  const auto relations = graph->GetExtAttr<FreeLaunchRelations>(kFreeLaunchRelationsAttr);
  ASSERT_NE(relations, nullptr);
  ASSERT_EQ(relations->size(), 1U);
  EXPECT_EQ((*relations)[0], std::make_pair(free_hold_node, launch_node));
}

TEST_F(RemoveLaunchFreeEdgeUT, OneLaunchTwoProducersTwoFreesBuildsCartesianOrderingAndRelations) {
  auto producer0 =
      ValueHolder::CreateSingleDataOutput("AllocMemory", {ValueHolder::CreateFeed(0), ValueHolder::CreateFeed(1)});
  auto producer1 =
      ValueHolder::CreateSingleDataOutput("AllocMemory", {ValueHolder::CreateFeed(2), ValueHolder::CreateFeed(3)});
  auto free0 = ValueHolder::CreateVoidGuarder("FreeMemHbm", producer0, {});
  auto free1 = ValueHolder::CreateVoidGuarder("FreeBatchHbm", producer1, {});
  auto launch = ValueHolder::CreateVoid<ValueHolder>("ExecuteOpLaunch", {ValueHolder::CreateFeed(4)});
  ASSERT_TRUE(ValueHolder::AddDependency(launch, free0).IsSuccess());
  ASSERT_TRUE(ValueHolder::AddDependency(launch, free1).IsSuccess());

  auto frame = ValueHolder::PopGraphFrame();
  ASSERT_NE(frame, nullptr);
  auto graph = frame->GetExecuteGraph();
  ASSERT_NE(graph, nullptr);

  bool changed = false;
  ASSERT_EQ(RemoveLaunchFreeEdge().Run(graph.get(), changed), ge::GRAPH_SUCCESS);
  ASSERT_TRUE(changed);

  auto producer0_node = producer0->GetFastNode();
  auto producer1_node = producer1->GetFastNode();
  auto free0_node = free0->GetFastNode();
  auto free1_node = free1->GetFastNode();
  auto launch_node = launch->GetFastNode();
  ASSERT_NE(producer0_node, nullptr);
  ASSERT_NE(producer1_node, nullptr);
  ASSERT_NE(free0_node, nullptr);
  ASSERT_NE(free1_node, nullptr);
  ASSERT_NE(launch_node, nullptr);

  EXPECT_STREQ(free0_node->GetTypePtr(), "FreeMemHbmHoldAddr");
  EXPECT_STREQ(free1_node->GetTypePtr(), "FreeBatchHbmHoldAddr");
  EXPECT_TRUE(HasDirectDependency(producer0_node, free0_node));
  EXPECT_TRUE(HasDirectDependency(producer0_node, free1_node));
  EXPECT_TRUE(HasDirectDependency(producer1_node, free0_node));
  EXPECT_TRUE(HasDirectDependency(producer1_node, free1_node));
  EXPECT_EQ(CountControlEdges(producer0_node, free0_node), 0U);
  EXPECT_EQ(CountControlEdges(producer0_node, free1_node), 1U);
  EXPECT_EQ(CountControlEdges(producer1_node, free0_node), 1U);
  EXPECT_EQ(CountControlEdges(producer1_node, free1_node), 0U);
  EXPECT_EQ(FastNodeTopoChecker(launch_node).StrictConnectTo(-1, {}), "success");

  const auto relations = graph->GetExtAttr<FreeLaunchRelations>(kFreeLaunchRelationsAttr);
  ASSERT_NE(relations, nullptr);
  ASSERT_EQ(relations->size(), 2U);
  EXPECT_EQ((*relations)[0], std::make_pair(free0_node, launch_node));
  EXPECT_EQ((*relations)[1], std::make_pair(free1_node, launch_node));
}

TEST_F(RemoveLaunchFreeEdgeUT, ExistingCartesianControlEdgeIsNotDuplicated) {
  auto producer0 =
      ValueHolder::CreateSingleDataOutput("AllocMemory", {ValueHolder::CreateFeed(0), ValueHolder::CreateFeed(1)});
  auto producer1 =
      ValueHolder::CreateSingleDataOutput("AllocMemory", {ValueHolder::CreateFeed(2), ValueHolder::CreateFeed(3)});
  auto free0 = ValueHolder::CreateVoidGuarder("FreeMemory", producer0, {});
  auto free1 = ValueHolder::CreateVoidGuarder("FreeMemory", producer1, {});
  auto launch = ValueHolder::CreateVoid<ValueHolder>("ExecuteOpLaunch", {ValueHolder::CreateFeed(4)});
  ASSERT_TRUE(ValueHolder::AddDependency(producer0, free1).IsSuccess());
  ASSERT_TRUE(ValueHolder::AddDependency(launch, free0).IsSuccess());
  ASSERT_TRUE(ValueHolder::AddDependency(launch, free1).IsSuccess());

  auto frame = ValueHolder::PopGraphFrame();
  ASSERT_NE(frame, nullptr);
  auto graph = frame->GetExecuteGraph();
  ASSERT_NE(graph, nullptr);

  bool changed = false;
  ASSERT_EQ(RemoveLaunchFreeEdge().Run(graph.get(), changed), ge::GRAPH_SUCCESS);
  ASSERT_TRUE(changed);
  EXPECT_EQ(CountControlEdges(producer0->GetFastNode(), free1->GetFastNode()), 1U);
}

TEST_F(RemoveLaunchFreeEdgeUT, UnregisteredProducerDoesNotQualifyFreeEdge) {
  auto producer = ValueHolder::CreateSingleDataOutput("NormalResourceProducer", {ValueHolder::CreateFeed(0)});
  ValueHolder::CreateVoidGuarder("FreeMemory", producer, {});
  auto stream = ValueHolder::CreateFeed(1);
  auto launch = ValueHolder::CreateVoid<ValueHolder>("ExecuteOpLaunch", {stream});
  auto free = producer->GetGuarder();
  ASSERT_NE(free, nullptr);
  ASSERT_TRUE(ValueHolder::AddDependency(launch, free).IsSuccess());

  auto frame = ValueHolder::PopGraphFrame();
  ASSERT_NE(frame, nullptr);
  auto graph = frame->GetExecuteGraph();
  ASSERT_NE(graph, nullptr);

  bool changed = false;
  ASSERT_EQ(RemoveLaunchFreeEdge().Run(graph.get(), changed), ge::GRAPH_SUCCESS);
  EXPECT_FALSE(changed);
  EXPECT_NE(FindNodeByType(graph.get(), "FreeMemory"), nullptr);
  EXPECT_EQ(ge::ExecuteGraphUtils::FindFirstNodeMatchType(graph.get(), "FreeMemoryHoldAddr"), nullptr);

  auto launch_node = FindNodeByType(graph.get(), "ExecuteOpLaunch");
  EXPECT_EQ(FastNodeTopoChecker(launch_node).StrictConnectTo(-1, {{"FreeMemory", -1}}), "success");
  EXPECT_FALSE(ge::AttrUtils::HasAttr(producer->GetFastNode()->GetOpDescBarePtr(), kRemoveLaunchFreeEdgeAllocAttr));
}

TEST_F(RemoveLaunchFreeEdgeUT, LaunchProducerDoesNotQualifyFreeEdge) {
  auto producer = ValueHolder::CreateSingleDataOutput("LaunchResourceProducerForRemoveLaunchFreeEdgeUT",
                                                      {ValueHolder::CreateFeed(0)});
  auto free = ValueHolder::CreateVoidGuarder("FreeMemory", producer, {});
  auto launch = ValueHolder::CreateVoid<ValueHolder>("ExecuteOpLaunch", {ValueHolder::CreateFeed(1)});
  ASSERT_TRUE(ValueHolder::AddDependency(launch, free).IsSuccess());

  auto frame = ValueHolder::PopGraphFrame();
  ASSERT_NE(frame, nullptr);
  auto graph = frame->GetExecuteGraph();
  ASSERT_NE(graph, nullptr);

  bool changed = false;
  ASSERT_EQ(RemoveLaunchFreeEdge().Run(graph.get(), changed), ge::GRAPH_SUCCESS);
  EXPECT_FALSE(changed);
  EXPECT_STREQ(free->GetFastNode()->GetTypePtr(), "FreeMemory");
  EXPECT_EQ(FastNodeTopoChecker(launch->GetFastNode()).StrictConnectTo(-1, {{"FreeMemory", -1}}), "success");
  EXPECT_FALSE(ge::AttrUtils::HasAttr(producer->GetFastNode()->GetOpDescBarePtr(), kRemoveLaunchFreeEdgeAllocAttr));
}

TEST_F(RemoveLaunchFreeEdgeUT, HostMemoryProducerDoesNotQualifyFreeEdge) {
  auto allocator = ValueHolder::CreateFeed(0);
  auto size = ValueHolder::CreateFeed(1);
  auto alloc_host = ValueHolder::CreateSingleDataOutput("AllocMemHost", {allocator, size});
  auto free = ValueHolder::CreateVoidGuarder("FreeMemory", alloc_host, {});
  auto stream = ValueHolder::CreateFeed(2);
  auto launch = ValueHolder::CreateVoid<ValueHolder>("ExecuteOpLaunch", {stream});
  ASSERT_TRUE(ValueHolder::AddDependency(launch, free).IsSuccess());

  auto frame = ValueHolder::PopGraphFrame();
  ASSERT_NE(frame, nullptr);
  auto graph = frame->GetExecuteGraph();
  ASSERT_NE(graph, nullptr);

  bool changed = false;
  ASSERT_EQ(RemoveLaunchFreeEdge().Run(graph.get(), changed), ge::GRAPH_SUCCESS);
  EXPECT_FALSE(changed);
  EXPECT_NE(FindNodeByType(graph.get(), "FreeMemory"), nullptr);
  EXPECT_EQ(ge::ExecuteGraphUtils::FindFirstNodeMatchType(graph.get(), "FreeMemoryHoldAddr"), nullptr);

  auto launch_node = FindNodeByType(graph.get(), "ExecuteOpLaunch");
  auto alloc_host_node = FindNodeByType(graph.get(), "AllocMemHost");
  EXPECT_EQ(FastNodeTopoChecker(launch_node).StrictConnectTo(-1, {{"FreeMemory", -1}}), "success");
  EXPECT_FALSE(ge::AttrUtils::HasAttr(alloc_host_node->GetOpDescBarePtr(), kRemoveLaunchFreeEdgeAllocAttr));
}

TEST_F(RemoveLaunchFreeEdgeUT, DynamicPlacementProducerDoesNotQualifyFreeEdge) {
  auto access = ValueHolder::CreateSingleDataOutput("AccessMemCrossStream",
                                                    {ValueHolder::CreateFeed(0), ValueHolder::CreateFeed(1)});
  auto free = ValueHolder::CreateVoidGuarder("FreeMemory", access, {});
  auto launch = ValueHolder::CreateVoid<ValueHolder>("ExecuteOpLaunch", {ValueHolder::CreateFeed(2)});
  ASSERT_TRUE(ValueHolder::AddDependency(launch, free).IsSuccess());

  auto frame = ValueHolder::PopGraphFrame();
  ASSERT_NE(frame, nullptr);
  auto graph = frame->GetExecuteGraph();
  ASSERT_NE(graph, nullptr);

  bool changed = false;
  ASSERT_EQ(RemoveLaunchFreeEdge().Run(graph.get(), changed), ge::GRAPH_SUCCESS);
  EXPECT_FALSE(changed);
  EXPECT_NE(FindNodeByType(graph.get(), "FreeMemory"), nullptr);
  EXPECT_EQ(ge::ExecuteGraphUtils::FindFirstNodeMatchType(graph.get(), "FreeMemoryHoldAddr"), nullptr);
  EXPECT_EQ(FastNodeTopoChecker(launch->GetFastNode()).StrictConnectTo(-1, {{"FreeMemory", -1}}), "success");
}

TEST_F(RemoveLaunchFreeEdgeUT, ZeroDataInputFreeDoesNotQualify) {
  auto stream = ValueHolder::CreateFeed(0);
  auto launch = ValueHolder::CreateVoid<ValueHolder>("ExecuteOpLaunch", {stream});
  auto free = ValueHolder::CreateVoid<ValueHolder>("FreeMemory", {});
  ASSERT_TRUE(ValueHolder::AddDependency(launch, free).IsSuccess());

  auto frame = ValueHolder::PopGraphFrame();
  ASSERT_NE(frame, nullptr);
  auto graph = frame->GetExecuteGraph();
  ASSERT_NE(graph, nullptr);
  ASSERT_TRUE(free->GetFastNode()->GetInDataNodes().empty());

  bool changed = false;
  ASSERT_EQ(RemoveLaunchFreeEdge().Run(graph.get(), changed), ge::GRAPH_SUCCESS);
  EXPECT_FALSE(changed);
  EXPECT_NE(FindNodeByType(graph.get(), "FreeMemory"), nullptr);
  EXPECT_EQ(ge::ExecuteGraphUtils::FindFirstNodeMatchType(graph.get(), "FreeMemoryHoldAddr"), nullptr);

  auto launch_node = FindNodeByType(graph.get(), "ExecuteOpLaunch");
  EXPECT_EQ(FastNodeTopoChecker(launch_node).StrictConnectTo(-1, {{"FreeMemory", -1}}), "success");
}

TEST_F(RemoveLaunchFreeEdgeUT, MultiInputFreeOnlyChecksReleasedResourceProducer) {
  auto size = ValueHolder::CreateFeed(0);
  auto allocator = ValueHolder::CreateFeed(1);
  auto alloc = ValueHolder::CreateSingleDataOutput("AllocMemory", {size, allocator});
  auto aux = ValueHolder::CreateSingleDataOutput("NormalResourceProducer", {ValueHolder::CreateFeed(2)});
  auto free = ValueHolder::CreateVoidGuarder("FreeMemory", alloc, {aux});
  auto stream = ValueHolder::CreateFeed(3);
  auto launch = ValueHolder::CreateVoid<ValueHolder>("ExecuteOpLaunch", {stream});
  ASSERT_TRUE(ValueHolder::AddDependency(launch, free).IsSuccess());

  auto frame = ValueHolder::PopGraphFrame();
  ASSERT_NE(frame, nullptr);
  auto graph = frame->GetExecuteGraph();
  ASSERT_NE(graph, nullptr);

  bool changed = false;
  ASSERT_EQ(RemoveLaunchFreeEdge().Run(graph.get(), changed), ge::GRAPH_SUCCESS);
  ASSERT_TRUE(changed);

  auto alloc_node = FindNodeByType(graph.get(), "AllocMemory");
  auto aux_node = FindNodeByType(graph.get(), "NormalResourceProducer");
  auto launch_node = FindNodeByType(graph.get(), "ExecuteOpLaunch");
  auto free_hold_node = FindNodeByType(graph.get(), "FreeMemoryHoldAddr");

  int64_t remove_edge_alloc = 0;
  ASSERT_TRUE(ge::AttrUtils::GetInt(alloc_node->GetOpDescBarePtr(), kRemoveLaunchFreeEdgeAllocAttr, remove_edge_alloc));
  EXPECT_EQ(remove_edge_alloc, 1);
  EXPECT_FALSE(ge::AttrUtils::HasAttr(aux_node->GetOpDescBarePtr(), kRemoveLaunchFreeEdgeAllocAttr));
  EXPECT_EQ(FastNodeTopoChecker(free_hold_node).StrictConnectFrom({{"AllocMemory", 0}, {"NormalResourceProducer", 0}}),
            "success");
  EXPECT_EQ(FastNodeTopoChecker(launch_node).StrictConnectTo(-1, {}), "success");
}

TEST_F(RemoveLaunchFreeEdgeUT, MultipleLaunchesToSameEligibleFreeAreAllRemoved) {
  auto size = ValueHolder::CreateFeed(0);
  auto allocator = ValueHolder::CreateFeed(1);
  auto alloc = ValueHolder::CreateSingleDataOutput("AllocMemory", {size, allocator});
  auto free = ValueHolder::CreateVoidGuarder("FreeMemory", alloc, {});
  auto stream0 = ValueHolder::CreateFeed(2);
  auto stream1 = ValueHolder::CreateFeed(3);
  auto launch0 = ValueHolder::CreateVoid<ValueHolder>("ExecuteOpLaunch", {stream0});
  auto launch1 = ValueHolder::CreateVoid<ValueHolder>("ExecuteOpLaunch", {stream1});
  ASSERT_TRUE(ValueHolder::AddDependency(launch0, free).IsSuccess());
  ASSERT_TRUE(ValueHolder::AddDependency(launch1, free).IsSuccess());

  auto frame = ValueHolder::PopGraphFrame();
  ASSERT_NE(frame, nullptr);
  auto graph = frame->GetExecuteGraph();
  ASSERT_NE(graph, nullptr);

  bool changed = false;
  ASSERT_EQ(RemoveLaunchFreeEdge().Run(graph.get(), changed), ge::GRAPH_SUCCESS);
  EXPECT_TRUE(changed);

  auto alloc_node = FindNodeByType(graph.get(), "AllocMemory");
  auto free_hold_node = FindNodeByType(graph.get(), "FreeMemoryHoldAddr");
  auto launch_nodes = ge::ExecuteGraphUtils::FindNodesByTypeFromAllNodes(graph.get(), "ExecuteOpLaunch");
  ASSERT_EQ(launch_nodes.size(), 2UL);

  int64_t remove_edge_alloc = 0;
  ASSERT_TRUE(ge::AttrUtils::GetInt(alloc_node->GetOpDescBarePtr(), kRemoveLaunchFreeEdgeAllocAttr, remove_edge_alloc));
  EXPECT_EQ(remove_edge_alloc, 1);
  EXPECT_EQ(FastNodeTopoChecker(free_hold_node).StrictConnectFrom({{"AllocMemory", 0}}), "success");
  EXPECT_EQ(FastNodeTopoChecker(launch_nodes[0]).StrictConnectTo(-1, {}), "success");
  EXPECT_EQ(FastNodeTopoChecker(launch_nodes[1]).StrictConnectTo(-1, {}), "success");

  const auto relations = graph->GetExtAttr<FreeLaunchRelations>(kFreeLaunchRelationsAttr);
  ASSERT_NE(relations, nullptr);
  ASSERT_EQ(relations->size(), 2U);
  EXPECT_EQ((*relations)[0], std::make_pair(free_hold_node, launch0->GetFastNode()));
  EXPECT_EQ((*relations)[1], std::make_pair(free_hold_node, launch1->GetFastNode()));
}

TEST_F(RemoveLaunchFreeEdgeUT, ExistingExactRelationIsNotDuplicated) {
  auto producer =
      ValueHolder::CreateSingleDataOutput("AllocMemory", {ValueHolder::CreateFeed(0), ValueHolder::CreateFeed(1)});
  auto free = ValueHolder::CreateVoidGuarder("FreeMemory", producer, {});
  auto launch = ValueHolder::CreateVoid<ValueHolder>("ExecuteOpLaunch", {ValueHolder::CreateFeed(2)});
  ASSERT_TRUE(ValueHolder::AddDependency(launch, free).IsSuccess());

  auto frame = ValueHolder::PopGraphFrame();
  ASSERT_NE(frame, nullptr);
  auto graph = frame->GetExecuteGraph();
  ASSERT_NE(graph, nullptr);
  ASSERT_TRUE(
      graph->SetExtAttr(kFreeLaunchRelationsAttr, FreeLaunchRelations{{free->GetFastNode(), launch->GetFastNode()}}));

  bool changed = false;
  ASSERT_EQ(RemoveLaunchFreeEdge().Run(graph.get(), changed), ge::GRAPH_SUCCESS);
  ASSERT_TRUE(changed);

  const auto relations = graph->GetExtAttr<FreeLaunchRelations>(kFreeLaunchRelationsAttr);
  ASSERT_NE(relations, nullptr);
  ASSERT_EQ(relations->size(), 1U);
  EXPECT_EQ((*relations)[0], std::make_pair(free->GetFastNode(), launch->GetFastNode()));
}

TEST_F(RemoveLaunchFreeEdgeUT, RelationStorageFailurePreservesLaunchFreeEdgeAndFreeType) {
  auto producer =
      ValueHolder::CreateSingleDataOutput("AllocMemory", {ValueHolder::CreateFeed(0), ValueHolder::CreateFeed(1)});
  auto free = ValueHolder::CreateVoidGuarder("FreeMemory", producer, {});
  auto launch = ValueHolder::CreateVoid<ValueHolder>("ExecuteOpLaunch", {ValueHolder::CreateFeed(2)});
  ASSERT_TRUE(ValueHolder::AddDependency(launch, free).IsSuccess());

  auto frame = ValueHolder::PopGraphFrame();
  ASSERT_NE(frame, nullptr);
  auto graph = frame->GetExecuteGraph();
  ASSERT_NE(graph, nullptr);
  ASSERT_TRUE(graph->SetExtAttr(kFreeLaunchRelationsAttr, int32_t{1}));

  bool changed = false;
  EXPECT_NE(RemoveLaunchFreeEdge().Run(graph.get(), changed), ge::GRAPH_SUCCESS);
  EXPECT_FALSE(changed);
  EXPECT_STREQ(free->GetFastNode()->GetTypePtr(), "FreeMemory");
  EXPECT_EQ(FastNodeTopoChecker(launch->GetFastNode()).StrictConnectTo(-1, {{"FreeMemory", -1}}), "success");
  EXPECT_FALSE(ge::AttrUtils::HasAttr(producer->GetFastNode()->GetOpDescBarePtr(), kRemoveLaunchFreeEdgeAllocAttr));
}

TEST_F(RemoveLaunchFreeEdgeUT, SubgraphRelationIsStoredAndMutatedOnOwningGraph) {
  auto subgraph_call = ValueHolder::CreateVoid<ValueHolder>("SubgraphCall", {});
  ASSERT_NE(ValueHolder::PushGraphFrame(subgraph_call, "Subgraph"), nullptr);
  auto producer =
      ValueHolder::CreateSingleDataOutput("AllocMemory", {ValueHolder::CreateFeed(0), ValueHolder::CreateFeed(1)});
  auto free = ValueHolder::CreateVoidGuarder("FreeMemory", producer, {});
  auto launch = ValueHolder::CreateVoid<ValueHolder>("ExecuteOpLaunch", {ValueHolder::CreateFeed(2)});
  ASSERT_TRUE(ValueHolder::AddDependency(launch, free).IsSuccess());
  auto subgraph_frame = ValueHolder::PopGraphFrame();
  ASSERT_NE(subgraph_frame, nullptr);
  auto root_frame = ValueHolder::PopGraphFrame();
  ASSERT_NE(root_frame, nullptr);
  auto subgraph = subgraph_frame->GetExecuteGraph();
  auto root_graph = root_frame->GetExecuteGraph();
  ASSERT_NE(subgraph, nullptr);
  ASSERT_NE(root_graph, nullptr);

  bool changed = false;
  ASSERT_EQ(RemoveLaunchFreeEdge().Run(root_graph.get(), changed), ge::GRAPH_SUCCESS);
  ASSERT_TRUE(changed);
  EXPECT_EQ(root_graph->GetExtAttr<FreeLaunchRelations>(kFreeLaunchRelationsAttr), nullptr);
  const auto relations = subgraph->GetExtAttr<FreeLaunchRelations>(kFreeLaunchRelationsAttr);
  ASSERT_NE(relations, nullptr);
  ASSERT_EQ(relations->size(), 1U);
  EXPECT_EQ((*relations)[0], std::make_pair(free->GetFastNode(), launch->GetFastNode()));
  EXPECT_STREQ(free->GetFastNode()->GetTypePtr(), "FreeMemoryHoldAddr");
  EXPECT_EQ(FastNodeTopoChecker(launch->GetFastNode()).StrictConnectTo(-1, {}), "success");
}

TEST_F(RemoveLaunchFreeEdgeUT, FreeTensorMemoryDoesNotQualifyWithoutHoldAddrVariant) {
  auto tensor = ValueHolder::CreateSingleDataOutput("BuildTensor", {ValueHolder::CreateFeed(0)});
  auto free = ValueHolder::CreateVoidGuarder("FreeTensorMemory", tensor, {});
  auto stream = ValueHolder::CreateFeed(1);
  auto launch = ValueHolder::CreateVoid<ValueHolder>("ExecuteOpLaunch", {stream});
  ASSERT_TRUE(ValueHolder::AddDependency(launch, free).IsSuccess());

  auto frame = ValueHolder::PopGraphFrame();
  ASSERT_NE(frame, nullptr);
  auto graph = frame->GetExecuteGraph();
  ASSERT_NE(graph, nullptr);

  bool changed = false;
  ASSERT_EQ(RemoveLaunchFreeEdge().Run(graph.get(), changed), ge::GRAPH_SUCCESS);
  EXPECT_FALSE(changed);

  auto tensor_node = FindNodeByType(graph.get(), "BuildTensor");
  auto launch_node = FindNodeByType(graph.get(), "ExecuteOpLaunch");
  EXPECT_NE(FindNodeByType(graph.get(), "FreeTensorMemory"), nullptr);
  EXPECT_EQ(FastNodeTopoChecker(launch_node).StrictConnectTo(-1, {{"FreeTensorMemory", -1}}), "success");
  EXPECT_FALSE(ge::AttrUtils::HasAttr(tensor_node->GetOpDescBarePtr(), kRemoveLaunchFreeEdgeAllocAttr));
}
}  // namespace bg
}  // namespace gert
