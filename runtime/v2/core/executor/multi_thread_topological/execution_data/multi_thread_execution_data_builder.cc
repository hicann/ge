/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "multi_thread_execution_data_builder.h"
#include "multi_thread_execution_data.h"
#include "core/executor/multi_thread_topological/executor/schedule/scheduler/task_scheduler_factory.h"
#include "core/executor/multi_thread_topological/executor/schedule/producer/task_producer_factory.h"
#include "core/executor/multi_thread_topological/executor/schedule/producer/producers/kernel_tags/critical_section_config.h"
#include "core/utils/rt2_executor_utils.h"
#include "multi_thread_exe_graph_resource_guard.h"
#include "framework/runtime/executor_option/multi_thread_executor_option.h"
#include "register/kernel_registry.h"

namespace gert {
namespace {
bool IsHoldAddressFree(const ge::FastNode *const node) {
  static const std::vector<const char *> kHoldAddressFreeTypes = {"FreeMemoryHoldAddr", "FreeMemHbmHoldAddr",
                                                                  "FreeBatchHbmHoldAddr"};
  return std::any_of(kHoldAddressFreeTypes.begin(), kHoldAddressFreeTypes.end(),
                     [node](const char *const type) { return strcmp(node->GetTypePtr(), type) == 0; });
}

bool IsLaunchKernel(const ge::FastNode *const node) {
  const auto kernel_info = KernelRegistry::GetInstance().FindKernelInfo(node->GetTypePtr());
  return (kernel_info != nullptr) && (kernel_info->critical_section == kKernelLaunch);
}

ge::graphStatus GetExecutionNodeId(const ge::FastNode *const graph_node,
                                   const std::vector<std::pair<ge::FastNode *, Node *>> &graph_to_exe_nodes,
                                   NodeIdentity &node_id) {
  const auto iter = std::find_if(
      graph_to_exe_nodes.begin(), graph_to_exe_nodes.end(),
      [graph_node](const std::pair<ge::FastNode *, Node *> &mapping) { return mapping.first == graph_node; });
  GE_ASSERT_TRUE(iter != graph_to_exe_nodes.end(), "Relation node %s is not mapped to an execution node",
                 graph_node->GetNamePtr());
  GE_ASSERT_NOTNULL(iter->second);
  node_id = iter->second->node_id;
  GE_ASSERT_TRUE(node_id < graph_to_exe_nodes.size(), "Execution node id %zu for relation node %s is out of range %zu",
                 node_id, graph_node->GetNamePtr(), graph_to_exe_nodes.size());
  return ge::GRAPH_SUCCESS;
}
}  // namespace

MultiThreadExecutionDataBuilder::MultiThreadExecutionDataBuilder(GraphExecutorBuilder &executor_builder)
    : ExecutionDataBuilder(executor_builder), base_ed_builder_(executor_builder) {}

ge::graphStatus MultiThreadExecutionDataBuilder::BuildFreeLaunchRelationCsr(
    const std::vector<std::pair<ge::FastNode *, Node *>> &graph_to_exe_nodes,
    MultiThreadResourceGuard &resource_guard) {
  const auto graph = GetExecutorBuilder().GetExeGraph();
  GE_ASSERT_NOTNULL(graph);
  std::vector<std::pair<NodeIdentity, NodeIdentity>> relation_ids;
  std::vector<ge::ExecuteGraph *> relation_graphs;
  for (const auto &mapping : graph_to_exe_nodes) {
    GE_ASSERT_NOTNULL(mapping.first);
    GE_ASSERT_NOTNULL(mapping.first->GetExtendInfo());
    const auto owner_graph = mapping.first->GetExtendInfo()->GetOwnerGraphBarePtr();
    GE_ASSERT_NOTNULL(owner_graph);
    if (std::find(relation_graphs.begin(), relation_graphs.end(), owner_graph) == relation_graphs.end()) {
      relation_graphs.emplace_back(owner_graph);
    }
  }
  for (const auto relation_graph : relation_graphs) {
    const auto relations = relation_graph->GetExtAttr<FreeLaunchRelations>(kFreeLaunchRelationsAttr);
    if (relations == nullptr) {
      continue;
    }
    for (const auto &relation : *relations) {
      const auto free_node = relation.first;
      const auto launch_node = relation.second;
      GE_ASSERT_NOTNULL(free_node);
      GE_ASSERT_NOTNULL(launch_node);
      GE_ASSERT_NOTNULL(free_node->GetExtendInfo());
      GE_ASSERT_NOTNULL(launch_node->GetExtendInfo());
      GE_ASSERT_TRUE(free_node->GetExtendInfo()->GetOwnerGraphBarePtr() == relation_graph,
                     "Free relation node %s is not owned by graph %s", free_node->GetNamePtr(),
                     relation_graph->GetName().c_str());
      GE_ASSERT_TRUE(launch_node->GetExtendInfo()->GetOwnerGraphBarePtr() == relation_graph,
                     "Launch relation node %s is not owned by graph %s", launch_node->GetNamePtr(),
                     relation_graph->GetName().c_str());
      GE_ASSERT_TRUE(IsHoldAddressFree(free_node), "Relation source %s type %s is not a hold-address Free",
                     free_node->GetNamePtr(), free_node->GetTypePtr());
      GE_ASSERT_TRUE(IsLaunchKernel(launch_node), "Relation target %s type %s is not a launch kernel",
                     launch_node->GetNamePtr(), launch_node->GetTypePtr());

      NodeIdentity free_id = 0U;
      NodeIdentity launch_id = 0U;
      GE_ASSERT_GRAPH_SUCCESS(GetExecutionNodeId(free_node, graph_to_exe_nodes, free_id));
      GE_ASSERT_GRAPH_SUCCESS(GetExecutionNodeId(launch_node, graph_to_exe_nodes, launch_id));
      relation_ids.emplace_back(free_id, launch_id);
    }
  }

  std::sort(relation_ids.begin(), relation_ids.end());
  relation_ids.erase(std::unique(relation_ids.begin(), relation_ids.end()), relation_ids.end());
  std::vector<NodeIdentity> offsets(graph_to_exe_nodes.size() + 1U, 0U);
  std::vector<NodeIdentity> launch_ids;
  launch_ids.reserve(relation_ids.size());
  for (const auto &relation : relation_ids) {
    ++offsets[relation.first + 1U];
    launch_ids.emplace_back(relation.second);
  }
  for (size_t i = 1U; i < offsets.size(); ++i) {
    offsets[i] += offsets[i - 1U];
  }

  auto offsets_guarder = CreateCArray(offsets);
  GE_ASSERT_NOTNULL(offsets_guarder);
  auto launch_ids_guarder = CreateCArray(launch_ids);
  if (!launch_ids.empty()) {
    GE_ASSERT_NOTNULL(launch_ids_guarder);
  }
  (void)resource_guard.ResetFreeLaunchRelationCsr(std::move(offsets_guarder), std::move(launch_ids_guarder),
                                                  graph_to_exe_nodes.size(), launch_ids.size());
  return ge::GRAPH_SUCCESS;
}

ResourceGuardPtr MultiThreadExecutionDataBuilder::Build() {
  GraphNode graph_nodes;
  return Build(graph_nodes);
}

ResourceGuardPtr MultiThreadExecutionDataBuilder::Build(GraphNode &graph_nodes) {
  auto resource_guard = ge::MakeUnique<MultiThreadResourceGuard>();
  auto execution_data_holder = ge::MakeUnique<uint8_t[]>(sizeof(MultiThreadExecutionData));
  GE_ASSERT_NOTNULL(execution_data_holder);
  auto execution_data = reinterpret_cast<MultiThreadExecutionData *>(execution_data_holder.get());
  base_ed_builder_.ReOrderByPriority(false).Build(&(execution_data->topo_ed.base_ed), resource_guard.get(),
                                                  graph_nodes);

  GE_ASSERT_SUCCESS(graph_nodes.EnsureNodeExeInOrder(GetExecutorBuilder().GetExeGraph().get()));

  auto &graph_to_exe_nodes = base_ed_builder_.GetOrderedGraphToExeNodes();
  GE_ASSERT_SUCCESS(BuildFreeLaunchRelationCsr(graph_to_exe_nodes, *resource_guard));
  auto exe_nodes_size = graph_to_exe_nodes.size();
  graph_nodes.node_indegrees.resize(exe_nodes_size);
  graph_nodes.node_watchers.resize(exe_nodes_size);
  for (auto &graph_node_to_exe_node : graph_to_exe_nodes) {
    Watcher *watcher = nullptr;
    GE_ASSERT_SUCCESS(graph_nodes.ReadInTopoInfo(graph_node_to_exe_node, watcher));
    resource_guard->PushWatcher(watcher);
    SetPriorityForNode(graph_node_to_exe_node);
  }
  GE_ASSERT_SUCCESS(CreateExecutionData(graph_nodes, &(execution_data->topo_ed), resource_guard.get()));

  GE_ASSERT_SUCCESS(CreateKernelOutputs(base_ed_builder_.GetOrderedGraphToExeNodes()));
  auto multi_thread_executor_option =
      reinterpret_cast<MultiThreadExecutorOption *>(GetExecutorBuilder().GetExecutorOption());
  auto thread_num = multi_thread_executor_option->GetThreadNum();
  GE_ASSERT_TRUE(thread_num >= kLeastThreadNumber, "new thread num is less than least %u", kLeastThreadNumber);
  GELOGD("multi thread executor will create %zu threads", thread_num);

  TaskSchedulerConfig cfg;
  cfg.producer_cfg.type = TaskProducerFactory::GetInstance().GetProducerType();
  cfg.AddWorkers(1, ExecTaskType::MEMORY, TaskThreadMode::URGENT, 1);
  if (thread_num > kLeastThreadNumber) {
    cfg.AddWorkers(1, ExecTaskType::LAUNCH, TaskThreadMode::URGENT, 1);
    thread_num--;
  }
  cfg.AddWorkers(1, ExecTaskType::NORMAL, TaskThreadMode::URGENT, thread_num - 1);  // one for memory worker

  execution_data->scheduler = resource_guard->ResetTaskScheduler(
      std::unique_ptr<TaskScheduler>(TaskSchedulerFactory::GetInstance().Create(cfg)));
  GE_ASSERT_SUCCESS(execution_data->scheduler->Prepare(
      TaskScheduler::ScheduleData(&(execution_data->topo_ed), resource_guard->GetFreeLaunchRelationCsr())));

  resource_guard->ResetExecutionData(std::move(execution_data_holder));
  return resource_guard;
}

ge::graphStatus MultiThreadExecutionDataBuilder::CreateExecutionData(GraphNode &graph_node,
                                                                     TopologicalExecutionData *topo_execution_data,
                                                                     ResourceGuard *resource_guard) const {
  const auto topo_resource_guard = reinterpret_cast<TopologicalResourceGuard *>(resource_guard);
  topo_execution_data->start_num = graph_node.start_nodes.size();
  topo_execution_data->start_nodes =
      reinterpret_cast<Node **>(topo_resource_guard->ResetStartNodesArray(CreateCArray(graph_node.start_nodes)));

  topo_execution_data->node_indegrees =
      reinterpret_cast<int64_t *>(topo_resource_guard->ResetNodesIndgreeArray(CreateCArray(graph_node.node_indegrees)));
  topo_execution_data->node_indegrees_backup = reinterpret_cast<int64_t *>(
      topo_resource_guard->ResetNodesWaitIndgreeArray(CreateCArray(graph_node.node_indegrees)));
  topo_execution_data->node_watchers =
      reinterpret_cast<Watcher **>(topo_resource_guard->ResetWatchersArray(CreateCArray(graph_node.node_watchers)));

  topo_execution_data->ready_queue =
      topo_resource_guard->ResetReadyQueue(CreatePriorityQueue(graph_node.nodes.size() + 1U));
  return ge::GRAPH_SUCCESS;
}
}  // namespace gert
