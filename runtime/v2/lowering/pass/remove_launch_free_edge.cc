/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "remove_launch_free_edge.h"
#include "common/compile_profiling/ge_call_wrapper.h"
#include "core/builder/node_types.h"
#include "core/executor/multi_thread_topological/execution_data/free_launch_relation.h"
#include "exe_graph/lowering/exe_graph_attrs.h"
#include "graph/utils/graph_dump_utils.h"
#include "graph/utils/fast_node_utils.h"
#include "kernel/memory/memory_kernel.h"
#include "graph/utils/op_desc_utils_ex.h"
#include "register/kernel_registry.h"
#include "core/executor/multi_thread_topological/executor/schedule/producer/producers/kernel_tags/critical_section_config.h"
namespace gert {
namespace bg {
namespace {
bool IsLaunchTargetNode(const ge::FastNode *const node) {
  const auto &node_type = node->GetTypePtr();
  const auto kernel_info = KernelRegistry::GetInstance().FindKernelInfo(node_type);
  return (kernel_info != nullptr) && (kernel_info->critical_section == kKernelLaunch);
}
bool IsMemoryTargetNode(const ge::FastNode *const node) {
  const auto kernel_info = KernelRegistry::GetInstance().FindKernelInfo(node->GetTypePtr());
  return (kernel_info != nullptr) && (kernel_info->critical_section == kKernelUseMemory);
}
bool IsDynamicPlacementMemoryNode(const ge::FastNode *const node) {
  return strcmp(node->GetTypePtr(), "AccessMemCrossStream") == 0;
}
bool IsFreeTargetNode(const ge::FastNode *const node) {
  const auto &node_type = node->GetTypePtr();
  static std::vector<const char *> kFreeKernels = {"FreeMemory", "FreeMemHbm", "FreeBatchHbm"};
  auto func = [&node_type](const char *const type) { return (strcmp(node_type, type) == 0); };
  return std::any_of(kFreeKernels.begin(), kFreeKernels.end(), func);
}

template <typename T>
void AppendUnique(std::vector<T> &values, const T &value) {
  if (std::find(values.begin(), values.end(), value) == values.end()) {
    values.emplace_back(value);
  }
}

bool HasDirectDependency(const ge::FastNode *const src, const ge::FastNode *const dst) {
  const auto &in_nodes = dst->GetAllInNodes();
  return std::find(in_nodes.begin(), in_nodes.end(), src) != in_nodes.end();
}

ge::FastNode *GetReleasedResourceProducer(const ge::FastNode *const free_node) {
  int32_t release_index = 0;
  if (!ge::AttrUtils::GetInt(free_node->GetOpDescBarePtr(), kReleaseResourceIndex, release_index)) {
    return nullptr;
  }
  return ge::FastNodeUtils::GetInDataNodeByIndex(free_node, release_index);
}

ge::graphStatus ReplaceFreeNode(const ge::FastNode *free_node) {
  static std::map<std::string, std::string> origin_free_to_new_free_types = {
      {kernel::kFreeMemory, kernel::kFreeMemoryHoldAddr},
      {kernel::kFreeMemHbm, kernel::kFreeMemHbmHoldAddr},
      {kernel::kFreeBatchHbm, kernel::kFreeBatchHbmHoldAddr}};
  auto op_desc = free_node->GetOpDescPtr();
  GE_ASSERT_NOTNULL(op_desc);
  const auto iter = origin_free_to_new_free_types.find(free_node->GetType());
  GE_ASSERT_TRUE(iter != origin_free_to_new_free_types.end(), "free node %s type %s is invalid",
                 free_node->GetNamePtr(), free_node->GetTypePtr());
  ge::OpDescUtilsEx::SetType(op_desc, iter->second);
  op_desc->SetName(iter->second + "_" + free_node->GetName());
  return ge::GRAPH_SUCCESS;
}

struct LaunchFreePlan {
  ge::ExecuteGraph *owner_graph;
  ge::FastNode *launch_node;
  std::vector<ge::FastNode *> alloc_nodes;
  std::vector<ge::FastNode *> free_nodes;
  std::vector<ge::FastEdge *> launch_free_edges;
};
}  // namespace
ge::graphStatus RemoveLaunchFreeEdge::Run(ge::ExecuteGraph *const graph, bool &changed) {
  GE_TIMESTAMP_START(RemoveLaunchFreeEdge);
  const auto launch_nodes = graph->GetAllNodes(IsLaunchTargetNode);
  std::vector<LaunchFreePlan> plans;
  std::vector<ge::FastNode *> target_alloc_nodes;
  std::vector<ge::FastNode *> target_free_nodes;
  for (const auto launch_node : launch_nodes) {
    GE_ASSERT_NOTNULL(launch_node->GetExtendInfo());
    auto owner_graph = launch_node->GetExtendInfo()->GetOwnerGraphBarePtr();
    GE_ASSERT_NOTNULL(owner_graph);
    LaunchFreePlan plan{owner_graph, launch_node, {}, {}, {}};
    for (const auto out_ctrl_edge : launch_node->GetAllOutControlEdges()) {
      const auto node = out_ctrl_edge->dst;
      if (IsFreeTargetNode(node)) {
        const auto alloc_node = GetReleasedResourceProducer(node);
        if ((alloc_node != nullptr) && !IsAllocHostNode(alloc_node->GetTypePtr()) &&
            !IsDynamicPlacementMemoryNode(alloc_node) && IsMemoryTargetNode(alloc_node)) {
          AppendUnique(plan.alloc_nodes, alloc_node);
          AppendUnique(plan.free_nodes, node);
          plan.launch_free_edges.emplace_back(out_ctrl_edge);
          AppendUnique(target_alloc_nodes, alloc_node);
          AppendUnique(target_free_nodes, node);
        }
      }
    }
    if (!plan.launch_free_edges.empty()) {
      plans.emplace_back(std::move(plan));
    }
  }

  for (const auto &plan : plans) {
    for (const auto alloc_node : plan.alloc_nodes) {
      for (const auto free_node : plan.free_nodes) {
        if (HasDirectDependency(alloc_node, free_node)) {
          continue;
        }
        GE_ASSERT_NOTNULL(
            plan.owner_graph->AddEdge(alloc_node, ge::kControlEdgeIndex, free_node, ge::kControlEdgeIndex));
        GELOGD("add control edge from %s to %s", alloc_node->GetNamePtr(), free_node->GetNamePtr());
      }
    }
  }

  std::vector<ge::ExecuteGraph *> relation_owner_graphs;
  for (const auto &plan : plans) {
    AppendUnique(relation_owner_graphs, plan.owner_graph);
  }
  for (const auto owner_graph : relation_owner_graphs) {
    FreeLaunchRelations relations;
    const auto stored_relations = owner_graph->GetExtAttr<FreeLaunchRelations>(kFreeLaunchRelationsAttr);
    if (stored_relations != nullptr) {
      relations = *stored_relations;
    }
    for (const auto &plan : plans) {
      if (plan.owner_graph != owner_graph) {
        continue;
      }
      for (const auto free_node : plan.free_nodes) {
        AppendUnique(relations, FreeLaunchRelation{free_node, plan.launch_node});
      }
    }
    GE_ASSERT_TRUE(owner_graph->SetExtAttr(kFreeLaunchRelationsAttr, relations),
                   "Failed to store free-launch relations for graph %s", owner_graph->GetName().c_str());
  }

  for (const auto alloc_node : target_alloc_nodes) {
    auto alloc_op_desc = alloc_node->GetOpDescBarePtr();
    GE_ASSERT_NOTNULL(alloc_op_desc);
    GE_ASSERT_TRUE(ge::AttrUtils::SetInt(alloc_op_desc, "remove_launch_free_edge_alloc", 1));
  }
  for (const auto free_node : target_free_nodes) {
    GE_ASSERT_SUCCESS(ReplaceFreeNode(free_node));
  }

  bool changed_in_pass = false;
  for (const auto &plan : plans) {
    for (const auto edge : plan.launch_free_edges) {
      GELOGD("remove ctrl edge from %s to %s", edge->src->GetNamePtr(), edge->dst->GetNamePtr());
      GE_ASSERT_GRAPH_SUCCESS(plan.owner_graph->RemoveEdge(edge));
      changed_in_pass = true;
    }
  }
  if (changed_in_pass) {
    changed = true;
    ge::DumpGraph(graph, "RemoveLaunchFreeEdgeAfter");
  }
  GE_TIMESTAMP_EVENT_END(RemoveLaunchFreeEdge, "Pass::RemoveLaunchFreeEdge");
  return ge::GRAPH_SUCCESS;
}
}  // namespace bg
}  // namespace gert
