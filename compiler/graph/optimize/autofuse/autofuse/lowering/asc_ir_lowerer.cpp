/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "asc_ir_lowerer.h"
#include "lowerings.h"
#include "backend/backend_spec.h"
#include "utils/auto_fuse_config.h"
#include "graph/utils/op_type_utils.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/debug/ge_op_types.h"
#include "graph/ascendc_ir/utils/asc_graph_utils.h"
#include "utils/autofuse_attrs.h"
#include "op_helper/lower_concat_helper.h"
#include "op_helper/lower_split_helper.h"
#include "common/scope_tracing_recorder.h"

namespace ge {
using namespace autofuse;
namespace {
bool IsNodeShouldPrune(const NodePtr &node) {
  if (!node->GetOutNodes().empty()) {
    return false;
  }
  if (OpTypeUtils::IsDataNode(node->GetType())) {
    GELOGI("Skip prune unused data %s", node->GetName().c_str());
    return false;
  }
  return true;
}

bool IsAscBackendOpNode(const NodePtr &node) {
  return (node->GetType() == kAscBackend || node->GetType() == kAscBackendNoKernelOp);
}

void GetOriginNamesAndTypes(OpDescPtr &op_desc, std::vector<std::string> &original_names,
                            std::vector<std::string> &original_types) {
  std::vector<std::string> origin_op_names;
  std::vector<std::string> origin_op_types;
  bool is_has_attr = ge::AttrUtils::GetListStr(op_desc, ge::ATTR_NAME_DATA_DUMP_ORIGIN_OP_NAMES, origin_op_names);
  ge::AttrUtils::GetListStr(op_desc, ge::ATTR_NAME_DATA_DUMP_ORIGIN_OP_TYPES, origin_op_types);
  if (!is_has_attr) {
    original_names.push_back(op_desc->GetName());
    original_types.push_back(op_desc->GetType());
  } else {
    if (!origin_op_names.empty()) {
      for (auto &node_name : origin_op_names) {
        if (!node_name.empty()) {
          original_names.push_back(node_name);
        }
      }
    }
    if (!origin_op_types.empty()) {
      for (auto &node_type : origin_op_types) {
        if (!node_type.empty()) {
          original_types.push_back(node_type);
        }
      }
    }
  }
}

graphStatus SetDataDumpAttrForAscBackend(NodePtr &node) {
  GE_ASSERT_NOTNULL(node->GetOpDesc());
  auto fuse_attrs = node->GetOpDesc()->GetAttrsGroup<AutoFuseAttrs>();
  GE_ASSERT_NOTNULL(fuse_attrs);
  std::vector<const ge::Node *> original_nodes;
  std::set<const ge::Node *> original_nodes_set;
  for (const ge::Node *ge_node : fuse_attrs->GetOriginNodes()) {
    GE_ASSERT_NOTNULL(ge_node);
    original_nodes_set.insert(ge_node);
  }
  for (auto ge_node : original_nodes_set) {
    original_nodes.emplace_back(ge_node);
  }
  std::vector<std::string> original_names;
  std::vector<std::string> original_types;
  for (auto &ge_node : original_nodes) {
    ge::OpDescPtr op_desc = ge_node->GetOpDesc();
    GE_ASSERT_NOTNULL(op_desc);
    (void)GetOriginNamesAndTypes(op_desc, original_names, original_types);
  }
  (void)ge::AttrUtils::SetListStr(node->GetOpDesc(), ge::ATTR_NAME_DATA_DUMP_ORIGIN_OP_NAMES, original_names);
  (void)ge::AttrUtils::SetListStr(node->GetOpDesc(), ge::ATTR_NAME_DATA_DUMP_ORIGIN_OP_TYPES, original_types);
  return GRAPH_SUCCESS;
}
}  // namespace

graphStatus AscIrLowerer::Lowering(const ComputeGraphPtr &graph) {
  TRACING_PERF_SCOPE(TracingModule::kModelCompile, "Lowering", graph->GetName());
  auto nodes = graph->GetAllNodes();
  if (std::any_of(nodes.begin(), nodes.end(), [](const NodePtr &node) {
        return IsAscBackendOpNode(node);
      })) {
    GELOGI("Skip lowering for graph %s as it has been lowered", graph->GetName().c_str());
    do_lowered_ = false;
    return GRAPH_SUCCESS;
  }
  const auto backend_spec = optimize::BackendSpec::GetInstance();
  GE_CHECK_NOTNULL(backend_spec);
  LoweringConfig config;
  config.max_loop_ops = AutoFuseConfig::LoweringConfig().max_fused_loop_ops;
  config.max_loop_loads = backend_spec->max_load_num;
  GELOGD("Max load num in one Ascbackend is %u.", config.max_loop_loads);
  config.max_buffer_readers = AutoFuseConfig::LoweringConfig().max_buffer_readers;
  GE_ASSERT_GRAPH_SUCCESS(LoweringManager::LoweringGraph(graph, config));
  GE_ASSERT_GRAPH_SUCCESS(LoweringManager::FusedLoopToAscBackendOp(graph, kAscBackendFuseConfig, counter_));
  auto graphs = graph->GetAllSubgraphs();
  if (std::find(graphs.begin(), graphs.end(), graph) == graphs.end()) {
    graphs.insert(graphs.begin(), graph);
  }
  for (const auto &subgraph : graphs) {
    GE_ASSERT_GRAPH_SUCCESS(RemoveDirectNodeUnusedEdges(subgraph));
  }
  do_lowered_ = true;
  return GRAPH_SUCCESS;
}

graphStatus AscIrLowerer::Lifting(const ComputeGraphPtr &graph) const {
  TRACING_PERF_SCOPE(TracingModule::kModelCompile, "Lifting", graph->GetName());
  if (!do_lowered_) {
    GELOGI("Skip lifting for graph %s as it is not lowered this time", graph->GetName().c_str());
    return GRAPH_SUCCESS;
  }
  if (ge::AutoFuseConfig::LoweringConfig().experimental_disable_lifting) {
    GELOGI("Skip lifting for graph %s as params disable_lifting is true", graph->GetName().c_str());
    return GRAPH_SUCCESS;
  }
  GE_ASSERT_GRAPH_SUCCESS(LowerConcatHelper::LiftingPoorPerfFusedAscBackendOps(graph));
  GE_ASSERT_GRAPH_SUCCESS(LoweringManager::LiftingOneNodeAscBackendOp(graph));
  GE_ASSERT_GRAPH_SUCCESS(DfxForAscBackendOp(graph));
  GE_ASSERT_SUCCESS(PruneUnusedNodesAfterLifting(graph), "Failed to prune unused nodes in graph %s",
                    graph->GetName().c_str());
  return GRAPH_SUCCESS;
}

graphStatus AscIrLowerer::RemoveDirectNodeUnusedEdges(const ComputeGraphPtr &graph) {
  std::deque<NodePtr> used_deque;
  for (auto &node : graph->GetDirectNode()) {
    bool force_skip_prune = false;
    (void)AttrUtils::GetBool(node->GetOpDesc(), ATTR_NAME_SKIP_PRUNE_OPTIMIZE, force_skip_prune);
    if (node->GetType() == ge::NETOUTPUT || force_skip_prune) {
      GELOGD("Add node %s to used start nodes", node->GetName().c_str());
      used_deque.push_back(node);
    }
  }
  std::set<const ge::Node *> used_nodes;
  while (!used_deque.empty()) {
    NodePtr node = used_deque.front();
    used_deque.pop_front();
    if (!used_nodes.insert(node.get()).second) {
      continue;
    }
    for (const auto &in_node : node->GetInAllNodes()) {
      GELOGD("Add node %s to used nodes", in_node->GetName().c_str());
      used_deque.push_back(in_node);
    }
  }
  for (const auto &node : graph->GetDirectNode()) {
    if (!IsAscBackendOpNode(node)) {
      continue;
    }
    const auto fuse_attr = node->GetOpDesc()->GetAttrsGroup<AutoFuseAttrs>();
    GE_ASSERT_NOTNULL(fuse_attr);
    const auto &buffers = fuse_attr->GetOriginOutputBuffers();
    std::for_each(buffers.begin(), buffers.end(),
                  [this](const OutDataAnchor *buffer) { replaced_nodes_.insert(buffer->GetOwnerNode()); });
    std::vector<std::pair<NodePtr, InDataAnchorPtr>> unused_node_and_anchors;
    for (auto &node_and_anchor : node->GetOutDataNodesAndAnchors()) {
      auto &dst_node = node_and_anchor.first;
      if (IsAscBackendOpNode(dst_node) || used_nodes.count(dst_node.get()) > 0U) {
        continue;
      }
      unused_node_and_anchors.emplace_back(node_and_anchor);
    }
    for (auto &node_and_anchor : unused_node_and_anchors) {
      auto anchor = node_and_anchor.second;
      GELOGI("Remove unused edge %s->%s before backend fuse", loop::BufferName(anchor->GetPeerOutAnchor()).c_str(),
             loop::BufferName(anchor).c_str());
      GE_WARN_ASSERT_GRAPH_SUCCESS(GraphUtils::RemoveEdge(anchor->GetPeerOutAnchor(), anchor));
    }
  }
  return GRAPH_SUCCESS;
}

graphStatus AscIrLowerer::PruneUnusedNodesAfterLifting(const ge::ComputeGraphPtr &graph) const {
  GE_ASSERT_NOTNULL(graph);
  std::set<NodePtr> seen_nodes;
  std::deque<NodePtr> nodes_to_remove;
  for (const NodePtr &node : replaced_nodes_) {
    if (node->GetOutNodes().empty()) {
      nodes_to_remove.push_back(node);
    }
  }
  for (const NodePtr &node : graph->GetAllNodes()) {
    if (IsAscBackendOpNode(node) && node->GetOutNodes().empty()) {
      nodes_to_remove.push_back(node);
    }
  }
  std::set<NodePtr> removed;
  while (!nodes_to_remove.empty()) {
    NodePtr node = nodes_to_remove.front();
    nodes_to_remove.pop_front();
    if (!removed.insert(node).second) {
      continue;
    }
    std::vector<NodePtr> input_nodes;
    for (auto &in_node : node->GetInAllNodes()) {
      input_nodes.emplace_back(in_node);
    }
    GELOGD("Remove unused node %s after lifting", node->GetName().c_str());
    (void)NodeUtils::RemoveSubgraphsOnNode(node);
    (void)graph->RemoveNode(node);
    for (auto &in_node : input_nodes) {
      if (IsNodeShouldPrune(in_node)) {
        nodes_to_remove.push_back(in_node);
      }
    }
  }
  return GRAPH_SUCCESS;
}

graphStatus AscIrLowerer::DfxForAscBackendOp(const ComputeGraphPtr &graph) const {
  GE_ASSERT_NOTNULL(graph);
  const std::string aiv_cnt_key = "_op_vectorcore_num";
  for (auto &node : graph->GetAllNodes()) {
    if (!IsAscBackendOpNode(node) && (node->GetType() != "FusedAscBackend")) {
      continue;
    }
    GE_ASSERT_NOTNULL(node->GetOpDesc());
    auto fuse_attrs = node->GetOpDesc()->GetAttrsGroup<AutoFuseAttrs>();
    if (fuse_attrs == nullptr) {
      continue;
    }
    // reconstruct the original computer graph for each ascbackend node
    GE_ASSERT_GRAPH_SUCCESS(LoweringManager::GetFusedOriginComputeGraph(*fuse_attrs, node));
    // set data dump attr for ascbackend node
    GE_ASSERT_SUCCESS(SetDataDumpAttrForAscBackend(node));

    int32_t vector_core_num = GetInterAttrs(fuse_attrs).vector_core_num;
    GE_CHECK_GE(vector_core_num, 0);
    if (vector_core_num > 0){
      (void)ge::AttrUtils::SetStr(node->GetOpDesc(), aiv_cnt_key, std::to_string(vector_core_num));
    }
  }
  return GRAPH_SUCCESS;
}
}  // namespace ge