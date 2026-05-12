/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph/build/stream/dag_adapter.h"
#include "framework/common/debug/ge_log.h"
#include "common/checker.h"

namespace minidag {
ge::graphStatus DAGAdapter::ToGEStatus(graphStatus status) {
  switch (status) {
    case graphStatus::SUCCESS:
      return ge::GRAPH_SUCCESS;
    case graphStatus::FAILED:
      return ge::GRAPH_FAILED;
    case graphStatus::INVALID_NODE:
    case graphStatus::INVALID_EDGE:
      return ge::GRAPH_FAILED;
    case graphStatus::NODE_NOT_FOUND:
      return ge::GE_GRAPH_GRAPH_NODE_NULL;
    default:
      return ge::GRAPH_FAILED;
  }
}

ge::graphStatus DAGAdapter::FromGEGraph(const ge::ConstGraphPtr &ge_graph,
                                         std::shared_ptr<DAGGraph> &dag) {
  if (ge_graph == nullptr) {
    GELOGE(ge::GRAPH_FAILED, "FromGEGraph failed: ge_graph is null");
    return ge::GRAPH_FAILED;
  }

  // 1. 获取图名称
  ge::AscendString name;
  GE_ASSERT_SUCCESS(ge_graph->GetName(name), "FromGEGraph failed: GetName returned error");
  GELOGI("FromGEGraph start: graph name=%s", name.GetString());
  dag = std::make_shared<DAGGraph>(name.GetString());
  GE_ASSERT_NOTNULL(dag, "FromGEGraph failed: create DAGGraph failed");

  // 2. 转换节点（并设置topo_id）
  auto nodes_ret = ConvertNodes(ge_graph, *dag);
  if (nodes_ret != graphStatus::SUCCESS) {
    GELOGE(ge::GRAPH_FAILED, "FromGEGraph failed: ConvertNodes returned %d",
           static_cast<int>(nodes_ret));
    return ToGEStatus(nodes_ret);  // 传递 ConvertNodes 错误
  }

  // 3. 转换边（数据边和控制边）
  auto edges_ret = ConvertEdges(ge_graph, *dag);
  if (edges_ret != graphStatus::SUCCESS) {
    GELOGE(ge::GRAPH_FAILED, "FromGEGraph failed: ConvertEdges returned %d",
           static_cast<int>(edges_ret));
    return ToGEStatus(edges_ret);  // 传递 ConvertEdges 错误
  }

  GELOGI("FromGEGraph done: nodes=%zu, edges=%zu",
          dag->GetNodeCount(), dag->GetEdgeCount());
  return ge::GRAPH_SUCCESS;
}

graphStatus DAGAdapter::ConvertNodes(const ge::ConstGraphPtr &ge_graph, DAGGraph &dag) {
  auto gnodes = ge_graph->GetDirectNode();
  int64_t topo_id = 0;

  for (const auto& gnode : gnodes) {
    ge::AscendString name, type;
    GE_ASSERT_SUCCESS(gnode.GetName(name), "ConvertNodes failed: GetName failed for gnode");
    GE_ASSERT_SUCCESS(gnode.GetType(type), "ConvertNodes failed: GetType failed for gnode %s", name.GetString());

    auto dag_node = dag.AddNode(name.GetString(), type.GetString());
    if (dag_node == nullptr) {
      GELOGE(ge::GRAPH_FAILED, "ConvertNodes failed: duplicate node name %s, type %s",
              name.GetString(), type.GetString());
      return graphStatus::DUPLICATE_NODE;
    }

    dag_node->SetTopoId(topo_id++);
  }

  GELOGI("ConvertNodes done: total=%zu", gnodes.size());
  return graphStatus::SUCCESS;
}

graphStatus DAGAdapter::ConvertEdges(const ge::ConstGraphPtr &ge_graph, DAGGraph &dag) {
  int64_t data_edge_count = 0;
  int64_t control_edge_count = 0;

  for (const auto &gnode : ge_graph->GetDirectNode()) {
    ge::AscendString src_name;
    GE_ASSERT_SUCCESS(gnode.GetName(src_name), "ConvertEdges failed: GetName failed for src gnode");
    auto src_node = dag.FindNode(src_name.GetString());
    if (src_node == nullptr) {
      GELOGE(ge::GRAPH_FAILED, "ConvertEdges failed: src_node not found in dag: %s", src_name.GetString());
      return graphStatus::NODE_NOT_FOUND;
    }

    auto data_ret = ConvertDataEdgesForNode(gnode, src_node, dag, data_edge_count);
    if (data_ret != graphStatus::SUCCESS) {
      return data_ret;
    }

    auto control_ret = ConvertControlEdgesForNode(gnode, src_node, dag, control_edge_count);
    if (control_ret != graphStatus::SUCCESS) {
      return control_ret;
    }
  }

  GELOGI("ConvertEdges done: data_edges=%ld, control_edges=%ld",
          data_edge_count, control_edge_count);
  return graphStatus::SUCCESS;
}

graphStatus DAGAdapter::ConvertDataEdgesForNode(
    const ge::GNode &gnode,
    const std::shared_ptr<DAGNode> &src_node,
    DAGGraph &dag,
    int64_t &edge_count) {
  for (size_t i = 0; i < gnode.GetOutputsSize(); ++i) {
    auto dst_pairs = gnode.GetOutDataNodesAndPortIndexs(static_cast<int32_t>(i));
    for (const auto& [dst_gnode, dst_port] : dst_pairs) {
      if (dst_gnode == nullptr) {
        continue;
      }
      ge::AscendString dst_name;
      if (dst_gnode->GetName(dst_name) != ge::GRAPH_SUCCESS) {
        GELOGE(ge::GRAPH_FAILED, "ConvertDataEdgesForNode failed: GetName failed for dst gnode");
        return graphStatus::INVALID_NODE;
      }
      auto dst_node = dag.FindNode(dst_name.GetString());
      if (dst_node == nullptr) {
        GELOGE(ge::GRAPH_FAILED, "ConvertDataEdgesForNode failed: dst_node not found: %s", dst_name.GetString());
        return graphStatus::NODE_NOT_FOUND;
      }

      graphStatus ret = dag.AddEdge(src_node, static_cast<int32_t>(i), dst_node, dst_port);
      if (ret != graphStatus::SUCCESS) {
        GELOGE(ge::GRAPH_FAILED, "ConvertDataEdgesForNode failed: AddEdge failed");
        return ret;
      }
      ++edge_count;
    }
  }
  return graphStatus::SUCCESS;
}

graphStatus DAGAdapter::ConvertControlEdgesForNode(
    const ge::GNode &gnode,
    const std::shared_ptr<DAGNode> &src_node,
    DAGGraph &dag,
    int64_t &edge_count) {
  for (const auto &dst_gnode : gnode.GetOutControlNodes()) {
    if (dst_gnode == nullptr) {
      continue;
    }
    ge::AscendString dst_name;
    GE_ASSERT_SUCCESS(dst_gnode->GetName(dst_name),
                       "ConvertControlEdgesForNode failed: GetName failed for dst gnode");
    auto dst_node = dag.FindNode(dst_name.GetString());
    if (dst_node == nullptr) {
      GELOGE(ge::GRAPH_FAILED, "ConvertControlEdgesForNode failed: dst_node not found: %s", dst_name.GetString());
      return graphStatus::NODE_NOT_FOUND;
    }

    graphStatus ret = dag.AddEdge(src_node, -1, dst_node, -1);
    if (ret != graphStatus::SUCCESS) {
      GELOGE(ge::GRAPH_FAILED, "ConvertControlEdgesForNode failed: AddEdge failed");
      return ret;
    }
    ++edge_count;
  }
  return graphStatus::SUCCESS;
}

ge::graphStatus DAGAdapter::RefreshStreamIdsToGE(
    const DAGGraph &dag,
    const ge::ConstGraphPtr &ge_graph,
    ge::StreamPassContext &context) {
  if (ge_graph == nullptr) {
    GE_LOGE("RefreshStreamIdsToGE failed: ge_graph is null");
    return ge::GRAPH_FAILED;
  }

  int64_t success_count = 0;
  int64_t skip_count = 0;
  int64_t filtered_count = 0;

  for (const auto &dag_node : dag.GetAllNodes()) {
    // 通过节点名称查找GE GNode
    ge::AscendString node_name(dag_node->GetName().c_str());
    auto gnode = ge_graph->FindNodeByName(node_name);
    if (gnode == nullptr) {
      GELOGD("Node not found in GE graph: %s", dag_node->GetName().c_str());
      ++skip_count;
      continue;
    }

    // Adapter 层过滤：跳过 Data/NetOutput 类型节点
    ge::AscendString node_type;
    if (gnode->GetType(node_type) != ge::GRAPH_SUCCESS) {
      GELOGW("GetType failed for node %s, treat as unknown type and skip",
             dag_node->GetName().c_str());
      ++skip_count;
      continue;
    }
    std::string type_str(node_type.GetString());
    if (type_str == "Data" || type_str == "NetOutput") {
      GELOGD("Skip Data/NetOutput node: %s", dag_node->GetName().c_str());
      ++filtered_count;
      continue;
    }

    int64_t stream_id = dag_node->GetStreamId();
    if (stream_id == INVALID_STREAM_ID) {
      GELOGD("Node %s has invalid stream_id, skip", dag_node->GetName().c_str());
      ++skip_count;
      continue;
    }

    // 设置到GE
    auto ret = context.SetStreamId(*gnode, stream_id);
    if (ret != ge::GRAPH_SUCCESS) {
      GE_LOGE("SetStreamId failed for node %s, stream_id=%ld, ret=%d",
              dag_node->GetName().c_str(), stream_id, ret);
      return ret;
    }
    ++success_count;
  }

  GELOGI("RefreshStreamIdsToGE done: success=%ld, skip=%ld, filtered=%ld",
          success_count, skip_count, filtered_count);
  return ge::GRAPH_SUCCESS;
}
}  // namespace minidag