/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph/build/memory/block_mem_assigner.h"
#include <cinttypes>
#include <algorithm>
#include "common/checker.h"
#include "graph/debug/ge_attr_define.h"
#include "memory_block.h"

namespace ge {
namespace {
static bool CompareNodeId(const ge::InDataAnchor *const left, const ge::InDataAnchor *const right) {
  bool invalid_para =
      ((left == nullptr) || (left->GetPeerOutAnchor() == nullptr) ||
       (left->GetPeerOutAnchor()->GetOwnerNodeBarePtr() == nullptr) ||
       (left->GetPeerOutAnchor()->GetOwnerNodeBarePtr()->GetOpDescBarePtr() == nullptr) || (right == nullptr) ||
       (right->GetPeerOutAnchor() == nullptr) || (right->GetPeerOutAnchor()->GetOwnerNodeBarePtr() == nullptr) ||
       (right->GetPeerOutAnchor()->GetOwnerNodeBarePtr()->GetOpDescBarePtr() == nullptr));
  if (invalid_para) {
    return false;
  }
  return (left->GetPeerOutAnchor()->GetOwnerNodeBarePtr()->GetOpDescBarePtr()->GetId() <
          right->GetPeerOutAnchor()->GetOwnerNodeBarePtr()->GetOpDescBarePtr()->GetId());
}

}  // namespace

void GetDiffStreamMinLifeTime(const Node *const node, const int64_t src_stream,
                              const DiffStreamEdgeLife &in_stream_edge, int64_t &min_life_time) {
  const auto node_op_desc = node->GetOpDescBarePtr();
  const auto dst_stream = MemReuseUtils::GetStreamId(node_op_desc);
  if (dst_stream == src_stream) {
    min_life_time = node_op_desc->GetId();
    GELOGI("same stream, node[%s] id as min life[%" PRId64 "]", node_op_desc->GetNamePtr(), min_life_time);
    return;
  }

  const auto it = in_stream_edge.find(dst_stream);
  if (it != in_stream_edge.cend()) {
    const auto edges_it = it->second.find(src_stream);
    if (edges_it != it->second.cend()) {
      auto edge_it = edges_it->second.lower_bound({static_cast<size_t>(node_op_desc->GetId()), 0UL});
      if (edge_it != edges_it->second.end()) {
        if ((edge_it->node_id > static_cast<size_t>(node_op_desc->GetId())) && (edge_it != edges_it->second.begin())) {
          --edge_it;
        }
        if (edge_it->node_id <= static_cast<size_t>(node_op_desc->GetId())) {
          min_life_time = (*edge_it).peer_node_id;
          GELOGI("diff stream, get min life[%" PRId64 "], node[%s], id[%" PRId64 "], stream_id:[%" PRId64 "<-%" PRId64
                 "] life_time:[%zu<-%zu]",
                 min_life_time, node_op_desc->GetNamePtr(), node_op_desc->GetId(), dst_stream, src_stream,
                 (*edge_it).node_id, min_life_time);
          return;
        }
      }
    }
  }
  min_life_time = kMinLifeTime;
  GELOGI("diff stream, get default min life[%" PRId64 "], node[%s], id[%" PRId64 "], stream_id[%" PRId64 "<-%" PRId64
         "]",
         min_life_time, node_op_desc->GetNamePtr(), node_op_desc->GetId(), dst_stream, src_stream);
}

// Ensure that the memory release order is consistent with the topo order
std::vector<ge::InDataAnchor *> GetSortAllInDataAnchors(const ge::NodePtr &node, const bool memory_priority_mode) {
  std::vector<ge::InDataAnchor *> anchors;
  for (const auto in_anchor : node->GetAllInDataAnchorsPtr()) {
    anchors.emplace_back(in_anchor);
  }
  if (memory_priority_mode) {
    std::sort(anchors.begin(), anchors.end(), CompareNodeId);
  }
  return anchors;
}

void HandleDependentStreamRedundantInfo(
    std::pair<const int64_t, std::map<int64_t, std::set<ge::EdgeLife, ge::CompareEdgeLife>>> &in_stream_edge) {
  for (auto &depend_stream_info : in_stream_edge.second) {
    size_t in_node_id = 0U;
    size_t out_node_id = 0U;
    if (depend_stream_info.second.size() <= 1U) {
      continue;
    }
    for (auto iter = depend_stream_info.second.begin(); iter != depend_stream_info.second.end();) {
      if ((iter->node_id >= in_node_id) && (iter->peer_node_id <= out_node_id)) {
        GELOGI("[StreamEdge]In depend Node: stream_id:[%" PRId64 "<-%" PRId64 "] life_time:[%zu<-%zu] delete",
               in_stream_edge.first, depend_stream_info.first, iter->node_id, iter->peer_node_id);
        iter = depend_stream_info.second.erase(iter);
        continue;
      }
      in_node_id = iter->node_id;
      out_node_id = iter->peer_node_id;
      ++iter;
    }
  }
}

void HandleInStreamRedundantDependence(ge::DiffStreamEdgeLife &in_stream_edges) {
  for (auto &in_stream_edge : in_stream_edges) {
    HandleDependentStreamRedundantInfo(in_stream_edge);
  }
}

void GetDiffStreamMaxLifeTime(const Node *const node, const int64_t stream_id,
                              const DiffStreamEdgeLife &diff_stream_edge_life, int64_t &max_life_time) {
  max_life_time = kMaxLifeTime;
  auto node_op_desc = node->GetOpDescBarePtr();
  GE_CHECK_NOTNULL_JUST_RETURN(node_op_desc);
  const auto node_stream_id = MemReuseUtils::GetStreamId(node_op_desc);
  GELOGD("Out depend node:[%s] life begin:%" PRId64 " stream_id:[%" PRId64 "->%" PRId64 "]", node_op_desc->GetNamePtr(),
         node_op_desc->GetId(), node_stream_id, stream_id);
  if (node_stream_id == stream_id) {
    max_life_time = node_op_desc->GetId();
    return;
  }
  const auto it = diff_stream_edge_life.find(node_stream_id);
  if (it == diff_stream_edge_life.cend()) {
    return;
  }
  const auto edges_it = it->second.find(stream_id);
  if (edges_it == it->second.cend()) {
    return;
  }
  const auto edge_it = edges_it->second.lower_bound({static_cast<size_t>(node_op_desc->GetId()), 0UL});
  if (edge_it == edges_it->second.end()) {
    return;
  }
  GELOGD("Node:[%s] life begin:%" PRId64 " stream_id:[%" PRId64 "->%" PRId64 "] life_time:[%" PRId64 "->%" PRId64 "]",
         node_op_desc->GetNamePtr(), node_op_desc->GetId(), node_stream_id, stream_id, (*edge_it).node_id,
         (*edge_it).peer_node_id);
  max_life_time = (*edge_it).peer_node_id;
}

int64_t GetNodeMaxLifeBySymbol(const SymbolToAnchors &symbol_to_anchors, const Node *const n, uint32_t out_index,
                               int64_t &max_node_life_time_by_symbol, std::set<int64_t> &streams,
                               const DiffStreamEdgeLife &diff_stream_edge_life, int64_t stream_id = kInvalidStreamId) {
  NodeIndexIO out_node_index_io(n, out_index, kOut);
  const int64_t n_stream_id =
      (stream_id == kInvalidStreamId) ? MemReuseUtils::GetStreamId(n->GetOpDescBarePtr()) : stream_id;
  SymbolToAnchors::const_iterator iter = symbol_to_anchors.find(out_node_index_io.ToString());
  // 先初始化返回值的为该节点本身的起使生命周期
  int64_t max_node_life_time = n->GetOpDescBarePtr()->GetId();
  if (iter != symbol_to_anchors.cend()) {
    for (const auto &node_index_io : iter->second) {
      if ((node_index_io.io_type_ != kIn) || (node_index_io.node_ptr_ == nullptr) ||
          (node_index_io.node_ptr_->GetOpDescBarePtr() == nullptr)) {
        continue;
      }
      const int64_t in_anchor_stream_id = MemReuseUtils::GetStreamId(node_index_io.node_ptr_->GetOpDescBarePtr());
      if (node_index_io.node_ptr_->GetOpDescBarePtr()->GetOpKernelLibName() != kEngineNameGeLocal) {
        streams.emplace(in_anchor_stream_id);
      }
      /* max_node_life_time_by_symbol 返回值有使用，在函数SetOutStreamLifeTime会使用，不能赋值错误 */
      if (node_index_io.node_ptr_->GetOpDescBarePtr()->GetId() > max_node_life_time_by_symbol) {
        max_node_life_time_by_symbol = node_index_io.node_ptr_->GetOpDescBarePtr()->GetId();
        max_node_life_time =
            (max_node_life_time_by_symbol > max_node_life_time) ? max_node_life_time_by_symbol : max_node_life_time;
        GELOGI("Node[%s] stream[%" PRId64 "] output[%u]'s life time by symbol [%" PRId64 "][%" PRId64
               "], node_io[%s], stream_id[%" PRId64 "].",
               n->GetNamePtr(), n_stream_id, out_index, max_node_life_time_by_symbol, max_node_life_time,
               node_index_io.node_ptr_->GetNamePtr(), in_anchor_stream_id);
      }
      if (n_stream_id != in_anchor_stream_id) {
        int64_t diff_stream_life_time_end = kMaxLifeTime;
        GetDiffStreamMaxLifeTime(node_index_io.node_ptr_, n_stream_id, diff_stream_edge_life,
                                 diff_stream_life_time_end);
        GELOGI("Node[%s] stream[%" PRId64 "] output[%u]'s life time is max of [%" PRId64 "][%" PRId64 "][%" PRId64
               "], node_io[%s], stream_id[%" PRId64 "].",
               n->GetNamePtr(), n_stream_id, out_index, max_node_life_time_by_symbol, max_node_life_time,
               diff_stream_life_time_end, node_index_io.node_ptr_->GetNamePtr(), in_anchor_stream_id);
        /* max_node_life_time_by_symbol 在此分支中不能赋值,此分支只影响最大值 */
        max_node_life_time =
            std::max(max_node_life_time_by_symbol, std::max(diff_stream_life_time_end, max_node_life_time));
      }
    }
  }

  // info日志, 打印node的生命周期
  GELOGI("Node[%s] output[%u]'s max life time[%" PRId64 "][%" PRId64 "].", n->GetNamePtr(), out_index,
         max_node_life_time_by_symbol, max_node_life_time);

  return max_node_life_time;
}

int64_t GetNodeMaxLife(const SymbolToAnchors &symbol_to_anchors, const DiffStreamEdgeLife &diff_stream_edge_life,
                       const Node *const n, uint32_t out_index, int64_t &max_node_life_time_by_symbol,
                       std::set<int64_t> &streams, int64_t stream_id = kInvalidStreamId) {
  const int64_t max_node_life_time = GetNodeMaxLifeBySymbol(
      symbol_to_anchors, n, out_index, max_node_life_time_by_symbol, streams, diff_stream_edge_life, stream_id);
  GELOGD("Node[%s] output[%u]'s max life time[%" PRId64 "].", n->GetNamePtr(), out_index, max_node_life_time);
  return max_node_life_time;
}

void GetContinuousOutputMaxLife(const NodePtr &node, const SymbolToAnchors &symbol_to_anchors,
                                const DiffStreamEdgeLife &out_stream_edges, int64_t &max_life_time,
                                std::set<int64_t> &streams) {
  auto node_op_desc = node->GetOpDescBarePtr();
  GE_CHECK_NOTNULL_JUST_RETURN(node_op_desc);
  for (uint32_t index = 0U; index < static_cast<uint32_t>(node_op_desc->GetOutputsSize()); index++) {
    const int64_t life_time =
        GetNodeMaxLife(symbol_to_anchors, out_stream_edges, node.get(), index, max_life_time, streams);
    if (life_time > max_life_time) {
      max_life_time = life_time;
    }
  }
  GELOGI("Continuous output node:%s max life time:%" PRId64 "", node->GetNamePtr(), max_life_time);
}

void GetContinuousOutputMaxLifeBySymbol(const Node *const node, const SymbolToAnchors &symbol_to_anchors,
                                        int64_t &max_life_time, const DiffStreamEdgeLife &diff_stream_edge_life) {
  std::set<int64_t> streams;
  const auto node_op_desc = node->GetOpDescBarePtr();
  GE_CHECK_NOTNULL_JUST_RETURN(node_op_desc);
  for (uint32_t index = 0U; index < static_cast<uint32_t>(node_op_desc->GetOutputsSize()); index++) {
    /* max_life_time 在此函数中已经进行最大值的赋值处理 */
    (void)GetNodeMaxLifeBySymbol(symbol_to_anchors, node, index, max_life_time, streams, diff_stream_edge_life);
  }
  GELOGI("Continuous output node:%s max life time:%" PRId64 " by symbol", node->GetNamePtr(), max_life_time);
}

void SetLastUsedInputMemAttr(const NodePtr &node, int32_t input_index, std::vector<TAttr<bool>> &bool_attr) {
  if (node == nullptr) {
    return;
  }
  auto node_op_desc = node->GetOpDescBarePtr();
  if (node_op_desc != nullptr) {
    auto input_desc = node_op_desc->MutableInputDesc(input_index);
    if (input_desc == nullptr) {
      return;
    }
    bool_attr.emplace_back(input_desc.get(), node_op_desc, input_index, ATTR_NAME_IS_END_OF_INPUTMEM_LIFECYCLE, true);
  }
}

/*
 * stream1  stream2
 *   10--+
 *   20   \   30
 *   40 ---\->50
 *   60     \ 70
 *            90
 * 10->90
 * 40->50
 * only keep edge 40->50, 可以简单记为缩短peer_node_id与node_id差值
 */
void EraseIntersectedEdge(std::set<EdgeLife, CompareEdgeLife> &in_edge_set, const EdgeLife &old_in_edge,
                          const EdgeLife &new_in_edge, const int64_t src_node_stream_id,
                          const int64_t dst_node_stream_id) {
  if (((old_in_edge.node_id > new_in_edge.node_id) && (old_in_edge.peer_node_id < new_in_edge.peer_node_id)) ||
      ((old_in_edge.node_id < new_in_edge.node_id) && (old_in_edge.peer_node_id > new_in_edge.peer_node_id))) {
    GELOGI("[StreamEdge]In depend Node: stream_id:[%" PRId64 "<-%" PRId64
           "] erase life_time:[%zu<-%zu], will insert new "
           "life_time:[%zu<-%zu].",
           dst_node_stream_id, src_node_stream_id, old_in_edge.node_id, old_in_edge.peer_node_id, new_in_edge.node_id,
           new_in_edge.peer_node_id);
    auto it = in_edge_set.find(old_in_edge);
    if ((it != in_edge_set.end()) && (it->peer_node_id == old_in_edge.peer_node_id)) {
      in_edge_set.erase(it);
    }
  }
}

Status GetNetoutputInNodeStream(const Node *const netoutput, const Node *const parent_node,
                                std::unordered_map<const Node *, std::vector<int64_t>> &parent_nodes_to_stream_ids) {
  auto &parent_node_to_stream_ids = parent_nodes_to_stream_ids[parent_node];
  const auto &netoutput_op_desc = netoutput->GetOpDesc();
  const auto input_size = netoutput->GetAllInDataAnchorsSize();
  for (uint32_t i = 0U; i < input_size; ++i) {
    const auto input_desc = netoutput_op_desc->GetInputDesc(i);
    uint32_t parent_out_index = 0U;
    if (!AttrUtils::GetInt(input_desc, ATTR_NAME_PARENT_NODE_INDEX, parent_out_index) ||
        parent_out_index >= parent_node_to_stream_ids.size() ||
        parent_node_to_stream_ids[parent_out_index] == kInvalidStreamId) {
      continue;
    }
    const auto in_data_anchor = netoutput->GetInDataAnchor(i);
    GE_ASSERT_NOTNULL(in_data_anchor);
    GE_ASSERT_NOTNULL(in_data_anchor->GetPeerOutAnchor());
    const auto input_node = in_data_anchor->GetPeerOutAnchor()->GetOwnerNodeBarePtr();
    GE_ASSERT_NOTNULL(input_node);
    const auto input_node_iter = parent_nodes_to_stream_ids.find(input_node);
    int64_t input_node_stream_id = kInvalidStreamId;
    // input_node is not a parent node
    if (input_node_iter == parent_nodes_to_stream_ids.end()) {
      input_node_stream_id = MemReuseUtils::GetStreamId(input_node->GetOpDescBarePtr());
    } else {
      const auto input_node_out_index = in_data_anchor->GetPeerOutAnchor()->GetIdx();
      GE_ASSERT_TRUE(static_cast<size_t>(input_node_out_index) <= input_node_iter->second.size(),
                     "input_node_out_index: %d, input node output size: %zu, input node: %s", input_node_out_index,
                     input_node_iter->second.size(), input_node->GetNamePtr());
      input_node_stream_id = input_node_iter->second[input_node_out_index];
    }
    if ((parent_node_to_stream_ids[parent_out_index] != kParentNodeDefaultStreamId) &&
        (parent_node_to_stream_ids[parent_out_index] != input_node_stream_id)) {
      GELOGI(
          "subgraph node has multi streams, set no reuse. node:%s(%s) output: %u, new stream id: %lld,"
          ", original stream id: %lld, input_node: %s",
          parent_node->GetNamePtr(), parent_node->GetTypePtr(), parent_out_index, input_node_stream_id,
          parent_node_to_stream_ids[parent_out_index], input_node->GetNamePtr());
      input_node_stream_id = kInvalidStreamId;  // means no reuse
    }
    parent_node_to_stream_ids[parent_out_index] = input_node_stream_id;
    GELOGI("get stream id from subgraph node. node:%s(%s) output: %u stream id: %lld, input_node: %s",
           parent_node->GetNamePtr(), parent_node->GetTypePtr(), parent_out_index, input_node_stream_id,
           input_node->GetNamePtr());
  }
  return SUCCESS;
}

std::set<int64_t> GetStreamMergeAndOutStreams(const ge::ComputeGraphPtr &graph) {
  std::set<int64_t> merge_and_out_streams;
  for (const NodePtr &node : graph->GetAllNodes()) {
    if (!MemReuseUtils::IsMergeNode(node)) {
      continue;
    }
    if (merge_and_out_streams.insert(MemReuseUtils::GetStreamId(node->GetOpDescBarePtr())).second) {
      GELOGD("Stream %" PRId64 " not reuse memory with other streams",
             MemReuseUtils::GetStreamId(node->GetOpDescBarePtr()));
    }
    for (const auto &out_node : node->GetOutAllNodes()) {
      if (merge_and_out_streams.insert(MemReuseUtils::GetStreamId(out_node->GetOpDescBarePtr())).second) {
        GELOGD("Stream %" PRId64 " not reuse memory with other streams",
               MemReuseUtils::GetStreamId(out_node->GetOpDescBarePtr()));
      }
    }
  }
  return merge_and_out_streams;
}

}  // namespace ge
