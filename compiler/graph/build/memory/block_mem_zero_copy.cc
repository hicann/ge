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
#include "graph/utils/type_utils.h"
#include "graph/utils/op_type_utils.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/tensor_utils.h"
#include "graph/ge_context.h"
#include "graph/optimize/mem_layout_conflict_optimize/mem_layout_conflict_util.h"
#include "framework/common/op/ge_op_utils.h"
#include "graph/custom_op_factory.h"
#include "common/ge_common/ge_types.h"
#include "common/omg_util/omg_util.h"
#include "common/context/local_context.h"
#include "memory_block.h"
#include "block_mem_zero_copy.h"

namespace ge {
// 地址不可刷新时不能做零拷贝，也不能和零拷贝节点内存进行复用
// 和Data、Netoutput连接时用于判断是否可零拷贝，和其他节点连接是用于判断是否可以和零拷贝节点进行复用
bool IsNodeSupportZeroCopy(const ge::NodePtr &node) {
  bool is_support_zero_copy = ge::MemLayoutConflictUtil::IsAddressRefreshable(node);
  if (!is_support_zero_copy) {
    GELOGI("Op[%s] not support zero copy", node->GetName().c_str());
  } else {
    // IsAddressRefreshable在动态shape静态子图场景认为hccl算子都是可刷新的，实际上因为处理阶段有差异，
    // 比如hccl判断时还未拆图，可能导致结果不一致，另外静态子图里hccl支持刷新会有性能劣化（hccl内部会做数据拷贝等额外处理）
    // 这里还是保持原有处理，统一使用IsHcomNodeNotSupportAddrRefresh的结果（不可刷新）
    const auto root_graph = GraphUtils::FindRootGraph(node->GetOwnerComputeGraph());
    const bool is_dynamic_shape_sub_graph = (root_graph != nullptr) && root_graph->GetGraphUnknownFlag() &&
                                            (!node->GetOwnerComputeGraph()->GetGraphUnknownFlag());
    if (is_dynamic_shape_sub_graph && ge::OpUtils::IsHcomNodeNotSupportAddrRefresh(node->GetOpDesc())) {
      GELOGI("hccl engine op[%s] not support zero copy", node->GetName().c_str());
      is_support_zero_copy = false;
    }
  }
  return is_support_zero_copy;
}

// 编译图的子图中连接netoutput的节点不进行零拷贝
bool IsOutNodeInCurComputeGraph(const ge::Node *const node, const ge::ComputeGraphPtr &graph) {
  return (node->GetType() == ge::NETOUTPUT) && (node->GetOwnerComputeGraphBarePtr() == graph.get());
}

const std::list<ge::NodeIndexIO> &FindNodeOutputSameAnchors(const ge::NodeIndexIO &node_index_io,
                                                            const ge::AnchorToSymbol &anchor_to_symbol,
                                                            const ge::SymbolToAnchors &symbol_to_anchors) {
  static const std::list<ge::NodeIndexIO> res = {};
  const auto &symbol_iter = anchor_to_symbol.find(node_index_io.ToString());
  if (symbol_iter != anchor_to_symbol.cend()) {
    const auto &anchors_iter = symbol_to_anchors.find(symbol_iter->second);
    if (anchors_iter != symbol_to_anchors.cend()) {
      return anchors_iter->second;
    }
  }
  return res;
}

size_t GetOutputFlowToNetoutputNum(const ge::NodePtr &node, uint32_t output_index, const ge::ComputeGraphPtr &graph,
                                   const ge::SymbolToAnchors &symbol_to_anchors,
                                   const ge::AnchorToSymbol &anchor_to_symbol) {
  auto out_anchor = node->GetOutDataAnchor(static_cast<int32_t>(output_index));
  if (out_anchor == nullptr) {
    return 0U;
  }
  size_t num_anchors_to_netoutput = 0U;
  ge::NodeIndexIO out_node_index_io(node, output_index, ge::kOut);
  const auto &same_anchors = FindNodeOutputSameAnchors(out_node_index_io, anchor_to_symbol, symbol_to_anchors);
  bool include_not_support_zero_copy_node = false;
  if (!same_anchors.empty()) {
    for (const auto &node_index_io : same_anchors) {
      if ((node_index_io.io_type_ != ge::kIn) && (node_index_io.node_ptr_ != nullptr) &&
          (node_index_io.node_ptr_->GetOpDescBarePtr() != nullptr)) {
        if (!IsNodeSupportZeroCopy(node_index_io.node_)) {
          include_not_support_zero_copy_node = true;
        }
        continue;
      }

      if (IsOutNodeInCurComputeGraph(node_index_io.node_.get(), graph)) {  // Combined with NET-OUTPUT of root graph
        num_anchors_to_netoutput++;
      }
    }
  }

  if (num_anchors_to_netoutput > 0U) {
    GELOGI("Node %s output %u flow to %zu root graph output", node->GetNamePtr(), output_index,
           num_anchors_to_netoutput);
    if (include_not_support_zero_copy_node) {
      GELOGI("Node %s output %u symbol is same as not support zero copy node", node->GetNamePtr(), output_index);
      num_anchors_to_netoutput = 0U;
    }
  }
  return num_anchors_to_netoutput;
}

void MarkZeroCopyBlockAttr(std::vector<TAttr<bool>> &bool_attr, const OpDesc *const op_desc, bool is_zero_copy,
                           OpMemoryType mem_type, uint32_t out_index) {
  if (is_zero_copy && (mem_type == OpMemoryType::kOutput)) {
    auto output_desc = op_desc->MutableOutputDesc(out_index);
    if (output_desc != nullptr) {
      bool_attr.emplace_back(output_desc.get(), op_desc, out_index, ATTR_IS_ZERO_COPY_BLOCK, true);
    } else {
      GELOGE(PARAM_INVALID, "Node %s output %u is zero copy block but not marked as output desc is nullptr",
             op_desc->GetNamePtr(), out_index);
    }
  }
}

bool IsOutputIndexRef(const OpDesc *const op_desc, uint32_t index) {
  auto output_tensor = op_desc->GetOutputDescPtr(index);
  if (output_tensor == nullptr) {
    return false;
  }
  bool dst_reuse_input = false;
  (void)ge::TensorUtils::GetReuseInput(*output_tensor, dst_reuse_input);
  if (dst_reuse_input) {
    return true;
  }

  bool is_ref = false;
  (void)ge::AttrUtils::GetBool(op_desc, ATTR_NAME_REFERENCE, is_ref);
  if (is_ref) {
    std::string output_name = op_desc->GetOutputNameByIndex(index);
    for (const auto &input_name : op_desc->GetAllInputNames()) {
      if (output_name == input_name) {
        return true;
      }
    }
  }
  return false;
}

bool IsSubgraphDataRefConstInput(const NodePtr &node) {
  std::string op_type;
  const auto &in_node = ge::NodeUtils::GetParentInput(node);
  return ge::NodeUtils::GetConstOpType(in_node, op_type) ||
         ((in_node != nullptr) && ge::OpTypeUtils::IsVariableNode(in_node->GetType()));
}

bool IsOutputBlock(const ge::InDataAnchor *const in_data_anchor) {
  auto peer_out_anchor = in_data_anchor->GetPeerOutAnchor();
  GE_IF_BOOL_EXEC(peer_out_anchor == nullptr, REPORT_INNER_ERR_MSG("E19999", "Peer out anchor is nullptr.");
                  GELOGE(FAILED, "[Check][Param] Peer out anchor is nullptr."); return false);
  auto src = peer_out_anchor->GetOwnerNodeBarePtr();
  int32_t index = peer_out_anchor->GetIdx();
  auto iter = GetLocalOmgContext().out_nodes_map.find(src->GetNamePtr());
  if (iter != GetLocalOmgContext().out_nodes_map.end()) {
    for (auto id : iter->second) {
      if (index == id) {
        return true;
      }
    }
  }
  return false;
}

bool IsKnownSubgraphData(const Node *node) {
  if ((node == nullptr) || NodeUtils::IsDynamicShape(*node)) {
    return false;
  }

  return node->GetOpDescBarePtr()->HasAttr(ATTR_NAME_PARENT_NODE_INDEX);
}

void SetReleaseBlockLifeEnd(MemoryBlock *to_release, int64_t stream_id) {
  const auto &to_release_out_stream_life_time = to_release->NodeTypeIndexList().back().out_stream_life_time_;
  if (to_release_out_stream_life_time.size() == 1) {
    for (const auto &item : to_release_out_stream_life_time) {
      size_t end_life_time = item.second.second;
      int64_t release_stream_id = to_release->stream_id_;
      if (to_release->stream_id_ != item.first) {
        // kMaxLifeTime means cannot return self stream, so need to set kMaxLifeTime to self stream, set real end time
        // to out stream.
        if (end_life_time == kMaxLifeTime) {
          release_stream_id = stream_id;
          end_life_time = item.second.first;
        }
      }
      to_release->SetLifeTimeEnd(end_life_time, release_stream_id);
      // stream 0->1->2->0场景，增加一个0->1的结束点，2上面的内存有机会和0上的复用
      if (to_release->diff_stream_prior_ && (to_release->stream_id_ != stream_id)) {
        to_release->SetLifeTimeEnd(item.second.first, stream_id);
      }
    }
    return;
  }
  size_t max_end_life_time = 0U;
  for (const auto &item : to_release_out_stream_life_time) {
    if (max_end_life_time < item.second.second) {
      max_end_life_time = item.second.second;
    }
  }
  if (max_end_life_time == kMaxLifeTime) {
    to_release->same_stream_ = false;
  }
  to_release->SetLifeTimeEnd(max_end_life_time, to_release->stream_id_);
}

void MarkReuseZeroCopyBlockFlag(const NodePtr &n, MemoryBlock *const block, const uint32_t index) {
  auto node_op_desc = n->GetOpDescBarePtr();

  // 输出连续内存不能做零拷贝，同时NoPadding且Reuse的也不能做零拷贝
  bool can_reuse_zero_copy = true;
  bool is_continuous = ge::MemLayoutConflictUtil::IsContinuousOutput(n);
  bool is_nopadding_continuous = false;

  if (!is_continuous) {
    (void)ge::AttrUtils::GetBool(*node_op_desc, ATTR_NAME_NOPADDING_CONTINUOUS_OUTPUT, is_nopadding_continuous);
    if (is_nopadding_continuous) {
      bool attr_reuse = false;
      (void)ge::AttrUtils::GetBool(*node_op_desc, ATTR_NAME_OUTPUT_REUSE_INPUT, attr_reuse);
      can_reuse_zero_copy = !attr_reuse;
    }
  } else {
    can_reuse_zero_copy = false;
  }

  if (!can_reuse_zero_copy) {
    block->is_reuse_zero_copy_ = false;
  }

  GELOGD("Node name: %s index: %d, can_reuse_zero_copy: %s.", n->GetNamePtr(), index,
         can_reuse_zero_copy ? "true" : "false");
}

bool IsNodeAndPeerNodeTaskSupportZeroCopy(const ge::NodePtr &node, uint32_t output_index) {
  GELOGD("Check node %s and peer node of output %u task zero copy supported", node->GetNamePtr(), output_index);
  if (!IsNodeSupportZeroCopy(node)) {
    return false;
  }

  const auto out_anchor = node->GetOutDataAnchor(static_cast<int32_t>(output_index));
  if (out_anchor == nullptr) {
    return false;
  }
  const auto peer_anchors = out_anchor->GetPeerInDataAnchorsPtr();
  const bool support = std::all_of(peer_anchors.begin(), peer_anchors.end(), [](const ge::InDataAnchor *anchor) {
    return ((anchor != nullptr) && (anchor->GetOwnerNodeBarePtr() != nullptr) &&
            IsNodeSupportZeroCopy(anchor->GetOwnerNode()));
  });

  GELOGD("Task of node %s and peer node of output %u %s zero copy", node->GetNamePtr(), output_index,
         (support ? "support" : "not support"));
  return support;
}

}  // namespace ge
