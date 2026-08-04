/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GE_GRAPH_BUILD_MEMORY_GRAPH_MEM_ASSIGNER_H_
#define GE_GRAPH_BUILD_MEMORY_GRAPH_MEM_ASSIGNER_H_

#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include "framework/common/ge_inner_error_codes.h"
#include "graph/node.h"
#include "rt_external_mem.h"
#include "graph/build/memory/hybrid_mem_assigner.h"
#include "graph/build/memory/graph_mem_splitter.h"
#include "graph/build/memory/atomic_memory_assigner.h"

namespace ge {

class VariableMemoryAssigner {
 public:
  explicit VariableMemoryAssigner(ComputeGraphPtr compute_graph) : compute_graph_(std::move(compute_graph)) {}

  VariableMemoryAssigner(const VariableMemoryAssigner &) = delete;

  VariableMemoryAssigner &operator=(const VariableMemoryAssigner &) = delete;

  virtual ~VariableMemoryAssigner() = default;

  /// @ingroup ge_graph
  /// @brief assign memory offset
  /// @return Status result of function
  Status Assign();

  /// @ingroup ge_graph
  /// @brief assign variable attr to nodes
  /// @return Status result of function
  Status AssignVarAttr2Nodes();

  Status AssignMemory2HasRefAttrNode();

 private:
  ComputeGraphPtr compute_graph_;
};

using VariableMemoryAssignerPtr = std::shared_ptr<VariableMemoryAssigner>;
using BlockMemAssignerPtr = std::shared_ptr<BlockMemAssigner>;
using HybridMemAssignerPtr = std::shared_ptr<HybridMemAssigner>;

class GraphMemoryAssigner {
 public:
  explicit GraphMemoryAssigner(ComputeGraphPtr compute_graph)
      : compute_graph_(std::move(compute_graph)), mem_assigner_(nullptr), graph_mem_splitter_(nullptr) {}

  GraphMemoryAssigner(const GraphMemoryAssigner &) = delete;

  GraphMemoryAssigner &operator=(const GraphMemoryAssigner &) = delete;

  virtual ~GraphMemoryAssigner() = default;

  /// @ingroup ge_graph
  /// @brief assign memory offset
  /// @return Status result of function
  Status AssignMemory(const bool has_assigned_var_mem = false);

  /// @ingroup ge_graph
  /// @brief assign variable memory offset
  /// @return Status result of function
  static Status AssignVarMemory(const ComputeGraphPtr &compute_graph);

  /// @ingroup ge_graph
  /// @brief assign variable attr to nodes,
  /// must be called after all memory assigned.
  /// @return Status result of function
  Status AssignVarAttr2Nodes();

  ge::Status AssignMemory2HasRefAttrNode() const;

  ge::Status ReAssignMemory(std::map<uint64_t, size_t> &mem_type_to_offset);

  Status AssignZeroCopyMemory(std::map<uint64_t, size_t> &mem_offset, size_t &zero_mem_copy_size);

  Status ReAssignContinuousMemory();

  Status SetMemReuseInfo() const;

  void RecordSubsequentReuseNodeInfo(const MemoryBlock *const memory_block,
                                     const std::vector<MemReuseInfo> &parent_mem_resue_info,
                                     std::vector<MemReuseInfo> &total_child_mem_resue_info, uint32_t depth = 0U) const;

  Status SetInputOffset() const;

  Status UpdateOpInputOffset(const NodePtr &node) const;
  Status UpdateRefOpOutputOffset(const NodePtr &node, const std::map<int32_t, int32_t> &out2ins, const int32_t ref_in,
                                 const int64_t input_offset) const;
  Status AtomicCleanCheck() const;
  Status ReuseCheck() const;
  Status CheckOffset() const;
  Status UpdateParentNodeOffset() const;
  Status CheckRefNodeOffset(const NodePtr &node) const;

  Status AssignReferenceMemory() const;

  void MarkDistanceAttr();

  const GraphMemSplitterPtr GetGraphMemSplitter() const {
    return graph_mem_splitter_;
  }

  const BlockMemAssignerPtr GetMemAssignerPtr() const {
    if (mem_assigner_ != nullptr) {
      return mem_assigner_->GetPriorityAssinger();
    }
    return nullptr;
  }

  Status GetMemType(const Node *const node, const IOType &io_type, const uint32_t index, uint32_t &mem_type) const;

  Status SetAtomicCleanOffset() const;

 private:
  Status AssignReferenceMemory(const NodePtr &node) const;

  Status OffsetValidCheck() const;

  Status TryGetNodeRefIndexes(const NodePtr &node, std::map<int32_t, int32_t> &out2ins) const;

  bool IsAssignContinuousInputMemoryDirectly(const NodePtr &input_continuous_node,
                                             std::map<NodePtr, uint32_t> &node_2_continuous_type) const;

  Status SetMemOffset(const NodePtr &node, const InDataAnchorPtr &in_data_anchor, bool reverse_refresh,
                      int64_t &mem_offset) const;

  Status AssignContinuousInputMemory(const NodePtr &node, uint32_t continuous_type, bool reverse_refresh = false);

  Status AssignContinuousOutputMemory(const NodePtr &node, int64_t memory_type, uint32_t continuous_type) const;

  Status UpdateSymbolOutputOffset(const NodePtr &node, int64_t output_index, int64_t offset) const;

  Status UpdateOpInputOffset(const NodePtr &node, std::vector<int64_t> &input_list) const;

  Status UpdateConstArgsOffset(const NodePtr &node, std::vector<int64_t> &input_list) const;

  Status UpdateOpInputDescOffset(const NodePtr &node) const;

  NodePtr GetKnownInputNode(const NodePtr &node) const;

  Status GetNodeMemoryType(const NodePtr &node, int64_t &memory_type, std::string input_or_output) const;

  bool CheckContinuousMemType(std::vector<int64_t> mem_type_list) const;

  Status AssignBufferPoolMemory();

  bool IsRefFromInputOpCascade(const NodePtr &node) const;

  Status UpdateRefOpOffsetReverse(const NodePtr &node) const;

  bool IsOutputVisitedByMultiStream(const NodePtr &peer_out_node, int64_t out_anchor_index) const;

  void UpdatePrevNodeInputDesc(const NodePtr &prev_node, const std::vector<int64_t> &prev_node_input_index_vec,
                               int64_t distance) const;

  void UpdateCurNodeInputDesc(const NodePtr &cur_node, int64_t cur_node_input_index, int64_t distance) const;

  void CheckNeedCalcDistAndUpdateVisitInfo(
      const NodePtr &peer_out_node, const OutDataAnchorPtr &peer_out_anchor, size_t matched_mem_offset,
      std::map<size_t, std::pair<NodePtr, std::vector<int64_t>>> &mem_block_visit_info,
      bool &is_need_calc_distance) const;

  void CalcDistanceAndUpdateDesc(const std::map<std::string, int64_t> &node_index_in_stream,
                                 const InDataAnchorPtr &in_data_anchor, size_t matched_mem_offset, const NodePtr &node,
                                 std::map<size_t, std::pair<NodePtr, std::vector<int64_t>>> &mem_block_visit_info,
                                 bool &is_need_skip) const;

  void DeleteVisitInfoWhenLifecycleEnded(
      const NodePtr &node, const InDataAnchorPtr &in_data_anchor, size_t matched_mem_offset,
      std::map<size_t, std::pair<NodePtr, std::vector<int64_t>>> &mem_block_visit_info) const;

  void MarkNodeDistanceAttr(const NodePtr &node,
                            std::map<size_t, std::pair<NodePtr, std::vector<int64_t>>> &mem_block_visit_info,
                            const std::map<std::string, int64_t> &node_index_in_stream);

  MemoryOffsetMap memory_offset_;
  ComputeGraphPtr compute_graph_;
  HybridMemAssignerPtr mem_assigner_;
  GraphMemSplitterPtr graph_mem_splitter_;
  AtomicMemoryAssignerPtr atomic_memory_assigner_;
};
}  // namespace ge

#endif  // GE_GRAPH_BUILD_MEMORY_GRAPH_MEM_ASSIGNER_H_
