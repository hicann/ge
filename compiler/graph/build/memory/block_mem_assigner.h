/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GE_GRAPH_BUILD_MEMORY_BLOCK_MEM_ASSIGNER_H_
#define GE_GRAPH_BUILD_MEMORY_BLOCK_MEM_ASSIGNER_H_

#include <map>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>
#include <list>

#include "rt_external.h"
#include "framework/common/ge_inner_error_codes.h"
#include "framework/common/framework_types_internal.h"
#include "framework/common/util.h"
#include "graph/build/memory/mem_assigner.h"
#include "graph/manager/graph_var_manager.h"
#include "graph/compute_graph.h"
#include "graph/utils/graph_utils.h"
#include "common/sgt_slice_type.h"
#include "graph/build/memory/mem_reuse_strategy.h"
#include "memory_block.h"
#include "block_type_list.h"
#include "continuous_mem.h"

namespace ge {

using StreamIdToBlocks = std::unordered_map<int64_t, std::vector<MemoryBlock *>>;
using MemoryTypeToSubGraphIdBlocks = std::unordered_map<int64_t, StreamIdToBlocks>;

struct ApplyMemoryParam {
  size_t block_size;  // block_size applied memory block size
  size_t real_size;   // real_size actual memory size required
  size_t no_align_size;
  OpMemoryType mem_type;
  uint32_t out_index;          // out_index output node index
  const bool is_op_reuse_mem;  // is_op_reuse_mem whether the op reuses memory
  const bool continuous;       // whether the memory of op is continuous
  uint64_t memory_type;        // device memory type
  bool is_zero_copy;
};

struct MemAssistInfo {
  ComputeGraphPtr compute_graph;
  AnchorToSymbol anchor_to_symbol;
  SymbolToAnchors symbol_to_anchors;
  std::unordered_map<const Node *, std::vector<int64_t>> parent_nodes_to_stream_ids;
};

bool CheckIsZeroMemNodeType(const std::string &node_type);
Status GetNoNeedAssignMemoryFlag(const NodePtr &n, uint32_t out_index, bool &no_need_assign_memory);
uint64_t GetWorkSpaceMemoryType(const size_t no_reuse_scope_size, const size_t index, const bool is_p2p_memory,
                                const bool session_scope_memory, std::vector<bool> &workspace_reuse_flag);

class BlockMemAssigner : public MemAssigner {
 public:
  BlockMemAssigner(const MemAssistInfo &mem_assist_info);

  BlockMemAssigner(const BlockMemAssigner &) = delete;

  BlockMemAssigner &operator=(const BlockMemAssigner &) = delete;

  ~BlockMemAssigner() override;

  Status Assign() override;

  const std::map<uint64_t, size_t> &GetMemOffsets() const {
    return mem_offsets_;
  }

  const std::map<uint64_t, MemoryStat> &GetMemoryStat() const {
    return memory_stat_;
  }

  int64_t GetAtomicAddrCleanId() const {
    return atomic_addr_clean_id_;
  }

  std::vector<MemoryBlock *> GetMemoryBlocks() const {
    return memory_blocks_;
  }

  void SetReuseStrategy(const ReuseStrategy &reuse_strategy) {
    reuse_strategy_ = reuse_strategy;
  }

  bool IsMemoryPriorityMode() const {
    return memory_priority_mode_;
  }

  /// @ingroup domi
  /// @brief   memory size fixed for reuse. get memory range
  /// @param [out] ranges return memory range
  /// @return Status result
  virtual Status GetMemoryRanges(std::vector<int64_t> &ranges) = 0;
  /// @ingroup domi
  /// @brief traverse all nodes' outputs and needed workspace mem, apply memory, consider reuse memory
  /// @param [in] ranges memory range provided
  /// @author
  Status AssignMemoryWithReuse(std::vector<int64_t> &ranges);

  std::string GetMaxBatchLabel() const {
    return max_batch_label_;
  }

  /// PreAssign and SetOpMemOffset are not thread safe, can only be called from a single thread
  /// Other function must ensure thread safety, there will be multiple thread calls
  static Status PreparationForAssign(MemAssistInfo &mem_assist_info);
  static Status SetRealStreamIdForParentNode(MemAssistInfo &mem_assist_info);

  void SetOpMemOffset(bool is_zero_copy) const;
  void SetOpMemOffset(const std::vector<MemoryBlock *> &zero_copy_blocks) const;
  void SetOffsetForContinuousMem() const;
  bool HasSameOutAnchorWithDiffStream(const Node *n, const uint32_t index) const;

 protected:
  /// @ingroup domi
  /// @brief traverse all memory size, resize, and calculate offset
  /// @param [in&out] memory_blocks memory size, resize and calculate memory address after offset
  Status ResizeMemoryBlocks();

  Status GetOutAndWorkSpaceMem(std::vector<int64_t> &all_memory_size);

  void GetNodeWorkSpaceSize(const ge::NodePtr &node, std::vector<int64_t> &workspace_memory, int64_t &total_size) const;
  /// @ingroup GE
  /// @brief check if input node reuse memory
  /// @param [in] n input node
  /// @return bool
  bool GetInputNodeReuseMemFlag(const NodePtr &n) const;

  /// @ingroup GE
  /// @brief check if a input of net output node reuse memory
  /// @param [in] index input index of netoutput node
  /// @return bool
  bool GetOutputNodeReuseMemFlagByIndex(const int32_t index) const;

  /// @ingroup GE
  /// @brief Check pre_reuse flag & post_reuse glag for each symbol
  /// @return void
  void InitReuseFlag();

  /// @ingroup GE
  /// @brief get pre_reuse flag
  /// @param [in] cur_node_index_io
  /// @param [out] symbol
  /// @return bool
  bool IsPreReuse(const NodeIndexIO &cur_node_index_io, std::string &symbol) const;

  /// @ingroup GE
  /// @brief get post_reuse flag
  /// @param [in] symbol
  /// @param [out] diff_stream_prior
  /// @return bool
  bool IsPostReuse(const std::string &symbol, bool &diff_stream_prior) const;

  /// @ingroup GE
  /// @brief get post_reuse flag
  /// @param [in] mem_block
  /// @return bool
  bool IsPostReuse(const ge::MemoryBlock *const mem_block) const;

  /// @ingroup GE
  /// @brief check if symbol of cur node_index_io has block
  /// @param [in] node_index_io
  /// @param [out] symbol
  /// @return bool
  bool IsSymbolExist(const NodeIndexIO &node_index_io, std::string &symbol, MemoryBlock *&block) const;

  /// @ingroup GE
  /// @brief check if symbol of cur node_index_io has output description block
  /// @param [in] node_index_io
  /// @param [out] symbol
  /// @return bool
  bool IsSymbolDescBlockExist(const NodeIndexIO &node_index_io, std::string &symbol, MemoryBlock *&block) const;

  /// @ingroup GE
  /// @brief Print symbol
  /// @return void
  void PrintSymbolMap();

 public:
  /// @ingroup GE
  /// @brief Get the memory type corresponding to the current symbol.
  /// @param [in] node_index_io_list
  /// @param [out] memory_type
  /// @return void
  static void GetSymbolMemType(const std::list<NodeIndexIO> &node_index_io_list, int64_t &memory_type);

  /// @ingroup GE
  /// @brief add the memory type with symbol.
  /// @param [in] symbol
  /// @param [in] memory_type
  /// @return void
  void AddSymbolMemType(const std::string &symbol, int64_t memory_type);

  /// @ingroup GE
  /// @brief Update input tensor or output tensor of op to new memory type attr.
  /// @param [in] node_index_io_list
  /// @param [in] memory_type
  /// @return void
  void UpdateOpTensorMemType(const std::list<NodeIndexIO> &node_index_io_list, int64_t memory_type);

  /// @ingroup GE
  /// @brief Print memory block info
  /// @return void
  void PrintMemBlock();

  /// @ingroup GE
  /// @brief Nano Determine whether it is the type of zero memory output node.
  /// @param [in] node type.
  /// @return bool true: is zero memory node; false: is not zero memory node
  /// @author
  bool CheckIsZeroMemNodeOutputIndex(const NodePtr &n, uint32_t index) const;

  virtual bool NeedLevel2Reuse() {
    return true;
  };

  Status GetRealStreamIdForParentNode(const NodePtr &node, const uint32_t out_index, int64_t &stream_id,
                                      bool &is_reuse) const;

  std::map<uint64_t, size_t> mem_offsets_;
  ge::ComputeGraphPtr compute_graph_;
  std::vector<MemoryBlock *> memory_blocks_;
  std::vector<MemoryBlock *> blocks_store_;
  std::vector<NodeTypeIndex> zero_memory_list_;

  // ref mapping
  const SymbolToAnchors &symbol_to_anchors_;
  const AnchorToSymbol &anchor_to_symbol_;
  std::map<std::string, MemoryReuseInfo> symbol_mem_reuse_info_;

 private:
  Status GetOutputTotalSizeAndOutCount(const NodePtr &n, uint32_t output_index, size_t &max_size, size_t &no_align_size,
                                       int32_t &out_count, bool is_separate_clean_continuous_inputs) const;
  /// @ingroup GE
  /// @brief Traversing the compute_graph_ to apply for output memory while considering reuse
  /// @param [in] n: node in compute_graph_
  /// @param [in] index: output node index
  /// @param [in] ranges: available memory specifications
  /// @param [in] is_op_reuse_mem: Whether the op reuses the memory, true: reuse; false: not reuse
  /// @param [in] out_node_need_continuous_input: Whether the downstream node's need continuous input memory
  /// @return MemoryBlock*
  /// @author
  MemoryBlock *ApplyOutMemory(const ge::NodePtr &n, uint32_t index, const std::vector<int64_t> &ranges,
                              const bool is_op_reuse_mem, const bool out_node_need_continuous_input);

  Status AssignOutputMemoryWithReuse(const NodePtr &node, std::vector<int64_t> &ranges);

  Status AssignWorkSpaceMemoryWithReuse(const NodePtr &node, std::vector<int64_t> &ranges);

  /// @ingroup GE
  /// @brief Traversing the compute_graph_ to apply for output description memory
  /// @param [in] n: node in compute_graph_
  /// @param [in] index: output node index
  /// @param [in] ranges: available memory specifications
  /// @return MemoryBlock*
  /// @author
  MemoryBlock *ApplyOutDescMemory(const NodePtr &n, uint32_t index, const std::vector<int64_t> &ranges);

  /// @ingroup GE
  /// @brief Traversing the compute_graph_ to apply for memory while considering reuse
  /// @param [in] n node in compute_graph_
  /// @param [in] workspace_reuse_flag reuse flag for workspace
  /// @param [in] ApplyMemoryParam apply memory param
  /// @return MemoryBlock*
  /// @author
  MemoryBlock *ApplyMemory(const NodePtr &n, const std::vector<bool> &workspace_reuse_flag,
                           const ApplyMemoryParam &param);
  bool IsNodeOutputUseSameMemWithNetOutput(const ge::NodePtr &node, uint32_t out_index) const;
  /// @ingroup GE
  /// @brief Get the block: release first, reuse first
  /// @param [in] block_size applied memory block size
  /// @param [in] batch_label batch label
  /// @param [in] reusable_blocks all reusable blocks
  /// @param [in] node_type_index node to assign memory
  /// @return MemoryBlock*
  /// @author
  MemoryBlock *GetFirstReleaseBlock(const size_t block_size, const std::string &batch_label,
                                    std::vector<MemoryBlock *> &reusable_blocks,
                                    const NodeTypeIndex &node_type_index) const;

  /// @ingroup GE
  /// @brief Get the block: release first, reuse first
  /// @param [in] block_size applied memory block size
  /// @param [in] batch_label batch label
  /// @param [in] reusable_blocks all reusable blocks
  /// @param [in] node_type_index node to assign memory
  /// @return MemoryBlock*
  /// @return MemoryBlock*
  /// @author
  MemoryBlock *GetLastReleaseBlock(const size_t block_size, const std::string &batch_label,
                                   std::vector<MemoryBlock *> &reusable_blocks,
                                   const NodeTypeIndex &node_type_index) const;
  /// @ingroup GE
  /// @brief check workspace_reuse_flag to judge if add workspace block wait reuse
  /// @param [in] workspace_reuse_flag mark out index if support resue
  /// @param [in] index out index
  /// @param [in] stream_id which stream op in
  /// @param [in] mem_block node workspace mem_block
  /// @param [in] memory_type workspace memory type
  /// @return void
  /// @author
  void CheckWorkspaceReuse(const std::vector<bool> &workspace_reuse_flag, uint32_t index, int64_t stream_id,
                           MemoryBlock *const mem_block, uint64_t memory_type);

  /// @ingroup GE
  /// @brief Release memory block to reusable list
  /// @param [in] to_release memory block to be released
  /// @param [in] reusable_memory reusable list
  /// @return void
  /// @author
  void ReleaseMemory(MemoryBlock *const to_release, std::vector<MemoryBlock *> &reusable_memory, int64_t stream_id,
                     const std::string &symbol, bool no_release = false);

  /// @ingroup GE
  /// @brief Release memory blocks to reusable list
  /// @param [in] to_releases memory blocks to be released
  /// @param [in] reusable_memory reusable list
  /// @return void
  /// @author
  void ReleaseMemorys(StreamIdToBlocks &to_releases, StreamIdToBlocks &reusable_memory);

  /// @ingroup GE
  /// @brief Release memory block to reusable list
  /// @param [in] n node in compute_graph_
  /// @param [in] node_out_blocks output memory blocks for ops
  /// @param [in] reusable_memory reusable list
  /// @return void
  /// @author
  void ReleaseInputNodeOutMemory(const NodePtr &node);

  void AssignContinuousBlocks();
  bool IsZeroCopyBlock(const NodePtr &node, uint32_t output_index, bool continuous, size_t output_size = 0) const;
  bool IsAtomicOutputMemory(const ge::NodePtr &node, uint32_t output_index, bool is_atomic,
                            bool out_node_set_continuous_input) const;
  bool IsOutNodeSetContinuousInput(const NodePtr &n, uint32_t out_index, InDataAnchor *&continuous_in_anchor,
                                   bool &is_reuse_zero_copy, std::set<int64_t> &streams);
  bool IsContinuousMemoryReuse(const Node *const n, uint32_t out_index, const Node *const continuous_node,
                               std::set<int64_t> &streams);

  Status CalNodeAsContinuousInputMaxLife(const Node *const n, uint32_t out_index, const Node *const continuous_node,
                                         int64_t &first_node_max_life, std::set<int64_t> &streams);

  void CalExitSymbolNodeLifeTime(const Node *const n, uint32_t out_index, size_t &max_life_time);
  /// @ingroup GE
  /// @|+++++++++block1++++++++|                               |+++++++++block1++++++++|
  /// @|+++++++++block1++++++++||++block2++|                   |+++++++++block1++++++++||++block2++|
  /// @                         |++block2++||++block3++|  ==>  |++block3++|             |++block2++|
  /// @                                     |++block3++|       |++block3++|
  /// @return void
  /// @author
  void ReuseBlocksByLifeTime();

  void ContinuousOutRefCheck(bool &is_all_output_ref, bool &is_output_has_ref, const NodePtr &n);

  Status ApplyContinuousMemory(const NodePtr &n, const std::vector<int64_t> &ranges, const bool is_op_reuse_mem);
  Status ApplyContinuousMemWithMng(const NodePtr &n, int32_t idx, const std::vector<int64_t> &ranges);
  Status GetContinuousMemType(const ContinuousMem &continuous_mem, uint64_t &memory_type) const;
  void CheckAndReleaseSuspendedBlock(const NodePtr &node, uint32_t idx, MemoryBlock *block);

  int32_t GetAllRefCount(const NodeIndexIO &out_node_index_io, bool &is_reuse_zero_copy) const;
  Status GetContinuousMemLifeTime(const ContinuousMem &continuous_mem, int64_t &begin_time, int64_t &out_time,
                                  int64_t &end_time, size_t &out_streams_cnt) const;
  static void OptimizeStreamIdForMemoryReuse(const NodePtr &node);
  static void SetRealStreamIdForDataNode(const Node *const node);

  void GetDiffStreamEdgeLife(const NodePtr &node, const std::set<int64_t> &exclude_merge_streams);
  void AddInStreamEdge(const ge::OpDesc *const node_desc, const ge::OpDesc *const in_node_desc);
  void InsertStreamOutEdge();
  void InsertStreamInEdge(std::set<EdgeLife, CompareEdgeLife> &in_edge_set, const EdgeLife &new_in_edge,
                          const int64_t src_stream_id, const int64_t dst_stream_id,
                          const std::pair<const char *, const char *> &node_names = {nullptr, nullptr});
  /// @ingroup GE
  /// @brief Cascade memory scenarios to obtain the actual life time begin of continuous input memory
  /// @return void
  /// @author
  void GetContinuousNodeLifeTimeBegin(const Node *const node, const int32_t in_index);

  void SetContinuousNodeLifeTimeBegin(const Node *const node);

  void GetRefContinuousInputNodeAndFixedAddrPriorFlag(const std::string &symbol, const std::list<NodeIndexIO> &anchors);

  bool IsNoNeedAssignMemory(const NodePtr &n, const NodeIndexIO &out_node_index_io, const uint32_t index) const;

  void SetOffsetSize(const NodeTypeIndex &node_type, const MemoryBlock &block, size_t real_size, size_t no_align_size,
                     int32_t child_block_level) const;

  void SetBlockOpMemOffset(const MemoryBlock *const block, int32_t child_block_level, bool &is_fixed_addr_prior) const;

  void ParseGraphIoAllocMode();
  void ParseIoReuseMemOption();
  Status InitIoReuseFlag();
  void AddMemoryStat(uint64_t memory_type, size_t real_size, bool is_reuse_memory);

  void InitDiffStreamSameOutTable();
  // [memory type][sub graph id][stream id]
  MemoryTypeToSubGraphIdBlocks reusable_blocks_;

  MemoryTypeToSubGraphIdBlocks stream_workspace_blocks_;

  std::unordered_map<std::string, MemoryBlock *> symbol_blocks_;

  std::unordered_map<std::string, MemoryBlock *> symbol_desc_blocks_;

  // 用于给带Padding连续输入节点分配连续的内存，记录所有输入的block
  // <int64_t, <input_index, blockPtr>>
  std::unordered_map<int64_t, std::unordered_map<uint32_t, MemoryBlock *>> node_continuous_input_blocks_;

  // 记录带Padding连续输入节点
  std::unordered_map<int64_t, std::pair<std::string, uint32_t>> node_continuous_input_counts_;

  std::unordered_map<std::string, size_t> cascade_min_life_time_;

  // reuse memory
  std::unordered_set<std::string> op_no_reuse_mem_set_;  // names and types of Op which is not reuse memory

  bool op_reuse_env_valid_ = false;  // init flag for op_no_reuse_mem_vec_

  bool is_ge_reuse_mem_ = true;  // global, controlled by ge option ge.exec.disableReuseMemory

  bool is_op_reuse_mem_ = true;  // op-level, changed and shared in the process of an op

  bool is_separate_clean_continuous_inputs_ = false;  // op-output-level, changed and shared in the process of an output

  size_t life_time_;

  int64_t atomic_addr_clean_id_ = 0;

  std::map<uint64_t, MemoryStat> memory_stat_;  // key: device memory type

  std::string max_batch_label_;

  size_t life_begin_ = 0U;

  size_t life_end_ = 0U;

  bool root_unknown_shape_flag_ = false;

  // [sub graph id] streamid, out streamid, nodeid, outnodeid
  DiffStreamEdgeLife out_stream_edges_;

  // [sub graph id] streamid, in streamid, nodeid, innodeid
  DiffStreamEdgeLife in_stream_edges_;

  bool memory_priority_mode_ = false;

  bool is_io_alloc_by_ge_in_run_graph_ = false;

  bool is_feature_map_refreshable_ = false;

  uint64_t input_fusion_size_ = 0U;

  ReuseStrategy reuse_strategy_{};

  // Saved and finally modified by a single thread
  std::vector<TAttr<bool>> bool_attr_;
  std::vector<TAttr<int64_t>> int_attr_;
  std::vector<bool> input_index_to_reuse_mem_flag_;
  std::vector<bool> output_index_to_reuse_mem_flag_;

  std::unordered_map<OutDataAnchor *, std::list<OutDataAnchor *> *> same_out_group_;
  std::list<std::list<OutDataAnchor *>> same_out_group_holder_;
  bool is_static_model_addr_fixed_ = false;
  const std::unordered_map<const Node *, std::vector<int64_t>> &parent_nodes_to_stream_ids_;
  ContinuousMemMng continuous_mem_mng_;
};
using BlockMemAssignerPtr = std::shared_ptr<BlockMemAssigner>;
}  // namespace ge
#endif  // GE_GRAPH_BUILD_MEMORY_BLOCK_MEM_ASSIGNER_H_
