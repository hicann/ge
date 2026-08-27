/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GE_GRAPH_BUILD_MEMORY_MEMORY_BLOCK_H_
#define GE_GRAPH_BUILD_MEMORY_MEMORY_BLOCK_H_

#include <map>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>
#include <list>
#include <set>

#include "rt_external.h"
#include "framework/common/ge_inner_error_codes.h"
#include "framework/common/framework_types_internal.h"
#include "framework/common/util.h"
#include "graph/manager/graph_var_manager.h"
#include "graph/compute_graph.h"
#include "graph/utils/graph_utils.h"
#include "common/sgt_slice_type.h"
#include "graph/build/memory/mem_reuse_strategy.h"
#include "block_type_list.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/utils/attr_utils.h"

namespace ge {

inline const std::string &GetBatchLabel(const ge::OpDesc *op_desc) {
  const auto *ptr = ge::AttrUtils::GetStr(op_desc, ATTR_NAME_BATCH_LABEL);
  static const std::string empty;
  return (ptr != nullptr) ? *ptr : empty;
}

constexpr size_t kMaxLifeTime = 0xffffffffUL;
constexpr size_t kMinLifeTime = 1U;
const size_t kDefaultLifeTime = 0xffffffffUL - 1;
const uint64_t kSessionScopeMemory = 0x100000000UL;
const int64_t kOutputMemoryGlobalType = 2;
constexpr size_t kMaxLogLen = 512UL;
constexpr uint32_t kMaxDepthNum = 100U;
constexpr int64_t kParentNodeDefaultStreamId = -2;

enum class MemoryNoReuseScope { kReuse, kSessionNoReuse, kGraphNoReuse };

using DependStreamLife = std::map<int32_t, std::map<int64_t, std::map<int64_t, size_t>>>;

struct EdgeLife {
  size_t node_id;
  size_t peer_node_id;
};

struct CompareEdgeLife {
  bool operator()(const ge::EdgeLife &left, const ge::EdgeLife &right) const {
    return left.node_id < right.node_id;
  }
};

using DiffStreamEdgeLife = std::map<int64_t, std::map<int64_t, std::set<EdgeLife, CompareEdgeLife>>>;

enum class OpMemoryType { kOutput, kWorkspace, kOutputDesc, kInput };

struct ReuseStrategy {
  explicit ReuseStrategy(bool use_range = false, bool ascending_sort = true, bool reuse_first_release = false,
                         bool memory_priority_mode = false)
      : use_range_(use_range),
        ascending_sort_(ascending_sort),
        reuse_first_release_(reuse_first_release),
        memory_priority_mode_(memory_priority_mode) {}
  bool use_range_ = false;
  bool ascending_sort_ = true;
  bool reuse_first_release_ = false;
  bool memory_priority_mode_ = false;
};

struct MemoryStat {
  size_t theory_min_memory_size_ = 0;
  size_t theory_memory_size_ = 0;
  size_t theory_no_reuse_memory_size_ = 0;
  size_t total_memory_size_ = 0;
  size_t stream_count_ = 0;
};

struct MemoryReuseInfo {
  size_t size_ = 0U;
  uint64_t mem_type_ = RT_MEMORY_HBM;
  bool pre_reuse_flag_ = true;
  bool post_reuse_flag_ = true;
  bool no_assign_mem_ = false;
  bool is_fixed_addr_prior_ = false;
  bool diff_stream_prior_ = false;
};

template <class T>
struct TAttr {
  TAttr(ge::AttrHolder *const ptr, const ge::OpDesc *const desc, int32_t index, const std::string &name, const T value)
      : ptr_(ptr), desc_(desc), name_(name), index_(index), value_(value) {}
  ge::AttrHolder *ptr_;
  const ge::OpDesc *desc_;
  const std::string &name_;
  const int32_t index_;
  const T value_;
};

struct NodeTypeIndex {
  NodeTypeIndex(const ge::Node *node, OpMemoryType mem_type, uint32_t index, bool ref_input = false, size_t begin = 0,
                int64_t stream_id = kInvalidStreamId, bool is_subgraph_workspace = false,
                size_t symbol_end_time = kDefaultLifeTime)
      : node_(node),
        mem_type_(mem_type),
        index_(index),
        ref_input_(ref_input),
        is_subgraph_workspace_(is_subgraph_workspace),
        life_time_begin_(begin),
        symbol_max_life_time_end_(symbol_end_time),
        stream_id_(stream_id) {
    if ((node_ != nullptr) && (node_->GetOpDesc() != nullptr)) {
      is_subgraph_out_ = (node_->GetOpDesc()->GetType() == ge::PARTITIONEDCALL) && (mem_type_ == OpMemoryType::kOutput);
      node_id_ = static_cast<size_t>(node_->GetOpDesc()->GetId());
      life_time_begin_is_min_ = (life_time_begin_ > 0) && (life_time_begin_ < node_id_);
    } else {
      node_id_ = life_time_begin_;
    }
  }

  const std::string GetMemType() const {
    return GetMemType(mem_type_);
  }
  static std::string GetMemType(const OpMemoryType &mem_type) {
    switch (mem_type) {
      case OpMemoryType::kOutput:
        return "output";
      case OpMemoryType::kWorkspace:
        return "workspace";
      case OpMemoryType::kOutputDesc:
        return "output_desc";
      case OpMemoryType::kInput:
        return "input";
      default:
        return "unknown";
    }
  }

  size_t GetLifeBegin(bool for_sort = false) const {
    if (life_time_begin_is_min_) {
      return life_time_begin_;
    } else {
      if (!for_sort && is_subgraph_out_) {
        return life_time_end_;
      } else {
        return node_id_;
      }
    }
  }

  std::string GetLifeBeginDesc() const {
    if (node_ == nullptr) {
      return std::to_string(life_time_begin_);
    }

    auto life_begin = GetLifeBegin();
    if (life_begin != node_id_) {
      return std::to_string(life_begin) + "--" + std::to_string(node_id_);
    }
    return std::to_string(life_begin);
  }

  std::vector<size_t> GetLifeEnd() const {
    std::vector<size_t> life_end_list;
    bool single_end = true;
    for (auto pair : diff_stream_life_time_) {
      if (pair.second != life_time_end_) {
        life_end_list.emplace_back(pair.second);
        single_end = false;
      }
    }
    life_end_list.emplace_back(life_time_end_);
    if (single_end && (ref_life_time_end_ != kDefaultLifeTime) && (ref_life_time_end_ != life_time_end_)) {
      life_end_list.emplace_back(ref_life_time_end_);
    }
    return life_end_list;
  }

  std::string GetLifeEndDesc() const {
    std::string end_desc;
    const auto life_end_list = GetLifeEnd();
    for (const auto life_end : life_end_list) {
      if (!end_desc.empty()) {
        end_desc.append("--");
      }
      end_desc.append(std::to_string(life_end));
    }
    return end_desc;
  }

  void SetOutStreamCount(size_t count) {
    if (count > out_stream_count_) {
      out_stream_count_ = count;
    }
  }
  void SetFirstContinuousNode(const bool flag) {
    first_continuous_node_ = flag;
  }
  void SetLastContinuousNode(const bool flag) {
    last_continuous_node_ = flag;
  }
  void SetContinuousNode(const bool flag) {
    continuous_node_ = flag;
  }
  bool GetFirstContinuousNodeFlag() const {
    return first_continuous_node_;
  }
  bool GetLastContinuousNodeFlag() const {
    return last_continuous_node_;
  }
  bool GetContinuousNodeFlag() const {
    return continuous_node_;
  }

  const ge::Node *node_ = nullptr;
  OpMemoryType mem_type_ = OpMemoryType::kOutput;
  uint32_t index_ = 0;
  bool ref_input_ = false;
  bool is_subgraph_workspace_ = false;
  bool is_subgraph_out_ = false;
  bool life_time_begin_is_min_ = false;
  bool next_is_ref_input_ = false;
  size_t life_time_begin_ = 0;
  size_t life_time_end_ = kDefaultLifeTime;
  size_t symbol_max_life_time_end_ = kDefaultLifeTime;
  int64_t stream_id_ = kInvalidStreamId;
  size_t node_id_ = 0U;
  std::map<int64_t, size_t> diff_stream_life_time_;
  std::map<int64_t, std::pair<size_t, size_t>> out_stream_life_time_;
  size_t ref_life_time_end_ = kDefaultLifeTime;
  size_t out_stream_count_ = 1U;
  size_t no_align_size_ = 0U;
  bool continuous_node_ = false;
  bool first_continuous_node_ = false;
  bool last_continuous_node_ = false;
};

class BlockMemAssigner;
class MemoryBlock {
 public:
  explicit MemoryBlock(const ReuseStrategy &reuse_strategy, size_t block_size, int64_t stream_id = 0,
                       bool reuse_mem = true, uint64_t memory_type = RT_MEMORY_HBM)
      : ref_count_(0),
        stream_id_(stream_id),
        child_block_(false),
        reuse_mem_(reuse_mem),
        same_stream_(true),
        has_sub_graph_in_out_node_(false),
        input_index_(0),
        continuous_block_(false),
        first_continuous_block_(false),
        last_continuous_block_(false),
        is_zero_copy_(false),
        is_reuse_zero_copy_(true),
        memory_type_(memory_type),
        memory_type_logic_base_(0),
        need_same_offset_in_batch_(false),
        max_real_size_(0),
        max_block_size_(0),
        window_size_(1U),
        thread_dim_(1U),
        post_reuse_flag_(true),
        is_fixed_addr_prior_(false),
        diff_stream_prior_(false),
        used_by_diff_streams_(false),
        block_size_(block_size),
        head_offset_(0U),
        tail_offset_(0U),
        child_offset_(0U),
        batch_used_size_(0U),
        reuse_strategy_(reuse_strategy) {
    switch (memory_type_) {
      case RT_MEMORY_HOST:
        memory_type_logic_base_ = kMemoryHostFeatureMapLogicBase;
        break;
      case RT_MEMORY_HOST_SVM:
        memory_type_logic_base_ = kMemoryHostSVMFeatureMapLogicBase;
        break;
      default:
        break;
    }
  }

  MemoryBlock(const MemoryBlock &) = delete;

  MemoryBlock &operator=(const MemoryBlock &) = delete;

  ~MemoryBlock() {
    node_type_index_list_.clear();
    symbol_list_.clear();
  }

  size_t Size() const {
    return block_size_;
  }

  void SetSize(size_t size) {
    if (size > block_size_) {
      block_size_ = size;
    }
  }

  size_t AlignSize();

  Status SetHeadOffset(size_t offset);

  void SetTailOffset(size_t offset);

  size_t HeadOffset() const {
    return head_offset_;
  }

  size_t TailOffset() const {
    return tail_offset_;
  }

  void AddNodeTypeIndex(const NodeTypeIndex &node_type_index, size_t real_size, size_t no_align_size,
                        int64_t stream_id) {
    if (node_type_index.ref_input_) {
      if (!node_type_index_list_.empty()) {
        node_type_index_list_.back().next_is_ref_input_ = true;
      }
    }
    node_type_index_list_.emplace_back(node_type_index);
    node_type_index_list_.back().no_align_size_ = no_align_size;
    real_size_list_.emplace_back(real_size);
    no_align_size_list_.emplace_back(no_align_size);

    if (stream_id != stream_id_) {
      same_stream_ = false;
    }

    // need recompute max real size
    max_real_size_ = 0;
    last_continuous_block_ = last_continuous_block_ || node_type_index.GetLastContinuousNodeFlag();
    first_continuous_block_ = first_continuous_block_ || node_type_index.GetFirstContinuousNodeFlag();
    continuous_block_ = continuous_block_ || node_type_index.GetContinuousNodeFlag();
    block_type_list_.WithAdded(node_type_index);
  }

  bool IsBlockTypeConflict(const MemoryBlock &other) const {
    return block_type_list_.IsConflictWithBlock(other.block_type_list_);
  }

  bool IsBlockTypeConflictWithNode(const NodeTypeIndex &node_type_index) const {
    return block_type_list_.IsConflictWithOneNode(node_type_index);
  }

  std::string BlockTypeStr() const {
    return block_type_list_.ToString();
  }

  void AddSymbol(const std::string &symbol) {
    symbol_list_.emplace_back(symbol);
  }

  void ClearOutStreamLifeInfo() {
    node_type_index_list_.back().out_stream_life_time_.clear();
  }

  void ClearDiffStreamLifeInfo() {
    node_type_index_list_.back().diff_stream_life_time_.clear();
  }

  const std::vector<NodeTypeIndex> &NodeTypeIndexList() const {
    return node_type_index_list_;
  }
  const std::vector<std::string> &SymbolList() const {
    return symbol_list_;
  }
  const std::vector<size_t> &RealSizeList() const {
    return real_size_list_;
  }
  const std::vector<MemoryBlock *> &ChildBlockList() const {
    return child_blocks_;
  }
  const std::map<std::string, std::vector<MemoryBlock *>> &BatchBlockList() const {
    return batch_to_blocks_;
  }
  const std::vector<size_t> &NoAlignSizeList() const {
    return no_align_size_list_;
  }
  const std::vector<MemoryBlock *> &ChildSubGraphBlockList() const {
    return sub_graph_blocks_;
  }
  bool IsNoAlignSizeReuseBlock() const {
    return continuous_block_;
  }
  bool IsRealSizeReuseBlock() const {
    return is_zero_copy_;
  }
  std::vector<MemoryBlock *> AllChildBlockList() const;

  inline void SetRefLifeTimeEnd() {
    for (size_t index = 0U; index < node_type_index_list_.size(); ++index) {
      auto &node_type_index = node_type_index_list_[index];
      if (!node_type_index.next_is_ref_input_ || node_type_index.ref_input_) {
        continue;
      }
      size_t ref_end_index = index;
      for (size_t i = index + 1U; i < node_type_index_list_.size(); ++i) {
        if (!node_type_index_list_[i].ref_input_) {
          break;
        }
        ref_end_index = i;
      }
      node_type_index.ref_life_time_end_ = node_type_index_list_[ref_end_index].life_time_end_;
    }
  }

  void Resize();

  std::string String() const;

  bool IsSameBatchLabel() const;

  // if the block is used by graph input, if true, return input size
  bool IsGraphInputAndGetSize(const ComputeGraphPtr &computeGraph, size_t &size) const;

  void AddContinuousLifeReuseBlock(MemoryBlock &block);

  void AddZeroCopyLifeReuseBlock(MemoryBlock &block);

  bool AddLifeReuseBlock(const BlockMemAssigner *const mem_assigner, MemoryBlock *block,
                         std::vector<MemoryBlock *> &clone_blocks, uint32_t depth,
                         DiffStreamEdgeLife &diff_stream_edge_life, bool child_reuse = false);

  void SetLifeTimeEnd(size_t time, int64_t stream_id);

  void SetOutStreamLifeTime(size_t out_time, size_t end_time, int64_t stream_id);

  size_t GetLifeBegin(bool for_sort = false) const;

  size_t GetLifeEnd(int64_t stream_id) const;

  size_t GetLifeEnd(int64_t stream_id, int64_t &end_stream_id) const;

  size_t GetSymbolLifeEnd() const;

  void SetSymbolLifeEnd(size_t symbol_life_end);

  size_t GetDependLifeBegin(int64_t stream_id, DiffStreamEdgeLife &diff_stream_edge_life) const;

  bool CrossLifeTimeNode(const std::vector<NodeTypeIndex>::const_iterator &it, const MemoryBlock &child_block) const;

  MemoryBlock *Clone() const;

  std::vector<NodeTypeIndex>::const_iterator DelNode(std::vector<NodeTypeIndex>::const_iterator &it);

  void Swap(MemoryBlock &block);

  bool AddChildBlock(MemoryBlock *block) {
    block->child_block_ = true;
    sub_graph_blocks_.emplace_back(block);
    return true;
  }

  bool AddBatchChildBlock(MemoryBlock *block) {
    if ((batch_used_size_ <= Size()) && (block->Size() <= (Size() - batch_used_size_))) {
      block->child_block_ = true;
      batch_used_size_ += block->Size();
      batch_to_blocks_[block->batch_label_].emplace_back(block);
      return true;
    }
    return false;
  }

  void Reset() {
    batch_used_size_ = 0U;
  }

  void SetOutStreamCount(size_t end_stream_count) {
    if (!node_type_index_list_.empty()) {
      node_type_index_list_.back().SetOutStreamCount(end_stream_count);
    }
  }
  void UpdateContinuousFlag();
  void SetFirstContinuousBlock() {
    first_continuous_block_ = true;
    if (!node_type_index_list_.empty()) {
      node_type_index_list_.back().SetFirstContinuousNode(true);
    }
  }
  void SetLastContinuousBlock() {
    last_continuous_block_ = true;
    if (!node_type_index_list_.empty()) {
      node_type_index_list_.back().SetLastContinuousNode(true);
    }
  }
  void SetContinuousBlock() {
    continuous_block_ = true;
    if (!node_type_index_list_.empty()) {
      node_type_index_list_.back().SetContinuousNode(true);
    }
  }
  bool GetFirstContinuousFlag() const {
    return first_continuous_block_;
  }
  bool GetLastContinuousFlag() const {
    return last_continuous_block_;
  }
  bool GetContinuousFlag() const {
    return continuous_block_;
  }

  const ReuseStrategy &GetReuseStrategy() const {
    return reuse_strategy_;
  }

  int32_t ref_count_;
  int64_t stream_id_;
  bool child_block_;
  bool reuse_mem_;
  bool same_stream_;
  bool has_sub_graph_in_out_node_;
  uint32_t input_index_;
  bool continuous_block_;
  bool first_continuous_block_;
  bool last_continuous_block_;
  bool is_zero_copy_;
  bool is_reuse_zero_copy_;
  uint64_t memory_type_;
  int64_t memory_type_logic_base_;
  std::string batch_label_;
  bool need_same_offset_in_batch_;
  size_t max_real_size_;
  size_t max_block_size_;
  uint32_t window_size_;
  uint32_t thread_dim_;
  bool post_reuse_flag_;
  bool is_fixed_addr_prior_;
  bool diff_stream_prior_;
  bool used_by_diff_streams_;

 private:
  size_t block_size_;
  std::vector<size_t> real_size_list_;
  std::vector<size_t> no_align_size_list_;
  size_t head_offset_;
  size_t tail_offset_;
  size_t child_offset_;
  size_t batch_used_size_;
  std::vector<NodeTypeIndex> node_type_index_list_;
  std::vector<std::string> symbol_list_;
  std::vector<MemoryBlock *> child_blocks_;
  std::vector<MemoryBlock *> sub_graph_blocks_;
  std::map<std::string, std::vector<MemoryBlock *>> batch_to_blocks_;
  const ReuseStrategy &reuse_strategy_;
  BlockTypeList block_type_list_;
};

bool CanNotLifeReuse(const ge::MemoryBlock &block, bool child_reuse = false);
bool CanReuseBlock(size_t life_begin, const ge::MemoryBlock &reusable_block, size_t block_size);
bool ReuseBlock(ge::MemoryBlock &block, const size_t block_size, const size_t life_begin,
                const std::string &batch_label, const ge::NodeTypeIndex &node_type_index);
bool CrossLifeTime(const NodeTypeIndex &left, const NodeTypeIndex &right);
bool CanBlockLifeReuse(const BlockMemAssigner *const mem_assigner, const MemoryBlock &in_block,
                       const MemoryBlock &out_block, DiffStreamEdgeLife &diff_stream_edge_life);
Status SetChildHeadOffset(size_t offset, size_t max_offset, std::vector<MemoryBlock *> &blocks);
void SetChildTailOffset(size_t offset, std::vector<MemoryBlock *> &blocks);
std::string ToString(const ge::NodeTypeIndex &x);
std::string GetName(const ge::MemoryBlock &block, bool last_node = false);
bool CompareBlockIndex(const MemoryBlock *const left, const MemoryBlock *const right);
struct CompareLifeInterval {
  explicit CompareLifeInterval(const ReuseStrategy &reuse_strategy) : reuse_strategy_(reuse_strategy) {}
  bool operator()(MemoryBlock *const left, MemoryBlock *const right) const;
  ReuseStrategy reuse_strategy_;
};
Status AddBlockMemOffset(std::map<uint64_t, size_t> &mem_offsets, MemoryBlock &block);

}  // namespace ge

#endif  // GE_GRAPH_BUILD_MEMORY_MEMORY_BLOCK_H_
