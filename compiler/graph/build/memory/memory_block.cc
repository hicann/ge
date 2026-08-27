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
#include <sstream>
#include "common/checker.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/optimize/mem_layout_conflict_optimize/mem_layout_conflict_util.h"
#include "common/ge_common/ge_types.h"
#include "memory_block.h"

namespace ge {

std::string ToString(const ge::NodeTypeIndex &x) {
  std::stringstream ss;
  if (x.node_ != nullptr) {
    ss << "[" << x.node_->GetNamePtr() << "(" << x.node_->GetTypePtr() << "), ";
  } else {
    ss << "[ (Subgraph)";
  }
  switch (x.mem_type_) {
    case ge::OpMemoryType::kOutput:
      ss << "Output, ";
      break;
    case ge::OpMemoryType::kWorkspace:
      ss << "Workspace, ";
      break;
    case ge::OpMemoryType::kOutputDesc:
      ss << "OutputDesc, ";
      break;
    default:
      break;
  }
  ss << x.index_ << ", ref_input:" << x.ref_input_ << ", begin:" << x.life_time_begin_ << ", end:" << x.life_time_end_
     << ", symbol end:" << x.symbol_max_life_time_end_;
  return ss.str();
}

std::string GetName(const ge::MemoryBlock &block, bool last_node) {
  if (!block.NodeTypeIndexList().empty()) {
    if (last_node) {
      return ToString(block.NodeTypeIndexList().back());
    } else {
      return ToString(block.NodeTypeIndexList().front());
    }
  }
  return "";
}

bool CanNotLifeReuse(const ge::MemoryBlock &block, bool child_reuse) {
  if ((!block.reuse_mem_) || ((!child_reuse) && block.child_block_)) {
    return true;
  }
  return false;
}

bool CanReuseBlock(size_t life_begin, const ge::MemoryBlock &reusable_block, size_t block_size) {
  bool can_reuse = false;
  if (reusable_block.Size() == block_size) {
    // in some continuous input case, continuous first input node's is not same as topo first node.
    if (life_begin > 0) {
      if (life_begin > reusable_block.GetLifeEnd(reusable_block.stream_id_)) {
        can_reuse = true;
      }
    } else {
      can_reuse = true;
    }
  }
  return (can_reuse && (!CanNotLifeReuse(reusable_block)));
}

bool ReuseBlock(ge::MemoryBlock &block, const size_t block_size, const size_t life_begin,
                const std::string &batch_label, const ge::NodeTypeIndex &node_type_index) {
  if (block.IsNoAlignSizeReuseBlock() || block.IsRealSizeReuseBlock() || (block.batch_label_ != batch_label)) {
    return false;
  }

  if (block.IsBlockTypeConflictWithNode(node_type_index)) {
    return false;
  }

  if (block.diff_stream_prior_) {
    return false;
  }

  // A node can reuse blocks of the same stream and preorder streams
  if (CanReuseBlock(life_begin, block, block_size)) {
    return true;
  }
  return false;
}

bool CrossLifeTime(const NodeTypeIndex &left, const NodeTypeIndex &right) {
  if ((left.node_ == nullptr) || (right.node_ == nullptr)) {
    return true;
  }
  auto left_node_op_desc = left.node_->GetOpDescBarePtr();
  auto right_node_op_desc = right.node_->GetOpDescBarePtr();
  if ((left_node_op_desc != nullptr) && (right_node_op_desc != nullptr)) {
    if (left.GetLifeBegin() < right.GetLifeBegin()) {
      if (left.life_time_end_ >= right.GetLifeBegin()) {
        return true;
      }
    } else if (left.GetLifeBegin() == right.GetLifeBegin()) {
      return true;
    } else {
      if (right.life_time_end_ >= left.GetLifeBegin()) {
        return true;
      }
    }
  }
  return false;
}

Status SetChildHeadOffset(size_t offset, size_t max_offset, std::vector<MemoryBlock *> &blocks) {
  for (auto block : blocks) {
    if (block != nullptr) {
      GE_ASSERT_SUCCESS(block->SetHeadOffset(offset),
                        "set head offset failed, offset: %zu, block head offset: %zu,"
                        " max_offset: %zu",
                        offset, block->HeadOffset(), max_offset);
      offset += block->Size();
      GE_ASSERT_TRUE(offset <= max_offset, "offset: %zu, max_offset: %zu", offset, max_offset);
    }
  }
  return SUCCESS;
}

void SetChildTailOffset(size_t offset, std::vector<MemoryBlock *> &blocks) {
  for (auto block : blocks) {
    if (block != nullptr) {
      offset += block->Size();
      block->SetTailOffset(offset - 1UL);
    }
  }
}

bool CanBlockLifeReuse(const BlockMemAssigner *const mem_assigner, const MemoryBlock &in_block,
                       const MemoryBlock &out_block, DiffStreamEdgeLife &diff_stream_edge_life) {
  const auto first_node = out_block.NodeTypeIndexList().front();
  if ((first_node.mem_type_ == OpMemoryType::kOutput) &&
      mem_assigner->HasSameOutAnchorWithDiffStream(first_node.node_, first_node.index_)) {
    GELOGD("out_block first node %s(topoid: %lld) output %u use same memory with node on other stream, return false.",
           first_node.node_->GetNamePtr(), first_node.node_->GetOpDescBarePtr()->GetId(), first_node.index_);
    return false;
  }
  if (in_block.IsBlockTypeConflict(out_block)) {
    GELOGD("block type conflict, in_block: %s(%s), out_block: %s.", GetName(in_block).c_str(),
           in_block.BlockTypeStr().c_str(), GetName(out_block).c_str(), out_block.BlockTypeStr().c_str());
    return false;
  }
  GELOGD("in_block[%s] out_block[%s]", GetName(in_block).c_str(), GetName(out_block).c_str());
  if (in_block.stream_id_ == out_block.stream_id_) {
    return (out_block.GetLifeBegin() > in_block.GetLifeEnd(out_block.stream_id_));
  } else {
    auto depend_node_id = out_block.GetDependLifeBegin(in_block.stream_id_, diff_stream_edge_life);
    /// |-stream 1-|         |-stream 2-|
    /// |node1-node3|        |--block---|
    /// |node2-node4-node6|  |--block---|
    /// |--block4--|       \ |--block5---|
    /// |--block---|        \_
    ///                      |node11-node13-node15|
    ///                      |node17-node19|
    /// edge node(node6) is the last in the block, node17 can reuse node6,node3
    size_t tail_node_id = 0UL;
    const auto &node_type_index = in_block.NodeTypeIndexList().back();
    if (node_type_index.node_ != nullptr) {
      auto node_op_desc = node_type_index.node_->GetOpDescBarePtr();
      if (node_op_desc != nullptr) {
        tail_node_id = static_cast<size_t>(node_op_desc->GetId());
      }
    }
    if ((tail_node_id != 0UL) && (depend_node_id >= tail_node_id)) {
      if (!in_block.GetReuseStrategy().memory_priority_mode_) {
        return (depend_node_id > in_block.GetLifeEnd(out_block.stream_id_));
      }
      int64_t end_stream_id = kInvalidStreamId;
      auto in_block_life_end = in_block.GetLifeEnd(out_block.stream_id_, end_stream_id);
      if (end_stream_id == out_block.stream_id_) {
        return (out_block.GetLifeBegin() > in_block_life_end);
      }
      if ((end_stream_id != in_block.stream_id_) && (end_stream_id != kInvalidStreamId)) {
        return (out_block.GetDependLifeBegin(end_stream_id, diff_stream_edge_life) >= in_block_life_end);
      }
      return (depend_node_id > in_block_life_end);
    }
  }
  return false;
}

namespace {

/// When child block's life time are not cross with parent block, they can be reused(only same stream).
/// |-----------------------------parent block---------------------|
/// |------child block1--------------||------child block2------|
/// |--child block1-1-|
bool CanIntervalLifeReuse(const MemoryBlock &parent_block, MemoryBlock &child_block,
                          std::vector<MemoryBlock *> &clone_blocks) {
  // judge by interval life time, only same stream can be judged by interval life time
  bool not_same_stream = ((parent_block.stream_id_ != child_block.stream_id_) || (!parent_block.same_stream_) ||
                          (!child_block.same_stream_) || parent_block.NodeTypeIndexList().empty() ||
                          child_block.NodeTypeIndexList().empty() ||
                          (parent_block.NodeTypeIndexList().back().diff_stream_life_time_.size() > 0U) ||
                          (child_block.NodeTypeIndexList().back().diff_stream_life_time_.size() > 0U));
  if (not_same_stream) {
    return false;
  }
  if (parent_block.IsBlockTypeConflict(child_block)) {
    GELOGD("block type conflict, parent_block: %s(%s), child_block: %s(%s).", GetName(parent_block).c_str(),
           parent_block.BlockTypeStr().c_str(), GetName(child_block).c_str(), child_block.BlockTypeStr().c_str());
    return false;
  }
  bool can_interval_life_reuse = false;
  auto clone_block = child_block.Clone();
  if (clone_block == nullptr) {
    return false;
  }

  bool same_size = ((child_block.NodeTypeIndexList().size() == child_block.RealSizeList().size()) &&
                    (child_block.NodeTypeIndexList().size() == child_block.NoAlignSizeList().size()));
  // ref node must keep in same block
  bool pre_node_cross = false;
  if (same_size) {
    for (auto it = child_block.NodeTypeIndexList().cbegin(); it != child_block.NodeTypeIndexList().cend();) {
      bool cross_node = (((*it).ref_input_ && pre_node_cross) ||
                         ((!(*it).ref_input_) && parent_block.CrossLifeTimeNode(it, child_block)));
      if (cross_node) {
        size_t node_pos = it - child_block.NodeTypeIndexList().cbegin();
        clone_block->AddNodeTypeIndex(*it, child_block.RealSizeList()[node_pos],
                                      child_block.NoAlignSizeList()[node_pos], child_block.stream_id_);
        it = child_block.DelNode(it);
        pre_node_cross = true;
      } else {
        can_interval_life_reuse = true;
        pre_node_cross = false;
        ++it;
      }
    }
  }
  child_block.UpdateContinuousFlag();
  // all life times cross, keep this block
  if (child_block.NodeTypeIndexList().empty()) {
    child_block.Swap(*clone_block);
    delete clone_block;
  } else {
    // partial life times cross, clone a new cross block
    if (!clone_block->NodeTypeIndexList().empty()) {
      clone_blocks.emplace_back(clone_block);
    } else {
      // no life time cross
      delete clone_block;
    }
  }
  if (can_interval_life_reuse) {
    GELOGD("Block size[%zu, %zu] life time are not cross.", parent_block.Size(), child_block.Size());
  }
  return can_interval_life_reuse;
}

}  // namespace

Status MemoryBlock::SetHeadOffset(size_t offset) {
  head_offset_ = offset;
  GE_ASSERT_TRUE(head_offset_ < std::numeric_limits<size_t>::max() - block_size_, "head_offset_: %zu, block_size_: %zu",
                 head_offset_, block_size_);
  const auto max_offset = head_offset_ + block_size_;
  GE_ASSERT_SUCCESS(SetChildHeadOffset(head_offset_, max_offset, child_blocks_),
                    "set child block failed, head_offset: %zu, max_offset: %zu", head_offset_, max_offset);
  GE_ASSERT_SUCCESS(SetChildHeadOffset(head_offset_, max_offset, sub_graph_blocks_),
                    "set subgraph block failed, head_offset: %zu, max_offset: %zu", head_offset_, max_offset);
  for (auto &blocks : batch_to_blocks_) {
    GE_ASSERT_SUCCESS(SetChildHeadOffset(head_offset_, max_offset, blocks.second),
                      "set batch block failed, head_offset: %zu, max_offset: %zu", head_offset_, max_offset);
  }
  return SUCCESS;
}

void MemoryBlock::SetTailOffset(size_t offset) {
  tail_offset_ = offset;
  SetChildTailOffset(head_offset_, child_blocks_);
  SetChildTailOffset(head_offset_, sub_graph_blocks_);
  for (auto &blocks : batch_to_blocks_) {
    SetChildTailOffset(head_offset_, blocks.second);
  }
}

std::vector<MemoryBlock *> MemoryBlock::AllChildBlockList() const {
  std::vector<MemoryBlock *> return_child_blocks;
  return_child_blocks.insert(return_child_blocks.end(), sub_graph_blocks_.cbegin(), sub_graph_blocks_.cend());
  for (auto &batch_blocks : batch_to_blocks_) {
    return_child_blocks.insert(return_child_blocks.end(), batch_blocks.second.cbegin(), batch_blocks.second.cend());
  }
  return_child_blocks.insert(return_child_blocks.end(), child_blocks_.cbegin(), child_blocks_.cend());
  return return_child_blocks;
}

void MemoryBlock::Resize() {
  size_t child_block_size = 0;
  for (auto block : child_blocks_) {
    if (block != nullptr) {
      block->Resize();
      child_block_size += block->Size();
    }
  }
  auto iter = std::max_element(real_size_list_.begin(), real_size_list_.end());
  if (iter == real_size_list_.end()) {
    GELOGW("real_size_list_ is empty");
    return;
  } else {
    size_t block_size = (child_block_size > *iter) ? child_block_size : *iter;
    if ((block_size > 0UL) && (block_size % MEM_ALIGN_SIZE != 0UL)) {
      MemReuseUtils::AlignMemOffset(block_size);
    }
    block_size_ = block_size;
  }
}

size_t MemoryBlock::AlignSize() {
  // Only one calculation, performance optimization
  if (max_real_size_ == 0UL) {
    auto iter = std::max_element(real_size_list_.begin(), real_size_list_.end());
    if (iter == real_size_list_.end()) {
      GELOGW("real_size_list_ is empty");
    } else {
      max_real_size_ = *iter;
      if ((max_real_size_ > 0UL) && ((max_real_size_ % MEM_ALIGN_SIZE) != 0UL)) {
        MemReuseUtils::AlignMemOffset(max_real_size_);
      }
    }
  }
  return max_real_size_;
}

bool MemoryBlock::IsSameBatchLabel() const {
  // only same batch label can reuse
  if (batch_label_.empty() || node_type_index_list_.empty()) {
    return false;
  }

  bool all_same_label = true;
  for (size_t index = 1UL; index < node_type_index_list_.size(); ++index) {
    if (node_type_index_list_[index].node_ == nullptr) {
      continue;
    }
    const auto index_op_desc = node_type_index_list_[index].node_->GetOpDescBarePtr();
    GE_IF_BOOL_EXEC(index_op_desc == nullptr, continue);
    const auto &batch_label = GetBatchLabel(index_op_desc);
    if (batch_label_ != batch_label) {
      all_same_label = false;
      break;
    }
  }
  return all_same_label;
}

bool MemoryBlock::IsGraphInputAndGetSize(const ComputeGraphPtr &compute_graph, size_t &size) const {
  for (const auto &node_type_index : node_type_index_list_) {
    const auto node = node_type_index.node_;
    if (MemReuseUtils::IsDirectInputNode(node, compute_graph)) {
      size = node_type_index.no_align_size_;
      GELOGD("Node:%s is input of %s, size=%zu", node->GetNamePtr(), compute_graph->GetName().c_str(), size);
      return true;
    }
  }
  return false;
}

void MemoryBlock::AddContinuousLifeReuseBlock(MemoryBlock &block) {
  // continuous memory case:only real_size is maximum can be reused and only one continuous memory in one block
  auto it_block = std::max_element(std::begin(block.NoAlignSizeList()), std::end(block.NoAlignSizeList()));
  auto it_this = std::max_element(std::begin(NoAlignSizeList()), std::end(NoAlignSizeList()));
  if (it_block != std::end(block.NoAlignSizeList()) && it_this != std::end(NoAlignSizeList())) {
    if ((IsNoAlignSizeReuseBlock() && block.IsNoAlignSizeReuseBlock()) ||
        (IsNoAlignSizeReuseBlock() && (*it_this < *it_block)) ||
        (block.IsNoAlignSizeReuseBlock() && (*it_this > *it_block))) {
      GELOGD("Conflict current block size:%zu continuous:%d, reuse block max size:%zu continuous:%d.", *it_this,
             GetContinuousFlag(), *it_block, block.GetContinuousFlag());
      return;
    }
  }
  if (IsBlockTypeConflict(block)) {
    GELOGD("block type conflict, this: %s(%s), param block: %s(%s).", GetName(*this).c_str(), BlockTypeStr().c_str(),
           GetName(block).c_str(), block.BlockTypeStr().c_str());
    return;
  }
  // merge small block to large block
  MemoryBlock *parent = nullptr;
  MemoryBlock *child = nullptr;
  if (((child_offset_ + block.AlignSize()) <= *it_this) && (IsNoAlignSizeReuseBlock())) {
    parent = this;
    child = &block;
  } else if (((block.child_offset_ + AlignSize()) <= *it_block) && (block.IsNoAlignSizeReuseBlock()) &&
             (AlignSize() == block.AlignSize()) && child_blocks_.empty()) {
    parent = &block;
    child = this;
  } else {
    return;
  }

  parent->child_blocks_.emplace_back(child);
  parent->child_offset_ += child->AlignSize();
  child->child_block_ = true;
  GELOGI(
      "[no_align_size_block_reuse]"
      "Add block[%s size:%zu, stream id:%" PRId64
      ", life time[begin:%zu, end:%zu], continuous:%d]"
      " to block[%s size:%zu, stream id:%" PRId64 ", life time[begin:%zu, end:%zu], continuous:%d]",
      GetName(*child).c_str(), child->block_size_, child->stream_id_, child->GetLifeBegin(),
      child->GetLifeEnd(child->stream_id_), child->GetContinuousFlag(), GetName(*parent).c_str(), parent->block_size_,
      parent->stream_id_, parent->GetLifeBegin(), parent->GetLifeEnd(parent->stream_id_), parent->GetContinuousFlag());

  return;
}

void MemoryBlock::AddZeroCopyLifeReuseBlock(MemoryBlock &block) {
  auto it_block = std::max_element(block.real_size_list_.begin(), block.real_size_list_.end());
  auto it_this = std::max_element(real_size_list_.begin(), real_size_list_.end());
  if ((it_block == block.real_size_list_.end()) || (it_this == real_size_list_.end())) {
    return;
  }
  if ((is_zero_copy_ && block.is_zero_copy_) || (is_zero_copy_ && (*it_this < *it_block)) ||
      (block.is_zero_copy_ && (*it_this > *it_block))) {
    GELOGD(
        "Conflict current block size:%zu is_reuse_zero_copy:%d is_zero_copy:%d, "
        "reuse block max size:%zu is_reuse_zero_copy:%d is_zero_copy:%d.",
        *it_this, is_reuse_zero_copy_, is_zero_copy_, *it_block, block.is_reuse_zero_copy_, block.is_zero_copy_);
    return;
  }
  if (IsBlockTypeConflict(block)) {
    GELOGD("block type conflict, this: %s(%s), param block: %s(%s).", GetName(*this).c_str(), BlockTypeStr().c_str(),
           GetName(block).c_str(), block.BlockTypeStr().c_str());
    return;
  }
  MemoryBlock *parent = nullptr;
  MemoryBlock *child = nullptr;
  // 如果child_offset_ 都为0，且 real_size 都相等，也是允许复用
  if ((((child_offset_ + block.AlignSize()) <= *it_this) ||
       ((child_offset_ == 0UL) && (block.child_offset_ == 0UL) && (*it_block == *it_this))) &&
      is_zero_copy_) {
    parent = this;
    child = &block;
  } else if ((((block.child_offset_ + AlignSize()) <= *it_block) ||
              ((child_offset_ == 0UL) && (block.child_offset_ == 0UL) && (*it_block == *it_this))) &&
             block.is_zero_copy_ && (AlignSize() == block.AlignSize()) && child_blocks_.empty()) {
    parent = &block;
    child = this;
  } else {
    return;
  }

  if ((parent->is_zero_copy_) && (!child->is_reuse_zero_copy_)) {
    return;
  }

  parent->child_blocks_.emplace_back(child);
  parent->child_offset_ += child->AlignSize();
  child->child_block_ = true;
  parent->is_reuse_zero_copy_ = (child->is_reuse_zero_copy_ && parent->is_reuse_zero_copy_);
  GELOGI(
      "[zero_copy_size_block_reuse]"
      "Add block[%s size:%zu, stream id:%" PRId64
      ", life time[begin:%zu, end:%zu], continuous:%d, is_zero_copy:%d]"
      " to block[%s size:%zu, stream id:%" PRId64 ", life time[begin:%zu, end:%zu], continuous:%d, is_zero_copy:%d]",
      GetName(*child).c_str(), child->block_size_, child->stream_id_, child->GetLifeBegin(),
      child->GetLifeEnd(child->stream_id_), child->GetContinuousFlag(), child->is_zero_copy_, GetName(*parent).c_str(),
      parent->block_size_, parent->stream_id_, parent->GetLifeBegin(), parent->GetLifeEnd(parent->stream_id_),
      parent->GetContinuousFlag(), parent->is_zero_copy_);

  return;
}

bool MemoryBlock::AddLifeReuseBlock(const BlockMemAssigner *const mem_assigner, MemoryBlock *block,
                                    std::vector<MemoryBlock *> &clone_blocks, uint32_t depth,
                                    DiffStreamEdgeLife &diff_stream_edge_life, bool child_reuse) {
  GELOGD("this[%s size:%zu, stream id:%" PRId64
         " life time[begin:%zu, end:%zu] childs:%zu] "
         "block[%s size:%zu, stream id:%" PRId64 ", life time[begin:%zu, end:%zu] childs:%zu]",
         GetName(*this).c_str(), block_size_, stream_id_, GetLifeBegin(), GetLifeEnd(block->stream_id_),
         child_blocks_.size(), GetName(*block).c_str(), block->block_size_, block->stream_id_, block->GetLifeBegin(),
         block->GetLifeEnd(stream_id_), block->child_blocks_.size());
  ++depth;
  const bool can_not_life_reuse =
      (CanNotLifeReuse(*this, child_reuse) || CanNotLifeReuse(*block) || (batch_label_ != block->batch_label_) ||
       (memory_type_ != block->memory_type_) || (depth > kMaxDepthNum));
  if (can_not_life_reuse || (!block->child_blocks_.empty())) {
    return false;
  }

  // Different streams must use stream dependency to judge the life cycle
  // In case same stream if it has child block, can judge all the child block's life time in CanIntervalLifeReuse
  bool can_block_life_reuse = CanBlockLifeReuse(mem_assigner, *this, *block, diff_stream_edge_life) ||
                              CanBlockLifeReuse(mem_assigner, *block, *this, diff_stream_edge_life);
  const bool is_continue_reuse_zero_copy =
      (is_zero_copy_ &&
       (block->GetFirstContinuousFlag() || block->GetLastContinuousFlag() || block->GetContinuousFlag())) ||
      (block->is_zero_copy_ && (GetFirstContinuousFlag() || GetLastContinuousFlag() || GetContinuousFlag()));
  GELOGD("continuous cannot reuse zero copy, is_continue_not_reuse_zero_copy:%d", is_continue_reuse_zero_copy);
  if (is_continue_reuse_zero_copy) {
    return false;
  }
  // continuous block reuse proc
  const bool is_no_align_size_reuse_block = IsNoAlignSizeReuseBlock() || block->IsNoAlignSizeReuseBlock();
  if (is_no_align_size_reuse_block) {
    if (can_block_life_reuse) {
      AddContinuousLifeReuseBlock(*block);
    }
    return true;
  }
  // zero copy block reuse proc
  const bool is_real_size_reuse_block = IsRealSizeReuseBlock() || block->IsRealSizeReuseBlock();
  if (is_real_size_reuse_block) {
    if (can_block_life_reuse) {
      AddZeroCopyLifeReuseBlock(*block);
    }
    return true;
  }

  if (!can_block_life_reuse && !CanIntervalLifeReuse(*this, *block, clone_blocks)) {
    return false;
  }

  // |-parent block---------------------------------------|
  // |-child block level 1----|-child block level 1----|
  // |-child block level 2-|
  for (auto child_block : child_blocks_) {
    if ((child_block != nullptr) &&
        child_block->AddLifeReuseBlock(mem_assigner, block, clone_blocks, depth, diff_stream_edge_life, true)) {
      return true;
    }
  }

  // merge small block to large block
  // noalign size         802816 + 802816 = 1605632       can reuse
  // after 32 align size  802848 + 802848 > 1605664       can't reuse
  // after 512 align size 803328 + 803328 > 1606144       can't reuse
  // so                   803328 + 803328 = 1606144 + 512 can reuse
  if (block->AlignSize() != MEM_ALIGN_SIZE) {
    if ((child_offset_ + block->AlignSize()) > (AlignSize() + MEM_ALIGN_SIZE)) {
      return false;
    }
  } else {
    if ((child_offset_ + block->AlignSize()) > AlignSize()) {
      return false;
    }
  }

  child_blocks_.emplace_back(block);
  is_reuse_zero_copy_ = (block->is_reuse_zero_copy_ && is_reuse_zero_copy_);
  child_offset_ += block->AlignSize();
  block->child_block_ = true;
  GELOGI("Add block[%s size:%zu, stream id:%" PRId64
         " life time[begin:%zu, end:%zu]] to"
         " block[%s size:%zu, stream id:%" PRId64 ", life time[begin:%zu, end:%zu]]",
         GetName(*block).c_str(), block->block_size_, block->stream_id_, block->GetLifeBegin(),
         block->GetLifeEnd(stream_id_), GetName(*this).c_str(), block_size_, stream_id_, GetLifeBegin(),
         GetLifeEnd(block->stream_id_));
  return true;
}

size_t MemoryBlock::GetLifeBegin(bool for_sort) const {
  if (!node_type_index_list_.empty()) {
    return node_type_index_list_.front().GetLifeBegin(for_sort);
  }
  return 0UL;
}

/// |-stream 1-|   |-stream 2-|
/// |--block1--|   |--block---|
/// |--block2--|   |--block---|
/// |--block3--|\  |--block---|
/// |--block4--| \ |--block5---|
/// |--block---|  \|--block6---|
/// |--block---|   |--block7--|
/// |--block---|   |--block---|
/// block7's first node's input node's life begin > block2's life end, block7 can reuse block1~block2
size_t MemoryBlock::GetDependLifeBegin(int64_t stream_id, DiffStreamEdgeLife &diff_stream_edge_life) const {
  GELOGD("In depend node:[%s] stream_id:[%" PRId64 "->%" PRId64 "] self life time[%" PRId64 "-%" PRId64 "]",
         NodeTypeIndexList().front().node_->GetNamePtr(), stream_id_, stream_id, GetLifeBegin(), GetLifeEnd(stream_id));
  const auto it = diff_stream_edge_life.find(stream_id_);
  if (it == diff_stream_edge_life.cend()) {
    return 0UL;
  }
  const auto edges_it = it->second.find(stream_id);
  if (edges_it == it->second.cend()) {
    return 0UL;
  }

  /// |-stream 1-|         |-stream 2-|
  /// |node1-node3|        |--block---|
  /// |node2-node4-node6|  |--block---|
  /// |--block4--|       \ |node7-node9|
  /// |--block---|        \_
  ///                      |node11-node13-node15|
  ///                      |node17-node19|
  auto first_node_id = GetLifeBegin();
  auto edge_it = edges_it->second.lower_bound({first_node_id, 0UL});
  if (edges_it->second.empty()) {
    return 0UL;
  }
  // lower_bound find node17, not found, so use node11-->node6
  if ((edge_it == edges_it->second.end()) || ((*edge_it).node_id > first_node_id)) {
    // lower_bound find node7, get node11-->node6, because node11 > node7, so return no depend node
    if (edge_it == edges_it->second.begin()) {
      GELOGD("Depend lower node id:%" PRId64 " > node id:%" PRId64 ".", (*edge_it).node_id, GetLifeBegin());
      return 0UL;
    }
    // not found, use tail data
    --edge_it;
  }

  // lower_bound find node11, get node11-->node6
  GELOGD("Node:[%s] life begin:%" PRId64 " stream_id:[%" PRId64 "->%" PRId64 "] depend life_time:[%" PRId64 "->%" PRId64
         "]",
         NodeTypeIndexList().front().node_->GetNamePtr(), first_node_id, stream_id_, stream_id, (*edge_it).node_id,
         (*edge_it).peer_node_id);
  return (*edge_it).peer_node_id;
}

// 这里stream_id和self stream_id可能不同，最终会和stream_id block->GetDependLifeBegin(self stream_id)比较确保正确性
size_t MemoryBlock::GetLifeEnd(int64_t stream_id) const {
  if (!node_type_index_list_.empty()) {
    const bool only_to_one_stream = (node_type_index_list_.back().out_stream_count_ == 1U) &&
                                    (node_type_index_list_.back().diff_stream_life_time_.size() == 1U);
    const auto it = node_type_index_list_.back().diff_stream_life_time_.find(stream_id);
    if (only_to_one_stream && (it != node_type_index_list_.back().diff_stream_life_time_.cend())) {
      GELOGD("block %s stream[%" PRId64 "] [%" PRId64 "] life[%" PRId64 "]", GetName(*this).c_str(), stream_id_,
             stream_id, it->second);
      return it->second;
    }

    GELOGD("block %s stream[%" PRId64 "] [%" PRId64 "] life[%" PRId64 "]", GetName(*this).c_str(), stream_id_,
           stream_id, node_type_index_list_.back().life_time_end_);
    return node_type_index_list_.back().life_time_end_;
  }
  return kMaxLifeTime;
}

/// |-stream 1-|         |-stream 2-|     |-stream 3-|
/// |node1-node3|        |--block---|     |--block---|
/// |node2-node4-node6|
/// |--block4--|       \                  |--block---|
/// |--block---|        \                 |--block---|
///                      |node11|
///                      |node17-node19|
///                      |--block---|  \  |--block---|
///                                     \ |--block---|
///                                      |node30-node32|
///                                       |--block---|
size_t MemoryBlock::GetLifeEnd(int64_t stream_id, int64_t &end_stream_id) const {
  end_stream_id = stream_id_;
  if (!node_type_index_list_.empty()) {
    const bool only_to_one_stream = (node_type_index_list_.back().out_stream_count_ == 1U) &&
                                    (node_type_index_list_.back().diff_stream_life_time_.size() == 1U);
    if (!only_to_one_stream) {
      // stream_id is 1, return normal end life time in stream 1 or kMaxLifeTime
      GELOGD("block %s stream[%" PRId64 "] [%" PRId64 "] life[%" PRId64 "]", GetName(*this).c_str(), stream_id_,
             stream_id, node_type_index_list_.back().life_time_end_);
      return node_type_index_list_.back().life_time_end_;
    }
    const auto it = node_type_index_list_.back().diff_stream_life_time_.find(stream_id);
    // out to only one diff stream, stream_id is 2, end_stream_id is 2, return node11
    if (it != node_type_index_list_.back().diff_stream_life_time_.cend()) {
      GELOGD("block %s stream[%" PRId64 "] [%" PRId64 "] life[%" PRId64 "]", GetName(*this).c_str(), stream_id_,
             stream_id, it->second);
      end_stream_id = stream_id;
      return it->second;
    }
    // out to only one diff stream, stream_id is 3, end_stream_id is 2, return node11
    if (stream_id != stream_id_) {
      end_stream_id = node_type_index_list_.back().diff_stream_life_time_.begin()->first;
      GELOGD("block %s stream[%" PRId64 "] [%" PRId64 "] [%" PRId64 "] life[%" PRId64 "]", GetName(*this).c_str(),
             stream_id_, end_stream_id, stream_id, node_type_index_list_.back().diff_stream_life_time_.begin()->second);
      return node_type_index_list_.back().diff_stream_life_time_.begin()->second;
    }
  }
  return kMaxLifeTime;
}

size_t MemoryBlock::GetSymbolLifeEnd() const {
  if (!node_type_index_list_.empty()) {
    return node_type_index_list_.back().symbol_max_life_time_end_;
  }
  return kDefaultLifeTime;
}

void MemoryBlock::SetSymbolLifeEnd(size_t symbol_life_end) {
  if (!node_type_index_list_.empty()) {
    if ((node_type_index_list_.back().symbol_max_life_time_end_ == kDefaultLifeTime) ||
        symbol_life_end > node_type_index_list_.back().symbol_max_life_time_end_) {
      node_type_index_list_.back().symbol_max_life_time_end_ = symbol_life_end;
    }
  }
}

void MemoryBlock::SetLifeTimeEnd(size_t time, int64_t stream_id) {
  if (!node_type_index_list_.empty()) {
    if (stream_id != stream_id_) {
      auto it = node_type_index_list_.back().diff_stream_life_time_.find(stream_id);
      if (it == node_type_index_list_.back().diff_stream_life_time_.end()) {
        node_type_index_list_.back().diff_stream_life_time_[stream_id] = time;
      } else if (time > it->second) {
        it->second = time;
      } else {
      }

      if (node_type_index_list_.back().life_time_end_ == kDefaultLifeTime) {
        node_type_index_list_.back().life_time_end_ = kMaxLifeTime;
      }
    } else {
      if ((node_type_index_list_.back().life_time_end_ == kDefaultLifeTime) ||
          (time > node_type_index_list_.back().life_time_end_)) {
        node_type_index_list_.back().life_time_end_ = time;
      }
    }
  }
}

void MemoryBlock::SetOutStreamLifeTime(size_t out_time, size_t end_time, int64_t stream_id) {
  const size_t symbol_life_time = GetSymbolLifeEnd();
  if ((symbol_life_time != kDefaultLifeTime) && (end_time < symbol_life_time)) {
    end_time = symbol_life_time;
    GELOGI("Block %s has continuous input node, which include multiple ref, end time is: %" PRId64 "",
           GetName(*this, true).c_str(), end_time);
  }
  end_time = (end_time < out_time) ? kMaxLifeTime : end_time;
  if (!node_type_index_list_.empty()) {
    auto iter = node_type_index_list_.back().out_stream_life_time_.find(stream_id);
    if (iter == node_type_index_list_.back().out_stream_life_time_.end()) {
      node_type_index_list_.back().out_stream_life_time_.emplace(stream_id, std::make_pair(out_time, end_time));
      node_type_index_list_.back().SetOutStreamCount(node_type_index_list_.back().out_stream_life_time_.size());
      return;
    }

    if (out_time > iter->second.first) {
      iter->second.first = out_time;
    }
    if (end_time > iter->second.second) {
      iter->second.second = end_time;
    }
  }
}

bool MemoryBlock::CrossLifeTimeNode(const std::vector<NodeTypeIndex>::const_iterator &it,
                                    const MemoryBlock &child_block) const {
  if (node_type_index_list_.empty()) {
    return false;
  }

  const NodeTypeIndex &node_type_index = *it;
  // quick judge life time by begin and end
  if (!((node_type_index.life_time_end_ < node_type_index_list_.front().GetLifeBegin()) ||
        (node_type_index.GetLifeBegin() > node_type_index_list_.back().life_time_end_))) {
    for (const auto &node : node_type_index_list_) {
      if (CrossLifeTime(node, node_type_index)) {
        return true;
      }
    }
  }

  if (node_type_index.next_is_ref_input_) {
    // all ref node must in same block, judge all the ref node and return same result
    auto ref_it = it;
    ref_it++;
    for (; ref_it != child_block.NodeTypeIndexList().cend(); ++ref_it) {
      if (!(*ref_it).ref_input_) {
        break;
      }
      for (const auto &node : node_type_index_list_) {
        if (CrossLifeTime(node, *ref_it)) {
          return true;
        }
      }
    }
  }
  return false;
}

MemoryBlock *MemoryBlock::Clone() const {
  auto block = new (std::nothrow) MemoryBlock(reuse_strategy_, block_size_, stream_id_, reuse_mem_, memory_type_);
  if (block != nullptr) {
    // 复用中作为判断条件的字段都需要clone，其他字段不需要clone
    block->same_stream_ = same_stream_;
    block->is_zero_copy_ = is_zero_copy_;
    block->is_reuse_zero_copy_ = is_reuse_zero_copy_;
    block->memory_type_logic_base_ = memory_type_logic_base_;
    block->need_same_offset_in_batch_ = need_same_offset_in_batch_;
    block->ref_count_ = ref_count_;
    block->input_index_ = input_index_;
    block->batch_label_ = batch_label_;
    block->has_sub_graph_in_out_node_ = has_sub_graph_in_out_node_;
    block->post_reuse_flag_ = post_reuse_flag_;
    block->is_fixed_addr_prior_ = is_fixed_addr_prior_;
    block->block_type_list_ = block_type_list_;
  }
  return block;
}

void MemoryBlock::UpdateContinuousFlag() {
  first_continuous_block_ = false;
  last_continuous_block_ = false;
  continuous_block_ = false;
  for (const auto &node : node_type_index_list_) {
    if (node.GetFirstContinuousNodeFlag()) {
      first_continuous_block_ = true;
    }
    if (node.GetLastContinuousNodeFlag()) {
      last_continuous_block_ = true;
    }
    if (node.GetContinuousNodeFlag()) {
      continuous_block_ = true;
    }
  }
}

void MemoryBlock::Swap(MemoryBlock &block) {
  node_type_index_list_.swap(block.node_type_index_list_);
  real_size_list_.swap(block.real_size_list_);
  no_align_size_list_.swap(block.no_align_size_list_);
  block_type_list_.swap(block.block_type_list_);
}

// call UpdateContinuousFlag after DelNode
std::vector<NodeTypeIndex>::const_iterator MemoryBlock::DelNode(std::vector<NodeTypeIndex>::const_iterator &it) {
  // vector sizes are same
  if ((node_type_index_list_.size() == real_size_list_.size()) &&
      (node_type_index_list_.size() == no_align_size_list_.size())) {
    const auto to_delete = *it;
    size_t node_pos = it - node_type_index_list_.begin();
    auto return_it = node_type_index_list_.erase(it);
    real_size_list_.erase(real_size_list_.cbegin() + node_pos);
    no_align_size_list_.erase(no_align_size_list_.cbegin() + node_pos);
    block_type_list_.WithDeleted(*this, to_delete);
    return return_it;
  }
  return ++it;
}

std::string MemoryBlock::String() const {
  std::stringstream ss;
  ss << "Block size: " << Size() << " from " << HeadOffset() << " to " << TailOffset() << " ";
  ss << "ref_count: " << ref_count_ << " ";
  ss << "stream_id: " << stream_id_ << " ";
  ss << "is_zero_copy: " << is_zero_copy_ << " ";
  ss << "reuse_mem_: " << reuse_mem_ << " ";
  ss << "no_align_size: " << ToString(no_align_size_list_) << " ";
  ss << "real_size_list: " << ToString(real_size_list_) << " ";
  ss << "members: ";
  for (auto x : NodeTypeIndexList()) {
    ss << "__node: " << ToString(x) << " ";
  }
  for (const auto &symbol : SymbolList()) {
    ss << "__symbol: " << symbol << " ";
  }
  ss << "memory_type: " << memory_type_ << " ";
  return ss.str();
}

// ascending order
bool CompareBlockIndex(const MemoryBlock *const left, const MemoryBlock *const right) {
  return (left != nullptr) && (right != nullptr) && (left->input_index_ < right->input_index_);
}

bool CompareLifeInterval::operator()(MemoryBlock *const left, MemoryBlock *const right) const {
  if ((left != nullptr) && (right != nullptr)) {
    auto left_size = left->AlignSize();
    auto right_size = right->AlignSize();
    if (left->GetContinuousFlag()) {
      auto it = std::max_element(std::begin(left->NoAlignSizeList()), std::end(left->NoAlignSizeList()));
      if (it != left->NoAlignSizeList().end()) {
        left_size = *it;
      }
    }

    if (right->GetContinuousFlag()) {
      auto it = std::max_element(std::begin(right->NoAlignSizeList()), std::end(right->NoAlignSizeList()));
      if (it != right->NoAlignSizeList().end()) {
        right_size = *it;
      }
    }

    if (left_size == right_size) {
      if (!reuse_strategy_.ascending_sort_) {
        return (left->GetLifeBegin(true) > right->GetLifeBegin(true));
      }
      if (left->NodeTypeIndexList().size() == right->NodeTypeIndexList().size()) {
        return (left->GetLifeBegin(true) < right->GetLifeBegin(true));
      } else {
        return (left->NodeTypeIndexList().size() < right->NodeTypeIndexList().size());
      }
    } else {
      return (left_size > right_size);
    }
  }
  return false;
}

Status AddBlockMemOffset(std::map<uint64_t, size_t> &mem_offsets, MemoryBlock &block) {
  auto it = mem_offsets.find(block.memory_type_);
  if (it == mem_offsets.end()) {
    auto result = mem_offsets.insert(std::pair<int64_t, size_t>(block.memory_type_, block.memory_type_logic_base_));
    GE_ASSERT_TRUE(result.second);
    it = result.first;
  }
  GE_ASSERT_TRUE(it != mem_offsets.end());
  auto &mem_offset = it->second;
  block.Resize();
  GE_ASSERT_SUCCESS(block.SetHeadOffset(mem_offset));
  mem_offset += block.Size();
  block.SetTailOffset(mem_offset - 1);
  return SUCCESS;
}

}  // namespace ge
