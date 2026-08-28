/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph/build/memory/atomic_memory_assigner.h"
#include <cinttypes>
#include <algorithm>
#include "common/checker.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/utils/tensor_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/op_type_utils.h"
#include "graph/manager/graph_var_manager.h"
#include "runtime/subscriber/global_profiler.h"
#include "graph/optimize/mem_layout_conflict_optimize/mem_layout_conflict_util.h"
#include "graph/build/memory/checker/special_node_checker.h"

namespace ge {

namespace {
const int32_t kAllInputAddrIsAtomic = -1;
constexpr size_t kMaxLogCharNum = 1200U;
constexpr const char_t *TBE_OP_ATOMIC_DTYPES = "tbe_op_atomic_dtypes";
constexpr const char_t *TBE_OP_ATOMIC_INT64_VALUES = "tbe_op_atomic_int64_values";
constexpr const char_t *TBE_OP_ATOMIC_FLOAT_VALUES = "tbe_op_atomic_float_values";
}  // namespace

std::string GraphNameId(const ge::ComputeGraph *const graph) {
  return ge::MemReuseUtils::GetGraphNameId(graph);
}

bool IsZeroCopyOut(const ge::OpDesc *op_desc, int64_t index) {
  const auto tensor_desc = op_desc->MutableOutputDesc(index);
  bool is_zero_block = false;
  (void)ge::AttrUtils::GetBool(tensor_desc, ge::ATTR_IS_ZERO_COPY_BLOCK, is_zero_block);
  return is_zero_block;
}

std::vector<int32_t> GetAtomicDataTypeList(const ge::Node *atomic_node) {
  std::vector<int32_t> data_type_list;
  (void)ge::AttrUtils::GetListInt(atomic_node->GetOpDesc(), TBE_OP_ATOMIC_DTYPES, data_type_list);
  return data_type_list;
}

std::vector<int64_t> GetAtomicIntValList(const ge::Node *atomic_node) {
  std::vector<int64_t> int_list;
  (void)ge::AttrUtils::GetListInt(atomic_node->GetOpDesc(), TBE_OP_ATOMIC_INT64_VALUES, int_list);
  return int_list;
}

std::vector<float32_t> GetAtomicFloatValList(const ge::Node *atomic_node) {
  std::vector<float32_t> float_list;
  (void)ge::AttrUtils::GetListFloat(atomic_node->GetOpDesc(), TBE_OP_ATOMIC_FLOAT_VALUES, float_list);
  return float_list;
}

std::vector<int32_t> GetMemsetDataTypeList(const ge::NodePtr &atomic_node) {
  std::vector<int32_t> data_type_list;
  (void)ge::AttrUtils::GetListInt(atomic_node->GetOpDesc(), ge::ATTR_NAME_ATOMIC_MEMSET_DTYPES, data_type_list);
  return data_type_list;
}

bool IsCrossSplitSegment(const std::map<int64_t, int64_t> &split_offset_to_size, const ge::CleanMemInfo &lh,
                         const ge::CleanMemInfo &rh) {
  if (split_offset_to_size.size() <= 1U) {
    return false;
  }

  // 查找lh.offset所在的split段
  auto it = split_offset_to_size.upper_bound(lh.offset);
  if (it != split_offset_to_size.begin()) {
    --it;  // 移动到包含lh.offset的段
  }

  // 检查lh和rh是否都在同一个split段内
  if (it != split_offset_to_size.end()) {
    const int64_t split_start = it->first;
    const int64_t split_end = split_start + it->second;

    // 如果lh和rh都在同一个split段内
    if ((lh.offset >= split_start) && (rh.offset + rh.size <= split_end)) {
      return false;  // 不跨越split边界
    }
  }

  return true;  // 跨越了split边界
}

Status AtomicMemoryAssigner::ReAssignAtomicMemory() {
  // batch_lable, memset_node, atomic_nodes
  std::map<std::string, std::map<NodePtr, std::vector<NodePtr>>> batch_to_memset_to_atomic_nodes;
  Status status = FilterAtomicNodes(batch_to_memset_to_atomic_nodes);
  if (status != SUCCESS) {
    GELOGE(status, "[Filter][AtomicNode]failed in graph_id:%u, graph_name:%s", compute_graph_->GetGraphID(),
           compute_graph_->GetName().c_str());
    return status;
  }

  std::map<int64_t, size_t> mem_type_to_batch_atomic_mem_start;
  std::map<int64_t, size_t> mem_type_to_batch_max_offset;
  for (const auto &offset_iter : memory_offset_) {
    mem_type_to_batch_atomic_mem_start[offset_iter.first] = offset_iter.second.mem_offset_;
    mem_type_to_batch_max_offset[offset_iter.first] = offset_iter.second.mem_offset_;
  }

  for (auto &iter_batch : batch_to_memset_to_atomic_nodes) {
    for (auto &offset_iter : memory_offset_) {
      offset_iter.second.mem_offset_ = mem_type_to_batch_atomic_mem_start[offset_iter.first];
    }

    for (auto &iter : iter_batch.second) {
      std::map<int64_t, size_t> mem_type_to_atomic_mem_start;
      for (const auto &offset_iter : memory_offset_) {
        mem_type_to_atomic_mem_start[offset_iter.first] = offset_iter.second.mem_offset_;
      }
      std::map<int64_t, std::vector<int64_t>> type_to_atomic_nodes_mem_starts;
      std::map<int64_t, std::vector<int64_t>> type_to_atomic_nodes_mem_sizes;
      for (auto &atomic_node : iter.second) {
        std::map<int64_t, std::vector<int64_t>> mem_type_to_offset_ends;
        std::map<int64_t, std::vector<int64_t>> mem_type_to_real_atomic_sizes;
        GE_ASSERT_SUCCESS(
            AssignAtomicOutputAndWorkspaceMemory(atomic_node, mem_type_to_offset_ends, mem_type_to_real_atomic_sizes),
            "[Assign][Memory]output atomic mem and workspace mem, fail for node name is %s.",
            atomic_node->GetNamePtr());
      }

      for (const auto &offset_iter : memory_offset_) {
        mem_type_to_batch_max_offset[offset_iter.first] =
            std::max(mem_type_to_batch_max_offset[offset_iter.first], offset_iter.second.mem_offset_);
      }
    }

    for (auto &offset_iter : memory_offset_) {
      offset_iter.second.mem_offset_ = mem_type_to_batch_max_offset[offset_iter.first];
      mem_type_to_batch_atomic_mem_start[offset_iter.first] = mem_type_to_batch_max_offset[offset_iter.first];
    }
  }
  return SUCCESS;
}

void Print(const ge::NodePtr &node, const std::vector<ge::CleanMemInfo> &clean_mem_infos,
           const ge::MemsetNodeAddrAndAttr &addr_and_type) {
  if (!IsLogEnable(GE_MODULE_NAME, DLOG_INFO)) {
    return;
  }
  std::stringstream ss;
  ss << "memset_node: " << node->GetName() << "(" << node->GetType() << "), clean_mem_infos: ";
  for (const auto &clean_mem_info : clean_mem_infos) {
    ss << "[" << clean_mem_info.ToStr() << "]";
    if (ss.str().length() > kMaxLogCharNum) {
      GELOGI("[AtomicClean]%s", ss.str().c_str());
      ss.str("");
      ss.clear();
    }
  }
  GELOGI("[AtomicClean]%s", ss.str().c_str());
  ss.str("");
  ss.clear();
  ss << "memset node offsets: " << ToString(addr_and_type.offsets) << ", sizes: " << ToString(addr_and_type.sizes);
  GELOGI("[AtomicClean]%s", ss.str().c_str());
  ss.str("");
  ss.clear();
  ss << "data_types: " << ToString(addr_and_type.data_type_list) << ", int_list: " << ToString(addr_and_type.int_list)
     << ", float_list: " << ToString(addr_and_type.float_list);
  GELOGI("[AtomicClean]%s", ss.str().c_str());
}

Status AtomicMemoryAssigner::SetAtomicCleanOffset() const {
  GE_CHECK_NOTNULL(compute_graph_);
  const auto split_offset_to_size = GetSplitOffsetSize();
  for (const auto &node : compute_graph_->GetAllNodes()) {
    if (!NodeUtils::IsLikeAtomicClean(node)) {
      continue;
    }
    std::set<CleanMemInfo> clean_mem_infos;
    GE_ASSERT_SUCCESS(CollectAtomicNodeCleanMemInfos(node, clean_mem_infos),
                      "collect atomic clean memory infos failed, node: %s(%s)", node->GetNamePtr(), node->GetTypePtr());
    if (clean_mem_infos.empty()) {
      continue;
    }
    const auto merged_clean_mem_infos = MergeCleanMemInfos(clean_mem_infos, split_offset_to_size);
    const auto memset_addr_attr = ConstructMemsetAddrAndAttr(merged_clean_mem_infos);
    Print(node, merged_clean_mem_infos, memset_addr_attr);
    GE_ASSERT_SUCCESS(AppendAttrsToMemSetOp(node, memset_addr_attr));
    GE_ASSERT_SUCCESS(AppendAddrSizeToMemSetOp(node, memset_addr_attr));
  }
  return SUCCESS;
}

Status AtomicMemoryAssigner::CollectAtomicNodeCleanMemInfos(const NodePtr &memset_node,
                                                            std::set<CleanMemInfo> &clean_mem_infos) const {
  GE_ASSERT_NOTNULL(memset_node);
  GELOGI("[AtomicClean]start to collect atomic clean memory infos for memset node: %s(%s), start size: %zu",
         memset_node->GetNamePtr(), memset_node->GetTypePtr(), clean_mem_infos.size());
  const auto &out_ctl_anchor = memset_node->GetOutControlAnchor();
  GE_CHECK_NOTNULL(out_ctl_anchor);
  const auto all_peer_in_ctrl_anchors = out_ctl_anchor->GetPeerInControlAnchorsPtr();
  for (const auto &in_ctl_anchor : all_peer_in_ctrl_anchors) {
    const auto atomic_node = in_ctl_anchor->GetOwnerNode();
    GE_ASSERT_NOTNULL(atomic_node);
    const auto atomic_op_desc = atomic_node->GetOpDescBarePtr();
    GE_ASSERT_NOTNULL(atomic_op_desc);
    bool is_atomic_node = false;
    // If GetBool fail, is_atomic_node is false.
    (void)ge::AttrUtils::GetBool(atomic_op_desc, ATOMIC_ATTR_IS_ATOMIC_NODE, is_atomic_node);
    if (!is_atomic_node) {
      // hcom算子要求对所有输入清零，但是atomic_addr_clean_pass.cc不会打ATOMIC_ATTR_IS_ATOMIC_NODE属性
      const auto has_atomic_input = atomic_op_desc->HasAttr(ATOMIC_ATTR_INPUT_INDEX);
      const auto has_atomic_output = atomic_op_desc->HasAttr(ATOMIC_ATTR_OUTPUT_INDEX);
      const auto atomic_workspace_index_size = atomic_op_desc->TryGetExtAttr(
          EXT_ATTR_ATOMIC_WORKSPACE_INFO, std::map<std::string, std::map<int64_t, int64_t>>{});
      if ((!has_atomic_input) && (!has_atomic_output) && atomic_workspace_index_size.empty()) {
        continue;
      }
    }
    AtomicNodeCleanTypeVals type_vals;
    GE_ASSERT_SUCCESS(type_vals.Init(atomic_node.get()), "atomic_node: %s(%s) get atomic attrs failed",
                      atomic_node->GetNamePtr());
    GE_ASSERT_SUCCESS(GetInputCleanMemInfos(atomic_node, clean_mem_infos),
                      "collect atomic node offsets failed, memset_node: %s", memset_node->GetNamePtr());
    GE_ASSERT_SUCCESS(GetOutputCleanMemInfos(atomic_node, type_vals, clean_mem_infos),
                      "collect atomic node offsets failed, memset_node: %s", memset_node->GetNamePtr());
    GE_ASSERT_SUCCESS(GetWorkspaceCleanMemInfos(atomic_node, type_vals, clean_mem_infos),
                      "collect atomic node offsets failed, memset_node: %s", memset_node->GetNamePtr());
  }
  GELOGI(
      "[AtomicClean]finish to collect atomic clean memory infos for memset node: %s(%s),"
      " now clean_mem_infos size: %zu, control out nodes: %zu",
      memset_node->GetNamePtr(), memset_node->GetTypePtr(), clean_mem_infos.size(), all_peer_in_ctrl_anchors.size());
  return SUCCESS;
}

// 把相邻的地址合并到一起, 但是不能跨越拆分的边界
std::vector<CleanMemInfo> AtomicMemoryAssigner::MergeCleanMemInfos(
    const std::set<CleanMemInfo> &clean_mem_infos, const std::map<int64_t, int64_t> &split_offset_to_size) const {
  std::vector<CleanMemInfo> merged;
  merged.reserve(clean_mem_infos.size());
  auto origin_iter = clean_mem_infos.begin();
  merged.emplace_back(*origin_iter++);

  while (origin_iter != clean_mem_infos.end()) {
    if (merged.back().CanMerge(*origin_iter) &&
        (!IsCrossSplitSegment(split_offset_to_size, merged.back(), *origin_iter))) {
      merged.back().Merge(*origin_iter++);
    } else {
      merged.emplace_back(*origin_iter++);
    }
  }
  return merged;
}

MemsetNodeAddrAndAttr AtomicMemoryAssigner::ConstructMemsetAddrAndAttr(
    const std::vector<CleanMemInfo> &clean_mem_infos) const {
  MemsetNodeAddrAndAttr memset_addr_and_attr(clean_mem_infos.size());
  if (!clean_mem_infos.empty()) {
    bool clear_memory_type = true;
    int64_t first_memory_type = clean_mem_infos.front().memory_type;
    for (const auto &mem_info : clean_mem_infos) {
      memset_addr_and_attr.offsets.emplace_back(mem_info.offset);
      memset_addr_and_attr.sizes.emplace_back(mem_info.size);
      memset_addr_and_attr.memory_types.emplace_back(mem_info.memory_type);
      if ((first_memory_type != mem_info.memory_type) || (mem_info.memory_type == RT_MEMORY_P2P_DDR)) {
        clear_memory_type = false;
      }
      memset_addr_and_attr.data_type_list.emplace_back(mem_info.type_val.data_type);
      if (IsFloatType(static_cast<ge::DataType>(mem_info.type_val.data_type))) {
        memset_addr_and_attr.float_list.emplace_back(mem_info.type_val.float_val);
      } else {
        memset_addr_and_attr.int_list.emplace_back(mem_info.type_val.int_val);
      }
    }
    if (clear_memory_type) {
      memset_addr_and_attr.memory_types.clear();
    }
  }

  return memset_addr_and_attr;
}

Status AtomicMemoryAssigner::GetInputCleanMemInfos(const NodePtr &node, std::set<CleanMemInfo> &clean_mem_infos) const {
  GE_ASSERT_NOTNULL(node);
  const auto op_desc = node->GetOpDescBarePtr();
  GE_ASSERT_NOTNULL(op_desc);
  std::vector<int64_t> atomic_input_index;
  (void)ge::AttrUtils::GetListInt(op_desc, ATOMIC_ATTR_INPUT_INDEX, atomic_input_index);
  if (atomic_input_index.empty()) {
    return SUCCESS;
  }
  const auto input_offsets = op_desc->GetInputOffset();
  GE_ASSERT_TRUE(input_offsets.size() >= atomic_input_index.size(),
                 "node %s input_offsets.size[%zu] < atomic_input_index.size[%zu]", node->GetNamePtr(),
                 input_offsets.size(), atomic_input_index.size());
  if ((atomic_input_index.size() == 1U) && (atomic_input_index.at(0) == kAllInputAddrIsAtomic)) {
    atomic_input_index.clear();
    for (int64_t i = 0; static_cast<size_t>(i) < input_offsets.size(); ++i) {
      atomic_input_index.emplace_back(i);
    }
  }
  GE_ASSERT_TRUE(AtomicMemoryAssigner::CheckInputIsSupportAtomic(node.get()));

  for (const auto index : atomic_input_index) {
    GE_ASSERT_TRUE(static_cast<size_t>(index) < input_offsets.size(),
                   "node %s atomic_input_index[%lld] >= input_offsets.size[%zu]", node->GetNamePtr(), index,
                   input_offsets.size());
    const auto tensor_desc = op_desc->MutableInputDesc(index);
    GE_ASSERT_NOTNULL(tensor_desc);
    int64_t get_size = 0;
    (void)TensorUtils::GetSize(*tensor_desc, get_size);
    if (get_size <= 0) {
      GELOGI("[AtomicClean]node: %s(%s), input index: %lld get size: %lld, no need clean", node->GetNamePtr(),
             node->GetTypePtr(), index, get_size);
      continue;
    }
    int64_t aligned_size = get_size;
    ge::AlignMemOffset(aligned_size);
    uint32_t mem_type = RT_MEMORY_HBM;
    GE_ASSERT_SUCCESS(GetMemType(node.get(), kIn, index, mem_type),
                      "node %s get output memory type failed, index: %lld", node->GetNamePtr(), index);
    // fe只对输出和workspace设置数据类型和初始值列表
    CleanMemInfo clean_mem_info;
    clean_mem_info.offset = input_offsets.at(index);
    clean_mem_info.size = aligned_size;
    clean_mem_info.memory_type = mem_type;
    clean_mem_infos.insert(clean_mem_info);
    GELOGI("[AtomicClean]input need clean, node: %s(%s), input index: %lld, get_size: %lld, clean_mem_info: %s",
           node->GetNamePtr(), node->GetTypePtr(), index, get_size, clean_mem_info.ToStr().c_str());
  }
  return SUCCESS;
}

Status AtomicMemoryAssigner::GetOutputCleanMemInfos(const NodePtr &node, AtomicNodeCleanTypeVals &type_vals,
                                                    std::set<CleanMemInfo> &clean_mem_infos) const {
  GE_ASSERT_NOTNULL(node);
  const auto op_desc = node->GetOpDescBarePtr();
  GE_ASSERT_NOTNULL(op_desc);

  std::vector<int64_t> atomic_output_index;
  (void)ge::AttrUtils::GetListInt(node->GetOpDesc(), ATOMIC_ATTR_OUTPUT_INDEX, atomic_output_index);
  if (atomic_output_index.empty()) {
    return SUCCESS;
  }
  const auto output_offsets = op_desc->GetOutputOffset();
  GE_ASSERT_TRUE(output_offsets.size() >= atomic_output_index.size(),
                 "node %s output_offsets.size[%zu] < atomic_output_index.size[%zu]", node->GetNamePtr(),
                 output_offsets.size(), atomic_output_index.size());

  for (const auto index : atomic_output_index) {
    GE_ASSERT_TRUE(static_cast<size_t>(index) < output_offsets.size(),
                   "node %s atomic_output_index[%lld] >= output_offsets.size[%zu]", node->GetNamePtr(), index,
                   output_offsets.size());
    const auto tensor_desc = op_desc->MutableOutputDesc(index);
    GE_ASSERT_NOTNULL(tensor_desc);
    CleanMemInfo clean_mem_info;
    // 获取要清零的初始值和数据类型
    GE_ASSERT_SUCCESS(type_vals.GetNextAttr(clean_mem_info.type_val), "atomic_node: %s(%s), output index: %lld",
                      node->GetNamePtr(), node->GetTypePtr(), index);
    // 获取内存大小
    int64_t get_size = 0;
    (void)TensorUtils::GetSize(*tensor_desc, get_size);
    if (get_size <= 0) {
      GELOGI("[AtomicClean]node: %s(%s), output index: %lld get size: %lld, no need clean", node->GetNamePtr(),
             node->GetTypePtr(), index, get_size);
      continue;
    }

    // 如果不是零拷贝内存，做512字节对齐
    int64_t aligned_size = get_size;
    const bool is_zero_copy = IsZeroCopyOut(op_desc, index);
    if (!is_zero_copy) {
      ge::AlignMemOffset(aligned_size);
    }

    // 获取内存类型
    uint32_t mem_type = RT_MEMORY_HBM;
    GE_ASSERT_SUCCESS(GetMemType(node.get(), kOut, index, mem_type),
                      "node %s get output memory type failed, index: %lld", node->GetNamePtr(), index);

    clean_mem_info.offset = output_offsets.at(index);
    clean_mem_info.size = aligned_size;
    clean_mem_info.memory_type = mem_type;
    clean_mem_info.is_zero_copy = is_zero_copy;
    clean_mem_infos.insert(clean_mem_info);
    GELOGI(
        "[AtomicClean]output need clean, node: %s(%s), output index: %lld, is_zero_copy: %d, get_size: %lld, "
        "clean_mem_info: %s",
        node->GetNamePtr(), node->GetTypePtr(), index, is_zero_copy, get_size, clean_mem_info.ToStr().c_str());
  }
  return SUCCESS;
}

Status AtomicMemoryAssigner::GetWorkspaceCleanMemInfos(const NodePtr &node, AtomicNodeCleanTypeVals &type_vals,
                                                       std::set<CleanMemInfo> &clean_mem_infos) const {
  GE_ASSERT_NOTNULL(node);
  const auto op_desc = node->GetOpDescBarePtr();
  GE_ASSERT_NOTNULL(op_desc);
  const auto sub_node_to_workspace_info =
      op_desc->TryGetExtAttr(EXT_ATTR_ATOMIC_WORKSPACE_INFO, std::map<std::string, std::map<int64_t, int64_t>>{});
  if (sub_node_to_workspace_info.empty()) {
    return SUCCESS;
  }
  // 融合算子的清零workspace内存分配，没有将地址写到atomic node中
  bool is_fusion_node = false;
  (void)ge::AttrUtils::GetBool(op_desc, ATOMIC_ATTR_IS_FUSION_NODE, is_fusion_node);
  if (is_fusion_node) {
    GELOGI("[AtomicClean]fusion node: %s(%s)", node->GetNamePtr(), node->GetTypePtr());
    return GetFusionWorkspaceCleanMemInfos(node, clean_mem_infos);
  }
  const auto workspace_offsets = op_desc->GetWorkspace();
  if (workspace_offsets.empty()) {
    GELOGI("[AtomicClean]workspace_offsets empty, node: %s(%s)", node->GetNamePtr(), node->GetTypePtr());
    return SUCCESS;
  }

  const auto workspace_size = node->GetOpDescBarePtr()->GetWorkspaceBytes();
  GE_ASSERT_TRUE(workspace_offsets.size() == workspace_size.size(),
                 "node %s workspace_offsets.size[%zu] != workspace_size.size[%zu]", workspace_offsets.size(),
                 workspace_size.size());

  std::vector<int64_t> tvm_workspace_types;
  const bool has_tvm_workspace_mem_type_attr =
      ge::AttrUtils::GetListInt(op_desc, TVM_ATTR_NAME_WORKSPACE_TYPE, tvm_workspace_types);

  std::vector<int64_t> workspace_type_list;
  const bool has_workspace_type_list_attr =
      ge::AttrUtils::GetListInt(op_desc, ATTR_NAME_WORKSPACE_TYPE_LIST, workspace_type_list);
  for (const auto &sub_node_iter : sub_node_to_workspace_info) {
    for (const auto &index_size_pair : sub_node_iter.second) {
      const auto index = static_cast<size_t>(index_size_pair.first);
      GE_ASSERT_TRUE(index < workspace_offsets.size());
      if (has_tvm_workspace_mem_type_attr && (index < tvm_workspace_types.size())) {
        // 这两种类型不分配内存
        if ((tvm_workspace_types.at(index) == RT_MEM_TYPE_L1) || (tvm_workspace_types.at(index) == kRtMemoryUB)) {
          GELOGI("[AtomicClean]tvm_workspace_types[%lld]: %lld, not assign memory, node: %s(%s), clean_mem_info: %s",
                 index, tvm_workspace_types.at(index), node->GetNamePtr(), node->GetTypePtr());
          continue;
        }
      }
      CleanMemInfo clean_mem_info;
      // 获取要清零的初始值和数据类型
      GE_ASSERT_SUCCESS(type_vals.GetNextAttr(clean_mem_info.type_val), "atomic_node: %s(%s), output index: %lld",
                        node->GetNamePtr(), node->GetTypePtr(), index);
      // ascend c算子的workspace 默认-1，要跳过
      if (workspace_size.at(index) < 0) {
        GELOGI("[AtomicClean]workspace_size[%lld]: %lld < 0, node: %s(%s), clean_mem_info: %s", index,
               workspace_size.at(index), node->GetNamePtr(), node->GetTypePtr());
        continue;
      }
      // 获取内存类型
      uint32_t mem_type = RT_MEMORY_HBM;
      if (has_workspace_type_list_attr && (index < workspace_type_list.size())) {
        mem_type = workspace_type_list.at(index) == RT_MEMORY_P2P_DDR ? RT_MEMORY_P2P_DDR : RT_MEMORY_HBM;
      }
      int64_t align_size = workspace_size.at(index);
      ge::AlignMemOffset(align_size);
      clean_mem_info.offset = workspace_offsets.at(index);
      clean_mem_info.size = align_size;
      clean_mem_info.memory_type = mem_type;
      clean_mem_infos.insert(clean_mem_info);
      GELOGI("[AtomicClean]workspace need clean, node: %s(%s), index: %lld, clean_mem_info: %s", node->GetNamePtr(),
             node->GetTypePtr(), index, clean_mem_info.ToStr().c_str());
    }
  }
  return SUCCESS;
}

Status AtomicMemoryAssigner::GetFusionWorkspaceCleanMemInfos(const NodePtr &node,
                                                             std::set<CleanMemInfo> &clean_mem_infos) const {
  const auto sub_node_to_workspace_info = node->GetOpDesc()->TryGetExtAttr(
      EXT_ATTR_ATOMIC_WORKSPACE_INFO, std::map<std::string, std::map<int64_t, int64_t>>{});
  const auto sub_node_to_offset = node->GetOpDesc()->TryGetExtAttr(EXT_ATTR_ATOMIC_WORKSPACE_OFFSET,
                                                                   std::map<std::string, std::map<int64_t, int64_t>>{});
  for (const auto &sub_node_iter : sub_node_to_workspace_info) {
    if (sub_node_iter.second.empty()) {
      continue;
    }
    for (const auto &index_size : sub_node_iter.second) {
      const auto &iter = sub_node_to_offset.find(sub_node_iter.first);
      GE_ASSERT_TRUE(iter != sub_node_to_offset.end());
      const auto &index_offset = iter->second.find(index_size.first);
      GE_ASSERT_TRUE(index_offset != iter->second.end());
      CleanMemInfo mem_info;
      mem_info.offset = index_offset->second;
      int64_t mem_align_size = index_size.second;
      ge::AlignMemOffset(mem_align_size);
      mem_info.size = mem_align_size;
      clean_mem_infos.insert(mem_info);
    }
  }
  return SUCCESS;
}

Status AtomicMemoryAssigner::GetMemType(const Node *const node, const IOType &io_type, const uint32_t index,
                                        uint32_t &mem_type) const {
  GE_ASSERT_NOTNULL(node);
  if (block_mem_assigner_ == nullptr) {
    std::vector<int64_t> mem_type_list;
    std::string mem_type_str;
    if (io_type == IOType::kIn) {
      mem_type_str = ATTR_NAME_INPUT_MEM_TYPE_LIST;
    } else if (io_type == IOType::kOut) {
      mem_type_str = ATTR_NAME_OUTPUT_MEM_TYPE_LIST;
    }
    (void)ge::AttrUtils::GetListInt(node->GetOpDesc(), mem_type_str, mem_type_list);
    if (index < mem_type_list.size()) {
      mem_type = mem_type_list.at(index);
    }
    return SUCCESS;
  }
  NodeIndexIO node_index_io{node, index, io_type};
  const auto &anchor_str = node_index_io.ToString();
  const auto symbol_anchor_iter = block_mem_assigner_->anchor_to_symbol_.find(anchor_str);
  GE_ASSERT_TRUE(symbol_anchor_iter != block_mem_assigner_->anchor_to_symbol_.end(), "cannot find symbol by anchor %s",
                 anchor_str.c_str());

  const auto &anchor_iter = block_mem_assigner_->symbol_to_anchors_.find(symbol_anchor_iter->second);
  GE_ASSERT_TRUE(anchor_iter != block_mem_assigner_->symbol_to_anchors_.end(), "cannot find anchor by symbol %s",
                 symbol_anchor_iter->second.c_str());

  int64_t type = RT_MEMORY_HBM;
  block_mem_assigner_->GetSymbolMemType(anchor_iter->second, type);
  mem_type = static_cast<uint32_t>(type);
  return SUCCESS;
}

Status AtomicMemoryAssigner::FilterAtomicNodes(
    std::map<std::string, std::map<NodePtr, std::vector<NodePtr>>> &atomic_nodes) {
  GE_CHECK_NOTNULL(compute_graph_);
  for (const auto &node : compute_graph_->GetAllNodes()) {
    if (!NodeUtils::IsLikeAtomicClean(node)) {
      continue;
    }
    std::map<std::string, std::vector<NodePtr>> tmp_normal_atomic_nodes;
    const auto &out_control_anchor = node->GetOutControlAnchor();
    GE_CHECK_NOTNULL(out_control_anchor);
    for (const auto peer_in_control_anchor : out_control_anchor->GetPeerInControlAnchorsPtr()) {
      GE_ASSERT_NOTNULL(peer_in_control_anchor);
      auto peer_in_node = peer_in_control_anchor->GetOwnerNode();
      GE_ASSERT_NOTNULL(peer_in_node);
      auto peer_in_node_desc = peer_in_node->GetOpDescBarePtr();
      GE_ASSERT_NOTNULL(peer_in_node_desc);
      bool is_atomic_node = false;
      // If GetBool fail, is_atomic_node is false.
      (void)ge::AttrUtils::GetBool(peer_in_node_desc, ATOMIC_ATTR_IS_ATOMIC_NODE, is_atomic_node);
      if (!is_atomic_node) {
        continue;
      }
      if (!CheckAtomicNodeIsSupportRef(peer_in_node)) {
        REPORT_INNER_ERR_MSG("E19999", "Op:%s check atomic node is support ref failed",
                             peer_in_node_desc->GetName().c_str());
        GELOGE(FAILED, "[Check][Attr]Op:%s check atomic node is support ref failed",
               peer_in_node_desc->GetName().c_str());
        return ge::PARAM_INVALID;
      }

      const auto &batch_label = GetBatchLabel(peer_in_node_desc);
      tmp_normal_atomic_nodes[batch_label].emplace_back(peer_in_node);
    }

    for (auto &it_atomic_node : tmp_normal_atomic_nodes) {
      if (!it_atomic_node.second.empty()) {
        atomic_nodes[it_atomic_node.first][node] = it_atomic_node.second;
      }
    }
  }
  return SUCCESS;
}

bool AtomicMemoryAssigner::CheckAtomicNodeIsSupportRef(const NodePtr &node) const {
  GE_ASSERT_NOTNULL(node);
  GE_ASSERT_NOTNULL(node->GetOpDesc());
  const auto op_desc = node->GetOpDesc();
  std::vector<int64_t> atomic_output_index;
  (void)ge::AttrUtils::GetListInt(op_desc, ATOMIC_ATTR_OUTPUT_INDEX, atomic_output_index);
  if (atomic_output_index.size() > op_desc->GetOutputsSize()) {
    REPORT_INNER_ERR_MSG("E19999", "op[%s]: The size [%zu] of atomic output index is greater than output size[%zu].",
                         op_desc->GetName().c_str(), atomic_output_index.size(), op_desc->GetOutputsSize());
    GELOGE(FAILED, "op[%s]: The size [%zu] of atomic output index is greater than output size[%zu].",
           op_desc->GetName().c_str(), atomic_output_index.size(), op_desc->GetOutputsSize());
    return false;
  }
  int32_t reuse_in_index;
  for (size_t i = 0; i < atomic_output_index.size(); ++i) {
    const auto out_anchor = node->GetOutDataAnchor(i);
    if (GraphUtils::IsRefFromInput(out_anchor, reuse_in_index)) {
      REPORT_INNER_ERR_MSG("E19999", "op[%s] output index[%zu] is both atomic and reference, not support now.",
                           op_desc->GetName().c_str(), i);
      GELOGE(FAILED, "[Check][Attr]op[%s] output index[%zu] is both atomic and reference, not support now.",
             op_desc->GetName().c_str(), i);
      return false;
    }
  }
  return true;
}

Status AtomicMemoryAssigner::AssignAtomicOutputAndWorkspaceMemory(
    const ge::NodePtr &node, std::map<int64_t, std::vector<int64_t>> &mem_type_to_offset_end,
    std::map<int64_t, std::vector<int64_t>> &mem_type_to_real_atomic_sizes) {
  auto node_op_desc = node->GetOpDesc();
  GE_CHECK_NOTNULL(node_op_desc);
  // Assign atomic node output memory
  Status ret = AssignAtomicOutputMemory(node, mem_type_to_offset_end, mem_type_to_real_atomic_sizes);
  if (ret != SUCCESS) {
    GELOGE(ret, "[Assign][Memory:Output:Atomic]Failed for node:%s.", node_op_desc->GetName().c_str());
    return ret;
  }

  // Check and assign atomic node workspace memory
  auto atomic_workspace_info =
      node_op_desc->TryGetExtAttr(EXT_ATTR_ATOMIC_WORKSPACE_INFO, std::map<std::string, std::map<int64_t, int64_t>>{});
  if (!atomic_workspace_info.empty()) {
    bool is_fusion_node = false;
    // If GetBool fail, is_fusion_node is false.
    (void)ge::AttrUtils::GetBool(node_op_desc, ATOMIC_ATTR_IS_FUSION_NODE, is_fusion_node);

    if (is_fusion_node) {
      // Assign fusion atomic node workspace memory
      ret = AssignFusionAtomicWorkspaceMemory(node_op_desc, atomic_workspace_info, mem_type_to_offset_end,
                                              mem_type_to_real_atomic_sizes);
    } else {
      // Assign single ordinary atomic node workspace memory, not include fusion node
      ret = AssignOrdinaryAtomicWorkspaceMemory(node_op_desc, atomic_workspace_info, mem_type_to_offset_end,
                                                mem_type_to_real_atomic_sizes);
    }
    if (ret != SUCCESS) {
      GELOGE(ret, "[Assign][Memory:Atomic:Workspace]fail for node:%s.", node_op_desc->GetName().c_str());
      return ret;
    }
  } else {
    GELOGW("Current atomic node %s does not have attr ATOMIC_WORKSPACE_INFO.", node->GetName().c_str());
  }

  return SUCCESS;
}

bool AtomicMemoryAssigner::CheckInputIsSupportAtomic(const ge::Node *node) {
  for (const auto in_data_anchor : node->GetAllInDataAnchorsPtr()) {
    auto peer_out_data_anchor = in_data_anchor->GetPeerOutAnchor();
    if (peer_out_data_anchor == nullptr) {
      continue;
    }
    auto peer_op_desc = peer_out_data_anchor->GetOwnerNodeBarePtr()->GetOpDescBarePtr();
    if (peer_op_desc == nullptr) {
      continue;
    }
    const auto type = peer_op_desc->GetType();
    if (OpTypeUtils::IsConstNode(type) || OpTypeUtils::IsVarLikeNode(type) ||
        (peer_op_desc->GetType() == AIPP_DATA_TYPE)) {
      REPORT_INNER_ERR_MSG("E19999",
                           "node(type:%s, name:%s) link to atomic node(name:%s), "
                           "this situation not supported now",
                           peer_op_desc->GetType().c_str(), peer_op_desc->GetName().c_str(), node->GetName().c_str());
      GELOGE(ge::FAILED,
             "[Check][Link]node(type:%s, name:%s) link to atomic node(name:%s), "
             "this situation not supported now",
             peer_op_desc->GetType().c_str(), peer_op_desc->GetName().c_str(), node->GetName().c_str());
      return false;
    }
  }
  return true;
}

// 更新父节点的offset
Status AtomicMemoryAssigner::UpdateParentNodeOutputOffset(const ge::NodePtr &node, int64_t output_index,
                                                          int64_t offset) const {
  const auto *anchors = FindSymbolAnchors(block_mem_assigner_, node, output_index);
  if (anchors == nullptr) {
    return SUCCESS;
  }
  for (const auto &anchor : *anchors) {
    auto op_desc = anchor.node_ptr_->GetOpDescBarePtr();
    if ((anchor.io_type_ != kOut) || (op_desc == nullptr) || op_desc->GetSubgraphInstanceNames().empty()) {
      continue;
    }
    auto output_offsets = op_desc->GetOutputOffset();
    if (output_offsets.size() > anchor.index_) {
      output_offsets[anchor.index_] = offset;
      op_desc->SetOutputOffset(output_offsets);
      GELOGI("parent node %s(%s) output[%u] offset is updated to %lld, from node %s out_index %lld",
             op_desc->GetNamePtr(), op_desc->GetTypePtr(), anchor.index_, offset, node->GetNamePtr(), output_index);
    }
  }
  return SUCCESS;
}

const std::list<NodeIndexIO> *FindSymbolAnchors(const BlockMemAssignerPtr &mem_assigner, const ge::NodePtr &node,
                                                int64_t output_index) {
  if (mem_assigner == nullptr) {
    return nullptr;
  }
  NodeIndexIO node_index_io{node, output_index, kOut};
  const auto &anchor_str = node_index_io.ToString();
  const auto symbol_anchor_iter = mem_assigner->anchor_to_symbol_.find(anchor_str);
  if (symbol_anchor_iter == mem_assigner->anchor_to_symbol_.end()) {
    GELOGW("cannot find symbol by anchor %s", anchor_str.c_str());
    return nullptr;
  }
  const auto &anchor_iter = mem_assigner->symbol_to_anchors_.find(symbol_anchor_iter->second);
  if (anchor_iter == mem_assigner->symbol_to_anchors_.end()) {
    return nullptr;
  }
  return &(anchor_iter->second);
}

Status AtomicMemoryAssigner::AssignAtomicOutputMemory(
    const ge::NodePtr &node, std::map<int64_t, std::vector<int64_t>> &mem_type_to_offset_end,
    std::map<int64_t, std::vector<int64_t>> &mem_type_to_real_atomic_sizes) {
  auto op_desc = node->GetOpDesc();
  GE_ASSERT_NOTNULL(op_desc);
  GELOGD("Begin to assign atomic output memory, node = %s.", op_desc->GetNamePtr());

  std::vector<int64_t> atomic_output_index;
  // If GetListInt fail, atomic_output_index is empty.
  (void)ge::AttrUtils::GetListInt(op_desc, ATOMIC_ATTR_OUTPUT_INDEX, atomic_output_index);

  // Check atomic output
  std::vector<int64_t> output_list = op_desc->GetOutputOffset();
  const auto out_num = op_desc->GetAllOutputsDescPtr().size();
  while (output_list.size() < out_num) {
    output_list.emplace_back(kInvalidOffset);
  }
  if (atomic_output_index.size() > output_list.size()) {
    std::string error = "Op:" + FmtToStr(node->GetName()) + "'s size:" + FmtToStr(atomic_output_index.size()) +
                        " of atomic_output_index is more than the size:" + FmtToStr(output_list.size()) +
                        " of output_list";
    GE_ERRORLOG_AND_ERRORMSG(FAILED, error.c_str());
    return ge::FAILED;
  }
  auto output_list_size = static_cast<int64_t>(output_list.size());
  for (auto &output_index : atomic_output_index) {
    if (output_index >= output_list_size) {
      std::string error = "Op:" + FmtToStr(node->GetName()) + "'s atomic_output index:" + FmtToStr(output_index) +
                          " is more than the size:" + FmtToStr(output_list_size) + " of output_list.";
      GE_ERRORLOG_AND_ERRORMSG(ge::PARAM_INVALID, error.c_str());
      return ge::PARAM_INVALID;
    }

    // If the input of the cascade op needs to clear the atomic addr, there is no need to clear it separately here
    bool is_assigned_mem = false;
    if (GetMemoryAssignmentStatus(node, output_index, is_assigned_mem) != SUCCESS) {
      GELOGE(ge::FAILED, "[Get][MemoryAssignmentStatus]fail for node %s, out_index:%" PRId64 "",
             node->GetName().c_str(), output_index);
      return ge::FAILED;
    }

    // If you have already assigned an atomic address, skip it, and you don't need to reassign it.
    if (is_assigned_mem) {
      continue;
    }

    auto output_desc = op_desc->GetAllOutputsDescPtr().at(output_index);
    GE_CHECK_NOTNULL(output_desc);
    int64_t size = 0;
    if (ge::TensorUtils::GetSize(*output_desc, size) != SUCCESS) {
      GELOGI("Tensor has no size");
    }
    int64_t memory_type = RT_MEMORY_HBM;
    if (block_mem_assigner_ != nullptr) {
      NodeIndexIO node_index_io(node.get(), output_index, kOut);
      const auto &symbol_to_anchors = block_mem_assigner_->symbol_to_anchors_;
      const auto &anchors_to_symbol = block_mem_assigner_->anchor_to_symbol_;
      const auto symbol_iter = anchors_to_symbol.find(node_index_io.ToString());
      if (symbol_iter != anchors_to_symbol.end()) {
        const auto &anchor_iter = symbol_to_anchors.find(symbol_iter->second);
        if (anchor_iter != symbol_to_anchors.end()) {
          BlockMemAssigner::GetSymbolMemType(anchor_iter->second, memory_type);
        }
      }
    }

    auto iter = memory_offset_.find(memory_type);
    GE_ASSERT_TRUE(iter != memory_offset_.end(),
                   "InnerData memory_offset_ does not have type[HBM], not expected, "
                   "graph_id:%u, graph_name:%s",
                   compute_graph_->GetGraphID(), compute_graph_->GetName().c_str());
    output_list[output_index] = iter->second.mem_offset_;
    const auto &batch_label = GetBatchLabel(op_desc.get());
    iter->second.mem_offset_ += size;
    AlignMemOffset(MEM_ALIGN_SIZE, memory_type);
    mem_type_to_offset_end[memory_type].emplace_back(iter->second.mem_offset_);
    mem_type_to_real_atomic_sizes[memory_type].emplace_back(size);
    iter->second.theory_min_ += (iter->second.mem_offset_ - output_list[output_index]);
    GELOGI("[IMAS]Atomic output : Set %s name[%s] optype[%s] output[%" PRId64 "] offset to [%zu] stream_id[%" PRId64
           "] memtype[%u] "
           "size[%" PRId64 "] real_size[%" PRId64 "] batch[%s].",
           GraphNameId(compute_graph_.get()).c_str(), op_desc->GetName().substr(0, kMaxLogLen).c_str(),
           node->GetType().c_str(), output_index, output_list[output_index], op_desc->GetStreamId(), RT_MEMORY_HBM,
           (iter->second.mem_offset_ - output_list[output_index]), size, batch_label.c_str());
    GE_ASSERT_SUCCESS(UpdateParentNodeOutputOffset(node, output_index, output_list[output_index]));
    CANN_PROFILING_REPORT_STATIC_OP_MEM_INFO(compute_graph_, op_desc, size, kMinLifeTime, kMaxLifeTime);
  }

  op_desc->SetOutputOffset(output_list);

  return ge::SUCCESS;
}

Status AtomicMemoryAssigner::AssignOrdinaryAtomicWorkspaceMemory(
    const ge::OpDescPtr &op_desc, std::map<std::string, std::map<int64_t, int64_t>> &workspace_info,
    std::map<int64_t, std::vector<int64_t>> &mem_type_to_offset_end,
    std::map<int64_t, std::vector<int64_t>> &mem_type_to_real_atomic_sizes) {
  GELOGI("Begin to reassign normal atomic memory, node = %s.", op_desc->GetName().c_str());
  auto mem_type_iter = memory_offset_.find(RT_MEMORY_HBM);
  if (mem_type_iter == memory_offset_.end()) {
    REPORT_INNER_ERR_MSG("E19999",
                         "InnerData memory_offset_ does not have type[HBM], not expected, "
                         "graph_id:%u, graph_name:%s",
                         compute_graph_->GetGraphID(), compute_graph_->GetName().c_str());
    GELOGE(FAILED,
           "[Check][InnerData]memory_offset_ does not have memory type[HBM]"
           "graph_id:%u, graph_name:%s",
           compute_graph_->GetGraphID(), compute_graph_->GetName().c_str());
    return FAILED;
  }
  std::vector<int64_t> workspace_vector = op_desc->GetWorkspace();

  for (auto iter = workspace_info.begin(); iter != workspace_info.end(); ++iter) {
    if (op_desc->GetName() != iter->first) {
      std::string error = "The node name" + FmtToStr(op_desc->GetName()) + " and the node name" +
                          FmtToStr(iter->first) + " in workspace info are inconsistent.";
      GE_ERRORLOG_AND_ERRORMSG(ge::PARAM_INVALID, error.c_str());
      return ge::PARAM_INVALID;
    }

    if (iter->second.empty()) {
      continue;
    }

    for (auto &info_iter : iter->second) {
      auto workspace_index = static_cast<uint64_t>(info_iter.first);
      auto workspace_size = info_iter.second;
      if (workspace_index >= workspace_vector.size()) {
        std::string error = "The workspace index:" + FmtToStr(workspace_index) +
                            " is more than the size:" + FmtToStr(workspace_vector.size()) +
                            " of workspace vector in op:" + op_desc->GetName().c_str();
        GE_ERRORLOG_AND_ERRORMSG(ge::PARAM_INVALID, error.c_str());
        return ge::PARAM_INVALID;
      }

      workspace_vector[workspace_index] = mem_type_iter->second.mem_offset_;
      const auto &batch_label = GetBatchLabel(op_desc.get());
      size_t tmp_mem_offset = mem_type_iter->second.mem_offset_;
      mem_type_iter->second.mem_offset_ += workspace_size;
      AlignMemOffset(MEM_ALIGN_SIZE, RT_MEMORY_HBM);
      mem_type_to_offset_end[RT_MEMORY_HBM].emplace_back(mem_type_iter->second.mem_offset_);
      mem_type_to_real_atomic_sizes[RT_MEMORY_HBM].emplace_back(mem_type_iter->second.mem_offset_ - tmp_mem_offset);
      mem_type_iter->second.theory_min_ += (mem_type_iter->second.mem_offset_ - tmp_mem_offset);
      GELOGI("[IMAS]Atomic ordinary workspace : Set %s name[%s] optype[%s] workspace[%" PRIu64
             "] offset to [%zu] stream_id[%" PRId64
             "] "
             "memtype[%u] size[%" PRId64 "] real_size[%" PRId64 "] batch[%s].",
             GraphNameId(compute_graph_.get()).c_str(), op_desc->GetName().substr(0, kMaxLogLen).c_str(),
             op_desc->GetType().c_str(), workspace_index, mem_type_iter->second.mem_offset_, op_desc->GetStreamId(),
             RT_MEMORY_HBM, (mem_type_iter->second.mem_offset_ - tmp_mem_offset), workspace_size, batch_label.c_str());
      CANN_PROFILING_REPORT_STATIC_OP_MEM_INFO(compute_graph_, op_desc, workspace_size, kMinLifeTime, kMaxLifeTime);
    }
  }
  op_desc->SetWorkspace(workspace_vector);

  return SUCCESS;
}

Status AtomicMemoryAssigner::AssignFusionAtomicWorkspaceMemory(
    const ge::OpDescPtr &op_desc, std::map<std::string, std::map<int64_t, int64_t>> &workspace_info,
    std::map<int64_t, std::vector<int64_t>> &mem_type_to_offset_end,
    std::map<int64_t, std::vector<int64_t>> &mem_type_to_real_atomic_sizes) {
  GELOGI("[AtomicClean]Begin to reassign fusion atomic memory, node = %s.", op_desc->GetName().c_str());
  auto mem_type_iter = memory_offset_.find(RT_MEMORY_HBM);
  if (mem_type_iter == memory_offset_.end()) {
    REPORT_INNER_ERR_MSG("E19999",
                         "InnerData memory_offset_ does not have type[HBM], not expected, "
                         "graph_id:%u, graph_name:%s",
                         compute_graph_->GetGraphID(), compute_graph_->GetName().c_str());
    GELOGE(FAILED,
           "[Check][InnerData]memory_offset_ does not have memory type[HBM]"
           "graph_id:%u, graph_name:%s",
           compute_graph_->GetGraphID(), compute_graph_->GetName().c_str());
    return FAILED;
  }
  std::map<std::string, std::map<int64_t, int64_t>> sub_node_workspace_offset;

  for (auto &iter : workspace_info) {
    if (iter.second.empty()) {
      continue;
    }

    std::map<int64_t, int64_t> index_offset;
    for (auto &info_iter : iter.second) {
      auto workspace_index = static_cast<uint64_t>(info_iter.first);
      auto workspace_size = info_iter.second;

      size_t workspace_offset = mem_type_iter->second.mem_offset_;
      const auto &batch_label = GetBatchLabel(op_desc.get());
      GELOGI("[AtomicClean][IMAS]Atomic fusion workspace : Set %s name[%s] optype[%s] workspace[%" PRIu64
             "] offset to [%zu]"
             " stream_id[%" PRId64 "] memtype[%u] ssize[%" PRId64 "] real_size[%" PRId64 "] batch[%s].",
             GraphNameId(compute_graph_.get()).c_str(), op_desc->GetName().substr(0, kMaxLogLen).c_str(),
             op_desc->GetType().c_str(), workspace_index, mem_type_iter->second.mem_offset_, op_desc->GetStreamId(),
             RT_MEMORY_HBM, workspace_size, workspace_size, batch_label.c_str());
      CANN_PROFILING_REPORT_STATIC_OP_MEM_INFO(compute_graph_, op_desc, workspace_size, kMinLifeTime, kMaxLifeTime);
      size_t tmp_mem_offset = mem_type_iter->second.mem_offset_;
      mem_type_iter->second.mem_offset_ += workspace_size;
      AlignMemOffset(MEM_ALIGN_SIZE, RT_MEMORY_HBM);
      mem_type_to_offset_end[RT_MEMORY_HBM].emplace_back(mem_type_iter->second.mem_offset_);
      index_offset.insert(std::make_pair(workspace_index, workspace_offset));
      mem_type_to_real_atomic_sizes[RT_MEMORY_HBM].emplace_back(mem_type_iter->second.mem_offset_ - tmp_mem_offset);
      mem_type_iter->second.theory_min_ += (mem_type_iter->second.mem_offset_ - tmp_mem_offset);
    }
    sub_node_workspace_offset.insert(std::make_pair(iter.first, index_offset));
  }
  if (!(op_desc->SetExtAttr(EXT_ATTR_ATOMIC_WORKSPACE_OFFSET, sub_node_workspace_offset))) {
    REPORT_INNER_ERR_MSG("E19999", "Set Attr:%s fail for node:%s", EXT_ATTR_ATOMIC_WORKSPACE_OFFSET.c_str(),
                         op_desc->GetName().c_str());
    GELOGE(FAILED, "[Set][Attr:%s]fail for node:%s.", EXT_ATTR_ATOMIC_WORKSPACE_OFFSET.c_str(),
           op_desc->GetName().c_str());
    return FAILED;
  }

  return SUCCESS;
}

void AtomicMemoryAssigner::AlignMemOffset(int64_t mem_align_size, int64_t memory_type) {
  if (mem_align_size <= 0) {
    return;
  }
  auto iter = memory_offset_.find(memory_type);
  if (iter == memory_offset_.end()) {
    GELOGW("Memory offset don't have memory type[%" PRId64 "].", memory_type);
    return;
  }
  iter->second.mem_offset_ = (iter->second.mem_offset_ + mem_align_size - 1) / mem_align_size * mem_align_size;
}

// 算子的输入不能调用这个接口，因为这些属性只有输出和workspace才设置
ge::Status AtomicNodeCleanTypeVals::GetNextAttr(CleanDataTypeValue &type_value) {
  if (data_type_index_ >= data_types_.size()) {
    type_value.data_type = static_cast<int32_t>(ge::DT_FLOAT);
    type_value.float_val = 0.0;
    return ge::SUCCESS;
  }
  type_value.data_type = data_types_[data_type_index_];
  data_type_index_++;
  if (IsFloatType(static_cast<ge::DataType>(type_value.data_type))) {
    GE_ASSERT_TRUE(float_val_index_ < float_vals_.size(), "float_val_index[%zu] >= float_vals.size[%zu], %s",
                   float_val_index_, float_vals_.size(), ToStr().c_str());
    type_value.float_val = float_vals_.at(float_val_index_);
    float_val_index_++;
  } else {
    GE_ASSERT_TRUE(int_val_index_ < int_vals_.size(), "int_val_index_[%zu] >= int_vals.size[%zu], %s", int_val_index_,
                   int_vals_.size(), ToStr().c_str());
    type_value.int_val = int_vals_.at(int_val_index_);
    int_val_index_++;
  }
  return ge::SUCCESS;
}

std::string AtomicNodeCleanTypeVals::ToStr() const {
  std::stringstream ss;
  std::vector<int64_t> atomic_output_index;
  (void)ge::AttrUtils::GetListInt(node_->GetOpDesc(), ge::ATOMIC_ATTR_OUTPUT_INDEX, atomic_output_index);
  ss << "atomic node[" << node_->GetName() << "(" << node_->GetType() << ")] data type list"
     << ge::ToString(data_types_) << ", int value list" << ge::ToString(int_vals_) << ", float value list"
     << ge::ToString(float_vals_) << ", atomic_output_index" << ge::ToString(atomic_output_index);
  return ss.str();
}

ge::Status AtomicNodeCleanTypeVals::Init(const ge::Node *node) {
  GE_ASSERT_NOTNULL(node);
  GE_ASSERT_NOTNULL(node->GetOpDescBarePtr());
  node_ = node;

  data_types_ = GetAtomicDataTypeList(node);
  if (data_types_.empty()) {
    return ge::SUCCESS;
  }
  int_vals_ = GetAtomicIntValList(node);
  float_vals_ = GetAtomicFloatValList(node);

  // 严格校验属性
  GE_ASSERT_TRUE(data_types_.size() == (int_vals_.size() + float_vals_.size()),
                 "data type list size[%zu] is not equal to val list size[int:%zu, float:%zu], %s", ToStr().c_str());
  if (IsLogEnable(GE_MODULE_NAME, DLOG_INFO)) {
    GELOGI("[AtomicClean] %s", ToStr().c_str());
  }
  return ge::SUCCESS;
}

Status AtomicMemoryAssigner::ReAssign() {
  return ReAssignAtomicMemory();
}

std::map<int64_t, int64_t> AtomicMemoryAssigner::GetSplitOffsetSize() const {
  std::map<int64_t, int64_t> offset_to_size;
  if (graph_mem_splitter_ != nullptr) {
    const auto &sub_mem_infos = graph_mem_splitter_->GetSubMemInfo();
    for (const auto &sub_mem : sub_mem_infos) {
      offset_to_size[sub_mem.mem_offset_base] = sub_mem.mem_size;
    }
  }
  return offset_to_size;
}

ge::Status AtomicMemoryAssigner::AppendAddrSizeToMemSetOp(const NodePtr &memset_node,
                                                          const MemsetNodeAddrAndAttr &addr_type) const {
  const auto &memset_op_desc = memset_node->GetOpDesc();
  std::vector<int64_t> workspace_vector = memset_op_desc->GetWorkspace();
  std::vector<int64_t> workspace_byte_vector = memset_op_desc->GetWorkspaceBytes();
  workspace_vector.insert(workspace_vector.cend(), addr_type.offsets.cbegin(), addr_type.offsets.cend());
  workspace_byte_vector.insert(workspace_byte_vector.cend(), addr_type.sizes.cbegin(), addr_type.sizes.cend());
  memset_op_desc->SetWorkspace(workspace_vector);
  memset_op_desc->SetWorkspaceBytes(workspace_byte_vector);

  std::vector<int64_t> mem_type_list;
  if (ge::AttrUtils::GetListInt(memset_op_desc, ATTR_NAME_WORKSPACE_TYPE_LIST, mem_type_list) ||
      (!addr_type.memory_types.empty())) {
    mem_type_list.insert(mem_type_list.cend(), addr_type.memory_types.cbegin(), addr_type.memory_types.cend());
    GE_ASSERT_TRUE(ge::AttrUtils::SetListInt(memset_op_desc, ATTR_NAME_WORKSPACE_TYPE_LIST, mem_type_list),
                   "[Set][Attr:%s]fail for op_name:%s", ATTR_NAME_WORKSPACE_TYPE_LIST.c_str(),
                   memset_node->GetNamePtr());
  }

  std::vector<int64_t> mem_start_vector;
  (void)ge::AttrUtils::GetListInt(memset_op_desc, ATTR_NAME_AUTOMIC_ADD_START, mem_start_vector);
  mem_start_vector.insert(mem_start_vector.cend(), addr_type.offsets.cbegin(), addr_type.offsets.cend());
  GE_ASSERT_TRUE(ge::AttrUtils::SetListInt(memset_op_desc, ATTR_NAME_AUTOMIC_ADD_START, mem_start_vector),
                 "[Set][Attr:%s]fail for op_name:%s", ATTR_NAME_AUTOMIC_ADD_START.c_str(),
                 memset_op_desc->GetName().c_str());

  std::vector<int64_t> mem_size_vector;
  (void)ge::AttrUtils::GetListInt(memset_op_desc, ATTR_NAME_ATOMIC_MEMSET_SIZES, mem_size_vector);
  mem_size_vector.insert(mem_size_vector.cend(), addr_type.sizes.cbegin(), addr_type.sizes.cend());
  GE_ASSERT_TRUE(ge::AttrUtils::SetListInt(memset_op_desc, ATTR_NAME_ATOMIC_MEMSET_SIZES, mem_size_vector),
                 "[Set][Attr:%s]fail for op_name:%s", ATTR_NAME_ATOMIC_MEMSET_SIZES.c_str(),
                 memset_op_desc->GetName().c_str());
  std::vector<int32_t> data_type_list = GetMemsetDataTypeList(memset_node);
  GE_ASSERT_TRUE(data_type_list.empty() || data_type_list.size() == mem_size_vector.size(),
                 "[Check][ListSize] failed, data type size[%zu] of memset node[%s] should be equal to"
                 " mem_size_vector size[%zu]",
                 data_type_list.size(), memset_node->GetName().c_str(), mem_size_vector.size());
  // compatible for atomic_addr_clean
  mem_size_vector.clear();
  (void)ge::AttrUtils::GetListInt(memset_op_desc, ATTR_NAME_AUTOMIC_ADD_MEM_SIZE, mem_size_vector);
  mem_size_vector.insert(mem_size_vector.cend(), addr_type.sizes.cbegin(), addr_type.sizes.cend());
  GE_ASSERT_TRUE(ge::AttrUtils::SetListInt(memset_op_desc, ATTR_NAME_AUTOMIC_ADD_MEM_SIZE, mem_size_vector),
                 "[Set][Attr:%s]fail for op_name:%s", ATTR_NAME_AUTOMIC_ADD_MEM_SIZE.c_str(),
                 memset_op_desc->GetName().c_str());
  GELOGI(
      "[AtomicClean]Append mem size and start to memset node[%s, mem_size_vector size = %zu,"
      " mem_start_vector size = %zu], data_type_list size = %zu, workspace_vector size = %zu,"
      " workspace_byte_vector size = %zu",
      memset_node->GetName().c_str(), mem_size_vector.size(), mem_start_vector.size(), data_type_list.size(),
      workspace_vector.size(), workspace_byte_vector.size());
  return SUCCESS;
}

ge::Status AtomicMemoryAssigner::AppendAttrsToMemSetOp(const NodePtr &memset_node,
                                                       const MemsetNodeAddrAndAttr &addr_type) const {
  GE_ASSERT_NOTNULL(memset_node);
  GE_ASSERT_NOTNULL(memset_node->GetOpDesc());
  const auto &memset_op = memset_node->GetOpDesc();
  if (!addr_type.data_type_list.empty()) {
    GE_ASSERT_TRUE(ge::AttrUtils::SetListInt(memset_op, ge::ATTR_NAME_ATOMIC_MEMSET_DTYPES, addr_type.data_type_list),
                   "[Set][Attr:%s] failed for memset_op[%s]", ge::ATTR_NAME_ATOMIC_MEMSET_DTYPES.c_str(),
                   memset_op->GetName().c_str());
  }
  if (!addr_type.int_list.empty()) {
    GE_ASSERT_TRUE(ge::AttrUtils::SetListInt(memset_op, ge::ATTR_NAME_ATOMIC_MEMSET_VALUES_INT, addr_type.int_list),
                   "[Set][Attr:%s] failed for memset_op[%s], atomic_node[%s]",
                   ge::ATTR_NAME_ATOMIC_MEMSET_VALUES_INT.c_str(), memset_op->GetName().c_str());
  }
  if (!addr_type.float_list.empty()) {
    GE_ASSERT_TRUE(
        ge::AttrUtils::SetListFloat(memset_op, ge::ATTR_NAME_ATOMIC_MEMSET_VALUES_FLOAT, addr_type.float_list),
        "[Set][Attr:%s] failed for memset_op[%s], atomic_node[%s]", ge::ATTR_NAME_ATOMIC_MEMSET_VALUES_FLOAT.c_str(),
        memset_op->GetName().c_str());
  }
  if (!IsLogEnable(GE_MODULE_NAME, DLOG_INFO)) {
    return SUCCESS;
  }
  std::stringstream mem_starts_ss;
  for (auto mem_start : addr_type.offsets) {
    mem_starts_ss << mem_start << " ";
  }
  std::stringstream mem_sizes_ss;
  for (auto mem_size : addr_type.sizes) {
    mem_sizes_ss << mem_size << " ";
  }
  GELOGI(
      "[AtomicClean][IMAS]AppendAttrsToMemSetOp : Set %s atomic_node name[%s] optype[%s] workspace[0] offset to [%s]"
      " streamid[%" PRId64 "] size[%s]",
      GraphNameId(compute_graph_.get()).c_str(), memset_node->GetName().substr(0, kMaxLogLen).c_str(),
      memset_node->GetType().c_str(), mem_starts_ss.str().c_str(), memset_node->GetOpDesc()->GetStreamId(),
      mem_sizes_ss.str().c_str());
  return SUCCESS;
}

Status AtomicMemoryAssigner::GetMemoryAssignmentStatus(const ge::NodePtr &node, int64_t output_index,
                                                       bool &is_mem_assigned) const {
  if (static_cast<size_t>(output_index) >= node->GetAllOutDataAnchorsSize()) {
    std::string error = "Op:" + FmtToStr(node->GetName()) + "'s output index:" + FmtToStr(output_index) +
                        " is more than the size:" + FmtToStr(node->GetAllOutDataAnchors().size()) +
                        " of node's AllOutDataAnchors.";
    GE_ERRORLOG_AND_ERRORMSG(ge::PARAM_INVALID, error.c_str());
    return ge::PARAM_INVALID;
  }
  auto out_data_anchor = node->GetAllOutDataAnchors().at(output_index);
  GE_CHECK_NOTNULL(out_data_anchor);
  const auto input_anchors = out_data_anchor->GetPeerInDataAnchorsPtr();
  for (auto const input_anchor : input_anchors) {
    auto output_node = input_anchor->GetOwnerNodeBarePtr();
    GE_CHECK_NOTNULL(output_node->GetOpDesc());
    const auto continous_input = MemLayoutConflictUtil::IsContinuousInput(output_node);
    if (!continous_input) {
      continue;
    }
    /// Get input atomic attr of peer output op, if atomic_input_index[0] = -1, indicates that the atomic address
    /// has been assigned
    std::vector<int64_t> atomic_input_index;
    (void)ge::AttrUtils::GetListInt(output_node->GetOpDesc(), ATOMIC_ATTR_INPUT_INDEX, atomic_input_index);
    if (!atomic_input_index.empty() && (atomic_input_index[0] == kAllInputAddrIsAtomic)) {
      GELOGI(
          "[AtomicClean]node %s(%s) atomic output[%lld] peer is continuous input node,"
          " don't assign continuous atomic memory for it.",
          node->GetNamePtr(), node->GetTypePtr(), output_index);
      is_mem_assigned = true;
      return SUCCESS;
    }
  }
  if (IsZeroCopyOut(node->GetOpDescBarePtr(), output_index)) {
    GELOGI("[AtomicClean]node %s(%s) atomic output[%lld] is zero copy, don't assign continuous atomic memory for it.",
           node->GetNamePtr(), node->GetTypePtr(), output_index);
    is_mem_assigned = true;
  }
  return SUCCESS;
}

}  // namespace ge
