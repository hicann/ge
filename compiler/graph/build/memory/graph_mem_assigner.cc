/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph/build/memory/graph_mem_assigner.h"
#include <cinttypes>
#include <cstring>
#include <set>
#include "common/math/ge_math_util.h"
#include "common/plugin/ge_make_unique_util.h"
#include "graph/debug/ge_util.h"
#include "base/err_msg.h"
#include "framework/common/debug/log.h"
#include "graph/build/memory/hybrid_mem_assigner.h"
#include "graph/build/memory/var_mem_assign_util.h"
#include "graph/build/memory/block_mem_assigner.h"
#include "graph/build/memory/checker/special_node_checker.h"
#include "common/omg_util/omg_util.h"
#include "common/checker.h"
#include "utils/extern_math_util.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/ge_attr_value.h"
#include "graph/manager/graph_var_manager.h"
#include "graph/utils/tensor_utils.h"
#include "graph/utils/type_utils.h"
#include "graph/build/memory/buffer_pool_mem_assigner.h"
#include "graph/optimize/mem_layout_conflict_optimize/mem_layout_conflict_util.h"
#include "checker/atomic_clean_checker.h"
#include "graph/types.h"
#include "graph/utils/op_type_utils.h"
#include "graph/unfold/graph_unfolder.h"
#include "graph/ge_context.h"
#include "common/platform_info_util/platform_info_util.h"
#include "common/compile_profiling/ge_call_wrapper.h"
#include "graph/normal_graph/compute_graph_impl.h"
#include "runtime/subscriber/global_profiler.h"

namespace {
const int32_t kPrevNextDistanceNum = 2;
const int64_t kInvalidStream = -1;
const uint32_t kSelfNodeMask = 0x10000U;
constexpr uint32_t kExtraDepth = 5U;  // 由于添加了subgraph这种虚拟block，所以深度会增加
// One state per bit cannot be repeated
struct ContinuousType {
  static const uint32_t kTypeInput = 1U;
  static const uint32_t kTypeInputNoPadding = 2U;
  static const uint32_t kTypeOutput = 4U;
  static const uint32_t kTypeOutputNoPadding = 8U;
};

int64_t GetSymbolOutputOffset(const ge::AnchorToSymbol &anchor_to_symbol, const ge::SymbolToAnchors &symbol_to_anchors,
                              const ge::NodePtr &node, const uint32_t i) {
  ge::NodeIndexIO cur_node_index_io(node, i, ge::kOut);
  auto iter1 = anchor_to_symbol.find(cur_node_index_io.ToString());
  if (iter1 == anchor_to_symbol.end()) {
    return ge::kInvalidOffset;
  }
  auto out_symbol = iter1->second;
  auto iter2 = symbol_to_anchors.find(out_symbol);
  if (iter2 == symbol_to_anchors.end()) {
    return ge::kInvalidOffset;
  }
  for (const auto &node_index_io : iter2->second) {
    if (node_index_io.node_->GetType() == ge::NETOUTPUT && node_index_io.io_type_ == ge::kIn) {
      if ((node->GetOpDesc() == nullptr) || (node_index_io.node_->GetOpDesc() == nullptr)) {
        return ge::kInvalidOffset;
      }
      std::vector<int64_t> output_list = node->GetOpDesc()->GetOutputOffset();
      std::vector<int64_t> netoutput_input_offsets = node_index_io.node_->GetOpDesc()->GetInputOffset();
      if (node_index_io.index_ >= netoutput_input_offsets.size()) {
        return ge::kInvalidOffset;
      }
      GELOGI("Node %s %uth output offset is %" PRId64 ", netoutput %s input offset is %" PRId64 ".",
             node->GetName().c_str(), i, output_list[i], iter2->first.c_str(),
             netoutput_input_offsets.at(node_index_io.index_));
      return netoutput_input_offsets.at(node_index_io.index_);
    }
  }
  return ge::kInvalidOffset;
}

ge::Status GetOffsetFromPeerOutputList(const ge::NodePtr &node, int32_t input_index, int64_t &offset, bool &found) {
  found = false;
  const auto in_data_anchor_list = node->GetAllInDataAnchors();
  if ((input_index < 0) || (static_cast<size_t>(input_index) >= in_data_anchor_list.size())) {
    return ge::SUCCESS;
  }
  const auto &in_data_anchor = in_data_anchor_list.at(input_index);
  GE_CHECK_NOTNULL(in_data_anchor);
  const auto peer_out_anchor = in_data_anchor->GetPeerOutAnchor();
  GE_CHECK_NOTNULL(peer_out_anchor);
  auto peer_node = peer_out_anchor->GetOwnerNode();
  GE_CHECK_NOTNULL(peer_node);
  auto peer_op_desc = peer_node->GetOpDesc();
  GE_CHECK_NOTNULL(peer_op_desc);

  int32_t reuse_in_index = -1;
  // 输入有ref或者是data或atomic，认为冲突
  if (peer_node->GetType() == ge::DATA) {
    GELOGE(ge::FAILED, "Conflict: node %s's input[%d] node %s's type is %s", node->GetName().c_str(), input_index,
           peer_node->GetName().c_str(), peer_node->GetType().c_str());
    return ge::FAILED;
  }

  bool is_atomic_node = false;
  (void)ge::AttrUtils::GetBool(peer_op_desc, ge::ATOMIC_ATTR_IS_ATOMIC_NODE, is_atomic_node);
  if (ge::GraphUtils::IsRefFromInput(peer_out_anchor, reuse_in_index) || is_atomic_node) {
    GELOGE(ge::FAILED, "Conflict: node %s's input[%d] node %s is atomic node:%d or set reuse input %d",
           node->GetName().c_str(), input_index, peer_node->GetName().c_str(), is_atomic_node, reuse_in_index);
    return ge::FAILED;
  }

  std::vector<int64_t> offset_list;
  if (!ge::AttrUtils::GetListInt(peer_op_desc, ge::ATTR_NAME_OUTPUT_OFFSET_LIST_FOR_CONTINUOUS, offset_list) ||
      offset_list.empty()) {
    return ge::SUCCESS;
  }
  int32_t peer_out_idx = peer_out_anchor->GetIdx();
  if (peer_out_idx >= static_cast<int32_t>(offset_list.size())) {
    return ge::SUCCESS;
  }
  offset = offset_list.at(peer_out_idx);
  found = true;
  return ge::SUCCESS;
}

ge::Status GetOffsetFromPeerInputList(const ge::NodePtr &node, int32_t output_index, int64_t &offset, bool &found) {
  found = false;
  const auto out_data_anchor = node->GetOutDataAnchor(output_index);
  GE_CHECK_NOTNULL(out_data_anchor);
  for (const auto &peer_in_anchor : out_data_anchor->GetPeerInDataAnchorsPtr()) {
    if ((peer_in_anchor == nullptr) || (peer_in_anchor->GetOwnerNodeBarePtr() == nullptr) ||
        (peer_in_anchor->GetOwnerNodeBarePtr()->GetOpDescBarePtr() == nullptr)) {
      continue;
    }

    // 输出是netoutput，认为冲突
    if (peer_in_anchor->GetOwnerNodeBarePtr()->GetType() == ge::NETOUTPUT) {
      GELOGE(ge::FAILED, "Conflict: node %s's output[%d] node %s type is %s", node->GetName().c_str(), output_index,
             peer_in_anchor->GetOwnerNodeBarePtr()->GetName().c_str(),
             peer_in_anchor->GetOwnerNodeBarePtr()->GetType().c_str());
      return ge::FAILED;
    }

    for (const auto &peer_in_node_out_anchor : peer_in_anchor->GetOwnerNodeBarePtr()->GetAllOutDataAnchors()) {
      int32_t reuse_in_index = -1;
      // 输出有ref，认为冲突
      const bool reuse_input = ge::GraphUtils::IsRefFromInput(peer_in_node_out_anchor, reuse_in_index);
      if (reuse_input && (reuse_in_index == peer_in_anchor->GetIdx())) {
        GELOGE(ge::FAILED, "Conflict: node %s's output[%d] node %s is set reuse input %d", node->GetName().c_str(),
               output_index, peer_in_anchor->GetOwnerNodeBarePtr()->GetName().c_str(), reuse_in_index);
        return ge::FAILED;
      }
    }

    std::vector<int64_t> offset_list;
    if (!ge::AttrUtils::GetListInt(peer_in_anchor->GetOwnerNodeBarePtr()->GetOpDescBarePtr(),
                                   ge::ATTR_NAME_INPUT_OFFSET_LIST_FOR_CONTINUOUS, offset_list) ||
        offset_list.empty()) {
      continue;
    }
    int32_t peer_in_idx = peer_in_anchor->GetIdx();
    if (peer_in_idx >= static_cast<int32_t>(offset_list.size())) {
      continue;
    }
    offset = offset_list.at(peer_in_idx);
    found = true;
    return ge::SUCCESS;
  }
  return ge::SUCCESS;
}

ge::Status GetCustomOffset(const ge::NodePtr &node, int32_t input_index, int32_t output_index, int64_t &offset) {
  offset = 0;
  auto op_desc = node->GetOpDesc();
  GE_CHECK_NOTNULL(op_desc);
  bool custom_input_output_offset = false;
  (void)ge::AttrUtils::GetBool(op_desc, ge::ATTR_NAME_CUSTOM_INPUT_OUTPUT_OFFSET, custom_input_output_offset);
  bool output_reuse_input = false;
  (void)ge::AttrUtils::GetBool(op_desc, ge::ATTR_NAME_OUTPUT_REUSE_INPUT, output_reuse_input);
  if (!custom_input_output_offset || !output_reuse_input) {
    return ge::SUCCESS;
  }
  // 只支持单输入单输出
  if ((op_desc->GetInputsSize() != 1U) || (op_desc->GetOutputsSize() != 1U)) {
    GELOGE(ge::FAILED, "Conflict: node %s's input size:%zu out put size:%zu", node->GetName().c_str(),
           op_desc->GetInputsSize(), op_desc->GetOutputsSize());
    return ge::FAILED;
  }

  int64_t input_relative_offset = 0;
  bool has_input_offset_list = false;
  // 输入输出共用一块内存，input节点从offset右侧位置开始写数据
  // |--offset--|--input--|
  // |--------output------|
  GE_ASSERT_SUCCESS(GetOffsetFromPeerOutputList(node, input_index, input_relative_offset, has_input_offset_list));

  int64_t output_relative_offset = 0;
  bool has_output_offset_list = false;
  // 输入输出共用一块内存，自身节点从offset右侧位置开始写数据
  // |--------input--------|
  // |--offset--|--output--|
  GE_ASSERT_SUCCESS(GetOffsetFromPeerInputList(node, output_index, output_relative_offset, has_output_offset_list));

  if (has_input_offset_list && has_output_offset_list) {
    GELOGE(ge::FAILED, "Conflict: node [%s] cannot set both %s on input peer and %s on output peer",
           node->GetName().c_str(), ge::ATTR_NAME_OUTPUT_OFFSET_LIST_FOR_CONTINUOUS.c_str(),
           ge::ATTR_NAME_INPUT_OFFSET_LIST_FOR_CONTINUOUS.c_str());
    return ge::FAILED;
  }

  // 后面统一处理input + custom_offset，因此这里有正值也可能有负值
  offset = output_relative_offset - input_relative_offset;
  GELOGI("Node [%s] GetCustomOffset: input[%d] relative[%lld], output[%d] relative[%lld], offset[%lld]",
         node->GetName().c_str(), input_index, input_relative_offset, output_index, output_relative_offset, offset);
  return ge::SUCCESS;
}

bool isVariableMemoryNode(const ge::NodePtr &node) {
  return (node->GetType() == ge::VARIABLE) || (node->GetType() == ge::CONSTANTOP);
}

static bool CompareLife(const ge::MemReuseInfo &left, const ge::MemReuseInfo &right) {
  if ((left.node != nullptr) && (right.node != nullptr)) {
    const auto left_op_desc = left.node->GetOpDescBarePtr();
    const auto right_op_desc = right.node->GetOpDescBarePtr();
    if ((left_op_desc != nullptr) && (right_op_desc != nullptr)) {
      return (left_op_desc->GetId() < right_op_desc->GetId());
    }
  }
  return false;
}

void SetMemoryReuseInfoToNodeAttr(std::vector<ge::MemReuseInfo> &mem_reuse_info, const uint32_t depth) {
  if (mem_reuse_info.empty()) {
    return;
  }
  // 1. sort by node id
  std::sort(mem_reuse_info.begin(), mem_reuse_info.end(), CompareLife);
  for (size_t index = 0U; index < (mem_reuse_info.size() - 1U); ++index) {
    if ((static_cast<uint32_t>(mem_reuse_info[index].mem_type) & kSelfNodeMask) != 0U) {
      const uint32_t mem_type = static_cast<uint32_t>(mem_reuse_info[index].mem_type) & (~kSelfNodeMask);
      const auto &node = mem_reuse_info[index].node;
      if (node == nullptr) {
        continue;
      }
      const auto &op_desc = node->GetOpDesc();
      if (op_desc == nullptr) {
        continue;
      }

      // Get reuse attr that already have reuse info on the current node
      std::map<std::string, std::vector<ge::MemReuseInfo>> node_to_reuse_info = op_desc->TryGetExtAttr(
          ge::ATTR_NAME_MEMORY_REUSE_INFO, std::map<std::string, std::vector<ge::MemReuseInfo>>{});
      // Get the key of current node and identify output index or workspace index: [memtype][index], eg:output0
      std::string key = (mem_type == static_cast<uint32_t>(ge::MemType::OUTPUT_MEM)) ? "output" : "workspace";
      key.append(std::to_string(mem_reuse_info[index].index));
      node_to_reuse_info[key].insert(node_to_reuse_info[key].cend(), mem_reuse_info.cbegin() + index + 1U,
                                     mem_reuse_info.cend());
      (void)node->GetOpDesc()->SetExtAttr(ge::ATTR_NAME_MEMORY_REUSE_INFO, node_to_reuse_info);
      GELOGD("Level[%u] Set Reuse info for node[%s], id[%" PRId64
             "], key[%s], total mem info size[%zu], saved reuse mem info"
             " size[%zu], detai:",
             depth, node->GetName().c_str(), op_desc->GetId(), key.c_str(), mem_reuse_info.size(),
             node_to_reuse_info[key].size());
      if (!IsLogEnable(GE_MODULE_NAME, DLOG_DEBUG)) {
        continue;
      }
      for (const auto &info : node_to_reuse_info[key]) {
        if ((info.node == nullptr) || (info.node->GetOpDesc() == nullptr)) {
          continue;
        }
        const uint32_t info_mem_type = static_cast<uint32_t>(info.mem_type) & (~kSelfNodeMask);
        GELOGD("Node[%s], memory type[%s], index[%u], id[%" PRId64 "]", info.node->GetName().c_str(),
               (info_mem_type == static_cast<uint32_t>(ge::MemType::OUTPUT_MEM)) ? "output" : "workspace", info.index,
               info.node->GetOpDesc()->GetId());
      }
    }
  }
}

std::string GetMemoryOption() {
  std::string topo_sorting_mode;
  ge::GetContext().GetOption(ge::OPTION_TOPOSORTING_MODE, topo_sorting_mode);
  if (topo_sorting_mode.empty()) {
    topo_sorting_mode = ge::GetContext().GetTrainGraphFlag() ? "0" : "1";
  }
  topo_sorting_mode = ge::GetTopoSortingModeStr(static_cast<ge::TopoSortingMode>(std::atoi(topo_sorting_mode.c_str())));
  std::string memory_optimization_policy;
  ge::GetContext().GetOption(ge::MEMORY_OPTIMIZATION_POLICY, memory_optimization_policy);
  std::string alloc_mode;
  (void)ge::GetContext().GetOption(ge::OPTION_GRAPH_IO_MEM_ALLOC_MODE, alloc_mode);
  const auto &graph_option = ge::GetThreadLocalContext().GetAllGraphOptions();
  const auto in_reuse = (graph_option.find(ge::OPTION_INPUT_REUSE_MEM_INDEXES) != graph_option.end()) ? "1" : "0";
  const auto out_reuse = (graph_option.find(ge::OPTION_OUTPUT_REUSE_MEM_INDEXES) != graph_option.end()) ? "1" : "0";
  const auto smp = ge::VarManager::IsGeUseExtendSizeMemoryFull()
                       ? ge::kDynamicAndStaticExpandable
                       : (ge::VarManager::IsGeUseExtendSizeMemory(true)
                              ? ge::kDynamicExpandable
                              : (ge::VarManager::IsGeUseExtendSizeMemory() ? ge::kExtendSizeType : "0"));
  return "topo_mode[" + topo_sorting_mode + "], " + "mop[" + memory_optimization_policy + "], " + "io_reuse[" +
         in_reuse + ":" + out_reuse + "], alloc_mode[" + alloc_mode + "], smp[" + smp + "]";
}

bool IsDynamicShapeStaticSubGraph(const ge::ComputeGraphPtr &graph) {
  const auto root_graph = ge::GraphUtils::FindRootGraph(graph);
  if (root_graph != nullptr) {
    bool dynamic_shape_partition = false;
    (void)ge::AttrUtils::GetBool(root_graph, ge::ATTR_NAME_DYNAMIC_SHAPE_PARTITIONED, dynamic_shape_partition);
    if (root_graph->GetGraphUnknownFlag() || dynamic_shape_partition) {
      return true;
    }
  }
  return false;
}

}  // namespace
namespace ge {
Status VariableMemoryAssigner::Assign() {
  Status result = ge::VarMemAssignUtil::AssignVarMemory(compute_graph_);
  if (result != ge::SUCCESS) {
    return result;
  }
  return ge::SUCCESS;
}

Status VariableMemoryAssigner::AssignVarAttr2Nodes() {
  Status result = ge::VarMemAssignUtil::AssignVarAttr2Nodes(compute_graph_);
  if (result != ge::SUCCESS) {
    return result;
  }
  return ge::SUCCESS;
}

Status VariableMemoryAssigner::AssignMemory2HasRefAttrNode() {
  Status result = ge::VarMemAssignUtil::AssignMemory2HasRefAttrNode(compute_graph_);
  if (result != ge::SUCCESS) {
    return result;
  }
  return ge::SUCCESS;
}

Status GraphMemoryAssigner::AssignMemory(const bool has_assigned_var_mem) {
  GE_ASSERT_SUCCESS(SpecialNodeChecker::CheckBeforeAssign(compute_graph_), "check attrs failed, graph: %s.",
                    compute_graph_->GetName().c_str());
  ge::HybridMemAssignerPtr mem_assigner(new (std::nothrow) HybridMemAssigner(compute_graph_));
  GE_CHECK_NOTNULL(mem_assigner);
  if (mem_assigner->Assign() != ge::SUCCESS) {
    GELOGE(ge::FAILED, "[Assign][GraphMem]graph_id:%u, graph_name:%s", compute_graph_->GetGraphID(),
           compute_graph_->GetName().c_str());
    return ge::FAILED;
  }

  for (const auto pair : mem_assigner->GetMemOffsets()) {
    const auto it = mem_assigner->GetMemoryStat().find(pair.first);
    const size_t theory_min = (it != mem_assigner->GetMemoryStat().cend()) ? it->second.theory_min_memory_size_ : 0UL;
    MemoryOffset offset(pair.first, pair.second, theory_min);
    memory_offset_.emplace(pair.first, offset);
  }

  // base memtype offset must be exist
  if (mem_assigner->GetMemOffsets().find(RT_MEMORY_HBM) == mem_assigner->GetMemOffsets().cend()) {
    const auto it = mem_assigner->GetMemoryStat().find(RT_MEMORY_HBM);
    const size_t theory_min = (it != mem_assigner->GetMemoryStat().cend()) ? it->second.theory_min_memory_size_ : 0UL;
    MemoryOffset memory_offset(RT_MEMORY_HBM, 0UL, theory_min);
    memory_offset_.emplace(RT_MEMORY_HBM, memory_offset);
  }

  if (mem_assigner->GetMemOffsets().find(RT_MEMORY_P2P_DDR) == mem_assigner->GetMemOffsets().cend()) {
    const auto it = mem_assigner->GetMemoryStat().find(RT_MEMORY_P2P_DDR);
    const size_t theory_min = (it != mem_assigner->GetMemoryStat().cend()) ? it->second.theory_min_memory_size_ : 0UL;
    MemoryOffset p2p_memory_offset(RT_MEMORY_P2P_DDR, 0UL, theory_min);
    memory_offset_.emplace(RT_MEMORY_P2P_DDR, p2p_memory_offset);
  }
  if (!has_assigned_var_mem) {
    GE_ASSERT_SUCCESS(AssignVarMemory(compute_graph_));
  }
  mem_assigner_ = std::move(mem_assigner);

  // 编译时动态shape的静态子图或者纯静态图均做切分处理；
  // 执行时使用切分结果(1.纯静态图存扩展模式打开 2.外部设置fix 3.静态子图， 其它场景不使用该结果)
  GE_CHECK_NOTNULL(mem_assigner_);
  BlockMemAssignerPtr priority_assigner = mem_assigner_->GetPriorityAssinger();
  GE_CHECK_NOTNULL(priority_assigner);
  // 纯静态图扩展模式按实际block大小切分
  const auto split_size = IsDynamicShapeStaticSubGraph(compute_graph_) ? kSubMemoryMaximumSize : 512U;
  graph_mem_splitter_ = MakeShared<GraphMemSplitter>(priority_assigner->GetMemoryBlocks(), split_size);
  GE_CHECK_NOTNULL(graph_mem_splitter_);
  graph_mem_splitter_->Split(mem_assigner_->GetMemOffsets());
  atomic_memory_assigner_ =
      ComGraphMakeUnique<AtomicMemoryAssigner>(compute_graph_, memory_offset_, priority_assigner, graph_mem_splitter_);
  GE_CHECK_NOTNULL(atomic_memory_assigner_);

  return SetMemReuseInfo();
}

ge::Status GraphMemoryAssigner::AssignVarMemory(const ComputeGraphPtr &compute_graph) {
  FuncPerfScope func_perf_scope("GraphMemoryAssigner", __FUNCTION__);
  const auto session_id = compute_graph->GetSessionID();
  const auto session_var_mng = ge::VarManager::Instance(session_id);
  GE_ASSERT_NOTNULL(session_var_mng);
  const int64_t var_size_before_assign = session_var_mng->GetVarMemSize(RT_MEMORY_HBM);
  const auto variable_assigner =
      std::unique_ptr<ge::VariableMemoryAssigner>(new (std::nothrow) ge::VariableMemoryAssigner(compute_graph));
  GE_ASSERT_NOTNULL(variable_assigner);
  GE_ASSERT_SUCCESS(variable_assigner->Assign(), "%s assign var memory failed.", compute_graph->GetName().c_str());
  const int64_t var_size_assign = session_var_mng->GetVarMemSize(RT_MEMORY_HBM) - var_size_before_assign;
  GELOGD("graph %s assign variable size = %" PRId64 "", compute_graph->GetName().c_str(), var_size_assign);
  return SUCCESS;
}

ge::Status GraphMemoryAssigner::AssignVarAttr2Nodes() {
  auto variable_assigner =
      std::unique_ptr<ge::VariableMemoryAssigner>(new (std::nothrow) ge::VariableMemoryAssigner(compute_graph_));
  if (variable_assigner == nullptr) {
    GELOGE(ge::FAILED, "[New][Object:VariableMemoryAssigner]graph_id:%u, graph_name:%s", compute_graph_->GetGraphID(),
           compute_graph_->GetName().c_str());
    REPORT_INNER_ERR_MSG("E19999",
                         "New Object:VariableMemoryAssigner failed, "
                         "graph_id:%u, graph_name:%s",
                         compute_graph_->GetGraphID(), compute_graph_->GetName().c_str());
    return ge::FAILED;
  }
  if (variable_assigner->AssignVarAttr2Nodes() != ge::SUCCESS) {
    return ge::FAILED;
  }
  return ge::SUCCESS;
}

ge::Status GraphMemoryAssigner::AssignMemory2HasRefAttrNode() const {
  auto variable_assigner = MakeUnique<VariableMemoryAssigner>(compute_graph_);
  GE_CHECK_NOTNULL(variable_assigner);
  if (variable_assigner->AssignMemory2HasRefAttrNode() != ge::SUCCESS) {
    return ge::FAILED;
  }
  return ge::SUCCESS;
}

Status GraphMemoryAssigner::SetMemReuseInfo() const {
  GE_CHECK_NOTNULL(mem_assigner_);
  BlockMemAssignerPtr priority_assigner = mem_assigner_->GetPriorityAssinger();
  if (priority_assigner == nullptr) {
    REPORT_INNER_ERR_MSG("E19999", "InnerData priority_assigner nullptr, not expected, graph_id:%u, graph_name:%s",
                         compute_graph_->GetGraphID(), compute_graph_->GetName().c_str());
    GELOGE(FAILED, "[Check][InnerData:priority_assigner]nullptr is invalid, graph_id:%u, graph_name:%s",
           compute_graph_->GetGraphID(), compute_graph_->GetName().c_str());
    return FAILED;
  }

  for (const auto &memory_block : priority_assigner->GetMemoryBlocks()) {
    if ((memory_block == nullptr) || (memory_block->is_zero_copy_) || (memory_block->child_block_)) {
      continue;
    }
    // Get the value, which is the reuse info of current node output or workspace
    const std::vector<MemReuseInfo> parent_mem_reuse_info;
    std::vector<MemReuseInfo> mem_reuse_info;
    RecordSubsequentReuseNodeInfo(memory_block, parent_mem_reuse_info, mem_reuse_info);
  }
  return SUCCESS;
}

void GraphMemoryAssigner::RecordSubsequentReuseNodeInfo(const MemoryBlock *const memory_block,
                                                        const std::vector<MemReuseInfo> &parent_mem_resue_info,
                                                        std::vector<MemReuseInfo> &total_child_mem_resue_info,
                                                        uint32_t depth) const {
  if (depth > (kMaxDepthNum + kExtraDepth)) {
    GELOGW("The system will stop recording subsequent reuse node info, because too many nesting levels(%u)", depth);
    return;
  }

  // add parent nodes
  std::vector<MemReuseInfo> tmp_parent_mem_reuse_info = parent_mem_resue_info;
  std::vector<MemReuseInfo> self_mem_reuse_info;

  /*
   * partitioned_call的输出和对应的子图的输出同block，子图的输出可能会和子图内做复用，得保留。
   * 当前一个节点是partitioned_call的时候，且当前节点是ref的时候，认为是子图的输出节点
   */
  bool last_node_is_subgraph_out = false;
  for (const auto &node_type_index : memory_block->NodeTypeIndexList()) {
    if ((node_type_index.node_ != nullptr) && (node_type_index.node_->GetOpDesc() != nullptr) &&
        (!node_type_index.ref_input_ || last_node_is_subgraph_out)) {
      last_node_is_subgraph_out = node_type_index.is_subgraph_out_;
      // record sub sequent node and reuse info
      MemReuseInfo reuse_info;
      reuse_info.node.reset(const_cast<Node *>(node_type_index.node_), [](Node *) {});
      reuse_info.index = node_type_index.index_;
      reuse_info.mem_type =
          (node_type_index.mem_type_ == OpMemoryType::kOutput) ? MemType::OUTPUT_MEM : MemType::WORKSPACE_MEM;
      GELOGD("Level[%u] node[%s], op memory type[%s], index[%u], id[%" PRId64 "]", depth,
             reuse_info.node->GetName().c_str(), node_type_index.GetMemType().c_str(), node_type_index.index_,
             reuse_info.node->GetOpDesc()->GetId());
      tmp_parent_mem_reuse_info.emplace_back(reuse_info);
      self_mem_reuse_info.emplace_back(reuse_info);
    }
  }
  std::vector<MemReuseInfo> child_mem_reuse_info;
  for (auto &child_block : memory_block->AllChildBlockList()) {
    if (child_block != nullptr) {
      std::vector<MemReuseInfo> tmp_total_mem_reuse_info;
      RecordSubsequentReuseNodeInfo(child_block, tmp_parent_mem_reuse_info, tmp_total_mem_reuse_info, depth + 1U);
      child_mem_reuse_info.insert(child_mem_reuse_info.cend(), tmp_total_mem_reuse_info.cbegin(),
                                  tmp_total_mem_reuse_info.cend());
    }
  }

  // add child nodes to total
  total_child_mem_resue_info.insert(total_child_mem_resue_info.cend(), child_mem_reuse_info.cbegin(),
                                    child_mem_reuse_info.cend());
  // add self nodes to total
  total_child_mem_resue_info.insert(total_child_mem_resue_info.cend(), self_mem_reuse_info.cbegin(),
                                    self_mem_reuse_info.cend());
  // save reuse info to node
  std::vector<MemReuseInfo> mem_reuse_info;
  // 1. parent nodes
  mem_reuse_info.insert(mem_reuse_info.cend(), parent_mem_resue_info.cbegin(), parent_mem_resue_info.cend());
  // 2. self nodes
  for (auto &info : self_mem_reuse_info) {
    info.mem_type = static_cast<MemType>((static_cast<uint32_t>(info.mem_type) | kSelfNodeMask));
  }
  mem_reuse_info.insert(mem_reuse_info.cend(), self_mem_reuse_info.cbegin(), self_mem_reuse_info.cend());
  // 3. child nodes
  mem_reuse_info.insert(mem_reuse_info.cend(), child_mem_reuse_info.cbegin(), child_mem_reuse_info.cend());
  // 4. save to node
  SetMemoryReuseInfoToNodeAttr(mem_reuse_info, depth);
}

ge::Status CalculateTensorRealSizeAndOutSize(const ge::ConstGeTensorDescPtr &output_desc, int64_t dim_index,
                                             int64_t &output_mem_size, int64_t &batch_dim_num, int64_t &out_size) {
  graphStatus graph_status = ge::TensorUtils::GetSize(*output_desc, out_size);
  if (graph_status != GRAPH_SUCCESS) {
    GELOGE(FAILED, "[Get][TensorSize]");
    REPORT_INNER_ERR_MSG("E19999", "Get tensor size failed");
    return FAILED;
  }

  GeShape output_shape = output_desc->GetShape();
  std::vector<int64_t> output_dims = output_shape.GetDims();
  if ((dim_index != 0L) && (dim_index >= static_cast<int64_t>(output_dims.size()))) {
    REPORT_INNER_ERR_MSG("E19999",
                         "Inner param dim_index value:%" PRId64 " invalid, bigger than dim size:%zu in shape:%s",
                         dim_index, output_dims.size(), output_shape.ToString().c_str());
    GELOGE(FAILED, "[Check][Param:dim_index]value:%" PRId64 " invalid, bigger than dim size:%zu in shape:%s", dim_index,
           output_dims.size(), output_shape.ToString().c_str());
    return FAILED;
  }

  for (int64_t index = 0; index < dim_index; index++) {
    FMK_INT64_MULCHECK(batch_dim_num, output_dims[index]);
    batch_dim_num *= output_dims[index];
    output_dims[index] = 1;
  }

  output_shape = GeShape(output_dims);
  Format out_format = output_desc->GetFormat();
  DataType data_type = output_desc->GetDataType();

  graph_status = ge::TensorUtils::CalcTensorMemSize(output_shape, out_format, data_type, output_mem_size);
  if (graph_status != GRAPH_SUCCESS) {
    GELOGE(graph_status, "[Calc][TensorSize]");
    return FAILED;
  }

  if (output_mem_size < 0) {
    REPORT_INNER_ERR_MSG("E19999",
                         "After calculating, tensor memory size:%" PRId64
                         " invalid, less than 0. "
                         "shape:%s, format:%s, dtype:%s, maybe has dynamic shape",
                         output_mem_size, output_shape.ToString().c_str(),
                         TypeUtils::FormatToSerialString(out_format).c_str(),
                         TypeUtils::DataTypeToSerialString(data_type).c_str());
    GELOGE(FAILED,
           "[Check][TensorSize]value:%" PRId64
           " invalid after calc, less than 0. shape:%s, format:%s, dtype:%s, "
           "maybe has dynamic shape",
           output_mem_size, output_shape.ToString().c_str(), TypeUtils::FormatToSerialString(out_format).c_str(),
           TypeUtils::DataTypeToSerialString(data_type).c_str());
    return FAILED;
  }

  return SUCCESS;
}

Status GraphMemoryAssigner::ReAssignMemory(std::map<uint64_t, size_t> &mem_type_to_offset) {
  if (memory_offset_.empty()) {
    REPORT_INNER_ERR_MSG("E19999", "InnerData memory_offset_ empty, not expected, graph_id:%u, graph_name:%s",
                         compute_graph_->GetGraphID(), compute_graph_->GetName().c_str());
    GELOGE(FAILED,
           "[Check][InnerData:memory_offset_]empty is not expected, "
           "graph_id:%u, graph_name:%s",
           compute_graph_->GetGraphID(), compute_graph_->GetName().c_str());
    return ge::FAILED;
  }

  GE_CHK_STATUS_RET(AssignBufferPoolMemory(), "[Assign][BufferPoolMemory] Failed! graph:%s",
                    compute_graph_->GetName().c_str());
  GE_ASSERT_NOTNULL(atomic_memory_assigner_);
  GE_CHK_STATUS_RET(atomic_memory_assigner_->ReAssign(), "[ReAssign][AtomicMemory] Failed! graph:%s",
                    compute_graph_->GetName().c_str());

  size_t total_mem_offset = 0U;
  for (auto pair : memory_offset_) {
    mem_type_to_offset[pair.first] = pair.second.mem_offset_;
    if ((pair.first != RT_MEMORY_HOST) && (pair.first != RT_MEMORY_HOST_SVM)) {
      total_mem_offset += pair.second.mem_offset_;
    }
  }

  if (graph_mem_splitter_ != nullptr) {
    graph_mem_splitter_->AddContinuousMemoryInfo(mem_type_to_offset);
  }

  if (((mem_assigner_ != nullptr) && (mem_assigner_->GetPriorityAssinger() != nullptr) &&
       mem_assigner_->GetPriorityAssinger()->IsMemoryPriorityMode())) {
    return SUCCESS;
  }
  auto session_id = compute_graph_->GetSessionID();
  GE_CHECK_NOTNULL(VarManager::Instance(session_id));
  const auto var_mem_size = static_cast<size_t>(VarManager::Instance(session_id)->GetVarMemSize(RT_MEMORY_HBM));
  const auto max_mem_size = PlatformInfoUtil::GetMemorySize();
  GE_CHK_STATUS_RET(CheckSizeTAddOverflow(total_mem_offset, var_mem_size));
  if (total_mem_offset + var_mem_size > max_mem_size) {
    REPORT_INNER_ERR_MSG("E19999",
                         "Sum of total_mem_offset:%zu and var_mem_size:%zu"
                         " is greater than memory manager malloc max size %zu, "
                         "graph_id:%u, graph_name:%s, reduce your batchsize or scale your model may solve problem",
                         total_mem_offset, var_mem_size, max_mem_size, compute_graph_->GetGraphID(),
                         compute_graph_->GetName().c_str());
    GELOGE(ge::FAILED,
           "[Check] sum of total_mem_offset:%zu and var_mem_size:%zu"
           " is greater than memory manager malloc max size %zu, "
           "graph_id:%u, graph_name:%s, reduce your batchsize or scale your model may solve problem",
           total_mem_offset, var_mem_size, max_mem_size, compute_graph_->GetGraphID(),
           compute_graph_->GetName().c_str());
    size_t zero_copy_mem_size = 0U;
    (void)AssignZeroCopyMemory(mem_type_to_offset, zero_copy_mem_size);
    for (auto iter : mem_type_to_offset) {
      if (mem_assigner_ == nullptr) {
        continue;
      }
      MemoryOffsetMap::const_iterator it = memory_offset_.find(iter.first);
      std::map<uint64_t, MemoryStat>::const_iterator it_memory_stat = mem_assigner_->GetMemoryStat().find(iter.first);
      if (it_memory_stat == mem_assigner_->GetMemoryStat().cend()) {
        continue;
      }
      GEEVENT("[IMAS]AfterAssignMemory : %s memoffset[%zu], memtype[%" PRId64
              "], theory_min[%zu], zero_copy[%zu], "
              "total_size[%zu], no_reuse[%zu], streams[%zu] %s",
              GraphNameId(compute_graph_.get()).c_str(), iter.second, iter.first,
              (it != memory_offset_.cend()) ? it->second.theory_min_ : 0UL, zero_copy_mem_size,
              it_memory_stat->second.total_memory_size_, it_memory_stat->second.theory_no_reuse_memory_size_,
              it_memory_stat->second.stream_count_, GetMemoryOption().c_str());
    }
    return ACL_ERROR_GE_MEMORY_ALLOCATION;
  }
  return SUCCESS;
}

Status GraphMemoryAssigner::AssignZeroCopyMemory(std::map<uint64_t, size_t> &mem_offset, size_t &zero_mem_copy_size) {
  GE_CHECK_NOTNULL(mem_assigner_);
  BlockMemAssignerPtr priority_assigner = mem_assigner_->GetPriorityAssinger();
  if (priority_assigner == nullptr) {
    REPORT_INNER_ERR_MSG("E19999", "InnerData priority_assigner nullptr, not expected, graph_id:%u, graph_name:%s",
                         compute_graph_->GetGraphID(), compute_graph_->GetName().c_str());
    GELOGE(FAILED,
           "[Check][InnerData:priority_assigner]nullptr is invalid, "
           "graph_id:%u, graph_name:%s",
           compute_graph_->GetGraphID(), compute_graph_->GetName().c_str());
    return ge::FAILED;
  }

  size_t mem_offset_tmp = mem_offset[RT_MEMORY_HBM];
  std::vector<std::pair<MemoryBlock *, size_t>> graph_input_blocks;
  // 记录设置offset的block顺序，用于输出内存报告日志
  std::vector<MemoryBlock *> zero_copy_blocks;

  // set offset for zero copy block
  for (auto &memory_block : priority_assigner->GetMemoryBlocks()) {
    if (memory_block == nullptr || memory_block->child_block_ || !memory_block->is_zero_copy_) {
      continue;
    }
    size_t size = 0U;
    if (memory_block->IsGraphInputAndGetSize(compute_graph_, size)) {
      graph_input_blocks.emplace_back(std::make_pair(memory_block, size));
      continue;
    }

    // set offset for non-input-fusion block
    memory_block->Resize();
    GE_ASSERT_SUCCESS(memory_block->SetHeadOffset(mem_offset[RT_MEMORY_HBM]));
    mem_offset[RT_MEMORY_HBM] += memory_block->Size();
    memory_block->SetTailOffset(mem_offset[RT_MEMORY_HBM] - 1);
    zero_copy_blocks.emplace_back(memory_block);
  }

  GELOGI("%s non-input-copy block start %zu, size %zu", compute_graph_->GetName().c_str(), mem_offset_tmp,
         mem_offset[RT_MEMORY_HBM] - mem_offset_tmp);

  // sort graph input blocks by size
  std::sort(graph_input_blocks.begin(), graph_input_blocks.end(),
            [](const std::pair<MemoryBlock *, size_t> &a, const std::pair<MemoryBlock *, size_t> &b) {
              return a.second < b.second;
            });

  // put graph-input-blocks together, so input data can merge H2D copy for model execution
  // set offset for zero copy block of graph-input
  size_t mem_offset_for_input = mem_offset[RT_MEMORY_HBM];
  for (auto &item : graph_input_blocks) {
    auto memory_block = item.first;
    memory_block->Resize();
    GE_ASSERT_SUCCESS(memory_block->SetHeadOffset(mem_offset[RT_MEMORY_HBM]));
    mem_offset[RT_MEMORY_HBM] += memory_block->Size();
    memory_block->SetTailOffset(mem_offset[RT_MEMORY_HBM] - 1);
    zero_copy_blocks.emplace_back(memory_block);
    GELOGI("graph-input-block size %zu, offset %zu, input size: %zu", memory_block->Size(), memory_block->HeadOffset(),
           item.second);
  }
  GELOGI("%s graph-input-block num is %zu, start %zu, size %zu", compute_graph_->GetName().c_str(),
         graph_input_blocks.size(), mem_offset_for_input, mem_offset[RT_MEMORY_HBM] - mem_offset_for_input);

  // set offset for zero copy nodes
  priority_assigner->SetOpMemOffset(zero_copy_blocks);
  zero_mem_copy_size = mem_offset[RT_MEMORY_HBM] - mem_offset_tmp;
  auto iter = memory_offset_.find(RT_MEMORY_HBM);
  if (iter == memory_offset_.end()) {
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
  iter->second.mem_offset_ = mem_offset[RT_MEMORY_HBM];
  iter->second.zero_copy_size_ = zero_mem_copy_size;

  GELOGD("max_mem_offset:%zu, mem_offset:%zu, zero_mem_copy_size:%zu.", mem_offset[RT_MEMORY_HBM], mem_offset_tmp,
         zero_mem_copy_size);
  if (graph_mem_splitter_ != nullptr) {
    graph_mem_splitter_->AddIoMemoryInfo(mem_offset);
  }
  return SUCCESS;
}

uint32_t GetContinuousMemoryType(const NodePtr &node) {
  if (node->GetOpDescBarePtr() == nullptr) {
    return 0;
  };

  bool is_continuous = MemLayoutConflictUtil::IsContinuousInput(node);
  uint32_t continuous_type = 0;
  if (is_continuous) {
    continuous_type |= ContinuousType::kTypeInput;
  } else {
    (void)ge::AttrUtils::GetBool(node->GetOpDescBarePtr(), ATTR_NAME_NOPADDING_CONTINUOUS_INPUT, is_continuous);
    if (is_continuous) {
      bool attr_reuse = false;
      (void)ge::AttrUtils::GetBool(node->GetOpDescBarePtr(), ATTR_NAME_OUTPUT_REUSE_INPUT, attr_reuse);
      if (attr_reuse) {
        continuous_type |= ContinuousType::kTypeInputNoPadding;
      }
    }
  }

  is_continuous = MemLayoutConflictUtil::IsContinuousOutput(node);
  if (is_continuous) {
    continuous_type |= ContinuousType::kTypeOutput;
  } else {
    (void)ge::AttrUtils::GetBool(node->GetOpDescBarePtr(), ATTR_NAME_NOPADDING_CONTINUOUS_OUTPUT, is_continuous);
    if (is_continuous) {
      bool attr_reuse = false;
      (void)ge::AttrUtils::GetBool(node->GetOpDescBarePtr(), ATTR_NAME_OUTPUT_REUSE_INPUT, attr_reuse);
      if (attr_reuse) {
        continuous_type |= ContinuousType::kTypeOutputNoPadding;
      }
    }
  }

  if (continuous_type != 0) {
    GELOGI("[Get][MemType:Continuous]Current node %s, value is %d", node->GetName().c_str(), continuous_type);
  }
  return continuous_type;
}

Status GetMemorySize(const OpDescPtr &op_desc, const ge::ConstGeTensorDescPtr &output_desc, uint32_t continuous_type,
                     int64_t &tensor_size, int64_t &nopadding_size) {
  GE_ASSERT_NOTNULL(op_desc);
  GE_ASSERT_NOTNULL(output_desc);
  tensor_size = 0;
  nopadding_size = 0;
  bool is_nopadding = ((continuous_type & ContinuousType::kTypeInputNoPadding) != 0) ||
                      ((continuous_type & ContinuousType::kTypeOutputNoPadding) != 0);
  if (is_nopadding) {
    int64_t attr_dim_index;
    bool get_attr_dim_flag = ge::AttrUtils::GetInt(op_desc, ATTR_NAME_REUSE_INPUT_ON_DIM_INDEX, attr_dim_index);
    if (!get_attr_dim_flag) {
      REPORT_INNER_ERR_MSG("E19999", "Get Attr:%s failed, op_name:%s", ATTR_NAME_REUSE_INPUT_ON_DIM_INDEX.c_str(),
                           op_desc->GetName().c_str());
      GELOGE(FAILED, "[Get][Attr:%s]fail for op_name:%s", ATTR_NAME_REUSE_INPUT_ON_DIM_INDEX.c_str(),
             op_desc->GetName().c_str());
      return FAILED;
    }

    // Calculate tensor real size of each piece of data and out size of complete data
    int64_t batch_dim_num = 1;
    if (CalculateTensorRealSizeAndOutSize(output_desc, attr_dim_index, nopadding_size, batch_dim_num, tensor_size) !=
        SUCCESS) {
      REPORT_INNER_ERR_MSG("E19999", "CalculateTensorRealSizeAndOutSize failed, attr_dim_index:%" PRId64 ", op_name:%s",
                           attr_dim_index, op_desc->GetName().c_str());
      GELOGE(FAILED, "[Calculate][NopaddingSize]failed for node %s, attr_dim_index:%" PRId64 "",
             op_desc->GetName().c_str(), attr_dim_index);
      return FAILED;
    }
  } else {
    if (ge::TensorUtils::GetSize(*output_desc, tensor_size) != ge::SUCCESS) {
      REPORT_INNER_ERR_MSG("E19999", "Get Tensor Size failed, op_name:%s", op_desc->GetName().c_str());
      GELOGE(FAILED, "[Get][TensorSize]failed in padding case, op_name:%s", op_desc->GetName().c_str());
      return FAILED;
    }
  }
  if ((tensor_size < 0) || (nopadding_size < 0)) {
    REPORT_INNER_ERR_MSG("E19999",
                         "GetMemorySize fail, "
                         "tensor_size:%" PRId64 " or nopadding_size:%" PRId64 " less than 0, invalid, op_name:%s",
                         tensor_size, nopadding_size, op_desc->GetName().c_str());
    GELOGE(FAILED,
           "[Get][MemorySize]tensor_size:%" PRId64 " or nopadding_size:%" PRId64 " less than 0, invalid, op_name:%s",
           tensor_size, nopadding_size, op_desc->GetName().c_str());
    return FAILED;
  }
  return SUCCESS;
}

ge::Status UpdateOffsetsByOffsetListAttr(const NodePtr &node, const int64_t input_offset,
                                         const int64_t origin_inner_offset, bool &has_input_offset_flag) {
  auto op_desc = node->GetOpDesc();
  const auto session_id = node->GetOwnerComputeGraphBarePtr()->GetSessionID();
  auto origin_output_list = op_desc->GetOutputOffset();
  GE_ASSERT_NOTNULL(ge::VarManager::Instance(session_id), "Get var manager failed, session_id=%llu", session_id);
  const auto input_offset_is_var = ge::VarManager::Instance(session_id)->IsVarAddr(input_offset - origin_inner_offset);
  for (const auto &out_data_anchor : node->GetAllOutDataAnchors()) {
    GE_CHECK_NOTNULL(out_data_anchor);
    for (const auto &peer_in_data_anchor : out_data_anchor->GetPeerInDataAnchors()) {
      GE_CHECK_NOTNULL(peer_in_data_anchor);
      GE_CHECK_NOTNULL(peer_in_data_anchor->GetOwnerNodeBarePtr());
      auto peer_op_desc = peer_in_data_anchor->GetOwnerNodeBarePtr()->GetOpDesc();
      GE_CHECK_NOTNULL(peer_op_desc);
      std::vector<int64_t> offset_list = {};
      bool has_offset_list =
          ge::AttrUtils::GetListInt(peer_op_desc, ATTR_NAME_INPUT_OFFSET_LIST_FOR_CONTINUOUS, offset_list);
      if (has_offset_list && !offset_list.empty()) {
        GE_ASSERT_TRUE(peer_in_data_anchor->GetIdx() < static_cast<int32_t>(offset_list.size()),
                       "Peer node:[%s] anchor_index:[%d] is out of range:[%u]", peer_op_desc->GetName().c_str(),
                       peer_in_data_anchor->GetIdx(), offset_list.size());
        GE_ASSERT_TRUE(!AddOverflow(input_offset, offset_list[peer_in_data_anchor->GetIdx()],
                                    origin_output_list[out_data_anchor->GetIdx()]));
        // 如果input_offset是变量内存逻辑地址，将'input_offset+偏移'作为算子输入输出offset，也需要给这个算子设置inner_offset，
        // 否则加载阶段，根据'input_offset+偏移'无法判定是变量内存
        int64_t new_inner_offset = 0;
        if (input_offset_is_var) {
          GE_ASSERT_TRUE(
              !AddOverflow(origin_inner_offset, offset_list[peer_in_data_anchor->GetIdx()], new_inner_offset));
        }
        if (new_inner_offset != 0) {
          GE_ASSERT_TRUE(ge::AttrUtils::SetInt(op_desc->MutableOutputDesc(out_data_anchor->GetIdx()),
                                               ATTR_NAME_INNER_OFFSET, new_inner_offset));
        }
        GELOGI(
            "Node [%s]' input[%d] has _input_offset_list_for_continuous [%lld],"
            " offset is set to %lld, new_inner_offset[%lld].",
            peer_op_desc->GetName().c_str(), peer_in_data_anchor->GetIdx(), offset_list[peer_in_data_anchor->GetIdx()],
            origin_output_list[out_data_anchor->GetIdx()], new_inner_offset);
        has_input_offset_flag = true;
        break;
      }
    }
  }
  op_desc->SetOutputOffset(origin_output_list);
  return ge::SUCCESS;
}

/*
 * NoPadding连续输出节点是输出引用输入的，输入Offset如果变化，必须更新所有输出Offset
 */
ge::Status UpdateNoPaddingContinousOutputOffsets(const NodePtr &node, const int64_t input_offset,
                                                 const int64_t origin_inner_offset) {
  auto op_desc = node->GetOpDesc();
  GE_CHECK_NOTNULL(op_desc);
  const auto continuous_type = GetContinuousMemoryType(node);
  if ((continuous_type & ContinuousType::kTypeOutputNoPadding) == 0) {
    return ge::SUCCESS;
  }
  GE_CHECK_NOTNULL(node->GetOwnerComputeGraphBarePtr());
  const auto session_id = node->GetOwnerComputeGraphBarePtr()->GetSessionID();
  GE_CHECK_NOTNULL(ge::VarManager::Instance(session_id));

  bool has_input_offset_flag = false;
  GE_ASSERT_SUCCESS(UpdateOffsetsByOffsetListAttr(node, input_offset, origin_inner_offset, has_input_offset_flag));
  if (has_input_offset_flag) {
    return ge::SUCCESS;
  }
  auto origin_output_list = op_desc->GetOutputOffset();
  int64_t output_offset = input_offset;
  const bool input_offset_is_var = ge::VarManager::Instance(session_id)->IsVarAddr(input_offset - origin_inner_offset);
  for (const auto &out_data_anchor : node->GetAllOutDataAnchors()) {
    GE_CHECK_NOTNULL(out_data_anchor);
    origin_output_list[out_data_anchor->GetIdx()] = output_offset;

    // 如果input_offset是变量内存逻辑地址，将'input_offset+偏移'作为算子输入输出offset，也需要给这个算子设置inner_offset，
    // 否则加载阶段，根据'input_offset+偏移'无法判定是变量内存
    int64_t new_inner_offset = 0;
    if (input_offset_is_var) {
      GE_ASSERT_TRUE(!AddOverflow(origin_inner_offset, (output_offset - input_offset), new_inner_offset));
    }
    if (new_inner_offset != 0) {
      GE_ASSERT_TRUE(ge::AttrUtils::SetInt(op_desc->MutableOutputDesc(out_data_anchor->GetIdx()),
                                           ATTR_NAME_INNER_OFFSET, new_inner_offset));
    }
    GELOGI("NoPaddingContinuousOutput node [%s]'s output[%d] offset is set to %lld, new_inner_offset[%lld]",
           node->GetNamePtr(), out_data_anchor->GetIdx(), origin_output_list[out_data_anchor->GetIdx()],
           new_inner_offset);
    int64_t tensor_desc_size = 0;
    int64_t nopadding_size = 0;
    GE_ASSERT_SUCCESS(GetMemorySize(op_desc, op_desc->GetOutputDescPtr(out_data_anchor->GetIdx()), continuous_type,
                                    tensor_desc_size, nopadding_size));
    GE_ASSERT_TRUE(!AddOverflow(output_offset, nopadding_size, output_offset));
  }
  op_desc->SetOutputOffset(origin_output_list);
  return ge::SUCCESS;
}

void AlignMemOffset(int64_t &mem_align_size) {
  if (mem_align_size <= 0) {
    return;
  }
  mem_align_size = (mem_align_size + MEM_ALIGN_SIZE - 1) / MEM_ALIGN_SIZE * MEM_ALIGN_SIZE;
}

/// op1 -> node -> op2
/// return true when node is ref from input, and op1 or op2 is reuse input from output
bool GraphMemoryAssigner::IsRefFromInputOpCascade(const NodePtr &node) const {
  std::unordered_set<int32_t> ref_input_index;
  int32_t reuse_in_index = -1;
  for (const auto &out_anchor : node->GetAllOutDataAnchors()) {
    bool reuse_input = GraphUtils::IsRefFromInput(out_anchor, reuse_in_index);
    if (reuse_input) {
      GELOGD("IsRefFromInputOpCascade: cur node:%s:%d is ref", node->GetName().c_str(), reuse_in_index);
      ref_input_index.insert(reuse_in_index);
    }
  }
  bool ref_from_input = !ref_input_index.empty();
  if (!ref_from_input) {
    return false;
  }

  for (const auto &in_anchor : node->GetAllInDataAnchors()) {
    const auto &peer_out_anchor = in_anchor->GetPeerOutAnchor();
    GE_IF_BOOL_EXEC(peer_out_anchor == nullptr, continue);
    if (ref_input_index.count(in_anchor->GetIdx()) == 0) {
      continue;
    }
    auto in_node = peer_out_anchor->GetOwnerNode();
    if (isVariableMemoryNode(in_node)) {
      GELOGD("Reuse variable memory, input node:%s, type:%s.", in_node->GetName().c_str(), in_node->GetType().c_str());
      return false;
    }
    if (GraphUtils::IsRefFromInput(peer_out_anchor, reuse_in_index)) {
      GELOGD("IsRefFromInputOpCascade: in node[%s] is ref, reuse index is:%d", in_node->GetName().c_str(),
             reuse_in_index);
      return true;
    }
  }

  for (const auto &out_anchor : node->GetAllOutDataAnchors()) {
    const auto &peer_in_anchors = out_anchor->GetPeerInDataAnchors();
    for (const auto &peer_in_anchor : peer_in_anchors) {
      auto peer_in_node = peer_in_anchor->GetOwnerNode();
      GE_IF_BOOL_EXEC(peer_in_node == nullptr, continue);
      for (const auto &peer_in_node_out_anchor : peer_in_node->GetAllOutDataAnchors()) {
        if (GraphUtils::IsRefFromInput(peer_in_node_out_anchor, reuse_in_index)) {
          GELOGD("IsRefFromInputOpCascade: out node[%s] is ref, reuse index is:%d",
                 peer_in_node_out_anchor->GetOwnerNode()->GetName().c_str(), reuse_in_index);
          return true;
        }
      }
    }
  }
  return false;
}

/// node:in0(in0 reuse out0) -> peer_node:out0
/// update peer_node's 0th output offset with node's 0th output offset
Status GraphMemoryAssigner::UpdateRefOpOffsetReverse(const NodePtr &node) const {
  std::map<int32_t, int32_t> out2ins;
  GE_CHK_STATUS_RET(TryGetNodeRefIndexes(node, out2ins), "[Get][RefIndexes]fail for node:%s", node->GetName().c_str());
  auto op_desc = node->GetOpDesc();
  GE_CHECK_NOTNULL(op_desc);
  std::vector<int64_t> output_list = op_desc->GetOutputOffset();
  for (const auto &out2in : out2ins) {
    auto reuse_in_anchor = node->GetInDataAnchor(out2in.second);
    GE_CHECK_NOTNULL(reuse_in_anchor);
    auto peer_out_anchor = reuse_in_anchor->GetPeerOutAnchor();
    GE_CHECK_NOTNULL(peer_out_anchor);
    auto peer_node = peer_out_anchor->GetOwnerNode();
    GE_CHECK_NOTNULL(peer_node);
    if (isVariableMemoryNode(peer_node)) {
      GELOGW("Peer node to update is %s, skip it. Node name:%s.", peer_node->GetType().c_str(),
             peer_node->GetName().c_str());
      continue;
    }
    int64_t custom_offset = 0;
    GE_ASSERT_SUCCESS(GetCustomOffset(node, out2in.second, out2in.first, custom_offset));
    // 设置自定义offset时，不需要更新
    if (custom_offset != 0) {
      continue;
    }
    auto peer_op_desc = peer_node->GetOpDesc();
    GE_CHECK_NOTNULL(peer_op_desc);
    std::vector<int64_t> peer_output_list = peer_op_desc->GetOutputOffset();
    if ((peer_out_anchor->GetIdx() >= static_cast<int32_t>(peer_output_list.size())) ||
        (out2in.first >= static_cast<int32_t>(output_list.size()))) {
      GELOGW("out of range, peer_out_anchor:%d, peer_output_list size:%zu, out2in:%d, output_list size:%zu",
             peer_out_anchor->GetIdx(), peer_output_list.size(), out2in.first, output_list.size());
      continue;
    }
    int64_t origin_input_offset = 0;
    const auto origin_input_offsets = op_desc->GetInputOffset();
    if (origin_input_offsets.size() > static_cast<size_t>(out2in.second)) {
      origin_input_offset = origin_input_offsets.at(out2in.second);
      GELOGI("UpdateRefOpOffsetReverse: Node[%s] input index[%d] has origin offset[%" PRId64 "]",
             node->GetName().c_str(), out2in.second, origin_input_offset);
    }
    peer_output_list.at(peer_out_anchor->GetIdx()) = output_list.at(out2in.first) - origin_input_offset;
    peer_op_desc->SetOutputOffset(peer_output_list);
    GELOGI("UpdateRefOpOffsetReverse: Node[%s] output[%d] is set from node[%s] output index[%d] offset[%" PRId64
           " - %" PRId64 "]",
           peer_node->GetName().c_str(), peer_out_anchor->GetIdx(), node->GetName().c_str(), out2in.first,
           output_list.at(out2in.first), origin_input_offset);
  }
  return SUCCESS;
}

Status GraphMemoryAssigner::ReAssignContinuousMemory() {
  // Stored nodes which need assign continuous input memory in `reverse topo order`
  std::vector<NodePtr> nodes_stack;
  std::map<NodePtr, uint32_t> node_2_continuous_type;

  // Traverse nodes
  for (auto &node : compute_graph_->GetAllNodes()) {
    GE_CHECK_NOTNULL(node);
    uint32_t continuous_type;
    std::map<NodePtr, uint32_t>::const_iterator iter = node_2_continuous_type.find(node);
    if (iter == node_2_continuous_type.cend()) {
      continuous_type = GetContinuousMemoryType(node);
      node_2_continuous_type.emplace(node, continuous_type);
    } else {
      continuous_type = iter->second;
    }
    // Assign continuous input memory
    bool continuous_input = ((continuous_type & ContinuousType::kTypeInput) != 0) ||
                            ((continuous_type & ContinuousType::kTypeInputNoPadding) != 0);
    if (IsRefFromInputOpCascade(node)) {
      nodes_stack.push_back(node);
      GELOGD("Ref: Push node:%s to stack", node->GetName().c_str());
    } else if (continuous_input) {
      if (IsAssignContinuousInputMemoryDirectly(node, node_2_continuous_type)) {
        GE_CHK_STATUS_RET(AssignContinuousInputMemory(node, continuous_type),
                          "[Assign][Memory:Continuous:Input]fail for node:%s", node->GetName().c_str());
      } else {
        nodes_stack.push_back(node);
        GELOGD("Continuous: Push node:%s to stack", node->GetName().c_str());
      }
    }
    // Assign continuous output memory
    int64_t memory_type = RT_MEMORY_HBM;
    bool continuous_output = ((continuous_type & ContinuousType::kTypeOutput) != 0) ||
                             ((continuous_type & ContinuousType::kTypeOutputNoPadding) != 0);
    if (continuous_output) {
      GE_CHK_STATUS_RET(GetNodeMemoryType(node, memory_type, "output"), "[Get][MemType]fail for node:%s",
                        node->GetName().c_str());
      GE_CHK_STATUS_RET(AssignContinuousOutputMemory(node, memory_type, continuous_type),
                        "[Assign][Memory:Continuous:Output]fail for node:%s", node->GetName().c_str());
    }
  }
  // Assign continuous input memory in `reverse topo order` which stored before
  while (!nodes_stack.empty()) {
    auto node = nodes_stack.back();
    nodes_stack.pop_back();
    auto iter = node_2_continuous_type.find(node);
    if (iter == node_2_continuous_type.end()) {
      REPORT_INNER_ERR_MSG("E19999", "Get ContinuousType from node_2_continuous_type map failed for node:%s",
                           node->GetName().c_str());
      GELOGE(FAILED, "[Get][ContinuousType] find fail for node:%s", node->GetName().c_str());
      return FAILED;
    }
    if (((iter->second & ContinuousType::kTypeInput) != 0) ||
        ((iter->second & ContinuousType::kTypeInputNoPadding) != 0)) {
      GE_CHK_STATUS_RET(AssignContinuousInputMemory(node, iter->second, true),
                        "[Assign][Memory:Continuous:Input]fail for node:%s.", node->GetName().c_str());
    } else {
      GE_CHK_STATUS_RET(UpdateRefOpOffsetReverse(node), "[Update][Memory:Reference:Output]fail for node:%s",
                        node->GetName().c_str());
    }
  }
  for (auto pair : memory_offset_) {
    GELOGD("[Reassign][Memory:Continuous]At last, memory type = %" PRId64 ", mem offset = %zu", pair.first,
           pair.second.mem_offset_);
  }
  return ge::SUCCESS;
}

Status GraphMemoryAssigner::SetMemOffset(const ge::NodePtr &node, const InDataAnchorPtr &in_data_anchor,
                                         bool reverse_refresh, int64_t &mem_offset) const {
  auto op_desc = node->GetOpDesc();
  GE_CHECK_NOTNULL(op_desc);

  auto peer_out_data_anchor = in_data_anchor->GetPeerOutAnchor();
  GE_CHECK_NOTNULL(peer_out_data_anchor);
  auto peer_op_desc = peer_out_data_anchor->GetOwnerNode()->GetOpDesc();
  std::vector<int64_t> output_list = peer_op_desc->GetOutputOffset();
  if (peer_out_data_anchor->GetIdx() >= static_cast<int32_t>(output_list.size())) {
    std::string error = "peer node:" + FmtToStr(peer_op_desc->GetName()) +
                        " anchor_index:" + FmtToStr(peer_out_data_anchor->GetIdx()) +
                        " is out of range:" + FmtToStr(output_list.size());
    GE_ERRORLOG_AND_ERRORMSG(FAILED, error.c_str());
    return FAILED;
  }

  // when continuous input has been allocated first input is beginning offset
  bool is_allocated_first_input = (in_data_anchor->GetIdx() == 0);
  if (is_allocated_first_input) {
    std::map<int32_t, int32_t> out2ins;
    GE_CHK_STATUS_RET(TryGetNodeRefIndexes(node, out2ins), "[Get][RefIndexes]fail for node: %s",
                      node->GetName().c_str());
    // output is beginning offset, set offset for input; only support this case now
    if ((out2ins.size() == 1) && (out2ins.begin()->second == 0) && (reverse_refresh)) {
      std::vector<int64_t> output_list_this = op_desc->GetOutputOffset();
      if (output_list_this.empty()) {
        REPORT_INNER_ERR_MSG("E19999", "No output offset in node :%s, not expected", node->GetName().c_str());
        GELOGE(FAILED, "[Get][OutputOffset] empty is invalid, node:%s", node->GetName().c_str());
        return FAILED;
      }
      auto peer_output_offset = output_list.at(peer_out_data_anchor->GetIdx());
      output_list.at(peer_out_data_anchor->GetIdx()) = output_list_this.at(out2ins.begin()->first);
      peer_op_desc->SetOutputOffset(output_list);
      GELOGI("[Update][Offset]Node %s out %d ref in %d input node %s, use output offset %" PRId64 " update %" PRId64 "",
             node->GetName().c_str(), out2ins.begin()->first, out2ins.begin()->second, peer_op_desc->GetName().c_str(),
             output_list_this.at(out2ins.begin()->first), peer_output_offset);
    } else {
      GELOGD("Node %s out %d ref in %d input node %s with total ref numbers %zu.", node->GetName().c_str(),
             out2ins.begin()->first, out2ins.begin()->second, peer_op_desc->GetName().c_str(), out2ins.size());
    }
    // first input is beginning offset
    mem_offset = output_list.at(peer_out_data_anchor->GetIdx());
  } else {
    // set offset for input
    output_list.at(peer_out_data_anchor->GetIdx()) = mem_offset;
    peer_op_desc->SetOutputOffset(output_list);
  }

  return SUCCESS;
}

Status GraphMemoryAssigner::AssignContinuousInputMemory(const ge::NodePtr &node, uint32_t continuous_type,
                                                        bool reverse_refresh) {
  int64_t memory_type = RT_MEMORY_HBM;
  GE_CHK_STATUS_RET(GetNodeMemoryType(node, memory_type, "input"), "[Get][MemType]fail for node:%s",
                    node->GetName().c_str());
  GELOGI("[Assign][Memory:Input:Continuous]start for Current node %s", node->GetName().c_str());
  const auto iter = memory_offset_.find(memory_type);
  GE_ASSERT_TRUE(iter != memory_offset_.end(),
                 "find memory offset fail for mem_type:%" PRId64
                 ", "
                 "for node:%s, ",
                 memory_type, node->GetName().c_str());

  GE_CHECK_NOTNULL(node->GetOpDesc());

  int64_t mem_offset = iter->second.mem_offset_;
  int64_t extra_memory_size = 0;

  for (auto &in_data_anchor : node->GetAllInDataAnchors()) {
    GE_IF_BOOL_EXEC(in_data_anchor == nullptr, continue);
    auto peer_out_data_anchor = in_data_anchor->GetPeerOutAnchor();
    GE_IF_BOOL_EXEC(peer_out_data_anchor == nullptr, continue);
    auto peer_op_desc = peer_out_data_anchor->GetOwnerNode()->GetOpDesc();
    GE_IF_BOOL_EXEC(peer_op_desc == nullptr, continue);

    int64_t tensor_desc_size = 0;
    int64_t nopadding_size = 0;
    int64_t real_size = 0;
    std::vector<int64_t> offsets_of_fusion = {};
    std::vector<int64_t> offset_list = {};
    bool lx_fusion = AttrUtils::GetListInt(peer_op_desc, ATTR_NAME_OUTPUT_OFFSET_FOR_BUFFER_FUSION, offsets_of_fusion);
    bool has_offset_list =
        AttrUtils::GetListInt(peer_op_desc, ATTR_NAME_OUTPUT_OFFSET_LIST_FOR_CONTINUOUS, offset_list);
    lx_fusion = lx_fusion && !offsets_of_fusion.empty();
    if (lx_fusion) {
      if (peer_out_data_anchor->GetIdx() >= static_cast<int32_t>(offsets_of_fusion.size())) {
        std::string error = "fusion: peer node:" + FmtToStr(peer_op_desc->GetName()) +
                            " anchor_index:" + FmtToStr(peer_out_data_anchor->GetIdx()) +
                            " is out of range:" + FmtToStr(offsets_of_fusion.size());
        GE_ERRORLOG_AND_ERRORMSG(FAILED, error.c_str());
        return FAILED;
      }
      nopadding_size = offsets_of_fusion[peer_out_data_anchor->GetIdx()];
      tensor_desc_size = nopadding_size;
      GELOGI("node %s(%lld) has %s attr, get size: %lld", peer_op_desc->GetName().substr(0, kMaxLogLen).c_str(),
             peer_op_desc->GetId(), ATTR_NAME_OUTPUT_OFFSET_FOR_BUFFER_FUSION.c_str(), nopadding_size);
    } else if (has_offset_list && !offset_list.empty()) {
      GE_ASSERT_TRUE(peer_out_data_anchor->GetIdx() < static_cast<int32_t>(offset_list.size()),
                     "Peer node:[%s] anchor_index:[%d] is out of range:[%u]", peer_op_desc->GetName().c_str(),
                     peer_out_data_anchor->GetIdx(), offset_list.size());
      const auto this_output_offset = node->GetOpDesc()->GetOutputOffset();
      GE_ASSERT_TRUE(!this_output_offset.empty(), "Node [%s] output offset empty.", node->GetName().c_str());
      int64_t continuous_output_offset = offset_list[peer_out_data_anchor->GetIdx()] + this_output_offset[0];
      std::vector<int64_t> output_list = peer_op_desc->GetOutputOffset();
      output_list.at(peer_out_data_anchor->GetIdx()) = continuous_output_offset;
      peer_op_desc->SetOutputOffset(output_list);
      GELOGI("Node [%s] has _output_offset_list_for_continuous [%d], add to output offset.",
             peer_op_desc->GetName().c_str(), offset_list[peer_out_data_anchor->GetIdx()]);
      GELOGI("[IMAS]Continuous input : Set %s name[%s] optype[%s] output[%d] offset to [%" PRId64 "] stream_id[%" PRId64
             "] "
             "memtype[%" PRId64 "] size[%zu] realsize[%" PRId64 "] nopadding[%d]",
             GraphNameId(compute_graph_.get()).c_str(), peer_op_desc->GetName().substr(0, kMaxLogLen).c_str(),
             peer_op_desc->GetType().c_str(), peer_out_data_anchor->GetIdx(), continuous_output_offset,
             peer_op_desc->GetStreamId(), memory_type, 0U, 0L, true);
      continue;
    } else {
      GE_ASSERT_SUCCESS(GetMemorySize(node->GetOpDesc(), peer_op_desc->GetOutputDescPtr(peer_out_data_anchor->GetIdx()),
                                      continuous_type, tensor_desc_size, nopadding_size));
    }
    GE_ASSERT_SUCCESS(SetMemOffset(node, in_data_anchor, reverse_refresh, mem_offset));

    int64_t align_size = tensor_desc_size;
    bool is_nopadding = ((continuous_type & ContinuousType::kTypeInputNoPadding) != 0) || lx_fusion;
    if (is_nopadding) {
      mem_offset += nopadding_size;
      extra_memory_size += (tensor_desc_size - nopadding_size);
      real_size = nopadding_size;
    } else {
      ge::AlignMemOffset(align_size);
      mem_offset += align_size;
      real_size = tensor_desc_size;
    }
    std::vector<int64_t> output_list = peer_op_desc->GetOutputOffset();
    GELOGI("[IMAS]Continuous input : Set %s name[%s] optype[%s] output[%d] offset to [%" PRId64 "] stream_id[%" PRId64
           "] memtype[%" PRId64
           "] "
           "size[%zu] realsize[%" PRId64 "] nopadding[%d]",
           GraphNameId(compute_graph_.get()).c_str(), peer_op_desc->GetName().substr(0, kMaxLogLen).c_str(),
           peer_op_desc->GetType().c_str(), peer_out_data_anchor->GetIdx(),
           output_list.at(peer_out_data_anchor->GetIdx()), peer_op_desc->GetStreamId(), memory_type, 0UL, real_size,
           is_nopadding);
  }

  mem_offset += extra_memory_size;
  ge::AlignMemOffset(mem_offset);

  return SUCCESS;
}

Status GetFirstInputPeerOutOutputOffset(const ge::NodePtr &node, int64_t &mem_offset) {
  auto in_data_anchor_list = node->GetAllInDataAnchors();
  if (in_data_anchor_list.empty()) {
    REPORT_INNER_ERR_MSG("E19999", "InAnchor list empty in node:%s, not expect", node->GetName().c_str());
    GELOGE(FAILED, "[Get][InAnchor]empty is invalid, node:%s", node->GetName().c_str());
    return FAILED;
  }
  auto peer_out_data_anchor = in_data_anchor_list.at(0)->GetPeerOutAnchor();
  GE_IF_BOOL_EXEC(peer_out_data_anchor == nullptr,
                  REPORT_INNER_ERR_MSG("E19999", "PeerAcnhor is null, not expect for node:%s", node->GetName().c_str());
                  GELOGE(ge::FAILED, "[Check][PeerAnchor]null is invalid, node:%s", node->GetName().c_str());
                  return ge::FAILED);
  auto peer_op_desc = peer_out_data_anchor->GetOwnerNode()->GetOpDesc();
  GE_IF_BOOL_EXEC(peer_op_desc == nullptr,
                  REPORT_INNER_ERR_MSG("E19999", "PeerOpDesc is null, not expect for node:%s", node->GetName().c_str());
                  GELOGE(ge::FAILED, "[Check][PeerOpDesc]null is invalid, node:%s", node->GetName().c_str());
                  return ge::FAILED);
  std::vector<int64_t> in_node_output_offsets = peer_op_desc->GetOutputOffset();
  if (peer_out_data_anchor->GetIdx() >= static_cast<int32_t>(in_node_output_offsets.size())) {
    REPORT_INNER_ERR_MSG("E19999",
                         "PeerAnchorIndex:%d bigger than in_offset size:%" PRIu64 ", judge invalid for node:%s",
                         peer_out_data_anchor->GetIdx(), in_node_output_offsets.size(), node->GetName().c_str());
    GELOGE(FAILED, "[Check][Index:PeerOutDataAnchor]PeerIndex:%d bigger than in_offset size:%" PRIu64 ", node:%s",
           peer_out_data_anchor->GetIdx(), in_node_output_offsets.size(), node->GetName().c_str());
    return FAILED;
  }
  mem_offset = in_node_output_offsets.at(peer_out_data_anchor->GetIdx());
  return SUCCESS;
}

Status GraphMemoryAssigner::AssignContinuousOutputMemory(const ge::NodePtr &node, int64_t memory_type,
                                                         uint32_t continuous_type) const {
  GELOGI("Current node %s needs continuous output.", node->GetName().c_str());
  auto op_desc = node->GetOpDesc();
  GE_CHECK_NOTNULL(op_desc);
  if ((op_desc->GetOutputsSize() > op_desc->GetOutputOffset().size()) || (op_desc->GetOutputOffset().size() == 0)) {
    REPORT_INNER_ERR_MSG("E19999", "Output size:%zu more than output offset size:%zu, invalid in node:%s",
                         op_desc->GetOutputsSize(), op_desc->GetOutputOffset().size(), node->GetName().c_str());
    GELOGE(ge::FAILED, "[Check][InnerData]Output size:%zu more than output offset size:%zu, invalid in node:%s",
           op_desc->GetOutputsSize(), op_desc->GetOutputOffset().size(), node->GetName().c_str());
    return ge::FAILED;
  }

  int64_t mem_offset = 0;
  bool has_input_offset_flag = false;
  bool is_nopadding = ((continuous_type & ContinuousType::kTypeOutputNoPadding) != 0);
  if (is_nopadding) {
    // out tensor memory must be reused input tensor memory
    GE_ASSERT_SUCCESS(GetFirstInputPeerOutOutputOffset(node, mem_offset));

    // 如果通过属性设置了偏移量，不用再计算内存大小，根据偏移量设置输出的值
    GE_ASSERT_SUCCESS(UpdateOffsetsByOffsetListAttr(node, mem_offset, 0U, has_input_offset_flag));
  } else {
    // Get the reference type of the node, default is false
    bool is_ref = false;
    // If GetBool fail, is_ref is false.
    (void)ge::AttrUtils::GetBool(node->GetOpDesc(), ATTR_NAME_REFERENCE, is_ref);

    // If the output is ref type and refers to the ref of an input, the name of the output
    // and the input are the same. Ge encounters ref type, finds matching relationship according
    // to the names of input and output, and allocates the same memory address, eg: HCOMBroadcast
    if (is_ref) {
      GELOGI("Current node %s no needs assign continuous output because reference input by name.",
             node->GetName().c_str());
      return SUCCESS;
    }
    mem_offset = op_desc->GetOutputOffset()[0];
  }

  std::vector<int64_t> output_list = op_desc->GetOutputOffset();
  for (auto &out_data_anchor : node->GetAllOutDataAnchors()) {
    if (!has_input_offset_flag) {
      output_list[out_data_anchor->GetIdx()] = mem_offset;
    }
    int64_t tensor_desc_size = 0;
    int64_t nopadding_size = 0;
    GE_ASSERT_SUCCESS(GetMemorySize(op_desc, op_desc->GetOutputDescPtr(out_data_anchor->GetIdx()), continuous_type,
                                    tensor_desc_size, nopadding_size));

    if (is_nopadding) {
      mem_offset += nopadding_size;
    } else {
      mem_offset += tensor_desc_size;
      ge::AlignMemOffset(mem_offset);
    }
    GELOGI("[IMAS]Continuous output : Set %s name[%s] optype[%s] output[%d] offset to [%zu] stream_id[%" PRId64
           "] memtype[%" PRId64 "] size[%zu] realsize[%" PRId64 "] nopadding[%d].",
           GraphNameId(compute_graph_.get()).c_str(), op_desc->GetName().substr(0, kMaxLogLen).c_str(),
           node->GetType().c_str(), out_data_anchor->GetIdx(), output_list[out_data_anchor->GetIdx()],
           op_desc->GetStreamId(), memory_type, 0UL, is_nopadding ? nopadding_size : tensor_desc_size, is_nopadding);
  }
  op_desc->SetOutputOffset(output_list);
  return ge::SUCCESS;
}

Status GraphMemoryAssigner::SetAtomicCleanOffset() const {
  GE_ASSERT_NOTNULL(atomic_memory_assigner_);
  return atomic_memory_assigner_->SetAtomicCleanOffset();
}

Status GraphMemoryAssigner::GetMemType(const Node *const node, const IOType &io_type, const uint32_t index,
                                       uint32_t &mem_type) const {
  GE_ASSERT_NOTNULL(atomic_memory_assigner_);
  return atomic_memory_assigner_->GetMemType(node, io_type, index, mem_type);
}

Status GraphMemoryAssigner::AssignReferenceMemory() const {
  for (auto &node : compute_graph_->GetAllNodes()) {
    auto out_op_desc = node->GetOpDesc();
    GE_CHECK_NOTNULL(out_op_desc);
    std::vector<int64_t> output_list = out_op_desc->GetOutputOffset();
    // output ref var
    for (const auto &out_data_anchor : node->GetAllOutDataAnchors()) {
      std::string var_name;
      if (ge::AttrUtils::GetStr(out_op_desc->GetOutputDescPtr(out_data_anchor->GetIdx()), ASSIGN_VAR_NAME, var_name) &&
          !var_name.empty()) {
        const auto session_id = compute_graph_->GetSessionID();
        GeTensorDesc var_tensor_desc;
        GE_ASSERT_NOTNULL(VarManager::Instance(session_id), "Get var manager failed, session_id=%llu", session_id);
        if (VarManager::Instance(session_id)->GetCurVarDesc(var_name, var_tensor_desc) != SUCCESS) {
          continue;
        }
        uint8_t *dev_ptr = nullptr;
        if (VarManager::Instance(session_id)->GetVarAddr(var_name, var_tensor_desc, dev_ptr) != SUCCESS) {
          continue;
        }
        output_list[out_data_anchor->GetIdx()] = static_cast<int64_t>(PtrToValue(dev_ptr));
        GELOGI("Op[%s] output[%u] set to var[%s]'s offset[%" PRId64 "].", out_op_desc->GetName().c_str(),
               out_data_anchor->GetIdx(), var_name.c_str(), output_list[out_data_anchor->GetIdx()]);
      }
    }
    out_op_desc->SetOutputOffset(output_list);
    GE_ASSERT_SUCCESS(AssignReferenceMemory(node), "node: %s(%s)", node->GetNamePtr(), node->GetTypePtr());
  }
  return ge::SUCCESS;
}

Status GraphMemoryAssigner::AssignReferenceMemory(const NodePtr &node) const {
  std::map<int32_t, int32_t> out2ins;
  GE_ASSERT_SUCCESS(TryGetNodeRefIndexes(node, out2ins), "[Get][RefIndexes]fail for node: %s", node->GetName().c_str());
  const auto type = node->GetType();
  const auto not_update_by_ref =
      (type == HCOMBROADCAST) || (type == HVDCALLBACKBROADCAST) || OpTypeUtils::IsDataNode(type);
  if (not_update_by_ref || out2ins.empty()) {
    return ge::SUCCESS;
  }
  for (const auto &anchor : node->GetAllInDataAnchors()) {
    bool be_ref = false;
    for (const auto &out2in : out2ins) {
      if (out2in.second == anchor->GetIdx()) {
        be_ref = true;
      }
    }
    if (!be_ref) {
      continue;
    }
    auto peer_out_anchor = anchor->GetPeerOutAnchor();
    if (peer_out_anchor == nullptr) {
      continue;
    }

    GE_CHECK_NOTNULL(peer_out_anchor->GetOwnerNode());
    const auto peer_node_op_desc = peer_out_anchor->GetOwnerNode()->GetOpDesc();
    GE_CHECK_NOTNULL(peer_node_op_desc);
    const auto peer_output_list = peer_node_op_desc->GetOutputOffset();
    const auto out_index = static_cast<unsigned long>(peer_out_anchor->GetIdx());
    if (peer_output_list.size() > static_cast<size_t>(out_index)) {
      int64_t peer_out_offset = peer_output_list.at(out_index);
      GE_CHK_STATUS_RET(UpdateRefOpOutputOffset(node, out2ins, anchor->GetIdx(), peer_out_offset),
                        "[Update][RefOffset]fail for node: %s", node->GetName().c_str());
    }
  }

  // ref output offset 设好后，同步到 symbol 链上的其他节点（含子图 netoutput）
  if (node->GetType() == WHILE) {
    for (const auto &out2in : out2ins) {
      auto output_list = node->GetOpDesc()->GetOutputOffset();
      if (static_cast<size_t>(out2in.first) < output_list.size()) {
        GE_CHK_STATUS_RET(UpdateSymbolOutputOffset(node, out2in.first, output_list[out2in.first]),
                          "[Update][SymbolOutputOffset]fail for node: %s", node->GetName().c_str());
      }
    }
  }
  return ge::SUCCESS;
}

Status GraphMemoryAssigner::UpdateSymbolOutputOffset(const NodePtr &node, int64_t output_index, int64_t offset) const {
  const auto *anchors = FindSymbolAnchors(GetMemAssignerPtr(), node, output_index);
  if (anchors == nullptr) {
    return SUCCESS;
  }
  for (const auto &anchor : *anchors) {
    auto op_desc = anchor.node_ptr_->GetOpDescBarePtr();
    if ((anchor.io_type_ != kOut) || (op_desc == nullptr)) {
      continue;
    }
    auto output_offsets = op_desc->GetOutputOffset();
    if (output_offsets.size() > anchor.index_) {
      output_offsets[anchor.index_] = offset;
      op_desc->SetOutputOffset(output_offsets);
      GELOGI("node %s(%s) output[%u] offset is updated to %lld, from node %s out_index %lld", op_desc->GetNamePtr(),
             op_desc->GetTypePtr(), anchor.index_, offset, node->GetNamePtr(), output_index);
    }
  }
  return SUCCESS;
}

Status GraphMemoryAssigner::CheckOffset() const {
  // 输出引用输入检查
  GE_ASSERT_SUCCESS(OffsetValidCheck());
  // 检查连续内存
  GE_ASSERT_SUCCESS(SpecialNodeChecker::Check(compute_graph_), "special node memory check failed.");
  // 内存复用检查
  GE_ASSERT_SUCCESS(ReuseCheck(), "reuse check failed.");
  // 集中清零检查
  GE_ASSERT_SUCCESS(AtomicCleanCheck());
  return SUCCESS;
}

Status GraphMemoryAssigner::ReuseCheck() const {
  auto reuse_checker = mem_assigner_->GetReuseChecker();
  if (reuse_checker != nullptr) {
    size_t max_offset = 0U;
    for (auto type_2_offset : memory_offset_) {
      if (type_2_offset.second.mem_offset_ > max_offset) {
        max_offset = type_2_offset.second.mem_offset_;
      }
    }
    GELOGI("set max feature offset %zu for memory reuse checker", max_offset);
    reuse_checker->SetMaxOffset(max_offset);
    GE_ASSERT_SUCCESS(reuse_checker->Check());
    mem_assigner_->ReuseCheckerDeInit();
  }
  return SUCCESS;
}

Status GraphMemoryAssigner::AtomicCleanCheck() const {
  AtomicCleanChecker checker(this);
  GE_ASSERT_SUCCESS(checker.Check(compute_graph_), "atomic clean address and size check failed, graph: %s",
                    compute_graph_->GetName().c_str());
  return SUCCESS;
}

Status GraphMemoryAssigner::OffsetValidCheck() const {
  for (const ge::NodePtr &node : compute_graph_->GetAllNodes()) {
    GE_CHECK_NOTNULL(node->GetOpDesc());
    std::vector<int64_t> input_list = node->GetOpDesc()->GetInputOffset();
    for (auto input : input_list) {
      if (input == ge::kInvalidOffset) {
        std::string error =
            "Invalid input offset" + FmtToStr(ge::kInvalidOffset) + +" in node" + FmtToStr(node->GetName());
        GE_ERRORLOG_AND_ERRORMSG(FAILED, error.c_str());
        return FAILED;
      }
    }

    std::vector<int64_t> output_list = node->GetOpDesc()->GetOutputOffset();
    for (auto output : output_list) {
      if (output == ge::kInvalidOffset) {
        std::string error =
            "Invalid output offset" + FmtToStr(ge::kInvalidOffset) + +" in node" + FmtToStr(node->GetName());
        GE_ERRORLOG_AND_ERRORMSG(FAILED, error.c_str());
        return FAILED;
      }
    }

    std::vector<int64_t> workspace_list = node->GetOpDesc()->GetWorkspace();
    for (auto workspace : workspace_list) {
      if (workspace == ge::kInvalidOffset) {
        std::string error =
            "Invalid workspace" + FmtToStr(ge::kInvalidOffset) + +" in node" + FmtToStr(node->GetName());
        GE_ERRORLOG_AND_ERRORMSG(FAILED, error.c_str());
        return FAILED;
      }
    }
    // check reuse input and output
    GE_CHK_STATUS_RET(CheckRefNodeOffset(node), "[Check][Offset]fail for node: %s", node->GetName().c_str());
  }
  return SUCCESS;
}

Status GraphMemoryAssigner::UpdateParentNodeOffset() const {
  GE_CHECK_NOTNULL(mem_assigner_);
  const AnchorToSymbol &anchor_to_symbol = mem_assigner_->GetAnchorToSymbol();
  const SymbolToAnchors &symbol_to_anchors = mem_assigner_->GetSymbolToAnchors();
  for (const ge::NodePtr &node : compute_graph_->GetAllNodes()) {
    GE_CHECK_NOTNULL(node->GetOpDesc());
    if (node->GetOpDesc()->GetSubgraphInstanceNames().empty()) {
      continue;
    }
    bool need_update_output = false;
    std::vector<int64_t> output_list = node->GetOpDesc()->GetOutputOffset();
    for (uint32_t i = 0; i < output_list.size(); ++i) {
      auto symbol_offset = GetSymbolOutputOffset(anchor_to_symbol, symbol_to_anchors, node, i);
      if (symbol_offset != ge::kInvalidOffset && output_list[i] != symbol_offset) {
        output_list[i] = symbol_offset;
        need_update_output = true;
      }
    }
    if (need_update_output) {
      node->GetOpDesc()->SetOutputOffset(output_list);
    }
  }
  return SUCCESS;
}

void SetAnchorIdx2InputIdxMap(const NodePtr &node, std::unordered_map<int32_t, int32_t> &anchor_idx_2_input_idx) {
  int32_t input_idx = 0;
  for (size_t idx = 0U; idx < node->GetOpDesc()->GetAllInputsSize(); ++idx) {
    const auto &input_desc = node->GetOpDesc()->MutableInputDesc(static_cast<uint32_t>(idx));
    if (input_desc == nullptr) {
      GELOGD("Node[%s] has unfed input, anchor index[%zu], node type[%s].", node->GetName().c_str(), idx,
             node->GetType().c_str());
    } else {
      anchor_idx_2_input_idx[static_cast<int32_t>(idx)] = input_idx;
      ++input_idx;
    }
  }
}
ge::Status GraphMemoryAssigner::CheckRefNodeOffset(const NodePtr &node) const {
  GE_CHECK_NOTNULL(node);
  std::map<int32_t, int32_t> out2ins;
  GE_CHK_STATUS_RET(TryGetNodeRefIndexes(node, out2ins), "[Get][RefIndexes]fail for node: %s", node->GetName().c_str());
  auto opdesc = node->GetOpDesc();
  GE_CHECK_NOTNULL(opdesc);
  auto output_list = opdesc->GetOutputOffset();
  auto input_list = opdesc->GetInputOffset();
  std::unordered_map<int32_t, int32_t> anchor_idx_2_input_idx;
  const bool is_node_has_unfed_optional_input = (opdesc->GetAllInputsSize() != opdesc->GetInputsSize());
  if (is_node_has_unfed_optional_input) {
    SetAnchorIdx2InputIdxMap(node, anchor_idx_2_input_idx);
    GE_CHK_BOOL_RET_STATUS(
        (anchor_idx_2_input_idx.size() == node->GetOpDesc()->GetInputOffset().size()), ge::PARAM_INVALID,
        "[Check][Failed]Node[%s], type[%s], input desc num[%zu] must be equal to input offset size[%zu].",
        node->GetName().c_str(), node->GetType().c_str(), anchor_idx_2_input_idx.size(), input_list.size());
  }
  for (const auto &out2in : out2ins) {
    auto out_i = out2in.first;
    if (static_cast<size_t>(out_i) >= output_list.size()) {
      std::string error = "Node" + FmtToStr(opdesc->GetName()) + "output offset size" + FmtToStr(output_list.size()) +
                          "should bigger than ref out index" + FmtToStr(out_i);
      GE_ERRORLOG_AND_ERRORMSG(ge::FAILED, error.c_str());
      return ge::FAILED;
    }
    auto in_i = out2in.second;
    if (is_node_has_unfed_optional_input) {
      auto iter = anchor_idx_2_input_idx.find(in_i);
      if (iter != anchor_idx_2_input_idx.cend()) {
        in_i = iter->second;
      } else {
        GELOGE(ge::PARAM_INVALID, "[Check][Failed]Anchor index[%d] has no valid input desc.", in_i);
      }
    }
    if (static_cast<size_t>(in_i) >= input_list.size()) {
      std::string error = "Node" + FmtToStr(opdesc->GetName()) + "input offset size" + FmtToStr(input_list.size()) +
                          "should bigger than ref input index" + FmtToStr(in_i);
      GE_ERRORLOG_AND_ERRORMSG(ge::FAILED, error.c_str());
      return ge::FAILED;
    }
    int64_t custom_offset = 0;
    GE_ASSERT_SUCCESS(GetCustomOffset(node, in_i, out_i, custom_offset));
    if (output_list[out_i] != input_list[in_i] + custom_offset) {
      std::string error = "Node" + FmtToStr(opdesc->GetName()) + "input offset " + FmtToStr(input_list[in_i]) + " + " +
                          FmtToStr(custom_offset) + " should equal to output offset" + FmtToStr(output_list[out_i]) +
                          " with ref in" + FmtToStr(in_i) + "to output" + FmtToStr(out_i);
      GE_ERRORLOG_AND_ERRORMSG(ge::FAILED, error.c_str());
      return ge::FAILED;
    }
  }
  return ge::SUCCESS;
}

ge::Status GraphMemoryAssigner::SetInputOffset() const {
  if (memory_offset_.empty()) {
    REPORT_INNER_ERR_MSG("E19999", "InnerData memory_offset_ empty, not expected, graph_id:%u, graph_name:%s",
                         compute_graph_->GetGraphID(), compute_graph_->GetName().c_str());
    GELOGE(FAILED,
           "[Check][InnerData:memory_offset_]empty is not expected, "
           "graph_id:%u, graph_name:%s",
           compute_graph_->GetGraphID(), compute_graph_->GetName().c_str());
    return ge::FAILED;
  }
  for (auto pair : memory_offset_) {
    if ((pair.first != RT_MEMORY_HBM) && (pair.second.mem_offset_ == 0)) {
      continue;
    }
    if (mem_assigner_ == nullptr) {
      continue;
    }
    std::map<uint64_t, MemoryStat>::const_iterator it_memory_stat = mem_assigner_->GetMemoryStat().find(pair.first);
    if (it_memory_stat == mem_assigner_->GetMemoryStat().cend()) {
      continue;
    }
    GEEVENT("[IMAS]AfterAssignMemory : %s memoffset[%zu], memtype[%" PRId64
            "], theory_min[%zu], zero_copy[%zu], "
            "total_size[%zu], no_reuse[%zu], streams[%zu], %s",
            GraphNameId(compute_graph_.get()).c_str(), pair.second.mem_offset_, pair.first, pair.second.theory_min_,
            pair.second.zero_copy_size_, it_memory_stat->second.total_memory_size_,
            it_memory_stat->second.theory_no_reuse_memory_size_, it_memory_stat->second.stream_count_,
            GetMemoryOption().c_str());

    // report graph total memory size, op_desc is nullptr
    CANN_PROFILING_REPORT_STATIC_OP_MEM_INFO(compute_graph_, nullptr, it_memory_stat->second.total_memory_size_,
                                             kMinLifeTime, kMaxLifeTime);
  }

  for (const ge::NodePtr &node : compute_graph_->GetAllNodes()) {
    if (UpdateOpInputOffset(node) != ge::SUCCESS) {
      GELOGE(ge::FAILED, "[Update][Offset:Input]fail for op:%s", node->GetName().c_str());
      return ge::FAILED;
    }
    if (UpdateOpInputDescOffset(node) != ge::SUCCESS) {
      GELOGE(ge::FAILED, "[Update][Offset:Input Desc]fail for op:%s", node->GetName().c_str());
      return ge::FAILED;
    }
  }
  return ge::SUCCESS;
}

NodePtr GraphMemoryAssigner::GetKnownInputNode(const NodePtr &node) const {
  if ((node->GetOpDesc() != nullptr) && (!node->GetOpDesc()->HasAttr(ATTR_NAME_PARENT_NODE_INDEX))) {
    return node;
  }

  if (NodeUtils::IsDynamicShape(node)) {
    return node;
  }

  return NodeUtils::GetParentInput(node);
}

ge::Status GraphMemoryAssigner::UpdateConstArgsOffset(const NodePtr &node, std::vector<int64_t> &input_list) const {
  GE_CHECK_NOTNULL(node->GetOpDesc());
  uint32_t parent_index = 0;
  if (!AttrUtils::GetInt(node->GetOpDesc(), ATTR_NAME_PARENT_NODE_INDEX, parent_index)) {
    return SUCCESS;
  }

  // Memory allocated for dynamic shape subgraph Data.
  const auto &owner = node->GetOwnerComputeGraph();
  bool dynamic_shape_partition = false;
  (void)AttrUtils::GetBool(owner->GetParentGraph(), ATTR_NAME_DYNAMIC_SHAPE_PARTITIONED, dynamic_shape_partition);
  if (owner->GetParentGraph()->GetGraphUnknownFlag() || dynamic_shape_partition) {
    return SUCCESS;
  }

  // Subgraph Data Node, check for constant input.
  std::string op_type;
  const auto &in_node = NodeUtils::GetParentInput(node);
  GE_CHECK_NOTNULL(in_node);
  if (gert::GraphUnfolder::IsDataNotNeedRefConst(node)) {
    return SUCCESS;
  }
  if (NodeUtils::GetConstOpType(in_node, op_type) || OpTypeUtils::IsVariableNode(in_node->GetType())) {
    input_list = in_node->GetOpDesc()->GetOutputOffset();
    node->GetOpDesc()->SetOutputOffset(input_list);  // Set Data output same as const output.
    if (!input_list.empty()) {
      GELOGI("update node %s output offset to [%lld]", node->GetNamePtr(), input_list[0U]);
    }
    return SUCCESS;  // Constant/Variable input.
  }

  const auto &parent_desc = owner->GetParentNode()->GetOpDesc();
  const auto parent_inputs = parent_desc->GetInputOffset();
  if (parent_inputs.size() <= parent_index) {
    std::string error = "Get Parent input offset failed, node is " + FmtToStr(node->GetName()) + +", input_size is " +
                        FmtToStr(parent_inputs.size()) + ", parent index is " + FmtToStr(parent_index);
    GE_ERRORLOG_AND_ERRORMSG(FAILED, error.c_str());
    return FAILED;
  }

  input_list = {parent_inputs[parent_index]};
  node->GetOpDesc()->SetOutputOffset(input_list);  // Set Data output same as parent input.
  if (!input_list.empty()) {
    GELOGI("update node %s output offset to [%lld]", node->GetNamePtr(), input_list[0U]);
  }
  return SUCCESS;
}

ge::Status GraphMemoryAssigner::UpdateOpInputDescOffset(const NodePtr &node) const {
  if (node->GetType() == DATA) {
    int32_t parent_index;
    auto op_desc = node->GetOpDesc();
    GE_CHECK_NOTNULL(op_desc);
    if (!AttrUtils::GetInt(op_desc, ATTR_NAME_PARENT_NODE_INDEX, parent_index)) {
      return SUCCESS;
    }
    const auto &graph = node->GetOwnerComputeGraph();
    GE_CHECK_NOTNULL(graph);
    const auto &parent_node = graph->GetParentNode();
    GE_CHECK_NOTNULL(parent_node);
    const auto &parent_desc = parent_node->GetOpDesc();
    GE_CHECK_NOTNULL(parent_desc);
    auto data_tensor = op_desc->MutableOutputDesc(0U);
    GE_CHECK_NOTNULL(data_tensor);
    bool is_no_tiling = false;
    (void)AttrUtils::GetBool(data_tensor, ATTR_NAME_TENSOR_NO_TILING_MEM_TYPE, is_no_tiling);
    const auto &input_tensor = parent_desc->GetInputDescPtr(static_cast<uint32_t>(parent_index));
    GE_CHECK_NOTNULL(input_tensor);
    if (is_no_tiling) {
      int64_t mem_offset;
      if (!AttrUtils::GetInt(input_tensor, ATTR_NAME_TENSOR_DESC_MEM_OFFSET, mem_offset)) {
        REPORT_INNER_ERR_MSG("E19999", "Update op[%s] input no tiling mem offset by parent failed, parent node[%s].",
                             node->GetName().c_str(), parent_desc->GetName().c_str());
        GELOGE(FAILED, "Update op[%s] input no tiling mem offset by parent failed, parent node[%s].",
               node->GetName().c_str(), parent_desc->GetName().c_str());
        return FAILED;
      }
      (void)AttrUtils::SetInt(data_tensor, ATTR_NAME_TENSOR_DESC_MEM_OFFSET, mem_offset);
      GELOGD("Set data node[%s] output desc memory offset[%" PRId64 "] by parent node[%s] input tensor[%d]",
             op_desc->GetName().c_str(), mem_offset, parent_node->GetName().c_str(), parent_index);
    }
    return SUCCESS;
  }
  for (const auto &anchor : node->GetAllInDataAnchors()) {
    const auto &peer_anchor = anchor->GetPeerOutAnchor();
    if (peer_anchor == nullptr) {
      continue;
    }
    const auto &peer_node = peer_anchor->GetOwnerNode();
    GE_CHECK_NOTNULL(peer_node);
    const auto &peer_op_desc = peer_node->GetOpDesc();
    GE_CHECK_NOTNULL(peer_op_desc);
    const auto out_index = peer_anchor->GetIdx();
    auto op_desc = node->GetOpDesc();
    GE_CHECK_NOTNULL(op_desc);
    auto in_tensor_desc = op_desc->MutableInputDesc(anchor->GetIdx());
    GE_CHECK_NOTNULL(in_tensor_desc);
    bool is_no_tiling = false;
    (void)AttrUtils::GetBool(in_tensor_desc, ATTR_NAME_TENSOR_NO_TILING_MEM_TYPE, is_no_tiling);
    const auto &out_tensor_desc = peer_op_desc->GetOutputDescPtr(static_cast<uint32_t>(out_index));
    GE_CHECK_NOTNULL(out_tensor_desc);
    if (is_no_tiling) {
      int64_t mem_offset;
      if (!AttrUtils::GetInt(out_tensor_desc, ATTR_NAME_TENSOR_DESC_MEM_OFFSET, mem_offset)) {
        REPORT_INNER_ERR_MSG("E19999", "Update op[%s] input no tiling mem offset failed, peer node[%s].",
                             node->GetName().c_str(), peer_op_desc->GetName().c_str());
        GELOGE(FAILED, "Update op[%s] input no tiling mem offset failed, peer node[%s].", node->GetName().c_str(),
               peer_op_desc->GetName().c_str());
        return FAILED;
      }
      (void)AttrUtils::SetInt(in_tensor_desc, ATTR_NAME_TENSOR_DESC_MEM_OFFSET, mem_offset);
      GELOGD("Set node[%s] tensor[%d] desc memory offset[%" PRId64 "] by peer node[%s] tensor[%d]",
             op_desc->GetName().c_str(), anchor->GetIdx(), mem_offset, peer_op_desc->GetName().c_str(), out_index);
    }
  }
  return SUCCESS;
}

ge::Status GraphMemoryAssigner::UpdateOpInputOffset(const NodePtr &node, std::vector<int64_t> &input_list) const {
  std::vector<int64_t> origin_input_list;
  std::vector<int64_t> memory_type;
  auto tmp_op_desc = node->GetOpDesc();
  GE_CHECK_NOTNULL(tmp_op_desc);
  origin_input_list = tmp_op_desc->GetInputOffset();
  int64_t valid_input_index = 0;
  bool has_mem_type_attr = ge::AttrUtils::GetListInt(tmp_op_desc, ATTR_NAME_INPUT_MEM_TYPE_LIST, memory_type);
  std::map<int32_t, int32_t> out2ins;
  GE_CHK_STATUS_RET(TryGetNodeRefIndexes(node, out2ins), "[Get][RefIndexes]fail for node: %s", node->GetName().c_str());
  for (const auto &anchor : node->GetAllInDataAnchors()) {
    std::vector<int64_t> output_list;
    auto peer_out_anchor = anchor->GetPeerOutAnchor();
    if (peer_out_anchor == nullptr) {
      continue;
    }

    // If the current node not broadcast, the OutputOffset of the previous node is used to update the input_list
    auto last_peer_out_node = peer_out_anchor->GetOwnerNode();
    auto last_peer_out_op_desc = last_peer_out_node->GetOpDesc();
    GE_CHECK_NOTNULL(last_peer_out_op_desc);
    output_list = last_peer_out_op_desc->GetOutputOffset();
    auto out_index = static_cast<unsigned long>(peer_out_anchor->GetIdx());
    if (output_list.size() > static_cast<size_t>(out_index)) {
      int64_t peer_out_inner_offset = 0;
      if (ge::AttrUtils::GetInt(last_peer_out_op_desc->MutableOutputDesc(out_index), ATTR_NAME_INNER_OFFSET,
                                peer_out_inner_offset)) {
        (void)ge::AttrUtils::SetInt(tmp_op_desc->MutableInputDesc(anchor->GetIdx()), ATTR_NAME_INNER_OFFSET,
                                    peer_out_inner_offset);
      }
      bool is_l1_type = false;
      bool is_ub_type = false;
      int64_t input_offset = output_list.at(out_index);
      if (has_mem_type_attr && !origin_input_list.empty()) {
        auto input_size = tmp_op_desc->GetInputsSize();
        auto ori_input_offset_list_size = origin_input_list.size();
        auto mem_type_size = memory_type.size();
        if ((input_size != mem_type_size) || (input_size != ori_input_offset_list_size)) {
          std::string error = "Node" + FmtToStr(tmp_op_desc->GetName()) + +" input_size" + FmtToStr(input_size) +
                              " diff from memory_type_size" + FmtToStr(mem_type_size) +
                              " from ori_input_offset_list_size" + FmtToStr(ori_input_offset_list_size);
          GE_ERRORLOG_AND_ERRORMSG(ge::FAILED, error.c_str());
          return ge::FAILED;
        }
        int64_t inner_offset = 0;
        (void)ge::AttrUtils::GetInt(tmp_op_desc->MutableInputDesc(anchor->GetIdx()), ATTR_NAME_INNER_OFFSET,
                                    inner_offset);
        GELOGD("Node[%s] input[%d] has origin offset[%" PRId64 "] origin_inner_offset[%" PRId64 "]",
               tmp_op_desc->GetName().c_str(), anchor->GetIdx(), origin_input_list[valid_input_index], inner_offset);
        // L1 keep original input_offset
        is_l1_type = (memory_type[valid_input_index] == RT_MEMORY_L1);
        is_ub_type = (memory_type[valid_input_index] == static_cast<int64_t>(kRtMemoryUB));
        if (is_l1_type || is_ub_type) {
          input_offset = origin_input_list[valid_input_index];
        } else {
          // hbm input_offset = original input_offset + output_offset
          if ((origin_input_list[valid_input_index] != 0) && (!tmp_op_desc->GetSubgraphInstanceNames().empty())) {
            std::string error = "Node" + FmtToStr(tmp_op_desc->GetName()) +
                                +" has subgraphs which is conflict with has origin_input_list" +
                                FmtToStr(origin_input_list[valid_input_index]);
            GE_ERRORLOG_AND_ERRORMSG(ge::FAILED, error.c_str());
            return ge::FAILED;
          }
          input_offset = origin_input_list[valid_input_index] + output_list.at(out_index);
          (void)ge::AttrUtils::SetInt(tmp_op_desc->MutableInputDesc(anchor->GetIdx()), ATTR_NAME_INNER_OFFSET,
                                      origin_input_list[valid_input_index] + inner_offset);
        }
      }
      const auto &in_node = GetKnownInputNode(last_peer_out_node);
      if ((in_node != nullptr) && (in_node->GetType() == CONSTANT) &&
          (!gert::GraphUnfolder::IsDataNotNeedRefConst(last_peer_out_node)) && (!is_ub_type)) {
        GeTensorDescPtr tensor_desc = tmp_op_desc->MutableInputDesc(static_cast<uint32_t>(anchor->GetIdx()));
        GE_CHECK_NOTNULL(tensor_desc);
        GE_CHK_STATUS(TensorUtils::GetDataOffset(*tensor_desc, input_offset));
        int64_t inner_offset{0};
        (void)AttrUtils::GetInt(tensor_desc, ATTR_NAME_INNER_OFFSET, inner_offset);
        input_offset += inner_offset;
        TensorUtils::SetDataOffset(*tensor_desc, input_offset);
      }

      if ((!is_l1_type) && (!is_ub_type)) {
        // update ref output_offset when input change
        GE_CHK_STATUS_RET(UpdateRefOpOutputOffset(node, out2ins, anchor->GetIdx(), input_offset),
                          "[Update][RefOffset]fail for node: %s", node->GetName().c_str());
      }
      GELOGD("Node[%s] input[%d] is set from node[%s] out index[%" PRIu64 "] offset[%" PRId64 "]",
             tmp_op_desc->GetName().c_str(), anchor->GetIdx(),
             peer_out_anchor->GetOwnerNode()->GetOpDesc()->GetName().c_str(), out_index, input_offset);
      input_list.emplace_back(input_offset);
      valid_input_index++;
    }
  }
  return ge::SUCCESS;
}

ge::Status GraphMemoryAssigner::UpdateRefOpOutputOffset(const NodePtr &node, const std::map<int32_t, int32_t> &out2ins,
                                                        const int32_t ref_in, const int64_t input_offset) const {
  auto opdesc = node->GetOpDesc();
  GE_CHECK_NOTNULL(opdesc);
  GE_CHECK_NOTNULL(opdesc->MutableInputDesc(ref_in));
  int64_t inner_offset = 0;
  bool has_inner_offset = ge::AttrUtils::GetInt(opdesc->MutableInputDesc(ref_in), ATTR_NAME_INNER_OFFSET, inner_offset);
  for (const auto &out2in : out2ins) {
    auto out_i = out2in.first;
    auto in_i = out2in.second;
    if (in_i == ref_in) {
      int64_t custom_offset = 0;
      GE_ASSERT_SUCCESS(GetCustomOffset(node, ref_in, out_i, custom_offset));
      // 设置自定义offset时，不需要更新
      if (custom_offset != 0) {
        continue;
      }
      auto origin_output_list = opdesc->GetOutputOffset();
      if (static_cast<size_t>(out_i) >= origin_output_list.size()) {
        std::string error = "Node" + FmtToStr(opdesc->GetName()) + "output offset size" +
                            FmtToStr(origin_output_list.size()) + "should bigger than ref out index" + FmtToStr(out_i);
        GE_ERRORLOG_AND_ERRORMSG(ge::FAILED, error.c_str());
        return ge::FAILED;
      }
      int64_t origin_offset = origin_output_list[out_i];
      origin_output_list[out_i] = input_offset;
      opdesc->SetOutputOffset(origin_output_list);
      if (has_inner_offset) {
        GE_CHECK_NOTNULL(opdesc->MutableOutputDesc(out_i));
        (void)ge::AttrUtils::SetInt(opdesc->MutableOutputDesc(out_i), ATTR_NAME_INNER_OFFSET, inner_offset);
      }
      GELOGI("Node[%s] output[%d] is updated from reuse input index[%d] from offset[%" PRId64 "] to offset[%" PRId64
             "], inner_offset[%" PRId64 "]",
             opdesc->GetName().c_str(), out_i, ref_in, origin_offset, origin_output_list[out_i], inner_offset);
      GE_ASSERT_SUCCESS(UpdateNoPaddingContinousOutputOffsets(node, input_offset, inner_offset));
    }
  }
  return ge::SUCCESS;
}

ge::Status GraphMemoryAssigner::UpdateOpInputOffset(const NodePtr &node) const {
  GE_CHECK_NOTNULL(node->GetOpDesc());
  std::vector<int64_t> input_list;
  if (node->GetType() == HCOMBROADCAST || node->GetType() == HVDCALLBACKBROADCAST) {
    for (const auto &anchor : node->GetAllInDataAnchors()) {
      std::vector<int64_t> output_list;
      auto peer_out_anchor = anchor->GetPeerOutAnchor();
      if (peer_out_anchor == nullptr) {
        continue;
      }

      auto last_peer_out_node = peer_out_anchor->GetOwnerNode();
      // If the current node is broadcast and the preceding node is variable, because InputOffset has been set
      // in function:AssignVarAttr2Nodes, then the InputOffset of the broadcast node is taken to update the input_list.
      // Otherwise, the OutputOffset of the previous node is used to update the input_list.
      if (last_peer_out_node->GetType() != VARIABLE) {
        auto last_peer_out_op_desc = last_peer_out_node->GetOpDesc();
        GE_CHECK_NOTNULL(last_peer_out_op_desc);
        output_list = last_peer_out_op_desc->GetOutputOffset();
        if (output_list.size() > static_cast<size_t>(peer_out_anchor->GetIdx())) {
          input_list.emplace_back(output_list.at(peer_out_anchor->GetIdx()));
        }
      } else {
        std::vector<int64_t> cur_node_input_list;
        auto cur_node_op_desc = node->GetOpDesc();
        GE_CHECK_NOTNULL(cur_node_op_desc);
        cur_node_input_list = cur_node_op_desc->GetInputOffset();
        if (cur_node_input_list.size() > static_cast<size_t>(anchor->GetIdx())) {
          input_list.emplace_back(cur_node_input_list.at(anchor->GetIdx()));
        }
      }
    }
    if (MemReuseUtils::IsAllOutRefAllInput(node)) {
      node->GetOpDesc()->SetOutputOffset(input_list);
      GELOGI("node: %s(%s) update output offset from input offset", node->GetNamePtr(), node->GetTypePtr());
    }
  } else if (OpTypeUtils::IsDataNode(node->GetType())) {
    if (UpdateConstArgsOffset(node, input_list) != SUCCESS) {
      GELOGE(FAILED, "[Update][Offset:Input:Const]fail for node:%s ", node->GetName().c_str());
      return FAILED;
    }
  } else {
    if (UpdateOpInputOffset(node, input_list) != SUCCESS) {
      GELOGE(FAILED, "[Update][Offset:Input]fail for node:%s", node->GetName().c_str());
      return FAILED;
    }
  }

  node->GetOpDesc()->SetInputOffset(input_list);
  return SUCCESS;
}

ge::Status GraphMemoryAssigner::GetNodeMemoryType(const NodePtr &node, int64_t &memory_type,
                                                  std::string input_or_output) const {
  memory_type = RT_MEMORY_HBM;
  uint32_t anchors_size = 0U;
  std::string mem_type_str;
  if (input_or_output == "input") {
    mem_type_str = ATTR_NAME_INPUT_MEM_TYPE_LIST;
    anchors_size = node->GetAllInDataAnchorsSize();
  } else if (input_or_output == "output") {
    mem_type_str = ATTR_NAME_OUTPUT_MEM_TYPE_LIST;
    anchors_size = node->GetAllOutDataAnchorsSize();
  }
  std::vector<int64_t> mem_type_list;
  (void)ge::AttrUtils::GetListInt(node->GetOpDesc(), mem_type_str, mem_type_list);
  if (mem_type_list.empty()) {
    if (memory_offset_.find(memory_type) == memory_offset_.end()) {
      std::string error = "Memory offset map does not have memory type" + FmtToStr(memory_type) + +", opname is " +
                          FmtToStr(node->GetName()) + ", optype is " + FmtToStr(node->GetType());
      GE_ERRORLOG_AND_ERRORMSG(FAILED, error.c_str());
      return FAILED;
    }
    return SUCCESS;
  }

  if (mem_type_list.size() != anchors_size) {
    std::string error = "input or output : " + input_or_output + ", The size" + FmtToStr(mem_type_list.size()) +
                        " of mem type list is not equal to the size of data anchor" + FmtToStr(anchors_size) +
                        ", opname is " + FmtToStr(node->GetName()) + ", optype is " + FmtToStr(node->GetType());
    GE_ERRORLOG_AND_ERRORMSG(FAILED, error.c_str());
    return FAILED;
  }

  if (!CheckContinuousMemType(mem_type_list)) {
    GELOGE(FAILED, "[Check][MemType:Continuous]fail for node:%s", node->GetName().c_str());
    return FAILED;
  }
  // It is continuous memory and memory type is the same, so use the first memory.
  memory_type = mem_type_list[0];
  return SUCCESS;
}

bool GraphMemoryAssigner::CheckContinuousMemType(std::vector<int64_t> mem_type_list) const {
  if (mem_type_list.size() == 0) {
    return true;
  }
  int64_t mem_type_tmp = mem_type_list[0];
  for (auto mem_type : mem_type_list) {
    if (mem_type != mem_type_tmp) {
      REPORT_INNER_ERR_MSG(
          "E19999", "The memory is continuous, but the type of the input memory is inconsistent. They are %s and %s",
          FmtToStr(mem_type_tmp).c_str(), FmtToStr(mem_type).c_str());
      GELOGW("The memory is continuous, but the type of the input memory is inconsistent. They are [%" PRId64
             "] and [%" PRId64 "].",
             mem_type_tmp, mem_type);
      return false;
    }
  }
  if (memory_offset_.find(mem_type_tmp) == memory_offset_.end()) {
    REPORT_INNER_ERR_MSG("E19999", "Memory offset map does not have memory type %s", FmtToStr(mem_type_tmp).c_str());
    GELOGW("Memory offset map does not have memory type[%" PRId64 "].", mem_type_tmp);
    return false;
  }
  return true;
}

ge::Status GraphMemoryAssigner::TryGetNodeRefIndexes(const NodePtr &node, std::map<int32_t, int32_t> &out2ins) const {
  // data and netoutput no need check because only data's output or netoutput's input is used
  if (OpTypeUtils::IsDataNode(node->GetType()) || (node->GetType() == NETOUTPUT)) {
    return ge::SUCCESS;
  }
  for (const auto &out_data_anchor : node->GetAllOutDataAnchors()) {
    int32_t reuse_in_index = -1;
    // nopadding means output[0] reuse input[0], but as history reason,
    // other output index also return true for mem assign in block_mem_assigner
    if (GraphUtils::IsNoPaddingRefFromInput(out_data_anchor, reuse_in_index)) {
      out2ins.emplace(out_data_anchor->GetIdx(), reuse_in_index);
      return ge::SUCCESS;
    }
    bool reuse_input_flag = GraphUtils::IsRefFromInput(out_data_anchor, reuse_in_index);
    if (reuse_input_flag) {
      if (node->GetInDataAnchor(reuse_in_index) != nullptr) {
        out2ins.emplace(out_data_anchor->GetIdx(), reuse_in_index);
      } else {
        REPORT_INNER_ERR_MSG("E19999",
                             "Invalid reuse_input value %d on output %d of node %s, "
                             "please check attr reuse_input",
                             reuse_in_index, out_data_anchor->GetIdx(), node->GetName().c_str());
        GELOGE(FAILED,
               "[Check][Attr]Invalid reuse_input value %d on output %d of node %s, "
               "please check attr reuse_input",
               reuse_in_index, out_data_anchor->GetIdx(), node->GetName().c_str());
        return FAILED;
      }
    }
  }

  return ge::SUCCESS;
}

bool GraphMemoryAssigner::IsAssignContinuousInputMemoryDirectly(
    const NodePtr &input_continuous_node, std::map<NodePtr, uint32_t> &node_2_continuous_type) const {
  for (const auto &in_node : input_continuous_node->GetInDataNodes()) {
    if (in_node->GetType() == VARIABLE) {
      GELOGI("node %s 's precursor node %s is variable, do not store.", input_continuous_node->GetName().c_str(),
             in_node->GetName().c_str());
      return true;
    }
    std::map<NodePtr, uint32_t>::const_iterator iter = node_2_continuous_type.find(in_node);
    // In node's topo order in the front, so function cannot be exception
    auto continuous_type = iter->second;
    bool continuous_input = ((continuous_type & ContinuousType::kTypeInput) != 0) ||
                            ((continuous_type & ContinuousType::kTypeInputNoPadding) != 0);
    if (continuous_input) {
      GELOGI("[Store][Node] of %s cause it's precursor node %s need assign continuous input memory",
             input_continuous_node->GetName().c_str(), in_node->GetName().c_str());
      return false;
    }
  }
  for (const auto &out_node : input_continuous_node->GetOutDataNodes()) {
    auto continuous_type = GetContinuousMemoryType(out_node);
    node_2_continuous_type.emplace(out_node, continuous_type);
    bool continuous_input = ((continuous_type & ContinuousType::kTypeInput) != 0) ||
                            ((continuous_type & ContinuousType::kTypeInputNoPadding) != 0);
    if (continuous_input) {
      GELOGI("[Store][Node] of %s cause it's succeed node %s need assign continuous input memory",
             input_continuous_node->GetName().c_str(), out_node->GetName().c_str());
      return false;
    }
  }

  return true;
}

Status GraphMemoryAssigner::AssignBufferPoolMemory() {
  auto is_buffer_pool_mem_enable = [](const ComputeGraphPtr &graph) -> bool {
    for (NodePtr &node : graph->GetAllNodes()) {
      auto op_desc = node->GetOpDesc();
      if (op_desc == nullptr) {
        continue;
      }
      bool has_attrs = op_desc->HasAttr(ATTR_NAME_BUFFER_POOL_ID) && op_desc->HasAttr(ATTR_NAME_BUFFER_POOL_SIZE);
      if (has_attrs) {
        return true;
      }
    }
    return false;
  };
  auto root_graph = GraphUtils::FindRootGraph(compute_graph_);
  GE_CHECK_NOTNULL(root_graph);
  if (root_graph->GetGraphUnknownFlag()) {
    GELOGI("[Check][Enable]Unknown root graph does not support buffer pool memory, graph:%s.",
           compute_graph_->GetName().c_str());
    return SUCCESS;
  }
  if (!is_buffer_pool_mem_enable(compute_graph_)) {
    GELOGD("[Check][Enable]Buffer pool memory is not enable, graph:%s.", compute_graph_->GetName().c_str());
    return SUCCESS;
  }
  std::map<int64_t, size_t> mem_type_to_offset;
  for (const auto &pair : memory_offset_) {
    mem_type_to_offset[pair.first] = pair.second.mem_offset_;
  }
  BufferPoolMemAssigner buffer_pool_mem_assigner(compute_graph_, mem_type_to_offset);
  Status status = buffer_pool_mem_assigner.Assign();
  if (status != SUCCESS) {
    GELOGE(status, "[Assign][BufferPoolMem]Graph:%s.", compute_graph_->GetName().c_str());
    REPORT_INNER_ERR_MSG("E19999", "Failed to assign buffer pool memory, graph:%s.", compute_graph_->GetName().c_str());
    return status;
  }
  int64_t mem_type = buffer_pool_mem_assigner.GetMemType();
  auto iter = memory_offset_.find(mem_type);
  if (iter == memory_offset_.end()) {
    GELOGE(FAILED, "[Check][MemType]Memory type is not supported, graph:%s, mem type:%" PRId64 ".",
           compute_graph_->GetName().c_str(), mem_type);
    REPORT_INNER_ERR_MSG("E19999", "Memory type is not supported, graph:%s, mem type:%" PRId64 ".",
                         compute_graph_->GetName().c_str(), mem_type);
    return FAILED;
  }
  iter->second.mem_offset_ = buffer_pool_mem_assigner.GetMemOffset();
  GELOGI("[Assign][BufferPoolMem]Assign buffer pool memory successfully, graph:%s, mem type:%" PRId64
         ", mem offset:%zu.",
         compute_graph_->GetName().c_str(), mem_type, buffer_pool_mem_assigner.GetMemOffset());
  return SUCCESS;
}

// if producer and customers in the same stream, or customers on the same stream when producer not assign a stream,
// then return false.
bool GraphMemoryAssigner::IsOutputVisitedByMultiStream(const NodePtr &peer_out_node, int64_t out_anchor_index) const {
  GE_IF_BOOL_EXEC(peer_out_node->GetOpDesc() == nullptr, return true);
  int64_t unique_stream_id = peer_out_node->GetOpDesc()->GetStreamId();

  GE_IF_BOOL_EXEC(peer_out_node->GetOutDataAnchor(out_anchor_index) == nullptr, return true);
  for (const auto &in_data_anchor : peer_out_node->GetOutDataAnchor(out_anchor_index)->GetPeerInDataAnchors()) {
    auto node = in_data_anchor->GetOwnerNode();
    GE_IF_BOOL_EXEC(node == nullptr || node->GetOpDesc() == nullptr, continue);
    if (node->GetOpDesc()->GetStreamId() == kInvalidStream) {
      continue;
    }
    if (unique_stream_id == kInvalidStream) {  // peer_out_node not belong to any stream
      unique_stream_id = node->GetOpDesc()->GetStreamId();
      continue;
    }
    if (node->GetOpDesc()->GetStreamId() != unique_stream_id) {
      return true;
    }
  }
  return false;
}

void GraphMemoryAssigner::UpdatePrevNodeInputDesc(const NodePtr &prev_node,
                                                  const std::vector<int64_t> &prev_node_input_index_vec,
                                                  int64_t distance) const {
  GE_IF_BOOL_EXEC(prev_node == nullptr, return);
  auto prev_node_op_desc = prev_node->GetOpDesc();
  GE_IF_BOOL_EXEC(prev_node_op_desc == nullptr, return);

  for (const auto prev_node_input_index : prev_node_input_index_vec) {
    const auto &input_desc = prev_node_op_desc->MutableInputDesc(prev_node_input_index);
    std::vector<int64_t> prev_next_distances;
    if (!ge::AttrUtils::GetListInt(input_desc, ATTR_NAME_DATA_VISIT_DISTANCE, prev_next_distances)) {
      GELOGW("Get [%s] input [%" PRId64 "] ATTR_NAME_DATA_VISIT_DISTANCE failed", prev_node_op_desc->GetName().c_str(),
             prev_node_input_index);
      continue;
    }

    if (prev_next_distances.size() == kPrevNextDistanceNum) {
      prev_next_distances[1] = distance;
    } else {
      GELOGW("Size of prev_next_distances is not %d.", kPrevNextDistanceNum);
      continue;
    }
    if (!ge::AttrUtils::SetListInt(input_desc, ATTR_NAME_DATA_VISIT_DISTANCE, prev_next_distances)) {
      GELOGW("Set [%s] input [%" PRId64 "] ATTR_NAME_DATA_VISIT_DISTANCE failed.", prev_node_op_desc->GetName().c_str(),
             prev_node_input_index);
      continue;
    }
    GELOGD("Set the next distance[%" PRId64 "] to node[%s], input index[%" PRId64 "]", distance,
           prev_node->GetName().c_str(), prev_node_input_index);
  }
  return;
}

void GraphMemoryAssigner::UpdateCurNodeInputDesc(const NodePtr &cur_node, int64_t cur_node_input_index,
                                                 int64_t distance) const {
  GE_IF_BOOL_EXEC(cur_node == nullptr, return);
  GE_IF_BOOL_EXEC(cur_node->GetOpDesc() == nullptr, return);
  const auto &input_desc = cur_node->GetOpDesc()->MutableInputDesc(cur_node_input_index);
  std::vector<int64_t> prev_next_distances{distance, -1};

  if (!ge::AttrUtils::SetListInt(input_desc, ATTR_NAME_DATA_VISIT_DISTANCE, prev_next_distances)) {
    GELOGW("Set [%s] input[%" PRId64 "] ATTR_NAME_DATA_VISIT_DISTANCE failed.",
           cur_node->GetOpDesc()->GetName().c_str(), cur_node_input_index);
    return;
  }
  GELOGD("Set the prev distance[%" PRId64 "] to node[%s], input index[%" PRId64 "]", distance,
         cur_node->GetName().c_str(), cur_node_input_index);
  return;
}

void GraphMemoryAssigner::CheckNeedCalcDistAndUpdateVisitInfo(
    const NodePtr &peer_out_node, const OutDataAnchorPtr &peer_out_anchor, size_t matched_mem_offset,
    std::unordered_map<size_t, std::pair<NodePtr, std::vector<int64_t>>> &mem_block_visit_info,
    bool &is_need_calc_distance) const {
  const auto iter = mem_block_visit_info.find(matched_mem_offset);
  // cannot find visit info, peer_out_node must be a producer and this data is the first time to be visited.
  if (iter == mem_block_visit_info.end()) {
    if (IsOutputVisitedByMultiStream(peer_out_node, peer_out_anchor->GetIdx())) {
      std::vector<int64_t> temp;
      mem_block_visit_info.insert(std::make_pair(matched_mem_offset, std::make_pair(nullptr, temp)));
      is_need_calc_distance = false;
      return;
    } else {
      std::vector<int64_t> temp = {-1};
      // producer's prev_node_index set to -1 as default
      mem_block_visit_info.insert(std::make_pair(matched_mem_offset, std::make_pair(peer_out_node, temp)));
      is_need_calc_distance = true;
      return;
    }
  } else {
    if (iter->second.first == nullptr) {
      // multi-stream visit, no need to calculate
      is_need_calc_distance = false;
      return;
    }
    if (peer_out_node->GetOpDesc()->GetStreamId() != iter->second.first->GetOpDesc()->GetStreamId()) {
      // cur node and peer_out_node not in the same stream, no need to calculate
      is_need_calc_distance = false;
      return;
    }
  }
  is_need_calc_distance = true;
  return;
}

// calculate distance, update visit info, update prev_node input desc, update cur node input desc
void GraphMemoryAssigner::CalcDistanceAndUpdateDesc(
    const std::unordered_map<std::string, int64_t> &node_index_in_stream, const InDataAnchorPtr &in_data_anchor,
    size_t matched_mem_offset, const NodePtr &node,
    std::unordered_map<size_t, std::pair<NodePtr, std::vector<int64_t>>> &mem_block_visit_info,
    bool &is_need_skip) const {
  int64_t distance = -1;
  auto &visit_info = mem_block_visit_info[matched_mem_offset];
  auto prev_node = visit_info.first;
  auto prev_node_input_index_vec = visit_info.second;
  GE_IF_BOOL_EXEC(prev_node == nullptr, is_need_skip = true; return);
  if (prev_node_input_index_vec.size() == 1 && prev_node_input_index_vec[0] == -1) {
    // prev_node is producer and the data is just be produced(not visited by other node)
    GE_IF_BOOL_EXEC(prev_node->GetOpDesc() == nullptr, is_need_skip = true; return);
    if (prev_node->GetOpDesc()->GetStreamId() == -1) {  // producer not assigned a stream
      distance = 0;
    } else {
      auto iter = node_index_in_stream.find(prev_node->GetName());
      if (iter == node_index_in_stream.end()) {
        distance = 0;
      } else {
        distance = node_index_in_stream.at(node->GetName()) - iter->second - 1;
      }
    }
    visit_info.first = node;
    visit_info.second.clear();
    visit_info.second.push_back(in_data_anchor->GetIdx());
  } else {  // the data is visit by other customer just before.
    if (prev_node_input_index_vec.empty()) {
      GELOGW("Missing prev node[%s] input index.", prev_node->GetName().c_str());
      is_need_skip = true;
      return;
    }
    if (prev_node == node) {  // scene: multiple anchors of a node access the same data
      std::vector<int64_t> prev_next_distances;
      GE_IF_BOOL_EXEC(prev_node->GetOpDesc() == nullptr, is_need_skip = true; return);
      auto input_desc = prev_node->GetOpDesc()->GetInputDesc(prev_node_input_index_vec[0]);
      if (!ge::AttrUtils::GetListInt(input_desc, ATTR_NAME_DATA_VISIT_DISTANCE, prev_next_distances)) {
        GELOGW("Get ATTR_NAME_DATA_VISIT_DISTANCE failed.");
        is_need_skip = true;
        return;
      }
      if (prev_next_distances.size() != kPrevNextDistanceNum) {
        GELOGW("Size of prev_next_distance is not %d.", kPrevNextDistanceNum);
        is_need_skip = true;
        return;
      } else {
        distance = prev_next_distances[0];  // use the same prev_distance as previous anchor
      }
      visit_info.second.push_back(in_data_anchor->GetIdx());
    } else {
      distance = node_index_in_stream.at(node->GetName()) - node_index_in_stream.at(prev_node->GetName()) - 1;
      UpdatePrevNodeInputDesc(prev_node, prev_node_input_index_vec, distance);
      visit_info.first = node;
      visit_info.second.clear();
      visit_info.second.push_back(in_data_anchor->GetIdx());
    }
  }
  UpdateCurNodeInputDesc(node, in_data_anchor->GetIdx(), distance);
}

void GraphMemoryAssigner::DeleteVisitInfoWhenLifecycleEnded(
    const NodePtr &node, const InDataAnchorPtr &in_data_anchor, size_t matched_mem_offset,
    std::unordered_map<size_t, std::pair<NodePtr, std::vector<int64_t>>> &mem_block_visit_info) const {
  GE_IF_BOOL_EXEC(node->GetOpDesc() == nullptr, return);
  auto input_desc = node->GetOpDesc()->GetInputDesc(in_data_anchor->GetIdx());
  bool is_end_of_inputmem_lifecycle = false;
  // if is_end_of_inputmem_lifecycle is true, indicating that cur node is the last customer of this data,
  // then we need to delete the visit info of the block in case that the memblock be reused and visited.
  if (ge::AttrUtils::GetBool(input_desc, ATTR_NAME_IS_END_OF_INPUTMEM_LIFECYCLE, is_end_of_inputmem_lifecycle) &&
      is_end_of_inputmem_lifecycle) {
    GELOGD("ATTR_NAME_IS_END_OF_INPUTMEM_LIFECYCLE is true, node name is [%s], in_data_anchor index is [%d]",
           node->GetName().c_str(), in_data_anchor->GetIdx());
    const auto iter = mem_block_visit_info.find(matched_mem_offset);
    if (iter != mem_block_visit_info.cend()) {
      mem_block_visit_info.erase(iter);
    }
  }
}

void GraphMemoryAssigner::MarkNodeDistanceAttr(
    const NodePtr &node, std::unordered_map<size_t, std::pair<NodePtr, std::vector<int64_t>>> &mem_block_visit_info,
    const std::unordered_map<std::string, int64_t> &node_index_in_stream) {
  GELOGD("Begin to mark node distance attr, node name is [%s]", node->GetName().c_str());
  for (const auto &in_data_anchor : node->GetAllInDataAnchors()) {
    auto peer_out_anchor = in_data_anchor->GetPeerOutAnchor();
    GE_IF_BOOL_EXEC(peer_out_anchor == nullptr, continue);
    auto peer_out_node = peer_out_anchor->GetOwnerNode();
    GE_IF_BOOL_EXEC(peer_out_node == nullptr, continue);

    GE_IF_BOOL_EXEC(peer_out_node->GetOpDesc() == nullptr, continue);
    auto matched_mem_offset = peer_out_node->GetOpDesc()->GetOutputOffset().at(peer_out_anchor->GetIdx());

    bool is_need_calc_distance = false;
    CheckNeedCalcDistAndUpdateVisitInfo(peer_out_node, peer_out_anchor, matched_mem_offset, mem_block_visit_info,
                                        is_need_calc_distance);
    if (!is_need_calc_distance) {
      continue;
    }

    bool is_need_skip = false;
    CalcDistanceAndUpdateDesc(node_index_in_stream, in_data_anchor, matched_mem_offset, node, mem_block_visit_info,
                              is_need_skip);
    if (is_need_skip) {
      continue;
    }

    DeleteVisitInfoWhenLifecycleEnded(node, in_data_anchor, matched_mem_offset, mem_block_visit_info);
  }
}

void GraphMemoryAssigner::MarkDistanceAttr() {
  // key: mem_offset of the memory which we visited. value: node we visited and input index of this node
  std::unordered_map<size_t, std::pair<NodePtr, std::vector<int64_t>>> mem_block_visit_info;
  // key: node name, value: topo order of node in it's belonged stream(exclude ge_local_op)
  std::unordered_map<std::string, int64_t> node_index_in_stream;
  // key: stream id, value: cur nodes num in that stream
  std::unordered_map<int64_t, int64_t> stream_nodes_num;

  for (auto &node : compute_graph_->GetAllNodes()) {
    auto node_op_desc = node->GetOpDesc();
    GE_IF_BOOL_EXEC(node_op_desc == nullptr, return);
    // Only sinking computing nodes need to be calculated, excluding the nodes which don't have task
    if ((node_op_desc->GetOpKernelLibName() != kEngineNameGeLocal) && (!node_op_desc->HasAttr(ATTR_NAME_NOTASK))) {
      int64_t stream_id = node_op_desc->GetStreamId();
      auto stream_count = stream_nodes_num.emplace(stream_id, 0).first;
      ++stream_count->second;
      node_index_in_stream.emplace(node->GetName(), stream_count->second - 1);
      (void)AttrUtils::SetInt(node_op_desc, ATTR_NAME_OP_READ_WRITE_INDEX, stream_count->second - 1);

      MarkNodeDistanceAttr(node, mem_block_visit_info, node_index_in_stream);
    } else {
      GELOGD("node[%s] is ge_local_op, no need to calculate distance.", node->GetName().c_str());
    }
  }
}
}  // namespace ge
