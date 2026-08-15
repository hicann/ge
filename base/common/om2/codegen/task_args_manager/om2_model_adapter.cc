/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "om2_model_adapter.h"
#include <regex>
#include <sstream>
#include "common/compile_profiling/ge_call_wrapper.h"
#include "common/aclrt_malloc_helper.h"
#include "common/file_constant_utils/file_constant_utils.h"
#include "graph/manager/graph_external_weight_manager.h"
#include "framework/common/op/ge_op_utils.h"
#include "graph/ge_context.h"
#include "base/err_mgr.h"
#include "graph/model_serialize.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/op_type_utils.h"
#include "common/checker.h"
#include "runtime/subscriber/global_profiler.h"
#include "graph/debug/ge_attr_define.h"
#include "framework/common/util.h"
#include "om2_model_args_utils.h"
#include "graph/utils/op_desc_utils.h"

namespace ge {
namespace om2 {
namespace {
constexpr uint32_t kDataIndex = 0U;
constexpr uint32_t kAlign32B = 32;
constexpr uint32_t kGetDynamicDimsCount = 1U;
constexpr uint8_t kConstructInputLogicalAllcationLoop = 2;

bool IsInputOfNetoutputCanZeroCopy(const NodePtr &node, const int32_t anchor_idx) {
  if ((node->GetInDataAnchor(anchor_idx) == nullptr) ||
      (node->GetInDataAnchor(anchor_idx)->GetPeerOutAnchor() == nullptr) ||
      (node->GetInDataAnchor(anchor_idx)->GetPeerOutAnchor()->GetOwnerNode() == nullptr) ||
      (node->GetInDataAnchor(anchor_idx)->GetPeerOutAnchor()->GetOwnerNode()->GetOpDesc() == nullptr)) {
    GELOGE(PARAM_INVALID, "[OM2] Peer node of net-output %s input %d is invalid", node->GetName().c_str(), anchor_idx);
    return false;
  }

  const auto src_node = node->GetInDataAnchor(anchor_idx)->GetPeerOutAnchor()->GetOwnerNode();
  const int32_t src_output_index = node->GetInDataAnchor(anchor_idx)->GetPeerOutAnchor()->GetIdx();
  const auto output_desc = src_node->GetOpDesc()->GetOutputDescPtr(static_cast<uint32_t>(src_output_index));

  bool is_zero_copy_block = false;
  const bool determinate =
      (output_desc != nullptr) && AttrUtils::GetBool(output_desc, ATTR_IS_ZERO_COPY_BLOCK, is_zero_copy_block);

  GELOGI("[OM2] Net-output %s input %d from %s output %d can zero copy: %s", node->GetName().c_str(), anchor_idx,
         src_node->GetName().c_str(), src_output_index,
         (determinate ? (is_zero_copy_block ? "true" : "false") : "indeterminate"));

  return is_zero_copy_block;
}
}  // namespace

ModelAdapter::ModelAdapter() = default;

ModelAdapter::~ModelAdapter() noexcept = default;

Status ModelAdapter::Init(const GeModelPtr &ge_model) {
  logLevel_ = dlog_getlevel(GE_MODULE_NAME, nullptr);
  GE_CHK_BOOL_RET_STATUS(ge_model != nullptr, PARAM_INVALID, "[OM2] ge_model is nullptr.");
  ge_model_ = ge_model;
  const ComputeGraphPtr compute_graph = ge_model_->GetGraph();
  GE_CHK_BOOL_RET_STATUS(compute_graph != nullptr, INTERNAL_ERROR, "[OM2] compute_graph is nullptr.");
  version_ = ge_model_->GetVersion();
  name_ = ge_model_->GetName();
  runtime_param_.graph_id = compute_graph->GetGraphID();
  runtime_param_.graph_name = compute_graph->GetName();
  GE_ASSERT_SUCCESS(ModelUtils::InitRuntimeParams(ge_model_, runtime_param_));
  GELOGD("[OM2] InitRuntimeParams: model_name=%s, runtime_param=%s.", name_.c_str(), runtime_param_.ToString().c_str());
  if (ge_model_->GetWeightSize() != 0U) {
    runtime_param_.weight_base = memory_segment_planner_.Allocate(SegmentType::kWeight, ge_model_->GetWeightSize());
  }
  if (runtime_param_.mem_size != 0U) {
    runtime_param_.mem_base = memory_segment_planner_.Allocate(SegmentType::kFeatureMap, runtime_param_.mem_size);
  }
  if (!runtime_param_.fm_memory_infos.empty()) {
    runtime_param_.fm_memory_infos[0U].memory_base =
        PtrToPtr<void, uint8_t>(reinterpret_cast<void *>(runtime_param_.mem_base));
  }
  GELOGD("[OM2] CurrentRuntimeParams: model_name=%s, runtime_param=%s.", name_.c_str(),
         runtime_param_.ToString().c_str());

  GE_CHK_STATUS_RET(InitIoNodes(compute_graph), "[OM2] InitIoNodes failed, model_name: %s", name_.c_str());

  for (const auto node : compute_graph->GetAllNodesPtr()) {
    const auto &op_desc = node->GetOpDesc();
    GE_CHECK_NOTNULL(op_desc);
    op_list_[op_desc->GetId()] = op_desc;
  }
  return SUCCESS;
}

Status ModelAdapter::InitIoNodes(const ComputeGraphPtr &compute_graph) {
  uint32_t data_op_index = 0U;
  std::map<uint32_t, OpDescPtr> index_to_data;
  std::vector<OpDescPtr> output_op_list;
  std::set<uint64_t> input_outside_addrs;
  std::set<uint64_t> output_outside_addrs;
  for (const auto &node : compute_graph->GetAllNodes()) {
    const auto &op_desc = node->GetOpDesc();
    GE_CHECK_NOTNULL(op_desc);
    if (OpTypeUtils::IsDataNode(op_desc->GetType())) {
      GE_CHK_STATUS_RET_NOLOG(InitDataOp(compute_graph, node, data_op_index, index_to_data, input_outside_addrs));
    } else if (op_desc->GetType() == NETOUTPUT) {
      GE_CHK_STATUS_RET_NOLOG(InitNetOutput(compute_graph, node, output_op_list, output_outside_addrs));
    }
  }
  GE_CHK_STATUS_RET_NOLOG(GenInputOutputInfo(index_to_data, output_op_list));
  GE_CHK_STATUS_RET_NOLOG(GenMemAllocations(index_to_data, output_op_list));
  return SUCCESS;
}

Status ModelAdapter::InitDataOp(const ComputeGraphPtr &graph, const NodePtr &node, uint32_t &data_op_index,
                                std::map<uint32_t, OpDescPtr> &index_to_data, std::set<uint64_t> &input_outside_addrs) {
  (void)input_outside_addrs;
  const auto op_desc = node->GetOpDesc();
  if (node->GetOwnerComputeGraph() != graph) {
    GELOGI("[OM2] Skip Data node: %s in subgraph.", op_desc->GetName().c_str());
    return SUCCESS;
  }
  if (node->GetOwnerComputeGraphBarePtr()->GetParentNode() != nullptr) {
    if (std::strcmp(node->GetTypePtr(), REFDATA) == 0) {
      GELOGD("[OM2] Skip RefData node: %s in subgraph %s.", op_desc->GetName().c_str(),
             node->GetOwnerComputeGraphBarePtr()->GetName().c_str());
      return SUCCESS;
    }
  }

  uint32_t data_index = data_op_index++;
  const auto &index_attr = (GraphUtils::FindRootGraph(graph) == graph) ? ATTR_NAME_INDEX : ATTR_NAME_PARENT_NODE_INDEX;
  if (AttrUtils::GetInt(op_desc, index_attr, data_index)) {
    GELOGD("[OM2] Get new index %u, old %u", data_index, data_op_index - 1U);
  }
  GELOGI("[OM2] Init data node: %s, index: %u.", op_desc->GetName().c_str(), data_index);

  const auto &anchor = node->GetOutDataAnchor(0);
  if ((anchor != nullptr) && (anchor->GetFirstPeerAnchor() != nullptr) &&
      (anchor->GetFirstPeerAnchor()->GetOwnerNode() != nullptr)) {
    const auto &node_desc = anchor->GetFirstPeerAnchor()->GetOwnerNode()->GetOpDesc();
    const size_t anchor_idx = static_cast<size_t>(anchor->GetFirstPeerAnchor()->GetIdx());
    std::vector<int64_t> op_max_size;
    if (AttrUtils::GetListInt(node_desc, "_op_max_size", op_max_size) && (op_max_size.size() > anchor_idx)) {
      (void)AttrUtils::SetInt(op_desc, "_op_max_size", op_max_size[anchor_idx]);
    }
  }
  index_to_data[data_index] = op_desc;
  return SUCCESS;
}

Status ModelAdapter::InitNetOutput(const ComputeGraphPtr &graph, const NodePtr &node,
                                   std::vector<OpDescPtr> &output_op_list, std::set<uint64_t> &output_outside_addrs) {
  (void)output_outside_addrs;
  const auto op_desc = node->GetOpDesc();
  if (node->GetOwnerComputeGraph() != graph) {
    GELOGI("[OM2] Skip subgraph NetOutput node: %s.", op_desc->GetName().c_str());
    (void)op_list_.erase(op_desc->GetId());
    (void)operator_list_.erase(op_desc->GetId());
    return SUCCESS;
  }
  output_op_list.push_back(op_desc);
  return SUCCESS;
}

Status ModelAdapter::GenInputOutputInfo(const std::map<uint32_t, OpDescPtr> &index_to_data,
                                        const std::vector<OpDescPtr> &output_op_list) {
  GELOGD("[OM2] Data node size: %zu, NetOutput node size: %zu.", index_to_data.size(), output_op_list.size());
  for (auto &item : index_to_data) {
    const auto output_addrs = ModelUtils::GetOutputAddrsValue(runtime_param_, item.second);
    GELOGD("[OM2] Data node is: %s, output addr size: %zu", item.second->GetName().c_str(), output_addrs.size());
    input_addrs_list_.emplace_back(output_addrs);
    GE_CHK_STATUS_RET(InitInputDescInfo(item.second), "[OM2] InitInputDescInfo failed, node: %s",
                      item.second->GetName().c_str());
  }

  std::vector<std::string> out_node_name;
  (void)AttrUtils::GetListStr(ge_model_, ATTR_MODEL_OUT_NODES_NAME, out_node_name);
  GELOGD("[OM2] Output node size: %zu, out nodes name is: %zu", output_op_list.size(), out_node_name.size());
  for (const auto &op_desc : output_op_list) {
    const auto input_addrs = ModelUtils::GetInputAddrsValue(runtime_param_, op_desc);
    GELOGD("[OM2] NetOutput node is: %s, input addr size: %zu", op_desc->GetName().c_str(), input_addrs.size());
    output_addrs_list_.emplace_back(input_addrs);

    bool getnext_sink_dynamic = false;
    if (AttrUtils::GetBool(op_desc, ATTR_GETNEXT_SINK_DYNMAIC, getnext_sink_dynamic) && getnext_sink_dynamic) {
      GELOGI("[OM2] ATTR_GETNEXT_SINK_DYNMAIC has been set and is true, node: %s", op_desc->GetName().c_str());
      is_getnext_sink_dynamic_ = true;
    }

    std::vector<std::string> shape_info;
    if (AttrUtils::GetListStr(op_desc, ATTR_NAME_DYNAMIC_OUTPUT_DIMS, shape_info)) {
      (void)dynamic_output_shape_info_.insert(dynamic_output_shape_info_.cend(), shape_info.cbegin(),
                                              shape_info.cend());
    }

    if (InitOutputTensorInfo(op_desc) != SUCCESS) {
      return INTERNAL_ERROR;
    }

    GE_CHK_STATUS_RET(InitOutputDescInfo(op_desc, out_node_name), "[OM2][Init][OutputDescInfo] failed, node: %s",
                      op_desc->GetName().c_str());
  }

  return SUCCESS;
}

Status ModelAdapter::GenMemAllocations(const std::map<uint32_t, OpDescPtr> &index_to_data,
                                       const std::vector<OpDescPtr> &output_op_list) {
  GE_ASSERT_SUCCESS(GenSliceOutputMemAllocations(output_op_list));

  GE_ASSERT_SUCCESS(GenFmMemAllocations());

  GE_ASSERT_SUCCESS(GenFixedFmMemAllocations());

  GE_ASSERT_SUCCESS(GenInputMemAllocations(index_to_data));

  GE_ASSERT_SUCCESS(GenOutputMemAllocations(output_op_list));

  MemAllocation not_change_mem_item = {static_cast<uint32_t>(logical_mem_allocations_.size()),
                                       0U,
                                       UINT64_MAX,
                                       MemAllocation::Type::ABSOLUTE,
                                       0U,
                                       kAbsoluteMemType,
                                       0UL,
                                       0UL};
  GELOGI("[OM2][mem allocation][absolute] model name %s, %s.", name_.c_str(), not_change_mem_item.ToString().c_str());
  logical_mem_allocations_.emplace_back(not_change_mem_item);
  return SUCCESS;
}

Status ModelAdapter::GenSliceOutputMemAllocations(const std::vector<OpDescPtr> &output_op_list) {
  int64_t fm_mem_size = 0;
  GE_ASSERT_SUCCESS(GetTotalMemSizeExcludeZeroCopy(fm_mem_size));
  const ComputeGraphPtr compute_graph = ge_model_->GetGraph();
  GE_CHECK_NOTNULL(compute_graph);

  uint32_t output_index = 0U;
  for (const auto &op_desc : output_op_list) {
    const auto node = compute_graph->FindNode(op_desc->GetName());
    GE_CHECK_NOTNULL(node);
    std::vector<uint64_t> mem_types;
    const std::vector<int64_t> input_size_list = ModelUtils::GetInputSize(op_desc);
    const std::vector<uint64_t> virtual_addr_list = ModelUtils::GetInputAddrsValue(runtime_param_, op_desc, mem_types);
    const std::vector<int64_t> v_input_offset = op_desc->GetInputOffset();

    GELOGD("[OM2] NetOutput node is: %s, input size is %zu, virtual_addr size is %zu.", op_desc->GetName().c_str(),
           input_size_list.size(), virtual_addr_list.size());
    GE_ASSERT_EQ(input_size_list.size(), virtual_addr_list.size());
    GE_ASSERT_EQ(virtual_addr_list.size(), mem_types.size());
    GE_ASSERT(virtual_addr_list.size() <= v_input_offset.size());

    size_t actual_output_size = virtual_addr_list.size();
    if (is_getnext_sink_dynamic_) {
      actual_output_size -= kGetDynamicDimsCount;
      GELOGD(
          "[OM2] In getnext sink dynamic scene, output size will minus 1 as GetNextDynamic is not model output, "
          "actual output size:%zu",
          actual_output_size);
    }

    output_index_to_allocation_ids_.resize(actual_output_size, UINT32_MAX);
    for (size_t i = 0UL; i < actual_output_size; ++i) {
      const uint64_t logical_addr = virtual_addr_list[i];
      const int64_t offset = v_input_offset[i];
      if (IsInputOfNetoutputCanZeroCopy(node, static_cast<int32_t>(i)) && (offset >= 0) && (offset < fm_mem_size)) {
        output_data_to_slice_flag_[output_index] = true;
      }

      if (!output_data_to_slice_flag_[output_index]) {
        output_index++;
        continue;
      }

      refreshable_output_index_and_allocation_ids_.emplace_back(
          std::make_pair(output_index, static_cast<uint32_t>(logical_mem_allocations_.size())));
      MemAllocation mem_allocation = {static_cast<uint32_t>(logical_mem_allocations_.size()),
                                      logical_addr,
                                      static_cast<uint64_t>(input_size_list[i]),
                                      MemAllocation::Type::OUTPUT,
                                      output_index,
                                      mem_types[i],
                                      0UL,
                                      0UL};
      GELOGI("[OM2][mem allocation][output][slice] model name %s, %s.", name_.c_str(),
             mem_allocation.ToString().c_str());
      logical_mem_allocations_.emplace_back(mem_allocation);
      output_index_to_allocation_ids_[output_index] = mem_allocation.id;
      zero_copy_output_indexes_.push_back(output_index);
      output_index++;
    }
  }

  return SUCCESS;
}

Status ModelAdapter::GenFmMemAllocations() {
  fm_mem_allocations_start_id_ = logical_mem_allocations_.size();

  for (const auto &mem_info : runtime_param_.fm_memory_infos) {
    refreshable_fm_index_and_allocation_ids_.emplace_back(
        std::make_pair(static_cast<uint32_t>(logical_fm_mem_allocations_size_),
                       static_cast<uint32_t>(logical_mem_allocations_.size())));

    MemAllocation fm_mem_allocation = {static_cast<uint32_t>(logical_mem_allocations_.size()),
                                       PtrToValue(mem_info.memory_base),
                                       static_cast<uint64_t>(mem_info.memory_size),
                                       MemAllocation::Type::FEATURE_MAP,
                                       static_cast<uint32_t>(logical_fm_mem_allocations_size_),
                                       kFmMemType,
                                       0UL,
                                       0UL};
    GELOGI("[OM2][mem allocation][feature map] model name %s, %s.", name_.c_str(),
           fm_mem_allocation.ToString().c_str());
    logical_mem_allocations_.emplace_back(fm_mem_allocation);
    ++logical_fm_mem_allocations_size_;
  }
  return SUCCESS;
}

Status ModelAdapter::GenFixedFmMemAllocations() {
  fixed_fm_mem_allocations_start_id_ = logical_mem_allocations_.size();
  for (const auto &mem_info : runtime_param_.fixed_fm_memory_infos) {
    fixed_fm_index_and_allocation_ids_.emplace_back(
        std::make_pair(static_cast<uint32_t>(logical_fixed_fm_mem_allocations_size_),
                       static_cast<uint32_t>(logical_mem_allocations_.size())));
    MemAllocation fm_mem_allocation = {static_cast<uint32_t>(logical_mem_allocations_.size()),
                                       PtrToValue(mem_info.memory_base),
                                       static_cast<uint64_t>(mem_info.memory_size),
                                       MemAllocation::Type::FIXED_FEATURE_MAP,
                                       static_cast<uint32_t>(logical_fixed_fm_mem_allocations_size_),
                                       kFmMemType,
                                       0UL,
                                       0UL};
    GELOGI("[OM2][mem allocation][fixed feature map] model name %s, %s.", name_.c_str(),
           fm_mem_allocation.ToString().c_str());
    logical_mem_allocations_.emplace_back(fm_mem_allocation);
    ++logical_fixed_fm_mem_allocations_size_;
  }
  return SUCCESS;
}

void ModelAdapter::PrintNoFrozenInputIndexes() {
  std::string input_indexes_nofrozen = "";
  std::string refreshable_ids_nofrozen_str = "";
  for (size_t i = 0; i < zero_copy_input_indexes_no_frozen_.size(); i++) {
    input_indexes_nofrozen += std::to_string(zero_copy_input_indexes_no_frozen_[i]);
    input_indexes_nofrozen += ", ";
    refreshable_ids_nofrozen_str += "(";
    refreshable_ids_nofrozen_str += std::to_string(refreshable_input_index_no_frozen_and_allocation_ids_[i].first);
    refreshable_ids_nofrozen_str += ", ";
    refreshable_ids_nofrozen_str += std::to_string(refreshable_input_index_no_frozen_and_allocation_ids_[i].second);
    refreshable_ids_nofrozen_str += "), ";
  }
  GELOGI("[OM2][Gen][FrozenInputIndexes], zero_copy_input_indexes_no_frozen is: %s", input_indexes_nofrozen.c_str());
  GELOGI("[OM2][Gen][RefreshableFrozenInputIndexes], refreshable_input_index_no_frozen_and_allocation_ids is: %s",
         refreshable_ids_nofrozen_str.c_str());
}

Status ModelAdapter::GenInputMemAllocations(const std::map<uint32_t, OpDescPtr> &index_to_data) {
  GE_ASSERT_SUCCESS(ParseHostInputIndexOption(index_to_data.size()));
  copy_host_input_infos_.clear();
  copy_host_input_infos_.resize(index_to_data.size());

  input_index_to_allocation_ids_.resize(index_to_data.size(), UINT32_MAX);
  uint32_t input_base_allocation_id = logical_mem_allocations_.size();
  for (size_t construct_input_logical_allcation_loop = 0;
       construct_input_logical_allcation_loop < kConstructInputLogicalAllcationLoop;
       construct_input_logical_allcation_loop++) {
    uint32_t input_index = 0U;
    for (const auto &item : index_to_data) {
      if ((construct_input_logical_allcation_loop == 0 && frozen_input_indexes_.count(input_index) == 0) ||
          (construct_input_logical_allcation_loop != 0 && frozen_input_indexes_.count(input_index) != 0)) {
        input_index++;
        continue;
      }
      std::vector<uint64_t> mem_types;
      const auto virtual_addr_list = ModelUtils::GetOutputAddrsValue(runtime_param_, item.second, mem_types);
      const auto output_size_list = ModelUtils::GetOutputSize(item.second);

      GELOGD("[OM2] Data node is: %s, output size is %zu, virtual_addr size is %zu.", item.second->GetName().c_str(),
             output_size_list.size(), virtual_addr_list.size());
      GE_ASSERT_EQ(output_size_list.size(), virtual_addr_list.size());
      GE_ASSERT_EQ(virtual_addr_list.size(), mem_types.size());
      if (virtual_addr_list.empty() || output_size_list.empty()) {
        GELOGE(PARAM_INVALID, "[OM2][Check][Param] Data[%s] failed: output size is %zu, virtual_addr size is %zu.",
               item.second->GetName().c_str(), output_size_list.size(), virtual_addr_list.size());
        return PARAM_INVALID;
      }

      const uint64_t logical_addr = virtual_addr_list[kDataIndex];
      const uint64_t data_size = static_cast<uint64_t>(output_size_list[kDataIndex]);
      MemAllocationAndOffset mem_allocation_and_offset{};
      if (GetMemAllocationByLogicAddr(logical_addr, mem_allocation_and_offset) == SUCCESS) {
        input_indexes_to_copy_info_[input_index] = {static_cast<uint32_t>(mem_allocation_and_offset.id),
                                                    mem_allocation_and_offset.offset, data_size};
        GELOGW(
            "[OM2][mem allocation][input] model_name %s, input_index %u, op_name %s op_type %s not support zero copy, "
            "%s.",
            name_.c_str(), input_index, item.second->GetName().c_str(), item.second->GetType().c_str(),
            input_indexes_to_copy_info_[input_index].ToString().c_str());
        GE_ASSERT_TRUE((item.second->GetType() != REFDATA),
                       "[OM2] model_name %s, input_index %u, op_name %s op_type %s not support zero copy",
                       name_.c_str(), input_index, item.second->GetName().c_str(), item.second->GetType().c_str());
        if (copy_host_input_indexes_.count(input_index) != 0U) {
          GELOGW("[OM2] model_name %s, host_input_index %u, op_name %s op_type %s not support zero copy", name_.c_str(),
                 input_index, item.second->GetName().c_str(), item.second->GetType().c_str());
        }

        input_index++;
        continue;
      }

      refreshable_input_index_and_allocation_ids_.emplace_back(
          std::make_pair(input_index, static_cast<uint32_t>(logical_mem_allocations_.size())));

      uint64_t tensor_size = data_size;
      int64_t size = 0L;
      const OpDescPtr &op_desc = item.second;
      const auto tensor_desc = op_desc->GetOutputDescPtr(kDataIndex);
      if ((tensor_desc != nullptr) && (TensorUtils::GetTensorSizeInBytes(*tensor_desc, size) == GRAPH_SUCCESS)) {
        tensor_size = static_cast<uint64_t>(size);
      }

      MemAllocation mem_allocation = {static_cast<uint32_t>(logical_mem_allocations_.size()),
                                      logical_addr,
                                      data_size,
                                      MemAllocation::Type::INPUT,
                                      input_index,
                                      mem_types[kDataIndex],
                                      0UL,
                                      0UL};
      mem_allocation.tensor_size = tensor_size;
      GELOGI(
          "[OM2][mem allocation][input] model_name %s, input_index %u, op_name %s op_type %s, %s, tensor_size %" PRIu64,
          name_.c_str(), input_index, item.second->GetName().c_str(), item.second->GetType().c_str(),
          mem_allocation.ToString().c_str(), tensor_size);
      logical_mem_allocations_.emplace_back(mem_allocation);
      input_index_to_allocation_ids_[input_index] = mem_allocation.id;
      zero_copy_input_indexes_.push_back(input_index);
      if (copy_host_input_indexes_.count(input_index) > 0U) {
        GE_ASSERT_TRUE((item.second->GetType() != REFDATA),
                       "[OM2] model_name %s, input_index %u, op_name %s op_type %s not support host input index ",
                       name_.c_str(), input_index, item.second->GetName().c_str(), item.second->GetType().c_str());
        CopyHostInputInfo copy_host_input = {};
        copy_host_input.input_index = input_index;
        copy_host_input.tensor_size = tensor_size;
        copy_host_input_infos_[input_index] = std::move(copy_host_input);
        host_input_size_ += tensor_size;
      }

      if (frozen_input_indexes_.count(input_index) == 0) {
        refreshable_input_index_no_frozen_and_allocation_ids_.push_back(std::make_pair(input_index, mem_allocation.id));
        zero_copy_input_indexes_no_frozen_.push_back(input_index);
      }
      input_index++;
    }
  }

  if (host_input_size_ > 0U) {
    host_input_size_ = ge::MemSizeAlign(host_input_size_, kAlign32B);
  }

  no_frozen_input_allocation_base_id_ = frozen_input_indexes_.size() + input_base_allocation_id;
  if (logLevel_ <= DLOG_INFO) {
    PrintNoFrozenInputIndexes();
  }
  return SUCCESS;
}

Status ModelAdapter::GenOutputMemAllocations(const std::vector<OpDescPtr> &output_op_list) {
  uint32_t output_index = 0U;

  for (const auto &op_desc : output_op_list) {
    std::vector<uint64_t> mem_types;
    const std::vector<int64_t> input_size_list = ModelUtils::GetInputSize(op_desc);
    const std::vector<uint64_t> virtual_addr_list = ModelUtils::GetInputAddrsValue(runtime_param_, op_desc, mem_types);

    GELOGD("[OM2] NetOutput node is: %s, input size is %zu, virtual_addr size is %zu.", op_desc->GetName().c_str(),
           input_size_list.size(), virtual_addr_list.size());
    GE_ASSERT_EQ(input_size_list.size(), virtual_addr_list.size());
    GE_ASSERT_EQ(virtual_addr_list.size(), mem_types.size());

    size_t actual_output_size = virtual_addr_list.size();
    if (is_getnext_sink_dynamic_) {
      actual_output_size -= kGetDynamicDimsCount;
      GELOGD(
          "[OM2] In getnext sink dynamic scene, output size will minus 1 as GetNextDynamic is not model output, "
          "actual output size:%zu",
          actual_output_size);
    }

    for (size_t i = 0UL; i < actual_output_size; ++i) {
      int64_t data_size;
      const auto &tensor_desc = op_desc->GetInputDescPtr(static_cast<uint32_t>(i));
      GE_ASSERT_NOTNULL(tensor_desc);
      GE_ASSERT_SUCCESS(TensorUtils::GetTensorSizeInBytes(*tensor_desc, data_size));
      output_indexes_to_tensor_size_[output_index] = static_cast<uint64_t>(data_size);
      const uint64_t logical_addr = virtual_addr_list[i];
      if (output_data_to_slice_flag_[output_index]) {
        output_index++;
        continue;
      }

      MemAllocationAndOffset mem_allocation_and_offset{};
      const auto ret = GetMemAllocationByLogicAddr(logical_addr, mem_allocation_and_offset);
      GELOGI("[OM2][mem allocation][output] model_name=%s, output_index %u, logical_addr 0x%" PRIx64
             " ret %u id %u offset 0x%" PRIx64 ".",
             name_.c_str(), output_index, logical_addr, ret, mem_allocation_and_offset.id,
             mem_allocation_and_offset.offset);
      if ((mem_types[i] == kVarMemType) || (ret == SUCCESS)) {
        uint32_t id = 0xFFFFFFFFU;
        uint64_t offset = logical_addr;
        if (mem_types[i] != kVarMemType) {
          GE_ASSERT_TRUE(ret == SUCCESS, "[OM2] not find 0x%" PRIx64 " in allocating table", logical_addr);
          id = static_cast<uint32_t>(mem_allocation_and_offset.id);
          offset = mem_allocation_and_offset.offset;
        }

        output_indexes_to_copy_info_[output_index] = {id, offset, static_cast<uint64_t>(data_size)};
        GELOGI("[OM2][mem allocation][output] model_name=%s, output_index %u, add output copy info, %s.", name_.c_str(),
               output_index, output_indexes_to_copy_info_[output_index].ToString().c_str());
        output_index++;
        continue;
      }

      refreshable_output_index_and_allocation_ids_.emplace_back(
          std::make_pair(output_index, static_cast<uint32_t>(logical_mem_allocations_.size())));
      MemAllocation mem_allocation = {static_cast<uint32_t>(logical_mem_allocations_.size()),
                                      virtual_addr_list[i],
                                      static_cast<uint64_t>(input_size_list[i]),
                                      MemAllocation::Type::OUTPUT,
                                      output_index,
                                      mem_types[i],
                                      0UL,
                                      0UL};
      GELOGI("[OM2] [mem allocation][output] model_name=%s, %s.", name_.c_str(), mem_allocation.ToString().c_str());
      logical_mem_allocations_.emplace_back(mem_allocation);
      output_index_to_allocation_ids_[output_index] = mem_allocation.id;
      zero_copy_output_indexes_.push_back(output_index);
      output_index++;
    }
  }

  return SUCCESS;
}

Status ModelAdapter::GetMemAllocationByLogicAddr(const uint64_t addr, MemAllocationAndOffset &allocation_info) const {
  for (const auto &item : logical_mem_allocations_) {
    if ((addr >= item.logical_addr) && (addr < (item.logical_addr + item.data_size))) {
      allocation_info.id = static_cast<size_t>(item.id);
      allocation_info.offset = (addr - item.logical_addr);
      return SUCCESS;
    }
  }
  return INTERNAL_ERROR;
}

void ModelAdapter::SetInputDimsInfo(const std::vector<int64_t> &input_dims, const Format format,
                                    ShapeDescription &shape_info) const {
  const size_t n = static_cast<size_t>((format == FORMAT_NHWC) ? NHWC_DIM_N : NCHW_DIM_N);
  const size_t c = static_cast<size_t>((format == FORMAT_NHWC) ? NHWC_DIM_C : NCHW_DIM_C);
  const size_t h = static_cast<size_t>((format == FORMAT_NHWC) ? NHWC_DIM_H : NCHW_DIM_H);
  const size_t w = static_cast<size_t>((format == FORMAT_NHWC) ? NHWC_DIM_W : NCHW_DIM_W);

  if (input_dims.size() == static_cast<size_t>(NORMAL_TENSOR_SIZE)) {
    shape_info.num = input_dims[n];
    shape_info.height = input_dims[h];
    shape_info.width = input_dims[w];
    shape_info.channel = input_dims[c];
  }
  for (size_t k = 0U; k < input_dims.size(); ++k) {
    shape_info.dims.push_back(input_dims[k]);
  }
}

void ModelAdapter::CreateInputDimsInfo(const OpDescPtr &op_desc, const Format format, ShapeDescription &shape_info,
                                       ShapeDescription &dims_info) const {
  GE_CHECK_NOTNULL_JUST_RETURN(op_desc->GetInputDescPtr(0U));
  if (op_desc->HasAttr(ATTR_DYNAMIC_AIPP_INPUT_DIMS)) {
    std::vector<int64_t> dynamic_aipp_input_dims;
    (void)AttrUtils::GetListInt(op_desc, ATTR_DYNAMIC_AIPP_INPUT_DIMS, dynamic_aipp_input_dims);
    SetInputDimsInfo(dynamic_aipp_input_dims, format, shape_info);
  } else {
    if (!op_desc->HasAttr(ATTR_MBATCH_ORIGIN_INPUT_DIMS)) {
      const std::vector<int64_t> input_dims = op_desc->GetInputDescPtr(0U)->GetShape().GetDims();
      SetInputDimsInfo(input_dims, format, shape_info);
    } else {
      std::vector<int64_t> origin_input_dims;
      (void)AttrUtils::GetListInt(op_desc, ATTR_MBATCH_ORIGIN_INPUT_DIMS, origin_input_dims);
      SetInputDimsInfo(origin_input_dims, format, shape_info);
    }
  }

  if (op_desc->HasAttr(ATTR_NAME_INPUT_DIMS)) {
    std::vector<int64_t> model_input_dims;
    (void)AttrUtils::GetListInt(op_desc, ATTR_NAME_INPUT_DIMS, model_input_dims);
    SetInputDimsInfo(model_input_dims, format, dims_info);
  } else {
    dims_info = shape_info;
  }
}

Status ModelAdapter::InitInputDescInfo(const OpDescPtr &op_desc) {
  GE_CHECK_NOTNULL(op_desc->GetInputDescPtr(0U));
  GE_CHECK_NOTNULL(op_desc->GetOutputDescPtr(0U));

  InputOutputDescInfo input;
  input.data_type = op_desc->GetInputDescPtr(0U)->GetDataType();
  input.name = op_desc->GetName();
  int64_t input_size = 0;
  if (AttrUtils::GetInt(*op_desc->GetOutputDescPtr(0U), ATTR_NAME_SPECIAL_INPUT_SIZE, input_size) && (input_size > 0)) {
    GELOGI("[OM2] data[%s] output has special size [%" PRId64 "]", op_desc->GetName().c_str(), input_size);
  } else {
    GE_CHK_STATUS_RET(TensorUtils::GetSize(*op_desc->GetInputDescPtr(0U), input_size),
                      "[OM2][Get][InputSize] failed in op: %s.", op_desc->GetName().c_str());
  }
  input.size = static_cast<uint64_t>(input_size);

  const Format format = op_desc->GetInputDescPtr(0U)->GetFormat();
  const std::vector<int64_t> input_dims = op_desc->GetInputDescPtr(0U)->GetShape().GetDims();
  InputOutputDescInfo origin_input = input;
  SetInputDimsInfo(input_dims, format, origin_input.shape_info);
  origin_input_descs_.push_back(origin_input);
  ShapeDescription dims_info;
  CreateInputDimsInfo(op_desc, format, input.shape_info, dims_info);

  input_formats_.push_back(format);
  input_descs_.push_back(input);

  input.shape_info = dims_info;
  input_descs_dims_.push_back(input);
  return SUCCESS;
}

void ModelAdapter::CreateOutput(const size_t index, const OpDescPtr &op_desc, InputOutputDescInfo &output,
                                uint32_t &format_result) const {
  const auto input_desc = op_desc->GetInputDescPtr(static_cast<uint32_t>(index));
  GE_IF_BOOL_EXEC(
      input_desc == nullptr,
      REPORT_INNER_ERR_MSG("E19999",
                           "[OM2] input_desc index:%zu in op:%s(%s) does not exist, model_name:%s, check invalid",
                           index, op_desc->GetName().c_str(), op_desc->GetType().c_str(), name_.c_str());
      GELOGE(FAILED, "[OM2][Get][InputDescPtr] input_desc index:%zu in op:%s(%s) does not exist, model_name:%s", index,
             op_desc->GetName().c_str(), op_desc->GetType().c_str(), name_.c_str());
      return);
  const auto format = input_desc->GetFormat();
  const auto shape = input_desc->GetShape();

  int64_t dims[] = {1, 1, 1, 1};
  format_result = format;
  if (format == FORMAT_ND) {  // for ND tensor
    for (size_t i = 0U; (i < shape.GetDimNum()) && (i < (sizeof(dims) / sizeof(dims[0]))); ++i) {
      dims[i] = shape.GetDim(i);
    }
  } else {  // FOR FORMAT_NHWC or FORMAT_NCHW
    dims[0] = shape.GetDim(static_cast<size_t>((format == FORMAT_NHWC) ? NHWC_DIM_N : NCHW_DIM_N));  // 0: first dim
    dims[1] = shape.GetDim(static_cast<size_t>((format == FORMAT_NHWC) ? NHWC_DIM_C : NCHW_DIM_C));  // 1: second dim
    dims[2] = shape.GetDim(static_cast<size_t>((format == FORMAT_NHWC) ? NHWC_DIM_H : NCHW_DIM_H));  // 2: third dim
    dims[3] = shape.GetDim(static_cast<size_t>((format == FORMAT_NHWC) ? NHWC_DIM_W : NCHW_DIM_W));  // 3: forth dim
  }
  output.shape_info.num = dims[0];      // 0: first dim
  output.shape_info.channel = dims[1];  // 1: second dim
  output.shape_info.height = dims[2];   // 2: third dim
  output.shape_info.width = dims[3];    // 3: forth dim

  if (input_desc->GetFormat() == FORMAT_FRACTAL_Z) {  // FraczToHWCK
    const int64_t k = shape.GetDim(0U);               // 0: first dim
    const int64_t c = shape.GetDim(1U);               // 1: second dim
    const int64_t h = shape.GetDim(2U);               // 2: third dim
    const int64_t w = shape.GetDim(3U);               // 3: forth dim
    output.shape_info.dims.push_back(h);
    output.shape_info.dims.push_back(w);
    output.shape_info.dims.push_back(c);
    output.shape_info.dims.push_back(k);
    format_result = FORMAT_HWCN;
  } else {
    for (size_t j = 0U; j < shape.GetDimNum(); ++j) {
      output.shape_info.dims.push_back(shape.GetDim(j));
    }
  }

  int64_t tensor_size = 0;
  if (AttrUtils::GetInt(input_desc, ATTR_NAME_SPECIAL_OUTPUT_SIZE, tensor_size) && (tensor_size > 0)) {
    GELOGI("[OM2] netoutput[%s] [%zu]th input has special size [%" PRId64 "]", op_desc->GetName().c_str(), index,
           tensor_size);
  } else {
    (void)TensorUtils::GetTensorSizeInBytes(*input_desc, tensor_size);  // no need to check value
  }
  output.size = static_cast<uint64_t>(tensor_size);
  output.data_type = static_cast<uint32_t>(input_desc->GetDataType());
}

Status ModelAdapter::InitOutputDescInfo(const OpDescPtr &op_desc, const std::vector<std::string> &out_node_name) {
  const size_t out_size = op_desc->GetInputsSize();
  for (size_t i = 0U; i < out_size; ++i) {
    std::string output_name;
    InputOutputDescInfo output;
    uint32_t format_result;
    CreateOutput(i, op_desc, output, format_result);

    const auto src_name = op_desc->GetSrcName();
    const auto src_index = op_desc->GetSrcIndex();
    GE_CHK_BOOL_RET_STATUS((src_name.size() > i) && (src_index.size() > i), INTERNAL_ERROR,
                           "[OM2][Check][Param] construct output failed, as index:%zu >= src name size:%zu, "
                           "or index >= src index size:%zu, op:%s.",
                           i, src_name.size(), src_index.size(), op_desc->GetName().c_str());
    if (out_size == out_node_name.size()) {
      const bool contains_colon = out_node_name[i].find(":") != std::string::npos;
      output_name = contains_colon ? out_node_name[i] : out_node_name[i] + ":" + std::to_string(src_index[i]);
    } else {
      output_name = std::string("output_") + std::to_string(i) + "_" + src_name[i] + "_" + std::to_string(src_index[i]);
    }
    output.name = output_name;
    output_descs_.push_back(output);
    output_formats_.push_back(format_result);
  }

  return SUCCESS;
}

Status ModelAdapter::InitOutputTensorInfo(const OpDescPtr &op_desc) {
  size_t input_num = op_desc->GetInputsSize();
  if (is_getnext_sink_dynamic_) {
    GE_CHECK_GE(input_num, kGetDynamicDimsCount);
    input_num = input_num - kGetDynamicDimsCount;
  }

  for (size_t i = 0U; i < input_num; ++i) {
    int64_t size = 0;
    const auto input_desc = op_desc->GetInputDescPtr(static_cast<uint32_t>(i));
    GE_CHECK_NOTNULL(input_desc);
    const auto ret = TensorUtils::GetTensorSizeInBytes(*input_desc, size);
    if (ret != GRAPH_SUCCESS) {
      REPORT_INNER_ERR_MSG("E19999", "[OM2] Get input TensorSize in op:%s(%s) failed, input_index:%zu, model_name:%s",
                           op_desc->GetName().c_str(), op_desc->GetType().c_str(), i, name_.c_str());
      GELOGE(ret, "[OM2][Get][InputTensorSize] in op:%s(%s) failed, input_index:%zu, model_name:%s",
             op_desc->GetName().c_str(), op_desc->GetType().c_str(), i, name_.c_str());
      return ret;
    }
    const GeShape &shape = input_desc->GetShape();
    bool is_no_tiling = false;
    (void)AttrUtils::GetBool(input_desc, ATTR_NAME_TENSOR_NO_TILING_MEM_TYPE, is_no_tiling);
    GELOGI("[OM2] Output size is %" PRId64 ", output shape is %s, no tiling is %d.", size,
           ToString(shape.GetDims()).c_str(), static_cast<int32_t>(is_no_tiling));
    output_buffer_size_.emplace_back(size);
    output_shape_info_.emplace_back(shape);
    output_no_tiling_flag_.push_back(is_no_tiling);
    if (is_no_tiling) {
      has_no_tiling_output_ = true;
    }
  }

  return SUCCESS;
}

uintptr_t ModelAdapter::MallocDynamicMemory(SegmentType type, const size_t size) {
  return memory_segment_planner_.Allocate(type, size);
}

Status ModelAdapter::GetTotalMemSizeExcludeZeroCopy(int64_t &total_useful_size) {
  if (runtime_param_.mem_size < static_cast<uint64_t>(runtime_param_.zero_copy_size)) {
    REPORT_INNER_ERR_MSG("E19999",
                         "[OM2]total mem size[%" PRIu64
                         "] is less than zero copy size["
                         "%" PRId64 "] ",
                         runtime_param_.mem_size, runtime_param_.zero_copy_size);
    GELOGE(FAILED,
           "[OM2][Check][TotalMemSizeExcludeZeroCopy] failed, total mem size[%" PRIu64
           "] is less than "
           "zero copy size[%" PRId64 "]",
           runtime_param_.mem_size, runtime_param_.zero_copy_size);
    return FAILED;
  }
  total_useful_size = (static_cast<int64_t>(runtime_param_.mem_size) - runtime_param_.zero_copy_size);
  return SUCCESS;
}

bool ModelAdapter::IsFeatureBaseRefreshable() const {
  return feature_base_refreshable_;
}

bool ModelAdapter::GetPhysicalMemoryRefreshable() const {
  return support_extend_memory_full_;
}

Status ModelAdapter::ParseHostInputIndexOption(const size_t input_num) {
  copy_host_input_indexes_.clear();
  string copy_host_inputs;
  (void)ge::GetContext().GetOption(OPTION_EXEC_HOST_INPUT_INDEXES, copy_host_inputs);
  if (copy_host_inputs.empty()) {
    GELOGI("[OM2] host input indexes is empty.");
    return SUCCESS;
  }

  // copy host input indexes: ids(1;2;4;5)
  std::vector<std::string> copy_host_input_vec = StringUtils::Split(copy_host_inputs, ';');
  for (auto &input : copy_host_input_vec) {
    int32_t input_index;
    GE_ASSERT_SUCCESS(ConvertToInt32(input, input_index));
    GE_ASSERT_TRUE((input_index >= 0) && static_cast<uint32_t>(input_index) < input_num,
                   "[OM2] host input index:%d no less than input num:%zu", input_index, input_num);
    GELOGI("[OM2] model name:%s, host input index:%d", name_.c_str(), input_index);
    (void)copy_host_input_indexes_.insert(input_index);
  }

  return SUCCESS;
}
}  // namespace om2
}  // namespace ge
