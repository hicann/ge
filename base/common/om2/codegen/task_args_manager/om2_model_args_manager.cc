/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "om2_model_args_manager.h"
#include <algorithm>
#include "common/checker.h"
#include "common/dump/kernel_tracing_utils.h"
#include "common/compile_profiling/ge_call_wrapper.h"
#include "framework/common/op/ge_op_utils.h"
#include "graph/ge_context.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/tensor_utils_ex.h"
#include "om2_memory_app_type_classifier.h"
#include "om2_model_args_layout_planner.h"
#include "om2_task_args_refresh_type_classifier.h"
#include "om2_task_node_map.h"
#include "common/om2/codegen/task_code_builder/task_code_builder.h"

namespace ge {
namespace om2 {
Status PlanFixedMemoryLayout(const TaskNodeMap &task_node_map,
                             const TaskArgsRefreshTypeClassifier::FixedAddrs &fixed_addrs, int64_t &total_len,
                             std::vector<int64_t> &offsets) {
  offsets.resize(fixed_addrs.size());
  for (size_t i = 0U; i < fixed_addrs.size(); ++i) {
    offsets[i] = total_len;

    const auto &fixed_addr = fixed_addrs[i].at(0);
    auto &node_info = task_node_map.FindNodeByTaskIndex(fixed_addr.task_index);
    GE_ASSERT_TRUE(node_info.node_id != -1);
    const auto op_desc = node_info.node->GetOpDesc();

    switch (fixed_addr.iow_index_type) {
      case TaskArgsRefreshTypeClassifier::kInput: {
        const auto td = op_desc->GetInputDescPtr(static_cast<uint32_t>(fixed_addr.iow_index));
        GE_ASSERT_NOTNULL(td, "[OM2] Failed to calculate fixed address for task %zu, op %s, null input, index %zu",
                          fixed_addr.task_index, op_desc->GetName().c_str(), fixed_addr.iow_index);
        int64_t size{0};
        GE_ASSERT_GRAPH_SUCCESS(TensorUtilsEx::GetTensorMemorySizeInBytesWithAutoPadding(*td, size));
        GE_ASSERT_TRUE(!AddOverflow(total_len, size, total_len));
        break;
      }
      case TaskArgsRefreshTypeClassifier::kOutput: {
        const auto td = op_desc->GetOutputDescPtr(static_cast<uint32_t>(fixed_addr.iow_index));
        GE_ASSERT_NOTNULL(td, "[OM2] Failed to calculate fixed address for task %zu, op %s, null output, index %zu",
                          fixed_addr.task_index, op_desc->GetName().c_str(), fixed_addr.iow_index);
        int64_t size{0};
        GE_ASSERT_GRAPH_SUCCESS(TensorUtilsEx::GetTensorMemorySizeInBytesWithAutoPadding(*td, size));
        GE_ASSERT_TRUE(!AddOverflow(total_len, size, total_len));
        break;
      }
      case TaskArgsRefreshTypeClassifier::kWorkspace: {
        auto ws_sizes = op_desc->GetWorkspaceBytes();
        GE_ASSERT_TRUE(
            fixed_addr.iow_index < ws_sizes.size(),
            "[OM2] Failed to calculate fixed address for task %zu, op %s, workspace index out of range %zu, max %zu",
            fixed_addr.task_index, op_desc->GetName().c_str(), fixed_addr.iow_index, ws_sizes.size());
        GE_ASSERT_TRUE(!AddOverflow(total_len, ws_sizes.at(fixed_addr.iow_index), total_len));
        break;
      }
      default:
        GELOGE(INTERNAL_ERROR, "[OM2] Failed to calculate fixed address for task %zu, op %s, unexpected iow type %d",
               fixed_addr.task_index, op_desc->GetName().c_str(), static_cast<int32_t>(fixed_addr.iow_index_type));
        return FAILED;
    }
  }
  return SUCCESS;
}
void DebugLogTaskRunParam(const size_t task_index, const int64_t op_index, const TaskRunParam &param,
                          const OpDescPtr &op_desc) {
  std::stringstream ss;
  ss << "Task index " << task_index << " op index " << op_index << ", args num " << param.args_descs.size() << ',';
  if (!param.args_descs.empty()) {
    ss << " len/placement: ";
    for (const auto &args_desc : param.args_descs) {
      ss << args_desc.args_len << '/' << GetArgsPlacementStr(args_desc.placement) << ',';
    }
  }

  ss << " inputs num " << param.parsed_input_addrs.size() << ',' << " outputs num " << param.parsed_output_addrs.size()
     << ',' << " workspaces num " << param.parsed_workspace_addrs.size() << ',' << " persistent workspaces num "
     << param.persistent_workspace_descs.size() << ',';
  if (!param.persistent_workspace_descs.empty()) {
    ss << " len/placement: ";
    for (const auto &pw_desc : param.persistent_workspace_descs) {
      ss << pw_desc.args_len << '/' << GetArgsPlacementStr(pw_desc.placement) << ',';
    }
  }

  if (op_desc != nullptr) {
    ss << " op type " << op_desc->GetType().c_str() << ',' << " op name " << op_desc->GetName().c_str() << '.';
  }
  GELOGD("[OM2] DebugLogTaskRunParam: %s", ss.str().c_str());
}
constexpr const char *kUpdatePolicyStr[ModelArgsManager::kUpdatePolicyEnd + 1] = {
    "no_need_update",   // kNoNeedUpdate
    "host_input",       // KUpdateHostInput
    "model-io",         // kUpdateModelIo
    "fm-and-model-io",  // kUpdateFmAndModelIo
    "all-one-time",     // kInitOneTime
    "unknown"};
const char *GetUpdatePolicyStr(ModelArgsManager::UpdatePolicy up) {
  if (up > ModelArgsManager::kUpdatePolicyEnd) {
    up = ModelArgsManager::kUpdatePolicyEnd;
  }
  return kUpdatePolicyStr[up];
}

void UseMin(uint64_t new_dev_addr, void *new_host_addr, uint64_t &dev_addr, void *&host_addr) {
  if (dev_addr > new_dev_addr) {
    dev_addr = new_dev_addr;
    host_addr = new_host_addr;
  }
}

ModelArgsManager::ModelArgsManager() = default;

ModelArgsManager::~ModelArgsManager() noexcept = default;

Status ModelArgsManager::Init(const GeModelPtr &model, const std::vector<TaskCodeBuilderPtr> *task_list_ptr) {
  logLevel_ = dlog_getlevel(GE_MODULE_NAME, nullptr);
  GE_ASSERT_NOTNULL(model);
  const auto &model_task_def = model->GetModelTaskDefPtr();
  GE_ASSERT_NOTNULL(model_task_def);

  GE_ASSERT_NOTNULL(task_list_ptr);
  task_list_ptr_ = task_list_ptr;
  if (static_cast<size_t>(model_task_def->task_size()) != task_list_ptr_->size()) {
    GELOGE(INTERNAL_ERROR, "[OM2] mode_task_def size do not match task_list size");
    return FAILED;
  }
  GE_ASSERT_SUCCESS(model_adapter_.Init(model));
  return InitTaskInfoV2(*model_task_def);
}

Status ModelArgsManager::GenModelArgsRefreshInfosForTask(std::vector<TaskArgsRefreshInfo> &infos,
                                                         PisToArgs &pls_to_args, const NodePtr &node) {
  for (const auto &info : infos) {
    ModelArgsRefreshInfo m_info;
    const size_t pls = static_cast<size_t>(info.placement);
    m_info.id = info.id;
    m_info.offset = info.offset;
    m_info.placement = info.placement;
    GE_ASSERT_TRUE(info.placement < ArgsPlacement::kEnd);
    GE_ASSERT_TRUE(info.args_offset < static_cast<uint64_t>(pls_to_args[pls].len),
                   "[OM2] op_name:%s, op_type:%s, args offset:%" PRIu64
                   " is more than pls:%zu, len:%d, task args refresh info:[%s]",
                   node->GetOpDesc()->GetName().c_str(), node->GetOpDesc()->GetType().c_str(), info.args_offset, pls,
                   pls_to_args[pls].len, info.ToString().c_str());
    GE_ASSERT_TRUE(pls_to_args[pls].host_addr != nullptr);
    m_info.host_args_addr = ValueToPtr(PtrToValue(pls_to_args[pls].host_addr) + info.args_offset);
    m_info.device_args_addr = pls_to_args[pls].dev_addr + info.args_offset;
    m_info.base_args_offset = PtrToValue(pls_to_args[pls].host_addr) + info.args_offset -
                              PtrToValue(model_args_[pls].model_args_host_addr.get());
    GELOGI(
        "[OM2][Args][Init] op_name:%s, op_type:%s, pls:%zu, pls host addr:0x%llx, pls dev addr:0x%llx, "
        "task args refresh info:[%s], after transfer, model args refresh info:[%s].",
        node->GetOpDesc()->GetName().c_str(), node->GetOpDesc()->GetType().c_str(), pls,
        PtrToValue(pls_to_args[pls].host_addr), pls_to_args[pls].dev_addr, info.ToString().c_str(),
        m_info.ToString().c_str());
    if (info.args_format_policy == ArgsFormatPolicy::kAddrAll) {
      allocation_ids_to_model_args_refresh_infos_addr_all[m_info.id].emplace_back(std::move(m_info));
    } else if (info.args_format_policy == ArgsFormatPolicy::kAddrLow32Bit) {
      allocation_ids_to_model_args_refresh_infos_addr_low_32bit[m_info.id].emplace_back(std::move(m_info));
    } else if (info.args_format_policy == ArgsFormatPolicy::kAddrHigh32Bit) {
      allocation_ids_to_model_args_refresh_infos_addr_high_32bit[m_info.id].emplace_back(std::move(m_info));
    }
  }
  return SUCCESS;
}

Status ModelArgsManager::InitTaskInfoV2(domi::ModelTaskDef &model_task_def) {
  if (model_task_def.task_size() == 0) {
    GELOGW("[OM2] No task defs in model task def");
    return SUCCESS;
  }
  GELOGI("[OM2] Begin to init all task info, task count %zu", model_task_def.task_size());
  allocation_ids_to_model_args_refresh_infos_addr_all.resize(model_adapter_.GetLogicalMemAllocation().size());
  allocation_ids_to_model_args_refresh_infos_addr_low_32bit.resize(model_adapter_.GetLogicalMemAllocation().size());
  allocation_ids_to_model_args_refresh_infos_addr_high_32bit.resize(model_adapter_.GetLogicalMemAllocation().size());
  const size_t task_size = static_cast<size_t>(model_task_def.task_size());
  std::vector<TaskRunParam> task_indexes_to_run_param(task_size);
  TaskNodeMap task_node_map;
  GE_ASSERT_SUCCESS(task_node_map.Init(model_adapter_.GetCompiledComputeGraph(), task_size));
  GE_ASSERT_SUCCESS(ParseModelTaskDef(model_task_def, task_indexes_to_run_param, task_node_map));
  const auto logical_addrs_to_memory_type =
      MemoryAppTypeClassifier(model_adapter_.GetLogicalMemAllocation(), model_adapter_.GetFmMemAllocationsStartId())
          .ClassifyByTaskRunParams(task_indexes_to_run_param);
  std::vector<TaskArgsRefreshTypeClassifier::TaskRefreshType> task_indexes_to_refresh_type;
  TaskArgsRefreshTypeClassifier::FixedAddrs fixed_addrs;
  GE_ASSERT_SUCCESS(TaskArgsRefreshTypeClassifier(task_node_map, logical_addrs_to_memory_type,
                                                  model_adapter_.IsFeatureBaseRefreshable())
                        .ClassifyMultiTasks(task_indexes_to_run_param, task_indexes_to_refresh_type, fixed_addrs,
                                            model_adapter_.GetPhysicalMemoryRefreshable()));
  ModelArgsLayoutPlannedResult planned_model_args_layout_result;
  GE_ASSERT_SUCCESS(ModelArgsLayoutPlanner(task_indexes_to_refresh_type, task_indexes_to_run_param, host_input_size_)
                        .Plan(planned_model_args_layout_result, AddrUseFor::kAddrUseForArgs));
  GE_ASSERT_SUCCESS(
      AllocModelArgs(planned_model_args_layout_result, model_args_, model_args_len_, op_refresh_placement_));
  GE_ASSERT_SUCCESS(ConstructUpdateData(task_node_map, planned_model_args_layout_result, task_indexes_to_run_param,
                                        task_indexes_to_args_));
  GE_ASSERT_SUCCESS(AllocFixedAddrs(task_node_map, fixed_addrs));
  std::vector<IowAddrs> task_indexes_to_init_param;
  GE_ASSERT_SUCCESS(ConstructTaskInitParams(task_indexes_to_refresh_type, logical_addrs_to_memory_type,
                                            std::move(task_indexes_to_run_param), task_indexes_to_init_param));
  for (size_t i = 0UL; i < task_list_ptr_->size(); ++i) {
    const auto task_info = task_list_ptr_->at(i);
    GE_ASSERT_SUCCESS(
        task_info->Init(model_task_def.task(static_cast<int32_t>(i)), model_adapter_.GetLogicalMemAllocation(),
                        task_indexes_to_args_.at(i), task_indexes_to_init_param.at(i)),
        "Failed to init task index %zu, related node %s", i,
        task_node_map.FindNodeByTaskIndex(i).node->GetName().c_str());
    std::vector<TaskArgsRefreshInfo> infos;
    GE_ASSERT_SUCCESS(task_info->GetTaskArgsRefreshInfos(infos),
                      "Failed to get task args refresh infos, task index %zu, related node %s", i,
                      task_node_map.FindNodeByTaskIndex(i).node->GetName().c_str());
    GE_ASSERT_SUCCESS(
        GenModelArgsRefreshInfosForTask(infos, task_indexes_to_args_[i], task_node_map.FindNodeByTaskIndex(i).node));
    if (update_version_ != 1) {
      InitForUpdate();
    }
  }
  return SUCCESS;
}

void ModelArgsManager::InitForUpdate() {
  const size_t size = model_adapter_.GetLogicalMemAllocation().size();
  last_bases_.resize(size, UINT64_MAX);
  id_to_plicy_.resize(size);

  id_to_len_.resize(size);
  const auto logical_mem_allocations = model_adapter_.GetLogicalMemAllocation();
  for (size_t id = 0U; id < size; id++) {
    id_to_len_[id] = logical_mem_allocations[id].data_size;
  }

  const uint32_t absolute_mem_id = static_cast<uint32_t>(size - 1U);
  id_to_plicy_[absolute_mem_id] = static_cast<uint32_t>(kInitOneTime);

  const size_t fm_start_id = model_adapter_.GetFmMemAllocationsStartId();
  const size_t fm_size = model_adapter_.GetFmMemAllocationsSize();
  for (size_t id = 0U; id < absolute_mem_id; id++) {
    if ((id >= fm_start_id) && (id < (fm_start_id + fm_size))) {
      id_to_plicy_[id] = static_cast<uint32_t>(kUpdateFmAndModelIo);
    } else {
      id_to_plicy_[id] = static_cast<uint32_t>(kUpdateModelIo);
    }
  }
}

Status ModelArgsManager::AllocModelArgs(const ModelArgsLayoutPlannedResult &layout, std::vector<ModelArgs> &model_args,
                                        std::vector<uint64_t> &model_args_len, ArgsPlacement &pls) {
  model_args.reserve(static_cast<size_t>(ArgsPlacement::kEnd));
  for (size_t pli = 0; pli < static_cast<size_t>(ArgsPlacement::kEnd); ++pli) {
    int64_t len = 0;
    ModelArgs placed_model_args;
    placed_model_args.placement = static_cast<ArgsPlacement>(pli);
    for (size_t pai = 0; pai < static_cast<size_t>(UpdateTriggerType::kEnd); ++pai) {
      const auto partition_len = layout.placements_to_partitions_to_len[pli][pai];
      if (partition_len == 0) {
        continue;
      }
      if ((pli == static_cast<size_t>(ArgsPlacement::kArgsPlacementHbm)) &&
          (pai == static_cast<size_t>(UpdateTriggerType::KTriggerByHostInput))) {
        host_input_partition_len_ = partition_len;
      }
      placed_model_args.model_args_partitions.push_back({static_cast<UpdateTriggerType>(pai), len, partition_len});
      GE_ASSERT_TRUE(!AddOverflow(len, partition_len, len));
    }
    const size_t built_in_len = static_cast<size_t>(len);
    const size_t reserved_len = 0UL;
    if (built_in_len == 0UL) {
      continue;
    }
    size_t total_len = 0UL;
    GE_ASSERT_TRUE(!AddOverflow(built_in_len, reserved_len, total_len));
    placed_model_args.model_args_host_addr = ge::MakeUnique<uint8_t[]>(total_len);
    GE_ASSERT_NOTNULL(placed_model_args.model_args_host_addr, "Failed to alloc args %zu at host, total_len %zu", pli,
                      total_len);
    SegmentType segment_type = SegmentType::kHbmArgs;
    if (placed_model_args.placement == ArgsPlacement::kArgsPlacementTs) {
      segment_type = SegmentType::kTsArgs;
    } else if (placed_model_args.placement == ArgsPlacement::kArgsPlacementSqe) {
      segment_type = SegmentType::kSqeArgs;
    } else if (placed_model_args.placement == ArgsPlacement::kArgsPlacementHostSvm) {
      segment_type = SegmentType::kHostSvmArgs;
    }
    const auto model_args_device_addr = model_adapter_.MallocDynamicMemory(segment_type, total_len);
    placed_model_args.model_args_device_addr = model_args_device_addr;
    GELOGI("[OM2] Alloc model args built_in=%zu, reserved=%zu, placement=%s, addr=0x%llx for model_name=%s",
           built_in_len, reserved_len, GetArgsPlacementStr(placed_model_args.placement),
           placed_model_args.model_args_device_addr, model_adapter_.GetOmName().c_str());
    model_args.emplace_back(std::move(placed_model_args));
    model_args_len.emplace_back(static_cast<size_t>(len));
    pls = placed_model_args.placement;
  }
  return SUCCESS;
}

Status ModelArgsManager::ConstructUpdateData(const TaskNodeMap &task_node_map,
                                             const ModelArgsLayoutPlannedResult &layout,
                                             const std::vector<TaskRunParam> &task_indexes_to_param,
                                             std::vector<PisToArgs> &task_indexes_to_args) {
  const bool need_debug_log = IsLogEnable(GE_MODULE_NAME, DLOG_DEBUG);
  auto trigger_types_to_update_policies = GenerateTriggerTypesToCorrespondingUpdatePolicies();
  std::array<const ModelArgs *, static_cast<size_t>(ArgsPlacement::kEnd)> pis_to_model_args{nullptr};
  for (const auto &placed_model_arg : model_args_) {
    pis_to_model_args[static_cast<size_t>(placed_model_arg.placement)] = &placed_model_arg;
  }
  const auto task_size = layout.task_indexes_to_arg_results.size();
  task_indexes_to_args.resize(task_size);
  for (size_t i = 0U; i < task_size; ++i) {
    const auto &task_arg_results = layout.task_indexes_to_arg_results[i];
    if (task_arg_results.empty()) {
      continue;
    }
    OneTaskUpdateData one_task_update_data{{i, task_list_ptr_->at(i).get(), {}}, false, {}, &task_indexes_to_args};
    GE_ASSERT_SUCCESS(ConstructOneTaskUpdateData(i, task_arg_results, task_indexes_to_param, pis_to_model_args,
                                                 one_task_update_data, AddrUseFor::kAddrUseForArgs));
    const auto &upis = trigger_types_to_update_policies.at(static_cast<size_t>(task_arg_results.at(0).trigger_type));
    if (need_debug_log) {
      DebugLogTaskUpdatePolicies(task_node_map, upis, i);
    }
    GE_ASSERT_SUCCESS(AddToTaskUpdateDataToPolicies(i, upis, one_task_update_data));
  }
  if (host_input_size_ > 0U) {
    update_policies_to_model_data_[KUpdateHostInput] = MakeUnique<ArgsUpdateData>();
    GE_ASSERT_NOTNULL(update_policies_to_model_data_[KUpdateHostInput]);
  }
  for (int32_t i = 0; i < kUpdatePolicyEnd; ++i) {
    const auto model_update_data = update_policies_to_model_data_[static_cast<size_t>(i)].get();
    if (model_update_data == nullptr) {
      continue;
    }
    for (const auto &model_arg : model_args_) {
      H2DCopyArg cp_arg{};
      const auto ret = ConstructH2DCopyParams(model_arg, static_cast<UpdatePolicy>(i), cp_arg);
      if (ret == GE_GRAPH_GRAPH_NOT_EXIST) {
        continue;
      } else if (ret == SUCCESS) {
        model_update_data->h2d_copy_datas.emplace_back(cp_arg);
      } else {
        return ret;
      }
    }
  }
  return SUCCESS;
}

void ModelArgsManager::DebugLogTaskUpdatePolicies(const TaskNodeMap &task_node_map, const TriggerPolicies &upis,
                                                  size_t task_index) const {
  std::stringstream ss;
  for (const auto upi : upis) {
    ss << GetUpdatePolicyStr(upi) << ",";
  }
  std::string node_name = "unknown";
  auto node_info = task_node_map.FindNodeByTaskIndex(task_index);
  if (node_info.node != nullptr) {
    node_name = node_info.node->GetName();
  }
  GELOGD("[OM2] The args of node %s task index %zu will be updated in policies %s", node_name.c_str(), task_index,
         ss.str().c_str());
}

Status ModelArgsManager::ConstructOneTaskUpdateData(
    const size_t task_index, const OneTaskArgsLayoutResult &task_arg_results,
    const std::vector<TaskRunParam> &task_indexes_to_param,
    const std::array<const ModelArgs *, static_cast<size_t>(ArgsPlacement::kEnd)> &pis_to_model_args,
    OneTaskUpdateData &task_update_data, const AddrUseFor addr_use_for) const {
  for (size_t j = 0UL; j < task_arg_results.size(); ++j) {
    const auto &task_arg_ret = task_arg_results[j];
    auto &args_desc = (addr_use_for == AddrUseFor::kAddrUseForArgs)
                          ? task_indexes_to_param[task_index].args_descs[j]
                          : task_indexes_to_param[task_index].persistent_workspace_descs[j];
    const auto store_placement = task_arg_ret.placement;
    const auto require_placement = args_desc.placement;
    const auto placed_model_args = pis_to_model_args[static_cast<size_t>(store_placement)];

    void *host_addr = nullptr;
    uint64_t device_addr = 0UL;
    uint64_t offset = 0UL;
    if (placed_model_args != nullptr) {
      host_addr = placed_model_args->model_args_host_addr.get() + task_arg_ret.offset;
      device_addr = placed_model_args->model_args_device_addr + static_cast<uint64_t>(task_arg_ret.offset);
      offset = static_cast<uint64_t>(task_arg_ret.offset);
    }

    task_update_data.update_data.host_args.emplace_back(HostArg{host_addr, args_desc.args_len, require_placement});
    (*task_update_data.task_indexes_to_args)[task_index][static_cast<size_t>(require_placement)] = {
        device_addr, host_addr, args_desc.args_len, offset};

    if (require_placement == ArgsPlacement::kArgsPlacementSqe) {
      GE_ASSERT_TRUE(!task_update_data.has_sqe_placement,
                     "[OM2] More than one placement-sqe tasks found in task %zu, not support yet", task_index);
      task_update_data.has_sqe_placement = true;
      task_update_data.sqe_update_arg.stream_id = std::numeric_limits<uint32_t>::max();
      task_update_data.sqe_update_arg.task_id = std::numeric_limits<uint32_t>::max();
      task_update_data.sqe_update_arg.dev_addr = device_addr;
      task_update_data.sqe_update_arg.len = static_cast<uint64_t>(args_desc.args_len);
    }
  }
  return SUCCESS;
}

Status ModelArgsManager::AddToTaskUpdateDataToPolicies(
    const size_t task_index,
    const SmallVector<ModelArgsManager::UpdatePolicy, ModelArgsManager::kUpdatePolicyEnd> &upis,
    const OneTaskUpdateData &one_task_update_data) {
  for (const auto upi : upis) {
    GE_ASSERT_TRUE(
        upi < kUpdatePolicyEnd,
        "[OM2] Failed to construct update data, found trigger by fm partition when fm refresh disabled, task index %zu",
        task_index);
    if (update_policies_to_model_data_[upi] == nullptr) {
      update_policies_to_model_data_[upi] = MakeUnique<ArgsUpdateData>();
      GE_ASSERT_NOTNULL(update_policies_to_model_data_[upi]);
    }
    auto model_update_data = update_policies_to_model_data_[upi].get();
    model_update_data->update_datas.emplace_back(one_task_update_data.update_data);
  }
  return SUCCESS;
}

Status ModelArgsManager::ConstructH2DCopyParams(const ModelArgs &model_arg, const ModelArgsManager::UpdatePolicy up,
                                                ModelArgsManager::H2DCopyArg &cp_arg) {
  switch (up) {
    case KUpdateHostInput: {
      for (const auto &partition : model_arg.model_args_partitions) {
        if (partition.partition_type == UpdateTriggerType::KTriggerByHostInput) {
          cp_arg.len = static_cast<uint64_t>(partition.len);
          cp_arg.device_addr = model_arg.model_args_device_addr + static_cast<uint64_t>(partition.offset);
          cp_arg.host_addr =
              ValueToPtr(PtrToValue(model_arg.model_args_host_addr.get()) + static_cast<uint64_t>(partition.offset));
          return SUCCESS;
        }
      }
      return GE_GRAPH_GRAPH_NOT_EXIST;
    }
    case kUpdateModelIo: {
      bool has_partition = false;
      cp_arg.len = 0UL;
      cp_arg.device_addr = std::numeric_limits<uint64_t>::max();
      for (const auto &partition : model_arg.model_args_partitions) {
        if ((partition.partition_type == UpdateTriggerType::kTriggerByFmAndIo) ||
            (partition.partition_type == UpdateTriggerType::KTriggerByHostInput)) {
          cp_arg.len += static_cast<uint64_t>(partition.len);
          UseMin(model_arg.model_args_device_addr + static_cast<uint64_t>(partition.offset),
                 ValueToPtr(PtrToValue(model_arg.model_args_host_addr.get()) + static_cast<uint64_t>(partition.offset)),
                 cp_arg.device_addr, cp_arg.host_addr);
          has_partition = true;
        }
      }
      return has_partition ? SUCCESS : GE_GRAPH_GRAPH_NOT_EXIST;
    }
    case kUpdateFmAndModelIo: {
      bool has_partition = false;
      cp_arg.len = 0UL;
      cp_arg.device_addr = std::numeric_limits<uint64_t>::max();
      for (const auto &partition : model_arg.model_args_partitions) {
        if ((partition.partition_type == UpdateTriggerType::kTriggerByFmAndIo) ||
            (partition.partition_type == UpdateTriggerType::kTriggerByFm) ||
            (partition.partition_type == UpdateTriggerType::KTriggerByHostInput)) {
          cp_arg.len += static_cast<uint64_t>(partition.len);
          UseMin(model_arg.model_args_device_addr + static_cast<uint64_t>(partition.offset),
                 ValueToPtr(PtrToValue(model_arg.model_args_host_addr.get()) + static_cast<uint64_t>(partition.offset)),
                 cp_arg.device_addr, cp_arg.host_addr);
          has_partition = true;
        }
      }
      return has_partition ? SUCCESS : GE_GRAPH_GRAPH_NOT_EXIST;
    }
    case kInitOneTime:
      cp_arg.len = 0UL;
      cp_arg.device_addr = model_arg.model_args_device_addr;
      cp_arg.host_addr = model_arg.model_args_host_addr.get();
      for (const auto &partition : model_arg.model_args_partitions) {
        cp_arg.len += static_cast<uint64_t>(partition.len);
      }
      GE_ASSERT_TRUE(cp_arg.len > 0UL, "[OM2] Placement %s does not have a partition",
                     GetArgsPlacementStr(model_arg.placement));
      return SUCCESS;
    default:
      GELOGE(INTERNAL_ERROR, "[OM2] unexpected update policy %d", static_cast<int32_t>(up));
      return FAILED;
  }
}

Status ModelArgsManager::AllocFixedAddrs(const TaskNodeMap &task_node_map,
                                         const TaskArgsRefreshTypeClassifier::FixedAddrs &fixed_addrs) {
  std::vector<int64_t> offsets;
  int64_t total_len = 0;
  GE_ASSERT_SUCCESS(PlanFixedMemoryLayout(task_node_map, fixed_addrs, total_len, offsets));
  if (total_len == 0) {
    GELOGD("[OM2] No need to alloc fixed memory for model %s", model_adapter_.GetOmName().c_str());
    return SUCCESS;
  }

  fixed_addr_bulk_.device_addr = model_adapter_.MallocDynamicMemory(SegmentType::kInferFeatureMap, total_len);
  GELOGI("Alloc fixed memory size %lld, addr=0x%llx for model %s", total_len, fixed_addr_bulk_.device_addr,
         model_adapter_.GetOmName().c_str());
  fixed_addr_bulk_.pieces.reserve(offsets.size() * 2UL);
  for (size_t i = 0U; i < offsets.size(); ++i) {
    for (const auto &fixed_addr : fixed_addrs.at(i)) {
      fixed_addr_bulk_.pieces.push_back({fixed_addr, fixed_addr_bulk_.device_addr + static_cast<uint64_t>(offsets[i])});
    }
  }
  return SUCCESS;
}

Status ModelArgsManager::ConstructTaskInitParams(
    const std::vector<TaskArgsRefreshTypeClassifier::TaskRefreshType> &task_indexes_to_refresh_type,
    const std::map<std::pair<uint64_t, uint64_t>, MemoryAppType> &logical_addrs_to_mem_app_type,
    std::vector<TaskRunParam> &&task_indexes_to_param, std::vector<IowAddrs> &task_indexes_to_init_param) const {
  task_indexes_to_init_param.reserve(task_indexes_to_param.size());
  for (size_t i = 0UL; i < task_indexes_to_refresh_type.size(); ++i) {
    auto &param = task_indexes_to_param[i];
    IowAddrs init_param = {std::move(param.parsed_input_addrs), std::move(param.parsed_output_addrs),
                           std::move(param.parsed_workspace_addrs)};
    for (size_t j = 0UL; j < init_param.input_logic_addrs.size(); ++j) {
      auto &addr = init_param.input_logic_addrs[j];
      addr.support_refresh = static_cast<bool>(task_indexes_to_refresh_type[i].input_refresh_types[j]);
      addr.memory_type = static_cast<uint64_t>(
          logical_addrs_to_mem_app_type.at(std::pair<uint64_t, uint64_t>(addr.memory_type, addr.logic_addr)));
    }
    for (size_t j = 0UL; j < init_param.output_logic_addrs.size(); ++j) {
      auto &addr = init_param.output_logic_addrs[j];
      addr.support_refresh = static_cast<bool>(task_indexes_to_refresh_type[i].output_refresh_types[j]);
      addr.memory_type = static_cast<uint64_t>(
          logical_addrs_to_mem_app_type.at(std::pair<uint64_t, uint64_t>(addr.memory_type, addr.logic_addr)));
    }
    for (size_t j = 0UL; j < init_param.workspace_logic_addrs.size(); ++j) {
      auto &addr = init_param.workspace_logic_addrs[j];
      addr.support_refresh = static_cast<bool>(task_indexes_to_refresh_type[i].workspace_refresh_types[j]);
      addr.memory_type = static_cast<uint64_t>(
          logical_addrs_to_mem_app_type.at(std::pair<uint64_t, uint64_t>(addr.memory_type, addr.logic_addr)));
    }
    task_indexes_to_init_param.emplace_back(std::move(init_param));
  }

  for (const auto &fap : fixed_addr_bulk_.pieces) {
    AddrDesc *addr_desc;
    switch (fap.desc.iow_index_type) {
      case TaskArgsRefreshTypeClassifier::kInput:
        addr_desc = &(task_indexes_to_init_param.at(fap.desc.task_index).input_logic_addrs.at(fap.desc.iow_index));
        break;
      case TaskArgsRefreshTypeClassifier::kOutput:
        addr_desc = &(task_indexes_to_init_param.at(fap.desc.task_index).output_logic_addrs.at(fap.desc.iow_index));
        break;
      case TaskArgsRefreshTypeClassifier::kWorkspace:
        addr_desc = &(task_indexes_to_init_param.at(fap.desc.task_index).workspace_logic_addrs.at(fap.desc.iow_index));
        break;
      default:
        GELOGE(INTERNAL_ERROR, "[OM2] Unexpected iow type %d when init task infos",
               static_cast<int32_t>(fap.desc.iow_index_type));
        return FAILED;
    }
    addr_desc->logic_addr = fap.device_addr;
    addr_desc->memory_type = static_cast<uint64_t>(MemoryAppType::kMemoryTypeFix);
    addr_desc->support_refresh = false;
  }

  return SUCCESS;
}
Status ModelArgsManager::ValidateTaskRunParam(const std::vector<TaskArgsDesc> &args_descs) const {
  std::map<ArgsPlacement, int32_t> placement_counts;
  for (const auto &args_desc : args_descs) {
    GE_ASSERT_TRUE((++placement_counts[args_desc.placement] <= 1), "[OM2] Placement %d has multiple records",
                   static_cast<int32_t>(args_desc.placement));
  }
  return SUCCESS;
}
Status ModelArgsManager::ParseModelTaskDef(domi::ModelTaskDef &model_task_def,
                                           std::vector<TaskRunParam> &task_indexes_to_run_param,
                                           TaskNodeMap &task_node_map) {
  const auto need_log = IsLogEnable(GE_MODULE_NAME, DLOG_DEBUG);
  const size_t task_size = static_cast<size_t>(model_task_def.task_size());

  for (size_t i = 0UL; i < task_size; ++i) {
    domi::TaskDef *const task_def = model_task_def.mutable_task(static_cast<int32_t>(i));

    auto &task_info = task_list_ptr_->at(i);
    const auto op_index = task_info->ParseOpIndex(*task_def);
    const OpDescPtr op_desc = model_adapter_.GetOpByIndex(static_cast<uint32_t>(op_index));
    const RuntimeParam &rts_param = model_adapter_.GetRuntimeParam();
    GE_ASSERT_SUCCESS(task_info->ParseTaskRunParam(*task_def, rts_param, op_desc, task_indexes_to_run_param[i]),
                      "[OM2] task index:%zu ParseTaskRunParam failed", i);
    GE_ASSERT_SUCCESS(ValidateTaskRunParam(task_indexes_to_run_param[i].args_descs),
                      "[OM2] task index %zu occurred multiple placement, task_type is %d", i, task_def->type());
    has_args_ = (has_args_) || (!task_indexes_to_run_param[i].args_descs.empty());

    GE_ASSERT_SUCCESS(task_node_map.AddRelation(i, op_index));

    if (need_log) {
      DebugLogTaskRunParam(i, op_index, task_indexes_to_run_param[i], op_desc);
    }
  }
  if (!has_args_) {
    GELOGW("[OM2] There no args need be managed in model");
  }
  return SUCCESS;
}

ModelArgsManager::TriggerTypesToPolicies ModelArgsManager::GenerateTriggerTypesToCorrespondingUpdatePolicies() const {
  if (model_adapter_.IsFeatureBaseRefreshable()) {
    return {SmallVector<UpdatePolicy, kUpdatePolicyEnd>{kInitOneTime},
            SmallVector<UpdatePolicy, kUpdatePolicyEnd>{kUpdateFmAndModelIo, kInitOneTime},
            SmallVector<UpdatePolicy, kUpdatePolicyEnd>{kUpdateModelIo, kUpdateFmAndModelIo, kInitOneTime},
            SmallVector<UpdatePolicy, kUpdatePolicyEnd>{KUpdateHostInput, kUpdateModelIo, kUpdateFmAndModelIo,
                                                        kInitOneTime}};
  } else {
    return {SmallVector<UpdatePolicy, kUpdatePolicyEnd>{kInitOneTime},
            SmallVector<UpdatePolicy, kUpdatePolicyEnd>{kUpdatePolicyEnd},
            SmallVector<UpdatePolicy, kUpdatePolicyEnd>{kUpdateModelIo, kInitOneTime},
            SmallVector<UpdatePolicy, kUpdatePolicyEnd>{KUpdateHostInput, kUpdateModelIo, kInitOneTime}};
  }
}

Status ModelArgsManager::GenerateArgsDataForProgramGenerator(Om2CodegenModel &codegen_model) {
  GE_ASSERT_NOTNULL(&codegen_model);
  auto &args_table = codegen_model.args_table;
  // 根据model_args_和model_args_len_,组装整个模型所需args数据结构(model_args_semantic)
  GE_ASSERT_EQ(model_args_.size(), model_args_len_.size());
  auto &model_args_semantic = args_table.model_args_semantic;
  model_args_semantic.clear();
  model_args_semantic.reserve(model_args_.size());
  for (size_t i = 0UL; i < model_args_.size(); ++i) {
    model_args_semantic.emplace_back(
        ModelArgsSemantic{model_args_[i].placement, model_args_len_[i], model_args_[i].model_args_partitions});
  }

  // 根据task_indexes_to_args_,组装每个task的args数据结构(task_indexes_to_args_semantic),按task index序排列
  auto &task_args_semantic = args_table.task_indexes_to_args_semantic;
  task_args_semantic.clear();
  task_args_semantic.resize(task_indexes_to_args_.size());
  for (size_t task_index = 0UL; task_index < task_indexes_to_args_.size(); ++task_index) {
    for (size_t placement = 0UL; placement < task_indexes_to_args_[task_index].size(); ++placement) {
      task_args_semantic[task_index][placement].offset = task_indexes_to_args_[task_index][placement].offset;
      task_args_semantic[task_index][placement].len = task_indexes_to_args_[task_index][placement].len;
    }
  }

  // 根据allocation_ids_to_model_args_refresh_infos_addr_all组装allocation id变化与要刷新的地址的映射关系
  auto convert_refresh_infos = [](const std::vector<std::vector<ModelArgsRefreshInfo>> &src,
                                  std::vector<std::vector<ModelArgsRefreshInfoSemantic>> &dst) {
    dst.clear();
    dst.resize(src.size());
    for (size_t allocation_id = 0UL; allocation_id < src.size(); ++allocation_id) {
      dst[allocation_id].reserve(src[allocation_id].size());
      for (const auto &info : src[allocation_id]) {
        dst[allocation_id].emplace_back(
            ModelArgsRefreshInfoSemantic{info.base_args_offset, info.offset, info.placement});
      }
    }
  };
  convert_refresh_infos(allocation_ids_to_model_args_refresh_infos_addr_all,
                        args_table.allocation_ids_to_model_args_refresh_infos_addr_all_semantic);

  // 组装input index和allocation id的关系
  args_table.input_index_to_allocation_ids = model_adapter_.GetInputIndexToAllocationIds();

  // 组装output index和allocation id的关系
  args_table.output_index_to_allocation_ids = model_adapter_.GetOutputIndexToAllocationIds();
  return SUCCESS;
}
}  // namespace om2
}  // namespace ge
