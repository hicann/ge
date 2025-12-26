/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "common/preload/model/pre_davinci_model.h"
#include "common/preload/model/pre_model_partition_utils.h"
#include "common/compile_profiling/ge_call_wrapper.h"
#include "framework/common/taskdown_common.h"
#include "common/preload/task_info/pre_generate_task_registry.h"

namespace ge {
void PreDavinciModel::Assign(const GeModelPtr &ge_model) {
  ge_model_ = ge_model;
}

void PreDavinciModel::DoReset() const {
  // other reset
  PreModelPartitionUtils::GetInstance().Reset();  // inst reset
}

Status PreDavinciModel::Init() {
  GELOGI("begin init pre davinci model.");
  GE_ASSERT_NOTNULL(ge_model_, "GeModel is null");
  const ComputeGraphPtr compute_graph = ge_model_->GetGraph();
  GE_ASSERT_NOTNULL(compute_graph, "compute_graph is null");
  InitRuntimeParams();
  InitKernelOffset();
  DoReset();
  GE_CHK_STATUS_RET(InitNodes(compute_graph), "[Init][Nodes] failed, graph:%s.", compute_graph->GetName().c_str());
  GE_TIMESTAMP_START(DoTaskSink);
  GE_CHK_STATUS_RET(DoTaskSink(EngineType::kDefaultEngine), "[Call][DoTaskSink] failed, model_id:%u.", model_id_);
  GE_TIMESTAMP_END(DoTaskSink, "PreDavinciModel::DoTaskSink");

  GE_TIMESTAMP_START(DoPartitionProcess);
  GE_CHK_STATUS_RET(DoPartitionProcess(), "[Call][DoPartitionProcess] failed, model_id:%u.", model_id_);
  GE_TIMESTAMP_END(DoPartitionProcess, "PreDavinciModel::DoPartitionProcess");
  GELOGI("success init pre davinci model.");
  return SUCCESS;
}
Status PreDavinciModel::DoTaskSink(const EngineType engine_type) {
  // task sink is supported as model_task_def is set
  const auto &model_task_def = ge_model_->GetModelTaskDefPtr();
  GE_ASSERT_NOTNULL(model_task_def, "model_task_def is null");

  task_num_ = static_cast<uint32_t>(model_task_def->task_size());
  for (int32_t i = 0; i < static_cast<int32_t>(task_num_); ++i) {
    // dynamic shape will create task_list_ before
    const auto &task_def = model_task_def->task(i);
    const domi::KernelDef &kernel_def = task_def.kernel();
    const domi::KernelContext &context = kernel_def.context();
    string engine_name;
    const auto op_desc = GetOpByIndex(context.op_index());
    GE_ASSERT_NOTNULL(op_desc, "[Call][GetOpByIndex] get op fail, op index is %u", context.op_index());
    GE_CHK_STATUS_RET(GetEngineName(engine_type, task_def.type(), context.kernel_type(), engine_name),
                      "[Call][GetEngineName] op[%s] failed.", op_desc->GetName().c_str());
    PreTaskInput pre_task_input;
    pre_task_input.rts_param = runtime_param_;
    pre_task_input.names_to_bin_offset = names_to_bin_offset_;
    const auto func = PreGenerateTaskRegistry::GetInstance().FindPreGenerateTask(engine_name);
    GE_ASSERT_NOTNULL(func, "[Call][FindPreGenerateTask] op[%s] can't find func from engine_name:%s",
                      op_desc->GetName().c_str(), engine_name.c_str());

    GELOGD("DoTaskSink generate task no:%d, op_desc:%s", i, op_desc->GetName().c_str());
    const auto task_result = func(task_def, op_desc, pre_task_input);
    if (!task_result.status.IsSuccess()) {
      GELOGE(FAILED, "[Call][func] func execution failed, error message:%s", task_result.status.GetErrorMessage());
      return FAILED;
    }
    PreModelPartitionUtils::GetInstance().AddPreTaskDescInfo(task_result.pre_task_desc_infos);
  }
  return SUCCESS;
}
Status PreDavinciModel::InitNodes(const ComputeGraphPtr &compute_graph) {
  const auto &nodes = compute_graph->GetAllNodes();
  for (size_t i = 0UL; i < nodes.size(); ++i) {
    const auto &node = nodes.at(i);
    const auto &op_desc = node->GetOpDesc();
    GE_CHECK_NOTNULL(op_desc);
    GELOGI("add op[%d] to list", op_desc->GetId());
    op_list_[op_desc->GetId()] = op_desc;
  }
  return SUCCESS;
}
void PreDavinciModel::InitRuntimeParams() {
  PreModelUtils::InitRuntimeParams(ge_model_, runtime_param_);
}
Status PreDavinciModel::DoPartitionProcess() {
  GE_CHK_STATUS_RET(
      PreModelPartitionUtils::GetInstance().InitTaskBuildMem(huge_stream_size_, runtime_param_.stream_num),
      "[Call][PreModelPartitionUtils][InitTaskBuildMem] failed.");
  // refresh partition data
  GE_CHK_STATUS_RET(PreModelPartitionUtils::GetInstance().PreparePartitionData(EngineType::kDefaultEngine),
                    "[Call][PreModelPartitionUtils][PreparePartitionData] failed.");
  return SUCCESS;
}
void PreDavinciModel::InitKernelOffset() {
  const TBEKernelStore tbe_kernel_store = ge_model_->GetTBEKernelStore();
  names_to_bin_offset_ = tbe_kernel_store.GetKernelOffset();
  GELOGI("names_to_bin_offset_ size:%u", names_to_bin_offset_.size());
}
// get Op
OpDescPtr PreDavinciModel::GetOpByIndex(const uint32_t op_index) const {
  const auto it = op_list_.find(static_cast<int64_t>(op_index));
  GE_ASSERT_TRUE(!(it == op_list_.end()));
  return it->second;
}

Status PreDavinciModel::GetEngineName(const EngineType engine_type, const uint32_t task_type,
                                      const uint32_t kernel_type, std::string &engine_name) const {
  GELOGD("GetEngineName engine_type:%u, task_type:%u, kernel_type:%u.", engine_type, task_type, kernel_type);
  if (task_type == static_cast<uint32_t>(ModelTaskType::MODEL_TASK_KERNEL)) {
    switch (engine_type) {
      case EngineType::kDefaultEngine:
        engine_name = GetEngineNameByType(kernel_type, kKernelTypeToEngineName);
        break;
      case EngineType::kNanoEngine:
        engine_name = GetEngineNameByType(kernel_type, kKernelTypeToNanoEngineName);
        break;
      default:
        engine_name = "";
        GELOGE(FAILED, "there are unsupported engine_type in the model, engine_type:%u, kernel_type:%u.", engine_type,
               kernel_type);
        break;
    }
  } else {
    switch (engine_type) {
      case EngineType::kDefaultEngine:
      case EngineType::kNanoEngine:
        engine_name = GetEngineNameByType(task_type, kTaskTypeToEngineName);
        break;
      default:
        engine_name = "";
        GELOGE(FAILED, "there are unsupported engine_type in the model, engine_type:%u, task_type:%u.", engine_type,
               task_type);
        break;
    }
  }
  if (engine_name == "") {
    GELOGE(FAILED, "[Call] there are unsupported task in the model, engine_type:%u, task_type:%u, kernel_type:%u.",
           engine_type, task_type, kernel_type);
    return FAILED;
  }
  GELOGD("success get engine name[%s]", engine_name.c_str());
  return SUCCESS;
}

std::string PreDavinciModel::GetEngineNameByType(const uint32_t type,
                                                 const std::map<uint32_t, std::string> type_to_engine_name) const {
  const auto it = type_to_engine_name.find(type);
  if (it == type_to_engine_name.end()) {
    GELOGE(FAILED, "[Call][GetEngineNameByType] failed find engine name from type:%u.", type);
    return "";
  }
  return it->second;
}
}  // namespace ge