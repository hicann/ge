/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "framework/runtime/dump/profiling_impl.h"

#include <algorithm>
#include <cstring>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include "common/checker.h"
#include "framework/common/debug/ge_log.h"
#include "framework/runtime/dump/profiling_config.h"
#include "graph_metadef/common/opskernel/ops_kernel_info_types.h"
#include "mmpa/mmpa_api.h"
#include "aprof_pub.h"
#include "profiling/prof_common.h"

namespace ge {
namespace dump {
namespace {
constexpr uint32_t kAgingFlag = 1U;
constexpr uint32_t kNonAgingFlag = 0U;
constexpr uint32_t kModelGraphIdMapDataLen = 16U;
constexpr uint32_t kOm1ModelLoadType = MSPROF_REPORT_MODEL_GRAPH_ID_MAP_TYPE + 2U;
// 模型级 / Node 级 profiling type ID，对齐 gert::GeProfInfoType 枚举值
constexpr uint32_t kProfModelExecuteType = MSPROF_REPORT_MODEL_GRAPH_ID_MAP_TYPE + 1U;
constexpr uint32_t kProfInputCopyType = MSPROF_REPORT_MODEL_GRAPH_ID_MAP_TYPE + 3U;
constexpr uint32_t kProfOutputCopyType = MSPROF_REPORT_MODEL_GRAPH_ID_MAP_TYPE + 4U;
constexpr uint32_t kProfStepInfoType = MSPROF_REPORT_NODE_GE_API_BASE_TYPE + 6U;
constexpr uint32_t kProfLaunchType = MSPROF_REPORT_NODE_LAUNCH_TYPE;
constexpr uint32_t kTensorInfoBytes = 44U;
constexpr uint32_t kTensorInfoBytesWithCap = 56U;
constexpr uint32_t kInvalidContextId = 0xFFFFFFFFU;
constexpr uint32_t kFusionOpInfoCap = 52U;
constexpr uint32_t kHashOffset = 8U;

const std::map<ModelTaskType, MsprofGeTaskType> kModelTaskTypeToProfTaskType = {
    {ModelTaskType::MODEL_TASK_KERNEL, MSPROF_GE_TASK_TYPE_AI_CORE},
    {ModelTaskType::MODEL_TASK_VECTOR_KERNEL, MSPROF_GE_TASK_TYPE_AIV},
    {ModelTaskType::MODEL_TASK_VECTOR_ALL_KERNEL, MSPROF_GE_TASK_TYPE_AIV},
    {ModelTaskType::MODEL_TASK_KERNEL_EX, MSPROF_GE_TASK_TYPE_AI_CPU},
    {ModelTaskType::MODEL_TASK_DSA, MSPROF_GE_TASK_TYPE_DSA},
    {ModelTaskType::MODEL_TASK_HCCL, MSPROF_GE_TASK_TYPE_HCCL},
    {ModelTaskType::MODEL_TASK_ALL_KERNEL, MSPROF_GE_TASK_TYPE_AI_CORE},
    {ModelTaskType::MODEL_TASK_SUPER_KERNEL, MSPROF_GE_TASK_TYPE_AI_CORE},
    {ModelTaskType::MODEL_TASK_FUSION_KERNEL, MSPROF_GE_TASK_TYPE_AI_CORE},
    {ModelTaskType::MODEL_TASK_KERNEL_LAUNCH_V2, MSPROF_GE_TASK_TYPE_AI_CORE},
    {ModelTaskType::MODEL_TASK_CUSTOM_KERNEL, MSPROF_GE_TASK_TYPE_AI_CORE},
};

Status CheckMsprofRet(int32_t ret, const char *action, const char *name) {
  if ((ret == MSPROF_ERROR_NONE) || (ret == MSPROF_ERROR_UNINITIALIZE)) {
    return SUCCESS;
  }
  GELOGW("%s failed, name=%s, ret=%d", action, name, ret);
  return FAILED;
}

void FillProfTensorDesc(const TaskDescInfo &task_desc_info, size_t tensor_index, size_t offset_idx,
                        MsprofTensorInfo *tensor_info) {
  const bool is_input = tensor_index < task_desc_info.input_shape.size();
  const auto &formats = is_input ? task_desc_info.input_format : task_desc_info.output_format;
  const auto &data_types = is_input ? task_desc_info.input_data_type : task_desc_info.output_data_type;
  const auto &shapes = is_input ? task_desc_info.input_shape : task_desc_info.output_shape;
  const size_t desc_index = is_input ? tensor_index : tensor_index - task_desc_info.input_shape.size();

  auto &tensor_data = tensor_info->tensorData[offset_idx];
  tensor_data.tensorType = static_cast<uint32_t>(is_input ? MSPROF_GE_TENSOR_TYPE_INPUT : MSPROF_GE_TENSOR_TYPE_OUTPUT);
  tensor_data.format = static_cast<uint32_t>(formats[desc_index]);
  const DataType data_type = data_types[desc_index];
  tensor_data.dataType = (static_cast<uint32_t>(data_type) < static_cast<uint32_t>(DT_MAX))
                             ? static_cast<uint32_t>(data_type)
                             : static_cast<uint32_t>(DT_UNDEFINED);
  const size_t shape_size = shapes[desc_index].size();
  const size_t copy_size = std::min(static_cast<size_t>(MSPROF_GE_TENSOR_DATA_SHAPE_LEN), shape_size);
  for (size_t i = 0U; i < copy_size; ++i) {
    tensor_data.shape[i] = static_cast<uint32_t>(shapes[desc_index][i]);
  }
  if (shape_size < static_cast<size_t>(MSPROF_GE_TENSOR_DATA_SHAPE_LEN)) {
    tensor_data.shape[shape_size] = 0U;
  }
}

void BuildSingleTensorInfo(const TaskDescInfo &task_desc_info, uint32_t tid, size_t index, uint32_t tensor_num,
                           MsprofAdditionalInfo &tensor_info) {
  tensor_info.type = MSPROF_REPORT_NODE_TENSOR_INFO_TYPE;
  tensor_info.level = static_cast<uint16_t>(MSPROF_REPORT_NODE_LEVEL);
  tensor_info.timeStamp = task_desc_info.prof_time;
  tensor_info.threadId = tid;
  tensor_info.dataLen = kTensorInfoBytesWithCap + (tensor_num - 1U) * kTensorInfoBytes;
  auto *tensor_data = reinterpret_cast<MsprofTensorInfo *>(tensor_info.data);
  tensor_data->opName = MsprofGetHashId(task_desc_info.op_name.c_str(), task_desc_info.op_name.length());
  tensor_data->tensorNum = tensor_num;
  for (size_t i = 0U; i < static_cast<size_t>(tensor_num); ++i) {
    FillProfTensorDesc(task_desc_info, index * static_cast<size_t>(MSPROF_GE_TENSOR_DATA_NUM) + i, i, tensor_data);
  }
}

void AppendTensorInfo(const Om2TaskIoEntry *entries, uint64_t entry_num, const char *op_name,
                      std::vector<Format> &formats, std::vector<DataType> &data_types,
                      std::vector<std::vector<int64_t>> &shapes) {
  if (entries == nullptr) {
    if (entry_num != 0U) {
      GELOGW("Task io entries is null, op_name=%s, entry_num=%llu", op_name,
             static_cast<unsigned long long>(entry_num));
    }
    return;
  }

  for (uint64_t i = 0U; i < entry_num; ++i) {
    const Om2Tensor *tensor = entries[i].tensor;
    if (tensor == nullptr) {
      GELOGW("Task io tensor is null, op_name=%s, index=%llu", op_name, static_cast<unsigned long long>(i));
      continue;
    }
    if ((tensor->shape_dims == nullptr) && (tensor->shape_dims_num != 0U)) {
      GELOGW("Task io tensor shape is null, op_name=%s, index=%llu, shape_dims_num=%llu", op_name,
             static_cast<unsigned long long>(i), static_cast<unsigned long long>(tensor->shape_dims_num));
      continue;
    }

    formats.emplace_back(static_cast<Format>(tensor->format));
    data_types.emplace_back(static_cast<DataType>(tensor->data_type));
    std::vector<int64_t> shape;
    if (tensor->shape_dims_num != 0U) {
      shape.assign(tensor->shape_dims, tensor->shape_dims + tensor->shape_dims_num);
    }
    shapes.emplace_back(std::move(shape));
  }
}
}  // namespace

bool ProfilingImpl::IsProfilingEnabled() const {
  return ProfilingConfig::Instance().IsEnabled();
}

Status ProfilingImpl::ReportModelLoadBegin(const ModelDumpInfo &model_info) const {
  if (!ProfilingConfig::Instance().IsModelLoadEnabled()) {
    GELOGD("Skip reporting OM2 profiling model load begin, model_load profiling disabled, model_id=%u",
           model_info.model_id);
    return SUCCESS;
  }

  const char *model_name = (model_info.model_name != nullptr) ? model_info.model_name : "";
  GELOGD("Report OM2 profiling model load begin, model_id=%u, model_name=%s", model_info.model_id, model_name);
  MsprofEvent model_load_event{};
  model_load_event.type = kOm1ModelLoadType;
  model_load_event.itemId = model_info.model_id;
  model_load_event.level = MSPROF_REPORT_MODEL_LEVEL;
  model_load_event.timeStamp = MsprofSysCycleTime();
  model_load_event.threadId = static_cast<uint32_t>(mmGetTid());
  model_load_event.requestId = 0U;
  GELOGD("[OM2][Prof] ReportModelLoadBegin: model_id=%u, model_name=%s, type=%u, level=%u, timeStamp=%lu, threadId=%u",
         model_info.model_id, model_name, model_load_event.type, model_load_event.level, model_load_event.timeStamp,
         model_load_event.threadId);
  return CheckMsprofRet(MsprofReportEvent(kNonAgingFlag, &model_load_event), "Report model load begin", model_name);
}

Status ProfilingImpl::ReportModelLoadEnd(const ModelDumpInfo &model_info) const {
  if (!ProfilingConfig::Instance().IsModelLoadEnabled()) {
    GELOGD("Skip reporting OM2 profiling model load end, model_load profiling disabled, model_id=%u",
           model_info.model_id);
    return SUCCESS;
  }

  const char *model_name = (model_info.model_name != nullptr) ? model_info.model_name : "";
  GELOGD("Report OM2 profiling model load end, model_id=%u, model_name=%s", model_info.model_id, model_name);
  const uint64_t prof_time = MsprofSysCycleTime();
  const uint32_t tid = static_cast<uint32_t>(mmGetTid());

  MsprofAdditionalInfo graph_id_info{};
  graph_id_info.level = MSPROF_REPORT_MODEL_LEVEL;
  graph_id_info.type = MSPROF_REPORT_MODEL_GRAPH_ID_MAP_TYPE;
  graph_id_info.timeStamp = prof_time;
  graph_id_info.threadId = tid;
  graph_id_info.dataLen = kModelGraphIdMapDataLen;
  auto *graph_id_data = reinterpret_cast<MsprofGraphIdInfo *>(graph_id_info.data);
  graph_id_data->graphId = UINT32_MAX;
  graph_id_data->modelName = MsprofGetHashId(model_name, strlen(model_name));
  graph_id_data->modelId = model_info.model_id;
  GELOGD(
      "[OM2][Prof] ReportModelGraphIdMap: model_id=%u, model_name=%s, type=%u, level=%u, timeStamp=%lu, "
      "threadId=%u, graphId=%u, modelName(hash)=%lu",
      model_info.model_id, model_name, graph_id_info.type, graph_id_info.level, graph_id_info.timeStamp,
      graph_id_info.threadId, graph_id_data->graphId, graph_id_data->modelName);
  GE_CHK_STATUS_RET(CheckMsprofRet(
      MsprofReportAdditionalInfo(kNonAgingFlag, &graph_id_info, static_cast<uint32_t>(sizeof(MsprofAdditionalInfo))),
      "Report model graph id map", model_name));

  MsprofEvent model_load_event{};
  model_load_event.type = kOm1ModelLoadType;
  model_load_event.itemId = model_info.model_id;
  model_load_event.level = MSPROF_REPORT_MODEL_LEVEL;
  model_load_event.timeStamp = prof_time;
  model_load_event.threadId = tid;
  model_load_event.requestId = 0U;
  GELOGD(
      "[OM2][Prof] ReportModelLoadEnd: model_id=%u, model_name=%s, type=%u, level=%u, timeStamp=%lu, "
      "threadId=%u, itemId=%lu, requestId=%u",
      model_info.model_id, model_name, model_load_event.type, model_load_event.level, model_load_event.timeStamp,
      model_load_event.threadId, model_load_event.itemId, model_load_event.requestId);
  return CheckMsprofRet(MsprofReportEvent(kNonAgingFlag, &model_load_event), "Report model load end", model_name);
}

Status ProfilingImpl::SaveTaskInfo(const Om2TaskInfo &task_info, const ModelDumpInfo &model_info) const {
  const char *op_name = (task_info.op_name != nullptr) ? task_info.op_name : "";
  if (!ProfilingConfig::Instance().IsTaskReportEnabled()) {
    GELOGD("Skip saving OM2 profiling task info, task profiling disabled, op_name=%s, task_type=%u", op_name,
           task_info.task_type);
    return SUCCESS;
  }

  GELOGD(
      "Save OM2 profiling task info, op_name=%s, op_type=%s, task_type=%u, task_id=%u, stream_id=%u, "
      "block_dim=%u, input_num=%llu, output_num=%u, workspace_num=%u, context_id=%u, thread_id=%u",
      op_name, task_info.op_type != nullptr ? task_info.op_type : "", task_info.task_type, task_info.task_id,
      task_info.stream_id, task_info.block_dim, static_cast<unsigned long long>(task_info.input_num),
      task_info.output_num, task_info.workspace_num, task_info.context_id, task_info.thread_id);
  TaskDescInfo task_desc_info{};
  uint32_t prof_task_type = static_cast<uint32_t>(MSPROF_GE_TASK_TYPE_INVALID);
  GE_CHK_STATUS_RET(BuildTaskDescInfo(task_info, model_info, task_desc_info, prof_task_type));
  if (prof_task_type == static_cast<uint32_t>(MSPROF_GE_TASK_TYPE_INVALID)) {
    GELOGD("Skip reporting OM2 profiling task info, unsupported task type, op_name=%s, task_type=%u", op_name,
           task_info.task_type);
    return SUCCESS;
  }
  GE_CHK_STATUS_RET(ReportTaskDescInfo(task_desc_info, prof_task_type, task_info.thread_id));
  GE_CHK_STATUS_RET(ReportFusionOpInfo(task_info, model_info.model_id));
  GE_CHK_STATUS_RET(ReportLaunchInfo(task_info, task_desc_info.prof_time), "ReportLaunchInfo failed");
  return SUCCESS;
}

Status ProfilingImpl::BuildTaskDescInfo(const Om2TaskInfo &task_info, const ModelDumpInfo &model_info,
                                        TaskDescInfo &task_desc_info, uint32_t &prof_task_type) const {
  const auto model_task_type = static_cast<ModelTaskType>(task_info.task_type);
  const auto iter = kModelTaskTypeToProfTaskType.find(model_task_type);
  if (iter == kModelTaskTypeToProfTaskType.end()) {
    GELOGD("Skip unsupported profiling task type: %u", task_info.task_type);
    prof_task_type = static_cast<uint32_t>(MSPROF_GE_TASK_TYPE_INVALID);
    return SUCCESS;
  }

  const char *op_name = (task_info.op_name != nullptr) ? task_info.op_name : "";
  task_desc_info.prof_time = MsprofSysCycleTime();
  task_desc_info.model_name = (model_info.model_name != nullptr) ? model_info.model_name : "";
  task_desc_info.op_name = op_name;
  task_desc_info.op_type = (task_info.op_type != nullptr) ? task_info.op_type : "";
  task_desc_info.block_dim = task_info.block_dim;
  task_desc_info.task_id = task_info.task_id;
  task_desc_info.stream_id = task_info.stream_id;
  task_desc_info.cur_iter_num = 0;
  task_desc_info.task_type = std::to_string(static_cast<uint32_t>(iter->second));
  task_desc_info.context_id = task_info.context_id;
  prof_task_type = static_cast<uint32_t>(iter->second);

  AppendTensorInfo(task_info.inputs, task_info.input_num, op_name, task_desc_info.input_format,
                   task_desc_info.input_data_type, task_desc_info.input_shape);
  AppendTensorInfo(task_info.outputs, task_info.output_num, op_name, task_desc_info.output_format,
                   task_desc_info.output_data_type, task_desc_info.output_shape);
  GELOGD("Build OM2 profiling task desc, op_name=%s, prof_task_type=%u, input_desc_num=%zu, output_desc_num=%zu",
         op_name, prof_task_type, task_desc_info.input_shape.size(), task_desc_info.output_shape.size());
  return SUCCESS;
}

Status ProfilingImpl::ReportTaskDescInfo(const TaskDescInfo &task_desc_info, uint32_t prof_task_type,
                                         uint32_t tid) const {
  MsprofCompactInfo node_basic_info{};
  node_basic_info.level = static_cast<uint16_t>(MSPROF_REPORT_NODE_LEVEL);
  node_basic_info.type = MSPROF_REPORT_NODE_BASIC_INFO_TYPE;
  node_basic_info.timeStamp = task_desc_info.prof_time;
  node_basic_info.threadId = tid;
  auto &prof_node_basic_info = node_basic_info.data.nodeBasicInfo;
  prof_node_basic_info.opName = MsprofGetHashId(task_desc_info.op_name.c_str(), task_desc_info.op_name.length());
  prof_node_basic_info.opType = MsprofGetHashId(task_desc_info.op_type.c_str(), task_desc_info.op_type.length());
  prof_node_basic_info.taskType = prof_task_type;
  prof_node_basic_info.blockDim = task_desc_info.block_dim;
  GELOGD(
      "[OM2][Prof] ReportTaskDescInfo: op_name=%s, opName(hash)=%lu, opType(hash)=%lu, "
      "prof_task_type=%u, block_dim=%u, task_id=%u, stream_id=%u, tid=%u, level=%u, type=%u, timeStamp=%lu",
      task_desc_info.op_name.c_str(), prof_node_basic_info.opName, prof_node_basic_info.opType, prof_task_type,
      task_desc_info.block_dim, task_desc_info.task_id, task_desc_info.stream_id, tid, node_basic_info.level,
      node_basic_info.type, node_basic_info.timeStamp);
  const int32_t ret =
      MsprofReportCompactInfo(kAgingFlag, &node_basic_info, static_cast<uint32_t>(sizeof(MsprofCompactInfo)));
  if ((ret != MSPROF_ERROR_NONE) && (ret != MSPROF_ERROR_UNINITIALIZE)) {
    GELOGW("Report profiling compact info failed, op_name=%s, ret=%d", task_desc_info.op_name.c_str(), ret);
    return FAILED;
  }
  GE_CHK_STATUS_RET(ReportTensorInfo(task_desc_info, node_basic_info.threadId));
  return SUCCESS;
}

Status ProfilingImpl::ReportTensorInfo(const TaskDescInfo &task_desc_info, uint32_t tid) const {
  const size_t total_num = task_desc_info.input_shape.size() + task_desc_info.output_shape.size();
  GELOGD(
      "[OM2][Prof] ReportTensorInfo: op_name=%s, input_num=%zu, output_num=%zu, total_num=%zu, tid=%u, "
      "batch_num=%zu",
      task_desc_info.op_name.c_str(), task_desc_info.input_shape.size(), task_desc_info.output_shape.size(), total_num,
      tid,
      (total_num + static_cast<size_t>(MSPROF_GE_TENSOR_DATA_NUM) - 1U) /
          static_cast<size_t>(MSPROF_GE_TENSOR_DATA_NUM));
  const size_t batch_num = total_num / static_cast<size_t>(MSPROF_GE_TENSOR_DATA_NUM);
  for (size_t i = 0U; i < batch_num; ++i) {
    MsprofAdditionalInfo tensor_info{};
    BuildSingleTensorInfo(task_desc_info, tid, i, static_cast<uint32_t>(MSPROF_GE_TENSOR_DATA_NUM), tensor_info);
    GELOGD(
        "[OM2][Prof] ReportTensorInfo batch[%zu]: op_name=%s, level=%u, type=%u, timeStamp=%lu, threadId=%u, "
        "dataLen=%u, tensorNum=%u",
        i, task_desc_info.op_name.c_str(), tensor_info.level, tensor_info.type, tensor_info.timeStamp,
        tensor_info.threadId, tensor_info.dataLen,
        reinterpret_cast<const MsprofTensorInfo *>(tensor_info.data)->tensorNum);
    GE_CHK_STATUS_RET(CheckMsprofRet(
        MsprofReportAdditionalInfo(kAgingFlag, &tensor_info, static_cast<uint32_t>(sizeof(MsprofAdditionalInfo))),
        "Report profiling tensor info", task_desc_info.op_name.c_str()));
  }

  const size_t remain_num = total_num % static_cast<size_t>(MSPROF_GE_TENSOR_DATA_NUM);
  if (remain_num == 0U) {
    return SUCCESS;
  }
  MsprofAdditionalInfo tensor_info{};
  BuildSingleTensorInfo(task_desc_info, tid, batch_num, static_cast<uint32_t>(remain_num), tensor_info);
  GELOGD(
      "[OM2][Prof] ReportTensorInfo last batch[%zu]: op_name=%s, level=%u, type=%u, timeStamp=%lu, threadId=%u, "
      "dataLen=%u, tensorNum=%u",
      batch_num, task_desc_info.op_name.c_str(), tensor_info.level, tensor_info.type, tensor_info.timeStamp,
      tensor_info.threadId, tensor_info.dataLen,
      reinterpret_cast<const MsprofTensorInfo *>(tensor_info.data)->tensorNum);
  return CheckMsprofRet(
      MsprofReportAdditionalInfo(kAgingFlag, &tensor_info, static_cast<uint32_t>(sizeof(MsprofAdditionalInfo))),
      "Report profiling tensor info", task_desc_info.op_name.c_str());
}

Status ProfilingImpl::ReportContextIdInfo(const TaskDescInfo &task_desc_info, uint32_t tid) const {
  if (task_desc_info.context_id == kInvalidContextId) {
    GELOGD("Skip reporting OM2 profiling context id, op_name=%s, context_id=%u", task_desc_info.op_name.c_str(),
           task_desc_info.context_id);
    return SUCCESS;
  }

  MsprofAdditionalInfo context_info{};
  context_info.level = static_cast<uint16_t>(MSPROF_REPORT_NODE_LEVEL);
  context_info.type = MSPROF_REPORT_NODE_CONTEXT_ID_INFO_TYPE;
  context_info.timeStamp = task_desc_info.prof_time;
  context_info.threadId = tid;
  context_info.dataLen = static_cast<uint32_t>(sizeof(MsprofContextIdInfo));
  auto *context_data = reinterpret_cast<MsprofContextIdInfo *>(context_info.data);
  context_data->opName = MsprofGetHashId(task_desc_info.op_name.c_str(), task_desc_info.op_name.length());
  context_data->ctxIdNum = 1U;
  context_data->ctxIds[0] = task_desc_info.context_id;
  GELOGD(
      "[OM2][Prof] ReportContextIdInfo: op_name=%s, opName(hash)=%lu, context_id=%u, ctxIdNum=%u, level=%u, "
      "type=%u, timeStamp=%lu, threadId=%u, dataLen=%u",
      task_desc_info.op_name.c_str(), context_data->opName, task_desc_info.context_id, context_data->ctxIdNum,
      context_info.level, context_info.type, context_info.timeStamp, context_info.threadId, context_info.dataLen);
  return CheckMsprofRet(
      MsprofReportAdditionalInfo(kAgingFlag, &context_info, static_cast<uint32_t>(sizeof(MsprofAdditionalInfo))),
      "Report profiling context id info", task_desc_info.op_name.c_str());
}

Status ProfilingImpl::ReportModelLevelProf(const Om2ProfInfos &prof_info, uint32_t model_id) const {
  if (!ProfilingConfig::Instance().IsTaskTimeEnabled()) {
    GELOGD("[OM2][Prof] Skip reporting OM2 model level profiling, model_execute profiling disabled, model_id=%u",
           model_id);
    return SUCCESS;
  }

  GELOGD("[OM2][Prof] Report OM2 model level profiling, model_id=%u, count=%u", model_id, prof_info.count);

  for (uint32_t i = 0U; i < prof_info.count; ++i) {
    auto &unit = prof_info.profUnit[i];
    if (unit.type >= OM2_PROF_TYPE_COUNT) {
      GELOGW("[OM2][Prof] Invalid prof type=%u at index=%u, model_id=%u, skipping", unit.type, i, model_id);
      continue;
    }
    switch (unit.type) {
      case OM2_PROF_INPUT_COPY:
        GE_CHK_STATUS_RET(ReportProfApi(MSPROF_REPORT_MODEL_LEVEL, static_cast<uint32_t>(kProfInputCopyType),
                                        static_cast<uint64_t>(model_id), unit, "ReportInputCopy"));
        GELOGD("[OM2][Prof] InputCopy reported, model_id=%u, begin=%lu, end=%lu", model_id, unit.begin_time,
               unit.end_time);
        break;
      case OM2_PROF_MODEL_EXECUTE:
        GE_CHK_STATUS_RET(ReportProfModelExecute(unit, model_id, prof_info.step_id));
        GELOGD("[OM2][Prof] ModelExecute reported, model_id=%u, begin=%lu, end=%lu", model_id, unit.begin_time,
               unit.end_time);
        break;
      case OM2_PROF_OUTPUT_COPY:
        GE_CHK_STATUS_RET(ReportProfApi(MSPROF_REPORT_MODEL_LEVEL, static_cast<uint32_t>(kProfOutputCopyType),
                                        static_cast<uint64_t>(model_id), unit, "ReportOutputCopy"));
        GELOGD("[OM2][Prof] OutputCopy reported, model_id=%u, begin=%lu, end=%lu", model_id, unit.begin_time,
               unit.end_time);
        break;
      case OM2_PROF_STEP_INFO_START:
        GE_CHK_STATUS_RET(ReportProfApi(MSPROF_REPORT_NODE_LEVEL, static_cast<uint32_t>(kProfStepInfoType), 0U, unit,
                                        "ReportStepInfo start"));
        GELOGD("[OM2][Prof] StepInfo start reported, model_id=%u, time=%lu", model_id, unit.begin_time);
        break;
      case OM2_PROF_STEP_INFO_END:
        GE_CHK_STATUS_RET(ReportProfApi(MSPROF_REPORT_NODE_LEVEL, static_cast<uint32_t>(kProfStepInfoType), 1U, unit,
                                        "ReportStepInfo end"));
        GELOGD("[OM2][Prof] StepInfo end reported, model_id=%u, time=%lu", model_id, unit.end_time);
        break;
      default:
        GELOGW("[OM2][Prof] Unhandled prof type=%u at index=%u, model_id=%u", unit.type, i, model_id);
        break;
    }
  }

  GELOGD("[OM2][Prof] ReportModelLevelProf done, model_id=%u, total=%u entries", model_id, prof_info.count);
  return SUCCESS;
}

Status ProfilingImpl::ReportProfApi(uint32_t level, uint32_t type, uint64_t item_id, const Om2ProfUnit &unit,
                                    const char *tag) const {
  MsprofApi api{};
  api.level = static_cast<uint16_t>(level);
  api.type = type;
  api.beginTime = unit.begin_time;
  api.endTime = unit.end_time;
  api.itemId = item_id;
  api.threadId = unit.thread_id;
  GELOGD("[OM2][Prof] %s: level=%u, type=%u, beginTime=%lu, endTime=%lu, itemId=%lu, threadId=%u", tag, api.level,
         api.type, api.beginTime, api.endTime, api.itemId, api.threadId);
  return CheckMsprofRet(MsprofReportApi(kAgingFlag, &api), tag, "");
}

Status ProfilingImpl::ReportProfModelExecute(const Om2ProfUnit &unit, uint32_t model_id, uint64_t step_id) const {
  MsprofEvent event{};
  event.level = MSPROF_REPORT_MODEL_LEVEL;
  event.type = kProfModelExecuteType;
  event.itemId = static_cast<uint64_t>(model_id);
  event.threadId = unit.thread_id;
  event.requestId = static_cast<uint32_t>(step_id);
  GELOGD("[OM2][Prof] ReportModelExecute: model_id=%u, step_id=%lu, begin=%lu, end=%lu, threadId=%u", model_id, step_id,
         unit.begin_time, unit.end_time, unit.thread_id);
  event.timeStamp = unit.begin_time;
  GE_CHK_STATUS_RET(CheckMsprofRet(MsprofReportEvent(kAgingFlag, &event), "ReportModelExecute begin", ""));
  event.timeStamp = unit.end_time;
  return CheckMsprofRet(MsprofReportEvent(kAgingFlag, &event), "ReportModelExecute end", "");
}

Status ProfilingImpl::ReportLaunchInfo(const Om2TaskInfo &task_info, uint64_t prof_time) const {
  if (task_info.launch_begin == 0U) {
    if (task_info.op_name != nullptr) {
      GELOGD("[OM2][Prof] Launch timing not recorded, op_name=%s", task_info.op_name);
    }
    return SUCCESS;
  }
  const char *op_name = (task_info.op_name != nullptr) ? task_info.op_name : "";
  GELOGD("[OM2][Prof] Report OM2 launch info, op_name=%s", op_name);

  MsprofApi api{};
  api.level = MSPROF_REPORT_NODE_LEVEL;
  api.type = kProfLaunchType;
  api.beginTime = task_info.launch_begin;
  api.endTime = prof_time;
  api.itemId = MsprofGetHashId(op_name, strlen(op_name));
  api.threadId = task_info.thread_id;
  GELOGD("[OM2][Prof] ReportLaunchInfo: op_name=%s, beginTime=%lu, endTime=%lu, itemId=%lu, threadId=%u", op_name,
         api.beginTime, api.endTime, api.itemId, api.threadId);
  return CheckMsprofRet(MsprofReportApi(kNonAgingFlag, &api), "ReportLaunchInfo", op_name);
}

Status ProfilingImpl::ReportFusionOpInfo(const Om2TaskInfo &task_info, uint32_t model_id) const {
  if (task_info.original_op_names == nullptr) {
    return SUCCESS;  // 非融合算子，跳过
  }

  const char *op_name = (task_info.op_name != nullptr) ? task_info.op_name : "";
  const std::vector<std::string> origin_op_names = SplitFusionOpNames(task_info.original_op_names);
  const uint64_t prof_time = MsprofSysCycleTime();
  const uint32_t tid = task_info.thread_id;
  const uint64_t op_name_hash = MsprofGetHashId(op_name, strlen(op_name));
  const size_t total_op_num = origin_op_names.size();

  GELOGD(
      "[OM2][Prof] Report OM2 fusion op info, op_name=%s, model_id=%u, fusion_num=%zu, "
      "input_mem=%lu, output_mem=%lu, workspace_mem=%lu, weight_mem=%lu",
      op_name, model_id, total_op_num, task_info.input_mem_size, task_info.output_mem_size,
      task_info.workspace_mem_size, task_info.weight_mem_size);

  // 参照 V1 BuildFusionOpInfo 分批上报
  auto report_batch = [&prof_time, &tid, &op_name_hash, &task_info, &origin_op_names, &op_name](size_t batch_begin,
                                                                                                size_t batch_num) {
    MsprofAdditionalInfo info{};
    info.level = MSPROF_REPORT_NODE_LEVEL;
    info.type = MSPROF_REPORT_NODE_FUSION_OP_INFO_TYPE;
    info.timeStamp = prof_time;
    info.threadId = tid;
    info.dataLen = kFusionOpInfoCap + static_cast<uint32_t>(batch_num) * kHashOffset;
    auto *fusion_info = reinterpret_cast<ProfFusionOpInfo *>(info.data);
    fusion_info->opName = op_name_hash;
    fusion_info->fusionOpNum = static_cast<uint32_t>(batch_num);
    fusion_info->inputMemsize = task_info.input_mem_size;
    fusion_info->outputMemsize = task_info.output_mem_size;
    fusion_info->workspaceMemSize = task_info.workspace_mem_size;
    fusion_info->weightMemSize = task_info.weight_mem_size;
    fusion_info->totalMemSize = fusion_info->inputMemsize + fusion_info->outputMemsize + fusion_info->workspaceMemSize +
                                fusion_info->weightMemSize;
    for (size_t i = 0U; i < batch_num; ++i) {
      fusion_info->fusionOpId[i] =
          MsprofGetHashId(origin_op_names[batch_begin + i].c_str(), origin_op_names[batch_begin + i].length());
    }
    GELOGD(
        "[OM2][Prof] ReportFusionOpInfo batch[%zu]: op_name=%s, opName(hash)=%lu, level=%u, type=%u, "
        "timeStamp=%lu, threadId=%u, dataLen=%u, fusionOpNum=%u, input=%lu, output=%lu, workspace=%lu, "
        "weight=%lu, total=%lu",
        batch_begin, op_name, fusion_info->opName, info.level, info.type, info.timeStamp, info.threadId, info.dataLen,
        fusion_info->fusionOpNum, fusion_info->inputMemsize, fusion_info->outputMemsize, fusion_info->workspaceMemSize,
        fusion_info->weightMemSize, fusion_info->totalMemSize);
    return CheckMsprofRet(
        MsprofReportAdditionalInfo(kNonAgingFlag, &info, static_cast<uint32_t>(sizeof(MsprofAdditionalInfo))),
        "ReportFusionOpInfo", op_name);
  };

  const size_t batch_cnt = total_op_num / static_cast<size_t>(MSPROF_GE_FUSION_OP_NUM);
  for (size_t i = 0U; i < batch_cnt; ++i) {
    GE_CHK_STATUS_RET(
        report_batch(i * static_cast<size_t>(MSPROF_GE_FUSION_OP_NUM), static_cast<size_t>(MSPROF_GE_FUSION_OP_NUM)));
  }
  const size_t remain = total_op_num % static_cast<size_t>(MSPROF_GE_FUSION_OP_NUM);
  if (remain != 0U) {
    GE_CHK_STATUS_RET(report_batch(batch_cnt * static_cast<size_t>(MSPROF_GE_FUSION_OP_NUM), remain));
  }
  return SUCCESS;
}

std::vector<std::string> ProfilingImpl::SplitFusionOpNames(const char *names_str) const {
  std::vector<std::string> result;
  if (names_str == nullptr) {
    return result;
  }
  std::string str(names_str);
  size_t start = 0U;
  size_t end = 0U;
  while ((end = str.find(';', start)) != std::string::npos) {
    result.emplace_back(str.substr(start, end - start));
    start = end + 1U;
  }
  if (start < str.length()) {
    result.emplace_back(str.substr(start));
  }
  return result;
}

Status ProfilingImpl::RegisterModelToProfilingRuntime(const ModelDumpInfo &model_info) const {
  const int32_t ret = MsprofSetDeviceIdByGeModelIdx(model_info.model_id, model_info.device_id);
  if (ret != MSPROF_ERROR_NONE) {
    GELOGW("[OM2][Prof] Register model_id to profiling runtime failed, model_id=%u, device_id=%u, ret=%d",
           model_info.model_id, model_info.device_id, ret);
    return FAILED;
  }
  GELOGD("[OM2][Prof] Register model_id to profiling runtime success, model_id=%u, device_id=%u", model_info.model_id,
         model_info.device_id);
  return SUCCESS;
}

Status ProfilingImpl::UnregisterModelFromProfilingRuntime(uint32_t model_id) const {
  const int32_t ret = MsprofUnsetDeviceIdByGeModelIdx(model_id, 0U);
  if (ret != MSPROF_ERROR_NONE) {
    GELOGW("[OM2][Prof] Unregister model_id from profiling runtime failed, model_id=%u, ret=%d", model_id, ret);
    return FAILED;
  }
  GELOGD("[OM2][Prof] Unregister model_id from profiling runtime success, model_id=%u", model_id);
  return SUCCESS;
}

}  // namespace dump
}  // namespace ge
