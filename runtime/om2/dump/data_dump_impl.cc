/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "framework/runtime/dump/data_dump_impl.h"
#include "framework/runtime/dump/dump_config.h"
#include "framework/common/debug/ge_log.h"
#include "framework/common/framework_types_internal.h"
#include "graph/def_types.h"
#include "rt_external.h"
#include "acl/acl_rt.h"

namespace ge {
namespace dump {
namespace {
constexpr uint32_t kAicpuLoadFlag = 1U;
constexpr uint32_t kAddrLength = static_cast<uint32_t>(sizeof(void *));
constexpr uint64_t kOpDebugShape = 2048U;
constexpr uint64_t kOpDebugSize = 2048U;
const std::string OP_DEBUG_NAME = "Node_OpDebug";
const std::string OP_DEBUG_TYPE = "Opdebug";
}  // namespace

DataDumpImpl::DataDumpImpl() = default;

DataDumpImpl::~DataDumpImpl() {
  Clear();
}

Status DataDumpImpl::SaveTask(const GertModelTaskDesc &task_info, ModelTaskType task_type, rtStream_t stream,
                              bool is_op_debug) {
  const char *op_name = (task_info.op_name != nullptr) ? task_info.op_name : "";
  const char *op_type = (task_info.op_type != nullptr) ? task_info.op_type : "";
  GELOGD("SaveTask: op_name=%s, task_id=%u, stream_id=%u, is_op_debug=%d", op_name, task_info.task_id,
         task_info.stream_id, is_op_debug);

  InnerDumpInfo dump_info = {};
  dump_info.task_id = static_cast<uint32_t>(task_info.task_id);
  dump_info.stream_id = static_cast<uint32_t>(task_info.stream_id);
  dump_info.context_id = static_cast<uint32_t>(task_info.context_id);
  dump_info.thread_id = static_cast<uint32_t>(task_info.thread_id);
  dump_info.task_type = task_type;
  dump_info.stream = stream;
  dump_info.is_op_debug = is_op_debug;
  dump_info.args_base = task_info.args_base;
  dump_info.args_size = task_info.args_size;
  dump_info.op_name = op_name;
  dump_info.op_type = op_type;
  dump_info.is_raw_address = task_info.is_raw_address;

  const auto copy_io_entries = [](const GertModelTaskIoEntry *entries, const uint32_t entry_num,
                                  std::vector<InnerTensorInfo> &inner_tensors) -> Status {
    if ((entry_num > 0U) && (entries == nullptr)) {
      GELOGE(PARAM_INVALID, "[Check][Param] OM2 task io entries is null, entry_num=%u.", entry_num);
      return PARAM_INVALID;
    }
    inner_tensors.reserve(entry_num);
    for (uint32_t i = 0U; i < entry_num; ++i) {
      const auto &entry = entries[i];
      if (entry.tensor == nullptr) {
        GELOGE(PARAM_INVALID, "[Check][Param] OM2 task io tensor is null, index=%u.", i);
        return PARAM_INVALID;
      }
      const auto &tensor = *entry.tensor;
      InnerTensorInfo inner_tensor{};
      inner_tensor.offset = entry.offset;
      inner_tensor.device_address = PtrToValue(tensor.GetAddr());
      inner_tensor.size = tensor.GetSize();
      inner_tensor.data_type = tensor.GetDataType();
      inner_tensor.format = tensor.GetStorageFormat();
      if (tensor.GetStorageShape().GetDimNum() > 0U) {
        inner_tensor.shape_dims.clear();
        inner_tensor.shape_dims.reserve(tensor.GetStorageShape().GetDimNum());
        for (auto i = 0U; i < tensor.GetStorageShape().GetDimNum(); ++i) {
          inner_tensor.shape_dims.push_back(tensor.GetStorageShape().GetDim(i));
        }
      }
      inner_tensors.push_back(inner_tensor);
    }
    return SUCCESS;
  };

  Status ret = copy_io_entries(task_info.inputs, task_info.input_num, dump_info.inputs);
  if (ret != SUCCESS) {
    return ret;
  }
  ret = copy_io_entries(task_info.outputs, task_info.output_num, dump_info.outputs);
  if (ret != SUCCESS) {
    return ret;
  }

  if ((task_info.workspace_num > 0U) &&
      ((task_info.workspace_addrs == nullptr) || (task_info.workspace_sizes == nullptr))) {
    GELOGE(PARAM_INVALID, "[Check][Param] OM2 task workspace info is null, workspace_num=%u.", task_info.workspace_num);
    return PARAM_INVALID;
  }
  for (uint32_t i = 0U; i < task_info.workspace_num; ++i) {
    dump_info.workspace_addrs.push_back(task_info.workspace_addrs[i]);
    dump_info.workspace_sizes.push_back(task_info.workspace_sizes[i]);
  }

  task_list_.push_back(dump_info);
  return SUCCESS;
}

Status DataDumpImpl::ExecuteLoadDumpInfo(const std::vector<uint8_t> &payload) {
  const size_t payload_size = payload.size();
  if (payload_size == 0U || payload_size > UINT32_MAX) {
    GELOGE(PARAM_INVALID, "Invalid dump payload size %zu.", payload_size);
    return PARAM_INVALID;
  }

  if (dev_mem_load_ != nullptr) {
    GELOGW("dev_mem_load_ has been used.");
    (void)aclrtFree(dev_mem_load_);
    dev_mem_load_ = nullptr;
  }

  aclError rt_ret = aclrtMalloc(&dev_mem_load_, payload_size, ACL_MEM_MALLOC_HUGE_FIRST);
  if (rt_ret != ACL_SUCCESS) {
    GELOGE(RT_FAILED, "[Call][aclrtMalloc] failed, size:%zu, ret:%d", payload_size, rt_ret);
    return RT_FAILED;
  }

  rt_ret = aclrtMemcpy(dev_mem_load_, payload_size, payload.data(), payload_size, ACL_MEMCPY_HOST_TO_DEVICE);
  if (rt_ret != ACL_SUCCESS) {
    GELOGE(RT_FAILED, "[Call][aclrtMemcpy] failed, size:%zu, ret:%d", payload_size, rt_ret);
    (void)aclrtFree(dev_mem_load_);
    dev_mem_load_ = nullptr;
    return RT_FAILED;
  }

  rt_ret = rtDatadumpInfoLoad(dev_mem_load_, static_cast<uint32_t>(payload_size));
  if (rt_ret != RT_ERROR_NONE) {
    GELOGE(RT_FAILED, "[Call][rtDatadumpInfoLoad] failed, length:%zu, ret:%d", payload_size, rt_ret);
    (void)aclrtFree(dev_mem_load_);
    dev_mem_load_ = nullptr;
    return RT_FAILED;
  }

  load_flag_ = true;
  GELOGI("LoadDumpInfo success, payload size is: %zu.", payload_size);
  return SUCCESS;
}

Status DataDumpImpl::BuildDumpTransportBasicInfo(const ModelDumpInfo &model_info, DumpTransportInfo &result) {
  if (dump_transport_base_info_initialized_) {
    result = dump_transport_base_info_;
    return ge::SUCCESS;
  }
  auto &dump_transport_info = dump_transport_base_info_.MutableModel();
  const char *model_name = (model_info.model_name != nullptr) ? model_info.model_name : "";
  dump_transport_info.SetDumpPath(DumpConfig::Instance().GetDumpPath() + std::to_string(model_info.device_id) + "/");
  dump_transport_info.SetModelName(model_name);
  dump_transport_info.SetModelId(model_info.model_id);
  dump_transport_info.SetDumpStep(DumpConfig::Instance().GetDumpStep());
  dump_transport_info.SetFlag(kAicpuLoadFlag);

  // step_id_addr 分配设备内存并初始化为 0，先释放旧的
  if (step_id_dev_addr_ != nullptr) {
    (void)aclrtFree(step_id_dev_addr_);
    step_id_dev_addr_ = nullptr;
  }

  void *step_id_dev_addr = nullptr;
  const aclError ret = aclrtMalloc(&step_id_dev_addr, sizeof(uint32_t), ACL_MEM_MALLOC_HUGE_FIRST);
  if (ret != ACL_SUCCESS) {
    GELOGE(RT_FAILED, "Malloc step_id_addr failed, ret=%d", ret);
    return RT_FAILED;
  }
  const uint32_t zero_val = 0U;
  const aclError cpy_ret =
      aclrtMemcpy(step_id_dev_addr, sizeof(uint32_t), &zero_val, sizeof(uint32_t), ACL_MEMCPY_HOST_TO_DEVICE);
  if (cpy_ret != ACL_SUCCESS) {
    GELOGE(RT_FAILED, "Memcpy step_id_addr failed, ret=%d", cpy_ret);
    (void)aclrtFree(step_id_dev_addr);
    return RT_FAILED;
  }
  step_id_dev_addr_ = step_id_dev_addr;
  dump_transport_info.SetStepIdAddr(PtrToValue(step_id_dev_addr));

  // loop_cond_addr 和 iterations_per_loop_addr 保持原逻辑
  if (model_info.loop_cond_addr != 0U) {
    dump_transport_info.SetLoopCondAddr(model_info.loop_cond_addr);
  }
  if (model_info.iterations_per_loop_addr != 0U) {
    dump_transport_info.SetIterationsPerLoopAddr(model_info.iterations_per_loop_addr);
  }

  // 设置 dump_data
  const std::string dump_data_str = DumpConfig::Instance().GetDumpData();
  if (dump_data_str == "stats") {
    dump_transport_info.SetDumpData(DumpTransDumpData::kStats);
  } else {
    dump_transport_info.SetDumpData(DumpTransDumpData::kTensor);
  }
  result = dump_transport_base_info_;
  dump_transport_base_info_initialized_ = true;
  return ge::SUCCESS;
}

Status DataDumpImpl::BuildTaskList(DumpTransportInfo &dump_transport_info) const {
  for (const auto &dump_info : task_list_) {
    DumpTransTaskInfo *task = &dump_transport_info.AddTask();
    GE_CHECK_NOTNULL(task);
    GELOGD("BuildTaskList: task_id=%u, stream_id=%u, args_base=0x%lx, args_size=%zu, is_raw_address=%u",
           dump_info.task_id, dump_info.stream_id, dump_info.args_base, dump_info.args_size, dump_info.is_raw_address);
    task->SetTaskId(dump_info.task_id);
    task->SetStreamId(dump_info.stream_id);
    task->SetContextId(dump_info.context_id);
    task->SetThreadId(dump_info.thread_id);

    task->SetOpName(dump_info.op_name);
    task->SetOpType(dump_info.op_type);
    task->SetEndGraph(false);
    task->SetTaskType(DumpTransTaskType::kAiCore);

    BuildTaskInputs(dump_info, *task);
    BuildTaskOutputs(dump_info, *task);
    BuildTaskWorkspaces(dump_info, *task);
  }
  return SUCCESS;
}

void DataDumpImpl::BuildTaskInputs(const InnerDumpInfo &dump_info, DumpTransTaskInfo &task) const {
  const std::string &dump_mode = DumpConfig::Instance().GetDumpMode();
  const bool need_dump_input =
      (dump_mode == GE_DUMP_MODE_INPUT) || (dump_mode == GE_DUMP_MODE_ALL) || dump_info.is_op_debug;
  if (!need_dump_input) {
    GELOGD("Skip dump input for task_id=%u, dump_mode=%s, is_op_debug=%u", dump_info.task_id, dump_mode.c_str(),
           dump_info.is_op_debug);
    return;
  }

  for (size_t i = 0; i < dump_info.inputs.size(); ++i) {
    DumpTransInputInfo *input_tensor = &task.AddInput();
    const auto &tensor = dump_info.inputs[i];

    // 对齐 v1：直接写 args + offset 地址给 AICPU，不解引用
    uint64_t device_address = tensor.device_address;
    auto addr_type = DumpTransAddressType::kTraditional;
    if (dump_info.is_raw_address) {
      addr_type = DumpTransAddressType::kRaw;
    } else if (dump_info.args_base != 0U && tensor.offset != UINT64_MAX) {
      device_address = dump_info.args_base + tensor.offset;
    }

    GELOGD(
        "BuildTaskInputs: task_id=%u, input[%zu], args_base=0x%lx, offset=0x%lx, device_address=0x%lx, size=%lu, "
        "is_raw=%d",
        dump_info.task_id, i, dump_info.args_base, static_cast<uint64_t>(tensor.offset), device_address, tensor.size,
        dump_info.is_raw_address);

    input_tensor->SetDataType(tensor.data_type);
    input_tensor->SetFormat(tensor.format);
    input_tensor->SetAddress(device_address);
    input_tensor->SetSize(tensor.size);
    input_tensor->SetAddrType(addr_type);
    input_tensor->SetShape(std::vector<uint64_t>(tensor.shape_dims.begin(), tensor.shape_dims.end()));
    input_tensor->SetOriginShape({});
    input_tensor->SetOffset(0U);
  }
}

void DataDumpImpl::BuildTaskOutputs(const InnerDumpInfo &dump_info, DumpTransTaskInfo &task) const {
  const std::string &dump_mode = DumpConfig::Instance().GetDumpMode();
  const bool need_dump_output =
      (dump_mode == GE_DUMP_MODE_OUTPUT) || (dump_mode == GE_DUMP_MODE_ALL) || dump_info.is_op_debug;
  if (!need_dump_output) {
    GELOGD("Skip dump output for task_id=%u, dump_mode=%s, is_op_debug=%u", dump_info.task_id, dump_mode.c_str(),
           dump_info.is_op_debug);
    return;
  }

  for (size_t i = 0; i < dump_info.outputs.size(); ++i) {
    DumpTransOutputInfo *output_tensor = &task.AddOutput();
    const auto &tensor = dump_info.outputs[i];

    // 对齐 v1：直接写 args + offset 地址给 AICPU，不解引用
    uint64_t device_address = tensor.device_address;
    auto addr_type = DumpTransAddressType::kTraditional;
    if (dump_info.is_raw_address) {
      addr_type = DumpTransAddressType::kRaw;
    } else if (dump_info.args_base != 0U && tensor.offset != UINT64_MAX) {
      device_address = dump_info.args_base + tensor.offset;
    }

    GELOGD(
        "BuildTaskOutputs: task_id=%u, output[%zu], args_base=0x%lx, offset=0x%lx, device_address=0x%lx, size=%lu, "
        "is_raw=%d",
        dump_info.task_id, i, dump_info.args_base, static_cast<uint64_t>(tensor.offset), device_address, tensor.size,
        dump_info.is_raw_address);

    output_tensor->SetDataType(tensor.data_type);
    output_tensor->SetFormat(tensor.format);
    output_tensor->SetAddress(device_address);
    output_tensor->SetSize(tensor.size);
    output_tensor->SetAddrType(addr_type);
    output_tensor->SetShape(std::vector<uint64_t>(tensor.shape_dims.begin(), tensor.shape_dims.end()));
    output_tensor->SetOriginShape({});
    output_tensor->SetOffset(0U);
    output_tensor->SetOriginalName("");
    output_tensor->SetOriginalOutputIndex(0);
    output_tensor->SetOriginalOutputDataType(0);
    output_tensor->SetOriginalOutputFormat(0);
  }
}

void DataDumpImpl::BuildTaskWorkspaces(const InnerDumpInfo &dump_info, DumpTransTaskInfo &task) const {
  // workspace 只在 op_debug 模式下才需要 dump（用于溢出/异常调试）
  if (!dump_info.is_op_debug) {
    GELOGD("Skip dump workspace for task_id=%u, is_op_debug=%u", dump_info.task_id, dump_info.is_op_debug);
    return;
  }

  for (size_t i = 0; i < dump_info.workspace_addrs.size(); ++i) {
    DumpTransWorkspaceInfo *workspace = &task.AddWorkspace();
    GELOGD("BuildTaskWorkspaces: task_id=%u, workspace[%zu], data_addr=0x%lx, size=%lu", dump_info.task_id, i,
           dump_info.workspace_addrs[i], dump_info.workspace_sizes[i]);
    workspace->SetDataAddr(dump_info.workspace_addrs[i]);
    workspace->SetSize(dump_info.workspace_sizes[i]);
    workspace->SetType(DumpTransWorkspaceType::kLog);
  }
}

void DataDumpImpl::SetOpDebugInfo(uint32_t task_id, uint32_t stream_id, void *debug_addr) {
  is_op_debug_ = true;
  op_debug_task_id_ = task_id;
  op_debug_stream_id_ = stream_id;
  op_debug_addr_ = debug_addr;
}

void DataDumpImpl::BuildOpDebugTask(DumpTransportInfo &dump_transport_info) const {
  if (!is_op_debug_) {
    return;
  }

  GELOGI("Add op_debug_info to aicpu, task_id=%u, stream_id=%u", op_debug_task_id_, op_debug_stream_id_);

  auto &task = dump_transport_info.AddTask();
  task.SetEndGraph(false);
  task.SetTaskId(op_debug_task_id_);
  task.SetStreamId(op_debug_stream_id_);
  task.SetOpName(OP_DEBUG_NAME);
  task.SetOpType(OP_DEBUG_TYPE);

  // set output
  auto &output = task.AddOutput();
  output.SetOriginalName(OP_DEBUG_NAME);
  output.SetOriginalOutputIndex(0);
  output.SetOriginalOutputFormat(FORMAT_ND);
  output.SetOriginalOutputDataType(DT_UINT8);
  output.SetDataType(DT_UINT8);
  output.SetFormat(FORMAT_ND);
  output.SetShape({kOpDebugShape});
  output.SetAddress(PtrToValue(op_debug_addr_));
  output.SetSize(kOpDebugSize);
  output.SetAddrType(DumpTransAddressType::kTraditional);

  output.SetOriginShape({});
  output.SetOffset(0U);
  task.SetContextId(0U);
  task.SetThreadId(0U);
  task.SetTaskType(DumpTransTaskType::kAiCore);
}

Status DataDumpImpl::BuildAndLoadDumpTransportInfo(const ModelDumpInfo &model_info) {
  GELOGI("BuildAndLoadDumpTransportInfo: model_id=%u, task_count=%zu, dump_data=%s, dump_mode=%s, is_op_debug=%d",
         model_info.model_id, task_list_.size(), DumpConfig::Instance().GetDumpData().c_str(),
         DumpConfig::Instance().GetDumpMode().c_str(), is_op_debug_);

  // 如果没有普通算子要 dump，也没有 overflow dump，直接返回
  if (task_list_.empty() && !is_op_debug_) {
    GELOGI("No task to dump, skip build and load op mapping info");
    return SUCCESS;
  }

  DumpTransportInfo dump_transport_info;
  Status ret = BuildDumpTransportBasicInfo(model_info, dump_transport_info);
  if (ret != SUCCESS) {
    GELOGE(ret, "Build op mapping basic info failed, ret=%u", ret);
    return ret;
  }

  ret = BuildTaskList(dump_transport_info);
  if (ret != SUCCESS) {
    GELOGE(ret, "[Build][TaskList] failed, ret:%u", ret);
    return ret;
  }

  // 添加 overflow dump 的特殊 Task（包含 p2p_debug_addr）
  BuildOpDebugTask(dump_transport_info);

  std::vector<uint8_t> payload;
  const auto encode_status = dump_transport_info.Serialize(payload);
  if (encode_status != DumpTransStatus::kOk) {
    GELOGE(PARAM_INVALID, "Serialize dump payload failed, model_id=%u, version=%u, status=%u.", model_info.model_id,
           dump_transport_info.GetVersion(), static_cast<uint32_t>(encode_status));
    return PARAM_INVALID;
  }
  ret = ExecuteLoadDumpInfo(payload);
  if (ret != SUCCESS) {
    GELOGE(ret, "[Execute][LoadDumpInfo] failed, ret:%u", ret);
    return ret;
  }

  dump_transport_info_ = std::move(dump_transport_info);
  GELOGI("BuildAndLoadDumpTransportInfo success, task_count=%zu", task_list_.size());
  return SUCCESS;
}

void DataDumpImpl::Clear() {
  if (dev_mem_load_ != nullptr) {
    (void)aclrtFree(dev_mem_load_);
    dev_mem_load_ = nullptr;
  }
  if (step_id_dev_addr_ != nullptr) {
    (void)aclrtFree(step_id_dev_addr_);
    step_id_dev_addr_ = nullptr;
  }
  task_list_.clear();
  dump_transport_info_.Clear();
  dump_transport_base_info_.Clear();
  dump_transport_base_info_initialized_ = false;
  load_flag_ = false;
}

}  // namespace dump
}  // namespace ge
