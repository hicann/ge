/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "framework/runtime/dump/dump_op_impl.h"
#include "framework/common/debug/ge_log.h"
#include "rt_external.h"
#include "acl/acl_rt.h"
#include "aicpu_task_struct.h"
#include "proto/op_mapping.pb.h"

namespace ge {
namespace dump {
namespace {
const std::string kDumpKernelsDumpOp = "DumpDataInfo";

// Local replacement of ge::AclrtMalloc: device (HBM) memory is allocated through the
// ACL high-bandwidth allocation policy, which is the only memory type ever requested here.
aclError DumpAclrtMalloc(void **ptr, size_t size) {
  if (ptr == nullptr) {
    return ACL_ERROR_INVALID_PARAM;
  }
  *ptr = nullptr;
  return aclrtMalloc(ptr, size, ACL_MEM_TYPE_HIGH_BAND_WIDTH);
}
}  // namespace

DumpOp::~DumpOp() {
  if (proto_dev_mem_ != nullptr) {
    (void)aclrtFree(proto_dev_mem_);
    proto_dev_mem_ = nullptr;
  }
  proto_dev_mem_capacity_ = 0U;

  if (proto_size_dev_mem_ != nullptr) {
    (void)aclrtFree(proto_size_dev_mem_);
    proto_size_dev_mem_ = nullptr;
  }
}

Status DumpOp::ExecutorDumpOp(const std::string &op_name, aclrtStream stream) {
  std::string proto_msg;
  const size_t proto_size = op_mapping_info_.ByteSizeLong();
  const bool ret = op_mapping_info_.SerializeToString(&proto_msg);
  if ((!ret) || (proto_size == 0U)) {
    GELOGE(ACL_ERROR_GE_INTERNAL_ERROR, "[Serialize][Protobuf]Failed, proto_size is %zu", proto_size);
    REPORT_INNER_ERR_MSG("E19999", "[Serialize][Protobuf]Failed, proto_size is %zu", proto_size);
    return ACL_ERROR_GE_INTERNAL_ERROR;
  }

  const Status status = ProtoMallocAndMemcpy(proto_size, proto_msg);
  if (status != SUCCESS) {
    return status;
  }

  constexpr uint32_t io_addr_num = 2U;
  constexpr uint32_t args_size =
      static_cast<uint32_t>(sizeof(aicpu::AicpuParamHead)) + (io_addr_num * static_cast<uint32_t>(sizeof(uint64_t)));
  std::array<uint8_t, args_size> args = {};
  size_t args_pos = 0UL;
  aicpu::AicpuParamHead &param_head = *(static_cast<aicpu::AicpuParamHead *>(static_cast<void *>(&args[args_pos])));
  args_pos += sizeof(aicpu::AicpuParamHead);
  param_head.length = args_size;
  param_head.ioAddrNum = io_addr_num;
  *(static_cast<uint64_t *>(static_cast<void *>(&args[args_pos]))) =
      static_cast<uint64_t>(reinterpret_cast<uintptr_t>(proto_dev_mem_));
  args_pos += sizeof(uint64_t);
  *(reinterpret_cast<uint64_t *>(static_cast<void *>(&args[args_pos]))) =
      static_cast<uint64_t>(reinterpret_cast<uintptr_t>(proto_size_dev_mem_));
  rtArgsEx_t args_for_launch = {};
  args_for_launch.args = &args[0U];
  args_for_launch.isNoNeedH2DCopy = 0U;
  args_for_launch.argsSize = args_size;
  const rtError_t rt_ret = rtCpuKernelLaunchWithFlag(nullptr, kDumpKernelsDumpOp.c_str(), 1U, &args_for_launch, nullptr,
                                                     stream, RT_KERNEL_DEFAULT);
  if (rt_ret != RT_ERROR_NONE) {
    GELOGE(RT_ERROR_TO_GE_STATUS(rt_ret), "[Call][rtCpuKernelLaunchWithFlag]Failed, ret %d", rt_ret);
    REPORT_INNER_ERR_MSG("E19999", "Call rtCpuKernelLaunchWithFlag failed, ret %d", rt_ret);
    return RT_ERROR_TO_GE_STATUS(rt_ret);
  }
  GELOGI("Kernel launch dump op %s success", op_name.c_str());
  return SUCCESS;
}

Status DumpOp::ProtoMallocAndMemcpy(const size_t proto_size, const std::string &proto_msg) {
  size_t proto_capacity = proto_size;

  GE_FREE_RT_LOG(proto_dev_mem_);
  proto_dev_mem_capacity_ = 0U;
  aclError rt_ret = DumpAclrtMalloc(&proto_dev_mem_, proto_capacity);
  if (rt_ret != ACL_SUCCESS) {
    GELOGE(RT_ERROR_TO_GE_STATUS(rt_ret), "[Call][aclrtMalloc]Failed, ret: %d", rt_ret);
    REPORT_INNER_ERR_MSG("E19999", "Call aclrtMalloc failed, ret: %d", rt_ret);
    return RT_ERROR_TO_GE_STATUS(rt_ret);
  }
  proto_dev_mem_capacity_ = proto_capacity;

  rt_ret =
      aclrtMemcpy(proto_dev_mem_, proto_dev_mem_capacity_, proto_msg.c_str(), proto_size, ACL_MEMCPY_HOST_TO_DEVICE);
  if (rt_ret != ACL_SUCCESS) {
    GELOGE(RT_ERROR_TO_GE_STATUS(rt_ret), "[Call][aclrtMemcpy]Failed, ret: %d", rt_ret);
    REPORT_INNER_ERR_MSG("E19999", "Call aclrtMemcpy failed, ret: %d", rt_ret);
    return RT_ERROR_TO_GE_STATUS(rt_ret);
  }

  GE_FREE_RT_LOG(proto_size_dev_mem_);
  rt_ret = DumpAclrtMalloc(&proto_size_dev_mem_, sizeof(size_t));
  if (rt_ret != ACL_SUCCESS) {
    GELOGE(RT_ERROR_TO_GE_STATUS(rt_ret), "[Call][aclrtMalloc]Failed, ret: %d", rt_ret);
    REPORT_INNER_ERR_MSG("E19999", "Call aclrtMalloc failed, ret: %d", rt_ret);
    return RT_ERROR_TO_GE_STATUS(rt_ret);
  }
  rt_ret = aclrtMemcpy(proto_size_dev_mem_, sizeof(size_t), &proto_size, sizeof(size_t), ACL_MEMCPY_HOST_TO_DEVICE);
  if (rt_ret != ACL_SUCCESS) {
    GELOGE(RT_ERROR_TO_GE_STATUS(rt_ret), "[Call][aclrtMemcpy]Failed, ret %d", rt_ret);
    REPORT_INNER_ERR_MSG("E19999", "Call aclrtMemcpy failed, ret %d", rt_ret);
    return RT_ERROR_TO_GE_STATUS(rt_ret);
  }
  return SUCCESS;
}

Status DumpOp::BuildTaskInputs(const GertModelTaskDesc &task_desc) {
  toolkit::aicpu::dump::Task *task = op_mapping_info_.add_task();
  GE_CHECK_NOTNULL(task);
  aclError rt_ret = SetTaskBasicInfo(task_desc, task);
  if (rt_ret != ACL_SUCCESS) {
    GELOGE(RT_ERROR_TO_GE_STATUS(rt_ret), "[Call][SetTaskBasicInfo]Failed, ret: %d", rt_ret);
    return RT_ERROR_TO_GE_STATUS(rt_ret);
  }

  if (task_desc.inputs == nullptr) {
    return SUCCESS;
  }

  for (uint32_t i = 0U; i < task_desc.input_num; ++i) {
    toolkit::aicpu::dump::Input *input_tensor = task->add_input();
    const auto &entry = task_desc.inputs[i];
    if (entry.tensor == nullptr) {
      GELOGE(PARAM_INVALID, "[Check][Param] OM2 task io tensor is null, index=%u.", i);
      return PARAM_INVALID;
    }
    const auto &tensor = *entry.tensor;

    // 对齐 v1：直接写 args + offset 地址给 AICPU，不解引用
    uint64_t device_address = reinterpret_cast<uint64_t>(tensor.GetAddr());
    auto addr_type = toolkit::aicpu::dump::AddressType::TRADITIONAL_ADDR;

    GELOGD("BuildTaskInputs: task_id=%u, input[%zu], device_address=0x%lx, size=%lu", task_desc.task_id, i,
           device_address, tensor.GetSize());
    input_tensor->set_data_type(tensor.GetDataType());
    input_tensor->set_format(tensor.GetStorageFormat());
    input_tensor->set_address(device_address);
    input_tensor->set_size(tensor.GetSize());
    input_tensor->set_addr_type(addr_type);
    for (auto i = 0U; i < tensor.GetStorageShape().GetDimNum(); ++i) {
      input_tensor->mutable_shape()->add_dim(tensor.GetStorageShape().GetDim(i));
    }
  }

  return SUCCESS;
}

Status DumpOp::BuildTaskOutputs(const GertModelTaskDesc &task_desc) {
  toolkit::aicpu::dump::Task *task = op_mapping_info_.add_task();
  GE_CHECK_NOTNULL(task);
  aclError rt_ret = SetTaskBasicInfo(task_desc, task);
  if (rt_ret != ACL_SUCCESS) {
    GELOGE(RT_ERROR_TO_GE_STATUS(rt_ret), "[Call][SetTaskBasicInfo]Failed, ret: %d", rt_ret);
    return RT_ERROR_TO_GE_STATUS(rt_ret);
  }

  if (task_desc.outputs == nullptr) {
    return SUCCESS;
  }

  for (uint32_t i = 0U; i < task_desc.output_num; ++i) {
    toolkit::aicpu::dump::Output *output_tensor = task->add_output();
    const auto &entry = task_desc.outputs[i];
    if (entry.tensor == nullptr) {
      GELOGE(PARAM_INVALID, "[Check][Param] OM2 task io tensor is null, index=%u.", i);
      return PARAM_INVALID;
    }
    const auto &tensor = *entry.tensor;

    // 对齐 v1：直接写 args + offset 地址给 AICPU，不解引用
    uint64_t device_address = reinterpret_cast<uint64_t>(tensor.GetAddr());
    auto addr_type = toolkit::aicpu::dump::AddressType::TRADITIONAL_ADDR;

    GELOGD("BuildTaskOutputs: task_id=%u, input[%zu], device_address=0x%lx, size=%lu", task_desc.task_id, i,
           device_address, tensor.GetSize());
    output_tensor->set_data_type(tensor.GetDataType());
    output_tensor->set_format(tensor.GetStorageFormat());
    output_tensor->set_address(device_address);
    output_tensor->set_size(tensor.GetSize());
    output_tensor->set_addr_type(addr_type);
    for (auto i = 0U; i < tensor.GetStorageShape().GetDimNum(); ++i) {
      output_tensor->mutable_shape()->add_dim(tensor.GetStorageShape().GetDim(i));
    }
  }

  return SUCCESS;
}

Status DumpOp::SetTaskBasicInfo(const GertModelTaskDesc &task_desc, toolkit::aicpu::dump::Task *task) {
  const char *op_name = (task_desc.op_name != nullptr) ? task_desc.op_name : "";
  const char *op_type = (task_desc.op_type != nullptr) ? task_desc.op_type : "";
  GELOGD("SetTaskBasicInfo: op_name=%s, task_id=%u, stream_id=%u", op_name, task_desc.task_id, task_desc.stream_id);
  GE_CHECK_NOTNULL(task);
  task->set_task_id(task_desc.task_id);
  task->set_stream_id(task_desc.stream_id);
  task->set_context_id(task_desc.context_id);
  task->set_thread_id(task_desc.thread_id);

  // 设置 op 信息
  toolkit::aicpu::dump::Op *op = task->mutable_op();
  if (op != nullptr) {
    op->set_op_name(op_name);
    op->set_op_type(op_type);
  }
  return SUCCESS;
}
}  // namespace dump
}  // namespace ge
