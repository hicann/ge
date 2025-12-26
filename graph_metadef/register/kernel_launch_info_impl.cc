/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "kernel_launch_info_impl.h"
#include "graph/debug/ge_util.h"
#include "common/checker.h"
#include "runtime/rt_model.h"
#include "ge/framework/common/taskdown_common.h"
#include "common/opskernel/ops_kernel_info_types.h"

namespace ge {
namespace {
bool IsAllKernel(const domi::TaskDef &task_def) {
  return (static_cast<ModelTaskType>(task_def.type()) == ModelTaskType::MODEL_TASK_ALL_KERNEL) ||
      (static_cast<ModelTaskType>(task_def.type()) == ModelTaskType::MODEL_TASK_VECTOR_ALL_KERNEL);
}
}
KernelLaunchInfoImplPtr KernelLaunchInfoImpl::LoadFromData(const gert::ExeResGenerationContext *context,
    const std::vector<uint8_t> &data) {
  GE_ASSERT_NOTNULL(context);
  auto impl_ptr = ComGraphMakeUnique<KernelLaunchInfoImpl>();
  GE_ASSERT_NOTNULL(impl_ptr);
  GE_ASSERT_TRUE(impl_ptr->task_def_.ParseFromArray(data.data(), data.size()));
  impl_ptr->context_ = const_cast<gert::ExeResGenerationContext *>(context);
  return impl_ptr;
}
KernelLaunchInfoImplPtr KernelLaunchInfoImpl::CreateAicpuKfcTask(const gert::ExeResGenerationContext *context,
    const char *so_name, const char *kernel_name) {
  GE_ASSERT_NOTNULL(context);
  auto impl_ptr = ComGraphMakeUnique<KernelLaunchInfoImpl>();
  GE_ASSERT_NOTNULL(impl_ptr);
  impl_ptr->context_ = const_cast<gert::ExeResGenerationContext *>(context);
  impl_ptr->task_def_.set_type(static_cast<int32_t>(ModelTaskType::MODEL_TASK_KERNEL));
  auto kernel_def = impl_ptr->task_def_.mutable_kernel();
  GE_ASSERT_NOTNULL(kernel_def);
  kernel_def->set_so_name(so_name);
  kernel_def->set_kernel_name(kernel_name);
  auto kernel_context = kernel_def->mutable_context();
  GE_ASSERT_NOTNULL(kernel_context);
  kernel_context->set_kernel_type(static_cast<uint32_t>(ccKernelType::AI_CPU_KFC));
  kernel_context->set_op_index(context->GetOpId());
  return impl_ptr;
}

KernelLaunchInfoImplPtr KernelLaunchInfoImpl::CreateHcomRecordTask(const gert::ExeResGenerationContext *context,
    const char *group_name) {
  GE_ASSERT_NOTNULL(context);
  GE_ASSERT_NOTNULL(group_name);
  auto impl_ptr = ComGraphMakeUnique<KernelLaunchInfoImpl>();
  GE_ASSERT_NOTNULL(impl_ptr);
  impl_ptr->context_ = const_cast<gert::ExeResGenerationContext *>(context);
  impl_ptr->task_def_.set_id(context->GetOpId());
  impl_ptr->task_def_.set_notify_id(UINT32_MAX);
  impl_ptr->task_def_.set_type(static_cast<int32_t>(ModelTaskType::MODEL_TASK_NOTIFY_RECORD));
  impl_ptr->task_def_.set_private_def(group_name);
  return impl_ptr;
}

KernelLaunchInfoImplPtr KernelLaunchInfoImpl::CreateHcomWaitTask(const gert::ExeResGenerationContext *context,
    const char *group_name) {
  GE_ASSERT_NOTNULL(context);
  GE_ASSERT_NOTNULL(group_name);
  auto impl_ptr = ComGraphMakeUnique<KernelLaunchInfoImpl>();
  GE_ASSERT_NOTNULL(impl_ptr);
  impl_ptr->context_ = const_cast<gert::ExeResGenerationContext *>(context);
  impl_ptr->task_def_.set_id(context->GetOpId());
  impl_ptr->task_def_.set_notify_id(UINT32_MAX);
  impl_ptr->task_def_.set_type(static_cast<int32_t>(ModelTaskType::MODEL_TASK_NOTIFY_WAIT));
  impl_ptr->task_def_.set_private_def(group_name);
  return impl_ptr;
}

std::vector<uint8_t> KernelLaunchInfoImpl::Serialize() {
  auto buffer_size = task_def_.ByteSizeLong();
  std::vector<uint8_t> buffer(buffer_size, 0);
  GE_ASSERT_TRUE(task_def_.SerializeToArray(buffer.data(), buffer_size));
  return buffer;
}
uint32_t KernelLaunchInfoImpl::GetStreamId() const {
  return task_def_.stream_id();
}
void KernelLaunchInfoImpl::SetStreamId(uint32_t stream_id) {
  task_def_.set_stream_id(stream_id);
}

uint32_t KernelLaunchInfoImpl::GetBlockDim() const {
  uint32_t block_dim = 0;
  if (static_cast<ModelTaskType>(task_def_.type()) == ModelTaskType::MODEL_TASK_KERNEL) {
    block_dim = task_def_.kernel().block_dim();
  } else if (IsAllKernel(task_def_)) {
    block_dim = task_def_.kernel_with_handle().block_dim();
  } else {
    GELOGE(FAILED, "Only aicpu and aicore task has block_dim, but get[%d]",
        task_def_.type());
  }
  return block_dim;
}

graphStatus KernelLaunchInfoImpl::SetBlockDim(uint32_t block_dim) {
  if (static_cast<ModelTaskType>(task_def_.type()) == ModelTaskType::MODEL_TASK_KERNEL) {
    auto kernel_def = task_def_.mutable_kernel();
    GE_ASSERT_NOTNULL(kernel_def);
    kernel_def->set_block_dim(block_dim);
  } else if (IsAllKernel(task_def_)) {
    auto kernel_with_handle = task_def_.mutable_kernel_with_handle();
    GE_ASSERT_NOTNULL(kernel_with_handle);
    kernel_with_handle->set_block_dim(block_dim);
  } else {
    // 报错
    GE_ASSERT_TRUE(false, "Only aicpu and aicore task can set args format, but get[%d]",
        task_def_.type());
  }
  return SUCCESS;
}

const char *KernelLaunchInfoImpl::GetArgsFormat() const {
  domi::KernelContext kernel_context;
  if (static_cast<ModelTaskType>(task_def_.type()) == ModelTaskType::MODEL_TASK_KERNEL) {
    return task_def_.kernel().context().args_format().c_str();
  }
  if (IsAllKernel(task_def_)) {
    return task_def_.kernel_with_handle().context().args_format().c_str();
  }
  GELOGE(FAILED, "Only aicpu and aicore task has args format, but get[%d]",
      task_def_.type());
  return nullptr;
}
graphStatus KernelLaunchInfoImpl::SetArgsFormat(const char *args_format) {
  GE_ASSERT_NOTNULL(args_format);
  domi::KernelContext *kernel_context = nullptr;
  if (static_cast<ModelTaskType>(task_def_.type()) == ModelTaskType::MODEL_TASK_KERNEL) {
    auto kernel_def = task_def_.mutable_kernel();
    GE_ASSERT_NOTNULL(kernel_def);
    kernel_context = kernel_def->mutable_context();
  } else if (IsAllKernel(task_def_)) {
    auto kernel_with_handle = task_def_.mutable_kernel_with_handle();
    GE_ASSERT_NOTNULL(kernel_with_handle);
    kernel_context = kernel_with_handle->mutable_context();
  } else {
    GELOGE(FAILED, "Only aicpu and aicore task can set args format, but get[%d]",
        task_def_.type());
  }
  GE_ASSERT_NOTNULL(kernel_context);
  kernel_context->set_args_format(args_format);
  return SUCCESS;
}

const char *KernelLaunchInfoImpl::GetSoName() const {
  return task_def_.kernel().so_name().c_str();
}
const char *KernelLaunchInfoImpl::GetKernelName() const {
  return task_def_.kernel().kernel_name().c_str();
}
}