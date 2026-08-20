/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "aicpu_launch_kernel.h"

#include <cstddef>
#include <iomanip>
#include <cinttypes>
#include <memory>
#include <new>
#include "aicpu_ext_info_handle.h"
#include "graph/error_codes.h"
#include "register/kernel_registry.h"
#include "framework/common/debug/log.h"
#include "exe_graph/runtime/runtime_tensor.h"
#include "kernel/memory/mem_block.h"
#include "aicpu_engine_struct.h"
#include "rt_external_mem.h"
#include "common/checker.h"
#include "rt_external_kernel.h"
#include "aicpu_task_struct.h"
#include "framework/common/taskdown_common.h"
#include "proto/task.pb.h"
#include "common/plugin/ge_make_unique_util.h"
#include "core/debug/kernel_tracing.h"
#include "core/executor/multi_thread_topological/executor/schedule/producer/producers/kernel_tags/critical_section_config.h"
#include "aicpu_resource_manager.h"
#include "fused_host_cpu_compute.h"
#include "engine/aicpu/graph_builder/bg_aicpu_arg.h"
#include "aicpu_args_handler.h"
#include "block_op_utils.h"
#include "kernel/kernel_log.h"
#include "kernel/memory/mem_block.h"
#include "kernel/memory/host_mem_allocator.h"
#include "core/utils/rt2_tensor_utils.h"
#include "exe_graph/runtime/gert_tensor_data.h"
#include "graph/load/model_manager/model_manager.h"
#include "aicpu_bin_handler.h"
#include "exe_graph/runtime/storage_shape.h"

using namespace ge;

namespace gert {
namespace kernel {
namespace {
const int32_t kBlockdimAxisDefaultIndex = -1;
constexpr size_t kTensorListStreamIdx = 0;
constexpr size_t kTensorListPushBackDataIdx = 2;
constexpr size_t kTensorListPushBackOutputHandleIdx = 3;
constexpr size_t kTensorListPopBackInputHandleIdx = 1;

using CustAICPUKernelPtr = std::shared_ptr<ge::OpKernelBin>;

enum class AllocHostCpuOutputMemoryInputs { kSize, kAllocator };

inline int64_t CeilDivisor(const int64_t x, const int64_t base) {
  int64_t ret = 0;
  if (base != 0) {
    ret = x / base;
  }
  if ((base != 0) && (x % base) != 0) {
    ret++;
  }
  return ret;
}

template <typename T, bool IsPointer = std::is_pointer<T>::value>
void PrintHex(const T *p, size_t num, std::stringstream &ss) {
  if (p == nullptr) {
    return;
  }
  constexpr uint64_t TWICE = 2UL;
  for (size_t i = 0; i < num; ++i) {
    if (!IsPointer) {
      ss << "0x" << std::setfill('0') << std::setw(sizeof(T) * TWICE) << std::hex << +p[i] << ' ';
    } else {
      ss << p[i] << ' ';
    }
  }
}

// call after kernel launch
void PrintStreamIdAndTaskId(const KernelContext *context, std::vector<std::string> &msgs) {
  (void)context;
  std::stringstream ss;
  uint32_t stream_id = 0U;
  uint32_t flip_task_id = 0U;
  if (rtGetTaskIdAndStreamID(&flip_task_id, &stream_id) == RT_ERROR_NONE) {
    const uint32_t task_id = flip_task_id & 0xFFFF;  // lower 16bits
    const uint32_t flip_num = flip_task_id >> 16U;   // high 16bits
    ss << "stream_id=" << stream_id << ", task_id=" << task_id << ", flip_num=" << flip_num
       << ", flip_task_id=" << flip_task_id;
  }
  msgs.emplace_back(ss.str());
}

void PrintIoAddresses(const KernelContext &context, std::vector<std::string> &msgs) {
  std::stringstream ss;
  auto handle = context.MutableInputPointer<AicpuArgsHandler>(static_cast<size_t>(AicpuLaunchCommon::kArgsHandler));
  GE_CHECK_NOTNULL_JUST_RETURN(handle);
  auto io_addrs = PtrToPtr<uint8_t, uint64_t>(handle->GetIoAddr());
  auto io_num = handle->GetIoNum();
  ss << " Launch aicpu inputs/outputs addresses: ";
  if (io_num > 0U) {
    PrintHex(io_addrs, io_num, ss);
    ss << ", Launch aicpu inputs/outputs sizes: ";
    for (size_t i = 0U; i < io_num; i++) {
      ss << handle->GetIoSize(i) << " ";
    }
  } else {
    ss << " No inputs/outputs";
  }
  msgs.emplace_back(ss.str());
}

void PrintArgsOffset(const KernelContext &context, std::vector<std::string> &msgs) {
  std::stringstream ss;
  ss << " Launch aicpu args offset: ";
  auto handle = context.GetInputPointer<AicpuArgsHandler>(static_cast<size_t>(AicpuLaunchCommon::kArgsHandler));
  GE_CHECK_NOTNULL_JUST_RETURN(handle);
  auto &args = handle->GetArgsEx();
  ss << " soNameAddrOffset: " << args.soNameAddrOffset << " kernelNameAddrOffset: " << args.kernelNameAddrOffset;
  ss << " hostInputInfoNum: " << args.hostInputInfoNum << " kernelOffsetInfoNum: " << args.kernelOffsetInfoNum;
  for (size_t i = 0U; i < args.hostInputInfoNum; ++i) {
    ss << " host[" << i << "] addr offset is " << args.hostInputInfoPtr[i].addrOffset << " data offset is "
       << args.hostInputInfoPtr[i].dataOffset;
  }
  for (size_t i = 0U; i < args.kernelOffsetInfoNum; ++i) {
    ss << " kernel[" << i << "] addr offset is " << args.kernelOffsetInfoPtr[i].addrOffset << " data offset is "
       << args.kernelOffsetInfoPtr[i].dataOffset;
  }
  msgs.emplace_back(ss.str());
}

std::vector<std::string> PrintCCLaunchArgs(const KernelContext *context) {
  auto args_handler = context->GetInputPointer<AicpuArgsHandler>(static_cast<size_t>(AicpuLaunchCommon::kArgsHandler));
  if (args_handler == nullptr) {
    return {};
  }
  auto stream = context->GetInputValue<rtStream_t>(static_cast<size_t>(AicpuLaunchCommon::kStream));
  auto block_dim = context->GetInputValue<uint32_t>(static_cast<size_t>(AicpuCCLaunch::kBlockDim));
  auto kernel_type = context->GetInputValue<ccKernelType>(static_cast<size_t>(AicpuCCLaunch::kKernelType));

  uint32_t flag = RT_KERNEL_DEFAULT;
  if (kernel_type == ccKernelType::CUST_AI_CPU) {
    flag |= static_cast<uint32_t>(RT_KERNEL_CUSTOM_AICPU);
  }
  std::vector<std::string> msgs;
  std::stringstream ss;
  ss << "Launch cc function arguments: "
     << "op_name " << args_handler->GetNodeName() << ", block_dim " << block_dim << ", stream " << stream
     << ", args_ex " << &(args_handler->GetArgsEx()) << " flag " << flag;
  msgs.emplace_back(ss.str());
  PrintArgsOffset(*context, msgs);
  PrintIoAddresses(*context, msgs);
  PrintStreamIdAndTaskId(context, msgs);
  return msgs;
}

std::vector<std::string> PrintTfLaunchArgs(const KernelContext *context) {
  auto args_handler = context->GetInputPointer<AicpuArgsHandler>(static_cast<size_t>(AicpuLaunchCommon::kArgsHandler));
  if (args_handler == nullptr) {
    return {};
  }
  auto stream = context->GetInputValue<rtStream_t>(static_cast<size_t>(AicpuLaunchCommon::kStream));
  std::vector<std::string> msgs;
  std::stringstream ss;
  ss << "Launch tf function arguments: "
     << "op_name " << args_handler->GetNodeName() << ", stream " << stream << ", args_ex "
     << &(args_handler->GetArgsEx()) << ", io_num " << args_handler->GetIoNum() << " , host input size "
     << args_handler->GetHostInputSize();
  msgs.emplace_back(ss.str());
  PrintArgsOffset(*context, msgs);
  PrintIoAddresses(*context, msgs);
  PrintStreamIdAndTaskId(context, msgs);
  return msgs;
}
}  // namespace

ge::graphStatus UpdateAicpuIoAddr(KernelContext *context) {
  auto args_handler = context->MutableInputPointer<AicpuArgsHandler>(0U);
  GE_ASSERT_NOTNULL(args_handler);

  auto io_num = args_handler->GetIoNum();
  uint64_t *io_addrs = PtrToPtr<uint8_t, uint64_t>(args_handler->GetIoAddr());
  GE_ASSERT_NOTNULL(io_addrs);
  for (size_t i = 1U; i < io_num + 1U; i++) {
    auto tensor_data = context->GetInputValue<gert::GertTensorData *>(i);
    GE_ASSERT_NOTNULL(tensor_data);
    if (!TensorPlacementUtils::IsOnHost(tensor_data->GetPlacement())) {
      *io_addrs = PtrToValue(tensor_data->GetAddr());
      args_handler->SetIoSize((i - 1U), tensor_data->GetSize());
    }
    io_addrs++;
  }
  return SUCCESS;
}
REGISTER_KERNEL(UpdateAicpuIoAddr).RunFunc(UpdateAicpuIoAddr);

ge::graphStatus ExpandAicpuOptionalInputAddrs(KernelContext *context) {
  auto compute_node_info = reinterpret_cast<const ComputeNodeInfo *>(context->GetComputeNodeExtend());
  GE_ASSERT_NOTNULL(compute_node_info);
  const auto node_name = compute_node_info->GetNodeName();
  const size_t total_input_num = context->GetOutputNum();
  GE_ASSERT_TRUE(context->GetInputNum() > 0U);
  const auto empty_input_placement = context->GetInputValue<TensorPlacement>(context->GetInputNum() - 1U);
  const size_t actual_input_num = context->GetInputNum() - 1U;

  size_t actual_input_index = 0U;
  for (size_t input_index = 0U; input_index < total_input_num; ++input_index) {
    auto ins_info = compute_node_info->GetInputInstanceInfo(input_index);
    auto out_tensor_data = context->GetOutputPointer<gert::GertTensorData>(input_index);
    GE_ASSERT_NOTNULL(out_tensor_data);
    if ((ins_info == nullptr) || (ins_info->GetInstanceNum() == 0U)) {
      // -1 means invalid stream id for empty optional input placeholder.
      GE_ASSERT_SUCCESS(out_tensor_data->ShareFrom(gert::GertTensorData(nullptr, 0U, empty_input_placement, -1)));
      GELOGI("Kernel %s expand optional input index %zu to nullptr.", node_name, input_index);
      continue;
    }

    GE_ASSERT_TRUE(actual_input_index < actual_input_num);
    auto in_tensor_data = context->GetInputPointer<gert::GertTensorData>(actual_input_index);
    GE_ASSERT_NOTNULL(in_tensor_data);
    GELOGD("Kernel %s tensor data info is: input index %zu, addr is %p, size is %zu.", node_name, input_index,
           static_cast<void *>(in_tensor_data->GetAddr()), in_tensor_data->GetSize());
    GE_ASSERT_SUCCESS(out_tensor_data->ShareFrom(*in_tensor_data));
    ++actual_input_index;
  }

  GE_ASSERT_TRUE(actual_input_index == actual_input_num);
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus CreateExpandAicpuOptionalInputAddrsOutputs(const ge::FastNode *node, KernelContext *context) {
  (void)node;
  for (size_t i = 0U; i < context->GetOutputNum(); ++i) {
    auto chain = context->GetOutput(i);
    GE_ASSERT_NOTNULL(chain);
    auto td = new (std::nothrow) GertTensorData();
    GE_ASSERT_NOTNULL(td);
    chain->SetWithDefaultDeleter(td);
  }
  return ge::GRAPH_SUCCESS;
}

REGISTER_KERNEL(ExpandAicpuOptionalInputAddrs)
    .RunFunc(ExpandAicpuOptionalInputAddrs)
    .OutputsCreator(CreateExpandAicpuOptionalInputAddrsOutputs)
    .ConcurrentCriticalSectionKey(kKernelUseMemory);

ge::graphStatus DistributeAsyncWaitTask(rtStream stream, const std::string &op_name,
                                        const AicpuArgsHandler *args_handler) {
  const auto is_block_op = args_handler->IsBlockOp();
  if (is_block_op) {
    GE_ASSERT_RT_OK(DistributeWaitTaskForAicpuBlockingOp(stream, args_handler, op_name.c_str()));
  }
  return SUCCESS;
}

ge::graphStatus AicpuLaunchTfKernel(KernelContext *context) {
  auto args_handler = context->GetInputPointer<AicpuArgsHandler>(static_cast<size_t>(AicpuLaunchCommon::kArgsHandler));
  auto stream = context->GetInputValue<rtStream_t>(static_cast<size_t>(AicpuLaunchCommon::kStream));
  auto bin_handle = context->GetInputPointer<aclrtBinHandle>(static_cast<size_t>(AicpuTfLaunch::kBinHandler));
  GE_ASSERT_NOTNULL(args_handler);
  GE_ASSERT_NOTNULL(bin_handle);
  auto node_type = context->GetInputValue<char_t *>(static_cast<size_t>(AicpuTfLaunch::kNodeType));
  GE_ASSERT_NOTNULL(node_type);
  const auto &op_name = args_handler->GetNodeName();
  if (!OpJsonBinHandler::IsSupportBinHandle() || *bin_handle == nullptr) {
    GELOGI("launch tf kernel %s with compatible.", op_name.c_str());
    const auto &arg_ex = args_handler->GetArgsEx();
    GE_ASSERT_RT_OK(rtAicpuKernelLaunchExWithArgs(rtKernelType_t::KERNEL_TYPE_FWK, op_name.c_str(), 1U, &arg_ex,
                                                  nullptr, stream, RT_KERNEL_DEFAULT));
  } else {
    std::vector<aclrtPlaceHolderInfo> placeHolder_info;
    for (const auto &kernel_info : args_handler->GetKernelOffset()) {
      placeHolder_info.emplace_back(aclrtPlaceHolderInfo({kernel_info.addrOffset, kernel_info.dataOffset}));
    }

    for (const auto &host_Info : args_handler->GetHostInputOffset()) {
      placeHolder_info.emplace_back(aclrtPlaceHolderInfo({host_Info.addrOffset, host_Info.dataOffset}));
    }
    GELOGI("launch tf kernel %s with new interface, place_size=%lu, args_size=%lu, node_type=%s", op_name.c_str(),
           placeHolder_info.size(), args_handler->GetArgsEx().argsSize, node_type);
    aclrtLaunchKernelAttr launch_attr = {};
    aclrtLaunchKernelCfg cfg = {&launch_attr, 0UL};
    aclrtFuncHandle func_handle = nullptr;
    GE_ASSERT_SUCCESS(aclrtBinaryGetFunction(*bin_handle, node_type, &func_handle));
    GE_ASSERT_NOTNULL(func_handle);
    GE_ASSERT_RT_OK(aclrtLaunchKernelWithHostArgs(func_handle, 1U, stream, &cfg, args_handler->GetArgsEx().args,
                                                  args_handler->GetArgsEx().argsSize, placeHolder_info.data(),
                                                  placeHolder_info.size()));
  }
  return DistributeAsyncWaitTask(stream, op_name, args_handler);
}
REGISTER_KERNEL(AicpuLaunchTfKernel)
    .RunFunc(AicpuLaunchTfKernel)
    .TracePrinter(PrintTfLaunchArgs)
    .ConcurrentCriticalSectionKey(kKernelLaunch);

ge::graphStatus AicpuLaunchCCKernelWithNewInterface(const KernelContext *context) {
  auto args_handler = context->GetInputPointer<AicpuArgsHandler>(static_cast<size_t>(AicpuLaunchCommon::kArgsHandler));
  auto stream = context->GetInputValue<rtStream_t>(static_cast<size_t>(AicpuLaunchCommon::kStream));
  auto block_dim = context->GetInputValue<uint32_t>(static_cast<size_t>(AicpuCCLaunch::kBlockDim));
  auto bin_handle = context->GetInputPointer<aclrtBinHandle>(static_cast<size_t>(AicpuCCLaunch::kBinHandler));
  std::vector<aclrtPlaceHolderInfo> placeHolder_info;
  for (auto &kernel_info : args_handler->GetKernelOffset()) {
    placeHolder_info.emplace_back(aclrtPlaceHolderInfo({kernel_info.addrOffset, kernel_info.dataOffset}));
  }

  for (auto &host_Info : args_handler->GetHostInputOffset()) {
    placeHolder_info.emplace_back(aclrtPlaceHolderInfo({host_Info.addrOffset, host_Info.dataOffset}));
  }

  const auto &op_name = args_handler->GetNodeName();
  auto node_type = context->GetInputValue<char_t *>(static_cast<size_t>(AicpuCCLaunch::kNodeType));
  GE_ASSERT_NOTNULL(node_type);
  GELOGI("launch cc kernel %s with new interface, block_dim=%u, place_size=%lu, args_size=%lu, node_type=%s",
         op_name.c_str(), block_dim, placeHolder_info.size(), args_handler->GetArgsEx().argsSize, node_type);
  aclrtLaunchKernelAttr launch_attr = {};
  aclrtLaunchKernelCfg cfg = {&launch_attr, 0UL};
  aclrtFuncHandle func_handle = nullptr;
  GE_ASSERT_SUCCESS(aclrtBinaryGetFunction(*bin_handle, node_type, &func_handle));
  GE_ASSERT_NOTNULL(func_handle);
  GE_ASSERT_RT_OK(aclrtLaunchKernelWithHostArgs(func_handle, block_dim, stream, &cfg, args_handler->GetArgsEx().args,
                                                args_handler->GetArgsEx().argsSize, placeHolder_info.data(),
                                                placeHolder_info.size()));
  return SUCCESS;
}

ge::graphStatus AicpuLaunchCCKernel(KernelContext *context) {
  auto args_handler = context->GetInputPointer<AicpuArgsHandler>(static_cast<size_t>(AicpuLaunchCommon::kArgsHandler));
  auto stream = context->GetInputValue<rtStream_t>(static_cast<size_t>(AicpuLaunchCommon::kStream));
  auto block_dim = context->GetInputValue<uint32_t>(static_cast<size_t>(AicpuCCLaunch::kBlockDim));
  auto kernel_type = context->GetInputValue<ccKernelType>(static_cast<size_t>(AicpuCCLaunch::kKernelType));
  auto bin_handle = context->GetInputPointer<rtBinHandle>(static_cast<size_t>(AicpuCCLaunch::kBinHandler));
  GE_ASSERT_NOTNULL(args_handler);
  GE_ASSERT_NOTNULL(bin_handle);

  uint32_t flag = RT_KERNEL_DEFAULT;
  uint32_t rt_kernel_type = static_cast<uint32_t>(rtKernelType_t::KERNEL_TYPE_AICPU);
  const auto &op_name = args_handler->GetNodeName();
  if ((kernel_type == ccKernelType::CUST_AI_CPU) || (!OpJsonBinHandler::IsSupportBinHandle()) ||
      (*bin_handle == nullptr)) {
    if (kernel_type == ccKernelType::CUST_AI_CPU) {
      flag |= static_cast<uint32_t>(RT_KERNEL_CUSTOM_AICPU);
      rt_kernel_type = static_cast<uint32_t>(rtKernelType_t::KERNEL_TYPE_AICPU_CUSTOM);
    }
    GELOGI("launch cc kernel %s with compatible, kernel_type is %u.", op_name.c_str(), kernel_type);
    const auto &arg_ex = args_handler->GetArgsEx();
    GE_ASSERT_RT_OK(
        rtAicpuKernelLaunchExWithArgs(rt_kernel_type, op_name.c_str(), block_dim, &arg_ex, nullptr, stream, flag));
  } else {
    GE_ASSERT_SUCCESS(AicpuLaunchCCKernelWithNewInterface(context));
  }

  return DistributeAsyncWaitTask(stream, op_name, args_handler);
}
REGISTER_KERNEL(AicpuLaunchCCKernel)
    .RunFunc(AicpuLaunchCCKernel)
    .TracePrinter(PrintCCLaunchArgs)
    .ConcurrentCriticalSectionKey(kKernelLaunch);

struct SharedPtrNoDeleter {
  void operator()(ge::OpKernelBin *) {}
};

ge::graphStatus LaunchAicpuCustKernel(KernelContext *context) {
  GE_ASSERT_NOTNULL(context);
  const auto aicpu_kernel_holder = context->GetInputValue<ge::OpKernelBin *>(0U);
  GE_ASSERT_NOTNULL(aicpu_kernel_holder);
  const CustAICPUKernelPtr aicpu_kernel(aicpu_kernel_holder, SharedPtrNoDeleter());
  GE_ASSERT_NOTNULL(aicpu_kernel);
  auto so_name = context->GetInputValue<char *>(static_cast<size_t>(1));
  bool loaded = false;
  ge::ModelManager::GetInstance().LoadCustAicpuSo(aicpu_kernel, so_name, loaded);
  GELOGI("launch aicpu cust kernel loaded=%d.", loaded);
  ge::ModelManager::GetInstance().LaunchCustAicpuSo();
  return SUCCESS;
}
REGISTER_KERNEL(LaunchAicpuCustKernel).RunFunc(LaunchAicpuCustKernel);

ge::graphStatus ClearAicpuCustKernel(KernelContext *context) {
  GE_ASSERT_NOTNULL(context);
  ge::ModelManager::GetInstance().ClearAicpuSo();
  return SUCCESS;
}
REGISTER_KERNEL(ClearAicpuCustKernel).RunFunc(ClearAicpuCustKernel);

ge::graphStatus AicpuHostCompute(KernelContext *context) {
  auto args_handler = context->GetInputPointer<AicpuArgsHandler>(0U);
  GE_ASSERT_NOTNULL(args_handler);

  auto io_num = args_handler->GetIoNum();
  uint64_t *io_addrs = PtrToPtr<uint8_t, uint64_t>(args_handler->GetIoAddr());
  GE_ASSERT_NOTNULL(io_addrs);
  for (size_t i = 1U; i < io_num + 1U; i++) {  // skip AicpuArgsHandler
    auto tensor_data = context->GetInputValue<gert::GertTensorData *>(i);
    GE_ASSERT_NOTNULL(tensor_data);
    *io_addrs = PtrToValue(tensor_data->GetAddr());
    io_addrs++;
  }

  auto args = args_handler->GetArgs();
  auto run_cpu_kernel = AicpuResourceManager::GetInstance().GetRunCpuKernel();
  GE_ASSERT_NOTNULL(args);
  GE_ASSERT_NOTNULL(run_cpu_kernel);
  GE_ASSERT_SUCCESS(run_cpu_kernel(args));
  return SUCCESS;
}
REGISTER_KERNEL(AicpuHostCompute).RunFunc(AicpuHostCompute);

namespace {
struct FusedHostCpuTensorState {
  const void *data = nullptr;
  size_t data_size = 0U;
  std::vector<int64_t> dims;
  bool initialized = false;
};

struct FusedHostCpuCallState {
  std::string register_name;
  void *kernel_state = nullptr;
  FusedHostCpuDestroyFunc destroy_func = nullptr;
  FusedHostCpuRunFunc run_func = nullptr;
  std::vector<FusedHostCpuTensorBinding> bindings;
  std::vector<FusedHostCpuTensorState> tensor_states;
};

bool HasSameShape(const gert::Shape &shape, const FusedHostCpuTensorState &state) {
  if (state.dims.size() != shape.GetDimNum()) {
    return false;
  }
  for (size_t i = 0U; i < shape.GetDimNum(); ++i) {
    if (state.dims[i] != shape.GetDim(i)) {
      return false;
    }
  }
  return true;
}

ge::graphStatus BuildFusedHostCpuBinding(const StorageShape *storage_shape, GertTensorData *tensor_data,
                                         FusedHostCpuTensorState &state, FusedHostCpuTensorBinding &binding) {
  GE_ASSERT_NOTNULL(storage_shape);
  GE_ASSERT_NOTNULL(tensor_data);
  GE_ASSERT_TRUE((tensor_data->GetSize() == 0U) || (tensor_data->GetAddr() != nullptr));
  const auto &origin_shape = storage_shape->GetOriginShape();
  const bool shape_changed = !state.initialized || !HasSameShape(origin_shape, state);
  if (shape_changed) {
    if (state.dims.size() != origin_shape.GetDimNum()) {
      state.dims.resize(origin_shape.GetDimNum());
    }
    for (size_t i = 0U; i < state.dims.size(); ++i) {
      state.dims[i] = origin_shape.GetDim(i);
    }
    binding.dims = state.dims.data();
    binding.dim_num = state.dims.size();
  }
  const void *data = tensor_data->GetAddr();
  const size_t data_size = tensor_data->GetSize();
  const bool data_changed = !state.initialized || (state.data != data) || (state.data_size != data_size);
  if (data_changed) {
    binding.data = reinterpret_cast<uint8_t *>(tensor_data->GetAddr());
    binding.data_size = data_size;
    state.data = data;
    state.data_size = data_size;
  }
  binding.flags = (shape_changed ? kFusedHostCpuShapeChanged : 0U) | (data_changed ? kFusedHostCpuDataChanged : 0U);
  state.initialized = true;
  return ge::GRAPH_SUCCESS;
}
}  // namespace

void *CreateFusedHostCpuComputeState(const char *register_name, void *kernel_state,
                                     const FusedHostCpuDestroyFunc destroy_func, const FusedHostCpuRunFunc run_func,
                                     const FusedHostCpuTensorMeta *tensor_metas, const size_t io_num) {
  if ((register_name == nullptr) || (kernel_state == nullptr) || (destroy_func == nullptr) || (run_func == nullptr) ||
      (tensor_metas == nullptr) || (io_num == 0U)) {
    GELOGE(ge::PARAM_INVALID,
           "Invalid fused HostCPU compute state arguments: register_null[%d], kernel_state_null[%d], "
           "destroy_null[%d], run_null[%d], tensor_metas_null[%d], io_num[%zu].",
           static_cast<int32_t>(register_name == nullptr), static_cast<int32_t>(kernel_state == nullptr),
           static_cast<int32_t>(destroy_func == nullptr), static_cast<int32_t>(run_func == nullptr),
           static_cast<int32_t>(tensor_metas == nullptr), io_num);
    return nullptr;
  }
  std::unique_ptr<FusedHostCpuCallState> state = std::make_unique<FusedHostCpuCallState>();
  state->register_name = register_name;
  state->kernel_state = kernel_state;
  state->destroy_func = destroy_func;
  state->run_func = run_func;
  state->bindings.reserve(io_num);
  state->tensor_states.reserve(io_num);
  for (size_t i = 0U; i < io_num; ++i) {
    std::vector<int64_t> dims(tensor_metas[i].dim_num, ge::UNKNOWN_DIM);
    FusedHostCpuTensorState tensor_state;
    tensor_state.dims = std::move(dims);
    state->tensor_states.emplace_back(std::move(tensor_state));
    state->bindings.emplace_back(
        FusedHostCpuTensorBinding{state->tensor_states.back().dims.data(), nullptr, tensor_metas[i].dim_num, 0U, 0U});
  }
  return state.release();
}

void DestroyFusedHostCpuComputeState(void *compute_state) {
  FusedHostCpuCallState *state = static_cast<FusedHostCpuCallState *>(compute_state);
  if (state == nullptr) {
    return;
  }
  state->destroy_func(state->kernel_state);
  delete state;
}

// Runtime callback for the ExecuteGraph kernel registered as FusedHostCpuCompute.
ge::graphStatus RunFusedHostCpuCompute(KernelContext *context) {
  GE_ASSERT_NOTNULL(context);
  const auto compute_meta = context->GetInputPointer<FusedHostCpuComputeMeta>(0U);
  GE_ASSERT_NOTNULL(compute_meta);
  FusedHostCpuCallState *call_state = static_cast<FusedHostCpuCallState *>(compute_meta->compute_state);
  GE_ASSERT_NOTNULL(call_state);
  GE_ASSERT_NOTNULL(call_state->kernel_state);
  GE_ASSERT_NOTNULL(call_state->run_func);
  const auto input_num = compute_meta->input_num;
  const auto output_num = compute_meta->output_num;
  const auto io_num = input_num + output_num;
  GE_ASSERT_TRUE(call_state->bindings.size() == io_num);
  GE_ASSERT_TRUE(call_state->tensor_states.size() == io_num);

  const size_t input_shape_start = 1U;
  const size_t input_addr_start = input_shape_start + input_num;
  const size_t output_shape_start = input_addr_start + input_num;
  const size_t output_addr_start = output_shape_start + output_num;
  GE_ASSERT_TRUE(context->GetInputNum() == (output_addr_start + output_num));

  uint32_t binding_flags = 0U;
  for (size_t i = 0U; i < input_num; ++i) {
    const auto storage_shape = context->GetInputPointer<StorageShape>(input_shape_start + i);
    auto tensor_data = context->MutableInputPointer<GertTensorData>(input_addr_start + i);
    GE_ASSERT_SUCCESS(
        BuildFusedHostCpuBinding(storage_shape, tensor_data, call_state->tensor_states[i], call_state->bindings[i]));
    binding_flags |= call_state->bindings[i].flags;
  }

  for (size_t i = 0U; i < output_num; ++i) {
    const auto storage_shape = context->GetInputPointer<StorageShape>(output_shape_start + i);
    auto tensor_data = context->MutableInputPointer<GertTensorData>(output_addr_start + i);
    const size_t tensor_index = input_num + i;
    GE_ASSERT_SUCCESS(BuildFusedHostCpuBinding(storage_shape, tensor_data, call_state->tensor_states[tensor_index],
                                               call_state->bindings[tensor_index]));
    binding_flags |= call_state->bindings[tensor_index].flags;
  }

  const uint32_t ret = call_state->run_func(call_state->kernel_state,
                                            static_cast<const void *>(call_state->bindings.data()), binding_flags);
  GE_ASSERT_TRUE(ret == 0U, "Fused HostCPU private entry failed: register_name[%s], ret=%u.",
                 call_state->register_name.c_str(), ret);
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus CreateFusedHostCpuComputeOutputs(const ge::FastNode *node, KernelContext *context) {
  (void)node;
  GE_ASSERT_NOTNULL(context);
  GE_ASSERT_TRUE(context->GetInputNum() >= context->GetOutputNum());
  const size_t output_addr_start = context->GetInputNum() - context->GetOutputNum();
  for (size_t i = 0U; i < context->GetOutputNum(); ++i) {
    auto chain = context->GetOutput(i);
    auto tensor_data = context->MutableInputPointer<GertTensorData>(output_addr_start + i);
    GE_ASSERT_NOTNULL(chain);
    GE_ASSERT_NOTNULL(tensor_data);
    chain->Set(tensor_data, nullptr);
  }
  return ge::GRAPH_SUCCESS;
}

REGISTER_KERNEL(FusedHostCpuCompute)
    .RunFunc(RunFusedHostCpuCompute)
    .OutputsCreator(CreateFusedHostCpuComputeOutputs)
    .ConcurrentCriticalSectionKey(kKernelUseMemory);

ge::graphStatus ReleaseFusedHostCpuKernelState(KernelContext *context) {
  GE_ASSERT_NOTNULL(context);
  const auto destroy_meta = context->GetInputPointer<FusedHostCpuDestroyMeta>(0U);
  GE_ASSERT_NOTNULL(destroy_meta);
  GE_ASSERT_NOTNULL(destroy_meta->compute_state);
  DestroyFusedHostCpuComputeState(destroy_meta->compute_state);
  return ge::GRAPH_SUCCESS;
}
REGISTER_KERNEL(ReleaseFusedHostCpuKernelState).RunFunc(ReleaseFusedHostCpuKernelState);

ge::graphStatus AicpuHostExecFunc(KernelContext *context) {
  const auto input_size = context->GetInputNum();
  // func取输入的最后一个，因为前面输入个数不固定。
  const auto aicpu_host_execute_addr = context->GetInputPointer<void *>(input_size - 1);
  GE_ASSERT_NOTNULL(aicpu_host_execute_addr);
  AicpuHostProcFunc aicpu_host_execute_func = *(AicpuHostProcFunc *)aicpu_host_execute_addr;
  GE_ASSERT_NOTNULL(aicpu_host_execute_func);
  GE_ASSERT_SUCCESS(aicpu_host_execute_func(context));
  return ge::GRAPH_SUCCESS;
}

REGISTER_KERNEL(AicpuHostExecFunc).RunFunc(AicpuHostExecFunc);

ge::graphStatus CalcBlockDim(KernelContext *context) {
  auto block_dim_index = context->GetInputValue<int32_t>(0U);
  auto input_shape = context->GetInputPointer<Shape>(1U);
  GE_ASSERT_NOTNULL(input_shape);

  int64_t total = 1;
  if (block_dim_index == kBlockdimAxisDefaultIndex) {
    total = input_shape->GetShapeSize();
  } else {
    total = input_shape->GetDim(static_cast<size_t>(block_dim_index));
  }
  int64_t ai_cpu_cnt = 1;
  int32_t device_id = 0;
  aclError ret = aclrtGetDevice(&device_id);
  if (ret == ACL_SUCCESS) {
    (void)aclrtGetDeviceInfo(static_cast<uint32_t>(device_id), ACL_DEV_ATTR_AICPU_CORE_NUM, &ai_cpu_cnt);
  }
  const int64_t max_shard_num = ai_cpu_cnt * 2;  // magic num
  const int64_t per_unit_size = total / std::min(std::max(int64_t{1}, ai_cpu_cnt), total);
  int64_t block_size = std::max(int64_t{1}, std::min(total, per_unit_size));
  int64_t shard_num = CeilDivisor(total, block_size);
  shard_num = std::min(max_shard_num, shard_num);
  GE_ASSERT_TRUE(shard_num != 0);
  block_size = CeilDivisor(total, shard_num);

  auto block_num = context->GetOutputPointer<uint32_t>(static_cast<int32_t>(0));
  GE_ASSERT_NOTNULL(block_num);
  *block_num = static_cast<uint32_t>(CeilDivisor(total, block_size));
  return ge::GRAPH_SUCCESS;
}
REGISTER_KERNEL(CalcBlockDim).RunFunc(CalcBlockDim);

ge::graphStatus TensorListOp(KernelContext *context) {
  auto compute_node_info = reinterpret_cast<const ComputeNodeInfo *>(context->GetComputeNodeExtend());
  GE_CHECK_NOTNULL(compute_node_info);
  const std::string op_type = compute_node_info->GetNodeType();
  const std::string op_name = compute_node_info->GetNodeName();
  auto stream = context->GetInputValue<rtStream_t>(kTensorListStreamIdx);

  if (op_type == "TensorListPushBack") {
    auto tensor_data = context->GetInputValue<gert::GertTensorData *>(kTensorListPushBackDataIdx);
    auto handle_data = context->GetInputValue<gert::GertTensorData *>(kTensorListPushBackOutputHandleIdx);
    GE_ASSERT_NOTNULL(tensor_data);
    GE_ASSERT_NOTNULL(handle_data);
    GE_ASSERT_SUCCESS(AicpuResourceManager::GetInstance().PushTensor(op_name, stream, tensor_data, handle_data));
  }

  if (op_type == "TensorListPopBack") {
    auto handle_data = context->GetInputValue<gert::GertTensorData *>(kTensorListPopBackInputHandleIdx);
    GE_ASSERT_NOTNULL(handle_data);
    GE_ASSERT_SUCCESS(AicpuResourceManager::GetInstance().PopTensor(op_name, stream, handle_data));
  }
  return ge::GRAPH_SUCCESS;
}
REGISTER_KERNEL(TensorListOp).RunFunc(TensorListOp);

ge::graphStatus CreateEvent(KernelContext *context) {
  auto rt_event = context->GetOutputPointer<aclrtEvent>(0U);
  auto event_id = context->GetOutputPointer<uint32_t>(1U);
  GE_ASSERT_NOTNULL(rt_event);
  GE_ASSERT_NOTNULL(event_id);

  GE_ASSERT_RT_OK(aclrtCreateEventWithFlag(rt_event, ACL_EVENT_SYNC));
  GE_ASSERT_RT_OK(aclrtGetEventId(*rt_event, event_id));
  return ge::GRAPH_SUCCESS;
}
REGISTER_KERNEL(CreateEvent).RunFunc(CreateEvent);

ge::graphStatus DestroyEvent(KernelContext *context) {
  auto rt_event = context->GetInputValue<aclrtEvent>(0U);
  GE_ASSERT_NOTNULL(rt_event);

  GE_ASSERT_RT_OK(aclrtDestroyEvent(rt_event));
  return ge::GRAPH_SUCCESS;
}
REGISTER_KERNEL(DestroyEvent).RunFunc(DestroyEvent);

ge::graphStatus AllocHostCpuOutputMemory(KernelContext *context) {
  const auto size = context->GetInputValue<size_t>(static_cast<size_t>(AllocHostCpuOutputMemoryInputs::kSize));
  auto gert_allocator =
      context->GetInputValue<GertAllocator *>(static_cast<size_t>(AllocHostCpuOutputMemoryInputs::kAllocator));
  const auto out_tensor_data = context->GetOutputPointer<gert::GertTensorData>(0);
  GE_ASSERT_NOTNULL(out_tensor_data);

  if ((out_tensor_data->GetAddr() == nullptr) || (out_tensor_data->GetSize() < size)) {
    auto host_block = gert_allocator->Malloc(size);
    KERNEL_CHECK_NOTNULL(host_block);
    KERNEL_CHECK(host_block->GetAddr() != nullptr, "malloc failed, tensor size=%zu", size);
    *out_tensor_data = {size, gert_allocator->GetPlacement(), gert_allocator->GetStreamId(), host_block};
  }
  return ge::GRAPH_SUCCESS;
}
ge::graphStatus CreateOutputsForAllocHostCpuOutput(const ge::FastNode *node, KernelContext *context) {
  (void)node;
  for (size_t i = 0U; i < context->GetOutputNum(); ++i) {
    auto chain = context->GetOutput(i);
    auto tensor_data = ge::MakeUnique<GertTensorData>(0, kOnHost, -1, nullptr);
    GE_ASSERT_NOTNULL(chain);
    GE_ASSERT_NOTNULL(tensor_data);
    chain->SetWithDefaultDeleter(tensor_data.release());
  }
  return ge::GRAPH_SUCCESS;
}
REGISTER_KERNEL(AllocHostCpuOutputMemory)
    .RunFunc(AllocHostCpuOutputMemory)
    .OutputsCreator(CreateOutputsForAllocHostCpuOutput)
    .ConcurrentCriticalSectionKey(kKernelUseMemory);
}  // namespace kernel
}  // namespace gert
