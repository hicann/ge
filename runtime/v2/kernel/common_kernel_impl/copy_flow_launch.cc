/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "copy_flow_launch.h"
#include "memory_copy.h"
#include "common/checker.h"
#include "graph/error_codes.h"
#include "register/kernel_registry.h"
#include "exe_graph/runtime/tensor_data.h"
#include "exe_graph/runtime/storage_shape.h"
#include "common/table_driven.h"
#include "core/debug/kernel_tracing.h"
#include "kernel/kernel_log.h"
#include "kernel/memory/mem_block.h"
#include "kernel/memory/caching_mem_allocator.h"
#include "exe_graph/runtime/gert_mem_allocator.h"
#include "aicore/launch_kernel/rt_kernel_launch_args_ex.h"
#include "graph/utils/type_utils.h"
#include "core/executor/multi_thread_topological/executor/schedule/producer/producers/kernel_tags/critical_section_config.h"
#include "core/utils/rt2_tensor_utils.h"
#include "exe_graph/runtime/gert_tensor_data.h"
#include "graph/utils/attr_utils.h"
#include "graph_metadef/common/ge_common/util.h"
#include "kernel/memory/memory_kernel.h"

using namespace ge;
namespace gert {
namespace kernel {
namespace {
constexpr size_t kAlignBytes4 = 4U;

ge::graphStatus GetCopyFlowCount(const FastNode *node, size_t &copy_flow_count) {
  GE_ASSERT_NOTNULL(node);
  const auto op_desc = node->GetOpDescBarePtr();
  GE_ASSERT_NOTNULL(op_desc);
  int64_t count = 0;
  GE_ASSERT_TRUE(ge::AttrUtils::GetInt(op_desc, kCopyFlowCountAttr, count));
  GE_ASSERT_TRUE(count > 0);
  copy_flow_count = static_cast<size_t>(count);
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus ValidateCopyFlowInputNum(const KernelContext *context, const size_t input_start,
                                         const size_t copy_flow_count) {
  size_t input_num = 0U;
  GE_ASSERT_TRUE(!ge::MulOverflow(copy_flow_count, kSizeOfCopyToDevice, input_num));
  GE_ASSERT_TRUE(!ge::AddOverflow(input_start, input_num, input_num));
  if (input_num != context->GetInputNum()) {
    GELOGE(ge::GRAPH_FAILED, "input num is not matched, input start %zu, copy flow count %zu, total input num %zu",
           input_start, copy_flow_count, context->GetInputNum());
    return ge::GRAPH_FAILED;
  }
  return ge::GRAPH_SUCCESS;
}
}  // namespace

ge::graphStatus CreateCopyFlowAllocSizes(const FastNode *node, KernelContext *context) {
  size_t copy_flow_count = 0U;
  GE_ASSERT_SUCCESS(GetCopyFlowCount(node, copy_flow_count));
  auto chain = context->GetOutput(0U);
  GE_ASSERT_NOTNULL(chain);
  auto sizes = ContinuousVector::Create<size_t>(copy_flow_count);
  GE_ASSERT_NOTNULL(sizes);
  chain->SetWithDefaultDeleter<uint8_t[]>(sizes.release());
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus CreateCopyFlowLaunchTensorData(const FastNode *node, KernelContext *context) {
  (void)node;
  auto out_num = context->GetOutputNum();
  for (size_t i = static_cast<size_t>(CopyFlowLaunchOutputs::kAddress); i < out_num; ++i) {
    auto chain = context->GetOutput(i);
    if (chain == nullptr) {
      return ge::GRAPH_FAILED;
    }
    auto tensor_data = new (std::nothrow) GertTensorData(0, kOnDeviceHbm, -1, nullptr);
    if (tensor_data == nullptr) {
      return ge::GRAPH_FAILED;
    }
    chain->SetWithDefaultDeleter(tensor_data);
  }

  return ge::GRAPH_SUCCESS;
}

ge::graphStatus CreatePrepareCopyFlowResultOutputs(const FastNode *node, KernelContext *context) {
  size_t copy_flow_count = 0U;
  GE_ASSERT_SUCCESS(GetCopyFlowCount(node, copy_flow_count));
  GE_ASSERT_TRUE(context->GetOutputNum() == copy_flow_count, "copy flow output num %zu does not match count %zu",
                 context->GetOutputNum(), copy_flow_count);
  return CreateCopyFlowLaunchTensorData(node, context);
}

ge::graphStatus CopyTensorToDevice(KernelContext *context, const size_t copy_index) {
  auto stream = context->GetInputValue<aclrtStream>(static_cast<size_t>(CopyFlowLaunchInputs::kStream));
  auto gert_allocator =
      context->MutableInputPointer<GertAllocator>(static_cast<size_t>(CopyFlowLaunchInputs::kAllocator));
  GE_ASSERT_NOTNULL(gert_allocator);
  auto addr_index = copy_index * kSizeOfCopyToDevice + static_cast<size_t>(CopyFlowLaunchInputs::kAddrAndLengthStart);
  auto tensor_data = context->GetInputValue<gert::GertTensorData *>(addr_index);
  auto tensor_size = context->GetInputValue<size_t>(addr_index + 1U);
  auto src_storage_shape = context->GetInputPointer<StorageShape>(addr_index + 2U);
  auto data_type = context->GetInputValue<ge::DataType>(addr_index + 3U);
  auto out_tensor_data = context->GetOutputPointer<gert::GertTensorData>(
      copy_index + static_cast<size_t>(CopyFlowLaunchOutputs::kAddress));
  GE_ASSERT_NOTNULL(tensor_data);
  GE_ASSERT_NOTNULL(out_tensor_data);
  GE_ASSERT_NOTNULL(src_storage_shape);
  auto mem_block = reinterpret_cast<memory::MultiStreamMemBlock *>(gert_allocator->Malloc(tensor_size));
  KERNEL_CHECK_NOTNULL(mem_block);
  KERNEL_CHECK(mem_block->GetAddr() != nullptr, "malloc failed, tensor size=%zu", tensor_size);
  KERNEL_TRACE_ALLOC_MEM(gert_allocator->GetStreamId(), mem_block, mem_block->GetAddr(), mem_block->GetSize());
  *out_tensor_data =
      TensorUtils::ToGertTensorData(mem_block, gert_allocator->GetPlacement(), gert_allocator->GetStreamId());

  auto host_tensor_size = ge::GetSizeInBytes(src_storage_shape->GetStorageShape().GetShapeSize(), data_type);
  GELOGD("StreamCopyH2D, host addr %p, host tensor size %zu, device addr %p, alloc device size %zu",
         tensor_data->GetAddr(), host_tensor_size, mem_block->GetAddr(), tensor_size);
  if (host_tensor_size > 0U) {
    GE_ASSERT_RT_OK(aclrtMemcpyAsync(mem_block->GetAddr(), tensor_size, tensor_data->GetAddr(), host_tensor_size,
                                     ACL_MEMCPY_HOST_TO_BUF_TO_DEVICE, stream));
  }
  out_tensor_data->SetPlacement(kOnDeviceHbm);

  return ge::GRAPH_SUCCESS;
}

ge::graphStatus CalcCopyFlowAllocSizes(KernelContext *context) {
  auto output_num = context->GetInputValue<size_t>(static_cast<size_t>(CalcCopyFlowAllocSizesInputs::kInputsNum));
  auto args = context->MutableInputPointer<gert::RtKernelLaunchArgsEx>(
      static_cast<size_t>(CalcCopyFlowAllocSizesInputs::kRtArg));
  auto alloc_sizes = context->GetOutputPointer<TypedContinuousVector<size_t>>(0U);
  GE_ASSERT_NOTNULL(args);
  GE_ASSERT_NOTNULL(alloc_sizes);
  GE_ASSERT_TRUE(output_num == alloc_sizes->GetCapacity(), "copy flow output num %zu does not match capacity %zu",
                 output_num, alloc_sizes->GetCapacity());

  alloc_sizes->SetSize(output_num);
  auto sizes_data = alloc_sizes->MutableData();
  GE_ASSERT_NOTNULL(sizes_data);
  size_t host_input_data_size = args->GetMergedCopySize();
  const auto max_host_input_data_len = kMaxHostInputDataLen + args->GetMergedCopySize();
  if (ValidateCopyFlowInputNum(context, static_cast<size_t>(CalcCopyFlowAllocSizesInputs::kAddrAndLengthStart),
                               output_num) != ge::GRAPH_SUCCESS) {
    return ge::GRAPH_FAILED;
  }
  for (size_t i = 0U; i < output_num; ++i) {
    sizes_data[i] = 0U;
    auto addr_index = i * kSizeOfCopyToDevice + static_cast<size_t>(CalcCopyFlowAllocSizesInputs::kAddrAndLengthStart);
    auto tensor_data = context->GetInputValue<gert::GertTensorData *>(addr_index);
    auto tensor_size = context->GetInputValue<size_t>(addr_index + 1U);
    auto src_storage_shape = context->GetInputPointer<StorageShape>(addr_index + 2U);
    auto data_type = context->GetInputValue<ge::DataType>(addr_index + 3U);
    GE_ASSERT_NOTNULL(tensor_data);
    GE_ASSERT_NOTNULL(src_storage_shape);
    if (!TensorPlacementUtils::IsOnHost(tensor_data->GetPlacement())) {
      continue;
    }

    const auto host_tensor_size = ge::GetSizeInBytes(src_storage_shape->GetStorageShape().GetShapeSize(), data_type);
    if (host_tensor_size < 0) {
      GELOGE(ge::GRAPH_FAILED, "shape_size[%" PRId64 "], data_type[%s]",
             src_storage_shape->GetStorageShape().GetShapeSize(),
             ge::TypeUtils::DataTypeToSerialString(data_type).c_str());
      return ge::GRAPH_FAILED;
    }
    const size_t align_size = ge::RoundUp(static_cast<uint64_t>(host_tensor_size), kAlignBytes4);
    const auto new_host_input_data_size = host_input_data_size + align_size;
    if (new_host_input_data_size > max_host_input_data_len) {
      sizes_data[i] = tensor_size;
    } else {
      host_input_data_size = new_host_input_data_size;
    }
  }
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus PrepareCopyFlowResult(KernelContext *context) {
  auto output_num = context->GetOutputNum();
  if (ValidateCopyFlowInputNum(context, static_cast<size_t>(PrepareCopyFlowResultInputs::kAddrAndLengthStart),
                               output_num) != ge::GRAPH_SUCCESS) {
    return ge::GRAPH_FAILED;
  }

  auto input_num = context->GetInputPointer<size_t>(static_cast<size_t>(PrepareCopyFlowResultInputs::kInputsNum));
  GE_CHECK_NOTNULL(input_num);
  if (*input_num != output_num) {
    GELOGE(ge::GRAPH_FAILED, "host input num %zu, is not match output num %zu,", *input_num, output_num);
    return ge::GRAPH_FAILED;
  }

  auto args = context->MutableInputPointer<gert::RtKernelLaunchArgsEx>(
      static_cast<size_t>(PrepareCopyFlowResultInputs::kRtArg));
  auto allocated_addrs = context->GetInputPointer<TypedContinuousVector<GertTensorData *>>(
      static_cast<size_t>(PrepareCopyFlowResultInputs::kAllocatedAddrs));
  auto inputs_index_cvv =
      context->GetInputValue<ContinuousVectorVector *>(static_cast<size_t>(PrepareCopyFlowResultInputs::kInputsIndex));
  GE_ASSERT_NOTNULL(args);
  GE_ASSERT_NOTNULL(allocated_addrs);
  GE_ASSERT_NOTNULL(inputs_index_cvv);
  GE_ASSERT_TRUE(allocated_addrs->GetSize() == output_num, "allocated addr num %zu is not match output num %zu",
                 allocated_addrs->GetSize(), output_num);
  GE_ASSERT_TRUE(inputs_index_cvv->GetSize() == output_num, "input index num %zu is not match output num %zu",
                 inputs_index_cvv->GetSize(), output_num);
  GE_ASSERT_SUCCESS(args->UpdateMergedCopyInfo());

  auto allocated_data = allocated_addrs->GetData();
  GE_ASSERT_NOTNULL(allocated_data);
  for (size_t i = 0U; i < output_num; ++i) {
    auto addr_index = i * kSizeOfCopyToDevice + static_cast<size_t>(PrepareCopyFlowResultInputs::kAddrAndLengthStart);
    auto tensor_data = context->GetInputValue<gert::GertTensorData *>(addr_index);
    auto src_storage_shape = context->GetInputPointer<StorageShape>(addr_index + 2U);
    auto data_type = context->GetInputValue<ge::DataType>(addr_index + 3U);
    auto out_tensor_data =
        context->GetOutputPointer<gert::GertTensorData>(i + static_cast<size_t>(CopyFlowLaunchOutputs::kAddress));
    GE_ASSERT_NOTNULL(tensor_data);
    GE_ASSERT_NOTNULL(src_storage_shape);
    GE_ASSERT_NOTNULL(out_tensor_data);
    if (TensorPlacementUtils::IsOnDevice(tensor_data->GetPlacement())) {
      out_tensor_data->ShareFrom(*tensor_data);
      continue;
    }
    if (!TensorPlacementUtils::IsOnHost(tensor_data->GetPlacement())) {
      GELOGE(ge::GRAPH_FAILED, "unsupported copy form placement %d to device hbm",
             static_cast<int32_t>(tensor_data->GetPlacement()));
      return ge::GRAPH_FAILED;
    }

    const auto host_tensor_size = ge::GetSizeInBytes(src_storage_shape->GetStorageShape().GetShapeSize(), data_type);
    if (host_tensor_size < 0) {
      GELOGE(ge::GRAPH_FAILED, "shape_size[%" PRId64 "], data_type[%s]",
             src_storage_shape->GetStorageShape().GetShapeSize(),
             ge::TypeUtils::DataTypeToSerialString(data_type).c_str());
      return ge::GRAPH_FAILED;
    }
    const auto allocated_tensor_data = allocated_data[i];
    GE_ASSERT_NOTNULL(allocated_tensor_data);
    if (allocated_tensor_data->GetAddr() != nullptr) {
      out_tensor_data->ShareFrom(*allocated_tensor_data);
      continue;
    }

    auto inputs_index_cv = inputs_index_cvv->Get(i);
    GE_ASSERT_NOTNULL(inputs_index_cv);
    RtKernelLaunchArgsEx::HostInputInfo host_input{tensor_data->GetAddr(), inputs_index_cv,
                                                   static_cast<size_t>(host_tensor_size)};
    GE_ASSERT_SUCCESS(args->UpdateHostInputArgs(host_input));
  }
  GE_ASSERT_GRAPH_SUCCESS(args->AlignHostInputSize());
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus LaunchCopyFlowH2D(KernelContext *context) {
  GE_ASSERT_TRUE(context->GetOutputNum() == 0U, "LaunchCopyFlowH2D output num must be 0, but got %zu",
                 context->GetOutputNum());
  auto input_num = context->GetInputPointer<size_t>(static_cast<size_t>(LaunchCopyFlowH2DInputs::kInputsNum));
  GE_CHECK_NOTNULL(input_num);
  GE_ASSERT_TRUE(*input_num > 0U);
  if (ValidateCopyFlowInputNum(context, static_cast<size_t>(LaunchCopyFlowH2DInputs::kAddrAndLengthStart),
                               *input_num) != ge::GRAPH_SUCCESS) {
    return ge::GRAPH_FAILED;
  }

  auto stream = context->GetInputValue<aclrtStream>(static_cast<size_t>(LaunchCopyFlowH2DInputs::kStream));
  auto allocated_addrs = context->GetInputPointer<TypedContinuousVector<GertTensorData *>>(
      static_cast<size_t>(LaunchCopyFlowH2DInputs::kAllocatedAddrs));
  GE_ASSERT_NOTNULL(allocated_addrs);
  GE_ASSERT_TRUE(allocated_addrs->GetSize() == *input_num, "allocated addr num %zu is not match input num %zu",
                 allocated_addrs->GetSize(), *input_num);
  auto allocated_data = allocated_addrs->GetData();
  GE_ASSERT_NOTNULL(allocated_data);

  for (size_t i = 0U; i < *input_num; ++i) {
    const auto addr_index = i * kSizeOfCopyToDevice + static_cast<size_t>(LaunchCopyFlowH2DInputs::kAddrAndLengthStart);
    auto tensor_data = context->GetInputValue<gert::GertTensorData *>(addr_index);
    auto src_storage_shape = context->GetInputPointer<StorageShape>(addr_index + 2U);
    auto data_type = context->GetInputValue<ge::DataType>(addr_index + 3U);
    GE_ASSERT_NOTNULL(tensor_data);
    GE_ASSERT_NOTNULL(src_storage_shape);
    if (TensorPlacementUtils::IsOnDevice(tensor_data->GetPlacement())) {
      continue;
    }
    if (!TensorPlacementUtils::IsOnHost(tensor_data->GetPlacement())) {
      GELOGE(ge::GRAPH_FAILED, "unsupported copy form placement %d to device hbm",
             static_cast<int32_t>(tensor_data->GetPlacement()));
      return ge::GRAPH_FAILED;
    }

    const auto host_tensor_size = ge::GetSizeInBytes(src_storage_shape->GetStorageShape().GetShapeSize(), data_type);
    if (host_tensor_size < 0) {
      GELOGE(ge::GRAPH_FAILED, "shape_size[%" PRId64 "], data_type[%s]",
             src_storage_shape->GetStorageShape().GetShapeSize(),
             ge::TypeUtils::DataTypeToSerialString(data_type).c_str());
      return ge::GRAPH_FAILED;
    }
    const auto allocated_tensor_data = allocated_data[i];
    GE_ASSERT_NOTNULL(allocated_tensor_data);
    if ((allocated_tensor_data->GetAddr() == nullptr) || (host_tensor_size == 0)) {
      continue;
    }
    GELOGD("StreamCopyH2D, host addr %p, host tensor size %zu, device addr %p, alloc device size %zu",
           tensor_data->GetAddr(), static_cast<size_t>(host_tensor_size), allocated_tensor_data->GetAddr(),
           allocated_tensor_data->GetSize());
    GE_ASSERT_RT_OK(aclrtMemcpyAsync(allocated_tensor_data->GetAddr(), allocated_tensor_data->GetSize(),
                                     tensor_data->GetAddr(), static_cast<size_t>(host_tensor_size),
                                     ACL_MEMCPY_HOST_TO_BUF_TO_DEVICE, stream));
  }
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus CopyFlowLaunch(KernelContext *context) {
  auto output_num = context->GetOutputNum();
  if (static_cast<size_t>(CopyFlowLaunchInputs::kAddrAndLengthStart) + (output_num * kSizeOfCopyToDevice) !=
      context->GetInputNum()) {
    GELOGE(ge::GRAPH_FAILED, "input num is not matched, input start %zu, output num %zu, total input num %zu",
           static_cast<size_t>(CopyFlowLaunchInputs::kAddrAndLengthStart), output_num, context->GetInputNum());
    return ge::GRAPH_FAILED;
  }

  auto input_num = context->GetInputPointer<size_t>(static_cast<size_t>(CopyFlowLaunchInputs::kInputsNum));
  GE_CHECK_NOTNULL(input_num);
  GELOGD("host input num is %zu, output num is %zu.", *input_num, output_num);
  if (*input_num != output_num) {
    GELOGE(ge::GRAPH_FAILED, "host input num %zu, is not match output num %zu,", *input_num, output_num);
    return ge::GRAPH_FAILED;
  }

  auto args =
      context->MutableInputPointer<gert::RtKernelLaunchArgsEx>(static_cast<size_t>(CopyFlowLaunchInputs::kRtArg));
  GE_CHECK_NOTNULL(args);
  // 更新host input data的offset，从图上保证先做tiling，然后 CopyFlowLaunch 进行随路拷贝
  GE_ASSERT_SUCCESS(args->UpdateMergedCopyInfo());

  auto inputs_index_cvv =
      context->GetInputValue<ContinuousVectorVector *>(static_cast<size_t>(CopyFlowLaunchInputs::kInputsIndex));
  GE_ASSERT_NOTNULL(inputs_index_cvv);
  for (size_t i = 0U; i < output_num; ++i) {
    auto addr_index = i * kSizeOfCopyToDevice + static_cast<size_t>(CopyFlowLaunchInputs::kAddrAndLengthStart);
    auto tensor_data = context->GetInputValue<gert::GertTensorData *>(addr_index);
    auto src_storage_shape = context->GetInputPointer<StorageShape>(addr_index + 2U);
    GE_CHECK_NOTNULL(src_storage_shape);
    auto data_type = context->GetInputValue<ge::DataType>(addr_index + 3U);
    auto out_tensor_data =
        context->GetOutputPointer<gert::GertTensorData>(i + static_cast<size_t>(CopyFlowLaunchOutputs::kAddress));
    GE_ASSERT_NOTNULL(tensor_data);
    GE_ASSERT_NOTNULL(out_tensor_data);
    if (TensorPlacementUtils::IsOnDevice(tensor_data->GetPlacement())) {
      GELOGD("The [%zu]th tensor data placement is %d, no need to optimize", i,
             static_cast<int32_t>(tensor_data->GetPlacement()));
      out_tensor_data->ShareFrom(*tensor_data);
    } else if (TensorPlacementUtils::IsOnHost(tensor_data->GetPlacement())) {
      const auto host_tensor_size = ge::GetSizeInBytes(src_storage_shape->GetStorageShape().GetShapeSize(), data_type);
      if (host_tensor_size < 0) {
        GELOGE(ge::GRAPH_FAILED, "shape_size[%" PRId64 "], data_type[%s]",
               src_storage_shape->GetStorageShape().GetShapeSize(),
               ge::TypeUtils::DataTypeToSerialString(data_type).c_str());
        return ge::GRAPH_FAILED;
      }
      size_t align_size = ge::RoundUp(static_cast<uint64_t>(host_tensor_size), kAlignBytes4);
      GELOGD("shape_size[%" PRId64 "], data_type[%s], host_tensor_size[%" PRId64 "], align_size[%zu]",
             src_storage_shape->GetStorageShape().GetShapeSize(),
             ge::TypeUtils::DataTypeToSerialString(data_type).c_str(), host_tensor_size, align_size);
      auto host_input_data_size = args->GetHostInputDataSize();
      host_input_data_size += align_size;
      auto max_host_input_data_len = kMaxHostInputDataLen + args->GetMergedCopySize();
      if (host_input_data_size > max_host_input_data_len) {
        GE_ASSERT_SUCCESS(CopyTensorToDevice(context, i));
      } else {
        auto inputs_index_cv = inputs_index_cvv->Get(i);
        GE_ASSERT_NOTNULL(inputs_index_cv);
        RtKernelLaunchArgsEx::HostInputInfo host_input{tensor_data->GetAddr(), inputs_index_cv,
                                                       static_cast<size_t>(host_tensor_size)};
        GE_ASSERT_SUCCESS(args->UpdateHostInputArgs(host_input));
      }
    } else {
      GELOGE(ge::GRAPH_FAILED, "unsupported copy form placement %d to device hbm",
             static_cast<int32_t>(tensor_data->GetPlacement()));
      return ge::GRAPH_FAILED;
    }
  }
  // copy flow launch之后，将字节进行对齐
  GE_ASSERT_GRAPH_SUCCESS(args->AlignHostInputSize());
  return ge::GRAPH_SUCCESS;
}
// Legacy mixed CopyFlowLaunch remains for single-thread and dynamic-multistream execution.
REGISTER_KERNEL(CopyFlowLaunch)
    .RunFunc(CopyFlowLaunch)
    .OutputsCreator(CreateCopyFlowLaunchTensorData)
    .ConcurrentCriticalSectionKey(kKernelUseMemory);
REGISTER_KERNEL(CalcCopyFlowAllocSizes).RunFunc(CalcCopyFlowAllocSizes).OutputsCreator(CreateCopyFlowAllocSizes);
REGISTER_KERNEL(PrepareCopyFlowResult)
    .RunFunc(PrepareCopyFlowResult)
    .OutputsCreator(CreatePrepareCopyFlowResultOutputs)
    .ConcurrentCriticalSectionKey(kKernelUseMemory);
REGISTER_KERNEL(LaunchCopyFlowH2D).RunFunc(LaunchCopyFlowH2D).ConcurrentCriticalSectionKey(kKernelLaunch);
}  // namespace kernel
}  // namespace gert
