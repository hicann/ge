/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef METADEF_CXX_INC_EXE_GRAPH_RUNTIME_HOST_CPU_OP_EXECUTION_CONTEXT_H_
#define METADEF_CXX_INC_EXE_GRAPH_RUNTIME_HOST_CPU_OP_EXECUTION_CONTEXT_H_

#include <type_traits>

#include "exe_graph/runtime/extended_kernel_context.h"

namespace gert {
class HostCpuOpExecutionContext : public ExtendedKernelContext {
 public:
  /**
   * 根据输入 index 获取输入 Tensor 指针。
   * @param index 输入 index
   * @return Tensor 指针，异常时返回空指针
   */
  const Tensor *GetInputTensor(size_t index) const {
    if (static_cast<int64_t>(index) >= GetAdditionalInputStartIndex()) {
      return nullptr;
    }
    return GetInputPointer<Tensor>(index);
  }

  /**
   * 基于算子 IR 原型定义，获取 REQUIRED_INPUT 类型的输入 Tensor 指针。
   * @param ir_index IR 原型定义中的 index
   * @return Tensor 指针，异常时返回空指针
   */
  const Tensor *GetRequiredInputTensor(size_t ir_index) const {
    return GetDynamicInputPointer<Tensor>(ir_index, 0);
  }

  /**
   * 基于算子 IR 原型定义，获取 OPTIONAL_INPUT 类型的输入 Tensor 指针。
   * @param ir_index IR 原型定义中的 index
   * @return Tensor 指针，异常时返回空指针
   */
  const Tensor *GetOptionalInputTensor(size_t ir_index) const {
    return GetDynamicInputPointer<Tensor>(ir_index, 0);
  }

  /**
   * 基于算子 IR 原型定义，获取 DYNAMIC_INPUT 类型的输入 Tensor 指针。
   * @param ir_index IR 原型定义中的 index
   * @param relative_index 该输入实例化后的相对 index
   * @return Tensor 指针，异常时返回空指针
   */
  const Tensor *GetDynamicInputTensor(size_t ir_index, size_t relative_index) const {
    return GetDynamicInputPointer<Tensor>(ir_index, relative_index);
  }

  /**
   * 根据输出 index 获取输出 Tensor 指针。
   * @param index 输出 index
   * @return Tensor 指针，异常时返回空指针
   */
  const Tensor *GetOutputTensor(size_t index) const {
    return GetOutputPointer<Tensor>(index);
  }

  /**
   * 为某个 Host CPU 输出 tensor 申请 host 内存，同时初始化输出 tensor 的基本信息。
   * @param index 输出 index
   * @param shape 输出 tensor 的 shape
   * @param format 输出 tensor 的 format
   * @param dtype 输出 tensor 的 data type
   * @return Tensor 指针，异常时返回空指针
   */
  Tensor *MallocOutputTensor(size_t index, const StorageShape &shape, const StorageFormat &format, ge::DataType dtype);

  /**
   * 指定某输出的内存地址引用自某个输入，同时初始化tensor的基本信息。
   * @param output_index 输出 index
   * @param input_index 输入 index
   * @return output_index 对应的输出 Tensor 指针，异常时返回空指针
   */
  Tensor *MakeOutputRefInput(size_t output_index, size_t input_index);

  enum class AdditionalInputIndex : uint32_t { kHostAllocator = 0U, kNum };

 protected:
  int64_t GetAdditionalInputStartIndex() const {
    const auto compute_node_info = GetComputeNodeInfo();
    if (compute_node_info == nullptr) {
      return -1;
    }
    return compute_node_info->GetInputsNum();
  }
};

// The runtime casts a KernelContext view to this context type. Keep it a standard-layout,
// zero-member extension.
static_assert(std::is_standard_layout<HostCpuOpExecutionContext>::value,
              "HostCpuOpExecutionContext must be standard layout");
static_assert(sizeof(HostCpuOpExecutionContext) == sizeof(ExtendedKernelContext),
              "HostCpuOpExecutionContext must not add member variables");
}  // namespace gert

#endif  // METADEF_CXX_INC_EXE_GRAPH_RUNTIME_HOST_CPU_OP_EXECUTION_CONTEXT_H_
