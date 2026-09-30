/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef METADEF_CXX_INC_EXE_GRAPH_RUNTIME_RESOURCE_USAGE_CONTEXT_H_
#define METADEF_CXX_INC_EXE_GRAPH_RUNTIME_RESOURCE_USAGE_CONTEXT_H_

#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <vector>

#include "exe_graph/runtime/extended_kernel_context.h"
#include "graph/ascend_string.h"
#include "graph/error_codes.h"
#include "graph/tensor.h"

namespace gert {
/**
 * 资源申报上下文，供 ResourceUsageReporter::DeclareResourceUsage 在编译期使用。
 * 继承自 ExtendedKernelContext，可通过基类接口获取节点名、节点类型、算子属性
 * 以及输入/输出 tensor 描述等编译期信息。
 * @since 9.3.0(2026-09)
 */
class ResourceUsageContext : public ExtendedKernelContext {
 public:
  /**
   * 上报本节点运行期将请求的辅流列表。
   * 框架按 key 去重统计。
   * 空 key 将导致上报失败并使编译失败。
   * @param keys 辅流 key 列表
   * @return GRAPH_SUCCESS 表示上报成功，否则返回错误码
   * @since 9.3.0(2026-09)
   */
  ge::graphStatus ReportAttachedStream(const std::vector<ge::AscendString> &keys);

  /**
   * 根据输入 index 获取输入 Tensor 指针，可读取编译期静态 shape、format、dtype。
   * @param index 输入索引
   * @return Tensor 指针，异常或越界时返回 nullptr
   * @since 9.3.0(2026-09)
   */
  const Tensor *GetInputTensor(size_t index) const {
    const auto additional_input_start = GetAdditionalInputStartIndex();
    if ((additional_input_start < 0) || (index >= static_cast<size_t>(additional_input_start))) {
      return nullptr;
    }
    return GetInputPointer<Tensor>(index);
  }

  /**
   * 根据输出 index 获取输出 Tensor 指针，可读取编译期静态 shape、format、dtype。
   * @param index 输出索引
   * @return Tensor 指针，异常或越界时返回 nullptr
   * @since 9.3.0(2026-09)
   */
  const Tensor *GetOutputTensor(size_t index) const {
    const auto compute_node_info = GetComputeNodeInfo();
    if (compute_node_info == nullptr) {
      return nullptr;
    }
    if (index >= compute_node_info->GetOutputsNum()) {
      return nullptr;
    }
    return GetOutputPointer<Tensor>(index);
  }

  /**
   * 基于算子 IR 原型定义，获取 REQUIRED_INPUT 类型的输入 Tensor 指针。
   * @param ir_index IR 原型定义中的 index
   * @return Tensor 指针，异常或该输入未实例化时返回 nullptr
   * @since 9.3.0(2026-09)
   */
  const Tensor *GetRequiredInputTensor(size_t ir_index) const {
    return GetDynamicInputPointer<Tensor>(ir_index, 0);
  }

  /**
   * 基于算子 IR 原型定义，获取 OPTIONAL_INPUT 类型的输入 Tensor 指针。
   * @param ir_index IR 原型定义中的 index
   * @return Tensor 指针，异常或该输入未实例化时返回 nullptr
   * @since 9.3.0(2026-09)
   */
  const Tensor *GetOptionalInputTensor(size_t ir_index) const {
    return GetDynamicInputPointer<Tensor>(ir_index, 0);
  }

  /**
   * 基于算子 IR 原型定义，获取 DYNAMIC_INPUT 类型的输入 Tensor 指针。
   * @param ir_index IR 原型定义中的 index
   * @param relative_index 该输入实例化后的相对 index，例如某个 DYNAMIC_INPUT 实例化了 3 个输入，
   * 那么 relative_index 的有效范围是 [0, 2]
   * @return Tensor 指针，异常或越界时返回 nullptr
   * @since 9.3.0(2026-09)
   */
  const Tensor *GetDynamicInputTensor(size_t ir_index, size_t relative_index) const {
    return GetDynamicInputPointer<Tensor>(ir_index, relative_index);
  }

  /**
   * 基于算子 IR 原型定义，获取 REQUIRED_OUTPUT 类型的输出 Tensor 指针。
   * @param ir_index IR 原型定义中的 index
   * @return Tensor 指针，异常或该输出未实例化时返回 nullptr
   * @since 9.3.0(2026-09)
   */
  const Tensor *GetRequiredOutputTensor(size_t ir_index) const {
    const auto ins_info = GetIrOutputInstanceInfo(ir_index);
    if (ins_info == nullptr) {
      return nullptr;
    }
    if (ins_info->GetInstanceNum() == 0U) {
      return nullptr;
    }
    return GetOutputPointer<Tensor>(ins_info->GetInstanceStart());
  }

  /**
   * 基于算子 IR 原型定义，获取 DYNAMIC_OUTPUT 类型的输出 Tensor 指针。
   * @param ir_index IR 原型定义中的 index
   * @param relative_index 该输出实例化后的相对 index，例如某个 DYNAMIC_OUTPUT 实例化了 3 个输出，
   * 那么 relative_index 的有效范围是 [0, 2]
   * @return Tensor 指针，异常或越界时返回 nullptr
   * @since 9.3.0(2026-09)
   */
  const Tensor *GetDynamicOutputTensor(size_t ir_index, size_t relative_index) const {
    const auto ins_info = GetIrOutputInstanceInfo(ir_index);
    if (ins_info == nullptr) {
      return nullptr;
    }
    if (ins_info->GetInstanceNum() <= relative_index) {
      return nullptr;
    }
    return GetOutputPointer<Tensor>(ins_info->GetInstanceStart() + relative_index);
  }

 protected:
  int64_t GetAdditionalInputStartIndex() const {
    const auto compute_node_info = GetComputeNodeInfo();
    if (compute_node_info == nullptr) {
      return -1;
    }
    return compute_node_info->GetInputsNum();
  }
};

static_assert(std::is_standard_layout<ResourceUsageContext>::value, "ResourceUsageContext must be standard layout");
static_assert(sizeof(ResourceUsageContext) == sizeof(ExtendedKernelContext),
              "ResourceUsageContext must not add member variables");
}  // namespace gert

#endif  // METADEF_CXX_INC_EXE_GRAPH_RUNTIME_RESOURCE_USAGE_CONTEXT_H_
