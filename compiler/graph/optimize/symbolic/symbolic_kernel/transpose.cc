/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <algorithm>
#include <vector>

#include "common/checker.h"
#include "common/plugin/ge_make_unique_util.h"
#include "framework/common/framework_types_internal.h"
#include "graph/optimize/symbolic/infer_symbolic_shape/symbolic_infer_util.h"
#include "graph/optimize/symbolic/symbol_compute_context.h"
#include "graph/optimize/symbolic/symbolic_kernel_factory.h"

namespace ge {
namespace {
constexpr size_t kInputXIndex = 0U;
constexpr size_t kInputPermIndex = 1U;
constexpr size_t kOutputIndex = 0U;

bool GetPerm(const gert::SymbolTensor *perm_tensor, std::vector<int64_t> &perm) {
  if (perm_tensor == nullptr || perm_tensor->GetSymbolicValue() == nullptr) {
    return false;
  }
  perm.clear();
  for (const auto &expr : *perm_tensor->GetSymbolicValue()) {
    int64_t value = 0L;
    if (!expr.GetConstValue(value)) {
      return false;
    }
    perm.emplace_back(value);
  }
  return true;
}

std::vector<int64_t> GetStrides(const std::vector<int64_t> &dims) {
  std::vector<int64_t> strides(dims.size(), 1L);
  for (int64_t i = static_cast<int64_t>(dims.size()) - 2L; i >= 0L; --i) {
    strides[static_cast<size_t>(i)] = strides[static_cast<size_t>(i + 1L)] * dims[static_cast<size_t>(i + 1L)];
  }
  return strides;
}

}  // namespace

graphStatus CheckTransposeInputs(gert::InferSymbolComputeContext *context, std::vector<int64_t> &dims,
                                 std::vector<int64_t> &perm, const std::vector<Expression> *&values) {
  const auto tensor = context->GetInputSymbolTensor(kInputXIndex);
  GE_UNSUPPORTED_IF_NULL(tensor);
  values = tensor->GetSymbolicValue();
  GE_UNSUPPORTED_IF_NULL(values);
  if (!context->GetConstInputDims(kInputXIndex, dims)) {
    GELOGW("Transpose symbolic compute unsupported: input shape is not constant, node %s[%s].", context->GetNodeName(),
           context->GetNodeType());
    return UNSUPPORTED;
  }
  if (!GetPerm(context->GetInputSymbolTensor(kInputPermIndex), perm) || perm.size() != dims.size()) {
    GELOGW("Transpose symbolic compute unsupported: perm is missing, non-constant, or rank-mismatched, node %s[%s].",
           context->GetNodeName(), context->GetNodeType());
    return UNSUPPORTED;
  }
  std::vector<bool> visited(dims.size(), false);
  for (auto &axis : perm) {
    axis = axis < 0L ? axis + static_cast<int64_t>(dims.size()) : axis;
    if (axis < 0L || axis >= static_cast<int64_t>(dims.size()) || visited[static_cast<size_t>(axis)]) {
      GELOGW("Transpose symbolic compute unsupported: perm contains an invalid or duplicate axis, node %s[%s].",
             context->GetNodeName(), context->GetNodeType());
      return UNSUPPORTED;
    }
    visited[static_cast<size_t>(axis)] = true;
  }
  return GRAPH_SUCCESS;
}

/**
 * Transpose算子的符号化计算
 * 【算子功能】按照perm重新排列输入张量的维度和元素布局。
 * 【推导逻辑】先读取并解析perm，转换负轴后校验其长度等于输入秩且每个轴只出现一次；输出Shape按
 *          perm顺序选择输入维度。随后计算输入和输出的行主序stride，遍历每个输出线性位置，将输出
 *          坐标按perm映射回输入线性位置，从而重排输入SymbolicValue。
 * 【算子约束】perm必须是合法的完整置换，输入Shape必须可确定，输出元素数必须与输入值数量一致且不超过上限。
 * 【举例】输入Shape=[2,3]、输入value=[0,1,2,3,4,5]、perm value=[1,0]时，输出Shape为[3,2]，
 *          输出value为[0,3,1,4,2,5]。
 */
graphStatus TransposeSymbolicKernelCompute(gert::InferSymbolComputeContext *context) {
  GE_ASSERT_NOTNULL(context);
  GE_ASSERT(context->GetComputeNodeInputNum() == 2U, "InputNum=%zu", context->GetComputeNodeInputNum());
  GE_ASSERT(context->GetComputeNodeOutputNum() == 1U, "OutputNum=%zu", context->GetComputeNodeOutputNum());

  std::vector<int64_t> input_dims;
  std::vector<int64_t> perm;
  const std::vector<Expression> *x_values = nullptr;
  const auto ret = CheckTransposeInputs(context, input_dims, perm, x_values);
  if (ret != GRAPH_SUCCESS) {
    return ret;
  }
  const auto x_tensor = context->GetInputSymbolTensor(kInputXIndex);

  const auto input_shape = x_tensor->GetOriginSymbolShape();
  std::vector<Expression> output_shape;
  std::vector<int64_t> output_dims;
  output_shape.reserve(perm.size());
  output_dims.reserve(perm.size());
  for (const auto axis : perm) {
    output_shape.emplace_back(input_shape.GetDim(static_cast<size_t>(axis)));
    output_dims.emplace_back(input_dims[static_cast<size_t>(axis)]);
  }

  int64_t output_size = 1L;
  for (const auto dim : output_dims) {
    output_size *= dim;
  }
  if (output_size < 0L || static_cast<size_t>(output_size) != x_values->size() || output_size > 200L) {
    GELOGW(
        "Transpose symbolic compute unsupported: output size[%ld] is invalid, mismatched, or exceeds limit, node "
        "%s[%s].",
        output_size, context->GetNodeName(), context->GetNodeType());
    return UNSUPPORTED;
  }

  const auto input_strides = GetStrides(input_dims);
  const auto output_strides = GetStrides(output_dims);
  auto output_tensor = context->GetOutputSymbolTensor(kOutputIndex);
  GE_ASSERT_NOTNULL(output_tensor);
  output_tensor->MutableOriginSymbolShape().MutableDims() = std::move(output_shape);
  auto output_values = output_tensor->MutableSymbolicValue();
  GE_ASSERT_NOTNULL(output_values);
  output_values->reserve(static_cast<size_t>(output_size));
  for (int64_t output_index = 0L; output_index < output_size; ++output_index) {
    int64_t input_index = 0L;
    for (size_t output_axis = 0U; output_axis < perm.size(); ++output_axis) {
      const int64_t coordinate = (output_index / output_strides[output_axis]) % output_dims[output_axis];
      input_index += coordinate * input_strides[static_cast<size_t>(perm[output_axis])];
    }
    output_values->emplace_back(x_values->at(static_cast<size_t>(input_index)));
  }
  return GRAPH_SUCCESS;
}

REGISTER_SYMBOLIC_KERNEL(Transpose, TransposeSymbolicKernelCompute);
}  // namespace ge
