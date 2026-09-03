/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <limits>
#include <vector>

#include "common/checker.h"
#include "common/plugin/ge_make_unique_util.h"
#include "framework/common/framework_types_internal.h"
#include "graph/optimize/symbolic/infer_symbolic_shape/symbolic_infer_util.h"
#include "graph/optimize/symbolic/symbol_compute_context.h"
#include "graph/optimize/symbolic/symbolic_kernel_factory.h"

namespace ge {
namespace {
constexpr size_t kSliceInputNum = 3U;
constexpr size_t kSliceOutputNum = 1U;
constexpr size_t kXIndex = 0U;
constexpr size_t kOffsetsIndex = 1U;
constexpr size_t kSizeIndex = 2U;
constexpr size_t kMaxSymbolicValueSize = 200U;

bool GetIndexValues(const gert::SymbolTensor *tensor, std::vector<int64_t> &values) {
  if (tensor == nullptr || tensor->GetSymbolicValue() == nullptr) {
    return false;
  }
  values.clear();
  for (const auto &expr : *tensor->GetSymbolicValue()) {
    int64_t value = 0L;
    if (!expr.GetConstValue(value)) {
      return false;
    }
    values.emplace_back(value);
  }
  return true;
}

}  // namespace

graphStatus GetSliceInputs(gert::InferSymbolComputeContext *context, std::vector<int64_t> &x_dims,
                           std::vector<int64_t> &offsets, std::vector<int64_t> &sizes,
                           const std::vector<Expression> *&x_values) {
  const auto x_tensor = context->GetInputSymbolTensor(kXIndex);
  GE_UNSUPPORTED_IF_NULL(x_tensor);
  x_values = x_tensor->GetSymbolicValue();
  GE_UNSUPPORTED_IF_NULL(x_values);
  if (!context->GetConstInputDims(kXIndex, x_dims)) {
    GELOGW("Slice symbolic compute unsupported: input shape is not constant, node %s[%s].", context->GetNodeName(),
           context->GetNodeType());
    return UNSUPPORTED;
  }
  if (!GetIndexValues(context->GetInputSymbolTensor(kOffsetsIndex), offsets) ||
      !GetIndexValues(context->GetInputSymbolTensor(kSizeIndex), sizes) || offsets.size() != x_dims.size() ||
      sizes.size() != x_dims.size()) {
    GELOGW("Slice symbolic compute unsupported: offsets or sizes are invalid or rank-mismatched, node %s[%s].",
           context->GetNodeName(), context->GetNodeType());
    return UNSUPPORTED;
  }
  return GRAPH_SUCCESS;
}

graphStatus BuildSliceShape(gert::InferSymbolComputeContext *context, const std::vector<int64_t> &x_dims,
                            const std::vector<int64_t> &offsets, const std::vector<int64_t> &sizes,
                            std::vector<Expression> &output_shape, std::vector<int64_t> &output_dims) {
  for (size_t i = 0U; i < x_dims.size(); ++i) {
    if (offsets[i] < 0L || offsets[i] > x_dims[i] || sizes[i] < -1L) {
      GELOGW("Slice symbolic compute unsupported: invalid offset or size at dimension %zu, node %s[%s].", i,
             context->GetNodeName(), context->GetNodeType());
      return UNSUPPORTED;
    }
    const int64_t output_dim = sizes[i] == -1L ? x_dims[i] - offsets[i] : sizes[i];
    if (output_dim < 0L || offsets[i] + output_dim > x_dims[i]) {
      GELOGW("Slice symbolic compute unsupported: slice range exceeds input at dimension %zu, node %s[%s].", i,
             context->GetNodeName(), context->GetNodeType());
      return UNSUPPORTED;
    }
    output_shape.emplace_back(sizes[i] == -1L ? context->GetInputSymbolShape(kXIndex)->GetDim(i) - Symbol(offsets[i])
                                              : Symbol(output_dim));
    output_dims.emplace_back(output_dim);
  }
  return GRAPH_SUCCESS;
}

graphStatus BuildSliceValues(gert::InferSymbolComputeContext *context, const std::vector<int64_t> &x_dims,
                             const std::vector<int64_t> &offsets, const std::vector<int64_t> &output_dims,
                             const std::vector<Expression> *x_values, std::vector<Expression> &output_values) {
  int64_t output_size = 1L;
  for (const auto dim : output_dims) {
    if (dim != 0L && output_size > static_cast<int64_t>(kMaxSymbolicValueSize) / dim) {
      GELOGW("Slice symbolic compute unsupported: output symbolic value size overflows the limit, node %s[%s].",
             context->GetNodeName(), context->GetNodeType());
      return UNSUPPORTED;
    }
    output_size *= dim;
  }
  if (output_size > static_cast<int64_t>(kMaxSymbolicValueSize)) {
    GELOGW("Slice symbolic compute unsupported: output symbolic value size[%ld] exceeds limit[%u], node %s[%s].",
           output_size, kMaxSymbolicValueSize, context->GetNodeName(), context->GetNodeType());
    return UNSUPPORTED;
  }
  std::vector<int64_t> x_strides(x_dims.size(), 1L);
  std::vector<int64_t> output_strides(output_dims.size(), 1L);
  auto input_strides = x_strides;
  for (int64_t i = static_cast<int64_t>(x_dims.size()) - 2L; i >= 0L; --i) {
    input_strides[static_cast<size_t>(i)] =
        input_strides[static_cast<size_t>(i + 1L)] * x_dims[static_cast<size_t>(i + 1L)];
    output_strides[static_cast<size_t>(i)] =
        output_strides[static_cast<size_t>(i + 1L)] * output_dims[static_cast<size_t>(i + 1L)];
  }
  for (int64_t output_index = 0L; output_index < output_size; ++output_index) {
    int64_t source_index = 0L;
    for (size_t i = 0U; i < output_dims.size(); ++i) {
      source_index += ((output_index / output_strides[i]) % output_dims[i] + offsets[i]) * input_strides[i];
    }
    if (source_index < 0L || static_cast<size_t>(source_index) >= x_values->size()) {
      GELOGW("Slice symbolic compute unsupported: calculated source index[%ld] is out of range, node %s[%s].",
             source_index, context->GetNodeName(), context->GetNodeType());
      return UNSUPPORTED;
    }
    output_values.emplace_back(x_values->at(static_cast<size_t>(source_index)));
  }
  return GRAPH_SUCCESS;
}

/**
 * Slice算子的符号化计算
 * 【算子功能】按照offsets和sizes从输入张量中提取连续切片。
 * 【推导逻辑】先解析offsets和sizes并校验每个轴的起点、长度及边界，得到输出Shape；再计算输入输出
 *          的行主序stride，遍历输出坐标加上offset映射回输入坐标，按行主序填充输出SymbolicValue。
 * 【算子约束】输入Shape、offsets和sizes必须可确定，切片范围合法，输出SymbolicValue元素数不超过上限。
 * 【举例】输入Shape=[2,3]、输入value=[0,1,2,3,4,5]、offsets value=[0,1]、sizes value=[2,2]时，
 *          输出Shape为[2,2]，输出value为[1,2,4,5]。
 */
graphStatus SliceSymbolicKernelCompute(gert::InferSymbolComputeContext *context) {
  GE_ASSERT_NOTNULL(context);
  GE_ASSERT(context->GetComputeNodeInputNum() == kSliceInputNum, "InputNum=%zu", context->GetComputeNodeInputNum());
  GE_ASSERT(context->GetComputeNodeOutputNum() == kSliceOutputNum, "OutputNum=%zu", context->GetComputeNodeOutputNum());

  std::vector<int64_t> x_dims;
  std::vector<int64_t> offsets;
  std::vector<int64_t> sizes;
  const std::vector<Expression> *x_values = nullptr;
  auto ret = GetSliceInputs(context, x_dims, offsets, sizes, x_values);
  if (ret != GRAPH_SUCCESS) {
    return ret;
  }

  std::vector<Expression> output_shape;
  std::vector<int64_t> output_dims;
  output_shape.reserve(x_dims.size());
  output_dims.reserve(x_dims.size());
  ret = BuildSliceShape(context, x_dims, offsets, sizes, output_shape, output_dims);
  if (ret != GRAPH_SUCCESS) {
    return ret;
  }

  auto output_tensor = context->GetOutputSymbolTensor(0U);
  GE_ASSERT_NOTNULL(output_tensor);
  output_tensor->MutableOriginSymbolShape().MutableDims() = std::move(output_shape);
  auto output_values = output_tensor->MutableSymbolicValue();
  GE_ASSERT_NOTNULL(output_values);
  ret = BuildSliceValues(context, x_dims, offsets, output_dims, x_values, *output_values);
  if (ret != GRAPH_SUCCESS) {
    return ret;
  }
  return GRAPH_SUCCESS;
}

REGISTER_SYMBOLIC_KERNEL(Slice, SliceSymbolicKernelCompute);
}  // namespace ge
