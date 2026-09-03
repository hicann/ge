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
#include <functional>
#include <numeric>
#include <set>
#include <vector>

#include "common/checker.h"
#include "framework/common/framework_types_internal.h"
#include "graph/optimize/symbolic/infer_symbolic_shape/symbolic_infer_util.h"
#include "graph/optimize/symbolic/symbol_compute_context.h"
#include "graph/optimize/symbolic/symbolic_kernel_factory.h"
#include "graph/symbolizer/symbol_operator.h"

namespace ge {
namespace {
constexpr size_t kInputIndex = 0U;
constexpr size_t kOutputIndex = 0U;
constexpr size_t kAxesAttrIndex = 0U;
constexpr int64_t kMaxSymbolicValueSize = 200L;

graphStatus NormalizeAxes(const std::vector<int64_t> &dims, std::vector<int64_t> &axes) {
  std::set<int64_t> unique_axes;
  for (auto axis : axes) {
    axis = axis < 0L ? axis + static_cast<int64_t>(dims.size()) : axis;
    if (axis < 0L || axis >= static_cast<int64_t>(dims.size())) {
      GELOGW("SoftmaxV2 symbolic compute unsupported: axis[%lld] is out of range, rank[%zu].", axis, dims.size());
      return UNSUPPORTED;
    }
    unique_axes.insert(axis);
  }
  axes.assign(unique_axes.begin(), unique_axes.end());
  return GRAPH_SUCCESS;
}

std::vector<int64_t> MakeStrides(const std::vector<int64_t> &dims) {
  std::vector<int64_t> strides(dims.size(), 1L);
  for (int64_t i = static_cast<int64_t>(dims.size()) - 2L; i >= 0L; --i) {
    strides[static_cast<size_t>(i)] = strides[static_cast<size_t>(i + 1L)] * dims[static_cast<size_t>(i + 1L)];
  }
  return strides;
}

std::vector<int64_t> MakeCoordinates(int64_t index, const std::vector<int64_t> &dims,
                                     const std::vector<int64_t> &strides) {
  std::vector<int64_t> coordinates(dims.size(), 0L);
  for (size_t i = 0U; i < dims.size(); ++i) {
    coordinates[i] = (index / strides[i]) % dims[i];
  }
  return coordinates;
}

bool SameNonReduceCoordinates(const std::vector<int64_t> &lhs, const std::vector<int64_t> &rhs,
                              const std::set<int64_t> &axes) {
  for (size_t i = 0U; i < lhs.size(); ++i) {
    if (axes.count(static_cast<int64_t>(i)) == 0U && lhs[i] != rhs[i]) {
      return false;
    }
  }
  return true;
}
}  // namespace

graphStatus BuildSoftmaxValues(const std::vector<int64_t> &dims, const std::vector<int64_t> &axes,
                               const std::vector<Expression> &input, std::vector<Expression> &output) {
  const auto strides = MakeStrides(dims);
  const std::set<int64_t> axis_set(axes.begin(), axes.end());
  std::vector<Expression> exp_values;
  exp_values.reserve(input.size());
  for (const auto &value : input) {
    exp_values.emplace_back(sym::Exp(value));
  }
  for (int64_t i = 0L; i < static_cast<int64_t>(input.size()); ++i) {
    const auto coordinates = MakeCoordinates(i, dims, strides);
    Expression denominator(Symbol(0));
    for (int64_t j = 0L; j < static_cast<int64_t>(input.size()); ++j) {
      if (SameNonReduceCoordinates(coordinates, MakeCoordinates(j, dims, strides), axis_set)) {
        denominator = denominator + exp_values[static_cast<size_t>(j)];
      }
    }
    output.emplace_back(exp_values[static_cast<size_t>(i)] / denominator);
  }
  return GRAPH_SUCCESS;
}

/**
 * SoftmaxV2算子的符号化计算
 * 【算子功能】沿axes指定的维度对输入执行softmax，输出每个元素的指数值除以对应归约组的指数和。
 * 【算子约束】axes必须是常量属性且轴有效；输入Shape和SymbolicValue必须可获得，且输入元素数不超过200。
 * 【推导逻辑】先读取并归一化axes，计算输入行主序stride；逐元素构造Exp(x)，再根据非归约维度坐标
 *          将元素划分到同一归约组并累加指数值。对每个输出位置使用本组指数和作为分母，生成Exp(x)/sum
 *          表达式，最后复制输入Shape并按原线性顺序写回输出SymbolicValue。
 * 【举例】x Shape=[2,2]、value=[0,1,2,3]、axes=[-1]时，输出Shape=[2,2]，
 *          输出value为[Exp(0)/(Exp(0)+Exp(1)), Exp(1)/(Exp(0)+Exp(1)), ...]。
 */
graphStatus SoftmaxV2SymbolicKernelCompute(gert::InferSymbolComputeContext *context) {
  GE_ASSERT_NOTNULL(context);
  std::vector<int64_t> dims;
  if (!context->GetConstInputDims(kInputIndex, dims)) {
    GELOGW("SoftmaxV2 symbolic compute unsupported: input Shape is not constant, node %s[%s].", context->GetNodeName(),
           context->GetNodeType());
    return UNSUPPORTED;
  }
  auto input_tensor = context->GetInputSymbolTensor(kInputIndex);
  GE_UNSUPPORTED_IF_NULL(input_tensor);
  auto input_values = input_tensor->GetSymbolicValue();
  GE_UNSUPPORTED_IF_NULL(input_values);
  int64_t element_count = std::accumulate(dims.begin(), dims.end(), 1L, std::multiplies<int64_t>());
  if (element_count <= 0L || element_count > kMaxSymbolicValueSize ||
      input_values->size() != static_cast<size_t>(element_count)) {
    GELOGW(
        "SoftmaxV2 symbolic compute unsupported: input element count[%lld] or SymbolicValue size[%zu] is invalid, "
        "node %s[%s].",
        element_count, input_values->size(), context->GetNodeName(), context->GetNodeType());
    return UNSUPPORTED;
  }

  auto attrs = context->GetAttrs();
  GE_UNSUPPORTED_IF_NULL(attrs);
  auto axes_attr = attrs->GetListInt(kAxesAttrIndex);
  std::vector<int64_t> axes = {-1L};
  if (axes_attr != nullptr && axes_attr->GetSize() > 0U) {
    axes.assign(axes_attr->GetData(), axes_attr->GetData() + axes_attr->GetSize());
  }
  auto ret = NormalizeAxes(dims, axes);
  if (ret != GRAPH_SUCCESS) {
    return ret;
  }
  auto output_tensor = context->GetOutputSymbolTensor(kOutputIndex);
  GE_ASSERT_NOTNULL(output_tensor);
  output_tensor->MutableOriginSymbolShape() = input_tensor->GetOriginSymbolShape();
  auto output_values = output_tensor->MutableSymbolicValue();
  GE_ASSERT_NOTNULL(output_values);
  output_values->reserve(input_values->size());
  ret = BuildSoftmaxValues(dims, axes, *input_values, *output_values);
  if (ret != GRAPH_SUCCESS) {
    return ret;
  }
  return GRAPH_SUCCESS;
}

REGISTER_SYMBOLIC_KERNEL(SoftmaxV2, SoftmaxV2SymbolicKernelCompute);
}  // namespace ge
