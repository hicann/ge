/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <algorithm>
#include <cstdint>
#include <functional>
#include <list>
#include <numeric>
#include <set>
#include "framework/common/framework_types_internal.h"
#include "common/plugin/ge_make_unique_util.h"
#include "common/util/mem_utils.h"
#include "common/checker.h"
#include "graph/optimize/symbolic/infer_symbolic_shape/symbolic_infer_util.h"
#include "graph/optimize/symbolic/symbol_compute_context.h"
#include "graph/optimize/symbolic/symbolic_kernel_factory.h"

namespace ge {
namespace {
constexpr size_t kXInputIndex = 0UL;
constexpr size_t kAxisInputIndex = 1UL;
constexpr size_t kOutputIndex = 0UL;

graphStatus GetAxisDims(const gert::InferSymbolComputeContext *context, std::vector<int64_t> &axis_dims) {
  std::vector<int64_t> axis_input_shape;
  context->GetConstInputDims(kAxisInputIndex, axis_input_shape);
  if (axis_input_shape.size() != 1) {
    GELOGW("SymbolicKernel compute unsupported, reason: axis_input_shape(%zu) not equal 1, node %s[%s].",
           axis_input_shape.size(), context->GetNodeName(), context->GetNodeType());
    return UNSUPPORTED;
  }
  auto axis_tensor = context->GetInputSymbolTensor(kAxisInputIndex);
  GE_UNSUPPORTED_IF_NULL(axis_tensor);
  auto axis_symbols = axis_tensor->GetSymbolicValue();
  GE_UNSUPPORTED_IF_NULL(axis_symbols);
  for (size_t i = 0UL; i < axis_symbols->size(); i++) {
    int64_t dim = 0L;
    if (!(*axis_symbols)[i].GetConstValue(dim)) {
      GELOGW("SymbolicKernel compute unsupported, reason: get %zu input const value failed, node %s[%s].", i,
             context->GetNodeName(), context->GetNodeType());
      return UNSUPPORTED;
    }
    axis_dims.emplace_back(dim);
  }
  return SUCCESS;
}

Status NormalizeAxisDims(const int64_t input_dims_size, std::vector<int64_t> &axis_dims) {
  for (size_t i = 0UL; i < axis_dims.size(); i++) {
    axis_dims[i] = axis_dims[i] < 0 ? axis_dims[i] + input_dims_size : axis_dims[i];
    GE_ASSERT_TRUE((axis_dims[i] >= 0) && (axis_dims[i] < input_dims_size), "axis: %lld should in range [0, %lld)",
                   axis_dims[i], input_dims_size);
  }
  std::sort(axis_dims.begin(), axis_dims.end());
  GELOGI("Axis dims: %s", SymbolicInferUtil::VectorToStr(axis_dims).c_str());
  return SUCCESS;
}
Status CalcOutputShape(const std::vector<int64_t> &input_dims, const std::vector<int64_t> &axis_dims,
                       const bool keep_dims, std::vector<Expression> &output_symbol_shape) {
  std::set<int64_t> axis_set(axis_dims.begin(), axis_dims.end());
  for (size_t i = 0UL; i < input_dims.size(); i++) {
    if (axis_set.count(i) > 0) {
      if (keep_dims) {
        output_symbol_shape.emplace_back(Symbol(1));
      }
      continue;
    }
    output_symbol_shape.emplace_back(Symbol(input_dims[i]));
  }
  return SUCCESS;
}
Status ReduceProdOutputSymbolValue(const std::vector<Expression> &input_symbols, const std::vector<int64_t> &input_dims,
                                   const std::vector<int64_t> &axis_dims, std::vector<Expression> &output_symbols) {
  std::vector<Expression> last_output_symbols = input_symbols;
  for (size_t axis_pos = 0UL; axis_pos < axis_dims.size(); axis_pos++) {
    output_symbols.clear();
    int64_t reduce_axis = axis_dims[axis_pos];
    // 累乘得到blocksize
    int64_t block_size =
        std::accumulate(input_dims.begin() + reduce_axis + 1, input_dims.end(), 1, std::multiplies<int64_t>());
    int64_t block_num = static_cast<int64_t>(last_output_symbols.size()) / block_size / input_dims[reduce_axis];
    GELOGI("block num: %lld, block size: %lld, index: %lld", block_num, block_size, reduce_axis);
    for (int64_t i = 0UL; i < block_num; i++) {
      std::vector<Expression> temp_result(last_output_symbols.begin() + i * input_dims[reduce_axis] * block_size,
                                          last_output_symbols.begin() + (i * input_dims[reduce_axis] + 1) * block_size);
      for (int64_t j = 1L; j < input_dims[reduce_axis]; j++) {
        auto start_iter = last_output_symbols.begin() + (i * input_dims[reduce_axis] + j) * block_size;
        std::transform(temp_result.begin(), temp_result.end(), start_iter, temp_result.begin(),
                       [](const Expression &a, const Expression &b) -> Expression { return a * b; });
      }
      output_symbols.insert(output_symbols.end(), temp_result.begin(), temp_result.end());
    }
    last_output_symbols = output_symbols;
  }
  return SUCCESS;
}
}  // namespace

static graphStatus ReduceProdSymbolicKernelCompute(gert::InferSymbolComputeContext *context) {
  GE_ASSERT_NOTNULL(context);
  GELOGD("ReduceProd Symbolic Kernel in, node %s[%s].", context->GetNodeName(), context->GetNodeType());

  // 获取输入shape
  std::vector<int64_t> input_x_dims;
  if (!context->GetConstInputDims(kXInputIndex, input_x_dims)) {
    GELOGW("SymbolicKernel compute unsupported, reason: get const input dim failed, node %s[%s].",
           context->GetNodeName(), context->GetNodeType());
    return UNSUPPORTED;
  }
  // 获取轴属性
  bool keep_dims = false;
  auto attrs = context->GetAttrs();
  GE_ASSERT_NOTNULL(attrs);
  auto keep_dims_ptr = attrs->GetBool(0);
  GE_ASSERT_NOTNULL(keep_dims_ptr);
  keep_dims = *keep_dims_ptr;

  // 获取InputSymbolsValue
  // 获取输入的值
  auto input_x_tensor = context->GetInputSymbolTensor(kXInputIndex);
  GE_UNSUPPORTED_IF_NULL(input_x_tensor);
  auto input_x_symbols = input_x_tensor->GetSymbolicValue();
  GE_UNSUPPORTED_IF_NULL(input_x_symbols);
  // 获取axis
  std::vector<int64_t> axis_dims;
  auto ret = GetAxisDims(context, axis_dims);
  if (ret != SUCCESS) {
    return ret;
  }

  GE_ASSERT_TRUE(axis_dims.size() <= input_x_dims.size(), "Axis num: %zu should not more than input shape dims: %zu",
                 axis_dims.size(), input_x_dims.size());

  // 归一化轴信息
  GE_ASSERT_SUCCESS(NormalizeAxisDims(static_cast<int64_t>(input_x_dims.size()), axis_dims));
  std::vector<Expression> output_symbol_shape;
  GE_ASSERT_SUCCESS(CalcOutputShape(input_x_dims, axis_dims, keep_dims, output_symbol_shape));

  auto out_symbolic_tensor = context->GetOutputSymbolTensor(kOutputIndex);
  GE_ASSERT_NOTNULL(out_symbolic_tensor);
  out_symbolic_tensor->MutableOriginSymbolShape().MutableDims() = output_symbol_shape;

  // 扩展x_symbols
  auto output_symbols = out_symbolic_tensor->MutableSymbolicValue();
  GE_ASSERT_NOTNULL(output_symbols);
  GE_ASSERT_SUCCESS(ReduceProdOutputSymbolValue(*input_x_symbols, input_x_dims, axis_dims, *output_symbols));
  GELOGD("%s[%s] kernel success, %s", context->GetNodeName(), context->GetNodeType(),
         SymbolicInferUtil::DumpSymbolTensor(*out_symbolic_tensor).c_str());
  return SUCCESS;
}

using ReduceCompute = std::function<Expression(const Expression &, const Expression &)>;

Status ReduceOutputSymbolValue(const std::vector<Expression> &input_symbols, const std::vector<int64_t> &input_dims,
                               const std::vector<int64_t> &axis_dims, const ReduceCompute &compute,
                               std::vector<Expression> &output_symbols) {
  std::vector<Expression> current_symbols = input_symbols;
  std::vector<int64_t> current_dims = input_dims;
  for (auto axis_iter = axis_dims.rbegin(); axis_iter != axis_dims.rend(); ++axis_iter) {
    const int64_t axis = *axis_iter;
    if (axis < 0L || static_cast<size_t>(axis) >= current_dims.size() || current_dims[axis] <= 0L) {
      return UNSUPPORTED;
    }
    int64_t block_size = 1L;
    for (size_t i = static_cast<size_t>(axis) + 1U; i < current_dims.size(); ++i) {
      if (current_dims[i] != 0L && block_size > INT64_MAX / current_dims[i]) {
        return UNSUPPORTED;
      }
      block_size *= current_dims[i];
    }
    const int64_t reduce_size = current_dims[axis];
    const int64_t group_size = reduce_size * block_size;
    if (group_size <= 0L || current_symbols.size() % static_cast<size_t>(group_size) != 0U) {
      return UNSUPPORTED;
    }
    const int64_t group_count = static_cast<int64_t>(current_symbols.size()) / group_size;
    std::vector<Expression> reduced;
    reduced.reserve(static_cast<size_t>(group_count * block_size));
    for (int64_t group = 0L; group < group_count; ++group) {
      const int64_t group_start = group * group_size;
      for (int64_t offset = 0L; offset < block_size; ++offset) {
        Expression value = current_symbols[static_cast<size_t>(group_start + offset)];
        for (int64_t reduce_index = 1L; reduce_index < reduce_size; ++reduce_index) {
          const int64_t index = group_start + reduce_index * block_size + offset;
          value = compute(value, current_symbols[static_cast<size_t>(index)]);
        }
        reduced.emplace_back(std::move(value));
      }
    }
    current_symbols = std::move(reduced);
    current_dims.erase(current_dims.begin() + axis);
  }
  output_symbols = std::move(current_symbols);
  return SUCCESS;
}

static graphStatus ReduceSymbolicKernelCompute(gert::InferSymbolComputeContext *context, const ReduceCompute &compute) {
  GE_ASSERT_NOTNULL(context);
  std::vector<int64_t> input_dims;
  if (!context->GetConstInputDims(kXInputIndex, input_dims)) {
    GELOGW("SymbolicKernel compute unsupported, reason: get const input dim failed, node %s[%s].",
           context->GetNodeName(), context->GetNodeType());
    return UNSUPPORTED;
  }
  auto input_tensor = context->GetInputSymbolTensor(kXInputIndex);
  GE_UNSUPPORTED_IF_NULL(input_tensor);
  auto input_symbols = input_tensor->GetSymbolicValue();
  GE_UNSUPPORTED_IF_NULL(input_symbols);

  auto attrs = context->GetAttrs();
  GE_ASSERT_NOTNULL(attrs);
  auto keep_dims = attrs->GetBool(0);
  GE_ASSERT_NOTNULL(keep_dims);

  std::vector<int64_t> axis_dims;
  GE_ASSERT_SUCCESS(GetAxisDims(context, axis_dims));
  GE_ASSERT_TRUE(axis_dims.size() <= input_dims.size());
  GE_ASSERT_SUCCESS(NormalizeAxisDims(static_cast<int64_t>(input_dims.size()), axis_dims));
  axis_dims.erase(std::unique(axis_dims.begin(), axis_dims.end()), axis_dims.end());

  bool noop_with_empty_axes = true;
  const bool *noop_attr = attrs->GetAttrPointer<bool>(1);
  if (noop_attr != nullptr) {
    noop_with_empty_axes = *noop_attr;
  }
  if (axis_dims.empty() && !noop_with_empty_axes) {
    axis_dims.resize(input_dims.size());
    std::iota(axis_dims.begin(), axis_dims.end(), 0L);
  }

  auto output_tensor = context->GetOutputSymbolTensor(kOutputIndex);
  GE_ASSERT_NOTNULL(output_tensor);
  std::vector<Expression> output_shape;
  GE_ASSERT_SUCCESS(CalcOutputShape(input_dims, axis_dims, *keep_dims, output_shape));
  output_tensor->MutableOriginSymbolShape().MutableDims() = output_shape;

  std::vector<Expression> output_symbols;
  if (axis_dims.empty()) {
    output_symbols = *input_symbols;
  } else {
    GE_ASSERT_SUCCESS(ReduceOutputSymbolValue(*input_symbols, input_dims, axis_dims, compute, output_symbols));
  }
  output_tensor->SetSymbolicValue(ge::MakeUnique<std::vector<Expression>>(std::move(output_symbols)));
  return SUCCESS;
}

static graphStatus ReduceSumSymbolicKernelCompute(gert::InferSymbolComputeContext *context) {
  return ReduceSymbolicKernelCompute(context, [](const Expression &x1, const Expression &x2) { return x1 + x2; });
}

static graphStatus ReduceMaxSymbolicKernelCompute(gert::InferSymbolComputeContext *context) {
  return ReduceSymbolicKernelCompute(context,
                                     [](const Expression &x1, const Expression &x2) { return sym::Max(x1, x2); });
}

static graphStatus ReduceMinSymbolicKernelCompute(gert::InferSymbolComputeContext *context) {
  return ReduceSymbolicKernelCompute(context,
                                     [](const Expression &x1, const Expression &x2) { return sym::Min(x1, x2); });
}

REGISTER_SYMBOLIC_KERNEL(ReduceProd, ReduceProdSymbolicKernelCompute);
REGISTER_SYMBOLIC_KERNEL(ReduceSum, ReduceSumSymbolicKernelCompute);
REGISTER_SYMBOLIC_KERNEL(ReduceMax, ReduceMaxSymbolicKernelCompute);
REGISTER_SYMBOLIC_KERNEL(ReduceMin, ReduceMinSymbolicKernelCompute);
}  // namespace ge
