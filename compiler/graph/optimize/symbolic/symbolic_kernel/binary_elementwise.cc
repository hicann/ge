/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "common/plugin/ge_make_unique_util.h"
#include "framework/common/framework_types_internal.h"
#include "common/util/mem_utils.h"
#include "common/checker.h"
#include "graph/optimize/symbolic/symbol_compute_context.h"
#include "graph/optimize/symbolic/infer_symbolic_shape/symbolic_infer_util.h"
#include "graph/optimize/symbolic/symbolic_kernel_factory.h"

namespace ge {
namespace {
constexpr size_t kX1InputIndex = 0UL;
constexpr size_t kX2InputIndex = 1UL;
constexpr size_t kOutputIndex = 0UL;

graphStatus BroadcastData(const std::vector<Expression> &src_data, const std::vector<int64_t> &src_shape,
                          const std::vector<int64_t> &dst_shape, std::vector<Expression> &dst_data) {
  std::vector<int64_t> aligned_src = src_shape;
  while (aligned_src.size() < dst_shape.size()) {
    aligned_src.insert(aligned_src.begin(), 1);
  }
  if (aligned_src.size() != dst_shape.size()) {
    GELOGW("Cannot broadcast, aligned_size: %zu, dst_size: %zu", aligned_src.size(), dst_shape.size());
    return UNSUPPORTED;
  }
  for (size_t i = 0UL; i < dst_shape.size(); ++i) {
    if (aligned_src[i] != 1 && aligned_src[i] != dst_shape[i]) {
      GELOGW("Dim: %zu cannot broadcast, src_value: %lld, dst_value: %lld", i, aligned_src[i], dst_shape[i]);
      return UNSUPPORTED;
    }
  }
  int64_t total_elements = 1L;
  for (auto dim : dst_shape) {
    total_elements *= dim;
  }
  if (static_cast<int64_t>(src_data.size()) == total_elements) {
    dst_data = src_data;
    return SUCCESS;
  }
  auto compute_strides = [](const std::vector<int64_t> &shape) {
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; --i) {
      strides[static_cast<size_t>(i)] = strides[static_cast<size_t>(i + 1)] * shape[static_cast<size_t>(i + 1)];
    }
    return strides;
  };
  std::vector<int64_t> src_strides = compute_strides(aligned_src);
  std::vector<int64_t> dst_strides = compute_strides(dst_shape);
  dst_data.reserve(static_cast<size_t>(total_elements));
  for (int64_t pos = 0L; pos < total_elements; ++pos) {
    int64_t src_linear = 0L;
    for (size_t i = 0UL; i < dst_shape.size(); ++i) {
      int64_t dim = aligned_src[i];
      int64_t idx = (pos / dst_strides[i]) % dst_shape[i];
      src_linear += (idx % dim) * src_strides[i];
    }
    dst_data.push_back(src_data[static_cast<size_t>(src_linear)]);
  }
  return SUCCESS;
}

graphStatus SetOutputShapeAndValue(gert::InferSymbolComputeContext *context, const std::vector<int64_t> &output_dims,
                                   std::vector<Expression> &&output_values) {
  auto output_tensor = context->GetOutputSymbolTensor(kOutputIndex);
  GE_ASSERT_NOTNULL(output_tensor);
  std::vector<Expression> output_shape_symbols;
  for (auto dim : output_dims) {
    output_shape_symbols.emplace_back(Symbol(dim));
  }
  output_tensor->MutableOriginSymbolShape().MutableDims() = output_shape_symbols;
  auto output_values_unique = ge::MakeUnique<std::vector<ge::Expression>>(std::move(output_values));
  if (output_values_unique != nullptr) {
    output_tensor->SetSymbolicValue(std::move(output_values_unique));
  }
  GELOGD("%s[%s] kernel success, %s", context->GetNodeName(), context->GetNodeType(),
         SymbolicInferUtil::DumpSymbolTensor(*output_tensor).c_str());
  return SUCCESS;
}

graphStatus ComputeBroadcastShape(const gert::InferSymbolComputeContext *context, const std::vector<int64_t> &x1_dims,
                                  const std::vector<int64_t> &x2_dims, std::vector<int64_t> &output_dims) {
  size_t max_rank = std::max(x1_dims.size(), x2_dims.size());
  std::vector<int64_t> aligned_x1 = x1_dims;
  std::vector<int64_t> aligned_x2 = x2_dims;
  while (aligned_x1.size() < max_rank) {
    aligned_x1.insert(aligned_x1.begin(), 1);
  }
  while (aligned_x2.size() < max_rank) {
    aligned_x2.insert(aligned_x2.begin(), 1);
  }
  output_dims.clear();
  for (size_t i = 0UL; i < max_rank; ++i) {
    if (aligned_x1[i] == aligned_x2[i]) {
      output_dims.push_back(aligned_x1[i]);
    } else if (aligned_x1[i] == 1) {
      output_dims.push_back(aligned_x2[i]);
    } else if (aligned_x2[i] == 1) {
      output_dims.push_back(aligned_x1[i]);
    } else {
      GELOGW("Add broadcast failed, x1[%zu]=%lld, x2[%zu]=%lld, node %s[%s].", i, aligned_x1[i], i, aligned_x2[i],
             context->GetNodeName(), context->GetNodeType());
      return UNSUPPORTED;
    }
  }
  return SUCCESS;
}

template <typename BinaryOp>
graphStatus BinaryElementwiseSymbolicKernelCompute(gert::InferSymbolComputeContext *context, BinaryOp binary_op) {
  GE_ASSERT_NOTNULL(context);
  std::vector<int64_t> x1_dims;
  if (!context->GetConstInputDims(kX1InputIndex, x1_dims)) {
    GELOGW("SymbolicKernel compute unsupported, reason: get x1 const input dim failed, node %s[%s].",
           context->GetNodeName(), context->GetNodeType());
    return UNSUPPORTED;
  }
  std::vector<int64_t> x2_dims;
  if (!context->GetConstInputDims(kX2InputIndex, x2_dims)) {
    GELOGW("SymbolicKernel compute unsupported, reason: get x2 const input dim failed, node %s[%s].",
           context->GetNodeName(), context->GetNodeType());
    return UNSUPPORTED;
  }
  auto x1_tensor = context->GetInputSymbolTensor(kX1InputIndex);
  GE_UNSUPPORTED_IF_NULL(x1_tensor);
  auto x1_symbols = x1_tensor->GetSymbolicValue();
  if (x1_symbols == nullptr) {
    GELOGW("SymbolicKernel compute unsupported, reason: get x1 symbolic value failed, node %s[%s].",
           context->GetNodeName(), context->GetNodeType());
    return UNSUPPORTED;
  }
  auto x2_tensor = context->GetInputSymbolTensor(kX2InputIndex);
  GE_UNSUPPORTED_IF_NULL(x2_tensor);
  auto x2_symbols = x2_tensor->GetSymbolicValue();
  if (x2_symbols == nullptr) {
    GELOGW("SymbolicKernel compute unsupported, reason: get x2 symbolic value failed, node %s[%s].",
           context->GetNodeName(), context->GetNodeType());
    return UNSUPPORTED;
  }
  std::vector<int64_t> output_dims;
  GE_ASSERT_SUCCESS(ComputeBroadcastShape(context, x1_dims, x2_dims, output_dims));
  std::vector<Expression> x1_broadcast;
  GE_ASSERT_SUCCESS(BroadcastData(*x1_symbols, x1_dims, output_dims, x1_broadcast));
  std::vector<Expression> x2_broadcast;
  GE_ASSERT_SUCCESS(BroadcastData(*x2_symbols, x2_dims, output_dims, x2_broadcast));
  GE_ASSERT_TRUE(x1_broadcast.size() == x2_broadcast.size(), "broadcast size mismatch: %zu vs %zu", x1_broadcast.size(),
                 x2_broadcast.size());
  std::vector<Expression> output_values;
  output_values.reserve(x1_broadcast.size());
  for (size_t i = 0UL; i < x1_broadcast.size(); ++i) {
    output_values.push_back(binary_op(x1_broadcast[i], x2_broadcast[i]));
  }
  return SetOutputShapeAndValue(context, output_dims, std::move(output_values));
}
}  // namespace

/**
 * Add算子的符号化计算
 * 【算子功能】对两个输入张量执行逐元素加法。
 * 【推导逻辑】先获取x1和x2的确定Shape及SymbolicValue，按尾部维度补1并逐维校验广播兼容性，生成输出
 *          Shape；然后根据输入/输出stride将两个SymbolicValue映射到输出线性位置，在每个位置构造x1+x2
 *          表达式，最后写入输出Shape和SymbolicValue。
 * 【算子约束】输入Shape必须可确定并满足广播规则，两个输入必须存在且长度匹配的SymbolicValue。
 * 【举例】x1 Shape=[2,2]、value=[1,2,3,4]，x2 Shape=[1,2]、value=[10,20]时，
 *          x2广播后输出Shape为[2,2]，输出value为[11,22,13,24]。
 */
graphStatus AddSymbolicKernelCompute(gert::InferSymbolComputeContext *context) {
  GELOGD("Add Symbolic Kernel in, node %s[%s].", context->GetNodeName(), context->GetNodeType());
  return BinaryElementwiseSymbolicKernelCompute(context,
                                                [](const Expression &x1, const Expression &x2) { return x1 + x2; });
}

/**
 * Sub算子的符号化计算
 * 【算子功能】对两个输入张量执行逐元素减法。
 * 【推导逻辑】先获取两个输入Shape并按尾部维度执行广播校验，再依据stride展开各输入的SymbolicValue；
 *          对广播后的每个线性位置构造x1-x2表达式，保持输出布局与广播Shape一致，最后写回输出结果。
 * 【算子约束】输入Shape必须可确定并满足广播规则，两个输入必须存在且长度匹配的SymbolicValue。
 * 【举例】x1 Shape=[2,2]、value=[10,20,30,40]，x2 Shape=[1,2]、value=[1,2]时，
 *          输出Shape为[2,2]，输出value为[9,18,29,38]。
 */
graphStatus SubSymbolicKernelCompute(gert::InferSymbolComputeContext *context) {
  GELOGD("Sub Symbolic Kernel in, node %s[%s].", context->GetNodeName(), context->GetNodeType());
  return BinaryElementwiseSymbolicKernelCompute(context,
                                                [](const Expression &x1, const Expression &x2) { return x1 - x2; });
}

/**
 * Mul算子的符号化计算
 * 【算子功能】对两个输入张量执行逐元素乘法。
 * 【推导逻辑】先获取两个输入Shape并按尾部维度执行广播校验，再依据stride展开各输入的SymbolicValue；
 *          对广播后的每个线性位置构造x1*x2表达式，保持输出布局与广播Shape一致，最后写回输出结果。
 * 【算子约束】输入Shape必须可确定并满足广播规则，两个输入必须存在且长度匹配的SymbolicValue。
 * 【举例】x1 Shape=[2,2]、value=[2,3,4,5]，x2 Shape=[1,2]、value=[10,20]时，
 *          输出Shape为[2,2]，输出value为[20,60,40,100]。
 */
graphStatus MulSymbolicKernelCompute(gert::InferSymbolComputeContext *context) {
  GELOGD("Mul Symbolic Kernel in, node %s[%s].", context->GetNodeName(), context->GetNodeType());
  return BinaryElementwiseSymbolicKernelCompute(context,
                                                [](const Expression &x1, const Expression &x2) { return x1 * x2; });
}

REGISTER_SYMBOLIC_KERNEL(Add, AddSymbolicKernelCompute);
REGISTER_SYMBOLIC_KERNEL(Sub, SubSymbolicKernelCompute);
REGISTER_SYMBOLIC_KERNEL(Mul, MulSymbolicKernelCompute);
}  // namespace ge
