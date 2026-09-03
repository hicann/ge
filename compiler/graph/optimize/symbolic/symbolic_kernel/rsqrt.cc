/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <set>

#include "common/checker.h"
#include "common/plugin/ge_make_unique_util.h"
#include "framework/common/framework_types_internal.h"
#include "graph/optimize/symbolic/infer_symbolic_shape/symbolic_infer_util.h"
#include "graph/optimize/symbolic/symbol_compute_context.h"
#include "graph/optimize/symbolic/symbolic_kernel_factory.h"

namespace ge {
namespace {
constexpr size_t kRsqrtInputNum = 1U;
constexpr size_t kRsqrtOutputNum = 1U;
constexpr size_t kInputIndex = 0U;
constexpr size_t kOutputIndex = 0U;
constexpr int32_t rsqrtDen = 2;
const std::set<DataType> kRsqrtSupportedTypes = {DT_FLOAT16, DT_FLOAT, DT_DOUBLE};

}  // namespace

/**
 * Rsqrt算子的符号化计算
 * 【算子功能】对输入张量逐元素计算倒数平方根。
 * 【推导逻辑】先校验输入dtype和SymbolicValue，再复制输入Shape；遍历每个输入表达式，添加大于0的约束，
 *          并通过Pow(x, -1/2)构造对应输出表达式，最后按输入顺序写回输出SymbolicValue。
 * 【算子约束】输入dtype必须受支持，输入SymbolicValue不能为空，且输入值必须大于0。
 * 【举例】输入Shape=[2]、输入value=[4,16]时，输出Shape为[2]，输出value为
 *          [Pow(4,-1/2),Pow(16,-1/2)]。
 */
graphStatus RsqrtSymbolicKernelCompute(gert::InferSymbolComputeContext *context) {
  GE_ASSERT_NOTNULL(context);
  GELOGD("Rsqrt Symbolic Kernel in, node %s[%s].", context->GetNodeName(), context->GetNodeType());
  GE_ASSERT(context->GetComputeNodeInputNum() == kRsqrtInputNum, "InputNum=%zu", context->GetComputeNodeInputNum());
  GE_ASSERT(context->GetComputeNodeOutputNum() == kRsqrtOutputNum, "OutputNum=%zu", context->GetComputeNodeOutputNum());

  const auto input_desc = context->GetInputDesc(kInputIndex);
  GE_UNSUPPORTED_IF_NULL(input_desc);
  if (kRsqrtSupportedTypes.count(input_desc->GetDataType()) == 0U) {
    GELOGW("Rsqrt symbolic compute unsupported, data type[%d], node %s[%s].",
           static_cast<int32_t>(input_desc->GetDataType()), context->GetNodeName(), context->GetNodeType());
    return UNSUPPORTED;
  }

  const auto input_tensor = context->GetInputSymbolTensor(kInputIndex);
  GE_UNSUPPORTED_IF_NULL(input_tensor);
  const auto input_values = input_tensor->GetSymbolicValue();
  GE_UNSUPPORTED_IF_NULL(input_values);
  if (input_values->empty()) {
    GELOGW("Rsqrt symbolic compute unsupported: input symbolic value is empty, node %s[%s].", context->GetNodeName(),
           context->GetNodeType());
    return UNSUPPORTED;
  }

  auto output_tensor = context->GetOutputSymbolTensor(kOutputIndex);
  GE_ASSERT_NOTNULL(output_tensor);
  output_tensor->MutableOriginSymbolShape() = input_tensor->GetOriginSymbolShape();
  auto output_values = output_tensor->MutableSymbolicValue();
  GE_ASSERT_NOTNULL(output_values);
  output_values->reserve(input_values->size());
  for (const auto &input_value : *input_values) {
    ASSERT_SYMBOL_GT(input_value, kSymbolZero);
    output_values->emplace_back(sym::Pow(input_value, sym::Rational(-1, rsqrtDen)));
  }

  GELOGD("%s[%s] kernel success, %s", context->GetNodeName(), context->GetNodeType(),
         SymbolicInferUtil::DumpSymbolTensor(*output_tensor).c_str());
  return GRAPH_SUCCESS;
}

REGISTER_SYMBOLIC_KERNEL(Rsqrt, RsqrtSymbolicKernelCompute);
}  // namespace ge
