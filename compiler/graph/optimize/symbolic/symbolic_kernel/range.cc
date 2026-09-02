/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE. See
 * LICENSE in the root of the software repository for the full text of the License.
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
constexpr size_t kRangeInputNum = 3U;
constexpr size_t kRangeOutputNum = 1U;
constexpr size_t kStartIndex = 0U;
constexpr size_t kLimitIndex = 1U;
constexpr size_t kDeltaIndex = 2U;
constexpr int64_t kMaxSymbolicValueSize = 200L;
const std::set<DataType> kRangeSupportedTypes = {DT_INT32, DT_FLOAT, DT_DOUBLE, DT_INT64};

const Expression *GetScalarValue(gert::InferSymbolComputeContext *context, size_t index) {
  const auto input_tensor = context->GetInputSymbolTensor(index);
  if (input_tensor == nullptr) {
    return nullptr;
  }
  const auto values = input_tensor->GetSymbolicValue();
  if (values == nullptr || values->empty()) {
    return nullptr;
  }
  return &values->front();
}

}  // namespace

bool ResolveRangeValues(gert::InferSymbolComputeContext *context, Expression &start, Expression &limit,
                        Expression &delta) {
  const auto start_value = GetScalarValue(context, kStartIndex);
  const auto limit_value = GetScalarValue(context, kLimitIndex);
  const auto delta_value = GetScalarValue(context, kDeltaIndex);
  start = Expression(Symbol(0));
  limit = start;
  delta = Expression(Symbol(1));
  if (start_value != nullptr && limit_value != nullptr && delta_value != nullptr) {
    start = *start_value;
    limit = *limit_value;
    delta = *delta_value;
    return true;
  }
  if (start_value != nullptr && limit_value != nullptr && delta_value == nullptr) {
    start = *start_value;
    limit = *limit_value;
    return true;
  }
  if (start_value != nullptr && limit_value == nullptr && delta_value == nullptr) {
    limit = *start_value;
    return true;
  }
  return false;
}

graphStatus CheckRangeTypes(gert::InferSymbolComputeContext *context) {
  const auto start_desc = context->GetInputDesc(kStartIndex);
  const auto limit_desc = context->GetInputDesc(kLimitIndex);
  const auto delta_desc = context->GetInputDesc(kDeltaIndex);
  GE_UNSUPPORTED_IF_NULL(start_desc);
  GE_UNSUPPORTED_IF_NULL(limit_desc);
  GE_UNSUPPORTED_IF_NULL(delta_desc);
  if (start_desc->GetDataType() != limit_desc->GetDataType() ||
      start_desc->GetDataType() != delta_desc->GetDataType() ||
      kRangeSupportedTypes.count(start_desc->GetDataType()) == 0U) {
    GELOGW("Range symbolic compute unsupported: input data types are inconsistent or unsupported, node %s[%s].",
           context->GetNodeName(), context->GetNodeType());
    return UNSUPPORTED;
  }
  return GRAPH_SUCCESS;
}

/**
 * Range算子的符号化计算
 * 【算子功能】按照给定起点、终点和步长生成等差序列张量。
 * 【推导逻辑】读取start、limit、delta的常量或Hint值，校验delta非零并依据正负方向计算序列长度；输出
 *          Shape为一维序列长度，随后从start开始按delta递推生成每个元素的SymbolicValue并写入输出。
 * 【算子约束】输入值必须可用，序列长度必须是非负常量且不超过SymbolicValue物化上限。
 * 【举例】start、limit、delta的Shape均为[]，value分别为0、5、2时，输出Shape为[3]，
 *          输出value为[0,2,4]。
 */
graphStatus RangeSymbolicKernelCompute(gert::InferSymbolComputeContext *context) {
  GE_ASSERT_NOTNULL(context);
  GELOGD("Range Symbolic Kernel in, node %s[%s].", context->GetNodeName(), context->GetNodeType());
  GE_ASSERT(context->GetComputeNodeInputNum() == kRangeInputNum, "InputNum=%zu", context->GetComputeNodeInputNum());
  GE_ASSERT(context->GetComputeNodeOutputNum() == kRangeOutputNum, "OutputNum=%zu", context->GetComputeNodeOutputNum());

  Expression start_value;
  Expression limit_value;
  Expression delta_value;
  if (!ResolveRangeValues(context, start_value, limit_value, delta_value)) {
    GELOGW("Range symbolic compute unsupported: required start/limit values are missing, node %s[%s].",
           context->GetNodeName(), context->GetNodeType());
    return UNSUPPORTED;
  }

  const auto ret = CheckRangeTypes(context);
  if (ret != GRAPH_SUCCESS) {
    return ret;
  }

  const auto range_num = sym::Ceiling((limit_value - start_value) / delta_value);
  int64_t range_num_hint = 0L;
  if (!range_num.GetConstValue(range_num_hint) || range_num_hint < 0L || range_num_hint > kMaxSymbolicValueSize) {
    GELOGW("Range symbolic compute unsupported: output length is not a valid constant within the limit, node %s[%s].",
           context->GetNodeName(), context->GetNodeType());
    return UNSUPPORTED;
  }

  auto output_tensor = context->GetOutputSymbolTensor(0U);
  GE_ASSERT_NOTNULL(output_tensor);
  output_tensor->MutableOriginSymbolShape().MutableDims() = {range_num};
  std::vector<Expression> values;
  values.reserve(static_cast<size_t>(range_num_hint));
  for (int64_t i = 0L; i < range_num_hint; ++i) {
    values.emplace_back(start_value + delta_value * Symbol(i));
  }
  auto values_unique = ge::MakeUnique<std::vector<Expression>>(std::move(values));
  GE_ASSERT_NOTNULL(values_unique);
  output_tensor->SetSymbolicValue(std::move(values_unique));
  GELOGD("%s[%s] kernel success, %s", context->GetNodeName(), context->GetNodeType(),
         SymbolicInferUtil::DumpSymbolTensor(*output_tensor).c_str());
  return GRAPH_SUCCESS;
}

REGISTER_SYMBOLIC_KERNEL(Range, RangeSymbolicKernelCompute);
}  // namespace ge
