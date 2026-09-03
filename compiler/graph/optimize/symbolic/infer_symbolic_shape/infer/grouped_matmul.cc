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
#include "common/framework_types_internal.h"
#include "exe_graph/runtime/infer_symbol_shape_context.h"
#include "graph/optimize/symbolic/infer_symbolic_shape/symbolic_infer_util.h"

namespace ge {
namespace {
constexpr size_t kXIndex = 0UL;
constexpr size_t kWeightIndex = 1UL;
constexpr size_t kGroupListIndex = 7UL;
constexpr size_t kTransposeWeightAttr = 2UL;
constexpr size_t kTransposeXAttr = 3UL;
constexpr size_t kGroupTypeAttr = 4UL;
constexpr size_t kGroupListTypeAttr = 5UL;
constexpr int64_t kNoSplit = -1L;
constexpr int64_t kSplitM = 0L;
constexpr int64_t kSplitK = 2L;
constexpr int64_t kGroupListSparse = 2L;

const gert::SymbolShape *GetDynamicShape(gert::InferSymbolShapeContext *context, size_t index, size_t offset) {
  return context->GetDynamicInputSymbolShape(index, offset);
}

bool GetBoolAttr(gert::InferSymbolShapeContext *context, size_t index, bool default_value) {
  const auto attrs = context->GetAttrs();
  if (attrs == nullptr) {
    return default_value;
  }
  const auto value = attrs->GetAttrPointer<bool>(index);
  return value == nullptr ? default_value : *value;
}

int64_t GetIntAttr(gert::InferSymbolShapeContext *context, size_t index, int64_t default_value) {
  const auto attrs = context->GetAttrs();
  if (attrs == nullptr) {
    return default_value;
  }
  const auto value = attrs->GetAttrPointer<int64_t>(index);
  return value == nullptr ? default_value : *value;
}

Expression GetWeightN(const gert::SymbolShape &weight, bool transpose_weight) {
  const auto rank = weight.GetDimNum();
  return weight.GetDim(transpose_weight ? rank - 2UL : rank - 1UL);
}

Expression GetXLogicalM(const gert::SymbolShape &x, bool transpose_x) {
  const auto rank = x.GetDimNum();
  return x.GetDim(transpose_x ? rank - 1UL : rank - 2UL);
}

void BuildMatmulShape(const gert::SymbolShape &x, const gert::SymbolShape &weight, bool transpose_x,
                      bool transpose_weight, std::vector<Expression> &output) {
  output = x.GetDims();
  if (output.size() < 2UL || weight.GetDimNum() < 2UL) {
    output.clear();
    return;
  }
  output[output.size() - 1UL] = GetWeightN(weight, transpose_weight);
  if (transpose_x) {
    output[output.size() - 2UL] = x.GetDim(x.GetDimNum() - 1UL);
  }
}

const std::vector<Expression> *GetGroupListValue(const gert::InferSymbolShapeContext *context) {
  const auto group_list = context->GetOptionalInputSymbolTensor(kGroupListIndex);
  return group_list == nullptr ? nullptr : group_list->GetSymbolicValue();
}

graphStatus InferSplitMOutputShape(gert::InferSymbolShapeContext *context, size_t output_index,
                                   const gert::SymbolShape &weight, int64_t group_list_type,
                                   std::vector<Expression> &output) {
  const auto values = GetGroupListValue(context);
  if (values == nullptr) {
    GELOGW("GroupedMatmul symbolic shape inference unsupported: group_list value is missing, node %s[%s].",
           context->GetNodeName(), context->GetNodeType());
    return UNSUPPORTED;
  }
  const size_t value_index = group_list_type == kGroupListSparse ? output_index * 2UL + 1UL : output_index;
  if (value_index >= values->size()) {
    GELOGW(
        "GroupedMatmul symbolic shape inference unsupported: group_list index[%zu] exceeds value size[%zu], node "
        "%s[%s].",
        value_index, values->size(), context->GetNodeName(), context->GetNodeType());
    return UNSUPPORTED;
  }
  output = {(*values)[value_index], GetWeightN(weight, false)};
  return GRAPH_SUCCESS;
}
}  // namespace

size_t GetDynamicInputCount(gert::InferSymbolShapeContext *context, size_t index) {
  size_t count = 0UL;
  while (GetDynamicShape(context, index, count) != nullptr) {
    ++count;
  }
  return count;
}

graphStatus SetSplitKOutputs(gert::InferSymbolShapeContext *context, size_t output_count, size_t weight_count,
                             const gert::SymbolShape &x, const gert::SymbolShape &weight, bool transpose_x,
                             bool transpose_weight) {
  const auto x_m = GetXLogicalM(x, transpose_x);
  const auto group_list = context->GetOptionalInputSymbolShape(kGroupListIndex);
  const Expression group_num = weight_count > 1UL ? Symbol(static_cast<int64_t>(weight_count))
                                                  : (group_list == nullptr ? Symbol(1) : group_list->GetDim(0UL));
  for (size_t i = 0UL; i < output_count; ++i) {
    auto output = context->GetOutputSymbolShape(i);
    GE_UNSUPPORTED_IF_NULL(output);
    output->MutableDims() = {group_num, x_m, GetWeightN(weight, transpose_weight)};
  }
  return GRAPH_SUCCESS;
}

/**
 * GroupedMatmul算子的符号化Shape推导
 * 【算子功能】对分组的x和weight执行矩阵乘，输出为一个或多个分组矩阵乘结果。
 * 【推导逻辑】先读取x和weight TensorList中每个有效Tensor的符号Shape，并依据transpose_x/transpose_w
 *          交换矩阵的最后两个维度；再根据group_type确定单组矩阵乘、分组矩阵乘或批维广播关系，
 *          校验K维匹配后拼接批维与M/N维得到矩阵乘输出Shape，最后将结果写入对应输出TensorList。
 *            K轴切分时增加group维，M轴切分且多输出时使用group_list的符号值推导各输出的M维。
 * 【算子约束】当前支持不切分、M轴切分和K轴切分的Shape推导，不参与量化参数的值计算。
 * 【举例】x=[M,K]、weight=[K,N]且group_type=-1时，输出为[M,N]；x=[M,K]、两个
 *          weight=[K,N]且group_type=2时，输出为[2,M,N]。
 */
graphStatus InferGroupedMatmulSymbolShape(gert::InferSymbolShapeContext *context) {
  GE_ASSERT_NOTNULL(context);
  const auto x0 = GetDynamicShape(context, kXIndex, 0UL);
  const auto weight0 = GetDynamicShape(context, kWeightIndex, 0UL);
  GE_UNSUPPORTED_IF_NULL(x0);
  GE_UNSUPPORTED_IF_NULL(weight0);

  const auto transpose_weight = GetBoolAttr(context, kTransposeWeightAttr, false);
  const auto transpose_x = GetBoolAttr(context, kTransposeXAttr, false);
  const auto group_type = GetIntAttr(context, kGroupTypeAttr, kNoSplit);
  const auto group_list_type = GetIntAttr(context, kGroupListTypeAttr, 0L);
  const size_t x_count = GetDynamicInputCount(context, kXIndex);
  const size_t weight_count = GetDynamicInputCount(context, kWeightIndex);
  const size_t output_count = context->GetComputeNodeOutputNum();

  if (group_type == kSplitK) {
    return SetSplitKOutputs(context, output_count, weight_count, *x0, *weight0, transpose_x, transpose_weight);
  }

  for (size_t i = 0UL; i < output_count; ++i) {
    auto output = context->GetOutputSymbolShape(i);
    GE_UNSUPPORTED_IF_NULL(output);
    const auto x = GetDynamicShape(context, kXIndex, x_count == 1UL ? 0UL : i);
    const auto weight = GetDynamicShape(context, kWeightIndex, weight_count == 1UL ? 0UL : i);
    GE_UNSUPPORTED_IF_NULL(x);
    GE_UNSUPPORTED_IF_NULL(weight);
    std::vector<Expression> output_dims;
    if (group_type == kSplitM && x_count == 1UL && output_count > 1UL) {
      const auto ret = InferSplitMOutputShape(context, i, *weight, group_list_type, output_dims);
      if (ret != GRAPH_SUCCESS) {
        return ret;
      }
    } else {
      BuildMatmulShape(*x, *weight, transpose_x, transpose_weight, output_dims);
      if (output_dims.empty()) {
        GELOGW(
            "GroupedMatmul symbolic shape inference unsupported: output Shape is empty, output index[%zu], node "
            "%s[%s].",
            i, context->GetNodeName(), context->GetNodeType());
        return UNSUPPORTED;
      }
    }
    output->MutableDims() = std::move(output_dims);
  }
  return GRAPH_SUCCESS;
}

IMPL_OP_INFER_SYMBOL_SHAPE_INNER(GroupedMatmul).InferSymbolShape(InferGroupedMatmulSymbolShape);
}  // namespace ge
