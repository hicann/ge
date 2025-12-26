/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */


#include <cstdlib>
#include "graph/compute_graph.h"
#include "exe_graph/runtime/infer_symbol_shape_context.h"
#include "common/checker.h"
#include "common/types.h"
#include "graph/optimize/symbolic/infer_symbolic_shape/symbolic_infer_util.h"

namespace ge {
namespace {
constexpr size_t kPadDimNum = 2U;
/**
 * Pad算子 算子的符号化shape推导
 * 【算子功能】对张量进行填充
 * 【算子约束】
 *         1. paddings入参是一个整数张量，形状为[n, 2]，其中n是输入tensor的秩
 * 【推导逻辑】
 *         1.输出shape等于输入shape + 在tensor前填充的数量 + 在tensor后填充的数量
 * 【举例】
 *      in_shape = [2, 3, 2]
 *      paddings = [[1, 2], [2, 1], [3, 3]]
 *      out_shape = [1+2+2, 2+3+1, 3+2+3]
 */
graphStatus InferShape4Pad(gert::InferSymbolShapeContext *context) {
  const auto in_shape = context->GetInputSymbolShape(0);
  GE_UNSUPPORTED_IF_NULL(in_shape);

  // paddings不能为空
  const auto paddings_tensor = context->GetInputSymbolTensor(1);
  GE_UNSUPPORTED_IF_NULL(paddings_tensor);
  if (paddings_tensor->GetSymbolicValue() == nullptr) {
    GELOGW("Symbol Infer unsupported, get symbolic value is nullptr, node %s[%s]", context->GetNodeName(), context->GetNodeType());
    return UNSUPPORTED;
  }

  const auto paddings_size = paddings_tensor->GetSymbolicValue()->size();
  GE_ASSERT(paddings_size > 0, "Invalid paddings, must be non-empty!");

  // paddings的shape必须是[inShapeDimNum,2]
  const auto paddings_shape = paddings_tensor->GetOriginSymbolShape();
  GE_ASSERT(paddings_shape.GetDim(0).IsConstExpr());
  int64_t paddings_dim_0 = -1;
  paddings_shape.GetDim(0).GetConstValue(paddings_dim_0);
  GE_ASSERT(paddings_dim_0 == static_cast<int64_t>(in_shape->GetDimNum()),
            "Paddings failed, as paddings dim0 %ld not equals to in shape dim num %u", paddings_dim_0,
            in_shape->GetDimNum());

  GE_ASSERT(paddings_shape.GetDim(1).IsConstExpr());
  int64_t paddings_dim_1 = -1;
  paddings_shape.GetDim(1).GetConstValue(paddings_dim_1);
  GE_ASSERT(paddings_dim_1 == kPadDimNum, "Padding failed, as paddings dim1 %ld not equals to 2", paddings_dim_1);

  const auto out_shape = context->GetOutputSymbolShape(0);
  GE_ASSERT_NOTNULL(out_shape);

  out_shape->Clear();
  auto paddings = *paddings_tensor->GetSymbolicValue();
  for (size_t i = 0; i < in_shape->GetDimNum(); ++i) {
    const auto dim = in_shape->GetDim(i) + paddings[kPadDimNum * i] + paddings[kPadDimNum * i + 1];
    out_shape->AppendDim(dim);
  }

  return GRAPH_SUCCESS;
}

graphStatus InferShape4PadD(gert::InferSymbolShapeContext *context) {
  const auto in_shape = context->GetInputSymbolShape(0);
  GE_UNSUPPORTED_IF_NULL(in_shape);
  const auto out_shape = context->GetOutputSymbolShape(0);
  GE_ASSERT_NOTNULL(out_shape);

  auto attrs = context->GetAttrs();
  GE_ASSERT_NOTNULL(attrs);
  const gert::ContinuousVectorVector *cvv_padding = nullptr;
  cvv_padding = attrs->GetListListInt(0);
  GE_ASSERT_NOTNULL(cvv_padding);

  out_shape->Clear();
  GE_ASSERT_EQ(cvv_padding->GetSize(), in_shape->GetDimNum());
  for (size_t i = 0U; i < cvv_padding->GetSize(); ++i) {
    const auto cv_padding = cvv_padding->Get(i);
    GE_ASSERT_NOTNULL(cv_padding);
    GE_ASSERT_EQ(cv_padding->GetSize(), 2U);  // 如果不为2是异常场景
    const int64_t *data = reinterpret_cast<const int64_t *>(cv_padding->GetData());
    GE_ASSERT_NOTNULL(data);
    const auto dim0 = *(data + 0U);
    const auto dim1 = *(data + 1U);
    auto sym0 = Symbol(dim0);
    auto sym1 = Symbol(dim1);
    const auto dim = in_shape->GetDim(i) + sym0 + sym1;
    out_shape->AppendDim(dim);
  }
  return GRAPH_SUCCESS;
}

IMPL_OP_INFER_SYMBOL_SHAPE_INNER(Pad).InferSymbolShape(InferShape4Pad);
IMPL_OP_INFER_SYMBOL_SHAPE_INNER(PadD).InferSymbolShape(InferShape4PadD);
}  // namespace
}  // namespace ge