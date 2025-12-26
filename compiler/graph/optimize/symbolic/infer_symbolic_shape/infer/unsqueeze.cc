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
graphStatus InferShape4UnSqueeze(gert::InferSymbolShapeContext *context) {
  const auto in_shape = context->GetInputSymbolShape(0);
  GE_UNSUPPORTED_IF_NULL(in_shape);
  const auto out_shape = context->GetOutputSymbolShape(0);
  GE_ASSERT_NOTNULL(out_shape);
  const auto attrs = context->GetAttrs();
  GE_ASSERT_NOTNULL(attrs);
  const auto axes = attrs->GetListInt(0);
  GE_ASSERT_NOTNULL(axes);
  // 如果传入的维度为空，则输出shape和输入shape一致
  if (axes->GetSize() == 0) {
    *out_shape = *in_shape;
    return ge::GRAPH_SUCCESS;
  }
  const auto in_dim_num = in_shape->GetDimNum();
  const auto out_dim_num = in_dim_num + axes->GetSize();
  GE_ASSERT(out_dim_num <= gert::Shape::kMaxDimNum,
            "Unsqueeze failed, DimNum of output shape is %zu, larger than kMaxDimNum is %zu!", out_dim_num,
            gert::Shape::kMaxDimNum);
  out_shape->Clear();
  for (size_t i = 0; i < out_dim_num; i++) {
    out_shape->AppendDim(ge::Symbol(0));
  }
  const auto out_dim_num_signed = static_cast<int64_t>(out_dim_num);
  for (size_t i = 0; i < axes->GetSize(); i++) {
    const int64_t raw_axis = axes->GetData()[i];
    const int64_t real_axis = raw_axis >= 0 ? raw_axis : raw_axis + out_dim_num_signed;
    // 维度值范围校验
    GE_ASSERT(real_axis >= 0 && real_axis < out_dim_num_signed,
              "Unsqueeze failed, as axes val[%zu] is out of range[-%zu, %zu].", raw_axis, out_dim_num, out_dim_num);
    // 校验当前维度值是否已经为1，判断axis是否存在重复值
    GE_ASSERT(out_shape->GetDim(real_axis) != 1, "Unsqueeze failed, axis repeated");
    out_shape->MutableDim(real_axis) = ge::Symbol(1);
  }
  size_t in_index = 0;
  for (size_t i = 0; i < out_dim_num; i++) {
    if (out_shape->GetDim(i) != 1) {
      out_shape->MutableDim(i) = in_shape->GetDim(in_index++);
    }
  }
  return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFER_SYMBOL_SHAPE_INNER(Unsqueeze).InferSymbolShape(InferShape4UnSqueeze);
}  // namespace
}  // namespace ge