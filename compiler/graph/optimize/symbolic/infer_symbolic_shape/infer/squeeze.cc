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
graphStatus InferShape4Squeeze(gert::InferSymbolShapeContext *context) {
  const auto in_shape = context->GetInputSymbolShape(0);
  GE_UNSUPPORTED_IF_NULL(in_shape);
  const auto out_shape = context->GetOutputSymbolShape(0);
  GE_ASSERT_NOTNULL(out_shape);
  const auto attrs = context->GetAttrs();
  GE_ASSERT_NOTNULL(attrs);
  const auto axes = attrs->GetListInt(0);
  GE_ASSERT_NOTNULL(axes);
  if (axes->GetSize() == 0) {
    // 传入的轴为空，压缩所有维度等于的1维度
    out_shape->Clear();
    for (size_t i = 0; i < in_shape->GetDimNum(); i++) {
      if (EXPECT_SYMBOL_NE(in_shape->GetDim(i), Symbol(1))) {
        out_shape->AppendDim(in_shape->GetDim(i));
      }
    }
  } else {
    // 传入的轴不为空，根据axis入参压缩维度值等于1的维度
    bool squeeze_dims[gert::Shape::kMaxDimNum] = {false};
    const auto dim_num = static_cast<int64_t>(in_shape->GetDimNum());
    for (size_t i = 0; i < axes->GetSize(); i++) {
      const int64_t raw_axis = axes->GetData()[i];
      const int64_t real_axis = raw_axis >= 0 ? raw_axis : raw_axis + dim_num;
      GE_ASSERT(real_axis >= 0 && real_axis < dim_num, "Squeeze failed, as axes val[%ld] is out of range[-%ld, %ld].",
                raw_axis, dim_num, dim_num);
      squeeze_dims[real_axis] = true;
    }
    out_shape->Clear();
    for (size_t i = 0; i < in_shape->GetDimNum(); i++) {
      if (!squeeze_dims[i]) {
        out_shape->AppendDim(in_shape->GetDim(i));
      } else {
        const auto dim = in_shape->GetDim(i);
        ASSERT_SYMBOL_EQ(dim, Symbol(1));
      }
    }
  }
  return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFER_SYMBOL_SHAPE_INNER(Squeeze).InferSymbolShape(InferShape4Squeeze);
}  // namespace
}  // namespace ge