/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
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
#include "common/framework_types_internal.h"
#include "graph/optimize/symbolic/infer_symbolic_shape/symbolic_infer_util.h"

namespace ge {
namespace {
/**
 * Flatten算子的符号化Shape推导
 * 【算子功能】将输入张量从指定轴开始的多个维度展平为两个维度。
 * 【算子约束】axis必须位于输入秩允许的范围内；标量输入仅支持axis=0。
 * 【推导逻辑】先将负axis转换为非负轴并校验其位于[0, rank)；遍历axis左侧维度并按顺序相乘形成
 *          输出第0维，再遍历axis及右侧维度相乘形成输出第1维。乘法直接作用于符号表达式，因此动态
 *          维度会保留为乘积表达式；标量按两个1维处理。
 * 【举例】输入Shape=[B,S,H,W]、axis=1时，输出Shape=[B,S*H*W]；输入Shape=[2,3,4]、axis=-1时，
 *          输出Shape=[6,4]。
 */
graphStatus InferShape4Flatten(gert::InferSymbolShapeContext *context) {
  const auto in_shape = context->GetInputSymbolShape(0);
  GE_UNSUPPORTED_IF_NULL(in_shape);
  const auto out_shape = context->GetOutputSymbolShape(0);
  GE_ASSERT_NOTNULL(out_shape);
  const auto attrs = context->GetAttrs();
  GE_ASSERT_NOTNULL(attrs);
  const auto axis_ptr = attrs->GetInt(0);
  GE_ASSERT_NOTNULL(axis_ptr);

  const int64_t dim_num = static_cast<int64_t>(in_shape->GetDimNum());
  const int64_t axis = *axis_ptr;
  const int64_t real_axis = axis >= 0 ? axis : axis + dim_num;
  if (dim_num == 0) {
    GE_ASSERT(axis == 0, "Flatten failed, scalar input only supports axis=0, but got axis[%ld]. node %s[%s]", axis,
              context->GetNodeName(), context->GetNodeType());
    out_shape->Clear();
    out_shape->AppendDim(Symbol(1));
    out_shape->AppendDim(Symbol(1));
    return GRAPH_SUCCESS;
  }

  GE_ASSERT(real_axis >= 0 && real_axis < dim_num, "Flatten failed, axis[%ld] is out of range[-%ld, %ld]. node %s[%s]",
            axis, dim_num, dim_num - 1, context->GetNodeName(), context->GetNodeType());

  auto dim0 = Expression(Symbol(1));
  for (int64_t i = 0; i < real_axis; ++i) {
    dim0 = dim0 * in_shape->GetDim(i);
  }
  auto dim1 = Expression(Symbol(1));
  for (int64_t i = real_axis; i < dim_num; ++i) {
    dim1 = dim1 * in_shape->GetDim(i);
  }

  out_shape->Clear();
  out_shape->AppendDim(dim0);
  out_shape->AppendDim(dim1);
  return GRAPH_SUCCESS;
}

graphStatus InferShape4FlattenV2(gert::InferSymbolShapeContext *context) {
  auto const in_shape = context->GetInputSymbolShape(0);
  GE_UNSUPPORTED_IF_NULL(in_shape);
  auto const out_shape = context->GetOutputSymbolShape(0);
  GE_ASSERT_NOTNULL(out_shape);
  const auto attrs = context->GetAttrs();
  GE_ASSERT_NOTNULL(attrs);
  const auto attr_axis_ptr = attrs->GetInt(0);
  GE_ASSERT_NOTNULL(attr_axis_ptr);
  const auto attr_end_axis_ptr = attrs->GetInt(1);
  GE_ASSERT_NOTNULL(attr_end_axis_ptr);

  const int64_t axis = *attr_axis_ptr;
  const int64_t end_axis = *attr_end_axis_ptr;
  GELOGD("FlattenV2: axis=%ld, end_axis=%ld. node %s[%s]", axis, end_axis, context->GetNodeName(),
         context->GetNodeType());

  const int64_t dim_num = static_cast<int64_t>(in_shape->GetDimNum());
  const int64_t real_axis = axis >= 0 ? axis : axis + dim_num;
  const int64_t real_end_axis = end_axis >= 0 ? end_axis : end_axis + dim_num;

  GE_ASSERT(real_axis >= 0 && real_axis < dim_num,
            "FlattenV2 failed, as axes val[%ld] is out of range[-%ld, %ld]. node %s[%s]", real_axis, dim_num, dim_num,
            context->GetNodeName(), context->GetNodeType());

  GE_ASSERT(real_axis <= real_end_axis,
            "FlattenV2 failed, as axes val[%ld] must be less than or equal to end_axes val[%ld]. node %s[%s]",
            real_axis, real_end_axis, context->GetNodeName(), context->GetNodeType());

  GE_ASSERT(real_end_axis >= 0 && real_end_axis < dim_num,
            "FlattenV2 failed, as end_axes val[%ld] is out of range[-%ld, %ld]. node %s[%s]", real_end_axis, dim_num,
            dim_num, context->GetNodeName(), context->GetNodeType());

  out_shape->Clear();
  auto product = Expression(Symbol(1));
  for (int64_t i = 0; i < dim_num; i++) {
    if (i >= real_axis && i <= real_end_axis) {
      product = product * in_shape->GetDim(i);
      if (i == real_end_axis) {
        out_shape->AppendDim(product);
      }
    } else {
      out_shape->AppendDim(in_shape->GetDim(i));
    }
  }
  return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFER_SYMBOL_SHAPE_INNER(Flatten).InferSymbolShape(InferShape4Flatten);
IMPL_OP_INFER_SYMBOL_SHAPE_INNER(FlattenV2).InferSymbolShape(InferShape4FlattenV2);
}  // namespace
}  // namespace ge
