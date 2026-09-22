/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "common/checker.h"
#include "common/framework_types_internal.h"
#include "graph/compute_graph.h"
#include "graph/optimize/symbolic/infer_symbolic_shape/symbolic_infer_util.h"
#include "exe_graph/runtime/infer_symbol_shape_context.h"

namespace ge {
namespace {

/**
 * ArgMaxWithValue 和 ArgMinWithValue 算子的符号化shape推导
 * 【算子功能】在指定维度上找到最大值(ArgMaxWithValue)或最小值(ArgMinWithValue)及其索引
 *            输出两个张量：索引张量(indices)和值张量(values)
 * 【算子约束】
 *      1. dimension参数指定操作维度，范围为[-dim_num, dim_num)
 *      2. keep_dims参数决定是否保留操作维度
 *      举例：输入=(2,3,4)，dimension=1，keep_dims=true，则输出=(2,1,4)
 *            输入=(2,3,4)，dimension=1，keep_dims=false，则输出=(2,4)
 * 【推导逻辑】
 *      1. 获取输入shape和属性参数(dimension, keep_dims)
 *      2. 处理负数维度，转换为正数索引
 *      3. 遍历输入shape的每个维度：
 *         - 如果是操作维度且keep_dims=true，输出维度大小为1
 *         - 如果是操作维度且keep_dims=false，移除该维度
 *         - 其他维度保持不变
 *      4. 索引张量和值张量具有相同的输出shape
 *      举例：输入shape=(2,3,4)，dimension=-2，keep_dims=false，则输出=(2,4)
 */
graphStatus InferShape4ArgWithValue(gert::InferSymbolShapeContext *context) {
  // 获取输入形状
  auto shape_x = context->GetInputSymbolShape(0);
  GE_UNSUPPORTED_IF_NULL(shape_x);
  auto dim_num_x = shape_x->GetDimNum();

  // 获取属性
  auto attrs = context->GetAttrs();
  GE_ASSERT_NOTNULL(attrs);
  const auto dimension = attrs->GetAttrPointer<int64_t>(0);
  GE_ASSERT_NOTNULL(dimension);
  const auto keep_dims = attrs->GetAttrPointer<bool>(1);
  GE_ASSERT_NOTNULL(keep_dims);

  // 确定操作维度
  int64_t actual_dim = *dimension;
  if (actual_dim < 0) {
    actual_dim = dim_num_x + actual_dim;
  }
  GE_ASSERT_TRUE(actual_dim >= 0 && static_cast<size_t>(actual_dim) < dim_num_x,
                 "dimension %ld out of range for input with %zu dimensions", actual_dim, dim_num_x);

  // 计算输出形状
  auto shape_indice = context->GetOutputSymbolShape(0);
  GE_ASSERT_NOTNULL(shape_indice);
  auto shape_values = context->GetOutputSymbolShape(1);
  GE_ASSERT_NOTNULL(shape_values);

  // 构建输出形状
  for (size_t i = 0; i < dim_num_x; ++i) {
    if (i == static_cast<size_t>(actual_dim)) {
      if (*keep_dims) {
        // 保持维度，大小为1
        shape_indice->AppendDim(kSymbolOne);
        shape_values->AppendDim(kSymbolOne);
      }
      // 否则，移除该维度
    } else {
      // 保持原维度
      shape_indice->AppendDim(shape_x->GetDim(i));
      shape_values->AppendDim(shape_x->GetDim(i));
    }
  }

  return GRAPH_SUCCESS;
}

/**
 * ArgMaxV2 的符号 Shape 推导。
 * 【算子功能】在 dimension 指定的维度上查找最大值索引，并输出索引张量 y。
 * 【算子约束】dimension 必须是包含一个常量值的 int32 或 int64 数据依赖输入，取值范围为 [-rank, rank-1]；
 *             输入秩不大于1时输出标量。
 * 【推导逻辑】读取 x 的符号 Shape 和 dimension 的符号值，解析负轴后删除 x 中对应维度，其他维度保持原有Expression。
 * 【举例】x=[B,S,H]、dimension=-1 时，y=[B,S]；x=[B,S,H]、dimension=1 时，y=[B,H]。
 */
graphStatus InferShape4ArgMaxV2(gert::InferSymbolShapeContext *context) {
  const auto x_shape = context->GetInputSymbolShape(0);
  GE_UNSUPPORTED_IF_NULL(x_shape);
  const auto y_shape = context->GetOutputSymbolShape(0);
  GE_ASSERT_NOTNULL(y_shape);

  if (x_shape->GetDimNum() <= 1U) {
    y_shape->MutableDims().clear();
    return GRAPH_SUCCESS;
  }

  const auto dimension_tensor = context->GetInputSymbolTensor(1);
  GE_UNSUPPORTED_IF_NULL(dimension_tensor);
  const auto dimension_value = dimension_tensor->GetSymbolicValue();
  GE_UNSUPPORTED_IF_NULL(dimension_value);
  if (dimension_value->size() != 1U) {
    GELOGW("ArgMaxV2 dimension must contain exactly one value, actual size[%zu]", dimension_value->size());
    return ge::UNSUPPORTED;
  }

  int64_t dimension = 0;
  GE_ASSERT_TRUE(dimension_value->at(0).GetConstValue<int64_t>(dimension),
                 "ArgMaxV2 dimension must be an int32 or int64 constant");
  return SymbolicInferUtil::ReduceDimsWithoutKeepDims<int64_t>(x_shape, {dimension}, 1U, y_shape);
}

IMPL_OP_INFER_SYMBOL_SHAPE_INNER(ArgMaxWithValue).InferSymbolShape(InferShape4ArgWithValue);
IMPL_OP_INFER_SYMBOL_SHAPE_INNER(ArgMinWithValue).InferSymbolShape(InferShape4ArgWithValue);
IMPL_OP_INFER_SYMBOL_SHAPE_INNER(ArgMaxV2).InferSymbolShape(InferShape4ArgMaxV2);

}  // namespace
}  // namespace ge
