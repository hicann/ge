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
#include "common/framework_types_internal.h"
#include "graph/optimize/symbolic/infer_symbolic_shape/symbolic_infer_util.h"

namespace ge {
namespace {
graphStatus GetConstInt(const Expression &expr, DataType dt, int64_t &value) {
  if (dt == DT_INT32) {
    int32_t tmp_value = 0;
    if (expr.GetConstValue<int32_t>(tmp_value) == false) {
      return UNSUPPORTED;
    }
    value = static_cast<int64_t>(tmp_value);
  } else if (dt == DT_INT64) {
    if (expr.GetConstValue<int64_t>(value) == false) {
      return UNSUPPORTED;
    }
  } else {
    GELOGE(PARAM_INVALID, "dt must in [int32, int64]");
    return ge::PARAM_INVALID;
  }
  return ge::GRAPH_SUCCESS;
}

// Reshape 尺寸约束收尾：无未知维度(-1)时登记总量一致 guard；有未知维度时按 hint
// 整除性求解（见 ResolveIntegralDim）并回填该维度
graphStatus FinalizeReshapeSize(const gert::InferSymbolShapeContext *context, const gert::SymbolShape *in_shape,
                                gert::SymbolShape *out_shape, size_t unknown_dim_idx) {
  Expression in_shape_size = in_shape->GetSymbolShapeSize();
  Expression out_shape_size = out_shape->GetSymbolShapeSize();
  if (unknown_dim_idx == std::numeric_limits<size_t>::max()) {
    // 添加guard out_shape_size == in_shape_size
    ASSERT_SYMBOL_EQ(in_shape_size, out_shape_size);
    return ge::GRAPH_SUCCESS;
  }
  Expression dynamic_dim;
  if (ResolveIntegralDim(in_shape_size, out_shape_size, dynamic_dim) != SUCCESS) {
    GELOGW("Symbol Infer unsupported, cannot infer integral unknown dimension, node %s[%s]", context->GetNodeName(),
           context->GetNodeType());
    return UNSUPPORTED;
  }
  out_shape->MutableDims()[unknown_dim_idx] = dynamic_dim;
  return ge::GRAPH_SUCCESS;
}

graphStatus ReshapeInferCommon(const gert::InferSymbolShapeContext *context, const gert::SymbolShape *in_shape,
                               gert::SymbolShape *out_shape, const gert::SymbolTensor *shape_tensor, DataType dt) {
  auto reshape_dim_num = shape_tensor->GetSymbolicValue()->size();
  size_t unknown_dim_idx = std::numeric_limits<size_t>::max();
  // expr可能是常量或者符号
  for (size_t i = 0; i < reshape_dim_num; i++) {
    auto dim_expr = shape_tensor->GetSymbolicValue()->at(i);
    // 常量维度直接取值；非常量维度尝试按 hint 取值（运行时真实数据），hint 不可得
    // 时保留符号维度原样传播（合法：符号维度本就可存在于 shape 中，不构成值污染）。
    int64_t dim = -2;
    bool has_dim_value = false;
    if (dim_expr.IsConstExpr()) {
      // 如果dim是常量，只能是int32或者int64类型
      if (GetConstInt(dim_expr, dt, dim) == UNSUPPORTED) {
        GELOGW("Symbol Infer unsupported, get dim at index[%zu] is not constvalue, node %s[%s]", i,
               context->GetNodeName(), context->GetNodeType());
        return UNSUPPORTED;
      }
      has_dim_value = true;
    } else if (dim_expr.GetHint(dim)) {
      has_dim_value = true;
    }
    if (!has_dim_value) {
      out_shape->AppendDim(dim_expr);
      continue;
    }
    if (dim == 0) {
      // 输入为0表示使用输入的维度；hint 判定为 0 时登记 guard 固化该假设
      if (!dim_expr.IsConstExpr()) {
        (void)EXPECT_SYMBOL_EQ(dim_expr, kSymbolZero);
      }
      GE_ASSERT_TRUE(i < in_shape->GetDimNum());
      out_shape->AppendDim(in_shape->GetDim(i));
    } else if (dim == -1) {
      // 输入为-1表示该维度不确定需要等其它维度确定后最后计算，先用1占位，并记录该维度。
      // 多个 -1 属算子语义非法（兼容 TF 约定 at most one -1：两维未知时总元素量方程
      // 欠定无唯一解），直接报错
      GE_ASSERT_TRUE(unknown_dim_idx == std::numeric_limits<size_t>::max(),
                     "Reshape symbolic infer: more than one -1 in shape, node %s[%s].", context->GetNodeName(),
                     context->GetNodeType());
      // hint 来源的 -1 必须登记 guard：运行时 shape 输入变为其它合法排列（如 [-1,3] 变
      // [2,6]，总元素数不变）时，总元素量约束仍成立，仅靠它无法拦截，需要 Eq(dim, -1)
      // 假设校验命中旧编译结果
      if (!dim_expr.IsConstExpr()) {
        (void)EXPECT_SYMBOL_EQ(dim_expr, Symbol(-1));
      }
      out_shape->AppendDim(Symbol(1));
      unknown_dim_idx = i;
    } else {
      out_shape->AppendDim(dim_expr);
    }
  }
  return FinalizeReshapeSize(context, in_shape, out_shape, unknown_dim_idx);
}

/**
 * Reshape算子符号化推导，该算子能改变输入数据的形状，根据指定的形状参数对输入的数据进行重排列，输出重新排列后的数据。
 * data：输入数据
 * shape：一个列表，用来定义输出数据的维度，0表示该维度跟输入数据一致，-1表示剩下的数据放入该维度
 * 例如
 * data: {2, 3, 4}
 * shape [1, 0, -1, 2]
 * output:
 * dim1 = 1
 * dim2 = 3
 * dim4 = 2
 * dim3 = (2 * 3 * 4) / (1 * 3 * 2) = 4
 * {1, 3, 4, 2}
 */
graphStatus InferShape4Reshape(gert::InferSymbolShapeContext *context) {
  auto in_shape = context->GetInputSymbolShape(0);
  GE_UNSUPPORTED_IF_NULL(in_shape);
  auto shape_tensor = context->GetInputSymbolTensor(1);
  GE_UNSUPPORTED_IF_NULL(shape_tensor);
  auto out_shape = context->GetOutputSymbolShape(0);
  GE_ASSERT_NOTNULL(out_shape);
  // 输入为空列表不需要进行reshape
  if (shape_tensor->GetSymbolicValue() == nullptr || shape_tensor->GetSymbolicValue()->empty()) {
    GELOGW("Symbol Infer unsupported, get symbolic value is nullptr or empty, node %s[%s]", context->GetNodeName(),
           context->GetNodeType());
    return UNSUPPORTED;
  }
  auto shape_desc = context->GetInputDesc(1);
  GE_ASSERT_NOTNULL(shape_desc);
  auto dt = shape_desc->GetDataType();
  return ReshapeInferCommon(context, in_shape, out_shape, shape_tensor, dt);
}

/**
 * Shape算子的符号化Shape推导
 * 【算子功能】获取输入张量的秩，并将秩表示为一个一维Shape张量的长度。
 * 【算子约束】输入必须存在有效的符号Shape；输出为单个一维张量。
 * 【推导逻辑】读取输入符号Shape的维度数量R，创建一个一维输出符号Shape并将其唯一维度设为R；
 *          该推导只处理Shape，不展开输入Tensor的元素值，输出SymbolicValue由对应符号计算阶段生成。
 * 【举例】输入Shape=[s0,s1,s2]时，输出Shape为[3]。
 */
graphStatus InferShape4Shape(gert::InferSymbolShapeContext *context) {
  const auto in_shape = context->GetInputSymbolShape(0);
  GE_UNSUPPORTED_IF_NULL(in_shape);
  const auto out_shape = context->GetOutputSymbolShape(0);
  GE_ASSERT_NOTNULL(out_shape);

  out_shape->Clear();
  out_shape->AppendDim(Symbol(static_cast<int64_t>(in_shape->GetDimNum())));
  return GRAPH_SUCCESS;
}

IMPL_OP_INFER_SYMBOL_SHAPE_INNER(Shape).InferSymbolShape(InferShape4Shape);
IMPL_OP_INFER_SYMBOL_SHAPE_INNER(Reshape).InferSymbolShape(InferShape4Reshape);
}  // namespace
}  // namespace ge
