/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <limits>

#include "common/checker.h"
#include "common/plugin/ge_make_unique_util.h"
#include "framework/common/framework_types_internal.h"
#include "graph/optimize/symbolic/infer_symbolic_shape/symbolic_infer_util.h"
#include "graph/optimize/symbolic/symbol_compute_context.h"
#include "graph/optimize/symbolic/symbolic_kernel_factory.h"
#include "graph_metadef/common/ge_common/util.h"
#include "common/util/mem_utils.h"

namespace ge {
namespace {
constexpr size_t kReshapeInputNum = 2U;
constexpr size_t kReshapeOutputNum = 1U;
constexpr size_t kDataIndex = 0U;
constexpr size_t kShapeIndex = 1U;
constexpr size_t kShapeInputOutputSize = 1U;

bool GetShapeValue(const Expression &expr, DataType dtype, int64_t &value) {
  if (dtype == DT_INT32) {
    int32_t value32 = 0;
    if (!expr.GetConstValue(value32)) {
      return false;
    }
    value = value32;
    return true;
  }
  if (dtype == DT_INT64) {
    return expr.GetConstValue(value);
  }
  return false;
}

// 常量维度直接取值；非常量维度按 hint（运行时真实数据）取值并标记 is_hint_value，
// 供调用方对 hint 判定结果登记 guard。hint 不可得返回 UNSUPPORTED 由调用方回退。
graphStatus GetShapeValueWithHint(const Expression &expr, DataType dtype, int64_t &value, bool &is_hint_value) {
  is_hint_value = false;
  if (GetShapeValue(expr, dtype, value)) {
    return GRAPH_SUCCESS;
  }
  if (!expr.GetHint(value)) {
    return UNSUPPORTED;
  }
  is_hint_value = true;
  return GRAPH_SUCCESS;
}

}  // namespace

// Reshape 未知维度收尾：无未知维度(-1)时登记总量一致 guard；有未知维度时按 hint
// 整除性求解（见 ResolveIntegralDim）并回填
graphStatus FinalizeReshapeDims(const gert::InferSymbolComputeContext *context, const Expression &input_size,
                                const Expression &known, const size_t unknown, std::vector<Expression> &dims) {
  if (unknown == std::numeric_limits<size_t>::max()) {
    ASSERT_SYMBOL_EQ(input_size, known);
    return GRAPH_SUCCESS;
  }
  Expression dynamic_dim;
  if (ResolveIntegralDim(input_size, known, dynamic_dim) != SUCCESS) {
    GELOGW("Reshape symbolic compute unsupported: cannot infer integral unknown dimension, node %s[%s].",
           context->GetNodeName(), context->GetNodeType());
    return UNSUPPORTED;
  }
  dims[unknown] = dynamic_dim;
  return GRAPH_SUCCESS;
}

graphStatus BuildReshapeDims(gert::InferSymbolComputeContext *context, const gert::SymbolTensor *shape_tensor,
                             std::vector<Expression> &dims) {
  const auto values = shape_tensor->GetSymbolicValue();
  const auto desc = context->GetInputDesc(kShapeIndex);
  GE_UNSUPPORTED_IF_NULL(values);
  GE_UNSUPPORTED_IF_NULL(desc);
  if (values->empty()) {
    GELOGW("Reshape symbolic compute unsupported: shape symbolic value is empty, node %s[%s].", context->GetNodeName(),
           context->GetNodeType());
    return UNSUPPORTED;
  }
  const auto input_shape = context->GetInputSymbolTensor(kDataIndex)->GetOriginSymbolShape();
  size_t unknown = std::numeric_limits<size_t>::max();
  Expression known(Symbol(1));
  for (size_t i = 0U; i < values->size(); ++i) {
    int64_t dim = 0L;
    bool is_hint_value = false;
    const auto &dim_expr = values->at(i);
    const auto ret = GetShapeValueWithHint(dim_expr, desc->GetDataType(), dim, is_hint_value);
    if (ret != GRAPH_SUCCESS) {
      // hint 不可得的维度不再乘进 known 做符号除法（Rational 分数表达式在 symengine
      // 的 subs/replace 上存在缺陷），直接回退由上层走传统推导
      return ret;
    }
    // hint 来源的取值按分类登记假设 guard：0 为复制输入维度、正数为显式目标维度
    if (is_hint_value && dim >= 0L) {
      (void)(dim == 0L ? EXPECT_SYMBOL_EQ(dim_expr, kSymbolZero) : EXPECT_SYMBOL_GT(dim_expr, kSymbolZero));
    }
    if (dim == 0L && i < input_shape.GetDimNum()) {
      dims.emplace_back(input_shape.GetDim(i));
      known = known * input_shape.GetDim(i);
    } else if (dim > 0L) {
      dims.emplace_back(dim_expr);
      known = known * dim_expr;
    } else if (dim == -1L && unknown == std::numeric_limits<size_t>::max()) {
      // hint 来源的 -1 登记 guard：运行时 shape 变为其它等元素量排列时，仅靠
      // 总元素量约束无法拦截（同 infer 侧 ReshapeInferCommon）
      if (is_hint_value) {
        (void)EXPECT_SYMBOL_EQ(dim_expr, Symbol(-1));
      }
      unknown = i;
      dims.emplace_back(Symbol(1));
    } else {
      // 第二个及以后的 -1（算子语义非法，兼容 TF 约定 at most one -1）或非法维度值
      // （负数/0 越界），直接报错
      GE_ASSERT_TRUE(false, "Reshape symbolic compute: invalid reshape dimension %s, node %s[%s].",
                     dim_expr.Serialize().get(), context->GetNodeName(), context->GetNodeType());
    }
  }
  return FinalizeReshapeDims(context, input_shape.GetSymbolShapeSize(), known, unknown, dims);
}

/**
 * Reshape算子的符号化计算
 * 【算子功能】在不改变元素数量和排列顺序的前提下调整输入张量Shape。
 * 【推导逻辑】先读取shape输入的SymbolicValue，逐项解析目标维度：0复制输入对应维度，正数直接作为
 *          输出维度，-1暂存为待推导维度，其他负数拒绝；同时累乘已知输出维度。若存在-1，则用输入
 *          元素总数除以已知输出元素数得到该维度，否则校验输入输出元素总数一致。输出值按元素顺序
 *          直接复用输入SymbolicValue。
 * 【算子约束】shape输入必须有效，最多允许一个-1，且所有显式维度必须为正数或0。
 * 【举例】输入Shape=[2,3,4]、输入value=[x0,x1,...,x23]、shape value=[0,-1]时，输出Shape为[2,12]，
 *          输出value按原元素顺序透传。
 */
graphStatus ReshapeSymbolicKernelCompute(gert::InferSymbolComputeContext *context) {
  GE_ASSERT_NOTNULL(context);
  GE_ASSERT(context->GetComputeNodeInputNum() == kReshapeInputNum, "InputNum=%zu", context->GetComputeNodeInputNum());
  GE_ASSERT(context->GetComputeNodeOutputNum() == kReshapeOutputNum, "OutputNum=%zu",
            context->GetComputeNodeOutputNum());

  const auto input_tensor = context->GetInputSymbolTensor(kDataIndex);
  const auto shape_tensor = context->GetInputSymbolTensor(kShapeIndex);
  GE_UNSUPPORTED_IF_NULL(input_tensor);
  GE_UNSUPPORTED_IF_NULL(shape_tensor);
  std::vector<Expression> output_dims;
  const auto ret = BuildReshapeDims(context, shape_tensor, output_dims);
  if (ret != GRAPH_SUCCESS) {
    return ret;
  }

  auto output_tensor = context->GetOutputSymbolTensor(0U);
  GE_ASSERT_NOTNULL(output_tensor);
  output_tensor->MutableOriginSymbolShape().MutableDims() = std::move(output_dims);
  const auto input_values = input_tensor->GetSymbolicValue();
  if (input_values != nullptr) {
    auto output_values = ge::MakeUnique<std::vector<Expression>>(*input_values);
    GE_ASSERT_NOTNULL(output_values);
    output_tensor->SetSymbolicValue(std::move(output_values));
  }
  GELOGD("%s[%s] kernel success, %s", context->GetNodeName(), context->GetNodeType(),
         SymbolicInferUtil::DumpSymbolTensor(*output_tensor).c_str());
  return GRAPH_SUCCESS;
}

static graphStatus ShapeSymbolicKernelCompute(gert::InferSymbolComputeContext *context) {
  GE_CHECK_NOTNULL(context);
  GELOGD("Shape Symbolic Kernel in, node %s[%s].", context->GetNodeName(), context->GetNodeType());
  GE_ASSERT(context->GetComputeNodeInputNum() == kShapeInputOutputSize, "InputNum=%zu",
            context->GetComputeNodeInputNum());
  GE_ASSERT(context->GetComputeNodeOutputNum() == kShapeInputOutputSize, "OutputNum=%zu",
            context->GetComputeNodeOutputNum());

  auto input_tensor = context->GetInputSymbolTensor(0U);
  GE_UNSUPPORTED_IF_NULL(input_tensor);
  const auto dims = input_tensor->GetOriginSymbolShape();
  auto symbolic_tensor = context->GetOutputSymbolTensor(0U);
  GE_ASSERT_NOTNULL(symbolic_tensor);
  auto symbolic_value_unique = ge::MakeUnique<std::vector<ge::Expression>>(dims.GetDims());
  if (symbolic_value_unique != nullptr) {
    symbolic_tensor->SetSymbolicValue(std::move(symbolic_value_unique));
  }
  symbolic_tensor->MutableOriginSymbolShape().MutableDims() = {ge::Symbol(dims.GetDimNum())};

  GELOGD("%s[%s] kernel success, %s", context->GetNodeName(), context->GetNodeType(),
         SymbolicInferUtil::DumpSymbolTensor(*symbolic_tensor).c_str());
  return SUCCESS;
}

REGISTER_SYMBOLIC_KERNEL(Shape, ShapeSymbolicKernelCompute);
REGISTER_SYMBOLIC_KERNEL(Reshape, ReshapeSymbolicKernelCompute);
}  // namespace ge
