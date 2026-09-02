/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE. See
 * LICENSE in the root of the software repository for the full text of the License.
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

}  // namespace

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
    if (!GetShapeValue(values->at(i), desc->GetDataType(), dim)) {
      dims.emplace_back(values->at(i));
      known = known * values->at(i);
    } else if (dim == 0L && i < input_shape.GetDimNum()) {
      dims.emplace_back(input_shape.GetDim(i));
      known = known * input_shape.GetDim(i);
    } else if (dim > 0L) {
      dims.emplace_back(values->at(i));
      known = known * values->at(i);
    } else if (dim == -1L && unknown == std::numeric_limits<size_t>::max()) {
      unknown = i;
      dims.emplace_back(Symbol(1));
    } else {
      GELOGW("Reshape symbolic compute unsupported: invalid reshape dimension, node %s[%s].", context->GetNodeName(),
             context->GetNodeType());
      return UNSUPPORTED;
    }
  }
  const auto input_size = input_shape.GetSymbolShapeSize();
  if (unknown != std::numeric_limits<size_t>::max()) {
    dims[unknown] = input_size / known;
  } else {
    ASSERT_SYMBOL_EQ(input_size, known);
  }
  return GRAPH_SUCCESS;
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
