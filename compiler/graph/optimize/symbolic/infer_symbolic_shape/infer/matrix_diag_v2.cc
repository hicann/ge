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
#include "graph/utils/type_utils.h"
#include "graph/optimize/symbolic/infer_symbolic_shape/symbolic_infer_util.h"

namespace ge {
namespace {

constexpr size_t kSingleDiagIndexNum = 1U;
constexpr size_t kMaxDiagIndexNum = 2U;
constexpr size_t kMatrixDimsNum = 2U;
constexpr size_t kEyeOutputIdx = 0U;
constexpr size_t kNumRowsAttrIdx = 0U;
constexpr size_t kNumColumnsAttrIdx = 1U;
constexpr size_t kBatchShapeAttrIdx = 2U;

graphStatus ValidateInputShapes(const gert::SymbolShape *diagonal_shape, const gert::SymbolShape *k_shape) {
  GE_ASSERT_TRUE(diagonal_shape->GetDimNum() >= 1,
                 "[InferSymbolShape4MatrixDiagV2] diagonal_shape is invalid, dim should be at least 1");

  GE_ASSERT_TRUE(k_shape->GetDimNum() <= 1,
                 "[InferSymbolShape4MatrixDiagV2] k_shape is invalid, dim should be at most 1");

  return GRAPH_SUCCESS;
}

graphStatus ExtractDiagIndices(const std::vector<Expression> *k_value, int32_t &lower_diag_index,
                               int32_t &upper_diag_index) {
  const size_t num_elements = k_value->size();

  GE_ASSERT_TRUE(num_elements >= kSingleDiagIndexNum && num_elements <= kMaxDiagIndexNum,
                 "[InferSymbolShape4MatrixDiagV2] input[k] must be scalar or a vector with one or two elements");
  GE_ASSERT_TRUE(k_value->at(0).GetConstValue<int32_t>(lower_diag_index),
                 "[InferSymbolShape4MatrixDiagV2] k_value[0] must be a scalar");

  if (num_elements == kSingleDiagIndexNum) {
    upper_diag_index = lower_diag_index;
  } else {
    GE_ASSERT_TRUE(k_value->at(1).GetConstValue<int32_t>(upper_diag_index),
                   "[InferSymbolShape4MatrixDiagV2] k_value[1] must be a scalar");
  }

  GE_ASSERT_TRUE(lower_diag_index <= upper_diag_index,
                 "[InferSymbolShape4MatrixDiagV2] lower_diag_index must be less than or equal to upper_diag_index");

  GELOGI("lower_diag_index %d, upper_diag_index %d", lower_diag_index, upper_diag_index);

  return GRAPH_SUCCESS;
}

graphStatus ValidateMultiDiag(const std::vector<Expression> &diagonal_dims, size_t diagonal_rank,
                              int32_t lower_diag_index, int32_t upper_diag_index) {
  GE_ASSERT_TRUE(diagonal_rank >= kMatrixDimsNum,
                 "[InferSymbolShape4MatrixDiagV2] diagonal_shape is invalid, dim should be at least 2");

  auto num_diags = diagonal_dims[diagonal_rank - kMatrixDimsNum];
  auto expected_num_diags = Symbol(upper_diag_index - lower_diag_index + 1);
  ASSERT_SYMBOL_EQ(num_diags, expected_num_diags);

  return GRAPH_SUCCESS;
}

void ComputeMinDimensions(const std::vector<Expression> &diagonal_dims, size_t diagonal_rank, int32_t lower_diag_index,
                          int32_t upper_diag_index, Expression &min_num_rows, Expression &min_num_cols) {
  auto max_diag_len = diagonal_dims[diagonal_rank - 1];
  min_num_rows = max_diag_len - Symbol(std::min(lower_diag_index, 0));
  min_num_cols = max_diag_len + Symbol(std::max(upper_diag_index, 0));

  GELOGI("max_diag_len %s, min_num_rows %s, min_num_cols %s", max_diag_len.Serialize().get(),
         min_num_rows.Serialize().get(), min_num_cols.Serialize().get());
}

graphStatus ProcessNumRowsCols(const gert::SymbolTensor *num_rows_tensor, const gert::SymbolTensor *num_cols_tensor,
                               const Expression &min_num_rows, const Expression &min_num_cols, Expression &num_rows,
                               Expression &num_cols) {
  auto num_rows_value = num_rows_tensor->GetSymbolicValue();
  auto num_cols_value = num_cols_tensor->GetSymbolicValue();

  if (num_rows_value != nullptr) {
    num_rows = num_rows_value->at(0);
  }
  if (num_cols_value != nullptr) {
    num_cols = num_cols_value->at(0);
  }

  if (num_rows_value == nullptr && num_cols_value == nullptr) {
    num_rows = ge::sym::Max(min_num_rows, min_num_cols);
    num_cols = num_rows;
  } else if (num_rows_value == nullptr) {
    num_rows = min_num_rows;
  } else if (EXPECT_SYMBOL_LE(num_rows, Symbol(0))) {
    num_rows = min_num_rows;
  } else {
    GELOGI("[InferSymbolShape4MatrixDiagV2] num_rows%s is invalid, it must be greater than or equal to min_num_rows %s",
           num_rows.Serialize().get(), min_num_rows.Serialize().get());
    ASSERT_SYMBOL_GE(num_rows, min_num_rows);
  }

  if (num_cols_value == nullptr) {
    num_cols = min_num_cols;
  } else if (EXPECT_SYMBOL_LE(num_cols, Symbol(0))) {
    num_cols = min_num_cols;
    GELOGI("num_cols is <= 0, use min_num_cols %s", min_num_cols.Serialize().get());
  } else {
    GELOGI("[InferSymbolShape4MatrixDiagV2] num_cols%s is invalid, it must be greater than or equal to min_num_cols %s",
           num_cols.Serialize().get(), min_num_cols.Serialize().get());
    ASSERT_SYMBOL_GE(num_cols, min_num_cols);
  }

  return GRAPH_SUCCESS;
}

void BuildOutputShape(const std::vector<Expression> &diagonal_dims, size_t diagonal_rank, int32_t lower_diag_index,
                      int32_t upper_diag_index, const Expression &num_rows, const Expression &num_cols,
                      gert::SymbolShape *output_shape) {
  if (lower_diag_index == upper_diag_index) {
    for (size_t i = 0; i < diagonal_rank - 1; i++) {
      output_shape->MutableDims().push_back(diagonal_dims[i]);
    }
    output_shape->MutableDims().push_back(num_rows);
    output_shape->MutableDims().push_back(num_cols);
  } else {
    for (size_t i = 0; i < diagonal_rank; i++) {
      if (i == diagonal_rank - kMatrixDimsNum) {
        output_shape->MutableDims().push_back(num_rows);
      } else if (i == diagonal_rank - 1) {
        output_shape->MutableDims().push_back(num_cols);
      } else {
        output_shape->MutableDims().push_back(diagonal_dims[i]);
      }
    }
  }
}

graphStatus InferShape4MatrixDiagV2(gert::InferSymbolShapeContext *context) {
  auto diagonal_shape = context->GetInputSymbolShape(0);
  GE_UNSUPPORTED_IF_NULL(diagonal_shape);
  auto k_shape = context->GetInputSymbolShape(1);
  GE_UNSUPPORTED_IF_NULL(k_shape);
  auto k_tensor = context->GetInputSymbolTensor(1);
  GE_UNSUPPORTED_IF_NULL(k_tensor);
  auto num_rows_tensor = context->GetInputSymbolTensor(2);
  GE_UNSUPPORTED_IF_NULL(num_rows_tensor);
  auto num_cols_tensor = context->GetInputSymbolTensor(3);
  GE_UNSUPPORTED_IF_NULL(num_cols_tensor);
  auto output_shape = context->GetOutputSymbolShape(0);
  GE_ASSERT_NOTNULL(output_shape);
  output_shape->MutableDims().clear();

  GE_ASSERT_GRAPH_SUCCESS(ValidateInputShapes(diagonal_shape, k_shape));

  auto k_value = k_tensor->GetSymbolicValue();
  GE_UNSUPPORTED_IF_NULL(k_value);

  int32_t lower_diag_index = 0;
  int32_t upper_diag_index = 0;
  GE_ASSERT_GRAPH_SUCCESS(ExtractDiagIndices(k_value, lower_diag_index, upper_diag_index));

  auto diagonal_dims = diagonal_shape->GetDims();
  size_t diagonal_rank = diagonal_shape->GetDimNum();

  if (lower_diag_index < upper_diag_index) {
    GE_ASSERT_GRAPH_SUCCESS(ValidateMultiDiag(diagonal_dims, diagonal_rank, lower_diag_index, upper_diag_index));
  }

  Expression min_num_rows;
  Expression min_num_cols;
  ComputeMinDimensions(diagonal_dims, diagonal_rank, lower_diag_index, upper_diag_index, min_num_rows, min_num_cols);

  Expression num_rows;
  Expression num_cols;
  GE_ASSERT_GRAPH_SUCCESS(
      ProcessNumRowsCols(num_rows_tensor, num_cols_tensor, min_num_rows, min_num_cols, num_rows, num_cols));

  BuildOutputShape(diagonal_dims, diagonal_rank, lower_diag_index, upper_diag_index, num_rows, num_cols, output_shape);

  return GRAPH_SUCCESS;
}

/**
 * MatrixDiag 的符号 Shape 推导。
 * 【算子功能】根据输入的批量对角线元素构造批量对角矩阵。
 * 【算子约束】输入至少包含一个维度；输出保留输入全部维度，并在末尾追加一个与输入最后一维相同的维度。
 * 【推导逻辑】读取输入符号 Shape，依次复制输入维度，再追加输入最后一维作为矩阵列维度。
 * 【举例】输入 x=[B,M,N] 时，输出 y=[B,M,N,N]。
 */
graphStatus InferShape4MatrixDiag(gert::InferSymbolShapeContext *context) {
  const auto input_shape = context->GetInputSymbolShape(0);
  GE_UNSUPPORTED_IF_NULL(input_shape);
  GE_ASSERT_TRUE(input_shape->GetDimNum() >= 1, "MatrixDiag input rank must be at least 1");
  const auto output_shape = context->GetOutputSymbolShape(0);
  GE_ASSERT_NOTNULL(output_shape);
  output_shape->MutableDims() = input_shape->GetDims();
  output_shape->MutableDims().push_back(input_shape->GetDim(input_shape->GetDimNum() - 1));
  return GRAPH_SUCCESS;
}

/**
 * Eye 的符号 Shape 推导。
 * 【算子功能】创建对角线为 1、其他位置为 0 的二维单位矩阵，并支持在前置 batch 维度上扩展。
 * 【算子约束】num_rows 必须为正数；num_columns 小于等于 0 时取 num_rows；batch_shape 中的每个维度必须为正数。
 * 【推导逻辑】读取 num_rows、num_columns 和 batch_shape 属性，先写入 batch_shape，再追加 num_rows 和有效 num_columns，
 *            生成输出 Shape。
 * 【举例】num_rows=3、num_columns=4、batch_shape=[2] 时，输出 y 的符号 Shape 为 [2,3,4]。
 */
graphStatus InferShape4Eye(gert::InferSymbolShapeContext *context) {
  const auto out_shape = context->GetOutputSymbolShape(kEyeOutputIdx);
  GE_ASSERT_NOTNULL(out_shape);
  const auto attrs = context->GetAttrs();
  GE_ASSERT_NOTNULL(attrs);
  const auto num_rows = attrs->GetAttrPointer<int64_t>(kNumRowsAttrIdx);
  const auto num_columns = attrs->GetAttrPointer<int64_t>(kNumColumnsAttrIdx);
  const auto batch_shape = attrs->GetAttrPointer<gert::ContinuousVector>(kBatchShapeAttrIdx);
  GE_ASSERT_NOTNULL(num_rows);
  GE_ASSERT_NOTNULL(num_columns);
  GE_ASSERT_NOTNULL(batch_shape);
  const auto batch_dims = static_cast<const int64_t *>(batch_shape->GetData());
  GE_ASSERT(batch_shape->GetSize() == 0U || batch_dims != nullptr, "Eye batch_shape data is null");
  GE_ASSERT(*num_rows > 0, "Eye num_rows must be greater than 0, actual value[%ld]", *num_rows);

  out_shape->MutableDims().clear();
  for (size_t i = 0U; i < batch_shape->GetSize(); ++i) {
    GE_ASSERT(batch_dims[i] > 0, "Eye batch_shape must be greater than 0, actual value[%ld]", batch_dims[i]);
    out_shape->AppendDim(Symbol(batch_dims[i]));
  }
  out_shape->AppendDim(Symbol(*num_rows));
  out_shape->AppendDim(Symbol(*num_columns > 0 ? *num_columns : *num_rows));
  return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFER_SYMBOL_SHAPE_INNER(MatrixDiagV2).InferSymbolShape(InferShape4MatrixDiagV2);
IMPL_OP_INFER_SYMBOL_SHAPE_INNER(MatrixDiag).InferSymbolShape(InferShape4MatrixDiag);
IMPL_OP_INFER_SYMBOL_SHAPE_INNER(Eye).InferSymbolShape(InferShape4Eye);
}  // namespace
}  // namespace ge
