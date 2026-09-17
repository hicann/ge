/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <utility>
#include <vector>
#include "common/plugin/ge_make_unique_util.h"
#include "common/checker.h"
#include "framework/common/framework_types_internal.h"
#include "graph/optimize/symbolic/infer_symbolic_shape/symbolic_infer_util.h"
#include "graph/optimize/symbolic/symbol_compute_context.h"
#include "graph/optimize/symbolic/symbolic_kernel_factory.h"

namespace ge {
namespace {
constexpr size_t kAxesAttrIndex = 0UL;
constexpr size_t kOutputIndex = 0UL;
constexpr size_t kAxesPairSize = 2UL;

graphStatus GatherShapesSymbolicKernelCompute(gert::InferSymbolComputeContext *context) {
  GE_ASSERT_NOTNULL(context);
  GELOGD("GatherShapes Symbolic Kernel in, node %s[%s].", context->GetNodeName(), context->GetNodeType());
  GE_ASSERT_TRUE(context->GetComputeNodeInputNum() != 0U, "GatherShapes input num should not be zero.");
  GE_ASSERT_TRUE(context->GetComputeNodeOutputNum() == 1U, "GatherShapes output num should be one.");

  const auto attrs = context->GetAttrs();
  GE_ASSERT_NOTNULL(attrs);
  const auto *attr_axes = attrs->GetListListInt(kAxesAttrIndex);
  GE_ASSERT_NOTNULL(attr_axes);
  const auto axes_size = attr_axes->GetSize();
  GE_ASSERT_TRUE(axes_size != 0U, "GatherShapes axes should not be empty.");

  std::vector<Expression> output_values;
  output_values.reserve(axes_size);
  for (size_t i = 0U; i < axes_size; ++i) {
    const auto axis = attr_axes->Get(i);
    GE_ASSERT_NOTNULL(axis);
    GE_ASSERT_TRUE(axis->GetSize() == kAxesPairSize, "GatherShapes axes[%zu] should contain input and dimension index.",
                   i);
    const auto *data = reinterpret_cast<const uint64_t *>(axis->GetData());
    GE_ASSERT_NOTNULL(data);
    const auto input_index = data[0U];
    const auto dim_index = data[1U];
    GE_ASSERT_TRUE(input_index < context->GetComputeNodeInputNum(),
                   "GatherShapes input index[%lu] is out of range, input num[%zu], axis[%zu].", input_index,
                   context->GetComputeNodeInputNum(), i);

    const auto input_shape = context->GetInputSymbolShape(input_index);
    GE_UNSUPPORTED_IF_NULL(input_shape);
    GE_ASSERT_TRUE(dim_index < input_shape->GetDimNum(),
                   "GatherShapes dimension index[%lu] is out of range, dimension num[%zu], axis[%zu].", dim_index,
                   input_shape->GetDimNum(), i);
    output_values.emplace_back(input_shape->GetDim(dim_index));
  }

  auto output = context->GetOutputSymbolTensor(kOutputIndex);
  GE_ASSERT_NOTNULL(output);
  output->MutableOriginSymbolShape().MutableDims() = {Symbol(static_cast<int64_t>(axes_size))};
  auto output_value = ge::MakeUnique<std::vector<Expression>>(std::move(output_values));
  GE_ASSERT_NOTNULL(output_value);
  output->SetSymbolicValue(std::move(output_value));

  GELOGD("%s[%s] kernel success, %s", context->GetNodeName(), context->GetNodeType(),
         SymbolicInferUtil::DumpSymbolTensor(*output).c_str());
  return SUCCESS;
}
}  // namespace

REGISTER_SYMBOLIC_KERNEL(GatherShapes, GatherShapesSymbolicKernelCompute);
}  // namespace ge
