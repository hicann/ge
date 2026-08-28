/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "exe_graph/runtime/host_cpu_op_execution_context.h"

#include "common/checker.h"
#include "exe_graph/runtime/gert_mem_allocator.h"
#include "graph/operator_factory.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/math_util.h"
#include "graph/op_desc.h"

namespace gert {
namespace {
constexpr size_t kMemAlignment = 512U;

void SetTensorDesc(const StorageShape &shape, const StorageFormat &format, ge::DataType dtype, Tensor *dst) {
  auto &storage_shape = dst->MutableStorageShape();
  storage_shape = shape.GetStorageShape();
  auto &origin_shape = dst->MutableOriginShape();
  origin_shape = shape.GetOriginShape();
  dst->SetStorageFormat(format.GetStorageFormat());
  dst->SetOriginFormat(format.GetOriginFormat());
  dst->SetDataType(dtype);
}

ge::OpDescPtr GetOpDescPtr(const HostCpuOpExecutionContext &ctx) {
  const auto node_type = ctx.GetNodeType();
  auto const node_op = ge::OperatorFactory::CreateOperator("_", node_type);
  if (node_op.IsEmpty()) {
    GELOGE(ge::FAILED, "get op from OperatorFactory fail. opType: %s", node_type);
    return nullptr;
  }
  GELOGD("get op from OperatorFactory success. opType is %s", node_type);
  auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(node_op);
  return op_desc;
}
}  // namespace

Tensor *HostCpuOpExecutionContext::MallocOutputTensor(size_t index, const StorageShape &shape,
                                                      const StorageFormat &format, ge::DataType dtype) {
  const auto additional_start_index = GetAdditionalInputStartIndex();
  GE_ASSERT_TRUE(additional_start_index >= 0);
  auto *gert_allocator = GetInputValue<GertAllocator *>(additional_start_index +
                                                        static_cast<int64_t>(AdditionalInputIndex::kHostAllocator));
  GE_ASSERT_NOTNULL(gert_allocator);
  GE_ASSERT_TRUE(gert_allocator->GetPlacement() == kOnHost, "Host CPU output allocator placement must be host.");

  auto op_desc = GetOpDescPtr(*this);
  if (op_desc != nullptr) {
    auto output_name = op_desc->GetOutputNameByIndex(index);
    GE_ASSERT_TRUE(op_desc->GetInputIndexByName(output_name) == -1, "[MallocOutputTensor] output name exists in input");
  }

  auto output_tensor = GetOutputPointer<Tensor>(index);
  GE_ASSERT_NOTNULL(output_tensor);
  SetTensorDesc(shape, format, dtype, output_tensor);

  const size_t tensor_size = shape.GetStorageShape().GetShapeSize() * GetSizeByDataType(dtype);
  size_t aligned_tensor_size = tensor_size;
  GE_ASSERT_TRUE(!ge::RoundUpOverflow(tensor_size, kMemAlignment, aligned_tensor_size));
  if (output_tensor->GetTensorData().GetSize() > 0U) {
    GE_ASSERT_TRUE(output_tensor->GetPlacement() == kOnHost, "Host CPU output tensor placement must be host.");
    return output_tensor;
  }

  auto new_tensor_data = gert_allocator->MallocTensorDataFromL1(aligned_tensor_size);
  GE_ASSERT_TRUE((new_tensor_data.GetAddr() != nullptr) && (new_tensor_data.GetSize() > 0U),
                 "Malloc host output tensor data failed, size: %zu", aligned_tensor_size);
  GE_ASSERT_SUCCESS(output_tensor->MutableTensorData().ShareFrom(new_tensor_data));
  return output_tensor;
}

Tensor *HostCpuOpExecutionContext::MakeOutputRefInput(size_t output_index, size_t input_index) {
  const auto additional_start_index = GetAdditionalInputStartIndex();
  GE_ASSERT_TRUE(additional_start_index >= 0);

  auto op_desc = GetOpDescPtr(*this);
  if (op_desc != nullptr) {
    auto input_name = op_desc->GetInputNameByIndex(input_index);
    auto output_name = op_desc->GetOutputNameByIndex(output_index);
    GE_ASSERT_TRUE(input_name == output_name, "[MakeOutputRefInput] output name does not exist in input");
  }
  auto *output_tensor = GetOutputPointer<Tensor>(output_index);
  GE_ASSERT_NOTNULL(output_tensor);

  auto input_tensor = GetInputPointer<Tensor>(input_index);
  GE_ASSERT_NOTNULL(input_tensor);
  SetTensorDesc(input_tensor->GetShape(), input_tensor->GetFormat(), input_tensor->GetDataType(), output_tensor);
  output_tensor->MutableTensorData().ShareFrom(input_tensor->GetTensorData());
  return output_tensor;
}
}  // namespace gert
