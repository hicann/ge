/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "exe_graph/runtime/eager_op_execution_context.h"
#include "exe_graph/runtime/gert_mem_allocator.h"
#include "common/checker.h"

namespace gert {
namespace {
void SetTensorDesc(const StorageShape &shape, const StorageFormat &format, ge::DataType dtype, Tensor *dst) {
  auto &storage_shape = dst->MutableStorageShape();
  storage_shape = shape.GetStorageShape();
  auto &origin_shape = dst->MutableOriginShape();
  origin_shape = shape.GetOriginShape();
  dst->SetStorageFormat(format.GetStorageFormat());
  dst->SetOriginFormat(format.GetOriginFormat());
  dst->SetDataType(dtype);
}
} // namespace

rtStream EagerOpExecutionContext::GetStream() const {
  auto start_index = GetAdditionalInputStartIndex();
  GE_ASSERT_TRUE(start_index >= 0);

  const auto av = GetInput(start_index + static_cast<int64_t>(AdditionalInputIndex::kStream));
  GE_ASSERT_NOTNULL(av);
  return av->GetValue<rtStream>();
}

Tensor *EagerOpExecutionContext::MallocOutputTensor(size_t index, const StorageShape &shape,
                                                    const StorageFormat &format, ge::DataType dtype,
                                                    size_t tensor_size) {
  auto additional_start_index = GetAdditionalInputStartIndex();
  GE_ASSERT_TRUE(additional_start_index >= 0);
  auto gert_allocator =
      GetInputValue<GertAllocator *>(additional_start_index + static_cast<int64_t>(AdditionalInputIndex::kDeviceAllocator));
  GE_ASSERT_NOTNULL(gert_allocator);
  auto output_tensor = GetOutputPointer<Tensor>(index);
  GE_ASSERT_NOTNULL(output_tensor);
  SetTensorDesc(shape, format, dtype, output_tensor);

  auto gert_tensor_data = gert_allocator->MallocTensorData(tensor_size);
  output_tensor->SetData(std::move(gert_tensor_data.MutableTensorData()));
  return output_tensor;
}

Tensor *EagerOpExecutionContext::MakeOutputRefInput(size_t output_index, size_t input_index) const {
  Tensor *output_tensor = const_cast<Tensor *>(GetOutputPointer<Tensor>(output_index));
  GE_ASSERT_NOTNULL(output_tensor);
  auto input_tensor = GetInputPointer<Tensor>(input_index);
  GE_ASSERT_NOTNULL(input_tensor);

  SetTensorDesc(input_tensor->GetShape(), input_tensor->GetFormat(), input_tensor->GetDataType(), output_tensor);
  output_tensor->MutableTensorData().ShareFrom(input_tensor->GetTensorData());
  return output_tensor;
}

void *EagerOpExecutionContext::MallocWorkSpace(size_t size) {
  auto additional_input_index = GetAdditionalInputStartIndex();
  GE_ASSERT_TRUE(additional_input_index >= 0);
  auto gert_allocator = GetInputValue<GertAllocator *>(additional_input_index +
                                                       static_cast<int64_t>(AdditionalInputIndex::kDeviceAllocator));
  GE_ASSERT_NOTNULL(gert_allocator);

  auto additional_output_start = GetAdditionalOutputStartIndex();
  auto memory_vec =
      GetOutputPointer<std::vector<GertMemBlock *>>(additional_output_start + static_cast<size_t>(AdditionalOutputIndex::kWorkSpace));
  GE_ASSERT_NOTNULL(memory_vec);

  auto mem_block = gert_allocator->Malloc(size);
  GE_ASSERT_NOTNULL(mem_block);
  (void)memory_vec->emplace_back(mem_block);
  return mem_block->GetAddr();
}
}  // namespace gert