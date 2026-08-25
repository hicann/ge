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

#include <cstdint>
#include <initializer_list>
#include <memory>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "exe_graph/runtime/storage_shape.h"
#include "faker/allocator_faker.h"
#include "faker/kernel_run_context_faker.h"

class HostCpuOpAllocatorFaker : public gert::AllocatorFaker {
 public:
  HostCpuOpAllocatorFaker() {
    SetPlacement(gert::kOnHost);
  }

  gert::TensorData MallocTensorDataFromL1(size_t size) override {
    return gert::TensorData(MallocL1(size), nullptr, size, GetPlacement());
  }

 private:
  std::vector<std::unique_ptr<uint8_t[]>> l1_blocks_;

  gert::TensorAddress MallocL1(size_t size) {
    if (size == 0U) {
      return nullptr;
    }
    std::unique_ptr<uint8_t[]> block(new uint8_t[size]);
    gert::TensorAddress tensor_address = block.get();
    l1_blocks_.emplace_back(std::move(block));
    return tensor_address;
  }
};

namespace gert {
class HostCpuOpExecutionContextUT : public testing::Test {
 public:
  void SetUp() override {
    BuildIn2Out2Case();
  }

 protected:
  HostCpuOpAllocatorFaker gert_allocator_;
  std::vector<Tensor> input_tensors_;
  std::vector<Tensor> output_tensors_;
  FakeKernelContextHolder context_holder_;

 private:
  void BuildIn2Out2Case() {
    input_tensors_.resize(2);
    input_tensors_[0] = {{{8, 3, 224, 224}, {8, 1, 224, 224, 16}},   // shape
                         {ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0, {}},  // format
                         kOnHost,                                    // placement
                         ge::DT_FLOAT16,                             // data type
                         (void *)0x12345};
    input_tensors_[1] = {{{2, 2, 3, 8}, {2, 2, 3, 8}},                // shape
                         {ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, {}},  // format
                         kOnHost,                                     // placement
                         ge::DT_FLOAT16,                              // data type
                         (void *)0x234565};
    output_tensors_.resize(2);
    output_tensors_[0] = {
        {{8, 3, 224, 224}, {8, 1, 224, 224, 16}}, {ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, {}}, ge::DT_FLOAT16};
    output_tensors_[1] = {
        {{8, 3, 224, 224}, {8, 1, 224, 224, 16}}, {ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, {}}, ge::DT_FLOAT16};
    context_holder_ = HostCpuOpExecutionContextFaker()
                          .IrInstanceNum({1, 1})
                          .NodeIoNum(2, 2)
                          .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0)
                          .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ)
                          .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ)
                          .NodeOutputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ)
                          .InputTensor({&input_tensors_[0], &input_tensors_[1]})
                          .OutputTensor({&output_tensors_[0], &output_tensors_[1]})
                          .Allocator(&gert_allocator_)
                          .Build();
  }
};

TEST_F(HostCpuOpExecutionContextUT, MallocOutputTensorOk) {
  auto context = context_holder_.GetContext<HostCpuOpExecutionContext>();

  ASSERT_NE(context, nullptr);
  std::initializer_list<int64_t> origin_shape = {2, 1, 3, 4};
  std::initializer_list<int64_t> storage_shape = {1, 2, 3, 4};
  auto output_tensor = context->MallocOutputTensor(1, {origin_shape, storage_shape},
                                                   {ge::FORMAT_ND, ge::FORMAT_ND, ExpandDimsType()}, ge::DT_FLOAT16);
  ASSERT_NE(output_tensor, nullptr);
  EXPECT_EQ(output_tensor->GetOriginShape(), origin_shape);
  EXPECT_EQ(output_tensor->GetStorageShape(), storage_shape);
  EXPECT_EQ(output_tensor->GetOriginFormat(), ge::FORMAT_ND);
  EXPECT_EQ(output_tensor->GetStorageFormat(), ge::FORMAT_ND);
  EXPECT_EQ(output_tensor->GetDataType(), ge::DT_FLOAT16);
  EXPECT_EQ(output_tensor->GetPlacement(), kOnHost);
  EXPECT_EQ(output_tensor->GetSize(), 512UL);
  EXPECT_NE(output_tensor->GetAddr(), nullptr);
}

TEST_F(HostCpuOpExecutionContextUT, MallocOutputTensorRefOutputError) {
  auto context = context_holder_.GetContext<HostCpuOpExecutionContext>();

  ASSERT_NE(context, nullptr);
  std::initializer_list<int64_t> origin_shape = {2, 1, 3, 4};
  std::initializer_list<int64_t> storage_shape = {1, 2, 3, 4};
  auto output_tensor = context->MallocOutputTensor(0, {origin_shape, storage_shape},
                                                   {ge::FORMAT_ND, ge::FORMAT_ND, ExpandDimsType()}, ge::DT_FLOAT16);
  EXPECT_EQ(output_tensor, nullptr);
}

TEST_F(HostCpuOpExecutionContextUT, MakeOutputRefInput) {
  auto context = context_holder_.GetContext<HostCpuOpExecutionContext>();
  ASSERT_NE(context, nullptr);

  auto output_tensor = context->MakeOutputRefInput(0, 0);
  ASSERT_NE(output_tensor, nullptr);
  EXPECT_EQ(output_tensor->GetOriginShape(), input_tensors_[0].GetOriginShape());
  EXPECT_EQ(output_tensor->GetStorageShape(), input_tensors_[0].GetStorageShape());
  EXPECT_EQ(output_tensor->GetOriginFormat(), input_tensors_[0].GetOriginFormat());
  EXPECT_EQ(output_tensor->GetStorageFormat(), input_tensors_[0].GetStorageFormat());
  EXPECT_EQ(output_tensor->GetDataType(), input_tensors_[0].GetDataType());
  EXPECT_EQ(output_tensor->GetPlacement(), kOnHost);
  EXPECT_EQ(output_tensor->GetAddr(), input_tensors_[0].GetAddr());
}

TEST_F(HostCpuOpExecutionContextUT, MakeOutputRefInputError) {
  auto context = context_holder_.GetContext<HostCpuOpExecutionContext>();
  ASSERT_NE(context, nullptr);

  auto output_tensor = context->MakeOutputRefInput(1, 0);
  EXPECT_EQ(output_tensor, nullptr);
}

}  // namespace gert
