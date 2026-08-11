/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "kernel/common_kernel_impl/copy_flow_launch.h"
#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include "faker/kernel_run_context_facker.h"
#include "faker/node_faker.h"
#include "register/kernel_registry.h"
#include "aicore/launch_kernel/rt_kernel_launch_args_ex.h"
#include "kernel/memory/mem_block.h"
#include "kernel/memory/single_stream_l2_allocator.h"
#include "kernel/memory/caching_mem_allocator.h"
#include "exe_graph/runtime/gert_mem_allocator.h"
#include "kernel/tensor_attr.h"
#include "faker/fake_value.h"
#include "faker/kernel_outputs_faker.h"
#include "faker/continuous_vector_builder.h"
#include "common/plugin/ge_make_unique_util.h"
#include "stub/gert_runtime_stub.h"
#include "checker/memory_profiling_log_matcher.h"
#include "core/executor/multi_thread_topological/executor/schedule/producer/producers/kernel_tags/critical_section_config.h"
#include "core/utils/rt2_tensor_utils.h"
#include "kernel/memory/single_stream_l2_allocator.h"
#include "common/tiling_fwk_data_helper.h"
#include "tests/depends/runtime/src/runtime_stub.h"

namespace gert {
struct CopyFlowLaunchKernelTest : public testing::Test {
  memory::CachingMemAllocator caching_mem_allocator_{0};
  memory::SingleStreamL2Allocator single_stream_l2_allocator_{&caching_mem_allocator_};
  KernelRegistry &registry = KernelRegistry::GetInstance();
};
namespace {
class MockCopyFlowAclRuntime : public ge::AclRuntimeStub {
 public:
  MOCK_METHOD6(aclrtMemcpyAsync, int32_t(void *dst, size_t dest_max, const void *src, size_t src_count,
                                         aclrtMemcpyKind kind, aclrtStream stream));
};

int64_t GetOffset(void *addr1, void *addr2) {
  auto value2 = reinterpret_cast<size_t>(addr2);
  auto value1 = reinterpret_cast<size_t>(addr1);
  if (value2 >= value1) {
    return static_cast<int64_t>(value2 - value1);
  } else {
    return static_cast<int64_t>(value1 - value2) * -1;
  }
}

std::unique_ptr<uint8_t[]> GetContinuousVector2DByVector2D(const std::vector<std::vector<int32_t>> &vector_2d,
                                                           size_t &total_size) {
  total_size = ContinuousVectorVector::GetOverHeadLength(vector_2d.size());
  for (const auto &inner_vec : vector_2d) {
    size_t inner_vec_length = 0U;
    GE_ASSERT_TRUE(!ge::MulOverflow(inner_vec.size(), sizeof(int32_t), inner_vec_length));
    GE_ASSERT_TRUE(!ge::AddOverflow(inner_vec_length, sizeof(ContinuousVector), inner_vec_length));
    GE_ASSERT_TRUE(!ge::AddOverflow(total_size, inner_vec_length, total_size));
  }
  auto holder = ge::MakeUnique<uint8_t[]>(total_size);
  auto cvv = new (holder.get()) ContinuousVectorVector();
  GE_ASSERT_NOTNULL(cvv);
  cvv->Init(vector_2d.size());

  for (const auto &inner_vec : vector_2d) {
    auto cv = cvv->Add<int32_t>(inner_vec.size());
    GE_ASSERT_NOTNULL(cv);
    if (!inner_vec.empty()) {
      const size_t copy_size = inner_vec.size() * sizeof(int32_t);
      GE_ASSERT_EOK(memcpy_s(cv->MutableData(), cv->GetCapacity() * sizeof(int32_t), inner_vec.data(), copy_size));
    }
  }
  return holder;
}
}  // namespace

// input 0 host 输入，input 1 device 输入，1 随路拷贝
TEST_F(CopyFlowLaunchKernelTest, CopyFlowLaunch_1H1D) {
  auto tensor_holder1 = TensorFaker().Placement(kOnHost).Shape({}).DataType(ge::DT_INT32).Value<int32_t>({2}).Build();
  auto shape1 = tensor_holder1.GetTensor()->GetShape(); /**/
  auto data_type1 = tensor_holder1.GetTensor()->GetDataType();
  auto size1 = ge::RoundUp(ge::GetSizeInBytes(shape1.GetStorageShape().GetShapeSize(), data_type1), 512);

  auto tensor_holder2 = TensorFaker().Placement(kOnDeviceHbm).Shape({10, 20}).Build();
  auto shape2 = tensor_holder2.GetTensor()->GetShape();
  auto data_type2 = tensor_holder2.GetTensor()->GetDataType();
  auto size2 = ge::RoundUp(ge::GetSizeInBytes(shape2.GetStorageShape().GetShapeSize(), data_type2), 512);
  auto outputs = OutputsHolder::Fake(2);

  std::vector<std::vector<int32_t>> host_inputs_addr_index = {{0}, {1}};
  size_t cvv_total_size = 0U;
  auto const_index_data = GetContinuousVector2DByVector2D(host_inputs_addr_index, cvv_total_size);
  // 这里指的是addr 的input out 个数，和 CopyFlowLaunch 输入输出有区别
  size_t input_num = 2;
  size_t output_num = 2;
  std::unique_ptr<uint8_t[]> launch_args = CreateLaunchArg(input_num, output_num);
  GertTensorData input_tensor_data1;
  TensorUtils::RefTdToGtd(tensor_holder1.GetTensor()->GetTensorData(), -1, input_tensor_data1);
  GertTensorData input_tensor_data2;
  TensorUtils::RefTdToGtd(tensor_holder2.GetTensor()->GetTensorData(), -1, input_tensor_data2);
  auto context_holder = KernelRunContextFaker()
                            .KernelIONum(13, 2)
                            .Inputs(std::vector<void *>(
                                {reinterpret_cast<void *>(input_num), reinterpret_cast<void *>(const_index_data.get()),
                                 reinterpret_cast<void *>(launch_args.get()), reinterpret_cast<void *>(0x11),
                                 &single_stream_l2_allocator_, &input_tensor_data1, reinterpret_cast<void *>(size1),
                                 &shape1, reinterpret_cast<void *>(data_type1), &input_tensor_data2,
                                 reinterpret_cast<void *>(size2), &shape2, reinterpret_cast<void *>(data_type2)}))
                            .Outputs(outputs.pointer)
                            .Build();

  auto run_context = context_holder.GetContext<KernelContext>();
  auto funcs = KernelRegistry::GetInstance().FindKernelFuncs(kernel::kCopyFlowLaunch);
  ASSERT_NE(funcs, nullptr);
  ASSERT_NE(funcs->outputs_creator, nullptr);
  ASSERT_NE(funcs->run_func, nullptr);

  ASSERT_EQ(funcs->outputs_creator(nullptr, run_context), ge::GRAPH_SUCCESS);
  EXPECT_EQ(funcs->run_func(run_context), ge::GRAPH_SUCCESS);
  auto addr_start = static_cast<size_t>(kernel::CopyFlowLaunchOutputs::kAddress);

  auto tensor_data2 = run_context->GetOutputPointer<GertTensorData>(addr_start + 1);
  ASSERT_NE(tensor_data2, nullptr);
  EXPECT_EQ(tensor_data2->GetAddr(), tensor_holder2.GetTensor()->GetAddr());
  EXPECT_EQ(tensor_data2->GetSize(), tensor_holder2.GetTensor()->GetSize());
  EXPECT_EQ(tensor_data2->GetPlacement(), kOnDeviceHbm);

  auto args = run_context->MutableInputPointer<gert::RtKernelLaunchArgsEx>(
      static_cast<size_t>(kernel::CopyFlowLaunchInputs::kRtArg));
  ASSERT_NE(args, nullptr);
  EXPECT_EQ(args->GetBase()->hostInputInfoNum, 1);
  EXPECT_EQ(args->GetBase()->hostInputInfoPtr[0].dataOffset,
            GetOffset(args->GetArgBase(), args->GetArgsPointer<void>(RtKernelLaunchArgsEx::ArgsType::kHostInputData)));

  int32_t input0_value =
      *(reinterpret_cast<int32_t *>(args->GetArgsPointer<void>(RtKernelLaunchArgsEx::ArgsType::kHostInputData)));
  EXPECT_EQ(input0_value, 2);

  context_holder.FreeAll();
}

// 两个输入都为host，所有host input 输入和等于kMaxHostInputDataLen,  都进行随路拷贝
TEST_F(CopyFlowLaunchKernelTest, CopyFlowLaunch_2H) {
  auto tensor_holder1 = TensorFaker().Placement(kOnHost).Shape({}).DataType(ge::DT_INT32).Value<int32_t>({2}).Build();
  auto shape1 = tensor_holder1.GetTensor()->GetShape();
  auto data_type1 = tensor_holder1.GetTensor()->GetDataType();
  auto size1 = ge::RoundUp(ge::GetSizeInBytes(shape1.GetStorageShape().GetShapeSize(), data_type1), 512);

  auto tensor_holder2 = TensorFaker().Placement(kOnHost).Shape({}).DataType(ge::DT_INT32).Value<int32_t>({5}).Build();
  auto shape2 = tensor_holder2.GetTensor()->GetShape();
  auto data_type2 = tensor_holder2.GetTensor()->GetDataType();
  auto size2 = ge::RoundUp(ge::GetSizeInBytes(shape2.GetStorageShape().GetShapeSize(), data_type2), 512);
  auto outputs = OutputsHolder::Fake(2);

  std::vector<std::vector<int32_t>> host_inputs_addr_index = {{0}, {1}};
  size_t cvv_total_size = 0U;
  auto const_index_data = GetContinuousVector2DByVector2D(host_inputs_addr_index, cvv_total_size);
  // 这里指的是addr 的input out 个数，和 CopyFlowLaunch 输入输出有区别
  size_t input_num = 2;
  size_t output_num = 2;
  std::unique_ptr<uint8_t[]> launch_args = CreateLaunchArg(input_num, output_num);
  GertTensorData input_tensor_data1;
  TensorUtils::RefTdToGtd(tensor_holder1.GetTensor()->GetTensorData(), -1, input_tensor_data1);
  GertTensorData input_tensor_data2;
  TensorUtils::RefTdToGtd(tensor_holder2.GetTensor()->GetTensorData(), -1, input_tensor_data2);
  auto context_holder = KernelRunContextFaker()
                            .KernelIONum(13, 2)
                            .Inputs(std::vector<void *>(
                                {reinterpret_cast<void *>(input_num), reinterpret_cast<void *>(const_index_data.get()),
                                 reinterpret_cast<void *>(launch_args.get()), reinterpret_cast<void *>(0x11),
                                 &single_stream_l2_allocator_, &input_tensor_data1, reinterpret_cast<void *>(size1),
                                 &shape1, reinterpret_cast<void *>(data_type1), &input_tensor_data2,
                                 reinterpret_cast<void *>(size2), &shape2, reinterpret_cast<void *>(data_type2)}))
                            .Outputs(outputs.pointer)
                            .Build();

  auto run_context = context_holder.GetContext<KernelContext>();
  auto funcs = KernelRegistry::GetInstance().FindKernelFuncs(kernel::kCopyFlowLaunch);
  ASSERT_NE(funcs, nullptr);
  ASSERT_NE(funcs->outputs_creator, nullptr);
  ASSERT_NE(funcs->run_func, nullptr);

  ASSERT_EQ(funcs->outputs_creator(nullptr, run_context), ge::GRAPH_SUCCESS);
  EXPECT_EQ(funcs->run_func(run_context), ge::GRAPH_SUCCESS);

  auto args = run_context->MutableInputPointer<gert::RtKernelLaunchArgsEx>(
      static_cast<size_t>(kernel::CopyFlowLaunchInputs::kRtArg));
  ASSERT_NE(args, nullptr);
  EXPECT_EQ(args->GetBase()->hostInputInfoNum, 2);
  EXPECT_EQ(args->GetBase()->hostInputInfoPtr[0].dataOffset,
            GetOffset(args->GetArgBase(), args->GetArgsPointer<void>(RtKernelLaunchArgsEx::ArgsType::kHostInputData)));

  int32_t input0_value =
      *(reinterpret_cast<int32_t *>(args->GetArgsPointer<void>(RtKernelLaunchArgsEx::ArgsType::kHostInputData)));
  EXPECT_EQ(input0_value, 2);
  auto input1_value_addr =
      ge::ValueToPtr((ge::PtrToValue(args->GetArgBase()) + args->GetBase()->hostInputInfoPtr[1].dataOffset));
  int32_t input1_value = *(reinterpret_cast<int32_t *>(input1_value_addr));
  EXPECT_EQ(input1_value, 5);

  context_holder.FreeAll();
}

// 两个host input 输入和等于kMaxHostInputDataLen, 都进行随路拷贝。第2个host输入输出给KernelLaunch节点的第1和2个输入。
TEST_F(CopyFlowLaunchKernelTest, CopyFlowLaunch_2Host3Inputs) {
  auto tensor_holder1 = TensorFaker().Placement(kOnHost).Shape({}).DataType(ge::DT_INT32).Value<int32_t>({2}).Build();
  auto shape1 = tensor_holder1.GetTensor()->GetShape();
  auto data_type1 = tensor_holder1.GetTensor()->GetDataType();
  auto size1 = ge::RoundUp(ge::GetSizeInBytes(shape1.GetStorageShape().GetShapeSize(), data_type1), 512);

  auto tensor_holder2 = TensorFaker().Placement(kOnHost).Shape({}).DataType(ge::DT_INT32).Value<int32_t>({5}).Build();
  auto shape2 = tensor_holder2.GetTensor()->GetShape();
  auto data_type2 = tensor_holder2.GetTensor()->GetDataType();
  auto size2 = ge::RoundUp(ge::GetSizeInBytes(shape2.GetStorageShape().GetShapeSize(), data_type2), 512);
  auto outputs = OutputsHolder::Fake(2);

  std::vector<std::vector<int32_t>> host_inputs_addr_index = {{0}, {1, 2}};
  size_t cvv_total_size = 0U;
  auto const_index_data = GetContinuousVector2DByVector2D(host_inputs_addr_index, cvv_total_size);
  // 这里指的是addr 的input out 个数，和 CopyFlowLaunch 输入输出有区别
  size_t input_num = 2;
  size_t output_num = 2;
  std::unique_ptr<uint8_t[]> launch_args = CreateLaunchArg(input_num + 1, output_num);
  GertTensorData input_tensor_data1;
  TensorUtils::RefTdToGtd(tensor_holder1.GetTensor()->GetTensorData(), -1, input_tensor_data1);
  GertTensorData input_tensor_data2;
  TensorUtils::RefTdToGtd(tensor_holder2.GetTensor()->GetTensorData(), -1, input_tensor_data2);
  auto context_holder = KernelRunContextFaker()
                            .KernelIONum(13, 2)
                            .Inputs(std::vector<void *>(
                                {reinterpret_cast<void *>(input_num), reinterpret_cast<void *>(const_index_data.get()),
                                 reinterpret_cast<void *>(launch_args.get()), reinterpret_cast<void *>(0x11),
                                 &single_stream_l2_allocator_, &input_tensor_data1, reinterpret_cast<void *>(size1),
                                 &shape1, reinterpret_cast<void *>(data_type1), &input_tensor_data2,
                                 reinterpret_cast<void *>(size2), &shape2, reinterpret_cast<void *>(data_type2)}))
                            .Outputs(outputs.pointer)
                            .Build();

  auto run_context = context_holder.GetContext<KernelContext>();
  auto funcs = KernelRegistry::GetInstance().FindKernelFuncs(kernel::kCopyFlowLaunch);
  ASSERT_NE(funcs, nullptr);
  ASSERT_NE(funcs->outputs_creator, nullptr);
  ASSERT_NE(funcs->run_func, nullptr);

  ASSERT_EQ(funcs->outputs_creator(nullptr, run_context), ge::GRAPH_SUCCESS);
  EXPECT_EQ(funcs->run_func(run_context), ge::GRAPH_SUCCESS);

  auto args = run_context->MutableInputPointer<gert::RtKernelLaunchArgsEx>(
      static_cast<size_t>(kernel::CopyFlowLaunchInputs::kRtArg));
  ASSERT_NE(args, nullptr);
  EXPECT_EQ(args->GetBase()->hostInputInfoNum, 3);
  EXPECT_EQ(args->GetBase()->hostInputInfoPtr[0].dataOffset,
            GetOffset(args->GetArgBase(), args->GetArgsPointer<void>(RtKernelLaunchArgsEx::ArgsType::kHostInputData)));
  EXPECT_EQ(
      args->GetBase()->hostInputInfoPtr[1].dataOffset,
      GetOffset(args->GetArgBase(), args->GetArgsPointer<void>(RtKernelLaunchArgsEx::ArgsType::kHostInputData)) + 4);
  EXPECT_EQ(args->GetBase()->hostInputInfoPtr[2].dataOffset, args->GetBase()->hostInputInfoPtr[1].dataOffset);

  int32_t input0_value =
      *(reinterpret_cast<int32_t *>(args->GetArgsPointer<void>(RtKernelLaunchArgsEx::ArgsType::kHostInputData)));
  EXPECT_EQ(input0_value, 2);
  auto input1_value_addr =
      ge::ValueToPtr((ge::PtrToValue(args->GetArgBase()) + args->GetBase()->hostInputInfoPtr[1].dataOffset));
  int32_t input1_value = *(reinterpret_cast<int32_t *>(input1_value_addr));
  EXPECT_EQ(input1_value, 5);

  context_holder.FreeAll();
}

// 两个输入都为host，host1 input 小于kMaxHostInputDataLen，所有host input 输入和大于kMaxHostInputDataLen,
// 只有host1 随路拷贝
TEST_F(CopyFlowLaunchKernelTest, CopyFlowLaunch_2H_1) {
  auto tensor_holder1 = TensorFaker().Placement(kOnHost).Shape({}).DataType(ge::DT_INT32).Value<int32_t>({2}).Build();
  auto shape1 = tensor_holder1.GetTensor()->GetShape();
  auto data_type1 = tensor_holder1.GetTensor()->GetDataType();
  auto size1 = ge::RoundUp(ge::GetSizeInBytes(shape1.GetStorageShape().GetShapeSize(), data_type1), 512);

  auto tensor_holder2 = TensorFaker().Placement(kOnHost).Shape({10, 5}).Build();
  auto shape2 = tensor_holder2.GetTensor()->GetShape();
  auto data_type2 = tensor_holder2.GetTensor()->GetDataType();
  auto size2 = ge::RoundUp(ge::GetSizeInBytes(shape2.GetStorageShape().GetShapeSize(), data_type2), 512);
  auto outputs = OutputsHolder::Fake(2);

  std::vector<std::vector<int32_t>> host_inputs_addr_index = {{0}, {1}};
  size_t cvv_total_size = 0U;
  auto const_index_data = GetContinuousVector2DByVector2D(host_inputs_addr_index, cvv_total_size);
  // 这里指的是addr 的input out 个数，和 CopyFlowLaunch 输入输出有区别
  size_t input_num = 2;
  size_t output_num = 2;
  std::unique_ptr<uint8_t[]> launch_args = CreateLaunchArg(input_num, output_num);
  GertTensorData input_tensor_data1;
  TensorUtils::RefTdToGtd(tensor_holder1.GetTensor()->GetTensorData(), -1, input_tensor_data1);
  GertTensorData input_tensor_data2;
  TensorUtils::RefTdToGtd(tensor_holder2.GetTensor()->GetTensorData(), -1, input_tensor_data2);
  auto context_holder = KernelRunContextFaker()
                            .KernelIONum(13, 2)
                            .Inputs(std::vector<void *>(
                                {reinterpret_cast<void *>(input_num), reinterpret_cast<void *>(const_index_data.get()),
                                 reinterpret_cast<void *>(launch_args.get()), reinterpret_cast<void *>(0x11),
                                 &single_stream_l2_allocator_, &input_tensor_data1, reinterpret_cast<void *>(size1),
                                 &shape1, reinterpret_cast<void *>(data_type1), &input_tensor_data2,
                                 reinterpret_cast<void *>(size2), &shape2, reinterpret_cast<void *>(data_type2)}))
                            .Outputs(outputs.pointer)
                            .Build();

  auto run_context = context_holder.GetContext<KernelContext>();
  auto funcs = KernelRegistry::GetInstance().FindKernelFuncs(kernel::kCopyFlowLaunch);
  ASSERT_NE(funcs, nullptr);
  ASSERT_NE(funcs->outputs_creator, nullptr);
  ASSERT_NE(funcs->run_func, nullptr);

  ASSERT_EQ(funcs->outputs_creator(nullptr, run_context), ge::GRAPH_SUCCESS);
  GertRuntimeStub runtime_stub;
  runtime_stub.GetSlogStub().NoConsoleOut().SetLevelInfo();
  EXPECT_EQ(funcs->run_func(run_context), ge::GRAPH_SUCCESS);
  ASSERT_TRUE(runtime_stub.GetSlogStub().FindInfoLogRegex(kAllocRe) >= 0);

  auto addr_start = static_cast<size_t>(kernel::CopyFlowLaunchOutputs::kAddress);
  auto tensor_data2 = run_context->GetOutputPointer<GertTensorData>(addr_start + 1);
  ASSERT_NE(tensor_data2, nullptr);
  EXPECT_NE(tensor_data2->GetAddr(), tensor_holder2.GetTensor()->GetAddr());
  EXPECT_EQ(memcmp(tensor_data2->GetAddr(), tensor_holder2.GetTensor()->GetAddr(),
                   ge::GetSizeInBytes(shape1.GetStorageShape().GetShapeSize(), data_type2)),
            0);
  EXPECT_EQ(tensor_data2->GetPlacement(), kOnDeviceHbm);

  auto args = run_context->MutableInputPointer<gert::RtKernelLaunchArgsEx>(
      static_cast<size_t>(kernel::CopyFlowLaunchInputs::kRtArg));
  ASSERT_NE(args, nullptr);
  EXPECT_EQ(args->GetBase()->hostInputInfoNum, 1);

  int32_t input0_value =
      *(reinterpret_cast<int32_t *>(args->GetArgsPointer<void>(RtKernelLaunchArgsEx::ArgsType::kHostInputData)));
  EXPECT_EQ(input0_value, 2);

  context_holder.FreeAll();
}

// 两个输入都为host，host1 input 大于kMaxHostInputDataLen，host 2 小于kMaxHostInputDataLen, 只有host2 随路拷贝
TEST_F(CopyFlowLaunchKernelTest, CopyFlowLaunch_2H_2) {
  auto tensor_holder1 = TensorFaker().Placement(kOnHost).Shape({10, 5}).Build();
  auto shape1 = tensor_holder1.GetTensor()->GetShape();
  auto data_type1 = tensor_holder1.GetTensor()->GetDataType();
  auto size1 = ge::RoundUp(ge::GetSizeInBytes(shape1.GetStorageShape().GetShapeSize(), data_type1), 512);

  auto tensor_holder2 = TensorFaker().Placement(kOnHost).Shape({5, 5}).Build();
  auto shape2 = tensor_holder2.GetTensor()->GetShape();
  auto data_type2 = tensor_holder2.GetTensor()->GetDataType();
  auto size2 = ge::RoundUp(ge::GetSizeInBytes(shape2.GetStorageShape().GetShapeSize(), data_type2), 512);
  auto outputs = OutputsHolder::Fake(2);

  std::vector<std::vector<int32_t>> host_inputs_addr_index = {{0}, {1}};
  size_t cvv_total_size = 0U;
  auto const_index_data = GetContinuousVector2DByVector2D(host_inputs_addr_index, cvv_total_size);
  // 这里指的是addr 的input out 个数，和 CopyFlowLaunch 输入输出有区别
  size_t input_num = 2;
  size_t output_num = 2;
  std::unique_ptr<uint8_t[]> launch_args = CreateLaunchArg(input_num, output_num);
  GertTensorData input_tensor_data1;
  TensorUtils::RefTdToGtd(tensor_holder1.GetTensor()->GetTensorData(), -1, input_tensor_data1);
  GertTensorData input_tensor_data2;
  TensorUtils::RefTdToGtd(tensor_holder2.GetTensor()->GetTensorData(), -1, input_tensor_data2);
  auto context_holder = KernelRunContextFaker()
                            .KernelIONum(12, 3)
                            .Inputs(std::vector<void *>(
                                {reinterpret_cast<void *>(input_num), reinterpret_cast<void *>(const_index_data.get()),
                                 reinterpret_cast<void *>(launch_args.get()), reinterpret_cast<void *>(0x11),
                                 &single_stream_l2_allocator_, &input_tensor_data1, reinterpret_cast<void *>(size1),
                                 &shape1, reinterpret_cast<void *>(data_type1), &input_tensor_data2,
                                 reinterpret_cast<void *>(size2), &shape2, reinterpret_cast<void *>(data_type2)}))
                            .Outputs(outputs.pointer)
                            .Build();

  auto run_context = context_holder.GetContext<KernelContext>();
  auto funcs = KernelRegistry::GetInstance().FindKernelFuncs(kernel::kCopyFlowLaunch);
  ASSERT_NE(funcs, nullptr);
  ASSERT_NE(funcs->outputs_creator, nullptr);
  ASSERT_NE(funcs->run_func, nullptr);

  ASSERT_EQ(funcs->outputs_creator(nullptr, run_context), ge::GRAPH_SUCCESS);
  GertRuntimeStub runtime_stub;
  runtime_stub.GetSlogStub().NoConsoleOut().SetLevelInfo();
  EXPECT_EQ(funcs->run_func(run_context), ge::GRAPH_SUCCESS);
  ASSERT_TRUE(runtime_stub.GetSlogStub().FindInfoLogRegex(kAllocRe) >= 0);

  auto addr_start = static_cast<size_t>(kernel::CopyFlowLaunchOutputs::kAddress);
  auto tensor_data1 = run_context->GetOutputPointer<GertTensorData>(addr_start);
  ASSERT_NE(tensor_data1, nullptr);
  EXPECT_NE(tensor_data1->GetAddr(), tensor_holder1.GetTensor()->GetAddr());
  EXPECT_EQ(memcmp(tensor_data1->GetAddr(), tensor_holder1.GetTensor()->GetAddr(),
                   ge::GetSizeInBytes(shape1.GetStorageShape().GetShapeSize(), data_type1)),
            0);
  EXPECT_EQ(tensor_data1->GetPlacement(), kOnDeviceHbm);

  auto args = run_context->MutableInputPointer<gert::RtKernelLaunchArgsEx>(
      static_cast<size_t>(kernel::CopyFlowLaunchInputs::kRtArg));
  ASSERT_NE(args, nullptr);
  EXPECT_EQ(args->GetBase()->hostInputInfoNum, 1);
  context_holder.FreeAll();
}

// 两个输入都为host，host1 和 host2 input 都大于kMaxHostInputDataLen，都不随路拷贝
TEST_F(CopyFlowLaunchKernelTest, NoCopyFlowLaunch_2H) {
  auto tensor_holder1 = TensorFaker().Placement(kOnHost).Shape({10, 5}).Build();
  auto shape1 = tensor_holder1.GetTensor()->GetShape();
  auto data_type1 = tensor_holder1.GetTensor()->GetDataType();
  auto size1 = ge::RoundUp(ge::GetSizeInBytes(shape1.GetStorageShape().GetShapeSize(), data_type1), 512);

  auto tensor_holder2 = TensorFaker().Placement(kOnHost).Shape({5, 10}).Build();
  auto shape2 = tensor_holder2.GetTensor()->GetShape();
  auto data_type2 = tensor_holder2.GetTensor()->GetDataType();
  auto size2 = ge::RoundUp(ge::GetSizeInBytes(shape2.GetStorageShape().GetShapeSize(), data_type2), 512);
  auto outputs = OutputsHolder::Fake(2);

  std::vector<std::vector<int32_t>> host_inputs_addr_index = {{0}, {1}};
  size_t cvv_total_size = 0U;
  auto const_index_data = GetContinuousVector2DByVector2D(host_inputs_addr_index, cvv_total_size);
  // 这里指的是addr 的input out 个数，和 CopyFlowLaunch 输入输出有区别
  size_t input_num = 2;
  size_t output_num = 2;
  std::unique_ptr<uint8_t[]> launch_args = CreateLaunchArg(input_num, output_num);
  GertTensorData input_tensor_data1;
  TensorUtils::RefTdToGtd(tensor_holder1.GetTensor()->GetTensorData(), -1, input_tensor_data1);
  GertTensorData input_tensor_data2;
  TensorUtils::RefTdToGtd(tensor_holder2.GetTensor()->GetTensorData(), -1, input_tensor_data2);
  auto context_holder = KernelRunContextFaker()
                            .KernelIONum(13, 2)
                            .Inputs(std::vector<void *>(
                                {reinterpret_cast<void *>(input_num), reinterpret_cast<void *>(const_index_data.get()),
                                 reinterpret_cast<void *>(launch_args.get()), reinterpret_cast<void *>(0x11),
                                 &single_stream_l2_allocator_, &input_tensor_data1, reinterpret_cast<void *>(size1),
                                 &shape1, reinterpret_cast<void *>(data_type1), &input_tensor_data2,
                                 reinterpret_cast<void *>(size2), &shape2, reinterpret_cast<void *>(data_type2)}))
                            .Outputs(outputs.pointer)
                            .Build();

  auto run_context = context_holder.GetContext<KernelContext>();
  auto funcs = KernelRegistry::GetInstance().FindKernelFuncs(kernel::kCopyFlowLaunch);
  ASSERT_NE(funcs, nullptr);
  ASSERT_NE(funcs->outputs_creator, nullptr);
  ASSERT_NE(funcs->run_func, nullptr);

  ASSERT_EQ(funcs->outputs_creator(nullptr, run_context), ge::GRAPH_SUCCESS);
  EXPECT_EQ(funcs->run_func(run_context), ge::GRAPH_SUCCESS);

  auto addr_start = static_cast<size_t>(kernel::CopyFlowLaunchOutputs::kAddress);
  auto tensor_data1 = run_context->GetOutputPointer<GertTensorData>(addr_start);
  ASSERT_NE(tensor_data1, nullptr);
  EXPECT_NE(tensor_data1->GetAddr(), tensor_holder1.GetTensor()->GetAddr());
  EXPECT_EQ(memcmp(tensor_data1->GetAddr(), tensor_holder1.GetTensor()->GetAddr(),
                   ge::GetSizeInBytes(shape1.GetStorageShape().GetShapeSize(), data_type1)),
            0);
  EXPECT_EQ(tensor_data1->GetPlacement(), kOnDeviceHbm);

  auto tensor_data2 = run_context->GetOutputPointer<GertTensorData>(addr_start + 1);
  ASSERT_NE(tensor_data2, nullptr);
  EXPECT_NE(tensor_data2->GetAddr(), tensor_holder2.GetTensor()->GetAddr());
  EXPECT_EQ(memcmp(tensor_data2->GetAddr(), tensor_holder2.GetTensor()->GetAddr(),
                   ge::GetSizeInBytes(shape1.GetStorageShape().GetShapeSize(), data_type1)),
            0);
  EXPECT_EQ(tensor_data2->GetPlacement(), kOnDeviceHbm);

  auto args = run_context->MutableInputPointer<gert::RtKernelLaunchArgsEx>(
      static_cast<size_t>(kernel::CopyFlowLaunchInputs::kRtArg));
  ASSERT_NE(args, nullptr);
  EXPECT_EQ(args->GetBase()->hostInputInfoNum, 0);
  context_holder.FreeAll();
}

// 1 host, 1device 输入， host输入大小大于kMaxHostInputDataLen, 不随路拷贝
TEST_F(CopyFlowLaunchKernelTest, NoCopyFlowLaunch_1H1D) {
  auto tensor_holder1 = TensorFaker().Placement(kOnHost).Shape({10, 20}).Build();
  auto shape1 = tensor_holder1.GetTensor()->GetShape();
  auto data_type1 = tensor_holder1.GetTensor()->GetDataType();
  auto size1 = ge::RoundUp(ge::GetSizeInBytes(shape1.GetStorageShape().GetShapeSize(), data_type1), 512);

  auto tensor_holder2 = TensorFaker().Placement(kOnDeviceHbm).Shape({10, 20}).Build();
  auto shape2 = tensor_holder2.GetTensor()->GetShape();
  auto data_type2 = tensor_holder2.GetTensor()->GetDataType();
  auto size2 = ge::RoundUp(ge::GetSizeInBytes(shape2.GetStorageShape().GetShapeSize(), data_type2), 512);

  // 这里指的是addr 的input out 个数，和 OptimizeHostInputs 输入输出有区别
  size_t input_num = 2;
  size_t output_num = 2;
  auto outputs = OutputsHolder::Fake(2);
  std::unique_ptr<uint8_t[]> launch_args = CreateLaunchArg(input_num, output_num);

  std::vector<std::vector<int32_t>> host_inputs_addr_index = {{0}, {1}};
  size_t cvv_total_size = 0U;
  auto const_index_data = GetContinuousVector2DByVector2D(host_inputs_addr_index, cvv_total_size);
  GertTensorData input_tensor_data1;
  TensorUtils::RefTdToGtd(tensor_holder1.GetTensor()->GetTensorData(), -1, input_tensor_data1);
  GertTensorData input_tensor_data2;
  TensorUtils::RefTdToGtd(tensor_holder2.GetTensor()->GetTensorData(), -1, input_tensor_data2);
  auto context_holder = KernelRunContextFaker()
                            .KernelIONum(13, 2)
                            .Inputs(std::vector<void *>(
                                {reinterpret_cast<void *>(input_num), reinterpret_cast<void *>(const_index_data.get()),
                                 reinterpret_cast<void *>(launch_args.get()), reinterpret_cast<void *>(0x11),
                                 &single_stream_l2_allocator_, &input_tensor_data1, reinterpret_cast<void *>(size1),
                                 &shape1, reinterpret_cast<void *>(data_type1), &input_tensor_data2,
                                 reinterpret_cast<void *>(size2), &shape2, reinterpret_cast<void *>(data_type2)}))
                            .Outputs(outputs.pointer)
                            .Build();

  auto run_context = context_holder.GetContext<KernelContext>();
  auto funcs = KernelRegistry::GetInstance().FindKernelFuncs(kernel::kCopyFlowLaunch);
  ASSERT_NE(funcs, nullptr);
  ASSERT_NE(funcs->outputs_creator, nullptr);
  ASSERT_NE(funcs->run_func, nullptr);

  ASSERT_EQ(funcs->outputs_creator(nullptr, run_context), ge::GRAPH_SUCCESS);
  EXPECT_EQ(funcs->run_func(run_context), ge::GRAPH_SUCCESS);
  auto addr_start = static_cast<size_t>(kernel::CopyFlowLaunchOutputs::kAddress);
  auto tensor_data1 = run_context->GetOutputPointer<GertTensorData>(addr_start);
  ASSERT_NE(tensor_data1, nullptr);
  EXPECT_NE(tensor_data1->GetAddr(), tensor_holder1.GetTensor()->GetAddr());
  EXPECT_EQ(memcmp(tensor_data1->GetAddr(), tensor_holder1.GetTensor()->GetAddr(),
                   ge::GetSizeInBytes(shape1.GetStorageShape().GetShapeSize(), data_type1)),
            0);
  EXPECT_EQ(tensor_data1->GetPlacement(), kOnDeviceHbm);

  auto tensor_data2 = run_context->GetOutputPointer<GertTensorData>(addr_start + 1);
  ASSERT_NE(tensor_data2, nullptr);
  EXPECT_EQ(tensor_data2->GetAddr(), tensor_holder2.GetTensor()->GetAddr());
  EXPECT_EQ(tensor_data2->GetSize(), tensor_holder2.GetTensor()->GetSize());
  EXPECT_EQ(tensor_data2->GetPlacement(), kOnDeviceHbm);

  auto args = run_context->MutableInputPointer<gert::RtKernelLaunchArgsEx>(
      static_cast<size_t>(kernel::CopyFlowLaunchInputs::kRtArg));
  ASSERT_NE(args, nullptr);
  EXPECT_EQ(args->GetBase()->hostInputInfoNum, 0);
  context_holder.FreeAll();
}

// 所有输入都是device输入，不需要随路拷贝
TEST_F(CopyFlowLaunchKernelTest, NoCopyFlowLaunch_2D) {
  auto tensor_holder1 = TensorFaker().Placement(kOnDeviceHbm).Shape({10, 20}).Build();
  auto shape1 = tensor_holder1.GetTensor()->GetShape();
  auto data_type1 = tensor_holder1.GetTensor()->GetDataType();
  auto size1 = ge::RoundUp(ge::GetSizeInBytes(shape1.GetStorageShape().GetShapeSize(), data_type1), 512);

  auto tensor_holder2 = TensorFaker().Placement(kOnDeviceHbm).Shape({10, 20}).Build();
  auto shape2 = tensor_holder2.GetTensor()->GetShape();
  auto data_type2 = tensor_holder2.GetTensor()->GetDataType();
  auto size2 = ge::RoundUp(ge::GetSizeInBytes(shape2.GetStorageShape().GetShapeSize(), data_type2), 512);

  // 这里指的是addr 的input out 个数，和 OptimizeHostInputs 输入输出有区别
  size_t input_num = 2;
  size_t output_num = 2;
  auto outputs = OutputsHolder::Fake(2);
  std::unique_ptr<uint8_t[]> launch_args = CreateLaunchArg(input_num, output_num);

  std::vector<std::vector<int32_t>> host_inputs_addr_index = {{0}, {1}};
  size_t cvv_total_size = 0U;
  auto const_index_data = GetContinuousVector2DByVector2D(host_inputs_addr_index, cvv_total_size);
  GertTensorData input_tensor_data1;
  TensorUtils::RefTdToGtd(tensor_holder1.GetTensor()->GetTensorData(), -1, input_tensor_data1);
  GertTensorData input_tensor_data2;
  TensorUtils::RefTdToGtd(tensor_holder2.GetTensor()->GetTensorData(), -1, input_tensor_data2);
  auto context_holder = KernelRunContextFaker()
                            .KernelIONum(13, 2)
                            .Inputs(std::vector<void *>(
                                {reinterpret_cast<void *>(input_num), reinterpret_cast<void *>(const_index_data.get()),
                                 reinterpret_cast<void *>(launch_args.get()), reinterpret_cast<void *>(0x11),
                                 &single_stream_l2_allocator_, &input_tensor_data1, reinterpret_cast<void *>(size1),
                                 &shape1, reinterpret_cast<void *>(data_type1), &input_tensor_data2,
                                 reinterpret_cast<void *>(size2), &shape2, reinterpret_cast<void *>(data_type2)}))
                            .Outputs(outputs.pointer)
                            .Build();

  auto run_context = context_holder.GetContext<KernelContext>();
  auto funcs = KernelRegistry::GetInstance().FindKernelFuncs(kernel::kCopyFlowLaunch);
  ASSERT_NE(funcs, nullptr);
  ASSERT_NE(funcs->outputs_creator, nullptr);
  ASSERT_NE(funcs->run_func, nullptr);

  ASSERT_EQ(funcs->outputs_creator(nullptr, run_context), ge::GRAPH_SUCCESS);
  EXPECT_EQ(funcs->run_func(run_context), ge::GRAPH_SUCCESS);
  auto addr_start = static_cast<size_t>(kernel::CopyFlowLaunchOutputs::kAddress);
  auto tensor_data1 = run_context->GetOutputPointer<GertTensorData>(addr_start);
  ASSERT_NE(tensor_data1, nullptr);
  EXPECT_EQ(tensor_data1->GetAddr(), tensor_holder1.GetTensor()->GetAddr());
  EXPECT_EQ(tensor_data1->GetSize(), tensor_holder1.GetTensor()->GetSize());
  EXPECT_EQ(tensor_data1->GetPlacement(), kOnDeviceHbm);

  auto tensor_data2 = run_context->GetOutputPointer<GertTensorData>(addr_start + 1);
  ASSERT_NE(tensor_data2, nullptr);
  EXPECT_EQ(tensor_data2->GetAddr(), tensor_holder2.GetTensor()->GetAddr());
  EXPECT_EQ(tensor_data2->GetSize(), tensor_holder2.GetTensor()->GetSize());
  EXPECT_EQ(tensor_data2->GetPlacement(), kOnDeviceHbm);

  auto args = run_context->MutableInputPointer<gert::RtKernelLaunchArgsEx>(
      static_cast<size_t>(kernel::CopyFlowLaunchInputs::kRtArg));
  ASSERT_NE(args, nullptr);
  EXPECT_EQ(args->GetBase()->hostInputInfoNum, 0);
  context_holder.FreeAll();
}

// 1、3、5 host输入， 2、4、6 device输入，满足随路拷贝条件，1、3、5正常随路拷贝
TEST_F(CopyFlowLaunchKernelTest, CopyFlowLaunch_3H3D) {
  auto tensor_holder1 = TensorFaker().Placement(kOnHost).Shape({}).DataType(ge::DT_INT32).Value<int32_t>({2}).Build();
  auto shape1 = tensor_holder1.GetTensor()->GetShape();
  auto data_type1 = tensor_holder1.GetTensor()->GetDataType();
  auto size1 = ge::RoundUp(ge::GetSizeInBytes(shape1.GetStorageShape().GetShapeSize(), data_type1), 512);
  GertTensorData input_tensor_data1;
  TensorUtils::RefTdToGtd(tensor_holder1.GetTensor()->GetTensorData(), -1, input_tensor_data1);

  auto tensor_holder2 = TensorFaker().Placement(kOnDeviceHbm).Shape({10, 20}).Build();
  auto shape2 = tensor_holder2.GetTensor()->GetShape();
  auto data_type2 = tensor_holder2.GetTensor()->GetDataType();
  auto size2 = ge::RoundUp(ge::GetSizeInBytes(shape2.GetStorageShape().GetShapeSize(), data_type2), 512);
  GertTensorData input_tensor_data2;
  TensorUtils::RefTdToGtd(tensor_holder2.GetTensor()->GetTensorData(), -1, input_tensor_data2);

  auto tensor_holder3 = TensorFaker().Placement(kOnHost).Shape({}).DataType(ge::DT_INT32).Value<int32_t>({4}).Build();
  auto shape3 = tensor_holder1.GetTensor()->GetShape();
  auto data_type3 = tensor_holder1.GetTensor()->GetDataType();
  auto size3 = ge::RoundUp(ge::GetSizeInBytes(shape1.GetStorageShape().GetShapeSize(), data_type1), 512);
  GertTensorData input_tensor_data3;
  TensorUtils::RefTdToGtd(tensor_holder3.GetTensor()->GetTensorData(), -1, input_tensor_data3);

  auto tensor_holder4 = TensorFaker().Placement(kOnDeviceHbm).Shape({10, 20}).Build();
  auto shape4 = tensor_holder2.GetTensor()->GetShape();
  auto data_type4 = tensor_holder2.GetTensor()->GetDataType();
  auto size4 = ge::RoundUp(ge::GetSizeInBytes(shape2.GetStorageShape().GetShapeSize(), data_type2), 512);
  GertTensorData input_tensor_data4;
  TensorUtils::RefTdToGtd(tensor_holder4.GetTensor()->GetTensorData(), -1, input_tensor_data4);

  auto tensor_holder5 = TensorFaker().Placement(kOnHost).Shape({}).DataType(ge::DT_INT32).Value<int32_t>({6}).Build();
  auto shape5 = tensor_holder1.GetTensor()->GetShape();
  auto data_type5 = tensor_holder1.GetTensor()->GetDataType();
  auto size5 = ge::RoundUp(ge::GetSizeInBytes(shape1.GetStorageShape().GetShapeSize(), data_type1), 512);
  GertTensorData input_tensor_data5;
  TensorUtils::RefTdToGtd(tensor_holder5.GetTensor()->GetTensorData(), -1, input_tensor_data5);

  auto tensor_holder6 = TensorFaker().Placement(kOnDeviceHbm).Shape({10, 20}).Build();
  auto shape6 = tensor_holder2.GetTensor()->GetShape();
  auto data_type6 = tensor_holder2.GetTensor()->GetDataType();
  auto size6 = ge::RoundUp(ge::GetSizeInBytes(shape2.GetStorageShape().GetShapeSize(), data_type2), 512);
  GertTensorData input_tensor_data6;
  TensorUtils::RefTdToGtd(tensor_holder6.GetTensor()->GetTensorData(), -1, input_tensor_data6);

  // 这里指的是addr 的input out 个数，和 CopyFlowLaunchInputs 输入输出有区别
  size_t input_num = 6;
  size_t output_num = 6;
  auto outputs = OutputsHolder::Fake(output_num);
  std::unique_ptr<uint8_t[]> launch_args = CreateLaunchArg(input_num, output_num);

  std::vector<std::vector<int32_t>> host_inputs_addr_index = {{0}, {1}, {2}, {3}, {4}, {5}};
  size_t cvv_total_size = 0U;
  auto const_index_data = GetContinuousVector2DByVector2D(host_inputs_addr_index, cvv_total_size);
  const size_t kernel_input_num =
      static_cast<size_t>(kernel::CopyFlowLaunchInputs::kAddrAndLengthStart) + 4 * output_num;
  auto context_holder = KernelRunContextFaker()
                            .KernelIONum(kernel_input_num, output_num)
                            .Inputs(std::vector<void *>({
                                reinterpret_cast<void *>(input_num),
                                reinterpret_cast<void *>(const_index_data.get()),
                                reinterpret_cast<void *>(launch_args.get()),
                                reinterpret_cast<void *>(0x11),
                                &single_stream_l2_allocator_,
                                &input_tensor_data1,
                                reinterpret_cast<void *>(size1),
                                &shape1,
                                reinterpret_cast<void *>(data_type1),
                                &input_tensor_data2,
                                reinterpret_cast<void *>(size2),
                                &shape2,
                                reinterpret_cast<void *>(data_type2),
                                &input_tensor_data3,
                                reinterpret_cast<void *>(size3),
                                &shape3,
                                reinterpret_cast<void *>(data_type3),
                                &input_tensor_data4,
                                reinterpret_cast<void *>(size4),
                                &shape4,
                                reinterpret_cast<void *>(data_type4),
                                &input_tensor_data5,
                                reinterpret_cast<void *>(size5),
                                &shape5,
                                reinterpret_cast<void *>(data_type5),
                                &input_tensor_data6,
                                reinterpret_cast<void *>(size6),
                                &shape6,
                                reinterpret_cast<void *>(data_type6),
                            }))
                            .Outputs(outputs.pointer)
                            .Build();

  auto run_context = context_holder.GetContext<KernelContext>();
  auto funcs = KernelRegistry::GetInstance().FindKernelFuncs(kernel::kCopyFlowLaunch);
  ASSERT_NE(funcs, nullptr);
  ASSERT_NE(funcs->outputs_creator, nullptr);
  ASSERT_NE(funcs->run_func, nullptr);

  ASSERT_EQ(funcs->outputs_creator(nullptr, run_context), ge::GRAPH_SUCCESS);
  EXPECT_EQ(funcs->run_func(run_context), ge::GRAPH_SUCCESS);
  auto addr_start = static_cast<size_t>(kernel::CopyFlowLaunchOutputs::kAddress);

  auto tensor_data2 = run_context->GetOutputPointer<GertTensorData>(addr_start + 1);
  ASSERT_NE(tensor_data2, nullptr);
  EXPECT_EQ(tensor_data2->GetAddr(), tensor_holder2.GetTensor()->GetAddr());
  EXPECT_EQ(tensor_data2->GetSize(), tensor_holder2.GetTensor()->GetSize());
  EXPECT_EQ(tensor_data2->GetPlacement(), kOnDeviceHbm);

  auto tensor_data4 = run_context->GetOutputPointer<GertTensorData>(addr_start + 3);
  ASSERT_NE(tensor_data4, nullptr);
  EXPECT_EQ(tensor_data4->GetAddr(), tensor_holder4.GetTensor()->GetAddr());
  EXPECT_EQ(tensor_data4->GetSize(), tensor_holder4.GetTensor()->GetSize());
  EXPECT_EQ(tensor_data4->GetPlacement(), kOnDeviceHbm);

  auto tensor_data6 = run_context->GetOutputPointer<GertTensorData>(addr_start + 5);
  ASSERT_NE(tensor_data6, nullptr);
  EXPECT_EQ(tensor_data6->GetAddr(), tensor_holder6.GetTensor()->GetAddr());
  EXPECT_EQ(tensor_data6->GetSize(), tensor_holder6.GetTensor()->GetSize());
  EXPECT_EQ(tensor_data6->GetPlacement(), kOnDeviceHbm);

  auto args = run_context->MutableInputPointer<gert::RtKernelLaunchArgsEx>(
      static_cast<size_t>(kernel::CopyFlowLaunchInputs::kRtArg));
  ASSERT_NE(args, nullptr);
  EXPECT_EQ(args->GetBase()->hostInputInfoNum, 3);

  int32_t host_input0_value =
      *(reinterpret_cast<int32_t *>(args->GetArgsPointer<void>(RtKernelLaunchArgsEx::ArgsType::kHostInputData)));
  EXPECT_EQ(host_input0_value, 2);
  auto host_intput1_addr =
      ge::ValueToPtr((ge::PtrToValue(args->GetArgBase()) + args->GetBase()->hostInputInfoPtr[1].dataOffset));
  int32_t host_intput1_value = *(reinterpret_cast<int32_t *>(host_intput1_addr));
  EXPECT_EQ(host_intput1_value, 4);
  auto host_intput2_addr =
      ge::ValueToPtr((ge::PtrToValue(args->GetArgBase()) + args->GetBase()->hostInputInfoPtr[2].dataOffset));
  int32_t host_intput2_value = *(reinterpret_cast<int32_t *>(host_intput2_addr));
  EXPECT_EQ(host_intput2_value, 6);
  context_holder.FreeAll();
}

// 1 host输入，满足随路拷贝条件，datatype为DT_INT8, 输入size不满足4字节对齐，可以正常随路拷贝
TEST_F(CopyFlowLaunchKernelTest, CopyFlowLaunch_HostNotAlign4Byte) {
  // 这里构造输入host tensor,
  // shape为{3}，不满足4字节对齐，同时tensordata构造了4个字节，用于最终校验是否会按照实际shape拷贝
  auto tensor_holder1 =
      TensorFaker().Placement(kOnHost).Shape({3}).DataType(ge::DT_INT8).Value<char>({'1', '2', '3', '4'}).Build();
  auto shape1 = tensor_holder1.GetTensor()->GetShape();
  auto data_type1 = tensor_holder1.GetTensor()->GetDataType();
  auto size1 = ge::RoundUp(ge::GetSizeInBytes(shape1.GetStorageShape().GetShapeSize(), data_type1), 32);
  auto outputs = OutputsHolder::Fake(1);

  std::vector<std::vector<int32_t>> host_inputs_addr_index = {{0}};
  size_t cvv_total_size = 0U;
  auto const_index_data = GetContinuousVector2DByVector2D(host_inputs_addr_index, cvv_total_size);
  // 这里指的是addr 的input out 个数，和 CopyFlowLaunch 输入输出有区别
  size_t input_num = 1;
  size_t output_num = 1;
  std::unique_ptr<uint8_t[]> launch_args = CreateLaunchArg(input_num, output_num);
  GertTensorData input_tensor_data1;
  TensorUtils::RefTdToGtd(tensor_holder1.GetTensor()->GetTensorData(), -1, input_tensor_data1);
  auto context_holder = KernelRunContextFaker()
                            .KernelIONum(9, 1)
                            .Inputs(std::vector<void *>(
                                {reinterpret_cast<void *>(input_num), reinterpret_cast<void *>(const_index_data.get()),
                                 reinterpret_cast<void *>(launch_args.get()), reinterpret_cast<void *>(0x11),
                                 &single_stream_l2_allocator_, &input_tensor_data1, reinterpret_cast<void *>(size1),
                                 &shape1, reinterpret_cast<void *>(data_type1)}))
                            .Outputs(outputs.pointer)
                            .Build();

  auto run_context = context_holder.GetContext<KernelContext>();
  auto funcs = KernelRegistry::GetInstance().FindKernelFuncs(kernel::kCopyFlowLaunch);
  ASSERT_NE(funcs, nullptr);
  ASSERT_NE(funcs->outputs_creator, nullptr);
  ASSERT_NE(funcs->run_func, nullptr);

  ASSERT_EQ(funcs->outputs_creator(nullptr, run_context), ge::GRAPH_SUCCESS);
  EXPECT_EQ(funcs->run_func(run_context), ge::GRAPH_SUCCESS);

  auto args = run_context->MutableInputPointer<gert::RtKernelLaunchArgsEx>(
      static_cast<size_t>(kernel::CopyFlowLaunchInputs::kRtArg));
  ASSERT_NE(args, nullptr);
  EXPECT_EQ(args->GetBase()->hostInputInfoNum, 1);
  EXPECT_EQ(args->GetBase()->hostInputInfoPtr[0].dataOffset,
            GetOffset(args->GetArgBase(), args->GetArgsPointer<void>(RtKernelLaunchArgsEx::ArgsType::kHostInputData)));

  std::string input0_value =
      std::string(reinterpret_cast<char *>(args->GetArgsPointer<void>(RtKernelLaunchArgsEx::ArgsType::kHostInputData)));
  // 按照实际shape来拷贝，只拷贝3个字节
  EXPECT_EQ(input0_value, "123");

  context_holder.FreeAll();
}
// 测试 CalcCopyFlowAllocSizes - 全部 device 输入，alloc size 全为 0
TEST_F(CopyFlowLaunchKernelTest, CalcCopyFlowAllocSizes_AllDeviceInputs) {
  auto tensor_holder1 = TensorFaker().Placement(kOnDeviceHbm).Shape({10, 20}).Build();
  auto shape1 = tensor_holder1.GetTensor()->GetShape();
  auto data_type1 = tensor_holder1.GetTensor()->GetDataType();
  auto size1 = ge::RoundUp(ge::GetSizeInBytes(shape1.GetStorageShape().GetShapeSize(), data_type1), 512);

  size_t output_num = 1;
  auto launch_args = CreateLaunchArg(output_num, output_num);
  GertTensorData input_tensor_data1;
  TensorUtils::RefTdToGtd(tensor_holder1.GetTensor()->GetTensorData(), -1, input_tensor_data1);

  auto outputs = OutputsHolder::Fake(1);
  auto context_holder =
      KernelRunContextFaker()
          .KernelIONum(static_cast<size_t>(kernel::CalcCopyFlowAllocSizesInputs::kAddrAndLengthStart) + 4, 1)
          .Inputs(std::vector<void *>({reinterpret_cast<void *>(output_num), launch_args.get(), &input_tensor_data1,
                                       reinterpret_cast<void *>(size1), &shape1, reinterpret_cast<void *>(data_type1)}))
          .Outputs(outputs.pointer)
          .Build();

  auto run_context = context_holder.GetContext<KernelContext>();
  auto funcs = KernelRegistry::GetInstance().FindKernelFuncs(kernel::kCalcCopyFlowAllocSizes);
  ASSERT_NE(funcs, nullptr);
  ASSERT_NE(funcs->outputs_creator, nullptr);
  ASSERT_NE(funcs->run_func, nullptr);

  auto graph = std::make_shared<ge::ExecuteGraph>("calc_copy_flow_all_device");
  auto node = FastNodeFaker(graph).Attr("copy_flow_count", static_cast<int64_t>(output_num)).Build();
  ASSERT_EQ(funcs->outputs_creator(node, run_context), ge::GRAPH_SUCCESS);
  EXPECT_EQ(funcs->run_func(run_context), ge::GRAPH_SUCCESS);

  auto alloc_sizes = run_context->GetOutputPointer<TypedContinuousVector<size_t>>(0U);
  ASSERT_NE(alloc_sizes, nullptr);
  EXPECT_EQ(alloc_sizes->GetSize(), output_num);
  EXPECT_EQ(alloc_sizes->GetData()[0], 0U);

  context_holder.FreeAll();
}

TEST_F(CopyFlowLaunchKernelTest, CalcCopyFlowAllocSizes_CapacityMatchesCopyFlowCount) {
  auto funcs = KernelRegistry::GetInstance().FindKernelFuncs(kernel::kCalcCopyFlowAllocSizes);
  ASSERT_NE(funcs, nullptr);
  ASSERT_NE(funcs->outputs_creator, nullptr);

  for (const int64_t copy_flow_count : {1, 16, 17, 32}) {
    auto outputs = OutputsHolder::Fake(1);
    auto context_holder = KernelRunContextFaker().KernelIONum(0, 1).Outputs(outputs.pointer).Build();
    auto run_context = context_holder.GetContext<KernelContext>();
    auto graph = std::make_shared<ge::ExecuteGraph>("copy_flow_count_test");
    auto node = FastNodeFaker(graph)
                    .NameAndType("calc_copy_flow_alloc_sizes", kernel::kCalcCopyFlowAllocSizes)
                    .Attr("copy_flow_count", copy_flow_count)
                    .Build();
    ASSERT_NE(node, nullptr);

    ASSERT_EQ(funcs->outputs_creator(node, run_context), ge::GRAPH_SUCCESS);
    auto alloc_sizes = run_context->GetOutputPointer<TypedContinuousVector<size_t>>(0U);
    ASSERT_NE(alloc_sizes, nullptr);
    EXPECT_EQ(alloc_sizes->GetCapacity(), static_cast<size_t>(copy_flow_count));
    context_holder.FreeAll();
  }
}

TEST_F(CopyFlowLaunchKernelTest, CalcCopyFlowAllocSizes_RejectsInvalidCopyFlowCount) {
  auto funcs = KernelRegistry::GetInstance().FindKernelFuncs(kernel::kCalcCopyFlowAllocSizes);
  ASSERT_NE(funcs, nullptr);
  ASSERT_NE(funcs->outputs_creator, nullptr);

  for (const bool set_zero_count : {false, true}) {
    auto outputs = OutputsHolder::Fake(1);
    auto context_holder = KernelRunContextFaker().KernelIONum(0, 1).Outputs(outputs.pointer).Build();
    auto run_context = context_holder.GetContext<KernelContext>();
    auto graph = std::make_shared<ge::ExecuteGraph>("invalid_copy_flow_count_test");
    FastNodeFaker node_faker(graph);
    if (set_zero_count) {
      node_faker.Attr("copy_flow_count", static_cast<int64_t>(0));
    }
    auto node = node_faker.Build();
    ASSERT_NE(node, nullptr);
    EXPECT_NE(funcs->outputs_creator(node, run_context), ge::GRAPH_SUCCESS);
    context_holder.FreeAll();
  }
}

// 测试 CalcCopyFlowAllocSizes - 全部 host 输入且 size 小于 kMaxHostInputDataLen，均随路拷贝，alloc size 全为 0
TEST_F(CopyFlowLaunchKernelTest, CalcCopyFlowAllocSizes_HostWithinLimit) {
  auto tensor_holder1 = TensorFaker().Placement(kOnHost).Shape({}).DataType(ge::DT_INT32).Value<int32_t>({2}).Build();
  auto shape1 = tensor_holder1.GetTensor()->GetShape();
  auto data_type1 = tensor_holder1.GetTensor()->GetDataType();
  auto size1 = ge::RoundUp(ge::GetSizeInBytes(shape1.GetStorageShape().GetShapeSize(), data_type1), 512);

  auto tensor_holder2 = TensorFaker().Placement(kOnHost).Shape({}).DataType(ge::DT_INT32).Value<int32_t>({5}).Build();
  auto shape2 = tensor_holder2.GetTensor()->GetShape();
  auto data_type2 = tensor_holder2.GetTensor()->GetDataType();
  auto size2 = ge::RoundUp(ge::GetSizeInBytes(shape2.GetStorageShape().GetShapeSize(), data_type2), 512);

  size_t output_num = 2;
  auto launch_args = CreateLaunchArg(output_num, output_num);
  GertTensorData input_tensor_data1;
  TensorUtils::RefTdToGtd(tensor_holder1.GetTensor()->GetTensorData(), -1, input_tensor_data1);
  GertTensorData input_tensor_data2;
  TensorUtils::RefTdToGtd(tensor_holder2.GetTensor()->GetTensorData(), -1, input_tensor_data2);

  auto outputs = OutputsHolder::Fake(1);
  auto context_holder =
      KernelRunContextFaker()
          .KernelIONum(static_cast<size_t>(kernel::CalcCopyFlowAllocSizesInputs::kAddrAndLengthStart) + 8, 1)
          .Inputs(std::vector<void *>({reinterpret_cast<void *>(output_num), launch_args.get(), &input_tensor_data1,
                                       reinterpret_cast<void *>(size1), &shape1, reinterpret_cast<void *>(data_type1),
                                       &input_tensor_data2, reinterpret_cast<void *>(size2), &shape2,
                                       reinterpret_cast<void *>(data_type2)}))
          .Outputs(outputs.pointer)
          .Build();

  auto run_context = context_holder.GetContext<KernelContext>();
  auto funcs = KernelRegistry::GetInstance().FindKernelFuncs(kernel::kCalcCopyFlowAllocSizes);
  ASSERT_NE(funcs, nullptr);
  ASSERT_NE(funcs->outputs_creator, nullptr);
  ASSERT_NE(funcs->run_func, nullptr);

  auto graph = std::make_shared<ge::ExecuteGraph>("calc_copy_flow_host_within_limit");
  auto node = FastNodeFaker(graph).Attr("copy_flow_count", static_cast<int64_t>(output_num)).Build();
  ASSERT_EQ(funcs->outputs_creator(node, run_context), ge::GRAPH_SUCCESS);
  EXPECT_EQ(funcs->run_func(run_context), ge::GRAPH_SUCCESS);

  auto alloc_sizes = run_context->GetOutputPointer<TypedContinuousVector<size_t>>(0U);
  ASSERT_NE(alloc_sizes, nullptr);
  EXPECT_EQ(alloc_sizes->GetSize(), output_num);
  EXPECT_EQ(alloc_sizes->GetData()[0], 0U);
  EXPECT_EQ(alloc_sizes->GetData()[1], 0U);

  context_holder.FreeAll();
}

// 测试 CalcCopyFlowAllocSizes - host 输入超过 kMaxHostInputDataLen，需要独立 HBM 分配
TEST_F(CopyFlowLaunchKernelTest, CalcCopyFlowAllocSizes_HostExceedLimit) {
  auto tensor_holder1 = TensorFaker().Placement(kOnHost).Shape({10, 20}).Build();
  auto shape1 = tensor_holder1.GetTensor()->GetShape();
  auto data_type1 = tensor_holder1.GetTensor()->GetDataType();
  auto size1 = ge::RoundUp(ge::GetSizeInBytes(shape1.GetStorageShape().GetShapeSize(), data_type1), 512);

  size_t output_num = 1;
  auto launch_args = CreateLaunchArg(output_num, output_num);
  GertTensorData input_tensor_data1;
  TensorUtils::RefTdToGtd(tensor_holder1.GetTensor()->GetTensorData(), -1, input_tensor_data1);

  auto outputs = OutputsHolder::Fake(1);
  auto context_holder =
      KernelRunContextFaker()
          .KernelIONum(static_cast<size_t>(kernel::CalcCopyFlowAllocSizesInputs::kAddrAndLengthStart) + 4, 1)
          .Inputs(std::vector<void *>({reinterpret_cast<void *>(output_num), launch_args.get(), &input_tensor_data1,
                                       reinterpret_cast<void *>(size1), &shape1, reinterpret_cast<void *>(data_type1)}))
          .Outputs(outputs.pointer)
          .Build();

  auto run_context = context_holder.GetContext<KernelContext>();
  auto funcs = KernelRegistry::GetInstance().FindKernelFuncs(kernel::kCalcCopyFlowAllocSizes);
  ASSERT_NE(funcs, nullptr);
  ASSERT_NE(funcs->outputs_creator, nullptr);
  ASSERT_NE(funcs->run_func, nullptr);

  auto graph = std::make_shared<ge::ExecuteGraph>("calc_copy_flow_host_exceed_limit");
  auto node = FastNodeFaker(graph).Attr("copy_flow_count", static_cast<int64_t>(output_num)).Build();
  ASSERT_EQ(funcs->outputs_creator(node, run_context), ge::GRAPH_SUCCESS);
  EXPECT_EQ(funcs->run_func(run_context), ge::GRAPH_SUCCESS);

  auto alloc_sizes = run_context->GetOutputPointer<TypedContinuousVector<size_t>>(0U);
  ASSERT_NE(alloc_sizes, nullptr);
  EXPECT_EQ(alloc_sizes->GetSize(), output_num);
  EXPECT_EQ(alloc_sizes->GetData()[0], size1);

  context_holder.FreeAll();
}

// 测试 CalcCopyFlowAllocSizes - 混合 device 和 host 输入，第一个 host 小于 limit，第二个 host 超限
TEST_F(CopyFlowLaunchKernelTest, CalcCopyFlowAllocSizes_MixedDeviceAndHostInputs) {
  auto tensor_holder1 = TensorFaker().Placement(kOnDeviceHbm).Shape({10, 20}).Build();
  auto shape1 = tensor_holder1.GetTensor()->GetShape();
  auto data_type1 = tensor_holder1.GetTensor()->GetDataType();
  auto size1 = ge::RoundUp(ge::GetSizeInBytes(shape1.GetStorageShape().GetShapeSize(), data_type1), 512);
  GertTensorData input_tensor_data1;
  TensorUtils::RefTdToGtd(tensor_holder1.GetTensor()->GetTensorData(), -1, input_tensor_data1);

  auto tensor_holder2 = TensorFaker().Placement(kOnHost).Shape({}).DataType(ge::DT_INT32).Value<int32_t>({3}).Build();
  auto shape2 = tensor_holder2.GetTensor()->GetShape();
  auto data_type2 = tensor_holder2.GetTensor()->GetDataType();
  auto size2 = ge::RoundUp(ge::GetSizeInBytes(shape2.GetStorageShape().GetShapeSize(), data_type2), 512);
  GertTensorData input_tensor_data2;
  TensorUtils::RefTdToGtd(tensor_holder2.GetTensor()->GetTensorData(), -1, input_tensor_data2);

  auto tensor_holder3 = TensorFaker().Placement(kOnHost).Shape({10, 20}).Build();
  auto shape3 = tensor_holder3.GetTensor()->GetShape();
  auto data_type3 = tensor_holder3.GetTensor()->GetDataType();
  auto size3 = ge::RoundUp(ge::GetSizeInBytes(shape3.GetStorageShape().GetShapeSize(), data_type3), 512);
  GertTensorData input_tensor_data3;
  TensorUtils::RefTdToGtd(tensor_holder3.GetTensor()->GetTensorData(), -1, input_tensor_data3);

  size_t output_num = 3;
  auto launch_args = CreateLaunchArg(output_num, output_num);

  auto outputs = OutputsHolder::Fake(1);
  auto context_holder =
      KernelRunContextFaker()
          .KernelIONum(static_cast<size_t>(kernel::CalcCopyFlowAllocSizesInputs::kAddrAndLengthStart) + 12, 1)
          .Inputs(std::vector<void *>({reinterpret_cast<void *>(output_num), launch_args.get(), &input_tensor_data1,
                                       reinterpret_cast<void *>(size1), &shape1, reinterpret_cast<void *>(data_type1),
                                       &input_tensor_data2, reinterpret_cast<void *>(size2), &shape2,
                                       reinterpret_cast<void *>(data_type2), &input_tensor_data3,
                                       reinterpret_cast<void *>(size3), &shape3, reinterpret_cast<void *>(data_type3)}))
          .Outputs(outputs.pointer)
          .Build();

  auto run_context = context_holder.GetContext<KernelContext>();
  auto funcs = KernelRegistry::GetInstance().FindKernelFuncs(kernel::kCalcCopyFlowAllocSizes);
  ASSERT_NE(funcs, nullptr);
  ASSERT_NE(funcs->outputs_creator, nullptr);
  ASSERT_NE(funcs->run_func, nullptr);

  auto graph = std::make_shared<ge::ExecuteGraph>("calc_copy_flow_mixed_inputs");
  auto node = FastNodeFaker(graph).Attr("copy_flow_count", static_cast<int64_t>(output_num)).Build();
  ASSERT_EQ(funcs->outputs_creator(node, run_context), ge::GRAPH_SUCCESS);
  EXPECT_EQ(funcs->run_func(run_context), ge::GRAPH_SUCCESS);

  auto alloc_sizes = run_context->GetOutputPointer<TypedContinuousVector<size_t>>(0U);
  ASSERT_NE(alloc_sizes, nullptr);
  EXPECT_EQ(alloc_sizes->GetSize(), output_num);
  EXPECT_EQ(alloc_sizes->GetData()[0], 0U);     // device input
  EXPECT_EQ(alloc_sizes->GetData()[1], 0U);     // small host input within limit
  EXPECT_EQ(alloc_sizes->GetData()[2], size3);  // large host input exceeds limit

  context_holder.FreeAll();
}

TEST_F(CopyFlowLaunchKernelTest, CalcCopyFlowAllocSizes_InputNumMismatch) {
  auto tensor_holder1 = TensorFaker().Placement(kOnDeviceHbm).Shape({10, 20}).Build();
  auto shape1 = tensor_holder1.GetTensor()->GetShape();
  auto data_type1 = tensor_holder1.GetTensor()->GetDataType();
  auto size1 = ge::RoundUp(ge::GetSizeInBytes(shape1.GetStorageShape().GetShapeSize(), data_type1), 512);
  GertTensorData input_tensor_data1;
  TensorUtils::RefTdToGtd(tensor_holder1.GetTensor()->GetTensorData(), -1, input_tensor_data1);

  size_t output_num = 1;
  auto launch_args = CreateLaunchArg(output_num, output_num);
  auto outputs = OutputsHolder::Fake(1);
  size_t fewer_inputs = static_cast<size_t>(kernel::CalcCopyFlowAllocSizesInputs::kAddrAndLengthStart) + 2;
  auto context_holder = KernelRunContextFaker()
                            .KernelIONum(fewer_inputs, 1)
                            .Inputs(std::vector<void *>({reinterpret_cast<void *>(output_num), launch_args.get(),
                                                         &input_tensor_data1, reinterpret_cast<void *>(size1)}))
                            .Outputs(outputs.pointer)
                            .Build();

  auto run_context = context_holder.GetContext<KernelContext>();
  auto funcs = KernelRegistry::GetInstance().FindKernelFuncs(kernel::kCalcCopyFlowAllocSizes);
  ASSERT_NE(funcs, nullptr);
  ASSERT_NE(funcs->outputs_creator, nullptr);
  ASSERT_NE(funcs->run_func, nullptr);

  auto graph = std::make_shared<ge::ExecuteGraph>("calc_copy_flow_input_mismatch");
  auto node = FastNodeFaker(graph).Attr("copy_flow_count", static_cast<int64_t>(output_num)).Build();
  ASSERT_EQ(funcs->outputs_creator(node, run_context), ge::GRAPH_SUCCESS);
  GertRuntimeStub runtime_stub;
  runtime_stub.GetSlogStub().NoConsoleOut().SetLevelInfo();
  EXPECT_EQ(funcs->run_func(run_context), ge::GRAPH_FAILED);

  context_holder.FreeAll();
}

TEST_F(CopyFlowLaunchKernelTest, PrepareAndLaunchCopyFlowRegistrationContracts) {
  const auto prepare_funcs = registry.FindKernelFuncs("PrepareCopyFlowResult");
  ASSERT_NE(prepare_funcs, nullptr);
  EXPECT_NE(prepare_funcs->outputs_creator, nullptr);
  EXPECT_NE(prepare_funcs->run_func, nullptr);
  const auto prepare_info = registry.FindKernelInfo("PrepareCopyFlowResult");
  ASSERT_NE(prepare_info, nullptr);
  EXPECT_EQ(prepare_info->critical_section, kKernelUseMemory);

  const auto launch_funcs = registry.FindKernelFuncs("LaunchCopyFlowH2D");
  ASSERT_NE(launch_funcs, nullptr);
  ASSERT_NE(launch_funcs->outputs_creator, nullptr);
  EXPECT_NE(launch_funcs->run_func, nullptr);
  const auto launch_info = registry.FindKernelInfo("LaunchCopyFlowH2D");
  ASSERT_NE(launch_info, nullptr);
  EXPECT_EQ(launch_info->critical_section, kKernelLaunch);

  auto zero_output_context_holder = KernelRunContextFaker().KernelIONum(0U, 0U).Build();
  auto zero_output_context = zero_output_context_holder.GetContext<KernelContext>();
  ASSERT_NE(zero_output_context, nullptr);
  EXPECT_EQ(zero_output_context->GetOutputNum(), 0U);
  EXPECT_EQ(launch_funcs->outputs_creator(nullptr, zero_output_context), ge::GRAPH_SUCCESS);
  EXPECT_EQ(zero_output_context->GetOutputNum(), 0U);
  zero_output_context_holder.FreeAll();
}

TEST_F(CopyFlowLaunchKernelTest, PrepareCopyFlowResult_MixedInputsMaterializesWithoutMemcpy) {
  auto separate_host = TensorFaker().Placement(kOnHost).Shape({}).DataType(ge::DT_INT32).Value<int32_t>({7}).Build();
  auto merged_host =
      TensorFaker().Placement(kOnHost).Shape({3}).DataType(ge::DT_INT8).Value<char>({'1', '2', '3'}).Build();
  auto device = TensorFaker().Placement(kOnDeviceHbm).Shape({1}).DataType(ge::DT_INT32).Build();
  auto zero_host = TensorFaker().Placement(kOnHost).Shape({}).Build();
  zero_host.GetTensor()->MutableTensorData().SetAddr(nullptr, nullptr);
  zero_host.GetTensor()->MutableOriginShape().AppendDim(0U);
  zero_host.GetTensor()->MutableStorageShape().AppendDim(0U);
  zero_host.GetTensor()->SetSize(0U);

  auto separate_shape = separate_host.GetTensor()->GetShape();
  auto merged_shape = merged_host.GetTensor()->GetShape();
  auto device_shape = device.GetTensor()->GetShape();
  auto zero_shape = zero_host.GetTensor()->GetShape();
  const auto separate_type = separate_host.GetTensor()->GetDataType();
  const auto merged_type = merged_host.GetTensor()->GetDataType();
  const auto device_type = device.GetTensor()->GetDataType();
  const auto zero_type = zero_host.GetTensor()->GetDataType();
  GertTensorData separate_src;
  GertTensorData merged_src;
  GertTensorData device_src;
  GertTensorData zero_src;
  TensorUtils::RefTdToGtd(separate_host.GetTensor()->GetTensorData(), -1, separate_src);
  TensorUtils::RefTdToGtd(merged_host.GetTensor()->GetTensorData(), -1, merged_src);
  TensorUtils::RefTdToGtd(device.GetTensor()->GetTensorData(), -1, device_src);
  TensorUtils::RefTdToGtd(zero_host.GetTensor()->GetTensorData(), -1, zero_src);

  std::vector<uint8_t> separate_dst(32U, 0U);
  std::vector<uint8_t> zero_dst(32U, 0U);
  GertTensorData separate_alloc = {separate_dst.data(), separate_dst.size(), kOnDeviceHbm, -1};
  GertTensorData merged_alloc = {nullptr, 0U, kOnDeviceHbm, -1};
  GertTensorData device_alloc = {nullptr, 0U, kOnDeviceHbm, -1};
  GertTensorData zero_alloc = {zero_dst.data(), zero_dst.size(), kOnDeviceHbm, -1};
  auto allocated_addrs =
      ContinuousVectorBuilder::Create<GertTensorData *>({&separate_alloc, &merged_alloc, &device_alloc, &zero_alloc});

  constexpr size_t kCopyFlowCount = 4U;
  std::vector<std::vector<int32_t>> input_indexes = {{0}, {1}, {2}, {3}};
  size_t cvv_total_size = 0U;
  auto indexes = GetContinuousVector2DByVector2D(input_indexes, cvv_total_size);
  auto launch_args = CreateLaunchArg(kCopyFlowCount, kCopyFlowCount);
  auto outputs = OutputsHolder::Fake(kCopyFlowCount);
  auto context_holder = KernelRunContextFaker()
                            .KernelIONum(4U + kCopyFlowCount * 4U, kCopyFlowCount)
                            .Inputs({reinterpret_cast<void *>(kCopyFlowCount),
                                     indexes.get(),
                                     launch_args.get(),
                                     allocated_addrs.get(),
                                     &separate_src,
                                     reinterpret_cast<void *>(separate_host.GetTensor()->GetSize()),
                                     &separate_shape,
                                     reinterpret_cast<void *>(separate_type),
                                     &merged_src,
                                     reinterpret_cast<void *>(merged_host.GetTensor()->GetSize()),
                                     &merged_shape,
                                     reinterpret_cast<void *>(merged_type),
                                     &device_src,
                                     reinterpret_cast<void *>(device.GetTensor()->GetSize()),
                                     &device_shape,
                                     reinterpret_cast<void *>(device_type),
                                     &zero_src,
                                     reinterpret_cast<void *>(zero_host.GetTensor()->GetSize()),
                                     &zero_shape,
                                     reinterpret_cast<void *>(zero_type)})
                            .Outputs(outputs.pointer)
                            .Build();
  auto run_context = context_holder.GetContext<KernelContext>();
  auto funcs = registry.FindKernelFuncs("PrepareCopyFlowResult");
  ASSERT_NE(funcs, nullptr);

  auto graph = std::make_shared<ge::ExecuteGraph>("prepare_copy_flow_result");
  auto node = FastNodeFaker(graph).Attr("copy_flow_count", static_cast<int64_t>(kCopyFlowCount)).Build();
  ASSERT_EQ(funcs->outputs_creator(node, run_context), ge::GRAPH_SUCCESS);
  auto acl_runtime_stub = std::make_shared<MockCopyFlowAclRuntime>();
  ge::AclRuntimeStub::SetInstance(acl_runtime_stub);
  EXPECT_CALL(*acl_runtime_stub, aclrtMemcpyAsync).Times(0);
  const auto result = funcs->run_func(run_context);
  ge::AclRuntimeStub::Reset();
  ASSERT_EQ(result, ge::GRAPH_SUCCESS);

  auto separate_result = run_context->GetOutputPointer<GertTensorData>(0U);
  auto merged_result = run_context->GetOutputPointer<GertTensorData>(1U);
  auto device_result = run_context->GetOutputPointer<GertTensorData>(2U);
  auto zero_result = run_context->GetOutputPointer<GertTensorData>(3U);
  ASSERT_NE(separate_result, nullptr);
  ASSERT_NE(merged_result, nullptr);
  ASSERT_NE(device_result, nullptr);
  ASSERT_NE(zero_result, nullptr);
  EXPECT_EQ(separate_result->GetAddr(), separate_alloc.GetAddr());
  EXPECT_EQ(merged_result->GetAddr(), nullptr);
  EXPECT_EQ(device_result->GetAddr(), device_src.GetAddr());
  EXPECT_EQ(zero_result->GetAddr(), zero_alloc.GetAddr());

  auto args = reinterpret_cast<RtKernelLaunchArgsEx *>(launch_args.get());
  EXPECT_EQ(args->GetBase()->hostInputInfoNum, 1U);
  EXPECT_EQ(args->GetHostInputDataSize(), 64U);
  EXPECT_EQ(memcmp(args->GetArgsPointer<void>(RtKernelLaunchArgsEx::ArgsType::kHostInputData), "123", 3U), 0);
  context_holder.FreeAll();
}

TEST_F(CopyFlowLaunchKernelTest, LaunchCopyFlowH2D_SubmitsOnlySeparatelyAllocatedNonZeroHostInput) {
  auto separate_host = TensorFaker().Placement(kOnHost).Shape({}).DataType(ge::DT_INT32).Value<int32_t>({7}).Build();
  auto merged_host =
      TensorFaker().Placement(kOnHost).Shape({3}).DataType(ge::DT_INT8).Value<char>({'1', '2', '3'}).Build();
  auto device = TensorFaker().Placement(kOnDeviceHbm).Shape({1}).DataType(ge::DT_INT32).Build();
  auto zero_host = TensorFaker().Placement(kOnHost).Shape({}).Build();
  zero_host.GetTensor()->MutableTensorData().SetAddr(nullptr, nullptr);
  zero_host.GetTensor()->MutableOriginShape().AppendDim(0U);
  zero_host.GetTensor()->MutableStorageShape().AppendDim(0U);
  zero_host.GetTensor()->SetSize(0U);

  auto separate_shape = separate_host.GetTensor()->GetShape();
  auto merged_shape = merged_host.GetTensor()->GetShape();
  auto device_shape = device.GetTensor()->GetShape();
  auto zero_shape = zero_host.GetTensor()->GetShape();
  const auto separate_type = separate_host.GetTensor()->GetDataType();
  const auto merged_type = merged_host.GetTensor()->GetDataType();
  const auto device_type = device.GetTensor()->GetDataType();
  const auto zero_type = zero_host.GetTensor()->GetDataType();
  GertTensorData separate_src;
  GertTensorData merged_src;
  GertTensorData device_src;
  GertTensorData zero_src;
  TensorUtils::RefTdToGtd(separate_host.GetTensor()->GetTensorData(), -1, separate_src);
  TensorUtils::RefTdToGtd(merged_host.GetTensor()->GetTensorData(), -1, merged_src);
  TensorUtils::RefTdToGtd(device.GetTensor()->GetTensorData(), -1, device_src);
  TensorUtils::RefTdToGtd(zero_host.GetTensor()->GetTensorData(), -1, zero_src);

  std::vector<uint8_t> separate_dst(32U, 0U);
  std::vector<uint8_t> zero_dst(32U, 0U);
  GertTensorData separate_alloc = {separate_dst.data(), separate_dst.size(), kOnDeviceHbm, -1};
  GertTensorData merged_alloc = {nullptr, 0U, kOnDeviceHbm, -1};
  GertTensorData device_alloc = {separate_dst.data(), separate_dst.size(), kOnDeviceHbm, -1};
  GertTensorData zero_alloc = {zero_dst.data(), zero_dst.size(), kOnDeviceHbm, -1};
  auto allocated_addrs =
      ContinuousVectorBuilder::Create<GertTensorData *>({&separate_alloc, &merged_alloc, &device_alloc, &zero_alloc});

  constexpr size_t kCopyFlowCount = 4U;
  auto stream = reinterpret_cast<aclrtStream>(0x11);
  auto context_holder = KernelRunContextFaker()
                            .KernelIONum(3U + kCopyFlowCount * 4U, 0U)
                            .Inputs({reinterpret_cast<void *>(kCopyFlowCount), stream, allocated_addrs.get(),
                                     &separate_src, reinterpret_cast<void *>(separate_host.GetTensor()->GetSize()),
                                     &separate_shape, reinterpret_cast<void *>(separate_type), &merged_src,
                                     reinterpret_cast<void *>(merged_host.GetTensor()->GetSize()), &merged_shape,
                                     reinterpret_cast<void *>(merged_type), &device_src,
                                     reinterpret_cast<void *>(device.GetTensor()->GetSize()), &device_shape,
                                     reinterpret_cast<void *>(device_type), &zero_src,
                                     reinterpret_cast<void *>(zero_host.GetTensor()->GetSize()), &zero_shape,
                                     reinterpret_cast<void *>(zero_type)})
                            .Build();
  auto funcs = registry.FindKernelFuncs("LaunchCopyFlowH2D");
  ASSERT_NE(funcs, nullptr);
  EXPECT_EQ(context_holder.GetContext<KernelContext>()->GetOutputNum(), 0U);

  auto acl_runtime_stub = std::make_shared<MockCopyFlowAclRuntime>();
  ge::AclRuntimeStub::SetInstance(acl_runtime_stub);
  EXPECT_CALL(*acl_runtime_stub,
              aclrtMemcpyAsync(separate_alloc.GetAddr(), separate_alloc.GetSize(), separate_src.GetAddr(),
                               sizeof(int32_t), ACL_MEMCPY_HOST_TO_BUF_TO_DEVICE, stream))
      .WillOnce(testing::Return(0));
  const auto result = funcs->run_func(context_holder.GetContext<KernelContext>());
  ge::AclRuntimeStub::Reset();
  EXPECT_EQ(result, ge::GRAPH_SUCCESS);
  context_holder.FreeAll();
}

TEST_F(CopyFlowLaunchKernelTest, PrepareCopyFlowResult_RejectsStaticOutputCountMismatch) {
  constexpr size_t kStaticCopyFlowCount = 2U;
  auto outputs = OutputsHolder::Fake(1U);
  auto context_holder = KernelRunContextFaker().KernelIONum(0U, 1U).Outputs(outputs.pointer).Build();
  auto funcs = registry.FindKernelFuncs("PrepareCopyFlowResult");
  ASSERT_NE(funcs, nullptr);
  auto graph = std::make_shared<ge::ExecuteGraph>("prepare_copy_flow_static_count_mismatch");
  auto node = FastNodeFaker(graph).Attr("copy_flow_count", static_cast<int64_t>(kStaticCopyFlowCount)).Build();
  GertRuntimeStub runtime_stub;
  runtime_stub.GetSlogStub().NoConsoleOut().SetLevelInfo();
  EXPECT_NE(funcs->outputs_creator(node, context_holder.GetContext<KernelContext>()), ge::GRAPH_SUCCESS);
  context_holder.FreeAll();
}

TEST_F(CopyFlowLaunchKernelTest, PrepareCopyFlowResult_RejectsInputIndexCardinalityMismatch) {
  constexpr size_t kCopyFlowCount = 1U;
  auto tensor_holder = TensorFaker().Placement(kOnDeviceHbm).Shape({1}).Build();
  auto shape = tensor_holder.GetTensor()->GetShape();
  const auto data_type = tensor_holder.GetTensor()->GetDataType();
  GertTensorData tensor_data;
  TensorUtils::RefTdToGtd(tensor_holder.GetTensor()->GetTensorData(), -1, tensor_data);
  GertTensorData allocated_data = {nullptr, 0U, kOnDeviceHbm, -1};
  auto allocated_addrs = ContinuousVectorBuilder::Create<GertTensorData *>({&allocated_data});
  size_t cvv_total_size = 0U;
  auto empty_indexes = GetContinuousVector2DByVector2D({}, cvv_total_size);
  auto launch_args = CreateLaunchArg(kCopyFlowCount, kCopyFlowCount);
  auto outputs = OutputsHolder::Fake(kCopyFlowCount);
  auto context_holder =
      KernelRunContextFaker()
          .KernelIONum(8U, kCopyFlowCount)
          .Inputs({reinterpret_cast<void *>(kCopyFlowCount), empty_indexes.get(), launch_args.get(),
                   allocated_addrs.get(), &tensor_data, reinterpret_cast<void *>(tensor_holder.GetTensor()->GetSize()),
                   &shape, reinterpret_cast<void *>(data_type)})
          .Outputs(outputs.pointer)
          .Build();
  auto funcs = registry.FindKernelFuncs("PrepareCopyFlowResult");
  ASSERT_NE(funcs, nullptr);
  auto graph = std::make_shared<ge::ExecuteGraph>("prepare_copy_flow_index_count_mismatch");
  auto node = FastNodeFaker(graph).Attr("copy_flow_count", static_cast<int64_t>(kCopyFlowCount)).Build();
  ASSERT_EQ(funcs->outputs_creator(node, context_holder.GetContext<KernelContext>()), ge::GRAPH_SUCCESS);
  GertRuntimeStub runtime_stub;
  runtime_stub.GetSlogStub().NoConsoleOut().SetLevelInfo();
  EXPECT_NE(funcs->run_func(context_holder.GetContext<KernelContext>()), ge::GRAPH_SUCCESS);
  context_holder.FreeAll();
}

TEST_F(CopyFlowLaunchKernelTest, PrepareCopyFlowResult_RejectsAllocatedAddressCardinalityMismatch) {
  constexpr size_t kCopyFlowCount = 1U;
  auto tensor_holder = TensorFaker().Placement(kOnDeviceHbm).Shape({1}).Build();
  auto shape = tensor_holder.GetTensor()->GetShape();
  const auto data_type = tensor_holder.GetTensor()->GetDataType();
  GertTensorData tensor_data;
  TensorUtils::RefTdToGtd(tensor_holder.GetTensor()->GetTensorData(), -1, tensor_data);
  auto allocated_addrs = ContinuousVectorBuilder::Create<GertTensorData *>({});
  std::vector<std::vector<int32_t>> input_indexes = {{0}};
  size_t cvv_total_size = 0U;
  auto indexes = GetContinuousVector2DByVector2D(input_indexes, cvv_total_size);
  auto launch_args = CreateLaunchArg(kCopyFlowCount, kCopyFlowCount);
  auto outputs = OutputsHolder::Fake(kCopyFlowCount);
  auto context_holder =
      KernelRunContextFaker()
          .KernelIONum(8U, kCopyFlowCount)
          .Inputs({reinterpret_cast<void *>(kCopyFlowCount), indexes.get(), launch_args.get(), allocated_addrs.get(),
                   &tensor_data, reinterpret_cast<void *>(tensor_holder.GetTensor()->GetSize()), &shape,
                   reinterpret_cast<void *>(data_type)})
          .Outputs(outputs.pointer)
          .Build();
  auto funcs = registry.FindKernelFuncs("PrepareCopyFlowResult");
  ASSERT_NE(funcs, nullptr);
  auto graph = std::make_shared<ge::ExecuteGraph>("prepare_copy_flow_allocated_count_mismatch");
  auto node = FastNodeFaker(graph).Attr("copy_flow_count", static_cast<int64_t>(kCopyFlowCount)).Build();
  ASSERT_EQ(funcs->outputs_creator(node, context_holder.GetContext<KernelContext>()), ge::GRAPH_SUCCESS);
  GertRuntimeStub runtime_stub;
  runtime_stub.GetSlogStub().NoConsoleOut().SetLevelInfo();
  EXPECT_NE(funcs->run_func(context_holder.GetContext<KernelContext>()), ge::GRAPH_SUCCESS);
  context_holder.FreeAll();
}

TEST_F(CopyFlowLaunchKernelTest, LaunchCopyFlowH2D_RejectsAllocatedAddressCardinalityMismatch) {
  constexpr size_t kCopyFlowCount = 1U;
  auto tensor_holder = TensorFaker().Placement(kOnDeviceHbm).Shape({1}).Build();
  auto shape = tensor_holder.GetTensor()->GetShape();
  const auto data_type = tensor_holder.GetTensor()->GetDataType();
  GertTensorData tensor_data;
  TensorUtils::RefTdToGtd(tensor_holder.GetTensor()->GetTensorData(), -1, tensor_data);
  auto allocated_addrs = ContinuousVectorBuilder::Create<GertTensorData *>({});
  auto context_holder =
      KernelRunContextFaker()
          .KernelIONum(7U, 0U)
          .Inputs({reinterpret_cast<void *>(kCopyFlowCount), reinterpret_cast<void *>(0x11), allocated_addrs.get(),
                   &tensor_data, reinterpret_cast<void *>(tensor_holder.GetTensor()->GetSize()), &shape,
                   reinterpret_cast<void *>(data_type)})
          .Build();
  auto funcs = registry.FindKernelFuncs("LaunchCopyFlowH2D");
  ASSERT_NE(funcs, nullptr);
  GertRuntimeStub runtime_stub;
  runtime_stub.GetSlogStub().NoConsoleOut().SetLevelInfo();
  EXPECT_NE(funcs->run_func(context_holder.GetContext<KernelContext>()), ge::GRAPH_SUCCESS);
  context_holder.FreeAll();
}

TEST_F(CopyFlowLaunchKernelTest, PrepareCopyFlowResult_RejectsRuntimeCardinalityMismatch) {
  constexpr size_t kCopyFlowCount = 1U;
  auto launch_args = CreateLaunchArg(kCopyFlowCount, kCopyFlowCount);
  std::vector<std::vector<int32_t>> input_indexes = {{0}};
  size_t cvv_total_size = 0U;
  auto indexes = GetContinuousVector2DByVector2D(input_indexes, cvv_total_size);
  GertTensorData allocated_data = {nullptr, 0U, kOnDeviceHbm, -1};
  auto allocated_addrs = ContinuousVectorBuilder::Create<GertTensorData *>({&allocated_data});
  auto outputs = OutputsHolder::Fake(kCopyFlowCount);
  auto context_holder =
      KernelRunContextFaker()
          .KernelIONum(4U, kCopyFlowCount)
          .Inputs({reinterpret_cast<void *>(kCopyFlowCount), indexes.get(), launch_args.get(), allocated_addrs.get()})
          .Outputs(outputs.pointer)
          .Build();
  auto funcs = registry.FindKernelFuncs("PrepareCopyFlowResult");
  ASSERT_NE(funcs, nullptr);
  auto graph = std::make_shared<ge::ExecuteGraph>("prepare_copy_flow_cardinality");
  auto node = FastNodeFaker(graph).Attr("copy_flow_count", static_cast<int64_t>(kCopyFlowCount)).Build();
  ASSERT_EQ(funcs->outputs_creator(node, context_holder.GetContext<KernelContext>()), ge::GRAPH_SUCCESS);
  GertRuntimeStub runtime_stub;
  runtime_stub.GetSlogStub().NoConsoleOut().SetLevelInfo();
  EXPECT_EQ(funcs->run_func(context_holder.GetContext<KernelContext>()), ge::GRAPH_FAILED);
  context_holder.FreeAll();
}

TEST_F(CopyFlowLaunchKernelTest, LaunchCopyFlowH2D_RejectsRuntimeCardinalityMismatch) {
  constexpr size_t kCopyFlowCount = 1U;
  GertTensorData allocated_data = {nullptr, 0U, kOnDeviceHbm, -1};
  auto allocated_addrs = ContinuousVectorBuilder::Create<GertTensorData *>({&allocated_data});
  auto context_holder =
      KernelRunContextFaker()
          .KernelIONum(3U, 0U)
          .Inputs({reinterpret_cast<void *>(kCopyFlowCount), reinterpret_cast<void *>(0x11), allocated_addrs.get()})
          .Build();
  auto funcs = registry.FindKernelFuncs("LaunchCopyFlowH2D");
  ASSERT_NE(funcs, nullptr);
  GertRuntimeStub runtime_stub;
  runtime_stub.GetSlogStub().NoConsoleOut().SetLevelInfo();
  EXPECT_EQ(funcs->run_func(context_holder.GetContext<KernelContext>()), ge::GRAPH_FAILED);
  context_holder.FreeAll();
}
}  // namespace gert
