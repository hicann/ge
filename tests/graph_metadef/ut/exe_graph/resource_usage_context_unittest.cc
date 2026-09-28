/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "exe_graph/runtime/resource_usage_context.h"

#include <cstddef>
#include <limits>
#include <set>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include "exe_graph/lowering/kernel_run_context_builder.h"
#include "runtime/resource_usage_collector.h"
#include "graph/op_desc.h"

namespace gert {
namespace {
constexpr const char *kOpName = "resource_usage_op";
constexpr const char *kOpType = "ResourceUsageOp";

ge::OpDescPtr MakeOpDesc() {
  auto op_desc = std::make_shared<ge::OpDesc>(kOpName, kOpType);
  const ge::GeTensorDesc tensor_desc(ge::GeShape({2, 8}), ge::FORMAT_ND, ge::DT_FLOAT);
  (void)op_desc->AddInputDesc("x", tensor_desc);
  (void)op_desc->AddOutputDesc("y", tensor_desc);
  return op_desc;
}

Tensor MakeTensor(void *addr) {
  return Tensor({{2, 8}, {2, 8}}, {ge::FORMAT_ND, ge::FORMAT_ND, {}}, kOnDeviceHbm, ge::DT_FLOAT, addr);
}

// 带 IR 原型信息的 OpDesc：输入 x（必需，1 实例）、dyn（动态，2 实例）、opt（可选，未实例化）；
// 输出 y（必需，1 实例）、outs（动态，2 实例）。扁平实例序为 [x, dyn0, dyn1] 与 [y, outs0, outs1]
ge::OpDescPtr MakeIrOpDesc() {
  auto op_desc = std::make_shared<ge::OpDesc>("resource_usage_ir_op", "ResourceUsageIrOp");
  const ge::GeTensorDesc tensor_desc(ge::GeShape({2, 8}), ge::FORMAT_ND, ge::DT_FLOAT);
  (void)op_desc->AddInputDesc("x", tensor_desc);
  (void)op_desc->AddInputDesc("dyn0", tensor_desc);
  (void)op_desc->AddInputDesc("dyn1", tensor_desc);
  op_desc->AppendIrInput("x", ge::kIrInputRequired);
  op_desc->AppendIrInput("dyn", ge::kIrInputDynamic);
  op_desc->AppendIrInput("opt", ge::kIrInputOptional);
  (void)op_desc->AddOutputDesc("y", tensor_desc);
  (void)op_desc->AddOutputDesc("outs0", tensor_desc);
  (void)op_desc->AddOutputDesc("outs1", tensor_desc);
  op_desc->AppendIrOutput("y", ge::kIrOutputRequired);
  op_desc->AppendIrOutput("outs", ge::kIrOutputDynamic);
  return op_desc;
}
}  // namespace

TEST(ResourceUsageContextUT, ReportsKeysAndDedupsAcrossCalls) {
  ResourceUsageCollector collector;
  Tensor input_tensor = MakeTensor(reinterpret_cast<void *>(0x1000U));
  Tensor output_tensor = MakeTensor(reinterpret_cast<void *>(0x2000U));
  // 收集器挂在附加输入槽 ResourceUsageInput::kCollector，与生产侧 model_builder.cc 同款布局
  auto holder =
      KernelRunContextBuilder().Inputs({&input_tensor, &collector}).Outputs({&output_tensor}).Build(MakeOpDesc());
  auto *context = reinterpret_cast<ResourceUsageContext *>(holder.GetKernelContext());
  ASSERT_NE(context, nullptr);

  // 模拟同 op_type 两个节点按静态 shape 申报不同 key 集：{key_s, key_v_1024} 与 {key_s, key_v_2048}
  EXPECT_EQ(context->ReportAttachedStream({ge::AscendString("key_s"), ge::AscendString("key_v_1024")}),
            ge::GRAPH_SUCCESS);
  EXPECT_EQ(context->ReportAttachedStream({ge::AscendString("key_s"), ge::AscendString("key_v_2048")}),
            ge::GRAPH_SUCCESS);
  EXPECT_FALSE(collector.HasError());
  const std::set<std::string> expected_keys{"key_s", "key_v_1024", "key_v_2048"};
  EXPECT_EQ(collector.GetAttachedStreamKeys(), expected_keys);
}

TEST(ResourceUsageContextUT, RejectsEmptyKeyWithStickyError) {
  ResourceUsageCollector collector;
  Tensor input_tensor = MakeTensor(reinterpret_cast<void *>(0x1000U));
  Tensor output_tensor = MakeTensor(reinterpret_cast<void *>(0x2000U));
  auto holder =
      KernelRunContextBuilder().Inputs({&input_tensor, &collector}).Outputs({&output_tensor}).Build(MakeOpDesc());
  auto *context = reinterpret_cast<ResourceUsageContext *>(holder.GetKernelContext());
  ASSERT_NE(context, nullptr);

  // 空 key 之前的合法 key 已入 union，空 key 处失败返回
  EXPECT_NE(context->ReportAttachedStream({ge::AscendString("key_a"), ge::AscendString("")}), ge::GRAPH_SUCCESS);
  EXPECT_TRUE(collector.HasError());
  const std::set<std::string> expected_keys{"key_a"};
  EXPECT_EQ(collector.GetAttachedStreamKeys(), expected_keys);

  // 错误状态粘滞：即使算子吞掉错误码后继续成功上报，HasError 仍为 true，供编译窗口复核拦截
  EXPECT_EQ(context->ReportAttachedStream({ge::AscendString("key_b")}), ge::GRAPH_SUCCESS);
  EXPECT_TRUE(collector.HasError());
}

TEST(ResourceUsageContextUT, EmptyKeyListSucceeds) {
  ResourceUsageCollector collector;
  Tensor input_tensor = MakeTensor(reinterpret_cast<void *>(0x1000U));
  Tensor output_tensor = MakeTensor(reinterpret_cast<void *>(0x2000U));
  auto holder =
      KernelRunContextBuilder().Inputs({&input_tensor, &collector}).Outputs({&output_tensor}).Build(MakeOpDesc());
  auto *context = reinterpret_cast<ResourceUsageContext *>(holder.GetKernelContext());
  ASSERT_NE(context, nullptr);

  EXPECT_EQ(context->ReportAttachedStream({}), ge::GRAPH_SUCCESS);
  EXPECT_FALSE(collector.HasError());
  EXPECT_TRUE(collector.GetAttachedStreamKeys().empty());
}

TEST(ResourceUsageContextUT, FailsWhenCollectorSlotIsNull) {
  Tensor input_tensor = MakeTensor(reinterpret_cast<void *>(0x1000U));
  Tensor output_tensor = MakeTensor(reinterpret_cast<void *>(0x2000U));
  auto holder =
      KernelRunContextBuilder().Inputs({&input_tensor, nullptr}).Outputs({&output_tensor}).Build(MakeOpDesc());
  auto *context = reinterpret_cast<ResourceUsageContext *>(holder.GetKernelContext());
  ASSERT_NE(context, nullptr);
  EXPECT_NE(context->ReportAttachedStream({ge::AscendString("key_s")}), ge::GRAPH_SUCCESS);
}

TEST(ResourceUsageContextUT, ExposesNodeInfoAndTensors) {
  ResourceUsageCollector collector;
  Tensor input_tensor = MakeTensor(reinterpret_cast<void *>(0x1000U));
  Tensor output_tensor = MakeTensor(reinterpret_cast<void *>(0x2000U));
  auto holder =
      KernelRunContextBuilder().Inputs({&input_tensor, &collector}).Outputs({&output_tensor}).Build(MakeOpDesc());
  auto *context = reinterpret_cast<ResourceUsageContext *>(holder.GetKernelContext());
  ASSERT_NE(context, nullptr);

  // 节点信息全部继承自 ExtendedKernelContext
  ASSERT_NE(context->GetNodeName(), nullptr);
  EXPECT_STREQ(context->GetNodeName(), kOpName);
  ASSERT_NE(context->GetNodeType(), nullptr);
  EXPECT_STREQ(context->GetNodeType(), kOpType);

  // tensor 薄封装：输入含附加槽位越界保护，输出按 ComputeNodeInfo 数量保护
  EXPECT_EQ(context->GetInputTensor(0U), &input_tensor);
  EXPECT_EQ(context->GetInputTensor(1U), nullptr);
  EXPECT_EQ(context->GetInputTensor(std::numeric_limits<size_t>::max()), nullptr);
  EXPECT_EQ(context->GetOutputTensor(0U), &output_tensor);
  EXPECT_EQ(context->GetOutputTensor(1U), nullptr);

  // 编译期静态 shape / dtype 可读取（算子按 shape 分桶申报 key 的依据）
  const auto *tensor = context->GetInputTensor(0U);
  ASSERT_NE(tensor, nullptr);
  const auto &shape = tensor->GetShape().GetStorageShape();
  ASSERT_EQ(shape.GetDimNum(), 2U);
  EXPECT_EQ(shape.GetDim(0U), 2);
  EXPECT_EQ(shape.GetDim(1U), 8);
  EXPECT_EQ(tensor->GetDataType(), ge::DT_FLOAT);
}

// IR 原型索引系列：按 ir_index（+ relative_index）定位实例，未实例化/越界均返回 nullptr
TEST(ResourceUsageContextUT, ExposesIrIndexedTensors) {
  ResourceUsageCollector collector;
  Tensor input_x = MakeTensor(reinterpret_cast<void *>(0x1000U));
  Tensor input_dyn0 = MakeTensor(reinterpret_cast<void *>(0x1100U));
  Tensor input_dyn1 = MakeTensor(reinterpret_cast<void *>(0x1200U));
  Tensor output_y = MakeTensor(reinterpret_cast<void *>(0x2000U));
  Tensor output_s0 = MakeTensor(reinterpret_cast<void *>(0x2100U));
  Tensor output_s1 = MakeTensor(reinterpret_cast<void *>(0x2200U));
  auto holder = KernelRunContextBuilder()
                    .Inputs({&input_x, &input_dyn0, &input_dyn1, &collector})
                    .Outputs({&output_y, &output_s0, &output_s1})
                    .Build(MakeIrOpDesc());
  auto *context = reinterpret_cast<ResourceUsageContext *>(holder.GetKernelContext());
  ASSERT_NE(context, nullptr);

  // 输入侧：必需输入 1 实例，动态输入 2 实例
  EXPECT_EQ(context->GetRequiredInputTensor(0U), &input_x);
  EXPECT_EQ(context->GetDynamicInputTensor(1U, 0U), &input_dyn0);
  EXPECT_EQ(context->GetDynamicInputTensor(1U, 1U), &input_dyn1);
  // relative_index 越界、可选输入未实例化、ir_index 越界均为 nullptr
  EXPECT_EQ(context->GetDynamicInputTensor(1U, 2U), nullptr);
  EXPECT_EQ(context->GetOptionalInputTensor(2U), nullptr);
  EXPECT_EQ(context->GetRequiredInputTensor(3U), nullptr);

  // 输出侧：必需输出 1 实例，动态输出 2 实例
  EXPECT_EQ(context->GetRequiredOutputTensor(0U), &output_y);
  EXPECT_EQ(context->GetDynamicOutputTensor(1U, 0U), &output_s0);
  EXPECT_EQ(context->GetDynamicOutputTensor(1U, 1U), &output_s1);
  EXPECT_EQ(context->GetDynamicOutputTensor(1U, 2U), nullptr);
  EXPECT_EQ(context->GetRequiredOutputTensor(2U), nullptr);

  // 扁平索引口径不受影响，附加槽仍被隔离；申报链路正常
  EXPECT_EQ(context->GetInputTensor(0U), &input_x);
  EXPECT_EQ(context->GetInputTensor(3U), nullptr);
  EXPECT_EQ(context->GetOutputTensor(0U), &output_y);
  EXPECT_EQ(context->ReportAttachedStream({ge::AscendString("key_ir")}), ge::GRAPH_SUCCESS);
  EXPECT_FALSE(collector.HasError());
}

// OpDesc 不带 IR 原型信息时，IR 索引系列安全退化为 nullptr（不影响扁平索引与申报）
TEST(ResourceUsageContextUT, IrIndexedTensorsReturnNullWithoutIrPrototype) {
  ResourceUsageCollector collector;
  Tensor input_tensor = MakeTensor(reinterpret_cast<void *>(0x1000U));
  Tensor output_tensor = MakeTensor(reinterpret_cast<void *>(0x2000U));
  auto holder =
      KernelRunContextBuilder().Inputs({&input_tensor, &collector}).Outputs({&output_tensor}).Build(MakeOpDesc());
  auto *context = reinterpret_cast<ResourceUsageContext *>(holder.GetKernelContext());
  ASSERT_NE(context, nullptr);

  EXPECT_EQ(context->GetRequiredInputTensor(0U), nullptr);
  EXPECT_EQ(context->GetDynamicInputTensor(0U, 0U), nullptr);
  EXPECT_EQ(context->GetRequiredOutputTensor(0U), nullptr);
  EXPECT_EQ(context->GetDynamicOutputTensor(0U, 0U), nullptr);
  EXPECT_EQ(context->GetInputTensor(0U), &input_tensor);
  EXPECT_EQ(context->ReportAttachedStream({ge::AscendString("key_no_ir")}), ge::GRAPH_SUCCESS);
}

// 附加输入槽位布局守护：只允许在 kNum 前追加，禁止插入、重排或修改已有值。
// 生产侧 model_builder.cc 与消费侧 resource_usage_context.cc 共用该内部枚举定位收集器槽位
TEST(ResourceUsageContextUT, AdditionalInputLayoutIsAppendOnly) {
  EXPECT_EQ(static_cast<uint32_t>(ResourceUsageInput::kCollector), 0U);
  EXPECT_EQ(static_cast<uint32_t>(ResourceUsageInput::kNum), 1U);
}
}  // namespace gert
