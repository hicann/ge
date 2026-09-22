/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>

#include <memory>
#include <string>
#include <vector>

#include "exe_graph/lowering/kernel_run_context_builder.h"
#include "exe_graph/runtime/eager_op_execution_context.h"
#include "framework/runtime/attached_stream_provider.h"
#include "framework/runtime/args_handler.h"
#include "graph/load/model_manager/task_info/ge/sink_op_args_handler.h"
#include "graph/op_desc.h"

namespace gert {
namespace {
class FakeProvider final : public AttachedStreamProvider {
 public:
  rtStream RequestAttachedStream(const ge::AscendString &key) override {
    last_key = key.GetString();
    return stream;
  }
  rtStream stream = reinterpret_cast<rtStream>(0x1234U);
  std::string last_key;
};
}  // namespace

TEST(EagerOpExecutionContextTest, PublicLayoutAndIndicesRemainStable) {
  EXPECT_EQ(static_cast<uint32_t>(EagerOpExecutionContext::AdditionalInputIndex::kDeviceAllocator), 0U);
  EXPECT_EQ(static_cast<uint32_t>(EagerOpExecutionContext::AdditionalInputIndex::kStream), 1U);
  EXPECT_EQ(static_cast<uint32_t>(EagerOpExecutionContext::AdditionalOutputIndex::kWorkSpace), 0U);
  EXPECT_EQ(static_cast<uint32_t>(EagerOpExecutionContext::AdditionalOutputIndex::kArgsHandler), 1U);
}

TEST(EagerOpExecutionContextTest, ProviderDefaultIsNull) {
  class DefaultHandler final : public ArgsHandler {
   public:
    const KernelArgs *MallocReadOnlyDevArgs(void *, size_t) override {
      return nullptr;
    }
    const std::deque<KernelArgs> &GetKernelArgs(Placement) const override {
      return args;
    }

   private:
    std::deque<KernelArgs> args;
  } handler;
  EXPECT_EQ(handler.GetAttachedStreamProvider(), nullptr);
}

TEST(EagerOpExecutionContextTest, ProviderPreservesKey) {
  FakeProvider provider;
  EXPECT_EQ(provider.RequestAttachedStream(ge::AscendString("aux")), provider.stream);
  EXPECT_EQ(provider.last_key, "aux");
}

namespace {
// 构造与 CustomTaskInfo::Distribute 相同布局的 eager context：
// 附加输入 = {device allocator, stream}，附加输出 = {workspace vector, args handler}
gert::KernelContextHolder BuildEagerContext(gert::ArgsHandler &handler, std::vector<int> &ws_vec,
                                            const ge::OpDescPtr &op_desc) {
  std::vector<void *> additional_inputs = {nullptr, nullptr};  // 本组用例不读取附加输入
  std::vector<void *> outputs = {&ws_vec, static_cast<gert::ArgsHandler *>(&handler)};
  return gert::KernelRunContextBuilder().Inputs(additional_inputs).Outputs(outputs).Build(op_desc);
}
}  // namespace

TEST(EagerOpExecutionContextTest, RequestAttachedStreamForwardsViaArgsHandlerSlot) {
  auto op_desc = std::make_shared<ge::OpDesc>("eager_attached_stream_op", "EagerAttachedStreamOp");
  std::vector<int> ws_vec;
  FakeProvider provider;
  ge::SinkOpArgsHandler handler(nullptr);
  handler.SetAttachedStreamProvider(&provider);

  auto holder = BuildEagerContext(handler, ws_vec, op_desc);
  auto *context = reinterpret_cast<EagerOpExecutionContext *>(holder.GetKernelContext());
  ASSERT_NE(context, nullptr);
  EXPECT_EQ(context->RequestAttachedStream(ge::AscendString("fwd_key")), provider.stream);
  EXPECT_EQ(provider.last_key, "fwd_key");
}

TEST(EagerOpExecutionContextTest, RequestAttachedStreamReturnsNullptrWithoutProvider) {
  auto op_desc = std::make_shared<ge::OpDesc>("eager_no_provider_op", "EagerNoProviderOp");
  std::vector<int> ws_vec;
  ge::SinkOpArgsHandler handler(nullptr);  // 未注入 provider，对应声明式算子等不支持辅流的场景

  auto holder = BuildEagerContext(handler, ws_vec, op_desc);
  auto *context = reinterpret_cast<EagerOpExecutionContext *>(holder.GetKernelContext());
  ASSERT_NE(context, nullptr);
  EXPECT_EQ(context->RequestAttachedStream(ge::AscendString("any_key")), nullptr);
}
}  // namespace gert
