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

#include <new>

#include "runtime/custom_op/python_custom_op_adapter.h"

namespace ge {
namespace custom_op {
namespace {
struct MockPythonCustomOpHolder {};

void *CreateMockPythonCustomOpHolder(const PythonCustomOpDescriptor *desc) {
  return (desc == nullptr) ? nullptr : new (std::nothrow) MockPythonCustomOpHolder();
}

void DestroyMockPythonCustomOpHolder(void *holder) {
  delete static_cast<MockPythonCustomOpHolder *>(holder);
}

graphStatus ExecuteMockPythonCustomOp(const void *holder, gert::EagerOpExecutionContext *ctx) {
  (void)ctx;
  EXPECT_NE(holder, nullptr);
  return (holder != nullptr) ? GRAPH_SUCCESS : GRAPH_FAILED;
}
}  // namespace

TEST(PythonCustomOpAdapter, forwards_execute_without_ir_meta_pod) {
  PythonCustomOpDescriptor desc;
  desc.descriptor_key = "python_adapter_without_ir_meta";
  desc.op_type = "PythonCustomOpAdapterUt";
  AddCustomOpCapability(desc.capabilities, CustomOpCapability::kEagerExecute);

  PythonCustomOpCallbacks callbacks;
  callbacks.create = CreateMockPythonCustomOpHolder;
  callbacks.destroy = DestroyMockPythonCustomOpHolder;
  callbacks.execute = ExecuteMockPythonCustomOp;

  ASSERT_TRUE(PythonCustomOpRuntimeRegistry::Register(desc, callbacks));
  {
    PythonCustomOpAdapter adapter(desc);
    ASSERT_TRUE(adapter.IsValid());
    EXPECT_EQ(adapter.Execute(nullptr), GRAPH_SUCCESS);
  }
  EXPECT_TRUE(PythonCustomOpRuntimeRegistry::Unregister(desc.descriptor_key));
}

TEST(PythonCustomOpAdapter, keeps_legacy_execute_without_registered_ir) {
  PythonCustomOpDescriptor desc;
  desc.descriptor_key = "python_adapter_legacy_without_ir";
  desc.op_type = "PythonCustomOpLegacyWithoutIrUt";
  AddCustomOpCapability(desc.capabilities, CustomOpCapability::kEagerExecute);

  PythonCustomOpCallbacks callbacks;
  callbacks.create = CreateMockPythonCustomOpHolder;
  callbacks.destroy = DestroyMockPythonCustomOpHolder;
  callbacks.execute = ExecuteMockPythonCustomOp;

  ASSERT_TRUE(PythonCustomOpRuntimeRegistry::Register(desc, callbacks));
  {
    PythonCustomOpAdapter adapter(desc);
    ASSERT_TRUE(adapter.IsValid());
    EXPECT_EQ(adapter.Execute(nullptr), GRAPH_SUCCESS);
  }
  EXPECT_TRUE(PythonCustomOpRuntimeRegistry::Unregister(desc.descriptor_key));
}
}  // namespace custom_op
}  // namespace ge
