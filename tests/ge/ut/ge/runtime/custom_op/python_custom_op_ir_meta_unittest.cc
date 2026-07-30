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

#include "graph/operator_reg.h"
#include "runtime/custom_op/python_custom_op_adapter.h"
#include "runtime/custom_op/python_custom_op_ir_meta.h"

namespace ge {
REG_OP(PythonCustomOpIrMetaUt)
    .INPUT(required_input, TensorType::ALL())
    .OPTIONAL_INPUT(optional_input, TensorType::ALL())
    .DYNAMIC_INPUT(dynamic_input, TensorType::ALL())
    .OUTPUT(required_output, TensorType::ALL())
    .DYNAMIC_OUTPUT(dynamic_output, TensorType::ALL())
    .ATTR(z_attr, Int, 0)
    .REQUIRED_ATTR(a_attr, Float)
    .OP_END_FACTORY_REG(PythonCustomOpIrMetaUt);
}  // namespace ge

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

graphStatus ExecuteMockPythonCustomOpWithIrMeta(const void *holder, gert::EagerOpExecutionContext *ctx,
                                                const PythonCustomOpIrMetaView *ir_meta) {
  (void)ctx;
  if ((holder == nullptr) || (ir_meta == nullptr)) {
    ADD_FAILURE() << "Python custom op holder or IR meta view is null";
    return GRAPH_FAILED;
  }

  EXPECT_STREQ(ir_meta->op_type, "PythonCustomOpIrMetaUt");
  EXPECT_EQ(ir_meta->input_count, 3U);
  EXPECT_EQ(ir_meta->attr_count, 2U);
  EXPECT_EQ(ir_meta->output_count, 2U);
  if ((ir_meta->input_count != 3U) || (ir_meta->attr_count != 2U) || (ir_meta->output_count != 2U)) {
    return GRAPH_FAILED;
  }

  EXPECT_STREQ(ir_meta->inputs[0].name, "required_input");
  EXPECT_EQ(ir_meta->inputs[0].kind, 0U);
  EXPECT_STREQ(ir_meta->inputs[1].name, "optional_input");
  EXPECT_EQ(ir_meta->inputs[1].kind, 1U);
  EXPECT_STREQ(ir_meta->inputs[2].name, "dynamic_input");
  EXPECT_EQ(ir_meta->inputs[2].kind, 2U);
  EXPECT_STREQ(ir_meta->attrs[0].name, "z_attr");
  EXPECT_STREQ(ir_meta->attrs[0].type, "VT_INT");
  EXPECT_STREQ(ir_meta->attrs[1].name, "a_attr");
  EXPECT_STREQ(ir_meta->attrs[1].type, "VT_FLOAT");
  EXPECT_STREQ(ir_meta->outputs[0].name, "required_output");
  EXPECT_EQ(ir_meta->outputs[0].kind, 0U);
  EXPECT_STREQ(ir_meta->outputs[1].name, "dynamic_output");
  EXPECT_EQ(ir_meta->outputs[1].kind, 1U);
  return GRAPH_SUCCESS;
}

graphStatus ExecuteMockLegacyPythonCustomOp(const void *holder, gert::EagerOpExecutionContext *ctx,
                                            const PythonCustomOpIrMetaView *ir_meta) {
  (void)ctx;
  EXPECT_NE(holder, nullptr);
  EXPECT_EQ(ir_meta, nullptr);
  return ((holder != nullptr) && (ir_meta == nullptr)) ? GRAPH_SUCCESS : GRAPH_FAILED;
}
}  // namespace

TEST(PythonCustomOpIrMeta, collects_ir_in_definition_order) {
  CustomOpIrMeta ir_meta;
  ASSERT_EQ(CollectCustomOpIrMeta("PythonCustomOpIrMetaUt", ir_meta), GRAPH_SUCCESS);

  EXPECT_EQ(ir_meta.op_type, "PythonCustomOpIrMetaUt");
  ASSERT_EQ(ir_meta.inputs.size(), 3U);
  EXPECT_EQ(ir_meta.inputs[0].name, "required_input");
  EXPECT_EQ(ir_meta.inputs[0].kind, kIrInputRequired);
  EXPECT_EQ(ir_meta.inputs[1].name, "optional_input");
  EXPECT_EQ(ir_meta.inputs[1].kind, kIrInputOptional);
  EXPECT_EQ(ir_meta.inputs[2].name, "dynamic_input");
  EXPECT_EQ(ir_meta.inputs[2].kind, kIrInputDynamic);

  ASSERT_EQ(ir_meta.attrs.size(), 2U);
  EXPECT_EQ(ir_meta.attrs[0].name, "z_attr");
  EXPECT_EQ(ir_meta.attrs[0].type, "VT_INT");
  EXPECT_EQ(ir_meta.attrs[1].name, "a_attr");
  EXPECT_EQ(ir_meta.attrs[1].type, "VT_FLOAT");

  ASSERT_EQ(ir_meta.outputs.size(), 2U);
  EXPECT_EQ(ir_meta.outputs[0].name, "required_output");
  EXPECT_EQ(ir_meta.outputs[0].kind, kIrOutputRequired);
  EXPECT_EQ(ir_meta.outputs[1].name, "dynamic_output");
  EXPECT_EQ(ir_meta.outputs[1].kind, kIrOutputDynamic);
}

TEST(PythonCustomOpIrMeta, rejects_unregistered_op_without_overwriting_output) {
  CustomOpIrMeta ir_meta;
  ir_meta.op_type = "keep_me";

  EXPECT_NE(CollectCustomOpIrMeta("PythonCustomOpIrMetaMissingUt", ir_meta), GRAPH_SUCCESS);
  EXPECT_EQ(ir_meta.op_type, "keep_me");
}

TEST(PythonCustomOpIrMeta, forwards_pod_view_to_python_callback) {
  PythonCustomOpDescriptor desc;
  desc.descriptor_key = "python_adapter_ir_meta";
  desc.op_type = "PythonCustomOpIrMetaUt";
  AddCustomOpCapability(desc.capabilities, CustomOpCapability::kEagerExecute);

  PythonCustomOpCallbacks callbacks;
  callbacks.create = CreateMockPythonCustomOpHolder;
  callbacks.destroy = DestroyMockPythonCustomOpHolder;
  callbacks.execute = ExecuteMockPythonCustomOpWithIrMeta;

  ASSERT_TRUE(PythonCustomOpRuntimeRegistry::Register(desc, callbacks));
  {
    PythonCustomOpAdapter adapter(desc);
    ASSERT_TRUE(adapter.IsValid());
    EXPECT_EQ(adapter.Execute(nullptr), GRAPH_SUCCESS);
  }
  EXPECT_TRUE(PythonCustomOpRuntimeRegistry::Unregister(desc.descriptor_key));
}

TEST(PythonCustomOpIrMeta, keeps_legacy_execute_without_registered_ir) {
  PythonCustomOpDescriptor desc;
  desc.descriptor_key = "python_adapter_legacy_without_ir";
  desc.op_type = "PythonCustomOpLegacyWithoutIrUt";
  AddCustomOpCapability(desc.capabilities, CustomOpCapability::kEagerExecute);

  PythonCustomOpCallbacks callbacks;
  callbacks.create = CreateMockPythonCustomOpHolder;
  callbacks.destroy = DestroyMockPythonCustomOpHolder;
  callbacks.execute = ExecuteMockLegacyPythonCustomOp;

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
