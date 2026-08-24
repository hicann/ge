/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>

#include <new>
#include <string>

#include "graph/custom_op/infer_meta.h"
#include "graph/custom_op/cast.h"
#include "graph/custom_op_factory.h"
#include "graph/operator_factory_impl.h"
#include "runtime/custom_op/python_custom_op_adapter.h"

namespace ge {
namespace custom_op {
namespace {

graphStatus g_infer_meta_call_count = 0;
graphStatus g_infer_meta_return_value = GRAPH_SUCCESS;

graphStatus FakeInferMetaBridge(const std::string &op_type, gert::InferShapeContext *ctx,
                                CustomOpInferMetaResult *result) {
  (void)op_type;
  (void)ctx;
  ++g_infer_meta_call_count;
  if (result != nullptr) {
    result->outputs.resize(2U);
    result->outputs[0U].shape = gert::StorageShape{{2}, {2}};
    result->outputs[0U].data_type = DT_FLOAT;
    result->outputs[1U].shape = gert::StorageShape{{3}, {3}};
    result->outputs[1U].data_type = DT_INT32;
  }
  return g_infer_meta_return_value;
}

graphStatus FakePythonInferMetaBridge(const PythonCustomOpStringView *, gert::InferShapeContext *,
                                      PythonCustomOpInferMetaResultView *) {
  return GRAPH_SUCCESS;
}

class InferMetaProviderTestOp final : public CustomOpInferMetaProvider {
 public:
  graphStatus InferMeta(gert::InferShapeContext *ctx, CustomOpInferMetaResult *result) override {
    return FakeInferMetaBridge("InferMetaProviderTestOp", ctx, result);
  }
};

class NonInferMetaTestOp final : public BaseCustomOp {};

class PythonCustomOpInferMetaProviderTest : public testing::Test {
 protected:
  void SetUp() override {
    g_infer_meta_call_count = 0;
    g_infer_meta_return_value = GRAPH_SUCCESS;
  }

  void TearDown() override {
    CustomOpFactory::RemoveCustomOps({AscendString("InferMetaProviderTestOp")});
    CustomOpFactory::RemoveCustomOps({AscendString("NonInferMetaTestOp")});
    CustomOpFactory::RemoveCustomOps({AscendString("PythonInferOnlyUt")});
  }
};

TEST_F(PythonCustomOpInferMetaProviderTest, dynamic_cast_identifies_infer_meta_provider) {
  ASSERT_EQ(CustomOpFactory::RegisterCustomOpCreator(
                AscendString("InferMetaProviderTestOp"),
                []() -> std::unique_ptr<BaseCustomOp> { return std::make_unique<InferMetaProviderTestOp>(); }),
            GRAPH_SUCCESS);

  auto op = CustomOpFactory::CreateOrGetCustomOp(AscendString("InferMetaProviderTestOp"));
  ASSERT_NE(op, nullptr);

  auto *provider = dynamic_cast<CustomOpInferMetaProvider *>(op);
  EXPECT_NE(provider, nullptr);

  CustomOpInferMetaResult result;
  auto ret = provider->InferMeta(nullptr, &result);
  EXPECT_EQ(ret, GRAPH_SUCCESS);
  EXPECT_EQ(g_infer_meta_call_count, 1);
  EXPECT_EQ(result.outputs.size(), 2U);
  EXPECT_EQ(result.outputs[0].shape.GetStorageShape(), gert::Shape({2}));
  EXPECT_EQ(result.outputs[1].shape.GetStorageShape(), gert::Shape({3}));
  EXPECT_EQ(result.outputs[0].data_type, DT_FLOAT);
  EXPECT_EQ(result.outputs[1].data_type, DT_INT32);
}

TEST_F(PythonCustomOpInferMetaProviderTest, dynamic_cast_returns_null_for_non_provider) {
  ASSERT_EQ(CustomOpFactory::RegisterCustomOpCreator(
                AscendString("NonInferMetaTestOp"),
                []() -> std::unique_ptr<BaseCustomOp> { return std::make_unique<NonInferMetaTestOp>(); }),
            GRAPH_SUCCESS);

  auto op = CustomOpFactory::CreateOrGetCustomOp(AscendString("NonInferMetaTestOp"));
  ASSERT_NE(op, nullptr);

  auto *provider = dynamic_cast<CustomOpInferMetaProvider *>(op);
  EXPECT_EQ(provider, nullptr);
}

TEST_F(PythonCustomOpInferMetaProviderTest, custom_op_cast_preserves_shape_infer_op_fallback) {
  ASSERT_EQ(CustomOpFactory::RegisterCustomOpCreator(
                AscendString("NonInferMetaTestOp"),
                []() -> std::unique_ptr<BaseCustomOp> { return std::make_unique<NonInferMetaTestOp>(); }),
            GRAPH_SUCCESS);

  auto op = CustomOpFactory::CreateOrGetCustomOp(AscendString("NonInferMetaTestOp"));
  ASSERT_NE(op, nullptr);

  auto *shape_infer = CustomOpCast<ShapeInferOp>(op);
  EXPECT_EQ(shape_infer, nullptr);
}

TEST_F(PythonCustomOpInferMetaProviderTest, infer_meta_returns_failure_propagates) {
  ASSERT_EQ(CustomOpFactory::RegisterCustomOpCreator(
                AscendString("InferMetaProviderTestOp"),
                []() -> std::unique_ptr<BaseCustomOp> { return std::make_unique<InferMetaProviderTestOp>(); }),
            GRAPH_SUCCESS);

  auto op = CustomOpFactory::CreateOrGetCustomOp(AscendString("InferMetaProviderTestOp"));
  ASSERT_NE(op, nullptr);
  auto *provider = dynamic_cast<CustomOpInferMetaProvider *>(op);
  ASSERT_NE(provider, nullptr);

  g_infer_meta_return_value = GRAPH_FAILED;
  CustomOpInferMetaResult result;
  auto ret = provider->InferMeta(nullptr, &result);
  EXPECT_EQ(ret, GRAPH_FAILED);
}

TEST_F(PythonCustomOpInferMetaProviderTest, infer_meta_result_default_is_empty) {
  CustomOpInferMetaResult result;
  EXPECT_TRUE(result.outputs.empty());
}

TEST_F(PythonCustomOpInferMetaProviderTest, python_custom_op_adapter_inherits_infer_meta_provider) {
  PythonCustomOpAdapterDescriptor desc;
  desc.op_type = "PythonAdapterInferMetaUt";
  desc.impl_descriptor_key = "ut:python_adapter_infer_meta";
  AddCustomOpCapability(desc.capabilities, CustomOpCapability::kEagerExecute);

  PythonCustomOpAdapterCallbacks callbacks;
  callbacks.create_impl_holder = [](const PythonCustomOpAdapterDescriptorView *) -> void * { return new int(1); };
  callbacks.destroy_impl_holder = [](void *holder) { delete static_cast<int *>(holder); };
  callbacks.execute = [](const void *, gert::EagerOpExecutionContext *) -> graphStatus { return GRAPH_SUCCESS; };

  ASSERT_TRUE(PythonCustomOpImplRuntimeRegistry::Register(desc, callbacks));

  PythonCustomOpAdapter adapter(desc);
  ASSERT_TRUE(adapter.IsValid());

  auto *provider = dynamic_cast<CustomOpInferMetaProvider *>(&adapter);
  EXPECT_NE(provider, nullptr);
  EXPECT_EQ(CustomOpCast<CustomOpInferMetaProvider>(&adapter), nullptr);
  EXPECT_EQ(CustomOpCast<ShapeInferOp>(&adapter), nullptr);

  PythonCustomOpImplRuntimeRegistry::Unregister(desc.impl_descriptor_key);
}

TEST_F(PythonCustomOpInferMetaProviderTest, python_custom_op_adapter_infer_only_via_descriptor) {
  PythonCustomOpAdapterDescriptor desc;
  desc.op_type = "PythonInferOnlyUt";
  desc.infer_meta = &FakePythonInferMetaBridge;
  AddCustomOpCapability(desc.capabilities, CustomOpCapability::kShapeInfer);
  AddCustomOpCapability(desc.capabilities, CustomOpCapability::kInferMeta);

  PythonCustomOpAdapter adapter(desc);
  EXPECT_TRUE(adapter.IsValid());

  auto *provider = dynamic_cast<CustomOpInferMetaProvider *>(&adapter);
  EXPECT_NE(provider, nullptr);
  EXPECT_NE(CustomOpCast<CustomOpInferMetaProvider>(&adapter), nullptr);
  auto *shape_infer = CustomOpCast<ShapeInferOp>(&adapter);
  EXPECT_NE(shape_infer, nullptr);
}

TEST_F(PythonCustomOpInferMetaProviderTest, python_custom_op_adapter_infer_only_registered_as_custom_op_creator) {
  ASSERT_EQ(CustomOpFactory::RegisterCustomOpCreator(
                AscendString("PythonInferOnlyUt"),
                []() -> std::unique_ptr<BaseCustomOp> {
                  PythonCustomOpAdapterDescriptor desc;
                  desc.op_type = "PythonInferOnlyUt";
                  desc.infer_meta = &FakePythonInferMetaBridge;
                  AddCustomOpCapability(desc.capabilities, CustomOpCapability::kShapeInfer);
                  AddCustomOpCapability(desc.capabilities, CustomOpCapability::kInferMeta);
                  return std::make_unique<PythonCustomOpAdapter>(desc);
                }),
            GRAPH_SUCCESS);
  auto op = CustomOpFactory::CreateOrGetCustomOp(AscendString("PythonInferOnlyUt"));
  ASSERT_NE(op, nullptr);
  EXPECT_NE(dynamic_cast<CustomOpInferMetaProvider *>(op), nullptr);
  EXPECT_NE(dynamic_cast<ShapeInferOp *>(op), nullptr);
}

TEST_F(PythonCustomOpInferMetaProviderTest, python_custom_op_adapter_infer_meta_fails_without_callback) {
  PythonCustomOpAdapterDescriptor desc;
  desc.op_type = "PythonAdapterInferMetaNoCallbackUt";
  desc.impl_descriptor_key = "ut:python_adapter_infer_meta_no_callback";
  AddCustomOpCapability(desc.capabilities, CustomOpCapability::kEagerExecute);

  PythonCustomOpAdapterCallbacks callbacks;
  callbacks.create_impl_holder = [](const PythonCustomOpAdapterDescriptorView *) -> void * { return new int(1); };
  callbacks.destroy_impl_holder = [](void *holder) { delete static_cast<int *>(holder); };
  callbacks.execute = [](const void *, gert::EagerOpExecutionContext *) -> graphStatus { return GRAPH_SUCCESS; };

  ASSERT_TRUE(PythonCustomOpImplRuntimeRegistry::Register(desc, callbacks));

  PythonCustomOpAdapter adapter(desc);
  ASSERT_TRUE(adapter.IsValid());

  CustomOpInferMetaResult result;
  auto ret = adapter.InferMeta(nullptr, &result);
  EXPECT_EQ(ret, GRAPH_FAILED);

  PythonCustomOpImplRuntimeRegistry::Unregister(desc.impl_descriptor_key);
}

}  // namespace
}  // namespace custom_op
}  // namespace ge
