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
#include <stdexcept>

#include "graph/custom_op.h"
#include "graph/custom_op_factory.h"
#include "graph/custom_op_registry.h"

namespace ge {
namespace {
class TestPriorityOp final : public BaseCustomOp {};

class TestPriorityOtherOp final : public BaseCustomOp {};

class TestPriorityRegisteredOp final : public BaseCustomOp {};
REG_OP_WITH_PRIORITY(TestPriorityRegisteredOp, "CustomOpRegistryPriorityMacroOp", OpBackend::kDevice,
                     OpRegistrationPriority::kBottom, OpEngine::kAiCore);

class TestPriorityShapeInferOp final : public ShapeInferOp {
 public:
  graphStatus InferShape(gert::InferShapeContext *) override {
    return GRAPH_SUCCESS;
  }

  graphStatus InferDataType(gert::InferDataTypeContext *) override {
    return GRAPH_SUCCESS;
  }
};
REG_OP_WITH_PRIORITY(TestPriorityShapeInferOp, "CustomOpRegistryCapabilityMacroOp", OpBackend::kDevice,
                     OpRegistrationPriority::kBottom, OpEngine::kAiCore);

class TestDefaultRegisteredOp final : public BaseCustomOp {};
REG_OP_BACKEND(TestDefaultRegisteredOp, "CustomOpRegistryDefaultMacroOp", OpBackend::kHostCPU);
}  // namespace

TEST(CustomOpRegistryPriorityTest, RegisterAndQueryByEngineBackendAndPriority) {
  CustomOpRegistry registry;
  const AscendString op_type("CustomOpRegistryPriorityRegisterOp");
  const OpEngine engine = OpEngine::kAiCore;
  const auto ret =
      registry.RegisterCreator(op_type, OpBackend::kHostCPU, OpRegistrationPriority::kBottom, engine,
                               []() -> std::unique_ptr<BaseCustomOp> { return std::make_unique<TestPriorityOp>(); });
  ASSERT_EQ(ret, GRAPH_SUCCESS);

  EXPECT_TRUE(registry.HasCreator(op_type, OpBackend::kHostCPU, OpRegistrationPriority::kBottom, engine));
  EXPECT_FALSE(registry.HasCreator(op_type, OpBackend::kHostCPU, OpRegistrationPriority::kTop, engine));
  EXPECT_FALSE(registry.HasCreator(op_type, OpBackend::kDevice, OpRegistrationPriority::kBottom, engine));
  EXPECT_FALSE(
      registry.HasCreator(op_type, OpBackend::kHostCPU, OpRegistrationPriority::kBottom, OpEngine::kVectorCore));
  EXPECT_FALSE(registry.HasCreator(op_type, OpBackend::kHostCPU));
}

TEST(CustomOpRegistryPriorityTest, TopAndBottomCanBeRegisteredIndependently) {
  CustomOpRegistry registry;
  const AscendString op_type("CustomOpRegistryPrioritySelectOp");
  const OpEngine engine = OpEngine::kAiCore;
  ASSERT_EQ(
      registry.RegisterCreator(op_type, OpBackend::kHostCPU, OpRegistrationPriority::kBottom, engine,
                               []() -> std::unique_ptr<BaseCustomOp> { return std::make_unique<TestPriorityOp>(); }),
      GRAPH_SUCCESS);
  ASSERT_EQ(registry.RegisterCreator(
                op_type, OpBackend::kHostCPU, OpRegistrationPriority::kTop, engine,
                []() -> std::unique_ptr<BaseCustomOp> { return std::make_unique<TestPriorityOtherOp>(); }),
            GRAPH_SUCCESS);

  EXPECT_TRUE(registry.HasCreator(op_type, OpBackend::kHostCPU, OpRegistrationPriority::kTop, engine));
  EXPECT_TRUE(registry.HasCreator(op_type, OpBackend::kHostCPU, OpRegistrationPriority::kBottom, engine));
}

TEST(CustomOpRegistryPriorityTest, CreateOrGetKeepsInstancesIndependentByPriority) {
  CustomOpRegistry registry;
  const AscendString op_type("CustomOpRegistryPriorityInstanceOp");
  const OpEngine engine = OpEngine::kAiCore;
  ASSERT_EQ(
      registry.RegisterCreator(op_type, OpBackend::kHostCPU, OpRegistrationPriority::kTop, engine,
                               []() -> std::unique_ptr<BaseCustomOp> { return std::make_unique<TestPriorityOp>(); }),
      GRAPH_SUCCESS);
  ASSERT_EQ(
      registry.RegisterCreator(op_type, OpBackend::kHostCPU, OpRegistrationPriority::kBottom, engine,
                               []() -> std::unique_ptr<BaseCustomOp> { return std::make_unique<TestPriorityOp>(); }),
      GRAPH_SUCCESS);

  auto *top_first = registry.CreateOrGetCustomOp(op_type, OpBackend::kHostCPU, OpRegistrationPriority::kTop, engine);
  auto *top_second = registry.CreateOrGetCustomOp(op_type, OpBackend::kHostCPU, OpRegistrationPriority::kTop, engine);
  auto *bottom = registry.CreateOrGetCustomOp(op_type, OpBackend::kHostCPU, OpRegistrationPriority::kBottom, engine);
  ASSERT_NE(top_first, nullptr);
  EXPECT_EQ(top_first, top_second);
  ASSERT_NE(bottom, nullptr);
  EXPECT_NE(top_first, bottom);
}

TEST(CustomOpRegistryPriorityTest, RejectsInvalidAndDuplicateRegistration) {
  CustomOpRegistry registry;
  const AscendString op_type("CustomOpRegistryPriorityInvalidOp");
  const OpEngine engine = OpEngine::kAiCore;
  EXPECT_EQ(
      registry.RegisterCreator(op_type, OpBackend::kHostCPU, OpRegistrationPriority::kTop, engine, BaseOpCreator()),
      GRAPH_PARAM_INVALID);
  EXPECT_EQ(
      registry.RegisterCreator(op_type, OpBackend::kHostCPU, OpRegistrationPriority::kTop, static_cast<OpEngine>(100U),
                               []() -> std::unique_ptr<BaseCustomOp> { return std::make_unique<TestPriorityOp>(); }),
      GRAPH_PARAM_INVALID);
  ASSERT_EQ(
      registry.RegisterCreator(op_type, OpBackend::kHostCPU, OpRegistrationPriority::kTop, engine,
                               []() -> std::unique_ptr<BaseCustomOp> { return std::make_unique<TestPriorityOp>(); }),
      GRAPH_SUCCESS);
  EXPECT_EQ(
      registry.RegisterCreator(op_type, OpBackend::kHostCPU, OpRegistrationPriority::kTop, engine,
                               []() -> std::unique_ptr<BaseCustomOp> { return std::make_unique<TestPriorityOp>(); }),
      GRAPH_FAILED);
}

TEST(CustomOpRegistryPriorityTest, CreatorExceptionPropagates) {
  CustomOpRegistry registry;
  const AscendString op_type("CustomOpRegistryPriorityExceptionOp");
  const OpEngine engine = OpEngine::kAiCore;
  ASSERT_EQ(
      registry.RegisterCreator(op_type, OpBackend::kHostCPU, OpRegistrationPriority::kBottom, engine,
                               []() -> std::unique_ptr<BaseCustomOp> { throw std::runtime_error("creator failed"); }),
      GRAPH_SUCCESS);

  EXPECT_THROW(
      (void)registry.CreateOrGetCustomOp(op_type, OpBackend::kHostCPU, OpRegistrationPriority::kBottom, engine),
      std::runtime_error);
}

TEST(CustomOpRegistryPriorityTest, MacroRegistersExplicitPriorityCreator) {
  const AscendString op_type("CustomOpRegistryPriorityMacroOp");
  const OpEngine engine = OpEngine::kAiCore;
  EXPECT_TRUE(CustomOpFactory::IsExistOp(op_type, OpBackend::kDevice, OpRegistrationPriority::kBottom, engine));
  EXPECT_NE(CustomOpFactory::CreateOrGetCustomOp(op_type, OpBackend::kDevice, OpRegistrationPriority::kBottom, engine),
            nullptr);
}

TEST(CustomOpRegistryPriorityTest, CommonCapabilityCanQueryExplicitPriorityAndEngine) {
  const AscendString op_type("CustomOpRegistryCapabilityMacroOp");
  const OpEngine engine = OpEngine::kAiCore;
  auto *custom_op =
      CustomOpFactory::GetCustomOpCommonCapability<ShapeInferOp>(op_type, OpRegistrationPriority::kBottom, engine);
  EXPECT_NE(custom_op, nullptr);
  EXPECT_EQ(CustomOpFactory::GetCustomOpCommonCapability<ShapeInferOp>(op_type, OpRegistrationPriority::kTop, engine),
            nullptr);
  EXPECT_EQ(CustomOpFactory::GetCustomOpCommonCapability<ShapeInferOp>(op_type, OpRegistrationPriority::kBottom,
                                                                       OpEngine::kVectorCore),
            nullptr);
}

TEST(CustomOpRegistryPriorityTest, RegOpBackendUsesCustomEngineTopPriority) {
  const AscendString op_type("CustomOpRegistryDefaultMacroOp");
  EXPECT_TRUE(CustomOpFactory::IsExistOp(op_type, OpBackend::kHostCPU));
  EXPECT_TRUE(
      CustomOpFactory::IsExistOp(op_type, OpBackend::kHostCPU, OpRegistrationPriority::kTop, OpEngine::kCustom));
  EXPECT_FALSE(
      CustomOpFactory::IsExistOp(op_type, OpBackend::kHostCPU, OpRegistrationPriority::kBottom, OpEngine::kCustom));
  EXPECT_FALSE(
      CustomOpFactory::IsExistOp(op_type, OpBackend::kHostCPU, OpRegistrationPriority::kTop, OpEngine::kAiCore));
}
}  // namespace ge
