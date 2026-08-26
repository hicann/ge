/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>

#include <algorithm>
#include <new>
#include <string>
#include <vector>

#include "framework/common/framework_types_internal.h"
#include "../graph/custom_ops_stub.h"
#include "graph/custom_op/cast.h"
#include "graph/custom_op_factory.h"
#include "graph/custom_op_registry.h"
#include "runtime/custom_op/python_custom_op_adapter.h"
#include "ge/ge_api_error_codes.h"
#include "macro_utils/dt_public_scope.h"
#include "macro_utils/dt_public_unscope.h"
#include "securec.h"

using namespace ge;
using namespace ge::custom_op;
namespace {
class RegistryTestOp : public BaseCustomOp {};
class RegistryDuplicateReplacementOp : public BaseCustomOp {};
class RegistryHostExecuteOp : public HostCpuExecuteOp {
 public:
  graphStatus Execute(gert::HostCpuOpExecutionContext *ctx) override {
    (void)ctx;
    return GRAPH_SUCCESS;
  }
};

class RegistryShapeInferOp : public ShapeInferOp {
 public:
  graphStatus InferShape(gert::InferShapeContext *ctx) override {
    (void)ctx;
    return GRAPH_SUCCESS;
  }

  graphStatus InferDataType(gert::InferDataTypeContext *ctx) override {
    (void)ctx;
    return GRAPH_SUCCESS;
  }
};

class RegistrySharedShapeInferOp : public ShapeInferOp {
 public:
  graphStatus InferShape(gert::InferShapeContext *ctx) override {
    (void)ctx;
    return GRAPH_SUCCESS;
  }

  graphStatus InferDataType(gert::InferDataTypeContext *ctx) override {
    (void)ctx;
    return GRAPH_SUCCESS;
  }
};

class RegistryDestructCountOp : public BaseCustomOp {
 public:
  explicit RegistryDestructCountOp(size_t *destruct_count) : destruct_count_(destruct_count) {}
  ~RegistryDestructCountOp() override {
    if (destruct_count_ != nullptr) {
      ++(*destruct_count_);
    }
  }

 private:
  size_t *destruct_count_;
};

class CastEagerOnlyOp : public EagerExecuteOp {
 public:
  graphStatus Execute(gert::EagerOpExecutionContext *ctx) override {
    (void)ctx;
    return GRAPH_SUCCESS;
  }
};

class RegistryPortableOp : public PortableOp {
 public:
  graphStatus Serialize(std::vector<uint8_t> &buffer) override {
    buffer = {1U, 2U, 3U};
    return GRAPH_SUCCESS;
  }

  graphStatus Deserialize(const std::vector<uint8_t> &buffer) override {
    deserialized_buffer = buffer;
    return GRAPH_SUCCESS;
  }

  std::vector<uint8_t> deserialized_buffer;
};

class FactoryCallbackPortableOp : public PortableOp {
 public:
  graphStatus Serialize(std::vector<uint8_t> &buffer) override {
    buffer.clear();
    return GRAPH_SUCCESS;
  }

  graphStatus Deserialize(const std::vector<uint8_t> &buffer) override {
    (void)buffer;
    const auto dependency_op = CustomOpFactory::CreateOrGetCustomOp("FactoryCallbackDependencyOp", OpBackend::kDevice);
    return (dependency_op == nullptr) ? GRAPH_FAILED : GRAPH_SUCCESS;
  }
};

struct MockPythonCustomOpHolder {
  bool executed{false};
};

void *CreateMockPythonCustomOpHolder(const PythonCustomOpAdapterDescriptorView *desc) {
  if ((desc == nullptr) || (desc->impl_descriptor_key.data == nullptr)) {
    return nullptr;
  }
  return new (std::nothrow) MockPythonCustomOpHolder();
}

void DestroyMockPythonCustomOpHolder(void *holder) {
  delete static_cast<MockPythonCustomOpHolder *>(holder);
}

graphStatus ExecuteMockPythonCustomOp(const void *holder, gert::EagerOpExecutionContext *ctx) {
  (void)ctx;
  auto *mock_holder = const_cast<MockPythonCustomOpHolder *>(static_cast<const MockPythonCustomOpHolder *>(holder));
  if (mock_holder == nullptr) {
    return GRAPH_FAILED;
  }
  mock_holder->executed = true;
  return GRAPH_SUCCESS;
}

graphStatus DeclareMockPythonCustomOp(const void *holder, gert::AnnotatedArgsContext *ctx) {
  (void)ctx;
  return (holder == nullptr) ? GRAPH_FAILED : GRAPH_SUCCESS;
}

graphStatus CompileMockPythonCustomOp(const void *holder, gert::OpCompileContext *ctx) {
  return ((holder != nullptr) && (ctx != nullptr)) ? GRAPH_SUCCESS : GRAPH_FAILED;
}

void *FailCreatePythonCustomOpHolder(const PythonCustomOpAdapterDescriptorView *) {
  return nullptr;
}

graphStatus FailExecutePythonCustomOp(const void *, gert::EagerOpExecutionContext *) {
  return GRAPH_FAILED;
}

std::vector<uint8_t> BuildCustomOpPartition(const std::string &name, const std::vector<uint8_t> &bin) {
  ge::CustomKernelItemHeader header{ge::kCustomKernelItemMagic, static_cast<uint32_t>(name.size()),
                                    static_cast<uint32_t>(bin.size())};
  std::vector<uint8_t> payload(sizeof(header) + name.size() + bin.size(), 0U);
  (void)memcpy_s(payload.data(), payload.size(), &header, sizeof(header));
  (void)memcpy_s(payload.data() + sizeof(header), payload.size() - sizeof(header), name.data(), name.size());
  if (!bin.empty()) {
    (void)memcpy_s(payload.data() + sizeof(header) + name.size(), payload.size() - sizeof(header) - name.size(),
                   bin.data(), bin.size());
  }
  return payload;
}
}  // namespace

class UtestCustomOpFactory : public testing::Test {
 protected:
  void SetUp() {}

  void TearDown() {}
};

TEST(UtestCustomOpRegistry, rejects_null_creator) {
  CustomOpRegistry registry;
  EXPECT_EQ(ge::GRAPH_PARAM_INVALID, registry.RegisterCreator("RegistryNullCreator", OpBackend::kDevice, nullptr));
  EXPECT_EQ(false, registry.HasCreator("RegistryNullCreator"));
}

TEST(UtestCustomOpRegistry, rejects_invalid_backend) {
  CustomOpRegistry registry;
  const auto invalid_backend = static_cast<OpBackend>(99U);
  EXPECT_EQ(ge::GRAPH_PARAM_INVALID,
            registry.RegisterCreator("RegistryInvalidBackend", invalid_backend, []() -> std::unique_ptr<BaseCustomOp> {
              return std::make_unique<RegistryTestOp>();
            }));
  EXPECT_EQ(nullptr, registry.CreateOrGetCustomOp("RegistryInvalidBackend", invalid_backend));
}

TEST(UtestCustomOpRegistry, compatibility_overloads_use_device_backend) {
  CustomOpRegistry registry;
  EXPECT_EQ(ge::GRAPH_SUCCESS, registry.RegisterCreator("RegistryCompatibilityOverloads", OpBackend::kDevice,
                                                        []() -> std::unique_ptr<BaseCustomOp> {
                                                          return std::make_unique<RegistryTestOp>();
                                                        }));

  const auto *created = registry.CreateOrGetCustomOp("RegistryCompatibilityOverloads", OpBackend::kDevice);
  EXPECT_NE(nullptr, created);
  EXPECT_TRUE(registry.HasCustomOp("RegistryCompatibilityOverloads"));
  EXPECT_EQ(registry.CreateOrGetCustomOp("RegistryCompatibilityOverloads", OpBackend::kHostCPU), nullptr);
}

TEST(UtestCustomOpRegistry, creator_returning_null_does_not_create_custom_op) {
  CustomOpRegistry registry;
  EXPECT_EQ(ge::GRAPH_SUCCESS, registry.RegisterCreator("RegistryCreatorReturnsNull", OpBackend::kDevice,
                                                        []() -> std::unique_ptr<BaseCustomOp> { return nullptr; }));

  EXPECT_EQ(nullptr, registry.CreateOrGetCustomOp("RegistryCreatorReturnsNull", OpBackend::kDevice));
  EXPECT_FALSE(registry.HasCustomOp("RegistryCreatorReturnsNull"));
}

TEST(UtestCustomOpRegistry, rejects_duplicate_creator) {
  CustomOpRegistry registry;
  const auto creator = []() -> std::unique_ptr<BaseCustomOp> { return std::make_unique<RegistryTestOp>(); };
  EXPECT_EQ(ge::GRAPH_SUCCESS, registry.RegisterCreator("RegistryDuplicateCreator", OpBackend::kDevice, creator));
  EXPECT_EQ(ge::GRAPH_FAILED, registry.RegisterCreator("RegistryDuplicateCreator", OpBackend::kDevice, creator));
}

TEST(UtestCustomOpRegistry, same_op_type_allows_different_backends) {
  CustomOpRegistry registry;
  EXPECT_EQ(ge::GRAPH_SUCCESS, registry.RegisterCreator("RegistryMultiBackendOp", OpBackend::kDevice,
                                                        []() -> std::unique_ptr<BaseCustomOp> {
                                                          return std::make_unique<RegistryTestOp>();
                                                        }));
  EXPECT_EQ(ge::GRAPH_SUCCESS, registry.RegisterCreator("RegistryMultiBackendOp", OpBackend::kHostCPU,
                                                        []() -> std::unique_ptr<BaseCustomOp> {
                                                          return std::make_unique<RegistryHostExecuteOp>();
                                                        }));

  EXPECT_TRUE(registry.HasCreator("RegistryMultiBackendOp"));
  EXPECT_TRUE(registry.HasCreator("RegistryMultiBackendOp", OpBackend::kDevice));
  EXPECT_TRUE(registry.HasCreator("RegistryMultiBackendOp", OpBackend::kHostCPU));
  EXPECT_NE(nullptr,
            dynamic_cast<RegistryTestOp *>(registry.CreateOrGetCustomOp("RegistryMultiBackendOp", OpBackend::kDevice)));
  EXPECT_NE(nullptr, dynamic_cast<RegistryHostExecuteOp *>(
                         registry.CreateOrGetCustomOp("RegistryMultiBackendOp", OpBackend::kHostCPU)));
  EXPECT_NE(nullptr,
            dynamic_cast<RegistryTestOp *>(registry.CreateOrGetCustomOp("RegistryMultiBackendOp", OpBackend::kDevice)));
}

TEST(UtestCustomOpRegistry, remove_custom_ops_removes_all_backends) {
  CustomOpRegistry registry;
  EXPECT_EQ(ge::GRAPH_SUCCESS, registry.RegisterCreator("RegistryRemoveMultiBackendOp", OpBackend::kDevice,
                                                        []() -> std::unique_ptr<BaseCustomOp> {
                                                          return std::make_unique<RegistryTestOp>();
                                                        }));
  EXPECT_EQ(ge::GRAPH_SUCCESS, registry.RegisterCreator("RegistryRemoveMultiBackendOp", OpBackend::kHostCPU,
                                                        []() -> std::unique_ptr<BaseCustomOp> {
                                                          return std::make_unique<RegistryHostExecuteOp>();
                                                        }));

  registry.RemoveCustomOps({AscendString("RegistryRemoveMultiBackendOp")});
  EXPECT_FALSE(registry.HasCreator("RegistryRemoveMultiBackendOp", OpBackend::kDevice));
  EXPECT_FALSE(registry.HasCreator("RegistryRemoveMultiBackendOp", OpBackend::kHostCPU));
  EXPECT_FALSE(registry.HasCreator("RegistryRemoveMultiBackendOp"));
}

TEST(UtestCustomOpRegistry, get_all_registered_ops_returns_all_backends) {
  CustomOpRegistry registry;
  EXPECT_EQ(ge::GRAPH_SUCCESS, registry.RegisterCreator("RegistryAllOpsDeviceOnly", OpBackend::kDevice,
                                                        []() -> std::unique_ptr<BaseCustomOp> {
                                                          return std::make_unique<RegistryTestOp>();
                                                        }));
  EXPECT_EQ(ge::GRAPH_SUCCESS, registry.RegisterCreator("RegistryAllOpsHostOnly", OpBackend::kHostCPU,
                                                        []() -> std::unique_ptr<BaseCustomOp> {
                                                          return std::make_unique<RegistryHostExecuteOp>();
                                                        }));

  std::vector<AscendString> all_ops;
  EXPECT_EQ(ge::GRAPH_SUCCESS, registry.GetAllRegisteredOps(all_ops));
  EXPECT_TRUE(std::any_of(all_ops.begin(), all_ops.end(),
                          [](const AscendString &op_name) { return op_name == "RegistryAllOpsDeviceOnly"; }));
  EXPECT_TRUE(std::any_of(all_ops.begin(), all_ops.end(),
                          [](const AscendString &op_name) { return op_name == "RegistryAllOpsHostOnly"; }));
}

TEST(UtestCustomOpRegistry, get_custom_op_common_capability_returns_unique_owner) {
  CustomOpRegistry registry;
  EXPECT_EQ(ge::GRAPH_SUCCESS, registry.RegisterCreator("RegistryCapabilityOwnerOp", OpBackend::kDevice,
                                                        []() -> std::unique_ptr<BaseCustomOp> {
                                                          return std::make_unique<RegistryShapeInferOp>();
                                                        }));
  EXPECT_EQ(ge::GRAPH_SUCCESS, registry.RegisterCreator("RegistryCapabilityOwnerOp", OpBackend::kHostCPU,
                                                        []() -> std::unique_ptr<BaseCustomOp> {
                                                          return std::make_unique<RegistryHostExecuteOp>();
                                                        }));

  EXPECT_NE(nullptr, dynamic_cast<RegistryShapeInferOp *>(registry.GetCustomOpCommonCapability(
                         "RegistryCapabilityOwnerOp", CustomOpCapability::kShapeInfer)));
  EXPECT_EQ(nullptr,
            registry.GetCustomOpCommonCapability("RegistryCapabilityOwnerOp", CustomOpCapability::kHostCpuExecute));
  EXPECT_NE(nullptr, dynamic_cast<RegistryHostExecuteOp *>(
                         registry.CreateOrGetCustomOp("RegistryCapabilityOwnerOp", OpBackend::kHostCPU)));
}

TEST(UtestCustomOpRegistry, get_custom_op_common_capability_shares_same_instance_across_backends) {
  CustomOpRegistry registry;
  EXPECT_EQ(ge::GRAPH_SUCCESS, registry.RegisterCreator("RegistrySharedCapabilityOp", OpBackend::kDevice,
                                                        []() -> std::unique_ptr<BaseCustomOp> {
                                                          return std::make_unique<RegistrySharedShapeInferOp>();
                                                        }));
  EXPECT_EQ(ge::GRAPH_SUCCESS, registry.RegisterCreator("RegistrySharedCapabilityOp", OpBackend::kHostCPU,
                                                        []() -> std::unique_ptr<BaseCustomOp> {
                                                          return std::make_unique<RegistrySharedShapeInferOp>();
                                                        }));

  auto *device_op = registry.CreateOrGetCustomOp("RegistrySharedCapabilityOp", OpBackend::kDevice);
  auto *host_op = registry.CreateOrGetCustomOp("RegistrySharedCapabilityOp", OpBackend::kHostCPU);
  ASSERT_NE(nullptr, device_op);
  EXPECT_EQ(device_op, host_op);
  EXPECT_EQ(device_op,
            registry.GetCustomOpCommonCapability("RegistrySharedCapabilityOp", CustomOpCapability::kShapeInfer));
}

TEST(UtestCustomOpRegistry, get_custom_op_common_capability_rejects_different_types_across_backends) {
  CustomOpRegistry registry;
  EXPECT_EQ(ge::GRAPH_SUCCESS, registry.RegisterCreator("RegistryCapabilityConflictOp", OpBackend::kDevice,
                                                        []() -> std::unique_ptr<BaseCustomOp> {
                                                          return std::make_unique<RegistryShapeInferOp>();
                                                        }));
  EXPECT_EQ(ge::GRAPH_SUCCESS, registry.RegisterCreator("RegistryCapabilityConflictOp", OpBackend::kHostCPU,
                                                        []() -> std::unique_ptr<BaseCustomOp> {
                                                          return std::make_unique<RegistrySharedShapeInferOp>();
                                                        }));

  EXPECT_EQ(nullptr,
            registry.GetCustomOpCommonCapability("RegistryCapabilityConflictOp", CustomOpCapability::kShapeInfer));
}

TEST(UtestCustomOpRegistry, duplicate_creator_keeps_original_creator) {
  CustomOpRegistry registry;
  EXPECT_EQ(ge::GRAPH_SUCCESS, registry.RegisterCreator("RegistryDuplicateKeepsOriginal", OpBackend::kDevice,
                                                        []() -> std::unique_ptr<BaseCustomOp> {
                                                          return std::make_unique<RegistryTestOp>();
                                                        }));
  EXPECT_EQ(ge::GRAPH_FAILED, registry.RegisterCreator("RegistryDuplicateKeepsOriginal", OpBackend::kDevice,
                                                       []() -> std::unique_ptr<BaseCustomOp> {
                                                         return std::make_unique<RegistryDuplicateReplacementOp>();
                                                       }));

  const auto op = registry.CreateOrGetCustomOp("RegistryDuplicateKeepsOriginal", OpBackend::kDevice);
  EXPECT_NE(nullptr, dynamic_cast<RegistryTestOp *>(op));
  EXPECT_EQ(nullptr, dynamic_cast<RegistryDuplicateReplacementOp *>(op));
}

TEST(UtestCustomOpRegistry, create_or_get_returns_same_instance) {
  CustomOpRegistry registry;
  EXPECT_EQ(ge::GRAPH_SUCCESS,
            registry.RegisterCreator("RegistryCreateOnce", OpBackend::kDevice, []() -> std::unique_ptr<BaseCustomOp> {
              return std::make_unique<RegistryTestOp>();
            }));

  const auto first = registry.CreateOrGetCustomOp("RegistryCreateOnce", OpBackend::kDevice);
  const auto second = registry.CreateOrGetCustomOp("RegistryCreateOnce", OpBackend::kDevice);
  EXPECT_NE(nullptr, first);
  EXPECT_EQ(first, second);
}

TEST(UtestCustomOpRegistry, remove_custom_ops_erases_creator_and_created_instance) {
  size_t destruct_count = 0U;
  CustomOpRegistry registry;
  EXPECT_EQ(ge::GRAPH_SUCCESS,
            registry.RegisterCreator("RegistryRemoveCustomOp", OpBackend::kDevice,
                                     [&destruct_count]() -> std::unique_ptr<BaseCustomOp> {
                                       return std::make_unique<RegistryDestructCountOp>(&destruct_count);
                                     }));

  EXPECT_EQ(true, registry.HasCreator("RegistryRemoveCustomOp"));
  EXPECT_NE(nullptr, registry.CreateOrGetCustomOp("RegistryRemoveCustomOp", OpBackend::kDevice));
  EXPECT_EQ(true, registry.HasCustomOp("RegistryRemoveCustomOp"));

  registry.RemoveCustomOps({AscendString("RegistryRemoveCustomOp")});
  EXPECT_EQ(1U, destruct_count);
  EXPECT_EQ(false, registry.HasCreator("RegistryRemoveCustomOp"));
  EXPECT_EQ(false, registry.HasCustomOp("RegistryRemoveCustomOp"));
  EXPECT_EQ(nullptr, registry.CreateOrGetCustomOp("RegistryRemoveCustomOp", OpBackend::kDevice));
}

TEST(UtestCustomOpCast, falls_back_to_dynamic_cast_for_cpp_custom_op) {
  CastEagerOnlyOp op;
  BaseCustomOp *base = &op;

  EXPECT_EQ(static_cast<EagerExecuteOp *>(&op), CustomOpCast<EagerExecuteOp>(base));
  EXPECT_EQ(nullptr, CustomOpCast<CompilableOp>(base));
  EXPECT_EQ(nullptr, CustomOpCast<ShapeInferOp>(base));
}

TEST(UtestCustomOpCast, supports_host_cpu_execute_op) {
  RegistryHostExecuteOp op;
  BaseCustomOp *base = &op;

  EXPECT_EQ(static_cast<HostCpuExecuteOp *>(&op), CustomOpCast<HostCpuExecuteOp>(base));
  EXPECT_EQ(nullptr, CustomOpCast<EagerExecuteOp>(base));
}

TEST(UtestCustomOpCast, filters_python_adapter_by_capability) {
  PythonCustomOpAdapterDescriptor desc;
  desc.impl_descriptor_key = "python_adapter_eager_only";
  desc.op_type = "PythonAdapterEagerOnly";
  AddCustomOpCapability(desc.capabilities, CustomOpCapability::kEagerExecute);

  PythonCustomOpAdapterCallbacks callbacks;
  callbacks.create_impl_holder = CreateMockPythonCustomOpHolder;
  callbacks.destroy_impl_holder = DestroyMockPythonCustomOpHolder;
  callbacks.execute = ExecuteMockPythonCustomOp;

  ASSERT_TRUE(PythonCustomOpImplRuntimeRegistry::Register(desc, callbacks));
  {
    PythonCustomOpAdapter adapter(desc);
    EXPECT_TRUE(adapter.IsValid());

    BaseCustomOp *base = &adapter;
    EXPECT_NE(nullptr, CustomOpCast<EagerExecuteOp>(base));
    EXPECT_EQ(nullptr, CustomOpCast<AnnotatedArgsOp>(base));
    EXPECT_EQ(nullptr, CustomOpCast<CompilableOp>(base));
    EXPECT_EQ(nullptr, CustomOpCast<ShapeInferOp>(base));
    EXPECT_EQ(nullptr, CustomOpCast<PortableOp>(base));
    EXPECT_EQ(nullptr, CustomOpCast<ArgsUpdater>(base));

    EXPECT_EQ(GRAPH_SUCCESS, CustomOpCast<EagerExecuteOp>(base)->Execute(nullptr));
    EXPECT_EQ(GRAPH_FAILED, adapter.Compile(nullptr));
    EXPECT_FALSE(PythonCustomOpImplRuntimeRegistry::Unregister(desc.impl_descriptor_key));
  }
  EXPECT_TRUE(PythonCustomOpImplRuntimeRegistry::Unregister(desc.impl_descriptor_key));
}

TEST(UtestCustomOpCast, filters_python_adapter_annotated_args_by_capability) {
  PythonCustomOpAdapterDescriptor desc;
  desc.impl_descriptor_key = "python_adapter_annotated_args_only";
  desc.op_type = "PythonAdapterAnnotatedArgsOnly";
  AddCustomOpCapability(desc.capabilities, CustomOpCapability::kAnnotatedArgs);

  PythonCustomOpAdapterCallbacks callbacks;
  callbacks.create_impl_holder = CreateMockPythonCustomOpHolder;
  callbacks.destroy_impl_holder = DestroyMockPythonCustomOpHolder;
  callbacks.declare_launch_args = DeclareMockPythonCustomOp;

  ASSERT_TRUE(PythonCustomOpImplRuntimeRegistry::Register(desc, callbacks));
  {
    PythonCustomOpAdapter adapter(desc);
    EXPECT_TRUE(adapter.IsValid());

    BaseCustomOp *base = &adapter;
    EXPECT_EQ(nullptr, CustomOpCast<EagerExecuteOp>(base));
    EXPECT_NE(nullptr, CustomOpCast<AnnotatedArgsOp>(base));
    EXPECT_EQ(nullptr, CustomOpCast<CompilableOp>(base));
  }
  EXPECT_TRUE(PythonCustomOpImplRuntimeRegistry::Unregister(desc.impl_descriptor_key));
}

TEST(UtestCustomOpCast, exposes_each_python_adapter_capability_in_dual_mode) {
  PythonCustomOpAdapterDescriptor desc;
  desc.impl_descriptor_key = "python_adapter_dual_capability";
  desc.op_type = "PythonAdapterDualCapability";
  AddCustomOpCapability(desc.capabilities, CustomOpCapability::kEagerExecute);
  AddCustomOpCapability(desc.capabilities, CustomOpCapability::kAnnotatedArgs);

  PythonCustomOpAdapterCallbacks callbacks;
  callbacks.create_impl_holder = CreateMockPythonCustomOpHolder;
  callbacks.destroy_impl_holder = DestroyMockPythonCustomOpHolder;
  callbacks.execute = ExecuteMockPythonCustomOp;
  callbacks.declare_launch_args = DeclareMockPythonCustomOp;

  ASSERT_TRUE(PythonCustomOpImplRuntimeRegistry::Register(desc, callbacks));
  {
    PythonCustomOpAdapter adapter(desc);
    EXPECT_TRUE(adapter.IsValid());
    BaseCustomOp *base = &adapter;
    EXPECT_NE(nullptr, CustomOpCast<EagerExecuteOp>(base));
    EXPECT_NE(nullptr, CustomOpCast<AnnotatedArgsOp>(base));
  }
  EXPECT_TRUE(PythonCustomOpImplRuntimeRegistry::Unregister(desc.impl_descriptor_key));
}

TEST(UtestCustomOpCast, exposes_python_adapter_compilable_capability) {
  PythonCustomOpAdapterDescriptor desc;
  desc.impl_descriptor_key = "python_adapter_compilable";
  desc.op_type = "PythonAdapterCompilable";
  AddCustomOpCapability(desc.capabilities, CustomOpCapability::kCompilable);

  PythonCustomOpAdapterCallbacks callbacks;
  callbacks.create_impl_holder = CreateMockPythonCustomOpHolder;
  callbacks.destroy_impl_holder = DestroyMockPythonCustomOpHolder;
  callbacks.compile_impl = CompileMockPythonCustomOp;

  ASSERT_TRUE(PythonCustomOpImplRuntimeRegistry::Register(desc, callbacks));
  {
    PythonCustomOpAdapter adapter(desc);
    EXPECT_TRUE(adapter.IsValid());

    BaseCustomOp *base = &adapter;
    EXPECT_EQ(nullptr, CustomOpCast<EagerExecuteOp>(base));
    EXPECT_NE(nullptr, CustomOpCast<CompilableOp>(base));
    EXPECT_EQ(GRAPH_FAILED, CustomOpCast<CompilableOp>(base)->Compile(nullptr));
  }
  EXPECT_TRUE(PythonCustomOpImplRuntimeRegistry::Unregister(desc.impl_descriptor_key));
}

TEST(UtestCustomOpCast, rejects_unsupported_python_adapter_capability) {
  PythonCustomOpAdapterDescriptor desc;
  desc.impl_descriptor_key = "python_adapter_shape_unsupported";
  desc.op_type = "PythonAdapterShapeUnsupported";
  AddCustomOpCapability(desc.capabilities, CustomOpCapability::kShapeInfer);

  PythonCustomOpAdapterCallbacks callbacks;
  callbacks.create_impl_holder = CreateMockPythonCustomOpHolder;
  callbacks.destroy_impl_holder = DestroyMockPythonCustomOpHolder;

  EXPECT_FALSE(PythonCustomOpImplRuntimeRegistry::Register(desc, callbacks));
}

TEST(UtestCustomOpCast, rejects_python_adapter_with_missing_impl_key_or_callback) {
  PythonCustomOpAdapterDescriptor desc;
  desc.impl_descriptor_key = "python_adapter_required_fields";
  desc.op_type = "PythonAdapterRequiredFields";
  AddCustomOpCapability(desc.capabilities, CustomOpCapability::kEagerExecute);

  PythonCustomOpAdapterCallbacks callbacks;
  callbacks.create_impl_holder = CreateMockPythonCustomOpHolder;
  callbacks.destroy_impl_holder = DestroyMockPythonCustomOpHolder;
  callbacks.execute = ExecuteMockPythonCustomOp;
  auto invalid_callbacks = callbacks;
  invalid_callbacks.create_impl_holder = nullptr;
  EXPECT_FALSE(invalid_callbacks.IsValid(desc.capabilities));
  invalid_callbacks = callbacks;
  invalid_callbacks.destroy_impl_holder = nullptr;
  EXPECT_FALSE(invalid_callbacks.IsValid(desc.capabilities));
  invalid_callbacks = callbacks;
  invalid_callbacks.execute = nullptr;
  EXPECT_FALSE(invalid_callbacks.IsValid(desc.capabilities));
}

TEST(UtestCustomOpCast, releases_runtime_lease_when_holder_creation_fails) {
  PythonCustomOpAdapterCallbacks callbacks;
  callbacks.create_impl_holder = CreateMockPythonCustomOpHolder;
  callbacks.destroy_impl_holder = DestroyMockPythonCustomOpHolder;
  callbacks.execute = ExecuteMockPythonCustomOp;
  PythonCustomOpAdapterDescriptor create_desc;
  create_desc.impl_descriptor_key = "python_adapter_fail_create";
  create_desc.op_type = "PythonAdapterFailCreate";
  AddCustomOpCapability(create_desc.capabilities, CustomOpCapability::kEagerExecute);
  callbacks.create_impl_holder = FailCreatePythonCustomOpHolder;
  ASSERT_TRUE(PythonCustomOpImplRuntimeRegistry::Register(create_desc, callbacks));
  {
    PythonCustomOpAdapter adapter(create_desc);
    EXPECT_FALSE(adapter.IsValid());
  }
  EXPECT_TRUE(PythonCustomOpImplRuntimeRegistry::Unregister(create_desc.impl_descriptor_key));

  PythonCustomOpAdapterDescriptor execute_desc;
  execute_desc.impl_descriptor_key = "python_adapter_fail_execute";
  execute_desc.op_type = "PythonAdapterFailExecute";
  AddCustomOpCapability(execute_desc.capabilities, CustomOpCapability::kEagerExecute);
  callbacks.create_impl_holder = CreateMockPythonCustomOpHolder;
  callbacks.destroy_impl_holder = DestroyMockPythonCustomOpHolder;
  callbacks.execute = FailExecutePythonCustomOp;
  ASSERT_TRUE(PythonCustomOpImplRuntimeRegistry::Register(execute_desc, callbacks));
  {
    PythonCustomOpAdapter adapter(execute_desc);
    ASSERT_TRUE(adapter.IsValid());
    EXPECT_EQ(adapter.Execute(nullptr), GRAPH_FAILED);
  }
  EXPECT_TRUE(PythonCustomOpImplRuntimeRegistry::Unregister(execute_desc.impl_descriptor_key));
}

TEST(UtestCustomOpRegistry, load_custom_ops_partition_deserializes_registered_portable_op) {
  CustomOpRegistry registry;
  EXPECT_EQ(ge::GRAPH_SUCCESS, registry.RegisterCreator("RegistryPortablePartition", OpBackend::kDevice,
                                                        []() -> std::unique_ptr<BaseCustomOp> {
                                                          return std::make_unique<RegistryPortableOp>();
                                                        }));
  const std::vector<uint8_t> kernel_bin = {0x1U, 0x2U, 0x3U};
  const auto payload = BuildCustomOpPartition("RegistryPortablePartition", kernel_bin);

  EXPECT_EQ(ge::GRAPH_SUCCESS, registry.LoadCustomOpsPartition(payload.data(), payload.size()));
  const auto *op =
      dynamic_cast<RegistryPortableOp *>(registry.CreateOrGetCustomOp("RegistryPortablePartition", OpBackend::kDevice));
  ASSERT_NE(nullptr, op);
  EXPECT_EQ(kernel_bin, op->deserialized_buffer);
}

TEST(UtestCustomOpFactory, facade_registers_and_creates_through_global_registry) {
  const auto ret = CustomOpFactory::RegisterCustomOpCreator(
      "FactoryGlobalRegistryOp", []() -> std::unique_ptr<BaseCustomOp> { return std::make_unique<RegistryTestOp>(); });
  EXPECT_TRUE((ret == ge::GRAPH_SUCCESS) || (ret == ge::GRAPH_FAILED));

  EXPECT_EQ(true, CustomOpFactory::IsExistOp("FactoryGlobalRegistryOp"));
  EXPECT_NE(nullptr, CustomOpFactory::CreateOrGetCustomOp("FactoryGlobalRegistryOp", OpBackend::kDevice));
}

TEST(UtestCustomOpFactory, load_custom_ops_partition_allows_factory_callback_from_deserialize) {
  const auto dependency_ret = CustomOpFactory::RegisterCustomOpCreator(
      "FactoryCallbackDependencyOp",
      []() -> std::unique_ptr<BaseCustomOp> { return std::make_unique<RegistryTestOp>(); });
  EXPECT_TRUE((dependency_ret == ge::GRAPH_SUCCESS) || (dependency_ret == ge::GRAPH_FAILED));

  const auto callback_ret = CustomOpFactory::RegisterCustomOpCreator(
      "FactoryCallbackPortableOp",
      []() -> std::unique_ptr<BaseCustomOp> { return std::make_unique<FactoryCallbackPortableOp>(); });
  EXPECT_TRUE((callback_ret == ge::GRAPH_SUCCESS) || (callback_ret == ge::GRAPH_FAILED));

  const auto payload = BuildCustomOpPartition("FactoryCallbackPortableOp", {0x1U});
  EXPECT_EQ(ge::GRAPH_SUCCESS, CustomOpFactory::LoadCustomOpsPartition(payload.data(), payload.size()));
}

TEST(UtestCustomOpFactory, create_or_get_custom_op) {
  EXPECT_EQ(true, CustomOpFactory::IsExistOp("MyEagerExecuteOp"));
  EXPECT_EQ(true, CustomOpFactory::IsExistOp("MyPortableOp"));
  const auto op = CustomOpFactory::CreateOrGetCustomOp("MyEagerExecuteOp", OpBackend::kDevice);
  const auto op2 = CustomOpFactory::CreateOrGetCustomOp("NonExists", OpBackend::kDevice);
  EXPECT_EQ(true, op != nullptr);
  EXPECT_EQ(true, op2 == nullptr);
}

TEST(UtestCustomOpFactory, get_all_ops) {
  std::vector<AscendString> all_registered_ops;
  const auto ret = CustomOpFactory::GetAllRegisteredOps(all_registered_ops);
  EXPECT_EQ(ge::SUCCESS, ret);
  const auto has_my_custom_op = std::any_of(all_registered_ops.begin(), all_registered_ops.end(),
                                            [](const AscendString &op_name) { return op_name == "MyEagerExecuteOp"; });
  const auto has_my_serializable_custom_op =
      std::any_of(all_registered_ops.begin(), all_registered_ops.end(),
                  [](const AscendString &op_name) { return op_name == "MyPortableOp"; });
  EXPECT_EQ(true, has_my_custom_op);
  EXPECT_EQ(true, has_my_serializable_custom_op);
  EXPECT_GE(all_registered_ops.size(), 4U);
}

TEST(UtestCustomOpFactory, register_custom_op_creator) {
  const auto reg_ret = CustomOpFactory::RegisterCustomOpCreator(
      "MyCustomOp3", []() -> std::unique_ptr<BaseCustomOp> { return std::make_unique<MyEagerExecuteOp>(); });
  const auto reg_null_ret = CustomOpFactory::RegisterCustomOpCreator("MyCustomOp4", nullptr);
  EXPECT_TRUE((reg_ret == ge::SUCCESS) || (reg_ret == ge::GRAPH_FAILED));
  EXPECT_EQ(ge::GRAPH_PARAM_INVALID, reg_null_ret);
  EXPECT_EQ(true, CustomOpFactory::IsExistOp("MyCustomOp3"));
  EXPECT_EQ(false, CustomOpFactory::IsExistOp("MyCustomOp4"));
}

TEST(UtestCustomOpFactory, creator_register) {
  CustomOpCreatorRegister("MyCustomOp5", []() -> std::unique_ptr<BaseCustomOp> { return nullptr; });
  CustomOpCreatorRegister("MyCustomOp5", []() -> std::unique_ptr<BaseCustomOp> { return nullptr; });
  EXPECT_EQ(true, CustomOpFactory::IsExistOp("MyCustomOp5"));
  EXPECT_EQ(nullptr, CustomOpFactory::CreateOrGetCustomOp("MyCustomOp5", OpBackend::kDevice));
}

TEST(UtestCustomOpFactory, test_compilable_op) {
  const auto my_custom_op_2 = CustomOpFactory::CreateOrGetCustomOp("MyEagerExecuteOp", OpBackend::kDevice);
  const auto my_custom_op_3 = CustomOpFactory::CreateOrGetCustomOp("MyCompilableOp", OpBackend::kDevice);
  const auto legacy_custom_op = CustomOpFactory::CreateOrGetCustomOp("MyCompilableOp");
  EXPECT_EQ(legacy_custom_op, my_custom_op_3);
  const auto compilable_op_2 = dynamic_cast<CompilableOp *>(my_custom_op_2);
  const auto compilable_op_3 = dynamic_cast<CompilableOp *>(my_custom_op_3);
  EXPECT_EQ(true, compilable_op_2 == nullptr);
  EXPECT_EQ(true, compilable_op_3 != nullptr);
  compilable_op_3->Compile(nullptr);
  const auto casted_my_custom_op_3 = dynamic_cast<MyCompilableOp *>(my_custom_op_3);
  std::string res;
  casted_my_custom_op_3->GetStubCompiledPath(res);
  EXPECT_EQ("stub_compiled_path", res);
}

TEST(UtestCustomOpFactory, create_or_get_returns_same_instance) {
  const auto base_custom_op = CustomOpFactory::CreateOrGetCustomOp("MyCompilableOp", OpBackend::kDevice);
  const auto base_custom_op2 = CustomOpFactory::CreateOrGetCustomOp("MyCompilableOp", OpBackend::kDevice);
  EXPECT_EQ(true, base_custom_op == base_custom_op2);
}

TEST(UtestCustomOpFactory, load_custom_kernels_partition_invalid_data) {
  uint8_t payload[1] = {0U};
  EXPECT_EQ(ge::GRAPH_PARAM_INVALID, CustomOpFactory::LoadCustomOpsPartition(nullptr, sizeof(payload)));
  EXPECT_EQ(ge::GRAPH_PARAM_INVALID, CustomOpFactory::LoadCustomOpsPartition(payload, 0U));
}

TEST(UtestCustomOpFactory, load_custom_kernels_partition_header_too_short_fail) {
  uint8_t payload[2] = {0x1U, 0x2U};
  EXPECT_EQ(ge::GRAPH_FAILED, CustomOpFactory::LoadCustomOpsPartition(payload, sizeof(payload)));
}

TEST(UtestCustomOpFactory, load_custom_kernels_partition_invalid_magic_fail) {
  const std::string name = "MyPortableOp";
  ge::CustomKernelItemHeader header{0x12345678U, static_cast<uint32_t>(name.size()), 3U};
  std::vector<uint8_t> payload(sizeof(header) + name.size() + 3U, 0U);
  (void)memcpy_s(payload.data(), payload.size(), &header, sizeof(header));
  (void)memcpy_s(payload.data() + sizeof(header), payload.size() - sizeof(header), name.data(), name.size());
  EXPECT_EQ(ge::GRAPH_FAILED, CustomOpFactory::LoadCustomOpsPartition(payload.data(), payload.size()));
}

TEST(UtestCustomOpFactory, load_custom_kernels_partition_unregistered_op_fail) {
  const std::string name = "NoSuchCustomOpX";
  ge::CustomKernelItemHeader header{ge::kCustomKernelItemMagic, static_cast<uint32_t>(name.size()), 1U};
  std::vector<uint8_t> payload(sizeof(header) + name.size() + 1U, 0U);
  (void)memcpy_s(payload.data(), payload.size(), &header, sizeof(header));
  (void)memcpy_s(payload.data() + sizeof(header), payload.size() - sizeof(header), name.data(), name.size());
  payload.back() = 0x1U;
  EXPECT_EQ(ge::GRAPH_FAILED, CustomOpFactory::LoadCustomOpsPartition(payload.data(), payload.size()));
}

TEST(UtestCustomOpFactory, load_custom_kernels_partition_not_portable_op_fail) {
  const std::string name = "MyCustomOp";
  ge::CustomKernelItemHeader header{ge::kCustomKernelItemMagic, static_cast<uint32_t>(name.size()), 1U};
  std::vector<uint8_t> payload(sizeof(header) + name.size() + 1U, 0U);
  (void)memcpy_s(payload.data(), payload.size(), &header, sizeof(header));
  (void)memcpy_s(payload.data() + sizeof(header), payload.size() - sizeof(header), name.data(), name.size());
  payload.back() = 0x1U;
  EXPECT_EQ(ge::GRAPH_FAILED, CustomOpFactory::LoadCustomOpsPartition(payload.data(), payload.size()));
}

TEST(UtestCustomOpFactory, load_custom_kernels_partition_success) {
  const std::string name = "MyPortableOp";
  ge::CustomKernelItemHeader header{ge::kCustomKernelItemMagic, static_cast<uint32_t>(name.size()), 3U};
  std::vector<uint8_t> payload(sizeof(header) + name.size() + 3U, 0U);
  (void)memcpy_s(payload.data(), payload.size(), &header, sizeof(header));
  (void)memcpy_s(payload.data() + sizeof(header), payload.size() - sizeof(header), name.data(), name.size());
  payload[sizeof(header) + name.size() + 0U] = 0x1U;
  payload[sizeof(header) + name.size() + 1U] = 0x2U;
  payload[sizeof(header) + name.size() + 2U] = 0x3U;
  EXPECT_EQ(ge::SUCCESS, CustomOpFactory::LoadCustomOpsPartition(payload.data(), payload.size()));
}

TEST(UtestCustomOpFactory, load_custom_kernels_partition_deserialize_fail_propagate) {
  const auto reg_ret = CustomOpFactory::RegisterCustomOpCreator(
      "MyPortableOpFailedForPartition",
      []() -> std::unique_ptr<BaseCustomOp> { return std::make_unique<MyPortableOpFailed>(); });
  EXPECT_TRUE((reg_ret == ge::SUCCESS) || (reg_ret == ge::GRAPH_FAILED));

  const std::string name = "MyPortableOpFailedForPartition";
  ge::CustomKernelItemHeader header{ge::kCustomKernelItemMagic, static_cast<uint32_t>(name.size()), 1U};
  std::vector<uint8_t> payload(sizeof(header) + name.size() + 1U, 0U);
  (void)memcpy_s(payload.data(), payload.size(), &header, sizeof(header));
  (void)memcpy_s(payload.data() + sizeof(header), payload.size() - sizeof(header), name.data(), name.size());
  payload.back() = 0x9U;
  EXPECT_EQ(ge::GRAPH_FAILED, CustomOpFactory::LoadCustomOpsPartition(payload.data(), payload.size()));
}

TEST(UtestCustomOpFactory, IncCov_RemoveCustomOps_RemovesRegisteredOps) {
  const auto reg_ret = CustomOpFactory::RegisterCustomOpCreator(
      "IncCovRemoveOp", []() -> std::unique_ptr<BaseCustomOp> { return std::make_unique<RegistryTestOp>(); });
  EXPECT_TRUE((reg_ret == ge::SUCCESS) || (reg_ret == ge::GRAPH_FAILED));
  EXPECT_EQ(true, CustomOpFactory::IsExistOp("IncCovRemoveOp"));

  std::vector<AscendString> op_types{AscendString("IncCovRemoveOp")};
  CustomOpFactory::RemoveCustomOps(op_types);
  EXPECT_EQ(false, CustomOpFactory::IsExistOp("IncCovRemoveOp"));
  EXPECT_EQ(nullptr, CustomOpFactory::CreateOrGetCustomOp("IncCovRemoveOp", OpBackend::kDevice));
}

TEST(UtestCustomOpRegistry, IncCov_LoadCustomOpsPartition_EntrySizeExceedsData) {
  CustomOpRegistry registry;
  const std::string name = "TestEntrySizeOp";
  ge::CustomKernelItemHeader header{ge::kCustomKernelItemMagic, static_cast<uint32_t>(name.size()), 10U};
  std::vector<uint8_t> payload(sizeof(header) + name.size() + 2U, 0U);
  (void)memcpy_s(payload.data(), payload.size(), &header, sizeof(header));
  (void)memcpy_s(payload.data() + sizeof(header), payload.size() - sizeof(header), name.data(), name.size());
  EXPECT_EQ(ge::GRAPH_FAILED, registry.LoadCustomOpsPartition(payload.data(), payload.size()));
}

TEST(UtestCustomOpRegistry, IncCov_LoadCustomOpsPartition_RegisteredNonPortableOp) {
  CustomOpRegistry registry;
  EXPECT_EQ(ge::GRAPH_SUCCESS,
            registry.RegisterCreator("IncCovNonPortableOp", OpBackend::kDevice, []() -> std::unique_ptr<BaseCustomOp> {
              return std::make_unique<RegistryTestOp>();
            }));
  const auto payload = BuildCustomOpPartition("IncCovNonPortableOp", {0x1U});
  EXPECT_EQ(ge::GRAPH_FAILED, registry.LoadCustomOpsPartition(payload.data(), payload.size()));
}
