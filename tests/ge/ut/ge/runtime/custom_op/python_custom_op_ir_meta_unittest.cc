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
#include <string>

#include "common/python_runtime/python_artifact_utils.h"
#include "common/python_runtime/python_bridge_loader_utils.h"
#include "exe_graph/runtime/op_compile_context.h"
#include "runtime/custom_op/python_custom_op_adapter.h"
#include "runtime/custom_op/python_custom_op_bridge_c_api.h"

namespace ge {
namespace custom_op {
namespace {
namespace artifact = ::ge::python_artifact;
namespace bridge_loader = ::ge::python_bridge_loader;

struct MockPythonCustomOpHolder {};

void *CreateMockPythonCustomOpHolder(const PythonCustomOpAdapterDescriptorView *desc) {
  return ((desc == nullptr) || (desc->impl_descriptor_key.data == nullptr)) ? nullptr
                                                                            : new (std::nothrow)
                                                                                  MockPythonCustomOpHolder();
}

void DestroyMockPythonCustomOpHolder(void *holder) {
  delete static_cast<MockPythonCustomOpHolder *>(holder);
}

graphStatus ExecuteMockPythonCustomOp(const void *holder, gert::EagerOpExecutionContext *ctx) {
  (void)ctx;
  EXPECT_NE(holder, nullptr);
  return (holder != nullptr) ? GRAPH_SUCCESS : GRAPH_FAILED;
}

graphStatus DeclareMockPythonCustomOp(const void *holder, gert::AnnotatedArgsContext *ctx) {
  return ((holder != nullptr) && (ctx != nullptr)) ? GRAPH_SUCCESS : GRAPH_FAILED;
}

graphStatus CompileMockPythonCustomOp(const void *holder, gert::OpCompileContext *ctx) {
  return ((holder != nullptr) && (ctx != nullptr)) ? GRAPH_SUCCESS : GRAPH_FAILED;
}

struct MockPythonCustomOpBridgeLoadState {
  PythonCustomOpBridgeApi api{};
  const PythonCustomOpBridgeApi *api_to_return{nullptr};
  uint32_t close_count{0U};
  uint32_t set_config_count{0U};
  uint32_t register_count{0U};
};

int g_mock_python_custom_op_bridge_handle = 0;
MockPythonCustomOpBridgeLoadState g_mock_python_custom_op_bridge_load_state;

Status SetMockPythonCustomOpArtifactConfig(const PythonCustomOpBridgeArtifactConfig *config) {
  if ((config == nullptr) || (config->artifact_root == nullptr) || (config->native_module_path == nullptr)) {
    return FAILED;
  }
  ++g_mock_python_custom_op_bridge_load_state.set_config_count;
  return SUCCESS;
}

Status RegisterMockPythonCustomOps(const PythonCustomOpRegistrar *registrar) {
  (void)registrar;
  ++g_mock_python_custom_op_bridge_load_state.register_count;
  return SUCCESS;
}

void ResetMockPythonCustomOpBridgeState() {}

void ShutdownMockPythonCustomOpBridge() {}

const PythonCustomOpBridgeApi *GetMockPythonCustomOpBridgeApi() {
  return g_mock_python_custom_op_bridge_load_state.api_to_return;
}

std::string ResolveMockPythonCustomOpBridgePath(const char *path) {
  return (path == nullptr) ? std::string() : std::string("/tmp/libge_python_custom_op_bridge.so");
}

void *OpenMockPythonCustomOpBridge(const char *path, int flags) {
  (void)flags;
  return (path == nullptr) ? nullptr : static_cast<void *>(&g_mock_python_custom_op_bridge_handle);
}

int CloseMockPythonCustomOpBridge(void *handle) {
  if (handle != nullptr) {
    ++g_mock_python_custom_op_bridge_load_state.close_count;
  }
  return 0;
}

void *LookupMockPythonCustomOpBridgeSymbol(void *handle, const char *symbol) {
  if ((handle == nullptr) || (symbol == nullptr)) {
    return nullptr;
  }
  return reinterpret_cast<void *>(&GetMockPythonCustomOpBridgeApi);
}

artifact::PythonRuntimeKey ResolveMockPythonRuntimeKey() {
  return {};
}

bool IsMockPythonCustomOpBridgeApiValid(const PythonCustomOpBridgeApi *api, const uint32_t expected_abi) {
  return (api != nullptr) && (api->abi_version == expected_abi) && (api->set_artifact_config != nullptr) &&
         (api->register_custom_ops != nullptr) && (api->reset_bridge_state != nullptr) &&
         (api->shutdown_bridge != nullptr);
}

bridge_loader::BridgeLoadDependencies MakeMockPythonCustomOpBridgeLoadDependencies() {
  return bridge_loader::BridgeLoadDependencies{
      &ResolveMockPythonCustomOpBridgePath, &OpenMockPythonCustomOpBridge,
      &CloseMockPythonCustomOpBridge,       &LookupMockPythonCustomOpBridgeSymbol,
      &ResolveMockPythonRuntimeKey,         kPythonCustomOpBridgeGetApiSymbol,
      kPythonCustomOpBridgeAbiVersion,      0,
  };
}

void ResetMockPythonCustomOpBridgeLoadState(const uint32_t abi_version) {
  g_mock_python_custom_op_bridge_load_state = MockPythonCustomOpBridgeLoadState{};
  g_mock_python_custom_op_bridge_load_state.api = PythonCustomOpBridgeApi{
      abi_version,
      &SetMockPythonCustomOpArtifactConfig,
      &RegisterMockPythonCustomOps,
      &ResetMockPythonCustomOpBridgeState,
      &ShutdownMockPythonCustomOpBridge,
  };
  g_mock_python_custom_op_bridge_load_state.api_to_return = &g_mock_python_custom_op_bridge_load_state.api;
}

bridge_loader::BridgeLoadStatus LoadMockPythonCustomOpBridge(
    bridge_loader::LoadedBridgeCandidate<PythonCustomOpBridgeApi> &loaded_bridge) {
  const artifact::BridgeLibraryCandidate candidate{
      "libge_python_custom_op_bridge.so",
      "/tmp/custom_op/python_custom_op_artifacts/cp311-linux-aarch64",
      "/tmp/custom_op/python_custom_op_artifacts/cp311-linux-aarch64/_ge_custom_op_native.so",
  };
  return bridge_loader::TryLoadBridgeCandidate<PythonCustomOpBridgeApi, PythonCustomOpBridgeArtifactConfig>(
      artifact::PythonRuntimeKey{}, candidate, MakeMockPythonCustomOpBridgeLoadDependencies(),
      &IsMockPythonCustomOpBridgeApiValid, loaded_bridge);
}
}  // namespace

TEST(PythonCustomOpAdapter, forwards_execute_without_ir_meta_pod) {
  PythonCustomOpAdapterDescriptor desc;
  desc.impl_descriptor_key = "python_adapter_without_ir_meta";
  desc.op_type = "PythonCustomOpAdapterUt";
  AddCustomOpCapability(desc.capabilities, CustomOpCapability::kEagerExecute);

  PythonCustomOpAdapterCallbacks callbacks;
  callbacks.create_impl_holder = CreateMockPythonCustomOpHolder;
  callbacks.destroy_impl_holder = DestroyMockPythonCustomOpHolder;
  callbacks.execute = ExecuteMockPythonCustomOp;

  ASSERT_TRUE(PythonCustomOpImplRuntimeRegistry::Register(desc, callbacks));
  {
    PythonCustomOpAdapter adapter(desc);
    ASSERT_TRUE(adapter.IsValid());
    EXPECT_EQ(adapter.Execute(nullptr), GRAPH_SUCCESS);
  }
  EXPECT_TRUE(PythonCustomOpImplRuntimeRegistry::Unregister(desc.impl_descriptor_key));
}

TEST(PythonCustomOpAdapter, keeps_legacy_execute_without_registered_ir) {
  PythonCustomOpAdapterDescriptor desc;
  desc.impl_descriptor_key = "python_adapter_legacy_without_ir";
  desc.op_type = "PythonCustomOpLegacyWithoutIrUt";
  AddCustomOpCapability(desc.capabilities, CustomOpCapability::kEagerExecute);

  PythonCustomOpAdapterCallbacks callbacks;
  callbacks.create_impl_holder = CreateMockPythonCustomOpHolder;
  callbacks.destroy_impl_holder = DestroyMockPythonCustomOpHolder;
  callbacks.execute = ExecuteMockPythonCustomOp;

  ASSERT_TRUE(PythonCustomOpImplRuntimeRegistry::Register(desc, callbacks));
  {
    PythonCustomOpAdapter adapter(desc);
    ASSERT_TRUE(adapter.IsValid());
    EXPECT_EQ(adapter.Execute(nullptr), GRAPH_SUCCESS);
  }
  EXPECT_TRUE(PythonCustomOpImplRuntimeRegistry::Unregister(desc.impl_descriptor_key));
}

TEST(PythonCustomOpAdapter, validates_annotated_args_callback_by_capability) {
  PythonCustomOpAdapterDescriptor desc;
  desc.impl_descriptor_key = "python_adapter_annotated_args_callback";
  desc.op_type = "PythonCustomOpIrMetaUt";
  AddCustomOpCapability(desc.capabilities, CustomOpCapability::kAnnotatedArgs);

  PythonCustomOpAdapterCallbacks callbacks;
  callbacks.create_impl_holder = CreateMockPythonCustomOpHolder;
  callbacks.destroy_impl_holder = DestroyMockPythonCustomOpHolder;
  EXPECT_FALSE(callbacks.IsValid(desc.capabilities));

  callbacks.declare_launch_args = DeclareMockPythonCustomOp;
  EXPECT_TRUE(callbacks.IsValid(desc.capabilities));
  EXPECT_TRUE(PythonCustomOpImplRuntimeRegistry::Register(desc, callbacks));
  EXPECT_TRUE(PythonCustomOpImplRuntimeRegistry::Unregister(desc.impl_descriptor_key));
}

TEST(PythonCustomOpAdapter, validates_and_forwards_compile_callback_by_capability) {
  PythonCustomOpAdapterDescriptor desc;
  desc.impl_descriptor_key = "python_adapter_compile_callback";
  desc.op_type = "PythonCustomOpCompileUt";
  AddCustomOpCapability(desc.capabilities, CustomOpCapability::kCompilable);

  PythonCustomOpAdapterCallbacks callbacks;
  callbacks.create_impl_holder = CreateMockPythonCustomOpHolder;
  callbacks.destroy_impl_holder = DestroyMockPythonCustomOpHolder;
  EXPECT_FALSE(callbacks.IsValid(desc.capabilities));

  callbacks.compile_impl = CompileMockPythonCustomOp;
  EXPECT_TRUE(callbacks.IsValid(desc.capabilities));
  ASSERT_TRUE(PythonCustomOpImplRuntimeRegistry::Register(desc, callbacks));
  {
    PythonCustomOpAdapter adapter(desc);
    ASSERT_TRUE(adapter.IsValid());
    gert::OpCompileContext ctx;
    EXPECT_EQ(adapter.Compile(&ctx), GRAPH_SUCCESS);
    EXPECT_EQ(adapter.Compile(nullptr), GRAPH_FAILED);
  }
  EXPECT_TRUE(PythonCustomOpImplRuntimeRegistry::Unregister(desc.impl_descriptor_key));
}

TEST(PythonCustomOpBridgeAbi, rejects_mismatched_abi_before_registration_and_accepts_current) {
  bridge_loader::LoadedBridgeCandidate<PythonCustomOpBridgeApi> loaded_bridge;
  ResetMockPythonCustomOpBridgeLoadState(kPythonCustomOpBridgeAbiVersion + 1U);

  EXPECT_EQ(LoadMockPythonCustomOpBridge(loaded_bridge), bridge_loader::BridgeLoadStatus::kInvalidApi);
  EXPECT_EQ(g_mock_python_custom_op_bridge_load_state.close_count, 1U);
  EXPECT_EQ(g_mock_python_custom_op_bridge_load_state.set_config_count, 0U);
  EXPECT_EQ(g_mock_python_custom_op_bridge_load_state.register_count, 0U);

  ResetMockPythonCustomOpBridgeLoadState(kPythonCustomOpBridgeAbiVersion);
  ASSERT_EQ(LoadMockPythonCustomOpBridge(loaded_bridge), bridge_loader::BridgeLoadStatus::kSuccess);
  EXPECT_EQ(g_mock_python_custom_op_bridge_load_state.close_count, 0U);
  EXPECT_EQ(g_mock_python_custom_op_bridge_load_state.set_config_count, 1U);
  ASSERT_NE(loaded_bridge.api, nullptr);
  EXPECT_EQ(loaded_bridge.api->register_custom_ops(nullptr), SUCCESS);
  EXPECT_EQ(g_mock_python_custom_op_bridge_load_state.register_count, 1U);
}

}  // namespace custom_op
}  // namespace ge
