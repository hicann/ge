/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <cstring>

#include "runtime/custom_op/python_custom_op_bridge_c_api.h"
#include "runtime/custom_op/python_custom_op_bridge_types.h"

namespace ge {
namespace custom_op {
namespace {
constexpr const char *kScenarioEnvName = "GE_PYTHON_CUSTOM_OP_LOADER_UT_SCENARIO";
constexpr const char *kMultiProtoFailure = "multi_proto_failure";
constexpr const char *kAdapterFailure = "adapter_failure";
constexpr const char *kSuccess = "success";
constexpr const char *kCppProtoImpl = "cpp_proto_impl";

constexpr const char *kMultiProtoOp = "PythonLoaderMultiProtoRollbackUt";
constexpr const char *kAdapterOpA = "PythonLoaderAdapterRollbackAUt";
constexpr const char *kAdapterOpB = "PythonLoaderAdapterRollbackBUt";
constexpr const char *kSuccessOp = "PythonLoaderSuccessUt";
constexpr const char *kCppProtoOp = "PythonLoaderCppProtoOwnershipUt";

constexpr const char *kMultiProtoKey = "loader_ut:multi_proto";
constexpr const char *kAdapterProtoKeyA = "loader_ut:adapter_proto_a";
constexpr const char *kAdapterProtoKeyB = "loader_ut:adapter_proto_b";
constexpr const char *kAdapterImplKeyA = "loader_ut:adapter_impl_a";
constexpr const char *kAdapterImplKeyB = "loader_ut:adapter_impl_b";
constexpr const char *kSuccessProtoKey = "loader_ut:success_proto";
constexpr const char *kSuccessImplKey = "loader_ut:success_impl";
constexpr const char *kCppProtoImplKey = "loader_ut:cpp_proto_impl";

std::atomic<uint32_t> g_register_count{0U};
std::atomic<uint32_t> g_reset_count{0U};
int g_holder = 0;

PythonCustomOpStringView StringView(const char *value) {
  return PythonCustomOpStringView{value, (value == nullptr) ? 0U : std::strlen(value)};
}

graphStatus InferMeta(const PythonCustomOpStringView *, gert::InferShapeContext *,
                      PythonCustomOpInferMetaResultView *) {
  return GRAPH_SUCCESS;
}

PythonCustomOpProtoDescriptorView MakeProto(const char *descriptor_key, const char *op_type) {
  return PythonCustomOpProtoDescriptorView{
      StringView(descriptor_key), StringView(op_type), nullptr, 0U, nullptr, 0U, nullptr, 0U, &InferMeta};
}

PythonCustomOpAdapterDescriptorView MakeAdapter(const char *op_type, const char *impl_key) {
  return PythonCustomOpAdapterDescriptorView{StringView(op_type), StringView(impl_key),
                                             static_cast<CustomOpCapabilityMask>(CustomOpCapability::kEagerExecute)};
}

void *CreateImplHolder(const PythonCustomOpAdapterDescriptorView *desc) {
  return (desc == nullptr) ? nullptr : &g_holder;
}

void DestroyImplHolder(void *holder) {
  (void)holder;
}

graphStatus Execute(const void *holder, gert::EagerOpExecutionContext *ctx) {
  (void)ctx;
  return (holder == nullptr) ? GRAPH_FAILED : GRAPH_SUCCESS;
}

PythonCustomOpAdapterCallbacks MakeCallbacks(const bool reject) {
  PythonCustomOpAdapterCallbacks callbacks;
  callbacks.create_impl_holder = &CreateImplHolder;
  callbacks.destroy_impl_holder = &DestroyImplHolder;
  callbacks.execute = &Execute;
  if (reject) {
    callbacks.create_impl_holder = nullptr;
  }
  (void)reject;
  return callbacks;
}

Status SetArtifactConfig(const PythonCustomOpBridgeArtifactConfig *config) {
  if ((config == nullptr) || (config->artifact_root == nullptr) || (config->native_module_path == nullptr)) {
    return static_cast<Status>(GRAPH_FAILED);
  }
  return static_cast<Status>(GRAPH_SUCCESS);
}

Status RegisterMultiProtoFailure(const PythonCustomOpRegistrar &registrar) {
  const auto first = MakeProto(kMultiProtoKey, kMultiProtoOp);
  if (!registrar.register_op_proto(&first)) {
    return static_cast<Status>(GRAPH_FAILED);
  }
  const auto invalid = MakeProto("loader_ut:invalid_proto", "");
  return static_cast<Status>(registrar.register_op_proto(&invalid) ? GRAPH_SUCCESS : GRAPH_FAILED);
}

Status RegisterAdapterFailure(const PythonCustomOpRegistrar &registrar) {
  const auto proto_a = MakeProto(kAdapterProtoKeyA, kAdapterOpA);
  const auto proto_b = MakeProto(kAdapterProtoKeyB, kAdapterOpB);
  if ((!registrar.register_op_proto(&proto_a)) || (!registrar.register_op_proto(&proto_b))) {
    return static_cast<Status>(GRAPH_FAILED);
  }

  const auto callbacks = MakeCallbacks(false);
  const auto adapter_a = MakeAdapter(kAdapterOpA, kAdapterImplKeyA);
  if (!registrar.register_op_impl(&adapter_a, &callbacks)) {
    return static_cast<Status>(GRAPH_FAILED);
  }
  const auto rejecting_callbacks = MakeCallbacks(true);
  const auto adapter_b = MakeAdapter(kAdapterOpB, kAdapterImplKeyB);
  return static_cast<Status>(registrar.register_op_impl(&adapter_b, &rejecting_callbacks) ? GRAPH_SUCCESS
                                                                                          : GRAPH_FAILED);
}

Status RegisterSuccess(const PythonCustomOpRegistrar &registrar) {
  const auto proto = MakeProto(kSuccessProtoKey, kSuccessOp);
  if (!registrar.register_op_proto(&proto)) {
    return static_cast<Status>(GRAPH_FAILED);
  }
  const auto callbacks = MakeCallbacks(false);
  const auto adapter = MakeAdapter(kSuccessOp, kSuccessImplKey);
  return static_cast<Status>(registrar.register_op_impl(&adapter, &callbacks) ? GRAPH_SUCCESS : GRAPH_FAILED);
}

Status RegisterCppProtoImpl(const PythonCustomOpRegistrar &registrar) {
  const auto callbacks = MakeCallbacks(false);
  const auto adapter = MakeAdapter(kCppProtoOp, kCppProtoImplKey);
  return static_cast<Status>(registrar.register_op_impl(&adapter, &callbacks) ? GRAPH_SUCCESS : GRAPH_FAILED);
}

Status RegisterCustomOps(const PythonCustomOpRegistrar *registrar) {
  ++g_register_count;
  if ((registrar == nullptr) || (registrar->register_op_proto == nullptr) || (registrar->register_op_impl == nullptr)) {
    return static_cast<Status>(GRAPH_FAILED);
  }
  const char *scenario = std::getenv(kScenarioEnvName);
  if (scenario == nullptr) {
    return static_cast<Status>(GRAPH_FAILED);
  }
  if (std::strcmp(scenario, kMultiProtoFailure) == 0) {
    return RegisterMultiProtoFailure(*registrar);
  }
  if (std::strcmp(scenario, kAdapterFailure) == 0) {
    return RegisterAdapterFailure(*registrar);
  }
  if (std::strcmp(scenario, kSuccess) == 0) {
    return RegisterSuccess(*registrar);
  }
  if (std::strcmp(scenario, kCppProtoImpl) == 0) {
    return RegisterCppProtoImpl(*registrar);
  }
  return static_cast<Status>(GRAPH_FAILED);
}

void ResetBridgeState() {
  ++g_reset_count;
}

void ShutdownBridge() {}
}  // namespace
}  // namespace custom_op
}  // namespace ge

extern "C" __attribute__((visibility("default"))) uint32_t GePythonCustomOpLoaderUtGetRegisterCount() {
  return ge::custom_op::g_register_count.load();
}

extern "C" __attribute__((visibility("default"))) uint32_t GePythonCustomOpLoaderUtGetResetCount() {
  return ge::custom_op::g_reset_count.load();
}

extern "C" __attribute__((visibility("default"))) void GePythonCustomOpLoaderUtResetCounters() {
  ge::custom_op::g_register_count.store(0U);
  ge::custom_op::g_reset_count.store(0U);
}

extern "C" __attribute__((visibility("default"))) const ge::custom_op::PythonCustomOpBridgeApi *
GeGetPythonCustomOpBridgeApi() {
  static const ge::custom_op::PythonCustomOpBridgeApi kBridgeApi = {
      ge::custom_op::kPythonCustomOpBridgeAbiVersion,
      &ge::custom_op::SetArtifactConfig,
      &ge::custom_op::RegisterCustomOps,
      &ge::custom_op::ResetBridgeState,
      &ge::custom_op::ShutdownBridge,
  };
  return &kBridgeApi;
}
