/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "runtime/custom_op/python_custom_op_bridge_loader.h"

#include <dlfcn.h>
#include <dirent.h>
#include <sys/stat.h>

#include <cstdlib>
#include <cstring>
#include <exception>
#include <map>
#include <mutex>
#include <new>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "common/ge_common/string_util.h"
#include "common/python_runtime/python_artifact_utils.h"
#include "common/python_runtime/python_bridge_loader_utils.h"
#include "framework/common/debug/ge_log.h"
#include "graph/ascend_string.h"
#include "graph/custom_op_factory.h"
#include "graph_metadef/graph/utils/file_utils.h"
#include "mmpa/mmpa_api.h"
#include "runtime/custom_op/python_custom_op_adapter.h"
#include "runtime/custom_op/python_custom_op_bridge_c_api.h"
#include "runtime/custom_op/python_custom_op_proto.h"

namespace ge {
namespace custom_op {
namespace {

constexpr const char *kCustomOpArtifactsRelativePath = "custom_op/python_custom_op_artifacts";
constexpr const char *kPythonFileSuffix = ".py";
constexpr const char *kPythonPackageInitFile = "__init__.py";

namespace artifact = ::ge::python_artifact;
namespace bridge_loader = ::ge::python_bridge_loader;

bool IsPythonFile(const std::string &path) {
  return (path.size() > strlen(kPythonFileSuffix)) &&
         (path.compare(path.size() - strlen(kPythonFileSuffix), strlen(kPythonFileSuffix), kPythonFileSuffix) == 0);
}

bool IsSkippedModuleEntry(const char *name) {
  if (name == nullptr) {
    return true;
  }
  return (name[0] == '_') || (strcmp(name, ".") == 0) || (strcmp(name, "..") == 0);
}

bool HasPackageInitFile(const std::string &dir) {
  struct stat path_stat{};
  return stat((dir + "/" + kPythonPackageInitFile).c_str(), &path_stat) == 0;
}

bool HasPythonCustomOpEntry(const std::string &path, const struct stat &path_stat) {
  if (S_ISREG(path_stat.st_mode)) {
    return IsPythonFile(path);
  }
  if (!S_ISDIR(path_stat.st_mode)) {
    return false;
  }

  DIR *dir = opendir(path.c_str());
  if (dir == nullptr) {
    GELOGI("Skip scanning python custom op directory[%s] because opendir failed.", path.c_str());
    return false;
  }
  struct dirent *entry = nullptr;
  while ((entry = readdir(dir)) != nullptr) {
    if (IsSkippedModuleEntry(entry->d_name)) {
      continue;
    }
    const std::string entry_path = path + "/" + entry->d_name;
    struct stat child_stat{};
    if (stat(entry_path.c_str(), &child_stat) != 0) {
      GELOGI("Skip scanning python custom op path[%s] because stat failed.", entry_path.c_str());
      continue;
    }
    if (S_ISREG(child_stat.st_mode) && IsPythonFile(entry_path)) {
      (void)closedir(dir);
      return true;
    }
    if (S_ISDIR(child_stat.st_mode) && HasPackageInitFile(entry_path)) {
      (void)closedir(dir);
      return true;
    }
  }
  (void)closedir(dir);
  return false;
}

Status FindPythonCustomOpEntryInEnv(const char *env_value, bool &found) {
  found = false;
  const std::string custom_opp_path = env_value;
  const std::vector<std::string> custom_opp_paths = StringUtils::Split(custom_opp_path, ':');
  if (custom_opp_paths.empty()) {
    return SUCCESS;
  }
  for (auto path : custom_opp_paths) {
    if (StringUtils::Trim(path).empty()) {
      continue;
    }
    struct stat path_stat{};
    if (stat(path.c_str(), &path_stat) != 0) {
      if (IsPythonFile(path)) {
        GELOGE(FAILED, "Python custom op path[%s] does not exist or is inaccessible.", path.c_str());
        return FAILED;
      }
      GELOGW("Skip inaccessible custom op path[%s].", path.c_str());
      continue;
    }
    if (HasPythonCustomOpEntry(path, path_stat)) {
      found = true;
      return SUCCESS;
    }
  }
  return SUCCESS;
}

std::string GetLoaderLibraryPath() {
  Dl_info dl_info{};
  if ((dladdr(reinterpret_cast<void *>(&LoadPythonCustomOps), &dl_info) == 0) || (dl_info.dli_fname == nullptr) ||
      (dl_info.dli_fname[0] == '\0')) {
    return "";
  }
  const auto real_path = RealPath(dl_info.dli_fname);
  return real_path.empty() ? std::string(dl_info.dli_fname) : real_path;
}

bool IsBridgeApiValid(const PythonCustomOpBridgeApi *api, const uint32_t expected_abi) {
  return (api != nullptr) && (api->abi_version == expected_abi) && (api->set_artifact_config != nullptr) &&
         (api->register_custom_ops != nullptr) && (api->reset_bridge_state != nullptr) &&
         (api->shutdown_bridge != nullptr);
}

struct PythonCustomOpRegistrationEntry {
  std::string op_type;
  std::string proto_descriptor_key;
  bool has_proto{false};
  PythonCustomOpInferMetaFn infer_meta{nullptr};
  bool has_impl{false};
  PythonCustomOpAdapterDescriptor impl_desc;
  PythonCustomOpAdapterCallbacks callbacks;
};

bridge_loader::BridgeLoadDependencies BuildBridgeLoadDependencies() {
  return bridge_loader::BridgeLoadDependencies{
      &RealPath,
      &dlopen,
      &dlclose,
      &dlsym,
      &artifact::ResolveLoadedPythonRuntimeKey,
      kPythonCustomOpBridgeGetApiSymbol,
      kPythonCustomOpBridgeAbiVersion,
      RTLD_NOW | RTLD_GLOBAL,
  };
}

class PythonCustomOpBridgeLoader {
 public:
  static PythonCustomOpBridgeLoader &GetInstance() {
    static PythonCustomOpBridgeLoader loader;
    return loader;
  }

  Status Load() {
    std::lock_guard<std::mutex> lock(mutex_);
    const auto ret = EnsureLoaded();
    if (ret != SUCCESS) {
      return ret;
    }
    return RegisterCustomOpsFromBridge();
  }

  void Unload() {
    std::lock_guard<std::mutex> lock(mutex_);
    GELOGI("Unload python custom ops with bridge library[%s].", loaded_path_.c_str());
    ClearPythonCustomOpRegistrations();
    ClearRegisteredState();
    if ((api_ != nullptr) && (api_->reset_bridge_state != nullptr)) {
      api_->reset_bridge_state();
    }
  }

  void ShutdownForProcess() {
    std::lock_guard<std::mutex> lock(mutex_);
    GELOGI("Shutdown python custom op bridge for process, current library[%s].", loaded_path_.c_str());
    if ((api_ != nullptr) && (api_->shutdown_bridge != nullptr)) {
      api_->shutdown_bridge();
    }
    api_ = nullptr;
    if (handle_ != nullptr) {
      if (dlclose(handle_) != 0) {
        GELOGW("Close python custom op bridge library failed: %s", dlerror());
      }
      handle_ = nullptr;
    }
    loaded_path_.clear();
    ClearRegisteredState();
  }

 private:
  void ClearPythonCustomOpRegistrations() {
    std::vector<AscendString> adapter_op_types;
    adapter_op_types.reserve(registered_op_type_to_adapter_.size());
    for (const auto &op_type : registered_op_type_to_adapter_) {
      adapter_op_types.emplace_back(op_type.c_str());
    }
    CustomOpFactory::RemoveCustomOps(adapter_op_types);
    ClearPythonCustomOpRuntimeRegistry();
    std::vector<std::string> proto_op_types;
    proto_op_types.reserve(python_custom_op_registrations_.size());
    for (const auto &item : python_custom_op_registrations_) {
      if (item.second.has_proto) {
        proto_op_types.emplace_back(item.first);
      }
    }
    UnregisterPythonCustomOpProtos(proto_op_types);
  }

  Status RegisterCustomOpsFromBridge() {
    static constexpr PythonCustomOpRegistrar kRegistrar = {
        &RegisterOpProtoFromBridge,
        &RegisterOpImplFromBridge,
    };
    GELOGI("Register python custom ops with bridge library[%s].", loaded_path_.c_str());
    const auto ret = api_->register_custom_ops(&kRegistrar);
    if (ret != SUCCESS) {
      GELOGE(ret, "[Register][PythonCustomOps] failed with bridge library[%s].", loaded_path_.c_str());
      return ret;
    }
    return CommitPythonCustomOpRegistrations();
  }

  static bool RegisterOpProtoFromBridge(const PythonCustomOpProtoDescriptorView *desc) {
    if (desc == nullptr) {
      return false;
    }
    return GetInstance().RegisterOpProto(*desc);
  }

  static bool RegisterOpImplFromBridge(const PythonCustomOpAdapterDescriptorView *desc,
                                       const PythonCustomOpAdapterCallbacks *callbacks) {
    if ((desc == nullptr) || (callbacks == nullptr)) {
      return false;
    }
    return GetInstance().RegisterOpImpl(*desc, *callbacks);
  }

  bool RegisterOpProto(const PythonCustomOpProtoDescriptorView &view) {
    PythonCustomOpProto proto;
    if (ParsePythonCustomOpProto(view, proto) != GRAPH_SUCCESS) {
      GELOGE(FAILED, "[Parse][PythonCustomOpProto] failed.");
      return false;
    }
    auto &registration = python_custom_op_registrations_[proto.op_type];
    if (registration.op_type.empty()) {
      registration.op_type = proto.op_type;
    }
    if (registration.has_proto) {
      if (registration.proto_descriptor_key == proto.descriptor_key) {
        return true;
      }
      GELOGE(FAILED,
             "Python custom op proto conflict, op type[%s], existing source[descriptor key:%s], "
             "current source[descriptor key:%s].",
             proto.op_type.c_str(), registration.proto_descriptor_key.c_str(), proto.descriptor_key.c_str());
      return false;
    }
    if (RegisterPythonCustomOpProto(proto) != GRAPH_SUCCESS) {
      GELOGE(FAILED, "[Register][PythonCustomOpProto] failed, descriptor key[%s], op type[%s].",
             proto.descriptor_key.c_str(), proto.op_type.c_str());
      return false;
    }
    registration.proto_descriptor_key = proto.descriptor_key;
    registration.has_proto = true;
    registration.infer_meta = proto.infer_meta;
    return true;
  }

  static bool CopyStringView(const PythonCustomOpStringView &view, const bool allow_empty, std::string &value) {
    if ((view.size != 0U) && (view.data == nullptr)) {
      return false;
    }
    value.assign(view.data == nullptr ? "" : view.data, view.size);
    return (allow_empty || (!value.empty())) && (value.find('\0') == std::string::npos);
  }

  bool ParseAdapterDescriptor(const PythonCustomOpAdapterDescriptorView &view, PythonCustomOpAdapterDescriptor &desc) {
    if ((!CopyStringView(view.op_type, false, desc.op_type)) ||
        (!CopyStringView(view.impl_descriptor_key, false, desc.impl_descriptor_key))) {
      return false;
    }
    desc.capabilities = view.capabilities;
    return true;
  }

  bool RegisterOpImpl(const PythonCustomOpAdapterDescriptorView &view,
                      const PythonCustomOpAdapterCallbacks &callbacks) {
    PythonCustomOpAdapterDescriptor desc;
    if (!ParseAdapterDescriptor(view, desc)) {
      GELOGE(FAILED, "[Parse][PythonCustomOpAdapter] failed.");
      return false;
    }
    auto &registration = python_custom_op_registrations_[desc.op_type];
    if (registration.op_type.empty()) {
      registration.op_type = desc.op_type;
    }
    if (registration.has_impl) {
      if (registration.impl_desc.impl_descriptor_key == desc.impl_descriptor_key) {
        return true;
      }
      GELOGE(FAILED,
             "Python custom op adapter conflict, op type[%s], existing source[impl key:%s], "
             "current source[impl key:%s].",
             desc.op_type.c_str(), registration.impl_desc.impl_descriptor_key.c_str(),
             desc.impl_descriptor_key.c_str());
      return false;
    }
    if (CustomOpFactory::IsExistOp(AscendString(desc.op_type.c_str()))) {
      GELOGE(FAILED,
             "[Check][PythonCustomOpAdapter]Op type[%s] conflicts, existing source[CustomOpFactory creator], "
             "current source[Python impl descriptor key:%s].",
             desc.op_type.c_str(), desc.impl_descriptor_key.c_str());
      return false;
    }
    if (!callbacks.IsValid(desc.capabilities)) {
      GELOGE(FAILED, "Invalid python custom op implementation, descriptor key[%s], op type[%s].",
             desc.impl_descriptor_key.c_str(), desc.op_type.c_str());
      return false;
    }
    registration.has_impl = true;
    registration.impl_desc = desc;
    registration.callbacks = callbacks;
    return true;
  }

  Status RegisterPythonCustomOpImpls() {
    for (const auto &item : python_custom_op_registrations_) {
      const auto &registration = item.second;
      if (!registration.has_impl ||
          (registered_op_type_to_adapter_.find(registration.op_type) != registered_op_type_to_adapter_.cend())) {
        continue;
      }
      if (!PythonCustomOpImplRuntimeRegistry::Register(registration.impl_desc, registration.callbacks)) {
        GELOGE(FAILED, "[Register][PythonCustomOpImplRuntimeRegistry] failed, descriptor key[%s], op type[%s].",
               registration.impl_desc.impl_descriptor_key.c_str(), registration.op_type.c_str());
        return FAILED;
      }
    }
    return SUCCESS;
  }

  bool BuildAdapterDescriptor(const PythonCustomOpRegistrationEntry &registration,
                              PythonCustomOpAdapterDescriptor &desc) {
    desc.op_type = registration.op_type;
    if (registration.has_impl) {
      desc = registration.impl_desc;
    }
    if (registration.has_proto) {
      desc.infer_meta = registration.infer_meta;
      AddCustomOpCapability(desc.capabilities, CustomOpCapability::kShapeInfer);
      AddCustomOpCapability(desc.capabilities, CustomOpCapability::kInferMeta);
    }
    const auto infer_capabilities = static_cast<CustomOpCapabilityMask>(CustomOpCapability::kShapeInfer) |
                                    static_cast<CustomOpCapabilityMask>(CustomOpCapability::kInferMeta);
    const auto impl_capabilities = desc.capabilities & ~infer_capabilities;
    if ((desc.capabilities == 0U) ||
        (HasCustomOpCapability(desc.capabilities, CustomOpCapability::kInferMeta) && (desc.infer_meta == nullptr)) ||
        ((impl_capabilities != 0U) && desc.impl_descriptor_key.empty()) ||
        ((impl_capabilities == 0U) && !desc.impl_descriptor_key.empty())) {
      GELOGE(FAILED, "Invalid python custom op adapter descriptor, op type[%s].", registration.op_type.c_str());
      return false;
    }
    return true;
  }

  Status RegisterPythonCustomOpCreator(const PythonCustomOpAdapterDescriptor &desc) {
    const auto ret = CustomOpFactory::RegisterCustomOpCreator(
        AscendString(desc.op_type.c_str()), [registered_desc = desc]() -> std::unique_ptr<BaseCustomOp> {
          auto *adapter = new (std::nothrow) PythonCustomOpAdapter(registered_desc);
          if ((adapter == nullptr) || (!adapter->IsValid())) {
            delete adapter;
            return std::unique_ptr<BaseCustomOp>();
          }
          return std::unique_ptr<BaseCustomOp>(adapter);
        });
    if (ret != GRAPH_SUCCESS) {
      GELOGE(ret, "Register python custom op creator failed, op type[%s].", desc.op_type.c_str());
      return FAILED;
    }
    registered_op_type_to_adapter_.insert(desc.op_type);
    GELOGI("Python custom op creator is registered, op type[%s].", desc.op_type.c_str());
    return SUCCESS;
  }

  Status RegisterPythonCustomOpCreators() {
    for (const auto &item : python_custom_op_registrations_) {
      const auto &registration = item.second;
      const auto &op_type = registration.op_type;
      if (registered_op_type_to_adapter_.find(op_type) != registered_op_type_to_adapter_.cend()) {
        continue;
      }
      if (CustomOpFactory::IsExistOp(AscendString(op_type.c_str()))) {
        GELOGE(FAILED, "Python custom op conflict, op type[%s], existing source[CustomOpFactory creator].",
               op_type.c_str());
        return FAILED;
      }

      PythonCustomOpAdapterDescriptor desc;
      if (!BuildAdapterDescriptor(registration, desc)) {
        return FAILED;
      }
      if (RegisterPythonCustomOpCreator(desc) != SUCCESS) {
        return FAILED;
      }
    }
    return SUCCESS;
  }

  Status CommitPythonCustomOpRegistrations() {
    if (RegisterPythonCustomOpImpls() != SUCCESS) {
      return FAILED;
    }
    return RegisterPythonCustomOpCreators();
  }

  void ClearRegisteredState() {
    python_custom_op_registrations_.clear();
    registered_op_type_to_adapter_.clear();
  }

  Status EnsureLoaded() {
    if (api_ != nullptr) {
      GELOGI("Reuse already loaded python custom op bridge library[%s].", loaded_path_.c_str());
      return SUCCESS;
    }
    const auto runtime_key = artifact::ResolveLoadedPythonRuntimeKey();
    if (!runtime_key.has_python_symbols) {
      GELOGE(FAILED, "[Check][PythonRuntime]Python symbols are not loaded before custom op preprocessing, runtime[%s].",
             runtime_key.ToString().c_str());
      return FAILED;
    }
    if (!runtime_key.is_initialized) {
      GELOGE(FAILED,
             "[Check][PythonRuntime]Python interpreter is not initialized before custom op preprocessing, "
             "runtime[%s].",
             runtime_key.ToString().c_str());
      return FAILED;
    }
    GELOGI("Python custom op runtime key before loading bridge: %s.", runtime_key.ToString().c_str());
    if (TryLoadPrebuiltBridge(runtime_key)) {
      return SUCCESS;
    }
    GELOGE(FAILED, "Load python custom op bridge library failed.");
    return FAILED;
  }

  bool TryLoadPrebuiltBridge(const artifact::PythonRuntimeKey &runtime_key) {
    const auto candidates = artifact::BuildPrebuiltBridgeLibraryCandidates(
        runtime_key, GetLoaderLibraryPath(), kCustomOpArtifactsRelativePath, kPythonCustomOpBridgeAbiVersion);
    return TryLoadBridgeCandidates(runtime_key, candidates);
  }

  bool TryLoadBridgeCandidates(const artifact::PythonRuntimeKey &runtime_key,
                               const std::vector<artifact::BridgeLibraryCandidate> &candidates) {
    for (const auto &candidate : candidates) {
      bridge_loader::LoadedBridgeCandidate<PythonCustomOpBridgeApi> loaded_bridge;
      const auto deps = BuildBridgeLoadDependencies();
      const auto status =
          bridge_loader::TryLoadBridgeCandidate<PythonCustomOpBridgeApi, PythonCustomOpBridgeArtifactConfig>(
              runtime_key, candidate, deps, &IsBridgeApiValid, loaded_bridge);
      if (status != bridge_loader::BridgeLoadStatus::kSuccess) {
        const auto error_suffix = bridge_loader::BuildBridgeLoadErrorSuffix(status, dlerror());
        GELOGW("Skip python custom op bridge candidate[%s], artifact_root[%s], native_module[%s], status[%s]%s.",
               candidate.bridge_path.c_str(), candidate.artifact_root.c_str(), candidate.native_module_path.c_str(),
               bridge_loader::BridgeLoadStatusToString(status), error_suffix.c_str());
        continue;
      }
      handle_ = loaded_bridge.handle;
      api_ = loaded_bridge.api;
      loaded_path_ = loaded_bridge.real_path;
      GELOGI("Load python custom op bridge from [%s] success.", loaded_path_.c_str());
      return true;
    }
    return false;
  }

  std::mutex mutex_;
  void *handle_{nullptr};
  const PythonCustomOpBridgeApi *api_{nullptr};
  std::string loaded_path_;
  std::map<std::string, PythonCustomOpRegistrationEntry> python_custom_op_registrations_;
  std::set<std::string> registered_op_type_to_adapter_;
};
}  // namespace

Status CheckNeedLoadPythonCustomOps(bool &need_load) {
  need_load = false;
  const char_t *custom_opp_path_env = nullptr;
  MM_SYS_GET_ENV(MM_ENV_ASCEND_CUSTOM_OPP_PATH, custom_opp_path_env);
  if ((custom_opp_path_env == nullptr) || (custom_opp_path_env[0] == '\0')) {
    GELOGI("Skip loading python custom ops because ASCEND_CUSTOM_OPP_PATH is empty.");
    return SUCCESS;
  }
  const auto ret = FindPythonCustomOpEntryInEnv(custom_opp_path_env, need_load);
  if (ret != SUCCESS) {
    return ret;
  }
  if (!need_load) {
    GELOGI(
        "Skip loading python custom ops because no loadable python custom op entry is found in "
        "ASCEND_CUSTOM_OPP_PATH.");
  }
  return SUCCESS;
}

Status LoadPythonCustomOps() {
  return PythonCustomOpBridgeLoader::GetInstance().Load();
}

void UnloadPythonCustomOps() {
  PythonCustomOpBridgeLoader::GetInstance().Unload();
}

void ShutdownPythonCustomOpsForProcess() {
  PythonCustomOpBridgeLoader::GetInstance().ShutdownForProcess();
}
}  // namespace custom_op
}  // namespace ge
