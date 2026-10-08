/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "parser/parser/onnx/python_onnx_plugin_bridge/onnx_plugin_bridge_loader.h"

#include <dirent.h>
#include <dlfcn.h>
#include <sys/stat.h>

#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>

#include "base/err_msg.h"
#include "common/python_runtime/ge_python_runtime_manager.h"
#include "common/python_runtime/python_artifact_utils.h"
#include "common/python_runtime/python_bridge_loader_utils.h"
#include "common/ge_common/debug/ge_log.h"
#include "graph/def_types.h"
#include "graph_metadef/graph/utils/file_utils.h"
#include "parser/parser/onnx/python_onnx_plugin_bridge/onnx_plugin_bridge_c_api.h"

namespace ge {
namespace {

constexpr const char *kOnnxPluginArtifactsRelativePath = "onnx_plugin/python_onnx_plugin_artifacts";
constexpr const char *kPythonFileSuffix = ".py";
constexpr const char *kPythonPackageInitFile = "__init__.py";
constexpr char kEnvPathSeparator = ':';

bool IsPythonFile(const std::string &path) {
  const auto suffix_size = std::strlen(kPythonFileSuffix);
  return (path.size() > suffix_size) && (path.compare(path.size() - suffix_size, suffix_size, kPythonFileSuffix) == 0);
}

bool IsSkippedModuleEntry(const char *name) {
  return (name[0] == '_') || (strcmp(name, ".") == 0) || (strcmp(name, "..") == 0);
}

bool HasPackageInitFile(const std::string &dir) {
  struct stat path_stat{};
  return stat((dir + "/" + kPythonPackageInitFile).c_str(), &path_stat) == 0;
}

bool DirHasPythonPluginEntry(const std::string &dir) {
  DIR *dir_handle = opendir(dir.c_str());
  if (dir_handle == nullptr) {
    GELOGW("Skip scanning ONNX python plugin directory[%s] because opendir failed.", dir.c_str());
    return false;
  }
  struct dirent *entry = nullptr;
  while ((entry = readdir(dir_handle)) != nullptr) {
    if (IsSkippedModuleEntry(entry->d_name)) {
      continue;
    }
    const std::string entry_path = dir + "/" + entry->d_name;
    struct stat entry_stat{};
    if (stat(entry_path.c_str(), &entry_stat) != 0) {
      GELOGW("Skip scanning ONNX python plugin path[%s] because stat failed.", entry_path.c_str());
      continue;
    }
    if (S_ISREG(entry_stat.st_mode) && IsPythonFile(entry_path)) {
      (void)closedir(dir_handle);
      return true;
    }
    if (S_ISDIR(entry_stat.st_mode) && HasPackageInitFile(entry_path)) {
      (void)closedir(dir_handle);
      return true;
    }
  }
  (void)closedir(dir_handle);
  return false;
}

bool PathHasPythonPluginEntry(const std::string &path) {
  struct stat path_stat{};
  if (stat(path.c_str(), &path_stat) != 0) {
    GELOGW("Skip scanning ONNX python plugin path[%s] because it does not exist or is inaccessible.", path.c_str());
    return false;
  }
  if (S_ISREG(path_stat.st_mode)) {
    return IsPythonFile(path);
  }
  return S_ISDIR(path_stat.st_mode) && DirHasPythonPluginEntry(path);
}

std::string TrimBlank(const std::string &value) {
  const auto first = value.find_first_not_of(" \t");
  if (first == std::string::npos) {
    return "";
  }
  const auto last = value.find_last_not_of(" \t");
  return value.substr(first, last - first + 1U);
}

namespace artifact = ::ge::python_artifact;
namespace bridge_loader = ::ge::python_bridge_loader;
namespace onnx_bridge = ::ge::onnx_plugin_bridge;

std::string GetLoaderLibraryPath() {
  Dl_info dl_info{};
  // pass this function's address via PtrToPtr; decltype names the function type so the
  // template parameter list stays in sync with the real signature.
  if ((dladdr(PtrToPtr<decltype(LoadOnnxPythonPluginBridge), const void>(&LoadOnnxPythonPluginBridge), &dl_info) ==
       0) ||
      (dl_info.dli_fname == nullptr) || (dl_info.dli_fname[0] == '\0')) {
    return "";
  }
  const auto real_path = RealPath(dl_info.dli_fname);
  return real_path.empty() ? std::string(dl_info.dli_fname) : real_path;
}

bool IsBridgeApiValid(const onnx_bridge::PythonOnnxPluginBridgeApi *api, const uint32_t expected_abi) {
  return (api != nullptr) && (api->abi_version == expected_abi) && (api->set_artifact_config != nullptr) &&
         (api->register_plugins != nullptr) && (api->reset_bridge_state != nullptr);
}

bridge_loader::BridgeLoadDependencies BuildBridgeLoadDependencies() {
  return bridge_loader::BridgeLoadDependencies{
      &RealPath,
      &dlopen,
      &dlclose,
      &dlsym,
      &artifact::ResolveLoadedPythonRuntimeKey,
      onnx_bridge::kPythonOnnxPluginBridgeGetApiSymbol,
      onnx_bridge::kPythonOnnxPluginBridgeAbiVersion,
      RTLD_NOW | RTLD_GLOBAL,
  };
}

class OnnxPluginBridgeLoader {
 public:
  static OnnxPluginBridgeLoader &Instance() {
    static OnnxPluginBridgeLoader loader;
    return loader;
  }

  Status Load(const onnx_bridge::PythonOnnxPluginRegistrar *registrar) {
    if (!NeedLoad()) {
      return SUCCESS;
    }
    if (registrar == nullptr) {
      GELOGE(PARAM_INVALID, "Register Python ONNX plugins without a registrar.");
      return PARAM_INVALID;
    }
    if (GePythonRuntimeManager::Instance().EnsureReady() != SUCCESS) {
      REPORT_INNER_ERR_MSG(
          "E19999",
          "Prepare Python runtime for ONNX plugin bridge failed: no loadable python3/libpython is found in "
          "PATH. The Python ONNX plugin bridge requires a usable python runtime, check the python3 and "
          "libpython installation, or remove python plugin entries from ASCEND_CUSTOM_OPP_PATH if Python "
          "ONNX plugins are not used.");
      GELOGE(FAILED, "Prepare Python runtime for ONNX plugin bridge failed.");
      return FAILED;
    }

    std::lock_guard<std::mutex> lock(mutex_);
    if (EnsureLoaded() != SUCCESS) {
      return FAILED;
    }
    const auto ret = api_->register_plugins(registrar);
    if (ret == SUCCESS) {
      bridge_active_ = true;
    }
    return ret;
  }

  void Unload() {
    std::lock_guard<std::mutex> lock(mutex_);
    if ((api_ == nullptr) || !bridge_active_) {
      return;
    }
    api_->reset_bridge_state();
    bridge_active_ = false;
  }

 private:
  bool NeedLoad() const {
    const char *plugin_path = std::getenv("ASCEND_CUSTOM_OPP_PATH");
    if ((plugin_path == nullptr) || (plugin_path[0] == '\0')) {
      return false;
    }
    if (!HasPythonOnnxPluginEntryInEnv(plugin_path)) {
      GELOGI(
          "Skip loading ONNX Python plugin bridge because no loadable python plugin entry is found in "
          "ASCEND_CUSTOM_OPP_PATH.");
      return false;
    }
    return true;
  }

  Status EnsureLoaded() {
    if (api_ != nullptr) {
      return SUCCESS;
    }

    const auto runtime_key = artifact::ResolveLoadedPythonRuntimeKey();
    const auto loader_library_path = GetLoaderLibraryPath();
    const auto dependencies = BuildBridgeLoadDependencies();
    const auto candidates = artifact::BuildPrebuiltBridgeLibraryCandidates(
        runtime_key, loader_library_path, kOnnxPluginArtifactsRelativePath,
        onnx_bridge::kPythonOnnxPluginBridgeAbiVersion);
    for (const auto &candidate : candidates) {
      bridge_loader::LoadedBridgeCandidate<onnx_bridge::PythonOnnxPluginBridgeApi> loaded_bridge;
      const auto status = bridge_loader::TryLoadBridgeCandidate<onnx_bridge::PythonOnnxPluginBridgeApi,
                                                                onnx_bridge::PythonOnnxPluginBridgeArtifactConfig>(
          runtime_key, candidate, dependencies, &IsBridgeApiValid, loaded_bridge);
      if (status != bridge_loader::BridgeLoadStatus::kSuccess) {
        GELOGW("Skip ONNX Python plugin bridge candidate[%s], status[%s].", candidate.bridge_path.c_str(),
               bridge_loader::BridgeLoadStatusToString(status));
        continue;
      }
      api_ = loaded_bridge.api;
      GELOGI("Load ONNX Python plugin bridge from [%s] success.", loaded_bridge.real_path.c_str());
      return SUCCESS;
    }
    ReportIncompatibleArtifacts(runtime_key, loader_library_path);
    return FAILED;
  }

  static std::string CollectAvailableArtifactSummary(const std::string &loader_library_path) {
    std::string summary;
    for (const auto &manifest_path :
         artifact::BuildArtifactManifestCandidates(loader_library_path, kOnnxPluginArtifactsRelativePath)) {
      artifact::PythonArtifactSet artifact_set;
      if (!artifact::LoadArtifactManifest(manifest_path, artifact_set)) {
        continue;
      }
      if (!summary.empty()) {
        summary += ", ";
      }
      summary += artifact_set.python_tag + "-" + artifact_set.platform + "(bridge_abi " +
                 std::to_string(artifact_set.bridge_abi) + ") at " + artifact_set.root;
    }
    return summary;
  }

  static void ReportIncompatibleArtifacts(const artifact::PythonRuntimeKey &runtime_key,
                                          const std::string &loader_library_path) {
    const auto available_artifacts = CollectAvailableArtifactSummary(loader_library_path);
    const char *python_path = std::getenv(artifact::kPythonPathEnvName);
    REPORT_INNER_ERR_MSG(
        "E19999",
        "No compatible ONNX Python plugin bridge artifact found for runtime[%s], available artifacts[%s], "
        "loader[%s], PYTHONPATH[%s]. The python interpreter resolved by the process must match a prebuilt "
        "artifact under <ge package>/onnx_plugin/python_onnx_plugin_artifacts; align the python3 version with "
        "the artifact python tag or install the matching ge python package. If Python ONNX plugins are not "
        "used, remove python plugin entries from ASCEND_CUSTOM_OPP_PATH to skip this bridge.",
        runtime_key.ToString().c_str(), available_artifacts.empty() ? "none" : available_artifacts.c_str(),
        loader_library_path.c_str(), python_path == nullptr ? "" : python_path);
    GELOGE(FAILED,
           "No compatible ONNX Python plugin bridge artifact found for runtime[%s], available artifacts[%s], "
           "loader[%s], PYTHONPATH[%s].",
           runtime_key.ToString().c_str(), available_artifacts.empty() ? "none" : available_artifacts.c_str(),
           loader_library_path.c_str(), python_path == nullptr ? "" : python_path);
  }

  std::mutex mutex_;
  const onnx_bridge::PythonOnnxPluginBridgeApi *api_{nullptr};
  bool bridge_active_{false};
};

}  // namespace

Status LoadOnnxPythonPluginBridge(const onnx_plugin_bridge::PythonOnnxPluginRegistrar *registrar) {
  return OnnxPluginBridgeLoader::Instance().Load(registrar);
}

void UnloadOnnxPythonPluginBridge() {
  OnnxPluginBridgeLoader::Instance().Unload();
}

bool HasPythonOnnxPluginEntryInEnv(const char *env_value) {
  if ((env_value == nullptr) || (env_value[0] == '\0')) {
    return false;
  }
  const std::string env_paths(env_value);
  size_t start = 0U;
  while (start <= env_paths.size()) {
    const auto end = env_paths.find(kEnvPathSeparator, start);
    const auto segment = TrimBlank(env_paths.substr(start, end - start));
    if (!segment.empty() && PathHasPythonPluginEntry(segment)) {
      return true;
    }
    if (end == std::string::npos) {
      break;
    }
    start = end + 1U;
  }
  return false;
}

}  // namespace ge
