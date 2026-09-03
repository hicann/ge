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

#include <dlfcn.h>

#include <cstdlib>
#include <mutex>
#include <string>

#include "common/python_runtime/ge_python_runtime_manager.h"
#include "common/python_runtime/python_artifact_utils.h"
#include "common/python_runtime/python_bridge_loader_utils.h"
#include "framework/common/debug/ge_log.h"
#include "graph/def_types.h"
#include "graph_metadef/graph/utils/file_utils.h"
#include "parser/parser/onnx/python_onnx_plugin_bridge/onnx_plugin_bridge_c_api.h"

namespace ge {
namespace {

constexpr const char *kOnnxPluginArtifactsRelativePath = "onnx_plugin/python_onnx_plugin_artifacts";

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

  Status Load() {
    if (!NeedLoad()) {
      return SUCCESS;
    }
    if (GePythonRuntimeManager::Instance().EnsureReady() != SUCCESS) {
      GELOGE(FAILED, "Prepare Python runtime for ONNX plugin bridge failed.");
      return FAILED;
    }

    std::lock_guard<std::mutex> lock(mutex_);
    if (EnsureLoaded() != SUCCESS) {
      return FAILED;
    }
    const auto ret = api_->register_plugins();
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
    return (plugin_path != nullptr) && (plugin_path[0] != '\0');
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
    const auto manifests =
        artifact::BuildArtifactManifestCandidates(loader_library_path, kOnnxPluginArtifactsRelativePath);
    const char *python_path = std::getenv(artifact::kPythonPathEnvName);
    GELOGE(FAILED,
           "No compatible ONNX Python plugin bridge artifact found for runtime[%s], loader[%s], "
           "PYTHONPATH[%s], manifests[%zu].",
           runtime_key.ToString().c_str(), loader_library_path.c_str(), python_path == nullptr ? "" : python_path,
           manifests.size());
    return FAILED;
  }

  std::mutex mutex_;
  const onnx_bridge::PythonOnnxPluginBridgeApi *api_{nullptr};
  bool bridge_active_{false};
};

}  // namespace

Status LoadOnnxPythonPluginBridge() {
  return OnnxPluginBridgeLoader::Instance().Load();
}

void UnloadOnnxPythonPluginBridge() {
  OnnxPluginBridgeLoader::Instance().Unload();
}

}  // namespace ge
