/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef PARSER_PARSER_ONNX_PYTHON_ONNX_PLUGIN_BRIDGE_ONNX_PLUGIN_BRIDGE_C_API_H_
#define PARSER_PARSER_ONNX_PYTHON_ONNX_PLUGIN_BRIDGE_ONNX_PLUGIN_BRIDGE_C_API_H_

#include <cstddef>

#include "ge/ge_api_types.h"

namespace ge {
namespace onnx_plugin_bridge {

struct PythonOnnxPluginBridgeArtifactConfig {
  const char *artifact_root;
  const char *native_module_path;
};

// Descriptor view borrowed from the bridge for the duration of one register call only.
struct PythonOnnxPluginDescriptorView {
  const char *target;
  const char *origin;
  const char *const *callback_kinds;
  size_t callback_kinds_count;
};

// Entry points owned by the bridge; the bridge keeps them valid for the process lifetime.
struct PythonOnnxPluginParseCallbacks {
  Status (*parse_params)(const void *message, void *operator_dest);
  Status (*parse_params_by_operator)(const char *origin, const void *operator_src, void *operator_dest);
  Status (*parse_op_to_graph)(const char *origin, const void *operator_src, void *subgraph);
};

// Registrar implemented by the host side; the bridge reports every descriptor through it.
struct PythonOnnxPluginRegistrar {
  bool (*register_onnx_plugin)(const PythonOnnxPluginDescriptorView *desc,
                               const PythonOnnxPluginParseCallbacks *callbacks);
};

struct PythonOnnxPluginBridgeApi {
  uint32_t abi_version;
  Status (*set_artifact_config)(const PythonOnnxPluginBridgeArtifactConfig *config);
  Status (*register_plugins)(const PythonOnnxPluginRegistrar *registrar);
  void (*reset_bridge_state)();
};

constexpr uint32_t kPythonOnnxPluginBridgeAbiVersion = 1U;
constexpr const char *kPythonOnnxPluginBridgeGetApiSymbol = "GeGetPythonOnnxPluginBridgeApi";

}  // namespace onnx_plugin_bridge
}  // namespace ge

extern "C" __attribute__((visibility("default"))) const ge::onnx_plugin_bridge::PythonOnnxPluginBridgeApi *
GeGetPythonOnnxPluginBridgeApi();

#endif  // PARSER_PARSER_ONNX_PYTHON_ONNX_PLUGIN_BRIDGE_ONNX_PLUGIN_BRIDGE_C_API_H_
