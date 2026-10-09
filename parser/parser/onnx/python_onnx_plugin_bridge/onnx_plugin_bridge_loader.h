/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef PARSER_PARSER_ONNX_PYTHON_ONNX_PLUGIN_BRIDGE_ONNX_PLUGIN_BRIDGE_LOADER_H_
#define PARSER_PARSER_ONNX_PYTHON_ONNX_PLUGIN_BRIDGE_ONNX_PLUGIN_BRIDGE_LOADER_H_

#include "ge/ge_api_types.h"

namespace ge {
namespace onnx_plugin_bridge {
struct PythonOnnxPluginRegistrar;
}  // namespace onnx_plugin_bridge

__attribute__((visibility("default"))) Status
LoadOnnxPythonPluginBridge(const onnx_plugin_bridge::PythonOnnxPluginRegistrar *registrar);
__attribute__((visibility("default"))) void UnloadOnnxPythonPluginBridge();
// 检查 ASCEND_CUSTOM_OPP_PATH 取值中是否存在可加载的 Python ONNX 插件入口：
// 冒号分隔的各路径段为 .py 文件，或目录下存在非下划线开头的 .py 文件 / 含 __init__.py 的子目录。
__attribute__((visibility("default"))) bool HasPythonOnnxPluginEntryInEnv(const char *env_value);

}  // namespace ge

#endif  // PARSER_PARSER_ONNX_PYTHON_ONNX_PLUGIN_BRIDGE_ONNX_PLUGIN_BRIDGE_LOADER_H_
