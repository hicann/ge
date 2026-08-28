/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "onnx_plugin_bridge.h"

#include "onnx_plugin_bridge_c_api.h"

#include "Python.h"
#include "pybind11/embed.h"
#include "pybind11/stl.h"

#include "common/python_runtime/ge_python_runtime_manager.h"
#include "framework/common/debug/ge_log.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/graph.h"
#include "graph/operator.h"
#include "parser/common/op_registration_tbe.h"
#include "parser/common/op_parser_factory.h"
#include "proto/onnx/ge_onnx.pb.h"
#include "register/op_registry.h"
#include "register/register_fmk_types.h"

#include <cstdlib>
#include <stdexcept>
#include <mutex>
#include <string>
#include <vector>

namespace ge {
namespace {
namespace py = pybind11;

constexpr const char *kBridgeModuleName = "ge.onnx_plugin._bridge";
constexpr const char *kNativeModuleName = "ge.onnx_plugin._ge_onnx_plugin_native";
constexpr const char *kPluginPathEnv = "ASCEND_CUSTOM_OPP_PATH";
constexpr const char *kParseNodeCallbackKind = "parse_node";
constexpr const char *kParseOperatorCallbackKind = "parse_operator";
constexpr const char *kDecomposeCallbackKind = "decompose";

class OnnxPluginBridge {
 public:
  static OnnxPluginBridge &Instance() {
    static OnnxPluginBridge bridge;
    return bridge;
  }

  Status SetArtifactConfig(const onnx_plugin_bridge::PythonOnnxPluginBridgeArtifactConfig *config) {
    std::lock_guard<std::mutex> lock(mutex_);
    if ((config == nullptr) || (config->native_module_path == nullptr) || (config->native_module_path[0] == '\0')) {
      // LCOV_EXCL_START
      GELOGE(PARAM_INVALID, "Invalid Python ONNX plugin bridge artifact config.");
      return PARAM_INVALID;
      // LCOV_EXCL_STOP
    }
    native_module_path_ = config->native_module_path;
    return SUCCESS;
  }

  Status Initialize() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (initialized_) {
      return SUCCESS;
    }
    py::gil_scoped_acquire gil;
    try {
      SyncPluginPathUnlocked();
      LoadNativeModuleUnlocked();
      bridge_module_ = py::module_::import(kBridgeModuleName);
      invalid_return_exception_ = bridge_module_.attr("_InvalidParseNodeReturn");
      invalid_decompose_return_exception_ = bridge_module_.attr("_InvalidDecomposeReturn");
      const py::object descriptors = bridge_module_.attr("load_and_get_onnx_plugin_descriptors")();
      for (const py::handle item : descriptors) {
        const py::dict descriptor = py::reinterpret_borrow<py::dict>(item);
        const auto target = py::cast<std::string>(descriptor["target"]);
        const auto origins = py::cast<std::vector<std::string>>(descriptor["origin_types"]);
        std::vector<std::string> callback_kinds;
        if (descriptor.contains("callback_kinds")) {
          callback_kinds = py::cast<std::vector<std::string>>(descriptor["callback_kinds"]);
        } else if (descriptor.contains("callback_kind")) {
          callback_kinds.emplace_back(py::cast<std::string>(descriptor["callback_kind"]));
        } else {
          callback_kinds.emplace_back(kParseNodeCallbackKind);
        }
        for (const auto &origin : origins) {
          if (!RegisterDescriptor(target, origin, callback_kinds)) {
            // LCOV_EXCL_START
            GELOGE(FAILED, "Register Python ONNX plugin failed, target[%s], origin[%s].", target.c_str(),
                   origin.c_str());
            ResetBridgeStateUnlocked();
            return FAILED;
            // LCOV_EXCL_STOP
          }
        }
      }
      // LCOV_EXCL_START
    } catch (const py::error_already_set &error) {
      GELOGE(FAILED, "Load Python ONNX plugins failed: %s", error.what());
      ResetBridgeStateUnlocked();
      return FAILED;
    } catch (const std::exception &error) {
      GELOGE(FAILED, "Register Python ONNX plugins failed: %s", error.what());
      ResetBridgeStateUnlocked();
      return FAILED;
    }
    // LCOV_EXCL_STOP
    initialized_ = true;
    return SUCCESS;
  }

  void ResetBridgeState() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (Py_IsInitialized() == 0) {
      // LCOV_EXCL_START
      (void)bridge_module_.release();
      (void)invalid_return_exception_.release();
      (void)invalid_decompose_return_exception_.release();
      initialized_ = false;
      return;
      // LCOV_EXCL_STOP
    }
    py::gil_scoped_acquire gil;
    ResetBridgeStateUnlocked();
  }

  Status ParseParams(const google::protobuf::Message *message, Operator &operator_dest) {
    if ((message == nullptr) || (message->GetTypeName() != ge::onnx::NodeProto::descriptor()->full_name())) {
      GELOGE(PARAM_INVALID, "Python ONNX plugin received an invalid NodeProto message.");
      return PARAM_INVALID;
    }

    const auto *node = static_cast<const ge::onnx::NodeProto *>(message);
    std::lock_guard<std::mutex> lock(mutex_);
    if (!initialized_) {
      GELOGE(FAILED, "Python ONNX plugin bridge is not initialized.");
      return FAILED;
    }
    py::gil_scoped_acquire gil;
    try {
      const py::object python_node = py::cast(node, py::return_value_policy::reference);
      const auto handle = reinterpret_cast<uintptr_t>(&operator_dest);
      (void)bridge_module_.attr("call_parse_node")(node->op_type(), python_node, handle);
      operator_dest.SetAttr(ATTR_NAME_FRAMEWORK_ORIGINAL_TYPE, node->op_type());
      return SUCCESS;
    } catch (const py::error_already_set &error) {
      if (error.matches(invalid_return_exception_.ptr())) {
        GELOGE(PARAM_INVALID, "Python ONNX plugin parse_node returned an invalid value.");
        return PARAM_INVALID;
      }
      GELOGE(FAILED, "Python ONNX plugin parse_node failed: %s", error.what());
      return FAILED;
      // LCOV_EXCL_START
    } catch (const std::exception &error) {
      GELOGE(FAILED, "Python ONNX plugin bridge failed: %s", error.what());
      return FAILED;
    }
    // LCOV_EXCL_STOP
  }

  Status ParseParamsByOperator(const std::string &origin, const Operator &operator_src, Operator &operator_dest) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!initialized_) {
      GELOGE(FAILED, "Python ONNX plugin bridge is not initialized.");
      return FAILED;
    }
    py::gil_scoped_acquire gil;
    try {
      const auto source_handle = reinterpret_cast<uintptr_t>(&operator_src);
      const auto target_handle = reinterpret_cast<uintptr_t>(&operator_dest);
      (void)bridge_module_.attr("call_parse_operator")(origin, source_handle, target_handle);
      operator_dest.SetAttr(ATTR_NAME_FRAMEWORK_ORIGINAL_TYPE, origin);
      return SUCCESS;
    } catch (const py::error_already_set &error) {
      if (error.matches(invalid_return_exception_.ptr())) {
        GELOGE(PARAM_INVALID, "Python ONNX plugin parse_operator returned an invalid value.");
        return PARAM_INVALID;
      }
      GELOGE(FAILED, "Python ONNX plugin parse_operator failed: %s", error.what());
      return FAILED;
      // LCOV_EXCL_START
    } catch (const std::exception &error) {
      GELOGE(FAILED, "Python ONNX plugin bridge failed: %s", error.what());
      return FAILED;
    }
    // LCOV_EXCL_STOP
  }

  Status ParseOpToGraph(const std::string &origin, const Operator &operator_src, Graph &subgraph) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!initialized_) {
      GELOGE(FAILED, "Python ONNX plugin bridge is not initialized.");
      return FAILED;
    }
    py::gil_scoped_acquire gil;
    try {
      const auto source_handle = reinterpret_cast<uintptr_t>(&operator_src);
      const py::object replacement = bridge_module_.attr("call_decompose")(origin, source_handle);
      const auto graph_handle = py::cast<uintptr_t>(replacement.attr("_handle").attr("value"));
      const auto *replacement_graph = reinterpret_cast<const Graph *>(graph_handle);
      return subgraph.CopyFrom(*replacement_graph) == GRAPH_SUCCESS ? SUCCESS : FAILED;
    } catch (const py::error_already_set &error) {
      if (error.matches(invalid_decompose_return_exception_.ptr())) {
        GELOGE(PARAM_INVALID, "Python ONNX plugin decompose returned an invalid graph.");
        return PARAM_INVALID;
      }
      GELOGE(FAILED, "Python ONNX plugin decompose failed: %s", error.what());
      return FAILED;
      // LCOV_EXCL_START
    } catch (const std::exception &error) {
      GELOGE(FAILED, "Python ONNX plugin decompose bridge failed: %s", error.what());
      return FAILED;
    }
    // LCOV_EXCL_STOP
  }

 private:
  void SyncPluginPathUnlocked() {
    const char *plugin_path = std::getenv(kPluginPathEnv);
    const std::string plugin_path_value = (plugin_path == nullptr) ? std::string() : plugin_path;
    const py::object environ = py::module_::import("os").attr("environ");
    (void)environ.attr("pop")(kPluginPathEnv, py::none());
    if (!plugin_path_value.empty()) {
      environ[kPluginPathEnv] = py::str(plugin_path_value);
    }
  }

  void LoadNativeModuleUnlocked() {
    const py::dict modules = py::module_::import("sys").attr("modules");
    if (!modules.attr("get")(py::str(kNativeModuleName), py::none()).is_none()) {
      return;
    }
    if (native_module_path_.empty()) {
      (void)py::module_::import(kNativeModuleName);
      return;
    }

    const py::module_ importlib_util = py::module_::import("importlib.util");
    const py::object spec = importlib_util.attr("spec_from_file_location")(kNativeModuleName, native_module_path_);
    if (spec.is_none()) {
      throw std::runtime_error("Create Python ONNX native module spec failed");
    }
    const py::object module = importlib_util.attr("module_from_spec")(spec);
    modules[kNativeModuleName] = module;
    try {
      spec.attr("loader").attr("exec_module")(module);
    } catch (...) {
      (void)modules.attr("pop")(kNativeModuleName, py::none());
      throw;
    }
  }

  void ResetBridgeStateUnlocked() {
    bridge_module_ = py::object();
    invalid_return_exception_ = py::object();
    invalid_decompose_return_exception_ = py::object();
    initialized_ = false;
  }

  bool RegisterCallback(const std::string &origin, const std::string &callback_kind, OpRegistrationData &registration,
                        bool &has_parse_params_callback, bool &has_graph_callback) {
    if (callback_kind == kParseNodeCallbackKind) {
      has_parse_params_callback = true;
      const domi::ParseParamFunc parse_params = [](const google::protobuf::Message *message,
                                                   Operator &operator_dest) -> Status {
        return OnnxPluginBridge::Instance().ParseParams(message, operator_dest);
      };
      registration.ParseParamsFn(parse_params);
    } else if (callback_kind == kParseOperatorCallbackKind) {
      has_parse_params_callback = true;
      const domi::ParseParamByOpFunc parse_params = [origin](const Operator &operator_src,
                                                             Operator &operator_dest) -> Status {
        return OnnxPluginBridge::Instance().ParseParamsByOperator(origin, operator_src, operator_dest);
      };
      registration.ParseParamsByOperatorFn(parse_params);
    } else if (callback_kind == kDecomposeCallbackKind) {
      has_graph_callback = true;
      const domi::ParseOpToGraphFunc parse_op_to_graph = [origin](const Operator &operator_src,
                                                                  Graph &subgraph) -> Status {
        return OnnxPluginBridge::Instance().ParseOpToGraph(origin, operator_src, subgraph);
      };
      registration.ParseOpToGraphFn(parse_op_to_graph);
    } else {
      // LCOV_EXCL_START
      GELOGE(PARAM_INVALID, "Unknown Python ONNX plugin callback kind[%s].", callback_kind.c_str());
      return false;
      // LCOV_EXCL_STOP
    }
    return true;
  }

  bool RegisterDescriptor(const std::string &target, const std::string &origin,
                          const std::vector<std::string> &callback_kinds) {
    std::string registered_target;
    if (domi::OpRegistry::Instance()->GetOmTypeByOriOpType(origin, registered_target)) {
      if (registered_target != target) {
        GELOGW("Skip Python ONNX plugin for origin[%s], existing registration maps it to target[%s].", origin.c_str(),
               registered_target.c_str());
      } else {
        GELOGI("Skip duplicate Python ONNX plugin registration for target[%s], origin[%s].", target.c_str(),
               origin.c_str());
      }
      return true;
    }
    OpRegistrationData registration(target.c_str());
    registration.FrameworkType(domi::ONNX).OriginOpType(origin.c_str());
    bool has_parse_params_callback = false;
    bool has_graph_callback = false;
    for (const auto &callback_kind : callback_kinds) {
      if (!RegisterCallback(origin, callback_kind, registration, has_parse_params_callback, has_graph_callback)) {
        return false;
      }
    }
    if (!has_parse_params_callback && has_graph_callback) {
      registration.ParseParamsFn([origin](const google::protobuf::Message *, Operator &operator_dest) -> Status {
        operator_dest.SetAttr(ATTR_NAME_FRAMEWORK_ORIGINAL_TYPE, origin);
        return SUCCESS;
      });
    }
    const auto parser_factory = OpParserFactory::Instance(domi::ONNX);
    if (parser_factory == nullptr) {
      GELOGE(FAILED, "Get ONNX parser factory failed, target[%s], origin[%s].", target.c_str(), origin.c_str());
      return false;
    }
    if (!parser_factory->OpParserIsRegistered(target) && !OpRegistrationTbe::Instance()->Finalize(registration)) {
      GELOGE(FAILED, "Finalize Python ONNX plugin registration failed, target[%s], origin[%s].", target.c_str(),
             origin.c_str());
      return false;
    }
    if (!domi::OpRegistry::Instance()->Register(registration)) {
      GELOGE(FAILED, "Register Python ONNX plugin failed, target[%s], origin[%s].", target.c_str(), origin.c_str());
      return false;
    }
    return true;
  }

  std::mutex mutex_;
  bool initialized_ = false;
  std::string native_module_path_;
  py::object bridge_module_;
  py::object invalid_return_exception_;
  py::object invalid_decompose_return_exception_;
};

}  // namespace

extern "C" Status SetOnnxPluginBridgeArtifactConfig(
    const onnx_plugin_bridge::PythonOnnxPluginBridgeArtifactConfig *config) {
  return OnnxPluginBridge::Instance().SetArtifactConfig(config);
}

extern "C" Status RegisterOnnxPluginBridgePlugins() {
  return OnnxPluginBridge::Instance().Initialize();
}

extern "C" void ResetOnnxPluginBridgeState() {
  OnnxPluginBridge::Instance().ResetBridgeState();
}

extern "C" Status InitOnnxPluginBridge() {
  if (GePythonRuntimeManager::Instance().EnsureReady() != SUCCESS) {
    return FAILED;
  }
  return OnnxPluginBridge::Instance().Initialize();
}

extern "C" const onnx_plugin_bridge::PythonOnnxPluginBridgeApi *GeGetPythonOnnxPluginBridgeApi() {
  static const onnx_plugin_bridge::PythonOnnxPluginBridgeApi api = {
      onnx_plugin_bridge::kPythonOnnxPluginBridgeAbiVersion,
      &SetOnnxPluginBridgeArtifactConfig,
      &RegisterOnnxPluginBridgePlugins,
      &ResetOnnxPluginBridgeState,
  };
  return &api;
}
}  // namespace ge
