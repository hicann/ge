/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "parser/onnx/python_onnx_plugin_bridge/onnx_plugin_bridge_registrar.h"

#include "common/ge_common/debug/ge_log.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/graph.h"
#include "graph/operator.h"
#include "parser/common/op_parser_factory.h"
#include "parser/common/op_registration_tbe.h"
#include "register/op_registry.h"
#include "register/register_fmk_types.h"

#include <string>

namespace ge {
namespace {
namespace onnx_bridge = ::ge::onnx_plugin_bridge;

constexpr const char *kParseNodeCallbackKind = "parse_node";
constexpr const char *kParseOperatorCallbackKind = "parse_operator";
constexpr const char *kDecomposeCallbackKind = "decompose";

bool RegisterCallbacks(const std::string &origin, const onnx_bridge::PythonOnnxPluginDescriptorView &desc,
                       const onnx_bridge::PythonOnnxPluginParseCallbacks *callbacks,
                       domi::OpRegistrationData &registration, bool &has_parse_params_callback,
                       bool &has_graph_callback) {
  for (size_t i = 0U; i < desc.callback_kinds_count; ++i) {
    const char *const callback_kind = desc.callback_kinds[i];
    if (callback_kind == nullptr) {
      GELOGE(FAILED, "Python ONNX plugin origin[%s] has a null callback kind.", origin.c_str());
      return false;
    }
    if (std::string(callback_kind) == kParseNodeCallbackKind) {
      has_parse_params_callback = true;
      const domi::ParseParamFunc parse_params = [callbacks](const google::protobuf::Message *message,
                                                            Operator &operator_dest) -> Status {
        return callbacks->parse_params(message, &operator_dest);
      };
      registration.ParseParamsFn(parse_params);
    } else if (std::string(callback_kind) == kParseOperatorCallbackKind) {
      has_parse_params_callback = true;
      const domi::ParseParamByOpFunc parse_params = [origin, callbacks](const Operator &operator_src,
                                                                        Operator &operator_dest) -> Status {
        return callbacks->parse_params_by_operator(origin.c_str(), &operator_src, &operator_dest);
      };
      registration.ParseParamsByOperatorFn(parse_params);
    } else if (std::string(callback_kind) == kDecomposeCallbackKind) {
      has_graph_callback = true;
      const domi::ParseOpToGraphFunc parse_op_to_graph = [origin, callbacks](const Operator &operator_src,
                                                                             Graph &subgraph) -> Status {
        return callbacks->parse_op_to_graph(origin.c_str(), &operator_src, &subgraph);
      };
      registration.ParseOpToGraphFn(parse_op_to_graph);
    } else {
      GELOGE(PARAM_INVALID, "Unknown Python ONNX plugin callback kind[%s], origin[%s].", callback_kind, origin.c_str());
      return false;
    }
  }
  return true;
}

bool IsOriginRegistered(const std::string &target, const std::string &origin) {
  std::string registered_target;
  if (!domi::OpRegistry::Instance()->GetOmTypeByOriOpType(origin, registered_target)) {
    return false;
  }
  if (registered_target != target) {
    GELOGW("Skip Python ONNX plugin for origin[%s], existing registration maps it to target[%s].", origin.c_str(),
           registered_target.c_str());
  } else {
    GELOGI("Skip duplicate Python ONNX plugin registration for target[%s], origin[%s].", target.c_str(),
           origin.c_str());
  }
  return true;
}

bool RegisterOnnxPluginFromBridge(const onnx_bridge::PythonOnnxPluginDescriptorView *desc,
                                  const onnx_bridge::PythonOnnxPluginParseCallbacks *callbacks) {
  if ((desc == nullptr) || (desc->target == nullptr) || (desc->origin == nullptr) ||
      (desc->callback_kinds == nullptr) || (desc->callback_kinds_count == 0U) || (callbacks == nullptr)) {
    GELOGE(FAILED, "Invalid Python ONNX plugin descriptor or callbacks.");
    return false;
  }
  const std::string target(desc->target);
  const std::string origin(desc->origin);
  if (IsOriginRegistered(target, origin)) {
    return true;
  }
  domi::OpRegistrationData registration(target.c_str());
  registration.FrameworkType(domi::ONNX).OriginOpType(origin.c_str());
  bool has_parse_params_callback = false;
  bool has_graph_callback = false;
  if (!RegisterCallbacks(origin, *desc, callbacks, registration, has_parse_params_callback, has_graph_callback)) {
    return false;
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

}  // namespace

const onnx_plugin_bridge::PythonOnnxPluginRegistrar *GetOnnxPluginBridgeRegistrar() {
  static const onnx_bridge::PythonOnnxPluginRegistrar registrar = {&RegisterOnnxPluginFromBridge};
  return &registrar;
}

}  // namespace ge
