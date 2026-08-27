/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <dlfcn.h>
#include <gtest/gtest.h>
#include <cstdlib>
#include <string>

#include "common/python_runtime/ge_python_runtime_manager.h"
#include "framework/omg/parser/parser_factory.h"
#include "common/ge_common/ge_inner_error_codes.h"
#include "ge/ge_api_error_codes.h"
#include "graph/operator.h"
#include "graph/utils/attr_utils.h"
#include "parser/common/op_registration_tbe.h"
#include "parser/onnx/python_onnx_plugin_bridge/onnx_plugin_bridge_c_api.h"
#include "parser/onnx/python_onnx_plugin_bridge/onnx_plugin_bridge_loader.h"
#include "proto/onnx/ge_onnx.pb.h"
#include "register/op_registry.h"
#include "register/register_fmk_types.h"
#include "onnx_plugin_test_helper.h"
#include "pybind11/embed.h"
#include "pybind11/eval.h"

namespace ge {
namespace {
namespace py = pybind11;

Status ParseCppPriority(const google::protobuf::Message *, Operator &op_dest) {
  op_dest.SetAttr("source", "cpp");
  return SUCCESS;
}

class ScopedBadSyntaxPlugin {
 public:
  ScopedBadSyntaxPlugin() {
    py::gil_scoped_acquire gil;
    py::exec(R"PY(
import ge.onnx_plugin.bootstrap as _bootstrap
_bootstrap._mde_original_loader = _bootstrap.load_plugins_from_env
def _mde_bad_loader(*args, **kwargs):
    compile("def broken(:\n    pass\n", "<memory-bad-plugin>", "exec")
_bootstrap.load_plugins_from_env = _mde_bad_loader
)PY");
  }

  ~ScopedBadSyntaxPlugin() {
    try {
      py::gil_scoped_acquire gil;
      py::exec(R"PY(
import ge.onnx_plugin.bootstrap as _bootstrap
_bootstrap.load_plugins_from_env = _bootstrap._mde_original_loader
del _bootstrap._mde_original_loader
)PY");
    } catch (...) {
      ADD_FAILURE() << "Failed to restore the ONNX plugin loader.";
    }
  }
};

}  // namespace

using onnx_plugin_test::ScopedInMemoryPlugin;

TEST(OnnxPythonPluginBridge, ResetBeforePythonRuntimeDoesNotCrash) {
  void *bridge = dlopen(ONNX_PYTHON_PLUGIN_BRIDGE_PATH, RTLD_NOW | RTLD_GLOBAL);
  ASSERT_NE(bridge, nullptr);
  using ResetBridgeFunc = void (*)();
  const auto reset_bridge = reinterpret_cast<ResetBridgeFunc>(dlsym(bridge, "ResetOnnxPluginBridgeState"));
  ASSERT_NE(reset_bridge, nullptr);
  reset_bridge();
}

TEST(OnnxPythonPluginBridge, InitializeFailsOnBadSyntaxPlugin) {
  ASSERT_EQ(GePythonRuntimeManager::Instance().EnsureReady(), SUCCESS);

  void *bridge = dlopen(ONNX_PYTHON_PLUGIN_BRIDGE_PATH, RTLD_NOW | RTLD_GLOBAL);
  ASSERT_NE(bridge, nullptr);
  using InitBridgeFunc = Status (*)();
  const auto init_bridge = reinterpret_cast<InitBridgeFunc>(dlsym(bridge, "InitOnnxPluginBridge"));
  ASSERT_NE(init_bridge, nullptr);

  ASSERT_EQ(unsetenv("ASCEND_CUSTOM_OPP_PATH"), 0);
  EXPECT_EQ(init_bridge(), SUCCESS);
  using ResetBridgeFunc = void (*)();
  const auto reset_bridge = reinterpret_cast<ResetBridgeFunc>(dlsym(bridge, "ResetOnnxPluginBridgeState"));
  ASSERT_NE(reset_bridge, nullptr);
  reset_bridge();

  std::string native_module_path;
  {
    py::gil_scoped_acquire gil;
    const auto native_module = py::module_::import("ge.onnx_plugin._ge_onnx_plugin_native");
    native_module_path = py::str(native_module.attr("__file__"));
    const auto modules = py::module_::import("sys").attr("modules");
    (void)modules.attr("pop")("ge.onnx_plugin._ge_onnx_plugin_native", py::none());
  }
  using SetConfigFunc = Status (*)(const onnx_plugin_bridge::PythonOnnxPluginBridgeArtifactConfig *);
  const auto set_config = reinterpret_cast<SetConfigFunc>(dlsym(bridge, "SetOnnxPluginBridgeArtifactConfig"));
  ASSERT_NE(set_config, nullptr);
  onnx_plugin_bridge::PythonOnnxPluginBridgeArtifactConfig config{nullptr, native_module_path.c_str()};
  ASSERT_EQ(set_config(&config), SUCCESS);

  ScopedBadSyntaxPlugin bad_syntax_plugin;
  ASSERT_EQ(setenv("ASCEND_CUSTOM_OPP_PATH", "__mde_bad_syntax_plugin_in_memory__", 1), 0);
  EXPECT_NE(init_bridge(), SUCCESS);
  unsetenv("ASCEND_CUSTOM_OPP_PATH");
}

TEST(OnnxPythonPluginBridge, ParseParamsUsesNativeNodeAndPreservesCppPriority) {
  ScopedInMemoryPlugin in_memory_plugin;
  ASSERT_EQ(setenv("ASCEND_CUSTOM_OPP_PATH", "__ge_py_onnx_plugin_in_memory__", 1), 0);
  domi::OpRegistrationData cpp_registration("BridgePriorityTarget");
  cpp_registration.FrameworkType(domi::ONNX)
      .OriginOpType("test.domain::1::BridgePriority")
      .ParseParamsFn(ParseCppPriority);
  (void)domi::OpRegTbeParserFactory::Instance()->Finalize(cpp_registration);
  ASSERT_TRUE(domi::OpRegistry::Instance()->Register(cpp_registration));

  void *bridge = dlopen(ONNX_PYTHON_PLUGIN_BRIDGE_PATH, RTLD_NOW | RTLD_GLOBAL);
  ASSERT_NE(bridge, nullptr);
  using InitBridgeFunc = Status (*)();
  const auto init_bridge = reinterpret_cast<InitBridgeFunc>(dlsym(bridge, "InitOnnxPluginBridge"));
  ASSERT_NE(init_bridge, nullptr);
  ASSERT_EQ(init_bridge(), SUCCESS);
  ASSERT_EQ(init_bridge(), SUCCESS);

  ge::onnx::NodeProto node;
  node.set_name("bridge_node");
  node.set_op_type("test.domain::1::BridgeElu");
  node.add_input("x");
  auto *alpha = node.add_attribute();
  alpha->set_name("alpha");
  alpha->set_type(ge::onnx::AttributeProto_AttributeType_FLOAT);
  alpha->set_f(0.5F);

  Operator op("bridge_node", "BridgeEluTarget");
  const auto parse_elu = domi::OpRegistry::Instance()->GetParseParamFunc("BridgeEluTarget", node.op_type());
  ASSERT_NE(parse_elu, nullptr);
  EXPECT_EQ(parse_elu(&node, op), SUCCESS);
  float alpha_value = 0.0F;
  EXPECT_EQ(op.GetAttr("alpha", alpha_value), GRAPH_SUCCESS);
  EXPECT_FLOAT_EQ(alpha_value, 0.5F);

  node.set_op_type("test.domain::1::BridgeError");
  Operator error_op("error_node", "BridgeErrorTarget");
  const auto parse_error = domi::OpRegistry::Instance()->GetParseParamFunc("BridgeErrorTarget", node.op_type());
  ASSERT_NE(parse_error, nullptr);
  EXPECT_EQ(parse_error(&node, error_op), FAILED);

  node.set_op_type("test.domain::1::BridgeReturn");
  Operator return_op("return_node", "BridgeReturnTarget");
  const auto parse_return = domi::OpRegistry::Instance()->GetParseParamFunc("BridgeReturnTarget", node.op_type());
  ASSERT_NE(parse_return, nullptr);
  EXPECT_NE(parse_return(&node, return_op), SUCCESS);

  Operator source_op("operator_source", "BridgeOperator");
  source_op.SetAttr("alpha", 0.5F);
  Operator operator_target("operator_target", "BridgeOperatorTarget");
  const auto parse_operator =
      domi::OpRegistry::Instance()->GetParseParamByOperatorFunc("test.domain::1::BridgeOperator");
  ASSERT_NE(parse_operator, nullptr);
  EXPECT_EQ(parse_operator(source_op, operator_target), SUCCESS);
  float copied_alpha = 0.0F;
  EXPECT_EQ(operator_target.GetAttr("copied_alpha", copied_alpha), GRAPH_SUCCESS);
  EXPECT_FLOAT_EQ(copied_alpha, 0.5F);

  node.set_op_type("test.domain::1::BridgeBoth");
  Operator both_node_target("both_node", "BridgeBothTarget");
  const auto parse_both_node = domi::OpRegistry::Instance()->GetParseParamFunc("BridgeBothTarget", node.op_type());
  ASSERT_NE(parse_both_node, nullptr);
  EXPECT_EQ(parse_both_node(&node, both_node_target), SUCCESS);
  std::string callback;
  EXPECT_EQ(both_node_target.GetAttr("callback", callback), GRAPH_SUCCESS);
  EXPECT_EQ(callback, "parse_node");

  Operator both_operator_target("both_operator", "BridgeBothTarget");
  const auto parse_both_operator = domi::OpRegistry::Instance()->GetParseParamByOperatorFunc(node.op_type());
  ASSERT_NE(parse_both_operator, nullptr);
  EXPECT_EQ(parse_both_operator(source_op, both_operator_target), SUCCESS);
  EXPECT_EQ(both_operator_target.GetAttr("callback", callback), GRAPH_SUCCESS);
  EXPECT_EQ(callback, "parse_operator");

  const auto parse_operator_error =
      domi::OpRegistry::Instance()->GetParseParamByOperatorFunc("test.domain::1::BridgeOperatorError");
  ASSERT_NE(parse_operator_error, nullptr);
  EXPECT_EQ(parse_operator_error(source_op, operator_target), FAILED);

  const auto parse_operator_return =
      domi::OpRegistry::Instance()->GetParseParamByOperatorFunc("test.domain::1::BridgeOperatorReturn");
  ASSERT_NE(parse_operator_return, nullptr);
  EXPECT_NE(parse_operator_return(source_op, operator_target), SUCCESS);

  node.set_op_type("test.domain::1::BridgePriority");
  Operator priority_op("priority_node", "BridgePriorityTarget");
  const auto parse_priority = domi::OpRegistry::Instance()->GetParseParamFunc("BridgePriorityTarget", node.op_type());
  ASSERT_NE(parse_priority, nullptr);
  EXPECT_EQ(parse_priority(&node, priority_op), SUCCESS);
  std::string source;
  EXPECT_EQ(priority_op.GetAttr("source", source), GRAPH_SUCCESS);
  EXPECT_EQ(source, "cpp");

  Operator null_op("null_node", "BridgeEluTarget");
  EXPECT_NE(parse_elu(nullptr, null_op), SUCCESS);

  ge::onnx::AttributeProto wrong_msg;
  EXPECT_NE(parse_elu(&wrong_msg, null_op), SUCCESS);

  using ResetBridgeFunc = void (*)();
  const auto reset_bridge = reinterpret_cast<ResetBridgeFunc>(dlsym(bridge, "ResetOnnxPluginBridgeState"));
  ASSERT_NE(reset_bridge, nullptr);
  reset_bridge();
  EXPECT_NE(parse_operator(source_op, operator_target), SUCCESS);
  EXPECT_NE(parse_elu(&node, priority_op), SUCCESS);
}

TEST(OnnxPythonPluginBridge, LoadThroughCommonLoader) {
  const char *old_plugin_path = std::getenv("ASCEND_CUSTOM_OPP_PATH");
  const std::string old_plugin_path_value = old_plugin_path == nullptr ? "" : old_plugin_path;
  const bool had_plugin_path = old_plugin_path != nullptr;
  const char *old_python_path = std::getenv("PYTHONPATH");
  const std::string old_python_path_value = old_python_path == nullptr ? "" : old_python_path;
  const bool had_python_path = old_python_path != nullptr;

  ASSERT_EQ(unsetenv("ASCEND_CUSTOM_OPP_PATH"), 0);
  EXPECT_EQ(LoadOnnxPythonPluginBridge(), SUCCESS);
  ASSERT_EQ(setenv("ASCEND_CUSTOM_OPP_PATH", "__ge_py_onnx_plugin_in_memory__", 1), 0);
  ASSERT_EQ(setenv("PYTHONPATH", "", 1), 0);
  EXPECT_EQ(LoadOnnxPythonPluginBridge(), FAILED);

  if (had_python_path) {
    ASSERT_EQ(setenv("PYTHONPATH", old_python_path_value.c_str(), 1), 0);
  } else {
    ASSERT_EQ(unsetenv("PYTHONPATH"), 0);
  }
  ScopedInMemoryPlugin in_memory_plugin;
  ASSERT_EQ(LoadOnnxPythonPluginBridge(), SUCCESS);
  ASSERT_EQ(LoadOnnxPythonPluginBridge(), SUCCESS);
  UnloadOnnxPythonPluginBridge();
  ASSERT_EQ(LoadOnnxPythonPluginBridge(), SUCCESS);
  UnloadOnnxPythonPluginBridge();

  if (had_plugin_path) {
    ASSERT_EQ(setenv("ASCEND_CUSTOM_OPP_PATH", old_plugin_path_value.c_str(), 1), 0);
  } else {
    ASSERT_EQ(unsetenv("ASCEND_CUSTOM_OPP_PATH"), 0);
  }
}

TEST(OnnxPythonPluginBridge, RejectsInvalidArtifactConfig) {
  void *bridge = dlopen(ONNX_PYTHON_PLUGIN_BRIDGE_PATH, RTLD_NOW | RTLD_GLOBAL);
  ASSERT_NE(bridge, nullptr);
  using SetConfigFunc = Status (*)(const onnx_plugin_bridge::PythonOnnxPluginBridgeArtifactConfig *);
  const auto set_config = reinterpret_cast<SetConfigFunc>(dlsym(bridge, "SetOnnxPluginBridgeArtifactConfig"));
  ASSERT_NE(set_config, nullptr);
  EXPECT_EQ(set_config(nullptr), ge::PARAM_INVALID);
  onnx_plugin_bridge::PythonOnnxPluginBridgeArtifactConfig config{nullptr, nullptr};
  EXPECT_EQ(set_config(&config), ge::PARAM_INVALID);
  config.native_module_path = "";
  EXPECT_EQ(set_config(&config), ge::PARAM_INVALID);
}

}  // namespace ge
