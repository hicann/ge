/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef TESTS_GE_UT_GE_ONNX_PLUGIN_TEST_HELPER_H_
#define TESTS_GE_UT_GE_ONNX_PLUGIN_TEST_HELPER_H_

#include <gtest/gtest.h>

#include "pybind11/embed.h"
#include "pybind11/eval.h"

namespace ge {
namespace onnx_plugin_test {
namespace py = pybind11;

constexpr const char *kInMemoryPluginSource = R"PY(
from ge.onnx_plugin import onnx_plugin

elu = onnx_plugin(
    source="BridgeElu", domain="test.domain", opsets=(1,), target="BridgeEluTarget"
)
@elu.parse_node
def parse_elu(node, target):
    target.set_attr("alpha", node.attrs["alpha"])

error = onnx_plugin(
    source="BridgeError", domain="test.domain", opsets=(1,), target="BridgeErrorTarget"
)

@error.parse_node
def parse_error(node, target):
    del node, target
    raise RuntimeError("python callback failed")
returned = onnx_plugin(
    source="BridgeReturn", domain="test.domain", opsets=(1,), target="BridgeReturnTarget"
)

@returned.parse_node
def parse_returned(node, target):
    del node, target
    return False

priority = onnx_plugin(
    source="BridgePriority", domain="test.domain", opsets=(1,), target="BridgePriorityTarget"
)

@priority.parse_node
def parse_priority(node, target):
    del node
    target.set_attr("source", "python")
)PY";

class ScopedInMemoryPlugin {
 public:
  ScopedInMemoryPlugin() {
    py::gil_scoped_acquire gil;
    py::dict scope;
    scope["module_name"] = py::str("_ge_py_onnx_plugin_in_memory");
    scope["source"] = py::str(kInMemoryPluginSource);
    py::exec(R"PY(
import sys
import ge.onnx_plugin.bootstrap as _bootstrap
module = sys.modules.get(module_name)
if module is None:
    module = __import__("types").ModuleType(module_name)
    module.__file__ = "<memory-onnx-plugin>"
    exec(compile(source, module.__file__, "exec"), module.__dict__)
    sys.modules[module.__name__] = module
_bootstrap._mde_memory_original_loader = _bootstrap.load_plugins_from_env
_bootstrap.load_plugins_from_env = lambda *args, **kwargs: [module]
)PY",
             scope, scope);
    module_ = py::reinterpret_borrow<py::object>(scope["module"]);
  }

  ~ScopedInMemoryPlugin() {
    try {
      py::gil_scoped_acquire gil;
      py::dict scope;
      scope["module"] = module_;
      py::exec(R"PY(
import sys
import ge.onnx_plugin.bootstrap as _bootstrap
_bootstrap.load_plugins_from_env = _bootstrap._mde_memory_original_loader
del _bootstrap._mde_memory_original_loader
)PY",
               scope, scope);
      (void)module_.release();
    } catch (...) {
      ADD_FAILURE() << "Failed to restore the ONNX plugin loader.";
    }
  }

 private:
  py::object module_;
};

}  // namespace onnx_plugin_test
}  // namespace ge

#endif  // TESTS_GE_UT_GE_ONNX_PLUGIN_TEST_HELPER_H_
