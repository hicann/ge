/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "binding_utils.h"
#include "bindings.h"

#include <vector>

#include "ge/fusion/graph_fuse_inspector_utils.h"

namespace ge {
namespace python_pass_native {
namespace {
py::tuple CanFuse(const py::iterable &node_objects) {
  const py::object node_type = py::module_::import("ge.graph").attr("Node");
  std::vector<GNode> nodes;
  for (const py::handle node_object : node_objects) {
    if (!py::isinstance(node_object, node_type)) {
      throw py::type_error("nodes must contain only ge.graph.Node objects");
    }
    const auto *node = BorrowNodeFromPython(node_object);
    if (node == nullptr) {
      throw std::runtime_error("Node handle is empty");
    }
    nodes.emplace_back(*node);
  }

  AscendString failed_reason;
  const bool ok = fusion::GraphFuseInspectorUtils::CanFuse(nodes, failed_reason);
  const char *const reason = failed_reason.GetString();
  return py::make_tuple(ok, reason == nullptr ? "" : reason);
}
}  // namespace

void BindGraphFuseInspector(py::module_ &m) {
  m.def("can_fuse", &CanFuse, py::arg("nodes"), "Check whether nodes can be safely fused into one node");
}

}  // namespace python_pass_native
}  // namespace ge
