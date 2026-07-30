/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <stdexcept>
#include <string>

#include "binding_utils.h"
#include "bindings.h"
#include "ge_common/ge_api_error_codes.h"
#include "ge/fusion/infer_shape_util.h"
#include "ge/fusion/subgraph_boundary.h"

namespace ge {
namespace python_pass_native {
namespace {
using ::ge::fusion::InferShapeUtil;
using ::ge::fusion::MatchResult;
using ::ge::fusion::SubgraphBoundary;

std::string GetGraphName(const Graph &graph) {
  AscendString graph_name;
  if (graph.GetName(graph_name) != GRAPH_SUCCESS) {
    return "";
  }
  return AscendStringToString(graph_name);
}

std::string GetMatchResultName(const MatchResult &match_result) {
  return GetGraphName(match_result.GetPatternGraph());
}

std::string GetNodeName(const GNode &node) {
  AscendString node_name;
  if (node.GetName(node_name) != GRAPH_SUCCESS) {
    return "";
  }
  return AscendStringToString(node_name);
}

void CheckInferShapeStatus(Status status, const Graph &replacement_graph, const char *source_type,
                           const std::string &source_name) {
  if (status != SUCCESS) {
    std::string message = "infer_shape failed, replacement_graph=" + GetGraphName(replacement_graph) +
                          ", source_type=" + std::string(source_type);
    if (!source_name.empty()) {
      message += ", source_name=" + source_name;
    }
    throw std::runtime_error(message);
  }
}

void InferShape(const py::handle &replacement_obj, const py::handle &source_obj) {
  const auto *replacement_graph = BorrowGraphFromPython(replacement_obj);
  if (replacement_graph == nullptr) {
    throw std::runtime_error("replacement Graph handle is empty");
  }

  const py::module_ passes_module = py::module_::import("ge.passes");
  const py::object match_result_type = passes_module.attr("MatchResult");
  if (py::isinstance(source_obj, match_result_type)) {
    const auto *match_result = BorrowMatchResultFromPython(source_obj);
    CheckInferShapeStatus(InferShapeUtil::InferShape(*replacement_graph, *match_result), *replacement_graph,
                          "MatchResult", GetMatchResultName(*match_result));
    return;
  }

  const py::module_ graph_module = py::module_::import("ge.graph");
  const py::object node_type = graph_module.attr("Node");
  if (py::isinstance(source_obj, node_type)) {
    const auto *node = BorrowNodeFromPython(source_obj);
    if (node == nullptr) {
      throw std::runtime_error("source Node handle is empty");
    }
    CheckInferShapeStatus(InferShapeUtil::InferShape(*replacement_graph, *node), *replacement_graph, "Node",
                          GetNodeName(*node));
    return;
  }

  const auto &boundary = source_obj.cast<const SubgraphBoundary &>();
  CheckInferShapeStatus(InferShapeUtil::InferShape(*replacement_graph, boundary), *replacement_graph,
                        "SubgraphBoundary", "");
}
}  // namespace

void BindInferShape(py::module_ &m) {
  m.def("infer_shape", &InferShape, py::arg("replacement"), py::arg("source"),
        "Infer shape, data type, and format for a replacement graph");
}

}  // namespace python_pass_native
}  // namespace ge
