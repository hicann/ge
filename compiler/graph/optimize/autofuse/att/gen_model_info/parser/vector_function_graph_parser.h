/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025 All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef AUTOFUSE_VECTOR_FUNCTION_GRAPH_PARSER_H
#define AUTOFUSE_VECTOR_FUNCTION_GRAPH_PARSER_H

#include "ascendc_ir_core/ascendc_ir.h"
#include "gen_model_info/parser/tuning_space.h"

namespace att {
class VectorFunctionGraphParser {
 public:
  VectorFunctionGraphParser(const ge::AscNodePtr &asc_node, const ge::AscGraph &graph)
      : asc_node_(asc_node), graph_(graph) {}
  ~VectorFunctionGraphParser() = default;
  ge::Status Parse();
  [[nodiscard]] const std::vector<NodeInfo> &GetNodesInfos() const { return nodes_infos_; }

 private:
  ge::Status ParseNodeInfos(NodeInfo &node_info);
  ge::Status ParseInputTensors(NodeInfo &node_info);
  ge::Status ParseOutputTensors(NodeInfo &node_info);
  ge::Status GetVectorizedAxes(const TensorPtr &tensor, const ge::AscTensorAttr &tensor_attr) const;
  ge::Status ParseTensorInfo(const ge::AscTensorAttr &attr, const TensorPtr &tensor, size_t index);

  std::vector<NodeInfo> nodes_infos_;
  const ge::AscNodePtr &asc_node_;
  const ge::AscGraph &graph_;
};
}  // namespace att

#endif  // AUTOFUSE_VECTOR_FUNCTION_GRAPH_PARSER_H
