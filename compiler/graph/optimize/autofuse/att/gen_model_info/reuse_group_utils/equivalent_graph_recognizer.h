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

#ifndef AUTOFUSE_ATT_GEN_MODEL_INFO_REUSE_GROUP_UTILS_EQUIVALENT_GRAPH_RECOGNIZER_H_
#define AUTOFUSE_ATT_GEN_MODEL_INFO_REUSE_GROUP_UTILS_EQUIVALENT_GRAPH_RECOGNIZER_H_

#include "base/model_info.h"
namespace att {
class EquivalentGraphRecognizer {
 public:
  EquivalentGraphRecognizer(const ge::AscGraph &graph_to, const ge::AscGraph &graph_from,
                            const ReuseScheduleGroupInfo &group_info_to, const ReuseScheduleGroupInfo &group_info_from);
  ~EquivalentGraphRecognizer() = default;
  bool IsEquivalent();
  const std::vector<std::string> &GetMappedInputAxesNames() const {
    return graph_to_ordered_input_names_;
  }
 private:
  bool IsAscNodeEquivalent(ge::AscNode &node1, ge::AscNode &node2);
  bool IsAscTensorEquivalent(const ge::AscTensorAttr &tensor1, const ge::AscTensorAttr &tensor2);
  bool IsAscNodeAttrEquivalent(const ge::AscNodeAttr &node_attr_to, const ge::AscNodeAttr &node_attr_from);
  bool CompareExpression(const ge::Expression &expr1, const ge::Expression &expr2);
  bool CanExprEquivalentAfterReplace(const ge::Expression &replace_expr, const ge::Expression &reuse_expr);
  bool IsTensorViewEquivalent(const ge::AscTensorAttr &tensor1,
                              const ge::AscTensorAttr &tensor2);
  bool CompareExprs(const std::vector<ge::Expression> &exprs1, const std::vector<ge::Expression> &exprs2);
  bool CompareAxis(const int64_t axis_id, const int64_t axis_id2) const;
  bool IsMemEquivalent(const ge::AscTensorAttr &tensor1, const ge::AscTensorAttr &tensor2) const;
  bool IsInputNodeSame(const ge::AscNodePtr &asc_node1, const ge::AscNodePtr &asc_node2);
  bool IsInputVar(const Expr &expr1, const Expr &expr2) const;
  std::string ReplaceSearchVarStr(const std::string &str) const;
  bool UpdateOrderedInputNames();
  bool IsInputAxesFromDuplicityMapped() const;
  // 检查graph_to_是否可以复用graph_from_的图
  ge::AscGraph graph_to_;
  ge::AscGraph graph_from_;
  const ReuseScheduleGroupInfo &group_info_to_;
  const ReuseScheduleGroupInfo &group_info_from_;
  std::set<std::string> search_axes_name_to_;
  std::set<std::string> search_axes_name_from_;
  std::set<std::string> input_axes_name_to_;
  std::set<std::string> input_axes_name_from_;
  std::map<int64_t, ge::AxisPtr> axis_id_to_axis_map_to_;
  std::map<int64_t, ge::AxisPtr> axis_id_to_axis_map_from_;
  // 按照graph_from_的输入轴映射到graph_to_输入轴名称
  std::map<std::string, std::string> mapped_input_axes_names_;
  // 按照顺序排布的graph_to_的输入轴，与graph_from_的一一映射(输出)
  std::vector<std::string> graph_to_ordered_input_names_;
};
}
#endif  // AUTOFUSE_ATT_GEN_MODEL_INFO_REUSE_GROUP_UTILS_EQUIVALENT_GRAPH_RECOGNIZER_H_
