/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "if_subgraph_adapter.h"
#include "subgraph_adapter_factory.h"
#include "common/util.h"
#include "framework/common/debug/ge_log.h"
#include "base/err_msg.h"

namespace ge {
using parser::IF;
namespace {
const std::map<std::string, int> kAttrNameToIndex = {{"then_branch", 0}, {"else_branch", 1}};
const int kIfNodeAttrSize = 2;
const char *kIf = "If";

void CollectGraphFreeInputs(const ge::onnx::GraphProto &onnx_graph, std::set<std::string> &free_inputs);

void CollectGraphDefinedValues(const ge::onnx::GraphProto &onnx_graph, std::set<std::string> &defined_values) {
  for (int i = 0; i < onnx_graph.input_size(); i++) {
    const std::string &input_name = onnx_graph.input(i).name();
    if (!input_name.empty()) {
      defined_values.emplace(input_name);
    }
  }
  for (int i = 0; i < onnx_graph.initializer_size(); i++) {
    const std::string &initializer_name = onnx_graph.initializer(i).name();
    if (!initializer_name.empty()) {
      defined_values.emplace(initializer_name);
    }
  }
}

void CollectNestedGraphFreeInputs(const ge::onnx::NodeProto &node_proto, std::set<std::string> &used_values) {
  for (int j = 0; j < node_proto.attribute_size(); j++) {
    const ge::onnx::AttributeProto &attribute = node_proto.attribute(j);
    if (attribute.has_g()) {
      std::set<std::string> nested_free_inputs;
      CollectGraphFreeInputs(attribute.g(), nested_free_inputs);
      used_values.insert(nested_free_inputs.begin(), nested_free_inputs.end());
    }
    for (int k = 0; k < attribute.graphs_size(); k++) {
      std::set<std::string> nested_free_inputs;
      CollectGraphFreeInputs(attribute.graphs(k), nested_free_inputs);
      used_values.insert(nested_free_inputs.begin(), nested_free_inputs.end());
    }
  }
}

void CollectNodeUsedAndDefinedValues(const ge::onnx::NodeProto &node_proto, std::set<std::string> &used_values,
                                     std::set<std::string> &defined_values) {
  for (int j = 0; j < node_proto.input_size(); j++) {
    const std::string &input_name = node_proto.input(j);
    if (!input_name.empty()) {
      used_values.emplace(input_name);
    }
  }
  for (int j = 0; j < node_proto.output_size(); j++) {
    const std::string &output_name = node_proto.output(j);
    if (!output_name.empty()) {
      defined_values.emplace(output_name);
    }
  }
  CollectNestedGraphFreeInputs(node_proto, used_values);
}

// 收集当前图及其节点属性嵌套子图中引用的名字，排除当前图作用域内已定义的值。
// ONNX 的 If 分支可能直接引用祖先图中的值而不在 GraphProto 输入中声明，
// 因此嵌套子图的自由输入集合必须逐层向上传播。
void CollectGraphFreeInputs(const ge::onnx::GraphProto &onnx_graph, std::set<std::string> &free_inputs) {
  std::set<std::string> used_values;
  std::set<std::string> defined_values;
  CollectGraphDefinedValues(onnx_graph, defined_values);

  for (int i = 0; i < onnx_graph.node_size(); i++) {
    CollectNodeUsedAndDefinedValues(onnx_graph.node(i), used_values, defined_values);
  }

  // 图输出本身也可能是外层作用域的值（无本地节点产生时），将其视为一次使用，
  // 避免捕获信息在图边界丢失。
  for (int i = 0; i < onnx_graph.output_size(); i++) {
    const std::string &output_name = onnx_graph.output(i).name();
    if (!output_name.empty()) {
      used_values.emplace(output_name);
    }
  }

  for (const std::string &used_value : used_values) {
    if (defined_values.count(used_value) == 0) {
      free_inputs.emplace(used_value);
    }
  }
}
}  // namespace
domi::Status IfSubgraphAdapter::AdaptAndFindAllSubgraphs(
    ge::onnx::NodeProto *parent_node, std::vector<ge::onnx::GraphProto *> &onnx_graphs,
    std::map<std::string, ge::onnx::GraphProto *> &name_to_onnx_graph, const std::string &parent_graph_name) {
  GE_CHECK_NOTNULL(parent_node);
  GELOGI("Onnx parent node name=%s, op type=%s, adapt subgraph.", parent_node->name().c_str(),
         parent_node->op_type().c_str());

  auto ret = ParseIfNodeSubgraphs(*parent_node, onnx_graphs, name_to_onnx_graph, parent_graph_name);
  if (ret != SUCCESS) {
    GELOGE(ret, "[Parse][Node] Parse if node failed.");
    return ret;
  }

  return SUCCESS;
}

domi::Status IfSubgraphAdapter::ParseIfNodeSubgraphs(ge::onnx::NodeProto &parent_node,
                                                     std::vector<ge::onnx::GraphProto *> &onnx_graphs,
                                                     std::map<std::string, ge::onnx::GraphProto *> &name_to_onnx_graph,
                                                     const std::string &parent_graph_name) const {
  if (parent_node.attribute_size() != kIfNodeAttrSize) {
    GELOGE(FAILED, "[Parse][Node] Invalid graph, if node attribute size:%d must be 2.", parent_node.attribute_size());
    REPORT_INNER_ERR_MSG("E19999", "Invalid graph, if node attribute size:%d must be 2.", parent_node.attribute_size());
    return FAILED;
  }

  GELOGD("node attribute size:%d.", parent_node.attribute_size());
  std::set<std::string> all_inputs;
  // for onnx graph, the first attribute may be else branch and the second attribute may be then branch
  for (int i = 0; i < parent_node.attribute_size(); i++) {
    ge::onnx::AttributeProto *attribute = parent_node.mutable_attribute(i);
    GE_CHECK_NOTNULL(attribute);
    std::string attr_name = attribute->name();
    auto itr = kAttrNameToIndex.find(attr_name);
    if (itr == kAttrNameToIndex.end()) {
      GELOGE(FAILED, "[Parse][Attribute] Invalid attribute name:%s, it should be then_branch or else_branch.",
             attr_name.c_str());
      REPORT_INNER_ERR_MSG("E19999", "Invalid attribute name:%s, it should be then_branch or else_branch.",
                           attr_name.c_str());
      return FAILED;
    }
    std::string unique_subgraph_name;
    std::string node_name = parent_node.name();
    if (!parent_graph_name.empty()) {
      node_name = OnnxUtil::GenUniqueNodeName(parent_graph_name, node_name);
    }
    OnnxUtil::GenUniqueSubgraphName(itr->second, itr->first, node_name, unique_subgraph_name);
    GELOGI("Adapt if node attribute:%s, subgraph name:%s.", attr_name.c_str(), unique_subgraph_name.c_str());
    ge::onnx::GraphProto *onnx_graph = attribute->mutable_g();
    name_to_onnx_graph[unique_subgraph_name] = onnx_graph;
    onnx_graphs.emplace_back(onnx_graph);

    auto ret = GetSubgraphsAllInputs(*onnx_graph, all_inputs);
    if (ret != SUCCESS) {
      GELOGE(ret, "[Get][Inputs] Get subgraph all inputs failed, attr_name:%s.", attr_name.c_str());
      return ret;
    }
  }

  for (auto &onnx_graph : onnx_graphs) {
    AddInputNodeForGraph(all_inputs, *onnx_graph);
  }

  AddInputForParentNode(all_inputs, parent_node);
  return SUCCESS;
}

domi::Status IfSubgraphAdapter::GetSubgraphsAllInputs(ge::onnx::GraphProto &onnx_graph,
                                                      std::set<std::string> &all_inputs) const {
  std::set<std::string> graph_free_inputs;
  CollectGraphFreeInputs(onnx_graph, graph_free_inputs);
  for (const auto &free_input : graph_free_inputs) {
    GELOGD("[Collect][FreeInput] Subgraph %s captures outer-scope value %s.", onnx_graph.name().c_str(),
           free_input.c_str());
  }
  all_inputs.insert(graph_free_inputs.begin(), graph_free_inputs.end());
  return SUCCESS;
}

void IfSubgraphAdapter::AddInputNodeForGraph(const std::set<std::string> &all_inputs,
                                             ge::onnx::GraphProto &onnx_graph) const {
  std::set<std::string> existing_inputs;
  for (int i = 0; i < onnx_graph.input_size(); i++) {
    existing_inputs.emplace(onnx_graph.input(i).name());
  }
  for (const auto &input_name : all_inputs) {
    if (!existing_inputs.emplace(input_name).second) {
      continue;
    }
    GELOGI("[Add][SubgraphInput] Add outer-scope input %s to subgraph %s.", input_name.c_str(),
           onnx_graph.name().c_str());
    ge::onnx::ValueInfoProto *value_info = onnx_graph.add_input();
    value_info->set_name(input_name);
  }
}

void IfSubgraphAdapter::AddInputForParentNode(const std::set<std::string> &all_inputs,
                                              ge::onnx::NodeProto &parent_node) const {
  std::set<std::string> existing_inputs;
  // input[0] 固定是 cond 槽位，不参与去重：若分支子图闭包捕获了 cond，
  // 子图侧会把它追加为新的子图 input（对应 If.input 的下一个槽位），
  // 此时 If 侧必须同步追加同名输入，不能复用 input[0]，否则子图 Data 的
  // parent_index = data_index + 1 会越界。
  for (int i = 1; i < parent_node.input_size(); i++) {
    existing_inputs.emplace(parent_node.input(i));
  }
  for (const auto &input_name : all_inputs) {
    if (!existing_inputs.emplace(input_name).second) {
      continue;
    }
    GELOGI("[Add][ParentNodeInput] Add outer-scope input %s to if node %s.", input_name.c_str(),
           parent_node.name().c_str());
    parent_node.add_input(input_name);
  }
}
REGISTER_SUBGRAPH_ADAPTER_CREATOR(kIf, IfSubgraphAdapter);
}  // namespace ge
