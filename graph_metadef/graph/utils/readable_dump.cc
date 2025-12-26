/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstring>
#include <stack>

#include "graph/utils/op_desc_utils.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/anchor_utils.h"
#include "graph/utils/readable_dump.h"
#include "graph/utils/tensor_adapter.h"
#include "graph/utils/default_attr_utils.h"
#include "graph/utils/tensor_value_utils.h"

namespace ge {
Status ReadableDump::GenReadableDump(std::stringstream &readable_ss, const ComputeGraphPtr &graph) {
  GE_ASSERT_NOTNULL(graph);
  const auto graph_name = graph->GetName();
  OutputHandler output_handler;
  output_handler.GenNodeToOutputsMap(graph);
  readable_ss << "graph(\"" + graph_name + "\"):" << std::endl;
  std::stringstream graph_output_ss("");
  for (const auto &node : graph->GetDirectNode()) {
    GE_ASSERT_NOTNULL(node);
    GE_ASSERT_NOTNULL(node->GetOpDesc());
    if (node->GetOpDesc()->GetType() != kNetOutput) {
      GenNodeDump(readable_ss, output_handler, node.get());
    } else {
      GenGraphOutput(graph_output_ss, node.get(), output_handler);
    }
  }
  if (!graph_output_ss.str().empty()) {
    readable_ss << std::endl;
    readable_ss << kIndentTwo << graph_output_ss.str();
    readable_ss << std::endl;
  }

  return SUCCESS;
}

void ReadableDump::GenNodeDump(std::stringstream &readable_ss, OutputHandler &output_handler, const Node *node) {
  readable_ss << GetInstanceName(node->GetName()) << " : ";
  readable_ss << GetNodeOutDegree(node) << " = ";
  readable_ss << GetNodeType(node);
  readable_ss << " (";
  GenNodeInputs(readable_ss, node, output_handler);
  GenNodeAttrs(readable_ss, node);
  readable_ss << ")" << std::endl;
  GenMultipleOutputsIfNeeded(readable_ss, node, output_handler);
}

std::string ReadableDump::GetInstanceName(const std::string &name, const std::string &indent) {
  return indent + "%" + name;
}

std::string ReadableDump::GetNodeOutDegree(const Node *node) {
  GE_ASSERT_NOTNULL(node);
  return "[#users=" + std::to_string(node->GetAllOutDataAnchorsPtr().size()) + "]";
}

std::string ReadableDump::GetNodeType(const Node *node) {
  GE_ASSERT_NOTNULL(node);
  return "Node[type=" + node->GetType() + "]";
}

std::string ReadableDump::GetNodeInputInstance(const Node *node, OutputHandler &output_handler) {
  GE_ASSERT_NOTNULL(node);
  std::stringstream input_instance_ss;
  auto in_data_anchors = node->GetAllInDataAnchors();
  bool first = true;
  for (uint32_t i = 0; i < node->GetAllInDataAnchorsSize(); i++) {
    if (in_data_anchors.at(i)->GetPeerAnchorsSize() == 0) {
      continue;
    }
    auto input_node_anchor_pair = NodeUtils::GetInDataNodeAndAnchorByIndex(*node, static_cast<int32_t>(i));
    if (input_node_anchor_pair.first == nullptr || input_node_anchor_pair.second == nullptr) {
      continue;
    }
    auto input_node_name = input_node_anchor_pair.first->GetName();
    auto input_node_index = AnchorUtils::GetIdx(input_node_anchor_pair.second);
    auto input_name_vec = output_handler.GetNodeToOutputsMap().at(input_node_name);
    GE_ASSERT_TRUE(input_node_index < static_cast<int32_t>(input_name_vec->size()));
    auto input_instance_name = input_name_vec->at(input_node_index);
    if (first) {
      first = false;
    } else {
      input_instance_ss << ", ";
    }
    input_instance_ss << GetInstanceName(input_instance_name, kIndentZero);
  }

  return input_instance_ss.str();
}

void ReadableDump::GenNodeInputs(std::stringstream &readable_ss, const Node *node, OutputHandler &output_handler) {
  if (node->GetInDataNodesSize() == 0 || node->GetAllInDataAnchorsSize() == 0) {
    return;
  }
  readable_ss << "inputs = (";
  readable_ss << GetNodeInputInstance(node, output_handler);
  readable_ss << ")";
}

std::string ReadableDump::GetAttrValueStr(const OpDescPtr &op_desc, const std::string &attr_name,
                                          const AnyValue &attr_value, const std::string &av_type) {
  if (av_type == "VT_TENSOR") {
    auto attr_tensor_value = attr_value.Get<GeTensor>();
    GE_ASSERT_NOTNULL(attr_tensor_value);
    auto attr_tensor_type = attr_tensor_value->GetTensorDesc().GetDataType();
    return TensorValueUtils::ConvertTensorValue(TensorAdapter::AsTensor(*attr_tensor_value), attr_tensor_type, " ", true);
  }

  std::string attr_value_str;
  try {
    attr_value_str = AttrString::GetDefaultValueString(op_desc, attr_name, av_type, true);
  } catch (...) {
    GELOGW("Unable to get the attribute %s value", attr_name.c_str());
    return attr_value_str;
  }
  if (av_type == "VT_DATA_TYPE" || av_type == "VT_LIST_DATA_TYPE") {
    const std::string prefix = "ge::";
    size_t pos = 0;
    while ((pos = attr_value_str.find(prefix, pos)) != std::string::npos) {
      attr_value_str.erase(pos, prefix.length());
    }
  }
  return attr_value_str;
}

void ReadableDump::GenNodeAttrs(std::stringstream &readable_ss, const Node *node) {
  const auto op_desc = node->GetOpDesc();
  const auto op = OpDescUtils::CreateOperatorFromOpDesc(op_desc);
  const std::map<std::string, AnyValue> attr_map = op_desc->GetAllAttrs();
  const auto &ir_attr_names = op_desc->GetIrAttrNames();
  std::unordered_set<std::string> ir_attr_names_set(ir_attr_names.begin(), ir_attr_names.end());
  std::map<AscendString, AscendString> ir_names_to_type;
  if (op.GetAllIrAttrNamesAndTypes(ir_names_to_type) != GRAPH_SUCCESS) {
    GELOGW("[ReadableDump][GenNodeAttrs] failed, get node %s attr names to type failed",
           node->GetName().c_str());
    return;
  }

  std::vector<std::pair<std::string, AnyValue>> ir_attrs_vec{};
  for (const auto &attr_name : ir_attr_names) {
    auto attr_pair = attr_map.find(attr_name);
    if (attr_pair != attr_map.end()) {  // only handle IR attrs
      ir_attrs_vec.emplace_back(*attr_pair);
    }
  }

  if (ir_attrs_vec.empty()) {
    return;
  }

  std::stringstream attr_contents;
  bool first = true;
  for (const auto &attr_pair : ir_attrs_vec) {
    AscendString attr_asc_name(attr_pair.first.c_str());
    auto attr_value_str =
        GetAttrValueStr(op_desc, attr_pair.first, attr_pair.second, ir_names_to_type.at(attr_asc_name).GetString());
    if (attr_value_str.empty() || attr_value_str == "\"\"") {
      continue;
    }
    if (first) {
      first = false;
    } else {
      attr_contents << ", ";
    }
    attr_contents << attr_pair.first << ": ";
    attr_contents << attr_value_str;
  }

  if (node->GetInDataNodesSize() != 0 && !attr_contents.str().empty()) {
    readable_ss << ", ";
  }

  if (!attr_contents.str().empty()) {
    readable_ss << "attrs = {";
    readable_ss << attr_contents.str();
    readable_ss << "}";
  }
}

std::string ReadableDump::GetOutputOutDegree(const Node *node, int32_t index) {
  GE_ASSERT_NOTNULL(node);
  GE_ASSERT_TRUE(index >= 0 && index < static_cast<int32_t>(node->GetAllOutDataAnchorsPtr().size()));
  auto output_pair = node->GetAllOutDataAnchorsPtr().at(index);
  return "[#users=" + std::to_string(output_pair->GetPeerInDataNodesSize()) + "]";
}

void ReadableDump::GenMultipleOutputsIfNeeded(std::stringstream &readable_ss, const Node *node,
                                              OutputHandler &output_handler) {
  auto out_anchors = node->GetAllOutDataAnchorsPtr();
  if (out_anchors.size() <= 1) {
    return;
  }

  auto node_name = node->GetName();
  auto node_outputs = output_handler.GetNodeToOutputsMap().at(node_name);
  for (uint32_t i = 0; i < node_outputs.get()->size(); ++i) {
    readable_ss << GetInstanceName(node_outputs.get()->at(i)) << " : ";
    readable_ss << GetOutputOutDegree(node, static_cast<int32_t>(i)) << " = ";
    readable_ss << "get_element[node=%" << node_name << "]";
    readable_ss << "(" << i << ")" << std::endl;
  }
}

void ReadableDump::GenGraphOutput(std::stringstream &graph_output_ss, const Node *net_output,
                                  OutputHandler &output_handler) {
  if (net_output->GetAllInDataAnchorsSize() == 0) {
    return;
  }
  graph_output_ss << "return (";
  graph_output_ss << GetNodeInputInstance(net_output, output_handler);
  graph_output_ss << ")";
}
}  // namespace ge
