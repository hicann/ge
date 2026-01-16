/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <algorithm>
#include <cstring>
#include <set>
#include <vector>

#include "graph/debug/ge_op_types.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/anchor_utils.h"
#include "graph/utils/readable_dump.h"
#include "graph/utils/tensor_adapter.h"
#include "graph/utils/default_attr_utils.h"
#include "graph/utils/tensor_value_utils.h"

namespace ge {
namespace {
constexpr size_t kForNodeInputStartIndex = 1U;
constexpr int32_t kMaxRecursionDepth = 10;
}  // namespace

Status ReadableDump::GenReadableDump(std::stringstream &readable_ss, const ComputeGraphPtr &graph) {
  DumpContext ctx;
  return GenReadableDump(readable_ss, graph, ctx);
}

Status ReadableDump::GenReadableDump(std::stringstream &readable_ss, const ComputeGraphPtr &graph,
                                     DumpContext &ctx, int32_t recursion_depth) {
  if (recursion_depth > kMaxRecursionDepth) {
    REPORT_INNER_ERR_MSG("E18888", "recursion_depth:%d is bigger than kMaxRecursionDepth:%d", recursion_depth,
                     kMaxRecursionDepth);
    GELOGE(GRAPH_FAILED, "[ReadableDump][GenReadableDump] recursion depth is too large, abort");
    return GRAPH_FAILED;
  }

  GE_ASSERT_NOTNULL(graph);
  const auto graph_name = graph->GetName();
  OutputHandler output_handler;
  output_handler.GenNodeToOutputsMap(graph);
  readable_ss << "graph(\"" << graph_name << "\"):" << std::endl;
  std::stringstream graph_output_ss("");
  std::vector<ComputeGraphPtr> subgraphs_to_dump;
  for (const auto &node : graph->GetDirectNode()) {
    GE_ASSERT_NOTNULL(node);
    GE_ASSERT_NOTNULL(node->GetOpDesc());
    if (node->GetOpDesc()->GetType() != kNetOutput) {
      GenNodeDump(readable_ss, output_handler, node.get(), subgraphs_to_dump, ctx);
    } else {
      GenGraphOutput(graph_output_ss, node.get(), output_handler);
    }
  }
  if (!graph_output_ss.str().empty()) {
    readable_ss << std::endl;
    readable_ss << kIndentTwo << graph_output_ss.str();
    readable_ss << std::endl;
  }

  for (const auto &subgraph : subgraphs_to_dump) {
    readable_ss << std::endl;
    GE_ASSERT_SUCCESS(GenReadableDump(readable_ss, subgraph, ctx, recursion_depth + 1));
  }

  return SUCCESS;
}

void ReadableDump::GenNodeDump(std::stringstream &readable_ss, OutputHandler &output_handler, const Node *node,
                               std::vector<ComputeGraphPtr> &subgraphs_to_dump, DumpContext &ctx) {
  readable_ss << GetInstanceName(node->GetName()) << " : ";
  readable_ss << GetNodeOutDegree(node) << " = ";
  readable_ss << GetNodeType(node);
  readable_ss << " (";
  GenNodeInputs(readable_ss, node, output_handler);
  GenNodeAttrs(readable_ss, node, subgraphs_to_dump, ctx);
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

std::string ReadableDump::GetInputInstanceName(const Node *node, size_t input_index, OutputHandler &output_handler) {
  GE_ASSERT_NOTNULL(node);
  auto in_data_anchors = node->GetAllInDataAnchorsPtr();
  GE_ASSERT_TRUE(input_index < in_data_anchors.size());
  if (in_data_anchors.at(input_index)->GetPeerAnchorsSize() == 0U) {
    return "";
  }

  auto input_node_anchor_pair = NodeUtils::GetInDataNodeAndAnchorByIndex(*node, static_cast<int32_t>(input_index));
  if (input_node_anchor_pair.first == nullptr || input_node_anchor_pair.second == nullptr) {
    return "";
  }

  auto input_node_name = input_node_anchor_pair.first->GetName();
  auto input_node_index = AnchorUtils::GetIdx(input_node_anchor_pair.second);
  const auto &node_to_outputs_map = output_handler.GetNodeToOutputsMap();
  const auto it = node_to_outputs_map.find(input_node_name);
  if (it == node_to_outputs_map.end()) {
    return "";
  }
  auto input_name_vec = it->second;
  GE_ASSERT_TRUE(input_node_index < static_cast<int32_t>(input_name_vec->size()));
  return input_name_vec->at(input_node_index);
}

void ReadableDump::AppendInputInstance(std::stringstream &ss, bool &first, const std::string &param_name,
                                       const std::string &instance_name) {
  if (instance_name.empty()) {
    return;
  }

  if (first) {
    first = false;
  } else {
    ss << ", ";
  }
  ss << param_name << "=" << GetInstanceName(instance_name, kIndentZero);
}

std::string ReadableDump::GetNodeInputInstance(const Node *node, OutputHandler &output_handler) {
  GE_ASSERT_NOTNULL(node);
  const auto op_desc = node->GetOpDesc();
  GE_ASSERT_NOTNULL(op_desc);

  const auto &ir_inputs = op_desc->GetIrInputs();
  std::map<size_t, std::pair<size_t, size_t>> ir_input_to_range;
  GE_ASSERT_GRAPH_SUCCESS(ge::OpDescUtils::GetIrInputRawDescRange(op_desc, ir_input_to_range));

  std::stringstream input_instance_ss;
  bool first = true;
  for (size_t ir_input_index = 0U; ir_input_index < ir_inputs.size(); ++ir_input_index) {
    const auto &ir_input = ir_inputs[ir_input_index];
    const auto &ir_input_name = ir_input.first;
    const auto &ir_input_type = ir_input.second;

    const auto &ir_input_to_range_iter = ir_input_to_range.find(ir_input_index);
    GE_ASSERT_TRUE(ir_input_to_range_iter != ir_input_to_range.end());
    const size_t start_index = ir_input_to_range_iter->second.first;
    const size_t count = ir_input_to_range_iter->second.second;

    if (ir_input_type == ge::IrInputType::kIrInputRequired) {
      GE_ASSERT_EQ(count, 1U);
      GE_ASSERT_TRUE(start_index < node->GetAllInDataAnchorsSize());
      auto instance_name = GetInputInstanceName(node, start_index, output_handler);
      AppendInputInstance(input_instance_ss, first, ir_input_name, instance_name);
    } else if (ir_input_type == ge::IrInputType::kIrInputDynamic) {
      GE_ASSERT_TRUE(count > 0U);
      size_t dump_start_index = 0U;
      if (node->GetType() == FOR) {
        dump_start_index = kForNodeInputStartIndex;  // FOR input_0 为start，因此动态输入 index 从 1 开始，其他从 0 开始
      }

      for (size_t dy_i = 0U; dy_i < count; ++dy_i) {
        const size_t actual_input_index = start_index + dy_i;
        auto instance_name = GetInputInstanceName(node, actual_input_index, output_handler);
        std::string suffix = "_" + std::to_string(dump_start_index + dy_i);
        AppendInputInstance(input_instance_ss, first, ir_input_name + suffix, instance_name);
      }
    } else {
      GE_ASSERT_TRUE(ir_input_type == ge::IrInputType::kIrInputOptional);
      if (count == 0U) {
        continue;
      }
      GE_ASSERT_EQ(1U, count);
      auto instance_name = GetInputInstanceName(node, start_index, output_handler);
      AppendInputInstance(input_instance_ss, first, ir_input_name, instance_name);
    }
  }

  return input_instance_ss.str();
}

void ReadableDump::GenNodeInputs(std::stringstream &readable_ss, const Node *node, OutputHandler &output_handler) {
  if (node->GetInDataNodesSize() == 0U) {
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

std::unordered_map<std::string, std::string> ReadableDump::CollectSubgraphsAndBuildIrToInstanceMap(
    const Node* node, std::vector<ComputeGraphPtr> &subgraphs_to_dump, DumpContext &ctx) {
  GE_ASSERT_NOTNULL(node);
  const auto op_desc = node->GetOpDesc();
  GE_ASSERT_NOTNULL(op_desc);
  std::unordered_map<std::string, std::string> ir_name_to_instance;
  const auto &subgraph_instance_names = op_desc->GetSubgraphInstanceNames();
  if (subgraph_instance_names.empty()) {
    return ir_name_to_instance;
  }
  const auto root_graph = GraphUtils::FindRootGraph(node->GetOwnerComputeGraph());
  if (root_graph == nullptr) {
    GELOGW("[ReadableDump][GenNodeGraphAttrs] can not find the root graph, node: %s.", node->GetName().c_str());
    return ir_name_to_instance;
  }

  for (const auto &instance_name : subgraph_instance_names) {
    if (instance_name.empty()) {
      continue;
    }

    if (ctx.visited_subgraph_instances.insert(instance_name).second) {
      const auto subgraph = root_graph->GetSubgraph(instance_name);
      if (subgraph != nullptr) {
        subgraphs_to_dump.emplace_back(subgraph);
      }
    }

    std::string ir_name;
    GE_ASSERT_SUCCESS(op_desc->GetSubgraphNameByInstanceName(instance_name, ir_name));
    ir_name_to_instance.emplace(ir_name, instance_name);
  }

  return ir_name_to_instance;
}

std::vector<std::string> ReadableDump::GetSortedSubgraphIrNames(const Node *node) {
  GE_ASSERT_NOTNULL(node);
  const auto op_desc = node->GetOpDesc();
  GE_ASSERT_NOTNULL(op_desc);
  const auto &ir_names_to_index = op_desc->GetSubgraphNameIndexes();
  std::vector<std::pair<std::string, uint32_t>> sorted_ir_name_pairs;
  sorted_ir_name_pairs.reserve(ir_names_to_index.size());

  for (const auto &ir_name_pair : ir_names_to_index) {
    sorted_ir_name_pairs.emplace_back(ir_name_pair);
  }
  std::sort(sorted_ir_name_pairs.begin(), sorted_ir_name_pairs.end(),
            [](const std::pair<std::string, uint32_t> &a, const std::pair<std::string, uint32_t> &b) {
              return a.second < b.second;
            });

  std::vector<std::string> sorted_ir_names;
  sorted_ir_names.reserve(sorted_ir_name_pairs.size());
  for (const auto &ir_name_pair : sorted_ir_name_pairs) {
    sorted_ir_names.emplace_back(ir_name_pair.first);
  }
  return sorted_ir_names;
}

void ReadableDump::AppendSubgraphAttrs(std::stringstream &attr_contents, const Node *node,
                                       std::vector<ComputeGraphPtr> &subgraphs_to_dump, DumpContext &ctx) {
  auto ir_name_to_instance = CollectSubgraphsAndBuildIrToInstanceMap(node, subgraphs_to_dump, ctx);
  if (ir_name_to_instance.empty()) {
    return;
  }

  bool first = true;
  auto sorted_subgraph_ir_names = GetSortedSubgraphIrNames(node);
  for (const auto &ir_name : sorted_subgraph_ir_names) {
    const auto it = ir_name_to_instance.find(ir_name);
    if (it == ir_name_to_instance.end()) {
      continue;
    }
    const auto &instance_name = it->second;
    if (instance_name.empty()) {
      continue;
    }

    if (first) {
      first = false;
    } else {
      attr_contents << ", ";
    }
    attr_contents << ir_name << ": " << GetInstanceName(instance_name, kIndentZero);
  }
}

void ReadableDump::AppendNodeAttrs(std::stringstream &attr_contents, const Node *node) {
  const auto op_desc = node->GetOpDesc();
  const auto op = OpDescUtils::CreateOperatorFromOpDesc(op_desc);
  const std::map<std::string, AnyValue> attr_map = op_desc->GetAllAttrs();
  const auto &ir_attr_names = op_desc->GetIrAttrNames();
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

  bool first = attr_contents.str().empty();
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
}

void ReadableDump::GenNodeAttrs(std::stringstream &readable_ss, const Node *node,
                                std::vector<ComputeGraphPtr> &subgraphs_to_dump, DumpContext &ctx) {
  std::stringstream attr_contents;
  AppendSubgraphAttrs(attr_contents, node, subgraphs_to_dump, ctx);
  AppendNodeAttrs(attr_contents, node);
  std::string attr_str = attr_contents.str();
  if (node->GetInDataNodesSize() != 0 && !attr_str.empty()) {
    readable_ss << ", ";
  }

  if (!attr_str.empty()) {
    readable_ss << "attrs = {" << attr_str << "}";
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

std::string ReadableDump::GetGraphOutputInstance(const Node *net_output, OutputHandler &output_handler) {
  std::stringstream input_instance_ss;
  bool first = true;
  int32_t retval_index = 0;
  const size_t output_num = net_output->GetInDataNodesSize();
  for (size_t i = 0U; i < net_output->GetAllInDataAnchorsSize(); ++i) {
    auto input_instance_name = GetInputInstanceName(net_output, i, output_handler);
    if (input_instance_name.empty()) {
      continue;
    }

    if (first) {
      first = false;
    } else {
      input_instance_ss << ", ";
    }

    // 输出个数为1，直接返回实例名；否则使用 output_index= 格式
    if (output_num > 1U) {
      input_instance_ss << "output_" << retval_index << "=" ;
    }
    input_instance_ss << GetInstanceName(input_instance_name, kIndentZero);
    retval_index++;
  }
  return input_instance_ss.str();
}

void ReadableDump::GenGraphOutput(std::stringstream &graph_output_ss, const Node *net_output,
                                  OutputHandler &output_handler) {
  if (net_output->GetInDataNodesSize() == 0) {
    return;
  }

  graph_output_ss << "return (";
  graph_output_ss << GetGraphOutputInstance(net_output, output_handler);
  graph_output_ss << ")";
}
}  // namespace ge
