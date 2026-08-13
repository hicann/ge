/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "common/helper/om2/rt_var_resource_builder.h"

#include <unordered_set>

#include "graph/manager/graph_var_manager.h"
#include "graph/compute_graph.h"
#include "graph/node.h"
#include "graph/utils/attr_utils.h"
#include "graph/utils/tensor_utils.h"
#include "graph/debug/ge_attr_define.h"
#include "common/const_place_holder_utils/const_place_holder_utils.h"
#include "common/om2/codegen/om2_codegen_types.h"
#include "common/ge_inner_error_codes.h"
#include "common/ge_common/debug/ge_log.h"

namespace gert {
namespace {

ge::Om2TensorDesc ConvertTensorDesc(const ge::GeTensorDesc &ge_desc) {
  ge::Om2TensorDesc desc;
  desc.SetFormat(ge_desc.GetFormat());
  desc.SetDataType(ge_desc.GetDataType());
  desc.SetShape(ge_desc.GetShape().GetDims());
  desc.SetName(ge_desc.GetName());
  std::vector<std::pair<int64_t, int64_t>> shape_range;
  if (ge_desc.GetShapeRange(shape_range) == ge::GRAPH_SUCCESS) {
    desc.SetShapeRange(shape_range);
  }
  int64_t size = 0;
  if (ge::TensorUtils::GetTensorSizeInBytes(ge_desc, size) == ge::GRAPH_SUCCESS) {
    desc.SetSize(static_cast<size_t>(size));
  }
  return desc;
}

RTVarTransRoad ConvertTransRoad(const ge::VarTransRoad &road) {
  RTVarTransRoad rt_road;
  rt_road.reserve(road.size());
  for (const auto &node : road) {
    RTTransNodeInfo info;
    info.node_type = node.node_type;
    info.input = ConvertTensorDesc(node.input);
    info.output = ConvertTensorDesc(node.output);
    rt_road.push_back(std::move(info));
  }
  return rt_road;
}

std::vector<uint8_t> ExtractInitDataFromWeights(const ge::OpDescPtr &op_desc) {
  ge::ConstGeTensorPtr weight = nullptr;
  if (!ge::AttrUtils::GetTensor(*op_desc, ge::ATTR_NAME_WEIGHTS, weight)) {
    return {};
  }
  if (weight == nullptr) {
    return {};
  }
  const auto &data = weight->GetData();
  if (data.size() == 0U) {
    return {};
  }
  return std::vector<uint8_t>(data.data(), data.data() + data.size());
}

std::vector<uint8_t> ExtractInitValueFromTensorDesc(const ge::GeTensorDesc &ge_desc) {
  ge::ConstGeTensorPtr init_value = nullptr;
  if (!ge::AttrUtils::GetTensor(ge_desc, ge::ATTR_NAME_INIT_VALUE, init_value)) {
    return {};
  }
  if (init_value == nullptr) {
    return {};
  }
  const auto &data = init_value->GetData();
  if (data.size() == 0U) {
    return {};
  }
  return std::vector<uint8_t>(data.data(), data.data() + data.size());
}

ge::Status FillCopyInfo(const ge::ComputeGraphPtr &compute_graph, const ge::OpDescPtr &op_desc, RTVarEntry &entry) {
  const auto *copy_from = ge::AttrUtils::GetStr(*op_desc, "_copy_from_var_node");
  if (copy_from == nullptr || copy_from->empty()) {
    return ge::SUCCESS;
  }
  entry.copy_info.src_var_name = *copy_from;
  const auto src_node = compute_graph->FindNode(*copy_from);
  if (src_node == nullptr) {
    return ge::SUCCESS;
  }
  const auto src_op_desc = src_node->GetOpDesc();
  if (src_op_desc == nullptr) {
    return ge::SUCCESS;
  }
  const auto src_output_desc = src_op_desc->MutableOutputDesc(0U);
  if (src_output_desc != nullptr) {
    entry.copy_info.src_tensor_desc = ConvertTensorDesc(*src_output_desc);
  }
  return ge::SUCCESS;
}

ge::Status FillTypeSpecificData(const ge::ComputeGraphPtr &compute_graph, const ge::OpDescPtr &op_desc,
                                const ge::GeTensorDesc &cur_desc, RTVarEntry &entry) {
  entry.op_type = op_desc->GetType();
  if (entry.op_type == ge::CONSTPLACEHOLDER) {
    uint8_t *dev_addr = nullptr;
    GE_ASSERT_SUCCESS(ge::GetConstPlaceHolderAddr(op_desc, dev_addr));
    entry.extern_dev_addr = dev_addr;
  } else if (entry.op_type == "Constant" || entry.op_type == "ConstantOp") {
    entry.init_data = ExtractInitDataFromWeights(op_desc);
  } else if (entry.op_type == "Variable") {
    entry.init_data = ExtractInitValueFromTensorDesc(cur_desc);
    return FillCopyInfo(compute_graph, op_desc, entry);
  }
  return ge::SUCCESS;
}

ge::Status BuildSingleEntry(ge::VarManager &var_manager, const ge::ComputeGraphPtr &compute_graph,
                            const std::string &var_name, const ge::GeTensorDesc &cur_desc, RTVarEntry &entry) {
  const auto om2_desc = ConvertTensorDesc(cur_desc);
  const auto var_key = RTVarResource::BuildVarKey(var_name, om2_desc);

  entry.var_name = var_name;
  entry.var_key = var_key;
  entry.tensor_desc = om2_desc;
  entry.size = om2_desc.GetSize();

  uint8_t *dev_ptr = nullptr;
  rtMemType_t memory_type = RT_MEMORY_HBM;
  if (var_manager.GetVarAddr(var_name, cur_desc, dev_ptr, memory_type) == ge::SUCCESS) {
    entry.logic_addr = reinterpret_cast<uint64_t>(dev_ptr);
    entry.memory_type = static_cast<uint32_t>(memory_type);
  }

  ge::OpDescPtr op_desc = nullptr;
  if (compute_graph != nullptr) {
    const auto node = compute_graph->FindNode(var_name);
    if (node != nullptr) {
      op_desc = node->GetOpDesc();
    }
  }
  if (op_desc != nullptr) {
    GE_ASSERT_SUCCESS(FillTypeSpecificData(compute_graph, op_desc, cur_desc, entry));
  }

  const auto *trans_road = var_manager.GetTransRoad(var_name);
  if (trans_road != nullptr && !trans_road->empty()) {
    entry.trans_road = ConvertTransRoad(*trans_road);
  }

  uint32_t graph_id = 0U;
  if (var_manager.GetChangedGraphId(var_name, graph_id) == ge::SUCCESS) {
    entry.changed_graph_id = graph_id;
  }
  if (var_manager.GetAllocatedGraphId(var_name, graph_id) == ge::SUCCESS) {
    entry.allocated_graph_id = graph_id;
  }

  return ge::SUCCESS;
}

}  // namespace

ge::Status BuildRTVarResource(ge::VarManager &var_manager, const ge::ComputeGraphPtr &compute_graph,
                              const std::vector<ge::Om2VarMeta> &var_metas, std::unique_ptr<RTVarResource> &resource) {
  resource = std::make_unique<RTVarResource>();
  if (var_metas.empty()) {
    return ge::SUCCESS;
  }

  std::unordered_set<std::string> processed_names;
  std::vector<std::string> pending_var_names;
  pending_var_names.reserve(var_metas.size());
  for (const auto &meta : var_metas) {
    pending_var_names.push_back(meta.var_name);
  }

  while (!pending_var_names.empty()) {
    const auto current_var_name = pending_var_names.back();
    pending_var_names.pop_back();
    if (!processed_names.insert(current_var_name).second) {
      continue;
    }

    ge::GeTensorDesc cur_desc;
    GE_ASSERT_SUCCESS(var_manager.GetCurVarDesc(current_var_name, cur_desc));

    RTVarEntry entry;
    GE_ASSERT_SUCCESS(BuildSingleEntry(var_manager, compute_graph, current_var_name, cur_desc, entry));

    if (!entry.copy_info.src_var_name.empty()) {
      pending_var_names.push_back(entry.copy_info.src_var_name);
    }

    const auto add_ret = resource->AddEntry(std::move(entry));
    if (add_ret != ge::SUCCESS) {
      GELOGW("[OM2][Var] AddEntry failed for var=%s.", current_var_name.c_str());
    }
  }

  GELOGI("[OM2][Var] BuildRTVarResource completed, %zu entries from %zu var_metas.", resource->GetAllEntries().size(),
         var_metas.size());
  return ge::SUCCESS;
}

}  // namespace gert
