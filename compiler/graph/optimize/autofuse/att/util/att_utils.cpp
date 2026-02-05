/**
* Copyright (c) 2025 Huawei Technologies Co., Ltd.
* This program is free software, you can redistribute it and/or modify it under the terms and conditions of
* CANN Open Software License Agreement Version 2.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
 */
#include "att_utils.h"
namespace att {
bool AttUtils::IsLoadNode(ge::AscNode *node) {
  GE_ASSERT_NOTNULL(node);
  const auto input_size = node->inputs().size();
  std::vector<size_t> indices(input_size);
  std::iota(indices.begin(), indices.end(), 0U);
  bool is_any_input_gm = std::any_of(indices.begin(), indices.end(), [&node](size_t id) {
    const auto &input = node->inputs[id];
    return (input.attr.mem.hardware == ge::MemHardware::kMemHardwareGM);
  });
  return is_any_input_gm;
}

bool AttUtils::IsStoreNode(ge::AscNode *node) {
  GE_ASSERT_NOTNULL(node);
  const auto output_size = node->outputs().size();
  std::vector<size_t> indices(output_size);
  std::iota(indices.begin(), indices.end(), 0U);
  bool is_any_output_gm = std::any_of(indices.begin(), indices.end(), [&node](size_t id) {
    const auto &output = node->outputs[id];
    return (output.attr.mem.hardware == ge::MemHardware::kMemHardwareGM);
  });
  return is_any_output_gm;
}

bool AttUtils::IsLoadStoreNode(ge::AscNode *node) {
  return IsLoadNode(node) || IsStoreNode(node);
}

bool AttUtils::IsTileSplitAxis(const AttAxisPtr &axis) {
  return (axis->axis_pos == AxisPosition::INNER) && (!axis->bind_multicore);
}

bool AttUtils::GetLastTileSplitAxisName(ge::AscNode &node, const ge::AscGraph &graph, std::string &axis_name) {
  if (node.outputs().empty()) {
    return false;
  }
  const auto &node_attr = node.outputs[0].attr;
  if (node_attr.axis.empty()) {
    return false;
  }
  const auto &last_axis_id = node_attr.axis.back();
  for (const auto &axis : graph.GetAllAxis()) {
    if (axis->id == last_axis_id) {
      axis_name = axis->name;
      return true;
    }
  }
  return false;
}
}  // namespace att