/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "norm_like_reduce_checker.h"
#include "ascgen_log.h"
#include "graph/symbolizer/symbolic_utils.h"

namespace optimize {
namespace {

static constexpr size_t kMaxFullLoadAxisSize = 3UL;

static bool CalculateRAxisTotalSize(const ge::AscTensorAttr &input_attr,
                                     const ge::AscTensorAttr &output_attr,
                                     int64_t &r_axis_total_size,
                                     int64_t &a_axis_total_size) {
  r_axis_total_size = 1;
  a_axis_total_size = 1;
  if (output_attr.repeats.empty() || output_attr.repeats.size() > kMaxFullLoadAxisSize) {
    GELOGD("Output repeats size %zu exceeds max full load axis size %zu",
           output_attr.repeats.size(), kMaxFullLoadAxisSize);
    return false;
  }
  if (ge::SymbolicUtils::StaticCheckEq(input_attr.repeats[0], output_attr.repeats[0]) != ge::TriBool::kTrue) {
    GELOGD("First axis of input and output not equal, not AR/ARA pattern");
    return false;
  }

  for (size_t i = 0; i < input_attr.repeats.size(); ++i) {
    if (!input_attr.repeats[i].IsConstExpr() || !output_attr.repeats[i].IsConstExpr()) {
      return false;
    }

    int64_t input_size = 0;
    int64_t output_size = 0;
    if (!input_attr.repeats[i].GetConstValue(input_size) || !output_attr.repeats[i].GetConstValue(output_size)) {
      return false;
    }

    if (input_size > output_size) {
      r_axis_total_size *= input_size;
    } else {
      a_axis_total_size *= output_size;
    }
  }

  return true;
}

static bool IsStaticShape(const ge::AscNodePtr &node) {
  if (node == nullptr || node->outputs().empty()) {
    return false;
  }
  for (const auto &node_out : node->outputs()) {
    if (node_out->attr.repeats.empty()) {
      return false;
    }
    for (const auto &repeat : node_out->attr.repeats) {
      if (!repeat.IsConstExpr()) {
        return false;
      }
    }
  }
  return true;
}

static bool IsStaticGraph(const ge::AscGraph &graph) {
  for (const auto &node : graph.GetAllNodes()) {
    if (!NormLikeReduceChecker::IsLoad(node)) {
      continue;
    }
    if (!IsStaticShape(node)) {
      return false;
    }
  }
  GELOGD("Graph[%s] is static", graph.GetName().c_str());
  return true;
}

static bool CheckReduceNodeNormLike(const ge::AscNodePtr &asc_node) {
  constexpr int64_t kThresholdTR = 32;
  constexpr int64_t kThresholdTA = 128;

  GELOGD("Found reduce node: %s", asc_node->GetNamePtr());

  if (asc_node->inputs().empty() || asc_node->outputs().empty()) {
    GELOGW("Reduce node %s has no inputs or outputs", asc_node->GetNamePtr());
    return false;
  }

  const ge::AscTensorAttr *input_attr_ptr = nullptr;

  ge::OutDataAnchorPtr current_out_anchor = nullptr;
  if (asc_node->GetAllInDataAnchorsSize() > 0 && asc_node->GetInDataAnchor(0) != nullptr) {
    current_out_anchor = asc_node->GetInDataAnchor(0)->GetPeerOutAnchor();
  }

  int traverse_depth = 0;
  const int max_traverse_depth = 100;

  while (current_out_anchor != nullptr && traverse_depth++ < max_traverse_depth) {
    auto current_tensor_attr = ge::AscTensorAttr::GetTensorAttrPtr(*current_out_anchor);
    auto current_node = std::dynamic_pointer_cast<ge::AscNode>(current_out_anchor->GetOwnerNode());

    bool is_load = (current_node != nullptr && NormLikeReduceChecker::IsLoad(current_node));
    bool has_valid_attr = (current_tensor_attr != nullptr && !current_tensor_attr->repeats.empty());
    if (has_valid_attr || is_load) {
      if (has_valid_attr) {
        input_attr_ptr = current_tensor_attr;
      }
      break;
    }

    if (current_node == nullptr || current_node->GetAllInDataAnchorsSize() <= 0 ||
        current_node->GetInDataAnchor(0) == nullptr) {
      return false;
    }

    auto next_in_anchor = current_node->GetInDataAnchor(0);
    current_out_anchor = next_in_anchor->GetPeerOutAnchor();
  }

  const auto &output = asc_node->outputs[0];
  if (input_attr_ptr == nullptr || input_attr_ptr->repeats.empty() || input_attr_ptr->repeats.size() != output.attr.repeats.size()) {
    return false;
  }

  int64_t r_axis_total_size = 1;
  int64_t a_axis_total_size = 1;
  if (!CalculateRAxisTotalSize(*input_attr_ptr, output.attr, r_axis_total_size, a_axis_total_size)) {
    GELOGD("Reduce node %s: failed to calculate R/A axis size (non-const shape)", asc_node->GetNamePtr());
    return false;
  }

  if (r_axis_total_size > kThresholdTR || a_axis_total_size < kThresholdTA) {
    return false;
  }

  GELOGI("Reduce node %s passed check: R_axis=%ld (threshold=%ld), A_axis=%ld (threshold=%ld)",
         asc_node->GetNamePtr(), r_axis_total_size, kThresholdTR, a_axis_total_size, kThresholdTA);
  return true;
}

}  // namespace

bool NormLikeReduceChecker::IsNormLikeReduceGraph(const ge::AscGraph &graph) {
  if (!IsStaticGraph(graph)) {
    GELOGI("AscGraph is not static shape, return false for IsNormLikeReduceGraph");
    return false;
  }

  bool has_reduce = false;
  for (const auto &asc_node : graph.GetAllNodes()) {
    if (!IsReduce(asc_node)) {
      continue;
    }

    has_reduce = true;
    if (!CheckReduceNodeNormLike(asc_node)) {
      return false;
    }
  }

  if (!has_reduce) {
    GELOGD("AscGraph has no reduce node, return false for IsNormLikeReduceGraph");
    return false;
  }

  GELOGD("AscGraph passed IsNormLikeReduceGraph check");
  return true;
}

}  // namespace optimize