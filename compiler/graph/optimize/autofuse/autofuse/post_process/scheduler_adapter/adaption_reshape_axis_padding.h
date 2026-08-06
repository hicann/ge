/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AUTOFUSE_POST_PROCESS_SCHEDULER_ADAPTER_ADAPTION_RESHAPE_AXIS_PADDING_H
#define AUTOFUSE_POST_PROCESS_SCHEDULER_ADAPTER_ADAPTION_RESHAPE_AXIS_PADDING_H
#include <algorithm>
#include <unordered_map>
#include <unordered_set>
#include "fusion/autofuse_attrs.h"
#include "fusion/fuse_type.h"
#include "graph/symbolizer/symbolic_utils.h"
#include "post_process/scheduler_adapter/adaption_complete_node_attrs.h"
#include "can_fuse/backend/asc_graph_axis_mapping.h"
#include "ascir_ops.h"

namespace ge {
namespace asc_adapt {
inline bool IsSameRepeat(const Expression &lhs, const Expression &rhs) {
  return SymbolicUtils::StaticCheckEq(lhs, rhs) == TriBool::kTrue;
}

inline bool IsUnitRepeat(const Expression &repeat) {
  return IsSameRepeat(repeat, kSymbolOne);
}

inline bool IsSameRepeats(const std::vector<Expression> &lhs, const std::vector<Expression> &rhs) {
  return (lhs.size() == rhs.size()) && std::equal(lhs.begin(), lhs.end(), rhs.begin(), IsSameRepeat);
}

inline std::vector<Expression> RemoveUnitRepeats(const std::vector<Expression> &repeats) {
  std::vector<Expression> non_unit_repeats;
  for (const auto &repeat : repeats) {
    if (!IsUnitRepeat(repeat)) {
      non_unit_repeats.push_back(repeat);
    }
  }
  return non_unit_repeats;
}

inline std::vector<size_t> BuildUnitRepeatGapCounts(const std::vector<Expression> &repeats) {
  const auto non_unit_repeats = RemoveUnitRepeats(repeats);
  std::vector<size_t> unit_gap_counts(non_unit_repeats.size() + 1U, 0U);
  size_t non_unit_idx = 0U;
  for (const auto &repeat : repeats) {
    if (IsUnitRepeat(repeat)) {
      ++unit_gap_counts[non_unit_idx];
    } else if (non_unit_idx < non_unit_repeats.size()) {
      ++non_unit_idx;
    }
  }
  return unit_gap_counts;
}

inline void MergeUnitRepeatGapCounts(const std::vector<size_t> &candidate_gaps, std::vector<size_t> &target_gaps) {
  if (target_gaps.empty()) {
    target_gaps = candidate_gaps;
    return;
  }
  if (target_gaps.size() != candidate_gaps.size()) {
    return;
  }
  for (size_t i = 0U; i < target_gaps.size(); ++i) {
    target_gaps[i] = std::max(target_gaps[i], candidate_gaps[i]);
  }
}

inline std::vector<Expression> ApplyUnitRepeatGaps(const std::vector<Expression> &non_unit_repeats,
                                                   const std::vector<size_t> &unit_gap_counts) {
  if (unit_gap_counts.size() != (non_unit_repeats.size() + 1U)) {
    return non_unit_repeats;
  }
  std::vector<Expression> target_repeats;
  for (size_t non_unit_idx = 0U; non_unit_idx < non_unit_repeats.size(); ++non_unit_idx) {
    target_repeats.insert(target_repeats.end(), unit_gap_counts[non_unit_idx], kSymbolOne);
    target_repeats.push_back(non_unit_repeats[non_unit_idx]);
  }
  target_repeats.insert(target_repeats.end(), unit_gap_counts.back(), kSymbolOne);
  return target_repeats;
}

inline void AppendUniqueTargetRepeats(const std::vector<Expression> &target_repeats,
                                      std::vector<std::vector<Expression>> &target_candidates) {
  if (target_repeats.empty()) {
    return;
  }
  const auto it = std::find_if(target_candidates.begin(), target_candidates.end(),
                               [&target_repeats](const auto &saved) { return IsSameRepeats(saved, target_repeats); });
  if (it == target_candidates.end()) {
    target_candidates.push_back(target_repeats);
  }
}

inline std::vector<Expression> BuildNoOpReshapeTargetRepeats(const af::ReshapeAxisChangeInfo &axis_change) {
  const auto before_non_unit_repeats = RemoveUnitRepeats(axis_change.before_repeats);
  const auto after_non_unit_repeats = RemoveUnitRepeats(axis_change.after_repeats);
  if (!IsSameRepeats(before_non_unit_repeats, after_non_unit_repeats)) {
    return axis_change.before_repeats.size() >= axis_change.after_repeats.size() ? axis_change.before_repeats
                                                                                 : axis_change.after_repeats;
  }

  std::vector<size_t> unit_gap_counts;
  MergeUnitRepeatGapCounts(BuildUnitRepeatGapCounts(axis_change.before_repeats), unit_gap_counts);
  MergeUnitRepeatGapCounts(BuildUnitRepeatGapCounts(axis_change.after_repeats), unit_gap_counts);
  return ApplyUnitRepeatGaps(before_non_unit_repeats, unit_gap_counts);
}

inline bool IsCompletedReshapeAxis(const AxisPtr &axis_info) {
  return (axis_info != nullptr) && (axis_info->name.rfind("reshape_axis_optimized_", 0U) == 0U);
}

inline std::string GetCompletedReshapeAxisName(const int64_t axis_id) {
  return "reshape_axis_optimized_" + std::to_string(axis_id);
}

inline void RefreshCompletedReshapeAxisName(const AxisPtr &axis_info) {
  if (IsCompletedReshapeAxis(axis_info)) {
    axis_info->name = GetCompletedReshapeAxisName(axis_info->id);
  }
}

inline AxisPtr MakeNoOpReshapeAxis(const int64_t axis_id, const Expression &repeat) {
  auto axis_info = ComGraphMakeShared<Axis>();
  GE_ASSERT_NOTNULL(axis_info);
  axis_info->id = axis_id;
  axis_info->name = GetCompletedReshapeAxisName(axis_id);
  axis_info->type = Axis::kAxisTypeOriginal;
  axis_info->size = repeat;
  return axis_info;
}

inline void ShiftAxisIdFrom(const int64_t insert_axis_id, int64_t &axis_id) {
  if (axis_id >= insert_axis_id) {
    ++axis_id;
  }
}

inline void ShiftAxisInfoIdFrom(const int64_t insert_axis_id, const AxisPtr &axis_info) {
  if (axis_info == nullptr) {
    return;
  }
  const auto old_axis_id = axis_info->id;
  ShiftAxisIdFrom(insert_axis_id, axis_info->id);
  if (axis_info->id != old_axis_id) {
    RefreshCompletedReshapeAxisName(axis_info);
  }
}

inline void ShiftAxisIdsFrom(const int64_t insert_axis_id, std::vector<int64_t> &axis) {
  for (auto &axis_id : axis) {
    ShiftAxisIdFrom(insert_axis_id, axis_id);
  }
}

inline void ShiftReshapeAxisChangesFrom(const int64_t insert_axis_id,
                                        std::vector<af::ReshapeAxisChangeInfo> &reshape_axis_changes) {
  for (auto &change : reshape_axis_changes) {
    ShiftAxisIdsFrom(insert_axis_id, change.before_axis);
    ShiftAxisIdsFrom(insert_axis_id, change.after_axis);
  }
}

inline Status ShiftAscGraphAxisIdsFrom(const AscGraph &asc_graph, const int64_t insert_axis_id) {
  for (const auto &node : AscGraphUtils::GetComputeGraph(asc_graph)->GetAllNodes()) {
    const auto &op_desc = node->GetOpDesc();
    GE_ASSERT_NOTNULL(op_desc);
    auto node_attr = op_desc->GetAttrsGroup<AscNodeAttr>();
    if (node_attr != nullptr) {
      ShiftAxisIdsFrom(insert_axis_id, node_attr->sched.axis);
      ShiftAxisIdFrom(insert_axis_id, node_attr->sched.loop_axis);
    }
    for (size_t i = 0U; i < node->GetAllInDataAnchorsSize(); ++i) {
      const auto input_tensor_desc = op_desc->MutableInputDesc(i);
      GE_ASSERT_NOTNULL(input_tensor_desc);
      auto tensor_attr = input_tensor_desc->GetAttrsGroup<AscTensorAttr>();
      if (tensor_attr == nullptr) {
        continue;
      }
      ShiftAxisIdsFrom(insert_axis_id, tensor_attr->axis);
    }
    for (size_t i = 0U; i < node->GetAllOutDataAnchorsSize(); ++i) {
      const auto output_tensor_desc = op_desc->MutableOutputDesc(i);
      GE_ASSERT_NOTNULL(output_tensor_desc);
      auto tensor_attr = output_tensor_desc->GetAttrsGroup<AscTensorAttr>();
      if (tensor_attr == nullptr) {
        continue;
      }
      ShiftAxisIdsFrom(insert_axis_id, tensor_attr->axis);
    }
  }
  return SUCCESS;
}

template <typename ScoreFunc>
inline std::vector<size_t> BuildBestOverlapTargetIndexes(const size_t source_size, const size_t target_size,
                                                         const ScoreFunc &score_func) {
  if ((source_size == 0U) || (source_size > target_size)) {
    return {};
  }
  constexpr int32_t kInvalidScore = -1000000;
  std::vector<std::vector<int32_t>> dp(source_size + 1U, std::vector<int32_t>(target_size + 1U, kInvalidScore));
  for (size_t target_idx = 0U; target_idx <= target_size; ++target_idx) {
    dp[source_size][target_idx] = 0;
  }
  for (size_t source_idx = source_size; source_idx > 0U; --source_idx) {
    for (size_t target_idx = target_size; target_idx > 0U; --target_idx) {
      const auto source_pos = source_idx - 1U;
      const auto target_pos = target_idx - 1U;
      if ((source_size - source_pos) > (target_size - target_pos)) {
        continue;
      }
      const auto match_score = score_func(source_pos, target_pos) + dp[source_pos + 1U][target_pos + 1U];
      const auto skip_score = dp[source_pos][target_pos + 1U];
      dp[source_pos][target_pos] = std::max(match_score, skip_score);
    }
  }

  std::vector<size_t> target_indexes(source_size, target_size);
  size_t source_idx = 0U;
  size_t target_idx = 0U;
  while ((source_idx < source_size) && (target_idx < target_size)) {
    const auto match_score = score_func(source_idx, target_idx) + dp[source_idx + 1U][target_idx + 1U];
    const auto skip_score = dp[source_idx][target_idx + 1U];
    if (((target_size - target_idx - 1U) >= (source_size - source_idx)) && (skip_score >= match_score)) {
      ++target_idx;
      continue;
    }
    target_indexes[source_idx++] = target_idx++;
  }
  return target_indexes;
}

inline int64_t GetNextNoOpReshapeAxisId(const std::vector<AxisPtr> &graph_axis) {
  int64_t max_axis_id = -1;
  for (const auto &axis_info : graph_axis) {
    if (axis_info != nullptr) {
      max_axis_id = std::max(max_axis_id, axis_info->id);
    }
  }
  return max_axis_id + 1;
}

inline int64_t GetNextNoOpReshapeAxisId(const std::vector<int64_t> &graph_axis_ids) {
  int64_t max_axis_id = -1;
  for (const auto axis_id : graph_axis_ids) {
    max_axis_id = std::max(max_axis_id, axis_id);
  }
  return max_axis_id + 1;
}

inline std::vector<int64_t> CollectAxisIds(const std::vector<AxisPtr> &graph_axis) {
  std::vector<int64_t> axis_ids;
  axis_ids.reserve(graph_axis.size());
  for (const auto &axis_info : graph_axis) {
    if (axis_info != nullptr) {
      axis_ids.push_back(axis_info->id);
    }
  }
  return axis_ids;
}

template <typename T>
inline void SortUniqueVector(std::vector<T> &values) {
  std::sort(values.begin(), values.end());
  values.erase(std::unique(values.begin(), values.end()), values.end());
}

inline void ShiftAxisGroupFrom(const int64_t insert_axis_id, optimize::autoschedule::AxisGroup &axis_group) {
  auto shift_axis_ids = [insert_axis_id](std::vector<int64_t> &axis_ids) {
    for (auto &axis_id : axis_ids) {
      ShiftAxisIdFrom(insert_axis_id, axis_id);
    }
    SortUniqueVector(axis_ids);
  };

  shift_axis_ids(axis_group.x_group);
  shift_axis_ids(axis_group.y_group);
  shift_axis_ids(axis_group.r_group);
  shift_axis_ids(axis_group.n_group);

  for (auto &axis_order : axis_group.axes_order) {
    if (axis_order >= static_cast<size_t>(insert_axis_id)) {
      ++axis_order;
    }
  }
  SortUniqueVector(axis_group.axes_order);
}

inline Status RefreshReshapeAxisGroupByInsertIndexes(const std::vector<int64_t> &axis_before_insert,
                                                     const std::vector<size_t> &insert_indexes,
                                                     optimize::autoschedule::AxisGroup &axis_group) {
  if (insert_indexes.empty() || axis_before_insert.empty() || axis_group.IsEmpty()) {
    return SUCCESS;
  }

  auto current_axis_ids = axis_before_insert;
  auto sorted_insert_indexes = insert_indexes;
  std::sort(sorted_insert_indexes.begin(), sorted_insert_indexes.end());
  for (const auto insert_index : sorted_insert_indexes) {
    const auto insert_axis_id = (insert_index < current_axis_ids.size()) ? current_axis_ids[insert_index]
                                                                         : GetNextNoOpReshapeAxisId(current_axis_ids);
    ShiftAxisGroupFrom(insert_axis_id, axis_group);
    for (auto &axis_id : current_axis_ids) {
      ShiftAxisIdFrom(insert_axis_id, axis_id);
    }
    current_axis_ids.insert(
        current_axis_ids.begin() + static_cast<ptrdiff_t>(std::min(insert_index, current_axis_ids.size())),
        insert_axis_id);
    if (std::find(axis_group.y_group.begin(), axis_group.y_group.end(), insert_axis_id) == axis_group.y_group.end()) {
      axis_group.y_group.push_back(insert_axis_id);
    }
    if (std::find(axis_group.axes_order.begin(), axis_group.axes_order.end(), static_cast<size_t>(insert_axis_id)) ==
        axis_group.axes_order.end()) {
      axis_group.axes_order.push_back(static_cast<size_t>(insert_axis_id));
    }
  }

  SortUniqueVector(axis_group.x_group);
  SortUniqueVector(axis_group.y_group);
  SortUniqueVector(axis_group.r_group);
  SortUniqueVector(axis_group.n_group);
  SortUniqueVector(axis_group.axes_order);
  return SUCCESS;
}

inline void CopyTensorAttrs(const AscTensorAttr &src_attr, AscTensorAttr &dst_attr) {
  dst_attr.axis = src_attr.axis;
  dst_attr.repeats = src_attr.repeats;
  dst_attr.strides = src_attr.strides;
}

inline void DumpTensorAttrs(const char *stage, const NodePtr &node, const AscTensorAttr &tensor_attr) {
  GELOGD("%s node %s(%s) tensor attr axis:%s, repeats:%s, strides:%s.", stage, node->GetNamePtr(),
         node->GetType().c_str(), AutofuseUtils::VectorToStr(tensor_attr.axis).c_str(),
         AutofuseUtils::VectorToStr(tensor_attr.repeats).c_str(),
         AutofuseUtils::VectorToStr(tensor_attr.strides).c_str());
}

inline void ReorderGraphAxesToOriginalOrder(std::vector<AxisPtr> &axis) {
  std::stable_sort(axis.begin(), axis.end(), [](const AxisPtr &lhs, const AxisPtr &rhs) {
    if ((lhs == nullptr) || (rhs == nullptr)) {
      return lhs != nullptr;
    }
    return lhs->id < rhs->id;
  });
}

inline void ReorderAxisIdsByGraphOrder(const std::vector<int64_t> &graph_axis_order, std::vector<int64_t> &axis) {
  std::vector<int64_t> reordered_axis;
  for (const auto graph_axis_id : graph_axis_order) {
    auto it = std::find(axis.begin(), axis.end(), graph_axis_id);
    if (it != axis.end()) {
      reordered_axis.push_back(graph_axis_id);
    }
  }
  for (const auto axis_id : axis) {
    if (std::find(graph_axis_order.begin(), graph_axis_order.end(), axis_id) == graph_axis_order.end()) {
      reordered_axis.push_back(axis_id);
    }
  }
  axis = reordered_axis;
}

inline void ReorderTensorAttrsByGraphOrder(const std::vector<int64_t> &graph_axis_order, AscTensorAttr &tensor_attr) {
  if (tensor_attr.axis.empty()) {
    return;
  }
  if ((tensor_attr.repeats.size() != tensor_attr.axis.size()) ||
      (tensor_attr.strides.size() != tensor_attr.axis.size())) {
    return;
  }
  AscTensorAttr reordered_attr = tensor_attr;
  reordered_attr.axis.clear();
  reordered_attr.repeats.clear();
  reordered_attr.strides.clear();
  for (const auto graph_axis_id : graph_axis_order) {
    auto it = std::find(tensor_attr.axis.begin(), tensor_attr.axis.end(), graph_axis_id);
    if (it == tensor_attr.axis.end()) {
      continue;
    }
    const auto idx = static_cast<size_t>(std::distance(tensor_attr.axis.begin(), it));
    reordered_attr.axis.push_back(tensor_attr.axis[idx]);
    reordered_attr.repeats.push_back(tensor_attr.repeats[idx]);
    reordered_attr.strides.push_back(tensor_attr.strides[idx]);
  }
  for (size_t i = 0U; i < tensor_attr.axis.size(); ++i) {
    if (std::find(graph_axis_order.begin(), graph_axis_order.end(), tensor_attr.axis[i]) != graph_axis_order.end()) {
      continue;
    }
    reordered_attr.axis.push_back(tensor_attr.axis[i]);
    reordered_attr.repeats.push_back(tensor_attr.repeats[i]);
    reordered_attr.strides.push_back(tensor_attr.strides[i]);
  }
  CopyTensorAttrs(reordered_attr, tensor_attr);
}

inline Status ReorderAscGraphAttrsByGraphOrder(const AscGraph &asc_graph,
                                               const std::vector<int64_t> &graph_axis_order) {
  for (const auto &node : AscGraphUtils::GetComputeGraph(asc_graph)->GetAllNodes()) {
    const auto &op_desc = node->GetOpDesc();
    GE_ASSERT_NOTNULL(op_desc);
    auto node_attr = op_desc->GetAttrsGroup<AscNodeAttr>();
    if (node_attr != nullptr) {
      ReorderAxisIdsByGraphOrder(graph_axis_order, node_attr->sched.axis);
    }
    for (size_t i = 0U; i < node->GetAllOutDataAnchorsSize(); ++i) {
      const auto output_tensor_desc = op_desc->MutableOutputDesc(i);
      GE_ASSERT_NOTNULL(output_tensor_desc);
      auto tensor_attr = output_tensor_desc->GetAttrsGroup<AscTensorAttr>();
      if (tensor_attr == nullptr) {
        continue;
      }
      ReorderTensorAttrsByGraphOrder(graph_axis_order, *tensor_attr);
    }
  }
  return SUCCESS;
}

inline Status ReorderAscGraphAttrsByCurrentGraphOrder(const AscGraph &asc_graph, const AscGraphAttr &graph_attr) {
  std::vector<int64_t> graph_axis_order;
  graph_axis_order.reserve(graph_attr.axis.size());
  for (const auto &axis_info : graph_attr.axis) {
    GE_ASSERT_NOTNULL(axis_info);
    graph_axis_order.push_back(axis_info->id);
  }
  GE_ASSERT_SUCCESS(ReorderAscGraphAttrsByGraphOrder(asc_graph, graph_axis_order));
  return SUCCESS;
}

inline bool BuildTargetToTensorIndexByGraphOrder(const std::vector<int64_t> &graph_axis_order,
                                                 const AscTensorAttr &tensor_attr,
                                                 std::vector<size_t> &target_to_tensor_index) {
  const auto matched_target_indexes =
      BuildBestOverlapTargetIndexes(tensor_attr.axis.size(), graph_axis_order.size(),
                                    [&tensor_attr, &graph_axis_order](size_t source_idx, size_t target_idx) -> int32_t {
                                      return tensor_attr.axis[source_idx] == graph_axis_order[target_idx] ? 1 : 0;
                                    });
  if (matched_target_indexes.size() != tensor_attr.axis.size()) {
    return false;
  }

  const size_t kInvalidIndex = graph_axis_order.size();
  target_to_tensor_index.assign(graph_axis_order.size(), kInvalidIndex);
  for (size_t tensor_axis_idx = 0U; tensor_axis_idx < matched_target_indexes.size(); ++tensor_axis_idx) {
    target_to_tensor_index[matched_target_indexes[tensor_axis_idx]] = tensor_axis_idx;
  }
  return true;
}

inline bool AppendCompletedTensorAxisByGraphOrder(const size_t target_idx, const size_t kInvalidIndex,
                                                  const std::vector<int64_t> &graph_axis_order,
                                                  const AscTensorAttr &tensor_attr, AscTensorAttr &completed_attr,
                                                  std::vector<int64_t> &inserted_axis_ids) {
  if (target_idx != kInvalidIndex) {
    completed_attr.axis.push_back(tensor_attr.axis[target_idx]);
    if (!tensor_attr.repeats.empty()) {
      completed_attr.repeats.push_back(tensor_attr.repeats[target_idx]);
    }
    if (!tensor_attr.strides.empty()) {
      completed_attr.strides.push_back(tensor_attr.strides[target_idx]);
    }
    return false;
  }
  const auto graph_axis_id = graph_axis_order[completed_attr.axis.size()];
  completed_attr.axis.push_back(graph_axis_id);
  if (!tensor_attr.repeats.empty()) {
    completed_attr.repeats.push_back(kSymbolOne);
  }
  if (!tensor_attr.strides.empty()) {
    completed_attr.strides.push_back(kSymbolZero);
  }
  inserted_axis_ids.push_back(graph_axis_id);
  return true;
}

inline void CompleteTensorAttrsByGraphOrderPreserveStrides(const NodePtr &node, const size_t output_idx,
                                                           const std::vector<int64_t> &graph_axis_order,
                                                           AscTensorAttr &tensor_attr) {
  const auto old_axis = tensor_attr.axis;
  std::vector<int64_t> inserted_axis_ids;
  if (tensor_attr.axis.size() >= graph_axis_order.size()) {
    return;
  }

  std::vector<size_t> target_to_tensor_index;
  if (!BuildTargetToTensorIndexByGraphOrder(graph_axis_order, tensor_attr, target_to_tensor_index)) {
    return;
  }

  AscTensorAttr completed_attr = tensor_attr;
  completed_attr.axis.clear();
  if (!tensor_attr.repeats.empty()) {
    completed_attr.repeats.clear();
  }
  if (!tensor_attr.strides.empty()) {
    completed_attr.strides.clear();
  }
  const size_t kInvalidIndex = graph_axis_order.size();
  for (size_t i = 0U; i < graph_axis_order.size(); ++i) {
    if (AppendCompletedTensorAxisByGraphOrder(target_to_tensor_index[i], kInvalidIndex, graph_axis_order, tensor_attr,
                                              completed_attr, inserted_axis_ids)) {
      GELOGD("node %s(%s) output %zu complete preserve tensor attrs with reshape axis id %ld at graph axis idx %zu.",
             node->GetName().c_str(), node->GetType().c_str(), output_idx, graph_axis_order[i], i);
    }
  }
  CopyTensorAttrs(completed_attr, tensor_attr);
  if (!inserted_axis_ids.empty()) {
    GELOGD("node %s(%s) output %zu complete preserve tensor attrs with reshape axes %s, axis from %s to %s.",
           node->GetName().c_str(), node->GetType().c_str(), output_idx,
           AutofuseUtils::VectorToStr(inserted_axis_ids).c_str(), AutofuseUtils::VectorToStr(old_axis).c_str(),
           AutofuseUtils::VectorToStr(tensor_attr.axis).c_str());
  }
}

inline Status InsertNoOpAxisPreserveOrder(const int64_t axis_id, std::vector<int64_t> &axis,
                                          std::vector<Expression> *repeats = nullptr,
                                          std::vector<Expression> *strides = nullptr) {
  if (std::find(axis.begin(), axis.end(), axis_id) != axis.end()) {
    return SUCCESS;
  }
  const auto insert_it = std::find_if(axis.begin(), axis.end(),
                                      [axis_id](const int64_t current_axis_id) { return current_axis_id > axis_id; });
  const auto insert_index = static_cast<size_t>(std::distance(axis.begin(), insert_it));
  axis.insert(insert_it, axis_id);
  if ((repeats != nullptr) && !repeats->empty()) {
    GE_ASSERT_TRUE(insert_index <= repeats->size());
    repeats->insert(repeats->begin() + static_cast<ptrdiff_t>(insert_index), kSymbolOne);
  }
  if ((strides != nullptr) && !strides->empty()) {
    GE_ASSERT_TRUE(insert_index <= strides->size());
    strides->insert(strides->begin() + static_cast<ptrdiff_t>(insert_index), kSymbolZero);
  }
  return SUCCESS;
}

inline Status CompleteTensorAttrsByInsertedAxisIdsPreserveOrder(const NodePtr &node, const size_t output_idx,
                                                                const std::vector<int64_t> &inserted_axis_ids,
                                                                AscTensorAttr &tensor_attr) {
  const auto old_axis = tensor_attr.axis;
  if (tensor_attr.axis.empty()) {
    return SUCCESS;
  }
  for (const auto axis_id : inserted_axis_ids) {
    GE_ASSERT_SUCCESS(
        InsertNoOpAxisPreserveOrder(axis_id, tensor_attr.axis, &tensor_attr.repeats, &tensor_attr.strides));
  }
  if (old_axis != tensor_attr.axis) {
    GELOGD("node %s(%s) output %zu complete preserve tensor attrs with reshape axes %s, axis from %s to %s.",
           node->GetName().c_str(), node->GetType().c_str(), output_idx,
           AutofuseUtils::VectorToStr(inserted_axis_ids).c_str(), AutofuseUtils::VectorToStr(old_axis).c_str(),
           AutofuseUtils::VectorToStr(tensor_attr.axis).c_str());
  }
  return SUCCESS;
}

inline Status UpdateTensorAttrsPreserveStrides(const NodePtr &node, const std::vector<int64_t> &axis,
                                               const std::vector<Expression> &repeats) {
  const auto &op_desc = node->GetOpDesc();
  GE_ASSERT_NOTNULL(op_desc);
  for (size_t i = 0U; i < node->GetAllOutDataAnchorsSize(); ++i) {
    const auto output_tensor_desc = op_desc->MutableOutputDesc(i);
    GE_ASSERT_NOTNULL(output_tensor_desc);
    auto tensor_attr = output_tensor_desc->GetAttrsGroup<AscTensorAttr>();
    GE_ASSERT_NOTNULL(tensor_attr);
    if (node->GetType() == kScalarType) {
      tensor_attr->axis = axis;
      tensor_attr->repeats.assign(axis.size(), kSymbolOne);
      tensor_attr->strides.assign(axis.size(), kSymbolZero);
      continue;
    }
    if ((tensor_attr->axis.empty()) && (node->GetType() != kDataType)) {
      GE_ASSERT_SUCCESS(UpdateTensorAttrsIfEmpty(node, tensor_attr, axis, repeats));
      continue;
    }
    CompleteTensorAttrsByGraphOrderPreserveStrides(node, i, axis, *tensor_attr);
    GELOGD("after preserve attrs: node %s(%s), axis:%s, repeats:%s stride:%s.", node->GetName().c_str(),
           node->GetType().c_str(), AutofuseUtils::VectorToStr(tensor_attr->axis).c_str(),
           AutofuseUtils::VectorToStr(tensor_attr->repeats).c_str(),
           AutofuseUtils::VectorToStr(tensor_attr->strides).c_str());
  }
  return SUCCESS;
}

inline Status UpdateTensorAttrsByInsertedAxisIdsPreserveOrder(const NodePtr &node, const std::vector<int64_t> &axis,
                                                              const std::vector<Expression> &repeats,
                                                              const std::vector<int64_t> &inserted_axis_ids) {
  const auto &op_desc = node->GetOpDesc();
  GE_ASSERT_NOTNULL(op_desc);
  for (size_t i = 0U; i < node->GetAllOutDataAnchorsSize(); ++i) {
    const auto output_tensor_desc = op_desc->MutableOutputDesc(i);
    GE_ASSERT_NOTNULL(output_tensor_desc);
    auto tensor_attr = output_tensor_desc->GetAttrsGroup<AscTensorAttr>();
    GE_ASSERT_NOTNULL(tensor_attr);
    if (node->GetType() == kScalarType) {
      tensor_attr->axis = axis;
      tensor_attr->repeats.assign(axis.size(), kSymbolOne);
      tensor_attr->strides.assign(axis.size(), kSymbolZero);
      continue;
    }
    if ((tensor_attr->axis.empty()) && (node->GetType() != kDataType)) {
      GE_ASSERT_SUCCESS(UpdateTensorAttrsIfEmpty(node, tensor_attr, axis, repeats));
      continue;
    }
    GE_ASSERT_SUCCESS(CompleteTensorAttrsByInsertedAxisIdsPreserveOrder(node, i, inserted_axis_ids, *tensor_attr));
    GELOGD("after preserve attrs: node %s(%s), axis:%s, repeats:%s stride:%s.", node->GetName().c_str(),
           node->GetType().c_str(), AutofuseUtils::VectorToStr(tensor_attr->axis).c_str(),
           AutofuseUtils::VectorToStr(tensor_attr->repeats).c_str(),
           AutofuseUtils::VectorToStr(tensor_attr->strides).c_str());
  }
  return SUCCESS;
}

inline bool IsAxisSubsequence(const std::vector<int64_t> &axis, const std::vector<int64_t> &target_axis) {
  size_t target_idx = 0U;
  for (const auto axis_id : axis) {
    while ((target_idx < target_axis.size()) && (target_axis[target_idx] != axis_id)) {
      ++target_idx;
    }
    if (target_idx >= target_axis.size()) {
      return false;
    }
    ++target_idx;
  }
  return true;
}

inline void CompleteGatherInputTensorAttrsByTargetAxis(const std::vector<int64_t> &target_axis,
                                                       AscTensorAttr &tensor_attr) {
  std::unordered_map<int64_t, size_t> axis_to_index;
  for (size_t i = 0U; i < tensor_attr.axis.size(); ++i) {
    axis_to_index[tensor_attr.axis[i]] = i;
  }

  AscTensorAttr completed_attr = tensor_attr;
  completed_attr.axis.clear();
  completed_attr.repeats.clear();
  completed_attr.strides.clear();
  for (const auto axis_id : target_axis) {
    completed_attr.axis.push_back(axis_id);
    const auto iter = axis_to_index.find(axis_id);
    if (iter != axis_to_index.end()) {
      const auto idx = iter->second;
      completed_attr.repeats.push_back(idx < tensor_attr.repeats.size() ? tensor_attr.repeats[idx] : kSymbolOne);
      completed_attr.strides.push_back(idx < tensor_attr.strides.size() ? tensor_attr.strides[idx] : kSymbolZero);
    } else {
      completed_attr.repeats.push_back(kSymbolOne);
      completed_attr.strides.push_back(kSymbolZero);
    }
  }
  CopyTensorAttrs(completed_attr, tensor_attr);
}

inline void CompleteGatherInputTensorAttrsByDimOrder(const std::vector<int64_t> &target_axis,
                                                     AscTensorAttr &tensor_attr) {
  AscTensorAttr completed_attr = tensor_attr;
  completed_attr.axis = target_axis;
  for (size_t i = 0U; i < completed_attr.axis.size(); ++i) {
    if (i >= completed_attr.repeats.size()) {
      completed_attr.repeats.push_back(kSymbolOne);
    }
    if (i >= completed_attr.strides.size()) {
      completed_attr.strides.push_back(kSymbolZero);
    }
  }
  completed_attr.repeats.resize(completed_attr.axis.size());
  completed_attr.strides.resize(completed_attr.axis.size());
  CopyTensorAttrs(completed_attr, tensor_attr);
}

inline void CompleteGatherInputTensorAttrsByAxes(const std::vector<int64_t> &target_axis, AscTensorAttr &tensor_attr) {
  if (IsAxisSubsequence(tensor_attr.axis, target_axis)) {
    CompleteGatherInputTensorAttrsByTargetAxis(target_axis, tensor_attr);
    return;
  }
  CompleteGatherInputTensorAttrsByDimOrder(target_axis, tensor_attr);
}

inline Status GetGatherInputDataNodes(const NodePtr &gather_node, NodePtr &params_data_node,
                                      NodePtr &indices_data_node) {
  std::vector<NodePtr> peer_out_nodes;
  GE_ASSERT_SUCCESS(asc_adapt::GetPeerOutNodes(gather_node, peer_out_nodes));
  GE_ASSERT_TRUE(peer_out_nodes.size() == 2U);
  params_data_node = peer_out_nodes[0];
  indices_data_node = peer_out_nodes[1];
  return SUCCESS;
}

inline bool IsGatherFuseType(const NodePtr &asc_node) {
  const auto autofuse_attr = BackendUtils::GetNodeAutoFuseAttr(asc_node);
  return (autofuse_attr != nullptr) && autofuse_attr->HasFuseType(loop::FuseType::kGather);
}

inline Status GetGatherReplacementAxisIndex(const NodePtr &gather_node, const AscTensorAttr &params_attr,
                                            const AscTensorAttr &indices_attr, const AscTensorAttr &gather_output_attr,
                                            size_t &replacement_axis_index) {
  (void)indices_attr;
  int64_t gather_axis_index = 0;
  GE_ASSERT_SUCCESS(asc_adapt::GetGatherAxis(gather_node, gather_axis_index));
  GE_ASSERT_TRUE(gather_axis_index >= 0);
  GE_ASSERT_TRUE(static_cast<size_t>(gather_axis_index) < params_attr.axis.size());
  const auto replacement_axis_id = params_attr.axis[static_cast<size_t>(gather_axis_index)];
  const auto iter = std::find(gather_output_attr.axis.begin(), gather_output_attr.axis.end(), replacement_axis_id);
  GE_ASSERT_TRUE(iter != gather_output_attr.axis.end());
  replacement_axis_index = static_cast<size_t>(std::distance(gather_output_attr.axis.begin(), iter));
  if (replacement_axis_index != static_cast<size_t>(gather_axis_index)) {
    GE_ASSERT_SUCCESS(asc_adapt::SetGatherAxis(gather_node, static_cast<int64_t>(replacement_axis_index)));
  }
  return SUCCESS;
}

inline Status CompleteGatherInputAttrsPreserveStrides(const AscGraph &asc_graph, const NodePtr &gather_node) {
  NodePtr params_data_node = nullptr;
  NodePtr indices_data_node = nullptr;
  GE_ASSERT_SUCCESS(GetGatherInputDataNodes(gather_node, params_data_node, indices_data_node));
  AscTensorAttr *gather_output_attr = nullptr;
  GE_ASSERT_SUCCESS(asc_adapt::GetOutputTensorAttr(gather_node, gather_output_attr));
  AscTensorAttr *params_attr = nullptr;
  GE_ASSERT_SUCCESS(asc_adapt::GetOutputTensorAttr(params_data_node, params_attr));
  AscTensorAttr *indices_attr = nullptr;
  GE_ASSERT_SUCCESS(asc_adapt::GetOutputTensorAttr(indices_data_node, indices_attr));
  if ((gather_output_attr->axis.empty()) || (params_attr->axis.empty()) || (indices_attr->axis.empty())) {
    return SUCCESS;
  }

  size_t replacement_axis_index = 0U;
  GE_ASSERT_SUCCESS(GetGatherReplacementAxisIndex(gather_node, *params_attr, *indices_attr, *gather_output_attr,
                                                  replacement_axis_index));
  GE_ASSERT_TRUE(replacement_axis_index + indices_attr->axis.size() <= gather_output_attr->axis.size());
  GELOGD("node %s(%s) complete gather input attrs in graph %s, replacement_axis_index:%zu.", gather_node->GetNamePtr(),
         gather_node->GetType().c_str(), asc_graph.GetName().c_str(), replacement_axis_index);
  DumpTensorAttrs("before complete gather output", gather_node, *gather_output_attr);
  DumpTensorAttrs("before complete gather params", params_data_node, *params_attr);
  DumpTensorAttrs("before complete gather indices", indices_data_node, *indices_attr);

  std::vector<int64_t> params_axis;
  std::vector<int64_t> indices_axis;
  for (size_t i = 0U; i < gather_output_attr->axis.size(); ++i) {
    if (i == replacement_axis_index) {
      params_axis.push_back(gather_output_attr->axis[i]);
      indices_axis.push_back(gather_output_attr->axis[i]);
    } else if ((i > replacement_axis_index) && (i < replacement_axis_index + indices_attr->axis.size())) {
      indices_axis.push_back(gather_output_attr->axis[i]);
    } else {
      params_axis.push_back(gather_output_attr->axis[i]);
    }
  }
  GELOGI("node %s(%s) complete gather params axis from %s to %s in graph %s.", params_data_node->GetName().c_str(),
         params_data_node->GetType().c_str(), AutofuseUtils::VectorToStr(params_attr->axis).c_str(),
         AutofuseUtils::VectorToStr(params_axis).c_str(), asc_graph.GetName().c_str());
  GELOGI("node %s(%s) complete gather indices axis from %s to %s in graph %s.", indices_data_node->GetName().c_str(),
         indices_data_node->GetType().c_str(), AutofuseUtils::VectorToStr(indices_attr->axis).c_str(),
         AutofuseUtils::VectorToStr(indices_axis).c_str(), asc_graph.GetName().c_str());
  CompleteGatherInputTensorAttrsByAxes(params_axis, *params_attr);
  CompleteGatherInputTensorAttrsByAxes(indices_axis, *indices_attr);
  DumpTensorAttrs("after complete gather params", params_data_node, *params_attr);
  DumpTensorAttrs("after complete gather indices", indices_data_node, *indices_attr);
  return SUCCESS;
}

inline Status CompleteGatherInputAttrsOnAscGraphPreserveStrides(const AscGraph &asc_graph) {
  for (const auto &node : AscGraphUtils::GetComputeGraph(asc_graph)->GetAllNodes()) {
    if (node->GetType() != kGatherType) {
      continue;
    }
    GE_ASSERT_SUCCESS(CompleteGatherInputAttrsPreserveStrides(asc_graph, node));
  }
  return SUCCESS;
}

inline Status CompleteNodeAttrsOnAscGraphPreserveStrides(AscGraph &asc_graph, const NodePtr &asc_node) {
  const auto is_gather_fuse_type = IsGatherFuseType(asc_node);
  TensorAttrInfo graph_attr;
  GE_ASSERT_SUCCESS(BackendUtils::GetGraphAttrInfo(asc_graph, graph_attr));
  GELOGI("max sched axis %s in graph %s, preserve tensor strides.", AutofuseUtils::VectorToStr(graph_attr.axis).c_str(),
         asc_graph.GetName().c_str());

  for (const auto &node : AscGraphUtils::GetComputeGraph(asc_graph)->GetAllNodes()) {
    if (IsTorchDataType(node)) {
      GELOGI("torch node %s(%s) not complete node attr.", node->GetName().c_str(), node->GetType().c_str());
      continue;
    }
    if (IsCubeRelatedAscNode(node)) {
      GELOGI("cube related node %s(%s) not complete node attr.", node->GetName().c_str(), node->GetType().c_str());
      continue;
    }
    if (((!is_gather_fuse_type) || (!IsGatherData(node))) && (!BackendUtils::IsOutputNode(node))) {
      GE_ASSERT_SUCCESS(UpdateTensorAttrsPreserveStrides(node, graph_attr.axis, graph_attr.repeats));
    }
    const auto &op_desc = node->GetOpDesc();
    GE_ASSERT_NOTNULL(op_desc);
    auto node_attr = op_desc->GetAttrsGroup<AscNodeAttr>();
    GE_ASSERT_NOTNULL(node_attr);
    GELOGI("node %s(%s) before complete sched axis %s to %s in graph %s, preserve tensor strides.",
           node->GetName().c_str(), node->GetType().c_str(), AutofuseUtils::VectorToStr(node_attr->sched.axis).c_str(),
           AutofuseUtils::VectorToStr(graph_attr.axis).c_str(), asc_graph.GetName().c_str());
    node_attr->sched.axis = graph_attr.axis;
    GELOGI("node %s(%s) after complete sched axis %s to %s in graph %s, preserve tensor strides.",
           node->GetName().c_str(), node->GetType().c_str(), AutofuseUtils::VectorToStr(node_attr->sched.axis).c_str(),
           AutofuseUtils::VectorToStr(graph_attr.axis).c_str(), asc_graph.GetName().c_str());
  }
  if (is_gather_fuse_type) {
    GE_ASSERT_SUCCESS(CompleteGatherInputAttrsOnAscGraphPreserveStrides(asc_graph));
  }
  return SUCCESS;
}

inline Status CompleteNodeAttrsByInsertedAxisIdsPreserveOrder(AscGraph &asc_graph, const NodePtr &asc_node,
                                                              const std::vector<int64_t> &inserted_axis_ids) {
  const auto is_gather_fuse_type = IsGatherFuseType(asc_node);
  TensorAttrInfo graph_attr;
  GE_ASSERT_SUCCESS(BackendUtils::GetGraphAttrInfo(asc_graph, graph_attr));
  GELOGI("complete inserted reshape axes %s in graph %s, preserve node axis order.",
         AutofuseUtils::VectorToStr(inserted_axis_ids).c_str(), asc_graph.GetName().c_str());

  for (const auto &node : AscGraphUtils::GetComputeGraph(asc_graph)->GetAllNodes()) {
    if (IsTorchDataType(node)) {
      GELOGI("torch node %s(%s) not complete node attr.", node->GetName().c_str(), node->GetType().c_str());
      continue;
    }
    if (IsCubeRelatedAscNode(node)) {
      GELOGI("cube related node %s(%s) not complete node attr.", node->GetName().c_str(), node->GetType().c_str());
      continue;
    }
    if (((!is_gather_fuse_type) || (!IsGatherData(node))) && (!BackendUtils::IsOutputNode(node))) {
      GE_ASSERT_SUCCESS(UpdateTensorAttrsByInsertedAxisIdsPreserveOrder(node, graph_attr.axis, graph_attr.repeats,
                                                                        inserted_axis_ids));
    }
    const auto &op_desc = node->GetOpDesc();
    GE_ASSERT_NOTNULL(op_desc);
    auto node_attr = op_desc->GetAttrsGroup<AscNodeAttr>();
    GE_ASSERT_NOTNULL(node_attr);
    const auto old_sched_axis = node_attr->sched.axis;
    if (node_attr->sched.axis.empty()) {
      node_attr->sched.axis = graph_attr.axis;
    } else {
      for (const auto axis_id : inserted_axis_ids) {
        GE_ASSERT_SUCCESS(InsertNoOpAxisPreserveOrder(axis_id, node_attr->sched.axis));
      }
    }
    GELOGI("node %s(%s) complete sched axis from %s to %s in graph %s, preserve node axis order.",
           node->GetName().c_str(), node->GetType().c_str(), AutofuseUtils::VectorToStr(old_sched_axis).c_str(),
           AutofuseUtils::VectorToStr(node_attr->sched.axis).c_str(), asc_graph.GetName().c_str());
  }
  if (is_gather_fuse_type) {
    GE_ASSERT_SUCCESS(CompleteGatherInputAttrsOnAscGraphPreserveStrides(asc_graph));
  }
  return SUCCESS;
}

inline Status DumpPadLeadingUnitAxisResult(const AscGraph &asc_graph, const NodePtr &asc_node, const char *process_name,
                                           const ComputeGraphPtr &fused_graph) {
  GELOGI("AscBackendPostProcessor: End to run the process(%s) on the graph, graph: %s, parent node: %s(%s).",
         process_name, fused_graph->GetName().c_str(), asc_node->GetNamePtr(), asc_node->GetType().c_str());
  GELOGD("dump node:%s(%s) asc graph info(with tensor attr info):", asc_node->GetNamePtr(),
         asc_node->GetType().c_str());
  (void)asc_graph;
  BackendUtils::DumpAscGraph(asc_node);
  return SUCCESS;
}

struct InsertIndexMergeState {
  bool initialized = false;
  size_t original_rank = 0U;
  std::vector<size_t> gap_counts;
};

inline std::vector<size_t> BuildInsertGapCounts(const size_t original_rank, const std::vector<size_t> &insert_indexes) {
  std::vector<size_t> gap_counts(original_rank + 1U, 0U);
  auto sorted_insert_indexes = insert_indexes;
  std::sort(sorted_insert_indexes.begin(), sorted_insert_indexes.end());

  size_t inserted_count = 0U;
  for (const auto insert_index : sorted_insert_indexes) {
    const size_t gap_index = std::min(insert_index - std::min(insert_index, inserted_count), original_rank);
    ++gap_counts[gap_index];
    ++inserted_count;
  }
  return gap_counts;
}

inline void BuildInsertIndexesFromGapCounts(const std::vector<size_t> &gap_counts,
                                            std::vector<size_t> &insert_indexes) {
  insert_indexes.clear();
  size_t inserted_count = 0U;
  for (size_t gap_index = 0U; gap_index < gap_counts.size(); ++gap_index) {
    for (size_t i = 0U; i < gap_counts[gap_index]; ++i) {
      insert_indexes.push_back(gap_index + inserted_count);
      ++inserted_count;
    }
  }
}

inline Status MergeRelationInsertIndexes(const size_t original_rank, const std::vector<size_t> &relation_insert_indexes,
                                         InsertIndexMergeState &merge_state,
                                         std::vector<size_t> &merged_insert_indexes) {
  if (relation_insert_indexes.empty()) {
    return SUCCESS;
  }
  if (!merge_state.initialized) {
    merge_state.initialized = true;
    merge_state.original_rank = original_rank;
    merge_state.gap_counts.assign(original_rank + 1U, 0U);
  }
  if (merge_state.original_rank != original_rank) {
    GELOGD("skip merging relation insert indexes %s because original rank %zu does not match merged rank %zu.",
           AutofuseUtils::VectorToStr(relation_insert_indexes).c_str(), original_rank, merge_state.original_rank);
    return SUCCESS;
  }

  const auto relation_gap_counts = BuildInsertGapCounts(original_rank, relation_insert_indexes);
  for (size_t i = 0U; i < merge_state.gap_counts.size(); ++i) {
    merge_state.gap_counts[i] = std::max(merge_state.gap_counts[i], relation_gap_counts[i]);
  }
  BuildInsertIndexesFromGapCounts(merge_state.gap_counts, merged_insert_indexes);
  GELOGD("merge relation insert indexes %s to merged insert indexes %s by original rank %zu.",
         AutofuseUtils::VectorToStr(relation_insert_indexes).c_str(),
         AutofuseUtils::VectorToStr(merged_insert_indexes).c_str(), original_rank);
  return SUCCESS;
}

inline Status CompleteNoOpReshapeAxesByInsertIndexes(AscGraph &asc_graph, const std::vector<size_t> &insert_indexes,
                                                     AscGraphAttr *graph_attr,
                                                     std::vector<af::ReshapeAxisChangeInfo> *reshape_axis_changes,
                                                     std::vector<int64_t> &inserted_axis_ids) {
  GE_ASSERT_NOTNULL(graph_attr);
  if (insert_indexes.empty()) {
    return SUCCESS;
  }

  auto sorted_insert_indexes = insert_indexes;
  std::sort(sorted_insert_indexes.begin(), sorted_insert_indexes.end());
  for (const auto insert_index : sorted_insert_indexes) {
    int64_t insert_axis_id = GetNextNoOpReshapeAxisId(graph_attr->axis);
    if (insert_index < graph_attr->axis.size()) {
      GE_ASSERT_NOTNULL(graph_attr->axis[insert_index]);
      insert_axis_id = graph_attr->axis[insert_index]->id;
    }
    for (const auto &axis_info : graph_attr->axis) {
      ShiftAxisInfoIdFrom(insert_axis_id, axis_info);
    }
    if (reshape_axis_changes != nullptr) {
      ShiftReshapeAxisChangesFrom(insert_axis_id, *reshape_axis_changes);
    }
    GE_ASSERT_SUCCESS(ShiftAscGraphAxisIdsFrom(asc_graph, insert_axis_id));
    const auto graph_insert_it =
        graph_attr->axis.begin() + static_cast<ptrdiff_t>(std::min(insert_index, graph_attr->axis.size()));
    graph_attr->axis.insert(graph_insert_it, MakeNoOpReshapeAxis(insert_axis_id, kSymbolOne));
    inserted_axis_ids.push_back(insert_axis_id);
    GELOGD(
        "graph %s complete no-op reshape axis id %ld repeat %s at insert index %zu by relation before complete attrs.",
        asc_graph.GetName().c_str(), insert_axis_id, kSymbolOne.Str().get(), insert_index);
  }
  return SUCCESS;
}

inline Status PadLeadingUnitAxisByInsertIndexesAndCompleteAttrs(AscGraph &asc_graph, const NodePtr &asc_node,
                                                                const std::vector<size_t> &insert_indexes) {
  if (insert_indexes.empty()) {
    return SUCCESS;
  }
  constexpr const char *kPadLeadingUnitAxisProcName = "pad_leading_unit_axis";
  const auto fused_graph = AscGraphUtils::GetComputeGraph(asc_graph);
  GE_ASSERT_NOTNULL(fused_graph);
  GE_ASSERT_SUCCESS(BackendUtils::AddInputOutputNodesForAscGraph(fused_graph));
  GE_ASSERT_SUCCESS(CacheGraphBeforePostProcess(asc_node, kPadLeadingUnitAxisProcName, fused_graph));
  auto autofuse_attr = BackendUtils::GetNodeAutoFuseAttr(asc_node);
  GE_ASSERT_NOTNULL(autofuse_attr);
  const auto graph_attr = fused_graph->GetAttrsGroup<AscGraphAttr>();
  GE_ASSERT_NOTNULL(graph_attr);
  ReorderGraphAxesToOriginalOrder(graph_attr->axis);
  const auto axis_before_insert = CollectAxisIds(graph_attr->axis);
  std::vector<int64_t> inserted_axis_ids;
  GELOGD("node %s(%s) graph %s start to pad leading unit axes by relation insert indexes %s.",
         asc_node->GetName().c_str(), asc_node->GetType().c_str(), asc_graph.GetName().c_str(),
         AutofuseUtils::VectorToStr(insert_indexes).c_str());
  GE_ASSERT_SUCCESS(CompleteNoOpReshapeAxesByInsertIndexes(asc_graph, insert_indexes, graph_attr,
                                                           &autofuse_attr->GetMutableInterAttrs().reshape_axis_changes,
                                                           inserted_axis_ids));
  GE_ASSERT_SUCCESS(CompleteNodeAttrsByInsertedAxisIdsPreserveOrder(asc_graph, asc_node, inserted_axis_ids));
  GE_ASSERT_SUCCESS(RefreshReshapeAxisGroupByInsertIndexes(axis_before_insert, insert_indexes,
                                                           GetInterAttrs(autofuse_attr).axis_group));
  GE_ASSERT_SUCCESS(DumpPadLeadingUnitAxisResult(asc_graph, asc_node, kPadLeadingUnitAxisProcName, fused_graph));
  return SUCCESS;
}

inline bool IsSameReshapeAxisChange(const af::ReshapeAxisChangeInfo &lhs, const af::ReshapeAxisChangeInfo &rhs) {
  return (lhs.before_axis == rhs.before_axis) && (lhs.after_axis == rhs.after_axis) &&
         (lhs.before_repeats.size() == rhs.before_repeats.size()) &&
         (lhs.after_repeats.size() == rhs.after_repeats.size()) &&
         std::equal(lhs.before_repeats.begin(), lhs.before_repeats.end(), rhs.before_repeats.begin(), IsSameRepeat) &&
         std::equal(lhs.after_repeats.begin(), lhs.after_repeats.end(), rhs.after_repeats.begin(), IsSameRepeat);
}

inline void AppendUniqueReshapeAxisChange(const af::ReshapeAxisChangeInfo &axis_change,
                                          std::vector<af::ReshapeAxisChangeInfo> &axis_changes) {
  const auto it = std::find_if(axis_changes.begin(), axis_changes.end(), [&axis_change](const auto &saved_change) {
    return IsSameReshapeAxisChange(axis_change, saved_change);
  });
  if (it == axis_changes.end()) {
    axis_changes.push_back(axis_change);
  }
}

inline Status CollectNodeReshapeAxisChanges(const NodePtr &node, std::vector<af::ReshapeAxisChangeInfo> &axis_changes) {
  if (!BackendUtils::IsBackendFuseNode(node)) {
    return SUCCESS;
  }
  const auto attr = node->GetOpDescBarePtr()->GetAttrsGroup<AutoFuseAttrs>();
  GE_ASSERT_NOTNULL(attr);
  for (const auto &axis_change : attr->GetReshapeAxisChanges()) {
    AppendUniqueReshapeAxisChange(axis_change, axis_changes);
  }
  if (node->GetType() == kAscBackendType) {
    return SUCCESS;
  }
  if (node->GetType() != kFusedAscBackendType) {
    return SUCCESS;
  }
  GE_ASSERT_NOTNULL(attr->GetFuseComputeGraph());
  for (const auto &inner_node : attr->GetFuseComputeGraph()->GetAllNodes()) {
    if ((inner_node == nullptr) || !BackendUtils::IsBackendFuseNode(inner_node) ||
        (inner_node->GetType() != kAscBackendType)) {
      continue;
    }
    GE_ASSERT_SUCCESS(CollectNodeReshapeAxisChanges(inner_node, axis_changes));
  }
  return SUCCESS;
}

inline Status SaveMergedReshapeAxisChanges(const NodePtr &node,
                                           const std::vector<af::ReshapeAxisChangeInfo> &axis_changes) {
  if (!BackendUtils::IsBackendFuseNode(node) || axis_changes.empty()) {
    return SUCCESS;
  }
  const auto attr = BackendUtils::GetNodeAutoFuseAttr(node);
  GE_ASSERT_NOTNULL(attr);
  auto merged_axis_changes = attr->GetReshapeAxisChanges();
  for (const auto &axis_change : axis_changes) {
    AppendUniqueReshapeAxisChange(axis_change, merged_axis_changes);
  }
  attr->SetReshapeAxisChanges(merged_axis_changes);
  return SUCCESS;
}

inline void AppendEndpointTargetCandidate(const std::vector<Expression> &current_repeats,
                                          const std::vector<af::ReshapeAxisChangeInfo> &axis_changes,
                                          std::vector<std::vector<Expression>> &target_candidates) {
  if (current_repeats.empty()) {
    return;
  }
  const auto current_non_unit_repeats = RemoveUnitRepeats(current_repeats);
  for (const auto &axis_change : axis_changes) {
    if (IsSameRepeats(current_non_unit_repeats, RemoveUnitRepeats(axis_change.before_repeats)) ||
        IsSameRepeats(current_non_unit_repeats, RemoveUnitRepeats(axis_change.after_repeats))) {
      AppendUniqueTargetRepeats(BuildNoOpReshapeTargetRepeats(axis_change), target_candidates);
    }
  }
}

inline Status BuildRelationTargetRepeats(const std::vector<Expression> &node1_repeats,
                                         const std::vector<Expression> &node2_repeats,
                                         const std::vector<af::ReshapeAxisChangeInfo> &node1_axis_changes,
                                         const std::vector<af::ReshapeAxisChangeInfo> &node2_axis_changes,
                                         std::vector<Expression> &target_repeats) {
  target_repeats.clear();
  if (node1_repeats.empty() && node2_repeats.empty()) {
    return SUCCESS;
  }
  if (!node1_repeats.empty() && !node2_repeats.empty() &&
      !IsSameRepeats(RemoveUnitRepeats(node1_repeats), RemoveUnitRepeats(node2_repeats))) {
    GELOGD(
        "skip relation reshape target because endpoint non-unit repeats are different, node1 repeats %s, node2 "
        "repeats %s.",
        AutofuseUtils::VectorToStr(node1_repeats).c_str(), AutofuseUtils::VectorToStr(node2_repeats).c_str());
    return SUCCESS;
  }

  std::vector<std::vector<Expression>> candidates;
  AppendUniqueTargetRepeats(node1_repeats, candidates);
  AppendUniqueTargetRepeats(node2_repeats, candidates);
  AppendEndpointTargetCandidate(node1_repeats, node1_axis_changes, candidates);
  AppendEndpointTargetCandidate(node2_repeats, node2_axis_changes, candidates);

  const auto base_repeats = !node1_repeats.empty() ? node1_repeats : node2_repeats;
  const auto base_non_unit_repeats = RemoveUnitRepeats(base_repeats);
  std::vector<size_t> unit_gap_counts = BuildUnitRepeatGapCounts(base_repeats);
  bool matched = false;
  for (const auto &candidate : candidates) {
    if (!IsSameRepeats(base_non_unit_repeats, RemoveUnitRepeats(candidate))) {
      continue;
    }
    MergeUnitRepeatGapCounts(BuildUnitRepeatGapCounts(candidate), unit_gap_counts);
    matched = true;
  }
  if (!matched) {
    return SUCCESS;
  }
  target_repeats = ApplyUnitRepeatGaps(base_non_unit_repeats, unit_gap_counts);
  return SUCCESS;
}

inline Status CollectInsertIndexesByTargetRepeats(const std::vector<Expression> &current_repeats,
                                                  const std::vector<Expression> &target_repeats,
                                                  std::vector<size_t> &insert_indexes) {
  if (current_repeats.empty() || target_repeats.empty() || (target_repeats.size() <= current_repeats.size())) {
    return SUCCESS;
  }
  const auto current_non_unit_repeats = RemoveUnitRepeats(current_repeats);
  const auto target_non_unit_repeats = RemoveUnitRepeats(target_repeats);
  if (!IsSameRepeats(current_non_unit_repeats, target_non_unit_repeats)) {
    return SUCCESS;
  }

  const auto current_gap_counts = BuildUnitRepeatGapCounts(current_repeats);
  const auto target_gap_counts = BuildUnitRepeatGapCounts(target_repeats);
  if (current_gap_counts.size() != target_gap_counts.size()) {
    return SUCCESS;
  }

  insert_indexes.clear();
  size_t current_gap_start = 0U;
  for (size_t gap_idx = 0U; gap_idx < current_gap_counts.size(); ++gap_idx) {
    if (target_gap_counts[gap_idx] < current_gap_counts[gap_idx]) {
      return SUCCESS;
    }
    const auto extra_unit_count = target_gap_counts[gap_idx] - current_gap_counts[gap_idx];
    for (size_t i = 0U; i < extra_unit_count; ++i) {
      insert_indexes.push_back(current_gap_start + current_gap_counts[gap_idx] + i);
    }
    current_gap_start += current_gap_counts[gap_idx];
    if (gap_idx + 1U < current_gap_counts.size()) {
      ++current_gap_start;
    }
  }
  GELOGD("collect reshape insert indexes %s by current repeats %s and edge target repeats %s.",
         AutofuseUtils::VectorToStr(insert_indexes).c_str(), AutofuseUtils::VectorToStr(current_repeats).c_str(),
         AutofuseUtils::VectorToStr(target_repeats).c_str());
  return SUCCESS;
}

inline Status PadNodeLeadingUnitAxisByInsertIndexes(const NodePtr &node, const std::vector<size_t> &insert_indexes) {
  if (insert_indexes.empty() || !BackendUtils::IsBackendFuseNode(node)) {
    return SUCCESS;
  }
  const auto attr = node->GetOpDescBarePtr()->GetAttrsGroup<AutoFuseAttrs>();
  GE_ASSERT_NOTNULL(attr);
  if (node->GetType() == kAscBackendType) {
    GE_ASSERT_NOTNULL(attr->GetAscGraph());
    GE_ASSERT_SUCCESS(PadLeadingUnitAxisByInsertIndexesAndCompleteAttrs(*(attr->GetAscGraph()), node, insert_indexes));
    return SUCCESS;
  }
  if (node->GetType() != kFusedAscBackendType) {
    return SUCCESS;
  }
  ComputeGraphPtr fused_graph;
  GE_ASSERT_SUCCESS(BackendUtils::GetNodeFusedGraph(node, fused_graph));
  GE_ASSERT_NOTNULL(fused_graph);
  const auto fused_graph_attr = fused_graph->GetAttrsGroup<AscGraphAttr>();
  GE_ASSERT_NOTNULL(fused_graph_attr);
  const auto axis_before_insert = CollectAxisIds(fused_graph_attr->axis);
  GE_ASSERT_NOTNULL(attr->GetFuseComputeGraph());
  for (const auto &inner_node : attr->GetFuseComputeGraph()->GetAllNodes()) {
    if ((inner_node == nullptr) || !BackendUtils::IsBackendFuseNode(inner_node) ||
        (inner_node->GetType() != kAscBackendType)) {
      continue;
    }
    GE_ASSERT_SUCCESS(PadNodeLeadingUnitAxisByInsertIndexes(inner_node, insert_indexes));
  }
  GE_ASSERT_SUCCESS(
      RefreshReshapeAxisGroupByInsertIndexes(axis_before_insert, insert_indexes, GetInterAttrs(attr).axis_group));
  return SUCCESS;
}

inline bool IsReshapeSearchTerminal(const NodePtr &node, const bool search_forward) {
  if (search_forward) {
    return BackendUtils::IsOutputNode(node);
  }
  return (node->GetType() == kDataType) || (node->GetType() == kScalarType);
}

inline bool FindDirectionalNonUnitRepeatByAxisId(const NodePtr &node, const int64_t axis_id, const bool search_forward,
                                                 std::unordered_set<const void *> &visited, Expression &repeat) {
  if ((node == nullptr) || (visited.find(node.get()) != visited.end())) {
    return false;
  }
  visited.insert(node.get());
  if (IsReshapeSearchTerminal(node, search_forward)) {
    return false;
  }

  const auto op_desc = node->GetOpDesc();
  if (op_desc == nullptr) {
    return false;
  }
  for (size_t i = 0U; i < node->GetAllOutDataAnchorsSize(); ++i) {
    const auto output_desc = op_desc->MutableOutputDesc(i);
    if (output_desc == nullptr) {
      continue;
    }
    const auto tensor_attr = output_desc->GetAttrsGroup<AscTensorAttr>();
    if (tensor_attr == nullptr) {
      continue;
    }
    const auto axis_it = std::find(tensor_attr->axis.begin(), tensor_attr->axis.end(), axis_id);
    if (axis_it == tensor_attr->axis.end()) {
      continue;
    }
    const auto axis_index = static_cast<size_t>(std::distance(tensor_attr->axis.begin(), axis_it));
    if ((axis_index < tensor_attr->repeats.size()) && !BackendUtils::IsEqOne(tensor_attr->repeats[axis_index])) {
      repeat = tensor_attr->repeats[axis_index];
      return true;
    }
  }

  std::vector<NodePtr> peer_nodes;
  if (search_forward) {
    for (size_t i = 0U; i < node->GetAllOutDataAnchorsSize(); ++i) {
      peer_nodes.clear();
      if (asc_adapt::GetPeerInNodes(node, peer_nodes, static_cast<int32_t>(i)) != SUCCESS) {
        return false;
      }
      for (const auto &peer_node : peer_nodes) {
        if (FindDirectionalNonUnitRepeatByAxisId(peer_node, axis_id, search_forward, visited, repeat)) {
          return true;
        }
      }
    }
    return false;
  }

  if (asc_adapt::GetPeerOutNodes(node, peer_nodes) != SUCCESS) {
    return false;
  }
  for (const auto &peer_node : peer_nodes) {
    if (FindDirectionalNonUnitRepeatByAxisId(peer_node, axis_id, search_forward, visited, repeat)) {
      return true;
    }
  }
  return false;
}

inline bool FindAnchorRepeatByRepeatIndex(const NodePtr &node, const std::vector<int64_t> &axis,
                                          const size_t repeat_index, const bool search_forward, Expression &repeat) {
  if (repeat_index >= axis.size()) {
    return false;
  }
  std::unordered_set<const void *> visited;
  return FindDirectionalNonUnitRepeatByAxisId(node, axis[repeat_index], search_forward, visited, repeat);
}

inline void BuildUnitRepeatGapIndexes(const std::vector<Expression> &repeats,
                                      std::vector<std::vector<size_t>> &gap_indexes, std::vector<size_t> &gap_starts) {
  gap_indexes.assign(RemoveUnitRepeats(repeats).size() + 1U, {});
  gap_starts.assign(gap_indexes.size(), 0U);
  size_t gap_idx = 0U;
  for (size_t i = 0U; i < repeats.size(); ++i) {
    if (BackendUtils::IsEqOne(repeats[i])) {
      gap_indexes[gap_idx].push_back(i);
      continue;
    }
    ++gap_idx;
    if (gap_idx < gap_starts.size()) {
      gap_starts[gap_idx] = i + 1U;
    }
  }
}

inline void BuildUnitRepeatGapIndexes(const std::vector<Expression> &repeats,
                                      std::vector<std::vector<size_t>> &gap_indexes) {
  std::vector<size_t> unused_gap_starts;
  BuildUnitRepeatGapIndexes(repeats, gap_indexes, unused_gap_starts);
}

inline void RemoveInsertIndexesInRange(const size_t range_start, const size_t range_end,
                                       std::vector<size_t> &insert_indexes) {
  insert_indexes.erase(std::remove_if(insert_indexes.begin(), insert_indexes.end(),
                                      [range_start, range_end](const size_t insert_index) {
                                        return (insert_index >= range_start) && (insert_index < range_end);
                                      }),
                       insert_indexes.end());
}

inline bool CollectAnchorAlignedInsertIndexes(const NodePtr &node, const std::vector<int64_t> &axis,
                                              const std::vector<size_t> &current_gap_indexes, const bool search_forward,
                                              const NodePtr &peer_node, const std::vector<int64_t> &peer_axis,
                                              const std::vector<size_t> &peer_gap_indexes,
                                              const bool peer_search_forward, const size_t gap_start,
                                              std::vector<size_t> &aligned_insert_indexes) {
  constexpr size_t kInvalidIndex = static_cast<size_t>(-1);
  std::vector<size_t> peer_offset_to_current_index(peer_gap_indexes.size(), kInvalidIndex);
  for (const auto current_index : current_gap_indexes) {
    Expression current_anchor;
    if (!FindAnchorRepeatByRepeatIndex(node, axis, current_index, search_forward, current_anchor)) {
      continue;
    }
    for (size_t peer_offset = 0U; peer_offset < peer_gap_indexes.size(); ++peer_offset) {
      Expression peer_anchor;
      if (FindAnchorRepeatByRepeatIndex(peer_node, peer_axis, peer_gap_indexes[peer_offset], peer_search_forward,
                                        peer_anchor) &&
          IsSameRepeat(current_anchor, peer_anchor)) {
        peer_offset_to_current_index[peer_offset] = current_index;
        break;
      }
    }
  }
  if (std::all_of(peer_offset_to_current_index.begin(), peer_offset_to_current_index.end(),
                  [](const size_t current_index) { return current_index == kInvalidIndex; })) {
    return false;
  }

  aligned_insert_indexes.clear();
  for (size_t peer_offset = 0U; peer_offset < peer_offset_to_current_index.size(); ++peer_offset) {
    if (peer_offset_to_current_index[peer_offset] != kInvalidIndex) {
      continue;
    }
    size_t insert_index = gap_start + current_gap_indexes.size();
    for (size_t next_offset = peer_offset + 1U; next_offset < peer_offset_to_current_index.size(); ++next_offset) {
      if (peer_offset_to_current_index[next_offset] != kInvalidIndex) {
        insert_index = peer_offset_to_current_index[next_offset];
        break;
      }
    }
    aligned_insert_indexes.push_back(insert_index);
  }
  return true;
}

inline void AdjustInsertIndexesByAnchorAlignment(const NodePtr &node, const std::vector<int64_t> &axis,
                                                 const std::vector<Expression> &repeats, const bool search_forward,
                                                 const NodePtr &peer_node, const std::vector<int64_t> &peer_axis,
                                                 const std::vector<Expression> &peer_repeats,
                                                 const bool peer_search_forward, std::vector<size_t> &insert_indexes) {
  std::vector<std::vector<size_t>> gap_indexes;
  std::vector<size_t> gap_starts;
  std::vector<std::vector<size_t>> peer_gap_indexes;
  BuildUnitRepeatGapIndexes(repeats, gap_indexes, gap_starts);
  BuildUnitRepeatGapIndexes(peer_repeats, peer_gap_indexes);
  if (gap_indexes.size() != peer_gap_indexes.size()) {
    return;
  }

  for (size_t gap_idx = 0U; gap_idx < gap_indexes.size(); ++gap_idx) {
    if ((peer_gap_indexes[gap_idx].size() <= gap_indexes[gap_idx].size()) || (peer_gap_indexes[gap_idx].size() <= 1U)) {
      continue;
    }
    std::vector<size_t> aligned_insert_indexes;
    if (!CollectAnchorAlignedInsertIndexes(node, axis, gap_indexes[gap_idx], search_forward, peer_node, peer_axis,
                                           peer_gap_indexes[gap_idx], peer_search_forward, gap_starts[gap_idx],
                                           aligned_insert_indexes)) {
      continue;
    }
    RemoveInsertIndexesInRange(gap_starts[gap_idx], gap_starts[gap_idx] + peer_gap_indexes[gap_idx].size(),
                               insert_indexes);
    insert_indexes.insert(insert_indexes.end(), aligned_insert_indexes.begin(), aligned_insert_indexes.end());
  }
  std::sort(insert_indexes.begin(), insert_indexes.end());
}

inline Status CollectVerticalRelationInsertIndexes(const NodePtr &node1, const NodePtr &node2,
                                                   const NodeFuseInfo &fuse_info,
                                                   const std::vector<af::ReshapeAxisChangeInfo> &node1_axis_changes,
                                                   const std::vector<af::ReshapeAxisChangeInfo> &node2_axis_changes,
                                                   InsertIndexMergeState &node1_insert_index_state,
                                                   InsertIndexMergeState &node2_insert_index_state,
                                                   std::vector<size_t> &node1_insert_indexes,
                                                   std::vector<size_t> &node2_insert_indexes) {
  AscGraphAxisMapping axis_mapping(false);
  for (const auto &relation : fuse_info.GetNode1ToNode2LinkMap()) {
    std::vector<Expression> node1_output_repeats;
    std::vector<Expression> node2_input_repeats;
    std::vector<Expression> dims;
    std::vector<int64_t> node1_output_axis;
    std::vector<int64_t> node2_input_axis;
    std::vector<Expression> target_repeats;
    std::vector<size_t> relation_node1_insert_indexes;
    std::vector<size_t> relation_node2_insert_indexes;
    GE_ASSERT_SUCCESS(
        axis_mapping.GetPreNodeAttrs(node2, relation.second, dims, node1_output_axis, node1_output_repeats));
    GE_ASSERT_SUCCESS(axis_mapping.GetCurNodeAttrs(node2, relation.second, node2_input_axis, node2_input_repeats));

    GE_ASSERT_SUCCESS(BuildRelationTargetRepeats(node1_output_repeats, node2_input_repeats, node1_axis_changes,
                                                 node2_axis_changes, target_repeats));
    GE_ASSERT_SUCCESS(
        CollectInsertIndexesByTargetRepeats(node1_output_repeats, target_repeats, relation_node1_insert_indexes));
    GE_ASSERT_SUCCESS(
        CollectInsertIndexesByTargetRepeats(node2_input_repeats, target_repeats, relation_node2_insert_indexes));
    AdjustInsertIndexesByAnchorAlignment(node1, node1_output_axis, node1_output_repeats, false, node2, node2_input_axis,
                                         node2_input_repeats, true, relation_node1_insert_indexes);
    AdjustInsertIndexesByAnchorAlignment(node2, node2_input_axis, node2_input_repeats, true, node1, node1_output_axis,
                                         node1_output_repeats, false, relation_node2_insert_indexes);
    GE_ASSERT_SUCCESS(MergeRelationInsertIndexes(node1_output_repeats.size(), relation_node1_insert_indexes,
                                                 node1_insert_index_state, node1_insert_indexes));
    GE_ASSERT_SUCCESS(MergeRelationInsertIndexes(node2_input_repeats.size(), relation_node2_insert_indexes,
                                                 node2_insert_index_state, node2_insert_indexes));
    GELOGD(
        "collect vertical reshape insert indexes, node1 %s out %d repeats %s indexes %s, node2 %s in %d repeats %s "
        "indexes %s, target repeats %s.",
        node1->GetName().c_str(), relation.first, AutofuseUtils::VectorToStr(node1_output_repeats).c_str(),
        AutofuseUtils::VectorToStr(node1_insert_indexes).c_str(), node2->GetName().c_str(), relation.second,
        AutofuseUtils::VectorToStr(node2_input_repeats).c_str(),
        AutofuseUtils::VectorToStr(node2_insert_indexes).c_str(), AutofuseUtils::VectorToStr(target_repeats).c_str());
  }
  return SUCCESS;
}

inline Status CollectCommonInputRelationInsertIndexes(const NodePtr &node1, const NodePtr &node2,
                                                      const NodeFuseInfo &fuse_info,
                                                      const std::vector<af::ReshapeAxisChangeInfo> &node1_axis_changes,
                                                      const std::vector<af::ReshapeAxisChangeInfo> &node2_axis_changes,
                                                      InsertIndexMergeState &node1_insert_index_state,
                                                      InsertIndexMergeState &node2_insert_index_state,
                                                      std::vector<size_t> &node1_insert_indexes,
                                                      std::vector<size_t> &node2_insert_indexes) {
  AscGraphAxisMapping axis_mapping(false);
  for (const auto &relation : fuse_info.GetSameInputMap()) {
    std::vector<Expression> node1_input_repeats;
    std::vector<Expression> node2_input_repeats;
    std::vector<int64_t> node1_input_axis;
    std::vector<int64_t> node2_input_axis;
    std::vector<Expression> target_repeats;
    std::vector<size_t> relation_node1_insert_indexes;
    std::vector<size_t> relation_node2_insert_indexes;
    GE_ASSERT_SUCCESS(axis_mapping.GetCurNodeAttrs(node1, relation.first, node1_input_axis, node1_input_repeats));
    GE_ASSERT_SUCCESS(axis_mapping.GetCurNodeAttrs(node2, relation.second, node2_input_axis, node2_input_repeats));

    GE_ASSERT_SUCCESS(BuildRelationTargetRepeats(node1_input_repeats, node2_input_repeats, node1_axis_changes,
                                                 node2_axis_changes, target_repeats));
    GE_ASSERT_SUCCESS(
        CollectInsertIndexesByTargetRepeats(node1_input_repeats, target_repeats, relation_node1_insert_indexes));
    GE_ASSERT_SUCCESS(
        CollectInsertIndexesByTargetRepeats(node2_input_repeats, target_repeats, relation_node2_insert_indexes));
    AdjustInsertIndexesByAnchorAlignment(node1, node1_input_axis, node1_input_repeats, true, node2, node2_input_axis,
                                         node2_input_repeats, true, relation_node1_insert_indexes);
    AdjustInsertIndexesByAnchorAlignment(node2, node2_input_axis, node2_input_repeats, true, node1, node1_input_axis,
                                         node1_input_repeats, true, relation_node2_insert_indexes);
    GE_ASSERT_SUCCESS(MergeRelationInsertIndexes(node1_input_repeats.size(), relation_node1_insert_indexes,
                                                 node1_insert_index_state, node1_insert_indexes));
    GE_ASSERT_SUCCESS(MergeRelationInsertIndexes(node2_input_repeats.size(), relation_node2_insert_indexes,
                                                 node2_insert_index_state, node2_insert_indexes));
    GELOGD(
        "collect common-input reshape insert indexes, node1 %s in %d repeats %s indexes %s, node2 %s in %d repeats "
        "%s indexes %s, target repeats %s.",
        node1->GetName().c_str(), relation.first, AutofuseUtils::VectorToStr(node1_input_repeats).c_str(),
        AutofuseUtils::VectorToStr(node1_insert_indexes).c_str(), node2->GetName().c_str(), relation.second,
        AutofuseUtils::VectorToStr(node2_input_repeats).c_str(),
        AutofuseUtils::VectorToStr(node2_insert_indexes).c_str(), AutofuseUtils::VectorToStr(target_repeats).c_str());
  }
  return SUCCESS;
}

inline Status CollectRelationInsertIndexes(const NodePtr &node1, const NodePtr &node2, const NodeFuseInfo &fuse_info,
                                           const std::vector<af::ReshapeAxisChangeInfo> &node1_axis_changes,
                                           const std::vector<af::ReshapeAxisChangeInfo> &node2_axis_changes,
                                           std::vector<size_t> &node1_insert_indexes,
                                           std::vector<size_t> &node2_insert_indexes) {
  node1_insert_indexes.clear();
  node2_insert_indexes.clear();
  InsertIndexMergeState node1_insert_index_state;
  InsertIndexMergeState node2_insert_index_state;
  GE_ASSERT_SUCCESS(CollectVerticalRelationInsertIndexes(
      node1, node2, fuse_info, node1_axis_changes, node2_axis_changes, node1_insert_index_state,
      node2_insert_index_state, node1_insert_indexes, node2_insert_indexes));
  GE_ASSERT_SUCCESS(CollectCommonInputRelationInsertIndexes(
      node1, node2, fuse_info, node1_axis_changes, node2_axis_changes, node1_insert_index_state,
      node2_insert_index_state, node1_insert_indexes, node2_insert_indexes));
  return SUCCESS;
}

// Backend can-fuse 前置处理接口：根据 node1/node2 的真实连接关系补齐 AscBackend/FusedAscBackend 的
// reshape no-op 单位轴，避免后续比较或合并属性时使用不一致的轴空间。
inline Status CompletePairReshapeAxes(const NodePtr &node1, const NodePtr &node2, const NodeFuseInfo &fuse_info) {
  std::vector<af::ReshapeAxisChangeInfo> node1_axis_changes;
  std::vector<af::ReshapeAxisChangeInfo> node2_axis_changes;
  GE_ASSERT_SUCCESS(CollectNodeReshapeAxisChanges(node1, node1_axis_changes));
  GE_ASSERT_SUCCESS(CollectNodeReshapeAxisChanges(node2, node2_axis_changes));
  if (node1_axis_changes.empty() && node2_axis_changes.empty()) {
    return SUCCESS;
  }

  std::vector<size_t> node1_insert_indexes;
  std::vector<size_t> node2_insert_indexes;
  GE_ASSERT_SUCCESS(CollectRelationInsertIndexes(node1, node2, fuse_info, node1_axis_changes, node2_axis_changes,
                                                 node1_insert_indexes, node2_insert_indexes));
  GE_ASSERT_SUCCESS(PadNodeLeadingUnitAxisByInsertIndexes(node1, node1_insert_indexes));
  GE_ASSERT_SUCCESS(PadNodeLeadingUnitAxisByInsertIndexes(node2, node2_insert_indexes));
  std::vector<af::ReshapeAxisChangeInfo> merged_axis_changes;
  GE_ASSERT_SUCCESS(CollectNodeReshapeAxisChanges(node1, merged_axis_changes));
  GE_ASSERT_SUCCESS(CollectNodeReshapeAxisChanges(node2, merged_axis_changes));
  GE_ASSERT_SUCCESS(SaveMergedReshapeAxisChanges(node1, merged_axis_changes));
  GE_ASSERT_SUCCESS(SaveMergedReshapeAxisChanges(node2, merged_axis_changes));
  return SUCCESS;
}

// Backend 融合属性继承接口：两个 backend 节点融合成新 backend 节点时，将两侧保存的 reshape 轴变化
// 元数据继承到新节点。此处不重映射 axis id，后续由 FlushReshapeAxisChanges 统一刷新。
inline void InheritReshapeAxisChanges(AutofuseInnerAttrs &attr_new, const AutofuseInnerAttrs &attr1,
                                      const AutofuseInnerAttrs &attr2) {
  attr_new.reshape_axis_changes.clear();
  for (const auto &axis_change : attr1.reshape_axis_changes) {
    AppendUniqueReshapeAxisChange(axis_change, attr_new.reshape_axis_changes);
  }
  for (const auto &axis_change : attr2.reshape_axis_changes) {
    AppendUniqueReshapeAxisChange(axis_change, attr_new.reshape_axis_changes);
  }
}

inline Status FlushReshapeAxisByRepeats(const NodePtr &node, const NodePtr &asc_node,
                                        const std::vector<int64_t> &axis_before_Flush,
                                        const std::vector<int64_t> &axis_after_Flush,
                                        const std::vector<Expression> &reshape_repeats,
                                        std::vector<int64_t> &reshape_axis, const char *info_name) {
  if (axis_before_Flush.empty() || axis_after_Flush.empty() || reshape_axis.empty()) {
    return SUCCESS;
  }

  GE_ASSERT_TRUE(axis_before_Flush.size() == axis_after_Flush.size(),
                 "axis_before_Flush size %zu must equal to axis_after_Flush size %zu", axis_before_Flush.size(),
                 axis_after_Flush.size());
  GE_ASSERT_TRUE(reshape_axis.size() == reshape_repeats.size(), "reshape axis size %zu must equal repeats size %zu",
                 reshape_axis.size(), reshape_repeats.size());

  std::unordered_map<int64_t, size_t> axis_before_to_index;
  for (size_t i = 0U; i < axis_before_Flush.size(); ++i) {
    axis_before_to_index[axis_before_Flush[i]] = i;
  }

  std::vector<int64_t> updated_axis;
  updated_axis.reserve(reshape_axis.size());
  for (size_t i = 0U; i < reshape_axis.size(); ++i) {
    const auto axis = reshape_axis[i];
    auto it = axis_before_to_index.find(axis);
    if (it != axis_before_to_index.end()) {
      updated_axis.push_back(axis_after_Flush[it->second]);
      continue;
    }
    if (BackendUtils::IsEqOne(reshape_repeats[i]) && (reshape_axis.size() == axis_after_Flush.size())) {
      updated_axis.push_back(axis_after_Flush[i]);
      continue;
    }
    GELOGW("Axis %ld in %s not found in axis_before_Flush for asc_node %s, keep original value", axis, info_name,
           asc_node->GetName().c_str());
    updated_axis.push_back(axis);
  }

  GELOGD("Flush %s for node %s, asc_node %s, before: %s, after: %s", info_name, node->GetName().c_str(),
         asc_node->GetName().c_str(), AutofuseUtils::VectorToStr(reshape_axis).c_str(),
         AutofuseUtils::VectorToStr(updated_axis).c_str());
  reshape_axis = updated_axis;
  return SUCCESS;
}

// Backend reshape 轴元数据刷新接口：FlushAscSubGraphAxisInfo 转换 AscGraph 轴 id 后，同步刷新 backend
// 节点中保存的 before_axis/after_axis，避免后续 can-fuse 或后处理读取到过期 axis id。
inline Status FlushReshapeAxisChanges(const NodePtr &node, const NodePtr &asc_node,
                                      const std::vector<int64_t> &axis_before_Flush,
                                      const std::vector<int64_t> &axis_after_Flush) {
  auto autofuse_attr = BackendUtils::GetNodeAutoFuseAttr(node);
  GE_ASSERT_NOTNULL(autofuse_attr);
  auto reshape_axis_changes = autofuse_attr->GetReshapeAxisChanges();
  if (reshape_axis_changes.empty()) {
    return SUCCESS;
  }

  for (auto &change : reshape_axis_changes) {
    GE_ASSERT_SUCCESS(FlushReshapeAxisByRepeats(node, asc_node, axis_before_Flush, axis_after_Flush,
                                                change.before_repeats, change.before_axis, "reshape_before_axis"));
    GE_ASSERT_SUCCESS(FlushReshapeAxisByRepeats(node, asc_node, axis_before_Flush, axis_after_Flush,
                                                change.after_repeats, change.after_axis, "reshape_after_axis"));
  }
  autofuse_attr->SetReshapeAxisChanges(reshape_axis_changes);

  return SUCCESS;
}

}  // namespace asc_adapt
}  // namespace ge
#endif  // AUTOFUSE_POST_PROCESS_SCHEDULER_ADAPTER_ADAPTION_RESHAPE_AXIS_PADDING_H
