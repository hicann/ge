/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "slices_to_split_pass.h"
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <algorithm>
#include <functional>
#include "graph/utils/graph_utils.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "common/checker.h"
#include "utils/autofuse_utils.h"
#include "operator_reg.h"
#include "debug/ge_util.h"

namespace ge {
namespace {
constexpr auto kOpTypeSlice = "Slice";
constexpr auto kOpTypeSliceD = "SliceD";
constexpr const char *kOpTypeSplitV = "SplitV";
constexpr size_t kMinSlicesToConvert = 3UL; // 至少需要3个slice才值得转换为SplitV

using SliceKey = std::pair<int64_t, int32_t>;

struct SliceKeyHash {
  size_t operator()(const SliceKey &key) const noexcept {
    return std::hash<int64_t>{}(key.first) ^ (std::hash<int32_t>{}(key.second) << 1);
  }
};

struct ControlEdge {
  NodePtr src;
  NodePtr dst;
};

struct ControlEdgeHash {
  size_t operator()(const ControlEdge &e) const noexcept {
    return std::hash<NodePtr>{}(e.src) ^ (std::hash<NodePtr>{}(e.dst) << 1);
  }
};

struct ControlEdgeEqual {
  bool operator()(const ControlEdge &a, const ControlEdge &b) const noexcept {
    return a.src == b.src && a.dst == b.dst;
  }
};

struct SliceInfo {
  NodePtr slice_node;
  std::vector<int64_t> offsets;
  std::vector<int64_t> sizes;
  int32_t input_output_index = 0;
  int64_t split_dim = -1;
  int64_t start = 0;
  int64_t end = 0;
};

bool IsSliceNode(const NodePtr &node) {
  return node != nullptr &&
         (node->GetType() == kOpTypeSlice || node->GetType() == kOpTypeSliceD);
}

bool GetSliceInfo(const NodePtr &slice_node, SliceInfo &info) {
  info.slice_node = slice_node;
  if (AutofuseUtils::GetListIntByInputOrAttr(slice_node, info.offsets, "offsets", "offsets") != SUCCESS) {
    GELOGD("Failed to get offsets from slice node: %s", slice_node->GetNamePtr());
    return false;
  }

  if (AutofuseUtils::GetListIntByInputOrAttr(slice_node, info.sizes, "size", "size") != SUCCESS) {
    GELOGD("Failed to get sizes from slice node: %s", slice_node->GetNamePtr());
    return false;
  }

  if (info.offsets.size() != info.sizes.size()) {
    GELOGE(FAILED, "Offsets and sizes size mismatch for slice: %s", slice_node->GetNamePtr());
    return false;
  }

  return true;
}

// 检查除目标维度外，其他维度是否相同（offset=0 且 size 相同）
bool IsValidSplitDimension(const SliceInfo &slice, const SliceInfo &first_slice,
                           size_t target_dim, size_t rank, bool &out_has_variation) {
  for (size_t d = 0; d < rank; ++d) {
    if (d == target_dim) {
      if (!out_has_variation && slice.offsets[d] != first_slice.offsets[d]) {
        out_has_variation = true;
      }
      continue;
    }
    if (slice.offsets[d] != 0 || slice.sizes[d] != first_slice.sizes[d]) {
      return false;
    }
  }
  return true;
}

// 查找 slice 变化的维度
bool FindSplitDimension(const std::vector<SliceInfo> &slices, int64_t &split_dim) {
  if (slices.empty()) {
    return false;
  }

  const auto rank = slices[0].offsets.size();
  for (size_t dim = 0; dim < rank; ++dim) {
    bool has_variation = false;
    bool valid_dim = true;
    for (const auto &slice: slices) {
      if (!IsValidSplitDimension(slice, slices[0], dim, rank, has_variation)) {
        valid_dim = false;
        break;
      }
    }

    if (valid_dim && has_variation) {
      split_dim = static_cast<int64_t>(dim);
      return true;
    }
  }

  return false;
}

// 按 split 维度的 offset 排序
void SortSlicesByDimension(std::vector<SliceInfo> &slices, int64_t split_dim) {
  std::sort(slices.begin(), slices.end(),
            [split_dim](const SliceInfo &a, const SliceInfo &b) {
              return a.offsets[split_dim] < b.offsets[split_dim];
            });
}

// 查找 slice 变化的维度，并按该维度排序
bool FindSplitDimAndSort(const std::vector<SliceInfo> &slices, int64_t &split_dim,
                         std::vector<SliceInfo> &sorted_slices) {
  if (!FindSplitDimension(slices, split_dim)) {
    return false;
  }

  sorted_slices = slices;
  SortSlicesByDimension(sorted_slices, split_dim);
  return true;
}

// 获取输入节点在指定维度的尺寸（仅支持静态shape）
bool GetDimensionSize(const NodePtr &input_node, int32_t output_idx,
                      int64_t split_dim, int64_t &out_dim_size) {
  out_dim_size = -1;
  if (!input_node || !input_node->GetOpDesc()) {
    return false;
  }

  auto &shape = input_node->GetOpDesc()->GetOutputDesc(output_idx).GetShape();
  if (shape.GetDimNum() > static_cast<size_t>(split_dim)) {
    out_dim_size = shape.GetDim(split_dim);
  }

  return out_dim_size >= 0;
}

// 验证单个切片的范围并设置 start/end
bool ValidateSliceRange(SliceInfo &slice, int64_t split_dim, int64_t dim_size, size_t idx, size_t total) {
  slice.split_dim = split_dim;
  slice.start = slice.offsets[split_dim];

  const int64_t size = slice.sizes[split_dim];
  if (size == -1) {
    if (idx != total - 1) {
      GELOGD("Only last slice can have size=-1, but slice %zu has size=-1", idx);
      return false;
    }
    slice.end = dim_size;
  } else if (size <= 0) {
    GELOGD("Slice %zu has invalid size=%ld (must be > 0 or -1)", idx, size);
    return false;
  } else {
    slice.end = slice.start + size;
  }

  if (slice.end > dim_size) {
    GELOGD("Slice %zu exceeds dimension: end=%ld > dim_size=%ld", idx, slice.end, dim_size);
    return false;
  }

  return true;
}

// 检查切片是否连续且覆盖整个维度
bool CheckSliceContinuity(std::vector<SliceInfo> &slices, int64_t split_dim, int64_t dim_size) {
  if (slices[0].offsets[split_dim] != 0) {
    GELOGD("Slices must start from 0, but start=%ld", slices[0].offsets[split_dim]);
    return false;
  }

  for (size_t i = 0; i < slices.size(); ++i) {
    if (!ValidateSliceRange(slices[i], split_dim, dim_size, i, slices.size())) {
      return false;
    }

    if (i > 0 && slices[i].start != slices[i - 1].end) {
      GELOGD("Slices not continuous: slice[%zu].start=%ld != slice[%zu].end=%ld",
             i, slices[i].start, i - 1, slices[i - 1].end);
      return false;
    }
  }

  if (slices.back().end != dim_size) {
    GELOGD("Slices don't cover entire dimension: end=%ld != dim_size=%ld",
           slices.back().end, dim_size);
    return false;
  }

  return true;
}

bool CheckContinuity(std::vector<SliceInfo> &slices, int64_t split_dim, const NodePtr &input_node) {
  if (slices.empty() || split_dim < 0) {
    GELOGE(FAILED, "Invalid input: slices.empty()=%d, split_dim=%ld",
           slices.empty(), split_dim);
    return false;
  }

  const int32_t output_idx = slices[0].input_output_index;
  int64_t dim_size = -1;
  if (!GetDimensionSize(input_node, output_idx, split_dim, dim_size)) {
    GELOGD("Unsupported dynamic shape or failed to get dim_size (output_idx=%d, dim_size=%ld)",
           output_idx, dim_size);
    return false;
  }

  if (!CheckSliceContinuity(slices, split_dim, dim_size)) {
    return false;
  }

  GELOGI("SlicesToSplit: output_idx=%d, split_dim=%ld, dim_size=%ld, %zu slices",
         output_idx, split_dim, dim_size, slices.size());
  return true;
}

NodePtr CreateConstNode(const ComputeGraphPtr &graph, const std::vector<int64_t> &data,
                        const std::string &name_suffix) {
  GeShape shape({static_cast<int64_t>(data.size())});
  GeTensorDesc desc(shape, FORMAT_ND, DT_INT64);
  desc.SetOriginShape(shape);
  desc.SetOriginDataType(DT_INT64);
  desc.SetOriginFormat(FORMAT_ND);

  const auto *ptr = reinterpret_cast<const uint8_t *>(data.data());
  std::vector<uint8_t> buffer(ptr, ptr + data.size() * sizeof(int64_t));
  auto tensor = ComGraphMakeShared<GeTensor>(desc, buffer);
  GE_ASSERT_NOTNULL(tensor);

  auto op_desc = ge::OpDescUtils::CreateConstOpZeroCopy(tensor);
  GE_ASSERT_NOTNULL(op_desc);
  op_desc->SetName(name_suffix);

  auto node = graph->AddNode(op_desc);
  GE_ASSERT_NOTNULL(node);
  return node;
}

inline graphStatus AddDataEdge(const NodePtr &src, int32_t src_idx,
                               const NodePtr &dst, int32_t dst_idx) {
  return GraphUtils::AddEdge(src->GetOutDataAnchor(src_idx), dst->GetInDataAnchor(dst_idx));
}

NodePtr CreateSplitNode(const ComputeGraphPtr &graph, const NodePtr &input_node,
                        const std::vector<SliceInfo> &slices, int64_t split_dim) {
  auto op = ge::OperatorFactory::CreateOperator(("SlicesToSplit_" + input_node->GetName()).c_str(), kOpTypeSplitV);
  const auto num_outputs = static_cast<uint32_t>(slices.size());
  op.DynamicOutputRegister("y", num_outputs);
  op.SetAttr("num_split", num_outputs);
  op.BreakConnect();

  auto split_node = graph->AddNode(ge::OpDescUtils::GetOpDescFromOperator(op));
  GE_ASSERT_NOTNULL(split_node);

  // 获取实际连接的输出索引
  const int32_t input_output_idx = slices.empty() ? 0 : slices[0].input_output_index;
  // 连接输入边（使用实际输出索引）
  if (AddDataEdge(input_node, input_output_idx, split_node, 0) != GRAPH_SUCCESS) {
    return nullptr;
  }

  // 计算 size_splits
  std::vector<int64_t> size_splits;
  size_splits.reserve(slices.size());
  for (const auto &s: slices) {
    size_splits.push_back(s.end - s.start);
  }

  const std::string base_name = "SlicesToSplit_" + input_node->GetName();
  auto size_splits_const = CreateConstNode(graph, size_splits, base_name + "_size_splits");
  if (!size_splits_const || AddDataEdge(size_splits_const, 0, split_node, 1) != GRAPH_SUCCESS) {
    return nullptr;
  }

  auto split_dim_const = CreateConstNode(graph, {split_dim}, base_name + "_split_dim");
  if (!split_dim_const || AddDataEdge(split_dim_const, 0, split_node, 2) != GRAPH_SUCCESS) {
    return nullptr;
  }

  // 设置输入输出描述
  auto split_op_desc = split_node->GetOpDesc();
  GE_ASSERT_NOTNULL(split_op_desc);

  split_op_desc->UpdateInputDesc(0, input_node->GetOpDesc()->GetOutputDesc(input_output_idx));
  split_op_desc->UpdateInputDesc(1, size_splits_const->GetOpDesc()->GetOutputDesc(0));
  split_op_desc->UpdateInputDesc(2, split_dim_const->GetOpDesc()->GetOutputDesc(0));
  split_op_desc->SetIsInputConst({false, true, true});

  for (size_t i = 0; i < slices.size(); ++i) {
    if (split_op_desc->UpdateOutputDesc(i, slices[i].slice_node->GetOpDesc()->GetOutputDesc(0)) != GRAPH_SUCCESS) {
      return nullptr;
    }
  }

  return split_node;
}

// 收集所有slice节点的控制边
void CollectControlEdges(const std::vector<SliceInfo> &slices,
                         std::unordered_set<ControlEdge, ControlEdgeHash, ControlEdgeEqual> &in_ctrl_edges,
                         std::unordered_set<ControlEdge, ControlEdgeHash, ControlEdgeEqual> &out_ctrl_edges) {
  for (const auto &slice: slices) {
    for (const auto &ctrl_in: slice.slice_node->GetInControlNodes()) {
      in_ctrl_edges.insert({ctrl_in, slice.slice_node});
    }
    for (const auto &ctrl_out: slice.slice_node->GetOutControlNodes()) {
      out_ctrl_edges.insert({slice.slice_node, ctrl_out});
    }
  }
}

// 迁移数据边从slice节点到split节点
graphStatus TransferDataEdges(const NodePtr &split_node, const std::vector<SliceInfo> &slices) {
  for (size_t i = 0; i < slices.size(); ++i) {
    const auto &slice = slices[i];
    for (const auto &out_anchor: slice.slice_node->GetAllOutDataAnchors()) {
      for (const auto &in_anchor: out_anchor->GetPeerInDataAnchors()) {
        GraphUtils::RemoveEdge(out_anchor, in_anchor);
        if (GraphUtils::AddEdge(split_node->GetOutDataAnchor(static_cast<int32_t>(i)), in_anchor) != GRAPH_SUCCESS) {
          return GRAPH_FAILED;
        }
      }
    }
  }
  return GRAPH_SUCCESS;
}

// 迁移控制边到新节点
graphStatus TransferControlEdges(
  const NodePtr &split_node,
  const std::unordered_set<ControlEdge, ControlEdgeHash, ControlEdgeEqual> &ctrl_edges,
  bool is_input) {
  for (const auto &e: ctrl_edges) {
    if (is_input) {
      GraphUtils::RemoveEdge(e.src->GetOutControlAnchor(), e.dst->GetInControlAnchor());
      if (GraphUtils::AddEdge(e.src->GetOutControlAnchor(), split_node->GetInControlAnchor()) != GRAPH_SUCCESS) {
        return GRAPH_FAILED;
      }
    } else {
      GraphUtils::RemoveEdge(e.src->GetOutControlAnchor(), e.dst->GetInControlAnchor());
      if (GraphUtils::AddEdge(split_node->GetOutControlAnchor(), e.dst->GetInControlAnchor()) != GRAPH_SUCCESS) {
        return GRAPH_FAILED;
      }
    }
  }
  return GRAPH_SUCCESS;
}

// 删除slice节点
void RemoveSliceNodes(const ComputeGraphPtr &graph, const std::vector<SliceInfo> &slices) {
  for (const auto &slice: slices) {
    NodeUtils::UnlinkAll(*slice.slice_node);
    graph->RemoveNode(slice.slice_node);
  }
}

graphStatus ReplaceSlicesWithSplit(const ComputeGraphPtr &graph,
                                   const std::vector<SliceInfo> &slices,
                                   const NodePtr &input_node, int64_t split_dim) {
  auto split_node = CreateSplitNode(graph, input_node, slices, split_dim);
  if (!split_node) {
    return GRAPH_FAILED;
  }

  std::unordered_set<ControlEdge, ControlEdgeHash, ControlEdgeEqual> in_ctrl_edges, out_ctrl_edges;
  CollectControlEdges(slices, in_ctrl_edges, out_ctrl_edges);

  if (TransferDataEdges(split_node, slices) != GRAPH_SUCCESS) {
    return GRAPH_FAILED;
  }

  if (TransferControlEdges(split_node, in_ctrl_edges, true) != GRAPH_SUCCESS) {
    return GRAPH_FAILED;
  }

  if (TransferControlEdges(split_node, out_ctrl_edges, false) != GRAPH_SUCCESS) {
    return GRAPH_FAILED;
  }

  RemoveSliceNodes(graph, slices);

  GELOGI("SlicesToSplit: replaced %zu slices with %s (in-ctrl=%zu, out-ctrl=%zu)",
         slices.size(), split_node->GetName().c_str(),
         in_ctrl_edges.size(), out_ctrl_edges.size());
  return GRAPH_SUCCESS;
}

// 收集并按 (input_node_id, output_index) 分组 Slice 节点
void CollectSliceNodes(const ComputeGraphPtr &graph,
                       std::unordered_map<SliceKey, std::pair<NodePtr, std::vector<SliceInfo> >, SliceKeyHash> &
                       input_to_slices) {
  for (const auto &node: graph->GetDirectNode()) {
    if (!IsSliceNode(node)) {
      continue;
    }

    SliceInfo info;
    if (!GetSliceInfo(node, info)) {
      GELOGD("Failed to get slice info from node: %s, skip", node->GetNamePtr());
      continue;
    }

    auto input_anchor = node->GetInDataAnchor(0);
    if (!input_anchor || !input_anchor->GetPeerOutAnchor()) {
      GELOGD("Slice node %s has no valid input anchor", node->GetNamePtr());
      continue;
    }

    auto input_node = NodeUtils::GetInDataNodeByIndex(*node, 0);
    if (!input_node) {
      continue;
    }

    info.input_output_index = input_anchor->GetPeerOutAnchor()->GetIdx();
    SliceKey key{input_node->GetOpDesc()->GetId(), info.input_output_index};
    input_to_slices[key].first = input_node;
    input_to_slices[key].second.push_back(info);
  }
}

// 获取排序后的分组 key 列表，保证确定性处理顺序
std::vector<SliceKey> GetSortedSliceKeys(
  const std::unordered_map<SliceKey, std::pair<NodePtr, std::vector<SliceInfo> >, SliceKeyHash> &input_to_slices) {
  std::vector<SliceKey> sorted_keys;
  sorted_keys.reserve(input_to_slices.size());
  for (const auto &entry: input_to_slices) {
    sorted_keys.push_back(entry.first);
  }
  std::sort(sorted_keys.begin(), sorted_keys.end(),
            [](const SliceKey &a, const SliceKey &b) {
              return a.first < b.first || (a.first == b.first && a.second < b.second);
            });
  return sorted_keys;
}

// 处理单组 Slice 节点，返回是否替换成功
bool ProcessSliceGroup(const ComputeGraphPtr &graph, const NodePtr &input_node,
                       const std::vector<SliceInfo> &slices) {
  if (slices.size() < kMinSlicesToConvert) {
    GELOGD("Only %zu slices (need at least %zu), skip conversion",
           slices.size(), kMinSlicesToConvert);
    return false;
  }

  int64_t split_dim = -1;
  std::vector<SliceInfo> sorted_slices;
  if (!FindSplitDimAndSort(slices, split_dim, sorted_slices)) {
    GELOGD("Failed to find split dim for input: %s", input_node->GetNamePtr());
    return false;
  }

  if (!CheckContinuity(sorted_slices, split_dim, input_node)) {
    GELOGD("Slices are not continuous for input: %s", input_node->GetNamePtr());
    return false;
  }

  return ReplaceSlicesWithSplit(graph, sorted_slices, input_node, split_dim) == GRAPH_SUCCESS;
}
} // namespace

graphStatus SlicesToSplitPass::Run(const ComputeGraphPtr &graph) const {
  GE_ASSERT_NOTNULL(graph);

  std::unordered_map<SliceKey, std::pair<NodePtr, std::vector<SliceInfo> >, SliceKeyHash> input_to_slices;

  CollectSliceNodes(graph, input_to_slices);

  auto sorted_keys = GetSortedSliceKeys(input_to_slices);

  uint32_t replaced_count = 0U;
  for (const auto &key: sorted_keys) {
    const auto &entry = input_to_slices[key];
    if (ProcessSliceGroup(graph, entry.first, entry.second)) {
      replaced_count++;
    }
  }

  if (replaced_count > 0U) {
    GELOGI("Replaced %u groups of slices with split nodes", replaced_count);
  }

  return GRAPH_SUCCESS;
}
} // namespace ge
