/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "liftings.h"
#include <algorithm>
#include "common/checker.h"
#include "graph_metadef/graph/debug/ge_util.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/debug/ge_op_types.h"
#include "graph/graph.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "fusion/autofuse_attrs.h"
#include "utils/autofuse_utils.h"
#include "asc_lowerer/asc_overrides.h"
#include "asc_lowerer/loop_common.h"
#include "lowerings.h"
#include "lowering/op_lowering_impl/lowering_impl.h"
#include "common/autofuse_backend_spec_api.h"
#include "op_helper/lower_split_helper.h"

namespace ge {
constexpr size_t kMinComputeNodes = 2U;
constexpr size_t kMinOneNodeInData = 64U;
constexpr size_t kMatmulMinInputNum = 2U;
constexpr size_t kNumOne = 1U;
constexpr int32_t kConvFilterInputIndex = 1;
// 二次 tiling 缺省 workspace，单位字节，对应 2MB。
constexpr int64_t kDefaultAscendcOpParaSize = 2 * 1024 * 1024;
const char *const kMatmulSubgraph = "matmul_subgraph";
const char *const kConvSubgraph = "conv_subgraph";
const char *const kMMV3Type = "MatMulV3";
const char *const kBMMV3Type = "BatchMatMulV3";
const char *const kConv2DType = "Conv2D";
const char *const kConv2DV2Type = "Conv2DV2";
const char *const kExtendConv2DType = "ExtendConv2D";

GeShape TransferShapeBetweenHwcnNchw(const GeShape &old_shape, const Format &old_format, const Format &new_format) {
  if (old_shape.GetDimNum() != 4U) {
    return old_shape;
  }
  if (old_format == new_format) {
    return old_shape;
  }
  std::vector<int64_t> dims = old_shape.GetDims();
  if (old_format == FORMAT_HWCN && new_format == FORMAT_NCHW) {
    return GeShape({dims[3], dims[2], dims[0], dims[1]});
  }
  if (old_format == FORMAT_NCHW && new_format == FORMAT_HWCN) {
    return GeShape({dims[2], dims[3], dims[1], dims[0]});
  }
  return old_shape;
}

NodePtr AddConvSubgraphInputNode(const ComputeGraphPtr &sub_graph, const Node *org_node,
                                 const GeTensorDesc &conv_input_desc, const int32_t input_index,
                                 const OutDataAnchorPtr &peer_anchor) {
  OpDescPtr input_op_desc;
  if (peer_anchor != nullptr) {
    const auto peer_node = peer_anchor->GetOwnerNode();
    GE_ASSERT_NOTNULL(peer_node);
    input_op_desc = GraphUtils::CopyOpDesc(peer_node->GetOpDesc(), nullptr);
    GE_ASSERT_NOTNULL(input_op_desc);
    input_op_desc->SetName(peer_node->GetName());
  } else {
    const auto input_name = org_node->GetName() + "_autofuse_input_" + std::to_string(input_index);
    input_op_desc = ComGraphMakeShared<OpDesc>(input_name, DATA);
    GE_ASSERT_NOTNULL(input_op_desc);
    GE_ASSERT_GRAPH_SUCCESS(input_op_desc->AddOutputDesc(conv_input_desc));
    GELOGI("[AutoFuseConvSubgraph] Restore disconnected input %s:%d with Data node %s.", org_node->GetNamePtr(),
           input_index, input_name.c_str());
  }
  return sub_graph->AddNode(input_op_desc);
}

graphStatus CreateConvSubgraphAttr(const NodePtr &node, vector<const Node *> &compute_ops, size_t &cube_real_inputs) {
  // ExtendConv2D 存在 optional 输入，按真实锚点连边，不再用 cube_real_inputs 做连续紧凑索引。
  (void)cube_real_inputs;
  const auto &sub_graph = ComGraphMakeShared<ComputeGraph>(kConvSubgraph + node->GetName());
  GE_ASSERT_NOTNULL(sub_graph);
  for (auto *org_node : compute_ops) {
    // Conv2DV2 / ExtendConv2D 都需要落盘到 conv_subgraph，供运行时二次 tiling 使用。
    if (org_node->GetType() == kConv2DV2Type || org_node->GetType() == kExtendConv2DType) {
      const auto &op_desc = GraphUtils::CopyOpDesc(org_node->GetOpDesc(), nullptr);
      GE_ASSERT_NOTNULL(op_desc);
      op_desc->SetName(org_node->GetName());
      const auto &ir_attr_names = op_desc->GetIrAttrNames();
      // ops-nn 增加了 private attr fixed_shift_value（紧跟 ascendc_op_para_size）。
      // 子图拷贝后若缺 ascendc_op_para_size，二次 tiling 按 IR attr 序构造上下文会越界，这里显式补齐。
      if (std::find(ir_attr_names.cbegin(), ir_attr_names.cend(), "ascendc_op_para_size") == ir_attr_names.cend()) {
        op_desc->AppendIrAttrName("ascendc_op_para_size");
        GELOGI("[AutoFuseConvSubgraph] Node:%s(%s) missing IR attr ascendc_op_para_size, append it.",
               op_desc->GetNamePtr(), op_desc->GetTypePtr());
      }
      if (!AttrUtils::HasAttr(op_desc, "ascendc_op_para_size")) {
        GELOGI("[AutoFuseConvSubgraph] Node:%s(%s) missing attr ascendc_op_para_size, set default %" PRId64 " bytes.",
               op_desc->GetNamePtr(), op_desc->GetTypePtr(), kDefaultAscendcOpParaSize);
        GE_ASSERT_TRUE(AttrUtils::SetInt(op_desc, "ascendc_op_para_size", kDefaultAscendcOpParaSize));
      }
      auto conv_node = sub_graph->AddNode(op_desc);
      GE_ASSERT_NOTNULL(conv_node);
      const auto &org_in_anchors = org_node->GetAllInDataAnchors();
      for (const auto &org_in_anchor : org_in_anchors) {
        GE_ASSERT_NOTNULL(org_in_anchor);
        const auto org_input_index = org_in_anchor->GetIdx();
        auto peer_anchor = org_in_anchor->GetPeerOutAnchor();
        auto conv_input_desc = op_desc->MutableInputDesc(org_input_index);
        if (conv_input_desc == nullptr) {
          GELOGI("[AutoFuseConvSubgraph] Node:%s(%s), input:%d has no input desc.", op_desc->GetNamePtr(),
                 op_desc->GetTypePtr(), org_input_index);
          continue;
        }
        const bool is_input_desc_valid = conv_input_desc->IsValid() == GRAPH_SUCCESS;
        if (!is_input_desc_valid) {
          GE_ASSERT_TRUE(org_input_index > kConvFilterInputIndex,
                         "Node:%s(%s) required input:%d has invalid tensor desc.", op_desc->GetNamePtr(),
                         op_desc->GetTypePtr(), org_input_index);
          GELOGI("[AutoFuseConvSubgraph] Node:%s(%s), optional input:%d has invalid tensor desc.",
                 op_desc->GetNamePtr(), op_desc->GetTypePtr(), org_input_index);
          continue;
        }
        // 针对5102，二次 tiling 侧 filter 期望 FRACTAL_Z；同步更新融合节点与子图节点，避免 format 校验失败。
        if (org_input_index == kConvFilterInputIndex) {
          auto fused_filter_desc = node->GetOpDesc()->MutableInputDesc(kConvFilterInputIndex);
          GE_ASSERT_NOTNULL(fused_filter_desc);
          fused_filter_desc->SetFormat(FORMAT_FRACTAL_Z);
          // conv_input_desc 是拷出来挂到 conv_subgraph 上的那份。如果convfixpipe前移并完成format转换，
          // 这里应该不需要手动设置。
          conv_input_desc->SetFormat(FORMAT_FRACTAL_Z);
        }
        // 原始边仍在时复制真实 peer；边已被无用边清理或融合流程删除时，根据有效 input desc 创建 Data 占位。
        auto conv_peer_node =
            AddConvSubgraphInputNode(sub_graph, org_node, *conv_input_desc, org_input_index, peer_anchor);
        GE_ASSERT_NOTNULL(conv_peer_node);
        // Data 占位只有 output 0；复制真实 peer 时则保持其原始输出索引。
        const auto peer_output_index = peer_anchor == nullptr ? 0 : peer_anchor->GetIdx();
        const auto &peer_node_out_anchor = conv_peer_node->GetOutDataAnchor(peer_output_index);
        GE_ASSERT_NOTNULL(peer_node_out_anchor);
        GE_ASSERT_GRAPH_SUCCESS(GraphUtils::AddEdge(peer_node_out_anchor, conv_node->GetInDataAnchor(org_input_index)));
      }
      break;
    }
  }

  auto op_desc = node->GetOpDesc();
  GE_ASSERT_NOTNULL(op_desc);
  GE_ASSERT_TRUE(op_desc->SetExtAttr(kConvSubgraph, sub_graph));
  return GRAPH_SUCCESS;
}

graphStatus RecordLiftingSkipReason(const NodePtr &fuse_node, const std::string &reason) {
  GE_ASSERT_NOTNULL(fuse_node->GetOpDesc());
  auto fuse_attrs = fuse_node->GetOpDesc()->GetAttrsGroup<AutoFuseAttrs>();
  GE_ASSERT_NOTNULL(fuse_attrs);
  const std::vector<const ge::Node *> origin_nodes = fuse_attrs->GetOriginNodes();
  for (const auto *origin_node : origin_nodes) {
    if (origin_node != nullptr) {
      GraphFusionReasonStore::CountNodeFuseFailReason(
          origin_node->GetName(), "Skip lifting: " + reason,
          GraphFusionReasonStore::FailReasonCategory::TEMPORARILY_NOT_SUPPORTED);
    }
  }
  return GRAPH_SUCCESS;
}

graphStatus CreateMMSubgraphAttr(const NodePtr &node, vector<const Node *> &compute_ops, size_t &cube_real_inputs) {
  const auto &sub_graph = ComGraphMakeShared<ComputeGraph>(kMatmulSubgraph + node->GetName());
  GE_ASSERT_NOTNULL(sub_graph);
  for (auto *org_node : compute_ops) {
    if ((org_node->GetType() == "MatMulV3") || (org_node->GetType() == "BatchMatMulV3")) {
      const auto &op_desc = GraphUtils::CopyOpDesc(org_node->GetOpDesc(), nullptr);
      GE_ASSERT_NOTNULL(op_desc);
      op_desc->SetName(org_node->GetName());
      auto mm_node = sub_graph->AddNode(op_desc);
      GE_ASSERT_NOTNULL(mm_node);
      bool is_a_b_same_input = cube_real_inputs > node->GetAllInDataAnchors().size();
      for (auto i = 0U; i < cube_real_inputs; i++) {
        // a矩阵、b矩阵同输入存在ascgraph的matmul有两个输入，Ascackend只有一个输入，需多加一个输入再生成kernel函数
        auto i_node = is_a_b_same_input ? (i == 0U ? 0U : i - 1U) : i;
        const auto &src_anchor = node->GetInDataAnchor(i_node);  // cube垂直向后融合可以保证cube输入在前
        GE_ASSERT_NOTNULL(src_anchor);
        auto peer_anchor = src_anchor->GetPeerOutAnchor();
        GE_ASSERT_NOTNULL(peer_anchor);
        auto peer_node = peer_anchor->GetOwnerNode();
        GE_ASSERT_NOTNULL(peer_node);
        const auto &op_desc = GraphUtils::CopyOpDesc(peer_node->GetOpDesc(), nullptr);
        GE_ASSERT_NOTNULL(op_desc);
        op_desc->SetName(peer_node->GetName());
        auto mm_peer_node = sub_graph->AddNode(op_desc);
        GE_ASSERT_NOTNULL(mm_peer_node);
        const auto &peer_node_out_anchor = mm_peer_node->GetOutDataAnchor(peer_anchor->GetIdx());
        GE_ASSERT_NOTNULL(peer_node_out_anchor);
        GE_ASSERT_GRAPH_SUCCESS(GraphUtils::AddEdge(peer_node_out_anchor, mm_node->GetInDataAnchor(i)));
      }
      break;
    }
  }

  auto op_desc = node->GetOpDesc();
  GE_ASSERT_NOTNULL(op_desc);
  GE_ASSERT_TRUE(op_desc->SetExtAttr(kMatmulSubgraph, sub_graph));
  return GRAPH_SUCCESS;
}

bool IsCubeSkipLifting(const NodePtr &node, const size_t min_compute_nodes, const AutoFuseAttrs *fuse_attrs,
                       bool is_fuse_from_lowering) {
  auto origin_nodes = fuse_attrs->GetOriginNodes();
  vector<const Node *> compute_ops =
      AutofuseUtils::GetComputeOps(origin_nodes);  // GetComputeOps里面融合reshape等节点不会统计成compute节点
  if ((compute_ops.size() < min_compute_nodes) &&
      is_fuse_from_lowering) {  // 需要is_fuse_from_lowering标记判断是否经过canfuse融合
    return false;
  }

  size_t cube_real_inputs = kMatmulMinInputNum;
  const auto asc_graph = fuse_attrs->GetAscGraph();
  GE_ASSERT_NOTNULL(asc_graph);

  bool has_conv = false;
  for (const auto &asc_node : asc_graph->GetAllNodes()) {
    if (!AutofuseUtils::IsCubeNodeType(asc_node)) {
      continue;
    }
    if (asc_node->GetType() == kConv2DType || asc_node->GetType() == kConv2DBias ||
        asc_node->GetType() == kConv2DOffset || asc_node->GetType() == kConv2DOffsetBias ||
        asc_node->GetType() == kExtendConv2DType || asc_node->GetType() == kExtendConv2DBias ||
        asc_node->GetType() == kExtendConv2DScale || asc_node->GetType() == kExtendConv2DBiasScale) {
      has_conv = true;
    } else {
      // 当前cube只有matmul和conv，非conv就是matmul
    }
    cube_real_inputs =
        asc_node->GetInNodes().size();  // ascgraph里面的cube节点，即使输入是某个节点输出的多引用，也有至少两个输入
    break;
  }
  if (has_conv) {
    GE_ASSERT_SUCCESS(CreateConvSubgraphAttr(node, compute_ops, cube_real_inputs));
    GELOGI("Skip lifting node: %s, cube_real_inputs %zu, has_conv", node->GetNamePtr(), cube_real_inputs);
  } else {
    GE_ASSERT_SUCCESS(CreateMMSubgraphAttr(node, compute_ops, cube_real_inputs));
    GELOGI("Skip lifting node: %s, cube_real_inputs %zu, has_matmul", node->GetNamePtr(), cube_real_inputs);
  }
  return true;
}

bool IsSingleTransposeShouldSkipLifting(const NodePtr &node) {
  const auto fuse_attrs = node->GetOpDesc()->GetAttrsGroup<AutoFuseAttrs>();
  GE_ASSERT_NOTNULL(fuse_attrs);
  const auto asc_graph = fuse_attrs->GetAscGraph();
  GE_ASSERT_NOTNULL(asc_graph);
  for (const auto &asc_node : asc_graph->GetAllNodes()) {
    if (asc_node->GetType() == af::ascir_op::Transpose::Type) {
      const auto input_size = asc_node->inputs[0].attr.axis.size();
      GE_ASSERT_TRUE(input_size > 0, "input_size %d out of range", input_size);
      const auto &input_tail_axis = asc_node->inputs[0].attr.axis[input_size - 1];
      const auto &output_tail_axis = asc_node->outputs[0].attr.axis[input_size - 1];
      const auto repeat = asc_node->inputs[0].attr.repeats[input_size - 1];
      int64_t dim = -1;
      GE_ASSERT_TRUE(repeat.GetHint(dim), "Failed to get int value, expr = %s",
                     ge::SymbolicUtils::ToString(repeat).c_str());
      const auto data_type_size = GetSizeByDataType(asc_node->inputs[0].attr.dtype);
      GE_ASSERT_TRUE(data_type_size > 0, "data_type_size must be greater than 0",
                     ge::SymbolicUtils::ToString(repeat).c_str());
      constexpr int64_t limited_tail_size = 512U;
      const auto limited_size = limited_tail_size / data_type_size;
      // 目前仅非尾轴转置且大尾轴场景跳过Lifting
      if ((input_tail_axis == output_tail_axis) && (dim >= limited_size)) {
        return true;
      }
    }
  }
  return false;
}

bool IsSpecificConditionSkipLifting(const NodePtr &node) {
  if (IsSingleTransposeShouldSkipLifting(node)) {
    return true;
  }
  // 可新增其他特殊场景
  return false;
}

bool IsSkipLifting(const NodePtr &node, size_t min_compute_nodes) {
  auto fuse_attrs = node->GetOpDesc()->GetAttrsGroup<AutoFuseAttrs>();
  GE_ASSERT_NOTNULL(fuse_attrs);
  bool disable_lifting = false;
  if (AttrUtils::GetBool(node->GetOpDesc(), "_disable_lifting", disable_lifting) && disable_lifting) {
    GELOGI("Skip lifting node: %s, as it has disable lifting flag", node->GetNamePtr());
    return true;
  }
  // step1: cube type
  if (fuse_attrs->HasFuseType(loop::FuseType::kCube)) {
    return IsCubeSkipLifting(node, min_compute_nodes, fuse_attrs, GetInterAttrs(fuse_attrs).is_fuse_from_lowering);
  }
  // step2: fused from can_fuse
  auto &inner_fuse_attr = GetInterAttrs(fuse_attrs);
  if (inner_fuse_attr.split_global_id == kNonSplitGlobalId && !inner_fuse_attr.is_fuse_from_lowering) {
    GELOGI("Skip lifting node: %s, as it is fused from can_fuse, and is not split fuse type.", node->GetNamePtr());
    return true;
  }
  // step3: split type
  if (fuse_attrs->HasFuseType(loop::FuseType::kSplit)) {
    bool need_lifting = false;
    LowerSplitHelper split_helper(node);
    split_helper.NeedLifting(need_lifting);
    return !need_lifting;
  }
  auto origin_nodes = fuse_attrs->GetOriginNodes();
  if ((origin_nodes.size() > kNumOne) && (fuse_attrs->HasFuseType(loop::FuseType::kSliceSplit))) {
    GELOGI("Skip lifting node: %s, as slice fuse other node, origin node size is %zu", node->GetNamePtr(),
           origin_nodes.size());
    return true;
  }
  // step4: compute node num
  vector<const Node *> compute_nodes = AutofuseUtils::GetComputeOps(origin_nodes);
  if (compute_nodes.size() >= min_compute_nodes) {
    GELOGD("Skip lifting node: %s, as num fused nodes num %zu >= %zu", node->GetNamePtr(), compute_nodes.size(),
           min_compute_nodes);
    return true;
  }

  if (fuse_attrs->GetOriginOutputBuffers().size() > kNumOne) {
    GELOGD("Skip lifting node: %s, as num origin output anchors %zu > 1", node->GetNamePtr(),
           fuse_attrs->GetOriginOutputBuffers().size());
    return true;
  }

  if (!fuse_attrs->GetOptimizedInputBuffers().empty()) {
    auto optimized_input_buffers = fuse_attrs->GetOptimizedInputBuffers();
    for (auto optimized_input_buffer : optimized_input_buffers) {
      GE_ASSERT_NOTNULL(optimized_input_buffer);
      GE_ASSERT_NOTNULL(optimized_input_buffer->GetOwnerNode());
      auto optimized_input_node = optimized_input_buffer->GetOwnerNode();
      if (!OpTypeUtils::IsConstNode(optimized_input_node->GetType())) {
        GELOGD("Skip lifting node: %s, as it optimize buffer loads %s", node->GetNamePtr(),
               loop::BufferName(*fuse_attrs->GetOptimizedInputBuffers().begin()).c_str());
        return true;
      }
    }
  }

  auto min_one_node_in_data = kMinOneNodeInData;
  const auto backend_spec = ge::GetAutofuseBackendSpec();
  if (backend_spec != nullptr) {
    min_one_node_in_data = backend_spec->concat_max_input_num + 1;
  }
  if ((origin_nodes.size() == kNumOne) && (origin_nodes.at(0) != nullptr) &&
      (origin_nodes.at(0)->GetAllInDataAnchorsSize() >= min_one_node_in_data)) {
    GELOGI("Skip lifting node: %s, as it has only one node but origin input size is %u", node->GetNamePtr(),
           fuse_attrs->GetOriginNodes().at(0)->GetAllInDataAnchorsSize());
    return true;
  }

  // AscIR只包含Transpose类型节点时跳过lifting
  if ((origin_nodes.size() == kNumOne) && (IsSpecificConditionSkipLifting(node))) {
    GELOGI(
        "Skip lifting node: %s, as the origin node is "
        "Non-tail axis Transpose with Tail axis greater than or equal to 512B",
        node->GetNamePtr());
    return true;
  }
  return false;
}

graphStatus LiftingAscBackendOp(const NodePtr &node) {
  const auto fuse_attr = node->GetOpDesc()->GetAttrsGroup<AutoFuseAttrs>();
  GE_WARN_ASSERT(fuse_attr != nullptr, "Node %s has no AutoFuseAttrs", node->GetNamePtr());
  std::vector<const ge::Node *> origin_nodes = fuse_attr->GetOriginNodes();

  const std::map<size_t, std::set<const ge::InDataAnchor *>> &concrete_edges = fuse_attr->GetConcreteEdges();
  for (auto &edges : concrete_edges) {
    auto in_anchor = node->GetInDataAnchor(static_cast<int32_t>(edges.first));
    GE_ASSERT_NOTNULL(in_anchor, "Node %s has no input anchor %zu", node->GetNamePtr(), edges.first);
    auto src = in_anchor->GetPeerOutAnchor();
    GE_ASSERT_NOTNULL(src, "Node %s input %zu has no peer out anchor", node->GetNamePtr(), edges.first);
    for (auto &dst : edges.second) {
      GE_ASSERT_NOTNULL(dst);
      if (!dst->IsLinkedWith(src)) {
        GELOGI("Lifting recover edge %s->%s", loop::BufferName(src).c_str(), loop::BufferName(dst).c_str());
        GE_ASSERT_GRAPH_SUCCESS(GraphUtils::AddEdge(src, const_cast<ge::InDataAnchor *>(dst)->shared_from_this()));
      }
    }
  }
  auto origin_index = 0U;
  for (auto asc_output : node->GetAllOutDataAnchors()) {
    GE_ASSERT_NOTNULL(asc_output);
    GE_ASSERT_TRUE(fuse_attr->GetOriginOutputBuffers().size() > origin_index);
    const auto origin_output = fuse_attr->GetOriginOutputBuffers()[origin_index++];
    for (auto &peer : asc_output->GetPeerAnchors()) {
      GELOGD("Replace src of edge %s->%s to %s", loop::BufferName(asc_output).c_str(), loop::BufferName(peer).c_str(),
             loop::BufferName(origin_output).c_str());
      GE_ASSERT_GRAPH_SUCCESS(GraphUtils::RemoveEdge(asc_output, peer->shared_from_this()));
      GE_ASSERT_GRAPH_SUCCESS(GraphUtils::AddEdge(origin_output->shared_from_this(), peer));
    }

    auto origin_control = origin_output->GetOwnerNode()->GetOutControlAnchor();
    auto asc_control = asc_output->GetOwnerNode()->GetOutControlAnchor();
    for (auto &peer : asc_control->GetPeerAnchors()) {
      GELOGD("Replace src of edge %s->%s to %s", loop::BufferName(asc_control).c_str(), loop::BufferName(peer).c_str(),
             loop::BufferName(origin_control).c_str());
      GE_ASSERT_GRAPH_SUCCESS(GraphUtils::RemoveEdge(asc_control, peer));
      GE_ASSERT_GRAPH_SUCCESS(GraphUtils::AddEdge(origin_control, peer));
    }
  }
  return GRAPH_SUCCESS;
}

graphStatus LiftingAscBackendOps(const std::vector<NodePtr> &nodes) {
  for (auto &node : nodes) {
    GE_ASSERT_GRAPH_SUCCESS(LiftingAscBackendOp(node));
  }
  return GRAPH_SUCCESS;
}

graphStatus LiftingMultiOutNode(const NodePtr &node, const NodePtr &origin_node, AutoFuseAttrs *fuse_attrs,
                                std::map<NodePtr, std::vector<NodePtr>> &node_maybe_lifting_outputs) {
  auto &maybe_lifting = node_maybe_lifting_outputs[origin_node];
  maybe_lifting.push_back(node);
  GELOGD("Maybe lifting %s of node %s", node->GetNamePtr(), origin_node->GetNamePtr());
  size_t num_of_out_anchors = 0U;
  for (const auto &lifting_node : maybe_lifting) {
    num_of_out_anchors += lifting_node->GetAllOutDataAnchorsSize();
  }
  if (num_of_out_anchors == origin_node->GetAllOutDataAnchorsSize()) {
    GELOGI("Lift AscBackend nodes %s, node list is %s, as: Num fused nodes %zu < %zu.",
           loop::StrJoin(maybe_lifting, [](const NodePtr &n) { return n->GetName(); }).c_str(),
           loop::StrJoin(fuse_attrs->GetOriginNodes(),
                         [](const Node *n) { return n->GetType() + "(" + n->GetName() + ")"; })
               .c_str(),
           fuse_attrs->GetOriginNodes().size(), kMinComputeNodes);
    GE_ASSERT_GRAPH_SUCCESS(LiftingAscBackendOps(maybe_lifting));
    maybe_lifting.clear();
  }
  return GRAPH_SUCCESS;
}

graphStatus LiftingManager::LiftingGraph(const ComputeGraphPtr &graph) {
  GE_ASSERT_NOTNULL(graph);
  std::map<NodePtr, std::vector<NodePtr>> node_maybe_lifting_outputs;
  for (auto &node : graph->GetAllNodes()) {
    if (node->GetType() != kAscBackend && node->GetType() != kAscBackendNoKernelOp) {
      continue;
    }
    auto fuse_attrs = node->GetOpDesc()->GetAttrsGroup<AutoFuseAttrs>();
    if (fuse_attrs == nullptr) {
      GELOGD("Skip lifting node: %s, as it has no auto fuse attrs", node->GetNamePtr());
      continue;
    }
    auto &origin_output = fuse_attrs->GetOriginOutputBuffers()[0];
    GE_ASSERT_NOTNULL(origin_output);
    const auto &origin_node = origin_output->GetOwnerNode();
    GE_ASSERT_NOTNULL(origin_node);

    if (IsSkipLifting(node, kMinComputeNodes)) {
      GELOGD("Skip lifting node: %s(%s), as it need skip lifting process.", node->GetNamePtr(),
             node->GetType().c_str());
      continue;
    }

    if (origin_node->GetAllOutDataAnchorsSize() > kNumOne) {
      GE_ASSERT_GRAPH_SUCCESS(LiftingMultiOutNode(node, origin_node, fuse_attrs, node_maybe_lifting_outputs));
      continue;
    }

    GELOGI("Lift AscBackend node %s, node list is %s, as: Num fused nodes %zu < %zu.", node->GetNamePtr(),
           loop::StrJoin(fuse_attrs->GetOriginNodes(),
                         [](const Node *n) { return n->GetType() + "(" + n->GetName() + ")"; })
               .c_str(),
           fuse_attrs->GetOriginNodes().size(), kMinComputeNodes);
    (void)RecordLiftingSkipReason(node, "Not satisfy requierent of skiping lifting , need do lifting");
    GE_ASSERT_GRAPH_SUCCESS(LiftingAscBackendOp(node));
  }
  return GRAPH_SUCCESS;
}
}  // namespace ge
