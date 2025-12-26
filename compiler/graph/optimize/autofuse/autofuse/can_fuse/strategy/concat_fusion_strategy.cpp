/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "concat_fusion_strategy.h"

#include "backend/backend_spec.h"
#include "can_fuse/backend/backend_utils.h"
#include "utils/autofuse_attrs.h"
#include "can_fuse/strategy/fusion_strategy_registry.h"
#include "utils/auto_fuse_config.h"
#include "utils/autofuse_utils.h"
#include "utils/not_fuse_reason_code.h"

namespace ge {
namespace {
constexpr int32_t kConcatAlgTranspose = 0;
}  // namespace
using namespace autofuse;;
bool ConcatFusionStrategy::CanFuse(const NodePtr &node1, const NodePtr &node2) {
  const auto attr1 = BackendUtils::GetNodeAutoFuseAttr(node1);
  GE_ASSERT_NOTNULL(attr1);
  const auto attr2 = BackendUtils::GetNodeAutoFuseAttr(node2);
  GE_ASSERT_NOTNULL(attr2);

  // 1.concat不能和concat融合，不区分垂直融合还是水平融合
  if (attr1->HasFuseType(loop::FuseType::kConcat) && attr2->HasFuseType(loop::FuseType::kConcat)) {
    GELOGI(
        "node1 %s(%s) and node2 %s(%s) can not fuse, the reason is [%s][node1 and node2 are both Concat type, Concat "
        "can not fuse with Concat, regardless of whether it is vertical fuse or horizontal fuse]",
        node1->GetNamePtr(), node1->GetType().c_str(), node2->GetNamePtr(), node2->GetType().c_str(),
        ge::NotFuseReasonCode(ge::NotFuseReason::kConcatCanNotFuseWithConcat));
    return false;
  }

  // 2.concat不能和reduce融，不区分垂直融合还是水平融合
  if ((attr1->HasFuseType(loop::FuseType::kConcat) && attr2->HasFuseType(loop::FuseType::kReduction)) ||
      (attr1->HasFuseType(loop::FuseType::kReduction) && attr2->HasFuseType(loop::FuseType::kConcat))) {
    GELOGI(
        "node1 %s(%s) and node2 %s(%s) can not fuse, the reason is [%s][Concat can not fuse with Reduce, regardless of "
        "whether it is vertical fuse or horizontal fuse]",
        node1->GetNamePtr(), node1->GetType().c_str(), node2->GetNamePtr(), node2->GetType().c_str(),
        ge::NotFuseReasonCode(ge::NotFuseReason::kConcatCanNotFuseWithReduce));
    return false;
  }

  // 3. 后融合只支持Concat->Pointwise
  if (BackendUtils::IsVertical(node1, node2) && attr1->HasFuseType(loop::FuseType::kConcat)) {
    if (CanFuseBackward(node1, node2, attr2) != SUCCESS) {
      return false;
    }
  }

  // 4. 判断是否能前融合Split
  // Split不支持前融合，此处不需要判断
  if (attr1->HasFuseType(loop::FuseType::kSplit) && attr2->HasFuseType(loop::FuseType::kConcat)) {
    if (CanFuseSplit(node1, node2) != SUCCESS) {
      return false;
    }
  }

  // 5.调度轴个数不同不能融合，不区分垂直融合还是水平融合
  ComputeGraphPtr graph1;
  ComputeGraphPtr graph2;
  GE_ASSERT_SUCCESS(BackendUtils::GetNodeFusedGraph(node1, graph1));
  GE_ASSERT_SUCCESS(BackendUtils::GetNodeFusedGraph(node2, graph2));
  const auto graph_attr1 = graph1->GetOrCreateAttrsGroup<AscGraphAttr>();
  GE_ASSERT_NOTNULL(graph_attr1);
  const auto graph_attr2 = graph2->GetOrCreateAttrsGroup<AscGraphAttr>();
  GE_ASSERT_NOTNULL(graph_attr2);
  if (attr1->HasFuseType(loop::FuseType::kConcat) && graph_attr1->axis.size() < graph_attr2->axis.size()) {
    GELOGI(
        "node1 %s(%s) and node2 %s(%s) can not fuse, the reason is [%s][In concat fusion occasion, node1 sched axis "
        "size(%zu) less than node2 sched axis size(%zu)]", node1->GetNamePtr(), node1->GetType().c_str(),
        node2->GetNamePtr(), node2->GetType().c_str(), ge::NotFuseReasonCode(ge::NotFuseReason::kConcatNodeSchedAxisSizeNotEqual),
        graph_attr1->axis.size(), graph_attr2->axis.size());
    return false;
  }
  if (attr2->HasFuseType(loop::FuseType::kConcat) && graph_attr2->axis.size() < graph_attr1->axis.size()) {
    GELOGI(
        "node1 %s(%s) and node2 %s(%s) can not fuse, the reason is [%s][In concat fusion occasion, node1 sched axis "
        "size(%zu) more than node2 sched axis size(%zu)]", node1->GetNamePtr(), node1->GetType().c_str(),
        node2->GetNamePtr(), node2->GetType().c_str(), ge::NotFuseReasonCode(ge::NotFuseReason::kConcatNodeSchedAxisSizeNotEqual),
        graph_attr1->axis.size(), graph_attr2->axis.size());
    return false;
  }
  return true;
}

bool ConcatFusionStrategy::CanMergeLoop(const NodePtr &node1, const NodePtr &node2) {
  auto attr_1 = BackendUtils::GetNodeAutoFuseAttr(node1);
  GE_ASSERT_NOTNULL(attr_1);
  auto attr_2 = BackendUtils::GetNodeAutoFuseAttr(node2);
  GE_ASSERT_NOTNULL(attr_2);
  // concat不能循环合并，有FusedAscBackend场景使用HasFuseType判断
  if (attr_1->HasFuseType(loop::FuseType::kConcat) || attr_2->HasFuseType(loop::FuseType::kConcat)) {
    GELOGI("node1 %s(%s) and node2 %s(%s) can not merge.", node1->GetNamePtr(), node1->GetTypePtr(),
           node2->GetNamePtr(), node2->GetTypePtr());
    return false;
  }
  return true;
}

uint32_t ConcatFusionStrategy::GetFusionPairPriority(const NodePtr &node1, const NodePtr &node2) {
  auto attr = BackendUtils::GetNodeAutoFuseAttr(node2);
  GE_ASSERT_NOTNULL(attr);
  uint32_t fusion_priority = kDefaultFusionPriority;
  // 首轮融合才要处理，只有AscBackend场景
  if (attr->GetFuseType() == loop::FuseType::kConcat) {
    fusion_priority = kHighFusionPriority;
    GELOGI("node1 %s(*) and node2 %s(Concat) priority:%u.", node1->GetNamePtr(), node2->GetNamePtr(),
           fusion_priority);
  } else {
    auto attr = BackendUtils::GetNodeAutoFuseAttr(node1);
    GE_ASSERT_NOTNULL(attr);
    if (attr->GetFuseType() == loop::FuseType::kConcat) {
      fusion_priority = kHighFusionPriority;
      GELOGI("node1 %s(Concat) and node2 %s(*) priority:%u.", node1->GetNamePtr(), node2->GetNamePtr(),
             fusion_priority);
    }
  }
  return fusion_priority;
}

uint64_t ConcatFusionStrategy::GetMaxFusionNodesSize(const NodePtr &node1, const NodePtr &node2) {
  const auto &config = AutoFuseConfig::Config().GetFusionStrategySolver();
  uint64_t max_fusion_size = config.max_fusion_size;
  // concat和输入融合时不限制个数
  if (BackendUtils::IsVertical(node1, node2)) {
    auto attr = BackendUtils::GetNodeAutoFuseAttr(node2);
    GE_ASSERT_NOTNULL(attr);
    if (attr->HasFuseType(loop::FuseType::kConcat)) {
      max_fusion_size = std::numeric_limits<uint64_t>::max();
    }
  }
  GELOGI("node1 %s(*) and node2 %s(*) max_fusion_nodes_size:%lu.", node1->GetNamePtr(), node2->GetNamePtr(),
         max_fusion_size);
  return max_fusion_size;
}

uint32_t ConcatFusionStrategy::GetMaxFusionNodeInputSize(const NodePtr &node1, const NodePtr &node2) {
  const auto &config = AutoFuseConfig::Config().GetFusionStrategySolver();
  uint32_t max_input_nums_after_fuse = config.max_input_nums_after_fuse;
  // concat和输入融合时不限制个数
  if (BackendUtils::IsVertical(node1, node2)) {
    auto attr = BackendUtils::GetNodeAutoFuseAttr(node2);
    GE_ASSERT_NOTNULL(attr);
    if (attr->HasFuseType(loop::FuseType::kConcat)) {
      max_input_nums_after_fuse = std::numeric_limits<uint32_t>::max();
    }
  }
  GELOGI("node1 %s(*) and node2 %s(*) max_fusion_node_input_size:%u.", node1->GetNamePtr(), node2->GetNamePtr(),
         max_input_nums_after_fuse);
  return max_input_nums_after_fuse;
}

bool ConcatFusionStrategy::CheckSameSchedAxis(const NodePtr &node1, const NodePtr &node2,
                                              const AxisPairSet &node1_map, const AxisPairSet &node2_map,
                                              const NodeFuseInfo &node_fuse_info) {
  // 此时node1和node2有水平融合，如果：
  // 1.node1和node2只有水平融合，这种场景就不融合
  if (!BackendUtils::IsVertical(node1, node2) && !BackendUtils::IsVertical(node2, node1)) {
    GELOGD("node1 %s(%s) and node2 %s(%s) only horizontal fuse, can not fuse.", node1->GetNamePtr(),
           node1->GetType().c_str(), node2->GetNamePtr(), node2->GetType().c_str());
    return false;
  }
  // 2.node1和node2既有水平融合也有垂直融合，需要进行以下判断
  for (const auto &same_input_map : node_fuse_info.GetSameInputMap()) {
    AxisIdAndSizePair node1_axis_id_and_size_pair_list;
    GE_ASSERT_SUCCESS(BackendUtils::GetScheduleAxisInfo(node1, same_input_map.first, node1_map,
                                                        node1_axis_id_and_size_pair_list));
    AxisIdAndSizePair node2_axis_id_and_size_pair_list;
    GE_ASSERT_SUCCESS(BackendUtils::GetScheduleAxisInfo(node2, same_input_map.second, node2_map,
                                                        node2_axis_id_and_size_pair_list));
    // 判断两个节点的循环轴和轴的大小是否相同
    if (node1_axis_id_and_size_pair_list.size() != node2_axis_id_and_size_pair_list.size()) {
      GELOGD(
          "node1 %s(%s) node1_axis_id_and_size_pair_list size: %zu and node2 %s(%s) node2_axis_id_and_size_pair_list "
          "size: %zu are not equal, can not fuse.", node1->GetNamePtr(), node1->GetType().c_str(),
          node1_axis_id_and_size_pair_list.size(), node2->GetNamePtr(), node2->GetType().c_str(),
          node2_axis_id_and_size_pair_list.size());
      return false;
    }

    for (auto i = 0U; i < node1_axis_id_and_size_pair_list.size(); ++i) {
      const auto node1_axis_id_and_size_pair = node1_axis_id_and_size_pair_list[i];
      const auto node2_axis_id_and_size_pair = node2_axis_id_and_size_pair_list[i];
      if (node1_axis_id_and_size_pair.first != node2_axis_id_and_size_pair.first ||
          node1_axis_id_and_size_pair.second != node2_axis_id_and_size_pair.second) {
        GELOGD(
            "node1 %s(%s) axis id(%ld) and size(%s) and node2 %s(%s) axis id(%ld) and size(%s) are not equal, "
            "can not fuse.", node1->GetNamePtr(), node1->GetType().c_str(), node1_axis_id_and_size_pair.first,
            std::string(node1_axis_id_and_size_pair.second.Str().get()).c_str(), node2->GetNamePtr(),
            node2->GetType().c_str(), node2_axis_id_and_size_pair.first,
            std::string(node2_axis_id_and_size_pair.second.Str().get()).c_str());
        return false;
      }
    }
  }
  return true;
}

Status ConcatFusionStrategy::CanFuseBackward(const NodePtr &node1, const NodePtr &node2,
                                             const AutoFuseAttrs *attr2) {
  const auto backend_spec = optimize::BackendSpec::GetInstance();
  GE_ASSERT_NOTNULL(backend_spec);
  const auto support_backward_fusion_ = backend_spec->concat_alg != kConcatAlgTranspose;
  if (!support_backward_fusion_) {
    GELOGI(
        "node1 %s(%s) and node2 %s(%s) can not fuse, the reason is [%s][node1 and node2 are vertical fuse, and node1 "
        "is Concat type, concat can not fuse backward]",
        node1->GetNamePtr(), node1->GetType().c_str(), node2->GetNamePtr(), node2->GetType().c_str(),
        ge::NotFuseReasonCode(ge::NotFuseReason::kConcatCanNotFuseBackward));
    return NOT_CHANGED;
  }

  const auto attr = BackendUtils::GetAscGraphOutputAttr(node1, node2);
  GE_ASSERT_NOTNULL(attr);
  // 后融合只支持来源于Concat输出	
  if (attr->GetFuseType() != loop::FuseType::kConcat) {
    GELOGI(
        "node1 %s(%s) and node2 %s(%s) can not fuse, the reason is [%s][node1 and node2 are vertical fuse, and node1 "
        "is Concat type, but concat node not linked to node2]",
        node1->GetNamePtr(), node1->GetType().c_str(), node2->GetNamePtr(), node2->GetType().c_str(),
        ge::NotFuseReasonCode(ge::NotFuseReason::kFusedAscBackendCanNotBackwardFuseFromNonConcat));
    return NOT_CHANGED;
  }
  // 后融合只支持融合Pointwise
  if (attr2->GetAllFuseType() != (1UL << static_cast<uint64_t>(loop::FuseType::kPointwise))) {
    GELOGI(
        "node1 %s(%s) and node2 %s(%s) can not fuse, the reason is [%s][node1 and node2 are vertical fuse, and node1 "
        "is Concat type, concat can not backward fuse non Pointwise type]",
        node1->GetNamePtr(), node1->GetType().c_str(), node2->GetNamePtr(), node2->GetType().c_str(),
        ge::NotFuseReasonCode(ge::NotFuseReason::kConcatCanNotBackwardFuseNonSimplestPointwise));
    return NOT_CHANGED;
  }

  for (const auto node1_out_anchor : node1->GetAllOutDataAnchorsPtr()) {
    GE_ASSERT_NOTNULL(node1_out_anchor);
    for (const auto &peer_in_anchor : node1_out_anchor->GetPeerInDataAnchorsPtr()) {
      if (node2 == peer_in_anchor->GetOwnerNode()) {
        std::vector<ViewOpAttrInfo> attr_infos;
        GE_CHK_BOOL_RET_SPECIAL_STATUS(
            (!BackendUtils::CurNodeInputIsSimplestLoad(node2, peer_in_anchor->GetIdx(), attr_infos)), NOT_CHANGED,
            "node1 %s(%s) and node2 %s(%s) can not fuse, the reason is [%s][node1 and node2 are vertical fuse, and "
            "node1 is Concat type, concat can not backward fuse Broadcast]",
            node1->GetNamePtr(), node1->GetType().c_str(), node2->GetNamePtr(), node2->GetType().c_str(),
            ge::NotFuseReasonCode(ge::NotFuseReason::kFusedAscBackendCanNotBackwardFuseBroadcast));
      }
    }
  }
  GELOGD("check can fuse backward success");
  return SUCCESS;
}

Status ConcatFusionStrategy::CanFuseSplit(const NodePtr &node1, const NodePtr &node2) {
  auto concat_node = FindConcatNode(node2);
  GE_ASSERT_NOTNULL(concat_node);
  if (IsConcatOrSplitFirstDim(concat_node)) {
    GELOGD("[%s] concat on first dim, can fuse split", concat_node->GetNamePtr());
    return SUCCESS;
  }
  ComputeGraphPtr subgraph;
  GE_ASSERT_SUCCESS(BackendUtils::GetNodeFusedGraph(node1, subgraph));
  for (const auto &n : subgraph->GetAllNodes()) {
    if (n->GetType() == kSplitType) {
      auto split_node = std::dynamic_pointer_cast<AscNode>(n);
      GE_ASSERT_NOTNULL(split_node);
      if (!IsConcatOrSplitFirstDim(split_node)) {
        GELOGI(
            "node1 %s(%s) and node2 %s(%s) can not fuse, the reason is [%s][neither Concat nor Split is operating on "
            "the first dim]",
            node1->GetNamePtr(), node1->GetType().c_str(), node2->GetNamePtr(), node2->GetType().c_str(),
            ge::NotFuseReasonCode(ge::NotFuseReason::kSplitCanNotFuseConcatBackward));
        return NOT_CHANGED;
      }
    }
  }
  GELOGD("[%s] split on first dim, can fuse split", concat_node->GetNamePtr());
  return SUCCESS;
}

AscNodePtr ConcatFusionStrategy::FindConcatNode(const NodePtr &backend_node) {
  ComputeGraphPtr graph;
  GE_ASSERT_SUCCESS(BackendUtils::GetNodeFusedGraph(backend_node, graph));
  for (const auto &node : graph->GetAllNodes()) {
    if (node->GetType() == kConcatType) {
      return std::dynamic_pointer_cast<AscNode>(node);
    } else if (node->GetType() == kAscBackendType) {
      ComputeGraphPtr subgraph;
      GE_ASSERT_SUCCESS(BackendUtils::GetNodeFusedGraph(node, subgraph));
      for (const auto &subnode : subgraph->GetAllNodes()) {
        if (subnode->GetType() == kConcatType) {
          return std::dynamic_pointer_cast<AscNode>(subnode);
        }
      }
    } else {
      // do nothing
    }
  }
  return nullptr;
}

bool ConcatFusionStrategy::IsConcatOrSplitFirstDim(const AscNodePtr &concat_node) {
  const auto &input_repeats = concat_node->inputs[0].attr.repeats;
  const auto &output_repeats = concat_node->outputs[0].attr.repeats;
  GE_ASSERT_TRUE(input_repeats.size() == output_repeats.size(),
                 "input_repeats.size() = %zu, mismatches output_repeats.size() = %zu", input_repeats.size(),
                 output_repeats.size());
  size_t non_one_count = 0U;
  size_t concat_dim = 0UL;
  bool is_first_dim = false;
  for (size_t i = 0U; i < input_repeats.size(); ++i) {
    if (SymbolicUtils::StaticCheckEq(input_repeats[i], output_repeats[i]) != kTrue) {
      concat_dim = i;
      is_first_dim = (non_one_count == 0);
      break;
    }
    if (SymbolicUtils::StaticCheckEq(input_repeats[i], ge::Symbol(1)) != kTrue) {
      ++non_one_count;
    }
  }
  is_first_dim = (is_first_dim || (concat_dim == 0UL));  // 单输入时，当成首轴转store处理
  GELOGI("node:%s input_shape = %s, output_shape = %s, is_first_dim = %d, diff_dim = %zu",
         concat_node->GetName().c_str(), ToString(input_repeats).c_str(), ToString(output_repeats).c_str(),
         is_first_dim, concat_dim);
  return is_first_dim;
}

REGISTER_FUSION_STRATEGY(ConcatFusionStrategy, loop::FuseType::kConcat);
}  // namespace ge
