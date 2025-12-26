/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "pow_equiv_substitution_pass.h"
#include <regex>
#include <queue>
#include "ascir_ops_utils.h"
#include "ascir_ops.h"
#include "graph_utils.h"
#include "symbolizer/symbolic_utils.h"
#include "schedule_utils.h"

namespace {
const double kRelEpsilonFloat = 1e-7;
const double kRelEpsilonInt = 1e-20;
const double kScalarHalf = 0.5f;
const uint32_t kScalarOne = 1U;
const double kScalarTwo = 2U;
constexpr uint64_t kScalarZeroMask = 0x7FFFFFFFFFFFFFFFUL;

enum class TargetValue : uint8_t {
  kZero,  // 0.0
  kHalf,  // 0.5
  kOne,   // 1.0
  kTwo,   // 2.0
  kNone
};

union DoubleBits {
  double d;
  uint64_t u;
};

bool IsEqual(double a, double b, double eps = kRelEpsilonFloat) {
  double diff = std::fabs(a - b);
  return diff < eps;
}

TargetValue CheckStringValue(const std::string &s) {
  DoubleBits val{0};
  std::istringstream iss(s);
  if (!(iss >> val.d)) {
    return TargetValue::kNone;
  }

  if (IsEqual(val.d, kScalarHalf)) {
    return TargetValue::kHalf;
  }

  if ((val.u & kScalarZeroMask) == 0UL) {
    return TargetValue::kZero;
  }

  if (IsEqual(val.d, kScalarOne, kRelEpsilonInt)) {
    return TargetValue::kOne;
  }

  if (IsEqual(val.d, kScalarTwo, kRelEpsilonInt)) {
    return TargetValue::kTwo;
  }
  return TargetValue::kNone;
}
}  // namespace

namespace optimize {
using ge::ops::IsOps;

Status PowEquivSubstitutionPass::RunPass(ge::AscGraph &graph) {
  auto all_nodes = graph.GetAllNodes();
  std::set<ge::NodePtr> data_nodes;
  for (const auto &node : all_nodes) {
    if (node == nullptr) {
      continue;
    }
    std::string scalar_val;
    if (!IsOps<ge::ascir_op::Pow>(node) || !GetScalarInput(node, scalar_val)) {
      continue;
    }
    switch (CheckStringValue(scalar_val)) {
      case TargetValue::kZero:
        GELOGD("Pow node [%s] scalar=0.0, replace with Scalar+Broadcast", node->GetName().c_str());
        GE_ASSERT_SUCCESS(ReplaceWithScalarBrc(graph, node, data_nodes));
        break;
      case TargetValue::kHalf:
        GELOGD("Pow node [%s] scalar=0.5, replace with Sqrt", node->GetName().c_str());
        GE_ASSERT_SUCCESS(ReplaceWithSqrt(graph, node, data_nodes));
        break;
      case TargetValue::kOne:
        GELOGD("Pow node [%s] scalar=1.0, remove useless node", node->GetName().c_str());
        GE_ASSERT_SUCCESS(RemoveUseLessPow(node, data_nodes));
        break;
      case TargetValue::kTwo:
        GELOGD("Pow node [%s] scalar=2.0, replace with Mul", node->GetName().c_str());
        GE_ASSERT_SUCCESS(ReplaceWithMul(graph, node, data_nodes));
        break;
      default:
        break;
    }
  }

  if (!useless_nodes_.empty()) {
    auto compute_graph = ge::AscGraphUtils::GetComputeGraph(graph);
    GE_ASSERT_NOTNULL(compute_graph);

    for (const auto &node : useless_nodes_) {
      GELOGD("Remove useless node: %s", node->GetNamePtr());
      GE_CHK_STATUS_RET(compute_graph->RemoveNode(node), "Failed to remove node: %s", node->GetNamePtr());
    }
    for (const auto &node : graph.GetAllNodes()) {
      if (!IsOps<ge::ascir_op::Output>(node) && !IsOps<ge::ascir_op::Workspace>(node)) {
        continue;
      }
      for (const auto &data_node : data_nodes) {
        ge::GraphUtils::AddEdge(data_node->GetOutControlAnchor(), node->GetInControlAnchor());
      }
      break;
    }
    GE_ASSERT_GRAPH_SUCCESS(ScheduleUtils::TopologicalSorting(graph));
  }

  return ge::SUCCESS;
}

bool PowEquivSubstitutionPass::GetScalarInput(const ge::AscNodePtr &pow_node, std::string &scalar_val) {
  auto pow_in_anchor = pow_node->GetInDataAnchor(1);
  while (pow_in_anchor != nullptr && pow_in_anchor->GetPeerOutAnchor() != nullptr) {
    auto target_node = std::dynamic_pointer_cast<ge::AscNode>(pow_in_anchor->GetPeerOutAnchor()->GetOwnerNode());
    GE_ASSERT_NOTNULL(target_node);
    if (IsOps<ge::ascir_op::Scalar>(target_node)) {
      auto ir_attr = target_node->attr.ir_attr.get();
      GE_ASSERT_NOTNULL(ir_attr);
      GE_ASSERT_SUCCESS(ir_attr->GetAttrValue("value", scalar_val));
      return true;
    } else if (IsOps<ge::ascir_op::Broadcast>(target_node)) {
      pow_in_anchor = target_node->GetInDataAnchor(0);
    } else {
      return false;
    }
  }
  return false;
}

Status PowEquivSubstitutionPass::ReplaceWithSqrt(ge::AscGraph &graph, const ge::AscNodePtr &pow_node,
                                                 std::set<ge::NodePtr> &data_nodes) {
  std::string sqrt_name = pow_node->GetName() + "_Sqrt";
  ge::ascir_op::Sqrt sqrt(sqrt_name.c_str());
  auto sqrt_node = graph.AddNode(sqrt);
  GE_ASSERT_NOTNULL(sqrt_node);
  sqrt_node->attr.sched = pow_node->attr.sched;
  sqrt_node->outputs[0].attr = pow_node->outputs[0].attr;

  auto pow_in_anchor = pow_node->GetInDataAnchor(0);
  GE_ASSERT_NOTNULL(pow_in_anchor);
  auto target_out = pow_in_anchor->GetPeerOutAnchor();
  GE_ASSERT_SUCCESS(ge::GraphUtils::RemoveEdge(target_out, pow_in_anchor));
  GE_ASSERT_SUCCESS(ge::GraphUtils::AddEdge(target_out, sqrt_node->GetInDataAnchor(0)));

  GE_ASSERT_SUCCESS(RelinkPowOutToTargetNode(pow_node, sqrt_node, data_nodes));
  return ge::SUCCESS;
}

Status PowEquivSubstitutionPass::ReplaceWithMul(ge::AscGraph &graph, const ge::AscNodePtr &pow_node,
                                                std::set<ge::NodePtr> &data_nodes) {
  std::string mul_name = pow_node->GetName() + "_Mul";
  ge::ascir_op::Mul mul(mul_name.c_str());
  auto mul_node = graph.AddNode(mul);
  GE_ASSERT_NOTNULL(mul_node);
  mul_node->attr.sched = pow_node->attr.sched;
  mul_node->outputs[0].attr = pow_node->outputs[0].attr;

  auto pow_in_anchor = pow_node->GetInDataAnchor(0);
  GE_ASSERT_NOTNULL(pow_in_anchor);
  auto target_out = pow_in_anchor->GetPeerOutAnchor();
  GE_ASSERT_SUCCESS(ge::GraphUtils::RemoveEdge(target_out, pow_in_anchor));
  GE_ASSERT_SUCCESS(ge::GraphUtils::AddEdge(target_out, mul_node->GetInDataAnchor(0)));
  GE_ASSERT_SUCCESS(ge::GraphUtils::AddEdge(target_out, mul_node->GetInDataAnchor(1)));

  GE_ASSERT_SUCCESS(RelinkPowOutToTargetNode(pow_node, mul_node, data_nodes));
  return ge::SUCCESS;
}

ge::Status PowEquivSubstitutionPass::RelinkPowOutToTargetNode(const ge::AscNodePtr &pow_node,
                                                              const ge::AscNodePtr &target_node,
                                                              std::set<ge::NodePtr> &data_nodes) {
  auto new_out_anchor = target_node->GetOutDataAnchor(0);
  auto pre_out_anchor = pow_node->GetOutDataAnchor(0);
  GE_ASSERT_NOTNULL(pre_out_anchor);
  for (const auto &cur_next_in_anchor : pre_out_anchor->GetPeerInDataAnchors()) {
    GE_ASSERT_SUCCESS(ge::GraphUtils::RemoveEdge(pre_out_anchor, cur_next_in_anchor));
    GE_ASSERT_SUCCESS(ge::GraphUtils::AddEdge(new_out_anchor, cur_next_in_anchor));
  }
  std::queue<ge::NodePtr> ques;
  std::set<ge::NodePtr> seen_nodes;
  ques.push(pow_node);
  while (!ques.empty()) {
    auto top = ques.front();
    seen_nodes.emplace(top);
    ques.pop();
    useless_nodes_.emplace(top);
    for (auto &input_node : top->GetInDataNodes()) {
      if (IsOps<ge::ascir_op::Data>(input_node)) {
        data_nodes.emplace(input_node);
        continue;
      }
      bool is_input_useless = true;
      for (const auto &target_out : input_node->GetOutDataNodes()) {
        if (useless_nodes_.count(target_out) == 0UL) {
          is_input_useless = false;
          break;
        }
      }
      if (is_input_useless && seen_nodes.count(input_node) == 0UL) {
        ques.push(input_node);
      }
    }
  }

  return ge::SUCCESS;
}

Status PowEquivSubstitutionPass::ReplaceWithScalarBrc(ge::AscGraph &graph, const ge::AscNodePtr &pow_node,
                                                      std::set<ge::NodePtr> &data_nodes) {
  std::string scalar_name = pow_node->GetName() + "_One";
  ge::ascir_op::Scalar scalar_one(scalar_name.c_str(), graph);
  scalar_one.ir_attr.SetValue(ge::SymbolicUtils::ToString(ge::sym::kSymbolOne));

  std::string brc_name = pow_node->GetName() + "_Brc";
  ge::ascir_op::Broadcast brc(brc_name.c_str());
  auto brc_node = graph.AddNode(brc);
  GE_ASSERT_NOTNULL(brc_node);
  brc_node->attr.sched = pow_node->attr.sched;
  brc_node->outputs[0].attr = pow_node->outputs[0].attr;
  brc.x = scalar_one.y;

  GE_ASSERT_SUCCESS(RelinkPowOutToTargetNode(pow_node, brc_node, data_nodes));
  return ge::SUCCESS;
}

Status PowEquivSubstitutionPass::RemoveUseLessPow(const ge::AscNodePtr &pow_node, std::set<ge::NodePtr> &data_nodes) {
  auto pow_in_anchor = pow_node->GetInDataAnchor(0);
  GE_ASSERT_NOTNULL(pow_in_anchor);
  auto target_out = pow_in_anchor->GetPeerOutAnchor();
  GE_ASSERT_NOTNULL(target_out);
  auto pre_node = std::dynamic_pointer_cast<ge::AscNode>(target_out->GetOwnerNode());
  GE_ASSERT_NOTNULL(pre_node);

  GE_ASSERT_SUCCESS(ge::GraphUtils::RemoveEdge(target_out, pow_in_anchor));
  GE_ASSERT_SUCCESS(RelinkPowOutToTargetNode(pow_node, pre_node, data_nodes));
  GELOGI("Successfully removed useless Pow node [%s]", pow_node->GetName().c_str());
  return ge::SUCCESS;
}
}  // namespace optimize
