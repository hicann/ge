/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "symbolic_infer_util.h"

#include <algorithm>
#include <vector>
#include "graph/utils/node_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/attr_utils.h"
#include "graph/debug/ge_attr_define.h"
#include "base/registry/op_impl_space_registry_v2.h"

#include <op_type_utils.h>

namespace ge {
constexpr static size_t kByteBitCount = 8UL;

namespace {
constexpr const char *const kValueDependentIdxsAttr = "_ge_value_dependent_idxs";
}  // namespace

graphStatus SymbolicInferUtil::GetConstInt(const gert::SymbolTensor *tensor, DataType dt, int64_t &value) {
  if (dt == DT_INT32) {
    int32_t tmp_value = 0;
    GE_ASSERT_TRUE(tensor->GetSymbolicValue()->at(0).GetConstValue<int32_t>(tmp_value),
                   "error info GetConstValue failed");
    value = static_cast<int64_t>(tmp_value);
  } else if (dt == DT_INT64) {
    GE_ASSERT_TRUE(tensor->GetSymbolicValue()->at(0).GetConstValue<int64_t>(value), "error info GetConstValue failed");
  } else {
    GELOGE(PARAM_INVALID, "dt must in [int32, int64]");
    return ge::PARAM_INVALID;
  }
  return ge::GRAPH_SUCCESS;
}
Status SymbolicInferUtil::Broadcast(const std::vector<std::vector<Expression>> &shapes,
                                    std::vector<Expression> &b_shape) {
  if (shapes.empty()) {
    b_shape.clear();
    return SUCCESS;
  }
  // 1. 挑选rank最大值
  size_t max_rank = 0;
  for (const auto &shape : shapes) {
    max_rank = std::max(max_rank, shape.size());
  }
  // 2. 初始化输出结果和广播标志
  b_shape.clear();
  // 3. 右对齐方式计算broadcast，比如[4, 2, 3]和[2, 3]，输出为[4, 2, 3]
  for (size_t dim = 0U; dim < max_rank; ++dim) {
    bool found = false;
    for (const auto &shape : shapes) {
      int32_t true_idx = dim + shape.size() - max_rank;
      if (true_idx >= 0) {
        const Expression &s = shape[true_idx];
        if (!found) {
          b_shape.emplace_back(s);
          found = true;
          continue;
        }
        if (EXPECT_SYMBOL_EQ(b_shape[dim], s)) {
          if (s.IsConstExpr()) {
            b_shape[dim] = s;
          }
          GELOGI("Symbol input0[%zu]:%s is equal to input1[%zu]:%s, no need broadcast.", true_idx,
                 b_shape[dim].Serialize().get(), true_idx, s.Serialize().get());
        } else if (EXPECT_SYMBOL_EQ(s, Symbol(1))) {
          GELOGI("Symbol input1[%zu]:%s is equal to symbol(1), should broadcast to input0[%zu]:%s.", true_idx,
                 s.Serialize().get(), true_idx, b_shape[dim].Serialize().get());
        } else if (EXPECT_SYMBOL_EQ(b_shape[dim], Symbol(1))) {
          GELOGI("Symbol input0[%zu]:%s is equal to symbol(1), should broadcast to input1[%zu]:%s.", true_idx,
                 b_shape[dim].Serialize().get(), true_idx, s.Serialize().get());
          b_shape[dim] = s;
        } else {
          GELOGE(ge::FAILED, "Symbol input0[%zu]:%s is not equal to input1[%zu]:%s which cannot broadcast.", true_idx,
                 b_shape[dim].Serialize().get(), true_idx, s.Serialize().get());
          return FAILED;
        }
      }
    }
  }
  return SUCCESS;
}

std::string SymbolicInferUtil::DumpSymbolTensor(const gert::SymbolTensor &symbolic_tensor) {
  std::string debug_msg = "origin symbol shape: ";
  debug_msg += SymbolicInferUtil::VectorExpressionToStr(symbolic_tensor.GetOriginSymbolShape().GetDims());
  debug_msg += ", symbolic value: ";
  if (symbolic_tensor.GetSymbolicValue() != nullptr) {
    debug_msg += SymbolicInferUtil::VectorExpressionToStr(*symbolic_tensor.GetSymbolicValue());
  } else {
    debug_msg += "None";
  }
  return debug_msg;
}

bool SymbolicInferUtil::IsSupportCondNode(const ge::NodePtr &node) {
  GE_WARN_ASSERT(node != nullptr);
  std::string node_type = node->GetType();
  return (kIfOpTypes.find(node_type) != kIfOpTypes.end()) || (kCaseOpTypes.find(node_type) != kCaseOpTypes.end());
}

NodePtr SymbolicInferUtil::GetCondInput(const NodePtr &node) {
  GE_WARN_ASSERT(IsSupportCondNode(node));
  auto cond_input = NodeUtils::GetInDataNodeByIndex(*node, 0);
  GELOGD("Get cond node[%s] input[%s] success.", node->GetNamePtr(), cond_input->GetNamePtr());
  GE_WARN_ASSERT(cond_input != nullptr);
  // 如果node节点的输入时cast，size，stringlength需要再往上找
  const std::set<string> vaild_types = {"Cast", "Size", "StringLength"};
  if (vaild_types.find(cond_input->GetType()) != vaild_types.end()) {
    cond_input = NodeUtils::GetInDataNodeByIndex(*cond_input, 0);
    GELOGD("Cond node[%s] input is cast/size/stringlength, get input[%s] success.", node->GetNamePtr(),
           cond_input->GetNamePtr());
    GE_ASSERT_NOTNULL(cond_input);
  }
  // 如果不是data节点, 直接返回
  if (!OpTypeUtils::IsDataNode(cond_input->GetType())) {
    return cond_input;
  }
  // 如果是data节点找根图的节点
  auto parent_input = NodeUtils::GetParentInput(*cond_input);
  return parent_input == nullptr ? cond_input : parent_input;
}

bool SymbolicInferUtil::IsValueDependentDataNode(const NodePtr &data_node) {
  const auto space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
  for (const auto *out_anchor : data_node->GetAllOutDataAnchorsPtr()) {
    if (out_anchor == nullptr) {
      continue;
    }
    for (const auto *peer_anchor : out_anchor->GetPeerInDataAnchorsPtr()) {
      if (peer_anchor == nullptr) {
        continue;
      }
      auto *owner_node = peer_anchor->GetOwnerNodeBarePtr();
      if (owner_node == nullptr) {
        continue;
      }
      const auto &consumer_op = owner_node->GetOpDesc();
      if (consumer_op == nullptr) {
        continue;
      }
      const size_t input_idx = static_cast<size_t>(peer_anchor->GetIdx());

      auto functions = gert::OpImplInferSymbolShapeRegistry::GetInstance().GetOpImpl(consumer_op->GetType().c_str());
      if (functions != nullptr) {
        const gert::OpImplKernelRegistry::OpImplFunctionsV2 *function_new = functions;
        if (space_registry != nullptr) {
          const auto *space_func = space_registry->GetOpImpl(consumer_op->GetType().c_str());
          if (space_func != nullptr) {
            function_new = space_func;
          }
        }
        size_t ir_index = 0UL;
        if (ge::OpDescUtils::GetInputIrIndexByInstanceIndex(consumer_op, input_idx, ir_index) != GRAPH_SUCCESS) {
          ir_index = input_idx;
        }
        if (function_new->IsInputDataDependency(ir_index)) {
          GELOGI("data node %s is value-dependent, consumer %s input idx %zu is data dependency.",
                 data_node->GetNamePtr(), consumer_op->GetNamePtr(), input_idx);
          return true;
        }
      }

      const auto &op_infer_depends = consumer_op->GetOpInferDepends();
      if (op_infer_depends.empty()) {
        continue;
      }
      auto input_name = consumer_op->GetValidInputNameByIndex(static_cast<uint32_t>(input_idx));
      if (std::find(op_infer_depends.cbegin(), op_infer_depends.cend(), input_name) != op_infer_depends.cend()) {
        GELOGI("data node %s is value-dependent, consumer %s input name %s in op_infer_depends.",
               data_node->GetNamePtr(), consumer_op->GetNamePtr(), input_name.c_str());
        return true;
      }
    }
  }
  return false;
}

Status SymbolicInferUtil::GetValueDependentInputIdxs(const ComputeGraphPtr &graph,
                                                     std::set<size_t> &value_dependent_idxs) {
  if (graph == nullptr) {
    return SUCCESS;
  }
  std::vector<int64_t> cached_idxs;
  if (ge::AttrUtils::GetListInt(graph, kValueDependentIdxsAttr, cached_idxs)) {
    for (const auto idx : cached_idxs) {
      value_dependent_idxs.insert(static_cast<size_t>(idx));
    }
    return SUCCESS;
  }
  std::set<size_t> computed_idxs;
  for (const auto &node : graph->GetDirectNode()) {
    if (node == nullptr) {
      continue;
    }
    const auto &op_desc = node->GetOpDesc();
    if (op_desc == nullptr || !OpTypeUtils::IsDataNode(op_desc->GetType())) {
      continue;
    }
    int32_t data_index = -1;
    (void)AttrUtils::GetInt(op_desc, ATTR_NAME_INDEX, data_index);
    if (data_index >= 0 && IsValueDependentDataNode(node)) {
      computed_idxs.insert(static_cast<size_t>(data_index));
      GELOGI("graph %s input data index %d is value-dependent.", graph->GetName().c_str(), data_index);
    }
  }
  std::vector<int64_t> cached_vec(computed_idxs.cbegin(), computed_idxs.cend());
  (void)ge::AttrUtils::SetListInt(graph, kValueDependentIdxsAttr, cached_vec);
  value_dependent_idxs.insert(computed_idxs.cbegin(), computed_idxs.cend());
  return SUCCESS;
}

}  // namespace ge
