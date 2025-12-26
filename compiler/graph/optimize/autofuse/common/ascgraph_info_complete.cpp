/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ascgraph_info_complete.h"
#include <map>
#include <queue>
#include "ascir_ops.h"
#include "ascendc_ir_def.h"
#include "graph/symbolizer/symbolic.h"
#include "graph/attribute_group/attr_group_shape_env.h"
#include "ascir_ops_utils.h"

using namespace ge::ascir_op;

namespace optimize {
namespace {
static Status GetNodeIrAttrOffset(const ge::NodePtr &node, ge::Expression &offset) {
  auto asc_node = std::dynamic_pointer_cast<ge::AscNode>(node);
  GE_ASSERT_NOTNULL(asc_node);
  GE_ASSERT_NOTNULL(asc_node->attr.ir_attr);
  return asc_node->attr.ir_attr->GetAttrValue("offset", offset);
}

void InsertFreeSymbolsIntoVarSet(const ge::Expression &exp, SizeVarSet &size_vars) {
  std::vector<ge::Expression> free_symbols = exp.FreeSymbols();
  size_vars.insert(free_symbols.begin(), free_symbols.end());
}

void CompleteDataApiInfo(ge::AscNodePtr &node) {
  node->attr.api.compute_type = ge::ComputeType::kComputeInvalid;
  node->attr.api.type = ge::ApiType::kAPITypeBuffer;
  node->attr.api.unit = ge::ComputeUnit::kUnitNone;
}

void CompleteLoadApiInfo(ge::AscNodePtr &node) {
  node->attr.api.compute_type = ge::ComputeType::kComputeLoad;
  node->attr.api.type = ge::ApiType::kAPITypeCompute;
  node->attr.api.unit = ge::ComputeUnit::kUnitMTE2;
}

void CompleteStoreApiInfo(ge::AscNodePtr &node) {
  node->attr.api.compute_type = ge::ComputeType::kComputeStore;
  node->attr.api.type = ge::ApiType::kAPITypeCompute;
  node->attr.api.unit = ge::ComputeUnit::kUnitMTE2;
}

void CompleteElewiseApiInfo(ge::AscNodePtr &node) {
  node->attr.api.compute_type = ge::ComputeType::kComputeElewise;
  node->attr.api.type = ge::ApiType::kAPITypeCompute;
  node->attr.api.unit = ge::ComputeUnit::kUnitVector;
}

void CompleteBroadcastApiInfo(ge::AscNodePtr &node) {
  node->attr.api.compute_type = ge::ComputeType::kComputeBroadcast;
  node->attr.api.type = ge::ApiType::kAPITypeCompute;
  node->attr.api.unit = ge::ComputeUnit::kUnitVector;
}

void CompleteReduceApiInfo(ge::AscNodePtr &node) {
  node->attr.api.compute_type = ge::ComputeType::kComputeReduce;
  node->attr.api.type = ge::ApiType::kAPITypeCompute;
  node->attr.api.unit = ge::ComputeUnit::kUnitVector;
}

void CompleteConcatApiInfo(ge::AscNodePtr &node) {
  node->attr.api.compute_type = ge::ComputeType::kComputeConcat;
  node->attr.api.type = ge::ApiType::kAPITypeCompute;
  node->attr.api.unit = ge::ComputeUnit::kUnitVector;
}

void CompleteSplitApiInfo(ge::AscNodePtr &node) {
  node->attr.api.compute_type = ge::ComputeType::kComputeSplit;
  node->attr.api.type = ge::ApiType::kAPITypeCompute;
  node->attr.api.unit = ge::ComputeUnit::kUnitVector;
}

void CompleteGatherApiInfo(ge::AscNodePtr &node) {
  if (node->attr.api.compute_type >= ge::ComputeType::kComputeInvalid) {
    node->attr.api.compute_type = ge::ComputeType::kComputeGather;
  }
  node->attr.api.type = ge::ApiType::kAPITypeCompute;
  node->attr.api.unit = ge::ComputeUnit::kUnitMTE2;
}

void CompleteCubeApiInfo(ge::AscNodePtr &node) {
  node->attr.api.compute_type = ge::ComputeType::kComputeCube;
  node->attr.api.type = ge::ApiType::kAPITypeCompute;
  node->attr.api.unit = ge::ComputeUnit::kUnitCube;
}
}  // namespace

using CompleteApiInfoFunc = std::function<void(ge::AscNodePtr &)>;
struct Completer {
  CompleteApiInfoFunc complete_api_info;
};

void CompleteTransposeApiInfo(ge::AscNodePtr &node) {
  node->attr.api.compute_type = ge::ComputeType::kComputeTranspose;
  node->attr.api.type = ge::ApiType::kAPITypeCompute;
  node->attr.api.unit = ge::ComputeUnit::kUnitVector;
}

static const std::map<std::string, Completer> kOpTypeToCompleter = {
    {Workspace::Type, {&CompleteDataApiInfo}},      {Data::Type, {&CompleteDataApiInfo}},
    {Scalar::Type, {&CompleteDataApiInfo}},         {Output::Type, {&CompleteDataApiInfo}},
    {IndexExpr::Type, {&CompleteDataApiInfo}},      {Nddma::Type, {&CompleteLoadApiInfo}},

    {Load::Type, {&CompleteLoadApiInfo}},           {Store::Type, {&CompleteStoreApiInfo}},

    {Sum::Type, {&CompleteReduceApiInfo}},          {Max::Type, {&CompleteReduceApiInfo}},
    {Mean::Type, {&CompleteReduceApiInfo}},         {Min::Type, {&CompleteReduceApiInfo}},
    {Prod::Type, {&CompleteReduceApiInfo}},         {All::Type, {&CompleteReduceApiInfo}},
    {Any::Type, {&CompleteReduceApiInfo}},

    {Broadcast::Type, {&CompleteBroadcastApiInfo}},
    {RemovePad::Type, {&CompleteElewiseApiInfo}},
    {Pad::Type, {&CompleteElewiseApiInfo}},

    {Cast::Type, {&CompleteElewiseApiInfo}},        {Abs::Type, {&CompleteElewiseApiInfo}},
    {Neg::Type, {&CompleteElewiseApiInfo}},         {Exp::Type, {&CompleteElewiseApiInfo}},
    {Sqrt::Type, {&CompleteElewiseApiInfo}},        {Rsqrt::Type, {&CompleteElewiseApiInfo}},
    {Relu::Type, {&CompleteElewiseApiInfo}},        {Reciprocal::Type, {&CompleteElewiseApiInfo}},
    {Erf::Type, {&CompleteElewiseApiInfo}},         {Sign::Type, {&CompleteElewiseApiInfo}},
    {Tanh::Type, {&CompleteElewiseApiInfo}},        {Isnan::Type, {&CompleteElewiseApiInfo}},
    {IsFinite::Type, {&CompleteElewiseApiInfo}},    {Ln::Type, {&CompleteElewiseApiInfo}},
    {LogicalNot::Type, {&CompleteElewiseApiInfo}},

    {Add::Type, {&CompleteElewiseApiInfo}},         {Sub::Type, {&CompleteElewiseApiInfo}},
    {Mul::Type, {&CompleteElewiseApiInfo}},         {Div::Type, {&CompleteElewiseApiInfo}},
    {TrueDiv::Type, {&CompleteElewiseApiInfo}},     {Minimum::Type, {&CompleteElewiseApiInfo}},
    {Maximum::Type, {&CompleteElewiseApiInfo}},     {LogicalOr::Type, {&CompleteElewiseApiInfo}},
    {LogicalAnd::Type, {&CompleteElewiseApiInfo}},

    {Ge::Type, {&CompleteElewiseApiInfo}},          {Eq::Type, {&CompleteElewiseApiInfo}},
    {Ne::Type, {&CompleteElewiseApiInfo}},          {Gt::Type, {&CompleteElewiseApiInfo}},
    {Le::Type, {&CompleteElewiseApiInfo}},          {Lt::Type, {&CompleteElewiseApiInfo}},
    {Broadcast::Type, {&CompleteElewiseApiInfo}},   {Sigmoid::Type, {&CompleteElewiseApiInfo}},
    {Concat::Type, {&CompleteConcatApiInfo}},       {Gather::Type, {&CompleteGatherApiInfo}},

    {Where::Type, {&CompleteElewiseApiInfo}},       {Select::Type, {&CompleteElewiseApiInfo}},
    {ClipByValue::Type, {&CompleteElewiseApiInfo}}, {Pow::Type, {&CompleteElewiseApiInfo}},
    {Transpose::Type, {&CompleteTransposeApiInfo}},
    {BitwiseAnd::Type, {&CompleteElewiseApiInfo}},  {LeakyRelu::Type, {&CompleteElewiseApiInfo}},
    {FloorDiv::Type, {&CompleteElewiseApiInfo}},    {Gelu::Type, {&CompleteElewiseApiInfo}},
    {Axpy::Type, {&CompleteElewiseApiInfo}},
    {Split::Type, {&CompleteSplitApiInfo}},
    {MatMul::Type, {&CompleteCubeApiInfo}},         {MatMulBias::Type, {&CompleteCubeApiInfo}},
    {MatMulOffset::Type, {&CompleteCubeApiInfo}},   {MatMulOffsetBias::Type, {&CompleteCubeApiInfo}},
    {BatchMatMul::Type, {&CompleteCubeApiInfo}},    {BatchMatMulBias::Type, {&CompleteCubeApiInfo}},
    {BatchMatMulOffset::Type, {&CompleteCubeApiInfo}},
    {BatchMatMulOffsetBias::Type, {&CompleteCubeApiInfo}},
};

Status AscGraphInfoComplete::CompleteApiInfo(const ge::AscGraph &optimize_graph) {
  for (auto node : optimize_graph.GetAllNodes()) {
    auto it = kOpTypeToCompleter.find(node->GetType());
    GE_ASSERT_TRUE((it != kOpTypeToCompleter.end()), "CompleteApiInfo unsupported node name:[%s], type: [%s].",
                   node->GetNamePtr(), node->GetTypePtr());
    it->second.complete_api_info(node);
  }
  return ge::SUCCESS;
}

void AscGraphInfoComplete::AppendOriginalSizeVar(const ge::AscGraph &graph, SizeVarSet &size_vars) {
  auto axes = graph.GetAllAxis();
  for (const auto &axis : axes) {
    InsertFreeSymbolsIntoVarSet(axis->size, size_vars);
  }
  auto all_nodes = graph.GetAllNodes();
  for (const auto &node : all_nodes) {
    if (!ge::ops::IsOps<Store>(node) && !ge::ops::IsOps<Load>(node) && !ge::ops::IsOps<Gather>(node)) {
      continue;
    }

    ge::Expression cur_load_offset;
    if (GetNodeIrAttrOffset(node, cur_load_offset) == ge::SUCCESS) {
      InsertFreeSymbolsIntoVarSet(cur_load_offset, size_vars);
    }

    if (ge::ops::IsOps<Gather>(node)) {
      for (const auto &exp : node->inputs[0].attr.repeats) {
        InsertFreeSymbolsIntoVarSet(exp, size_vars);
      }
    }

    for (const auto &exp : node->outputs[0].attr.repeats) {
      InsertFreeSymbolsIntoVarSet(exp, size_vars);
    }
    for (const auto &exp : node->outputs[0].attr.strides) {
      InsertFreeSymbolsIntoVarSet(exp, size_vars);
    }
  }
}
}  // namespace optimize
