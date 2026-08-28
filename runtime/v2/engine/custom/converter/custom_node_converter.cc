/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "custom_node_converter.h"
#include "graph_builder/converter_checker.h"
#include "graph_builder/bg_tensor.h"
#include "graph_builder/bg_infer_shape.h"
#include "common/checker.h"
#include "exe_graph/lowering/frame_selector.h"
#include "kernel/common_kernel_impl/build_tensor.h"
#include "graph/custom_op_registry.h"
#include "lowering/placement/placed_lowering_result.h"
#include "exe_graph/lowering/lowering_definitions.h"
#include "exe_graph/runtime/eager_op_execution_context.h"
#include "common/ge_common/ge_types.h"
#include "graph/utils/inference_rule.h"
#include "exe_graph/lowering/data_dependent_interpreter.h"
#include "graph/utils/attr_utils.h"
#include "graph/utils/op_desc_utils.h"

namespace gert {
namespace {
// Default threshold for non-tensor InputKind values. When the "_custom_op_non_tensor_kind_base"
// attribute is absent (e.g. older OM models without the attribute), this value is used so that
// InputKind enums starting from 3 (INT, FLOAT, BOOL, ...) are recognised as non-tensor.
constexpr int64_t kDefaultNonTensorInputKindBase = 3L;

bool NeedHostInput(const ge::OpDescPtr &op_desc, size_t instance_index, const gert::DataDependentInterpreter &ddi) {
  std::vector<int64_t> input_kinds;
  if (!ge::AttrUtils::GetListInt(op_desc, "input_kinds", input_kinds) || input_kinds.empty()) {
    return false;
  }
  size_t ir_index = 0U;
  if (ge::OpDescUtils::GetInputIrIndexByInstanceIndex(op_desc, instance_index, ir_index) != ge::SUCCESS) {
    return false;
  }
  // Read the non-tensor kind base from the attribute set by codegen; fall back to default for
  // backward compatibility with OMs that do not carry this attribute.
  int64_t non_tensor_base = kDefaultNonTensorInputKindBase;
  (void)ge::AttrUtils::GetInt(op_desc, "_custom_op_non_tensor_kind_base", non_tensor_base);
  if (ir_index >= input_kinds.size() || input_kinds[ir_index] < non_tensor_base) {
    return false;
  }
  bool is_data_dependent = false;
  if (ddi.IsDataDependent(static_cast<int32_t>(instance_index), is_data_dependent) != ge::GRAPH_SUCCESS) {
    return false;
  }
  if (is_data_dependent) {
    GELOGI("NeedHostInput: op=%s, input instance_index=%zu, ir_index=%zu, input_kind=%ld, non_tensor_base=%ld",
           op_desc->GetNamePtr(), instance_index, ir_index, input_kinds[ir_index], non_tensor_base);
  }
  return is_data_dependent;
}

bool NeedCustomOpInferShape(const ge::NodePtr &node, const LoweringGlobalData &global_data) {
  GE_ASSERT_NOTNULL(node);
  const auto op_desc = node->GetOpDesc();
  GE_ASSERT_NOTNULL(op_desc);
  if (bg::FindShapeInferOpInCustomOpRegistry(node->GetTypePtr(), global_data) != nullptr) {
    return true;
  }
  const std::string infer_rule = ge::InferenceRule::GetInferenceRule(op_desc);
  if (!infer_rule.empty()) {
    return true;
  }
  return false;
}

bg::ValueHolderPtr FindCustomExecutorFunc(const ge::NodePtr &node, const LowerInput &lower_input) {
  auto builder = [&node, &lower_input]() -> std::vector<bg::ValueHolderPtr> {
    return bg::FrameSelector::OnInitRoot([&node, &lower_input]() -> std::vector<bg::ValueHolderPtr> {
      auto node_type = bg::ValueHolder::CreateConst(node->GetTypePtr(), node->GetType().size() + 1, true);
      ge::CustomOpRegistry *custom_op_registry = lower_input.global_data->GetCustomOpRegistry().get();
      auto registry_holder = bg::ValueHolder::CreateConst(&custom_op_registry, sizeof(custom_op_registry));
      return {bg::ValueHolder::CreateSingleDataOutput("FindCustomOp", {node_type, registry_holder})};
    });
  };
  return lower_input.global_data->GetOrCreateUniqueValueHolder(node->GetType() + "_FindCustomOp_", builder)[0];
}

bg::ValueHolderPtr FindHostCpuCustomExecutorFunc(const ge::NodePtr &node, const LowerInput &lower_input) {
  auto builder = [&node, &lower_input]() -> std::vector<bg::ValueHolderPtr> {
    return bg::FrameSelector::OnInitRoot([&node, &lower_input]() -> std::vector<bg::ValueHolderPtr> {
      auto node_type = bg::ValueHolder::CreateConst(node->GetTypePtr(), node->GetType().size() + 1, true);
      ge::CustomOpRegistry *custom_op_registry = lower_input.global_data->GetCustomOpRegistry().get();
      auto registry_holder = bg::ValueHolder::CreateConst(&custom_op_registry, sizeof(custom_op_registry));
      return {bg::ValueHolder::CreateSingleDataOutput("FindHostCpuCustomOp", {node_type, registry_holder})};
    });
  };
  return lower_input.global_data->GetOrCreateUniqueValueHolder(node->GetType() + "_FindHostCpuCustomOp_", builder)[0];
}

ge::graphStatus BuildInputTensors(const ge::NodePtr &node, const LowerInput &lower_input,
                                  std::vector<bg::ValueHolderPtr> &input_tensor_holders,
                                  std::vector<bg::ValueHolderPtr> &input_addr_holders) {
  const ge::OpDescPtr op_desc = node->GetOpDesc();
  gert::DataDependentInterpreter ddi(op_desc, lower_input.global_data->GetSpaceRegistriesV2());
  size_t instance_index = 0U;
  for (const ge::InDataAnchorPtr &in_data_anchor : node->GetAllInDataAnchors()) {
    GE_ASSERT_NOTNULL(in_data_anchor);
    // optional场景
    ge::OutDataAnchorPtr out_data_anchor = in_data_anchor->GetPeerOutAnchor();
    if (out_data_anchor == nullptr) {
      continue;
    }
    ge::NodePtr peer_node = out_data_anchor->GetOwnerNode();
    GE_ASSERT_NOTNULL(peer_node);
    const auto *const_lower_result = peer_node->GetOpDesc()->GetExtAttr<PlacedLoweringResult>(kLoweringResult);
    GE_ASSERT_NOTNULL(const_lower_result, "Lowering result of node [%s, %s] is not found.", peer_node->GetNamePtr(),
                      peer_node->GetTypePtr());
    auto *lower_result = const_cast<PlacedLoweringResult *>(const_lower_result);
    GE_ASSERT_NOTNULL(lower_result);
    const int32_t input_placement = NeedHostInput(op_desc, instance_index, ddi) ? kOnHost : kOnDeviceHbm;
    const OutputLowerResult *result = lower_result->GetOutputTensorResult(
        *lower_input.global_data, out_data_anchor->GetIdx(), {input_placement, node->GetOpDesc()->GetStreamId()});
    GE_ASSERT_NOTNULL(result, "Lowering result of node [%s, %s] output[%d] is nullptr.", peer_node->GetNamePtr(),
                      peer_node->GetTypePtr(), out_data_anchor->GetIdx());
    GE_ASSERT_NOTNULL(result->shape);
    input_tensor_holders.emplace_back(result->shape);
    input_addr_holders.emplace_back(result->address);
    ++instance_index;
  }
  GE_ASSERT_TRUE(lower_input.input_shapes.size() == input_tensor_holders.size(),
                 "Size[%zu] of input shapes and size[%zu] of input tensor is not same.",
                 lower_input.input_shapes.size(), input_tensor_holders.size());
  return ge::SUCCESS;
}

ge::graphStatus BuildHostInputTensors(const ge::NodePtr &node, const LowerInput &lower_input,
                                      std::vector<bg::ValueHolderPtr> &input_tensor_holders,
                                      std::vector<bg::ValueHolderPtr> &input_addr_holders) {
  for (const ge::InDataAnchorPtr &in_data_anchor : node->GetAllInDataAnchors()) {
    GE_ASSERT_NOTNULL(in_data_anchor);
    // optional场景
    ge::OutDataAnchorPtr out_data_anchor = in_data_anchor->GetPeerOutAnchor();
    if (out_data_anchor == nullptr) {
      continue;
    }
    ge::NodePtr peer_node = out_data_anchor->GetOwnerNode();
    GE_ASSERT_NOTNULL(peer_node);
    const auto *const_lower_result = peer_node->GetOpDesc()->GetExtAttr<PlacedLoweringResult>(kLoweringResult);
    GE_ASSERT_NOTNULL(const_lower_result, "Lowering result of node [%s, %s] is not found.", peer_node->GetNamePtr(),
                      peer_node->GetTypePtr());
    auto *lower_result = const_cast<PlacedLoweringResult *>(const_lower_result);
    GE_ASSERT_NOTNULL(lower_result);
    const OutputLowerResult *result = lower_result->GetOutputTensorResult(
        *lower_input.global_data, out_data_anchor->GetIdx(), {kOnHost, node->GetOpDesc()->GetStreamId()});
    GE_ASSERT_NOTNULL(result, "Lowering result of node [%s, %s] output[%d] is nullptr.", peer_node->GetNamePtr(),
                      peer_node->GetTypePtr(), out_data_anchor->GetIdx());
    GE_ASSERT_NOTNULL(result->shape);
    input_tensor_holders.emplace_back(result->shape);
    input_addr_holders.emplace_back(result->address);
  }
  GE_ASSERT_TRUE(lower_input.input_shapes.size() == input_tensor_holders.size(),
                 "Size[%zu] of input shapes and size[%zu] of input tensor is not same.",
                 lower_input.input_shapes.size(), input_tensor_holders.size());
  return ge::SUCCESS;
}
}  // namespace

LowerResult LoweringCustomNode(const ge::NodePtr &node, const LowerInput &lower_input) {
  LOWER_REQUIRE_HYPER_SUCCESS(CheckLowerInput(lower_input));
  std::vector<bg::ValueHolderPtr> input_holders;
  std::vector<bg::ValueHolderPtr> input_addr_holders;
  LOWER_REQUIRE_SUCCESS(BuildInputTensors(node, lower_input, input_holders, input_addr_holders));
  // Allocate
  auto allocator_holder =
      lower_input.global_data->GetOrCreateAllocator({kOnDeviceHbm, AllocatorUsage::kAllocNodeOutput});
  auto custom_executor_func = FindCustomExecutorFunc(node, lower_input);
  // Create op executeFunc
  input_holders.emplace_back(allocator_holder);
  input_holders.emplace_back(lower_input.global_data->GetStream());
  input_holders.emplace_back(custom_executor_func);
  // Check inference_rule
  const auto op_desc = node->GetOpDesc();
  LOWER_REQUIRE_NOTNULL(op_desc);
  std::string kernel_type = "ExecuteCustomOp";
  std::vector<bg::ValueHolderPtr> infer_output_shapes;
  if (NeedCustomOpInferShape(node, *lower_input.global_data)) {
    kernel_type = "ExecuteCustomOpWithInferShape";
    infer_output_shapes = bg::InferCustomOpShape(node, lower_input.input_shapes, *lower_input.global_data);
    input_holders.insert(input_holders.end(), infer_output_shapes.begin(), infer_output_shapes.end());
  }
  // 最后需要workspace地址和args_handler地址
  std::vector<bg::ValueHolderPtr> output_tensor_holders = bg::ValueHolder::CreateDataOutput(
      kernel_type.c_str(), input_holders,
      node->GetAllOutDataAnchorsSize() +
          static_cast<size_t>(gert::EagerOpExecutionContext::AdditionalOutputIndex::kNum));
  std::vector<bg::ValueHolderPtr> output_shapes;
  std::vector<bg::DevMemValueHolderPtr> output_addrs;
  for (size_t i = 0UL; i < node->GetAllOutDataAnchorsSize(); i++) {
    auto split_outputs = bg::DevMemValueHolder::CreateDataOutput(
        kernel::kSplitDataTensor, {output_tensor_holders[i], allocator_holder},
        static_cast<size_t>(kernel::SplitTensorOutputs::kNum), op_desc->GetStreamId());
    CONVERTER_CHECK_HOLDERS_ALL_OK(split_outputs, static_cast<size_t>(kernel::SplitTensorOutputs::kNum));
    LOWER_REQUIRE_NOTNULL(bg::ValueHolder::CreateVoidGuarder(
        "FreeMemory", split_outputs[static_cast<size_t>(kernel::SplitTensorOutputs::kTensorData)], {}));
    output_shapes.emplace_back(split_outputs[static_cast<size_t>(kernel::SplitTensorOutputs::kShape)]);
    output_addrs.emplace_back(split_outputs[static_cast<size_t>(kernel::SplitTensorOutputs::kTensorData)]);
    for (size_t j = 0UL; j < input_addr_holders.size(); j++) {
      auto guarder = input_addr_holders[j]->GetGuarder();
      if (guarder != nullptr) {
        GELOGI("[Dumper] Add dependency from node %s input[%zu] to output[%zu] of node %s.", node->GetNamePtr(), j, i,
               node->GetNamePtr());
        GE_ASSERT_HYPER_SUCCESS(bg::ValueHolder::AddDependency(
            split_outputs[static_cast<size_t>(kernel::SplitTensorOutputs::kTensorData)], guarder));
      }
    }
  }
  LOWER_REQUIRE_NOTNULL(bg::ValueHolder::CreateVoidGuarder(
      "FreeCustomOpWorkspaces", output_tensor_holders[node->GetAllOutDataAnchorsSize()], {allocator_holder}));
  LOWER_REQUIRE_NOTNULL(bg::ValueHolder::CreateVoidGuarder(
      "FreeArgsGuarder",
      output_tensor_holders[node->GetAllOutDataAnchorsSize() +
                            static_cast<size_t>(gert::EagerOpExecutionContext::AdditionalOutputIndex::kArgsHandler)],
      {}));
  // 输入tensor需要添加对地址guard的依赖边，否则会出现提前释放
  for (auto &addr : input_addr_holders) {
    auto guarder = addr->GetGuarder();
    if (guarder != nullptr) {
      GE_ASSERT_HYPER_SUCCESS(bg::ValueHolder::AddDependency(output_tensor_holders.front(), guarder));
    }
  }
  return {HyperStatus::Success(), {}, output_shapes, output_addrs};
}
REGISTER_NODE_CONVERTER_PLACEMENT(ge::kCustomOpKernelLibName.c_str(), kOnDeviceHbm, LoweringCustomNode);

LowerResult LoweringHostCustomNode(const ge::NodePtr &node, const LowerInput &lower_input) {
  LOWER_REQUIRE_HYPER_SUCCESS(CheckLowerInput(lower_input));
  std::vector<bg::ValueHolderPtr> input_holders;
  std::vector<bg::ValueHolderPtr> input_addr_holders;
  LOWER_REQUIRE_SUCCESS(BuildHostInputTensors(node, lower_input, input_holders, input_addr_holders));
  // Allocate
  auto allocator_holder = lower_input.global_data->GetOrCreateAllocator({kOnHost, AllocatorUsage::kAllocNodeOutput});
  // Create op executeFunc
  auto custom_executor_func = FindHostCpuCustomExecutorFunc(node, lower_input);
  input_holders.emplace_back(allocator_holder);
  input_holders.emplace_back(custom_executor_func);
  // Check inference_rule
  const auto op_desc = node->GetOpDesc();
  LOWER_REQUIRE_NOTNULL(op_desc);
  std::string kernel_type = "ExecuteHostCustomOp";
  if (NeedCustomOpInferShape(node, *lower_input.global_data)) {
    kernel_type = "ExecuteHostCustomOpWithInferShape";
    std::vector<bg::ValueHolderPtr> infer_output_shapes =
        bg::InferCustomOpShape(node, lower_input.input_shapes, *lower_input.global_data);
    input_holders.insert(input_holders.end(), infer_output_shapes.begin(), infer_output_shapes.end());
  }
  std::vector<bg::ValueHolderPtr> output_tensor_holders =
      bg::ValueHolder::CreateDataOutput(kernel_type.c_str(), input_holders, node->GetAllOutDataAnchorsSize());
  std::vector<bg::ValueHolderPtr> output_shapes;
  std::vector<bg::DevMemValueHolderPtr> output_addrs;
  for (size_t i = 0UL; i < node->GetAllOutDataAnchorsSize(); i++) {
    auto split_outputs = bg::DevMemValueHolder::CreateDataOutput(
        kernel::kSplitDataTensor, {output_tensor_holders[i], allocator_holder},
        static_cast<size_t>(kernel::SplitTensorOutputs::kNum), op_desc->GetStreamId());
    CONVERTER_CHECK_HOLDERS_ALL_OK(split_outputs, static_cast<size_t>(kernel::SplitTensorOutputs::kNum));
    auto output_addr = split_outputs[static_cast<size_t>(kernel::SplitTensorOutputs::kTensorData)];
    output_addr->SetPlacement(kOnHost);
    LOWER_REQUIRE_NOTNULL(bg::ValueHolder::CreateVoidGuarder("FreeMemory", output_addr, {}));
    output_shapes.emplace_back(split_outputs[static_cast<size_t>(kernel::SplitTensorOutputs::kShape)]);
    output_addrs.emplace_back(output_addr);
  }

  for (auto &addr : input_addr_holders) {
    auto guarder = addr->GetGuarder();
    if ((guarder != nullptr) && (!output_tensor_holders.empty())) {
      GE_ASSERT_HYPER_SUCCESS(bg::ValueHolder::AddDependency(output_tensor_holders.front(), guarder));
    }
  }
  return {HyperStatus::Success(), {}, output_shapes, output_addrs};
}
REGISTER_NODE_CONVERTER_PLACEMENT(ge::kHostCpuCustomOpLowerFunc.c_str(), kOnHost, LoweringHostCustomNode);
}  // namespace gert
