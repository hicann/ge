/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "lowering/pass/split_mixed_launch_memory.h"

#include <algorithm>
#include <numeric>
#include <queue>
#include <string>
#include <unordered_set>
#include <gtest/gtest.h>
#include "common/dump/dump_manager.h"
#include "common/dump/dump_properties.h"
#include "common/bg_test.h"
#include "common/model_v2_executor_test_helper.h"
#include "common/summary_checker.h"
#include "common/topo_checker.h"
#include "exe_graph/runtime/tiling_context.h"
#include "exe_graph/lowering/exe_graph_attrs.h"
#include "exe_graph/lowering/value_holder.h"
#include "faker/exe_graph_model_level_data_faker.h"
#include "framework/runtime/executor_option/multi_thread_executor_option.h"
#include "graph/utils/execute_graph_utils.h"
#include "graph/utils/fast_node_utils.h"
#include "kernel/common_kernel_impl/memory_copy.h"
#include "kernel/common_kernel_impl/copy_flow_launch.h"
#include "kernel/memory/memory_kernel.h"
#include "core/builder/graph_node.h"
#include "core/builder/node_types.h"
#include "core/builder/graph_executor_builder.h"
#include "core/executor/multi_thread_topological/execution_data/multi_thread_execution_data_builder.h"
#include "core/executor/multi_thread_topological/execution_data/multi_thread_exe_graph_resource_guard.h"
#include "core/executor/multi_thread_topological/execution_data/free_launch_relation.h"
#include "kernel/tensor_attr.h"
#include "kernel/common_kernel_impl/tiling.h"
#include "lowering/node_priority_calculator.h"
#include "lowering/pass/copy_flow_launch_fuse.h"
#include "lowering/pass_changed_kernels_info.h"
#include "lowering/pass/remove_launch_free_edge.h"
#include "lowering/pass/offline_optimizer.h"
#include "subscriber/dumper/executor_dumper.h"
#include "subscriber/profiler/cann_profiler_v2.h"
#include "stub/gert_runtime_stub.h"
#include "faker/kernel_run_context_facker.h"
#include "mmpa/mmpa_api.h"

namespace gert {
bool IsEnableRt2MultiThread();

namespace bg {
namespace {
class ScopedEnvRestore {
 public:
  explicit ScopedEnvRestore(const mmEnvId env_id) : env_id_(env_id) {
    const char_t *value = nullptr;
    GetRaw(value);
    if (value != nullptr) {
      original_value_ = value;
      had_original_value_ = true;
    }
  }

  ~ScopedEnvRestore() {
    EXPECT_EQ(had_original_value_ ? Set(original_value_.c_str()) : Unset(), 0);
  }

  int32_t Set(const char *const value) const {
    int32_t mm_ret = 0;
    if (env_id_ == MM_ENV_MAX_RUNTIME_CORE_NUMBER) {
      MM_SYS_SET_ENV(MM_ENV_MAX_RUNTIME_CORE_NUMBER, value, 1, mm_ret);
    } else {
      MM_SYS_SET_ENV(MM_ENV_ENABLE_DYNAMIC_SHAPE_MULTI_STREAM, value, 1, mm_ret);
    }
    return mm_ret;
  }

  int32_t Unset() const {
    int32_t mm_ret = 0;
    if (env_id_ == MM_ENV_MAX_RUNTIME_CORE_NUMBER) {
      MM_SYS_UNSET_ENV(MM_ENV_MAX_RUNTIME_CORE_NUMBER, mm_ret);
    } else {
      MM_SYS_UNSET_ENV(MM_ENV_ENABLE_DYNAMIC_SHAPE_MULTI_STREAM, mm_ret);
    }
    return mm_ret;
  }

  bool Get(std::string &value) const {
    const char_t *current_value = nullptr;
    GetRaw(current_value);
    if (current_value == nullptr) {
      return false;
    }
    value = current_value;
    return true;
  }

  ScopedEnvRestore(const ScopedEnvRestore &) = delete;
  ScopedEnvRestore &operator=(const ScopedEnvRestore &) = delete;

 private:
  void GetRaw(const char_t *&value) const {
    if (env_id_ == MM_ENV_MAX_RUNTIME_CORE_NUMBER) {
      MM_SYS_GET_ENV(MM_ENV_MAX_RUNTIME_CORE_NUMBER, value);
    } else {
      MM_SYS_GET_ENV(MM_ENV_ENABLE_DYNAMIC_SHAPE_MULTI_STREAM, value);
    }
  }

  mmEnvId env_id_;
  std::string original_value_;
  bool had_original_value_ = false;
};

class SplitMixedLaunchMemoryUT : public BgTestAutoCreateFrame {
 protected:
  void SetRt2MultiThreadEnabled() {
    EXPECT_EQ(dynamic_multi_stream_env_.Unset(), 0);
    EXPECT_EQ(max_runtime_core_env_.Set("3"), 0);
  }

  void SetRt2SingleThreadEnabled() {
    EXPECT_EQ(dynamic_multi_stream_env_.Unset(), 0);
    EXPECT_EQ(max_runtime_core_env_.Set("1"), 0);
  }

  void SetDynamicShapeMultiStreamEnabled() {
    EXPECT_EQ(dynamic_multi_stream_env_.Set("1"), 0);
  }

  ScopedEnvRestore max_runtime_core_env_{MM_ENV_MAX_RUNTIME_CORE_NUMBER};
  ScopedEnvRestore dynamic_multi_stream_env_{MM_ENV_ENABLE_DYNAMIC_SHAPE_MULTI_STREAM};
};

std::vector<ValueHolderPtr> CreateTiling() {
  std::vector<ValueHolderPtr> inputs = {
      ValueHolder::CreateFeed(0),
      ValueHolder::CreateFeed(0),
      ValueHolder::CreateFeed(0),
      ValueHolder::CreateSingleDataOutput("InnerData", {}),
      ValueHolder::CreateSingleDataOutput("InnerData", {}),
      ValueHolder::CreateSingleDataOutput("InnerData", {}),
  };
  return ValueHolder::CreateDataOutput("Tiling", inputs, static_cast<size_t>(kernel::TilingExOutputIndex::kNum));
}

std::vector<ValueHolderPtr> CreateLaunchKernelWithFlagCommonInputs() {
  auto tiling_out = CreateTiling();
  std::vector<ValueHolderPtr> inputs = {
      ValueHolder::CreateFeed(0),
      ValueHolder::CreateSingleDataOutput("InnerData", {}),
      tiling_out[TilingContext::kOutputBlockDim],
      ValueHolder::CreateSingleDataOutput("InnerData", {}),
      ValueHolder::CreateSingleDataOutput("InnerData", {}),
      ValueHolder::CreateSingleDataOutput("InnerData", {}),
      ValueHolder::CreateSingleDataOutput("InnerData", {}),
      tiling_out[TilingContext::kOutputScheduleMode],
      ValueHolder::CreateSingleDataOutput("InnerData", {}),
      tiling_out[static_cast<size_t>(kernel::TilingExOutputIndex::kRtArg)],
      ValueHolder::CreateFeed(0),
      ValueHolder::CreateSingleDataOutput("InnerData", {}),
  };
  return inputs;
}

std::vector<ValueHolderPtr> CreateLaunchKernelWithFlagCommonFeedInputs() {
  std::vector<ValueHolderPtr> tiling_inputs(6U);
  std::generate(tiling_inputs.begin(), tiling_inputs.end(), []() { return ValueHolder::CreateFeed(0); });
  auto tiling_out =
      ValueHolder::CreateDataOutput("Tiling", tiling_inputs, static_cast<size_t>(kernel::TilingExOutputIndex::kNum));
  std::vector<ValueHolderPtr> inputs = {
      ValueHolder::CreateFeed(0),
      ValueHolder::CreateFeed(0),
      tiling_out[TilingContext::kOutputBlockDim],
      ValueHolder::CreateFeed(0),
      ValueHolder::CreateFeed(0),
      ValueHolder::CreateFeed(0),
      ValueHolder::CreateFeed(0),
      tiling_out[TilingContext::kOutputScheduleMode],
      ValueHolder::CreateFeed(0),
      tiling_out[static_cast<size_t>(kernel::TilingExOutputIndex::kRtArg)],
      ValueHolder::CreateFeed(0),
      ValueHolder::CreateFeed(0),
  };
  return inputs;
}

ValueHolderPtr CreateCopyH2D(const bool use_feed_inputs = false) {
  auto data = ValueHolder::CreateFeed(0);
  auto dt_holder = ValueHolder::CreateConst("Hello", 5, true);
  auto tensor_size = ValueHolder::CreateSingleDataOutput("CalcTensorSizeFromStorage", {dt_holder, data});
  auto split_outputs = ValueHolder::CreateDataOutput("SplitTensor", {}, 2);
  std::vector<ValueHolderPtr> copy_inputs = {
      ValueHolder::CreateFeed(0),
      use_feed_inputs ? ValueHolder::CreateFeed(0) : ValueHolder::CreateSingleDataOutput("InnerData", {}),
      split_outputs[1],
      tensor_size,
      split_outputs[0],
      use_feed_inputs ? ValueHolder::CreateFeed(0) : ValueHolder::CreateSingleDataOutput("InnerData", {})};
  auto copy_h2d = ValueHolder::CreateSingleDataOutput("CopyH2D", copy_inputs);
  ValueHolder::CreateVoidGuarder("FreeMemory", copy_h2d, {});
  return copy_h2d;
}

ge::ExecuteGraphPtr BuildCopyToLaunchGraph(const size_t copy_count = 1U, const bool use_feed_inputs = false) {
  auto launch_inputs =
      use_feed_inputs ? CreateLaunchKernelWithFlagCommonFeedInputs() : CreateLaunchKernelWithFlagCommonInputs();
  for (size_t i = 0U; i < copy_count; ++i) {
    launch_inputs.emplace_back(CreateCopyH2D(use_feed_inputs));
  }
  launch_inputs.emplace_back(
      ValueHolder::CreateSingleDataOutput("AllocBatchHbm", {ValueHolder::CreateFeed(0), ValueHolder::CreateFeed(0)}));
  auto launch = ValueHolder::CreateSingleDataOutput("LaunchKernelWithFlag", launch_inputs);
  auto frame = ValueHolder::PopGraphFrame({launch}, {});
  EXPECT_NE(frame, nullptr);
  return frame->GetExecuteGraph();
}

ge::ExecuteGraphPtr BuildTwoOutputCopyToTwoLaunchesGraph(
    const bool use_feed_inputs = false, std::vector<std::string> *const consumer_launch_names = nullptr) {
  auto stream = ValueHolder::CreateFeed(0);
  auto allocator = use_feed_inputs ? ValueHolder::CreateFeed(0) : ValueHolder::CreateSingleDataOutput("InnerData", {});
  std::vector<ValueHolderPtr> copy_inputs = {stream, allocator};
  for (size_t i = 0U; i < 2U; ++i) {
    auto data = ValueHolder::CreateFeed(0);
    auto dt_holder = ValueHolder::CreateConst("Hello", 5, true);
    auto tensor_size = ValueHolder::CreateSingleDataOutput("CalcTensorSizeFromStorage", {dt_holder, data});
    auto split_outputs = ValueHolder::CreateDataOutput("SplitTensor", {}, 2);
    auto data_type =
        use_feed_inputs ? ValueHolder::CreateFeed(0) : ValueHolder::CreateSingleDataOutput("InnerData", {});
    copy_inputs.insert(copy_inputs.end(), {split_outputs[1], tensor_size, split_outputs[0], data_type});
  }
  auto copy_outputs = ValueHolder::CreateDataOutput("CopyH2D", copy_inputs, 2U);
  ValueHolder::CreateVoidGuarder("FreeMemory", copy_outputs[0], {});
  ValueHolder::CreateVoidGuarder("FreeMemory", copy_outputs[1], {});

  std::vector<ValueHolderPtr> launches;
  for (size_t i = 0U; i < copy_outputs.size(); ++i) {
    auto launch_inputs =
        use_feed_inputs ? CreateLaunchKernelWithFlagCommonFeedInputs() : CreateLaunchKernelWithFlagCommonInputs();
    launch_inputs.emplace_back(copy_outputs[i]);
    launch_inputs.emplace_back(
        ValueHolder::CreateSingleDataOutput("AllocBatchHbm", {ValueHolder::CreateFeed(0), ValueHolder::CreateFeed(0)}));
    auto launch = ValueHolder::CreateSingleDataOutput("LaunchKernelWithFlag", launch_inputs);
    if (consumer_launch_names != nullptr) {
      consumer_launch_names->emplace_back(launch->GetFastNode()->GetName());
    }
    launches.emplace_back(std::move(launch));
  }
  auto frame = ValueHolder::PopGraphFrame(launches, {});
  EXPECT_NE(frame, nullptr);
  return frame->GetExecuteGraph();
}

size_t GetGuarderCount(const ge::FastNode *const node, const size_t output_index) {
  const auto &out_data_edges = node->GetAllOutDataEdgesRef();
  if (output_index >= out_data_edges.size()) {
    return 0U;
  }
  return static_cast<size_t>(std::count_if(out_data_edges[output_index].cbegin(), out_data_edges[output_index].cend(),
                                           [](const ge::Edge<ge::FastNode> *const edge) {
                                             return (edge != nullptr) && (edge->dst != nullptr) &&
                                                    IsFreeNode(edge->dst->GetTypePtr());
                                           }));
}

bool HasControlEdge(const ge::FastNode *const src, const ge::FastNode *const dst) {
  return std::any_of(dst->GetAllInControlEdgesRef().cbegin(), dst->GetAllInControlEdgesRef().cend(),
                     [src](const ge::FastEdge *const edge) { return (edge != nullptr) && (edge->src == src); });
}

ge::graphStatus AssignUniqueFeedIndexes(const ge::ExecuteGraphPtr &graph) {
  int64_t feed_index = 0;
  for (const auto node : ge::ExecuteGraphUtils::FindNodesByTypeFromAllNodes(graph.get(), "Data")) {
    GE_ASSERT_TRUE(ge::AttrUtils::SetInt(node->GetOpDescBarePtr(), "index", feed_index));
    ++feed_index;
  }
  return ge::GRAPH_SUCCESS;
}

UINT32 BuildOnlyKernel(KernelContext *) {
  return ge::GRAPH_SUCCESS;
}

bool IsAcyclicAfterStrictOrder(const ge::ExecuteGraph *const graph, const GraphNode &graph_node) {
  std::unordered_map<const ge::FastNode *, int64_t> indegrees;
  for (const auto node : graph->GetDirectNode()) {
    if (!IsNodeNeedExec(node->GetTypePtr())) {
      continue;
    }
    const auto count_in_edge = [](const ge::FastEdge *const edge) {
      return (edge != nullptr) && IsNodeNeedExec(edge->src->GetTypePtr());
    };
    int64_t indegree = static_cast<int64_t>(
        std::count_if(node->GetAllInDataEdgesRef().cbegin(), node->GetAllInDataEdgesRef().cend(), count_in_edge));
    indegree += static_cast<int64_t>(
        std::count_if(node->GetAllInControlEdgesRef().cbegin(), node->GetAllInControlEdgesRef().cend(), count_in_edge));
    const auto additional_indegree = graph_node.additional_indegree_info.find(node);
    if (additional_indegree != graph_node.additional_indegree_info.cend()) {
      indegree += additional_indegree->second;
    }
    indegrees.emplace(node, indegree);
  }

  std::queue<const ge::FastNode *> ready_nodes;
  for (const auto &node_and_indegree : indegrees) {
    if (node_and_indegree.second == 0) {
      ready_nodes.emplace(node_and_indegree.first);
    }
  }
  size_t visited_count = 0U;
  while (!ready_nodes.empty()) {
    const auto node = ready_nodes.front();
    ready_nodes.pop();
    ++visited_count;
    const auto decrease_indegree = [&indegrees, &ready_nodes](const ge::FastNode *const dst) {
      const auto iter = indegrees.find(dst);
      if ((iter != indegrees.end()) && (--iter->second == 0)) {
        ready_nodes.emplace(dst);
      }
    };
    for (const auto &out_edges : node->GetAllOutDataEdgesRef()) {
      for (const auto edge : out_edges) {
        if (edge != nullptr) {
          decrease_indegree(edge->dst);
        }
      }
    }
    for (const auto edge : node->GetAllOutControlEdgesRef()) {
      if (edge != nullptr) {
        decrease_indegree(edge->dst);
      }
    }
    const auto additional_outputs = graph_node.additional_add_info.find(node);
    if (additional_outputs != graph_node.additional_add_info.cend()) {
      for (const auto dst : additional_outputs->second) {
        decrease_indegree(dst);
      }
    }
  }
  return visited_count == indegrees.size();
}

ge::ExecuteGraphPtr BuildTwoOutputCopyToSingleLaunchGraph(
    const bool repeat_first_output = false, const bool use_feed_inputs = false,
    std::vector<ValueHolderPtr> *const original_copy_outputs = nullptr) {
  auto stream = ValueHolder::CreateFeed(0);
  auto allocator = use_feed_inputs ? ValueHolder::CreateFeed(0) : ValueHolder::CreateSingleDataOutput("InnerData", {});
  std::vector<ValueHolderPtr> copy_inputs = {stream, allocator};
  for (size_t i = 0U; i < 2U; ++i) {
    auto data = ValueHolder::CreateFeed(0);
    auto dt_holder = ValueHolder::CreateConst("Hello", 5, true);
    auto tensor_size = ValueHolder::CreateSingleDataOutput("CalcTensorSizeFromStorage", {dt_holder, data});
    auto split_outputs = ValueHolder::CreateDataOutput("SplitTensor", {}, 2);
    auto data_type =
        use_feed_inputs ? ValueHolder::CreateFeed(0) : ValueHolder::CreateSingleDataOutput("InnerData", {});
    copy_inputs.insert(copy_inputs.end(), {split_outputs[1], tensor_size, split_outputs[0], data_type});
  }
  auto copy_outputs = ValueHolder::CreateDataOutput("CopyH2D", copy_inputs, 2U);
  if (original_copy_outputs != nullptr) {
    *original_copy_outputs = copy_outputs;
  }
  ValueHolder::CreateVoidGuarder("FreeMemory", copy_outputs[0], {});
  ValueHolder::CreateVoidGuarder("FreeMemory", copy_outputs[1], {});

  auto launch_inputs =
      use_feed_inputs ? CreateLaunchKernelWithFlagCommonFeedInputs() : CreateLaunchKernelWithFlagCommonInputs();
  launch_inputs.insert(launch_inputs.end(), copy_outputs.cbegin(), copy_outputs.cend());
  if (repeat_first_output) {
    launch_inputs.emplace_back(copy_outputs[0U]);
  }
  launch_inputs.emplace_back(
      ValueHolder::CreateSingleDataOutput("AllocBatchHbm", {ValueHolder::CreateFeed(0), ValueHolder::CreateFeed(0)}));
  auto launch = ValueHolder::CreateSingleDataOutput("LaunchKernelWithFlag", launch_inputs);
  auto frame = ValueHolder::PopGraphFrame({launch}, {});
  EXPECT_NE(frame, nullptr);
  return frame->GetExecuteGraph();
}

ge::ExecuteGraphPtr BuildCopyToConsumerGraph() {
  auto copy_h2d = CreateCopyH2D();
  auto consumer = ValueHolder::CreateSingleDataOutput("Consumer", {copy_h2d});
  auto frame = ValueHolder::PopGraphFrame({consumer}, {});
  EXPECT_NE(frame, nullptr);
  return frame->GetExecuteGraph();
}

ge::ExecuteGraphPtr BuildLegacyCopyFlowToConsumerGraph() {
  std::vector<ValueHolderPtr> inputs = {
      ValueHolder::CreateSingleDataOutput("InputNum", {}),   ValueHolder::CreateSingleDataOutput("InputIndex", {}),
      ValueHolder::CreateSingleDataOutput("RtArg", {}),      ValueHolder::CreateSingleDataOutput("Stream", {}),
      ValueHolder::CreateSingleDataOutput("Allocator", {}),  ValueHolder::CreateSingleDataOutput("TensorData", {}),
      ValueHolder::CreateSingleDataOutput("TensorSize", {}), ValueHolder::CreateSingleDataOutput("StorageShape", {}),
      ValueHolder::CreateSingleDataOutput("DataType", {})};
  auto copy_flow = ValueHolder::CreateSingleDataOutput(kernel::kCopyFlowLaunch, inputs);
  auto ctrl_before = ValueHolder::CreateSingleDataOutput("CtrlBeforeCopyFlow", {});
  EXPECT_TRUE(ValueHolder::AddDependency(ctrl_before, copy_flow).IsSuccess());
  ValueHolder::CreateVoidGuarder("FreeMemory", copy_flow, {});
  auto consumer = ValueHolder::CreateSingleDataOutput("Consumer", {copy_flow});
  auto ctrl_after = ValueHolder::CreateSingleDataOutput("CtrlAfterCopyFlow", {});
  EXPECT_TRUE(ValueHolder::AddDependency(copy_flow, ctrl_after).IsSuccess());
  auto frame = ValueHolder::PopGraphFrame({consumer, ctrl_after}, {});
  EXPECT_NE(frame, nullptr);
  return frame->GetExecuteGraph();
}

ge::ExecuteGraphPtr BuildCopyToDirectLaunchConsumerGraph(const char *const consumer_type = "ExecuteOpLaunch") {
  auto copy_h2d = CreateCopyH2D();
  auto launch = ValueHolder::CreateVoid<bg::ValueHolder>(consumer_type, {copy_h2d});
  auto frame = ValueHolder::PopGraphFrame({}, {launch});
  EXPECT_NE(frame, nullptr);
  return frame->GetExecuteGraph();
}

ge::ExecuteGraphPtr BuildCopyToCustomConsumerGraph() {
  auto copy_h2d = CreateCopyH2D();
  auto build_ref_tensor = ValueHolder::CreateSingleDataOutput("BuildRefTensor", {copy_h2d});
  auto custom = ValueHolder::CreateSingleDataOutput("ExecuteCustomOp", {build_ref_tensor});
  auto frame = ValueHolder::PopGraphFrame({custom}, {});
  EXPECT_NE(frame, nullptr);
  return frame->GetExecuteGraph();
}

ge::ExecuteGraphPtr BuildCopyToDirectLaunchWithFreeControlGraph() {
  auto copy_h2d = CreateCopyH2D(true);
  auto launch = ValueHolder::CreateVoid<bg::ValueHolder>("ExecuteOpLaunch", {copy_h2d});
  EXPECT_TRUE(ValueHolder::AddDependency(launch, copy_h2d->GetGuarder()).IsSuccess());
  auto output_data = ValueHolder::CreateSingleDataOutput("OutputData", {copy_h2d});
  auto frame = ValueHolder::PopGraphFrame({output_data}, {launch});
  EXPECT_NE(frame, nullptr);
  return frame == nullptr ? nullptr : frame->GetExecuteGraph();
}

ge::ExecuteGraphPtr BuildAclnnPrepareCopyGraph(const bool add_copy_flow = false) {
  auto stream = ValueHolder::CreateFeed(0);
  auto allocator = ValueHolder::CreateFeed(1);
  auto src_addr = ValueHolder::CreateFeed(2);
  auto tensor_size = ValueHolder::CreateFeed(3);
  auto storage_shape = ValueHolder::CreateFeed(4);
  auto data_type = ValueHolder::CreateFeed(5);
  auto copy = ValueHolder::CreateSingleDataOutput("MakeSureTensorAtDevice",
                                                  {stream, allocator, src_addr, tensor_size, storage_shape, data_type});
  ValueHolder::CreateVoidGuarder("FreeMemory", copy, {});

  kernel::BuildTensorAttr tensor_attr{};
  tensor_attr.placement = kOnDeviceHbm;
  tensor_attr.data_type = ge::DT_FLOAT;
  auto tensor_attr_holder = ValueHolder::CreateConst(&tensor_attr, sizeof(tensor_attr));
  auto build_ref_tensor =
      ValueHolder::CreateSingleDataOutput("BuildRefTensor", {storage_shape, copy, tensor_attr_holder});

  auto execute_option = ValueHolder::CreateFeed(6);
  auto fwk_data = ValueHolder::CreateFeed(7);
  auto execute_op_prepare =
      ValueHolder::CreateDataOutput("ExecuteOpPrepare", {build_ref_tensor, execute_option, fwk_data, stream}, 2U);
  auto workspace = ValueHolder::CreateSingleDataOutput("AllocBatchHbm", {allocator, execute_op_prepare[1U]});
  auto execute_op_launch = ValueHolder::CreateVoid<bg::ValueHolder>(
      "ExecuteOpLaunch",
      {build_ref_tensor, execute_op_prepare[0U], workspace, execute_op_prepare[1U], stream, fwk_data});

  if (add_copy_flow) {
    std::vector<ValueHolderPtr> copy_flow_inputs = {
        ValueHolder::CreateSingleDataOutput("InputNum", {}),   ValueHolder::CreateSingleDataOutput("InputIndex", {}),
        ValueHolder::CreateSingleDataOutput("RtArg", {}),      ValueHolder::CreateSingleDataOutput("Stream", {}),
        ValueHolder::CreateSingleDataOutput("Allocator", {}),  ValueHolder::CreateSingleDataOutput("TensorData", {}),
        ValueHolder::CreateSingleDataOutput("TensorSize", {}), ValueHolder::CreateSingleDataOutput("StorageShape", {}),
        ValueHolder::CreateSingleDataOutput("DataType", {})};
    auto copy_flow = ValueHolder::CreateSingleDataOutput(kernel::kCopyFlowLaunch, copy_flow_inputs);
    auto copy_flow_free = ValueHolder::CreateVoidGuarder("FreeMemory", copy_flow, {});
    auto copy_flow_launch = ValueHolder::CreateVoid<bg::ValueHolder>("ExecuteOpLaunch", {copy_flow});
    EXPECT_TRUE(ValueHolder::AddDependency(copy_flow_launch, copy_flow_free).IsSuccess());
  }

  auto frame = ValueHolder::PopGraphFrame({execute_op_prepare[0U]}, {});
  EXPECT_NE(frame, nullptr);
  return frame == nullptr ? nullptr : frame->GetExecuteGraph();
}

ge::ExecuteGraphPtr BuildSingleStageAclnnCopyGraph() {
  auto stream = ValueHolder::CreateFeed(0);
  auto allocator = ValueHolder::CreateFeed(1);
  auto src_addr = ValueHolder::CreateFeed(2);
  auto tensor_size = ValueHolder::CreateFeed(3);
  auto storage_shape = ValueHolder::CreateFeed(4);
  auto data_type = ValueHolder::CreateFeed(5);
  auto copy = ValueHolder::CreateSingleDataOutput("MakeSureTensorAtDevice",
                                                  {stream, allocator, src_addr, tensor_size, storage_shape, data_type});
  ValueHolder::CreateVoidGuarder("FreeMemory", copy, {});

  kernel::BuildTensorAttr tensor_attr{};
  tensor_attr.placement = kOnDeviceHbm;
  tensor_attr.data_type = ge::DT_FLOAT;
  auto tensor_attr_holder = ValueHolder::CreateConst(&tensor_attr, sizeof(tensor_attr));
  auto build_ref_tensor =
      ValueHolder::CreateSingleDataOutput("BuildRefTensor", {storage_shape, copy, tensor_attr_holder});
  auto execute_op = ValueHolder::CreateSingleDataOutput("ExecuteOpFunc", {build_ref_tensor});
  EXPECT_TRUE(ValueHolder::AddDependency(execute_op, copy->GetGuarder()).IsSuccess());

  auto frame = ValueHolder::PopGraphFrame({execute_op}, {});
  EXPECT_NE(frame, nullptr);
  return frame == nullptr ? nullptr : frame->GetExecuteGraph();
}

ge::ExecuteGraphPtr BuildRootGraphWithAclnnPrepareCopySubgraph() {
  auto sub_graph = BuildAclnnPrepareCopyGraph();
  GE_ASSERT_NOTNULL(sub_graph);

  ValueHolder::PushGraphFrame();
  auto call_node = ValueHolder::CreateVoid<bg::ValueHolder>("PartitionedCall", {});
  auto root_frame = ValueHolder::PopGraphFrame({}, {call_node});
  GE_ASSERT_NOTNULL(root_frame);
  auto root_graph = root_frame->GetExecuteGraph();
  GE_ASSERT_NOTNULL(root_graph);

  root_graph->AddSubGraph(sub_graph);
  sub_graph->SetParentGraph(root_graph.get());
  sub_graph->SetParentNode(call_node->GetFastNode());
  call_node->GetFastNode()->GetOpDescPtr()->AddSubgraphName("body");
  call_node->GetFastNode()->GetOpDescPtr()->SetSubgraphInstanceName(0U, sub_graph->GetName());
  return root_graph;
}

ge::ExecuteGraphPtr BuildMainGraphWithAclnnPrepareCopySubgraph() {
  auto call_node = ValueHolder::CreateSingleDataOutput("PartitionedCall", {});
  GE_ASSERT_NOTNULL(call_node);
  GE_ASSERT_NOTNULL(ValueHolder::PushGraphFrame(call_node, "NestedOwner"));
  auto sub_graph = BuildAclnnPrepareCopyGraph();
  GE_ASSERT_NOTNULL(sub_graph);

  auto main_frame = ValueHolder::PopGraphFrame({}, {call_node});
  GE_ASSERT_NOTNULL(main_frame);
  auto main_graph = main_frame->GetExecuteGraph();
  GE_ASSERT_NOTNULL(main_graph);
  return main_graph;
}

ge::ExecuteGraphPtr BuildZeroCopyAndH2DCopyGraph() {
  auto compute_graph = std::make_shared<ge::ComputeGraph>("zero_copy_h2d_compute_graph");
  auto h2d_compute_desc = ge::MakeShared<ge::OpDesc>("h2d_compute", "Compute");
  auto zero_copy_compute_desc = ge::MakeShared<ge::OpDesc>("zero_copy_compute", "Compute");
  GE_ASSERT_NOTNULL(h2d_compute_desc);
  GE_ASSERT_NOTNULL(zero_copy_compute_desc);
  const auto h2d_compute = compute_graph->AddNode(h2d_compute_desc);
  const auto zero_copy_compute = compute_graph->AddNode(zero_copy_compute_desc);
  GE_ASSERT_NOTNULL(h2d_compute);
  GE_ASSERT_NOTNULL(zero_copy_compute);

  ValueHolder::SetCurrentComputeNode(h2d_compute);
  auto h2d_copy = CreateCopyH2D(true);
  auto h2d_consumer = ValueHolder::CreateVoid<bg::ValueHolder>("ExecuteOpLaunch", {h2d_copy});
  EXPECT_TRUE(ValueHolder::AddDependency(h2d_consumer, h2d_copy->GetGuarder()).IsSuccess());

  ValueHolder::SetCurrentComputeNode(zero_copy_compute);
  auto allocator = ValueHolder::CreateFeed(20);
  auto size = ValueHolder::CreateFeed(21);
  auto shape = ValueHolder::CreateFeed(22);
  auto stream = ValueHolder::CreateFeed(-1);
  auto output_data = ValueHolder::CreateSingleDataOutput("OutputData", {});
  auto tensor_attrs = ValueHolder::CreateConst("Hello", 5);
  auto alloc = ValueHolder::CreateSingleDataOutput("AllocMemory", {allocator, size});
  auto free = ValueHolder::CreateVoidGuarder("FreeMemory", alloc, {});
  auto copy_d2d = ValueHolder::CreateSingleDataOutput("CopyD2D", {alloc});
  auto zero_copy_consumer = ValueHolder::CreateVoid<bg::ValueHolder>("ExecuteOpLaunch", {copy_d2d, stream});
  EXPECT_TRUE(ValueHolder::AddDependency(zero_copy_consumer, free).IsSuccess());
  auto ensure = ValueHolder::CreateVoid<bg::ValueHolder>("EnsureTensorAtOutMemory",
                                                         {shape, alloc, tensor_attrs, stream, output_data});
  auto frame = ValueHolder::PopGraphFrame({output_data}, {h2d_consumer, zero_copy_consumer, copy_d2d, ensure});
  EXPECT_NE(frame, nullptr);
  return frame == nullptr ? nullptr : frame->GetExecuteGraph();
}

}  // namespace

TEST_F(SplitMixedLaunchMemoryUT, IsEnableRt2MultiThreadFollowsMaxRuntimeCoreNumber) {
  SetRt2SingleThreadEnabled();
  EXPECT_FALSE(IsEnableRt2MultiThread());

  SetRt2MultiThreadEnabled();
  EXPECT_TRUE(IsEnableRt2MultiThread());
}

TEST_F(SplitMixedLaunchMemoryUT, IsEnableRt2MultiThreadDisabledByDynamicShapeMultiStream) {
  SetRt2MultiThreadEnabled();
  SetDynamicShapeMultiStreamEnabled();
  EXPECT_FALSE(IsEnableRt2MultiThread());
}

TEST_F(SplitMixedLaunchMemoryUT, ScopedEnvRestorePreservesInheritedAndMissingValues) {
  ASSERT_EQ(max_runtime_core_env_.Set("inherited_core_count"), 0);
  {
    ScopedEnvRestore restore(MM_ENV_MAX_RUNTIME_CORE_NUMBER);
    ASSERT_EQ(restore.Set("3"), 0);
  }
  std::string value;
  ASSERT_TRUE(max_runtime_core_env_.Get(value));
  EXPECT_EQ(value, "inherited_core_count");

  ASSERT_EQ(dynamic_multi_stream_env_.Unset(), 0);
  {
    ScopedEnvRestore restore(MM_ENV_ENABLE_DYNAMIC_SHAPE_MULTI_STREAM);
    ASSERT_EQ(restore.Set("1"), 0);
  }
  EXPECT_FALSE(dynamic_multi_stream_env_.Get(value));
}

TEST_F(SplitMixedLaunchMemoryUT, SplitLegacyCopyFlowLaunchWhenPassRuns) {
  auto graph = BuildCopyToLaunchGraph();
  ASSERT_NE(graph, nullptr);

  EXPECT_EQ(ge::ExecuteGraphUtils::FindNodesByTypeFromAllNodes(graph.get(), "CopyH2D").size(), 1UL);
  EXPECT_EQ(ge::ExecuteGraphUtils::FindNodesByTypeFromAllNodes(graph.get(), "CopyFlowLaunch").size(), 0UL);
  EXPECT_EQ(ge::ExecuteGraphUtils::FindNodesByTypeFromAllNodes(graph.get(), "PrepareCopyFlowResult").size(), 0UL);
  EXPECT_EQ(ge::ExecuteGraphUtils::FindNodesByTypeFromAllNodes(graph.get(), "LaunchCopyFlowH2D").size(), 0UL);

  bool changed = false;
  ASSERT_EQ(CopyFlowLaunchFuse().Run(graph.get(), changed), ge::GRAPH_SUCCESS);
  EXPECT_TRUE(changed);

  EXPECT_EQ(ge::ExecuteGraphUtils::FindNodesByTypeFromAllNodes(graph.get(), "CopyH2D").size(), 0UL);
  auto legacy_copy_flow_nodes =
      ge::ExecuteGraphUtils::FindNodesByTypeFromAllNodes(graph.get(), kernel::kCopyFlowLaunch);
  ASSERT_EQ(legacy_copy_flow_nodes.size(), 1UL);
  constexpr int64_t kComputeIndex = 17;
  ASSERT_TRUE(ge::AttrUtils::SetInt(legacy_copy_flow_nodes[0]->GetOpDescBarePtr(), kComputeNodeIndex, kComputeIndex));
  EXPECT_EQ(ge::ExecuteGraphUtils::FindNodesByTypeFromAllNodes(graph.get(), "PrepareCopyFlowResult").size(), 0UL);
  EXPECT_EQ(ge::ExecuteGraphUtils::FindNodesByTypeFromAllNodes(graph.get(), "LaunchCopyFlowH2D").size(), 0UL);

  changed = false;
  ASSERT_EQ(SplitMixedLaunchMemory().Run(graph.get(), changed), ge::GRAPH_SUCCESS);
  EXPECT_TRUE(changed);

  EXPECT_EQ(ge::ExecuteGraphUtils::FindNodesByTypeFromAllNodes(graph.get(), "CopyH2D").size(), 0UL);
  EXPECT_EQ(ge::ExecuteGraphUtils::FindNodesByTypeFromAllNodes(graph.get(), "CopyFlowLaunch").size(), 0UL);
  auto calc_nodes = ge::ExecuteGraphUtils::FindNodesByTypeFromAllNodes(graph.get(), "CalcCopyFlowAllocSizes");
  ASSERT_EQ(calc_nodes.size(), 1UL);
  auto alloc_nodes = ge::ExecuteGraphUtils::FindNodesByTypeFromAllNodes(graph.get(), kernel::kAllocCopyFlowHbm);
  ASSERT_EQ(alloc_nodes.size(), 1UL);
  EXPECT_TRUE(IsAllocNode(alloc_nodes[0]->GetTypePtr()));
  EXPECT_EQ(ge::ExecuteGraphUtils::FindNodesByTypeFromAllNodes(graph.get(), "AllocBatchHbm").size(), 1UL);
  int64_t calc_count = 0;
  int64_t alloc_count = 0;
  ASSERT_TRUE(ge::AttrUtils::GetInt(calc_nodes[0]->GetOpDescBarePtr(), kernel::kCopyFlowCountAttr, calc_count));
  ASSERT_TRUE(ge::AttrUtils::GetInt(alloc_nodes[0]->GetOpDescBarePtr(), kernel::kCopyFlowCountAttr, alloc_count));
  EXPECT_EQ(calc_count, 1);
  EXPECT_EQ(alloc_count, calc_count);
  auto prepare_nodes = ge::ExecuteGraphUtils::FindNodesByTypeFromAllNodes(graph.get(), "PrepareCopyFlowResult");
  auto launch_copy_nodes = ge::ExecuteGraphUtils::FindNodesByTypeFromAllNodes(graph.get(), "LaunchCopyFlowH2D");
  ASSERT_EQ(prepare_nodes.size(), 1UL);
  ASSERT_EQ(launch_copy_nodes.size(), 1UL);
  int64_t prepare_count = 0;
  int64_t launch_count = 0;
  ASSERT_TRUE(ge::AttrUtils::GetInt(prepare_nodes[0]->GetOpDescBarePtr(), kernel::kCopyFlowCountAttr, prepare_count));
  ASSERT_TRUE(
      ge::AttrUtils::GetInt(launch_copy_nodes[0]->GetOpDescBarePtr(), kernel::kCopyFlowCountAttr, launch_count));
  EXPECT_EQ(prepare_count, calc_count);
  EXPECT_EQ(launch_count, calc_count);
  int64_t prepare_compute_index = 0;
  ASSERT_TRUE(ge::AttrUtils::GetInt(prepare_nodes[0]->GetOpDescBarePtr(), kComputeNodeIndex, prepare_compute_index));
  EXPECT_EQ(prepare_compute_index, kComputeIndex);
  EXPECT_EQ(FastNodeTopoChecker(calc_nodes[0]).OutChecker().DataToByType(kernel::kAllocCopyFlowHbm).Result(),
            "success");
  EXPECT_EQ(FastNodeTopoChecker(alloc_nodes[0]).OutChecker().DataToByType("PrepareCopyFlowResult").Result(), "success");
  EXPECT_EQ(FastNodeTopoChecker(prepare_nodes[0]).OutChecker().CtrlToByType("LaunchCopyFlowH2D").Result(), "success");
  EXPECT_EQ(FastNodeTopoChecker(prepare_nodes[0]).OutChecker().DataToByType("LaunchKernelWithFlag").Result(),
            "success");
  EXPECT_EQ(FastNodeTopoChecker(launch_copy_nodes[0]).OutChecker().CtrlToByType("LaunchKernelWithFlag").Result(),
            "success");
  EXPECT_EQ(launch_copy_nodes[0]->GetDataOutNum(), 0U);
}

TEST_F(SplitMixedLaunchMemoryUT, OfflineOptimizerPropagatesComputeNodeIndexThroughLegacyCopyFlow) {
  SetRt2MultiThreadEnabled();
  std::unique_ptr<GraphFrame> init_frame;
  std::unique_ptr<GraphFrame> de_init_frame;
  Create3StageFrames(init_frame, de_init_frame);
  ASSERT_NE(init_frame, nullptr);
  auto init_netoutput_desc = ge::MakeShared<ge::OpDesc>("init_netoutput", "InnerNetOutput");
  ASSERT_NE(init_netoutput_desc, nullptr);
  ASSERT_NE(init_frame->GetExecuteGraph()->AddNode(init_netoutput_desc), nullptr);

  auto compute_graph = std::make_shared<ge::ComputeGraph>("copy_flow_priority_compute_graph");
  auto compute_op_desc = ge::MakeShared<ge::OpDesc>("copy_flow_priority_compute", "Compute");
  ASSERT_NE(compute_op_desc, nullptr);
  const auto compute_node = compute_graph->AddNode(compute_op_desc);
  ASSERT_NE(compute_node, nullptr);
  ValueHolder::SetCurrentComputeNode(compute_node);

  auto main_graph = BuildTwoOutputCopyToSingleLaunchGraph(true, true);
  ASSERT_NE(main_graph, nullptr);
  const auto consumer_nodes =
      ge::ExecuteGraphUtils::FindNodesByTypeFromAllNodes(main_graph.get(), "LaunchKernelWithFlag");
  ASSERT_EQ(consumer_nodes.size(), 1UL);
  int64_t source_compute_index = -1;
  ASSERT_TRUE(ge::AttrUtils::GetInt(consumer_nodes[0]->GetOpDescBarePtr(), kComputeNodeIndex, source_compute_index));
  EXPECT_EQ(source_compute_index, 0);

  auto output_data_desc = ge::MakeShared<ge::OpDesc>("output_data", "OutputData");
  ASSERT_NE(output_data_desc, nullptr);
  ASSERT_NE(main_graph->AddNode(output_data_desc), nullptr);
  auto root_frame = ValueHolder::PopGraphFrame();
  ASSERT_NE(root_frame, nullptr);
  auto root_graph = root_frame->GetExecuteGraph();
  ASSERT_NE(root_graph, nullptr);

  LoweringOption option;
  LoweringGlobalData global_data;
  ASSERT_EQ(OfflineOptimizer(option, global_data).Run(root_graph.get()), ge::GRAPH_SUCCESS);

  const auto prepare_nodes =
      ge::ExecuteGraphUtils::FindNodesByTypeFromAllNodes(root_graph.get(), kernel::kPrepareCopyFlowResult);
  const auto launch_copy_nodes =
      ge::ExecuteGraphUtils::FindNodesByTypeFromAllNodes(root_graph.get(), kernel::kLaunchCopyFlowH2D);
  ASSERT_EQ(prepare_nodes.size(), 1UL);
  ASSERT_EQ(launch_copy_nodes.size(), 1UL);
  int64_t prepare_compute_index = -1;
  int64_t launch_compute_index = -1;
  ASSERT_TRUE(ge::AttrUtils::GetInt(prepare_nodes[0]->GetOpDescBarePtr(), kComputeNodeIndex, prepare_compute_index));
  ASSERT_TRUE(ge::AttrUtils::GetInt(launch_copy_nodes[0]->GetOpDescBarePtr(), kComputeNodeIndex, launch_compute_index));
  EXPECT_EQ(prepare_compute_index, source_compute_index);
  EXPECT_EQ(launch_compute_index, source_compute_index);

  ASSERT_EQ(root_graph->TopologicalSorting(), ge::GRAPH_SUCCESS);
  const auto root_graph_nodes = root_graph->GetAllNodes();
  const auto main_graph_nodes = main_graph->GetAllNodes();
  ASSERT_EQ(NodePriorityCalculator(*root_frame).CalcNodeExecutionPriorities(main_graph_nodes, root_graph_nodes.size()),
            ge::GRAPH_SUCCESS);
  GraphNode graph_node;
  ASSERT_EQ(graph_node.EnsureNodeExeInOrder(root_graph.get()), ge::GRAPH_SUCCESS);
  EXPECT_TRUE(IsAcyclicAfterStrictOrder(main_graph.get(), graph_node));
}

TEST_F(SplitMixedLaunchMemoryUT, LegacyCopyFlowUsesPreparedDataAndLaunchTailControlsOnlyRequiredConsumers) {
  auto graph = BuildLegacyCopyFlowToConsumerGraph();
  ASSERT_NE(graph, nullptr);

  bool changed = false;
  ASSERT_EQ(SplitMixedLaunchMemory().Run(graph.get(), changed), ge::GRAPH_SUCCESS);
  ASSERT_TRUE(changed);

  auto prepare_nodes = ge::ExecuteGraphUtils::FindNodesByTypeFromAllNodes(graph.get(), "PrepareCopyFlowResult");
  auto launch_nodes = ge::ExecuteGraphUtils::FindNodesByTypeFromAllNodes(graph.get(), "LaunchCopyFlowH2D");
  auto calc_nodes = ge::ExecuteGraphUtils::FindNodesByTypeFromAllNodes(graph.get(), kernel::kCalcCopyFlowAllocSizes);
  auto free_nodes = ge::ExecuteGraphUtils::FindNodesByTypeFromAllNodes(graph.get(), "FreeMemory");
  ASSERT_EQ(prepare_nodes.size(), 1UL);
  ASSERT_EQ(launch_nodes.size(), 1UL);
  ASSERT_EQ(calc_nodes.size(), 1UL);
  ASSERT_EQ(free_nodes.size(), 1UL);
  EXPECT_EQ(FastNodeTopoChecker(calc_nodes[0]).InChecker().CtrlFromByType("CtrlBeforeCopyFlow").Result(), "success");
  EXPECT_NE(FastNodeTopoChecker(prepare_nodes[0]).InChecker().CtrlFromByType("CtrlBeforeCopyFlow").Result(), "success");
  EXPECT_EQ(FastNodeTopoChecker(prepare_nodes[0]).OutChecker().DataToByType("Consumer").Result(), "success");
  EXPECT_EQ(FastNodeTopoChecker(prepare_nodes[0]).OutChecker().DataToByType("FreeMemory").Result(), "success");
  EXPECT_EQ(FastNodeTopoChecker(prepare_nodes[0]).OutChecker().CtrlToByType("LaunchCopyFlowH2D").Result(), "success");
  EXPECT_EQ(FastNodeTopoChecker(launch_nodes[0]).OutChecker().CtrlToByType("CtrlAfterCopyFlow").Result(), "success");
  EXPECT_NE(FastNodeTopoChecker(launch_nodes[0]).OutChecker().CtrlToByType("Consumer").Result(), "success");
}

TEST_F(SplitMixedLaunchMemoryUT, CopyFlowLaunchFuseKeepsLegacyCopyFlowLaunchForSingleThread) {
  auto graph = BuildCopyToLaunchGraph();
  ASSERT_NE(graph, nullptr);

  EXPECT_EQ(ge::ExecuteGraphUtils::FindNodesByTypeFromAllNodes(graph.get(), "CopyH2D").size(), 1UL);
  EXPECT_EQ(ge::ExecuteGraphUtils::FindNodesByTypeFromAllNodes(graph.get(), "CopyFlowLaunch").size(), 0UL);
  EXPECT_EQ(ge::ExecuteGraphUtils::FindNodesByTypeFromAllNodes(graph.get(), "PrepareCopyFlowResult").size(), 0UL);
  EXPECT_EQ(ge::ExecuteGraphUtils::FindNodesByTypeFromAllNodes(graph.get(), "LaunchCopyFlowH2D").size(), 0UL);

  bool changed = false;
  ASSERT_EQ(CopyFlowLaunchFuse().Run(graph.get(), changed), ge::GRAPH_SUCCESS);
  EXPECT_TRUE(changed);

  EXPECT_EQ(ge::ExecuteGraphUtils::FindNodesByTypeFromAllNodes(graph.get(), "CopyH2D").size(), 0UL);
  EXPECT_EQ(ge::ExecuteGraphUtils::FindNodesByTypeFromAllNodes(graph.get(), "CopyFlowLaunch").size(), 1UL);
  EXPECT_EQ(ge::ExecuteGraphUtils::FindNodesByTypeFromAllNodes(graph.get(), "PrepareCopyFlowResult").size(), 0UL);
  EXPECT_EQ(ge::ExecuteGraphUtils::FindNodesByTypeFromAllNodes(graph.get(), "LaunchCopyFlowH2D").size(), 0UL);
}

TEST_F(SplitMixedLaunchMemoryUT, OfflineOptimizerKeepsCopyFlowLaunchForSingleThread) {
  SetRt2SingleThreadEnabled();
  std::unique_ptr<GraphFrame> init_frame;
  std::unique_ptr<GraphFrame> de_init_frame;
  Create3StageFrames(init_frame, de_init_frame);
  ASSERT_NE(init_frame, nullptr);
  auto init_netoutput_desc = ge::MakeShared<ge::OpDesc>("init_netoutput", "InnerNetOutput");
  ASSERT_NE(init_netoutput_desc, nullptr);
  ASSERT_NE(init_frame->GetExecuteGraph()->AddNode(init_netoutput_desc), nullptr);
  auto copy_flow = ValueHolder::CreateSingleDataOutput(kernel::kCopyFlowLaunch, {ValueHolder::CreateFeed(0)});
  auto main_frame = ValueHolder::PopGraphFrame({copy_flow}, {});
  ASSERT_NE(main_frame, nullptr);
  auto main_graph = main_frame->GetExecuteGraph();
  ASSERT_NE(main_graph, nullptr);
  auto output_data_desc = ge::MakeShared<ge::OpDesc>("output_data", "OutputData");
  ASSERT_NE(output_data_desc, nullptr);
  ASSERT_NE(main_graph->AddNode(output_data_desc), nullptr);
  auto root_frame = ValueHolder::PopGraphFrame();
  ASSERT_NE(root_frame, nullptr);
  auto root_graph = root_frame->GetExecuteGraph();
  ASSERT_NE(root_graph, nullptr);

  LoweringOption option;
  LoweringGlobalData global_data;
  ASSERT_EQ(OfflineOptimizer(option, global_data).Run(root_graph.get()), ge::GRAPH_SUCCESS);
  EXPECT_EQ(ge::ExecuteGraphUtils::FindNodesByTypeFromAllNodes(root_graph.get(), kernel::kCopyFlowLaunch).size(), 1UL);
  EXPECT_EQ(ge::ExecuteGraphUtils::FindNodesByTypeFromAllNodes(root_graph.get(), kernel::kAllocCopyFlowHbm).size(),
            0UL);
  EXPECT_EQ(ge::ExecuteGraphUtils::FindNodesByTypeFromAllNodes(root_graph.get(), "PrepareCopyFlowResult").size(), 0UL);
  EXPECT_EQ(ge::ExecuteGraphUtils::FindNodesByTypeFromAllNodes(root_graph.get(), "LaunchCopyFlowH2D").size(), 0UL);
}

TEST_F(SplitMixedLaunchMemoryUT, OfflineOptimizerPreservesLaunchFreeEdgeForSingleThread) {
  SetRt2SingleThreadEnabled();
  std::unique_ptr<GraphFrame> init_frame;
  std::unique_ptr<GraphFrame> de_init_frame;
  Create3StageFrames(init_frame, de_init_frame);
  ASSERT_NE(init_frame, nullptr);
  auto init_netoutput_desc = ge::MakeShared<ge::OpDesc>("init_netoutput", "InnerNetOutput");
  ASSERT_NE(init_netoutput_desc, nullptr);
  ASSERT_NE(init_frame->GetExecuteGraph()->AddNode(init_netoutput_desc), nullptr);
  auto main_graph = BuildCopyToDirectLaunchWithFreeControlGraph();
  ASSERT_NE(main_graph, nullptr);
  auto output_data_desc = ge::MakeShared<ge::OpDesc>("output_data", "OutputData");
  ASSERT_NE(output_data_desc, nullptr);
  ASSERT_NE(main_graph->AddNode(output_data_desc), nullptr);
  auto root_frame = ValueHolder::PopGraphFrame();
  ASSERT_NE(root_frame, nullptr);
  auto root_graph = root_frame->GetExecuteGraph();
  ASSERT_NE(root_graph, nullptr);

  LoweringOption option;
  LoweringGlobalData global_data;
  ASSERT_EQ(OfflineOptimizer(option, global_data).Run(root_graph.get()), ge::GRAPH_SUCCESS);

  EXPECT_EQ(ge::ExecuteGraphUtils::FindNodesByTypeFromAllNodes(root_graph.get(), "CalcDeviceCopySizes").size(), 0UL);
  EXPECT_EQ(ge::ExecuteGraphUtils::FindNodesByTypeFromAllNodes(root_graph.get(), "AllocMemHbm").size(), 0UL);
  EXPECT_EQ(ge::ExecuteGraphUtils::FindNodesByTypeFromAllNodes(root_graph.get(), "LaunchH2DCopy").size(), 0UL);
  EXPECT_EQ(ge::ExecuteGraphUtils::FindNodesByTypeFromAllNodes(root_graph.get(), "ShareH2DCopyResult").size(), 0UL);
  EXPECT_EQ(ge::ExecuteGraphUtils::FindNodesByTypeFromAllNodes(root_graph.get(), "PrepareCopyFlowResult").size(), 0UL);
  EXPECT_EQ(ge::ExecuteGraphUtils::FindNodesByTypeFromAllNodes(root_graph.get(), "LaunchCopyFlowH2D").size(), 0UL);
  EXPECT_EQ(ge::ExecuteGraphUtils::FindNodesByTypeFromAllNodes(root_graph.get(), "FreeMemoryHoldAddr").size(), 0UL);

  const auto launch_nodes = ge::ExecuteGraphUtils::FindNodesByTypeFromAllNodes(root_graph.get(), "ExecuteOpLaunch");
  const auto free_nodes = ge::ExecuteGraphUtils::FindNodesByTypeFromAllNodes(root_graph.get(), "FreeMemory");
  ASSERT_EQ(launch_nodes.size(), 1UL);
  ASSERT_EQ(free_nodes.size(), 1UL);
  EXPECT_TRUE(HasControlEdge(launch_nodes[0], free_nodes[0]));
  EXPECT_EQ(main_graph->GetExtAttr<FreeLaunchRelations>(kFreeLaunchRelationsAttr), nullptr);
}

TEST_F(SplitMixedLaunchMemoryUT, OfflineOptimizerPreservesLaunchFreeEdgeAndEmptyCsrForDynamicMultiStream) {
  SetRt2MultiThreadEnabled();
  SetDynamicShapeMultiStreamEnabled();
  ASSERT_FALSE(IsEnableRt2MultiThread());

  std::unique_ptr<GraphFrame> init_frame;
  std::unique_ptr<GraphFrame> de_init_frame;
  Create3StageFrames(init_frame, de_init_frame);
  ASSERT_NE(init_frame, nullptr);
  auto init_netoutput_desc = ge::MakeShared<ge::OpDesc>("init_netoutput", "InnerNetOutput");
  ASSERT_NE(init_netoutput_desc, nullptr);
  ASSERT_NE(init_frame->GetExecuteGraph()->AddNode(init_netoutput_desc), nullptr);

  auto main_graph = BuildCopyToDirectLaunchWithFreeControlGraph();
  ASSERT_NE(main_graph, nullptr);
  auto output_data_desc = ge::MakeShared<ge::OpDesc>("output_data", "OutputData");
  ASSERT_NE(output_data_desc, nullptr);
  ASSERT_NE(main_graph->AddNode(output_data_desc), nullptr);
  auto root_frame = ValueHolder::PopGraphFrame();
  ASSERT_NE(root_frame, nullptr);
  auto root_graph = root_frame->GetExecuteGraph();
  ASSERT_NE(root_graph, nullptr);

  LoweringOption option;
  LoweringGlobalData global_data;
  ASSERT_EQ(OfflineOptimizer(option, global_data).Run(root_graph.get()), ge::GRAPH_SUCCESS);

  EXPECT_EQ(ge::ExecuteGraphUtils::FindNodesByTypeFromAllNodes(root_graph.get(), "CalcDeviceCopySizes").size(), 0UL);
  EXPECT_EQ(ge::ExecuteGraphUtils::FindNodesByTypeFromAllNodes(root_graph.get(), "AllocMemHbm").size(), 0UL);
  EXPECT_EQ(ge::ExecuteGraphUtils::FindNodesByTypeFromAllNodes(root_graph.get(), "LaunchH2DCopy").size(), 0UL);
  EXPECT_EQ(ge::ExecuteGraphUtils::FindNodesByTypeFromAllNodes(root_graph.get(), "ShareH2DCopyResult").size(), 0UL);
  EXPECT_EQ(ge::ExecuteGraphUtils::FindNodesByTypeFromAllNodes(root_graph.get(), "PrepareCopyFlowResult").size(), 0UL);
  EXPECT_EQ(ge::ExecuteGraphUtils::FindNodesByTypeFromAllNodes(root_graph.get(), "LaunchCopyFlowH2D").size(), 0UL);
  EXPECT_EQ(ge::ExecuteGraphUtils::FindNodesByTypeFromAllNodes(root_graph.get(), "FreeMemoryHoldAddr").size(), 0UL);

  const auto launch_nodes = ge::ExecuteGraphUtils::FindNodesByTypeFromAllNodes(root_graph.get(), "ExecuteOpLaunch");
  const auto free_nodes = ge::ExecuteGraphUtils::FindNodesByTypeFromAllNodes(root_graph.get(), "FreeMemory");
  ASSERT_EQ(launch_nodes.size(), 1UL);
  ASSERT_EQ(free_nodes.size(), 1UL);
  EXPECT_TRUE(HasControlEdge(launch_nodes[0], free_nodes[0]));
  EXPECT_EQ(main_graph->GetExtAttr<FreeLaunchRelations>(kFreeLaunchRelationsAttr), nullptr);

  ASSERT_EQ(root_graph->TopologicalSorting(), ge::GRAPH_SUCCESS);
  ASSERT_EQ(AssignUniqueFeedIndexes(main_graph), ge::GRAPH_SUCCESS);
  const auto main_parent = main_graph->GetParentNodeBarePtr();
  ASSERT_NE(main_parent, nullptr);
  ASSERT_EQ(main_parent->GetOpDescBarePtr()->AddOutputDesc(ge::GeTensorDesc()), ge::GRAPH_SUCCESS);
  main_parent->GetExtendInfo()->UpdateOutputSymbols(1U);
  GertRuntimeStub runtime_stub;
  runtime_stub.GetKernelStub().SetUp("SplitTensor", BuildOnlyKernel);
  auto model_level_data = ExeGraphModelLevelDataFaker(root_graph).Build();
  GraphExecutorBuilder executor_builder(model_level_data.GetModelLevelData(), main_graph,
                                        &model_level_data.symbols_to_value);
  auto executor_option = MultiThreadExecutorOption(kLeastThreadNumber);
  executor_builder.ExecutorOpt(executor_option);
  MultiThreadExecutionDataBuilder execution_data_builder(executor_builder);
  auto resource_guard = execution_data_builder.Build();
  ASSERT_NE(resource_guard, nullptr);
  const auto &csr = static_cast<MultiThreadResourceGuard *>(resource_guard.get())->GetFreeLaunchRelationCsr();
  ASSERT_EQ(csr.relation_num, 0U);
  ASSERT_NE(csr.offsets, nullptr);
  for (NodeIdentity node_id = 0U; node_id < csr.node_num; ++node_id) {
    const auto range = csr.GetLaunchIds(node_id);
    EXPECT_EQ(range.size, 0U);
  }
}

TEST_F(SplitMixedLaunchMemoryUT, OfflineOptimizerKeepsMultiOutputLegacyCopyFlowForDynamicMultiStream) {
  SetRt2MultiThreadEnabled();
  SetDynamicShapeMultiStreamEnabled();
  ASSERT_FALSE(IsEnableRt2MultiThread());
  std::unique_ptr<GraphFrame> init_frame;
  std::unique_ptr<GraphFrame> de_init_frame;
  Create3StageFrames(init_frame, de_init_frame);
  ASSERT_NE(init_frame, nullptr);
  auto init_netoutput_desc = ge::MakeShared<ge::OpDesc>("init_netoutput", "InnerNetOutput");
  ASSERT_NE(init_netoutput_desc, nullptr);
  ASSERT_NE(init_frame->GetExecuteGraph()->AddNode(init_netoutput_desc), nullptr);

  auto main_graph = BuildTwoOutputCopyToSingleLaunchGraph(false, true);
  ASSERT_NE(main_graph, nullptr);
  auto output_data_desc = ge::MakeShared<ge::OpDesc>("output_data", "OutputData");
  ASSERT_NE(output_data_desc, nullptr);
  ASSERT_NE(main_graph->AddNode(output_data_desc), nullptr);
  auto root_frame = ValueHolder::PopGraphFrame();
  ASSERT_NE(root_frame, nullptr);
  auto root_graph = root_frame->GetExecuteGraph();
  ASSERT_NE(root_graph, nullptr);

  LoweringOption option;
  LoweringGlobalData global_data;
  ASSERT_EQ(OfflineOptimizer(option, global_data).Run(root_graph.get()), ge::GRAPH_SUCCESS);

  const auto copy_flow_nodes =
      ge::ExecuteGraphUtils::FindNodesByTypeFromAllNodes(root_graph.get(), kernel::kCopyFlowLaunch);
  ASSERT_EQ(copy_flow_nodes.size(), 1UL);
  EXPECT_EQ(copy_flow_nodes[0U]->GetDataOutNum(), 2U);
  const auto input_num_node = ge::ExecuteGraphUtils::FindNodeFromAllNodes(
      root_graph.get(), ("Const_" + copy_flow_nodes[0U]->GetName() + "_Num").c_str());
  const auto indexes_node = ge::ExecuteGraphUtils::FindNodeFromAllNodes(
      root_graph.get(), ("Const_" + copy_flow_nodes[0U]->GetName() + "_Index").c_str());
  ASSERT_NE(input_num_node, nullptr);
  ASSERT_NE(indexes_node, nullptr);
  ge::Buffer input_num_buffer;
  ge::Buffer indexes_buffer;
  ASSERT_TRUE(ge::AttrUtils::GetZeroCopyBytes(input_num_node->GetOpDescBarePtr(), "value", input_num_buffer));
  ASSERT_TRUE(ge::AttrUtils::GetZeroCopyBytes(indexes_node->GetOpDescBarePtr(), "value", indexes_buffer));
  ASSERT_NE(input_num_buffer.GetData(), nullptr);
  EXPECT_EQ(*reinterpret_cast<const size_t *>(input_num_buffer.GetData()), 2U);
  const auto indexes = reinterpret_cast<const ContinuousVectorVector *>(indexes_buffer.GetData());
  ASSERT_NE(indexes, nullptr);
  ASSERT_EQ(indexes->GetSize(), 2U);
  ASSERT_NE(indexes->Get(0U), nullptr);
  ASSERT_NE(indexes->Get(1U), nullptr);
  EXPECT_EQ(indexes->Get(0U)->GetSize(), 1U);
  EXPECT_EQ(indexes->Get(1U)->GetSize(), 1U);
  EXPECT_EQ(
      ge::ExecuteGraphUtils::FindNodesByTypeFromAllNodes(root_graph.get(), kernel::kCalcCopyFlowAllocSizes).size(),
      0UL);
  EXPECT_EQ(ge::ExecuteGraphUtils::FindNodesByTypeFromAllNodes(root_graph.get(), kernel::kAllocCopyFlowHbm).size(),
            0UL);
  EXPECT_EQ(ge::ExecuteGraphUtils::FindNodesByTypeFromAllNodes(root_graph.get(), kernel::kPrepareCopyFlowResult).size(),
            0UL);
  EXPECT_EQ(ge::ExecuteGraphUtils::FindNodesByTypeFromAllNodes(root_graph.get(), kernel::kLaunchCopyFlowH2D).size(),
            0UL);
}

TEST_F(SplitMixedLaunchMemoryUT, SplitCopyH2DToSeparateMemoryAndLaunchKernels) {
  SetRt2MultiThreadEnabled();
  auto graph = BuildCopyToConsumerGraph();
  ASSERT_NE(graph, nullptr);

  bool changed = false;
  ASSERT_EQ(SplitMixedLaunchMemory().Run(graph.get(), changed), ge::GRAPH_SUCCESS);
  EXPECT_TRUE(changed);

  EXPECT_EQ(ge::ExecuteGraphUtils::FindNodesByTypeFromAllNodes(graph.get(), "CopyH2D").size(), 0UL);
  EXPECT_EQ(ge::ExecuteGraphUtils::FindNodesByTypeFromAllNodes(graph.get(), "CalcDeviceCopySizes").size(), 1UL);
  EXPECT_EQ(ge::ExecuteGraphUtils::FindNodesByTypeFromAllNodes(graph.get(), "AllocMemHbm").size(), 1UL);
  auto calc_nodes = ge::ExecuteGraphUtils::FindNodesByTypeFromAllNodes(graph.get(), "CalcDeviceCopySizes");
  ASSERT_EQ(calc_nodes.size(), 1UL);
  EXPECT_EQ(FastNodeTopoChecker(calc_nodes[0]).InChecker().DataFromByType("CalcTensorSizeFromStorage").Result(),
            "success");
  auto launch_nodes = ge::ExecuteGraphUtils::FindNodesByTypeFromAllNodes(graph.get(), "LaunchH2DCopy");
  ASSERT_EQ(launch_nodes.size(), 1UL);
  EXPECT_EQ(launch_nodes[0]->GetDataOutNum(), 0U);
  EXPECT_NE(FastNodeTopoChecker(launch_nodes[0]).OutChecker().DataToByType("Consumer").Result(), "success");
  auto share_nodes = ge::ExecuteGraphUtils::FindNodesByTypeFromAllNodes(graph.get(), kernel::kShareH2DCopyResult);
  ASSERT_EQ(share_nodes.size(), 1UL);
  EXPECT_EQ(FastNodeTopoChecker(share_nodes[0]).InChecker().DataFromByType("AllocMemHbm").Result(), "success");
  EXPECT_EQ(FastNodeTopoChecker(share_nodes[0]).OutChecker().CtrlToByType("LaunchH2DCopy").Result(), "success");
  EXPECT_EQ(FastNodeTopoChecker(share_nodes[0]).OutChecker().DataToByType("Consumer").Result(), "success");
  auto free_nodes = ge::ExecuteGraphUtils::FindNodesByTypeFromAllNodes(graph.get(), "FreeMemory");
  ASSERT_EQ(free_nodes.size(), 1UL);
  EXPECT_EQ(FastNodeTopoChecker(free_nodes[0]).StrictConnectFrom({{kernel::kShareH2DCopyResult, 0}, {"Consumer", -1}}),
            "success");
}

TEST_F(SplitMixedLaunchMemoryUT, DirectLaunchConsumerUsesSharedDataAndWaitsForCopySubmission) {
  SetRt2MultiThreadEnabled();
  auto graph = BuildCopyToDirectLaunchConsumerGraph();
  ASSERT_NE(graph, nullptr);

  bool changed = false;
  ASSERT_EQ(SplitMixedLaunchMemory().Run(graph.get(), changed), ge::GRAPH_SUCCESS);
  ASSERT_TRUE(changed);

  auto launch_copy_nodes = ge::ExecuteGraphUtils::FindNodesByTypeFromAllNodes(graph.get(), "LaunchH2DCopy");
  auto share_nodes = ge::ExecuteGraphUtils::FindNodesByTypeFromAllNodes(graph.get(), kernel::kShareH2DCopyResult);
  auto consumer_nodes = ge::ExecuteGraphUtils::FindNodesByTypeFromAllNodes(graph.get(), "ExecuteOpLaunch");
  ASSERT_EQ(launch_copy_nodes.size(), 1UL);
  ASSERT_EQ(share_nodes.size(), 1UL);
  ASSERT_EQ(consumer_nodes.size(), 1UL);
  EXPECT_EQ(FastNodeTopoChecker(share_nodes[0]).OutChecker().DataToByType("ExecuteOpLaunch").Result(), "success");
  EXPECT_EQ(FastNodeTopoChecker(launch_copy_nodes[0]).OutChecker().CtrlToByType("ExecuteOpLaunch").Result(), "success");
  EXPECT_NE(FastNodeTopoChecker(launch_copy_nodes[0]).OutChecker().DataToByType("ExecuteOpLaunch").Result(), "success");
}

TEST_F(SplitMixedLaunchMemoryUT, HcomLaunchConsumerWaitsForCopySubmission) {
  SetRt2MultiThreadEnabled();
  auto graph = BuildCopyToDirectLaunchConsumerGraph("LaunchHcomKernel");
  ASSERT_NE(graph, nullptr);

  bool changed = false;
  ASSERT_EQ(SplitMixedLaunchMemory().Run(graph.get(), changed), ge::GRAPH_SUCCESS);
  ASSERT_TRUE(changed);

  auto launch_copy_nodes = ge::ExecuteGraphUtils::FindNodesByTypeFromAllNodes(graph.get(), "LaunchH2DCopy");
  ASSERT_EQ(launch_copy_nodes.size(), 1UL);
  EXPECT_EQ(FastNodeTopoChecker(launch_copy_nodes[0]).OutChecker().CtrlToByType("LaunchHcomKernel").Result(),
            "success");
}

TEST_F(SplitMixedLaunchMemoryUT, CriticalSectionLaunchConsumerWaitsForCopySubmission) {
  SetRt2MultiThreadEnabled();
  auto graph = BuildCopyToDirectLaunchConsumerGraph("GenerateSqeAndLaunchTask");
  ASSERT_NE(graph, nullptr);

  bool changed = false;
  ASSERT_EQ(SplitMixedLaunchMemory().Run(graph.get(), changed), ge::GRAPH_SUCCESS);
  ASSERT_TRUE(changed);

  auto launch_copy_nodes = ge::ExecuteGraphUtils::FindNodesByTypeFromAllNodes(graph.get(), "LaunchH2DCopy");
  ASSERT_EQ(launch_copy_nodes.size(), 1UL);
  EXPECT_EQ(FastNodeTopoChecker(launch_copy_nodes[0]).OutChecker().CtrlToByType("GenerateSqeAndLaunchTask").Result(),
            "success");
}

TEST_F(SplitMixedLaunchMemoryUT, CustomConsumerBehindBuildRefTensorWaitsForCopySubmission) {
  SetRt2MultiThreadEnabled();
  auto graph = BuildCopyToCustomConsumerGraph();
  ASSERT_NE(graph, nullptr);

  bool changed = false;
  ASSERT_EQ(SplitMixedLaunchMemory().Run(graph.get(), changed), ge::GRAPH_SUCCESS);
  ASSERT_TRUE(changed);

  auto launch_copy_nodes = ge::ExecuteGraphUtils::FindNodesByTypeFromAllNodes(graph.get(), "LaunchH2DCopy");
  ASSERT_EQ(launch_copy_nodes.size(), 1UL);
  EXPECT_EQ(FastNodeTopoChecker(launch_copy_nodes[0]).OutChecker().CtrlToByType("ExecuteCustomOp").Result(), "success");
}

TEST_F(SplitMixedLaunchMemoryUT, OfflineOptimizerKeepsFusedMultiOutputCopyFlowRuntimeContract) {
  SetRt2MultiThreadEnabled();
  std::unique_ptr<GraphFrame> init_frame;
  std::unique_ptr<GraphFrame> de_init_frame;
  Create3StageFrames(init_frame, de_init_frame);
  ASSERT_NE(init_frame, nullptr);
  auto init_netoutput_desc = ge::MakeShared<ge::OpDesc>("init_netoutput", "InnerNetOutput");
  ASSERT_NE(init_netoutput_desc, nullptr);
  ASSERT_NE(init_frame->GetExecuteGraph()->AddNode(init_netoutput_desc), nullptr);

  auto main_graph = BuildTwoOutputCopyToSingleLaunchGraph(true, true);
  ASSERT_NE(main_graph, nullptr);
  auto output_data_desc = ge::MakeShared<ge::OpDesc>("output_data", "OutputData");
  ASSERT_NE(output_data_desc, nullptr);
  ASSERT_NE(main_graph->AddNode(output_data_desc), nullptr);
  auto root_frame = ValueHolder::PopGraphFrame();
  ASSERT_NE(root_frame, nullptr);
  auto root_graph = root_frame->GetExecuteGraph();
  ASSERT_NE(root_graph, nullptr);

  LoweringOption option;
  LoweringGlobalData global_data;
  ASSERT_EQ(OfflineOptimizer(option, global_data).Run(root_graph.get()), ge::GRAPH_SUCCESS);

  auto prepare_nodes =
      ge::ExecuteGraphUtils::FindNodesByTypeFromAllNodes(root_graph.get(), kernel::kPrepareCopyFlowResult);
  auto launch_nodes = ge::ExecuteGraphUtils::FindNodesByTypeFromAllNodes(root_graph.get(), kernel::kLaunchCopyFlowH2D);
  ASSERT_EQ(prepare_nodes.size(), 1UL);
  ASSERT_EQ(launch_nodes.size(), 1UL);
  EXPECT_EQ(launch_nodes[0]->GetDataOutNum(), 0U);

  const auto input_num_edge =
      prepare_nodes[0]->GetInDataEdgeByIndex(static_cast<int32_t>(kernel::PrepareCopyFlowResultInputs::kInputsNum));
  ASSERT_NE(input_num_edge, nullptr);
  const std::string launch_suffix = "_LaunchCopyFlowH2D";
  ASSERT_GT(launch_nodes[0]->GetName().size(), launch_suffix.size());
  const auto copy_flow_name =
      launch_nodes[0]->GetName().substr(0U, launch_nodes[0]->GetName().size() - launch_suffix.size());
  const auto input_num_node =
      ge::ExecuteGraphUtils::FindNodeFromAllNodes(root_graph.get(), ("Const_" + copy_flow_name + "_Num").c_str());
  ASSERT_NE(input_num_node, nullptr);
  ge::Buffer input_num_buffer;
  ASSERT_TRUE(ge::AttrUtils::GetZeroCopyBytes(input_num_node->GetOpDescBarePtr(), "value", input_num_buffer));
  ASSERT_NE(input_num_buffer.GetData(), nullptr);
  EXPECT_EQ(*reinterpret_cast<const size_t *>(input_num_buffer.GetData()), 2U);

  const auto indexes_edge =
      prepare_nodes[0]->GetInDataEdgeByIndex(static_cast<int32_t>(kernel::PrepareCopyFlowResultInputs::kInputsIndex));
  ASSERT_NE(indexes_edge, nullptr);
  const auto indexes_node =
      ge::ExecuteGraphUtils::FindNodeFromAllNodes(root_graph.get(), ("Const_" + copy_flow_name + "_Index").c_str());
  ASSERT_NE(indexes_node, nullptr);
  ge::Buffer indexes_buffer;
  ASSERT_TRUE(ge::AttrUtils::GetZeroCopyBytes(indexes_node->GetOpDescBarePtr(), "value", indexes_buffer));
  const auto indexes = reinterpret_cast<const ContinuousVectorVector *>(indexes_buffer.GetData());
  ASSERT_NE(indexes, nullptr);
  ASSERT_EQ(indexes->GetSize(), 2U);
  const auto first_output_indexes = indexes->Get(0U);
  const auto second_output_indexes = indexes->Get(1U);
  ASSERT_NE(first_output_indexes, nullptr);
  ASSERT_NE(second_output_indexes, nullptr);
  ASSERT_EQ(first_output_indexes->GetSize(), 2U);
  ASSERT_EQ(second_output_indexes->GetSize(), 1U);
  const auto first_output_index_data = reinterpret_cast<const int32_t *>(first_output_indexes->GetData());
  const auto second_output_index_data = reinterpret_cast<const int32_t *>(second_output_indexes->GetData());
  ASSERT_NE(first_output_index_data, nullptr);
  ASSERT_NE(second_output_index_data, nullptr);
  EXPECT_EQ(first_output_index_data[0U], 1);
  EXPECT_EQ(first_output_index_data[1U], 3);
  EXPECT_EQ(second_output_index_data[0U], 2);

  const auto allocated_addrs_edge =
      launch_nodes[0]->GetInDataEdgeByIndex(static_cast<int32_t>(kernel::LaunchCopyFlowH2DInputs::kAllocatedAddrs));
  ASSERT_NE(allocated_addrs_edge, nullptr);
  EXPECT_STREQ(allocated_addrs_edge->src->GetTypePtr(), kernel::kAllocCopyFlowHbm);
  EXPECT_EQ(allocated_addrs_edge->src_output, 0);

  const auto consumer_launch_nodes =
      ge::ExecuteGraphUtils::FindNodesByTypeFromAllNodes(root_graph.get(), "LaunchKernelWithFlag");
  const auto free_nodes = ge::ExecuteGraphUtils::FindNodesByTypeFromAllNodes(root_graph.get(), "FreeMemory");
  const auto free_hold_addr_nodes =
      ge::ExecuteGraphUtils::FindNodesByTypeFromAllNodes(root_graph.get(), "FreeMemoryHoldAddr");
  ASSERT_EQ(consumer_launch_nodes.size(), 1UL);
  EXPECT_EQ(free_nodes.size(), 0UL);
  ASSERT_EQ(free_hold_addr_nodes.size(), 2UL);
  EXPECT_EQ(launch_nodes[0]->GetDataOutNum(), 0U);
  EXPECT_EQ(FastNodeTopoChecker(prepare_nodes[0]).OutChecker().DataToByType("LaunchKernelWithFlag").Result(),
            "success");
  EXPECT_FALSE(HasControlEdge(consumer_launch_nodes[0], free_hold_addr_nodes[0]));
  EXPECT_FALSE(HasControlEdge(consumer_launch_nodes[0], free_hold_addr_nodes[1]));
  for (const auto free_hold_addr_node : free_hold_addr_nodes) {
    EXPECT_EQ(FastNodeTopoChecker(free_hold_addr_node).InChecker().DataFromByType("PrepareCopyFlowResult").Result(),
              "success");
  }
  const auto relations = main_graph->GetExtAttr<FreeLaunchRelations>(kFreeLaunchRelationsAttr);
  ASSERT_NE(relations, nullptr);
  ASSERT_EQ(relations->size(), 2U);
  for (const auto free_hold_addr_node : free_hold_addr_nodes) {
    EXPECT_NE(std::find(relations->cbegin(), relations->cend(),
                        std::make_pair(free_hold_addr_node, consumer_launch_nodes[0])),
              relations->cend());
  }
}

TEST_F(SplitMixedLaunchMemoryUT, OfflineOptimizerKeepsSeventeenCopyFlowOutputsAndRelations) {
  constexpr size_t kCopyFlowCount = 17U;
  SetRt2MultiThreadEnabled();
  std::unique_ptr<GraphFrame> init_frame;
  std::unique_ptr<GraphFrame> de_init_frame;
  Create3StageFrames(init_frame, de_init_frame);
  ASSERT_NE(init_frame, nullptr);
  auto init_netoutput_desc = ge::MakeShared<ge::OpDesc>("init_netoutput", "InnerNetOutput");
  ASSERT_NE(init_netoutput_desc, nullptr);
  ASSERT_NE(init_frame->GetExecuteGraph()->AddNode(init_netoutput_desc), nullptr);

  auto main_graph = BuildCopyToLaunchGraph(kCopyFlowCount, true);
  ASSERT_NE(main_graph, nullptr);
  auto output_data_desc = ge::MakeShared<ge::OpDesc>("output_data", "OutputData");
  ASSERT_NE(output_data_desc, nullptr);
  ASSERT_NE(main_graph->AddNode(output_data_desc), nullptr);
  auto root_frame = ValueHolder::PopGraphFrame();
  ASSERT_NE(root_frame, nullptr);
  auto root_graph = root_frame->GetExecuteGraph();
  ASSERT_NE(root_graph, nullptr);

  LoweringOption option;
  LoweringGlobalData global_data;
  ASSERT_EQ(OfflineOptimizer(option, global_data).Run(root_graph.get()), ge::GRAPH_SUCCESS);

  const auto prepare_nodes =
      ge::ExecuteGraphUtils::FindNodesByTypeFromAllNodes(root_graph.get(), kernel::kPrepareCopyFlowResult);
  const auto launch_copy_nodes =
      ge::ExecuteGraphUtils::FindNodesByTypeFromAllNodes(root_graph.get(), kernel::kLaunchCopyFlowH2D);
  const auto consumer_nodes =
      ge::ExecuteGraphUtils::FindNodesByTypeFromAllNodes(root_graph.get(), "LaunchKernelWithFlag");
  const auto free_hold_addr_nodes =
      ge::ExecuteGraphUtils::FindNodesByTypeFromAllNodes(root_graph.get(), "FreeMemoryHoldAddr");
  ASSERT_EQ(prepare_nodes.size(), 1U);
  ASSERT_EQ(launch_copy_nodes.size(), 1U);
  ASSERT_EQ(consumer_nodes.size(), 1U);
  ASSERT_EQ(free_hold_addr_nodes.size(), kCopyFlowCount);

  int64_t prepare_count = 0;
  int64_t launch_count = 0;
  ASSERT_TRUE(ge::AttrUtils::GetInt(prepare_nodes[0]->GetOpDescBarePtr(), kernel::kCopyFlowCountAttr, prepare_count));
  ASSERT_TRUE(
      ge::AttrUtils::GetInt(launch_copy_nodes[0]->GetOpDescBarePtr(), kernel::kCopyFlowCountAttr, launch_count));
  EXPECT_EQ(prepare_count, static_cast<int64_t>(kCopyFlowCount));
  EXPECT_EQ(launch_count, static_cast<int64_t>(kCopyFlowCount));

  std::vector<int32_t> expected_output_indexes(kCopyFlowCount);
  std::iota(expected_output_indexes.begin(), expected_output_indexes.end(), 0);
  std::vector<int32_t> consumer_prepare_outputs;
  for (const auto input_edge : consumer_nodes[0]->GetAllInDataEdgesRef()) {
    if ((input_edge != nullptr) && (input_edge->src == prepare_nodes[0])) {
      consumer_prepare_outputs.emplace_back(input_edge->src_output);
    }
  }
  EXPECT_EQ(consumer_prepare_outputs, expected_output_indexes);

  std::set<int32_t> free_prepare_outputs;
  const auto relations = main_graph->GetExtAttr<FreeLaunchRelations>(kFreeLaunchRelationsAttr);
  ASSERT_NE(relations, nullptr);
  ASSERT_EQ(relations->size(), kCopyFlowCount);
  for (const auto free_hold_addr_node : free_hold_addr_nodes) {
    const auto input_edge = free_hold_addr_node->GetInDataEdgeByIndex(0);
    ASSERT_NE(input_edge, nullptr);
    EXPECT_EQ(input_edge->src, prepare_nodes[0]);
    free_prepare_outputs.emplace(input_edge->src_output);
    EXPECT_NE(std::find(relations->cbegin(), relations->cend(), std::make_pair(free_hold_addr_node, consumer_nodes[0])),
              relations->cend());
  }
  EXPECT_EQ(free_prepare_outputs, std::set<int32_t>(expected_output_indexes.cbegin(), expected_output_indexes.cend()));
}

TEST_F(SplitMixedLaunchMemoryUT, OfflineOptimizerMapsLegacyCopyOutputsToSurvivingPrepareForSubscribers) {
  SetRt2MultiThreadEnabled();
  std::unique_ptr<GraphFrame> init_frame;
  std::unique_ptr<GraphFrame> de_init_frame;
  Create3StageFrames(init_frame, de_init_frame);
  ASSERT_NE(init_frame, nullptr);
  auto init_netoutput_desc = ge::MakeShared<ge::OpDesc>("init_netoutput", "InnerNetOutput");
  ASSERT_NE(init_netoutput_desc, nullptr);
  ASSERT_NE(init_frame->GetExecuteGraph()->AddNode(init_netoutput_desc), nullptr);

  std::vector<ValueHolderPtr> original_copy_outputs;
  auto main_graph = BuildTwoOutputCopyToSingleLaunchGraph(false, true, &original_copy_outputs);
  ASSERT_NE(main_graph, nullptr);
  ASSERT_EQ(original_copy_outputs.size(), 2UL);
  const auto original_copy_op_desc = original_copy_outputs[0U]->GetFastNode()->GetOpDescPtr();
  ASSERT_NE(original_copy_op_desc, nullptr);
  const auto consumer_launch_nodes =
      ge::ExecuteGraphUtils::FindNodesByTypeFromAllNodes(main_graph.get(), "LaunchKernelWithFlag");
  ASSERT_EQ(consumer_launch_nodes.size(), 1UL);
  const auto consumer_launch_name = consumer_launch_nodes[0U]->GetName();

  auto output_data_desc = ge::MakeShared<ge::OpDesc>("output_data", "OutputData");
  ASSERT_NE(output_data_desc, nullptr);
  ASSERT_NE(main_graph->AddNode(output_data_desc), nullptr);
  auto root_frame = ValueHolder::PopGraphFrame();
  ASSERT_NE(root_frame, nullptr);
  auto root_graph = root_frame->GetExecuteGraph();
  ASSERT_NE(root_graph, nullptr);

  LoweringOption option;
  LoweringGlobalData global_data;
  ASSERT_EQ(OfflineOptimizer(option, global_data).Run(root_graph.get()), ge::GRAPH_SUCCESS);

  const auto prepare_nodes =
      ge::ExecuteGraphUtils::FindNodesByTypeFromAllNodes(root_graph.get(), kernel::kPrepareCopyFlowResult);
  ASSERT_EQ(prepare_nodes.size(), 1UL);
  const auto prepare_node = prepare_nodes[0U];
  const auto pass_changed_info = original_copy_op_desc->TryGetExtAttr(kPassChangedInfo, PassChangedKernels{});
  ASSERT_EQ(pass_changed_info.pass_changed_kernels.size(), 2UL);
  for (int32_t output_index = 0; output_index < 2; ++output_index) {
    const auto mapping = std::find_if(
        pass_changed_info.pass_changed_kernels.cbegin(), pass_changed_info.pass_changed_kernels.cend(),
        [output_index, &consumer_launch_name](const std::pair<KernelNameAndIdx, KernelNameAndIdx> &candidate) {
          return (candidate.first.idx == output_index) && (candidate.first.launch_name == consumer_launch_name);
        });
    ASSERT_NE(mapping, pass_changed_info.pass_changed_kernels.cend());
    EXPECT_EQ(mapping->second.kernel_name, prepare_node->GetName());
    EXPECT_EQ(mapping->second.idx, output_index);
    EXPECT_EQ(ge::ExecuteGraphUtils::FindNodeFromAllNodes(root_graph.get(), mapping->second.kernel_name.c_str()),
              prepare_node);
  }

  AsyncAnyValue output_chain_0;
  AsyncAnyValue output_chain_1;
  constexpr uint64_t kOutputSymbol0 = 101U;
  constexpr uint64_t kOutputSymbol1 = 102U;
  ASSERT_EQ(prepare_node->GetExtendInfo()->SetOutputSymbol(0U, kOutputSymbol0), ge::GRAPH_SUCCESS);
  ASSERT_EQ(prepare_node->GetExtendInfo()->SetOutputSymbol(1U, kOutputSymbol1), ge::GRAPH_SUCCESS);
  const auto output_symbol_0 = prepare_node->GetExtendInfo()->GetOutputSymbol(0U);
  const auto output_symbol_1 = prepare_node->GetExtendInfo()->GetOutputSymbol(1U);
  ASSERT_NE(output_symbol_0, ge::kInvalidSymbol);
  ASSERT_NE(output_symbol_1, ge::kInvalidSymbol);
  SymbolsToValue symbols_to_value{{output_symbol_0, &output_chain_0}, {output_symbol_1, &output_chain_1}};
  const auto extend_info = ge::MakeShared<const SubscriberExtendInfo>(nullptr, root_graph, nullptr, ge::ModelData{},
                                                                      nullptr, symbols_to_value, 0U, "", nullptr,
                                                                      std::unordered_map<std::string, TraceAttr>{});
  ASSERT_NE(extend_info, nullptr);
  CannProfilerV2 profiler(extend_info);
  std::unordered_map<std::string, ge::FastNode *> kernel_names_to_nodes{{prepare_node->GetName(), prepare_node}};
  EXPECT_EQ(profiler.InitOutputChainFromEquivalentDataEdges(original_copy_outputs[0U], kernel_names_to_nodes),
            reinterpret_cast<Chain *>(&output_chain_0));
  EXPECT_EQ(profiler.InitOutputChainFromEquivalentDataEdges(original_copy_outputs[1U], kernel_names_to_nodes),
            reinterpret_cast<Chain *>(&output_chain_1));

  GertRuntimeStub stub;
  stub.GetSlogStub().NoConsoleOut();
  ExecutorDumper dumper(extend_info);
  auto compute_graph = std::make_shared<ge::ComputeGraph>("subscriber_compute_graph");
  auto compute_op_desc = ge::MakeShared<ge::OpDesc>("subscriber_compute", "Compute");
  ASSERT_NE(compute_op_desc, nullptr);
  NodeDumpUnit dump_unit;
  dump_unit.node = compute_graph->AddNode(compute_op_desc);
  ASSERT_NE(dump_unit.node, nullptr);
  dumper.compute_node_name_to_launch_kernel_name_[compute_op_desc->GetName()] = consumer_launch_name;
  for (int32_t output_index = 0; output_index < 2; ++output_index) {
    const auto resolved = dumper.GetKernelNameAndIdxAfterPass(
        original_copy_op_desc.get(), {original_copy_op_desc->GetName(), output_index}, &dump_unit);
    EXPECT_EQ(resolved.kernel_name, prepare_node->GetName());
    EXPECT_EQ(resolved.idx, output_index);
  }
}

TEST_F(SplitMixedLaunchMemoryUT, OfflineOptimizerCopiesOnlyExplicitCrossLaunchGuarderMappings) {
  SetRt2MultiThreadEnabled();
  std::unique_ptr<GraphFrame> init_frame;
  std::unique_ptr<GraphFrame> de_init_frame;
  Create3StageFrames(init_frame, de_init_frame);
  ASSERT_NE(init_frame, nullptr);
  auto init_netoutput_desc = ge::MakeShared<ge::OpDesc>("init_netoutput", "InnerNetOutput");
  ASSERT_NE(init_netoutput_desc, nullptr);
  ASSERT_NE(init_frame->GetExecuteGraph()->AddNode(init_netoutput_desc), nullptr);

  std::vector<std::string> consumer_launch_names;
  auto main_graph = BuildTwoOutputCopyToTwoLaunchesGraph(true, &consumer_launch_names);
  ASSERT_NE(main_graph, nullptr);
  ASSERT_EQ(consumer_launch_names.size(), 2UL);
  auto output_data_desc = ge::MakeShared<ge::OpDesc>("output_data", "OutputData");
  ASSERT_NE(output_data_desc, nullptr);
  ASSERT_NE(main_graph->AddNode(output_data_desc), nullptr);
  auto root_frame = ValueHolder::PopGraphFrame();
  ASSERT_NE(root_frame, nullptr);
  auto root_graph = root_frame->GetExecuteGraph();
  ASSERT_NE(root_graph, nullptr);

  LoweringOption option;
  LoweringGlobalData global_data;
  ASSERT_EQ(OfflineOptimizer(option, global_data).Run(root_graph.get()), ge::GRAPH_SUCCESS);

  const auto prepare_nodes =
      ge::ExecuteGraphUtils::FindNodesByTypeFromAllNodes(root_graph.get(), kernel::kPrepareCopyFlowResult);
  ASSERT_EQ(prepare_nodes.size(), 2UL);
  for (size_t launch_index = 0U; launch_index < consumer_launch_names.size(); ++launch_index) {
    const auto prepare_name = "CopyFlowLaunch_To_" + consumer_launch_names[launch_index];
    const auto prepare = ge::ExecuteGraphUtils::FindNodeFromAllNodes(root_graph.get(), prepare_name.c_str());
    ASSERT_NE(prepare, nullptr);
    EXPECT_EQ(prepare->GetType(), kernel::kPrepareCopyFlowResult);
    EXPECT_EQ(GetGuarderCount(prepare, 0U), launch_index == 0U ? 1U : 0U);
    EXPECT_EQ(GetGuarderCount(prepare, 1U), launch_index == 1U ? 1U : 0U);
  }
}

TEST_F(SplitMixedLaunchMemoryUT, SplitSingleStageAclnnOrdersCopySubmissionBeforeExecuteOpFunc) {
  SetRt2MultiThreadEnabled();
  auto graph = BuildSingleStageAclnnCopyGraph();
  ASSERT_NE(graph, nullptr);

  bool changed = false;
  ASSERT_EQ(SplitMixedLaunchMemory().Run(graph.get(), changed), ge::GRAPH_SUCCESS);
  ASSERT_TRUE(changed);

  auto launch_copy_nodes = ge::ExecuteGraphUtils::FindNodesByTypeFromAllNodes(graph.get(), "LaunchH2DCopy");
  auto share_nodes = ge::ExecuteGraphUtils::FindNodesByTypeFromAllNodes(graph.get(), kernel::kShareH2DCopyResult);
  ASSERT_EQ(launch_copy_nodes.size(), 1UL);
  ASSERT_EQ(share_nodes.size(), 1UL);
  EXPECT_EQ(FastNodeTopoChecker(launch_copy_nodes[0]).OutChecker().CtrlToByType("ExecuteOpFunc").Result(), "success");
  EXPECT_NE(FastNodeTopoChecker(launch_copy_nodes[0]).OutChecker().CtrlToByType("BuildRefTensor").Result(), "success");
  EXPECT_EQ(FastNodeTopoChecker(share_nodes[0]).OutChecker().DataToByType("BuildRefTensor").Result(), "success");
}

TEST_F(SplitMixedLaunchMemoryUT, OfflineOptimizerSplitsSingleStageAclnnCopyForMultiThread) {
  SetRt2MultiThreadEnabled();
  std::unique_ptr<GraphFrame> init_frame;
  std::unique_ptr<GraphFrame> de_init_frame;
  Create3StageFrames(init_frame, de_init_frame);
  ASSERT_NE(init_frame, nullptr);
  auto init_netoutput_desc = ge::MakeShared<ge::OpDesc>("init_netoutput", "InnerNetOutput");
  ASSERT_NE(init_netoutput_desc, nullptr);
  ASSERT_NE(init_frame->GetExecuteGraph()->AddNode(init_netoutput_desc), nullptr);

  auto main_graph = BuildSingleStageAclnnCopyGraph();
  ASSERT_NE(main_graph, nullptr);
  auto output_data_desc = ge::MakeShared<ge::OpDesc>("output_data", "OutputData");
  ASSERT_NE(output_data_desc, nullptr);
  ASSERT_NE(main_graph->AddNode(output_data_desc), nullptr);
  auto root_frame = ValueHolder::PopGraphFrame();
  ASSERT_NE(root_frame, nullptr);
  auto root_graph = root_frame->GetExecuteGraph();
  ASSERT_NE(root_graph, nullptr);

  LoweringOption option;
  LoweringGlobalData global_data;
  ASSERT_EQ(OfflineOptimizer(option, global_data).Run(root_graph.get()), ge::GRAPH_SUCCESS);

  auto launch_copy_nodes = ge::ExecuteGraphUtils::FindNodesByTypeFromAllNodes(root_graph.get(), "LaunchH2DCopy");
  auto share_nodes = ge::ExecuteGraphUtils::FindNodesByTypeFromAllNodes(root_graph.get(), kernel::kShareH2DCopyResult);
  ASSERT_EQ(launch_copy_nodes.size(), 1UL);
  ASSERT_EQ(share_nodes.size(), 1UL);
  EXPECT_EQ(FastNodeTopoChecker(launch_copy_nodes[0]).OutChecker().CtrlToByType("ExecuteOpFunc").Result(), "success");
  EXPECT_NE(FastNodeTopoChecker(launch_copy_nodes[0]).OutChecker().CtrlToByType("BuildRefTensor").Result(), "success");
  EXPECT_EQ(FastNodeTopoChecker(share_nodes[0]).OutChecker().DataToByType("BuildRefTensor").Result(), "success");
}

TEST_F(SplitMixedLaunchMemoryUT, SplitAclnnPrepareCopyInMainGraph) {
  SetRt2MultiThreadEnabled();
  auto graph = BuildAclnnPrepareCopyGraph();
  ASSERT_NE(graph, nullptr);

  bool changed = false;
  ASSERT_EQ(SplitMixedLaunchMemory().Run(graph.get(), changed), ge::GRAPH_SUCCESS);
  EXPECT_TRUE(changed);
  auto calc_nodes = ge::ExecuteGraphUtils::FindNodesByTypeFromAllNodes(graph.get(), "CalcDeviceCopySizes");
  ASSERT_EQ(calc_nodes.size(), 1UL);
  EXPECT_NE(
      calc_nodes[0]->GetInDataEdgeByIndex(static_cast<int32_t>(kernel::CalcDeviceCopySizesInputs::kOriginalTensorSize)),
      nullptr);

  auto launch_copy_nodes = ge::ExecuteGraphUtils::FindNodesByTypeFromAllNodes(graph.get(), "LaunchH2DCopy");
  ASSERT_EQ(launch_copy_nodes.size(), 1UL);
  EXPECT_FALSE(ge::AttrUtils::HasAttr(launch_copy_nodes[0]->GetOpDescBarePtr(), kComputeNodeIndex));
  EXPECT_EQ(launch_copy_nodes[0]->GetDataOutNum(), 0U);
  EXPECT_EQ(FastNodeTopoChecker(launch_copy_nodes[0]).OutChecker().CtrlToByType("ExecuteOpLaunch").Result(), "success");
  EXPECT_NE(FastNodeTopoChecker(launch_copy_nodes[0]).OutChecker().DataToByType("BuildRefTensor").Result(), "success");
  EXPECT_NE(FastNodeTopoChecker(launch_copy_nodes[0]).OutChecker().CtrlToByType("BuildRefTensor").Result(), "success");
  EXPECT_NE(FastNodeTopoChecker(launch_copy_nodes[0]).OutChecker().CtrlToByType("ExecuteOpPrepare").Result(),
            "success");
  ASSERT_EQ(launch_copy_nodes[0]->GetAllOutControlEdges().size(), 1UL);

  auto share_nodes = ge::ExecuteGraphUtils::FindNodesByTypeFromAllNodes(graph.get(), "ShareH2DCopyResult");
  ASSERT_EQ(share_nodes.size(), 1UL);
  EXPECT_EQ(FastNodeTopoChecker(share_nodes[0]).OutChecker().DataToByType("BuildRefTensor").Result(), "success");

  auto free_nodes = ge::ExecuteGraphUtils::FindNodesByTypeFromAllNodes(graph.get(), "FreeMemory");
  ASSERT_EQ(free_nodes.size(), 1UL);
  EXPECT_EQ(FastNodeTopoChecker(free_nodes[0]).InChecker().DataFromByType("ShareH2DCopyResult").Result(), "success");
  EXPECT_EQ(FastNodeTopoChecker(free_nodes[0]).InChecker().CtrlFromByType("ExecuteOpLaunch").Result(), "success");

  changed = false;
  ASSERT_EQ(RemoveLaunchFreeEdge().Run(graph.get(), changed), ge::GRAPH_SUCCESS);
  EXPECT_TRUE(changed);

  free_nodes = ge::ExecuteGraphUtils::FindNodesByTypeFromAllNodes(graph.get(), "FreeMemory");
  EXPECT_EQ(free_nodes.size(), 0UL);
  auto free_hold_addr_nodes = ge::ExecuteGraphUtils::FindNodesByTypeFromAllNodes(graph.get(), "FreeMemoryHoldAddr");
  ASSERT_EQ(free_hold_addr_nodes.size(), 1UL);
  EXPECT_NE(FastNodeTopoChecker(free_hold_addr_nodes[0]).InChecker().CtrlFromByType("ExecuteOpLaunch").Result(),
            "success");
}

TEST_F(SplitMixedLaunchMemoryUT, SplitAclnnPrepareCopyInSubgraph) {
  SetRt2MultiThreadEnabled();
  auto graph = BuildRootGraphWithAclnnPrepareCopySubgraph();
  ASSERT_NE(graph, nullptr);

  bool changed = false;
  ASSERT_EQ(SplitMixedLaunchMemory().Run(graph.get(), changed), ge::GRAPH_SUCCESS);
  EXPECT_TRUE(changed);
  auto calc_nodes = ge::ExecuteGraphUtils::FindNodesByTypeFromAllNodes(graph.get(), "CalcDeviceCopySizes");
  ASSERT_EQ(calc_nodes.size(), 1UL);
  EXPECT_NE(
      calc_nodes[0]->GetInDataEdgeByIndex(static_cast<int32_t>(kernel::CalcDeviceCopySizesInputs::kOriginalTensorSize)),
      nullptr);

  auto launch_copy_nodes = ge::ExecuteGraphUtils::FindNodesByTypeFromAllNodes(graph.get(), "LaunchH2DCopy");
  auto share_nodes = ge::ExecuteGraphUtils::FindNodesByTypeFromAllNodes(graph.get(), "ShareH2DCopyResult");
  ASSERT_EQ(launch_copy_nodes.size(), 1UL);
  ASSERT_EQ(share_nodes.size(), 1UL);
  EXPECT_EQ(launch_copy_nodes[0]->GetDataOutNum(), 0U);
  EXPECT_EQ(FastNodeTopoChecker(launch_copy_nodes[0]).OutChecker().CtrlToByType("ExecuteOpLaunch").Result(), "success");
  EXPECT_NE(FastNodeTopoChecker(launch_copy_nodes[0]).OutChecker().CtrlToByType("BuildRefTensor").Result(), "success");
  EXPECT_NE(FastNodeTopoChecker(launch_copy_nodes[0]).OutChecker().CtrlToByType("ExecuteOpPrepare").Result(),
            "success");
  EXPECT_EQ(FastNodeTopoChecker(share_nodes[0]).OutChecker().DataToByType("BuildRefTensor").Result(), "success");

  changed = false;
  ASSERT_EQ(RemoveLaunchFreeEdge().Run(graph.get(), changed), ge::GRAPH_SUCCESS);
  EXPECT_TRUE(changed);
  auto free_nodes = ge::ExecuteGraphUtils::FindNodesByTypeFromAllNodes(graph.get(), "FreeMemory");
  EXPECT_EQ(free_nodes.size(), 0UL);
  auto free_hold_addr_nodes = ge::ExecuteGraphUtils::FindNodesByTypeFromAllNodes(graph.get(), "FreeMemoryHoldAddr");
  ASSERT_EQ(free_hold_addr_nodes.size(), 1UL);
  EXPECT_NE(FastNodeTopoChecker(free_hold_addr_nodes[0]).InChecker().CtrlFromByType("ExecuteOpLaunch").Result(),
            "success");
}

TEST_F(SplitMixedLaunchMemoryUT, OfflineOptimizerAclnnPrepareLaunchFinalTopology) {
  SetRt2MultiThreadEnabled();
  std::unique_ptr<GraphFrame> init_frame;
  std::unique_ptr<GraphFrame> de_init_frame;
  Create3StageFrames(init_frame, de_init_frame);
  ASSERT_NE(init_frame, nullptr);
  auto init_netoutput_desc = ge::MakeShared<ge::OpDesc>("init_netoutput", "InnerNetOutput");
  ASSERT_NE(init_netoutput_desc, nullptr);
  ASSERT_NE(init_frame->GetExecuteGraph()->AddNode(init_netoutput_desc), nullptr);

  auto compute_graph = std::make_shared<ge::ComputeGraph>("aclnn_priority_compute_graph");
  auto compute_op_desc = ge::MakeShared<ge::OpDesc>("aclnn_priority_compute", "Compute");
  ASSERT_NE(compute_op_desc, nullptr);
  const auto compute_node = compute_graph->AddNode(compute_op_desc);
  ASSERT_NE(compute_node, nullptr);
  ValueHolder::SetCurrentComputeNode(compute_node);

  auto main_graph = BuildAclnnPrepareCopyGraph();
  ASSERT_NE(main_graph, nullptr);
  const auto source_copy_nodes =
      ge::ExecuteGraphUtils::FindNodesByTypeFromAllNodes(main_graph.get(), kernel::kMakeSureTensorAtDevice);
  ASSERT_EQ(source_copy_nodes.size(), 1UL);
  int64_t source_compute_index = -1;
  ASSERT_TRUE(ge::AttrUtils::GetInt(source_copy_nodes[0]->GetOpDescBarePtr(), kComputeNodeIndex, source_compute_index));
  EXPECT_EQ(source_compute_index, 0);
  auto output_data_desc = ge::MakeShared<ge::OpDesc>("output_data", "OutputData");
  ASSERT_NE(output_data_desc, nullptr);
  ASSERT_NE(main_graph->AddNode(output_data_desc), nullptr);
  auto root_frame = ValueHolder::PopGraphFrame();
  ASSERT_NE(root_frame, nullptr);
  auto root_graph = root_frame->GetExecuteGraph();
  ASSERT_NE(root_graph, nullptr);

  LoweringOption option;
  LoweringGlobalData global_data;
  ASSERT_EQ(OfflineOptimizer(option, global_data).Run(root_graph.get()), ge::GRAPH_SUCCESS);

  const auto launch_copy_nodes = ge::ExecuteGraphUtils::FindNodesByTypeFromAllNodes(root_graph.get(), "LaunchH2DCopy");
  const auto share_nodes =
      ge::ExecuteGraphUtils::FindNodesByTypeFromAllNodes(root_graph.get(), kernel::kShareH2DCopyResult);
  const auto prepare_nodes = ge::ExecuteGraphUtils::FindNodesByTypeFromAllNodes(root_graph.get(), "ExecuteOpPrepare");
  const auto launch_nodes = ge::ExecuteGraphUtils::FindNodesByTypeFromAllNodes(root_graph.get(), "ExecuteOpLaunch");
  const auto free_nodes = ge::ExecuteGraphUtils::FindNodesByTypeFromAllNodes(root_graph.get(), "FreeMemory");
  const auto free_hold_addr_nodes =
      ge::ExecuteGraphUtils::FindNodesByTypeFromAllNodes(root_graph.get(), "FreeMemoryHoldAddr");
  ASSERT_EQ(launch_copy_nodes.size(), 1UL);
  ASSERT_EQ(share_nodes.size(), 1UL);
  ASSERT_EQ(prepare_nodes.size(), 1UL);
  ASSERT_EQ(launch_nodes.size(), 1UL);
  EXPECT_EQ(free_nodes.size(), 0UL);
  ASSERT_EQ(free_hold_addr_nodes.size(), 1UL);

  EXPECT_FALSE(IsLaunchNode(launch_copy_nodes[0]->GetTypePtr()));
  EXPECT_TRUE(IsLaunchOrHasSubGraphNode(launch_copy_nodes[0]));
  EXPECT_EQ(launch_copy_nodes[0]->GetDataOutNum(), 0U);
  int64_t launch_compute_index = -1;
  ASSERT_TRUE(ge::AttrUtils::GetInt(launch_copy_nodes[0]->GetOpDescBarePtr(), kComputeNodeIndex, launch_compute_index));
  EXPECT_EQ(launch_compute_index, source_compute_index);
  EXPECT_EQ(FastNodeTopoChecker(share_nodes[0]).OutChecker().DataToByType("BuildRefTensor").Result(), "success");
  EXPECT_NE(FastNodeTopoChecker(launch_copy_nodes[0]).OutChecker().CtrlToByType("BuildRefTensor").Result(), "success");
  EXPECT_NE(FastNodeTopoChecker(launch_copy_nodes[0]).OutChecker().CtrlToByType("ExecuteOpPrepare").Result(),
            "success");
  EXPECT_EQ(FastNodeTopoChecker(launch_copy_nodes[0]).OutChecker().CtrlToByType("ExecuteOpLaunch").Result(), "success");
  EXPECT_NE(FastNodeTopoChecker(launch_nodes[0]).OutChecker().CtrlToByType("FreeMemoryHoldAddr").Result(), "success");
  EXPECT_EQ(FastNodeTopoChecker(free_hold_addr_nodes[0]).InChecker().DataFromByType("ShareH2DCopyResult").Result(),
            "success");
  EXPECT_NE(FastNodeTopoChecker(free_hold_addr_nodes[0]).InChecker().CtrlFromByType("ExecuteOpLaunch").Result(),
            "success");

  const auto relations = main_graph->GetExtAttr<FreeLaunchRelations>(kFreeLaunchRelationsAttr);
  ASSERT_NE(relations, nullptr);
  ASSERT_EQ(relations->size(), 1U);
  EXPECT_EQ((*relations)[0], std::make_pair(free_hold_addr_nodes[0], launch_nodes[0]));

  ASSERT_EQ(root_graph->TopologicalSorting(), ge::GRAPH_SUCCESS);
  const auto root_graph_nodes = root_graph->GetAllNodes();
  const auto main_graph_nodes = main_graph->GetAllNodes();
  ASSERT_EQ(NodePriorityCalculator(*root_frame).CalcNodeExecutionPriorities(main_graph_nodes, root_graph_nodes.size()),
            ge::GRAPH_SUCCESS);
  GraphNode graph_node;
  ASSERT_EQ(graph_node.EnsureNodeExeInOrder(root_graph.get()), ge::GRAPH_SUCCESS);
  EXPECT_TRUE(IsAcyclicAfterStrictOrder(main_graph.get(), graph_node));
}

TEST_F(SplitMixedLaunchMemoryUT, OfflineOptimizerInternalCopyHelpersDoNotAdvanceDumpRange) {
  SetRt2MultiThreadEnabled();
  std::unique_ptr<GraphFrame> init_frame;
  std::unique_ptr<GraphFrame> de_init_frame;
  Create3StageFrames(init_frame, de_init_frame);
  ASSERT_NE(init_frame, nullptr);
  auto init_netoutput_desc = ge::MakeShared<ge::OpDesc>("init_netoutput", "InnerNetOutput");
  ASSERT_NE(init_netoutput_desc, nullptr);
  ASSERT_NE(init_frame->GetExecuteGraph()->AddNode(init_netoutput_desc), nullptr);

  auto compute_graph = std::make_shared<ge::ComputeGraph>("dump_range_compute_graph");
  auto compute_op_desc = ge::MakeShared<ge::OpDesc>("dump_range_compute", "Compute");
  ASSERT_NE(compute_op_desc, nullptr);
  const auto compute_node = compute_graph->AddNode(compute_op_desc);
  ASSERT_NE(compute_node, nullptr);
  ValueHolder::SetCurrentComputeNode(compute_node);

  auto main_graph = BuildAclnnPrepareCopyGraph(true);
  ASSERT_NE(main_graph, nullptr);
  auto output_data_desc = ge::MakeShared<ge::OpDesc>("output_data", "OutputData");
  ASSERT_NE(output_data_desc, nullptr);
  ASSERT_NE(main_graph->AddNode(output_data_desc), nullptr);
  auto root_frame = ValueHolder::PopGraphFrame();
  ASSERT_NE(root_frame, nullptr);
  auto root_graph = root_frame->GetExecuteGraph();
  ASSERT_NE(root_graph, nullptr);

  LoweringOption option;
  LoweringGlobalData global_data;
  ASSERT_EQ(OfflineOptimizer(option, global_data).Run(root_graph.get()), ge::GRAPH_SUCCESS);

  const auto h2d_launch_nodes = ge::ExecuteGraphUtils::FindNodesByTypeFromAllNodes(root_graph.get(), "LaunchH2DCopy");
  const auto copy_flow_launch_nodes =
      ge::ExecuteGraphUtils::FindNodesByTypeFromAllNodes(root_graph.get(), kernel::kLaunchCopyFlowH2D);
  const auto execute_launch_nodes =
      ge::ExecuteGraphUtils::FindNodesByTypeFromAllNodes(root_graph.get(), "ExecuteOpLaunch");
  ASSERT_EQ(h2d_launch_nodes.size(), 1U);
  ASSERT_EQ(copy_flow_launch_nodes.size(), 1U);
  ASSERT_EQ(execute_launch_nodes.size(), 2U);
  EXPECT_EQ(h2d_launch_nodes[0]->GetDataOutNum(), 0U);
  EXPECT_EQ(copy_flow_launch_nodes[0]->GetDataOutNum(), 0U);
  EXPECT_FALSE(IsLaunchNode(h2d_launch_nodes[0]->GetTypePtr()));
  EXPECT_FALSE(IsLaunchNode(copy_flow_launch_nodes[0]->GetTypePtr()));

  int64_t compute_index = -1;
  ASSERT_TRUE(ge::AttrUtils::GetInt(h2d_launch_nodes[0]->GetOpDescBarePtr(), kComputeNodeIndex, compute_index));
  for (const auto node : {copy_flow_launch_nodes[0], execute_launch_nodes[0], execute_launch_nodes[1]}) {
    int64_t node_compute_index = -1;
    ASSERT_TRUE(ge::AttrUtils::GetInt(node->GetOpDescBarePtr(), kComputeNodeIndex, node_compute_index));
    EXPECT_EQ(node_compute_index, compute_index);
  }

  ASSERT_EQ(root_graph->TopologicalSorting(), ge::GRAPH_SUCCESS);
  ASSERT_EQ(AssignUniqueFeedIndexes(main_graph), ge::GRAPH_SUCCESS);
  const auto main_parent = main_graph->GetParentNodeBarePtr();
  ASSERT_NE(main_parent, nullptr);
  ASSERT_EQ(main_parent->GetOpDescBarePtr()->AddOutputDesc(ge::GeTensorDesc()), ge::GRAPH_SUCCESS);
  main_parent->GetExtendInfo()->UpdateOutputSymbols(1U);
  GertRuntimeStub runtime_stub;
  runtime_stub.GetKernelStub().AllKernelRegisteredAndSuccess();
  auto model_level_data = ExeGraphModelLevelDataFaker(root_graph).Build();
  auto runtime_compute_info =
      const_cast<ComputeNodeInfo *>(model_level_data.GetComputeNodeInf()->Get<ComputeNodeInfo>(0U));
  ASSERT_NE(runtime_compute_info, nullptr);
  runtime_compute_info->SetNodeName(compute_op_desc->GetNamePtr());
  runtime_compute_info->SetNodeType(compute_op_desc->GetTypePtr());
  GraphExecutorBuilder executor_builder(model_level_data.GetModelLevelData(), main_graph,
                                        &model_level_data.symbols_to_value);
  ExeGraphExecutor main_executor;
  ASSERT_EQ(executor_builder.Build(main_executor), ge::GRAPH_SUCCESS);
  auto execution_data = reinterpret_cast<ExecutionData *>(const_cast<void *>(main_executor.GetExecutionData()));
  ASSERT_NE(execution_data, nullptr);

  const auto runtime_h2d_launches = ModelV2ExecutorTestHelper::GetNodesByKernelType(execution_data, "LaunchH2DCopy");
  const auto runtime_copy_flow_launches =
      ModelV2ExecutorTestHelper::GetNodesByKernelType(execution_data, kernel::kLaunchCopyFlowH2D);
  const auto runtime_execute_launches =
      ModelV2ExecutorTestHelper::GetNodesByKernelType(execution_data, "ExecuteOpLaunch");
  ASSERT_EQ(runtime_h2d_launches.size(), 1U);
  ASSERT_EQ(runtime_copy_flow_launches.size(), 1U);
  ASSERT_EQ(runtime_execute_launches.size(), 2U);
  EXPECT_EQ(runtime_h2d_launches[0]->context.output_size, 0U);
  EXPECT_EQ(runtime_copy_flow_launches[0]->context.output_size, 0U);
  for (const auto node : {runtime_h2d_launches[0], runtime_copy_flow_launches[0], runtime_execute_launches[0],
                          runtime_execute_launches[1]}) {
    EXPECT_EQ(node->context.compute_node_info, runtime_compute_info);
  }

  auto node_iter =
      std::find(execution_data->base_ed.nodes, execution_data->base_ed.nodes + execution_data->base_ed.node_num,
                runtime_copy_flow_launches[0]);
  ASSERT_NE(node_iter, execution_data->base_ed.nodes + execution_data->base_ed.node_num);
  std::rotate(node_iter, node_iter + 1, execution_data->base_ed.nodes + execution_data->base_ed.node_num);

  GlobalDumper::GetInstance()->SetEnableFlags(0U);
  ModelV2Executor model_executor;
  ModelV2ExecutorTestHelper::SetExecutionData(execution_data, kMainExeGraph, &model_executor);
  ge::ModelData model_data{};
  const std::string model_name = "dump_range_model";
  const auto extend_info = ge::MakeShared<const SubscriberExtendInfo>(
      &model_executor, root_graph, compute_graph, model_data, nullptr, model_level_data.symbols_to_value, 0U,
      model_name, nullptr, std::unordered_map<std::string, TraceAttr>{});
  ASSERT_NE(extend_info, nullptr);
  ExecutorDumper dumper(extend_info);
  ASSERT_EQ(dumper.CollectLaunchKernelName(), ge::SUCCESS);
  const auto mapped_launch = dumper.compute_node_name_to_launch_kernel_name_.find(compute_op_desc->GetName());
  ASSERT_NE(mapped_launch, dumper.compute_node_name_to_launch_kernel_name_.cend());
  std::unordered_set<std::string> execute_launch_names;
  for (const auto node : runtime_execute_launches) {
    const auto kernel_extend_info = static_cast<const KernelExtendInfo *>(node->context.kernel_extend_info);
    ASSERT_NE(kernel_extend_info, nullptr);
    execute_launch_names.emplace(kernel_extend_info->GetKernelName());
  }
  EXPECT_EQ(execute_launch_names.count(mapped_launch->second), 1U);

  struct DumpPropertiesCleaner {
    ~DumpPropertiesCleaner() {
      ge::DumpManager::GetInstance().RemoveDumpProperties(ge::kInferSessionId);
      GlobalDumper::GetInstance()->SetEnableFlags(0U);
    }
  } dump_properties_cleaner;
  ge::DumpManager::GetInstance().RemoveDumpProperties(ge::kInferSessionId);
  ge::DumpProperties dump_properties;
  dump_properties.SetOpDumpRange(model_name, {{compute_op_desc->GetName(), compute_op_desc->GetName()}});
  ge::DumpManager::GetInstance().AddDumpProperties(ge::kInferSessionId, dump_properties);
  dumper.session_id_ = ge::kInferSessionId;
  ASSERT_EQ(dumper.ResetDumpFsmState(), ge::SUCCESS);
  ASSERT_EQ(dumper.dump_fsm_state_.size(), 1U);
  EXPECT_EQ(dumper.dump_fsm_state_[0], ge::DumpProcState::kInit);

  EXPECT_EQ(dumper.DataDump(runtime_h2d_launches[0], kExecuteStart), ge::SUCCESS);
  EXPECT_EQ(dumper.dump_fsm_state_[0], ge::DumpProcState::kInit);
  EXPECT_TRUE(dumper.dump_op_in_range_.empty());
  EXPECT_EQ(dumper.DataDump(runtime_copy_flow_launches[0], kExecuteStart), ge::SUCCESS);
  EXPECT_EQ(dumper.dump_fsm_state_[0], ge::DumpProcState::kInit);
  EXPECT_TRUE(dumper.dump_op_in_range_.empty());
  EXPECT_EQ(dumper.DataDump(runtime_execute_launches[0], kExecuteStart), ge::SUCCESS);
  EXPECT_EQ(dumper.dump_fsm_state_[0], ge::DumpProcState::kStop);
  EXPECT_EQ(dumper.dump_op_in_range_, std::unordered_set<std::string>({compute_op_desc->GetName()}));

  ge::DumpManager::GetInstance().RemoveDumpProperties(ge::kInferSessionId);
  const std::string ordinary_compute_name = "ordinary_compute";
  ge::DumpProperties ordinary_dump_properties;
  ordinary_dump_properties.SetOpDumpRange(model_name, {{ordinary_compute_name, ordinary_compute_name}});
  ge::DumpManager::GetInstance().AddDumpProperties(ge::kInferSessionId, ordinary_dump_properties);
  ASSERT_EQ(dumper.ResetDumpFsmState(), ge::SUCCESS);
  auto ordinary_context = KernelRunContextFaker()
                              .NodeName(ordinary_compute_name)
                              .NodeType("Compute")
                              .KernelName("ordinary_launch")
                              .KernelType("LaunchKernelWithHandle")
                              .KernelIONum(0U, 0U)
                              .Build();
  Node ordinary_launch{};
  memcpy(&ordinary_launch.context, ordinary_context.context, sizeof(KernelRunContext));
  EXPECT_EQ(dumper.DataDump(&ordinary_launch, kExecuteStart), ge::SUCCESS);
  EXPECT_EQ(dumper.dump_fsm_state_[0], ge::DumpProcState::kStop);
  EXPECT_EQ(dumper.dump_op_in_range_, std::unordered_set<std::string>({ordinary_compute_name}));
}

TEST_F(SplitMixedLaunchMemoryUT, OfflineOptimizerKeepsNestedAclnnRelationOnOwnerAndMapsItToMainCsr) {
  SetRt2MultiThreadEnabled();
  std::unique_ptr<GraphFrame> init_frame;
  std::unique_ptr<GraphFrame> de_init_frame;
  Create3StageFrames(init_frame, de_init_frame);
  ASSERT_NE(init_frame, nullptr);
  auto init_netoutput_desc = ge::MakeShared<ge::OpDesc>("init_netoutput", "InnerNetOutput");
  ASSERT_NE(init_netoutput_desc, nullptr);
  ASSERT_NE(init_frame->GetExecuteGraph()->AddNode(init_netoutput_desc), nullptr);

  auto main_graph = BuildMainGraphWithAclnnPrepareCopySubgraph();
  ASSERT_NE(main_graph, nullptr);
  auto call_node = ge::ExecuteGraphUtils::FindFirstNodeMatchType(main_graph.get(), "PartitionedCall");
  ASSERT_NE(call_node, nullptr);
  auto nested_graph = ge::FastNodeUtils::GetSubgraphFromNode(call_node, 0U);
  ASSERT_NE(nested_graph, nullptr);
  auto output_data_desc = ge::MakeShared<ge::OpDesc>("output_data", "OutputData");
  ASSERT_NE(output_data_desc, nullptr);
  ASSERT_NE(main_graph->AddNode(output_data_desc), nullptr);
  auto root_frame = ValueHolder::PopGraphFrame();
  ASSERT_NE(root_frame, nullptr);
  auto root_graph = root_frame->GetExecuteGraph();
  ASSERT_NE(root_graph, nullptr);

  LoweringOption option;
  LoweringGlobalData global_data;
  ASSERT_EQ(OfflineOptimizer(option, global_data).Run(root_graph.get()), ge::GRAPH_SUCCESS);

  const auto launch_copy_nodes = ge::ExecuteGraphUtils::FindNodesByTypeFromAllNodes(nested_graph, "LaunchH2DCopy");
  const auto share_nodes =
      ge::ExecuteGraphUtils::FindNodesByTypeFromAllNodes(nested_graph, kernel::kShareH2DCopyResult);
  const auto prepare_nodes = ge::ExecuteGraphUtils::FindNodesByTypeFromAllNodes(nested_graph, "ExecuteOpPrepare");
  const auto launch_nodes = ge::ExecuteGraphUtils::FindNodesByTypeFromAllNodes(nested_graph, "ExecuteOpLaunch");
  const auto free_hold_addr_nodes =
      ge::ExecuteGraphUtils::FindNodesByTypeFromAllNodes(nested_graph, "FreeMemoryHoldAddr");
  ASSERT_EQ(launch_copy_nodes.size(), 1UL);
  ASSERT_EQ(share_nodes.size(), 1UL);
  ASSERT_EQ(prepare_nodes.size(), 1UL);
  ASSERT_EQ(launch_nodes.size(), 1UL);
  ASSERT_EQ(free_hold_addr_nodes.size(), 1UL);
  EXPECT_EQ(launch_copy_nodes[0]->GetDataOutNum(), 0U);
  EXPECT_EQ(FastNodeTopoChecker(share_nodes[0]).OutChecker().DataToByType("BuildRefTensor").Result(), "success");
  EXPECT_NE(FastNodeTopoChecker(launch_copy_nodes[0]).OutChecker().CtrlToByType("ExecuteOpPrepare").Result(),
            "success");
  EXPECT_EQ(FastNodeTopoChecker(launch_copy_nodes[0]).OutChecker().CtrlToByType("ExecuteOpLaunch").Result(), "success");
  EXPECT_EQ(free_hold_addr_nodes[0]->GetExtendInfo()->GetOwnerGraphBarePtr(), nested_graph);
  EXPECT_EQ(launch_nodes[0]->GetExtendInfo()->GetOwnerGraphBarePtr(), nested_graph);

  EXPECT_EQ(root_graph->GetExtAttr<FreeLaunchRelations>(kFreeLaunchRelationsAttr), nullptr);
  EXPECT_EQ(main_graph->GetExtAttr<FreeLaunchRelations>(kFreeLaunchRelationsAttr), nullptr);
  const auto relations = nested_graph->GetExtAttr<FreeLaunchRelations>(kFreeLaunchRelationsAttr);
  ASSERT_NE(relations, nullptr);
  ASSERT_EQ(relations->size(), 1U);
  EXPECT_EQ((*relations)[0], std::make_pair(free_hold_addr_nodes[0], launch_nodes[0]));

  ASSERT_EQ(root_graph->TopologicalSorting(), ge::GRAPH_SUCCESS);
  GertRuntimeStub runtime_stub;
  runtime_stub.GetKernelStub().SetUp("PartitionedCall", BuildOnlyKernel);
  auto model_level_data = ExeGraphModelLevelDataFaker(root_graph).Build();
  GraphExecutorBuilder executor_builder(model_level_data.GetModelLevelData(), main_graph,
                                        &model_level_data.symbols_to_value);
  auto executor_option = MultiThreadExecutorOption(kLeastThreadNumber);
  executor_builder.ExecutorOpt(executor_option);
  MultiThreadExecutionDataBuilder execution_data_builder(executor_builder);
  auto resource_guard = execution_data_builder.Build();
  ASSERT_NE(resource_guard, nullptr);
  const auto &csr = static_cast<MultiThreadResourceGuard *>(resource_guard.get())->GetFreeLaunchRelationCsr();
  ASSERT_EQ(csr.relation_num, 1U);
  ASSERT_NE(csr.offsets, nullptr);
  ASSERT_NE(csr.launch_ids, nullptr);
  size_t relation_range_count = 0U;
  for (NodeIdentity node_id = 0U; node_id < csr.node_num; ++node_id) {
    const auto range = csr.GetLaunchIds(node_id);
    if (range.size == 0U) {
      continue;
    }
    ++relation_range_count;
    ASSERT_EQ(range.size, 1U);
    EXPECT_LT(range.data[0], csr.node_num);
  }
  EXPECT_EQ(relation_range_count, 1U);
}

TEST_F(SplitMixedLaunchMemoryUT, OfflineOptimizerRunsZeroCopyBeforeHoldAddressFreeOptimization) {
  SetRt2MultiThreadEnabled();
  std::unique_ptr<GraphFrame> init_frame;
  std::unique_ptr<GraphFrame> de_init_frame;
  Create3StageFrames(init_frame, de_init_frame);
  ASSERT_NE(init_frame, nullptr);
  auto init_netoutput_desc = ge::MakeShared<ge::OpDesc>("init_netoutput", "InnerNetOutput");
  ASSERT_NE(init_netoutput_desc, nullptr);
  ASSERT_NE(init_frame->GetExecuteGraph()->AddNode(init_netoutput_desc), nullptr);

  auto main_graph = BuildZeroCopyAndH2DCopyGraph();
  ASSERT_NE(main_graph, nullptr);
  auto root_frame = ValueHolder::PopGraphFrame();
  ASSERT_NE(root_frame, nullptr);
  auto root_graph = root_frame->GetExecuteGraph();
  ASSERT_NE(root_graph, nullptr);

  LoweringOption option;
  LoweringGlobalData global_data;
  ASSERT_EQ(OfflineOptimizer(option, global_data).Run(root_graph.get()), ge::GRAPH_SUCCESS);

  EXPECT_EQ(ge::ExecuteGraphUtils::FindNodesByTypeFromAllNodes(root_graph.get(), "AllocMemory").size(), 0UL);
  const auto alloc_nodes = ge::ExecuteGraphUtils::FindNodesByTypeFromAllNodes(root_graph.get(), "AllocModelOutTensor");
  const auto consumer_launch_nodes =
      ge::ExecuteGraphUtils::FindNodesByTypeFromAllNodes(root_graph.get(), "ExecuteOpLaunch");
  const auto h2d_launch_nodes = ge::ExecuteGraphUtils::FindNodesByTypeFromAllNodes(root_graph.get(), "LaunchH2DCopy");
  const auto share_nodes =
      ge::ExecuteGraphUtils::FindNodesByTypeFromAllNodes(root_graph.get(), kernel::kShareH2DCopyResult);
  const auto copy_d2d_nodes = ge::ExecuteGraphUtils::FindNodesByTypeFromAllNodes(root_graph.get(), "CopyD2D");
  const auto ensure_nodes =
      ge::ExecuteGraphUtils::FindNodesByTypeFromAllNodes(root_graph.get(), "EnsureTensorAtOutMemory");
  const auto free_hold_addr_nodes =
      ge::ExecuteGraphUtils::FindNodesByTypeFromAllNodes(root_graph.get(), "FreeMemoryHoldAddr");
  ASSERT_EQ(alloc_nodes.size(), 1UL);
  ASSERT_EQ(consumer_launch_nodes.size(), 2UL);
  ASSERT_EQ(h2d_launch_nodes.size(), 1UL);
  ASSERT_EQ(share_nodes.size(), 1UL);
  ASSERT_EQ(copy_d2d_nodes.size(), 1UL);
  ASSERT_EQ(ensure_nodes.size(), 1UL);
  ASSERT_EQ(free_hold_addr_nodes.size(), 2UL);
  EXPECT_EQ(h2d_launch_nodes[0]->GetDataOutNum(), 0U);

  EXPECT_EQ(FastNodeTopoChecker(copy_d2d_nodes[0]).InChecker().DataFromByType("AllocModelOutTensor").Result(),
            "success");
  EXPECT_EQ(FastNodeTopoChecker(ensure_nodes[0]).InChecker().DataFromByType("AllocModelOutTensor").Result(), "success");

  ge::FastNode *h2d_consumer = nullptr;
  ge::FastNode *zero_copy_consumer = nullptr;
  for (const auto consumer_launch : consumer_launch_nodes) {
    if (HasControlEdge(h2d_launch_nodes[0], consumer_launch)) {
      h2d_consumer = consumer_launch;
    } else if (FastNodeTopoChecker(consumer_launch).InChecker().DataFromByType("CopyD2D").Result() == "success") {
      zero_copy_consumer = consumer_launch;
    }
  }
  ASSERT_NE(h2d_consumer, nullptr);
  ASSERT_NE(zero_copy_consumer, nullptr);
  EXPECT_EQ(FastNodeTopoChecker(h2d_consumer).InChecker().DataFromByType(kernel::kShareH2DCopyResult).Result(),
            "success");
  EXPECT_FALSE(HasControlEdge(h2d_launch_nodes[0], copy_d2d_nodes[0]));

  ge::FastNode *h2d_free = nullptr;
  ge::FastNode *zero_copy_free = nullptr;
  for (const auto free_hold_addr_node : free_hold_addr_nodes) {
    const auto input_edge = free_hold_addr_node->GetInDataEdgeByIndex(0);
    ASSERT_NE(input_edge, nullptr);
    if (input_edge->src == share_nodes[0]) {
      h2d_free = free_hold_addr_node;
    } else if (input_edge->src == alloc_nodes[0]) {
      zero_copy_free = free_hold_addr_node;
    }
  }
  ASSERT_NE(h2d_free, nullptr);
  ASSERT_NE(zero_copy_free, nullptr);
  EXPECT_FALSE(HasControlEdge(h2d_launch_nodes[0], h2d_free));

  int64_t remove_launch_free_edge_alloc = 0;
  ASSERT_TRUE(ge::AttrUtils::GetInt(alloc_nodes[0]->GetOpDescBarePtr(), "remove_launch_free_edge_alloc",
                                    remove_launch_free_edge_alloc));
  EXPECT_EQ(remove_launch_free_edge_alloc, 1);
  const auto relations = main_graph->GetExtAttr<FreeLaunchRelations>(kFreeLaunchRelationsAttr);
  ASSERT_NE(relations, nullptr);
  ASSERT_GE(relations->size(), 2U);
  EXPECT_NE(std::find(relations->cbegin(), relations->cend(), std::make_pair(h2d_free, h2d_consumer)),
            relations->cend());
  EXPECT_NE(std::find(relations->cbegin(), relations->cend(), std::make_pair(zero_copy_free, zero_copy_consumer)),
            relations->cend());

  ASSERT_EQ(root_graph->TopologicalSorting(), ge::GRAPH_SUCCESS);
  const auto root_graph_nodes = root_graph->GetAllNodes();
  const auto main_graph_nodes = main_graph->GetAllNodes();
  ASSERT_EQ(NodePriorityCalculator(*root_frame).CalcNodeExecutionPriorities(main_graph_nodes, root_graph_nodes.size()),
            ge::GRAPH_SUCCESS);
  GraphNode graph_node;
  ASSERT_EQ(graph_node.EnsureNodeExeInOrder(root_graph.get()), ge::GRAPH_SUCCESS);
  const auto &h2d_additional_edges = graph_node.additional_add_info[h2d_consumer];
  EXPECT_NE(std::find(h2d_additional_edges.cbegin(), h2d_additional_edges.cend(), copy_d2d_nodes[0]),
            h2d_additional_edges.cend());
  EXPECT_GT(graph_node.additional_indegree_info[copy_d2d_nodes[0]], 0U);
  EXPECT_TRUE(IsAcyclicAfterStrictOrder(main_graph.get(), graph_node));
}
}  // namespace bg
}  // namespace gert
