/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "split_mixed_launch_memory.h"

#include <limits>
#include <map>
#include "common/checker.h"
#include "exe_graph/lowering/exe_graph_attrs.h"
#include "graph/utils/fast_node_utils.h"
#include "graph/utils/execute_graph_utils.h"
#include "graph/utils/graph_dump_utils.h"
#include "kernel/common_kernel_impl/memory_copy.h"
#include "kernel/common_kernel_impl/copy_flow_launch.h"
#include "kernel/memory/memory_kernel.h"
#include "common/compile_profiling/ge_call_wrapper.h"
#include "core/builder/node_types.h"
#include "core/executor/multi_thread_topological/executor/schedule/producer/producers/kernel_tags/critical_section_config.h"
#include "register/kernel_registry.h"

namespace gert {
namespace bg {
namespace {
constexpr const char *kBuildRefTensor = "BuildRefTensor";
constexpr const char *kFreeMemory = "FreeMemory";

bool IsLaunchConsumer(const ge::FastNode *const node) {
  if (gert::IsLaunchNode(node->GetTypePtr())) {
    return true;
  }
  const auto kernel_info = KernelRegistry::GetInstance().FindKernelInfo(node->GetTypePtr());
  return (kernel_info != nullptr) && (kernel_info->critical_section == kKernelLaunch);
}

bool IsRefTensorForLaunch(const ge::FastNode *const node) {
  if (strcmp(node->GetTypePtr(), kBuildRefTensor) != 0) {
    return false;
  }
  for (const auto out_node : node->GetOutDataNodes()) {
    if ((out_node != nullptr) && IsLaunchConsumer(out_node)) {
      return true;
    }
  }
  return false;
}

bool HasControlEdge(const ge::FastNode *const src_node, const ge::FastNode *const dst_node) {
  for (const auto edge : src_node->GetAllOutControlEdges()) {
    if ((edge != nullptr) && (edge->dst == dst_node)) {
      return true;
    }
  }
  return false;
}

ge::graphStatus AddControlEdgeIfAbsent(ge::ExecuteGraph *const graph, ge::FastNode *const src_node,
                                       ge::FastNode *const dst_node) {
  if (!HasControlEdge(src_node, dst_node)) {
    GE_ASSERT_NOTNULL(graph->AddEdge(src_node, ge::kControlEdgeIndex, dst_node, ge::kControlEdgeIndex));
  }
  return ge::GRAPH_SUCCESS;
}

struct SplitCopyFlowLaunchNodes {
  ge::FastNode *calc_alloc_sizes_node;
  ge::FastNode *alloc_batch_node;
  ge::FastNode *prepare_result_node;
  ge::FastNode *launch_copy_node;
};

bool IsLegacyCopyFlowLaunchNode(const ge::FastNode *const node) {
  return node->GetType() == kernel::kCopyFlowLaunch;
}

ge::graphStatus AddDynamicInputDescFromCopyFlow(const ge::OpDescPtr &op_desc,
                                                const ge::FastNode *const copy_flow_node) {
  const auto copy_flow_op_desc = copy_flow_node->GetOpDescBarePtr();
  GE_ASSERT_NOTNULL(copy_flow_op_desc);
  for (uint32_t i = static_cast<uint32_t>(kernel::CopyFlowLaunchInputs::kAddrAndLengthStart);
       i < copy_flow_node->GetDataInNum(); ++i) {
    GE_ASSERT_SUCCESS(op_desc->AddInputDesc(copy_flow_op_desc->GetInputDesc(i)));
  }
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus AddCalcAllocSizesInputDescFromCopyFlow(const ge::OpDescPtr &op_desc,
                                                       const ge::FastNode *const copy_flow_node) {
  const auto copy_flow_op_desc = copy_flow_node->GetOpDescBarePtr();
  GE_ASSERT_NOTNULL(copy_flow_op_desc);
  GE_ASSERT_SUCCESS(op_desc->AddInputDesc(
      copy_flow_op_desc->GetInputDesc(static_cast<uint32_t>(kernel::CopyFlowLaunchInputs::kInputsNum))));
  GE_ASSERT_SUCCESS(op_desc->AddInputDesc(
      copy_flow_op_desc->GetInputDesc(static_cast<uint32_t>(kernel::CopyFlowLaunchInputs::kRtArg))));
  GE_ASSERT_SUCCESS(AddDynamicInputDescFromCopyFlow(op_desc, copy_flow_node));
  GE_ASSERT_SUCCESS(op_desc->AddOutputDesc(ge::GeTensorDesc()));
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus AddAllocBatchInputDescFromCopyFlow(const ge::OpDescPtr &op_desc,
                                                   const ge::FastNode *const copy_flow_node) {
  const auto copy_flow_op_desc = copy_flow_node->GetOpDescBarePtr();
  GE_ASSERT_NOTNULL(copy_flow_op_desc);
  GE_ASSERT_SUCCESS(op_desc->AddInputDesc(
      copy_flow_op_desc->GetInputDesc(static_cast<uint32_t>(kernel::CopyFlowLaunchInputs::kAllocator))));
  GE_ASSERT_SUCCESS(op_desc->AddInputDesc(ge::GeTensorDesc()));
  GE_ASSERT_SUCCESS(op_desc->AddOutputDesc(ge::GeTensorDesc()));
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus AddPrepareCopyFlowResultInputDescFromCopyFlow(const ge::OpDescPtr &op_desc,
                                                              const ge::FastNode *const copy_flow_node) {
  const auto copy_flow_op_desc = copy_flow_node->GetOpDescBarePtr();
  GE_ASSERT_NOTNULL(copy_flow_op_desc);
  GE_ASSERT_SUCCESS(op_desc->AddInputDesc(
      copy_flow_op_desc->GetInputDesc(static_cast<uint32_t>(kernel::CopyFlowLaunchInputs::kInputsNum))));
  GE_ASSERT_SUCCESS(op_desc->AddInputDesc(
      copy_flow_op_desc->GetInputDesc(static_cast<uint32_t>(kernel::CopyFlowLaunchInputs::kInputsIndex))));
  GE_ASSERT_SUCCESS(op_desc->AddInputDesc(
      copy_flow_op_desc->GetInputDesc(static_cast<uint32_t>(kernel::CopyFlowLaunchInputs::kRtArg))));
  GE_ASSERT_SUCCESS(op_desc->AddInputDesc(ge::GeTensorDesc()));
  GE_ASSERT_SUCCESS(AddDynamicInputDescFromCopyFlow(op_desc, copy_flow_node));
  for (uint32_t i = 0U; i < copy_flow_node->GetDataOutNum(); ++i) {
    GE_ASSERT_SUCCESS(op_desc->AddOutputDesc(copy_flow_op_desc->GetOutputDesc(i)));
  }
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus AddLaunchCopyFlowH2DInputDescFromCopyFlow(const ge::OpDescPtr &op_desc,
                                                          const ge::FastNode *const copy_flow_node) {
  const auto copy_flow_op_desc = copy_flow_node->GetOpDescBarePtr();
  GE_ASSERT_NOTNULL(copy_flow_op_desc);
  GE_ASSERT_SUCCESS(op_desc->AddInputDesc(
      copy_flow_op_desc->GetInputDesc(static_cast<uint32_t>(kernel::CopyFlowLaunchInputs::kInputsNum))));
  GE_ASSERT_SUCCESS(op_desc->AddInputDesc(
      copy_flow_op_desc->GetInputDesc(static_cast<uint32_t>(kernel::CopyFlowLaunchInputs::kStream))));
  GE_ASSERT_SUCCESS(op_desc->AddInputDesc(ge::GeTensorDesc()));
  return AddDynamicInputDescFromCopyFlow(op_desc, copy_flow_node);
}

SplitCopyFlowLaunchNodes CreateSplitCopyFlowLaunchNodesFromCopyFlow(ge::ExecuteGraph *const graph,
                                                                    ge::FastNode *const copy_flow_node) {
  const auto copy_flow_count = static_cast<size_t>(copy_flow_node->GetDataOutNum());
  GE_ASSERT_TRUE((copy_flow_count > 0U) &&
                 (copy_flow_count <= static_cast<size_t>(std::numeric_limits<int64_t>::max())));
  const auto split_node_name = copy_flow_node->GetName();
  auto calc_op_desc = ge::MakeShared<ge::OpDesc>(split_node_name + "_CalcAllocSizes", kernel::kCalcCopyFlowAllocSizes);
  GE_ASSERT_NOTNULL(calc_op_desc);
  GE_ASSERT_TRUE(
      ge::AttrUtils::SetInt(calc_op_desc, kernel::kCopyFlowCountAttr, static_cast<int64_t>(copy_flow_count)));
  GE_ASSERT_SUCCESS(AddCalcAllocSizesInputDescFromCopyFlow(calc_op_desc, copy_flow_node));
  auto calc_node = graph->AddNode(calc_op_desc);
  GE_ASSERT_NOTNULL(calc_node);

  auto alloc_op_desc = ge::MakeShared<ge::OpDesc>(split_node_name + "_AllocCopyFlowHbm", kernel::kAllocCopyFlowHbm);
  GE_ASSERT_NOTNULL(alloc_op_desc);
  GE_ASSERT_TRUE(
      ge::AttrUtils::SetInt(alloc_op_desc, kernel::kCopyFlowCountAttr, static_cast<int64_t>(copy_flow_count)));
  GE_ASSERT_SUCCESS(AddAllocBatchInputDescFromCopyFlow(alloc_op_desc, copy_flow_node));
  auto alloc_node = graph->AddNode(alloc_op_desc);
  GE_ASSERT_NOTNULL(alloc_node);

  auto prepare_op_desc = ge::MakeShared<ge::OpDesc>(split_node_name, kernel::kPrepareCopyFlowResult);
  GE_ASSERT_NOTNULL(prepare_op_desc);
  GE_ASSERT_TRUE(
      ge::AttrUtils::SetInt(prepare_op_desc, kernel::kCopyFlowCountAttr, static_cast<int64_t>(copy_flow_count)));
  int64_t compute_node_index = 0;
  const bool has_compute_node_index =
      ge::AttrUtils::GetInt(copy_flow_node->GetOpDescBarePtr(), kComputeNodeIndex, compute_node_index);
  if (has_compute_node_index) {
    GE_ASSERT_TRUE(ge::AttrUtils::SetInt(prepare_op_desc, kComputeNodeIndex, compute_node_index));
  }
  GE_ASSERT_SUCCESS(AddPrepareCopyFlowResultInputDescFromCopyFlow(prepare_op_desc, copy_flow_node));
  auto prepare_node = graph->AddNode(prepare_op_desc);
  GE_ASSERT_NOTNULL(prepare_node);

  auto launch_op_desc = ge::MakeShared<ge::OpDesc>(split_node_name + "_LaunchCopyFlowH2D", kernel::kLaunchCopyFlowH2D);
  GE_ASSERT_NOTNULL(launch_op_desc);
  GE_ASSERT_TRUE(
      ge::AttrUtils::SetInt(launch_op_desc, kernel::kCopyFlowCountAttr, static_cast<int64_t>(copy_flow_count)));
  if (has_compute_node_index) {
    GE_ASSERT_TRUE(ge::AttrUtils::SetInt(launch_op_desc, kComputeNodeIndex, compute_node_index));
  }
  GE_ASSERT_SUCCESS(AddLaunchCopyFlowH2DInputDescFromCopyFlow(launch_op_desc, copy_flow_node));
  auto launch_node = graph->AddNode(launch_op_desc);
  GE_ASSERT_NOTNULL(launch_node);
  GE_ASSERT_NOTNULL(graph->AddEdge(prepare_node, ge::kControlEdgeIndex, launch_node, ge::kControlEdgeIndex));
  return {calc_node, alloc_node, prepare_node, launch_node};
}

ge::graphStatus AddSplitCopyFlowInputEdges(ge::FastNode *const copy_flow_node,
                                           const SplitCopyFlowLaunchNodes &split_nodes) {
  auto graph = copy_flow_node->GetExtendInfo()->GetOwnerGraphBarePtr();
  GE_ASSERT_NOTNULL(graph);
  const auto input_num_edge =
      copy_flow_node->GetInDataEdgeByIndex(static_cast<int32_t>(kernel::CopyFlowLaunchInputs::kInputsNum));
  const auto inputs_index_edge =
      copy_flow_node->GetInDataEdgeByIndex(static_cast<int32_t>(kernel::CopyFlowLaunchInputs::kInputsIndex));
  const auto rt_arg_edge =
      copy_flow_node->GetInDataEdgeByIndex(static_cast<int32_t>(kernel::CopyFlowLaunchInputs::kRtArg));
  const auto stream_edge =
      copy_flow_node->GetInDataEdgeByIndex(static_cast<int32_t>(kernel::CopyFlowLaunchInputs::kStream));
  const auto allocator_edge =
      copy_flow_node->GetInDataEdgeByIndex(static_cast<int32_t>(kernel::CopyFlowLaunchInputs::kAllocator));
  GE_ASSERT_NOTNULL(input_num_edge);
  GE_ASSERT_NOTNULL(inputs_index_edge);
  GE_ASSERT_NOTNULL(rt_arg_edge);
  GE_ASSERT_NOTNULL(stream_edge);
  GE_ASSERT_NOTNULL(allocator_edge);

  GE_ASSERT_NOTNULL(graph->AddEdge(input_num_edge->src, input_num_edge->src_output, split_nodes.calc_alloc_sizes_node,
                                   static_cast<int32_t>(kernel::CalcCopyFlowAllocSizesInputs::kInputsNum)));
  GE_ASSERT_NOTNULL(graph->AddEdge(rt_arg_edge->src, rt_arg_edge->src_output, split_nodes.calc_alloc_sizes_node,
                                   static_cast<int32_t>(kernel::CalcCopyFlowAllocSizesInputs::kRtArg)));
  int32_t calc_input_index = static_cast<int32_t>(kernel::CalcCopyFlowAllocSizesInputs::kAddrAndLengthStart);
  for (int32_t i = static_cast<int32_t>(kernel::CopyFlowLaunchInputs::kAddrAndLengthStart);
       i < static_cast<int32_t>(copy_flow_node->GetDataInNum()); ++i) {
    const auto in_edge = copy_flow_node->GetInDataEdgeByIndex(i);
    GE_ASSERT_NOTNULL(in_edge);
    GE_ASSERT_NOTNULL(
        graph->AddEdge(in_edge->src, in_edge->src_output, split_nodes.calc_alloc_sizes_node, calc_input_index++));
  }

  GE_ASSERT_NOTNULL(graph->AddEdge(allocator_edge->src, allocator_edge->src_output, split_nodes.alloc_batch_node, 0));
  GE_ASSERT_NOTNULL(graph->AddEdge(split_nodes.calc_alloc_sizes_node, 0, split_nodes.alloc_batch_node, 1));

  GE_ASSERT_NOTNULL(graph->AddEdge(input_num_edge->src, input_num_edge->src_output, split_nodes.prepare_result_node,
                                   static_cast<int32_t>(kernel::PrepareCopyFlowResultInputs::kInputsNum)));
  GE_ASSERT_NOTNULL(graph->AddEdge(inputs_index_edge->src, inputs_index_edge->src_output,
                                   split_nodes.prepare_result_node,
                                   static_cast<int32_t>(kernel::PrepareCopyFlowResultInputs::kInputsIndex)));
  GE_ASSERT_NOTNULL(graph->AddEdge(rt_arg_edge->src, rt_arg_edge->src_output, split_nodes.prepare_result_node,
                                   static_cast<int32_t>(kernel::PrepareCopyFlowResultInputs::kRtArg)));
  GE_ASSERT_NOTNULL(graph->AddEdge(split_nodes.alloc_batch_node, 0, split_nodes.prepare_result_node,
                                   static_cast<int32_t>(kernel::PrepareCopyFlowResultInputs::kAllocatedAddrs)));

  GE_ASSERT_NOTNULL(graph->AddEdge(input_num_edge->src, input_num_edge->src_output, split_nodes.launch_copy_node,
                                   static_cast<int32_t>(kernel::LaunchCopyFlowH2DInputs::kInputsNum)));
  GE_ASSERT_NOTNULL(graph->AddEdge(stream_edge->src, stream_edge->src_output, split_nodes.launch_copy_node,
                                   static_cast<int32_t>(kernel::LaunchCopyFlowH2DInputs::kStream)));
  GE_ASSERT_NOTNULL(graph->AddEdge(split_nodes.alloc_batch_node, 0, split_nodes.launch_copy_node,
                                   static_cast<int32_t>(kernel::LaunchCopyFlowH2DInputs::kAllocatedAddrs)));

  int32_t prepare_input_index = static_cast<int32_t>(kernel::PrepareCopyFlowResultInputs::kAddrAndLengthStart);
  int32_t launch_input_index = static_cast<int32_t>(kernel::LaunchCopyFlowH2DInputs::kAddrAndLengthStart);
  for (int32_t i = static_cast<int32_t>(kernel::CopyFlowLaunchInputs::kAddrAndLengthStart);
       i < static_cast<int32_t>(copy_flow_node->GetDataInNum()); ++i) {
    const auto in_edge = copy_flow_node->GetInDataEdgeByIndex(i);
    GE_ASSERT_NOTNULL(in_edge);
    GE_ASSERT_NOTNULL(
        graph->AddEdge(in_edge->src, in_edge->src_output, split_nodes.prepare_result_node, prepare_input_index++));
    GE_ASSERT_NOTNULL(
        graph->AddEdge(in_edge->src, in_edge->src_output, split_nodes.launch_copy_node, launch_input_index++));
  }

  GE_ASSERT_SUCCESS(ge::ExecuteGraphUtils::CopyInCtrlEdges(copy_flow_node, split_nodes.calc_alloc_sizes_node));
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoveCopyFlowOutputEdges(ge::FastNode *const copy_flow_node, ge::FastNode *const prepare_result_node,
                                        ge::FastNode *const launch_copy_node) {
  auto graph = copy_flow_node->GetExtendInfo()->GetOwnerGraphBarePtr();
  GE_ASSERT_NOTNULL(graph);
  std::vector<ge::Edge<ge::FastNode> *> output_edges;
  const auto &all_out_edges = copy_flow_node->GetAllOutDataEdgesRef();
  for (const auto &out_edges : all_out_edges) {
    for (const auto edge : out_edges) {
      if (edge != nullptr) {
        output_edges.emplace_back(edge);
      }
    }
  }
  for (const auto edge : output_edges) {
    const auto dst_endpoint = ge::FastNodeUtils::GetDstEndpoint(edge);
    const auto src_output = edge->src_output;
    GE_ASSERT_GRAPH_SUCCESS(graph->RemoveEdge(edge));
    GE_ASSERT_NOTNULL(graph->AddEdge(prepare_result_node, src_output, dst_endpoint.node, dst_endpoint.index));
    if (IsLaunchConsumer(dst_endpoint.node)) {
      GE_ASSERT_SUCCESS(AddControlEdgeIfAbsent(graph, launch_copy_node, dst_endpoint.node));
    }
  }
  GE_ASSERT_SUCCESS(ge::ExecuteGraphUtils::CopyOutCtrlEdges(copy_flow_node, launch_copy_node));
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus SplitLegacyCopyFlowLaunchNode(ge::FastNode *const copy_flow_node, bool &changed) {
  auto graph = copy_flow_node->GetExtendInfo()->GetOwnerGraphBarePtr();
  GE_ASSERT_NOTNULL(graph);
  const auto split_nodes = CreateSplitCopyFlowLaunchNodesFromCopyFlow(graph, copy_flow_node);
  GE_ASSERT_SUCCESS(AddSplitCopyFlowInputEdges(copy_flow_node, split_nodes));
  GE_ASSERT_SUCCESS(
      MoveCopyFlowOutputEdges(copy_flow_node, split_nodes.prepare_result_node, split_nodes.launch_copy_node));
  GE_ASSERT_SUCCESS(ge::ExecuteGraphUtils::IsolateNode(copy_flow_node, {}));
  GE_ASSERT_SUCCESS(ge::ExecuteGraphUtils::RemoveNodeWithoutRelink(graph, copy_flow_node));
  changed = true;
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus SplitLegacyCopyFlowLaunchNodes(ge::ExecuteGraph *const graph, bool &changed) {
  const auto copy_flow_nodes = graph->GetAllNodes(IsLegacyCopyFlowLaunchNode);
  for (const auto copy_flow_node : copy_flow_nodes) {
    GE_ASSERT_SUCCESS(SplitLegacyCopyFlowLaunchNode(copy_flow_node, changed));
  }
  return ge::GRAPH_SUCCESS;
}

bool IsMixedDeviceCopyNode(const ge::FastNode *const node) {
  return (node->GetType() == kernel::kCopyH2D) || (node->GetType() == kernel::kMakeSureTensorAtDevice);
}

ge::graphStatus AddCalcDeviceCopySizesInputDesc(const ge::OpDescPtr &op_desc, const ge::FastNode *const copy_node,
                                                size_t output_index) {
  const auto copy_op_desc = copy_node->GetOpDescBarePtr();
  GE_ASSERT_NOTNULL(copy_op_desc);
  const size_t addr_index = static_cast<size_t>(kernel::MakeSureTensorAtDeviceInputs::kAddrAndLengthStart) +
                            (output_index * kernel::kSizeOfCopyToDevice);
  GE_ASSERT_SUCCESS(op_desc->AddInputDesc(copy_op_desc->GetInputDesc(static_cast<uint32_t>(addr_index))));
  GE_ASSERT_SUCCESS(op_desc->AddInputDesc(
      copy_op_desc->GetInputDesc(static_cast<uint32_t>(kernel::MakeSureTensorAtDeviceInputs::kAllocator))));
  GE_ASSERT_SUCCESS(op_desc->AddInputDesc(copy_op_desc->GetInputDesc(static_cast<uint32_t>(addr_index + 3U))));
  GE_ASSERT_SUCCESS(op_desc->AddInputDesc(copy_op_desc->GetInputDesc(static_cast<uint32_t>(addr_index + 2U))));
  GE_ASSERT_SUCCESS(op_desc->AddInputDesc(
      copy_op_desc->GetInputDesc(static_cast<uint32_t>(kernel::MakeSureTensorAtDeviceInputs::kStream))));
  GE_ASSERT_SUCCESS(op_desc->AddInputDesc(copy_op_desc->GetInputDesc(static_cast<uint32_t>(addr_index + 1U))));
  GE_ASSERT_SUCCESS(op_desc->AddOutputDesc(ge::GeTensorDesc()));
  GE_ASSERT_SUCCESS(op_desc->AddOutputDesc(ge::GeTensorDesc()));
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus AddAllocMemHbmInputDesc(const ge::OpDescPtr &op_desc, const ge::FastNode *const copy_node) {
  const auto copy_op_desc = copy_node->GetOpDescBarePtr();
  GE_ASSERT_NOTNULL(copy_op_desc);
  GE_ASSERT_SUCCESS(op_desc->AddInputDesc(
      copy_op_desc->GetInputDesc(static_cast<uint32_t>(kernel::MakeSureTensorAtDeviceInputs::kAllocator))));
  GE_ASSERT_SUCCESS(op_desc->AddInputDesc(ge::GeTensorDesc()));
  GE_ASSERT_SUCCESS(op_desc->AddOutputDesc(ge::GeTensorDesc()));
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus AddH2DCopyInputDesc(const ge::OpDescPtr &op_desc, const ge::FastNode *const copy_node,
                                    size_t output_index) {
  const auto copy_op_desc = copy_node->GetOpDescBarePtr();
  GE_ASSERT_NOTNULL(copy_op_desc);
  const size_t addr_index = static_cast<size_t>(kernel::MakeSureTensorAtDeviceInputs::kAddrAndLengthStart) +
                            (output_index * kernel::kSizeOfCopyToDevice);
  GE_ASSERT_SUCCESS(op_desc->AddInputDesc(ge::GeTensorDesc()));
  GE_ASSERT_SUCCESS(op_desc->AddInputDesc(
      copy_op_desc->GetInputDesc(static_cast<uint32_t>(kernel::MakeSureTensorAtDeviceInputs::kStream))));
  GE_ASSERT_SUCCESS(op_desc->AddInputDesc(copy_op_desc->GetInputDesc(static_cast<uint32_t>(addr_index))));
  GE_ASSERT_SUCCESS(op_desc->AddInputDesc(ge::GeTensorDesc()));
  return ge::GRAPH_SUCCESS;
}

struct SplitDeviceCopyNodes {
  ge::FastNode *calc_copy_sizes_node;
  ge::FastNode *alloc_node;
  ge::FastNode *share_copy_result_node;
  ge::FastNode *launch_copy_node;
};

SplitDeviceCopyNodes CreateSplitDeviceCopyNodes(ge::ExecuteGraph *const graph, ge::FastNode *const copy_node,
                                                size_t output_index) {
  const auto split_node_name = copy_node->GetName() + "_Split_" + std::to_string(output_index);
  auto calc_op_desc = ge::MakeShared<ge::OpDesc>(split_node_name + "_CalcDeviceCopySizes", "CalcDeviceCopySizes");
  GE_ASSERT_NOTNULL(calc_op_desc);
  GE_ASSERT_SUCCESS(AddCalcDeviceCopySizesInputDesc(calc_op_desc, copy_node, output_index));
  auto calc_node = graph->AddNode(calc_op_desc);
  GE_ASSERT_NOTNULL(calc_node);

  auto alloc_op_desc = ge::MakeShared<ge::OpDesc>(split_node_name + "_AllocMemHbm", "AllocMemHbm");
  GE_ASSERT_NOTNULL(alloc_op_desc);
  GE_ASSERT_SUCCESS(AddAllocMemHbmInputDesc(alloc_op_desc, copy_node));
  auto alloc_node = graph->AddNode(alloc_op_desc);
  GE_ASSERT_NOTNULL(alloc_node);

  auto share_op_desc = ge::MakeShared<ge::OpDesc>(split_node_name + "_ShareH2DCopyResult", kernel::kShareH2DCopyResult);
  GE_ASSERT_NOTNULL(share_op_desc);
  GE_ASSERT_SUCCESS(AddH2DCopyInputDesc(share_op_desc, copy_node, output_index));
  GE_ASSERT_SUCCESS(
      share_op_desc->AddOutputDesc(copy_node->GetOpDescBarePtr()->GetOutputDesc(static_cast<uint32_t>(output_index))));
  auto share_node = graph->AddNode(share_op_desc);
  GE_ASSERT_NOTNULL(share_node);

  auto launch_op_desc = ge::MakeShared<ge::OpDesc>(split_node_name + "_LaunchH2DCopy", "LaunchH2DCopy");
  GE_ASSERT_NOTNULL(launch_op_desc);
  int64_t compute_node_index = 0;
  if (ge::AttrUtils::GetInt(copy_node->GetOpDescBarePtr(), kComputeNodeIndex, compute_node_index)) {
    GE_ASSERT_TRUE(ge::AttrUtils::SetInt(launch_op_desc, kComputeNodeIndex, compute_node_index));
  }
  GE_ASSERT_SUCCESS(AddH2DCopyInputDesc(launch_op_desc, copy_node, output_index));
  auto launch_node = graph->AddNode(launch_op_desc);
  GE_ASSERT_NOTNULL(launch_node);
  return {calc_node, alloc_node, share_node, launch_node};
}

ge::graphStatus AddH2DCopyInputEdges(ge::ExecuteGraph *const graph, const ge::Edge<ge::FastNode> *const stream_edge,
                                     const ge::Edge<ge::FastNode> *const addr_edge,
                                     const SplitDeviceCopyNodes &split_nodes, ge::FastNode *const node) {
  GE_ASSERT_NOTNULL(
      graph->AddEdge(split_nodes.alloc_node, 0, node, static_cast<int32_t>(kernel::LaunchH2DCopyInputs::kDstAddress)));
  GE_ASSERT_NOTNULL(graph->AddEdge(stream_edge->src, stream_edge->src_output, node,
                                   static_cast<int32_t>(kernel::LaunchH2DCopyInputs::kStream)));
  GE_ASSERT_NOTNULL(graph->AddEdge(addr_edge->src, addr_edge->src_output, node,
                                   static_cast<int32_t>(kernel::LaunchH2DCopyInputs::kSrcAddress)));
  GE_ASSERT_NOTNULL(graph->AddEdge(split_nodes.calc_copy_sizes_node,
                                   static_cast<int32_t>(kernel::CalcDeviceCopySizesOutputs::kCopySize), node,
                                   static_cast<int32_t>(kernel::LaunchH2DCopyInputs::kTensorSize)));
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus AddSplitDeviceCopyInputEdges(ge::FastNode *const copy_node, size_t output_index,
                                             const SplitDeviceCopyNodes &split_nodes) {
  auto graph = copy_node->GetExtendInfo()->GetOwnerGraphBarePtr();
  GE_ASSERT_NOTNULL(graph);
  const auto stream_edge =
      copy_node->GetInDataEdgeByIndex(static_cast<int32_t>(kernel::MakeSureTensorAtDeviceInputs::kStream));
  const auto allocator_edge =
      copy_node->GetInDataEdgeByIndex(static_cast<int32_t>(kernel::MakeSureTensorAtDeviceInputs::kAllocator));
  const int32_t addr_index = static_cast<int32_t>(kernel::MakeSureTensorAtDeviceInputs::kAddrAndLengthStart) +
                             static_cast<int32_t>(output_index * kernel::kSizeOfCopyToDevice);
  const auto addr_edge = copy_node->GetInDataEdgeByIndex(addr_index);
  const auto size_edge = copy_node->GetInDataEdgeByIndex(addr_index + 1);
  const auto shape_edge = copy_node->GetInDataEdgeByIndex(addr_index + 2);
  const auto data_type_edge = copy_node->GetInDataEdgeByIndex(addr_index + 3);
  GE_ASSERT_NOTNULL(stream_edge);
  GE_ASSERT_NOTNULL(allocator_edge);
  GE_ASSERT_NOTNULL(addr_edge);
  GE_ASSERT_NOTNULL(size_edge);
  GE_ASSERT_NOTNULL(shape_edge);
  GE_ASSERT_NOTNULL(data_type_edge);

  GE_ASSERT_NOTNULL(graph->AddEdge(addr_edge->src, addr_edge->src_output, split_nodes.calc_copy_sizes_node,
                                   static_cast<int32_t>(kernel::CalcDeviceCopySizesInputs::kSrcAddress)));
  GE_ASSERT_NOTNULL(graph->AddEdge(allocator_edge->src, allocator_edge->src_output, split_nodes.calc_copy_sizes_node,
                                   static_cast<int32_t>(kernel::CalcDeviceCopySizesInputs::kAllocator)));
  GE_ASSERT_NOTNULL(graph->AddEdge(data_type_edge->src, data_type_edge->src_output, split_nodes.calc_copy_sizes_node,
                                   static_cast<int32_t>(kernel::CalcDeviceCopySizesInputs::kDataType)));
  GE_ASSERT_NOTNULL(graph->AddEdge(shape_edge->src, shape_edge->src_output, split_nodes.calc_copy_sizes_node,
                                   static_cast<int32_t>(kernel::CalcDeviceCopySizesInputs::kStorageShape)));
  GE_ASSERT_NOTNULL(graph->AddEdge(stream_edge->src, stream_edge->src_output, split_nodes.calc_copy_sizes_node,
                                   static_cast<int32_t>(kernel::CalcDeviceCopySizesInputs::kStream)));
  GE_ASSERT_NOTNULL(graph->AddEdge(size_edge->src, size_edge->src_output, split_nodes.calc_copy_sizes_node,
                                   static_cast<int32_t>(kernel::CalcDeviceCopySizesInputs::kOriginalTensorSize)));

  GE_ASSERT_NOTNULL(graph->AddEdge(allocator_edge->src, allocator_edge->src_output, split_nodes.alloc_node, 0));
  GE_ASSERT_NOTNULL(graph->AddEdge(split_nodes.calc_copy_sizes_node,
                                   static_cast<int32_t>(kernel::CalcDeviceCopySizesOutputs::kAllocSize),
                                   split_nodes.alloc_node, 1));

  GE_ASSERT_SUCCESS(
      AddH2DCopyInputEdges(graph, stream_edge, addr_edge, split_nodes, split_nodes.share_copy_result_node));
  GE_ASSERT_SUCCESS(AddH2DCopyInputEdges(graph, stream_edge, addr_edge, split_nodes, split_nodes.launch_copy_node));
  GE_ASSERT_SUCCESS(AddControlEdgeIfAbsent(graph, split_nodes.share_copy_result_node, split_nodes.launch_copy_node));
  GE_ASSERT_SUCCESS(ge::ExecuteGraphUtils::CopyInCtrlEdges(copy_node, split_nodes.calc_copy_sizes_node));
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus AddRefTensorLaunchControlEdges(ge::ExecuteGraph *const graph, ge::FastNode *const launch_copy_node,
                                               const std::vector<ge::FastNode *> &build_ref_tensors,
                                               const std::vector<ge::FastNode *> &free_nodes) {
  std::map<std::string, ge::FastNode *> execute_nodes;
  for (const auto build_ref_tensor : build_ref_tensors) {
    for (const auto out_node : build_ref_tensor->GetOutDataNodes()) {
      if ((out_node != nullptr) && IsLaunchConsumer(out_node)) {
        execute_nodes.emplace(out_node->GetNamePtr(), out_node);
      }
    }
  }
  for (const auto &execute_node : execute_nodes) {
    GE_ASSERT_SUCCESS(AddControlEdgeIfAbsent(graph, launch_copy_node, execute_node.second));
    for (const auto free_node : free_nodes) {
      GE_ASSERT_SUCCESS(AddControlEdgeIfAbsent(graph, execute_node.second, free_node));
    }
  }
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoveDeviceCopyOutputEdges(ge::FastNode *const copy_node, size_t output_index,
                                          ge::FastNode *const share_copy_result_node,
                                          ge::FastNode *const launch_copy_node) {
  auto graph = copy_node->GetExtendInfo()->GetOwnerGraphBarePtr();
  GE_ASSERT_NOTNULL(graph);
  std::vector<ge::Edge<ge::FastNode> *> output_edges;
  std::vector<ge::FastNode *> build_ref_tensors;
  std::vector<ge::FastNode *> free_nodes;
  const auto &all_out_edges = copy_node->GetAllOutDataEdgesRef();
  if (output_index < all_out_edges.size()) {
    for (const auto edge : all_out_edges[output_index]) {
      if (edge != nullptr) {
        output_edges.emplace_back(edge);
        if (IsRefTensorForLaunch(edge->dst)) {
          build_ref_tensors.emplace_back(edge->dst);
        } else if (strcmp(edge->dst->GetTypePtr(), kFreeMemory) == 0) {
          free_nodes.emplace_back(edge->dst);
        }
      }
    }
  }
  for (const auto edge : output_edges) {
    const auto dst_endpoint = ge::FastNodeUtils::GetDstEndpoint(edge);
    GE_ASSERT_GRAPH_SUCCESS(graph->RemoveEdge(edge));
    GE_ASSERT_NOTNULL(graph->AddEdge(share_copy_result_node, 0, dst_endpoint.node, dst_endpoint.index));
    if (IsLaunchConsumer(dst_endpoint.node)) {
      GE_ASSERT_SUCCESS(AddControlEdgeIfAbsent(graph, launch_copy_node, dst_endpoint.node));
    }
  }
  GE_ASSERT_SUCCESS(AddRefTensorLaunchControlEdges(graph, launch_copy_node, build_ref_tensors, free_nodes));
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus SplitMixedDeviceCopyNode(ge::FastNode *const copy_node, bool &changed) {
  auto graph = copy_node->GetExtendInfo()->GetOwnerGraphBarePtr();
  GE_ASSERT_NOTNULL(graph);
  const size_t output_num = copy_node->GetDataOutNum();
  for (size_t i = 0U; i < output_num; ++i) {
    const auto split_nodes = CreateSplitDeviceCopyNodes(graph, copy_node, i);
    GE_ASSERT_SUCCESS(AddSplitDeviceCopyInputEdges(copy_node, i, split_nodes));
    GE_ASSERT_SUCCESS(
        MoveDeviceCopyOutputEdges(copy_node, i, split_nodes.share_copy_result_node, split_nodes.launch_copy_node));
    GE_ASSERT_SUCCESS(ge::ExecuteGraphUtils::CopyOutCtrlEdges(copy_node, split_nodes.launch_copy_node));
  }
  GE_ASSERT_SUCCESS(ge::ExecuteGraphUtils::IsolateNode(copy_node, {}));
  GE_ASSERT_SUCCESS(ge::ExecuteGraphUtils::RemoveNodeWithoutRelink(graph, copy_node));
  changed = changed || (output_num > 0U);
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus SplitRemainingMixedDeviceCopyNodes(ge::ExecuteGraph *const graph, bool &changed) {
  const auto copy_nodes = graph->GetAllNodes(IsMixedDeviceCopyNode);
  for (const auto copy_node : copy_nodes) {
    GE_ASSERT_SUCCESS(SplitMixedDeviceCopyNode(copy_node, changed));
  }
  return ge::GRAPH_SUCCESS;
}
}  // namespace

ge::graphStatus SplitMixedLaunchMemory::Run(ge::ExecuteGraph *const graph, bool &changed) {
  GE_TIMESTAMP_START(SplitMixedLaunchMemory);
  GE_ASSERT_SUCCESS(SplitLegacyCopyFlowLaunchNodes(graph, changed));
  GE_ASSERT_SUCCESS(SplitRemainingMixedDeviceCopyNodes(graph, changed));
  if (changed) {
    ge::DumpGraph(graph, "AfterSplitMixedLaunchMemory");
  }
  GE_TIMESTAMP_EVENT_END(SplitMixedLaunchMemory, "Pass::SplitMixedLaunchMemory");
  return ge::GRAPH_SUCCESS;
}
}  // namespace bg
}  // namespace gert
