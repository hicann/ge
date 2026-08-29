/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>

#include <cstdlib>
#include <limits>
#include <set>
#include <utility>

#include "macro_utils/dt_public_scope.h"
#include "framework/common/host_cpu_fusion_attr.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/ge_local_context.h"
#include "graph/partition/optimizer/host_cpu_fusion_pass.h"
#include "graph/utils/attr_utils.h"
#include "graph/utils/graph_utils.h"
#include "macro_utils/dt_public_unscope.h"

namespace ge {
namespace {
constexpr char kHostCpuEngineName[] = "DNN_VM_HOST_CPU";
constexpr char kHostCpuKernelLibName[] = "DNN_VM_HOST_CPU_OP_STORE";
constexpr char kHostCpuTaskKernelLibName[] = "HOSTCPUKernel";

bool SupportAllHostCpuOps(const std::string &) {
  return true;
}

NodePtr AddNode(const ComputeGraphPtr &graph, const std::string &name, const std::string &type,
                const size_t input_count, const size_t output_count) {
  auto op_desc = std::make_shared<OpDesc>(name, type);
  const GeTensorDesc tensor_desc(GeShape({2}), FORMAT_ND, DT_INT64);
  for (size_t i = 0U; i < input_count; ++i) {
    EXPECT_EQ(op_desc->AddInputDesc("input" + std::to_string(i), tensor_desc), GRAPH_SUCCESS);
  }
  for (size_t i = 0U; i < output_count; ++i) {
    EXPECT_EQ(op_desc->AddOutputDesc("output" + std::to_string(i), tensor_desc), GRAPH_SUCCESS);
  }
  return graph->AddNode(op_desc);
}

void AddEdge(const NodePtr &source, const size_t source_index, const NodePtr &target, const size_t target_index) {
  ASSERT_EQ(GraphUtils::AddEdge(source->GetOutDataAnchor(static_cast<int32_t>(source_index)),
                                target->GetInDataAnchor(static_cast<int32_t>(target_index))),
            GRAPH_SUCCESS);
}

void MarkCandidate(const NodePtr &node) {
  node->GetOpDesc()->SetOpEngineName(kHostCpuEngineName);
  node->GetOpDesc()->SetOpKernelLibName(kHostCpuKernelLibName);
  ASSERT_TRUE(AttrUtils::SetBool(node->GetOpDesc(), "SmallShapeHostcpu", true));
}

ComputeGraphPtr BuildNonConvergingGraph() {
  auto graph = std::make_shared<ComputeGraph>("host_cpu_branch");
  auto data = AddNode(graph, "data", "Data", 0U, 1U);
  auto a = AddNode(graph, "a", "HostA", 1U, 1U);
  auto b = AddNode(graph, "b", "HostB", 1U, 1U);
  auto c = AddNode(graph, "c", "HostC", 1U, 1U);
  auto d = AddNode(graph, "d", "HostD", 1U, 1U);
  auto e = AddNode(graph, "e", "HostE", 1U, 1U);
  auto output = AddNode(graph, "output", "NetOutput", 2U, 0U);
  AddEdge(data, 0U, a, 0U);
  AddEdge(a, 0U, b, 0U);
  AddEdge(a, 0U, c, 0U);
  AddEdge(b, 0U, d, 0U);
  AddEdge(c, 0U, e, 0U);
  AddEdge(d, 0U, output, 0U);
  AddEdge(e, 0U, output, 1U);
  for (const auto &node : {a, b, c, d, e}) {
    MarkCandidate(node);
  }
  EXPECT_TRUE(AttrUtils::SetInt(a->GetOpDesc(), "axis", 1));
  EXPECT_TRUE(AttrUtils::SetFloat(a->GetOpDesc(), "scale", 0.5F));
  EXPECT_TRUE(AttrUtils::SetBool(a->GetOpDesc(), "keep", true));
  EXPECT_TRUE(AttrUtils::SetStr(a->GetOpDesc(), "label", "a\\\"b\n"));
  EXPECT_TRUE(AttrUtils::SetListInt(a->GetOpDesc(), "sizes", {1, 2}));
  EXPECT_TRUE(AttrUtils::SetListFloat(a->GetOpDesc(), "ratios", {0.25F, 0.75F}));
  for (const auto &name : {"axis", "scale", "keep", "label", "sizes", "ratios"}) {
    a->GetOpDesc()->AppendIrAttrName(name);
  }
  return graph;
}

ComputeGraphPtr BuildConvergingGraph() {
  auto graph = std::make_shared<ComputeGraph>("host_cpu_converge");
  auto data = AddNode(graph, "data", "Data", 0U, 1U);
  auto a = AddNode(graph, "a", "HostA", 1U, 1U);
  auto b = AddNode(graph, "b", "HostB", 1U, 1U);
  auto c = AddNode(graph, "c", "HostC", 1U, 1U);
  auto d = AddNode(graph, "d", "HostD", 2U, 1U);
  auto output = AddNode(graph, "output", "NetOutput", 1U, 0U);
  AddEdge(data, 0U, a, 0U);
  AddEdge(a, 0U, b, 0U);
  AddEdge(a, 0U, c, 0U);
  AddEdge(b, 0U, d, 0U);
  AddEdge(c, 0U, d, 1U);
  AddEdge(d, 0U, output, 0U);
  for (const auto &node : {a, b, c, d}) {
    MarkCandidate(node);
  }
  return graph;
}

ComputeGraphPtr BuildPartiallyConvergingGraph() {
  auto graph = std::make_shared<ComputeGraph>("host_cpu_partial_converge");
  auto data = AddNode(graph, "data", "Data", 0U, 1U);
  auto a = AddNode(graph, "a", "HostA", 1U, 1U);
  auto b = AddNode(graph, "b", "HostB", 1U, 1U);
  auto c = AddNode(graph, "c", "HostC", 1U, 1U);
  auto d = AddNode(graph, "d", "HostD", 2U, 1U);
  auto e = AddNode(graph, "e", "HostE", 1U, 1U);
  auto f = AddNode(graph, "f", "HostF", 1U, 1U);
  auto output = AddNode(graph, "output", "NetOutput", 2U, 0U);
  AddEdge(data, 0U, a, 0U);
  AddEdge(a, 0U, b, 0U);
  AddEdge(a, 0U, c, 0U);
  AddEdge(a, 0U, e, 0U);
  AddEdge(b, 0U, d, 0U);
  AddEdge(c, 0U, d, 1U);
  AddEdge(e, 0U, f, 0U);
  AddEdge(d, 0U, output, 0U);
  AddEdge(f, 0U, output, 1U);
  for (const auto &node : {a, b, c, d, e, f}) {
    MarkCandidate(node);
  }
  return graph;
}

ComputeGraphPtr BuildSingleCandidateGraph() {
  auto graph = std::make_shared<ComputeGraph>("host_cpu_single");
  auto data = AddNode(graph, "data", "Data", 0U, 1U);
  auto host = AddNode(graph, "host", "HostA", 1U, 1U);
  auto output = AddNode(graph, "output", "NetOutput", 1U, 0U);
  AddEdge(data, 0U, host, 0U);
  AddEdge(host, 0U, output, 0U);
  MarkCandidate(host);
  return graph;
}

ComputeGraphPtr BuildMultiOutputFanoutGraph() {
  auto graph = std::make_shared<ComputeGraph>("host_cpu_multi_output");
  auto data = AddNode(graph, "data", "Data", 0U, 1U);
  auto first = AddNode(graph, "first", "HostA", 1U, 2U);
  auto second = AddNode(graph, "second", "HostB", 1U, 1U);
  auto device = AddNode(graph, "device", "DeviceOp", 1U, 1U);
  auto output = AddNode(graph, "output", "NetOutput", 2U, 0U);
  AddEdge(data, 0U, first, 0U);
  AddEdge(first, 0U, second, 0U);
  AddEdge(first, 1U, device, 0U);
  AddEdge(second, 0U, output, 0U);
  AddEdge(device, 0U, output, 1U);
  MarkCandidate(first);
  MarkCandidate(second);
  return graph;
}

ComputeGraphPtr BuildIndependentComponentsGraph() {
  auto graph = std::make_shared<ComputeGraph>("host_cpu_independent_components");
  auto data0 = AddNode(graph, "data0", "Data", 0U, 1U);
  auto data1 = AddNode(graph, "data1", "Data", 0U, 1U);
  auto a = AddNode(graph, "a", "HostA", 1U, 1U);
  auto b = AddNode(graph, "b", "HostB", 1U, 1U);
  auto c = AddNode(graph, "c", "HostC", 1U, 1U);
  auto d = AddNode(graph, "d", "HostD", 1U, 1U);
  auto output = AddNode(graph, "output", "NetOutput", 2U, 0U);
  AddEdge(data0, 0U, a, 0U);
  AddEdge(a, 0U, b, 0U);
  AddEdge(b, 0U, output, 0U);
  AddEdge(data1, 0U, c, 0U);
  AddEdge(c, 0U, d, 0U);
  AddEdge(d, 0U, output, 1U);
  for (const auto &node : {a, b, c, d}) {
    MarkCandidate(node);
  }
  return graph;
}

ComputeGraphPtr BuildLinearCandidateGraph(const size_t candidate_count) {
  auto graph = std::make_shared<ComputeGraph>("host_cpu_linear");
  auto previous = AddNode(graph, "data", "Data", 0U, 1U);
  for (size_t i = 0U; i < candidate_count; ++i) {
    auto current = AddNode(graph, "host_" + std::to_string(i), "HostA", 1U, 1U);
    AddEdge(previous, 0U, current, 0U);
    MarkCandidate(current);
    previous = current;
  }
  auto output = AddNode(graph, "output", "NetOutput", 1U, 0U);
  AddEdge(previous, 0U, output, 0U);
  return graph;
}

ComputeGraphPtr BuildGatherPackGraph() {
  auto graph = std::make_shared<ComputeGraph>("host_cpu_gather_pack");
  auto data = AddNode(graph, "data", "Data", 0U, 1U);
  auto indices = AddNode(graph, "indices", "Const", 0U, 1U);
  auto gather = AddNode(graph, "gather", "Gather", 2U, 1U);
  auto pack = AddNode(graph, "pack", "Pack", 1U, 1U);
  auto output = AddNode(graph, "output", "NetOutput", 1U, 0U);
  AddEdge(data, 0U, gather, 0U);
  AddEdge(indices, 0U, gather, 1U);
  AddEdge(gather, 0U, pack, 0U);
  AddEdge(pack, 0U, output, 0U);
  MarkCandidate(gather);
  MarkCandidate(pack);
  return graph;
}

class FakeCompiler : public HostCpuFusionCompiler {
 public:
  explicit FakeCompiler(const Status status, const size_t fail_on_call = std::numeric_limits<size_t>::max())
      : status_(status), fail_on_call_(fail_on_call) {}

  Status Compile(const std::string &source, std::vector<uint8_t> &so_data) const override {
    const size_t call_index = sources.size();
    sources.emplace_back(source);
    if (status_ != SUCCESS) {
      return status_;
    }
    if (call_index == fail_on_call_) {
      return FAILED;
    }
    so_data.assign(20U, 0U);
    so_data[0] = 0x7FU;
    so_data[1] = 'E';
    so_data[2] = 'L';
    so_data[3] = 'F';
    so_data[4] = 2U;
    so_data[5] = 1U;
    so_data[6] = 1U;
    so_data[16] = 3U;
    return SUCCESS;
  }

  mutable std::vector<std::string> sources;

 private:
  Status status_;
  size_t fail_on_call_;
};

class TransitionInvalidCompiler final : public HostCpuFusionCompiler {
 public:
  explicit TransitionInvalidCompiler(const ComputeGraphPtr &graph) : graph_(graph) {}

  Status Compile(const std::string &, std::vector<uint8_t> &so_data) const override {
    so_data.assign(20U, 0U);
    so_data[0] = 0x7FU;
    so_data[1] = 'E';
    so_data[2] = 'L';
    so_data[3] = 'F';
    so_data[4] = 2U;
    so_data[5] = 1U;
    so_data[6] = 1U;
    so_data[16] = 3U;
    if (!mutated_) {
      mutated_ = true;
      const auto source = graph_->FindNode("d");
      const auto target = graph_->FindNode("a");
      if ((source == nullptr) || (target == nullptr) ||
          (GraphUtils::AddEdge(source->GetOutControlAnchor(), target->GetInControlAnchor()) != GRAPH_SUCCESS)) {
        return FAILED;
      }
    }
    return SUCCESS;
  }

 private:
  ComputeGraphPtr graph_;
  mutable bool mutated_ = false;
};

class EdgeRemovingCompiler final : public HostCpuFusionCompiler {
 public:
  explicit EdgeRemovingCompiler(const ComputeGraphPtr &graph) : graph_(graph) {}

  Status Compile(const std::string &, std::vector<uint8_t> &so_data) const override {
    so_data.assign(20U, 0U);
    so_data[0] = 0x7FU;
    so_data[1] = 'E';
    so_data[2] = 'L';
    so_data[3] = 'F';
    so_data[4] = 2U;
    so_data[5] = 1U;
    so_data[6] = 1U;
    so_data[16] = 3U;
    if (!mutated_) {
      mutated_ = true;
      const auto source = graph_->FindNode("e");
      const auto output = graph_->FindNode("output");
      if ((source == nullptr) || (output == nullptr) ||
          (GraphUtils::RemoveEdge(source->GetOutDataAnchor(0), output->GetInDataAnchor(1)) != GRAPH_SUCCESS)) {
        return FAILED;
      }
    }
    return SUCCESS;
  }

 private:
  ComputeGraphPtr graph_;
  mutable bool mutated_ = false;
};

size_t CountNodesByType(const ComputeGraphPtr &graph, const std::string &type) {
  size_t count = 0U;
  for (const auto &node : graph->GetDirectNode()) {
    if (node->GetType() == type) {
      ++count;
    }
  }
  return count;
}
}  // namespace

TEST(HostCpuFusionPassTest, SplitsNonConvergingBranchesAndGeneratesAllSupportedAttrs) {
  const auto graph = BuildNonConvergingGraph();
  HostCpuFusionPass pass(std::make_shared<FakeCompiler>(SUCCESS), SupportAllHostCpuOps);
  std::vector<std::vector<HostCpuFusionRegion>> components;
  ASSERT_EQ(pass.BuildFusionRegions(graph, components), SUCCESS);
  ASSERT_EQ(components.size(), 1U);
  ASSERT_EQ(components[0].size(), 2U);
  EXPECT_EQ(components[0][0].nodes.size(), 3U);
  EXPECT_EQ(components[0][1].nodes.size(), 3U);
  EXPECT_EQ(components[0][0].external_outputs.size(), 1U);
  EXPECT_EQ(components[0][1].external_outputs.size(), 1U);

  HostCpuFusionCodegen codegen;
  HostCpuFusionCodegenResult result;
  ASSERT_EQ(codegen.Generate(components[0][0], result), SUCCESS);
  EXPECT_NE(result.source.find("SetAttr(\"axis\""), std::string::npos);
  EXPECT_NE(result.source.find("SetAttr(\"scale\""), std::string::npos);
  EXPECT_NE(result.source.find("SetAttr(\"keep\""), std::string::npos);
  EXPECT_NE(result.source.find("std::string(\"a\\\\\\\"b\\n\", 5U)"), std::string::npos);
  EXPECT_NE(result.source.find("std::vector<int64_t>{1LL, 2LL}"), std::string::npos);
  EXPECT_NE(result.source.find("std::vector<float>{0.25F, 0.75F}"), std::string::npos);
  EXPECT_NE(result.source.find("#include \"graph/operator.h\""), std::string::npos);
  EXPECT_NE(result.source.find("#include \"graph/tensor.h\""), std::string::npos);
  EXPECT_EQ(result.source.find("host_cpu_kernel_registry.h"), std::string::npos);
  EXPECT_NE(result.source.find("#include \"aicpu/cpu_kernels/cpu_kernel.h\""), std::string::npos);
  EXPECT_NE(result.source.find("REGISTER_CPU_KERNEL(kFusedHostCpuKernel_"), std::string::npos);
  EXPECT_NE(result.source.find("CreateCpuConstantFoldingFusedChainPlan"), std::string::npos);
  EXPECT_NE(result.source.find("RunCpuConstantFoldingFusedChainPlan"), std::string::npos);
  EXPECT_EQ(result.source.find("RunCpuConstantFoldingFusedPlan"), std::string::npos);
  EXPECT_EQ(result.source.find("dlsym"), std::string::npos);
  EXPECT_EQ(result.source.find("UpdateInputDesc"), std::string::npos);
  EXPECT_EQ(result.source.find("UpdateOutputDesc"), std::string::npos);
  EXPECT_NE(result.source.find("ValidateFusedHostCpuKernelRegistration"), std::string::npos);
  EXPECT_NE(result.source.find("void *CreateFusedHostCpuKernelState()"), std::string::npos);
  EXPECT_NE(result.source.find("uint32_t RunFusedHostCpuKernel(void *kernel_state, const void *binding_data"),
            std::string::npos);
  EXPECT_NE(result.source.find("FusedHostCpuTensorBinding"), std::string::npos);
  EXPECT_NE(result.source.find("ComputeBindings"), std::string::npos);
  EXPECT_NE(result.source.find("BuildFusedHostCpuRuntimeTensor"), std::string::npos);
  EXPECT_NE(result.source.find("InitializeBindings"), std::string::npos);
  EXPECT_NE(result.source.find("RunCpuConstantFoldingFusedChainPlanBindings"), std::string::npos);
  EXPECT_EQ(result.source.find("BindFusedHostCpuTensor"), std::string::npos);
  EXPECT_NE(result.source.find("FusedHostCpuOrchestration_"), std::string::npos);
  EXPECT_NE(result.source.find("CpuKernelContext &ctx"), std::string::npos);
  EXPECT_EQ(result.source.find("REGISTER_HOST_CPU_OP_BUILDER"), std::string::npos);
  EXPECT_EQ(result.source.find("CreateHostCpuOp"), std::string::npos);
  EXPECT_EQ(result.source.find("_Kernel_Creator"), std::string::npos);
  EXPECT_EQ(result.source.find("graph/op_desc.h"), std::string::npos);
  EXPECT_EQ(result.source.find("graph/utils/op_desc_utils.h"), std::string::npos);
  EXPECT_EQ(result.source.find("GeTensorDesc"), std::string::npos);
  EXPECT_EQ(result.source.find("OpDescUtils"), std::string::npos);
  EXPECT_EQ(result.source.find(".at("), std::string::npos);
  EXPECT_NE(result.source.find("if (state == nullptr) { return nullptr; }"), std::string::npos);
  EXPECT_EQ(result.source.find("state->Initialize()"), std::string::npos);
}

TEST(HostCpuFusionPassTest, KeepsConvergingDagInOneRegion) {
  const auto graph = BuildConvergingGraph();
  HostCpuFusionPass pass(std::make_shared<FakeCompiler>(SUCCESS), SupportAllHostCpuOps);
  std::vector<std::vector<HostCpuFusionRegion>> components;
  ASSERT_EQ(pass.BuildFusionRegions(graph, components), SUCCESS);
  ASSERT_EQ(components.size(), 1U);
  ASSERT_EQ(components[0].size(), 1U);
  EXPECT_EQ(components[0][0].nodes.size(), 4U);
}

TEST(HostCpuFusionPassTest, SupportsComponentWithMoreThanFifteenNodes) {
  const auto graph = BuildLinearCandidateGraph(16U);
  HostCpuFusionPass pass(std::make_shared<FakeCompiler>(SUCCESS), SupportAllHostCpuOps);
  std::vector<std::vector<HostCpuFusionRegion>> components;
  ASSERT_EQ(pass.BuildFusionRegions(graph, components), SUCCESS);
  ASSERT_EQ(components.size(), 1U);
  ASSERT_EQ(components[0].size(), 1U);
  EXPECT_EQ(components[0][0].nodes.size(), 16U);

  HostCpuFusionCodegenResult result;
  EXPECT_EQ(HostCpuFusionCodegen().Generate(components[0][0], result), SUCCESS);
  EXPECT_FALSE(result.source.empty());
}

TEST(HostCpuFusionPassTest, CloneAndSplitKeepsConvergedChildrenTogether) {
  const auto graph = BuildPartiallyConvergingGraph();
  HostCpuFusionPass pass(std::make_shared<FakeCompiler>(SUCCESS), SupportAllHostCpuOps);
  std::vector<std::vector<HostCpuFusionRegion>> components;
  ASSERT_EQ(pass.BuildFusionRegions(graph, components), SUCCESS);
  ASSERT_EQ(components.size(), 1U);
  ASSERT_EQ(components[0].size(), 2U);
  std::set<std::set<std::string>> actual_regions;
  for (const auto &region : components[0]) {
    std::set<std::string> node_names;
    for (const auto &node : region.nodes) {
      node_names.emplace(node->GetName());
    }
    actual_regions.emplace(std::move(node_names));
  }
  const std::set<std::set<std::string>> expected_regions = {{"a", "b", "c", "d"}, {"a", "e", "f"}};
  EXPECT_EQ(actual_regions, expected_regions);
  EXPECT_NE(graph->FindNode("a"), nullptr);
  EXPECT_EQ(CountNodesByType(graph, kFusedHostCpuOpType), 0U);
}

TEST(HostCpuFusionPassTest, RejectsNullAndEmptyGraphs) {
  HostCpuFusionPass pass(std::make_shared<FakeCompiler>(SUCCESS), SupportAllHostCpuOps);
  std::vector<std::vector<HostCpuFusionRegion>> components;
  EXPECT_EQ(pass.BuildFusionRegions(nullptr, components), PARAM_INVALID);
  EXPECT_EQ(pass.BuildFusionRegions(std::make_shared<ComputeGraph>("empty"), components), PARAM_INVALID);
}

TEST(HostCpuFusionPassTest, DoesNotFuseSingleCandidateNode) {
  const auto graph = BuildSingleCandidateGraph();
  auto compiler = std::make_shared<FakeCompiler>(SUCCESS);
  HostCpuFusionPass pass(compiler, SupportAllHostCpuOps);
  NodeEngineMap atomic_map;
  NodeEngineMap composite_map;
  EXPECT_EQ(pass.Run(graph, atomic_map, composite_map), NOT_CHANGED);
  EXPECT_NE(graph->FindNode("host"), nullptr);
  EXPECT_EQ(CountNodesByType(graph, kFusedHostCpuOpType), 0U);
  EXPECT_TRUE(compiler->sources.empty());
}

TEST(HostCpuFusionPassTest, UnsupportedInnerCpuKernelKeepsOriginalGraph) {
  const auto graph = BuildGatherPackGraph();
  auto compiler = std::make_shared<FakeCompiler>(SUCCESS);
  HostCpuFusionPass pass(compiler, [](const std::string &op_type) { return op_type != "Gather"; });
  NodeEngineMap atomic_map;
  NodeEngineMap composite_map;

  EXPECT_EQ(pass.Run(graph, atomic_map, composite_map), NOT_CHANGED);
  EXPECT_NE(graph->FindNode("gather"), nullptr);
  EXPECT_NE(graph->FindNode("pack"), nullptr);
  EXPECT_EQ(CountNodesByType(graph, kFusedHostCpuOpType), 0U);
  EXPECT_TRUE(compiler->sources.empty());
}

TEST(HostCpuFusionPassTest, ReconnectsExternalConsumerOfInternalMultiOutput) {
  const auto graph = BuildMultiOutputFanoutGraph();
  auto compiler = std::make_shared<FakeCompiler>(SUCCESS);
  HostCpuFusionPass pass(compiler, SupportAllHostCpuOps);
  NodeEngineMap atomic_map;
  NodeEngineMap composite_map;
  ASSERT_EQ(pass.Run(graph, atomic_map, composite_map), SUCCESS);
  ASSERT_EQ(CountNodesByType(graph, kFusedHostCpuOpType), 1U);
  const auto device = graph->FindNode("device");
  const auto output = graph->FindNode("output");
  ASSERT_NE(device, nullptr);
  ASSERT_NE(output, nullptr);
  const auto device_source = device->GetInDataAnchor(0)->GetPeerOutAnchor();
  const auto output_source = output->GetInDataAnchor(0)->GetPeerOutAnchor();
  ASSERT_NE(device_source, nullptr);
  ASSERT_NE(output_source, nullptr);
  EXPECT_EQ(device_source->GetOwnerNode()->GetType(), kFusedHostCpuOpType);
  EXPECT_EQ(output_source->GetOwnerNode()->GetType(), kFusedHostCpuOpType);
  EXPECT_EQ(device_source->GetOwnerNode(), output_source->GetOwnerNode());
  EXPECT_EQ(device_source->GetIdx(), 0);
  EXPECT_EQ(output_source->GetIdx(), 1);
  ASSERT_EQ(compiler->sources.size(), 1U);
  EXPECT_NE(compiler->sources[0].find("node_output_0_1 = &outputs_[0U]"), std::string::npos);
  EXPECT_NE(compiler->sources[0].find("node_output_1_0 = &outputs_[1U]"), std::string::npos);
}

TEST(HostCpuFusionPassTest, FusesIndependentComponentsSeparately) {
  const auto graph = BuildIndependentComponentsGraph();
  auto compiler = std::make_shared<FakeCompiler>(SUCCESS);
  HostCpuFusionPass pass(compiler, SupportAllHostCpuOps);
  NodeEngineMap atomic_map;
  NodeEngineMap composite_map;
  ASSERT_EQ(pass.Run(graph, atomic_map, composite_map), SUCCESS);
  EXPECT_EQ(CountNodesByType(graph, kFusedHostCpuOpType), 2U);
  EXPECT_EQ(compiler->sources.size(), 2U);
  EXPECT_EQ(atomic_map.size(), 2U);
  EXPECT_EQ(composite_map.size(), 2U);
  const auto first_source = graph->FindNode("output")->GetInDataAnchor(0)->GetPeerOutAnchor();
  const auto second_source = graph->FindNode("output")->GetInDataAnchor(1)->GetPeerOutAnchor();
  ASSERT_NE(first_source, nullptr);
  ASSERT_NE(second_source, nullptr);
  EXPECT_EQ(first_source->GetOwnerNode()->GetType(), kFusedHostCpuOpType);
  EXPECT_EQ(second_source->GetOwnerNode()->GetType(), kFusedHostCpuOpType);
  EXPECT_NE(first_source->GetOwnerNode(), second_source->GetOwnerNode());
}

TEST(HostCpuFusionPassTest, FusesCandidateChainInDynamicPartitionSubgraph) {
  const ComputeGraphPtr root_graph = std::make_shared<ComputeGraph>("host_cpu_root");
  const NodePtr root_data = AddNode(root_graph, "root_data", "Data", 0U, 1U);
  const NodePtr partitioned_call = AddNode(root_graph, "partitioned_call", "PartitionedCall", 1U, 1U);
  const NodePtr root_output = AddNode(root_graph, "root_output", "NetOutput", 1U, 0U);
  AddEdge(root_data, 0U, partitioned_call, 0U);
  AddEdge(partitioned_call, 0U, root_output, 0U);

  const ComputeGraphPtr subgraph = BuildConvergingGraph();
  partitioned_call->GetOpDesc()->RegisterSubgraphIrName("body", SubgraphType::kStatic);
  partitioned_call->GetOpDesc()->AddSubgraphName(subgraph->GetName());
  partitioned_call->GetOpDesc()->SetSubgraphInstanceName(0U, subgraph->GetName());
  subgraph->SetParentNode(partitioned_call);
  subgraph->SetParentGraph(root_graph);
  ASSERT_EQ(root_graph->AddSubgraph(subgraph), GRAPH_SUCCESS);

  NodeEngineMap atomic_map;
  NodeEngineMap composite_map;
  for (const char *const name : {"a", "b", "c", "d"}) {
    const NodePtr node = subgraph->FindNode(name);
    ASSERT_NE(node, nullptr);
    atomic_map[node] = kHostCpuEngineName;
    composite_map[node] = kHostCpuEngineName;
  }
  const std::shared_ptr<FakeCompiler> compiler = std::make_shared<FakeCompiler>(SUCCESS);
  HostCpuFusionPass pass(compiler, SupportAllHostCpuOps);

  ASSERT_EQ(pass.Run(root_graph, atomic_map, composite_map), SUCCESS);
  EXPECT_EQ(CountNodesByType(root_graph, kFusedHostCpuOpType), 0U);
  ASSERT_EQ(CountNodesByType(subgraph, kFusedHostCpuOpType), 1U);
  EXPECT_EQ(compiler->sources.size(), 1U);
  for (const char *const name : {"a", "b", "c", "d"}) {
    EXPECT_EQ(subgraph->FindNode(name), nullptr);
  }

  NodePtr fused_node;
  for (const NodePtr &node : subgraph->GetDirectNode()) {
    if (node->GetType() == kFusedHostCpuOpType) {
      fused_node = node;
      break;
    }
  }
  ASSERT_NE(fused_node, nullptr);
  ASSERT_EQ(atomic_map.size(), 1U);
  ASSERT_EQ(composite_map.size(), 1U);
  EXPECT_EQ(atomic_map.at(fused_node), kHostCpuEngineName);
  EXPECT_EQ(composite_map.at(fused_node), kHostCpuEngineName);
  EXPECT_EQ(subgraph->FindNode("output")->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode(), fused_node);

  std::string so_key;
  Buffer so_data;
  ASSERT_TRUE(AttrUtils::GetStr(fused_node->GetOpDesc(), kFusedHostCpuSoKey, so_key));
  EXPECT_TRUE(AttrUtils::GetBytes(root_graph, so_key, so_data));
  EXPECT_FALSE(AttrUtils::HasAttr(subgraph, so_key));
}

TEST(HostCpuFusionPassTest, SemanticChangesProduceDifferentChainIds) {
  const auto first_graph = BuildNonConvergingGraph();
  const auto second_graph = BuildNonConvergingGraph();
  ASSERT_TRUE(AttrUtils::SetInt(second_graph->FindNode("a")->GetOpDesc(), "axis", 2));
  HostCpuFusionPass pass(std::make_shared<FakeCompiler>(SUCCESS), SupportAllHostCpuOps);
  std::vector<std::vector<HostCpuFusionRegion>> first_components;
  std::vector<std::vector<HostCpuFusionRegion>> second_components;
  ASSERT_EQ(pass.BuildFusionRegions(first_graph, first_components), SUCCESS);
  ASSERT_EQ(pass.BuildFusionRegions(second_graph, second_components), SUCCESS);
  ASSERT_EQ(first_components.size(), 1U);
  ASSERT_EQ(second_components.size(), 1U);
  ASSERT_EQ(first_components[0].size(), second_components[0].size());
  EXPECT_NE(first_components[0][0].chain_id, second_components[0][0].chain_id);
}

TEST(HostCpuFusionPassTest, UnsupportedSemanticAttrSkipsCodeGeneration) {
  const auto graph = BuildConvergingGraph();
  const auto node = graph->FindNode("a");
  ASSERT_TRUE(AttrUtils::SetDataType(node->GetOpDesc(), "dtype", DT_INT64));
  node->GetOpDesc()->AppendIrAttrName("dtype");
  auto compiler = std::make_shared<FakeCompiler>(SUCCESS);
  HostCpuFusionPass pass(compiler, SupportAllHostCpuOps);
  std::vector<std::vector<HostCpuFusionRegion>> components;
  ASSERT_EQ(pass.BuildFusionRegions(graph, components), SUCCESS);
  HostCpuFusionCodegenResult result;
  EXPECT_EQ(HostCpuFusionCodegen().Generate(components[0][0], result), UNSUPPORTED);
  EXPECT_TRUE(result.register_name.empty());
  EXPECT_TRUE(result.source.empty());
  EXPECT_TRUE(result.so_data.empty());
  NodeEngineMap atomic_map;
  NodeEngineMap composite_map;
  EXPECT_EQ(pass.Run(graph, atomic_map, composite_map), NOT_CHANGED);
  EXPECT_TRUE(compiler->sources.empty());
  EXPECT_EQ(CountNodesByType(graph, kFusedHostCpuOpType), 0U);
  EXPECT_NE(graph->FindNode("a"), nullptr);
  EXPECT_NE(graph->FindNode("d"), nullptr);
}

TEST(HostCpuFusionPassTest, ReplacesOriginalComponentAndUpdatesEngineMaps) {
  const auto graph = BuildNonConvergingGraph();
  auto compiler = std::make_shared<FakeCompiler>(SUCCESS);
  HostCpuFusionPass pass(compiler, SupportAllHostCpuOps);
  NodeEngineMap atomic_map;
  NodeEngineMap composite_map;
  for (const auto &node : graph->GetDirectNode()) {
    if (node->GetOpDesc()->GetOpEngineName() == kHostCpuEngineName) {
      atomic_map[node] = kHostCpuEngineName;
      composite_map[node] = kHostCpuEngineName;
    }
  }
  ASSERT_EQ(pass.Run(graph, atomic_map, composite_map), SUCCESS);
  EXPECT_EQ(CountNodesByType(graph, kFusedHostCpuOpType), 2U);
  EXPECT_EQ(graph->FindNode("a"), nullptr);
  EXPECT_EQ(graph->FindNode("b"), nullptr);
  EXPECT_EQ(graph->FindNode("c"), nullptr);
  EXPECT_EQ(graph->FindNode("d"), nullptr);
  EXPECT_EQ(graph->FindNode("e"), nullptr);
  EXPECT_EQ(atomic_map.size(), 2U);
  EXPECT_EQ(composite_map.size(), 2U);
  EXPECT_EQ(compiler->sources.size(), 2U);
  const auto output = graph->FindNode("output");
  ASSERT_NE(output, nullptr);
  const auto source0 = output->GetInDataAnchor(0)->GetPeerOutAnchor();
  const auto source1 = output->GetInDataAnchor(1)->GetPeerOutAnchor();
  ASSERT_NE(source0, nullptr);
  ASSERT_NE(source1, nullptr);
  EXPECT_EQ(source0->GetOwnerNode()->GetType(), kFusedHostCpuOpType);
  EXPECT_EQ(source1->GetOwnerNode()->GetType(), kFusedHostCpuOpType);
  EXPECT_NE(source0->GetOwnerNode(), source1->GetOwnerNode());
  for (const auto &node : graph->GetDirectNode()) {
    if (node->GetType() != kFusedHostCpuOpType) {
      continue;
    }
    std::string so_key;
    std::string task_kernel_lib;
    int64_t unknown_shape_type = -1;
    Buffer so_data;
    ASSERT_TRUE(AttrUtils::GetStr(node->GetOpDesc(), kFusedHostCpuSoKey, so_key));
    ASSERT_TRUE(AttrUtils::GetStr(node->GetOpDesc(), "opKernelLib", task_kernel_lib));
    ASSERT_TRUE(AttrUtils::GetInt(node->GetOpDesc(), ATTR_NAME_UNKNOWN_SHAPE_TYPE, unknown_shape_type));
    ASSERT_TRUE(AttrUtils::GetBytes(graph, so_key, so_data));
    EXPECT_EQ(task_kernel_lib, kHostCpuTaskKernelLibName);
    EXPECT_EQ(unknown_shape_type, DEPEND_IN_SHAPE);
    EXPECT_EQ(node->GetOpDesc()->GetInputNameByIndex(0U), "input_0_output0");
    ASSERT_GE(so_data.GetSize(), 4U);
    EXPECT_EQ(so_data[0], 0x7FU);
  }
}

TEST(HostCpuFusionPassTest, CompileFailureDoesNotModifyGraphOrEngineMaps) {
  const auto graph = BuildNonConvergingGraph();
  HostCpuFusionPass pass(std::make_shared<FakeCompiler>(FAILED), SupportAllHostCpuOps);
  NodeEngineMap atomic_map;
  NodeEngineMap composite_map;
  for (const auto &node : graph->GetDirectNode()) {
    if (node->GetOpDesc()->GetOpEngineName() == kHostCpuEngineName) {
      atomic_map[node] = kHostCpuEngineName;
      composite_map[node] = kHostCpuEngineName;
    }
  }
  const size_t original_atomic_size = atomic_map.size();
  ASSERT_EQ(pass.Run(graph, atomic_map, composite_map), NOT_CHANGED);
  EXPECT_EQ(CountNodesByType(graph, kFusedHostCpuOpType), 0U);
  EXPECT_NE(graph->FindNode("a"), nullptr);
  EXPECT_NE(graph->FindNode("e"), nullptr);
  EXPECT_EQ(atomic_map.size(), original_atomic_size);
  EXPECT_EQ(composite_map.size(), original_atomic_size);
  EXPECT_EQ(graph->FindNode("output")->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode()->GetName(), "d");
  EXPECT_EQ(graph->FindNode("output")->GetInDataAnchor(1)->GetPeerOutAnchor()->GetOwnerNode()->GetName(), "e");
}

TEST(HostCpuFusionPassTest, LaterComponentCompileFailureKeepsEntireGraphUnchanged) {
  const auto graph = BuildIndependentComponentsGraph();
  auto compiler = std::make_shared<FakeCompiler>(SUCCESS, 1U);
  HostCpuFusionPass pass(compiler, SupportAllHostCpuOps);
  std::vector<std::vector<HostCpuFusionRegion>> components;
  ASSERT_EQ(pass.BuildFusionRegions(graph, components), SUCCESS);
  ASSERT_EQ(components.size(), 2U);
  for (const auto &regions : components) {
    for (const auto &region : regions) {
      EXPECT_FALSE(AttrUtils::HasAttr(graph, std::string(kFusedHostCpuSoDataPrefix) + region.chain_id));
    }
  }
  NodeEngineMap atomic_map;
  NodeEngineMap composite_map;
  for (const auto &node : graph->GetDirectNode()) {
    if (node->GetOpDesc()->GetOpEngineName() == kHostCpuEngineName) {
      atomic_map[node] = kHostCpuEngineName;
      composite_map[node] = kHostCpuEngineName;
    }
  }
  const size_t original_engine_count = atomic_map.size();

  EXPECT_EQ(pass.Run(graph, atomic_map, composite_map), NOT_CHANGED);
  EXPECT_EQ(compiler->sources.size(), 2U);
  EXPECT_EQ(CountNodesByType(graph, kFusedHostCpuOpType), 0U);
  for (const auto &name : {"a", "b", "c", "d"}) {
    EXPECT_NE(graph->FindNode(name), nullptr);
  }
  EXPECT_EQ(atomic_map.size(), original_engine_count);
  EXPECT_EQ(composite_map.size(), original_engine_count);
  EXPECT_EQ(graph->FindNode("output")->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode()->GetName(), "b");
  EXPECT_EQ(graph->FindNode("output")->GetInDataAnchor(1)->GetPeerOutAnchor()->GetOwnerNode()->GetName(), "d");
  for (const auto &regions : components) {
    for (const auto &region : regions) {
      EXPECT_FALSE(AttrUtils::HasAttr(graph, std::string(kFusedHostCpuSoDataPrefix) + region.chain_id));
    }
  }
}

TEST(HostCpuFusionPassTest, RejectsMalformedFusionCodegenRegions) {
  const auto graph = BuildConvergingGraph();
  HostCpuFusionPass pass(std::make_shared<FakeCompiler>(SUCCESS), SupportAllHostCpuOps);
  std::vector<std::vector<HostCpuFusionRegion>> components;
  ASSERT_EQ(pass.BuildFusionRegions(graph, components), SUCCESS);
  ASSERT_EQ(components.size(), 1U);
  ASSERT_EQ(components[0].size(), 1U);
  const HostCpuFusionRegion valid = components[0][0];
  HostCpuFusionCodegen codegen;
  HostCpuFusionCodegenResult result;

  auto invalid = valid;
  invalid.nodes.resize(1U);
  EXPECT_EQ(codegen.Generate(invalid, result), PARAM_INVALID);
  invalid = valid;
  invalid.chain_id.clear();
  EXPECT_EQ(codegen.Generate(invalid, result), PARAM_INVALID);
  invalid = valid;
  invalid.external_outputs.clear();
  EXPECT_EQ(codegen.Generate(invalid, result), PARAM_INVALID);

  invalid = valid;
  invalid.chain_id = "1starts_with_digit";
  EXPECT_EQ(codegen.Generate(invalid, result), PARAM_INVALID);
  invalid = valid;
  invalid.chain_id = "contains-dash";
  EXPECT_EQ(codegen.Generate(invalid, result), PARAM_INVALID);
  invalid = valid;
  invalid.chain_id.assign(200U, 'a');
  EXPECT_EQ(codegen.Generate(invalid, result), PARAM_INVALID);

  invalid = valid;
  invalid.nodes[0] = nullptr;
  EXPECT_EQ(codegen.Generate(invalid, result), PARAM_INVALID);
  invalid = valid;
  invalid.nodes.push_back(invalid.nodes.front());
  EXPECT_EQ(codegen.Generate(invalid, result), PARAM_INVALID);
  invalid = valid;
  invalid.external_inputs[0] = nullptr;
  EXPECT_EQ(codegen.Generate(invalid, result), PARAM_INVALID);
  invalid = valid;
  invalid.external_inputs.push_back(invalid.external_inputs.front());
  EXPECT_EQ(codegen.Generate(invalid, result), PARAM_INVALID);
  invalid = valid;
  invalid.external_outputs[0].source = nullptr;
  EXPECT_EQ(codegen.Generate(invalid, result), PARAM_INVALID);
  invalid = valid;
  invalid.external_outputs.push_back(invalid.external_outputs.front());
  EXPECT_EQ(codegen.Generate(invalid, result), PARAM_INVALID);

  auto internal_output_binding = valid;
  internal_output_binding.external_outputs.push_back({valid.nodes.front()->GetOutDataAnchor(0), {}});
  EXPECT_EQ(codegen.Generate(internal_output_binding, result), SUCCESS);

  invalid = valid;
  std::swap(invalid.nodes[0], invalid.nodes[1]);
  EXPECT_EQ(codegen.Generate(invalid, result), PARAM_INVALID);

  invalid = valid;
  invalid.external_inputs[0] = valid.nodes[0]->GetOutDataAnchor(0);
  EXPECT_EQ(codegen.Generate(invalid, result), PARAM_INVALID);
  invalid = valid;
  invalid.external_outputs[0].source = graph->FindNode("data")->GetOutDataAnchor(0);
  EXPECT_EQ(codegen.Generate(invalid, result), PARAM_INVALID);

  invalid = valid;
  invalid.external_inputs.clear();
  EXPECT_EQ(codegen.Generate(invalid, result), PARAM_INVALID);

  invalid = valid;
  GraphUtils::RemoveEdge(graph->FindNode("a")->GetInDataAnchor(0)->GetPeerOutAnchor(),
                         graph->FindNode("a")->GetInDataAnchor(0));
  EXPECT_EQ(codegen.Generate(invalid, result), UNSUPPORTED);
}

TEST(HostCpuFusionPassTest, RejectsInvalidGeneratedTensorAndAttributeMetadata) {
  const auto make_region = []() {
    const auto graph = BuildConvergingGraph();
    HostCpuFusionPass pass(std::make_shared<FakeCompiler>(SUCCESS), SupportAllHostCpuOps);
    std::vector<std::vector<HostCpuFusionRegion>> components;
    EXPECT_EQ(pass.BuildFusionRegions(graph, components), SUCCESS);
    EXPECT_EQ(components.size(), 1U);
    EXPECT_EQ(components[0].size(), 1U);
    return std::make_pair(graph, components[0][0]);
  };
  HostCpuFusionCodegenResult result;

  {
    const auto holder = make_region();
    auto invalid = holder.second;
    invalid.nodes[0]->GetOpDesc()->SetName(std::string("node\0name", 9U));
    EXPECT_EQ(HostCpuFusionCodegen().Generate(invalid, result), UNSUPPORTED);
  }
  {
    const auto holder = make_region();
    auto invalid = holder.second;
    invalid.nodes[1]->GetOpDesc()->MutableAllInputName().clear();
    EXPECT_EQ(HostCpuFusionCodegen().Generate(invalid, result), UNSUPPORTED);
  }
  {
    const auto holder = make_region();
    auto invalid = holder.second;
    invalid.nodes[1]->GetOpDesc()->MutableAllOutputName().clear();
    EXPECT_EQ(HostCpuFusionCodegen().Generate(invalid, result), UNSUPPORTED);
  }
  {
    const auto holder = make_region();
    auto invalid = holder.second;
    invalid.nodes[0]->GetOpDesc()->MutableOutputDesc(0)->SetShape(GeShape({-1}));
    EXPECT_EQ(HostCpuFusionCodegen().Generate(invalid, result), UNSUPPORTED);
  }
  {
    const auto holder = make_region();
    auto invalid = holder.second;
    invalid.nodes[0]->GetOpDesc()->AppendIrAttrName("declared_but_missing");
    EXPECT_EQ(HostCpuFusionCodegen().Generate(invalid, result), UNSUPPORTED);
  }
  {
    const auto holder = make_region();
    auto invalid = holder.second;
    invalid.nodes[0]->GetOpDesc()->MutableAllOutputName().clear();
    EXPECT_EQ(HostCpuFusionCodegen().Generate(invalid, result), UNSUPPORTED);
  }
  {
    const auto holder = make_region();
    auto invalid = holder.second;
    const GeTensorDesc desc(GeShape({2}), FORMAT_ND, DT_INT64);
    ASSERT_EQ(invalid.nodes[0]->GetOpDesc()->AddInputDesc("extra", desc), GRAPH_SUCCESS);
    EXPECT_EQ(HostCpuFusionCodegen().Generate(invalid, result), PARAM_INVALID);
  }
  {
    const auto holder = make_region();
    auto invalid = holder.second;
    const GeTensorDesc desc(GeShape({2}), FORMAT_ND, DT_INT64);
    ASSERT_EQ(invalid.nodes[0]->GetOpDesc()->AddOutputDesc("extra", desc), GRAPH_SUCCESS);
    EXPECT_EQ(HostCpuFusionCodegen().Generate(invalid, result), PARAM_INVALID);
  }
  {
    const auto holder = make_region();
    auto invalid = holder.second;
    invalid.nodes[0]->GetOpDesc()->AppendIrAttrName("");
    EXPECT_EQ(HostCpuFusionCodegen().Generate(invalid, result), UNSUPPORTED);
  }
  {
    const auto holder = make_region();
    auto invalid = holder.second;
    ASSERT_TRUE(
        AttrUtils::SetListFloat(invalid.nodes[0]->GetOpDesc(), "nan_list", {std::numeric_limits<float>::quiet_NaN()}));
    invalid.nodes[0]->GetOpDesc()->AppendIrAttrName("nan_list");
    EXPECT_EQ(HostCpuFusionCodegen().Generate(invalid, result), UNSUPPORTED);
  }
  {
    const auto holder = make_region();
    auto invalid = holder.second;
    ASSERT_TRUE(
        AttrUtils::SetFloat(invalid.nodes[0]->GetOpDesc(), "nan_attr", std::numeric_limits<float>::quiet_NaN()));
    invalid.nodes[0]->GetOpDesc()->AppendIrAttrName("nan_attr");
    EXPECT_EQ(HostCpuFusionCodegen().Generate(invalid, result), UNSUPPORTED);
  }
}

TEST(HostCpuFusionPassTest, RejectsHostCpuCandidateWhenOutputShapeIsUnknown) {
  const auto graph = BuildSingleCandidateGraph();
  const auto host = graph->FindNode("host");
  ASSERT_NE(host, nullptr);
  host->GetOpDesc()->MutableOutputDesc(0)->SetShape(GeShape({-1}));
  HostCpuFusionPass pass(std::make_shared<FakeCompiler>(SUCCESS), SupportAllHostCpuOps);
  std::vector<std::vector<HostCpuFusionRegion>> components;
  EXPECT_EQ(pass.BuildFusionRegions(graph, components), NOT_CHANGED);
  EXPECT_TRUE(components.empty());
}

TEST(HostCpuFusionPassTest, RejectsCandidatesWithUnsafeExecutionContracts) {
  const auto expect_not_candidate = [](const ComputeGraphPtr &graph) {
    HostCpuFusionPass pass(std::make_shared<FakeCompiler>(SUCCESS), SupportAllHostCpuOps);
    std::vector<std::vector<HostCpuFusionRegion>> components;
    EXPECT_EQ(pass.BuildFusionRegions(graph, components), NOT_CHANGED);
    EXPECT_TRUE(components.empty());
  };

  {
    auto graph = std::make_shared<ComputeGraph>("already_fused");
    auto fused = AddNode(graph, "fused", kFusedHostCpuOpType, 0U, 1U);
    auto output = AddNode(graph, "output", "NetOutput", 1U, 0U);
    AddEdge(fused, 0U, output, 0U);
    MarkCandidate(fused);
    expect_not_candidate(graph);
  }
  {
    const auto graph = BuildSingleCandidateGraph();
    ASSERT_EQ(GraphUtils::AddEdge(graph->FindNode("data")->GetOutControlAnchor(),
                                  graph->FindNode("host")->GetInControlAnchor()),
              GRAPH_SUCCESS);
    expect_not_candidate(graph);
  }
  {
    const auto graph = BuildSingleCandidateGraph();
    ASSERT_EQ(GraphUtils::RemoveEdge(graph->FindNode("data")->GetOutDataAnchor(0),
                                     graph->FindNode("host")->GetInDataAnchor(0)),
              GRAPH_SUCCESS);
    expect_not_candidate(graph);
  }
  {
    const auto graph = BuildSingleCandidateGraph();
    graph->FindNode("host")->GetOpDesc()->MutableInputDesc(0)->SetDataType(DT_RESOURCE);
    expect_not_candidate(graph);
  }
  {
    const auto graph = BuildSingleCandidateGraph();
    graph->FindNode("host")->GetOpDesc()->MutableOutputDesc(0)->SetOriginDataType(DT_VARIANT);
    expect_not_candidate(graph);
  }
  {
    const auto graph = BuildSingleCandidateGraph();
    ASSERT_EQ(GraphUtils::AddEdge(graph->FindNode("host")->GetOutDataAnchor(0),
                                  graph->FindNode("output")->GetInControlAnchor()),
              GRAPH_SUCCESS);
    expect_not_candidate(graph);
  }
  {
    auto graph = std::make_shared<ComputeGraph>("no_output_anchor");
    auto host = AddNode(graph, "host", "HostA", 0U, 0U);
    MarkCandidate(host);
    expect_not_candidate(graph);
  }
}

TEST(HostCpuFusionPassTest, RejectsCandidatesWithInvalidHostCpuMetadata) {
  const auto expect_not_candidate = [](const ComputeGraphPtr &graph,
                                       const HostCpuFusionOpSupportChecker &checker = SupportAllHostCpuOps) {
    HostCpuFusionPass pass(std::make_shared<FakeCompiler>(SUCCESS), checker);
    std::vector<std::vector<HostCpuFusionRegion>> components;
    EXPECT_EQ(pass.BuildFusionRegions(graph, components), NOT_CHANGED);
    EXPECT_TRUE(components.empty());
  };
  {
    auto graph = BuildSingleCandidateGraph();
    graph->FindNode("host")->GetOpDesc()->SetOpEngineName("DNN_VM_CPU");
    expect_not_candidate(graph);
  }
  {
    auto graph = BuildSingleCandidateGraph();
    graph->FindNode("host")->GetOpDesc()->SetOpKernelLibName("OTHER");
    expect_not_candidate(graph);
  }
  {
    auto graph = BuildSingleCandidateGraph();
    ASSERT_TRUE(AttrUtils::SetBool(graph->FindNode("host")->GetOpDesc(), "SmallShapeHostcpu", false));
    expect_not_candidate(graph);
  }
  {
    auto graph = BuildSingleCandidateGraph();
    ASSERT_EQ(graph->FindNode("host")->GetOpDesc()->DelAttr("SmallShapeHostcpu"), GRAPH_SUCCESS);
    ASSERT_TRUE(AttrUtils::SetListStr(graph->FindNode("host")->GetOpDesc(), "_resource_list", {"resource"}));
    expect_not_candidate(graph);
  }
  {
    auto graph = BuildSingleCandidateGraph();
    expect_not_candidate(graph, [](const std::string &) { return false; });
  }
  {
    auto graph = BuildSingleCandidateGraph();
    graph->FindNode("host")->GetOpDesc()->MutableOutputDesc(0)->SetDataType(DT_RESOURCE);
    expect_not_candidate(graph);
  }
}

TEST(HostCpuFusionPassTest, RejectsNullRunAndUsesDefaultKernelSupportChecker) {
  NodeEngineMap atomic_map;
  NodeEngineMap composite_map;
  HostCpuFusionPass explicit_pass(std::make_shared<FakeCompiler>(SUCCESS), SupportAllHostCpuOps);
  EXPECT_EQ(explicit_pass.Run(nullptr, atomic_map, composite_map), PARAM_INVALID);

  HostCpuFusionPass default_pass(std::make_shared<FakeCompiler>(SUCCESS));
  std::vector<std::vector<HostCpuFusionRegion>> components;
  EXPECT_EQ(default_pass.BuildFusionRegions(BuildSingleCandidateGraph(), components), NOT_CHANGED);
  EXPECT_TRUE(components.empty());
}

TEST(HostCpuFusionPassTest, RejectsGraphWithInvalidTopologyBeforeFusion) {
  auto graph = std::make_shared<ComputeGraph>("host_cpu_cycle");
  auto first = AddNode(graph, "first", "HostA", 1U, 1U);
  auto second = AddNode(graph, "second", "HostB", 1U, 1U);
  ASSERT_NE(first, nullptr);
  ASSERT_NE(second, nullptr);
  ASSERT_EQ(GraphUtils::AddEdge(first->GetOutDataAnchor(0), second->GetInDataAnchor(0)), GRAPH_SUCCESS);
  ASSERT_EQ(GraphUtils::AddEdge(second->GetOutDataAnchor(0), first->GetInDataAnchor(0)), GRAPH_SUCCESS);
  MarkCandidate(first);
  MarkCandidate(second);
  HostCpuFusionPass pass(std::make_shared<FakeCompiler>(SUCCESS), SupportAllHostCpuOps);
  std::vector<std::vector<HostCpuFusionRegion>> components;
  EXPECT_EQ(pass.BuildFusionRegions(graph, components), FAILED);
}

TEST(HostCpuFusionPassTest, RunStopsWhenGraphTopologyIsInvalid) {
  auto graph = std::make_shared<ComputeGraph>("host_cpu_cycle_run");
  auto first = AddNode(graph, "first", "HostA", 1U, 1U);
  auto second = AddNode(graph, "second", "HostB", 1U, 1U);
  ASSERT_NE(first, nullptr);
  ASSERT_NE(second, nullptr);
  ASSERT_EQ(GraphUtils::AddEdge(first->GetOutDataAnchor(0), second->GetInDataAnchor(0)), GRAPH_SUCCESS);
  ASSERT_EQ(GraphUtils::AddEdge(second->GetOutDataAnchor(0), first->GetInDataAnchor(0)), GRAPH_SUCCESS);
  MarkCandidate(first);
  MarkCandidate(second);
  NodeEngineMap atomic_map;
  NodeEngineMap composite_map;
  HostCpuFusionPass pass(std::make_shared<FakeCompiler>(SUCCESS), SupportAllHostCpuOps);
  EXPECT_EQ(pass.Run(graph, atomic_map, composite_map), FAILED);
}

TEST(HostCpuFusionPassTest, RollsBackWhenTransitionGraphBecomesInvalid) {
  const auto graph = BuildConvergingGraph();
  NodeEngineMap atomic_map;
  NodeEngineMap composite_map;
  HostCpuFusionPass pass(std::make_shared<TransitionInvalidCompiler>(graph), SupportAllHostCpuOps);
  EXPECT_EQ(pass.Run(graph, atomic_map, composite_map), FAILED);
  EXPECT_NE(graph->FindNode("a"), nullptr);
  EXPECT_NE(graph->FindNode("d"), nullptr);
  EXPECT_TRUE(atomic_map.empty());
  EXPECT_TRUE(composite_map.empty());
}

TEST(HostCpuFusionPassTest, RollsBackWhenOutputEdgeReplacementFails) {
  const auto graph = BuildNonConvergingGraph();
  NodeEngineMap atomic_map;
  NodeEngineMap composite_map;
  HostCpuFusionPass pass(std::make_shared<EdgeRemovingCompiler>(graph), SupportAllHostCpuOps);
  EXPECT_EQ(pass.Run(graph, atomic_map, composite_map), FAILED);
  EXPECT_NE(graph->FindNode("d"), nullptr);
  EXPECT_NE(graph->FindNode("e"), nullptr);
  EXPECT_TRUE(atomic_map.empty());
  EXPECT_TRUE(composite_map.empty());
}

TEST(HostCpuFusionPassTest, SerializesDescriptorRangesAndEscapesBoundaryAttributes) {
  const auto graph = BuildConvergingGraph();
  const auto node = graph->FindNode("a");
  ASSERT_NE(node, nullptr);
  auto output_desc = node->GetOpDesc()->MutableOutputDesc(0);
  ASSERT_NE(output_desc, nullptr);
  output_desc->SetOriginShape(GeShape({1, 2}));
  ASSERT_EQ(output_desc->SetShapeRange({{1, 4}}), GRAPH_SUCCESS);
  output_desc->SetName(std::string("tensor\r\t") + static_cast<char>(1));
  output_desc->SetExpandDimsRule("expand\r\t");

  const std::string escaped_value = std::string("value\r\t") + static_cast<char>(1) + static_cast<char>(0x7F);
  ASSERT_TRUE(AttrUtils::SetStr(node->GetOpDesc(), "escaped", escaped_value));
  ASSERT_TRUE(AttrUtils::SetInt(node->GetOpDesc(), "minimum", std::numeric_limits<int64_t>::min()));
  ASSERT_TRUE(AttrUtils::SetFloat(node->GetOpDesc(), "integral_float", 2.0F));
  for (const auto &name : {"escaped", "minimum", "integral_float"}) {
    node->GetOpDesc()->AppendIrAttrName(name);
  }

  HostCpuFusionPass pass(std::make_shared<FakeCompiler>(SUCCESS), SupportAllHostCpuOps);
  std::vector<std::vector<HostCpuFusionRegion>> components;
  ASSERT_EQ(pass.BuildFusionRegions(graph, components), SUCCESS);
  HostCpuFusionCodegenResult result;
  ASSERT_EQ(HostCpuFusionCodegen().Generate(components[0][0], result), SUCCESS);
  EXPECT_NE(result.source.find("desc.SetOriginShape"), std::string::npos);
  EXPECT_NE(result.source.find("desc.SetShapeRange"), std::string::npos);
  EXPECT_NE(result.source.find("\\r\\t\\001\\177"), std::string::npos);
  EXPECT_NE(result.source.find("(-9223372036854775807LL - 1LL)"), std::string::npos);
  EXPECT_NE(result.source.find("2.0F"), std::string::npos);
}

TEST(HostCpuFusionPassTest, RejectsMissingDeclaredIrAttribute) {
  const auto graph = BuildConvergingGraph();
  graph->FindNode("a")->GetOpDesc()->AppendIrAttrName("missing");
  HostCpuFusionPass pass(std::make_shared<FakeCompiler>(SUCCESS), SupportAllHostCpuOps);
  std::vector<std::vector<HostCpuFusionRegion>> components;
  ASSERT_EQ(pass.BuildFusionRegions(graph, components), SUCCESS);
  HostCpuFusionCodegenResult result;
  EXPECT_EQ(HostCpuFusionCodegen().Generate(components[0][0], result), UNSUPPORTED);
  EXPECT_TRUE(result.source.empty());
}

TEST(HostCpuFusionPassTest, CompilesGeneratedSourceAndRejectsInvalidSource) {
#if defined(__linux__)
#if defined(__aarch64__) || defined(__arm64__)
  GTEST_SKIP() << "HostCPU fusion JIT compiler is unavailable on ARM64.";
#endif
  if ((std::getenv("ASCEND_OPP_PATH") == nullptr) || (std::getenv("ASCEND_HOME_PATH") == nullptr)) {
    GTEST_SKIP() << "HostCPU fusion JIT requires ASCEND_OPP_PATH and ASCEND_HOME_PATH.";
  }
  const auto graph = BuildConvergingGraph();
  HostCpuFusionPass pass(std::make_shared<FakeCompiler>(SUCCESS), SupportAllHostCpuOps);
  std::vector<std::vector<HostCpuFusionRegion>> components;
  ASSERT_EQ(pass.BuildFusionRegions(graph, components), SUCCESS);
  ASSERT_EQ(components.size(), 1U);
  ASSERT_EQ(components[0].size(), 1U);
  HostCpuFusionCodegenResult result;
  ASSERT_EQ(HostCpuFusionCodegen().Generate(components[0][0], result), SUCCESS);
  HostCpuFusionCompiler compiler;
  std::vector<uint8_t> so_data;
  EXPECT_EQ(compiler.Compile(result.source, so_data), SUCCESS);
  ASSERT_FALSE(so_data.empty());
  EXPECT_EQ(compiler.Compile("", so_data), UNSUPPORTED);
  EXPECT_TRUE(so_data.empty());
  EXPECT_EQ(compiler.Compile("this is not valid C++;", so_data), UNSUPPORTED);
  EXPECT_TRUE(so_data.empty());
#else
  GTEST_SKIP() << "HostCPU fusion JIT uses Linux memfd.";
#endif
}

TEST(HostCpuFusionPassTest, RejectsJitWhenToolkitEnvironmentIsUnavailable) {
#if !defined(__linux__)
  GTEST_SKIP() << "HostCPU fusion JIT uses Linux memfd.";
#else
  const char *old_opp_value = std::getenv("ASCEND_OPP_PATH");
  const char *old_home_value = std::getenv("ASCEND_HOME_PATH");
  const std::string old_opp = (old_opp_value == nullptr) ? std::string() : old_opp_value;
  const std::string old_home = (old_home_value == nullptr) ? std::string() : old_home_value;
  ASSERT_EQ(setenv("ASCEND_OPP_PATH", "", 1), 0);
  ASSERT_EQ(setenv("ASCEND_HOME_PATH", "", 1), 0);
  const auto old_options = GetThreadLocalContext().GetAllGlobalOptions();
  HostCpuFusionCompiler compiler;
  std::vector<uint8_t> so_data;
  GetThreadLocalContext().SetGlobalOption({{OPTION_HOST_ENV_CPU, "aarch64"}});
  EXPECT_EQ(compiler.Compile("", so_data), UNSUPPORTED);
  EXPECT_EQ(compiler.Compile("int value = 0;", so_data), UNSUPPORTED);
  GetThreadLocalContext().SetGlobalOption({{OPTION_HOST_ENV_CPU, "x86_64"}});
  EXPECT_EQ(compiler.Compile("int value = 0;", so_data), UNSUPPORTED);
  GetThreadLocalContext().SetGlobalOption({{OPTION_HOST_ENV_CPU, "unknown_cpu"}});
  EXPECT_EQ(compiler.Compile("int value = 0;", so_data), UNSUPPORTED);

  ASSERT_EQ(setenv("ASCEND_OPP_PATH", "/tmp/no_such_opp/opp", 1), 0);
  ASSERT_EQ(setenv("ASCEND_HOME_PATH", "/tmp/no_such_home", 1), 0);
  EXPECT_EQ(compiler.Compile("int value = 0;", so_data), UNSUPPORTED);
  GetThreadLocalContext().SetGlobalOption(old_options);
  if (old_opp_value == nullptr) {
    EXPECT_EQ(unsetenv("ASCEND_OPP_PATH"), 0);
  } else {
    EXPECT_EQ(setenv("ASCEND_OPP_PATH", old_opp.c_str(), 1), 0);
  }
  if (old_home_value == nullptr) {
    EXPECT_EQ(unsetenv("ASCEND_HOME_PATH"), 0);
  } else {
    EXPECT_EQ(setenv("ASCEND_HOME_PATH", old_home.c_str(), 1), 0);
  }
#endif
}

TEST(HostCpuFusionPassTest, ReportsJitCompilerDiagnosticsForInvalidSource) {
#if !defined(__linux__)
  GTEST_SKIP() << "HostCPU fusion JIT uses Linux memfd.";
#else
  const char *old_opp_value = std::getenv("ASCEND_OPP_PATH");
  const char *old_home_value = std::getenv("ASCEND_HOME_PATH");
  const std::string old_opp = (old_opp_value == nullptr) ? std::string() : old_opp_value;
  const std::string old_home = (old_home_value == nullptr) ? std::string() : old_home_value;
  ASSERT_EQ(setenv("ASCEND_OPP_PATH", "/usr/local/Ascend/cann-9.2.0/opp", 1), 0);
  ASSERT_EQ(setenv("ASCEND_HOME_PATH", "/usr/local/Ascend/cann-9.2.0/x86_64-linux", 1), 0);
  const auto old_options = GetThreadLocalContext().GetAllGlobalOptions();
  GetThreadLocalContext().SetGlobalOption({{OPTION_HOST_ENV_CPU, "x86_64"}});
  HostCpuFusionCompiler compiler;
  std::vector<uint8_t> so_data;
  EXPECT_EQ(compiler.Compile("this is not valid C++;", so_data), UNSUPPORTED);
  EXPECT_TRUE(so_data.empty());
  GetThreadLocalContext().SetGlobalOption(old_options);
  if (old_opp_value == nullptr) {
    EXPECT_EQ(unsetenv("ASCEND_OPP_PATH"), 0);
  } else {
    EXPECT_EQ(setenv("ASCEND_OPP_PATH", old_opp.c_str(), 1), 0);
  }
  if (old_home_value == nullptr) {
    EXPECT_EQ(unsetenv("ASCEND_HOME_PATH"), 0);
  } else {
    EXPECT_EQ(setenv("ASCEND_HOME_PATH", old_home.c_str(), 1), 0);
  }
#endif
}

TEST(HostCpuFusionPassTest, ExistingFusedNodeRollsBackGraphCommit) {
  const auto graph = BuildConvergingGraph();
  HostCpuFusionPass region_pass(std::make_shared<FakeCompiler>(SUCCESS), SupportAllHostCpuOps);
  std::vector<std::vector<HostCpuFusionRegion>> components;
  ASSERT_EQ(region_pass.BuildFusionRegions(graph, components), SUCCESS);
  ASSERT_EQ(components.size(), 1U);
  ASSERT_EQ(components[0].size(), 1U);
  const std::string fused_name = std::string(kFusedHostCpuOpType) + "_" + components[0][0].chain_id;
  ASSERT_NE(graph->AddNode(std::make_shared<OpDesc>(fused_name, kFusedHostCpuOpType)), nullptr);

  NodeEngineMap atomic_map;
  NodeEngineMap composite_map;
  HostCpuFusionPass pass(std::make_shared<FakeCompiler>(SUCCESS), SupportAllHostCpuOps);
  EXPECT_EQ(pass.Run(graph, atomic_map, composite_map), FAILED);
  EXPECT_EQ(CountNodesByType(graph, kFusedHostCpuOpType), 1U);
  for (const auto &name : {"a", "b", "c", "d"}) {
    EXPECT_NE(graph->FindNode(name), nullptr);
  }
  EXPECT_EQ(graph->FindNode("output")->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode()->GetName(), "d");
  EXPECT_FALSE(AttrUtils::HasAttr(graph, std::string(kFusedHostCpuSoDataPrefix) + components[0][0].chain_id));
}

TEST(HostCpuFusionPassTest, ExistingSoDataPreventsCommitWithoutChangingGraph) {
  const auto graph = BuildConvergingGraph();
  HostCpuFusionPass region_pass(std::make_shared<FakeCompiler>(SUCCESS), SupportAllHostCpuOps);
  std::vector<std::vector<HostCpuFusionRegion>> components;
  ASSERT_EQ(region_pass.BuildFusionRegions(graph, components), SUCCESS);
  ASSERT_EQ(components.size(), 1U);
  ASSERT_EQ(components[0].size(), 1U);
  const std::string so_key = std::string(kFusedHostCpuSoDataPrefix) + components[0][0].chain_id;
  const std::vector<uint8_t> existing_so{'e', 'x', 'i', 's', 't', 'i', 'n', 'g'};
  ASSERT_TRUE(AttrUtils::SetBytes(graph, so_key, Buffer::CopyFrom(existing_so.data(), existing_so.size())));
  NodeEngineMap atomic_map;
  NodeEngineMap composite_map;
  EXPECT_EQ(region_pass.Run(graph, atomic_map, composite_map), FAILED);
  EXPECT_TRUE(AttrUtils::HasAttr(graph, so_key));
  EXPECT_NE(graph->FindNode("a"), nullptr);
  EXPECT_NE(graph->FindNode("d"), nullptr);
}

TEST(HostCpuFusionPassTest, RejectsNegativeOutputShapeSize) {
  const auto graph = BuildSingleCandidateGraph();
  const auto host = graph->FindNode("host");
  ASSERT_NE(host, nullptr);
  host->GetOpDesc()->MutableOutputDesc(0)->SetShape(GeShape({std::numeric_limits<int64_t>::max(), 2}));
  HostCpuFusionPass pass(std::make_shared<FakeCompiler>(SUCCESS), SupportAllHostCpuOps);
  std::vector<std::vector<HostCpuFusionRegion>> components;
  EXPECT_EQ(pass.BuildFusionRegions(graph, components), NOT_CHANGED);
  EXPECT_TRUE(components.empty());
}

TEST(HostCpuFusionPassTest, KeepsDanglingCandidateComponentUnchanged) {
  auto graph = std::make_shared<ComputeGraph>("host_cpu_dangling");
  auto data = AddNode(graph, "data", "Data", 0U, 1U);
  auto first = AddNode(graph, "first", "HostA", 1U, 1U);
  auto second = AddNode(graph, "second", "HostB", 1U, 1U);
  ASSERT_NE(data, nullptr);
  ASSERT_NE(first, nullptr);
  ASSERT_NE(second, nullptr);
  AddEdge(data, 0U, first, 0U);
  AddEdge(first, 0U, second, 0U);
  MarkCandidate(first);
  MarkCandidate(second);
  HostCpuFusionPass pass(std::make_shared<FakeCompiler>(SUCCESS), SupportAllHostCpuOps);
  std::vector<std::vector<HostCpuFusionRegion>> components;
  EXPECT_EQ(pass.BuildFusionRegions(graph, components), NOT_CHANGED);
  EXPECT_TRUE(components.empty());
}

TEST(HostCpuFusionPassTest, SerializesMultipleRangesAndRejectsUnsupportedTensorSize) {
  const auto graph = BuildConvergingGraph();
  const auto first = graph->FindNode("a");
  ASSERT_NE(first, nullptr);
  auto input_desc = first->GetOpDesc()->MutableInputDesc(0);
  auto output_desc = first->GetOpDesc()->MutableOutputDesc(0);
  ASSERT_NE(input_desc, nullptr);
  ASSERT_NE(output_desc, nullptr);
  input_desc->SetOriginShape(GeShape({3, 4}));
  output_desc->SetOriginShape(GeShape({3, 4}));
  ASSERT_EQ(input_desc->SetShapeRange({{1, 3}, {2, 4}}), GRAPH_SUCCESS);
  ASSERT_EQ(output_desc->SetShapeRange({{1, 3}, {2, 4}}), GRAPH_SUCCESS);

  HostCpuFusionPass pass(std::make_shared<FakeCompiler>(SUCCESS), SupportAllHostCpuOps);
  std::vector<std::vector<HostCpuFusionRegion>> components;
  ASSERT_EQ(pass.BuildFusionRegions(graph, components), SUCCESS);
  ASSERT_EQ(components.size(), 1U);
  HostCpuFusionCodegenResult result;
  ASSERT_EQ(HostCpuFusionCodegen().Generate(components[0][0], result), SUCCESS);
  EXPECT_NE(result.source.find("std::vector<std::pair<int64_t, int64_t>>{{1LL, 3LL}, {2LL, 4LL}}"), std::string::npos);
  EXPECT_NE(result.source.find("desc.SetOriginShape(ge::Shape(std::vector<int64_t>{3, 4}))"), std::string::npos);

  const auto invalid_graph = BuildConvergingGraph();
  const auto invalid = invalid_graph->FindNode("b");
  ASSERT_NE(invalid, nullptr);
  HostCpuFusionPass invalid_pass(std::make_shared<FakeCompiler>(SUCCESS), SupportAllHostCpuOps);
  components.clear();
  ASSERT_EQ(invalid_pass.BuildFusionRegions(invalid_graph, components), SUCCESS);
  invalid->GetOpDesc()->MutableOutputDesc(0)->SetShape(GeShape({std::numeric_limits<int64_t>::max(), 2}));
  EXPECT_EQ(HostCpuFusionCodegen().Generate(components[0][0], result), UNSUPPORTED);

  invalid->GetOpDesc()->MutableOutputDesc(0)->SetShape(GeShape(std::vector<int64_t>{}));
  invalid->GetOpDesc()->MutableOutputDesc(0)->SetDataType(static_cast<DataType>(DT_MAX));
  EXPECT_EQ(HostCpuFusionCodegen().Generate(components[0][0], result), UNSUPPORTED);
}

TEST(HostCpuFusionPassTest, SanitizesHostCpuFusionInputNames) {
  const auto graph = BuildSingleCandidateGraph();
  const auto data = graph->FindNode("data");
  ASSERT_NE(data, nullptr);
  auto &output_names = data->GetOpDesc()->MutableAllOutputName();
  ASSERT_EQ(output_names.size(), 1U);
  output_names.clear();
  output_names.emplace(std::string(80U, 'x') + "-name", 0U);
  EXPECT_EQ(GetHostCpuFusionInputName(data->GetOutDataAnchor(0), 3U).size(), std::string("input_3_").size() + 64U);
  output_names.clear();
  EXPECT_EQ(GetHostCpuFusionInputName(data->GetOutDataAnchor(0), 4U), "input_4_tensor");
  EXPECT_EQ(GetHostCpuFusionInputName(nullptr, 5U), "input_5_tensor");
}

TEST(HostCpuFusionPassTest, RejectsGeneratedSourceThatExceedsLimit) {
  const auto graph = BuildConvergingGraph();
  const auto node = graph->FindNode("a");
  ASSERT_NE(node, nullptr);
  const std::string huge_value(1100U * 1024U, 'x');
  ASSERT_TRUE(AttrUtils::SetStr(node->GetOpDesc(), "huge_attr", huge_value));
  node->GetOpDesc()->AppendIrAttrName("huge_attr");
  HostCpuFusionPass pass(std::make_shared<FakeCompiler>(SUCCESS), SupportAllHostCpuOps);
  std::vector<std::vector<HostCpuFusionRegion>> components;
  ASSERT_EQ(pass.BuildFusionRegions(graph, components), SUCCESS);
  HostCpuFusionCodegenResult result;
  EXPECT_EQ(HostCpuFusionCodegen().Generate(components[0][0], result), UNSUPPORTED);
  EXPECT_TRUE(result.source.empty());
}

TEST(HostCpuFusionPassTest, RunSkipsEmptyGraphWithoutPreparingFusion) {
  const auto graph = std::make_shared<ComputeGraph>("host_cpu_empty_run");
  NodeEngineMap atomic_map;
  NodeEngineMap composite_map;
  HostCpuFusionPass pass(std::make_shared<FakeCompiler>(SUCCESS), SupportAllHostCpuOps);

  EXPECT_EQ(pass.Run(graph, atomic_map, composite_map), NOT_CHANGED);
  EXPECT_TRUE(atomic_map.empty());
  EXPECT_TRUE(composite_map.empty());
}

TEST(HostCpuFusionPassTest, IncludesReferencePortIndexesInChainFingerprint) {
  const auto graph = BuildConvergingGraph();
  const auto node = graph->FindNode("a");
  ASSERT_NE(node, nullptr);
  auto input_desc = node->GetOpDesc()->MutableInputDesc(0);
  ASSERT_NE(input_desc, nullptr);
  input_desc->SetRefPortByIndex({0U, 1U});

  HostCpuFusionPass pass(std::make_shared<FakeCompiler>(SUCCESS), SupportAllHostCpuOps);
  std::vector<std::vector<HostCpuFusionRegion>> components;
  ASSERT_EQ(pass.BuildFusionRegions(graph, components), SUCCESS);
  ASSERT_EQ(components.size(), 1U);
  ASSERT_EQ(components[0].size(), 1U);
  EXPECT_FALSE(components[0][0].chain_id.empty());
}

TEST(HostCpuFusionPassTest, GeneratesTensorDescriptorWithMultipleDimensions) {
  const auto graph = BuildConvergingGraph();
  const auto node = graph->FindNode("a");
  ASSERT_NE(node, nullptr);
  auto output_desc = node->GetOpDesc()->MutableOutputDesc(0);
  ASSERT_NE(output_desc, nullptr);
  output_desc->SetShape(GeShape({2, 3}));

  HostCpuFusionPass pass(std::make_shared<FakeCompiler>(SUCCESS), SupportAllHostCpuOps);
  std::vector<std::vector<HostCpuFusionRegion>> components;
  ASSERT_EQ(pass.BuildFusionRegions(graph, components), SUCCESS);
  ASSERT_EQ(components.size(), 1U);
  HostCpuFusionCodegenResult result;
  ASSERT_EQ(HostCpuFusionCodegen().Generate(components[0][0], result), SUCCESS);
  EXPECT_NE(result.source.find("std::vector<int64_t>{2, 3}"), std::string::npos);
}

}  // namespace ge
