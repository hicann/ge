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

#include <algorithm>
#include <cstdlib>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "framework/common/host_cpu_fusion_attr.h"
#include "graph/custom_op.h"
#include "graph/custom_op_factory.h"
#include "graph/op_so_bin.h"
#include "graph/partition/optimizer/host_cpu_fusion_codegen.h"
#include "graph/partition/optimizer/host_cpu_fusion_pass.h"
#include "graph/utils/attr_utils.h"
#include "graph/utils/graph_utils.h"

namespace ge {
namespace {
constexpr char kHostCpuEngineName[] = "DNN_VM_HOST_CPU";
constexpr char kHostCpuKernelLibName[] = "DNN_VM_HOST_CPU_OP_STORE";
constexpr char kSmallShapeHostCpu[] = "SmallShapeHostcpu";

std::string GetToolkitHome() {
  const char *home = std::getenv("ASCEND_HOME_PATH");
  if ((home != nullptr) && (home[0] != '\0')) {
    return home;
  }
  const char *install_path = std::getenv("ASCEND_INSTALL_PATH");
  if ((install_path != nullptr) && (install_path[0] != '\0')) {
    return install_path;
  }
  return "/usr/local/Ascend/cann-9.2.0/x86_64-linux";
}

class EnvGuard final {
 public:
  EnvGuard(const char *name, const char *value) : name_(name), had_old_(false) {
    const char *old = std::getenv(name);
    if (old != nullptr) {
      old_ = old;
      had_old_ = true;
    }
    EXPECT_EQ(setenv(name, value, 1), 0);
  }
  ~EnvGuard() {
    if (had_old_) {
      (void)setenv(name_.c_str(), old_.c_str(), 1);
    } else {
      (void)unsetenv(name_.c_str());
    }
  }

 private:
  std::string name_;
  std::string old_;
  bool had_old_;
};

NodePtr AddNode(const ComputeGraphPtr &graph, const std::string &name, const std::string &type,
                const size_t input_count, const size_t output_count) {
  const GeTensorDesc desc(GeShape({2}), FORMAT_ND, DT_INT64);
  auto op_desc = std::make_shared<OpDesc>(name, type);
  for (size_t i = 0U; i < input_count; ++i) {
    EXPECT_EQ(op_desc->AddInputDesc("input" + std::to_string(i), desc), GRAPH_SUCCESS);
  }
  for (size_t i = 0U; i < output_count; ++i) {
    EXPECT_EQ(op_desc->AddOutputDesc("output" + std::to_string(i), desc), GRAPH_SUCCESS);
  }
  return graph->AddNode(op_desc);
}

void Connect(const NodePtr &source, const size_t source_index, const NodePtr &target, const size_t target_index) {
  ASSERT_EQ(GraphUtils::AddEdge(source->GetOutDataAnchor(static_cast<int32_t>(source_index)),
                                target->GetInDataAnchor(static_cast<int32_t>(target_index))),
            GRAPH_SUCCESS);
}

void MarkHostCpuCandidate(const NodePtr &node) {
  node->GetOpDesc()->SetOpEngineName(kHostCpuEngineName);
  node->GetOpDesc()->SetOpKernelLibName(kHostCpuKernelLibName);
  ASSERT_TRUE(AttrUtils::SetBool(node->GetOpDesc(), kSmallShapeHostCpu, true));
}

ComputeGraphPtr BuildLinearGraph() {
  auto graph = std::make_shared<ComputeGraph>("host_cpu_fusion_linear");
  auto data = AddNode(graph, "data", "Data", 0U, 1U);
  auto first = AddNode(graph, "first", "HostA", 1U, 1U);
  auto second = AddNode(graph, "second", "HostB", 1U, 1U);
  auto output = AddNode(graph, "output", "NetOutput", 1U, 0U);
  Connect(data, 0U, first, 0U);
  Connect(first, 0U, second, 0U);
  Connect(second, 0U, output, 0U);
  MarkHostCpuCandidate(first);
  MarkHostCpuCandidate(second);
  return graph;
}

ComputeGraphPtr BuildSplitGraph() {
  auto graph = std::make_shared<ComputeGraph>("host_cpu_fusion_split");
  auto data = AddNode(graph, "data", "Data", 0U, 1U);
  auto first = AddNode(graph, "first", "HostA", 1U, 1U);
  auto left = AddNode(graph, "left", "HostB", 1U, 1U);
  auto right = AddNode(graph, "right", "HostC", 1U, 1U);
  auto left_out = AddNode(graph, "left_out", "NetOutput", 1U, 0U);
  auto right_out = AddNode(graph, "right_out", "NetOutput", 1U, 0U);
  Connect(data, 0U, first, 0U);
  Connect(first, 0U, left, 0U);
  Connect(first, 0U, right, 0U);
  Connect(left, 0U, left_out, 0U);
  Connect(right, 0U, right_out, 0U);
  MarkHostCpuCandidate(first);
  MarkHostCpuCandidate(left);
  MarkHostCpuCandidate(right);
  return graph;
}

class MinimalCustomOpCompiler final : public HostCpuFusionCompiler {
 public:
  Status Compile(const std::string &, std::vector<uint8_t> &so_data) const override {
    return HostCpuFusionCompiler::Compile(
        "extern \"C\" __attribute__((visibility(\"default\"))) unsigned int "
        "GetRegisteredCustomOpCreatorAbiVersion() { return 2U; }\n"
        "extern \"C\" __attribute__((visibility(\"default\"))) unsigned long "
        "GetRegisteredCustomOpCreatorNum() { return 0U; }\n"
        "extern \"C\" __attribute__((visibility(\"default\"))) int "
        "GetRegisteredCustomOpCreators(void *, unsigned long, unsigned long) { return 0; }\n",
        so_data);
  }
};

std::vector<uint8_t> MakeMinimalElfData() {
  std::vector<uint8_t> so_data(20U, 0U);
  so_data[0] = 0x7FU;
  so_data[1] = 'E';
  so_data[2] = 'L';
  so_data[3] = 'F';
  so_data[4] = 2U;
  so_data[5] = 1U;
  so_data[6] = 1U;
  so_data[16] = 3U;
  return so_data;
}

class MinimalElfCompiler final : public HostCpuFusionCompiler {
 public:
  Status Compile(const std::string &, std::vector<uint8_t> &so_data) const override {
    so_data = MakeMinimalElfData();
    return SUCCESS;
  }
};

class MissingCreatorSymbolsCompiler final : public HostCpuFusionCompiler {
 public:
  Status Compile(const std::string &, std::vector<uint8_t> &so_data) const override {
    return HostCpuFusionCompiler::Compile(
        "extern \"C\" __attribute__((visibility(\"default\"))) int HostCpuFusionMissingCreatorSymbols() "
        "{ return 0; }\n",
        so_data);
  }
};

class CustomOpRegistrationGuard final {
 public:
  explicit CustomOpRegistrationGuard(const std::string &op_type) : op_type_(op_type.c_str()) {}
  ~CustomOpRegistrationGuard() {
    CustomOpFactory::RemoveCustomOps({op_type_});
  }

 private:
  AscendString op_type_;
};

bool SupportAll(const std::string &) {
  return true;
}

HostCpuFusionCodegenResult GetCodegenResult(HostCpuFusionPass &pass, const ComputeGraphPtr &graph) {
  std::vector<std::vector<HostCpuFusionRegion>> components;
  EXPECT_EQ(pass.BuildFusionRegions(graph, components), SUCCESS);
  EXPECT_EQ(components.size(), 1U);
  HostCpuFusionCodegenResult result;
  if (components.size() != 1U) {
    return result;
  }
  EXPECT_EQ(components[0].size(), 1U);
  if (components[0].size() != 1U) {
    return result;
  }
  EXPECT_EQ(HostCpuFusionCodegen().Generate(components[0][0], result), SUCCESS);
  return result;
}

OpSoBinPtr MakeSoBin(const std::string &so_name, const std::vector<uint8_t> &so_data) {
  auto data = std::make_unique<char_t[]>(so_data.size());
  std::copy(so_data.cbegin(), so_data.cend(), data.get());
  return std::make_shared<OpSoBin>(so_name, kFusedHostCpuSoVendor, std::move(data),
                                   static_cast<uint32_t>(so_data.size()), SoBinType::kCustomOp);
}

void ExpectExistingFusedNodeRollsBack(const bool with_existing_so) {
  const auto graph = BuildLinearGraph();
  NodeEngineMap atomic_map;
  NodeEngineMap composite_map;
  HostCpuFusionPass pass(std::make_shared<MinimalElfCompiler>(), SupportAll);
  const auto codegen_result = GetCodegenResult(pass, graph);
  ASSERT_FALSE(codegen_result.register_name.empty());
  ASSERT_EQ(
      CustomOpFactory::RegisterCustomOpCreator(AscendString(codegen_result.register_name.c_str()), OpBackend::kHostCPU,
                                               []() -> std::unique_ptr<BaseCustomOp> { return nullptr; }),
      GRAPH_SUCCESS);
  const CustomOpRegistrationGuard registration_guard(codegen_result.register_name);
  ASSERT_NE(AddNode(graph, codegen_result.register_name, "ExistingFusedHostCpu", 0U, 0U), nullptr);

  OpSoBinPtr existing_so;
  if (with_existing_so) {
    existing_so = MakeSoBin("libunrelated.so", MakeMinimalElfData());
    ASSERT_NE(existing_so, nullptr);
    ASSERT_TRUE(graph->SetExtAttr("bin_file_buffer",
                                  std::map<std::string, OpSoBinPtr>{{"unrelated/libunrelated.so", existing_so}}));
  }

  EXPECT_EQ(pass.Run(graph, atomic_map, composite_map), FAILED);
  EXPECT_NE(graph->FindNode("first"), nullptr);
  EXPECT_NE(graph->FindNode("second"), nullptr);
  EXPECT_NE(graph->FindNode(codegen_result.register_name), nullptr);
  EXPECT_TRUE(atomic_map.empty());
  EXPECT_TRUE(composite_map.empty());
  const auto so_buffer = graph->GetExtAttr<std::map<std::string, OpSoBinPtr>>("bin_file_buffer");
  if (with_existing_so) {
    ASSERT_NE(so_buffer, nullptr);
    ASSERT_EQ(so_buffer->size(), 1U);
    EXPECT_EQ(so_buffer->at("unrelated/libunrelated.so"), existing_so);
  } else {
    EXPECT_EQ(so_buffer, nullptr);
  }
}
}  // namespace

TEST(HostCpuFusionPassTest, KeepsSharedAncestorBranchesInSingleRegion) {
  const auto graph = BuildSplitGraph();
  HostCpuFusionPass pass(std::make_shared<MinimalCustomOpCompiler>(), SupportAll);
  std::vector<std::vector<HostCpuFusionRegion>> components;
  ASSERT_EQ(pass.BuildFusionRegions(graph, components), SUCCESS);
  ASSERT_EQ(components.size(), 1U);
  ASSERT_EQ(components[0].size(), 1U);
  EXPECT_EQ(components[0][0].nodes.size(), 3U);
  EXPECT_EQ(components[0][0].external_outputs.size(), 2U);
}

TEST(HostCpuFusionPassTest, CommitsGeneratedCustomOpSoAndReplacesCandidates) {
#if defined(__linux__)
  const auto toolkit_home = GetToolkitHome();
  EnvGuard home("ASCEND_HOME_PATH", toolkit_home.c_str());
  EnvGuard opp("ASCEND_OPP_PATH", "");
  const auto graph = BuildLinearGraph();
  NodeEngineMap atomic_map;
  NodeEngineMap composite_map;
  HostCpuFusionPass pass(std::make_shared<MinimalCustomOpCompiler>(), SupportAll);
  std::vector<std::vector<HostCpuFusionRegion>> components;
  ASSERT_EQ(pass.BuildFusionRegions(graph, components), SUCCESS);
  ASSERT_EQ(components.size(), 1U);
  ASSERT_EQ(components[0].size(), 1U);
  HostCpuFusionCodegenResult codegen_result;
  ASSERT_EQ(HostCpuFusionCodegen().Generate(components[0][0], codegen_result), SUCCESS);
  // This UT verifies graph commit. The generated SO loading path is covered by ST; pre-registering the creator here
  // avoids retaining a process-global SO handle that would pollute later GEInitialize/GEFinalize tests in this binary.
  ASSERT_EQ(
      CustomOpFactory::RegisterCustomOpCreator(AscendString(codegen_result.register_name.c_str()), OpBackend::kHostCPU,
                                               []() -> std::unique_ptr<BaseCustomOp> { return nullptr; }),
      GRAPH_SUCCESS);
  const CustomOpRegistrationGuard registration_guard(codegen_result.register_name);
  ASSERT_EQ(pass.Run(graph, atomic_map, composite_map), SUCCESS);
  EXPECT_EQ(graph->FindNode("first"), nullptr);
  EXPECT_EQ(graph->FindNode("second"), nullptr);
  EXPECT_EQ(graph->GetDirectNodesSize(), 3U);
  EXPECT_EQ(atomic_map.size(), 1U);
  EXPECT_EQ(composite_map.size(), 1U);
#else
  GTEST_SKIP() << "HostCPU fusion JIT uses Linux memfd.";
#endif
}

TEST(HostCpuFusionPassTest, KeepsOriginalGraphWhenCompilerFails) {
  class FailingCompiler final : public HostCpuFusionCompiler {
   public:
    Status Compile(const std::string &, std::vector<uint8_t> &so_data) const override {
      so_data.clear();
      return FAILED;
    }
  };
  const auto graph = BuildLinearGraph();
  NodeEngineMap atomic_map;
  NodeEngineMap composite_map;
  HostCpuFusionPass pass(std::make_shared<FailingCompiler>(), SupportAll);
  EXPECT_EQ(pass.Run(graph, atomic_map, composite_map), NOT_CHANGED);
  EXPECT_NE(graph->FindNode("first"), nullptr);
  EXPECT_NE(graph->FindNode("second"), nullptr);
}

TEST(HostCpuFusionPassTest, KeepsOriginalGraphWhenCustomOpCreatorSymbolsAreMissing) {
#if defined(__linux__)
  const auto toolkit_home = GetToolkitHome();
  EnvGuard home("ASCEND_HOME_PATH", toolkit_home.c_str());
  EnvGuard opp("ASCEND_OPP_PATH", "");
  const auto graph = BuildLinearGraph();
  NodeEngineMap atomic_map;
  NodeEngineMap composite_map;
  HostCpuFusionPass pass(std::make_shared<MissingCreatorSymbolsCompiler>(), SupportAll);
  EXPECT_EQ(pass.Run(graph, atomic_map, composite_map), FAILED);
  EXPECT_NE(graph->FindNode("first"), nullptr);
  EXPECT_NE(graph->FindNode("second"), nullptr);
  EXPECT_TRUE(atomic_map.empty());
  EXPECT_TRUE(composite_map.empty());
  const auto so_buffer = graph->GetExtAttr<std::map<std::string, OpSoBinPtr>>("bin_file_buffer");
  EXPECT_EQ(so_buffer, nullptr);
#else
  GTEST_SKIP() << "HostCPU fusion JIT uses Linux memfd.";
#endif
}

TEST(HostCpuFusionPassTest, RejectsExistingCustomOpSoWithDifferentContents) {
  const auto graph = BuildLinearGraph();
  NodeEngineMap atomic_map;
  NodeEngineMap composite_map;
  HostCpuFusionPass pass(std::make_shared<MinimalElfCompiler>(), SupportAll);
  const auto codegen_result = GetCodegenResult(pass, graph);
  ASSERT_FALSE(codegen_result.register_name.empty());
  ASSERT_EQ(
      CustomOpFactory::RegisterCustomOpCreator(AscendString(codegen_result.register_name.c_str()), OpBackend::kHostCPU,
                                               []() -> std::unique_ptr<BaseCustomOp> { return nullptr; }),
      GRAPH_SUCCESS);
  const CustomOpRegistrationGuard registration_guard(codegen_result.register_name);

  auto different_so_data = MakeMinimalElfData();
  different_so_data.back() = 1U;
  const std::string so_name = "lib" + codegen_result.register_name + ".so";
  const std::string so_key = std::string(kFusedHostCpuSoVendor) + "/" + so_name;
  const auto existing_so = MakeSoBin(so_name, different_so_data);
  ASSERT_NE(existing_so, nullptr);
  ASSERT_TRUE(graph->SetExtAttr("bin_file_buffer", std::map<std::string, OpSoBinPtr>{{so_key, existing_so}}));

  EXPECT_EQ(pass.Run(graph, atomic_map, composite_map), PARAM_INVALID);
  EXPECT_NE(graph->FindNode("first"), nullptr);
  EXPECT_NE(graph->FindNode("second"), nullptr);
  EXPECT_TRUE(atomic_map.empty());
  EXPECT_TRUE(composite_map.empty());
  const auto so_buffer = graph->GetExtAttr<std::map<std::string, OpSoBinPtr>>("bin_file_buffer");
  ASSERT_NE(so_buffer, nullptr);
  ASSERT_EQ(so_buffer->size(), 1U);
  EXPECT_EQ(so_buffer->at(so_key), existing_so);
}

TEST(HostCpuFusionPassTest, KeepsOriginalGraphWhenCustomOpSoBufferHasWrongType) {
  const auto graph = BuildLinearGraph();
  NodeEngineMap atomic_map;
  NodeEngineMap composite_map;
  HostCpuFusionPass pass(std::make_shared<MinimalElfCompiler>(), SupportAll);
  const auto codegen_result = GetCodegenResult(pass, graph);
  ASSERT_FALSE(codegen_result.register_name.empty());
  ASSERT_EQ(
      CustomOpFactory::RegisterCustomOpCreator(AscendString(codegen_result.register_name.c_str()), OpBackend::kHostCPU,
                                               []() -> std::unique_ptr<BaseCustomOp> { return nullptr; }),
      GRAPH_SUCCESS);
  const CustomOpRegistrationGuard registration_guard(codegen_result.register_name);
  ASSERT_TRUE(graph->SetExtAttr("bin_file_buffer", std::string("invalid buffer type")));

  EXPECT_EQ(pass.Run(graph, atomic_map, composite_map), FAILED);
  EXPECT_NE(graph->FindNode("first"), nullptr);
  EXPECT_NE(graph->FindNode("second"), nullptr);
  EXPECT_TRUE(atomic_map.empty());
  EXPECT_TRUE(composite_map.empty());
  const auto so_buffer = graph->GetExtAttr<std::string>("bin_file_buffer");
  ASSERT_NE(so_buffer, nullptr);
  EXPECT_EQ(*so_buffer, "invalid buffer type");
}

TEST(HostCpuFusionPassTest, RollsBackCustomOpArtifactsWhenFusedNodeAlreadyExists) {
  ExpectExistingFusedNodeRollsBack(false);
  ExpectExistingFusedNodeRollsBack(true);
}

}  // namespace ge
