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
#include <string>
#include <vector>

#include "framework/common/host_cpu_fusion_attr.h"
#include "graph/op_so_bin.h"
#include "graph/partition/optimizer/host_cpu_fusion_codegen.h"
#include "graph/partition/optimizer/host_cpu_fusion_pass.h"
#include "graph/utils/attr_utils.h"
#include "graph/utils/graph_utils.h"

namespace ge {
namespace {

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

std::string GetToolkitOpp(const std::string &home) {
  constexpr char kX86Suffix[] = "/x86_64-linux";
  if (home.size() > (sizeof(kX86Suffix) - 1U) &&
      home.compare(home.size() - (sizeof(kX86Suffix) - 1U), sizeof(kX86Suffix) - 1U, kX86Suffix) == 0) {
    return home.substr(0U, home.size() - (sizeof(kX86Suffix) - 1U)) + "/opp";
  }
  return home + "/opp";
}

NodePtr MakeNode(const ComputeGraphPtr &graph, const std::string &name, const std::string &type, const size_t inputs,
                 const size_t outputs) {
  const GeTensorDesc desc(GeShape({2, 3}), FORMAT_ND, DT_INT64);
  auto op_desc = std::make_shared<OpDesc>(name, type);
  for (size_t i = 0U; i < inputs; ++i) {
    EXPECT_EQ(op_desc->AddInputDesc("input" + std::to_string(i), desc), GRAPH_SUCCESS);
  }
  for (size_t i = 0U; i < outputs; ++i) {
    EXPECT_EQ(op_desc->AddOutputDesc("output" + std::to_string(i), desc), GRAPH_SUCCESS);
  }
  return graph->AddNode(op_desc);
}

void Connect(const NodePtr &source, const size_t source_index, const NodePtr &target, const size_t target_index) {
  ASSERT_EQ(GraphUtils::AddEdge(source->GetOutDataAnchor(static_cast<int32_t>(source_index)),
                                target->GetInDataAnchor(static_cast<int32_t>(target_index))),
            GRAPH_SUCCESS);
}

HostCpuFusionRegion MakeRegion(const ComputeGraphPtr &graph) {
  const auto data = MakeNode(graph, "data", "Data", 0U, 1U);
  const auto first = MakeNode(graph, "first", "HostST", 1U, 2U);
  const auto second = MakeNode(graph, "second", "HostST", 1U, 1U);
  const auto side = MakeNode(graph, "side", "DeviceOp", 1U, 1U);
  const auto output = MakeNode(graph, "output", "NetOutput", 2U, 0U);
  Connect(data, 0U, first, 0U);
  Connect(first, 0U, second, 0U);
  Connect(first, 1U, side, 0U);
  Connect(second, 0U, output, 0U);
  Connect(side, 0U, output, 1U);

  HostCpuFusionRegion region;
  region.chain_id = "st_codegen_chain";
  region.nodes = {first, second};
  region.external_inputs = {data->GetOutDataAnchor(0)};
  region.external_outputs = {{first->GetOutDataAnchor(1), {side->GetInDataAnchor(0)}},
                             {second->GetOutDataAnchor(0), {output->GetInDataAnchor(0)}}};
  return region;
}

void MarkHostCpuCandidateForPassSt(const NodePtr &node) {
  node->GetOpDesc()->SetOpEngineName("DNN_VM_HOST_CPU");
  node->GetOpDesc()->SetOpKernelLibName("DNN_VM_HOST_CPU_OP_STORE");
  ASSERT_TRUE(AttrUtils::SetBool(node->GetOpDesc(), "SmallShapeHostcpu", true));
}

ComputeGraphPtr BuildPassGraph() {
  auto graph = std::make_shared<ComputeGraph>("host_cpu_fusion_pass_st");
  const auto data = MakeNode(graph, "pass_data", "Data", 0U, 1U);
  const auto first = MakeNode(graph, "pass_first", "HostPassA", 1U, 1U);
  const auto second = MakeNode(graph, "pass_second", "HostPassB", 1U, 1U);
  const auto output = MakeNode(graph, "pass_output", "NetOutput", 1U, 0U);
  Connect(data, 0U, first, 0U);
  Connect(first, 0U, second, 0U);
  Connect(second, 0U, output, 0U);
  MarkHostCpuCandidateForPassSt(first);
  MarkHostCpuCandidateForPassSt(second);
  return graph;
}

class MinimalCustomOpCompilerForPassSt final : public HostCpuFusionCompiler {
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

bool SupportAllForPassSt(const std::string &) {
  return true;
}

class EnvGuard final {
 public:
  EnvGuard(const char *name, const char *value) : name_(name), old_(), had_old_(false) {
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

}  // namespace

// 用例描述：验证 HostCPU 融合区域能够生成当前 CustomOp 编排源码并编译为 ELF SO。
// 预置条件：使用本机 Toolkit 头文件和 g++，构造带内部中间张量及两个外部输出的融合区域。
// 测试步骤：生成源码，检查输入/输出绑定与 CustomOp ABI，再执行 JIT 编译。
// 预期结果：源码生成成功；工具包提供 HostCPU ABI 时编译产物为有效 ELF 共享对象，否则返回 UNSUPPORTED。
TEST(HostCpuFusionCodegenST, GeneratesAndCompilesCurrentCustomOp) {
#if defined(__linux__)
  const auto toolkit_home = GetToolkitHome();
  EnvGuard home("ASCEND_HOME_PATH", toolkit_home.c_str());
  EnvGuard opp("ASCEND_OPP_PATH", "");
  auto graph = std::make_shared<ComputeGraph>("host_cpu_codegen_st");
  const auto region = MakeRegion(graph);
  HostCpuFusionCodegenResult result;
  ASSERT_EQ(HostCpuFusionCodegen().Generate(region, result), SUCCESS);
  EXPECT_EQ(result.register_name, "FusedHostCpu_st_codegen_chain");
  EXPECT_NE(result.source.find("internal_tensor_0"), std::string::npos);
  EXPECT_NE(result.source.find("std::array<const gert::Tensor *, 1U>"), std::string::npos);
  EXPECT_NE(result.source.find("std::array<gert::Tensor *, 2U>"), std::string::npos);
  EXPECT_NE(result.source.find("REG_OP_BACKEND"), std::string::npos);
  EXPECT_NE(result.source.find("GetRegisteredCustomOpCreatorAbiVersion"), std::string::npos);

  std::vector<uint8_t> so_data;
  const auto compile_status = HostCpuFusionCompiler().Compile(result.source, so_data);
  if (compile_status == SUCCESS) {
    ASSERT_GT(so_data.size(), 20U);
    EXPECT_EQ(so_data[0], 0x7FU);
    EXPECT_EQ(so_data[1], 'E');
    EXPECT_EQ(so_data[2], 'L');
    EXPECT_EQ(so_data[3], 'F');
  } else {
    // Older Toolkit packages may not expose the HostCpuExecuteOp ABI yet.
    EXPECT_EQ(compile_status, UNSUPPORTED);
    EXPECT_TRUE(so_data.empty());
  }
#else
  GTEST_SKIP() << "HostCPU fusion JIT uses Linux memfd.";
#endif
}

TEST(HostCpuFusionCodegenST, ExercisesCodegenRejectionAndNameBoundaries) {
  auto graph = std::make_shared<ComputeGraph>("host_cpu_codegen_rejection");
  const auto valid = MakeRegion(graph);
  HostCpuFusionCodegen codegen;
  HostCpuFusionCodegenResult result;

  auto invalid = valid;
  invalid.chain_id.clear();
  EXPECT_EQ(codegen.Generate(invalid, result), PARAM_INVALID);
  invalid = valid;
  invalid.chain_id = "9bad";
  EXPECT_EQ(codegen.Generate(invalid, result), PARAM_INVALID);
  invalid = valid;
  invalid.chain_id = "has-dash";
  EXPECT_EQ(codegen.Generate(invalid, result), PARAM_INVALID);
  invalid = valid;
  invalid.chain_id.assign(160U, 'a');
  EXPECT_EQ(codegen.Generate(invalid, result), PARAM_INVALID);
  invalid = valid;
  invalid.external_outputs.clear();
  EXPECT_EQ(codegen.Generate(invalid, result), PARAM_INVALID);
  invalid = valid;
  invalid.nodes[0] = nullptr;
  EXPECT_EQ(codegen.Generate(invalid, result), PARAM_INVALID);
  invalid = valid;
  invalid.nodes[1] = invalid.nodes[0];
  EXPECT_EQ(codegen.Generate(invalid, result), PARAM_INVALID);
  invalid = valid;
  invalid.external_inputs[0] = nullptr;
  EXPECT_EQ(codegen.Generate(invalid, result), PARAM_INVALID);
  invalid = valid;
  invalid.external_inputs.push_back(invalid.external_inputs[0]);
  EXPECT_EQ(codegen.Generate(invalid, result), PARAM_INVALID);
  invalid = valid;
  invalid.external_outputs[0].source = nullptr;
  EXPECT_EQ(codegen.Generate(invalid, result), PARAM_INVALID);
  invalid = valid;
  invalid.external_outputs.push_back(invalid.external_outputs[0]);
  EXPECT_EQ(codegen.Generate(invalid, result), PARAM_INVALID);

  invalid = valid;
  invalid.nodes[1]->GetOpDesc()->MutableOutputDesc(0)->SetShape(GeShape({-1}));
  EXPECT_EQ(codegen.Generate(invalid, result), UNSUPPORTED);
  invalid = valid;
  invalid.nodes[0]->GetOpDesc()->MutableOutputDesc(0)->SetShape(GeShape({-1}));
  EXPECT_EQ(codegen.Generate(invalid, result), UNSUPPORTED);
  invalid = valid;
  ASSERT_EQ(GraphUtils::RemoveEdge(invalid.nodes[0]->GetInDataAnchor(0)->GetPeerOutAnchor(),
                                   invalid.nodes[0]->GetInDataAnchor(0)),
            GRAPH_SUCCESS);
  EXPECT_EQ(codegen.Generate(invalid, result), UNSUPPORTED);

  EXPECT_EQ(GetHostCpuFusionInputName(nullptr, 1U), "input_1_tensor");
  const auto data = graph->FindNode("data");
  ASSERT_NE(data, nullptr);
  data->GetOpDesc()->MutableAllOutputName().clear();
  data->GetOpDesc()->MutableAllOutputName().emplace("with-dash", 0U);
  EXPECT_EQ(GetHostCpuFusionInputName(data->GetOutDataAnchor(0), 2U), "input_2_with_dash");
}

TEST(HostCpuFusionCodegenST, CompilerHandlesMissingToolkitAndDiagnostics) {
#if defined(__linux__)
  EnvGuard home("ASCEND_HOME_PATH", "");
  EnvGuard opp("ASCEND_OPP_PATH", "");
  HostCpuFusionCompiler compiler;
  std::vector<uint8_t> so_data;
  EXPECT_EQ(compiler.Compile("int value = 0;", so_data), UNSUPPORTED);
  EXPECT_TRUE(so_data.empty());

  const auto toolkit_home = GetToolkitHome();
  EnvGuard valid_home("ASCEND_HOME_PATH", toolkit_home.c_str());
  EXPECT_EQ(compiler.Compile("this is invalid C++;", so_data), UNSUPPORTED);
  EXPECT_TRUE(so_data.empty());
  {
    EnvGuard no_home("ASCEND_HOME_PATH", "");
    const auto toolkit_opp = GetToolkitOpp(toolkit_home);
    EnvGuard opp_only("ASCEND_OPP_PATH", toolkit_opp.c_str());
    EXPECT_EQ(compiler.Compile("", so_data), UNSUPPORTED);
  }
#else
  GTEST_SKIP() << "HostCPU fusion JIT uses Linux memfd.";
#endif
}

TEST(HostCpuFusionCodegenST, CommitsHostCpuFusionPassWithEmbeddedCustomOpSo) {
#if defined(__linux__)
  const auto toolkit_home = GetToolkitHome();
  EnvGuard home("ASCEND_HOME_PATH", toolkit_home.c_str());
  EnvGuard opp("ASCEND_OPP_PATH", "");
  const auto graph = BuildPassGraph();
  NodeEngineMap atomic_map;
  NodeEngineMap composite_map;
  HostCpuFusionPass pass(std::make_shared<MinimalCustomOpCompilerForPassSt>(), SupportAllForPassSt);
  ASSERT_EQ(pass.Run(graph, atomic_map, composite_map), SUCCESS);
  EXPECT_EQ(graph->FindNode("pass_first"), nullptr);
  EXPECT_EQ(graph->FindNode("pass_second"), nullptr);
  EXPECT_EQ(graph->GetDirectNodesSize(), 3U);
  EXPECT_EQ(atomic_map.size(), 1U);
  EXPECT_EQ(composite_map.size(), 1U);
#else
  GTEST_SKIP() << "HostCPU fusion JIT uses Linux memfd.";
#endif
}

}  // namespace ge
