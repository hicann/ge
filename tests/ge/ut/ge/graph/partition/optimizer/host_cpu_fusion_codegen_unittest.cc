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
#include <map>
#include <string>
#include <vector>

#include "graph/ge_local_context.h"
#include "graph/partition/optimizer/host_cpu_fusion_codegen.h"
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

void AddEdge(const NodePtr &source, const size_t source_index, const NodePtr &target, const size_t target_index) {
  ASSERT_EQ(GraphUtils::AddEdge(source->GetOutDataAnchor(static_cast<int32_t>(source_index)),
                                target->GetInDataAnchor(static_cast<int32_t>(target_index))),
            GRAPH_SUCCESS);
}

HostCpuFusionRegion BuildValidRegion(const ComputeGraphPtr &graph) {
  auto data = AddNode(graph, "data", "Data", 0U, 1U);
  auto first = AddNode(graph, "first", "HostA", 1U, 2U);
  auto second = AddNode(graph, "second", "HostA", 1U, 1U);
  auto device = AddNode(graph, "device", "Device", 1U, 1U);
  auto output = AddNode(graph, "output", "NetOutput", 2U, 0U);
  AddEdge(data, 0U, first, 0U);
  AddEdge(first, 0U, second, 0U);
  AddEdge(first, 1U, device, 0U);
  AddEdge(second, 0U, output, 0U);
  AddEdge(device, 0U, output, 1U);

  HostCpuFusionRegion region;
  region.chain_id = "coverage_chain";
  region.nodes = {first, second};
  region.external_inputs = {data->GetOutDataAnchor(0)};
  region.external_outputs = {{first->GetOutDataAnchor(1), {device->GetInDataAnchor(0)}},
                             {second->GetOutDataAnchor(0), {output->GetInDataAnchor(0)}}};
  return region;
}

class ScopedEnv final {
 public:
  ScopedEnv(const char *name, const char *value) : name_(name), old_(), had_old_(false) {
    const char *old = std::getenv(name);
    if (old != nullptr) {
      old_ = old;
      had_old_ = true;
    }
    EXPECT_EQ(setenv(name, value, 1), 0);
  }
  ~ScopedEnv() {
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

std::string GetNativeCpu() {
#if defined(__aarch64__)
  return "aarch64";
#elif defined(__x86_64__) || defined(__amd64__)
  return "x86_64";
#else
  return "";
#endif
}

std::string GetToolkitRoot(const std::string &home) {
  constexpr char kX86Suffix[] = "/x86_64-linux";
  constexpr char kAarch64Suffix[] = "/aarch64-linux";
  if (home.size() > (sizeof(kX86Suffix) - 1U) &&
      home.compare(home.size() - (sizeof(kX86Suffix) - 1U), sizeof(kX86Suffix) - 1U, kX86Suffix) == 0) {
    return home.substr(0U, home.size() - (sizeof(kX86Suffix) - 1U));
  }
  if (home.size() > (sizeof(kAarch64Suffix) - 1U) &&
      home.compare(home.size() - (sizeof(kAarch64Suffix) - 1U), sizeof(kAarch64Suffix) - 1U, kAarch64Suffix) == 0) {
    return home.substr(0U, home.size() - (sizeof(kAarch64Suffix) - 1U));
  }
  return home;
}

std::string GetHostCompilerLibraryPath(const std::string &toolkit_home) {
  const char *old = std::getenv("LIBRARY_PATH");
  if (GetNativeCpu() != "aarch64") {
    return old == nullptr ? "" : old;
  }

  const auto toolkit_root = GetToolkitRoot(toolkit_home);
  std::string library_path = toolkit_root + "/tools/hcc/aarch64-target-linux-gnu/lib64:" + toolkit_root +
                             "/tools/hcc/lib64/gcc/aarch64-target-linux-gnu/7.3.0";
  if ((old != nullptr) && (old[0] != '\0')) {
    library_path += ":";
    library_path += old;
  }
  return library_path;
}

class ScopedNativeHostEnv final {
 public:
  ScopedNativeHostEnv() : old_options_(GetThreadLocalContext().GetAllGraphOptions()) {
    auto options = old_options_;
    options["ge.host_env_os"] = "linux";
    const auto native_cpu = GetNativeCpu();
    if (!native_cpu.empty()) {
      options["ge.host_env_cpu"] = native_cpu;
    } else {
      options.erase("ge.host_env_cpu");
    }
    GetThreadLocalContext().SetGraphOption(options);
  }
  ~ScopedNativeHostEnv() {
    GetThreadLocalContext().SetGraphOption(old_options_);
  }

 private:
  std::map<std::string, std::string> old_options_;
};

}  // namespace

TEST(HostCpuFusionCodegenTest, GeneratesCurrentCustomOpSourceAndInternalBindings) {
  auto graph = std::make_shared<ComputeGraph>("codegen_coverage");
  auto region = BuildValidRegion(graph);
  region.nodes[0]->GetOpDesc()->SetName("first\n\"");
  HostCpuFusionCodegenResult result;

  ASSERT_EQ(HostCpuFusionCodegen().Generate(region, result), SUCCESS);
  EXPECT_EQ(result.register_name, "FusedHostCpu_coverage_chain");
  EXPECT_FALSE(result.source.empty());
  EXPECT_NE(result.source.find("HostCpuExecuteOp"), std::string::npos);
  EXPECT_NE(result.source.find("FusedHostCpuCustomOp_coverage_chain"), std::string::npos);
  EXPECT_NE(result.source.find("internal_tensor_0"), std::string::npos);
  EXPECT_NE(result.source.find("external_output_0"), std::string::npos);
  EXPECT_NE(result.source.find("external_output_1"), std::string::npos);
  EXPECT_NE(result.source.find("GetHostKernel_0"), std::string::npos);
  EXPECT_EQ(result.source.find("GetHostKernel_1"), std::string::npos);
  EXPECT_NE(result.source.find("REG_OP_BACKEND"), std::string::npos);
  EXPECT_NE(result.source.find("GetRegisteredCustomOpCreators"), std::string::npos);
  EXPECT_NE(result.source.find("first\\n\\\""), std::string::npos);
}

TEST(HostCpuFusionCodegenTest, CoversRegionValidationAndTensorFailures) {
  HostCpuFusionCodegen codegen;
  HostCpuFusionCodegenResult result;
  auto graph = std::make_shared<ComputeGraph>("codegen_validation");
  const auto valid = BuildValidRegion(graph);

  auto invalid = valid;
  invalid.nodes.resize(1U);
  EXPECT_EQ(codegen.Generate(invalid, result), PARAM_INVALID);
  invalid = valid;
  invalid.chain_id = "1invalid";
  EXPECT_EQ(codegen.Generate(invalid, result), PARAM_INVALID);
  invalid = valid;
  invalid.chain_id = "has-dash";
  EXPECT_EQ(codegen.Generate(invalid, result), PARAM_INVALID);
  invalid = valid;
  invalid.chain_id.assign(160U, 'a');
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
  invalid.external_inputs.clear();
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
}

TEST(HostCpuFusionCodegenTest, CoversInputNameSanitization) {
  EXPECT_EQ(GetHostCpuFusionInputName(nullptr, 0U), "input_0_tensor");
  auto graph = std::make_shared<ComputeGraph>("name_coverage");
  auto data = AddNode(graph, "data", "Data", 0U, 1U);
  ASSERT_NE(data, nullptr);
  data->GetOpDesc()->MutableAllOutputName().clear();
  data->GetOpDesc()->MutableAllOutputName().emplace("bad-name", 0U);
  EXPECT_EQ(GetHostCpuFusionInputName(data->GetOutDataAnchor(0), 2U), "input_2_bad_name");
  data->GetOpDesc()->MutableAllOutputName().clear();
  data->GetOpDesc()->MutableAllOutputName().emplace(std::string(100U, 'x'), 0U);
  EXPECT_EQ(GetHostCpuFusionInputName(data->GetOutDataAnchor(0), 3U).size(), 8U + 64U);
}

TEST(HostCpuFusionCodegenTest, CompilesSourceAndRejectsInvalidSource) {
#if defined(__linux__)
  const auto toolkit_home = GetToolkitHome();
  ScopedNativeHostEnv native_host_env;
  ScopedEnv home("ASCEND_HOME_PATH", toolkit_home.c_str());
  ScopedEnv opp("ASCEND_OPP_PATH", "");
  const auto library_path_value = GetHostCompilerLibraryPath(toolkit_home);
  ScopedEnv library_path("LIBRARY_PATH", library_path_value.c_str());
  HostCpuFusionCompiler compiler;
  std::vector<uint8_t> so_data;
  EXPECT_EQ(compiler.Compile("", so_data), UNSUPPORTED);
  EXPECT_TRUE(so_data.empty());
  ASSERT_EQ(compiler.Compile("extern \"C\" int codegen_test() { return 0; }", so_data), SUCCESS);
  ASSERT_FALSE(so_data.empty());
  EXPECT_EQ(so_data[0], 0x7FU);
  EXPECT_EQ(so_data[1], 'E');
  EXPECT_EQ(so_data[2], 'L');
  EXPECT_EQ(so_data[3], 'F');
  EXPECT_EQ(compiler.Compile("this is not valid C++;", so_data), UNSUPPORTED);
  EXPECT_TRUE(so_data.empty());
  {
    ScopedEnv no_home("ASCEND_HOME_PATH", "");
    const auto toolkit_opp = GetToolkitOpp(toolkit_home);
    ScopedEnv opp_only("ASCEND_OPP_PATH", toolkit_opp.c_str());
    EXPECT_EQ(compiler.Compile("", so_data), UNSUPPORTED);
  }
#else
  GTEST_SKIP() << "HostCPU fusion JIT uses Linux memfd.";
#endif
}

}  // namespace ge
