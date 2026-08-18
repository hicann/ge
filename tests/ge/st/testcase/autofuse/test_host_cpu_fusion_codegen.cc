/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdlib>
#include <limits>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "aicpu/cpu_kernels/cpu_kernel_register.h"
#include "common/env_path.h"
#include "framework/common/host_cpu_fusion_attr.h"
#include "graph/ge_local_context.h"
#include "graph/partition/optimizer/host_cpu_fusion_codegen.h"
#include "graph/partition/optimizer/host_cpu_fusion_pass.h"
#include "graph/utils/attr_utils.h"
#include "graph/utils/graph_utils.h"
#include "runtime/v2/engine/aicpu/kernel/aicpu_resource_manager.h"
#include "runtime/v2/engine/aicpu/kernel/fused_host_cpu_compute.h"

// The open-source ST links a placeholder constant-folding library, so provide the CpuKernel registration symbols
// that the generated JIT so normally resolves from the real libconstant_folding_ops.so on a product installation.
namespace aicpu {
namespace {
std::map<std::string, KERNEL_CREATOR_FUN> &GetTestCpuKernelCreators() {
  static std::map<std::string, KERNEL_CREATOR_FUN> creators;
  return creators;
}
}  // namespace

bool RegistCpuKernel(const std::string &type, const KERNEL_CREATOR_FUN &fun) {
  return GetTestCpuKernelCreators().emplace(type, fun).second;
}

CpuKernelRegister &CpuKernelRegister::Instance() {
  static CpuKernelRegister instance;
  return instance;
}

std::shared_ptr<CpuKernel> CpuKernelRegister::GetCpuKernel(const std::string &op_type) {
  const auto iter = GetTestCpuKernelCreators().find(op_type);
  return (iter == GetTestCpuKernelCreators().cend()) ? nullptr : iter->second();
}

std::string CpuKernelContext::GetOpType() const {
  return {};
}

Tensor *CpuKernelContext::Input(const uint32_t) const {
  return nullptr;
}

Tensor *CpuKernelContext::Output(const uint32_t) const {
  return nullptr;
}

uint32_t CpuKernelContext::GetInputsSize() const {
  return 0U;
}

uint32_t CpuKernelContext::GetOutputsSize() const {
  return 0U;
}

std::shared_ptr<TensorShape> Tensor::GetTensorShape() const {
  return nullptr;
}

DataType Tensor::GetDataType() const {
  return DT_UNDEFINED;
}

void *Tensor::GetData() const {
  return nullptr;
}

uint64_t Tensor::GetDataSize() const {
  return 0U;
}

Format TensorShape::GetFormat() const {
  return FORMAT_ND;
}

std::vector<int64_t> TensorShape::GetDimSizes() const {
  return {};
}

int64_t TensorShape::GetDimSize(int32_t) const {
  return 0;
}

int32_t TensorShape::GetDims() const {
  return 0;
}
}  // namespace aicpu

// The loader test does not load the product constant-folding library. Export the private chain-plan symbols so the
// generated JIT so can resolve them when validating registration; this test does not execute the plan.
size_t g_create_fused_chain_plan_count = 0U;

extern "C" __attribute__((visibility("default"))) void *CreateCpuConstantFoldingFusedChainPlan(const void *, size_t,
                                                                                               size_t, size_t) {
  ++g_create_fused_chain_plan_count;
  return reinterpret_cast<void *>(1U);
}

extern "C" __attribute__((visibility("default"))) int32_t RunCpuConstantFoldingFusedChainPlan(void *, uint32_t) {
  return 0;
}

extern "C" __attribute__((visibility("default"))) int32_t RunCpuConstantFoldingFusedChainPlanBindings(void *,
                                                                                                      const void *,
                                                                                                      uint32_t) {
  return 0;
}

extern "C" __attribute__((visibility("default"))) void DestroyCpuConstantFoldingFusedChainPlan(void *) {}

namespace ge {
namespace {
#if defined(__linux__)
constexpr char kHostFusionStA[] = "HostFusionStA";
constexpr char kHostFusionStB[] = "HostFusionStB";
constexpr char kHostFusionStMissing[] = "HostFusionStMissing";
constexpr char kHostFusionStAfterMissing[] = "HostFusionStAfterMissing";
constexpr char kHostCpuEngineName[] = "DNN_VM_HOST_CPU";
constexpr char kHostCpuKernelLibName[] = "DNN_VM_HOST_CPU_OP_STORE";

class ScopedEnvVar {
 public:
  ScopedEnvVar(std::string name, const std::string &value) : name_(std::move(name)) {
    const char *old_value = std::getenv(name_.c_str());
    if (old_value != nullptr) {
      had_old_value_ = true;
      old_value_ = old_value;
    }
    valid_ = (setenv(name_.c_str(), value.c_str(), 1) == 0);
  }

  ~ScopedEnvVar() {
    if (had_old_value_) {
      (void)setenv(name_.c_str(), old_value_.c_str(), 1);
    } else {
      (void)unsetenv(name_.c_str());
    }
  }

  bool IsValid() const {
    return valid_;
  }

 private:
  std::string name_;
  std::string old_value_;
  bool had_old_value_ = false;
  bool valid_ = false;
};

class StFakeCompiler final : public HostCpuFusionCompiler {
 public:
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
    return SUCCESS;
  }
};

NodePtr AddNode(const ComputeGraphPtr &graph, const std::string &name, const std::string &type,
                const size_t input_count, const size_t output_count) {
  const GeTensorDesc desc(GeShape({2}), FORMAT_ND, DT_INT64);
  auto op_desc = std::make_shared<OpDesc>(name, type);
  for (size_t i = 0U; i < input_count; ++i) {
    if (op_desc->AddInputDesc("x", desc) != GRAPH_SUCCESS) {
      return nullptr;
    }
  }
  for (size_t i = 0U; i < output_count; ++i) {
    if (op_desc->AddOutputDesc("y", desc) != GRAPH_SUCCESS) {
      return nullptr;
    }
  }
  return graph->AddNode(op_desc);
}

void MarkFusionCandidate(const NodePtr &node) {
  node->GetOpDesc()->SetOpEngineName(kHostCpuEngineName);
  node->GetOpDesc()->SetOpKernelLibName(kHostCpuKernelLibName);
  ASSERT_TRUE(AttrUtils::SetBool(node->GetOpDesc(), "SmallShapeHostcpu", true));
}

ComputeGraphPtr BuildCoverageChainGraph() {
  auto graph = std::make_shared<ComputeGraph>("host_cpu_coverage_chain");
  auto data = AddNode(graph, "data", "Data", 0U, 1U);
  auto first = AddNode(graph, "first", kHostFusionStA, 1U, 1U);
  auto second = AddNode(graph, "second", kHostFusionStB, 1U, 1U);
  auto output = AddNode(graph, "output", "NetOutput", 1U, 0U);
  if ((data == nullptr) || (first == nullptr) || (second == nullptr) || (output == nullptr)) {
    return nullptr;
  }
  if ((GraphUtils::AddEdge(data->GetOutDataAnchor(0), first->GetInDataAnchor(0)) != GRAPH_SUCCESS) ||
      (GraphUtils::AddEdge(first->GetOutDataAnchor(0), second->GetInDataAnchor(0)) != GRAPH_SUCCESS) ||
      (GraphUtils::AddEdge(second->GetOutDataAnchor(0), output->GetInDataAnchor(0)) != GRAPH_SUCCESS)) {
    return nullptr;
  }
  MarkFusionCandidate(first);
  MarkFusionCandidate(second);
  return graph;
}

HostCpuFusionRegion BuildCoverageRegion(const ComputeGraphPtr &graph) {
  HostCpuFusionRegion region;
  region.chain_id = "st_coverage_region";
  const auto data = graph->FindNode("data");
  const auto first = graph->FindNode("first");
  const auto second = graph->FindNode("second");
  const auto output = graph->FindNode("output");
  region.nodes = {first, second};
  region.external_inputs = {data->GetOutDataAnchor(0)};
  region.external_outputs = {{second->GetOutDataAnchor(0), {output->GetInDataAnchor(0)}}};
  return region;
}
#endif
}  // namespace

// 用例描述：验证 HostCPU 编排源码可真实编译，并注册到共享 CpuKernelRegister。
// 预置条件：Linux 环境已安装可用 Toolkit 头文件和目标架构 g++，测试提供 CpuKernelRegister 符号。
// 测试步骤：生成两节点融合源码并编译，同一 so 加载两次，创建状态后首次绑定执行，再释放引用并复用。
// 预期结果：模型加载期不创建整链 plan，首次执行才创建一次；引用归零后 so 仍可重新加载。
TEST(HostCpuFusionCodegenST, CompilesLoadsAndKeepsCpuKernelRegistered) {
#if !defined(__linux__)
  GTEST_SKIP() << "HostCPU fusion JIT uses Linux memfd.";
#else
  const std::string opp_path = EnvPath().GetAscendInstallPath() + "/opp";
  const ScopedEnvVar opp_env("ASCEND_OPP_PATH", opp_path);
  const ScopedEnvVar home_env("ASCEND_HOME_PATH", EnvPath().GetAirBasePath());
  ASSERT_TRUE(opp_env.IsValid());
  ASSERT_TRUE(home_env.IsValid());

  auto graph = std::make_shared<ComputeGraph>("host_cpu_fusion_st");
  auto data = AddNode(graph, "data", "Data", 0U, 1U);
  auto first = AddNode(graph, "first", kHostFusionStA, 1U, 1U);
  auto second = AddNode(graph, "second", kHostFusionStB, 1U, 1U);
  auto output = AddNode(graph, "output", "NetOutput", 1U, 0U);
  ASSERT_NE(data, nullptr);
  ASSERT_NE(first, nullptr);
  ASSERT_NE(second, nullptr);
  ASSERT_NE(output, nullptr);
  ASSERT_EQ(GraphUtils::AddEdge(data->GetOutDataAnchor(0), first->GetInDataAnchor(0)), GRAPH_SUCCESS);
  ASSERT_EQ(GraphUtils::AddEdge(first->GetOutDataAnchor(0), second->GetInDataAnchor(0)), GRAPH_SUCCESS);
  ASSERT_EQ(GraphUtils::AddEdge(second->GetOutDataAnchor(0), output->GetInDataAnchor(0)), GRAPH_SUCCESS);

  HostCpuFusionRegion region;
  region.chain_id = "st_actual_compile";
  region.nodes = {first, second};
  region.external_inputs = {data->GetOutDataAnchor(0)};
  region.external_outputs = {{second->GetOutDataAnchor(0), {output->GetInDataAnchor(0)}}};
  HostCpuFusionCodegenResult result;
  HostCpuFusionCodegen codegen;
  ASSERT_EQ(codegen.Generate(region, result), SUCCESS);
  HostCpuFusionCompiler compiler;
  ASSERT_EQ(compiler.Compile(result.source, result.so_data), SUCCESS);
  ASSERT_EQ(gert::AicpuResourceManager::GetInstance().LoadFusedHostCpuSo(result.register_name, result.so_data.data(),
                                                                         result.so_data.size()),
            GRAPH_SUCCESS);
  ASSERT_EQ(gert::AicpuResourceManager::GetInstance().LoadFusedHostCpuSo(result.register_name, result.so_data.data(),
                                                                         result.so_data.size()),
            GRAPH_SUCCESS);

  EXPECT_NE(aicpu::CpuKernelRegister::Instance().GetCpuKernel(result.register_name), nullptr);
  const auto kernel_funcs =
      gert::AicpuResourceManager::GetInstance().GetFusedHostCpuKernelFunctions(result.register_name);
  ASSERT_NE(kernel_funcs.create_func, nullptr);
  ASSERT_NE(kernel_funcs.destroy_func, nullptr);
  ASSERT_NE(kernel_funcs.run_func, nullptr);
  g_create_fused_chain_plan_count = 0U;
  void *kernel_state = kernel_funcs.create_func();
  ASSERT_NE(kernel_state, nullptr);
  EXPECT_EQ(g_create_fused_chain_plan_count, 0U);
  int64_t dims[] = {2};
  int64_t input_data[] = {1, 2};
  int64_t output_data[] = {0, 0};
  gert::FusedHostCpuTensorBinding bindings[] = {
      {dims, reinterpret_cast<uint8_t *>(input_data), 1U, sizeof(input_data),
       gert::kFusedHostCpuShapeChanged | gert::kFusedHostCpuDataChanged},
      {dims, reinterpret_cast<uint8_t *>(output_data), 1U, sizeof(output_data),
       gert::kFusedHostCpuShapeChanged | gert::kFusedHostCpuDataChanged}};
  EXPECT_EQ(
      kernel_funcs.run_func(kernel_state, bindings, gert::kFusedHostCpuShapeChanged | gert::kFusedHostCpuDataChanged),
      0U);
  EXPECT_EQ(g_create_fused_chain_plan_count, 1U);
  // 绑定未变化时应复用已创建的 plan，不重复构建内部执行计划。
  EXPECT_EQ(kernel_funcs.run_func(kernel_state, bindings, 0U), 0U);
  EXPECT_EQ(g_create_fused_chain_plan_count, 1U);
  kernel_funcs.destroy_func(kernel_state);
  ASSERT_EQ(gert::AicpuResourceManager::GetInstance().ReleaseFusedHostCpuSo(result.register_name), GRAPH_SUCCESS);
  EXPECT_NE(aicpu::CpuKernelRegister::Instance().GetCpuKernel(result.register_name), nullptr);
  ASSERT_EQ(gert::AicpuResourceManager::GetInstance().ReleaseFusedHostCpuSo(result.register_name), GRAPH_SUCCESS);
  EXPECT_NE(aicpu::CpuKernelRegister::Instance().GetCpuKernel(result.register_name), nullptr);
  EXPECT_EQ(gert::AicpuResourceManager::GetInstance().ReleaseFusedHostCpuSo(result.register_name), ge::PARAM_INVALID);
  ASSERT_EQ(gert::AicpuResourceManager::GetInstance().LoadFusedHostCpuSo(result.register_name, result.so_data.data(),
                                                                         result.so_data.size()),
            GRAPH_SUCCESS);
  ASSERT_EQ(gert::AicpuResourceManager::GetInstance().ReleaseFusedHostCpuSo(result.register_name), GRAPH_SUCCESS);
#endif
}

// 用例描述：验证同一进程连续加载两个不同的 HostCPU 融合 SO。
// 预置条件：Linux 环境已安装可用 Toolkit 头文件和目标架构 g++，测试提供 CpuKernelRegister 符号。
// 测试步骤：为同一融合区域生成两个不同注册名的 SO，依次加载并查询各自的 private C ABI。
// 预期结果：两个 SO 均独立完成静态注册，且分别返回有效的 private C ABI。
TEST(HostCpuFusionCodegenST, LoadsDifferentFusedSharedObjectsIndependently) {
#if !defined(__linux__)
  GTEST_SKIP() << "HostCPU fusion JIT uses Linux memfd.";
#else
  const std::string opp_path = EnvPath().GetAscendInstallPath() + "/opp";
  const ScopedEnvVar opp_env("ASCEND_OPP_PATH", opp_path);
  const ScopedEnvVar home_env("ASCEND_HOME_PATH", EnvPath().GetAirBasePath());
  ASSERT_TRUE(opp_env.IsValid());
  ASSERT_TRUE(home_env.IsValid());

  auto graph = std::make_shared<ComputeGraph>("host_cpu_fusion_multi_so_st");
  auto data = AddNode(graph, "data", "Data", 0U, 1U);
  auto first = AddNode(graph, "first", kHostFusionStA, 1U, 1U);
  auto second = AddNode(graph, "second", kHostFusionStB, 1U, 1U);
  auto output = AddNode(graph, "output", "NetOutput", 1U, 0U);
  ASSERT_NE(data, nullptr);
  ASSERT_NE(first, nullptr);
  ASSERT_NE(second, nullptr);
  ASSERT_NE(output, nullptr);
  ASSERT_EQ(GraphUtils::AddEdge(data->GetOutDataAnchor(0), first->GetInDataAnchor(0)), GRAPH_SUCCESS);
  ASSERT_EQ(GraphUtils::AddEdge(first->GetOutDataAnchor(0), second->GetInDataAnchor(0)), GRAPH_SUCCESS);
  ASSERT_EQ(GraphUtils::AddEdge(second->GetOutDataAnchor(0), output->GetInDataAnchor(0)), GRAPH_SUCCESS);

  HostCpuFusionRegion region;
  region.nodes = {first, second};
  region.external_inputs = {data->GetOutDataAnchor(0)};
  region.external_outputs = {{second->GetOutDataAnchor(0), {output->GetInDataAnchor(0)}}};
  HostCpuFusionCodegen codegen;
  HostCpuFusionCompiler compiler;
  HostCpuFusionCodegenResult first_result;
  region.chain_id = "st_multi_so_first";
  ASSERT_EQ(codegen.Generate(region, first_result), SUCCESS);
  ASSERT_EQ(compiler.Compile(first_result.source, first_result.so_data), SUCCESS);
  HostCpuFusionCodegenResult second_result;
  region.chain_id = "st_multi_so_second";
  ASSERT_EQ(codegen.Generate(region, second_result), SUCCESS);
  ASSERT_EQ(compiler.Compile(second_result.source, second_result.so_data), SUCCESS);
  ASSERT_NE(first_result.so_data, second_result.so_data);

  ASSERT_EQ(gert::AicpuResourceManager::GetInstance().LoadFusedHostCpuSo(
                first_result.register_name, first_result.so_data.data(), first_result.so_data.size()),
            GRAPH_SUCCESS);
  ASSERT_EQ(gert::AicpuResourceManager::GetInstance().LoadFusedHostCpuSo(
                second_result.register_name, second_result.so_data.data(), second_result.so_data.size()),
            GRAPH_SUCCESS);
  const gert::FusedHostCpuKernelFunctions first_funcs =
      gert::AicpuResourceManager::GetInstance().GetFusedHostCpuKernelFunctions(first_result.register_name);
  const gert::FusedHostCpuKernelFunctions second_funcs =
      gert::AicpuResourceManager::GetInstance().GetFusedHostCpuKernelFunctions(second_result.register_name);
  EXPECT_NE(first_funcs.create_func, nullptr);
  EXPECT_NE(first_funcs.destroy_func, nullptr);
  EXPECT_NE(first_funcs.run_func, nullptr);
  EXPECT_NE(second_funcs.create_func, nullptr);
  EXPECT_NE(second_funcs.destroy_func, nullptr);
  EXPECT_NE(second_funcs.run_func, nullptr);
  EXPECT_NE(aicpu::CpuKernelRegister::Instance().GetCpuKernel(first_result.register_name), nullptr);
  EXPECT_NE(aicpu::CpuKernelRegister::Instance().GetCpuKernel(second_result.register_name), nullptr);
  EXPECT_EQ(gert::AicpuResourceManager::GetInstance().ReleaseFusedHostCpuSo(first_result.register_name), GRAPH_SUCCESS);
  EXPECT_EQ(gert::AicpuResourceManager::GetInstance().ReleaseFusedHostCpuSo(second_result.register_name),
            GRAPH_SUCCESS);
#endif
}

// 用例描述：验证融合外层与内部原算子都不再依赖 HostCPU registry，稳态执行复用 CpuKernel plan。
// 预置条件：Linux 环境已安装可用 Toolkit 头文件和目标架构 g++。
// 测试步骤：生成三节点融合源码并编译，检查外层注册及内部 plan 创建、复用调用。
// 预期结果：外层使用 REGISTER_CPU_KERNEL；内部使用数组和线程级 plan，不包含 map 或 HostCpuOp API。
TEST(HostCpuFusionCodegenST, UsesCachedCpuKernelPlanForInternalNodes) {
#if !defined(__linux__)
  GTEST_SKIP() << "HostCPU fusion JIT uses Linux memfd.";
#else
  const std::string opp_path = EnvPath().GetAscendInstallPath() + "/opp";
  const ScopedEnvVar opp_env("ASCEND_OPP_PATH", opp_path);
  const ScopedEnvVar home_env("ASCEND_HOME_PATH", EnvPath().GetAirBasePath());
  ASSERT_TRUE(opp_env.IsValid());
  ASSERT_TRUE(home_env.IsValid());

  auto graph = std::make_shared<ComputeGraph>("host_cpu_fusion_missing_kernel_st");
  auto data = AddNode(graph, "data", "Data", 0U, 1U);
  auto first = AddNode(graph, "first", kHostFusionStA, 1U, 1U);
  auto missing = AddNode(graph, "missing", kHostFusionStMissing, 1U, 1U);
  auto after_missing = AddNode(graph, "after_missing", kHostFusionStAfterMissing, 1U, 1U);
  auto output = AddNode(graph, "output", "NetOutput", 1U, 0U);
  ASSERT_NE(data, nullptr);
  ASSERT_NE(first, nullptr);
  ASSERT_NE(missing, nullptr);
  ASSERT_NE(after_missing, nullptr);
  ASSERT_NE(output, nullptr);
  ASSERT_EQ(GraphUtils::AddEdge(data->GetOutDataAnchor(0), first->GetInDataAnchor(0)), GRAPH_SUCCESS);
  ASSERT_EQ(GraphUtils::AddEdge(first->GetOutDataAnchor(0), missing->GetInDataAnchor(0)), GRAPH_SUCCESS);
  ASSERT_EQ(GraphUtils::AddEdge(missing->GetOutDataAnchor(0), after_missing->GetInDataAnchor(0)), GRAPH_SUCCESS);
  ASSERT_EQ(GraphUtils::AddEdge(after_missing->GetOutDataAnchor(0), output->GetInDataAnchor(0)), GRAPH_SUCCESS);

  HostCpuFusionRegion region;
  region.chain_id = "st_missing_kernel";
  region.nodes = {first, missing, after_missing};
  region.external_inputs = {data->GetOutDataAnchor(0)};
  region.external_outputs = {{after_missing->GetOutDataAnchor(0), {output->GetInDataAnchor(0)}}};
  HostCpuFusionCodegenResult result;
  HostCpuFusionCodegen codegen;
  ASSERT_EQ(codegen.Generate(region, result), SUCCESS);
  HostCpuFusionCompiler compiler;
  ASSERT_EQ(compiler.Compile(result.source, result.so_data), SUCCESS);
  EXPECT_NE(result.source.find("REGISTER_CPU_KERNEL(kFusedHostCpuKernel_st_missing_kernel"), std::string::npos);
  EXPECT_EQ(result.source.find("REGISTER_HOST_CPU_OP_BUILDER"), std::string::npos);
  EXPECT_EQ(result.source.find("host_cpu_kernel_registry.h"), std::string::npos);
  EXPECT_EQ(result.source.find("HostCpuOp"), std::string::npos);
  EXPECT_EQ(result.source.find("CreateHostCpuOp"), std::string::npos);
  EXPECT_EQ(result.source.find("dlsym"), std::string::npos);
  EXPECT_EQ(result.source.find("UpdateInputDesc"), std::string::npos);
  EXPECT_EQ(result.source.find("UpdateOutputDesc"), std::string::npos);
  EXPECT_EQ(result.source.find("std::map"), std::string::npos);
  EXPECT_NE(result.source.find("FusedHostCpuChainPlanGuard chain_plan_"), std::string::npos);
  EXPECT_NE(result.source.find("CreateCpuConstantFoldingFusedChainPlan("), std::string::npos);
  EXPECT_NE(result.source.find("node_descs.data(), node_descs.size(), 1U, 1U"), std::string::npos);
  EXPECT_NE(result.source.find("RunCpuConstantFoldingFusedChainPlan(chain_plan_.Get(), binding_flags)"),
            std::string::npos);
  EXPECT_NE(result.source.find("RunCpuConstantFoldingFusedChainPlanBindings("), std::string::npos);
  EXPECT_NE(result.source.find("FusedHostCpuTensorState"), std::string::npos);
  EXPECT_NE(result.source.find("FusedHostCpuTensorBinding"), std::string::npos);
  EXPECT_NE(result.source.find("ComputeBindings"), std::string::npos);
  EXPECT_NE(result.source.find("BuildFusedHostCpuRuntimeTensor"), std::string::npos);
  EXPECT_NE(result.source.find("InitializeBindings"), std::string::npos);
  EXPECT_EQ(result.source.find("BindFusedHostCpuTensor"), std::string::npos);
  EXPECT_EQ(result.source.find("if ((binding_flags == 0U) && runtime_bound_) { return Run(0U); }"), std::string::npos);
  EXPECT_EQ(result.source.find("binding_flags |= inputs["), std::string::npos);
  EXPECT_EQ(result.source.find("binding_flags |= outputs["), std::string::npos);
  EXPECT_NE(result.source.find("HasSameFusedHostCpuShape"), std::string::npos);
  EXPECT_EQ(result.source.find("GetDimSizes"), std::string::npos);
  EXPECT_NE(result.source.find("bool bindings_changed = false"), std::string::npos);
  EXPECT_NE(result.source.find("node_input_binding_indices_0{{0}}"), std::string::npos);
  EXPECT_NE(result.source.find("node_output_binding_indices_0{{-1}}"), std::string::npos);
  EXPECT_NE(result.source.find("node_input_binding_indices_1{{-1}}"), std::string::npos);
  EXPECT_NE(result.source.find("node_output_binding_indices_1{{-1}}"), std::string::npos);
  EXPECT_NE(result.source.find("node_input_binding_indices_2{{-1}}"), std::string::npos);
  EXPECT_NE(result.source.find("node_output_binding_indices_2{{1}}"), std::string::npos);
  EXPECT_EQ(result.source.find("RunCpuConstantFoldingFusedPlan"), std::string::npos);
  EXPECT_NE(result.source.find("static thread_local ge::FusedHostCpuOrchestration_st_missing_kernel orchestration"),
            std::string::npos);
  EXPECT_NE(result.source.find("void *CreateFusedHostCpuKernelState()"), std::string::npos);
  EXPECT_NE(result.source.find("if (state == nullptr) { return nullptr; }"), std::string::npos);
  EXPECT_EQ(result.source.find("state->Initialize()"), std::string::npos);
  EXPECT_NE(result.source.find("void DestroyFusedHostCpuKernelState(void *kernel_state)"), std::string::npos);
  EXPECT_NE(result.source.find("uint32_t RunFusedHostCpuKernel(void *kernel_state, const void *binding_data"),
            std::string::npos);
  EXPECT_NE(result.source.find("const bool bindings_changed"), std::string::npos);
#endif
}

// 用例描述：验证 HostCpuFusionPass 从候选扫描、真实 JIT 到图提交和运行时加载的完整链路。
// 预置条件：Linux 环境已安装可用 Toolkit 头文件和目标架构 g++，测试提供融合 SO 所需符号。
// 测试步骤：构造两个 HostCPU 候选节点，运行融合 Pass，从根图读取生成 SO 并交给资源管理器加载。
// 预期结果：原节点被一个 FusedHostCpu 替换，SO 属性完整，注册名可成功加载和释放。
TEST(HostCpuFusionCodegenST, PassCompilesCommitsAndLoadsFusedSharedObject) {
#if !defined(__linux__)
  GTEST_SKIP() << "HostCPU fusion JIT uses Linux memfd.";
#else
  const std::string opp_path = EnvPath().GetAscendInstallPath() + "/opp";
  const ScopedEnvVar opp_env("ASCEND_OPP_PATH", opp_path);
  const ScopedEnvVar home_env("ASCEND_HOME_PATH", EnvPath().GetAirBasePath());
  ASSERT_TRUE(opp_env.IsValid());
  ASSERT_TRUE(home_env.IsValid());

  auto graph = std::make_shared<ComputeGraph>("host_cpu_fusion_pass_st");
  auto data = AddNode(graph, "data", "Data", 0U, 1U);
  auto first = AddNode(graph, "first", kHostFusionStA, 1U, 1U);
  auto second = AddNode(graph, "second", kHostFusionStB, 1U, 1U);
  auto output = AddNode(graph, "output", "NetOutput", 1U, 0U);
  ASSERT_NE(data, nullptr);
  ASSERT_NE(first, nullptr);
  ASSERT_NE(second, nullptr);
  ASSERT_NE(output, nullptr);
  ASSERT_EQ(GraphUtils::AddEdge(data->GetOutDataAnchor(0), first->GetInDataAnchor(0)), GRAPH_SUCCESS);
  ASSERT_EQ(GraphUtils::AddEdge(first->GetOutDataAnchor(0), second->GetInDataAnchor(0)), GRAPH_SUCCESS);
  ASSERT_EQ(GraphUtils::AddEdge(second->GetOutDataAnchor(0), output->GetInDataAnchor(0)), GRAPH_SUCCESS);
  MarkFusionCandidate(first);
  MarkFusionCandidate(second);

  NodeEngineMap atomic_map;
  NodeEngineMap composite_map;
  HostCpuFusionPass pass(std::make_shared<HostCpuFusionCompiler>(), [](const std::string &) { return true; });
  ASSERT_EQ(pass.Run(graph, atomic_map, composite_map), SUCCESS);
  EXPECT_EQ(graph->FindNode("first"), nullptr);
  EXPECT_EQ(graph->FindNode("second"), nullptr);

  NodePtr fused_node;
  for (const auto &node : graph->GetDirectNode()) {
    if (node->GetType() == kFusedHostCpuOpType) {
      fused_node = node;
      break;
    }
  }
  ASSERT_NE(fused_node, nullptr);
  ASSERT_EQ(atomic_map.size(), 1U);
  ASSERT_EQ(composite_map.size(), 1U);
  EXPECT_EQ(output->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode(), fused_node);

  std::string register_name;
  std::string so_key;
  Buffer so_data;
  ASSERT_TRUE(AttrUtils::GetStr(fused_node->GetOpDesc(), kFusedHostCpuRegisterName, register_name));
  ASSERT_TRUE(AttrUtils::GetStr(fused_node->GetOpDesc(), kFusedHostCpuSoKey, so_key));
  ASSERT_TRUE(AttrUtils::GetBytes(graph, so_key, so_data));
  ASSERT_GT(so_data.GetSize(), 0U);
  ASSERT_EQ(
      gert::AicpuResourceManager::GetInstance().LoadFusedHostCpuSo(register_name, so_data.GetData(), so_data.GetSize()),
      GRAPH_SUCCESS);
  EXPECT_NE(aicpu::CpuKernelRegister::Instance().GetCpuKernel(register_name), nullptr);
  EXPECT_EQ(gert::AicpuResourceManager::GetInstance().ReleaseFusedHostCpuSo(register_name), GRAPH_SUCCESS);
#endif
}

// 用例描述：验证 JIT 编译器对空源码和编译错误源码均安全回退。
// 预置条件：Linux 环境已安装可用 Toolkit 头文件和目标架构 g++。
// 测试步骤：依次提交空源码和语法错误源码，并检查返回码及输出缓存。
// 预期结果：两次调用均返回 UNSUPPORTED，且不会留下伪造的共享库数据。
TEST(HostCpuFusionCodegenST, CompilerRejectsEmptyAndInvalidSources) {
#if !defined(__linux__)
  GTEST_SKIP() << "HostCPU fusion JIT uses Linux memfd.";
#else
  const std::string opp_path = EnvPath().GetAscendInstallPath() + "/opp";
  const ScopedEnvVar opp_env("ASCEND_OPP_PATH", opp_path);
  const ScopedEnvVar home_env("ASCEND_HOME_PATH", EnvPath().GetAirBasePath());
  ASSERT_TRUE(opp_env.IsValid());
  ASSERT_TRUE(home_env.IsValid());

  HostCpuFusionCompiler compiler;
  std::vector<uint8_t> so_data{1U};
  EXPECT_EQ(compiler.Compile("", so_data), UNSUPPORTED);
  EXPECT_TRUE(so_data.empty());
  EXPECT_EQ(compiler.Compile("this is not valid C++;", so_data), UNSUPPORTED);
  EXPECT_TRUE(so_data.empty());
#endif
}

// 用例描述：覆盖 JIT 工具链路径为空、缺失头文件和不同目标 CPU 的编译器选择。
// 预置条件：Linux 环境；不要求实际 Toolkit 头文件存在。
// 测试步骤：清空或设置无效 Toolkit 路径，分别设置 aarch64、x86_64 和未知目标 CPU。
// 预期结果：编译在头文件校验前安全返回 UNSUPPORTED，不启动外部编译器。
TEST(HostCpuFusionCodegenST, CompilerRejectsMissingToolkitAndSelectsTargetCompiler) {
#if !defined(__linux__)
  GTEST_SKIP() << "HostCPU fusion JIT uses Linux memfd.";
#else
  const auto original_options = GetThreadLocalContext().GetAllGlobalOptions();
  const ScopedEnvVar empty_opp("ASCEND_OPP_PATH", "");
  const ScopedEnvVar empty_home("ASCEND_HOME_PATH", "");
  ASSERT_TRUE(empty_opp.IsValid());
  ASSERT_TRUE(empty_home.IsValid());
  HostCpuFusionCompiler compiler;
  std::vector<uint8_t> so_data;

  GetThreadLocalContext().SetGlobalOption({{OPTION_HOST_ENV_CPU, "aarch64"}});
  EXPECT_EQ(compiler.Compile("int value = 0;", so_data), UNSUPPORTED);
  GetThreadLocalContext().SetGlobalOption({{OPTION_HOST_ENV_CPU, "x86_64"}});
  EXPECT_EQ(compiler.Compile("int value = 0;", so_data), UNSUPPORTED);
  GetThreadLocalContext().SetGlobalOption({{OPTION_HOST_ENV_CPU, "unsupported_cpu"}});
  EXPECT_EQ(compiler.Compile("int value = 0;", so_data), UNSUPPORTED);
  GetThreadLocalContext().SetGlobalOption(original_options);

  const ScopedEnvVar invalid_opp("ASCEND_OPP_PATH", "/tmp/nonexistent/opp/");
  const ScopedEnvVar invalid_home("ASCEND_HOME_PATH", "/tmp/nonexistent/home/");
  ASSERT_TRUE(invalid_opp.IsValid());
  ASSERT_TRUE(invalid_home.IsValid());
  EXPECT_EQ(compiler.Compile("int value = 0;", so_data), UNSUPPORTED);
#endif
}

// 用例描述：覆盖融合源码生成器的输入校验、描述符序列化和属性序列化边界。
// 预置条件：Linux 环境可构造 GE 图；本用例不依赖实际设备执行。
// 测试步骤：构造有效融合区域，再逐项注入非法节点、边、名称、Shape 和属性。
// 预期结果：非法输入均返回 PARAM_INVALID 或 UNSUPPORTED，合法输入能生成稳定源码。
TEST(HostCpuFusionCodegenST, CoversCodegenValidationAndDescriptorBoundaries) {
#if !defined(__linux__)
  GTEST_SKIP() << "HostCPU fusion JIT uses Linux memfd.";
#else
  const auto graph = BuildCoverageChainGraph();
  ASSERT_NE(graph, nullptr);
  const auto valid = BuildCoverageRegion(graph);
  HostCpuFusionCodegen codegen;
  HostCpuFusionCodegenResult result;
  ASSERT_EQ(codegen.Generate(valid, result), SUCCESS);
  EXPECT_FALSE(result.source.empty());

  auto invalid = valid;
  invalid.nodes.resize(1U);
  EXPECT_EQ(codegen.Generate(invalid, result), PARAM_INVALID);
  invalid = valid;
  invalid.chain_id = "1bad";
  EXPECT_EQ(codegen.Generate(invalid, result), PARAM_INVALID);
  invalid = valid;
  invalid.chain_id = "bad-name";
  EXPECT_EQ(codegen.Generate(invalid, result), PARAM_INVALID);
  invalid = valid;
  invalid.chain_id.assign(200U, 'x');
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

  const auto missing_peer_graph = BuildCoverageChainGraph();
  ASSERT_NE(missing_peer_graph, nullptr);
  ASSERT_EQ(GraphUtils::RemoveEdge(missing_peer_graph->FindNode("data")->GetOutDataAnchor(0),
                                   missing_peer_graph->FindNode("first")->GetInDataAnchor(0)),
            GRAPH_SUCCESS);
  EXPECT_EQ(codegen.Generate(BuildCoverageRegion(missing_peer_graph), result), UNSUPPORTED);

  const auto invalid_name_graph = BuildCoverageChainGraph();
  ASSERT_NE(invalid_name_graph, nullptr);
  invalid_name_graph->FindNode("first")->GetOpDesc()->MutableAllInputName().clear();
  EXPECT_EQ(codegen.Generate(BuildCoverageRegion(invalid_name_graph), result), UNSUPPORTED);

  const auto invalid_output_name_graph = BuildCoverageChainGraph();
  ASSERT_NE(invalid_output_name_graph, nullptr);
  invalid_output_name_graph->FindNode("second")->GetOpDesc()->MutableAllOutputName().clear();
  EXPECT_EQ(codegen.Generate(BuildCoverageRegion(invalid_output_name_graph), result), UNSUPPORTED);

  const auto invalid_size_graph = BuildCoverageChainGraph();
  ASSERT_NE(invalid_size_graph, nullptr);
  invalid_size_graph->FindNode("first")->GetOpDesc()->MutableOutputDesc(0)->SetShape(
      GeShape({std::numeric_limits<int64_t>::max(), 2}));
  EXPECT_EQ(codegen.Generate(BuildCoverageRegion(invalid_size_graph), result), UNSUPPORTED);

  const auto scalar_graph = BuildCoverageChainGraph();
  ASSERT_NE(scalar_graph, nullptr);
  auto first_desc = scalar_graph->FindNode("first")->GetOpDesc();
  first_desc->MutableOutputDesc(0)->SetShape(GeShape(std::vector<int64_t>{}));
  first_desc->MutableOutputDesc(0)->SetOriginShape(GeShape({3, 4}));
  ASSERT_EQ(first_desc->MutableOutputDesc(0)->SetShapeRange({{1, 3}, {2, 4}}), GRAPH_SUCCESS);
  ASSERT_TRUE(AttrUtils::SetInt(first_desc, "axis", 1));
  ASSERT_TRUE(AttrUtils::SetFloat(first_desc, "scale", 2.0F));
  ASSERT_TRUE(AttrUtils::SetBool(first_desc, "keep", true));
  ASSERT_TRUE(AttrUtils::SetStr(first_desc, "label", "value\r\t"));
  ASSERT_TRUE(AttrUtils::SetListInt(first_desc, "sizes", {1, 2}));
  ASSERT_TRUE(AttrUtils::SetListFloat(first_desc, "ratios", {0.25F, 0.75F}));
  for (const auto &name : {"axis", "scale", "keep", "label", "sizes", "ratios"}) {
    first_desc->AppendIrAttrName(name);
  }
  ASSERT_EQ(codegen.Generate(BuildCoverageRegion(scalar_graph), result), SUCCESS);
  EXPECT_NE(result.source.find("SetOriginShape"), std::string::npos);
  EXPECT_NE(result.source.find("SetShapeRange"), std::string::npos);
  EXPECT_NE(result.source.find("std::vector<int64_t>{1LL, 2LL}"), std::string::npos);
  EXPECT_NE(result.source.find("std::vector<float>{0.25F, 0.75F}"), std::string::npos);

  const auto unsupported_type_graph = BuildCoverageChainGraph();
  ASSERT_NE(unsupported_type_graph, nullptr);
  unsupported_type_graph->FindNode("first")->GetOpDesc()->MutableOutputDesc(0)->SetShape(
      GeShape(std::vector<int64_t>{}));
  unsupported_type_graph->FindNode("first")->GetOpDesc()->MutableOutputDesc(0)->SetDataType(DT_UNDEFINED);
  EXPECT_EQ(codegen.Generate(BuildCoverageRegion(unsupported_type_graph), result), UNSUPPORTED);
#endif
}

// 用例描述：覆盖 HostCPU 融合 Pass 的候选拒绝和提交回滚路径。
// 预置条件：使用假的编译器返回合法 ELF，避免测试依赖 JIT 工具链。
// 测试步骤：构造候选链，分别注入不支持属性、控制边、Shape 边界和重复融合节点。
// 预期结果：Pass 保持原图不变，并返回对应状态。
TEST(HostCpuFusionCodegenST, CoversFusionPassCandidateAndRollbackPaths) {
#if !defined(__linux__)
  GTEST_SKIP() << "HostCPU fusion JIT uses Linux memfd.";
#else
  const auto graph = BuildCoverageChainGraph();
  ASSERT_NE(graph, nullptr);
  HostCpuFusionPass pass(std::make_shared<StFakeCompiler>(), [](const std::string &) { return false; });
  std::vector<std::vector<HostCpuFusionRegion> > components;
  EXPECT_EQ(pass.BuildFusionRegions(graph, components), NOT_CHANGED);
  EXPECT_TRUE(components.empty());

  const auto control_graph = BuildCoverageChainGraph();
  ASSERT_NE(control_graph, nullptr);
  ASSERT_EQ(GraphUtils::AddEdge(control_graph->FindNode("data")->GetOutControlAnchor(),
                                control_graph->FindNode("first")->GetInControlAnchor()),
            GRAPH_SUCCESS);
  components.clear();
  HostCpuFusionPass control_pass(std::make_shared<StFakeCompiler>(), [](const std::string &) { return true; });
  EXPECT_EQ(control_pass.BuildFusionRegions(control_graph, components), NOT_CHANGED);
  EXPECT_TRUE(components.empty());

  const auto shape_graph = BuildCoverageChainGraph();
  ASSERT_NE(shape_graph, nullptr);
  shape_graph->FindNode("first")->GetOpDesc()->MutableOutputDesc(0)->SetShape(GeShape({-1}));
  components.clear();
  EXPECT_EQ(control_pass.BuildFusionRegions(shape_graph, components), NOT_CHANGED);
  EXPECT_TRUE(components.empty());

  const auto empty_graph = std::make_shared<ComputeGraph>("empty_host_cpu_coverage");
  components.clear();
  EXPECT_EQ(control_pass.BuildFusionRegions(nullptr, components), PARAM_INVALID);
  EXPECT_EQ(control_pass.BuildFusionRegions(empty_graph, components), PARAM_INVALID);

  const auto rollback_graph = BuildCoverageChainGraph();
  ASSERT_NE(rollback_graph, nullptr);
  HostCpuFusionPass region_pass(std::make_shared<StFakeCompiler>(), [](const std::string &) { return true; });
  ASSERT_EQ(region_pass.BuildFusionRegions(rollback_graph, components), SUCCESS);
  ASSERT_EQ(components.size(), 1U);
  ASSERT_EQ(components[0].size(), 1U);
  const std::string fused_name = std::string(kFusedHostCpuOpType) + "_" + components[0][0].chain_id;
  ASSERT_NE(rollback_graph->AddNode(std::make_shared<OpDesc>(fused_name, kFusedHostCpuOpType)), nullptr);
  NodeEngineMap atomic_map;
  NodeEngineMap composite_map;
  EXPECT_EQ(region_pass.Run(rollback_graph, atomic_map, composite_map), FAILED);
  EXPECT_NE(rollback_graph->FindNode("first"), nullptr);
  EXPECT_NE(rollback_graph->FindNode("second"), nullptr);
#endif
}
}  // namespace ge
