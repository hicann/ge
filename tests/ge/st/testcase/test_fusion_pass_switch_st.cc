/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include <atomic>
#include <fstream>
#include <map>
#include <string>
#include <unistd.h>

#include "ge_graph_dsl/graph_dsl.h"
#include "ge/ge_api.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/graph_utils_ex.h"
#include "graph/ge_local_context.h"
#include "ge_running_env/ge_running_env_faker.h"
#include "ge_running_env/fake_op.h"
#include "ge_running_env/fake_graph_optimizer.h"
#include "ge_running_env/fake_ops_kernel_builder.h"
#include "ge_running_env/op_reg.h"
#include "utils/mock_ops_kernel_builder.h"
#include "utils/taskdef_builder.h"
#include "register/register_custom_pass.h"
#include "register/pass_option_utils.h"
#include "register/optimization_option_registry.h"
#include "register/op_tiling_registry.h"
#include "ge/fusion/pass/pattern_fusion_pass.h"
#include "ge/fusion/pass/fusion_pass_reg.h"
#include "graph/fusion/fusion_utils.h"
#include "es_ge_test_ops.h"
#include "common/opskernel/ops_kernel_info_types.h"

#define private public
#include "compiler/graph/fusion/pass/pass_registry.h"
#include "compiler/graph/fusion/pass/fusion_pass_executor.h"
#undef private

namespace ge {
namespace fusion {
namespace {

std::string GetCodeDir() {
  char current_path[4096] = {'\0'};
  getcwd(current_path, sizeof(current_path));
  return current_path;
}

static std::atomic<int> g_pass_run_count{0};

graphStatus StubInferShape(Operator &op) {
  auto op_desc = OpDescUtils::GetOpDescFromOperator(op);
  if (op_desc->GetInputsSize() > 0U && op_desc->GetOutputsSize() > 0U) {
    *op_desc->MutableOutputDesc(0) = *op_desc->GetInputDescPtr(0);
  }
  return GRAPH_SUCCESS;
}

void MockGenerateTask() {
  auto aicore_func = [](const ge::Node &node, RunContext &context, std::vector<domi::TaskDef> &tasks) -> Status {
    auto op_desc = node.GetOpDesc();
    op_desc->SetOpKernelLibName("AiCoreLib");
    size_t arg_size = 100U;
    std::vector<uint8_t> args(arg_size, 0);
    domi::TaskDef task_def;
    task_def.set_type(static_cast<uint32_t>(ModelTaskType::MODEL_TASK_KERNEL));
    auto kernel_info = task_def.mutable_kernel();
    kernel_info->set_args(args.data(), args.size());
    kernel_info->set_args_size(arg_size);
    kernel_info->mutable_context()->set_kernel_type(static_cast<uint32_t>(ccKernelType::TE));
    kernel_info->set_kernel_name(node.GetName());
    kernel_info->set_block_dim(1);
    uint16_t args_offset[2] = {0};
    kernel_info->mutable_context()->set_args_offset(args_offset, 2 * sizeof(uint16_t));
    kernel_info->mutable_context()->set_op_index(node.GetOpDesc()->GetId());
    tasks.emplace_back(task_def);
    return SUCCESS;
  };
  MockForGenerateTask("AiCoreLib", aicore_func);
  MockForGenerateTask("AIcoreEngine", aicore_func);
}

// TransData → Data (删除 TransData，直接返回输入)
class TransDataRemovePass : public PatternFusionPass {
 protected:
  std::vector<PatternUniqPtr> Patterns() override {
    std::vector<PatternUniqPtr> patterns;
    auto pattern_graph = ge::es::EsGraphBuilder("pattern");
    auto esb_graph = pattern_graph.GetCGraphBuilder();
    auto data = EsCreateGraphInput(esb_graph, 0);
    auto transdata = EsTransData(data, "0", "29", 0, 0, 0);
    esb_graph->SetGraphOutput(transdata, 0);
    auto pattern = std::make_unique<Pattern>(std::move(*pattern_graph.BuildAndReset()));
    patterns.emplace_back(pattern.release());
    return patterns;
  }
  bool MeetRequirements(const std::unique_ptr<MatchResult> &match_result) override {
    return true;
  }
  GraphUniqPtr Replacement(const std::unique_ptr<MatchResult> &match_result) override {
    g_pass_run_count++;
    auto replace_graph = ge::es::EsGraphBuilder("replacement");
    auto esb_graph = replace_graph.GetCGraphBuilder();
    auto data = EsCreateGraphInput(esb_graph, 0);
    esb_graph->SetGraphOutput(data, 0);
    return replace_graph.BuildAndReset();
  }
};

void ResetPassRegistry() {
  PassRegistry::GetInstance().name_2_fusion_pass_regs_.clear();
  PassRegistry::GetInstance().descriptor_key_2_python_pass_descs_.clear();
  PassRegistry::GetInstance().pass_name_2_python_pass_create_contexts_.clear();
}

void ResetOoTable() {
  GetThreadLocalContext().GetOo().Initialize({}, OptionRegistry::GetInstance().GetRegisteredOptTable());
  GetThreadLocalContext().SetGlobalOption({});
  GetThreadLocalContext().SetSessionOption({});
  GetThreadLocalContext().SetGraphOption({});
}

bool HasNodeType(const ComputeGraphPtr &graph, const std::string &type) {
  for (const auto &node : graph->GetDirectNode()) {
    if (node->GetType() == type) {
      return true;
    }
  }
  return false;
}
}  // namespace

class FusionPassSwitchTest : public testing::Test {
 protected:
  void SetUp() override {
    ResetPassRegistry();
    g_pass_run_count = 0;
    MockGenerateTask();
  }
  void TearDown() override {
    ResetPassRegistry();
    ResetOoTable();
    OpsKernelBuilderRegistry::GetInstance().Unregister("AiCoreLib");
    OpsKernelBuilderRegistry::GetInstance().Unregister("AIcoreEngine");
    GeRunningEnvFaker env;
    env.InstallDefault();
  }
};

void InstallFusionEnv() {
  auto ge_env = GeRunningEnvFaker();
  ge_env.Reset()
      .Install(FakeEngine("DNN_VM_GE_LOCAL").KernelInfoStore("DNN_VM_GE_LOCAL_OP_STORE"))
      .Install(FakeEngine("AIcoreEngine").KernelInfoStore("AiCoreLib").GraphOptimizer("AIcoreEngine"))
      .Install(FakeOp(DATA).InfoStoreAndBuilder("DNN_VM_GE_LOCAL_OP_STORE").InferShape(StubInferShape))
      .Install(FakeOp(NETOUTPUT).InfoStoreAndBuilder("DNN_VM_GE_LOCAL_OP_STORE").InferShape(StubInferShape))
      .Install(FakeOp(TRANSDATA).InfoStoreAndBuilder("AiCoreLib").InferShape(StubInferShape));
  optiling::OpTilingFuncV2 tilingfun = [](const ge::Operator &op, const optiling::OpCompileInfoV2 &compile_info,
                                          optiling::OpRunInfoV2 &run_info) -> bool {
    run_info.SetWorkspaces({1024});
    return true;
  };
  optiling::OpTilingRegistryInterf_V2(TRANSDATA, tilingfun);
  REGISTER_OP_TILING_UNIQ_V2(TransData, tilingfun, 1);
}

Graph BuildSimpleTransDataGraph() {
  auto data_cfg = OP_CFG(DATA).TensorDesc(FORMAT_NCHW, DT_FLOAT, {1, 16, 16, 16}).Attr(ATTR_NAME_INDEX, 0);
  auto transdata_cfg = OP_CFG(TRANSDATA).TensorDesc(FORMAT_NCHW, DT_FLOAT, {1, 16, 16, 16});
  auto netoutput_cfg = OP_CFG(NETOUTPUT).TensorDesc(FORMAT_NCHW, DT_FLOAT, {1, 16, 16, 16});
  DEF_GRAPH(g1) {
    CHAIN(NODE("data_1", data_cfg)
              ->EDGE(0, 0)
              ->NODE("transdata_1", transdata_cfg)
              ->EDGE(0, 0)
              ->NODE(NODE_NAME_NET_OUTPUT, netoutput_cfg));
  };
  return ToGeGraph(g1);
}

/**
 * 用例描述：注册PassSwitch::kOff的融合Pass，不配置任何运行时开关时，Pass在编译流程中被跳过
 * 预置条件：
 *   1. Fake AIcoreEngine引擎及Data/TransData/NetOutput算子的InfoStore和Builder
 *   2. Mock GenerateTask桩住算子编译的Task生成
 *   3. 注册TransDataRemovePass，声明PassSwitch::kOff
 * 测试步骤：
 *   1. 构造Data→TransData→NetOutput图
 *   2. 创建Session，AddGraph，不配置任何开关
 *   3. 执行BuildGraph触发编译流程
 * 预期结果：
 *   1. BuildGraph返回SUCCESS
 *   2. g_pass_run_count == 0，证明PassSwitch::kOff默认关闭生效，Pass未执行Replacement
 */
TEST_F(FusionPassSwitchTest, DefaultOff_NoConfig_PassSkipped) {
  InstallFusionEnv();
  REG_FUSION_PASS(TransDataRemovePass).DefaultSwitch(PassSwitch::kOff).Stage(CustomPassStage::kBeforeInferShape);

  auto graph = BuildSimpleTransDataGraph();
  auto compute_graph = GraphUtilsEx::GetComputeGraph(graph);
  auto transdata_node = compute_graph->FindNode("transdata_1");
  ASSERT_NE(transdata_node, nullptr);
  transdata_node->GetOpDesc()->SetOpEngineName("AIcoreEngine");

  ResetOoTable();

  map<AscendString, AscendString> options;
  Session session(options);
  ASSERT_EQ(session.AddGraph(1, graph, options), SUCCESS);

  std::vector<InputTensorInfo> inputs;
  ASSERT_EQ(session.BuildGraph(1, inputs), SUCCESS);

  EXPECT_EQ(g_pass_run_count.load(), 0);
}

/**
 * 用例描述：注册PassSwitch::kOff的融合Pass，通过optimization_switch配置on时，Pass在编译流程中执行
 * 预置条件：
 *   1. Fake AIcoreEngine引擎及Data/TransData/NetOutput算子的InfoStore和Builder
 *   2. Mock GenerateTask桩住算子编译的Task生成
 *   3. 注册TransDataRemovePass，声明PassSwitch::kOff
 * 测试步骤：
 *   1. 构造Data→TransData→NetOutput图
 *   2. 创建Session时配置ge.optimizationSwitch=TransDataRemovePass:on
 *   3. 执行BuildGraph触发编译流程
 * 预期结果：
 *   1. BuildGraph返回SUCCESS
 *   2. g_pass_run_count > 0，证明第1层(graph option)覆盖了第4层(kOff)，Pass执行了Replacement
 */
TEST_F(FusionPassSwitchTest, DefaultOff_OptionOn_PassExecuted) {
  InstallFusionEnv();
  REG_FUSION_PASS(TransDataRemovePass).DefaultSwitch(PassSwitch::kOff).Stage(CustomPassStage::kBeforeInferShape);

  auto graph = BuildSimpleTransDataGraph();
  auto compute_graph = GraphUtilsEx::GetComputeGraph(graph);
  auto transdata_node = compute_graph->FindNode("transdata_1");
  ASSERT_NE(transdata_node, nullptr);
  transdata_node->GetOpDesc()->SetOpEngineName("AIcoreEngine");

  map<AscendString, AscendString> options;
  options[AscendString("ge.optimizationSwitch")] = AscendString("TransDataRemovePass:on");
  Session session(options);
  ASSERT_EQ(session.AddGraph(1, graph, options), SUCCESS);

  std::vector<InputTensorInfo> inputs;
  ASSERT_EQ(session.BuildGraph(1, inputs), SUCCESS);

  EXPECT_GT(g_pass_run_count.load(), 0);
}

/**
 * 用例描述：不调用DefaultSwitch（兼容写法）注册融合Pass，无开关配置时Pass在编译流程中执行
 * 预置条件：
 *   1. Fake AIcoreEngine引擎及Data/TransData/NetOutput算子的InfoStore和Builder
 *   2. Mock GenerateTask桩住算子编译的Task生成
 *   3. 注册TransDataRemovePass，不传PassSwitch参数（默认kOn）
 * 测试步骤：
 *   1. 构造Data→TransData→NetOutput图
 *   2. 创建Session，AddGraph，不配置任何开关
 *   3. 执行BuildGraph触发编译流程
 * 预期结果：
 *   1. BuildGraph返回SUCCESS
 *   2. g_pass_run_count > 0，证明不调用DefaultSwitch向后兼容，默认kOn与原return true行为一致
 */
TEST_F(FusionPassSwitchTest, NoSwitch_DefaultOn_PassExecuted) {
  InstallFusionEnv();
  REG_FUSION_PASS(TransDataRemovePass).Stage(CustomPassStage::kBeforeInferShape);

  auto graph = BuildSimpleTransDataGraph();
  auto compute_graph = GraphUtilsEx::GetComputeGraph(graph);
  auto transdata_node = compute_graph->FindNode("transdata_1");
  ASSERT_NE(transdata_node, nullptr);
  transdata_node->GetOpDesc()->SetOpEngineName("AIcoreEngine");

  map<AscendString, AscendString> options;
  Session session(options);
  ASSERT_EQ(session.AddGraph(1, graph, options), SUCCESS);

  std::vector<InputTensorInfo> inputs;
  ASSERT_EQ(session.BuildGraph(1, inputs), SUCCESS);

  EXPECT_GT(g_pass_run_count.load(), 0);
}

/**
 * 用例描述：注册PassSwitch::kOff的融合Pass，通过fusion_switch_file JSON精确匹配on时，Pass在编译流程中执行
 * 预置条件：
 *   1. Fake AIcoreEngine引擎及Data/TransData/NetOutput算子的InfoStore和Builder
 *   2. Mock GenerateTask桩住算子编译的Task生成
 *   3. 注册TransDataRemovePass，声明PassSwitch::kOff
 *   4. 编写fusion_switch.cfg，配置GraphFusion.TransDataRemovePass=on
 * 测试步骤：
 *   1. 构造Data→TransData→NetOutput图
 *   2. 创建Session时配置ge.fusionSwitchFile指向JSON配置文件
 *   3. 执行BuildGraph触发编译流程
 * 预期结果：
 *   1. BuildGraph返回SUCCESS
 *   2. g_pass_run_count > 0，证明第2层(JSON精确匹配)覆盖了第4层(kOff)，Pass执行了Replacement
 */
TEST_F(FusionPassSwitchTest, DefaultOff_JsonExactOn_PassExecuted) {
  InstallFusionEnv();
  REG_FUSION_PASS(TransDataRemovePass).DefaultSwitch(PassSwitch::kOff).Stage(CustomPassStage::kBeforeInferShape);

  std::string fusion_config_json =
      "{\n"
      "    \"Switch\":{\n"
      "        \"GraphFusion\":{\n"
      "          \"TransDataRemovePass\" : \"on\"\n"
      "        },\n"
      "        \"UBFusion\":{\n"
      "        }\n"
      "    }}";
  std::ofstream json_file("./fusion_switch_config.json");
  json_file << fusion_config_json << std::endl;
  std::string config_file_path = GetCodeDir() + "/fusion_switch_config.json";

  auto graph = BuildSimpleTransDataGraph();
  auto compute_graph = GraphUtilsEx::GetComputeGraph(graph);
  auto transdata_node = compute_graph->FindNode("transdata_1");
  ASSERT_NE(transdata_node, nullptr);
  transdata_node->GetOpDesc()->SetOpEngineName("AIcoreEngine");

  map<AscendString, AscendString> options;
  options[AscendString("ge.fusionSwitchFile")] = AscendString(config_file_path.c_str());
  Session session(options);
  ASSERT_EQ(session.AddGraph(1, graph, options), SUCCESS);

  std::vector<InputTensorInfo> inputs;
  ASSERT_EQ(session.BuildGraph(1, inputs), SUCCESS);

  EXPECT_GT(g_pass_run_count.load(), 0);
  remove("./fusion_switch_config.json");
}

}  // namespace fusion
}  // namespace ge
