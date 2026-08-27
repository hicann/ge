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
#include <memory>
#include <string>
#include "register/core_num_utils.h"
#include "graph/compute_graph.h"
#include "graph/op_desc.h"
#include "graph/utils/attr_utils.h"
#include "graph/ge_local_context.h"
#include "ge_common/ge_common_api_types.h"

namespace ge {
namespace {

class CoreNumValidateUT : public testing::Test {
 protected:
  void SetUp() {
    platform_info_.soc_info.ai_core_cnt = 32;
    platform_info_.soc_info.vector_core_cnt = 16;
  }

  void TearDown() {
    // 统一清理 thread-local 选项，消除跨用例 SOC_VERSION 污染隐患（GetOption 查 graph/session/global）。
    GetThreadLocalContext().SetGlobalOption({});
    GetThreadLocalContext().SetSessionOption({});
    GetThreadLocalContext().SetGraphOption({});
  }

  fe::PlatformInfo platform_info_;
};

// --- ValidateCoreNumWithOpDesc ---

TEST_F(CoreNumValidateUT, ValidateCoreNumWithOpDesc_NullOpDesc_ReturnsError) {
  OpDescPtr null_op;
  EXPECT_NE(CoreNumUtils::ValidateCoreNumWithOpDesc(platform_info_, null_op), GRAPH_SUCCESS);
}

TEST_F(CoreNumValidateUT, ValidateCoreNumWithOpDesc_NoAttrs_ReturnsSuccess) {
  auto op_desc = std::make_shared<OpDesc>("test_op", "Relu");
  EXPECT_EQ(CoreNumUtils::ValidateCoreNumWithOpDesc(platform_info_, op_desc), GRAPH_SUCCESS);
}

// 属性存在但类型非 string（如 int），GetStr 失败 -> "It is not string." 分支。
TEST_F(CoreNumValidateUT, ValidateCoreNumWithOpDesc_AiCoreNumNotString_ReturnsError) {
  auto op_desc = std::make_shared<OpDesc>("test_op", "Relu");
  (void)ge::AttrUtils::SetInt(op_desc, kAiCoreNumOp, 16);
  EXPECT_NE(CoreNumUtils::ValidateCoreNumWithOpDesc(platform_info_, op_desc), GRAPH_SUCCESS);
}

TEST_F(CoreNumValidateUT, ValidateCoreNumWithOpDesc_VectorCoreNumNotString_ReturnsError) {
  auto op_desc = std::make_shared<OpDesc>("test_op", "Relu");
  (void)ge::AttrUtils::SetInt(op_desc, kVectorCoreNumOp, 8);
  EXPECT_NE(CoreNumUtils::ValidateCoreNumWithOpDesc(platform_info_, op_desc), GRAPH_SUCCESS);
}

TEST_F(CoreNumValidateUT, ValidateCoreNumWithOpDesc_ValidAiCoreNum_ReturnsSuccess) {
  auto op_desc = std::make_shared<OpDesc>("test_op", "Relu");
  (void)ge::AttrUtils::SetStr(op_desc, kAiCoreNumOp, "16");
  EXPECT_EQ(CoreNumUtils::ValidateCoreNumWithOpDesc(platform_info_, op_desc), GRAPH_SUCCESS);
}

TEST_F(CoreNumValidateUT, ValidateCoreNumWithOpDesc_ZeroAiCoreNum_ReturnsSuccess) {
  auto op_desc = std::make_shared<OpDesc>("test_op", "Relu");
  (void)ge::AttrUtils::SetStr(op_desc, kAiCoreNumOp, "0");
  EXPECT_EQ(CoreNumUtils::ValidateCoreNumWithOpDesc(platform_info_, op_desc), GRAPH_SUCCESS);
}

TEST_F(CoreNumValidateUT, ValidateCoreNumWithOpDesc_MaxAiCoreNum_ReturnsSuccess) {
  auto op_desc = std::make_shared<OpDesc>("test_op", "Relu");
  (void)ge::AttrUtils::SetStr(op_desc, kAiCoreNumOp, "32");
  EXPECT_EQ(CoreNumUtils::ValidateCoreNumWithOpDesc(platform_info_, op_desc), GRAPH_SUCCESS);
}

TEST_F(CoreNumValidateUT, ValidateCoreNumWithOpDesc_AiCoreNumOutOfRange_ReturnsError) {
  auto op_desc = std::make_shared<OpDesc>("test_op", "Relu");
  (void)ge::AttrUtils::SetStr(op_desc, kAiCoreNumOp, "33");
  EXPECT_NE(CoreNumUtils::ValidateCoreNumWithOpDesc(platform_info_, op_desc), GRAPH_SUCCESS);
}

TEST_F(CoreNumValidateUT, ValidateCoreNumWithOpDesc_AiCoreNumNotInteger_ReturnsError) {
  auto op_desc = std::make_shared<OpDesc>("test_op", "Relu");
  (void)ge::AttrUtils::SetStr(op_desc, kAiCoreNumOp, "abc");
  EXPECT_NE(CoreNumUtils::ValidateCoreNumWithOpDesc(platform_info_, op_desc), GRAPH_SUCCESS);
}

TEST_F(CoreNumValidateUT, ValidateCoreNumWithOpDesc_AiCoreNumNegative_ReturnsError) {
  auto op_desc = std::make_shared<OpDesc>("test_op", "Relu");
  (void)ge::AttrUtils::SetStr(op_desc, kAiCoreNumOp, "-1");
  EXPECT_NE(CoreNumUtils::ValidateCoreNumWithOpDesc(platform_info_, op_desc), GRAPH_SUCCESS);
}

TEST_F(CoreNumValidateUT, ValidateCoreNumWithOpDesc_ValidVectorCoreNum_ReturnsSuccess) {
  auto op_desc = std::make_shared<OpDesc>("test_op", "Relu");
  (void)ge::AttrUtils::SetStr(op_desc, kVectorCoreNumOp, "8");
  EXPECT_EQ(CoreNumUtils::ValidateCoreNumWithOpDesc(platform_info_, op_desc), GRAPH_SUCCESS);
}

TEST_F(CoreNumValidateUT, ValidateCoreNumWithOpDesc_VectorCoreNumOutOfRange_ReturnsError) {
  auto op_desc = std::make_shared<OpDesc>("test_op", "Relu");
  (void)ge::AttrUtils::SetStr(op_desc, kVectorCoreNumOp, "17");
  EXPECT_NE(CoreNumUtils::ValidateCoreNumWithOpDesc(platform_info_, op_desc), GRAPH_SUCCESS);
}

TEST_F(CoreNumValidateUT, ValidateCoreNumWithOpDesc_VectorCoreNumNotInteger_ReturnsError) {
  auto op_desc = std::make_shared<OpDesc>("test_op", "Relu");
  (void)ge::AttrUtils::SetStr(op_desc, kVectorCoreNumOp, "xyz");
  EXPECT_NE(CoreNumUtils::ValidateCoreNumWithOpDesc(platform_info_, op_desc), GRAPH_SUCCESS);
}

TEST_F(CoreNumValidateUT, ValidateCoreNumWithOpDesc_BothAttrsValid_ReturnsSuccess) {
  auto op_desc = std::make_shared<OpDesc>("test_op", "Relu");
  (void)ge::AttrUtils::SetStr(op_desc, kAiCoreNumOp, "16");
  (void)ge::AttrUtils::SetStr(op_desc, kVectorCoreNumOp, "8");
  EXPECT_EQ(CoreNumUtils::ValidateCoreNumWithOpDesc(platform_info_, op_desc), GRAPH_SUCCESS);
}

TEST_F(CoreNumValidateUT, ValidateCoreNumWithOpDesc_BothAttrsOneInvalid_ReturnsError) {
  auto op_desc = std::make_shared<OpDesc>("test_op", "Relu");
  (void)ge::AttrUtils::SetStr(op_desc, kAiCoreNumOp, "16");
  (void)ge::AttrUtils::SetStr(op_desc, kVectorCoreNumOp, "99");
  EXPECT_NE(CoreNumUtils::ValidateCoreNumWithOpDesc(platform_info_, op_desc), GRAPH_SUCCESS);
}

// --- ValidateCoreNumWithOpDesc edge cases ---

TEST_F(CoreNumValidateUT, ValidateCoreNumWithOpDesc_AiCoreNumEmptyString_ReturnsError) {
  auto op_desc = std::make_shared<OpDesc>("test_op", "Relu");
  (void)ge::AttrUtils::SetStr(op_desc, kAiCoreNumOp, "");
  EXPECT_NE(CoreNumUtils::ValidateCoreNumWithOpDesc(platform_info_, op_desc), GRAPH_SUCCESS);
}

TEST_F(CoreNumValidateUT, ValidateCoreNumWithOpDesc_AiCoreNumOverflow_ReturnsError) {
  auto op_desc = std::make_shared<OpDesc>("test_op", "Relu");
  (void)ge::AttrUtils::SetStr(op_desc, kAiCoreNumOp, "99999999999999999999");
  EXPECT_NE(CoreNumUtils::ValidateCoreNumWithOpDesc(platform_info_, op_desc), GRAPH_SUCCESS);
}

TEST_F(CoreNumValidateUT, ValidateCoreNumWithOpDesc_AiCoreNumLeadingZero_ReturnsError) {
  auto op_desc = std::make_shared<OpDesc>("test_op", "Relu");
  (void)ge::AttrUtils::SetStr(op_desc, kAiCoreNumOp, "01");
  EXPECT_NE(CoreNumUtils::ValidateCoreNumWithOpDesc(platform_info_, op_desc), GRAPH_SUCCESS);
}

TEST_F(CoreNumValidateUT, ValidateCoreNumWithOpDesc_VectorCoreNumLeadingZero_ReturnsError) {
  auto op_desc = std::make_shared<OpDesc>("test_op", "Relu");
  (void)ge::AttrUtils::SetStr(op_desc, kVectorCoreNumOp, "01");
  EXPECT_NE(CoreNumUtils::ValidateCoreNumWithOpDesc(platform_info_, op_desc), GRAPH_SUCCESS);
}

// --- ValidateCoreNumWithGraph(compute_graph) single-param overload ---

TEST_F(CoreNumValidateUT, ValidateCoreNumWithGraphSingleParam_NullGraph_ReturnsError) {
  ComputeGraphPtr null_graph;
  EXPECT_NE(CoreNumUtils::ValidateCoreNumWithGraph(null_graph), GRAPH_SUCCESS);
}

TEST_F(CoreNumValidateUT, ValidateCoreNumWithGraphSingleParam_NoCoreNumAttrs_ReturnsSuccess) {
  auto graph = std::make_shared<ComputeGraph>("test_graph");
  auto op1 = std::make_shared<OpDesc>("op1", "Relu");
  graph->AddNode(op1);

  auto op2 = std::make_shared<OpDesc>("op2", "Add");
  graph->AddNode(op2);

  EXPECT_EQ(CoreNumUtils::ValidateCoreNumWithGraph(graph), GRAPH_SUCCESS);
}

TEST_F(CoreNumValidateUT, ValidateCoreNumWithGraphSingleParam_EmptyGraph_ReturnsSuccess) {
  auto graph = std::make_shared<ComputeGraph>("test_graph");
  EXPECT_EQ(CoreNumUtils::ValidateCoreNumWithGraph(graph), GRAPH_SUCCESS);
}

TEST_F(CoreNumValidateUT, ValidateCoreNumWithGraphSingleParam_AiCoreNumValid_ReturnsSuccess) {
  std::map<std::string, std::string> global_opts = {{"ge.socVersion", "Ascend910B2"}};
  GetThreadLocalContext().SetGlobalOption(global_opts);

  auto graph = std::make_shared<ComputeGraph>("test_graph");
  auto op1 = std::make_shared<OpDesc>("op1", "Relu");
  (void)ge::AttrUtils::SetStr(op1, kAiCoreNumOp, "16");
  graph->AddNode(op1);

  EXPECT_EQ(CoreNumUtils::ValidateCoreNumWithGraph(graph), GRAPH_SUCCESS);
}

TEST_F(CoreNumValidateUT, ValidateCoreNumWithGraphSingleParam_AiCoreNumInvalid_ReturnsError) {
  std::map<std::string, std::string> global_opts = {{"ge.socVersion", "Ascend910B2"}};
  GetThreadLocalContext().SetGlobalOption(global_opts);

  auto graph = std::make_shared<ComputeGraph>("test_graph");
  auto op1 = std::make_shared<OpDesc>("op1", "Relu");
  (void)ge::AttrUtils::SetStr(op1, kAiCoreNumOp, "99999");
  graph->AddNode(op1);

  EXPECT_NE(CoreNumUtils::ValidateCoreNumWithGraph(graph), GRAPH_SUCCESS);
}

// fail-fast：图中存在核数属性但 SOC_VERSION 未设置时，应报错而非静默跳过（外部错误码 E10001）。
TEST_F(CoreNumValidateUT, ValidateCoreNumWithGraphSingleParam_CoreNumWithoutSocVersion_ReturnsError) {
  // 清理可能由其他用例残留的 SOC_VERSION，确保本用例处于"未设置"状态。
  GetThreadLocalContext().SetGlobalOption({});
  GetThreadLocalContext().SetSessionOption({});

  auto graph = std::make_shared<ComputeGraph>("test_graph");
  auto op1 = std::make_shared<OpDesc>("op1", "Relu");
  (void)ge::AttrUtils::SetStr(op1, kAiCoreNumOp, "16");
  graph->AddNode(op1);

  EXPECT_NE(CoreNumUtils::ValidateCoreNumWithGraph(graph), GRAPH_SUCCESS);
}

// fail-fast：vectorcore 属性同样在 SOC_VERSION 缺失时报错。
TEST_F(CoreNumValidateUT, ValidateCoreNumWithGraphSingleParam_VectorCoreNumWithoutSocVersion_ReturnsError) {
  GetThreadLocalContext().SetGlobalOption({});
  GetThreadLocalContext().SetSessionOption({});

  auto graph = std::make_shared<ComputeGraph>("test_graph");
  auto op1 = std::make_shared<OpDesc>("op1", "Relu");
  (void)ge::AttrUtils::SetStr(op1, kVectorCoreNumOp, "8");
  graph->AddNode(op1);

  EXPECT_NE(CoreNumUtils::ValidateCoreNumWithGraph(graph), GRAPH_SUCCESS);
}

// --- GetCoreNumFromGraph ---

TEST_F(CoreNumValidateUT, GetCoreNumFromGraph_NullGraph_ReturnsError) {
  ComputeGraphPtr null_graph;
  int32_t aicore_num = -1;
  int32_t vectorcore_num = -1;
  EXPECT_NE(CoreNumUtils::GetCoreNumFromGraph(null_graph, aicore_num, vectorcore_num), GRAPH_SUCCESS);
}

TEST_F(CoreNumValidateUT, GetCoreNumFromGraph_NoAttrs_KeepsDefaultValue) {
  auto graph = std::make_shared<ComputeGraph>("test_graph");
  int32_t aicore_num = -1;
  int32_t vectorcore_num = -1;
  EXPECT_EQ(CoreNumUtils::GetCoreNumFromGraph(graph, aicore_num, vectorcore_num), GRAPH_SUCCESS);
  EXPECT_EQ(aicore_num, -1);
  EXPECT_EQ(vectorcore_num, -1);
}

TEST_F(CoreNumValidateUT, GetCoreNumFromGraph_EmptyAttrValue_KeepsDefaultValue) {
  auto graph = std::make_shared<ComputeGraph>("test_graph");
  (void)ge::AttrUtils::SetStr(graph, AICORE_NUM, "");
  (void)ge::AttrUtils::SetStr(graph, kVectorCoreNum, "");
  int32_t aicore_num = -1;
  int32_t vectorcore_num = -1;
  EXPECT_EQ(CoreNumUtils::GetCoreNumFromGraph(graph, aicore_num, vectorcore_num), GRAPH_SUCCESS);
  EXPECT_EQ(aicore_num, -1);
  EXPECT_EQ(vectorcore_num, -1);
}

TEST_F(CoreNumValidateUT, GetCoreNumFromGraph_BothAttrsValid_ReturnsParsedValue) {
  auto graph = std::make_shared<ComputeGraph>("test_graph");
  (void)ge::AttrUtils::SetStr(graph, AICORE_NUM, "8");
  (void)ge::AttrUtils::SetStr(graph, kVectorCoreNum, "16");
  int32_t aicore_num = -1;
  int32_t vectorcore_num = -1;
  EXPECT_EQ(CoreNumUtils::GetCoreNumFromGraph(graph, aicore_num, vectorcore_num), GRAPH_SUCCESS);
  EXPECT_EQ(aicore_num, 8);
  EXPECT_EQ(vectorcore_num, 16);
}

TEST_F(CoreNumValidateUT, GetCoreNumFromGraph_OnlyAiCoreNum_VectorCoreNumKeepsDefaultValue) {
  auto graph = std::make_shared<ComputeGraph>("test_graph");
  (void)ge::AttrUtils::SetStr(graph, AICORE_NUM, "8");
  int32_t aicore_num = -1;
  int32_t vectorcore_num = -1;
  EXPECT_EQ(CoreNumUtils::GetCoreNumFromGraph(graph, aicore_num, vectorcore_num), GRAPH_SUCCESS);
  EXPECT_EQ(aicore_num, 8);
  EXPECT_EQ(vectorcore_num, -1);
}

TEST_F(CoreNumValidateUT, GetCoreNumFromGraph_ZeroAiCoreNum_ReturnsZero) {
  auto graph = std::make_shared<ComputeGraph>("test_graph");
  (void)ge::AttrUtils::SetStr(graph, AICORE_NUM, "0");
  int32_t aicore_num = -1;
  int32_t vectorcore_num = -1;
  EXPECT_EQ(CoreNumUtils::GetCoreNumFromGraph(graph, aicore_num, vectorcore_num), GRAPH_SUCCESS);
  EXPECT_EQ(aicore_num, 0);
}

TEST_F(CoreNumValidateUT, GetCoreNumFromGraph_AiCoreNumNotInteger_ReturnsError) {
  auto graph = std::make_shared<ComputeGraph>("test_graph");
  (void)ge::AttrUtils::SetStr(graph, AICORE_NUM, "abc");
  int32_t aicore_num = -1;
  int32_t vectorcore_num = -1;
  EXPECT_NE(CoreNumUtils::GetCoreNumFromGraph(graph, aicore_num, vectorcore_num), GRAPH_SUCCESS);
}

TEST_F(CoreNumValidateUT, GetCoreNumFromGraph_AiCoreNumNegative_ReturnsError) {
  auto graph = std::make_shared<ComputeGraph>("test_graph");
  (void)ge::AttrUtils::SetStr(graph, AICORE_NUM, "-1");
  int32_t aicore_num = -1;
  int32_t vectorcore_num = -1;
  EXPECT_NE(CoreNumUtils::GetCoreNumFromGraph(graph, aicore_num, vectorcore_num), GRAPH_SUCCESS);
}

TEST_F(CoreNumValidateUT, GetCoreNumFromGraph_VectorCoreNumLeadingZero_ReturnsError) {
  auto graph = std::make_shared<ComputeGraph>("test_graph");
  (void)ge::AttrUtils::SetStr(graph, kVectorCoreNum, "08");
  int32_t aicore_num = -1;
  int32_t vectorcore_num = -1;
  EXPECT_NE(CoreNumUtils::GetCoreNumFromGraph(graph, aicore_num, vectorcore_num), GRAPH_SUCCESS);
}

// --- FillCoreNumOptions ---

TEST_F(CoreNumValidateUT, FillCoreNumOptions_NegativeMeansUnset_WritesNothing) {
  std::map<std::string, std::string> options;
  EXPECT_EQ(CoreNumUtils::FillCoreNumOptions(-1, -1, options), GRAPH_SUCCESS);
  EXPECT_TRUE(options.empty());
}

TEST_F(CoreNumValidateUT, FillCoreNumOptions_BothSet_WritesBoth) {
  std::map<std::string, std::string> options;
  EXPECT_EQ(CoreNumUtils::FillCoreNumOptions(8, 16, options), GRAPH_SUCCESS);
  EXPECT_EQ(options[AICORE_NUM], "8");
  EXPECT_EQ(options[kVectorCoreNum], "16");
}

TEST_F(CoreNumValidateUT, FillCoreNumOptions_OnlyAiCoreSet_LeavesVectorCoreUnset) {
  std::map<std::string, std::string> options;
  EXPECT_EQ(CoreNumUtils::FillCoreNumOptions(8, -1, options), GRAPH_SUCCESS);
  EXPECT_EQ(options[AICORE_NUM], "8");
  EXPECT_EQ(options.count(kVectorCoreNum), 0U);
}

// 0 是合法配置值(表示不限制)，必须写进 options，不能被当成"未配置"。
TEST_F(CoreNumValidateUT, FillCoreNumOptions_ZeroIsConfigured_WritesZero) {
  std::map<std::string, std::string> options;
  EXPECT_EQ(CoreNumUtils::FillCoreNumOptions(0, 0, options), GRAPH_SUCCESS);
  EXPECT_EQ(options[AICORE_NUM], "0");
  EXPECT_EQ(options[kVectorCoreNum], "0");
}

TEST_F(CoreNumValidateUT, FillCoreNumOptions_KeepsUnrelatedEntries) {
  std::map<std::string, std::string> options{{"other.option", "keep"}};
  EXPECT_EQ(CoreNumUtils::FillCoreNumOptions(8, -1, options), GRAPH_SUCCESS);
  EXPECT_EQ(options["other.option"], "keep");
  EXPECT_EQ(options[AICORE_NUM], "8");
}

// --- GetCoreNumOptionsFromGraph ---

TEST_F(CoreNumValidateUT, GetCoreNumOptionsFromGraph_NullGraph_ReturnsError) {
  ComputeGraphPtr null_graph;
  std::map<std::string, std::string> options;
  EXPECT_NE(CoreNumUtils::GetCoreNumOptionsFromGraph(null_graph, options), GRAPH_SUCCESS);
}

TEST_F(CoreNumValidateUT, GetCoreNumOptionsFromGraph_NoAttrs_WritesNothing) {
  auto graph = std::make_shared<ComputeGraph>("test_graph");
  std::map<std::string, std::string> options;
  EXPECT_EQ(CoreNumUtils::GetCoreNumOptionsFromGraph(graph, options), GRAPH_SUCCESS);
  EXPECT_TRUE(options.empty());
}

TEST_F(CoreNumValidateUT, GetCoreNumOptionsFromGraph_RootGraphAttrs_WritesOptions) {
  auto graph = std::make_shared<ComputeGraph>("test_graph");
  (void)ge::AttrUtils::SetStr(graph, AICORE_NUM, "8");
  (void)ge::AttrUtils::SetStr(graph, kVectorCoreNum, "16");
  std::map<std::string, std::string> options;
  EXPECT_EQ(CoreNumUtils::GetCoreNumOptionsFromGraph(graph, options), GRAPH_SUCCESS);
  EXPECT_EQ(options[AICORE_NUM], "8");
  EXPECT_EQ(options[kVectorCoreNum], "16");
}

// 静态编译子图的 GeModel 持有的是根图的子图，模型级核数只写在根图上，必须能上溯拿到。
TEST_F(CoreNumValidateUT, GetCoreNumOptionsFromGraph_SubGraph_ReadsFromRootGraph) {
  auto root_graph = std::make_shared<ComputeGraph>("root_graph");
  (void)ge::AttrUtils::SetStr(root_graph, AICORE_NUM, "8");
  (void)ge::AttrUtils::SetStr(root_graph, kVectorCoreNum, "16");
  auto sub_graph = std::make_shared<ComputeGraph>("root_graph_sub_1_know");
  sub_graph->SetParentGraph(root_graph);

  std::map<std::string, std::string> options;
  EXPECT_EQ(CoreNumUtils::GetCoreNumOptionsFromGraph(sub_graph, options), GRAPH_SUCCESS);
  EXPECT_EQ(options[AICORE_NUM], "8");
  EXPECT_EQ(options[kVectorCoreNum], "16");
}

// 子图自身带同名属性不生效，避免子图上的残留值盖掉模型级配置。
TEST_F(CoreNumValidateUT, GetCoreNumOptionsFromGraph_SubGraphAttrIgnored_RootGraphWins) {
  auto root_graph = std::make_shared<ComputeGraph>("root_graph");
  (void)ge::AttrUtils::SetStr(root_graph, AICORE_NUM, "8");
  auto sub_graph = std::make_shared<ComputeGraph>("root_graph_sub_1_know");
  (void)ge::AttrUtils::SetStr(sub_graph, AICORE_NUM, "20");
  sub_graph->SetParentGraph(root_graph);

  std::map<std::string, std::string> options;
  EXPECT_EQ(CoreNumUtils::GetCoreNumOptionsFromGraph(sub_graph, options), GRAPH_SUCCESS);
  EXPECT_EQ(options[AICORE_NUM], "8");
}

// OM 被篡改成非法核数时必须失败，不能静默按未配置处理。
TEST_F(CoreNumValidateUT, GetCoreNumOptionsFromGraph_InvalidAttr_ReturnsError) {
  auto graph = std::make_shared<ComputeGraph>("test_graph");
  (void)ge::AttrUtils::SetStr(graph, AICORE_NUM, "abc");
  std::map<std::string, std::string> options;
  EXPECT_NE(CoreNumUtils::GetCoreNumOptionsFromGraph(graph, options), GRAPH_SUCCESS);
}

}  // namespace
}  // namespace ge
