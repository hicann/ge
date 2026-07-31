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
#include <vector>
#include <string>
#include "register/op_def_registry.h"

namespace ge {
static ge::graphStatus InferShape4Cov(gert::InferShapeContext *context) {
  return GRAPH_SUCCESS;
}
static ge::graphStatus InferShapeRange4Cov(gert::InferShapeRangeContext *context) {
  return GRAPH_SUCCESS;
}
static ge::graphStatus InferDataType4Cov(gert::InferDataTypeContext *context) {
  return GRAPH_SUCCESS;
}
}  // namespace ge

namespace ops {

class OpDefCovUT : public testing::Test {
 protected:
  void SetUp() {}
  void TearDown() {}
};

// ---- op_def.cc coverage ----

TEST_F(OpDefCovUT, OpDef_AssignmentOperator) {
  OpDef opDef1("TestAssign");
  opDef1.Input("x").DataType({ge::DT_FLOAT16});
  opDef1.Output("y").DataType({ge::DT_FLOAT16});
  OpDef opDef2("TestAssign2");
  opDef2 = opDef1;
  EXPECT_EQ(opDef2.GetOpType(), ge::AscendString("TestAssign"));
}

TEST_F(OpDefCovUT, OpDef_FindAttr) {
  OpDef opDef("TestFindAttr");
  opDef.Attr("attr1").AttrType(REQUIRED).String();
  OpAttrDef *attr = nullptr;
  auto status = opDef.FindAttr("attr1", &attr);
  EXPECT_EQ(status, ItemFindStatus::ITEM_FIND);
  status = opDef.FindAttr("nonexistent", &attr);
  EXPECT_EQ(status, ItemFindStatus::ITEM_NOEXIST);
}

TEST_F(OpDefCovUT, OpDef_AddAttr) {
  OpDef opDef1("TestAddAttr1");
  opDef1.Attr("attr1").AttrType(REQUIRED).String();
  OpDef opDef2("TestAddAttr2");
  auto &attr = opDef1.Attr("attr1");
  opDef2.AddAttr(attr);
  EXPECT_EQ(opDef2.GetAttrs().size(), 1U);
}

TEST_F(OpDefCovUT, OpDef_GetInputsAndGetOutputs) {
  OpDef opDef("TestGetIO");
  opDef.Input("x").DataType({ge::DT_FLOAT16});
  opDef.Output("y").DataType({ge::DT_FLOAT16});
  auto &inputs = opDef.GetInputs();
  auto &outputs = opDef.GetOutputs();
  EXPECT_EQ(inputs.size(), 1U);
  EXPECT_EQ(outputs.size(), 1U);
}

TEST_F(OpDefCovUT, OpDef_AICPU) {
  OpDef opDef("TestAICPU");
  opDef.AICPU().ExtendCfgInfo("key1", "value1");
  EXPECT_EQ(opDef.AICPU().GetCfgKeys().size(), 9U);
}

TEST_F(OpDefCovUT, OpDef_HostCPU) {
  OpDef opDef("TestHostCPU");
  opDef.HostCPU().ExtendCfgInfo("key1", "value1");
  EXPECT_EQ(opDef.HostCPU().GetCfgKeys().size(), 9U);
}

TEST_F(OpDefCovUT, OpDef_FollowImpl) {
  OpDef opDef("TestFollowImpl");
  opDef.Input("x").DataType({ge::DT_FLOAT16});
  opDef.Output("y").Follow("x").DataType({ge::DT_FLOAT16});
  opDef.FollowImpl();
  auto followMap = opDef.GetFollowMap();
  EXPECT_FALSE(followMap.empty());
}

TEST_F(OpDefCovUT, OpDef_GetFollowShapeMapAndTypeMap) {
  OpDef opDef("TestFollowMaps");
  opDef.Input("x").DataType({ge::DT_FLOAT16});
  opDef.Output("y").Follow("x").DataType({ge::DT_FLOAT16});
  opDef.FollowImpl();
  auto shapeMap = opDef.GetFollowShapeMap();
  auto typeMap = opDef.GetFollowTypeMap();
  EXPECT_FALSE(shapeMap.empty());
  EXPECT_FALSE(typeMap.empty());
}

TEST_F(OpDefCovUT, OpDef_GetParamDef) {
  OpDef opDef("TestGetParamDef");
  opDef.Input("x").DataType({ge::DT_FLOAT16});
  opDef.Output("y").DataType({ge::DT_FLOAT16});
  auto paramDef = opDef.GetParamDef(ge::AscendString("x"), OpDef::PortStat::IN);
  EXPECT_EQ(paramDef.GetParamName(), ge::AscendString("x"));
  auto paramDefOut = opDef.GetParamDef(ge::AscendString("y"), OpDef::PortStat::OUT);
  EXPECT_EQ(paramDefOut.GetParamName(), ge::AscendString("y"));
}

TEST_F(OpDefCovUT, OpDef_GetInferShapeRangeAndDataType) {
  OpDef opDef("TestGetInferFuncs");
  opDef.SetInferShape(ge::InferShape4Cov);
  opDef.SetInferShapeRange(ge::InferShapeRange4Cov);
  opDef.SetInferDataType(ge::InferDataType4Cov);
  EXPECT_NE(opDef.GetInferShape(), nullptr);
  EXPECT_NE(opDef.GetInferShapeRange(), nullptr);
  EXPECT_NE(opDef.GetInferDataType(), nullptr);
}

TEST_F(OpDefCovUT, OpDef_GetAttrs) {
  OpDef opDef("TestGetAttrs");
  opDef.Attr("attr1").AttrType(REQUIRED).String();
  opDef.Attr("attr2").AttrType(OPTIONAL).Int();
  auto &attrs = opDef.GetAttrs();
  EXPECT_EQ(attrs.size(), 2U);
}

TEST_F(OpDefCovUT, OpDef_GetOrCreateAttr) {
  OpDef opDef("TestGetOrCreate");
  opDef.GetOrCreateAttr("attr1").AttrType(REQUIRED).String();
  opDef.GetOrCreateAttr("attr1").AttrType(OPTIONAL).Int();
  EXPECT_EQ(opDef.GetAttrs().size(), 1U);
}

TEST_F(OpDefCovUT, OpDef_GetMergeInputsOutputs) {
  OpDef opDef("TestMergeIO");
  opDef.Input("x").DataType({ge::DT_FLOAT16});
  opDef.Output("y").DataType({ge::DT_FLOAT16});
  opDef.AICore().AddConfig("ascend910");
  auto aicoreMap = opDef.AICore().GetAICoreConfigs();
  auto aicore = aicoreMap["ascend910"];
  auto mergeIO = opDef.GetMergeInputsOutputs(aicore);
  EXPECT_EQ(mergeIO.size(), 2U);
}

// ---- op_def_aicore.cc coverage ----

TEST_F(OpDefCovUT, OpAICoreConfig_Output) {
  OpAICoreConfig config;
  config.Input("x").DataType({ge::DT_FLOAT16});
  config.Output("y").DataType({ge::DT_FLOAT16});
  auto &output = config.Output("y");
  EXPECT_EQ(output.GetParamName(), ge::AscendString("y"));
}

TEST_F(OpDefCovUT, OpAICoreConfig_GetInputsAndGetOutputs) {
  OpAICoreConfig config;
  config.Input("x").DataType({ge::DT_FLOAT16});
  config.Output("y").DataType({ge::DT_FLOAT16});
  auto &inputs = config.GetInputs();
  auto &outputs = config.GetOutputs();
  EXPECT_EQ(inputs.size(), 1U);
  EXPECT_EQ(outputs.size(), 1U);
}

TEST_F(OpDefCovUT, OpAICoreConfig_AddCfgItemAndGetCfgKeys) {
  OpAICoreConfig config;
  config.AddCfgItem("key1", "value1");
  config.AddCfgItem("key2", "value2");
  auto &keys = config.GetCfgKeys();
  EXPECT_EQ(keys.size(), 2U);
  auto &val = config.GetConfigValue("key1");
  EXPECT_EQ(val, ge::AscendString("value1"));
}

TEST_F(OpDefCovUT, OpAICoreDef_CopyConstructor) {
  OpDef opDef("TestAICoreCopy");
  opDef.AICore().AddConfig("ascend910");
  OpAICoreDef &aicore = opDef.AICore();
  OpAICoreDef copy(aicore);
  auto configs = copy.GetAICoreConfigs();
  EXPECT_TRUE(configs.find("ascend910") != configs.end());
}

TEST_F(OpDefCovUT, OpAICoreDef_AssignmentOperator) {
  OpDef opDef1("TestAICoreAssign1");
  opDef1.AICore().AddConfig("ascend910");
  OpDef opDef2("TestAICoreAssign2");
  opDef2.AICore() = opDef1.AICore();
  auto configs = opDef2.AICore().GetAICoreConfigs();
  EXPECT_TRUE(configs.find("ascend910") != configs.end());
}

TEST_F(OpDefCovUT, OpAICoreDef_Log) {
  OpDef opDef("TestAICoreLog");
  opDef.AICore().Log("TestOp", "test info");
}

// ---- op_def_aicpu.cc coverage ----

TEST_F(OpDefCovUT, OpAICPUDef_CopyConstructor) {
  OpDef opDef("TestAICPUCopy");
  opDef.AICPU().ExtendCfgInfo("key1", "value1");
  OpAICPUDef &aicpu = opDef.AICPU();
  OpAICPUDef copy(aicpu);
  EXPECT_EQ(copy.GetCfgKeys().size(), 9U);
}

TEST_F(OpDefCovUT, OpAICPUDef_AssignmentOperator) {
  OpDef opDef1("TestAICPUAssign1");
  opDef1.AICPU().ExtendCfgInfo("key1", "value1");
  OpDef opDef2("TestAICPUAssign2");
  opDef2.AICPU() = opDef1.AICPU();
  EXPECT_EQ(opDef2.AICPU().GetCfgKeys().size(), 9U);
}

TEST_F(OpDefCovUT, OpAICPUDef_GetCfgKeysAndGetConfigValue) {
  OpDef opDef("TestAICPUCfg");
  opDef.AICPU().ExtendCfgInfo("key1", "value1");
  auto &keys = opDef.AICPU().GetCfgKeys();
  EXPECT_EQ(keys.size(), 9U);
  auto &val = opDef.AICPU().GetConfigValue("key1");
  EXPECT_EQ(val, ge::AscendString("value1"));
}

TEST_F(OpDefCovUT, OpAICPUDef_EraseCfgInfo) {
  OpDef opDef("TestAICPUErase");
  opDef.AICPU().ExtendCfgInfo("key1", "value1");
  opDef.AICPU().ExtendCfgInfo("key1", "");
  EXPECT_EQ(opDef.AICPU().GetCfgKeys().size(), 9U);
}

// ---- op_def_hostcpu.cc coverage ----

TEST_F(OpDefCovUT, OpHostCPUDef_CopyConstructor) {
  OpDef opDef("TestHostCPUCopy");
  opDef.HostCPU().ExtendCfgInfo("key1", "value1");
  OpHostCPUDef &hostcpu = opDef.HostCPU();
  OpHostCPUDef copy(hostcpu);
  EXPECT_EQ(copy.GetCfgKeys().size(), 9U);
}

TEST_F(OpDefCovUT, OpHostCPUDef_AssignmentOperator) {
  OpDef opDef1("TestHostCPUAssign1");
  opDef1.HostCPU().ExtendCfgInfo("key1", "value1");
  OpDef opDef2("TestHostCPUAssign2");
  opDef2.HostCPU() = opDef1.HostCPU();
  EXPECT_EQ(opDef2.HostCPU().GetCfgKeys().size(), 9U);
}

TEST_F(OpDefCovUT, OpHostCPUDef_GetCfgKeysAndGetConfigValue) {
  OpDef opDef("TestHostCPUCfg");
  opDef.HostCPU().ExtendCfgInfo("key1", "value1");
  auto &keys = opDef.HostCPU().GetCfgKeys();
  EXPECT_EQ(keys.size(), 9U);
  auto &val = opDef.HostCPU().GetConfigValue("key1");
  EXPECT_EQ(val, ge::AscendString("value1"));
}

TEST_F(OpDefCovUT, OpHostCPUDef_EraseCfgInfo) {
  OpDef opDef("TestHostCPUErase");
  opDef.HostCPU().ExtendCfgInfo("key1", "value1");
  opDef.HostCPU().ExtendCfgInfo("key1", "");
  EXPECT_EQ(opDef.HostCPU().GetCfgKeys().size(), 9U);
}

// ---- op_def_attr.cc coverage ----

TEST_F(OpDefCovUT, OpAttrDef_CopyConstructor) {
  OpDef opDef("TestAttrCopy");
  auto &attr = opDef.Attr("attr1").AttrType(REQUIRED).String();
  OpAttrDef copy(attr);
  EXPECT_EQ(copy.GetName(), ge::AscendString("attr1"));
}

TEST_F(OpDefCovUT, OpAttrDef_AssignmentOperator) {
  OpDef opDef1("TestAttrAssign1");
  opDef1.Attr("attr1").AttrType(REQUIRED).String();
  OpDef opDef2("TestAttrAssign2");
  auto &attr2 = opDef2.Attr("attr2").AttrType(OPTIONAL).Int();
  attr2 = opDef1.Attr("attr1");
  EXPECT_EQ(attr2.GetName(), ge::AscendString("attr1"));
}

TEST_F(OpDefCovUT, OpAttrDef_Float) {
  OpDef opDef("TestAttrFloat");
  auto &attr = opDef.Attr("attr1").AttrType(REQUIRED).Float();
  EXPECT_EQ(attr.GetName(), ge::AscendString("attr1"));
}

TEST_F(OpDefCovUT, OpAttrDef_Int) {
  OpDef opDef("TestAttrInt");
  auto &attr = opDef.Attr("attr1").AttrType(REQUIRED).Int();
  EXPECT_EQ(attr.GetName(), ge::AscendString("attr1"));
}

TEST_F(OpDefCovUT, OpAttrDef_ListBool) {
  OpDef opDef("TestAttrListBool");
  auto &attr = opDef.Attr("attr1").AttrType(REQUIRED).ListBool();
  EXPECT_EQ(attr.GetName(), ge::AscendString("attr1"));
}

TEST_F(OpDefCovUT, OpAttrDef_ListFloat) {
  OpDef opDef("TestAttrListFloat");
  auto &attr = opDef.Attr("attr1").AttrType(REQUIRED).ListFloat();
  EXPECT_EQ(attr.GetName(), ge::AscendString("attr1"));
}

TEST_F(OpDefCovUT, OpAttrDef_ListInt) {
  OpDef opDef("TestAttrListInt");
  auto &attr = opDef.Attr("attr1").AttrType(REQUIRED).ListInt();
  EXPECT_EQ(attr.GetName(), ge::AscendString("attr1"));
}

TEST_F(OpDefCovUT, OpAttrDef_ListListInt) {
  OpDef opDef("TestAttrListListInt");
  auto &attr = opDef.Attr("attr1").AttrType(REQUIRED).ListListInt();
  EXPECT_EQ(attr.GetName(), ge::AscendString("attr1"));
}

TEST_F(OpDefCovUT, OpAttrDef_GetNameAndIsRequired) {
  OpDef opDef("TestAttrGetName");
  auto &attr = opDef.Attr("attr1").AttrType(REQUIRED).String();
  EXPECT_EQ(attr.GetName(), ge::AscendString("attr1"));
  EXPECT_TRUE(attr.IsRequired());
}

// ---- op_def_factory.cc coverage ----

TEST_F(OpDefCovUT, OpDefFactory_OpDefRegister) {
  auto creator = [](const char *) -> OpDef { return OpDef("TestRegisterOp"); };
  int ret = OpDefFactory::OpDefRegister("TestRegisterOpType", creator);
  EXPECT_GE(ret, 0);
}

// ---- op_def_mc2.cc coverage ----

TEST_F(OpDefCovUT, OpMC2Def_CopyConstructor) {
  OpDef opDef("TestMC2Copy");
  opDef.MC2().HcclGroup("group1");
  OpMC2Def &mc2 = opDef.MC2();
  OpMC2Def copy(mc2);
  auto groups = copy.GetHcclGroups();
  EXPECT_EQ(groups.size(), 1U);
}

TEST_F(OpDefCovUT, OpMC2Def_AssignmentOperator) {
  OpDef opDef1("TestMC2Assign1");
  opDef1.MC2().HcclGroup("group1");
  OpDef opDef2("TestMC2Assign2");
  opDef2.MC2() = opDef1.MC2();
  auto groups = opDef2.MC2().GetHcclGroups();
  EXPECT_EQ(groups.size(), 1U);
}

// ---- op_def_param.cc coverage ----

TEST_F(OpDefCovUT, OpParamDef_MergeParam) {
  OpDef opDef("TestParamMerge");
  auto &param1 = opDef.Input("x").DataType({ge::DT_FLOAT16}).Format({ge::FORMAT_ND});
  auto &param2 = opDef.Input("x").DataType({ge::DT_FLOAT}).Format({ge::FORMAT_NCHW});
  param1.MergeParam(param2);
}

TEST_F(OpDefCovUT, OpParamDef_IsDtypeAndIsDtypeList) {
  OpDef opDef("TestParamDtypeCheck");
  auto &param1 = opDef.Input("x").DataType({ge::DT_FLOAT16});
  EXPECT_TRUE(param1.IsDtype());
  EXPECT_FALSE(param1.IsDtypeList());
  auto &param2 = opDef.Input("y").DataTypeList({ge::DT_FLOAT16, ge::DT_FLOAT});
  EXPECT_FALSE(param2.IsDtype());
  EXPECT_TRUE(param2.IsDtypeList());
}

TEST_F(OpDefCovUT, OpParamDef_IsFormatAndIsFormatList) {
  OpDef opDef("TestParamFormatCheck");
  auto &param1 = opDef.Input("x").Format({ge::FORMAT_ND});
  EXPECT_TRUE(param1.IsFormat());
  EXPECT_FALSE(param1.IsFormatList());
  auto &param2 = opDef.Input("y").FormatList({ge::FORMAT_ND, ge::FORMAT_NCHW});
  EXPECT_FALSE(param2.IsFormat());
  EXPECT_TRUE(param2.IsFormatList());
}

TEST_F(OpDefCovUT, OpParamDef_IsScalarOrScalarList) {
  OpDef opDef("TestParamScalar");
  auto &param1 = opDef.Input("x").DataType({ge::DT_FLOAT16}).Scalar();
  EXPECT_TRUE(param1.IsScalarOrScalarList());
  auto &param2 = opDef.Input("y").DataType({ge::DT_FLOAT16}).ScalarList();
  EXPECT_TRUE(param2.IsScalarOrScalarList());
}

TEST_F(OpDefCovUT, OpParamDef_IsScalarTypeSetAndNameSet) {
  OpDef opDef("TestParamScalarType");
  auto &param1 = opDef.Input("x").DataType({ge::DT_FLOAT16}).Scalar().To(ge::DT_INT32).To("x");
  EXPECT_TRUE(param1.IsScalarTypeSet());
  EXPECT_TRUE(param1.IsScalarNameSet());
}

TEST_F(OpDefCovUT, OpParamDef_IsValueDepend) {
  OpDef opDef("TestParamValueDepend");
  auto &param1 = opDef.Input("x").DataType({ge::DT_FLOAT16}).ValueDepend(Option::REQUIRED);
  EXPECT_TRUE(param1.IsValueDepend());
  auto &param2 = opDef.Input("y").DataType({ge::DT_FLOAT16});
  EXPECT_FALSE(param2.IsValueDepend());
}

TEST_F(OpDefCovUT, OpParamDef_GetDataTypesList) {
  OpDef opDef("TestParamGetDtypesList");
  auto &param = opDef.Input("x").DataTypeList({ge::DT_FLOAT16, ge::DT_FLOAT});
  auto &types = param.GetDataTypesList();
  EXPECT_EQ(types.size(), 2U);
}

TEST_F(OpDefCovUT, OpParamDef_GetFormatsList) {
  OpDef opDef("TestParamGetFormatsList");
  auto &param = opDef.Input("x").FormatList({ge::FORMAT_ND, ge::FORMAT_NCHW});
  auto &formats = param.GetFormatsList();
  EXPECT_EQ(formats.size(), 2U);
}

TEST_F(OpDefCovUT, OpParamDef_IsSetDtypeForBinAndFormatForBin) {
  OpDef opDef("TestParamForBin");
  auto &param = opDef.Input("x").DataType({ge::DT_FLOAT16}).DataTypeForBinQuery({ge::DT_FLOAT});
  EXPECT_TRUE(param.IsSetDtypeForBin());
  auto &param2 = opDef.Input("y").Format({ge::FORMAT_ND}).FormatForBinQuery({ge::FORMAT_NCHW});
  EXPECT_TRUE(param2.IsSetFormatForBin());
}

TEST_F(OpDefCovUT, OpParamDef_GetScalarName) {
  OpDef opDef("TestParamScalarName");
  auto &param = opDef.Input("x").DataType({ge::DT_FLOAT16}).Scalar().To("y");
  auto &name = param.GetScalarName();
  EXPECT_EQ(name, ge::AscendString("y"));
}

TEST_F(OpDefCovUT, OpDef_MergeParam) {
  OpDef opDef("TestMergeParamCov");
  opDef.Input("x").DataType({ge::DT_FLOAT16}).Format({ge::FORMAT_ND});
  opDef.Output("y").DataType({ge::DT_FLOAT16}).Format({ge::FORMAT_ND});
  std::vector<OpParamDef> merge;
  std::vector<OpParamDef> aicore_params;
  auto &input = opDef.GetInputs();
  aicore_params = input;
  opDef.MergeParam(merge, aicore_params);
  EXPECT_FALSE(merge.empty());
}

TEST_F(OpDefCovUT, OpDef_DfsDataType) {
  OpDef opDef("TestDfsDataTypeCov");
  opDef.Input("x").DataType({ge::DT_FLOAT16, ge::DT_FLOAT}).Format({ge::FORMAT_ND});
  opDef.Output("y").DataType({ge::DT_FLOAT16, ge::DT_FLOAT}).Format({ge::FORMAT_ND});
  std::vector<OpParamDef> all_param = opDef.GetInputs();
  for (auto &out : opDef.GetOutputs()) {
    all_param.push_back(out);
  }
  OpDef::DfsParam dfs_param;
  opDef.DfsDataType(dfs_param, all_param, 0U, 0U);
}

TEST_F(OpDefCovUT, OpDef_DfsFormat) {
  OpDef opDef("TestDfsFormatCov");
  opDef.Input("x").DataType({ge::DT_FLOAT16}).Format({ge::FORMAT_ND, ge::FORMAT_NCHW});
  opDef.Output("y").DataType({ge::DT_FLOAT16}).Format({ge::FORMAT_ND, ge::FORMAT_NCHW});
  std::vector<OpParamDef> all_param = opDef.GetInputs();
  for (auto &out : opDef.GetOutputs()) {
    all_param.push_back(out);
  }
  OpDef::DfsParam dfs_param;
  opDef.DfsFormat(dfs_param, all_param, 0U, 0U);
}

TEST_F(OpDefCovUT, OpDef_DfsFullPermutation) {
  OpDef opDef("TestDfsFullPermCov");
  opDef.Input("x").DataType({ge::DT_FLOAT16, ge::DT_FLOAT}).Format({ge::FORMAT_ND, ge::FORMAT_NCHW});
  opDef.Output("y").DataType({ge::DT_FLOAT16, ge::DT_FLOAT}).Format({ge::FORMAT_ND, ge::FORMAT_NCHW});
  std::vector<OpParamDef> all_param = opDef.GetInputs();
  for (auto &out : opDef.GetOutputs()) {
    all_param.push_back(out);
  }
  OpDef::DfsParam dfs_param;
  opDef.DfsFullPermutation(dfs_param, all_param, 0U, 0U);
}

TEST_F(OpDefCovUT, OpDef_IsNonListTypes) {
  OpDef opDef("TestIsNonListTypesCov");
  auto &param1 = opDef.Input("x").DataType({ge::DT_FLOAT16});
  EXPECT_TRUE(opDef.IsNonListTypes(param1));
  auto &param2 = opDef.Input("y").DataTypeList({ge::DT_FLOAT16, ge::DT_FLOAT});
  EXPECT_FALSE(opDef.IsNonListTypes(param2));
}

TEST_F(OpDefCovUT, OpDef_IsNonListFormats) {
  OpDef opDef("TestIsNonListFormatsCov");
  auto &param1 = opDef.Input("x").Format({ge::FORMAT_ND});
  EXPECT_TRUE(opDef.IsNonListFormats(param1));
  auto &param2 = opDef.Input("y").FormatList({ge::FORMAT_ND, ge::FORMAT_NCHW});
  EXPECT_FALSE(opDef.IsNonListFormats(param2));
}

TEST_F(OpDefCovUT, OpDef_GetNonListLen) {
  OpDef opDef("TestGetNonListLenCov");
  opDef.Input("x").DataType({ge::DT_FLOAT16}).Format({ge::FORMAT_ND});
  opDef.Input("y").DataType({ge::DT_FLOAT}).Format({ge::FORMAT_NCHW});
  opDef.Output("z").DataType({ge::DT_FLOAT16}).Format({ge::FORMAT_ND});
  auto &inputs = opDef.GetInputs();
  auto &outputs = opDef.GetOutputs();
  uint32_t len = opDef.GetNonListLen(inputs, outputs);
  EXPECT_GE(len, 0U);
}

TEST_F(OpDefCovUT, OpDef_UpdateDtypeImpl) {
  OpDef opDef("TestUpdateDtypeCov");
  opDef.Input("x").DataType({ge::DT_FLOAT16, ge::DT_FLOAT}).Format({ge::FORMAT_ND});
  opDef.Output("y").DataType({ge::DT_FLOAT16}).Format({ge::FORMAT_ND});
  std::vector<OpParamDef> all_param = opDef.GetInputs();
  OpDef::DfsParam dfs_param;
  opDef.DfsDataType(dfs_param, all_param, 0U, 0U);
  OpParamDef param = opDef.GetInputs()[0];
  opDef.UpdateDtypeImpl(dfs_param, param, 0U);
}

TEST_F(OpDefCovUT, OpDef_UpdateFormatImpl) {
  OpDef opDef("TestUpdateFormatCov");
  opDef.Input("x").DataType({ge::DT_FLOAT16}).Format({ge::FORMAT_ND, ge::FORMAT_NCHW});
  opDef.Output("y").DataType({ge::DT_FLOAT16}).Format({ge::FORMAT_ND});
  std::vector<OpParamDef> all_param = opDef.GetInputs();
  OpDef::DfsParam dfs_param;
  opDef.DfsFormat(dfs_param, all_param, 0U, 0U);
  OpParamDef param = opDef.GetInputs()[0];
  opDef.UpdateFormatImpl(dfs_param, param, 0U);
}

TEST_F(OpDefCovUT, OpDef_UpdateInput) {
  OpDef opDef("TestUpdateInputCov");
  opDef.Input("x").DataType({ge::DT_FLOAT16, ge::DT_FLOAT}).Format({ge::FORMAT_ND, ge::FORMAT_NCHW});
  opDef.Output("y").DataType({ge::DT_FLOAT16}).Format({ge::FORMAT_ND});
  std::vector<OpParamDef> all_param = opDef.GetInputs();
  for (auto &out : opDef.GetOutputs()) {
    all_param.push_back(out);
  }
  OpDef::DfsParam dfs_param;
  opDef.DfsFullPermutation(dfs_param, all_param, 0U, 0U);
  std::vector<OpParamDef> input = opDef.GetInputs();
  opDef.UpdateInput(dfs_param, input);
}

TEST_F(OpDefCovUT, OpDef_UpdateOutput) {
  OpDef opDef("TestUpdateOutputCov");
  opDef.Input("x").DataType({ge::DT_FLOAT16, ge::DT_FLOAT}).Format({ge::FORMAT_ND, ge::FORMAT_NCHW});
  opDef.Output("y").DataType({ge::DT_FLOAT16}).Format({ge::FORMAT_ND});
  std::vector<OpParamDef> all_param = opDef.GetInputs();
  for (auto &out : opDef.GetOutputs()) {
    all_param.push_back(out);
  }
  OpDef::DfsParam dfs_param;
  opDef.DfsFullPermutation(dfs_param, all_param, 0U, 0U);
  std::vector<OpParamDef> output = opDef.GetOutputs();
  opDef.UpdateOutput(dfs_param, output);
}

TEST_F(OpDefCovUT, OpDef_SetPermutedParam) {
  OpDef opDef("TestSetPermutedParamCov");
  opDef.Input("x").DataType({ge::DT_FLOAT16, ge::DT_FLOAT}).Format({ge::FORMAT_ND, ge::FORMAT_NCHW});
  opDef.Output("y").DataType({ge::DT_FLOAT16}).Format({ge::FORMAT_ND});
  std::vector<OpParamDef> all_param = opDef.GetInputs();
  for (auto &out : opDef.GetOutputs()) {
    all_param.push_back(out);
  }
  OpDef::DfsParam dfs_param;
  opDef.DfsFullPermutation(dfs_param, all_param, 0U, 0U);
  std::vector<OpParamDef> input = opDef.GetInputs();
  std::vector<OpParamDef> output = opDef.GetOutputs();
  opDef.SetPermutedParam(dfs_param, input, output);
}

TEST_F(OpDefCovUT, OpDef_CheckIncompatible) {
  OpDef opDef("TestCheckIncompatibleCov");
  opDef.Input("x").DataType({ge::DT_FLOAT16}).Format({ge::FORMAT_ND});
  opDef.Output("y").DataType({ge::DT_FLOAT16}).Format({ge::FORMAT_ND});
  std::vector<OpParamDef> all_param = opDef.GetInputs();
  for (auto &out : opDef.GetOutputs()) {
    all_param.push_back(out);
  }
  opDef.CheckIncompatible(all_param);
}

TEST_F(OpDefCovUT, OpDef_FullPermutation) {
  OpDef opDef("TestFullPermutationCov");
  opDef.Input("x").DataType({ge::DT_FLOAT16, ge::DT_FLOAT}).Format({ge::FORMAT_ND, ge::FORMAT_NCHW});
  opDef.Output("y").DataType({ge::DT_FLOAT16}).Format({ge::FORMAT_ND});
  std::vector<OpParamDef> input = opDef.GetInputs();
  std::vector<OpParamDef> output = opDef.GetOutputs();
  opDef.FullPermutation(input, output);
}

TEST_F(OpDefCovUT, OpDef_SetDefaultND) {
  OpDef opDef("TestSetDefaultNDCov");
  opDef.Input("x").DataType({ge::DT_FLOAT16}).Format({ge::FORMAT_ND});
  opDef.Output("y").DataType({ge::DT_FLOAT16}).Format({ge::FORMAT_ND});
  std::vector<OpParamDef> defs = opDef.GetInputs();
  for (auto &out : opDef.GetOutputs()) {
    defs.push_back(out);
  }
  opDef.SetDefaultND(defs);
}

TEST_F(OpDefCovUT, OpDef_FollowListImpl) {
  OpDef opDef("TestFollowListImplCov");
  opDef.Input("x").DataType({ge::DT_FLOAT16, ge::DT_FLOAT}).Format({ge::FORMAT_ND});
  opDef.Output("y").Follow("x").DataType({ge::DT_FLOAT16, ge::DT_FLOAT}).Format({ge::FORMAT_ND});
  opDef.FollowImpl();
  std::vector<OpParamDef> all_param = opDef.GetInputs();
  for (auto &out : opDef.GetOutputs()) {
    all_param.push_back(out);
  }
  OpDef::DfsParam dfs_param;
  opDef.DfsFullPermutation(dfs_param, all_param, 0U, 0U);
  std::vector<OpParamDef> input = opDef.GetInputs();
  std::vector<OpParamDef> output = opDef.GetOutputs();
  opDef.FollowListImpl(dfs_param, input, output);
}
}  // namespace ops
