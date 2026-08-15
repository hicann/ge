/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 ("the License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>

#include <limits>
#include <memory>
#include <vector>

#include "common/model/ge_model.h"
#include "common/om2/codegen/task_args_manager/om2_model_args_utils.h"
#include "framework/common/framework_types_internal.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/op_desc.h"
#include "graph/utils/tensor_utils.h"

namespace ge {
namespace om2 {
namespace {

struct ModelFixture {
  OpDescPtr op_desc;
};

ModelFixture BuildAddOp(size_t input_num, size_t output_num) {
  ModelFixture fixture;
  fixture.op_desc = std::make_shared<OpDesc>("add", "Add");
  for (size_t i = 0; i < input_num; ++i) {
    GeTensorDesc input_desc(GeShape({1}), FORMAT_ND, DT_FLOAT);
    TensorUtils::SetSize(input_desc, 4);
    EXPECT_EQ(fixture.op_desc->AddInputDesc(input_desc), GRAPH_SUCCESS);
  }
  for (size_t i = 0; i < output_num; ++i) {
    GeTensorDesc output_desc(GeShape({1}), FORMAT_ND, DT_FLOAT);
    TensorUtils::SetSize(output_desc, 4);
    EXPECT_EQ(fixture.op_desc->AddOutputDesc(output_desc), GRAPH_SUCCESS);
  }
  EXPECT_NE(fixture.op_desc, nullptr);
  return fixture;
}

void SetTensorSize(const GeTensorDescPtr &desc, int64_t size) {
  ASSERT_NE(desc, nullptr);
  TensorUtils::SetSize(*desc, size);
}

RuntimeParam BuildRuntimeParamForAddressTests() {
  RuntimeParam param;
  param.mem_size = 0x1000U;
  param.logic_mem_base = 0x1000U;
  param.mem_base = 0x50000000U;
  param.weight_size = 0x1000U;
  param.logic_weight_base = 0x9000U;
  param.weight_base = 0x60000000U;
  param.var_size = 0x1000U;
  param.logic_var_base = 0x20000U;
  param.var_base = 0x70000000U;
  param.host_mem_size = 0x1000U;
  param.host_logic_mem_base = 0x900000000U;
  param.host_mem_base = 0x80000000U;
  param.host_svm_size = 0x1000U;
  param.host_svm_logic_mem_base = 0xA00000000U;
  param.host_svm_mem_base = 0x90000000U;

  auto &p2p_info = param.memory_infos[RT_MEMORY_P2P_DDR];
  p2p_info.memory_type = RT_MEMORY_P2P_DDR;
  p2p_info.logic_memory_base = 0x1000U + param.mem_size;
  p2p_info.memory_size = 0x1000U;
  p2p_info.memory_base = reinterpret_cast<uint8_t *>(0xA0000000U);

  auto &host_info = param.memory_infos[RT_MEMORY_HOST];
  host_info.memory_type = RT_MEMORY_HOST;
  host_info.logic_memory_base = param.host_logic_mem_base;
  host_info.memory_size = param.host_mem_size;
  host_info.memory_base = reinterpret_cast<uint8_t *>(param.host_mem_base);

  auto &host_svm_info = param.memory_infos[RT_MEMORY_HOST_SVM];
  host_svm_info.memory_type = RT_MEMORY_HOST_SVM;
  host_svm_info.logic_memory_base = param.host_svm_logic_mem_base;
  host_svm_info.memory_size = param.host_svm_size;
  host_svm_info.memory_base = reinterpret_cast<uint8_t *>(param.host_svm_mem_base);

  return param;
}

std::shared_ptr<GeModel> BuildGeModel() {
  auto ge_model = std::make_shared<GeModel>();
  (void)AttrUtils::SetInt(ge_model, ATTR_MODEL_MEMORY_SIZE, 0x2000);
  (void)AttrUtils::SetInt(ge_model, ATTR_MODEL_WEIGHT_SIZE, 0x400);
  (void)AttrUtils::SetInt(ge_model, MODEL_ATTR_TASK_GEN_BASE_ADDR, 0x1000);
  (void)AttrUtils::SetInt(ge_model, MODEL_ATTR_TASK_GEN_WEIGHT_ADDR, 0x2000);
  (void)AttrUtils::SetInt(ge_model, ATTR_MODEL_TASK_GEN_VAR_ADDR, 0x3000);
  (void)AttrUtils::SetInt(ge_model, ATTR_MODEL_VAR_SIZE, 0x1000);
  (void)AttrUtils::SetInt(ge_model, ATTR_MODEL_ZERO_COPY_MEMORY_SIZE, 0);
  (void)AttrUtils::SetInt(ge_model, MODEL_ATTR_HOST_MEMORY_SIZE, 0x1000);
  (void)AttrUtils::SetInt(ge_model, MODEL_ATTR_TASK_GEN_HOST_BASE_ADDR, 0x4000);
  (void)AttrUtils::SetInt(ge_model, MODEL_ATTR_HOST_SVM_SIZE, 0x1000);
  (void)AttrUtils::SetInt(ge_model, MODEL_ATTR_TASK_GEN_HOST_SVM_BASE_ADDR, 0x5000);
  (void)AttrUtils::SetInt(ge_model, ATTR_MODEL_P2P_MEMORY_SIZE, 0x1000);
  (void)AttrUtils::SetInt(ge_model, ATTR_MODEL_SESSION_SCOPE_MEMORY_SIZE, 0x1000);
  (void)AttrUtils::SetInt(ge_model, MODEL_ATTR_SESSION_ID, 1);
  (void)AttrUtils::SetInt(ge_model, ATTR_MODEL_STREAM_NUM, 2);
  (void)AttrUtils::SetInt(ge_model, ATTR_MODEL_NOTIFY_NUM, 3);
  (void)AttrUtils::SetInt(ge_model, ATTR_MODEL_EVENT_NUM, 4);
  (void)AttrUtils::SetInt(ge_model, ATTR_MODEL_LABEL_NUM, 5);
  (void)AttrUtils::SetInt(ge_model, ATTR_MODEL_BATCH_NUM, 6);
  (void)AttrUtils::SetListListInt(
      ge_model, ATTR_MODEL_SUB_MEMORY_INFO,
      std::vector<std::vector<int64_t>>{
          {RT_MEMORY_HBM, 0x1000, 0x1000, 0}, {RT_MEMORY_HBM, 0x2000, 0x1000, 1}, {RT_MEMORY_HBM, 0x3000, 0x1000, 0}});
  return ge_model;
}

}  // namespace

class ModelUtilsUT : public testing::Test {};

TEST_F(ModelUtilsUT, GetSizeHelpers_Work) {
  auto fixture = BuildAddOp(2U, 2U);
  auto op_desc = fixture.op_desc;

  SetTensorSize(op_desc->MutableInputDesc(0), 64);
  SetTensorSize(op_desc->MutableInputDesc(1), 32);
  SetTensorSize(op_desc->MutableOutputDesc(0), 16);
  SetTensorSize(op_desc->MutableOutputDesc(1), 8);
  op_desc->MutableOutputDesc(1)->SetDataType(DT_STRING);
  (void)AttrUtils::SetInt(op_desc, "_op_max_size", 128);
  op_desc->SetOutputOffset({0x10, 0x20});
  op_desc->SetWorkspace({0x10, 0x20});
  op_desc->SetWorkspaceBytes({0x100, 0x200});

  EXPECT_EQ(ModelUtils::GetInputSize(op_desc), (std::vector<int64_t>{64, 32}));
  EXPECT_EQ(ModelUtils::GetOutputSize(op_desc), (std::vector<int64_t>{16, 128}));
  EXPECT_EQ(ModelUtils::GetWorkspaceSize(op_desc), (std::vector<int64_t>{0x100, 0x200}));
}

TEST_F(ModelUtilsUT, GetSizeHelpers_HandleInvalidInput) {
  auto fixture = BuildAddOp(2U, 1U);
  auto op_desc = fixture.op_desc;
  op_desc->SetOutputOffset({});
  op_desc->SetWorkspace({0x10});
  op_desc->SetWorkspaceBytes({});

  EXPECT_TRUE(ModelUtils::GetInputSize(nullptr).empty());
  EXPECT_TRUE(ModelUtils::GetOutputSize(op_desc).empty());
  EXPECT_TRUE(ModelUtils::GetWorkspaceSize(op_desc).empty());
}

TEST_F(ModelUtilsUT, GetInputAndOutputAddrs_Work) {
  auto fixture = BuildAddOp(2U, 2U);
  auto op_desc = fixture.op_desc;
  auto runtime_param = BuildRuntimeParamForAddressTests();

  op_desc->SetIsInputConst({true, false});
  op_desc->SetInputOffset({0x0, 0x10});
  op_desc->SetOutputOffset({0x40, 0x80});

  auto input0 = op_desc->MutableInputDesc(0);
  SetTensorSize(input0, 64);
  input0->SetShape(GeShape({1, 1}));
  input0->SetOriginShape(GeShape({1, 1}));
  TensorUtils::SetDataOffset(*input0, 0);

  auto input1 = op_desc->MutableInputDesc(1);
  SetTensorSize(input1, 32);
  input1->SetShape(GeShape({1, 1}));
  input1->SetOriginShape(GeShape({1, 1}));
  TensorUtils::SetDataOffset(*input1, 0x10);
  (void)AttrUtils::SetInt(input1, ATTR_NAME_TENSOR_MEM_TYPE, RT_MEMORY_P2P_DDR);

  auto output0 = op_desc->MutableOutputDesc(0);
  SetTensorSize(output0, 16);
  output0->SetShape(GeShape({1, 1}));
  output0->SetOriginShape(GeShape({1, 1}));

  auto output1 = op_desc->MutableOutputDesc(1);
  SetTensorSize(output1, 8);
  output1->SetShape(GeShape({1, 1}));
  output1->SetOriginShape(GeShape({1, 1}));

  const auto input_addrs = ModelUtils::GetInputAddrs(runtime_param, op_desc);
  const auto input_addrs_value = ModelUtils::GetInputAddrsValue(runtime_param, op_desc);
  const auto input_data_addrs = ModelUtils::GetInputDataAddrs(runtime_param, op_desc);
  std::vector<uint64_t> input_data_mem_type;
  const auto input_data_addrs_value = ModelUtils::GetInputDataAddrsValue(runtime_param, op_desc, input_data_mem_type);
  const auto output_addrs = ModelUtils::GetOutputAddrs(runtime_param, op_desc);
  const auto output_addrs_value = ModelUtils::GetOutputAddrsValue(runtime_param, op_desc);
  const auto output_data_addrs = ModelUtils::GetOutputDataAddrs(runtime_param, op_desc);
  const auto output_data_addrs_value = ModelUtils::GetOutputDataAddrsValue(runtime_param, op_desc);

  EXPECT_EQ(input_addrs.size(), 2U);
  EXPECT_EQ(input_addrs_value.size(), 2U);
  EXPECT_EQ(input_data_addrs.size(), 2U);
  EXPECT_EQ(input_data_addrs_value.size(), 2U);
  EXPECT_EQ(output_addrs.size(), 2U);
  EXPECT_EQ(output_addrs_value.size(), 2U);
  EXPECT_EQ(output_data_addrs.size(), 2U);
  EXPECT_EQ(output_data_addrs_value.size(), 2U);
}

TEST_F(ModelUtilsUT, GetInputOutputDescAddrs_CoversFillSinkTensorDesc) {
  auto fixture = BuildAddOp(0U, 1U);
  auto op_desc = fixture.op_desc;
  auto output_desc = op_desc->MutableOutputDesc(0);
  ASSERT_NE(output_desc, nullptr);

  output_desc->SetShape(GeShape({2, 3}));
  output_desc->SetOriginShape(GeShape({2, 3}));
  output_desc->SetFormat(FORMAT_ND);
  output_desc->SetDataType(DT_FLOAT);
  (void)AttrUtils::SetInt(output_desc, ATTR_NAME_TENSOR_DESC_MEM_OFFSET, 0x100);

  std::vector<uint8_t> backing(0x1000, 0);
  RuntimeParam runtime_param = BuildRuntimeParamForAddressTests();
  runtime_param.mem_base = reinterpret_cast<uintptr_t>(backing.data());
  runtime_param.mem_size = backing.size();

  std::vector<void *> addrs{reinterpret_cast<void *>(0x12345678UL)};
  std::vector<uint64_t> mem_types;
  EXPECT_EQ(
      ModelUtils::GetInputOutputDescAddrs(runtime_param, op_desc, op_desc->GetAllOutputsDescPtr(), mem_types, addrs),
      SUCCESS);
  EXPECT_EQ(addrs[0], reinterpret_cast<void *>(backing.data() + 0x100));

  auto bad_fixture = BuildAddOp(0U, 1U);
  auto bad_op_desc = bad_fixture.op_desc;
  auto bad_output_desc = bad_op_desc->MutableOutputDesc(0);
  ASSERT_NE(bad_output_desc, nullptr);
  bad_output_desc->SetShape(GeShape(std::vector<int64_t>(33, 1)));
  bad_output_desc->SetOriginShape(GeShape(std::vector<int64_t>(33, 1)));
  bad_output_desc->SetFormat(FORMAT_ND);
  bad_output_desc->SetDataType(DT_FLOAT);
  (void)AttrUtils::SetInt(bad_output_desc, ATTR_NAME_TENSOR_DESC_MEM_OFFSET, 0x200);

  std::vector<void *> bad_addrs{reinterpret_cast<void *>(0x87654321UL)};
  EXPECT_EQ(ModelUtils::GetInputOutputDescAddrs(runtime_param, bad_op_desc, bad_op_desc->GetAllOutputsDescPtr(),
                                                mem_types, bad_addrs),
            FAILED);
}

TEST_F(ModelUtilsUT, GetOutputDataAddrs_OptionalOutputAndMemTypes) {
  auto fixture = BuildAddOp(1U, 3U);
  auto op_desc = fixture.op_desc;
  auto runtime_param = BuildRuntimeParamForAddressTests();

  op_desc->SetOutputOffset({0x40, 0x80, 0xC0});
  SetTensorSize(op_desc->MutableOutputDesc(0), 16);
  SetTensorSize(op_desc->MutableOutputDesc(1), 0);
  SetTensorSize(op_desc->MutableOutputDesc(2), 16);
  (void)AttrUtils::SetInt(op_desc->MutableOutputDesc(1), ATTR_NAME_MEMORY_SIZE_CALC_TYPE,
                          static_cast<int32_t>(MemorySizeCalcType::ALWAYS_EMPTY));

  std::vector<uint64_t> mem_types;
  const auto addrs = ModelUtils::GetOutputDataAddrs(runtime_param, op_desc, mem_types, true);
  EXPECT_EQ(addrs.size(), 3U);
  EXPECT_NE(addrs[0], nullptr);
  EXPECT_EQ(addrs[1], nullptr);
  EXPECT_NE(addrs[2], nullptr);
}

TEST_F(ModelUtilsUT, GetWorkspaceDataAddrs_HandlesPriorityPaths) {
  auto fixture = BuildAddOp(1U, 1U);
  auto op_desc = fixture.op_desc;
  auto runtime_param = BuildRuntimeParamForAddressTests();

  op_desc->SetWorkspace({0x10, 0x20, 0x30, 0x40, 0x50});
  op_desc->SetWorkspaceBytes({0x100, 0x200, 0x300, 0x400, 0x500});
  AttrUtils::SetListInt(
      op_desc, TVM_ATTR_NAME_WORKSPACE_TYPE,
      std::vector<int64_t>{RT_MEMORY_P2P_DDR, RT_MEMORY_L1, kRtMemoryUB, RT_MEMORY_HBM, RT_MEMORY_HBM});
  AttrUtils::SetListInt(
      op_desc, ATTR_NAME_WORKSPACE_TYPE_LIST,
      std::vector<int64_t>{RT_MEMORY_P2P_DDR, RT_MEMORY_L1, kRtMemoryUB, RT_MEMORY_HBM, RT_MEMORY_HBM});
  AttrUtils::SetListInt(op_desc, ATTR_NAME_WORKSPACE_MEMORY_NO_REUSE_SCOPE, std::vector<int32_t>{0, 0, 0, 1, 0});

  const uint64_t session_scope_key = 0x100000000UL | RT_MEMORY_HBM;
  auto &session_scope_info = runtime_param.memory_infos[session_scope_key];
  session_scope_info.memory_type = session_scope_key;
  session_scope_info.logic_memory_base = 0x40U;
  session_scope_info.memory_size = 0x1000U;
  session_scope_info.memory_base = reinterpret_cast<uint8_t *>(0xB0000000U);

  const auto addrs = ModelUtils::GetWorkspaceDataAddrs(runtime_param, op_desc);
  const auto addrs_value = ModelUtils::GetWorkspaceDataAddrsValue(runtime_param, op_desc);

  EXPECT_EQ(addrs.size(), 5U);
  EXPECT_EQ(addrs_value.size(), 5U);
  EXPECT_EQ(addrs[1], reinterpret_cast<void *>(0x20));
  EXPECT_EQ(addrs[2], reinterpret_cast<void *>(0x30));
  EXPECT_NE(addrs[0], nullptr);
  EXPECT_NE(addrs[3], nullptr);
  EXPECT_NE(addrs[4], nullptr);
}

TEST_F(ModelUtilsUT, HelperFunctions_Work) {
  EXPECT_TRUE(ModelUtils::IsSuppoprtAddrRefreshable(static_cast<uint64_t>(MemoryAppType::kMemoryTypeFeatureMap)));
  EXPECT_TRUE(ModelUtils::IsSuppoprtAddrRefreshable(static_cast<uint64_t>(MemoryAppType::kMemoryTypeModelIo)));
  EXPECT_FALSE(ModelUtils::IsSuppoprtAddrRefreshable(static_cast<uint64_t>(MemoryAppType::kMemoryTypeFix)));

  std::vector<uint64_t> mem_types{static_cast<uint64_t>(MemoryAppType::kMemoryTypeFix),
                                  static_cast<uint64_t>(MemoryAppType::kMemoryTypeFeatureMap),
                                  static_cast<uint64_t>(MemoryAppType::kMemoryTypeModelIo)};
  std::vector<uint8_t> flags;
  ModelUtils::GetAddrRefreshableFlagsByMemTypes(mem_types, flags);
  EXPECT_EQ(flags, (std::vector<uint8_t>{0U, 1U, 1U}));

  EXPECT_TRUE(ModelUtils::IsFeatureMapOrModelIoType(kFmMemType));
  EXPECT_TRUE(ModelUtils::IsFeatureMapOrModelIoType(RT_MEMORY_HBM));
  EXPECT_TRUE(ModelUtils::IsFeatureMapOrModelIoType(RT_MEMORY_L2));
  EXPECT_TRUE(ModelUtils::IsFeatureMapOrModelIoType(RT_MEMORY_DEFAULT));
  EXPECT_FALSE(ModelUtils::IsFeatureMapOrModelIoType(kFixMemType));

  EXPECT_TRUE(ModelUtils::IsAICoreKernel(ccKernelType::TE));
  EXPECT_FALSE(ModelUtils::IsAICoreKernel(ccKernelType::INVALID));
}

TEST_F(ModelUtilsUT, GetInputAndOutputDescHelpers_CoverBranches) {
  auto fixture = BuildAddOp(2U, 2U);
  auto op_desc = fixture.op_desc;

  op_desc->SetIsInputConst({true, false});
  SetTensorSize(op_desc->MutableInputDesc(0), 16);
  SetTensorSize(op_desc->MutableInputDesc(1), 16);
  op_desc->MutableInputDesc(1)->SetShape(GeShape({static_cast<int64_t>(INT32_MAX) + 1, 1}));
  op_desc->MutableInputDesc(1)->SetOriginShape(GeShape({static_cast<int64_t>(INT32_MAX) + 1, 1}));

  SetTensorSize(op_desc->MutableOutputDesc(0), 16);
  SetTensorSize(op_desc->MutableOutputDesc(1), 16);
  op_desc->MutableOutputDesc(1)->SetShape(GeShape({static_cast<int64_t>(INT32_MAX) + 1, 1}));
  op_desc->MutableOutputDesc(1)->SetOriginShape(GeShape({static_cast<int64_t>(INT32_MAX) + 1, 1}));

  const auto input_descs = ModelUtils::GetInputDescs(op_desc);
  const auto output_descs = ModelUtils::GetOutputDescs(op_desc);

  EXPECT_EQ(input_descs.size(), 1U);
  EXPECT_EQ(output_descs.size(), 2U);
}

TEST_F(ModelUtilsUT, GetInputAddrs_CoverBranches) {
  auto fixture = BuildAddOp(4U, 1U);
  auto op_desc = fixture.op_desc;
  auto runtime_param = BuildRuntimeParamForAddressTests();

  op_desc->SetIsInputConst({false, false, false, false});
  op_desc->SetInputOffset({0x10, 0x20, 0x30, 0x40});
  for (size_t i = 0; i < 4; ++i) {
    auto tensor_desc = op_desc->MutableInputDesc(static_cast<uint32_t>(i));
    SetTensorSize(tensor_desc, 16);
    tensor_desc->SetShape(GeShape({1, 1}));
    tensor_desc->SetOriginShape(GeShape({1, 1}));
    TensorUtils::SetDataOffset(*tensor_desc, 0);
  }
  (void)AttrUtils::SetInt(op_desc->MutableInputDesc(2), ATTR_NAME_TENSOR_MEM_TYPE, RT_MEMORY_P2P_DDR);
  (void)AttrUtils::SetListInt(op_desc, ATTR_NAME_INPUT_MEM_TYPE_LIST,
                              std::vector<int64_t>{RT_MEMORY_L1, RT_MEMORY_HOST, RT_MEMORY_L2, RT_MEMORY_DEFAULT});
  runtime_param.fileconstant_addr_mapping[0x10] = 0xdead0000U;

  std::vector<uint64_t> input_mem_type;
  const auto input_addrs = ModelUtils::GetInputAddrs(runtime_param, op_desc, input_mem_type, false);
  std::vector<uint64_t> input_data_mem_type;
  const auto input_data_addrs_value = ModelUtils::GetInputDataAddrsValue(runtime_param, op_desc, input_data_mem_type);

  EXPECT_EQ(input_addrs.size(), 4U);
  EXPECT_EQ(input_data_addrs_value.size(), 4U);
  EXPECT_EQ(input_mem_type[0], kConstantMemType);
  EXPECT_EQ(input_mem_type[1], RT_MEMORY_HOST);
  EXPECT_EQ(input_mem_type[2], RT_MEMORY_P2P_DDR);
  EXPECT_EQ(input_mem_type[3], RT_MEMORY_DEFAULT);
  EXPECT_EQ(input_addrs[0], reinterpret_cast<void *>(0xdead0000U));
  EXPECT_NE(input_addrs[1], nullptr);
  EXPECT_NE(input_addrs[2], nullptr);
  EXPECT_NE(input_addrs[3], nullptr);
}

TEST_F(ModelUtilsUT, GetOutputAddrs_CoverBranches) {
  auto fixture = BuildAddOp(1U, 3U);
  auto op_desc = fixture.op_desc;
  auto runtime_param = BuildRuntimeParamForAddressTests();

  op_desc->SetOutputOffset({0x50, 0x60, 0x70});
  auto out0 = op_desc->MutableOutputDesc(0);
  auto out1 = op_desc->MutableOutputDesc(1);
  auto out2 = op_desc->MutableOutputDesc(2);
  SetTensorSize(out0, 16);
  SetTensorSize(out1, 0);
  SetTensorSize(out2, 16);
  out0->SetShape(GeShape({1, 1}));
  out0->SetOriginShape(GeShape({1, 1}));
  out1->SetShape(GeShape({1, 1}));
  out1->SetOriginShape(GeShape({1, 1}));
  out2->SetShape(GeShape({1, 1}));
  out2->SetOriginShape(GeShape({1, 1}));
  (void)AttrUtils::SetInt(out1, ATTR_NAME_MEMORY_SIZE_CALC_TYPE,
                          static_cast<int32_t>(MemorySizeCalcType::ALWAYS_EMPTY));
  (void)AttrUtils::SetInt(out2, ATTR_NAME_TENSOR_MEM_TYPE, RT_MEMORY_HOST_SVM);
  (void)AttrUtils::SetListInt(op_desc, ATTR_NAME_OUTPUT_MEM_TYPE_LIST,
                              std::vector<int64_t>{RT_MEMORY_L1, RT_MEMORY_HBM, RT_MEMORY_P2P_DDR});
  runtime_param.fileconstant_addr_mapping[0x50] = 0xbeef0000U;

  std::vector<uint64_t> output_mem_type;
  const auto output_addrs = ModelUtils::GetOutputAddrs(runtime_param, op_desc, output_mem_type, true);
  std::vector<uint64_t> output_data_mem_type;
  const auto output_data_addrs = ModelUtils::GetOutputDataAddrs(runtime_param, op_desc, output_data_mem_type, true);
  std::vector<uint64_t> output_data_mem_type_value;
  const auto output_data_addrs_value =
      ModelUtils::GetOutputDataAddrsValue(runtime_param, op_desc, output_data_mem_type_value);
  std::vector<uint64_t> output_mem_type_value;
  const auto output_addrs_value = ModelUtils::GetOutputAddrsValue(runtime_param, op_desc, output_mem_type_value, true);

  EXPECT_EQ(output_addrs.size(), 3U);
  EXPECT_EQ(output_data_addrs.size(), 3U);
  EXPECT_EQ(output_addrs_value.size(), 3U);
  EXPECT_EQ(output_data_addrs_value.size(), 2U);
  EXPECT_EQ(output_data_mem_type.size(), 3U);
  EXPECT_EQ(output_mem_type[0], kConstantMemType);
  EXPECT_EQ(output_mem_type[1], kFixMemType);
  EXPECT_EQ(output_mem_type[2], RT_MEMORY_HOST_SVM);
  EXPECT_EQ(output_addrs[0], reinterpret_cast<void *>(0xbeef0000U));
  EXPECT_EQ(output_addrs[1], nullptr);
  EXPECT_NE(output_addrs[2], nullptr);
}

TEST_F(ModelUtilsUT, WorkspaceDataAddrs_CoverBranches) {
  auto fixture = BuildAddOp(1U, 1U);
  auto op_desc = fixture.op_desc;

  op_desc->SetWorkspace({0x10, 0x20, 0x30, 0x40, 0x50});
  op_desc->SetWorkspaceBytes({0x100, 0x200, 0x300, 0x400, 0x500});
  AttrUtils::SetListInt(
      op_desc, TVM_ATTR_NAME_WORKSPACE_TYPE,
      std::vector<int64_t>{RT_MEMORY_P2P_DDR, RT_MEMORY_L1, kRtMemoryUB, RT_MEMORY_HBM, RT_MEMORY_HBM});
  AttrUtils::SetListInt(
      op_desc, ATTR_NAME_WORKSPACE_TYPE_LIST,
      std::vector<int64_t>{RT_MEMORY_P2P_DDR, RT_MEMORY_L1, kRtMemoryUB, RT_MEMORY_HBM, RT_MEMORY_HBM});
  AttrUtils::SetListInt(op_desc, ATTR_NAME_WORKSPACE_MEMORY_NO_REUSE_SCOPE, std::vector<int32_t>{0, 0, 0, 1, 0});

  auto runtime_param = BuildRuntimeParamForAddressTests();
  runtime_param.is_single_op = true;
  const uint64_t session_scope_key = 0x100000000UL | RT_MEMORY_HBM;
  auto &session_scope_info = runtime_param.memory_infos[session_scope_key];
  session_scope_info.memory_type = session_scope_key;
  session_scope_info.logic_memory_base = 0x40U;
  session_scope_info.memory_size = 0x1000U;
  session_scope_info.memory_base = reinterpret_cast<uint8_t *>(0xB0000000U);

  std::vector<uint64_t> workspace_mem_type;
  const auto workspace_addrs = ModelUtils::GetWorkspaceDataAddrs(runtime_param, op_desc, workspace_mem_type);
  const auto workspace_addrs_value = ModelUtils::GetWorkspaceDataAddrsValue(runtime_param, op_desc, workspace_mem_type);
  EXPECT_EQ(workspace_addrs.size(), 5U);
  EXPECT_EQ(workspace_addrs_value.size(), 5U);
  EXPECT_EQ(workspace_mem_type[0], RT_MEMORY_P2P_DDR);
  EXPECT_EQ(workspace_mem_type[1], RT_MEMORY_L1);
  EXPECT_EQ(workspace_mem_type[2], kRtMemoryUB);
  EXPECT_EQ(workspace_mem_type[3], (0x100000000UL | RT_MEMORY_HBM));
  EXPECT_EQ(workspace_mem_type[4], RT_MEMORY_HBM);

  auto mismatch_op_desc = BuildAddOp(1U, 1U).op_desc;
  mismatch_op_desc->SetWorkspace({0x10});
  mismatch_op_desc->SetWorkspaceBytes({0x100, 0x200});
  EXPECT_TRUE(ModelUtils::GetWorkspaceSize(mismatch_op_desc).empty());
  EXPECT_TRUE(ModelUtils::GetWorkspaceDataAddrs(runtime_param, mismatch_op_desc).empty());

  auto aicpu_op_desc = BuildAddOp(1U, 1U).op_desc;
  aicpu_op_desc->SetWorkspace({0x10});
  aicpu_op_desc->SetWorkspaceBytes({0x100});
  AttrUtils::SetListBool(aicpu_op_desc, "workspace_reuse_flag", std::vector<bool>{false});
  runtime_param.is_single_op = false;
  EXPECT_TRUE(ModelUtils::GetWorkspaceDataAddrs(runtime_param, aicpu_op_desc).empty());
}

TEST_F(ModelUtilsUT, RuntimeHelpers_CoverBranches) {
  auto runtime_param = BuildRuntimeParamForAddressTests();
  auto empty_model = std::make_shared<GeModel>();
  AttrUtils::SetInt(empty_model, ATTR_MODEL_MEMORY_SIZE, 0x2000);
  AttrUtils::SetInt(empty_model, ATTR_MODEL_ZERO_COPY_MEMORY_SIZE, 0x100);
  std::vector<MemInfo> all_mem_info;
  EXPECT_EQ(ModelUtils::GetHbmFeatureMapMemInfo(empty_model, all_mem_info), SUCCESS);
  ASSERT_EQ(all_mem_info.size(), 1U);
  EXPECT_EQ(all_mem_info[0].memory_type, RT_MEMORY_HBM);
  EXPECT_EQ(all_mem_info[0].memory_size, 0x1F00);

  RuntimeParam init_param;
  init_param.fixed_mem_base = 1U;
  EXPECT_EQ(ModelUtils::InitRuntimeParams(BuildGeModel(), init_param), SUCCESS);
  EXPECT_FALSE(init_param.fixed_fm_memory_infos.empty());
  EXPECT_FALSE(init_param.fm_memory_infos.empty());

  uint8_t *mem_addr = nullptr;
  uint64_t mem_type = 0UL;
  EXPECT_EQ(ModelUtils::GetRtAddress(runtime_param, std::numeric_limits<uintptr_t>::max(), mem_addr), SUCCESS);
  EXPECT_EQ(mem_addr, nullptr);
  EXPECT_EQ(ModelUtils::GetRtAddress(runtime_param, runtime_param.logic_mem_base, mem_addr, mem_type), SUCCESS);
  EXPECT_EQ(mem_type, kFmMemType);
  EXPECT_EQ(ModelUtils::GetRtAddress(runtime_param, runtime_param.logic_weight_base, mem_addr, mem_type), SUCCESS);
  EXPECT_EQ(mem_type, kWeightMemType);
  EXPECT_EQ(ModelUtils::GetRtAddress(runtime_param, runtime_param.logic_var_base, mem_addr, mem_type), SUCCESS);
  EXPECT_EQ(mem_type, kConstantMemType);
  EXPECT_EQ(ModelUtils::GetRtAddress(runtime_param, runtime_param.host_logic_mem_base, mem_addr, mem_type), SUCCESS);
  EXPECT_EQ(mem_type, RT_MEMORY_HOST);
  EXPECT_EQ(ModelUtils::GetRtAddress(runtime_param, runtime_param.host_svm_logic_mem_base, mem_addr, mem_type),
            SUCCESS);
  EXPECT_EQ(mem_type, RT_MEMORY_HOST_SVM);
  EXPECT_EQ(ModelUtils::GetRtAddress(runtime_param, runtime_param.logic_mem_base + runtime_param.mem_size, mem_addr,
                                     mem_type),
            SUCCESS);
  EXPECT_EQ(mem_type, RT_MEMORY_P2P_DDR);
  EXPECT_EQ(ModelUtils::GetRtAddress(runtime_param, 0x8888U, mem_addr, mem_type), PARAM_INVALID);
  EXPECT_EQ(ModelUtils::GetRtAddress(runtime_param, 0U, mem_addr, mem_type), SUCCESS);
}

TEST_F(ModelUtilsUT, GetRtAddress_CoversMainBranches) {
  auto runtime_param = BuildRuntimeParamForAddressTests();
  uint8_t mem_base = 0;
  uint8_t *addr = nullptr;
  uint64_t mem_type = 0UL;
  runtime_param.mem_base = reinterpret_cast<uintptr_t>(&mem_base);

  EXPECT_EQ(ModelUtils::GetRtAddress(runtime_param, std::numeric_limits<uintptr_t>::max(), addr), SUCCESS);
  EXPECT_EQ(addr, nullptr);

  EXPECT_EQ(ModelUtils::GetRtAddress(runtime_param, runtime_param.logic_mem_base, addr, mem_type), SUCCESS);
  EXPECT_EQ(mem_type, kFmMemType);
  EXPECT_EQ(ModelUtils::GetRtAddress(runtime_param, runtime_param.logic_weight_base, addr, mem_type), SUCCESS);
  EXPECT_EQ(mem_type, kWeightMemType);
  EXPECT_EQ(ModelUtils::GetRtAddress(runtime_param, runtime_param.logic_var_base, addr, mem_type), SUCCESS);
  EXPECT_EQ(mem_type, kConstantMemType);
  EXPECT_EQ(ModelUtils::GetRtAddress(runtime_param, runtime_param.host_logic_mem_base, addr, mem_type), SUCCESS);
  EXPECT_EQ(mem_type, RT_MEMORY_HOST);
  EXPECT_EQ(ModelUtils::GetRtAddress(runtime_param, runtime_param.host_svm_logic_mem_base, addr, mem_type), SUCCESS);
  EXPECT_EQ(mem_type, RT_MEMORY_HOST_SVM);
  EXPECT_EQ(
      ModelUtils::GetRtAddress(runtime_param, runtime_param.logic_mem_base + runtime_param.mem_size, addr, mem_type),
      SUCCESS);
  EXPECT_EQ(mem_type, RT_MEMORY_P2P_DDR);
  EXPECT_EQ(ModelUtils::GetRtAddress(runtime_param, 0x8888U, addr, mem_type), PARAM_INVALID);
  EXPECT_EQ(ModelUtils::GetRtAddress(runtime_param, 0U, addr, mem_type), SUCCESS);
}

TEST_F(ModelUtilsUT, MemoryInfoHelpers_Work) {
  auto ge_model = BuildGeModel();

  std::vector<MemInfo> all_mem_info;
  EXPECT_EQ(ModelUtils::GetHbmFeatureMapMemInfo(ge_model, all_mem_info), SUCCESS);
  EXPECT_EQ(all_mem_info.size(), 2U);

  std::vector<MemInfo> all_mem_info_with_zero_copy;
  EXPECT_EQ(ModelUtils::GetHbmFeatureMapMemInfo(ge_model, all_mem_info_with_zero_copy, true), SUCCESS);
  EXPECT_EQ(all_mem_info_with_zero_copy.size(), 3U);

  const auto all_memory_type_size = ModelUtils::GetAllMemoryTypeSize(ge_model);
  EXPECT_GE(all_memory_type_size.size(), 6U);

  RuntimeParam runtime_param;
  EXPECT_EQ(ModelUtils::InitRuntimeParams(ge_model, runtime_param), SUCCESS);
  EXPECT_EQ(runtime_param.session_id, 1U);
  EXPECT_EQ(runtime_param.stream_num, 2U);
  EXPECT_EQ(runtime_param.notify_num, 3U);
  EXPECT_EQ(runtime_param.memory_infos.count(RT_MEMORY_P2P_DDR), 1U);
  EXPECT_EQ(runtime_param.memory_infos.count(RT_MEMORY_HOST), 1U);
  EXPECT_EQ(runtime_param.memory_infos.count(RT_MEMORY_HOST_SVM), 1U);
}

}  // namespace om2
}  // namespace ge
