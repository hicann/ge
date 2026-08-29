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

#include <vector>

#include "common/share_graph.h"
#define private public
#include "framework/common/helper/model_helper.h"
#undef private
#include "framework/common/framework_types_internal.h"
#include "graph/utils/graph_utils_ex.h"
#include "stub/gert_runtime_stub.h"

namespace ge {
namespace {

class UtestModelHelperMisc : public testing::Test {
 protected:
  void SetUp() override {}
  void TearDown() override {}
};

TEST_F(UtestModelHelperMisc, CheckOsCpuInfoAndOppVersion_Success) {
  ModelHelper model_helper;
  std::vector<char> data(256);
  ModelFileHeader *file_header = reinterpret_cast<ModelFileHeader *>(data.data());
  file_header->need_check_os_cpu_info = static_cast<uint8_t>(OsCpuInfoCheckTyep::NO_CHECK);
  model_helper.file_header_ = file_header;
  model_helper.is_unknown_shape_model_ = true;
  gert::GertRuntimeStub stub;
  stub.GetSlogStub().Clear();
  stub.GetSlogStub().SetLevelDebug();
  EXPECT_EQ(model_helper.CheckOsCpuInfoAndOppVersion(), SUCCESS);
  ASSERT_TRUE(stub.GetSlogStub().FindLog(DLOG_DEBUG, "Check opp version[] success") >= 0);
}

TEST_F(UtestModelHelperMisc, UpdateSessionGraphId) {
  ModelHelper model_helper;
  bool refreshed = false;
  auto graph = gert::ShareGraph::BuildWithKnownSubgraphWithTwoConst();
  auto ret = model_helper.UpdateSessionGraphId(graph, "1", refreshed);
  EXPECT_EQ(ret, SUCCESS);
}

TEST_F(UtestModelHelperMisc, LoadModel_AlreadyLoaded_ReturnsRepeated) {
  ModelHelper model_helper;
  model_helper.is_assign_model_ = true;
  ModelData model_data;
  model_data.model_data = nullptr;
  model_data.model_len = 0U;
  EXPECT_EQ(model_helper.LoadModel(model_data), ACL_ERROR_GE_EXEC_MODEL_DATA_SIZE_INVALID);
}

TEST_F(UtestModelHelperMisc, LoadRootModel_AlreadyLoaded_ReturnsRepeated) {
  ModelHelper model_helper;
  model_helper.is_assign_model_ = true;
  ModelData model_data;
  model_data.model_data = nullptr;
  model_data.model_len = 0U;
  EXPECT_EQ(model_helper.LoadRootModel(model_data), ACL_ERROR_GE_EXEC_LOAD_MODEL_REPEATED);
}

TEST_F(UtestModelHelperMisc, UpdatePlatfromInfoWithRuntime_OfflineScene_ReturnsSuccess) {
  ModelHelper model_helper;
  fe::PlatformInfo platform_info;
  int32_t virtual_type = 0;
  EXPECT_EQ(model_helper.UpdatePlatfromInfoWithRuntime(-1, 1, 1, platform_info, virtual_type), SUCCESS);
}

TEST_F(UtestModelHelperMisc, SetSaveMode_Test) {
  ModelHelper model_helper;
  model_helper.SetSaveMode(true);
  model_helper.SetSaveMode(false);
}

TEST_F(UtestModelHelperMisc, GetBaseNameFromFileName_WithExtension) {
  ModelHelper model_helper;
  std::string base_name;
  EXPECT_EQ(model_helper.GetBaseNameFromFileName("model.om", base_name), SUCCESS);
  EXPECT_EQ(base_name, "model");
}

TEST_F(UtestModelHelperMisc, GetBaseNameFromFileName_WithDoubleExtension) {
  ModelHelper model_helper;
  std::string base_name;
  EXPECT_EQ(model_helper.GetBaseNameFromFileName("model.exe.om", base_name), SUCCESS);
  EXPECT_EQ(base_name, "model.exe");
}

TEST_F(UtestModelHelperMisc, GetBaseNameFromFileName_NoExtension) {
  ModelHelper model_helper;
  std::string base_name;
  EXPECT_EQ(model_helper.GetBaseNameFromFileName("model", base_name), SUCCESS);
}

TEST_F(UtestModelHelperMisc, GetBaseNameFromFileName_WithPath) {
  ModelHelper model_helper;
  std::string base_name;
  EXPECT_EQ(model_helper.GetBaseNameFromFileName("/path/to/model.om", base_name), SUCCESS);
}

TEST_F(UtestModelHelperMisc, LoadModel_NullModelData) {
  ModelHelper model_helper;
  ModelData model_data;
  model_data.model_data = nullptr;
  model_data.model_len = 0U;
  EXPECT_NE(model_helper.LoadModel(model_data), SUCCESS);
}

TEST_F(UtestModelHelperMisc, LoadRootModel_NullModelData) {
  ModelHelper model_helper;
  ModelData model_data;
  model_data.model_data = nullptr;
  model_data.model_len = 0U;
  EXPECT_NE(model_helper.LoadRootModel(model_data), SUCCESS);
}

TEST_F(UtestModelHelperMisc, CheckOsCpuInfoAndOppVersion_NeedCheck) {
#if defined(__aarch64__) || defined(__arm64__)
  GTEST_SKIP() << "Model helper dump dependency is unavailable on native arm64";
#endif
  ModelHelper model_helper;
  std::vector<char> data(256);
  ModelFileHeader *file_header = reinterpret_cast<ModelFileHeader *>(data.data());
  file_header->need_check_os_cpu_info = static_cast<uint8_t>(OsCpuInfoCheckTyep::NEED_CHECK);
  model_helper.file_header_ = file_header;
  model_helper.is_unknown_shape_model_ = true;
  gert::GertRuntimeStub stub;
  stub.GetSlogStub().Clear();
  EXPECT_NE(model_helper.CheckOsCpuInfoAndOppVersion(), SUCCESS);
}

TEST_F(UtestModelHelperMisc, SaveOriginalGraphToOmModel_EmptyOutputFile) {
  ModelHelper model_helper;
  auto graph = std::make_shared<ComputeGraph>("test_graph");
  auto data = graph->AddNode(std::make_shared<OpDesc>("data1", DATA));
  EXPECT_NE(model_helper.SaveOriginalGraphToOmModel(GraphUtilsEx::CreateGraphFromComputeGraph(graph), ""), SUCCESS);
}

TEST_F(UtestModelHelperMisc, SetSaveMode_True) {
  ModelHelper model_helper;
  model_helper.SetSaveMode(true);
  EXPECT_TRUE(model_helper.is_offline_);
}

TEST_F(UtestModelHelperMisc, SetSaveMode_False) {
  ModelHelper model_helper;
  model_helper.SetSaveMode(false);
  EXPECT_FALSE(model_helper.is_offline_);
}

TEST_F(UtestModelHelperMisc, SaveModelWeights_NoWeight) {
  ModelHelper model_helper;
  auto ge_model = std::make_shared<GeModel>();
  auto om_helper = std::make_shared<OmFileSaveHelper>();
  EXPECT_EQ(model_helper.SaveModelWeights(om_helper, ge_model, 0U), SUCCESS);
}

TEST_F(UtestModelHelperMisc, SaveModelTbeKernel_NoKernel) {
  ModelHelper model_helper;
  auto ge_model = std::make_shared<GeModel>();
  auto om_helper = std::make_shared<OmFileSaveHelper>();
  EXPECT_EQ(model_helper.SaveModelTbeKernel(om_helper, ge_model, 0U), SUCCESS);
}

TEST_F(UtestModelHelperMisc, SaveModelCustAICPU_NoKernel) {
  ModelHelper model_helper;
  auto ge_model = std::make_shared<GeModel>();
  auto om_helper = std::make_shared<OmFileSaveHelper>();
  EXPECT_EQ(model_helper.SaveModelCustAICPU(om_helper, ge_model, 0U), SUCCESS);
}

TEST_F(UtestModelHelperMisc, SaveModelDef_BasicModel) {
  ModelHelper model_helper;
  auto ge_model = std::make_shared<GeModel>();
  auto compute_graph = std::make_shared<ComputeGraph>("test_def_graph");
  auto data = compute_graph->AddNode(std::make_shared<OpDesc>("data1", DATA));
  ge_model->SetGraph(compute_graph);
  ge_model->SetName("test_model_def");
  auto om_helper = std::make_shared<OmFileSaveHelper>();
  ge::Buffer model_buffer;
  EXPECT_EQ(model_helper.SaveModelDef(om_helper, ge_model, model_buffer, 0U), SUCCESS);
}

TEST_F(UtestModelHelperMisc, SaveAllModelPartiton_BasicModel) {
  ModelHelper model_helper;
  auto ge_model = std::make_shared<GeModel>();
  auto compute_graph = std::make_shared<ComputeGraph>("test_all_part_graph");
  auto data = compute_graph->AddNode(std::make_shared<OpDesc>("data1", DATA));
  ge_model->SetGraph(compute_graph);
  ge_model->SetName("test_all_part");
  auto task_def = std::make_shared<domi::ModelTaskDef>();
  task_def->add_task();
  ge_model->SetModelTaskDef(task_def);
  auto om_helper = std::make_shared<OmFileSaveHelper>();
  ge::Buffer model_buffer;
  ge::Buffer task_buffer;
  EXPECT_EQ(model_helper.SaveAllModelPartiton(om_helper, ge_model, model_buffer, task_buffer, 0U), SUCCESS);
}

TEST_F(UtestModelHelperMisc, SetModelAttributes_Basic) {
  ModelHelper model_helper;
  auto ge_model = std::make_shared<GeModel>();
  EXPECT_EQ(model_helper.SetModelAttributes(ge_model), SUCCESS);
}

TEST_F(UtestModelHelperMisc, SaveToOmModel_EmptyOutputFile) {
  ModelHelper model_helper;
  auto ge_model = std::make_shared<GeModel>();
  ge_model->SetName("test_empty_out");
  ModelBufferData model;
  EXPECT_EQ(model_helper.SaveToOmModel(ge_model, "", model, nullptr), FAILED);
}

TEST_F(UtestModelHelperMisc, SaveBundleModelBufferToMem_EmptyBuffers) {
  ModelHelper model_helper;
  std::vector<ModelBufferData> model_buffers;
  ModelBufferData output_buffer;
  EXPECT_EQ(model_helper.SaveBundleModelBufferToMem(model_buffers, 0UL, output_buffer), SUCCESS);
}

TEST_F(UtestModelHelperMisc, UpdatePlatfromInfoWithRuntime_OnlineScene) {
  ModelHelper model_helper;
  fe::PlatformInfo platform_info;
  int32_t virtual_type = 0;
  EXPECT_EQ(model_helper.UpdatePlatfromInfoWithRuntime(0, 0, 0, platform_info, virtual_type), SUCCESS);
}
}  // namespace
}  // namespace ge
