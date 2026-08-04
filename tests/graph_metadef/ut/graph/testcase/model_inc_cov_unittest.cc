/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <stdio.h>
#include <gtest/gtest.h>
#include <iostream>
#include <fstream>
#include "test_structs.h"
#include "func_counter.h"
#include "graph/buffer.h"
#include "graph/attr_store.h"
#include "graph/model.h"
#include "graph/node.h"
#include "graph_builder_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/graph_utils_ex.h"
#include "graph/utils/node_utils.h"
#include "graph/debug/ge_attr_define.h"
#include "graph_metadef/graph/utils/file_utils.h"
#include "mmpa/mmpa_api.h"
#include "ge_ir.pb.h"
#include "graph/graph_buffer.h"

extern "C" ge::graphStatus GeApiWrapper_ModelSaveToString(const ge::Graph &graph, const std::string &node_name,
                                                          std::string &model_str);

namespace ge {
namespace {
class SubModel2 : public Model {
 public:
  SubModel2() {}
  SubModel2(const std::string &name, const std::string &custom_version) : Model(name, custom_version) {}
  virtual ~SubModel2() = default;
};

static ge::Graph BuildTestGraph() {
  ge::Graph graph("test_graph");
  auto compute_graph = std::make_shared<ComputeGraph>("test_compute_graph");
  auto op_desc = std::make_shared<OpDesc>("test_node", "TestOp");
  op_desc->AddInputDesc(GeTensorDesc(GeShape({1, 1, 1, 1}), FORMAT_NCHW, DT_FLOAT));
  op_desc->AddOutputDesc(GeTensorDesc(GeShape({1, 1, 1, 1}), FORMAT_NCHW, DT_FLOAT));
  compute_graph->AddNode(op_desc);
  graph = ge::GraphUtilsEx::CreateGraphFromComputeGraph(compute_graph);
  return graph;
}

}  // namespace

class ModelIncCovUt : public testing::Test {
 protected:
  void SetUp() override {}

  void TearDown() override {
    system("rm -rf ./tmp_model_inc_cov");
  }
};

TEST_F(ModelIncCovUt, IncCov2_ModelCharPtrConstructor_NullName) {
  Model model(static_cast<const char *>(nullptr), "v1");
  EXPECT_EQ(model.GetName(), "");
}

TEST_F(ModelIncCovUt, IncCov2_ModelCharPtrConstructor_NullVersion) {
  Model model("test_model", static_cast<const char *>(nullptr));
  EXPECT_EQ(model.GetPlatformVersion(), "");
}

TEST_F(ModelIncCovUt, IncCov2_ModelCharPtrConstructor_BothNull) {
  Model model(static_cast<const char *>(nullptr), static_cast<const char *>(nullptr));
  EXPECT_EQ(model.GetName(), "");
  EXPECT_EQ(model.GetPlatformVersion(), "");
}

TEST_F(ModelIncCovUt, IncCov2_ModelCharPtrConstructor_ValidParams) {
  Model model("test_name", "test_version");
  EXPECT_EQ(model.GetName(), "test_name");
  EXPECT_EQ(model.GetPlatformVersion(), "test_version");
}

TEST_F(ModelIncCovUt, IncCov2_SetName_GetName) {
  Model model("old_name", "v1");
  model.SetName("new_name");
  EXPECT_EQ(model.GetName(), "new_name");
}

TEST_F(ModelIncCovUt, IncCov2_GetVersion_Default) {
  Model model("test", "v1");
  EXPECT_EQ(model.GetVersion(), 0U);
}

TEST_F(ModelIncCovUt, IncCov2_GetPlatformVersion) {
  Model model("test", "custom_v2");
  EXPECT_EQ(model.GetPlatformVersion(), "custom_v2");
}

TEST_F(ModelIncCovUt, IncCov2_SetGraph_GetGraph) {
  Model model("test", "v1");
  auto graph = std::make_shared<ComputeGraph>("test_graph");
  model.SetGraph(graph);
  EXPECT_EQ(model.GetGraph(), graph);
}

TEST_F(ModelIncCovUt, IncCov2_IsValid_NullGraph) {
  Model model("test", "v1");
  EXPECT_FALSE(model.IsValid());
}

TEST_F(ModelIncCovUt, IncCov2_IsValid_ValidGraph) {
  auto md = SubModel2("test", "v1");
  auto graph = BuildTestGraph();
  md.SetGraph(GraphUtilsEx::GetComputeGraph(graph));
  EXPECT_TRUE(md.IsValid());
}

TEST_F(ModelIncCovUt, IncCov2_Save_Success) {
  auto md = SubModel2("test_save", "v1");
  auto graph = BuildTestGraph();
  md.SetGraph(GraphUtilsEx::GetComputeGraph(graph));
  Buffer buffer;
  EXPECT_EQ(md.Save(buffer, false), GRAPH_SUCCESS);
  EXPECT_GT(buffer.GetSize(), 0U);
}

TEST_F(ModelIncCovUt, IncCov2_Save_Failure) {
  Model model("test_fail", "v1");
  Buffer buffer;
  EXPECT_EQ(model.Save(buffer, false), GRAPH_FAILED);
}

TEST_F(ModelIncCovUt, IncCov2_SaveWithoutSeparate_Success) {
  auto md = SubModel2("test_save_wo_sep", "v1");
  auto graph = BuildTestGraph();
  md.SetGraph(GraphUtilsEx::GetComputeGraph(graph));
  Buffer buffer;
  EXPECT_EQ(md.SaveWithoutSeparate(buffer, false), GRAPH_SUCCESS);
  EXPECT_GT(buffer.GetSize(), 0U);
}

TEST_F(ModelIncCovUt, IncCov2_SaveWithPath_Success) {
  auto md = SubModel2("test_save_path", "v1");
  auto graph = BuildTestGraph();
  md.SetGraph(GraphUtilsEx::GetComputeGraph(graph));
  Buffer buffer;
  EXPECT_EQ(md.Save(buffer, "./tmp_model_inc_cov/test_save.air", false), GRAPH_SUCCESS);
  EXPECT_GT(buffer.GetSize(), 0U);
}

TEST_F(ModelIncCovUt, IncCov2_SaveSeparateModel_Success) {
  auto md = SubModel2("test_save_sep", "v1");
  auto graph = BuildTestGraph();
  md.SetGraph(GraphUtilsEx::GetComputeGraph(graph));
  Buffer buffer;
  EXPECT_EQ(md.SaveSeparateModel(buffer, "./tmp_model_inc_cov/test_sep.air", false), GRAPH_SUCCESS);
}

TEST_F(ModelIncCovUt, IncCov2_SaveModelDef_Success) {
  auto md = SubModel2("test_save_def", "v1");
  auto graph = BuildTestGraph();
  md.SetGraph(GraphUtilsEx::GetComputeGraph(graph));
  proto::ModelDef model_def;
  EXPECT_EQ(md.Save(model_def, false), GRAPH_SUCCESS);
}

TEST_F(ModelIncCovUt, IncCov2_SetAttr_GetAttrMap) {
  Model model("test_attr", "v1");
  ProtoAttrMap attrs;
  model.SetAttr(attrs);
  EXPECT_TRUE(model.GetAttrMap().GetAllAttrNames().empty());
}

TEST_F(ModelIncCovUt, IncCov2_LoadFromBuffer_Success) {
  auto md = SubModel2("test_load_buf", "v1");
  auto graph = BuildTestGraph();
  md.SetGraph(GraphUtilsEx::GetComputeGraph(graph));
  Buffer buffer;
  md.Save(buffer, false);
  ASSERT_GT(buffer.GetSize(), 0U);
  Model loaded_model;
  EXPECT_EQ(Model::Load(buffer.GetData(), buffer.GetSize(), loaded_model), GRAPH_SUCCESS);
}

TEST_F(ModelIncCovUt, IncCov2_LoadFromBuffer_InvalidData) {
  uint8_t invalid_data[] = {0x00, 0x01, 0x02, 0x03};
  Model loaded_model;
  EXPECT_EQ(Model::Load(invalid_data, sizeof(invalid_data), loaded_model), GRAPH_FAILED);
}

TEST_F(ModelIncCovUt, IncCov2_LoadWithMultiThread_Success) {
  auto md = SubModel2("test_load_mt", "v1");
  auto graph = BuildTestGraph();
  md.SetGraph(GraphUtilsEx::GetComputeGraph(graph));
  Buffer buffer;
  md.Save(buffer, false);
  ASSERT_GT(buffer.GetSize(), 0U);
  Model loaded_model;
  EXPECT_EQ(Model::LoadWithMultiThread(buffer.GetData(), buffer.GetSize(), loaded_model), GRAPH_SUCCESS);
}

TEST_F(ModelIncCovUt, IncCov2_LoadWithMultiThread_InvalidData) {
  uint8_t invalid_data[] = {0xFF, 0xFE, 0xFD};
  Model loaded_model;
  EXPECT_EQ(Model::LoadWithMultiThread(invalid_data, sizeof(invalid_data), loaded_model), GRAPH_FAILED);
}

TEST_F(ModelIncCovUt, IncCov2_LoadFromModelDef_Success) {
  auto md = SubModel2("test_load_def", "v1");
  auto graph = BuildTestGraph();
  md.SetGraph(GraphUtilsEx::GetComputeGraph(graph));
  proto::ModelDef model_def;
  md.Save(model_def, false);
  Model loaded_model;
  EXPECT_EQ(loaded_model.Load(model_def), GRAPH_SUCCESS);
}

TEST_F(ModelIncCovUt, IncCov2_SaveToFile_NoDirForceSeparate) {
  system("mkdir -p ./tmp_model_inc_cov");
  auto md = SubModel2("test_save_file_sep", "v1");
  auto graph = BuildTestGraph();
  md.SetGraph(GraphUtilsEx::GetComputeGraph(graph));
  EXPECT_EQ(md.SaveToFile("./tmp_model_inc_cov/test_sep_file.air", true), GRAPH_SUCCESS);
  system("rm -f ./tmp_model_inc_cov/test_sep_file.air");
}

TEST_F(ModelIncCovUt, IncCov2_SaveToFile_WithDir) {
  system("mkdir -p ./tmp_model_inc_cov/subdir");
  auto md = SubModel2("test_save_file_dir", "v1");
  auto graph = BuildTestGraph();
  md.SetGraph(GraphUtilsEx::GetComputeGraph(graph));
  EXPECT_EQ(md.SaveToFile("./tmp_model_inc_cov/subdir/test_file.air", false), GRAPH_SUCCESS);
  system("rm -f ./tmp_model_inc_cov/subdir/test_file.air");
}

TEST_F(ModelIncCovUt, IncCov2_SaveToFile_EmptyFileName) {
  auto md = SubModel2("test_save_file_empty", "v1");
  auto graph = BuildTestGraph();
  md.SetGraph(GraphUtilsEx::GetComputeGraph(graph));
  EXPECT_NE(md.SaveToFile("", false), GRAPH_SUCCESS);
}

TEST_F(ModelIncCovUt, IncCov2_LoadFromFile_TooLongName) {
  std::string long_name(MMPA_MAX_PATH + 10, 'a');
  Model model;
  EXPECT_EQ(model.LoadFromFile(long_name), GRAPH_FAILED);
}

TEST_F(ModelIncCovUt, IncCov2_LoadFromFile_NonExistFile) {
  Model model;
  EXPECT_EQ(model.LoadFromFile("./tmp_model_inc_cov/nonexist_file.air"), GRAPH_FAILED);
}

TEST_F(ModelIncCovUt, IncCov2_LoadFromFile_CorruptedFile) {
  system("mkdir -p ./tmp_model_inc_cov");
  std::string file_path = "./tmp_model_inc_cov/corrupted.air";
  std::ofstream ofs(file_path, std::ios::binary);
  if (ofs.is_open()) {
    ofs.write("corrupted_data_not_protobuf", 26);
    ofs.close();
  }
  Model model;
  EXPECT_EQ(model.LoadFromFile(file_path), GRAPH_FAILED);
  system(("rm -f " + file_path).c_str());
}

TEST_F(ModelIncCovUt, IncCov2_LoadFromFile_ValidFile) {
  system("mkdir -p ./tmp_model_inc_cov");
  std::string file_path = "./tmp_model_inc_cov/valid_model.air";
  auto md = SubModel2("test_load_file_valid", "v1");
  auto graph = BuildTestGraph();
  md.SetGraph(GraphUtilsEx::GetComputeGraph(graph));
  EXPECT_EQ(md.SaveToFile(file_path, false), GRAPH_SUCCESS);
  Model loaded_model;
  EXPECT_EQ(loaded_model.LoadFromFile(file_path), GRAPH_SUCCESS);
  system(("rm -f " + file_path).c_str());
}

TEST_F(ModelIncCovUt, IncCov2_LoadFromFile_WithModelDef) {
  system("mkdir -p ./tmp_model_inc_cov");
  std::string file_path = "./tmp_model_inc_cov/valid_model_def.air";
  auto md = SubModel2("test_load_def_file", "v1");
  auto graph = BuildTestGraph();
  md.SetGraph(GraphUtilsEx::GetComputeGraph(graph));
  EXPECT_EQ(md.SaveToFile(file_path, false), GRAPH_SUCCESS);
  proto::ModelDef model_def;
  Model model;
  EXPECT_EQ(model.Load(model_def, file_path), GRAPH_FAILED);
  system(("rm -f " + file_path).c_str());
}

TEST_F(ModelIncCovUt, IncCov2_ModelSaveToString_Success) {
  auto graph = BuildTestGraph();
  std::string model_str;
  EXPECT_EQ(GeApiWrapper_ModelSaveToString(graph, "test_node", model_str), ge::SUCCESS);
  EXPECT_FALSE(model_str.empty());
}

TEST_F(ModelIncCovUt, IncCov2_ModelSaveToString_InvalidGraph) {
  ge::Graph empty_graph;
  std::string model_str;
  EXPECT_NE(GeApiWrapper_ModelSaveToString(empty_graph, "test_node", model_str), ge::SUCCESS);
}

TEST_F(ModelIncCovUt, IncCov2_SaveToFile_NoDirNoWorkPath) {
  auto md = SubModel2("test_save_no_work", "v1");
  auto graph = BuildTestGraph();
  md.SetGraph(GraphUtilsEx::GetComputeGraph(graph));
  unsetenv("ASCEND_WORK_PATH");
  EXPECT_EQ(md.SaveToFile("just_a_file_inc.air"), GRAPH_SUCCESS);
  system("rm -f just_a_file_inc.air");
}

TEST_F(ModelIncCovUt, IncCov2_SaveToFile_NoGraph) {
  Model model("test_no_graph", "v1");
  system("mkdir -p ./tmp_model_inc_cov");
  EXPECT_NE(model.SaveToFile("./tmp_model_inc_cov/no_graph.air", false), GRAPH_SUCCESS);
  system("rm -f ./tmp_model_inc_cov/no_graph.air");
}

TEST_F(ModelIncCovUt, IncCov2_SaveToFile_NoGraphForceSeparate) {
  Model model("test_no_graph_sep", "v1");
  system("mkdir -p ./tmp_model_inc_cov");
  EXPECT_NE(model.SaveToFile("./tmp_model_inc_cov/no_graph_sep.air", true), GRAPH_SUCCESS);
  system("rm -f ./tmp_model_inc_cov/no_graph_sep.air");
}

TEST_F(ModelIncCovUt, IncCov2_LoadFromFile_EmptyFile) {
  system("mkdir -p ./tmp_model_inc_cov");
  std::string file_path = "./tmp_model_inc_cov/empty.air";
  std::ofstream ofs(file_path, std::ios::binary);
  if (ofs.is_open()) {
    ofs.close();
  }
  Model model;
  EXPECT_EQ(model.LoadFromFile(file_path), GRAPH_FAILED);
  system(("rm -f " + file_path).c_str());
}

TEST_F(ModelIncCovUt, IncCov2_LoadFromFile_PermissionDenied) {
  system("mkdir -p ./tmp_model_inc_cov");
  std::string file_path = "./tmp_model_inc_cov/noperm.air";
  std::ofstream ofs(file_path, std::ios::binary);
  if (ofs.is_open()) {
    ofs.write("data", 4);
    ofs.close();
  }
  system(("chmod 000 " + file_path).c_str());
  Model model;
  EXPECT_EQ(model.LoadFromFile(file_path), GRAPH_FAILED);
  system(("chmod 644 " + file_path).c_str());
  system(("rm -f " + file_path).c_str());
}

TEST_F(ModelIncCovUt, IncCov2_LoadFromFile_InvalidProtobufContent) {
  system("mkdir -p ./tmp_model_inc_cov");
  std::string file_path = "./tmp_model_inc_cov/invalid_proto.air";
  std::ofstream ofs(file_path, std::ios::binary);
  if (ofs.is_open()) {
    std::string invalid_data(128, '\xFF');
    ofs.write(invalid_data.data(), static_cast<std::streamsize>(invalid_data.size()));
    ofs.close();
  }
  Model model;
  EXPECT_EQ(model.LoadFromFile(file_path), GRAPH_FAILED);
  system(("rm -f " + file_path).c_str());
}
}  // namespace ge
