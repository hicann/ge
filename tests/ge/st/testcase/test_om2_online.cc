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

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <future>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "common/env_path.h"
#include "common/om2/om2_model_data.h"
#include "common/helper/om2/om2_utils.h"
#include "common/memory/tensor_trans_utils.h"
#include "common/path_utils.h"
#include "framework/common/helper/om2_package_helper.h"
#include "graph/execute/model_executor.h"
#include "graph/ge_local_context.h"
#include "graph/manager/graph_manager_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/graph.h"
#include "graph/op_desc.h"
#include "graph/utils/file_utils.h"
#include "init_ge.h"
#include "mmpa/mmpa_api.h"
#include "common/model/ge_root_model.h"

#include "ge_runtime_stub/include/common/share_graph.h"
#include "ge_runtime_stub/include/faker/aicore_taskdef_faker.h"
#include "ge_runtime_stub/include/faker/ge_model_builder.h"

namespace ge {
namespace {

class ScopedEnvVar {
 public:
  ScopedEnvVar(const char *name, const char *value) : name_(name) {
    const char *old_value = getenv(name);
    if (old_value != nullptr) {
      old_value_ = old_value;
      has_old_value_ = true;
    }
    (void)setenv(name, value, 1);
  }

  ~ScopedEnvVar() {
    if (has_old_value_) {
      (void)setenv(name_.c_str(), old_value_.c_str(), 1);
      return;
    }
    (void)unsetenv(name_.c_str());
  }

 private:
  std::string name_;
  std::string old_value_;
  bool has_old_value_ = false;
};

GeRootModelPtr CreateSimpleGeRootModel() {
  auto compute_graph = std::make_shared<ComputeGraph>("test_graph");
  auto data_op = std::make_shared<OpDesc>("data1", "Data");
  data_op->AddInputDesc(GeTensorDesc(GeShape({2, 16}), FORMAT_ND, DT_FLOAT));
  data_op->AddOutputDesc(GeTensorDesc(GeShape({2, 16}), FORMAT_ND, DT_FLOAT));
  auto data_node = compute_graph->AddNode(data_op);

  auto output_op = std::make_shared<OpDesc>("output1", "NetOutput");
  output_op->AddInputDesc(GeTensorDesc(GeShape({2, 16}), FORMAT_ND, DT_FLOAT));
  auto output_node = compute_graph->AddNode(output_op);

  (void)ge::GraphUtils::AddEdge(data_node->GetOutDataAnchor(0), output_node->GetInDataAnchor(0));

  auto ge_root_model = std::make_shared<GeRootModel>();
  (void)ge_root_model->Initialize(compute_graph);
  auto ge_model = std::make_shared<GeModel>();
  ge_model->SetGraph(compute_graph);
  ge_root_model->SetSubgraphInstanceNameToModel(compute_graph->GetName(), ge_model);
  return ge_root_model;
}
}  // namespace

class Om2OnlineSessionTest : public testing::Test {
 protected:
  void SetUp() override {
    test_work_dir_ = EnvPath().GetOrCreateCaseTmpPath("Om2OnlineSessionTest");
    setenv("ASCEND_WORK_PATH", test_work_dir_.c_str(), 1);
    const auto ascend_install_path = EnvPath().GetAscendInstallPath();
    setenv("ASCEND_HOME_PATH", ascend_install_path.c_str(), 1);
  }

  void TearDown() override {
    unsetenv("ASCEND_WORK_PATH");
    unsetenv("ASCEND_HOME_PATH");
    unsetenv("ENABLE_RUNTIME_OM2");
    std::error_code ec;
    (void)std::filesystem::remove_all(test_work_dir_, ec);
  }

  std::string test_work_dir_;
};

// Test: OM2 online mode detection via env var
TEST_F(Om2OnlineSessionTest, Om2ModeDetection) {
  unsetenv("ENABLE_RUNTIME_OM2");
  EXPECT_FALSE(IsOm2OnlineMode());

  setenv("ENABLE_RUNTIME_OM2", "1", 1);
  EXPECT_TRUE(IsOm2OnlineMode());

  unsetenv("ENABLE_RUNTIME_OM2");
  EXPECT_FALSE(IsOm2OnlineMode());
}

// Test: GeRootModel OM2 data lifecycle
TEST_F(Om2OnlineSessionTest, GeRootModel_Om2DataLifecycle) {
  auto ge_root_model = CreateSimpleGeRootModel();
  ASSERT_NE(ge_root_model, nullptr);

  // Initially no OM2 data
  EXPECT_FALSE(ge_root_model->HasOm2ModelData());

  // Set OM2 data
  auto om2_data = std::make_shared<gert::Om2ModelData>();
  om2_data->model_meta.model_name = "test_model";
  om2_data->model_meta.work_size = 2048U;
  ge_root_model->SetOm2ModelData(om2_data);

  EXPECT_TRUE(ge_root_model->HasOm2ModelData());
  EXPECT_EQ(ge_root_model->GetOm2ModelData().model_meta.model_name, "test_model");
  EXPECT_EQ(ge_root_model->GetOm2ModelData().model_meta.work_size, 2048U);

  // Clear OM2 data
  ge_root_model->SetOm2ModelData(nullptr);
  EXPECT_FALSE(ge_root_model->HasOm2ModelData());
}

// Test: Om2ModelData structure integrity
TEST_F(Om2OnlineSessionTest, Om2ModelData_StructureIntegrity) {
  gert::Om2ModelData model_data;

  model_data.program_body.so_artifact.file_name = "libtest.so";
  model_data.program_body.so_artifact.data = {0x7f, 0x45, 0x4c, 0x46};

  // Populate model meta
  model_data.model_meta.model_name = "test_model";
  model_data.model_meta.root_graph_name = "root_graph";
  model_data.model_meta.work_size = 4096U;

  // Populate kernel binaries
  gert::Om2KernelBinary kernel;
  kernel.name = "test_kernel";
  kernel.data = {0x01, 0x02, 0x03};
  model_data.kernel_binaries.push_back(kernel);

  model_data.constants_data.weight_data = {0xAA, 0xBB, 0xCC, 0xDD};

  EXPECT_EQ(model_data.program_body.so_artifact.file_name, "libtest.so");
  EXPECT_EQ(model_data.program_body.so_artifact.data.size(), 4U);
  EXPECT_EQ(model_data.model_meta.work_size, 4096U);
  EXPECT_EQ(model_data.kernel_binaries.size(), 1U);
  EXPECT_EQ(model_data.kernel_binaries[0].name, "test_kernel");
  EXPECT_EQ(model_data.constants_data.weight_data.size(), 4U);
}

// Test: Multiple GeRootModel instances with OM2 data
TEST_F(Om2OnlineSessionTest, MultipleGeRootModels_IndependentOm2Data) {
  auto model1 = CreateSimpleGeRootModel();
  auto model2 = CreateSimpleGeRootModel();
  ASSERT_NE(model1, nullptr);
  ASSERT_NE(model2, nullptr);

  auto data1 = std::make_shared<gert::Om2ModelData>();
  data1->model_meta.model_name = "model_1";
  model1->SetOm2ModelData(data1);

  auto data2 = std::make_shared<gert::Om2ModelData>();
  data2->model_meta.model_name = "model_2";
  model2->SetOm2ModelData(data2);

  // Each model has its own independent OM2 data
  EXPECT_TRUE(model1->HasOm2ModelData());
  EXPECT_TRUE(model2->HasOm2ModelData());
  EXPECT_EQ(model1->GetOm2ModelData().model_meta.model_name, "model_1");
  EXPECT_EQ(model2->GetOm2ModelData().model_meta.model_name, "model_2");

  // Clearing one doesn't affect the other
  model1->SetOm2ModelData(nullptr);
  EXPECT_FALSE(model1->HasOm2ModelData());
  EXPECT_TRUE(model2->HasOm2ModelData());
}

// Test: Om2ModelData shared_ptr semantics (Fork scenario)
TEST_F(Om2OnlineSessionTest, Om2ModelData_SharedPtrFork) {
  auto ge_root_model = CreateSimpleGeRootModel();
  ASSERT_NE(ge_root_model, nullptr);

  auto om2_data = std::make_shared<gert::Om2ModelData>();
  om2_data->model_meta.model_name = "shared_model";
  om2_data->constants_data.weight_data = {0x01, 0x02, 0x03};
  ge_root_model->SetOm2ModelData(om2_data);

  // Simulate fork: shared_ptr copy
  auto forked_data = om2_data;  // shared_ptr copy, same underlying data

  EXPECT_TRUE(ge_root_model->HasOm2ModelData());
  EXPECT_EQ(forked_data->model_meta.model_name, "shared_model");
  EXPECT_EQ(forked_data->constants_data.weight_data.size(), 3U);

  // Both point to the same data
  EXPECT_EQ(&ge_root_model->GetOm2ModelData(), forked_data.get());
}

namespace {
class EnvValueGuard {
 public:
  explicit EnvValueGuard(const char *name) : name_(name) {
    const char *value = std::getenv(name_.c_str());
    if (value != nullptr) {
      old_value_ = value;
      had_value_ = true;
    }
  }

  ~EnvValueGuard() {
    if (had_value_) {
      (void)setenv(name_.c_str(), old_value_.c_str(), 1);
    } else {
      (void)unsetenv(name_.c_str());
    }
  }

 private:
  std::string name_;
  std::string old_value_;
  bool had_value_ = false;
};

void EnableOm2OnlineMode() {
  ASSERT_EQ(setenv("ENABLE_RUNTIME_OM2", "1", 1), 0);
}

std::string MakeFakeOm2SoSource() {
  return R"(
#include <cstdint>
#include <cstddef>
extern "C" {
int Om2ModelCreate(void **model_handle, void **rt_model_handle, const char **, const void **,
                   size_t *, int, void **, void *, uint64_t *, unsigned int, void *) {
  if (model_handle) *model_handle = (void*)0x1;
  if (rt_model_handle) *rt_model_handle = (void*)0x2;
  return 0;
}
int Om2ModelLoad(void **) { return 0; }
int Om2ModelRun(void **, int, void **, int, void **, int) { return 0; }
int Om2ModelRunAsync(void **, void *, int, void **, int, void **) { return 0; }
int Om2ModelDestroy(void **) { return 0; }
}
)";
}

std::vector<uint8_t> ReadFileBytes(const std::string &path) {
  std::ifstream ifs(path, std::ios::binary | std::ios::ate);
  if (!ifs.is_open()) {
    return {};
  }
  const auto size = ifs.tellg();
  ifs.seekg(0, std::ios::beg);
  std::vector<uint8_t> data(static_cast<size_t>(size));
  ifs.read(reinterpret_cast<char *>(data.data()), size);
  return data;
}

gert::Om2ModelData MakeOm2ModelDataWithFakeSo(const std::string &so_path) {
  gert::Om2ModelData model_data;
  model_data.model_meta.model_name = "st_om2_test_model";
  model_data.model_meta.root_graph_name = "test_graph";
  model_data.model_meta.work_size = 1024U;

  ge::Om2TensorDesc input_desc;
  input_desc.SetName("input");
  input_desc.SetDataType(ge::DT_FLOAT);
  input_desc.SetShape({1, 4});
  input_desc.SetSize(16U);
  model_data.model_meta.input_desc.push_back(input_desc);
  model_data.model_meta.input_desc_v2.push_back(input_desc);

  ge::Om2TensorDesc output_desc;
  output_desc.SetName("output");
  output_desc.SetDataType(ge::DT_FLOAT);
  output_desc.SetShape({1, 4});
  output_desc.SetSize(16U);
  model_data.model_meta.output_desc.push_back(output_desc);
  model_data.model_meta.output_desc_v2.push_back(output_desc);

  auto so_bytes = ReadFileBytes(so_path);
  model_data.program_body.so_artifact.file_name = "libst_test_model_om2.so";
  model_data.program_body.so_artifact.data = std::string(so_bytes.begin(), so_bytes.end());

  return model_data;
}
}  // namespace

class Om2OnlineModelExecutorTest : public testing::Test {
 protected:
  static void SetUpTestSuite() {
    test_work_dir_ = EnvPath().GetOrCreateCaseTmpPath("Om2OnlineModelExecutorTest");
    setenv("ASCEND_WORK_PATH", test_work_dir_.c_str(), 1);

    const std::string src_path = PathUtils::Join({test_work_dir_, "fake_om2.cpp"});
    fake_so_path_ = PathUtils::Join({test_work_dir_, "libfake_om2.so"});
    std::ofstream ofs(src_path, std::ios::binary | std::ios::trunc);
    ASSERT_TRUE(ofs.is_open());
    ofs << MakeFakeOm2SoSource();
    ofs.close();
    const std::string compile_cmd = "ASAN_OPTIONS=detect_leaks=0 LSAN_OPTIONS=detect_leaks=0 g++ -shared -fPIC -o " +
                                    fake_so_path_ + " " + src_path;
    ASSERT_EQ(std::system(compile_cmd.c_str()), 0);
    ASSERT_EQ(mmAccess2(fake_so_path_.c_str(), M_F_OK), EOK);
  }

  static void TearDownTestSuite() {
    unsetenv("ASCEND_WORK_PATH");
    EnvPath().RemoveRfCaseTmpPath("Om2OnlineModelExecutorTest");
  }

  void TearDown() override {
    unsetenv("ENABLE_RUNTIME_OM2");
  }

  static std::string test_work_dir_;
  static std::string fake_so_path_;
};

std::string Om2OnlineModelExecutorTest::test_work_dir_;
std::string Om2OnlineModelExecutorTest::fake_so_path_;

TEST_F(Om2OnlineModelExecutorTest, LoadAndUnload_Success) {
  EnvValueGuard guard("ENABLE_RUNTIME_OM2");
  EnableOm2OnlineMode();

  ModelExecutor model_executor;
  EXPECT_EQ(model_executor.Initialize({}, 0), SUCCESS);

  auto compute_graph = std::make_shared<ComputeGraph>("test_graph");
  GeRootModelPtr ge_root_model = std::make_shared<GeRootModel>();
  EXPECT_EQ(ge_root_model->Initialize(compute_graph), SUCCESS);

  GeModelPtr ge_model = std::make_shared<GeModel>();
  ge_model->SetGraph(compute_graph);
  ge_root_model->SetSubgraphInstanceNameToModel(compute_graph->GetName(), ge_model);

  GraphId graph_id = 5001;
  GraphNodePtr graph_node = std::make_shared<ge::GraphNode>(graph_id);
  graph_node->SetGeRootModel(ge_root_model);

  auto model_data = std::make_shared<gert::Om2ModelData>(MakeOm2ModelDataWithFakeSo(fake_so_path_));
  ge_root_model->SetOm2ModelData(model_data);

  EXPECT_EQ(model_executor.LoadGraph(ge_root_model, graph_node), SUCCESS);
  EXPECT_EQ(model_executor.UnloadGraph(ge_root_model, graph_id), SUCCESS);
  EXPECT_EQ(model_executor.Finalize(), SUCCESS);
}

TEST_F(Om2OnlineModelExecutorTest, RunGraph_Success) {
  EnvValueGuard guard("ENABLE_RUNTIME_OM2");
  EnableOm2OnlineMode();

  ModelExecutor model_executor;
  EXPECT_EQ(model_executor.Initialize({}, 0), SUCCESS);

  auto compute_graph = std::make_shared<ComputeGraph>("test_graph");
  GeRootModelPtr ge_root_model = std::make_shared<GeRootModel>();
  EXPECT_EQ(ge_root_model->Initialize(compute_graph), SUCCESS);

  GeModelPtr ge_model = std::make_shared<GeModel>();
  ge_model->SetGraph(compute_graph);
  ge_root_model->SetSubgraphInstanceNameToModel(compute_graph->GetName(), ge_model);

  GraphId graph_id = 5002;
  GraphNodePtr graph_node = std::make_shared<ge::GraphNode>(graph_id);
  graph_node->SetGeRootModel(ge_root_model);

  auto model_data = std::make_shared<gert::Om2ModelData>(MakeOm2ModelDataWithFakeSo(fake_so_path_));
  ge_root_model->SetOm2ModelData(model_data);

  EXPECT_EQ(model_executor.LoadGraph(ge_root_model, graph_node), SUCCESS);

  std::vector<uint8_t> data(16U, 0U);
  gert::Tensor input;
  input.SetPlacement(gert::kOnDeviceHbm);
  input.SetData(gert::TensorData(data.data(), nullptr, data.size(), gert::kOnDeviceHbm));

  gert::Tensor output;
  output.SetPlacement(gert::kOnDeviceHbm);
  output.SetData(gert::TensorData(data.data(), nullptr, data.size(), gert::kOnDeviceHbm));

  std::vector<gert::Tensor> inputs;
  inputs.push_back(std::move(input));
  std::vector<gert::Tensor> outputs;
  outputs.push_back(std::move(output));
  EXPECT_EQ(model_executor.RunGraph(graph_node, graph_id, inputs, outputs), SUCCESS);

  EXPECT_EQ(model_executor.UnloadGraph(ge_root_model, graph_id), SUCCESS);
  EXPECT_EQ(model_executor.Finalize(), SUCCESS);
}

TEST_F(Om2OnlineModelExecutorTest, RunGraph_EmptyOutputs_PrepareOm2Outputs) {
  EnvValueGuard guard("ENABLE_RUNTIME_OM2");
  EnableOm2OnlineMode();

  ModelExecutor model_executor;
  EXPECT_EQ(model_executor.Initialize({}, 0), SUCCESS);

  auto compute_graph = std::make_shared<ComputeGraph>("test_graph");
  GeRootModelPtr ge_root_model = std::make_shared<GeRootModel>();
  EXPECT_EQ(ge_root_model->Initialize(compute_graph), SUCCESS);

  GeModelPtr ge_model = std::make_shared<GeModel>();
  ge_model->SetGraph(compute_graph);
  ge_root_model->SetSubgraphInstanceNameToModel(compute_graph->GetName(), ge_model);

  GraphId graph_id = 5003;
  GraphNodePtr graph_node = std::make_shared<ge::GraphNode>(graph_id);
  graph_node->SetGeRootModel(ge_root_model);

  auto model_data = std::make_shared<gert::Om2ModelData>(MakeOm2ModelDataWithFakeSo(fake_so_path_));
  ge_root_model->SetOm2ModelData(model_data);

  EXPECT_EQ(model_executor.LoadGraph(ge_root_model, graph_node), SUCCESS);

  std::vector<uint8_t> data(16U, 0U);
  gert::Tensor input;
  input.SetPlacement(gert::kOnDeviceHbm);
  input.SetData(gert::TensorData(data.data(), nullptr, data.size(), gert::kOnDeviceHbm));

  std::vector<gert::Tensor> inputs;
  inputs.push_back(std::move(input));
  std::vector<gert::Tensor> outputs;
  EXPECT_EQ(model_executor.RunGraph(graph_node, graph_id, inputs, outputs), SUCCESS);
  EXPECT_EQ(outputs.size(), 1U);

  EXPECT_EQ(model_executor.UnloadGraph(ge_root_model, graph_id), SUCCESS);
  EXPECT_EQ(model_executor.Finalize(), SUCCESS);
}

TEST_F(Om2OnlineModelExecutorTest, RunGraph_HostInput_ReturnsUnsupported) {
  EnvValueGuard guard("ENABLE_RUNTIME_OM2");
  EnableOm2OnlineMode();

  ModelExecutor model_executor;
  EXPECT_EQ(model_executor.Initialize({}, 0), SUCCESS);

  auto compute_graph = std::make_shared<ComputeGraph>("test_graph");
  GeRootModelPtr ge_root_model = std::make_shared<GeRootModel>();
  EXPECT_EQ(ge_root_model->Initialize(compute_graph), SUCCESS);

  GeModelPtr ge_model = std::make_shared<GeModel>();
  ge_model->SetGraph(compute_graph);
  ge_root_model->SetSubgraphInstanceNameToModel(compute_graph->GetName(), ge_model);

  GraphId graph_id = 5004;
  GraphNodePtr graph_node = std::make_shared<ge::GraphNode>(graph_id);
  graph_node->SetGeRootModel(ge_root_model);

  auto model_data = std::make_shared<gert::Om2ModelData>(MakeOm2ModelDataWithFakeSo(fake_so_path_));
  ge_root_model->SetOm2ModelData(model_data);

  EXPECT_EQ(model_executor.LoadGraph(ge_root_model, graph_node), SUCCESS);

  gert::Tensor host_input;
  host_input.SetPlacement(gert::kOnHost);
  std::vector<uint8_t> data(16U, 0U);
  host_input.SetData(gert::TensorData(data.data(), nullptr, data.size(), gert::kOnHost));

  gert::Tensor output;
  output.SetPlacement(gert::kOnDeviceHbm);
  output.SetData(gert::TensorData(data.data(), nullptr, data.size(), gert::kOnDeviceHbm));

  std::vector<gert::Tensor> inputs;
  inputs.push_back(std::move(host_input));
  std::vector<gert::Tensor> outputs;
  outputs.push_back(std::move(output));
  EXPECT_EQ(model_executor.RunGraph(graph_node, graph_id, inputs, outputs), GE_GRAPH_UNSUPPORTED);

  EXPECT_EQ(model_executor.UnloadGraph(ge_root_model, graph_id), SUCCESS);
  EXPECT_EQ(model_executor.Finalize(), SUCCESS);
}

TEST_F(Om2OnlineModelExecutorTest, RunGraph_NotLoaded_ReturnsGraphNotExist) {
  EnvValueGuard guard("ENABLE_RUNTIME_OM2");
  EnableOm2OnlineMode();

  ModelExecutor model_executor;
  EXPECT_EQ(model_executor.Initialize({}, 0), SUCCESS);

  auto compute_graph = std::make_shared<ComputeGraph>("test_graph");
  GeRootModelPtr ge_root_model = std::make_shared<GeRootModel>();
  EXPECT_EQ(ge_root_model->Initialize(compute_graph), SUCCESS);

  GeModelPtr ge_model = std::make_shared<GeModel>();
  ge_model->SetGraph(compute_graph);
  ge_root_model->SetSubgraphInstanceNameToModel(compute_graph->GetName(), ge_model);

  auto model_data = std::make_shared<gert::Om2ModelData>(MakeOm2ModelDataWithFakeSo(fake_so_path_));
  ge_root_model->SetOm2ModelData(model_data);

  GraphId graph_id = 5005;
  GraphNodePtr graph_node = std::make_shared<ge::GraphNode>(graph_id);
  graph_node->SetGeRootModel(ge_root_model);

  std::vector<gert::Tensor> inputs(1);
  std::vector<gert::Tensor> outputs(1);
  EXPECT_EQ(model_executor.RunGraph(graph_node, graph_id, inputs, outputs), GE_GRAPH_GRAPH_NOT_EXIST);

  EXPECT_EQ(model_executor.Finalize(), SUCCESS);
}

TEST_F(Om2OnlineModelExecutorTest, UnloadGraph_NotLoaded_ReturnsSuccess) {
  EnvValueGuard guard("ENABLE_RUNTIME_OM2");
  EnableOm2OnlineMode();

  ModelExecutor model_executor;
  EXPECT_EQ(model_executor.Initialize({}, 0), SUCCESS);

  auto compute_graph = std::make_shared<ComputeGraph>("test_graph");
  GeRootModelPtr ge_root_model = std::make_shared<GeRootModel>();
  EXPECT_EQ(ge_root_model->Initialize(compute_graph), SUCCESS);

  GraphId graph_id = 5006;
  EXPECT_EQ(model_executor.UnloadGraph(ge_root_model, graph_id), SUCCESS);

  EXPECT_EQ(model_executor.Finalize(), SUCCESS);
}

TEST_F(Om2OnlineModelExecutorTest, DumpDebugJSONPrint_ReturnsUnsupported) {
  EnvValueGuard guard("ENABLE_RUNTIME_OM2");
  EnableOm2OnlineMode();

  ModelExecutor model_executor;
  AscendString json_result;
  EXPECT_EQ(model_executor.DumpDebugJSONPrint(1U, 1U, 0U, json_result), GE_GRAPH_UNSUPPORTED);
}

TEST_F(Om2OnlineModelExecutorTest, UpdateFeatureMemoryBase_ReturnsUnsupported) {
  EnvValueGuard guard("ENABLE_RUNTIME_OM2");
  EnableOm2OnlineMode();

  ModelExecutor model_executor;
  EXPECT_EQ(model_executor.Initialize({}, 0), SUCCESS);

  GraphNodePtr graph_node = std::make_shared<ge::GraphNode>(1U);
  graph_node->SetGeRootModel(std::make_shared<GeRootModel>());

  EXPECT_EQ(model_executor.UpdateFeatureMemoryBase(graph_node, 0U, 0U), GE_GRAPH_UNSUPPORTED);
  EXPECT_EQ(model_executor.Finalize(), SUCCESS);
}

TEST_F(Om2OnlineModelExecutorTest, PaRemapped_ReturnsUnsupported) {
  EnvValueGuard guard("ENABLE_RUNTIME_OM2");
  EnableOm2OnlineMode();

  ModelExecutor model_executor;
  EXPECT_EQ(model_executor.Initialize({}, 0), SUCCESS);

  GraphNodePtr graph_node = std::make_shared<ge::GraphNode>(1U);
  graph_node->SetGeRootModel(std::make_shared<GeRootModel>());

  std::vector<std::pair<uint64_t, uint64_t>> cross_ranges;
  EXPECT_EQ(model_executor.PaRemapped(graph_node, 0U, 0U, 0U, cross_ranges), GE_GRAPH_UNSUPPORTED);
  EXPECT_EQ(model_executor.Finalize(), SUCCESS);
}

TEST_F(Om2OnlineModelExecutorTest, RunGraphWithStream_Om2Mode_Success) {
  EnvValueGuard guard("ENABLE_RUNTIME_OM2");
  EnableOm2OnlineMode();

  ModelExecutor model_executor;
  EXPECT_EQ(model_executor.Initialize({}, 0), SUCCESS);

  auto compute_graph = std::make_shared<ComputeGraph>("test_graph");
  GeRootModelPtr ge_root_model = std::make_shared<GeRootModel>();
  EXPECT_EQ(ge_root_model->Initialize(compute_graph), SUCCESS);

  GeModelPtr ge_model = std::make_shared<GeModel>();
  ge_model->SetGraph(compute_graph);
  ge_root_model->SetSubgraphInstanceNameToModel(compute_graph->GetName(), ge_model);

  GraphId graph_id = 5007;
  GraphNodePtr graph_node = std::make_shared<ge::GraphNode>(graph_id);
  graph_node->SetGeRootModel(ge_root_model);

  auto model_data = std::make_shared<gert::Om2ModelData>(MakeOm2ModelDataWithFakeSo(fake_so_path_));
  ge_root_model->SetOm2ModelData(model_data);
  EXPECT_EQ(model_executor.LoadGraph(ge_root_model, graph_node), SUCCESS);

  GeTensorDesc input_desc(GeShape({1, 4}), FORMAT_ND, DT_FLOAT);
  input_desc.SetPlacement(kPlacementDevice);
  GeTensorDesc output_desc(GeShape({1, 4}), FORMAT_ND, DT_FLOAT);
  output_desc.SetPlacement(kPlacementDevice);
  std::vector<uint8_t> input_data(16U, 0U);
  std::vector<uint8_t> output_data(16U, 0U);
  std::vector<GeTensor> inputs{GeTensor(input_desc, input_data.data(), input_data.size())};
  std::vector<GeTensor> outputs{GeTensor(output_desc, output_data.data(), output_data.size())};

  EXPECT_EQ(model_executor.RunGraphWithStream(graph_node, graph_id, nullptr, inputs, outputs), SUCCESS);
  EXPECT_EQ(model_executor.UnloadGraph(ge_root_model, graph_id), SUCCESS);
  EXPECT_EQ(model_executor.Finalize(), SUCCESS);
}

TEST_F(Om2OnlineModelExecutorTest, ExecuteGraphWithStream_Om2Mode_Success) {
  EnvValueGuard guard("ENABLE_RUNTIME_OM2");
  EnableOm2OnlineMode();

  ModelExecutor model_executor;
  EXPECT_EQ(model_executor.Initialize({}, 0), SUCCESS);

  auto compute_graph = std::make_shared<ComputeGraph>("test_graph");
  GeRootModelPtr ge_root_model = std::make_shared<GeRootModel>();
  EXPECT_EQ(ge_root_model->Initialize(compute_graph), SUCCESS);

  GeModelPtr ge_model = std::make_shared<GeModel>();
  ge_model->SetGraph(compute_graph);
  ge_root_model->SetSubgraphInstanceNameToModel(compute_graph->GetName(), ge_model);

  GraphId graph_id = 5008;
  GraphNodePtr graph_node = std::make_shared<ge::GraphNode>(graph_id);
  graph_node->SetGeRootModel(ge_root_model);

  auto model_data = std::make_shared<gert::Om2ModelData>(MakeOm2ModelDataWithFakeSo(fake_so_path_));
  ge_root_model->SetOm2ModelData(model_data);
  EXPECT_EQ(model_executor.LoadGraph(ge_root_model, graph_node), SUCCESS);

  std::vector<uint8_t> data(16U, 0U);
  gert::Tensor input;
  input.SetPlacement(gert::kOnDeviceHbm);
  input.SetData(gert::TensorData(data.data(), nullptr, data.size(), gert::kOnDeviceHbm));

  gert::Tensor output;
  output.SetPlacement(gert::kOnDeviceHbm);
  output.SetData(gert::TensorData(data.data(), nullptr, data.size(), gert::kOnDeviceHbm));

  std::vector<gert::Tensor> inputs;
  inputs.push_back(std::move(input));
  std::vector<gert::Tensor> outputs;
  outputs.push_back(std::move(output));

  EXPECT_EQ(model_executor.ExecuteGraphWithStream(graph_node, graph_id, nullptr, inputs, outputs), SUCCESS);
  EXPECT_EQ(model_executor.UnloadGraph(ge_root_model, graph_id), SUCCESS);
  EXPECT_EQ(model_executor.Finalize(), SUCCESS);
}

TEST_F(Om2OnlineModelExecutorTest, RunThread_Om2Mode_ReturnsUnsupported) {
  EnvValueGuard guard("ENABLE_RUNTIME_OM2");
  EnableOm2OnlineMode();

  ModelExecutor model_executor;
  EXPECT_EQ(model_executor.Initialize({}, 0), SUCCESS);
  model_executor.StartRunThread();

  GraphId graph_id = 5009;
  error_message::ErrorManagerContext error_context;
  GEThreadLocalContext context = GetThreadLocalContext();

  std::promise<Status> status_promise;
  auto status_future = status_promise.get_future();
  const auto callback = [&status_promise](Status status, std::vector<gert::Tensor> &outputs) {
    status_promise.set_value(status);
  };

  auto compute_graph = std::make_shared<ComputeGraph>("test_graph");
  GeRootModelPtr ge_root_model = std::make_shared<GeRootModel>();
  EXPECT_EQ(ge_root_model->Initialize(compute_graph), SUCCESS);

  GeModelPtr ge_model = std::make_shared<GeModel>();
  ge_model->SetGraph(compute_graph);
  ge_root_model->SetSubgraphInstanceNameToModel(compute_graph->GetName(), ge_model);

  GraphNodePtr graph_node = std::make_shared<ge::GraphNode>(graph_id);
  graph_node->SetGeRootModel(ge_root_model);
  graph_node->SetLoadFlag(true);
  graph_node->IncreaseLoadCount();
  graph_node->Lock();

  std::vector<gert::Tensor> input_tensors(1);

  auto run_args = std::make_shared<RunArgs>();
  ASSERT_TRUE(run_args != nullptr);
  run_args->graph_node = graph_node;
  run_args->graph_id = graph_id;
  run_args->session_id = 0;
  run_args->error_context = error_context;
  run_args->input_tensor = std::move(input_tensors);
  run_args->context = context;
  run_args->callback = callback;
  EXPECT_EQ(model_executor.PushRunArgs(run_args), SUCCESS);

  auto status = status_future.wait_for(std::chrono::seconds(5));
  ASSERT_EQ(status, std::future_status::ready);
  EXPECT_EQ(status_future.get(), GE_GRAPH_UNSUPPORTED);

  EXPECT_EQ(model_executor.Finalize(), SUCCESS);
}

TEST_F(Om2OnlineModelExecutorTest, LoadGraph_ExternalConstAndFeatureMemory) {
  EnvValueGuard guard("ENABLE_RUNTIME_OM2");
  EnableOm2OnlineMode();

  ModelExecutor model_executor;
  EXPECT_EQ(model_executor.Initialize({}, 0), SUCCESS);

  auto compute_graph = std::make_shared<ComputeGraph>("test_graph");
  GeRootModelPtr ge_root_model = std::make_shared<GeRootModel>();
  EXPECT_EQ(ge_root_model->Initialize(compute_graph), SUCCESS);

  GeModelPtr ge_model = std::make_shared<GeModel>();
  ge_model->SetGraph(compute_graph);
  ge_root_model->SetSubgraphInstanceNameToModel(compute_graph->GetName(), ge_model);

  GraphId graph_id = 5010;
  GraphNodePtr graph_node = std::make_shared<ge::GraphNode>(graph_id);
  graph_node->SetGeRootModel(ge_root_model);

  std::vector<uint8_t> const_mem(16U, 0U);
  std::vector<uint8_t> feature_mem(1024U, 0U);
  graph_node->SetConstMemoryBase(const_mem.data(), const_mem.size());
  graph_node->SetFeatureMemoryBase(feature_mem.data(), feature_mem.size());

  auto model_data = std::make_shared<gert::Om2ModelData>(MakeOm2ModelDataWithFakeSo(fake_so_path_));
  ge_root_model->SetOm2ModelData(model_data);

  EXPECT_EQ(model_executor.LoadGraph(ge_root_model, graph_node), SUCCESS);
  EXPECT_EQ(model_executor.UnloadGraph(ge_root_model, graph_id), SUCCESS);
  EXPECT_EQ(model_executor.Finalize(), SUCCESS);
}

TEST_F(Om2OnlineModelExecutorTest, RunGraph_InputCountMismatch_ReturnsParamInvalid) {
  EnvValueGuard guard("ENABLE_RUNTIME_OM2");
  EnableOm2OnlineMode();

  ModelExecutor model_executor;
  EXPECT_EQ(model_executor.Initialize({}, 0), SUCCESS);

  auto compute_graph = std::make_shared<ComputeGraph>("test_graph");
  GeRootModelPtr ge_root_model = std::make_shared<GeRootModel>();
  EXPECT_EQ(ge_root_model->Initialize(compute_graph), SUCCESS);

  GeModelPtr ge_model = std::make_shared<GeModel>();
  ge_model->SetGraph(compute_graph);
  ge_root_model->SetSubgraphInstanceNameToModel(compute_graph->GetName(), ge_model);

  GraphId graph_id = 5011;
  GraphNodePtr graph_node = std::make_shared<ge::GraphNode>(graph_id);
  graph_node->SetGeRootModel(ge_root_model);

  auto model_data = std::make_shared<gert::Om2ModelData>(MakeOm2ModelDataWithFakeSo(fake_so_path_));
  ge_root_model->SetOm2ModelData(model_data);

  EXPECT_EQ(model_executor.LoadGraph(ge_root_model, graph_node), SUCCESS);

  std::vector<gert::Tensor> inputs;
  std::vector<gert::Tensor> outputs(1);
  EXPECT_EQ(model_executor.RunGraph(graph_node, graph_id, inputs, outputs), PARAM_INVALID);

  EXPECT_EQ(model_executor.UnloadGraph(ge_root_model, graph_id), SUCCESS);
  EXPECT_EQ(model_executor.Finalize(), SUCCESS);
}

}  // namespace ge
