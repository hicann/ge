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
#include <mutex>
#include <future>
#include <chrono>
#include <condition_variable>
#include <fstream>

#include "macro_utils/dt_public_scope.h"
#include "graph/load/graph_loader.h"
#include "graph/execute/model_executor.h"
#include "graph/manager/graph_manager.h"
#include "graph/manager/active_memory_allocator.h"
#include "graph/manager/mem_manager.h"
#include "graph/manager/graph_var_manager.h"
#include "graph/load/model_manager/model_manager.h"
#include "compiler/graph/build/graph_compile_summary_impl.h"
#include "tests/ge/ut/ge/graph/passes/graph_builder_utils.h"
#include "base/common/model/external_allocator_manager.h"
#include "graph/ge_context.h"
#include "hcom/hcom_topo_info.h"
#include "executor/ge_executor.h"
#include "graph_metadef/common/ge_common/util.h"
#include "graph/custom_op_factory.h"
#include "common/om2/om2_model_data.h"
#include "common/helper/om2/om2_utils.h"
#include "common/memory/tensor_trans_utils.h"
#include "common/env_path.h"
#include "common/path_utils.h"
#include "mmpa/mmpa_api.h"

using namespace std;

namespace ge {
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
  model_data.model_meta.model_name = "test_model";
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
  model_data.program_body.so_artifact.file_name = "libtest_model_om2.so";
  model_data.program_body.so_artifact.data = std::string(so_bytes.begin(), so_bytes.end());

  return model_data;
}

gert::Om2ModelData MakeMinimalOm2ModelData(size_t work_size = 1024U, size_t tensor_size = 16U) {
  gert::Om2ModelData model_data;
  model_data.model_meta.model_name = "om2_ut_model";
  model_data.model_meta.root_graph_name = "test_graph";
  model_data.model_meta.work_size = work_size;

  ge::Om2TensorDesc input_desc;
  input_desc.SetName("input0");
  input_desc.SetDataType(DT_FLOAT);
  input_desc.SetFormat(FORMAT_ND);
  input_desc.SetShape({1, 4});
  input_desc.SetSize(tensor_size);
  model_data.model_meta.input_desc.emplace_back(input_desc);

  ge::Om2TensorDesc output_desc;
  output_desc.SetName("output0");
  output_desc.SetDataType(DT_FLOAT);
  output_desc.SetFormat(FORMAT_ND);
  output_desc.SetShape({1, 4});
  output_desc.SetSize(tensor_size);
  model_data.model_meta.output_desc.emplace_back(output_desc);
  return model_data;
}

class ExternalAllocatorUtStub : public Allocator {
 public:
  MemBlock *Malloc(size_t size) override {
    block_ = new (std::nothrow) MemBlock(*this, &mem, size);
    return block_;
  }
  MemBlock *MallocAdvise(size_t size, void *addr) override {
    block_ = new (std::nothrow) MemBlock(*this, &mem, size);
    advise_cnt++;
    return block_;
  }
  void Free(MemBlock *block) override {
    delete block;
    if (block == block_) {
      block_ = nullptr;
    }
  }
  MemBlock *GetBlockAddr() {
    return block_;
  }
  uint64_t GetAdviseCnt() {
    return advise_cnt;
  }

 private:
  uint64_t mem = 0;
  MemBlock *block_{nullptr};
  uint64_t advise_cnt = 0U;
};

ge::ComputeGraphPtr CreateGraphWithConstOutput() {
  ge::ut::GraphBuilder builder("graph");
  auto data = builder.AddNode("data1", "Data", 1, 1);
  auto netoutput = builder.AddNode("Node_Output", "NetOutput", 1, 0);
  builder.AddDataEdge(data, 0, netoutput, 0);
  data->GetOpDesc()->SetOutputOffset({1});
  netoutput->GetOpDesc()->SetInputOffset({1});
  return builder.GetGraph();
}

void CreateSummaryCompiledModel(GraphNodePtr &graph_node, GeModelPtr &ge_model, GeRootModelPtr &ge_root_model,
                                bool has_p2p = true) {
  auto compute_graph = CreateGraphWithConstOutput();
  ge_root_model = MakeShared<GeRootModel>();
  EXPECT_EQ(ge_root_model->Initialize(compute_graph), SUCCESS);
  auto flow_root_compute_graph = MakeShared<ComputeGraph>("test_graph");
  flow_root_compute_graph->SetAllSubgraphs({compute_graph});

  AttrUtils::SetStr(compute_graph, "_split_logic_stream_2_origin_logic_stream", "");

  ge_model = MakeShared<GeModel>();
  ge_model->SetGraph(compute_graph);
  ge_root_model->SetSubgraphInstanceNameToModel(compute_graph->GetName(), ge_model);
  ge_root_model->SetModelId(1U);

  GraphId graph_id = 1;
  graph_node = MakeShared<ge::GraphNode>(graph_id);
  graph_node->SetGeRootModel(ge_root_model);
  graph_node->SetComputeGraph(compute_graph);

  AttrUtils::SetInt(ge_model, ATTR_MODEL_WEIGHT_SIZE, 512);
  AttrUtils::SetInt(ge_model, ATTR_MODEL_MEMORY_SIZE, 1024);
  AttrUtils::SetInt(ge_model, ATTR_MODEL_ZERO_COPY_MEMORY_SIZE, 0);
  AttrUtils::SetInt(ge_model, ATTR_MODEL_STREAM_NUM, 1);
  AttrUtils::SetInt(ge_model, ATTR_MODEL_EVENT_NUM, 2);
  if (has_p2p) {
    AttrUtils::SetInt(ge_model, ATTR_MODEL_P2P_MEMORY_SIZE, 1024);
  }

  uint64_t mem = 0UL;
  std::vector<std::vector<int64_t>> sub_mem_infos;
  std::vector<int64_t> sub_mem_offset;
  sub_mem_offset.emplace_back(0x2U);             // mem_type RT_MEMORY_HBM 0x2U
  sub_mem_offset.emplace_back((int64_t)(&mem));  // mem_offset_base
  sub_mem_offset.emplace_back(sizeof(mem));      // mem_size
  sub_mem_offset.emplace_back(1UL);              // is_fixed_addr_prior
  sub_mem_infos.emplace_back(sub_mem_offset);
  AttrUtils::SetListListInt(ge_model, ATTR_MODEL_SUB_MEMORY_INFO, sub_mem_infos);

  std::map<std::string, std::string> graph_options;
  graph_options.emplace(ge::OPTION_FEATURE_BASE_REFRESHABLE, "1");
  GetThreadLocalContext().SetGraphOption(graph_options);
  graph_node->SetOptions(graph_options);
}
}  // namespace
class UtestModelExecutorTest : public testing::Test {
 protected:
  void SetUp() {}
  void TearDown() {
    EXPECT_TRUE(ModelManager::GetInstance().model_map_.empty());
    EXPECT_TRUE(ModelManager::GetInstance().hybrid_model_map_.empty());
  }
};

TEST_F(UtestModelExecutorTest, test_get_total_memory_size) {
  ModelExecutor model_executor;
  size_t free_mem = 0U;
  size_t total_mem_size = 0;
  EXPECT_EQ(model_executor.GetDeviceMemorySize(free_mem, total_mem_size), SUCCESS);
  EXPECT_EQ(total_mem_size, 128UL * 1024UL * 1024UL);
}

TEST_F(UtestModelExecutorTest, test_load_graph_sync) {
  ModelExecutor model_executor;
  EXPECT_EQ(model_executor.Initialize({}, 0), SUCCESS);
  model_executor.StartRunThread();

  auto compute_graph = MakeShared<ComputeGraph>("test_graph");
  GeRootModelPtr ge_root_model = MakeShared<GeRootModel>();
  EXPECT_EQ(ge_root_model->Initialize(compute_graph), SUCCESS);
  ge_root_model->SetCustomOpRegistry(CustomOpFactory::GetGlobalRegistryPtr());

  GeModelPtr ge_model = MakeShared<GeModel>();
  ge_model->SetGraph(compute_graph);

  shared_ptr<domi::ModelTaskDef> model_task_def = std::make_shared<domi::ModelTaskDef>();
  ge_model->SetModelTaskDef(model_task_def);

  ge_root_model->SetSubgraphInstanceNameToModel(compute_graph->GetName(), ge_model);

  GraphId graph_id = 1;
  GraphNodePtr graph_node = MakeShared<ge::GraphNode>(graph_id);
  graph_node->SetGeRootModel(ge_root_model);
  graph_node->SetLoadFlag(true);
  graph_node->SetAsync(false);

  EXPECT_EQ(model_executor.LoadGraph(ge_root_model, graph_node), SUCCESS);
  EXPECT_EQ(model_executor.UnloadGraph(ge_root_model, graph_id), SUCCESS);

  EXPECT_EQ(model_executor.Finalize(), SUCCESS);
}

TEST_F(UtestModelExecutorTest, test_load_graph_async) {
  ModelExecutor model_executor;
  EXPECT_EQ(model_executor.Initialize({}, 0), SUCCESS);
  model_executor.StartRunThread();

  Graph graph("test_graph");
  auto compute_graph = MakeShared<ComputeGraph>("test_graph");
  GeRootModelPtr ge_root_model = MakeShared<GeRootModel>();
  EXPECT_EQ(ge_root_model->Initialize(compute_graph), SUCCESS);
  ge_root_model->SetCustomOpRegistry(CustomOpFactory::GetGlobalRegistryPtr());
  GeModelPtr ge_model = MakeShared<GeModel>();
  ge_model->SetGraph(compute_graph);

  shared_ptr<domi::ModelTaskDef> model_task_def = std::make_shared<domi::ModelTaskDef>();
  ge_model->SetModelTaskDef(model_task_def);

  ge_root_model->SetSubgraphInstanceNameToModel(compute_graph->GetName(), ge_model);

  GraphId graph_id = 1001;
  GraphNodePtr graph_node = MakeShared<ge::GraphNode>(graph_id);
  graph_node->SetGeRootModel(ge_root_model);
  graph_node->SetLoadFlag(true);
  graph_node->SetAsync(true);

  EXPECT_EQ(model_executor.LoadGraph(ge_root_model, graph_node), SUCCESS);

  EXPECT_EQ(model_executor.UnloadGraph(ge_root_model, graph_id), SUCCESS);

  EXPECT_EQ(model_executor.Finalize(), SUCCESS);
}

TEST_F(UtestModelExecutorTest, test_recover_graph) {
  ModelExecutor model_executor;
  EXPECT_EQ(model_executor.Initialize({}, 0), SUCCESS);
  model_executor.StartRunThread();

  Graph graph("test_graph");
  auto compute_graph = MakeShared<ComputeGraph>("test_graph");
  GeRootModelPtr ge_root_model = MakeShared<GeRootModel>();
  EXPECT_EQ(ge_root_model->Initialize(compute_graph), SUCCESS);
  ge_root_model->SetCustomOpRegistry(CustomOpFactory::GetGlobalRegistryPtr());

  GeModelPtr ge_model = MakeShared<GeModel>();
  ge_model->SetGraph(compute_graph);

  shared_ptr<domi::ModelTaskDef> model_task_def = std::make_shared<domi::ModelTaskDef>();
  ge_model->SetModelTaskDef(model_task_def);

  ge_root_model->SetSubgraphInstanceNameToModel(compute_graph->GetName(), ge_model);

  GraphId graph_id = 1001;
  GraphNodePtr graph_node = MakeShared<ge::GraphNode>(graph_id);
  graph_node->SetGeRootModel(ge_root_model);
  graph_node->SetLoadFlag(true);
  graph_node->SetAsync(true);

  EXPECT_EQ(model_executor.LoadGraph(ge_root_model, graph_node), SUCCESS);

  GeExecutor ge_executor;
  EXPECT_EQ(ge_executor.RecoverAllModel(0), SUCCESS);

  EXPECT_EQ(model_executor.UnloadGraph(ge_root_model, graph_id), SUCCESS);

  EXPECT_EQ(model_executor.Finalize(), SUCCESS);
}

TEST_F(UtestModelExecutorTest, test_load_graph_failed) {
  ModelExecutor model_executor;
  EXPECT_EQ(model_executor.Initialize({}, 0), SUCCESS);
  model_executor.StartRunThread();

  Graph graph("test_graph");
  auto compute_graph = MakeShared<ComputeGraph>("test_graph");
  GeRootModelPtr ge_root_model = MakeShared<GeRootModel>();
  EXPECT_EQ(ge_root_model->Initialize(compute_graph), SUCCESS);

  GraphId graph_id = 1;
  GraphNodePtr graph_node = MakeShared<ge::GraphNode>(graph_id);
  graph_node->SetGeRootModel(ge_root_model);
  graph_node->SetLoadFlag(true);
  graph_node->SetAsync(true);

  // GeModel is null, DavinciModel::Assign will return FAILED
  setenv(kEnvGeuseStaticMemory.c_str(), "1", true);
  EXPECT_NE(model_executor.LoadGraph(ge_root_model, graph_node), SUCCESS);  // GeModel is null
  EXPECT_EQ(model_executor.UnloadGraph(ge_root_model, graph_id), SUCCESS);

  EXPECT_EQ(model_executor.Finalize(), SUCCESS);
  unsetenv(kEnvGeuseStaticMemory.c_str());
}

TEST_F(UtestModelExecutorTest, test_check_and_release_event_success) {
  dlog_setlevel(GE_MODULE_NAME, DLOG_DEBUG, 0);
  ModelExecutor model_executor;
  EXPECT_EQ(model_executor.Initialize({}, 0), SUCCESS);
  model_executor.StartRunThread();
  model_executor.AddGraphNode(0, nullptr);

  uint32_t graph_num = 5;
  for (uint32_t i = 1; i < graph_num; i++) {
    GraphId graph_id = i;
    string graph_name = "test_graph" + to_string(graph_id);

    auto compute_graph = MakeShared<ComputeGraph>(graph_name);
    GeRootModelPtr ge_root_model = MakeShared<GeRootModel>();
    EXPECT_EQ(ge_root_model->Initialize(compute_graph), SUCCESS);
    ge_root_model->SetCustomOpRegistry(CustomOpFactory::GetGlobalRegistryPtr());

    GeModelPtr ge_model = MakeShared<GeModel>();
    ge_model->SetGraph(compute_graph);
    ge_root_model->SetSubgraphInstanceNameToModel(compute_graph->GetName(), ge_model);

    GraphNodePtr graph_node = MakeShared<ge::GraphNode>(graph_id);
    graph_node->SetGeRootModel(ge_root_model);
    graph_node->SetLoadFlag(true);
    graph_node->SetAsync(false);

    uint32_t event_num_dev_avail = 0;
    switch (graph_id) {
      case 1: {
        shared_ptr<domi::ModelTaskDef> model_task_def = std::make_shared<domi::ModelTaskDef>();
        ge_model->SetModelTaskDef(model_task_def);
        EXPECT_TRUE(AttrUtils::SetInt(ge_model, ATTR_MODEL_EVENT_NUM, 1));
        EXPECT_EQ(model_executor.LoadGraph(ge_root_model, graph_node), SUCCESS);
        break;
      }
      case 2: {
        shared_ptr<domi::ModelTaskDef> model_task_def = std::make_shared<domi::ModelTaskDef>();
        ge_model->SetModelTaskDef(model_task_def);

        (void)aclrtGetEventAvailNum(&event_num_dev_avail);
        EXPECT_TRUE(AttrUtils::SetInt(ge_model, ATTR_MODEL_EVENT_NUM, event_num_dev_avail + 1));
        EXPECT_EQ(model_executor.LoadGraph(ge_root_model, graph_node), SUCCESS);
        break;
      }
      case 3: {
        shared_ptr<domi::ModelTaskDef> model_task_def = std::make_shared<domi::ModelTaskDef>();
        ge_model->SetModelTaskDef(model_task_def);
        EXPECT_TRUE(AttrUtils::SetInt(ge_model, ATTR_MODEL_EVENT_NUM, 1));
        EXPECT_EQ(model_executor.LoadGraph(ge_root_model, graph_node), SUCCESS);
        break;
      }
      case 4: {
        shared_ptr<OpDesc> op_desc = std::make_shared<OpDesc>("add", "add");
        AttrUtils::SetBool(op_desc, ATTR_NAME_IS_BLOCKING_OP, true);
        compute_graph->AddNode(op_desc);
        AttrUtils::SetListStr(op_desc, "_hccl_group_id_list", {"group0", "group1"});
        EXPECT_EQ(HcomTopoInfo::Instance().SetGroupOrderedStream(0, "group0", (void *)1), GRAPH_SUCCESS);
        EXPECT_EQ(HcomTopoInfo::Instance().SetGroupOrderedStream(0, "group1", (void *)2), GRAPH_SUCCESS);

        shared_ptr<domi::ModelTaskDef> model_task_def = std::make_shared<domi::ModelTaskDef>();
        ge_model->SetModelTaskDef(model_task_def);

        (void)aclrtGetEventAvailNum(&event_num_dev_avail);
        EXPECT_TRUE(AttrUtils::SetInt(ge_model, ATTR_MODEL_EVENT_NUM,
                                      event_num_dev_avail - 2));  // 释放1个，kfc占用2个event， block占用1个event
        EXPECT_EQ(model_executor.LoadGraph(ge_root_model, graph_node), SUCCESS);
        EXPECT_EQ(model_executor.UnloadGraph(ge_root_model, graph_id), SUCCESS);
      }
      default:
        break;
    }
  }

  EXPECT_EQ(model_executor.Finalize(), SUCCESS);
  HcomTopoInfo::Instance().UnsetGroupOrderedStream(0, "group0");
  HcomTopoInfo::Instance().UnsetGroupOrderedStream(0, "group1");
  dlog_setlevel(GE_MODULE_NAME, DLOG_ERROR, 0);
}

TEST_F(UtestModelExecutorTest, test_check_and_release_event_failed) {
  ModelExecutor model_executor;
  EXPECT_EQ(model_executor.Initialize({}, 0), SUCCESS);
  model_executor.StartRunThread();
  model_executor.AddGraphNode(0, nullptr);

  uint32_t graph_num = 4;
  for (uint32_t i = 1; i < graph_num; i++) {
    GraphId graph_id = i;
    string graph_name = "test_graph" + to_string(graph_id);

    auto compute_graph = MakeShared<ComputeGraph>(graph_name);
    GeRootModelPtr ge_root_model = MakeShared<GeRootModel>();
    EXPECT_EQ(ge_root_model->Initialize(compute_graph), SUCCESS);
    ge_root_model->SetCustomOpRegistry(CustomOpFactory::GetGlobalRegistryPtr());

    GeModelPtr ge_model = MakeShared<GeModel>();
    ge_model->SetGraph(compute_graph);
    ge_root_model->SetSubgraphInstanceNameToModel(compute_graph->GetName(), ge_model);

    GraphNodePtr graph_node = MakeShared<ge::GraphNode>(graph_id);
    graph_node->SetGeRootModel(ge_root_model);
    graph_node->SetLoadFlag(true);
    graph_node->SetAsync(false);

    uint32_t event_num_dev_avail = 0;
    switch (graph_id) {
      case 1: {
        shared_ptr<domi::ModelTaskDef> model_task_def = std::make_shared<domi::ModelTaskDef>();
        ge_model->SetModelTaskDef(model_task_def);
        EXPECT_TRUE(AttrUtils::SetInt(ge_model, ATTR_MODEL_EVENT_NUM, 1));
        EXPECT_EQ(model_executor.LoadGraph(ge_root_model, graph_node), SUCCESS);
        break;
      }
      case 2: {
        shared_ptr<OpDesc> op_desc = std::make_shared<OpDesc>("add", "add");
        AttrUtils::SetBool(op_desc, ATTR_NAME_IS_BLOCKING_OP, true);
        compute_graph->AddNode(op_desc);
        shared_ptr<domi::ModelTaskDef> model_task_def = std::make_shared<domi::ModelTaskDef>();
        ge_model->SetModelTaskDef(model_task_def);

        (void)aclrtGetEventAvailNum(&event_num_dev_avail);
        EXPECT_TRUE(AttrUtils::SetInt(ge_model, ATTR_MODEL_EVENT_NUM, event_num_dev_avail + 32));
        EXPECT_EQ(model_executor.LoadGraph(ge_root_model, graph_node), FAILED);
        break;
      }
      case 3: {
        shared_ptr<OpDesc> op_desc = std::make_shared<OpDesc>("add", "add");
        AttrUtils::SetBool(op_desc, ATTR_NAME_IS_BLOCKING_OP, false);
        compute_graph->AddNode(op_desc);

        shared_ptr<domi::ModelTaskDef> model_task_def = std::make_shared<domi::ModelTaskDef>();
        ge_model->SetModelTaskDef(model_task_def);

        (void)aclrtGetEventAvailNum(&event_num_dev_avail);
        EXPECT_TRUE(AttrUtils::SetInt(ge_model, ATTR_MODEL_EVENT_NUM, event_num_dev_avail + 32));
        EXPECT_EQ(model_executor.LoadGraph(ge_root_model, graph_node), FAILED);
        break;
      }
      default:
        break;
    }
  }

  EXPECT_EQ(model_executor.Finalize(), SUCCESS);
}

TEST_F(UtestModelExecutorTest, test_release_model_check_hccl_task) {
  {
    auto listener = MakeShared<RunAsyncListener>();
    shared_ptr<DavinciModel> davinci_model = MakeShared<DavinciModel>(1, listener);
    davinci_model->SetId(1);
    davinci_model->has_hccl_task_ = true;
    ModelManager::GetInstance().InsertModel(1, davinci_model);
  }

  ModelExecutor model_executor;
  EXPECT_EQ(model_executor.Initialize({}, 0), SUCCESS);
  model_executor.StartRunThread();

  string graph_name = "test_graph";
  auto compute_graph = MakeShared<ComputeGraph>(graph_name);

  GeRootModelPtr ge_root_model = MakeShared<GeRootModel>();
  EXPECT_EQ(ge_root_model->Initialize(compute_graph), SUCCESS);
  GeModelPtr ge_model = MakeShared<GeModel>();
  ge_model->SetGraph(compute_graph);
  ge_root_model->SetSubgraphInstanceNameToModel(compute_graph->GetName(), ge_model);
  ge_root_model->SetModelId(1);

  GraphNodePtr graph_node = MakeShared<ge::GraphNode>(1);
  graph_node->SetGeRootModel(ge_root_model);
  graph_node->SetLoadFlag(true);
  graph_node->SetAsync(false);

  EXPECT_EQ(model_executor.ReleaseModel(ge_root_model, graph_node), false);

  ModelManager::GetInstance().GetModel(1)->has_hccl_task_ = false;
  EXPECT_EQ(model_executor.ReleaseModel(ge_root_model, graph_node), true);
  EXPECT_EQ(model_executor.Finalize(), SUCCESS);
}

TEST_F(UtestModelExecutorTest, test_unload_model_check_hccl_task) {
  {
    auto listener = MakeShared<RunAsyncListener>();
    shared_ptr<DavinciModel> davinci_model = MakeShared<DavinciModel>(1, listener);
    davinci_model->SetId(1);
    davinci_model->has_hccl_task_ = true;
    ModelManager::GetInstance().InsertModel(1, davinci_model);
  }

  ModelExecutor model_executor;
  EXPECT_EQ(model_executor.Initialize({}, 0), SUCCESS);
  model_executor.StartRunThread();

  string graph_name = "test_graph";
  auto compute_graph = MakeShared<ComputeGraph>(graph_name);
  GeRootModelPtr ge_root_model = MakeShared<GeRootModel>();
  EXPECT_EQ(ge_root_model->Initialize(compute_graph), SUCCESS);
  GeModelPtr ge_model = MakeShared<GeModel>();
  ge_model->SetGraph(compute_graph);
  ge_root_model->SetSubgraphInstanceNameToModel(compute_graph->GetName(), ge_model);

  ge_root_model->SetModelId(1);

  GraphNodePtr graph_node = MakeShared<ge::GraphNode>(1);
  graph_node->SetGeRootModel(ge_root_model);
  graph_node->SetLoadFlag(true);
  graph_node->SetAsync(false);

  EXPECT_EQ(model_executor.ReleaseMemory(ge_root_model, graph_node), false);

  ModelManager::GetInstance().GetModel(1)->has_hccl_task_ = false;
  EXPECT_EQ(model_executor.UnloadPneModel(1, 0, 1), SUCCESS);
  EXPECT_EQ(model_executor.Finalize(), SUCCESS);
}

TEST_F(UtestModelExecutorTest, test_check_and_release_memory) {
  {
    auto listener = MakeShared<RunAsyncListener>();
    shared_ptr<DavinciModel> davinci_model1 = MakeShared<DavinciModel>(1, listener);
    davinci_model1->SetId(1);
    davinci_model1->is_async_mode_ = true;
    ModelManager::GetInstance().InsertModel(1, davinci_model1);
    shared_ptr<DavinciModel> davinci_model2 = MakeShared<DavinciModel>(2, listener);
    davinci_model2->SetId(2);
    davinci_model2->is_async_mode_ = true;
    ModelManager::GetInstance().InsertModel(2, davinci_model2);
  }

  ModelExecutor model_executor;
  EXPECT_EQ(model_executor.Initialize({}, 0), SUCCESS);
  model_executor.StartRunThread();

  GeModelPtr ge_model = std::make_shared<GeModel>();
  int64_t memory_size = 25 * 1024UL * 1024UL * 1024UL;
  int64_t weight_size = 25 * 1024UL * 1024UL * 1024UL;
  uint64_t session_id = 0;
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, ATTR_MODEL_MEMORY_SIZE, memory_size));
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, ATTR_MODEL_WEIGHT_SIZE, weight_size));
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, MODEL_ATTR_SESSION_ID, session_id));

  GraphId graph_id = 1;
  GraphNodePtr graph_node = MakeShared<GraphNode>(graph_id);
  model_executor.AddGraphNode(graph_id, graph_node);

  ComputeGraphPtr compute_graph = MakeShared<ComputeGraph>("test_graph");
  GeRootModelPtr ge_root_model = MakeShared<GeRootModel>();
  EXPECT_EQ(ge_root_model->Initialize(compute_graph), SUCCESS);
  ge_root_model->SetModelId(1);
  ge_root_model->SetModelId(2);
  graph_node->SetGeRootModel(ge_root_model);
  graph_node->SetLoadFlag(true);
  ge_model->model_id_to_session_id_map_[1] = session_id;
  ge_model->model_id_to_session_id_map_[2] = session_id;

  shared_ptr<domi::ModelTaskDef> model_task_def = std::make_shared<domi::ModelTaskDef>();
  ge_model->SetModelTaskDef(model_task_def);

  ge_root_model->SetSubgraphInstanceNameToModel(compute_graph->GetName(), ge_model);

  EXPECT_EQ(model_executor.CheckAndReleaseMemory(ge_root_model, graph_node), SUCCESS);
  EXPECT_EQ(model_executor.Finalize(), SUCCESS);

  (void)ModelManager::GetInstance().DeleteModel(1U);
  (void)ModelManager::GetInstance().DeleteModel(2U);
}

TEST_F(UtestModelExecutorTest, test_check_and_release_memory_with_refreshable_fm) {
  {
    auto listener = MakeShared<RunAsyncListener>();
    shared_ptr<DavinciModel> davinci_model1 = MakeShared<DavinciModel>(1, listener);
    davinci_model1->SetId(1);
    davinci_model1->is_async_mode_ = true;
    ModelManager::GetInstance().InsertModel(1, davinci_model1);
    shared_ptr<DavinciModel> davinci_model2 = MakeShared<DavinciModel>(2, listener);
    davinci_model2->SetId(2);
    davinci_model2->is_async_mode_ = true;
    ModelManager::GetInstance().InsertModel(2, davinci_model2);
  }

  ModelExecutor model_executor;
  EXPECT_EQ(model_executor.Initialize({}, 0), SUCCESS);
  model_executor.StartRunThread();

  GeModelPtr ge_model = std::make_shared<GeModel>();
  int64_t memory_size = 25 * 1024UL * 1024UL * 1024UL;
  int64_t weight_size = 25 * 1024UL * 1024UL * 1024UL;
  uint64_t session_id = 0;
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, ATTR_MODEL_MEMORY_SIZE, memory_size));
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, ATTR_MODEL_WEIGHT_SIZE, weight_size));
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, MODEL_ATTR_SESSION_ID, session_id));

  GraphId graph_id = 1;
  GraphNodePtr graph_node = MakeShared<GraphNode>(graph_id);
  model_executor.AddGraphNode(graph_id, graph_node);

  ComputeGraphPtr compute_graph = MakeShared<ComputeGraph>("test_graph");
  GeRootModelPtr ge_root_model = MakeShared<GeRootModel>();
  EXPECT_EQ(ge_root_model->Initialize(compute_graph), SUCCESS);
  ge_root_model->SetModelId(1);
  ge_root_model->SetModelId(2);
  graph_node->SetGeRootModel(ge_root_model);
  graph_node->SetLoadFlag(true);
  auto memory = malloc(10);
  graph_node->SetRefreshableFeatureMemoryBase(memory, 10);

  ge_model->model_id_to_session_id_map_[1] = session_id;
  ge_model->model_id_to_session_id_map_[2] = session_id;

  shared_ptr<domi::ModelTaskDef> model_task_def = std::make_shared<domi::ModelTaskDef>();
  ge_model->SetModelTaskDef(model_task_def);

  ge_root_model->SetSubgraphInstanceNameToModel(compute_graph->GetName(), ge_model);

  EXPECT_EQ(model_executor.CheckAndReleaseMemory(ge_root_model, graph_node), SUCCESS);
  EXPECT_EQ(model_executor.Finalize(), SUCCESS);

  (void)ModelManager::GetInstance().DeleteModel(1U);
  (void)ModelManager::GetInstance().DeleteModel(2U);
  free(memory);
}

TEST_F(UtestModelExecutorTest, test_check_and_release_memory_with_fix_fm) {
  {
    auto listener = MakeShared<RunAsyncListener>();
    shared_ptr<DavinciModel> davinci_model1 = MakeShared<DavinciModel>(1, listener);
    davinci_model1->SetId(1);
    davinci_model1->is_async_mode_ = true;
    ModelManager::GetInstance().InsertModel(1, davinci_model1);
    shared_ptr<DavinciModel> davinci_model2 = MakeShared<DavinciModel>(2, listener);
    davinci_model2->SetId(2);
    davinci_model2->is_async_mode_ = true;
    ModelManager::GetInstance().InsertModel(2, davinci_model2);
  }

  ModelExecutor model_executor;
  EXPECT_EQ(model_executor.Initialize({}, 0), SUCCESS);
  model_executor.StartRunThread();

  GeModelPtr ge_model = std::make_shared<GeModel>();
  int64_t memory_size = 25 * 1024UL * 1024UL * 1024UL;
  int64_t weight_size = 25 * 1024UL * 1024UL * 1024UL;
  uint64_t session_id = 0;
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, ATTR_MODEL_MEMORY_SIZE, memory_size));
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, ATTR_MODEL_WEIGHT_SIZE, weight_size));
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, MODEL_ATTR_SESSION_ID, session_id));

  GraphId graph_id = 1;
  GraphNodePtr graph_node = MakeShared<GraphNode>(graph_id);
  model_executor.AddGraphNode(graph_id, graph_node);

  ComputeGraphPtr compute_graph = MakeShared<ComputeGraph>("test_graph");
  GeRootModelPtr ge_root_model = MakeShared<GeRootModel>();
  EXPECT_EQ(ge_root_model->Initialize(compute_graph), SUCCESS);
  ge_root_model->SetModelId(1);
  ge_root_model->SetModelId(2);
  auto memory = malloc(10);
  ge_root_model->MutableFixedFeatureMemory().insert(
      {RT_MEMORY_HBM, {RT_MEMORY_HBM, memory, 10, true, false, false, 0U, nullptr}});
  graph_node->SetGeRootModel(ge_root_model);
  graph_node->SetLoadFlag(true);
  ge_model->model_id_to_session_id_map_[1] = session_id;
  ge_model->model_id_to_session_id_map_[2] = session_id;

  shared_ptr<domi::ModelTaskDef> model_task_def = std::make_shared<domi::ModelTaskDef>();
  ge_model->SetModelTaskDef(model_task_def);

  ge_root_model->SetSubgraphInstanceNameToModel(compute_graph->GetName(), ge_model);

  EXPECT_EQ(model_executor.CheckAndReleaseMemory(ge_root_model, graph_node), SUCCESS);
  EXPECT_EQ(model_executor.Finalize(), SUCCESS);

  (void)ModelManager::GetInstance().DeleteModel(1U);
  (void)ModelManager::GetInstance().DeleteModel(2U);
  free(memory);
}

TEST_F(UtestModelExecutorTest, test_flow_model_check_and_release_memory) {
  {
    auto listener = MakeShared<RunAsyncListener>();
    shared_ptr<DavinciModel> davinci_model1 = MakeShared<DavinciModel>(1, listener);
    davinci_model1->SetId(1);
    davinci_model1->is_async_mode_ = true;
    ModelManager::GetInstance().InsertModel(1, davinci_model1);
    shared_ptr<DavinciModel> davinci_model2 = MakeShared<DavinciModel>(2, listener);
    davinci_model2->SetId(2);
    davinci_model2->is_async_mode_ = true;
    ModelManager::GetInstance().InsertModel(2, davinci_model2);
  }

  ModelExecutor model_executor;
  EXPECT_EQ(model_executor.Initialize({}, 0), SUCCESS);
  model_executor.StartRunThread();

  GeModelPtr ge_model = std::make_shared<GeModel>();
  int64_t memory_size = 25 * 1024UL * 1024UL * 1024UL;
  int64_t weight_size = 25 * 1024UL * 1024UL * 1024UL;
  uint64_t session_id = 0;
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, ATTR_MODEL_MEMORY_SIZE, memory_size));
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, ATTR_MODEL_WEIGHT_SIZE, weight_size));
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, MODEL_ATTR_SESSION_ID, session_id));

  GraphId graph_id = 1;
  GraphNodePtr graph_node = MakeShared<GraphNode>(graph_id);
  model_executor.AddGraphNode(graph_id, graph_node);

  ComputeGraphPtr compute_graph = MakeShared<ComputeGraph>("test_graph");
  GeRootModelPtr ge_root_model = MakeShared<GeRootModel>();
  EXPECT_EQ(ge_root_model->Initialize(compute_graph), SUCCESS);
  ge_root_model->SetModelId(1);
  ge_root_model->SetModelId(2);
  graph_node->SetGeRootModel(ge_root_model);
  graph_node->SetLoadFlag(true);
  ge_model->model_id_to_session_id_map_[1] = session_id;
  ge_model->model_id_to_session_id_map_[2] = session_id;

  shared_ptr<domi::ModelTaskDef> model_task_def = std::make_shared<domi::ModelTaskDef>();
  ge_model->SetModelTaskDef(model_task_def);

  ge_root_model->SetSubgraphInstanceNameToModel(compute_graph->GetName(), ge_model);

  EXPECT_EQ(model_executor.CheckAndReleaseMemory(ge_root_model, graph_node), SUCCESS);
  EXPECT_EQ(model_executor.Finalize(), SUCCESS);

  (void)ModelManager::GetInstance().DeleteModel(1U);
  (void)ModelManager::GetInstance().DeleteModel(2U);
}

TEST_F(UtestModelExecutorTest, test_check_and_release_memory_extend_size_static_memory) {
  {
    auto listener = MakeShared<RunAsyncListener>();
    shared_ptr<DavinciModel> davinci_model1 = MakeShared<DavinciModel>(1, listener);
    davinci_model1->SetId(1);
    ModelManager::GetInstance().InsertModel(1, davinci_model1);
    shared_ptr<DavinciModel> davinci_model2 = MakeShared<DavinciModel>(2, listener);
    davinci_model1->SetId(2);
    ModelManager::GetInstance().InsertModel(2, davinci_model2);
  }

  ModelExecutor model_executor;
  EXPECT_EQ(model_executor.Initialize({}, 0), SUCCESS);
  model_executor.StartRunThread();

  GeModelPtr ge_model = std::make_shared<GeModel>();
  int64_t memory_size = 2 * 1024UL;
  int64_t weight_size = 2 * 1024UL;
  int64_t zero_copy_memory_size = 1 * 1024UL;
  uint64_t session_id = 0;
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, ATTR_MODEL_MEMORY_SIZE, memory_size));
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, ATTR_MODEL_WEIGHT_SIZE, weight_size));
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, MODEL_ATTR_SESSION_ID, session_id));
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, ATTR_MODEL_ZERO_COPY_MEMORY_SIZE, zero_copy_memory_size));

  shared_ptr<domi::ModelTaskDef> model_task_def = std::make_shared<domi::ModelTaskDef>();
  ge_model->SetModelTaskDef(model_task_def);

  GraphId graph_id = 1;
  GraphNodePtr graph_node = MakeShared<GraphNode>(graph_id);
  model_executor.AddGraphNode(graph_id, graph_node);

  ComputeGraphPtr compute_graph = MakeShared<ComputeGraph>("test_graph");
  GeRootModelPtr ge_root_model = MakeShared<GeRootModel>();
  EXPECT_EQ(ge_root_model->Initialize(compute_graph), SUCCESS);
  ge_root_model->SetModelId(1);
  ge_root_model->SetModelId(2);
  graph_node->SetGeRootModel(ge_root_model);
  graph_node->SetLoadFlag(true);

  ge_root_model->SetSubgraphInstanceNameToModel(compute_graph->GetName(), ge_model);

  // set extend size static memory option
  std::map<std::string, std::string> graph_options;
  graph_options[STATIC_MEMORY_POLICY] = "2";
  graph_options[OPTION_EXEC_REUSE_ZERO_COPY_MEMORY] = "1";
  GetThreadLocalContext().SetGraphOption(graph_options);

  int64_t sum_size = 0;
  bool is_resuse = false;
  // static memory is not malloced, cannot reuse static memory
  EXPECT_EQ(model_executor.GetMemorySizeAfterReuse({ge_model}, graph_node, sum_size, is_resuse), SUCCESS);
  EXPECT_EQ(sum_size, memory_size - zero_copy_memory_size + weight_size);
  EXPECT_FALSE(is_resuse);

  // malloc static memory, reuse static memory
  auto mem_instance = SessionMemAllocator<ActiveMemoryAllocator>::Instance().GetMemAllocator(0, 0);
  LogicalMemorys logical_memorys;
  logical_memorys.emplace_back(0, memory_size);
  std::vector<std::pair<uint8_t *, size_t>> mem_size;
  (void)mem_instance->MallocMemory("", logical_memorys, mem_size, 0);
  EXPECT_EQ(model_executor.GetMemorySizeAfterReuse({ge_model}, graph_node, sum_size, is_resuse), SUCCESS);
  EXPECT_EQ(sum_size, weight_size);
  EXPECT_TRUE(is_resuse);

  EXPECT_EQ(model_executor.CheckAndReleaseMemory(ge_root_model, graph_node), SUCCESS);
  EXPECT_EQ(model_executor.Finalize(), SUCCESS);

  (void)mem_instance->FreeMemory(0);
  EXPECT_EQ(ModelManager::GetInstance().DeleteModel(1U), SUCCESS);
  EXPECT_EQ(ModelManager::GetInstance().DeleteModel(2U), SUCCESS);
  graph_options[STATIC_MEMORY_POLICY] = "";
  graph_options[OPTION_EXEC_REUSE_ZERO_COPY_MEMORY] = "";
  GetThreadLocalContext().SetGraphOption(graph_options);
}

TEST_F(UtestModelExecutorTest, test_run_thread) {
  std::map<std::string, std::string> graph_options;
  graph_options[STATIC_MEMORY_POLICY] = "4";
  GetThreadLocalContext().SetGraphOption(graph_options);
  ModelExecutor model_executor;
  EXPECT_EQ(model_executor.Initialize({}, 0), SUCCESS);
  model_executor.StartRunThread();

  GraphId graph_id = 1;
  uint64_t session_id = 0;
  error_message::ErrorManagerContext error_context;
  GEThreadLocalContext context = GetThreadLocalContext();
  const auto callback = [](Status status, std::vector<gert::Tensor> &outputs) {};

  auto compute_graph = MakeShared<ComputeGraph>("test_graph");
  GeRootModelPtr ge_root_model = MakeShared<GeRootModel>();
  EXPECT_EQ(ge_root_model->Initialize(compute_graph), SUCCESS);
  ge_root_model->SetCustomOpRegistry(CustomOpFactory::GetGlobalRegistryPtr());

  GeModelPtr ge_model = MakeShared<GeModel>();
  ge_model->SetGraph(compute_graph);

  shared_ptr<domi::ModelTaskDef> model_task_def = std::make_shared<domi::ModelTaskDef>();
  ge_model->SetModelTaskDef(model_task_def);

  ge_root_model->SetSubgraphInstanceNameToModel(compute_graph->GetName(), ge_model);

  GraphNodePtr graph_node = MakeShared<ge::GraphNode>(graph_id);
  graph_node->SetGeRootModel(ge_root_model);
  graph_node->SetLoadFlag(false);
  graph_node->SetAsync(false);
  graph_node->IncreaseLoadCount();
  graph_node->Lock();

  std::vector<gert::Tensor> input_tensors(1);

  std::shared_ptr<RunArgs> run_args;
  run_args = std::make_shared<RunArgs>();
  ASSERT_TRUE(run_args != nullptr);
  run_args->graph_node = graph_node;
  run_args->graph_id = graph_id;
  run_args->session_id = session_id;
  run_args->error_context = error_context;
  run_args->input_tensor = std::move(input_tensors);
  run_args->context = context;
  run_args->callback = callback;
  EXPECT_EQ(model_executor.PushRunArgs(run_args), SUCCESS);

  while (model_executor.run_args_q_.Size() > 0) {
    usleep(10);  // 0.01ms, Wait for RunThread.
  }
  EXPECT_EQ(model_executor.Finalize(), SUCCESS);
  EXPECT_EQ(ModelManager::GetInstance().DeleteModel(ge_root_model->GetModelId()), SUCCESS);
  graph_options[STATIC_MEMORY_POLICY] = "";
  GetThreadLocalContext().SetGraphOption(graph_options);
}

TEST_F(UtestModelExecutorTest, test_run_thread_2) {
  GraphId graph_id = 1;
  uint64_t session_id = 0;
  error_message::ErrorManagerContext error_context;
  GEThreadLocalContext context;

  auto compute_graph = MakeShared<ComputeGraph>("test_graph");
  GeRootModelPtr ge_root_model = MakeShared<GeRootModel>();
  EXPECT_EQ(ge_root_model->Initialize(compute_graph), SUCCESS);
  ge_root_model->SetCustomOpRegistry(CustomOpFactory::GetGlobalRegistryPtr());

  GeModelPtr ge_model = MakeShared<GeModel>();
  ge_model->SetGraph(compute_graph);

  shared_ptr<domi::ModelTaskDef> model_task_def = std::make_shared<domi::ModelTaskDef>();
  ge_model->SetModelTaskDef(model_task_def);

  ge_root_model->SetSubgraphInstanceNameToModel(compute_graph->GetName(), ge_model);

  GraphNodePtr graph_node = MakeShared<ge::GraphNode>(graph_id);
  graph_node->SetGeRootModel(ge_root_model);
  graph_node->SetLoadFlag(false);
  graph_node->SetAsync(true);
  graph_node->IncreaseLoadCount();

  std::vector<gert::Tensor> input_tensors(1);

  std::shared_ptr<RunArgs> run_args;
  run_args = std::make_shared<RunArgs>();
  ASSERT_TRUE(run_args != nullptr);
  run_args->graph_node = graph_node;
  run_args->graph_id = graph_id;
  run_args->session_id = session_id;
  run_args->error_context = error_context;
  run_args->input_tensor = std::move(input_tensors);
  run_args->context = context;

  {
    ModelExecutor model_executor;
    EXPECT_EQ(model_executor.Initialize({}, session_id), SUCCESS);
    // Callback for execute.
    std::mutex run_mutex;
    std::condition_variable model_run_cv;
    Status run_status = FAILED;
    const RunAsyncCallbackV2 callback = [&](Status status, std::vector<gert::Tensor> &outputs) {
      std::unique_lock<std::mutex> lock(run_mutex);
      run_status = status;
      model_run_cv.notify_one();
    };
    run_args->callback = callback;
    graph_node->Lock();
    EXPECT_EQ(model_executor.PushRunArgs(run_args), SUCCESS);
    model_executor.StartRunThread();
    std::unique_lock<std::mutex> lock(run_mutex);
    EXPECT_EQ(model_run_cv.wait_for(lock, std::chrono::seconds(10)), std::cv_status::no_timeout);
    EXPECT_EQ(run_status, SUCCESS);
    EXPECT_EQ(model_executor.Finalize(), SUCCESS);
  }

  {
    graph_node->SetGeRootModel(nullptr);
    ModelExecutor model_executor;
    EXPECT_EQ(model_executor.Initialize({}, session_id), SUCCESS);
    // Callback for execute.
    std::condition_variable model_run_cv;
    Status run_status = FAILED;
    bool call_flag = false;
    size_t sleep_time_max = 5U;
    size_t sleep_time = 0U;
    const RunAsyncCallbackV2 callback = [&](Status status, std::vector<gert::Tensor> &outputs) {
      run_status = status;
      model_run_cv.notify_one();
      call_flag = true;
    };
    run_args->callback = callback;
    graph_node->Lock();
    EXPECT_EQ(model_executor.PushRunArgs(run_args), SUCCESS);
    model_executor.StartRunThread();
    while (!call_flag) {
      sleep(1);
      if (++sleep_time >= sleep_time_max) {
        break;
      }
    }
    EXPECT_EQ(run_status, PARAM_INVALID);
    EXPECT_EQ(model_executor.Finalize(), SUCCESS);
  }

  graph_node->SetGeRootModel(ge_root_model);
  {
    ModelExecutor model_executor;
    EXPECT_EQ(model_executor.Initialize({}, session_id), SUCCESS);
    // Callback for execute.
    std::mutex run_mutex;
    std::condition_variable model_run_cv;
    Status run_status = FAILED;
    const RunAsyncCallbackV2 callback = [&](Status status, std::vector<gert::Tensor> &outputs) {
      std::unique_lock<std::mutex> lock(run_mutex);
      run_status = status;
      model_run_cv.notify_one();
    };
    run_args->callback = callback;
    graph_node->SetLoadFlag(false);
    graph_node->SetLoadCount(0);
    graph_node->SetLoadRecord(0);
    graph_node->Lock();
    ModelManager::GetInstance().DeleteModel(ge_root_model->GetModelId());
    EXPECT_EQ(model_executor.PushRunArgs(run_args), SUCCESS);
    model_executor.StartRunThread();
    std::unique_lock<std::mutex> lock(run_mutex);
    EXPECT_EQ(model_run_cv.wait_for(lock, std::chrono::seconds(10)), std::cv_status::no_timeout);
    EXPECT_NE(run_status, SUCCESS);
    EXPECT_EQ(model_executor.Finalize(), SUCCESS);
  }
  ModelManager::GetInstance().DeleteModel(ge_root_model->GetModelId());
}

TEST_F(UtestModelExecutorTest, test_run_thread_3) {
  GraphId graph_id = 1;
  uint64_t session_id = 0;
  error_message::ErrorManagerContext error_context;
  GEThreadLocalContext context;

  auto compute_graph = CreateGraphWithConstOutput();
  GeRootModelPtr ge_root_model = MakeShared<GeRootModel>();
  EXPECT_EQ(ge_root_model->Initialize(compute_graph), SUCCESS);
  ge_root_model->SetModelId(1);

  GeModelPtr ge_model = MakeShared<GeModel>();
  ge_model->SetGraph(compute_graph);

  shared_ptr<domi::ModelTaskDef> model_task_def = std::make_shared<domi::ModelTaskDef>();
  ge_model->SetModelTaskDef(model_task_def);

  ge_root_model->SetSubgraphInstanceNameToModel(compute_graph->GetName(), ge_model);

  GraphNodePtr graph_node = MakeShared<ge::GraphNode>(graph_id);
  graph_node->SetGeRootModel(ge_root_model);
  graph_node->SetLoadFlag(false);
  graph_node->SetAsync(true);
  graph_node->SetLoadCount(0);
  graph_node->SetLoadRecord(0);

  std::vector<gert::Tensor> input_tensors(1);

  std::shared_ptr<RunArgs> run_args;
  run_args = std::make_shared<RunArgs>();
  ASSERT_TRUE(run_args != nullptr);
  run_args->graph_node = graph_node;
  run_args->graph_id = graph_id;
  run_args->session_id = session_id;
  run_args->error_context = error_context;
  run_args->input_tensor = std::move(input_tensors);
  run_args->context = context;

  {
    ModelExecutor model_executor;
    EXPECT_EQ(model_executor.Initialize({}, session_id), SUCCESS);
    // Callback for execute.
    std::mutex run_mutex;
    std::condition_variable model_run_cv;
    Status run_status = FAILED;
    const RunAsyncCallbackV2 callback = [&](Status status, std::vector<gert::Tensor> &outputs) {
      std::unique_lock<std::mutex> lock(run_mutex);
      run_status = status;
      model_run_cv.notify_one();
    };
    run_args->callback = callback;
    graph_node->Lock();
    auto hybrid_model_ptr = ge::hybrid::HybridDavinciModel::Create(ge_root_model);
    auto shared_model = std::shared_ptr<hybrid::HybridDavinciModel>(hybrid_model_ptr.release());
    ModelManager::GetInstance().InsertModel(ge_root_model->GetModelId(), shared_model);
    shared_model->SetModelId(1);
    shared_model->Init();
    EXPECT_EQ(model_executor.PushRunArgs(run_args), SUCCESS);
    model_executor.StartRunThread();
    std::unique_lock<std::mutex> lock(run_mutex);
    EXPECT_EQ(model_run_cv.wait_for(lock, std::chrono::seconds(10)), std::cv_status::no_timeout);
    EXPECT_NE(run_status, SUCCESS);
    EXPECT_EQ(ModelManager::GetInstance().DeleteModel(ge_root_model->GetModelId()), SUCCESS);
    EXPECT_EQ(model_executor.Finalize(), SUCCESS);
  }
}

TEST_F(UtestModelExecutorTest, test_run_thread_4) {
  GraphId graph_id = 1;
  uint64_t session_id = 0;
  error_message::ErrorManagerContext error_context;
  GEThreadLocalContext context;

  auto compute_graph = CreateGraphWithConstOutput();
  GeRootModelPtr ge_root_model = MakeShared<GeRootModel>();
  EXPECT_EQ(ge_root_model->Initialize(compute_graph), SUCCESS);
  ge_root_model->SetModelId(1);

  GeModelPtr ge_model = MakeShared<GeModel>();
  ge_model->SetGraph(compute_graph);

  shared_ptr<domi::ModelTaskDef> model_task_def = std::make_shared<domi::ModelTaskDef>();
  ge_model->SetModelTaskDef(model_task_def);

  ge_root_model->SetSubgraphInstanceNameToModel(compute_graph->GetName(), ge_model);

  GraphNodePtr graph_node = MakeShared<ge::GraphNode>(graph_id);
  graph_node->SetGeRootModel(ge_root_model);
  graph_node->SetLoadFlag(false);
  graph_node->SetAsync(true);
  graph_node->SetLoadCount(0);
  graph_node->SetLoadRecord(0);

  std::vector<gert::Tensor> input_tensors(1);

  std::shared_ptr<RunArgs> run_args;
  run_args = std::make_shared<RunArgs>();
  ASSERT_TRUE(run_args != nullptr);
  run_args->graph_node = graph_node;
  run_args->graph_id = graph_id;
  run_args->session_id = session_id;
  run_args->error_context = error_context;
  run_args->input_tensor = std::move(input_tensors);
  run_args->context = context;

  {
    ModelExecutor model_executor;
    EXPECT_EQ(model_executor.Initialize({}, session_id), SUCCESS);
    // Callback for execute.
    const RunAsyncCallbackV2 callback = [&](Status status, std::vector<gert::Tensor> &outputs) {};
    run_args->callback = callback;
    graph_node->Lock();
    auto hybrid_model_ptr = ge::hybrid::HybridDavinciModel::Create(ge_root_model);
    auto shared_model = std::shared_ptr<hybrid::HybridDavinciModel>(hybrid_model_ptr.release());
    ModelManager::GetInstance().InsertModel(ge_root_model->GetModelId(), shared_model);
    shared_model->SetModelId(1);
    shared_model->Init();
    shared_ptr<RunAsyncListener> listerner_ptr = MakeShared<RunAsyncListener>();
    shared_model->SetListener(listerner_ptr);
    EXPECT_EQ(model_executor.PushRunArgs(run_args), SUCCESS);
    model_executor.StartRunThread();
    sleep(1);  // wait for thread
    ASSERT_NE(listerner_ptr->sem_.Size() + listerner_ptr->sem_v2_.Size(), 0);
    EXPECT_EQ(ModelManager::GetInstance().DeleteModel(ge_root_model->GetModelId()), SUCCESS);
    EXPECT_EQ(model_executor.Finalize(), SUCCESS);
  }
}

static void test_run_graph(ModelExecutor &model_executor) {
  auto compute_graph = MakeShared<ComputeGraph>("test_graph");
  GeRootModelPtr ge_root_model = MakeShared<GeRootModel>();
  EXPECT_EQ(ge_root_model->Initialize(compute_graph), SUCCESS);
  ge_root_model->SetCustomOpRegistry(CustomOpFactory::GetGlobalRegistryPtr());
  GeModelPtr ge_model = MakeShared<GeModel>();
  ge_model->SetGraph(compute_graph);
  shared_ptr<domi::ModelTaskDef> model_task_def = std::make_shared<domi::ModelTaskDef>();
  ge_model->SetModelTaskDef(model_task_def);
  ge_root_model->SetSubgraphInstanceNameToModel(compute_graph->GetName(), ge_model);

  GraphId graph_id = 1;
  GraphNodePtr graph_node = MakeShared<ge::GraphNode>(graph_id);
  graph_node->SetGeRootModel(ge_root_model);
  graph_node->SetLoadFlag(false);
  graph_node->SetAsync(false);  // RunGraph is Synchronization.
  EXPECT_EQ(model_executor.LoadGraph(ge_root_model, graph_node), SUCCESS);

  std::vector<gert::Tensor> inputs;
  std::vector<gert::Tensor> outputs;
  EXPECT_EQ(model_executor.RunGraph(graph_node, graph_id, inputs, outputs), SUCCESS);
  EXPECT_EQ(ModelManager::GetInstance().DeleteModel(ge_root_model->GetModelId()), SUCCESS);
}

TEST_F(UtestModelExecutorTest, test_run_graph_train) {
  GetThreadLocalContext().SetGlobalOption({{OPTION_GRAPH_RUN_MODE, "1"}});
  ModelExecutor model_executor;
  EXPECT_EQ(model_executor.Initialize({}, 0), SUCCESS);
  model_executor.StartRunThread();
  test_run_graph(model_executor);
  EXPECT_EQ(model_executor.Finalize(), SUCCESS);
}

TEST_F(UtestModelExecutorTest, test_run_graph_infer) {
  GetThreadLocalContext().SetGlobalOption({});
  GetThreadLocalContext().SetSessionOption({});
  GetThreadLocalContext().SetGraphOption({});
  ModelExecutor model_executor;
  EXPECT_EQ(model_executor.Initialize({}, 0), SUCCESS);
  model_executor.StartRunThread();
  test_run_graph(model_executor);
  EXPECT_EQ(model_executor.Finalize(), SUCCESS);
}

TEST_F(UtestModelExecutorTest, test_run_graph_with_stream) {
  ModelExecutor model_executor;
  EXPECT_EQ(model_executor.Initialize({}, 0), SUCCESS);
  model_executor.StartRunThread();

  GraphId graph_id = 1;
  auto compute_graph = CreateGraphWithConstOutput();
  GeRootModelPtr ge_root_model = MakeShared<GeRootModel>();
  EXPECT_EQ(ge_root_model->Initialize(compute_graph), SUCCESS);

  GeModelPtr ge_model = MakeShared<GeModel>();
  ge_model->SetGraph(compute_graph);

  shared_ptr<domi::ModelTaskDef> model_task_def = std::make_shared<domi::ModelTaskDef>();
  ge_model->SetModelTaskDef(model_task_def);
  ge_root_model->SetSubgraphInstanceNameToModel(compute_graph->GetName(), ge_model);

  GraphNodePtr graph_node = MakeShared<ge::GraphNode>(graph_id);
  graph_node->SetGeRootModel(ge_root_model);
  graph_node->SetComputeGraph(compute_graph);
  graph_node->SetLoadFlag(false);
  graph_node->SetAsync(true);

  GeTensor tensor;
  std::vector<GeTensor> inputs{tensor};
  std::vector<GeTensor> outputs;

  rtStream_t stream = nullptr;
  aclrtCreateStreamWithConfig(&stream, 0, 0);
  EXPECT_EQ(model_executor.RunGraphWithStream(graph_node, graph_id, stream, inputs, outputs), 1343225857);

  EXPECT_EQ(model_executor.Finalize(), SUCCESS);

  aclrtDestroyStream(stream);
}

TEST_F(UtestModelExecutorTest, test_execute_graph_with_stream) {
  ModelExecutor model_executor;
  EXPECT_EQ(model_executor.Initialize({}, 0), SUCCESS);
  model_executor.StartRunThread();

  GraphId graph_id = 1;
  auto compute_graph = CreateGraphWithConstOutput();
  GeRootModelPtr ge_root_model = MakeShared<GeRootModel>();
  EXPECT_EQ(ge_root_model->Initialize(compute_graph), SUCCESS);

  GeModelPtr ge_model = MakeShared<GeModel>();
  ge_model->SetGraph(compute_graph);

  shared_ptr<domi::ModelTaskDef> model_task_def = std::make_shared<domi::ModelTaskDef>();
  ge_model->SetModelTaskDef(model_task_def);
  ge_root_model->SetSubgraphInstanceNameToModel(compute_graph->GetName(), ge_model);

  GraphNodePtr graph_node = MakeShared<ge::GraphNode>(graph_id);
  graph_node->SetGeRootModel(ge_root_model);
  graph_node->SetComputeGraph(compute_graph);
  graph_node->SetLoadFlag(false);
  graph_node->SetAsync(true);

  std::vector<gert::Tensor> gert_inputs;
  gert_inputs.resize(1);
  std::vector<int32_t> input_data_1(1 * 2 * 3 * 4, 0);
  gert_inputs[0] = {{{1, 2, 3, 4}, {1, 2, 3, 4}},                // shape
                    {ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, {}},  // format
                    gert::kOnDeviceHbm,                          // placement
                    ge::DT_INT32,                                // data type
                    (void *)input_data_1.data()};
  std::vector<gert::Tensor> gert_outputs;

  rtStream_t stream = nullptr;
  aclrtCreateStreamWithConfig(&stream, 0, 0);
  GeTensor tensor;
  std::vector<GeTensor> inputs{tensor};
  std::vector<GeTensor> outputs;
  EXPECT_EQ(model_executor.RunGraphWithStream(graph_node, graph_id, stream, inputs, outputs), PARAM_INVALID);
  EXPECT_EQ(model_executor.ExecuteGraphWithStream(graph_node, graph_id, stream, gert_inputs, gert_outputs),
            ACL_ERROR_GE_EXEC_MODEL_ID_INVALID);
  GraphNodePtr graph_node_2 = MakeShared<ge::GraphNode>(2);
  EXPECT_NE(model_executor.ExecuteGraphWithStream(graph_node_2, 2, stream, gert_inputs, gert_outputs), SUCCESS);
  EXPECT_EQ(model_executor.Finalize(), SUCCESS);
  aclrtDestroyStream(stream);
}

static bool is_err_cb_called = false;
static void err_cb_stub(Status sta, std::vector<gert::Tensor> &tens) {
  is_err_cb_called = true;
}

TEST_F(UtestModelExecutorTest, ReturnError) {
  RunAsyncCallbackV2 callback = err_cb_stub;
  Status ret = 0;
  string log_info = string("err log info");

  ModelExecutor model_executor;
  model_executor.ReturnError(callback, ret, log_info);
  EXPECT_TRUE(is_err_cb_called);
}

TEST_F(UtestModelExecutorTest, UpdateFeatureMemoryBase_UpdateOk) {
  auto listener = MakeShared<RunAsyncListener>();
  shared_ptr<DavinciModel> davinci_model = MakeShared<DavinciModel>(1, listener);
  davinci_model->SetId(1);
  davinci_model->is_async_mode_ = true;
  davinci_model->runtime_param_.fm_memory_infos.resize(1U);
  davinci_model->runtime_param_.fm_memory_infos[0U].memory_size = 4;
  ModelManager::GetInstance().InsertModel(1, davinci_model);

  ModelExecutor model_executor;
  EXPECT_EQ(model_executor.Initialize({}, 0), SUCCESS);
  GeModelPtr ge_model = std::make_shared<GeModel>();
  shared_ptr<domi::ModelTaskDef> model_task_def = std::make_shared<domi::ModelTaskDef>();
  ge_model->SetModelTaskDef(model_task_def);

  GraphId graph_id = 1;
  GraphNodePtr graph_node = MakeShared<GraphNode>(graph_id);
  model_executor.AddGraphNode(graph_id, graph_node);

  ComputeGraphPtr compute_graph = MakeShared<ComputeGraph>("test_graph");
  GeRootModelPtr ge_root_model = MakeShared<GeRootModel>();
  EXPECT_EQ(ge_root_model->Initialize(compute_graph), SUCCESS);
  ge_root_model->SetModelId(1);
  graph_node->SetGeRootModel(ge_root_model);
  graph_node->SetLoadFlag(true);
  ge_root_model->SetSubgraphInstanceNameToModel(compute_graph->GetName(), ge_model);

  EXPECT_EQ(model_executor.UpdateFeatureMemoryBase(graph_node, 0x1234, 4), SUCCESS);
  EXPECT_EQ(davinci_model->mem_base_, 0x1234);
  EXPECT_EQ(reinterpret_cast<uintptr_t>(davinci_model->runtime_param_.fm_memory_infos[0U].memory_base), 0x1234);
  EXPECT_EQ(davinci_model->mem_base_size_, 4);
  (void)ModelManager::GetInstance().DeleteModel(1U);
}

TEST_F(UtestModelExecutorTest, MallocAndFreeFixedFeatureMemoryIfNeed_Success_WhenUserNotSetHbmFixedMem) {
  GraphManager graph_manager;
  GeModelPtr ge_model;
  GraphNodePtr graph_node;
  GeRootModelPtr ge_root_model;
  GraphId graph_id = 1;
  CreateSummaryCompiledModel(graph_node, ge_model, ge_root_model);
  graph_manager.AddGraphNode(graph_id, graph_node);
  graph_node->SetBuildFlag(true);
  graph_node->SetCompiledFlag(true);

  ModelExecutor model_executor;
  model_executor.AddGraphNode(graph_id, graph_node);
  EXPECT_EQ(model_executor.Initialize({}, 0), SUCCESS);

  graph_manager.executor_ = &model_executor;
  CompiledGraphSummaryPtr summary = nullptr;
  EXPECT_EQ(graph_manager.GetCompiledGraphSummary(graph_id, summary), SUCCESS);
  ASSERT_NE(summary, nullptr);

  EXPECT_TRUE(ge_root_model->GetFixedFeatureMemory().empty());
  EXPECT_EQ(model_executor.MallocFixedFeatureMemoryIfNeed(graph_node, ge_root_model, nullptr), SUCCESS);
  auto all_fixed_feature_mem = ge_root_model->GetFixedFeatureMemory();
  ASSERT_EQ(all_fixed_feature_mem.size(), 2U);

  const auto hbm_iter = all_fixed_feature_mem.find(RT_MEMORY_HBM);
  ASSERT_NE(hbm_iter, all_fixed_feature_mem.end());
  EXPECT_NE(hbm_iter->second.addr, nullptr);
  EXPECT_EQ(hbm_iter->second.block, nullptr);
  EXPECT_EQ(hbm_iter->second.size, 8);
  EXPECT_TRUE(hbm_iter->second.ge_alloc);
  EXPECT_FALSE(hbm_iter->second.user_alloc);

  const auto p2p_iter = all_fixed_feature_mem.find(RT_MEMORY_P2P_DDR);
  ASSERT_NE(p2p_iter, all_fixed_feature_mem.cend());
  EXPECT_NE(p2p_iter->second.addr, nullptr);
  EXPECT_EQ(p2p_iter->second.block, nullptr);
  EXPECT_EQ(p2p_iter->second.size, 1024);
  EXPECT_TRUE(p2p_iter->second.ge_alloc);
  EXPECT_FALSE(p2p_iter->second.user_alloc);

  EXPECT_EQ(model_executor.FreeFixedFeatureMemoryIfNeed(ge_root_model), SUCCESS);
  EXPECT_TRUE(ge_root_model->GetFixedFeatureMemory().empty());
}

TEST_F(UtestModelExecutorTest, MallocAndFreeFixedFeatureMemoryIfNeed_Success_UseFixedBaseExpandableAllocator) {
  mmSetEnv("GE_USE_STATIC_MEMORY", "4", 1);
  GraphManager graph_manager;
  GeModelPtr ge_model;
  GraphNodePtr graph_node;
  GeRootModelPtr ge_root_model;
  GraphId graph_id = 1;
  CreateSummaryCompiledModel(graph_node, ge_model, ge_root_model);
  graph_manager.AddGraphNode(graph_id, graph_node);
  graph_node->SetBuildFlag(true);
  graph_node->SetCompiledFlag(true);

  ModelExecutor model_executor;
  uint64_t session_id = 12091530;
  model_executor.AddGraphNode(graph_id, graph_node);
  EXPECT_EQ(model_executor.Initialize({}, session_id), SUCCESS);

  graph_manager.executor_ = &model_executor;
  CompiledGraphSummaryPtr summary = nullptr;
  EXPECT_EQ(graph_manager.GetCompiledGraphSummary(graph_id, summary), SUCCESS);
  ASSERT_NE(summary, nullptr);

  EXPECT_TRUE(ge_root_model->GetFixedFeatureMemory().empty());

  EXPECT_EQ(model_executor.MallocFixedFeatureMemoryIfNeed(graph_node, ge_root_model, nullptr), SUCCESS);
  auto all_fixed_feature_mem = ge_root_model->GetFixedFeatureMemory();
  ASSERT_EQ(all_fixed_feature_mem.size(), 2U);

  const auto hbm_iter = all_fixed_feature_mem.find(RT_MEMORY_HBM);
  ASSERT_NE(hbm_iter, all_fixed_feature_mem.end());
  EXPECT_NE(hbm_iter->second.addr, nullptr);
  EXPECT_NE(hbm_iter->second.block, nullptr);
  EXPECT_EQ(hbm_iter->second.size, 8);
  EXPECT_TRUE(hbm_iter->second.ge_alloc);
  EXPECT_FALSE(hbm_iter->second.user_alloc);

  const auto p2p_iter = all_fixed_feature_mem.find(RT_MEMORY_P2P_DDR);
  ASSERT_NE(p2p_iter, all_fixed_feature_mem.cend());
  EXPECT_NE(p2p_iter->second.addr, nullptr);
  EXPECT_NE(p2p_iter->second.block, nullptr);
  EXPECT_EQ(p2p_iter->second.size, 1024);
  EXPECT_TRUE(p2p_iter->second.ge_alloc);
  EXPECT_FALSE(p2p_iter->second.user_alloc);

  auto hbm_session_allocator = SessionMemAllocator<FixedBaseExpandableAllocator>::Instance().GetMemAllocator(
      session_id, GetContext().DeviceId(), RT_MEMORY_HBM);
  auto p2p_session_allocator = SessionMemAllocator<FixedBaseExpandableAllocator>::Instance().GetMemAllocator(
      session_id, GetContext().DeviceId(), RT_MEMORY_P2P_DDR);

  auto hbm_mem_block = hbm_session_allocator->Malloc(2048);
  ASSERT_NE(hbm_mem_block, nullptr);
  auto p2p_mem_block = p2p_session_allocator->Malloc(2048);
  ASSERT_NE(p2p_mem_block, nullptr);

  ASSERT_EQ(hbm_mem_block->GetAddr(), hbm_iter->second.addr);
  ASSERT_EQ(p2p_mem_block->GetAddr(), p2p_iter->second.addr);

  hbm_mem_block->Free();
  p2p_mem_block->Free();

  EXPECT_EQ(model_executor.FreeFixedFeatureMemoryIfNeed(ge_root_model), SUCCESS);
  EXPECT_TRUE(ge_root_model->GetFixedFeatureMemory().empty());
  mmSetEnv("GE_USE_STATIC_MEMORY", "0", 1);
}

TEST_F(UtestModelExecutorTest, FreeFixedFeatureMemory_WhenLoadFailed) {
  ModelExecutor model_executor;
  EXPECT_EQ(model_executor.Initialize({}, 0), SUCCESS);
  model_executor.StartRunThread();

  Graph graph("test_graph");
  auto compute_graph = MakeShared<ComputeGraph>("test_graph");
  compute_graph->SetGraphUnknownFlag(true);
  GeRootModelPtr ge_root_model = MakeShared<GeRootModel>();
  EXPECT_EQ(ge_root_model->Initialize(compute_graph), SUCCESS);

  GraphId graph_id = 1;
  GraphNodePtr graph_node = MakeShared<ge::GraphNode>(graph_id);
  graph_node->SetGeRootModel(ge_root_model);
  graph_node->SetLoadFlag(true);
  graph_node->SetAsync(true);

  const std::string purpose = MemTypeUtils::ToString(RT_MEMORY_HBM) + " fixed feature base";
  auto &mem_instance = MemManager::Instance().MemInstance(RT_MEMORY_HBM);
  auto addr = mem_instance.MallocMemory(purpose, 8 * 1024 * 1024, GetContext().DeviceId());
  ASSERT_NE(addr, nullptr);
  (void)ge_root_model->MutableFixedFeatureMemory().insert(
      {RT_MEMORY_HBM, {RT_MEMORY_HBM, addr, 8 * 1024 * 1024, false, true, false, 0, nullptr}});

  // GeModel is null, DavinciModel::Assign will return FAILED
  EXPECT_EQ(model_executor.LoadGraph(ge_root_model, graph_node), PARAM_INVALID);  // GeModel is null

  // 加载失败后，这里的内存被释放
  EXPECT_TRUE(ge_root_model->MutableFixedFeatureMemory().empty());
  EXPECT_EQ(model_executor.UnloadGraph(ge_root_model, graph_id), SUCCESS);

  EXPECT_EQ(model_executor.Finalize(), SUCCESS);
  unsetenv(kEnvGeuseStaticMemory.c_str());
}

TEST_F(UtestModelExecutorTest, MallocAndFreeFixedFeatureMemoryIfNeed_Success_WhenUserSetHbmFixedMem) {
  GraphManager graph_manager;
  GeModelPtr ge_model;
  GraphNodePtr graph_node;
  GeRootModelPtr ge_root_model;
  GraphId graph_id = 1;
  CreateSummaryCompiledModel(graph_node, ge_model, ge_root_model);
  graph_manager.AddGraphNode(graph_id, graph_node);
  graph_node->SetBuildFlag(true);
  graph_node->SetCompiledFlag(true);

  ModelExecutor model_executor;
  model_executor.AddGraphNode(graph_id, graph_node);
  EXPECT_EQ(model_executor.Initialize({}, 0), SUCCESS);

  graph_manager.executor_ = &model_executor;
  CompiledGraphSummaryPtr summary = nullptr;
  EXPECT_EQ(graph_manager.GetCompiledGraphSummary(graph_id, summary), SUCCESS);
  ASSERT_NE(summary, nullptr);

  EXPECT_TRUE(ge_root_model->GetFixedFeatureMemory().empty());
  graph_manager.SetFixedFeatureMemoryBase(graph_id, MemoryType::MEMORY_TYPE_DEFAULT, (void *)0x1234, 12U);
  EXPECT_EQ(ge_root_model->GetFixedFeatureMemory().size(), 1U);

  EXPECT_EQ(model_executor.MallocFixedFeatureMemoryIfNeed(graph_node, ge_root_model, nullptr), SUCCESS);
  auto all_fixed_feature_mem = ge_root_model->GetFixedFeatureMemory();
  ASSERT_EQ(all_fixed_feature_mem.size(), 2U);

  const auto hbm_iter = all_fixed_feature_mem.find(RT_MEMORY_HBM);
  ASSERT_NE(hbm_iter, all_fixed_feature_mem.end());
  EXPECT_NE(hbm_iter->second.addr, nullptr);
  EXPECT_EQ(hbm_iter->second.block, nullptr);
  EXPECT_EQ(hbm_iter->second.size, 12);
  EXPECT_FALSE(hbm_iter->second.ge_alloc);  // ge_alloc is false
  EXPECT_TRUE(hbm_iter->second.user_alloc);

  const auto p2p_iter = all_fixed_feature_mem.find(RT_MEMORY_P2P_DDR);
  ASSERT_NE(p2p_iter, all_fixed_feature_mem.cend());
  EXPECT_NE(p2p_iter->second.addr, nullptr);
  EXPECT_EQ(p2p_iter->second.block, nullptr);
  EXPECT_EQ(p2p_iter->second.size, 1024);
  EXPECT_TRUE(p2p_iter->second.ge_alloc);
  EXPECT_FALSE(p2p_iter->second.user_alloc);

  EXPECT_EQ(model_executor.FreeFixedFeatureMemoryIfNeed(ge_root_model), SUCCESS);
  EXPECT_EQ(ge_root_model->GetFixedFeatureMemory().size(), 1U);
}

TEST_F(UtestModelExecutorTest, MallocAndFreeFixedFeatureMemoryIfNeed_ExternalAllocator_Success) {
  std::shared_ptr<Allocator> external_allocator = MakeShared<ExternalAllocatorUtStub>();
  ExternalAllocatorManager::SetExternalAllocator(rtStream_t(0x1), external_allocator);
  GraphManager graph_manager;
  GeModelPtr ge_model;
  GraphNodePtr graph_node;
  GeRootModelPtr ge_root_model;
  GraphId graph_id = 1;
  CreateSummaryCompiledModel(graph_node, ge_model, ge_root_model);
  graph_manager.AddGraphNode(graph_id, graph_node);
  graph_node->SetBuildFlag(true);
  graph_node->SetCompiledFlag(true);

  ModelExecutor model_executor;
  model_executor.AddGraphNode(graph_id, graph_node);
  EXPECT_EQ(model_executor.Initialize({}, 0), SUCCESS);

  graph_manager.executor_ = &model_executor;
  CompiledGraphSummaryPtr summary = nullptr;
  EXPECT_EQ(graph_manager.GetCompiledGraphSummary(graph_id, summary), SUCCESS);
  ASSERT_NE(summary, nullptr);
  EXPECT_TRUE(ge_root_model->GetFixedFeatureMemory().empty());

  EXPECT_EQ(model_executor.MallocFixedFeatureMemoryIfNeed(graph_node, ge_root_model, rtStream_t(0x1)), SUCCESS);
  auto all_fixed_feature_mem = ge_root_model->GetFixedFeatureMemory();
  ASSERT_EQ(all_fixed_feature_mem.size(), 2U);

  const auto hbm_iter = all_fixed_feature_mem.find(RT_MEMORY_HBM);
  ASSERT_NE(hbm_iter, all_fixed_feature_mem.end());
  EXPECT_NE(hbm_iter->second.addr, nullptr);
  ASSERT_NE(hbm_iter->second.block, nullptr);  // external allocator, block is not nullptr
  EXPECT_EQ(hbm_iter->second.size, 8);
  EXPECT_TRUE(hbm_iter->second.ge_alloc);
  EXPECT_FALSE(hbm_iter->second.user_alloc);

  const auto p2p_iter = all_fixed_feature_mem.find(RT_MEMORY_P2P_DDR);
  ASSERT_NE(p2p_iter, all_fixed_feature_mem.cend());
  EXPECT_NE(p2p_iter->second.addr, nullptr);
  EXPECT_EQ(p2p_iter->second.block, nullptr);
  EXPECT_EQ(p2p_iter->second.size, 1024);
  EXPECT_TRUE(p2p_iter->second.ge_alloc);
  EXPECT_FALSE(p2p_iter->second.user_alloc);

  EXPECT_EQ(model_executor.FreeFixedFeatureMemoryIfNeed(ge_root_model), SUCCESS);
  EXPECT_TRUE(ge_root_model->GetFixedFeatureMemory().empty());
  ExternalAllocatorManager::DeleteExternalAllocator(rtStream_t(0x1));
}

TEST_F(UtestModelExecutorTest, MallocAndFreeFixedFeatureMemoryIfNeed_UserHasSetFeatureMemoryBase_NoNeedToMallocFixed) {
  std::shared_ptr<Allocator> external_allocator = MakeShared<ExternalAllocatorUtStub>();
  ExternalAllocatorManager::SetExternalAllocator(rtStream_t(0x1), external_allocator);
  GraphManager graph_manager;
  GeModelPtr ge_model;
  GraphNodePtr graph_node;
  GeRootModelPtr ge_root_model;
  GraphId graph_id = 1;
  CreateSummaryCompiledModel(graph_node, ge_model, ge_root_model);
  graph_manager.AddGraphNode(graph_id, graph_node);
  graph_node->SetBuildFlag(true);
  graph_node->SetCompiledFlag(true);
  void *p = (void *)0x1234;
  EXPECT_EQ(graph_manager.UpdateFeatureMemoryBase(graph_id, p, 1024), SUCCESS);

  ModelExecutor model_executor;
  model_executor.AddGraphNode(graph_id, graph_node);
  EXPECT_EQ(model_executor.Initialize({}, 0), SUCCESS);

  graph_manager.executor_ = &model_executor;
  CompiledGraphSummaryPtr summary = nullptr;
  EXPECT_EQ(graph_manager.GetCompiledGraphSummary(graph_id, summary), SUCCESS);
  ASSERT_NE(summary, nullptr);
  EXPECT_TRUE(ge_root_model->GetFixedFeatureMemory().empty());

  EXPECT_EQ(model_executor.MallocFixedFeatureMemoryIfNeed(graph_node, ge_root_model, rtStream_t(0x1)), SUCCESS);
  auto all_fixed_feature_mem = ge_root_model->GetFixedFeatureMemory();
  ASSERT_EQ(all_fixed_feature_mem.size(), 1U);

  const auto hbm_iter = all_fixed_feature_mem.find(RT_MEMORY_HBM);
  ASSERT_EQ(hbm_iter, all_fixed_feature_mem.end());

  const auto p2p_iter = all_fixed_feature_mem.find(RT_MEMORY_P2P_DDR);
  ASSERT_NE(p2p_iter, all_fixed_feature_mem.cend());
  EXPECT_NE(p2p_iter->second.addr, nullptr);
  EXPECT_EQ(p2p_iter->second.block, nullptr);
  EXPECT_EQ(p2p_iter->second.size, 1024);
  EXPECT_TRUE(p2p_iter->second.ge_alloc);
  EXPECT_FALSE(p2p_iter->second.user_alloc);

  EXPECT_EQ(model_executor.FreeFixedFeatureMemoryIfNeed(ge_root_model), SUCCESS);
  EXPECT_TRUE(ge_root_model->GetFixedFeatureMemory().empty());
  ExternalAllocatorManager::DeleteExternalAllocator(rtStream_t(0x1));
}

// ============================================================================
// OM2 Online Mode Tests
// ============================================================================

TEST_F(UtestModelExecutorTest, Om2Mode_EnvNotSet_IsFalse) {
  unsetenv("ENABLE_RUNTIME_OM2");
  EXPECT_FALSE(IsOm2OnlineMode());
}

TEST_F(UtestModelExecutorTest, Om2Mode_EnvSetToOne_IsTrue) {
  setenv("ENABLE_RUNTIME_OM2", "1", 1);
  EXPECT_TRUE(IsOm2OnlineMode());
  unsetenv("ENABLE_RUNTIME_OM2");
}

TEST_F(UtestModelExecutorTest, Om2Mode_DynamicSwitch) {
  unsetenv("ENABLE_RUNTIME_OM2");
  EXPECT_FALSE(IsOm2OnlineMode());

  setenv("ENABLE_RUNTIME_OM2", "1", 1);
  EXPECT_TRUE(IsOm2OnlineMode());

  unsetenv("ENABLE_RUNTIME_OM2");
  EXPECT_FALSE(IsOm2OnlineMode());
}

TEST_F(UtestModelExecutorTest, Om2Mode_NonOneValues_AreFalse) {
  setenv("ENABLE_RUNTIME_OM2", "0", 1);
  EXPECT_FALSE(IsOm2OnlineMode());

  setenv("ENABLE_RUNTIME_OM2", "true", 1);
  EXPECT_FALSE(IsOm2OnlineMode());

  setenv("ENABLE_RUNTIME_OM2", "", 1);
  EXPECT_FALSE(IsOm2OnlineMode());

  unsetenv("ENABLE_RUNTIME_OM2");
}

TEST_F(UtestModelExecutorTest, RunGraphWithStreamOm2_ConvertsCallerOutputsToGertOutputs) {
  GeTensorDesc desc(GeShape({1, 4}), FORMAT_ND, DT_FLOAT);
  desc.SetPlacement(kPlacementDevice);
  GeTensor output(desc);
  std::vector<uint8_t> data(16U, 0U);
  ASSERT_EQ(output.SetData(data.data(), data.size()), SUCCESS);

  std::vector<GeTensor> ge_outputs{output};
  std::vector<gert::Tensor> gert_outputs;
  ASSERT_EQ(TensorTransUtils::GeTensors2GertTensors(ge_outputs, gert_outputs), SUCCESS);

  ASSERT_EQ(gert_outputs.size(), 1U);
  EXPECT_NE(gert_outputs[0].GetAddr(), nullptr);
  EXPECT_EQ(gert_outputs[0].GetSize(), data.size());
}

TEST_F(UtestModelExecutorTest, DumpDebugJSONPrint_ReturnsUnsupportedInOm2Mode) {
  EnvValueGuard guard("ENABLE_RUNTIME_OM2");
  EnableOm2OnlineMode();
  ModelExecutor model_executor;
  AscendString json_result;
  EXPECT_EQ(model_executor.DumpDebugJSONPrint(1U, 1U, 0U, json_result), GE_GRAPH_UNSUPPORTED);
}

class UtestModelExecutorOm2Test : public testing::Test {
 protected:
  static void SetUpTestSuite() {
    test_work_dir_ = EnvPath().GetOrCreateCaseTmpPath("UtestModelExecutorOm2");
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
    EnvPath().RemoveRfCaseTmpPath("UtestModelExecutorOm2");
  }

  void TearDown() override {
    unsetenv("ENABLE_RUNTIME_OM2");
  }

  static std::string test_work_dir_;
  static std::string fake_so_path_;
};

std::string UtestModelExecutorOm2Test::test_work_dir_;
std::string UtestModelExecutorOm2Test::fake_so_path_;

TEST_F(UtestModelExecutorOm2Test, RunGraph_HostInput_ReturnsUnsupported) {
  EnvValueGuard guard("ENABLE_RUNTIME_OM2");
  EnableOm2OnlineMode();

  ModelExecutor model_executor;
  EXPECT_EQ(model_executor.Initialize({}, 0), SUCCESS);

  auto compute_graph = MakeShared<ComputeGraph>("test_graph");
  GeRootModelPtr ge_root_model = MakeShared<GeRootModel>();
  EXPECT_EQ(ge_root_model->Initialize(compute_graph), SUCCESS);

  GeModelPtr ge_model = MakeShared<GeModel>();
  ge_model->SetGraph(compute_graph);
  ge_root_model->SetSubgraphInstanceNameToModel(compute_graph->GetName(), ge_model);

  GraphId graph_id = 2001;
  GraphNodePtr graph_node = MakeShared<ge::GraphNode>(graph_id);
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

TEST_F(UtestModelExecutorOm2Test, RunGraph_NotLoaded_ReturnsGraphNotExist) {
  EnvValueGuard guard("ENABLE_RUNTIME_OM2");
  EnableOm2OnlineMode();

  ModelExecutor model_executor;
  EXPECT_EQ(model_executor.Initialize({}, 0), SUCCESS);

  auto compute_graph = MakeShared<ComputeGraph>("test_graph");
  GeRootModelPtr ge_root_model = MakeShared<GeRootModel>();
  EXPECT_EQ(ge_root_model->Initialize(compute_graph), SUCCESS);

  GeModelPtr ge_model = MakeShared<GeModel>();
  ge_model->SetGraph(compute_graph);
  ge_root_model->SetSubgraphInstanceNameToModel(compute_graph->GetName(), ge_model);

  auto model_data = std::make_shared<gert::Om2ModelData>(MakeOm2ModelDataWithFakeSo(fake_so_path_));
  ge_root_model->SetOm2ModelData(model_data);

  GraphId graph_id = 3001;
  GraphNodePtr graph_node = MakeShared<ge::GraphNode>(graph_id);
  graph_node->SetGeRootModel(ge_root_model);

  std::vector<gert::Tensor> inputs(1);
  std::vector<gert::Tensor> outputs(1);
  EXPECT_EQ(model_executor.RunGraph(graph_node, graph_id, inputs, outputs), GE_GRAPH_GRAPH_NOT_EXIST);

  EXPECT_EQ(model_executor.Finalize(), SUCCESS);
}

TEST_F(UtestModelExecutorOm2Test, UnloadGraph_NotLoaded_ReturnsSuccess) {
  EnvValueGuard guard("ENABLE_RUNTIME_OM2");
  EnableOm2OnlineMode();

  ModelExecutor model_executor;
  EXPECT_EQ(model_executor.Initialize({}, 0), SUCCESS);

  auto compute_graph = MakeShared<ComputeGraph>("test_graph");
  GeRootModelPtr ge_root_model = MakeShared<GeRootModel>();
  EXPECT_EQ(ge_root_model->Initialize(compute_graph), SUCCESS);

  GraphId graph_id = 3002;
  EXPECT_EQ(model_executor.UnloadGraph(ge_root_model, graph_id), SUCCESS);

  EXPECT_EQ(model_executor.Finalize(), SUCCESS);
}

TEST_F(UtestModelExecutorOm2Test, RunGraph_InputCountMismatch_ReturnsParamInvalid) {
  EnvValueGuard guard("ENABLE_RUNTIME_OM2");
  EnableOm2OnlineMode();

  ModelExecutor model_executor;
  EXPECT_EQ(model_executor.Initialize({}, 0), SUCCESS);

  auto compute_graph = MakeShared<ComputeGraph>("test_graph");
  GeRootModelPtr ge_root_model = MakeShared<GeRootModel>();
  EXPECT_EQ(ge_root_model->Initialize(compute_graph), SUCCESS);

  GeModelPtr ge_model = MakeShared<GeModel>();
  ge_model->SetGraph(compute_graph);
  ge_root_model->SetSubgraphInstanceNameToModel(compute_graph->GetName(), ge_model);

  GraphId graph_id = 3003;
  GraphNodePtr graph_node = MakeShared<ge::GraphNode>(graph_id);
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

TEST_F(UtestModelExecutorOm2Test, RunGraph_OutputCountMismatch_ReturnsParamInvalid) {
  EnvValueGuard guard("ENABLE_RUNTIME_OM2");
  EnableOm2OnlineMode();

  ModelExecutor model_executor;
  EXPECT_EQ(model_executor.Initialize({}, 0), SUCCESS);

  auto compute_graph = MakeShared<ComputeGraph>("test_graph");
  GeRootModelPtr ge_root_model = MakeShared<GeRootModel>();
  EXPECT_EQ(ge_root_model->Initialize(compute_graph), SUCCESS);

  GeModelPtr ge_model = MakeShared<GeModel>();
  ge_model->SetGraph(compute_graph);
  ge_root_model->SetSubgraphInstanceNameToModel(compute_graph->GetName(), ge_model);

  GraphId graph_id = 3004;
  GraphNodePtr graph_node = MakeShared<ge::GraphNode>(graph_id);
  graph_node->SetGeRootModel(ge_root_model);

  auto model_data = std::make_shared<gert::Om2ModelData>(MakeOm2ModelDataWithFakeSo(fake_so_path_));
  ge_root_model->SetOm2ModelData(model_data);

  EXPECT_EQ(model_executor.LoadGraph(ge_root_model, graph_node), SUCCESS);

  std::vector<uint8_t> data(16U, 0U);
  gert::Tensor input;
  input.SetPlacement(gert::kOnDeviceHbm);
  input.SetData(gert::TensorData(data.data(), nullptr, data.size(), gert::kOnDeviceHbm));

  gert::Tensor output1;
  output1.SetPlacement(gert::kOnDeviceHbm);
  output1.SetData(gert::TensorData(data.data(), nullptr, data.size(), gert::kOnDeviceHbm));
  gert::Tensor output2;
  output2.SetPlacement(gert::kOnDeviceHbm);
  output2.SetData(gert::TensorData(data.data(), nullptr, data.size(), gert::kOnDeviceHbm));

  std::vector<gert::Tensor> inputs;
  inputs.push_back(std::move(input));
  std::vector<gert::Tensor> outputs;
  outputs.push_back(std::move(output1));
  outputs.push_back(std::move(output2));
  EXPECT_EQ(model_executor.RunGraph(graph_node, graph_id, inputs, outputs), PARAM_INVALID);

  EXPECT_EQ(model_executor.UnloadGraph(ge_root_model, graph_id), SUCCESS);
  EXPECT_EQ(model_executor.Finalize(), SUCCESS);
}

TEST_F(UtestModelExecutorOm2Test, RunGraph_InputSizeInsufficient_ReturnsParamInvalid) {
  EnvValueGuard guard("ENABLE_RUNTIME_OM2");
  EnableOm2OnlineMode();

  ModelExecutor model_executor;
  EXPECT_EQ(model_executor.Initialize({}, 0), SUCCESS);

  auto compute_graph = MakeShared<ComputeGraph>("test_graph");
  GeRootModelPtr ge_root_model = MakeShared<GeRootModel>();
  EXPECT_EQ(ge_root_model->Initialize(compute_graph), SUCCESS);

  GeModelPtr ge_model = MakeShared<GeModel>();
  ge_model->SetGraph(compute_graph);
  ge_root_model->SetSubgraphInstanceNameToModel(compute_graph->GetName(), ge_model);

  GraphId graph_id = 3005;
  GraphNodePtr graph_node = MakeShared<ge::GraphNode>(graph_id);
  graph_node->SetGeRootModel(ge_root_model);

  auto model_data = std::make_shared<gert::Om2ModelData>(MakeOm2ModelDataWithFakeSo(fake_so_path_));
  ge_root_model->SetOm2ModelData(model_data);

  EXPECT_EQ(model_executor.LoadGraph(ge_root_model, graph_node), SUCCESS);

  std::vector<uint8_t> small_data(4U, 0U);
  gert::Tensor input;
  input.SetPlacement(gert::kOnDeviceHbm);
  input.SetData(gert::TensorData(small_data.data(), nullptr, small_data.size(), gert::kOnDeviceHbm));

  gert::Tensor output;
  output.SetPlacement(gert::kOnDeviceHbm);
  std::vector<uint8_t> out_data(16U, 0U);
  output.SetData(gert::TensorData(out_data.data(), nullptr, out_data.size(), gert::kOnDeviceHbm));

  std::vector<gert::Tensor> inputs;
  inputs.push_back(std::move(input));
  std::vector<gert::Tensor> outputs;
  outputs.push_back(std::move(output));
  EXPECT_EQ(model_executor.RunGraph(graph_node, graph_id, inputs, outputs), PARAM_INVALID);

  EXPECT_EQ(model_executor.UnloadGraph(ge_root_model, graph_id), SUCCESS);
  EXPECT_EQ(model_executor.Finalize(), SUCCESS);
}

TEST_F(UtestModelExecutorOm2Test, RunGraph_EmptyOutputs_PrepareOm2Outputs) {
  EnvValueGuard guard("ENABLE_RUNTIME_OM2");
  EnableOm2OnlineMode();

  ModelExecutor model_executor;
  EXPECT_EQ(model_executor.Initialize({}, 0), SUCCESS);

  auto compute_graph = MakeShared<ComputeGraph>("test_graph");
  GeRootModelPtr ge_root_model = MakeShared<GeRootModel>();
  EXPECT_EQ(ge_root_model->Initialize(compute_graph), SUCCESS);

  GeModelPtr ge_model = MakeShared<GeModel>();
  ge_model->SetGraph(compute_graph);
  ge_root_model->SetSubgraphInstanceNameToModel(compute_graph->GetName(), ge_model);

  GraphId graph_id = 3006;
  GraphNodePtr graph_node = MakeShared<ge::GraphNode>(graph_id);
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

TEST_F(UtestModelExecutorOm2Test, LoadAndUnload_Success) {
  EnvValueGuard guard("ENABLE_RUNTIME_OM2");
  EnableOm2OnlineMode();

  ModelExecutor model_executor;
  EXPECT_EQ(model_executor.Initialize({}, 0), SUCCESS);

  auto compute_graph = MakeShared<ComputeGraph>("test_graph");
  GeRootModelPtr ge_root_model = MakeShared<GeRootModel>();
  EXPECT_EQ(ge_root_model->Initialize(compute_graph), SUCCESS);

  GeModelPtr ge_model = MakeShared<GeModel>();
  ge_model->SetGraph(compute_graph);
  ge_root_model->SetSubgraphInstanceNameToModel(compute_graph->GetName(), ge_model);

  GraphId graph_id = 3007;
  GraphNodePtr graph_node = MakeShared<ge::GraphNode>(graph_id);
  graph_node->SetGeRootModel(ge_root_model);

  auto model_data = std::make_shared<gert::Om2ModelData>(MakeOm2ModelDataWithFakeSo(fake_so_path_));
  ge_root_model->SetOm2ModelData(model_data);

  EXPECT_EQ(model_executor.LoadGraph(ge_root_model, graph_node), SUCCESS);
  EXPECT_EQ(model_executor.UnloadGraph(ge_root_model, graph_id), SUCCESS);

  EXPECT_EQ(model_executor.Finalize(), SUCCESS);
}

TEST_F(UtestModelExecutorOm2Test, LoadGraph_ExternalConstAndFeatureMemory_Success) {
  EnvValueGuard guard("ENABLE_RUNTIME_OM2");
  EnableOm2OnlineMode();

  ModelExecutor model_executor;
  EXPECT_EQ(model_executor.Initialize({}, 0), SUCCESS);

  auto compute_graph = MakeShared<ComputeGraph>("test_graph");
  GeRootModelPtr ge_root_model = MakeShared<GeRootModel>();
  EXPECT_EQ(ge_root_model->Initialize(compute_graph), SUCCESS);

  GeModelPtr ge_model = MakeShared<GeModel>();
  ge_model->SetGraph(compute_graph);
  ge_root_model->SetSubgraphInstanceNameToModel(compute_graph->GetName(), ge_model);

  GraphId graph_id = 3011;
  GraphNodePtr graph_node = MakeShared<ge::GraphNode>(graph_id);
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

TEST_F(UtestModelExecutorOm2Test, UnloadOm2Graph_UnknownGraph_ReturnsSuccess) {
  ModelExecutor model_executor;
  EXPECT_EQ(model_executor.UnloadOm2Graph(3012U), SUCCESS);
}

TEST_F(UtestModelExecutorOm2Test, RunGraphWithStream_Om2Mode_Success) {
  EnvValueGuard guard("ENABLE_RUNTIME_OM2");
  EnableOm2OnlineMode();

  ModelExecutor model_executor;
  EXPECT_EQ(model_executor.Initialize({}, 0), SUCCESS);

  auto compute_graph = MakeShared<ComputeGraph>("test_graph");
  GeRootModelPtr ge_root_model = MakeShared<GeRootModel>();
  EXPECT_EQ(ge_root_model->Initialize(compute_graph), SUCCESS);

  GeModelPtr ge_model = MakeShared<GeModel>();
  ge_model->SetGraph(compute_graph);
  ge_root_model->SetSubgraphInstanceNameToModel(compute_graph->GetName(), ge_model);

  GraphId graph_id = 3013;
  GraphNodePtr graph_node = MakeShared<ge::GraphNode>(graph_id);
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

TEST_F(UtestModelExecutorOm2Test, RunGraph_InputNullAddr_ReturnsParamInvalid) {
  EnvValueGuard guard("ENABLE_RUNTIME_OM2");
  EnableOm2OnlineMode();

  ModelExecutor model_executor;
  EXPECT_EQ(model_executor.Initialize({}, 0), SUCCESS);

  auto compute_graph = MakeShared<ComputeGraph>("test_graph");
  GeRootModelPtr ge_root_model = MakeShared<GeRootModel>();
  EXPECT_EQ(ge_root_model->Initialize(compute_graph), SUCCESS);

  GeModelPtr ge_model = MakeShared<GeModel>();
  ge_model->SetGraph(compute_graph);
  ge_root_model->SetSubgraphInstanceNameToModel(compute_graph->GetName(), ge_model);

  GraphId graph_id = 3008;
  GraphNodePtr graph_node = MakeShared<ge::GraphNode>(graph_id);
  graph_node->SetGeRootModel(ge_root_model);

  auto model_data = std::make_shared<gert::Om2ModelData>(MakeOm2ModelDataWithFakeSo(fake_so_path_));
  ge_root_model->SetOm2ModelData(model_data);

  EXPECT_EQ(model_executor.LoadGraph(ge_root_model, graph_node), SUCCESS);

  gert::Tensor input;
  input.SetPlacement(gert::kOnDeviceHbm);

  gert::Tensor output;
  output.SetPlacement(gert::kOnDeviceHbm);
  std::vector<uint8_t> out_data(16U, 0U);
  output.SetData(gert::TensorData(out_data.data(), nullptr, out_data.size(), gert::kOnDeviceHbm));

  std::vector<gert::Tensor> inputs;
  inputs.push_back(std::move(input));
  std::vector<gert::Tensor> outputs;
  outputs.push_back(std::move(output));
  EXPECT_EQ(model_executor.RunGraph(graph_node, graph_id, inputs, outputs), PARAM_INVALID);

  EXPECT_EQ(model_executor.UnloadGraph(ge_root_model, graph_id), SUCCESS);
  EXPECT_EQ(model_executor.Finalize(), SUCCESS);
}

TEST_F(UtestModelExecutorOm2Test, RunGraph_OutputNullAddr_ReturnsParamInvalid) {
  EnvValueGuard guard("ENABLE_RUNTIME_OM2");
  EnableOm2OnlineMode();

  ModelExecutor model_executor;
  EXPECT_EQ(model_executor.Initialize({}, 0), SUCCESS);

  auto compute_graph = MakeShared<ComputeGraph>("test_graph");
  GeRootModelPtr ge_root_model = MakeShared<GeRootModel>();
  EXPECT_EQ(ge_root_model->Initialize(compute_graph), SUCCESS);

  GeModelPtr ge_model = MakeShared<GeModel>();
  ge_model->SetGraph(compute_graph);
  ge_root_model->SetSubgraphInstanceNameToModel(compute_graph->GetName(), ge_model);

  GraphId graph_id = 3009;
  GraphNodePtr graph_node = MakeShared<ge::GraphNode>(graph_id);
  graph_node->SetGeRootModel(ge_root_model);

  auto model_data = std::make_shared<gert::Om2ModelData>(MakeOm2ModelDataWithFakeSo(fake_so_path_));
  ge_root_model->SetOm2ModelData(model_data);

  EXPECT_EQ(model_executor.LoadGraph(ge_root_model, graph_node), SUCCESS);

  std::vector<uint8_t> input_data(16U, 0U);
  gert::Tensor input;
  input.SetPlacement(gert::kOnDeviceHbm);
  input.SetData(gert::TensorData(input_data.data(), nullptr, input_data.size(), gert::kOnDeviceHbm));

  gert::Tensor output;
  output.SetPlacement(gert::kOnDeviceHbm);

  std::vector<gert::Tensor> inputs;
  inputs.push_back(std::move(input));
  std::vector<gert::Tensor> outputs;
  outputs.push_back(std::move(output));
  EXPECT_EQ(model_executor.RunGraph(graph_node, graph_id, inputs, outputs), PARAM_INVALID);

  EXPECT_EQ(model_executor.UnloadGraph(ge_root_model, graph_id), SUCCESS);
  EXPECT_EQ(model_executor.Finalize(), SUCCESS);
}

TEST_F(UtestModelExecutorOm2Test, RunGraph_OutputSizeInsufficient_ReturnsParamInvalid) {
  EnvValueGuard guard("ENABLE_RUNTIME_OM2");
  EnableOm2OnlineMode();

  ModelExecutor model_executor;
  EXPECT_EQ(model_executor.Initialize({}, 0), SUCCESS);

  auto compute_graph = MakeShared<ComputeGraph>("test_graph");
  GeRootModelPtr ge_root_model = MakeShared<GeRootModel>();
  EXPECT_EQ(ge_root_model->Initialize(compute_graph), SUCCESS);

  GeModelPtr ge_model = MakeShared<GeModel>();
  ge_model->SetGraph(compute_graph);
  ge_root_model->SetSubgraphInstanceNameToModel(compute_graph->GetName(), ge_model);

  GraphId graph_id = 3010;
  GraphNodePtr graph_node = MakeShared<ge::GraphNode>(graph_id);
  graph_node->SetGeRootModel(ge_root_model);

  auto model_data = std::make_shared<gert::Om2ModelData>(MakeOm2ModelDataWithFakeSo(fake_so_path_));
  ge_root_model->SetOm2ModelData(model_data);

  EXPECT_EQ(model_executor.LoadGraph(ge_root_model, graph_node), SUCCESS);

  std::vector<uint8_t> input_data(16U, 0U);
  gert::Tensor input;
  input.SetPlacement(gert::kOnDeviceHbm);
  input.SetData(gert::TensorData(input_data.data(), nullptr, input_data.size(), gert::kOnDeviceHbm));

  std::vector<uint8_t> small_output_data(4U, 0U);
  gert::Tensor output;
  output.SetPlacement(gert::kOnDeviceHbm);
  output.SetData(gert::TensorData(small_output_data.data(), nullptr, small_output_data.size(), gert::kOnDeviceHbm));

  std::vector<gert::Tensor> inputs;
  inputs.push_back(std::move(input));
  std::vector<gert::Tensor> outputs;
  outputs.push_back(std::move(output));
  EXPECT_EQ(model_executor.RunGraph(graph_node, graph_id, inputs, outputs), PARAM_INVALID);

  EXPECT_EQ(model_executor.UnloadGraph(ge_root_model, graph_id), SUCCESS);
  EXPECT_EQ(model_executor.Finalize(), SUCCESS);
}

TEST_F(UtestModelExecutorOm2Test, UpdateFeatureMemoryBase_ReturnsUnsupported) {
  EnvValueGuard guard("ENABLE_RUNTIME_OM2");
  EnableOm2OnlineMode();

  ModelExecutor model_executor;
  EXPECT_EQ(model_executor.Initialize({}, 0), SUCCESS);

  auto compute_graph = MakeShared<ComputeGraph>("test_graph");
  GraphNodePtr graph_node = MakeShared<ge::GraphNode>(1U);
  graph_node->SetGeRootModel(MakeShared<GeRootModel>());

  EXPECT_EQ(model_executor.UpdateFeatureMemoryBase(graph_node, 0U, 0U), GE_GRAPH_UNSUPPORTED);
  EXPECT_EQ(model_executor.Finalize(), SUCCESS);
}

TEST_F(UtestModelExecutorOm2Test, PaRemapped_ReturnsUnsupported) {
  EnvValueGuard guard("ENABLE_RUNTIME_OM2");
  EnableOm2OnlineMode();

  ModelExecutor model_executor;
  EXPECT_EQ(model_executor.Initialize({}, 0), SUCCESS);

  auto compute_graph = MakeShared<ComputeGraph>("test_graph");
  GraphNodePtr graph_node = MakeShared<ge::GraphNode>(1U);
  graph_node->SetGeRootModel(MakeShared<GeRootModel>());

  std::vector<std::pair<uint64_t, uint64_t>> cross_ranges;
  EXPECT_EQ(model_executor.PaRemapped(graph_node, 0U, 0U, 0U, cross_ranges), GE_GRAPH_UNSUPPORTED);
  EXPECT_EQ(model_executor.Finalize(), SUCCESS);
}

TEST_F(UtestModelExecutorOm2Test, RunThread_Om2Mode_ReturnsUnsupported) {
  EnvValueGuard guard("ENABLE_RUNTIME_OM2");
  EnableOm2OnlineMode();

  ModelExecutor model_executor;
  EXPECT_EQ(model_executor.Initialize({}, 0), SUCCESS);
  model_executor.StartRunThread();

  GraphId graph_id = 4001;
  uint64_t session_id = 0;
  error_message::ErrorManagerContext error_context;
  GEThreadLocalContext context = GetThreadLocalContext();

  std::promise<Status> status_promise;
  auto status_future = status_promise.get_future();
  const auto callback = [&status_promise](Status status, std::vector<gert::Tensor> &outputs) {
    status_promise.set_value(status);
  };

  auto compute_graph = MakeShared<ComputeGraph>("test_graph");
  GeRootModelPtr ge_root_model = MakeShared<GeRootModel>();
  EXPECT_EQ(ge_root_model->Initialize(compute_graph), SUCCESS);

  GeModelPtr ge_model = MakeShared<GeModel>();
  ge_model->SetGraph(compute_graph);
  ge_root_model->SetSubgraphInstanceNameToModel(compute_graph->GetName(), ge_model);

  GraphNodePtr graph_node = MakeShared<ge::GraphNode>(graph_id);
  graph_node->SetGeRootModel(ge_root_model);
  graph_node->SetLoadFlag(true);
  graph_node->IncreaseLoadCount();
  graph_node->Lock();

  std::vector<gert::Tensor> input_tensors(1);

  auto run_args = std::make_shared<RunArgs>();
  ASSERT_TRUE(run_args != nullptr);
  run_args->graph_node = graph_node;
  run_args->graph_id = graph_id;
  run_args->session_id = session_id;
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

TEST_F(UtestModelExecutorOm2Test, ExecuteGraphWithStream_Om2Mode_Success) {
  EnvValueGuard guard("ENABLE_RUNTIME_OM2");
  EnableOm2OnlineMode();

  ModelExecutor model_executor;
  EXPECT_EQ(model_executor.Initialize({}, 0), SUCCESS);

  auto compute_graph = MakeShared<ComputeGraph>("test_graph");
  GeRootModelPtr ge_root_model = MakeShared<GeRootModel>();
  EXPECT_EQ(ge_root_model->Initialize(compute_graph), SUCCESS);

  GeModelPtr ge_model = MakeShared<GeModel>();
  ge_model->SetGraph(compute_graph);
  ge_root_model->SetSubgraphInstanceNameToModel(compute_graph->GetName(), ge_model);

  GraphId graph_id = 4002;
  GraphNodePtr graph_node = MakeShared<ge::GraphNode>(graph_id);
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

}  // namespace ge
