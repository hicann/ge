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

#include "macro_utils/dt_public_scope.h"
#include "dflow/base/exec_runtime/execution_runtime.h"
#include "session_v2/inner_ge_session.h"
#include "ge/ge_api_v2.h"
#include "framework/omg/omg_inner_types.h"
#include "graph/ge_local_context.h"
#include "depends/runtime/src/runtime_stub.h"
#include "graph/utils/graph_utils_ex.h"
#include "graph/ge_global_options.h"
#include "compiler/api/gelib/gelib.h"
#include "common/model/external_allocator_manager.h"
#include "graph/load/graph_loader.h"
#include "graph/load/model_manager/model_manager.h"
#include "graph/load/model_manager/davinci_model.h"
#include "graph/manager/graph_manager_utils.h"
#include "graph/manager/graph_manager.h"
#include "graph/execute/model_executor.h"
#include "common/env_path.h"
#include "depends/mmpa/src/mmpa_stub.h"
#include "ge_running_env/ge_running_env_faker.h"
#include "ge_running_env/fake_op.h"
#include "ge_running_env/dir_env.h"
#include "faker/space_registry_faker.h"
#include "faker/magic_ops.h"
#include "ge/st/stubs/utils/mock_ops_kernel_builder.h"
#include "ge/framework/common/taskdown_common.h"
#include "stub/gert_runtime_stub.h"
#include "common/share_graph.h"

using namespace std;
namespace ge {
namespace {
class RuntimeMock910A : public RuntimeStub {
 public:
  rtError_t rtGetSocVersion(char *version, const uint32_t maxLen) {
    (void)strcpy_s(version, maxLen, "Ascend910A");
    return RT_ERROR_NONE;
  }
};

class RuntimeMock910B1 : public RuntimeStub {
 public:
  rtError_t rtGetSocVersion(char *version, const uint32_t maxLen) {
    (void)strcpy_s(version, maxLen, "Ascend910B1");
    return RT_ERROR_NONE;
  }
};

class ExternalAllocatorUtStub : public Allocator {
 public:
  MemBlock *Malloc(size_t size) override {
    return nullptr;
  }
  void Free(MemBlock *block) override {
  }
};

class StubExecutor : public Executor {
 public:
  Status LoadGraph(const GeRootModelPtr &ge_root_model, const GraphNodePtr &graph_node,
                   const rtStream_t stream = nullptr) override {
    return SUCCESS;
  }

  Status UnloadGraph(const GeRootModelPtr &ge_root_model, const uint32_t graph_id) override {
    return SUCCESS;
  }

  Status PushRunArgs(const std::shared_ptr<RunArgs> &args) override {
    return SUCCESS;
  }

  Status PushRunArgs(const std::shared_ptr<RunArgsV2> &args) override {
    return SUCCESS;
  }

  Status RunGraph(const GraphNodePtr &graph_node, const GraphId graph_id,
                  const std::vector<GeTensor> &inputs, std::vector<GeTensor> &outputs) override {
    return SUCCESS;
  }

  Status RunGraph(const GraphNodePtr &graph_node, const GraphId graph_id,
                  const std::vector<gert::Tensor> &inputs, std::vector<gert::Tensor> &outputs) override {
    return SUCCESS;
  }

  Status RunGraphWithStream(const GraphNodePtr &graph_node, const GraphId graph_id, const rtStream_t stream,
                            const std::vector<GeTensor> &inputs, std::vector<GeTensor> &outputs) override {
    return SUCCESS;
  }

  Status ExecuteGraphWithStream(const GraphNodePtr &graph_node, const GraphId graph_id,
                              rtStream_t const stream, const std::vector<gert::Tensor> &inputs,
                              std::vector<gert::Tensor> &outputs) override {
    return SUCCESS;
  }

  Status UpdateFeatureMemoryBase(const GraphNodePtr &graph_node, const uintptr_t mem_base, const size_t size) override {
    (void)graph_node;
    mem_base_ = mem_base;
    mem_base_size_ = size;
    return SUCCESS;
  }

  Status PaRemapped(const GraphNodePtr &graph_node, const uint64_t va, const uint64_t new_pa,
                    const uint64_t len, std::vector<std::pair<uint64_t, uint64_t>> &cross_ranges) override {
    return SUCCESS;
  }

  uintptr_t mem_base_;
  size_t mem_base_size_;
};

Status InitializeHeterogeneousRuntime(const std::map<std::string, std::string> &options) {
  return SUCCESS;
}
}  // namespace

class UtestInnerGeSession : public testing::Test {
 protected:
  void SetUp() override {}

  void TearDown() override {}

  const std::string run_data_path = PathUtils::Join({EnvPath().GetAirBasePath(), "tests/ge/st/st_run_data/"});
};

Status Callback1(uint32_t, const std::map<std::string, ge::Tensor> &){
  return SUCCESS;
}

Status Callback2(uint32_t, const std::map<AscendString, ge::Tensor> &){
  return SUCCESS;
}


TEST_F(UtestInnerGeSession, build_graph_success) {
  std::map <string, string> options;
  uint64_t session_id = 1;
  InnerGeSession inner_seesion(session_id, options);
  std::vector<ge::Tensor> inputs;
  ge::Tensor tensor;
  inputs.emplace_back(tensor);
  Status ret = inner_seesion.CompileGraph(1, inputs);
  EXPECT_NE(ret, ge::SUCCESS);
}

TEST_F(UtestInnerGeSession, initialize) {
  std::map<std::string, std::string> options = {};
  uint64_t session_id = 1;
  InnerGeSession inner_session(session_id, options);
  EXPECT_EQ(inner_session.Initialize(), SUCCESS);
  EXPECT_EQ(inner_session.Finalize(), SUCCESS);
}

TEST_F(UtestInnerGeSession, check_op_precision_mode) {
  std::map<std::string, std::string> options = {
    {ge::OP_PRECISION_MODE, "./op_precision_mode.ini"}
  };
  uint64_t session_id = 1;
  InnerGeSession inner_session(session_id, options);
  auto ret = inner_session.Initialize();
  EXPECT_NE(ret, ge::SUCCESS);
  options["ge.autoTuneMode"] = "RA";
  InnerGeSession inner_session1(session_id, options);
  ret = inner_session1.Initialize();
  EXPECT_NE(ret, ge::SUCCESS);
}

TEST_F(UtestInnerGeSession, InnerLoadGraph_test) {
  uint32_t graph_id = 1;
  std::map<std::string, std::string> options_init = {};
  uint64_t session_id = 1;
  InnerGeSession inner_session(session_id, options_init);
  std::map<AscendString, AscendString> options = {};
  EXPECT_NE(inner_session.LoadGraph(graph_id, options, nullptr), SUCCESS);
  EXPECT_EQ(inner_session.Initialize(), SUCCESS);
  ComputeGraphPtr compute_graph = MakeShared<ComputeGraph>("test_graph");
  GraphNodePtr graph_node = MakeShared<ge::GraphNode>(graph_id);
  auto root_model = MakeShared<GeRootModel>();
  root_model->Initialize(compute_graph);
  graph_node->SetGeRootModel(root_model);
  inner_session.graph_manager_.AddGraphNode(graph_id, graph_node);
  // not compiled
  EXPECT_NE(inner_session.LoadGraph(graph_id, options, nullptr), SUCCESS);

  // graph running
  graph_node->SetRunFlag(true);
  EXPECT_NE(inner_session.LoadGraph(graph_id, options, nullptr), SUCCESS);

  // graph compiled
  graph_node->SetRunFlag(false);
  options.emplace("ge.exec.frozenInputIndexes", "1;2");
  EXPECT_NE(inner_session.LoadGraph(graph_id, options, nullptr), SUCCESS);
  
  // graph has been loaded
  EXPECT_NE(inner_session.LoadGraph(graph_id, options, nullptr), SUCCESS);
  
  // without executor
  inner_session.graph_manager_.executor_ = nullptr;
  EXPECT_NE(inner_session.LoadGraph(graph_id, options, nullptr), SUCCESS);
  EXPECT_EQ(inner_session.Finalize(), SUCCESS);
}

TEST_F(UtestInnerGeSession, RunGraph_Failed_GraphNodeNull) {
  uint32_t graph_id = 1;
  std::map<std::string, std::string> options_init = {};
  uint64_t session_id = 1;
  InnerGeSession inner_session(session_id, options_init);
  std::map<AscendString, AscendString> options = {};
  EXPECT_EQ(inner_session.Initialize(), SUCCESS);
  ComputeGraphPtr compute_graph = MakeShared<ComputeGraph>("test_graph");
  inner_session.graph_manager_.AddGraphNode(graph_id, nullptr);
  std::vector<gert::Tensor> inputs;
  std::vector<gert::Tensor> outputs;
  EXPECT_NE(inner_session.RunGraph(graph_id, inputs, outputs), SUCCESS);
  EXPECT_EQ(inner_session.Finalize(), SUCCESS);
}

TEST_F(UtestInnerGeSession, RunGraph_Failed_GraphNodeStateError) {
  uint32_t graph_id = 1;
  std::map<std::string, std::string> options_init = {};
  uint64_t session_id = 1;
  InnerGeSession inner_session(session_id, options_init);
  std::map<AscendString, AscendString> options = {};
  EXPECT_NE(inner_session.LoadGraph(graph_id, options, nullptr), SUCCESS);
  EXPECT_EQ(inner_session.Initialize(), SUCCESS);
  ComputeGraphPtr compute_graph = MakeShared<ComputeGraph>("test_graph");
  GraphNodePtr graph_node = MakeShared<ge::GraphNode>(graph_id);
  auto root_model = MakeShared<GeRootModel>();
  root_model->Initialize(compute_graph);
  graph_node->SetGeRootModel(root_model);
  inner_session.graph_manager_.AddGraphNode(graph_id, graph_node);
  graph_node->SetRunFlag(true);
  // not compiled
  std::vector<gert::Tensor> inputs;
  std::vector<gert::Tensor> outputs;
  EXPECT_NE(inner_session.RunGraph(graph_id, inputs, outputs), SUCCESS);
  EXPECT_EQ(inner_session.Finalize(), SUCCESS);
}

TEST_F(UtestInnerGeSession, InnerLoadGraph_test_with_invalid_frozenInputIndexes) {
  uint32_t graph_id = 1;
  std::map<std::string, std::string> options_init = {};
  uint64_t session_id = 1;
  InnerGeSession inner_session(session_id, options_init);
  std::map<AscendString, AscendString> options = {};
  // invalid option
  options.emplace("ge.exec.frozenInputIndexes", "a");
  
  EXPECT_NE(inner_session.LoadGraph(graph_id, options, nullptr), SUCCESS);
  EXPECT_EQ(inner_session.Initialize(), SUCCESS);
  ComputeGraphPtr compute_graph = MakeShared<ComputeGraph>("test_graph");
  GraphNodePtr graph_node = MakeShared<ge::GraphNode>(graph_id);
  auto root_model = MakeShared<GeRootModel>();
  root_model->Initialize(compute_graph);
  graph_node->SetGeRootModel(root_model);
  inner_session.graph_manager_.AddGraphNode(graph_id, graph_node);

  // graph compiled
  graph_node->SetRunFlag(false);
  EXPECT_NE(inner_session.LoadGraph(graph_id, options, nullptr), SUCCESS);
  graph_node->SetLoadFlag(false);

  // empty option
  auto iter = options.find("ge.exec.frozenInputIndexes");
  options.erase(iter);
  options.emplace("ge.exec.frozenInputIndexes", "");
  EXPECT_NE(inner_session.LoadGraph(graph_id, options, nullptr), SUCCESS);
  EXPECT_EQ(inner_session.Finalize(), SUCCESS);
}

TEST_F(UtestInnerGeSession, CheckPrecisionModeParamValid_Failed) {
  ge::GetThreadLocalContext().SetGlobalOption({});
  ge::GetThreadLocalContext().SetSessionOption({});
  ge::GetThreadLocalContext().SetGraphOption({});
  std::map<std::string, std::string> options = {
    {ge::PRECISION_MODE, "Im am invalid, hahah"}
  };
  uint64_t session_id = 1;
  InnerGeSession inner_session(session_id, options);
  auto ret = inner_session.Initialize();
  EXPECT_NE(ret, ge::SUCCESS);
  std::string ge_option;
  EXPECT_NE(ge::GetThreadLocalContext().GetOption(ge::PRECISION_MODE, ge_option), ge::SUCCESS);
}

TEST_F(UtestInnerGeSession, CheckPrecisionModeV2ParamValid_Failed) {
  ge::GetThreadLocalContext().SetGlobalOption({});
  ge::GetThreadLocalContext().SetSessionOption({});
  ge::GetThreadLocalContext().SetGraphOption({});
  std::map<std::string, std::string> options = {
    {ge::PRECISION_MODE_V2, "I am invalid too, hahaha"}
  };
  uint64_t session_id = 1;
  InnerGeSession inner_session(session_id, options);
  auto ret = inner_session.Initialize();
  EXPECT_NE(ret, ge::SUCCESS);
  std::string ge_option;
  EXPECT_NE(ge::GetThreadLocalContext().GetOption(ge::PRECISION_MODE_V2, ge_option), ge::SUCCESS);
}

TEST_F(UtestInnerGeSession, CheckPrecisionModeV2ParamValid_Failed_WhenConfigPrecisionModeAtTheSameTime) {
  ge::GetThreadLocalContext().SetGlobalOption({});
  ge::GetThreadLocalContext().SetSessionOption({});
  ge::GetThreadLocalContext().SetGraphOption({});
  std::map<std::string, std::string> options = {
    {ge::PRECISION_MODE_V2, "fp16"},
    {ge::PRECISION_MODE, "force_fp16"}
  };
  uint64_t session_id = 1;
  InnerGeSession inner_session(session_id, options);
  auto ret = inner_session.Initialize();
  EXPECT_NE(ret, ge::SUCCESS);
  std::string ge_option;
  EXPECT_NE(ge::GetThreadLocalContext().GetOption(ge::PRECISION_MODE_V2, ge_option), ge::SUCCESS);
  EXPECT_EQ(inner_session.Finalize(), SUCCESS);
}

TEST_F(UtestInnerGeSession, CheckPrecisionModeParamValid_Success) {
  std::map<std::string, std::string> options = {
      {ge::PRECISION_MODE, "force_fp16"}
  };
  uint64_t session_id = 1;
  InnerGeSession inner_session(session_id, options);
  auto ret = inner_session.Initialize();
  EXPECT_EQ(ret, ge::SUCCESS);
  std::string ge_option;
  EXPECT_EQ(ge::GetThreadLocalContext().GetOption(ge::PRECISION_MODE, ge_option), ge::SUCCESS);
  EXPECT_STREQ(ge_option.c_str(), "force_fp16");
  EXPECT_EQ(inner_session.Finalize(), SUCCESS);
}

TEST_F(UtestInnerGeSession, CheckPrecisionModeV2ParamValid_Success) {
  std::map<std::string, std::string> options = {
      {ge::PRECISION_MODE_V2, "fp16"}
  };
  uint64_t session_id = 1;
  InnerGeSession inner_session(session_id, options);
  auto ret = inner_session.Initialize();
  EXPECT_EQ(ret, ge::SUCCESS);
  std::string ge_option;
  EXPECT_EQ(ge::GetThreadLocalContext().GetOption(ge::PRECISION_MODE_V2, ge_option), ge::SUCCESS);
  EXPECT_STREQ(ge_option.c_str(), "fp16");
  EXPECT_EQ(inner_session.Finalize(), SUCCESS);
}

TEST_F(UtestInnerGeSession, set_train_flag) {
  std::map<std::string, std::string> options = {
    {ge::OPTION_GRAPH_RUN_MODE, "1"}
  };
  uint64_t session_id = 1;
  InnerGeSession inner_session(session_id, options);
  EXPECT_EQ(inner_session.Initialize(), SUCCESS);
  inner_session.SetTrainFlagOption();
  EXPECT_EQ(domi::GetContext().train_flag, true);
  EXPECT_EQ(inner_session.Finalize(), SUCCESS);
}

TEST_F(UtestInnerGeSession, Initialize_01) {
  std::map<std::string, std::string> options = {
    {"ge.exec.disableReuseMemory", "100"}
  };
  uint64_t session_id = 1;
  InnerGeSession inner_session(session_id, options);
  inner_session.is_initialized_ = true;
  EXPECT_EQ(inner_session.Initialize(), SUCCESS);
  inner_session.is_initialized_ = false;
  EXPECT_EQ(inner_session.Initialize(), FAILED);
  options["ge.exec.disableReuseMemory"] = "1";
  EXPECT_EQ(inner_session.Initialize(), FAILED);
  options["ge.exec.modify_mixlist"] = "";
  EXPECT_EQ(inner_session.Initialize(), FAILED);
  (void)GetThreadLocalContext();
  options["ge.session_device_id"] = "session_device_id";
  EXPECT_EQ(inner_session.Initialize(), FAILED);
  options["ge.session_device_id"] = "9999999999";
  EXPECT_EQ(inner_session.Initialize(), FAILED);
  options["ge.session_device_id"] = "1";
  EXPECT_EQ(inner_session.Initialize(), FAILED);
}

TEST_F(UtestInnerGeSession, Initialize_02) {
  std::map<std::string, std::string> options = {
    {"ge.exec.disableReuseMemory", "1"}
  };
  options.insert({"ge.exec.modify_mixlist", "0"});
  options.insert({"ge.session_device_id", "1"});
  options.insert({"ge.exec.precision_mode", "allow_mix_precision"});

  uint64_t session_id = 1;
  InnerGeSession inner_session(session_id, options);
  inner_session.is_initialized_ = true;
  EXPECT_EQ(inner_session.Initialize(), SUCCESS);
  inner_session.is_initialized_ = false;
  EXPECT_EQ(inner_session.Initialize(), SUCCESS);
  EXPECT_EQ(inner_session.Finalize(), SUCCESS);
}

TEST_F(UtestInnerGeSession, Initialize_03) {
  std::map<std::string, std::string> options = {
    {"ge.exec.disableReuseMemory", "1"}
  };
  options.insert({"ge.exec.modify_mixlist", "0"});
  options.insert({"ge.session_device_id", "abcdefghijk"});
  options.insert({"ge.exec.precision_mode", "allow_mix_precision"});

  uint64_t session_id = 1;
  InnerGeSession inner_session(session_id, options);
  inner_session.is_initialized_ = true;
  EXPECT_EQ(inner_session.Initialize(), SUCCESS);
  inner_session.is_initialized_ = false;
  EXPECT_EQ(inner_session.Initialize(), SUCCESS);
  EXPECT_EQ(inner_session.Finalize(), SUCCESS);
}

TEST_F(UtestInnerGeSession, InitializeExecutionRuntime) {
  class MockMmpa : public MmpaStubApiGe {
   public:
    void *DlSym(void *handle, const char *func_name) override {
      if (std::string(func_name) == "InitializeHeterogeneousRuntime") {
        return (void *) &InitializeHeterogeneousRuntime;
      }
      return dlsym(handle, func_name);
    }
  };

  std::map<std::string, std::string> options = {
    {"ge.exec.disableReuseMemory", "1"}
  };
  options.insert({"ge.exec.modify_mixlist", "0"});
  options.insert({"ge.session_device_id", "1"});
  options.insert({"ge.exec.precision_mode", "allow_mix_precision"});

  ExecutionRuntime::instance_ = nullptr;
  uint64_t session_id = 0;
  InnerGeSession local_session(session_id, options);
  local_session.is_initialized_ = false;
  EXPECT_EQ(local_session.Initialize(), SUCCESS);
  EXPECT_EQ(local_session.Finalize(), SUCCESS);

  setenv("RESOURCE_CONFIG_PATH", "fake_numa_config.json", 1);
  MmpaStub::GetInstance().SetImpl(std::make_shared<MockMmpa>());
  ExecutionRuntime::handle_ = (void *)0xffffffff;

  session_id = 1;
  InnerGeSession inner_session(session_id, options);
  inner_session.is_initialized_ = true;
  EXPECT_EQ(inner_session.Initialize(), SUCCESS);
  inner_session.is_initialized_ = false;
  EXPECT_EQ(inner_session.Initialize(), SUCCESS);
  EXPECT_EQ(inner_session.Finalize(), SUCCESS);

  ExecutionRuntime::handle_ = nullptr;
  EXPECT_EQ(GEFinalizeV2(), SUCCESS);
  ExecutionRuntime::instance_ = nullptr;
  MmpaStub::GetInstance().Reset();
  unsetenv("RESOURCE_CONFIG_PATH");
}

TEST_F(UtestInnerGeSession, AddGraph) {
  std::map<std::string, std::string> options = {
    {"ge.exec.disableReuseMemory", "100"}
  };
  uint64_t session_id = 1;
  InnerGeSession inner_session(session_id, options);
  Graph g("graph");
  EXPECT_NE(inner_session.AddGraph(1, g, {}), SUCCESS);
  options["ge.autoTuneMode"] = "RA";
  EXPECT_EQ(inner_session.AddGraph(1, g, options), FAILED);
}

TEST_F(UtestInnerGeSession, CompileGraph) {
  std::map<std::string, std::string> options = {
    {"ge.exec.disableReuseMemory", "100"}
  };
  uint64_t session_id = 1;
  InnerGeSession inner_session(session_id, options);
  uint32_t graph_id = 0;
  std::vector<ge::Tensor> inputs(1);
  EXPECT_NE(inner_session.CompileGraph(graph_id, inputs), SUCCESS);
}

TEST_F(UtestInnerGeSession, Finalize) {
  std::map<std::string, std::string> options = {
    {"ge.exec.disableReuseMemory", "100"}
  };
  uint64_t session_id = 1;
  InnerGeSession inner_session(session_id, options);
  inner_session.is_initialized_ = true;
  EXPECT_EQ(inner_session.Finalize(), SUCCESS);
}
namespace {
int32_t g_so_addr = 0;
class MockMmpa : public ge::MmpaStubApiGe {
 public:
  void *DlOpen(const char *file_name, int32_t mode) override {
    if (string("libmodel_deployer.so") == file_name) {
      return (void *) &g_so_addr;
    }
    return MmpaStubApiGe::DlOpen(file_name, mode);
  }

  void *DlSym(void *handle, const char *func_name) override {
    if (std::string(func_name) == "InitializeHeterogeneousRuntime") {
      return (void *) &InitializeHeterogeneousRuntime;
    }
    return dlsym(handle, func_name);
  }
  int32_t DlClose(void *handle) override {
    return 0;
  }
};
}

TEST_F(UtestInnerGeSession, InitializeWithJitCompileTrueCheckSuccess) {
  RuntimeStub::SetInstance(std::make_shared<RuntimeMock910B1>());
  std::map<std::string, std::string> options = {{JIT_COMPILE, "1"}};
  uint64_t session_id = 1;
  InnerGeSession inner_session(session_id, options);
  EXPECT_EQ(inner_session.Initialize(), SUCCESS);
  std::string jit_compile_option;
  GetThreadLocalContext().GetOption(JIT_COMPILE, jit_compile_option);
  EXPECT_STREQ(jit_compile_option.c_str(), "1");
  EXPECT_EQ(inner_session.Finalize(), SUCCESS);
}

TEST_F(UtestInnerGeSession, AddGraphWithJitCompileTrueOn910B1_CheckSuccess) {
  RuntimeStub::SetInstance(std::make_shared<RuntimeMock910B1>());
  std::map<std::string, std::string> options = {};
  uint64_t session_id = 1;
  ge::GELib::Initialize({});
  InnerGeSession inner_session(session_id, options);
  EXPECT_EQ(inner_session.Initialize(), SUCCESS);

  // use default value
  std::string jit_compile_option;
  GetThreadLocalContext().GetOption(JIT_COMPILE, jit_compile_option);
  EXPECT_STREQ(jit_compile_option.c_str(), "2");

  ComputeGraphPtr compute_graph = ge::MakeShared<ComputeGraph>("g");
  EXPECT_NE(compute_graph, nullptr);
  ge::Graph g = ge::GraphUtilsEx::CreateGraphFromComputeGraph(compute_graph);
  std::map<std::string, std::string> map_options = {{JIT_COMPILE, "1"}};
  EXPECT_EQ(inner_session.AddGraph(1, g, map_options), SUCCESS);

  // use option on graph
  GetThreadLocalContext().GetOption(JIT_COMPILE, jit_compile_option);
  EXPECT_STREQ(jit_compile_option.c_str(), "1");
  EXPECT_EQ(inner_session.Finalize(), SUCCESS);
}

TEST_F(UtestInnerGeSession, AddGraphWithJitCompileFalseOn910A_CheckSuccess) {
  RuntimeStub::SetInstance(std::make_shared<RuntimeMock910A>());
  std::map<std::string, std::string> options = {};
  uint64_t session_id = 1;
  ge::GELib::Initialize({});
  InnerGeSession inner_session(session_id, options);
  EXPECT_EQ(inner_session.Initialize(), SUCCESS);

  // use default value
  std::string jit_compile_option;
  GetThreadLocalContext().GetOption(JIT_COMPILE, jit_compile_option);
  EXPECT_STREQ(jit_compile_option.c_str(), "2");

  ComputeGraphPtr compute_graph = ge::MakeShared<ComputeGraph>("g");
  EXPECT_NE(compute_graph, nullptr);
  ge::Graph g = ge::GraphUtilsEx::CreateGraphFromComputeGraph(compute_graph);
  std::map<std::string, std::string> map_options = {{JIT_COMPILE, "0"}};
  EXPECT_EQ(inner_session.AddGraph(1, g, map_options), SUCCESS);

  // use option on graph
  GetThreadLocalContext().GetOption(JIT_COMPILE, jit_compile_option);
  EXPECT_STREQ(jit_compile_option.c_str(), "0");
  EXPECT_EQ(inner_session.Finalize(), SUCCESS);
}

TEST_F(UtestInnerGeSession, ExternalAllocator_test) {
  uint32_t stream = 10;
  std::shared_ptr<Allocator> external_allocator = MakeShared<ExternalAllocatorUtStub>();
  std::map <string, string> options;
  uint64_t session_id = 1;
  InnerGeSession inner_session(session_id, options);
  EXPECT_EQ(inner_session.RegisterExternalAllocator(&stream, external_allocator), SUCCESS);
  EXPECT_EQ(ExternalAllocatorManager::GetExternalAllocator(&stream), external_allocator);
  EXPECT_EQ(inner_session.UnregisterExternalAllocator(&stream), SUCCESS);
  EXPECT_EQ(ExternalAllocatorManager::GetExternalAllocator(&stream), nullptr);
}

TEST_F(UtestInnerGeSession, CopyGeOutputsMemToUserOutputs_test) {
  uint32_t stream = 10;
  std::shared_ptr<Allocator> external_allocator = MakeShared<ExternalAllocatorUtStub>();
  std::map <string, string> options;
  uint64_t session_id = 1;
  InnerGeSession inner_session(session_id, options);

  std::vector<ge::Tensor> outputs;
  std::vector<ge::GeTensor> ge_outputs;

  const auto deleter = [](uint8_t *device_data) {
    (void)device_data;
  };

  ge::GeTensorDesc tensor_desc(ge::GeShape({1}), ge::FORMAT_ND, ge::DT_FLOAT16);
  ge::GeTensor ge_tensor(tensor_desc);
  uint8_t *mem1 = (uint8_t *)0x10;
  size_t size1 = 0x1;
  ge_tensor.SetData(mem1, size1, deleter);
  ge_outputs.emplace_back(std::move(ge_tensor));

  CopyGeOutputsMemToUserOutputs(&stream, ge_outputs, outputs);
  EXPECT_EQ(outputs.size(), 0);

  EXPECT_EQ(inner_session.RegisterExternalAllocator(&stream, external_allocator), SUCCESS);
  EXPECT_EQ(ExternalAllocatorManager::GetExternalAllocator(&stream), external_allocator);

  CopyGeOutputsMemToUserOutputs(&stream, ge_outputs, outputs);
  EXPECT_EQ(outputs.size(), 1);
  EXPECT_EQ(outputs[0].GetData(), mem1);
}

TEST_F(UtestInnerGeSession, ForkGraph_OriginGraphNotCompiled_Failed) {
  uint64_t session_id = 1UL;
  std::map<std::string, std::string> options;
  InnerGeSession inner_session(session_id, options);
  EXPECT_EQ(inner_session.Initialize(), SUCCESS);

  ComputeGraphPtr origin_compute_graph = ge::MakeShared<ComputeGraph>("origin_g");
  EXPECT_NE(origin_compute_graph, nullptr);
  ge::Graph origin_g = ge::GraphUtilsEx::CreateGraphFromComputeGraph(origin_compute_graph);
  GraphId origin_graph_id = 1;
  EXPECT_EQ(inner_session.AddGraph(origin_graph_id, origin_g, {}), SUCCESS);
  GraphId forked_graph_id = 2;
  EXPECT_NE(inner_session.ForkGraph(origin_graph_id, forked_graph_id), SUCCESS);
  EXPECT_EQ(inner_session.Finalize(), SUCCESS);
}

TEST_F(UtestInnerGeSession, ForkGraph_GraphIdExists_Failed) {
  uint64_t session_id = 1UL;
  std::map<std::string, std::string> options;
  InnerGeSession inner_session(session_id, options);
  EXPECT_EQ(inner_session.Initialize(), SUCCESS);

  ComputeGraphPtr origin_compute_graph = ge::MakeShared<ComputeGraph>("origin_g");
  EXPECT_NE(origin_compute_graph, nullptr);
  ge::Graph origin_g = ge::GraphUtilsEx::CreateGraphFromComputeGraph(origin_compute_graph);
  GraphId origin_graph_id = 1;
  EXPECT_EQ(inner_session.AddGraph(origin_graph_id, origin_g, {}), SUCCESS);
  EXPECT_NE(inner_session.ForkGraph(origin_graph_id, origin_graph_id), SUCCESS);
  EXPECT_EQ(inner_session.Finalize(), SUCCESS);
}

TEST_F(UtestInnerGeSession, ForkAndLoadGraph_SUCCESS) {
  ge::GELib::GetInstance()->OpsKernelManagerObj().ops_kernel_store_.clear();
  GeRunningEnvFaker().InstallDefault();
  GeRunningEnvFaker ge_env;
  ge_env.Reset()
      .Install(FakeEngine("DNN_VM_GE_LOCAL").KernelInfoStore("DNN_VM_GE_LOCAL_OP_STORE"))
      .Install(FakeEngine("AIcoreEngine").KernelInfoStore("AIcoreEngine"))
      .Install(FakeEngine("DNN_VM_RTS").KernelInfoStore("DNN_VM_RTS_OP_STORE"))
      .Install(FakeOp(NETOUTPUT).InfoStoreAndBuilder("DNN_VM_GE_LOCAL_OP_STORE"))
      .Install(FakeOp(CONSTANT).InfoStoreAndBuilder("DNN_VM_GE_LOCAL_OP_STORE"))
      .Install(FakeOp(DATA).InfoStoreAndBuilder("DNN_VM_GE_LOCAL_OP_STORE"))
      .Install(FakeOp(PARTITIONEDCALL).InfoStoreAndBuilder("DNN_VM_GE_LOCAL_OP_STORE"))
      .Install(FakeOp(RESHAPE).InfoStoreAndBuilder("DNN_VM_GE_LOCAL_OP_STORE"))
      .Install(FakeOp(ADD).Inputs({"x1", "x2"}).Outputs({"y"}).InfoStoreAndBuilder("AIcoreEngine"));

  MockForAiCoreGenerateTask();
  gert::GertRuntimeStub rtstub;
  rtstub.StubByNodeTypes({"Data", "Add", "NetOutput"});
  rtstub.GetRtsRuntimeStub().Clear();
  rtstub.StubByNodeTypes({"Data", "Add", "NetOutput"});
  rtstub.GetKernelStub().AllKernelRegisteredAndSuccess();

  std::vector<Tensor> inputs;
  ge::Tensor tensor1;
  TensorDesc tensor_desc1(Shape({3, 3, 3}), FORMAT_NCHW, DT_FLOAT);
  tensor1.SetTensorDesc(tensor_desc1);
  std::vector<uint8_t> data({1, 2, 3, 4});
  tensor1.SetData(data);
  inputs.emplace_back(tensor1);

  ge::Tensor tensor2;
  TensorDesc tensor_desc2(Shape({3, 3, 3}), FORMAT_NCHW, DT_FLOAT);
  tensor2.SetTensorDesc(tensor_desc2);
  std::vector<uint8_t> data2({1, 2, 3, 4});
  tensor2.SetData(data2);
  inputs.emplace_back(tensor2);

  std::map<std::string, std::string> options;
  std::map<AscendString, AscendString> init_options {
    {OPTION_GRAPH_RUN_MODE, "0"}
  };
  EXPECT_EQ(GEInitializeV2(init_options), SUCCESS);
  InnerGeSession inner_session(0, options);
  EXPECT_EQ(inner_session.Initialize(), SUCCESS);

  ComputeGraphPtr com_graph = gert::ShareGraph::AicoreGraph();
  auto graph = GraphUtilsEx::CreateGraphFromComputeGraph(com_graph);
  GraphId origin_graph_id = 1;
  EXPECT_EQ(inner_session.AddGraph(origin_graph_id, graph, options), SUCCESS);
  EXPECT_EQ(inner_session.CompileGraph(origin_graph_id, inputs), SUCCESS); // load fail but compile success

  GraphId forked_graph_id = 2;
  EXPECT_EQ(inner_session.ForkGraph(origin_graph_id, forked_graph_id), SUCCESS);

  std::map<AscendString, AscendString> load_options;
  EXPECT_NE(inner_session.LoadGraph(forked_graph_id, load_options, nullptr), SUCCESS); // mock lowering func failed

  // remove origin graph, fork graph will not removed together
  EXPECT_EQ(inner_session.getGraphManagerObj().graph_ids_.size(), 2);
  EXPECT_EQ(inner_session.RemoveGraph(origin_graph_id), SUCCESS);
  EXPECT_TRUE(inner_session.getGraphManagerObj().graph_ids_to_forked_ids_.empty());
  EXPECT_EQ(inner_session.getGraphManagerObj().graph_ids_.size(), 1);

  EXPECT_EQ(inner_session.RemoveGraph(forked_graph_id), SUCCESS);
  EXPECT_EQ(inner_session.getGraphManagerObj().graph_ids_.size(), 0);
  EXPECT_EQ(inner_session.Finalize(), SUCCESS);

  TearDownForAiCoreGenerateTask();
  rtstub.Clear();
  ge_env.InstallDefault();
  ge_env.Reset();
}
}  // namespace ge
