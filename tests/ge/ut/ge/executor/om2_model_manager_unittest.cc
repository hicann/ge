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

#include <cstdlib>
#include <fstream>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "om2/om2_model_manager.h"
#include "common/om2/om2_model_data.h"
#include "framework/runtime/om2_model_executor.h"
#include "common/env_path.h"
#include "common/path_utils.h"
#include "graph/utils/file_utils.h"
#include "mmpa/mmpa_api.h"
#include "ge/ge_error_codes.h"

namespace ge {
namespace {
constexpr uint32_t kModelId1 = 9001U;
constexpr uint32_t kModelId2 = 9002U;
constexpr uint32_t kModelId3 = 9003U;
constexpr uint32_t kModelId4 = 9004U;
constexpr uint64_t kSessionId = 0U;

void WriteTextFile(const std::string &file_path, const std::string &content) {
  std::ofstream ofs(file_path, std::ios::binary | std::ios::trunc);
  ASSERT_TRUE(ofs.is_open());
  ofs << content;
}

std::string MakeFakeSoSource() {
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

gert::Om2ModelLoadArg MakeLoadArg(uint32_t model_id) {
  gert::Om2ModelLoadArg load_arg;
  load_arg.device_id = 0;
  load_arg.model_id = model_id;
  return load_arg;
}

// Build a minimal Om2ModelData with a valid fake .so for testing.
gert::Om2ModelData MakeOm2ModelDataWithFakeSo(const std::string &so_bytes_path) {
  gert::Om2ModelData model_data;
  model_data.model_meta.model_name = "test_model";
  model_data.model_meta.root_graph_name = "test_graph";
  model_data.model_meta.work_size = 1024U;

  // Add a minimal input/output descriptor so model desc is valid
  ge::Om2TensorDesc input_desc;
  input_desc.SetName("input");
  input_desc.SetDataType(ge::DT_FLOAT);
  input_desc.SetShape({1, 2, 3, 4});
  model_data.model_meta.input_desc.push_back(input_desc);
  model_data.model_meta.input_desc_v2.push_back(input_desc);

  ge::Om2TensorDesc output_desc;
  output_desc.SetName("output");
  output_desc.SetDataType(ge::DT_FLOAT);
  output_desc.SetShape({1, 2, 3, 4});
  model_data.model_meta.output_desc.push_back(output_desc);
  model_data.model_meta.output_desc_v2.push_back(output_desc);

  // Load fake .so bytes
  auto so_bytes = ReadFileBytes(so_bytes_path);
  model_data.program_body.so_artifact.file_name = "libtest_model_om2.so";
  model_data.program_body.so_artifact.data = std::string(so_bytes.begin(), so_bytes.end());

  return model_data;
}
}  // namespace

class Om2ModelManagerTest : public testing::Test {
 protected:
  static void SetUpTestSuite() {
    test_work_dir_ = EnvPath().GetOrCreateCaseTmpPath("Om2ModelManagerUt");
    setenv("ASCEND_WORK_PATH", test_work_dir_.c_str(), 1);

    // Compile a minimal fake .so
    const std::string src_path = PathUtils::Join({test_work_dir_, "fake_om2.cpp"});
    fake_so_path_ = PathUtils::Join({test_work_dir_, "libfake_om2.so"});
    WriteTextFile(src_path, MakeFakeSoSource());
    const std::string compile_cmd = "ASAN_OPTIONS=detect_leaks=0 LSAN_OPTIONS=detect_leaks=0 g++ -shared -fPIC -o " +
                                    fake_so_path_ + " " + src_path;
    ASSERT_EQ(std::system(compile_cmd.c_str()), 0);
    ASSERT_EQ(mmAccess2(fake_so_path_.c_str(), M_F_OK), EOK);
  }

  static void TearDownTestSuite() {
    unsetenv("ASCEND_WORK_PATH");
    EnvPath().RemoveRfCaseTmpPath("Om2ModelManagerUt");
  }

  void TearDown() override {
    // Clean up any models loaded during the test
    auto &mgr = Om2ModelManager::GetInstance();
    (void)mgr.UnloadModel(kModelId1);
    (void)mgr.UnloadModel(kModelId2);
    (void)mgr.UnloadModel(kModelId3);
    (void)mgr.UnloadModel(kModelId4);
  }

  static std::string test_work_dir_;
  static std::string fake_so_path_;
};

std::string Om2ModelManagerTest::test_work_dir_;
std::string Om2ModelManagerTest::fake_so_path_;

// --- Error path tests (no .so needed) ---

TEST_F(Om2ModelManagerTest, RunModel_NotLoaded_ReturnsNotLoaded) {
  std::vector<gert::Tensor *> inputs;
  std::vector<gert::Tensor *> outputs;
  EXPECT_EQ(Om2ModelManager::GetInstance().RunModel(kModelId1, nullptr, inputs, outputs), GE_RTI_MODEL_NOT_LOADED);
}

TEST_F(Om2ModelManagerTest, UnloadModel_NotFound_ReturnsSuccess) {
  EXPECT_EQ(Om2ModelManager::GetInstance().UnloadModel(kModelId1), SUCCESS);
}

TEST_F(Om2ModelManagerTest, LoadModel_EmptySoData_Fails) {
  gert::Om2ModelData model_data;
  // so_artifact is empty by default → LoadSoFromBuffer fails
  auto load_arg = MakeLoadArg(kModelId1);
  EXPECT_NE(Om2ModelManager::GetInstance().LoadModel(kModelId1, model_data, load_arg, kSessionId), SUCCESS);
}

TEST_F(Om2ModelManagerTest, LoadModel_EmptySoData_NotStoredInMap) {
  gert::Om2ModelData model_data;
  auto load_arg = MakeLoadArg(kModelId1);
  // Load fails
  EXPECT_NE(Om2ModelManager::GetInstance().LoadModel(kModelId1, model_data, load_arg, kSessionId), SUCCESS);
  // Verify model is NOT in the map by trying to run it
  std::vector<gert::Tensor *> inputs;
  std::vector<gert::Tensor *> outputs;
  EXPECT_EQ(Om2ModelManager::GetInstance().RunModel(kModelId1, nullptr, inputs, outputs), GE_RTI_MODEL_NOT_LOADED);
}

// --- Success path tests (using fake .so) ---

TEST_F(Om2ModelManagerTest, LoadModel_Success) {
  auto model_data = MakeOm2ModelDataWithFakeSo(fake_so_path_);
  auto load_arg = MakeLoadArg(kModelId1);
  EXPECT_EQ(Om2ModelManager::GetInstance().LoadModel(kModelId1, model_data, load_arg, kSessionId), SUCCESS);
}

TEST_F(Om2ModelManagerTest, LoadModel_DuplicateId_ReturnsAlreadyExist) {
  auto model_data = MakeOm2ModelDataWithFakeSo(fake_so_path_);
  auto load_arg = MakeLoadArg(kModelId1);
  ASSERT_EQ(Om2ModelManager::GetInstance().LoadModel(kModelId1, model_data, load_arg, kSessionId), SUCCESS);
  EXPECT_EQ(Om2ModelManager::GetInstance().LoadModel(kModelId1, model_data, load_arg, kSessionId),
            GE_GRAPH_GRAPH_ALREADY_EXIST);
}

TEST_F(Om2ModelManagerTest, RunModel_AfterLoad) {
  auto model_data = MakeOm2ModelDataWithFakeSo(fake_so_path_);
  auto load_arg = MakeLoadArg(kModelId1);
  ASSERT_EQ(Om2ModelManager::GetInstance().LoadModel(kModelId1, model_data, load_arg, kSessionId), SUCCESS);

  // RunModel requires tensors matching model desc (2 inputs, 1 output from the fake .so)
  // But the fake .so's RunAsync just returns 0, so we test with empty tensors to verify routing
  std::vector<gert::Tensor *> inputs;
  std::vector<gert::Tensor *> outputs;
  // RunModel looks up executor and calls RunAsync - the executor's RunAsync validates counts
  // but the key test is that RunModel finds the model and delegates to executor
  auto ret = Om2ModelManager::GetInstance().RunModel(kModelId1, nullptr, inputs, outputs);
  // The fake .so RunAsync accepts any input/output counts since we pass empty vectors
  // The actual return value depends on the executor's internal validation
  // We just verify it doesn't return GE_RTI_MODEL_NOT_LOADED (meaning it found the model)
  EXPECT_NE(ret, GE_RTI_MODEL_NOT_LOADED);
}

TEST_F(Om2ModelManagerTest, UnloadModel_RemovesFromMap) {
  auto model_data = MakeOm2ModelDataWithFakeSo(fake_so_path_);
  auto load_arg = MakeLoadArg(kModelId1);
  ASSERT_EQ(Om2ModelManager::GetInstance().LoadModel(kModelId1, model_data, load_arg, kSessionId), SUCCESS);

  EXPECT_EQ(Om2ModelManager::GetInstance().UnloadModel(kModelId1), SUCCESS);

  // After unload, RunModel should fail
  std::vector<gert::Tensor *> inputs;
  std::vector<gert::Tensor *> outputs;
  EXPECT_EQ(Om2ModelManager::GetInstance().RunModel(kModelId1, nullptr, inputs, outputs), GE_RTI_MODEL_NOT_LOADED);
}

TEST_F(Om2ModelManagerTest, UnloadModel_Idempotent) {
  auto model_data = MakeOm2ModelDataWithFakeSo(fake_so_path_);
  auto load_arg = MakeLoadArg(kModelId1);
  ASSERT_EQ(Om2ModelManager::GetInstance().LoadModel(kModelId1, model_data, load_arg, kSessionId), SUCCESS);

  EXPECT_EQ(Om2ModelManager::GetInstance().UnloadModel(kModelId1), SUCCESS);
  EXPECT_EQ(Om2ModelManager::GetInstance().UnloadModel(kModelId1), SUCCESS);  // second unload
}

TEST_F(Om2ModelManagerTest, LoadModel_MultipleIds) {
  auto model_data = MakeOm2ModelDataWithFakeSo(fake_so_path_);

  auto load_arg1 = MakeLoadArg(kModelId1);
  EXPECT_EQ(Om2ModelManager::GetInstance().LoadModel(kModelId1, model_data, load_arg1, kSessionId), SUCCESS);

  auto load_arg2 = MakeLoadArg(kModelId2);
  EXPECT_EQ(Om2ModelManager::GetInstance().LoadModel(kModelId2, model_data, load_arg2, kSessionId), SUCCESS);
}

// --- Thread safety tests ---

TEST_F(Om2ModelManagerTest, ConcurrentLoad_DifferentIds) {
  auto model_data = MakeOm2ModelDataWithFakeSo(fake_so_path_);
  std::vector<std::thread> threads;
  std::vector<ge::Status> results(4);

  const uint32_t ids[] = {kModelId1, kModelId2, kModelId3, kModelId4};
  for (int i = 0; i < 4; ++i) {
    threads.emplace_back([&, i]() {
      auto load_arg = MakeLoadArg(ids[i]);
      results[i] = Om2ModelManager::GetInstance().LoadModel(ids[i], model_data, load_arg, kSessionId);
    });
  }
  for (auto &t : threads) {
    t.join();
  }

  for (int i = 0; i < 4; ++i) {
    EXPECT_EQ(results[i], SUCCESS) << "Thread " << i << " failed with model_id=" << ids[i];
  }

  // Cleanup all
  for (int i = 0; i < 4; ++i) {
    (void)Om2ModelManager::GetInstance().UnloadModel(ids[i]);
  }
}

TEST_F(Om2ModelManagerTest, ConcurrentLoad_SameId_OneSucceeds) {
  auto model_data = MakeOm2ModelDataWithFakeSo(fake_so_path_);
  std::vector<std::thread> threads;
  std::vector<ge::Status> results(4);

  for (int i = 0; i < 4; ++i) {
    threads.emplace_back([&, i]() {
      auto load_arg = MakeLoadArg(kModelId1);
      results[i] = Om2ModelManager::GetInstance().LoadModel(kModelId1, model_data, load_arg, kSessionId);
    });
  }
  for (auto &t : threads) {
    t.join();
  }

  int success_count = 0;
  int already_exist_count = 0;
  for (const auto &r : results) {
    if (r == SUCCESS) {
      ++success_count;
    } else if (r == GE_GRAPH_GRAPH_ALREADY_EXIST) {
      ++already_exist_count;
    }
  }
  EXPECT_EQ(success_count, 1);
  EXPECT_EQ(already_exist_count, 3);

  // Cleanup
  (void)Om2ModelManager::GetInstance().UnloadModel(kModelId1);
}

TEST_F(Om2ModelManagerTest, RunModel_WithStream_CallsRunAsync) {
  auto model_data = MakeOm2ModelDataWithFakeSo(fake_so_path_);
  auto load_arg = MakeLoadArg(kModelId1);
  ASSERT_EQ(Om2ModelManager::GetInstance().LoadModel(kModelId1, model_data, load_arg, kSessionId), SUCCESS);

  void *fake_stream = reinterpret_cast<void *>(0xDEAD);
  std::vector<gert::Tensor *> inputs;
  std::vector<gert::Tensor *> outputs;
  auto ret = Om2ModelManager::GetInstance().RunModel(kModelId1, fake_stream, inputs, outputs);
  EXPECT_NE(ret, GE_RTI_MODEL_NOT_LOADED);
}

TEST_F(Om2ModelManagerTest, GenModelId_ReturnsMonotonicallyIncreasing) {
  auto &mgr = Om2ModelManager::GetInstance();
  const uint32_t id1 = mgr.GenModelId();
  const uint32_t id2 = mgr.GenModelId();
  const uint32_t id3 = mgr.GenModelId();
  EXPECT_EQ(id2, id1 + 1U);
  EXPECT_EQ(id3, id2 + 1U);
}

}  // namespace ge
