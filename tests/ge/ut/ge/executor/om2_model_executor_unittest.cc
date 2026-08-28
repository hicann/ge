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
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <vector>

#include "framework/runtime/dump/model_dump_manager.h"
#include "framework/runtime/om2_model_executor.h"
#include "framework/runtime/rt_session.h"
#include "ge/ge_ir_build.h"
#include "common/env_path.h"
#include "common/helper/om2/json_file.h"
#include "depends/ascendcl/src/ascendcl_stub.h"
#include "common/helper/om2/zip_archive_writer.h"
#include "common/path_utils.h"
#include "graph/utils/file_utils.h"
#include "mmpa/mmpa_api.h"
#include "graph_metadef/depends/checker/tensor_check_utils.h"
#include "ge/ge_error_codes.h"
#include "rt_external_mem.h"
#include "runtime/om2/om2_aipp_utils.h"

namespace ge {
namespace {
constexpr const char *kOm2BaseName = "om2_model_executor_test";
constexpr const char *kModelName = "g1";
constexpr uint32_t kTestModelId = 9527U;
constexpr uintptr_t kFakeRtModelHandleValue = 0x12345678U;

struct ModelDataHolder {
  ge::ModelData model_data{};
  std::unique_ptr<char[]> buffer;
  std::shared_ptr<uint8_t> shared_buffer;
};

std::string GetParentDir(const std::string &path) {
  const auto pos = path.find_last_of('/');
  if (pos == std::string::npos) {
    return {};
  }
  return path.substr(0, pos);
}

void WriteTextFile(const std::string &file_path, const std::string &content) {
  const auto parent_dir = GetParentDir(file_path);
  ASSERT_FALSE(parent_dir.empty());
  ASSERT_EQ(CreateDir(parent_dir), 0);
  std::ofstream ofs(file_path, std::ios::binary | std::ios::trunc);
  ASSERT_TRUE(ofs.is_open());
  ofs << content;
  ASSERT_TRUE(ofs.good());
}

void WriteBinaryFile(const std::string &file_path, const std::vector<uint8_t> &content) {
  const auto parent_dir = GetParentDir(file_path);
  ASSERT_FALSE(parent_dir.empty());
  ASSERT_EQ(CreateDir(parent_dir), 0);
  std::ofstream ofs(file_path, std::ios::binary | std::ios::trunc);
  ASSERT_TRUE(ofs.is_open());
  ofs.write(reinterpret_cast<const char *>(content.data()), static_cast<std::streamsize>(content.size()));
  ASSERT_TRUE(ofs.good());
}

void RunCommandOrAssert(const std::string &command) {
  const std::string wrapped_command =
      "env ASAN_OPTIONS=detect_leaks=0:halt_on_error=0 LSAN_OPTIONS=exitcode=0 " + command;
  ASSERT_EQ(system(wrapped_command.c_str()), 0) << wrapped_command;
}

std::string MakeManifestJson() {
  return R"({
    "atc_command": "",
    "model_num": 1,
    "om2_version": "1.0"
})";
}

std::string MakeModelMetaJsonWithDynamicBatch() {
  return R"({
    "dynamic_dims": {
      "dynamic_type": 1,
      "user_designate_shape_order": ["data"],
      "gears": [
        {"inputs": [1], "outputs": [[1, 1000]]},
        {"inputs": [2], "outputs": [[2, 1000]]},
        {"inputs": [4], "outputs": [[4, 1000]]},
        {"inputs": [8], "outputs": [[8, 1000]]}
      ]
    },
    "inputs": [
        {
            "data_type": "DT_FLOAT",
            "format": "NCHW",
            "index": 0,
            "name": "data",
            "shape": [-1, 3, 224, 224],
            "max_gear_shape": [8, 3, 224, 224],
            "shape_range": [],
            "size": 0
        }
    ],
    "name": "g1",
    "outputs": [
        {
            "data_type": "DT_FLOAT",
            "format": "ND",
            "index": 0,
            "name": "output",
            "shape": [-1, 1000],
            "shape_range": [],
            "size": 0
        }
    ],
    "work_size": 2048,
    "zero_copy_size": 0
})";
}

std::string MakeModelMetaJsonWithDynamicHW() {
  return R"({
    "dynamic_dims": {
      "dynamic_type": 2,
      "user_designate_shape_order": ["data"],
      "gears": [
        {"inputs": [224, 224], "outputs": [[1, 1000]]},
        {"inputs": [448, 448], "outputs": [[1, 1000]]}
      ]
    },
    "inputs": [
        {
            "data_type": "DT_FLOAT",
            "format": "NCHW",
            "index": 0,
            "name": "data",
            "shape": [1, 3, -1, -1],
            "max_gear_shape": [1, 3, 448, 448],
            "shape_range": [],
            "size": 0
        }
    ],
    "name": "g1",
    "outputs": [
        {
            "data_type": "DT_FLOAT",
            "format": "ND",
            "index": 0,
            "name": "output",
            "shape": [1, 1000],
            "shape_range": [],
            "size": 0
        }
    ],
    "work_size": 2048,
    "zero_copy_size": 0
})";
}

std::string MakeModelMetaJsonWithDynamicDims() {
  return R"({
    "dynamic_dims": {
      "dynamic_type": 3,
      "user_designate_shape_order": ["data"],
      "gears": [
        {"inputs": [1, 128], "outputs": [[1, 128]]},
        {"inputs": [1, 256], "outputs": [[1, 256]]},
        {"inputs": [1, 512], "outputs": [[1, 512]]}
      ]
    },
    "inputs": [
        {
            "data_type": "DT_FLOAT",
            "format": "ND",
            "index": 0,
            "name": "data",
            "shape": [-1, -1],
            "max_gear_shape": [1, 512],
            "shape_range": [],
            "size": 0
        }
    ],
    "name": "g1",
    "outputs": [
        {
            "data_type": "DT_FLOAT",
            "format": "ND",
            "index": 0,
            "name": "output",
            "shape": [-1, -1],
            "shape_range": [],
            "size": 0
        }
    ],
    "work_size": 2048,
    "zero_copy_size": 0
})";
}

std::string MakeModelMetaJson() {
  return R"({
    "inputs": [
        {
            "data_type": "DT_FLOAT",
            "format": "ND",
            "index": 0,
            "name": "data1",
            "shape": [1, 2, 3, 4],
            "shape_range": [],
            "size": 0
        },
        {
            "data_type": "DT_FLOAT",
            "format": "NCHW",
            "index": 1,
            "name": "data2",
            "shape": [1, 1, 224, 224],
            "shape_range": [],
            "size": 0
        }
    ],
    "name": "g1",
    "outputs": [
        {
            "data_type": "DT_FLOAT",
            "format": "ND",
            "index": 0,
            "name": "output_0_reshape1_0",
            "shape": [],
            "shape_range": [],
            "size": 4
        }
    ],
    "work_size": 2048,
    "zero_copy_size": 0
})";
}

std::string MakeModelMetaJsonWithZeroCopySize() {
  return R"({
    "inputs": [
        {
            "data_type": "DT_FLOAT",
            "format": "ND",
            "index": 0,
            "name": "data1",
            "shape": [1, -1, 3, 4],
            "max_gear_shape": [1, 2, 3, 4],
            "shape_aclmdlGetInputDimsV2": [1, 8, 3, 4],
            "shape_range": [],
            "size": 0
        },
        {
            "data_type": "DT_FLOAT",
            "format": "NCHW",
            "index": 1,
            "name": "data2",
            "shape": [1, 1, -1, 224],
            "max_gear_shape": [1, 1, 224, 224],
            "shape_aclmdlGetInputDimsV2": [1, 1, 448, 224],
            "shape_range": [],
            "size": 0
        }
    ],
    "name": "g1",
    "outputs": [
        {
            "data_type": "DT_FLOAT",
            "format": "ND",
            "index": 0,
            "name": "output_0_reshape1_0",
            "shape": [],
            "shape_range": [],
            "size": 4
        }
    ],
    "work_size": 2048,
    "zero_copy_size": 1024
})";
}

std::string MakeModelMetaJsonWithoutRootGraphName() {
  return R"({
    "inputs": [],
    "name": "g1",
    "outputs": [],
    "work_size": 2048,
    "zero_copy_size": 0
})";
}

std::string MakeModelMetaJsonWithoutInputShape() {
  return R"({
    "inputs": [
        {
            "data_type": "DT_FLOAT",
            "format": "ND",
            "index": 0,
            "name": "data1",
            "shape_range": [],
            "size": 0
        }
    ],
    "name": "g1",
    "outputs": [],
    "work_size": 2048,
    "zero_copy_size": 0,
    "user_designate_shape_order": []
})";
}

std::string MakeVariablesConfigJson() {
  ge::JsonFile tensor_desc;
  (void)tensor_desc.Set("name", "var_0");
  (void)tensor_desc.Set("shape", std::vector<int64_t>{1});
  (void)tensor_desc.Set("data_type", "DT_FLOAT");
  (void)tensor_desc.Set("format", "ND");
  (void)tensor_desc.Set("size", 4U);
  (void)tensor_desc.Set("shape_range", std::vector<std::pair<int64_t, int64_t>>{});

  ge::JsonFile meta;
  (void)meta.Set("index", 0U);
  (void)meta.Set("var_name", "var_0");
  (void)meta.Set("op_type", "VARIABLE");
  (void)meta.Set("op_name", "var_0");
  (void)meta.Set("tensor_desc", tensor_desc.Raw());
  auto metas = ge::JsonFile::json::array();
  metas.push_back(meta.Raw());

  ge::JsonFile root;
  (void)root.Set("graph_id", 7U);
  (void)root.Set("var_metas", metas);
  return root.Dump();
}

std::string MakeVarResourceJson(const size_t init_data_offset, const size_t init_data_size) {
  ge::JsonFile tensor_desc;
  (void)tensor_desc.Set("name", "var_0");
  (void)tensor_desc.Set("shape", std::vector<int64_t>{1});
  (void)tensor_desc.Set("data_type", "DT_FLOAT");
  (void)tensor_desc.Set("format", "ND");
  (void)tensor_desc.Set("size", 4U);
  (void)tensor_desc.Set("shape_range", std::vector<std::pair<int64_t, int64_t>>{});

  const std::string var_key = "var_00_0";
  ge::JsonFile entry;
  (void)entry.Set("var_name", "var_0");
  (void)entry.Set("var_key", var_key);
  (void)entry.Set("op_type", "VARIABLE");
  (void)entry.Set("logic_addr", 0U);
  (void)entry.Set("size", 4U);
  (void)entry.Set("memory_type", static_cast<uint32_t>(RT_MEMORY_HBM));
  (void)entry.Set("changed_graph_id", 7U);
  (void)entry.Set("allocated_graph_id", 7U);
  (void)entry.Set("tensor_desc", tensor_desc.Raw());
  (void)entry.Set("trans_road", ge::JsonFile::json::array());
  (void)entry.Set("copy_info", ge::JsonFile::json::object());
  (void)entry.Set("init_data_offset", init_data_offset);
  (void)entry.Set("init_data_size", init_data_size);

  auto entries = ge::JsonFile::json::object();
  entries[var_key] = entry.Raw();
  ge::JsonFile root;
  (void)root.Set("entries", entries);
  return root.Dump();
}

static std::string interface_header_src = R"(#pragma once

#include <cstddef>
#include <cstdint>

namespace gert {
  class Tensor;
}

namespace om2 {
struct FakeModel {
  uint64_t session_id;
};
}
struct GertModelLoadConfig {
  uint64_t struct_size = sizeof(GertModelLoadConfig);
  const char **bin_files = nullptr;
  const void **bin_data = nullptr;
  uint64_t *bin_size = nullptr;
  uint64_t bin_num = 0;
  void **constants = nullptr;
  void **var_addrs = nullptr;
  void *work_ptr = nullptr;
  uint64_t *session_id = nullptr;
  uint64_t model_id = 0; // used for logging
  void *instance_handle = nullptr;
  const struct GertModelCallbacks *callbacks = nullptr;
  int64_t priority = 0;
};

struct GertModelRunConfig {
  uint64_t struct_size = sizeof(GertModelRunConfig);
  uint64_t input_count = 0;
  gert::Tensor **input_data = nullptr;
  uint64_t output_count = 0;
  gert::Tensor **output_data = nullptr;
  uint64_t stream_sync_timeout_ms = 0;
};

struct GertModelUnloadConfig {
  uint64_t struct_size = sizeof(GertModelUnloadConfig);
};

struct GertModelLoadOutput {
  uint64_t struct_size = sizeof(GertModelLoadOutput);
};

struct GertModelRunOutput {
  uint64_t struct_size = sizeof(GertModelRunOutput);
  void *prof_info = nullptr;
};

struct GertModelUnloadOutput {
  uint64_t struct_size = sizeof(GertModelUnloadOutput);
};

extern "C" {
typedef void *GertModelHandle;

int GertModelLoad(const struct GertModelLoadConfig *config, GertModelHandle *model_handle, struct GertModelLoadOutput *output);

int GertModelRunAsync(GertModelHandle model_handle, void *stream, const struct GertModelRunConfig *config, struct GertModelRunOutput *output);

int GertModelRun(GertModelHandle model_handle, const struct GertModelRunConfig *config, struct GertModelRunOutput *output);

int GertModelUnload(GertModelHandle model_handle, const struct GertModelUnloadConfig *config, struct GertModelUnloadOutput *output);
}
)";

std::string MakeInterfaceHeader() {
  return interface_header_src;
}

std::string MakeLoadAndRunCpp() {
  return R"(#include "g1_interface.h"

#include <cstdlib>
#include <fstream>
#include <new>
#include <string>

namespace {
constexpr uintptr_t kFakeRtModelHandleValue = 0x12345678U;

bool CheckWorkPtr(void *work_ptr) {
  const char *mode = std::getenv("OM2_EXPECT_WORK_PTR_MODE");
  if ((mode == nullptr) || (mode[0] == '\0')) {
    return true;
  }
  const std::string mode_str(mode);
  if (mode_str == "NON_NULL") {
    return work_ptr != nullptr;
  }
  if (mode_str == "EQUAL") {
    const char *value = std::getenv("OM2_EXPECT_WORK_PTR_VALUE");
    if (value == nullptr) {
      return false;
    }
    const auto expect_ptr = reinterpret_cast<void *>(std::stoull(value, nullptr, 16));
    return work_ptr == expect_ptr;
  }
  return false;
}

bool CheckConst0(void **constants) {
  const char *mode = std::getenv("OM2_EXPECT_CONST0_MODE");
  if ((mode == nullptr) || (mode[0] == '\0')) {
    return true;
  }
  if ((constants == nullptr) || (constants[0] == nullptr)) {
    return false;
  }
  const char *value = std::getenv("OM2_EXPECT_CONST0_FIRST_BYTE");
  if (value == nullptr) {
    return false;
  }
  const auto expect = static_cast<unsigned char>(std::stoul(value, nullptr, 10));
  return *(static_cast<unsigned char *>(constants[0])) == expect;
}

bool CheckConst0Ptr(void **constants) {
  const char *mode = std::getenv("OM2_EXPECT_CONST0_PTR_MODE");
  if ((mode == nullptr) || (mode[0] == '\0')) {
    return true;
  }
  if ((constants == nullptr) || (constants[0] == nullptr)) {
    return false;
  }
  const std::string mode_str(mode);
  if (mode_str == "EQUAL") {
    const char *value = std::getenv("OM2_EXPECT_CONST0_PTR_VALUE");
    if (value == nullptr) {
      return false;
    }
    const auto expect_ptr = reinterpret_cast<void *>(std::stoull(value, nullptr, 16));
    return constants[0] == expect_ptr;
  }
  return false;
}

bool CheckConstByIndex(void **constants, size_t index, const char *mode_env, const char *value_env) {
  const char *mode = std::getenv(mode_env);
  if ((mode == nullptr) || (mode[0] == '\0')) {
    return true;
  }
  if ((constants == nullptr) || (constants[index] == nullptr)) {
    return false;
  }
  const char *value = std::getenv(value_env);
  if (value == nullptr) {
    return false;
  }
  const auto expect = static_cast<unsigned char>(std::stoul(value, nullptr, 10));
  return *(static_cast<unsigned char *>(constants[index])) == expect;
}

bool CheckConstPtrEqual(void **constants) {
  const char *value = std::getenv("OM2_EXPECT_CONST1_CONST2_PTR_EQUAL");
  if ((value == nullptr) || (value[0] == '\0')) {
    return true;
  }
  if ((constants == nullptr) || (constants[1] == nullptr) || (constants[2] == nullptr)) {
    return false;
  }
  return constants[1] == constants[2];
}

bool CheckVar0(void **var_addrs) {
  const char *mode = std::getenv("OM2_EXPECT_VAR0_MODE");
  if ((mode == nullptr) || (mode[0] == '\0')) {
    return true;
  }
  return std::string(mode) == "NON_NULL" && var_addrs != nullptr && var_addrs[0] != nullptr;
}

bool CheckSessionId(uint64_t *session_id) {
  const char *value = std::getenv("OM2_EXPECT_SESSION_ID");
  if ((value == nullptr) || (value[0] == '\0')) {
    return true;
  }
  if (session_id == nullptr) {
    return false;
  }
  if (std::string(value) == "ANY") {
    return true;
  }
  const auto expect = static_cast<uint64_t>(std::stoull(value, nullptr, 10));
  return *session_id == expect;
}

bool CheckModelId(uint32_t model_id) {
  const char *value = std::getenv("OM2_EXPECT_MODEL_ID");
  if ((value == nullptr) || (value[0] == '\0')) {
    return true;
  }
  const auto expect = static_cast<uint32_t>(std::stoul(value, nullptr, 10));
  return model_id == expect;
}

bool CheckInstanceHandle(void *instance_handle) {
  const char *mode = std::getenv("OM2_EXPECT_INSTANCE_HANDLE_MODE");
  if ((mode == nullptr) || (mode[0] == '\0')) {
    return true;
  }
  return std::string(mode) == "NON_NULL" && instance_handle != nullptr;
}
}  // namespace

extern "C" int GertModelLoad(const struct GertModelLoadConfig *config, GertModelHandle *model_handle, struct GertModelLoadOutput *output) {
  if ((model_handle == nullptr) || (config == nullptr)) {
    return 1;
  }
  if (!CheckWorkPtr(config->work_ptr) || !CheckConst0(config->constants) || !CheckConst0Ptr(config->constants) ||
      !CheckConstByIndex(config->constants, 1U, "OM2_EXPECT_CONST1_MODE", "OM2_EXPECT_CONST1_FIRST_BYTE") ||
      !CheckConstByIndex(config->constants, 2U, "OM2_EXPECT_CONST2_MODE", "OM2_EXPECT_CONST2_FIRST_BYTE") ||
      !CheckConstPtrEqual(config->constants) || !CheckVar0(config->var_addrs) ||
      !CheckSessionId(config->session_id) || !CheckModelId(config->model_id) || !CheckInstanceHandle(config->instance_handle)) {
    return 1;
  }
  auto *model = new (std::nothrow) om2::FakeModel();
  if (model == nullptr) {
    return 1;
  }
  model->session_id = (config->session_id == nullptr) ? 0UL : *config->session_id;
  *model_handle = model;
  const char *trace = std::getenv("OM2_CALL_TRACE");
  if (trace != nullptr) {
    std::ofstream ofs(trace, std::ios::app);
    ofs << "create\n";
  }
  return 0;
}

extern "C" int GertModelRunAsync(GertModelHandle model_handle, void *stream, const struct GertModelRunConfig *config, struct GertModelRunOutput *output) {
  if ((model_handle == nullptr) || (config == nullptr) || (config->input_data == nullptr) || (config->output_data == nullptr)) {
    return 1;
  }
  return (config->input_count == 2 && config->output_count == 1) ? 0 : 1;
}

extern "C" int GertModelRun(GertModelHandle model_handle, const struct GertModelRunConfig *config, struct GertModelRunOutput *output) {
  if ((model_handle == nullptr) || (config == nullptr) || (config->input_data == nullptr) || (config->output_data == nullptr)) {
    return 1;
  }
  return (config->input_count == 2 && config->output_count == 1) ? 0 : 1;
}

extern "C" int GertModelUnload(GertModelHandle model_handle, const struct GertModelUnloadConfig *config, struct GertModelUnloadOutput *output) {
  if (model_handle == nullptr) {
    return 0;
  }
  delete static_cast<om2::FakeModel *>(model_handle);
  return 0;
}
)";
}

std::string MakeEmptyCpp(const std::string &header_name) {
  return "#include \"" + header_name + "\"\n";
}

std::string MakeCMakeLists() {
  return R"(cmake_minimum_required(VERSION 3.10)
project(g1_om2 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

add_library(g1_om2 SHARED
  g1_resources.cpp
  g1_kernel_reg.cpp
  g1_load_and_run.cpp
  g1_args_manager.cpp
)

set_target_properties(g1_om2 PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)
)";
}

std::string MakeConstantsConfigJson() {
  return R"({
    "internal_weight_size": 16,
    "consts": {
      "fc1_weight": {
        "file_name": "constant_0",
        "index": 0,
        "type": "INTERNAL",
        "offset": 0,
        "size": 16
      }
    }
  })";
}

std::string MakeIndividualConstantsConfigJson() {
  return R"({
    "internal_weight_size": 0,
    "consts": {
      "fc1_weight": {
        "file_name": "fc.bin",
        "index": 0,
        "type": "INDIVIDUAL",
        "offset": 1,
        "size": 2
      }
    }
  })";
}

std::string MakeIndividualConstantsConfigJsonWithZeroInternalWeightSize() {
  return R"({
    "internal_weight_size": 0,
    "consts": {
      "fc1_weight": {
        "file_name": "fc.bin",
        "index": 0,
        "type": "INDIVIDUAL",
        "offset": 1,
        "size": 2
      }
    }
  })";
}

std::string MakeCombinedConstantsConfigJson() {
  return R"({
    "internal_weight_size": 0,
    "consts": {
      "fc1_weight": {
        "file_name": "combined.bin",
        "index": 0,
        "type": "COMBINED",
        "offset": 1,
        "size": 2
      }
    }
  })";
}

std::string MakeMixedConstantsConfigJson() {
  return R"({
    "internal_weight_size": 16,
    "consts": {
      "fc0_weight": {
        "file_name": "constant_0",
        "index": 0,
        "type": "INTERNAL",
        "offset": 0,
        "size": 16
      },
      "fc1_weight": {
        "file_name": "mixed_fc.bin",
        "index": 1,
        "type": "INDIVIDUAL",
        "offset": 1,
        "size": 2
      },
      "fc2_weight": {
        "file_name": "mixed_combined.bin",
        "index": 2,
        "type": "COMBINED",
        "offset": 1,
        "size": 2
      }
    }
  })";
}

std::string MakeDuplicateIndividualConstantsConfigJson() {
  return R"({
    "internal_weight_size": 0,
    "consts": {
      "fc1_weight": {
        "file_name": "duplicate_fc.bin",
        "index": 1,
        "type": "INDIVIDUAL",
        "offset": 1,
        "size": 2
      },
      "fc2_weight": {
        "file_name": "duplicate_fc.bin",
        "index": 2,
        "type": "INDIVIDUAL",
        "offset": 1,
        "size": 2
      }
    }
  })";
}

std::string PtrToHexString(const void *ptr) {
  std::ostringstream oss;
  oss << std::hex << reinterpret_cast<uintptr_t>(ptr);
  return oss.str();
}

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

class AclRuntimeStubGuard {
 public:
  explicit AclRuntimeStubGuard(AclRuntimeStub *stub) : stub_(stub) {
    AclRuntimeStub::Install(stub_);
  }

  ~AclRuntimeStubGuard() {
    AclRuntimeStub::UnInstall(stub_);
  }

 private:
  AclRuntimeStub *stub_;
};

class VarInitDataRecordingAclRuntimeStub : public AclRuntimeStub {
 public:
  struct MemcpyRecord {
    void *dst = nullptr;
    size_t dest_max = 0U;
    std::vector<uint8_t> data;
    aclrtMemcpyKind kind = ACL_MEMCPY_HOST_TO_DEVICE;
  };

  aclError aclrtMemcpy(void *dst, size_t dest_max, const void *src, size_t count, aclrtMemcpyKind kind) override {
    MemcpyRecord record;
    record.dst = dst;
    record.dest_max = dest_max;
    record.kind = kind;
    if ((src != nullptr) && (count > 0U)) {
      const auto *src_bytes = static_cast<const uint8_t *>(src);
      record.data.assign(src_bytes, src_bytes + count);
    }
    memcpy_records.emplace_back(std::move(record));
    return AclRuntimeStub::aclrtMemcpy(dst, dest_max, src, count, kind);
  }

  std::vector<MemcpyRecord> memcpy_records;
};

enum class VarWeightOrder {
  kAbsent,
  kBeforeResource,
  kAfterResource,
};

gert::Om2ModelLoadArg MakeOm2LoadArg() {
  gert::Om2ModelLoadArg load_arg;
  load_arg.device_id = 0;
  load_arg.model_id = kTestModelId;
  return load_arg;
}

std::vector<std::string> ReadTraceFile(const std::string &trace_file) {
  std::ifstream ifs(trace_file);
  std::vector<std::string> lines;
  std::string line;
  while (std::getline(ifs, line)) {
    lines.push_back(line);
  }
  return lines;
}
}  // namespace

class Om2ModelExecutorUt : public testing::Test {
 protected:
  void SetUp() override {
    unsetenv("OM2_EXPECT_WORK_PTR_MODE");
    unsetenv("OM2_EXPECT_WORK_PTR_VALUE");
    unsetenv("OM2_EXPECT_CONST0_MODE");
    unsetenv("OM2_EXPECT_CONST0_FIRST_BYTE");
    unsetenv("OM2_EXPECT_CONST0_PTR_MODE");
    unsetenv("OM2_EXPECT_CONST0_PTR_VALUE");
    unsetenv("OM2_EXPECT_CONST1_MODE");
    unsetenv("OM2_EXPECT_CONST1_FIRST_BYTE");
    unsetenv("OM2_EXPECT_CONST2_MODE");
    unsetenv("OM2_EXPECT_CONST2_FIRST_BYTE");
    unsetenv("OM2_EXPECT_CONST1_CONST2_PTR_EQUAL");
    unsetenv("OM2_EXPECT_VAR0_MODE");
    unsetenv("OM2_EXPECT_SESSION_ID");
    unsetenv("OM2_EXPECT_MODEL_ID");
    unsetenv("OM2_EXPECT_INSTANCE_HANDLE_MODE");
    unsetenv("OM2_CALL_TRACE");
  }

  void TearDown() override {
    unsetenv("OM2_EXPECT_WORK_PTR_MODE");
    unsetenv("OM2_EXPECT_WORK_PTR_VALUE");
    unsetenv("OM2_EXPECT_CONST0_MODE");
    unsetenv("OM2_EXPECT_CONST0_FIRST_BYTE");
    unsetenv("OM2_EXPECT_CONST0_PTR_MODE");
    unsetenv("OM2_EXPECT_CONST0_PTR_VALUE");
    unsetenv("OM2_EXPECT_CONST1_MODE");
    unsetenv("OM2_EXPECT_CONST1_FIRST_BYTE");
    unsetenv("OM2_EXPECT_CONST2_MODE");
    unsetenv("OM2_EXPECT_CONST2_FIRST_BYTE");
    unsetenv("OM2_EXPECT_CONST1_CONST2_PTR_EQUAL");
    unsetenv("OM2_EXPECT_VAR0_MODE");
    unsetenv("OM2_EXPECT_SESSION_ID");
    unsetenv("OM2_EXPECT_MODEL_ID");
    unsetenv("OM2_EXPECT_INSTANCE_HANDLE_MODE");
    unsetenv("OM2_CALL_TRACE");
  }

  static void SetUpTestSuite() {
    test_work_dir_ = EnvPath().GetOrCreateCaseTmpPath("Om2ModelExecutorUt");
    setenv("ASCEND_WORK_PATH", test_work_dir_.c_str(), 1);
    om2_file_path_ = PathUtils::Join({test_work_dir_, std::string(kOm2BaseName) + ".om2"});
    om2_fileconst_file_path_ = PathUtils::Join({test_work_dir_, std::string(kOm2BaseName) + "_fileconst.om2"});
    om2_combined_file_path_ = PathUtils::Join({test_work_dir_, std::string(kOm2BaseName) + "_combined.om2"});
    om2_mixed_file_path_ = PathUtils::Join({test_work_dir_, std::string(kOm2BaseName) + "_mixed.om2"});
    om2_duplicate_individual_file_path_ =
        PathUtils::Join({test_work_dir_, std::string(kOm2BaseName) + "_duplicate_individual.om2"});
    PrepareOm2File();
    PrepareFileConstOm2File();
    PrepareCombinedFileConstOm2File();
    PrepareMixedOm2File();
    PrepareDuplicateIndividualOm2File();
  }

  static void TearDownTestSuite() {
    unsetenv("ASCEND_WORK_PATH");
    EnvPath().RemoveRfCaseTmpPath("Om2ModelExecutorUt");
  }

  static void PrepareOm2File() {
    std::call_once(prepare_once_, []() {
      const std::string runtime_dir = PathUtils::Join({test_work_dir_, "fake_runtime"});
      const std::string build_dir = PathUtils::Join({runtime_dir, "build"});
      const std::string so_path = PathUtils::Join({runtime_dir, "libg1_om2.so"});
      const std::string archive_constant_path = PathUtils::Join({test_work_dir_, "constant_0"});
      const std::string archive_constant_cfg_path = PathUtils::Join({test_work_dir_, "model_0_constants_config.json"});

      (void)PathUtils::RemoveDirectories(runtime_dir);
      ASSERT_EQ(CreateDir(runtime_dir), 0);
      WriteTextFile(PathUtils::Join({runtime_dir, "g1_interface.h"}), MakeInterfaceHeader());
      WriteTextFile(PathUtils::Join({runtime_dir, "g1_resources.cpp"}), MakeEmptyCpp("g1_interface.h"));
      WriteTextFile(PathUtils::Join({runtime_dir, "g1_kernel_reg.cpp"}), MakeEmptyCpp("g1_interface.h"));
      WriteTextFile(PathUtils::Join({runtime_dir, "g1_args_manager.cpp"}), MakeEmptyCpp("g1_interface.h"));
      WriteTextFile(PathUtils::Join({runtime_dir, "g1_load_and_run.cpp"}), MakeLoadAndRunCpp());
      WriteTextFile(PathUtils::Join({runtime_dir, "CMakeLists.txt"}), MakeCMakeLists());

      const std::string cmake_config_cmd = "cmake -S " + runtime_dir + " -B " + build_dir;
      const std::string cmake_build_cmd = "cmake --build " + build_dir + " -j1";
      RunCommandOrAssert(cmake_config_cmd);
      RunCommandOrAssert(cmake_build_cmd);
      ASSERT_EQ(mmAccess2(so_path.c_str(), M_F_OK), EOK);

      WriteBinaryFile(archive_constant_path,
                      std::vector<uint8_t>{1U, 2U, 3U, 4U, 5U, 6U, 7U, 8U, 9U, 10U, 11U, 12U, 13U, 14U, 15U, 16U});
      WriteTextFile(archive_constant_cfg_path, MakeConstantsConfigJson());

      ZipArchiveWriter zip_writer(om2_file_path_);
      ASSERT_TRUE(zip_writer.IsMemFileOpened());
      const auto manifest = MakeManifestJson();
      const auto model_meta = MakeModelMetaJson();
      ASSERT_TRUE(zip_writer.WriteBytes("manifest.json", manifest.data(), manifest.size(), false));
      ASSERT_TRUE(zip_writer.WriteBytes("data/model_0/model_meta.json", model_meta.data(), model_meta.size(), false));
      ASSERT_TRUE(zip_writer.WriteFile("data/model_0/runtime/CMakeLists.txt",
                                       PathUtils::Join({runtime_dir, "CMakeLists.txt"}), false));
      ASSERT_TRUE(zip_writer.WriteFile("data/model_0/runtime/g1_interface.h",
                                       PathUtils::Join({runtime_dir, "g1_interface.h"}), false));
      ASSERT_TRUE(zip_writer.WriteFile("data/model_0/runtime/g1_resources.cpp",
                                       PathUtils::Join({runtime_dir, "g1_resources.cpp"}), false));
      ASSERT_TRUE(zip_writer.WriteFile("data/model_0/runtime/g1_kernel_reg.cpp",
                                       PathUtils::Join({runtime_dir, "g1_kernel_reg.cpp"}), false));
      ASSERT_TRUE(zip_writer.WriteFile("data/model_0/runtime/g1_args_manager.cpp",
                                       PathUtils::Join({runtime_dir, "g1_args_manager.cpp"}), false));
      ASSERT_TRUE(zip_writer.WriteFile("data/model_0/runtime/g1_load_and_run.cpp",
                                       PathUtils::Join({runtime_dir, "g1_load_and_run.cpp"}), false));
      ASSERT_TRUE(zip_writer.WriteFile("data/model_0/runtime/libg1_om2.so", so_path, false));
      ASSERT_TRUE(zip_writer.WriteFile("data/constants/constant_0", archive_constant_path, false));
      ASSERT_TRUE(
          zip_writer.WriteFile("data/constants/model_0_constants_config.json", archive_constant_cfg_path, false));
      ASSERT_TRUE(zip_writer.SaveModelDataToFile());
      ASSERT_EQ(mmAccess2(om2_file_path_.c_str(), M_F_OK), EOK);
    });
  }

  static void PrepareFileConstOm2File() {
    std::call_once(prepare_fileconst_once_, []() {
      const std::string runtime_dir = PathUtils::Join({test_work_dir_, "fake_runtime_fileconst"});
      const std::string build_dir = PathUtils::Join({runtime_dir, "build"});
      const std::string so_path = PathUtils::Join({runtime_dir, "libg1_om2.so"});
      const std::string archive_constant_cfg_path = PathUtils::Join({test_work_dir_, "model_1_constants_config.json"});
      const std::string weight_dir = PathUtils::Join({test_work_dir_, "weight"});
      const std::string file_const_path = PathUtils::Join({weight_dir, "fc.bin"});

      (void)PathUtils::RemoveDirectories(runtime_dir);
      ASSERT_EQ(CreateDir(runtime_dir), 0);
      WriteTextFile(PathUtils::Join({runtime_dir, "g1_interface.h"}), MakeInterfaceHeader());
      WriteTextFile(PathUtils::Join({runtime_dir, "g1_resources.cpp"}), MakeEmptyCpp("g1_interface.h"));
      WriteTextFile(PathUtils::Join({runtime_dir, "g1_kernel_reg.cpp"}), MakeEmptyCpp("g1_interface.h"));
      WriteTextFile(PathUtils::Join({runtime_dir, "g1_args_manager.cpp"}), MakeEmptyCpp("g1_interface.h"));
      WriteTextFile(PathUtils::Join({runtime_dir, "g1_load_and_run.cpp"}), MakeLoadAndRunCpp());
      WriteTextFile(PathUtils::Join({runtime_dir, "CMakeLists.txt"}), MakeCMakeLists());

      const std::string cmake_config_cmd = "cmake -S " + runtime_dir + " -B " + build_dir;
      const std::string cmake_build_cmd = "cmake --build " + build_dir + " -j1";
      RunCommandOrAssert(cmake_config_cmd);
      RunCommandOrAssert(cmake_build_cmd);
      ASSERT_EQ(mmAccess2(so_path.c_str(), M_F_OK), EOK);

      WriteTextFile(archive_constant_cfg_path, MakeIndividualConstantsConfigJson());
      WriteBinaryFile(file_const_path, {21U, 22U, 23U, 24U});

      ZipArchiveWriter zip_writer(om2_fileconst_file_path_);
      ASSERT_TRUE(zip_writer.IsMemFileOpened());
      const auto manifest = MakeManifestJson();
      const auto model_meta = MakeModelMetaJson();
      ASSERT_TRUE(zip_writer.WriteBytes("manifest.json", manifest.data(), manifest.size(), false));
      ASSERT_TRUE(zip_writer.WriteBytes("data/model_0/model_meta.json", model_meta.data(), model_meta.size(), false));
      ASSERT_TRUE(zip_writer.WriteFile("data/model_0/runtime/CMakeLists.txt",
                                       PathUtils::Join({runtime_dir, "CMakeLists.txt"}), false));
      ASSERT_TRUE(zip_writer.WriteFile("data/model_0/runtime/g1_interface.h",
                                       PathUtils::Join({runtime_dir, "g1_interface.h"}), false));
      ASSERT_TRUE(zip_writer.WriteFile("data/model_0/runtime/g1_resources.cpp",
                                       PathUtils::Join({runtime_dir, "g1_resources.cpp"}), false));
      ASSERT_TRUE(zip_writer.WriteFile("data/model_0/runtime/g1_kernel_reg.cpp",
                                       PathUtils::Join({runtime_dir, "g1_kernel_reg.cpp"}), false));
      ASSERT_TRUE(zip_writer.WriteFile("data/model_0/runtime/g1_args_manager.cpp",
                                       PathUtils::Join({runtime_dir, "g1_args_manager.cpp"}), false));
      ASSERT_TRUE(zip_writer.WriteFile("data/model_0/runtime/g1_load_and_run.cpp",
                                       PathUtils::Join({runtime_dir, "g1_load_and_run.cpp"}), false));
      ASSERT_TRUE(zip_writer.WriteFile("data/model_0/runtime/libg1_om2.so", so_path, false));
      ASSERT_TRUE(
          zip_writer.WriteFile("data/constants/model_1_constants_config.json", archive_constant_cfg_path, false));
      ASSERT_TRUE(zip_writer.SaveModelDataToFile());
      ASSERT_EQ(mmAccess2(om2_fileconst_file_path_.c_str(), M_F_OK), EOK);
    });
  }

  static void PrepareCombinedFileConstOm2File() {
    std::call_once(prepare_combined_once_, []() {
      const std::string runtime_dir = PathUtils::Join({test_work_dir_, "fake_runtime_combined"});
      const std::string build_dir = PathUtils::Join({runtime_dir, "build"});
      const std::string so_path = PathUtils::Join({runtime_dir, "libg1_om2.so"});
      const std::string archive_constant_cfg_path = PathUtils::Join({test_work_dir_, "model_2_constants_config.json"});
      const std::string weight_dir = PathUtils::Join({test_work_dir_, "weight"});
      const std::string file_const_path = PathUtils::Join({weight_dir, "combined.bin"});

      (void)PathUtils::RemoveDirectories(runtime_dir);
      ASSERT_EQ(CreateDir(runtime_dir), 0);
      WriteTextFile(PathUtils::Join({runtime_dir, "g1_interface.h"}), MakeInterfaceHeader());
      WriteTextFile(PathUtils::Join({runtime_dir, "g1_resources.cpp"}), MakeEmptyCpp("g1_interface.h"));
      WriteTextFile(PathUtils::Join({runtime_dir, "g1_kernel_reg.cpp"}), MakeEmptyCpp("g1_interface.h"));
      WriteTextFile(PathUtils::Join({runtime_dir, "g1_args_manager.cpp"}), MakeEmptyCpp("g1_interface.h"));
      WriteTextFile(PathUtils::Join({runtime_dir, "g1_load_and_run.cpp"}), MakeLoadAndRunCpp());
      WriteTextFile(PathUtils::Join({runtime_dir, "CMakeLists.txt"}), MakeCMakeLists());

      const std::string cmake_config_cmd = "cmake -S " + runtime_dir + " -B " + build_dir;
      const std::string cmake_build_cmd = "cmake --build " + build_dir + " -j1";
      RunCommandOrAssert(cmake_config_cmd);
      RunCommandOrAssert(cmake_build_cmd);
      ASSERT_EQ(mmAccess2(so_path.c_str(), M_F_OK), EOK);

      WriteTextFile(archive_constant_cfg_path, MakeCombinedConstantsConfigJson());
      WriteBinaryFile(file_const_path, {41U, 42U, 43U, 44U});

      ZipArchiveWriter zip_writer(om2_combined_file_path_);
      ASSERT_TRUE(zip_writer.IsMemFileOpened());
      const auto manifest = MakeManifestJson();
      const auto model_meta = MakeModelMetaJson();
      ASSERT_TRUE(zip_writer.WriteBytes("manifest.json", manifest.data(), manifest.size(), false));
      ASSERT_TRUE(zip_writer.WriteBytes("data/model_0/model_meta.json", model_meta.data(), model_meta.size(), false));
      ASSERT_TRUE(zip_writer.WriteFile("data/model_0/runtime/CMakeLists.txt",
                                       PathUtils::Join({runtime_dir, "CMakeLists.txt"}), false));
      ASSERT_TRUE(zip_writer.WriteFile("data/model_0/runtime/g1_interface.h",
                                       PathUtils::Join({runtime_dir, "g1_interface.h"}), false));
      ASSERT_TRUE(zip_writer.WriteFile("data/model_0/runtime/g1_resources.cpp",
                                       PathUtils::Join({runtime_dir, "g1_resources.cpp"}), false));
      ASSERT_TRUE(zip_writer.WriteFile("data/model_0/runtime/g1_kernel_reg.cpp",
                                       PathUtils::Join({runtime_dir, "g1_kernel_reg.cpp"}), false));
      ASSERT_TRUE(zip_writer.WriteFile("data/model_0/runtime/g1_args_manager.cpp",
                                       PathUtils::Join({runtime_dir, "g1_args_manager.cpp"}), false));
      ASSERT_TRUE(zip_writer.WriteFile("data/model_0/runtime/g1_load_and_run.cpp",
                                       PathUtils::Join({runtime_dir, "g1_load_and_run.cpp"}), false));
      ASSERT_TRUE(zip_writer.WriteFile("data/model_0/runtime/libg1_om2.so", so_path, false));
      ASSERT_TRUE(
          zip_writer.WriteFile("data/constants/model_2_constants_config.json", archive_constant_cfg_path, false));
      ASSERT_TRUE(zip_writer.SaveModelDataToFile());
      ASSERT_EQ(mmAccess2(om2_combined_file_path_.c_str(), M_F_OK), EOK);
    });
  }

  static void PrepareMixedOm2File() {
    std::call_once(prepare_mixed_once_, []() {
      const std::string runtime_dir = PathUtils::Join({test_work_dir_, "fake_runtime_mixed"});
      const std::string build_dir = PathUtils::Join({runtime_dir, "build"});
      const std::string so_path = PathUtils::Join({runtime_dir, "libg1_om2.so"});
      const std::string archive_constant_path = PathUtils::Join({test_work_dir_, "constant_mixed_0"});
      const std::string archive_constant_cfg_path = PathUtils::Join({test_work_dir_, "model_3_constants_config.json"});
      const std::string weight_dir = PathUtils::Join({test_work_dir_, "weight"});
      const std::string individual_path = PathUtils::Join({weight_dir, "mixed_fc.bin"});
      const std::string combined_path = PathUtils::Join({weight_dir, "mixed_combined.bin"});

      (void)PathUtils::RemoveDirectories(runtime_dir);
      ASSERT_EQ(CreateDir(runtime_dir), 0);
      WriteTextFile(PathUtils::Join({runtime_dir, "g1_interface.h"}), MakeInterfaceHeader());
      WriteTextFile(PathUtils::Join({runtime_dir, "g1_resources.cpp"}), MakeEmptyCpp("g1_interface.h"));
      WriteTextFile(PathUtils::Join({runtime_dir, "g1_kernel_reg.cpp"}), MakeEmptyCpp("g1_interface.h"));
      WriteTextFile(PathUtils::Join({runtime_dir, "g1_args_manager.cpp"}), MakeEmptyCpp("g1_interface.h"));
      WriteTextFile(PathUtils::Join({runtime_dir, "g1_load_and_run.cpp"}), MakeLoadAndRunCpp());
      WriteTextFile(PathUtils::Join({runtime_dir, "CMakeLists.txt"}), MakeCMakeLists());

      const std::string cmake_config_cmd = "cmake -S " + runtime_dir + " -B " + build_dir;
      const std::string cmake_build_cmd = "cmake --build " + build_dir + " -j1";
      RunCommandOrAssert(cmake_config_cmd);
      RunCommandOrAssert(cmake_build_cmd);
      ASSERT_EQ(mmAccess2(so_path.c_str(), M_F_OK), EOK);

      WriteBinaryFile(archive_constant_path,
                      std::vector<uint8_t>{1U, 2U, 3U, 4U, 5U, 6U, 7U, 8U, 9U, 10U, 11U, 12U, 13U, 14U, 15U, 16U});
      WriteTextFile(archive_constant_cfg_path, MakeMixedConstantsConfigJson());
      WriteBinaryFile(individual_path, {61U, 62U, 63U, 64U});
      WriteBinaryFile(combined_path, {71U, 72U, 73U, 74U});

      ZipArchiveWriter zip_writer(om2_mixed_file_path_);
      ASSERT_TRUE(zip_writer.IsMemFileOpened());
      const auto manifest = MakeManifestJson();
      const auto model_meta = MakeModelMetaJson();
      ASSERT_TRUE(zip_writer.WriteBytes("manifest.json", manifest.data(), manifest.size(), false));
      ASSERT_TRUE(zip_writer.WriteBytes("data/model_0/model_meta.json", model_meta.data(), model_meta.size(), false));
      ASSERT_TRUE(zip_writer.WriteFile("data/model_0/runtime/CMakeLists.txt",
                                       PathUtils::Join({runtime_dir, "CMakeLists.txt"}), false));
      ASSERT_TRUE(zip_writer.WriteFile("data/model_0/runtime/g1_interface.h",
                                       PathUtils::Join({runtime_dir, "g1_interface.h"}), false));
      ASSERT_TRUE(zip_writer.WriteFile("data/model_0/runtime/g1_resources.cpp",
                                       PathUtils::Join({runtime_dir, "g1_resources.cpp"}), false));
      ASSERT_TRUE(zip_writer.WriteFile("data/model_0/runtime/g1_kernel_reg.cpp",
                                       PathUtils::Join({runtime_dir, "g1_kernel_reg.cpp"}), false));
      ASSERT_TRUE(zip_writer.WriteFile("data/model_0/runtime/g1_args_manager.cpp",
                                       PathUtils::Join({runtime_dir, "g1_args_manager.cpp"}), false));
      ASSERT_TRUE(zip_writer.WriteFile("data/model_0/runtime/g1_load_and_run.cpp",
                                       PathUtils::Join({runtime_dir, "g1_load_and_run.cpp"}), false));
      ASSERT_TRUE(zip_writer.WriteFile("data/model_0/runtime/libg1_om2.so", so_path, false));
      ASSERT_TRUE(zip_writer.WriteFile("data/constants/constant_0", archive_constant_path, false));
      ASSERT_TRUE(
          zip_writer.WriteFile("data/constants/model_3_constants_config.json", archive_constant_cfg_path, false));
      ASSERT_TRUE(zip_writer.SaveModelDataToFile());
      ASSERT_EQ(mmAccess2(om2_mixed_file_path_.c_str(), M_F_OK), EOK);
    });
  }

  static void PrepareDuplicateIndividualOm2File() {
    std::call_once(prepare_duplicate_individual_once_, []() {
      const std::string runtime_dir = PathUtils::Join({test_work_dir_, "fake_runtime_duplicate_individual"});
      const std::string build_dir = PathUtils::Join({runtime_dir, "build"});
      const std::string so_path = PathUtils::Join({runtime_dir, "libg1_om2.so"});
      const std::string archive_constant_cfg_path =
          PathUtils::Join({test_work_dir_, "model_duplicate_individual_constants_config.json"});
      const std::string weight_dir = PathUtils::Join({test_work_dir_, "weight"});
      const std::string individual_path = PathUtils::Join({weight_dir, "duplicate_fc.bin"});

      (void)PathUtils::RemoveDirectories(runtime_dir);
      ASSERT_EQ(CreateDir(runtime_dir), 0);
      WriteTextFile(PathUtils::Join({runtime_dir, "g1_interface.h"}), MakeInterfaceHeader());
      WriteTextFile(PathUtils::Join({runtime_dir, "g1_resources.cpp"}), MakeEmptyCpp("g1_interface.h"));
      WriteTextFile(PathUtils::Join({runtime_dir, "g1_kernel_reg.cpp"}), MakeEmptyCpp("g1_interface.h"));
      WriteTextFile(PathUtils::Join({runtime_dir, "g1_args_manager.cpp"}), MakeEmptyCpp("g1_interface.h"));
      WriteTextFile(PathUtils::Join({runtime_dir, "g1_load_and_run.cpp"}), MakeLoadAndRunCpp());
      WriteTextFile(PathUtils::Join({runtime_dir, "CMakeLists.txt"}), MakeCMakeLists());

      const std::string cmake_config_cmd = "cmake -S " + runtime_dir + " -B " + build_dir;
      const std::string cmake_build_cmd = "cmake --build " + build_dir + " -j1";
      RunCommandOrAssert(cmake_config_cmd);
      RunCommandOrAssert(cmake_build_cmd);
      ASSERT_EQ(mmAccess2(so_path.c_str(), M_F_OK), EOK);

      WriteTextFile(archive_constant_cfg_path, MakeDuplicateIndividualConstantsConfigJson());
      WriteBinaryFile(individual_path, {101U, 102U, 103U, 104U});

      ZipArchiveWriter zip_writer(om2_duplicate_individual_file_path_);
      ASSERT_TRUE(zip_writer.IsMemFileOpened());
      const auto manifest = MakeManifestJson();
      const auto model_meta = MakeModelMetaJson();
      ASSERT_TRUE(zip_writer.WriteBytes("manifest.json", manifest.data(), manifest.size(), false));
      ASSERT_TRUE(zip_writer.WriteBytes("data/model_0/model_meta.json", model_meta.data(), model_meta.size(), false));
      ASSERT_TRUE(zip_writer.WriteFile("data/model_0/runtime/CMakeLists.txt",
                                       PathUtils::Join({runtime_dir, "CMakeLists.txt"}), false));
      ASSERT_TRUE(zip_writer.WriteFile("data/model_0/runtime/g1_interface.h",
                                       PathUtils::Join({runtime_dir, "g1_interface.h"}), false));
      ASSERT_TRUE(zip_writer.WriteFile("data/model_0/runtime/g1_resources.cpp",
                                       PathUtils::Join({runtime_dir, "g1_resources.cpp"}), false));
      ASSERT_TRUE(zip_writer.WriteFile("data/model_0/runtime/g1_kernel_reg.cpp",
                                       PathUtils::Join({runtime_dir, "g1_kernel_reg.cpp"}), false));
      ASSERT_TRUE(zip_writer.WriteFile("data/model_0/runtime/g1_args_manager.cpp",
                                       PathUtils::Join({runtime_dir, "g1_args_manager.cpp"}), false));
      ASSERT_TRUE(zip_writer.WriteFile("data/model_0/runtime/g1_load_and_run.cpp",
                                       PathUtils::Join({runtime_dir, "g1_load_and_run.cpp"}), false));
      ASSERT_TRUE(zip_writer.WriteFile("data/model_0/runtime/libg1_om2.so", so_path, false));
      ASSERT_TRUE(zip_writer.WriteFile("data/constants/model_duplicate_individual_constants_config.json",
                                       archive_constant_cfg_path, false));
      ASSERT_TRUE(zip_writer.SaveModelDataToFile());
      ASSERT_EQ(mmAccess2(om2_duplicate_individual_file_path_.c_str(), M_F_OK), EOK);
    });
  }

  static ModelDataHolder LoadValidModelData() {
    PrepareOm2File();
    uint32_t model_buf_size = 0U;
    auto model_buf = GetBinDataFromFile(om2_file_path_, model_buf_size);
    EXPECT_NE(model_buf, nullptr);
    EXPECT_GT(model_buf_size, 0U);

    ModelDataHolder holder;
    holder.model_data.model_data = model_buf.get();
    holder.model_data.model_len = model_buf_size;
    holder.model_data.om_path = om2_file_path_;
    holder.buffer = std::move(model_buf);
    return holder;
  }

  static ModelDataHolder LoadValidFileConstModelData() {
    PrepareFileConstOm2File();
    uint32_t model_buf_size = 0U;
    auto model_buf = GetBinDataFromFile(om2_fileconst_file_path_, model_buf_size);
    EXPECT_NE(model_buf, nullptr);
    EXPECT_GT(model_buf_size, 0U);

    ModelDataHolder holder;
    holder.model_data.model_data = model_buf.get();
    holder.model_data.model_len = model_buf_size;
    holder.model_data.om_path = om2_fileconst_file_path_;
    holder.buffer = std::move(model_buf);
    return holder;
  }

  static ModelDataHolder LoadValidCombinedModelData() {
    PrepareCombinedFileConstOm2File();
    uint32_t model_buf_size = 0U;
    auto model_buf = GetBinDataFromFile(om2_combined_file_path_, model_buf_size);
    EXPECT_NE(model_buf, nullptr);
    EXPECT_GT(model_buf_size, 0U);

    ModelDataHolder holder;
    holder.model_data.model_data = model_buf.get();
    holder.model_data.model_len = model_buf_size;
    holder.model_data.om_path = om2_combined_file_path_;
    holder.buffer = std::move(model_buf);
    return holder;
  }

  static ModelDataHolder LoadValidMixedModelData() {
    PrepareMixedOm2File();
    uint32_t model_buf_size = 0U;
    auto model_buf = GetBinDataFromFile(om2_mixed_file_path_, model_buf_size);
    EXPECT_NE(model_buf, nullptr);
    EXPECT_GT(model_buf_size, 0U);

    ModelDataHolder holder;
    holder.model_data.model_data = model_buf.get();
    holder.model_data.model_len = model_buf_size;
    holder.model_data.om_path = om2_mixed_file_path_;
    holder.buffer = std::move(model_buf);
    return holder;
  }

  static ModelDataHolder LoadValidDuplicateIndividualModelData() {
    PrepareDuplicateIndividualOm2File();
    uint32_t model_buf_size = 0U;
    auto model_buf = GetBinDataFromFile(om2_duplicate_individual_file_path_, model_buf_size);
    EXPECT_NE(model_buf, nullptr);
    EXPECT_GT(model_buf_size, 0U);

    ModelDataHolder holder;
    holder.model_data.model_data = model_buf.get();
    holder.model_data.model_len = model_buf_size;
    holder.model_data.om_path = om2_duplicate_individual_file_path_;
    holder.buffer = std::move(model_buf);
    return holder;
  }

  static void ConstructIoTensors(std::vector<gert::Tensor> &input_tensors, std::vector<gert::Tensor> &output_tensors,
                                 std::vector<gert::Tensor *> &inputs, std::vector<gert::Tensor *> &outputs) {
    input_tensors.resize(2);
    output_tensors.resize(1);
    TensorCheckUtils::ConstructGertTensor(input_tensors[0], {2, 16}, DataType::DT_FLOAT, Format::FORMAT_ND);
    TensorCheckUtils::ConstructGertTensor(input_tensors[1], {2, 16}, DataType::DT_FLOAT, Format::FORMAT_ND);
    TensorCheckUtils::ConstructGertTensor(output_tensors[0], {2, 16}, DataType::DT_FLOAT, Format::FORMAT_ND);

    inputs = {&input_tensors[0], &input_tensors[1]};
    outputs = {&output_tensors[0]};
  }

  static ModelDataHolder MakeVariableArchiveModelData(const std::string &case_suffix,
                                                      const std::string &var_resource_json,
                                                      const std::vector<uint8_t> *var_weight_data,
                                                      const std::string &variables_config_json,
                                                      const VarWeightOrder weight_order) {
    PrepareOm2File();
    ge::ModelBufferData model_buf;
    const auto om_path = PathUtils::Join({test_work_dir_, "variable_archive_" + case_suffix + ".om2"});
    const auto so_path = PathUtils::Join({test_work_dir_, "fake_runtime", "libg1_om2.so"});
    ZipArchiveWriter zip_writer(om_path);
    EXPECT_TRUE(zip_writer.IsMemFileOpened());
    const auto model_meta = MakeModelMetaJson();
    EXPECT_TRUE(zip_writer.WriteBytes("data/model_0/model_meta.json", model_meta.data(), model_meta.size(), false));
    EXPECT_TRUE(zip_writer.WriteFile("data/model_0/runtime/libg1_om2.so", so_path, false));

    const auto write_weight = [&]() {
      EXPECT_TRUE(zip_writer.WriteBytes("data/variables/var_weight_data", var_weight_data->data(),
                                        var_weight_data->size(), false));
    };
    if (var_weight_data != nullptr && weight_order == VarWeightOrder::kBeforeResource) {
      write_weight();
    }
    EXPECT_TRUE(zip_writer.WriteBytes("data/variables/var_resource.json", var_resource_json.data(),
                                      var_resource_json.size(), false));
    if (var_weight_data != nullptr && weight_order == VarWeightOrder::kAfterResource) {
      write_weight();
    }
    EXPECT_TRUE(zip_writer.WriteBytes("data/variables/model_0_variables_config.json", variables_config_json.data(),
                                      variables_config_json.size(), false));
    EXPECT_TRUE(zip_writer.SaveModelData(model_buf, false));

    ModelDataHolder holder;
    holder.model_data.model_data = model_buf.data.get();
    holder.model_data.model_len = model_buf.length;
    holder.model_data.om_path = om_path;
    holder.shared_buffer = model_buf.data;
    return holder;
  }

  static std::string test_work_dir_;
  static std::string om2_file_path_;
  static std::string om2_fileconst_file_path_;
  static std::string om2_combined_file_path_;
  static std::string om2_mixed_file_path_;
  static std::string om2_duplicate_individual_file_path_;
  static std::once_flag prepare_once_;
  static std::once_flag prepare_fileconst_once_;
  static std::once_flag prepare_combined_once_;
  static std::once_flag prepare_mixed_once_;
  static std::once_flag prepare_duplicate_individual_once_;
};

std::string Om2ModelExecutorUt::test_work_dir_;
std::string Om2ModelExecutorUt::om2_file_path_;
std::string Om2ModelExecutorUt::om2_fileconst_file_path_;
std::string Om2ModelExecutorUt::om2_combined_file_path_;
std::string Om2ModelExecutorUt::om2_mixed_file_path_;
std::string Om2ModelExecutorUt::om2_duplicate_individual_file_path_;
std::once_flag Om2ModelExecutorUt::prepare_once_;
std::once_flag Om2ModelExecutorUt::prepare_fileconst_once_;
std::once_flag Om2ModelExecutorUt::prepare_combined_once_;
std::once_flag Om2ModelExecutorUt::prepare_mixed_once_;
std::once_flag Om2ModelExecutorUt::prepare_duplicate_individual_once_;

TEST_F(Om2ModelExecutorUt, load_invalid_model_data) {
  gert::Om2ModelExecutor executor;
  ModelData invalid_model_data{};
  auto load_arg = MakeOm2LoadArg();
  EXPECT_NE(executor.Load(invalid_model_data, load_arg, 1U), SUCCESS);
}

TEST_F(Om2ModelExecutorUt, load_ok) {
  auto model_data_holder = LoadValidModelData();
  gert::Om2ModelExecutor executor;
  auto load_arg = MakeOm2LoadArg();
  EXPECT_EQ(executor.Load(model_data_holder.model_data, load_arg, 1U), SUCCESS);
}

TEST_F(Om2ModelExecutorUt, load_ok_with_zip_archive_writer_base_name_prefix) {
  PrepareOm2File();
  ge::ModelBufferData model_buf;
  const std::string om2_mem_path = PathUtils::Join({test_work_dir_, "load_with_base_prefix.om2"});
  const std::string runtime_dir = PathUtils::Join({test_work_dir_, "fake_runtime"});
  const std::string so_path = PathUtils::Join({runtime_dir, "libg1_om2.so"});
  const std::string archive_constant_path = PathUtils::Join({test_work_dir_, "constant_0"});
  const std::string archive_constant_cfg_path = PathUtils::Join({test_work_dir_, "model_0_constants_config.json"});

  ZipArchiveWriter zip_writer(om2_mem_path);
  ASSERT_TRUE(zip_writer.IsMemFileOpened());
  const auto manifest = MakeManifestJson();
  const auto model_meta = MakeModelMetaJson();
  ASSERT_TRUE(zip_writer.WriteBytes("manifest.json", manifest.data(), manifest.size(), false));
  ASSERT_TRUE(zip_writer.WriteBytes("data/model_0/model_meta.json", model_meta.data(), model_meta.size(), false));
  ASSERT_TRUE(zip_writer.WriteFile("data/model_0/runtime/libg1_om2.so", so_path, false));
  ASSERT_TRUE(zip_writer.WriteFile("data/constants/constant_0", archive_constant_path, false));
  ASSERT_TRUE(zip_writer.WriteFile("data/constants/model_0_constants_config.json", archive_constant_cfg_path, false));
  ASSERT_TRUE(zip_writer.SaveModelData(model_buf, false));
  ASSERT_NE(model_buf.data, nullptr);
  ASSERT_GT(model_buf.length, 0U);

  ModelDataHolder holder;
  holder.model_data.model_data = model_buf.data.get();
  holder.model_data.model_len = model_buf.length;
  holder.model_data.om_path = om2_mem_path;
  holder.shared_buffer = model_buf.data;

  gert::Om2ModelExecutor executor;
  const auto load_arg = MakeOm2LoadArg();
  EXPECT_EQ(executor.Load(holder.model_data, load_arg, 1U), SUCCESS);
}

TEST_F(Om2ModelExecutorUt, load_deserializes_variable_entries_regardless_of_weight_order) {
  const auto resource_json = MakeVarResourceJson(0U, 4U);
  const auto config_json = MakeVariablesConfigJson();
  const std::vector<uint8_t> weight{1U, 2U, 3U, 4U};
  const std::vector<std::pair<const char *, VarWeightOrder>> cases{
      {"weight_before", VarWeightOrder::kBeforeResource},
      {"weight_after", VarWeightOrder::kAfterResource},
  };
  uint64_t session_id = 1001U;
  for (const auto &[suffix, order] : cases) {
    auto holder = MakeVariableArchiveModelData(suffix, resource_json, &weight, config_json, order);
    ASSERT_EQ(setenv("OM2_EXPECT_VAR0_MODE", "NON_NULL", 1), 0);
    VarInitDataRecordingAclRuntimeStub runtime_stub;
    AclRuntimeStubGuard runtime_stub_guard(&runtime_stub);
    gert::Om2ModelExecutor executor;
    ASSERT_EQ(executor.Load(holder.model_data, MakeOm2LoadArg(), session_id++), SUCCESS) << suffix;
    ASSERT_EQ(runtime_stub.memcpy_records.size(), 1U) << suffix;
    EXPECT_EQ(runtime_stub.memcpy_records[0].dest_max, weight.size()) << suffix;
    EXPECT_EQ(runtime_stub.memcpy_records[0].data, weight) << suffix;
    EXPECT_EQ(runtime_stub.memcpy_records[0].kind, ACL_MEMCPY_HOST_TO_DEVICE) << suffix;
  }
}

TEST_F(Om2ModelExecutorUt, load_skips_invalid_variable_init_data) {
  struct VariableWeightCase {
    const char *suffix;
    size_t offset;
    size_t size;
    const std::vector<uint8_t> *weight;
    VarWeightOrder order;
    uint64_t session_id;
  };

  const auto config_json = MakeVariablesConfigJson();
  const std::vector<uint8_t> short_weight{1U, 2U};
  const std::vector<VariableWeightCase> cases{
      {"missing_weight", 0U, 4U, nullptr, VarWeightOrder::kAbsent, 1003U},
      {"size_out_of_range", 1U, 4U, &short_weight, VarWeightOrder::kAfterResource, 1004U},
      {"offset_out_of_range", 3U, 1U, &short_weight, VarWeightOrder::kBeforeResource, 1005U},
  };
  for (const auto &test_case : cases) {
    const auto resource_json = MakeVarResourceJson(test_case.offset, test_case.size);
    auto holder =
        MakeVariableArchiveModelData(test_case.suffix, resource_json, test_case.weight, config_json, test_case.order);
    ASSERT_EQ(setenv("OM2_EXPECT_VAR0_MODE", "NON_NULL", 1), 0);
    VarInitDataRecordingAclRuntimeStub runtime_stub;
    AclRuntimeStubGuard runtime_stub_guard(&runtime_stub);
    gert::Om2ModelExecutor executor;
    ASSERT_EQ(executor.Load(holder.model_data, MakeOm2LoadArg(), test_case.session_id), SUCCESS) << test_case.suffix;
    EXPECT_TRUE(runtime_stub.memcpy_records.empty()) << test_case.suffix;
  }
}

TEST_F(Om2ModelExecutorUt, load_preserves_zero_copy_and_origin_input_dims_from_model_meta) {
  PrepareOm2File();
  ge::ModelBufferData model_buf;
  const std::string om2_mem_path = PathUtils::Join({test_work_dir_, "load_with_zero_copy.om2"});
  const std::string runtime_dir = PathUtils::Join({test_work_dir_, "fake_runtime"});
  const std::string so_path = PathUtils::Join({runtime_dir, "libg1_om2.so"});

  ZipArchiveWriter zip_writer(om2_mem_path);
  ASSERT_TRUE(zip_writer.IsMemFileOpened());
  const auto manifest = MakeManifestJson();
  const auto model_meta = MakeModelMetaJsonWithZeroCopySize();
  ASSERT_TRUE(zip_writer.WriteBytes("manifest.json", manifest.data(), manifest.size(), false));
  ASSERT_TRUE(zip_writer.WriteBytes("data/model_0/model_meta.json", model_meta.data(), model_meta.size(), false));
  ASSERT_TRUE(zip_writer.WriteFile("data/model_0/runtime/libg1_om2.so", so_path, false));
  ASSERT_TRUE(zip_writer.SaveModelData(model_buf, false));
  ASSERT_NE(model_buf.data, nullptr);
  ASSERT_GT(model_buf.length, 0U);

  ModelDataHolder holder;
  holder.model_data.model_data = model_buf.data.get();
  holder.model_data.model_len = model_buf.length;
  holder.model_data.om_path = om2_mem_path;
  holder.shared_buffer = model_buf.data;

  gert::Om2ModelExecutor executor;
  auto load_arg{MakeOm2LoadArg()};
  std::vector<uint8_t> external_work(1024U, 0U);
  load_arg.work_ptr = external_work.data();
  load_arg.work_size = external_work.size();
  ASSERT_EQ(executor.Load(holder.model_data, load_arg, 1U), SUCCESS);

  const std::vector<ge::Om2TensorDesc> *input_desc = nullptr;
  const std::vector<ge::Om2TensorDesc> *output_desc = nullptr;
  ASSERT_EQ(executor.GetModelDescInfo(input_desc, output_desc, false), SUCCESS);
  ASSERT_NE(input_desc, nullptr);
  ASSERT_EQ(input_desc->size(), 2U);
  EXPECT_EQ((*input_desc)[0].GetShape(), std::vector<int64_t>({1, 2, 3, 4}));
  EXPECT_EQ((*input_desc)[1].GetShape(), std::vector<int64_t>({1, 1, 224, 224}));

  const std::vector<ge::Om2TensorDesc> *input_desc_v2 = nullptr;
  const std::vector<ge::Om2TensorDesc> *output_desc_v2 = nullptr;
  ASSERT_EQ(executor.GetModelDescInfo(input_desc_v2, output_desc_v2, true), SUCCESS);
  ASSERT_NE(input_desc_v2, nullptr);
  ASSERT_EQ(input_desc_v2->size(), 2U);
  EXPECT_EQ((*input_desc_v2)[0].GetShape(), std::vector<int64_t>({1, 8, 3, 4}));
  EXPECT_EQ((*input_desc_v2)[1].GetShape(), std::vector<int64_t>({1, 1, 448, 224}));

  const auto &origin_input_dims = executor.GetOriginInputDims();
  ASSERT_EQ(origin_input_dims.size(), 2U);
  EXPECT_EQ(origin_input_dims[0], std::vector<int64_t>({1, -1, 3, 4}));
  EXPECT_EQ(origin_input_dims[1], std::vector<int64_t>({1, 1, -1, 224}));
}

TEST_F(Om2ModelExecutorUt, load_uses_external_work_ptr_when_size_is_enough) {
  auto model_data_holder = LoadValidModelData();
  gert::Om2ModelExecutor executor;
  auto load_arg = MakeOm2LoadArg();
  std::vector<uint8_t> external_work(4096U, 0U);
  load_arg.work_ptr = external_work.data();
  load_arg.work_size = external_work.size();

  EnvValueGuard guard_mode("OM2_EXPECT_WORK_PTR_MODE");
  EnvValueGuard guard_value("OM2_EXPECT_WORK_PTR_VALUE");
  ASSERT_EQ(setenv("OM2_EXPECT_WORK_PTR_MODE", "EQUAL", 1), 0);
  const std::string expected_ptr = PtrToHexString(external_work.data());
  ASSERT_EQ(setenv("OM2_EXPECT_WORK_PTR_VALUE", expected_ptr.c_str(), 1), 0);

  EXPECT_EQ(executor.Load(model_data_holder.model_data, load_arg, 1U), SUCCESS);
}

TEST_F(Om2ModelExecutorUt, load_runtime_so_without_creating_workspace) {
  auto model_data_holder = LoadValidModelData();
  const std::string ascend_work_path = PathUtils::Join({test_work_dir_, "load_without_workspace"});
  const std::string workspace_root = PathUtils::Join({ascend_work_path, ".ascend_temp/.tmp_om2_workspace"});
  (void)PathUtils::RemoveDirectories(ascend_work_path);
  ASSERT_EQ(CreateDir(ascend_work_path), 0);

  EnvValueGuard ascend_work_path_guard("ASCEND_WORK_PATH");
  ASSERT_EQ(setenv("ASCEND_WORK_PATH", ascend_work_path.c_str(), 1), 0);

  gert::Om2ModelExecutor executor;
  auto load_arg = MakeOm2LoadArg();
  ASSERT_EQ(executor.Load(model_data_holder.model_data, load_arg, 1U), SUCCESS);
  EXPECT_NE(mmAccess2(workspace_root.c_str(), M_F_OK), EOK);
}
TEST_F(Om2ModelExecutorUt, load_calls_model_load_after_model_create) {
  auto model_data_holder = LoadValidModelData();
  gert::Om2ModelExecutor executor;
  auto load_arg = MakeOm2LoadArg();
  const auto trace_file = PathUtils::Join({test_work_dir_, "dump_call_trace.txt"});
  (void)std::remove(trace_file.c_str());
  ASSERT_EQ(setenv("OM2_CALL_TRACE", trace_file.c_str(), 1), 0);

  EXPECT_EQ(executor.Load(model_data_holder.model_data, load_arg, 1U), SUCCESS);

  EXPECT_EQ(ReadTraceFile(trace_file), std::vector<std::string>({"create"}));
}

TEST_F(Om2ModelExecutorUt, load_fallbacks_root_graph_name_to_model_name_when_meta_missing) {
  const std::string om2_file_path = PathUtils::Join({test_work_dir_, "missing_root_graph_name.om2"});
  ZipArchiveWriter zip_writer(om2_file_path);
  ASSERT_TRUE(zip_writer.IsMemFileOpened());
  const auto manifest = MakeManifestJson();
  const auto model_meta = MakeModelMetaJsonWithoutRootGraphName();
  ASSERT_TRUE(zip_writer.WriteBytes("manifest.json", manifest.data(), manifest.size(), false));
  ASSERT_TRUE(zip_writer.WriteBytes("data/model_0/model_meta.json", model_meta.data(), model_meta.size(), false));
  ASSERT_TRUE(zip_writer.WriteFile("data/model_0/runtime/libg1_om2.so",
                                   PathUtils::Join({test_work_dir_, "fake_runtime", "libg1_om2.so"}), false));
  const std::string constants_config = "{}";
  ASSERT_TRUE(zip_writer.WriteBytes("data/constants/model_0_constants_config.json", constants_config.data(),
                                    constants_config.size(), false));
  ASSERT_TRUE(zip_writer.SaveModelDataToFile());
  uint32_t model_buf_size = 0U;
  auto model_buf = GetBinDataFromFile(om2_file_path, model_buf_size);
  ASSERT_NE(model_buf, nullptr);
  ModelDataHolder holder;
  holder.model_data.model_data = model_buf.get();
  holder.model_data.model_len = model_buf_size;
  holder.model_data.om_path = om2_file_path;
  holder.buffer = std::move(model_buf);

  gert::Om2ModelExecutor executor;
  auto load_arg = MakeOm2LoadArg();
  EXPECT_EQ(executor.Load(holder.model_data, load_arg, 1U), SUCCESS);
}

TEST_F(Om2ModelExecutorUt, load_failed_when_model_desc_is_invalid) {
  const std::string om2_file_path = PathUtils::Join({test_work_dir_, "invalid_model_desc.om2"});
  ZipArchiveWriter zip_writer(om2_file_path);
  ASSERT_TRUE(zip_writer.IsMemFileOpened());
  const auto manifest = MakeManifestJson();
  // Missing input shape should fail while parsing the cached model desc.
  const auto model_meta = MakeModelMetaJsonWithoutInputShape();
  ASSERT_TRUE(zip_writer.WriteBytes("manifest.json", manifest.data(), manifest.size(), false));
  ASSERT_TRUE(zip_writer.WriteBytes("data/model_0/model_meta.json", model_meta.data(), model_meta.size(), false));
  ASSERT_TRUE(zip_writer.WriteFile("data/model_0/runtime/libg1_om2.so",
                                   PathUtils::Join({test_work_dir_, "fake_runtime", "libg1_om2.so"}), false));
  const std::string constants_config = "{}";
  ASSERT_TRUE(zip_writer.WriteBytes("data/constants/model_0_constants_config.json", constants_config.data(),
                                    constants_config.size(), false));
  ASSERT_TRUE(zip_writer.SaveModelDataToFile());
  uint32_t model_buf_size = 0U;
  auto model_buf = GetBinDataFromFile(om2_file_path, model_buf_size);
  ASSERT_NE(model_buf, nullptr);
  ModelDataHolder holder;
  holder.model_data.model_data = model_buf.get();
  holder.model_data.model_len = model_buf_size;
  holder.model_data.om_path = om2_file_path;
  holder.buffer = std::move(model_buf);

  gert::Om2ModelExecutor executor;
  auto load_arg = MakeOm2LoadArg();
  EXPECT_NE(executor.Load(holder.model_data, load_arg, 1U), SUCCESS);
}

TEST_F(Om2ModelExecutorUt, load_generates_session_id_without_rt_session) {
  auto model_data_holder = LoadValidModelData();
  auto load_arg = MakeOm2LoadArg();
  ge::Status error_code = SUCCESS;
  ASSERT_EQ(setenv("OM2_EXPECT_SESSION_ID", "ANY", 1), 0);
  auto executor = gert::LoadOm2ExecutorFromData(model_data_holder.model_data, load_arg, error_code);
  EXPECT_EQ(error_code, SUCCESS);
  ASSERT_NE(executor, nullptr);
}

TEST_F(Om2ModelExecutorUt, load_uses_rt_session_id_when_rt_session_is_not_null) {
  auto model_data_holder = LoadValidModelData();
  auto load_arg = MakeOm2LoadArg();
  gert::RtSession rt_session(9527U);
  load_arg.rt_session = &rt_session;
  ge::Status error_code = SUCCESS;
  ASSERT_EQ(setenv("OM2_EXPECT_SESSION_ID", "9527", 1), 0);
  auto executor = gert::LoadOm2ExecutorFromData(model_data_holder.model_data, load_arg, error_code);
  EXPECT_EQ(error_code, SUCCESS);
  ASSERT_NE(executor, nullptr);
}

TEST_F(Om2ModelExecutorUt, load_passes_model_id_and_instance_handle_to_create) {
  auto model_data_holder = LoadValidModelData();
  auto load_arg = MakeOm2LoadArg();
  ge::Status error_code = SUCCESS;
  const auto expected_model_id = std::to_string(kTestModelId);
  ASSERT_EQ(setenv("OM2_EXPECT_MODEL_ID", expected_model_id.c_str(), 1), 0);
  ASSERT_EQ(setenv("OM2_EXPECT_INSTANCE_HANDLE_MODE", "NON_NULL", 1), 0);
  auto executor = gert::LoadOm2ExecutorFromData(model_data_holder.model_data, load_arg, error_code);
  EXPECT_EQ(error_code, SUCCESS);
  ASSERT_NE(executor, nullptr);
}

TEST_F(Om2ModelExecutorUt, load_failed_when_device_id_is_not_set) {
  auto model_data_holder = LoadValidModelData();
  gert::Om2ModelExecutor executor;
  auto load_arg = MakeOm2LoadArg();
  load_arg.device_id = -1;
  EXPECT_NE(executor.Load(model_data_holder.model_data, load_arg, 1U), SUCCESS);
}

TEST_F(Om2ModelExecutorUt, load_ok_with_external_work_ptr_and_internal_weight_from_archive) {
  auto model_data_holder = LoadValidModelData();
  gert::Om2ModelExecutor executor;
  auto load_arg = MakeOm2LoadArg();
  load_arg.work_ptr = reinterpret_cast<void *>(0x12345);
  load_arg.work_size = 2048U;
  ASSERT_EQ(setenv("OM2_EXPECT_WORK_PTR_MODE", "EQUAL", 1), 0);
  ASSERT_EQ(setenv("OM2_EXPECT_WORK_PTR_VALUE", PtrToHexString(load_arg.work_ptr).c_str(), 1), 0);
  ASSERT_EQ(setenv("OM2_EXPECT_CONST0_MODE", "NON_NULL", 1), 0);
  ASSERT_EQ(setenv("OM2_EXPECT_CONST0_FIRST_BYTE", "1", 1), 0);
  EXPECT_EQ(executor.Load(model_data_holder.model_data, load_arg, 1U), SUCCESS);
}

TEST_F(Om2ModelExecutorUt, load_ok_with_internal_work_ptr_and_external_device_weight) {
  auto model_data_holder = LoadValidModelData();
  gert::Om2ModelExecutor executor;
  std::vector<uint8_t> device_weight(16U, 0U);
  auto load_arg = MakeOm2LoadArg();
  load_arg.weight_ptr = device_weight.data();
  load_arg.weight_size = device_weight.size();
  ASSERT_EQ(setenv("OM2_EXPECT_WORK_PTR_MODE", "NON_NULL", 1), 0);
  ASSERT_EQ(setenv("OM2_EXPECT_CONST0_MODE", "NON_NULL", 1), 0);
  ASSERT_EQ(setenv("OM2_EXPECT_CONST0_FIRST_BYTE", "1", 1), 0);
  ASSERT_EQ(setenv("OM2_EXPECT_CONST0_PTR_MODE", "EQUAL", 1), 0);
  ASSERT_EQ(setenv("OM2_EXPECT_CONST0_PTR_VALUE", PtrToHexString(load_arg.weight_ptr).c_str(), 1), 0);
  EXPECT_EQ(executor.Load(model_data_holder.model_data, load_arg, 1U), SUCCESS);
}

TEST_F(Om2ModelExecutorUt, load_failed_when_external_device_weight_size_too_small) {
  auto model_data_holder = LoadValidModelData();
  gert::Om2ModelExecutor executor;
  std::vector<uint8_t> device_weight(8U, 0U);
  auto load_arg = MakeOm2LoadArg();
  load_arg.weight_ptr = device_weight.data();
  load_arg.weight_size = device_weight.size();
  EXPECT_NE(executor.Load(model_data_holder.model_data, load_arg, 1U), SUCCESS);
}

TEST_F(Om2ModelExecutorUt, load_ok_with_individual_fileconst_from_file) {
  auto model_data_holder = LoadValidFileConstModelData();
  gert::Om2ModelExecutor executor;
  auto load_arg = MakeOm2LoadArg();
  ASSERT_EQ(setenv("OM2_EXPECT_CONST0_MODE", "NON_NULL", 1), 0);
  ASSERT_EQ(setenv("OM2_EXPECT_CONST0_FIRST_BYTE", "22", 1), 0);
  EXPECT_EQ(executor.Load(model_data_holder.model_data, load_arg, 1U), SUCCESS);
}

TEST_F(Om2ModelExecutorUt, load_ok_with_individual_fileconst_from_user_mem) {
  auto model_data_holder = LoadValidFileConstModelData();
  gert::Om2ModelExecutor executor;
  std::vector<uint8_t> user_mem{31U, 32U, 33U, 34U};
  auto load_arg = MakeOm2LoadArg();
  load_arg.file_constant_mems.push_back({"fc.bin", user_mem.data(), user_mem.size()});
  ASSERT_EQ(setenv("OM2_EXPECT_CONST0_MODE", "NON_NULL", 1), 0);
  ASSERT_EQ(setenv("OM2_EXPECT_CONST0_FIRST_BYTE", "32", 1), 0);
  EXPECT_EQ(executor.Load(model_data_holder.model_data, load_arg, 1U), SUCCESS);
}

TEST_F(Om2ModelExecutorUt, load_ok_with_combined_fileconst_from_file) {
  auto model_data_holder = LoadValidCombinedModelData();
  gert::Om2ModelExecutor executor;
  auto load_arg = MakeOm2LoadArg();
  ASSERT_EQ(setenv("OM2_EXPECT_CONST0_MODE", "NON_NULL", 1), 0);
  ASSERT_EQ(setenv("OM2_EXPECT_CONST0_FIRST_BYTE", "42", 1), 0);
  EXPECT_EQ(executor.Load(model_data_holder.model_data, load_arg, 1U), SUCCESS);
}

TEST_F(Om2ModelExecutorUt, load_ok_with_combined_fileconst_from_user_mem) {
  auto model_data_holder = LoadValidCombinedModelData();
  gert::Om2ModelExecutor executor;
  std::vector<uint8_t> user_mem{51U, 52U, 53U, 54U};
  auto load_arg = MakeOm2LoadArg();
  load_arg.file_constant_mems.push_back({"combined.bin", user_mem.data(), user_mem.size()});
  ASSERT_EQ(setenv("OM2_EXPECT_CONST0_MODE", "NON_NULL", 1), 0);
  ASSERT_EQ(setenv("OM2_EXPECT_CONST0_FIRST_BYTE", "52", 1), 0);
  EXPECT_EQ(executor.Load(model_data_holder.model_data, load_arg, 1U), SUCCESS);
}

TEST_F(Om2ModelExecutorUt, load_ok_with_mixed_consts_from_file) {
  auto model_data_holder = LoadValidMixedModelData();
  gert::Om2ModelExecutor executor;
  auto load_arg = MakeOm2LoadArg();
  ASSERT_EQ(setenv("OM2_EXPECT_CONST0_MODE", "NON_NULL", 1), 0);
  ASSERT_EQ(setenv("OM2_EXPECT_CONST0_FIRST_BYTE", "1", 1), 0);
  ASSERT_EQ(setenv("OM2_EXPECT_CONST1_MODE", "NON_NULL", 1), 0);
  ASSERT_EQ(setenv("OM2_EXPECT_CONST1_FIRST_BYTE", "62", 1), 0);
  ASSERT_EQ(setenv("OM2_EXPECT_CONST2_MODE", "NON_NULL", 1), 0);
  ASSERT_EQ(setenv("OM2_EXPECT_CONST2_FIRST_BYTE", "72", 1), 0);
  EXPECT_EQ(executor.Load(model_data_holder.model_data, load_arg, 1U), SUCCESS);
}

TEST_F(Om2ModelExecutorUt, load_ok_with_mixed_consts_and_external_resources) {
  auto model_data_holder = LoadValidMixedModelData();
  gert::Om2ModelExecutor executor;
  std::vector<uint8_t> device_weight(16U, 0U);
  std::vector<uint8_t> individual_mem{80U, 81U, 82U, 83U};
  std::vector<uint8_t> combined_mem{90U, 91U, 92U, 93U};
  auto load_arg = MakeOm2LoadArg();
  load_arg.work_ptr = reinterpret_cast<void *>(0x34567);
  load_arg.work_size = 2048U;
  load_arg.weight_ptr = device_weight.data();
  load_arg.weight_size = device_weight.size();
  load_arg.file_constant_mems.push_back({"mixed_fc.bin", individual_mem.data(), individual_mem.size()});
  load_arg.file_constant_mems.push_back({"mixed_combined.bin", combined_mem.data(), combined_mem.size()});
  ASSERT_EQ(setenv("OM2_EXPECT_WORK_PTR_MODE", "EQUAL", 1), 0);
  ASSERT_EQ(setenv("OM2_EXPECT_WORK_PTR_VALUE", PtrToHexString(load_arg.work_ptr).c_str(), 1), 0);
  ASSERT_EQ(setenv("OM2_EXPECT_CONST0_MODE", "NON_NULL", 1), 0);
  ASSERT_EQ(setenv("OM2_EXPECT_CONST0_FIRST_BYTE", "1", 1), 0);
  ASSERT_EQ(setenv("OM2_EXPECT_CONST0_PTR_MODE", "EQUAL", 1), 0);
  ASSERT_EQ(setenv("OM2_EXPECT_CONST0_PTR_VALUE", PtrToHexString(load_arg.weight_ptr).c_str(), 1), 0);
  ASSERT_EQ(setenv("OM2_EXPECT_CONST1_MODE", "NON_NULL", 1), 0);
  ASSERT_EQ(setenv("OM2_EXPECT_CONST1_FIRST_BYTE", "81", 1), 0);
  ASSERT_EQ(setenv("OM2_EXPECT_CONST2_MODE", "NON_NULL", 1), 0);
  ASSERT_EQ(setenv("OM2_EXPECT_CONST2_FIRST_BYTE", "91", 1), 0);
  EXPECT_EQ(executor.Load(model_data_holder.model_data, load_arg, 1U), SUCCESS);
}

TEST_F(Om2ModelExecutorUt, load_reuses_duplicate_individual_fileconst_in_same_load) {
  auto model_data_holder = LoadValidDuplicateIndividualModelData();
  gert::Om2ModelExecutor executor;
  auto load_arg = MakeOm2LoadArg();
  ASSERT_EQ(setenv("OM2_EXPECT_CONST1_MODE", "NON_NULL", 1), 0);
  ASSERT_EQ(setenv("OM2_EXPECT_CONST1_FIRST_BYTE", "102", 1), 0);
  ASSERT_EQ(setenv("OM2_EXPECT_CONST2_MODE", "NON_NULL", 1), 0);
  ASSERT_EQ(setenv("OM2_EXPECT_CONST2_FIRST_BYTE", "102", 1), 0);
  ASSERT_EQ(setenv("OM2_EXPECT_CONST1_CONST2_PTR_EQUAL", "1", 1), 0);
  EXPECT_EQ(executor.Load(model_data_holder.model_data, load_arg, 1U), SUCCESS);
}

TEST_F(Om2ModelExecutorUt, run_before_load_failed) {
  gert::Om2ModelExecutor executor;
  std::vector<gert::Tensor> input_tensors;
  std::vector<gert::Tensor> output_tensors;
  std::vector<gert::Tensor *> inputs;
  std::vector<gert::Tensor *> outputs;
  ConstructIoTensors(input_tensors, output_tensors, inputs, outputs);
  EXPECT_NE(executor.Run(inputs, outputs), SUCCESS);
}

TEST_F(Om2ModelExecutorUt, run_ok_after_load) {
  auto model_data_holder = LoadValidModelData();
  gert::Om2ModelExecutor executor;
  auto load_arg = MakeOm2LoadArg();
  ASSERT_EQ(executor.Load(model_data_holder.model_data, load_arg, 1U), SUCCESS);

  std::vector<gert::Tensor> input_tensors;
  std::vector<gert::Tensor> output_tensors;
  std::vector<gert::Tensor *> inputs;
  std::vector<gert::Tensor *> outputs;
  ConstructIoTensors(input_tensors, output_tensors, inputs, outputs);
  EXPECT_EQ(executor.Run(inputs, outputs), SUCCESS);
}

TEST_F(Om2ModelExecutorUt, run_async_before_load_failed) {
  gert::Om2ModelExecutor executor;
  std::vector<gert::Tensor> input_tensors;
  std::vector<gert::Tensor> output_tensors;
  std::vector<gert::Tensor *> inputs;
  std::vector<gert::Tensor *> outputs;
  ConstructIoTensors(input_tensors, output_tensors, inputs, outputs);
  EXPECT_NE(executor.RunAsync(nullptr, inputs, outputs), SUCCESS);
}

TEST_F(Om2ModelExecutorUt, run_async_ok_after_load) {
  auto model_data_holder = LoadValidModelData();
  gert::Om2ModelExecutor executor;
  auto load_arg = MakeOm2LoadArg();
  ASSERT_EQ(executor.Load(model_data_holder.model_data, load_arg, 1U), SUCCESS);

  std::vector<gert::Tensor> input_tensors;
  std::vector<gert::Tensor> output_tensors;
  std::vector<gert::Tensor *> inputs;
  std::vector<gert::Tensor *> outputs;
  ConstructIoTensors(input_tensors, output_tensors, inputs, outputs);
  EXPECT_EQ(executor.RunAsync(nullptr, inputs, outputs), SUCCESS);
}

TEST_F(Om2ModelExecutorUt, get_model_desc_info_ok) {
  auto model_data_holder = LoadValidModelData();
  gert::Om2ModelExecutor executor;
  auto load_arg = MakeOm2LoadArg();
  ASSERT_EQ(executor.Load(model_data_holder.model_data, load_arg, 1U), SUCCESS);

  const std::vector<ge::Om2TensorDesc> *input_desc = nullptr;
  const std::vector<ge::Om2TensorDesc> *output_desc = nullptr;
  EXPECT_EQ(executor.GetModelDescInfo(input_desc, output_desc, false), SUCCESS);
  ASSERT_NE(input_desc, nullptr);
  ASSERT_NE(output_desc, nullptr);
  ASSERT_EQ(input_desc->size(), 2U);
  ASSERT_EQ(output_desc->size(), 1U);
  EXPECT_EQ((*input_desc)[0].GetName(), "data1");
  EXPECT_EQ((*input_desc)[1].GetName(), "data2");
  EXPECT_EQ((*output_desc)[0].GetName(), "output_0_reshape1_0");

  const std::vector<ge::Om2TensorDesc> *input_desc_v2 = nullptr;
  const std::vector<ge::Om2TensorDesc> *output_desc_v2 = nullptr;
  EXPECT_EQ(executor.GetModelDescInfo(input_desc_v2, output_desc_v2, true), SUCCESS);
  ASSERT_NE(input_desc_v2, nullptr);
  ASSERT_NE(output_desc_v2, nullptr);
  EXPECT_EQ(input_desc_v2->size(), input_desc->size());
  EXPECT_EQ(output_desc_v2->size(), output_desc->size());
}

TEST_F(Om2ModelExecutorUt, get_model_attrs_ok) {
  auto model_data_holder = LoadValidModelData();
  gert::Om2ModelExecutor executor;
  auto load_arg = MakeOm2LoadArg();
  ASSERT_EQ(executor.Load(model_data_holder.model_data, load_arg, 1U), SUCCESS);

  std::vector<std::string> dynamic_output_shape;
  EXPECT_EQ(executor.GetModelAttrs(dynamic_output_shape), SUCCESS);
  EXPECT_TRUE(dynamic_output_shape.empty());
}

TEST_F(Om2ModelExecutorUt, get_dynamic_batch_info_ok) {
  auto model_data_holder = LoadValidModelData();
  gert::Om2ModelExecutor executor;
  auto load_arg = MakeOm2LoadArg();
  ASSERT_EQ(executor.Load(model_data_holder.model_data, load_arg, 1U), SUCCESS);

  std::vector<std::vector<int64_t>> dynamic_batch_info;
  int32_t dynamic_type = -1;
  EXPECT_EQ(executor.GetDynamicBatchInfo(dynamic_batch_info, dynamic_type), SUCCESS);
  EXPECT_TRUE(dynamic_batch_info.empty());
  EXPECT_EQ(dynamic_type, 0);
}

TEST_F(Om2ModelExecutorUt, get_user_designate_shape_order_ok) {
  auto model_data_holder = LoadValidModelData();
  gert::Om2ModelExecutor executor;
  auto load_arg = MakeOm2LoadArg();
  ASSERT_EQ(executor.Load(model_data_holder.model_data, load_arg, 1U), SUCCESS);

  std::vector<std::string> user_designate_shape_order;
  EXPECT_EQ(executor.GetUserDesignateShapeOrder(user_designate_shape_order), SUCCESS);
  EXPECT_TRUE(user_designate_shape_order.empty());
}

TEST_F(Om2ModelExecutorUt, IsOm2Model_Ok_FromDataMultiSceneTest) {
  // invalid model data
  bool is_support = false;
  EXPECT_EQ(gert::IsOm2Model(nullptr, 10, is_support), ACL_ERROR_GE_PARAM_INVALID);

  // data size is too small
  const uint8_t data[] = {0x50, 0x4B};
  is_support = false;
  EXPECT_EQ(gert::IsOm2Model(data, 2, is_support), ACL_ERROR_GE_EXEC_MODEL_DATA_SIZE_INVALID);

  // valid magic
  const uint8_t valid_data[] = {0x50, 0x4B, 0x03, 0x04};
  is_support = false;
  EXPECT_EQ(gert::IsOm2Model(valid_data, 4, is_support), SUCCESS);
  EXPECT_TRUE(is_support);

  // invalid magic
  constexpr uint8_t invalid_data[] = {0x00, 0x00, 0x00, 0x00};
  is_support = false;
  EXPECT_EQ(gert::IsOm2Model(invalid_data, 4, is_support), SUCCESS);
  EXPECT_FALSE(is_support);
}

TEST_F(Om2ModelExecutorUt, IsOm2Model_Ok_FromFileMultiScene) {
  // file is not exist
  bool is_support = false;
  EXPECT_EQ(gert::IsOm2Model("/non/existent/file.om2", is_support), ACL_ERROR_GE_EXEC_MODEL_PATH_INVALID);

  // file size too small
  const std::string test_file = PathUtils::Join({test_work_dir_, "small_file.om2"});
  WriteBinaryFile(test_file, {0x50, 0x4B});
  is_support = false;
  EXPECT_EQ(gert::IsOm2Model(test_file.c_str(), is_support), ACL_ERROR_GE_EXEC_MODEL_DATA_SIZE_INVALID);

  // valid_magic
  PrepareOm2File();
  is_support = false;
  EXPECT_EQ(gert::IsOm2Model(om2_file_path_.c_str(), is_support), SUCCESS);
  EXPECT_TRUE(is_support);

  // invalid_magic
  const std::string test_file_invalid = PathUtils::Join({test_work_dir_, "invalid_magic.om2"});
  WriteBinaryFile(test_file, {0x00, 0x00, 0x00, 0x00});

  is_support = false;
  EXPECT_EQ(gert::IsOm2Model(test_file.c_str(), is_support), SUCCESS);
  EXPECT_FALSE(is_support);
}

TEST_F(Om2ModelExecutorUt, get_mem_and_weight_size_from_file_ok) {
  PrepareOm2File();
  size_t work_size = 0U;
  size_t weight_size = 0U;
  EXPECT_EQ(gert::GetOm2MemAndWeightSize(om2_file_path_, work_size, weight_size), SUCCESS);
  EXPECT_EQ(work_size, 2048U);
  EXPECT_EQ(weight_size, 16U);
}

TEST_F(Om2ModelExecutorUt, get_mem_and_weight_size_external_only_with_zero_internal_weight_size_ok) {
  const std::string om2_file_path =
      PathUtils::Join({test_work_dir_, "external_only_with_zero_internal_weight_size.om2"});
  ZipArchiveWriter zip_writer(om2_file_path);
  ASSERT_TRUE(zip_writer.IsMemFileOpened());
  const auto manifest = MakeManifestJson();
  const auto constants_config = MakeIndividualConstantsConfigJsonWithZeroInternalWeightSize();
  const auto model_meta = MakeModelMetaJson();
  ASSERT_TRUE(zip_writer.WriteBytes("manifest.json", manifest.data(), manifest.size(), false));
  ASSERT_TRUE(zip_writer.WriteBytes("data/constants/model_0_constants_config.json", constants_config.data(),
                                    constants_config.size(), false));
  ASSERT_TRUE(zip_writer.WriteBytes("data/model_0/model_meta.json", model_meta.data(), model_meta.size(), false));
  ASSERT_TRUE(zip_writer.SaveModelDataToFile());

  size_t work_size = 0U;
  size_t weight_size = 1024U;
  EXPECT_EQ(gert::GetOm2MemAndWeightSize(om2_file_path, work_size, weight_size), SUCCESS);
  EXPECT_EQ(work_size, 2048U);
  EXPECT_EQ(weight_size, 0U);
}

TEST_F(Om2ModelExecutorUt, get_mem_and_weight_size_from_mem_ok) {
  auto model_data_holder = LoadValidModelData();
  size_t work_size = 0U;
  size_t weight_size = 0U;
  EXPECT_EQ(gert::GetOm2MemAndWeightSize(model_data_holder.model_data.model_data,
                                         model_data_holder.model_data.model_len, work_size, weight_size),
            SUCCESS);
  EXPECT_EQ(work_size, 2048U);
  EXPECT_EQ(weight_size, 16U);
}

// 辅助函数：生成带属性的op_attr.json
static std::string MakeOpAttrJson() {
  return R"({"test_op":{"_datadump_original_op_names":{"type":"LIST_STRING","value":["original_op1","original_op2"]}}})";
}

// 辅助函数：生成空op_attr.json
static std::string MakeEmptyOpAttrJson() {
  return "{}";
}

// 辅助函数：生成无效的JSON
static std::string MakeInvalidOpAttrJson() {
  return "invalid json content";
}

// 辅助函数：生成多个算子属性的op_attr.json
static std::string MakeMultipleOpAttrJson() {
  return R"({"op1":{"_datadump_original_op_names":{"type":"LIST_STRING","value":["orig1","orig2"]},"_another_attr":{"type":"STRING","value":"test_value"}},"op2":{"_datadump_original_op_names":{"type":"LIST_STRING","value":["orig3"]}}})";
}

TEST_F(Om2ModelExecutorUt, GetOpAttr_ValidOpAttrJson_ReturnsParsedMap) {
  // 创建包含op_attr.json的OM2文件
  const std::string om2_with_attr = PathUtils::Join({test_work_dir_, "om2_with_op_attr.om2"});
  const std::string runtime_dir = PathUtils::Join({test_work_dir_, "fake_runtime_attr"});
  const std::string build_dir = PathUtils::Join({runtime_dir, "build"});
  const std::string so_path = PathUtils::Join({runtime_dir, "libg1_om2.so"});

  (void)PathUtils::RemoveDirectories(runtime_dir);
  ASSERT_EQ(CreateDir(runtime_dir), 0);
  WriteTextFile(PathUtils::Join({runtime_dir, "g1_interface.h"}), MakeInterfaceHeader());
  WriteTextFile(PathUtils::Join({runtime_dir, "g1_resources.cpp"}), MakeEmptyCpp("g1_interface.h"));
  WriteTextFile(PathUtils::Join({runtime_dir, "g1_kernel_reg.cpp"}), MakeEmptyCpp("g1_interface.h"));
  WriteTextFile(PathUtils::Join({runtime_dir, "g1_args_manager.cpp"}), MakeEmptyCpp("g1_interface.h"));
  WriteTextFile(PathUtils::Join({runtime_dir, "g1_load_and_run.cpp"}), MakeLoadAndRunCpp());
  WriteTextFile(PathUtils::Join({runtime_dir, "CMakeLists.txt"}), MakeCMakeLists());

  const std::string cmake_config_cmd = "cmake -S " + runtime_dir + " -B " + build_dir;
  const std::string cmake_build_cmd = "cmake --build " + build_dir + " -j1";
  RunCommandOrAssert(cmake_config_cmd);
  RunCommandOrAssert(cmake_build_cmd);
  ASSERT_EQ(mmAccess2(so_path.c_str(), M_F_OK), EOK);

  ZipArchiveWriter zip_writer(om2_with_attr);
  ASSERT_TRUE(zip_writer.IsMemFileOpened());
  const auto manifest = MakeManifestJson();
  const auto model_meta = MakeModelMetaJson();
  const auto op_attr = MakeOpAttrJson();
  ASSERT_TRUE(zip_writer.WriteBytes("manifest.json", manifest.data(), manifest.size(), false));
  ASSERT_TRUE(zip_writer.WriteBytes("data/model_0/model_meta.json", model_meta.data(), model_meta.size(), false));
  ASSERT_TRUE(zip_writer.WriteBytes("data/model_0/op_attr.json", op_attr.data(), op_attr.size(), false));
  ASSERT_TRUE(zip_writer.WriteFile("data/model_0/runtime/CMakeLists.txt",
                                   PathUtils::Join({runtime_dir, "CMakeLists.txt"}), false));
  ASSERT_TRUE(zip_writer.WriteFile("data/model_0/runtime/g1_interface.h",
                                   PathUtils::Join({runtime_dir, "g1_interface.h"}), false));
  ASSERT_TRUE(zip_writer.WriteFile("data/model_0/runtime/g1_resources.cpp",
                                   PathUtils::Join({runtime_dir, "g1_resources.cpp"}), false));
  ASSERT_TRUE(zip_writer.WriteFile("data/model_0/runtime/g1_kernel_reg.cpp",
                                   PathUtils::Join({runtime_dir, "g1_kernel_reg.cpp"}), false));
  ASSERT_TRUE(zip_writer.WriteFile("data/model_0/runtime/g1_args_manager.cpp",
                                   PathUtils::Join({runtime_dir, "g1_args_manager.cpp"}), false));
  ASSERT_TRUE(zip_writer.WriteFile("data/model_0/runtime/g1_load_and_run.cpp",
                                   PathUtils::Join({runtime_dir, "g1_load_and_run.cpp"}), false));
  ASSERT_TRUE(zip_writer.WriteFile("data/model_0/runtime/libg1_om2.so", so_path, false));
  ASSERT_TRUE(zip_writer.SaveModelDataToFile());

  uint32_t model_buf_size = 0U;
  auto model_buf = GetBinDataFromFile(om2_with_attr, model_buf_size);
  ASSERT_NE(model_buf, nullptr);
  ASSERT_GT(model_buf_size, 0U);

  ModelDataHolder holder;
  holder.model_data.model_data = model_buf.get();
  holder.model_data.model_len = model_buf_size;
  holder.model_data.om_path = om2_with_attr;
  holder.buffer = std::move(model_buf);

  auto load_arg = MakeOm2LoadArg();
  ge::Status status;
  auto executor = gert::LoadOm2ExecutorFromData(holder.model_data, load_arg, status);
  ASSERT_EQ(status, SUCCESS);

  std::map<std::string, std::map<std::string, std::string>> op_attr_map;
  status = executor->GetOpAttr(op_attr_map);
  EXPECT_EQ(status, SUCCESS);

  // 验证map内容
  EXPECT_FALSE(op_attr_map.empty());
  EXPECT_TRUE(op_attr_map.find("test_op") != op_attr_map.end());
  EXPECT_TRUE(op_attr_map["test_op"].find("_datadump_original_op_names") != op_attr_map["test_op"].end());

  const std::string &value_str = op_attr_map["test_op"]["_datadump_original_op_names"];
  // value应该是[N]value格式，与OM1 DavinciModel::GetNodeAttr一致
  EXPECT_EQ(value_str, "[12]original_op1[12]original_op2");
}

TEST_F(Om2ModelExecutorUt, GetOpAttr_EmptyOpAttrJson_ReturnsEmptyMap) {
  // 创建包含空op_attr.json的OM2文件
  const std::string om2_empty_attr = PathUtils::Join({test_work_dir_, "om2_empty_op_attr.om2"});
  const std::string runtime_dir = PathUtils::Join({test_work_dir_, "fake_runtime_empty_attr"});
  const std::string build_dir = PathUtils::Join({runtime_dir, "build"});
  const std::string so_path = PathUtils::Join({runtime_dir, "libg1_om2.so"});

  (void)PathUtils::RemoveDirectories(runtime_dir);
  ASSERT_EQ(CreateDir(runtime_dir), 0);
  WriteTextFile(PathUtils::Join({runtime_dir, "g1_interface.h"}), MakeInterfaceHeader());
  WriteTextFile(PathUtils::Join({runtime_dir, "g1_resources.cpp"}), MakeEmptyCpp("g1_interface.h"));
  WriteTextFile(PathUtils::Join({runtime_dir, "g1_kernel_reg.cpp"}), MakeEmptyCpp("g1_interface.h"));
  WriteTextFile(PathUtils::Join({runtime_dir, "g1_args_manager.cpp"}), MakeEmptyCpp("g1_interface.h"));
  WriteTextFile(PathUtils::Join({runtime_dir, "g1_load_and_run.cpp"}), MakeLoadAndRunCpp());
  WriteTextFile(PathUtils::Join({runtime_dir, "CMakeLists.txt"}), MakeCMakeLists());

  const std::string cmake_config_cmd = "cmake -S " + runtime_dir + " -B " + build_dir;
  const std::string cmake_build_cmd = "cmake --build " + build_dir + " -j1";
  RunCommandOrAssert(cmake_config_cmd);
  RunCommandOrAssert(cmake_build_cmd);
  ASSERT_EQ(mmAccess2(so_path.c_str(), M_F_OK), EOK);

  ZipArchiveWriter zip_writer(om2_empty_attr);
  ASSERT_TRUE(zip_writer.IsMemFileOpened());
  const auto manifest = MakeManifestJson();
  const auto model_meta = MakeModelMetaJson();
  const auto op_attr = MakeEmptyOpAttrJson();
  ASSERT_TRUE(zip_writer.WriteBytes("manifest.json", manifest.data(), manifest.size(), false));
  ASSERT_TRUE(zip_writer.WriteBytes("data/model_0/model_meta.json", model_meta.data(), model_meta.size(), false));
  ASSERT_TRUE(zip_writer.WriteBytes("data/model_0/op_attr.json", op_attr.data(), op_attr.size(), false));
  ASSERT_TRUE(zip_writer.WriteFile("data/model_0/runtime/CMakeLists.txt",
                                   PathUtils::Join({runtime_dir, "CMakeLists.txt"}), false));
  ASSERT_TRUE(zip_writer.WriteFile("data/model_0/runtime/g1_interface.h",
                                   PathUtils::Join({runtime_dir, "g1_interface.h"}), false));
  ASSERT_TRUE(zip_writer.WriteFile("data/model_0/runtime/g1_resources.cpp",
                                   PathUtils::Join({runtime_dir, "g1_resources.cpp"}), false));
  ASSERT_TRUE(zip_writer.WriteFile("data/model_0/runtime/g1_kernel_reg.cpp",
                                   PathUtils::Join({runtime_dir, "g1_kernel_reg.cpp"}), false));
  ASSERT_TRUE(zip_writer.WriteFile("data/model_0/runtime/g1_args_manager.cpp",
                                   PathUtils::Join({runtime_dir, "g1_args_manager.cpp"}), false));
  ASSERT_TRUE(zip_writer.WriteFile("data/model_0/runtime/g1_load_and_run.cpp",
                                   PathUtils::Join({runtime_dir, "g1_load_and_run.cpp"}), false));
  ASSERT_TRUE(zip_writer.WriteFile("data/model_0/runtime/libg1_om2.so", so_path, false));
  ASSERT_TRUE(zip_writer.SaveModelDataToFile());

  // 加载模型并验证GetOpAttr返回空map
  uint32_t model_buf_size = 0U;
  auto model_buf = GetBinDataFromFile(om2_empty_attr, model_buf_size);
  ASSERT_NE(model_buf, nullptr);
  ASSERT_GT(model_buf_size, 0U);

  ModelDataHolder holder;
  holder.model_data.model_data = model_buf.get();
  holder.model_data.model_len = model_buf_size;
  holder.model_data.om_path = om2_empty_attr;
  holder.buffer = std::move(model_buf);

  auto load_arg = MakeOm2LoadArg();
  ge::Status status;
  auto executor = gert::LoadOm2ExecutorFromData(holder.model_data, load_arg, status);
  ASSERT_EQ(status, SUCCESS);

  std::map<std::string, std::map<std::string, std::string>> op_attr_map;
  status = executor->GetOpAttr(op_attr_map);
  EXPECT_EQ(status, SUCCESS);
  EXPECT_TRUE(op_attr_map.empty());
}

TEST_F(Om2ModelExecutorUt, GetOpAttr_MissingOpAttrJson_ReturnsEmptyMap) {
  // 使用现有的om2_file_path_（不包含op_attr.json）
  PrepareOm2File();

  auto load_arg = MakeOm2LoadArg();
  ge::Status status;
  auto model_data_holder = LoadValidModelData();
  auto handle = gert::LoadOm2ExecutorFromData(model_data_holder.model_data, load_arg, status);
  ASSERT_EQ(status, SUCCESS);

  std::map<std::string, std::map<std::string, std::string>> op_attr_map;
  status = handle->GetOpAttr(op_attr_map);
  // 无op_attr.json时不报错，返回空map（fallback机制）
  EXPECT_EQ(status, SUCCESS);
  EXPECT_TRUE(op_attr_map.empty());
}

TEST_F(Om2ModelExecutorUt, GetOpAttr_InvalidOpAttrJson_ReturnsEmptyMap) {
  // 创建包含无效JSON的OM2文件
  const std::string om2_invalid_attr = PathUtils::Join({test_work_dir_, "om2_invalid_op_attr.om2"});
  const std::string runtime_dir = PathUtils::Join({test_work_dir_, "fake_runtime_invalid_attr"});
  const std::string build_dir = PathUtils::Join({runtime_dir, "build"});
  const std::string so_path = PathUtils::Join({runtime_dir, "libg1_om2.so"});

  (void)PathUtils::RemoveDirectories(runtime_dir);
  ASSERT_EQ(CreateDir(runtime_dir), 0);
  WriteTextFile(PathUtils::Join({runtime_dir, "g1_interface.h"}), MakeInterfaceHeader());
  WriteTextFile(PathUtils::Join({runtime_dir, "g1_resources.cpp"}), MakeEmptyCpp("g1_interface.h"));
  WriteTextFile(PathUtils::Join({runtime_dir, "g1_kernel_reg.cpp"}), MakeEmptyCpp("g1_interface.h"));
  WriteTextFile(PathUtils::Join({runtime_dir, "g1_args_manager.cpp"}), MakeEmptyCpp("g1_interface.h"));
  WriteTextFile(PathUtils::Join({runtime_dir, "g1_load_and_run.cpp"}), MakeLoadAndRunCpp());
  WriteTextFile(PathUtils::Join({runtime_dir, "CMakeLists.txt"}), MakeCMakeLists());

  const std::string cmake_config_cmd = "cmake -S " + runtime_dir + " -B " + build_dir;
  const std::string cmake_build_cmd = "cmake --build " + build_dir + " -j1";
  RunCommandOrAssert(cmake_config_cmd);
  RunCommandOrAssert(cmake_build_cmd);
  ASSERT_EQ(mmAccess2(so_path.c_str(), M_F_OK), EOK);

  ZipArchiveWriter zip_writer(om2_invalid_attr);
  ASSERT_TRUE(zip_writer.IsMemFileOpened());
  const auto manifest = MakeManifestJson();
  const auto model_meta = MakeModelMetaJson();
  const auto op_attr = MakeInvalidOpAttrJson();
  ASSERT_TRUE(zip_writer.WriteBytes("manifest.json", manifest.data(), manifest.size(), false));
  ASSERT_TRUE(zip_writer.WriteBytes("data/model_0/model_meta.json", model_meta.data(), model_meta.size(), false));
  ASSERT_TRUE(zip_writer.WriteBytes("data/model_0/op_attr.json", op_attr.data(), op_attr.size(), false));
  ASSERT_TRUE(zip_writer.WriteFile("data/model_0/runtime/CMakeLists.txt",
                                   PathUtils::Join({runtime_dir, "CMakeLists.txt"}), false));
  ASSERT_TRUE(zip_writer.WriteFile("data/model_0/runtime/g1_interface.h",
                                   PathUtils::Join({runtime_dir, "g1_interface.h"}), false));
  ASSERT_TRUE(zip_writer.WriteFile("data/model_0/runtime/g1_resources.cpp",
                                   PathUtils::Join({runtime_dir, "g1_resources.cpp"}), false));
  ASSERT_TRUE(zip_writer.WriteFile("data/model_0/runtime/g1_kernel_reg.cpp",
                                   PathUtils::Join({runtime_dir, "g1_kernel_reg.cpp"}), false));
  ASSERT_TRUE(zip_writer.WriteFile("data/model_0/runtime/g1_args_manager.cpp",
                                   PathUtils::Join({runtime_dir, "g1_args_manager.cpp"}), false));
  ASSERT_TRUE(zip_writer.WriteFile("data/model_0/runtime/g1_load_and_run.cpp",
                                   PathUtils::Join({runtime_dir, "g1_load_and_run.cpp"}), false));
  ASSERT_TRUE(zip_writer.WriteFile("data/model_0/runtime/libg1_om2.so", so_path, false));
  ASSERT_TRUE(zip_writer.SaveModelDataToFile());

  // 加载模型，无效JSON时fallback到空map
  uint32_t model_buf_size = 0U;
  auto model_buf = GetBinDataFromFile(om2_invalid_attr, model_buf_size);
  ASSERT_NE(model_buf, nullptr);
  ASSERT_GT(model_buf_size, 0U);

  ModelDataHolder holder;
  holder.model_data.model_data = model_buf.get();
  holder.model_data.model_len = model_buf_size;
  holder.model_data.om_path = om2_invalid_attr;
  holder.buffer = std::move(model_buf);

  auto load_arg = MakeOm2LoadArg();
  ge::Status status;
  auto handle = gert::LoadOm2ExecutorFromData(holder.model_data, load_arg, status);
  ASSERT_EQ(status, SUCCESS);

  std::map<std::string, std::map<std::string, std::string>> op_attr_map;
  status = handle->GetOpAttr(op_attr_map);
  EXPECT_EQ(status, SUCCESS);
  EXPECT_TRUE(op_attr_map.empty());
}

TEST_F(Om2ModelExecutorUt, ParseOpAttrJsonToMapInternal_MultipleAttrs_ParsesAllAttrs) {
  // 创建包含多个算子多个属性的OM2文件
  const std::string om2_multi_attr = PathUtils::Join({test_work_dir_, "om2_multi_op_attr.om2"});
  const std::string runtime_dir = PathUtils::Join({test_work_dir_, "fake_runtime_multi_attr"});
  const std::string build_dir = PathUtils::Join({runtime_dir, "build"});
  const std::string so_path = PathUtils::Join({runtime_dir, "libg1_om2.so"});

  (void)PathUtils::RemoveDirectories(runtime_dir);
  ASSERT_EQ(CreateDir(runtime_dir), 0);
  WriteTextFile(PathUtils::Join({runtime_dir, "g1_interface.h"}), MakeInterfaceHeader());
  WriteTextFile(PathUtils::Join({runtime_dir, "g1_resources.cpp"}), MakeEmptyCpp("g1_interface.h"));
  WriteTextFile(PathUtils::Join({runtime_dir, "g1_kernel_reg.cpp"}), MakeEmptyCpp("g1_interface.h"));
  WriteTextFile(PathUtils::Join({runtime_dir, "g1_args_manager.cpp"}), MakeEmptyCpp("g1_interface.h"));
  WriteTextFile(PathUtils::Join({runtime_dir, "g1_load_and_run.cpp"}), MakeLoadAndRunCpp());
  WriteTextFile(PathUtils::Join({runtime_dir, "CMakeLists.txt"}), MakeCMakeLists());

  const std::string cmake_config_cmd = "cmake -S " + runtime_dir + " -B " + build_dir;
  const std::string cmake_build_cmd = "cmake --build " + build_dir + " -j1";
  RunCommandOrAssert(cmake_config_cmd);
  RunCommandOrAssert(cmake_build_cmd);
  ASSERT_EQ(mmAccess2(so_path.c_str(), M_F_OK), EOK);

  ZipArchiveWriter zip_writer(om2_multi_attr);
  ASSERT_TRUE(zip_writer.IsMemFileOpened());
  const auto manifest = MakeManifestJson();
  const auto model_meta = MakeModelMetaJson();
  const auto op_attr = MakeMultipleOpAttrJson();
  ASSERT_TRUE(zip_writer.WriteBytes("manifest.json", manifest.data(), manifest.size(), false));
  ASSERT_TRUE(zip_writer.WriteBytes("data/model_0/model_meta.json", model_meta.data(), model_meta.size(), false));
  ASSERT_TRUE(zip_writer.WriteBytes("data/model_0/op_attr.json", op_attr.data(), op_attr.size(), false));
  ASSERT_TRUE(zip_writer.WriteFile("data/model_0/runtime/CMakeLists.txt",
                                   PathUtils::Join({runtime_dir, "CMakeLists.txt"}), false));
  ASSERT_TRUE(zip_writer.WriteFile("data/model_0/runtime/g1_interface.h",
                                   PathUtils::Join({runtime_dir, "g1_interface.h"}), false));
  ASSERT_TRUE(zip_writer.WriteFile("data/model_0/runtime/g1_resources.cpp",
                                   PathUtils::Join({runtime_dir, "g1_resources.cpp"}), false));
  ASSERT_TRUE(zip_writer.WriteFile("data/model_0/runtime/g1_kernel_reg.cpp",
                                   PathUtils::Join({runtime_dir, "g1_kernel_reg.cpp"}), false));
  ASSERT_TRUE(zip_writer.WriteFile("data/model_0/runtime/g1_args_manager.cpp",
                                   PathUtils::Join({runtime_dir, "g1_args_manager.cpp"}), false));
  ASSERT_TRUE(zip_writer.WriteFile("data/model_0/runtime/g1_load_and_run.cpp",
                                   PathUtils::Join({runtime_dir, "g1_load_and_run.cpp"}), false));
  ASSERT_TRUE(zip_writer.WriteFile("data/model_0/runtime/libg1_om2.so", so_path, false));
  ASSERT_TRUE(zip_writer.SaveModelDataToFile());

  // 加载模型并验证所有属性都被解析
  uint32_t model_buf_size = 0U;
  auto model_buf = GetBinDataFromFile(om2_multi_attr, model_buf_size);
  ASSERT_NE(model_buf, nullptr);
  ASSERT_GT(model_buf_size, 0U);

  ModelDataHolder holder;
  holder.model_data.model_data = model_buf.get();
  holder.model_data.model_len = model_buf_size;
  holder.model_data.om_path = om2_multi_attr;
  holder.buffer = std::move(model_buf);

  auto load_arg = MakeOm2LoadArg();
  ge::Status status;
  auto handle = gert::LoadOm2ExecutorFromData(holder.model_data, load_arg, status);
  ASSERT_EQ(status, SUCCESS);

  std::map<std::string, std::map<std::string, std::string>> op_attr_map;
  status = handle->GetOpAttr(op_attr_map);
  EXPECT_EQ(status, SUCCESS);

  // 验证包含2个算子
  EXPECT_EQ(op_attr_map.size(), 2U);
  EXPECT_TRUE(op_attr_map.find("op1") != op_attr_map.end());
  EXPECT_TRUE(op_attr_map.find("op2") != op_attr_map.end());

  // op1有2个属性
  EXPECT_EQ(op_attr_map["op1"].size(), 2U);
  EXPECT_TRUE(op_attr_map["op1"].find("_datadump_original_op_names") != op_attr_map["op1"].end());
  EXPECT_EQ(op_attr_map["op1"]["_datadump_original_op_names"], "[5]orig1[5]orig2");

  // op2有1个属性
  EXPECT_EQ(op_attr_map["op2"].size(), 1U);
  EXPECT_EQ(op_attr_map["op2"]["_datadump_original_op_names"], "[5]orig3");
}

TEST_F(Om2ModelExecutorUt, GetOpAttr_BeforeLoad_ReturnsError) {
  // 未加载模型前调用GetOpAttr应该返回错误
  auto executor = std::make_unique<gert::Om2ModelExecutor>();

  std::map<std::string, std::map<std::string, std::string>> op_attr_map;
  ge::Status status = executor->GetOpAttr(op_attr_map);
  // 未初始化，应该返回错误
  EXPECT_NE(status, SUCCESS);
  EXPECT_TRUE(op_attr_map.empty());
}

// ========== SetDynamicSize 和 GetCurrentShape 测试用例 ==========

TEST_F(Om2ModelExecutorUt, SetDynamicSize_BeforeLoad_ReturnsError) {
  auto executor = std::make_unique<gert::Om2ModelExecutor>();
  std::vector<uint64_t> batch_num = {1, 2, 4};
  int32_t dynamic_type = 1;  // DYNAMIC_BATCH

  ge::Status status = executor->SetDynamicSize(batch_num, dynamic_type);
  EXPECT_NE(status, SUCCESS);
}

TEST_F(Om2ModelExecutorUt, SetDynamicSize_TypeMismatch_ReturnsError) {
  auto model_data_holder = LoadValidModelData();
  gert::Om2ModelExecutor executor;
  auto load_arg = MakeOm2LoadArg();
  ASSERT_EQ(executor.Load(model_data_holder.model_data, load_arg, 1U), SUCCESS);

  // 静态模型 dynamic_type=0，尝试设置 DYNAMIC_BATCH=1 应该失败
  std::vector<uint64_t> batch_num = {1};
  int32_t dynamic_type = 1;  // DYNAMIC_BATCH

  ge::Status status = executor.SetDynamicSize(batch_num, dynamic_type);
  EXPECT_NE(status, SUCCESS);
}

TEST_F(Om2ModelExecutorUt, SetDynamicSize_InvalidGear_ReturnsError) {
  // 创建包含动态 batch 信息的模型
  const std::string so_path = PathUtils::Join({test_work_dir_, "fake_runtime/libg1_om2.so"});
  const std::string om2_dynamic_path = PathUtils::Join({test_work_dir_, "dynamic_batch.om2"});

  ZipArchiveWriter zip_writer(om2_dynamic_path);
  ASSERT_TRUE(zip_writer.IsMemFileOpened());
  const auto manifest = MakeManifestJson();
  const auto model_meta = MakeModelMetaJsonWithDynamicBatch();
  ASSERT_TRUE(zip_writer.WriteBytes("manifest.json", manifest.data(), manifest.size(), false));
  ASSERT_TRUE(zip_writer.WriteBytes("data/model_0/model_meta.json", model_meta.data(), model_meta.size(), false));
  ASSERT_TRUE(zip_writer.WriteFile("data/model_0/runtime/libg1_om2.so", so_path, false));
  ASSERT_TRUE(zip_writer.SaveModelDataToFile());

  // 加载模型
  uint32_t model_buf_size = 0U;
  auto model_buf = GetBinDataFromFile(om2_dynamic_path, model_buf_size);
  ASSERT_NE(model_buf, nullptr);
  ASSERT_GT(model_buf_size, 0U);

  ModelData model_data{};
  model_data.model_data = model_buf.get();
  model_data.model_len = model_buf_size;

  gert::Om2ModelExecutor executor;
  auto load_arg = MakeOm2LoadArg();
  ASSERT_EQ(executor.Load(model_data, load_arg, 1U), SUCCESS);

  // 尝试设置不存在的档位（模型支持 1,2,4,8）
  std::vector<uint64_t> invalid_batch = {3};
  int32_t dynamic_type = 1;  // DYNAMIC_BATCH

  ge::Status status = executor.SetDynamicSize(invalid_batch, dynamic_type);
  EXPECT_NE(status, SUCCESS);
}

TEST_F(Om2ModelExecutorUt, SetDynamicSize_ValidGear_Success) {
  // 创建包含动态 batch 信息的模型
  const std::string so_path = PathUtils::Join({test_work_dir_, "fake_runtime/libg1_om2.so"});
  const std::string om2_dynamic_path = PathUtils::Join({test_work_dir_, "dynamic_batch_valid.om2"});

  ZipArchiveWriter zip_writer(om2_dynamic_path);
  ASSERT_TRUE(zip_writer.IsMemFileOpened());
  const auto manifest = MakeManifestJson();
  const auto model_meta = MakeModelMetaJsonWithDynamicBatch();
  ASSERT_TRUE(zip_writer.WriteBytes("manifest.json", manifest.data(), manifest.size(), false));
  ASSERT_TRUE(zip_writer.WriteBytes("data/model_0/model_meta.json", model_meta.data(), model_meta.size(), false));
  ASSERT_TRUE(zip_writer.WriteFile("data/model_0/runtime/libg1_om2.so", so_path, false));
  ASSERT_TRUE(zip_writer.SaveModelDataToFile());

  uint32_t model_buf_size = 0U;
  auto model_buf = GetBinDataFromFile(om2_dynamic_path, model_buf_size);
  ASSERT_NE(model_buf, nullptr);
  ASSERT_GT(model_buf_size, 0U);

  ModelData model_data{};
  model_data.model_data = model_buf.get();
  model_data.model_len = model_buf_size;

  gert::Om2ModelExecutor executor;
  auto load_arg = MakeOm2LoadArg();
  ASSERT_EQ(executor.Load(model_data, load_arg, 1U), SUCCESS);

  // 设置有效档位 batch=4
  std::vector<uint64_t> batch_num = {4};
  int32_t dynamic_type = 1;  // DYNAMIC_BATCH

  ge::Status status = executor.SetDynamicSize(batch_num, dynamic_type);
  EXPECT_EQ(status, SUCCESS);

  // 验证 GetCurrentShape 返回正确的值
  std::vector<int64_t> current_shape;
  int32_t current_type = 0;
  status = executor.GetCurrentShape(current_shape, current_type);
  EXPECT_EQ(status, SUCCESS);
  EXPECT_EQ(current_shape.size(), 1U);
  EXPECT_EQ(current_shape[0], 4);
  EXPECT_EQ(current_type, dynamic_type);
}

TEST_F(Om2ModelExecutorUt, SetDynamicSize_DynamicHW_Success) {
  const std::string so_path = PathUtils::Join({test_work_dir_, "fake_runtime/libg1_om2.so"});
  const std::string om2_dynamic_path = PathUtils::Join({test_work_dir_, "dynamic_hw.om2"});

  ZipArchiveWriter zip_writer(om2_dynamic_path);
  ASSERT_TRUE(zip_writer.IsMemFileOpened());
  const auto manifest = MakeManifestJson();
  const auto model_meta = MakeModelMetaJsonWithDynamicHW();
  ASSERT_TRUE(zip_writer.WriteBytes("manifest.json", manifest.data(), manifest.size(), false));
  ASSERT_TRUE(zip_writer.WriteBytes("data/model_0/model_meta.json", model_meta.data(), model_meta.size(), false));
  ASSERT_TRUE(zip_writer.WriteFile("data/model_0/runtime/libg1_om2.so", so_path, false));
  ASSERT_TRUE(zip_writer.SaveModelDataToFile());

  uint32_t model_buf_size = 0U;
  auto model_buf = GetBinDataFromFile(om2_dynamic_path, model_buf_size);
  ASSERT_NE(model_buf, nullptr);
  ASSERT_GT(model_buf_size, 0U);

  ModelData model_data{};
  model_data.model_data = model_buf.get();
  model_data.model_len = model_buf_size;

  gert::Om2ModelExecutor executor;
  auto load_arg = MakeOm2LoadArg();
  ASSERT_EQ(executor.Load(model_data, load_arg, 1U), SUCCESS);

  // 设置有效档位 H=448, W=448
  std::vector<uint64_t> hw_num = {448, 448};
  int32_t dynamic_type = 2;  // DYNAMIC_IMAGE

  ge::Status status = executor.SetDynamicSize(hw_num, dynamic_type);
  EXPECT_EQ(status, SUCCESS);

  std::vector<int64_t> current_shape;
  int32_t current_type = 0;
  status = executor.GetCurrentShape(current_shape, current_type);
  EXPECT_EQ(status, SUCCESS);
  EXPECT_EQ(current_shape.size(), 2U);
  EXPECT_EQ(current_shape[0], 448);
  EXPECT_EQ(current_shape[1], 448);
  EXPECT_EQ(current_type, dynamic_type);
}

TEST_F(Om2ModelExecutorUt, SetDynamicSize_DynamicDims_Success) {
  const std::string so_path = PathUtils::Join({test_work_dir_, "fake_runtime/libg1_om2.so"});
  const std::string om2_dynamic_path = PathUtils::Join({test_work_dir_, "dynamic_dims.om2"});

  ZipArchiveWriter zip_writer(om2_dynamic_path);
  ASSERT_TRUE(zip_writer.IsMemFileOpened());
  const auto manifest = MakeManifestJson();
  const auto model_meta = MakeModelMetaJsonWithDynamicDims();
  ASSERT_TRUE(zip_writer.WriteBytes("manifest.json", manifest.data(), manifest.size(), false));
  ASSERT_TRUE(zip_writer.WriteBytes("data/model_0/model_meta.json", model_meta.data(), model_meta.size(), false));
  ASSERT_TRUE(zip_writer.WriteFile("data/model_0/runtime/libg1_om2.so", so_path, false));
  ASSERT_TRUE(zip_writer.SaveModelDataToFile());

  uint32_t model_buf_size = 0U;
  auto model_buf = GetBinDataFromFile(om2_dynamic_path, model_buf_size);
  ASSERT_NE(model_buf, nullptr);
  ASSERT_GT(model_buf_size, 0U);

  ModelData model_data{};
  model_data.model_data = model_buf.get();
  model_data.model_len = model_buf_size;

  gert::Om2ModelExecutor executor;
  auto load_arg = MakeOm2LoadArg();
  ASSERT_EQ(executor.Load(model_data, load_arg, 1U), SUCCESS);

  // 设置有效档位 [1, 256]
  std::vector<uint64_t> dims_num = {1, 256};
  int32_t dynamic_type = 3;  // DYNAMIC_DIMS

  ge::Status status = executor.SetDynamicSize(dims_num, dynamic_type);
  EXPECT_EQ(status, SUCCESS);

  std::vector<int64_t> current_shape;
  int32_t current_type = 0;
  status = executor.GetCurrentShape(current_shape, current_type);
  EXPECT_EQ(status, SUCCESS);
  EXPECT_EQ(current_shape.size(), 2U);
  EXPECT_EQ(current_shape[0], 1);
  EXPECT_EQ(current_shape[1], 256);
  EXPECT_EQ(current_type, dynamic_type);
}

TEST_F(Om2ModelExecutorUt, GetCurrentShape_BeforeSet_ReturnsFixed) {
  auto model_data_holder = LoadValidModelData();
  gert::Om2ModelExecutor executor;
  auto load_arg = MakeOm2LoadArg();
  ASSERT_EQ(executor.Load(model_data_holder.model_data, load_arg, 1U), SUCCESS);

  // 未调用 SetDynamicSize 时，GetCurrentShape 应该返回 FIXED 类型
  std::vector<int64_t> current_shape;
  int32_t current_type = -1;
  ge::Status status = executor.GetCurrentShape(current_shape, current_type);
  EXPECT_EQ(status, SUCCESS);
  EXPECT_TRUE(current_shape.empty());
  EXPECT_EQ(current_type, 0);  // FIXED = 0
}

TEST_F(Om2ModelExecutorUt, SetDynamicSize_EmptyBatchNum_ReturnsError) {
  // 创建包含动态 batch 信息的模型
  const std::string so_path = PathUtils::Join({test_work_dir_, "fake_runtime/libg1_om2.so"});
  const std::string om2_path = PathUtils::Join({test_work_dir_, "dynamic_batch_empty.om2"});

  ZipArchiveWriter zip_writer(om2_path);
  ASSERT_TRUE(zip_writer.IsMemFileOpened());
  const auto manifest = MakeManifestJson();
  const auto model_meta = MakeModelMetaJsonWithDynamicBatch();
  ASSERT_TRUE(zip_writer.WriteBytes("manifest.json", manifest.data(), manifest.size(), false));
  ASSERT_TRUE(zip_writer.WriteBytes("data/model_0/model_meta.json", model_meta.data(), model_meta.size(), false));
  ASSERT_TRUE(zip_writer.WriteFile("data/model_0/runtime/libg1_om2.so", so_path, false));
  ASSERT_TRUE(zip_writer.SaveModelDataToFile());

  uint32_t model_buf_size = 0U;
  auto model_buf = GetBinDataFromFile(om2_path, model_buf_size);
  ASSERT_NE(model_buf, nullptr);
  ASSERT_GT(model_buf_size, 0U);

  ModelData model_data{};
  model_data.model_data = model_buf.get();
  model_data.model_len = model_buf_size;

  gert::Om2ModelExecutor executor;
  auto load_arg = MakeOm2LoadArg();
  ASSERT_EQ(executor.Load(model_data, load_arg, 1U), SUCCESS);

  // 空 batch_num 应该返回错误（type 匹配但无有效档位）
  std::vector<uint64_t> empty_batch;
  int32_t dynamic_type = 1;  // DYNAMIC_BATCH
  ge::Status status = executor.SetDynamicSize(empty_batch, dynamic_type);
  EXPECT_NE(status, SUCCESS);
}

// 辅助 lambda：创建并加载包含动态 batch 信息的模型
static bool LoadDynamicBatchModel(const std::string &test_work_dir, const std::string &om2_suffix,
                                  gert::Om2ModelExecutor &executor) {
  const std::string so_path = PathUtils::Join({test_work_dir, "fake_runtime/libg1_om2.so"});
  const std::string om2_path = PathUtils::Join({test_work_dir, "dynamic_batch_" + om2_suffix + ".om2"});

  ZipArchiveWriter zip_writer(om2_path);
  if (!zip_writer.IsMemFileOpened()) return false;
  const auto manifest = MakeManifestJson();
  const auto model_meta = MakeModelMetaJsonWithDynamicBatch();
  if (!zip_writer.WriteBytes("manifest.json", manifest.data(), manifest.size(), false)) return false;
  if (!zip_writer.WriteBytes("data/model_0/model_meta.json", model_meta.data(), model_meta.size(), false)) return false;
  if (!zip_writer.WriteFile("data/model_0/runtime/libg1_om2.so", so_path, false)) return false;
  if (!zip_writer.SaveModelDataToFile()) return false;

  uint32_t model_buf_size = 0U;
  auto model_buf = GetBinDataFromFile(om2_path, model_buf_size);
  if (!model_buf || model_buf_size == 0U) return false;

  ModelData model_data{};
  model_data.model_data = model_buf.get();
  model_data.model_len = model_buf_size;

  auto load_arg = MakeOm2LoadArg();
  return executor.Load(model_data, load_arg, 1U) == SUCCESS;
}

TEST_F(Om2ModelExecutorUt, SetDynamicSize_MultipleUpdates_UpdatesCorrectly) {
  gert::Om2ModelExecutor executor;
  ASSERT_TRUE(LoadDynamicBatchModel(test_work_dir_, "multi", executor));

  // 第一次设置 batch=1
  std::vector<uint64_t> batch1 = {1};
  int32_t dynamic_type = 1;  // DYNAMIC_BATCH
  ge::Status status = executor.SetDynamicSize(batch1, dynamic_type);
  EXPECT_EQ(status, SUCCESS);

  std::vector<int64_t> current_shape;
  int32_t current_type = -1;
  status = executor.GetCurrentShape(current_shape, current_type);
  EXPECT_EQ(status, SUCCESS);
  EXPECT_EQ(current_shape.size(), 1U);
  EXPECT_EQ(current_shape[0], 1);
  EXPECT_EQ(current_type, dynamic_type);

  // 第二次设置 batch=2
  std::vector<uint64_t> batch2 = {2};
  status = executor.SetDynamicSize(batch2, dynamic_type);
  EXPECT_EQ(status, SUCCESS);

  current_shape.clear();
  current_type = -1;
  status = executor.GetCurrentShape(current_shape, current_type);
  EXPECT_EQ(status, SUCCESS);
  EXPECT_EQ(current_shape.size(), 1U);
  EXPECT_EQ(current_shape[0], 2);
  EXPECT_EQ(current_type, dynamic_type);
}

TEST_F(Om2ModelExecutorUt, SetDynamicSize_AllValidGears_Success) {
  gert::Om2ModelExecutor executor;
  ASSERT_TRUE(LoadDynamicBatchModel(test_work_dir_, "all_gears", executor));

  // 测试所有有效档位 [1, 2, 4, 8]
  std::vector<std::vector<uint64_t>> valid_gears = {{1}, {2}, {4}, {8}};
  int32_t dynamic_type = 1;  // DYNAMIC_BATCH

  for (const auto &gear : valid_gears) {
    ge::Status status = executor.SetDynamicSize(gear, dynamic_type);
    EXPECT_EQ(status, SUCCESS) << "Failed for gear size " << gear[0];

    std::vector<int64_t> current_shape;
    int32_t current_type = -1;
    status = executor.GetCurrentShape(current_shape, current_type);
    EXPECT_EQ(status, SUCCESS);
    EXPECT_EQ(current_shape.size(), 1U);
    EXPECT_EQ(current_shape[0], static_cast<int64_t>(gear[0]));
    EXPECT_EQ(current_type, dynamic_type);
  }
}

TEST_F(Om2ModelExecutorUt, SetDynamicSize_InvalidGears_AllFail) {
  gert::Om2ModelExecutor executor;
  ASSERT_TRUE(LoadDynamicBatchModel(test_work_dir_, "invalid", executor));

  // 测试所有无效档位 [3, 5, 6, 7, 9, 10]
  std::vector<std::vector<uint64_t>> invalid_gears = {{3}, {5}, {6}, {7}, {9}, {10}};
  int32_t dynamic_type = 1;  // DYNAMIC_BATCH

  for (const auto &gear : invalid_gears) {
    ge::Status status = executor.SetDynamicSize(gear, dynamic_type);
    EXPECT_NE(status, SUCCESS) << "Should fail for invalid gear size " << gear[0];
  }
}

TEST_F(Om2ModelExecutorUt, SetDynamicSize_TypeMismatch_AllTypes) {
  gert::Om2ModelExecutor executor;
  ASSERT_TRUE(LoadDynamicBatchModel(test_work_dir_, "mismatch", executor));

  // 模型是 DYNAMIC_BATCH (type=1)，测试其他类型不匹配
  std::vector<uint64_t> batch = {1};

  // FIXED (type=0)
  ge::Status status = executor.SetDynamicSize(batch, 0);
  EXPECT_NE(status, SUCCESS);

  // DYNAMIC_IMAGE (type=2)
  status = executor.SetDynamicSize(batch, 2);
  EXPECT_NE(status, SUCCESS);

  // DYNAMIC_DIMS (type=3)
  status = executor.SetDynamicSize(batch, 3);
  EXPECT_NE(status, SUCCESS);
}

TEST_F(Om2ModelExecutorUt, GetCurrentShape_AfterFailedSet_ReturnsEmpty) {
  gert::Om2ModelExecutor executor;
  ASSERT_TRUE(LoadDynamicBatchModel(test_work_dir_, "fail_shape", executor));

  // 设置失败后，GetCurrentShape 应该仍然返回空
  std::vector<uint64_t> invalid_batch = {3};
  int32_t dynamic_type = 1;  // DYNAMIC_BATCH
  ge::Status status = executor.SetDynamicSize(invalid_batch, dynamic_type);
  EXPECT_NE(status, SUCCESS);

  std::vector<int64_t> current_shape;
  int32_t current_type = -1;
  status = executor.GetCurrentShape(current_shape, current_type);
  EXPECT_EQ(status, SUCCESS);
  EXPECT_TRUE(current_shape.empty());
  EXPECT_EQ(current_type, 0);  // FIXED = 0
}

// ============================================================================
// OM2 AIPP Method Tests
// ============================================================================

namespace {
constexpr const char *kAippModelMetaJson = R"({
    "inputs": [
        {
            "data_type": "DT_FLOAT",
            "format": "NCHW",
            "index": 0,
            "name": "data1",
            "shape": [1, 3, 224, 224],
            "shape_range": [],
            "shape_aclmdlGetInputDimsV2": [1, 3, 224, 224],
            "size": 0
        }
    ],
    "name": "g1",
    "outputs": [
        {
            "data_type": "DT_FLOAT",
            "format": "ND",
            "index": 0,
            "name": "output_0",
            "shape": [1, 1000],
            "shape_range": [],
            "shape_aclmdlGetInputDimsV2": [1, 1000],
            "size": 0
        }
    ],
    "work_size": 2048,
    "zero_copy_size": 0
})";

constexpr const char *kAippJsonSectionStatic = R"("aipp": {
    "aipp_infos": [
      {
        "index": 0,
        "aipp_type": 1,
        "aipp_data_index": 0,
        "aipp_mode": 1,
        "input_format": 2,
        "src_image_size_w": 640,
        "src_image_size_h": 480,
        "crop": 0,
        "load_start_pos_w": 0,
        "load_start_pos_h": 0,
        "crop_size_w": 0,
        "crop_size_h": 0,
        "resize": 0,
        "resize_output_w": 0,
        "resize_output_h": 0,
        "padding": 0,
        "left_padding_size": 0,
        "right_padding_size": 0,
        "top_padding_size": 0,
        "bottom_padding_size": 0,
        "csc_switch": 1,
        "rbuv_swap_switch": 0,
        "ax_swap_switch": 0,
        "single_line_mode": 0,
        "matrix_r0c0": 256,
        "matrix_r0c1": 0,
        "matrix_r0c2": 0,
        "matrix_r1c0": 0,
        "matrix_r1c1": 256,
        "matrix_r1c2": 0,
        "matrix_r2c0": 0,
        "matrix_r2c1": 0,
        "matrix_r2c2": 256,
        "output_bias_0": 16,
        "output_bias_1": 128,
        "output_bias_2": 128,
        "input_bias_0": 16,
        "input_bias_1": 128,
        "input_bias_2": 128,
        "mean_chn_0": 104,
        "mean_chn_1": 117,
        "mean_chn_2": 123,
        "mean_chn_3": 0,
        "min_chn_0": 0.0,
        "min_chn_1": 0.0,
        "min_chn_2": 0.0,
        "min_chn_3": 0.0,
        "var_reci_chn_0": 1.0,
        "var_reci_chn_1": 1.0,
        "var_reci_chn_2": 1.0,
        "var_reci_chn_3": 1.0,
        "support_rotation": 0,
        "related_input_rank": 0,
        "max_src_image_size": 8192,
        "orig_input_format": 0,
        "orig_input_data_type": 0,
        "orig_input_dim_num": 4,
        "aipp_inputs": ["NCHW:DT_FLOAT:tensor_0:100:3:1,3,224,224"],
        "aipp_outputs": ["NCHW:DT_FLOAT:tensor_0_out:200:3:1,3,224,224"]
      }
    ]
  })";

constexpr const char *kAippJsonSectionDynamic = R"("aipp": {
    "aipp_infos": [
      {
        "index": 0,
        "aipp_type": 2,
        "aipp_data_index": 0,
        "aipp_mode": 1,
        "input_format": 2,
        "src_image_size_w": 640,
        "src_image_size_h": 480,
        "crop": 0,
        "load_start_pos_w": 0,
        "load_start_pos_h": 0,
        "crop_size_w": 0,
        "crop_size_h": 0,
        "resize": 0,
        "resize_output_w": 0,
        "resize_output_h": 0,
        "padding": 0,
        "left_padding_size": 0,
        "right_padding_size": 0,
        "top_padding_size": 0,
        "bottom_padding_size": 0,
        "csc_switch": 1,
        "rbuv_swap_switch": 0,
        "ax_swap_switch": 0,
        "single_line_mode": 0,
        "matrix_r0c0": 256,
        "matrix_r0c1": 0,
        "matrix_r0c2": 0,
        "matrix_r1c0": 0,
        "matrix_r1c1": 256,
        "matrix_r1c2": 0,
        "matrix_r2c0": 0,
        "matrix_r2c1": 0,
        "matrix_r2c2": 256,
        "output_bias_0": 16,
        "output_bias_1": 128,
        "output_bias_2": 128,
        "input_bias_0": 16,
        "input_bias_1": 128,
        "input_bias_2": 128,
        "mean_chn_0": 104,
        "mean_chn_1": 117,
        "mean_chn_2": 123,
        "mean_chn_3": 0,
        "min_chn_0": 0.0,
        "min_chn_1": 0.0,
        "min_chn_2": 0.0,
        "min_chn_3": 0.0,
        "var_reci_chn_0": 1.0,
        "var_reci_chn_1": 1.0,
        "var_reci_chn_2": 1.0,
        "var_reci_chn_3": 1.0,
        "support_rotation": 0,
        "related_input_rank": 0,
        "max_src_image_size": 8192,
        "orig_input_format": 0,
        "orig_input_data_type": 0,
        "orig_input_dim_num": 4,
        "aipp_inputs": ["NCHW:DT_FLOAT:tensor_0:100:3:1,3,224,224"],
        "aipp_outputs": ["NCHW:DT_FLOAT:tensor_0_out:200:3:1,3,224,224"]
      }
    ]
  })";

constexpr const char *kAippJsonSectionDynamicConf = R"("aipp": {
    "aipp_infos": [
      {
        "index": 0,
        "aipp_type": 3,
        "aipp_data_index": 0,
        "aipp_mode": 1,
        "input_format": 2,
        "src_image_size_w": 640,
        "src_image_size_h": 480,
        "crop": 0,
        "load_start_pos_w": 0,
        "load_start_pos_h": 0,
        "crop_size_w": 0,
        "crop_size_h": 0,
        "resize": 0,
        "resize_output_w": 0,
        "resize_output_h": 0,
        "padding": 0,
        "left_padding_size": 0,
        "right_padding_size": 0,
        "top_padding_size": 0,
        "bottom_padding_size": 0,
        "csc_switch": 1,
        "rbuv_swap_switch": 0,
        "ax_swap_switch": 0,
        "single_line_mode": 0,
        "matrix_r0c0": 256,
        "matrix_r0c1": 0,
        "matrix_r0c2": 0,
        "matrix_r1c0": 0,
        "matrix_r1c1": 256,
        "matrix_r1c2": 0,
        "matrix_r2c0": 0,
        "matrix_r2c1": 0,
        "matrix_r2c2": 256,
        "output_bias_0": 16,
        "output_bias_1": 128,
        "output_bias_2": 128,
        "input_bias_0": 16,
        "input_bias_1": 128,
        "input_bias_2": 128,
        "mean_chn_0": 104,
        "mean_chn_1": 117,
        "mean_chn_2": 123,
        "mean_chn_3": 0,
        "min_chn_0": 0.0,
        "min_chn_1": 0.0,
        "min_chn_2": 0.0,
        "min_chn_3": 0.0,
        "var_reci_chn_0": 1.0,
        "var_reci_chn_1": 1.0,
        "var_reci_chn_2": 1.0,
        "var_reci_chn_3": 1.0,
        "support_rotation": 0,
        "related_input_rank": 0,
        "max_src_image_size": 8192,
        "orig_input_format": 0,
        "orig_input_data_type": 0,
        "orig_input_dim_num": 4,
        "aipp_inputs": ["NCHW:DT_FLOAT:tensor_0:100:3:1,3,224,224"],
        "aipp_outputs": ["NCHW:DT_FLOAT:tensor_0_out:200:3:1,3,224,224"]
      }
    ]
  })";

std::string MakeAippJsonSection(const bool is_dynamic, const int32_t aipp_type) {
  // 根据 aipp_type 选择预构建的 JSON，避免运行时字符串拼接触发 clang-tidy
  if (is_dynamic && aipp_type == 2) {
    return kAippJsonSectionDynamic;
  }
  if (aipp_type == 3) {
    return kAippJsonSectionDynamicConf;
  }
  return kAippJsonSectionStatic;
}

std::string MakeModelMetaJsonWithAipp(const bool is_dynamic, const int32_t aipp_type) {
  std::string base_json = kAippModelMetaJson;
  // 找到最后一个 '}'（即 JSON 根对象的闭合括号）并移除它
  const size_t last_brace = base_json.rfind('}');
  if (last_brace == std::string::npos) {
    return base_json;
  }
  // 移除根对象闭合括号，追加 aipp 字段和新闭合括号
  std::string json_without_root = base_json.substr(0, last_brace);
  std::string result = json_without_root + ",\n  " + MakeAippJsonSection(is_dynamic, aipp_type) + "\n}";
  return result;
}

static bool LoadAippModel(const std::string &test_work_dir, const bool is_dynamic, const int32_t aipp_type,
                          gert::Om2ModelExecutor &executor) {
  const std::string so_path = PathUtils::Join({test_work_dir, "fake_runtime/libg1_om2.so"});
  const std::string om2_suffix =
      std::string("aipp_") + (is_dynamic ? "dynamic_" : "static_") + std::to_string(aipp_type);
  const std::string om2_path = PathUtils::Join({test_work_dir, om2_suffix + ".om2"});

  ZipArchiveWriter zip_writer(om2_path);
  if (!zip_writer.IsMemFileOpened()) {
    return false;
  }
  const auto manifest = MakeManifestJson();
  const auto model_meta = MakeModelMetaJsonWithAipp(is_dynamic, aipp_type);
  if (!zip_writer.WriteBytes("manifest.json", manifest.data(), manifest.size(), false)) {
    return false;
  }
  if (!zip_writer.WriteBytes("data/model_0/model_meta.json", model_meta.data(), model_meta.size(), false)) {
    return false;
  }
  if (!zip_writer.WriteFile("data/model_0/runtime/libg1_om2.so", so_path, false)) {
    return false;
  }
  if (!zip_writer.SaveModelDataToFile()) {
    return false;
  }

  uint32_t model_buf_size = 0U;
  auto model_buf = GetBinDataFromFile(om2_path, model_buf_size);
  if (!model_buf || model_buf_size == 0U) {
    return false;
  }

  ModelData model_data{};
  model_data.model_data = model_buf.get();
  model_data.model_len = model_buf_size;

  auto load_arg = MakeOm2LoadArg();
  return executor.Load(model_data, load_arg, 1U) == SUCCESS;
}
}  // namespace

TEST_F(Om2ModelExecutorUt, GetAippInfo_NoAipp_ReturnsNotExist) {
  gert::Om2ModelExecutor executor;
  ASSERT_TRUE(LoadDynamicBatchModel(test_work_dir_, "no_aipp", executor));

  ge::AippConfigInfo aipp_info{};
  const auto status = executor.GetAippInfo(0U, aipp_info);
  EXPECT_EQ(status, ACL_ERROR_GE_AIPP_NOT_EXIST);
}

TEST_F(Om2ModelExecutorUt, GetAippInfo_IndexOutOfRange_ReturnsNotExist) {
  gert::Om2ModelExecutor executor;
  // 模型有 1 个 AIPP 条目在 index 0，查询 index 5
  ASSERT_TRUE(LoadAippModel(test_work_dir_, false, 1, executor));

  ge::AippConfigInfo aipp_info{};
  const auto status = executor.GetAippInfo(5U, aipp_info);
  EXPECT_EQ(status, ACL_ERROR_GE_AIPP_NOT_EXIST);
}

TEST_F(Om2ModelExecutorUt, GetAippInfo_StaticAipp_ReturnsCorrectConfig) {
  gert::Om2ModelExecutor executor;
  ASSERT_TRUE(LoadAippModel(test_work_dir_, false, 1, executor));

  ge::AippConfigInfo aipp_info{};
  const auto status = executor.GetAippInfo(0U, aipp_info);
  EXPECT_EQ(status, SUCCESS);
  EXPECT_EQ(aipp_info.input_format, 2);
  EXPECT_EQ(aipp_info.src_image_size_w, 640);
  EXPECT_EQ(aipp_info.src_image_size_h, 480);
  EXPECT_EQ(aipp_info.csc_switch, 1);
  EXPECT_EQ(aipp_info.max_src_image_size, 8192U);
}

TEST_F(Om2ModelExecutorUt, GetAippInfo_DynamicAipp_ReturnsCorrectConfig) {
  gert::Om2ModelExecutor executor;
  ASSERT_TRUE(LoadAippModel(test_work_dir_, true, 2, executor));

  ge::AippConfigInfo aipp_info{};
  const auto status = executor.GetAippInfo(0U, aipp_info);
  EXPECT_EQ(status, SUCCESS);
  EXPECT_EQ(aipp_info.input_format, 2);
  EXPECT_EQ(aipp_info.src_image_size_w, 640);
  EXPECT_EQ(aipp_info.src_image_size_h, 480);
}

TEST_F(Om2ModelExecutorUt, GetAippType_NoAipp_ReturnsDataWithoutAipp) {
  gert::Om2ModelExecutor executor;
  ASSERT_TRUE(LoadDynamicBatchModel(test_work_dir_, "a_type_noaip", executor));

  ge::InputAippType aipp_type = ge::DYNAMIC_AIPP_NODE;
  size_t aipp_data_index = 999U;
  const auto status = executor.GetAippType(0U, aipp_type, aipp_data_index);
  EXPECT_EQ(status, SUCCESS);
  EXPECT_EQ(aipp_type, ge::DATA_WITHOUT_AIPP);
  EXPECT_EQ(aipp_data_index, 0xFFFFFFFFU);
}

TEST_F(Om2ModelExecutorUt, GetAippType_StaticAipp_ReturnsCorrectType) {
  gert::Om2ModelExecutor executor;
  // aipp_type=1 → DATA_WITH_STATIC_AIPP
  ASSERT_TRUE(LoadAippModel(test_work_dir_, false, 1, executor));

  ge::InputAippType aipp_type = ge::DATA_WITHOUT_AIPP;
  size_t aipp_data_index = 0U;
  const auto status = executor.GetAippType(0U, aipp_type, aipp_data_index);
  EXPECT_EQ(status, SUCCESS);
  EXPECT_EQ(aipp_type, ge::DATA_WITH_STATIC_AIPP);
  EXPECT_EQ(aipp_data_index, 0U);
}

TEST_F(Om2ModelExecutorUt, GetAippType_DynamicAipp_ReturnsCorrectType) {
  gert::Om2ModelExecutor executor;
  // aipp_type=2 → DATA_WITH_DYNAMIC_AIPP
  ASSERT_TRUE(LoadAippModel(test_work_dir_, true, 2, executor));

  ge::InputAippType aipp_type = ge::DATA_WITHOUT_AIPP;
  size_t aipp_data_index = 0U;
  const auto status = executor.GetAippType(0U, aipp_type, aipp_data_index);
  EXPECT_EQ(status, SUCCESS);
  EXPECT_EQ(aipp_type, ge::DATA_WITH_DYNAMIC_AIPP);
  EXPECT_EQ(aipp_data_index, 0U);
}

TEST_F(Om2ModelExecutorUt, GetAippType_DynamicAippConf_ReturnsCorrectType) {
  gert::Om2ModelExecutor executor;
  // aipp_type=3 → DYNAMIC_AIPP_NODE
  ASSERT_TRUE(LoadAippModel(test_work_dir_, false, 3, executor));

  ge::InputAippType aipp_type = ge::DATA_WITHOUT_AIPP;
  size_t aipp_data_index = 0U;
  const auto status = executor.GetAippType(0U, aipp_type, aipp_data_index);
  EXPECT_EQ(status, SUCCESS);
  EXPECT_EQ(aipp_type, ge::DYNAMIC_AIPP_NODE);
}

TEST_F(Om2ModelExecutorUt, GetOrigInputInfo_NoAipp_ReturnsNotExist) {
  gert::Om2ModelExecutor executor;
  ASSERT_TRUE(LoadDynamicBatchModel(test_work_dir_, "noaipp_orig", executor));

  ge::OriginInputInfo orig_info{};
  const auto status = executor.GetOrigInputInfo(0U, orig_info);
  EXPECT_EQ(status, ACL_ERROR_GE_AIPP_NOT_EXIST);
}

TEST_F(Om2ModelExecutorUt, GetOrigInputInfo_ValidAipp_ReturnsCorrectInfo) {
  gert::Om2ModelExecutor executor;
  ASSERT_TRUE(LoadAippModel(test_work_dir_, false, 1, executor));

  ge::OriginInputInfo orig_info{};
  const auto status = executor.GetOrigInputInfo(0U, orig_info);
  EXPECT_EQ(status, SUCCESS);
  EXPECT_EQ(orig_info.format, ge::FORMAT_NCHW);
  EXPECT_EQ(orig_info.data_type, ge::DT_FLOAT);
  EXPECT_EQ(orig_info.dim_num, 4U);
}

TEST_F(Om2ModelExecutorUt, GetAllAippInputOutputDims_NoAipp_ReturnsNotExist) {
  gert::Om2ModelExecutor executor;
  ASSERT_TRUE(LoadDynamicBatchModel(test_work_dir_, "noaipp_dims", executor));

  std::vector<ge::InputOutputDims> input_dims;
  std::vector<ge::InputOutputDims> output_dims;
  const auto status = executor.GetAllAippInputOutputDims(0U, input_dims, output_dims);
  EXPECT_EQ(status, ACL_ERROR_GE_AIPP_NOT_EXIST);
}

TEST_F(Om2ModelExecutorUt, GetAllAippInputOutputDims_ValidAipp_ReturnsCorrectDims) {
  gert::Om2ModelExecutor executor;
  ASSERT_TRUE(LoadAippModel(test_work_dir_, false, 1, executor));

  std::vector<ge::InputOutputDims> input_dims;
  std::vector<ge::InputOutputDims> output_dims;
  const auto status = executor.GetAllAippInputOutputDims(0U, input_dims, output_dims);
  EXPECT_EQ(status, SUCCESS);
  EXPECT_EQ(input_dims.size(), 1U);
  EXPECT_EQ(output_dims.size(), 1U);
  EXPECT_EQ(input_dims[0].name, "tensor_0");
  EXPECT_EQ(input_dims[0].size, 100U);
  EXPECT_EQ(input_dims[0].dim_num, 3U);
  EXPECT_EQ(output_dims[0].name, "tensor_0_out");
  EXPECT_EQ(output_dims[0].size, 200U);
  EXPECT_EQ(output_dims[0].dim_num, 3U);
}

TEST_F(Om2ModelExecutorUt, GetBatchInfoSize_NoBatchInfo_ReturnsOne) {
  gert::Om2ModelExecutor executor;
  ASSERT_TRUE(LoadAippModel(test_work_dir_, false, 1, executor));

  size_t shape_count = 0U;
  const auto status = executor.GetBatchInfoSize(shape_count);
  EXPECT_EQ(status, SUCCESS);
  // 模型没有 dynamic_batch_info 时，shape_count 应为 1
  EXPECT_EQ(shape_count, 1U);
}

TEST_F(Om2ModelExecutorUt, GetBatchInfoSize_WithBatchInfo_ReturnsCorrectSize) {
  gert::Om2ModelExecutor executor;
  // 使用动态 batch 模型（4 个档位: 1, 2, 4, 8）
  ASSERT_TRUE(LoadDynamicBatchModel(test_work_dir_, "batch_size", executor));

  size_t shape_count = 0U;
  const auto status = executor.GetBatchInfoSize(shape_count);
  EXPECT_EQ(status, SUCCESS);
  EXPECT_EQ(shape_count, 4U);
}

TEST_F(Om2ModelExecutorUt, SetDynamicAippData_NullAddr_ReturnsError) {
  gert::Om2ModelExecutor executor;
  ASSERT_TRUE(LoadAippModel(test_work_dir_, true, 2, executor));

  const std::vector<kAippDynamicBatchPara> batch_para(1);
  const kAippDynamicPara aipp_parms{};
  const auto status = executor.SetDynamicAippData(nullptr, 1024U, batch_para, aipp_parms);
  EXPECT_EQ(status, ACL_ERROR_GE_DYNAMIC_INPUT_ADDR_INVALID);
}

TEST_F(Om2ModelExecutorUt, SetDynamicAippData_EmptyBatchPara_ReturnsError) {
  gert::Om2ModelExecutor executor;
  ASSERT_TRUE(LoadAippModel(test_work_dir_, true, 2, executor));

  uint8_t dummy_buffer[4096] = {0};
  const std::vector<kAippDynamicBatchPara> empty_batch;
  const kAippDynamicPara aipp_parms{};
  const auto status = executor.SetDynamicAippData(dummy_buffer, sizeof(dummy_buffer), empty_batch, aipp_parms);
  EXPECT_EQ(status, ACL_ERROR_GE_AIPP_BATCH_EMPTY);
}

TEST_F(Om2ModelExecutorUt, SetDynamicAippData_StructLargerThanLength_ReturnsError) {
  gert::Om2ModelExecutor executor;
  ASSERT_TRUE(LoadAippModel(test_work_dir_, true, 2, executor));

  uint8_t dummy_buffer[8] = {0};
  const std::vector<kAippDynamicBatchPara> batch_para(10);
  const kAippDynamicPara aipp_parms{};
  // struct_len = 10 * sizeof(kAippDynamicBatchPara) + (sizeof(kAippDynamicPara) - sizeof(kAippDynamicBatchPara))
  // far exceeds length=8
  const auto status = executor.SetDynamicAippData(dummy_buffer, 8U, batch_para, aipp_parms);
  EXPECT_EQ(status, ACL_ERROR_GE_DYNAMIC_INPUT_LENGTH_INVALID);
}

TEST_F(Om2ModelExecutorUt, make_om2_load_arg_has_reuse_zero_copy_false_by_default) {
  auto load_arg = MakeOm2LoadArg();
  EXPECT_FALSE(load_arg.reuse_zero_copy);
}

TEST_F(Om2ModelExecutorUt, load_with_reuse_zero_copy_true_and_internal_work_allocation) {
  auto model_data_holder = LoadValidModelData();
  gert::Om2ModelExecutor executor;
  auto load_arg = MakeOm2LoadArg();
  load_arg.reuse_zero_copy = true;

  EnvValueGuard guard_mode("OM2_EXPECT_WORK_PTR_MODE");
  ASSERT_EQ(setenv("OM2_EXPECT_WORK_PTR_MODE", "NON_NULL", 1), 0);

  EXPECT_EQ(executor.Load(model_data_holder.model_data, load_arg, 1U), SUCCESS);
}

TEST_F(Om2ModelExecutorUt, load_with_reuse_zero_copy_false_and_internal_work_allocation) {
  auto model_data_holder = LoadValidModelData();
  gert::Om2ModelExecutor executor;
  auto load_arg = MakeOm2LoadArg();
  load_arg.reuse_zero_copy = false;

  EnvValueGuard guard_mode("OM2_EXPECT_WORK_PTR_MODE");
  ASSERT_EQ(setenv("OM2_EXPECT_WORK_PTR_MODE", "NON_NULL", 1), 0);

  EXPECT_EQ(executor.Load(model_data_holder.model_data, load_arg, 1U), SUCCESS);
}

TEST_F(Om2ModelExecutorUt, load_with_reuse_zero_copy_true_and_external_work_ptr) {
  auto model_data_holder = LoadValidModelData();
  gert::Om2ModelExecutor executor;
  auto load_arg = MakeOm2LoadArg();
  load_arg.reuse_zero_copy = true;
  std::vector<uint8_t> external_work(4096U, 0U);
  load_arg.work_ptr = external_work.data();
  load_arg.work_size = external_work.size();

  EnvValueGuard guard_mode("OM2_EXPECT_WORK_PTR_MODE");
  EnvValueGuard guard_value("OM2_EXPECT_WORK_PTR_VALUE");
  ASSERT_EQ(setenv("OM2_EXPECT_WORK_PTR_MODE", "EQUAL", 1), 0);
  const std::string expected_ptr = PtrToHexString(external_work.data());
  ASSERT_EQ(setenv("OM2_EXPECT_WORK_PTR_VALUE", expected_ptr.c_str(), 1), 0);

  EXPECT_EQ(executor.Load(model_data_holder.model_data, load_arg, 1U), SUCCESS);
}

}  // namespace ge

namespace gert {
namespace om2 {

class Om2AippUtilsUt : public ::testing::Test {
 protected:
  void SetUp() override {}
  void TearDown() override {}
};

TEST_F(Om2AippUtilsUt, ParseAippDimInfo_InvalidPartsCount) {
  ge::InputOutputDims dims_info;
  auto status = ParseAippDimInfo("too:few", dims_info);
  EXPECT_EQ(status, ge::FAILED);
}

TEST_F(Om2AippUtilsUt, ParseAippDimInfo_EmptyDimInShape) {
  ge::InputOutputDims dims_info;
  auto status = ParseAippDimInfo("NCHW:DT_FLOAT:data:0:4:1,,3,224", dims_info);
  EXPECT_EQ(status, ge::SUCCESS);
  EXPECT_EQ(dims_info.name, "data");
  EXPECT_EQ(dims_info.dim_num, 4U);
  EXPECT_EQ(dims_info.dims.size(), 3U);
}

TEST_F(Om2AippUtilsUt, ParseAippConfigFromJson_AllFields) {
  ge::JsonFile entry;
  entry.Set("aipp_mode", static_cast<int8_t>(1));
  entry.Set("input_format", static_cast<int8_t>(1));
  entry.Set("src_image_size_w", static_cast<int32_t>(224));
  entry.Set("src_image_size_h", static_cast<int32_t>(224));
  entry.Set("crop", static_cast<int8_t>(1));
  entry.Set("csc_switch", static_cast<int8_t>(1));
  entry.Set("matrix_r0c0", static_cast<int32_t>(1));
  entry.Set("matrix_r2c2", static_cast<int32_t>(9));
  entry.Set("mean_chn_0", static_cast<int32_t>(128));
  entry.Set("min_chn_0", static_cast<float32_t>(0.0F));
  entry.Set("var_reci_chn_0", static_cast<float32_t>(1.0F));
  entry.Set("max_src_image_size", static_cast<uint32_t>(4096U));
  entry.Set("support_rotation", static_cast<int8_t>(0));
  entry.Set("related_input_rank", static_cast<uint32_t>(0U));

  auto result = ParseAippConfigFromJson(entry);
  EXPECT_EQ(result.aipp_mode, 1);
  EXPECT_EQ(result.csc_switch, 1);
  EXPECT_EQ(result.matrix_r0c0, 1);
  EXPECT_EQ(result.matrix_r2c2, 9);
  EXPECT_EQ(result.mean_chn_0, 128);
  EXPECT_EQ(result.max_src_image_size, 4096U);
}

TEST_F(Om2AippUtilsUt, ParseOriginInputFromJson_Valid) {
  ge::JsonFile entry;
  entry.Set("orig_input_format", static_cast<int32_t>(0));
  entry.Set("orig_input_data_type", static_cast<int32_t>(1));
  entry.Set("orig_input_dim_num", static_cast<uint32_t>(4U));

  auto result = ParseOriginInputFromJson(entry);
  EXPECT_EQ(result.format, static_cast<ge::Format>(0));
  EXPECT_EQ(result.data_type, static_cast<ge::DataType>(1));
  EXPECT_EQ(result.dim_num, 4U);
}

TEST_F(Om2AippUtilsUt, ParseAippDimsFromJson_Valid) {
  ge::JsonFile entry;
  entry.Set("aipp_inputs", std::vector<std::string>{"NCHW:DT_FLOAT:data:0:4:1,3,224,224"});

  auto result = ParseAippDimsFromJson(entry, "aipp_inputs");
  EXPECT_EQ(result.size(), 1U);
  EXPECT_EQ(result[0].name, "data");
  EXPECT_EQ(result[0].dim_num, 4U);
  EXPECT_EQ(result[0].dims.size(), 4U);
}

TEST_F(Om2AippUtilsUt, ParseAippDimsFromJson_WithInvalidEntry) {
  ge::JsonFile entry;
  entry.Set("aipp_inputs", std::vector<std::string>{"invalid", "NCHW:DT_FLOAT:data:0:4:1,3,224"});

  auto result = ParseAippDimsFromJson(entry, "aipp_inputs");
  EXPECT_EQ(result.size(), 1U);
  EXPECT_EQ(result[0].name, "data");
}

TEST_F(Om2AippUtilsUt, ParseAippJson_NoAippInfosKey) {
  ge::JsonFile aipp_json;
  std::vector<Om2AippMeta> aipp_infos;
  bool has_aipp = false;

  auto status = ParseAippJson(aipp_json, aipp_infos, has_aipp);
  EXPECT_EQ(status, ge::FAILED);
  EXPECT_FALSE(has_aipp);
}

TEST_F(Om2AippUtilsUt, ParseAippJson_AippInfosNotArray) {
  ge::JsonFile aipp_json;
  aipp_json.Set("aipp_infos", "not_an_array");
  std::vector<Om2AippMeta> aipp_infos;
  bool has_aipp = false;

  auto status = ParseAippJson(aipp_json, aipp_infos, has_aipp);
  EXPECT_EQ(status, ge::SUCCESS);
  EXPECT_FALSE(has_aipp);
}

TEST_F(Om2AippUtilsUt, ParseAippJson_EmptyAippInfos) {
  ge::JsonFile aipp_json;
  aipp_json.Set("aipp_infos", nlohmann::json::array());
  std::vector<Om2AippMeta> aipp_infos;
  bool has_aipp = false;

  auto status = ParseAippJson(aipp_json, aipp_infos, has_aipp);
  EXPECT_EQ(status, ge::SUCCESS);
  EXPECT_TRUE(has_aipp);
}

TEST_F(Om2AippUtilsUt, ParseAippJson_ValidAippInfo) {
  nlohmann::json aipp_item;
  aipp_item["index"] = 0;
  aipp_item["aipp_type"] = 1;
  aipp_item["aipp_data_index"] = 0;
  aipp_item["aipp_mode"] = 1;
  aipp_item["input_format"] = 1;
  aipp_item["src_image_size_w"] = 224;
  aipp_item["src_image_size_h"] = 224;
  aipp_item["csc_switch"] = 1;
  aipp_item["matrix_r0c0"] = 1;
  aipp_item["matrix_r2c2"] = 9;
  aipp_item["mean_chn_0"] = 128;
  aipp_item["min_chn_0"] = 0.0F;
  aipp_item["var_reci_chn_0"] = 1.0F;
  aipp_item["max_src_image_size"] = 4096;
  aipp_item["orig_input_format"] = 0;
  aipp_item["orig_input_data_type"] = 1;
  aipp_item["orig_input_dim_num"] = 4;
  aipp_item["aipp_inputs"] = nlohmann::json::array({"NCHW:DT_FLOAT:data:0:4:1,3,224,224"});
  aipp_item["aipp_outputs"] = nlohmann::json::array();

  nlohmann::json root_json;
  root_json["aipp_infos"] = nlohmann::json::array({aipp_item});

  ge::JsonFile aipp_json(root_json);
  std::vector<Om2AippMeta> aipp_infos;
  bool has_aipp = false;

  auto status = ParseAippJson(aipp_json, aipp_infos, has_aipp);
  EXPECT_EQ(status, ge::SUCCESS);
  EXPECT_TRUE(has_aipp);
  EXPECT_EQ(aipp_infos.size(), 1U);
  EXPECT_EQ(aipp_infos[0].aipp_type, ge::DATA_WITH_STATIC_AIPP);
  EXPECT_EQ(aipp_infos[0].aipp_config_info.aipp_mode, 1);
  EXPECT_EQ(aipp_infos[0].aipp_input_dims.size(), 1U);
  EXPECT_EQ(aipp_infos[0].orig_input_info.dim_num, 4U);
}

TEST_F(Om2AippUtilsUt, ParseAippJson_SkipNonObjectItem) {
  nlohmann::json root_json;
  root_json["aipp_infos"] = nlohmann::json::array({"not_an_object", 42});

  ge::JsonFile aipp_json(root_json);
  std::vector<Om2AippMeta> aipp_infos;
  bool has_aipp = false;

  auto status = ParseAippJson(aipp_json, aipp_infos, has_aipp);
  EXPECT_EQ(status, ge::SUCCESS);
  EXPECT_TRUE(has_aipp);
}

TEST_F(Om2AippUtilsUt, ParseAippJson_IndexOutOfBounds) {
  nlohmann::json aipp_item;
  aipp_item["index"] = 2;
  aipp_item["aipp_type"] = 1;
  aipp_item["aipp_data_index"] = 0;
  aipp_item["aipp_mode"] = 1;
  aipp_item["input_format"] = 1;
  aipp_item["src_image_size_w"] = 224;
  aipp_item["src_image_size_h"] = 224;
  aipp_item["csc_switch"] = 0;
  aipp_item["matrix_r0c0"] = 0;
  aipp_item["matrix_r2c2"] = 0;
  aipp_item["mean_chn_0"] = 0;
  aipp_item["min_chn_0"] = 0.0F;
  aipp_item["var_reci_chn_0"] = 0.0F;
  aipp_item["max_src_image_size"] = 0;
  aipp_item["orig_input_format"] = 0;
  aipp_item["orig_input_data_type"] = 0;
  aipp_item["orig_input_dim_num"] = 0;
  aipp_item["aipp_inputs"] = nlohmann::json::array();
  aipp_item["aipp_outputs"] = nlohmann::json::array();

  nlohmann::json root_json;
  root_json["aipp_infos"] = nlohmann::json::array({aipp_item});

  ge::JsonFile aipp_json(root_json);
  std::vector<Om2AippMeta> aipp_infos;
  bool has_aipp = false;

  auto status = ParseAippJson(aipp_json, aipp_infos, has_aipp);
  EXPECT_EQ(status, ge::SUCCESS);
  EXPECT_EQ(aipp_infos.size(), 3U);
}

TEST_F(Om2AippUtilsUt, ParseAippJson_MultipleValidItems) {
  nlohmann::json item0;
  item0["index"] = 0;
  item0["aipp_type"] = 1;
  item0["aipp_data_index"] = 0;
  item0["aipp_mode"] = 1;
  item0["input_format"] = 1;
  item0["src_image_size_w"] = 224;
  item0["src_image_size_h"] = 224;
  item0["csc_switch"] = 0;
  item0["matrix_r0c0"] = 0;
  item0["matrix_r2c2"] = 0;
  item0["mean_chn_0"] = 0;
  item0["min_chn_0"] = 0.0F;
  item0["var_reci_chn_0"] = 0.0F;
  item0["max_src_image_size"] = 0;
  item0["orig_input_format"] = 0;
  item0["orig_input_data_type"] = 0;
  item0["orig_input_dim_num"] = 0;
  item0["aipp_inputs"] = nlohmann::json::array();
  item0["aipp_outputs"] = nlohmann::json::array();

  nlohmann::json item1;
  item1["index"] = 1;
  item1["aipp_type"] = 2;
  item1["aipp_data_index"] = 1;
  item1["aipp_mode"] = 2;
  item1["input_format"] = 2;
  item1["src_image_size_w"] = 512;
  item1["src_image_size_h"] = 512;
  item1["csc_switch"] = 0;
  item1["matrix_r0c0"] = 0;
  item1["matrix_r2c2"] = 0;
  item1["mean_chn_0"] = 0;
  item1["min_chn_0"] = 0.0F;
  item1["var_reci_chn_0"] = 0.0F;
  item1["max_src_image_size"] = 0;
  item1["orig_input_format"] = 0;
  item1["orig_input_data_type"] = 0;
  item1["orig_input_dim_num"] = 0;
  item1["aipp_inputs"] = nlohmann::json::array();
  item1["aipp_outputs"] = nlohmann::json::array();

  nlohmann::json root_json;
  root_json["aipp_infos"] = nlohmann::json::array({item0, item1});

  ge::JsonFile aipp_json(root_json);
  std::vector<Om2AippMeta> aipp_infos;
  bool has_aipp = false;

  auto status = ParseAippJson(aipp_json, aipp_infos, has_aipp);
  EXPECT_EQ(status, ge::SUCCESS);
  EXPECT_TRUE(has_aipp);
  EXPECT_EQ(aipp_infos.size(), 2U);
  EXPECT_EQ(aipp_infos[0].aipp_type, ge::DATA_WITH_STATIC_AIPP);
  EXPECT_EQ(aipp_infos[1].aipp_type, ge::DATA_WITH_DYNAMIC_AIPP);
}

}  // namespace om2
}  // namespace gert
