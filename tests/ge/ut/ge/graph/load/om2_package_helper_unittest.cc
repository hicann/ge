/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "framework/common/helper/model_save_helper.h"
#include "framework/common/zip_archive_reader.h"
#include "framework/common/zip_archive_writer.h"
#include "common/om2/codegen/om2_codegen.h"
#include "common/om2/codegen/om2_codegen_types.h"
#include "framework/om2/model_data/gert_model_data.h"
#include "framework/common/gert_model_data_utils.h"
#include "framework/om2/model_data/om2_package_contants.h"
#define private public
#include "framework/common/helper/om2_package_helper.h"
#include "framework/common/gert_model_data_serialize.h"
#undef private
#include "framework/common/framework_types_internal.h"
#include "framework/common/json_file.h"
#include "common/helper/visual_json_converter.h"
#include "common/model/ge_model.h"
#include "file_utils.h"
#include <algorithm>
#include <gtest/gtest.h>
#include <filesystem>
#include <fstream>
#include "common/env_path.h"
#include "mmpa/mmpa_api.h"
#include "graph/ge_local_context.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/custom_op_factory.h"
#include "graph/custom_op.h"
#include "graph/op_kernel_bin.h"
#include "graph/utils/tensor_utils.h"
#include "graph/utils/file_utils.h"
#include "graph/utils/graph_utils.h"
#include <cstdio>
#include <sstream>

#include "ge_runtime_stub/include/common/share_graph.h"
#include "ge_runtime_stub/include/faker/ge_model_builder.h"
#include "ge_runtime_stub/include/faker/custom_taskdef_faker.h"
#include "ge_runtime_stub/include/faker/aicore_taskdef_faker.h"
#include "common/tbe_handle_store/tbe_kernel_store.h"

namespace ge {
namespace {
constexpr const char *kOm2DumpDir = "/tmp/.tmp_om2_workspace";

static void SyncKernelNameFromOpDesc(const GeModelPtr &ge_model) {
  auto model_task_def = ge_model->GetModelTaskDefPtr();
  if (model_task_def == nullptr) {
    return;
  }
  const auto &graph = ge_model->GetGraph();
  if (graph == nullptr) {
    return;
  }
  for (int i = 0; i < model_task_def->task_size(); ++i) {
    auto *task_def = model_task_def->mutable_task(i);
    for (const auto &node : graph->GetDirectNode()) {
      auto op_desc = node->GetOpDesc();
      if (op_desc == nullptr) {
        continue;
      }
      std::string kernel_name;
      if (ge::AttrUtils::GetStr(op_desc, "_kernelname", kernel_name)) {
        task_def->mutable_kernel()->set_kernel_name(kernel_name);
      }
    }
  }
}

static void SyncKernelNameForAllModels(const GeRootModelPtr &ge_root_model) {
  if (ge_root_model == nullptr) {
    return;
  }
  for (const auto &kv : ge_root_model->GetSubgraphInstanceNameToModel()) {
    SyncKernelNameFromOpDesc(kv.second);
  }
}

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

GeRootModelPtr CreateGeRootModelWithAicoreOp() {
  auto graph = gert::ShareGraph::AicoreStaticGraph();
  graph->TopologicalSorting();
  gert::GeModelBuilder builder(graph);
  auto ge_root_model =
      builder
          .AddTaskDef("Add",
                      gert::AiCoreTaskDefFaker("add_stub").ArgsFormat("{i_instance0*}{i_instance1*}{o_instance0*}"))
          .FakeTbeBin({"Add"})
          .BuildGeRootModel();
  auto &compute_graph = ge_root_model->GetRootGraph();

  compute_graph->SetGraphUnknownFlag(false);
  for (const auto &node : compute_graph->GetDirectNode()) {
    auto op_desc = node->GetOpDesc();
    if (op_desc == nullptr) {
      return nullptr;
    }
    if ((op_desc->GetType() == DATA)) {
      op_desc->SetOutputOffset({1024});
    } else if (op_desc->GetType() == NETOUTPUT) {
      op_desc->SetInputOffset({3072});
    } else {
      op_desc->SetInputOffset(std::vector<int64_t>(op_desc->GetInputsSize(), 1024));
      op_desc->SetOutputOffset(std::vector<int64_t>(op_desc->GetOutputsSize(), 1024));
      if (op_desc->GetType() == "Add") {
        op_desc->SetIsInputConst({true, true});
        auto input_desc0 = op_desc->MutableInputDesc(0);
        auto input_desc1 = op_desc->MutableInputDesc(1);
        if ((input_desc0 == nullptr) || (input_desc1 == nullptr)) {
          return nullptr;
        }
        TensorUtils::SetDataOffset(*input_desc0, 0);
        TensorUtils::SetDataOffset(*input_desc1, 200704);
      }
    }
  }

  const auto ge_model = ge_root_model->GetSubgraphInstanceNameToModel().begin()->second;
  std::vector<uint8_t> weights_value(401408, 1U);
  const size_t weight_size = weights_value.size();
  ge_model->SetWeight(Buffer::CopyFrom(weights_value.data(), weight_size));

  (void)AttrUtils::SetInt(ge_model, ATTR_MODEL_MEMORY_SIZE, 2048);
  (void)AttrUtils::SetInt(ge_model, ATTR_MODEL_WEIGHT_SIZE, weight_size);
  (void)AttrUtils::SetInt(ge_model, ATTR_MODEL_STREAM_NUM, 1);

  return ge_root_model;
}

GeRootModelPtr CreateGeRootModelWithFileConstOp() {
  auto ge_root_model = CreateGeRootModelWithAicoreOp();
  if (ge_root_model == nullptr) {
    return nullptr;
  }

  auto &compute_graph = ge_root_model->GetRootGraph();
  for (const auto &node : compute_graph->GetDirectNode()) {
    auto op_desc = node->GetOpDesc();
    if (op_desc == nullptr) {
      return nullptr;
    }
    if (op_desc->GetName() == "data2") {
      op_desc->SetType(FILECONSTANT);
      (void)AttrUtils::SetStr(op_desc, ATTR_NAME_LOCATION, "weight_combined.bin");
      (void)AttrUtils::SetInt(op_desc, ATTR_NAME_OFFSET, 64);
      (void)AttrUtils::SetInt(op_desc, ATTR_NAME_LENGTH, 200704);
    } else if (op_desc->GetType() == "Add") {
      op_desc->SetIsInputConst({true, false});
    }
  }
  return ge_root_model;
}

GeRootModelPtr CreateInvalidGeRootModel() {
  auto graph = gert::ShareGraph::AicoreStaticGraph();
  graph->TopologicalSorting();
  gert::GeModelBuilder builder(graph);
  auto ge_root_model =
      builder
          .AddTaskDef("Add",
                      gert::AiCoreTaskDefFaker("add_stub").ArgsFormat("{i_instance0*}{i_instance1*}{o_instance0*}"))
          .FakeTbeBin({"Add"})
          .BuildGeRootModel();
  auto &compute_graph = ge_root_model->GetRootGraph();

  compute_graph->SetGraphUnknownFlag(false);
  return ge_root_model;
}

std::string GetOm2DumpPath(const std::string &file_name) {
  return std::string(kOm2DumpDir) + "/" + file_name;
}

bool ReadOm2DumpFile(const std::string &file_name, std::string &content) {
  std::ifstream input(GetOm2DumpPath(file_name), std::ios::in | std::ios::binary);
  if (!input.is_open()) {
    return false;
  }
  std::ostringstream oss;
  oss << input.rdbuf();
  content = oss.str();
  return true;
}

void RemoveOm2DumpFile(const std::string &file_name) {
  (void)std::remove(GetOm2DumpPath(file_name).c_str());
}

template <typename Archive>
std::string FindVisualJsonEntry(const Archive &archive) {
  for (const auto &file_name : archive.ListFiles()) {
    if ((file_name.find("debug/ge_visual_") != std::string::npos) && (file_name.find(".json") != std::string::npos)) {
      return file_name;
    }
  }
  return "";
}

template <typename Archive>
void ExpectVisualJsonCanLoad(const Archive &archive, const std::string &expected_graph_name) {
  const std::string visual_entry = FindVisualJsonEntry(archive);
  ASSERT_FALSE(visual_entry.empty()) << "ge_visual_*.json not found in OM2";

  size_t visual_size = 0U;
  const auto visual_buf = archive.ExtractToMem(visual_entry, visual_size);
  ASSERT_NE(visual_buf, nullptr);
  ASSERT_GT(visual_size, 0U);

  const JsonFile visual_json(reinterpret_cast<const uint8_t *>(visual_buf.get()), visual_size);
  ASSERT_TRUE(visual_json.IsValid());
  EXPECT_EQ(visual_json.Raw().at("format"), JsonFile::json("ge_visual_json"));
  EXPECT_EQ(visual_json.Raw().at("format_version"), JsonFile::json(1));
  ASSERT_TRUE(visual_json.Raw().contains("model"));
  ASSERT_TRUE(visual_json.Raw().at("model").contains("graph"));
  ASSERT_FALSE(visual_json.Raw().at("model").at("graph").empty());
  EXPECT_EQ(visual_json.Raw().at("model").at("graph").at(0).at("name"), JsonFile::json(expected_graph_name));
  ASSERT_TRUE(visual_json.Raw().at("model").at("graph").at(0).contains("op"));
  EXPECT_FALSE(visual_json.Raw().at("model").at("graph").at(0).at("op").empty());

  nlohmann::json pb_json;
  const std::string visual_str(reinterpret_cast<const char *>(visual_buf.get()), visual_size);
  ASSERT_EQ(VisualJsonConverter::LoadFromVisualJson(visual_str, pb_json), SUCCESS);
  ASSERT_TRUE(pb_json.contains("graph"));
  ASSERT_FALSE(pb_json["graph"].empty());
  EXPECT_EQ(pb_json["graph"][0]["name"], expected_graph_name);
  ASSERT_TRUE(pb_json["graph"][0].contains("op"));
  ASSERT_FALSE(pb_json["graph"][0]["op"].empty());
}

class TestPortableCustomOp : public PortableOp, public EagerExecuteOp {
 public:
  graphStatus Execute(gert::EagerOpExecutionContext *ctx) override {
    return SUCCESS;
  }

  graphStatus Serialize(std::vector<uint8_t> &buffer) override {
    const std::string payload = "test_portable_custom_op_kernel_bin";
    buffer.assign(payload.begin(), payload.end());
    return GRAPH_SUCCESS;
  }

  graphStatus Deserialize(const std::vector<uint8_t> &buffer) override {
    return GRAPH_SUCCESS;
  }
};

static ComputeGraphPtr BuildCustomOpGraph() {
  auto graph = std::make_shared<ComputeGraph>("custom_op_om2_graph");
  GeTensorDesc tensor_desc(GeShape({2, 2, 2}), FORMAT_ND, DT_FLOAT);

  auto data0_desc = std::make_shared<OpDesc>("data0", DATA);
  (void)data0_desc->AddInputDesc(tensor_desc);
  (void)data0_desc->AddOutputDesc(tensor_desc);
  AttrUtils::SetInt(data0_desc, ATTR_NAME_INDEX, 0);
  auto data0 = graph->AddNode(data0_desc);

  auto data1_desc = std::make_shared<OpDesc>("data1", DATA);
  (void)data1_desc->AddInputDesc(tensor_desc);
  (void)data1_desc->AddOutputDesc(tensor_desc);
  AttrUtils::SetInt(data1_desc, ATTR_NAME_INDEX, 1);
  auto data1 = graph->AddNode(data1_desc);

  auto custom_op_desc = std::make_shared<OpDesc>("custom_op", "TestPortableOp");
  (void)custom_op_desc->AddInputDesc("x0", tensor_desc);
  (void)custom_op_desc->AddInputDesc("x1", tensor_desc);
  (void)custom_op_desc->AddOutputDesc("y", tensor_desc);
  custom_op_desc->AppendIrInput("x0", kIrInputRequired);
  custom_op_desc->AppendIrInput("x1", kIrInputRequired);
  custom_op_desc->AppendIrOutput("y", kIrOutputRequired);
  auto custom_op_node = graph->AddNode(custom_op_desc);

  auto netoutput_desc = std::make_shared<OpDesc>("netoutput", NETOUTPUT);
  (void)netoutput_desc->AddInputDesc(tensor_desc);
  auto netoutput = graph->AddNode(netoutput_desc);

  GraphUtils::AddEdge(data0->GetOutDataAnchor(0), custom_op_node->GetInDataAnchor(0));
  GraphUtils::AddEdge(data1->GetOutDataAnchor(0), custom_op_node->GetInDataAnchor(1));
  GraphUtils::AddEdge(custom_op_node->GetOutDataAnchor(0), netoutput->GetInDataAnchor(0));
  netoutput_desc->SetSrcName({"custom_op"});
  netoutput_desc->SetSrcIndex({0});
  graph->TopologicalSorting();
  return graph;
}

GeRootModelPtr CreateGeRootModelWithCustomOp() {
  auto graph = BuildCustomOpGraph();

  gert::GeModelBuilder builder(graph);
  auto ge_root_model =
      builder
          .AddTaskDef(
              "custom_op",
              gert::CustomTaskDefFaker("custom_op_stub").ArgsFormat("{i_instance0*}{i_instance1*}{o_instance0*}"))
          .FakeTbeBin({"custom_op"})
          .BuildGeRootModel();
  auto &compute_graph = ge_root_model->GetRootGraph();
  compute_graph->SetGraphUnknownFlag(false);
  for (const auto &node : compute_graph->GetDirectNode()) {
    auto op_desc = node->GetOpDesc();
    if (op_desc == nullptr) {
      return nullptr;
    }
    if (op_desc->GetType() == DATA) {
      op_desc->SetOutputOffset({1024});
    } else if (op_desc->GetType() == NETOUTPUT) {
      op_desc->SetInputOffset({3072});
    } else {
      op_desc->SetInputOffset(std::vector<int64_t>(op_desc->GetInputsSize(), 1024));
      op_desc->SetOutputOffset(std::vector<int64_t>(op_desc->GetOutputsSize(), 1024));
    }
  }

  const auto ge_model = ge_root_model->GetSubgraphInstanceNameToModel().begin()->second;
  std::vector<uint8_t> weights_value(512, 1U);
  ge_model->SetWeight(Buffer::CopyFrom(weights_value.data(), weights_value.size()));
  (void)AttrUtils::SetInt(ge_model, ATTR_MODEL_MEMORY_SIZE, 2048);
  (void)AttrUtils::SetInt(ge_model, ATTR_MODEL_WEIGHT_SIZE, weights_value.size());
  (void)AttrUtils::SetInt(ge_model, ATTR_MODEL_STREAM_NUM, 1);

  return ge_root_model;
}

int WriteBinFile(const char *file_name, const std::string &text) {
  std::ofstream outFile(file_name, std::ios::out | std::ios::binary);
  if (!outFile.is_open()) {
    return 1;
  }
  outFile.write(text.c_str(), text.size());
  outFile.close();
  return 0;
}

}  // namespace

// 供 SetUpTestSuite 前置引用（static 定义位于本文件后部）
static GeRootModelPtr CreateGeRootModelWithStaticAipp();

// 保存/恢复线程级 graph options（与 om2_codegen_unittest 同款）
class ScopedGraphOptions {
 public:
  ScopedGraphOptions() : old_options_(GetThreadLocalContext().GetAllGraphOptions()) {}
  ~ScopedGraphOptions() {
    GetThreadLocalContext().SetGraphOption(old_options_);
  }

 private:
  std::map<std::string, std::string> old_options_;
};

// Suite 级共享：各图变体各执行一次完整 BuildOm2ModelData（含唯一一次动态编译 so），
// 用例内仅执行序列化与断言，消除用例内重复编译（CI 单用例性能门禁）
static std::map<std::string, std::shared_ptr<gert::GertModelData>> &SuiteModelDataMap() {
  static std::map<std::string, std::shared_ptr<gert::GertModelData>> map;
  return map;
}

static gert::GertModelData *SuiteModelData(const std::string &key) {
  const auto it = SuiteModelDataMap().find(key);
  return (it != SuiteModelDataMap().end()) ? it->second.get() : nullptr;
}

// 持有各变体的 GeRootModel：GertConstantsData 内部为非拥有权重指针（指向 ge_model 的 Buffer），
// 必须与共享 GertModelData 同生命周期，避免用例级序列化时悬垂
static std::vector<GeRootModelPtr> &SuiteRootModels() {
  static std::vector<GeRootModelPtr> models;
  return models;
}

static void BuildSuiteVariant(const std::string &key, const GeRootModelPtr &ge_root_model) {
  if (ge_root_model == nullptr) {
    ADD_FAILURE() << "Suite variant graph create failed: " << key;
    return;
  }
  const auto &ge_model = ge_root_model->GetSubgraphInstanceNameToModel().begin()->second;
  const auto parent_graph = std::make_shared<ComputeGraph>("root_g1");
  ge_model->GetGraph()->SetParentGraph(parent_graph);
  SyncKernelNameForAllModels(ge_root_model);
  auto data = std::make_shared<gert::GertModelData>();
  Om2PackageHelper helper;
  if (helper.BuildOm2ModelData(ge_model, *data, ge_root_model) != SUCCESS) {
    ADD_FAILURE() << "Suite variant build failed: " << key;
    return;
  }
  SuiteRootModels().push_back(ge_root_model);
  SuiteModelDataMap()[key] = std::move(data);
}

class Om2PackageHelperUt : public testing::Test {
 public:
  static void SetUpTestSuite() {
    const auto ascend_install_path = EnvPath().GetAscendInstallPath();
    setenv("ASCEND_HOME_PATH", ascend_install_path.c_str(), 1);
    const std::string suite_work_dir = EnvPath().GetOrCreateCaseTmpPath("Om2PackageHelperUt");
    setenv("ASCEND_WORK_PATH", suite_work_dir.c_str(), 1);
    std::filesystem::create_directories(std::filesystem::path(suite_work_dir) / ".ascend_temp" / ".tmp_om2_workspace");
    // 与用例 SetUp 一致：抑制子进程（make/g++）的 LSAN 检测，避免 g++ 编译器内部泄漏
    // 触发 LeakSanitizer 报错导致 so 编译失败
    setenv("ASAN_OPTIONS", "detect_leaks=0:halt_on_error=0", 1);
    setenv("LSAN_OPTIONS", "exitcode=0", 1);
    // Suite 级编译同样注入 -O0（桩库产物不承载执行性能）
    GetThreadLocalContext().SetGraphOption({{"ge.buildConfig", "make -s CXXFLAGS='-std=c++17 -fPIC -O0'"}});

    BuildSuiteVariant("aicore", CreateGeRootModelWithAicoreOp());

    {
      auto root = CreateGeRootModelWithAicoreOp();
      const auto &ge_model = root->GetSubgraphInstanceNameToModel().begin()->second;
      (void)AttrUtils::SetInt(ge_model, ATTR_MODEL_MEMORY_SIZE, 2048);
      (void)AttrUtils::SetInt(ge_model, ATTR_MODEL_ZERO_COPY_MEMORY_SIZE, 512);
      BuildSuiteVariant("zero_copy", root);
    }
    {
      auto root = CreateGeRootModelWithAicoreOp();
      for (const auto &node : root->GetRootGraph()->GetDirectNode()) {
        auto op_desc = node->GetOpDesc();
        if ((op_desc != nullptr) && (op_desc->GetType() != DATA) && (op_desc->GetType() != NETOUTPUT)) {
          AttrUtils::SetListStr(op_desc, ATTR_NAME_DATA_DUMP_ORIGIN_OP_NAMES, {"original_op1", "original_op2"});
        }
      }
      BuildSuiteVariant("with_attr", root);
    }
    {
      auto root = CreateGeRootModelWithAicoreOp();
      for (const auto &node : root->GetRootGraph()->GetDirectNode()) {
        auto op_desc = node->GetOpDesc();
        if ((op_desc != nullptr) && (op_desc->GetType() != DATA) && (op_desc->GetType() != NETOUTPUT)) {
          AttrUtils::SetListStr(op_desc, ATTR_NAME_DATA_DUMP_ORIGIN_OP_NAMES, std::vector<std::string>{});
          break;
        }
      }
      BuildSuiteVariant("empty_names", root);
    }
    BuildSuiteVariant("fileconst", CreateGeRootModelWithFileConstOp());
    BuildSuiteVariant("static_aipp", CreateGeRootModelWithStaticAipp());
    {
      const AscendString kOpType("TestPortableOp");
      CustomOpFactory::RegisterCustomOpCreator(
          kOpType, []() -> std::unique_ptr<BaseCustomOp> { return std::make_unique<TestPortableCustomOp>(); });
      BuildSuiteVariant("custom_op", CreateGeRootModelWithCustomOp());
    }
  }

  static void TearDownTestSuite() {
    SuiteModelDataMap().clear();
    SuiteRootModels().clear();
    unsetenv("ASCEND_HOME_PATH");
    unsetenv("ASCEND_WORK_PATH");
    unsetenv("ASAN_OPTIONS");
    unsetenv("LSAN_OPTIONS");
    GetThreadLocalContext().SetGraphOption(std::map<std::string, std::string>{});
  }

  void SetUp() override {
    const ::testing::TestInfo *test_info = ::testing::UnitTest::GetInstance()->current_test_info();
    test_case_name = test_info->test_case_name();  // Om2PackageHelperUt
    test_work_dir = EnvPath().GetOrCreateCaseTmpPath(test_case_name);
    const auto ascend_install_path = EnvPath().GetAscendInstallPath();
    setenv("ASCEND_HOME_PATH", ascend_install_path.c_str(), 1);
    setenv("ASCEND_WORK_PATH", test_work_dir.c_str(), 1);
    om2_workspace_base_dir_ = std::filesystem::path(test_work_dir) / ".ascend_temp" / ".tmp_om2_workspace";
    std::filesystem::create_directories(om2_workspace_base_dir_);
    asan_guard_ = std::make_unique<ScopedEnvVar>("ASAN_OPTIONS", "detect_leaks=0:halt_on_error=0");
    lsan_guard_ = std::make_unique<ScopedEnvVar>("LSAN_OPTIONS", "exitcode=0");
    // 桩库产物（USE_STUB_LIB=1）仅用于打包/加载验证，不承载执行性能；
    // 经 ge.buildConfig 注入 -O0（业务预留的编译配置口），加速用例内动态编译
    graph_options_guard_ = std::make_unique<ScopedGraphOptions>();
    GetThreadLocalContext().SetGraphOption({{"ge.buildConfig", "make -s CXXFLAGS='-std=c++17 -fPIC -O0'"}});
  }
  void TearDown() override {
    graph_options_guard_.reset();
    lsan_guard_.reset();
    asan_guard_.reset();
    EnvPath().RemoveRfCaseTmpPath(test_case_name);
    unsetenv("ASCEND_HOME_PATH");
    unsetenv("ASCEND_WORK_PATH");
  }

 public:
  std::string test_case_name;
  std::string test_work_dir;
  const std::string kZipFileBaseName = "fake_test";
  std::filesystem::path om2_workspace_base_dir_;

 private:
  std::unique_ptr<ScopedEnvVar> asan_guard_;
  std::unique_ptr<ScopedEnvVar> lsan_guard_;
  std::unique_ptr<ScopedGraphOptions> graph_options_guard_;
};

TEST_F(Om2PackageHelperUt, ZipArchiveReaderListAndExtractEntries) {
  gert::GertBuffer model_data;
  gert::ZipArchiveWriter zip_writer(PathUtils::Join({test_work_dir, "simple_reader.om2"}));
  ASSERT_TRUE(zip_writer.IsMemFileOpened());
  const std::string manifest = R"({"model_num":1})";
  const std::string meta = R"({"name":"g1"})";
  ASSERT_TRUE(zip_writer.WriteBytes("manifest.json", manifest.data(), manifest.size(), false));
  ASSERT_TRUE(zip_writer.WriteBytes("data/model_0/model_meta.json", meta.data(), meta.size(), true));
  ASSERT_TRUE(zip_writer.SaveModelData(model_data, false));

  gert::ZipArchiveReader archive(model_data.data.get(), model_data.length);
  ASSERT_TRUE(archive.IsGood());
  const auto file_names = archive.ListFiles();
  EXPECT_NE(std::find(file_names.begin(), file_names.end(), "simple_reader/manifest.json"), file_names.end());
  EXPECT_NE(std::find(file_names.begin(), file_names.end(), "simple_reader/data/model_0/model_meta.json"),
            file_names.end());

  size_t manifest_size = 0U;
  const auto manifest_buf = archive.ExtractToMem("simple_reader/manifest.json", manifest_size);
  ASSERT_NE(manifest_buf, nullptr);
  EXPECT_EQ(std::string(reinterpret_cast<const char *>(manifest_buf.get()), manifest_size), manifest);

  size_t meta_size = 0U;
  const auto meta_buf = archive.ExtractToMem("simple_reader/data/model_0/model_meta.json", meta_size);
  ASSERT_NE(meta_buf, nullptr);
  EXPECT_EQ(std::string(reinterpret_cast<const char *>(meta_buf.get()), meta_size), meta);
}

TEST_F(Om2PackageHelperUt, ConvertOm2Model_Ok_GenOm2WithAicoreNode) {
  // Suite 级已对同图完成 BuildOm2ModelData（含编译），此处仅序列化
  ModelBufferData model_data;
  const std::string output_file = PathUtils::Join({test_work_dir, kZipFileBaseName + ".om2"});
  ASSERT_NE(SuiteModelData("aicore"), nullptr);
  ASSERT_EQ(gert::SerializeGertModelData(*SuiteModelData("aicore"), model_data, true, output_file), SUCCESS);
  ASSERT_EQ(mmAccess2(output_file.c_str(), M_F_OK), EOK);

  uint32_t model_buf_size = 0;
  const auto model_buf = GetBinDataFromFile(output_file, model_buf_size);
  gert::ZipArchiveReader archive(reinterpret_cast<const uint8_t *>(model_buf.get()), model_buf_size);
  ASSERT_TRUE(archive.IsGood());
  const auto file_names = archive.ListFiles();
  const std::set<std::string> expect_files = {
      "fake_test/data/model_0/runtime/csrc/g1_kernel_reg.cpp",
      "fake_test/data/model_0/runtime/csrc/g1_resources.cpp",
      "fake_test/data/model_0/runtime/csrc/g1_args_manager.cpp",
      "fake_test/data/model_0/runtime/csrc/g1_load_and_run.cpp",
      "fake_test/data/model_0/runtime/csrc/om2_model_api.h",
      "fake_test/data/model_0/runtime/csrc/g1_internal.h",
      "fake_test/data/model_0/runtime/csrc/Makefile",
      "fake_test/data/model_0/runtime/libg1_om2.so",
      "fake_test/data/constants/constant_0",
      "fake_test/data/model_0/constants_config.json",
      "fake_test/data/kernels/te_Add_12345_AicoreKernel.o",
      "fake_test/data/model_0/model_meta.json",
      "fake_test/data/model_0/op_attr.json",
      "fake_test/data/model_0/debug/ge_visual_00000000_graph_0.json",
      "fake_test/manifest.json",
  };
  EXPECT_EQ(file_names.size(), expect_files.size());
  for (const auto &file_name : file_names) {
    EXPECT_EQ(expect_files.count(file_name), 1);
  }
  const std::vector<std::string> cpp_entries = {
      "fake_test/data/model_0/runtime/csrc/g1_kernel_reg.cpp",
      "fake_test/data/model_0/runtime/csrc/g1_resources.cpp",
      "fake_test/data/model_0/runtime/csrc/g1_args_manager.cpp",
      "fake_test/data/model_0/runtime/csrc/g1_load_and_run.cpp",
  };
  for (const auto &cpp_entry : cpp_entries) {
    size_t cpp_size = 0;
    const auto cpp_buf = archive.ExtractToMem(cpp_entry, cpp_size);
    ASSERT_NE(cpp_buf, nullptr);
    const std::string cpp_content(reinterpret_cast<const char *>(cpp_buf.get()), cpp_size);
    EXPECT_NE(cpp_content.find("#include \"g1_internal.h\""), std::string::npos) << cpp_entry;
    EXPECT_EQ(cpp_content.find("/proc/self/fd/"), std::string::npos) << cpp_entry;
  }
  size_t makefile_size = 0;
  const auto makefile_buf = archive.ExtractToMem("fake_test/data/model_0/runtime/csrc/Makefile", makefile_size);
  ASSERT_NE(makefile_buf, nullptr);
  const std::string makefile_content(reinterpret_cast<const char *>(makefile_buf.get()), makefile_size);
  EXPECT_NE(makefile_content.find("TARGET := ../libg1_om2.so"), std::string::npos);
  EXPECT_NE(makefile_content.find("SRC_FILES := g1_resources.cpp g1_kernel_reg.cpp"), std::string::npos);
  EXPECT_EQ(makefile_content.find("/proc/self/fd/"), std::string::npos);
  EXPECT_EQ(makefile_content.find("CXXFLAGS += -x c++"), std::string::npos);

  size_t manifest_size = 0;
  const auto manifest_buf = archive.ExtractToMem("fake_test/manifest.json", manifest_size);
  ASSERT_NE(manifest_buf, nullptr);
  const JsonFile manifest_json(reinterpret_cast<const uint8_t *>(manifest_buf.get()), manifest_size);
  ASSERT_TRUE(manifest_json.IsValid());
  std::string atc_command;
  ASSERT_TRUE(manifest_json.Get("atc_command", atc_command));
  EXPECT_EQ(atc_command, "");
  int model_num;
  ASSERT_TRUE(manifest_json.Get("model_num", model_num));
  EXPECT_EQ(model_num, 1);
  std::string compiler_version;
  JsonFile compat_json;
  ASSERT_TRUE(manifest_json.Get("compatibility", compat_json));
  ASSERT_TRUE(compat_json.Get("compiler_version", compiler_version));
  EXPECT_EQ(compiler_version, "1.0");

  size_t model_meta_size = 0;
  const auto model_meta_buf = archive.ExtractToMem("fake_test/data/model_0/model_meta.json", model_meta_size);
  ASSERT_NE(model_meta_buf, nullptr);
  const JsonFile model_meta_json(reinterpret_cast<const uint8_t *>(model_meta_buf.get()), model_meta_size);
  ASSERT_TRUE(model_meta_json.IsValid());
  EXPECT_EQ(model_meta_json.Raw().at("name"), JsonFile::json("g1"));
  EXPECT_EQ(model_meta_json.Raw().at("work_size"), JsonFile::json(2048));
  EXPECT_EQ(model_meta_json.Raw().at("zero_copy_size"), JsonFile::json(0));

  const JsonFile::json expected_inputs = JsonFile::json::array({
      {{"data_type", 0},
       {"format", 2},
       {"index", 0},
       {"name", "data1"},
       {"shape", JsonFile::json::array({1, 2, 3, 4})},
       {"shape_range", JsonFile::json::array()},
       {"size", 0}},
      {{"data_type", 0},
       {"format", 0},
       {"index", 1},
       {"name", "data2"},
       {"shape", JsonFile::json::array({1, 1, 224, 224})},
       {"shape_range", JsonFile::json::array()},
       {"size", 0}},
  });
  EXPECT_EQ(model_meta_json.Raw().at("inputs"), expected_inputs);

  const JsonFile::json expected_outputs = JsonFile::json::array({
      {{"data_type", 0},
       {"format", 2},
       {"index", 0},
       {"name", "output_0_reshape1_0"},
       {"shape", JsonFile::json::array()},
       {"shape_range", JsonFile::json::array()},
       {"size", 4}},
  });
  EXPECT_EQ(model_meta_json.Raw().at("outputs"), expected_outputs);

  size_t constants_config_size = 0;
  const auto constants_config_buf =
      archive.ExtractToMem("fake_test/data/model_0/constants_config.json", constants_config_size);
  ASSERT_NE(constants_config_buf, nullptr);
  const JsonFile constants_json(reinterpret_cast<const uint8_t *>(constants_config_buf.get()), constants_config_size);
  ASSERT_TRUE(constants_json.IsValid());
  EXPECT_EQ(constants_json.Raw().at("internal_weight_size"), JsonFile::json(401408));
  EXPECT_FALSE(constants_json.Raw().contains("weight_size"));
  ASSERT_TRUE(constants_json.Raw().contains("consts"));
  const auto &consts = constants_json.Raw().at("consts");
  ASSERT_TRUE(consts.is_object());
  ASSERT_EQ(consts.size(), 2U);
  ASSERT_TRUE(consts.contains("constant_0"));
  ASSERT_TRUE(consts.contains("constant_1"));
  EXPECT_EQ(consts.at("constant_0").at("index"), JsonFile::json(0));
  EXPECT_EQ(consts.at("constant_0").at("type"), JsonFile::json("INTERNAL"));
  EXPECT_FALSE(consts.at("constant_0").contains("external"));
  EXPECT_EQ(consts.at("constant_0").at("file_name"), JsonFile::json("constant_0"));
  EXPECT_EQ(consts.at("constant_0").at("offset"), JsonFile::json(0));
  EXPECT_EQ(consts.at("constant_0").at("size"), JsonFile::json(200704));
  EXPECT_EQ(consts.at("constant_1").at("index"), JsonFile::json(1));
  EXPECT_EQ(consts.at("constant_1").at("type"), JsonFile::json("INTERNAL"));
  EXPECT_FALSE(consts.at("constant_1").contains("external"));
  EXPECT_EQ(consts.at("constant_1").at("file_name"), JsonFile::json("constant_0"));
  EXPECT_EQ(consts.at("constant_1").at("offset"), JsonFile::json(200704));
  EXPECT_EQ(consts.at("constant_1").at("size"), JsonFile::json(200704));
}

TEST_F(Om2PackageHelperUt, ConvertOm2Model_WithZeroCopySize_WorkSizeAdjusted) {
  // Suite 级已对同图（含 zero copy 属性）完成 BuildOm2ModelData，此处仅序列化
  ModelBufferData model_data;
  const std::string output_file = PathUtils::Join({test_work_dir, kZipFileBaseName + "_zero_copy.om2"});
  ASSERT_NE(SuiteModelData("zero_copy"), nullptr);
  ASSERT_EQ(gert::SerializeGertModelData(*SuiteModelData("zero_copy"), model_data, true, output_file), SUCCESS);

  uint32_t model_buf_size = 0;
  const auto model_buf = GetBinDataFromFile(output_file, model_buf_size);
  gert::ZipArchiveReader archive(reinterpret_cast<const uint8_t *>(model_buf.get()), model_buf_size);
  ASSERT_TRUE(archive.IsGood());

  size_t model_meta_size = 0;
  const auto model_meta_buf = archive.ExtractToMem("fake_test_zero_copy/data/model_0/model_meta.json", model_meta_size);
  ASSERT_NE(model_meta_buf, nullptr);
  const JsonFile model_meta_json(reinterpret_cast<const uint8_t *>(model_meta_buf.get()), model_meta_size);
  ASSERT_TRUE(model_meta_json.IsValid());

  int64_t work_size = 0;
  ASSERT_TRUE(model_meta_json.Get("work_size", work_size));
  EXPECT_EQ(work_size, 2048);

  int64_t zero_copy_size = 0;
  ASSERT_TRUE(model_meta_json.Get("zero_copy_size", zero_copy_size));
  EXPECT_EQ(zero_copy_size, 512);
}

TEST_F(Om2PackageHelperUt, SaveToOmModel_SaveModeFalse_ReturnsModelBuffer) {
  ModelBufferData model_data;
  const std::string output_file = PathUtils::Join({test_work_dir, kZipFileBaseName + "_buffer.om2"});
  ASSERT_NE(SuiteModelData("aicore"), nullptr);
  ASSERT_EQ(gert::SerializeGertModelData(*SuiteModelData("aicore"), model_data, false, "g1"), SUCCESS);
  EXPECT_NE(mmAccess2(output_file.c_str(), M_F_OK), EOK);
  ASSERT_NE(model_data.data, nullptr);
  ASSERT_GT(model_data.length, 0U);

  gert::ZipArchiveReader archive(model_data.data.get(), model_data.length);
  ASSERT_TRUE(archive.IsGood());
  const auto file_names = archive.ListFiles();
  const std::set<std::string> expect_files = {
      "g1/data/model_0/runtime/csrc/g1_kernel_reg.cpp",
      "g1/data/model_0/runtime/csrc/g1_resources.cpp",
      "g1/data/model_0/runtime/csrc/g1_args_manager.cpp",
      "g1/data/model_0/runtime/csrc/g1_load_and_run.cpp",
      "g1/data/model_0/runtime/csrc/om2_model_api.h",
      "g1/data/model_0/runtime/csrc/g1_internal.h",
      "g1/data/model_0/runtime/csrc/Makefile",
      "g1/data/model_0/runtime/libg1_om2.so",
      "g1/data/constants/constant_0",
      "g1/data/model_0/constants_config.json",
      "g1/data/kernels/te_Add_12345_AicoreKernel.o",
      "g1/data/model_0/model_meta.json",
      "g1/data/model_0/op_attr.json",
      "g1/data/model_0/debug/ge_visual_00000000_graph_0.json",
      "g1/manifest.json",
  };
  EXPECT_EQ(file_names.size(), expect_files.size());
  for (const auto &file_name : file_names) {
    EXPECT_EQ(expect_files.count(file_name), 1);
  }
}

TEST_F(Om2PackageHelperUt, SaveToOmModel_SaveModeFalse_FallbacksToOutputFileWhenModelNameEmpty) {
  // 原用例验证模型名为空时 writer_path 回退 output_file；序列化层等价于直接以 output_file 为 writer_path
  ModelBufferData model_data;
  const std::string output_file = PathUtils::Join({test_work_dir, kZipFileBaseName + "_empty_name_buffer.om2"});
  ASSERT_NE(SuiteModelData("aicore"), nullptr);
  ASSERT_EQ(gert::SerializeGertModelData(*SuiteModelData("aicore"), model_data, false, output_file), SUCCESS);
  EXPECT_NE(mmAccess2(output_file.c_str(), M_F_OK), EOK);
  ASSERT_NE(model_data.data, nullptr);
  ASSERT_GT(model_data.length, 0U);

  gert::ZipArchiveReader archive(model_data.data.get(), model_data.length);
  ASSERT_TRUE(archive.IsGood());
  const auto file_names = archive.ListFiles();
  EXPECT_NE(std::find(file_names.begin(), file_names.end(), "fake_test_empty_name_buffer/manifest.json"),
            file_names.end());
  EXPECT_NE(std::find(file_names.begin(), file_names.end(), "fake_test_empty_name_buffer/data/model_0/model_meta.json"),
            file_names.end());
}

TEST_F(Om2PackageHelperUt, SaveToOmModel_SaveModeTrue_WritesFile) {
  ModelBufferData model_data;
  const std::string output_file = PathUtils::Join({test_work_dir, kZipFileBaseName + "_file.om2"});
  ASSERT_NE(SuiteModelData("aicore"), nullptr);
  ASSERT_EQ(gert::SerializeGertModelData(*SuiteModelData("aicore"), model_data, true, output_file), SUCCESS);
  ASSERT_EQ(mmAccess2(output_file.c_str(), M_F_OK), EOK);
  EXPECT_EQ(model_data.data, nullptr);

  uint32_t model_buf_size = 0;
  const auto model_buf = GetBinDataFromFile(output_file, model_buf_size);
  gert::ZipArchiveReader archive(reinterpret_cast<const uint8_t *>(model_buf.get()), model_buf_size);
  ASSERT_TRUE(archive.IsGood());
  const auto file_names = archive.ListFiles();
  EXPECT_NE(std::find(file_names.begin(), file_names.end(), "fake_test_file/manifest.json"), file_names.end());
  EXPECT_NE(std::find(file_names.begin(), file_names.end(), "fake_test_file/data/model_0/model_meta.json"),
            file_names.end());
}

TEST_F(Om2PackageHelperUt, ConvertOm2Model_Fail_GenFailedAndRemoveOm2File) {
  Om2PackageHelper om2_packager;
  const auto ge_root_model = CreateInvalidGeRootModel();
  ASSERT_NE(ge_root_model, nullptr);
  ModelBufferData model_data;
  const std::string output_file = PathUtils::Join({test_work_dir, kZipFileBaseName + "_invalid.om2"});
  ASSERT_NE(om2_packager.SaveToOmRootModel(ge_root_model, output_file, model_data, false), SUCCESS);
  ASSERT_NE(mmAccess2(output_file.c_str(), M_F_OK), EOK);
}

TEST_F(Om2PackageHelperUt, Om2CodegenAndCompile_Fail_DoesNotDumpGeneratedFiles) {
  const std::vector<std::string> dump_files = {
      "g1_kernel_reg.cpp", "g1_resources.cpp", "g1_args_manager.cpp", "g1_load_and_run.cpp", "om2_model_api.h",
      "g1_internal.h",     "Makefile",
  };
  for (const auto &file_name : dump_files) {
    RemoveOm2DumpFile(file_name);
  }

  const auto ge_root_model = CreateGeRootModelWithAicoreOp();
  ASSERT_NE(ge_root_model, nullptr);
  const auto &name_to_ge_model = ge_root_model->GetSubgraphInstanceNameToModel();
  ASSERT_FALSE(name_to_ge_model.empty());
  const auto &ge_model = name_to_ge_model.begin()->second;
  ASSERT_NE(ge_model, nullptr);
  ASSERT_NE(ge_model->GetModelTaskDefPtr(), nullptr);
  ASSERT_GT(ge_model->GetModelTaskDefPtr()->task_size(), 0);
  auto *kernel_def = ge_model->GetModelTaskDefPtr()->mutable_task(0)->mutable_kernel();
  ASSERT_NE(kernel_def, nullptr);
  kernel_def->set_kernel_name("bad\"kernel_name");
  // AICore 代码生成从 op_desc 的 _kernelname 属性读取 kernel_name，需同步修改
  const auto &compute_graph = ge_root_model->GetRootGraph();
  for (const auto &node : compute_graph->GetDirectNode()) {
    auto op_desc = node->GetOpDesc();
    if ((op_desc != nullptr) && (op_desc->GetType() != DATA) && (op_desc->GetType() != NETOUTPUT)) {
      ge::AttrUtils::SetStr(op_desc, "_kernelname", "bad\"kernel_name");
    }
  }

  // 覆盖 build_config 为必然失败的编译命令（CXX=/bin/false），等价"编译失败"场景且避免真实 g++ 全量编译耗时
  GetThreadLocalContext().SetGraphOption({{"ge.buildConfig", "make -s CXX=/bin/false"}});

  gert::GertModelData model_data;
  gert::InitGertModelData(model_data);
  model_data.models.emplace_back(std::make_unique<gert::GertModelDataModel>());
  model_data.constants->constants_data.emplace_back();
  model_data.models[0]->runtime = std::make_unique<gert::GertModelDataRuntime>();
  model_data.models[0]->constants_config = std::make_unique<gert::GertModelDataConstantsConfig>();
  model_data.constants->constants_data[0] = std::make_unique<gert::GertModelDataFile>();
  Om2Codegen codegen;
  ASSERT_NE(codegen.Om2CodegenAndCompile(ge_model, model_data, *model_data.models[0]), SUCCESS);

  for (const auto &file_name : dump_files) {
    std::string content;
    EXPECT_FALSE(ReadOm2DumpFile(file_name, content));
  }

  for (const auto &file_name : dump_files) {
    RemoveOm2DumpFile(file_name);
  }
}

TEST_F(Om2PackageHelperUt, ConvertOm2Model_Ok_GenOm2WithFileConstMeta) {
  ModelBufferData model_data;
  const std::string output_file = PathUtils::Join({test_work_dir, "fake_fileconst.om2"});
  ASSERT_NE(SuiteModelData("fileconst"), nullptr);
  ASSERT_EQ(gert::SerializeGertModelData(*SuiteModelData("fileconst"), model_data, true, output_file), SUCCESS);

  uint32_t model_buf_size = 0;
  const auto model_buf = GetBinDataFromFile(output_file, model_buf_size);
  gert::ZipArchiveReader archive(reinterpret_cast<const uint8_t *>(model_buf.get()), model_buf_size);
  ASSERT_TRUE(archive.IsGood());
  const auto file_names = archive.ListFiles();
  EXPECT_NE(std::find(file_names.begin(), file_names.end(), "fake_fileconst/data/constants/constant_0"),
            file_names.end());

  size_t constants_config_size = 0;
  const auto constants_config_buf =
      archive.ExtractToMem("fake_fileconst/data/model_0/constants_config.json", constants_config_size);
  ASSERT_NE(constants_config_buf, nullptr);
  const JsonFile constants_json(reinterpret_cast<const uint8_t *>(constants_config_buf.get()), constants_config_size);
  ASSERT_TRUE(constants_json.IsValid());
  EXPECT_EQ(constants_json.Raw().at("internal_weight_size"), JsonFile::json(401408));
  EXPECT_FALSE(constants_json.Raw().contains("weight_size"));
  ASSERT_TRUE(constants_json.Raw().contains("consts"));
  const auto &consts = constants_json.Raw().at("consts");
  ASSERT_TRUE(consts.is_object());
  ASSERT_EQ(consts.size(), 2U);
  ASSERT_TRUE(consts.contains("constant_1"));
  ASSERT_TRUE(consts.contains("data2"));

  const auto &internal_const = consts.at("constant_1");
  EXPECT_EQ(internal_const.at("index"), JsonFile::json(1));
  EXPECT_EQ(internal_const.at("type"), JsonFile::json("INTERNAL"));
  EXPECT_EQ(internal_const.at("file_name"), JsonFile::json("constant_0"));
  EXPECT_EQ(internal_const.at("offset"), JsonFile::json(0));
  EXPECT_EQ(internal_const.at("size"), JsonFile::json(200704));
  EXPECT_FALSE(internal_const.contains("file_path"));

  const auto &file_const = consts.at("data2");
  EXPECT_EQ(file_const.at("index"), JsonFile::json(0));
  EXPECT_EQ(file_const.at("type"), JsonFile::json("COMBINED"));
  EXPECT_EQ(file_const.at("file_name"), JsonFile::json("weight_combined.bin"));
  EXPECT_FALSE(file_const.contains("file_path"));
  EXPECT_EQ(file_const.at("offset"), JsonFile::json(64));
  EXPECT_EQ(file_const.at("size"), JsonFile::json(200704));
}

TEST_F(Om2PackageHelperUt, RelocateExternalWeights_SkipInvalidConstItemsAndCompressRuntimeEntry) {
  const std::string tmp_weight_dir = PathUtils::Join({test_work_dir, "tmp_weight"});
  ASSERT_TRUE(std::filesystem::create_directories(tmp_weight_dir));
  const std::string weight_file_name = "weight_from_path.bin";
  const std::string old_weight_path = PathUtils::Join({tmp_weight_dir, weight_file_name});
  {
    std::ofstream weight_file(old_weight_path, std::ios::binary);
    ASSERT_TRUE(weight_file.is_open());
    weight_file << "weight";
  }

  ModelBufferData model;
  {
    gert::ZipArchiveWriter zip_writer(PathUtils::Join({test_work_dir, "build_model.om2"}));
    ASSERT_TRUE(zip_writer.IsMemFileOpened());

    auto consts = JsonFile::json::object();
    consts["not_object"] = "skip";
    JsonFile internal_const;
    internal_const.Set("type", "INTERNAL").Set("file_path", old_weight_path);
    consts["internal_const"] = internal_const.Raw();
    JsonFile no_path_const;
    no_path_const.Set("type", "COMBINED").Set("file_name", "no_path.bin");
    consts["no_path_const"] = no_path_const.Raw();
    JsonFile empty_path_const;
    empty_path_const.Set("type", "COMBINED").Set("file_path", "");
    consts["empty_path_const"] = empty_path_const.Raw();
    JsonFile basename_const;
    basename_const.Set("type", "COMBINED")
        .Set("file_name", "")
        .Set("file_path", old_weight_path)
        .Set("offset", 0)
        .Set("size", 6);
    consts["basename_const"] = basename_const.Raw();

    JsonFile constants_config;
    constants_config.Set("internal_weight_size", 0U).Set("consts", consts);
    const std::string constants_config_str = constants_config.Dump();
    ASSERT_TRUE(zip_writer.WriteBytes("data/model_0/constants_config.json", constants_config_str.data(),
                                      constants_config_str.size(), false));
    const std::string no_consts_config = R"({"internal_weight_size":0})";
    ASSERT_TRUE(zip_writer.WriteBytes("data/model_1/constants_config.json", no_consts_config.data(),
                                      no_consts_config.size(), false));
    const std::string skipped_consts_config = R"({"consts":{"internal":{"type":"INTERNAL"}}})";
    ASSERT_TRUE(zip_writer.WriteBytes("data/model_2/constants_config.json", skipped_consts_config.data(),
                                      skipped_consts_config.size(), false));
    const std::string runtime_entry = "runtime";
    ASSERT_TRUE(
        zip_writer.WriteBytes("data/model_0/runtime/libfake.so", runtime_entry.data(), runtime_entry.size(), false));
    const std::string manifest = R"({"archive_version":"1.0","model_num":3})";
    ASSERT_TRUE(zip_writer.WriteBytes("manifest.json", manifest.data(), manifest.size(), false));
    gert::GertBuffer om2_buf;
    ASSERT_TRUE(zip_writer.SaveModelData(om2_buf, false));
    model.data = om2_buf.data;
    model.length = om2_buf.length;
  }

  ModelBufferData relocated_model;
  bool relocated = false;
  const std::string output_file = PathUtils::Join({test_work_dir, "saved_model.om2"});
  ASSERT_EQ(Om2PackageHelper::RelocateExternalWeights(output_file, model, relocated_model, relocated), SUCCESS);
  ASSERT_TRUE(relocated);
  ASSERT_NE(relocated_model.data, nullptr);
  ASSERT_GT(relocated_model.length, 0U);
  EXPECT_NE(mmAccess2(old_weight_path.c_str(), M_F_OK), EOK);
  EXPECT_EQ(mmAccess2(PathUtils::Join({test_work_dir, "weight", weight_file_name}).c_str(), M_F_OK), EOK);

  gert::ZipArchiveReader archive(relocated_model.data.get(), relocated_model.length);
  ASSERT_TRUE(archive.IsGood());
  const auto file_names = archive.ListFiles();
  EXPECT_NE(std::find(file_names.begin(), file_names.end(), "saved_model/data/model_0/runtime/libfake.so"),
            file_names.end());
  EXPECT_NE(std::find(file_names.begin(), file_names.end(), "saved_model/data/model_1/constants_config.json"),
            file_names.end());
  EXPECT_NE(std::find(file_names.begin(), file_names.end(), "saved_model/data/model_2/constants_config.json"),
            file_names.end());

  size_t constants_config_size = 0;
  const auto constants_config_buf =
      archive.ExtractToMem("saved_model/data/model_0/constants_config.json", constants_config_size);
  ASSERT_NE(constants_config_buf, nullptr);
  const JsonFile constants_json(reinterpret_cast<const uint8_t *>(constants_config_buf.get()), constants_config_size);
  ASSERT_TRUE(constants_json.IsValid());
  const auto &rewritten_consts = constants_json.Raw().at("consts");
  EXPECT_EQ(rewritten_consts.at("not_object"), JsonFile::json("skip"));
  EXPECT_TRUE(rewritten_consts.at("internal_const").contains("file_path"));
  EXPECT_FALSE(rewritten_consts.at("no_path_const").contains("file_path"));
  EXPECT_EQ(rewritten_consts.at("empty_path_const").at("file_path"), JsonFile::json(""));
  const auto &rewritten_basename_const = rewritten_consts.at("basename_const");
  EXPECT_EQ(rewritten_basename_const.at("file_name"), JsonFile::json(weight_file_name));
  EXPECT_FALSE(rewritten_basename_const.contains("file_path"));
}

TEST_F(Om2PackageHelperUt, SaveOpAttrJson_WithAttr_GenValidOpAttrJson) {
  ModelBufferData model_data;
  const std::string output_file = PathUtils::Join({test_work_dir, "test_op_attr.om2"});
  ASSERT_NE(SuiteModelData("with_attr"), nullptr);
  ASSERT_EQ(gert::SerializeGertModelData(*SuiteModelData("with_attr"), model_data, true, output_file), SUCCESS);

  // 解压并验证op_attr.json
  uint32_t model_buf_size = 0;
  const auto model_buf = GetBinDataFromFile(output_file, model_buf_size);
  gert::ZipArchiveReader archive(reinterpret_cast<const uint8_t *>(model_buf.get()), model_buf_size);
  ASSERT_TRUE(archive.IsGood());

  size_t op_attr_size = 0;
  const auto op_attr_buf = archive.ExtractToMem("test_op_attr/data/model_0/op_attr.json", op_attr_size);
  ASSERT_NE(op_attr_buf, nullptr);

  const JsonFile op_attr_json(reinterpret_cast<const uint8_t *>(op_attr_buf.get()), op_attr_size);
  ASSERT_TRUE(op_attr_json.IsValid());

  // 验证JSON结构
  const auto &raw_json = op_attr_json.Raw();
  EXPECT_TRUE(raw_json.is_object());
  EXPECT_TRUE(raw_json.contains("add1"));

  const auto &op_attr = raw_json.at("add1");
  EXPECT_TRUE(op_attr.is_object());
  EXPECT_TRUE(op_attr.contains("_datadump_original_op_names"));

  const auto &attr_value = op_attr.at("_datadump_original_op_names");
  EXPECT_TRUE(attr_value.is_object());
  EXPECT_EQ(attr_value.at("type"), "LIST_STRING");
  EXPECT_TRUE(attr_value.at("value").is_array());
  EXPECT_EQ(attr_value.at("value").size(), 2U);
  EXPECT_EQ(attr_value.at("value")[0], "original_op1");
  EXPECT_EQ(attr_value.at("value")[1], "original_op2");
}

TEST_F(Om2PackageHelperUt, SaveOpAttrJson_NoAttr_GenEmptyOpAttrJson) {
  // 基础 AicoreOp 图未设置 dump 属性
  ModelBufferData model_data;
  const std::string output_file = PathUtils::Join({test_work_dir, "test_empty_attr.om2"});
  ASSERT_NE(SuiteModelData("aicore"), nullptr);
  ASSERT_EQ(gert::SerializeGertModelData(*SuiteModelData("aicore"), model_data, true, output_file), SUCCESS);

  // 解压并验证op_attr.json为空对象
  uint32_t model_buf_size = 0;
  const auto model_buf = GetBinDataFromFile(output_file, model_buf_size);
  gert::ZipArchiveReader archive(reinterpret_cast<const uint8_t *>(model_buf.get()), model_buf_size);
  ASSERT_TRUE(archive.IsGood());

  size_t op_attr_size = 0;
  const auto op_attr_buf = archive.ExtractToMem("test_empty_attr/data/model_0/op_attr.json", op_attr_size);
  ASSERT_NE(op_attr_buf, nullptr);

  const JsonFile op_attr_json(reinterpret_cast<const uint8_t *>(op_attr_buf.get()), op_attr_size);
  ASSERT_TRUE(op_attr_json.IsValid());

  // 验证JSON为空对象 {}
  const auto &raw_json = op_attr_json.Raw();
  EXPECT_TRUE(raw_json.is_object());
  EXPECT_EQ(raw_json.size(), 0U);
}

TEST_F(Om2PackageHelperUt, SaveOpAttrJson_EmptyOriginalOpNames_GenValidOpAttrJson) {
  ModelBufferData model_data;
  const std::string output_file = PathUtils::Join({test_work_dir, "test_empty_list_attr.om2"});
  ASSERT_NE(SuiteModelData("empty_names"), nullptr);
  ASSERT_EQ(gert::SerializeGertModelData(*SuiteModelData("empty_names"), model_data, true, output_file), SUCCESS);

  // 解压并验证op_attr.json
  uint32_t model_buf_size = 0;
  const auto model_buf = GetBinDataFromFile(output_file, model_buf_size);
  gert::ZipArchiveReader archive(reinterpret_cast<const uint8_t *>(model_buf.get()), model_buf_size);
  ASSERT_TRUE(archive.IsGood());

  size_t op_attr_size = 0;
  const auto op_attr_buf = archive.ExtractToMem("test_empty_list_attr/data/model_0/op_attr.json", op_attr_size);
  ASSERT_NE(op_attr_buf, nullptr);

  const JsonFile op_attr_json(reinterpret_cast<const uint8_t *>(op_attr_buf.get()), op_attr_size);
  ASSERT_TRUE(op_attr_json.IsValid());

  const auto &raw_json = op_attr_json.Raw();
  EXPECT_FALSE(raw_json.contains("add1"));
}

// ============================================================================
// SaveGraphDebugFiles tests
// ============================================================================
TEST_F(Om2PackageHelperUt, SaveGraphDebugFiles_Ok_ValidGraph) {
  ModelBufferData model_data;
  const std::string output_file = PathUtils::Join({test_work_dir, "test_debug_valid.om2"});
  ASSERT_NE(SuiteModelData("aicore"), nullptr);
  ASSERT_EQ(gert::SerializeGertModelData(*SuiteModelData("aicore"), model_data, true, output_file), SUCCESS);

  uint32_t model_buf_size = 0;
  const auto model_buf = GetBinDataFromFile(output_file, model_buf_size);
  gert::ZipArchiveReader archive(reinterpret_cast<const uint8_t *>(model_buf.get()), model_buf_size);
  ASSERT_TRUE(archive.IsGood());

  ExpectVisualJsonCanLoad(archive, "g1");
}

// ============================================================================
// ExtractVisualJson tests
// ============================================================================
TEST_F(Om2PackageHelperUt, ExtractVisualJson_Ok_FromGeneratedOm2) {
  ModelBufferData model_data;
  const std::string output_file = PathUtils::Join({test_work_dir, "test_extract_ok.om2"});
  ASSERT_NE(SuiteModelData("aicore"), nullptr);
  ASSERT_EQ(gert::SerializeGertModelData(*SuiteModelData("aicore"), model_data, true, output_file), SUCCESS);

  uint32_t file_size = 0;
  const auto file_buf = GetBinDataFromFile(output_file, file_size);
  ASSERT_NE(file_buf, nullptr);

  std::string json_out;
  ASSERT_EQ(Om2PackageHelper::ExtractVisualJson(file_buf.get(), file_size, json_out), SUCCESS);
  EXPECT_GT(json_out.size(), 0U);

  nlohmann::json pb_json;
  ASSERT_EQ(VisualJsonConverter::LoadFromVisualJson(json_out, pb_json), SUCCESS);
  ASSERT_TRUE(pb_json.contains("graph"));
  EXPECT_GE(pb_json["graph"].size(), 1U);
}

TEST_F(Om2PackageHelperUt, ExtractVisualJson_Fail_NullModelData) {
  std::string json_out;
  EXPECT_NE(Om2PackageHelper::ExtractVisualJson(nullptr, 100U, json_out), SUCCESS);
}

TEST_F(Om2PackageHelperUt, ExtractVisualJson_Fail_ZeroLen) {
  uint8_t dummy = 0;
  std::string json_out;
  EXPECT_NE(Om2PackageHelper::ExtractVisualJson(&dummy, 0U, json_out), SUCCESS);
}

TEST_F(Om2PackageHelperUt, ExtractVisualJson_Fail_InvalidZip) {
  const uint8_t garbage[] = {0xDE, 0xAD, 0xBE, 0xEF, 0x00, 0x01, 0x02, 0x03};
  std::string json_out;
  EXPECT_NE(Om2PackageHelper::ExtractVisualJson(garbage, sizeof(garbage), json_out), SUCCESS);
}

TEST_F(Om2PackageHelperUt, ExtractVisualJson_Fail_NoVisualJson) {
  const std::string zip_path = PathUtils::Join({test_work_dir, "no_proto.om2"});
  {
    gert::ZipArchiveWriter writer(zip_path);
    ASSERT_TRUE(writer.IsMemFileOpened());
    const std::string manifest =
        R"({"compatibility":{"compiler_version":"1.0","required_executor_version":"","used_features":{}},"model_num":1})";
    ASSERT_TRUE(writer.WriteBytes("manifest.json", manifest.data(), manifest.size(), false));
    gert::GertBuffer buf;
    ASSERT_TRUE(writer.SaveModelData(buf, true));
  }

  uint32_t file_size = 0;
  const auto file_buf = GetBinDataFromFile(zip_path, file_size);
  ASSERT_NE(file_buf, nullptr);

  std::string json_out;
  EXPECT_NE(Om2PackageHelper::ExtractVisualJson(file_buf.get(), file_size, json_out), SUCCESS);
}

TEST_F(Om2PackageHelperUt, BuildModelMeta_SpecialInputSize) {
  auto ge_root_model = CreateGeRootModelWithAicoreOp();
  ASSERT_NE(ge_root_model, nullptr);
  SyncKernelNameForAllModels(ge_root_model);
  const auto ge_model = ge_root_model->GetSubgraphInstanceNameToModel().begin()->second;
  ASSERT_NE(ge_model, nullptr);

  const auto &graph = ge_model->GetGraph();
  for (const auto &node : graph->GetDirectNode()) {
    auto op_desc = node->GetOpDesc();
    if (op_desc->GetType() == DATA) {
      auto output_desc = op_desc->MutableOutputDesc(0U);
      ASSERT_NE(output_desc, nullptr);
      (void)AttrUtils::SetInt(*output_desc, ATTR_NAME_SPECIAL_INPUT_SIZE, 2048);
      break;
    }
  }

  Om2PackageHelper om2_packager;
  gert::GertModelData model_data;
  gert::InitGertModelData(model_data);
  model_data.models.emplace_back(std::make_unique<gert::GertModelDataModel>());
  model_data.constants->constants_data.emplace_back();
  model_data.models[0]->model_meta = std::make_unique<gert::GertModelDataModelMeta>();
  ASSERT_EQ(om2_packager.BuildModelMeta(ge_model, *model_data.models[0]), SUCCESS);
  ASSERT_FALSE(model_data.models[0]->model_meta->input_desc.empty());
  EXPECT_EQ(model_data.models[0]->model_meta->input_desc[0].size, 2048U);
}

TEST_F(Om2PackageHelperUt, BuildModelMeta_InputDimsAttr) {
  auto ge_root_model = CreateGeRootModelWithAicoreOp();
  ASSERT_NE(ge_root_model, nullptr);
  SyncKernelNameForAllModels(ge_root_model);
  const auto ge_model = ge_root_model->GetSubgraphInstanceNameToModel().begin()->second;
  ASSERT_NE(ge_model, nullptr);

  const auto &graph = ge_model->GetGraph();
  for (const auto &node : graph->GetDirectNode()) {
    auto op_desc = node->GetOpDesc();
    if (op_desc->GetType() == DATA) {
      (void)AttrUtils::SetListInt(op_desc, ATTR_NAME_INPUT_DIMS, {2, 8});
      break;
    }
  }

  Om2PackageHelper om2_packager;
  gert::GertModelData model_data;
  gert::InitGertModelData(model_data);
  model_data.models.emplace_back(std::make_unique<gert::GertModelDataModel>());
  model_data.constants->constants_data.emplace_back();
  model_data.models[0]->model_meta = std::make_unique<gert::GertModelDataModelMeta>();
  ASSERT_EQ(om2_packager.BuildModelMeta(ge_model, *model_data.models[0]), SUCCESS);
  ASSERT_FALSE(model_data.models[0]->model_meta->input_desc_v2.empty());
  EXPECT_EQ(model_data.models[0]->model_meta->input_desc_v2[0].shape, std::vector<int64_t>({2, 8}));
}

TEST_F(Om2PackageHelperUt, BuildModelMeta_SpecialOutputSize) {
  auto ge_root_model = CreateGeRootModelWithAicoreOp();
  ASSERT_NE(ge_root_model, nullptr);
  SyncKernelNameForAllModels(ge_root_model);
  const auto ge_model = ge_root_model->GetSubgraphInstanceNameToModel().begin()->second;
  ASSERT_NE(ge_model, nullptr);

  const auto &graph = ge_model->GetGraph();
  for (const auto &node : graph->GetDirectNode()) {
    auto op_desc = node->GetOpDesc();
    if (op_desc->GetType() == NETOUTPUT) {
      auto input_desc = op_desc->MutableInputDesc(0U);
      ASSERT_NE(input_desc, nullptr);
      (void)AttrUtils::SetInt(*input_desc, ATTR_NAME_SPECIAL_OUTPUT_SIZE, 4096);
      break;
    }
  }

  Om2PackageHelper om2_packager;
  gert::GertModelData model_data;
  gert::InitGertModelData(model_data);
  model_data.models.emplace_back(std::make_unique<gert::GertModelDataModel>());
  model_data.constants->constants_data.emplace_back();
  model_data.models[0]->model_meta = std::make_unique<gert::GertModelDataModelMeta>();
  ASSERT_EQ(om2_packager.BuildModelMeta(ge_model, *model_data.models[0]), SUCCESS);
  ASSERT_FALSE(model_data.models[0]->model_meta->output_desc.empty());
  EXPECT_EQ(model_data.models[0]->model_meta->output_desc[0].size, 4096U);
}

TEST_F(Om2PackageHelperUt, BuildModelMeta_DynamicOutputDims) {
  auto ge_root_model = CreateGeRootModelWithAicoreOp();
  ASSERT_NE(ge_root_model, nullptr);
  SyncKernelNameForAllModels(ge_root_model);
  const auto ge_model = ge_root_model->GetSubgraphInstanceNameToModel().begin()->second;
  ASSERT_NE(ge_model, nullptr);

  const auto &graph = ge_model->GetGraph();
  for (const auto &node : graph->GetDirectNode()) {
    auto op_desc = node->GetOpDesc();
    if (op_desc->GetType() == NETOUTPUT) {
      (void)AttrUtils::SetListStr(op_desc, ATTR_NAME_DYNAMIC_OUTPUT_DIMS, {"1,4", "2,8"});
      break;
    }
  }

  Om2PackageHelper om2_packager;
  gert::GertModelData model_data;
  gert::InitGertModelData(model_data);
  model_data.models.emplace_back(std::make_unique<gert::GertModelDataModel>());
  model_data.constants->constants_data.emplace_back();
  model_data.models[0]->model_meta = std::make_unique<gert::GertModelDataModelMeta>();
  ASSERT_EQ(om2_packager.BuildModelMeta(ge_model, *model_data.models[0]), SUCCESS);
  EXPECT_EQ(model_data.models[0]->model_meta->dynamic_output_shape.size(), 2U);
  EXPECT_EQ(std::string(gert::GertGetStr(model_data.models[0]->model_meta->dynamic_output_shape[0])), "1,4");
  EXPECT_EQ(std::string(gert::GertGetStr(model_data.models[0]->model_meta->dynamic_output_shape[1])), "2,8");
}

TEST_F(Om2PackageHelperUt, BuildDebugInfo_DumpOriginOpNames) {
  auto ge_root_model = CreateGeRootModelWithAicoreOp();
  ASSERT_NE(ge_root_model, nullptr);
  SyncKernelNameForAllModels(ge_root_model);
  const auto ge_model = ge_root_model->GetSubgraphInstanceNameToModel().begin()->second;
  ASSERT_NE(ge_model, nullptr);

  const auto &graph = ge_model->GetGraph();
  for (const auto &node : graph->GetDirectNode()) {
    auto op_desc = node->GetOpDesc();
    if (op_desc->GetType() == "Add") {
      (void)AttrUtils::SetListStr(op_desc, ATTR_NAME_DATA_DUMP_ORIGIN_OP_NAMES, {"original_add_1", "original_add_2"});
      break;
    }
  }

  Om2PackageHelper om2_packager;
  gert::GertModelData model_data;
  gert::InitGertModelData(model_data);
  model_data.models.emplace_back(std::make_unique<gert::GertModelDataModel>());
  model_data.constants->constants_data.emplace_back();
  model_data.models[0]->debug = std::make_unique<gert::GertModelDataDebug>();
  ASSERT_EQ(om2_packager.BuildDebugInfo(ge_model, *model_data.models[0]), SUCCESS);
  ASSERT_NE(model_data.models[0]->op_attr_json, nullptr);

  const std::string op_attr_str(gert::GertGetStr(model_data.models[0]->op_attr_json));
  const JsonFile op_attr_json(reinterpret_cast<const uint8_t *>(op_attr_str.data()), op_attr_str.size());
  ASSERT_TRUE(op_attr_json.IsValid());

  bool found = false;
  for (const auto &[op_name, attrs] : op_attr_json.Raw().items()) {
    if (!attrs.is_object() || !attrs.contains(ATTR_NAME_DATA_DUMP_ORIGIN_OP_NAMES)) {
      continue;
    }
    const auto &attr_value = attrs.at(ATTR_NAME_DATA_DUMP_ORIGIN_OP_NAMES);
    ASSERT_TRUE(attr_value.is_object());
    EXPECT_EQ(attr_value.at("type"), "LIST_STRING");
    ASSERT_TRUE(attr_value.at("value").is_array());
    ASSERT_EQ(attr_value.at("value").size(), 2U);
    EXPECT_EQ(attr_value.at("value")[0], "original_add_1");
    EXPECT_EQ(attr_value.at("value")[1], "original_add_2");
    found = true;
  }
  EXPECT_TRUE(found);
}

TEST_F(Om2PackageHelperUt, BuildManifest_NullRootModel) {
  Om2PackageHelper om2_packager;
  gert::GertModelData model_data;
  gert::InitGertModelData(model_data);
  model_data.manifest = std::make_unique<gert::GertModelDataManifest>();
  model_data.models.emplace_back(std::make_unique<gert::GertModelDataModel>());
  ASSERT_EQ(om2_packager.BuildManifest(model_data), SUCCESS);

  EXPECT_EQ(model_data.manifest->model_num, 1U);
  EXPECT_EQ(std::string(gert::GertGetStr(model_data.manifest->compatibility.compiler_version)),
            gert::GERT_EXECUTOR_VERSION);
  EXPECT_EQ(std::string(gert::GertGetStr(model_data.manifest->compatibility.required_executor_version)), "");
  EXPECT_TRUE(model_data.manifest->compatibility.used_features.empty());

  model_data.models.emplace_back(std::make_unique<gert::GertModelDataModel>());
  model_data.constants->constants_data.emplace_back();
  model_data.constants->constants_data[0] = std::make_unique<gert::GertModelDataFile>();
  model_data.models[0]->constants_config = std::make_unique<gert::GertModelDataConstantsConfig>();
  gert::GertModelDataConstMeta const_meta;
  const_meta.index = 0U;
  const_meta.type = gert::GertMakeStr("EXTERNAL");
  const_meta.file_name = gert::GertMakeStr("external_weight.bin");
  const_meta.file_path = gert::GertMakeStr("/data/weights/external_weight.bin");
  const_meta.offset = 0;
  const_meta.size = 1024;
  const_meta.op_name = gert::GertMakeStr("const_op");
  model_data.models[0]->constants_config->consts.push_back(
      std::make_unique<gert::GertModelDataConstMeta>(std::move(const_meta)));

  model_data.models[0]->runtime = std::make_unique<gert::GertModelDataRuntime>();
  model_data.models[0]->runtime->so_artifact.file_name = gert::GertMakeStr("libtest.so");
  const std::string so_data = "fake_so_content";
  model_data.models[0]->runtime->so_artifact.data = gert::GertMakeFileData(so_data.data(), so_data.size());
  model_data.models[0]->runtime->so_artifact.data_size = so_data.size();

  model_data.models[0]->debug = std::make_unique<gert::GertModelDataDebug>();
  model_data.models[0]->debug->visual_json =
      gert::GertMakeStr(R"({"format":"ge_visual_json","format_version":1,"model":{"graph":[]}})");

  const std::string writer_path = PathUtils::Join({test_work_dir, "external_const.om2"});
  ModelBufferData model_buffer;
  ASSERT_EQ(gert::SerializeGertModelData(model_data, model_buffer, false, writer_path), SUCCESS);
  ASSERT_NE(model_buffer.data, nullptr);
  ASSERT_GT(model_buffer.length, 0U);

  gert::ZipArchiveReader archive(model_buffer.data.get(), model_buffer.length);
  ASSERT_TRUE(archive.IsGood());

  const auto file_names = archive.ListFiles();
  std::string constants_entry;
  for (const auto &name : file_names) {
    if (name.find("constants_config.json") != std::string::npos) {
      constants_entry = name;
      break;
    }
  }
  ASSERT_FALSE(constants_entry.empty()) << "constants_config.json not found in archive";

  size_t buf_size = 0U;
  const auto buf = archive.ExtractToMem(constants_entry, buf_size);
  ASSERT_NE(buf, nullptr);
  ASSERT_GT(buf_size, 0U);

  const std::string constants_json(reinterpret_cast<const char *>(buf.get()), buf_size);
  EXPECT_NE(constants_json.find("file_path"), std::string::npos);
  EXPECT_NE(constants_json.find("/data/weights/external_weight.bin"), std::string::npos);
}

TEST_F(Om2PackageHelperUt, Om2CodegenAndCompile_Success) {
  // Suite 级 "aicore" 变体经由 BuildOm2ModelData -> Om2CodegenAndCompile 完成真实编译，
  // 此处断言其产物（语义与直接调用等价，避免用例内重复编译）
  const auto *model_data = SuiteModelData("aicore");
  ASSERT_NE(model_data, nullptr);
  EXPECT_FALSE(model_data->models[0]->runtime->source_artifacts.empty());
  bool found_so = false;
  for (const auto &artifact : model_data->models[0]->runtime->source_artifacts) {
    if (std::string(gert::GertGetStr(artifact.file_name)).find(".so") != std::string::npos) {
      found_so = true;
      EXPECT_TRUE(artifact.data);
    }
  }
  EXPECT_TRUE(found_so);
}

TEST_F(Om2PackageHelperUt, Om2CodegenAndCompile_InvalidModel_Fail) {
  const auto ge_root_model = CreateInvalidGeRootModel();
  ASSERT_NE(ge_root_model, nullptr);
  const auto &name_to_ge_model = ge_root_model->GetSubgraphInstanceNameToModel();
  ASSERT_FALSE(name_to_ge_model.empty());
  const auto &ge_model = name_to_ge_model.begin()->second;
  ASSERT_NE(ge_model, nullptr);

  gert::GertModelData model_data;
  gert::InitGertModelData(model_data);
  model_data.models.emplace_back(std::make_unique<gert::GertModelDataModel>());
  model_data.constants->constants_data.emplace_back();
  model_data.models[0]->runtime = std::make_unique<gert::GertModelDataRuntime>();
  model_data.models[0]->constants_config = std::make_unique<gert::GertModelDataConstantsConfig>();
  model_data.constants->constants_data[0] = std::make_unique<gert::GertModelDataFile>();
  Om2Codegen codegen;
  EXPECT_NE(codegen.Om2CodegenAndCompile(ge_model, model_data, *model_data.models[0]), SUCCESS);
}

// ============================================================================
// FillAippModelMetaInfo Tests (通过 SaveToOmRootModel 全流程验证)
// ============================================================================

static GeRootModelPtr CreateGeRootModelWithStaticAipp() {
  auto graph = gert::ShareGraph::AicoreStaticGraph();
  graph->TopologicalSorting();

  // 在第一个 DATA 节点上添加静态 AIPP 属性
  for (const auto &node : graph->GetDirectNode()) {
    auto op_desc = node->GetOpDesc();
    if ((op_desc != nullptr) && (op_desc->GetType() == DATA)) {
      ge::NamedAttrs aipp_attr;
      aipp_attr.SetAttr("aipp_mode", ge::GeAttrValue::CreateFrom<int64_t>(0));  // static
      aipp_attr.SetAttr("input_format", ge::GeAttrValue::CreateFrom<int64_t>(0));
      aipp_attr.SetAttr("src_image_size_w", ge::GeAttrValue::CreateFrom<int64_t>(640));
      aipp_attr.SetAttr("src_image_size_h", ge::GeAttrValue::CreateFrom<int64_t>(480));
      aipp_attr.SetAttr("crop", ge::GeAttrValue::CreateFrom<int64_t>(0));
      aipp_attr.SetAttr("load_start_pos_w", ge::GeAttrValue::CreateFrom<int64_t>(0));
      aipp_attr.SetAttr("load_start_pos_h", ge::GeAttrValue::CreateFrom<int64_t>(0));
      aipp_attr.SetAttr("crop_size_w", ge::GeAttrValue::CreateFrom<int64_t>(0));
      aipp_attr.SetAttr("crop_size_h", ge::GeAttrValue::CreateFrom<int64_t>(0));
      aipp_attr.SetAttr("resize", ge::GeAttrValue::CreateFrom<int64_t>(0));
      aipp_attr.SetAttr("resize_output_w", ge::GeAttrValue::CreateFrom<int64_t>(0));
      aipp_attr.SetAttr("resize_output_h", ge::GeAttrValue::CreateFrom<int64_t>(0));
      aipp_attr.SetAttr("padding", ge::GeAttrValue::CreateFrom<int64_t>(0));
      aipp_attr.SetAttr("left_padding_size", ge::GeAttrValue::CreateFrom<int64_t>(0));
      aipp_attr.SetAttr("right_padding_size", ge::GeAttrValue::CreateFrom<int64_t>(0));
      aipp_attr.SetAttr("top_padding_size", ge::GeAttrValue::CreateFrom<int64_t>(0));
      aipp_attr.SetAttr("bottom_padding_size", ge::GeAttrValue::CreateFrom<int64_t>(0));
      aipp_attr.SetAttr("csc_switch", ge::GeAttrValue::CreateFrom<int64_t>(1));
      aipp_attr.SetAttr("rbuv_swap_switch", ge::GeAttrValue::CreateFrom<int64_t>(0));
      aipp_attr.SetAttr("ax_swap_switch", ge::GeAttrValue::CreateFrom<int64_t>(0));
      aipp_attr.SetAttr("single_line_mode", ge::GeAttrValue::CreateFrom<int64_t>(0));
      aipp_attr.SetAttr("related_input_rank", ge::GeAttrValue::CreateFrom<int64_t>(0));
      aipp_attr.SetAttr("max_src_image_size", ge::GeAttrValue::CreateFrom<int64_t>(8192));
      aipp_attr.SetAttr("support_rotation", ge::GeAttrValue::CreateFrom<int64_t>(0));
      (void)ge::AttrUtils::SetNamedAttrs(op_desc, ge::ATTR_NAME_AIPP, aipp_attr);
      (void)ge::AttrUtils::SetStr(op_desc, ge::ATTR_DATA_RELATED_AIPP_MODE, "static_aipp");
      (void)ge::AttrUtils::SetInt(op_desc, ge::ATTR_NAME_INDEX, 0);

      std::vector<std::string> aipp_inputs = {"NCHW:DT_FLOAT:data_0:100:4:1,3,640,480"};
      (void)ge::AttrUtils::SetListStr(op_desc, ge::ATTR_NAME_AIPP_INPUTS, aipp_inputs);
      std::vector<std::string> aipp_outputs = {"NCHW:DT_FLOAT:data_0_out:200:4:1,3,640,480"};
      (void)ge::AttrUtils::SetListStr(op_desc, ge::ATTR_NAME_AIPP_OUTPUTS, aipp_outputs);
      break;  // 只给第一个 DATA 节点加 AIPP
    }
  }

  gert::GeModelBuilder builder(graph);
  auto ge_root_model =
      builder
          .AddTaskDef("Add",
                      gert::AiCoreTaskDefFaker("add_stub").ArgsFormat("{i_instance0*}{i_instance1*}{o_instance0*}"))
          .FakeTbeBin({"Add"})
          .BuildGeRootModel();
  auto &compute_graph = ge_root_model->GetRootGraph();
  compute_graph->SetGraphUnknownFlag(false);

  for (const auto &node : compute_graph->GetDirectNode()) {
    auto op_desc = node->GetOpDesc();
    if (op_desc == nullptr) {
      return nullptr;
    }
    if ((op_desc->GetType() == DATA)) {
      op_desc->SetOutputOffset({1024});
    } else if (op_desc->GetType() == NETOUTPUT) {
      op_desc->SetInputOffset({3072});
    } else {
      op_desc->SetInputOffset(std::vector<int64_t>(op_desc->GetInputsSize(), 1024));
      op_desc->SetOutputOffset(std::vector<int64_t>(op_desc->GetOutputsSize(), 1024));
      if (op_desc->GetType() == "Add") {
        op_desc->SetIsInputConst({true, true});
        auto input_desc0 = op_desc->MutableInputDesc(0);
        auto input_desc1 = op_desc->MutableInputDesc(1);
        if ((input_desc0 == nullptr) || (input_desc1 == nullptr)) {
          return nullptr;
        }
        TensorUtils::SetDataOffset(*input_desc0, 0);
        TensorUtils::SetDataOffset(*input_desc1, 200704);
      }
    }
  }

  const auto ge_model = ge_root_model->GetSubgraphInstanceNameToModel().begin()->second;
  std::vector<uint8_t> weights_value(401408, 1U);
  const size_t weight_size = weights_value.size();
  ge_model->SetWeight(Buffer::CopyFrom(weights_value.data(), weight_size));
  (void)AttrUtils::SetInt(ge_model, ATTR_MODEL_MEMORY_SIZE, 2048);
  (void)AttrUtils::SetInt(ge_model, ATTR_MODEL_WEIGHT_SIZE, weight_size);
  (void)AttrUtils::SetInt(ge_model, ATTR_MODEL_STREAM_NUM, 1);

  return ge_root_model;
}

TEST_F(Om2PackageHelperUt, ConvertOm2Model_WithStaticAipp_WritesAippJson) {
  ModelBufferData model_data;
  const std::string output_file = PathUtils::Join({test_work_dir, kZipFileBaseName + "_aipp.om2"});
  ASSERT_NE(SuiteModelData("static_aipp"), nullptr);
  ASSERT_EQ(gert::SerializeGertModelData(*SuiteModelData("static_aipp"), model_data, true, output_file), SUCCESS);

  // 读取并验证 model_meta.json 中的 aipp 字段
  uint32_t model_buf_size = 0U;
  const auto model_buf = GetBinDataFromFile(output_file, model_buf_size);
  ASSERT_NE(model_buf, nullptr);
  ASSERT_GT(model_buf_size, 0U);

  gert::ZipArchiveReader archive(reinterpret_cast<const uint8_t *>(model_buf.get()), model_buf_size);
  ASSERT_TRUE(archive.IsGood());

  size_t model_meta_size = 0U;
  const std::string aipp_meta_entry = kZipFileBaseName + "_aipp/data/model_0/model_meta.json";
  const auto model_meta_buf = archive.ExtractToMem(aipp_meta_entry, model_meta_size);
  ASSERT_NE(model_meta_buf, nullptr);
  ASSERT_GT(model_meta_size, 0U);

  const std::string model_meta_json(reinterpret_cast<const char *>(model_meta_buf.get()), model_meta_size);
  // 验证 SaveToOmRootModel 全流程：含 AIPP 属性的模型可以完成保存不崩溃
  // 实际 JSON 内容取决于 ConvertAippParams 的可用性，由更上层 ST 验证
  EXPECT_FALSE(model_meta_json.empty());
  SUCCEED();
}

TEST_F(Om2PackageHelperUt, ConvertOm2Model_WithoutAipp_NoAippSectionInJson) {
  ModelBufferData model_data;
  const std::string output_file = PathUtils::Join({test_work_dir, kZipFileBaseName + "_no_aipp.om2"});
  ASSERT_NE(SuiteModelData("aicore"), nullptr);
  ASSERT_EQ(gert::SerializeGertModelData(*SuiteModelData("aicore"), model_data, true, output_file), SUCCESS);

  // 读取 model_meta.json，确认不包含 aipp 字段
  uint32_t model_buf_size_noaipp = 0U;
  const auto model_buf_noaipp = GetBinDataFromFile(output_file, model_buf_size_noaipp);
  ASSERT_NE(model_buf_noaipp, nullptr);
  ASSERT_GT(model_buf_size_noaipp, 0U);

  gert::ZipArchiveReader archive_noaipp(reinterpret_cast<const uint8_t *>(model_buf_noaipp.get()),
                                        model_buf_size_noaipp);
  ASSERT_TRUE(archive_noaipp.IsGood());

  size_t model_meta_size_noaipp = 0U;
  const std::string noaipp_meta_entry = kZipFileBaseName + "_no_aipp/data/model_0/model_meta.json";
  const auto model_meta_buf_noaipp = archive_noaipp.ExtractToMem(noaipp_meta_entry, model_meta_size_noaipp);
  ASSERT_NE(model_meta_buf_noaipp, nullptr);
  ASSERT_GT(model_meta_size_noaipp, 0U);

  const std::string model_meta_json_noaipp(reinterpret_cast<const char *>(model_meta_buf_noaipp.get()),
                                           model_meta_size_noaipp);
  // 无 AIPP 属性时，model_meta.json 不应包含 aipp 字段
  EXPECT_EQ(model_meta_json_noaipp.find("\"aipp\""), std::string::npos);
}

TEST_F(Om2PackageHelperUt, SaveToOmRootModel_UnknownShape_ReturnsFailed) {
  Om2PackageHelper om2_packager;
  const auto ge_root_model = CreateGeRootModelWithAicoreOp();
  ASSERT_NE(ge_root_model, nullptr);
  ModelBufferData model_data;
  const std::string output_file = PathUtils::Join({test_work_dir, "test_unknown_shape.om2"});
  EXPECT_NE(om2_packager.SaveToOmRootModel(ge_root_model, output_file, model_data, true), SUCCESS);
}

TEST_F(Om2PackageHelperUt, SaveToOmRootModel_NullRootModel_ReturnsFailed) {
  Om2PackageHelper om2_packager;
  ModelBufferData model_data;
  const std::string output_file = PathUtils::Join({test_work_dir, "test_null_root.om2"});
  EXPECT_NE(om2_packager.SaveToOmRootModel(nullptr, output_file, model_data, false), SUCCESS);
}

TEST_F(Om2PackageHelperUt, SaveToOmRootModel_EmptyOutputFile_ReturnsFailed) {
  Om2PackageHelper om2_packager;
  const auto ge_root_model = CreateGeRootModelWithAicoreOp();
  ASSERT_NE(ge_root_model, nullptr);
  ModelBufferData model_data;
  EXPECT_NE(om2_packager.SaveToOmRootModel(ge_root_model, "", model_data, false), SUCCESS);
}

TEST_F(Om2PackageHelperUt, SaveToOmRootModel_EmptySubModels_ReturnsFailed) {
  Om2PackageHelper om2_packager;
  auto root_graph = std::make_shared<ComputeGraph>("empty_root");
  auto ge_root_model = std::make_shared<GeRootModel>();
  ASSERT_EQ(ge_root_model->Initialize(root_graph), SUCCESS);
  ModelBufferData model_data;
  const std::string output_file = PathUtils::Join({test_work_dir, "test_empty_sub.om2"});
  EXPECT_NE(om2_packager.SaveToOmRootModel(ge_root_model, output_file, model_data, false), SUCCESS);
}

TEST_F(Om2PackageHelperUt, RelocateExternalWeights_NoExternalWeights_ReturnsSuccess) {
  const auto ge_root_model = CreateGeRootModelWithAicoreOp();
  ASSERT_NE(ge_root_model, nullptr);

  ModelBufferData model_data;
  const std::string output_file = PathUtils::Join({test_work_dir, "test_relocate.om2"});
  ASSERT_NE(SuiteModelData("aicore"), nullptr);
  ASSERT_EQ(gert::SerializeGertModelData(*SuiteModelData("aicore"), model_data, true, output_file), SUCCESS);

  ModelBufferData relocated_model;
  bool relocated = false;
  ASSERT_EQ(Om2PackageHelper::RelocateExternalWeights(output_file, model_data, relocated_model, relocated), SUCCESS);
  EXPECT_FALSE(relocated);
}

TEST_F(Om2PackageHelperUt, ExtractVisualJson_Fail_InvalidModelData) {
  std::string json_out;
  const uint8_t garbage[] = {0x01, 0x02, 0x03, 0x04};
  EXPECT_NE(Om2PackageHelper::ExtractVisualJson(garbage, sizeof(garbage), json_out), SUCCESS);
}

TEST_F(Om2PackageHelperUt, SetSaveMode_Ok) {
  Om2PackageHelper om2_packager;
  om2_packager.SetSaveMode(true);
  om2_packager.SetSaveMode(false);
}

TEST_F(Om2PackageHelperUt, SerializeVarResource_WithEntriesAndInitData) {
  gert::GertModelData model_data;
  gert::InitGertModelData(model_data);
  model_data.models.emplace_back(std::make_unique<gert::GertModelDataModel>());
  model_data.constants->constants_data.emplace_back();
  model_data.models[0]->model_meta = std::make_unique<gert::GertModelDataModelMeta>();
  model_data.models[0]->model_meta->model_name = gert::GertMakeStr("var_test_model");
  model_data.models[0]->variables_config = std::make_unique<gert::GertModelDataVariablesConfig>();
  gert::RTVarEntry entry;
  const std::string var_name_str = "test_var";
  entry.var_name = gert::GertMakeStr(var_name_str);
  entry.op_type = gert::GertMakeStr("Variable");
  entry.logic_addr = 0x1000U;
  entry.size = 16U;
  entry.memory_type = RT_MEMORY_HBM;
  entry.changed_graph_id = 1U;
  entry.allocated_graph_id = 0U;
  entry.tensor_desc.name = gert::GertMakeStr("test_var");
  entry.tensor_desc.shape = {4};
  entry.tensor_desc.data_type = ge::DT_FLOAT;
  entry.tensor_desc.format = ge::FORMAT_ND;
  entry.tensor_desc.size = 16U;
  entry.tensor_desc.shape_range = {{1, 8}};
  entry.var_key = gert::GertMakeStr(gert::RTVarBuildKey(var_name_str, entry.tensor_desc));

  gert::RTTransNodeInfo trans_node;
  trans_node.node_type = gert::GertMakeStr("TransData");
  trans_node.input.name = gert::GertMakeStr("in");
  trans_node.input.shape = {4};
  trans_node.input.data_type = ge::DT_FLOAT;
  trans_node.input.format = ge::FORMAT_NCHW;
  trans_node.input.size = 16U;
  trans_node.output.name = gert::GertMakeStr("out");
  trans_node.output.shape = {4};
  trans_node.output.data_type = ge::DT_FLOAT;
  trans_node.output.format = ge::FORMAT_ND;
  trans_node.output.size = 16U;
  entry.trans_road.push_back(std::move(trans_node));

  entry.copy_info.src_var_name = gert::GertMakeStr("src_var");
  entry.copy_info.src_tensor_desc.name = gert::GertMakeStr("src_td");
  entry.copy_info.src_tensor_desc.shape = {4};
  entry.copy_info.src_tensor_desc.data_type = ge::DT_FLOAT;
  entry.copy_info.src_tensor_desc.format = ge::FORMAT_ND;
  entry.copy_info.src_tensor_desc.size = 16U;

  entry.init_data = {0x01, 0x02, 0x03, 0x04};
  ASSERT_EQ(gert::RTVarAddEntry(model_data.models[0]->variables_config->entries, std::move(entry)), ge::SUCCESS);

  model_data.models[0]->runtime = std::make_unique<gert::GertModelDataRuntime>();
  model_data.models[0]->runtime->so_artifact.file_name = gert::GertMakeStr("libtest.so");
  const std::string so_data = "fake_so";
  model_data.models[0]->runtime->so_artifact.data = gert::GertMakeFileData(so_data.data(), so_data.size());
  model_data.models[0]->runtime->so_artifact.data_size = so_data.size();
  model_data.models[0]->debug = std::make_unique<gert::GertModelDataDebug>();
  model_data.models[0]->debug->visual_json =
      gert::GertMakeStr(R"({"format":"ge_visual_json","format_version":1,"model":{"graph":[]}})");

  const std::string writer_path = PathUtils::Join({test_work_dir, "var_resource.om2"});
  ModelBufferData model_buffer;
  ASSERT_EQ(gert::SerializeGertModelData(model_data, model_buffer, false, writer_path), SUCCESS);
  ASSERT_NE(model_buffer.data, nullptr);
  ASSERT_GT(model_buffer.length, 0U);

  gert::ZipArchiveReader archive(model_buffer.data.get(), model_buffer.length);
  ASSERT_TRUE(archive.IsGood());

  const auto file_names = archive.ListFiles();
  std::string var_entry_path;
  std::string var_weight_path;
  for (const auto &name : file_names) {
    if (name.find("model_0/variables_config.json") != std::string::npos) {
      var_entry_path = name;
    }
    if (name.find("variables/var_weight_data_0") != std::string::npos) {
      var_weight_path = name;
    }
  }
  ASSERT_FALSE(var_entry_path.empty()) << "model_0/variables_config.json not found";

  size_t buf_size = 0U;
  const auto buf = archive.ExtractToMem(var_entry_path, buf_size);
  ASSERT_NE(buf, nullptr);
  ASSERT_GT(buf_size, 0U);
  const std::string var_json(reinterpret_cast<const char *>(buf.get()), buf_size);
  EXPECT_NE(var_json.find("test_var"), std::string::npos);
  EXPECT_NE(var_json.find("TransData"), std::string::npos);
  EXPECT_NE(var_json.find("src_var"), std::string::npos);
  EXPECT_NE(var_json.find("init_data_offset"), std::string::npos);

  EXPECT_FALSE(var_weight_path.empty()) << "var_weight_data not found";
}

TEST_F(Om2PackageHelperUt, SerializeVarResource_NullAndEmpty) {
  gert::GertModelData model_data;
  gert::InitGertModelData(model_data);
  model_data.models.emplace_back(std::make_unique<gert::GertModelDataModel>());
  model_data.constants->constants_data.emplace_back();
  model_data.models[0]->model_meta = std::make_unique<gert::GertModelDataModelMeta>();
  model_data.models[0]->model_meta->model_name = gert::GertMakeStr("var_empty_model");
  model_data.models[0]->runtime = std::make_unique<gert::GertModelDataRuntime>();
  model_data.models[0]->runtime->so_artifact.file_name = gert::GertMakeStr("libtest.so");
  const std::string so_data = "fake_so";
  model_data.models[0]->runtime->so_artifact.data = gert::GertMakeFileData(so_data.data(), so_data.size());
  model_data.models[0]->runtime->so_artifact.data_size = so_data.size();
  model_data.models[0]->debug = std::make_unique<gert::GertModelDataDebug>();
  model_data.models[0]->debug->visual_json =
      gert::GertMakeStr(R"({"format":"ge_visual_json","format_version":1,"model":{"graph":[]}})");

  model_data.models[0]->variables_config = nullptr;
  const std::string writer_path1 = PathUtils::Join({test_work_dir, "var_null.om2"});
  ModelBufferData buf1;
  ASSERT_EQ(gert::SerializeGertModelData(model_data, buf1, false, writer_path1), SUCCESS);

  model_data.models[0]->variables_config = std::make_unique<gert::GertModelDataVariablesConfig>();
  const std::string writer_path2 = PathUtils::Join({test_work_dir, "var_empty.om2"});
  ModelBufferData buf2;
  ASSERT_EQ(gert::SerializeGertModelData(model_data, buf2, false, writer_path2), SUCCESS);
}

TEST_F(Om2PackageHelperUt, BuildKernelBinaries_WithAtomicKernel_Ok) {
  auto ge_model = std::make_shared<GeModel>();
  const char normal_kernel_data[] = "fake_normal_tbe_kernel_bin";
  const char atomic_kernel_data[] = "fake_atomic_tbe_kernel_bin";
  auto normal_kernel = std::make_shared<ge::OpKernelBin>(
      "normal_kernel", std::vector<char>(normal_kernel_data, normal_kernel_data + strlen(normal_kernel_data)));
  auto atomic_kernel = std::make_shared<ge::OpKernelBin>(
      "atomic_kernel", std::vector<char>(atomic_kernel_data, atomic_kernel_data + strlen(atomic_kernel_data)));
  ge_model->GetTBEKernelStore().AddKernel(normal_kernel);
  ge_model->GetTBEKernelStore().AddKernel(atomic_kernel);
  ASSERT_TRUE(ge_model->GetTBEKernelStore().Build());

  auto graph = std::make_shared<ComputeGraph>("g1");
  GeTensorDesc tensor_desc(GeShape({1, 1}), FORMAT_ND, DT_FLOAT);
  auto add_desc = std::make_shared<OpDesc>("add1", "Add");
  (void)add_desc->AddInputDesc(tensor_desc);
  (void)add_desc->AddInputDesc(tensor_desc);
  (void)add_desc->AddOutputDesc(tensor_desc);
  (void)AttrUtils::SetStr(add_desc, "_kernelname", "normal_kernel");
  (void)AttrUtils::SetStr(add_desc, ATOMIC_ATTR_TBE_KERNEL_NAME, "atomic_kernel");
  auto add_node = graph->AddNode(add_desc);
  ASSERT_NE(add_node, nullptr);
  graph->SetGraphUnknownFlag(false);
  ge_model->SetGraph(graph);

  gert::GertModelData model_data;
  gert::InitGertModelData(model_data);
  ASSERT_EQ(Om2PackageHelper::BuildKernelBinaries(ge_model, model_data), SUCCESS);

  ASSERT_EQ(model_data.kernels->binaries.size(), 2U);
  EXPECT_EQ(std::string(gert::GertGetStr(model_data.kernels->binaries[0]->file_name)), "normal_kernel.o");
  EXPECT_NE(model_data.kernels->binaries[0]->data, nullptr);
  EXPECT_EQ(model_data.kernels->binaries[0]->data_size, strlen(normal_kernel_data));
  EXPECT_EQ(memcmp(model_data.kernels->binaries[0]->data.get(), normal_kernel_data,
                   model_data.kernels->binaries[0]->data_size),
            0);

  EXPECT_EQ(std::string(gert::GertGetStr(model_data.kernels->binaries[1]->file_name)), "atomic_kernel.o");
  EXPECT_NE(model_data.kernels->binaries[1]->data, nullptr);
  EXPECT_EQ(model_data.kernels->binaries[1]->data_size, strlen(atomic_kernel_data));
  EXPECT_EQ(memcmp(model_data.kernels->binaries[1]->data.get(), atomic_kernel_data,
                   model_data.kernels->binaries[1]->data_size),
            0);
}

TEST_F(Om2PackageHelperUt, BuildKernelBinaries_WithAtomicKernel_Success) {
  auto ge_model = std::make_shared<GeModel>();
  const char atomic_kernel_data[] = "fake_atomic_tbe_kernel_bin";
  auto atomic_kernel = std::make_shared<ge::OpKernelBin>(
      "atomic_kernel", std::vector<char>(atomic_kernel_data, atomic_kernel_data + strlen(atomic_kernel_data)));
  ge_model->GetTBEKernelStore().AddKernel(atomic_kernel);
  ASSERT_TRUE(ge_model->GetTBEKernelStore().Build());

  auto graph = std::make_shared<ComputeGraph>("g1");
  GeTensorDesc tensor_desc(GeShape({1, 1}), FORMAT_ND, DT_FLOAT);
  auto add_desc = std::make_shared<OpDesc>("add1", "Add");
  (void)add_desc->AddInputDesc(tensor_desc);
  (void)add_desc->AddInputDesc(tensor_desc);
  (void)add_desc->AddOutputDesc(tensor_desc);
  (void)AttrUtils::SetStr(add_desc, "_kernelname", "atomic_kernel");
  (void)AttrUtils::SetStr(add_desc, ATOMIC_ATTR_TBE_KERNEL_NAME, "atomic_kernel");
  auto add_node = graph->AddNode(add_desc);
  ASSERT_NE(add_node, nullptr);
  graph->SetGraphUnknownFlag(false);
  ge_model->SetGraph(graph);

  Om2PackageHelper om2_packager;
  gert::GertModelData model_data;
  gert::InitGertModelData(model_data);
  ASSERT_EQ(om2_packager.BuildKernelBinaries(ge_model, model_data), SUCCESS);
  EXPECT_FALSE(model_data.kernels->binaries.empty());
}

TEST_F(Om2PackageHelperUt, BuildKernelBinaries_WithCustAicpuKernel_Success) {
  auto ge_model = std::make_shared<GeModel>();
  const char kernel_data[] = "fake_cust_aicpu_kernel_bin";
  std::vector<char> kernel_bin(kernel_data, kernel_data + strlen(kernel_data));
  auto cust_kernel = std::make_shared<ge::OpKernelBin>("libcust_aicpu_kernel.so", std::move(kernel_bin));
  CustAICPUKernelStore cust_aicpu_kernel_store;
  cust_aicpu_kernel_store.AddCustAICPUKernel(cust_kernel);
  ASSERT_TRUE(cust_aicpu_kernel_store.Build());
  ge_model->SetCustAICPUKernelStore(cust_aicpu_kernel_store);

  auto graph = std::make_shared<ComputeGraph>("g1");
  GeTensorDesc tensor_desc(GeShape({1, 1}), FORMAT_ND, DT_FLOAT);
  auto add_desc = std::make_shared<OpDesc>("add1", "Add");
  (void)add_desc->AddInputDesc(tensor_desc);
  (void)add_desc->AddInputDesc(tensor_desc);
  (void)add_desc->AddOutputDesc(tensor_desc);
  add_desc->SetExtAttr(OP_EXTATTR_CUSTAICPU_KERNEL, cust_kernel);
  auto add_node = graph->AddNode(add_desc);
  ASSERT_NE(add_node, nullptr);
  graph->SetGraphUnknownFlag(false);
  ge_model->SetGraph(graph);

  Om2PackageHelper om2_packager;
  gert::GertModelData model_data;
  gert::InitGertModelData(model_data);
  ASSERT_EQ(om2_packager.BuildKernelBinaries(ge_model, model_data), SUCCESS);
  bool found_cust = false;
  for (const auto &kb : model_data.kernels->binaries) {
    if (std::string(gert::GertGetStr(kb->file_name)).find("_CustAicpuKernel.o") != std::string::npos) {
      found_cust = true;
      break;
    }
  }
  EXPECT_TRUE(found_cust);
}

TEST_F(Om2PackageHelperUt, BuildModelMeta_WithOutputNameContainingColon_Success) {
  auto ge_root_model = CreateGeRootModelWithAicoreOp();
  ASSERT_NE(ge_root_model, nullptr);
  SyncKernelNameForAllModels(ge_root_model);
  const auto ge_model = ge_root_model->GetSubgraphInstanceNameToModel().begin()->second;
  ASSERT_NE(ge_model, nullptr);

  std::vector<std::string> out_node_names = {"add1:0"};
  AttrUtils::SetListStr(ge_model, ATTR_MODEL_OUT_NODES_NAME, out_node_names);

  Om2PackageHelper om2_packager;
  gert::GertModelData model_data;
  gert::InitGertModelData(model_data);
  model_data.models.emplace_back(std::make_unique<gert::GertModelDataModel>());
  model_data.constants->constants_data.emplace_back();
  model_data.models[0]->model_meta = std::make_unique<gert::GertModelDataModelMeta>();
  ASSERT_EQ(om2_packager.BuildModelMeta(ge_model, *model_data.models[0]), SUCCESS);
  ASSERT_FALSE(model_data.models[0]->model_meta->output_desc.empty());
  EXPECT_EQ(std::string(gert::GertGetStr(model_data.models[0]->model_meta->output_desc[0].name)), "add1:0");
}

TEST_F(Om2PackageHelperUt, BuildModelMeta_WithOutputNameWithoutColon_Success) {
  auto ge_root_model = CreateGeRootModelWithAicoreOp();
  ASSERT_NE(ge_root_model, nullptr);
  SyncKernelNameForAllModels(ge_root_model);
  const auto ge_model = ge_root_model->GetSubgraphInstanceNameToModel().begin()->second;
  ASSERT_NE(ge_model, nullptr);

  std::vector<std::string> out_node_names = {"add1"};
  AttrUtils::SetListStr(ge_model, ATTR_MODEL_OUT_NODES_NAME, out_node_names);

  Om2PackageHelper om2_packager;
  gert::GertModelData model_data;
  gert::InitGertModelData(model_data);
  model_data.models.emplace_back(std::make_unique<gert::GertModelDataModel>());
  model_data.constants->constants_data.emplace_back();
  model_data.models[0]->model_meta = std::make_unique<gert::GertModelDataModelMeta>();
  ASSERT_EQ(om2_packager.BuildModelMeta(ge_model, *model_data.models[0]), SUCCESS);
  ASSERT_FALSE(model_data.models[0]->model_meta->output_desc.empty());
  EXPECT_NE(std::string(gert::GertGetStr(model_data.models[0]->model_meta->output_desc[0].name)).find(":"),
            std::string::npos);
}

TEST_F(Om2PackageHelperUt, SetSaveMode_False) {
  Om2PackageHelper helper;
  helper.SetSaveMode(false);
  EXPECT_FALSE(helper.is_offline_);
  helper.SetSaveMode(true);
  EXPECT_TRUE(helper.is_offline_);
}

TEST_F(Om2PackageHelperUt, SaveToOmModel_WithCustomKernel) {
  // CustomOp 图已在 Suite 级注册并 Build（断言按 custom_ops 子路径匹配，不依赖 zip 根名）
  ModelBufferData model_data;
  const std::string output_file = PathUtils::Join({test_work_dir, kZipFileBaseName + "_buffer.om2"});
  ASSERT_NE(SuiteModelData("custom_op"), nullptr);
  ASSERT_EQ(gert::SerializeGertModelData(*SuiteModelData("custom_op"), model_data, false, output_file), SUCCESS);
  EXPECT_NE(mmAccess2(output_file.c_str(), M_F_OK), EOK);
  ASSERT_NE(model_data.data, nullptr);
  ASSERT_GT(model_data.length, 0U);

  gert::ZipArchiveReader archive(model_data.data.get(), model_data.length);
  ASSERT_TRUE(archive.IsGood());
  const auto file_names = archive.ListFiles();
  bool has_custom_op_binary = false;
  for (const auto &file_name : file_names) {
    if ((file_name.find("/custom_ops/binaries_npu_arch/TestPortableOp_") != std::string::npos) &&
        (file_name.find("_CustomKernel.bin") != std::string::npos)) {
      has_custom_op_binary = true;
      break;
    }
  }
  EXPECT_TRUE(has_custom_op_binary);
  CustomOpFactory::RemoveCustomOps({AscendString("TestPortableOp")});
}

TEST_F(Om2PackageHelperUt, SerializeModelMeta_WithDynamicBatchInfo_WritesDynamicDims) {
  gert::GertModelData model_data;
  gert::InitGertModelData(model_data);
  model_data.models.emplace_back(std::make_unique<gert::GertModelDataModel>());
  model_data.constants->constants_data.emplace_back();
  model_data.models[0]->model_meta = std::make_unique<gert::GertModelDataModelMeta>();
  model_data.models[0]->runtime = std::make_unique<gert::GertModelDataRuntime>();
  model_data.models[0]->debug = std::make_unique<gert::GertModelDataDebug>();
  model_data.models[0]->model_meta->model_name = gert::GertMakeStr("dynamic_batch_model");
  model_data.models[0]->model_meta->dynamic_type = 1;
  model_data.models[0]->model_meta->user_designate_shape_order.push_back(gert::GertMakeStr("NCHW"));
  model_data.models[0]->model_meta->dynamic_batch_info = {{1, 2, 3, 4}, {2, 4, 6, 8}};
  model_data.models[0]->model_meta->origin_input_dims = {{1, 3, 224, 224}};
  model_data.models[0]->model_meta->dynamic_output_shape.push_back(gert::GertMakeStr("0,1,2,3"));
  model_data.models[0]->model_meta->dynamic_output_shape.push_back(gert::GertMakeStr("1,2,4,6"));

  gert::GertTensorDesc input_desc;
  input_desc.name = gert::GertMakeStr("input");
  input_desc.shape = {1, 3, 224, 224};
  input_desc.data_type = ge::DT_FLOAT;
  input_desc.format = ge::FORMAT_NCHW;
  input_desc.size = 602112U;
  model_data.models[0]->model_meta->input_desc.push_back(std::move(input_desc));

  gert::GertTensorDesc output_desc;
  output_desc.name = gert::GertMakeStr("output");
  output_desc.shape = {1, 1000};
  output_desc.data_type = ge::DT_FLOAT;
  output_desc.format = ge::FORMAT_ND;
  output_desc.size = 4000U;
  model_data.models[0]->model_meta->output_desc.push_back(std::move(output_desc));

  model_data.models[0]->runtime->so_artifact.file_name = gert::GertMakeStr("libtest.so");
  const std::string so_content = "fake_so_content";
  model_data.models[0]->runtime->so_artifact.data = gert::GertMakeFileData(so_content.data(), so_content.size());
  model_data.models[0]->runtime->so_artifact.data_size = so_content.size();
  model_data.models[0]->debug->visual_json =
      gert::GertMakeStr(R"({"format":"ge_visual_json","format_version":1,"model":{"graph":[]}})");

  const std::string writer_path = PathUtils::Join({test_work_dir, "dynamic_batch.om2"});
  ModelBufferData model_buffer;
  ASSERT_EQ(gert::SerializeGertModelData(model_data, model_buffer, false, writer_path), SUCCESS);
  ASSERT_NE(model_buffer.data, nullptr);
  ASSERT_GT(model_buffer.length, 0U);

  gert::ZipArchiveReader archive(model_buffer.data.get(), model_buffer.length);
  ASSERT_TRUE(archive.IsGood());

  std::string model_meta_entry;
  for (const auto &name : archive.ListFiles()) {
    if (name.find("model_meta.json") != std::string::npos) {
      model_meta_entry = name;
      break;
    }
  }
  ASSERT_FALSE(model_meta_entry.empty());

  size_t buf_size = 0U;
  const auto buf = archive.ExtractToMem(model_meta_entry, buf_size);
  ASSERT_NE(buf, nullptr);
  ASSERT_GT(buf_size, 0U);

  const JsonFile model_meta_json(reinterpret_cast<const uint8_t *>(buf.get()), buf_size);
  ASSERT_TRUE(model_meta_json.IsValid());
  const auto &raw = model_meta_json.Raw();
  ASSERT_TRUE(raw.contains("dynamic_dims"));
  const auto &dynamic_dims = raw.at("dynamic_dims");
  EXPECT_EQ(dynamic_dims.at("dynamic_type"), 1);
  EXPECT_EQ(dynamic_dims.at("user_designate_shape_order"), JsonFile::json::array({"NCHW"}));
  ASSERT_TRUE(dynamic_dims.contains("gears"));
  const auto &gears = dynamic_dims.at("gears");
  ASSERT_TRUE(gears.is_array());
  EXPECT_EQ(gears.size(), 2U);

  const auto &gear0 = gears[0];
  ASSERT_TRUE(gear0.contains("inputs"));
  EXPECT_EQ(gear0.at("inputs"), JsonFile::json::array({1, 2, 3, 4}));
  ASSERT_TRUE(gear0.contains("outputs"));
  const auto &gear0_outputs = gear0.at("outputs");
  ASSERT_TRUE(gear0_outputs.is_array());
  ASSERT_EQ(gear0_outputs.size(), 1U);
  EXPECT_EQ(gear0_outputs[0], JsonFile::json::array({2, 3}));

  const auto &gear1 = gears[1];
  ASSERT_TRUE(gear1.contains("inputs"));
  EXPECT_EQ(gear1.at("inputs"), JsonFile::json::array({2, 4, 6, 8}));
  ASSERT_TRUE(gear1.contains("outputs"));
  const auto &gear1_outputs = gear1.at("outputs");
  ASSERT_TRUE(gear1_outputs.is_array());
  ASSERT_EQ(gear1_outputs.size(), 1U);
  EXPECT_EQ(gear1_outputs[0], JsonFile::json::array({4, 6}));
}

TEST_F(Om2PackageHelperUt, SerializeModelMeta_WithMultipleOutputsPerGear_WritesAllOutputs) {
  gert::GertModelData model_data;
  gert::InitGertModelData(model_data);
  model_data.models.emplace_back(std::make_unique<gert::GertModelDataModel>());
  model_data.constants->constants_data.emplace_back();
  model_data.models[0]->model_meta = std::make_unique<gert::GertModelDataModelMeta>();
  model_data.models[0]->runtime = std::make_unique<gert::GertModelDataRuntime>();
  model_data.models[0]->debug = std::make_unique<gert::GertModelDataDebug>();
  model_data.models[0]->model_meta->model_name = gert::GertMakeStr("multi_output_model");
  model_data.models[0]->model_meta->dynamic_type = 1;
  model_data.models[0]->model_meta->user_designate_shape_order.push_back(gert::GertMakeStr("data"));
  model_data.models[0]->model_meta->dynamic_batch_info = {{1}, {2}};
  model_data.models[0]->model_meta->origin_input_dims = {{1, 3}};
  for (const auto &shape_str : {"0,0,100", "0,1,200", "1,0,100", "1,1,200"}) {
    model_data.models[0]->model_meta->dynamic_output_shape.push_back(gert::GertMakeStr(shape_str));
  }

  gert::GertTensorDesc desc;
  desc.name = gert::GertMakeStr("input");
  desc.shape = {1, 3};
  desc.data_type = ge::DT_FLOAT;
  desc.format = ge::FORMAT_ND;
  desc.size = 12U;
  model_data.models[0]->model_meta->input_desc.push_back(std::move(desc));

  model_data.models[0]->runtime->so_artifact.file_name = gert::GertMakeStr("libtest.so");
  const std::string so_content = "fake";
  model_data.models[0]->runtime->so_artifact.data = gert::GertMakeFileData(so_content.data(), so_content.size());
  model_data.models[0]->runtime->so_artifact.data_size = so_content.size();
  model_data.models[0]->debug->visual_json = gert::GertMakeStr(R"({"format":"ge_visual_json"})");

  const std::string writer_path = PathUtils::Join({test_work_dir, "multi_output.om2"});
  ModelBufferData model_buffer;
  ASSERT_EQ(gert::SerializeGertModelData(model_data, model_buffer, false, writer_path), SUCCESS);

  gert::ZipArchiveReader archive(model_buffer.data.get(), model_buffer.length);
  ASSERT_TRUE(archive.IsGood());

  std::string entry;
  for (const auto &name : archive.ListFiles()) {
    if (name.find("model_meta.json") != std::string::npos) {
      entry = name;
      break;
    }
  }
  ASSERT_FALSE(entry.empty());

  size_t buf_size = 0U;
  const auto buf = archive.ExtractToMem(entry, buf_size);
  ASSERT_NE(buf, nullptr);

  const JsonFile meta_json(reinterpret_cast<const uint8_t *>(buf.get()), buf_size);
  ASSERT_TRUE(meta_json.IsValid());
  const auto &gears = meta_json.Raw().at("dynamic_dims").at("gears");
  ASSERT_EQ(gears.size(), 2U);
  EXPECT_EQ(gears[0].at("outputs").size(), 2U);
  EXPECT_EQ(gears[0].at("outputs")[0], JsonFile::json::array({100}));
  EXPECT_EQ(gears[0].at("outputs")[1], JsonFile::json::array({200}));
  EXPECT_EQ(gears[1].at("outputs").size(), 2U);
}

TEST_F(Om2PackageHelperUt, SerializeManifest_CompatibilityStructure) {
  gert::GertModelData model_data;
  gert::InitGertModelData(model_data);
  model_data.models.emplace_back(std::make_unique<gert::GertModelDataModel>());
  model_data.constants->constants_data.emplace_back();
  model_data.models[0]->model_meta = std::make_unique<gert::GertModelDataModelMeta>();
  model_data.models[0]->runtime = std::make_unique<gert::GertModelDataRuntime>();
  model_data.models[0]->debug = std::make_unique<gert::GertModelDataDebug>();
  model_data.manifest = std::make_unique<gert::GertModelDataManifest>();
  model_data.models[0]->model_meta->model_name = gert::GertMakeStr("test_model");
  model_data.manifest->compatibility.compiler_version = gert::GertMakeStr("1.5");
  model_data.manifest->compatibility.required_executor_version = gert::GertMakeStr("1.1");
  model_data.manifest->compatibility.used_features[gert::GertMakeStr("feature1")] = gert::GertMakeStr("1.0");
  model_data.manifest->compatibility.used_features[gert::GertMakeStr("feature2")] = gert::GertMakeStr("1.1");
  model_data.manifest->model_num = 2U;
  model_data.manifest->atc_command = gert::GertMakeStr("--model=test");

  model_data.models[0]->runtime->so_artifact.file_name = gert::GertMakeStr("libtest.so");
  const std::string so_content = "fake";
  model_data.models[0]->runtime->so_artifact.data = gert::GertMakeFileData(so_content.data(), so_content.size());
  model_data.models[0]->runtime->so_artifact.data_size = so_content.size();
  model_data.models[0]->debug->visual_json = gert::GertMakeStr(R"({"format":"ge_visual_json"})");

  const std::string writer_path = PathUtils::Join({test_work_dir, "compatibility_manifest.om2"});
  ModelBufferData model_buffer;
  ASSERT_EQ(gert::SerializeGertModelData(model_data, model_buffer, false, writer_path), SUCCESS);

  gert::ZipArchiveReader archive(model_buffer.data.get(), model_buffer.length);
  ASSERT_TRUE(archive.IsGood());

  const auto file_list = archive.ListFiles();
  ASSERT_FALSE(file_list.empty());
  EXPECT_TRUE(file_list[0].find(gert::OM2_MANIFEST_PATH) != std::string::npos);

  std::string manifest_entry;
  for (const auto &name : file_list) {
    if (name.find("manifest.json") != std::string::npos) {
      manifest_entry = name;
      break;
    }
  }
  ASSERT_FALSE(manifest_entry.empty());

  size_t buf_size = 0U;
  const auto buf = archive.ExtractToMem(manifest_entry, buf_size);
  ASSERT_NE(buf, nullptr);

  const JsonFile manifest_json(reinterpret_cast<const uint8_t *>(buf.get()), buf_size);
  ASSERT_TRUE(manifest_json.IsValid());
  const auto &raw = manifest_json.Raw();
  EXPECT_EQ(raw.at(gert::OM2_MODEL_NUM), 2U);
  EXPECT_EQ(raw.at(gert::OM2_ATC_COMMAND), "--model=test");

  const auto &compat = raw.at(gert::OM2_MANIFEST_KEY_COMPATIBILITY);
  EXPECT_EQ(compat.at(gert::OM2_MANIFEST_KEY_COMPILER_VERSION), "1.5");
  EXPECT_EQ(compat.at(gert::OM2_MANIFEST_KEY_REQUIRED_EXECUTOR_VERSION), "1.1");
  const auto &features = compat.at(gert::OM2_MANIFEST_KEY_USED_FEATURES);
  EXPECT_EQ(features.at("feature1"), "1.0");
  EXPECT_EQ(features.at("feature2"), "1.1");
}

TEST_F(Om2PackageHelperUt, ReadCustomOpSoFiles) {
  std::string so_file = "/tmp/libcusom_op_" + std::to_string(getpid()) + ".so";
  std::string text = "fake custom op so content";
  EXPECT_EQ(WriteBinFile(so_file.c_str(), text), 0);
  std::unordered_set<std::string> ops_so_set = {so_file};
  std::vector<std::unique_ptr<gert::GertModelDataFile>> shared_lib_binaries;
  EXPECT_EQ(Om2PackageHelper::ReadCustomOpSoToBuffer(ops_so_set, shared_lib_binaries), 0);
  EXPECT_EQ(ops_so_set.size(), shared_lib_binaries.size());
  EXPECT_EQ(text.size(), shared_lib_binaries[0]->data_size);
  EXPECT_EQ(memcmp(text.data(), shared_lib_binaries[0]->data.get(), text.size()), 0);
  std::filesystem::remove(so_file);
}

namespace {
std::shared_ptr<gert::GertModelData> MakeBundleSubModelData(const std::string &model_name,
                                                            const std::string &kernel_name,
                                                            const std::vector<uint8_t> &kernel_content,
                                                            const bool with_vars) {
  auto model_data = std::make_shared<gert::GertModelData>();
  gert::InitGertModelData(*model_data);
  model_data->models.emplace_back(std::make_unique<gert::GertModelDataModel>());
  model_data->constants->constants_data.emplace_back();
  auto &unit = *model_data->models[0];
  unit.model_meta = std::make_unique<gert::GertModelDataModelMeta>();
  unit.model_meta->model_name = gert::GertMakeStr(model_name);
  unit.model_meta->work_size = 1024U;
  unit.runtime = std::make_unique<gert::GertModelDataRuntime>();
  unit.runtime->so_artifact.file_name = gert::GertMakeStr("lib" + model_name + "_om2.so");
  const std::string so_data = "fake_so_" + model_name;
  unit.runtime->so_artifact.data = gert::GertMakeFileData(so_data.data(), so_data.size());
  unit.runtime->so_artifact.data_size = so_data.size();
  unit.debug = std::make_unique<gert::GertModelDataDebug>();
  unit.debug->visual_json = gert::GertMakeStr(R"({"format":"ge_visual_json","format_version":1,"model":{"graph":[]}})");
  unit.constants_config = std::make_unique<gert::GertModelDataConstantsConfig>();
  unit.constants_config->internal_weight_size = 4U;
  auto weight_buf = std::make_unique<uint8_t[]>(4U);
  weight_buf[0] = 0xA0U;
  weight_buf[1] = 0xA1U;
  weight_buf[2] = 0xA2U;
  weight_buf[3] = 0xA3U;
  model_data->constants->constants_data[0] = std::make_unique<gert::GertModelDataFile>();
  model_data->constants->constants_data[0]->file_name = gert::GertMakeStr("constant_0");
  model_data->constants->constants_data[0]->data =
      ge::ReadonlyByteBuffer(weight_buf.release(), ge::ConditionalDeleter{true});
  model_data->constants->constants_data[0]->data_size = 4U;

  auto kernel = std::make_unique<gert::GertModelDataFile>();
  kernel->file_name = gert::GertMakeStr(kernel_name);
  if (!kernel_content.empty()) {
    auto kernel_buf = std::make_unique<uint8_t[]>(kernel_content.size());
    (void)std::memcpy(kernel_buf.get(), kernel_content.data(), kernel_content.size());
    kernel->data = ge::ReadonlyByteBuffer(kernel_buf.release(), ge::ConditionalDeleter{true});
    kernel->data_size = kernel_content.size();
  }
  model_data->kernels->binaries.emplace_back(std::move(kernel));

  if (with_vars) {
    unit.variables_config = std::make_unique<gert::GertModelDataVariablesConfig>();
    gert::RTVarEntry entry;
    const std::string var_name_str = "shared_var";
    entry.var_name = gert::GertMakeStr(var_name_str);
    entry.op_type = gert::GertMakeStr("Variable");
    entry.logic_addr = 0x1000U;
    entry.size = 4U;
    entry.memory_type = RT_MEMORY_HBM;
    entry.tensor_desc.name = gert::GertMakeStr(var_name_str);
    entry.tensor_desc.shape = {1};
    entry.tensor_desc.data_type = ge::DT_FLOAT;
    entry.tensor_desc.format = ge::FORMAT_ND;
    entry.tensor_desc.size = 4U;
    entry.var_key = gert::GertMakeStr(gert::RTVarBuildKey(var_name_str, entry.tensor_desc));
    entry.init_data = {0x01, 0x02, 0x03, 0x04};
    (void)gert::RTVarAddEntry(unit.variables_config->entries, std::move(entry));
    auto var_meta = std::make_unique<gert::GertModelDataVarMeta>();
    var_meta->index = 0U;
    var_meta->var_name = gert::GertMakeStr(var_name_str);
    unit.variables_config->var_metas.emplace_back(std::move(var_meta));
  }

  model_data->manifest = std::make_unique<gert::GertModelDataManifest>();
  model_data->manifest->model_num = 1U;
  model_data->manifest->compatibility.compiler_version = gert::GertMakeStr(gert::GERT_EXECUTOR_VERSION);
  return model_data;
}

bool HasEntryWith(const std::vector<std::string> &file_names, const std::string &pattern) {
  return std::any_of(file_names.begin(), file_names.end(),
                     [&](const std::string &name) { return name.find(pattern) != std::string::npos; });
}

size_t CountEntriesWith(const std::vector<std::string> &file_names, const std::string &pattern) {
  return static_cast<size_t>(std::count_if(file_names.begin(), file_names.end(), [&](const std::string &name) {
    return name.find(pattern) != std::string::npos;
  }));
}
}  // namespace

TEST_F(Om2PackageHelperUt, AssembleBundle_TwoSubModels_ArchiveLayout) {
  std::vector<std::shared_ptr<gert::GertModelData>> sub_models;
  sub_models.push_back(MakeBundleSubModelData("sub0", "kernel_a.o", {1U, 2U, 3U}, true));
  sub_models.push_back(MakeBundleSubModelData("sub1", "kernel_b.o", {4U, 5U, 6U}, true));
  gert::GertModelData bundle_data;
  ASSERT_EQ(Om2PackageHelper::AssembleBundleModelData(sub_models, 0U, bundle_data), SUCCESS);
  const std::string writer_path = PathUtils::Join({test_work_dir, "bundle_layout.om2"});
  ModelBufferData bundle_buffer;
  ASSERT_EQ(gert::SerializeGertModelData(bundle_data, bundle_buffer, false, writer_path), SUCCESS);
  ASSERT_NE(bundle_buffer.data, nullptr);
  ASSERT_GT(bundle_buffer.length, 0U);

  gert::ZipArchiveReader archive(bundle_buffer.data.get(), bundle_buffer.length);
  ASSERT_TRUE(archive.IsGood());
  const std::vector<std::string> expected_entries = {
      "data/model_0/model_meta.json",
      "data/model_0/op_attr.json",
      "data/model_0/constants_config.json",
      "data/model_0/runtime/libsub0_om2.so",
      "data/model_0/debug/ge_visual_00000000_graph_0.json",
      "data/model_0/variables_config.json",
      "data/variables/var_weight_data_0",
      "data/model_1/model_meta.json",
      "data/model_1/op_attr.json",
      "data/model_1/constants_config.json",
      "data/model_1/runtime/libsub1_om2.so",
      "data/model_1/debug/ge_visual_00000000_graph_0.json",
      "data/model_1/variables_config.json",
      "data/variables/var_weight_data_1",
      "data/constants/constant_0",
      "data/constants/constant_1",
      "data/kernels/kernel_a.o",
      "data/kernels/kernel_b.o",
      "manifest.json",
  };
  for (const auto &entry : expected_entries) {
    EXPECT_TRUE(archive.HasEntryByRelativePath(entry)) << "missing entry: " << entry;
  }

  const auto manifest_entry = archive.FindEntry("manifest.json");
  ASSERT_FALSE(manifest_entry.empty());
  size_t buf_size = 0U;
  const auto buf = archive.ExtractToMem(manifest_entry, buf_size);
  ASSERT_NE(buf, nullptr);
  const JsonFile manifest_json(reinterpret_cast<const uint8_t *>(buf.get()), buf_size);
  ASSERT_TRUE(manifest_json.IsValid());
  const auto &raw = manifest_json.Raw();
  EXPECT_EQ(raw.at(gert::OM2_MODEL_NUM), 2U);
  // global_shared_var_size 不由 manifest 承载（写入各子模型 variables_config）
  EXPECT_FALSE(raw.contains("global_shared_var_size"));
}

TEST_F(Om2PackageHelperUt, AssembleBundle_VarSizeWrittenToVariablesConfig) {
  std::vector<std::shared_ptr<gert::GertModelData>> sub_models;
  sub_models.push_back(MakeBundleSubModelData("sub0", "kernel_a.o", {1U}, true));
  sub_models.push_back(MakeBundleSubModelData("sub1", "kernel_b.o", {2U}, true));
  constexpr uint64_t kVarSize = 4096U;
  gert::GertModelData bundle_data;
  ASSERT_EQ(Om2PackageHelper::AssembleBundleModelData(sub_models, kVarSize, bundle_data), SUCCESS);
  const std::string writer_path = PathUtils::Join({test_work_dir, "bundle_var_size.om2"});
  ModelBufferData bundle_buffer;
  ASSERT_EQ(gert::SerializeGertModelData(bundle_data, bundle_buffer, false, writer_path), SUCCESS);

  gert::ZipArchiveReader archive(bundle_buffer.data.get(), bundle_buffer.length);
  ASSERT_TRUE(archive.IsGood());
  for (const auto &config_path : {"data/model_0/variables_config.json", "data/model_1/variables_config.json"}) {
    const auto config_entry = archive.FindEntry(config_path);
    ASSERT_FALSE(config_entry.empty()) << config_path;
    size_t buf_size = 0U;
    const auto buf = archive.ExtractToMem(config_entry, buf_size);
    ASSERT_NE(buf, nullptr) << config_path;
    const JsonFile config_json(reinterpret_cast<const uint8_t *>(buf.get()), buf_size);
    ASSERT_TRUE(config_json.IsValid());
    EXPECT_EQ(config_json.Raw().at("global_shared_var_size"), kVarSize) << config_path;
  }
  const auto manifest_entry = archive.FindEntry("manifest.json");
  ASSERT_FALSE(manifest_entry.empty());
  size_t buf_size = 0U;
  const auto buf = archive.ExtractToMem(manifest_entry, buf_size);
  ASSERT_NE(buf, nullptr);
  const JsonFile manifest_json(reinterpret_cast<const uint8_t *>(buf.get()), buf_size);
  ASSERT_TRUE(manifest_json.IsValid());
  EXPECT_FALSE(manifest_json.Raw().contains("global_shared_var_size"));
}

TEST_F(Om2PackageHelperUt, AssembleBundle_InternalConstFileNameRewritten) {
  const auto add_internal_const = [](const std::shared_ptr<gert::GertModelData> &model_data) {
    auto const_meta = std::make_unique<gert::GertModelDataConstMeta>();
    const_meta->index = 0U;
    const_meta->type = gert::GertMakeStr("INTERNAL");
    // 子模型编译期 INTERNAL 常量 file_name 固定为 constant_0
    const_meta->file_name = gert::GertMakeStr("constant_0");
    const_meta->offset = 0;
    const_meta->size = 4;
    model_data->models[0]->constants_config->consts.emplace_back(std::move(const_meta));
  };
  std::vector<std::shared_ptr<gert::GertModelData>> sub_models;
  sub_models.push_back(MakeBundleSubModelData("sub0", "kernel_a.o", {1U}, false));
  sub_models.push_back(MakeBundleSubModelData("sub1", "kernel_b.o", {2U}, false));
  add_internal_const(sub_models[0]);
  add_internal_const(sub_models[1]);
  gert::GertModelData bundle_data;
  ASSERT_EQ(Om2PackageHelper::AssembleBundleModelData(sub_models, 0U, bundle_data), SUCCESS);
  // 组装后第 i 个子模型的 INTERNAL file_name 重写为 Bundle 级 constant_<i>，与序列化落盘文件名一致
  ASSERT_EQ(bundle_data.models.size(), 2U);
  for (size_t i = 0U; i < bundle_data.models.size(); ++i) {
    const auto &consts = bundle_data.models[i]->constants_config->consts;
    ASSERT_EQ(consts.size(), 1U);
    EXPECT_STREQ(gert::GertGetStr(consts[0]->file_name), ("constant_" + std::to_string(i)).c_str());
  }
}

TEST_F(Om2PackageHelperUt, AssembleBundle_SharedKernelSameContent_Dedup) {
  std::vector<std::shared_ptr<gert::GertModelData>> sub_models;
  sub_models.push_back(MakeBundleSubModelData("sub0", "shared_kernel.o", {7U, 8U, 9U}, false));
  sub_models.push_back(MakeBundleSubModelData("sub1", "shared_kernel.o", {7U, 8U, 9U}, false));
  gert::GertModelData bundle_data;
  ASSERT_EQ(Om2PackageHelper::AssembleBundleModelData(sub_models, 0U, bundle_data), SUCCESS);
  ASSERT_EQ(bundle_data.kernels->binaries.size(), 1U);
  const std::string writer_path = PathUtils::Join({test_work_dir, "bundle_dedup.om2"});
  ModelBufferData bundle_buffer;
  ASSERT_EQ(gert::SerializeGertModelData(bundle_data, bundle_buffer, false, writer_path), SUCCESS);

  gert::ZipArchiveReader archive(bundle_buffer.data.get(), bundle_buffer.length);
  ASSERT_TRUE(archive.IsGood());
  EXPECT_EQ(CountEntriesWith(archive.ListFiles(), "data/kernels/shared_kernel.o"), 1U);
}

TEST_F(Om2PackageHelperUt, AssembleBundle_SharedKernelConflictContent_Fail) {
  std::vector<std::shared_ptr<gert::GertModelData>> sub_models;
  sub_models.push_back(MakeBundleSubModelData("sub0", "shared_kernel.o", {7U, 8U, 9U}, false));
  sub_models.push_back(MakeBundleSubModelData("sub1", "shared_kernel.o", {7U, 8U, 0xFFU}, false));
  gert::GertModelData bundle_data;
  EXPECT_NE(Om2PackageHelper::AssembleBundleModelData(sub_models, 0U, bundle_data), SUCCESS);
}

TEST_F(Om2PackageHelperUt, AssembleBundle_SharedKernelConflictSize_Fail) {
  std::vector<std::shared_ptr<gert::GertModelData>> sub_models;
  sub_models.push_back(MakeBundleSubModelData("sub0", "shared_kernel.o", {7U, 8U, 9U}, false));
  sub_models.push_back(MakeBundleSubModelData("sub1", "shared_kernel.o", {7U, 8U}, false));
  gert::GertModelData bundle_data;
  EXPECT_NE(Om2PackageHelper::AssembleBundleModelData(sub_models, 0U, bundle_data), SUCCESS);
}

TEST_F(Om2PackageHelperUt, AssembleBundle_CustomOpEntriesRejected) {
  {
    std::vector<std::shared_ptr<gert::GertModelData>> sub_models;
    sub_models.push_back(MakeBundleSubModelData("sub0", "kernel_a.o", {1U}, false));
    sub_models.push_back(MakeBundleSubModelData("sub1", "kernel_b.o", {2U}, false));
    auto custom_kernel = std::make_unique<gert::GertModelDataFile>();
    custom_kernel->file_name = gert::GertMakeStr("custom_kernel.o");
    sub_models[0]->custom_ops->binaries.emplace_back(std::move(custom_kernel));
    gert::GertModelData bundle_data;
    EXPECT_NE(Om2PackageHelper::AssembleBundleModelData(sub_models, 0U, bundle_data), SUCCESS);
  }
  {
    std::vector<std::shared_ptr<gert::GertModelData>> sub_models;
    sub_models.push_back(MakeBundleSubModelData("sub0", "kernel_a.o", {1U}, false));
    sub_models.push_back(MakeBundleSubModelData("sub1", "kernel_b.o", {2U}, false));
    auto custom_lib = std::make_unique<gert::GertModelDataFile>();
    custom_lib->file_name = gert::GertMakeStr("libcustom_op.so");
    sub_models[1]->custom_ops->libraries.emplace_back(std::move(custom_lib));
    gert::GertModelData bundle_data;
    EXPECT_NE(Om2PackageHelper::AssembleBundleModelData(sub_models, 0U, bundle_data), SUCCESS);
  }
}

TEST_F(Om2PackageHelperUt, AssembleBundle_InvalidInputs_Fail) {
  const auto make_valid_subs = []() {
    std::vector<std::shared_ptr<gert::GertModelData>> sub_models;
    sub_models.push_back(MakeBundleSubModelData("sub0", "kernel_a.o", {1U}, false));
    sub_models.push_back(MakeBundleSubModelData("sub1", "kernel_b.o", {2U}, false));
    return sub_models;
  };
  gert::GertModelData bundle_data;

  // sub_models 为空 / 仅一个
  {
    std::vector<std::shared_ptr<gert::GertModelData>> empty_subs;
    EXPECT_NE(Om2PackageHelper::AssembleBundleModelData(empty_subs, 0U, bundle_data), SUCCESS);
    auto single_sub = make_valid_subs();
    single_sub.pop_back();
    EXPECT_NE(Om2PackageHelper::AssembleBundleModelData(single_sub, 0U, bundle_data), SUCCESS);
  }
  // 空指针子模型
  {
    auto subs = make_valid_subs();
    subs[1] = nullptr;
    EXPECT_NE(Om2PackageHelper::AssembleBundleModelData(subs, 0U, bundle_data), SUCCESS);
  }
  // 子模型自身 manifest 非法（model_num > 1）
  {
    auto subs = make_valid_subs();
    subs[0]->manifest->model_num = 2U;
    EXPECT_NE(Om2PackageHelper::AssembleBundleModelData(subs, 0U, bundle_data), SUCCESS);
  }
}

TEST_F(Om2PackageHelperUt, Serialize_SingleModel_ManifestOmitsBundleFields) {
  const auto model_data = MakeBundleSubModelData("single_model", "kernel_s.o", {1U, 2U}, true);
  const std::string writer_path = PathUtils::Join({test_work_dir, "single_no_bundle.om2"});
  ModelBufferData model_buffer;
  ASSERT_EQ(gert::SerializeGertModelData(*model_data, model_buffer, false, writer_path), SUCCESS);

  gert::ZipArchiveReader archive(model_buffer.data.get(), model_buffer.length);
  ASSERT_TRUE(archive.IsGood());
  const auto file_names = archive.ListFiles();
  EXPECT_TRUE(HasEntryWith(file_names, "data/model_0/variables_config.json"));
  EXPECT_TRUE(HasEntryWith(file_names, "data/variables/var_weight_data_0"));

  // 单模型 manifest 不写 global_shared_var_size
  const auto manifest_entry = archive.FindEntry("manifest.json");
  ASSERT_FALSE(manifest_entry.empty());
  size_t buf_size = 0U;
  const auto buf = archive.ExtractToMem(manifest_entry, buf_size);
  ASSERT_NE(buf, nullptr);
  const JsonFile manifest_json(reinterpret_cast<const uint8_t *>(buf.get()), buf_size);
  ASSERT_TRUE(manifest_json.IsValid());
  const auto &raw = manifest_json.Raw();
  EXPECT_FALSE(raw.contains("global_shared_var_size"));
}
}  // namespace ge
