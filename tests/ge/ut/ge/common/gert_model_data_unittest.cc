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

#include "framework/om2/model_data/gert_model_data.h"
#include "framework/common/gert_model_data_deserialize.h"
#include "framework/common/gert_model_data_utils.h"
#include "framework/common/json_file.h"
#include "framework/om2/model_data/om2_package_contants.h"
#include "framework/common/zip_archive_writer.h"

namespace gert {
namespace {

class Om2ModelDataTest : public testing::Test {
 protected:
  void SetUp() override {}
  void TearDown() override {}
};

// Test default construction
TEST_F(Om2ModelDataTest, DefaultConstruction) {
  GertModelData model_data;
  gert::InitGertModelData(model_data);
  model_data.models.emplace_back(std::make_unique<gert::GertModelDataModel>());
  model_data.constants->constants_data.emplace_back();
  model_data.models[0]->runtime = std::make_unique<GertModelDataRuntime>();
  model_data.models[0]->model_meta = std::make_unique<GertModelDataModelMeta>();
  model_data.models[0]->constants_config = std::make_unique<GertModelDataConstantsConfig>();
  model_data.models[0]->debug = std::make_unique<GertModelDataDebug>();

  EXPECT_TRUE(model_data.models[0]->runtime->source_artifacts.empty());
  EXPECT_TRUE(!model_data.models[0]->runtime->so_artifact.file_name);
  EXPECT_TRUE(!model_data.models[0]->runtime->so_artifact.data);

  EXPECT_TRUE(!model_data.models[0]->model_meta->model_name);
  EXPECT_EQ(model_data.models[0]->model_meta->work_size, 0U);
  EXPECT_EQ(model_data.models[0]->model_meta->zero_copy_size, 0);
  EXPECT_TRUE(model_data.models[0]->model_meta->input_desc.empty());
  EXPECT_TRUE(model_data.models[0]->model_meta->output_desc.empty());
  EXPECT_TRUE(model_data.models[0]->model_meta->input_desc_v2.empty());
  EXPECT_TRUE(model_data.models[0]->model_meta->output_desc_v2.empty());
  EXPECT_TRUE(model_data.models[0]->model_meta->dynamic_batch_info.empty());
  EXPECT_EQ(model_data.models[0]->model_meta->dynamic_type, 0);
  EXPECT_TRUE(model_data.models[0]->model_meta->dynamic_output_shape.empty());
  EXPECT_TRUE(model_data.models[0]->model_meta->user_designate_shape_order.empty());
  EXPECT_TRUE(model_data.models[0]->model_meta->origin_input_dims.empty());

  EXPECT_EQ(model_data.models[0]->constants_config->internal_weight_size, 0U);
  EXPECT_TRUE(model_data.models[0]->constants_config->consts.empty());

  EXPECT_EQ(model_data.constants->constants_data[0], nullptr);
  EXPECT_TRUE(model_data.kernels->binaries.empty());

  EXPECT_TRUE(model_data.models[0]->op_attr_json == nullptr);
  EXPECT_TRUE(!model_data.models[0]->debug->visual_json);
}

// Test populating codegen output
TEST_F(Om2ModelDataTest, PopulateCodegenOutput) {
  GertModelData model_data;
  gert::InitGertModelData(model_data);
  model_data.models.emplace_back(std::make_unique<gert::GertModelDataModel>());
  model_data.constants->constants_data.emplace_back();
  model_data.models[0]->runtime = std::make_unique<GertModelDataRuntime>();

  gert::GertModelDataProgramBody artifact1;
  artifact1.file_name = gert::GertMakeStr("model.cpp");
  const std::string data1 = "int main() { return 0; }";
  artifact1.data = gert::GertMakeStr(data1);
  artifact1.data_len = data1.size();
  model_data.models[0]->runtime->source_artifacts.push_back(std::move(artifact1));

  gert::GertModelDataProgramBody artifact2;
  artifact2.file_name = gert::GertMakeStr("model.h");
  const std::string data2 = "#pragma once\nvoid func();";
  artifact2.data = gert::GertMakeStr(data2);
  artifact2.data_len = data2.size();
  model_data.models[0]->runtime->source_artifacts.push_back(std::move(artifact2));

  model_data.models[0]->runtime->so_artifact.file_name = gert::GertMakeStr("libmodel.so");
  const std::string so_data = "binary data";
  model_data.models[0]->runtime->so_artifact.data = gert::GertMakeStr(so_data);
  model_data.models[0]->runtime->so_artifact.data_len = so_data.size();

  EXPECT_EQ(model_data.models[0]->runtime->source_artifacts.size(), 2U);
  EXPECT_EQ(std::string(gert::GertGetStr(model_data.models[0]->runtime->source_artifacts[0].file_name)), "model.cpp");
  EXPECT_EQ(std::string(gert::GertGetStr(model_data.models[0]->runtime->source_artifacts[1].file_name)), "model.h");
  EXPECT_EQ(std::string(gert::GertGetStr(model_data.models[0]->runtime->so_artifact.file_name)), "libmodel.so");
}

// Test populating model metadata
TEST_F(Om2ModelDataTest, PopulateModelMeta) {
  GertModelData model_data;
  gert::InitGertModelData(model_data);
  model_data.models.emplace_back(std::make_unique<gert::GertModelDataModel>());
  model_data.constants->constants_data.emplace_back();
  model_data.models[0]->model_meta = std::make_unique<GertModelDataModelMeta>();

  model_data.models[0]->model_meta->model_name = gert::GertMakeStr("test_model");
  model_data.models[0]->model_meta->work_size = 1024 * 1024;

  model_data.models[0]->model_meta->input_desc.push_back(
      gert::MakeGertTensorDesc("input", ge::DT_FLOAT, ge::FORMAT_ND, {1, 3, 224, 224}));
  model_data.models[0]->model_meta->output_desc.push_back(
      gert::MakeGertTensorDesc("output", ge::DT_FLOAT, ge::FORMAT_ND, {1, 1000}));

  model_data.models[0]->model_meta->dynamic_batch_info = {{1}, {2}, {4}, {8}};
  model_data.models[0]->model_meta->dynamic_type = 1;

  model_data.models[0]->model_meta->dynamic_output_shape.push_back(gert::GertMakeStr("1,1000"));

  model_data.models[0]->model_meta->origin_input_dims = {{1, 3, 224, 224}};

  EXPECT_EQ(std::string(gert::GertGetStr(model_data.models[0]->model_meta->model_name)), "test_model");
  EXPECT_EQ(model_data.models[0]->model_meta->work_size, 1024 * 1024);
  EXPECT_EQ(model_data.models[0]->model_meta->input_desc.size(), 1U);
  EXPECT_EQ(std::string(gert::GertGetStr(model_data.models[0]->model_meta->input_desc[0].name)), "input");
  EXPECT_EQ(model_data.models[0]->model_meta->output_desc.size(), 1U);
  EXPECT_EQ(std::string(gert::GertGetStr(model_data.models[0]->model_meta->output_desc[0].name)), "output");
  EXPECT_EQ(model_data.models[0]->model_meta->dynamic_batch_info.size(), 4U);
  EXPECT_EQ(model_data.models[0]->model_meta->dynamic_type, 1);
}

// Test populating constants config
TEST_F(Om2ModelDataTest, PopulateConstantsConfig) {
  GertModelData model_data;
  gert::InitGertModelData(model_data);
  model_data.models.emplace_back(std::make_unique<gert::GertModelDataModel>());
  model_data.constants->constants_data.emplace_back();
  model_data.models[0]->constants_config = std::make_unique<GertModelDataConstantsConfig>();

  model_data.models[0]->constants_config->internal_weight_size = 2048;

  gert::GertModelDataConstMeta const1;
  const1.index = 0;
  const1.type = gert::GertMakeStr("weight");
  const1.file_name = gert::GertMakeStr("weight0.bin");
  const1.offset = 0;
  const1.size = 1024;
  model_data.models[0]->constants_config->consts.push_back(
      std::make_unique<gert::GertModelDataConstMeta>(std::move(const1)));

  gert::GertModelDataConstMeta const2;
  const2.index = 1;
  const2.type = gert::GertMakeStr("bias");
  const2.file_name = gert::GertMakeStr("bias0.bin");
  const2.offset = 1024;
  const2.size = 1024;
  model_data.models[0]->constants_config->consts.push_back(
      std::make_unique<gert::GertModelDataConstMeta>(std::move(const2)));

  EXPECT_EQ(model_data.models[0]->constants_config->internal_weight_size, 2048);
  EXPECT_EQ(model_data.models[0]->constants_config->consts.size(), 2U);
  EXPECT_EQ(model_data.models[0]->constants_config->consts[0]->size, 1024);
  EXPECT_EQ(model_data.models[0]->constants_config->consts[1]->size, 1024);
}

// Test populating weight data
TEST_F(Om2ModelDataTest, PopulateWeightData) {
  GertModelData model_data;
  gert::InitGertModelData(model_data);
  model_data.models.emplace_back(std::make_unique<gert::GertModelDataModel>());
  model_data.constants->constants_data.emplace_back();
  model_data.models[0]->constants_config = std::make_unique<GertModelDataConstantsConfig>();
  model_data.constants->constants_data[0] = std::make_unique<GertModelDataConstantsData>();

  auto buf = std::make_unique<uint8_t[]>(5);
  buf[0] = 0x01;
  buf[1] = 0x02;
  buf[2] = 0x03;
  buf[3] = 0x04;
  buf[4] = 0x05;
  model_data.constants->constants_data[0]->data = ge::ReadonlyByteBuffer(buf.release(), ge::ConditionalDeleter{true});
  model_data.constants->constants_data[0]->size = 5U;
  model_data.models[0]->constants_config->internal_weight_size = 5U;

  EXPECT_NE(model_data.constants->constants_data[0]->data, nullptr);
  EXPECT_EQ(model_data.constants->constants_data[0]->data.get()[0], 0x01);
  EXPECT_EQ(model_data.constants->constants_data[0]->data.get()[4], 0x05);
}

// Test populating kernel binaries
TEST_F(Om2ModelDataTest, PopulateKernelBinaries) {
  GertModelData model_data;
  gert::InitGertModelData(model_data);

  GertModelDataKernelBinary kernel1;
  kernel1.name = gert::GertMakeStr("kernel_add");
  auto buf1 = std::make_unique<uint8_t[]>(3);
  buf1[0] = 0x10;
  buf1[1] = 0x20;
  buf1[2] = 0x30;
  kernel1.data = ge::ReadonlyByteBuffer(buf1.release(), ge::ConditionalDeleter{true});
  kernel1.data_size = 3U;
  model_data.kernels->binaries.push_back(std::make_unique<GertModelDataKernelBinary>(std::move(kernel1)));

  GertModelDataKernelBinary kernel2;
  kernel2.name = gert::GertMakeStr("kernel_mul");
  auto buf2 = std::make_unique<uint8_t[]>(3);
  buf2[0] = 0x40;
  buf2[1] = 0x50;
  buf2[2] = 0x60;
  kernel2.data = ge::ReadonlyByteBuffer(buf2.release(), ge::ConditionalDeleter{true});
  kernel2.data_size = 3U;
  model_data.kernels->binaries.push_back(std::make_unique<GertModelDataKernelBinary>(std::move(kernel2)));

  // Verify
  EXPECT_EQ(model_data.kernels->binaries.size(), 2U);
  EXPECT_EQ(std::string(gert::GertGetStr(model_data.kernels->binaries[0]->name)), "kernel_add");
  EXPECT_EQ(model_data.kernels->binaries[0]->data_size, 3U);
  EXPECT_EQ(std::string(gert::GertGetStr(model_data.kernels->binaries[1]->name)), "kernel_mul");
}

// Test populating debug info
TEST_F(Om2ModelDataTest, PopulateDebugInfo) {
  GertModelData model_data;
  gert::InitGertModelData(model_data);
  model_data.models.emplace_back(std::make_unique<gert::GertModelDataModel>());
  model_data.constants->constants_data.emplace_back();
  model_data.models[0]->debug = std::make_unique<GertModelDataDebug>();

  model_data.models[0]->debug->visual_json = gert::GertMakeStr(R"({"format":"ge_visual_json","format_version":1})");

  model_data.models[0]->op_attr_json = gert::GertMakeStr(R"({"add":{"alpha":"1.0","beta":"1.0"}})");

  EXPECT_EQ(std::string(gert::GertGetStr(model_data.models[0]->debug->visual_json)),
            R"({"format":"ge_visual_json","format_version":1})");
  ASSERT_NE(model_data.models[0]->op_attr_json, nullptr);
  EXPECT_EQ(std::string(gert::GertGetStr(model_data.models[0]->op_attr_json)),
            R"({"add":{"alpha":"1.0","beta":"1.0"}})");
}

// Test populating manifest
TEST_F(Om2ModelDataTest, PopulateManifest) {
  GertModelData model_data;
  gert::InitGertModelData(model_data);
  model_data.manifest = std::make_unique<GertModelDataManifest>();
  model_data.manifest->atc_command = gert::GertMakeStr("");
  model_data.manifest->model_num = 1U;

  EXPECT_EQ(model_data.manifest->model_num, 1U);
  EXPECT_EQ(std::string(gert::GertGetStr(model_data.manifest->atc_command)), "");
}

// Test move semantics
TEST_F(Om2ModelDataTest, MoveSemantics) {
  GertModelData model_data1;
  gert::InitGertModelData(model_data1);
  model_data1.models.emplace_back(std::make_unique<gert::GertModelDataModel>());
  model_data1.constants->constants_data.emplace_back();
  model_data1.models[0]->model_meta = std::make_unique<GertModelDataModelMeta>();
  model_data1.models[0]->constants_config = std::make_unique<GertModelDataConstantsConfig>();
  model_data1.constants->constants_data[0] = std::make_unique<GertModelDataConstantsData>();
  model_data1.models[0]->model_meta->model_name = gert::GertMakeStr("test_model");
  auto buf = std::make_unique<uint8_t[]>(3);
  buf[0] = 0x01;
  buf[1] = 0x02;
  buf[2] = 0x03;
  model_data1.constants->constants_data[0]->data = ge::ReadonlyByteBuffer(buf.release(), ge::ConditionalDeleter{true});
  model_data1.constants->constants_data[0]->size = 3U;
  model_data1.models[0]->constants_config->internal_weight_size = 3U;

  GertModelData model_data2 = std::move(model_data1);

  EXPECT_EQ(std::string(gert::GertGetStr(model_data2.models[0]->model_meta->model_name)), "test_model");
  EXPECT_EQ(model_data2.models[0]->constants_config->internal_weight_size, 3U);
  EXPECT_EQ(model_data1.constants, nullptr);
}

TEST_F(Om2ModelDataTest, KernelBinary_DefaultConstruction) {
  GertModelDataKernelBinary kernel;
  EXPECT_EQ(kernel.data, nullptr);
  EXPECT_EQ(kernel.data_size, 0U);
  EXPECT_TRUE(!kernel.name);
}

TEST_F(Om2ModelDataTest, KernelBinary_MoveSemantics) {
  GertModelDataKernelBinary kernel1;
  kernel1.name = gert::GertMakeStr("kernel_move");
  auto buf = std::make_unique<uint8_t[]>(4);
  buf[0] = 0xAA;
  buf[1] = 0xBB;
  buf[2] = 0xCC;
  buf[3] = 0xDD;
  kernel1.data = ge::ReadonlyByteBuffer(buf.release(), ge::ConditionalDeleter{true});
  kernel1.data_size = 4U;

  GertModelDataKernelBinary kernel2 = std::move(kernel1);
  EXPECT_EQ(kernel1.data, nullptr);
  EXPECT_EQ(std::string(gert::GertGetStr(kernel2.name)), "kernel_move");
  EXPECT_EQ(kernel2.data_size, 4U);
  EXPECT_NE(kernel2.data, nullptr);
  EXPECT_EQ(kernel2.data.get()[0], 0xAA);
  EXPECT_EQ(kernel2.data.get()[3], 0xDD);
}

TEST_F(Om2ModelDataTest, KernelBinary_NonOwningBuffer) {
  uint8_t raw_data[] = {0x10, 0x20, 0x30, 0x40};
  GertModelDataKernelBinary kernel;
  kernel.name = gert::GertMakeStr("non_owning_kernel");
  kernel.data = ge::ReadonlyByteBuffer(raw_data, ge::ConditionalDeleter{false});
  kernel.data_size = sizeof(raw_data);

  EXPECT_NE(kernel.data, nullptr);
  EXPECT_EQ(kernel.data.get(), raw_data);
  EXPECT_EQ(kernel.data_size, 4U);
  EXPECT_EQ(kernel.data.get()[0], 0x10);
  EXPECT_EQ(kernel.data.get()[3], 0x40);
}

TEST_F(Om2ModelDataTest, WeightData_NonOwningBuffer) {
  uint8_t raw_weights[] = {0x01, 0x02, 0x03, 0x04, 0x05, 0x06};
  GertModelData model_data;
  gert::InitGertModelData(model_data);
  model_data.models.emplace_back(std::make_unique<gert::GertModelDataModel>());
  model_data.constants->constants_data.emplace_back();
  model_data.models[0]->constants_config = std::make_unique<GertModelDataConstantsConfig>();
  model_data.constants->constants_data[0] = std::make_unique<GertModelDataConstantsData>();
  model_data.constants->constants_data[0]->data = ge::ReadonlyByteBuffer(raw_weights, ge::ConditionalDeleter{false});
  model_data.constants->constants_data[0]->size = sizeof(raw_weights);
  model_data.models[0]->constants_config->internal_weight_size = sizeof(raw_weights);

  EXPECT_NE(model_data.constants->constants_data[0]->data, nullptr);
  EXPECT_EQ(model_data.constants->constants_data[0]->data.get(), raw_weights);
  EXPECT_EQ(model_data.models[0]->constants_config->internal_weight_size, 6U);
  EXPECT_EQ(model_data.constants->constants_data[0]->data.get()[0], 0x01);
  EXPECT_EQ(model_data.constants->constants_data[0]->data.get()[5], 0x06);
}

TEST_F(Om2ModelDataTest, KernelBinaries_WithNullData) {
  GertModelData model_data;
  gert::InitGertModelData(model_data);

  GertModelDataKernelBinary kernel_with_data;
  kernel_with_data.name = gert::GertMakeStr("has_data");
  auto buf = std::make_unique<uint8_t[]>(2);
  buf[0] = 0xFF;
  buf[1] = 0xFE;
  kernel_with_data.data = ge::ReadonlyByteBuffer(buf.release(), ge::ConditionalDeleter{true});
  kernel_with_data.data_size = 2U;
  model_data.kernels->binaries.push_back(std::make_unique<GertModelDataKernelBinary>(std::move(kernel_with_data)));

  GertModelDataKernelBinary kernel_without_data;
  kernel_without_data.name = gert::GertMakeStr("no_data");
  model_data.kernels->binaries.push_back(std::make_unique<GertModelDataKernelBinary>(std::move(kernel_without_data)));

  EXPECT_EQ(model_data.kernels->binaries.size(), 2U);
  EXPECT_NE(model_data.kernels->binaries[0]->data, nullptr);
  EXPECT_EQ(model_data.kernels->binaries[0]->data_size, 2U);
  EXPECT_EQ(model_data.kernels->binaries[1]->data, nullptr);
  EXPECT_EQ(model_data.kernels->binaries[1]->data_size, 0U);
}

TEST_F(Om2ModelDataTest, WeightData_ContentVerification) {
  GertModelData model_data;
  gert::InitGertModelData(model_data);
  model_data.models.emplace_back(std::make_unique<gert::GertModelDataModel>());
  model_data.constants->constants_data.emplace_back();
  model_data.models[0]->constants_config = std::make_unique<GertModelDataConstantsConfig>();
  model_data.constants->constants_data[0] = std::make_unique<GertModelDataConstantsData>();
  constexpr size_t kSize = 8U;
  auto buf = std::make_unique<uint8_t[]>(kSize);
  for (size_t i = 0; i < kSize; ++i) {
    buf[i] = static_cast<uint8_t>(i * 0x11);
  }
  model_data.constants->constants_data[0]->data = ge::ReadonlyByteBuffer(buf.release(), ge::ConditionalDeleter{true});
  model_data.constants->constants_data[0]->size = kSize;
  model_data.models[0]->constants_config->internal_weight_size = kSize;

  EXPECT_NE(model_data.constants->constants_data[0]->data, nullptr);
  const uint8_t *ptr = model_data.constants->constants_data[0]->data.get();
  for (size_t i = 0; i < kSize; ++i) {
    EXPECT_EQ(ptr[i], static_cast<uint8_t>(i * 0x11));
  }
}

TEST_F(Om2ModelDataTest, ModelData_MoveWithKernelBinaries) {
  GertModelData model_data1;
  gert::InitGertModelData(model_data1);
  model_data1.models.emplace_back(std::make_unique<gert::GertModelDataModel>());
  model_data1.constants->constants_data.emplace_back();
  model_data1.models[0]->model_meta = std::make_unique<GertModelDataModelMeta>();
  model_data1.models[0]->constants_config = std::make_unique<GertModelDataConstantsConfig>();
  model_data1.constants->constants_data[0] = std::make_unique<GertModelDataConstantsData>();
  model_data1.models[0]->model_meta->model_name = gert::GertMakeStr("move_kernels");

  for (int i = 0; i < 3; ++i) {
    GertModelDataKernelBinary kb;
    kb.name = gert::GertMakeStr("kernel_" + std::to_string(i));
    auto buf = std::make_unique<uint8_t[]>(2);
    buf[0] = static_cast<uint8_t>(i);
    buf[1] = static_cast<uint8_t>(i + 1);
    kb.data = ge::ReadonlyByteBuffer(buf.release(), ge::ConditionalDeleter{true});
    kb.data_size = 2U;
    model_data1.kernels->binaries.push_back(std::make_unique<GertModelDataKernelBinary>(std::move(kb)));
  }

  auto wbuf = std::make_unique<uint8_t[]>(3);
  wbuf[0] = 0xAA;
  wbuf[1] = 0xBB;
  wbuf[2] = 0xCC;
  model_data1.constants->constants_data[0]->data = ge::ReadonlyByteBuffer(wbuf.release(), ge::ConditionalDeleter{true});
  model_data1.constants->constants_data[0]->size = 3U;
  model_data1.models[0]->constants_config->internal_weight_size = 3U;

  GertModelData model_data2 = std::move(model_data1);

  EXPECT_EQ(std::string(gert::GertGetStr(model_data2.models[0]->model_meta->model_name)), "move_kernels");
  EXPECT_EQ(model_data2.kernels->binaries.size(), 3U);
  EXPECT_EQ(std::string(gert::GertGetStr(model_data2.kernels->binaries[0]->name)), "kernel_0");
  EXPECT_EQ(model_data2.kernels->binaries[0]->data_size, 2U);
  EXPECT_EQ(model_data2.kernels->binaries[0]->data.get()[0], 0x00);
  EXPECT_EQ(std::string(gert::GertGetStr(model_data2.kernels->binaries[2]->name)), "kernel_2");
  EXPECT_EQ(model_data2.kernels->binaries[2]->data.get()[0], 0x02);
  EXPECT_NE(model_data2.constants->constants_data[0]->data, nullptr);
  EXPECT_EQ(model_data2.models[0]->constants_config->internal_weight_size, 3U);

  EXPECT_EQ(model_data1.constants, nullptr);
  EXPECT_EQ(model_data1.kernels, nullptr);
}

// aipp 解析已内化为反序列化内部逻辑，通过公共 C API 集成验证 GertModelDataAippMeta 字段
TEST_F(Om2ModelDataTest, DeserializeModelMetaWithAippFillsGertModelDataAippMeta) {
  const std::string path = "ut_gert_model_data_aipp.om2";
  gert::ZipArchiveWriter writer(path);
  ASSERT_TRUE(writer.IsMemFileOpened());
  const std::string manifest =
      R"({"compatibility":{"compiler_version":"1.0","required_executor_version":"","used_features":{}},"model_num":1})";
  ASSERT_TRUE(writer.WriteBytes(gert::OM2_MANIFEST_PATH, manifest.data(), manifest.size(), false));
  const std::string meta = R"({
    "inputs": [{"name": "x", "index": 0, "shape": [1, 3, 224, 224], "data_type": "DT_FLOAT", "format": "NCHW",
                "size": 602112, "shape_range": []}],
    "outputs": [{"name": "y", "index": 0, "shape": [1, 1000], "data_type": "DT_FLOAT", "format": "ND",
                 "size": 4000, "shape_range": []}],
    "work_size": 4096, "zero_copy_size": 0, "name": "aipp_model",
    "aipp": {"aipp_infos": [{"index": 0, "aipp_type": 1, "aipp_data_index": 0,
                              "aipp_mode": 0, "input_format": 0,
                              "src_image_size_w": 224, "src_image_size_h": 224,
                              "crop": 0, "crop_size_w": 0, "crop_size_h": 0,
                              "resize": 0, "resize_output_w": 0, "resize_output_h": 0,
                              "padding": 0, "csc_switch": 0, "support_rotation": 0,
                              "related_input_rank": 0, "max_src_image_size": 0,
                              "aipp_inputs": ["NCHW:DT_FLOAT:data:0:4:1,3,224,224"],
                              "aipp_outputs": ["NCHW:DT_FLOAT:data:0:3:1,1,1000"],
                              "orig_input_format": 0, "orig_input_data_type": 0, "orig_input_dim_num": 4}]}
  })";
  ASSERT_TRUE(
      writer.WriteBytes(gert::FormatOm2Path(gert::OM2_MODEL_META_PATH_FORMAT, "0"), meta.data(), meta.size(), false));
  const std::string fake_so(64, 'S');
  ASSERT_TRUE(writer.WriteBytes(gert::FormatOm2Path(gert::OM2_RUNTIME_DIR_FORMAT, "0") + "libaipp_om2.so",
                                fake_so.data(), fake_so.size(), false));
  const std::string op_attr = "{}";
  ASSERT_TRUE(writer.WriteBytes(gert::FormatOm2Path(gert::OM2_OP_ATTR_PATH_FORMAT, "0"), op_attr.data(), op_attr.size(),
                                false));
  const std::string consts = R"({"internal_weight_size": 0, "consts": {}})";
  ASSERT_TRUE(writer.WriteBytes(gert::FormatOm2Path(gert::OM2_CONSTANTS_CONFIG_PATH_FORMAT, "0", "0"), consts.data(),
                                consts.size(), false));
  gert::GertBuffer buf;
  ASSERT_TRUE(writer.SaveModelData(buf, false));
  ASSERT_NE(buf.data, nullptr);

  GertModelData model_data;
  gert::InitGertModelData(model_data);
  EXPECT_EQ(gert::DeserializeGertModelData(buf.data.get(), buf.length, &model_data), 0U);

  EXPECT_NE(model_data.models[0]->model_meta->has_aipp, 0U);
  ASSERT_EQ(model_data.models[0]->model_meta->aipp_infos.size(), 1U);
  ASSERT_NE(model_data.models[0]->model_meta->aipp_infos[0], nullptr);
  const auto &aipp = *model_data.models[0]->model_meta->aipp_infos[0];
  EXPECT_EQ(aipp.aipp_type, ge::DATA_WITH_STATIC_AIPP);
  EXPECT_EQ(aipp.aipp_data_index, 0U);
  ASSERT_NE(aipp.aipp_config_info, nullptr);
  EXPECT_EQ(aipp.aipp_config_info->src_image_size_w, 224);
  EXPECT_EQ(aipp.aipp_config_info->src_image_size_h, 224);
  ASSERT_EQ(aipp.aipp_input_dims.size(), 1U);
  ASSERT_NE(aipp.aipp_input_dims[0], nullptr);
  EXPECT_EQ(aipp.aipp_input_dims[0]->name, "data");
  EXPECT_EQ(aipp.aipp_input_dims[0]->dim_num, 4U);
  ASSERT_EQ(aipp.aipp_input_dims[0]->dims.size(), 4U);
  EXPECT_EQ(aipp.aipp_input_dims[0]->dims[0], 1);
  EXPECT_EQ(aipp.aipp_input_dims[0]->dims[3], 224);
  ASSERT_EQ(aipp.aipp_output_dims.size(), 1U);
  ASSERT_NE(aipp.aipp_output_dims[0], nullptr);
  EXPECT_EQ(aipp.aipp_output_dims[0]->dim_num, 3U);
  ASSERT_EQ(aipp.aipp_output_dims[0]->dims.size(), 3U);
}

// ====== 文件类别掩码模式（DeserializeGertModelData 带 GertDeserializeFiles）======

// 构造包含全部已知类别的最小 OM2 归档，用于掩码模式测试
void BuildOm2ArchiveForFileList(const std::string &path, gert::GertBuffer &buf) {
  gert::ZipArchiveWriter writer(path);
  ASSERT_TRUE(writer.IsMemFileOpened());
  const std::string manifest =
      R"({"compatibility":{"compiler_version":"1.0","required_executor_version":"","used_features":{}},"model_num":1})";
  ASSERT_TRUE(writer.WriteBytes(gert::OM2_MANIFEST_PATH, manifest.data(), manifest.size(), false));
  const std::string meta = R"({
    "inputs": [{"name": "x", "index": 0, "shape": [1, 2], "data_type": "DT_FLOAT", "format": "ND",
                "size": 8, "shape_range": []}],
    "outputs": [{"name": "y", "index": 0, "shape": [1, 2], "data_type": "DT_FLOAT", "format": "ND",
                 "size": 8, "shape_range": []}],
    "work_size": 8192, "zero_copy_size": 0, "name": "list_model"
  })";
  ASSERT_TRUE(
      writer.WriteBytes(gert::FormatOm2Path(gert::OM2_MODEL_META_PATH_FORMAT, "0"), meta.data(), meta.size(), false));
  const std::string fake_so(32, 'S');
  ASSERT_TRUE(writer.WriteBytes(gert::FormatOm2Path(gert::OM2_RUNTIME_DIR_FORMAT, "0") + "liblist_om2.so",
                                fake_so.data(), fake_so.size(), false));
  const std::string op_attr = R"({"attr": 1})";
  ASSERT_TRUE(writer.WriteBytes(gert::FormatOm2Path(gert::OM2_OP_ATTR_PATH_FORMAT, "0"), op_attr.data(), op_attr.size(),
                                false));
  const std::string consts = R"({"internal_weight_size": 4, "consts": {}})";
  ASSERT_TRUE(writer.WriteBytes(gert::FormatOm2Path(gert::OM2_CONSTANTS_CONFIG_PATH_FORMAT, "0", "0"), consts.data(),
                                consts.size(), false));
  const uint8_t weight_data[] = {0xAA, 0xBB, 0xCC, 0xDD};
  ASSERT_TRUE(writer.WriteBytes(std::string(gert::OM2_CONSTANTS_DIR) + gert::OM2_CONSTANTS_FILE_PREFIX + "0",
                                weight_data, sizeof(weight_data), false));
  const uint8_t kernel_0[] = {0x00};
  ASSERT_TRUE(writer.WriteBytes(std::string(gert::OM2_KERNELS_DIR) + "kernel_0.o", kernel_0, sizeof(kernel_0), false));
  const uint8_t kernel_1[] = {0x01};
  ASSERT_TRUE(writer.WriteBytes(std::string(gert::OM2_KERNELS_DIR) + "kernel_1.o", kernel_1, sizeof(kernel_1), false));
  const std::string visual_json = R"({"format":"ge_visual_json","format_version":1})";
  ASSERT_TRUE(writer.WriteBytes(gert::FormatOm2Path(gert::OM2_VISUAL_JSON_PATH_FORMAT, "0"), visual_json.data(),
                                visual_json.size(), false));
  ASSERT_TRUE(writer.SaveModelData(buf, false));
  ASSERT_NE(buf.data, nullptr);
}

// 掩码模式：仅选 kModelMeta，只填充 model_meta，其余字段不填充
TEST_F(Om2ModelDataTest, DeserializeWithFiles_ModelMetaOnly) {
  const std::string path = "ut_gert_model_data_files_meta.om2";
  gert::GertBuffer buf;
  BuildOm2ArchiveForFileList(path, buf);

  GertModelData model_data;
  gert::InitGertModelData(model_data);
  EXPECT_EQ(
      gert::DeserializeGertModelData(buf.data.get(), buf.length, &model_data, gert::GertDeserializeFiles::kModelMeta),
      0U);

  ASSERT_NE(model_data.models[0]->model_meta, nullptr);
  EXPECT_EQ(std::string(gert::GertGetStr(model_data.models[0]->model_meta->model_name)), "list_model");
  EXPECT_EQ(model_data.models[0]->model_meta->work_size, 8192U);
  EXPECT_EQ(model_data.models[0]->model_meta->input_desc.size(), 1U);
  EXPECT_EQ(model_data.models[0]->model_meta->output_desc.size(), 1U);

  EXPECT_EQ(std::string(gert::GertGetStr(model_data.models[0]->runtime->so_artifact.file_name)), "");
  EXPECT_TRUE(model_data.kernels->binaries.empty());
  EXPECT_EQ(model_data.models[0]->op_attr_json, nullptr);
  EXPECT_TRUE(model_data.models[0]->constants_config->consts.empty());
  EXPECT_EQ(model_data.constants->constants_data[0], nullptr);
  EXPECT_EQ(model_data.models[0]->debug->visual_json, nullptr);
}

// 掩码模式：选 kKernelBinaries，加载 kernels 目录下全部 kernel binary
TEST_F(Om2ModelDataTest, DeserializeWithFiles_KernelBinaries) {
  const std::string path = "ut_gert_model_data_files_kernel.om2";
  gert::GertBuffer buf;
  BuildOm2ArchiveForFileList(path, buf);

  GertModelData model_data;
  gert::InitGertModelData(model_data);
  EXPECT_EQ(gert::DeserializeGertModelData(buf.data.get(), buf.length, &model_data,
                                           gert::GertDeserializeFiles::kKernelBinaries),
            0U);

  ASSERT_EQ(model_data.kernels->binaries.size(), 2U);
  EXPECT_EQ(std::string(gert::GertGetStr(model_data.kernels->binaries[0]->name)), "kernel_0.o");
  EXPECT_EQ(model_data.kernels->binaries[0]->data_size, 1U);
  EXPECT_EQ(model_data.kernels->binaries[0]->data.get()[0], 0x00);
  EXPECT_EQ(std::string(gert::GertGetStr(model_data.kernels->binaries[1]->name)), "kernel_1.o");
  EXPECT_EQ(model_data.kernels->binaries[1]->data.get()[0], 0x01);
  EXPECT_EQ(std::string(gert::GertGetStr(model_data.models[0]->model_meta->model_name)), "");
}

// 掩码模式：多类别组合（op_attr + constants config + weight）
TEST_F(Om2ModelDataTest, DeserializeWithFiles_MultiFields) {
  const std::string path = "ut_gert_model_data_files_multi.om2";
  gert::GertBuffer buf;
  BuildOm2ArchiveForFileList(path, buf);

  GertModelData model_data;
  gert::InitGertModelData(model_data);
  const auto files = gert::GertDeserializeFiles::kOpAttr | gert::GertDeserializeFiles::kConstantsConfig |
                     gert::GertDeserializeFiles::kWeightData;
  EXPECT_EQ(gert::DeserializeGertModelData(buf.data.get(), buf.length, &model_data, files), 0U);

  EXPECT_NE(model_data.models[0]->op_attr_json, nullptr);
  EXPECT_EQ(std::string(gert::GertGetStr(model_data.models[0]->op_attr_json)), R"({"attr": 1})");
  EXPECT_EQ(model_data.models[0]->constants_config->internal_weight_size, 4U);
  ASSERT_NE(model_data.constants->constants_data[0]->data, nullptr);
  EXPECT_EQ(model_data.constants->constants_data[0]->data.get()[0], 0xAA);
  EXPECT_EQ(model_data.constants->constants_data[0]->data.get()[3], 0xDD);
  EXPECT_TRUE(model_data.kernels->binaries.empty());
  EXPECT_TRUE(model_data.models[0]->model_meta->input_desc.empty());
}

// 掩码模式：选中的可选类别文件在包中缺失时仅告警并返回成功，字段保持为空
TEST_F(Om2ModelDataTest, DeserializeWithFiles_SelectedFileMissing) {
  const std::string path = "ut_gert_model_data_files_missing.om2";
  gert::GertBuffer buf;
  BuildOm2ArchiveForFileList(path, buf);

  GertModelData model_data;
  gert::InitGertModelData(model_data);
  EXPECT_EQ(
      gert::DeserializeGertModelData(buf.data.get(), buf.length, &model_data, gert::GertDeserializeFiles::kVarResource),
      0U);
  EXPECT_EQ(model_data.models[0]->variables_config, nullptr);
}

// 掩码模式：选中的必选类别文件在包中缺失时报错（model_meta.json 不在精简归档中）
TEST_F(Om2ModelDataTest, DeserializeWithFiles_RequiredFileMissing) {
  const std::string path = "ut_gert_model_data_files_reqmissing.om2";
  gert::ZipArchiveWriter writer(path);
  ASSERT_TRUE(writer.IsMemFileOpened());
  const std::string manifest =
      R"({"compatibility":{"compiler_version":"1.0","required_executor_version":"","used_features":{}},"model_num":1})";
  ASSERT_TRUE(writer.WriteBytes(gert::OM2_MANIFEST_PATH, manifest.data(), manifest.size(), false));
  gert::GertBuffer buf;
  ASSERT_TRUE(writer.SaveModelData(buf, false));
  ASSERT_NE(buf.data, nullptr);

  GertModelData model_data;
  gert::InitGertModelData(model_data);
  EXPECT_NE(
      gert::DeserializeGertModelData(buf.data.get(), buf.length, &model_data, gert::GertDeserializeFiles::kModelMeta),
      0U);
}

// 掩码模式：目录扫描类别在包中无匹配文件时容忍（custom_ops 目录不存在）
TEST_F(Om2ModelDataTest, DeserializeWithFiles_EmptyDirTolerated) {
  const std::string path = "ut_gert_model_data_files_emptydir.om2";
  gert::GertBuffer buf;
  BuildOm2ArchiveForFileList(path, buf);

  GertModelData model_data;
  gert::InitGertModelData(model_data);
  EXPECT_EQ(gert::DeserializeGertModelData(buf.data.get(), buf.length, &model_data,
                                           gert::GertDeserializeFiles::kCustomKernels),
            0U);
  EXPECT_TRUE(model_data.custom_ops->binaries.empty());
}

// 掩码模式：kNone（未选择任何文件）报参数错误
TEST_F(Om2ModelDataTest, DeserializeWithFiles_NoneIsInvalid) {
  const std::string path = "ut_gert_model_data_files_none.om2";
  gert::GertBuffer buf;
  BuildOm2ArchiveForFileList(path, buf);

  GertModelData model_data;
  gert::InitGertModelData(model_data);
  EXPECT_NE(gert::DeserializeGertModelData(buf.data.get(), buf.length, &model_data, gert::GertDeserializeFiles::kNone),
            0U);
}

// model_index 越界：单模型归档（model_num=1）传 index=1 报参数错
TEST_F(Om2ModelDataTest, DeserializeModelIndexOutOfRange) {
  const std::string path = "ut_gert_model_data_index_oor.om2";
  gert::GertBuffer buf;
  BuildOm2ArchiveForFileList(path, buf);

  GertModelData model_data;
  gert::InitGertModelData(model_data);
  EXPECT_NE(gert::DeserializeGertModelData(buf.data.get(), buf.length, &model_data,
                                           gert::GertDeserializeFiles::kModelMeta, 1U),
            0U);
}

// model_index 显式传 0 与默认行为一致（单模型语义）
TEST_F(Om2ModelDataTest, DeserializeModelIndexZero) {
  const std::string path = "ut_gert_model_data_index_zero.om2";
  gert::GertBuffer buf;
  BuildOm2ArchiveForFileList(path, buf);

  GertModelData model_data;
  gert::InitGertModelData(model_data);
  EXPECT_EQ(gert::DeserializeGertModelData(buf.data.get(), buf.length, &model_data,
                                           gert::GertDeserializeFiles::kModelMeta, 0U),
            0U);
  EXPECT_EQ(model_data.models.size(), 1U);
}

// visual json：缺省全量模式不反序列化 debug 目录；kVisualJson 选中时填充 debug_info
TEST_F(Om2ModelDataTest, DeserializeWithFiles_VisualJson) {
  const std::string path = "ut_gert_model_data_files_visual.om2";
  gert::GertBuffer buf;
  BuildOm2ArchiveForFileList(path, buf);

  GertModelData full_mode;
  gert::InitGertModelData(full_mode);
  EXPECT_EQ(gert::DeserializeGertModelData(buf.data.get(), buf.length, &full_mode), 0U);
  EXPECT_EQ(full_mode.models[0]->debug->visual_json, nullptr);

  GertModelData selective_mode;
  gert::InitGertModelData(selective_mode);
  EXPECT_EQ(gert::DeserializeGertModelData(buf.data.get(), buf.length, &selective_mode,
                                           gert::GertDeserializeFiles::kVisualJson),
            0U);
  ASSERT_NE(selective_mode.models[0]->debug->visual_json, nullptr);
  EXPECT_EQ(std::string(gert::GertGetStr(selective_mode.models[0]->debug->visual_json)),
            R"({"format":"ge_visual_json","format_version":1})");
}

// 缺省参数 = 全量模式回归：三参调用填充全部类别（除 debug 目录）并通过必选检查
TEST_F(Om2ModelDataTest, DeserializeWithFiles_DefaultIsFullMode) {
  const std::string path = "ut_gert_model_data_files_default.om2";
  gert::GertBuffer buf;
  BuildOm2ArchiveForFileList(path, buf);

  GertModelData model_data;
  gert::InitGertModelData(model_data);
  EXPECT_EQ(gert::DeserializeGertModelData(buf.data.get(), buf.length, &model_data), 0U);

  EXPECT_EQ(std::string(gert::GertGetStr(model_data.models[0]->model_meta->model_name)), "list_model");
  EXPECT_NE(model_data.models[0]->op_attr_json, nullptr);
  EXPECT_NE(model_data.constants->constants_data[0]->data, nullptr);
  EXPECT_EQ(model_data.kernels->binaries.size(), 2U);
  EXPECT_NE(std::string(gert::GertGetStr(model_data.models[0]->runtime->so_artifact.file_name)), "");
  EXPECT_EQ(model_data.models[0]->debug->visual_json, nullptr);
}

}  // namespace
}  // namespace gert
