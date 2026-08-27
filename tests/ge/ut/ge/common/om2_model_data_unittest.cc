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

#include "common/om2/om2_model_data.h"

namespace gert {
namespace {

class Om2ModelDataTest : public testing::Test {
 protected:
  void SetUp() override {}
  void TearDown() override {}
};

// Test default construction
TEST_F(Om2ModelDataTest, DefaultConstruction) {
  Om2ModelData model_data;

  EXPECT_TRUE(model_data.program_body.source_artifacts.empty());
  EXPECT_TRUE(model_data.program_body.so_artifact.file_name.empty());
  EXPECT_TRUE(model_data.program_body.so_artifact.data.empty());

  // Verify default values for model_meta
  EXPECT_TRUE(model_data.model_meta.model_name.empty());
  EXPECT_EQ(model_data.model_meta.work_size, 0U);
  EXPECT_EQ(model_data.model_meta.zero_copy_size, 0);
  EXPECT_TRUE(model_data.model_meta.input_desc.empty());
  EXPECT_TRUE(model_data.model_meta.output_desc.empty());
  EXPECT_TRUE(model_data.model_meta.input_desc_v2.empty());
  EXPECT_TRUE(model_data.model_meta.output_desc_v2.empty());
  EXPECT_TRUE(model_data.model_meta.dynamic_batch_info.empty());
  EXPECT_EQ(model_data.model_meta.dynamic_type, 0);
  EXPECT_TRUE(model_data.model_meta.dynamic_output_shape.empty());
  EXPECT_TRUE(model_data.model_meta.user_designate_shape_order.empty());
  EXPECT_TRUE(model_data.model_meta.origin_input_dims.empty());

  EXPECT_EQ(model_data.constants_data.internal_weight_size, 0U);
  EXPECT_TRUE(model_data.constants_data.consts.empty());

  EXPECT_EQ(model_data.constants_data.weight_data, nullptr);
  EXPECT_TRUE(model_data.kernel_binaries.empty());

  // Verify debug_info
  EXPECT_TRUE(model_data.debug_info.op_attr_json.empty());
  EXPECT_TRUE(model_data.debug_info.visual_json.empty());
}

// Test populating codegen output
TEST_F(Om2ModelDataTest, PopulateCodegenOutput) {
  Om2ModelData model_data;

  // Add source artifacts
  ge::Om2CodegenArtifact artifact1;
  artifact1.file_name = "model.cpp";
  artifact1.data = "int main() { return 0; }";
  model_data.program_body.source_artifacts.push_back(artifact1);

  ge::Om2CodegenArtifact artifact2;
  artifact2.file_name = "model.h";
  artifact2.data = "#pragma once\nvoid func();";
  model_data.program_body.source_artifacts.push_back(artifact2);

  model_data.program_body.so_artifact.file_name = "libmodel.so";
  model_data.program_body.so_artifact.data = "binary data";

  EXPECT_EQ(model_data.program_body.source_artifacts.size(), 2U);
  EXPECT_EQ(model_data.program_body.source_artifacts[0].file_name, "model.cpp");
  EXPECT_EQ(model_data.program_body.source_artifacts[1].file_name, "model.h");
  EXPECT_EQ(model_data.program_body.so_artifact.file_name, "libmodel.so");
}

// Test populating model metadata
TEST_F(Om2ModelDataTest, PopulateModelMeta) {
  Om2ModelData model_data;

  model_data.model_meta.model_name = "test_model";
  model_data.model_meta.work_size = 1024 * 1024;

  // Add input descriptors
  ge::Om2TensorDesc input_desc;
  input_desc.SetName("input");
  input_desc.SetDataType(ge::DT_FLOAT);
  input_desc.SetShape({1, 3, 224, 224});
  model_data.model_meta.input_desc.push_back(input_desc);

  // Add output descriptors
  ge::Om2TensorDesc output_desc;
  output_desc.SetName("output");
  output_desc.SetDataType(ge::DT_FLOAT);
  output_desc.SetShape({1, 1000});
  model_data.model_meta.output_desc.push_back(output_desc);

  // Add dynamic batch info
  model_data.model_meta.dynamic_batch_info = {{1}, {2}, {4}, {8}};
  model_data.model_meta.dynamic_type = 1;

  // Add dynamic output shape
  model_data.model_meta.dynamic_output_shape = {"1,1000"};

  // Add origin input dims
  model_data.model_meta.origin_input_dims = {{1, 3, 224, 224}};

  // Verify
  EXPECT_EQ(model_data.model_meta.model_name, "test_model");
  EXPECT_EQ(model_data.model_meta.work_size, 1024 * 1024);
  EXPECT_EQ(model_data.model_meta.input_desc.size(), 1U);
  EXPECT_EQ(model_data.model_meta.input_desc[0].GetName(), "input");
  EXPECT_EQ(model_data.model_meta.output_desc.size(), 1U);
  EXPECT_EQ(model_data.model_meta.output_desc[0].GetName(), "output");
  EXPECT_EQ(model_data.model_meta.dynamic_batch_info.size(), 4U);
  EXPECT_EQ(model_data.model_meta.dynamic_type, 1);
}

// Test populating constants config
TEST_F(Om2ModelDataTest, PopulateConstantsConfig) {
  Om2ModelData model_data;

  model_data.constants_data.internal_weight_size = 2048;

  ge::Om2ConstMeta const1;
  const1.index = 0;
  const1.type = "weight";
  const1.file_name = "weight0.bin";
  const1.offset = 0;
  const1.size = 1024;
  model_data.constants_data.consts.push_back(const1);

  ge::Om2ConstMeta const2;
  const2.index = 1;
  const2.type = "bias";
  const2.file_name = "bias0.bin";
  const2.offset = 1024;
  const2.size = 1024;
  model_data.constants_data.consts.push_back(const2);

  EXPECT_EQ(model_data.constants_data.internal_weight_size, 2048);
  EXPECT_EQ(model_data.constants_data.consts.size(), 2U);
  EXPECT_EQ(model_data.constants_data.consts[0].size, 1024);
  EXPECT_EQ(model_data.constants_data.consts[1].size, 1024);
}

// Test populating weight data
TEST_F(Om2ModelDataTest, PopulateWeightData) {
  Om2ModelData model_data;

  auto buf = std::make_unique<uint8_t[]>(5);
  buf[0] = 0x01;
  buf[1] = 0x02;
  buf[2] = 0x03;
  buf[3] = 0x04;
  buf[4] = 0x05;
  model_data.constants_data.weight_data = ge::ReadonlyByteBuffer(buf.release(), ge::ConditionalDeleter{true});
  model_data.constants_data.internal_weight_size = 5U;

  EXPECT_NE(model_data.constants_data.weight_data, nullptr);
  EXPECT_EQ(model_data.constants_data.weight_data.get()[0], 0x01);
  EXPECT_EQ(model_data.constants_data.weight_data.get()[4], 0x05);
}

// Test populating kernel binaries
TEST_F(Om2ModelDataTest, PopulateKernelBinaries) {
  Om2ModelData model_data;

  Om2KernelBinary kernel1;
  kernel1.name = "kernel_add";
  auto buf1 = std::make_unique<uint8_t[]>(3);
  buf1[0] = 0x10;
  buf1[1] = 0x20;
  buf1[2] = 0x30;
  kernel1.data = ge::ReadonlyByteBuffer(buf1.release(), ge::ConditionalDeleter{true});
  kernel1.data_size = 3U;
  model_data.kernel_binaries.push_back(std::move(kernel1));

  Om2KernelBinary kernel2;
  kernel2.name = "kernel_mul";
  auto buf2 = std::make_unique<uint8_t[]>(3);
  buf2[0] = 0x40;
  buf2[1] = 0x50;
  buf2[2] = 0x60;
  kernel2.data = ge::ReadonlyByteBuffer(buf2.release(), ge::ConditionalDeleter{true});
  kernel2.data_size = 3U;
  model_data.kernel_binaries.push_back(std::move(kernel2));

  // Verify
  EXPECT_EQ(model_data.kernel_binaries.size(), 2U);
  EXPECT_EQ(model_data.kernel_binaries[0].name, "kernel_add");
  EXPECT_EQ(model_data.kernel_binaries[0].data_size, 3U);
  EXPECT_EQ(model_data.kernel_binaries[1].name, "kernel_mul");
}

// Test populating debug info
TEST_F(Om2ModelDataTest, PopulateDebugInfo) {
  Om2ModelData model_data;

  model_data.debug_info.visual_json = R"({"format":"ge_visual_json","format_version":1})";
  model_data.debug_info.op_attr_json = R"({"add":{"alpha":{"type":"FLOAT","value":1.0}}})";

  // Verify
  EXPECT_EQ(model_data.debug_info.visual_json, R"({"format":"ge_visual_json","format_version":1})");
  EXPECT_FALSE(model_data.debug_info.op_attr_json.empty());
  EXPECT_NE(model_data.debug_info.op_attr_json.find("add"), std::string::npos);
}

// Test populating manifest
TEST_F(Om2ModelDataTest, PopulateManifest) {
  Om2ModelData model_data;

  model_data.manifest["model_name"] = "test_model";
  model_data.manifest["version"] = "1.0";
  model_data.manifest["framework"] = "onnx";

  // Verify
  ASSERT_EQ(model_data.manifest.size(), 3U);
  EXPECT_EQ(model_data.manifest["model_name"], "test_model");
  EXPECT_EQ(model_data.manifest["version"], "1.0");
  EXPECT_EQ(model_data.manifest["framework"], "onnx");
}

// Test move semantics
TEST_F(Om2ModelDataTest, MoveSemantics) {
  Om2ModelData model_data1;
  model_data1.model_meta.model_name = "test_model";
  auto buf = std::make_unique<uint8_t[]>(3);
  buf[0] = 0x01;
  buf[1] = 0x02;
  buf[2] = 0x03;
  model_data1.constants_data.weight_data = ge::ReadonlyByteBuffer(buf.release(), ge::ConditionalDeleter{true});
  model_data1.constants_data.internal_weight_size = 3U;

  Om2ModelData model_data2 = std::move(model_data1);

  EXPECT_EQ(model_data2.model_meta.model_name, "test_model");
  EXPECT_EQ(model_data2.constants_data.internal_weight_size, 3U);
  EXPECT_EQ(model_data1.constants_data.weight_data, nullptr);
}

TEST_F(Om2ModelDataTest, KernelBinary_DefaultConstruction) {
  Om2KernelBinary kernel;
  EXPECT_EQ(kernel.data, nullptr);
  EXPECT_EQ(kernel.data_size, 0U);
  EXPECT_TRUE(kernel.name.empty());
}

TEST_F(Om2ModelDataTest, KernelBinary_MoveSemantics) {
  Om2KernelBinary kernel1;
  kernel1.name = "kernel_move";
  auto buf = std::make_unique<uint8_t[]>(4);
  buf[0] = 0xAA;
  buf[1] = 0xBB;
  buf[2] = 0xCC;
  buf[3] = 0xDD;
  kernel1.data = ge::ReadonlyByteBuffer(buf.release(), ge::ConditionalDeleter{true});
  kernel1.data_size = 4U;

  Om2KernelBinary kernel2 = std::move(kernel1);
  EXPECT_EQ(kernel1.data, nullptr);
  EXPECT_EQ(kernel2.name, "kernel_move");
  EXPECT_EQ(kernel2.data_size, 4U);
  EXPECT_NE(kernel2.data, nullptr);
  EXPECT_EQ(kernel2.data.get()[0], 0xAA);
  EXPECT_EQ(kernel2.data.get()[3], 0xDD);
}

TEST_F(Om2ModelDataTest, KernelBinary_NonOwningBuffer) {
  uint8_t raw_data[] = {0x10, 0x20, 0x30, 0x40};
  Om2KernelBinary kernel;
  kernel.name = "non_owning_kernel";
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
  Om2ModelData model_data;
  model_data.constants_data.weight_data = ge::ReadonlyByteBuffer(raw_weights, ge::ConditionalDeleter{false});
  model_data.constants_data.internal_weight_size = sizeof(raw_weights);

  EXPECT_NE(model_data.constants_data.weight_data, nullptr);
  EXPECT_EQ(model_data.constants_data.weight_data.get(), raw_weights);
  EXPECT_EQ(model_data.constants_data.internal_weight_size, 6U);
  EXPECT_EQ(model_data.constants_data.weight_data.get()[0], 0x01);
  EXPECT_EQ(model_data.constants_data.weight_data.get()[5], 0x06);
}

TEST_F(Om2ModelDataTest, KernelBinaries_WithNullData) {
  Om2ModelData model_data;

  Om2KernelBinary kernel_with_data;
  kernel_with_data.name = "has_data";
  auto buf = std::make_unique<uint8_t[]>(2);
  buf[0] = 0xFF;
  buf[1] = 0xFE;
  kernel_with_data.data = ge::ReadonlyByteBuffer(buf.release(), ge::ConditionalDeleter{true});
  kernel_with_data.data_size = 2U;
  model_data.kernel_binaries.push_back(std::move(kernel_with_data));

  Om2KernelBinary kernel_without_data;
  kernel_without_data.name = "no_data";
  model_data.kernel_binaries.push_back(std::move(kernel_without_data));

  EXPECT_EQ(model_data.kernel_binaries.size(), 2U);
  EXPECT_NE(model_data.kernel_binaries[0].data, nullptr);
  EXPECT_EQ(model_data.kernel_binaries[0].data_size, 2U);
  EXPECT_EQ(model_data.kernel_binaries[1].data, nullptr);
  EXPECT_EQ(model_data.kernel_binaries[1].data_size, 0U);
}

TEST_F(Om2ModelDataTest, WeightData_ContentVerification) {
  Om2ModelData model_data;
  constexpr size_t kSize = 8U;
  auto buf = std::make_unique<uint8_t[]>(kSize);
  for (size_t i = 0; i < kSize; ++i) {
    buf[i] = static_cast<uint8_t>(i * 0x11);
  }
  model_data.constants_data.weight_data = ge::ReadonlyByteBuffer(buf.release(), ge::ConditionalDeleter{true});
  model_data.constants_data.internal_weight_size = kSize;

  EXPECT_NE(model_data.constants_data.weight_data, nullptr);
  const uint8_t *ptr = model_data.constants_data.weight_data.get();
  for (size_t i = 0; i < kSize; ++i) {
    EXPECT_EQ(ptr[i], static_cast<uint8_t>(i * 0x11));
  }
}

TEST_F(Om2ModelDataTest, ModelData_MoveWithKernelBinaries) {
  Om2ModelData model_data1;
  model_data1.model_meta.model_name = "move_kernels";

  for (int i = 0; i < 3; ++i) {
    Om2KernelBinary kb;
    kb.name = "kernel_" + std::to_string(i);
    auto buf = std::make_unique<uint8_t[]>(2);
    buf[0] = static_cast<uint8_t>(i);
    buf[1] = static_cast<uint8_t>(i + 1);
    kb.data = ge::ReadonlyByteBuffer(buf.release(), ge::ConditionalDeleter{true});
    kb.data_size = 2U;
    model_data1.kernel_binaries.push_back(std::move(kb));
  }

  auto wbuf = std::make_unique<uint8_t[]>(3);
  wbuf[0] = 0xAA;
  wbuf[1] = 0xBB;
  wbuf[2] = 0xCC;
  model_data1.constants_data.weight_data = ge::ReadonlyByteBuffer(wbuf.release(), ge::ConditionalDeleter{true});
  model_data1.constants_data.internal_weight_size = 3U;

  Om2ModelData model_data2 = std::move(model_data1);

  EXPECT_EQ(model_data2.model_meta.model_name, "move_kernels");
  EXPECT_EQ(model_data2.kernel_binaries.size(), 3U);
  EXPECT_EQ(model_data2.kernel_binaries[0].name, "kernel_0");
  EXPECT_EQ(model_data2.kernel_binaries[0].data_size, 2U);
  EXPECT_EQ(model_data2.kernel_binaries[0].data.get()[0], 0x00);
  EXPECT_EQ(model_data2.kernel_binaries[2].name, "kernel_2");
  EXPECT_EQ(model_data2.kernel_binaries[2].data.get()[0], 0x02);
  EXPECT_NE(model_data2.constants_data.weight_data, nullptr);
  EXPECT_EQ(model_data2.constants_data.internal_weight_size, 3U);

  EXPECT_EQ(model_data1.constants_data.weight_data, nullptr);
  EXPECT_TRUE(model_data1.kernel_binaries.empty());
}

}  // namespace
}  // namespace gert
