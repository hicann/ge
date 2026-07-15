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
  EXPECT_TRUE(model_data.model_meta.root_graph_name.empty());
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

  EXPECT_TRUE(model_data.constants_data.weight_data.empty());
  EXPECT_TRUE(model_data.kernel_binaries.empty());

  // Verify debug_info
  EXPECT_TRUE(model_data.debug_info.op_attr_map.empty());
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
  model_data.model_meta.root_graph_name = "root_graph";
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
  EXPECT_EQ(model_data.model_meta.root_graph_name, "root_graph");
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

  std::vector<uint8_t> weights = {0x01, 0x02, 0x03, 0x04, 0x05};
  model_data.constants_data.weight_data = weights;

  EXPECT_EQ(model_data.constants_data.weight_data.size(), 5U);
  EXPECT_EQ(model_data.constants_data.weight_data[0], 0x01);
  EXPECT_EQ(model_data.constants_data.weight_data[4], 0x05);
}

// Test populating kernel binaries
TEST_F(Om2ModelDataTest, PopulateKernelBinaries) {
  Om2ModelData model_data;

  Om2KernelBinary kernel1;
  kernel1.name = "kernel_add";
  kernel1.data = {0x10, 0x20, 0x30};
  model_data.kernel_binaries.push_back(kernel1);

  Om2KernelBinary kernel2;
  kernel2.name = "kernel_mul";
  kernel2.data = {0x40, 0x50, 0x60};
  model_data.kernel_binaries.push_back(kernel2);

  // Verify
  EXPECT_EQ(model_data.kernel_binaries.size(), 2U);
  EXPECT_EQ(model_data.kernel_binaries[0].name, "kernel_add");
  EXPECT_EQ(model_data.kernel_binaries[0].data.size(), 3U);
  EXPECT_EQ(model_data.kernel_binaries[1].name, "kernel_mul");
}

// Test populating debug info
TEST_F(Om2ModelDataTest, PopulateDebugInfo) {
  Om2ModelData model_data;

  model_data.debug_info.visual_json = R"({"format":"ge_visual_json","format_version":1})";
  model_data.debug_info.op_attr_map["add"] = {{"alpha", "1.0"}, {"beta", "1.0"}};

  // Verify
  EXPECT_EQ(model_data.debug_info.visual_json, R"({"format":"ge_visual_json","format_version":1})");
  ASSERT_EQ(model_data.debug_info.op_attr_map.size(), 1U);
  EXPECT_EQ(model_data.debug_info.op_attr_map["add"]["alpha"], "1.0");
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
  model_data1.constants_data.weight_data = {0x01, 0x02, 0x03};

  Om2ModelData model_data2 = std::move(model_data1);

  EXPECT_EQ(model_data2.model_meta.model_name, "test_model");
  EXPECT_EQ(model_data2.constants_data.weight_data.size(), 3U);
  EXPECT_TRUE(model_data1.constants_data.weight_data.empty());
}

}  // namespace
}  // namespace gert
