/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <fstream>
#include <vector>
#include <string>
#include <gtest/gtest.h>

#include "codegen.h"
#include "optimize.h"
#include "backend_common.h"
#include "ascir_ops.h"
#include "platform_context.h"
#include "runtime_stub.h"

class TestBackendConcatE2e : public testing::Test {
 protected:
  void SetUp() override {
    dlog_setlevel(ASCGEN_MODULE_NAME, DLOG_ERROR, 0);
    unsetenv("AUTOFUSE_ENABLE_VF");
    ge::PlatformContext::GetInstance().Reset();
    auto stub_v2 = std::make_shared<ge::RuntimeStubV2>();
    ge::RuntimeStub::SetInstance(stub_v2);
  }
  void TearDown() override {
    dlog_setlevel(ASCGEN_MODULE_NAME, DLOG_ERROR, 0);
    ge::RuntimeStub::Reset();
  }

  static void CreateConcatAscGraph(ge::AscGraph &graph,
                                   const std::vector<std::string> &dim_sizes,
                                   ge::DataType dtype = ge::DT_INT32) {
    ge::ascir_op::Data x0("concat_data0", graph);
    x0.ir_attr.SetIndex(0);
    x0.y.dtype = dtype;

    ge::ascir_op::Data x1("concat_data1", graph);
    x1.ir_attr.SetIndex(1);
    x1.y.dtype = dtype;

    ge::ascir_op::Load load_0("load0");
    load_0.x = x0.y;
    load_0.y.dtype = dtype;

    ge::ascir_op::Load load_1("load1");
    load_1.x = x1.y;
    load_1.y.dtype = dtype;

    ge::ascir_op::Neg neg_0("neg_0");
    ge::ascir_op::Neg neg_1("neg_1");

    ge::ascir_op::Concat concat("concat");
    if (dtype == ge::DT_INT32) {
      neg_0.x = load_0.y;
      neg_0.y.dtype = dtype;
      neg_1.x = load_1.y;
      neg_1.y.dtype = dtype;
      concat.x = {neg_0.y, neg_1.y};
    } else {
      concat.x = {load_0.y, load_1.y};
    }
    concat.y.dtype = dtype;

    ge::ascir_op::Store x_out("store");
    x_out.x = concat.y;
    x_out.y.dtype = dtype;

    ge::ascir_op::Output y("output");
    y.x = x_out.y;
    y.ir_attr.SetIndex(0);
    y.y.dtype = dtype;

    ConstructVVAscGraphAxisInfo(graph, dim_sizes);
    auto concat_node = graph.FindNode("concat");
    auto size = concat_node->attr.sched.axis.size();
    auto repeats = concat_node->outputs()[0]->attr.repeats;
    repeats[size - 1] = repeats[size - 1] + repeats[size - 1];
    auto strides = concat_node->outputs()[0]->attr.strides;
    for (int i = dim_sizes.size() - 2; i >= 0; i--) {
      strides[i] = ge::sym::Mul(repeats[i + 1], strides[i + 1]);
    }
    concat_node->outputs()[0]->attr.strides = strides;
    concat_node->outputs()[0]->attr.repeats = repeats;
    auto store_node = graph.FindNode("store");
    store_node->outputs()[0]->attr.strides = strides;
    store_node->outputs()[0]->attr.repeats = repeats;
  }

  static void ConstructVVAscGraphAxisInfo(ge::AscGraph &graph, const std::vector<std::string> &dim_sizes) {
    std::vector<int64_t> axis;
    std::vector<ge::Expression> repeats;
    std::vector<ge::Expression> strides;
    auto ONE = ge::Symbol(1);

    // 构造符号、轴信息
    for (size_t i = 0; i < dim_sizes.size(); i++) {
      ge::Expression sym_s;
      if (dim_sizes[i][0] == 's') {
        sym_s = graph.CreateSizeVar(dim_sizes[i]);
      } else {
        sym_s = ge::Symbol(std::atoi(dim_sizes[i].c_str()));
      }
      std::string sym_str = "s" + std::to_string(i);
      std::string axis_str = "z" + std::to_string(i);
      auto aixs_z = graph.CreateAxis(axis_str.c_str(), sym_s);
      axis.push_back(aixs_z.id);
      repeats.push_back(sym_s);
      strides.push_back(ONE);
    }
    // 计算每个轴的stride
    for (int32_t i = static_cast<int32_t>(dim_sizes.size()) - 2; i >= 0; i--) {
      strides[i] = ge::sym::Mul(repeats[i + 1], strides[i + 1]);
    }
    // 将原始轴信息设置到图中所有节点上
    for (auto node : graph.GetAllNodes()) {
      node->attr.sched.axis = axis;
      for (auto output_attr : node->outputs()) {
        output_attr->attr.axis = axis;
        output_attr->attr.repeats = repeats;
        output_attr->attr.strides = strides;
      }
    }
  }

  static void CreateOneAxisConcatAscGraph(ge::AscGraph &graph, const std::vector<ge::Expression> &dim_sizes,
                                          ge::DataType dtype = ge::DT_INT32) {
    auto Zero = ge::Symbol(0);
    auto One = ge::Symbol(1);
    auto s0 = One;
    auto s1_1 = dim_sizes[0];
    auto s1_2 = dim_sizes[1];

    auto z0 = graph.CreateAxis("z0", s0);
    auto z1 = graph.CreateAxis("z1", (s1_1 + s1_1 + s1_2));

    ge::ascir_op::Data x0("concat_data0", graph);
    x0.ir_attr.SetIndex(0);
    x0.y.dtype = dtype;

    ge::ascir_op::Data x1("concat_data1", graph);
    x1.ir_attr.SetIndex(1);
    x1.y.dtype = dtype;

    ge::ascir_op::Load load_op("load");
    ge::ascir_op::Load load_op_1("load_1");
    ge::ascir_op::Load load_op2("load2");
    ge::ascir_op::Concat concat_op("concat");

    load_op.x = x0.y;
    load_op.attr.sched.axis = {z0.id, z1.id};
    *load_op.y.axis = {z0.id, z1.id};
    *load_op.y.repeats = {s0, s1_1};
    *load_op.y.strides = {Zero, One};
    load_op.y.dtype = dtype;

    load_op_1.x = x0.y;
    load_op_1.attr.sched.axis = {z0.id, z1.id};
    *load_op_1.y.axis = {z0.id, z1.id};
    *load_op_1.y.repeats = {s0, s1_1};
    *load_op_1.y.strides = {Zero, One};
    load_op_1.y.dtype = dtype;

    load_op2.x = x1.y;
    load_op2.attr.sched.axis = {z0.id, z1.id};
    *load_op2.y.axis = {z0.id, z1.id};
    *load_op2.y.repeats = {s0, s1_2};
    *load_op2.y.strides = {Zero, One};
    load_op2.y.dtype = dtype;

    // concat_op.x1 = load_op.y;
    // concat_op.x2 = load_op2.y;
    concat_op.x = {load_op.y, load_op_1.y, load_op2.y};
    concat_op.attr.sched.axis = {z0.id, z1.id};
    *concat_op.y.axis = {z0.id, z1.id};
    *concat_op.y.repeats = {s0, s1_1 + s1_1 + s1_2};
    *concat_op.y.strides = {Zero, One};
    concat_op.y.dtype = dtype;

    ge::ascir_op::Store x_out("store");
    x_out.x = concat_op.y;
    x_out.y.dtype = dtype;
    x_out.attr.sched.axis = {z0.id, z1.id};
    *x_out.y.axis = {z0.id, z1.id};
    *x_out.y.repeats = {s0, s1_1 + s1_1 + s1_2};
    *x_out.y.strides = {Zero, One};

    ge::ascir_op::Output y("output");
    y.x = x_out.y;
    y.ir_attr.SetIndex(0);
    y.y.dtype = dtype;
  }
};

TEST_F(TestBackendConcatE2e, ConcatNotAllAligned) {
  bool gen_success = true;
  std::string tilig_stub = R"(
#define REGISTER_TILING_DEFAULT(tiling)
#define GET_TILING_DATA(t, tiling)  AutofuseTilingData t = *(AutofuseTilingData*)tiling;
)";

  ge::AscGraph graph("concat_v2_test");
  CreateConcatAscGraph(graph, {"s0", "s1"});
  std::map<std::string, std::string> shape_info(
      {{"s0", "stub_s0"}, {"s1", "stub_s1"}}
  );
  std::cout << "KERNEL_SRC_LIST=" << KERNEL_SRC_LIST << std::endl;
  std::vector<std::string> parts = splitString(KERNEL_SRC_LIST, ':');
  std::string kernel_src_file_name = parts[0];      // add_abs_test_tiling.cpp
  std::string tiling_src_file_name = parts[1];      // add_abs_test_kernel.cpp
  std::string tiling_data_src_file_name = parts[2]; // autofuse_tiling_data.h

  try {
    optimize::Optimizer optimizer(optimize::OptimizerOptions{});
    codegen::Codegen codegen(codegen::CodegenOptions{});

    std::fstream kernel_file(kernel_src_file_name, std::ios::out);
    std::fstream tiling_file(tiling_src_file_name, std::ios::out);
    std::fstream tiling_data_file(tiling_data_src_file_name, std::ios::out);

    std::vector<::ascir::ScheduledResult> schedule_results;
    ascir::FusedScheduledResult fused_schedule_result;
    fused_schedule_result.node_idx_to_scheduled_results.push_back(schedule_results);
    EXPECT_EQ(optimizer.Optimize(graph, fused_schedule_result), 0);
    codegen::CodegenResult result;
    EXPECT_EQ(codegen.Generate(shape_info, fused_schedule_result, result), 0);
    kernel_file << tilig_stub << result.kernel;
    tiling_file << result.tiling;
    tiling_data_file << result.tiling_data;
  }
  catch (...) {
    gen_success = false;
  }

  EXPECT_EQ(gen_success, true);
}

TEST_F(TestBackendConcatE2e, ConcatNotAllAligned_B64) {
  bool gen_success = true;
  ge::AscGraph graph("concat_v2_test");
  CreateConcatAscGraph(graph, {"s0", "s1"}, ge::DT_INT64);
  std::map<std::string, std::string> shape_info(
      {{"s0", "stub_s0"}, {"s1", "stub_s1"}}
  );
  std::cout << "KERNEL_SRC_LIST=" << KERNEL_SRC_LIST << std::endl;
  std::vector<std::string> parts = splitString(KERNEL_SRC_LIST, ':');

  try {
    optimize::Optimizer optimizer(optimize::OptimizerOptions{});
    codegen::Codegen codegen(codegen::CodegenOptions{});

    std::vector<::ascir::ScheduledResult> schedule_results;
    ascir::FusedScheduledResult fused_schedule_result;
    fused_schedule_result.node_idx_to_scheduled_results.push_back(schedule_results);
    EXPECT_EQ(optimizer.Optimize(graph, fused_schedule_result), 0);
    codegen::CodegenResult result;
    EXPECT_EQ(codegen.Generate(shape_info, fused_schedule_result, result), 0);
    const auto &kernel = result.kernel;

    std::string expected = "const concat::ConcatTiling<2> concat_tiling {\n"
                           "  .num_rows = static_cast<uint32_t>(z0t_actual_size),\n"
                           "  .num_dst_cols = (ConvertToUint32((4 * t->s1))),\n"
                           "  .num_srcs_cols = {(ConvertToUint32((2 * t->s1))), (ConvertToUint32((2 * t->s1))), },\n"
                           "};\n";
    std::cout << kernel << std::endl;
    std::cout << expected << std::endl;
    EXPECT_TRUE(kernel.find(expected) != std::string::npos);
    expected = "concat::ConcatExtend<uint32_t, 2>((uint32_t *)";
    EXPECT_TRUE(kernel.find(expected) != std::string::npos);
  }
  catch (...) {
    gen_success = false;
  }

  EXPECT_EQ(gen_success, true);
}

TEST_F(TestBackendConcatE2e, ConcatNotAllAligned_B8) {
  bool gen_success = true;
  ge::AscGraph graph("concat_v2_test");
  CreateConcatAscGraph(graph, {"s0", "s1"}, ge::DT_INT8);
  std::map<std::string, std::string> shape_info(
      {{"s0", "stub_s0"}, {"s1", "stub_s1"}}
  );
  std::cout << "KERNEL_SRC_LIST=" << KERNEL_SRC_LIST << std::endl;
  std::vector<std::string> parts = splitString(KERNEL_SRC_LIST, ':');

  try {
    optimize::Optimizer optimizer(optimize::OptimizerOptions{});
    codegen::Codegen codegen(codegen::CodegenOptions{});

    std::vector<::ascir::ScheduledResult> schedule_results;
    ascir::FusedScheduledResult fused_schedule_result;
    fused_schedule_result.node_idx_to_scheduled_results.push_back(schedule_results);
    EXPECT_EQ(optimizer.Optimize(graph, fused_schedule_result), 0);
    codegen::CodegenResult result;
    EXPECT_EQ(codegen.Generate(shape_info, fused_schedule_result, result), 0);
    const auto &kernel = result.kernel;
    std::string expected = "const concat::ConcatTiling<2> concat_tiling {\n"
                           "  .num_rows = static_cast<uint32_t>(z0t_actual_size),\n"
                           "  .num_dst_cols = (ConvertToUint32((2 * t->s1))),\n"
                           "  .num_srcs_cols = {(ConvertToUint32(t->s1)), (ConvertToUint32(t->s1)), },\n"
                           "};\n";
    EXPECT_TRUE(kernel.find(expected) != std::string::npos);
    expected = "concat::ConcatExtend<int8_t, 2>((int8_t *)";
    EXPECT_TRUE(kernel.find(expected) != std::string::npos);
  }
  catch (...) {
    gen_success = false;
  }

  EXPECT_EQ(gen_success, true);
}

TEST_F(TestBackendConcatE2e, ConcatNotAllAligned_B8ToB16) {
  bool gen_success = true;
  ge::AscGraph graph("concat_v2_test");
  CreateConcatAscGraph(graph, {"s0", "14"}, ge::DT_INT8);
  std::map<std::string, std::string> shape_info(
      {{"s0", "stub_s0"}}
  );
  std::cout << "KERNEL_SRC_LIST=" << KERNEL_SRC_LIST << std::endl;
  std::vector<std::string> parts = splitString(KERNEL_SRC_LIST, ':');

  try {
    optimize::Optimizer optimizer(optimize::OptimizerOptions{});
    codegen::Codegen codegen(codegen::CodegenOptions{});

    std::vector<::ascir::ScheduledResult> schedule_results;
    ascir::FusedScheduledResult fused_schedule_result;
    fused_schedule_result.node_idx_to_scheduled_results.push_back(schedule_results);
    EXPECT_EQ(optimizer.Optimize(graph, fused_schedule_result), 0);
    codegen::CodegenResult result;
    EXPECT_EQ(codegen.Generate(shape_info, fused_schedule_result, result), 0);
    const auto &kernel = result.kernel;

    std::string expected = "const concat::ConcatTiling<2> concat_tiling {\n"
                           "  .num_rows = static_cast<uint32_t>(z0t_actual_size),\n"
                           "  .num_dst_cols = 14,\n"
                           "  .num_srcs_cols = {7, 7, },\n"
                           "};\n";
    EXPECT_TRUE(kernel.find(expected) != std::string::npos);
    expected = "concat::ConcatExtend<uint16_t, 2>((uint16_t *)";
    EXPECT_TRUE(kernel.find(expected) != std::string::npos);
  }
  catch (...) {
    gen_success = false;
  }

  EXPECT_EQ(gen_success, true);
}

TEST_F(TestBackendConcatE2e, ConcatAllAligned) {
  bool gen_success = true;
  ge::AscGraph graph("concat_v2_test");
  CreateConcatAscGraph(graph, {"s0", "32"}, ge::DT_INT8);
  std::map<std::string, std::string> shape_info(
      {{"s0", "stub_s0"}}
  );
  std::cout << "KERNEL_SRC_LIST=" << KERNEL_SRC_LIST << std::endl;
  std::vector<std::string> parts = splitString(KERNEL_SRC_LIST, ':');

  try {
    optimize::Optimizer optimizer(optimize::OptimizerOptions{});
    codegen::Codegen codegen(codegen::CodegenOptions{});

    std::vector<::ascir::ScheduledResult> schedule_results;
    ascir::FusedScheduledResult fused_schedule_result;
    fused_schedule_result.node_idx_to_scheduled_results.push_back(schedule_results);
    EXPECT_EQ(optimizer.Optimize(graph, fused_schedule_result), 0);
    codegen::CodegenResult result;
    EXPECT_EQ(codegen.Generate(shape_info, fused_schedule_result, result), 0);
    const auto &kernel = result.kernel;
    std::string expected = "constexpr ConcatTilingAllAligned<2> concat_tiling {\n"
                           "  .dst_col_size = 64,\n"
                           "  .src_col_sizes = { 32, 32, },\n"
                           "  .dst_offsets = { 0, 32, },\n"
                           "};\n";
    EXPECT_TRUE(kernel.find(expected) != std::string::npos);
    expected = "ConcatAllAligned<int8_t, 2>(";
    EXPECT_TRUE(kernel.find(expected) != std::string::npos);
  }
  catch (...) {
    gen_success = false;
  }

  EXPECT_EQ(gen_success, true);
}

TEST_F(TestBackendConcatE2e, ConcatOneAxis) {
  bool gen_success = true;
  ge::AscGraph graph("concat_one_axis_test");
  CreateOneAxisConcatAscGraph(graph, {ge::Symbol(1), ge::Symbol(2)}, ge::DT_FLOAT);
  std::map<std::string, std::string> shape_info(
      {{"s0", "stub_s0"}}
  );
  std::cout << "KERNEL_SRC_LIST=" << KERNEL_SRC_LIST << std::endl;
  std::vector<std::string> parts = splitString(KERNEL_SRC_LIST, ':');

  try {
    optimize::Optimizer optimizer(optimize::OptimizerOptions{});
    codegen::Codegen codegen(codegen::CodegenOptions{});

    std::vector<::ascir::ScheduledResult> schedule_results;
    ascir::FusedScheduledResult fused_schedule_result;
    fused_schedule_result.node_idx_to_scheduled_results.push_back(schedule_results);
    EXPECT_EQ(optimizer.Optimize(graph, fused_schedule_result), 0);
    codegen::CodegenResult result;
    EXPECT_EQ(codegen.Generate(shape_info, fused_schedule_result, result), 0);
    const auto &kernel = result.kernel;
    std::cout << kernel << std::endl;

    std::string expected = "concat::ConcatOneAxis";
    EXPECT_TRUE(kernel.find(expected) != std::string::npos);
  }
  catch (...) {
    gen_success = false;
  }

  EXPECT_EQ(gen_success, true);
}