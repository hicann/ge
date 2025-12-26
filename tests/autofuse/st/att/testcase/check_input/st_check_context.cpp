/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include "gtest/gtest.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/graph_utils.h"
#include "exe_graph/runtime/tiling_context.h"
#include "exe_graph/lowering/tiling_context_builder.h"
#include "exe_graph/lowering/device_tiling_context_builder.h"
#include "platform/platform_infos_def.h"
#include "check_input/src/context_func.cpp"
#include "kernel_context_holder_builder.h"
using namespace att;
class CheckContextST : public ::testing::Test {
 public:
  static void SetUpTestCase() {
    std::cout << "Test begin." << std::endl;
  }
  static void TearDownTestCase() {
    std::cout << "Test end." << std::endl;
  }

  void SetUp() override {}

  void TearDown() override {}
};

TEST_F(CheckContextST, CheckPreTiling) {
  KernelContextHolderBuilder builder;
  auto holder = builder.AddInput(InOutput(ge::GeShape({1}), ge::FORMAT_ND, ge::DT_INT32))
                    .SetTilingData(10240)
                    .SetWorkSpace(1600)
                    .SetPlatformInfo()
                    .SetCompileInfo(2)
                    .AddPrivateAtt({"test", ge::AnyValue::CreateFrom<int64_t>(10)})
                    .AddPrivateAtt({"head_num", ge::AnyValue::CreateFrom<int64_t>(10)})
                    .Build();
  auto tiling_context = reinterpret_cast<gert::TilingContext *>(holder.context_);
  TilingData tiling_data;
  EXPECT_EQ(PreTiling(tiling_data, tiling_context), true);
  EXPECT_EQ(tiling_data.get_block_dim(), 48u);
  EXPECT_EQ(tiling_data.get_hbm_size(), 10 * 1024 * 1024);
  EXPECT_EQ(tiling_data.get_ub_size(), 192 * 1024);
}

TEST_F(CheckContextST, CheckInputVars) {
  KernelContextHolderBuilder builder1;
  auto holder = builder1.AddInput(InOutput(ge::GeShape({2, 3}), ge::FORMAT_ND, ge::DT_FLOAT16))
                    .AddInput(InOutput(ge::GeShape({5, 6, 7, 8}), ge::FORMAT_ND, ge::DT_FLOAT16))
                    .SetTilingData(10240)
                    .SetWorkSpace(1600)
                    .SetPlatformInfo()
                    .SetCompileInfo(2)
                    .AddPrivateAtt({"test", ge::AnyValue::CreateFrom<int64_t>(10)})
                    .AddPrivateAtt({"head_num", ge::AnyValue::CreateFrom<int64_t>(10)})
                    .Build();
  auto tiling_context1 = reinterpret_cast<gert::TilingContext *>(holder.context_);
  EXPECT_FALSE(TilingInputVarsCheck0(tiling_context1));
  KernelContextHolderBuilder builder2;
  holder = builder2.AddInput(InOutput(ge::GeShape({2, 3, 4, 5, 6, 10, 8}), ge::FORMAT_ND, ge::DT_FLOAT16))
               .AddInput(InOutput(ge::GeShape({2, 3, 4, 5, 6, 7, 8}), ge::FORMAT_ND, ge::DT_FLOAT16))
               .SetTilingData(10240)
               .SetWorkSpace(1600)
               .SetPlatformInfo()
               .SetCompileInfo(2)
               .AddPrivateAtt({"test", ge::AnyValue::CreateFrom<int64_t>(10)})
               .AddPrivateAtt({"head_num", ge::AnyValue::CreateFrom<int64_t>(10)})
               .Build();
  auto tiling_context2 = reinterpret_cast<gert::TilingContext *>(holder.context_);
  EXPECT_TRUE(TilingInputVarsCheck0(tiling_context2));
}

TEST_F(CheckContextST, TestCheckandSet) {
  KernelContextHolderBuilder builder1;
  auto holder = builder1.AddInput(InOutput(ge::GeShape({1}), ge::FORMAT_ND, ge::DT_INT32))
                    .AddInput(InOutput(ge::GeShape({1}), ge::FORMAT_ND, ge::DT_INT32))
                    .SetTilingData(10240)
                    .SetWorkSpace(1600)
                    .SetPlatformInfo()
                    .SetCompileInfo(2)
                    .AddPrivateAtt({"test", ge::AnyValue::CreateFrom<int64_t>(10)})
                    .AddPrivateAtt({"head_num", ge::AnyValue::CreateFrom<int64_t>(2)})
                    .Build();
  TilingData tiling_data;
  auto tiling_context = reinterpret_cast<gert::TilingContext *>(holder.context_);
  EXPECT_FALSE(CheckandSetInput(tiling_data, tiling_context, 0u));

  KernelContextHolderBuilder builder2;
  holder = builder2.AddInput(InOutput(ge::GeShape({11, 1, 2, 3, 4, 5, 6}), ge::FORMAT_ND, ge::DT_FLOAT16))
               .AddInput(InOutput(ge::GeShape({11, 1, 2, 3, 4, 10, 6}), ge::FORMAT_NCHW, ge::DT_FLOAT16))
               .SetTilingData(10240)
               .SetWorkSpace(1600)
               .SetPlatformInfo()
               .SetCompileInfo(2)
               .AddPrivateAtt({"test", ge::AnyValue::CreateFrom<int64_t>(10)})
               .AddPrivateAtt({"head_num", ge::AnyValue::CreateFrom<int64_t>(2)})
               .Build();
  tiling_context = reinterpret_cast<gert::TilingContext *>(holder.context_);
  EXPECT_FALSE(CheckandSetInput(tiling_data, tiling_context, 0u));
  KernelContextHolderBuilder builder3;
  holder = builder3.AddInput(InOutput(ge::GeShape({11, 1, 2, 3, 4, 5, 6}), ge::FORMAT_ND, ge::DT_FLOAT16))
               .AddInput(InOutput(ge::GeShape({11, 1, 2, 3, 4, 10, 6}), ge::FORMAT_ND, ge::DT_INT32))
               .SetTilingData(10240)
               .SetWorkSpace(1600)
               .SetPlatformInfo()
               .SetCompileInfo(2)
               .AddPrivateAtt({"test", ge::AnyValue::CreateFrom<int64_t>(10)})
               .AddPrivateAtt({"head_num", ge::AnyValue::CreateFrom<int64_t>(2)})
               .Build();
  tiling_context = reinterpret_cast<gert::TilingContext *>(holder.context_);
  EXPECT_FALSE(CheckandSetInput(tiling_data, tiling_context, 0u));
  KernelContextHolderBuilder builder4;
  holder = builder4.AddInput(InOutput(ge::GeShape({11, 1, 2, 3, 4, 5, 6}), ge::FORMAT_ND, ge::DT_FLOAT16))
               .AddInput(InOutput(ge::GeShape({11, 1, 2, 3, 4, 10, 6}), ge::FORMAT_ND, ge::DT_FLOAT16))
               .SetTilingData(10240)
               .SetWorkSpace(1600)
               .SetPlatformInfo()
               .SetCompileInfo(2)
               .AddPrivateAtt({"test", ge::AnyValue::CreateFrom<int64_t>(10)})
               .AddPrivateAtt({"head_num", ge::AnyValue::CreateFrom<int64_t>(11)})
               .Build();
  tiling_context = reinterpret_cast<gert::TilingContext *>(holder.context_);
  EXPECT_FALSE(CheckandSetInput(tiling_data, tiling_context, 0u));
  KernelContextHolderBuilder builder5;
  holder = builder5.AddInput(InOutput(ge::GeShape({11, 1, 3, 3, 4, 5, 6}), ge::FORMAT_ND, ge::DT_FLOAT16))
               .AddInput(InOutput(ge::GeShape({11, 1, 2, 3, 4, 10, 6}), ge::FORMAT_ND, ge::DT_FLOAT16))
               .SetTilingData(10240)
               .SetWorkSpace(1600)
               .SetPlatformInfo()
               .SetCompileInfo(2)
               .AddPrivateAtt({"test", ge::AnyValue::CreateFrom<int64_t>(10)})
               .AddPrivateAtt({"head_num", ge::AnyValue::CreateFrom<int64_t>(11)})
               .Build();
  tiling_context = reinterpret_cast<gert::TilingContext *>(holder.context_);
  EXPECT_FALSE(CheckandSetInput(tiling_data, tiling_context, 0u));
  KernelContextHolderBuilder builder6;
  holder = builder6.AddInput(InOutput(ge::GeShape({11, 99999999999, 2, 3, 4, 5, 6}), ge::FORMAT_ND, ge::DT_FLOAT16))
               .AddInput(InOutput(ge::GeShape({11, 99999999999, 2, 3, 4, 10, 6}), ge::FORMAT_ND, ge::DT_FLOAT16))
               .SetTilingData(10240)
               .SetWorkSpace(1600)
               .SetPlatformInfo()
               .SetCompileInfo(2)
               .AddPrivateAtt({"test", ge::AnyValue::CreateFrom<int64_t>(10)})
               .AddPrivateAtt({"head_num", ge::AnyValue::CreateFrom<int64_t>(11)})
               .Build();
  tiling_context = reinterpret_cast<gert::TilingContext *>(holder.context_);
  EXPECT_FALSE(CheckandSetInput(tiling_data, tiling_context, 0u));
  KernelContextHolderBuilder builder7;
  holder = builder7.AddInput(InOutput(ge::GeShape({11,1,2,3,4,5,6}), ge::FORMAT_ND, ge::DT_FLOAT16))
               .AddInput(InOutput(ge::GeShape({11,1,2,3,4,10,6}), ge::FORMAT_ND, ge::DT_FLOAT16))
               .SetTilingData(10240)
               .SetWorkSpace(1600)
               .SetPlatformInfo()
               .SetCompileInfo(2)
               .AddPrivateAtt({"test", ge::AnyValue::CreateFrom<int64_t>(10)})
               .AddPrivateAtt({"head_num", ge::AnyValue::CreateFrom<int64_t>(2)})
               .Build();
  tiling_context = reinterpret_cast<gert::TilingContext *>(holder.context_);
  EXPECT_TRUE(CheckandSetInput(tiling_data, tiling_context, 0u));
  EXPECT_EQ(tiling_data.get_B(), 2);
  EXPECT_EQ(tiling_data.get_D(), 6);
  EXPECT_EQ(tiling_data.get_G(), 4);
  EXPECT_EQ(tiling_data.get_N(), 3);
  EXPECT_EQ(tiling_data.get_S1(), 5);
  EXPECT_EQ(tiling_data.get_S2(), 10);
}

TEST_F(CheckContextST, TestSetWorkspace) {
  KernelContextHolderBuilder builder1;
  auto holder = builder1.AddInput(InOutput(ge::GeShape({1}), ge::FORMAT_ND, ge::DT_INT32))
                    .AddInput(InOutput(ge::GeShape({1}), ge::FORMAT_ND, ge::DT_INT32))
                    .SetTilingData(10240)
                    .SetWorkSpace(1600)
                    .SetPlatformInfo()
                    .SetCompileInfo(2)
                    .AddPrivateAtt({"test", ge::AnyValue::CreateFrom<int64_t>(10)})
                    .AddPrivateAtt({"head_num", ge::AnyValue::CreateFrom<int64_t>(2)})
                    .Build();
  TilingData tiling_data;
  auto tiling_context = reinterpret_cast<gert::TilingContext *>(holder.context_);
  tiling_data.set_B(4);
  tiling_data.set_N(1024);
  tiling_data.set_G(10);
  EXPECT_EQ(GetWorkSpaceSize(tiling_data), true);
  EXPECT_EQ(tiling_data.get_workspaceSize(), 4 * 1024 * 10 * 16);
  
  EXPECT_EQ(PostTiling(tiling_data, tiling_context), true);
  EXPECT_EQ(tiling_context->GetWorkspaceSizes(1)[0], 4 * 1024 * 10 * 16 + 16 * 1024 * 1024);
}