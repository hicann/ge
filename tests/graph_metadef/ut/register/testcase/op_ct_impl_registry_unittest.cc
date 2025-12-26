/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "register/op_ct_impl_registry.h"
#include <gtest/gtest.h>
#include "exe_graph/runtime/kernel_context.h"
#include "graph/any_value.h"
#include "register/op_ct_impl_registry_api.h"
#include "exe_graph/runtime/exe_res_generation_context.h"
#include "register/op_ct_impl_kernel_registry.h"

namespace gert_test {
namespace {
ge::graphStatus CalcParamKernelFunc(gert::ExeResGenerationContext *context) {
  return ge::GRAPH_SUCCESS;
}
ge::graphStatus GenTaskKernelFunc(const gert::ExeResGenerationContext *context, std::vector<std::vector<uint8_t>> &tasks) {
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus CheckSupportFunc(const gert::OpCheckContext *context, ge::AscendString &result) {
  return ge::GRAPH_SUCCESS;
}
ge::graphStatus OpSelectFormatFunc(const gert::OpCheckContext *context, ge::AscendString &result) {
  return ge::GRAPH_SUCCESS;
}
ge::graphStatus GetOpSpecificInfoFunc(const gert::OpCheckContext *context, ge::AscendString &result) {
  return ge::GRAPH_SUCCESS;
}
}  // namespace
class OpCtImplRegistryUT : public testing::Test {

 protected:
  virtual void TearDown() {
    gert::OpCtImplRegistry::GetInstance().GetAllTypesToImpl().clear();
  }
};

TEST_F(OpCtImplRegistryUT, Register_Success_RegisterAll) {
  auto funcs = gert::OpCtImplRegistry::GetInstance().GetOpImpl("TestFoo");
  ASSERT_EQ(funcs, nullptr);

  IMPL_OP_CT(TestFoo)
      .CalcOpParam(CalcParamKernelFunc)
      .GenerateTask(GenTaskKernelFunc);
  auto impl_num = GetRegisteredOpCtNum();
  EXPECT_EQ(impl_num, 1);
  funcs = gert::OpCtImplRegistry::GetInstance().GetOpImpl("TestFoo");
  ASSERT_NE(funcs, nullptr);
  EXPECT_EQ(funcs->calc_op_param, &CalcParamKernelFunc);
  EXPECT_EQ(funcs->gen_task, &GenTaskKernelFunc);

  IMPL_OP_CT(TestConv2D).CalcOpParam(CalcParamKernelFunc).GenerateTask(GenTaskKernelFunc);
  impl_num = GetRegisteredOpCtNum();
  auto impl_funcs = std::unique_ptr<TypesToCtImpl[]>(new(std::nothrow) TypesToCtImpl[impl_num]);
  EXPECT_EQ(impl_num, 2);
  auto ret = GetOpCtImplFunctions(reinterpret_cast<TypesToCtImpl *>(impl_funcs.get()), impl_num);
  EXPECT_NE(ret, ge::GRAPH_FAILED);
  for (size_t i = 0; i < impl_num; ++i) {
    EXPECT_EQ(impl_funcs[i].funcs.calc_op_param, &CalcParamKernelFunc);
    EXPECT_EQ(impl_funcs[i].funcs.gen_task, &GenTaskKernelFunc);
  }
}

TEST_F(OpCtImplRegistryUT, Register_Ct_version_test) {
  auto funcs = gert::OpCtImplRegistry::GetInstance().GetOpImpl("TestFoo");
  ASSERT_EQ(funcs, nullptr);

  IMPL_OP_CT(TestFoo)
  .CalcOpParam(CalcParamKernelFunc)
      .GenerateTask(GenTaskKernelFunc);
  auto impl_num = GetRegisteredOpCtNum();
  EXPECT_EQ(impl_num, 1);
  funcs = gert::OpCtImplRegistry::GetInstance().GetOpImpl("TestFoo");
  ASSERT_NE(funcs, nullptr);
  EXPECT_EQ(funcs->calc_op_param, &CalcParamKernelFunc);
  EXPECT_EQ(funcs->gen_task, &GenTaskKernelFunc);

  IMPL_OP_CT(TestConv2D).CalcOpParam(CalcParamKernelFunc).GenerateTask(GenTaskKernelFunc);
  impl_num = GetRegisteredOpCtNum();
  size_t real_size = sizeof(gert::OpCtImplKernelRegistry::OpCtImplFunctions) + 8;
  size_t offset = real_size + sizeof(char*);
  auto mem_ptr = std::unique_ptr<uint8_t[]>(new(std::nothrow) uint8_t [impl_num * offset]);
  EXPECT_EQ(impl_num, 2);
  for (size_t i = 0; i < impl_num; ++i) {
    auto tmp_impl = reinterpret_cast<TypesToCtImpl *>(mem_ptr.get() + offset * i);
    tmp_impl->funcs.version = 2;
    tmp_impl->funcs.st_size = real_size;
  }
  auto ret = GetOpCtImplFunctions(reinterpret_cast<TypesToCtImpl *>(mem_ptr.get()), impl_num);
  EXPECT_NE(ret, ge::GRAPH_FAILED);
  for (size_t i = 0; i < impl_num; ++i) {
    auto tmp_impl = reinterpret_cast<TypesToCtImpl *>(mem_ptr.get() + offset * i);
    EXPECT_EQ(tmp_impl->funcs.version, 1);
    EXPECT_EQ(tmp_impl->funcs.st_size, sizeof(gert::OpCtImplKernelRegistry::OpCtImplFunctions));
    EXPECT_EQ(tmp_impl->funcs.calc_op_param, &CalcParamKernelFunc);
    EXPECT_EQ(tmp_impl->funcs.gen_task, &GenTaskKernelFunc);
  }
}

TEST_F(OpCtImplRegistryUT, register_all_success) {
  auto funcs = gert::OpCtImplRegistry::GetInstance().GetOpImpl("TestFoo");
  ASSERT_EQ(funcs, nullptr);

  IMPL_OP_CT(TestFoo)
      .CheckSupport(CheckSupportFunc)
      .OpSelectFormat(OpSelectFormatFunc)
      .GetOpSpecificInfo(GetOpSpecificInfoFunc);
  auto impl_num = GetRegisteredOpCtNum();
  EXPECT_EQ(impl_num, 1);
  funcs = gert::OpCtImplRegistry::GetInstance().GetOpImpl("TestFoo");
  ASSERT_NE(funcs, nullptr);
  EXPECT_EQ(funcs->check_support, &CheckSupportFunc);
  EXPECT_EQ(funcs->op_select_format, &OpSelectFormatFunc);
  EXPECT_EQ(funcs->get_op_specific_info, &GetOpSpecificInfoFunc);

  IMPL_OP_CT(TestConv2D)
      .CheckSupport(CheckSupportFunc)
      .OpSelectFormat(OpSelectFormatFunc)
      .GetOpSpecificInfo(GetOpSpecificInfoFunc);
  impl_num = GetRegisteredOpCtNum();
  auto impl_funcs = std::unique_ptr<TypesToCtImpl[]>(new(std::nothrow) TypesToCtImpl[impl_num]);
  EXPECT_EQ(impl_num, 2);
  auto ret = GetOpCtImplFunctions(reinterpret_cast<TypesToCtImpl *>(impl_funcs.get()), impl_num);
  EXPECT_NE(ret, ge::GRAPH_FAILED);
  for (size_t i = 0; i < impl_num; ++i) {
    EXPECT_EQ(impl_funcs[i].funcs.check_support, &CheckSupportFunc);
    EXPECT_EQ(impl_funcs[i].funcs.op_select_format, &OpSelectFormatFunc);
    EXPECT_EQ(impl_funcs[i].funcs.get_op_specific_info, &GetOpSpecificInfoFunc);
  }
}

}  // namespace gert_test
