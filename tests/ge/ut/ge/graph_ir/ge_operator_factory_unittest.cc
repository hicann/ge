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

#include "../graph/ops_stub.h"
#include "graph/operator_factory.h"

#include "macro_utils/dt_public_scope.h"
#include "operator_factory_impl.h"
#include "macro_utils/dt_public_unscope.h"

using namespace ge;
class UtestGeOperatorFactory : public testing::Test {
 protected:
  void SetUp() {}

  void TearDown() {}
};

TEST(UtestGeOperatorFactory, create_operator) {
  Operator acosh = OperatorFactory::CreateOperator("acosh", "Acosh");
  EXPECT_EQ("Acosh", acosh.GetOpType());
  EXPECT_EQ("acosh", acosh.GetName());
  EXPECT_EQ(false, acosh.IsEmpty());
}

TEST(UtestGeOperatorFactory, create_operator_nullptr) {
  Operator abc = OperatorFactory::CreateOperator("abc", "ABC");
  EXPECT_EQ(true, abc.IsEmpty());
}

TEST(UtestGeOperatorFactory, get_infer_shape_func) {
  OperatorFactoryImpl::RegisterInferShapeFunc("test", nullptr);
  InferShapeFunc infer_shape_func = OperatorFactoryImpl::GetInferShapeFunc("ABC");
  EXPECT_EQ(nullptr, infer_shape_func);
}

TEST(UtestGeOperatorFactory, get_verify_func) {
  OperatorFactoryImpl::RegisterVerifyFunc("test", nullptr);
  VerifyFunc verify_func = OperatorFactoryImpl::GetVerifyFunc("ABC");
  EXPECT_EQ(nullptr, verify_func);
}

TEST(UtestGeOperatorFactory, get_ops_type_list) {
  std::vector<std::string> all_ops;
  graphStatus status = OperatorFactory::GetOpsTypeList(all_ops);
  EXPECT_NE(0, all_ops.size());
  EXPECT_EQ(GRAPH_SUCCESS, status);
}

TEST(UtestGeOperatorFactory, is_exist_op) {
  graphStatus status = OperatorFactory::IsExistOp("Acosh");
  EXPECT_EQ(true, status);
  status = OperatorFactory::IsExistOp("ABC");
  EXPECT_EQ(false, status);
}

TEST(UtestGeOperatorFactory, register_func) {
  OperatorFactoryImpl::RegisterInferShapeFunc("test", nullptr);
  graphStatus status = OperatorFactoryImpl::RegisterInferShapeFunc("test", nullptr);
  EXPECT_EQ(GRAPH_FAILED, status);
  status = OperatorFactoryImpl::RegisterInferShapeFunc("ABC", nullptr);
  EXPECT_EQ(GRAPH_SUCCESS, status);

  OperatorFactoryImpl::RegisterVerifyFunc("test", nullptr);
  status = OperatorFactoryImpl::RegisterVerifyFunc("test", nullptr);
  EXPECT_EQ(GRAPH_FAILED, status);
  status = OperatorFactoryImpl::RegisterVerifyFunc("ABC", nullptr);
  EXPECT_EQ(GRAPH_SUCCESS, status);
}

TEST(UtestGeOperatorFactory, IncCov_GetInferFuncs) {
  InferFormatFunc fmt_func = OperatorFactoryImpl::GetInferFormatFunc("NonExistent");
  EXPECT_EQ(fmt_func, nullptr);
  InferDataSliceFunc slice_func = OperatorFactoryImpl::GetInferDataSliceFunc("NonExistent");
  EXPECT_EQ(slice_func, nullptr);
  InferAxisSliceFunc axis_slice_func = OperatorFactoryImpl::GetInferAxisSliceFunc("NonExistent");
  EXPECT_EQ(axis_slice_func, nullptr);
  InferAxisTypeInfoFunc axis_type_func = OperatorFactoryImpl::GetInferAxisTypeInfoFunc("NonExistent");
  EXPECT_EQ(axis_type_func, nullptr);
  auto value_range = OperatorFactoryImpl::GetInferValueRangePara("NonExistent");
  InferShapeV2Func shape_v2 = OperatorFactoryImpl::GetInferShapeV2Func();
  InferDataTypeFunc dt_func = OperatorFactoryImpl::GetInferDataTypeFunc();
  InferShapeRangeFunc range_func = OperatorFactoryImpl::GetInferShapeRangeFunc();
  InferFormatV2Func fmt_v2 = OperatorFactoryImpl::GetInferFormatV2Func();
  auto is_fmt_v2 = OperatorFactoryImpl::GetIsInferFormatV2RegisteredFunc();
  auto is_shape_v2 = OperatorFactoryImpl::GetIsInferShapeV2RegisteredFunc();
  auto custom_shape = OperatorFactoryImpl::GetCustomOpInferShapeFunc();
  auto custom_dt = OperatorFactoryImpl::GetCustomOpInferDataTypeFunc();
}

TEST(UtestGeOperatorFactory, IncCov_RegisterAndDuplicate) {
  graphStatus status = OperatorFactoryImpl::RegisterInferFormatFunc("CovFmt", nullptr);
  EXPECT_EQ(status, GRAPH_SUCCESS);
  status = OperatorFactoryImpl::RegisterInferFormatFunc("CovFmt", nullptr);
  EXPECT_EQ(status, GRAPH_FAILED);

  status = OperatorFactoryImpl::RegisterInferDataSliceFunc("CovSlice", nullptr);
  EXPECT_EQ(status, GRAPH_SUCCESS);
  status = OperatorFactoryImpl::RegisterInferDataSliceFunc("CovSlice", nullptr);
  EXPECT_EQ(status, GRAPH_FAILED);

  status = OperatorFactoryImpl::RegisterInferAxisSliceFunc("CovAxisSlice", nullptr);
  EXPECT_EQ(status, GRAPH_SUCCESS);
  status = OperatorFactoryImpl::RegisterInferAxisSliceFunc("CovAxisSlice", nullptr);
  EXPECT_EQ(status, GRAPH_FAILED);

  status = OperatorFactoryImpl::RegisterInferAxisTypeInfoFunc("CovAxisType", nullptr);
  EXPECT_EQ(status, GRAPH_SUCCESS);
  status = OperatorFactoryImpl::RegisterInferAxisTypeInfoFunc("CovAxisType", nullptr);
  EXPECT_EQ(status, GRAPH_FAILED);

  status = OperatorFactoryImpl::RegisterInferValueRangeFunc("CovValueRange");
  EXPECT_EQ(status, GRAPH_SUCCESS);
  status = OperatorFactoryImpl::RegisterInferValueRangeFunc("CovValueRange");
  EXPECT_EQ(status, GRAPH_FAILED);

  status = OperatorFactoryImpl::RegisterOperatorCreator(
      "CovCreator", [](const std::string &name) { return Operator(name, "CovType"); });
  EXPECT_EQ(status, GRAPH_SUCCESS);
  status = OperatorFactoryImpl::RegisterOperatorCreator(
      "CovCreator", [](const std::string &name) { return Operator(name, "CovType"); });
  EXPECT_EQ(status, GRAPH_FAILED);
}

TEST(UtestGeOperatorFactory, IncCov_OperatorCreatorV2) {
  graphStatus status = OperatorFactoryImpl::RegisterOperatorCreator(
      "CovV2", [](const AscendString &name) { return Operator(name.GetString(), "CovV2Type"); });
  EXPECT_EQ(status, GRAPH_SUCCESS);
  status = OperatorFactoryImpl::RegisterOperatorCreator(
      "CovV2", [](const AscendString &name) { return Operator(name.GetString(), "CovV2Type"); });
  EXPECT_EQ(status, GRAPH_FAILED);

  OperatorFactoryImpl::SetRegisterOverridable(true);
  status = OperatorFactoryImpl::RegisterOperatorCreator(
      "CovV2", [](const AscendString &name) { return Operator(name.GetString(), "CovV2Override"); });
  EXPECT_EQ(status, GRAPH_SUCCESS);
  OperatorFactoryImpl::SetRegisterOverridable(false);
}

TEST(UtestGeOperatorFactory, IncCov_RemoveCustomOpCreators) {
  OperatorFactoryImpl::RegisterOperatorCreator(
      "CovRemove", [](const AscendString &name) { return Operator(name.GetString(), "CovRemoveType"); });
  EXPECT_TRUE(OperatorFactory::IsExistOp("CovRemove"));
  OperatorFactoryImpl::RemoveCustomOpCreators({"CovRemove"});
  EXPECT_FALSE(OperatorFactory::IsExistOp("CovRemove"));
}

TEST(UtestGeOperatorFactory, IncCov_RegisterV2Funcs) {
  OperatorFactoryImpl::RegisterInferShapeV2Func(nullptr);
  OperatorFactoryImpl::RegisterInferDataTypeFunc(nullptr);
  OperatorFactoryImpl::RegisterInferShapeRangeFunc(nullptr);
  OperatorFactoryImpl::RegisterInferFormatV2Func(nullptr);
  OperatorFactoryImpl::RegisterIsInferFormatV2RegisteredFunc(nullptr);
  OperatorFactoryImpl::RegisterIsInferShapeV2RegisteredFunc(nullptr);
  OperatorFactoryImpl::RegisterCustomOpInferShapeFunc(nullptr);
  OperatorFactoryImpl::RegisterCustomOpInferDataTypeFunc(nullptr);
}

TEST(UtestGeOperatorFactory, IncCov_GetOpsTypeListFallback) {
  auto v2_temp = OperatorFactoryImpl::operator_creators_v2_;
  OperatorFactoryImpl::operator_creators_v2_ = nullptr;
  std::vector<std::string> all_ops;
  graphStatus status = OperatorFactoryImpl::GetOpsTypeList(all_ops);
  EXPECT_EQ(status, GRAPH_SUCCESS);
  EXPECT_NE(all_ops.size(), 0U);
  OperatorFactoryImpl::operator_creators_v2_ = v2_temp;
}
/*
TEST(UtestGeOperatorFactory, get_ops_type_list_fail) {
  auto operator_creators_temp = OperatorFactoryImpl::operator_creators_;
  OperatorFactoryImpl::operator_creators_ = nullptr;
  std::vector<std::string> all_ops;
  graphStatus status = OperatorFactoryImpl::GetOpsTypeList(all_ops);
  EXPECT_EQ(GRAPH_FAILED, status);
  OperatorFactoryImpl::operator_creators_ = operator_creators_temp;
}
*/
