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
#include "graph/cache_policy/compile_cache_desc.h"
#include "graph/cache_policy/cache_policy.h"
#include "graph/cache_policy/policy_register.h"
#include "graph/cache_policy/cache_state.h"
#include "exe_graph/runtime/shape.h"

namespace ge {
class UtestCompileCacheDescCov : public testing::Test {
 protected:
  void SetUp() {}
  void TearDown() {}
};

TEST_F(UtestCompileCacheDescCov, BinaryHolderMoveAssign_WithData) {
  uint8_t data = 42;
  BinaryHolder h1(&data, 1);
  BinaryHolder h2;
  h2 = std::move(h1);
  EXPECT_NE(h2.GetDataPtr(), nullptr);
  EXPECT_EQ(h2.GetDataLen(), 1U);
  EXPECT_EQ(*h2.GetDataPtr(), 42);
}

TEST_F(UtestCompileCacheDescCov, BinaryHolderMoveAssign_SelfAssign) {
  uint8_t data = 42;
  BinaryHolder h1(&data, 1);
  BinaryHolder &ref = h1;
  h1 = std::move(ref);
}

TEST_F(UtestCompileCacheDescCov, BinaryHolderCopyAssign_WithData) {
  uint8_t data = 99;
  BinaryHolder h1(&data, 1);
  BinaryHolder h2;
  h2 = h1;
  EXPECT_NE(h2.GetDataPtr(), nullptr);
  EXPECT_EQ(h2.GetDataLen(), 1U);
  EXPECT_EQ(*h2.GetDataPtr(), 99);
  EXPECT_NE(h1.GetDataPtr(), h2.GetDataPtr());
}

TEST_F(UtestCompileCacheDescCov, BinaryHolderCopyAssign_FromEmpty) {
  BinaryHolder h1;
  BinaryHolder h2;
  h2 = h1;
  EXPECT_EQ(h2.GetDataPtr(), nullptr);
  EXPECT_EQ(h2.GetDataLen(), 0UL);
}

TEST_F(UtestCompileCacheDescCov, BinaryHolderCopyAssign_OverwriteExisting) {
  uint8_t d1 = 1;
  uint8_t d2 = 2;
  BinaryHolder h1(&d1, 1);
  BinaryHolder h2(&d2, 1);
  h2 = h1;
  EXPECT_EQ(*h2.GetDataPtr(), 1);
}

TEST_F(UtestCompileCacheDescCov, BinaryHolderCopyCtor_WithData) {
  uint8_t data = 77;
  BinaryHolder h1(&data, 1);
  BinaryHolder h2(h1);
  EXPECT_NE(h2.GetDataPtr(), nullptr);
  EXPECT_EQ(h2.GetDataLen(), 1U);
  EXPECT_EQ(*h2.GetDataPtr(), 77);
  EXPECT_NE(h1.GetDataPtr(), h2.GetDataPtr());
}

TEST_F(UtestCompileCacheDescCov, BinaryHolderCopyCtor_Empty) {
  BinaryHolder h1;
  BinaryHolder h2(h1);
  EXPECT_EQ(h2.GetDataPtr(), nullptr);
  EXPECT_EQ(h2.GetDataLen(), 0UL);
}

TEST_F(UtestCompileCacheDescCov, BinaryHolderMoveCtor_WithData) {
  uint8_t data = 55;
  BinaryHolder h1(&data, 1);
  BinaryHolder h2(std::move(h1));
  EXPECT_EQ(h1.GetDataPtr(), nullptr);
  EXPECT_EQ(h1.GetDataLen(), 0U);
  EXPECT_NE(h2.GetDataPtr(), nullptr);
  EXPECT_EQ(h2.GetDataLen(), 1U);
  EXPECT_EQ(*h2.GetDataPtr(), 55);
}

TEST_F(UtestCompileCacheDescCov, BinaryHolderNeq_DifferentLength) {
  uint8_t d1[2] = {1, 2};
  uint8_t d2[3] = {1, 2, 3};
  BinaryHolder h1(d1, 2);
  BinaryHolder h2(d2, 3);
  EXPECT_TRUE(h1 != h2);
}

TEST_F(UtestCompileCacheDescCov, BinaryHolderNeq_OneNull) {
  uint8_t data = 1;
  BinaryHolder h1(&data, 1);
  BinaryHolder h2;
  EXPECT_TRUE(h1 != h2);
  EXPECT_TRUE(h2 != h1);
}

TEST_F(UtestCompileCacheDescCov, BinaryHolderNeq_BothNull) {
  BinaryHolder h1;
  BinaryHolder h2;
  EXPECT_FALSE(h1 != h2);
}

TEST_F(UtestCompileCacheDescCov, BinaryHolderNeq_DifferentValue) {
  uint8_t d1[2] = {1, 2};
  uint8_t d2[2] = {1, 3};
  BinaryHolder h1(d1, 2);
  BinaryHolder h2(d2, 2);
  EXPECT_TRUE(h1 != h2);
}

TEST_F(UtestCompileCacheDescCov, BinaryHolderNeq_Equal) {
  uint8_t d1[2] = {1, 2};
  uint8_t d2[2] = {1, 2};
  BinaryHolder h1(d1, 2);
  BinaryHolder h2(d2, 2);
  EXPECT_FALSE(h1 != h2);
}

TEST_F(UtestCompileCacheDescCov, BinaryHolderCreateFrom_ValidData) {
  auto ptr = std::unique_ptr<uint8_t[]>(new uint8_t[4]{1, 2, 3, 4});
  auto holder = BinaryHolder::createFrom(std::move(ptr), 4);
  ASSERT_NE(holder, nullptr);
  EXPECT_NE(holder->GetDataPtr(), nullptr);
  EXPECT_EQ(holder->GetDataLen(), 4U);
  EXPECT_EQ(holder->GetDataPtr()[0], 1);
  EXPECT_EQ(holder->GetDataPtr()[3], 4);
  EXPECT_EQ(ptr, nullptr);
}

TEST_F(UtestCompileCacheDescCov, TensorInfoMatch_FormatMismatch) {
  TensorInfoArgs t1(FORMAT_ND, FORMAT_ND, DT_FLOAT16);
  TensorInfoArgs t2(FORMAT_NCHW, FORMAT_ND, DT_FLOAT16);
  EXPECT_FALSE(t1.IsTensorInfoMatch(t2));
}

TEST_F(UtestCompileCacheDescCov, TensorInfoMatch_OriginFormatMismatch) {
  TensorInfoArgs t1(FORMAT_ND, FORMAT_ND, DT_FLOAT16);
  TensorInfoArgs t2(FORMAT_ND, FORMAT_NCHW, DT_FLOAT16);
  EXPECT_FALSE(t1.IsTensorInfoMatch(t2));
}

TEST_F(UtestCompileCacheDescCov, TensorInfoMatch_DtypeMismatch) {
  TensorInfoArgs t1(FORMAT_ND, FORMAT_ND, DT_FLOAT16);
  TensorInfoArgs t2(FORMAT_ND, FORMAT_ND, DT_FLOAT);
  EXPECT_FALSE(t1.IsTensorInfoMatch(t2));
}

TEST_F(UtestCompileCacheDescCov, TensorInfoMatch_SuccessExactShape) {
  TensorInfoArgs t1(FORMAT_ND, FORMAT_ND, DT_FLOAT16);
  std::vector<int64_t> shape{1, 2};
  t1.SetShape(shape);
  t1.SetOriginShape(shape);
  TensorInfoArgs t2(FORMAT_ND, FORMAT_ND, DT_FLOAT16);
  t2.SetShape(shape);
  t2.SetOriginShape(shape);
  EXPECT_TRUE(t1.IsTensorInfoMatch(t2));
}

TEST_F(UtestCompileCacheDescCov, TensorInfoMatch_SuccessDynamicShape) {
  TensorInfoArgs t1(FORMAT_ND, FORMAT_ND, DT_FLOAT16);
  std::vector<int64_t> shape{-1, -1};
  t1.SetShape(shape);
  t1.SetOriginShape(shape);
  std::vector<std::pair<int64_t, int64_t>> ranges{{1, 10}, {1, 10}};
  t1.SetShapeRange(ranges);
  TensorInfoArgs t2(FORMAT_ND, FORMAT_ND, DT_FLOAT16);
  std::vector<int64_t> shape2{5, 5};
  t2.SetShape(shape2);
  t2.SetOriginShape(shape2);
  EXPECT_TRUE(t1.IsTensorInfoMatch(t2));
}

TEST_F(UtestCompileCacheDescCov, IsShapeInRange_AllShape) {
  TensorInfoArgs t1(FORMAT_ND, FORMAT_ND, DT_FLOAT16);
  std::vector<int64_t> shape{-2};
  t1.SetShape(shape);
  t1.SetOriginShape(shape);
  TensorInfoArgs t2(FORMAT_ND, FORMAT_ND, DT_FLOAT16);
  std::vector<int64_t> shape2{5, 10};
  t2.SetShape(shape2);
  t2.SetOriginShape(shape2);
  EXPECT_TRUE(t1.IsShapeInRange(t2));
}

TEST_F(UtestCompileCacheDescCov, IsShapeInRange_DifferentRankShape) {
  TensorInfoArgs t1(FORMAT_ND, FORMAT_ND, DT_FLOAT16);
  std::vector<int64_t> shape1{1, 2};
  t1.SetShape(shape1);
  t1.SetOriginShape(shape1);
  TensorInfoArgs t2(FORMAT_ND, FORMAT_ND, DT_FLOAT16);
  std::vector<int64_t> shape2{1, 2, 3};
  t2.SetShape(shape2);
  t2.SetOriginShape(shape2);
  EXPECT_FALSE(t1.IsShapeInRange(t2));
}

TEST_F(UtestCompileCacheDescCov, IsShapeInRange_DifferentRankOriginShape) {
  TensorInfoArgs t1(FORMAT_ND, FORMAT_ND, DT_FLOAT16);
  std::vector<int64_t> shape1{1, 2};
  t1.SetShape(shape1);
  std::vector<int64_t> origin1{1, 2};
  t1.SetOriginShape(origin1);
  TensorInfoArgs t2(FORMAT_ND, FORMAT_ND, DT_FLOAT16);
  std::vector<int64_t> shape2{1, 2};
  t2.SetShape(shape2);
  std::vector<int64_t> origin2{1, 2, 3};
  t2.SetOriginShape(origin2);
  EXPECT_FALSE(t1.IsShapeInRange(t2));
}

TEST_F(UtestCompileCacheDescCov, IsShapeInRange_ShapeSizeMismatchRange) {
  TensorInfoArgs t1(FORMAT_ND, FORMAT_ND, DT_FLOAT16);
  std::vector<int64_t> shape{-1, -1};
  t1.SetShape(shape);
  t1.SetOriginShape(shape);
  std::vector<std::pair<int64_t, int64_t>> ranges{{1, 10}};
  t1.SetShapeRange(ranges);
  TensorInfoArgs t2(FORMAT_ND, FORMAT_ND, DT_FLOAT16);
  std::vector<int64_t> shape2{5, 5};
  t2.SetShape(shape2);
  t2.SetOriginShape(shape2);
  EXPECT_FALSE(t1.IsShapeInRange(t2));
}

TEST_F(UtestCompileCacheDescCov, IsShapeInRange_FirstGreaterThanOther) {
  TensorInfoArgs t1(FORMAT_ND, FORMAT_ND, DT_FLOAT16);
  std::vector<int64_t> shape{-1, -1};
  t1.SetShape(shape);
  t1.SetOriginShape(shape);
  std::vector<std::pair<int64_t, int64_t>> ranges{{10, 20}, {1, 10}};
  t1.SetShapeRange(ranges);
  TensorInfoArgs t2(FORMAT_ND, FORMAT_ND, DT_FLOAT16);
  std::vector<int64_t> shape2{5, 5};
  t2.SetShape(shape2);
  t2.SetOriginShape(shape2);
  EXPECT_FALSE(t1.IsShapeInRange(t2));
}

TEST_F(UtestCompileCacheDescCov, IsShapeInRange_SecondIsUnknownDim) {
  TensorInfoArgs t1(FORMAT_ND, FORMAT_ND, DT_FLOAT16);
  std::vector<int64_t> shape{-1, -1};
  t1.SetShape(shape);
  t1.SetOriginShape(shape);
  std::vector<std::pair<int64_t, int64_t>> ranges{{1, -1}, {1, -1}};
  t1.SetShapeRange(ranges);
  TensorInfoArgs t2(FORMAT_ND, FORMAT_ND, DT_FLOAT16);
  std::vector<int64_t> shape2{5, 100};
  t2.SetShape(shape2);
  t2.SetOriginShape(shape2);
  EXPECT_TRUE(t1.IsShapeInRange(t2));
}

TEST_F(UtestCompileCacheDescCov, IsShapeInRange_SecondSmallerThanOther) {
  TensorInfoArgs t1(FORMAT_ND, FORMAT_ND, DT_FLOAT16);
  std::vector<int64_t> shape{-1, -1};
  t1.SetShape(shape);
  t1.SetOriginShape(shape);
  std::vector<std::pair<int64_t, int64_t>> ranges{{1, 3}, {1, 3}};
  t1.SetShapeRange(ranges);
  TensorInfoArgs t2(FORMAT_ND, FORMAT_ND, DT_FLOAT16);
  std::vector<int64_t> shape2{5, 5};
  t2.SetShape(shape2);
  t2.SetOriginShape(shape2);
  EXPECT_FALSE(t1.IsShapeInRange(t2));
}

TEST_F(UtestCompileCacheDescCov, IsShapeInRange_ExactShapeMatched) {
  TensorInfoArgs t1(FORMAT_ND, FORMAT_ND, DT_FLOAT16);
  std::vector<int64_t> shape1{1, 2};
  t1.SetShape(shape1);
  t1.SetOriginShape(shape1);
  TensorInfoArgs t2(FORMAT_ND, FORMAT_ND, DT_FLOAT16);
  std::vector<int64_t> shape2{1, 2};
  t2.SetShape(shape2);
  t2.SetOriginShape(shape2);
  EXPECT_TRUE(t1.IsShapeInRange(t2));
}

TEST_F(UtestCompileCacheDescCov, IsShapeInRange_ExactShapeNotMatched) {
  TensorInfoArgs t1(FORMAT_ND, FORMAT_ND, DT_FLOAT16);
  std::vector<int64_t> shape1{1, 2};
  t1.SetShape(shape1);
  t1.SetOriginShape(shape1);
  TensorInfoArgs t2(FORMAT_ND, FORMAT_ND, DT_FLOAT16);
  std::vector<int64_t> shape2{3, 4};
  t2.SetShape(shape2);
  t2.SetOriginShape(shape2);
  EXPECT_FALSE(t1.IsShapeInRange(t2));
}

TEST_F(UtestCompileCacheDescCov, IsShapeInRange_ExactShapeOriginNotMatched) {
  TensorInfoArgs t1(FORMAT_ND, FORMAT_ND, DT_FLOAT16);
  std::vector<int64_t> shape1{1, 2};
  t1.SetShape(shape1);
  std::vector<int64_t> origin1{1, 2};
  t1.SetOriginShape(origin1);
  TensorInfoArgs t2(FORMAT_ND, FORMAT_ND, DT_FLOAT16);
  std::vector<int64_t> shape2{1, 2};
  t2.SetShape(shape2);
  std::vector<int64_t> origin2{3, 4};
  t2.SetOriginShape(origin2);
  EXPECT_FALSE(t1.IsShapeInRange(t2));
}

TEST_F(UtestCompileCacheDescCov, TensorInfoNeq_DifferentFormat) {
  TensorInfoArgs t1(FORMAT_ND, FORMAT_ND, DT_FLOAT16);
  TensorInfoArgs t2(FORMAT_NCHW, FORMAT_ND, DT_FLOAT16);
  EXPECT_TRUE(t1 != t2);
}

TEST_F(UtestCompileCacheDescCov, TensorInfoNeq_DifferentOriginFormat) {
  TensorInfoArgs t1(FORMAT_ND, FORMAT_ND, DT_FLOAT16);
  TensorInfoArgs t2(FORMAT_ND, FORMAT_NCHW, DT_FLOAT16);
  EXPECT_TRUE(t1 != t2);
}

TEST_F(UtestCompileCacheDescCov, TensorInfoNeq_DifferentDtype) {
  TensorInfoArgs t1(FORMAT_ND, FORMAT_ND, DT_FLOAT16);
  TensorInfoArgs t2(FORMAT_ND, FORMAT_ND, DT_FLOAT);
  EXPECT_TRUE(t1 != t2);
}

TEST_F(UtestCompileCacheDescCov, TensorInfoNeq_DifferentShape) {
  TensorInfoArgs t1(FORMAT_ND, FORMAT_ND, DT_FLOAT16);
  std::vector<int64_t> s1{1, 2};
  t1.SetShape(s1);
  t1.SetOriginShape(s1);
  TensorInfoArgs t2(FORMAT_ND, FORMAT_ND, DT_FLOAT16);
  std::vector<int64_t> s2{3, 4};
  t2.SetShape(s2);
  t2.SetOriginShape(s2);
  EXPECT_TRUE(t1 != t2);
}

TEST_F(UtestCompileCacheDescCov, TensorInfoNeq_DifferentShapeRange) {
  TensorInfoArgs t1(FORMAT_ND, FORMAT_ND, DT_FLOAT16);
  std::vector<int64_t> shape{-1, -1};
  t1.SetShape(shape);
  t1.SetOriginShape(shape);
  std::vector<std::pair<int64_t, int64_t>> r1{{1, 10}, {1, 10}};
  t1.SetShapeRange(r1);
  TensorInfoArgs t2(FORMAT_ND, FORMAT_ND, DT_FLOAT16);
  t2.SetShape(shape);
  t2.SetOriginShape(shape);
  std::vector<std::pair<int64_t, int64_t>> r2{{1, 20}, {1, 10}};
  t2.SetShapeRange(r2);
  EXPECT_TRUE(t1 != t2);
}

TEST_F(UtestCompileCacheDescCov, TensorInfoNeq_Equal) {
  TensorInfoArgs t1(FORMAT_ND, FORMAT_ND, DT_FLOAT16);
  std::vector<int64_t> shape{-1, -1};
  t1.SetShape(shape);
  t1.SetOriginShape(shape);
  std::vector<std::pair<int64_t, int64_t>> ranges{{1, 10}, {1, 10}};
  t1.SetShapeRange(ranges);
  TensorInfoArgs t2(FORMAT_ND, FORMAT_ND, DT_FLOAT16);
  t2.SetShape(shape);
  t2.SetOriginShape(shape);
  t2.SetShapeRange(ranges);
  EXPECT_FALSE(t1 != t2);
}

TEST_F(UtestCompileCacheDescCov, TensorInfoIsUnknownShape_True) {
  TensorInfoArgs t1(FORMAT_ND, FORMAT_ND, DT_FLOAT16);
  std::vector<int64_t> shape{-1, 2};
  t1.SetShape(shape);
  EXPECT_TRUE(t1.IsUnknownShape());
}

TEST_F(UtestCompileCacheDescCov, TensorInfoIsUnknownShape_False) {
  TensorInfoArgs t1(FORMAT_ND, FORMAT_ND, DT_FLOAT16);
  std::vector<int64_t> shape{1, 2};
  t1.SetShape(shape);
  EXPECT_FALSE(t1.IsUnknownShape());
}

TEST_F(UtestCompileCacheDescCov, TensorInfoGetters) {
  TensorInfoArgs t1(FORMAT_NCHW, FORMAT_NHWC, DT_INT32);
  EXPECT_EQ(t1.GetFormat(), FORMAT_NCHW);
  EXPECT_EQ(t1.GetOriginFormat(), FORMAT_NHWC);
  EXPECT_EQ(t1.GetDataType(), DT_INT32);
}

TEST_F(UtestCompileCacheDescCov, TensorInfoSetShape_SmallVector) {
  TensorInfoArgs t1(FORMAT_ND, FORMAT_ND, DT_FLOAT16);
  SmallVector<int64_t, kDefaultDimsNum> shape{1, 2, 3};
  t1.SetShape(shape);
  EXPECT_EQ(t1.shape_.size(), 3U);
  EXPECT_EQ(t1.shape_[0], 1);
  EXPECT_EQ(t1.shape_[2], 3);
}

TEST_F(UtestCompileCacheDescCov, TensorInfoSetOriginShape_SmallVector) {
  TensorInfoArgs t1(FORMAT_ND, FORMAT_ND, DT_FLOAT16);
  SmallVector<int64_t, kDefaultDimsNum> shape{4, 5, 6};
  t1.SetOriginShape(shape);
  EXPECT_EQ(t1.origin_shape_.size(), 3U);
  EXPECT_EQ(t1.origin_shape_[0], 4);
  EXPECT_EQ(t1.origin_shape_[2], 6);
}

TEST_F(UtestCompileCacheDescCov, CheckWithoutTensorInfo_OpTypeMismatch) {
  CompileCacheDescPtr desc1 = std::make_shared<CompileCacheDesc>();
  desc1->SetOpType("op_a");
  CompileCacheDescPtr desc2 = std::make_shared<CompileCacheDesc>();
  desc2->SetOpType("op_b");
  auto ccp =
      ge::CachePolicy::Create(ge::MatchPolicyType::MATCH_POLICY_EXACT_ONLY, ge::AgingPolicyType::AGING_POLICY_LRU);
  ccp->AddCache(desc1);
  EXPECT_EQ(ccp->FindCache(desc2), KInvalidCacheItemId);
}

TEST_F(UtestCompileCacheDescCov, CheckWithoutTensorInfo_TensorInfoSizeMismatch) {
  CompileCacheDescPtr desc1 = std::make_shared<CompileCacheDesc>();
  desc1->SetOpType("op_a");
  TensorInfoArgs t(FORMAT_ND, FORMAT_ND, DT_FLOAT16);
  desc1->AddTensorInfo(t);
  CompileCacheDescPtr desc2 = std::make_shared<CompileCacheDesc>();
  desc2->SetOpType("op_a");
  auto ccp =
      ge::CachePolicy::Create(ge::MatchPolicyType::MATCH_POLICY_EXACT_ONLY, ge::AgingPolicyType::AGING_POLICY_LRU);
  ccp->AddCache(desc1);
  EXPECT_EQ(ccp->FindCache(desc2), KInvalidCacheItemId);
}

TEST_F(UtestCompileCacheDescCov, CheckWithoutTensorInfo_ScopeIdMismatch) {
  CompileCacheDescPtr desc1 = std::make_shared<CompileCacheDesc>();
  desc1->SetOpType("op_a");
  desc1->SetScopeId({1, 2});
  CompileCacheDescPtr desc2 = std::make_shared<CompileCacheDesc>();
  desc2->SetOpType("op_a");
  desc2->SetScopeId({1, 3});
  auto ccp =
      ge::CachePolicy::Create(ge::MatchPolicyType::MATCH_POLICY_EXACT_ONLY, ge::AgingPolicyType::AGING_POLICY_LRU);
  ccp->AddCache(desc1);
  EXPECT_EQ(ccp->FindCache(desc2), KInvalidCacheItemId);
}

TEST_F(UtestCompileCacheDescCov, CheckWithoutTensorInfo_BinaryCountMismatch) {
  CompileCacheDescPtr desc1 = std::make_shared<CompileCacheDesc>();
  desc1->SetOpType("op_a");
  uint8_t val = 1;
  BinaryHolder h(&val, 1);
  desc1->AddBinary(h);
  CompileCacheDescPtr desc2 = std::make_shared<CompileCacheDesc>();
  desc2->SetOpType("op_a");
  auto ccp =
      ge::CachePolicy::Create(ge::MatchPolicyType::MATCH_POLICY_EXACT_ONLY, ge::AgingPolicyType::AGING_POLICY_LRU);
  ccp->AddCache(desc1);
  EXPECT_EQ(ccp->FindCache(desc2), KInvalidCacheItemId);
}

TEST_F(UtestCompileCacheDescCov, CheckWithoutTensorInfo_BinaryLenMismatch) {
  CompileCacheDescPtr desc1 = std::make_shared<CompileCacheDesc>();
  desc1->SetOpType("op_a");
  uint8_t v1 = 1;
  BinaryHolder h1(&v1, 1);
  desc1->AddBinary(h1);
  auto ccp =
      ge::CachePolicy::Create(ge::MatchPolicyType::MATCH_POLICY_EXACT_ONLY, ge::AgingPolicyType::AGING_POLICY_LRU);
  ccp->AddCache(desc1);

  CompileCacheDescPtr desc2 = std::make_shared<CompileCacheDesc>();
  desc2->SetOpType("op_a");
  uint8_t v2[2] = {1, 2};
  BinaryHolder h2(v2, 2);
  desc2->AddBinary(h2);
  EXPECT_EQ(ccp->FindCache(desc2), KInvalidCacheItemId);
}

TEST_F(UtestCompileCacheDescCov, CheckWithoutTensorInfo_BinaryNullptrCheck) {
  CompileCacheDescPtr desc1 = std::make_shared<CompileCacheDesc>();
  desc1->SetOpType("op_a");
  BinaryHolder empty_holder;
  desc1->AddBinary(empty_holder);
  auto ccp =
      ge::CachePolicy::Create(ge::MatchPolicyType::MATCH_POLICY_EXACT_ONLY, ge::AgingPolicyType::AGING_POLICY_LRU);
  ccp->AddCache(desc1);

  CompileCacheDescPtr desc2 = std::make_shared<CompileCacheDesc>();
  desc2->SetOpType("op_a");
  BinaryHolder empty_holder2;
  desc2->AddBinary(empty_holder2);
  EXPECT_EQ(ccp->FindCache(desc2), KInvalidCacheItemId);
}

TEST_F(UtestCompileCacheDescCov, CheckWithoutTensorInfo_BinaryValueMismatch) {
  CompileCacheDescPtr desc1 = std::make_shared<CompileCacheDesc>();
  desc1->SetOpType("op_a");
  uint8_t v1 = 1;
  BinaryHolder h1(&v1, 1);
  desc1->AddBinary(h1);
  auto ccp =
      ge::CachePolicy::Create(ge::MatchPolicyType::MATCH_POLICY_EXACT_ONLY, ge::AgingPolicyType::AGING_POLICY_LRU);
  ccp->AddCache(desc1);

  CompileCacheDescPtr desc2 = std::make_shared<CompileCacheDesc>();
  desc2->SetOpType("op_a");
  uint8_t v2 = 2;
  BinaryHolder h2(&v2, 1);
  desc2->AddBinary(h2);
  EXPECT_EQ(ccp->FindCache(desc2), KInvalidCacheItemId);
}

TEST_F(UtestCompileCacheDescCov, CheckWithoutTensorInfo_Success) {
  CompileCacheDescPtr desc1 = std::make_shared<CompileCacheDesc>();
  desc1->SetOpType("op_a");
  uint8_t v1 = 1;
  BinaryHolder h1(&v1, 1);
  desc1->AddBinary(h1);
  auto ccp =
      ge::CachePolicy::Create(ge::MatchPolicyType::MATCH_POLICY_EXACT_ONLY, ge::AgingPolicyType::AGING_POLICY_LRU);
  CacheItemId id = ccp->AddCache(desc1);
  ASSERT_NE(id, KInvalidCacheItemId);

  CompileCacheDescPtr desc2 = std::make_shared<CompileCacheDesc>();
  desc2->SetOpType("op_a");
  uint8_t v2 = 1;
  BinaryHolder h2(&v2, 1);
  desc2->AddBinary(h2);
  EXPECT_EQ(ccp->FindCache(desc2), id);
}

TEST_F(UtestCompileCacheDescCov, IsMatch_ShapeNotMatched) {
  CompileCacheDescPtr desc1 = std::make_shared<CompileCacheDesc>();
  desc1->SetOpType("op_a");
  TensorInfoArgs t1(FORMAT_ND, FORMAT_ND, DT_FLOAT16);
  std::vector<int64_t> s1{1, 2};
  t1.SetShape(s1);
  t1.SetOriginShape(s1);
  desc1->AddTensorInfo(t1);
  auto ccp =
      ge::CachePolicy::Create(ge::MatchPolicyType::MATCH_POLICY_EXACT_ONLY, ge::AgingPolicyType::AGING_POLICY_LRU);
  ccp->AddCache(desc1);

  CompileCacheDescPtr desc2 = std::make_shared<CompileCacheDesc>();
  desc2->SetOpType("op_a");
  TensorInfoArgs t2(FORMAT_ND, FORMAT_ND, DT_FLOAT16);
  std::vector<int64_t> s2{3, 4};
  t2.SetShape(s2);
  t2.SetOriginShape(s2);
  desc2->AddTensorInfo(t2);
  EXPECT_EQ(ccp->FindCache(desc2), KInvalidCacheItemId);
}

TEST_F(UtestCompileCacheDescCov, IsMatch_Success) {
  CompileCacheDescPtr desc1 = std::make_shared<CompileCacheDesc>();
  desc1->SetOpType("op_a");
  TensorInfoArgs t1(FORMAT_ND, FORMAT_ND, DT_FLOAT16);
  std::vector<int64_t> s1{1, 2};
  t1.SetShape(s1);
  t1.SetOriginShape(s1);
  desc1->AddTensorInfo(t1);
  auto ccp =
      ge::CachePolicy::Create(ge::MatchPolicyType::MATCH_POLICY_EXACT_ONLY, ge::AgingPolicyType::AGING_POLICY_LRU);
  CacheItemId id = ccp->AddCache(desc1);
  ASSERT_NE(id, KInvalidCacheItemId);

  CompileCacheDescPtr desc2 = std::make_shared<CompileCacheDesc>();
  desc2->SetOpType("op_a");
  TensorInfoArgs t2(FORMAT_ND, FORMAT_ND, DT_FLOAT16);
  t2.SetShape(s1);
  t2.SetOriginShape(s1);
  desc2->AddTensorInfo(t2);
  EXPECT_EQ(ccp->FindCache(desc2), id);
}

TEST_F(UtestCompileCacheDescCov, IsMatch_DynamicShapeInRange) {
  CompileCacheDescPtr desc1 = std::make_shared<CompileCacheDesc>();
  desc1->SetOpType("op_a");
  TensorInfoArgs t1(FORMAT_ND, FORMAT_ND, DT_FLOAT16);
  std::vector<int64_t> s1{-1, -1};
  t1.SetShape(s1);
  t1.SetOriginShape(s1);
  std::vector<std::pair<int64_t, int64_t>> ranges{{1, 10}, {1, 10}};
  t1.SetShapeRange(ranges);
  desc1->AddTensorInfo(t1);
  auto ccp =
      ge::CachePolicy::Create(ge::MatchPolicyType::MATCH_POLICY_EXACT_ONLY, ge::AgingPolicyType::AGING_POLICY_LRU);
  CacheItemId id = ccp->AddCache(desc1);
  ASSERT_NE(id, KInvalidCacheItemId);

  CompileCacheDescPtr desc2 = std::make_shared<CompileCacheDesc>();
  desc2->SetOpType("op_a");
  TensorInfoArgs t2(FORMAT_ND, FORMAT_ND, DT_FLOAT16);
  std::vector<int64_t> s2{5, 5};
  t2.SetShape(s2);
  t2.SetOriginShape(s2);
  desc2->AddTensorInfo(t2);
  EXPECT_EQ(ccp->FindCache(desc2), id);
}

TEST_F(UtestCompileCacheDescCov, IsEqual_TensorInfoNotMatched) {
  CompileCacheDescPtr desc1 = std::make_shared<CompileCacheDesc>();
  desc1->SetOpType("op_a");
  TensorInfoArgs t1(FORMAT_ND, FORMAT_ND, DT_FLOAT16);
  std::vector<int64_t> s1{1, 2};
  t1.SetShape(s1);
  t1.SetOriginShape(s1);
  desc1->AddTensorInfo(t1);
  auto ccp =
      ge::CachePolicy::Create(ge::MatchPolicyType::MATCH_POLICY_EXACT_ONLY, ge::AgingPolicyType::AGING_POLICY_LRU);
  ccp->AddCache(desc1);

  CompileCacheDescPtr desc2 = std::make_shared<CompileCacheDesc>();
  desc2->SetOpType("op_a");
  TensorInfoArgs t2(FORMAT_ND, FORMAT_ND, DT_FLOAT16);
  std::vector<int64_t> s2{1, 3};
  t2.SetShape(s2);
  t2.SetOriginShape(s2);
  desc2->AddTensorInfo(t2);
  EXPECT_NE(ccp->AddCache(desc2), KInvalidCacheItemId);
}

TEST_F(UtestCompileCacheDescCov, IsEqual_Success) {
  CompileCacheDescPtr desc1 = std::make_shared<CompileCacheDesc>();
  desc1->SetOpType("op_a");
  TensorInfoArgs t1(FORMAT_ND, FORMAT_ND, DT_FLOAT16);
  std::vector<int64_t> s1{1, 2};
  t1.SetShape(s1);
  t1.SetOriginShape(s1);
  desc1->AddTensorInfo(t1);
  EXPECT_TRUE(desc1->IsEqual(desc1));
}

TEST_F(UtestCompileCacheDescCov, IsMatch_DirectCall_Success) {
  CompileCacheDescPtr desc1 = std::make_shared<CompileCacheDesc>();
  desc1->SetOpType("op_a");
  TensorInfoArgs t1(FORMAT_ND, FORMAT_ND, DT_FLOAT16);
  std::vector<int64_t> s1{1, 2};
  t1.SetShape(s1);
  t1.SetOriginShape(s1);
  desc1->AddTensorInfo(t1);
  EXPECT_TRUE(desc1->IsMatch(desc1));
}

TEST_F(UtestCompileCacheDescCov, AddBinary_Rvalue) {
  CompileCacheDescPtr desc = std::make_shared<CompileCacheDesc>();
  uint8_t val = 42;
  BinaryHolder holder(&val, 1);
  desc->AddBinary(std::move(holder));
  EXPECT_EQ(desc->other_desc_.size(), 1U);
  EXPECT_EQ(desc->other_desc_[0].GetDataLen(), 1U);
}

TEST_F(UtestCompileCacheDescCov, AddBinary_Lvalue) {
  CompileCacheDescPtr desc = std::make_shared<CompileCacheDesc>();
  uint8_t val = 42;
  BinaryHolder holder(&val, 1);
  desc->AddBinary(holder);
  EXPECT_EQ(desc->other_desc_.size(), 1U);
  EXPECT_EQ(desc->other_desc_[0].GetDataLen(), 1U);
}

TEST_F(UtestCompileCacheDescCov, MutableTensorInfo_OutOfBounds) {
  CompileCacheDescPtr desc = std::make_shared<CompileCacheDesc>();
  desc->SetOpType("op_a");
  EXPECT_EQ(desc->MutableTensorInfo(0), nullptr);
}

TEST_F(UtestCompileCacheDescCov, MutableTensorInfo_Valid) {
  CompileCacheDescPtr desc = std::make_shared<CompileCacheDesc>();
  desc->SetOpType("op_a");
  TensorInfoArgs t(FORMAT_ND, FORMAT_ND, DT_FLOAT16);
  desc->AddTensorInfo(t);
  ASSERT_NE(desc->MutableTensorInfo(0), nullptr);
  EXPECT_EQ(desc->MutableTensorInfo(0)->GetFormat(), FORMAT_ND);
}

TEST_F(UtestCompileCacheDescCov, GetTensorInfoSize) {
  CompileCacheDescPtr desc = std::make_shared<CompileCacheDesc>();
  EXPECT_EQ(desc->GetTensorInfoSize(), 0U);
  TensorInfoArgs t(FORMAT_ND, FORMAT_ND, DT_FLOAT16);
  desc->AddTensorInfo(t);
  EXPECT_EQ(desc->GetTensorInfoSize(), 1U);
}

TEST_F(UtestCompileCacheDescCov, GetCacheDescHash) {
  CompileCacheDescPtr desc = std::make_shared<CompileCacheDesc>();
  desc->SetOpType("op_a");
  TensorInfoArgs t(FORMAT_ND, FORMAT_ND, DT_FLOAT16);
  desc->AddTensorInfo(t);
  CacheHashKey hash1 = desc->GetCacheDescHash();
  desc->SetOpType("op_b");
  CacheHashKey hash2 = desc->GetCacheDescHash();
  EXPECT_NE(hash1, hash2);
}

TEST_F(UtestCompileCacheDescCov, GetCacheDescHash_Empty) {
  CompileCacheDescPtr desc = std::make_shared<CompileCacheDesc>();
  CacheHashKey hash = desc->GetCacheDescHash();
  (void)hash;
}

TEST_F(UtestCompileCacheDescCov, SetScopeId) {
  CompileCacheDescPtr desc = std::make_shared<CompileCacheDesc>();
  desc->SetScopeId({10, 20, 30});
  EXPECT_EQ(desc->scope_id_.size(), 3U);
  EXPECT_EQ(desc->scope_id_[0], 10U);
  EXPECT_EQ(desc->scope_id_[1], 20U);
  EXPECT_EQ(desc->scope_id_[2], 30U);
}

TEST_F(UtestCompileCacheDescCov, TensorInfoSetShape_VectorOverwrite) {
  TensorInfoArgs t(FORMAT_ND, FORMAT_ND, DT_FLOAT16);
  std::vector<int64_t> s1{1, 2, 3};
  t.SetShape(s1);
  std::vector<int64_t> s2{4, 5};
  t.SetShape(s2);
  EXPECT_EQ(t.shape_.size(), 2U);
  EXPECT_EQ(t.shape_[0], 4);
}

TEST_F(UtestCompileCacheDescCov, TensorInfoSetOriginShape_VectorOverwrite) {
  TensorInfoArgs t(FORMAT_ND, FORMAT_ND, DT_FLOAT16);
  std::vector<int64_t> s1{1, 2, 3};
  t.SetOriginShape(s1);
  std::vector<int64_t> s2{4, 5};
  t.SetOriginShape(s2);
  EXPECT_EQ(t.origin_shape_.size(), 2U);
  EXPECT_EQ(t.origin_shape_[0], 4);
}

TEST_F(UtestCompileCacheDescCov, TensorInfoSetShapeRange_Overwrite) {
  TensorInfoArgs t(FORMAT_ND, FORMAT_ND, DT_FLOAT16);
  std::vector<std::pair<int64_t, int64_t>> r1{{1, 10}};
  t.SetShapeRange(r1);
  std::vector<std::pair<int64_t, int64_t>> r2{{1, 20}, {1, 30}};
  t.SetShapeRange(r2);
  EXPECT_EQ(t.shape_range_.size(), 2U);
  EXPECT_EQ(t.shape_range_[1].second, 30);
}

TEST_F(UtestCompileCacheDescCov, BinaryHolderConstructor_NullData) {
  BinaryHolder h(nullptr, 10);
  EXPECT_EQ(h.GetDataPtr(), nullptr);
  EXPECT_EQ(h.GetDataLen(), 0UL);
}

TEST_F(UtestCompileCacheDescCov, BinaryHolderConstructor_ZeroLength) {
  uint8_t data = 42;
  BinaryHolder h(&data, 0);
  EXPECT_EQ(h.GetDataPtr(), nullptr);
  EXPECT_EQ(h.GetDataLen(), 0UL);
}

TEST_F(UtestCompileCacheDescCov, BinaryHolderCreateFrom_NullPtr) {
  std::unique_ptr<uint8_t[]> null_ptr;
  auto holder = BinaryHolder::createFrom(std::move(null_ptr), 10);
  ASSERT_NE(holder, nullptr);
  EXPECT_EQ(holder->GetDataPtr(), nullptr);
  EXPECT_EQ(holder->GetDataLen(), 0UL);
}

TEST_F(UtestCompileCacheDescCov, BinaryHolderCreateFrom_ZeroLength) {
  auto ptr = std::unique_ptr<uint8_t[]>(new uint8_t[4]{1, 2, 3, 4});
  auto holder = BinaryHolder::createFrom(std::move(ptr), 0);
  ASSERT_NE(holder, nullptr);
  EXPECT_EQ(holder->GetDataPtr(), nullptr);
  EXPECT_EQ(holder->GetDataLen(), 0UL);
}

TEST_F(UtestCompileCacheDescCov, BinaryHolderCopyCtor_FromEmpty) {
  BinaryHolder h1;
  BinaryHolder h2(h1);
  EXPECT_EQ(h2.GetDataPtr(), nullptr);
  EXPECT_EQ(h2.GetDataLen(), 0UL);
}

TEST_F(UtestCompileCacheDescCov, BinaryHolderMoveCtor_Empty) {
  BinaryHolder h1;
  BinaryHolder h2(std::move(h1));
  EXPECT_EQ(h2.GetDataPtr(), nullptr);
  EXPECT_EQ(h2.GetDataLen(), 0UL);
}

TEST_F(UtestCompileCacheDescCov, BinaryHolderGetDataPtr_NullHolder) {
  BinaryHolder h;
  EXPECT_EQ(h.GetDataPtr(), nullptr);
  EXPECT_EQ(h.GetDataLen(), 0UL);
}

TEST_F(UtestCompileCacheDescCov, IsMatch_NullDesc) {
  CompileCacheDescPtr desc1 = std::make_shared<CompileCacheDesc>();
  desc1->SetOpType("op_a");
  EXPECT_FALSE(desc1->IsMatch(nullptr));
}

TEST_F(UtestCompileCacheDescCov, IsEqual_NullDesc) {
  CompileCacheDescPtr desc1 = std::make_shared<CompileCacheDesc>();
  desc1->SetOpType("op_a");
  EXPECT_FALSE(desc1->IsEqual(nullptr));
}

TEST_F(UtestCompileCacheDescCov, IsMatch_DirectCall_Fail) {
  CompileCacheDescPtr desc1 = std::make_shared<CompileCacheDesc>();
  desc1->SetOpType("op_a");
  TensorInfoArgs t1(FORMAT_ND, FORMAT_ND, DT_FLOAT16);
  std::vector<int64_t> s1{1, 2};
  t1.SetShape(s1);
  t1.SetOriginShape(s1);
  desc1->AddTensorInfo(t1);

  CompileCacheDescPtr desc2 = std::make_shared<CompileCacheDesc>();
  desc2->SetOpType("op_b");
  TensorInfoArgs t2(FORMAT_ND, FORMAT_ND, DT_FLOAT16);
  t2.SetShape(s1);
  t2.SetOriginShape(s1);
  desc2->AddTensorInfo(t2);
  EXPECT_FALSE(desc1->IsMatch(desc2));
}

TEST_F(UtestCompileCacheDescCov, IsEqual_DirectCall_Fail) {
  CompileCacheDescPtr desc1 = std::make_shared<CompileCacheDesc>();
  desc1->SetOpType("op_a");
  TensorInfoArgs t1(FORMAT_ND, FORMAT_ND, DT_FLOAT16);
  std::vector<int64_t> s1{1, 2};
  t1.SetShape(s1);
  t1.SetOriginShape(s1);
  desc1->AddTensorInfo(t1);

  CompileCacheDescPtr desc2 = std::make_shared<CompileCacheDesc>();
  desc2->SetOpType("op_b");
  TensorInfoArgs t2(FORMAT_ND, FORMAT_ND, DT_FLOAT16);
  t2.SetShape(s1);
  t2.SetOriginShape(s1);
  desc2->AddTensorInfo(t2);
  EXPECT_FALSE(desc1->IsEqual(desc2));
}

TEST_F(UtestCompileCacheDescCov, IsEqual_DirectCall_Success) {
  CompileCacheDescPtr desc1 = std::make_shared<CompileCacheDesc>();
  desc1->SetOpType("op_a");
  TensorInfoArgs t1(FORMAT_ND, FORMAT_ND, DT_FLOAT16);
  std::vector<int64_t> s1{1, 2};
  t1.SetShape(s1);
  t1.SetOriginShape(s1);
  desc1->AddTensorInfo(t1);

  CompileCacheDescPtr desc2 = std::make_shared<CompileCacheDesc>();
  desc2->SetOpType("op_a");
  TensorInfoArgs t2(FORMAT_ND, FORMAT_ND, DT_FLOAT16);
  t2.SetShape(s1);
  t2.SetOriginShape(s1);
  desc2->AddTensorInfo(t2);
  EXPECT_TRUE(desc1->IsEqual(desc2));
}

TEST_F(UtestCompileCacheDescCov, CheckWithoutTensorInfo_OpTypeMatch_BinaryMatch_Success) {
  CompileCacheDescPtr desc1 = std::make_shared<CompileCacheDesc>();
  desc1->SetOpType("op_a");
  uint8_t v1 = 1;
  BinaryHolder h1(&v1, 1);
  desc1->AddBinary(h1);

  CompileCacheDescPtr desc2 = std::make_shared<CompileCacheDesc>();
  desc2->SetOpType("op_a");
  uint8_t v2 = 1;
  BinaryHolder h2(&v2, 1);
  desc2->AddBinary(h2);
  EXPECT_TRUE(desc1->IsMatch(desc2));
}

TEST_F(UtestCompileCacheDescCov, TensorInfoMatch_AllShapeMatch) {
  TensorInfoArgs t1(FORMAT_ND, FORMAT_ND, DT_FLOAT16);
  std::vector<int64_t> shape{-2};
  t1.SetShape(shape);
  t1.SetOriginShape(shape);
  TensorInfoArgs t2(FORMAT_ND, FORMAT_ND, DT_FLOAT16);
  std::vector<int64_t> shape2{1, 2};
  t2.SetShape(shape2);
  t2.SetOriginShape(shape2);
  EXPECT_TRUE(t1.IsTensorInfoMatch(t2));
}

TEST_F(UtestCompileCacheDescCov, TensorInfoNeq_DifferentOriginShape) {
  TensorInfoArgs t1(FORMAT_ND, FORMAT_ND, DT_FLOAT16);
  std::vector<int64_t> s1{1, 2};
  t1.SetShape(s1);
  std::vector<int64_t> o1{1, 2};
  t1.SetOriginShape(o1);
  TensorInfoArgs t2(FORMAT_ND, FORMAT_ND, DT_FLOAT16);
  t2.SetShape(s1);
  std::vector<int64_t> o2{3, 4};
  t2.SetOriginShape(o2);
  EXPECT_TRUE(t1 != t2);
}
}  // namespace ge
