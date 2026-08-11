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

#include "common/fp16_t/fp16_t.h"

namespace ge {
namespace formats {
class UtestFP16 : public testing::Test {
 protected:
  void SetUp() {}
  void TearDown() {}
};

TEST_F(UtestFP16, fp16_to_other) {
  fp16_t test;
  float num = test.ToFloat();
  EXPECT_EQ(num, 0.0);

  double num2 = test.ToDouble();
  EXPECT_EQ(num2, 0);

  int16_t num3 = test.ToInt16();
  EXPECT_EQ(num3, 0);

  int32_t num4 = test.ToInt32();
  EXPECT_EQ(num4, 0);

  int8_t num5 = test.ToInt8();
  EXPECT_EQ(num5, 0);

  uint16_t num6 = test.ToUInt16();
  EXPECT_EQ(num6, 0);

  uint32_t num7 = test.ToUInt16();
  EXPECT_EQ(num7, 0);

  uint8_t num8 = test.ToUInt8();
  EXPECT_EQ(num8, 0);

  int32_t num9 = test.ToUInt32();
  EXPECT_EQ(num9, 0);
}

TEST_F(UtestFP16, OperatorAdd_success) {
  fp16_t test1(1);
  fp16_t test2(1);
  test1 = test1 + test2;
  EXPECT_EQ(test1.val, 2);
}

TEST_F(UtestFP16, OperatorSubtract_success) {
  fp16_t test1(1);
  fp16_t test2(1);
  test1 = test1 - test2;
  EXPECT_EQ(test1.val, 0);
}

TEST_F(UtestFP16, OperatorMultiply_success) {
  fp16_t test1(1);
  fp16_t test2(2);
  test1 = test1 * test2;
  EXPECT_EQ(test1.val, 0);
}

TEST_F(UtestFP16, OperatorEqual_success) {
  fp16_t test1(1);
  fp16_t test2(2);
  fp16_t test3(1);
  EXPECT_EQ(test1 == test2, false);
  EXPECT_EQ(test1 == test3, true);
}

TEST_F(UtestFP16, OperatorGreaterThan_success) {
  fp16_t test1(1);
  fp16_t test2(2);
  fp16_t test3(1);
  EXPECT_EQ(test1 > test2, false);
  EXPECT_EQ(test2 > test1, true);
  fp16_t test4((uint16_t)200000U);
  fp16_t test5((uint16_t)300000U);
  EXPECT_EQ(test4 > test5, true);
  EXPECT_EQ(test5 > test4, false);
}

TEST_F(UtestFP16, OperatorEqualOrGreaterThan_success) {
  fp16_t test1(1);
  fp16_t test2(2);
  fp16_t test3(1);
  EXPECT_EQ(test1 >= test2, false);
  EXPECT_EQ(test1 >= test3, true);
}

TEST_F(UtestFP16, OperatorEqualOrLessThan_success) {
  fp16_t test1(1);
  fp16_t test2(2);
  fp16_t test3(3);
  EXPECT_EQ(test1 <= test2, true);
  EXPECT_EQ(test3 <= test2, false);
  fp16_t test4((uint16_t)200000U);
  fp16_t test5((uint16_t)300000U);
  EXPECT_EQ(test4 <= test5, false);
  EXPECT_EQ(test5 <= test4, true);
}

TEST_F(UtestFP16, OperatorEqualToTagFp16_success) {
  fp16_t test(1);
  fp16_t test1(1);
  test1 = test;
  EXPECT_EQ(test1.val, 1);

  float32_t val_f32 = 4294967296;
  test1 = val_f32;
  EXPECT_EQ(test1.val, 31743);

  float64_t val_f64 = 8589934592;
  test1 = val_f64;
  EXPECT_EQ(test1.val, 31743);

  fp16_t test4((uint16_t)kFp64ExpMask);
  test1 = test4;
  EXPECT_EQ(test1.val, 0);

  fp16_t test5(0);
  test1 = test5;
  EXPECT_EQ(test1.val, 0);
}

TEST_F(UtestFP16, OperatorEqualTofloat32_t_success) {
  fp16_t test1(1);
  float32_t test2 = 2.0F;
  test1 = test2;
  EXPECT_EQ(test1.val, 16384);
}

TEST_F(UtestFP16, OperatorEqualTofloat64_t_success) {
  fp16_t test1(1);
  float64_t test2 = 2.0F;
  test1 = test2;
  EXPECT_EQ(test1.val, 16384);
}

TEST_F(UtestFP16, OperatorEqualToint32_t_success) {
  fp16_t test1(1);
  int32_t test2 = 2;
  test1 = test2;
  EXPECT_EQ(test1.val, 16384);
}

TEST_F(UtestFP16, Float32_t_success) {
  fp16_t test(0);
  EXPECT_EQ(float32_t(test), 0);
}

TEST_F(UtestFP16, Float64_t_success) {
  fp16_t test(0);
  EXPECT_EQ(float64_t(test), 0);
}

TEST_F(UtestFP16, Int8_t_success) {
  fp16_t test(0);
  EXPECT_EQ(int8_t(test), '\0');
  fp16_t test1(-1);
  EXPECT_EQ(int8_t(test1), -128);
  fp16_t test2(1024);
  EXPECT_EQ(int8_t(test2), '\0');
  fp16_t test3(31744);
  EXPECT_EQ(int8_t(test3), '\x7F');
  fp16_t test4((uint16_t)130944U);
  EXPECT_EQ(int8_t(test4), -128);
}

TEST_F(UtestFP16, Uint8_t_success) {
  fp16_t test(0);
  EXPECT_EQ(uint8_t(test), '\0');
  fp16_t test2(1024);
  EXPECT_EQ(uint8_t(test2), '\0');
  fp16_t test3(31744);
  EXPECT_EQ(uint8_t(test3), 255);
  fp16_t test4((uint16_t)130944U);
  EXPECT_EQ(uint8_t(test4), '\0');
}

TEST_F(UtestFP16, Int16_t_success) {
  fp16_t test(0);
  EXPECT_EQ(int16_t(test), 0);
  fp16_t test1(31744);
  EXPECT_EQ(int16_t(test1), 32767);
  fp16_t test2(64512);
  EXPECT_EQ(int16_t(test2), -32768);

  fp16_t test3(~kFp16ExpMask);
  EXPECT_EQ(int16_t(test3), 0);
}

TEST_F(UtestFP16, uint16_t_success) {
  fp16_t test(0);
  EXPECT_EQ(uint16_t(test), 0);
  fp16_t test1(31744);
  EXPECT_EQ(uint16_t(test1), 255);
  fp16_t test2(64512);
  EXPECT_EQ(uint16_t(test2), 0);
  fp16_t test3(~31744);
  EXPECT_EQ(uint16_t(test3), 0);
}

TEST_F(UtestFP16, Int32_t_success) {
  fp16_t test(0);
  EXPECT_EQ(int32_t(test), 0);

  fp16_t test1(kFp16ExpMask);
  EXPECT_EQ(int32_t(test1), 2147483647);

  fp16_t test2(~kFp16ExpMask);
  EXPECT_EQ(int32_t(test2), 0);
}

TEST_F(UtestFP16, Uint32_t_success) {
  fp16_t test(0);
  EXPECT_EQ(uint32_t(test), 0);
  fp16_t test1(64512);
  EXPECT_EQ(uint32_t(test1), 0);

  fp16_t test2(kFp16ExpMask);
  EXPECT_EQ(uint32_t(test2), 255);

  fp16_t test3(~kFp16ExpMask);
  EXPECT_EQ(uint32_t(test3), 0);
}

TEST_F(UtestFP16, Int64_t_success) {
  fp16_t test(0);
  EXPECT_EQ(int64_t(test), 0);
}

TEST_F(UtestFP16, uint64_t_success) {
  fp16_t test(0);
  EXPECT_EQ(uint64_t(test), 0);
}

TEST_F(UtestFP16, RightShift_success) {
  int16_t man = 0;
  fp16_t test;
  EXPECT_EQ(RightShift(man, 0), 0);
}

TEST_F(UtestFP16, GetManSum_success) {
  int16_t m_a = 0;
  int16_t m_b = 0;
  fp16_t test;
  EXPECT_EQ(GetManSum(0, m_a, 1, m_b), 0);
  EXPECT_EQ(GetManSum(1, m_a, 0, m_b), 0);
}

TEST_F(UtestFP16, Fp16ToFloat_Denormal_CovEnhance) {
  fp16_t denorm;
  denorm.val = 0x0001U;
  float f = denorm.ToFloat();
  EXPECT_NE(f, 0.0f);

  fp16_t denorm2;
  denorm2.val = 0x0200U;
  float f2 = denorm2.ToFloat();
  EXPECT_NE(f2, 0.0f);
}

TEST_F(UtestFP16, Fp16ToDouble_Denormal_CovEnhance) {
  fp16_t denorm;
  denorm.val = 0x0001U;
  double d = denorm.ToDouble();
  EXPECT_NE(d, 0.0);
}

TEST_F(UtestFP16, Int8_OverflowPaths_CovEnhance) {
  fp16_t pos_val;
  pos_val.val = 0x4C00U;  // 2^4 * 1.0 = 16.0
  int8_t i8_pos = pos_val.ToInt8();
  EXPECT_EQ(i8_pos, 16);

  fp16_t neg_val;
  neg_val.val = 0xCC00U;  // -16.0
  int8_t i8_neg = neg_val.ToInt8();
  EXPECT_EQ(i8_neg, -16);

  fp16_t small_pos;
  small_pos.val = 0x4180U;  // ~2.75
  int8_t i8_small = small_pos.ToInt8();
  EXPECT_EQ(i8_small, 3);
}

TEST_F(UtestFP16, Uint8_OverflowPath_CovEnhance) {
  fp16_t large_val;
  large_val.val = 0x5400U;  // 2^6 = 64.0
  uint8_t u8 = large_val.ToUInt8();
  EXPECT_EQ(u8, 64);

  fp16_t normal_val;
  normal_val.val = 0x4180U;  // ~2.75
  uint8_t u8_normal = normal_val.ToUInt8();
  EXPECT_EQ(u8_normal, 3);
}

TEST_F(UtestFP16, Int16_OverflowPaths_CovEnhance) {
  fp16_t large_pos;
  large_pos.val = 0x6C00U;  // 2^12 = 4096.0
  int16_t i16 = large_pos.ToInt16();
  EXPECT_EQ(i16, 4096);

  fp16_t large_neg;
  large_neg.val = 0xEC00U;  // -4096.0
  int16_t i16_neg = large_neg.ToInt16();
  EXPECT_EQ(i16_neg, -4096);

  fp16_t mid_val;
  mid_val.val = 0x4400U;  // 4.0
  int16_t i16_mid = mid_val.ToInt16();
  EXPECT_EQ(i16_mid, 4);
}

TEST_F(UtestFP16, Uint16_Conversion_CovEnhance) {
  fp16_t val;
  val.val = 0x4400U;  // 4.0
  uint16_t u16 = val.ToUInt16();
  EXPECT_EQ(u16, 4);

  fp16_t large_val;
  large_val.val = 0x6C00U;  // 2^12 = 4096.0
  uint16_t u16_large = large_val.ToUInt16();
  EXPECT_EQ(u16_large, 4096);
}

TEST_F(UtestFP16, Int32_Rounding_CovEnhance) {
  fp16_t val;
  val.val = 0x4180U;  // ~2.75
  int32_t i32 = val.ToInt32();
  EXPECT_EQ(i32, 3);

  fp16_t neg_val;
  neg_val.val = 0xC180U;  // ~-2.75
  int32_t i32_neg = neg_val.ToInt32();
  EXPECT_EQ(i32_neg, -3);
}

TEST_F(UtestFP16, Uint32_Conversion_CovEnhance) {
  fp16_t val;
  val.val = 0x4400U;  // 4.0
  uint32_t u32 = val.ToUInt32();
  EXPECT_EQ(u32, 4U);

  fp16_t neg_val;
  neg_val.val = 0xC400U;  // -4.0
  uint32_t u32_neg = neg_val.ToUInt32();
  EXPECT_EQ(u32_neg, 0U);
}

TEST_F(UtestFP16, OperatorGreaterThan_NegativeBoth_CovEnhance) {
  fp16_t neg1;
  neg1.val = 0xBC00U;  // -1.0
  fp16_t neg2;
  neg2.val = 0xBE00U;            // -1.5
  EXPECT_EQ(neg1 > neg2, true);  // -1 > -1.5

  fp16_t neg3;
  neg3.val = 0xC000U;             // -2.0
  EXPECT_EQ(neg3 > neg1, false);  // -2 > -1 is false (e_a >= e_b)
}

TEST_F(UtestFP16, OperatorGreaterThan_PositiveBoth_EDiff_CovEnhance) {
  fp16_t pos1;
  pos1.val = 0x4000U;  // 2.0
  fp16_t pos2;
  pos2.val = 0x4400U;             // 4.0
  EXPECT_EQ(pos1 > pos2, false);  // 2 > 4 is false, e_a < e_b
}

TEST_F(UtestFP16, OperatorAssign_Int32_Zero_CovEnhance) {
  fp16_t test(1);
  test = 0;
  EXPECT_EQ(test.val, 0U);
}

TEST_F(UtestFP16, OperatorAssign_Int32_Negative_CovEnhance) {
  fp16_t test(1);
  test = -2;
  EXPECT_EQ(test.val, 0xC000U);  // -2.0 in fp16_t

  test = -1;
  EXPECT_EQ(test.val, 0xBC00U);  // -1.0 in fp16_t
}

TEST_F(UtestFP16, OperatorAssign_Float64_Denormal_CovEnhance) {
  fp16_t test(1);
  double tiny = 5.960464477539063e-08;  // 2^-24, denormal range
  test = tiny;
  EXPECT_NE(test.val, 0U);

  fp16_t test2(1);
  double tinier = 2.980232238769531e-08;  // 2^-25, smaller than smallest denormal
  test2 = tinier;
  EXPECT_EQ(test2.val, 0U);

  fp16_t test3(1);
  double very_tiny = 1.0e-45;  // smaller than smallest denormal
  test3 = very_tiny;
  EXPECT_EQ(test3.val, 0U);

  fp16_t test4(1);
  double normal = 2.0;
  test4 = normal;
  EXPECT_EQ(test4.val, 0x4000U);

  fp16_t test5(1);
  double overflow_val = 1e20;
  test5 = overflow_val;
  EXPECT_EQ(test5.val, 0x7BFFU);
}

TEST_F(UtestFP16, OperatorAssign_Float32_Denormal_CovEnhance) {
  fp16_t test(1);
  float tiny = 5.960464477539063e-08F;  // 2^-24, denormal
  test = tiny;
  EXPECT_NE(test.val, 0U);

  fp16_t test2(1);
  float tinier = 2.980232238769531e-08F;  // 2^-25, smaller than smallest denormal
  test2 = tinier;
  EXPECT_EQ(test2.val, 0U);

  fp16_t test3(1);
  float zero = 0.0F;
  test3 = zero;
  EXPECT_EQ(test3.val, 0U);

  fp16_t test4(1);
  float overflow_val = 1e20F;
  test4 = overflow_val;
  EXPECT_EQ(test4.val, 0x7BFFU);
}

TEST_F(UtestFP16, Fp16Add_DifferentExponents_CovEnhance) {
  fp16_t a;
  a.val = 0x4000U;  // 2.0
  fp16_t b;
  b.val = 0x3C00U;  // 1.0
  fp16_t result = a + b;
  EXPECT_EQ(result.val, 0x4200U);  // 3.0

  fp16_t c;
  c.val = 0x3C00U;  // 1.0
  fp16_t d;
  d.val = 0x4000U;  // 2.0
  fp16_t result2 = c + d;
  EXPECT_EQ(result2.val, 0x4200U);  // 3.0
}

TEST_F(UtestFP16, Fp16Mul_ShiftPaths_CovEnhance) {
  fp16_t a;
  a.val = 0x4000U;  // 2.0
  fp16_t b;
  b.val = 0x4000U;  // 2.0
  fp16_t result = a * b;
  EXPECT_EQ(result.val, 0x4400U);  // 4.0

  fp16_t c;
  c.val = 0x3C00U;  // 1.0
  fp16_t d;
  d.val = 0x3C00U;  // 1.0
  fp16_t result2 = c * d;
  EXPECT_EQ(result2.val, 0x3C00U);  // 1.0

  fp16_t e;
  e.val = 0x4400U;  // 4.0
  fp16_t f;
  f.val = 0x4400U;  // 4.0
  fp16_t result3 = e * f;
  EXPECT_EQ(result3.val, 0x4C00U);  // 16.0
}

TEST_F(UtestFP16, Fp16Sub_DifferentSign_CovEnhance) {
  fp16_t a;
  a.val = 0x4200U;  // 3.0
  fp16_t b;
  b.val = 0x4000U;  // 2.0
  fp16_t result = a - b;
  EXPECT_EQ(result.val, 0x3C00U);  // 1.0

  fp16_t c;
  c.val = 0x3C00U;  // 1.0
  fp16_t d;
  d.val = 0x4000U;  // 2.0
  fp16_t result2 = c - d;
  EXPECT_EQ(result2.val, 0xBC00U);  // -1.0
}

TEST_F(UtestFP16, Int8_WhileLoopOverflow_CovEnhance) {
  fp16_t neg_val;
  neg_val.val = 0xD800U;
  EXPECT_EQ(neg_val.ToInt8(), -128);

  fp16_t pos_val;
  pos_val.val = 0x5800U;
  EXPECT_EQ(pos_val.ToInt8(), 127);
}

TEST_F(UtestFP16, Uint8_WhileLoopOverflow_CovEnhance) {
  fp16_t val;
  val.val = 0x5C00U;
  EXPECT_EQ(val.ToUInt8(), 255);
}

TEST_F(UtestFP16, Int16_WhileLoopOverflow_CovEnhance) {
  fp16_t neg_val;
  neg_val.val = 0xF801U;
  EXPECT_EQ(neg_val.ToInt16(), -32768);

  fp16_t pos_val;
  pos_val.val = 0x7800U;
  EXPECT_EQ(pos_val.ToInt16(), 32767);
}

TEST_F(UtestFP16, Int16_SmallExpRoundingAndSignReset_CovEnhance) {
  fp16_t val;
  val.val = 0x3880U;
  EXPECT_EQ(val.ToInt16(), 1);

  fp16_t neg_val;
  neg_val.val = 0xC140U;
  EXPECT_EQ(neg_val.ToInt16(), -3);

  fp16_t zero_neg;
  zero_neg.val = 0xB800U;
  EXPECT_EQ(zero_neg.ToInt16(), 0);
}

TEST_F(UtestFP16, Uint16_SmallExpAndRounding_CovEnhance) {
  fp16_t val;
  val.val = 0x3880U;
  EXPECT_EQ(val.ToUInt16(), 1);
}

TEST_F(UtestFP16, Int32_SmallExpAndRounding_CovEnhance) {
  fp16_t val;
  val.val = 0x3880U;
  EXPECT_EQ(val.ToInt32(), 1);
}

TEST_F(UtestFP16, Int8_NegativeRoundingCondition_CovEnhance) {
  fp16_t val;
  val.val = 0xC140U;
  EXPECT_EQ(val.ToInt8(), -3);
}

TEST_F(UtestFP16, OperatorGreaterThan_NegativeExpDiff_CovEnhance) {
  fp16_t lhs;
  lhs.val = 0xBC00U;
  fp16_t rhs;
  rhs.val = 0xC000U;
  EXPECT_EQ(lhs > rhs, true);
}

TEST_F(UtestFP16, Fp16Add_SameExpMantissaOverflow_CovEnhance) {
  fp16_t a;
  a.val = 0x3C00U;
  fp16_t b;
  b.val = 0x3C00U;
  fp16_t result = a + b;
  EXPECT_EQ(result.val, 0x4000U);
}

TEST_F(UtestFP16, Fp16Mul_NormalizeDenormalBoundary_CovEnhance) {
  fp16_t a;
  a.val = 0x0001U;
  fp16_t b;
  b.val = 0x6000U;
  fp16_t result = a * b;
  EXPECT_NE(result.val, 0U);
}

TEST_F(UtestFP16, Fp16Mul_ZeroShiftPath_CovEnhance) {
  fp16_t a;
  a.val = 0x0000U;
  fp16_t b;
  b.val = 0x7800U;
  fp16_t result = a * b;
  EXPECT_EQ(result.val, 0U);
}

TEST_F(UtestFP16, OperatorAssign_Float32_DenormalRounding_CovEnhance) {
  fp16_t test(1);
  test = 8.940696716308594e-08F;
  EXPECT_NE(test.val, 0U);

  fp16_t test2(1);
  test2 = 3.0e-08F;
  EXPECT_NE(test2.val, 0U);
}

TEST_F(UtestFP16, OperatorAssign_Float32_RoundingOverflow_CovEnhance) {
  fp16_t test(1);
  test = 1.9990234375F;
  EXPECT_EQ(test.val, 0x3FFFU);
}

TEST_F(UtestFP16, OperatorAssign_Float64_DenormalRounding_CovEnhance) {
  fp16_t test(1);
  test = 8.940696716308594e-08;
  EXPECT_NE(test.val, 0U);

  fp16_t test2(1);
  test2 = 3.0e-08;
  EXPECT_NE(test2.val, 0U);
}

TEST_F(UtestFP16, OperatorAssign_Float64_RoundingOverflow_CovEnhance) {
  fp16_t test(1);
  test = 1.9990234375;
  EXPECT_EQ(test.val, 0x3FFFU);
}

TEST_F(UtestFP16, OperatorAssign_Int32_LargeOverflow_CovEnhance) {
  fp16_t test(1);
  test = 2147483647;
  EXPECT_EQ(test.val, 0x7BFFU);
}
}  // namespace formats
}  // namespace ge
