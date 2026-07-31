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
#include <string>
#include <vector>
#include <cstdint>
#include <climits>
#include "graph/types.h"
#include "graph/error_codes.h"
#include "graph/op_types.h"
#include "graph/utils/type_utils.h"

namespace ge {
class UtestTypesCov : public testing::Test {
 protected:
  void SetUp() {}
  void TearDown() {}
};

TEST_F(UtestTypesCov, OpTypeContainerInstance) {
  auto &container = OpTypeContainer::Instance();
  EXPECT_NE(&container, nullptr);
  auto &container2 = OpTypeContainer::Instance();
  EXPECT_EQ(&container, &container2);
  container.Register("TestOpTypeCov");
  EXPECT_TRUE(container.IsExisting("TestOpTypeCov"));
  EXPECT_FALSE(container.IsExisting("NonExistentOpTypeCov"));
}

TEST_F(UtestTypesCov, GetFormatNameNormal) {
  const char_t *name = GetFormatName(FORMAT_NCHW);
  ASSERT_NE(name, nullptr);
  EXPECT_STREQ(name, "NCHW");

  name = GetFormatName(FORMAT_NHWC);
  ASSERT_NE(name, nullptr);
  EXPECT_STREQ(name, "NHWC");

  name = GetFormatName(FORMAT_ND);
  ASSERT_NE(name, nullptr);
  EXPECT_STREQ(name, "ND");

  name = GetFormatName(FORMAT_FRACTAL_Z);
  ASSERT_NE(name, nullptr);
  EXPECT_STREQ(name, "FRACTAL_Z");
}

TEST_F(UtestTypesCov, GetFormatNameBeyondEnd) {
  const char_t *name = GetFormatName(FORMAT_END);
  ASSERT_NE(name, nullptr);
  EXPECT_STREQ(name, "UNKNOWN");

  name = GetFormatName(static_cast<Format>(FORMAT_END + 1));
  ASSERT_NE(name, nullptr);
  EXPECT_STREQ(name, "UNKNOWN");

  name = GetFormatName(static_cast<Format>(0xFFFF));
  ASSERT_NE(name, nullptr);
  EXPECT_STREQ(name, "UNKNOWN");
}

TEST_F(UtestTypesCov, GetSizeInBytesNegativeCount) {
  int64_t result = GetSizeInBytes(-1, DT_FLOAT);
  EXPECT_EQ(result, -1);
  result = GetSizeInBytes(-100, DT_INT32);
  EXPECT_EQ(result, -1);
}

TEST_F(UtestTypesCov, GetSizeInBytesUnknownType) {
  int64_t result = GetSizeInBytes(10, DT_UNDEFINED);
  EXPECT_EQ(result, -1);
}

TEST_F(UtestTypesCov, GetSizeInBytesNormal) {
  int64_t result = GetSizeInBytes(10, DT_FLOAT);
  EXPECT_EQ(result, 40);
  result = GetSizeInBytes(100, DT_FLOAT16);
  EXPECT_EQ(result, 200);
  result = GetSizeInBytes(0, DT_FLOAT);
  EXPECT_EQ(result, 0);
}

TEST_F(UtestTypesCov, GetSizeInBytesOverflow) {
  int64_t result = GetSizeInBytes(INT64_MAX, DT_FLOAT);
  EXPECT_EQ(result, -1);
  result = GetSizeInBytes(INT64_MAX, DT_FLOAT16);
  EXPECT_EQ(result, -1);
  result = GetSizeInBytes(INT64_MAX, DT_INT64);
  EXPECT_EQ(result, -1);
}

TEST_F(UtestTypesCov, GetSizeInBytesBitType) {
  int64_t result = GetSizeInBytes(16, DT_INT4);
  EXPECT_GE(result, 0);
  result = GetSizeInBytes(8, DT_UINT1);
  EXPECT_GE(result, 0);
  result = GetSizeInBytes(16, DT_INT2);
  EXPECT_GE(result, 0);
  result = GetSizeInBytes(16, DT_UINT2);
  EXPECT_GE(result, 0);
}

TEST_F(UtestTypesCov, GetSizeInBytesBitTypeOverflow) {
  int64_t result = GetSizeInBytes(INT64_MAX, DT_INT4);
  EXPECT_EQ(result, -1);
  result = GetSizeInBytes(INT64_MAX, DT_INT2);
  EXPECT_EQ(result, -1);
}

TEST_F(UtestTypesCov, PromoteSymsWithNullData) {
  Promote promote({});
  auto syms = promote.Syms();
  EXPECT_TRUE(syms.empty());
}

TEST_F(UtestTypesCov, PromoteSymsNormal) {
  Promote promote({"T1", "T2", "T3"});
  auto syms = promote.Syms();
  EXPECT_EQ(syms.size(), 3U);
  EXPECT_STREQ(syms[0], "T1");
  EXPECT_STREQ(syms[1], "T2");
  EXPECT_STREQ(syms[2], "T3");
}

TEST_F(UtestTypesCov, PromoteSymsWithNullSym) {
  Promote promote({nullptr, "T2"});
  auto syms = promote.Syms();
  EXPECT_EQ(syms.size(), 2U);
  EXPECT_STREQ(syms[0], "");
  EXPECT_STREQ(syms[1], "T2");
}

TEST_F(UtestTypesCov, PromoteMoveConstructor) {
  Promote promote1({"T1", "T2"});
  auto syms1 = promote1.Syms();
  EXPECT_EQ(syms1.size(), 2U);
  Promote promote2(std::move(promote1));
  auto syms2 = promote2.Syms();
  EXPECT_EQ(syms2.size(), 2U);
  EXPECT_STREQ(syms2[0], "T1");
  EXPECT_STREQ(syms2[1], "T2");
}

TEST_F(UtestTypesCov, PromoteMoveAssignment) {
  Promote promote1({"A", "B", "C"});
  Promote promote2({"X"});
  auto syms2 = promote2.Syms();
  EXPECT_EQ(syms2.size(), 1U);
  promote2 = std::move(promote1);
  auto syms = promote2.Syms();
  EXPECT_EQ(syms.size(), 3U);
  EXPECT_STREQ(syms[0], "A");
  EXPECT_STREQ(syms[1], "B");
  EXPECT_STREQ(syms[2], "C");
}

TEST_F(UtestTypesCov, PromoteMoveSelfAssignment) {
  Promote promote1({"T1", "T2"});
  auto &ref = promote1;
  promote1 = std::move(ref);
  auto syms = promote1.Syms();
  EXPECT_EQ(syms.size(), 2U);
  EXPECT_STREQ(syms[0], "T1");
}
}  // namespace ge
