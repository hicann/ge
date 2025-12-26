/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "gtest/gtest.h"
#include "generator/extra_info_gen/extra_info_generator.h"

namespace att {
class CheckGenCheckUT : public ::testing::Test {
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

TEST_F(CheckGenCheckUT, GetDimCheckCode) {
  std::map<uint32_t, uint32_t> max_dim;
  std::map<uint32_t, uint32_t> min_dim;
  std::map<uint32_t, std::vector<int64_t>> map_A;
  std::map<uint32_t, std::vector<int64_t>> map_B;
  std::map<std::string, std::map<uint32_t, std::vector<int64_t>>> axis_map;
  map_A[0] = {0, -2};
  map_A[1] = {0, -2};
  map_A[2] = {INT64_MAX};
  axis_map["A"] = map_A;
  
  map_B[0] = {-1, 4};
  map_B[1] = {1, -3};
  map_B[2] = {0, -3};
  axis_map["B"] = map_B;
  EXPECT_EQ(GenDimMap(axis_map, max_dim, min_dim), ge::SUCCESS);
  EXPECT_FALSE(max_dim.size() == 0);
}

TEST_F(CheckGenCheckUT, CheckRequireCover) {
  std::vector<std::vector<int64_t>> intervals;
  intervals = {{0, -2}, {-1}};
  EXPECT_FALSE(RequireCoverCheck(intervals));

  intervals = {{0, -3}, {-1}};
  EXPECT_TRUE(RequireCoverCheck(intervals));

  intervals = {{0}, {1, 2}, {3, -4}, {-3, -2}, {-1}};
  EXPECT_FALSE(RequireCoverCheck(intervals));
  EXPECT_TRUE(GenCheckInputCoverFunc(0, intervals).size() == 0);

  intervals = {{0}, {1, 2}, {-3, -2}, {-1 }};
  EXPECT_TRUE(RequireCoverCheck(intervals));
  EXPECT_FALSE(GenCheckInputCoverFunc(0, intervals).size() == 0);
}
}