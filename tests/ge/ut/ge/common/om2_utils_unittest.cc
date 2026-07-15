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

#include "common/helper/om2/om2_utils.h"

namespace ge {
namespace {

class Om2UtilsTest : public testing::Test {
 protected:
  void SetUp() override {
    // Save original value
    const char *old = std::getenv("ENABLE_RUNTIME_OM2");
    if (old != nullptr) {
      has_old_value_ = true;
      old_value_ = old;
    }
  }

  void TearDown() override {
    // Restore original value
    if (has_old_value_) {
      (void)setenv("ENABLE_RUNTIME_OM2", old_value_.c_str(), 1);
    } else {
      (void)unsetenv("ENABLE_RUNTIME_OM2");
    }
  }

 private:
  bool has_old_value_{false};
  std::string old_value_;
};

TEST_F(Om2UtilsTest, EnvNotSet_ReturnsFalse) {
  (void)unsetenv("ENABLE_RUNTIME_OM2");
  EXPECT_FALSE(IsOm2OnlineMode());
}

TEST_F(Om2UtilsTest, EnvSetTo1_ReturnsTrue) {
  (void)setenv("ENABLE_RUNTIME_OM2", "1", 1);
  EXPECT_TRUE(IsOm2OnlineMode());
}

TEST_F(Om2UtilsTest, EnvSetTo0_ReturnsFalse) {
  (void)setenv("ENABLE_RUNTIME_OM2", "0", 1);
  EXPECT_FALSE(IsOm2OnlineMode());
}

TEST_F(Om2UtilsTest, EnvSetToYes_ReturnsFalse) {
  (void)setenv("ENABLE_RUNTIME_OM2", "yes", 1);
  EXPECT_FALSE(IsOm2OnlineMode());
}

TEST_F(Om2UtilsTest, EnvSetToEmpty_ReturnsFalse) {
  (void)setenv("ENABLE_RUNTIME_OM2", "", 1);
  EXPECT_FALSE(IsOm2OnlineMode());
}

TEST_F(Om2UtilsTest, DynamicSwitch) {
  (void)setenv("ENABLE_RUNTIME_OM2", "1", 1);
  EXPECT_TRUE(IsOm2OnlineMode());

  (void)setenv("ENABLE_RUNTIME_OM2", "0", 1);
  EXPECT_FALSE(IsOm2OnlineMode());
}

}  // namespace
}  // namespace ge
