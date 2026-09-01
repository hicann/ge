/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "framework/runtime/gert_model/gert_model_executor_callbacks.h"

#include <gtest/gtest.h>

namespace ge {
namespace {

class Om2KernelHandlesTest : public testing::Test {};

class BinHandleStoreLockGuard {
 public:
  BinHandleStoreLockGuard() {
    EXPECT_EQ(::LockBinHandleStore(), 0);
  }

  ~BinHandleStoreLockGuard() {
    EXPECT_EQ(::UnlockBinHandleStore(), 0);
  }
};

TEST_F(Om2KernelHandlesTest, QueryMissingBinReturnsNull) {
  BinHandleStoreLockGuard lock_guard;
  aclrtBinHandle bin_handle = reinterpret_cast<aclrtBinHandle>(0x1234);
  EXPECT_EQ(::QueryBinHandleFromStore("om2_missing_bin", &bin_handle), 0);
  EXPECT_EQ(bin_handle, nullptr);
}

TEST_F(Om2KernelHandlesTest, NullArgumentsReturnError) {
  uint8_t need_unload = 0U;
  aclrtBinHandle bin_handle = nullptr;

  EXPECT_EQ(::QueryBinHandleFromStore(nullptr, &bin_handle), -1);
  EXPECT_EQ(::ReleaseBinHandleFromStore(nullptr, &need_unload), -1);
  EXPECT_EQ(::ReleaseBinHandleFromStore("om2_null_bin", nullptr), -1);
  EXPECT_EQ(::SaveBinHandleToStore(nullptr, reinterpret_cast<aclrtBinHandle>(0x1)), -1);
}

TEST_F(Om2KernelHandlesTest, SaveAndQueryReturnsStoredHandle) {
  BinHandleStoreLockGuard lock_guard;
  constexpr auto kBinId = "om2_kernel_handles_save_query";
  const auto bin_handle = reinterpret_cast<aclrtBinHandle>(0x12345678);

  EXPECT_EQ(::SaveBinHandleToStore(kBinId, bin_handle), 0);

  aclrtBinHandle queried_handle = nullptr;
  EXPECT_EQ(::QueryBinHandleFromStore(kBinId, &queried_handle), 0);
  EXPECT_EQ(queried_handle, bin_handle);
}

TEST_F(Om2KernelHandlesTest, SaveSameBinIncrementsReferenceAndReleasePairs) {
  BinHandleStoreLockGuard lock_guard;
  constexpr auto kBinId = "om2_kernel_handles_ref_count";
  const auto bin_handle = reinterpret_cast<aclrtBinHandle>(0x22334455);
  uint8_t need_unload = 0U;
  aclrtBinHandle queried_handle = nullptr;

  EXPECT_EQ(::SaveBinHandleToStore(kBinId, bin_handle), 0);
  EXPECT_EQ(::SaveBinHandleToStore(kBinId, reinterpret_cast<aclrtBinHandle>(0x1)), 0);

  EXPECT_EQ(::QueryBinHandleFromStore(kBinId, &queried_handle), 0);
  EXPECT_EQ(queried_handle, bin_handle);

  EXPECT_EQ(::ReleaseBinHandleFromStore(kBinId, &need_unload), 0);
  EXPECT_EQ(need_unload, 0U);

  queried_handle = reinterpret_cast<aclrtBinHandle>(0xdeadbeef);
  EXPECT_EQ(::QueryBinHandleFromStore(kBinId, &queried_handle), 0);
  EXPECT_EQ(queried_handle, bin_handle);

  EXPECT_EQ(::ReleaseBinHandleFromStore(kBinId, &need_unload), 0);
  EXPECT_EQ(need_unload, 1U);

  queried_handle = reinterpret_cast<aclrtBinHandle>(0xdeadbeef);
  EXPECT_EQ(::QueryBinHandleFromStore(kBinId, &queried_handle), 0);
  EXPECT_EQ(queried_handle, nullptr);
}

TEST_F(Om2KernelHandlesTest, ReleaseMissingBinDoesNotUnload) {
  BinHandleStoreLockGuard lock_guard;
  uint8_t need_unload = 1U;

  EXPECT_EQ(::ReleaseBinHandleFromStore("om2_missing_release_bin", &need_unload), 0);
  EXPECT_EQ(need_unload, 0U);
}

TEST_F(Om2KernelHandlesTest, RecursiveLockAllowsNestedLockAndUnlock) {
  EXPECT_EQ(::LockBinHandleStore(), 0);
  EXPECT_EQ(::LockBinHandleStore(), 0);
  EXPECT_EQ(::UnlockBinHandleStore(), 0);
  EXPECT_EQ(::UnlockBinHandleStore(), 0);
}

}  // namespace
}  // namespace ge
