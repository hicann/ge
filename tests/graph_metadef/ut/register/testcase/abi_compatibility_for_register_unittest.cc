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
#include "graph/any_value.h"
#include "register/op_impl_registry.h"
#include "register/op_impl_registry_base.h"
#include "register/kernel_registry_impl.h"

namespace gert {
namespace {
constexpr const size_t kUint8Size = 1U;
constexpr const size_t kPointerSize = 8U;
constexpr const size_t kVectorSize = 24U;
constexpr const size_t kUnorderedSetSize = 56U;
constexpr const size_t kMapSize = 48U;
constexpr const size_t kVirtualTableSize = 8U;
constexpr const size_t kReservedFieldSize = 8U;
constexpr const size_t kReservedFieldSize2 = 40U;
constexpr const size_t kOpImplFunctionSize = 216U;
constexpr const size_t kOpImplReservedFieldSize = 502UL * sizeof(void*);
constexpr const size_t kOpImplSize = 4304UL;

constexpr const size_t kOpImplFunctionsSize = 200U;
constexpr const size_t kOpImplRegistrySize = 88U + kVirtualTableSize;
constexpr const size_t kOpImplRegisterSize = 56U;
}  // namespace

constexpr size_t OpImplKernelRegistry::OpImplFunctions::kByteBitCount;
class AbiCompatibilityForRegisterUT : public testing::Test {};
TEST_F(AbiCompatibilityForRegisterUT, OpImplRegistry_CheckMemLayoutNotChanged) {
  OpImplRegistry r;
  ASSERT_EQ(sizeof(r), kOpImplRegistrySize);
  ASSERT_EQ(reinterpret_cast<uintptr_t>(&r.types_to_impl_) - reinterpret_cast<uintptr_t>(&r), kVirtualTableSize);

  EXPECT_EQ(reinterpret_cast<uintptr_t>(&r.reserved_) - reinterpret_cast<uintptr_t>(&r.types_to_impl_),
            kMapSize);
  EXPECT_EQ(sizeof(r.reserved_), kReservedFieldSize2);
}

TEST_F(AbiCompatibilityForRegisterUT, OpImplFunctionsV2_CheckMemLayoutNotChanged) {
  gert::OpImplKernelRegistry::OpImplFunctionsV2 r;
  ASSERT_EQ(sizeof(r), kOpImplSize);
  EXPECT_EQ(sizeof(r.reserved_), kOpImplReservedFieldSize);
}
}  // namespace gert
