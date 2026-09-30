/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdint>
#include <gtest/gtest.h>

#include "exe_graph/runtime/annotated_args_handler.h"
#include "ge/ge_api_types.h"

namespace gert {
namespace {

TEST(DeclarativeCustomOpAttachedStreamSt, RequestAttachedStreamKeepsPerKeyIds) {
  AnnotatedArgsHandler handler;
  handler.SetAttachedStreamRequestFunc(
      [](const ge::AscendString &key) { return key.GetLength() == 0U ? 0U : static_cast<uint32_t>(key.GetLength()); });

  EXPECT_EQ(handler.RequestAttachedStream(ge::AscendString("aux")), 3U);
  EXPECT_EQ(handler.RequestAttachedStream(ge::AscendString("aux")), 3U);
  EXPECT_EQ(handler.GetAttachedStreamIds().size(), 2U);
}

TEST(DeclarativeCustomOpAttachedStreamSt, MissingRequestCallbackReturnsInvalidStream) {
  AnnotatedArgsHandler handler;
  EXPECT_EQ(handler.RequestAttachedStream(ge::AscendString("aux")), UINT32_MAX);
  EXPECT_TRUE(handler.GetAttachedStreamIds().empty());
}

}  // namespace
}  // namespace gert
