/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "bg_attached_stream_provider.h"

#include "exe_graph/lowering/frame_selector.h"

namespace gert {
namespace bg {
namespace {
const std::string kCreateAttachedStreamProvider = "CreateAttachedStreamProvider";
}  // namespace

ValueHolderPtr GetAttachedStreamProvider(LoweringGlobalData &global_data) {
  auto builder = []() -> std::vector<ValueHolderPtr> {
    return FrameSelector::OnInitRoot([]() -> std::vector<ValueHolderPtr> {
      return ValueHolder::CreateDataOutput(kCreateAttachedStreamProvider.c_str(), {}, 1U);
    });
  };
  const auto &providers = global_data.GetOrCreateUniqueValueHolder(kCreateAttachedStreamProvider, builder);
  GE_ASSERT_TRUE(!providers.empty());
  return providers[0];
}
}  // namespace bg
}  // namespace gert
