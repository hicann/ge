/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph/custom_op_load_context.h"
#include <cstdint>

namespace ge {
namespace {
thread_local uint32_t g_offline_custom_op_so_loading_depth = 0U;
}

ScopedOfflineCustomOpSoLoadGuard::ScopedOfflineCustomOpSoLoadGuard() {
  ++g_offline_custom_op_so_loading_depth;
}

ScopedOfflineCustomOpSoLoadGuard::~ScopedOfflineCustomOpSoLoadGuard() {
  if (g_offline_custom_op_so_loading_depth > 0U) {
    --g_offline_custom_op_so_loading_depth;
  }
}

bool IsOfflineCustomOpSoLoading() {
  return g_offline_custom_op_so_loading_depth > 0U;
}
}  // namespace ge
