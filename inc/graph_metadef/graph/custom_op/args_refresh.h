/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef METADEF_CXX_INC_GRAPH_CUSTOM_OP_ARGS_REFRESH_H_
#define METADEF_CXX_INC_GRAPH_CUSTOM_OP_ARGS_REFRESH_H_

#include <cstdint>

namespace ge {
enum class ArgsRefreshStrategy {
  kNone = 0,
  kAnnotatedArgs,
  kUpdateCallback,
};

// Serialized in ATTR_NAME_CUSTOM_TASK_ARGS_MODE. Keep the numeric values stable for OM compatibility.
enum class CustomTaskArgsMode : int64_t {
  kUnspecified = 0,
  kNone = 1,
  kAnnotatedArgs = 2,
  kUpdateCallback = 3,
};
}  // namespace ge

#endif  // METADEF_CXX_INC_GRAPH_CUSTOM_OP_ARGS_REFRESH_H_
