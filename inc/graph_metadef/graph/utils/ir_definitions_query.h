/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef INC_GRAPH_METADEF_GRAPH_UTILS_IR_DEFINITIONS_QUERY_H_
#define INC_GRAPH_METADEF_GRAPH_UTILS_IR_DEFINITIONS_QUERY_H_

#include <utility>
#include <vector>

#include "framework/common/ge_visibility.h"
#include "ge_common/ge_common_api_types.h"

extern "C" VISIBILITY_EXPORT ge::Status GetRegisteredIrDefFromGraph(
    const char *op_type, std::vector<std::pair<ge::AscendString, ge::AscendString>> &inputs,
    std::vector<std::pair<ge::AscendString, ge::AscendString>> &outputs,
    std::vector<std::pair<ge::AscendString, ge::AscendString>> &attrs);

#endif  // INC_GRAPH_METADEF_GRAPH_UTILS_IR_DEFINITIONS_QUERY_H_
