/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_RUNTIME_OM2_MALLOC_HELPER_H_
#define AIR_RUNTIME_OM2_MALLOC_HELPER_H_

#include <cstdint>

#include "acl/acl_rt.h"

namespace gert {

using rtMemType_t = uint32_t;

aclError Om2Malloc(void **ptr, size_t size, rtMemType_t mem_type, uint16_t module_id);

}  // namespace gert

#endif  // AIR_RUNTIME_OM2_MALLOC_HELPER_H_
