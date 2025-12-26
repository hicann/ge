/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GE_SESSION_V2_GE_SESSION_IMPL_H_
#define GE_SESSION_V2_GE_SESSION_IMPL_H_

#include <cstdint>
#include "ge/ge_api_v2.h"
namespace ge {
class GeSession::Impl {
  public:
    Impl();

    ~Impl();

    void SetSessionId(uint64_t session_id);

    uint64_t GetSessionId() const;

  private:
    uint64_t session_id_{0};
};
}  // namespace ge

#endif  // GE_SESSION_V2_GE_SESSION_IMPL_H_