/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "session_v2/ge_session_impl.h"

namespace ge {

  GeSession::Impl::Impl() {
  }

  GeSession::Impl::~Impl() {
  }

  void GeSession::Impl::SetSessionId(uint64_t session_id) {
    session_id_ = session_id;
  }

  uint64_t GeSession::Impl::GetSessionId() const {
    return session_id_;
  }
}  // namespace ge