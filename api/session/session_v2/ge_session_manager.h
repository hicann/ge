/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GE_SESSION_V2_GE_SESSION_MANAGER_H_
#define GE_SESSION_V2_GE_SESSION_MANAGER_H_

#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <vector>
#include "framework/common/ge_inner_error_codes.h"
#include "ge/ge_api_types.h"
#include "session_v2/inner_ge_session.h"
#include "runtime/base.h"

namespace ge {
using InnerGeSessionPtr = std::shared_ptr<InnerGeSession>;

class GeSessionManager {
 public:
  GeSessionManager() = default;

  ~GeSessionManager() = default;

  /// @ingroup ge_session
  /// @brief initialize session manager
  /// @return Status result of function
  Status Initialize();

  /// @ingroup ge_session
  /// @brief finalize session manager
  /// @return Status result of function
  Status Finalize();

  /// @ingroup ge_session
  /// @brief create session
  /// @param [in] options session config options
  /// @param [out] session_id session id
  /// @return Status result of function
  Status CreateSession(const std::map<std::string, std::string> &options, SessionId &session_id);

  /// @ingroup ge_session
  /// @brief destroy the session with specific session id
  /// @param [in] session_id session id
  /// @return Status result of function
  Status DestroySession(SessionId session_id);

  /// @ingroup ge_session
  /// @brief get session with specific session id
  /// @param [in] session_id session id
  /// @return InnerGeSessionPtr session
  InnerGeSessionPtr GetSession(SessionId session_id);

 private:
  Status GetNextSessionId(SessionId &next_session_id) const;

  Status SetRtContext(SessionId session_id, rtContext_t rt_context) const;

  std::map<SessionId, InnerGeSessionPtr> session_manager_map_;
  std::mutex mutex_;
  bool init_flag_ = false;
};
}  // namespace ge

#endif  // GE_SESSION_V2_GE_SESSION_MANAGER_H_
