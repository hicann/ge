/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "session_v2/ge_session_manager.h"
#include <memory>
#include <utility>
#include "common/plugin/ge_make_unique_util.h"
#include "framework/common/debug/ge_log.h"
#include "graph/ge_context.h"
#include "graph/manager/util/rt_context_util.h"
#include "graph/manager/session_id_manager.h"

namespace ge {
Status GeSessionManager::Initialize() {
  if (init_flag_) {
    GELOGW("GeSession Manager has been initialized.");
    return SUCCESS;
  }
  init_flag_ = true;
  return SUCCESS;
}

Status GeSessionManager::Finalize() {
  if (!init_flag_) {
    GELOGW("GeSession Manager has not been initialized.");
    return SUCCESS;
  }
  const std::lock_guard<std::mutex> lock(mutex_);
  for (auto iter = session_manager_map_.cbegin(); iter != session_manager_map_.cend(); ++iter) {
    (void)iter->second->Finalize();
  }
  session_manager_map_.clear();
  init_flag_ = false;
  return SUCCESS;
}

Status GeSessionManager::SetRtContext(SessionId session_id, rtContext_t rt_context) const {
  GELOGI("set rt_context RT_CTX_NORMAL_MODE, device id:%u.", GetContext().DeviceId());
  GE_CHK_STATUS_RET(rtCtxCreate(&rt_context, RT_CTX_NORMAL_MODE, static_cast<int32_t>(GetContext().DeviceId())));
  GE_CHK_RT_RET(rtCtxSetCurrent(rt_context));
  RtContextUtil::GetInstance().AddRtContext(session_id, rt_context);
  return SUCCESS;
}

Status GeSessionManager::CreateSession(const std::map<std::string, std::string> &options, SessionId &session_id) {
  if (!init_flag_) {
    GELOGE(GE_SESSION_MANAGER_NOT_INIT, "[Create][GeSession]fail for GeSession manager is not initialized.");
    REPORT_INNER_ERR_MSG("E19999", "CreateSession fail for GeSession manager is not initialized.");
    return GE_SESSION_MANAGER_NOT_INIT;
  }

  for (const auto &item : options) {
    GELOGI("GE option: %s, value: %s.", item.first.c_str(), item.second.c_str());
  }

  SessionId next_session_id = 0;

  const std::lock_guard<std::mutex> lock(mutex_);
  const auto nextSessionIdRet = GetNextSessionId(next_session_id);
  if (nextSessionIdRet != SUCCESS) {
    return nextSessionIdRet;
  }

  InnerGeSessionPtr sessionPtr = MakeShared<InnerGeSession>(next_session_id, options);
  if (sessionPtr == nullptr) {
    return MEMALLOC_FAILED;
  }
  Status ret = sessionPtr->Initialize();
  if (ret != SUCCESS) {
    return ret;
  }

  (void)session_manager_map_.emplace(std::pair<SessionId, InnerGeSessionPtr>(next_session_id, sessionPtr));
  session_id = next_session_id;

  // create a context
  ret = SetRtContext(session_id, rtContext_t());

  return ret;
}

Status GeSessionManager::DestroySession(SessionId session_id) {
  if (!init_flag_) {
    GELOGW("[Destroy][GeSession]GeSession manager is not initialized, session_id:%lu.", session_id);
    return SUCCESS;
  }
  const std::lock_guard<std::mutex> lock(mutex_);

  const auto it = session_manager_map_.find(session_id);
  if (it == session_manager_map_.end()) {
    return GE_SESSION_NOT_EXIST;
  }

  // Unified destruct rt_context
  RtContextUtil::GetInstance().DestroyRtContexts(session_id);

  const InnerGeSessionPtr &innerSession = it->second;
  const auto ret = innerSession->Finalize();
  if (ret != SUCCESS) {
    return ret;
  }
  (void)session_manager_map_.erase(session_id);
  return ret;
}

InnerGeSessionPtr GeSessionManager::GetSession(SessionId session_id) {
  if (!init_flag_) {
    GELOGE(GE_SESSION_MANAGER_NOT_INIT, "[Get][GeSession]fail for GeSession manager is not initialized, session_id:%lu.",
           session_id);
    REPORT_INNER_ERR_MSG("E19999", "GetSession fail for GeSession manager is not initialized, session_id:%lu.",
                         session_id);
    return nullptr;
  }
  const std::lock_guard<std::mutex> lock(mutex_);
  const auto it = session_manager_map_.find(session_id);
  if (it == session_manager_map_.end()) {
    GELOGE(GE_SESSION_NOT_EXIST, "[Find][InnerGeSession] fail for %lu does not exist", session_id);
    REPORT_INNER_ERR_MSG("E19999", "GetSession fail for InnerGeSession does not exist, session_id:%lu.", session_id);
    return nullptr;
  }
  return it->second;
}

Status GeSessionManager::GetNextSessionId(SessionId &next_session_id) const {
  if (!init_flag_) {
    GELOGE(GE_SESSION_MANAGER_NOT_INIT, "[Get][NextSessionId]fail for GeSession manager is not initialized.");
    REPORT_INNER_ERR_MSG("E19999", "GetNextSessionId fail for GeSession manager is not initialized.");
    return GE_SESSION_MANAGER_NOT_INIT;
  }

  next_session_id = SessionIdManager::GetNextSessionId();
  return SUCCESS;
}
}  // namespace ge
