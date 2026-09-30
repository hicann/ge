/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "attached_stream_collection.h"

#include "acl/acl_mdl.h"
#include "framework/common/debug/ge_log.h"
#include "graph/ascend_string.h"

namespace ge {
bool AttachedStreamCollection::IsValidKey(const AscendString &key) const {
  const char_t *const key_str = key.GetString();
  if ((key_str == nullptr) || (key_str[0] == '\0')) {
    GELOGE(PARAM_INVALID, "Invalid eager attached stream key: key is null or empty.");
    return false;
  }
  return true;
}

gert::rtStream AttachedStreamCollection::RequestAttachedStream(const AscendString &key) {
  if (!IsValidKey(key) || model_ == nullptr) {
    if (model_ == nullptr) {
      GELOGE(PARAM_INVALID, "Cannot request eager attached stream without rt model.");
    }
    return nullptr;
  }
  const std::string value(key.GetString());
  const auto found = streams_.find(value);
  if (found != streams_.end()) {
    return found->second;
  }

  rtStream_t stream = nullptr;
  if (rtStreamCreateWithFlags(&stream, priority_, stream_flags_) != RT_ERROR_NONE) {
    GELOGE(FAILED, "Failed to create eager attached stream, key=%s.", value.c_str());
    return nullptr;
  }
  // HEAD flag：辅流在模型执行时直接启动（与未被 StreamActive 激活的模型主流一致）。
  // DEFAULT flag 会使流成为 WAIT_ACTIVE 型，模型重放时无人激活导致辅流任务永不执行。
  // 主辅流之间的执行顺序由算子通过 event 同步保证（设计 E2：同步责任在算子侧）。
  const auto bind_ret = aclmdlRIBindStream(model_, stream, ACL_MODEL_STREAM_FLAG_HEAD);
  if (bind_ret != ACL_SUCCESS) {
    GELOGE(FAILED, "Failed to bind eager attached stream, key=%s, ret=%d.", value.c_str(), bind_ret);
    // 刚建好的流上没有任何任务，无需先同步即可销毁
    (void)aclrtDestroyStream(stream);
    return nullptr;
  }
  (void)streams_.emplace(value, stream);
  return stream;
}

void AttachedStreamCollection::UnbindAndDestroy() {
  for (const auto &item : streams_) {
    // 销毁前必须同步：aclrtDestroyStream 的前置约束要求流上任务已执行完，且解绑也会改动 rtModel 的流表。
    // 同步不设超时：超时后仍会继续解绑销毁、并由 DestroyResources 释放模型内存，等于把阻塞换成设备侧写已释放内存。
    // 时序与 Rt2AttachedStreamCollection::Destroy 一致（RT2 无 rtModel，故少一步解绑）。
    GELOGI("Synchronizing eager attached stream before destroy, key=%s.", item.first.c_str());
    GE_LOGW_IF(aclrtSynchronizeStream(item.second) != ACL_SUCCESS,
               "Failed to synchronize eager attached stream, key=%s.", item.first.c_str());
    if (model_ != nullptr) {
      GE_LOGW_IF(aclmdlRIUnbindStream(model_, item.second) != ACL_SUCCESS,
                 "Failed to unbind eager attached stream, key=%s.", item.first.c_str());
    }
    GE_LOGW_IF(aclrtDestroyStream(item.second) != ACL_SUCCESS, "Failed to destroy eager attached stream, key=%s.",
               item.first.c_str());
  }
  streams_.clear();
  model_ = nullptr;
}
}  // namespace ge
