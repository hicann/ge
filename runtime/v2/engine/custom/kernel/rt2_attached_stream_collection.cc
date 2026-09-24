/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "rt2_attached_stream_collection.h"

#include <string_view>

#include "acl/acl_rt.h"
#include "framework/common/debug/ge_log.h"
#include "graph/ascend_string.h"

namespace gert {
rtStream Rt2AttachedStreamCollection::RequestAttachedStream(const ge::AscendString &key) {
  // 用 GetString 判空而不用 GetLength：stub 版 libgraph.so 的 GetLength 恒返回 0
  const char *const key_str = key.GetString();
  if ((key_str == nullptr) || (key_str[0] == '\0')) {
    GELOGE(ge::PARAM_INVALID, "Invalid eager attached stream key: key is null or empty.");
    return nullptr;
  }
  const auto found = streams_.find(std::string_view(key_str));
  if (found != streams_.end()) {
    return found->second;
  }
  rtStream_t stream = nullptr;
  if (rtStreamCreateWithFlags(&stream, RT_STREAM_PRIORITY_DEFAULT, RT_STREAM_DEFAULT) != RT_ERROR_NONE) {
    GELOGE(ge::FAILED, "Failed to create eager attached stream, key=%s.", key_str);
    return nullptr;
  }
  (void)streams_.emplace(key_str, stream);
  GELOGI("Created eager attached stream, key=%s, total=%zu.", key_str, streams_.size());
  return stream;
}

void Rt2AttachedStreamCollection::Destroy() {
  for (const auto &item : streams_) {
    // 先流同步再销毁：aclrtDestroyStream 的前置约束要求流上任务已执行完。
    // 同步不设超时：超时后仍会继续销毁并释放执行器内存，等于把阻塞换成设备侧写已释放内存。
    // 时序与 V1 的 ge::AttachedStreamCollection::UnbindAndDestroy 一致（V1 多一步 rtModel 解绑）。
    GELOGI("Synchronizing eager attached stream before destroy, key=%s.", item.first.c_str());
    if (aclrtSynchronizeStream(item.second) != ACL_SUCCESS) {
      GELOGW("Failed to synchronize eager attached stream, key=%s.", item.first.c_str());
    }
    if (aclrtDestroyStream(item.second) != ACL_SUCCESS) {
      GELOGW("Failed to destroy eager attached stream, key=%s.", item.first.c_str());
    }
  }
  streams_.clear();
}
}  // namespace gert
