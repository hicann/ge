/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "framework/runtime/gert_model/gert_model_executor_callbacks.h"

#include <mutex>
#include <string>
#include <unordered_map>

#include "common/ge_common/debug/ge_log.h"

namespace ge {
struct BinHandleInfo {
  aclrtBinHandle bin_handle{nullptr};
  int64_t refer_count{0};
};

class Om2KernelHandlesManager {
 public:
  static Om2KernelHandlesManager &Instance() {
    static Om2KernelHandlesManager manager;
    return manager;
  }

  int32_t LockBinHandleStore() {
    GELOGD("[OM2][KernelHandles] Lock bin handle store.");
    mutex_.lock();
    return 0;
  }

  int32_t UnlockBinHandleStore() {
    GELOGD("[OM2][KernelHandles] Unlock bin handle store.");
    mutex_.unlock();
    return 0;
  }

  int32_t QueryBinHandleFromStore(const char *bin_id, aclrtBinHandle *bin_handle) {
    if (bin_id == nullptr) {
      return -1;
    }
    GELOGD("[OM2][KernelHandles] Query bin handle from store, bin_id=%s.", bin_id);
    const std::string bin_id_str(bin_id);
    auto iter = global_bin_handle_store_.find(bin_id_str);
    if (iter != global_bin_handle_store_.end()) {
      *bin_handle = iter->second.bin_handle;
      return 0;
    }
    *bin_handle = nullptr;
    return 0;
  }

  int32_t ReleaseBinHandleFromStore(const char *bin_id, uint8_t *need_unload) {
    if (bin_id == nullptr || need_unload == nullptr) {
      return -1;
    }
    *need_unload = 0U;
    GELOGD("[OM2][KernelHandles] Release bin handle from store, bin_id=%s.", bin_id);
    const std::string bin_id_str(bin_id);
    auto iter = global_bin_handle_store_.find(bin_id_str);
    if (iter != global_bin_handle_store_.end()) {
      iter->second.refer_count--;
      if (iter->second.refer_count <= 0) {
        *need_unload = 1U;
        (void)global_bin_handle_store_.erase(iter);
        return 0;
      }
      return 0;
    }
    return 0;
  }

  int32_t SaveBinHandleToStore(const char *bin_id, const aclrtBinHandle bin_handle) {
    if (bin_id == nullptr) {
      return -1;
    }
    const std::string bin_id_str(bin_id);
    GELOGD("[OM2][KernelHandles] Save bin handle to store, bin_id=%s.", bin_id);
    auto iter = global_bin_handle_store_.find(bin_id_str);
    if (iter != global_bin_handle_store_.end()) {
      iter->second.refer_count++;
      return 0;
    }
    BinHandleInfo kernel_bin_handle;
    kernel_bin_handle.bin_handle = bin_handle;
    kernel_bin_handle.refer_count = 1;
    global_bin_handle_store_.emplace(bin_id_str, kernel_bin_handle);
    return 0;
  }

 private:
  Om2KernelHandlesManager() = default;

  std::recursive_mutex mutex_;
  std::unordered_map<std::string, BinHandleInfo> global_bin_handle_store_;
};
}  // namespace ge

extern "C" {
int32_t LockBinHandleStore() {
  return ge::Om2KernelHandlesManager::Instance().LockBinHandleStore();
}

int32_t UnlockBinHandleStore() {
  return ge::Om2KernelHandlesManager::Instance().UnlockBinHandleStore();
}

int32_t QueryBinHandleFromStore(const char *bin_id, aclrtBinHandle *bin_handle) {
  return ge::Om2KernelHandlesManager::Instance().QueryBinHandleFromStore(bin_id, bin_handle);
}

int32_t ReleaseBinHandleFromStore(const char *bin_id, uint8_t *need_unload) {
  return ge::Om2KernelHandlesManager::Instance().ReleaseBinHandleFromStore(bin_id, need_unload);
}

int32_t SaveBinHandleToStore(const char *bin_id, const aclrtBinHandle bin_handle) {
  return ge::Om2KernelHandlesManager::Instance().SaveBinHandleToStore(bin_id, bin_handle);
}

}  // extern "C"
