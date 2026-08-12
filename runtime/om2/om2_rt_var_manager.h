/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_RUNTIME_OM2_RT_VAR_MANAGER_H_
#define AIR_RUNTIME_OM2_RT_VAR_MANAGER_H_

#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

#include "common/om2/rt_var_resource.h"

constexpr uint64_t kMemoryVarLogicBase = 137438953472U;  // 128GB

namespace gert {

struct RTVarRuntimeState {
  std::map<uint32_t, void *> dev_addrs;
  std::map<uint32_t, bool> is_loaded;
};

class Om2RTVarManager {
 public:
  Om2RTVarManager() = default;
  ~Om2RTVarManager();

  ge::Status Init(const RTVarResource &resource, void *external_var_addr = nullptr, uint64_t external_var_size = 0);
  ge::Status GetVarDevAddr(const std::string &var_name, uint32_t device_id, void *&dev_addr);
  ge::Status GetVarDevAddr(const RTVarEntry &entry, uint32_t device_id, void *&dev_addr);
  ge::Status TransAllVarData(const std::vector<std::string> &var_names, uint32_t device_id, uint32_t graph_id);
  ge::Status CopyVarData(const std::vector<std::string> &var_names, uint32_t device_id);
  ge::Status GetOrCreateVarAddr(const std::string &key, uint32_t device_id, size_t size, void *&addr);
  bool TryGetVarAddr(const std::string &key, uint32_t device_id, void *&addr) const;
  const RTVarResource *GetVarResource() const;
  void Finalize() noexcept;

 private:
  ge::Status AllocDevAddr(const RTVarEntry &entry, void *&dev_addr);
  ge::Status CopyVarFromDevice(const RTVarEntry &entry, const RTVarRuntimeState &state, uint32_t device_id,
                               std::vector<uint8_t> &host_buf);
  ge::Status CopyVarToDevice(const RTVarEntry &entry, const RTVarRuntimeState &state, uint32_t device_id,
                             const std::vector<uint8_t> &host_buf);
  ge::Status TransVarOnHost(const RTVarTransRoad &trans_road, std::vector<uint8_t> &data);
  ge::Status TransSingleVarData(const std::string &var_name, uint32_t device_id);

  RTVarRuntimeState &GetOrCreateRuntimeState(const std::string &var_key);
  const RTVarRuntimeState *GetRuntimeState(const std::string &var_key) const;

  uint64_t logic_var_base_{kMemoryVarLogicBase};
  RTVarResource var_resource_;
  std::unordered_map<std::string, RTVarRuntimeState> var_runtime_states_;
  void *external_var_addr_{nullptr};
  uint64_t external_var_size_{0};
  struct LegacyVarAddrInfo {
    void *addr = nullptr;
    size_t size = 0U;
  };

  mutable std::recursive_mutex mutex_;
  std::map<uint32_t, std::map<std::string, LegacyVarAddrInfo>> legacy_device_to_vars_;
};

using Om2RTVarManagerPtr = std::shared_ptr<Om2RTVarManager>;

class __attribute__((visibility("default"))) Om2RTVarManagerPool {
 public:
  static Om2RTVarManagerPool &Instance();
  ~Om2RTVarManagerPool();
  Om2RTVarManagerPtr GetManager(uint64_t session_id);
  void RemoveManager(uint64_t session_id);
  void Destroy() noexcept;

 private:
  Om2RTVarManagerPool() = default;
  std::mutex mutex_;
  std::map<uint64_t, Om2RTVarManagerPtr> session_id_to_manager_;
};

}  // namespace gert

#endif  // AIR_RUNTIME_OM2_RT_VAR_MANAGER_H_
