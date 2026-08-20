/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "om2_rt_var_manager.h"

#include <cinttypes>
#include <numeric>

#include "acl/acl_rt.h"
#include "common/checker.h"
#include "common/ge_inner_error_codes.h"
#include "common/ge_common/debug/ge_log.h"
#include "om2_thread_pool.h"
#include "graph_metadef/common/ge_common/util.h"
#include "formats/om2_formats.h"
#include "common/datatype_transfer/om2_datatype_transfer.h"
#include "om2_malloc_helper.h"
#include "rt_external_mem.h"

namespace gert {

namespace {
constexpr uint32_t kDefaultVarTransThreadNum = 16U;

bool IsNoNeedTrans(const std::string &node_type) {
  return (node_type == "RESHAPE") || (node_type == "REFORMAT") || (node_type == "SQUEEZEV2") ||
         (node_type == "UNSQUEEZEV2");
}

bool NeedRealTrans(const RTVarTransRoad &trans_road) {
  for (const auto &node : trans_road) {
    if (!IsNoNeedTrans(node.node_type)) {
      return true;
    }
  }
  return false;
}
}  // namespace

Om2RTVarManager::~Om2RTVarManager() {
  Finalize();
}

ge::Status Om2RTVarManager::Init(const RTVarResource &resource, void *external_var_addr, uint64_t external_var_size) {
  const std::lock_guard<std::recursive_mutex> lock(mutex_);
  external_var_addr_ = external_var_addr;
  external_var_size_ = external_var_size;
  for (const auto &[var_key, new_entry] : resource.GetAllEntries()) {
    if (var_resource_.GetEntry(var_key) != nullptr) {
      continue;
    }
    GE_RETURN_IF_ERROR(var_resource_.AddEntry(new_entry));
  }
  return ge::SUCCESS;
}

ge::Status Om2RTVarManager::AllocDevAddr(const RTVarEntry &entry, void *&dev_addr) {
  void *new_addr = nullptr;
  const auto malloc_ret = Om2Malloc(&new_addr, entry.size, entry.memory_type, 0);
  if (malloc_ret != ACL_SUCCESS) {
    GELOGE(ge::FAILED, "[OM2][Var][Alloc] aclrtMalloc failed, var=%s, size=%" PRIu64 ", ret=%u.",
           entry.var_name.c_str(), entry.size, malloc_ret);
    return ge::FAILED;
  }
  dev_addr = new_addr;
  return ge::SUCCESS;
}

RTVarRuntimeState &Om2RTVarManager::GetOrCreateRuntimeState(const std::string &var_key) {
  return var_runtime_states_[var_key];
}

const RTVarRuntimeState *Om2RTVarManager::GetRuntimeState(const std::string &var_key) const {
  auto it = var_runtime_states_.find(var_key);
  if (it == var_runtime_states_.end()) {
    return nullptr;
  }
  return &it->second;
}

ge::Status Om2RTVarManager::GetVarDevAddr(const std::string &var_name, uint32_t device_id, void *&dev_addr) {
  const auto *entry = var_resource_.GetEntryByName(var_name);
  if (entry == nullptr) {
    GELOGE(ge::PARAM_INVALID, "[OM2][Var] var_name=%s not found.", var_name.c_str());
    return ge::PARAM_INVALID;
  }
  return GetVarDevAddr(*entry, device_id, dev_addr);
}

ge::Status Om2RTVarManager::GetVarDevAddr(const RTVarEntry &entry, uint32_t device_id, void *&dev_addr) {
  const std::lock_guard<std::recursive_mutex> lock(mutex_);
  auto &state = GetOrCreateRuntimeState(entry.var_key);
  auto it = state.dev_addrs.find(device_id);
  if (it != state.dev_addrs.end()) {
    dev_addr = it->second;
    return ge::SUCCESS;
  }

  void *new_addr = nullptr;
  if (entry.extern_dev_addr != nullptr) {
    new_addr = entry.extern_dev_addr;
  } else if (external_var_addr_ != nullptr) {
    if (entry.logic_addr < logic_var_base_) {
      GELOGE(ge::FAILED, "[OM2][Var][Alloc] logic_addr %" PRIu64 " < base %" PRIu64 " for var=%s.", entry.logic_addr,
             logic_var_base_, entry.var_name.c_str());
      return ge::FAILED;
    }
    const uint64_t offset = entry.logic_addr - logic_var_base_;
    if (offset > external_var_size_ || entry.size > (external_var_size_ - offset)) {
      GELOGE(ge::FAILED,
             "[OM2][Var][Alloc] external arena overflow, var=%s, offset=%" PRIu64 ", size=%" PRIu64 ", arena=%" PRIu64
             ".",
             entry.var_name.c_str(), offset, entry.size, external_var_size_);
      return ge::FAILED;
    }
    new_addr = static_cast<uint8_t *>(external_var_addr_) + offset;
  } else {
    GE_RETURN_IF_ERROR(AllocDevAddr(entry, new_addr));
  }

  state.dev_addrs[device_id] = new_addr;
  dev_addr = new_addr;

  if (!entry.init_data.empty() && !state.is_loaded[device_id]) {
    if (entry.init_data.size() > entry.size) {
      GELOGE(ge::FAILED, "[OM2][Var][Alloc] init_data size %zu > entry size %" PRIu64 " for var=%s.",
             entry.init_data.size(), entry.size, entry.var_name.c_str());
      return ge::FAILED;
    }
    const auto ret =
        aclrtMemcpy(new_addr, entry.size, entry.init_data.data(), entry.init_data.size(), ACL_MEMCPY_HOST_TO_DEVICE);
    if (ret != ACL_SUCCESS) {
      GELOGE(ge::FAILED, "[OM2][Var][Alloc] init H2D failed, var=%s, ret=%u.", entry.var_name.c_str(), ret);
      return ge::FAILED;
    }
    state.is_loaded[device_id] = true;
  }

  return ge::SUCCESS;
}

const RTVarResource *Om2RTVarManager::GetVarResource() const {
  return &var_resource_;
}

void Om2RTVarManager::Finalize() noexcept {
  const std::lock_guard<std::recursive_mutex> lock(mutex_);
  for (auto &[var_key, state] : var_runtime_states_) {
    const auto *entry = var_resource_.GetEntry(var_key);
    if (entry == nullptr) {
      GELOGW("[OM2][Var][Finalize] var_key=%s not found in var_resource_, skip free.", var_key.c_str());
      continue;
    }
    for (auto &[_, addr] : state.dev_addrs) {
      (void)_;
      if (addr != nullptr && addr != entry->extern_dev_addr) {
        (void)aclrtFree(addr);
      }
    }
  }
  var_runtime_states_.clear();
  for (auto &[_, vars] : legacy_device_to_vars_) {
    (void)_;
    for (auto &[__, info] : vars) {
      (void)__;
      if (info.addr != nullptr) {
        (void)aclrtFree(info.addr);
      }
    }
  }
  legacy_device_to_vars_.clear();
}

ge::Status Om2RTVarManager::GetOrCreateVarAddr(const std::string &key, uint32_t device_id, size_t size, void *&addr) {
  addr = nullptr;
  if (size == 0U) {
    return ge::SUCCESS;
  }
  {
    const std::lock_guard<std::recursive_mutex> lock(mutex_);
    auto dev_it = legacy_device_to_vars_.find(device_id);
    if (dev_it != legacy_device_to_vars_.end()) {
      auto var_it = dev_it->second.find(key);
      if (var_it != dev_it->second.end()) {
        if (var_it->second.size != size) {
          GELOGE(ge::FAILED, "[OM2][Var][Legacy] size mismatch for key=%s, cached=%zu, requested=%zu.", key.c_str(),
                 var_it->second.size, size);
          return ge::FAILED;
        }
        addr = var_it->second.addr;
        return ge::SUCCESS;
      }
    }
  }
  void *new_addr = nullptr;
  const auto malloc_ret = Om2Malloc(&new_addr, size, RT_MEMORY_HBM, 0);
  if (malloc_ret != ACL_SUCCESS) {
    GELOGE(ge::FAILED, "[OM2][Var][Alloc] legacy malloc failed, key=%s, size=%zu.", key.c_str(), size);
    return ge::FAILED;
  }
  {
    const std::lock_guard<std::recursive_mutex> lock(mutex_);
    auto &dev_vars = legacy_device_to_vars_[device_id];
    auto var_it = dev_vars.find(key);
    if (var_it != dev_vars.end()) {
      if (var_it->second.size != size) {
        (void)aclrtFree(new_addr);
        GELOGE(ge::FAILED, "[OM2][Var][Legacy] size mismatch for key=%s, cached=%zu, requested=%zu.", key.c_str(),
               var_it->second.size, size);
        return ge::FAILED;
      }
      addr = var_it->second.addr;
      (void)aclrtFree(new_addr);
      return ge::SUCCESS;
    }
    dev_vars[key] = LegacyVarAddrInfo{new_addr, size};
    addr = new_addr;
  }
  return ge::SUCCESS;
}

bool Om2RTVarManager::TryGetVarAddr(const std::string &key, uint32_t device_id, void *&addr) const {
  const std::lock_guard<std::recursive_mutex> lock(mutex_);
  addr = nullptr;
  auto dev_it = legacy_device_to_vars_.find(device_id);
  if (dev_it == legacy_device_to_vars_.end()) {
    return false;
  }
  auto var_it = dev_it->second.find(key);
  if (var_it == dev_it->second.end()) {
    return false;
  }
  addr = var_it->second.addr;
  return addr != nullptr;
}

ge::Status Om2RTVarManager::CopyVarFromDevice(const RTVarEntry &entry, const RTVarRuntimeState &state,
                                              uint32_t device_id, std::vector<uint8_t> &host_buf) {
  auto it = state.dev_addrs.find(device_id);
  if (it == state.dev_addrs.end() || it->second == nullptr) {
    GELOGE(ge::FAILED, "[OM2][Var] dev_addr not allocated for var=%s, device=%u.", entry.var_name.c_str(), device_id);
    return ge::FAILED;
  }
  host_buf.resize(entry.size);
  const auto ret = aclrtMemcpy(host_buf.data(), entry.size, it->second, entry.size, ACL_MEMCPY_DEVICE_TO_HOST);
  if (ret != ACL_SUCCESS) {
    GELOGE(ge::FAILED, "[OM2][Var][D2H] aclrtMemcpy failed, var=%s, ret=%u.", entry.var_name.c_str(), ret);
    return ge::FAILED;
  }
  return ge::SUCCESS;
}

ge::Status Om2RTVarManager::CopyVarToDevice(const RTVarEntry &entry, const RTVarRuntimeState &state, uint32_t device_id,
                                            const std::vector<uint8_t> &host_buf) {
  auto it = state.dev_addrs.find(device_id);
  if (it == state.dev_addrs.end() || it->second == nullptr) {
    GELOGE(ge::FAILED, "[OM2][Var] dev_addr not allocated for var=%s, device=%u.", entry.var_name.c_str(), device_id);
    return ge::FAILED;
  }
  const auto ret = aclrtMemcpy(it->second, entry.size, host_buf.data(), host_buf.size(), ACL_MEMCPY_HOST_TO_DEVICE);
  if (ret != ACL_SUCCESS) {
    GELOGE(ge::FAILED, "[OM2][Var][H2D] aclrtMemcpy failed, var=%s, ret=%u.", entry.var_name.c_str(), ret);
    return ge::FAILED;
  }
  return ge::SUCCESS;
}

ge::Status Om2RTVarManager::TransVarOnHost(const RTVarTransRoad &trans_road, std::vector<uint8_t> &data) {
  ge::formats::TransResult last_result{};
  bool use_init_data = true;
  for (const auto &node : trans_road) {
    if (IsNoNeedTrans(node.node_type)) {
      continue;
    }
    uint8_t *src_data = nullptr;
    if (use_init_data) {
      src_data = data.data();
      use_init_data = false;
    } else {
      src_data = last_result.data.get();
    }

    ge::formats::TransResult tmp_result{};
    if (node.node_type == "TRANSDATA" || node.node_type == "TRANSPOSED") {
      const auto src_format = node.input.GetFormat();
      const auto dst_format = node.output.GetFormat();
      const auto src_shape = node.input.GetShape();
      const auto dst_shape = node.output.GetShape();
      const auto data_type = node.input.GetDataType();
      const ge::Format src_primary = static_cast<ge::Format>(ge::GetPrimaryFormat(static_cast<int32_t>(src_format)));
      const ge::Format dst_primary = static_cast<ge::Format>(ge::GetPrimaryFormat(static_cast<int32_t>(dst_format)));
      const ge::Format src_sub = static_cast<ge::Format>(ge::GetSubFormat(static_cast<int32_t>(src_format)));
      const ge::Format dst_sub = static_cast<ge::Format>(ge::GetSubFormat(static_cast<int32_t>(dst_format)));
      const int64_t src_c0 = ge::GetC0Value(static_cast<int32_t>(src_format));
      const int64_t dst_c0 = ge::GetC0Value(static_cast<int32_t>(dst_format));
      const auto ret = ge::formats::TransDataFormat({src_data, src_format, dst_format, src_primary, dst_primary,
                                                     src_sub, dst_sub, src_c0, dst_c0, src_shape, dst_shape, data_type},
                                                    tmp_result);
      if (ret != ge::SUCCESS) {
        GELOGE(ge::FAILED, "[OM2][Var][Trans] TransDataFormat failed, %s, dst_format=%d, ret=%u.",
               node.node_type.c_str(), static_cast<int>(dst_format), ret);
        return ret;
      }
    } else if (node.node_type == "CAST") {
      const auto &src_shape = node.input.GetShape();
      int64_t element_count = 1;
      for (const auto dim : src_shape) {
        element_count *= dim;
      }
      if (element_count == 0) {
        element_count = 1;
      }
      const auto src_dtype = node.input.GetDataType();
      const auto dst_dtype = node.output.GetDataType();
      const auto ret = ge::formats::TransTensorDataType(
          {src_data, static_cast<size_t>(element_count), src_dtype, dst_dtype}, tmp_result);
      if (ret != ge::SUCCESS) {
        GELOGE(ge::FAILED, "[OM2][Var][Trans] TransTensorDataType failed, ret=%u.", ret);
        return ret;
      }
    } else {
      GELOGE(ge::UNSUPPORTED, "[OM2][Var][Trans] unsupported node_type=%s.", node.node_type.c_str());
      return ge::UNSUPPORTED;
    }
    last_result = tmp_result;
  }

  if (last_result.data != nullptr && last_result.length > 0U) {
    data.assign(last_result.data.get(), last_result.data.get() + last_result.length);
  }
  return ge::SUCCESS;
}

ge::Status Om2RTVarManager::TransSingleVarData(const std::string &var_name, uint32_t device_id) {
  const auto *entry = var_resource_.GetEntryByName(var_name);
  if (entry == nullptr) {
    return ge::FAILED;
  }

  const auto &input_desc = entry->trans_road.front().input;
  const std::string old_var_key = RTVarResource::BuildVarKey(var_name, input_desc);
  const auto *old_entry = var_resource_.GetEntry(old_var_key);
  if (old_entry == nullptr) {
    GELOGW("[OM2][Var][Trans] old entry not found for var=%s, key=%s.", var_name.c_str(), old_var_key.c_str());
    return ge::FAILED;
  }

  void *old_dev_addr = nullptr;
  GE_RETURN_IF_ERROR(GetVarDevAddr(*old_entry, device_id, old_dev_addr));

  const auto *old_state = GetRuntimeState(old_entry->var_key);
  if (old_state == nullptr) {
    GELOGE(ge::FAILED, "[OM2][Var][Trans] old state not found for var=%s.", var_name.c_str());
    return ge::FAILED;
  }

  std::vector<uint8_t> host_buf;
  GE_RETURN_IF_ERROR(CopyVarFromDevice(*old_entry, *old_state, device_id, host_buf));
  GE_RETURN_IF_ERROR(TransVarOnHost(entry->trans_road, host_buf));

  void *new_dev_addr = nullptr;
  GE_RETURN_IF_ERROR(GetVarDevAddr(*entry, device_id, new_dev_addr));

  auto &new_state = GetOrCreateRuntimeState(entry->var_key);
  GE_RETURN_IF_ERROR(CopyVarToDevice(*entry, new_state, device_id, host_buf));

  new_state.is_loaded[device_id] = true;
  GELOGI("[OM2][Var][Trans] var=%s trans completed on device=%u.", var_name.c_str(), device_id);
  return ge::SUCCESS;
}

ge::Status Om2RTVarManager::TransAllVarData(const std::vector<std::string> &var_names, uint32_t device_id,
                                            uint32_t graph_id) {
  if (var_names.empty()) {
    return ge::SUCCESS;
  }

  std::vector<std::string> vars_to_trans;
  for (const auto &var_name : var_names) {
    const auto *entry = var_resource_.GetEntryByName(var_name);
    if (entry == nullptr || entry->trans_road.empty()) {
      continue;
    }
    if (entry->changed_graph_id != graph_id || entry->changed_graph_id == entry->allocated_graph_id) {
      continue;
    }
    const auto *state = GetRuntimeState(entry->var_key);
    if (state != nullptr) {
      auto loaded_it = state->is_loaded.find(device_id);
      if (loaded_it != state->is_loaded.end() && loaded_it->second) {
        continue;
      }
    }
    if (!NeedRealTrans(entry->trans_road)) {
      continue;
    }
    vars_to_trans.push_back(var_name);
  }

  if (vars_to_trans.empty()) {
    return ge::SUCCESS;
  }

  aclrtContext context = nullptr;
  if (aclrtGetCurrentContext(&context) != ACL_SUCCESS) {
    GELOGE(ge::FAILED, "[OM2][Var][Trans] get current context failed.");
    return ge::FAILED;
  }

  ge::om2::ThreadPool executor("om2_vartrans", kDefaultVarTransThreadNum);
  std::vector<std::future<ge::Status>> futures;

  for (const auto &var_name : vars_to_trans) {
    auto trans_func = [this, var_name, device_id, context]() -> ge::Status {
      if (aclrtSetCurrentContext(context) != ACL_SUCCESS) {
        GELOGE(ge::FAILED, "[OM2][Var][Trans] set context failed for var=%s.", var_name.c_str());
        return ge::FAILED;
      }
      return TransSingleVarData(var_name, device_id);
    };
    auto f = executor.commit(trans_func);
    if (!f.valid()) {
      GELOGE(ge::FAILED, "[OM2][Var][Trans] commit task failed for var=%s.", var_name.c_str());
      return ge::FAILED;
    }
    futures.push_back(std::move(f));
  }

  for (auto &f : futures) {
    const auto ret = f.get();
    if (ret != ge::SUCCESS) {
      return ret;
    }
  }

  return ge::SUCCESS;
}

ge::Status Om2RTVarManager::CopyVarData(const std::vector<std::string> &var_names, uint32_t device_id) {
  for (const auto &var_name : var_names) {
    const auto *entry = var_resource_.GetEntryByName(var_name);
    if (entry == nullptr) {
      continue;
    }
    if (entry->copy_info.src_var_name.empty()) {
      continue;
    }
    const auto *dst_state = GetRuntimeState(entry->var_key);
    if (dst_state != nullptr) {
      auto loaded_it = dst_state->is_loaded.find(device_id);
      if (loaded_it != dst_state->is_loaded.end() && loaded_it->second) {
        continue;
      }
    }

    const std::string src_var_key =
        RTVarResource::BuildVarKey(entry->copy_info.src_var_name, entry->copy_info.src_tensor_desc);
    const auto *src_entry = var_resource_.GetEntry(src_var_key);
    if (src_entry == nullptr) {
      GELOGW("[OM2][Var][Copy] src entry not found, var=%s, src_key=%s.", var_name.c_str(), src_var_key.c_str());
      continue;
    }

    void *src_addr = nullptr;
    GE_RETURN_IF_ERROR(GetVarDevAddr(*src_entry, device_id, src_addr));

    const auto *src_state = GetRuntimeState(src_entry->var_key);
    if (src_state == nullptr) {
      GELOGE(ge::FAILED, "[OM2][Var][Copy] src state not found, var=%s.", var_name.c_str());
      return ge::FAILED;
    }

    std::vector<uint8_t> host_buf;
    GE_RETURN_IF_ERROR(CopyVarFromDevice(*src_entry, *src_state, device_id, host_buf));

    if (src_entry->tensor_desc.GetDataType() != entry->tensor_desc.GetDataType()) {
      int64_t element_count = 1;
      for (const auto dim : src_entry->tensor_desc.GetShape()) {
        element_count *= dim;
      }
      if (element_count == 0) {
        element_count = 1;
      }
      ge::formats::TransResult cast_result{};
      const auto cast_ret =
          ge::formats::TransTensorDataType({host_buf.data(), static_cast<size_t>(element_count),
                                            src_entry->tensor_desc.GetDataType(), entry->tensor_desc.GetDataType()},
                                           cast_result);
      if (cast_ret != ge::SUCCESS) {
        GELOGE(ge::FAILED, "[OM2][Var][Copy] dtype cast failed, var=%s, ret=%u.", var_name.c_str(), cast_ret);
        return cast_ret;
      }
      host_buf.assign(cast_result.data.get(), cast_result.data.get() + cast_result.length);
    }

    void *dst_addr = nullptr;
    GE_RETURN_IF_ERROR(GetVarDevAddr(*entry, device_id, dst_addr));

    auto &new_dst_state = GetOrCreateRuntimeState(entry->var_key);
    GE_RETURN_IF_ERROR(CopyVarToDevice(*entry, new_dst_state, device_id, host_buf));

    new_dst_state.is_loaded[device_id] = true;
    GELOGI("[OM2][Var][Copy] var=%s copied from src=%s on device=%u.", var_name.c_str(),
           entry->copy_info.src_var_name.c_str(), device_id);
  }
  return ge::SUCCESS;
}

Om2RTVarManagerPool &Om2RTVarManagerPool::Instance() {
  static Om2RTVarManagerPool pool;
  return pool;
}

Om2RTVarManagerPool::~Om2RTVarManagerPool() {
  Destroy();
}

Om2RTVarManagerPtr Om2RTVarManagerPool::GetManager(uint64_t session_id) {
  const std::lock_guard<std::mutex> lock(mutex_);
  auto &manager = session_id_to_manager_[session_id];
  if (manager == nullptr) {
    manager = std::make_shared<Om2RTVarManager>();
    if (manager == nullptr) {
      GELOGE(ge::INTERNAL_ERROR, "[OM2][New][RTVarManager] failed, session_id=%" PRIu64 ".", session_id);
      return nullptr;
    }
  }
  return manager;
}

void Om2RTVarManagerPool::RemoveManager(uint64_t session_id) {
  Om2RTVarManagerPtr manager = nullptr;
  {
    const std::lock_guard<std::mutex> lock(mutex_);
    auto it = session_id_to_manager_.find(session_id);
    if (it != session_id_to_manager_.end()) {
      manager = it->second;
      (void)session_id_to_manager_.erase(it);
    }
  }
  if (manager != nullptr) {
    manager->Finalize();
  }
}

void Om2RTVarManagerPool::Destroy() noexcept {
  const std::lock_guard<std::mutex> lock(mutex_);
  for (const auto &[_, mgr] : session_id_to_manager_) {
    (void)_;
    if (mgr != nullptr) {
      mgr->Finalize();
    }
  }
  session_id_to_manager_.clear();
}

}  // namespace gert
