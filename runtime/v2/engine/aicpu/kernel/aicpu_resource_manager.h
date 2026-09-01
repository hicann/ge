/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_CXX_RUNTIME_V2_KERNEL_LAUNCH_KERNEL_AICPU_AICPU_RESOURCE_MANAGER_H_
#define AIR_CXX_RUNTIME_V2_KERNEL_LAUNCH_KERNEL_AICPU_AICPU_RESOURCE_MANAGER_H_

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <deque>
#include <map>
#include <vector>
#include "ge/ge_api_error_codes.h"
#include "exe_graph/runtime/tensor_data.h"
#include "graph/op_desc.h"
#include "graph/op_kernel_bin.h"
#include "common/tbe_handle_store/kernel_store.h"
#include "exe_graph/runtime/gert_tensor_data.h"
#include "rt_external_base.h"
#include "common/ge_rts_decl.h"
#include "exe_graph/runtime/kernel_context.h"

namespace gert {

using AicpuHostProcFunc = ge::graphStatus (*)(KernelContext *);
using FusedHostCpuCreateFunc = void *(*)();
using FusedHostCpuDestroyFunc = void (*)(void *);
using FusedHostCpuRunFunc = uint32_t (*)(void *, const void *, uint32_t);

struct FusedHostCpuKernelFunctions {
  FusedHostCpuCreateFunc create_func = nullptr;
  FusedHostCpuDestroyFunc destroy_func = nullptr;
  FusedHostCpuRunFunc run_func = nullptr;
};

class AicpuResourceManager {
 public:
  static AicpuResourceManager &GetInstance();
  ~AicpuResourceManager();

  ge::graphStatus LoadConstantFoldingLib();
  ge::graphStatus LoadFusedHostCpuSo(const std::string &register_name, const uint8_t *so_data, size_t so_size);
  ge::graphStatus ReleaseFusedHostCpuSo(const std::string &register_name);
  FusedHostCpuKernelFunctions GetFusedHostCpuKernelFunctions(const std::string &register_name);
  std::function<uint32_t(void *)> GetRunCpuKernel() const;
  std::function<AicpuHostProcFunc(std::string)> GetAicpuHostFindFunc() const;

  bool IsSingleOp() const {
    return is_single_op_;
  }

  void SetSingleOp(const bool is_single_op) {
    const std::lock_guard<std::mutex> lk(mutex_);
    is_single_op_ = is_single_op;
  }

  ge::graphStatus PushTensor(const std::string &op_name, const rtStream_t stream, const GertTensorData *tensor_data,
                             const GertTensorData *handle_data);
  ge::graphStatus PopTensor(const std::string &op_name, const rtStream_t stream, const GertTensorData *handle_data);
  void ClearTensors();
  ge::graphStatus HasLoadedCustAicpuSo(const std::string &so_name, bool &loaded);

 private:
  ge::graphStatus TryReuseFusedHostCpuSo(const std::string &register_name, const uint8_t *so_data, size_t so_size,
                                         uint64_t so_hash, bool &handled);
  ge::graphStatus LoadNewFusedHostCpuSo(const std::string &register_name, const uint8_t *so_data, size_t so_size,
                                        uint64_t so_hash);
  ge::graphStatus OpenFusedHostCpuSo(const std::string &register_name, const uint8_t *so_data, size_t so_size,
                                     void *&handle, int &fd, FusedHostCpuKernelFunctions &kernel_funcs);
  ge::graphStatus CheckOrCreateHandle(const std::string &op_name, const rtStream_t stream,
                                      const GertTensorData *handle_data);
  AicpuResourceManager() = default;

  bool is_single_op_ = false;
  std::mutex mutex_;
  std::function<uint32_t(void *)> run_cpu_kernel_ = nullptr;
  std::function<AicpuHostProcFunc(std::string)> aicpu_host_find_func_ = nullptr;
  void *so_handle_ = nullptr;
  // CpuKernelRegister 没有注销接口。引用计数仅表示活跃模型数；引用归零后仍保留 so 映射，避免其中的
  // std::function creator 因 dlclose 变成悬空指针，并支持后续模型复用同一产物。
  std::mutex fused_so_mutex_;
  std::map<std::string, uint64_t> fused_register_hashes_;
  std::map<std::string, size_t> fused_register_ref_counts_;
  std::map<std::string, FusedHostCpuKernelFunctions> fused_kernel_funcs_;
  std::map<uint64_t, void *> fused_so_handles_;
  std::map<uint64_t, int> fused_so_fds_;
  std::map<uint64_t, size_t> fused_so_ref_counts_;
  std::map<uint64_t, std::vector<uint8_t>> fused_so_contents_;
  std::vector<void *> fused_quarantined_so_handles_;
  std::vector<int> fused_quarantined_so_fds_;

  std::map<uint64_t, std::deque<GertTensorData>> tensors_;
  std::map<std::string, uint64_t> handles_;
  std::mutex cust_aicpu_so_mutex_;
  std::map<uintptr_t, std::string> cust_aicpu_context_so_;
};
}  // namespace gert
#endif  // AIR_CXX_RUNTIME_V2_KERNEL_LAUNCH_KERNEL_AICPU_AICPU_RESOURCE_MANAGER_H_
