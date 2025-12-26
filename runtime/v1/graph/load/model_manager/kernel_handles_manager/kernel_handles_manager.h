/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef EXECUTOR_GRAPH_LOAD_MODEL_MANAGER_KERNEL_MANAGER_KERBEL_HANDLES_MANAGER_H
#define EXECUTOR_GRAPH_LOAD_MODEL_MANAGER_KERNEL_MANAGER_KERBEL_HANDLES_MANAGER_H

#include <mutex>
#include <variant>
#include <unordered_map>
#include "rts/rts_kernel.h"
#include "graph/ge_error_codes.h"
#include "common/tbe_handle_store/kernel_store.h"

namespace ge {
struct KernelBinInfo {
  rtBinHandle bin_handle{nullptr};
  int64_t refer_cnt{0};
};

struct AicoreRegisterInfo {
  std::string kernel_bin_name;
  int32_t magic{0};
  KernelBinPtr kernel_bin{nullptr};
};

struct AicpuRegisterInfo {
  std::string op_type;
  std::string so_name;
  std::string kernel_name;
  std::string op_kernel_lib;
};

struct CustAicpuRegisterInfo {
  KernelBinPtr cust_aicpu_kernel_bin{nullptr};
};
using KernelRegisterInfo = std::variant<AicoreRegisterInfo, AicpuRegisterInfo, CustAicpuRegisterInfo>;
class KernelHandlesManager {
 public:
  KernelHandlesManager() = default;
  virtual ~KernelHandlesManager() = default;
  virtual graphStatus RegisterKernel(const KernelRegisterInfo &register_info,
      const std::string &bin_name) = 0;
  virtual std::string GenerateKey(const KernelRegisterInfo &register_info) = 0;
  rtBinHandle GetOrRegisterKernel(const KernelRegisterInfo &register_info,
      const std::string &bin_name);
  rtBinHandle FindKernel(const std::string &bin_name) const;
  graphStatus ClearKernel();
 protected:
  bool IsKernelHandleRegistered(const std::string &bin_name);
  void StoredKernelHandle(const rtBinHandle bin_handle, const std::string &bin_name);
  std::unordered_map<std::string, int64_t> local_refer_cnt_;
  static std::unordered_map<std::string, KernelBinInfo> global_bin_store_;
  static std::mutex mtx_;
};
} // namespace ge

#endif // EXECUTOR_GRAPH_LOAD_MODEL_MANAGER_KERNEL_MANAGER_KERBEL_HANDLES_MANAGER_H