/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef EXECUTOR_GRAPH_LOAD_MODEL_MANAGER_KERNEL_MANAGER_AICORE_KERBEL_HANDLES_MANAGER_H
#define EXECUTOR_GRAPH_LOAD_MODEL_MANAGER_KERNEL_MANAGER_AICORE_KERBEL_HANDLES_MANAGER_H

#include "kernel_handles_manager.h"

namespace ge {
class AicoreKernelHandlesManager : public KernelHandlesManager {
 public:
  AicoreKernelHandlesManager() = default;
  ~AicoreKernelHandlesManager() override = default;
  graphStatus RegisterKernel(const KernelRegisterInfo &register_info,
      const std::string &bin_name) override;
  std::string GenerateKey(const KernelRegisterInfo &register_info) override;
};
} // namespace ge
 
#endif // EXECUTOR_GRAPH_LOAD_MODEL_MANAGER_KERNEL_MANAGER_AICORE_KERBEL_HANDLES_MANAGER_H