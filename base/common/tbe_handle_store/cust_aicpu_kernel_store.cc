/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "common/tbe_handle_store/cust_aicpu_kernel_store.h"

namespace ge {
void CustAICPUKernelStore::AddCustAICPUKernel(const CustAICPUKernelPtr &kernel) {
  AddKernel(kernel);
}

void CustAICPUKernelStore::LoadCustAICPUKernelBinToOpDesc(const std::shared_ptr<OpDesc> &op_desc) const {
  GE_CHECK_NOTNULL_JUST_RETURN(op_desc);
  const auto &kernel_bin = FindKernel(op_desc->GetName());
  if (kernel_bin != nullptr) {
    GE_IF_BOOL_EXEC(!op_desc->SetExtAttr(OP_EXTATTR_CUSTAICPU_KERNEL, kernel_bin),
                    GELOGW("LoadKernelCustAICPUBinToOpDesc: SetExtAttr for kernel_bin failed"));
    GELOGI("Load cust aicpu kernel:%s, %zu", kernel_bin->GetName().c_str(), kernel_bin->GetBinDataSize());
  }
}
}  // namespace ge
