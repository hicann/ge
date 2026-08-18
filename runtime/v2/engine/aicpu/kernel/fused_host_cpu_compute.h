/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_CXX_RUNTIME_V2_ENGINE_AICPU_KERNEL_FUSED_HOST_CPU_COMPUTE_H_
#define AIR_CXX_RUNTIME_V2_ENGINE_AICPU_KERNEL_FUSED_HOST_CPU_COMPUTE_H_

#include "aicpu_resource_manager.h"

namespace gert {
struct FusedHostCpuTensorMeta {
  size_t dim_num;
};

enum FusedHostCpuBindingFlag : uint32_t { kFusedHostCpuShapeChanged = 1U, kFusedHostCpuDataChanged = 2U };

// Private C ABI payload shared by RT2 and the generated JIT SO. Keep it standard-layout; compiler, runtime and
// generated SO must be deployed as one ABI-compatible set.
struct FusedHostCpuTensorBinding {
  const int64_t *dims;
  uint8_t *data;
  size_t dim_num;
  size_t data_size;
  uint32_t flags;
};

struct FusedHostCpuComputeMeta {
  void *compute_state;
  size_t input_num;
  size_t output_num;
};

struct FusedHostCpuDestroyMeta {
  void *compute_state;
};

namespace kernel {
void *CreateFusedHostCpuComputeState(const char *register_name, void *kernel_state,
                                     FusedHostCpuDestroyFunc destroy_func, FusedHostCpuRunFunc run_func,
                                     const FusedHostCpuTensorMeta *tensor_metas, size_t io_num);
void DestroyFusedHostCpuComputeState(void *compute_state);
}  // namespace kernel
}  // namespace gert
#endif  // AIR_CXX_RUNTIME_V2_ENGINE_AICPU_KERNEL_FUSED_HOST_CPU_COMPUTE_H_
