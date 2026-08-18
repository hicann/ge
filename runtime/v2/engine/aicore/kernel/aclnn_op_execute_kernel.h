/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_CXX_RUNTIME_V2_KERNEL_ACLNN_OP_EXECUTE_KERNEL_H_
#define AIR_CXX_RUNTIME_V2_KERNEL_ACLNN_OP_EXECUTE_KERNEL_H_

#include "graph/error_codes.h"
#include "exe_graph/runtime/kernel_context.h"
#include "platform/platform_infos_def.h"
#include "kernel/common_kernel_impl/platform.h"

namespace gert {
namespace kernel {
struct AclnnDeterministicConfig {
  bool has_op_deterministic = false;
  int32_t op_deterministic = 0;
  bool has_op_deterministic_level = false;
  int32_t op_deterministic_level = 0;
};

struct AclnnOriginalDeterministicConfig {
  int32_t deterministic = 0;
  int32_t deterministic_level = 0;
  int8_t reserved[4];  // 填充以确保sizeof > 8。若 <=
                       // 8字节，框架会启用Inplace优化导致GetPointer解析错误，此处强制走指针模式
};

enum class SingleStageAclnnOpFwkDataIndex { kCoreNumInfos, kDeterministicConfig, kOriginalDeterministicConfig };

struct SingleStageAclnnOpFwkData {
  const CoreNumInfos *core_num_infos;
  const AclnnDeterministicConfig *deterministic_config;
  const AclnnOriginalDeterministicConfig *original_deterministic_config;
};

enum class DualStageAclnnOpFwkDataIndex {
  kExecutePrepareFunc,
  kExecuteLaunchFunc,
  kPlatformInfo,
  kCoreNumInfos,
  kDeterministicConfig,
  kOriginalDeterministicConfig
};

struct DualStageAclnnOpFwkData {
  void *op_execute_prepare_func;
  void *op_execute_launch_func;
  fe::PlatFormInfos *platform_info;
  const CoreNumInfos *core_num_infos;
  const AclnnDeterministicConfig *deterministic_config;
  const AclnnOriginalDeterministicConfig *original_deterministic_config;
};

ge::graphStatus FindOpExeFunc(KernelContext *context);
ge::graphStatus ExecuteOpFunc(KernelContext *context);
ge::graphStatus BuildSingleStageAclnnOpFwkData(KernelContext *context);
ge::graphStatus FindOpExe2PhaseFunc(KernelContext *context);
ge::graphStatus ExecuteOpPrepare(KernelContext *context);
ge::graphStatus ExecuteOpLaunch(KernelContext *context);
ge::graphStatus BuildDualStageAclnnOpFwkData(KernelContext *context);
}  // namespace kernel
}  // namespace gert
#endif  // AIR_CXX_RUNTIME_V2_KERNEL_ACLNN_OP_EXECUTE_KERNEL_H_
