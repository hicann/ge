/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GE_FRAMEWORK_RUNTIME_GERT_MODEL_GERT_MODEL_EXECUTOR_TYPES_H_
#define GE_FRAMEWORK_RUNTIME_GERT_MODEL_GERT_MODEL_EXECUTOR_TYPES_H_

#include "framework/runtime/dump/model_dump_c_api.h"

enum GertModelTaskLaunchType : uint64_t {
  ACL_RT_LAUNCH_KERNEL_V2 = 0,
  RT_STARS_TASK_LAUNCH_WITH_FLAG = 1,
};

struct GertModelLaunchKernelV2Params {
  uint64_t struct_size = sizeof(GertModelLaunchKernelV2Params);

  aclrtFuncHandle func_handle = nullptr;
  uint32_t block_dim = 0;
  // 用于填充空洞，保持结构体布局与 ACL 接口一致。
  uint32_t reserved_1 = 0;
  const void *args_data = nullptr;
  size_t args_size = 0;
  aclrtLaunchKernelCfg *config = nullptr;
  aclrtStream stream = nullptr;
};

struct GertModelLaunchStarsTaskWithFlagParams {
  uint64_t struct_size = sizeof(GertModelLaunchStarsTaskWithFlagParams);

  const void *task_sqe = nullptr;
  uint32_t sqe_len = 0;
  // 用于填充空洞，保持结构体布局与 ACL 接口一致。
  uint32_t reserved_1 = 0;
  aclrtStream stream = nullptr;
  uint32_t flag = 0;
  // 用于填充空洞，保持结构体布局与 ACL 接口一致。
  uint32_t reserved_2 = 0;
};

union GertModelTaskLaunchParams {
  GertModelLaunchKernelV2Params launch_kernel_v2_params;
  GertModelLaunchStarsTaskWithFlagParams launch_stars_task_params;
};

struct GertModelTaskLaunchInfo {
  uint64_t struct_size = sizeof(GertModelTaskLaunchInfo);

  GertModelTaskLaunchType launch_type = ACL_RT_LAUNCH_KERNEL_V2;
  GertModelTaskDesc *task_info = nullptr;
  const GertModelTaskLaunchParams *launch_params = nullptr;
};

#ifdef __cplusplus
extern "C" {
#endif

typedef int32_t (*GertModelLaunchFunc)(void *instance_handle, GertModelTaskLaunchInfo *launch_info);

#ifdef __cplusplus
}
#endif

struct GertModelLoadCallbacks {
  uint64_t struct_size = sizeof(GertModelLoadCallbacks);  // 布局变化时更新

  // codegen 在 InitResources 创建 rt_model_handle 后、Load 前回调；
  // executor 收到后完成 ReportModelBaseInfo（组装 ModelDumpInfo → SetModelDumpInfo）
  ReportModelBaseInfoFunc report_model_base_info = nullptr;
  GertModelLaunchFunc launch_func = nullptr;
};

#endif  // GE_FRAMEWORK_RUNTIME_GERT_MODEL_GERT_MODEL_EXECUTOR_TYPES_H_
