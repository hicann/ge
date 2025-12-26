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

#include "graph/ge_error_codes.h"
#include "exe_graph/runtime/kernel_context.h"
#include "platform/platform_infos_def.h"

namespace gert {
namespace kernel {
enum class AclnnOpFwkDataInput {
  kExecutePrepareFunc,
  kExecuteLaunchFunc,
  kPlatformInfo
};
struct AclnnOpFwkData {
  void *op_execute_prepare_func;
  void *op_execute_launch_func;
  fe::PlatFormInfos *platform_info;
};

ge::graphStatus FindOpExeFunc(KernelContext *context);
ge::graphStatus ExecuteOpFunc(KernelContext *context);
ge::graphStatus FindOpExe2PhaseFunc(KernelContext *context);
ge::graphStatus ExecuteOpPrepare(KernelContext *context);
ge::graphStatus ExecuteOpLaunch(KernelContext *context);
ge::graphStatus BuildAclnnOpFwkData(KernelContext *context);
}
}
#endif  // AIR_CXX_RUNTIME_V2_KERNEL_ACLNN_OP_EXECUTE_KERNEL_H_
