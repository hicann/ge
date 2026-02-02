/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "engines/custom_engine/custom_ops_kernel_builder.h"
#include <memory>
#include "framework/common/ge_inner_error_codes.h"
#include "framework/common/debug/ge_log.h"
#include "register/ops_kernel_builder_registry.h"
#include "common/ge_common/ge_types.h"

namespace ge {
namespace custom {
REGISTER_OPS_KERNEL_BUILDER(kCustomOpKernelLibName, CustomOpsKernelBuilder);

CustomOpsKernelBuilder::~CustomOpsKernelBuilder() {
  GELOGI("CustomOpsKernelBuilder destroyed");
}

Status CustomOpsKernelBuilder::Initialize(const std::map<std::string, std::string> &options) {
  (void)options;
  return SUCCESS;
}

Status CustomOpsKernelBuilder::Finalize() {
  return SUCCESS;
}

Status CustomOpsKernelBuilder::CalcOpRunningParam(Node &node) {
  (void)node;
  return SUCCESS;
}

Status CustomOpsKernelBuilder::GenerateTask(const Node &node,
    RunContext &context, std::vector<domi::TaskDef> &tasks) {
  (void)node;
  (void)context;
  (void)tasks;
  return SUCCESS;
}
}  // namespace ge_local
}  // namespace ge
