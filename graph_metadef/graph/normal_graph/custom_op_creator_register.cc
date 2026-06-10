/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph/custom_op.h"

#include <memory>

#include "graph/custom_op_factory.h"
#include "graph/custom_op_load_context.h"
#include "graph/custom_op_pull_registry.h"

namespace ge {
CustomOpCreatorRegister::CustomOpCreatorRegister(const AscendString &operator_type,
                                                 const CustomOpCreateFunc op_creator) {
  RegisterCustomOpLocalCreator(operator_type.GetString(), op_creator);
  if ((op_creator == nullptr) || IsOfflineCustomOpSoLoading()) {
    return;
  }
  CustomOpFactory::RegisterCustomOpCreator(operator_type, [op_creator]() -> std::unique_ptr<BaseCustomOp> {
    return std::unique_ptr<BaseCustomOp>(op_creator());
  });
}
}  // namespace ge
