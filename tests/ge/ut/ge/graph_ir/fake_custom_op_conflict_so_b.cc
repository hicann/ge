/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "framework/common/framework_types_internal.h"
#include "graph/custom_op.h"
#include "graph/custom_op_pull_registry.h"
#include "operator_reg.h"

namespace {
constexpr char kFakeOpType[] = "FakeCustomOpConflictUt";
constexpr char kFakeSoMark[] = "fake-conflict-b";

class FakeConflictedOpB : public ge::BaseCustomOp {};

ge::BaseCustomOp *CreateFakeConflictedOpB() {
  return new (std::nothrow) FakeConflictedOpB();
}
// 注意：不调用 RegisterCustomOpLocalCreator —— 其实现位于 custom_op_registry_static 库，
// fake SO 不链接该库（RTLD_NOW 下 dlopen 会报未定义符号）。pull ABI 由下方手写 extern "C" 自包含提供。
}  // namespace

namespace ge {
REG_OP(FakeCustomOpConflictUt).OUTPUT(y, TensorType::ALL()).OP_END_FACTORY_REG(FakeCustomOpConflictUt);
}  // namespace ge

extern "C" const char *GetFakeCustomOpSoMark() {
  return kFakeSoMark;
}

extern "C" uint32_t GetRegisteredCustomOpCreatorAbiVersion() {
  return ge::kCustomOpCreatorPullAbiVersionV2;
}

extern "C" size_t GetRegisteredCustomOpCreatorNum() {
  return 1U;
}

extern "C" int32_t GetRegisteredCustomOpCreators(ge::CustomOpTypeToCreator *creators, const size_t creator_num,
                                                 const size_t creator_struct_size) {
  if ((creators == nullptr) || (creator_num < 1U) || (creator_struct_size != sizeof(ge::CustomOpTypeToCreator))) {
    return -1;
  }
  creators[0] = ge::CustomOpTypeToCreator{sizeof(ge::CustomOpTypeToCreator), kFakeOpType, &CreateFakeConflictedOpB,
                                          ge::OpBackend::kDevice};
  return 0;
}
