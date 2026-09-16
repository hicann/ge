/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph/custom_op/op_proto_ledger.h"

namespace ge {
ScopedOpProtoLoadTxn::ScopedOpProtoLoadTxn(const CustomOpRegistryPtr &) {}
ScopedOpProtoLoadTxn::~ScopedOpProtoLoadTxn() {}

void OpProtoLedger::ResetForFinalize() {}
void OpProtoLedger::ClaimProviderMaps(const std::string &) {}
void OpProtoLedger::SetCurrentProvider(const std::string &, const std::string &) {}
void OpProtoLedger::AttachProviderHandle(const std::string &, const std::shared_ptr<void> &) {}
}  // namespace ge
