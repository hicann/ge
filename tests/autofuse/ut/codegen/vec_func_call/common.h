/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "runtime_stub.h"
#include "platform_context.h"

namespace ge {
    class RuntimeStubV2 : public RuntimeStub {
     public:
      rtError_t rtGetSocVersion(char *version, const uint32_t maxLen) override {
        (void)strcpy_s(version, maxLen, "Ascend910_9591");
        return RT_ERROR_NONE;
      }
    };

    // 公共函数：设置运行时存根
    void SetupRuntimeStub() {
        ge::PlatformContext::GetInstance().Reset();
        auto stub_v2 = std::make_shared<ge::RuntimeStubV2>();
        ge::RuntimeStub::SetInstance(stub_v2);
    }
}  // namespace ge
