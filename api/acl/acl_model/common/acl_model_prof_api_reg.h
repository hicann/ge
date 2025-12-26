/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ACL_PROF_API_REG_H_
#define ACL_PROF_API_REG_H_

#include "acl/acl_base.h"
#include "aprof_pub.h"

namespace acl {
        enum AclMdlProfType {
            // start with 0x020000U
            ModelProfTypeStart = MSPROF_REPORT_ACL_MODEL_BASE_TYPE,
            AclmdlExecute,
            AclmdlLoadFromMemWithQ,
            AclmdlLoadFromMemWithMem,
            AclmdlGetDesc,
            AclmdlLoadFromFile,
            AclmdlLoadFromFileWithMem,
            AclmdlLoadFromMem,
            AclmdlBundleLoadFromFile,
            AclmdlBundleLoadFromMem,
            AclmdlBundleUnload,
            AclmdlSetInputAIPP,
            AclmdlSetAIPPByInputIndex,
            AclmdlExecuteAsync,
            AclmdlQuerySize,
            AclmdlQuerySizeFromMem,
            AclmdlSetDynamicBatchSize,
            AclmdlSetDynamicHWSize,
            AclmdlSetInputDynamicDims,
            AclmdlLoadWithConfig,
            AclmdlLoadFromFileWithQ,
            AclmdlUnload,
            AclCreateTensorDesc,
            AclDestroyTensorDesc,
            AclTransTensorDescFormat,
            // this is the end, can not add after ModelProfTypeEnd
            ModelProfTypeEnd,
        };

    class AclMdlProfilingReporter {
    public:
        explicit AclMdlProfilingReporter(const AclMdlProfType apiId);
        virtual ~AclMdlProfilingReporter() noexcept;
    private:
        uint64_t startTime_ = 0UL;
        const AclMdlProfType aclApi_;
    };
}  // namespace acl

#define ACL_PROFILING_REG(apiId) \
    const acl::AclMdlProfilingReporter profilingReporter(apiId)
#endif 
