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

#define CBLAS_PROF_TYPE_START_OFFSET 0x006000U

namespace acl {
        enum AclCblasProfType {
            // start with 0x046000U
            CblasProfTypeStart = MSPROF_REPORT_ACL_OTHERS_BASE_TYPE + CBLAS_PROF_TYPE_START_OFFSET,
            AclblasGemmEx,
            AclblasCreateHandleForGemmEx,
            AclblasCreateHandleForHgemm,
            AclblasHgemm,
            AclblasS8gemm,
            AclblasCreateHandleForS8gemm,
            AclblasGemvEx,
            AclblasCreateHandleForGemvEx,
            AclblasHgemv,
            AclblasCreateHandleForHgemv,
            AclblasCreateHandleForS8gemv,
            AclblasS8gemv,
            // this is the end, can not add after CblasProfTypeEnd
            CblasProfTypeEnd
        };

    class AclCblasProfilingReporter {
    public:
        explicit AclCblasProfilingReporter(const AclCblasProfType apiId);
        virtual ~AclCblasProfilingReporter() noexcept;
    private:
        uint64_t startTime_ = 0UL;
        const AclCblasProfType aclApi_;
    };
}  // namespace acl

#define ACL_PROFILING_REG(apiId) \
    const acl::AclCblasProfilingReporter profilingReporter(apiId)
#endif 
