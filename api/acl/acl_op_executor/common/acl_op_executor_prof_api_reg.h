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

#define OP_EXEC_PROF_TYPE_START_OFFSET 0x008000U

namespace acl {
    enum AclOpExecProfType {
        // start with 0x018000U
        OpExecProfTypeStart = MSPROF_REPORT_ACL_OP_BASE_TYPE + OP_EXEC_PROF_TYPE_START_OFFSET,
        AclopLoad,
        AclopExecute,
        AclopCreateHandle,
        AclopDestroyHandle,
        AclopExecWithHandle,
        AclopExecuteV2,
        AclopCreateKernel,
        AclopUpdateParams,
        AclopInferShape,
        AclopCast,
        AclopCreateHandleForCast,
        AclopCreateAttr,
        AclopDestroyAttr,
        // this is the end, can not add after OpExecProfTypeEnd
        OpExecProfTypeEnd,
    };

    class AclOpExecProfilingReporter {
    public:
        explicit AclOpExecProfilingReporter(const AclOpExecProfType apiId);
        virtual ~AclOpExecProfilingReporter() noexcept;
    private:
        uint64_t startTime_ = 0UL;
        const AclOpExecProfType aclApi_;
    };
}  // namespace acl

#define ACL_PROFILING_REG(apiId) \
    const acl::AclOpExecProfilingReporter profilingReporter(apiId)
#endif 
