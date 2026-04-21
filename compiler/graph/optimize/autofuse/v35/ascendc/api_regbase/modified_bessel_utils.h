/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef __ASCENDC_API_REGBASE_MODIFIED_BESSEL_UTILS_H__
#define __ASCENDC_API_REGBASE_MODIFIED_BESSEL_UTILS_H__

template<typename T, int32_t iterationNum, int32_t factorLen, const float* factorList>
__simd_callee__ inline void mainIter(AscendC::Reg::RegTensor<T>& pReg, AscendC::Reg::RegTensor<T>& qReg, 
                                    AscendC::Reg::RegTensor<T>& constReg, AscendC::Reg::RegTensor<T>& xFactorReg,
                                    AscendC::Reg::MaskReg& branchMask) {
    AscendC::Reg::RegTensor<T> iterReg;

    AscendC::Reg::Adds(pReg, qReg, (T)0.0, branchMask);
    AscendC::Reg::Adds(qReg, constReg, (T)0.0, branchMask);
    AscendC::Reg::Mul(iterReg, xFactorReg, qReg, branchMask);
    AscendC::Reg::Sub(iterReg, iterReg, pReg, branchMask);
    AscendC::Reg::Adds(constReg, iterReg, (T)factorList[factorLen - iterationNum + 1], branchMask);
    if constexpr (iterationNum > 2) {
        mainIter<T, iterationNum - 1, factorLen, factorList>(pReg, qReg, constReg, xFactorReg, branchMask);
    }
}

#endif // __ASCENDC_API_REGBASE_MODIFIED_BESSEL_UTILS_H__