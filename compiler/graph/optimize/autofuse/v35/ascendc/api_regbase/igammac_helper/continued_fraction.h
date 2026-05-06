/**
* Copyright (c) 2026 Huawei Technologies Co., Ltd.
* This program is free software, you can redistribute it and/or modify it under the terms and conditions of
* CANN Open Software License Agreement Version 2.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/
#ifndef __ASCENDC_API_REGBASE_IGAMMAC_HELPER_CONTINUED_FRACTION_H__
#define __ASCENDC_API_REGBASE_IGAMMAC_HELPER_CONTINUED_FRACTION_H__

#include "series.h"

namespace AscendC {
namespace IGammaCInternal {

__simd_callee__ inline void Igammac_compute_ans_float(
    Reg::RegTensor<float>& ansReg, Reg::RegTensor<float>& cReg, Reg::RegTensor<float>& tReg,
    Reg::RegTensor<float>& yReg, Reg::RegTensor<float>& zReg,
    Reg::RegTensor<float>& pkm1Reg, Reg::RegTensor<float>& pkm2Reg,
    Reg::RegTensor<float>& qkm1Reg, Reg::RegTensor<float>& qkm2Reg,
    Reg::MaskReg& iterMask, Reg::MaskReg& scaleMask)
{
    static constexpr float machep = 5.9604644775390625e-8f;
    static constexpr float big = 16777216.0f;
    static constexpr float biginv = 5.9604644775390625e-8f;
    static constexpr int maxiter = 25;
    Reg::MaskReg fullMask = Reg::CreateMask<float, Reg::MaskPattern::ALL>();
    Reg::RegTensor<float> ycReg, rReg, pkReg, qkReg, tmpReg, tmpReg1;
    Reg::MaskReg qkZeroMask, qkNonZeroMask;

    for (int i = 0; i < maxiter; i++) {
        Reg::Adds(cReg, cReg, 1.0f, iterMask);
        Reg::Adds(yReg, yReg, 1.0f, iterMask);
        Reg::Adds(zReg, zReg, 2.0f, iterMask);
        Reg::Mul(ycReg, yReg, cReg, iterMask);

        Reg::Mul(tmpReg, pkm1Reg, zReg, iterMask);
        Reg::Mul(tmpReg1, pkm2Reg, ycReg, iterMask);
        Reg::Sub(pkReg, tmpReg, tmpReg1, iterMask);

        Reg::Mul(tmpReg, qkm1Reg, zReg, iterMask);
        Reg::Mul(tmpReg1, qkm2Reg, ycReg, iterMask);
        Reg::Sub(qkReg, tmpReg, tmpReg1, iterMask);

        Reg::Compares<float, CMPMODE::EQ>(qkZeroMask, qkReg, 0.0f, iterMask);
        Reg::Not(qkNonZeroMask, qkZeroMask, fullMask);
        Reg::And(qkNonZeroMask, qkNonZeroMask, iterMask, fullMask);

        Reg::Div(rReg, pkReg, qkReg, qkNonZeroMask);
        Reg::Sub(tmpReg, ansReg, rReg, qkNonZeroMask);
        Reg::Div(tmpReg, tmpReg, rReg, qkNonZeroMask);
        Reg::Abs(tReg, tmpReg, qkNonZeroMask);
        Reg::Copy(ansReg, rReg, qkNonZeroMask);

        Reg::Duplicate<float, Reg::MaskMergeMode::MERGING>(tReg, 1.0f, qkZeroMask);

        Reg::Copy(pkm2Reg, pkm1Reg, iterMask);
        Reg::Copy(pkm1Reg, pkReg, iterMask);
        Reg::Copy(qkm2Reg, qkm1Reg, iterMask);
        Reg::Copy(qkm1Reg, qkReg, iterMask);

        Reg::Abs(tmpReg, pkReg, iterMask);
        Reg::Compares<float, CMPMODE::GT>(scaleMask, tmpReg, big, iterMask);
        Reg::Muls(tmpReg, pkm2Reg, biginv, scaleMask);
        Reg::Select(pkm2Reg, tmpReg, pkm2Reg, scaleMask);
        Reg::Muls(tmpReg, pkm1Reg, biginv, scaleMask);
        Reg::Select(pkm1Reg, tmpReg, pkm1Reg, scaleMask);
        Reg::Muls(tmpReg, qkm2Reg, biginv, scaleMask);
        Reg::Select(qkm2Reg, tmpReg, qkm2Reg, scaleMask);
        Reg::Muls(tmpReg, qkm1Reg, biginv, scaleMask);
        Reg::Select(qkm1Reg, tmpReg, qkm1Reg, scaleMask);

        Reg::Compares<float, CMPMODE::GT>(scaleMask, tReg, machep, iterMask);
        Reg::And(iterMask, iterMask, scaleMask, fullMask);
    }
}

__simd_callee__ inline void Igammac_helper_continued_fraction_float(Reg::RegTensor<float>& dstReg, Reg::RegTensor<float>& src0Reg,
                                                                    Reg::RegTensor<float>& src1Reg, Reg::RegTensor<float>& lgammaReg,
                                                                    Reg::RegTensor<float>& powReg, Reg::MaskReg mask)
{
    Reg::MaskReg fullMask = Reg::CreateMask<float, Reg::MaskPattern::ALL>();
    Reg::MaskReg zeroMask, iterMask, scaleMask;
    Reg::RegTensor<float> axReg, ansReg, cReg, tReg, yReg, zReg;
    Reg::RegTensor<float> pkm1Reg, pkm2Reg, qkm1Reg, qkm2Reg;

    Igammac_helper_fac_float(axReg, src0Reg, src1Reg, powReg, lgammaReg, fullMask);

    Reg::Compares<float, CMPMODE::EQ>(zeroMask, axReg, 0.0f, fullMask);
    Reg::Duplicate<float, Reg::MaskMergeMode::MERGING>(dstReg, 0.0f, zeroMask);
    Reg::Not(iterMask, zeroMask, fullMask);
    Reg::And(iterMask, iterMask, mask, fullMask);

    Reg::Duplicate(yReg, 1.0f, iterMask);
    Reg::Sub(yReg, yReg, src0Reg, iterMask);
    Reg::Add(zReg, src1Reg, yReg, iterMask);
    Reg::Adds(zReg, zReg, 1.0f, iterMask);
    Reg::Duplicate(cReg, 0.0f, iterMask);
    Reg::Duplicate(pkm2Reg, 1.0f, iterMask);
    Reg::Copy(qkm2Reg, src1Reg, iterMask);
    Reg::Adds(pkm1Reg, src1Reg, 1.0f, iterMask);
    Reg::Mul(qkm1Reg, zReg, src1Reg, iterMask);
    Reg::Div(ansReg, pkm1Reg, qkm1Reg, iterMask);

    Igammac_compute_ans_float(ansReg, cReg, tReg, yReg, zReg, pkm1Reg, pkm2Reg, qkm1Reg, qkm2Reg, iterMask, scaleMask);
    Reg::Mul(dstReg, ansReg, axReg, mask);
}
} // namespace IGammaCInternal
} // namespace AscendC

#endif // __ASCENDC_API_REGBASE_IGAMMAC_HELPER_CONTINUED_FRACTION_H__