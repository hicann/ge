/**
* Copyright (c) 2026 Huawei Technologies Co., Ltd.
* This program is free software, you can redistribute it and/or modify it under the terms and conditions of
* CANN Open Software License Agreement Version 2.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/
#ifndef __ASCENDC_API_REGBASE_IGAMMAC_HELPER_SERIES_COMPLEMENT_H__
#define __ASCENDC_API_REGBASE_IGAMMAC_HELPER_SERIES_COMPLEMENT_H__

namespace AscendC {
namespace IGammaCInternal {

template <int32_t iterationNum>
__simd_callee__ inline void SeriesComplementIter(
    Reg::RegTensor<float>& facReg, Reg::RegTensor<float>& sumReg,
    Reg::RegTensor<float>& src0Reg, Reg::RegTensor<float>& src1Reg,
    Reg::MaskReg& fullMask)
{
    Reg::RegTensor<float> tmpReg, tmpReg1, termReg;

    Reg::Duplicate(tmpReg, -1.0f / iterationNum, fullMask);
    Reg::Mul(tmpReg, src1Reg, tmpReg, fullMask);
    Reg::Mul(facReg, facReg, tmpReg, fullMask);
    Reg::Duplicate(tmpReg1, iterationNum, fullMask);
    Reg::Add(tmpReg1, src0Reg, tmpReg1, fullMask);
    Reg::Div(termReg, facReg, tmpReg1, fullMask);
    Reg::Add(sumReg, sumReg, termReg, fullMask);

    if constexpr (iterationNum < 25) {
        SeriesComplementIter<iterationNum + 1>(facReg, sumReg, src0Reg, src1Reg, fullMask);
    }
}

__simd_callee__ inline void Igammac_helper_series_complement_float(
    Reg::RegTensor<float>& dstReg,
    Reg::RegTensor<float>& src0Reg,
    Reg::RegTensor<float>& src1Reg,
    Reg::RegTensor<float>& lgammaReg,
    Reg::MaskReg mask)
{
    constexpr float machep = 5.9604644775390625e-8f;

    Reg::RegTensor<float> tmpReg;
    Reg::RegTensor<float> tmpReg1;
    Reg::RegTensor<float> lgamma_a1_Reg;
    Reg::MaskReg fullMask = Reg::CreateMask<float, Reg::MaskPattern::ALL>();

    Reg::RegTensor<float> facReg;
    Reg::RegTensor<float> sumReg;
    Reg::RegTensor<float> termReg;
    Reg::RegTensor<float> logxReg;
    Reg::RegTensor<float> oneReg;

    Reg::Duplicate(facReg, 1.0f, fullMask);
    Reg::Duplicate(sumReg, 0.0f, fullMask);
    Reg::Duplicate(oneReg, 1.0f, fullMask);

    SeriesComplementIter<1>(facReg, sumReg, src0Reg, src1Reg, fullMask);

    Reg::Log(logxReg, src1Reg, fullMask);
    Reg::Log(lgamma_a1_Reg, src0Reg, fullMask);
    Reg::Add(lgamma_a1_Reg, lgamma_a1_Reg, lgammaReg, fullMask);
    Reg::Mul(tmpReg, src0Reg, logxReg, fullMask);
    Reg::Sub(tmpReg, tmpReg, lgamma_a1_Reg, fullMask);
    Reg::Exp(tmpReg, tmpReg, fullMask);
    Reg::Sub(tmpReg, oneReg, tmpReg, fullMask);
    Reg::Mul(tmpReg1, src0Reg, logxReg, fullMask);
    Reg::Sub(tmpReg1, tmpReg1, lgammaReg, fullMask);
    Reg::Exp(tmpReg1, tmpReg1, fullMask);
    Reg::Mul(tmpReg1, tmpReg1, sumReg, fullMask);
    Reg::Sub(dstReg, tmpReg, tmpReg1, fullMask);
}

} // namespace IGammaCInternal
} // namespace AscendC

#endif // __ASCENDC_API_REGBASE_IGAMMAC_HELPER_SERIES_COMPLEMENT_H__