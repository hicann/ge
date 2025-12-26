/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef REDUCEINIT_H
#define REDUCEINIT_H
#include "kernel_operator.h"
using namespace AscendC;

#ifndef INFINITY
#define INFINITY (1.0f / 0.0f)
#endif

constexpr int32_t kReduceOpMin = 0;
constexpr int32_t kReduceOpMax = 1;
constexpr int32_t kReduceOpSum = 2;
constexpr int32_t kReduceOpProd = 3;
constexpr int32_t kReduceOpAny = 4;
constexpr int32_t kReduceOpAll = 5;
constexpr int32_t kReduceOpMean = 6;

template <typename T, int reduce_type>
inline __aicore__ T GetPaddingValue() {
    T paddingValue = 1;
    if constexpr (reduce_type == kReduceOpMin) {
        paddingValue = INFINITY;
    } else if constexpr (reduce_type == kReduceOpMax) {
        paddingValue = -INFINITY;
    } else if constexpr (reduce_type == kReduceOpSum) {
        paddingValue = T(0);
    } else if constexpr (reduce_type == kReduceOpProd) {
        paddingValue = T(1);
    } else if constexpr (reduce_type == kReduceOpAny) {
        paddingValue = T(0);
    } else if constexpr (reduce_type == kReduceOpAll) {
        paddingValue = T(1);
    } else if constexpr (reduce_type ==kReduceOpMean) {
        paddingValue = T(0);
    }
    return paddingValue;
}

template <typename T>
__aicore__ inline void ReduceInitImpl(__local_mem__ T* dstUb, uint32_t repeat_times, uint32_t repeat_timesTail,
                                      uint32_t sreg, uint32_t sreg_tail, uint32_t sreg_one_blk, uint32_t strideElement,
                                      T padValue, uint32_t currentOffset, uint32_t baseOffset) {
    MicroAPI::MaskReg fullMask = MicroAPI::CreateMask<T, MicroAPI::MaskPattern::ALL>();
    uint32_t sreg_down = (sreg * sizeof(T) / 32) * 32 / sizeof(T);
    uint32_t sreg_new = sreg - sreg_down;
    MicroAPI::MaskReg preg = AscendC::MicroAPI::UpdateMask<T>(sreg_new);
    MicroAPI::MaskReg preg_one_block = AscendC::MicroAPI::UpdateMask<T>(sreg_one_blk);
    MicroAPI::MaskReg preg_pad;
    MicroAPI::MaskNot(preg_pad, preg, fullMask);
    MicroAPI::RegTensor<T> srcVreg;
    MicroAPI::UnalignReg ureg;


    for (uint32_t i = 0; i < static_cast<uint32_t>(repeat_times); ++i) {
        MicroAPI::DataCopy<T>(srcVreg, dstUb + baseOffset + i * strideElement);
        MicroAPI::Duplicate<T, AscendC::MicroAPI::MaskMergeMode::MERGING>(srcVreg, padValue, preg_pad);
        MicroAPI::DataCopy<T>(dstUb + baseOffset + i * strideElement, srcVreg, preg_one_block);
    }

    MicroAPI::RegTensor<T> dstReg;
    MicroAPI::MaskReg mask;
    MicroAPI::Duplicate(dstReg, padValue, fullMask);
    constexpr uint32_t repeatStrideTail = static_cast<uint32_t>(VECTOR_REG_WIDTH / sizeof(T));
    for (uint32_t i = 0; i < static_cast<uint32_t>(repeat_timesTail); ++i) {
        mask = AscendC::MicroAPI::UpdateMask<T>(sreg_tail);
        MicroAPI::DataCopy(dstUb + currentOffset + i * repeatStrideTail, dstReg, mask);
    }
}

template <typename T, int32_t ReduceType, bool isTailLast>
__aicore__ inline void ReduceInit(const LocalTensor<T> &dstTensor, const uint32_t dim_a, const uint32_t dim_r,
                                  const uint32_t dim_r_current, const uint32_t inner_r){
    static_assert(SupportType<T, half, float, uint8_t, int8_t, uint16_t, int16_t, uint32_t, int32_t, uint64_t, int64_t>());
    __local_mem__ T* dstUb = (__local_mem__ T*)dstTensor.GetPhyAddr();
    T padValue = GetPaddingValue<T, ReduceType>();
    uint32_t baseOffset = (inner_r * sizeof(T) / AscendC::ONE_BLK_SIZE) * AscendC::ONE_BLK_SIZE;
    baseOffset = baseOffset / sizeof(T);
    uint32_t repeatStride = AlignUp(inner_r * sizeof(T), AscendC::ONE_BLK_SIZE);
    uint32_t strideElement = repeatStride / sizeof(T);
    uint32_t repeat_times = (dim_a - 1 ) * dim_r / strideElement + dim_r_current / strideElement;
    uint32_t sreg = inner_r;
    uint32_t sreg_tail = dim_r - dim_r_current;
    constexpr uint32_t repeatStrideTail = static_cast<uint32_t>(VECTOR_REG_WIDTH / sizeof(T));
    uint32_t repeatTimesTail = static_cast<uint16_t>(CeilDivision(sreg_tail, repeatStrideTail));
    uint32_t currentOffset = (dim_a - 1) * dim_r + dim_r_current;
    uint32_t sreg_one_blk = (inner_r * sizeof(T) % 32 == 0) ? 0 : AscendC::ONE_BLK_SIZE / sizeof(T);
    VF_CALL<ReduceInitImpl<T>>(dstUb, repeat_times, repeatTimesTail, sreg, sreg_tail, sreg_one_blk, strideElement, padValue, currentOffset, baseOffset);
}
#endif