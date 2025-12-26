/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CAST_H 
#define CAST_H
#include "kernel_operator.h"
using namespace AscendC;

template <typename InT, typename OutT>
inline __aicore__ constexpr AscendC::RoundMode GetCastRoundMode() {
    bool getRint = SupportType<Tuple<OutT,InT>, Tuple<half, int16_t>, Tuple<bfloat16_t, float>>();
    bool getTrunc = SupportType<Tuple<OutT, InT>, Tuple<int32_t, bfloat16_t>, Tuple<float, int32_t>
        , Tuple<half, int32_t>, Tuple<float, float>, Tuple<int4x2_t, half>>();

    if (getRint) {
        return AscendC::RoundMode::CAST_RINT;
    }
    if (getTrunc) {
        return AscendC::RoundMode::CAST_TRUNC;
    }
    if constexpr (AscendC::IsSameType<InT, float>::value) {
        if constexpr (AscendC::IsSameType<OutT, half>::value){
            return AscendC::RoundMode::CAST_RINT;
        }
    if constexpr (AscendC::SupportType<OutT, int64_t, int32_t, int16_t>()){
        return AscendC::RoundMode::CAST_TRUNC;
        }
    }
    if constexpr (AscendC::IsSameType<InT, half>::value &&
                AscendC::SupportType<OutT, int32_t, int16_t, int8_t, uint8_t>()){
        return AscendC::RoundMode::CAST_TRUNC;
    }
    if constexpr (AscendC::IsSameType<InT, int64_t>::value && AscendC::SupportType<OutT, float>()){
        return AscendC::RoundMode::CAST_RINT;
    }
    return AscendC::RoundMode::CAST_NONE;
}

template<typename InT, typename OutT>
__aicore__ inline void GenLoadInstr(MicroAPI::RegTensor<InT> &srcVreg, __local_mem__ InT *srcAddr){
    if constexpr (SupportType<InT, int4x2_t>() && sizeof(OutT) == 2) {
    MicroAPI::UnPack<uint16_t,uint8_t>((MicroAPI::RegTensor<uint16_t> &)srcVreg, 
                                       (MicroAPI::RegTensor<uint8_t> &)srcVreg);
    MicroAPI::UnPack<uint32_t, uint16_t>((MicroAPI::RegTensor<uint32_t> &)srcVreg,
                                        (MicroAPI::RegTensor<uint16_t> &)srcVreg);
    } else if constexpr (sizeof(InT) == 1 && sizeof(OutT) == 2) {
        MicroAPI::DataCopy<InT, MicroAPI::LoadDist::DIST_UNPACK_B8>(srcVreg,srcAddr);
    } else if constexpr (sizeof(InT) == 2 && sizeof(OutT) == 4) {
        MicroAPI::DataCopy<InT, MicroAPI::LoadDist::DIST_UNPACK_B16>(srcVreg, srcAddr);
    } else {
        MicroAPI::DataCopy(srcVreg, srcAddr);
    } 
}


template<typename InT, typename OutT>
__aicore__ inline void GenStoreInstr(__local_mem__ OutT *dstAddr, 
                                     MicroAPI::RegTensor<OutT> & dstVreg,
                                     MicroAPI::MaskReg &maskReg){
    if constexpr (SupportType<OutT, int4x2_t>() && sizeof(InT) == 2) {
        MicroAPI::Pack<uint16_t,uint32_t>((MicroAPI::RegTensor<uint16_t> &)dstVreg,
                                          (MicroAPI::RegTensor<uint32_t> &)dstVreg);
        MicroAPI::Pack<uint8_t,uint16_t>((MicroAPI::RegTensor<uint8_t> &)dstVreg,
                                         (MicroAPI::RegTensor<uint16_t> &)dstVreg);
    } else if constexpr (sizeof(OutT) == 1 && sizeof(InT) == 2) {
        MicroAPI::DataCopy<OutT, MicroAPI::StoreDist::DIST_PACK_B16>(dstAddr, dstVreg, maskReg);
    } else if constexpr (sizeof(OutT) == 2 && sizeof(InT) == 4) {
        MicroAPI::DataCopy<OutT, MicroAPI::StoreDist::DIST_PACK_B32>(dstAddr, dstVreg, maskReg);
    } else {
        MicroAPI::DataCopy(dstAddr, dstVreg, maskReg);
    }
}

template <typename InT, typename OutT, AscendC::RoundMode roundMode>
inline __aicore__ void CastExtendCommon(__local_mem__ OutT* dstUb, __local_mem__ InT* srcUb,
                                        const int64_t count, uint32_t repeat_times,
                                        uint32_t innerLoopStride) {
    static constexpr MicroAPI::CastTrait cast_trait = {MicroAPI::RegLayout::ZERO,
                                                       MicroAPI::SatMode::NO_SAT,
                                                       MicroAPI::MaskMergeMode::ZEROING, roundMode};

    uint32_t sreg = static_cast<uint32_t>(count);
    MicroAPI::RegTensor<InT> srcVreg;
    MicroAPI::RegTensor<OutT> dstVreg;
    MicroAPI::MaskReg stMaskReg, exMaskReg;
    exMaskReg = MicroAPI::CreateMask<uint8_t, MicroAPI::MaskPattern::ALL>();

    for (uint32_t i = 0; i < repeat_times; i++){
        if constexpr (sizeof(InT) < sizeof(OutT)) {
            stMaskReg = MicroAPI::UpdateMask<OutT>(sreg);
        } else {
            stMaskReg = MicroAPI::UpdateMask<InT>(sreg);
        }
        if constexpr ((SupportType<OutT, int4x2_t>() && sizeof(InT) == 2)) {
                MicroAPI::MaskPack(stMaskReg, stMaskReg);
        }
        GenLoadInstr<InT,OutT>(srcVreg, srcUb + innerLoopStride * i);
        if constexpr (std::is_same_v<InT, int32_t> && std::is_same_v<OutT, half>) {
            MicroAPI::Cast<float, InT, cast_trait>((MicroAPI::RegTensor<float> &)dstVreg, srcVreg,
                                                    exMaskReg);
            MicroAPI::Cast<OutT, float, cast_trait>(dstVreg, (MicroAPI::RegTensor<float> &)dstVreg,
                                                    exMaskReg);
        } else if constexpr (std::is_same_v<InT, float> && std::is_same_v<OutT, float>) {
            MicroAPI::Truncate<InT, roundMode>(dstVreg, srcVreg, exMaskReg);
        } else {
            MicroAPI::Cast<OutT,InT, cast_trait>(dstVreg, srcVreg, exMaskReg);
        }
        GenStoreInstr<InT, OutT>(dstUb + innerLoopStride * i, dstVreg, stMaskReg);
    }
}

template <typename InT, typename OutT, AscendC::RoundMode roundMode>
inline __aicore__ void CastExtendB4(__local_mem__ OutT* dstUb, __local_mem__ InT* srcUb,
                                    const int64_t count, uint32_t repeat_times,
                                    uint32_t innerLoopStride) {
    static constexpr MicroAPI::CastTrait cast_trait = {MicroAPI::RegLayout::ZERO,
                                                       MicroAPI::SatMode::NO_SAT,
                                                       MicroAPI::MaskMergeMode::ZEROING, roundMode};

    MicroAPI::MaskReg ldMaskReg, stMaskReg, exMaskReg, dumpMaskReg;
    uint32_t sreg = static_cast<uint32_t>(count);
    MicroAPI::RegTensor<InT> srcVreg;
    MicroAPI::RegTensor<OutT> dstVreg;
    exMaskReg = MicroAPI::CreateMask<uint8_t, MicroAPI::MaskPattern::ALL>();

    for (uint32_t i = 0; i < repeat_times; i++){
        if constexpr (sizeof(InT) < sizeof(OutT)) {
            stMaskReg = MicroAPI::UpdateMask<OutT>(sreg);
            exMaskReg = stMaskReg;
            MicroAPI::MaskPack(ldMaskReg, stMaskReg);
            if constexpr ((SupportType<OutT, int4x2_t>() && sizeof(InT) == 2)) {
                MicroAPI::MaskPack(ldMaskReg, ldMaskReg);
                MicroAPI::MaskUnPack(stMaskReg, ldMaskReg);
                MicroAPI::MaskUnPack(exMaskReg,stMaskReg);
                MicroAPI::MaskInterleave<uint16_t>(stMaskReg, dumpMaskReg, stMaskReg, stMaskReg);
            }
        } else if constexpr (sizeof(InT) > sizeof(OutT)) {
            ldMaskReg = MicroAPI::UpdateMask<InT>(sreg);
            exMaskReg = ldMaskReg;
            MicroAPI::MaskPack(stMaskReg, ldMaskReg);
            if constexpr ((SupportType<OutT, int4x2_t>() && sizeof(InT) == 2)) {
                MicroAPI::MaskPack(stMaskReg, stMaskReg);
            }
        }
        if constexpr (std::is_same_v<InT, int4x2_t>) {
            GenLoadInstr<InT, OutT>(srcVreg, srcUb + innerLoopStride * i / 2);
        } else {
            GenLoadInstr<InT, OutT>(srcVreg, srcUb + innerLoopStride * i);
        }
        MicroAPI::Cast<OutT, InT, cast_trait>(dstVreg, srcVreg, exMaskReg);
        if constexpr (std::is_same_v<OutT, int4x2_t>) {
            GenStoreInstr<InT, OutT>(dstUb + innerLoopStride * i /2, dstVreg, stMaskReg);
        } else {
            GenStoreInstr<InT, OutT>(dstUb + innerLoopStride * i, dstVreg, stMaskReg);
        }
    }
}

template <typename InT, typename OutT, AscendC::RoundMode roundMode>
inline __aicore__ void CastExtendB8(__local_mem__ OutT* dstUb, __local_mem__ InT* srcUb,
                                    const int64_t count, uint32_t repeat_times,
                                    uint32_t innerLoopStride){

    static constexpr MicroAPI::CastTrait cast_trait_in = {
        MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::NO_SAT, MicroAPI::MaskMergeMode::ZEROING,
        GetCastRoundMode<InT, half>()};
    static constexpr MicroAPI::CastTrait cast_trait_out = {
        MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::NO_SAT, MicroAPI::MaskMergeMode::ZEROING, 
        GetCastRoundMode<half, OutT>()};

    uint32_t sreg = static_cast<uint32_t>(count);
    MicroAPI::RegTensor<InT> srcVreg;
    MicroAPI::RegTensor<OutT> dstVreg;
    MicroAPI::RegTensor<half> tmpVreg;
    MicroAPI::MaskReg ldMaskReg, stMaskReg, exMaskReg, dumpMaskReg;
    exMaskReg = MicroAPI::CreateMask<uint8_t, MicroAPI::MaskPattern::ALL>();

    if constexpr (sizeof(OutT) == sizeof(float)) {
        sreg = sreg * 2;
    }

    for (uint32_t i = 0; i < repeat_times; i++){
        stMaskReg = MicroAPI::UpdateMask<half>(sreg);
        GenLoadInstr<InT, half>(srcVreg, srcUb + innerLoopStride * i);
        MicroAPI::Cast<half, InT, cast_trait_in>(tmpVreg, srcVreg, exMaskReg);
        if constexpr (sizeof(half) < sizeof(OutT)) {
            MicroAPI::UnPack<uint32_t, uint16_t>((MicroAPI::RegTensor<uint32_t> &)tmpVreg, 
                                                 (MicroAPI::RegTensor<uint16_t> &)tmpVreg);
        }
        MicroAPI::Cast<OutT, half,cast_trait_out>(dstVreg,tmpVreg, exMaskReg);
        if constexpr (SupportType<OutT, int4x2_t>()) {
            GenStoreInstr<half, OutT>(dstUb + innerLoopStride * i / 2, dstVreg, stMaskReg);
        } else {
            GenStoreInstr<half, OutT>(dstUb + innerLoopStride * i, dstVreg, stMaskReg);
        }
    }
}

template <typename InT, typename OutT, AscendC::RoundMode roundMode>
inline __aicore__ void CastExtendB64(__local_mem__ OutT* dstUb, __local_mem__ InT* srcUb,
                                    const int64_t count, uint32_t repeat_times,
                                    uint32_t innerLoopStride){
    static constexpr MicroAPI::CastTrait cast_trait = {MicroAPI::RegLayout::ZERO, 
                                                       MicroAPI::SatMode::NO_SAT, 
                                                       MicroAPI::MaskMergeMode::ZEROING, roundMode};
    uint32_t sreg = static_cast<uint32_t>(count);
    uint32_t b64_sreg = static_cast<uint32_t>(2 * count);
    MicroAPI::RegTensor<InT> srcVreg;
    MicroAPI::RegTensor<OutT> dstVreg;
    MicroAPI::RegTensor<uint32_t> zeroVreg, tmpVreg;
    MicroAPI::MaskReg maskReg, b64MaskReg;
    constexpr uint8_t elePerBlkInT = GetDataBlockSizeInBytes() / sizeof(InT);
    constexpr uint8_t elePerBlkOutT = GetDataBlockSizeInBytes() / sizeof(OutT);
    MicroAPI::MaskReg fullMask = MicroAPI::CreateMask<uint32_t, MicroAPI::MaskPattern::ALL>();
    MicroAPI::Duplicate(zeroVreg, 0, fullMask);

    for (uint32_t i = 0; i < repeat_times; i++){
        maskReg = MicroAPI::UpdateMask<uint32_t>(sreg);
        b64MaskReg = MicroAPI::UpdateMask<uint32_t>(b64_sreg);
        if constexpr (sizeof(OutT) == sizeof(int64_t)) {

            MicroAPI::DataCopy((MicroAPI::RegTensor<uint32_t>&)srcVreg, 
                              (__local_mem__ uint32_t *&)srcUb + innerLoopStride * i);
            MicroAPI::Interleave((MicroAPI::RegTensor<uint32_t> &)srcVreg, tmpVreg, 
                                 (MicroAPI::RegTensor<uint32_t> &)srcVreg, zeroVreg);
        } else {

            MicroAPI::DataCopy((MicroAPI::RegTensor<uint32_t>&)srcVreg, 
                               (__local_mem__ uint32_t *&)srcUb + innerLoopStride * i * 2);
        }
        MicroAPI::Cast<OutT, InT, cast_trait>(dstVreg, srcVreg, b64MaskReg);
        if constexpr (sizeof(OutT) == sizeof(int64_t)) {
            
            // MicroAPI::LocalMemBar<MicroAPI::MemType::VEC_STORE, MicroAPI::MemType::VEC_STORE>();
            MicroAPI::DataCopy<uint32_t>((__local_mem__ uint32_t *&)dstUb + innerLoopStride *i * 2, 
                                         (MicroAPI::RegTensor<uint32_t> &)dstVreg, b64MaskReg);
        } else {

            MicroAPI::DeInterleave((MicroAPI::RegTensor<uint32_t> &)dstVreg, tmpVreg, 
                                   (MicroAPI::RegTensor<uint32_t> &)dstVreg, zeroVreg);
            MicroAPI::LocalMemBar<MicroAPI::MemType::VEC_STORE, MicroAPI::MemType::VEC_STORE>();
            MicroAPI::DataCopy((__local_mem__ uint32_t *&)dstUb + innerLoopStride * i,
                               (MicroAPI::RegTensor<uint32_t> &)dstVreg, b64MaskReg);
        }
    }
}

template <typename InT, typename OutT, AscendC::RoundMode roundMode>
inline __aicore__ void CastExtendB64Transfer(__local_mem__ OutT* dstUb, __local_mem__ InT* srcUb,
                                             const int64_t count, uint32_t repeat_times,
                                             uint32_t innerLoopStride) {
    static constexpr MicroAPI::CastTrait cast_trait_in = {
        MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::NO_SAT, MicroAPI::MaskMergeMode::ZEROING,
        GetCastRoundMode<InT, float>()};
    static constexpr MicroAPI::CastTrait cast_trait_out = {
        MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::NO_SAT, MicroAPI::MaskMergeMode::ZEROING,
        GetCastRoundMode<float, OutT>()};
    uint32_t sreg = static_cast<uint32_t>(count);
    uint32_t b64Sreg = static_cast<uint32_t>(count * 2);
    MicroAPI::RegTensor<InT> srcVreg;
    MicroAPI::RegTensor<OutT> dstVreg;
    MicroAPI::RegTensor<float> midVreg;
    MicroAPI::RegTensor<uint32_t> zeroVreg, tmpVreg;
    MicroAPI::MaskReg b64MaskReg, b32MaskReg, maskReg;
    constexpr uint8_t elePerBlkInT = GetDataBlockSizeInBytes() / sizeof(InT);
    constexpr uint8_t elePerBlkOutT = GetDataBlockSizeInBytes() / sizeof(OutT);
    MicroAPI::MaskReg fullPreg = MicroAPI::CreateMask<uint32_t, MicroAPI::MaskPattern::ALL>();
    MicroAPI::Duplicate(zeroVreg, 0, fullPreg);

    for (uint32_t i = 0; i < repeat_times; i++){
        b32MaskReg = MicroAPI::UpdateMask<uint32_t>(sreg);
        b64MaskReg = MicroAPI::UpdateMask<uint32_t>(b64Sreg);
        if constexpr (sizeof(InT) < sizeof(float)) {

            MicroAPI::MaskPack(maskReg, b32MaskReg);
            GenLoadInstr<InT, float>(srcVreg,srcUb + innerLoopStride * i);
            MicroAPI::Cast<float, InT, cast_trait_in>(midVreg, srcVreg, b64MaskReg);

            MicroAPI::Interleave((MicroAPI::RegTensor<uint32_t> &)midVreg, tmpVreg,
                                 (MicroAPI::RegTensor<uint32_t> &)midVreg, zeroVreg);
            MicroAPI::Cast<OutT, float, cast_trait_out>(dstVreg, midVreg, b64MaskReg);
            // MicroAPI::LocalMemBar<MicroAPI::MemType::VEC_STORE, MicroAPI::MemType::VEC_STORE>();
            MicroAPI::DataCopy((__local_mem__ uint32_t *&)dstUb + innerLoopStride * i * 2,
                               (MicroAPI::RegTensor<uint32_t> &)dstVreg, b64MaskReg);
        } else {

            MicroAPI::DataCopy((MicroAPI::RegTensor<uint32_t> &)srcVreg,
                               (__local_mem__ uint32_t *&)srcUb + innerLoopStride * i * 2);
            MicroAPI::Cast<float, InT, cast_trait_in>(midVreg, srcVreg, b64MaskReg);
            MicroAPI::DeInterleave((MicroAPI::RegTensor<uint32_t> &)midVreg, tmpVreg, 
                                   (MicroAPI::RegTensor<uint32_t> &)midVreg, zeroVreg);
            
            MicroAPI::MaskPack(maskReg, b64MaskReg);
            MicroAPI::Cast<OutT, float, cast_trait_out>(dstVreg, midVreg, b64MaskReg);
            GenStoreInstr<float, OutT>(dstUb + innerLoopStride * i, dstVreg, maskReg);           
        }
    }
}

template <typename InT, typename OutT, AscendC::RoundMode roundMode>
inline __aicore__ void CastExtendInT(__local_mem__ OutT* dstUb, __local_mem__ InT* srcUb,
                                    const int64_t count, uint32_t repeat_times,
                                    uint32_t innerLoopStride)
{
    uint32_t sreg = static_cast<uint32_t>(count);
    MicroAPI::RegTensor<InT> srcVreg;
    MicroAPI::RegTensor<OutT> dstVreg;
    MicroAPI::RegTensor<half> tmpVreg;
    MicroAPI::MaskReg stMaskReg, exMaskReg;
    exMaskReg = MicroAPI::CreateMask<InT, MicroAPI::MaskPattern::ALL>();
    for (uint32_t i = 0; i < repeat_times; i++) {
        stMaskReg = MicroAPI::UpdateMask<InT>(sreg);
        GenLoadInstr<InT, OutT>(srcVreg, srcUb + innerLoopStride * i);
        MicroAPI::Copy<OutT>(dstVreg, (MicroAPI::RegTensor<OutT> &)srcVreg);
        GenStoreInstr<InT, OutT>(dstUb + innerLoopStride * i, dstVreg, stMaskReg);
    }
}

template <typename InT, typename OutT, AscendC::RoundMode roundMode>
inline __aicore__ void CastExtendS64U8(__local_mem__ OutT* dstUb, __local_mem__ InT* srcUb,
                                    const int64_t count, uint32_t repeat_times,
                                    uint32_t innerLoopStride)
{
    uint32_t sreg = static_cast<uint32_t>(count);
    MicroAPI::RegTensor<InT> srcVreg;
    MicroAPI::RegTensor<OutT> dstVreg;
    MicroAPI::RegTensor<InT> zeroVreg;
    MicroAPI::RegTensor<InT> maxVreg;
    MicroAPI::RegTensor<int64_t> tmp64Vreg;
    MicroAPI::RegTensor<uint32_t> tmp32Vreg;
    MicroAPI::RegTensor<uint16_t> tmp16Vreg;
    MicroAPI::MaskReg stMaskReg, exMaskReg;
    MicroAPI::MaskReg cmpMask;

    MicroAPI::MaskReg ubMask = MicroAPI::CreateMask<uint8_t, MicroAPI::MaskPattern::VL32>();
    exMaskReg = MicroAPI::CreateMask<InT, MicroAPI::MaskPattern::ALL>();
    MicroAPI::Duplicate(zeroVreg, 0);
    MicroAPI::Duplicate(maxVreg, 255);
    
    uint32_t countTail = count % innerLoopStride;
    uint32_t tailCtrl = (countTail + count - 1) / count;
    for (uint32_t i = 0; i < tailCtrl; i++) {
        repeat_times -= 1;
    }
    for (uint32_t i = 0; i < repeat_times; i++) {
        stMaskReg = ubMask;
        MicroAPI::DataCopy(srcVreg, srcUb + innerLoopStride * i);
        MicroAPI::Copy(tmp64Vreg, srcVreg);
        MicroAPI::Compare<InT, CMPMODE::LT>(cmpMask, srcVreg, zeroVreg, exMaskReg);
        MicroAPI::Select(srcVreg, zeroVreg, tmp64Vreg, cmpMask);
        MicroAPI::Compare<InT, CMPMODE::GT>(cmpMask, srcVreg, maxVreg, exMaskReg);
        MicroAPI::Select(srcVreg, maxVreg, srcVreg, cmpMask);
        MicroAPI::Pack<uint32_t, int64_t, MicroAPI::HighLowPart::LOWEST>(tmp32Vreg, srcVreg);
        MicroAPI::Pack<uint16_t, uint32_t, MicroAPI::HighLowPart::LOWEST>(tmp16Vreg, tmp32Vreg);
        MicroAPI::Pack<uint8_t, uint16_t, MicroAPI::HighLowPart::LOWEST>(dstVreg, tmp16Vreg);
        MicroAPI::DataCopy(dstUb + innerLoopStride * i, dstVreg, stMaskReg);
    }
    for (uint32_t i = 0; i < tailCtrl; i++) {
        stMaskReg = MicroAPI::UpdateMask<OutT>(countTail);
        MicroAPI::DataCopy(srcVreg, srcUb + innerLoopStride * repeat_times);
        MicroAPI::Copy(tmp64Vreg, srcVreg);
        MicroAPI::Compare<InT, CMPMODE::LT>(cmpMask, srcVreg, zeroVreg, exMaskReg);
        MicroAPI::Select(srcVreg, zeroVreg, tmp64Vreg, cmpMask);
        MicroAPI::Compare<InT, CMPMODE::GT>(cmpMask, srcVreg, maxVreg, exMaskReg);
        MicroAPI::Select(srcVreg, maxVreg, srcVreg, cmpMask);
        MicroAPI::Pack<uint32_t, int64_t, MicroAPI::HighLowPart::LOWEST>(tmp32Vreg, srcVreg);
        MicroAPI::Pack<uint16_t, uint32_t, MicroAPI::HighLowPart::LOWEST>(tmp16Vreg, tmp32Vreg);
        MicroAPI::Pack<uint8_t, uint16_t, MicroAPI::HighLowPart::LOWEST>(dstVreg, tmp16Vreg);
        MicroAPI::DataCopy(dstUb + innerLoopStride * repeat_times, dstVreg, stMaskReg);
    }
}

template <auto func, typename InT, typename OutT, AscendC::RoundMode roundMode, uint8_t dim>
inline __aicore__ void CastExtendImpl(__local_mem__ OutT* dstUb, __local_mem__ InT* srcUb,
                                      const int64_t count, uint32_t repeat_times,
                                      uint32_t innerLoopStride, const uint32_t (&output_dims)[dim],
                                      const uint32_t (&output_stride)[dim],
                                      const uint32_t (&input_stride)[dim]) 
{
    static_assert(dim == 1 || dim == 2 || dim == 3, "CastExtend dim exceeds maximum 3");
    if constexpr(dim == 1) {
        func(dstUb, srcUb, count, repeat_times, innerLoopStride);
    } else if constexpr(dim == 2) {
        uint32_t loop_num = uint32_t(output_dims[0]);
        for (uint32_t i = 0; i < loop_num; i++) {
            if constexpr (std::is_same_v<InT, int4x2_t>) {
                func(dstUb + i * output_stride[0], srcUb + i * input_stride[0] / 2, count, repeat_times, 
                     innerLoopStride);
            } else if constexpr (std::is_same_v<OutT, int4x2_t>) {
                func(dstUb + i * output_stride[0] / 2, srcUb + i * input_stride[0], count, repeat_times, 
                     innerLoopStride);
            } else {
                func(dstUb + i * output_stride[0], srcUb + i * input_stride[0], count, repeat_times, 
                     innerLoopStride);
            }
        }
    } 
}

template <typename InT, typename OutT, uint8_t dim>
inline __aicore__ void CastExtend(const AscendC::LocalTensor<OutT> &dst,
                                  const AscendC::LocalTensor<InT> &src,
                                      const uint32_t (&output_dims)[dim],
                                      const uint32_t (&output_stride)[dim],
                                      const uint32_t (&input_stride)[dim]) {

    constexpr AscendC::RoundMode roundMode = GetCastRoundMode<InT, OutT>();
    __local_mem__ InT *srcUb = (__local_mem__ InT *)src.GetPhyAddr();
    __local_mem__ OutT *dstUb = (__local_mem__ OutT *)dst.GetPhyAddr();

    uint32_t count = output_dims[dim - 1];
    constexpr uint32_t repeatStrideSrc = static_cast<uint32_t>(VECTOR_REG_WIDTH / sizeof(InT));
    constexpr uint32_t repeatStrideDst = static_cast<uint32_t>(VECTOR_REG_WIDTH / sizeof(OutT));
    uint32_t innerLoopStride = repeatStrideSrc > repeatStrideDst ? repeatStrideDst : repeatStrideSrc;
    uint32_t repeat_times = static_cast<uint32_t>((count + innerLoopStride - 1) / innerLoopStride);

    if constexpr (AscendC::IsSameType<InT, uint8_t>::value && (AscendC::SupportType<OutT, int4x2_t>())) {
        innerLoopStride = static_cast<uint32_t>(VECTOR_REG_WIDTH / sizeof(half));
        repeat_times = static_cast<uint32_t>((count + innerLoopStride - 1) / innerLoopStride);
    }

    constexpr bool b4Cast = 
        SupportType<Tuple<OutT, InT>, Tuple<half, int4x2_t>, Tuple<int4x2_t, half>>(); 

    constexpr bool b8Cast = AscendC::IsSameType<InT, uint8_t>::value &&
                            AscendC::SupportType<OutT, float, int32_t, int16_t, int4x2_t>();

    constexpr bool b64Cast = 
        SupportType<Tuple<OutT, InT>, Tuple<float, int64_t>, Tuple<int64_t, float>,
                    Tuple<int32_t, int64_t>, Tuple<int64_t, int32_t>>();

    constexpr bool b64CastWithTransfer = 
        SupportType<Tuple<OutT, InT>, Tuple<half, int64_t>, Tuple<int64_t, half>>();
    
    constexpr bool castWithSameBit = 
        SupportType<Tuple<OutT, InT>, Tuple<uint8_t, int8_t>, Tuple<int8_t, uint8_t>, Tuple<uint16_t, int16_t>,
                    Tuple<int16_t, uint16_t>, Tuple<uint32_t, int32_t>, Tuple<int32_t, uint32_t>,
                    Tuple<uint64_t, int64_t>, Tuple<int64_t, uint64_t>>();
    
    constexpr bool s64U8Cast = SupportType<Tuple<OutT, InT>, Tuple<uint8_t, int64_t>>();

    if constexpr (b4Cast) {
        constexpr auto func = CastExtendB4<InT, OutT, roundMode>;
        VF_CALL<CastExtendImpl<func, InT, OutT, roundMode, dim>>(dstUb, srcUb, count, repeat_times, innerLoopStride,
                                                                 output_dims, output_stride, input_stride);
    } else if constexpr (b8Cast) {
        constexpr auto func = CastExtendB8<InT, OutT, roundMode>;
        VF_CALL<CastExtendImpl<func, InT, OutT, roundMode, dim>>(dstUb, srcUb, count, repeat_times, innerLoopStride,
                                                                 output_dims, output_stride, input_stride);
    } else if constexpr (b64CastWithTransfer) {
        constexpr auto func = CastExtendB64Transfer<InT, OutT, roundMode>;
        VF_CALL<CastExtendImpl<func, InT, OutT, roundMode, dim>>(dstUb, srcUb, count, repeat_times, innerLoopStride,
                                                                 output_dims, output_stride, input_stride);
    } else if constexpr (b64Cast) {
        constexpr auto func = CastExtendB64<InT, OutT, roundMode>;
        VF_CALL<CastExtendImpl<func, InT, OutT, roundMode, dim>>(dstUb, srcUb, count, repeat_times, innerLoopStride,
                                                                 output_dims, output_stride, input_stride);
    } else if constexpr (castWithSameBit) {
        constexpr auto func = CastExtendInT<InT, OutT, roundMode>;
        VF_CALL<CastExtendImpl<func, InT, OutT, roundMode, dim>>(dstUb, srcUb, count, repeat_times, innerLoopStride,
                                                                 output_dims, output_stride, input_stride);
    } else if constexpr (s64U8Cast) {
        constexpr auto func = CastExtendS64U8<InT, OutT, roundMode>;
        VF_CALL<CastExtendImpl<func, InT, OutT, roundMode, dim>>(dstUb, srcUb, count, repeat_times, innerLoopStride,
                                                                 output_dims, output_stride, input_stride);
    } else {
        constexpr auto func = CastExtendCommon<InT, OutT, roundMode>;
        VF_CALL<CastExtendImpl<func, InT, OutT, roundMode, dim>>(dstUb, srcUb, count, repeat_times, innerLoopStride,
                                                                 output_dims, output_stride, input_stride);
    }
}
#endif 