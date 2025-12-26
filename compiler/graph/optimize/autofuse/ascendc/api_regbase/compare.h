/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef __ASCENDC_API_REGBASE_COMPARE_H__
#define __ASCENDC_API_REGBASE_COMPARE_H__

#include "kernel_operator.h"
using namespace AscendC;
template <typename InT, CMPMODE mode, uint16_t vlSize, bool isScalar = false>
__simd_vf__ inline void CompareNormal2DVecImpl(__local_mem__ uint8_t *dst, __local_mem__ InT *src0,
                                               __local_mem__ InT *src1, const uint16_t dstStride, 
    const uint16_t srcStride, const uint16_t repeatTime, 
    const uint16_t counterFirst, uint32_t counterTail, InT scalar) 
{
    MicroAPI::MaskReg fullMask, tailMask, dstMask;
    MicroAPI::RegTensor<InT> src0Reg, src1Reg;
    MicroAPI::RegTensor<uint8_t> dstReg, oneAllReg, zeroAllReg;
    fullMask = MicroAPI::CreateMask<uint8_t>();
    MicroAPI::Duplicate(oneAllReg, 1);
    MicroAPI::Duplicate(zeroAllReg, 0);
    if constexpr (isScalar) {
        MicroAPI::Duplicate(src1Reg, scalar);
    }
    uint32_t mainBlockCount = GetVecLen() / sizeof(InT);
    MicroAPI::MaskReg mainBlockMask = MicroAPI::UpdateMask<uint8_t>(mainBlockCount);
    MicroAPI::MaskReg tailBlockMask = MicroAPI::UpdateMask<uint8_t>(counterTail);
    for (uint16_t j = 0U; j < counterFirst; ++j) {
      //mainBlock
        for (uint16_t i = 0U; i < repeatTime; ++i) {
          MicroAPI::DataCopy(src0Reg, src0 + j * srcStride + i * vlSize);
          if constexpr (!isScalar) {
              MicroAPI::DataCopy(src1Reg, src1 + j * srcStride + i * vlSize);
          }
          MicroAPI::Compare<InT, mode>(dstMask, src0Reg, src1Reg, fullMask);
          if constexpr (sizeof(InT) == 2) {
              MicroAPI::MaskPack(dstMask, dstMask);
          } else if constexpr (sizeof(InT) == 4) {
              MicroAPI::MaskPack(dstMask, dstMask);
              MicroAPI::MaskPack(dstMask, dstMask);
          } else if constexpr (sizeof(InT) == 8) {
              MicroAPI::MaskPack(dstMask, dstMask);
              MicroAPI::MaskPack(dstMask, dstMask);
              MicroAPI::MaskPack(dstMask, dstMask);
          }
          MicroAPI::Select(dstReg, oneAllReg, zeroAllReg, dstMask);
          MicroAPI::DataCopy(dst + j * dstStride + i * vlSize, dstReg, mainBlockMask);
        }
      //tailBlock
      MicroAPI::DataCopy(src0Reg, src0 + j * srcStride + repeatTime * vlSize);
      if constexpr (!isScalar) {
          MicroAPI::DataCopy(src1Reg, src1 + j * srcStride + repeatTime * vlSize);
      }
      MicroAPI::Compare<InT, mode>(dstMask, src0Reg, src1Reg, fullMask);
      if constexpr (sizeof(InT) == 2) {
          MicroAPI::MaskPack(dstMask, dstMask);
      } else if constexpr (sizeof(InT) == 4) {
          MicroAPI::MaskPack(dstMask, dstMask);
          MicroAPI::MaskPack(dstMask, dstMask);
      } else if constexpr (sizeof(InT) == 8) {
          MicroAPI::MaskPack(dstMask, dstMask);
          MicroAPI::MaskPack(dstMask, dstMask);
          MicroAPI::MaskPack(dstMask, dstMask);
      }
      MicroAPI::Select(dstReg, oneAllReg, zeroAllReg, dstMask);
      MicroAPI::DataCopy(dst + j * dstStride + repeatTime * vlSize, dstReg, tailBlockMask);
    }
}

template <typename InT, uint8_t dim, CMPMODE mode>
__aicore__ inline void CompareExtend(const LocalTensor<uint8_t> &dst,
                                     const LocalTensor<InT> &src0,
                                     const LocalTensor<InT> &src1,
                                     const uint16_t (&output_dims)[dim],
                                     const uint16_t (&output_stride)[dim],
                                     const uint16_t (&input_stride)[dim]) 
{
    CheckTensorPosition(dst, "dst", "VECIN, VECOUT, VECCALC");
    CheckTensorPosition(src0, "src0", "VECIN, VECOUT, VECCALC");
    CheckTensorPosition(src1, "src1", "VECIN, VECOUT, VECCALC");
    bool src1IsScalar = false;
    InT scalar = 0;
    constexpr uint16_t vlSize = static_cast<uint32_t>(GetVecLen() / sizeof(InT));
    if (src1.GetSize() * sizeof(InT) == 32) {
        src1IsScalar = true;
        scalar = src1.GetValue(0);
    }
    static_assert((dim == 1) || (dim == 2),"CompareExtend only support dim=1 or dim=2");
    __local_mem__ uint8_t *dstLocal = (__local_mem__ uint8_t *)dst.GetPhyAddr();
    __local_mem__ InT *src0Local = (__local_mem__ InT *)src0.GetPhyAddr();
    __local_mem__ InT *src1Local = (__local_mem__ InT *)src1.GetPhyAddr();
    const uint16_t dstStride = dim == 1 ? 1 : output_stride[0];
    const uint16_t srcStride = dim == 1 ? 1 : input_stride[0];
    uint16_t repeat = dim == 1 ? output_dims[0] / vlSize : output_dims[1] / vlSize;
    uint16_t counterFirst = dim == 1 ? 1 : output_dims[0];
    uint32_t counterTail = dim == 1 ? output_dims[0] - repeat * vlSize : output_dims[1] - repeat * vlSize;
    if ((counterTail == 0) && (repeat > 0)) {
        repeat--;
        counterTail += vlSize;
    }
    if (src1IsScalar) {
        CompareNormal2DVecImpl<InT, mode, vlSize, true>(
            dstLocal, src0Local, src1Local, dstStride, srcStride, repeat, counterFirst, counterTail, scalar);
    } else {
        CompareNormal2DVecImpl<InT, mode, vlSize>(
            dstLocal, src0Local, src1Local,dstStride, srcStride, repeat, counterFirst, counterTail, scalar);
    }
}

template <typename InT, uint8_t dim, CMPMODE mode>
__aicore__ inline void CompareScalarExtend(const LocalTensor<uint8_t> &dst,
                                           const LocalTensor<InT> &src0,
                                           const InT srcScalar,
                                           const uint16_t (&output_dims)[dim],
                                           const uint16_t (&output_stride)[dim],
                                           const uint16_t (&input_stride)[dim]) 
{
    CheckTensorPosition(dst, "dst", "VECIN, VECOUT, VECCALC");
    CheckTensorPosition(src0, "src0", "VECIN, VECOUT, VECCALC");
    bool src1IsScalar = true;
    constexpr uint16_t vlSize = static_cast<uint32_t>(GetVecLen() / sizeof(InT));
    static_assert((dim == 1) || (dim == 2),"CompareExtend only support dim=1 or dim=2");
    __local_mem__ uint8_t *dstLocal = (__local_mem__ uint8_t *)dst.GetPhyAddr();
    __local_mem__ InT *src0Local = (__local_mem__ InT *)src0.GetPhyAddr();
    __local_mem__ InT *src1Local = (__local_mem__ InT *)src0.GetPhyAddr();
    const uint16_t dstStride = dim == 1 ? 0 : output_stride[0];
    const uint16_t srcStride = dim == 1 ? 0 : input_stride[0];
    uint16_t repeat = dim == 1 ? output_dims[0] / vlSize : output_dims[1] / vlSize;
    uint16_t counterFirst = dim == 1 ? 1 : output_dims[0];
    uint32_t counterTail = dim == 1 ? output_dims[0] - repeat * vlSize : output_dims[1] - repeat * vlSize;
    if ((counterTail == 0) && (repeat > 0)) {
        repeat--;
        counterTail += vlSize;
    }
    CompareNormal2DVecImpl<InT, mode, vlSize, true>(
        dstLocal, src0Local, src1Local, dstStride, srcStride, repeat, counterFirst, counterTail, srcScalar);
}
#endif  //__ASCENDC_API_REGBASE__COMPARE_H__
