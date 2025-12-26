/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef __ASCENDC_API_REGBASE_LOGICAL_NOT_H
#define __ASCENDC_API_REGBASE_LOGICAL_NOT_H

template <typename T>
inline __aicore__ void LogicalNotExtend(const AscendC::LocalTensor<T> &dst, const AscendC::LocalTensor<T> &src,
                                        AscendC::LocalTensor<uint8_t> &tmp_buf, const uint32_t count) {
  if constexpr (AscendC::IsSameType<T, half>::value) {
    AscendC::LocalTensor<bool> boolean_buf = tmp_buf.ReinterpretCast<bool>();
    AscendC::LogicalNot(boolean_buf, src, count);
    AscendC::Cast(dst, tmp_buf, AscendC::RoundMode::CAST_NONE, count);
  }
  if constexpr (AscendC::IsSameType<T, float>::value || AscendC::IsSameType<T, int32_t>::value || AscendC::IsSameType<T, int16_t>::value ) {
    auto tmp_buf_uint8 = tmp_buf;
    AscendC::LocalTensor<bool> boolean_buf = tmp_buf_uint8.ReinterpretCast<bool>();
    AscendC::LogicalNot(boolean_buf, src, count);
    auto new_size = KernelUtils::BlkAlign<uint8_t>(count);
    auto tmp_buf_half = tmp_buf[new_size].ReinterpretCast<half>();
    AscendC::Cast(tmp_buf_half, tmp_buf_uint8, AscendC::RoundMode::CAST_NONE, count);
    auto cast_mode = AscendC::RoundMode::CAST_NONE;
    if constexpr (AscendC::IsSameType<T, int32_t>::value || AscendC::IsSameType<T, int16_t>::value ) {
      cast_mode = AscendC::RoundMode::CAST_TRUNC;
    }
    AscendC::Cast(dst, tmp_buf_half, cast_mode, count);
  }

  if constexpr (AscendC::IsSameType<T, uint8_t>::value) {
    auto dst_bool = dst.template ReinterpretCast<bool>();
    AscendC::LogicalNot(dst_bool, src, count);
  }
}
#endif  // __ASCENDC_API_REGBASE_LOGICAL_NOT_H
