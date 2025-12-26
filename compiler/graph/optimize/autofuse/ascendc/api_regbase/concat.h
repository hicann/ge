/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef __ASCENDC_API_REGBASE_CONCAT_H__
#define __ASCENDC_API_REGBASE_CONCAT_H__

namespace concat {
template<size_t INPUT_NUM>
struct ConcatTiling {
  uint32_t num_rows;
  uint32_t num_dst_cols;
  uint32_t num_srcs_cols[INPUT_NUM];
};

template<size_t INPUT_NUM>
struct ConcatTilingPadded {
  uint32_t num_rows;
  uint32_t num_dst_cols;
  uint32_t num_srcs_cols[INPUT_NUM];
  uint32_t src_row_strides[INPUT_NUM];
  uint32_t src_second_last_dim_strides[INPUT_NUM];
  uint32_t gather_mask_dim_sizes[INPUT_NUM];
};

template<size_t INPUT_NUM>
struct ConcatTilingOneAxis {
  uint32_t src_col_sizes[INPUT_NUM];
  uint32_t dst_col_offsets[INPUT_NUM];
};

template<typename T>
struct ArangeTypeGet {
};

template<>
struct ArangeTypeGet<uint32_t> {
  using T = int32_t;
};

template<>
struct ArangeTypeGet<uint16_t> {
  using T = int16_t;
};

template<typename U>
__aicore__ inline void GenSequence(__ubuf__ U *seq_addr) {
  uint32_t kVfLen = AscendC::VECTOR_REG_WIDTH / sizeof(U);
  using SeqType = typename ArangeTypeGet<U>::T;
  auto seq_buf_addr = (__ubuf__ SeqType *) seq_addr;
  __VEC_SCOPE__ {
    AscendC::MicroAPI::RegTensor<SeqType> reg_seq;
    AscendC::MicroAPI::Arange(reg_seq, 0);
    uint32_t num = kVfLen;
    AscendC::MicroAPI::MaskReg p0 = AscendC::MicroAPI::UpdateMask<SeqType>(num);
    AscendC::MicroAPI::DataCopy(seq_buf_addr, reg_seq, p0);
  }
}

template<typename T>
__aicore__ inline void Copy(__ubuf__ T *dst_addr_base,
                            __ubuf__ T *src_addr_base,
                            uint32_t rows,
                            uint32_t cols,
                            uint32_t row_stride) {
  constexpr uint32_t kVfLen = AscendC::VECTOR_REG_WIDTH / sizeof(T);
  auto num_rows = static_cast<uint16_t>(rows);
  uint16_t repeat_times = cols / kVfLen;
  uint32_t tail_cols = cols - repeat_times * kVfLen;

  auto src_addr = src_addr_base;
  __VEC_SCOPE__{
    AscendC::MicroAPI::UnalignReg u0;
    AscendC::MicroAPI::UnalignReg u_reg;
    AscendC::MicroAPI::RegTensor<T> vd0;
    for (uint16_t i = 0; i < num_rows; ++i) {
      auto dst_addr = dst_addr_base + i * row_stride;
      for (uint16_t j = 0; j < repeat_times; ++j) {
        AscendC::MicroAPI::DataCopyUnAlignPre(u0, src_addr);
        AscendC::MicroAPI::DataCopyUnAlign(vd0, u0, src_addr, kVfLen);
        AscendC::MicroAPI::DataCopyUnAlign(dst_addr, vd0, u_reg, kVfLen);
      }
      AscendC::MicroAPI::DataCopyUnAlignPre(u0, src_addr);
      AscendC::MicroAPI::DataCopyUnAlign(vd0, u0, src_addr, tail_cols);
      AscendC::MicroAPI::DataCopyUnAlign(dst_addr, vd0, u_reg, tail_cols);
      AscendC::MicroAPI::DataCopyUnAlignPost(dst_addr, u_reg, 0);
    }
  }
}

template<typename T>
__aicore__ inline void CopyPadded(__ubuf__ T *dst_addr_base,
                                  __ubuf__ T *src_addr_base,
                                  uint32_t rows,
                                  uint32_t src_row_stride,
                                  uint32_t row_stride,
                                  uint32_t gather_mask_repeat_stride,
                                  uint32_t gather_mask_dim_size) {
  constexpr uint32_t kVfLen = AscendC::VECTOR_REG_WIDTH / sizeof(T);
  auto num_rows = static_cast<uint16_t>(rows);
  uint16_t repeat_times = src_row_stride / gather_mask_repeat_stride;
  uint16_t sub_repeat_times = gather_mask_dim_size / kVfLen;
  uint16_t sub_tail_cols = gather_mask_dim_size - sub_repeat_times * kVfLen;
  __VEC_SCOPE__{
    AscendC::MicroAPI::UnalignReg u0;
    AscendC::MicroAPI::UnalignReg u_reg;
    AscendC::MicroAPI::RegTensor<T> vd0;
    AscendC::MicroAPI::RegTensor<T> vd1;
    for (uint16_t i = 0; i < num_rows; ++i) {
      auto dst_addr = dst_addr_base + i * row_stride;
      for (uint16_t j = 0; j < repeat_times; ++j) {
        auto src_addr = src_addr_base + i * src_row_stride + j * gather_mask_repeat_stride;
        for (uint16_t k = 0; k < sub_repeat_times; ++k) {
          AscendC::MicroAPI::DataCopyUnAlignPre(u0, src_addr);
          AscendC::MicroAPI::DataCopyUnAlign(vd0, u0, src_addr, kVfLen);
          AscendC::MicroAPI::DataCopyUnAlign(dst_addr, vd0, u_reg, kVfLen);
        }
        AscendC::MicroAPI::DataCopyUnAlignPre(u0, src_addr);
        AscendC::MicroAPI::DataCopyUnAlign(vd0, u0, src_addr, sub_tail_cols);
        AscendC::MicroAPI::DataCopyUnAlign(dst_addr, vd0, u_reg, sub_tail_cols);
        AscendC::MicroAPI::DataCopyUnAlignPost(dst_addr, u_reg, 0);
      }
    }
  }
}

template<typename U>
__aicore__ inline void GenScatterIndex(uint32_t src_cols,
                                       uint32_t dst_cols,
                                       uint32_t dst_col_offset,
                                       __ubuf__ U *seq_addr,
                                       __ubuf__ U *index_addr) {
  constexpr uint32_t kVfLen = AscendC::VECTOR_REG_WIDTH / sizeof(U);
  __VEC_SCOPE__{
    AscendC::MicroAPI::RegTensor<U> v0;
    AscendC::MicroAPI::RegTensor<U> v1;
    AscendC::MicroAPI::RegTensor<U> v2;
    AscendC::MicroAPI::RegTensor<U> v3;

    AscendC::MicroAPI::RegTensor<U> vd0;
    AscendC::MicroAPI::RegTensor<U> vd2;
    AscendC::MicroAPI::RegTensor<U> vd3;
    AscendC::MicroAPI::RegTensor<U> vd6;
    AscendC::MicroAPI::RegTensor<U> vd7;
    AscendC::MicroAPI::RegTensor<U> vd10;

    auto num = kVfLen;
    AscendC::MicroAPI::MaskReg p0 = AscendC::MicroAPI::UpdateMask<U>(num);
    AscendC::MicroAPI::DataCopy(v0, seq_addr);
    AscendC::MicroAPI::Duplicate(v1, (U) src_cols, p0);
    AscendC::MicroAPI::Duplicate(v2, (U) dst_col_offset, p0);
    AscendC::MicroAPI::Duplicate(v3, (U) dst_cols, p0);
    AscendC::MicroAPI::Div(vd2, v0, v1, p0);
    AscendC::MicroAPI::Mul(vd6, vd2, v1, p0);
    AscendC::MicroAPI::Sub(vd7, v0, vd6, p0);
    AscendC::MicroAPI::Mul(vd0, vd2, v3, p0);
    AscendC::MicroAPI::Add(vd3, vd0, vd7, p0);
    AscendC::MicroAPI::Add(vd10, vd3, v2, p0);
    AscendC::MicroAPI::DataCopy(index_addr, vd10, p0);
  }
}

template<typename T, typename U>
__aicore__ inline void ScatterInput(__ubuf__ T *dst_addr,
                                    __ubuf__ T *src_addr,
                                    __ubuf__ U *index_addr,
                                    uint32_t rows,
                                    uint32_t dst_cols,
                                    uint32_t src_cols) {
  constexpr uint32_t kVfLen = AscendC::VECTOR_REG_WIDTH / sizeof(U);
  auto rows_per_loop = static_cast<uint16_t>(kVfLen / src_cols);
  auto loop_times = static_cast<uint16_t>(rows / rows_per_loop);
  uint16_t tail_rows = rows - loop_times * rows_per_loop;

  __VEC_SCOPE__{
    AscendC::MicroAPI::RegTensor<T> vd2;
    AscendC::MicroAPI::RegTensor<T> src;
    AscendC::MicroAPI::RegTensor<T> tmp;
    AscendC::MicroAPI::RegTensor<T> dst0;
    AscendC::MicroAPI::RegTensor<T> dst1;
    AscendC::MicroAPI::RegTensor<U> vd0;
    AscendC::MicroAPI::RegTensor<U> vd1;
    AscendC::MicroAPI::UnalignReg u0;

    uint32_t num = rows_per_loop * src_cols;
    uint32_t tail_num = tail_rows * src_cols;
    uint32_t pnum = num;
    uint32_t tail_pnum = tail_num;
    AscendC::MicroAPI::MaskReg p0 = AscendC::MicroAPI::UpdateMask<U>(num);
    AscendC::MicroAPI::MaskReg p1 = AscendC::MicroAPI::UpdateMask<U>(tail_num);

    AscendC::MicroAPI::DataCopy(vd0, index_addr);
    AscendC::MicroAPI::DataCopyUnAlignPre(u0, src_addr);
    for (uint16_t i = 0; i < loop_times; ++i) {
      AscendC::MicroAPI::DataCopyUnAlign(vd2, u0, src_addr, pnum);
      AscendC::MicroAPI::Adds(vd1, vd0, static_cast<U>(i * rows_per_loop * dst_cols), p0);
      if constexpr (sizeof(T) == 1) {
        AscendC::MicroAPI::Interleave(dst0, dst1, vd2, tmp);
        AscendC::MicroAPI::DataCopyScatter(dst_addr, dst0, vd1, p0);
      } else {
        AscendC::MicroAPI::DataCopyScatter(dst_addr, vd2, vd1, p0);
      }
    }
    AscendC::MicroAPI::DataCopyUnAlign(vd2, u0, src_addr, tail_pnum);
    AscendC::MicroAPI::Adds(vd1, vd0, static_cast<U>(loop_times * rows_per_loop * dst_cols), p1);
    if constexpr (sizeof(T) == 1) {
      AscendC::MicroAPI::Interleave(dst0, dst1, vd2, tmp);
      AscendC::MicroAPI::DataCopyScatter(dst_addr, dst0, vd1, p1);
    } else {
      AscendC::MicroAPI::DataCopyScatter(dst_addr, vd2, vd1, p1);
    }
  }
}

template <typename U>
__aicore__ inline AscendC::MicroAPI::MaskReg GenMaskReg(uint32_t vf_len, uint32_t block_size,
                                                        uint32_t gather_mask_repeat_stride,
                                                        uint32_t gather_mask_dim_size, __ubuf__ uint32_t *index_addr) {
  constexpr uint16_t kDataBlockSize = 32 / sizeof(U);
  uint16_t loop_times = static_cast<uint16_t>(vf_len / block_size);
  // 不对齐, 必然有tail
  uint16_t all_num = (gather_mask_repeat_stride / kDataBlockSize) - 1;
  uint16_t tail_num = gather_mask_dim_size % (32 / sizeof(U));
  uint16_t index = 0;
  for (uint16_t i = 0; i < loop_times; ++i) {
    for (uint16_t j = 0; j < all_num; ++j) {
      index_addr[index++] = 0xffffffffU;
    }
    index_addr[index++] = (1U << (tail_num * sizeof(U))) - 1U;
  }
  for (; index < 8; ++index) {
    index_addr[index] = 0;
  }

  MicroAPI::MaskReg p_copy_mask = MicroAPI::CreateMask<U, MicroAPI::MaskPattern::ALLF>();
  MicroAPI::DataCopy(p_copy_mask, index_addr);
  return p_copy_mask;
}

template <typename T, typename U>
__aicore__ inline void ScatterInputPadded(__ubuf__ T *dst_addr, __ubuf__ T *src_addr, __ubuf__ U *index_addr,
                                          uint32_t rows, uint32_t dst_cols, uint32_t src_cols, uint32_t src_row_stride,
                                          uint32_t gather_mask_repeat_stride, uint32_t gather_mask_dim_size) {
  constexpr uint32_t kVfLen = VECTOR_REG_WIDTH / sizeof(U);
  auto rows_per_loop = static_cast<uint16_t>(kVfLen / src_row_stride);
  auto loop_times = static_cast<uint16_t>(rows / rows_per_loop);
  uint16_t tail_rows = rows - loop_times * rows_per_loop;
  __VEC_SCOPE__ {
    MicroAPI::RegTensor<T> vd_src;
    MicroAPI::RegTensor<T> vd_src_unpadded;
    MicroAPI::RegTensor<T> tmp;
    MicroAPI::RegTensor<T> dst0;
    MicroAPI::RegTensor<T> dst1;
    MicroAPI::RegTensor<U> vd_dst_index_base;
    MicroAPI::RegTensor<U> vd_dst_index;

    uint32_t num = rows_per_loop * src_cols;
    uint32_t tail_num = tail_rows * src_cols;
    MicroAPI::MaskReg p0 = MicroAPI::UpdateMask<U>(num);
    MicroAPI::MaskReg p1 = MicroAPI::UpdateMask<U>(tail_num);
    uint32_t main_num = rows_per_loop * src_row_stride;

    auto tmp_buf_addr = (__ubuf__ uint32_t *)(index_addr + kVfLen);
    MicroAPI::MaskReg copy_in_mask =
        GenMaskReg<U>(kVfLen, src_row_stride, gather_mask_repeat_stride, gather_mask_dim_size, tmp_buf_addr);
    MicroAPI::DataCopy(vd_dst_index_base, index_addr);

    for (uint16_t i = 0; i < loop_times; ++i) {
      MicroAPI::DataCopy<T, MicroAPI::PostLiteral::POST_MODE_UPDATE>(vd_src, src_addr, main_num);
      MicroAPI::GatherMask(vd_src_unpadded, vd_src, copy_in_mask);
      MicroAPI::Adds(vd_dst_index, vd_dst_index_base, static_cast<U>(i * rows_per_loop * dst_cols), p0);
      if constexpr (sizeof(T) == 1) {
        MicroAPI::Interleave(dst0, dst1, vd_src_unpadded, tmp);
        MicroAPI::DataCopyScatter(dst_addr, dst0, vd_dst_index, p0);
      } else {
        MicroAPI::DataCopyScatter(dst_addr, vd_src_unpadded, vd_dst_index, p0);
      }
    }
    if (tail_rows > 0) {
      MicroAPI::DataCopy(vd_src, src_addr);
      MicroAPI::GatherMask(vd_src_unpadded, vd_src, copy_in_mask);
      MicroAPI::Adds(vd_dst_index, vd_dst_index_base, static_cast<U>(loop_times * rows_per_loop * dst_cols), p1);
      if constexpr (sizeof(T) == 1) {
        MicroAPI::Interleave(dst0, dst1, vd_src_unpadded, tmp);
        MicroAPI::DataCopyScatter(dst_addr, dst0, vd_dst_index, p1);
      } else {
        MicroAPI::DataCopyScatter(dst_addr, vd_src_unpadded, vd_dst_index, p1);
      }
    }
  }
}

template <typename T, typename U, size_t INPUT_NUM>
__aicore__ inline void ConcatExtendInner(T *dst_addr, T *src_addrs[INPUT_NUM], LocalTensor<uint8_t> &tmp_buf,
                                         const ConcatTiling<INPUT_NUM> &tiling) {
  const uint32_t kScatterMaxLen = VECTOR_REG_WIDTH / sizeof(T) / 2;
  auto seq_tensor = tmp_buf[512].ReinterpretCast<U>();
  auto index_tensor = tmp_buf.ReinterpretCast<U>();
  auto seq_addr = (__ubuf__ U *)seq_tensor.GetPhyAddr();
  auto index_addr = (__ubuf__ U *)index_tensor.GetPhyAddr();

  GenSequence(seq_addr);
  uint32_t dst_col_offset = 0;
  for (size_t i = 0UL; i < INPUT_NUM; ++i) {
    if (tiling.num_srcs_cols[i] > kScatterMaxLen) {
      Copy((__ubuf__ T *)(uint64_t)dst_addr + dst_col_offset, (__ubuf__ T *)(uint64_t)src_addrs[i], tiling.num_rows,
           tiling.num_srcs_cols[i], tiling.num_dst_cols);
    } else {
      GenScatterIndex(tiling.num_srcs_cols[i], tiling.num_dst_cols, dst_col_offset, seq_addr, index_addr);
      ScatterInput((__ubuf__ T *)(uint64_t)dst_addr, (__ubuf__ T *)(uint64_t)src_addrs[i], index_addr, tiling.num_rows,
                   tiling.num_dst_cols, tiling.num_srcs_cols[i]);
    }
    dst_col_offset += tiling.num_srcs_cols[i];
  }
}

template <typename T, typename U, size_t INPUT_NUM>
__aicore__ inline void ConcatExtendInner(T *dst_addr, T *src_addrs[INPUT_NUM], LocalTensor<uint8_t> &tmp_buf,
                                         const ConcatTilingPadded<INPUT_NUM> &tiling) {
  const uint32_t kScatterMaxLen = VECTOR_REG_WIDTH / sizeof(T) / 2;
  auto seq_tensor = tmp_buf[512].ReinterpretCast<U>();
  auto index_tensor = tmp_buf.ReinterpretCast<U>();
  auto seq_addr = (__ubuf__ U *)seq_tensor.GetPhyAddr();
  auto index_addr = (__ubuf__ U *)index_tensor.GetPhyAddr();

  GenSequence(seq_addr);
  uint32_t dst_col_offset = 0;
  for (size_t i = 0UL; i < INPUT_NUM; ++i) {
    if (tiling.num_srcs_cols[i] == tiling.src_row_strides[i]) {
      if (tiling.num_srcs_cols[i] > kScatterMaxLen) {
        Copy((__ubuf__ T *)(uint64_t)dst_addr + dst_col_offset, (__ubuf__ T *)(uint64_t)src_addrs[i], tiling.num_rows,
             tiling.num_srcs_cols[i], tiling.num_dst_cols);
      } else {
        GenScatterIndex(tiling.num_srcs_cols[i], tiling.num_dst_cols, dst_col_offset, seq_addr, index_addr);
        ScatterInput((__ubuf__ T *)(uint64_t)dst_addr, (__ubuf__ T *)(uint64_t)src_addrs[i], index_addr,
                     tiling.num_rows, tiling.num_dst_cols, tiling.num_srcs_cols[i]);
      }
    } else {
      if (tiling.src_row_strides[i] > kScatterMaxLen) {
        CopyPadded((__ubuf__ T *)(uint64_t)dst_addr + dst_col_offset, (__ubuf__ T *)(uint64_t)src_addrs[i],
                   tiling.num_rows, tiling.src_row_strides[i], tiling.num_dst_cols,
                   tiling.src_second_last_dim_strides[i], tiling.gather_mask_dim_sizes[i]);
      } else {
        GenScatterIndex(tiling.num_srcs_cols[i], tiling.num_dst_cols, dst_col_offset, seq_addr, index_addr);
        ScatterInputPadded((__ubuf__ T *)(uint64_t)dst_addr, (__ubuf__ T *)(uint64_t)src_addrs[i], index_addr,
                           tiling.num_rows, tiling.num_dst_cols, tiling.num_srcs_cols[i],
                           tiling.src_row_strides[i], tiling.src_second_last_dim_strides[i],
                           tiling.gather_mask_dim_sizes[i]);
      }
    }
    dst_col_offset += tiling.num_srcs_cols[i];
  }
}

template<typename T, size_t INPUT_NUM>
__aicore__ inline void ConcatExtend(T *dst_addr,
                                    T *src_addrs[INPUT_NUM],
                                    AscendC::LocalTensor<uint8_t> &tmp_buf,
                                    const ConcatTiling<INPUT_NUM> &tiling) {
  if constexpr (sizeof(T) == sizeof(uint32_t)) {
    ConcatExtendInner<T, uint32_t, INPUT_NUM>(dst_addr, src_addrs, tmp_buf, tiling);
  } else {
    ConcatExtendInner<T, uint16_t, INPUT_NUM>(dst_addr, src_addrs, tmp_buf, tiling);
  }
}

template<typename T, size_t INPUT_NUM>
__aicore__ inline void ConcatExtend(T *dst_addr,
                                    T *src_addrs[INPUT_NUM],
                                    AscendC::LocalTensor<uint8_t> &tmp_buf,
                                    const ConcatTilingPadded<INPUT_NUM> &tiling) {
  if constexpr (sizeof(T) == sizeof(uint32_t)) {
    ConcatExtendInner<T, uint32_t, INPUT_NUM>(dst_addr, src_addrs, tmp_buf, tiling);
  } else {
    ConcatExtendInner<T, uint16_t, INPUT_NUM>(dst_addr, src_addrs, tmp_buf, tiling);
  }
}

template<typename T>
__aicore__ inline void CopyOneAxis(__ubuf__ T *dst_addr, __ubuf__ T *src_addr, uint32_t size) {
  __VEC_SCOPE__ {
    AscendC::MicroAPI::UnalignReg u0;
    AscendC::MicroAPI::RegTensor<T> vd0;
    AscendC::MicroAPI::DataCopy(vd0, src_addr);
    AscendC::MicroAPI::DataCopyUnAlign(dst_addr, vd0, u0, size);
    AscendC::MicroAPI::DataCopyUnAlignPost(dst_addr, u0, 0);
  }
}

template <typename T, size_t INPUT_NUM, size_t... Is>
inline __aicore__ void ConcatOneAxisUnroll(T *dst_addr,
                                           T *src_addrs[INPUT_NUM],
                                           const ConcatTilingOneAxis<INPUT_NUM> &tiling,
                                           std::index_sequence<Is...>) {
  ((CopyOneAxis((__ubuf__ T *)((uint64_t)dst_addr + tiling.dst_col_offsets[Is] * sizeof(T)),
                (__ubuf__ T *)(uint64_t)src_addrs[Is],
                tiling.src_col_sizes[Is])), ...);
}

template <typename T, size_t INPUT_NUM>
inline __aicore__ void ConcatOneAxis(T *dst_addr,
                                     T *src_addrs[INPUT_NUM],
                                     const ConcatTilingOneAxis<INPUT_NUM> &tiling) {
  ConcatOneAxisUnroll(dst_addr, src_addrs, tiling, std::make_index_sequence<INPUT_NUM>{});
}
}  // namespace concat

template<size_t INPUT_NUM>
struct ConcatTilingAllAligned {
  uint32_t dst_col_size;
  uint32_t src_col_sizes[INPUT_NUM];
  uint32_t dst_offsets[INPUT_NUM];
};

template<typename T, uint32_t INPUT_NUM>
inline __aicore__ void ConcatAllAligned(uint32_t num_rows,
                                        const ConcatTilingAllAligned<INPUT_NUM> &tiling,
                                        LocalTensor<T> &dst_tensor,
                                        LocalTensor<T> (&src_tensors)[INPUT_NUM]) {
  constexpr uint32_t kDataBlockSize = 32U;
  constexpr auto align_size = static_cast<uint16_t>(kDataBlockSize / sizeof(T));
#pragma unroll
  for (uint32_t i = 0U; i < INPUT_NUM; ++i) {
    const auto size = tiling.src_col_sizes[i];
    DataCopyParams copy_params{static_cast<uint16_t>(num_rows), static_cast<uint16_t>(size / align_size), 0,
                               static_cast<uint16_t>((tiling.dst_col_size - size) / align_size)};
    DataCopy(dst_tensor[tiling.dst_offsets[i]], src_tensors[i], copy_params);
  }
}

#endif  // __ASCENDC_API_REGBASE_CONCAT_H__
