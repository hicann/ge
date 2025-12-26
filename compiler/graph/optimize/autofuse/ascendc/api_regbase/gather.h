/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef __ASCENDC_API_REGBASE_GATHER_H__
#define __ASCENDC_API_REGBASE_GATHER_H__

constexpr int32_t CASE1 = 0;  // 只有一根轴
constexpr int32_t CASE2 = 1;  // 对首轴gather
constexpr int32_t CASE3 = 2;  // 对尾轴gather
constexpr int32_t CASE4 = 3;   // 对中间轴gather
#define THREAD_NUMBER 2048
// 生成单个向量轴的参数声明
// 首先定义每个变量的宏
#define VECTORIZED_AXIS_SIZE_M(n)       vectorized_axis_##n##_size_m
#define VECTORIZED_AXIS_SIZE_SHIFT(n)   vectorized_axis_##n##_size_shift
#define VECTORIZED_AXIS_SIZE(n)         vectorized_axis_##n##_size
#define VECTORIZED_AXIS_STRIDE(n)       vectorized_axis_##n##_stride
#define Y_VECTORIZED_AXIS_SIZE_STRIDE(n) y_vectorized_axis_##n##_size_stride

#define DECLARE_VECTORIZED_AXIS_PARAMS_SIMD(n) \
  uint32_t VECTORIZED_AXIS_SIZE(n) = 0, \
  uint32_t VECTORIZED_AXIS_STRIDE(n) = 0, \
  uint32_t Y_VECTORIZED_AXIS_SIZE_STRIDE(n) = 0

// 宏定义：用于生成单个向量轴的参数声明
#define DECLARE_VECTORIZED_AXIS_PARAMS_SIMT(n) \
  uint32_t VECTORIZED_AXIS_SIZE_M(n), \
  uint32_t VECTORIZED_AXIS_SIZE_SHIFT(n), \
  uint32_t VECTORIZED_AXIS_SIZE(n), \
  uint32_t VECTORIZED_AXIS_STRIDE(n), \
  uint32_t Y_VECTORIZED_AXIS_SIZE_STRIDE(n)

// 宏定义：用于传递单个向量轴的参数
#define PASS_VECTORIZED_AXIS_PARAMS_SIMT(n) \
  VECTORIZED_AXIS_SIZE_M(n), \
  VECTORIZED_AXIS_SIZE_SHIFT(n), \
  VECTORIZED_AXIS_SIZE(n), \
  VECTORIZED_AXIS_STRIDE(n), \
  Y_VECTORIZED_AXIS_SIZE_STRIDE(n)

#define PASS_VECTORIZED_AXIS_PARAMS_SIMD(n) \
  VECTORIZED_AXIS_SIZE(n), \
  VECTORIZED_AXIS_STRIDE(n), \
  Y_VECTORIZED_AXIS_SIZE_STRIDE(n)

// 宏定义：声明单个向量轴的 magic 和 shift 变量
#define DECLARE_VECTORIZED_AXIS_MAGIC_SHIFT_VARIABLES(n) \
  uint32_t VECTORIZED_AXIS_SIZE_M(n) {0}; \
  uint32_t VECTORIZED_AXIS_SIZE_SHIFT(n) {0}

#define DECLARE_VECTORIZED_AXIS_MAGIC_SHIFT_PARAMS(n) \
  uint32_t &VECTORIZED_AXIS_SIZE_M(n), \
  uint32_t &VECTORIZED_AXIS_SIZE_SHIFT(n), \
  uint32_t VECTORIZED_AXIS_SIZE(n)

#define PASS_VECTORIZED_AXIS_MAGIC_SHIFT_PARAMS(n) \
  VECTORIZED_AXIS_SIZE_M(n), VECTORIZED_AXIS_SIZE_SHIFT(n), VECTORIZED_AXIS_SIZE(n)

template <typename T, AscendC::PaddingMode mode = AscendC::PaddingMode::Normal>
inline __aicore__ void GatherDataCopyPadExtend(const AscendC::LocalTensor<T> &dst, const AscendC::GlobalTensor<T> &src,
                                         uint32_t block_count, uint32_t block_len, uint32_t src_stride,
                                         uint32_t dst_stride) {
  uint32_t align_num = AscendC::ONE_BLK_SIZE / sizeof(T);
  AscendC::DataCopyExtParams param;
  param.blockCount = block_count;
  param.blockLen = block_len * sizeof(T);
  param.srcStride = src_stride * sizeof(T);
  param.dstStride = dst_stride / align_num;
  AscendC::DataCopyPadExtParams<T> pad_params = {true, 0, 0, 0};
  AscendC::DataCopyPad<T, mode>(dst, src, param, pad_params);
}

/**************************************************************************************** 模板1 通用SIMT模板 *********************************************************************/
template <typename INDEX_SIZE_T, uint32_t N, uint32_t VECTORIZED_AXIS_SIZE>
inline __aicore__ void ComputeDstpAndYOffset(const INDEX_SIZE_T vector_axis_size_m, const INDEX_SIZE_T vector_axis_size_shift, const INDEX_SIZE_T vector_axis_size,
                                             const INDEX_SIZE_T vector_axis_stride, const INDEX_SIZE_T y_vector_axis_size_stride,
                                             uint32_t &dst_p, INDEX_SIZE_T &y_offset, INDEX_SIZE_T &v_offset) {
  if constexpr (VECTORIZED_AXIS_SIZE >= N) {
    auto tmp = Simt::UintDiv(v_offset, vector_axis_size_m, vector_axis_size_shift);
    auto tmp1 = (v_offset - tmp * vector_axis_size);
    y_offset += tmp1 * y_vector_axis_size_stride;
    dst_p += tmp1 * vector_axis_stride;
    v_offset = tmp;
  }
}

template <typename INDEX_SIZE_T, uint32_t VECTORIZED_AXIS_SIZE>
inline __aicore__ void GetDstpAndYOffset(INDEX_SIZE_T v_offset, uint32_t &dst_p, INDEX_SIZE_T &y_offset,
                                                      DECLARE_VECTORIZED_AXIS_PARAMS_SIMT(1), DECLARE_VECTORIZED_AXIS_PARAMS_SIMT(2), DECLARE_VECTORIZED_AXIS_PARAMS_SIMT(3),
                                                      DECLARE_VECTORIZED_AXIS_PARAMS_SIMT(4), DECLARE_VECTORIZED_AXIS_PARAMS_SIMT(5), DECLARE_VECTORIZED_AXIS_PARAMS_SIMT(6),
                                                      DECLARE_VECTORIZED_AXIS_PARAMS_SIMT(7), DECLARE_VECTORIZED_AXIS_PARAMS_SIMT(8)) {
  ComputeDstpAndYOffset<INDEX_SIZE_T, 1, VECTORIZED_AXIS_SIZE>(PASS_VECTORIZED_AXIS_PARAMS_SIMT(1), dst_p, y_offset, v_offset);
  ComputeDstpAndYOffset<INDEX_SIZE_T, 2, VECTORIZED_AXIS_SIZE>(PASS_VECTORIZED_AXIS_PARAMS_SIMT(2), dst_p, y_offset, v_offset);
  ComputeDstpAndYOffset<INDEX_SIZE_T, 3, VECTORIZED_AXIS_SIZE>(PASS_VECTORIZED_AXIS_PARAMS_SIMT(3), dst_p, y_offset, v_offset);
  ComputeDstpAndYOffset<INDEX_SIZE_T, 4, VECTORIZED_AXIS_SIZE>(PASS_VECTORIZED_AXIS_PARAMS_SIMT(4), dst_p, y_offset, v_offset);
  ComputeDstpAndYOffset<INDEX_SIZE_T, 5, VECTORIZED_AXIS_SIZE>(PASS_VECTORIZED_AXIS_PARAMS_SIMT(5), dst_p, y_offset, v_offset);
  ComputeDstpAndYOffset<INDEX_SIZE_T, 6, VECTORIZED_AXIS_SIZE>(PASS_VECTORIZED_AXIS_PARAMS_SIMT(6), dst_p, y_offset, v_offset);
  ComputeDstpAndYOffset<INDEX_SIZE_T, 7, VECTORIZED_AXIS_SIZE>(PASS_VECTORIZED_AXIS_PARAMS_SIMT(7), dst_p, y_offset, v_offset);
  ComputeDstpAndYOffset<INDEX_SIZE_T, 8, VECTORIZED_AXIS_SIZE>(PASS_VECTORIZED_AXIS_PARAMS_SIMT(8), dst_p, y_offset, v_offset);
}

template <typename T1, typename T2, typename INDEX_SIZE_T, int32_t CASE, int32_t VECTORIZED_AXIS_SIZE>
__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_NUMBER) inline void GatherSimt(__ubuf__ T1 *dst, __gm__ T1 *x1_gm, __gm__ T2 *x2_gm,
                                                      uint32_t ub_actual_size, INDEX_SIZE_T offset,
                                                      INDEX_SIZE_T x1_gather_dim_size,
                                                      INDEX_SIZE_T x2_tensor_size_m, INDEX_SIZE_T x2_tensor_size_shift, INDEX_SIZE_T x2_tensor_size,
                                                      INDEX_SIZE_T x1_gather_dim_stride_m, INDEX_SIZE_T x1_gather_dim_stride_shift, INDEX_SIZE_T x1_gather_dim_stride,
                                                      DECLARE_VECTORIZED_AXIS_PARAMS_SIMT(1), DECLARE_VECTORIZED_AXIS_PARAMS_SIMT(2), DECLARE_VECTORIZED_AXIS_PARAMS_SIMT(3),
                                                      DECLARE_VECTORIZED_AXIS_PARAMS_SIMT(4), DECLARE_VECTORIZED_AXIS_PARAMS_SIMT(5), DECLARE_VECTORIZED_AXIS_PARAMS_SIMT(6),
                                                      DECLARE_VECTORIZED_AXIS_PARAMS_SIMT(7), DECLARE_VECTORIZED_AXIS_PARAMS_SIMT(8)){
    for (INDEX_SIZE_T i = static_cast<INDEX_SIZE_T>(Simt::GetThreadIdx()); i < ub_actual_size; i += static_cast<INDEX_SIZE_T>(Simt::GetThreadNum<0>())) {
      auto y_offset = offset;
      uint32_t dst_p = 0;
      GetDstpAndYOffset<INDEX_SIZE_T, VECTORIZED_AXIS_SIZE>(i, dst_p, y_offset, PASS_VECTORIZED_AXIS_PARAMS_SIMT(1), PASS_VECTORIZED_AXIS_PARAMS_SIMT(2),
                                                                                PASS_VECTORIZED_AXIS_PARAMS_SIMT(3), PASS_VECTORIZED_AXIS_PARAMS_SIMT(4),
                                                                                PASS_VECTORIZED_AXIS_PARAMS_SIMT(5), PASS_VECTORIZED_AXIS_PARAMS_SIMT(6),
                                                                                PASS_VECTORIZED_AXIS_PARAMS_SIMT(7), PASS_VECTORIZED_AXIS_PARAMS_SIMT(8));
      if constexpr (CASE == CASE1) {
        T2 param_offset = x2_gm[y_offset];
        if (unlikely(param_offset < 0)) {
          param_offset += x1_gather_dim_size;
        }
        dst[dst_p] = param_offset >= 0 ? x1_gm[param_offset] : 0;
      }
      if constexpr (CASE == CASE2) {
        INDEX_SIZE_T index_idx = Simt::UintDiv(y_offset, x1_gather_dim_stride_m, x1_gather_dim_stride_shift);
        T2 index_value = x2_gm[index_idx];
        if (unlikely(index_value < 0)) {
          index_value += x1_gather_dim_size;
        }
        auto param_offset = index_value * x1_gather_dim_stride + (y_offset - index_idx * x1_gather_dim_stride);
        dst[dst_p] = index_value >= 0 ? x1_gm[param_offset] : 0;
      }
      if constexpr (CASE == CASE3) {
        INDEX_SIZE_T tmp = Simt::UintDiv(y_offset, x2_tensor_size_m, x2_tensor_size_shift);
        INDEX_SIZE_T index_idx = y_offset - tmp * x2_tensor_size;
        T2 index_value = x2_gm[index_idx];
        if (unlikely(index_value < 0)) {
          index_value += x1_gather_dim_size;
        }
        auto param_offset = tmp * x1_gather_dim_size + index_value;
        dst[dst_p] = index_value >= 0 ? x1_gm[param_offset] : 0;
      }
      if constexpr (CASE == CASE4) {
        INDEX_SIZE_T tmp = Simt::UintDiv(y_offset, x1_gather_dim_stride_m, x1_gather_dim_stride_shift);
        INDEX_SIZE_T tmp1 = Simt::UintDiv(tmp, x2_tensor_size_m, x2_tensor_size_shift);
        INDEX_SIZE_T index_idx = tmp - tmp1 * x2_tensor_size;
        T2 index_value = x2_gm[index_idx];
        if (unlikely(index_value < 0)) {
          index_value += x1_gather_dim_size;
        }
        auto param_offset =  tmp1 * x1_gather_dim_size * x1_gather_dim_stride + index_value * \
							 x1_gather_dim_stride + (y_offset - tmp * x1_gather_dim_stride);
        dst[dst_p] = index_value >= 0 ? x1_gm[param_offset] : 0;
      }
    }
}

template <uint32_t N, uint32_t VECTORIZED_AXIS_SIZE>
inline __aicore__ void ComputeMagicShiftForVectiorizedSize(uint32_t &m, uint32_t &s, const uint32_t number) {
  if constexpr (VECTORIZED_AXIS_SIZE >= N) {
    AscendC::GetUintDivMagicAndShift(m, s, number);
  }
}

template <uint32_t VECTORIZED_AXIS_SIZE>
inline __aicore__ void GetMagicShiftForVectiorizedSize( DECLARE_VECTORIZED_AXIS_MAGIC_SHIFT_PARAMS(1), DECLARE_VECTORIZED_AXIS_MAGIC_SHIFT_PARAMS(2), DECLARE_VECTORIZED_AXIS_MAGIC_SHIFT_PARAMS(3),
                                                        DECLARE_VECTORIZED_AXIS_MAGIC_SHIFT_PARAMS(4), DECLARE_VECTORIZED_AXIS_MAGIC_SHIFT_PARAMS(5), DECLARE_VECTORIZED_AXIS_MAGIC_SHIFT_PARAMS(6),
                                                        DECLARE_VECTORIZED_AXIS_MAGIC_SHIFT_PARAMS(7), DECLARE_VECTORIZED_AXIS_MAGIC_SHIFT_PARAMS(8)) {
    ComputeMagicShiftForVectiorizedSize<1, VECTORIZED_AXIS_SIZE>(PASS_VECTORIZED_AXIS_MAGIC_SHIFT_PARAMS(1));
    ComputeMagicShiftForVectiorizedSize<2, VECTORIZED_AXIS_SIZE>(PASS_VECTORIZED_AXIS_MAGIC_SHIFT_PARAMS(2));
    ComputeMagicShiftForVectiorizedSize<3, VECTORIZED_AXIS_SIZE>(PASS_VECTORIZED_AXIS_MAGIC_SHIFT_PARAMS(3));
    ComputeMagicShiftForVectiorizedSize<4, VECTORIZED_AXIS_SIZE>(PASS_VECTORIZED_AXIS_MAGIC_SHIFT_PARAMS(4));
    ComputeMagicShiftForVectiorizedSize<5, VECTORIZED_AXIS_SIZE>(PASS_VECTORIZED_AXIS_MAGIC_SHIFT_PARAMS(5));
    ComputeMagicShiftForVectiorizedSize<6, VECTORIZED_AXIS_SIZE>(PASS_VECTORIZED_AXIS_MAGIC_SHIFT_PARAMS(6));
    ComputeMagicShiftForVectiorizedSize<7, VECTORIZED_AXIS_SIZE>(PASS_VECTORIZED_AXIS_MAGIC_SHIFT_PARAMS(7));
    ComputeMagicShiftForVectiorizedSize<8, VECTORIZED_AXIS_SIZE>(PASS_VECTORIZED_AXIS_MAGIC_SHIFT_PARAMS(8));
}

template <typename T1, typename T2, int32_t CASE, int32_t VECTORIZED_AXIS_SIZE>
inline __aicore__ void GatherExtendDefault(AscendC::LocalTensor<T1> &dst, const AscendC::GlobalTensor<T1> &src1, const AscendC::GlobalTensor<T2> &src2,
                              uint32_t ub_actual_size,uint64_t offset,
                              uint32_t x2_tensor_size, uint32_t x1_gather_dim_size, uint32_t x1_gather_dim_stride,
                              DECLARE_VECTORIZED_AXIS_PARAMS_SIMD(1), DECLARE_VECTORIZED_AXIS_PARAMS_SIMD(2),
                              DECLARE_VECTORIZED_AXIS_PARAMS_SIMD(3), DECLARE_VECTORIZED_AXIS_PARAMS_SIMD(4),
                              DECLARE_VECTORIZED_AXIS_PARAMS_SIMD(5), DECLARE_VECTORIZED_AXIS_PARAMS_SIMD(6),
                              DECLARE_VECTORIZED_AXIS_PARAMS_SIMD(7), DECLARE_VECTORIZED_AXIS_PARAMS_SIMD(8)) {

  int32_t event_id_mte2_to_v = static_cast<int32_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE2_V));
  int32_t event_id_v_to_mte2 = static_cast<int32_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::V_MTE2));
  __gm__ T1 *x1_gm = (__gm__ T1*)(src1.GetPhyAddr());
  __gm__ T2 *x2_gm = (__gm__ T2*)(src2.GetPhyAddr());
  __ubuf__ T1 *dst_p = (__ubuf__ T1*)(dst.GetPhyAddr());
  uint32_t x1_gather_dim_stride_m {0}, x1_gather_dim_stride_shift {0}, x2_tensor_size_m {0}, x2_tensor_size_shift {0};
  if constexpr (CASE == CASE2) {
      AscendC::GetUintDivMagicAndShift(x1_gather_dim_stride_m, x1_gather_dim_stride_shift, x1_gather_dim_stride);
  }
  if constexpr (CASE == CASE3) {
      AscendC::GetUintDivMagicAndShift(x2_tensor_size_m, x2_tensor_size_shift, x2_tensor_size);
  }
  if constexpr (CASE == CASE4) {
      AscendC::GetUintDivMagicAndShift(x1_gather_dim_stride_m, x1_gather_dim_stride_shift, x1_gather_dim_stride);
      AscendC::GetUintDivMagicAndShift(x2_tensor_size_m, x2_tensor_size_shift, x2_tensor_size);
  }
  DECLARE_VECTORIZED_AXIS_MAGIC_SHIFT_VARIABLES(1); DECLARE_VECTORIZED_AXIS_MAGIC_SHIFT_VARIABLES(2); DECLARE_VECTORIZED_AXIS_MAGIC_SHIFT_VARIABLES(3);
  DECLARE_VECTORIZED_AXIS_MAGIC_SHIFT_VARIABLES(4); DECLARE_VECTORIZED_AXIS_MAGIC_SHIFT_VARIABLES(5); DECLARE_VECTORIZED_AXIS_MAGIC_SHIFT_VARIABLES(6);
  DECLARE_VECTORIZED_AXIS_MAGIC_SHIFT_VARIABLES(7); DECLARE_VECTORIZED_AXIS_MAGIC_SHIFT_VARIABLES(8);
  GetMagicShiftForVectiorizedSize<VECTORIZED_AXIS_SIZE>(PASS_VECTORIZED_AXIS_MAGIC_SHIFT_PARAMS(1), PASS_VECTORIZED_AXIS_MAGIC_SHIFT_PARAMS(2), PASS_VECTORIZED_AXIS_MAGIC_SHIFT_PARAMS(3),
                                                        PASS_VECTORIZED_AXIS_MAGIC_SHIFT_PARAMS(4), PASS_VECTORIZED_AXIS_MAGIC_SHIFT_PARAMS(5), PASS_VECTORIZED_AXIS_MAGIC_SHIFT_PARAMS(6),
                                                        PASS_VECTORIZED_AXIS_MAGIC_SHIFT_PARAMS(7), PASS_VECTORIZED_AXIS_MAGIC_SHIFT_PARAMS(8));
  int32_t event_id_v_to_mte3 = static_cast<int32_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::V_MTE3));
  AscendC::Simt::VF_CALL<GatherSimt<T1, T2, uint32_t, CASE, VECTORIZED_AXIS_SIZE>>(AscendC::Simt::Dim3(THREAD_NUMBER), dst_p, x1_gm, x2_gm, ub_actual_size,
                static_cast<uint32_t>(offset), static_cast<uint32_t>(x1_gather_dim_size),
                static_cast<uint32_t>(x2_tensor_size_m), static_cast<uint32_t>(x2_tensor_size_shift), static_cast<uint32_t>(x2_tensor_size),
                static_cast<uint32_t>(x1_gather_dim_stride_m), static_cast<uint32_t>(x1_gather_dim_stride_shift), static_cast<uint32_t>(x1_gather_dim_stride),
                PASS_VECTORIZED_AXIS_PARAMS_SIMT(1), PASS_VECTORIZED_AXIS_PARAMS_SIMT(2),
                PASS_VECTORIZED_AXIS_PARAMS_SIMT(3), PASS_VECTORIZED_AXIS_PARAMS_SIMT(4),
                PASS_VECTORIZED_AXIS_PARAMS_SIMT(5), PASS_VECTORIZED_AXIS_PARAMS_SIMT(6),
                PASS_VECTORIZED_AXIS_PARAMS_SIMT(7), PASS_VECTORIZED_AXIS_PARAMS_SIMT(8));
  AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(event_id_v_to_mte3);
  AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(event_id_v_to_mte3);
}

/***********************************************************************************************************************************************************************************/

/**************************************************************************************** 模板2 长连续数据搬运模板 *********************************************************************/
template <typename T1>
__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_NUMBER) inline void GatherSimtContinuous(__ubuf__ T1 *dst, __gm__ T1 *x1_gm, uint32_t ub_actual_size, uint32_t dst_p, uint64_t param_offset){
    for (uint32_t i = Simt::GetThreadIdx(); i < ub_actual_size; i += Simt::GetThreadNum<0>()) {
      dst[i + dst_p] = x1_gm[i + param_offset];
    }
}

template <typename T1, typename T2, int32_t CASE, int32_t VECTORIZED_AXIS_SIZE>
inline __aicore__ void GatherExtendDataCopy(AscendC::LocalTensor<T1> &dst, const AscendC::GlobalTensor<T1> &src1, const AscendC::GlobalTensor<T2> &src2,
                              uint32_t ub_actual_size,uint64_t offset,
                              uint64_t x2_tensor_size, uint64_t x1_gather_dim_size, uint64_t x1_gather_dim_stride,
                              DECLARE_VECTORIZED_AXIS_PARAMS_SIMD(1), DECLARE_VECTORIZED_AXIS_PARAMS_SIMD(2),
                              DECLARE_VECTORIZED_AXIS_PARAMS_SIMD(3), DECLARE_VECTORIZED_AXIS_PARAMS_SIMD(4),
                              DECLARE_VECTORIZED_AXIS_PARAMS_SIMD(5), DECLARE_VECTORIZED_AXIS_PARAMS_SIMD(6),
                              DECLARE_VECTORIZED_AXIS_PARAMS_SIMD(7), DECLARE_VECTORIZED_AXIS_PARAMS_SIMD(8)) {
      uint64_t index_idx_from = (offset / x1_gather_dim_stride) % x2_tensor_size;
      uint64_t index_idx_to = ((offset+ub_actual_size - 1) / x1_gather_dim_stride) % x2_tensor_size;
      uint64_t pre_gather_idx_from =  (offset / x1_gather_dim_stride) / x2_tensor_size;
      uint64_t pre_gather_idx_to =  ((offset+ub_actual_size - 1) / x1_gather_dim_stride) / x2_tensor_size;
      if(pre_gather_idx_from == pre_gather_idx_to && index_idx_from < index_idx_to){
        __gm__ T1 *x1_gm = (__gm__ T1*)(src1.GetPhyAddr());
        __ubuf__ T1 *dst_p1 = (__ubuf__ T1*)(dst.GetPhyAddr());
        uint64_t back_gather_idx_first = offset % x1_gather_dim_stride;
        uint64_t back_gather_idx_last = (offset + ub_actual_size - 1) % x1_gather_dim_stride;
        uint64_t dst_p = 0, padding = 0, current_addr = 0, aligned_addr = 0;
        int32_t event_id_v_to_mte2 = static_cast<int32_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::V_MTE2));
        int32_t event_id_mte2_to_v = static_cast<int32_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE2_V));
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(event_id_v_to_mte2);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(event_id_v_to_mte2);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(event_id_mte2_to_v);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(event_id_mte2_to_v);
        T2 index_value = src2.GetValue(index_idx_from);
        uint64_t param_offset = pre_gather_idx_from * x1_gather_dim_size * x1_gather_dim_stride + index_value * x1_gather_dim_stride + back_gather_idx_first;
        GatherDataCopyPadExtend(dst[dst_p], src1[param_offset], 1, x1_gather_dim_stride - back_gather_idx_first , 1, 1);
        dst_p += (x1_gather_dim_stride - back_gather_idx_first);
        for (int i = index_idx_from + 1; i < index_idx_to; i++) {
          index_value = src2.GetValue(i);
          if ((dst_p * sizeof(T1)) % 32 != 0) { //判断目标地址是否32B对齐
            current_addr = dst_p * sizeof(T1);
            padding = (32 - (current_addr % 32)) % 32;
            if (padding / sizeof(T1) >= x1_gather_dim_stride) {
              param_offset = pre_gather_idx_from * x1_gather_dim_size * x1_gather_dim_stride + index_value * x1_gather_dim_stride;
              AscendC::Simt::VF_CALL<GatherSimtContinuous<T1>>(AscendC::Simt::Dim3(128), dst_p1, x1_gm, x1_gather_dim_stride, dst_p, param_offset);
              dst_p += x1_gather_dim_stride;
            }
            else {
              aligned_addr = current_addr + padding;
              param_offset = pre_gather_idx_from * x1_gather_dim_size * x1_gather_dim_stride + index_value * x1_gather_dim_stride;
              AscendC::Simt::VF_CALL<GatherSimtContinuous<T1>>(AscendC::Simt::Dim3(128), dst_p1, x1_gm, padding / sizeof(T1), dst_p, param_offset);
              dst_p = aligned_addr / sizeof(T1);
              param_offset += padding / sizeof(T1);
              GatherDataCopyPadExtend(dst[dst_p], src1[param_offset], 1, x1_gather_dim_stride - padding / sizeof(T1), 1, 1);
              dst_p += x1_gather_dim_stride - padding / sizeof(T1);
            }
          }
          else {
            param_offset = pre_gather_idx_from * x1_gather_dim_size * x1_gather_dim_stride + index_value * x1_gather_dim_stride;
            GatherDataCopyPadExtend(dst[dst_p], src1[param_offset], 1, x1_gather_dim_stride, 1, 1);
            dst_p += x1_gather_dim_stride;
          }
        }
        padding = 0;
        index_value = src2.GetValue(index_idx_to);
        if ((dst_p * sizeof(T1)) % 32 != 0) { //判断目标地址是否32B对齐
            current_addr = dst_p * sizeof(T1);
            padding = (32 - (current_addr % 32)) % 32;
            if (padding / sizeof(T1) >= back_gather_idx_last + 1) {
              param_offset = pre_gather_idx_from * x1_gather_dim_size * x1_gather_dim_stride + index_value * x1_gather_dim_stride;
              AscendC::Simt::VF_CALL<GatherSimtContinuous<T1>>(AscendC::Simt::Dim3(128), dst_p1, x1_gm, back_gather_idx_last + 1, dst_p, param_offset);
            }
            else {
              aligned_addr = current_addr + padding;
              param_offset = pre_gather_idx_from * x1_gather_dim_size * x1_gather_dim_stride + index_value * x1_gather_dim_stride;
              AscendC::Simt::VF_CALL<GatherSimtContinuous<T1>>(AscendC::Simt::Dim3(128), dst_p1, x1_gm, padding / sizeof(T1), dst_p, param_offset);
              dst_p = aligned_addr / sizeof(T1);
              param_offset += padding / sizeof(T1);
              GatherDataCopyPadExtend(dst[dst_p], src1[param_offset], 1, back_gather_idx_last + 1 - padding / sizeof(T1), 1, 1);
            }
        }
        else {
            param_offset = pre_gather_idx_from * x1_gather_dim_size * x1_gather_dim_stride + index_value * x1_gather_dim_stride;
            GatherDataCopyPadExtend(dst[dst_p], src1[param_offset], 1, x1_gather_dim_stride, 1, 1);
        }
      }
      else if (pre_gather_idx_from == pre_gather_idx_to && index_idx_from == index_idx_to) {
        AscendC::PipeBarrier<PIPE_MTE2>();
        T2 index_value = src2.GetValue(index_idx_from);
        uint64_t back_gather_idx_first = offset % x1_gather_dim_stride;
        uint64_t param_offset = pre_gather_idx_from * x1_gather_dim_size * x1_gather_dim_stride + index_value * x1_gather_dim_stride +back_gather_idx_first;
        GatherDataCopyPadExtend(dst, src1[param_offset], 1, x1_gather_dim_stride - back_gather_idx_first , 1, 1);
      }
      else {
        GatherExtendDefault<T1, T2, CASE, VECTORIZED_AXIS_SIZE>(dst, src1, src2, ub_actual_size, offset, x2_tensor_size, x1_gather_dim_size, x1_gather_dim_stride,
                                                      PASS_VECTORIZED_AXIS_PARAMS_SIMD(1), PASS_VECTORIZED_AXIS_PARAMS_SIMD(2), PASS_VECTORIZED_AXIS_PARAMS_SIMD(3),
                                                      PASS_VECTORIZED_AXIS_PARAMS_SIMD(4), PASS_VECTORIZED_AXIS_PARAMS_SIMD(5), PASS_VECTORIZED_AXIS_PARAMS_SIMD(6),
                                                      PASS_VECTORIZED_AXIS_PARAMS_SIMD(7), PASS_VECTORIZED_AXIS_PARAMS_SIMD(8));
      }
}

/***********************************************************************************************************************************************************************************/

/**************************************************************************************** 模板3 微指令处理单轴场景 *********************************************************************/
template <typename T1, typename T2>
__aicore__ inline void ConvertNegIndices(const LocalTensor<int32_t> &indicesLocal, int32_t num, int32_t dimSize) {
        __local_mem__ int32_t *indiceAddr = (__local_mem__ int32_t *)indicesLocal.GetPhyAddr();
        constexpr int16_t vfLen = AscendC::VECTOR_REG_WIDTH / sizeof(int32_t);
        uint16_t vfLoopNum = (num + vfLen - 1) / vfLen;
        __VEC_SCOPE__
        {
            MicroAPI::RegTensor<int32_t> indice;
            MicroAPI::RegTensor<int32_t> dst;
            MicroAPI::MaskReg ltPreg;
            uint32_t size = num;
            __local_mem__ int32_t *curIndiceAddr = indiceAddr;
            for (uint16_t i = 0; i < vfLoopNum; i++) {
                MicroAPI::MaskReg preg = AscendC::MicroAPI::UpdateMask<int32_t>(size);
                MicroAPI::DataCopy(indice, curIndiceAddr);
                MicroAPI::CompareScalar<int32_t, CMPMODE::LT>(ltPreg, indice, 0, preg);
                MicroAPI::Adds(dst, indice, dimSize, ltPreg);
                MicroAPI::Copy<int32_t, MicroAPI::MaskMergeMode::MERGING>(indice, dst, ltPreg);
                MicroAPI::DataCopy(curIndiceAddr, indice, preg);
                curIndiceAddr += vfLen;
            }
      }
}

template <typename TARGET_T, typename ORG_T>
inline __aicore__ void LoadIndicesORG64(MicroAPI::RegTensor<TARGET_T> &vregIndcie, MicroAPI::MaskReg &gatherMask, __local_mem__ ORG_T *indiceAddr, MicroAPI::MaskReg preg, ORG_T dimsize)
{
    if constexpr (sizeof(TARGET_T) == sizeof(int32_t)) {
        MicroAPI::RegTensor<int64_t, MicroAPI::RegTraitNumTwo> tmpReg;
        MicroAPI::DataCopy(tmpReg, indiceAddr);
        MicroAPI::MaskReg gtPreg;
        MicroAPI::MaskReg ltPreg;
        MicroAPI::CompareScalar<int64_t, CMPMODE::GE>(gtPreg, tmpReg, 0, preg);
        MicroAPI::CompareScalar<int64_t, CMPMODE::LT>(ltPreg, tmpReg, dimsize, preg);
        MicroAPI::MaskAnd(gatherMask, gtPreg, ltPreg, preg);
        MicroAPI::Pack((MicroAPI::RegTensor<uint32_t>&)vregIndcie, tmpReg);
    } else {
        constexpr int16_t vfLen = AscendC::VECTOR_REG_WIDTH / sizeof(int32_t);
        MicroAPI::RegTensor<int64_t, MicroAPI::RegTraitNumTwo> tmpReg0, tmpReg1;
        MicroAPI::RegTensor<int32_t> tmpB32Reg0, tmpB32Reg1;
        MicroAPI::MaskReg lowPreg, highPreg;
        MicroAPI::MaskInterleave<int16_t>(lowPreg, highPreg, preg, preg);
        MicroAPI::DataCopy(tmpReg0, indiceAddr);
        MicroAPI::DataCopy(tmpReg1, indiceAddr + vfLen);
        MicroAPI::Pack((MicroAPI::RegTensor<uint32_t>&)tmpB32Reg0, tmpReg0);
        MicroAPI::Pack((MicroAPI::RegTensor<uint32_t>&)tmpB32Reg1, tmpReg1);
        MicroAPI::DeInterleave<int16_t>((MicroAPI::RegTensor<int16_t>&)vregIndcie, (MicroAPI::RegTensor<int16_t>&)tmpB32Reg0, (MicroAPI::RegTensor<int16_t>&)tmpB32Reg0, (MicroAPI::RegTensor<int16_t>&)tmpB32Reg1);
        MicroAPI::MaskReg gtPreg, ltPreg;
        MicroAPI::CompareScalar<int64_t, CMPMODE::GE>(gtPreg, tmpReg0, 0, lowPreg);
        MicroAPI::CompareScalar<int64_t, CMPMODE::LT>(ltPreg, tmpReg0, dimsize, lowPreg);
        MicroAPI::MaskAnd(lowPreg, gtPreg, ltPreg, lowPreg);
        MicroAPI::CompareScalar<int64_t, CMPMODE::GE>(gtPreg, tmpReg1, 0, highPreg);
        MicroAPI::CompareScalar<int64_t, CMPMODE::LT>(ltPreg, tmpReg1, dimsize, highPreg);
        MicroAPI::MaskAnd(highPreg, gtPreg, ltPreg, highPreg);
        MicroAPI::MaskDeInterleave<int16_t>(gatherMask, ltPreg, lowPreg, highPreg);
    }
}
template <typename TARGET_T, typename ORG_T>
inline __aicore__ void LoadIndicesORG32(MicroAPI::RegTensor<TARGET_T> &vregIndcie, MicroAPI::MaskReg &gatherMask, __local_mem__ ORG_T *indiceAddr, MicroAPI::MaskReg preg, ORG_T dimsize)
{
    if constexpr (sizeof(TARGET_T) == sizeof(int32_t)) {
        MicroAPI::DataCopy((MicroAPI::RegTensor<int32_t>&)vregIndcie, indiceAddr);
        MicroAPI::MaskReg gtPreg;
        MicroAPI::MaskReg ltPreg;
        MicroAPI::CompareScalar<int32_t, CMPMODE::GE>(gtPreg, (MicroAPI::RegTensor<int32_t>&)vregIndcie, 0, preg);
        MicroAPI::CompareScalar<int32_t, CMPMODE::LT>(ltPreg,(MicroAPI::RegTensor<int32_t>&) vregIndcie, dimsize, preg);
        MicroAPI::MaskAnd(gatherMask, gtPreg, ltPreg, preg);
  } else {
        constexpr int16_t vfLen = AscendC::VECTOR_REG_WIDTH / sizeof(int32_t);
        MicroAPI::RegTensor<int32_t> tmpReg0, tmpReg1;
        MicroAPI::RegTensor<int32_t> tmpB32;
        MicroAPI::MaskReg lowPreg, highPreg;
        MicroAPI::MaskInterleave<int16_t>(lowPreg, highPreg, preg, preg);
        MicroAPI::DataCopy(tmpReg0, indiceAddr);
        MicroAPI::DataCopy(tmpReg1, indiceAddr + vfLen);

        MicroAPI::DeInterleave<int16_t>((MicroAPI::RegTensor<int16_t>&)vregIndcie, (MicroAPI::RegTensor<int16_t>&)tmpB32, (MicroAPI::RegTensor<int16_t>&)tmpReg0, (MicroAPI::RegTensor<int16_t>&)tmpReg1);
        MicroAPI::MaskReg gtPreg, ltPreg;
        MicroAPI::CompareScalar<int32_t, CMPMODE::GE>(gtPreg, tmpReg0, 0, lowPreg);
        MicroAPI::CompareScalar<int32_t, CMPMODE::LT>(ltPreg, tmpReg0, dimsize, lowPreg);
        MicroAPI::MaskAnd(lowPreg, gtPreg, ltPreg, lowPreg);
        MicroAPI::CompareScalar<int32_t, CMPMODE::GE>(gtPreg, tmpReg1, 0, highPreg);
        MicroAPI::CompareScalar<int32_t, CMPMODE::LT>(ltPreg, tmpReg1, dimsize, highPreg);
        MicroAPI::MaskAnd(highPreg, gtPreg, ltPreg, highPreg);
        MicroAPI::MaskDeInterleave<int16_t>(gatherMask, ltPreg, lowPreg, highPreg);
  }
}

template <typename TARGET_T, typename ORG_T>
inline __aicore__ void LoadIndices(MicroAPI::RegTensor<TARGET_T> &vregIndcie, MicroAPI::MaskReg &gatherMask, __local_mem__ ORG_T *indiceAddr, MicroAPI::MaskReg preg, ORG_T dimsize)
{
    if constexpr (sizeof(ORG_T) == sizeof(int64_t)) {
      LoadIndicesORG64(vregIndcie, gatherMask, indiceAddr, preg, dimsize);
    } else {
      LoadIndicesORG32(vregIndcie, gatherMask, indiceAddr, preg, dimsize);
    }
}

template <typename T>
struct IndexTypeGet {
    using type = typename std::conditional<sizeof(T) == sizeof(int8_t) || sizeof(T) == sizeof(int16_t), uint16_t, uint32_t>::type;
};


template<typename T1, typename T2>
inline __aicore__ void VRegGather(__local_mem__ T1 *yAddr, __local_mem__ T1 *xAddr, __local_mem__ int32_t *indiceAddr, uint32_t num_per_loop, int32_t gather_dim_size, int16_t vfLen, uint16_t vfLoopNum)
{
    using indiceType = typename IndexTypeGet<T1>::type;
    __VEC_SCOPE__
    {
        using RegDstT = typename std::conditional<sizeof(T1) == sizeof(int64_t), AscendC::MicroAPI::RegTensor<T1, AscendC::MicroAPI::RegTraitNumTwo>,
                                                AscendC::MicroAPI::RegTensor<T1>>::type;
        RegDstT vd0;
        AscendC::MicroAPI::RegTensor<indiceType> vregIndcie;
        AscendC::MicroAPI::MaskReg gatherMask;
        uint32_t size = num_per_loop;
        __local_mem__ int32_t *curIndiceAddr = indiceAddr;
        for (uint16_t i = 0; i < vfLoopNum; i++) {
            MicroAPI::MaskReg preg0 = AscendC::MicroAPI::UpdateMask<indiceType>(size);
            LoadIndices<indiceType, int32_t>(vregIndcie, gatherMask, curIndiceAddr, preg0, gather_dim_size);
            curIndiceAddr += vfLen;
            __local_mem__ T1 *curyAddr = yAddr + i * vfLen;
            __local_mem__ T1 *curXaddr = xAddr;
            if constexpr (sizeof(T1) == 1) {
              AscendC::MicroAPI::DataCopyGather((AscendC::MicroAPI::RegTensor<int16_t>&)vd0, curXaddr, vregIndcie, gatherMask);
              AscendC::MicroAPI::DataCopy<T1, AscendC::MicroAPI::StoreDist::DIST_PACK_B16>(
                curyAddr, vd0, preg0);
            } else {
              AscendC::MicroAPI::DataCopyGather(vd0, curXaddr, vregIndcie, gatherMask);
              AscendC::MicroAPI::DataCopy(curyAddr, vd0, preg0);
            }
       }
    }
}
template <typename T1, typename T2>
inline __aicore__ void GatherExtendVReg(int32_t offset, AscendC::LocalTensor<uint8_t> &tmp_buf, uint32_t gather_dim_size, const AscendC::LocalTensor<T1> &yLocal,  const AscendC::GlobalTensor<T1> &src1, const AscendC::GlobalTensor<T2> &src2, int32_t cols) {

    LocalTensor<T1> xLocal;
    xLocal = tmp_buf[0].template ReinterpretCast<T1>();

    LocalTensor<T2> indicesLocal;
    int32_t st_offset = gather_dim_size * sizeof(T1);
    int32_t indice_offset = st_offset + (32 - st_offset % 32);
    indicesLocal = tmp_buf[indice_offset].template ReinterpretCast<T2>();

    LocalTensor<int32_t> tmpLocal;

    DataCopyPadExtParams<T1> param_src_pad;
    param_src_pad.isPad = false;
    param_src_pad.leftPadding = 0;
    param_src_pad.rightPadding = 0;
    param_src_pad.paddingValue = 0;
    AscendC::DataCopyExtParams param_src;
    param_src.blockCount = 1;
    param_src.blockLen = gather_dim_size * sizeof(T1);
    param_src.srcStride = 0;
    param_src.dstStride = 0;
    AscendC::DataCopyPad(xLocal, src1, param_src, param_src_pad);
    int32_t maxOutCols;
    if(cols < 300) {
      maxOutCols = cols;
    }
    else {
      maxOutCols = 8 * (cols / 16);
    }
    int64_t outLoopNum = (cols + maxOutCols - 1) / maxOutCols;
    int32_t tailOutCols = cols - (outLoopNum - 1) * maxOutCols;
    __local_mem__ T1 *yAddr = (__local_mem__ T1 *)yLocal.GetPhyAddr();
    __local_mem__ T1 *xAddr = (__local_mem__ T1 *)xLocal.GetPhyAddr();
    int32_t event_id_mte2_to_v = static_cast<int32_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE2_V));
    int32_t event_id_v_to_mte2 = static_cast<int32_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::V_MTE2));

    for(int64_t j = 0; j < outLoopNum; j++){
      int32_t curOutCols = (j == outLoopNum - 1) ? tailOutCols : maxOutCols;
      AscendC::DataCopyExtParams indice_src;
      indice_src.blockCount = 1;
      indice_src.blockLen = curOutCols * sizeof(T2);
      indice_src.srcStride = 0;
      indice_src.dstStride = 0;
      AscendC::DataCopyPad(indicesLocal[0], src2[j * maxOutCols], indice_src, AscendC::DataCopyPadExtParams<T2>());

      __local_mem__ int32_t *indiceAddr;

      using indiceType = typename IndexTypeGet<T2>::type;

      constexpr static uint32_t VECTOR_LENGTH = AscendC::GetVecLen();
      constexpr static uint32_t SIZE_OF_DTYPE = sizeof(float);
      constexpr static uint32_t ELEMENT_PER_VECTOR_LENGTH = VECTOR_LENGTH / SIZE_OF_DTYPE;
      uint32_t num_per_loop = curOutCols;
      constexpr int16_t vfLen = VECTOR_LENGTH / sizeof(indiceType);
      uint16_t vfLoopNum = (num_per_loop + vfLen - 1) / vfLen;
      uint16_t vfRowsLoop = 1;
      int32_t dimSize = gather_dim_size;
      AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(event_id_mte2_to_v);
      AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(event_id_mte2_to_v);
      if constexpr (sizeof(T2) == sizeof(int64_t)) {
          Cast(tmpLocal, indicesLocal, AscendC::RoundMode::CAST_NONE, curOutCols);
          ConvertNegIndices<T1, T2>(tmpLocal, curOutCols, gather_dim_size);
          indiceAddr = (__local_mem__ int32_t *)tmpLocal[0].GetPhyAddr();
      } else {
          ConvertNegIndices<T1, T2>(indicesLocal, curOutCols, gather_dim_size);
          indiceAddr = (__local_mem__ int32_t *)indicesLocal[0].GetPhyAddr();
      }
      VRegGather<T1, T2>(yAddr, xAddr, indiceAddr, num_per_loop, dimSize, vfLen, vfLoopNum);
      yAddr += num_per_loop;
      AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(event_id_v_to_mte2);
      AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(event_id_v_to_mte2);
    }
}

/***********************************************************************************************************************************************************************************/

template <typename T1, typename T2, int32_t CASE, int32_t VECTORIZED_AXIS_SIZE>
inline __aicore__ void GatherExtend(AscendC::LocalTensor<T1> &dst, const AscendC::GlobalTensor<T1> &src1, const AscendC::GlobalTensor<T2> &src2,
                              uint32_t ub_actual_size,uint64_t offset,
                              uint64_t x2_tensor_size, uint64_t x1_gather_dim_size, uint64_t x1_gather_dim_stride,
                              AscendC::LocalTensor<uint8_t> &tmp_buf, uint32_t tmp_buf_size, uint32_t param_size, uint32_t param_axis_size,
                              DECLARE_VECTORIZED_AXIS_PARAMS_SIMD(1), DECLARE_VECTORIZED_AXIS_PARAMS_SIMD(2),
                              DECLARE_VECTORIZED_AXIS_PARAMS_SIMD(3), DECLARE_VECTORIZED_AXIS_PARAMS_SIMD(4),
                              DECLARE_VECTORIZED_AXIS_PARAMS_SIMD(5), DECLARE_VECTORIZED_AXIS_PARAMS_SIMD(6),
                              DECLARE_VECTORIZED_AXIS_PARAMS_SIMD(7), DECLARE_VECTORIZED_AXIS_PARAMS_SIMD(8)) {
  if(VECTORIZED_AXIS_SIZE == 1 && param_size <= 30000 && param_axis_size == 1 && tmp_buf_size > 8192){
    GatherExtendVReg<T1, T2>(offset, tmp_buf, param_size, dst, src1, src2[offset], ub_actual_size);
  }
  else if(VECTORIZED_AXIS_SIZE == 1 && Y_VECTORIZED_AXIS_SIZE_STRIDE(1) == 1 && x1_gather_dim_stride >= 2048){
    GatherExtendDataCopy<T1, T2, CASE, VECTORIZED_AXIS_SIZE>(dst, src1, src2, ub_actual_size, offset, x2_tensor_size, x1_gather_dim_size, x1_gather_dim_stride,
                                              PASS_VECTORIZED_AXIS_PARAMS_SIMD(1), PASS_VECTORIZED_AXIS_PARAMS_SIMD(2), PASS_VECTORIZED_AXIS_PARAMS_SIMD(3),
                                              PASS_VECTORIZED_AXIS_PARAMS_SIMD(4), PASS_VECTORIZED_AXIS_PARAMS_SIMD(5), PASS_VECTORIZED_AXIS_PARAMS_SIMD(6),
                                              PASS_VECTORIZED_AXIS_PARAMS_SIMD(7), PASS_VECTORIZED_AXIS_PARAMS_SIMD(8));
  } else {
    GatherExtendDefault<T1, T2, CASE, VECTORIZED_AXIS_SIZE>(dst, src1, src2, ub_actual_size, offset, x2_tensor_size, x1_gather_dim_size, x1_gather_dim_stride,
                                              PASS_VECTORIZED_AXIS_PARAMS_SIMD(1), PASS_VECTORIZED_AXIS_PARAMS_SIMD(2), PASS_VECTORIZED_AXIS_PARAMS_SIMD(3),
                                              PASS_VECTORIZED_AXIS_PARAMS_SIMD(4), PASS_VECTORIZED_AXIS_PARAMS_SIMD(5), PASS_VECTORIZED_AXIS_PARAMS_SIMD(6),
                                              PASS_VECTORIZED_AXIS_PARAMS_SIMD(7), PASS_VECTORIZED_AXIS_PARAMS_SIMD(8));
  }
}

#endif  // __ASCENDC_API_GATHER_REGBASE_H__