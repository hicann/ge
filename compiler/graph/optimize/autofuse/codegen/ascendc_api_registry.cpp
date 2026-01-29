/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "ascendc_api_registry.h"

namespace codegen {
namespace {
class Register {
 public:
  Register();
};

Register::Register() {
  const std::string kAscendcBitwise_andStr = {
#include "bitwise_and_str.h"

  };
  const std::string kAscendcDuplicateStr = {
#include "duplicate_str.h"

  };
  const std::string kAscendcBroadcastStr = {
#include "broadcast_str.h"

  };
  const std::string kAscendcCastStr = {
#include "cast_str.h"

  };
  const std::string kAscendcClipbyvalueStr = {
#include "clipbyvalue_str.h"

  };
  const std::string kAscendcCompareStr = {
#include "compare_str.h"

  };
  const std::string kAscendcCompareV2Str = {
#include "compare_v2_str.h"

  };
  const std::string kAscendcConcatStr = {
#include "concat_str.h"

  };
  const std::string kAscendcDatacopyStr = {
#include "datacopy_str.h"

  };
  const std::string kAscendcIsfiniteStr = {
#include "isfinite_str.h"

  };
  const std::string kAscendcIsnanStr = {
#include "isnan_str.h"

  };
  const std::string kAscendcLogical_notStr = {
#include "logical_not_str.h"

  };
  const std::string kAscendcLogicalStr = {
#include "logical_str.h"

  };
  const std::string kAscendcPowStr = {
#include "pow_str.h"

  };
  const std::string kAscendcAxpyStr = {
#include "axpy_str.h"

  };
  const std::string kAscendcReciprocalStr = {
#include "reciprocal_str.h"

  };
  const std::string kAscendcReduce_initStr = {
#include "reduce_init_str.h"

  };
  const std::string kAscendcRemovePadStr = {
#include "removepad_str.h"

  };
  const std::string kAscendcReduce_prodStr = {
#include "reduce_prod_str.h"

  };
  const std::string kAscendcReduceStr = {
#include "reduce_str.h"

  };
  const std::string kAscendcRsqrtStr = {
#include "rsqrt_str.h"

  };
  const std::string kAscendcScalar_divStr = {
#include "scalar_div_str.h"

  };
  const std::string kAscendcFloorDivStr = {
#include "floor_div_str.h"

  };

  const std::string kAscendcSigmoidStr = {
#include "sigmoid_str.h"

  };
  const std::string kAscendcSignStr = {
#include "sign_str.h"

  };
  const std::string kAscendcWhereStr = {
#include "where_str.h"

  };

  const std::string kAscendcGatherStr = {
#include "gather_str.h"

  };
  const std::string kAscendcSubsStr = {
#include "subs_str.h"

  };
  const std::string kAscendcTranposeBaseTypeStr = {
#include "transpose_base_type_str.h"

  };
  const std::string kAscendcTranposeStr = {
#include "transpose_str.h"

  };
  const std::string kAscendcMatmulStr = {
#include "matmul_str.h"

  };
  const std::string kAscendcmat_mul_v3_tiling_key_public = {
#include "mat_mul_v3_tiling_key_public_str.h"

  };
  const std::string kAscendcmat_mul_tiling_key = {
#include "mat_mul_tiling_key_str.h"

  };
  const std::string kAscendcmat_mul_v3_common = {
#include "mat_mul_v3_common_str.h"

  };
  const std::string kAscendcmat_mul_tiling_data = {
#include "mat_mul_tiling_data_str.h"

  };
  const std::string kAscendcbatch_mat_mul_v3_tiling_key = {
#include "batch_mat_mul_v3_tiling_key_str.h"

  };
  const std::string kAscendcmat_mul_asw_block = {
#include "mat_mul_asw_block_str.h"

  };
  const std::string kAscendcmat_mul_asw_kernel = {
#include "mat_mul_asw_kernel_str.h"

  };
  const std::string kAscendcmat_mul_stream_k_block = {
#include "mat_mul_stream_k_block_str.h"

  };
  const std::string kAscendcmat_mul_stream_k_kernel = {
#include "mat_mul_stream_k_kernel_str.h"

  };
  const std::string kAscendcmat_mul_v3_full_load_kernel_helper = {
#include "mat_mul_v3_full_load_kernel_helper_str.h"

  };
  const std::string kAscendcmat_mul_full_load = {
#include "mat_mul_full_load_str.h"

  };
  const std::string kAscendcmm_copy_cube_out = {
#include "mm_copy_cube_out_str.h"

  };
  const std::string kAscendcmm_custom_mm_policy = {
#include "mm_custom_mm_policy_str.h"

  };
  const std::string kAscendcmat_mul_fixpipe_opti = {
#include "mat_mul_fixpipe_opti_str.h"

  };
  const std::string kAscendcblock_scheduler_aswt = {
#include "block_scheduler_aswt_str.h"

  };
  const std::string kAscendcblock_scheduler_streamk = {
#include "block_scheduler_streamk_str.h"

  };
  const std::string kAscendcmat_mul_fixpipe_opti_basic_cmct = {
#include "mat_mul_fixpipe_opti_basic_cmct_str.h"

  };
  const std::string kAscendcbatch_mat_mul_v3_matmul2mul_block_scheduler = {
#include "batch_mat_mul_v3_matmul2mul_block_scheduler_str.h"

  };
  const std::string kAscendcbatch_mat_mul_v3_mergebatch_basicapi_block_scheduler = {
#include "batch_mat_mul_v3_mergebatch_basicapi_block_scheduler_str.h"

  };
  const std::string kAscendcbatch_mat_mul_v3_mergebatch_basicapi_cmct = {
#include "batch_mat_mul_v3_mergebatch_basicapi_cmct_str.h"

  };
  const std::string kAscendcbatch_mat_mul_v3_matmul2mul_cmct = {
#include "batch_mat_mul_v3_matmul2mul_cmct_str.h"

  };
  const std::string kAscendcmat_mul_pingpong_basic_cmct = {
#include "mat_mul_pingpong_basic_cmct_str.h"

  };
  const std::string kAscendcmat_mul_input_k_eq_zero_clear_output = {
#include "mat_mul_input_k_eq_zero_clear_output_str.h"

  };
  const std::string kAscendcmat_mul_streamk_basic_cmct = {
#include "mat_mul_streamk_basic_cmct_str.h"

  };
  const std::string kAscendcbatch_mat_mul_v3_asw_kernel_advanced = {
#include "batch_mat_mul_v3_asw_kernel_advanced_str.h"

  };
  const std::string kAscendcbatch_mat_mul_v3_asw_block_advanced = {
#include "batch_mat_mul_v3_asw_block_advanced_str.h"

  };
  const std::string kAscendcbatch_mat_mul_v3_asw_al1_full_load_kernel_advanced = {
#include "batch_mat_mul_v3_asw_al1_full_load_kernel_advanced_str.h"

  };
  const std::string kAscendcbatch_mat_mul_v3_asw_bl1_full_load_kernel_advanced = {
#include "batch_mat_mul_v3_asw_bl1_full_load_kernel_advanced_str.h"

  };
  const std::string kAscendcbatch_mat_mul_v3_iterbatch_block_advanced = {
#include "batch_mat_mul_v3_iterbatch_block_advanced_str.h"

  };
  const std::string kAscendcbatch_mat_mul_v3_iterbatch_kernel_advanced = {
#include "batch_mat_mul_v3_iterbatch_kernel_advanced_str.h"

  };
  const std::string kAscendcbatch_mat_mul_v3_iterbatch_basicapi_block_scheduler = {
#include "batch_mat_mul_v3_iterbatch_basicapi_block_scheduler_str.h"

  };
  const std::string kAscendcbatch_mat_mul_v3_iterbatch_basicapi_cmct = {
#include "batch_mat_mul_v3_iterbatch_basicapi_cmct_str.h"

  };
  const std::string kAscendcbatch_matmul = {
#include "batch_matmul_str.h"

  };
  std::unordered_map<std::string, std::string> api_to_file{
      {"bitwise_and.h", kAscendcBitwise_andStr},
      {"duplicate.h", kAscendcDuplicateStr},
      {"broadcast.h", kAscendcBroadcastStr},
      {"cast.h", kAscendcCastStr},
      {"clipbyvalue.h", kAscendcClipbyvalueStr},
      {"compare.h", kAscendcCompareStr},
      {"compare_v2.h", kAscendcCompareV2Str},
      {"concat.h", kAscendcConcatStr},
      {"datacopy.h", kAscendcDatacopyStr},
      {"isfinite.h", kAscendcIsfiniteStr},
      {"isnan.h", kAscendcIsnanStr},
      {"logical_not.h", kAscendcLogical_notStr},
      {"logical.h", kAscendcLogicalStr},
      {"pow.h", kAscendcPowStr},
      {"axpy.h", kAscendcAxpyStr},
      {"reciprocal.h", kAscendcReciprocalStr},
      {"reduce_init.h", kAscendcReduce_initStr},
      {"removepad.h", kAscendcRemovePadStr},
      {"reduce_prod.h", kAscendcReduce_prodStr},
      {"reduce.h", kAscendcReduceStr},
      {"rsqrt.h", kAscendcRsqrtStr},
      {"scalar_div.h", kAscendcScalar_divStr},
      {"floor_div.h", kAscendcFloorDivStr},
      {"sigmoid.h", kAscendcSigmoidStr},
      {"sign.h", kAscendcSignStr},
      {"where.h", kAscendcWhereStr},
      {"gather.h", kAscendcGatherStr},
      {"subs.h", kAscendcSubsStr},
      {"transpose_base_type.h", kAscendcTranposeBaseTypeStr},
      {"transpose.h", kAscendcTranposeStr},
      {"matmul.h", kAscendcMatmulStr},
      {"mat_mul_v3_tiling_key_public.h", kAscendcmat_mul_v3_tiling_key_public},
      {"mat_mul_tiling_key.h", kAscendcmat_mul_tiling_key},
      {"mat_mul_v3_common.h", kAscendcmat_mul_v3_common},
      {"mat_mul_tiling_data.h", kAscendcmat_mul_tiling_data},
      {"batch_mat_mul_v3_tiling_key.h", kAscendcbatch_mat_mul_v3_tiling_key},
      {"mat_mul_asw_block.h", kAscendcmat_mul_asw_block},
      {"mat_mul_asw_kernel.h", kAscendcmat_mul_asw_kernel},
      {"mat_mul_stream_k_block.h", kAscendcmat_mul_stream_k_block},
      {"mat_mul_stream_k_kernel.h", kAscendcmat_mul_stream_k_kernel},
      {"mat_mul_v3_full_load_kernel_helper.h", kAscendcmat_mul_v3_full_load_kernel_helper},
      {"mat_mul_full_load.h", kAscendcmat_mul_full_load},
      {"mm_copy_cube_out.h", kAscendcmm_copy_cube_out},
      {"mm_custom_mm_policy.h", kAscendcmm_custom_mm_policy},
      {"mat_mul_fixpipe_opti.h", kAscendcmat_mul_fixpipe_opti},
      {"block_scheduler_aswt.h", kAscendcblock_scheduler_aswt},
      {"block_scheduler_streamk.h", kAscendcblock_scheduler_streamk},
      {"mat_mul_fixpipe_opti_basic_cmct.h", kAscendcmat_mul_fixpipe_opti_basic_cmct},
      {"batch_mat_mul_v3_matmul2mul_block_scheduler.h", kAscendcbatch_mat_mul_v3_matmul2mul_block_scheduler},
      {"batch_mat_mul_v3_mergebatch_basicapi_block_scheduler.h", kAscendcbatch_mat_mul_v3_mergebatch_basicapi_block_scheduler},
      {"batch_mat_mul_v3_mergebatch_basicapi_cmct.h", kAscendcbatch_mat_mul_v3_mergebatch_basicapi_cmct},
      {"batch_mat_mul_v3_matmul2mul_cmct.h", kAscendcbatch_mat_mul_v3_matmul2mul_cmct},
      {"mat_mul_pingpong_basic_cmct.h", kAscendcmat_mul_pingpong_basic_cmct},
      {"mat_mul_input_k_eq_zero_clear_output.h", kAscendcmat_mul_input_k_eq_zero_clear_output},
      {"mat_mul_streamk_basic_cmct.h", kAscendcmat_mul_streamk_basic_cmct},
      {"batch_mat_mul_v3_asw_kernel_advanced.h", kAscendcbatch_mat_mul_v3_asw_kernel_advanced},
      {"batch_mat_mul_v3_asw_block_advanced.h", kAscendcbatch_mat_mul_v3_asw_block_advanced},
      {"batch_mat_mul_v3_asw_al1_full_load_kernel_advanced.h",
       kAscendcbatch_mat_mul_v3_asw_al1_full_load_kernel_advanced},
      {"batch_mat_mul_v3_asw_bl1_full_load_kernel_advanced.h",
       kAscendcbatch_mat_mul_v3_asw_bl1_full_load_kernel_advanced},
      {"batch_mat_mul_v3_iterbatch_block_advanced.h", kAscendcbatch_mat_mul_v3_iterbatch_block_advanced},
      {"batch_mat_mul_v3_iterbatch_kernel_advanced.h", kAscendcbatch_mat_mul_v3_iterbatch_kernel_advanced},
      {"batch_mat_mul_v3_iterbatch_basicapi_block_scheduler.h",
       kAscendcbatch_mat_mul_v3_iterbatch_basicapi_block_scheduler},
      {"batch_mat_mul_v3_iterbatch_basicapi_cmct.h", kAscendcbatch_mat_mul_v3_iterbatch_basicapi_cmct},
      {"batch_matmul.h", kAscendcbatch_matmul}};

  AscendCApiRegistry::GetInstance().RegisterApi(api_to_file);
}

Register __attribute__((unused)) api_register;
}  // namespace

AscendCApiRegistry &AscendCApiRegistry::GetInstance() {
  static AscendCApiRegistry instance;
  return instance;
}

const std::string &AscendCApiRegistry::GetFileContent(const std::string &api_name) {
  static const std::string kEmpty;
  auto it = api_to_file_content_.find(api_name);
  return it != api_to_file_content_.end() ? it->second : kEmpty;
}

void AscendCApiRegistry::RegisterApi(const std::unordered_map<std::string, std::string> &api_to_file_content) {
  api_to_file_content_.insert(api_to_file_content.cbegin(), api_to_file_content.cend());
}
}  // namespace codegen