/**
* Copyright (c) Huawei Technologies Co., Ltd. 2025 All rights reserved.
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
 */

#include <string>
#include <numeric>
#include "common/checker.h"
#include "base/att_const_values.h"
#include "api_perf_register/ascendc_api_perf.h"
#include "api_perf_register/api_perf_factory.h"
namespace att {
namespace {
ge::Status GetLoadCase(const NodeDetail &node_info, Expr &blk, int32_t &use_case) {
  size_t dim_size = node_info.input_dims.size();
  auto iter1 = kBlkEleMap.find(node_info.input_dtype[0]);
  GE_ASSERT_TRUE(iter1 != kBlkEleMap.end());
  Expr blocklen = node_info.input_dims[dim_size - 1UL];
  Expr blkThr = CreateExpr(256);
  if (blocklen.IsConstExpr()) {
    int32_t blklen;
    int32_t blkthreshold = 256U;
    blocklen.GetConstValue(blklen);
    if (blklen > blkthreshold) {  // blocklen大于512B
      use_case = kCaseOne;
    } else {
      use_case = kCaseTwo;
    }
  } else {
    blk = blocklen - blkThr;
    use_case = kCaseDefault;
  }
  GELOGD("input dtype is %s, dim_size[%zu], blocklen[%s], use_case[%d]", node_info.input_dtype[0].c_str(),
         dim_size, blocklen.Str().get(), use_case);
  return ge::SUCCESS;
}

ge::Status LoadPerf(const NodeDetail &node_info, PerfOutputInfo &perf) {
  Expr res_normal;
  Expr res_small_blk;
  Expr blk;
  Expr res_stride;
  int32_t use_case;
  GELOGD("Dma with Load: %s", node_info.ToString().c_str());
  GE_ASSERT_SUCCESS(GetPerf({kMoveGmToUb, node_info.input_dtype[0],node_info.output_dtype[0],
                             node_info.input_dims, CreateExpr(0)}, res_normal));
  GE_ASSERT_SUCCESS(GetPerf({kMoveGmToUb + "SmallBlk", node_info.input_dtype[0], node_info.output_dtype[0],
                             node_info.input_dims, CreateExpr(0)}, res_small_blk));
  GE_ASSERT_SUCCESS(GetPerf({kMoveGmToUb + "Stride", node_info.input_dtype[0], node_info.output_dtype[0],
                             node_info.input_dims, node_info.gm_stride}, res_stride));
  GE_ASSERT_SUCCESS(GetLoadCase(node_info, blk, use_case));
  if (use_case == kCaseOne) {
    perf.pipe_res[PipeType::AIV_MTE2] = res_normal + res_stride;
  } else if (use_case == kCaseTwo) {
    perf.pipe_res[PipeType::AIV_MTE2] = res_small_blk + res_stride;
  } else {
    Expr res = CreateExpr("load_node");
    std::shared_ptr<IfCase> branch_a = std::make_shared<IfCase>(res_normal + res_stride);
    GE_ASSERT_NOTNULL(branch_a);
    std::shared_ptr<IfCase> branch_b = std::make_shared<IfCase>(res_small_blk + res_stride);
    GE_ASSERT_NOTNULL(branch_b);
    // blocklen < 512B时走branch_b；否则走branch_a
    TenaryOp tenary_op = TenaryOp(CondType::K_LT, blk, CreateExpr(0), std::move(branch_b), std::move(branch_a));
    tenary_op.SetVariable(res);
    perf.tenary_ops[res] = tenary_op;
    perf.pipe_res[PipeType::AIV_MTE2] = res;
  }
  return ge::SUCCESS;
}

ge::Status StorePerf(const NodeDetail &node_info, PerfOutputInfo &perf) {
  Expr res_normal;
  int32_t use_case;
  Expr res_stride;
  GELOGD("Dma with Load: %s", node_info.ToString().c_str());
  GE_ASSERT_SUCCESS(GetPerf({kMoveUbToGm, node_info.input_dtype[0],node_info.output_dtype[0],
                             node_info.input_dims, CreateExpr(0)}, res_normal));
  GE_ASSERT_SUCCESS(GetPerf({kMoveUbToGm + "Stride", node_info.input_dtype[0], node_info.output_dtype[0],
                             node_info.input_dims, node_info.gm_stride}, res_stride));
  perf.pipe_res[PipeType::AIV_MTE3] = res_normal + res_stride;
  return ge::SUCCESS;
}

ge::Status NddmaPerf(const NodeDetail &node_info, PerfOutputInfo &perf) {
  Expr res;
  GELOGD("Dma with nddma: %s", node_info.ToString().c_str());
  GE_ASSERT_SUCCESS(GetPerf({kNddma, node_info.input_dtype[0],node_info.output_dtype[0],
                             node_info.input_dims, CreateExpr(0)}, res));
  perf.pipe_res[PipeType::AIV_MTE2] = res;
  return ge::SUCCESS;
}

REGISTER_ASCENDC_EVAL_FUNC_TAG(kLoad, V2, LoadPerf);
REGISTER_ASCENDC_EVAL_FUNC_TAG(kStore, V2, StorePerf);
REGISTER_ASCENDC_EVAL_FUNC_TAG(kNddma, V2, NddmaPerf);
}
}