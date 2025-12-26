/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "where_api_call.h"

#include <sstream>
#include "attr_utils.h"
#include "ascir_ops.h"
#include "common_utils.h"
#include "common/ge_common/debug/log.h"
#include "graph/ascendc_ir/utils//asc_tensor_utils.h"
#include "common/checker.h"
#include "../utils/api_call_factory.h"
#include "../utils/api_call_utils.h"

namespace codegen {
using namespace std;
using namespace ge::ops;
using namespace ge::ascir_op;
using namespace ascgen_utils;

Status WhereApiCall::Generate(const TPipe &tpipe, const std::vector<ascir::AxisId> &current_axis,
  const std::vector<std::reference_wrapper<const Tensor>> &inputs,
  const std::vector<std::reference_wrapper<const Tensor>> &outputs, std::string &result) const {
  size_t x1_idx = 0;
  size_t x2_idx = 1;
  size_t x3_idx = 2;
  auto x1 = inputs[x1_idx].get();
  auto x2 = inputs[x2_idx].get();
  auto x3 = inputs[x3_idx].get();
  auto y = outputs[0].get();

  GELOGD("x2, is_constant:%d, is_ub_scalar:%d, need_gen_get_value_of_ub_scalar:%d",
    static_cast<int32_t>(x2.is_constant),
    static_cast<int32_t>(x2.is_ub_scalar),
    static_cast<int32_t>(x2.need_gen_get_value_of_ub_scalar));
  GELOGD("x3, is_constant:%d, is_ub_scalar:%d, need_gen_get_value_of_ub_scalar:%d",
    static_cast<int32_t>(x3.is_constant),
    static_cast<int32_t>(x3.is_ub_scalar),
    static_cast<int32_t>(x3.need_gen_get_value_of_ub_scalar));

  // todo 暂不支持非标量且需要广播的场景
  ApiLoopParams param;
  VectorizedAxisLoopMergeStatus merge_info;
  std::vector<Tensor> ub_inputs;
  std::vector<Tensor> ub_outputs;
  ub_inputs.push_back(x1);
  if (!x2.is_constant && !x2.need_gen_get_value_of_ub_scalar) {
    ub_inputs.push_back(x2);
  }
  if (!x3.is_constant && !x3.need_gen_get_value_of_ub_scalar) {
    ub_inputs.push_back(x3);
  }
  ub_outputs.push_back(y);
  bool status = GenerateVectorizedAxisMergeStatus(ub_inputs, ub_outputs, merge_info, tpipe);
  GE_ASSERT_TRUE(status, "GenerateVectorizedAxisMergeStatus failed");
  SaveApiLoopAxisParams(merge_info, param);
  stringstream ss;

  const bool x2_is_scalar_scene = x2.IsAnyScalar();
  const bool x3_is_scalar_scene = x3.IsAnyScalar();
  // api层面x2 x3 dtype类型一样
  std::string x2_dtype_name;
  std::string x3_dtype_name;
  GE_CHK_STATUS_RET(Tensor::DtypeName(x2.dtype, x2_dtype_name),
    "Codegen get data type:%d failed", static_cast<int32_t>(x2.dtype));
  GE_CHK_STATUS_RET(Tensor::DtypeName(x3.dtype, x3_dtype_name),
  "Codegen get data type:%d failed", static_cast<int32_t>(x3.dtype));
  GE_ASSERT_TRUE(x2_dtype_name == x3_dtype_name, "x2_dtype_name:%s, x3_dtype_name:%s",
    x2_dtype_name.c_str(), x3_dtype_name.c_str());
  std::string x2_scalar = x2.need_gen_get_value_of_ub_scalar ? ("(" + x2_dtype_name + ")" + x2.ub_scalar_name) : x2.Str();
  std::string x3_scalar = x3.need_gen_get_value_of_ub_scalar ? ("(" + x3_dtype_name + ")" + x3.ub_scalar_name) : x3.Str();
  // 参考cast
  if (param.outer_repeats.size() == 0) {  // 没有loop，调用一维where接口
    ss << this->api_name_ << "(" << y << "[" << tpipe.tiler.TensorVectorizedOffset(current_axis, y) << "], "  // 输出
       << x1 << "[" << tpipe.tiler.TensorVectorizedOffset(current_axis, x1) << "], ";                    // 输入1
    if (x2_is_scalar_scene) {
      ss << x2_scalar << ", ";  // 输入2
    } else {
      ss << x2 << "[" << tpipe.tiler.TensorVectorizedOffset(current_axis, x2) << "], ";  // 输入2
    }
    if (x3_is_scalar_scene) {
      ss << x3_scalar << ", ";  // 输入3
    } else {
      ss << x3 << "[" << tpipe.tiler.TensorVectorizedOffset(current_axis, x3) << "], ";  // 输入3
    }
    ss << x1.actual_size << ", " << tpipe.tmp_buf << ");" << std::endl;
  } else {  // 有loop，调用两根轴输入的where接口
    std::stringstream ss1;
    if (x2_is_scalar_scene && x3_is_scalar_scene) {
      std::string scalar_local_blk_tensor_name_x2 = x2.IsConstScalar() ? "local_blk_tensor_of_" + x2.name : x2.name;
      std::string scalar_local_blk_tensor_name_x3 = x3.IsConstScalar() ? "local_blk_tensor_of_" + x3.name : x3.name;
      scalar_local_blk_tensor_name_x2 = x2.need_alloc_local_blk_tensor_from_tbuf ? scalar_local_blk_tensor_name_x2 + "_1" : scalar_local_blk_tensor_name_x2;
      scalar_local_blk_tensor_name_x3 = x3.need_alloc_local_blk_tensor_from_tbuf ? scalar_local_blk_tensor_name_x3 + "_1" : scalar_local_blk_tensor_name_x3;
     // 计算 output stride
      size_t output_strides_size = param.outputs_strides[0].size();
      std::vector<ascir::SizeExpr> inner_output_strides(param.outputs_strides[0].begin(),
                                                        param.outputs_strides[0].begin() + output_strides_size - 1);
      std::string output_inner_offset = output_strides_size == 1 ? "0" : CalcInnerOffset(tpipe, inner_output_strides);
      // 计算 input stride
      uint32_t index = 0U;
      size_t input0_strides_size = param.inputs_strides[index].size();
      std::vector<ascir::SizeExpr> inner0_input_strides(param.inputs_strides[index].begin(),
                                                      param.inputs_strides[index].begin() + input0_strides_size - 1);
      std::string input0_inner_offset = input0_strides_size == 1 ? "0" : CalcInnerOffset(tpipe, inner0_input_strides);
      ss1 << this->api_name_ << "<true, true>(" << y << "[" << output_inner_offset << "], " << x1 << "[" << input0_inner_offset << "], "
          << scalar_local_blk_tensor_name_x2 << "[0], "
          << scalar_local_blk_tensor_name_x3 << "[0], "
          << param.outer_repeats[param.outer_repeats.size() - 1] << ", " << tpipe.tiler.ActualSize(param.cal_count) << ", "
          << tpipe.tiler.Size(param.output_second_to_last_stride) << ", "
          << tpipe.tiler.Size(param.input_second_to_last_stride) << ", "
          << "ONE_BLK_SIZE / sizeof(float), "
          << "ONE_BLK_SIZE / sizeof(float), "
          << tpipe.tmp_buf << ", ONE_BLK_SIZE * 2);" << std::endl;
      if (param.outer_repeats.size() == 1) {
        ss << ss1.str();
      } else {
        std::vector<std::string> inner_outer_repeats(param.outer_repeats.begin(),
                                                     param.outer_repeats.begin() + param.outer_repeats.size() - 1);
        CreateComputeNodeOuterFor(param.outer_repeats, ss1, ss, 0);
      }
    } else if (x2_is_scalar_scene) {
      std::string scalar_local_blk_tensor_name_x2 = x2.IsConstScalar() ? "local_blk_tensor_of_" + x2.name : x2.name;
      scalar_local_blk_tensor_name_x2 = x2.need_alloc_local_blk_tensor_from_tbuf ? scalar_local_blk_tensor_name_x2 + "_1" : scalar_local_blk_tensor_name_x2;
      // 计算 output stride
      size_t output_strides_size = param.outputs_strides[0].size();
      std::vector<ascir::SizeExpr> inner_output_strides(param.outputs_strides[0].begin(),
                                                        param.outputs_strides[0].begin() + output_strides_size - 1);
      std::string output_inner_offset = output_strides_size == 1 ? "0" : CalcInnerOffset(tpipe, inner_output_strides);
      // 计算 input stride
      uint32_t index = 0U;
      size_t input0_strides_size = param.inputs_strides[index].size();
      std::vector<ascir::SizeExpr> inner0_input_strides(param.inputs_strides[index].begin(),
                                                      param.inputs_strides[index].begin() + input0_strides_size - 1);
      std::string input0_inner_offset = input0_strides_size == 1 ? "0" : CalcInnerOffset(tpipe, inner0_input_strides);

      index++;
      size_t input2_strides_size = param.inputs_strides[index].size();
      std::vector<ascir::SizeExpr> inner2_input_strides(param.inputs_strides[index].begin(),
                                                      param.inputs_strides[index].begin() + input2_strides_size - 1);
      std::string input2_inner_offset = input2_strides_size == 1 ? "0" : CalcInnerOffset(tpipe, inner2_input_strides);

      ss1 << this->api_name_ << "<true, false>(" << y << "[" << output_inner_offset << "], " << x1 << "[" << input0_inner_offset << "], "
          << scalar_local_blk_tensor_name_x2 << "[0], "
          << x3 << "[" << input2_inner_offset << "], "  // 输入3
          << param.outer_repeats[param.outer_repeats.size() - 1] << ", " << tpipe.tiler.ActualSize(param.cal_count) << ", "
          << tpipe.tiler.Size(param.output_second_to_last_stride) << ", "
          << tpipe.tiler.Size(param.input_second_to_last_stride) << ", "
          << "ONE_BLK_SIZE / sizeof(float), "
          << tpipe.tiler.Size(param.output_second_to_last_stride) << ", "
          << tpipe.tmp_buf << ", ONE_BLK_SIZE);" << std::endl;
      if (param.outer_repeats.size() == 1) {
        ss << ss1.str();
      } else {
        std::vector<std::string> inner_outer_repeats(param.outer_repeats.begin(),
                                                     param.outer_repeats.begin() + param.outer_repeats.size() - 1);
        CreateComputeNodeOuterFor(param.outer_repeats, ss1, ss, 0);
      }
    } else if (x3_is_scalar_scene) {
      std::string scalar_local_blk_tensor_name_x3 = x3.IsConstScalar() ? "local_blk_tensor_of_" + x3.name : x3.name;
      scalar_local_blk_tensor_name_x3 = x3.need_alloc_local_blk_tensor_from_tbuf ? scalar_local_blk_tensor_name_x3 + "_1" : scalar_local_blk_tensor_name_x3;
      // 计算 output stride
      size_t output_strides_size = param.outputs_strides[0].size();
      std::vector<ascir::SizeExpr> inner_output_strides(param.outputs_strides[0].begin(),
                                                        param.outputs_strides[0].begin() + output_strides_size - 1);
      std::string output_inner_offset = output_strides_size == 1 ? "0" : CalcInnerOffset(tpipe, inner_output_strides);
      // 计算 input stride
      uint32_t index = 0U;
      size_t input0_strides_size = param.inputs_strides[index].size();
      std::vector<ascir::SizeExpr> inner0_input_strides(param.inputs_strides[index].begin(),
                                                      param.inputs_strides[index].begin() + input0_strides_size - 1);
      std::string input0_inner_offset = input0_strides_size == 1 ? "0" : CalcInnerOffset(tpipe, inner0_input_strides);

      index++;
      size_t input1_strides_size = param.inputs_strides[index].size();
      std::vector<ascir::SizeExpr> inner1_input_strides(param.inputs_strides[index].begin(),
                                                      param.inputs_strides[index].begin() + input1_strides_size - 1);
      std::string input1_inner_offset = input1_strides_size == 1 ? "0" : CalcInnerOffset(tpipe, inner1_input_strides);

      ss1 << this->api_name_ << "<false, true>(" << y << "[" << output_inner_offset << "], " << x1 << "[" << input0_inner_offset << "], "
          << x2 << "[" << input1_inner_offset << "], "  // 输入2
          << scalar_local_blk_tensor_name_x3 << "[0], "
          << param.outer_repeats[param.outer_repeats.size() - 1] << ", " << tpipe.tiler.ActualSize(param.cal_count) << ", "
          << tpipe.tiler.Size(param.output_second_to_last_stride) << ", "
          << tpipe.tiler.Size(param.input_second_to_last_stride) << ", "
          << tpipe.tiler.Size(param.output_second_to_last_stride) << ", "
          << "ONE_BLK_SIZE / sizeof(float), "
          << tpipe.tmp_buf << ", ONE_BLK_SIZE);" << std::endl;
      if (param.outer_repeats.size() == 1) {
        ss << ss1.str();
      } else {
        std::vector<std::string> inner_outer_repeats(param.outer_repeats.begin(),
                                                     param.outer_repeats.begin() + param.outer_repeats.size() - 1);
        CreateComputeNodeOuterFor(param.outer_repeats, ss1, ss, 0);
      }
    } else {// 参考cast/compare
      // 计算 output stride
      size_t output_strides_size = param.outputs_strides[0].size();
      std::vector<ascir::SizeExpr> inner_output_strides(param.outputs_strides[0].begin(),
                                                        param.outputs_strides[0].begin() + output_strides_size - 1);
      std::string output_inner_offset = output_strides_size == 1 ? "0" : CalcInnerOffset(tpipe, inner_output_strides);
      // 计算 input stride
      uint32_t index = 0U;
      size_t input0_strides_size = param.inputs_strides[index].size();
      std::vector<ascir::SizeExpr> inner0_input_strides(param.inputs_strides[index].begin(),
                                                      param.inputs_strides[index].begin() + input0_strides_size - 1);
      std::string input0_inner_offset = input0_strides_size == 1 ? "0" : CalcInnerOffset(tpipe, inner0_input_strides);

      index++;
      size_t input1_strides_size = param.inputs_strides[index].size();
      std::vector<ascir::SizeExpr> inner1_input_strides(param.inputs_strides[index].begin(),
                                                      param.inputs_strides[index].begin() + input1_strides_size - 1);
      std::string input1_inner_offset = input1_strides_size == 1 ? "0" : CalcInnerOffset(tpipe, inner1_input_strides);

      index++;
      size_t input2_strides_size = param.inputs_strides[index].size();
      std::vector<ascir::SizeExpr> inner2_input_strides(param.inputs_strides[index].begin(),
                                                      param.inputs_strides[index].begin() + input2_strides_size - 1);
      std::string input2_inner_offset = input2_strides_size == 1 ? "0" : CalcInnerOffset(tpipe, inner2_input_strides);

      ss1 << this->api_name_ << "<false, false>(" << y << "[" << output_inner_offset << "], " << x1 << "[" << input0_inner_offset << "], "
          << x2 << "[" << input1_inner_offset << "], "  // 输入2
          << x3 << "[" << input2_inner_offset << "], "  // 输入3
          << param.outer_repeats[param.outer_repeats.size() - 1] << ", " << tpipe.tiler.ActualSize(param.cal_count) << ", "
          << tpipe.tiler.Size(param.output_second_to_last_stride) << ", "
          << tpipe.tiler.Size(param.input_second_to_last_stride) << ", "
          << tpipe.tiler.Size(param.output_second_to_last_stride) << ", "
          << tpipe.tiler.Size(param.output_second_to_last_stride) << ", "
          << tpipe.tmp_buf << ", 0);" << std::endl;
      if (param.outer_repeats.size() == 1) {
        ss << ss1.str();
      } else {
        std::vector<std::string> inner_outer_repeats(param.outer_repeats.begin(),
                                                     param.outer_repeats.begin() + param.outer_repeats.size() - 1);
        CreateComputeNodeOuterFor(param.outer_repeats, ss1, ss, 0);
      }
    }
  }

  result = ss.str();
  return ge::SUCCESS;
}

static ApiCallRegister<WhereApiCall> register_where_api_call("WhereApiCall");
}
