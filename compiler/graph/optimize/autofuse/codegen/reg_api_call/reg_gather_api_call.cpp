/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "reg_gather_api_call.h"
#include "../api_call/gather/gather_api_call_base.h"

#include <sstream>
#include "attr_utils.h"
#include "ascir_ops.h"
#include "common_utils.h"
#include "common/ge_common/debug/log.h"
#include "graph/ascendc_ir/utils/asc_tensor_utils.h"
#include "common/checker.h"
#include "../api_call/utils/api_call_factory.h"
#include "../api_call/utils/api_call_utils.h"
#include "graph/symbolizer/symbolic_utils.h"
#include "../api_call/gather/gather_api_call.h"

namespace codegen {
using namespace std;
using namespace ge::ops;
using namespace ge::ascir_op;
using namespace ascgen_utils;
using namespace gather_base;
Status GatherRegApiCall::GetGatherCase(const Tensor &x1, const Tensor &x2, const Tensor &y,
                                       std::string &result) const {
  std::stringstream ss;
  const string single_axis = "0";
  const string begin_axis = "1";
  const string end_axis = "2";
  const string mid_axis = "3";
  if (x1.axis.size() == 1 && this->axis == 0) {
    result = single_axis;
    return ge::SUCCESS;
  }
  else if (x1.axis.size() > 1 && this->axis == 0) {
    result = begin_axis;
    return ge::SUCCESS;
  }
  else if (x1.axis.size() > 1 && this->axis == x1.axis.size() - 1) {
    result = end_axis;
    return ge::SUCCESS;
  }
  else if (x1.axis.size() > 1 && this->axis != x1.axis.size() - 1 && this->axis > 0) { // 对中间轴gahter
    result = mid_axis;
    return ge::SUCCESS;
  }
  GELOGE(ge::FAILED, "gather axis(%d) is larger than x1 axis size(%d) or below 0", this->axis, x1.axis_size.size());
  return ge::FAILED;
}

std::string GenerateNonLastAxisGatherSimt(const std::vector<ascir::AxisId> &current_axis,
                                      const std::vector<std::reference_wrapper<const Tensor>> &inputs,
                                      const std::vector<std::reference_wrapper<const Tensor>> &outputs,
                                      int64_t gather_axis, const TPipe &tpipe) {
  stringstream ss;
  const auto &x1 = inputs[0].get();  // param_gm
  const auto &x2 = inputs[1].get();  // indices_gm
  const auto &y = outputs[0].get();  // dst_ub
  if (y.vectorized_axis.size() > 1) {
    auto axis0 = tpipe.tiler.GetAxis(y.vectorized_axis[0]);
    auto axis1 = tpipe.tiler.GetAxis(y.vectorized_axis[1]);
    std::vector<ascir::AxisId> param_outer_axes;
    std::vector<ascir::AxisId> param_inner_axes;
    CollectParamOuterAndInnerAxes(x1.axis, gather_axis, param_outer_axes, param_inner_axes);
    std::string outer_axis_offset = CalGatherOuterAxisOffset(current_axis, param_inner_axes, axis0.id, tpipe);
    ss << "for (" << axis0.AsArg() << " = 0; " << axis0 << " < " << axis0.actual_size << "; " << axis0 << "++) {"
       << std::endl;
    ss << CalGatherOuterAxesIndex(outer_axis_offset, param_outer_axes, x2.axis, tpipe);

    ss << "auto indices_index = " << outer_axis_offset << " % " << CalGatherIndicesAxesSize(x2.axis, tpipe) << ";"
       << std::endl;
    std::string indices_value = x2.Str() + ".GetValue(indices_index)";
    ss << "auto param_offset = " << CalGatherParamOffset(x1.axis, indices_value, gather_axis, axis1, tpipe);

    ss << "DataCopyPadExtend(" << y << "[" << axis0 << " * " << tpipe.tiler.Size(y.vectorized_strides[0]) << "], " << x1
       << "[param_offset], 1, " << axis1.actual_size << ", 0, 0);" << std::endl;
    ss << "}" << std::endl;
  } else {
    auto axis0 = tpipe.tiler.GetAxis(y.vectorized_axis[0]);
    std::vector<ascir::AxisId> param_outer_axes;
    std::vector<ascir::AxisId> param_inner_axes;
    CollectParamOuterAndInnerAxes(x1.axis, gather_axis, param_outer_axes, param_inner_axes);
    std::string outer_axis_offset = CalGatherOuterAxisOffset(current_axis, param_inner_axes, ge::kIdNone, tpipe);
    auto outer_axis = tpipe.tiler.GetAxis(axis0.split_pair_other_id);
    auto gather_dim_size = tpipe.tiler.Size(x1.axis_size[gather_axis]);
    auto inner_size = CalGatherInnerSize(x1.axis, gather_axis, tpipe);
    auto outer_size = CalGatherOuterSize(x1.axis, gather_axis, tpipe);
    auto gather_size = CalGatherSize(x2.axis, gather_axis, tpipe);
    ss << "auto y_index_base =  " << outer_axis_offset << " * " << inner_size << " + " << outer_axis << " * " << tpipe.tiler.Size(axis0.size) << ";"
       << std::endl;
    ss << "GatherSimtNonTailExtend(" << y << "[0], " << x1 << ", " << x2 << ", y_index_base, " << gather_size << ", " << outer_size << ", " << inner_size << ", " << gather_dim_size << ", "<<axis0.actual_size<<");"
       << std::endl;
  }
  return ss.str();
}

Status GatherRegApiCall::GenerateComputeTypeGather(const TPipe &tpipe, const std::vector<ascir::AxisId> &current_axis,
                                                   const std::vector<std::reference_wrapper<const Tensor>> &inputs,
                                                   const std::vector<std::reference_wrapper<const Tensor>> &outputs,
                                                   std::string &result) const {
  std::stringstream ss;
  const auto &x1 = inputs[0].get();  // param_gm
  const auto &x2 = inputs[1].get();  // indices_gm
  const auto &y = outputs[0].get();  // dst_ub
  DataCopyParams param_x1;
  DataCopyParams param_x2;
  DataCopyParams param;
  std::string x1_offset = tpipe.tiler.Offset(current_axis, x1.axis, x1.axis_strides);
  std::string dst_offset = tpipe.tiler.Offset(current_axis, y.axis, y.axis_strides);
  size_t pos = dst_offset.rfind('+');
  std::string x2_offset = (pos != std::string::npos) ? dst_offset.substr(pos + 1) : dst_offset;
  x2_offset.erase(0, x2_offset.find_first_not_of(" "));
  if (this->axis + 1 > x1.axis_size.size()) {
    GELOGE(ge::FAILED, "gather axis(%d) is larger than x1 axis size(%d)", this->axis, x1.axis_size.size());
    return ge::FAILED;
  }
  if (this->axis + 1 != x1.axis_size.size()) {
    ss << GenerateNonLastAxisGatherSimt(current_axis, inputs, outputs, this->axis, tpipe);
  } else {
    if (x1.axis_size.size() == 1) {
      ss << this->api_name_ << "(" << y << ", " << x1 << ", " << x2 << "[" << dst_offset << "], "
         << tpipe.tiler.Size(x1.axis_size[0], true) << ", " << y.actual_size << ", " << tpipe.tmp_buf << ");"
         << std::endl;
    } else {
      string first_merge_axis = "0";
      string block_inner_axis;
      for (size_t i = 0; i < current_axis.size(); i++) {
        if (tpipe.tiler.GetAxis(current_axis[i]).type == Axis::Type::kAxisTypeBlockInner) {
          block_inner_axis = tpipe.tiler.GetAxis(current_axis[i]).Str();
        }
      }
      if (block_inner_axis.length() > 1) {
        first_merge_axis = block_inner_axis.substr(0, block_inner_axis.length() - 1);  // 获取合轴之后的首轴未切分的轴
      }
      std::string param_last_axis_size = tpipe.tiler.Size(x1.axis_size[x1.axis_size.size() - 1], true);
      x1_offset = first_merge_axis + " * " + param_last_axis_size;
      ss << this->api_name_ << "(" << y << ", " << x1 << "[" << x1_offset << "], " << x2 << "[" << x2_offset << "], "
         << param_last_axis_size << ", " << y.actual_size << ", " << tpipe.tmp_buf << ");" << std::endl;
    }
  }
  result = ss.str();
  return ge::SUCCESS;
}

Status GatherRegApiCall::GenerateComputeTypeLoad(const TPipe &tpipe, const std::vector<ascir::AxisId> &current_axis,
                                                 const std::vector<std::reference_wrapper<const Tensor>> &inputs,
                                                 const std::vector<std::reference_wrapper<const Tensor>> &outputs,
                                                 std::string &result) const {
  std::stringstream ss;
  const auto &x1 = inputs[0].get();  // param_gm
  const auto &x2 = inputs[1].get();  // indices_gm
  const auto &y = outputs[0].get();  // dst_ub
  string dtypename = "";
  y.DtypeName(y.dtype, dtypename);
  ss << this->api_name_ << "<" << dtypename << ", ";
  x2.DtypeName(x2.dtype, dtypename);
  ss << dtypename << ", ";
  std::string case_;
  if (GetGatherCase(x1, x2, y, case_) == ge::FAILED) {
    GELOGE(ge::FAILED, "gather_dim status need add");
    return ge::FAILED;
  }
  ss << case_ << ", ";
  ss << y.vectorized_axis.size() << ">(";
  std::string offset = tpipe.tiler.Offset(current_axis, y.axis, y.axis_strides);
  ss << y <<  ", " << x1 << ", " << x2 << ", ";
  for (int i = y.vectorized_axis.size() - 1; i >= 0; i--) {
    ss << tpipe.tiler.GetAxis(y.vectorized_axis[i]).actual_size;
    if (i != 0) {
      ss << "*";
    }
  }
  ss << ", " << offset << ", ";
  for (int i = x2.axis_size.size() - 1; i >= 0; i--) {
     ss << tpipe.tiler.Size(x2.axis_size[i]) << " * ";
  }
  ss << "1, ";
  ss << tpipe.tiler.Size(x1.axis_size[this->axis]) << ", " << tpipe.tiler.Size(x1.axis_strides[this->axis]) << ", ";
  ge::Expression param_size = ge::Symbol(1);
  ss << "tmp_buf, " << "t->tmp_tbuf_size, ";
  for (int i=0; i< x1.axis_size.size();i++) {
    ss << tpipe.tiler.Size(x1.axis_size[i]) << " * ";
    param_size = ge::sym::Mul(param_size, x1.axis_size[i]);
  }
  ss << "1, ";
  ss << x1.axis_size.size() << ", ";
  for (int i = y.vectorized_axis.size() - 1; i >= 0; i--) {
    ss << tpipe.tiler.GetAxis(y.vectorized_axis[i]).actual_size<< ", " << tpipe.tiler.Size(y.vectorized_strides[i]) << ", " << tpipe.tiler.Size(y.axis_strides[y.vectorized_axis_pos[i]]);
    if (i != 0) {
      ss << ",";
    }
  }
  ss << ");" << std::endl;
  result = ss.str();
  return ge::SUCCESS;
}

Status GatherRegApiCall::Generate(const TPipe &tpipe, const std::vector<ascir::AxisId> &current_axis,
                               const std::vector<std::reference_wrapper<const Tensor>> &inputs,
                               const std::vector<std::reference_wrapper<const Tensor>> &outputs,
                               std::string &result) const {
  if (this->compute_type == ge::ComputeType::kComputeGather) {
    return GenerateComputeTypeGather(tpipe, current_axis, inputs, outputs, result);
  }
  else if (this->compute_type == ge::ComputeType::kComputeLoad) {
    return GenerateComputeTypeLoad(tpipe, current_axis, inputs, outputs, result);
  }
  GELOGE(ge::FAILED, "gather's compute_type(%d) must be kComputeLoad or kComputeGather", this->compute_type);
  return ge::FAILED;
}

Status GatherRegApiCall::ParseAttr(const ascir::NodeView &node) {
  GE_CHK_GRAPH_STATUS_RET(node->attr.ir_attr->GetAttrValue("axis", this->axis),
                          "Failed to get Gahter axis attr, node = %s", node->GetNamePtr());
  GELOGI("name:%s, axis:%lld", node->GetNamePtr(), this->axis);
  this->compute_type = node->attr.api.compute_type;
  return ge::SUCCESS;
}

static ApiCallRegister<GatherRegApiCall> register_gather_api_call("GatherRegApiCall");

}  // namespace codegen