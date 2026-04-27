/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "reg_load_api_call.h"

#include <sstream>

#include "ascir_ops.h"
#include "common/ge_common/debug/log.h"
#include "graph/ascendc_ir/utils/asc_tensor_utils.h"
#include "common/checker.h"
#include "reg_api_call_utils.h"
#include "api_call/utils/api_call_factory.h"
#include "codegen_api_param/codegen_api_param.h"

using namespace ge::ops;
using namespace ge::ascir_op;

namespace {
constexpr size_t kDmaMaxLen = 2U;
constexpr size_t kFourAxisNum = 4U;
}
namespace codegen {
Status LoadRegApiCall::ParseAttr(const ascir::NodeView &node) {
  (void)node->attr.ir_attr->GetAttrValue("offset", offset_);
  return ge::SUCCESS;
}

void BuildApiParamInCVFusion(CodegenApiParamPtr api_param, DmaSpecificParams &dma_specific_params, const Tensor &gm,
                             const Tensor &ub, std::string &dtype_name) {
  api_param->template_params.emplace_back("AscendC::PaddingMode::Normal");
  api_param->input_params.emplace_back(gm.Str(), true, "offset");
  api_param->output_params.emplace_back(ub.Str(), true, "0");
  dma_specific_params.data_copy_params.block_count = "curAivM";
  dma_specific_params.data_copy_params.block_len = "load_block_len";
  dma_specific_params.data_copy_params.src_stride = "load_src_stride";
  dma_specific_params.data_copy_params.dst_stride = "load_dst_stride";

  int dtype_size = GetSizeByDataType(gm.dtype);
  if (dtype_size == 1 || dtype_size == 2 || dtype_size == 4) {
    // LoadAlign仅支持字节大小为1、2、4的数据类型，否则GatherMask编译错误。
    // 超过4字节的数据类型，CV融合场景下目前一定是对齐拷入的，不需要RemovePad。
    std::stringstream ss;
    ss << "if (KernelUtils::BlkAlign<" << dtype_name << ">(curAlignN) != curAlignN) {" << std::endl;
    ss << "event_t eventID = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));" << std::endl;
    ss << "SetFlag<HardEvent::MTE2_V>(eventID);" << std::endl;
    ss << "WaitFlag<HardEvent::MTE2_V>(eventID);" << std::endl;
    ss << "uint8_t mask = 7;" << std::endl;
    ss << "uint64_t rsvdCnt = 0;" << std::endl;
    ss << "AscendC::GatherMask(" << ub << ", " << ub << ", mask, true, static_cast<uint32_t>(curAlignN)"
       << ", {1, static_cast<uint16_t>(curAivM), static_cast<uint16_t>(KernelUtils::BlkAlign<" << dtype_name << ">(curAlignN) * sizeof(" << dtype_name << ") / ONE_BLK_SIZE), 0}"
       << ", rsvdCnt);" << std::endl;
    ss << "}";
    api_param->api_post_process.emplace_back(ss.str());
  }
}

void SetDataCopyParams(const TPipe &tpipe, DataCopyParams& data_copy_param, DmaSpecificParams& dma_specific_params) {
  DmaParams dma_param;
  SetDmaParams(tpipe, data_copy_param, dma_param, true);
  dma_specific_params.data_copy_params.block_count = dma_param.block_count;
  dma_specific_params.data_copy_params.block_len = dma_param.block_len;
  dma_specific_params.data_copy_params.src_stride = dma_param.src_stride;
  dma_specific_params.data_copy_params.dst_stride = dma_param.dst_stride;
}

void SetLoopModeParams(const TPipe &tpipe, DataCopyParams& data_copy_param, DmaSpecificParams& dma_specific_params,
                       int64_t input_dtype_size, int64_t output_dtype_size) {
  LoopModeParams loop_mode_param;
  SetLoopModeParams(tpipe, data_copy_param, loop_mode_param, true);
  dma_specific_params.loop_mode_params.loop_sizes.emplace_back("static_cast<uint32_t>(" +
    loop_mode_param.loop_size[0] + ")");
  dma_specific_params.loop_mode_params.loop_sizes.emplace_back("static_cast<uint32_t>(" +
    loop_mode_param.loop_size[1] + ")");
  dma_specific_params.loop_mode_params.loop_src_strides.emplace_back("static_cast<uint64_t>(" +
    loop_mode_param.loop_src_stride[0] + " * " + std::to_string(input_dtype_size) + ")");
  dma_specific_params.loop_mode_params.loop_src_strides.emplace_back("static_cast<uint64_t>(" +
    loop_mode_param.loop_src_stride[1] + " * " + std::to_string(input_dtype_size) + ")");
  dma_specific_params.loop_mode_params.loop_dst_strides.emplace_back("static_cast<uint64_t>(" +
    loop_mode_param.loop_dst_stride[0] + " * " + std::to_string(output_dtype_size) + ")");
  dma_specific_params.loop_mode_params.loop_dst_strides.emplace_back("static_cast<uint64_t>(" +
    loop_mode_param.loop_dst_stride[1] + " * " + std::to_string(output_dtype_size) + ")");
}

Status BuildApiParamInNormal(const TPipe &tpipe, CodegenApiParamPtr api_param, DmaSpecificParams &dma_specific_params,
                             const Tensor &gm, const Tensor &ub, std::string &gm_offset) {
  DataCopyParams data_copy_param;
  GE_ASSERT_TRUE(CalculateDmaParams(tpipe, ub, ub, data_copy_param), "CalculateDmaParams failed");
  size_t total_len = data_copy_param.repeats.size();
  std::string padding_mode = GetPaddingMode(tpipe, ub, data_copy_param);
  api_param->template_params.emplace_back(padding_mode);
  std::string ub_offset = "0";

  SetDataCopyParams(tpipe, data_copy_param, dma_specific_params);
  if (total_len > kDmaMaxLen) {
    SetLoopModeParams(tpipe, data_copy_param, dma_specific_params, ge::GetSizeByDataType(gm.dtype),
                      ge::GetSizeByDataType(ub.dtype));
  }

  if (total_len > kFourAxisNum) {
    // 超过四层for循环，需要外抛
    std::vector<ascir::SizeExpr> gm_stride(data_copy_param.gm_strides.begin(),
                                           data_copy_param.gm_strides.end() - kFourAxisNum);
    std::vector<ascir::SizeExpr> ub_stride(data_copy_param.ub_strides.begin(),
                                           data_copy_param.ub_strides.end() - kFourAxisNum);
    std::vector<ascir::SizeExpr> repeats(data_copy_param.repeats.begin(),
                                         data_copy_param.repeats.end() - kFourAxisNum);
    std::string gm_inner_offset = CalcInnerOffset(tpipe, gm_stride);
    std::string ub_inner_offset = CalcInnerOffset(tpipe, ub_stride);
    gm_offset = gm_offset + " + " + gm_inner_offset;
    ub_offset = ub_inner_offset;
    for (const auto& repeat : repeats) {
      api_param->outer_loop_axes.emplace_back(tpipe.tiler.ActualSize(repeat));
    }
  }
  api_param->input_params.emplace_back(gm.Str(), true, gm_offset);
  api_param->output_params.emplace_back(ub.Str(), true, ub_offset);
  return ge::SUCCESS;
}

Status LoadRegApiCall::BuildApiParam(const TPipe &tpipe, const std::vector<ascir::AxisId> &current_axis,
                                     const std::vector<std::reference_wrapper<const Tensor>> &inputs,
                                     const std::vector<std::reference_wrapper<const Tensor>> &outputs) const {
  const auto &gm = inputs[0].get();
  const auto &ub = outputs[0].get();
  auto api_param = ge::ComGraphMakeShared<CodegenApiParam>();
  GE_ASSERT_NOTNULL(api_param);
  api_param->api_name = api_name_;
  std::string dtype_name;
  GE_CHK_STATUS_RET(Tensor::DtypeName(gm.dtype, dtype_name), "data type:%d failed", static_cast<int32_t>(gm.dtype));
  api_param->template_params.emplace_back(dtype_name);
  DmaSpecificParams dma_specific_params;
  if (tpipe.cv_fusion_type == ascir::CubeTemplateType::kUBFuse && !ub.is_ub_scalar) {
    BuildApiParamInCVFusion(api_param, dma_specific_params, gm, ub, dtype_name);
  } else {
    std::string gm_offset = ub.is_ub_scalar ? "0" : tpipe.tiler.Offset(current_axis, ub.axis, ub.axis_strides);
    gm_offset = gm_offset + " + " + tpipe.tiler.Size(offset_);
    BuildApiParamInNormal(tpipe, api_param, dma_specific_params, gm, ub, gm_offset);
  }
  api_param->specific_params = dma_specific_params;
  GE_CHK_STATUS_RET(CodegenApiParam::Register(this->node, api_param));
  return ge::SUCCESS;
}

Status LoadRegApiCall::GenDimensionParam(const CodegenApiParamPtr api_param, std::stringstream &ss) const {
  auto* dma_params = std::get_if<DmaSpecificParams>(&api_param->specific_params);
  GE_ASSERT_NOTNULL(dma_params, "dma_params is null, graph name: %s, node name: %s", graph_name.c_str(),
                    node_name.c_str());
  ss << dma_params->data_copy_params.block_count << ", ";
  ss << dma_params->data_copy_params.block_len << ", ";
  ss << dma_params->data_copy_params.src_stride << ", ";
  ss << dma_params->data_copy_params.dst_stride;
  if (dma_params->loop_mode_params.loop_sizes.size() > 0) {
    ss << ", " << "{" << dma_params->loop_mode_params.loop_sizes[0] << ", "
       << dma_params->loop_mode_params.loop_sizes[1] << ", "
       << dma_params->loop_mode_params.loop_src_strides[0] << ", "
       << dma_params->loop_mode_params.loop_dst_strides[0] << ", "
       << dma_params->loop_mode_params.loop_src_strides[1] << ", "
       << dma_params->loop_mode_params.loop_dst_strides[1] << "}";
  }
  ss << ");" << std::endl;
  return ge::SUCCESS;
}
static ApiCallRegister<LoadRegApiCall> register_load_reg_api_call("LoadRegApiCall");
}  // namespace codegen