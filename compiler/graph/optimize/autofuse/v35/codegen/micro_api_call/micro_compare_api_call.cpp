/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "micro_api_call_factory.h"
#include "ascir_ops.h"

#include "micro_compare_api_call.h"

namespace codegen {
Status MicroCompareApiCall::Generate(const codegen::TensorManager &tensor_mng, [[maybe_unused]] const TPipe &tpipe,
                                     CallParam &param, string &result) {
  std::stringstream ss;
  ss << "AscendC::MicroAPI::" << this->api_name_ << (this->second_input_scalar_ ? "s" : "") << "(";
  for (auto out_arg : this->outputs_) {
    ss << *(tensor_mng.GetTensor(out_arg.second)) << ", ";
  }
  ss << *(tensor_mng.GetTensor(this->inputs_[0].second)) << ", ";
  if (inputs_[1].first == TensorType::REG_TENSOR) {
    ss << *tensor_mng.GetTensor(inputs_[1].second) << ", ";
  } else {
    ss << *tpipe.GetTensor(inputs_[1].second) << ", ";
  }
  ss << param.p_reg << ");" << std::endl;
  result = ss.str();
  return ge::SUCCESS;
}

Status MicroCompareApiCall::Init(const ascir::NodeView &node) {
  // 判断第二个输入是否是scalar
  ge::AscNodeInputs node_inputs = node->inputs;
  auto &attr = node_inputs[1].attr;
  for (size_t i = 0; i < attr.vectorized_axis.size(); i++) {
    auto it = std::find(attr.axis.begin(), attr.axis.end(), attr.vectorized_axis[i]);
    GE_ASSERT_TRUE(it != attr.axis.end(), "Incorrect axis ID in vectorized_axis");
    auto axis_id = static_cast<uint64_t>(std::distance(attr.axis.begin(), it));
    if (ge::SymbolicUtils::StaticCheckEq(attr.repeats[axis_id], ge::sym::kSymbolOne) != ge::TriBool::kTrue) {
      this->second_input_scalar_ = false;
    }
  }
  GELOGI("name:%s, second input scalar:%d", node->GetNamePtr(), this->second_input_scalar_);
  return ge::SUCCESS;
}

static MicroApiCallRegister<MicroCompareApiCall> register_micro_compare_api_call("MicroCompareApiCall");
}  // namespace codegen
