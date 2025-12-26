/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph/load/model_manager/data_inputer.h"

#include "framework/common/debug/log.h"
#include "common/model/executor.h"

namespace ge {
/// @ingroup domi_ome
/// @brief init InputData
/// @param [in] input_data use input data to init InputData
/// @param [in] output_data use output data to init OutputData
InputDataWrapper::InputDataWrapper(const InputData &input_data, const OutputData &output_data) {
  input_ = input_data;
  output_ = output_data;
  args_ = nullptr;
}

InputDataWrapper::InputDataWrapper(const std::shared_ptr<RunArgs> &args) {
  args_ = args;
}
}  // namespace ge
