/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>

#include "add_custom_ir.h"
#include "graph/custom_op.h"

namespace {
constexpr size_t kInputIndexX = 0U;
constexpr size_t kInputIndexY = 1U;
constexpr size_t kOutputIndexZ = 0U;
}  // namespace

namespace ge {
class AddCustom final : public HostCpuExecuteOp, public ShapeInferOp {
 public:
  graphStatus Execute(gert::HostCpuOpExecutionContext *ctx) override {
    std::cout << "HostCpuExecuteOp::Execute for AddCustom" << std::endl;

    const gert::Tensor *input_x = ctx->GetInputTensor(kInputIndexX);
    const gert::Tensor *input_y = ctx->GetInputTensor(kInputIndexY);
    if ((input_x == nullptr) || (input_y == nullptr)) {
      std::cerr << "GetInputTensor failed, input_x=" << input_x << ", input_y=" << input_y << std::endl;
      return GRAPH_FAILED;
    }

    gert::Tensor *output_z =
        ctx->MallocOutputTensor(kOutputIndexZ, input_x->GetShape(), input_x->GetFormat(), input_x->GetDataType());
    if (output_z == nullptr) {
      std::cerr << "MallocOutputTensor failed" << std::endl;
      return GRAPH_FAILED;
    }

    const float *x = input_x->GetData<float>();
    const float *y = input_y->GetData<float>();
    float *z = output_z->GetData<float>();
    const int64_t shape_size = input_x->GetStorageShape().GetShapeSize();
    for (size_t i = 0U; i < shape_size; ++i) {
      z[i] = x[i] + y[i];
    }
    return GRAPH_SUCCESS;
  }

  graphStatus InferShape(gert::InferShapeContext *ctx) override {
    std::cout << "InferShape for AddCustom" << std::endl;
    const gert::Shape *input_shape = ctx->GetInputShape(kInputIndexX);
    gert::Shape *output_shape = ctx->GetOutputShape(kOutputIndexZ);
    if ((input_shape == nullptr) || (output_shape == nullptr)) {
      std::cerr << "InferShape failed, input_shape=" << input_shape << ", output_shape=" << output_shape << std::endl;
      return GRAPH_FAILED;
    }
    *output_shape = *input_shape;
    return GRAPH_SUCCESS;
  }

  graphStatus InferDataType(gert::InferDataTypeContext *ctx) override {
    std::cout << "InferDataType for AddCustom" << std::endl;
    return ctx->SetOutputDataType(kOutputIndexZ, ctx->GetInputDataType(kInputIndexX));
  }
};

REG_OP_BACKEND(AddCustom, "AddCustom", ge::OpBackend::kHostCPU);
}  // namespace ge
