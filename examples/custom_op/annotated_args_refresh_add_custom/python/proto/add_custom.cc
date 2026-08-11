/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "add_custom.h"

namespace {
ge::graphStatus InferShape(gert::InferShapeContext *ctx) {
  const auto *input_shape = ctx->GetInputShape(0U);
  auto *output_shape = ctx->GetOutputShape(0U);
  if ((input_shape == nullptr) || (output_shape == nullptr)) {
    return ge::GRAPH_FAILED;
  }
  *output_shape = *input_shape;
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus InferDataType(gert::InferDataTypeContext *ctx) {
  return ctx->SetOutputDataType(0U, ctx->GetInputDataType(0U));
}

IMPL_OP(AnnotatedAddCustom).InferShape(InferShape).InferDataType(InferDataType);
}  // namespace
