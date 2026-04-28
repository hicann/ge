/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef METADEF_CXX_INC_GRAPH_UTILS_PREPARE_INFER_SHAPE_CONTEXT_H_
#define METADEF_CXX_INC_GRAPH_UTILS_PREPARE_INFER_SHAPE_CONTEXT_H_

#include <memory>
#include <vector>

#include "exe_graph/lowering/kernel_run_context_builder.h"
#include "exe_graph/runtime/infer_shape_context.h"
#include "graph/op_desc.h"
#include "graph/operator.h"

namespace gert {
bool IsInputDescValid(const ge::GeTensorDesc &input_desc, size_t &invalid_index_num);

ge::graphStatus GetTensorAddress(const ge::Operator &op, const ge::OpDescPtr &op_desc, size_t input_index,
                                 size_t invalid_index_num, TensorAddress &address,
                                 std::vector<std::unique_ptr<ge::Tensor>> &ge_tensors_holder);

ge::graphStatus GetTensorHolder(const ge::GeTensorDesc &input_desc, const gert::StorageShape &storage_shape,
                                TensorAddress address, std::unique_ptr<uint8_t[]> &tensor_holder);

ge::graphStatus ConstructCompileKernelContextInputs(const ge::Operator &op, const ge::OpDescPtr &op_desc,
                                                    std::vector<std::unique_ptr<uint8_t[]>> &inputs,
                                                    std::vector<std::unique_ptr<ge::Tensor>> &ge_tensors_holder);

ge::graphStatus ConstructInferShapeContextInputs(const ge::Operator &op, const ge::OpDescPtr &op_desc,
                                                 std::vector<std::unique_ptr<uint8_t[]>> &inputs,
                                                 std::vector<std::unique_ptr<ge::Tensor>> &ge_tensors_holder);

ge::graphStatus ConstructCompileKernelContextOutputs(const ge::OpDescPtr &op_desc,
                                                     std::vector<std::unique_ptr<uint8_t[]>> &outputs);

void ConstructDataTypeContextInputs(const ge::OpDescPtr &op_desc, std::vector<void *> &inputs);

void ConstructDataTypeContextOutputs(const ge::OpDescPtr &op_desc, std::vector<void *> &outputs);

std::vector<void *> GetInputs(const std::vector<std::unique_ptr<uint8_t[]>> &inputs_holders);

std::vector<void *> GetInputs(const ge::Operator &op, const std::vector<std::unique_ptr<uint8_t[]>> &inputs_holders);

std::vector<void *> GetOutputs(const std::vector<std::unique_ptr<uint8_t[]>> &outputs_holders);

ge::graphStatus UpdateOpDescOutShape(const ge::OpDescPtr &op_desc, gert::InferShapeContext *infer_shape_ctx);

ge::graphStatus PrepareInferShapeContext(const ge::Operator &op, const ge::OpDescPtr &op_desc,
                                         std::vector<std::unique_ptr<uint8_t[]>> &inputs_holder,
                                         std::vector<std::unique_ptr<uint8_t[]>> &outputs_holder,
                                         std::vector<std::unique_ptr<ge::Tensor>> &ge_tensors_holder);

}  // namespace gert

#endif  // METADEF_CXX_INC_GRAPH_UTILS_PREPARE_INFER_SHAPE_CONTEXT_H_
