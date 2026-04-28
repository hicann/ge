/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph/utils/prepare_infer_shape_context.h"
#include <cstddef>
#include <limits>
#include <memory>
#include <vector>
#include "base/registry/op_impl_space_registry_v2.h"
#include "common/checker.h"
#include "exe_graph/lowering/kernel_run_context_builder.h"
#include "graph/utils/op_desc_utils.h"
#include "graph_metadef/graph/debug/ge_util.h"

namespace gert {
namespace {
void GetStorageShape(const ge::GeTensorDesc &input_desc, gert::StorageShape &storage_shape) {
  const auto &dims = input_desc.GetOriginShape().GetDims();
  for (const auto &dim : dims) {
    (void)storage_shape.MutableOriginShape().AppendDim(dim);
    (void)storage_shape.MutableStorageShape().AppendDim(dim);
  }
}
}  // namespace

bool IsInputDescValid(const ge::GeTensorDesc &input_desc, size_t &invalid_index_num) {
  if (input_desc.IsValid() != ge::GRAPH_SUCCESS) {
    if (invalid_index_num < std::numeric_limits<size_t>::max()) {
      ++invalid_index_num;
    }
    return false;
  }
  return true;
}

ge::graphStatus GetTensorAddress(const ge::Operator &op, const ge::OpDescPtr &op_desc, size_t input_index,
                                 size_t invalid_index_num, TensorAddress &address,
                                 std::vector<std::unique_ptr<ge::Tensor>> &ge_tensors_holder) {
  const auto *const space_registry =
      DefaultOpImplSpaceRegistryV2::GetInstance()
          .GetSpaceRegistry(static_cast<gert::OppImplVersionTag>(op_desc->GetOppImplVersion()))
          .get();
  if (space_registry == nullptr) {
    return ge::GRAPH_SUCCESS;
  }
  const auto &functions = space_registry->GetOpImpl(op_desc->GetType().c_str());
  if (functions == nullptr) {
    return ge::GRAPH_SUCCESS;
  }
  const size_t instance_index = input_index - invalid_index_num;
  const auto valid_op_ir_map = ge::OpDescUtils::GetInputIrIndexes2InstanceIndexesPairMap(op_desc);
  if (valid_op_ir_map.empty()) {
    return ge::GRAPH_PARAM_INVALID;
  }
  size_t ir_index;
  GE_ASSERT_GRAPH_SUCCESS(ge::OpDescUtils::GetInputIrIndexByInstanceIndex(op_desc, instance_index, ir_index),
                          "[Get][InputIrIndexByInstanceIndex] failed, op[%s], instance index[%zu], input_index[%zu]",
                          op_desc->GetName().c_str(), instance_index, input_index);
  if ((functions != nullptr) && functions->IsInputDataDependency(ir_index)) {
    ge_tensors_holder[input_index] = ge::ComGraphMakeUnique<ge::Tensor>();
    GE_ASSERT_NOTNULL(ge_tensors_holder[input_index], "Create ge tensor holder inputs failed.");
    const auto index_name = op_desc->GetInputNameByIndex(static_cast<uint32_t>(input_index));
    if (op.GetInputConstData(index_name.c_str(), *(ge_tensors_holder[input_index].get())) == ge::GRAPH_SUCCESS) {
      address = ge_tensors_holder[input_index]->GetData();
    }
  }
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus GetTensorHolder(const ge::GeTensorDesc &input_desc, const gert::StorageShape &storage_shape,
                                TensorAddress address, std::unique_ptr<uint8_t[]> &tensor_holder) {
  tensor_holder = ge::ComGraphMakeUnique<uint8_t[]>(sizeof(gert::Tensor));
  GE_ASSERT_NOTNULL(tensor_holder, "Create context holder inputs failed.");
  if (address == nullptr) {
    new (tensor_holder.get()) gert::Tensor(storage_shape,
                                           {input_desc.GetOriginFormat(), input_desc.GetFormat(), {}},
                                           input_desc.GetDataType());
  } else {
    new (tensor_holder.get()) gert::Tensor(storage_shape,
                                           {input_desc.GetOriginFormat(), input_desc.GetFormat(), {}},
                                           gert::kOnHost, input_desc.GetDataType(), address);
  }
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus ConstructCompileKernelContextInputs(const ge::Operator &op, const ge::OpDescPtr &op_desc,
                                                     std::vector<std::unique_ptr<uint8_t[]>> &inputs,
                                                     std::vector<std::unique_ptr<ge::Tensor>> &ge_tensors_holder) {
  ge_tensors_holder.resize(op_desc->GetAllInputsSize());
  size_t invalid_index_num = 0UL;
  for (size_t i = 0UL; i < op_desc->GetAllInputsSize(); ++i) {
    if (!IsInputDescValid(op_desc->GetInputDesc(static_cast<uint32_t>(i)), invalid_index_num)) {
      GELOGD("input desc is not valid, skip add input[%zu] into context inputs.", i);
      continue;
    }
    gert::StorageShape storage_shape;
    GetStorageShape(op_desc->GetInputDesc(static_cast<uint32_t>(i)), storage_shape);
    TensorAddress address = nullptr;
    auto status = GetTensorAddress(op, op_desc, i, invalid_index_num, address, ge_tensors_holder);
    if (status != ge::GRAPH_SUCCESS) {
      return status;
    }
    std::unique_ptr<uint8_t[]> tensor_holder;
    status = GetTensorHolder(op_desc->GetInputDesc(static_cast<uint32_t>(i)), storage_shape, address, tensor_holder);
    if (status != ge::GRAPH_SUCCESS) {
      return status;
    }
    inputs.emplace_back(std::move(tensor_holder));
  }
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus ConstructInferShapeContextInputs(const ge::Operator &op, const ge::OpDescPtr &op_desc,
                                                 std::vector<std::unique_ptr<uint8_t[]>> &inputs,
                                                 std::vector<std::unique_ptr<ge::Tensor>> &ge_tensors_holder) {
  GE_ASSERT_GRAPH_SUCCESS(ConstructCompileKernelContextInputs(op, op_desc, inputs, ge_tensors_holder));
  inputs.emplace_back(nullptr);
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus ConstructCompileKernelContextOutputs(const ge::OpDescPtr &op_desc,
                                                     std::vector<std::unique_ptr<uint8_t[]>> &outputs) {
  auto size = op_desc->GetAllOutputsDescSize();
  while (size-- > 0) {
    auto tensor_holder = ge::ComGraphMakeUnique<uint8_t[]>(sizeof(gert::Tensor));
    GE_ASSERT_NOTNULL(tensor_holder, "Create context holder outputs failed, op[%s]", op_desc->GetName().c_str());
    outputs.emplace_back(std::move(tensor_holder));
  }
  return ge::GRAPH_SUCCESS;
}

void ConstructDataTypeContextInputs(const ge::OpDescPtr &op_desc, std::vector<void *> &inputs) {
  inputs.reserve(op_desc->GetAllInputsSize());
  for (size_t i = 0UL; i < op_desc->GetAllInputsSize(); ++i) {
    const auto &compile_tensor = op_desc->MutableInputDesc(static_cast<uint32_t>(i));
    if (compile_tensor == nullptr) {
      GELOGD("OpDesc[%s]type[%s], input desc[%zu] is nullptr, skip constructing rt2 ctx for it.", op_desc->GetNamePtr(),
             op_desc->GetTypePtr(), i);
      continue;
    }
    inputs.emplace_back(reinterpret_cast<void *>(compile_tensor->GetDataType()));
  }
}

void ConstructDataTypeContextOutputs(const ge::OpDescPtr &op_desc, std::vector<void *> &outputs) {
  outputs.reserve(op_desc->GetAllOutputsDescSize());
  for (size_t i = 0UL; i < op_desc->GetAllOutputsDescSize(); ++i) {
    const auto &compile_tensor = op_desc->GetOutputDesc(static_cast<uint32_t>(i));
    outputs.emplace_back(reinterpret_cast<void *>(compile_tensor.GetDataType()));
  }
}

std::vector<void *> GetInputs(const std::vector<std::unique_ptr<uint8_t[]>> &inputs_holders) {
  std::vector<void *> inputs;
  inputs.reserve(inputs_holders.size());
  for (const auto &input_holder : inputs_holders) {
    inputs.emplace_back(input_holder.get());
  }
  return inputs;
}

std::vector<void *> GetInputs(const ge::Operator &op, const std::vector<std::unique_ptr<uint8_t[]>> &inputs_holders) {
  std::vector<void *> inputs;
  inputs.reserve(inputs_holders.size() + 1UL);
  for (const auto &input_holder : inputs_holders) {
    inputs.emplace_back(input_holder.get());
  }
  inputs.emplace_back(op.GetInferenceContext().get());
  return inputs;
}

std::vector<void *> GetOutputs(const std::vector<std::unique_ptr<uint8_t[]>> &outputs_holders) {
  std::vector<void *> outputs;
  outputs.reserve(outputs_holders.size());
  for (const auto &output_holder : outputs_holders) {
    outputs.emplace_back(output_holder.get());
  }
  return outputs;
}

ge::graphStatus UpdateOpDescOutShape(const ge::OpDescPtr &op_desc, gert::InferShapeContext *infer_shape_ctx) {
  for (size_t index = 0UL; index < op_desc->GetOutputsSize(); ++index) {
    auto &dst_out_shape = op_desc->MutableOutputDesc(static_cast<uint32_t>(index))->MutableShape();
    const auto *shape = infer_shape_ctx->GetOutputShape(index);
    GE_ASSERT_NOTNULL(shape);
    dst_out_shape.SetDimNum(shape->GetDimNum());
    for (size_t dim = 0UL; dim < shape->GetDimNum(); ++dim) {
      (void)dst_out_shape.SetDim(dim, shape->GetDim(dim));
    }
    op_desc->MutableOutputDesc(static_cast<uint32_t>(index))->SetOriginShape(dst_out_shape);
  }
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus PrepareInferShapeContext(const ge::Operator &op, const ge::OpDescPtr &op_desc,
                                         std::vector<std::unique_ptr<uint8_t[]>> &inputs_holder,
                                         std::vector<std::unique_ptr<uint8_t[]>> &outputs_holder,
                                         std::vector<std::unique_ptr<ge::Tensor>> &ge_tensors_holder) {
  ge_tensors_holder.clear();
  ge_tensors_holder.resize(op_desc->GetAllInputsSize());
  GE_ASSERT_GRAPH_SUCCESS(ConstructInferShapeContextInputs(op, op_desc, inputs_holder, ge_tensors_holder),
                          "[Construct][InferShapeContextInputs] failed, op_desc[%s]", op_desc->GetName().c_str());
  GE_ASSERT_GRAPH_SUCCESS(ConstructCompileKernelContextOutputs(op_desc, outputs_holder),
                          "[Construct][InferShapeContextOutputs] failed, op_desc[%s]", op_desc->GetName().c_str());
  return ge::GRAPH_SUCCESS;
}

}  // namespace gert
