/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <array>
#include <functional>
#include <numeric>
#include <vector>

#include "common/checker.h"
#include "common/plugin/ge_make_unique_util.h"
#include "framework/common/framework_types_internal.h"
#include "graph/optimize/symbolic/infer_symbolic_shape/symbolic_infer_util.h"
#include "graph/optimize/symbolic/symbol_compute_context.h"
#include "graph/optimize/symbolic/symbolic_kernel_factory.h"

namespace ge {
namespace {
constexpr size_t kXIndex = 0U;
constexpr size_t kIndicesIndex = 1U;
constexpr size_t kUpdatesIndex = 2U;
constexpr size_t kOutputIndex = 0U;
constexpr int64_t kMaxSymbolicValueSize = 200L;

int64_t ElementCount(const std::vector<int64_t> &dims) {
  return std::accumulate(dims.begin(), dims.end(), 1L, std::multiplies<int64_t>());
}

std::vector<int64_t> MakeStrides(const std::vector<int64_t> &dims) {
  std::vector<int64_t> strides(dims.size(), 1L);
  for (int64_t i = static_cast<int64_t>(dims.size()) - 2L; i >= 0L; --i) {
    strides[static_cast<size_t>(i)] = strides[static_cast<size_t>(i + 1L)] * dims[static_cast<size_t>(i + 1L)];
  }
  return strides;
}

std::vector<int64_t> Coordinates(int64_t index, const std::vector<int64_t> &dims, const std::vector<int64_t> &strides) {
  std::vector<int64_t> coordinates(dims.size(), 0L);
  for (size_t i = 0U; i < dims.size(); ++i) {
    coordinates[i] = (index / strides[i]) % dims[i];
  }
  return coordinates;
}

struct ScatterInputs {
  std::vector<int64_t> x_dims;
  std::vector<int64_t> indices_dims;
  std::vector<int64_t> updates_dims;
  const std::vector<Expression> *x_values = nullptr;
  const std::vector<Expression> *indices_values = nullptr;
  const std::vector<Expression> *updates_values = nullptr;
};

graphStatus GetInput(const gert::InferSymbolComputeContext *context, size_t index, std::vector<int64_t> &dims,
                     const std::vector<Expression> *&values) {
  if (!context->GetConstInputDims(index, dims)) {
    GELOGW("TensorScatterUpdate symbolic compute unsupported: input[%zu] Shape is not constant, node %s[%s].", index,
           context->GetNodeName(), context->GetNodeType());
    return UNSUPPORTED;
  }
  auto tensor = context->GetInputSymbolTensor(index);
  GE_UNSUPPORTED_IF_NULL(tensor);
  values = tensor->GetSymbolicValue();
  GE_UNSUPPORTED_IF_NULL(values);
  return GRAPH_SUCCESS;
}

graphStatus GetScatterInputs(const gert::InferSymbolComputeContext *context, ScatterInputs &inputs) {
  const std::array<std::vector<int64_t> *, 3U> dims = {&inputs.x_dims, &inputs.indices_dims, &inputs.updates_dims};
  const std::array<const std::vector<Expression> **, 3U> values = {&inputs.x_values, &inputs.indices_values,
                                                                   &inputs.updates_values};
  const std::array<size_t, 3U> input_indices = {kXIndex, kIndicesIndex, kUpdatesIndex};
  for (size_t i = 0U; i < input_indices.size(); ++i) {
    const auto ret = GetInput(context, input_indices[i], *dims[i], *values[i]);
    if (ret != GRAPH_SUCCESS) {
      return ret;
    }
  }
  return GRAPH_SUCCESS;
}

graphStatus ValidateScatterInputs(const gert::InferSymbolComputeContext *context, const ScatterInputs &inputs) {
  const int64_t x_count = ElementCount(inputs.x_dims);
  const int64_t indices_count = ElementCount(inputs.indices_dims);
  const int64_t updates_count = ElementCount(inputs.updates_dims);
  if (x_count <= 0L || x_count > kMaxSymbolicValueSize || indices_count < 0L || updates_count < 0L ||
      inputs.x_values->size() != static_cast<size_t>(x_count) ||
      inputs.indices_values->size() != static_cast<size_t>(indices_count) ||
      inputs.updates_values->size() != static_cast<size_t>(updates_count) || inputs.indices_dims.empty()) {
    GELOGW(
        "TensorScatterUpdate symbolic compute unsupported: input Shape or SymbolicValue size is invalid, node %s[%s].",
        context->GetNodeName(), context->GetNodeType());
    return UNSUPPORTED;
  }

  const int64_t index_depth = inputs.indices_dims.back();
  if (index_depth < 1L || index_depth > static_cast<int64_t>(inputs.x_dims.size())) {
    GELOGW(
        "TensorScatterUpdate symbolic compute unsupported: index depth[%lld] is invalid for x rank[%zu], node %s[%s].",
        index_depth, inputs.x_dims.size(), context->GetNodeName(), context->GetNodeType());
    return UNSUPPORTED;
  }

  std::vector<int64_t> expected_updates_dims(inputs.indices_dims.begin(), inputs.indices_dims.end() - 1);
  expected_updates_dims.insert(expected_updates_dims.end(), inputs.x_dims.begin() + index_depth, inputs.x_dims.end());
  if (expected_updates_dims != inputs.updates_dims) {
    GELOGW("TensorScatterUpdate symbolic compute unsupported: updates Shape does not match indices and x, node %s[%s].",
           context->GetNodeName(), context->GetNodeType());
    return UNSUPPORTED;
  }
  return GRAPH_SUCCESS;
}

graphStatus ApplyScatterUpdates(const gert::InferSymbolComputeContext *context, const ScatterInputs &inputs,
                                std::vector<Expression> &output) {
  const auto &indices = *inputs.indices_values;
  const auto &updates = *inputs.updates_values;
  const auto x_strides = MakeStrides(inputs.x_dims);
  const std::vector<int64_t> suffix_dims(inputs.updates_dims.begin() + inputs.indices_dims.size() - 1,
                                         inputs.updates_dims.end());
  const auto suffix_strides = MakeStrides(suffix_dims);
  const int64_t index_depth = inputs.indices_dims.back();
  const int64_t index_count = static_cast<int64_t>(indices.size()) / index_depth;
  const int64_t update_slice_size = static_cast<int64_t>(updates.size()) / index_count;
  for (int64_t update_index = 0L; update_index < index_count; ++update_index) {
    int64_t target_base = 0L;
    for (int64_t dim = 0L; dim < index_depth; ++dim) {
      int64_t coordinate = 0L;
      if (!indices[static_cast<size_t>(update_index * index_depth + dim)].GetConstValue(coordinate) ||
          coordinate < 0L || coordinate >= inputs.x_dims[static_cast<size_t>(dim)]) {
        GELOGW("TensorScatterUpdate symbolic compute unsupported: index[%lld][%lld] is invalid, node %s[%s].",
               update_index, dim, context->GetNodeName(), context->GetNodeType());
        return UNSUPPORTED;
      }
      target_base += coordinate * x_strides[static_cast<size_t>(dim)];
    }
    for (int64_t slice_index = 0L; slice_index < update_slice_size; ++slice_index) {
      const auto suffix_coordinates = Coordinates(slice_index, suffix_dims, suffix_strides);
      int64_t target_offset = target_base;
      for (size_t dim = 0U; dim < suffix_coordinates.size(); ++dim) {
        target_offset += suffix_coordinates[dim] * x_strides[static_cast<size_t>(index_depth) + dim];
      }
      output[static_cast<size_t>(target_offset)] =
          updates[static_cast<size_t>(update_index * update_slice_size + slice_index)];
    }
  }
  return GRAPH_SUCCESS;
}

}  // namespace

/**
 * TensorScatterUpdate算子的符号化计算
 * 【算子功能】根据indices指定的坐标，将updates中的值更新到输入x的对应位置，未更新位置保持x原值。
 * 【算子约束】indices最后一维长度K不超过x的Rank；indices值必须为有效的非负坐标；updates Shape必须为
 *          indices.shape[:-1]与x.shape[K:]的拼接，且参与计算的SymbolicValue元素数量不超过200。
 * 【推导逻辑】先读取三个输入的Shape和SymbolicValue，依据indices最后一维得到索引深度K，并校验updates
 *          Shape等于indices去掉最后一维后拼接x的[K:]后缀。输出Shape复制x，输出值先复制x；遍历每个索引
 *          向量计算目标切片起始线性位置，再结合更新切片坐标逐元素覆盖输出，重复索引按遍历顺序后写覆盖前写。
 * 【举例】x Shape=[2,3]、value=[0,1,2,3,4,5]，indices Shape=[2,1]、value=[0,1]，updates Shape=[2,3]、
 *          value=[10,11,12,20,21,22]时，输出Shape=[2,3]，value为[10,11,12,20,21,22]。
 */
graphStatus TensorScatterUpdateSymbolicKernelCompute(gert::InferSymbolComputeContext *context) {
  GE_ASSERT_NOTNULL(context);
  ScatterInputs inputs;
  auto ret = GetScatterInputs(context, inputs);
  if (ret != GRAPH_SUCCESS) {
    return ret;
  }
  ret = ValidateScatterInputs(context, inputs);
  if (ret != GRAPH_SUCCESS) {
    return ret;
  }

  auto output_tensor = context->GetOutputSymbolTensor(kOutputIndex);
  GE_ASSERT_NOTNULL(output_tensor);
  output_tensor->MutableOriginSymbolShape() = context->GetInputSymbolTensor(kXIndex)->GetOriginSymbolShape();
  auto output_values = output_tensor->MutableSymbolicValue();
  GE_ASSERT_NOTNULL(output_values);
  *output_values = *inputs.x_values;

  ret = ApplyScatterUpdates(context, inputs, *output_values);
  if (ret != GRAPH_SUCCESS) {
    return ret;
  }
  return GRAPH_SUCCESS;
}

REGISTER_SYMBOLIC_KERNEL(TensorScatterUpdate, TensorScatterUpdateSymbolicKernelCompute);
}  // namespace ge
