/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <vector>

#include "common/checker.h"
#include "ge_api_c_wrapper_utils.h"
#include "graph/tensor.h"

#ifdef __cplusplus

using namespace ge;
using namespace ge::c_wrapper;
namespace {
bool BuildShape(const int64_t *dims, const size_t dims_num, Shape &shape) {
  if ((dims_num > 0U) && (dims == nullptr)) {
    return false;
  }
  std::vector<int64_t> dims_vec;
  dims_vec.reserve(dims_num);
  for (size_t i = 0U; i < dims_num; ++i) {
    dims_vec.emplace_back(dims[i]);
  }
  shape = Shape(dims_vec);
  return true;
}
}  // namespace

extern "C" {
#endif

TensorDesc *GeApiWrapper_TensorDesc_Create(const int64_t *dims, const size_t dims_num, const C_Format format,
                                           const C_DataType dtype) {
  Shape shape;
  if (!BuildShape(dims, dims_num, shape)) {
    return nullptr;
  }
  return new (std::nothrow) TensorDesc(shape, static_cast<Format>(format), static_cast<DataType>(dtype));
}

void GeApiWrapper_TensorDesc_Destroy(const TensorDesc *tensor_desc) {
  delete tensor_desc;
}

graphStatus GeApiWrapper_TensorDesc_GetShape(const TensorDesc *tensor_desc, int64_t **dims, size_t *dims_num) {
  GE_ASSERT_NOTNULL(tensor_desc);
  GE_ASSERT_NOTNULL(dims);
  GE_ASSERT_NOTNULL(dims_num);
  *dims = VecDimsToArray(tensor_desc->GetShape().GetDims(), dims_num);
  return GRAPH_SUCCESS;
}

graphStatus GeApiWrapper_TensorDesc_SetShape(TensorDesc *tensor_desc, const int64_t *dims, const size_t dims_num) {
  GE_ASSERT_NOTNULL(tensor_desc);
  Shape shape;
  if (!BuildShape(dims, dims_num, shape)) {
    return GRAPH_PARAM_INVALID;
  }
  tensor_desc->SetShape(shape);
  return GRAPH_SUCCESS;
}

graphStatus GeApiWrapper_TensorDesc_GetOriginShape(const TensorDesc *tensor_desc, int64_t **dims, size_t *dims_num) {
  GE_ASSERT_NOTNULL(tensor_desc);
  GE_ASSERT_NOTNULL(dims);
  GE_ASSERT_NOTNULL(dims_num);
  *dims = VecDimsToArray(tensor_desc->GetOriginShape().GetDims(), dims_num);
  return GRAPH_SUCCESS;
}

graphStatus GeApiWrapper_TensorDesc_SetOriginShape(TensorDesc *tensor_desc, const int64_t *dims,
                                                   const size_t dims_num) {
  GE_ASSERT_NOTNULL(tensor_desc);
  Shape shape;
  if (!BuildShape(dims, dims_num, shape)) {
    return GRAPH_PARAM_INVALID;
  }
  tensor_desc->SetOriginShape(shape);
  return GRAPH_SUCCESS;
}

C_Format GeApiWrapper_TensorDesc_GetFormat(const TensorDesc *tensor_desc) {
  GE_ASSERT_NOTNULL(tensor_desc);
  return static_cast<C_Format>(tensor_desc->GetFormat());
}

graphStatus GeApiWrapper_TensorDesc_SetFormat(TensorDesc *tensor_desc, const C_Format format) {
  GE_ASSERT_NOTNULL(tensor_desc);
  tensor_desc->SetFormat(static_cast<Format>(format));
  return GRAPH_SUCCESS;
}

C_Format GeApiWrapper_TensorDesc_GetOriginFormat(const TensorDesc *tensor_desc) {
  GE_ASSERT_NOTNULL(tensor_desc);
  return static_cast<C_Format>(tensor_desc->GetOriginFormat());
}

graphStatus GeApiWrapper_TensorDesc_SetOriginFormat(TensorDesc *tensor_desc, const C_Format format) {
  GE_ASSERT_NOTNULL(tensor_desc);
  tensor_desc->SetOriginFormat(static_cast<Format>(format));
  return GRAPH_SUCCESS;
}

C_DataType GeApiWrapper_TensorDesc_GetDataType(const TensorDesc *tensor_desc) {
  GE_ASSERT_NOTNULL(tensor_desc);
  return static_cast<C_DataType>(tensor_desc->GetDataType());
}

graphStatus GeApiWrapper_TensorDesc_SetDataType(TensorDesc *tensor_desc, const C_DataType dtype) {
  GE_ASSERT_NOTNULL(tensor_desc);
  tensor_desc->SetDataType(static_cast<DataType>(dtype));
  return GRAPH_SUCCESS;
}

#ifdef __cplusplus
}
#endif
