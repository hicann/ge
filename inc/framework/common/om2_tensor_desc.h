/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_CXX_FRAMEWORK_COMMON_OM2_TENSOR_DESC_H_
#define AIR_CXX_FRAMEWORK_COMMON_OM2_TENSOR_DESC_H_

#include <vector>
#include <string>
#include <utility>

#include "framework/common/ge_visibility.h"
#include "graph/types.h"

namespace ge {

class VISIBILITY_EXPORT Om2TensorDesc {
 public:
  Om2TensorDesc() : size_(0) {}
  ~Om2TensorDesc() = default;

  Om2TensorDesc(const Om2TensorDesc &) = default;
  Om2TensorDesc(Om2TensorDesc &&) = default;
  Om2TensorDesc &operator=(const Om2TensorDesc &) = default;
  Om2TensorDesc &operator=(Om2TensorDesc &&) = default;

  void SetDataType(DataType data_type) {
    data_type_ = data_type;
  }
  void SetFormat(Format format) {
    format_ = format;
  }
  void SetShape(const std::vector<int64_t> &dims) {
    dims_ = dims;
  }
  void SetName(const std::string &name) {
    name_ = name;
  }
  void SetShapeRange(const std::vector<std::pair<int64_t, int64_t>> &shape_range) {
    shape_range_ = shape_range;
  }
  void SetSize(size_t size) {
    size_ = size;
  }

  DataType GetDataType() const {
    return data_type_;
  }
  Format GetFormat() const {
    return format_;
  }
  const std::vector<int64_t> &GetShape() const {
    return dims_;
  }
  const std::string &GetName() const {
    return name_;
  }
  const std::vector<std::pair<int64_t, int64_t>> &GetShapeRange() const {
    return shape_range_;
  }
  size_t GetSize() const {
    return size_;
  }

  Format GetOriginFormat() const {
    return format_;
  }
  const std::vector<int64_t> &GetOriginShape() const {
    return dims_;
  }

  int64_t GetElementNum() const {
    if (dims_.empty()) {
      return 0;
    }
    int64_t element_num = 1;
    for (int64_t dim : dims_) {
      if (dim < 0) {
        return 0;
      }
      element_num *= dim;
    }
    return element_num;
  }
  size_t GetByteSize() const {
    return size_;
  }

 private:
  DataType data_type_ = DT_UNDEFINED;
  Format format_ = FORMAT_ND;
  std::vector<int64_t> dims_;
  std::string name_;
  std::vector<std::pair<int64_t, int64_t>> shape_range_;
  size_t size_;
};

}  // namespace ge

#endif  // AIR_CXX_FRAMEWORK_COMMON_OM2_TENSOR_DESC_H_
