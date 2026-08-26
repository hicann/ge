/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef API_PYTHON_GE_GE_CUSTOM_OP_NATIVE_BINDINGS_RUNTIME_ATTRS_BINDING_H_
#define API_PYTHON_GE_GE_CUSTOM_OP_NATIVE_BINDINGS_RUNTIME_ATTRS_BINDING_H_

#include "binding_common.h"
#include "exe_graph/runtime/continuous_vector.h"
#include "exe_graph/runtime/runtime_attrs.h"
#include "runtime/native_bindings/runtime_type_wrappers.h"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

namespace ge {
namespace python_custom_op_native {
namespace runtime_attrs_binding_detail {
namespace runtime_native = ::ge::python_runtime_native;

template <typename T>
inline const T *GetRequiredAttr(const gert::RuntimeAttrs *attrs, size_t index, const char *type_name) {
  const auto *value = attrs->GetAttrPointer<T>(index);
  if (value == nullptr) {
    throw std::runtime_error(std::string("Failed to get runtime attr type[") + type_name + "] at index[" +
                             std::to_string(index) + "]");
  }
  return value;
}

template <typename T>
inline py::list BuildTypedList(const gert::TypedContinuousVector<T> *value) {
  if (value == nullptr) {
    throw std::runtime_error("Runtime attr list is null");
  }
  py::list result;
  const auto *data = value->GetData();
  for (size_t index = 0U; index < value->GetSize(); ++index) {
    result.append(data[index]);
  }
  return result;
}

inline py::list BuildDataTypeList(const gert::ContinuousVector *value) {
  if (value == nullptr) {
    throw std::runtime_error("Runtime attr data type list is null");
  }
  py::list result;
  const auto *data = static_cast<const ge::DataType *>(value->GetData());
  for (size_t index = 0U; index < value->GetSize(); ++index) {
    result.append(runtime_native::MakeGraphTypeEnum("DataType", static_cast<int32_t>(data[index])));
  }
  return result;
}

inline py::list BuildBoolList(const gert::ContinuousVector *value) {
  if (value == nullptr) {
    throw std::runtime_error("Runtime attr bool list is null");
  }
  py::list result;
  const auto *data = static_cast<const uint8_t *>(value->GetData());
  for (size_t index = 0U; index < value->GetSize(); ++index) {
    result.append(data[index] != 0U);
  }
  return result;
}

inline py::list BuildStringList(const gert::ContinuousVector *value) {
  if (value == nullptr) {
    throw std::runtime_error("Runtime attr string list is null");
  }
  py::list result;
  const auto *data = static_cast<const char *>(value->GetData());
  for (size_t index = 0U; index < value->GetSize(); ++index) {
    result.append(py::str(data));
    data += std::strlen(data) + 1U;
  }
  return result;
}

inline py::list BuildIntListList(const gert::ContinuousVectorVector *value) {
  if (value == nullptr) {
    throw std::runtime_error("Runtime attr nested int list is null");
  }
  py::list result;
  for (size_t index = 0U; index < value->GetSize(); ++index) {
    const auto *inner = value->Get(index);
    if (inner == nullptr) {
      throw std::runtime_error("Runtime attr nested int list element is null");
    }
    const auto *data = static_cast<const int64_t *>(inner->GetData());
    py::list inner_result;
    for (size_t inner_index = 0U; inner_index < inner->GetSize(); ++inner_index) {
      inner_result.append(data[inner_index]);
    }
    result.append(std::move(inner_result));
  }
  return result;
}
}  // namespace runtime_attrs_binding_detail

class BorrowedRuntimeAttrs {
 public:
  BorrowedRuntimeAttrs(const gert::RuntimeAttrs *attrs, std::shared_ptr<bool> valid)
      : BorrowedRuntimeAttrs(attrs, std::move(valid), false) {}

  BorrowedRuntimeAttrs(const gert::RuntimeAttrs *attrs, std::shared_ptr<bool> valid, bool read_only)
      : attrs_(attrs), valid_(std::move(valid)), read_only_(read_only) {}

  int64_t GetInt(size_t index) const {
    return *runtime_attrs_binding_detail::GetRequiredAttr<int64_t>(Get(), index, "VT_INT");
  }
  float GetFloat(size_t index) const {
    return *runtime_attrs_binding_detail::GetRequiredAttr<float>(Get(), index, "VT_FLOAT");
  }
  bool GetBool(size_t index) const {
    return *runtime_attrs_binding_detail::GetRequiredAttr<bool>(Get(), index, "VT_BOOL");
  }
  std::string GetStr(size_t index) const {
    return runtime_attrs_binding_detail::GetRequiredAttr<char>(Get(), index, "VT_STRING");
  }
  py::object GetDataType(size_t index) const {
    const auto value = *runtime_attrs_binding_detail::GetRequiredAttr<ge::DataType>(Get(), index, "VT_DATA_TYPE");
    return runtime_attrs_binding_detail::runtime_native::MakeGraphTypeEnum("DataType", static_cast<int32_t>(value));
  }
  py::object GetTensor(size_t index) const {
    const auto *tensor = runtime_attrs_binding_detail::GetRequiredAttr<gert::Tensor>(Get(), index, "VT_TENSOR");
    if (read_only_) {
      return py::cast(runtime_attrs_binding_detail::runtime_native::NativeTensor::Borrow(tensor, valid_));
    }
    return py::cast(
        runtime_attrs_binding_detail::runtime_native::NativeTensor::Borrow(const_cast<gert::Tensor *>(tensor), valid_));
  }
  py::list GetListInt(size_t index) const {
    return runtime_attrs_binding_detail::BuildTypedList(Get()->GetListInt(index));
  }
  py::list GetListFloat(size_t index) const {
    return runtime_attrs_binding_detail::BuildTypedList(Get()->GetListFloat(index));
  }
  py::list GetListBool(size_t index) const {
    return runtime_attrs_binding_detail::BuildBoolList(
        runtime_attrs_binding_detail::GetRequiredAttr<gert::ContinuousVector>(Get(), index, "VT_LIST_BOOL"));
  }
  py::list GetListStr(size_t index) const {
    return runtime_attrs_binding_detail::BuildStringList(
        runtime_attrs_binding_detail::GetRequiredAttr<gert::ContinuousVector>(Get(), index, "VT_LIST_STRING"));
  }
  py::list GetListDataType(size_t index) const {
    return runtime_attrs_binding_detail::BuildDataTypeList(
        runtime_attrs_binding_detail::GetRequiredAttr<gert::ContinuousVector>(Get(), index, "VT_LIST_DATA_TYPE"));
  }
  py::list GetListListInt(size_t index) const {
    return runtime_attrs_binding_detail::BuildIntListList(Get()->GetListListInt(index));
  }
  size_t GetAttrNum() const {
    return Get()->GetAttrNum();
  }

 private:
  const gert::RuntimeAttrs *Get() const {
    if ((valid_ == nullptr) || (!(*valid_)) || (attrs_ == nullptr)) {
      throw std::runtime_error("Borrowed runtime attrs have expired");
    }
    return attrs_;
  }

  const gert::RuntimeAttrs *attrs_{nullptr};
  std::shared_ptr<bool> valid_;
  bool read_only_{false};
};
}  // namespace python_custom_op_native
}  // namespace ge

#endif  // API_PYTHON_GE_GE_CUSTOM_OP_NATIVE_BINDINGS_RUNTIME_ATTRS_BINDING_H_
