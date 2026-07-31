/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "bindings.h"
#include "exe_graph/runtime/continuous_vector.h"
#include "exe_graph/runtime/eager_op_execution_context.h"
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
namespace {
namespace runtime_native = ::ge::python_runtime_native;

template <typename T>
const T *GetRequiredAttr(const gert::RuntimeAttrs *attrs, const size_t index, const char *type_name) {
  const auto *value = attrs->GetAttrPointer<T>(index);
  if (value == nullptr) {
    throw std::runtime_error(std::string("Failed to get runtime attr type[") + type_name + "] at index[" +
                             std::to_string(index) + "]");
  }
  return value;
}

template <typename T>
py::list BuildTypedList(const gert::TypedContinuousVector<T> *value) {
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

py::list BuildDataTypeList(const gert::ContinuousVector *value) {
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

py::list BuildBoolList(const gert::ContinuousVector *value) {
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

py::list BuildStringList(const gert::ContinuousVector *value) {
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

py::list BuildIntListList(const gert::ContinuousVectorVector *value) {
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

class BorrowedRuntimeAttrs {
 public:
  BorrowedRuntimeAttrs(const gert::RuntimeAttrs *attrs, std::shared_ptr<bool> valid)
      : attrs_(attrs), valid_(std::move(valid)) {}

  int64_t GetInt(size_t index) const {
    return *GetRequiredAttr<int64_t>(Get(), index, "VT_INT");
  }

  float GetFloat(size_t index) const {
    return *GetRequiredAttr<float>(Get(), index, "VT_FLOAT");
  }

  bool GetBool(size_t index) const {
    return *GetRequiredAttr<bool>(Get(), index, "VT_BOOL");
  }

  std::string GetStr(size_t index) const {
    return GetRequiredAttr<char>(Get(), index, "VT_STRING");
  }

  py::object GetDataType(size_t index) const {
    const auto value = *GetRequiredAttr<ge::DataType>(Get(), index, "VT_DATA_TYPE");
    return runtime_native::MakeGraphTypeEnum("DataType", static_cast<int32_t>(value));
  }

  py::object GetTensor(size_t index) const {
    const auto *tensor = GetRequiredAttr<gert::Tensor>(Get(), index, "VT_TENSOR");
    return py::cast(runtime_native::NativeTensor::Borrow(const_cast<gert::Tensor *>(tensor), valid_));
  }

  py::list GetListInt(size_t index) const {
    return BuildTypedList(Get()->GetListInt(index));
  }

  py::list GetListFloat(size_t index) const {
    return BuildTypedList(Get()->GetListFloat(index));
  }

  py::list GetListBool(size_t index) const {
    return BuildBoolList(GetRequiredAttr<gert::ContinuousVector>(Get(), index, "VT_LIST_BOOL"));
  }

  py::list GetListStr(size_t index) const {
    return BuildStringList(GetRequiredAttr<gert::ContinuousVector>(Get(), index, "VT_LIST_STRING"));
  }

  py::list GetListDataType(size_t index) const {
    return BuildDataTypeList(GetRequiredAttr<gert::ContinuousVector>(Get(), index, "VT_LIST_DATA_TYPE"));
  }

  py::list GetListListInt(size_t index) const {
    return BuildIntListList(Get()->GetListListInt(index));
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
};

class BorrowedEagerOpExecutionContext {
 public:
  explicit BorrowedEagerOpExecutionContext(gert::EagerOpExecutionContext *ctx)
      : ctx_(ctx), valid_(std::make_shared<bool>(true)) {}

  py::object GetInputTensor(size_t index) const {
    const auto *tensor = Get()->GetInputTensor(index);
    return CastRequiredTensor(tensor, "Failed to get input tensor");
  }

  size_t GetInputNum() const {
    return Get()->GetComputeNodeInputNum();
  }

  size_t GetDynamicInputNum(size_t ir_index) const {
    const auto *instance_info = Get()->GetIrInputInstanceInfo(ir_index);
    if (instance_info == nullptr) {
      throw std::runtime_error("Failed to get dynamic input instance info");
    }
    return instance_info->GetInstanceNum();
  }

  py::object GetAttrs() const {
    const auto *attrs = Get()->GetAttrs();
    if (attrs == nullptr) {
      throw std::runtime_error("Failed to get runtime attrs");
    }
    return py::cast(BorrowedRuntimeAttrs(attrs, valid_));
  }

  py::object GetRequiredInputTensor(size_t ir_index) const {
    const auto *tensor = Get()->GetRequiredInputTensor(ir_index);
    return CastRequiredTensor(tensor, "Failed to get required input tensor");
  }

  py::object GetOptionalInputTensor(size_t ir_index) const {
    const auto *tensor = Get()->GetOptionalInputTensor(ir_index);
    if (tensor == nullptr) {
      return py::none();
    }
    return CastTensor(tensor);
  }

  py::object GetDynamicInputTensor(size_t ir_index, size_t relative_index) const {
    const auto *tensor = Get()->GetDynamicInputTensor(ir_index, relative_index);
    return CastRequiredTensor(tensor, "Failed to get dynamic input tensor");
  }

  py::object GetOutputTensor(size_t index) const {
    const auto *tensor = Get()->GetOutputTensor(index);
    return CastRequiredTensor(tensor, "Failed to get output tensor");
  }

  py::object MakeOutputRefInput(size_t output_index, size_t input_index) const {
    auto *tensor = Get()->MakeOutputRefInput(output_index, input_index);
    if (tensor == nullptr) {
      throw std::runtime_error("Failed to make output reference input");
    }
    return py::cast(runtime_native::NativeTensor::Borrow(tensor, valid_));
  }

  py::object MallocOutputTensor(size_t index, const py::object &shape_obj, const py::object &format_obj,
                                int32_t dtype) const {
    const auto &shape = shape_obj.cast<const runtime_native::NativeStorageShape &>();
    const auto &format = format_obj.cast<const runtime_native::NativeStorageFormat &>();
    auto *tensor = Get()->MallocOutputTensor(index, *shape.Get(), *format.Get(), static_cast<ge::DataType>(dtype));
    if (tensor == nullptr) {
      throw std::runtime_error("Failed to malloc output tensor");
    }
    return py::cast(runtime_native::NativeTensor::Borrow(tensor, valid_));
  }

  uintptr_t MallocWorkSpace(size_t size) const {
    auto *workspace = Get()->MallocWorkSpace(size);
    if (workspace == nullptr) {
      throw std::runtime_error("Failed to malloc workspace");
    }
    return reinterpret_cast<uintptr_t>(workspace);
  }

  uintptr_t GetStream() const {
    return reinterpret_cast<uintptr_t>(Get()->GetStream());
  }

  void Invalidate() {
    if (valid_ != nullptr) {
      *valid_ = false;
    }
    ctx_ = nullptr;
  }

 private:
  gert::EagerOpExecutionContext *Get() const {
    if ((valid_ == nullptr) || (!(*valid_)) || (ctx_ == nullptr)) {
      throw std::runtime_error("Borrowed native object has expired");
    }
    return ctx_;
  }

  py::object CastTensor(const gert::Tensor *tensor) const {
    return py::cast(runtime_native::NativeTensor::Borrow(const_cast<gert::Tensor *>(tensor), valid_));
  }

  py::object CastRequiredTensor(const gert::Tensor *tensor, const char *message) const {
    if (tensor == nullptr) {
      throw std::runtime_error(message);
    }
    return CastTensor(tensor);
  }

  gert::EagerOpExecutionContext *ctx_{nullptr};
  std::shared_ptr<bool> valid_;
};

BorrowedEagerOpExecutionContext BorrowEagerOpExecutionContext(uintptr_t ctx_handle) {
  if (ctx_handle == 0U) {
    throw std::invalid_argument("ctx_handle is null");
  }
  return BorrowedEagerOpExecutionContext(reinterpret_cast<gert::EagerOpExecutionContext *>(ctx_handle));
}

}  // namespace

void BindEagerOpExecutionContext(py::module_ &m) {
  py::class_<BorrowedRuntimeAttrs>(m, "RuntimeAttrs", "Borrowed view of gert::RuntimeAttrs")
      .def("get_int", &BorrowedRuntimeAttrs::GetInt, py::arg("index"))
      .def("get_float", &BorrowedRuntimeAttrs::GetFloat, py::arg("index"))
      .def("get_bool", &BorrowedRuntimeAttrs::GetBool, py::arg("index"))
      .def("get_str", &BorrowedRuntimeAttrs::GetStr, py::arg("index"))
      .def("get_data_type", &BorrowedRuntimeAttrs::GetDataType, py::arg("index"))
      .def("get_tensor", &BorrowedRuntimeAttrs::GetTensor, py::arg("index"))
      .def("get_list_int", &BorrowedRuntimeAttrs::GetListInt, py::arg("index"))
      .def("get_list_float", &BorrowedRuntimeAttrs::GetListFloat, py::arg("index"))
      .def("get_list_bool", &BorrowedRuntimeAttrs::GetListBool, py::arg("index"))
      .def("get_list_str", &BorrowedRuntimeAttrs::GetListStr, py::arg("index"))
      .def("get_list_data_type", &BorrowedRuntimeAttrs::GetListDataType, py::arg("index"))
      .def("get_list_list_int", &BorrowedRuntimeAttrs::GetListListInt, py::arg("index"))
      .def("get_attr_num", &BorrowedRuntimeAttrs::GetAttrNum);

  py::class_<BorrowedEagerOpExecutionContext>(m, "EagerOpExecutionContext",
                                              "Borrowed view of gert::EagerOpExecutionContext")
      .def("get_input_tensor", &BorrowedEagerOpExecutionContext::GetInputTensor, py::arg("index"))
      .def("get_input_num", &BorrowedEagerOpExecutionContext::GetInputNum)
      .def("get_dynamic_input_num", &BorrowedEagerOpExecutionContext::GetDynamicInputNum, py::arg("ir_index"))
      .def("get_attrs", &BorrowedEagerOpExecutionContext::GetAttrs)
      .def("get_required_input_tensor", &BorrowedEagerOpExecutionContext::GetRequiredInputTensor, py::arg("ir_index"))
      .def("get_optional_input_tensor", &BorrowedEagerOpExecutionContext::GetOptionalInputTensor, py::arg("ir_index"))
      .def("get_dynamic_input_tensor", &BorrowedEagerOpExecutionContext::GetDynamicInputTensor, py::arg("ir_index"),
           py::arg("relative_index"))
      .def("malloc_output_tensor", &BorrowedEagerOpExecutionContext::MallocOutputTensor, py::arg("index"),
           py::arg("shape"), py::arg("format"), py::arg("dtype"))
      .def("make_output_ref_input", &BorrowedEagerOpExecutionContext::MakeOutputRefInput, py::arg("output_index"),
           py::arg("input_index"))
      .def("malloc_workspace", &BorrowedEagerOpExecutionContext::MallocWorkSpace, py::arg("size"))
      .def("get_output_tensor", &BorrowedEagerOpExecutionContext::GetOutputTensor, py::arg("index"))
      .def("get_stream", &BorrowedEagerOpExecutionContext::GetStream)
      .def("_invalidate", &BorrowedEagerOpExecutionContext::Invalidate);
  m.def("_borrow_eager_op_execution_context", &BorrowEagerOpExecutionContext, py::arg("ctx_handle"));
}

}  // namespace python_custom_op_native
}  // namespace ge
