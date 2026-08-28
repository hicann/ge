/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "custom_op_bindings.h"
#include "exe_graph/runtime/extended_kernel_context.h"
#include "exe_graph/runtime/infer_shape_context.h"
#include "exe_graph/runtime/storage_shape.h"
#include "runtime_attrs_binding.h"
#include "runtime/native_bindings/runtime_type_wrappers.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace ge {
namespace python_custom_op_native {
namespace {
namespace runtime_native = ::ge::python_runtime_native;

class BorrowedInferMetaContext {
 public:
  explicit BorrowedInferMetaContext(gert::InferShapeContext *ctx) : ctx_(ctx), valid_(std::make_shared<bool>(true)) {}

  py::object GetRequiredInputTensor(size_t ir_index) const {
    const auto *shape = GetDynamicInputShapePointer(ir_index, 0U);
    if (shape == nullptr) {
      throw std::runtime_error("Failed to get required input tensor at ir index " + std::to_string(ir_index));
    }
    return MakeTensorDesc(shape, GetInputDataTypeByIr(ir_index, 0U, true));
  }

  py::object GetOptionalInputTensor(size_t ir_index) const {
    const auto *ins_info = Get()->GetIrInputInstanceInfo(ir_index);
    if ((ins_info == nullptr) || (ins_info->GetInstanceNum() == 0U)) {
      return py::none();
    }
    const auto *shape = GetInputShape(ins_info->GetInstanceStart());
    if (shape == nullptr) {
      return py::none();
    }
    return MakeTensorDesc(shape, GetInputDataTypeByIr(ir_index, 0U, false));
  }

  size_t GetDynamicInputNum(size_t ir_index) const {
    const auto *ins_info = Get()->GetIrInputInstanceInfo(ir_index);
    if (ins_info == nullptr) {
      throw std::runtime_error("Failed to get dynamic input instance info at ir index " + std::to_string(ir_index));
    }
    return ins_info->GetInstanceNum();
  }

  py::object GetDynamicInputTensor(size_t ir_index, size_t relative_index) const {
    const auto *shape = GetDynamicInputShapePointer(ir_index, relative_index);
    if (shape == nullptr) {
      throw std::runtime_error("Failed to get dynamic input tensor at ir index " + std::to_string(ir_index) +
                               ", relative index " + std::to_string(relative_index));
    }
    return MakeTensorDesc(shape, GetInputDataTypeByIr(ir_index, relative_index, true));
  }

  py::object GetAttrs() const {
    const auto *attrs = Get()->GetAttrs();
    if (attrs == nullptr) {
      throw std::runtime_error("Failed to get runtime attrs");
    }
    return py::cast(BorrowedRuntimeAttrs(attrs, valid_));
  }

  size_t GetDynamicOutputNum(size_t ir_index) const {
    const auto *ins_info = Get()->GetIrOutputInstanceInfo(ir_index);
    if (ins_info == nullptr) {
      throw std::runtime_error("Failed to get dynamic output instance info at ir index " + std::to_string(ir_index));
    }
    return ins_info->GetInstanceNum();
  }

  void Invalidate() {
    if (valid_ != nullptr) {
      *valid_ = false;
    }
    ctx_ = nullptr;
  }

 private:
  gert::InferShapeContext *Get() const {
    if ((valid_ == nullptr) || (!(*valid_)) || (ctx_ == nullptr)) {
      throw std::runtime_error("Borrowed infer shape context has expired");
    }
    return ctx_;
  }

  const gert::Shape *GetInputShape(size_t flat_index) const {
    return Get()->GetInputShape(flat_index);
  }

  const gert::Shape *GetDynamicInputShapePointer(size_t ir_index, size_t relative_index) const {
    const auto *ins_info = Get()->GetIrInputInstanceInfo(ir_index);
    if (ins_info == nullptr) {
      return nullptr;
    }
    const auto start = ins_info->GetInstanceStart();
    if ((ins_info->GetInstanceNum() == 0U) || (relative_index >= ins_info->GetInstanceNum())) {
      return nullptr;
    }
    return GetInputShape(start + relative_index);
  }

  int32_t GetInputDataTypeByIr(size_t ir_index, size_t relative_index, bool required) const {
    const auto *ins_info = Get()->GetIrInputInstanceInfo(ir_index);
    if (ins_info == nullptr) {
      if (required) {
        throw std::runtime_error("Failed to get input instance info at ir index " + std::to_string(ir_index));
      }
      return -1;
    }
    const auto start = ins_info->GetInstanceStart();
    if ((ins_info->GetInstanceNum() == 0U) || (relative_index >= ins_info->GetInstanceNum())) {
      if (required) {
        throw std::runtime_error("Failed to get input data type at ir index " + std::to_string(ir_index));
      }
      return -1;
    }
    const auto *desc = Get()->GetInputDesc(start + relative_index);
    if (desc == nullptr) {
      if (required) {
        throw std::runtime_error("Failed to get input desc at ir index " + std::to_string(ir_index));
      }
      return -1;
    }
    return static_cast<int32_t>(desc->GetDataType());
  }

  py::object MakeTensorDesc(const gert::Shape *shape, int32_t data_type) const {
    const auto shape_dims = runtime_native::ShapeToDims(*shape);
    const auto shape_obj = py::cast(runtime_native::NativeStorageShape(shape_dims, shape_dims));
    const auto data_type_obj = runtime_native::MakeGraphTypeEnum("DataType", data_type);
    return py::cast(runtime_native::NativeTensorDesc(shape_obj, data_type_obj));
  }

  gert::InferShapeContext *ctx_{nullptr};
  std::shared_ptr<bool> valid_;
};

BorrowedInferMetaContext BorrowInferMetaContext(const py::capsule &ctx_handle) {
  if ((ctx_handle.get_pointer() == nullptr) || (ctx_handle.name() == nullptr) ||
      (std::string(ctx_handle.name()) != "gert::InferShapeContext")) {
    throw std::invalid_argument("ctx_handle is invalid");
  }
  return BorrowedInferMetaContext(static_cast<gert::InferShapeContext *>(ctx_handle.get_pointer()));
}

}  // namespace

void BindInferMetaContext(py::module_ &m) {
  py::class_<BorrowedInferMetaContext>(m, "InferMetaContext", "Borrowed context for Python infer_meta")
      .def("get_required_input_tensor", &BorrowedInferMetaContext::GetRequiredInputTensor, py::arg("ir_index"))
      .def("get_optional_input_tensor", &BorrowedInferMetaContext::GetOptionalInputTensor, py::arg("ir_index"))
      .def("get_dynamic_input_num", &BorrowedInferMetaContext::GetDynamicInputNum, py::arg("ir_index"))
      .def("get_dynamic_input_tensor", &BorrowedInferMetaContext::GetDynamicInputTensor, py::arg("ir_index"),
           py::arg("relative_index"))
      .def("get_attrs", &BorrowedInferMetaContext::GetAttrs)
      .def("get_dynamic_output_num", &BorrowedInferMetaContext::GetDynamicOutputNum, py::arg("ir_index"))
      .def("_invalidate", &BorrowedInferMetaContext::Invalidate);
  m.def("_borrow_infer_meta_context", &BorrowInferMetaContext, py::arg("ctx_handle"));
}

}  // namespace python_custom_op_native
}  // namespace ge
