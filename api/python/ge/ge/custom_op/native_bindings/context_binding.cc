/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "custom_op_bindings.h"
#include "exe_graph/runtime/annotated_args_context.h"
#include "exe_graph/runtime/eager_op_execution_context.h"
#include "exe_graph/runtime/op_compile_context.h"
#include "graph/ascend_string.h"
#include "platform/platform_infos_def.h"
#include "runtime_attrs_binding.h"
#include "runtime/native_bindings/runtime_type_wrappers.h"

#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace ge {
namespace python_custom_op_native {
namespace {
namespace runtime_native = ::ge::python_runtime_native;

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

class BorrowedCompilePlatformInfo {
 public:
  BorrowedCompilePlatformInfo(gert::OpCompileContext *ctx, std::shared_ptr<bool> active)
      : ctx_(ctx), active_(std::move(active)) {}

  std::string GetPlatformResource(const py::object &group_obj, const py::object &key_obj) const {
    EnsureActiveContext();
    const auto group = RequireString(group_obj, "group");
    const auto key = RequireString(key_obj, "key");
    if (group.empty()) {
      throw std::invalid_argument("platform resource group must not be empty");
    }
    if (key.empty()) {
      throw std::invalid_argument("platform resource key must not be empty");
    }
    EnsurePlatformSnapshot();
    std::string value;
    if (!platform_info_.GetPlatformResWithLock(group, key, value)) {
      throw py::key_error(group + ":" + key);
    }
    return value;
  }

  std::map<std::string, std::string> GetPlatformResourceGroup(const py::object &group_obj) const {
    EnsureActiveContext();
    const auto group = RequireString(group_obj, "group");
    if (group.empty()) {
      throw std::invalid_argument("platform resource group must not be empty");
    }
    EnsurePlatformSnapshot();
    std::map<std::string, std::string> values;
    if (!platform_info_.GetPlatformResWithLock(group, values)) {
      throw py::key_error(group);
    }
    return values;
  }

  uint32_t GetCoreNum(const py::object &core_type) const {
    EnsureActiveContext();
    if (core_type.is_none()) {
      EnsurePlatformSnapshot();
      return platform_info_.GetCoreNumWithLock();
    }
    if (!py::isinstance<py::str>(core_type)) {
      throw py::type_error("core_type must be a string or None");
    }
    const auto value = core_type.cast<std::string>();
    if (value.empty()) {
      throw std::invalid_argument("core_type must not be empty");
    }
    EnsurePlatformSnapshot();
    return platform_info_.GetCoreNumByType(value);
  }

  std::string GetSocVersion() const {
    EnsureActiveContext();
    EnsurePlatformSnapshot();
    return optional_infos_.GetSocVersion();
  }

  uint32_t GetAiCoreNum() const {
    EnsureActiveContext();
    EnsurePlatformSnapshot();
    return optional_infos_.GetAICoreNum();
  }

 private:
  static std::string RequireString(const py::object &value, const char *name) {
    if (!py::isinstance<py::str>(value)) {
      throw py::type_error(std::string(name) + " must be a string");
    }
    return value.cast<std::string>();
  }

  void EnsureActiveContext() const {
    if ((active_ == nullptr) || (!(*active_)) || (ctx_ == nullptr)) {
      throw std::runtime_error("Borrowed native object has expired");
    }
  }

  void EnsurePlatformSnapshot() const {
    EnsureActiveContext();
    if (platform_initialized_) {
      if (platform_failed_) {
        throw std::runtime_error("Failed to get platform infos");
      }
      return;
    }
    platform_initialized_ = true;
    const auto ret = ctx_->GetPlatformInfos(platform_info_, optional_infos_);
    if (ret != GRAPH_SUCCESS) {
      platform_failed_ = true;
      throw std::runtime_error("Failed to get platform infos");
    }
  }

  gert::OpCompileContext *ctx_{nullptr};
  std::shared_ptr<bool> active_;
  mutable bool platform_initialized_{false};
  mutable bool platform_failed_{false};
  mutable fe::PlatFormInfos platform_info_;
  mutable fe::OptionalInfos optional_infos_;
};

class BorrowedOpCompileContext {
 public:
  explicit BorrowedOpCompileContext(gert::OpCompileContext *ctx) : ctx_(ctx), active_(std::make_shared<bool>(true)) {}

  py::object GetRequiredInputTensor(size_t ir_index) const {
    return CastRequiredTensor(Get()->GetRequiredInputTensor(ir_index), "Failed to get required input tensor");
  }

  py::object GetOptionalInputTensor(size_t ir_index) const {
    const auto *tensor = Get()->GetOptionalInputTensor(ir_index);
    return (tensor == nullptr) ? py::none() : CastTensor(tensor);
  }

  size_t GetDynamicInputNum(size_t ir_index) const {
    const auto *instance_info = Get()->GetIrInputInstanceInfo(ir_index);
    if (instance_info == nullptr) {
      throw std::runtime_error("Failed to get dynamic input instance info");
    }
    return instance_info->GetInstanceNum();
  }

  py::object GetDynamicInputTensor(size_t ir_index, size_t relative_index) const {
    return CastRequiredTensor(Get()->GetDynamicInputTensor(ir_index, relative_index),
                              "Failed to get dynamic input tensor");
  }

  py::object GetRequiredOutputTensor(size_t ir_index) const {
    return CastRequiredTensor(Get()->GetRequiredOutputTensor(ir_index), "Failed to get required output tensor");
  }

  size_t GetDynamicOutputNum(size_t ir_index) const {
    const auto *instance_info = Get()->GetIrOutputInstanceInfo(ir_index);
    if (instance_info == nullptr) {
      throw std::runtime_error("Failed to get dynamic output instance info");
    }
    return instance_info->GetInstanceNum();
  }

  py::object GetDynamicOutputTensor(size_t ir_index, size_t relative_index) const {
    return CastRequiredTensor(Get()->GetDynamicOutputTensor(ir_index, relative_index),
                              "Failed to get dynamic output tensor");
  }

  py::object GetAttrs() const {
    const auto *attrs = Get()->GetAttrs();
    if (attrs == nullptr) {
      throw std::runtime_error("Failed to get runtime attrs");
    }
    return py::cast(BorrowedRuntimeAttrs(attrs, active_, true));
  }

  std::string GetOption(const py::object &option_key_obj) const {
    EnsureActiveContext();
    const auto option_key = RequireString(option_key_obj, "option_key");
    if (option_key.empty()) {
      throw std::invalid_argument("option_key must not be empty");
    }
    ge::AscendString option;
    const auto ret = Get()->GetOption(ge::AscendString(option_key.c_str()), option);
    if (ret != GRAPH_SUCCESS) {
      throw py::key_error(option_key);
    }
    const auto *option_value = option.GetString();
    return (option_value == nullptr) ? std::string() : std::string(option_value, option.GetLength());
  }

  BorrowedCompilePlatformInfo GetPlatformInfo() const {
    EnsureActiveContext();
    return BorrowedCompilePlatformInfo(ctx_, active_);
  }

  void Invalidate() {
    if (active_ != nullptr) {
      *active_ = false;
    }
    ctx_ = nullptr;
  }

 private:
  static std::string RequireString(const py::object &value, const char *name) {
    if (!py::isinstance<py::str>(value)) {
      throw py::type_error(std::string(name) + " must be a string");
    }
    return value.cast<std::string>();
  }

  gert::OpCompileContext *Get() const {
    EnsureActiveContext();
    return ctx_;
  }

  void EnsureActiveContext() const {
    if ((active_ == nullptr) || (!(*active_)) || (ctx_ == nullptr)) {
      throw std::runtime_error("Borrowed native object has expired");
    }
  }

  py::object CastTensor(const gert::Tensor *tensor) const {
    return py::cast(runtime_native::NativeTensor::Borrow(tensor, active_));
  }

  py::object CastRequiredTensor(const gert::Tensor *tensor, const char *message) const {
    if (tensor == nullptr) {
      throw std::runtime_error(message);
    }
    return CastTensor(tensor);
  }

  gert::OpCompileContext *ctx_{nullptr};
  std::shared_ptr<bool> active_;
};

BorrowedOpCompileContext BorrowOpCompileContext(uintptr_t ctx_handle) {
  if (ctx_handle == 0U) {
    throw std::invalid_argument("ctx_handle is null");
  }
  return BorrowedOpCompileContext(reinterpret_cast<gert::OpCompileContext *>(ctx_handle));
}

BorrowedEagerOpExecutionContext BorrowEagerOpExecutionContext(uintptr_t ctx_handle) {
  if (ctx_handle == 0U) {
    throw std::invalid_argument("ctx_handle is null");
  }
  return BorrowedEagerOpExecutionContext(reinterpret_cast<gert::EagerOpExecutionContext *>(ctx_handle));
}

void EnsureActive(const std::shared_ptr<bool> &active) {
  if ((active == nullptr) || (!(*active))) {
    throw std::runtime_error("Borrowed native object has expired");
  }
}

class NativeWorkspaceAddr {
 public:
  NativeWorkspaceAddr(gert::WorkspaceAddr workspace, std::shared_ptr<bool> active)
      : workspace_(workspace), active_(std::move(active)) {}

  uint32_t GetIndex() const {
    return Get().index;
  }

  uintptr_t GetAddr() const {
    return reinterpret_cast<uintptr_t>(Get().addr);
  }

  const gert::WorkspaceAddr &Get() const {
    EnsureActive(active_);
    return workspace_;
  }

 private:
  gert::WorkspaceAddr workspace_{};
  std::shared_ptr<bool> active_;
};

class NativeAnnotatedKernelLaunchInfo {
 public:
  NativeAnnotatedKernelLaunchInfo(std::string kernel_name, const py::bytes &kernel_bin, uint32_t block_dim,
                                  uint32_t stream_id)
      : kernel_name_(std::move(kernel_name)), block_dim_(block_dim), stream_id_(stream_id) {
    const std::string kernel_bin_str = kernel_bin;
    kernel_bin_.assign(kernel_bin_str.cbegin(), kernel_bin_str.cend());
    if (kernel_name_.empty()) {
      throw std::invalid_argument("kernel_name must not be empty");
    }
    if (kernel_bin_.empty()) {
      throw std::invalid_argument("kernel_bin must not be empty");
    }
    if (block_dim_ == 0U) {
      throw std::invalid_argument("block_dim must be greater than zero");
    }
  }

  gert::AnnotatedKernelLaunchInfo GetView() const {
    return gert::AnnotatedKernelLaunchInfo{kernel_name_.c_str(), kernel_bin_.data(), kernel_bin_.size(), block_dim_,
                                           stream_id_};
  }

  uint32_t GetStreamId() const {
    return stream_id_;
  }

 private:
  std::string kernel_name_;
  std::vector<uint8_t> kernel_bin_;
  uint32_t block_dim_{0U};
  uint32_t stream_id_{0U};
};

class NativeAnnotatedKernelArgs {
 public:
  NativeAnnotatedKernelArgs(gert::AnnotatedArgsContext *ctx, std::shared_ptr<bool> active)
      : ctx_(ctx), active_(std::move(active)) {}
  NativeAnnotatedKernelArgs(const NativeAnnotatedKernelArgs &) = delete;
  NativeAnnotatedKernelArgs &operator=(const NativeAnnotatedKernelArgs &) = delete;
  NativeAnnotatedKernelArgs(NativeAnnotatedKernelArgs &&) = default;
  NativeAnnotatedKernelArgs &operator=(NativeAnnotatedKernelArgs &&) = default;

  void AppendInput(uint32_t instance_index, const runtime_native::NativeTensor &tensor) {
    const auto input_num = GetContext()->GetComputeNodeInputNum();
    if (instance_index >= input_num) {
      throw std::out_of_range("input instance index " + std::to_string(instance_index) +
                              " is out of range for input count " + std::to_string(input_num));
    }
    Append(gert::InputAddr{instance_index, tensor.Get()->GetAddr()}, "append input");
  }

  void AppendOutput(uint32_t instance_index, const runtime_native::NativeTensor &tensor) {
    const auto output_num = GetContext()->GetComputeNodeOutputNum();
    if (instance_index >= output_num) {
      throw std::out_of_range("output instance index " + std::to_string(instance_index) +
                              " is out of range for output count " + std::to_string(output_num));
    }
    Append(gert::OutputAddr{instance_index, tensor.Get()->GetAddr()}, "append output");
  }

  void AppendWorkspace(const NativeWorkspaceAddr &workspace) {
    Append(workspace.Get(), "append workspace");
  }

  void AppendScalar(uint64_t value) {
    Append(value, "append scalar");
  }

  gert::AnnotatedKernelArgs Take() {
    (void)GetContext();
    consumed_ = true;
    return std::move(args_);
  }

 private:
  gert::AnnotatedArgsContext *GetContext() const {
    EnsureActive(active_);
    if ((ctx_ == nullptr) || consumed_) {
      throw std::runtime_error("AnnotatedKernelArgs has been consumed");
    }
    return ctx_;
  }

  template <typename T>
  void Append(const T &arg, const char *operation) {
    (void)GetContext();
    if (args_.AppendArg(arg) != GRAPH_SUCCESS) {
      throw std::runtime_error(std::string("Failed to ") + operation);
    }
  }

  gert::AnnotatedArgsContext *ctx_{nullptr};
  std::shared_ptr<bool> active_;
  gert::AnnotatedKernelArgs args_;
  bool consumed_{false};
};

class BorrowedAnnotatedArgsContext {
 public:
  explicit BorrowedAnnotatedArgsContext(gert::AnnotatedArgsContext *ctx)
      : ctx_(ctx), active_(std::make_shared<bool>(true)) {}

  py::object GetRequiredInputTensor(size_t ir_index) const {
    return CastRequiredTensor(Get()->GetRequiredInputTensor(ir_index), "Failed to get required input tensor");
  }

  py::object GetOptionalInputTensor(size_t ir_index) const {
    const auto *tensor = Get()->GetOptionalInputTensor(ir_index);
    return (tensor == nullptr) ? py::none() : CastTensor(tensor);
  }

  size_t GetDynamicInputNum(size_t ir_index) const {
    const auto *instance_info = Get()->GetIrInputInstanceInfo(ir_index);
    if (instance_info == nullptr) {
      throw std::runtime_error("Failed to get dynamic input instance info");
    }
    return instance_info->GetInstanceNum();
  }

  py::object GetDynamicInputTensor(size_t ir_index, size_t relative_index) const {
    return CastRequiredTensor(Get()->GetDynamicInputTensor(ir_index, relative_index),
                              "Failed to get dynamic input tensor");
  }

  py::object GetRequiredOutputTensor(size_t ir_index) const {
    return CastRequiredTensor(Get()->GetRequiredOutputTensor(ir_index), "Failed to get required output tensor");
  }

  size_t GetDynamicOutputNum(size_t ir_index) const {
    const auto *instance_info = Get()->GetIrOutputInstanceInfo(ir_index);
    if (instance_info == nullptr) {
      throw std::runtime_error("Failed to get dynamic output instance info");
    }
    return instance_info->GetInstanceNum();
  }

  py::object GetDynamicOutputTensor(size_t ir_index, size_t relative_index) const {
    return CastRequiredTensor(Get()->GetDynamicOutputTensor(ir_index, relative_index),
                              "Failed to get dynamic output tensor");
  }

  py::object GetAttrs() const {
    const auto *attrs = Get()->GetAttrs();
    if (attrs == nullptr) {
      throw std::runtime_error("Failed to get runtime attrs");
    }
    return py::cast(BorrowedRuntimeAttrs(attrs, active_));
  }

  NativeWorkspaceAddr MallocWorkspace(size_t size) const {
    if (size == 0U) {
      throw std::invalid_argument("workspace size must be greater than zero");
    }
    const auto workspace = Get()->MallocWorkSpace(size);
    if (workspace.addr == nullptr) {
      throw std::runtime_error("Failed to malloc workspace");
    }
    return NativeWorkspaceAddr(workspace, active_);
  }

  uint32_t GetStreamId() const {
    return Get()->GetStreamId();
  }

  NativeAnnotatedKernelArgs CreateKernelArgs() const {
    return NativeAnnotatedKernelArgs(Get(), active_);
  }

  void AddLaunch(const NativeAnnotatedKernelLaunchInfo &launch_info, NativeAnnotatedKernelArgs &args) const {
    auto *ctx = Get();
    if (launch_info.GetStreamId() != ctx->GetStreamId()) {
      throw std::invalid_argument("launch stream_id does not match current context stream_id");
    }
    auto native_args = args.Take();
    if (ctx->AddLaunch(launch_info.GetView(), std::move(native_args)) != GRAPH_SUCCESS) {
      throw std::runtime_error("Failed to add annotated kernel launch");
    }
  }

  void Invalidate() {
    if (active_ != nullptr) {
      *active_ = false;
    }
    ctx_ = nullptr;
  }

 private:
  gert::AnnotatedArgsContext *Get() const {
    EnsureActive(active_);
    if (ctx_ == nullptr) {
      throw std::runtime_error("Borrowed native object has expired");
    }
    return ctx_;
  }

  py::object CastTensor(const gert::Tensor *tensor) const {
    return py::cast(runtime_native::NativeTensor::Borrow(const_cast<gert::Tensor *>(tensor), active_));
  }

  py::object CastRequiredTensor(const gert::Tensor *tensor, const char *message) const {
    if (tensor == nullptr) {
      throw std::runtime_error(message);
    }
    return CastTensor(tensor);
  }

  gert::AnnotatedArgsContext *ctx_{nullptr};
  std::shared_ptr<bool> active_;
};

BorrowedAnnotatedArgsContext BorrowAnnotatedArgsContext(uintptr_t ctx_handle) {
  if (ctx_handle == 0U) {
    throw std::invalid_argument("ctx_handle is null");
  }
  return BorrowedAnnotatedArgsContext(reinterpret_cast<gert::AnnotatedArgsContext *>(ctx_handle));
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

void BindAnnotatedArgsContext(py::module_ &m) {
  py::class_<NativeWorkspaceAddr>(m, "WorkspaceAddr", "Borrowed annotated workspace address")
      .def_property_readonly("index", &NativeWorkspaceAddr::GetIndex)
      .def_property_readonly("addr", &NativeWorkspaceAddr::GetAddr);

  py::class_<NativeAnnotatedKernelLaunchInfo>(m, "AnnotatedKernelLaunchInfo", "Owned kernel launch metadata")
      .def(py::init<std::string, const py::bytes &, uint32_t, uint32_t>(), py::kw_only(), py::arg("kernel_name"),
           py::arg("kernel_bin"), py::arg("block_dim"), py::arg("stream_id"));

  py::class_<NativeAnnotatedKernelArgs>(m, "AnnotatedKernelArgs", "Borrowed annotated kernel args builder")
      .def("append_input", &NativeAnnotatedKernelArgs::AppendInput, py::arg("instance_index"), py::arg("tensor"))
      .def("append_output", &NativeAnnotatedKernelArgs::AppendOutput, py::arg("instance_index"), py::arg("tensor"))
      .def("append_workspace", &NativeAnnotatedKernelArgs::AppendWorkspace, py::arg("workspace"))
      .def("append_scalar", &NativeAnnotatedKernelArgs::AppendScalar, py::arg("value"));

  py::class_<BorrowedAnnotatedArgsContext>(m, "AnnotatedArgsContext", "Borrowed view of gert::AnnotatedArgsContext")
      .def("_get_required_input_tensor", &BorrowedAnnotatedArgsContext::GetRequiredInputTensor, py::arg("ir_index"))
      .def("_get_optional_input_tensor", &BorrowedAnnotatedArgsContext::GetOptionalInputTensor, py::arg("ir_index"))
      .def("_get_dynamic_input_num", &BorrowedAnnotatedArgsContext::GetDynamicInputNum, py::arg("ir_index"))
      .def("_get_dynamic_input_tensor", &BorrowedAnnotatedArgsContext::GetDynamicInputTensor, py::arg("ir_index"),
           py::arg("relative_index"))
      .def("_get_required_output_tensor", &BorrowedAnnotatedArgsContext::GetRequiredOutputTensor, py::arg("ir_index"))
      .def("_get_dynamic_output_num", &BorrowedAnnotatedArgsContext::GetDynamicOutputNum, py::arg("ir_index"))
      .def("_get_dynamic_output_tensor", &BorrowedAnnotatedArgsContext::GetDynamicOutputTensor, py::arg("ir_index"),
           py::arg("relative_index"))
      .def("_get_attrs", &BorrowedAnnotatedArgsContext::GetAttrs)
      .def("malloc_workspace", &BorrowedAnnotatedArgsContext::MallocWorkspace, py::arg("size"))
      .def("get_stream_id", &BorrowedAnnotatedArgsContext::GetStreamId)
      .def("create_kernel_args", &BorrowedAnnotatedArgsContext::CreateKernelArgs)
      .def("add_launch", &BorrowedAnnotatedArgsContext::AddLaunch, py::arg("launch_info"), py::arg("args"))
      .def("_invalidate", &BorrowedAnnotatedArgsContext::Invalidate);
  m.def("_borrow_annotated_args_context", &BorrowAnnotatedArgsContext, py::arg("ctx_handle"));
}

void BindOpCompileContext(py::module_ &m) {
  py::class_<BorrowedOpCompileContext>(m, "OpCompileContext", "Borrowed view of gert::OpCompileContext")
      .def("get_option", &BorrowedOpCompileContext::GetOption, py::arg("option_key"))
      .def("_get_platform_info", &BorrowedOpCompileContext::GetPlatformInfo)
      .def("_get_required_input_tensor", &BorrowedOpCompileContext::GetRequiredInputTensor, py::arg("ir_index"))
      .def("_get_optional_input_tensor", &BorrowedOpCompileContext::GetOptionalInputTensor, py::arg("ir_index"))
      .def("_get_dynamic_input_num", &BorrowedOpCompileContext::GetDynamicInputNum, py::arg("ir_index"))
      .def("_get_dynamic_input_tensor", &BorrowedOpCompileContext::GetDynamicInputTensor, py::arg("ir_index"),
           py::arg("relative_index"))
      .def("_get_required_output_tensor", &BorrowedOpCompileContext::GetRequiredOutputTensor, py::arg("ir_index"))
      .def("_get_dynamic_output_num", &BorrowedOpCompileContext::GetDynamicOutputNum, py::arg("ir_index"))
      .def("_get_dynamic_output_tensor", &BorrowedOpCompileContext::GetDynamicOutputTensor, py::arg("ir_index"),
           py::arg("relative_index"))
      .def("_get_attrs", &BorrowedOpCompileContext::GetAttrs)
      .def("_invalidate", &BorrowedOpCompileContext::Invalidate);
  py::class_<BorrowedCompilePlatformInfo>(m, "CompilePlatformInfo", "Borrowed platform information for compile")
      .def("get_platform_resource", &BorrowedCompilePlatformInfo::GetPlatformResource, py::arg("group"), py::arg("key"))
      .def("get_platform_resource_group", &BorrowedCompilePlatformInfo::GetPlatformResourceGroup, py::arg("group"))
      .def("get_core_num", &BorrowedCompilePlatformInfo::GetCoreNum, py::arg("core_type") = py::none())
      .def("get_soc_version", &BorrowedCompilePlatformInfo::GetSocVersion)
      .def("get_ai_core_num", &BorrowedCompilePlatformInfo::GetAiCoreNum);
  m.def("_borrow_op_compile_context", &BorrowOpCompileContext, py::arg("ctx_handle"));
}

}  // namespace python_custom_op_native
}  // namespace ge
