/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "runtime/custom_op/python_custom_op_adapter.h"

#include <map>
#include <mutex>
#include <new>
#include <utility>

#include "framework/common/debug/ge_log.h"
#include "graph_metadef/graph/debug/ge_util.h"
#include "graph/custom_op/infer_meta.h"

namespace ge {
namespace custom_op {
namespace {
graphStatus InvokePythonCustomOpInferMeta(PythonCustomOpInferMetaFn infer_meta, const std::string &op_type,
                                          gert::InferShapeContext *ctx, CustomOpInferMetaResult *result) {
  if ((infer_meta == nullptr) || (ctx == nullptr) || (result == nullptr)) {
    return GRAPH_FAILED;
  }
  result->outputs.clear();
  result->outputs.resize(ctx->GetComputeNodeOutputNum());
  std::vector<PythonCustomOpInferMetaOutputView> output_views;
  output_views.reserve(result->outputs.size());
  for (auto &output : result->outputs) {
    output_views.emplace_back(PythonCustomOpInferMetaOutputView{&output.shape, ge::DT_UNDEFINED});
  }
  const PythonCustomOpStringView op_type_view{op_type.data(), op_type.size()};
  PythonCustomOpInferMetaResultView result_view{output_views.data(), output_views.size()};
  const auto ret = infer_meta(&op_type_view, ctx, &result_view);
  if (ret != GRAPH_SUCCESS) {
    return ret;
  }
  for (size_t i = 0U; i < output_views.size(); ++i) {
    result->outputs[i].data_type = output_views[i].data_type;
  }
  return GRAPH_SUCCESS;
}

struct PythonCustomOpImplRuntimeEntry {
  explicit PythonCustomOpImplRuntimeEntry(PythonCustomOpAdapterDescriptor d, PythonCustomOpAdapterCallbacks cb)
      : desc(std::move(d)), callbacks(cb) {}

  PythonCustomOpAdapterDescriptor desc;
  PythonCustomOpAdapterCallbacks callbacks;
  size_t active_adapter_count{0U};
  std::mutex mutex;
};

class PythonCustomOpImplRuntimeRegistryImpl {
 public:
  bool Register(const PythonCustomOpAdapterDescriptor &desc, const PythonCustomOpAdapterCallbacks &callbacks) {
    if (desc.impl_descriptor_key.empty() || desc.op_type.empty() || (!callbacks.IsValid(desc.capabilities))) {
      GELOGW("Register python custom op runtime failed, descriptor key[%s], op type[%s].",
             desc.impl_descriptor_key.c_str(), desc.op_type.c_str());
      return false;
    }

    std::lock_guard<std::mutex> lock(mutex_);
    if (descriptor_key_to_runtime_entry_.find(desc.impl_descriptor_key) != descriptor_key_to_runtime_entry_.cend()) {
      GELOGW("Python custom op runtime descriptor key[%s] has already registered.", desc.impl_descriptor_key.c_str());
      return false;
    }
    auto runtime_entry = ComGraphMakeShared<PythonCustomOpImplRuntimeEntry>(desc, callbacks);
    if (runtime_entry == nullptr) {
      GELOGE(GRAPH_FAILED, "Create python custom op runtime entry failed, descriptor key[%s], op type[%s].",
             desc.impl_descriptor_key.c_str(), desc.op_type.c_str());
      return false;
    }
    descriptor_key_to_runtime_entry_.emplace(desc.impl_descriptor_key, std::move(runtime_entry));
    return true;
  }

  bool Unregister(const std::string &descriptor_key) {
    const std::lock_guard<std::mutex> map_lock(mutex_);
    const auto iter = descriptor_key_to_runtime_entry_.find(descriptor_key);
    if (iter == descriptor_key_to_runtime_entry_.cend()) {
      return false;
    }
    const auto &runtime_entry = iter->second;
    std::lock_guard<std::mutex> runtime_lock(runtime_entry->mutex);
    if (runtime_entry->active_adapter_count != 0U) {
      GELOGW("Python custom op runtime descriptor key[%s] is still in use.", descriptor_key.c_str());
      return false;
    }
    descriptor_key_to_runtime_entry_.erase(iter);
    return true;
  }

  bool Acquire(const PythonCustomOpAdapterDescriptor &desc, PythonCustomOpAdapterCallbacks &callbacks) {
    const std::lock_guard<std::mutex> map_lock(mutex_);
    const auto iter = descriptor_key_to_runtime_entry_.find(desc.impl_descriptor_key);
    if (iter == descriptor_key_to_runtime_entry_.cend()) {
      GELOGW("Acquire python custom op runtime failed because descriptor key[%s] is not registered.",
             desc.impl_descriptor_key.c_str());
      return false;
    }

    const auto &runtime_entry = iter->second;
    std::lock_guard<std::mutex> runtime_lock(runtime_entry->mutex);
    if ((runtime_entry->desc.op_type != desc.op_type) ||
        (runtime_entry->desc.impl_descriptor_key != desc.impl_descriptor_key) ||
        (runtime_entry->desc.capabilities != desc.capabilities)) {
      GELOGW("Acquire python custom op runtime descriptor mismatch, descriptor key[%s].",
             desc.impl_descriptor_key.c_str());
      return false;
    }
    ++runtime_entry->active_adapter_count;
    callbacks = runtime_entry->callbacks;
    return true;
  }

  void Release(const PythonCustomOpAdapterDescriptor &desc) {
    auto runtime_entry = Get(desc.impl_descriptor_key);
    if (runtime_entry == nullptr) {
      return;
    }

    std::lock_guard<std::mutex> lock(runtime_entry->mutex);
    if (runtime_entry->active_adapter_count == 0U) {
      return;
    }
    --runtime_entry->active_adapter_count;
  }

  void Clear() {
    std::lock_guard<std::mutex> lock(mutex_);
    descriptor_key_to_runtime_entry_.clear();
  }

 private:
  std::shared_ptr<PythonCustomOpImplRuntimeEntry> Get(const std::string &descriptor_key) {
    std::lock_guard<std::mutex> lock(mutex_);
    const auto iter = descriptor_key_to_runtime_entry_.find(descriptor_key);
    if (iter == descriptor_key_to_runtime_entry_.cend()) {
      return nullptr;
    }
    return iter->second;
  }

  std::mutex mutex_;
  std::map<std::string, std::shared_ptr<PythonCustomOpImplRuntimeEntry>> descriptor_key_to_runtime_entry_;
};

PythonCustomOpImplRuntimeRegistryImpl &GetPythonCustomOpImplRuntimeRegistryImpl() {
  static PythonCustomOpImplRuntimeRegistryImpl runtime_registry;
  return runtime_registry;
}

}  // namespace

PythonCustomOpImplHolder::PythonCustomOpImplHolder(const PythonCustomOpAdapterDescriptor &desc) : desc_(desc) {
  desc_.capabilities &= ~static_cast<CustomOpCapabilityMask>(CustomOpCapability::kShapeInfer);
  desc_.capabilities &= ~static_cast<CustomOpCapabilityMask>(CustomOpCapability::kInferMeta);
  if (!PythonCustomOpImplRuntimeRegistry::GetInstance().Acquire(desc_, callbacks_)) {
    return;
  }
  if (callbacks_.create_impl_holder == nullptr) {
    PythonCustomOpImplRuntimeRegistry::GetInstance().Release(desc_);
    return;
  }
  const PythonCustomOpAdapterDescriptorView descriptor_view = {
      {desc_.op_type.data(), desc_.op_type.size()},
      {desc_.impl_descriptor_key.data(), desc_.impl_descriptor_key.size()},
      desc_.capabilities};
  holder_ = callbacks_.create_impl_holder(&descriptor_view);
  if (holder_ == nullptr) {
    PythonCustomOpImplRuntimeRegistry::GetInstance().Release(desc_);
    return;
  }
  valid_ = true;
}

PythonCustomOpImplHolder::~PythonCustomOpImplHolder() {
  if (valid_) {
    if ((holder_ != nullptr) && (callbacks_.destroy_impl_holder != nullptr)) {
      callbacks_.destroy_impl_holder(holder_);
      holder_ = nullptr;
    }
    PythonCustomOpImplRuntimeRegistry::GetInstance().Release(desc_);
  }
}

bool PythonCustomOpImplHolder::IsValid() const {
  return valid_;
}

void *PythonCustomOpImplHolder::GetHolder() const {
  return holder_;
}

const PythonCustomOpAdapterCallbacks &PythonCustomOpImplHolder::GetCallbacks() const {
  return callbacks_;
}

const PythonCustomOpAdapterDescriptor &PythonCustomOpImplHolder::GetDescriptor() const {
  return desc_;
}

PythonCustomOpImplRuntimeRegistry &PythonCustomOpImplRuntimeRegistry::GetInstance() {
  static PythonCustomOpImplRuntimeRegistry runtime_registry;
  return runtime_registry;
}

bool PythonCustomOpImplRuntimeRegistry::Register(const PythonCustomOpAdapterDescriptor &desc,
                                                 const PythonCustomOpAdapterCallbacks &callbacks) {
  return GetPythonCustomOpImplRuntimeRegistryImpl().Register(desc, callbacks);
}

bool PythonCustomOpImplRuntimeRegistry::Unregister(const std::string &descriptor_key) {
  return GetPythonCustomOpImplRuntimeRegistryImpl().Unregister(descriptor_key);
}

bool PythonCustomOpImplRuntimeRegistry::Acquire(const PythonCustomOpAdapterDescriptor &desc,
                                                PythonCustomOpAdapterCallbacks &callbacks) {
  return GetPythonCustomOpImplRuntimeRegistryImpl().Acquire(desc, callbacks);
}

void PythonCustomOpImplRuntimeRegistry::Release(const PythonCustomOpAdapterDescriptor &desc) {
  GetPythonCustomOpImplRuntimeRegistryImpl().Release(desc);
}

void PythonCustomOpImplRuntimeRegistry::Clear() {
  GetPythonCustomOpImplRuntimeRegistryImpl().Clear();
}

void ClearPythonCustomOpRuntimeRegistry() {
  PythonCustomOpImplRuntimeRegistry::GetInstance().Clear();
}

PythonCustomOpAdapter::PythonCustomOpAdapter(PythonCustomOpAdapterDescriptor desc)
    : op_type_(desc.op_type),
      impl_descriptor_key_(desc.impl_descriptor_key),
      capabilities_(desc.capabilities),
      infer_meta_(desc.infer_meta) {
  if (!desc.impl_descriptor_key.empty()) {
    holder_ = std::make_unique<PythonCustomOpImplHolder>(desc);
  }
}

PythonCustomOpAdapter::~PythonCustomOpAdapter() = default;

bool PythonCustomOpAdapter::IsValid() const {
  if (capabilities_ == 0U) {
    return false;
  }
  if ((HasCustomOpCapability(capabilities_, CustomOpCapability::kShapeInfer) ||
       HasCustomOpCapability(capabilities_, CustomOpCapability::kInferMeta)) &&
      (infer_meta_ == nullptr)) {
    return false;
  }
  const auto infer_capabilities = static_cast<CustomOpCapabilityMask>(CustomOpCapability::kShapeInfer) |
                                  static_cast<CustomOpCapabilityMask>(CustomOpCapability::kInferMeta);
  const auto impl_capabilities = capabilities_ & ~infer_capabilities;
  if (impl_capabilities == 0U) {
    return impl_descriptor_key_.empty() && (holder_ == nullptr);
  }
  return (!impl_descriptor_key_.empty()) && (holder_ != nullptr) && holder_->IsValid();
}

bool PythonCustomOpAdapter::HasCapability(CustomOpCapability capability) const {
  return HasCustomOpCapability(capabilities_, capability);
}

graphStatus PythonCustomOpAdapter::Execute(gert::EagerOpExecutionContext *ctx) {
  if (!HasCapability(CustomOpCapability::kEagerExecute)) {
    return ReportUnsupported(CustomOpCapability::kEagerExecute, "Execute");
  }
  if ((holder_ == nullptr) || (!holder_->IsValid()) || (holder_->GetHolder() == nullptr) ||
      (holder_->GetCallbacks().execute == nullptr)) {
    GELOGE(GRAPH_FAILED, "Python custom op adapter is invalid, descriptor key[%s], op type[%s].",
           impl_descriptor_key_.c_str(), op_type_.c_str());
    return GRAPH_FAILED;
  }
  return holder_->GetCallbacks().execute(holder_->GetHolder(), ctx);
}

graphStatus PythonCustomOpAdapter::DeclareLaunchArgs(gert::AnnotatedArgsContext &ctx) {
  if (!HasCapability(CustomOpCapability::kAnnotatedArgs)) {
    return ReportUnsupported(CustomOpCapability::kAnnotatedArgs, "DeclareLaunchArgs");
  }
  if ((holder_ == nullptr) || (!holder_->IsValid()) || (holder_->GetHolder() == nullptr) ||
      (holder_->GetCallbacks().declare_launch_args == nullptr)) {
    GELOGE(GRAPH_FAILED, "Python custom op adapter is invalid, descriptor key[%s], op type[%s].",
           impl_descriptor_key_.c_str(), op_type_.c_str());
    return GRAPH_FAILED;
  }
  return holder_->GetCallbacks().declare_launch_args(holder_->GetHolder(), &ctx);
}

graphStatus PythonCustomOpAdapter::Compile(gert::OpCompileContext *ctx) {
  (void)ctx;
  return ReportUnsupported(CustomOpCapability::kCompilable, "Compile");
}

graphStatus PythonCustomOpAdapter::InferShape(gert::InferShapeContext *ctx) {
  if (!HasCapability(CustomOpCapability::kShapeInfer) || (infer_meta_ == nullptr)) {
    return ReportUnsupported(CustomOpCapability::kShapeInfer, "InferShape");
  }
  CustomOpInferMetaResult result;
  const auto ret = InferMeta(ctx, &result);
  if (ret != GRAPH_SUCCESS) {
    return ret;
  }
  for (size_t i = 0U; i < result.outputs.size(); ++i) {
    *ctx->GetOutputShape(i) = result.outputs[i].shape.GetStorageShape();
  }
  return GRAPH_SUCCESS;
}

graphStatus PythonCustomOpAdapter::InferDataType(gert::InferDataTypeContext *ctx) {
  (void)ctx;
  return ReportUnsupported(CustomOpCapability::kShapeInfer, "InferDataType");
}

graphStatus PythonCustomOpAdapter::InferMeta(gert::InferShapeContext *ctx, CustomOpInferMetaResult *result) {
  if ((ctx == nullptr) || (result == nullptr)) {
    GELOGE(GRAPH_FAILED, "Python custom op infer_meta context or result is null, op type[%s].", op_type_.c_str());
    return GRAPH_FAILED;
  }
  if (!HasCapability(CustomOpCapability::kInferMeta) || (infer_meta_ == nullptr)) {
    GELOGE(GRAPH_FAILED, "Python custom op infer_meta capability or callback is not registered, op type[%s].",
           op_type_.c_str());
    return GRAPH_FAILED;
  }
  return InvokePythonCustomOpInferMeta(infer_meta_, op_type_, ctx, result);
}

graphStatus PythonCustomOpAdapter::Serialize(std::vector<uint8_t> &buffer) {
  buffer.clear();
  return ReportUnsupported(CustomOpCapability::kPortable, "Serialize");
}

graphStatus PythonCustomOpAdapter::Deserialize(const std::vector<uint8_t> &buffer) {
  (void)buffer;
  return ReportUnsupported(CustomOpCapability::kPortable, "Deserialize");
}

graphStatus PythonCustomOpAdapter::UpdateHostArgs(gert::UpdateArgsContext *ctx) {
  (void)ctx;
  return ReportUnsupported(CustomOpCapability::kArgsUpdater, "UpdateHostArgs");
}

graphStatus PythonCustomOpAdapter::ReportUnsupported(CustomOpCapability capability, const char *method_name) const {
  GELOGE(GRAPH_FAILED, "Python custom op[%s] does not support %s capability[%u].", op_type_.c_str(), method_name,
         static_cast<uint32_t>(capability));
  return GRAPH_FAILED;
}

}  // namespace custom_op
}  // namespace ge
