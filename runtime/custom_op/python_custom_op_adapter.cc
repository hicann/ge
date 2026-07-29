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

#include "common/checker.h"
#include "framework/common/debug/ge_log.h"
#include "graph/operator_factory.h"
#include "graph_metadef/graph/debug/ge_util.h"
#include "runtime/custom_op/python_custom_op_ir_meta.h"

namespace ge {
namespace custom_op {
class PythonCustomOpIrMetaViewStorage {
 public:
  graphStatus Initialize(const std::string &op_type) {
    GE_ASSERT_GRAPH_SUCCESS(CollectCustomOpIrMeta(op_type, ir_meta_));

    input_views_.reserve(ir_meta_.inputs.size());
    for (const auto &input : ir_meta_.inputs) {
      uint32_t kind = 0U;
      GE_ASSERT_GRAPH_SUCCESS(ConvertInputKind(input.kind, kind));
      input_views_.emplace_back(PythonCustomOpIrInputView{input.name.c_str(), kind});
    }

    attr_views_.reserve(ir_meta_.attrs.size());
    for (const auto &attr : ir_meta_.attrs) {
      attr_views_.emplace_back(PythonCustomOpIrAttrView{attr.name.c_str(), attr.type.c_str()});
    }

    output_views_.reserve(ir_meta_.outputs.size());
    for (const auto &output : ir_meta_.outputs) {
      uint32_t kind = 0U;
      GE_ASSERT_GRAPH_SUCCESS(ConvertOutputKind(output.kind, kind));
      output_views_.emplace_back(PythonCustomOpIrOutputView{output.name.c_str(), kind});
    }

    view_ = PythonCustomOpIrMetaView{
        ir_meta_.op_type.c_str(), input_views_.empty() ? nullptr : input_views_.data(),
        input_views_.size(),      attr_views_.empty() ? nullptr : attr_views_.data(),
        attr_views_.size(),       output_views_.empty() ? nullptr : output_views_.data(),
        output_views_.size(),
    };
    return GRAPH_SUCCESS;
  }

  const PythonCustomOpIrMetaView &GetView() const {
    return view_;
  }

 private:
  static graphStatus ConvertInputKind(const ge::IrInputType kind, uint32_t &value) {
    switch (kind) {
      case ge::kIrInputRequired:
        value = 0U;
        return GRAPH_SUCCESS;
      case ge::kIrInputOptional:
        value = 1U;
        return GRAPH_SUCCESS;
      case ge::kIrInputDynamic:
        value = 2U;
        return GRAPH_SUCCESS;
      default:
        GE_ASSERT_TRUE(false, "Unsupported custom op IR input kind[%u].", static_cast<uint32_t>(kind));
    }
  }

  static graphStatus ConvertOutputKind(const ge::IrOutputType kind, uint32_t &value) {
    switch (kind) {
      case ge::kIrOutputRequired:
        value = 0U;
        return GRAPH_SUCCESS;
      case ge::kIrOutputDynamic:
        value = 1U;
        return GRAPH_SUCCESS;
      default:
        GE_ASSERT_TRUE(false, "Unsupported custom op IR output kind[%u].", static_cast<uint32_t>(kind));
    }
  }

  CustomOpIrMeta ir_meta_;
  std::vector<PythonCustomOpIrInputView> input_views_;
  std::vector<PythonCustomOpIrAttrView> attr_views_;
  std::vector<PythonCustomOpIrOutputView> output_views_;
  PythonCustomOpIrMetaView view_{};
};

namespace {
std::unique_ptr<PythonCustomOpIrMetaViewStorage> CreateIrMetaViewStorage(const std::string &op_type) {
  if (!OperatorFactory::IsExistOp(op_type.c_str())) {
    return nullptr;
  }
  auto storage = std::unique_ptr<PythonCustomOpIrMetaViewStorage>(new (std::nothrow) PythonCustomOpIrMetaViewStorage());
  if (storage == nullptr) {
    GELOGE(GRAPH_FAILED, "Create custom op IR meta view storage failed, op type[%s].", op_type.c_str());
    return nullptr;
  }
  const auto ret = storage->Initialize(op_type);
  if (ret != GRAPH_SUCCESS) {
    GELOGE(ret, "Initialize custom op IR meta view storage failed, op type[%s].", op_type.c_str());
    return nullptr;
  }
  return storage;
}

struct PythonCustomOpRuntimeEntry {
  explicit PythonCustomOpRuntimeEntry(PythonCustomOpDescriptor d, PythonCustomOpCallbacks cb)
      : desc(std::move(d)), callbacks(cb) {}

  PythonCustomOpDescriptor desc;
  PythonCustomOpCallbacks callbacks;
  // A descriptor_key maps to one shared runtime entry. Multiple adapters may
  // reference it, while each adapter owns one Python custom op instance.
  // active_adapter_count prevents unregister while callbacks are still in use.
  size_t active_adapter_count{0U};
  std::mutex mutex;
};

class PythonCustomOpRuntimeRegistryImpl {
 public:
  bool Register(const PythonCustomOpDescriptor &desc, const PythonCustomOpCallbacks &callbacks) {
    if (desc.descriptor_key.empty() || desc.op_type.empty() || (!callbacks.IsValid(desc.capabilities))) {
      GELOGW("Register python custom op runtime failed, descriptor key[%s], op type[%s].", desc.descriptor_key.c_str(),
             desc.op_type.c_str());
      return false;
    }

    std::lock_guard<std::mutex> lock(mutex_);
    if (descriptor_key_to_runtime_entry_.find(desc.descriptor_key) != descriptor_key_to_runtime_entry_.cend()) {
      GELOGW("Python custom op runtime descriptor key[%s] has already registered.", desc.descriptor_key.c_str());
      return false;
    }
    auto runtime_entry = ComGraphMakeShared<PythonCustomOpRuntimeEntry>(desc, callbacks);
    if (runtime_entry == nullptr) {
      GELOGE(GRAPH_FAILED, "Create python custom op runtime entry failed, descriptor key[%s], op type[%s].",
             desc.descriptor_key.c_str(), desc.op_type.c_str());
      return false;
    }
    descriptor_key_to_runtime_entry_.emplace(desc.descriptor_key, std::move(runtime_entry));
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

  bool Acquire(const PythonCustomOpDescriptor &desc, PythonCustomOpCallbacks &callbacks) {
    auto runtime_entry = Get(desc.descriptor_key);
    if (runtime_entry == nullptr) {
      GELOGW("Acquire python custom op runtime failed because descriptor key[%s] is not registered.",
             desc.descriptor_key.c_str());
      return false;
    }

    std::lock_guard<std::mutex> lock(runtime_entry->mutex);
    ++runtime_entry->active_adapter_count;
    callbacks = runtime_entry->callbacks;
    return true;
  }

  void Release(const PythonCustomOpDescriptor &desc) {
    auto runtime_entry = Get(desc.descriptor_key);
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
  std::shared_ptr<PythonCustomOpRuntimeEntry> Get(const std::string &descriptor_key) {
    std::lock_guard<std::mutex> lock(mutex_);
    const auto iter = descriptor_key_to_runtime_entry_.find(descriptor_key);
    if (iter == descriptor_key_to_runtime_entry_.cend()) {
      return nullptr;
    }
    return iter->second;
  }

  std::mutex mutex_;
  std::map<std::string, std::shared_ptr<PythonCustomOpRuntimeEntry>> descriptor_key_to_runtime_entry_;
};

PythonCustomOpRuntimeRegistryImpl &GetPythonCustomOpRuntimeRegistryImpl() {
  static PythonCustomOpRuntimeRegistryImpl runtime_registry;
  return runtime_registry;
}
}  // namespace

PythonCustomOpHolder::PythonCustomOpHolder(const PythonCustomOpDescriptor &desc) : desc_(desc) {
  if (!PythonCustomOpRuntimeRegistry::GetInstance().Acquire(desc_, callbacks_)) {
    return;
  }
  if (callbacks_.create == nullptr) {
    PythonCustomOpRuntimeRegistry::GetInstance().Release(desc_);
    return;
  }
  holder_ = callbacks_.create(&desc_);
  if (holder_ == nullptr) {
    PythonCustomOpRuntimeRegistry::GetInstance().Release(desc_);
    return;
  }
  valid_ = true;
}

PythonCustomOpHolder::~PythonCustomOpHolder() {
  if (valid_) {
    if ((holder_ != nullptr) && (callbacks_.destroy != nullptr)) {
      callbacks_.destroy(holder_);
      holder_ = nullptr;
    }
    PythonCustomOpRuntimeRegistry::GetInstance().Release(desc_);
  }
}

bool PythonCustomOpHolder::IsValid() const {
  return valid_;
}

void *PythonCustomOpHolder::GetHolder() const {
  return holder_;
}

const PythonCustomOpCallbacks &PythonCustomOpHolder::GetCallbacks() const {
  return callbacks_;
}

const PythonCustomOpDescriptor &PythonCustomOpHolder::GetDescriptor() const {
  return desc_;
}

PythonCustomOpRuntimeRegistry &PythonCustomOpRuntimeRegistry::GetInstance() {
  static PythonCustomOpRuntimeRegistry runtime_registry;
  return runtime_registry;
}

bool PythonCustomOpRuntimeRegistry::Register(const PythonCustomOpDescriptor &desc,
                                             const PythonCustomOpCallbacks &callbacks) {
  return GetPythonCustomOpRuntimeRegistryImpl().Register(desc, callbacks);
}

bool PythonCustomOpRuntimeRegistry::Unregister(const std::string &descriptor_key) {
  return GetPythonCustomOpRuntimeRegistryImpl().Unregister(descriptor_key);
}

bool PythonCustomOpRuntimeRegistry::Acquire(const PythonCustomOpDescriptor &desc, PythonCustomOpCallbacks &callbacks) {
  return GetPythonCustomOpRuntimeRegistryImpl().Acquire(desc, callbacks);
}

void PythonCustomOpRuntimeRegistry::Release(const PythonCustomOpDescriptor &desc) {
  GetPythonCustomOpRuntimeRegistryImpl().Release(desc);
}

void PythonCustomOpRuntimeRegistry::Clear() {
  GetPythonCustomOpRuntimeRegistryImpl().Clear();
}

PythonCustomOpAdapter::PythonCustomOpAdapter(PythonCustomOpDescriptor desc)
    : desc_(std::move(desc)),
      ir_meta_(CreateIrMetaViewStorage(desc_.op_type)),
      holder_(new (std::nothrow) PythonCustomOpHolder(desc_)) {}

PythonCustomOpAdapter::~PythonCustomOpAdapter() = default;

bool PythonCustomOpAdapter::IsValid() const {
  return (holder_ != nullptr) && holder_->IsValid();
}

bool PythonCustomOpAdapter::HasCapability(CustomOpCapability capability) const {
  return HasCustomOpCapability(desc_.capabilities, capability);
}

graphStatus PythonCustomOpAdapter::Execute(gert::EagerOpExecutionContext *ctx) {
  if (!HasCapability(CustomOpCapability::kEagerExecute)) {
    return ReportUnsupported(CustomOpCapability::kEagerExecute, "Execute");
  }
  if ((holder_ == nullptr) || (!holder_->IsValid()) || (holder_->GetHolder() == nullptr) ||
      (holder_->GetCallbacks().execute == nullptr)) {
    GELOGE(GRAPH_FAILED, "Python custom op adapter is invalid, descriptor key[%s], op type[%s].",
           desc_.descriptor_key.c_str(), desc_.op_type.c_str());
    return GRAPH_FAILED;
  }
  const auto *ir_meta = (ir_meta_ == nullptr) ? nullptr : &ir_meta_->GetView();
  return holder_->GetCallbacks().execute(holder_->GetHolder(), ctx, ir_meta);
}

graphStatus PythonCustomOpAdapter::Compile(gert::OpCompileContext *ctx) {
  (void)ctx;
  return ReportUnsupported(CustomOpCapability::kCompilable, "Compile");
}

graphStatus PythonCustomOpAdapter::InferShape(gert::InferShapeContext *ctx) {
  (void)ctx;
  return ReportUnsupported(CustomOpCapability::kShapeInfer, "InferShape");
}

graphStatus PythonCustomOpAdapter::InferDataType(gert::InferDataTypeContext *ctx) {
  (void)ctx;
  return ReportUnsupported(CustomOpCapability::kShapeInfer, "InferDataType");
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
  GELOGE(GRAPH_FAILED, "Python custom op[%s] does not support %s capability[%u].", desc_.op_type.c_str(), method_name,
         static_cast<uint32_t>(capability));
  return GRAPH_FAILED;
}

void ClearPythonCustomOpRuntimeRegistry() {
  PythonCustomOpRuntimeRegistry::GetInstance().Clear();
}
}  // namespace custom_op
}  // namespace ge
