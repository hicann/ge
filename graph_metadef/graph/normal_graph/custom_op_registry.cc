/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph/custom_op.h"
#include "graph/custom_op_registry.h"

#include <limits>
#include <string>
#include <typeinfo>
#include <utility>

#include "debug/ge_log.h"
#include "framework/common/framework_types_internal.h"
#include "graph/custom_op/cast.h"

namespace ge {
namespace {
const char *OpBackendToString(const OpBackend backend) {
  switch (backend) {
    case OpBackend::kDevice:
      return "device";
    case OpBackend::kHostCPU:
      return "host_cpu";
    default:
      return "unknown";
  }
}

const char *OpEngineToString(const OpEngine engine) {
  switch (engine) {
    case OpEngine::kCustom:
      return "CUSTOM";
    case OpEngine::kAiCore:
      return "AICORE";
    case OpEngine::kFftsPlus:
      return "FFTS+";
    case OpEngine::kVectorCore:
      return "VECTORCORE";
    case OpEngine::kDsa:
      return "DSA";
    case OpEngine::kRts:
      return "RTS";
    case OpEngine::kRtsFftsPlus:
      return "RTS_FFTS_PLUS";
    case OpEngine::kHccl:
      return "HCCL";
    case OpEngine::kAiCpuFftsPlus:
      return "AICPU_FFTS_PLUS";
    case OpEngine::kAiCpuAscendFftsPlus:
      return "AICPU_ASCEND_FFTS_PLUS";
    case OpEngine::kAiCpuAscend:
      return "AICPU_ASCEND";
    case OpEngine::kAiCpu:
      return "AICPU";
    case OpEngine::kDvpp:
      return "DVPP";
    case OpEngine::kGeLocal:
      return "GE_LOCAL";
    case OpEngine::kHostCpu:
      return "HOST_CPU";
    default:
      return "unknown";
  }
}

const char *OpPriorityToString(const OpRegistrationPriority priority) {
  switch (priority) {
    case OpRegistrationPriority::kTop:
      return "top";
    case OpRegistrationPriority::kBottom:
      return "bottom";
    default:
      return "unknown";
  }
}

bool IsValidOpBackend(const OpBackend backend) {
  return (backend == OpBackend::kDevice) || (backend == OpBackend::kHostCPU);
}

bool IsValidOpPriority(const OpRegistrationPriority priority) {
  return (priority == OpRegistrationPriority::kTop) || (priority == OpRegistrationPriority::kBottom);
}

bool IsValidOpEngine(const OpEngine engine) {
  return (engine >= OpEngine::kCustom) && (engine <= OpEngine::kHostCpu);
}

graphStatus ValidateCustomOpParams(const AscendString &op_type, const OpBackend backend,
                                   const OpRegistrationPriority priority, const OpEngine engine) {
  if (!IsValidOpEngine(engine)) {
    GELOGE(GRAPH_PARAM_INVALID, "[Check][Param] custom op engine for %s:%s:%s is invalid:%u.", op_type.GetString(),
           OpBackendToString(backend), OpPriorityToString(priority), static_cast<uint32_t>(engine));
    return GRAPH_PARAM_INVALID;
  }
  if (!IsValidOpBackend(backend)) {
    GELOGE(GRAPH_PARAM_INVALID, "[Check][Param] custom op backend for %s:%s:%s is invalid:%u.", op_type.GetString(),
           OpEngineToString(engine), OpPriorityToString(priority), static_cast<uint32_t>(backend));
    return GRAPH_PARAM_INVALID;
  }
  if (!IsValidOpPriority(priority)) {
    GELOGE(GRAPH_PARAM_INVALID, "[Check][Param] custom op priority for %s:%s:%s is invalid:%u.", op_type.GetString(),
           OpEngineToString(engine), OpBackendToString(backend), static_cast<uint32_t>(priority));
    return GRAPH_PARAM_INVALID;
  }
  return GRAPH_SUCCESS;
}

bool IsCommonCapability(const CustomOpCapability capability) {
  return (capability == CustomOpCapability::kShapeInfer) || (capability == CustomOpCapability::kInferMeta) ||
         (capability == CustomOpCapability::kPortable);
}

bool HasCommonCapability(const BaseCustomOp *op, const CustomOpCapability capability) {
  switch (capability) {
    case CustomOpCapability::kShapeInfer:
      return CustomOpCast<ShapeInferOp>(op) != nullptr;
    case CustomOpCapability::kInferMeta:
      return CustomOpCast<CustomOpInferMetaProvider>(op) != nullptr;
    case CustomOpCapability::kPortable:
      return CustomOpCast<PortableOp>(op) != nullptr;
    default:
      return false;
  }
}

struct ParsedCustomKernelItem {
  std::string op_type;
  const uint8_t *kernel_bin;
  size_t bin_len;
  size_t entry_size;
};

graphStatus ParseCustomKernelItem(const uint8_t *data, const size_t len, const size_t offset,
                                  ParsedCustomKernelItem &item) {
  const size_t header_size = sizeof(CustomKernelItemHeader);
  if (header_size > (len - offset)) {
    GELOGE(GRAPH_FAILED, "[CUSTOM OP] Insufficient data for CustomKernelItemHeader at offset %zu", offset);
    return GRAPH_FAILED;
  }

  const auto *header = reinterpret_cast<const CustomKernelItemHeader *>(data + offset);
  if (header->magic != kCustomKernelItemMagic) {
    GELOGE(GRAPH_FAILED, "[CUSTOM OP] Invalid magic in CustomKernelItemHeader: 0x%X, expected 0x%X", header->magic,
           kCustomKernelItemMagic);
    return GRAPH_FAILED;
  }

  const size_t name_len = static_cast<size_t>(header->name_len);
  const size_t bin_len = static_cast<size_t>(header->bin_len);
  if ((name_len > (std::numeric_limits<size_t>::max() - header_size)) ||
      (bin_len > (std::numeric_limits<size_t>::max() - header_size - name_len))) {
    GELOGE(GRAPH_FAILED, "[CUSTOM OP] Invalid kernel entry size at offset %zu, name len %zu, bin len %zu", offset,
           name_len, bin_len);
    return GRAPH_FAILED;
  }

  const size_t entry_size = header_size + name_len + bin_len;
  if (entry_size > (len - offset)) {
    GELOGE(GRAPH_FAILED, "[CUSTOM OP] Insufficient data for kernel entry at offset %zu, need %zu bytes", offset,
           entry_size);
    return GRAPH_FAILED;
  }

  const char *op_type_ptr = reinterpret_cast<const char *>(data + offset + header_size);
  item.op_type = std::string(op_type_ptr, name_len);
  item.kernel_bin = data + offset + header_size + name_len;
  item.bin_len = bin_len;
  item.entry_size = entry_size;
  return GRAPH_SUCCESS;
}

graphStatus DeserializeCustomKernelItem(CustomOpRegistry &registry, const ParsedCustomKernelItem &item) {
  auto op = registry.GetCustomOpCommonCapability(AscendString(item.op_type.c_str()), CustomOpCapability::kPortable);
  if (op == nullptr) {
    GELOGE(GRAPH_FAILED, "[CUSTOM OP] Custom op '%s' is not PortableOp or not found in registry", item.op_type.c_str());
    return GRAPH_FAILED;
  }

  auto *serializable_op = CustomOpCast<PortableOp>(op);
  if (serializable_op == nullptr) {
    GELOGE(GRAPH_FAILED,
           "[CUSTOM OP] Custom op '%s' is not PortableOp, type mismatch or version inconsistency detected",
           item.op_type.c_str());
    return GRAPH_FAILED;
  }

  const std::vector<uint8_t> kernel_bin_buffer(item.kernel_bin, item.kernel_bin + item.bin_len);
  const auto ret = serializable_op->Deserialize(kernel_bin_buffer);
  if (ret != GRAPH_SUCCESS) {
    GELOGE(ret, "[CUSTOM OP] Failed to deserialize custom op '%s'", item.op_type.c_str());
    return ret;
  }

  GELOGI("[CUSTOM OP] Successfully deserialized custom op '%s'", item.op_type.c_str());
  return GRAPH_SUCCESS;
}

template <typename EngineInstanceMap>
void MoveCustomOpInstancesForRemoval(EngineInstanceMap &engine_instances,
                                     std::vector<std::shared_ptr<BaseCustomOp>> &removed_custom_ops) {
  for (auto &engine_custom_op : engine_instances) {
    for (auto &priority_custom_op : engine_custom_op.second) {
      for (auto &backend_custom_op : priority_custom_op.second) {
        removed_custom_ops.emplace_back(std::move(backend_custom_op.second));
      }
    }
  }
}
}  // namespace

struct CustomOpRegistry::Impl {
  std::vector<OpProtoClaimRecord> proto_claims;
};

CustomOpRegistry::~CustomOpRegistry() {
  try {
    std::vector<OpProtoClaimRecord> claims;
    {
      const std::lock_guard<std::mutex> lock(mu_);
      if (impl_ != nullptr) {
        claims = impl_->proto_claims;
      }
    }
    if (!claims.empty()) {
      OpProtoLedger::ReleaseClaims(claims);  // 按事务日志精确扣减，替代盲删（锁外执行，避免与账本锁嵌套）
    }
  } catch (const std::exception &e) {
    GELOGW("[CUSTOM OP] Exception in CustomOpRegistry destructor: %s", e.what());
  } catch (...) {
    GELOGW("[CUSTOM OP] Unknown exception in CustomOpRegistry destructor.");
  }
}

graphStatus CustomOpRegistry::RegisterCreator(const AscendString &op_type, OpBackend backend,
                                              const BaseOpCreator &creator) {
  return RegisterCreator(op_type, backend, OpRegistrationPriority::kTop, OpEngine::kCustom, creator);
}

graphStatus CustomOpRegistry::RegisterCreator(const AscendString &op_type, OpBackend backend,
                                              OpRegistrationPriority priority, OpEngine engine,
                                              const BaseOpCreator &creator) {
  const std::lock_guard<std::mutex> lock(mu_);
  const auto validate_ret = ValidateCustomOpParams(op_type, backend, priority, engine);
  if (validate_ret != GRAPH_SUCCESS) {
    return validate_ret;
  }
  if (creator == nullptr) {
    GELOGE(GRAPH_PARAM_INVALID, "[Check][Param] custom op creator for %s:%s:%s:%s is null.", op_type.GetString(),
           OpEngineToString(engine), OpBackendToString(backend), OpPriorityToString(priority));
    return GRAPH_PARAM_INVALID;
  }
  auto creator_iter = creators_.find(op_type);
  if (creator_iter != creators_.cend()) {
    const auto engine_iter = creator_iter->second.find(engine);
    if (engine_iter != creator_iter->second.cend()) {
      const auto priority_iter = engine_iter->second.find(priority);
      if (priority_iter != engine_iter->second.cend()) {
        const auto backend_iter = priority_iter->second.find(backend);
        if (backend_iter != priority_iter->second.cend()) {
          GELOGW("[CUSTOM OP] custom op creator for %s:%s:%s:%s already exist.", op_type.GetString(),
                 OpEngineToString(engine), OpBackendToString(backend), OpPriorityToString(priority));
          return GRAPH_FAILED;
        }
      }
    }
  }
  if (creator_iter == creators_.cend()) {
    creator_iter = creators_.emplace(op_type, EngineCreatorMap()).first;
  }
  auto engine_iter = creator_iter->second.find(engine);
  if (engine_iter == creator_iter->second.cend()) {
    engine_iter = creator_iter->second.emplace(engine, PriorityCreatorMap()).first;
  }
  auto priority_iter = engine_iter->second.find(priority);
  if (priority_iter == engine_iter->second.cend()) {
    priority_iter = engine_iter->second.emplace(priority, BackendCreatorMap()).first;
  }
  (void)priority_iter->second.emplace(backend, creator);
  GELOGI("[CUSTOM OP] register custom operator creator for %s:%s:%s:%s.", op_type.GetString(), OpEngineToString(engine),
         OpBackendToString(backend), OpPriorityToString(priority));
  return GRAPH_SUCCESS;
}

void CustomOpRegistry::AddSoHandles(const std::vector<CustomOpSoHandlePtr> &so_handles) {
  const std::lock_guard<std::mutex> lock(mu_);
  so_handles_.insert(so_handles_.end(), so_handles.begin(), so_handles.end());
}

BaseCustomOp *CustomOpRegistry::CreateOrGetCustomOp(const AscendString &op_type, OpBackend backend) {
  return CreateOrGetCustomOp(op_type, backend, OpRegistrationPriority::kTop, OpEngine::kCustom);
}

BaseCustomOp *CustomOpRegistry::CreateOrGetCustomOp(const AscendString &op_type, OpBackend backend,
                                                    OpRegistrationPriority priority, OpEngine engine) {
  const std::lock_guard<std::mutex> lock(mu_);
  return CreateOrGetCustomOpLocked(op_type, backend, priority, engine);
}

BaseCustomOp *CustomOpRegistry::CacheCustomOpLocked(const AscendString &op_type, OpBackend backend,
                                                    OpRegistrationPriority priority, OpEngine engine,
                                                    std::unique_ptr<BaseCustomOp> base_custom_op) {
  auto &backend_custom_ops = custom_ops_[op_type][engine][priority];
  for (const auto &backend_custom_op : backend_custom_ops) {
    if ((backend_custom_op.second != nullptr) && (typeid(*backend_custom_op.second) == typeid(*base_custom_op))) {
      auto [ops_it, success] = backend_custom_ops.emplace(backend, backend_custom_op.second);
      if (success) {
        GELOGI("[CUSTOM OP] share custom op instance for %s:%s:%s:%s with existing %s backend.", op_type.GetString(),
               OpEngineToString(engine), OpBackendToString(backend), OpPriorityToString(priority),
               OpBackendToString(backend_custom_op.first));
        return ops_it->second.get();
      }
      GELOGW("[CUSTOM OP] custom op instance found for %s:%s:%s:%s.", op_type.GetString(), OpEngineToString(engine),
             OpBackendToString(backend), OpPriorityToString(priority));
      return ops_it->second.get();
    }
  }
  auto [ops_it, success] = backend_custom_ops.emplace(backend, std::move(base_custom_op));
  if (success) {
    return ops_it->second.get();
  }
  GELOGW("[CUSTOM OP] custom op instance found for %s:%s:%s:%s.", op_type.GetString(), OpEngineToString(engine),
         OpBackendToString(backend), OpPriorityToString(priority));
  return ops_it->second.get();
}

BaseCustomOp *CustomOpRegistry::CreateOrGetCustomOpLocked(const AscendString &op_type, OpBackend backend,
                                                          OpRegistrationPriority priority, OpEngine engine) {
  if (ValidateCustomOpParams(op_type, backend, priority, engine) != GRAPH_SUCCESS) {
    return nullptr;
  }
  if (const auto it = custom_ops_.find(op_type); it != custom_ops_.cend()) {
    if (const auto engine_it = it->second.find(engine); engine_it != it->second.cend()) {
      if (const auto priority_it = engine_it->second.find(priority); priority_it != engine_it->second.cend()) {
        if (const auto backend_it = priority_it->second.find(backend); backend_it != priority_it->second.cend()) {
          GELOGD("[CUSTOM OP] custom_op %s:%s:%s:%s already created.", op_type.GetString(), OpEngineToString(engine),
                 OpBackendToString(backend), OpPriorityToString(priority));
          return backend_it->second.get();
        }
      }
    }
  }
  if (const auto op_creator_it = creators_.find(op_type); op_creator_it != creators_.cend()) {
    const auto engine_creator_it = op_creator_it->second.find(engine);
    if (engine_creator_it == op_creator_it->second.cend()) {
      GELOGW("[CUSTOM OP] get custom operator creator failed for %s:%s:%s:%s.", op_type.GetString(),
             OpEngineToString(engine), OpBackendToString(backend), OpPriorityToString(priority));
      return nullptr;
    }
    const auto priority_creator_it = engine_creator_it->second.find(priority);
    if (priority_creator_it == engine_creator_it->second.cend()) {
      GELOGW("[CUSTOM OP] get custom operator creator failed for %s:%s:%s:%s.", op_type.GetString(),
             OpEngineToString(engine), OpBackendToString(backend), OpPriorityToString(priority));
      return nullptr;
    }
    const auto backend_creator_it = priority_creator_it->second.find(backend);
    if (backend_creator_it == priority_creator_it->second.cend()) {
      GELOGW("[CUSTOM OP] get custom operator creator failed for %s:%s:%s:%s.", op_type.GetString(),
             OpEngineToString(engine), OpBackendToString(backend), OpPriorityToString(priority));
      return nullptr;
    }
    if (backend_creator_it->second == nullptr) {
      GELOGE(GRAPH_PARAM_INVALID, "[Check][Param] custom op creator for %s:%s:%s:%s is null.", op_type.GetString(),
             OpEngineToString(engine), OpBackendToString(backend), OpPriorityToString(priority));
      return nullptr;
    }
    auto base_custom_op = backend_creator_it->second();
    if (base_custom_op == nullptr) {
      GELOGE(GRAPH_FAILED, "[CUSTOM OP] custom op creator returned null for %s:%s:%s:%s.", op_type.GetString(),
             OpEngineToString(engine), OpBackendToString(backend), OpPriorityToString(priority));
      return nullptr;
    }
    return CacheCustomOpLocked(op_type, backend, priority, engine, std::move(base_custom_op));
  }
  GELOGW("[CUSTOM OP] get custom operator creator failed for %s:%s:%s:%s.", op_type.GetString(),
         OpEngineToString(engine), OpBackendToString(backend), OpPriorityToString(priority));
  return nullptr;
}

BaseCustomOp *CustomOpRegistry::GetCustomOpCommonCapability(const AscendString &op_type,
                                                            CustomOpCapability capability) {
  return GetCustomOpCommonCapability(op_type, capability, OpRegistrationPriority::kTop, OpEngine::kCustom);
}

BaseCustomOp *CustomOpRegistry::GetCustomOpCommonCapability(const AscendString &op_type, CustomOpCapability capability,
                                                            OpRegistrationPriority priority, OpEngine engine) {
  if (!IsCommonCapability(capability)) {
    GELOGE(GRAPH_FAILED, "[CUSTOM OP] capability %u for %s is not a common capability.",
           static_cast<uint32_t>(capability), op_type.GetString());
    return nullptr;
  }
  const std::lock_guard<std::mutex> lock(mu_);
  const auto op_creator_it = creators_.find(op_type);
  if (op_creator_it == creators_.cend()) {
    GELOGW("[CUSTOM OP] get custom operator creator failed for %s.", op_type.GetString());
    return nullptr;
  }
  const auto engine_creator_it = op_creator_it->second.find(engine);
  if (engine_creator_it == op_creator_it->second.cend()) {
    GELOGW("[CUSTOM OP] get custom operator creator failed for %s:%s.", op_type.GetString(), OpEngineToString(engine));
    return nullptr;
  }
  const auto priority_creator_it = engine_creator_it->second.find(priority);
  if (priority_creator_it == engine_creator_it->second.cend()) {
    GELOGW("[CUSTOM OP] get custom operator creator failed for %s:%s:%s.", op_type.GetString(),
           OpEngineToString(engine), OpPriorityToString(priority));
    return nullptr;
  }

  BaseCustomOp *matched_op = nullptr;
  const OpBackend *matched_backend = nullptr;
  for (const auto &backend_creator : priority_creator_it->second) {
    const auto backend = backend_creator.first;
    auto *const custom_op = CreateOrGetCustomOpLocked(op_type, backend, priority, engine);
    if ((custom_op == nullptr) || (!HasCommonCapability(custom_op, capability))) {
      continue;
    }
    if (matched_op == nullptr) {
      matched_op = custom_op;
      matched_backend = &backend_creator.first;
      continue;
    }
    if (matched_op != custom_op) {
      GELOGE(GRAPH_FAILED,
             "[CUSTOM OP] capability %u for %s must have a unique provider, but found in both %s and %s backend.",
             static_cast<uint32_t>(capability), op_type.GetString(), OpBackendToString(*matched_backend),
             OpBackendToString(backend));
      return nullptr;
    }
  }
  if (matched_op == nullptr) {
    GELOGI("[CUSTOM OP] capability %u for %s is not implemented.", static_cast<uint32_t>(capability),
           op_type.GetString());
  }
  return matched_op;
}

void CustomOpRegistry::RemoveCustomOps(const std::vector<AscendString> &op_types) {
  std::vector<std::shared_ptr<BaseCustomOp>> removed_custom_ops;
  {
    const std::lock_guard<std::mutex> lock(mu_);
    for (const auto &op_type : op_types) {
      const auto custom_op_iter = custom_ops_.find(op_type);
      if (custom_op_iter != custom_ops_.cend()) {
        MoveCustomOpInstancesForRemoval(custom_op_iter->second, removed_custom_ops);
        (void)custom_ops_.erase(custom_op_iter);
      }

      const auto creator_iter = creators_.find(op_type);
      if (creator_iter != creators_.cend()) {
        (void)creators_.erase(creator_iter);
      }
    }
  }

  removed_custom_ops.clear();
}

ArgsRefreshStrategy CustomOpRegistry::GetArgsRefreshStrategy(const AscendString &op_type) {
  const auto *custom_op = CreateOrGetCustomOp(op_type, OpBackend::kDevice);
  if (custom_op == nullptr) {
    return ArgsRefreshStrategy::kNone;
  }
  if (CustomOpCast<ArgsUpdater>(custom_op) != nullptr) {
    return ArgsRefreshStrategy::kUpdateCallback;
  }
  if (CustomOpCast<AnnotatedArgsOp>(custom_op) != nullptr) {
    return ArgsRefreshStrategy::kAnnotatedArgs;
  }
  return ArgsRefreshStrategy::kNone;
}

bool CustomOpRegistry::IsAddressRefreshable(const AscendString &op_type) {
  return GetArgsRefreshStrategy(op_type) != ArgsRefreshStrategy::kNone;
}

bool CustomOpRegistry::HasCreator(const AscendString &op_type) const {
  const std::lock_guard<std::mutex> lock(mu_);
  const auto creator_iter = creators_.find(op_type);
  if (creator_iter == creators_.cend()) {
    return false;
  }
  const auto engine_iter = creator_iter->second.find(OpEngine::kCustom);
  if (engine_iter == creator_iter->second.cend()) {
    return false;
  }
  const auto priority_iter = engine_iter->second.find(OpRegistrationPriority::kTop);
  if (priority_iter == engine_iter->second.cend()) {
    return false;
  }
  return !priority_iter->second.empty();
}

bool CustomOpRegistry::HasCreator(const AscendString &op_type, OpBackend backend) const {
  return HasCreator(op_type, backend, OpRegistrationPriority::kTop, OpEngine::kCustom);
}

bool CustomOpRegistry::HasCreator(const AscendString &op_type, OpBackend backend, OpRegistrationPriority priority,
                                  OpEngine engine) const {
  const std::lock_guard<std::mutex> lock(mu_);
  const auto it = creators_.find(op_type);
  if (it == creators_.cend()) {
    return false;
  }
  const auto engine_iter = it->second.find(engine);
  if (engine_iter == it->second.cend()) {
    return false;
  }
  const auto priority_iter = engine_iter->second.find(priority);
  if (priority_iter == engine_iter->second.cend()) {
    return false;
  }
  return priority_iter->second.find(backend) != priority_iter->second.cend();
}

bool CustomOpRegistry::HasCustomOp(const AscendString &op_type) const {
  const std::lock_guard<std::mutex> lock(mu_);
  const auto custom_op_it = custom_ops_.find(op_type);
  if (custom_op_it == custom_ops_.cend()) {
    return false;
  }
  const auto engine_iter = custom_op_it->second.find(OpEngine::kCustom);
  if (engine_iter == custom_op_it->second.cend()) {
    return false;
  }
  const auto priority_iter = engine_iter->second.find(OpRegistrationPriority::kTop);
  if (priority_iter == engine_iter->second.cend()) {
    return false;
  }
  return !priority_iter->second.empty();
}

graphStatus CustomOpRegistry::GetAllRegisteredOps(std::vector<AscendString> &all_registered_ops) const {
  const std::lock_guard<std::mutex> lock(mu_);
  for (const auto &op_creator : creators_) {
    const auto engine_iter = op_creator.second.find(OpEngine::kCustom);
    if (engine_iter == op_creator.second.cend()) {
      continue;
    }
    const auto priority_iter = engine_iter->second.find(OpRegistrationPriority::kTop);
    if (priority_iter != engine_iter->second.cend() && !priority_iter->second.empty()) {
      all_registered_ops.push_back(op_creator.first);
    }
  }
  return GRAPH_SUCCESS;
}

graphStatus CustomOpRegistry::LoadCustomOpsPartition(const uint8_t *data, size_t len) {
  if ((data == nullptr) || (len == 0U)) {
    GELOGE(GRAPH_PARAM_INVALID, "[CUSTOM OP] custom ops partition data is invalid, data %p, len %zu.", data, len);
    return GRAPH_PARAM_INVALID;
  }

  size_t offset = 0U;
  while (offset < len) {
    ParsedCustomKernelItem item{};
    const auto parse_ret = ParseCustomKernelItem(data, len, offset, item);
    if (parse_ret != GRAPH_SUCCESS) {
      return parse_ret;
    }
    const auto deserialize_ret = DeserializeCustomKernelItem(*this, item);
    if (deserialize_ret != GRAPH_SUCCESS) {
      return deserialize_ret;
    }
    offset += item.entry_size;
  }

  GELOGI("[CUSTOM OP] load custom ops partition success.");
  return GRAPH_SUCCESS;
}

void CustomOpRegistry::AppendProtoClaims(std::vector<OpProtoClaimRecord> &&claims) {
  const std::lock_guard<std::mutex> lock(mu_);
  if (impl_ == nullptr) {
    impl_ = std::make_shared<Impl>();
  }
  impl_->proto_claims.insert(impl_->proto_claims.end(), std::make_move_iterator(claims.begin()),
                             std::make_move_iterator(claims.end()));
}
}  // namespace ge
