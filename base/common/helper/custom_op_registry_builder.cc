/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "common/helper/custom_op_registry_builder.h"

#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "common/helper/custom_op_so_loader.h"
#include "graph/custom_op.h"
#include "framework/common/debug/log.h"
#include "graph_metadef/common/ge_common/util.h"
#include "graph/custom_op_pull_registry.h"
#include "mmpa/mmpa_api.h"

namespace ge {
namespace {
constexpr const char *kGetCreatorAbiVersionSymbol = "GetRegisteredCustomOpCreatorAbiVersion";
constexpr const char *kGetCreatorNumSymbol = "GetRegisteredCustomOpCreatorNum";
constexpr const char *kGetCreatorsSymbol = "GetRegisteredCustomOpCreators";

using GetCreatorAbiVersionFunc = uint32_t (*)();
using GetCreatorNumFunc = size_t (*)();

// Keep the historical V1 layout private. It is only needed to decode old custom op SOs.
struct LegacyCustomOpTypeToCreator {
  uint32_t struct_size;
  const char *op_type;
  CustomOpCreateFunc creator;
};

using GetCreatorsFunc = int32_t (*)(LegacyCustomOpTypeToCreator *, size_t, size_t);
using GetCreatorsFuncV2 = int32_t (*)(CustomOpTypeToCreator *, size_t, size_t);

struct PullCreatorSymbols {
  uint32_t abi_version = kCustomOpCreatorPullAbiVersion;
  GetCreatorAbiVersionFunc get_abi_version = nullptr;
  GetCreatorNumFunc get_creator_num = nullptr;
  GetCreatorsFunc get_creators = nullptr;
  GetCreatorsFuncV2 get_creators_v2 = nullptr;
};

struct PendingCreator {
  std::string op_type;
  OpBackend backend;
  CustomOpCreateFunc creator = nullptr;
};

bool IsValidOpBackend(const OpBackend backend) {
  return (backend == OpBackend::kDevice) || (backend == OpBackend::kHostCPU);
}

Status ResolvePullCreatorSymbols(void *const so_handle, const CustomOpRegistryBuilder::DlsymFunc dlsym_func,
                                 PullCreatorSymbols &symbols) {
  symbols.get_abi_version =
      reinterpret_cast<GetCreatorAbiVersionFunc>(dlsym_func(so_handle, kGetCreatorAbiVersionSymbol));
  symbols.get_creator_num = reinterpret_cast<GetCreatorNumFunc>(dlsym_func(so_handle, kGetCreatorNumSymbol));
  auto get_creators = dlsym_func(so_handle, kGetCreatorsSymbol);
  if ((symbols.get_abi_version == nullptr) || (symbols.get_creator_num == nullptr) || (get_creators == nullptr)) {
    GELOGE(FAILED, "[CUSTOM OP] pull creator ABI symbols are incomplete.");
    return FAILED;
  }

  const uint32_t abi_version = symbols.get_abi_version();
  symbols.abi_version = abi_version;
  if (abi_version == kCustomOpCreatorPullAbiVersion) {
    symbols.get_creators = reinterpret_cast<GetCreatorsFunc>(get_creators);
  } else if (abi_version == kCustomOpCreatorPullAbiVersionV2) {
    symbols.get_creators_v2 = reinterpret_cast<GetCreatorsFuncV2>(get_creators);
  } else {
    GELOGE(FAILED, "[CUSTOM OP] unsupported pull creator ABI version %u.", abi_version);
    return FAILED;
  }
  return SUCCESS;
}

Status LoadRawCreatorsV1(const PullCreatorSymbols &symbols, std::vector<LegacyCustomOpTypeToCreator> &raw_creators) {
  const uint32_t abi_version = symbols.abi_version;
  if (abi_version != kCustomOpCreatorPullAbiVersion) {
    GELOGE(FAILED, "[CUSTOM OP] pull creator ABI version %u does not match expected %u.", abi_version,
           kCustomOpCreatorPullAbiVersion);
    return FAILED;
  }

  const size_t creator_num = symbols.get_creator_num();
  raw_creators.resize(creator_num);
  auto creator = raw_creators.empty() ? nullptr : raw_creators.data();
  const auto ret = symbols.get_creators(creator, raw_creators.size(), sizeof(LegacyCustomOpTypeToCreator));
  if (ret != 0) {
    GELOGE(FAILED, "[CUSTOM OP] get registered custom op creators failed, ret:%d.", ret);
    return FAILED;
  }
  return SUCCESS;
}

Status LoadRawCreatorsV2(const PullCreatorSymbols &symbols, std::vector<CustomOpTypeToCreator> &raw_creators) {
  const uint32_t abi_version = symbols.abi_version;
  if (abi_version != kCustomOpCreatorPullAbiVersionV2) {
    GELOGE(FAILED, "[CUSTOM OP] pull creator V2 ABI version %u does not match expected %u.", abi_version,
           kCustomOpCreatorPullAbiVersionV2);
    return FAILED;
  }

  const size_t creator_num = symbols.get_creator_num();
  raw_creators.resize(creator_num);
  auto creator = raw_creators.empty() ? nullptr : raw_creators.data();
  const auto ret = symbols.get_creators_v2(creator, raw_creators.size(), sizeof(CustomOpTypeToCreator));
  if (ret != 0) {
    GELOGE(FAILED, "[CUSTOM OP] get registered custom op creators V2 failed, ret:%d.", ret);
    return FAILED;
  }
  return SUCCESS;
}

Status ValidateAndCollectCreator(const std::string &op_type, const OpBackend backend, const CustomOpCreateFunc creator,
                                 const CustomOpRegistryPtr &registry,
                                 std::set<std::pair<std::string, OpBackend>> &pending_op_keys,
                                 std::vector<PendingCreator> &pending_creators) {
  if ((op_type.empty()) || (creator == nullptr) || (!IsValidOpBackend(backend))) {
    GELOGE(FAILED, "[CUSTOM OP] invalid custom op pull creator entry.");
    return FAILED;
  }

  const auto op_key = std::make_pair(op_type, backend);
  if (registry->HasCreator(AscendString(op_type.c_str()), backend) ||
      (pending_op_keys.find(op_key) != pending_op_keys.end())) {
    GELOGE(FAILED, "[CUSTOM OP] duplicate custom op creator for %s backend %u in model registry.", op_type.c_str(),
           static_cast<uint32_t>(backend));
    return FAILED;
  }

  (void)pending_op_keys.insert(op_key);
  pending_creators.push_back({op_type, backend, creator});
  return SUCCESS;
}

Status ValidateAndCollectCreator(const LegacyCustomOpTypeToCreator &raw_creator, const CustomOpRegistryPtr &registry,
                                 std::set<std::pair<std::string, OpBackend>> &pending_op_keys,
                                 std::vector<PendingCreator> &pending_creators) {
  if ((raw_creator.struct_size != sizeof(LegacyCustomOpTypeToCreator)) || (raw_creator.op_type == nullptr) ||
      (raw_creator.op_type[0] == '\0') || (raw_creator.creator == nullptr)) {
    GELOGE(FAILED, "[CUSTOM OP] invalid custom op pull creator entry.");
    return FAILED;
  }

  return ValidateAndCollectCreator(std::string(raw_creator.op_type), OpBackend::kDevice, raw_creator.creator, registry,
                                   pending_op_keys, pending_creators);
}

Status ValidateAndCollectCreator(const CustomOpTypeToCreator &raw_creator, const CustomOpRegistryPtr &registry,
                                 std::set<std::pair<std::string, OpBackend>> &pending_op_keys,
                                 std::vector<PendingCreator> &pending_creators) {
  if ((raw_creator.struct_size != sizeof(CustomOpTypeToCreator)) || (raw_creator.op_type == nullptr) ||
      (raw_creator.op_type[0] == '\0') || (raw_creator.creator == nullptr)) {
    GELOGE(FAILED, "[CUSTOM OP] invalid custom op pull creator V2 entry.");
    return FAILED;
  }

  return ValidateAndCollectCreator(std::string(raw_creator.op_type), raw_creator.backend, raw_creator.creator, registry,
                                   pending_op_keys, pending_creators);
}

Status LoadAndCollectCreators(const PullCreatorSymbols &symbols, const CustomOpRegistryPtr &registry,
                              std::set<std::pair<std::string, OpBackend>> &pending_op_keys,
                              std::vector<PendingCreator> &pending_creators) {
  if (symbols.abi_version == kCustomOpCreatorPullAbiVersionV2) {
    std::vector<CustomOpTypeToCreator> raw_creators;
    GE_CHK_STATUS_RET(LoadRawCreatorsV2(symbols, raw_creators), "[CUSTOM OP] load raw pull creators V2 failed.");
    for (const auto &raw_creator : raw_creators) {
      GE_CHK_STATUS_RET(ValidateAndCollectCreator(raw_creator, registry, pending_op_keys, pending_creators),
                        "[CUSTOM OP] validate pull creator V2 failed.");
    }
    return SUCCESS;
  }

  std::vector<LegacyCustomOpTypeToCreator> raw_creators;
  GE_CHK_STATUS_RET(LoadRawCreatorsV1(symbols, raw_creators), "[CUSTOM OP] load raw pull creators failed.");
  for (const auto &raw_creator : raw_creators) {
    GE_CHK_STATUS_RET(ValidateAndCollectCreator(raw_creator, registry, pending_op_keys, pending_creators),
                      "[CUSTOM OP] validate pull creator failed.");
  }
  return SUCCESS;
}

Status CollectCreatorsFromSoHandle(const CustomOpSoHandlePtr &so_handle, const CustomOpRegistryPtr &registry,
                                   const CustomOpRegistryBuilder::DlsymFunc dlsym_func,
                                   std::set<std::pair<std::string, OpBackend>> &pending_op_keys,
                                   std::vector<PendingCreator> &pending_creators) {
  if ((so_handle == nullptr) || (so_handle->GetHandle() == nullptr)) {
    GELOGE(FAILED, "[CUSTOM OP] custom op so handle is null.");
    return FAILED;
  }

  PullCreatorSymbols symbols;
  GE_CHK_STATUS_RET(ResolvePullCreatorSymbols(so_handle->GetHandle(), dlsym_func, symbols),
                    "[CUSTOM OP] resolve pull creator symbols failed.");
  return LoadAndCollectCreators(symbols, registry, pending_op_keys, pending_creators);
}
}  // namespace

Status CustomOpRegistryBuilder::AddCreatorsFromSoHandles(const std::vector<CustomOpSoHandlePtr> &so_handles,
                                                         const CustomOpRegistryPtr &registry) {
  return AddCreatorsFromSoHandles(so_handles, registry, mmDlsym);
}

Status CustomOpRegistryBuilder::AddCreatorsFromSoHandles(const std::vector<CustomOpSoHandlePtr> &so_handles,
                                                         const CustomOpRegistryPtr &registry,
                                                         const DlsymFunc dlsym_func) {
  if ((registry == nullptr) || (dlsym_func == nullptr)) {
    GELOGE(FAILED, "[CUSTOM OP] registry or dlsym function is null.");
    return FAILED;
  }

  std::set<std::pair<std::string, OpBackend>> pending_op_keys;
  std::vector<PendingCreator> pending_creators;
  for (const auto &so_handle : so_handles) {
    GE_CHK_STATUS_RET(CollectCreatorsFromSoHandle(so_handle, registry, dlsym_func, pending_op_keys, pending_creators),
                      "[CUSTOM OP] collect creators from custom op so handle failed.");
  }

  for (const auto &pending_creator : pending_creators) {
    const auto register_ret = registry->RegisterCreator(
        AscendString(pending_creator.op_type.c_str()), pending_creator.backend,
        [creator = pending_creator.creator]() { return std::unique_ptr<BaseCustomOp>(creator()); });
    GE_CHK_STATUS_RET(register_ret, "[CUSTOM OP] register pull creator to model registry failed.");
  }
  registry->AddSoHandles(so_handles);
  return SUCCESS;
}
}  // namespace ge
