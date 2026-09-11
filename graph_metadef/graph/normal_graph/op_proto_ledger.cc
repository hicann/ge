/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph/custom_op/op_proto_ledger.h"

#include "debug/ge_log.h"
#include "graph/custom_op_registry.h"
#include "graph/operator_factory_impl.h"

namespace ge {
namespace {
// 事务守卫与账本同 TU：thread-local 活动事务指针（加载全程同一线程）
thread_local ScopedOpProtoLoadTxn *g_active_proto_load_txn = nullptr;

ScopedOpProtoLoadTxn *GetActiveTxn() {
  return g_active_proto_load_txn;
}
}  // namespace

ScopedOpProtoLoadTxn::ScopedOpProtoLoadTxn(const CustomOpRegistryPtr &registry) : registry_(registry.get()) {
  if (g_active_proto_load_txn != nullptr) {
    GELOGW("[CUSTOM OP] nested op proto load txn detected, the inner txn is ignored.");
    return;
  }
  {
    auto &ledger = OpProtoLedger::GetInstance();
    const std::lock_guard<std::mutex> lock(ledger.mu_);
    // mu_ 内、构造返回前自增：回放侧在同一锁下的推迟判定与本事务的 map 读写因此不交错
    ++ledger.active_txn_count_;
  }
  g_active_proto_load_txn = this;
  active_ = true;
}

ScopedOpProtoLoadTxn::~ScopedOpProtoLoadTxn() {
  if (!active_) {
    return;
  }
  if (g_active_proto_load_txn == this) {
    g_active_proto_load_txn = nullptr;
  }
  if ((registry_ != nullptr) && (!claims_.empty())) {
    registry_->AppendProtoClaims(std::move(claims_));  // 提交事务日志，registry 析构时回放
  }
  std::vector<OpProtoClaimRecord> pending_batch;
  {
    auto &ledger = OpProtoLedger::GetInstance();
    const std::lock_guard<std::mutex> lock(ledger.mu_);
    if (ledger.active_txn_count_ > 0U) {
      --ledger.active_txn_count_;  // 防御 Finalize 复位后的下溢；正常路径恒大于 0
    }
    if ((ledger.active_txn_count_ == 0U) && (!ledger.pending_releases_.empty())) {
      pending_batch.swap(ledger.pending_releases_);  // 静默期：摘走被推迟的回放 claim，锁外清扫
    }
  }
  if (!pending_batch.empty()) {
    GELOGI("[CUSTOM OP] load quiescence reached, replaying %zu deferred op proto claim(s).", pending_batch.size());
    OpProtoLedger::ReleaseClaims(pending_batch);  // 清扫中若新事务开启，剩余 claim 自动再推迟（自收敛）
  }
}

void ScopedOpProtoLoadTxn::RecordConflictLocked(const std::string &op_type, const OpProtoMapKind map_kind,
                                                const std::string &incumbent_so_name,
                                                const std::string &incumbent_fingerprint) {
  const auto conflict_key = std::make_pair(op_type, map_kind);
  if (!conflict_index_.insert(conflict_key).second) {
    return;  // 同一加载事务内按 op_type 和 map kind 去重
  }
  OpProtoConflictRecord record;
  record.op_type = op_type;
  record.map_kind = map_kind;
  record.incumbent_so_name = incumbent_so_name;
  record.incumbent_fingerprint = incumbent_fingerprint;
  record.challenger_so_name = current_so_name_;
  record.challenger_fingerprint = current_fingerprint_;
  conflicts_.push_back(std::move(record));
}

OpProtoLedger &OpProtoLedger::GetInstance() {
  static OpProtoLedger instance;
  return instance;
}

void OpProtoLedger::ResetForFinalize() {
  auto &ledger = GetInstance();
  const std::lock_guard<std::mutex> lock(ledger.mu_);
  (void)ledger.entries_.clear();
  (void)ledger.pending_releases_.clear();  // 被推迟的 claim 随条目一并废弃
  ledger.active_txn_count_ = 0U;           // 测试隔离复位；Finalize 与加载并发属既有未定义场景
}

bool OpProtoLedger::HasActiveTxn() {
  return GetActiveTxn() != nullptr;
}

void OpProtoLedger::MarkGlobalOverride(const std::string &op_type, const OpProtoMapKind kind) {
  auto &ledger = GetInstance();
  const std::lock_guard<std::mutex> lock(ledger.mu_);
  const auto entry_it = ledger.entries_.find(op_type);
  if (entry_it != ledger.entries_.cend()) {
    const auto map_it = entry_it->second.find(kind);
    if (map_it != entry_it->second.cend()) {
      map_it->second.invalidated = true;
    }
  }
}

void OpProtoLedger::RecordOverride(const std::string &op_type, const OpProtoMapKind kind,
                                   std::function<void()> restore_action) {
  auto *txn = GetActiveTxn();
  if ((txn == nullptr) || (!restore_action)) {
    return;
  }
  auto &ledger = GetInstance();
  const std::lock_guard<std::mutex> lock(ledger.mu_);
  auto entry_it = ledger.entries_.find(op_type);
  if (entry_it == ledger.entries_.cend()) {
    entry_it = ledger.entries_.emplace(op_type, LedgerEntry{}).first;
  }
  auto &map_entry = entry_it->second[kind];
  map_entry.provider_fingerprint = txn->current_fingerprint_;
  map_entry.provider_so_name = txn->current_so_name_;
  map_entry.overridden = true;
  if (!map_entry.has_backup) {
    map_entry.has_backup = true;
    map_entry.restore_action = std::move(restore_action);
  }
  GELOGI("[CUSTOM OP] override recorded for op[%s], map kind[%u], provider so[%s], fingerprint[%s], backup saved[%d].",
         op_type.c_str(), static_cast<uint32_t>(kind), map_entry.provider_so_name.c_str(),
         map_entry.provider_fingerprint.c_str(), map_entry.has_backup ? 1 : 0);
  if (txn->claimed_map_keys_.insert({op_type, kind}).second) {
    txn->claims_.push_back({op_type, kind, OpProtoClaimKind::kOverridden});
    ++map_entry.refcount;
  }
}

void OpProtoLedger::SetCurrentProvider(const std::string &fingerprint, const std::string &so_name) {
  auto *txn = GetActiveTxn();
  if (txn == nullptr) {
    return;  // 单算子等无事务路径：no-op（R4）
  }
  txn->current_fingerprint_ = fingerprint;
  txn->current_so_name_ = so_name;
}

void OpProtoLedger::AttachProviderHandle(const std::string &fingerprint, const std::shared_ptr<void> &keepalive) {
  if ((GetActiveTxn() == nullptr) || (keepalive == nullptr)) {
    return;
  }
  auto &ledger = GetInstance();
  const std::lock_guard<std::mutex> lock(ledger.mu_);
  for (auto &entry : ledger.entries_) {
    for (auto &map_entry : entry.second) {
      if ((map_entry.second.provider_fingerprint == fingerprint) && (map_entry.second.keepalive == nullptr)) {
        map_entry.second.keepalive = keepalive;  // dlopen 已跑完静态构造，此刻回填条目 token
      }
    }
  }
}

OpProtoLedger::MapLedgerEntry OpProtoLedger::MakeProviderEntry(const ScopedOpProtoLoadTxn &txn) {
  MapLedgerEntry entry;
  entry.provider_fingerprint = txn.current_fingerprint_;
  entry.provider_so_name = txn.current_so_name_;
  return entry;
}

bool OpProtoLedger::IsProviderConsistentLocked(ScopedOpProtoLoadTxn &txn, const std::string &op_type,
                                               const LedgerEntry &entry) {
  for (const auto &map_item : entry) {
    const auto &map_entry = map_item.second;
    if ((!map_entry.provider_fingerprint.empty()) && (map_entry.provider_fingerprint != txn.current_fingerprint_)) {
      txn.RecordConflictLocked(op_type, map_item.first, map_entry.provider_so_name, map_entry.provider_fingerprint);
      GELOGW("[CUSTOM OP] op[%s] has foreign provider on map kind[%u], current provider[%s] blocked.", op_type.c_str(),
             static_cast<uint32_t>(map_item.first), txn.current_fingerprint_.c_str());
      return false;
    }
  }
  return true;
}

void OpProtoLedger::HandleFreshWriteLocked(ScopedOpProtoLoadTxn &txn, const std::string &op_type,
                                           const OpProtoMapKind kind, EntryIterator entry_it) {
  // 本次注册实际写入全局 map：created
  if (entry_it == entries_.cend()) {
    entry_it = entries_.emplace(op_type, LedgerEntry{}).first;
  }
  if (!IsProviderConsistentLocked(txn, op_type, entry_it->second)) {
    return;
  }
  auto map_it = entry_it->second.find(kind);
  if (map_it == entry_it->second.cend()) {
    map_it = entry_it->second.emplace(kind, MakeProviderEntry(txn)).first;
  } else if (map_it->second.provider_fingerprint != txn.current_fingerprint_) {
    if (map_it->second.refcount != 0U) {
      GELOGW("[CUSTOM OP] op[%s] stale ledger entry with pending claims, skip re-create.", op_type.c_str());
      txn.RecordConflictLocked(op_type, kind, map_it->second.provider_so_name, map_it->second.provider_fingerprint);
      return;
    }
    map_it->second = MakeProviderEntry(txn);
  }
  if (txn.claimed_map_keys_.insert({op_type, kind}).second) {
    txn.claims_.push_back({op_type, kind, OpProtoClaimKind::kCreated});
    ++map_it->second.refcount;
  }
}

void OpProtoLedger::HandleOccupiedWriteLocked(ScopedOpProtoLoadTxn &txn, const std::string &op_type,
                                              const OpProtoMapKind kind, EntryIterator entry_it) {
  // 全局 map 已被占用：按 provider 判断冲突。
  if (entry_it == entries_.cend()) {
    entry_it = entries_.emplace(op_type, LedgerEntry{}).first;
  }
  if (!IsProviderConsistentLocked(txn, op_type, entry_it->second)) {
    return;
  }
  auto map_it = entry_it->second.find(kind);
  if (map_it == entry_it->second.cend() || map_it->second.provider_fingerprint != txn.current_fingerprint_) {
    const std::string so_name = (map_it == entry_it->second.cend()) ? "" : map_it->second.provider_so_name;
    const std::string fingerprint = (map_it == entry_it->second.cend()) ? "" : map_it->second.provider_fingerprint;
    txn.RecordConflictLocked(op_type, kind, so_name, fingerprint);
    return;  // 冲突：不计数、不写日志账（拦截在 ModelHelper 聚合判定）
  }
  if (txn.claimed_map_keys_.insert({op_type, kind}).second) {
    txn.claims_.push_back({op_type, kind, OpProtoClaimKind::kBorrowed});
    ++map_it->second.refcount;
  }
}

bool OpProtoLedger::ShouldBlockRegister(const std::string &op_type, const OpProtoMapKind kind, const bool overwriting) {
  auto *txn = GetActiveTxn();
  if (txn == nullptr) {
    return false;  // 无事务（内置 OPP 初始化/在线注册/单算子）：零开销原路径（R4）
  }
  auto &ledger = GetInstance();
  const std::lock_guard<std::mutex> lock(ledger.mu_);
  const auto entry_it = ledger.entries_.find(op_type);
  if ((entry_it != ledger.entries_.cend()) && (!IsProviderConsistentLocked(*txn, op_type, entry_it->second))) {
    return true;
  }
  if ((entry_it == ledger.entries_.cend()) || (entry_it->second.find(kind) == entry_it->second.cend())) {
    // OM 加载事务允许接管或创建当前 map，调用方负责保存被覆盖值。
    return false;
  }
  const auto &entry = entry_it->second.at(kind);

  if (overwriting) {
    if (entry.provider_fingerprint == txn->current_fingerprint_) {
      return false;  // 同提供者等价值刷新，引用归零时可清理该 map
    }
    txn->RecordConflictLocked(op_type, kind, entry.provider_so_name, entry.provider_fingerprint);
    GELOGW("[CUSTOM OP] op[%s] overwrite of foreign provider entry under load txn blocked.", op_type.c_str());
    return true;
  }
  if (entry.provider_fingerprint == txn->current_fingerprint_ || entry.refcount == 0U) {
    return false;  // 同一提供者可以继续使用该 map；无引用条目：新鲜写入后由 OnRegistered 重建归账
  }
  // 活跃异实现且仍有借用方：此时写入无人认领、卸载不清理 → 在写入发生前拦截
  txn->RecordConflictLocked(op_type, kind, entry.provider_so_name, entry.provider_fingerprint);
  GELOGW("[CUSTOM OP] op[%s] fresh write onto live entry of provider so[%s] blocked.", op_type.c_str(),
         entry.provider_so_name.c_str());
  return true;
}

void OpProtoLedger::OnRegistered(const std::string &op_type, const OpProtoMapKind kind, const bool map_had_entry) {
  auto *txn = GetActiveTxn();
  if (txn == nullptr) {
    return;  // 无事务（内置 OPP 初始化/在线注册）：零开销原路径
  }
  auto &ledger = GetInstance();
  const std::lock_guard<std::mutex> lock(ledger.mu_);
  auto entry_it = ledger.entries_.find(op_type);
  if (!map_had_entry) {
    ledger.HandleFreshWriteLocked(*txn, op_type, kind, entry_it);
  } else {
    ledger.HandleOccupiedWriteLocked(*txn, op_type, kind, entry_it);
  }
}

void OpProtoLedger::ClaimProviderMaps(const std::string &op_type) {
  auto *txn = GetActiveTxn();
  if (txn == nullptr) {
    return;
  }
  if (txn->current_fingerprint_.empty()) {
    return;  // pull 提供者未设置：跳过，防误判冲突
  }
  auto &ledger = GetInstance();
  const std::lock_guard<std::mutex> lock(ledger.mu_);
  const auto entry_it = ledger.entries_.find(op_type);
  if (entry_it == ledger.entries_.cend()) {
    return;  // kernel-only 算子或原型来自内置：全局无本模型可清理之物
  }
  if (!IsProviderConsistentLocked(*txn, op_type, entry_it->second)) {
    return;
  }
  for (auto &map_item : entry_it->second) {
    const auto kind = map_item.first;
    auto &map_entry = map_item.second;
    if (txn->claimed_map_keys_.insert({op_type, kind}).second) {
      txn->claims_.push_back({op_type, kind, OpProtoClaimKind::kBorrowed});
      ++map_entry.refcount;
    }
  }
}

void OpProtoLedger::EraseFromGlobalMapLocked(const std::string &op_type, const OpProtoMapKind kind) const {
  if (kind == OpProtoMapKind::kCreatorV1) {
    if (OperatorFactoryImpl::operator_creators_ != nullptr) {
      (void)OperatorFactoryImpl::operator_creators_->erase(op_type);
    }
  }
  if (kind == OpProtoMapKind::kCreatorV2) {
    if (OperatorFactoryImpl::operator_creators_v2_ != nullptr) {
      (void)OperatorFactoryImpl::operator_creators_v2_->erase(op_type);
    }
  }
  if (kind == OpProtoMapKind::kInferShape) {
    if (OperatorFactoryImpl::operator_infershape_funcs_ != nullptr) {
      (void)OperatorFactoryImpl::operator_infershape_funcs_->erase(op_type);
    }
  }
  if (kind == OpProtoMapKind::kInferFormat) {
    if (OperatorFactoryImpl::operator_inferformat_funcs_ != nullptr) {
      (void)OperatorFactoryImpl::operator_inferformat_funcs_->erase(op_type);
    }
  }
  if (kind == OpProtoMapKind::kVerify) {
    if (OperatorFactoryImpl::operator_verify_funcs_ != nullptr) {
      (void)OperatorFactoryImpl::operator_verify_funcs_->erase(op_type);
    }
  }
  if (kind == OpProtoMapKind::kInferDataSlice) {
    if (OperatorFactoryImpl::operator_infer_data_slice_funcs_ != nullptr) {
      (void)OperatorFactoryImpl::operator_infer_data_slice_funcs_->erase(op_type);
    }
  }
  if (kind == OpProtoMapKind::kInferValueRange) {
    if (OperatorFactoryImpl::operator_infer_value_range_paras_ != nullptr) {
      (void)OperatorFactoryImpl::operator_infer_value_range_paras_->erase(op_type);
    }
  }
  if (kind == OpProtoMapKind::kAxisSlice) {
    if (OperatorFactoryImpl::operator_infer_axis_slice_funcs_ != nullptr) {
      (void)OperatorFactoryImpl::operator_infer_axis_slice_funcs_->erase(op_type);
    }
  }
  if (kind == OpProtoMapKind::kAxisTypeInfo) {
    if (OperatorFactoryImpl::operator_infer_axis_type_info_funcs_ != nullptr) {
      (void)OperatorFactoryImpl::operator_infer_axis_type_info_funcs_->erase(op_type);
    }
  }
}
void OpProtoLedger::ReleaseClaims(const std::vector<OpProtoClaimRecord> &claims) {
  auto &ledger = GetInstance();
  for (auto claim_iter = claims.cbegin(); claim_iter != claims.cend(); ++claim_iter) {
    std::shared_ptr<void> token_to_release;
    {
      const std::lock_guard<std::mutex> lock(ledger.mu_);
      // 加载↔卸载回放并发防护：检查必须与擦除同临界区——有加载事务 in-flight（其 dlopen
      // 静态构造正在读写全局 map）时，剩余 claim 整体推迟到静默期（最后一个事务析构清扫），
      // 避免延迟回放与裸 map 数据竞争
      if (ledger.active_txn_count_ != 0U) {
        ledger.pending_releases_.insert(ledger.pending_releases_.cend(), claim_iter, claims.cend());
        GELOGI("[CUSTOM OP] %zu op proto claim(s) deferred to load quiescence, %u txn(s) in flight.",
               static_cast<size_t>(claims.cend() - claim_iter), ledger.active_txn_count_);
        return;
      }
      auto entry_it = ledger.entries_.find(claim_iter->op_type);
      if (entry_it == ledger.entries_.cend()) {
        GELOGW("[CUSTOM OP] op proto ledger entry[%s] missing on release.", claim_iter->op_type.c_str());
        continue;
      }
      const auto map_it = entry_it->second.find(claim_iter->map_kind);
      if (map_it == entry_it->second.cend() || map_it->second.refcount == 0U) {
        GELOGW("[CUSTOM OP] op proto ledger entry[%s] map kind[%u] missing on release or refcount[%u] already zero.",
               claim_iter->op_type.c_str(), static_cast<uint32_t>(claim_iter->map_kind),
               map_it != entry_it->second.cend() ? map_it->second.refcount : 0U);
        continue;
      }
      --map_it->second.refcount;
      if (map_it->second.refcount > 0U) {
        continue;
      }
      bool restored = false;
      if (!map_it->second.invalidated) {
        if (map_it->second.overridden && map_it->second.has_backup && map_it->second.restore_action) {
          map_it->second.restore_action();
          restored = true;
        }
        if (!map_it->second.overridden) {
          ledger.EraseFromGlobalMapLocked(claim_iter->op_type, claim_iter->map_kind);
        }
        GELOGI("[CUSTOM OP] op[%s] map kind[%u] released to zero, %s, provider so[%s].", claim_iter->op_type.c_str(),
               static_cast<uint32_t>(claim_iter->map_kind), restored ? "builtin entry restored" : "global entry erased",
               map_it->second.provider_so_name.c_str());
      }
      token_to_release = std::move(map_it->second.keepalive);
      (void)entry_it->second.erase(map_it);
      if (entry_it->second.empty()) {
        (void)ledger.entries_.erase(entry_it);
      }
    }
    // 锁外释放 token：全部 map erase 完成后才可能触发 dlclose（先摘条目后 dlclose）
  }
}
}  // namespace ge
