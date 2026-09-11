/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CANN_GRAPH_ENGINE_OP_PROTO_LEDGER_H
#define CANN_GRAPH_ENGINE_OP_PROTO_LEDGER_H

#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <vector>

namespace ge {
class CustomOpRegistry;
using CustomOpRegistryPtr = std::shared_ptr<CustomOpRegistry>;

// 账本条目写入的全局 map 类别（清理清单，按位记录）
enum class OpProtoMapKind : uint16_t {
  kNone = 0x0U,
  kCreatorV1 = 0x1U,
  kCreatorV2 = 0x2U,
  kInferShape = 0x4U,
  kInferFormat = 0x8U,
  kVerify = 0x10U,
  kInferDataSlice = 0x20U,
  kInferValueRange = 0x40U,
  kAxisSlice = 0x80U,
  kAxisTypeInfo = 0x100U,
};

enum class OpProtoClaimKind : uint8_t { kCreated = 0U, kBorrowed = 1U, kOverridden = 2U };

struct OpProtoClaimRecord {
  std::string op_type;
  OpProtoMapKind map_kind = OpProtoMapKind::kNone;
  OpProtoClaimKind kind = OpProtoClaimKind::kCreated;

  OpProtoClaimRecord() = default;
  OpProtoClaimRecord(const std::string &type, const OpProtoClaimKind claim_kind) : op_type(type), kind(claim_kind) {}
  OpProtoClaimRecord(const std::string &type, const OpProtoMapKind proto_kind, const OpProtoClaimKind claim_kind)
      : op_type(type), map_kind(proto_kind), kind(claim_kind) {}
};

struct OpProtoConflictRecord {
  std::string op_type;
  OpProtoMapKind map_kind = OpProtoMapKind::kNone;
  std::string incumbent_so_name;
  std::string incumbent_fingerprint;
  std::string challenger_so_name;
  std::string challenger_fingerprint;
};

// OM 离线加载事务守卫：dlopen 静态构造与 pull 收集期间记账，析构时把事务日志提交到模型级 registry
class ScopedOpProtoLoadTxn {
 public:
  explicit ScopedOpProtoLoadTxn(const CustomOpRegistryPtr &registry);
  ~ScopedOpProtoLoadTxn();
  ScopedOpProtoLoadTxn(const ScopedOpProtoLoadTxn &) = delete;
  ScopedOpProtoLoadTxn &operator=(const ScopedOpProtoLoadTxn &) = delete;

  const std::vector<OpProtoConflictRecord> &GetConflicts() const {
    return conflicts_;
  }

 private:
  friend class OpProtoLedger;
  void RecordConflictLocked(const std::string &op_type, const OpProtoMapKind map_kind,
                            const std::string &incumbent_so_name, const std::string &incumbent_fingerprint);

  CustomOpRegistry *registry_;
  std::vector<OpProtoClaimRecord> claims_;
  std::vector<OpProtoConflictRecord> conflicts_;
  std::set<std::pair<std::string, OpProtoMapKind>> conflict_index_;
  std::set<std::pair<std::string, OpProtoMapKind>> claimed_map_keys_;
  std::string current_fingerprint_;
  std::string current_so_name_;
  bool active_ = false;
};

// 进程级原型账本：(op_type, map kind) → 提供者指纹/引用计数/SO 保活
class OpProtoLedger {
 public:
  static OpProtoLedger &GetInstance();
  // 清空全部账本状态（CustomOpSoLoader::Finalize 联动；测试隔离）
  static void ResetForFinalize();
  static bool HasActiveTxn();
  // 标记某个全局 map 项已被进程级注册覆盖，模型卸载时不得删除该项。
  static void MarkGlobalOverride(const std::string &op_type, const OpProtoMapKind kind);
  static void RecordOverride(const std::string &op_type, const OpProtoMapKind kind,
                             std::function<void()> restore_action);
  // 写前守门：RegisterXXX 实际变更全局 map 前咨询账本，返回 true 表示必须拒绝本次写入
  // （冲突已记录），使被拦截的写入根本不发生、不产生无人认领的全局残留。
  // overwriting=true 表示将覆盖已占用槽位的值（is_register_overridable 的 v2 override 路径），
  // false 表示新鲜写入（emplace 空槽位）。无事务时恒为 false（R4：零开销原路径）。
  static bool ShouldBlockRegister(const std::string &op_type, const OpProtoMapKind kind, const bool overwriting);
  // 注册入口挂钩：map_had_entry 表示调用时全局 map 是否已被占用。
  static void OnRegistered(const std::string &op_type, const OpProtoMapKind kind, const bool map_had_entry);
  // pull 阶段认领当前 op_type 的全部原型 map；条目缺失时不记录。
  static void ClaimProviderMaps(const std::string &op_type);
  // dlopen 前设置当前提供者（指纹 + so 名）；无事务时 no-op
  static void SetCurrentProvider(const std::string &fingerprint, const std::string &so_name);
  // dlopen 成功后立即绑定 SO 保活 token（先摘条目后 dlclose 的顺序保障）；无事务时 no-op
  static void AttachProviderHandle(const std::string &fingerprint, const std::shared_ptr<void> &keepalive);
  // 事务日志回放（~CustomOpRegistry 调用）
  static void ReleaseClaims(const std::vector<OpProtoClaimRecord> &claims);

 private:
  OpProtoLedger() = default;
  ~OpProtoLedger() = default;
  OpProtoLedger(const OpProtoLedger &) = delete;
  OpProtoLedger &operator=(const OpProtoLedger &) = delete;

  struct MapLedgerEntry {
    std::string provider_fingerprint;
    std::string provider_so_name;
    uint32_t refcount = 0U;
    bool overridden = false;
    bool has_backup = false;
    bool invalidated = false;
    std::function<void()> restore_action;
    std::shared_ptr<void> keepalive;
  };

  using LedgerEntry = std::map<OpProtoMapKind, MapLedgerEntry>;

  using EntryIterator = std::map<std::string, LedgerEntry>::iterator;

  // 构造"本 SO 写入一个全局 map"的记账条目。
  static MapLedgerEntry MakeProviderEntry(const ScopedOpProtoLoadTxn &txn);
  static bool IsProviderConsistentLocked(ScopedOpProtoLoadTxn &txn, const std::string &op_type,
                                         const LedgerEntry &entry);

  // 以下两个 Handle*Locked 的调用约定：调用方（OnRegistered）已持有 mu_；entry_it 为 entries_.find 结果，
  // 函数内部可能因新建/重建条目而更新该迭代器（外部借用分支的 emplace 不影响调用方后续，参数按值传递）
  // 全局 map 新鲜写入路径（map_had_entry=false）：created 记账 + 同事务 claim 去重
  void HandleFreshWriteLocked(ScopedOpProtoLoadTxn &txn, const std::string &op_type, const OpProtoMapKind kind,
                              EntryIterator entry_it);

  // 全局 map 占用路径（map_had_entry=true）：同提供者借用或异实现冲突。
  void HandleOccupiedWriteLocked(ScopedOpProtoLoadTxn &txn, const std::string &op_type, const OpProtoMapKind kind,
                                 EntryIterator entry_it);

  void EraseFromGlobalMapLocked(const std::string &op_type, const OpProtoMapKind kind) const;

  friend class ScopedOpProtoLoadTxn;
  std::mutex mu_;
  std::map<std::string, LedgerEntry> entries_;
  // 加载↔卸载回放并发防护（静默期延迟擦除）：加载事务 in-flight 期间回放整体推迟，
  // 最后一个事务析构时清扫，避免回放擦除与 dlopen 静态构造的全局 map 读写交错
  uint32_t active_txn_count_ = 0U;                    // 仅在 mu_ 下访问：in-flight 加载事务数
  std::vector<OpProtoClaimRecord> pending_releases_;  // 仅在 mu_ 下访问：被推迟的回放 claim
};
}  // namespace ge

#endif  // CANN_GRAPH_ENGINE_OP_PROTO_LEDGER_H
