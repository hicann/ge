/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>

#include <atomic>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "graph/custom_op_registry.h"
#include "graph/custom_op/op_proto_ledger.h"
#include "graph/operator_factory_impl.h"

using namespace ge;

namespace {
constexpr char kSoA[] = "fake_custom_op_v1.so";
constexpr char kSoB[] = "fake_custom_op_v2.so";
constexpr char kFpA[] = "fp_fake_v1";
constexpr char kFpB[] = "fp_fake_v2";

class OpProtoLedgerUT : public testing::Test {
 protected:
  void SetUp() override {
    OpProtoLedger::ResetForFinalize();
    CleanDirtyFactoryEntries();
  }
  void TearDown() override {
    OpProtoLedger::ResetForFinalize();
    CleanDirtyFactoryEntries();
  }
  // 集成用例通过真实工厂注册的条目兜底清理（防止断言失败中断留下脏数据）
  static void CleanDirtyFactoryEntries() {
    for (const auto &op_type : dirty_ops_) {
      OperatorFactoryImpl::RemoveCustomOpCreators({op_type});
    }
    dirty_ops_.clear();
  }
  static std::vector<std::string> dirty_ops_;
};
std::vector<std::string> OpProtoLedgerUT::dirty_ops_;

// 白盒观测助手（依赖 UT target 的 -fno-access-control，直访账本私有 entries_）：
// 注意：entries_ 在 registry 析构/回放后可能被修改，每个断言点须重新获取，不得跨改动复用指针
const OpProtoLedger::MapLedgerEntry *GetLedgerEntryForUt(const std::string &op_type,
                                                         const OpProtoMapKind kind = OpProtoMapKind::kCreatorV2) {
  const auto &entries = OpProtoLedger::GetInstance().entries_;
  const auto it = entries.find(op_type);
  if (it == entries.cend()) {
    return nullptr;
  }
  const auto map_it = it->second.find(kind);
  return (map_it == it->second.cend()) ? nullptr : &map_it->second;
}

bool HasLedgerEntryForUt(const std::string &op_type) {
  return OpProtoLedger::GetInstance().entries_.count(op_type) != 0U;
}
}  // namespace

// F1 矩阵行 1：无条目 → created，map 级 refcount=1
TEST_F(OpProtoLedgerUT, FreshCreateRecordsClaimAndEntry) {
  auto registry = std::make_shared<CustomOpRegistry>();
  {
    ScopedOpProtoLoadTxn txn(registry);
    OpProtoLedger::SetCurrentProvider(kFpA, kSoA);
    OpProtoLedger::OnRegistered("LedgerFreshOp", OpProtoMapKind::kCreatorV2, false);
    const auto *entry = GetLedgerEntryForUt("LedgerFreshOp");
    ASSERT_NE(entry, nullptr);
    EXPECT_EQ(entry->refcount, 1U);
    EXPECT_EQ(entry->provider_fingerprint, std::string(kFpA));
  }
}

// 同 op_type 第二类 map 写入：两个 map 分别记账
TEST_F(OpProtoLedgerUT, SecondKindMergesWrittenMapsWithoutNewClaim) {
  auto registry = std::make_shared<CustomOpRegistry>();
  {
    ScopedOpProtoLoadTxn txn(registry);
    OpProtoLedger::SetCurrentProvider(kFpA, kSoA);
    OpProtoLedger::OnRegistered("LedgerTwoKindOp", OpProtoMapKind::kCreatorV2, false);
    OpProtoLedger::OnRegistered("LedgerTwoKindOp", OpProtoMapKind::kInferShape, false);
    const auto *creator_entry = GetLedgerEntryForUt("LedgerTwoKindOp");
    const auto *infer_entry = GetLedgerEntryForUt("LedgerTwoKindOp", OpProtoMapKind::kInferShape);
    ASSERT_NE(creator_entry, nullptr);
    ASSERT_NE(infer_entry, nullptr);
    EXPECT_EQ(creator_entry->refcount, 1U);
    EXPECT_EQ(infer_entry->refcount, 1U);
  }
}

// F1 矩阵行 4（pull 路径）：同指纹借用 refcount+1，记 borrowed
TEST_F(OpProtoLedgerUT, PullClaimSameFingerprintBorrows) {
  auto reg1 = std::make_shared<CustomOpRegistry>();
  auto reg2 = std::make_shared<CustomOpRegistry>();
  {
    ScopedOpProtoLoadTxn txn(reg1);
    OpProtoLedger::SetCurrentProvider(kFpA, kSoA);
    OpProtoLedger::OnRegistered("LedgerBorrowOp", OpProtoMapKind::kCreatorV2, false);
    OpProtoLedger::OnRegistered("LedgerBorrowOp", OpProtoMapKind::kInferShape, false);
  }
  {
    ScopedOpProtoLoadTxn txn(reg2);
    OpProtoLedger::SetCurrentProvider(kFpA, kSoA);
    OpProtoLedger::ClaimProviderMaps("LedgerBorrowOp");
    const auto *entry = GetLedgerEntryForUt("LedgerBorrowOp");
    const auto *infer_entry = GetLedgerEntryForUt("LedgerBorrowOp", OpProtoMapKind::kInferShape);
    ASSERT_NE(entry, nullptr);
    ASSERT_NE(infer_entry, nullptr);
    EXPECT_EQ(entry->refcount, 2U);
    EXPECT_EQ(infer_entry->refcount, 2U);
  }
}

// F1 矩阵行 5（pull 路径）：异指纹 → 冲突记录，不计数
TEST_F(OpProtoLedgerUT, PullClaimDifferentFingerprintRecordsConflict) {
  auto reg1 = std::make_shared<CustomOpRegistry>();
  auto reg2 = std::make_shared<CustomOpRegistry>();
  {
    ScopedOpProtoLoadTxn txn(reg1);
    OpProtoLedger::SetCurrentProvider(kFpA, kSoA);
    OpProtoLedger::OnRegistered("LedgerConflictOp", OpProtoMapKind::kCreatorV2, false);
  }
  {
    ScopedOpProtoLoadTxn txn(reg2);
    OpProtoLedger::SetCurrentProvider(kFpB, kSoB);
    OpProtoLedger::ClaimProviderMaps("LedgerConflictOp");
    EXPECT_EQ(txn.GetConflicts().size(), 1U);
    EXPECT_EQ(txn.GetConflicts()[0].op_type, std::string("LedgerConflictOp"));
    EXPECT_EQ(txn.GetConflicts()[0].challenger_fingerprint, std::string(kFpB));
    EXPECT_EQ(txn.GetConflicts()[0].incumbent_fingerprint, std::string(kFpA));
  }
  const auto *entry = GetLedgerEntryForUt("LedgerConflictOp");
  ASSERT_NE(entry, nullptr);
  EXPECT_EQ(entry->refcount, 1U);  // 冲突不计数
}

// F1 矩阵行 5（注册路径）：map 占用 + 账本异指纹 → 冲突
TEST_F(OpProtoLedgerUT, RegisterOnOccupiedMapDifferentFingerprintRecordsConflict) {
  auto reg1 = std::make_shared<CustomOpRegistry>();
  auto reg2 = std::make_shared<CustomOpRegistry>();
  {
    ScopedOpProtoLoadTxn txn(reg1);
    OpProtoLedger::SetCurrentProvider(kFpA, kSoA);
    OpProtoLedger::OnRegistered("LedgerRegConflictOp", OpProtoMapKind::kCreatorV2, false);
  }
  {
    ScopedOpProtoLoadTxn txn(reg2);
    OpProtoLedger::SetCurrentProvider(kFpB, kSoB);
    OpProtoLedger::OnRegistered("LedgerRegConflictOp", OpProtoMapKind::kCreatorV2, true);
    EXPECT_EQ(txn.GetConflicts().size(), 1U);
  }
  const auto *entry = GetLedgerEntryForUt("LedgerRegConflictOp");
  ASSERT_NE(entry, nullptr);
  EXPECT_EQ(entry->refcount, 1U);
}

// 无事务时注册：账本零交互（单算子路径 R4）
TEST_F(OpProtoLedgerUT, NoTxnRegistrationLeavesLedgerUntouched) {
  dirty_ops_.push_back("LedgerNoTxnOp");
  const OpCreatorV2 creator = [](const AscendString &) { return Operator(); };
  ASSERT_EQ(OperatorFactoryImpl::RegisterOperatorCreator("LedgerNoTxnOp", creator), GRAPH_SUCCESS);
  EXPECT_FALSE(HasLedgerEntryForUt("LedgerNoTxnOp"));
}

// pull 提供者未设置（空指纹）：跳过（防误判冲突）
TEST_F(OpProtoLedgerUT, PullClaimWithoutProviderIsSkipped) {
  auto reg1 = std::make_shared<CustomOpRegistry>();
  auto reg2 = std::make_shared<CustomOpRegistry>();
  {
    ScopedOpProtoLoadTxn txn(reg1);
    OpProtoLedger::SetCurrentProvider(kFpA, kSoA);
    OpProtoLedger::OnRegistered("LedgerNoProviderOp", OpProtoMapKind::kCreatorV2, false);
  }
  {
    ScopedOpProtoLoadTxn txn(reg2);
    OpProtoLedger::ClaimProviderMaps("LedgerNoProviderOp");  // 未 SetCurrentProvider
    EXPECT_TRUE(txn.GetConflicts().empty());
  }
  const auto *entry = GetLedgerEntryForUt("LedgerNoProviderOp");
  ASSERT_NE(entry, nullptr);
  EXPECT_EQ(entry->refcount, 1U);
}

// 嵌套事务：内层失效，记账归外层
TEST_F(OpProtoLedgerUT, NestedTxnIsIgnored) {
  auto registry = std::make_shared<CustomOpRegistry>();
  {
    ScopedOpProtoLoadTxn outer(registry);
    OpProtoLedger::SetCurrentProvider(kFpA, kSoA);
    {
      ScopedOpProtoLoadTxn inner(registry);
      OpProtoLedger::SetCurrentProvider(kFpB, kSoB);  // 嵌套：不会生效
      OpProtoLedger::OnRegistered("LedgerNestedOp", OpProtoMapKind::kCreatorV2, false);
    }
    const auto *entry = GetLedgerEntryForUt("LedgerNestedOp");
    ASSERT_NE(entry, nullptr);
    EXPECT_EQ(entry->provider_fingerprint, std::string(kFpB));  // provider 写在 txn 数据上，嵌套不隔离
  }
}

// 回放条目缺失（幽灵 claim）：告警不崩溃
TEST_F(OpProtoLedgerUT, ReleaseGhostClaimWarnsWithoutCrash) {
  const std::vector<OpProtoClaimRecord> ghost{
      {"LedgerGhostOp", OpProtoMapKind::kCreatorV2, OpProtoClaimKind::kCreated}};
  OpProtoLedger::ReleaseClaims(ghost);
  EXPECT_FALSE(HasLedgerEntryForUt("LedgerGhostOp"));
}

// 集成：真实工厂注册路径（v2 creator）在事务内按 map 创建账本条目
TEST_F(OpProtoLedgerUT, FactoryRegisterV2InsideTxnCreatesEntry) {
  dirty_ops_.push_back("LedgerFactoryV2Op");
  auto registry = std::make_shared<CustomOpRegistry>();
  {
    ScopedOpProtoLoadTxn txn(registry);
    OpProtoLedger::SetCurrentProvider(kFpA, kSoA);
    const OpCreatorV2 creator = [](const AscendString &) { return Operator(); };
    ASSERT_EQ(OperatorFactoryImpl::RegisterOperatorCreator("LedgerFactoryV2Op", creator), GRAPH_SUCCESS);
    const auto *entry = GetLedgerEntryForUt("LedgerFactoryV2Op");
    ASSERT_NE(entry, nullptr);
    EXPECT_EQ(entry->refcount, 1U);
  }
}

// 集成：真实工厂注册（v1 creator）在事务外 → 账本零交互（已有用例覆盖语义，此处验证 v1 路径）
TEST_F(OpProtoLedgerUT, FactoryRegisterV1OutsideTxnNoLedgerEntry) {
  dirty_ops_.push_back("LedgerFactoryV1Op");
  const OpCreator creator = [](const std::string &) { return Operator(); };
  ASSERT_EQ(OperatorFactoryImpl::RegisterOperatorCreator("LedgerFactoryV1Op", creator), GRAPH_SUCCESS);
  EXPECT_TRUE(OperatorFactoryImpl::IsExistOp("LedgerFactoryV1Op"));
  EXPECT_FALSE(HasLedgerEntryForUt("LedgerFactoryV1Op"));
}

// F2：map 引用归零后清理对应全局 map；先摘条目后释放 token
TEST_F(OpProtoLedgerUT, ReleaseToZeroErasesGlobalMaps) {
  dirty_ops_.push_back("LedgerLifeOp");
  auto token = std::make_shared<int32_t>(1);
  {
    auto reg1 = std::make_shared<CustomOpRegistry>();
    {
      ScopedOpProtoLoadTxn txn(reg1);
      OpProtoLedger::SetCurrentProvider(kFpA, kSoA);
      const OpCreatorV2 creator = [](const AscendString &) { return Operator(); };
      ASSERT_EQ(OperatorFactoryImpl::RegisterOperatorCreator("LedgerLifeOp", creator), GRAPH_SUCCESS);
      OpProtoLedger::AttachProviderHandle(kFpA, std::shared_ptr<void>(token));
      ASSERT_EQ(token.use_count(), 2);  // 测试持有 + 账本条目持有
    }
    auto reg2 = std::make_shared<CustomOpRegistry>();
    {
      ScopedOpProtoLoadTxn txn(reg2);
      OpProtoLedger::SetCurrentProvider(kFpA, kSoA);
      OpProtoLedger::ClaimProviderMaps("LedgerLifeOp");
    }
    const auto *entry = GetLedgerEntryForUt("LedgerLifeOp");
    ASSERT_NE(entry, nullptr);
    ASSERT_EQ(entry->refcount, 2U);
    reg1.reset();  // 模型 A 卸载：rc 2→1，条目保留
    EXPECT_TRUE(OperatorFactoryImpl::IsExistOp("LedgerLifeOp"));
    entry = GetLedgerEntryForUt("LedgerLifeOp");  // 回放后重新获取，不复用旧指针
    ASSERT_NE(entry, nullptr);
    EXPECT_EQ(entry->refcount, 1U);
    reg2.reset();  // 全部卸载：rc 0 → 清理
  }
  EXPECT_FALSE(OperatorFactoryImpl::IsExistOp("LedgerLifeOp"));
  EXPECT_EQ(token.use_count(), 1);  // 保活 token 已释放（SO 可 dlclose）
  EXPECT_FALSE(HasLedgerEntryForUt("LedgerLifeOp"));
}

// 进程级覆盖只使被覆盖的 map 项失效，不影响同一模型中其他算子的清理。
TEST_F(OpProtoLedgerUT, GlobalOverrideInvalidatesOnlyTargetMap) {
  dirty_ops_.push_back("LedgerOverrideOp");
  dirty_ops_.push_back("LedgerKeepOp");
  auto registry = std::make_shared<CustomOpRegistry>();
  {
    ScopedOpProtoLoadTxn txn(registry);
    OpProtoLedger::SetCurrentProvider(kFpA, kSoA);
    const OpCreatorV2 creator = [](const AscendString &) { return Operator(); };
    ASSERT_EQ(OperatorFactoryImpl::RegisterOperatorCreator("LedgerOverrideOp", creator), GRAPH_SUCCESS);
    ASSERT_EQ(OperatorFactoryImpl::RegisterOperatorCreator("LedgerKeepOp", creator), GRAPH_SUCCESS);
    OpProtoLedger::MarkGlobalOverride("LedgerOverrideOp", OpProtoMapKind::kCreatorV2);
  }
  registry.reset();
  EXPECT_TRUE(OperatorFactoryImpl::IsExistOp("LedgerOverrideOp"));
  EXPECT_FALSE(OperatorFactoryImpl::IsExistOp("LedgerKeepOp"));
  EXPECT_FALSE(HasLedgerEntryForUt("LedgerOverrideOp"));
  EXPECT_FALSE(HasLedgerEntryForUt("LedgerKeepOp"));
}

// OM 事务可以覆盖已有的 CreatorV2 和 InferShape，卸载时恢复原有实现。
TEST_F(OpProtoLedgerUT, OverrideBuiltinMapsRestoresOnRelease) {
  dirty_ops_.push_back("LedgerBuiltinOverrideOp");
  const OpCreatorV2 builtin_creator = [](const AscendString &) { return Operator(); };
  const OpCreatorV2 om_creator = [](const AscendString &) { return Operator(); };
  const InferShapeFunc builtin_infer = [](Operator &) { return GRAPH_SUCCESS; };
  const InferShapeFunc om_infer = [](Operator &) { return GRAPH_FAILED; };
  ASSERT_EQ(OperatorFactoryImpl::RegisterOperatorCreator("LedgerBuiltinOverrideOp", builtin_creator), GRAPH_SUCCESS);
  ASSERT_EQ(OperatorFactoryImpl::RegisterInferShapeFunc("LedgerBuiltinOverrideOp", builtin_infer), GRAPH_SUCCESS);
  {
    auto registry = std::make_shared<CustomOpRegistry>();
    {
      ScopedOpProtoLoadTxn txn(registry);
      OpProtoLedger::SetCurrentProvider(kFpA, kSoA);
      EXPECT_EQ(OperatorFactoryImpl::RegisterOperatorCreator("LedgerBuiltinOverrideOp", om_creator), GRAPH_SUCCESS);
      EXPECT_EQ(OperatorFactoryImpl::RegisterInferShapeFunc("LedgerBuiltinOverrideOp", om_infer), GRAPH_SUCCESS);
      const auto *creator_entry = GetLedgerEntryForUt("LedgerBuiltinOverrideOp");
      const auto *infer_entry = GetLedgerEntryForUt("LedgerBuiltinOverrideOp", OpProtoMapKind::kInferShape);
      ASSERT_NE(creator_entry, nullptr);
      ASSERT_NE(infer_entry, nullptr);
      EXPECT_TRUE(creator_entry->overridden);
      EXPECT_TRUE(infer_entry->overridden);
      EXPECT_TRUE(creator_entry->has_backup);
      EXPECT_TRUE(infer_entry->has_backup);
    }
    EXPECT_NE(OperatorFactoryImpl::GetInferShapeFunc("LedgerBuiltinOverrideOp"), nullptr);
  }
  EXPECT_TRUE(OperatorFactoryImpl::IsExistOp("LedgerBuiltinOverrideOp"));
  EXPECT_NE(OperatorFactoryImpl::GetInferShapeFunc("LedgerBuiltinOverrideOp"), nullptr);
  EXPECT_FALSE(HasLedgerEntryForUt("LedgerBuiltinOverrideOp"));
}

// OM 覆盖后被 Python 在线注册接管时，OM 卸载不得恢复过期的内置 backup。
TEST_F(OpProtoLedgerUT, PythonOverridePreventsStaleRestore) {
  dirty_ops_.push_back("LedgerPythonOverrideOp");
  const OpCreatorV2 builtin_creator = [](const AscendString &) { return Operator(); };
  const OpCreatorV2 om_creator = [](const AscendString &) { return Operator(); };
  const OpCreatorV2 python_creator = [](const AscendString &) { return Operator(); };
  ASSERT_EQ(OperatorFactoryImpl::RegisterOperatorCreator("LedgerPythonOverrideOp", builtin_creator), GRAPH_SUCCESS);
  auto registry = std::make_shared<CustomOpRegistry>();
  {
    ScopedOpProtoLoadTxn txn(registry);
    OpProtoLedger::SetCurrentProvider(kFpA, kSoA);
    ASSERT_EQ(OperatorFactoryImpl::RegisterOperatorCreator("LedgerPythonOverrideOp", om_creator), GRAPH_SUCCESS);
  }
  OperatorFactoryImpl::SetRegisterOverridable(true);
  ASSERT_EQ(OperatorFactoryImpl::RegisterOperatorCreator("LedgerPythonOverrideOp", python_creator), GRAPH_SUCCESS);
  OperatorFactoryImpl::SetRegisterOverridable(false);
  registry.reset();
  EXPECT_TRUE(OperatorFactoryImpl::IsExistOp("LedgerPythonOverrideOp"));
  EXPECT_FALSE(HasLedgerEntryForUt("LedgerPythonOverrideOp"));
}

TEST_F(OpProtoLedgerUT, CreatorV1OverrideRestoresBuiltinValue) {
  dirty_ops_.push_back("LedgerCreatorV1OverrideOp");
  const OpCreator builtin_creator = [](const std::string &) { return Operator(); };
  const OpCreator om_creator = [](const std::string &) { return Operator(); };
  ASSERT_EQ(OperatorFactoryImpl::RegisterOperatorCreator("LedgerCreatorV1OverrideOp", builtin_creator), GRAPH_SUCCESS);
  {
    auto registry = std::make_shared<CustomOpRegistry>();
    ScopedOpProtoLoadTxn txn(registry);
    OpProtoLedger::SetCurrentProvider(kFpA, kSoA);
    EXPECT_EQ(OperatorFactoryImpl::RegisterOperatorCreator("LedgerCreatorV1OverrideOp", om_creator), GRAPH_SUCCESS);
    const auto *entry = GetLedgerEntryForUt("LedgerCreatorV1OverrideOp", OpProtoMapKind::kCreatorV1);
    ASSERT_NE(entry, nullptr);
    EXPECT_TRUE(entry->overridden);
    EXPECT_TRUE(entry->has_backup);
  }
  EXPECT_TRUE(OperatorFactoryImpl::IsExistOp("LedgerCreatorV1OverrideOp"));
  EXPECT_FALSE(HasLedgerEntryForUt("LedgerCreatorV1OverrideOp"));
}

// 同一 op_type 的不同 map 由不同 provider 提供时，按 op_type 级规则产生冲突。
TEST_F(OpProtoLedgerUT, DifferentMapsFromDifferentProvidersConflict) {
  auto registry = std::make_shared<CustomOpRegistry>();
  {
    ScopedOpProtoLoadTxn txn(registry);
    OpProtoLedger::SetCurrentProvider(kFpA, kSoA);
    OpProtoLedger::OnRegistered("LedgerSkipOp", OpProtoMapKind::kCreatorV2, false);
    const auto *entry = GetLedgerEntryForUt("LedgerSkipOp");
    ASSERT_NE(entry, nullptr);
    ASSERT_EQ(entry->refcount, 1U);

    // 同事务切换提供者指纹后注册另一类 map。
    OpProtoLedger::SetCurrentProvider(kFpB, kSoB);
    OpProtoLedger::OnRegistered("LedgerSkipOp", OpProtoMapKind::kInferShape, false);

    ASSERT_EQ(txn.GetConflicts().size(), 1U);
    EXPECT_EQ(txn.GetConflicts()[0].map_kind, OpProtoMapKind::kCreatorV2);
    EXPECT_EQ(GetLedgerEntryForUt("LedgerSkipOp", OpProtoMapKind::kInferShape), nullptr);
  }
}

// 写前守门判定矩阵：OM 事务内允许接管无账本来源的 map；不同 provider 的活跃 map 仍拦截。
TEST_F(OpProtoLedgerUT, ShouldBlockRegisterMatrix) {
  // 无事务：恒放行（内置 OPP 初始化/在线注册/单算子路径）
  EXPECT_FALSE(OpProtoLedger::ShouldBlockRegister("LedgerMatrixNoTxnOp", OpProtoMapKind::kCreatorV2, false));
  EXPECT_FALSE(OpProtoLedger::ShouldBlockRegister("LedgerMatrixNoTxnOp", OpProtoMapKind::kCreatorV2, true));

  auto registry = std::make_shared<CustomOpRegistry>();
  ScopedOpProtoLoadTxn txn(registry);
  // 事务内无条目：新鲜写入和覆盖均放行，由调用方保存被覆盖值。
  EXPECT_FALSE(OpProtoLedger::ShouldBlockRegister("LedgerMatrixFreshOp", OpProtoMapKind::kCreatorV2, false));
  EXPECT_FALSE(OpProtoLedger::ShouldBlockRegister("LedgerMatrixOverwriteOp", OpProtoMapKind::kCreatorV2, true));
  EXPECT_TRUE(txn.GetConflicts().empty());

  // 同一提供者的活跃条目：新鲜（追加类别）/覆盖（等价值刷新）均放行
  OpProtoLedger::SetCurrentProvider(kFpA, kSoA);
  OpProtoLedger::OnRegistered("LedgerMatrixSameFpOp", OpProtoMapKind::kCreatorV2, false);
  EXPECT_FALSE(OpProtoLedger::ShouldBlockRegister("LedgerMatrixSameFpOp", OpProtoMapKind::kCreatorV2, false));
  EXPECT_FALSE(OpProtoLedger::ShouldBlockRegister("LedgerMatrixSameFpOp", OpProtoMapKind::kCreatorV2, true));

  // 异提供者活跃条目：新鲜/覆盖均拦截（同 op_type 冲突去重，仍各记 1 条）
  OpProtoLedger::SetCurrentProvider(kFpB, kSoB);
  EXPECT_TRUE(OpProtoLedger::ShouldBlockRegister("LedgerMatrixSameFpOp", OpProtoMapKind::kCreatorV2, false));
  EXPECT_TRUE(OpProtoLedger::ShouldBlockRegister("LedgerMatrixSameFpOp", OpProtoMapKind::kCreatorV2, true));
  ASSERT_EQ(txn.GetConflicts().size(), 1U);
  EXPECT_EQ(txn.GetConflicts()[0].op_type, std::string("LedgerMatrixSameFpOp"));
  EXPECT_EQ(txn.GetConflicts()[0].incumbent_fingerprint, std::string(kFpA));
  EXPECT_EQ(txn.GetConflicts()[0].challenger_fingerprint, std::string(kFpB));

  // 白盒构造无引用条目：新鲜写入和覆盖均放行。
  auto &entries = OpProtoLedger::GetInstance().entries_;
  OpProtoLedger::LedgerEntry stale_entry;
  stale_entry[OpProtoMapKind::kCreatorV2].provider_fingerprint = kFpA;
  stale_entry[OpProtoMapKind::kCreatorV2].provider_so_name = kSoA;
  (void)entries.emplace("LedgerMatrixStaleOp", stale_entry);
  EXPECT_TRUE(OpProtoLedger::ShouldBlockRegister("LedgerMatrixStaleOp", OpProtoMapKind::kCreatorV2, false));
  EXPECT_TRUE(OpProtoLedger::ShouldBlockRegister("LedgerMatrixStaleOp", OpProtoMapKind::kCreatorV2, true));

  EXPECT_EQ(txn.GetConflicts().size(), 2U);
}

// 不同 provider 写入同一 op_type 的不同 map 时，在写入前拦截。
TEST_F(OpProtoLedgerUT, FactoryFreshWriteDifferentMapBlocked) {
  dirty_ops_.push_back("LedgerGateResidueOp");
  auto reg1 = std::make_shared<CustomOpRegistry>();
  {
    ScopedOpProtoLoadTxn txn(reg1);
    OpProtoLedger::SetCurrentProvider(kFpA, kSoA);
    const OpCreatorV2 creator = [](const AscendString &) { return Operator(); };
    ASSERT_EQ(OperatorFactoryImpl::RegisterOperatorCreator("LedgerGateResidueOp", creator), GRAPH_SUCCESS);
  }
  auto reg2 = std::make_shared<CustomOpRegistry>();
  {
    ScopedOpProtoLoadTxn txn(reg2);
    OpProtoLedger::SetCurrentProvider(kFpB, kSoB);
    // infershape 槽位为空，但 op_type 已由其他 provider 提供，写入应被拦截。
    const InferShapeFunc infer = [](Operator &) { return GRAPH_SUCCESS; };
    EXPECT_EQ(OperatorFactoryImpl::RegisterInferShapeFunc("LedgerGateResidueOp", infer), GRAPH_FAILED);
    EXPECT_EQ(OperatorFactoryImpl::GetInferShapeFunc("LedgerGateResidueOp"), nullptr);
    EXPECT_EQ(txn.GetConflicts().size(), 1U);
  }

  // 两个 map 分别记录各自 provider。
  EXPECT_TRUE(OperatorFactoryImpl::IsExistOp("LedgerGateResidueOp"));
  const auto *creator_entry = GetLedgerEntryForUt("LedgerGateResidueOp");
  ASSERT_NE(creator_entry, nullptr);
  EXPECT_EQ(creator_entry->provider_fingerprint, std::string(kFpA));
  EXPECT_EQ(GetLedgerEntryForUt("LedgerGateResidueOp", OpProtoMapKind::kInferShape), nullptr);
}

// 内置 creator 被 OM 接管时，同一 op_type 的 InferShape 也可由同一 provider 注册。
TEST_F(OpProtoLedgerUT, FactoryFreshKindAlongsideOverriddenAllowed) {
  dirty_ops_.push_back("LedgerGateExtOp");
  // 无事务注册 creator（模拟内置/在线来源，无账本条目）
  const OpCreatorV2 creator = [](const AscendString &) { return Operator(); };
  ASSERT_EQ(OperatorFactoryImpl::RegisterOperatorCreator("LedgerGateExtOp", creator), GRAPH_SUCCESS);

  auto registry = std::make_shared<CustomOpRegistry>();
  {
    ScopedOpProtoLoadTxn txn(registry);
    OpProtoLedger::SetCurrentProvider(kFpA, kSoA);
    // creator 槽位被占用，OM 接管并记录恢复信息。
    EXPECT_EQ(OperatorFactoryImpl::RegisterOperatorCreator("LedgerGateExtOp", creator), GRAPH_SUCCESS);
    const auto *entry = GetLedgerEntryForUt("LedgerGateExtOp");
    ASSERT_NE(entry, nullptr);
    EXPECT_TRUE(entry->overridden);
    EXPECT_EQ(entry->refcount, 1U);

    // infershape 槽位为空，独立创建并归当前 SO 所有。
    const InferShapeFunc infer = [](Operator &) { return GRAPH_SUCCESS; };
    EXPECT_EQ(OperatorFactoryImpl::RegisterInferShapeFunc("LedgerGateExtOp", infer), GRAPH_SUCCESS);
    EXPECT_NE(OperatorFactoryImpl::GetInferShapeFunc("LedgerGateExtOp"), nullptr);
    EXPECT_TRUE(txn.GetConflicts().empty());
    const auto *infer_entry = GetLedgerEntryForUt("LedgerGateExtOp", OpProtoMapKind::kInferShape);
    ASSERT_NE(infer_entry, nullptr);
    EXPECT_EQ(infer_entry->provider_fingerprint, std::string(kFpA));
  }
}

// 集成回归：事务内 override 无账本来源的槽位 → 覆盖并在卸载时恢复
TEST_F(OpProtoLedgerUT, FactoryOverrideUnderTxnRestoresBuiltinValue) {
  dirty_ops_.push_back("LedgerGateOvNoEntryOp");
  int32_t incumbent_calls = 0;
  int32_t challenger_calls = 0;
  {
    // 无事务注册（模拟在线来源）：无账本条目
    const OpCreatorV2 incumbent = [&incumbent_calls](const AscendString &) {
      ++incumbent_calls;
      return Operator();
    };
    ASSERT_EQ(OperatorFactoryImpl::RegisterOperatorCreator("LedgerGateOvNoEntryOp", incumbent), GRAPH_SUCCESS);
  }
  {
    auto registry = std::make_shared<CustomOpRegistry>();
    ScopedOpProtoLoadTxn txn(registry);
    OpProtoLedger::SetCurrentProvider(kFpA, kSoA);
    OperatorFactoryImpl::SetRegisterOverridable(true);
    const OpCreatorV2 challenger = [&challenger_calls](const AscendString &) {
      ++challenger_calls;
      return Operator();
    };
    EXPECT_EQ(OperatorFactoryImpl::RegisterOperatorCreator("LedgerGateOvNoEntryOp", challenger), GRAPH_SUCCESS);
    OperatorFactoryImpl::SetRegisterOverridable(false);
    EXPECT_TRUE(txn.GetConflicts().empty());

    (void)OperatorFactoryImpl::CreateOperator("", "LedgerGateOvNoEntryOp");
    EXPECT_EQ(incumbent_calls, 0);
    EXPECT_EQ(challenger_calls, 1);
  }
}

// 集成回归：事务内 override 异提供者活跃条目 → 拦截，条目与在位值均不受影响
TEST_F(OpProtoLedgerUT, FactoryOverrideCrossProviderUnderTxnBlocked) {
  dirty_ops_.push_back("LedgerGateOvDiffFpOp");
  auto reg1 = std::make_shared<CustomOpRegistry>();
  {
    ScopedOpProtoLoadTxn txn(reg1);
    OpProtoLedger::SetCurrentProvider(kFpA, kSoA);
    const OpCreatorV2 creator = [](const AscendString &) { return Operator(); };
    ASSERT_EQ(OperatorFactoryImpl::RegisterOperatorCreator("LedgerGateOvDiffFpOp", creator), GRAPH_SUCCESS);
  }
  {
    auto reg2 = std::make_shared<CustomOpRegistry>();
    ScopedOpProtoLoadTxn txn(reg2);
    OpProtoLedger::SetCurrentProvider(kFpB, kSoB);
    OperatorFactoryImpl::SetRegisterOverridable(true);
    const OpCreatorV2 challenger = [](const AscendString &) { return Operator(); };
    EXPECT_EQ(OperatorFactoryImpl::RegisterOperatorCreator("LedgerGateOvDiffFpOp", challenger), GRAPH_FAILED);
    OperatorFactoryImpl::SetRegisterOverridable(false);
    ASSERT_EQ(txn.GetConflicts().size(), 1U);
    EXPECT_EQ(txn.GetConflicts()[0].incumbent_fingerprint, std::string(kFpA));
  }
  const auto *entry = GetLedgerEntryForUt("LedgerGateOvDiffFpOp");
  ASSERT_NE(entry, nullptr);
  EXPECT_EQ(entry->provider_fingerprint, std::string(kFpA));  // 在位提供者不变
  EXPECT_EQ(entry->refcount, 1U);
}

// 集成回归：事务内 override 同提供者条目 → 放行，引用归零时可清理该 map
TEST_F(OpProtoLedgerUT, FactoryOverrideSameProviderUnderTxnAllowed) {
  dirty_ops_.push_back("LedgerGateOvSameFpOp");
  auto registry = std::make_shared<CustomOpRegistry>();
  {
    ScopedOpProtoLoadTxn txn(registry);
    OpProtoLedger::SetCurrentProvider(kFpA, kSoA);
    const OpCreatorV2 first = [](const AscendString &) { return Operator(); };
    ASSERT_EQ(OperatorFactoryImpl::RegisterOperatorCreator("LedgerGateOvSameFpOp", first), GRAPH_SUCCESS);
    OperatorFactoryImpl::SetRegisterOverridable(true);
    const OpCreatorV2 second = [](const AscendString &) { return Operator(); };
    EXPECT_EQ(OperatorFactoryImpl::RegisterOperatorCreator("LedgerGateOvSameFpOp", second), GRAPH_SUCCESS);
    OperatorFactoryImpl::SetRegisterOverridable(false);
    EXPECT_TRUE(txn.GetConflicts().empty());
  }
}

// 静默期延迟擦除：加载事务 in-flight 期间回放整体推迟（条目/全局 map/SO token 原样保活），
// 最后一个事务析构时统一清扫，终态与立即回放等价；多批被推迟的 claim 一并处理
TEST_F(OpProtoLedgerUT, ReleaseClaimsDeferredWhileTxnActive) {
  dirty_ops_.push_back("LedgerDeferOp");
  auto token = std::make_shared<int32_t>(1);
  auto reg1 = std::make_shared<CustomOpRegistry>();
  {
    ScopedOpProtoLoadTxn txn(reg1);
    OpProtoLedger::SetCurrentProvider(kFpA, kSoA);
    const OpCreatorV2 creator = [](const AscendString &) { return Operator(); };
    ASSERT_EQ(OperatorFactoryImpl::RegisterOperatorCreator("LedgerDeferOp", creator), GRAPH_SUCCESS);
    OpProtoLedger::AttachProviderHandle(kFpA, std::shared_ptr<void>(token));
  }
  // 第二个借用方（模拟更早加载的另一模型），其 claim 独立成批推迟
  auto reg0 = std::make_shared<CustomOpRegistry>();
  {
    ScopedOpProtoLoadTxn txn(reg0);
    OpProtoLedger::SetCurrentProvider(kFpA, kSoA);
    OpProtoLedger::ClaimProviderMaps("LedgerDeferOp");
  }

  auto reg2 = std::make_shared<CustomOpRegistry>();
  {
    ScopedOpProtoLoadTxn txn(reg2);  // 模拟并发加载 in-flight
    reg0.reset();                    // 卸载回放批次 1：推迟
    reg1.reset();                    // 卸载回放批次 2：推迟
    EXPECT_TRUE(OperatorFactoryImpl::IsExistOp("LedgerDeferOp"));
    const auto *entry = GetLedgerEntryForUt("LedgerDeferOp");
    ASSERT_NE(entry, nullptr);
    EXPECT_EQ(entry->refcount, 2U);   // 两批 claim 均未扣减
    EXPECT_EQ(token.use_count(), 2);  // SO 保活 token 未释放
    EXPECT_EQ(OpProtoLedger::GetInstance().active_txn_count_, 1U);
    EXPECT_EQ(OpProtoLedger::GetInstance().pending_releases_.size(), 2U);
  }  // 最后一个事务析构：静默期清扫两批 claim → 归零清理
  EXPECT_FALSE(OperatorFactoryImpl::IsExistOp("LedgerDeferOp"));
  EXPECT_FALSE(HasLedgerEntryForUt("LedgerDeferOp"));
  EXPECT_EQ(token.use_count(), 1);
  EXPECT_TRUE(OpProtoLedger::GetInstance().pending_releases_.empty());
  EXPECT_EQ(OpProtoLedger::GetInstance().active_txn_count_, 0U);
}

// 推迟窗口内同指纹借用：计数正确（推迟未扣减 + 新借用），清扫后条目归新借用方、不误删
TEST_F(OpProtoLedgerUT, BorrowDuringDeferralWindowKeepsCountsCorrect) {
  dirty_ops_.push_back("LedgerDeferBorrowOp");
  auto reg1 = std::make_shared<CustomOpRegistry>();
  {
    ScopedOpProtoLoadTxn txn(reg1);
    OpProtoLedger::SetCurrentProvider(kFpA, kSoA);
    const OpCreatorV2 creator = [](const AscendString &) { return Operator(); };
    ASSERT_EQ(OperatorFactoryImpl::RegisterOperatorCreator("LedgerDeferBorrowOp", creator), GRAPH_SUCCESS);
  }
  auto reg2 = std::make_shared<CustomOpRegistry>();
  {
    ScopedOpProtoLoadTxn txn(reg2);
    OpProtoLedger::SetCurrentProvider(kFpA, kSoA);
    reg1.reset();                                             // 卸载 A：回放推迟，条目仍活跃
    OpProtoLedger::ClaimProviderMaps("LedgerDeferBorrowOp");  // 同指纹借用（模拟共享同内容的新模型）
    const auto *entry = GetLedgerEntryForUt("LedgerDeferBorrowOp");
    ASSERT_NE(entry, nullptr);
    EXPECT_EQ(entry->refcount, 2U);  // 1（推迟未扣减）+ 1（新借用）
  }  // 事务析构：先提交 reg2 借用 claim，再清扫 reg1 推迟 claim → 2→1
  const auto *entry = GetLedgerEntryForUt("LedgerDeferBorrowOp");
  ASSERT_NE(entry, nullptr);
  EXPECT_EQ(entry->refcount, 1U);
  EXPECT_TRUE(OperatorFactoryImpl::IsExistOp("LedgerDeferBorrowOp"));  // 未被误删
  reg2.reset();                                                        // 最后借用方卸载 → 归零清理
  EXPECT_FALSE(OperatorFactoryImpl::IsExistOp("LedgerDeferBorrowOp"));
  EXPECT_FALSE(HasLedgerEntryForUt("LedgerDeferBorrowOp"));
}

// 推迟窗口内异指纹加载到不同 map：按 op_type 级 provider 规则拦截。
TEST_F(OpProtoLedgerUT, DifferentFingerprintDuringDeferralWindowConflicts) {
  dirty_ops_.push_back("LedgerDeferConflictOp");
  auto reg1 = std::make_shared<CustomOpRegistry>();
  {
    ScopedOpProtoLoadTxn txn(reg1);
    OpProtoLedger::SetCurrentProvider(kFpA, kSoA);
    const OpCreatorV2 creator = [](const AscendString &) { return Operator(); };
    ASSERT_EQ(OperatorFactoryImpl::RegisterOperatorCreator("LedgerDeferConflictOp", creator), GRAPH_SUCCESS);
  }
  auto reg2 = std::make_shared<CustomOpRegistry>();
  {
    ScopedOpProtoLoadTxn txn(reg2);
    OpProtoLedger::SetCurrentProvider(kFpB, kSoB);
    reg1.reset();  // 卸载 A：回放推迟 → 条目 fp_a 仍活跃
    const InferShapeFunc infer = [](Operator &) { return GRAPH_SUCCESS; };
    EXPECT_EQ(OperatorFactoryImpl::RegisterInferShapeFunc("LedgerDeferConflictOp", infer), GRAPH_FAILED);
    EXPECT_EQ(OperatorFactoryImpl::GetInferShapeFunc("LedgerDeferConflictOp"), nullptr);
    ASSERT_EQ(txn.GetConflicts().size(), 1U);
    EXPECT_EQ(txn.GetConflicts()[0].incumbent_fingerprint, std::string(kFpA));
  }  // 静默期清扫：fp_a 条目归零清理
  reg2.reset();
  EXPECT_FALSE(OperatorFactoryImpl::IsExistOp("LedgerDeferConflictOp"));
  EXPECT_FALSE(HasLedgerEntryForUt("LedgerDeferConflictOp"));
}

// 真实双线程（栅栏同步）：加载事务窗口内的卸载回放必然推迟（计数在 mu_ 内先于窗口可见），
// join 后终态与串行一致：无误删、计数归零
TEST_F(OpProtoLedgerUT, ConcurrentReleaseDefersToTxnQuiescence) {
  dirty_ops_.push_back("LedgerRaceOp");
  auto token = std::make_shared<int32_t>(1);
  auto reg1 = std::make_shared<CustomOpRegistry>();
  {
    ScopedOpProtoLoadTxn txn(reg1);
    OpProtoLedger::SetCurrentProvider(kFpA, kSoA);
    const OpCreatorV2 creator = [](const AscendString &) { return Operator(); };
    ASSERT_EQ(OperatorFactoryImpl::RegisterOperatorCreator("LedgerRaceOp", creator), GRAPH_SUCCESS);
    OpProtoLedger::AttachProviderHandle(kFpA, std::shared_ptr<void>(token));
  }
  auto reg2 = std::make_shared<CustomOpRegistry>();
  std::atomic<bool> txn_open{false};
  std::atomic<bool> release_done{false};
  auto loader = std::thread([&]() {
    ScopedOpProtoLoadTxn txn(reg2);
    OpProtoLedger::SetCurrentProvider(kFpA, kSoA);
    txn_open.store(true);
    while (!release_done.load()) {
      std::this_thread::yield();
    }
    OpProtoLedger::ClaimProviderMaps("LedgerRaceOp");  // 窗口内借用：必见完整条目
  });
  auto unloader = std::thread([&]() {
    while (!txn_open.load()) {
      std::this_thread::yield();
    }
    reg1.reset();  // 窗口内回放：计数已可见 → 必然推迟而非交错擦除
    release_done.store(true);
  });
  loader.join();
  unloader.join();

  // 确定性终态：推迟的 claim 已被 loader 线程的事务析构清扫，借用 claim 归 reg2
  EXPECT_TRUE(OperatorFactoryImpl::IsExistOp("LedgerRaceOp"));
  const auto *entry = GetLedgerEntryForUt("LedgerRaceOp");
  ASSERT_NE(entry, nullptr);
  EXPECT_EQ(entry->refcount, 1U);
  EXPECT_EQ(OpProtoLedger::GetInstance().active_txn_count_, 0U);
  EXPECT_TRUE(OpProtoLedger::GetInstance().pending_releases_.empty());
  EXPECT_EQ(token.use_count(), 2);
  reg2.reset();  // 最后借用方卸载 → 归零清理
  EXPECT_FALSE(OperatorFactoryImpl::IsExistOp("LedgerRaceOp"));
  EXPECT_EQ(token.use_count(), 1);
}
