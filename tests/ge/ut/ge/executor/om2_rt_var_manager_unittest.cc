/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include <vector>
#include "runtime/om2/om2_rt_var_manager.h"
#include "framework/common/gert_model_data_utils.h"
#include "rt_external_mem.h"
#include "common/ge_common/error_codes_define.h"
#include "depends/ascendcl/src/ascendcl_stub.h"

namespace gert {
namespace {

class FreeRecordingAclRuntimeStub : public ge::AclRuntimeStub {
 public:
  aclError aclrtFree(void *devPtr) override {
    freed_ptrs.push_back(devPtr);
    return ge::AclRuntimeStub::aclrtFree(devPtr);
  }

  std::vector<void *> freed_ptrs;
};

class AclRuntimeStubGuard {
 public:
  explicit AclRuntimeStubGuard(ge::AclRuntimeStub *stub) : stub_(stub) {
    ge::AclRuntimeStub::Install(stub_);
  }
  ~AclRuntimeStubGuard() {
    ge::AclRuntimeStub::UnInstall(stub_);
  }

 private:
  ge::AclRuntimeStub *stub_;
};

class Om2RTVarManagerTest : public testing::Test {
 protected:
  RTVarEntry MakeVarEntry(const std::string &name, const std::string &op_type, uint64_t size) {
    RTVarEntry entry;
    entry.var_name = gert::GertMakeStr(name);
    entry.op_type = gert::GertMakeStr(op_type);
    entry.size = size;
    entry.memory_type = RT_MEMORY_HBM;
    gert::GertTensorDesc desc;
    desc.format = ge::FORMAT_ND;
    desc.data_type = ge::DT_FLOAT;
    entry.var_key = gert::GertMakeStr(RTVarBuildKey(name, desc));
    entry.tensor_desc = std::move(desc);
    return entry;
  }

  std::vector<RTVarEntry> MakeEntries(RTVarEntry entry) {
    std::vector<RTVarEntry> entries;
    RTVarAddEntry(entries, std::move(entry));
    return entries;
  }
};

TEST_F(Om2RTVarManagerTest, InitMergesEntries) {
  Om2RTVarManager mgr;
  auto e1 = MakeVarEntry("v1", "VARIABLE", 1024);
  auto resource = MakeEntries(std::move(e1));
  ASSERT_EQ(mgr.Init(resource), ge::SUCCESS);
  EXPECT_NE(mgr.GetVarResource().GetEntryByName("v1"), nullptr);
}

TEST_F(Om2RTVarManagerTest, InitSkipsDuplicateKeys) {
  Om2RTVarManager mgr;
  auto e1 = MakeVarEntry("v1", "VARIABLE", 1024);
  auto r1 = MakeEntries(std::move(e1));
  ASSERT_EQ(mgr.Init(r1), ge::SUCCESS);

  auto e2 = MakeVarEntry("v1", "VARIABLE", 1024);
  auto r2 = MakeEntries(std::move(e2));
  ASSERT_EQ(mgr.Init(r2), ge::SUCCESS);
  EXPECT_EQ(mgr.GetVarResource().GetAllEntries().size(), 1U);
}

TEST_F(Om2RTVarManagerTest, GetVarDevAddrNotFound) {
  Om2RTVarManager mgr;
  void *addr = nullptr;
  EXPECT_NE(mgr.GetVarDevAddr("nonexistent", 0, addr), ge::SUCCESS);
}

TEST_F(Om2RTVarManagerTest, ConstPlaceHolderUsesExternAddr) {
  Om2RTVarManager mgr;
  auto entry = MakeVarEntry("ph1", "CONSTPLACEHOLDER", 2048);
  uint8_t fake_extern_addr[2048];
  entry.extern_dev_addr = fake_extern_addr;
  auto resource = MakeEntries(std::move(entry));
  ASSERT_EQ(mgr.Init(resource), ge::SUCCESS);

  void *addr = nullptr;
  ASSERT_EQ(mgr.GetVarDevAddr("ph1", 0, addr), ge::SUCCESS);
  EXPECT_EQ(addr, fake_extern_addr);
}

TEST_F(Om2RTVarManagerTest, MultiDeviceIsolation) {
  Om2RTVarManager mgr;
  auto entry = MakeVarEntry("v1", "VARIABLE", 512);
  auto resource = MakeEntries(std::move(entry));
  ASSERT_EQ(mgr.Init(resource), ge::SUCCESS);

  void *addr0 = nullptr;
  void *addr1 = nullptr;
  ASSERT_EQ(mgr.GetVarDevAddr("v1", 0, addr0), ge::SUCCESS);
  ASSERT_EQ(mgr.GetVarDevAddr("v1", 1, addr1), ge::SUCCESS);
  EXPECT_NE(addr0, nullptr);
  EXPECT_NE(addr1, nullptr);
  EXPECT_NE(addr0, addr1);
}

TEST_F(Om2RTVarManagerTest, SameDeviceReturnsSameAddr) {
  Om2RTVarManager mgr;
  auto entry = MakeVarEntry("v1", "VARIABLE", 512);
  auto resource = MakeEntries(std::move(entry));
  ASSERT_EQ(mgr.Init(resource), ge::SUCCESS);

  void *addr1 = nullptr;
  void *addr2 = nullptr;
  ASSERT_EQ(mgr.GetVarDevAddr("v1", 0, addr1), ge::SUCCESS);
  ASSERT_EQ(mgr.GetVarDevAddr("v1", 0, addr2), ge::SUCCESS);
  EXPECT_EQ(addr1, addr2);
}

TEST_F(Om2RTVarManagerTest, LegacyGetOrCreateVarAddr) {
  Om2RTVarManager mgr;
  void *addr = nullptr;
  ASSERT_EQ(mgr.GetOrCreateVarAddr("legacy_key", 0, 256, addr), ge::SUCCESS);
  EXPECT_NE(addr, nullptr);

  void *addr2 = nullptr;
  ASSERT_EQ(mgr.GetOrCreateVarAddr("legacy_key", 0, 256, addr2), ge::SUCCESS);
  EXPECT_EQ(addr, addr2);
}

TEST_F(Om2RTVarManagerTest, LegacyTryGetVarAddr) {
  Om2RTVarManager mgr;
  void *addr = nullptr;
  EXPECT_FALSE(mgr.TryGetVarAddr("missing", 0, addr));

  void *created = nullptr;
  ASSERT_EQ(mgr.GetOrCreateVarAddr("key1", 0, 128, created), ge::SUCCESS);
  EXPECT_TRUE(mgr.TryGetVarAddr("key1", 0, addr));
  EXPECT_EQ(addr, created);
}

TEST_F(Om2RTVarManagerTest, FinalizeFreesMemory) {
  auto mgr = std::make_unique<Om2RTVarManager>();
  auto entry = MakeVarEntry("v1", "VARIABLE", 512);
  auto resource = MakeEntries(std::move(entry));
  ASSERT_EQ(mgr->Init(resource), ge::SUCCESS);

  void *addr = nullptr;
  ASSERT_EQ(mgr->GetVarDevAddr("v1", 0, addr), ge::SUCCESS);
  EXPECT_NE(addr, nullptr);
  mgr.reset();
}

TEST_F(Om2RTVarManagerTest, PoolGetAndRemove) {
  auto &pool = Om2RTVarManagerPool::Instance();
  auto mgr = pool.GetManager(42);
  ASSERT_NE(mgr, nullptr);
  auto mgr2 = pool.GetManager(42);
  EXPECT_EQ(mgr.get(), mgr2.get());
  pool.RemoveManager(42);
}

TEST_F(Om2RTVarManagerTest, TransAllVarDataSkipsNoTransRoad) {
  Om2RTVarManager mgr;
  auto entry = MakeVarEntry("v1", "VARIABLE", 512);
  auto resource = MakeEntries(std::move(entry));
  ASSERT_EQ(mgr.Init(resource), ge::SUCCESS);
  ASSERT_EQ(mgr.TransAllVarData({"v1"}, 0, 1), ge::SUCCESS);
}

TEST_F(Om2RTVarManagerTest, CopyVarDataSkipsNoCopyInfo) {
  Om2RTVarManager mgr;
  auto entry = MakeVarEntry("v1", "VARIABLE", 512);
  auto resource = MakeEntries(std::move(entry));
  ASSERT_EQ(mgr.Init(resource), ge::SUCCESS);
  ASSERT_EQ(mgr.CopyVarData({"v1"}, 0), ge::SUCCESS);
}

TEST_F(Om2RTVarManagerTest, InitRejectsInconsistentSharedVarSize) {
  Om2RTVarManager mgr;
  auto e1 = MakeVarEntry("v1", "VARIABLE", 1024);
  auto r1 = MakeEntries(std::move(e1));
  ASSERT_EQ(mgr.Init(r1), ge::SUCCESS);

  // 同 var_key（name + tensor_desc 一致）但 size 不一致，Bundle 共享变量校验必须拒绝
  auto e2 = MakeVarEntry("v1", "VARIABLE", 2048);
  auto r2 = MakeEntries(std::move(e2));
  EXPECT_EQ(mgr.Init(r2), ge::PARAM_INVALID);
  EXPECT_EQ(mgr.GetVarResource().GetAllEntries().size(), 1U);
}

TEST_F(Om2RTVarManagerTest, InitRejectsInconsistentSharedVarLogicAddr) {
  Om2RTVarManager mgr;
  auto e1 = MakeVarEntry("v1", "VARIABLE", 1024);
  e1.logic_addr = kMemoryVarLogicBase + 0x1000U;
  auto r1 = MakeEntries(std::move(e1));
  ASSERT_EQ(mgr.Init(r1), ge::SUCCESS);

  // 同 var_key 但逻辑地址不一致，跨子模型共享语义被破坏，必须拒绝
  auto e2 = MakeVarEntry("v1", "VARIABLE", 1024);
  e2.logic_addr = kMemoryVarLogicBase + 0x2000U;
  auto r2 = MakeEntries(std::move(e2));
  EXPECT_EQ(mgr.Init(r2), ge::PARAM_INVALID);
}

TEST_F(Om2RTVarManagerTest, InitAcceptsConsistentSharedVar) {
  Om2RTVarManager mgr;
  auto e1 = MakeVarEntry("v1", "VARIABLE", 1024);
  e1.logic_addr = kMemoryVarLogicBase + 0x1000U;
  auto r1 = MakeEntries(std::move(e1));
  ASSERT_EQ(mgr.Init(r1), ge::SUCCESS);

  // 各字段一致的共享变量重复 Init 幂等（Bundle 多子模型共享场景）
  auto e2 = MakeVarEntry("v1", "VARIABLE", 1024);
  e2.logic_addr = kMemoryVarLogicBase + 0x1000U;
  auto r2 = MakeEntries(std::move(e2));
  EXPECT_EQ(mgr.Init(r2), ge::SUCCESS);
  EXPECT_EQ(mgr.GetVarResource().GetAllEntries().size(), 1U);
}

TEST_F(Om2RTVarManagerTest, GetVarDevAddrUsesExternalArena) {
  Om2RTVarManager mgr;
  auto entry = MakeVarEntry("v1", "VARIABLE", 512);
  entry.logic_addr = kMemoryVarLogicBase + 64U;
  auto resource = MakeEntries(std::move(entry));
  std::vector<uint8_t> arena(4096U, 0U);
  ASSERT_EQ(mgr.Init(resource, arena.data(), arena.size()), ge::SUCCESS);

  void *addr = nullptr;
  ASSERT_EQ(mgr.GetVarDevAddr("v1", 0, addr), ge::SUCCESS);
  EXPECT_EQ(addr, static_cast<void *>(arena.data() + 64U));

  void *addr_again = nullptr;
  ASSERT_EQ(mgr.GetVarDevAddr("v1", 0, addr_again), ge::SUCCESS);
  EXPECT_EQ(addr_again, addr);
}

TEST_F(Om2RTVarManagerTest, GetVarDevAddrExternalArenaInvalidLogicAddrFails) {
  std::vector<uint8_t> arena(1024U, 0U);
  {
    // offset + size 超出 arena 范围
    Om2RTVarManager mgr;
    auto entry = MakeVarEntry("v1", "VARIABLE", 512);
    entry.logic_addr = kMemoryVarLogicBase + 600U;
    auto resource = MakeEntries(std::move(entry));
    ASSERT_EQ(mgr.Init(resource, arena.data(), arena.size()), ge::SUCCESS);
    void *addr = nullptr;
    EXPECT_EQ(mgr.GetVarDevAddr("v1", 0, addr), ge::FAILED);
  }
  {
    // logic_addr 低于逻辑基址
    Om2RTVarManager mgr;
    auto entry = MakeVarEntry("v2", "VARIABLE", 512);
    entry.logic_addr = kMemoryVarLogicBase - 1U;
    auto resource = MakeEntries(std::move(entry));
    ASSERT_EQ(mgr.Init(resource, arena.data(), arena.size()), ge::SUCCESS);
    void *addr = nullptr;
    EXPECT_EQ(mgr.GetVarDevAddr("v2", 0, addr), ge::FAILED);
  }
}

TEST_F(Om2RTVarManagerTest, GetVarDevAddrInitDataOversizeFails) {
  Om2RTVarManager mgr;
  auto entry = MakeVarEntry("v1", "VARIABLE", 4);
  entry.init_data = std::vector<uint8_t>(8U, 0xFFU);
  auto resource = MakeEntries(std::move(entry));
  ASSERT_EQ(mgr.Init(resource), ge::SUCCESS);

  void *addr = nullptr;
  EXPECT_EQ(mgr.GetVarDevAddr("v1", 0, addr), ge::FAILED);
}

TEST_F(Om2RTVarManagerTest, FinalizeSkipsExternalArenaAddrs) {
  FreeRecordingAclRuntimeStub runtime_stub;
  AclRuntimeStubGuard runtime_stub_guard(&runtime_stub);
  std::vector<uint8_t> arena(4096U, 0U);
  uint8_t extern_addr[16] = {0};
  {
    Om2RTVarManager mgr;
    auto arena_entry = MakeVarEntry("arena_var", "VARIABLE", 512);
    arena_entry.logic_addr = kMemoryVarLogicBase + 128U;
    auto extern_entry = MakeVarEntry("extern_var", "CONSTPLACEHOLDER", 16);
    extern_entry.extern_dev_addr = extern_addr;
    std::vector<RTVarEntry> resource;
    RTVarAddEntry(resource, std::move(arena_entry));
    RTVarAddEntry(resource, std::move(extern_entry));
    ASSERT_EQ(mgr.Init(resource, arena.data(), arena.size()), ge::SUCCESS);

    void *arena_mapped_addr = nullptr;
    void *extern_mapped_addr = nullptr;
    ASSERT_EQ(mgr.GetVarDevAddr("arena_var", 0, arena_mapped_addr), ge::SUCCESS);
    ASSERT_EQ(mgr.GetVarDevAddr("extern_var", 0, extern_mapped_addr), ge::SUCCESS);
    EXPECT_EQ(arena_mapped_addr, static_cast<void *>(arena.data() + 128U));
    EXPECT_EQ(extern_mapped_addr, static_cast<void *>(extern_addr));
  }
  // 外部 arena 借用内存与 extern_dev_addr 均不归 GE 所有，Finalize 不得释放
  for (const auto freed : runtime_stub.freed_ptrs) {
    EXPECT_NE(freed, static_cast<void *>(arena.data() + 128U));
    EXPECT_NE(freed, static_cast<void *>(extern_addr));
  }
}

}  // namespace
}  // namespace gert
