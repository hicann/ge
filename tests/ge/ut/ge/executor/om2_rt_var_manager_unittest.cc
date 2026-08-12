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
#include "runtime/om2/om2_rt_var_manager.h"
#include "rt_external_mem.h"

namespace gert {
namespace {

class Om2RTVarManagerTest : public testing::Test {
 protected:
  RTVarEntry MakeVarEntry(const std::string &name, const std::string &op_type, uint64_t size) {
    RTVarEntry entry;
    entry.var_name = name;
    entry.op_type = op_type;
    entry.size = size;
    entry.memory_type = RT_MEMORY_HBM;
    ge::Om2TensorDesc desc;
    desc.SetFormat(ge::FORMAT_ND);
    desc.SetDataType(ge::DT_FLOAT);
    entry.tensor_desc = desc;
    entry.var_key = RTVarResource::BuildVarKey(name, desc);
    return entry;
  }

  RTVarResource MakeResource(std::vector<RTVarEntry> entries) {
    RTVarResource resource;
    for (auto &e : entries) {
      resource.AddEntry(std::move(e));
    }
    return resource;
  }
};

TEST_F(Om2RTVarManagerTest, InitMergesEntries) {
  Om2RTVarManager mgr;
  auto e1 = MakeVarEntry("v1", "VARIABLE", 1024);
  auto resource = MakeResource({std::move(e1)});
  ASSERT_EQ(mgr.Init(resource), ge::SUCCESS);
  ASSERT_NE(mgr.GetVarResource(), nullptr);
  EXPECT_NE(mgr.GetVarResource()->GetEntryByName("v1"), nullptr);
}

TEST_F(Om2RTVarManagerTest, InitSkipsDuplicateKeys) {
  Om2RTVarManager mgr;
  auto e1 = MakeVarEntry("v1", "VARIABLE", 1024);
  auto r1 = MakeResource({std::move(e1)});
  ASSERT_EQ(mgr.Init(r1), ge::SUCCESS);

  auto e2 = MakeVarEntry("v1", "VARIABLE", 1024);
  auto r2 = MakeResource({std::move(e2)});
  ASSERT_EQ(mgr.Init(r2), ge::SUCCESS);
  EXPECT_EQ(mgr.GetVarResource()->GetAllEntries().size(), 1U);
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
  auto resource = MakeResource({std::move(entry)});
  ASSERT_EQ(mgr.Init(resource), ge::SUCCESS);

  void *addr = nullptr;
  ASSERT_EQ(mgr.GetVarDevAddr("ph1", 0, addr), ge::SUCCESS);
  EXPECT_EQ(addr, fake_extern_addr);
}

TEST_F(Om2RTVarManagerTest, MultiDeviceIsolation) {
  Om2RTVarManager mgr;
  auto entry = MakeVarEntry("v1", "VARIABLE", 512);
  auto resource = MakeResource({std::move(entry)});
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
  auto resource = MakeResource({std::move(entry)});
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
  auto resource = MakeResource({std::move(entry)});
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
  auto resource = MakeResource({std::move(entry)});
  ASSERT_EQ(mgr.Init(resource), ge::SUCCESS);
  ASSERT_EQ(mgr.TransAllVarData({"v1"}, 0, 1), ge::SUCCESS);
}

TEST_F(Om2RTVarManagerTest, CopyVarDataSkipsNoCopyInfo) {
  Om2RTVarManager mgr;
  auto entry = MakeVarEntry("v1", "VARIABLE", 512);
  auto resource = MakeResource({std::move(entry)});
  ASSERT_EQ(mgr.Init(resource), ge::SUCCESS);
  ASSERT_EQ(mgr.CopyVarData({"v1"}, 0), ge::SUCCESS);
}

}  // namespace
}  // namespace gert
