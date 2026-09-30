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
#include <memory>

#include "graph/load/model_manager/attached_stream_collection.h"
#include "graph/ascend_string.h"
#include "depends/runtime/src/runtime_stub.h"
#include "depends/ascendcl/src/ascendcl_stub.h"

namespace ge {
namespace {
constexpr rtError_t kRtErrorForTest = 1;

class RuntimeForAttachedStreamTest final : public RuntimeStub {
 public:
  rtError_t rtStreamCreateWithFlags(rtStream_t *stream, int32_t, uint32_t) override {
    ++create_count_;
    if (create_fail) {
      return kRtErrorForTest;
    }
    *stream = reinterpret_cast<rtStream_t>(++next_stream_);
    return RT_ERROR_NONE;
  }

  int create_count_{0};
  bool create_fail{false};

 private:
  uintptr_t next_stream_{0x1000U};
};

class AclForAttachedStreamTest final : public AclRuntimeStub {
 public:
  aclError aclmdlRIBindStream(aclmdlRI, aclrtStream stream, uint32_t flag) override {
    bound_streams.push_back(stream);
    bind_flags.push_back(flag);
    events.push_back("bind");
    return bind_success ? ACL_SUCCESS : ACL_ERROR_FAILURE;
  }
  aclError aclrtSynchronizeStream(aclrtStream stream) override {
    synced_streams.push_back(stream);
    events.push_back("sync");
    return sync_success ? ACL_SUCCESS : ACL_ERROR_RT_FAILURE;
  }
  aclError aclmdlRIUnbindStream(aclmdlRI, aclrtStream stream) override {
    unbound_streams.push_back(stream);
    events.push_back("unbind");
    return ACL_SUCCESS;
  }
  aclError aclrtDestroyStream(aclrtStream stream) override {
    destroyed_streams.push_back(stream);
    events.push_back("destroy");
    return ACL_SUCCESS;
  }

  std::vector<aclrtStream> bound_streams;
  std::vector<uint32_t> bind_flags;
  std::vector<aclrtStream> synced_streams;
  std::vector<aclrtStream> unbound_streams;
  std::vector<aclrtStream> destroyed_streams;
  std::vector<std::string> events;
  bool bind_success{true};
  bool sync_success{true};
};

TEST(AttachedStreamCollectionTest, RejectsEmptyKeysBeforeRuntimeCall) {
  AttachedStreamCollection collection(nullptr, RT_STREAM_PRIORITY_DEFAULT, RT_STREAM_DEFAULT);
  EXPECT_EQ(collection.RequestAttachedStream(AscendString()), nullptr);
  EXPECT_EQ(collection.RequestAttachedStream(AscendString("")), nullptr);
}

TEST(AttachedStreamCollectionTest, AcceptsFrameworkReservedPrefixKey) {
  auto runtime = std::make_shared<RuntimeForAttachedStreamTest>();
  auto acl = std::make_shared<AclForAttachedStreamTest>();
  RuntimeStub::SetInstance(runtime);
  AclRuntimeStub::SetInstance(acl);

  AttachedStreamCollection collection(reinterpret_cast<rtModel_t>(0x55U), RT_STREAM_PRIORITY_DEFAULT,
                                      RT_STREAM_DEFAULT);
  // CANN-FMK- 前缀保留给 HCCL 等框架组件，属 API 文档软约束，代码不拦截
  EXPECT_NE(collection.RequestAttachedStream(AscendString("CANN-FMK-hccl")), nullptr);
  EXPECT_EQ(runtime->create_count_, 1);

  collection.UnbindAndDestroy();
  RuntimeStub::Reset();
  AclRuntimeStub::Reset();
}

TEST(AttachedStreamCollectionTest, ReusesKeyAndCleansUpAfterUnbind) {
  auto runtime = std::make_shared<RuntimeForAttachedStreamTest>();
  auto acl = std::make_shared<AclForAttachedStreamTest>();
  RuntimeStub::SetInstance(runtime);
  AclRuntimeStub::SetInstance(acl);

  AttachedStreamCollection collection(reinterpret_cast<rtModel_t>(0x55U), RT_STREAM_PRIORITY_DEFAULT,
                                      RT_STREAM_DEFAULT);
  const auto first = collection.RequestAttachedStream(AscendString("shared"));
  const auto second = collection.RequestAttachedStream(AscendString("shared"));
  ASSERT_NE(first, nullptr);
  EXPECT_EQ(first, second);
  EXPECT_EQ(runtime->create_count_, 1);
  ASSERT_EQ(acl->bound_streams.size(), 1U);
  // HEAD flag：辅流须在模型执行时直接启动；DEFAULT(WAIT_ACTIVE) 会导致辅流任务在重放中永不执行
  ASSERT_EQ(acl->bind_flags.size(), 1U);
  EXPECT_EQ(acl->bind_flags[0], static_cast<uint32_t>(ACL_MODEL_STREAM_FLAG_HEAD));

  collection.UnbindAndDestroy();
  EXPECT_EQ(acl->synced_streams.size(), 1U);
  EXPECT_EQ(acl->unbound_streams.size(), 1U);
  EXPECT_EQ(acl->destroyed_streams.size(), 1U);
  // 顺序必须是 bind -> sync -> unbind -> destroy：aclrtDestroyStream 要求流上任务已执行完，
  // 且解绑会改动 rtModel 流表，故同步先于解绑
  ASSERT_EQ(acl->events.size(), 4U);
  EXPECT_EQ(acl->events[0], "bind");
  EXPECT_EQ(acl->events[1], "sync");
  EXPECT_EQ(acl->events[2], "unbind");
  EXPECT_EQ(acl->events[3], "destroy");
  RuntimeStub::Reset();
  AclRuntimeStub::Reset();
}

TEST(AttachedStreamCollectionTest, MissingModelReturnsNullOnValidKey) {
  AttachedStreamCollection collection(nullptr, RT_STREAM_PRIORITY_DEFAULT, RT_STREAM_DEFAULT);
  EXPECT_EQ(collection.RequestAttachedStream(AscendString("eager_aux")), nullptr);
}

TEST(AttachedStreamCollectionTest, BindFailureRollsBackTemporaryStream) {
  auto runtime = std::make_shared<RuntimeForAttachedStreamTest>();
  auto acl = std::make_shared<AclForAttachedStreamTest>();
  acl->bind_success = false;
  RuntimeStub::SetInstance(runtime);
  AclRuntimeStub::SetInstance(acl);

  AttachedStreamCollection collection(reinterpret_cast<rtModel_t>(0x55U), RT_STREAM_PRIORITY_DEFAULT,
                                      RT_STREAM_DEFAULT);
  EXPECT_EQ(collection.RequestAttachedStream(AscendString("rollback")), nullptr);
  EXPECT_EQ(runtime->create_count_, 1);
  // 回滚路径销毁的是刚建好、未下发任何任务的流，不需要同步
  EXPECT_TRUE(acl->synced_streams.empty());
  EXPECT_EQ(acl->unbound_streams.size(), 0U);
  EXPECT_EQ(acl->destroyed_streams.size(), 1U);
  RuntimeStub::Reset();
  AclRuntimeStub::Reset();
}

// 同步失败仍须继续解绑销毁：跳过会让 streams_ 清空后句柄彻底丢失，变成确定性泄漏
TEST(AttachedStreamCollectionTest, SyncFailureStillUnbindsAndDestroys) {
  auto runtime = std::make_shared<RuntimeForAttachedStreamTest>();
  auto acl = std::make_shared<AclForAttachedStreamTest>();
  acl->sync_success = false;
  RuntimeStub::SetInstance(runtime);
  AclRuntimeStub::SetInstance(acl);

  AttachedStreamCollection collection(reinterpret_cast<rtModel_t>(0x55U), RT_STREAM_PRIORITY_DEFAULT,
                                      RT_STREAM_DEFAULT);
  ASSERT_NE(collection.RequestAttachedStream(AscendString("sync_fail")), nullptr);

  collection.UnbindAndDestroy();
  ASSERT_EQ(acl->events.size(), 4U);
  EXPECT_EQ(acl->events[1], "sync");
  EXPECT_EQ(acl->unbound_streams.size(), 1U);
  EXPECT_EQ(acl->destroyed_streams.size(), 1U);
  RuntimeStub::Reset();
  AclRuntimeStub::Reset();
}

TEST(AttachedStreamCollectionTest, CreateFailureReturnsNullptrWithoutBindOrDestroy) {
  auto runtime = std::make_shared<RuntimeForAttachedStreamTest>();
  auto acl = std::make_shared<AclForAttachedStreamTest>();
  runtime->create_fail = true;
  RuntimeStub::SetInstance(runtime);
  AclRuntimeStub::SetInstance(acl);

  AttachedStreamCollection collection(reinterpret_cast<rtModel_t>(0x55U), RT_STREAM_PRIORITY_DEFAULT,
                                      RT_STREAM_DEFAULT);
  EXPECT_EQ(collection.RequestAttachedStream(AscendString("create_fail")), nullptr);
  EXPECT_EQ(runtime->create_count_, 1);
  EXPECT_EQ(acl->bound_streams.size(), 0U);
  EXPECT_EQ(acl->unbound_streams.size(), 0U);
  EXPECT_EQ(acl->destroyed_streams.size(), 0U);
  RuntimeStub::Reset();
  AclRuntimeStub::Reset();
}

TEST(AttachedStreamCollectionTest, DistinctKeysGetDistinctStreams) {
  auto runtime = std::make_shared<RuntimeForAttachedStreamTest>();
  auto acl = std::make_shared<AclForAttachedStreamTest>();
  RuntimeStub::SetInstance(runtime);
  AclRuntimeStub::SetInstance(acl);

  AttachedStreamCollection collection(reinterpret_cast<rtModel_t>(0x55U), RT_STREAM_PRIORITY_DEFAULT,
                                      RT_STREAM_DEFAULT);
  const auto first = collection.RequestAttachedStream(AscendString("key_a"));
  const auto second = collection.RequestAttachedStream(AscendString("key_b"));
  ASSERT_NE(first, nullptr);
  ASSERT_NE(second, nullptr);
  EXPECT_NE(first, second);
  EXPECT_EQ(runtime->create_count_, 2);
  ASSERT_EQ(acl->bound_streams.size(), 2U);
  // 已存在的 key 复用同一条流，不重复建流
  EXPECT_EQ(collection.RequestAttachedStream(AscendString("key_a")), first);
  EXPECT_EQ(runtime->create_count_, 2);

  collection.UnbindAndDestroy();
  EXPECT_EQ(acl->unbound_streams.size(), 2U);
  EXPECT_EQ(acl->destroyed_streams.size(), 2U);
  RuntimeStub::Reset();
  AclRuntimeStub::Reset();
}
}  // namespace
}  // namespace ge
