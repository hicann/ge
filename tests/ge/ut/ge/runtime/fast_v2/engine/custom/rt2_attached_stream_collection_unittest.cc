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
#include <string>
#include <vector>

#include "engine/custom/kernel/rt2_attached_stream_collection.h"
#include "graph/ascend_string.h"
#include "depends/runtime/src/runtime_stub.h"
#include "depends/ascendcl/src/ascendcl_stub.h"

namespace gert {
namespace {
constexpr rtError_t kRtErrorForTest = 1;
// 辅流数量不设上限，取一个较大的数量验证不会被截断
constexpr size_t kLargeStreamNum = 16U;

class RuntimeForRt2AttachedStreamTest final : public ge::RuntimeStub {
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

class AclForRt2AttachedStreamTest final : public ge::AclRuntimeStub {
 public:
  aclError aclrtSynchronizeStream(aclrtStream stream) override {
    synced_streams.push_back(stream);
    events.push_back("sync");
    return ACL_SUCCESS;
  }
  aclError aclrtDestroyStream(aclrtStream stream) override {
    destroyed_streams.push_back(stream);
    events.push_back("destroy");
    return ACL_SUCCESS;
  }

  std::vector<aclrtStream> synced_streams;
  std::vector<aclrtStream> destroyed_streams;
  std::vector<std::string> events;
};

class Rt2AttachedStreamCollectionTest : public testing::Test {
 protected:
  void SetUp() override {
    runtime_ = std::make_shared<RuntimeForRt2AttachedStreamTest>();
    acl_ = std::make_shared<AclForRt2AttachedStreamTest>();
    ge::RuntimeStub::SetInstance(runtime_);
    ge::AclRuntimeStub::SetInstance(acl_);
  }
  void TearDown() override {
    ge::RuntimeStub::Reset();
    ge::AclRuntimeStub::Reset();
  }

  std::shared_ptr<RuntimeForRt2AttachedStreamTest> runtime_;
  std::shared_ptr<AclForRt2AttachedStreamTest> acl_;
};

TEST_F(Rt2AttachedStreamCollectionTest, RejectsEmptyKeyWithoutRuntimeCall) {
  Rt2AttachedStreamCollection collection;
  EXPECT_EQ(collection.RequestAttachedStream(ge::AscendString()), nullptr);
  EXPECT_EQ(collection.RequestAttachedStream(ge::AscendString("")), nullptr);
  EXPECT_EQ(runtime_->create_count_, 0);
}

TEST_F(Rt2AttachedStreamCollectionTest, ReusesSameKeyAndIsolatesDistinctKeys) {
  Rt2AttachedStreamCollection collection;
  const auto first = collection.RequestAttachedStream(ge::AscendString("rt2_aux"));
  const auto reuse = collection.RequestAttachedStream(ge::AscendString("rt2_aux"));
  const auto other = collection.RequestAttachedStream(ge::AscendString("rt2_other"));
  ASSERT_NE(first, nullptr);
  EXPECT_EQ(reuse, first);
  ASSERT_NE(other, nullptr);
  EXPECT_NE(other, first);
  EXPECT_EQ(runtime_->create_count_, 2);
}

TEST_F(Rt2AttachedStreamCollectionTest, AcceptsFrameworkReservedPrefixKey) {
  Rt2AttachedStreamCollection collection;
  // CANN-FMK- 前缀保留给 HCCL 等框架组件，属 API 文档软约束，代码不拦截
  EXPECT_NE(collection.RequestAttachedStream(ge::AscendString("CANN-FMK-hccl")), nullptr);
  EXPECT_EQ(runtime_->create_count_, 1);
}

TEST_F(Rt2AttachedStreamCollectionTest, CreateFailureReturnsNullptrWithoutDestroy) {
  runtime_->create_fail = true;
  Rt2AttachedStreamCollection collection;
  EXPECT_EQ(collection.RequestAttachedStream(ge::AscendString("rt2_aux")), nullptr);
  EXPECT_EQ(runtime_->create_count_, 1);
  EXPECT_TRUE(acl_->destroyed_streams.empty());
}

TEST_F(Rt2AttachedStreamCollectionTest, HasNoStreamNumLimit) {
  Rt2AttachedStreamCollection collection;
  for (size_t i = 0U; i < kLargeStreamNum; ++i) {
    ASSERT_NE(collection.RequestAttachedStream(ge::AscendString(("rt2_key_" + std::to_string(i)).c_str())), nullptr);
  }
  EXPECT_EQ(static_cast<size_t>(runtime_->create_count_), kLargeStreamNum);
  // 数量不受限后，复用与隔离语义仍须成立
  EXPECT_EQ(collection.RequestAttachedStream(ge::AscendString("rt2_key_0")),
            collection.RequestAttachedStream(ge::AscendString("rt2_key_0")));
  EXPECT_EQ(static_cast<size_t>(runtime_->create_count_), kLargeStreamNum);
}

TEST_F(Rt2AttachedStreamCollectionTest, DestroySyncsBeforeDestroyAndIsIdempotent) {
  Rt2AttachedStreamCollection collection;
  const auto first = collection.RequestAttachedStream(ge::AscendString("rt2_aux"));
  ASSERT_NE(first, nullptr);

  collection.Destroy();
  ASSERT_EQ(acl_->events.size(), 2U);
  EXPECT_EQ(acl_->events[0], "sync");
  EXPECT_EQ(acl_->events[1], "destroy");
  EXPECT_EQ(acl_->synced_streams[0], first);
  EXPECT_EQ(acl_->destroyed_streams[0], first);

  // 幂等：重复 Destroy 不再产生任何 RTS/ACL 调用
  collection.Destroy();
  EXPECT_EQ(acl_->events.size(), 2U);
}

TEST_F(Rt2AttachedStreamCollectionTest, RecreatesStreamAfterDestroy) {
  Rt2AttachedStreamCollection collection;
  const auto first = collection.RequestAttachedStream(ge::AscendString("rt2_aux"));
  ASSERT_NE(first, nullptr);
  collection.Destroy();

  // 防御性契约：Destroy 后对象仍可继续使用（生产路径 Destroy 只由析构调用，不会走到这里）
  const auto second = collection.RequestAttachedStream(ge::AscendString("rt2_aux"));
  ASSERT_NE(second, nullptr);
  EXPECT_NE(second, first);
  EXPECT_EQ(runtime_->create_count_, 2);
}
}  // namespace
}  // namespace gert
