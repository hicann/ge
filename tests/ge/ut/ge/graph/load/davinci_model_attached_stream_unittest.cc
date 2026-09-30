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

#include "graph/ascend_string.h"
#include "graph/load/model_manager/davinci_model.h"
#include "depends/runtime/src/runtime_stub.h"
#include "depends/ascendcl/src/ascendcl_stub.h"

namespace ge {
namespace {
class RuntimeForModelAttachedStreamTest final : public RuntimeStub {
 public:
  rtError_t rtStreamCreateWithFlags(rtStream_t *stream, int32_t, uint32_t) override {
    ++create_count_;
    *stream = reinterpret_cast<rtStream_t>(++next_stream_);
    return RT_ERROR_NONE;
  }

  int create_count_{0};

 private:
  uintptr_t next_stream_{0x2000U};
};

class AclForModelAttachedStreamTest : public AclRuntimeStub {
 public:
  aclError aclrtSynchronizeStream(aclrtStream stream) override {
    synced_streams.push_back(stream);
    events.push_back("sync");
    return ACL_SUCCESS;
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

  std::vector<aclrtStream> synced_streams;
  std::vector<aclrtStream> unbound_streams;
  std::vector<aclrtStream> destroyed_streams;
  std::vector<std::string> events;
};

class AclWithoutCtxForModelAttachedStreamTest final : public AclForModelAttachedStreamTest {
 public:
  aclError aclrtGetCurrentContext(aclrtContext *context) override {
    (void)context;
    return ACL_ERROR_RT_FAILURE;
  }
};

// RT2 内嵌 V1 静态子图场景：DavinciModelFinalizer 调用本接口清理 Eager 算子申请的辅流，
// 且 DestroyResources 会置 has_finalized_，析构不再兜底，因此本接口必须真正释放流且可重复调用。
TEST(DavinciModelAttachedStreamTest, UnbindAndDestroyAttachedStreamsIsIdempotent) {
  auto runtime = std::make_shared<RuntimeForModelAttachedStreamTest>();
  auto acl = std::make_shared<AclForModelAttachedStreamTest>();
  RuntimeStub::SetInstance(runtime);
  AclRuntimeStub::SetInstance(acl);

  DavinciModel model(0, nullptr);
  // UT 目标以 -fno-access-control 编译，可直接初始化私有成员
  model.attached_stream_collection_.Initialize(reinterpret_cast<rtModel_t>(0x55U), RT_STREAM_PRIORITY_DEFAULT,
                                               RT_STREAM_DEFAULT);
  ASSERT_NE(model.GetAttachedStreamProvider()->RequestAttachedStream(AscendString("eager_aux")), nullptr);
  ASSERT_EQ(runtime->create_count_, 1);

  model.UnbindAndDestroyAttachedStreams();
  // 顺序必须是 sync -> unbind -> destroy，与 RT2 的 Rt2AttachedStreamCollection::Destroy 一致
  ASSERT_EQ(acl->events.size(), 3U);
  EXPECT_EQ(acl->events[0], "sync");
  EXPECT_EQ(acl->events[1], "unbind");
  EXPECT_EQ(acl->events[2], "destroy");
  EXPECT_EQ(acl->synced_streams.size(), 1U);
  EXPECT_EQ(acl->unbound_streams.size(), 1U);
  EXPECT_EQ(acl->destroyed_streams.size(), 1U);

  // 幂等：Finalizer 与析构路径都可能调用，重复调用不得产生额外的同步/解绑/销毁
  model.UnbindAndDestroyAttachedStreams();
  EXPECT_EQ(acl->events.size(), 3U);
  EXPECT_EQ(acl->unbound_streams.size(), 1U);
  EXPECT_EQ(acl->destroyed_streams.size(), 1U);

  RuntimeStub::Reset();
  AclRuntimeStub::Reset();
}

// ~DavinciModel 兜底清理必须与 Finalizer 路径同一守卫：ctx 已销毁时不得再调 RT 接口，
// 否则每条辅流都会刷 sync/unbind/destroy 的 RT 错误日志（对齐 UnbindTaskSinkStream/DestroyStream/DestroyResources）
TEST(DavinciModelAttachedStreamTest, DestructorSkipsAttachedStreamCleanupWhenContextMissing) {
  auto runtime = std::make_shared<RuntimeForModelAttachedStreamTest>();
  auto acl = std::make_shared<AclWithoutCtxForModelAttachedStreamTest>();
  RuntimeStub::SetInstance(runtime);
  AclRuntimeStub::SetInstance(acl);
  {
    DavinciModel model(0, nullptr);
    model.attached_stream_collection_.Initialize(reinterpret_cast<rtModel_t>(0x55U), RT_STREAM_PRIORITY_DEFAULT,
                                                 RT_STREAM_DEFAULT);
    ASSERT_NE(model.GetAttachedStreamProvider()->RequestAttachedStream(AscendString("eager_aux")), nullptr);
    ASSERT_EQ(runtime->create_count_, 1);
  }
  EXPECT_TRUE(acl->events.empty());
  EXPECT_TRUE(acl->synced_streams.empty());
  EXPECT_TRUE(acl->unbound_streams.empty());
  EXPECT_TRUE(acl->destroyed_streams.empty());

  RuntimeStub::Reset();
  AclRuntimeStub::Reset();
}
}  // namespace
}  // namespace ge
