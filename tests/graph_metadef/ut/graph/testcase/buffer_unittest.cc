/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <securec.h>
#include <gtest/gtest.h>
#include "graph/buffer.h"
#include "buffer/buffer_impl.h"
#include "proto/ge_ir.pb.h"

namespace ge {
class BufferUT : public testing::Test {
 protected:
  void SetUp() {}

  void TearDown() {}
};

TEST_F(BufferUT, ShareFrom1) {
  uint8_t first_buf[100];
  for (int i = 0; i < 100; ++i) {
    first_buf[i] = i * 1024;
  }
  uint8_t second_buf[100];
  for (int i = 0; i < 100; ++i) {
    second_buf[i] = i * 1024;
  }
  second_buf[50] = 10;

  Buffer buf(100);
  memcpy_s(buf.GetData(), buf.GetSize(), first_buf, sizeof(first_buf));
  EXPECT_EQ(memcmp(buf.GetData(), first_buf, sizeof(first_buf)), 0);

  Buffer buf1 = BufferUtils::CreateShareFrom(buf);  // The buf1 and buf are ref from the same memory now
  buf1.GetData()[50] = 10;
  EXPECT_EQ(memcmp(buf1.GetData(), second_buf, sizeof(second_buf)), 0);
  EXPECT_EQ(memcmp(buf.GetData(), second_buf, sizeof(second_buf)), 0);
  EXPECT_NE(memcmp(buf.GetData(), first_buf, sizeof(first_buf)), 0);
}

TEST_F(BufferUT, ShareFrom2) {
  uint8_t first_buf[100];
  for (int i = 0; i < 100; ++i) {
    first_buf[i] = i * 1024;
  }
  uint8_t second_buf[100];
  for (int i = 0; i < 100; ++i) {
    second_buf[i] = i * 1024;
  }
  second_buf[50] = 10;

  Buffer buf(100);
  memcpy_s(buf.GetData(), buf.GetSize(), first_buf, sizeof(first_buf));
  EXPECT_EQ(memcmp(buf.GetData(), first_buf, sizeof(first_buf)), 0);

  Buffer buf1;
  BufferUtils::ShareFrom(buf, buf1);  // The buf1 and buf are ref from the same memory now
  buf1.GetData()[50] = 10;
  EXPECT_EQ(memcmp(buf1.GetData(), second_buf, sizeof(second_buf)), 0);
  EXPECT_EQ(memcmp(buf.GetData(), second_buf, sizeof(second_buf)), 0);
  EXPECT_NE(memcmp(buf.GetData(), first_buf, sizeof(first_buf)), 0);
}

TEST_F(BufferUT, OperatorAssign) {
  uint8_t first_buf[100];
  for (int i = 0; i < 100; ++i) {
    first_buf[i] = i * 1024;
  }
  uint8_t second_buf[100];
  for (int i = 0; i < 100; ++i) {
    second_buf[i] = i * 1024;
  }
  second_buf[50] = 10;

  Buffer buf(100);
  memcpy_s(buf.GetData(), buf.GetSize(), first_buf, sizeof(first_buf));
  EXPECT_EQ(memcmp(buf.GetData(), first_buf, sizeof(first_buf)), 0);

  Buffer buf1;
  buf1 = buf;  // The buf1 and buf are ref from the same memory now
  buf1.GetData()[50] = 10;
  EXPECT_EQ(memcmp(buf1.GetData(), second_buf, sizeof(second_buf)), 0);
  EXPECT_EQ(memcmp(buf.GetData(), second_buf, sizeof(second_buf)), 0);
  EXPECT_NE(memcmp(buf.GetData(), first_buf, sizeof(first_buf)), 0);
}

TEST_F(BufferUT, CreateShareFrom) {
  uint8_t first_buf[100];
  for (int i = 0; i < 100; ++i) {
    first_buf[i] = i * 1024;
  }
  uint8_t second_buf[100];
  for (int i = 0; i < 100; ++i) {
    second_buf[i] = i * 1024;
  }
  second_buf[50] = 10;

  Buffer buf(100);
  memcpy_s(buf.GetData(), buf.GetSize(), first_buf, sizeof(first_buf));
  EXPECT_EQ(memcmp(buf.GetData(), first_buf, sizeof(first_buf)), 0);

  Buffer buf1 = BufferUtils::CreateShareFrom(buf);  // The buf1 and buf are ref from the same memory now
  buf1.GetData()[50] = 10;
  EXPECT_EQ(memcmp(buf1.GetData(), second_buf, sizeof(second_buf)), 0);
  EXPECT_EQ(memcmp(buf.GetData(), second_buf, sizeof(second_buf)), 0);
  EXPECT_NE(memcmp(buf.GetData(), first_buf, sizeof(first_buf)), 0);
}

TEST_F(BufferUT, CreateCopyFrom1) {
  uint8_t first_buf[100];
  for (int i = 0; i < 100; ++i) {
    first_buf[i] = i * 2;
  }
  uint8_t second_buf[100];
  for (int i = 0; i < 100; ++i) {
    second_buf[i] = i * 2;
  }
  second_buf[50] = 250;

  Buffer buf(100);
  memcpy_s(buf.GetData(), buf.GetSize(), first_buf, sizeof(first_buf));
  EXPECT_EQ(memcmp(buf.GetData(), first_buf, sizeof(first_buf)), 0);

  Buffer buf1;
  BufferUtils::CopyFrom(buf, buf1);
  buf1.GetData()[50] = 250;
  EXPECT_EQ(memcmp(buf1.GetData(), second_buf, sizeof(second_buf)), 0);
  EXPECT_EQ(memcmp(buf.GetData(), first_buf, sizeof(first_buf)), 0);
}

TEST_F(BufferUT, CreateCopyFrom2) {
  uint8_t first_buf[100];
  for (int i = 0; i < 100; ++i) {
    first_buf[i] = i * 2;
  }
  uint8_t second_buf[100];
  for (int i = 0; i < 100; ++i) {
    second_buf[i] = i * 2;
  }
  second_buf[50] = 250;

  Buffer buf(100);
  memcpy_s(buf.GetData(), buf.GetSize(), first_buf, sizeof(first_buf));
  EXPECT_EQ(memcmp(buf.GetData(), first_buf, sizeof(first_buf)), 0);

  Buffer buf1 = BufferUtils::CreateCopyFrom(buf);  // The buf1 and buf are ref from the same memory now
  buf1.GetData()[50] = 250;
  EXPECT_EQ(memcmp(buf1.GetData(), second_buf, sizeof(second_buf)), 0);
  EXPECT_EQ(memcmp(buf.GetData(), first_buf, sizeof(first_buf)), 0);
}

TEST_F(BufferUT, Cov_DefaultConstructor) {
  Buffer buf;
  EXPECT_EQ(buf.GetSize(), 0UL);
  EXPECT_EQ(buf.GetData(), nullptr);
  EXPECT_EQ(buf.data(), nullptr);
  EXPECT_EQ(buf.size(), 0UL);
}

TEST_F(BufferUT, Cov_CopyConstructor) {
  uint8_t data[10] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
  Buffer buf1 = Buffer::CopyFrom(data, sizeof(data));
  Buffer buf2(buf1);
  EXPECT_EQ(buf2.GetSize(), 10UL);
  EXPECT_EQ(memcmp(buf2.GetData(), data, sizeof(data)), 0);
}

TEST_F(BufferUT, Cov_SizeConstructorWithDefaultVal) {
  Buffer buf(10, 0xAB);
  EXPECT_EQ(buf.GetSize(), 10UL);
  for (size_t i = 0; i < 10; ++i) {
    EXPECT_EQ(buf[i], 0xAB);
  }
}

TEST_F(BufferUT, Cov_GetDataMutable) {
  uint8_t data[5] = {1, 2, 3, 4, 5};
  Buffer buf = Buffer::CopyFrom(data, sizeof(data));
  uint8_t *mutable_data = buf.GetData();
  ASSERT_NE(mutable_data, nullptr);
  mutable_data[0] = 99;
  EXPECT_EQ(buf[0], 99);
}

TEST_F(BufferUT, Cov_GetDataMutable_EmptyBuffer) {
  Buffer buf;
  EXPECT_EQ(buf.GetData(), nullptr);
}

TEST_F(BufferUT, Cov_GetDataConst_NullBuffer) {
  Buffer buf;
  const Buffer &const_buf = buf;
  EXPECT_EQ(const_buf.GetData(), nullptr);
}

TEST_F(BufferUT, Cov_GetSize_EmptyBuffer) {
  Buffer buf;
  EXPECT_EQ(buf.GetSize(), 0UL);
}

TEST_F(BufferUT, Cv_GetSize_BufferWithSize) {
  Buffer buf(50);
  EXPECT_EQ(buf.GetSize(), 50UL);
}

TEST_F(BufferUT, Cov_ClearBuffer) {
  uint8_t data[10] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  Buffer buf = Buffer::CopyFrom(data, sizeof(data));
  EXPECT_EQ(buf.GetSize(), 10UL);
  buf.ClearBuffer();
  EXPECT_EQ(buf.GetSize(), 0UL);
}

TEST_F(BufferUT, Cov_Clear_EmptyBuffer) {
  Buffer buf;
  buf.clear();
  EXPECT_EQ(buf.size(), 0UL);
}

TEST_F(BufferUT, Cov_OperatorIndex_InBounds) {
  uint8_t data[5] = {10, 20, 30, 40, 50};
  Buffer buf = Buffer::CopyFrom(data, sizeof(data));
  EXPECT_EQ(buf[0], 10);
  EXPECT_EQ(buf[4], 50);
}

TEST_F(BufferUT, Cov_OperatorIndex_OutOfBounds) {
  Buffer buf(5);
  EXPECT_EQ(buf[100], 0xffU);
}

TEST_F(BufferUT, Cov_OperatorIndex_EmptyBuffer) {
  Buffer buf;
  EXPECT_EQ(buf[0], 0xffU);
}

TEST_F(BufferUT, Cov_OperatorAssign_SelfAssign) {
  uint8_t data[5] = {1, 2, 3, 4, 5};
  Buffer buf = Buffer::CopyFrom(data, sizeof(data));
  buf = buf;
  EXPECT_EQ(buf.GetSize(), 5UL);
  EXPECT_EQ(buf[0], 1);
}

TEST_F(BufferUT, Cov_DataAlias) {
  uint8_t data[5] = {1, 2, 3, 4, 5};
  Buffer buf = Buffer::CopyFrom(data, sizeof(data));
  EXPECT_EQ(buf.data(), buf.GetData());
}

TEST_F(BufferUT, Cov_SizeAlias) {
  Buffer buf(42);
  EXPECT_EQ(buf.size(), buf.GetSize());
}

TEST_F(BufferUT, Cov_ClearAlias) {
  uint8_t data[5] = {1, 2, 3, 4, 5};
  Buffer buf = Buffer::CopyFrom(data, sizeof(data));
  buf.clear();
  EXPECT_EQ(buf.size(), 0UL);
}

TEST_F(BufferUT, Cov_BufferUtils_CreateCopyFromData) {
  uint8_t data[5] = {1, 2, 3, 4, 5};
  Buffer buf = BufferUtils::CreateCopyFrom(data, sizeof(data));
  EXPECT_EQ(buf.GetSize(), 5UL);
  EXPECT_EQ(memcmp(buf.GetData(), data, sizeof(data)), 0);
}

TEST_F(BufferUT, Cov_BufferUtils_CreateCopyFromBuffer) {
  uint8_t data[5] = {1, 2, 3, 4, 5};
  Buffer buf1 = BufferUtils::CreateCopyFrom(data, sizeof(data));
  Buffer buf2 = BufferUtils::CreateCopyFrom(buf1);
  EXPECT_EQ(buf2.GetSize(), 5UL);
  EXPECT_EQ(memcmp(buf2.GetData(), data, sizeof(data)), 0);
  buf2.GetData()[0] = 99;
  EXPECT_EQ(buf1[0], 1);
}

TEST_F(BufferUT, Cov_CopyFrom_NullData) {
  Buffer buf = Buffer::CopyFrom(nullptr, 100);
  EXPECT_EQ(buf.GetSize(), 0UL);
}

TEST_F(BufferUT, Cov_DefaultValConstructor) {
  Buffer buf(20, 0x42);
  EXPECT_EQ(buf.GetSize(), 20UL);
  for (size_t i = 0; i < 20; ++i) {
    EXPECT_EQ(buf[i], 0x42);
  }
}

TEST_F(BufferUT, Cov_BufferImpl_ProtoOwnerAttrDef) {
  auto attr_def = std::make_shared<proto::AttrDef>();
  attr_def->set_bt("hello");
  BufferImpl impl(attr_def, attr_def.get());
  EXPECT_EQ(impl.GetSize(), 5UL);
  const BufferImpl &const_impl = impl;
  const uint8_t *ptr = const_impl.GetData();
  ASSERT_NE(ptr, nullptr);
}

TEST_F(BufferUT, Cov_BufferImpl_ProtoOwnerAttrDef_NullProtoMsg) {
  auto owner = std::make_shared<proto::AttrDef>();
  BufferImpl impl(owner, static_cast<proto::AttrDef *>(nullptr));
  EXPECT_EQ(impl.GetSize(), 0UL);
  EXPECT_EQ(impl.GetData(), nullptr);
  const BufferImpl &const_impl = impl;
  EXPECT_EQ(const_impl.GetData(), nullptr);
}

TEST_F(BufferUT, Cov_BufferImpl_ProtoOwnerString) {
  auto owner = std::make_shared<proto::AttrDef>();
  std::string buf = "test";
  BufferImpl impl(owner, &buf);
  EXPECT_EQ(impl.GetSize(), 4UL);
  const BufferImpl &const_impl = impl;
  const uint8_t *ptr = const_impl.GetData();
  ASSERT_NE(ptr, nullptr);
}

TEST_F(BufferUT, Cov_BufferImpl_ProtoOwnerString_NullString) {
  auto owner = std::make_shared<proto::AttrDef>();
  BufferImpl impl(owner, static_cast<std::string *>(nullptr));
  EXPECT_EQ(impl.GetSize(), 0UL);
  EXPECT_EQ(impl.GetData(), nullptr);
  const BufferImpl &const_impl = impl;
  EXPECT_EQ(const_impl.GetData(), nullptr);
}

TEST_F(BufferUT, Cov_BufferImpl_GetDataConst_WithData) {
  uint8_t data[5] = {1, 2, 3, 4, 5};
  BufferImpl impl;
  impl.CopyFrom(data, sizeof(data));
  const BufferImpl &const_impl = impl;
  const uint8_t *ptr = const_impl.GetData();
  ASSERT_NE(ptr, nullptr);
  EXPECT_EQ(memcmp(ptr, data, sizeof(data)), 0);
}

TEST_F(BufferUT, Cov_Buffer_ProtoOwnerAttrDefCtor) {
  auto attr_def = std::make_shared<proto::AttrDef>();
  attr_def->set_bt("test");
  ProtoMsgOwner owner = attr_def;
  Buffer buf(owner, attr_def.get());
  EXPECT_EQ(buf.GetSize(), 4UL);
  EXPECT_NE(buf.GetData(), nullptr);
}

TEST_F(BufferUT, Cov_Buffer_ProtoOwnerStringCtor) {
  auto attr_def = std::make_shared<proto::AttrDef>();
  ProtoMsgOwner owner = attr_def;
  std::string str = "hello";
  Buffer buf(owner, &str);
  EXPECT_EQ(buf.GetSize(), 5UL);
  EXPECT_NE(buf.GetData(), nullptr);
}

class GeAttrValueImp {
 public:
  static Buffer CreateBufferFromAttrDef(const ProtoMsgOwner &proto_owner, proto::AttrDef *buffer) {
    return Buffer(proto_owner, buffer);
  }
  static Buffer CreateBufferFromString(const ProtoMsgOwner &proto_owner, std::string *buffer) {
    return Buffer(proto_owner, buffer);
  }
};

TEST_F(BufferUT, Cov_Buffer_FromProtoOwnerAttrDef) {
  auto attr_def = std::make_shared<proto::AttrDef>();
  attr_def->set_bt("hello");
  Buffer buf = GeAttrValueImp::CreateBufferFromAttrDef(attr_def, attr_def.get());
  EXPECT_EQ(buf.GetSize(), 5UL);
  const Buffer &const_buf = buf;
  ASSERT_NE(const_buf.GetData(), nullptr);
  EXPECT_EQ(const_buf.data()[0], static_cast<uint8_t>('h'));
}

TEST_F(BufferUT, Cov_Buffer_FromProtoOwnerString) {
  auto owner = std::make_shared<proto::AttrDef>();
  std::string str = "test";
  Buffer buf = GeAttrValueImp::CreateBufferFromString(owner, &str);
  EXPECT_EQ(buf.GetSize(), 4UL);
  const Buffer &const_buf = buf;
  ASSERT_NE(const_buf.GetData(), nullptr);
  EXPECT_EQ(const_buf.data()[0], static_cast<uint8_t>('t'));
}
}  // namespace ge
