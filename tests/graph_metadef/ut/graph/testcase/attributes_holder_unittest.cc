/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include "test_structs.h"
#include "func_counter.h"
#include "graph/detail/attributes_holder.h"
#include "graph/ge_attr_value.h"
#include "graph/any_value.h"
#include "ge_ir.pb.h"

namespace ge {
namespace {

class SubAttrStore : public AttrStore {
 public:
  bool SetAnyValueByName(const std::string &name, const AnyValue &value);
  SubAttrStore &operator=(const SubAttrStore &other) {
    if (this == &other) {
      return *this;
    }
    CopyAttrStoreAllMembers(other);
    return *this;
  }
};

class SubAttrHolder : public AttrHolder {
 public:
  SubAttrHolder();
  virtual ~SubAttrHolder() = default;

 protected:
  ProtoAttrMap &MutableAttrMap() override;
  ConstProtoAttrMap &GetAttrMap() const override;

 public:
  SubAttrStore attrs_;
};

SubAttrHolder::SubAttrHolder() {
  attrs_ = SubAttrStore();
}

ProtoAttrMap &SubAttrHolder::MutableAttrMap() {
  return attrs_;
}

ConstProtoAttrMap &SubAttrHolder::GetAttrMap() const {
  return attrs_;
}

}  // namespace

void oper(AnyValue::OperateType ot, const AnyValue *av, void *out) {
  return;
}

class AttrHolderUt : public testing::Test {};

TEST_F(AttrHolderUt, All) {
  EXPECT_NO_THROW(GeIrProtoHelper<proto::TensorDescriptor> helper1; helper1.InitDefault();

                  GeIrProtoHelper<proto::ShapeDef> helper2; helper2.InitDefault();

                  GeIrProtoHelper<proto::NamedAttrs> helper3; helper3.InitDefault();

                  GeIrProtoHelper<proto::ModelDef> helper4; helper4.InitDefault();

                  GeIrProtoHelper<proto::OpDef> helper5; helper5.InitDefault();

                  GeIrProtoHelper<proto::GraphDef> helper6; helper6.InitDefault(););
}

TEST_F(AttrHolderUt, Plus) {
  SubAttrHolder sub_attr_hodler = SubAttrHolder();
  AnyValue av = AnyValue::CreateFrom<int>(1);
  av.operate_ = oper;
  EXPECT_EQ(sub_attr_hodler.SetAttr("name", av), GRAPH_SUCCESS);
  av.operate_ = nullptr;
  EXPECT_EQ(sub_attr_hodler.TrySetAttr("name", av), GRAPH_FAILED);
  EXPECT_EQ(sub_attr_hodler.AddRequiredAttr("name"), GRAPH_FAILED);
}

TEST_F(AttrHolderUt, ExtAttrGetSuccess) {
  SubAttrHolder holder;
  EXPECT_EQ(holder.GetExtAttr<int32_t>("TestName"), nullptr);
  holder.SetExtAttr<int32_t>("TestName", static_cast<int32_t>(10));
  auto pi = holder.GetExtAttr<int32_t>("TestName");
  ASSERT_NE(pi, nullptr);
  EXPECT_EQ(*pi, 10);
}

TEST_F(AttrHolderUt, ExtAttrGetWrongType) {
  SubAttrHolder holder;
  EXPECT_EQ(holder.GetExtAttr<int32_t>("TestName"), nullptr);
  holder.SetExtAttr<int32_t>("TestName", static_cast<int32_t>(10));
  auto pi = holder.GetExtAttr<int32_t>("TestName");
  ASSERT_NE(pi, nullptr);
  auto p_int64 = holder.GetExtAttr<int64_t>("TestName");
  ASSERT_EQ(p_int64, nullptr);
}
TEST_F(AttrHolderUt, ExtAttrGetClassSuccess) {
  SubAttrHolder holder;
  std::vector<int64_t> data = {1, 2, 10, 20, 100, 200, 1000, 2000};
  EXPECT_EQ(holder.GetExtAttr<std::vector<int64_t>>("TestName"), nullptr);
  holder.SetExtAttr<std::vector<int64_t>>("TestName", data);
  auto pd = holder.GetExtAttr<std::vector<int64_t>>("TestName");
  ASSERT_NE(pd, nullptr);
  EXPECT_EQ(*pd, data);
}

TEST_F(AttrHolderUt, ExtAttrGetSameAddress) {
  SubAttrHolder holder;
  std::vector<int64_t> data = {1, 2, 10, 20, 100, 200, 1000, 2000};
  holder.SetExtAttr<std::vector<int64_t>>("TestName", data);
  auto pd = holder.GetExtAttr<std::vector<int64_t>>("TestName");
  ASSERT_NE(pd, nullptr);

  auto pd2 = holder.GetExtAttr<std::vector<int64_t>>("TestName");
  ASSERT_NE(pd2, nullptr);

  EXPECT_TRUE(pd == pd2);
}

TEST_F(AttrHolderUt, ExtAttrTryGetSuccess) {
  SubAttrHolder holder;
  std::vector<int64_t> data1 = {1, 2, 10, 20, 100, 200, 1000, 2000};
  std::vector<int64_t> data2 = {1, 2, 10, 20, 100, 200, 1000, 2000, 10000, 20000};

  std::vector<int64_t> ret_data = holder.TryGetExtAttr("TestName", data1);
  EXPECT_EQ(ret_data, data1);

  holder.SetExtAttr<std::vector<int64_t>>("TestName", data2);
  ret_data = holder.TryGetExtAttr("TestName", data1);
  EXPECT_NE(ret_data, data1);
  EXPECT_EQ(ret_data, data2);
}

TEST_F(AttrHolderUt, ExtAttrEraseSuccess) {
  SubAttrHolder holder;
  holder.SetExtAttr<int32_t>("TestName", static_cast<int32_t>(10));
  auto pi = holder.GetExtAttr<int32_t>("TestName");
  ASSERT_NE(pi, nullptr);
  EXPECT_EQ(*pi, 10);
  EXPECT_TRUE(holder.DelExtAttr("TestName"));
  pi = holder.GetExtAttr<int32_t>("TestName");
  EXPECT_EQ(pi, nullptr);
}

TEST_F(AttrHolderUt, ExtAttrEraseFailedWhenAttrNotExsit) {
  SubAttrHolder holder;
  EXPECT_FALSE(holder.DelExtAttr("TestName"));
}
TEST_F(AttrHolderUt, GetOrCreateAttrsGroup_AutoCreate_Ok) {
  SubAttrHolder holder;
  ASSERT_NE(holder.GetOrCreateAttrsGroup<TestAttrGroup>(), nullptr);
}
TEST_F(AttrHolderUt, GetOrCreateAttrsGroup_MultiTimes_SameRet_Ok) {
  SubAttrHolder holder;
  auto ret1 = holder.GetOrCreateAttrsGroup<TestAttrGroup>();
  auto ret2 = holder.GetOrCreateAttrsGroup<TestAttrGroup>();
  ASSERT_EQ(ret1, ret2);
  ASSERT_NE(ret1, nullptr);
}
TEST_F(AttrHolderUt, GetAttrsGroup_NotExists_ReturnNull) {
  SubAttrHolder holder;
  ASSERT_EQ(holder.GetAttrsGroup<TestAttrGroup>(), nullptr);
}
TEST_F(AttrHolderUt, GetAttrsGroup_Ok) {
  SubAttrHolder holder;
  auto ret1 = holder.GetOrCreateAttrsGroup<TestAttrGroup>();
  ASSERT_NE(ret1, nullptr);
  auto ret2 = holder.GetAttrsGroup<TestAttrGroup>();
  ASSERT_EQ(ret1, ret2);
}
TEST_F(AttrHolderUt, CreateAttrGroupWith0Args) {
  SubAttrHolder s;
  auto ptr = s.CreateAttrsGroup<TestAttrGroup>();
  ASSERT_NE(ptr, nullptr);
  ASSERT_EQ(ptr->a, 0);
  ASSERT_EQ(ptr->b, 0);

  auto ptr_1 = s.CreateAttrsGroup<TestAttrGroup>();
  ASSERT_EQ(ptr_1, nullptr);

  auto ptr_2 = s.CreateAttrsGroup<TestAttrGroup>(1);
  ASSERT_EQ(ptr_2, nullptr);

  auto ptr_3 = s.CreateAttrsGroup<TestAttrGroup>(1, 2);
  ASSERT_EQ(ptr_3, nullptr);

  auto ptr_4 = s.GetAttrsGroup<TestAttrGroup>();
  ASSERT_EQ(ptr_4, ptr);

  auto ptr_5 = s.GetOrCreateAttrsGroup<TestAttrGroup>();
  ASSERT_EQ(ptr_5, ptr);
}
TEST_F(AttrHolderUt, CreateAttrGroupWith1Args) {
  SubAttrHolder s;
  auto ptr = s.CreateAttrsGroup<TestAttrGroup>(1);
  ASSERT_NE(ptr, nullptr);
  ASSERT_EQ(ptr->a, 1);
  ASSERT_EQ(ptr->b, 0);

  auto ptr_1 = s.CreateAttrsGroup<TestAttrGroup>();
  ASSERT_EQ(ptr_1, nullptr);

  auto ptr_2 = s.CreateAttrsGroup<TestAttrGroup>(1);
  ASSERT_EQ(ptr_2, nullptr);

  auto ptr_3 = s.CreateAttrsGroup<TestAttrGroup>(1, 2);
  ASSERT_EQ(ptr_3, nullptr);

  auto ptr_4 = s.GetAttrsGroup<TestAttrGroup>();
  ASSERT_EQ(ptr_4, ptr);

  auto ptr_5 = s.GetOrCreateAttrsGroup<TestAttrGroup>();
  ASSERT_EQ(ptr_5, ptr);
}
TEST_F(AttrHolderUt, CreateAttrGroupWith2Args) {
  SubAttrHolder s;
  auto ptr = s.CreateAttrsGroup<TestAttrGroup>(1, 2);
  ASSERT_NE(ptr, nullptr);
  ASSERT_EQ(ptr->a, 1);
  ASSERT_EQ(ptr->b, 2);

  auto ptr_1 = s.CreateAttrsGroup<TestAttrGroup>();
  ASSERT_EQ(ptr_1, nullptr);

  auto ptr_2 = s.CreateAttrsGroup<TestAttrGroup>(1);
  ASSERT_EQ(ptr_2, nullptr);

  auto ptr_3 = s.CreateAttrsGroup<TestAttrGroup>(1, 2);
  ASSERT_EQ(ptr_3, nullptr);

  auto ptr_4 = s.GetAttrsGroup<TestAttrGroup>();
  ASSERT_EQ(ptr_4, ptr);

  auto ptr_5 = s.GetOrCreateAttrsGroup<TestAttrGroup>();
  ASSERT_EQ(ptr_5, ptr);
}

TEST_F(AttrHolderUt, CovSetAttrEmptyValueFailed) {
  SubAttrHolder holder;
  AnyValue empty_av;
  EXPECT_EQ(holder.SetAttr("empty_key", empty_av), GRAPH_FAILED);
}

TEST_F(AttrHolderUt, CovTrySetAttrEmptyValueFailed) {
  SubAttrHolder holder;
  AnyValue empty_av;
  EXPECT_EQ(holder.TrySetAttr("empty_key", empty_av), GRAPH_FAILED);
}

TEST_F(AttrHolderUt, CovTrySetAttrAlreadyExists) {
  SubAttrHolder holder;
  AnyValue av = AnyValue::CreateFrom<int>(42);
  EXPECT_EQ(holder.SetAttr("existing_key", av), GRAPH_SUCCESS);
  AnyValue av2 = AnyValue::CreateFrom<int>(99);
  EXPECT_EQ(holder.TrySetAttr("existing_key", av2), GRAPH_SUCCESS);
  AnyValue got;
  EXPECT_EQ(holder.GetAttr("existing_key", got), GRAPH_SUCCESS);
}

TEST_F(AttrHolderUt, CovGetAttrNonExistent) {
  SubAttrHolder holder;
  AnyValue av;
  EXPECT_EQ(holder.GetAttr("nonexistent_key", av), GRAPH_FAILED);
}

TEST_F(AttrHolderUt, CovDelAttrNonExistent) {
  SubAttrHolder holder;
  EXPECT_EQ(holder.DelAttr("nonexistent_key"), GRAPH_FAILED);
}

TEST_F(AttrHolderUt, CovDelAttrExistent) {
  SubAttrHolder holder;
  AnyValue av = AnyValue::CreateFrom<int>(42);
  EXPECT_EQ(holder.SetAttr("key_to_delete", av), GRAPH_SUCCESS);
  EXPECT_EQ(holder.DelAttr("key_to_delete"), GRAPH_SUCCESS);
  EXPECT_EQ(holder.DelAttr("key_to_delete"), GRAPH_FAILED);
}

TEST_F(AttrHolderUt, CovHasAttrNonExistent) {
  SubAttrHolder holder;
  EXPECT_FALSE(holder.HasAttr("nonexistent_key"));
}

TEST_F(AttrHolderUt, CovAddRequiredAttrDuplicate) {
  SubAttrHolder holder;
  EXPECT_EQ(holder.AddRequiredAttr("req_attr"), GRAPH_SUCCESS);
  EXPECT_EQ(holder.AddRequiredAttr("req_attr"), GRAPH_FAILED);
  EXPECT_TRUE(holder.HasAttr("req_attr"));
}

TEST_F(AttrHolderUt, CovAddRequiredAttrWithType) {
  SubAttrHolder holder;
  EXPECT_EQ(holder.AddRequiredAttrWithType("typed_attr", "Int"), GRAPH_SUCCESS);
  EXPECT_EQ(holder.AddRequiredAttrWithType("typed_attr", "Float"), GRAPH_FAILED);
  EXPECT_TRUE(holder.HasAttr("typed_attr"));
}

TEST_F(AttrHolderUt, CovCopyAttrsFrom) {
  SubAttrHolder src;
  AnyValue av = AnyValue::CreateFrom<int>(42);
  EXPECT_EQ(src.SetAttr("copy_key", av), GRAPH_SUCCESS);

  SubAttrHolder dst;
  dst.CopyAttrsFrom(src);
  AnyValue got;
  EXPECT_EQ(dst.GetAttr("copy_key", got), GRAPH_SUCCESS);
}

TEST_F(AttrHolderUt, CovCopyFrom) {
  SubAttrHolder src;
  EXPECT_EQ(src.AddRequiredAttrWithType("copy_req", "Int"), GRAPH_SUCCESS);

  SubAttrHolder dst;
  dst.CopyFrom(src);
  EXPECT_TRUE(dst.HasAttr("copy_req"));
}

TEST_F(AttrHolderUt, CovGetAllAttrs) {
  SubAttrHolder holder;
  AnyValue av1 = AnyValue::CreateFrom<int>(1);
  AnyValue av2 = AnyValue::CreateFrom<int>(2);
  EXPECT_EQ(holder.SetAttr("key1", av1), GRAPH_SUCCESS);
  EXPECT_EQ(holder.SetAttr("key2", av2), GRAPH_SUCCESS);
  auto all_attrs = holder.GetAllAttrs();
  EXPECT_EQ(all_attrs.size(), 2U);
}

TEST_F(AttrHolderUt, CovGetAllAttrNames) {
  SubAttrHolder holder;
  AnyValue av = AnyValue::CreateFrom<int>(1);
  EXPECT_EQ(holder.SetAttr("name_key", av), GRAPH_SUCCESS);
  EXPECT_EQ(holder.AddRequiredAttr("req_name"), GRAPH_SUCCESS);
  auto names = holder.GetAllAttrNames();
  EXPECT_FALSE(names.empty());
}

TEST_F(AttrHolderUt, CovGetAllAttrsWithFilter) {
  SubAttrHolder holder;
  AnyValue av1 = AnyValue::CreateFrom<int>(1);
  AnyValue av2 = AnyValue::CreateFrom<int>(2);
  EXPECT_EQ(holder.SetAttr("keep_key", av1), GRAPH_SUCCESS);
  EXPECT_EQ(holder.SetAttr("skip_key", av2), GRAPH_SUCCESS);
  AttrNameFilter filter = [](const std::string &attr_name) { return attr_name == "keep_key"; };
  auto filtered = holder.GetAllAttrsWithFilter(filter);
  EXPECT_EQ(filtered.size(), 1U);
  EXPECT_NE(filtered.find("keep_key"), filtered.end());
}
}  // namespace ge
