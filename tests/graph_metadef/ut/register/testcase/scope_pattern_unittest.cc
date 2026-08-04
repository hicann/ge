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
#include <gmock/gmock.h>
#include <vector>

#include "register/scope/scope_pattern_impl.h"
#include "register/scope/scope_graph_impl.h"
#include "framework/common/debug/ge_log.h"
#include "graph/types.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/attr_utils.h"
#include "register/scope/scope_fusion_pass_register.h"

using namespace std;
using namespace testing;

namespace ge {

class ScopePatternUt : public testing::Test {
 protected:
  void SetUp() {}
  void TearDown() {}
};

TEST_F(ScopePatternUt, ScopeAttrValue1) {
  ScopeAttrValue scope_attr_value;

  float32_t value = 0.2;
  scope_attr_value.SetFloatValue(value);
  EXPECT_EQ(scope_attr_value.impl_->GetFloatValue(), static_cast<float32_t>(0.2));

  int64_t value2 = 2;
  scope_attr_value.SetIntValue(value2);
  EXPECT_EQ(scope_attr_value.impl_->GetIntValue(), 2);

  scope_attr_value.SetStringValue("abc");
  EXPECT_EQ(scope_attr_value.impl_->GetStrValue(), string("abc"));

  scope_attr_value.SetStringValue(string("def"));
  EXPECT_EQ(scope_attr_value.impl_->GetStrValue(), string("def"));

  scope_attr_value.SetBoolValue(true);
  EXPECT_TRUE(scope_attr_value.impl_->GetBoolValue());

  ScopeAttrValue scope_attr_value2(scope_attr_value);
  EXPECT_EQ(scope_attr_value2.impl_->GetFloatValue(), static_cast<float32_t>(0.2));
  EXPECT_EQ(scope_attr_value2.impl_->GetIntValue(), 2);
  EXPECT_EQ(scope_attr_value2.impl_->GetStrValue(), string("def"));
  EXPECT_TRUE(scope_attr_value2.impl_->GetBoolValue());

  ScopeAttrValue scope_attr_value3;
  scope_attr_value3 = scope_attr_value;
  EXPECT_EQ(scope_attr_value3.impl_->GetFloatValue(), static_cast<float32_t>(0.2));
  EXPECT_EQ(scope_attr_value3.impl_->GetIntValue(), 2);
  EXPECT_EQ(scope_attr_value3.impl_->GetStrValue(), string("def"));
  EXPECT_TRUE(scope_attr_value3.impl_->GetBoolValue());
}

TEST_F(ScopePatternUt, ScopeAttrValue2) {
  ScopeAttrValue scope_attr_value;
  scope_attr_value.impl_ = nullptr;

  float32_t value = 0.2;
  scope_attr_value.SetFloatValue(value);

  int64_t value2 = 2;
  scope_attr_value.SetIntValue(value2);
  scope_attr_value.SetStringValue("abc");
  scope_attr_value.SetStringValue(string("def"));
  scope_attr_value.SetBoolValue(true);

  EXPECT_TRUE(scope_attr_value.impl_ == nullptr);
}

TEST_F(ScopePatternUt, NodeOpTypeFeature) {
  // construct
  string nodeType = string("add");
  int32_t num = 1;
  int32_t step = 100;
  NodeOpTypeFeature notf(nodeType, num, step);
  EXPECT_EQ(notf.impl_->step_, step);
  NodeOpTypeFeature notf2("edf", num, 0);
  EXPECT_EQ(notf2.impl_->node_type_, string("edf"));

  // match
  Scope scope;
  scope.Init("name", "sub_type", nullptr);
  EXPECT_FALSE(notf.Match(nullptr));
  EXPECT_FALSE(notf.Match(&scope));
  EXPECT_FALSE(notf2.Match(&scope));

  // copy
  NodeOpTypeFeature notf3(notf);
  EXPECT_EQ(notf3.impl_->node_type_, string("add"));
  notf3 = notf3;
  notf3 = notf2;
  EXPECT_EQ(notf3.impl_->node_type_, string("edf"));

  notf3.impl_.reset();
  EXPECT_FALSE(notf3.Match(nullptr));
  EXPECT_EQ(notf3.impl_, nullptr);
}

TEST_F(ScopePatternUt, NodeAttrFeature) {
  // construct
  ScopeAttrValue scope_attr_value;
  scope_attr_value.SetStringValue("abc");
  NodeAttrFeature naf("node_type", "attr_name", DT_INT8, scope_attr_value);
  NodeAttrFeature naf2(string("node_type_2"), string("attr_name_2"), DT_INT8, scope_attr_value);
  EXPECT_EQ(naf.impl_->attr_value_.impl_->GetStrValue(), string("abc"));

  // copy
  NodeAttrFeature naf3(naf2);
  EXPECT_EQ(naf3.impl_->node_type_, string("node_type_2"));
  naf3 = naf3;
  naf3 = naf;
  EXPECT_EQ(naf3.impl_->attr_name_, string("attr_name"));

  // match
  Scope scope;
  scope.Init("name", "sub_type", nullptr);
  EXPECT_FALSE(naf3.impl_->Match(nullptr));
  EXPECT_FALSE(naf3.impl_->Match(&scope));
}

TEST_F(ScopePatternUt, CheckNodeAttrFeatureData) {
  ScopeAttrValue scope_attr_value;
  scope_attr_value.SetStringValue("abc");
  NodeAttrFeature naf("node_type", "attr_name", DT_INT8, scope_attr_value);

  bool init_value = true;
  ge::OpDescPtr op_desc(new ge::OpDesc("add1", "Add"));
  Scope scope;
  scope.Init("name", "sub_type", nullptr);

  auto ret = naf.impl_->CheckNodeAttrFeatureData(init_value, op_desc, &scope);
  EXPECT_EQ(ret, PARAM_INVALID);

  string init_value2 = "init_value";
  ret = naf.impl_->CheckNodeAttrFeatureData(init_value2, op_desc, &scope);
  EXPECT_EQ(ret, PARAM_INVALID);

  int64_t init_value3 = 1;
  ret = naf.impl_->CheckNodeAttrFeatureData(init_value3, op_desc, &scope);
  EXPECT_EQ(ret, PARAM_INVALID);

  float32_t init_value4 = 0.2;
  ret = naf.impl_->CheckNodeAttrFeatureData(init_value4, op_desc, &scope);
  EXPECT_EQ(ret, PARAM_INVALID);

  // match
  EXPECT_FALSE(naf.Match(nullptr));
  EXPECT_FALSE(naf.Match(&scope));
}

TEST_F(ScopePatternUt, CheckNodeAttrFeatureDataSuccess) {
  {
    ScopeAttrValue scope_attr_value;
    bool init_value = true;
    scope_attr_value.SetBoolValue(init_value);
    string attr_name("attr_name");
    NodeAttrFeature naf("node_type", attr_name, DT_INT8, scope_attr_value);

    ge::OpDescPtr op_desc(new ge::OpDesc("add1", "Add"));
    ge::AttrUtils::SetBool(op_desc, attr_name, init_value);
    Scope scope;
    scope.Init("name", "sub_type", nullptr);

    auto ret = naf.impl_->CheckNodeAttrFeatureData(init_value, op_desc, &scope);
    EXPECT_EQ(ret, SUCCESS);
  }
  {
    ScopeAttrValue scope_attr_value;
    string init_value = "true";
    scope_attr_value.SetStringValue(init_value.c_str());
    string attr_name("attr_name");
    NodeAttrFeature naf("node_type", attr_name, DT_INT8, scope_attr_value);

    ge::OpDescPtr op_desc(new ge::OpDesc("add1", "Add"));
    ge::AttrUtils::SetStr(op_desc, attr_name, init_value);
    Scope scope;
    scope.Init("name", "sub_type", nullptr);

    auto ret = naf.impl_->CheckNodeAttrFeatureData(init_value, op_desc, &scope);
    EXPECT_EQ(ret, SUCCESS);
  }
  {
    ScopeAttrValue scope_attr_value;
    float32_t init_value = 0.0f;
    scope_attr_value.SetFloatValue(init_value);
    string attr_name("attr_name");
    NodeAttrFeature naf("node_type", attr_name, DT_INT8, scope_attr_value);

    ge::OpDescPtr op_desc(new ge::OpDesc("add1", "Add"));
    ge::AttrUtils::SetFloat(op_desc, attr_name, init_value);
    Scope scope;
    scope.Init("name", "sub_type", nullptr);

    auto ret = naf.impl_->CheckNodeAttrFeatureData(init_value, op_desc, &scope);
    EXPECT_EQ(ret, SUCCESS);
  }
  {
    ScopeAttrValue scope_attr_value;
    int64_t init_value = 0;
    scope_attr_value.SetIntValue(init_value);
    string attr_name("attr_name");
    NodeAttrFeature naf("node_type", attr_name, DT_INT8, scope_attr_value);

    ge::OpDescPtr op_desc(new ge::OpDesc("add1", "Add"));
    ge::AttrUtils::SetInt(op_desc, attr_name, init_value);
    Scope scope;
    scope.Init("name", "sub_type", nullptr);

    auto ret = naf.impl_->CheckNodeAttrFeatureData(init_value, op_desc, &scope);
    EXPECT_EQ(ret, SUCCESS);
  }
}

TEST_F(ScopePatternUt, ScopeFeature) {
  // construct
  string sub_type = "sub_type";
  int32_t num = 3;
  string suffix = "suffix";
  string sub_scope_mask = "sub_scope_mask";
  int32_t step = 0;

  ScopeFeature sf(sub_type, num, suffix, sub_scope_mask, step);
  EXPECT_EQ(sf.impl_->sub_type_, sub_type);

  ScopeFeature sf2("sub_type_2", num, "suffix_2", "sub_scope_mask_2", step);
  EXPECT_EQ(sf2.impl_->sub_type_, string("sub_type_2"));

  // copy
  ScopeFeature sf3(sf2);
  EXPECT_EQ(sf3.impl_->sub_type_, string("sub_type_2"));

  sf2 = sf2;
  sf2 = sf;
  EXPECT_EQ(sf2.impl_->sub_type_, sub_type);

  // match
  Scope scope;
  scope.Init("name", "sub_type", nullptr);
  EXPECT_FALSE(sf.Match(&scope));
}

TEST_F(ScopePatternUt, ScopeFeature_Match) {
  std::vector<Scope *> scopes;
  Scope scope;
  scope.Init("name", "sub_type", nullptr);
  scopes.emplace_back(&scope);
  Scope scope2;
  scope2.Init("name_2", "sub_type_2", nullptr);
  scopes.emplace_back(&scope2);

  ScopeFeature sf2("sub_type_2", 1, "suffix_2", "sub_scope_mask_2", 1);
  auto ret = sf2.impl_->SubScopesMatch(scopes);
  EXPECT_FALSE(ret);
}

TEST_F(ScopePatternUt, ScopePattern) {
  ScopePattern scope_pat;
  EXPECT_NE(scope_pat.impl_, nullptr);

  scope_pat.SetSubType("sub_type");
  scope_pat.SetSubType(string("sub_type_2"));
  EXPECT_EQ(scope_pat.impl_->sub_type_, string("sub_type_2"));

  scope_pat.impl_.reset();
  scope_pat.SetSubType("sub_type");
  scope_pat.SetSubType(string("sub_type_2"));
  EXPECT_EQ(scope_pat.impl_, nullptr);
}

TEST_F(ScopePatternUt, AddFeature) {
  ScopePattern scope_pat;

  NodeOpTypeFeature notf("abc", 1, 0);
  scope_pat.AddNodeOpTypeFeature(notf);
  EXPECT_TRUE(scope_pat.impl_->node_optype_features_.size() > 0);

  ScopeAttrValue scope_attr_value;
  scope_attr_value.SetStringValue("abc");
  NodeAttrFeature naf("node_type", "attr_name", DT_INT8, scope_attr_value);
  scope_pat.AddNodeAttrFeature(naf);
  EXPECT_TRUE(scope_pat.impl_->node_attr_features_.size() > 0);

  ScopeFeature sf("sub_type", 1, "suffix", "sub_scope_mask", 1);
  scope_pat.AddScopeFeature(sf);
  EXPECT_TRUE(scope_pat.impl_->scopes_features_.size() > 0);
}

TEST_F(ScopePatternUt, AddFeature_Null) {
  ScopePattern scope_pat;
  scope_pat.impl_.reset();

  NodeOpTypeFeature notf("abc", 1, 0);
  scope_pat.AddNodeOpTypeFeature(notf);

  ScopeAttrValue scope_attr_value;
  scope_attr_value.SetStringValue("abc");
  NodeAttrFeature naf("node_type", "attr_name", DT_INT8, scope_attr_value);
  scope_pat.AddNodeAttrFeature(naf);

  ScopeFeature sf("sub_type", 1, "suffix", "sub_scope_mask", 1);
  scope_pat.AddScopeFeature(sf);

  EXPECT_EQ(scope_pat.impl_, nullptr);
}

TEST_F(ScopePatternUt, IncCov_NodeAttrFeature_NullImpl) {
  ScopeAttrValue empty_val;
  NodeAttrFeature naf("node_type", "attr_name", DT_INT8, empty_val);
  naf.impl_.reset();
  Scope scope;
  scope.Init("name", "sub_type", nullptr);
  EXPECT_FALSE(naf.Match(&scope));

  ScopeAttrValue other_val;
  NodeAttrFeature assigned("other", "other_attr", DT_INT8, other_val);
  assigned.impl_.reset();
  assigned = naf;
}

TEST_F(ScopePatternUt, IncCov_ScopeFeature_MatchFull) {
  Scope root;
  root.Init("parent/child", "", nullptr);
  Scope *sub1 = new Scope();
  sub1->Init("parent/child/sub1", "my_type", &root);
  sub1->impl_->SetSubType("my_type");
  root.impl_->AddSubScope(sub1);

  ScopeFeature sf("my_type", 1, "parent", "", 0);
  EXPECT_TRUE(sf.impl_->Match(&root));

  ScopeFeature sf2("my_type", 2, "parent", "", 0);
  EXPECT_FALSE(sf2.impl_->Match(&root));
}

TEST_F(ScopePatternUt, IncCov_ScopeFeature_SubScopesMatchContinueAndTrue) {
  Scope *sub1 = new Scope();
  sub1->Init("parent/child_a", "my_type", nullptr);
  sub1->impl_->SetSubType("my_type");
  Scope *sub2 = new Scope();
  sub2->Init("parent/child_b", "my_type", nullptr);
  sub2->impl_->SetSubType("my_type");

  std::vector<Scope *> scopes = {sub1, sub2};
  ScopeFeature sf("my_type", 2, "", "parent", 0);
  EXPECT_TRUE(sf.impl_->SubScopesMatch(scopes));

  ScopeFeature sf2("my_type", 1, "", "parent", 0);
  EXPECT_FALSE(sf2.impl_->SubScopesMatch(scopes));

  delete sub1;
  delete sub2;
}

TEST_F(ScopePatternUt, IncCov_ScopePattern_MatchEdgeCases) {
  ScopePattern scope_pat;
  EXPECT_FALSE(scope_pat.impl_->Match(nullptr));

  Scope scope;
  scope.Init("name", "sub_type", nullptr);
  OperatorPtr node(new ge::Operator("add1", "Add"));
  scope.impl_->AddNode(node);
  scope.impl_->OpsNumInc("Add");

  ScopeAttrValue attr_val;
  attr_val.SetBoolValue(true);
  NodeAttrFeature naf("Mul", "attr_name", DT_BOOL, attr_val);
  scope_pat.AddNodeAttrFeature(naf);
  EXPECT_FALSE(scope_pat.impl_->Match(&scope));

  ScopePattern scope_pat2;
  ScopeFeature sf("nonexistent", 1, "suffix", "mask", 0);
  scope_pat2.AddScopeFeature(sf);
  EXPECT_FALSE(scope_pat2.impl_->Match(&scope));

  Scope scope_retval;
  scope_retval.Init("retval_scope", "", nullptr);
  scope_retval.impl_->OpsNumInc("_Retval");
  ScopePattern scope_pat3;
  EXPECT_FALSE(scope_pat3.impl_->Match(&scope_retval));
}

TEST_F(ScopePatternUt, IncCov_ScopeAttrValue_SelfAssignment) {
  ScopeAttrValue scope_attr_value;
  scope_attr_value.SetIntValue(42);
  scope_attr_value.SetFloatValue(1.5F);
  scope_attr_value.SetStringValue("test");
  scope_attr_value.SetBoolValue(true);
  scope_attr_value = scope_attr_value;
  EXPECT_EQ(scope_attr_value.impl_->GetIntValue(), 42);
  EXPECT_EQ(scope_attr_value.impl_->GetStrValue(), string("test"));
}

TEST_F(ScopePatternUt, IncCov_ScopeAttrValue_AssignNullImpl) {
  ScopeAttrValue src_val;
  src_val.SetIntValue(42);
  ScopeAttrValue null_val;
  null_val.impl_ = nullptr;
  null_val = src_val;
  EXPECT_EQ(null_val.impl_, nullptr);
}

TEST_F(ScopePatternUt, IncCov_NodeOpTypeFeature_StepMatchSuccess) {
  Scope scope;
  scope.Init("step_scope", "sub_type", nullptr);
  scope.impl_->OpsNumInc("Conv2D");
  scope.impl_->OpsNumInc("Conv2D");
  scope.impl_->OpsNumInc("Conv2D");
  NodeOpTypeFeature notf("Conv2D", 1, 2);
  EXPECT_TRUE(notf.Match(&scope));
}

TEST_F(ScopePatternUt, IncCov_NodeOpTypeFeature_AssignNullImpl) {
  NodeOpTypeFeature notf1("add", 1, 0);
  NodeOpTypeFeature notf2("sub", 2, 1);
  notf2.impl_.reset();
  notf2 = notf1;
  EXPECT_EQ(notf2.impl_, nullptr);
}

TEST_F(ScopePatternUt, IncCov_NodeAttrFeature_MatchWithNodes) {
  Scope scope;
  scope.Init("attr_scope", "sub_type", nullptr);
  OperatorPtr node(new ge::Operator("add1", "Add"));
  auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(*node);
  ge::AttrUtils::SetBool(op_desc, "attr_name", true);
  scope.impl_->AddNode(node);

  ScopeAttrValue bool_val;
  bool_val.SetBoolValue(true);
  NodeAttrFeature naf_bool("Add", "attr_name", DT_BOOL, bool_val);
  EXPECT_TRUE(naf_bool.Match(&scope));

  ScopeAttrValue bool_val_false;
  bool_val_false.SetBoolValue(false);
  NodeAttrFeature naf_false("Add", "attr_name", DT_BOOL, bool_val_false);
  EXPECT_FALSE(naf_false.Match(&scope));

  ge::AttrUtils::SetInt(op_desc, "attr_name", 42);
  ScopeAttrValue int_val;
  int_val.SetIntValue(42);
  NodeAttrFeature naf_int("Add", "attr_name", DT_INT32, int_val);
  EXPECT_TRUE(naf_int.Match(&scope));

  ge::AttrUtils::SetFloat(op_desc, "attr_name", 1.5F);
  ScopeAttrValue float_val;
  float_val.SetFloatValue(1.5F);
  NodeAttrFeature naf_float("Add", "attr_name", DT_FLOAT, float_val);
  EXPECT_TRUE(naf_float.Match(&scope));

  ScopeAttrValue int8_val;
  int8_val.SetBoolValue(true);
  NodeAttrFeature naf_int8("Add", "attr_name", DT_INT8, int8_val);
  EXPECT_TRUE(naf_int8.Match(&scope));

  NodeAttrFeature naf_wrong("Mul", "attr_name", DT_BOOL, bool_val);
  EXPECT_FALSE(naf_wrong.Match(&scope));
}

TEST_F(ScopePatternUt, IncCov_ScopeFeature_SuffixNotMatch) {
  Scope scope;
  scope.Init("parent/child", "", nullptr);
  ScopeFeature sf("", 0, "wrong_suffix", "", 0);
  EXPECT_FALSE(sf.Match(&scope));
}

TEST_F(ScopePatternUt, IncCov_ScopeFeature_AssignNullImpl) {
  ScopeFeature sf1("sub_type", 1, "suffix", "mask", 0);
  ScopeFeature sf2("other", 2, "other_suffix", "other_mask", 1);
  sf2.impl_.reset();
  sf2 = sf1;
  EXPECT_EQ(sf2.impl_, nullptr);
}

TEST_F(ScopePatternUt, IncCov_ScopeFeature_MatchNullImpl) {
  ScopeFeature sf("sub_type", 1, "suffix", "mask", 0);
  sf.impl_.reset();
  Scope scope;
  scope.Init("name", "sub_type", nullptr);
  EXPECT_FALSE(sf.Match(&scope));
}
}  // namespace ge
