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
#include "util/tenary_op.h"
namespace att {
class TenaryOpsUtilsUnitTest : public testing::Test {
 public:
  void SetUp() override {}
  void TearDown() override {
  }
};

TEST_F(TenaryOpsUtilsUnitTest, TestConcursiveReplaceVars) {
  std::map<Expr, TenaryOp, ExprCmp> tenary_ops;
  Expr res1 = CreateExpr("res1");
  Expr res2 = CreateExpr("res2");
  Expr res3 = CreateExpr("res3");
  Expr expr1 = CreateExpr("expr1");
  Expr expr2 = CreateExpr("expr2");
  Expr expr3 = CreateExpr("expr3");
  TenaryOp tenary_op1 = TenaryOp(expr1 + res2);
  tenary_op1.SetVariable(res1);
  tenary_ops[res1] = tenary_op1;
  TenaryOp tenary_op2 = TenaryOp(CondType::K_EQ, expr2, CreateExpr(2), res3, expr3);
  tenary_op2.SetVariable(res2);
  tenary_ops[res2] = tenary_op2;
  TenaryOp tenary_op3 = TenaryOp(CreateExpr(3));
  tenary_op3.SetVariable(res3);
  tenary_ops[res3] = tenary_op3;
  auto res = ConcursiveReplaceVars(tenary_ops);
  EXPECT_TRUE(!res.empty());
  EXPECT_EQ(Str(res1.Replace(res)), "(TenaryOp(IsEqual(expr2, 2), 3, expr3) + expr1)");
  EXPECT_EQ(Str(res2.Replace(res)), "TenaryOp(IsEqual(expr2, 2), 3, expr3)");
  EXPECT_EQ(Str(res3.Replace(res)), "3");
}

TEST_F(TenaryOpsUtilsUnitTest, TestConcursiveReplaceVars2) {
  std::map<Expr, TenaryOp, ExprCmp> tenary_ops;
  Expr res1 = CreateExpr("res1");
  Expr res2 = CreateExpr("res2");
  Expr res3 = CreateExpr("res3");
  Expr res4 = CreateExpr("res4");
  Expr expr1 = CreateExpr("expr1");
  Expr expr2 = CreateExpr("expr2");
  Expr expr3 = CreateExpr("expr3");
  Expr expr4 = CreateExpr("expr4");
  Expr expr5 = CreateExpr("expr5");
  TenaryOp tenary_op1 = TenaryOp(res3 + res2);
  tenary_op1.SetVariable(res1);
  tenary_ops[res1] = tenary_op1;
  TenaryOp tenary_op2 = TenaryOp(CondType::K_EQ, expr2, res3, expr4, expr3);
  tenary_op2.SetVariable(res2);
  tenary_ops[res2] = tenary_op2;
  TenaryOp tenary_op3 = TenaryOp(CondType::K_LE, expr4, expr5, expr1, expr3);
  tenary_op3.SetVariable(res3);
  tenary_ops[res3] = tenary_op3;
  TenaryOp tenary_op4 = TenaryOp(CondType::K_GE, expr4, expr5, expr1, expr3);
  tenary_op4.SetVariable(res4);
  tenary_ops[res4] = tenary_op4;
  auto res = ConcursiveReplaceVars(tenary_ops);
  EXPECT_TRUE(!res.empty());
  EXPECT_EQ(Str(res1.Replace(res)), "(TenaryOp(IsEqual(expr2, TenaryOp(expr4 <= expr5, expr1, expr3)), expr4, expr3) + TenaryOp(expr4 <= expr5, expr1, expr3))");
  EXPECT_EQ(Str(res2.Replace(res)), "TenaryOp(IsEqual(expr2, TenaryOp(expr4 <= expr5, expr1, expr3)), expr4, expr3)");
  EXPECT_EQ(Str(res3.Replace(res)), "TenaryOp(expr4 <= expr5, expr1, expr3)");
  EXPECT_EQ(Str(res4.Replace(res)), "TenaryOp(expr4 >= expr5, expr1, expr3)");
}

TEST_F(TenaryOpsUtilsUnitTest, TestConcursiveRelatedVars) {
  std::map<Expr, TenaryOp, ExprCmp> tenary_ops;
  Expr res1 = CreateExpr("res1");
  Expr res2 = CreateExpr("res2");
  Expr res3 = CreateExpr("res3");
  Expr expr1 = CreateExpr("expr1");
  Expr expr2 = CreateExpr("expr2");
  Expr expr3 = CreateExpr("expr3");
  Expr expr4 = CreateExpr("expr4");
  Expr expr5 = CreateExpr("expr5");
  TenaryOp tenary_op1 = TenaryOp(res3 + res2);
  tenary_op1.SetVariable(res1);
  tenary_ops[res1] = tenary_op1;
  TenaryOp tenary_op2 = TenaryOp(CondType::K_EQ, expr2, res3, expr4, expr3);
  tenary_op2.SetVariable(res2);
  tenary_ops[res2] = tenary_op2;
  TenaryOp tenary_op3 = TenaryOp(CondType::K_LE, expr4, expr5, expr1, expr3);
  tenary_op3.SetVariable(res3);
  tenary_ops[res3] = tenary_op3;
  auto res = ConcursiveRelatedVars(tenary_ops);
  EXPECT_TRUE(!res.empty());
  EXPECT_EQ(GetVecString(res[res1]), "expr4,expr5,expr1,expr3,expr2,expr4,expr5,expr1,expr3,expr4,expr3,");
  EXPECT_EQ(GetVecString(res[res2]), "expr2,expr4,expr5,expr1,expr3,expr4,expr3,");
  EXPECT_EQ(GetVecString(res[res3]), "expr4,expr5,expr1,expr3,");
}

TEST_F(TenaryOpsUtilsUnitTest, TestUpdateReplaceVars) {
  std::map<Expr, TenaryOp, ExprCmp> tenary_ops;
  Expr rec = CreateExpr("rec");
  Expr res1 = CreateExpr("res1");
  Expr expr1 = CreateExpr("expr1");
  TenaryOp tenary_op1 = TenaryOp(expr1 + res1);
  std::vector<std::pair<Expr, Expr>> expr_map;
  expr_map.emplace_back(std::make_pair(res1, rec));
  tenary_op1.SetVariable(res1);
  tenary_ops[res1] = tenary_op1;
  tenary_ops[res1].UpdateRelatedVars(expr_map);
  auto related_maps = tenary_ops[res1].GetRelatedVars();
  EXPECT_TRUE(find(related_maps.begin(), related_maps.end(), rec) != related_maps.end());
}
} //namespace