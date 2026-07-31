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
#include "graph/ir/ir_data_type_symbol_store.h"
#include "graph/type/sym_dtype.h"
#include "graph/op_desc.h"
#include "graph/utils/type_utils.h"
#include "graph/operator_reg.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/normal_graph/op_desc_impl.h"

namespace ge {

class UtestIRDataTypeSymbolStore : public testing::Test {
 protected:
  void SetUp() {}
  void TearDown() {}
};

TEST_F(UtestIRDataTypeSymbolStore, CovDeclareSymbolOrderedTensorTypeList) {
  IRDataTypeSymbolStore store;
  OrderedTensorTypeList types({DT_FLOAT, DT_INT32, DT_INT64});
  auto *sym = store.DeclareSymbol("ordered_sym", types);
  ASSERT_NE(sym, nullptr);
  EXPECT_EQ(sym->Id(), "ordered_sym");
  EXPECT_TRUE(sym->IsOrderedList());
  auto named_syms = store.GetNamedSymbols();
  EXPECT_EQ(named_syms.size(), 1U);
}

TEST_F(UtestIRDataTypeSymbolStore, CovIsSupportOrderedSymbolicInferDtypeEmpty) {
  IRDataTypeSymbolStore store;
  EXPECT_FALSE(store.IsSupportOrderedSymbolicInferDtype());
}

TEST_F(UtestIRDataTypeSymbolStore, CovIsSupportOrderedSymbolicInferDtypeNoOrderedList) {
  IRDataTypeSymbolStore store;
  TensorType types({DT_FLOAT, DT_INT32});
  store.DeclareSymbol("sym1", types);
  store.SetInputSymbol("x", kIrInputRequired, "sym1");
  EXPECT_FALSE(store.IsSupportOrderedSymbolicInferDtype());
}

TEST_F(UtestIRDataTypeSymbolStore, CovIsSupportOrderedSymbolicInferDtypeSingleOrderedList) {
  IRDataTypeSymbolStore store;
  OrderedTensorTypeList types({DT_FLOAT, DT_INT32});
  store.DeclareSymbol("ordered_sym", types);
  EXPECT_TRUE(store.IsSupportOrderedSymbolicInferDtype());
}

TEST_F(UtestIRDataTypeSymbolStore, CovIsSupportOrderedSymbolicInferDtypeAllOrderedSameSize) {
  IRDataTypeSymbolStore store;
  OrderedTensorTypeList types1({DT_FLOAT, DT_INT32});
  OrderedTensorTypeList types2({DT_FLOAT16, DT_INT8});
  store.DeclareSymbol("ordered_sym1", types1);
  store.DeclareSymbol("ordered_sym2", types2);
  EXPECT_TRUE(store.IsSupportOrderedSymbolicInferDtype());
}

TEST_F(UtestIRDataTypeSymbolStore, CovIsSupportOrderedSymbolicInferDtypeMixedOrderedAndNonOrdered) {
  IRDataTypeSymbolStore store;
  OrderedTensorTypeList ordered_types({DT_FLOAT, DT_INT32});
  store.DeclareSymbol("ordered_sym", ordered_types);
  TensorType regular_types({DT_FLOAT});
  store.DeclareSymbol("regular_sym", regular_types);
  store.SetInputSymbol("x", kIrInputRequired, "regular_sym");
  EXPECT_FALSE(store.IsSupportOrderedSymbolicInferDtype());
}

TEST_F(UtestIRDataTypeSymbolStore, CovIsSupportOrderedSymbolicInferDtypeDifferentSizes) {
  IRDataTypeSymbolStore store;
  OrderedTensorTypeList types1({DT_FLOAT, DT_INT32});
  OrderedTensorTypeList types2({DT_FLOAT16, DT_INT8, DT_INT64});
  store.DeclareSymbol("ordered_sym1", types1);
  store.DeclareSymbol("ordered_sym2", types2);
  EXPECT_FALSE(store.IsSupportOrderedSymbolicInferDtype());
}

TEST_F(UtestIRDataTypeSymbolStore, CovIsSupportOrderedSymbolicInferDtypeEmptyOrderedList) {
  IRDataTypeSymbolStore store;
  OrderedTensorTypeList empty_types({});
  store.DeclareSymbol("empty_ordered_sym", empty_types);
  EXPECT_FALSE(store.IsSupportOrderedSymbolicInferDtype());
}

TEST_F(UtestIRDataTypeSymbolStore, CovIsSupportSymbolicInferDtype) {
  IRDataTypeSymbolStore empty_store;
  EXPECT_FALSE(empty_store.IsSupportSymbolicInferDtype());

  IRDataTypeSymbolStore store;
  TensorType types({DT_FLOAT, DT_INT32});
  store.DeclareSymbol("sym1", types);
  EXPECT_TRUE(store.IsSupportSymbolicInferDtype());
}

TEST_F(UtestIRDataTypeSymbolStore, CovGetOrCreateSymbolWithQuotes) {
  IRDataTypeSymbolStore store;
  auto *sym1 = store.GetOrCreateSymbol("\"test_sym\"");
  ASSERT_NE(sym1, nullptr);
  EXPECT_EQ(sym1->Id(), "test_sym");
  auto *sym2 = store.GetOrCreateSymbol("test_sym");
  EXPECT_EQ(sym1, sym2);
}

TEST_F(UtestIRDataTypeSymbolStore, CovDeclareSymbolPromote) {
  IRDataTypeSymbolStore store;
  store.SetInputSymbol("x", kIrInputRequired, "sym_x");
  store.SetInputSymbol("y", kIrInputRequired, "sym_y");
  Promote promote({"sym_x", "sym_y"});
  auto *sym = store.DeclareSymbol("promote_sym", promote);
  ASSERT_NE(sym, nullptr);
  EXPECT_EQ(sym->Id(), "promote_sym");
  auto named_syms = store.GetNamedSymbols();
  EXPECT_GE(named_syms.size(), 1U);
}

TEST_F(UtestIRDataTypeSymbolStore, CovDeclareSymbolListTensorType) {
  IRDataTypeSymbolStore store;
  ListTensorType types(TensorType({DT_FLOAT, DT_INT32, DT_INT64}));
  auto *sym = store.DeclareSymbol("list_sym", types);
  ASSERT_NE(sym, nullptr);
  EXPECT_EQ(sym->Id(), "list_sym");
  auto named_syms = store.GetNamedSymbols();
  EXPECT_EQ(named_syms.size(), 1U);
}

TEST_F(UtestIRDataTypeSymbolStore, CovSetOutputSymbol) {
  IRDataTypeSymbolStore store;
  TensorType types({DT_FLOAT});
  store.DeclareSymbol("out_sym", types);
  auto *sym = store.SetOutputSymbol("y", kIrOutputRequired, "out_sym");
  ASSERT_NE(sym, nullptr);
  auto out_syms = store.GetOutSymbols();
  EXPECT_EQ(out_syms.size(), 1U);
}

TEST_F(UtestIRDataTypeSymbolStore, CovGetPromoteIrInputList) {
  IRDataTypeSymbolStore store;
  store.SetInputSymbol("x", kIrInputRequired, "sym_x");
  store.SetInputSymbol("y", kIrInputRequired, "sym_y");
  Promote promote({"sym_x", "sym_y"});
  store.DeclareSymbol("promote_sym", promote);
  std::vector<std::vector<size_t>> promote_index_list;
  auto status = store.GetPromoteIrInputList(promote_index_list);
  EXPECT_EQ(status, GRAPH_SUCCESS);
  EXPECT_EQ(promote_index_list.size(), 1U);
}

TEST_F(UtestIRDataTypeSymbolStore, CovGetPromoteIrInputListEmpty) {
  IRDataTypeSymbolStore store;
  TensorType types({DT_FLOAT});
  store.DeclareSymbol("sym1", types);
  std::vector<std::vector<size_t>> promote_index_list;
  auto status = store.GetPromoteIrInputList(promote_index_list);
  EXPECT_EQ(status, GRAPH_SUCCESS);
  EXPECT_EQ(promote_index_list.size(), 0U);
}

REG_OP(OpTestInferDtypeDynOut)
    .INPUT(x, "T")
    .DYNAMIC_OUTPUT(y, "T")
    .DATATYPE(T, ListTensorType(TensorType({DT_FLOAT, DT_INT32})))
    .OP_END_FACTORY_REG(OpTestInferDtypeDynOut);

TEST_F(UtestIRDataTypeSymbolStore, CovInferDtypeDynamicOutput) {
  auto op = op::OpTestInferDtypeDynOut();
  auto desc = OpDescUtils::GetOpDescFromOperator(op);
  GeTensorDesc input_desc(GeShape({1}), FORMAT_ND, DT_FLOAT);
  desc->UpdateInputDesc("x", input_desc);
  desc->AddOutputDesc("y0", GeTensorDesc(GeShape({1}), FORMAT_ND, DT_FLOAT));
  const auto &sym_store = desc->impl_->GetIRMeta().GetIRDataTypeSymbolStore();
  auto ret = sym_store.InferDtype(desc);
  EXPECT_NE(ret, GRAPH_SUCCESS);
}

TEST_F(UtestIRDataTypeSymbolStore, CovInferDtypeNullptr) {
  IRDataTypeSymbolStore store;
  TensorType types({DT_FLOAT});
  store.DeclareSymbol("sym1", types);
  OpDescPtr null_op;
  auto ret = store.InferDtype(null_op);
  EXPECT_NE(ret, GRAPH_SUCCESS);
}

REG_OP(OpTestInferDtypeRequired)
    .INPUT(x, "T")
    .OUTPUT(y, "T")
    .DATATYPE(T, TensorType({DT_FLOAT, DT_INT32}))
    .OP_END_FACTORY_REG(OpTestInferDtypeRequired);

TEST_F(UtestIRDataTypeSymbolStore, CovInferDtypeRequiredOutput) {
  auto op = op::OpTestInferDtypeRequired();
  auto desc = OpDescUtils::GetOpDescFromOperator(op);
  GeTensorDesc input_desc(GeShape({1}), FORMAT_ND, DT_FLOAT);
  desc->UpdateInputDesc("x", input_desc);
  desc->UpdateOutputDesc("y", GeTensorDesc(GeShape({1}), FORMAT_ND, DT_FLOAT));
  const auto &sym_store = desc->impl_->GetIRMeta().GetIRDataTypeSymbolStore();
  auto ret = sym_store.InferDtype(desc);
  EXPECT_EQ(ret, GRAPH_SUCCESS);
}

REG_OP(OpTestInferDtypePromote)
    .INPUT(x, "T1")
    .INPUT(y, "T2")
    .OUTPUT(z, "T3")
    .DATATYPE(T1, TensorType({DT_FLOAT, DT_INT32}))
    .DATATYPE(T2, TensorType({DT_FLOAT, DT_INT32}))
    .DATATYPE(T3, Promote({"T1", "T2"}))
    .OP_END_FACTORY_REG(OpTestInferDtypePromote);

TEST_F(UtestIRDataTypeSymbolStore, CovInferDtypePromoteOutput) {
  auto op = op::OpTestInferDtypePromote();
  auto desc = OpDescUtils::GetOpDescFromOperator(op);
  GeTensorDesc input_desc1(GeShape({1}), FORMAT_ND, DT_FLOAT);
  GeTensorDesc input_desc2(GeShape({1}), FORMAT_ND, DT_INT32);
  desc->UpdateInputDesc("x", input_desc1);
  desc->UpdateInputDesc("y", input_desc2);
  desc->UpdateOutputDesc("z", GeTensorDesc(GeShape({1}), FORMAT_ND, DT_FLOAT));
  const auto &sym_store = desc->impl_->GetIRMeta().GetIRDataTypeSymbolStore();
  auto ret = sym_store.InferDtype(desc);
  EXPECT_EQ(ret, GRAPH_SUCCESS);
}

TEST_F(UtestIRDataTypeSymbolStore, CovIsSupportOrderedSymbolicInferDtypeNullptrSym) {
  IRDataTypeSymbolStore store;
  OrderedTensorTypeList types({DT_FLOAT, DT_INT32});
  store.DeclareSymbol("ordered_sym", types);
  store.GetOrCreateSymbol("non_ordered_sym");
  EXPECT_FALSE(store.IsSupportOrderedSymbolicInferDtype());
}

}  // namespace ge
