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

#include "register/graph_optimizer/fusion_common/unknown_shape_utils.h"
#include "graph/op_desc.h"
#include "graph/ge_tensor.h"

namespace fe {
class UnknownShapeUtilsCovUT : public testing::Test {
 protected:
  void SetUp() override {}

  void TearDown() override {}
};

TEST_F(UnknownShapeUtilsCovUT, IncCov_IsUnknownShapeValue) {
  EXPECT_TRUE(UnknownShapeUtils::IsUnknownShapeValue(ge::UNKNOWN_DIM));
  EXPECT_TRUE(UnknownShapeUtils::IsUnknownShapeValue(ge::UNKNOWN_DIM_NUM));
  EXPECT_FALSE(UnknownShapeUtils::IsUnknownShapeValue(1));
  EXPECT_FALSE(UnknownShapeUtils::IsUnknownShapeValue(0));
  EXPECT_FALSE(UnknownShapeUtils::IsUnknownShapeValue(100));
  EXPECT_FALSE(UnknownShapeUtils::IsUnknownShapeValue(-3));
}

TEST_F(UnknownShapeUtilsCovUT, IncCov_IsContainUnknownDimNumFalse) {
  ge::OpDescPtr op_desc = std::make_shared<ge::OpDesc>("test_op", "Relu");
  ge::GeTensorDesc input_desc(ge::GeShape({1, 2, 3, 4}));
  op_desc->AddInputDesc(input_desc);
  ge::GeTensorDesc output_desc(ge::GeShape({1, 2, 3, 4}));
  op_desc->AddOutputDesc(output_desc);
  EXPECT_FALSE(UnknownShapeUtils::IsContainUnknownDimNum(*op_desc));
}

TEST_F(UnknownShapeUtilsCovUT, IncCov_IsContainUnknownDimNumInputTrue) {
  ge::OpDescPtr op_desc = std::make_shared<ge::OpDesc>("test_op", "Relu");
  ge::GeShape unknown_shape;
  unknown_shape.SetIsUnknownDimNum();
  ge::GeTensorDesc input_desc(unknown_shape);
  op_desc->AddInputDesc(input_desc);
  ge::GeTensorDesc output_desc(ge::GeShape({1, 2, 3, 4}));
  op_desc->AddOutputDesc(output_desc);
  EXPECT_TRUE(UnknownShapeUtils::IsContainUnknownDimNum(*op_desc));
}

TEST_F(UnknownShapeUtilsCovUT, IncCov_IsContainUnknownDimNumOutputTrue) {
  ge::OpDescPtr op_desc = std::make_shared<ge::OpDesc>("test_op", "Relu");
  ge::GeTensorDesc input_desc(ge::GeShape({1, 2, 3, 4}));
  op_desc->AddInputDesc(input_desc);
  ge::GeShape unknown_shape;
  unknown_shape.SetIsUnknownDimNum();
  ge::GeTensorDesc output_desc(unknown_shape);
  op_desc->AddOutputDesc(output_desc);
  EXPECT_TRUE(UnknownShapeUtils::IsContainUnknownDimNum(*op_desc));
}

TEST_F(UnknownShapeUtilsCovUT, IncCov_IsContainUnknownDimNumEmpty) {
  ge::OpDescPtr op_desc = std::make_shared<ge::OpDesc>("test_op", "Relu");
  EXPECT_FALSE(UnknownShapeUtils::IsContainUnknownDimNum(*op_desc));
}

TEST_F(UnknownShapeUtilsCovUT, IncCov_IsUnknownShapeOpKnownShape) {
  ge::OpDescPtr op_desc = std::make_shared<ge::OpDesc>("test_op", "Relu");
  ge::GeTensorDesc input_desc(ge::GeShape({1, 2, 3, 4}));
  op_desc->AddInputDesc(input_desc);
  ge::GeTensorDesc output_desc(ge::GeShape({1, 2, 3, 4}));
  op_desc->AddOutputDesc(output_desc);
  EXPECT_FALSE(UnknownShapeUtils::IsUnknownShapeOp(*op_desc));
}

TEST_F(UnknownShapeUtilsCovUT, IncCov_IsUnknownShapeOpInputUnknown) {
  ge::OpDescPtr op_desc = std::make_shared<ge::OpDesc>("test_op", "Relu");
  ge::GeTensorDesc input_desc(ge::GeShape({ge::UNKNOWN_DIM, 2, 3, 4}));
  op_desc->AddInputDesc(input_desc);
  ge::GeTensorDesc output_desc(ge::GeShape({1, 2, 3, 4}));
  op_desc->AddOutputDesc(output_desc);
  EXPECT_TRUE(UnknownShapeUtils::IsUnknownShapeOp(*op_desc));
}

TEST_F(UnknownShapeUtilsCovUT, IncCov_IsUnknownShapeOpOutputUnknown) {
  ge::OpDescPtr op_desc = std::make_shared<ge::OpDesc>("test_op", "Relu");
  ge::GeTensorDesc input_desc(ge::GeShape({1, 2, 3, 4}));
  op_desc->AddInputDesc(input_desc);
  ge::GeTensorDesc output_desc(ge::GeShape({1, ge::UNKNOWN_DIM, 3, 4}));
  op_desc->AddOutputDesc(output_desc);
  EXPECT_TRUE(UnknownShapeUtils::IsUnknownShapeOp(*op_desc));
}

TEST_F(UnknownShapeUtilsCovUT, IncCov_IsUnknownShapeOpCachedAttr) {
  ge::OpDescPtr op_desc = std::make_shared<ge::OpDesc>("test_op", "Relu");
  ge::GeTensorDesc input_desc(ge::GeShape({1, 2, 3, 4}));
  op_desc->AddInputDesc(input_desc);
  ge::GeTensorDesc output_desc(ge::GeShape({1, 2, 3, 4}));
  op_desc->AddOutputDesc(output_desc);
  EXPECT_FALSE(UnknownShapeUtils::IsUnknownShapeOp(*op_desc));
  EXPECT_FALSE(UnknownShapeUtils::IsUnknownShapeOp(*op_desc));
}

TEST_F(UnknownShapeUtilsCovUT, IncCov_IsUnknownShapeOpCachedAttrTrue) {
  ge::OpDescPtr op_desc = std::make_shared<ge::OpDesc>("test_op", "Relu");
  ge::AttrUtils::SetBool(op_desc, "_unknown_shape", true);
  EXPECT_TRUE(UnknownShapeUtils::IsUnknownShapeOp(*op_desc));
}

TEST_F(UnknownShapeUtilsCovUT, IncCov_IsUnknownShapeOpNoInputsOutputs) {
  ge::OpDescPtr op_desc = std::make_shared<ge::OpDesc>("test_op", "Relu");
  EXPECT_FALSE(UnknownShapeUtils::IsUnknownShapeOp(*op_desc));
}

TEST_F(UnknownShapeUtilsCovUT, IncCov_IsUnknownShapeOpUnknownDimNum) {
  ge::OpDescPtr op_desc = std::make_shared<ge::OpDesc>("test_op", "Relu");
  ge::GeShape unknown_shape;
  unknown_shape.SetIsUnknownDimNum();
  ge::GeTensorDesc input_desc(unknown_shape);
  op_desc->AddInputDesc(input_desc);
  EXPECT_TRUE(UnknownShapeUtils::IsUnknownShapeOp(*op_desc));
}
}  // namespace fe
