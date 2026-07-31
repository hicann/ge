/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <vector>

#include <gtest/gtest.h>
#include "graph/ge_tensor.h"
#include "ge_ir.pb.h"
#include "graph_metadef/graph/debug/ge_util.h"
#include "graph/normal_graph/ge_tensor_impl.h"
#include "graph/utils/tensor_adapter.h"
#include "graph/utils/tensor_utils.h"
#include "graph/utils/tensor_utils_ex.h"
#include "graph/utils/attr_utils.h"
#include "graph/debug/ge_attr_define.h"

namespace ge {
class TensorUT : public testing::Test {
 protected:
  void SetUp() {}

  void TearDown() {}
};

TEST_F(TensorUT, SetData1NoShare) {
  GeTensor t1;
  std::vector<uint8_t> vec;
  for (uint8_t i = 0; i < 150; ++i) {
    vec.push_back(i);
  }
  ASSERT_EQ(t1.SetData(vec), GRAPH_SUCCESS);
  ASSERT_EQ(t1.GetData().GetSize(), vec.size());
  ASSERT_EQ(memcmp(t1.GetData().GetData(), vec.data(), vec.size()), 0);
  t1.MutableData().GetData()[10] = 250;
  ASSERT_NE(memcmp(t1.GetData().GetData(), vec.data(), vec.size()), 0);

  std::vector<uint8_t> vec2;
  for (uint8_t i = 0; i < 105; ++i) {
    vec2.push_back(i * 2);
  }
  vec = vec2;
  ASSERT_EQ(t1.SetData(std::move(vec2)), GRAPH_SUCCESS);
  ASSERT_EQ(t1.GetData().GetSize(), vec.size());
  ASSERT_EQ(memcmp(t1.GetData().GetData(), vec.data(), vec.size()), 0);

  vec.clear();
  for (uint8_t i = 0; i < 100; ++i) {
    vec.push_back(100 - i);
  }
  Buffer buffer = Buffer::CopyFrom(vec.data(), vec.size());
  ASSERT_EQ(t1.SetData(buffer), GRAPH_SUCCESS);
  ASSERT_EQ(t1.GetData().GetSize(), vec.size());
  ASSERT_EQ(memcmp(t1.GetData().GetData(), vec.data(), vec.size()), 0);

  vec.clear();
  for (uint8_t i = 0; i < 150; ++i) {
    vec.push_back(i);
  }
  ASSERT_EQ(t1.SetData(vec.data(), vec.size()), GRAPH_SUCCESS);
  ASSERT_EQ(t1.GetData().GetSize(), vec.size());
  ASSERT_EQ(memcmp(t1.GetData().GetData(), vec.data(), vec.size()), 0);

  vec.clear();
  for (uint8_t i = 0; i < 200; ++i) {
    vec.push_back(200 - i);
  }
  TensorData td;
  td.SetData(vec);
  ASSERT_EQ(memcmp(td.GetData(), vec.data(), vec.size()), 0);
  ASSERT_EQ(t1.SetData(td), GRAPH_SUCCESS);
  ASSERT_EQ(t1.GetData().GetSize(), vec.size());
  ASSERT_EQ(memcmp(t1.GetData().GetData(), vec.data(), vec.size()), 0);
}

TEST_F(TensorUT, Construct1_General) {
  GeTensor t1;
  ASSERT_EQ(t1.impl_->desc_.impl_, t1.GetData().impl_->tensor_descriptor_);

  GeTensorDesc td;

  GeIrProtoHelper<ge::proto::TensorDef> helper;
  helper.InitDefault();
  helper.GetProtoMsg()->mutable_data()->resize(200);
  GeTensor t2(helper.GetProtoOwner(), helper.GetProtoMsg());
  ASSERT_NE(t2.impl_->tensor_def_.GetProtoOwner(), nullptr);
  ASSERT_NE(t2.impl_->tensor_def_.GetProtoMsg(), nullptr);
  ASSERT_EQ(t2.impl_->tensor_data_.impl_->tensor_descriptor_, t2.impl_->desc_.impl_);
  ASSERT_EQ(reinterpret_cast<const char *>(t2.impl_->tensor_data_.GetData()),
            t2.impl_->tensor_def_.GetProtoMsg()->data().data());
}
TEST_F(TensorUT, Construct2_CopyDesc) {
  EXPECT_NO_THROW(GeTensorDesc desc; GeTensor t1(desc););
}
TEST_F(TensorUT, Construct3_ExceptionalScenes) {
  GeIrProtoHelper<ge::proto::TensorDef> helper;
  helper.InitDefault();
  GeTensor t1(nullptr, helper.GetProtoMsg());
  GeTensor t2(helper.GetProtoOwner(), nullptr);
  GeTensor t3(nullptr, nullptr);

  ASSERT_EQ(t1.impl_->tensor_def_.GetProtoMsg(), helper.GetProtoMsg());
  ASSERT_EQ(t1.impl_->tensor_def_.GetProtoOwner(), nullptr);
  ASSERT_EQ(t1.impl_->tensor_data_.impl_->tensor_descriptor_, t1.impl_->desc_.impl_);

  ASSERT_EQ(t2.impl_->tensor_def_.GetProtoMsg(), nullptr);
  ASSERT_EQ(t2.impl_->tensor_def_.GetProtoOwner(), helper.GetProtoOwner());
  ASSERT_EQ(t2.impl_->tensor_data_.impl_->tensor_descriptor_, t2.impl_->desc_.impl_);

  ASSERT_EQ(t3.impl_->tensor_def_.GetProtoMsg(), nullptr);
  ASSERT_EQ(t3.impl_->tensor_def_.GetProtoOwner(), nullptr);
  ASSERT_EQ(t3.impl_->tensor_data_.impl_->tensor_descriptor_, t3.impl_->desc_.impl_);
}
TEST_F(TensorUT, CopyConstruct1_NullTensorDef) {
  GeTensor t1;

  std::vector<uint8_t> vec;
  for (uint8_t i = 0; i < 100; ++i) {
    vec.push_back(i * 2);
  }
  t1.SetData(vec);
  GeTensor t2(t1);

  // The copy construct share tensor_data_, do not share tensor_desc
  ASSERT_EQ(t1.impl_->tensor_def_.GetProtoOwner(), nullptr);
  ASSERT_EQ(t1.impl_->tensor_def_.GetProtoMsg(), nullptr);
  ASSERT_EQ(t1.impl_->tensor_data_.impl_->tensor_descriptor_, t1.impl_->desc_.impl_);
  ASSERT_EQ(t2.impl_->tensor_data_.impl_->tensor_descriptor_, t2.impl_->desc_.impl_);
  ASSERT_EQ(t1.impl_->tensor_data_.GetData(), t2.impl_->tensor_data_.GetData());

  t1.MutableTensorDesc().SetFormat(FORMAT_NCHW);
  t2.MutableTensorDesc().SetFormat(FORMAT_NHWC);
  ASSERT_EQ(t1.GetTensorDesc().GetFormat(), FORMAT_NCHW);
  ASSERT_EQ(t2.GetTensorDesc().GetFormat(), FORMAT_NHWC);

  ASSERT_EQ(memcmp(t1.GetData().GetData(), vec.data(), vec.size()), 0);
  ASSERT_EQ(t1.GetData().GetData(), t2.GetData().GetData());
}

TEST_F(TensorUT, CopyConstruct2_WithTensorDef) {
  GeIrProtoHelper<ge::proto::TensorDef> helper;
  helper.InitDefault();
  helper.GetProtoMsg()->mutable_data()->resize(100);
  GeTensor t1(helper.GetProtoOwner(), helper.GetProtoMsg());

  std::vector<uint8_t> vec;
  for (uint8_t i = 0; i < 100; ++i) {
    vec.push_back(i * 2);
  }
  t1.SetData(vec);
  GeTensor t2(t1);

  // Copy construct should share tensordata only
  ASSERT_NE(t1.impl_->tensor_def_.GetProtoOwner(), nullptr);
  ASSERT_NE(t1.impl_->tensor_def_.GetProtoMsg(), nullptr);
  ASSERT_EQ(t1.impl_->tensor_data_.impl_->tensor_descriptor_, t1.impl_->desc_.impl_);
  ASSERT_EQ(t2.impl_->tensor_data_.impl_->tensor_descriptor_, t2.impl_->desc_.impl_);
  ASSERT_EQ(t1.impl_->tensor_data_.GetData(), t2.impl_->tensor_data_.GetData());

  t1.MutableTensorDesc().SetFormat(FORMAT_NCHW);
  ASSERT_EQ(t1.GetTensorDesc().GetFormat(), FORMAT_NCHW);
  t2.MutableTensorDesc().SetFormat(FORMAT_NHWC);
  ASSERT_EQ(t1.GetTensorDesc().GetFormat(), FORMAT_NCHW);
  ASSERT_EQ(t2.GetTensorDesc().GetFormat(), FORMAT_NHWC);

  ASSERT_EQ(memcmp(t1.GetData().GetData(), vec.data(), vec.size()), 0);
  ASSERT_EQ(t1.GetData().GetData(), t2.GetData().GetData());
}

TEST_F(TensorUT, SetData_SharedWithTensorDef) {
  GeIrProtoHelper<ge::proto::TensorDef> helper;
  helper.InitDefault();
  helper.GetProtoMsg()->mutable_data()->resize(100);
  GeTensor t1(helper.GetProtoOwner(), helper.GetProtoMsg());

  std::vector<uint8_t> vec;
  for (uint8_t i = 0; i < 100; ++i) {
    vec.push_back(i * 2);
  }
  t1.SetData(vec);
  GeTensor t2(t1);

  std::vector<uint8_t> vec2;
  for (uint8_t i = 0; i < 100; ++i) {
    vec2.push_back(i);
  }
  t2.SetData(vec2);
  ASSERT_EQ(memcmp(t2.GetData().GetData(), vec2.data(), vec2.size()), 0);
  // todo 这里存在bug，但是从目前来看，并没有被触发，因此暂时不修复了，重构后一起修复。
  //  触发bug的场景为：如果tensor1是通过tensor_def_持有TensorData，然后通过拷贝构造、拷贝赋值的方式，从tensor1构造了tensor2。
  //  那么通过tensor2.SetData后，会导致tensor1的GetData接口失效（获取到野指针）
  //  触发的表现就是，如下两条ASSERT_EQ并不成立
  // ASSERT_EQ(t1.GetData().GetData(), t2.GetData().GetData());
  // ASSERT_EQ(memcmp(t1.GetData().GetData(), vec2.data(), vec2.size()), 0);
}

TEST_F(TensorUT, SetData_SharedWithoutTensorDef) {
  GeTensor t1;

  std::vector<uint8_t> vec;
  for (uint8_t i = 0; i < 100; ++i) {
    vec.push_back(i * 2);
  }
  t1.SetData(vec);
  GeTensor t2(t1);

  std::vector<uint8_t> vec3;
  for (uint8_t i = 0; i < 100; ++i) {
    vec3.push_back(i);
  }
  t2.SetData(vec3);
  ASSERT_EQ(t2.GetData().size(), vec3.size());
  ASSERT_EQ(memcmp(t2.GetData().GetData(), vec3.data(), vec3.size()), 0);
  ASSERT_EQ(t1.GetData().size(), vec3.size());
  ASSERT_EQ(memcmp(t1.GetData().GetData(), vec3.data(), vec3.size()), 0);
  ASSERT_EQ(t1.GetData().GetData(), t2.GetData().GetData());

  std::vector<uint8_t> vec2;
  for (uint8_t i = 0; i < 105; ++i) {
    vec2.push_back(i);
  }
  t2.SetData(vec2);
  ASSERT_EQ(t2.GetData().size(), vec2.size());
  ASSERT_EQ(memcmp(t2.GetData().GetData(), vec2.data(), vec2.size()), 0);
  // after modify the data of t2 using a different size buffer, the t1 will not be modified
  ASSERT_EQ(t1.GetData().size(), vec3.size());
  ASSERT_EQ(memcmp(t1.GetData().GetData(), vec3.data(), vec3.size()), 0);
  ASSERT_NE(t1.GetData().GetData(), t2.GetData().GetData());
}

TEST_F(TensorUT, SetDataDelete_success) {
  auto deleter = [](uint8_t *ptr) {
    delete[] ptr;
    ptr = nullptr;
  };
  uint8_t *data_ptr = new uint8_t[10];
  GeTensor ge_tensor;
  ge_tensor.SetData(data_ptr, 10, deleter);
  auto length = ge_tensor.GetData().GetSize();
  ASSERT_EQ(length, 10);
}

TEST_F(TensorUT, TensorSetDataDelete_success) {
  auto deleter = [](uint8_t *ptr) {
    delete[] ptr;
    ptr = nullptr;
  };
  uint8_t *data_ptr = new uint8_t[10];
  Tensor tensor;
  EXPECT_EQ(tensor.SetData(data_ptr, 10, deleter), GRAPH_SUCCESS);
  EXPECT_EQ(tensor.GetSize(), 10);
}

TEST_F(TensorUT, TransTensorDescWithoutOriginShape2GeTensorDesc) {
  TensorDesc desc(Shape({1, 2, 3, 4}), FORMAT_NCHW);
  GeTensorDesc ge_desc = TensorAdapter::TensorDesc2GeTensorDesc(desc);
  ASSERT_EQ(desc.GetFormat(), ge_desc.GetFormat());
  ASSERT_EQ(desc.GetShape().GetDims().size(), ge_desc.GetShape().GetDims().size());
  for (size_t i = 0; i < desc.GetShape().GetDims().size(); i++) {
    ASSERT_EQ(desc.GetShape().GetDim(i), ge_desc.GetShape().GetDim(i));
  }
  bool origin_format_is_set = false;
  EXPECT_FALSE(AttrUtils::GetBool(ge_desc, ATTR_NAME_ORIGIN_FORMAT_IS_SET, origin_format_is_set));
}

TEST_F(TensorUT, TransTensorDescWithOriginShape2GeTensorDesc) {
  TensorDesc desc(Shape({1, 2, 3, 4}), FORMAT_NCHW);
  desc.SetOriginFormat(FORMAT_NHWC);
  desc.SetOriginShape(Shape({1, 3, 4, 2}));
  GeTensorDesc ge_desc = TensorAdapter::TensorDesc2GeTensorDesc(desc);

  ASSERT_EQ(desc.GetFormat(), ge_desc.GetFormat());
  ASSERT_EQ(desc.GetShape().GetDims().size(), ge_desc.GetShape().GetDims().size());
  for (size_t i = 0; i < desc.GetShape().GetDims().size(); i++) {
    ASSERT_EQ(desc.GetShape().GetDim(i), ge_desc.GetShape().GetDim(i));
  }

  ASSERT_EQ(desc.GetOriginFormat(), ge_desc.GetOriginFormat());
  ASSERT_EQ(desc.GetOriginShape().GetDims().size(), ge_desc.GetOriginShape().GetDims().size());
  for (size_t i = 0; i < desc.GetOriginShape().GetDims().size(); i++) {
    ASSERT_EQ(desc.GetOriginShape().GetDim(i), ge_desc.GetOriginShape().GetDim(i));
  }
  bool origin_format_is_set = false;
  EXPECT_TRUE(AttrUtils::GetBool(ge_desc, ATTR_NAME_ORIGIN_FORMAT_IS_SET, origin_format_is_set));
  EXPECT_TRUE(origin_format_is_set);
}

TEST_F(TensorUT, NormalizeGeTensorWithOriginShape) {
  TensorDesc desc(Shape({1, 2, 3, 4}), FORMAT_NCHW);
  desc.SetOriginFormat(FORMAT_NHWC);
  desc.SetOriginShape(Shape({1, 3, 4, 2}));
  Tensor tensor(desc);
  auto ge_tensor = TensorAdapter::AsGeTensor(tensor);
  auto &ge_desc = ge_tensor.MutableTensorDesc();

  bool origin_format_is_set = false;
  EXPECT_TRUE(AttrUtils::GetBool(ge_desc, ATTR_NAME_ORIGIN_FORMAT_IS_SET, origin_format_is_set));
  EXPECT_TRUE(origin_format_is_set);

  auto normalized_ge_tensor = TensorAdapter::NormalizeGeTensor(ge_tensor);
  auto &normalized_ge_desc = normalized_ge_tensor.MutableTensorDesc();

  EXPECT_TRUE(AttrUtils::GetBool(normalized_ge_desc, ATTR_NAME_ORIGIN_FORMAT_IS_SET, origin_format_is_set));
  EXPECT_FALSE(origin_format_is_set);

  auto storage_format = static_cast<int64_t>(FORMAT_MAX);
  EXPECT_TRUE(AttrUtils::GetInt(normalized_ge_desc, ATTR_NAME_STORAGE_FORMAT, storage_format));
  EXPECT_EQ(storage_format, static_cast<int64_t>(ge_desc.GetFormat()));

  std::vector<int64_t> storage_dims;
  EXPECT_TRUE(AttrUtils::GetListInt(normalized_ge_desc, ATTR_NAME_STORAGE_SHAPE, storage_dims));
  EXPECT_EQ(storage_dims.size(), ge_desc.GetShape().GetDims().size());
  for (size_t i = 0; i < storage_dims.size(); i++) {
    ASSERT_EQ(ge_desc.GetShape().GetDim(i), storage_dims[i]);
  }

  EXPECT_EQ(ge_desc.GetOriginFormat(), normalized_ge_desc.GetFormat());
  ASSERT_EQ(ge_desc.GetOriginShape().GetDims().size(), normalized_ge_desc.GetShape().GetDims().size());
  for (size_t i = 0; i < ge_desc.GetOriginShape().GetDims().size(); i++) {
    ASSERT_EQ(ge_desc.GetOriginShape().GetDim(i), normalized_ge_desc.GetShape().GetDim(i));
  }
}

TEST_F(TensorUT, GeShapeSetDimNum) {
  ge::GeShape shape;
  EXPECT_EQ(shape.GetDimNum(), 0);
  shape.SetDimNum(2);  // Normal dim nums
  EXPECT_EQ(shape.GetDimNum(), 2);
  EXPECT_EQ(shape.GetDim(0), ge::UNKNOWN_DIM);
  EXPECT_EQ(shape.GetDim(1), ge::UNKNOWN_DIM);
  shape.SetDimNum(0);  // Scalar dim nums
  EXPECT_EQ(shape.GetDimNum(), 0);
  shape.SetDimNum(20);  // Big dim nums
  EXPECT_EQ(shape.GetDimNum(), 20);
  for (int i = 0; i < 20; i++) {
    EXPECT_EQ(shape.GetDim(i), ge::UNKNOWN_DIM);
  }
}

TEST_F(TensorUT, GeShapeIsUnknownDimNum) {
  ge::GeShape shape;
  EXPECT_FALSE(shape.IsUnknownDimNum());
  shape.SetDimNum(2);
  EXPECT_FALSE(shape.IsUnknownDimNum());
  shape.SetIsUnknownDimNum();
  EXPECT_TRUE(shape.IsUnknownDimNum());
  shape.SetDimNum(2);
  EXPECT_FALSE(shape.IsUnknownDimNum());
}

TEST_F(TensorUT, GeShapeAppendDim) {
  ge::GeShape shape;
  EXPECT_EQ(shape.GetDimNum(), 0);
  shape.AppendDim(1);
  EXPECT_EQ(shape.GetDimNum(), 1);
  EXPECT_EQ(shape.GetDim(0), 1);
  shape.AppendDim(2);
  EXPECT_EQ(shape.GetDimNum(), 2);
  EXPECT_EQ(shape.GetDim(0), 1);
  EXPECT_EQ(shape.GetDim(1), 2);
  shape.SetIsUnknownDimNum();
  EXPECT_TRUE(shape.IsUnknownDimNum());
  shape.AppendDim(1);
  EXPECT_FALSE(shape.IsUnknownDimNum());
}

TEST_F(TensorUT, GeTensorDescGetShape) {
  ge::GeTensorDesc desc(ge::GeShape(std::vector<int64_t>({1, 2})));
  auto &shape = desc.GetShape();
  EXPECT_EQ(shape.GetDim(0), 1);
  EXPECT_EQ(shape.GetDim(1), 2);
  const_cast<ge::GeShape *>(&shape)->SetDim(0, 10);
  const_cast<ge::GeShape *>(&shape)->SetDim(1, 20);
  auto &shape2 = desc.GetShape();
  EXPECT_EQ(shape2.GetDim(0), 10);
  EXPECT_EQ(shape2.GetDim(1), 20);
}

TEST_F(TensorUT, GeTensorSerializeUtils_GeShape) {
  GeShape shape({1, 2, 3, 4});
  proto::ShapeDef shape_proto;
  GeTensorSerializeUtils::GeShapeAsProto(shape, &shape_proto);
  GeShape shape_from_proto;
  GeTensorSerializeUtils::AssembleGeShapeFromProto(&shape_proto, shape_from_proto);
  EXPECT_EQ(shape, shape_from_proto);
}

TEST_F(TensorUT, GeTensorSerializeUtils_GeTensorDesc) {
  GeShape shape({1, 2, 3, 4});
  GeTensorDesc desc(shape, FORMAT_NC1HWC0, DT_FLOAT16);
  desc.SetOriginDataType(DT_INT32);
  desc.SetOriginFormat(FORMAT_NHWC1C0);
  desc.SetOriginShape(GeShape({4, 3, 2, 1}));
  proto::TensorDescriptor desc_proto;
  GeTensorSerializeUtils::GeTensorDescAsProto(desc, &desc_proto);
  GeTensorDesc desc_from_proto;
  GeTensorSerializeUtils::AssembleGeTensorDescFromProto(&desc_proto, desc_from_proto);
  bool res = false;
  EXPECT_TRUE(AttrUtils::GetBool(desc_from_proto, "origin_shape_initialized", res));
  EXPECT_TRUE(res);
  EXPECT_EQ(desc, desc_from_proto);
}

TEST_F(TensorUT, GeTensorSerializeUtils_Dtype) {
  proto::TensorDescriptor desc_proto;
  ge::proto::AttrDef custom_dtype;
  custom_dtype.set_i(13);
  (void)desc_proto.mutable_attr()->insert({"__tensor_desc_data_type__", custom_dtype});
  ge::DataType dtype;
  GeTensorSerializeUtils::GetDtypeFromDescProto(&desc_proto, dtype);
  EXPECT_EQ(dtype, ge::DT_DUAL);
}

TEST_F(TensorUT, GeTensorSerializeUtils_GeTensor) {
  GeShape shape({1, 2, 3, 4});
  GeTensorDesc desc(shape, FORMAT_NC1HWC0, DT_FLOAT16);
  desc.SetOriginDataType(DT_INT32);
  desc.SetOriginFormat(FORMAT_NHWC1C0);
  desc.SetOriginShape(GeShape({4, 3, 2, 1}));
  GeTensor tensor(desc);
  proto::TensorDef tensor_proto;
  GeTensorSerializeUtils::GeTensorAsProto(tensor, &tensor_proto);
  GeTensor tensor_from_proto;
  GeTensorSerializeUtils::AssembleGeTensorFromProto(&tensor_proto, tensor_from_proto);
  EXPECT_EQ(tensor.GetTensorDesc(), desc);
  EXPECT_EQ(tensor.GetTensorDesc(), tensor_from_proto.GetTensorDesc());
}

TEST_F(TensorUT, GeShape_ModifyDimNum) {
  GeShape shape({1, 2, 3, 4});
  EXPECT_EQ(shape.GetShapeSize(), 24);
  EXPECT_EQ(shape.GetDimNum(), 4);
  shape.SetDimNum(2);
  EXPECT_EQ(shape.GetDimNum(), 2);
  EXPECT_FALSE(shape.IsUnknownDimNum());
  shape.SetIsUnknownDimNum();
  EXPECT_TRUE(shape.IsUnknownDimNum());
  EXPECT_EQ(shape.GetShapeSize(), -1);
  shape.SetDimNum(2);
  EXPECT_EQ(shape.GetDimNum(), 2);
  EXPECT_FALSE(shape.IsUnknownDimNum());
  shape.SetDim(0, 2);
  shape.SetDim(1, 2);
  EXPECT_EQ(shape.GetShapeSize(), 4);
  shape.SetDim(0, INT64_MAX);
  shape.SetDim(1, 2);
  EXPECT_EQ(shape.GetShapeSize(), -1);
}

TEST_F(TensorUT, GeShape_Unknown) {
  GeShape shape({-2});
  EXPECT_TRUE(shape.IsUnknownShape());
  EXPECT_TRUE(shape.IsUnknownDimNum());
  EXPECT_FALSE(shape.IsScalar());
  EXPECT_EQ(shape.GetDimNum(), 0U);
  EXPECT_EQ(shape.GetDims().size(), 1U);
}

TEST_F(TensorUT, Shape_Unknown) {
  Shape shape({-2});
  EXPECT_EQ(shape.GetDimNum(), 0U);
  EXPECT_EQ(shape.GetDims().size(), 1U);
}

TEST_F(TensorUT, GeTensorDesc_Update) {
  GeShape shape({1, 2, 3, 4});
  GeTensorDesc desc(shape, FORMAT_NC1HWC0, DT_FLOAT16);
  EXPECT_EQ(desc.GetShape(), shape);
  EXPECT_EQ(desc.GetFormat(), FORMAT_NC1HWC0);
  EXPECT_EQ(desc.GetDataType(), DT_FLOAT16);
  GeShape shape2({4, 3, 2, 1});
  desc.Update(shape2, FORMAT_NHWC, DT_INT32);
  EXPECT_EQ(desc.GetShape(), shape2);
  EXPECT_EQ(desc.GetFormat(), FORMAT_NHWC);
  EXPECT_EQ(desc.GetDataType(), DT_INT32);
}

TEST_F(TensorUT, AttrUtils_SetGeTensorDesc) {
  GeShape shape({1, 2, 3, 4});
  GeTensorDesc desc(shape, FORMAT_NC1HWC0, DT_FLOAT16);
  GeTensorDesc obj;
  ge::AttrUtils::SetTensorDesc(obj, "attr_tensor", desc);
  GeTensorDesc desc_from_attr;
  ge::AttrUtils::GetTensorDesc(obj, "attr_tensor", desc_from_attr);
  EXPECT_EQ(desc, desc_from_attr);
}

TEST_F(TensorUT, AttrUtils_SetListGeTensorDesc) {
  GeShape shape({1, 2, 3, 4});
  std::vector<GeTensorDesc> descs;
  descs.emplace_back(GeTensorDesc(GeShape({1, 2, 3, 4}), FORMAT_NC1HWC0, DT_FLOAT16));
  descs.emplace_back(GeTensorDesc(GeShape({4, 3, 2, 1}), FORMAT_NCHW, DT_INT32));
  GeTensorDesc obj;
  ge::AttrUtils::SetListTensorDesc(obj, "attr_tensors", descs);
  std::vector<GeTensorDesc> descs_from_attr;
  ge::AttrUtils::GetListTensorDesc(obj, "attr_tensors", descs_from_attr);
  EXPECT_EQ(descs.size(), descs_from_attr.size());
  for (size_t i = 0; i < descs.size(); i++) {
    EXPECT_EQ(descs[i], descs_from_attr[i]);
  }
}

class AscendStringUT : public testing::Test {
 protected:
  void SetUp() {}

  void TearDown() {}
};

TEST_F(AscendStringUT, Hash) {
  ge::AscendString ascend_string("ABC");
  EXPECT_EQ(std::hash<ge::AscendString>()(ascend_string), ascend_string.Hash());
  EXPECT_EQ(std::hash<std::string>()(ascend_string.GetString()), ascend_string.Hash());
  EXPECT_EQ(std::hash<std::string>()("ABC"), ascend_string.Hash());

  ge::AscendString empty_ascend_string;
  EXPECT_EQ(std::hash<ge::AscendString>()(empty_ascend_string), empty_ascend_string.Hash());
  EXPECT_EQ(std::hash<std::string>()(""), empty_ascend_string.Hash());
}

TEST_F(AscendStringUT, EmptyValueCompare) {
  ge::AscendString ascend_string;
  EXPECT_NE(ascend_string.GetString(), "");
  EXPECT_EQ(ascend_string.GetString(), std::string(""));
  EXPECT_TRUE(std::string(ascend_string.GetString()).empty());
}

TEST_F(TensorUT, TensorUtils_GetSteExtMeta) {
  GeTensorDesc desc;

#define TEST_EXT_META_INNER(NAME, TYPE, V, V1) \
  do {                                         \
    TYPE v = V;                                \
    TYPE v1 = V1;                              \
    TensorUtils::Set##NAME(desc, v);           \
    TensorUtils::Get##NAME(desc, v1);          \
    EXPECT_EQ(v, v1);                          \
  } while (false)

#define TEST_EXT_META_INT64(NAME) TEST_EXT_META_INNER(NAME, int64_t, 0, -1);
#define TEST_EXT_META_BOOL(NAME) TEST_EXT_META_INNER(NAME, bool, true, false);
#define TEST_EXT_META_UINT32(NAME) TEST_EXT_META_INNER(NAME, uint32_t, 0, 1);

  TEST_EXT_META_INT64(Size);
  TEST_EXT_META_INT64(DataOffset);

  TEST_EXT_META_UINT32(RealDimCnt);
  TEST_EXT_META_UINT32(ReuseInputIndex);

  TEST_EXT_META_BOOL(InputTensor);
  TEST_EXT_META_BOOL(OutputTensor);
  TEST_EXT_META_BOOL(ReuseInput);

  desc.SetName("foo");
  EXPECT_EQ(desc.GetName(), "foo");

  TensorUtils::SetWeightSize(desc, 2021);
  EXPECT_EQ(TensorUtils::GetWeightSize(desc), 2021);
}

TEST_F(TensorUT, Tensor_Construct3) {
  std::vector<int64_t> shape{4};
  uint8_t *data = new uint8_t[4]{1, 2, 3, 4};
  size_t size = 4;
  TensorDesc tensor_desc(Shape(shape), FORMAT_ND, DT_UINT8);
  Tensor tensor(tensor_desc, data, size);
  EXPECT_EQ(tensor.GetSize(), 4);
  delete[] data;
}

TEST_F(TensorUT, Tensor_Construct4) {
  std::vector<uint8_t> value{1, 2, 3};
  std::vector<int64_t> shape{3};
  TensorDesc tensor_desc(Shape(shape), FORMAT_ND, DT_UINT8);
  Tensor tensor(std::move(tensor_desc), std::move(value));
  EXPECT_EQ(tensor.GetSize(), 3);
}

TEST_F(TensorUT, Tensor_SetData) {
  Tensor t1;
  std::vector<uint8_t> vec;
  for (uint8_t i = 0; i < 10; ++i) {
    vec.push_back(i);
  }
  EXPECT_EQ(t1.SetData(vec), GRAPH_SUCCESS);

  Tensor t2;
  std::string str1 = "abc";
  EXPECT_EQ(t2.SetData(str1), GRAPH_SUCCESS);

  Tensor t3;
  std::vector<std::string> vec_str;
  EXPECT_EQ(t3.SetData(vec_str), GRAPH_FAILED);
  for (uint8_t i = 0; i < 10; ++i) {
    vec_str.push_back(std::to_string(i));
  }
  EXPECT_EQ(t3.SetData(vec_str), GRAPH_SUCCESS);

  Tensor t4;
  const char *str2 = "def";
  EXPECT_EQ(t4.SetData(str2), GRAPH_SUCCESS);

  Tensor t5;
  const char *str3[3] = {"123", "456", "789"};
  std::vector<AscendString> vec_asc_str;
  for (uint8_t i = 0; i < 3; ++i) {
    vec_asc_str.push_back(AscendString(str3[i]));
  }
  EXPECT_EQ(t5.SetData(vec_asc_str), GRAPH_SUCCESS);
}

TEST_F(TensorUT, Shape_SetDim) {
  size_t idx = 1;
  int64_t value = 2;

  Shape shape1;
  EXPECT_EQ(shape1.SetDim(idx, value), GRAPH_FAILED);

  std::vector<int64_t> dims;
  for (int64_t i = 0; i < 3; i++) {
    dims.push_back(i);
  }

  Shape shape2(dims);
  EXPECT_EQ(shape2.SetDim(idx, value), GRAPH_SUCCESS);
}

TEST_F(TensorUT, TensorDesc_Construct1) {
  std::vector<int64_t> shape{3};
  TensorDesc tensor_desc1(Shape(shape), FORMAT_ND, DT_UINT8);
  TensorDesc tensor_desc2(std::move(tensor_desc1));

  TensorDesc tensor_desc3(Shape(shape), FORMAT_ND, DT_UINT8);
  TensorDesc tensor_desc4 = std::move(tensor_desc3);

  tensor_desc4.Update(Shape(shape), FORMAT_ND, DT_UINT16);
  EXPECT_EQ(tensor_desc4.GetDataType(), DT_UINT16);

  TensorDesc tensor_desc5;
  EXPECT_EQ(tensor_desc5.GetShape().GetShapeSize(), 0);
}

TEST_F(TensorUT, TensorDesc_GetSetShape) {
  std::vector<std::pair<int64_t, int64_t>> range;
  TensorDesc tensor_desc1;
  tensor_desc1.GetShape();
  tensor_desc1.GetOriginShape();

  EXPECT_EQ(tensor_desc1.GetShapeRange(range), GRAPH_SUCCESS);
  EXPECT_EQ(tensor_desc1.SetShapeRange(range), GRAPH_SUCCESS);

  EXPECT_EQ(tensor_desc1.SetUnknownDimNumShape(), GRAPH_SUCCESS);

  std::vector<int64_t> shape{3};
  TensorDesc tensor_desc2(Shape(shape), FORMAT_ND, DT_UINT8);
  EXPECT_EQ(tensor_desc2.SetUnknownDimNumShape(), GRAPH_SUCCESS);
}

TEST_F(TensorUT, TensorDesc_SetDataType) {
  EXPECT_NO_THROW(std::vector<int64_t> shape{3}; TensorDesc tensor_desc1(Shape(shape), FORMAT_ND, DT_UINT8);
                  tensor_desc1.SetDataType(DT_UINT16););
}

TEST_F(TensorUT, TensorDesc_GetSetName) {
  std::vector<int64_t> shape{3};
  TensorDesc tensor_desc1(Shape(shape), FORMAT_ND, DT_UINT8);
  tensor_desc1.SetName("abc");

  AscendString name;
  tensor_desc1.GetName(name);
  EXPECT_EQ(name, AscendString("abc"));
}

TEST_F(TensorUT, TensorDesc_get_set_expand_dims_rule) {
  TensorDesc td;
  // init status
  AscendString expand_dims_rule;
  td.GetExpandDimsRule(expand_dims_rule);
  EXPECT_EQ(expand_dims_rule.GetLength(), 0);

  // test set and get
  expand_dims_rule = AscendString("1100");
  td.SetExpandDimsRule(expand_dims_rule);
  td.GetExpandDimsRule(expand_dims_rule);
  EXPECT_STREQ(expand_dims_rule.GetString(), "1100");
}

TEST_F(TensorUT, Tensor_SetTensorDesc_GetData) {
  std::vector<int64_t> shape{3};
  TensorDesc tensor_desc1(Shape(shape), FORMAT_ND, DT_UINT8);

  Tensor t1;
  auto ret = t1.SetTensorDesc(tensor_desc1);
  EXPECT_EQ(ret, GRAPH_SUCCESS);

  uint8_t *data1 = NULL;
  data1 = t1.GetData();
  EXPECT_NE(data1, nullptr);

  const uint8_t *data2 = NULL;
  data2 = t1.GetData();
  EXPECT_NE(data2, nullptr);
}

TEST_F(TensorUT, Tensor_AsGeTensorImpl) {
  std::vector<int64_t> shape{3};
  TensorDesc tensor_desc1(Shape(shape), FORMAT_ND, DT_UINT8);

  Tensor t1;
  auto ret = t1.SetTensorDesc(tensor_desc1);
  EXPECT_EQ(ret, GRAPH_SUCCESS);

  const GeTensor *gt_impl = TensorAdapter::AsBareGeTensorPtr(t1);
  EXPECT_NE(gt_impl, nullptr);

  t1.impl = nullptr;
  const GeTensor *gt_impl_2 = TensorAdapter::AsBareGeTensorPtr(t1);
  EXPECT_EQ(gt_impl_2, nullptr);
}

TEST_F(TensorUT, unique_ptr_Tensor_ResetData) {
  Tensor t1;
  std::unique_ptr<uint8_t[], Tensor::DeleteFunc> pt;
  EXPECT_NO_THROW(pt = t1.ResetData());
}

TEST_F(TensorUT, Tensor_IsValid_Clone) {
  Tensor t1;
  Tensor t2;

  std::vector<int64_t> shape{3};
  TensorDesc tensor_desc1(Shape(shape), FORMAT_ND, DT_UINT8);
  t1.SetTensorDesc(tensor_desc1);

  EXPECT_EQ(t1.IsValid(), GRAPH_FAILED);

  t2 = t1.Clone();
}

TEST_F(TensorUT, TensorAdapter_GetGeTensorFromTensor) {
  Tensor t1;
  GeTensor gt = TensorAdapter::AsGeTensorShared(t1);
  ConstGeTensorPtr cgtptr = TensorAdapter::AsGeTensorPtr(t1);
  EXPECT_NE(cgtptr, nullptr);
}

TEST_F(TensorUT, TensorAdapter_AsTensor) {
  GeTensor gt1;
  std::vector<uint8_t> vec;
  for (uint8_t i = 0; i < 100; ++i) {
    vec.push_back(i * 2);
  }
  gt1.SetData(vec);

  Tensor t1;
  t1 = TensorAdapter::AsTensor(gt1);
  EXPECT_EQ(t1.GetSize(), gt1.GetData().GetSize());
  const GeTensor gt2;
  const Tensor t2 = TensorAdapter::AsTensor(gt2);
  EXPECT_EQ(t2.GetSize(), gt2.GetData().GetSize());
}

TEST_F(TensorUT, TensorDesc2GeTensorDesc_expand_dims_rule) {
  TensorDesc td;
  // test set and get
  td.SetExpandDimsRule(AscendString("0011"));

  auto ge_tensor_desc = TensorAdapter::TensorDesc2GeTensorDesc(td);
  EXPECT_STREQ(ge_tensor_desc.GetExpandDimsRule().c_str(), "0011");
}

TEST_F(TensorUT, GeTensorDesc2TensorDesc_expand_dims_rule) {
  GeTensorDesc ge_tensor_desc;
  // test set and get
  ge_tensor_desc.SetExpandDimsRule("0011");

  auto tensor_desc = TensorAdapter::GeTensorDesc2TensorDesc(ge_tensor_desc);
  AscendString expand_dims_rule;
  tensor_desc.GetExpandDimsRule(expand_dims_rule);
  EXPECT_STREQ(expand_dims_rule.GetString(), "0011");
}

TEST_F(TensorUT, GeTensorDesc2TensorDesc_reuse_input) {
  GeTensorDesc ge_tensor_desc;
  TensorUtils::SetReuseInput(ge_tensor_desc, true);
  TensorUtils::SetReuseInputIndex(ge_tensor_desc, 1U);

  auto tensor_desc = TensorAdapter::GeTensorDesc2TensorDesc(ge_tensor_desc);
  auto converted_ge_tensor_desc = TensorAdapter::TensorDesc2GeTensorDesc(tensor_desc);

  bool reuse_input = false;
  uint32_t reuse_input_index = 0U;
  ASSERT_EQ(TensorUtils::GetReuseInput(converted_ge_tensor_desc, reuse_input), GRAPH_SUCCESS);
  ASSERT_EQ(TensorUtils::GetReuseInputIndex(converted_ge_tensor_desc, reuse_input_index), GRAPH_SUCCESS);
  EXPECT_TRUE(reuse_input);
  EXPECT_EQ(reuse_input_index, 1U);
}

TEST_F(TensorUT, GetPaddingSize_ReturnsValidValue) {
  const int64_t padding_size = TensorUtilsEx::GetPaddingSize();
  EXPECT_GE(padding_size, 0);
  EXPECT_LE(padding_size, 32);
}

TEST_F(TensorUT, GetPaddingSize_ReturnsCachedValue) {
  const int64_t first = TensorUtilsEx::GetPaddingSize();
  const int64_t second = TensorUtilsEx::GetPaddingSize();
  EXPECT_EQ(first, second);
}

TEST_F(TensorUT, IncCov_ShapeOverflowChecks) {
  Shape s1({INT64_MAX, 2});
  EXPECT_EQ(s1.GetShapeSize(), 0);
  Shape s2({2, INT64_MIN});
  EXPECT_EQ(s2.GetShapeSize(), 0);
  Shape s3({-3, INT64_MAX});
  EXPECT_EQ(s3.GetShapeSize(), 0);
  Shape s4({-3, -4});
  EXPECT_EQ(s4.GetShapeSize(), 12);
  Shape s5({-3, INT64_MIN + 1});
  EXPECT_EQ(s5.GetShapeSize(), 0);
}

TEST_F(TensorUT, IncCov_ShapeEdgeCases) {
  Shape shape({1, 2, 3});
  EXPECT_EQ(shape.GetDim(10), 0);
  EXPECT_EQ(shape.SetDim(10, 1), GRAPH_FAILED);
  Shape null_shape;
  null_shape.impl_ = nullptr;
  EXPECT_EQ(null_shape.GetDimNum(), 0U);
  EXPECT_EQ(null_shape.GetDim(0), 0);
  EXPECT_EQ(null_shape.SetDim(0, 1), GRAPH_FAILED);
  EXPECT_EQ(null_shape.GetDims(), std::vector<int64_t>());
  EXPECT_EQ(null_shape.GetShapeSize(), 0);
}

TEST_F(TensorUT, IncCov_TensorDescNullImpl) {
  TensorDesc desc;
  desc.impl = nullptr;
  EXPECT_EQ(desc.GetShape().GetDimNum(), 0U);
  EXPECT_EQ(desc.SetUnknownDimNumShape(), GRAPH_FAILED);
  EXPECT_EQ(desc.SetShapeRange({}), GRAPH_FAILED);
  std::vector<std::pair<int64_t, int64_t>> range;
  EXPECT_EQ(desc.GetShapeRange(range), GRAPH_FAILED);
  EXPECT_EQ(desc.GetOriginShape().GetDimNum(), 0U);
  EXPECT_EQ(desc.GetFormat(), FORMAT_RESERVED);
  EXPECT_EQ(desc.GetOriginFormat(), FORMAT_RESERVED);
  EXPECT_EQ(desc.GetDataType(), DT_UNDEFINED);
  EXPECT_EQ(desc.GetSize(), 0);
  EXPECT_EQ(desc.GetRealDimCnt(), 0);
  EXPECT_EQ(desc.GetName(), "");
  AscendString name;
  EXPECT_EQ(desc.GetName(name), GRAPH_FAILED);
  EXPECT_EQ(desc.GetPlacement(), kPlacementHost);
  uint8_t *const_data = nullptr;
  size_t const_data_len = 0;
  EXPECT_FALSE(desc.GetConstData(&const_data, const_data_len));
  AscendString rule;
  EXPECT_EQ(desc.GetExpandDimsRule(rule), GRAPH_FAILED);
}

TEST_F(TensorUT, IncCov_TensorDescConstDataAndCopy) {
  TensorDesc desc;
  auto const_data = std::unique_ptr<uint8_t[]>(new uint8_t[4]{1, 2, 3, 4});
  desc.SetConstData(std::move(const_data), 4);
  uint8_t *get_data = nullptr;
  size_t get_len = 0;
  EXPECT_TRUE(desc.GetConstData(&get_data, get_len));
  EXPECT_EQ(get_len, 4U);
  TensorDesc copy_desc(desc);
  uint8_t *copy_data = nullptr;
  size_t copy_len = 0;
  EXPECT_TRUE(copy_desc.GetConstData(&copy_data, copy_len));
  EXPECT_EQ(copy_len, 4U);
  TensorDesc assign_desc;
  assign_desc = desc;
  uint8_t *assign_data = nullptr;
  size_t assign_len = 0;
  EXPECT_TRUE(assign_desc.GetConstData(&assign_data, assign_len));
  EXPECT_EQ(assign_len, 4U);
  TensorDesc self_assign = desc;
  self_assign = self_assign;
  TensorDesc empty_desc;
  TensorDesc copy_empty(empty_desc);
  TensorDesc assign_empty;
  assign_empty = empty_desc;
  desc.impl = nullptr;
  desc.SetConstData(nullptr, 0);
  EXPECT_FALSE(desc.GetConstData(&get_data, get_len));
}

TEST_F(TensorUT, IncCov_TensorNullImpl) {
  Tensor tensor;
  tensor.impl = nullptr;
  EXPECT_EQ(tensor.GetTensorDesc().GetShape().GetDimNum(), 0U);
  EXPECT_EQ(tensor.SetTensorDesc(TensorDesc()), GRAPH_FAILED);
  EXPECT_EQ(tensor.GetData(), nullptr);
  EXPECT_EQ(tensor.GetSize(), 0U);
  EXPECT_EQ(tensor.ResetData(), nullptr);
  std::vector<uint8_t> vec = {1, 2};
  std::vector<uint8_t> vec2 = {1, 2};
  EXPECT_EQ(tensor.SetData(std::move(vec2)), GRAPH_FAILED);
  EXPECT_EQ(tensor.SetData(vec), GRAPH_FAILED);
  EXPECT_EQ(tensor.SetData(vec.data(), vec.size()), GRAPH_FAILED);
  EXPECT_EQ(tensor.SetData(std::string("test")), GRAPH_FAILED);
  EXPECT_EQ(tensor.SetData(std::vector<std::string>({"a"})), GRAPH_FAILED);
  EXPECT_EQ(tensor.SetData(static_cast<const char_t *>("test")), GRAPH_FAILED);
  EXPECT_EQ(tensor.SetData(std::vector<AscendString>({AscendString("a")})), GRAPH_FAILED);
  uint8_t *data_ptr = new uint8_t[10];
  EXPECT_EQ(tensor.SetData(data_ptr, 10, [](uint8_t *p) { delete[] p; }), GRAPH_FAILED);
  delete[] data_ptr;
  EXPECT_EQ(tensor.ResetData(data_ptr, 10, [](uint8_t *p) { delete[] p; }), GRAPH_FAILED);
}

TEST_F(TensorUT, IncCov_TensorSetDataStringAndVectors) {
  Tensor tensor;
  EXPECT_EQ(tensor.SetData(std::string("")), GRAPH_FAILED);
  EXPECT_EQ(tensor.SetData(std::string("hello")), GRAPH_SUCCESS);
  EXPECT_EQ(tensor.SetData(std::vector<std::string>({"a", "b"})), GRAPH_SUCCESS);
  EXPECT_EQ(tensor.SetData(std::vector<std::string>()), GRAPH_FAILED);
  EXPECT_EQ(tensor.SetData(static_cast<const char_t *>(nullptr)), GRAPH_FAILED);
  EXPECT_EQ(tensor.SetData(static_cast<const char_t *>("test")), GRAPH_SUCCESS);
}

TEST_F(TensorUT, IncCov_TensorAdapterNullTensor) {
  Tensor tensor;
  tensor.impl = nullptr;
  EXPECT_EQ(TensorAdapter::AsGeTensorPtr(static_cast<const Tensor &>(tensor)), nullptr);
  EXPECT_EQ(TensorAdapter::AsGeTensorPtr(tensor), nullptr);
  auto ge1 = TensorAdapter::AsGeTensor(static_cast<const Tensor &>(tensor));
  auto ge2 = TensorAdapter::AsGeTensor(tensor);
  EXPECT_EQ(TensorAdapter::AsBareGeTensorPtr(tensor), nullptr);
  auto ge3 = TensorAdapter::AsGeTensorShared(tensor);
  EXPECT_EQ(TensorAdapter::GeTensor2Tensor(nullptr).GetSize(), 0U);
}

TEST_F(TensorUT, IncCov_TensorAdapterMethods) {
  GeTensorDesc ge_desc(GeShape({1, 2}), FORMAT_NCHW, DT_FLOAT);
  GeTensor ge_tensor_from_desc(ge_desc);
  auto tensor = TensorAdapter::AsTensor(ge_tensor_from_desc);
  auto ge_tensor = TensorAdapter::AsGeTensor(static_cast<const Tensor &>(tensor));
  EXPECT_EQ(ge_tensor.GetTensorDesc().GetDataType(), DT_FLOAT);
  GeTensor ge_tensor2(ge_desc, std::vector<uint8_t>({1, 2, 3, 4}));
  auto tensor2 = TensorAdapter::AsTensor(ge_tensor2);
  EXPECT_EQ(tensor2.GetSize(), 4U);
  auto ge_tensor3 = TensorAdapter::AsGeTensorShared(tensor2);
  auto normalized = TensorAdapter::NormalizeGeTensor(ge_tensor2);
  TensorAdapter::NormalizeGeTensorDesc(ge_desc);
  auto ge_tensor_ptr = TensorAdapter::AsGeTensorPtr(tensor2);
  EXPECT_NE(ge_tensor_ptr, nullptr);
  auto const_ge_tensor_ptr = TensorAdapter::AsGeTensorPtr(static_cast<const Tensor &>(tensor2));
  EXPECT_NE(const_ge_tensor_ptr, nullptr);
  auto bare_ptr = TensorAdapter::AsBareGeTensorPtr(tensor2);
  EXPECT_NE(bare_ptr, nullptr);
  EXPECT_EQ(TensorAdapter::GeTensor2Tensor(ge_tensor_ptr).GetSize(), 4U);
}

TEST_F(TensorUT, IncCov_TensorCloneAndIsValid) {
  Tensor tensor(TensorDesc(Shape({2, 3}), FORMAT_NCHW, DT_FLOAT), std::vector<uint8_t>(24, 1));
  auto cloned = tensor.Clone();
  EXPECT_EQ(cloned.GetSize(), 24U);
  EXPECT_EQ(cloned.GetTensorDesc().GetDataType(), DT_FLOAT);

  TensorDesc desc(Shape({-1, 2}), FORMAT_NCHW, DT_FLOAT);
  Tensor tensor2(desc, std::vector<uint8_t>(8, 0));
  EXPECT_EQ(tensor2.IsValid(), GRAPH_SUCCESS);

  TensorDesc string_desc(Shape({1}), FORMAT_NCHW, DT_STRING);
  Tensor string_tensor(string_desc);
  string_tensor.SetData(std::string("test"));
  EXPECT_EQ(string_tensor.IsValid(), GRAPH_SUCCESS);
}

TEST_F(TensorUT, IncCov_TensorDescSetNameAndExpandDims) {
  TensorDesc desc;
  desc.SetName("test_name");
  EXPECT_EQ(desc.GetName(), "test_name");
  AscendString name;
  EXPECT_EQ(desc.GetName(name), GRAPH_SUCCESS);
  EXPECT_STREQ(name.GetString(), "test_name");
  desc.SetName(static_cast<const char_t *>(nullptr));
  EXPECT_EQ(desc.GetName(), "test_name");
  desc.SetExpandDimsRule(AscendString("0011"));
  AscendString rule;
  EXPECT_EQ(desc.GetExpandDimsRule(rule), GRAPH_SUCCESS);
  EXPECT_STREQ(rule.GetString(), "0011");
  desc.SetReuseInputIndex(5);
}
}  // namespace ge
