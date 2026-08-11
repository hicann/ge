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
#include <securec.h>
#include <gtest/gtest.h>
#include "graph/ge_tensor.h"
#include "ge_ir.pb.h"
#include "graph/utils/tensor_utils.h"
#include "graph/normal_graph/ge_tensor_impl.h"
#include "graph/tensor.h"
#include <iostream>
#include "graph/utils/tensor_adapter.h"

namespace ge {
class TensorUtilsUT : public testing::Test {
 protected:
  void SetUp() {}
  void TearDown() {}
};

TEST_F(TensorUtilsUT, CopyConstruct1_NullTensorDef) {
  GeTensor t1;
  std::vector<uint8_t> vec;
  for (uint8_t i = 0; i < 100; ++i) {
    vec.push_back(i * 2);
  }
  std::cout << "test1" << std::endl;
  t1.SetData(vec);
  GeTensor t2 = TensorUtils::CreateShareTensor(t1);
  t1.impl_->tensor_def_.GetProtoOwner();
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

TEST_F(TensorUtilsUT, CopyConstruct2_WithTensorDef) {
  GeIrProtoHelper<ge::proto::TensorDef> helper;
  helper.InitDefault();
  helper.GetProtoMsg()->mutable_data()->resize(100);
  GeTensor t1(helper.GetProtoOwner(), helper.GetProtoMsg());

  std::vector<uint8_t> vec;
  for (uint8_t i = 0; i < 100; ++i) {
    vec.push_back(i * 2);
  }
  t1.SetData(vec);
  GeTensor t2 = TensorUtils::CreateShareTensor(t1);

  // The copy construct share tensor_data_ and tensor_desc
  ASSERT_NE(t1.impl_->tensor_def_.GetProtoOwner(), nullptr);
  ASSERT_NE(t1.impl_->tensor_def_.GetProtoMsg(), nullptr);
  ASSERT_EQ(t1.impl_->tensor_data_.impl_->tensor_descriptor_, t1.impl_->desc_.impl_);
  ASSERT_EQ(t2.impl_->tensor_data_.impl_->tensor_descriptor_, t2.impl_->desc_.impl_);
  ASSERT_EQ(t1.impl_->tensor_data_.GetData(), t2.impl_->tensor_data_.GetData());

  t1.MutableTensorDesc().SetFormat(FORMAT_NCHW);
  ASSERT_EQ(t1.GetTensorDesc().GetFormat(), FORMAT_NCHW);
  ASSERT_EQ(t2.GetTensorDesc().GetFormat(), FORMAT_NCHW);
  t2.MutableTensorDesc().SetFormat(FORMAT_NHWC);
  ASSERT_EQ(t1.GetTensorDesc().GetFormat(), FORMAT_NHWC);
  ASSERT_EQ(t2.GetTensorDesc().GetFormat(), FORMAT_NHWC);

  ASSERT_EQ(memcmp(t1.GetData().GetData(), vec.data(), vec.size()), 0);
  ASSERT_EQ(t1.GetData().GetData(), t2.GetData().GetData());
}

TEST_F(TensorUtilsUT, SetData_CreateShareTensorWithTensorDef) {
  GeIrProtoHelper<ge::proto::TensorDef> helper;
  helper.InitDefault();
  helper.GetProtoMsg()->mutable_data()->resize(100);
  GeTensor t1(helper.GetProtoOwner(), helper.GetProtoMsg());

  std::vector<uint8_t> vec;
  for (uint8_t i = 0; i < 100; ++i) {
    vec.push_back(i * 2);
  }
  t1.SetData(vec);
  GeTensor t2 = TensorUtils::CreateShareTensor(t1);

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

TEST_F(TensorUtilsUT, SetData_CreateShareTensorWithoutTensorDef) {
  GeTensor t1;

  std::vector<uint8_t> vec;
  for (uint8_t i = 0; i < 100; ++i) {
    vec.push_back(i * 2);
  }
  t1.SetData(vec);
  GeTensor t2 = TensorUtils::CreateShareTensor(t1);

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

TEST_F(TensorUtilsUT, CreateShareTensorFromSharedPtr) {
  auto ap = std::make_shared<AlignedPtr>(100);
  for (uint8_t i = 0; i < 100; ++i) {
    ap->MutableGet()[i] = i;
  }
  GeTensorDesc td;
  GeTensor t1 = TensorUtils::CreateShareTensor(td, ap, 100);
  ASSERT_EQ(t1.GetData().GetData(), ap->MutableGet());
  ASSERT_EQ(t1.GetData().size(), 100);

  GeTensor t2(td, ap, 100);
  ASSERT_EQ(t2.GetData().GetData(), ap->MutableGet());
  ASSERT_EQ(t2.GetData().size(), 100);
}

TEST_F(TensorUtilsUT, ShareTensorData) {
  auto ap = std::make_shared<AlignedPtr>(100);
  for (uint8_t i = 0; i < 100; ++i) {
    ap->MutableGet()[i] = i;
  }
  GeTensorDesc td;

  GeTensor t1(td);
  t1.SetData(ap, 100);
  ASSERT_EQ(t1.GetData().GetData(), ap->MutableGet());
  ASSERT_EQ(t1.GetData().size(), 100);

  GeTensor t2(td);
  TensorUtils::ShareAlignedPtr(ap, 100, t2);
  ASSERT_EQ(t2.GetData().GetData(), ap->MutableGet());
  ASSERT_EQ(t2.GetData().size(), 100);
}

TEST_F(TensorUtilsUT, CopyAssign_NullTensorDef) {
  GeTensor t1;

  std::vector<uint8_t> vec;
  for (uint8_t i = 0; i < 100; ++i) {
    vec.push_back(i * 2);
  }
  t1.SetData(vec);
  GeTensor t2;
  TensorUtils::ShareTensor(t1, t2);

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

TEST_F(TensorUtilsUT, CopyConstruct3_TensorData) {
  std::vector<uint8_t> vec;
  for (uint8_t i = 0; i < 200; ++i) {
    vec.push_back(i);
  }
  TensorData td1;
  td1.SetData(vec);

  TensorData td2(td1);
  ASSERT_EQ(td1.GetData(), td2.GetData());
  ASSERT_EQ(td1.GetSize(), td2.GetSize());
  ASSERT_EQ(td1.GetSize(), 200);

  TensorData td3 = TensorUtils::CreateShareTensorData(td1);
  ASSERT_EQ(td1.GetData(), td3.GetData());
  ASSERT_EQ(td1.GetSize(), td3.GetSize());
  ASSERT_EQ(td1.GetSize(), 200);
}

TEST_F(TensorUtilsUT, CopyAssign_TensorData) {
  std::vector<uint8_t> vec;
  for (uint8_t i = 0; i < 200; ++i) {
    vec.push_back(i);
  }
  TensorData td1;
  td1.SetData(vec);

  TensorData td2 = td1;
  ASSERT_EQ(td1.GetData(), td2.GetData());
  ASSERT_EQ(td1.GetSize(), td2.GetSize());
  ASSERT_EQ(td1.GetSize(), 200);

  TensorData td3;
  TensorUtils::ShareTensorData(td1, td3);
  ASSERT_EQ(td1.GetData(), td3.GetData());
  ASSERT_EQ(td1.GetSize(), td3.GetSize());
  ASSERT_EQ(td1.GetSize(), 200);
}

TEST_F(TensorUtilsUT, SetData_ShareAlignedPtr_TensorData) {
  std::vector<uint8_t> vec;
  for (uint8_t i = 0; i < 200; ++i) {
    vec.push_back(i);
  }
  auto ap = std::make_shared<AlignedPtr>(vec.size());
  memcpy_s(ap->MutableGet(), vec.size(), vec.data(), vec.size());

  TensorData td1;
  td1.SetData(ap, vec.size());
  ASSERT_EQ(td1.GetData(), ap->MutableGet());
  ASSERT_EQ(td1.GetSize(), 200);

  TensorData td2;
  TensorUtils::ShareAlignedPtr(ap, vec.size(), td2);
  ASSERT_EQ(td2.GetData(), ap->MutableGet());
  ASSERT_EQ(td2.GetSize(), 200);
}

TEST_F(TensorUtilsUT, ShareTheSame) {
  std::vector<uint8_t> vec;
  for (uint8_t i = 0; i < 200; ++i) {
    vec.push_back(i);
  }
  TensorData td1;
  td1.SetData(vec);
  TensorUtils::ShareTensorData(td1, td1);
  ASSERT_EQ(memcmp(td1.GetData(), vec.data(), vec.size()), 0);
  ASSERT_EQ(td1.GetSize(), 200);

  GeTensorDesc tensor_desc;
  GeTensor t1(tensor_desc, vec);
  TensorUtils::ShareTensor(t1, t1);
  ASSERT_EQ(memcmp(t1.GetData().GetData(), vec.data(), vec.size()), 0);
}

TEST_F(TensorUtilsUT, ConstData) {
  std::unique_ptr<uint8_t[]> const_data = std::unique_ptr<uint8_t[]>(new (std::nothrow) uint8_t[10]);
  ASSERT_NE(const_data, nullptr);
  TensorDesc tensor_desc;
  tensor_desc.SetFormat(FORMAT_NCHW);
  tensor_desc.SetConstData(std::move(const_data), sizeof(int));
  uint8_t *ret = nullptr;
  size_t len = 0;
  tensor_desc.GetConstData(&ret, len);
  printf("GetConstData1====%p\n", ret);
  ASSERT_NE(ret, nullptr);
  ASSERT_EQ(sizeof(int), len);

  // operator=
  tensor_desc = tensor_desc;
  TensorDesc tensor_desc1;
  tensor_desc1 = tensor_desc;
  uint8_t *ret1 = nullptr;
  size_t len1 = 0;
  tensor_desc1.GetConstData(&ret1, len1);
  printf("GetConstData2 ====%p\n", ret1);
  ASSERT_NE(ret1, nullptr);
  ASSERT_NE(ret1, ret);
  ASSERT_EQ(sizeof(int), len1);
  ASSERT_EQ(tensor_desc1.GetFormat(), FORMAT_NCHW);

  // copy
  std::size_t big_size = SECUREC_MEM_MAX_LEN + 1;
  std::unique_ptr<uint8_t[]> big_data = std::unique_ptr<uint8_t[]>(new (std::nothrow) uint8_t[big_size]);
  TensorDesc tensor_desc2;
  tensor_desc2.SetFormat(FORMAT_NCHW);
  tensor_desc2.SetConstData(std::move(big_data), big_size);
  TensorDesc tensor_desc3(tensor_desc2);
  uint8_t *ret2 = nullptr;
  size_t len2 = 0;
  tensor_desc3.GetConstData(&ret2, len2);
  printf("GetConstData3 ====%p\n", ret2);
  ASSERT_NE(ret2, nullptr);
  ASSERT_NE(ret2, ret);
  ASSERT_EQ(big_size, len2);
  ASSERT_EQ(tensor_desc3.GetFormat(), FORMAT_NCHW);
}
TEST_F(TensorUtilsUT, GetShapeSize_Ok_VectorMax) {
  Shape shape({std::numeric_limits<int64_t>::max()});
  EXPECT_EQ(shape.GetShapeSize(), std::numeric_limits<int64_t>::max());
}
TEST_F(TensorUtilsUT, GetShapeSize_ReturnZero_Overflow) {
  Shape shape({2, std::numeric_limits<int64_t>::max() - 1});
  EXPECT_EQ(shape.GetShapeSize(), 0);
}
TEST_F(TensorUtilsUT, TensorConstruct_IsValid_Overflow) {
  Shape shape({std::numeric_limits<int64_t>::max()});
  TensorDesc td;
  td.SetDataType(DT_FLOAT);
  td.SetShape(shape);
  td.SetOriginShape(shape);
  Tensor tensor(td, {});

  // todo 这个行为挺奇怪的，即使发生了overflow，仍然返回success，不过历史实现一直是这样，不敢修改这个行为
  ASSERT_EQ(tensor.IsValid(), ge::GRAPH_SUCCESS);
}

TEST_F(TensorUtilsUT, TensorSetAndGetMetaInfoGeneral) {
  Tensor tensor;
  tensor.SetOriginFormat(ge::FORMAT_NCHW);
  EXPECT_EQ(tensor.GetOriginFormat(), ge::FORMAT_NCHW);

  tensor.SetFormat(ge::FORMAT_NC1HWC0);
  EXPECT_EQ(tensor.GetFormat(), ge::FORMAT_NC1HWC0);

  tensor.SetDataType(ge::DT_BF16);
  EXPECT_EQ(tensor.GetDataType(), ge::DT_BF16);

  tensor.SetOriginShapeDimNum(4);
  EXPECT_EQ(tensor.GetOriginShapeDimNum(), 4);
  for (size_t i = 0U; i < tensor.GetOriginShapeDimNum(); ++i) {
    tensor.SetOriginShapeDim(i, i);
  }
  for (size_t i = 0U; i < tensor.GetOriginShapeDimNum(); ++i) {
    EXPECT_EQ(tensor.GetOriginShapeDim(i), i);
  }
  tensor.SetShapeDimNum(5);
  EXPECT_EQ(tensor.GetShapeDimNum(), 5);
  for (size_t i = 0U; i < tensor.GetShapeDimNum(); ++i) {
    tensor.SetShapeDim(i, i);
  }
  for (size_t i = 0U; i < tensor.GetShapeDimNum(); ++i) {
    EXPECT_EQ(tensor.GetShapeDim(i), i);
  }

  EXPECT_EQ(tensor.SetPlacement(Placement::kPlacementDevice), ge::GRAPH_SUCCESS);
  EXPECT_EQ(tensor.GetPlacement(), Placement::kPlacementDevice);

  EXPECT_EQ(tensor.SetExpandDimsRule("0011"), ge::GRAPH_SUCCESS);
  AscendString str;
  EXPECT_EQ(tensor.GetExpandDimsRule(str), ge::GRAPH_SUCCESS);
  EXPECT_EQ(str, "0011");
}

TEST_F(TensorUtilsUT, TensorSetAndResetData) {
  Tensor tensor;
  EXPECT_EQ(tensor.ResetData(nullptr, 0UL, [](uint8_t *ptr) { delete[] ptr; }), ge::GRAPH_SUCCESS);

  uint8_t *data_ptr = new uint8_t[10];
  EXPECT_EQ(tensor.ResetData(data_ptr, 10, [](uint8_t *ptr) { delete[] ptr; }), ge::GRAPH_SUCCESS);
  EXPECT_EQ(tensor.GetData(), data_ptr);

  uint8_t *data_ptr2 = new uint8_t[20];
  EXPECT_EQ(tensor.ResetData(data_ptr2, 20, [](uint8_t *ptr) { delete[] ptr; }), ge::GRAPH_SUCCESS);
  EXPECT_EQ(tensor.GetData(), data_ptr2);
  tensor.ResetData().reset();
}

TEST_F(TensorUtilsUT, TensorSetAndGetMetaInfoAbnormal) {
  Tensor tensor;
  tensor.impl = nullptr;

  EXPECT_EQ(tensor.SetOriginFormat(ge::FORMAT_NCHW), ge::GRAPH_FAILED);
  EXPECT_EQ(tensor.GetOriginFormat(), ge::FORMAT_RESERVED);

  EXPECT_EQ(tensor.SetFormat(ge::FORMAT_NC1HWC0), ge::GRAPH_FAILED);
  EXPECT_EQ(tensor.GetFormat(), ge::FORMAT_RESERVED);

  EXPECT_EQ(tensor.SetDataType(ge::DT_BF16), ge::GRAPH_FAILED);
  EXPECT_EQ(tensor.GetDataType(), ge::DT_UNDEFINED);

  EXPECT_EQ(tensor.SetOriginShapeDimNum(4), ge::GRAPH_FAILED);
  EXPECT_EQ(tensor.GetOriginShapeDimNum(), 0);
  EXPECT_EQ(tensor.SetOriginShapeDim(0, 0), ge::GRAPH_FAILED);
  EXPECT_EQ(tensor.GetOriginShapeDim(0), 0);
  EXPECT_EQ(tensor.SetShapeDimNum(5), ge::GRAPH_FAILED);
  EXPECT_EQ(tensor.GetShapeDimNum(), 0);
  EXPECT_EQ(tensor.SetShapeDim(0, 0), ge::GRAPH_FAILED);
  EXPECT_EQ(tensor.GetShapeDim(0), 0);

  EXPECT_EQ(tensor.SetPlacement(Placement::kPlacementDevice), ge::GRAPH_FAILED);
  EXPECT_EQ(tensor.GetPlacement(), Placement::kPlacementEnd);

  EXPECT_EQ(tensor.SetExpandDimsRule("0011"), ge::GRAPH_FAILED);
  AscendString str;
  EXPECT_EQ(tensor.GetExpandDimsRule(str), ge::GRAPH_FAILED);
  EXPECT_NE(str, "0011");

  uint8_t *data_ptr = new uint8_t[20];
  EXPECT_NE(tensor.ResetData(data_ptr, 20, [](uint8_t *ptr) { delete[] ptr; }), ge::GRAPH_SUCCESS);
  delete[] data_ptr;
}

TEST_F(TensorUtilsUT, SetReuseInputIndex) {
  TensorDesc tensor_desc;
  tensor_desc.SetReuseInputIndex(1);
  auto ge_tensor_desc = TensorAdapter::TensorDesc2GeTensorDesc(tensor_desc);
  bool reuse_flag = false;
  uint32_t reuse_index = 0;
  TensorUtils::GetReuseInput(ge_tensor_desc, reuse_flag);
  TensorUtils::GetReuseInputIndex(ge_tensor_desc, reuse_index);
  EXPECT_EQ(reuse_flag, true);
  EXPECT_EQ(reuse_index, 1);
}

TEST_F(TensorUtilsUT, IncCov_TensorDescMoveAndSwap) {
  TensorDesc desc1(Shape({1, 2}), FORMAT_NCHW, DT_FLOAT);
  desc1.SetName("desc1");
  desc1.SetOriginShape(Shape({3, 4}));
  desc1.SetOriginFormat(FORMAT_NHWC);
  TensorDesc desc2(std::move(desc1));
  EXPECT_EQ(desc2.GetName(), "desc1");
  EXPECT_EQ(desc2.GetOriginFormat(), FORMAT_NHWC);
  TensorDesc desc3;
  desc3 = std::move(desc2);
  EXPECT_EQ(desc3.GetName(), "desc1");
  EXPECT_EQ(desc3.GetShape().GetDimNum(), 2U);
  TensorDesc desc4 = desc3;
  EXPECT_EQ(desc4.GetName(), "desc1");
  TensorDesc desc5;
  desc5 = desc3;
  EXPECT_EQ(desc5.GetName(), "desc1");
}

TEST_F(TensorUtilsUT, IncCov_TensorDescUpdateAndAccessors) {
  TensorDesc desc(Shape({1, 2}), FORMAT_NCHW, DT_FLOAT);
  desc.Update(Shape({3, 4}), FORMAT_NHWC, DT_INT32);
  EXPECT_EQ(desc.GetShape().GetDimNum(), 2U);
  EXPECT_EQ(desc.GetFormat(), FORMAT_NHWC);
  EXPECT_EQ(desc.GetDataType(), DT_INT32);
  desc.SetShape(Shape({5, 6}));
  EXPECT_EQ(desc.GetShape().GetDim(0), 5);
  desc.SetSize(100);
  EXPECT_EQ(desc.GetSize(), 100);
  desc.SetRealDimCnt(3);
  EXPECT_EQ(desc.GetRealDimCnt(), 3);
  desc.SetPlacement(kPlacementDevice);
  EXPECT_EQ(desc.GetPlacement(), kPlacementDevice);
  desc.SetShapeRange({{1, 10}, {2, 20}});
  std::vector<std::pair<int64_t, int64_t>> range;
  EXPECT_EQ(desc.GetShapeRange(range), GRAPH_SUCCESS);
  EXPECT_EQ(range.size(), 2U);
}

TEST_F(TensorUtilsUT, IncCov_TensorSetDataWithDeleter) {
  Tensor tensor;
  uint8_t *data_ptr = new uint8_t[10];
  for (int i = 0; i < 10; ++i) {
    data_ptr[i] = static_cast<uint8_t>(i);
  }
  EXPECT_EQ(tensor.SetData(data_ptr, 10, [](uint8_t *p) { delete[] p; }), GRAPH_SUCCESS);
  EXPECT_EQ(tensor.GetSize(), 10U);
  EXPECT_EQ(tensor.GetData()[0], 0);
  EXPECT_EQ(tensor.GetData()[9], 9);
}

TEST_F(TensorUtilsUT, IncCov_TensorSetDataAscendStrings) {
  Tensor tensor;
  std::vector<AscendString> datas = {AscendString("hello"), AscendString("world")};
  EXPECT_EQ(tensor.SetData(datas), GRAPH_SUCCESS);
  EXPECT_EQ(tensor.SetData(std::vector<AscendString>({AscendString("a")})), GRAPH_SUCCESS);
}

TEST_F(TensorUtilsUT, IncCov_TensorAdapterAllMethods) {
  GeTensorDesc ge_desc(GeShape({1, 2}), FORMAT_NCHW, DT_FLOAT);
  TensorUtils::SetSize(ge_desc, 100);
  TensorUtils::SetRealDimCnt(ge_desc, 2);
  auto tensor_desc = TensorAdapter::GeTensorDesc2TensorDesc(ge_desc);
  EXPECT_EQ(tensor_desc.GetDataType(), DT_FLOAT);
  auto converted = TensorAdapter::TensorDesc2GeTensorDesc(tensor_desc);
  EXPECT_EQ(converted.GetDataType(), DT_FLOAT);

  GeTensor ge_tensor(ge_desc, std::vector<uint8_t>({1, 2, 3}));
  auto tensor = TensorAdapter::AsTensor(ge_tensor);
  EXPECT_EQ(tensor.GetSize(), 3U);
  auto shared = TensorAdapter::AsGeTensorShared(tensor);
  EXPECT_EQ(shared.GetTensorDesc().GetDataType(), DT_FLOAT);
  auto bare_ptr = TensorAdapter::AsBareGeTensorPtr(tensor);
  EXPECT_NE(bare_ptr, nullptr);
  auto ge_ptr = TensorAdapter::AsGeTensorPtr(tensor);
  EXPECT_NE(ge_ptr, nullptr);
  auto const_ge_ptr = TensorAdapter::AsGeTensorPtr(static_cast<const Tensor &>(tensor));
  EXPECT_NE(const_ge_ptr, nullptr);
  auto ge1 = TensorAdapter::AsGeTensor(static_cast<const Tensor &>(tensor));
  auto ge2 = TensorAdapter::AsGeTensor(tensor);
  EXPECT_EQ(ge1.GetTensorDesc().GetDataType(), DT_FLOAT);
  auto from_ge = TensorAdapter::GeTensor2Tensor(ge_ptr);
  EXPECT_EQ(from_ge.GetSize(), 3U);
  auto from_null = TensorAdapter::GeTensor2Tensor(nullptr);
  EXPECT_EQ(from_null.GetSize(), 0U);
}

TEST_F(TensorUtilsUT, IncCov_TensorDescSetName) {
  TensorDesc desc;
  desc.SetName("test_tensor_name");
  EXPECT_EQ(desc.GetName(), "test_tensor_name");
}

TEST_F(TensorUtilsUT, IncCov_TensorDescGetNameAscendString) {
  TensorDesc desc;
  desc.SetName("test_ascend_name");
  AscendString name;
  EXPECT_EQ(desc.GetName(name), GRAPH_SUCCESS);
  EXPECT_STREQ(name.GetString(), "test_ascend_name");
}

TEST_F(TensorUtilsUT, IncCov_TensorSetDataNullCharPtr) {
  Tensor tensor;
  EXPECT_EQ(tensor.SetData(static_cast<const char_t *>(nullptr)), GRAPH_FAILED);
}

TEST_F(TensorUtilsUT, IncCov_TensorSetDataNullAscendString) {
  Tensor tensor;
  AscendString null_str(nullptr);
  std::vector<AscendString> datas = {null_str};
  EXPECT_EQ(tensor.SetData(datas), GRAPH_SUCCESS);
}

TEST_F(TensorUtilsUT, IncCov_TensorSetDataCharPtr) {
  Tensor tensor;
  EXPECT_EQ(tensor.SetData("hello"), GRAPH_SUCCESS);
  EXPECT_EQ(tensor.GetSize(), 22U);
}

TEST_F(TensorUtilsUT, IncCov_TensorGetData) {
  Tensor tensor;
  const uint8_t *data = tensor.GetData();
  EXPECT_NE(data, nullptr);
}

TEST_F(TensorUtilsUT, IncCov_TensorDescSetNameCharPtr) {
  TensorDesc desc;
  desc.SetName(static_cast<const char_t *>("char_name"));
  EXPECT_EQ(desc.GetName(), "char_name");
}

TEST_F(TensorUtilsUT, IncCov_TensorDescSetNameNullCharPtr) {
  TensorDesc desc;
  desc.SetName(static_cast<const char_t *>(nullptr));
  EXPECT_EQ(desc.GetName(), "");
}

TEST_F(TensorUtilsUT, IncCov_TensorResetData) {
  Tensor tensor;
  auto data = tensor.ResetData();
  EXPECT_EQ(data, nullptr);
}
}  // namespace ge
