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
#include "common/memory/tensor_trans_utils.h"
#include "../../../graph_metadef/depends/checker/tensor_check_utils.h"
#include "ge/ge_api_error_codes.h"
#include "graph_metadef/graph/utils/tensor_adapter.h"
#include "runtime/gert_api.h"
namespace ge {
namespace {
bool IsEqualWith(const gert::Shape &rt_shape, const ge::Shape &ge_shape) {
  if (rt_shape.GetDimNum() != ge_shape.GetDimNum()) {
    return false;
  }

  for (size_t i = 0U; i < rt_shape.GetDimNum(); ++i) {
    if (rt_shape.GetDim(i) != ge_shape.GetDim(i)) {
      return false;
    }
  }
  return true;
}
}
class UtestTensorTransUtils : public testing::Test {
 protected:
  void SetUp() {}
  void TearDown() {}
};

TEST_F(UtestTensorTransUtils, TransDeviceRtTensorToHostTensor_without_value_Success) {
  //  Status TensorTransUtils::TransRtTensorToTensor(const std::vector<gert::Tensor> &srcs,
  //                                                           std::vector<Tensor> &dsts, bool with_value);
  gert::Tensor tensor0 = {{{1000, 1000, 1, 1000}, {3, 1000, 1, 1000}},  // shape
                          {ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, {}},   // format
                          gert::kOnDeviceHbm,                           // placement
                          ge::DT_FLOAT,                                 // data type
                          nullptr};
  gert::Tensor tensor1 = {{{100, 100, 1, 100}, {3, 100, 1, 100}},  // shape
                          {ge::FORMAT_ND, ge::FORMAT_NC1HWC0, {}},   // format
                          gert::kOnDeviceHbm,                           // placement
                          ge::DT_FLOAT,                                 // data type
                          nullptr};
  std::vector<gert::Tensor> srcs;
  srcs.emplace_back(std::move(tensor0));
  srcs.emplace_back(std::move(tensor1));

  std::vector<Tensor> dsts;
  EXPECT_EQ(TensorTransUtils::TransRtTensorToTensor(srcs, dsts, false), GRAPH_SUCCESS);

  EXPECT_EQ(dsts.size(), srcs.size());
  EXPECT_EQ(dsts[1].GetPlacement(), ge::kPlacementEnd);
  const auto dst_tensor_desc = dsts[1].GetTensorDesc();
  EXPECT_EQ(dst_tensor_desc.GetPlacement(), ge::kPlacementEnd);
  EXPECT_TRUE(IsEqualWith(srcs[1].GetOriginShape(), dst_tensor_desc.GetOriginShape()));
  EXPECT_TRUE(IsEqualWith(srcs[1].GetStorageShape(), dst_tensor_desc.GetShape()));
  EXPECT_EQ(srcs[1].GetOriginFormat(), dst_tensor_desc.GetOriginFormat());
  EXPECT_EQ(srcs[1].GetFormat().GetStorageFormat(), dst_tensor_desc.GetFormat());
  EXPECT_EQ(srcs[1].GetFormat().GetOriginFormat(), dst_tensor_desc.GetOriginFormat());
}

TEST_F(UtestTensorTransUtils, TransDeviceRtTensorToHostTensor_empty_rttensor_with_value_Success) {
  gert::Tensor tensor0 = {{{1000, 1000, 0, 1000}, {3, 1000, 0, 1000}},  // shape
                          {ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, {}},   // format
                          gert::kOnDeviceHbm,                           // placement
                          ge::DT_FLOAT,                                 // data type
                          nullptr};
  std::vector<gert::Tensor> srcs;
  srcs.emplace_back(std::move(tensor0));

  std::vector<Tensor> dsts;
  EXPECT_EQ(TensorTransUtils::TransRtTensorToTensor(srcs, dsts, true), GRAPH_SUCCESS);

  EXPECT_EQ(dsts.size(), srcs.size());
  EXPECT_EQ(dsts[0].GetPlacement(), ge::kPlacementHost);
  const auto dst_tensor_desc = dsts[0].GetTensorDesc();
  EXPECT_EQ(dst_tensor_desc.GetPlacement(), ge::kPlacementHost);
  EXPECT_TRUE(IsEqualWith(srcs[0].GetOriginShape(), dst_tensor_desc.GetOriginShape()));
  EXPECT_TRUE(IsEqualWith(srcs[0].GetStorageShape(), dst_tensor_desc.GetShape()));
  EXPECT_EQ(srcs[0].GetOriginFormat(), dst_tensor_desc.GetOriginFormat());
  EXPECT_EQ(srcs[0].GetFormat().GetStorageFormat(), dst_tensor_desc.GetFormat());
  EXPECT_EQ(srcs[0].GetFormat().GetOriginFormat(), dst_tensor_desc.GetOriginFormat());
  EXPECT_EQ(dsts[0].GetTensorDesc().GetSize(), 0);
}

TEST_F(UtestTensorTransUtils, GertTensors2GeTensors) {
  std::vector<gert::Tensor> gert_tensors;
  gert_tensors.resize(2);
  TensorCheckUtils::ConstructGertTensor(gert_tensors[0]);
  TensorCheckUtils::ConstructGertTensor(gert_tensors[1]);

  std::vector<GeTensor> ge_tensors;
  ASSERT_EQ(TensorTransUtils::GertTensors2GeTensors(gert_tensors, ge_tensors), SUCCESS);
  ASSERT_EQ(ge_tensors.size(), gert_tensors.size());
  for (size_t i = 0; i < gert_tensors.size(); i++) {
    EXPECT_EQ(TensorCheckUtils::CheckGeTensorEqGertTensor(ge_tensors[i], gert_tensors[i]), SUCCESS);
  }
  ge_tensors.clear();
  gert_tensors.clear();
}

TEST_F(UtestTensorTransUtils, GertTensors2Tensors) {
  std::vector<gert::Tensor> gert_tensors;
  gert_tensors.resize(2);
  TensorCheckUtils::ConstructGertTensor(gert_tensors[0]);
  TensorCheckUtils::ConstructGertTensor(gert_tensors[1]);

  std::vector<Tensor> tensors;
  ASSERT_EQ(TensorTransUtils::GertTensors2Tensors(gert_tensors, tensors), SUCCESS);
  ASSERT_EQ(tensors.size(), gert_tensors.size());
  for (size_t i = 0; i < gert_tensors.size(); i++) {
    EXPECT_EQ(TensorCheckUtils::CheckTensorEqGertTensor(tensors[i], gert_tensors[i]), SUCCESS);
  }
  tensors.clear();
  gert_tensors.clear();
}

/*
 * 构造ge_tensor, 转换成为gert_tensor, 释放ge_tensor, 校验gert_tensor的数据，表示ge_tensor释放后，gert_tensor还可以访问数据
 */
TEST_F(UtestTensorTransUtils, GeTensors2GertTensors_CheckDataAfterFreeGeTensors) {
  std::vector<GeTensor> ge_tensors;
  ge_tensors.resize(2);
  TensorCheckUtils::ConstructGeTensor(ge_tensors[0]);
  TensorCheckUtils::ConstructGeTensor(ge_tensors[1]);

  std::vector<gert::Tensor> gert_tensors;
  ASSERT_EQ(TensorTransUtils::GeTensors2GertTensors(ge_tensors, gert_tensors), SUCCESS);
  ASSERT_EQ(ge_tensors.size(), gert_tensors.size());
  for (size_t i = 0; i < gert_tensors.size(); i++) {
    EXPECT_EQ(TensorCheckUtils::CheckGeTensorEqGertTensor(ge_tensors[i], gert_tensors[i]), SUCCESS);
  }
  const auto buffer_size = ge_tensors[0].MutableData().GetSize();
  auto buffer = new (std::nothrow) uint32_t [buffer_size];
  ASSERT_NE(buffer, nullptr);
  memccpy(buffer, ge_tensors[0].MutableData().GetData(), buffer_size, buffer_size);
  ge_tensors.clear();

  // ge_tensor释放后，gert_tensor数据仍然可用，且数据正确
  auto dst_data = reinterpret_cast<uint32_t *>(gert_tensors[0].MutableTensorData().GetAddr());
  ASSERT_EQ(buffer_size, gert_tensors[0].GetSize());

  for (size_t i = 0; i < buffer_size / sizeof(uint32_t); i++) {
    EXPECT_EQ(buffer[i], dst_data[i]);
  }
  delete [] buffer;
  gert_tensors.clear();
}

/*
 * 构造ge_tensor, 转换成为gert_tensor, 释放ge_tensor
 * gert_tensor_share和gert_tensor共享数据，分别释放
 */
TEST_F(UtestTensorTransUtils, GeTensors2GertTensors_GertTensorShare) {
  std::vector<GeTensor> ge_tensors;
  ge_tensors.resize(2);
  TensorCheckUtils::ConstructGeTensor(ge_tensors[0]);
  TensorCheckUtils::ConstructGeTensor(ge_tensors[1]);

  std::vector<gert::Tensor> gert_tensors;
  ASSERT_EQ(TensorTransUtils::GeTensors2GertTensors(ge_tensors, gert_tensors), SUCCESS);
  ASSERT_EQ(ge_tensors.size(), gert_tensors.size());
  for (size_t i = 0; i < gert_tensors.size(); i++) {
    EXPECT_EQ(TensorCheckUtils::CheckGeTensorEqGertTensor(ge_tensors[i], gert_tensors[i]), SUCCESS);
  }
  const auto buffer_size = ge_tensors[0].MutableData().GetSize();
  auto buffer = new (std::nothrow) uint32_t [buffer_size];
  ASSERT_NE(buffer, nullptr);
  memccpy(buffer, ge_tensors[0].MutableData().GetData(), buffer_size, buffer_size);
  ge_tensors.clear();

  // share from
  gert::Tensor gert_tensor_share(gert_tensors[0].GetShape(), gert_tensors[0].GetFormat(), gert_tensors[0].GetDataType());
  gert_tensor_share.MutableTensorData().ShareFrom(gert_tensors[0].GetTensorData());
  gert_tensors.clear();

  // 原gert_tensor释放后，访问gert_tensor_share
  auto dst_data = reinterpret_cast<uint32_t *>(gert_tensor_share.MutableTensorData().GetAddr());
  ASSERT_EQ(buffer_size, gert_tensor_share.GetSize());

  for (size_t i = 0; i < buffer_size / sizeof(uint32_t); i++) {
    EXPECT_EQ(buffer[i], dst_data[i]);
  }
  delete [] buffer;
}

TEST_F(UtestTensorTransUtils, Tensors2GertTensors) {
  std::vector<GeTensor> ge_tensors;
  ge_tensors.resize(10);
  for (size_t i = 0U; i < ge_tensors.size(); i++) {
    TensorCheckUtils::ConstructGeTensor(ge_tensors[i]);
  }

  std::vector<Tensor> tensors;
  tensors.resize(ge_tensors.size());
  for (size_t i = 0; i < ge_tensors.size(); i++) {
    tensors[i] = TensorAdapter::AsTensor(ge_tensors[i]);
  }

  GELOGE(FAILED, "call 10 start");
  std::vector<gert::Tensor> gert_tensors;
  ASSERT_EQ(TensorTransUtils::Tensors2GertTensors(tensors, gert_tensors), SUCCESS);
  GELOGE(FAILED, "call 10 finish");

  ASSERT_EQ(tensors.size(), gert_tensors.size());
  for (size_t i = 0; i < gert_tensors.size(); i++) {
    EXPECT_EQ(TensorCheckUtils::CheckTensorEqGertTensor(tensors[i], gert_tensors[i]), SUCCESS);
  }
  ge_tensors.clear();
  tensors.clear();
  gert_tensors.clear();
}

TEST_F(UtestTensorTransUtils, AsTensorsView) {
  constexpr size_t tensor_num = 10;
  std::vector<GeTensor> ge_tensors;
  ge_tensors.resize(tensor_num);
  for (size_t i = 0U; i < ge_tensors.size(); i++) {
    TensorCheckUtils::ConstructGeTensor(ge_tensors[i]);
  }

  std::vector<Tensor> tensors;
  tensors.resize(ge_tensors.size());
  for (size_t i = 0; i < ge_tensors.size(); i++) {
    tensors[i] = TensorAdapter::AsTensor(ge_tensors[i]);
  }

  GELOGE(FAILED, "call 10 start");
  std::vector<gert::Tensor> gert_tensors;
  ASSERT_EQ(TensorTransUtils::AsTensorsView(tensors, gert_tensors), SUCCESS);
  GELOGE(FAILED, "call 10 finish");

  ASSERT_EQ(tensors.size(), gert_tensors.size());
  for (size_t i = 0; i < gert_tensors.size(); i++) {
    EXPECT_EQ(TensorCheckUtils::CheckTensorEqGertTensor(tensors[i], gert_tensors[i]), SUCCESS);
  }

  gert_tensors.clear();
  ge_tensors.clear();
  tensors.clear();
}

/*
 * device上的gert_tensor转换成host上的gert_tensor，释放device上gert_tensor，然后校验host gert_tensor数据
 * 测试host gert_tensor的share from功能
 */
TEST_F(UtestTensorTransUtils, TransGertTensorToHost_WithShareFrom) {
  gert::Tensor host_gert_tensor;
  GeTensor ge_tensor;
  gert::Tensor device_gert_tensor;
  TensorCheckUtils::ConstructGeTensor(ge_tensor);

  ASSERT_EQ(TensorTransUtils::GeTensor2GertTensor(ge_tensor, device_gert_tensor), SUCCESS);
  EXPECT_EQ(TensorCheckUtils::CheckGeTensorEqGertTensor(ge_tensor, device_gert_tensor), SUCCESS);

  ASSERT_EQ(TensorTransUtils::TransGertTensorToHost(device_gert_tensor,
    host_gert_tensor), SUCCESS);

  // address 64 aligned
  auto addr = host_gert_tensor.GetAddr();
  EXPECT_EQ(PtrToValue(addr) % 64, 0);

  // 释放device gert_tensor
  EXPECT_EQ(TensorCheckUtils::CheckGeTensorEqGertTensor(ge_tensor, device_gert_tensor), SUCCESS);
  device_gert_tensor.MutableTensorData().Free();

  EXPECT_EQ(TensorCheckUtils::CheckGeTensorEqGertTensorWithData(ge_tensor, host_gert_tensor), SUCCESS);

  // share from，测试两个gert_tensor可单独释放
  gert::Tensor host_gert_tensor_share(host_gert_tensor.GetShape(), host_gert_tensor.GetFormat(),
    host_gert_tensor.GetDataType());
  host_gert_tensor_share.MutableTensorData().ShareFrom(host_gert_tensor.GetTensorData());

  // 原始host gert_tensor释放
  host_gert_tensor.MutableTensorData().Free();

  // share出来的host_gert_tensor_share数据访问
  EXPECT_EQ(TensorCheckUtils::CheckGeTensorEqGertTensorWithData(ge_tensor, host_gert_tensor_share), SUCCESS);
}

TEST_F(UtestTensorTransUtils, GetDimsFromGertShape_Success) {
  gert::Tensor gert_tensor;
  std::initializer_list<int64_t> shape = {99, 2, 3};
  const auto exp_dims = std::vector<int64_t>(shape);
  TensorCheckUtils::ConstructGertTensor(gert_tensor, shape);

  const auto dims = TensorTransUtils::GetDimsFromGertShape(gert_tensor.GetStorageShape());
  ASSERT_EQ(exp_dims.size(), dims.size());
  for (size_t i = 0U; i < dims.size(); i++) {
    EXPECT_EQ(exp_dims.at(i), dims.at(i));
  }
}
TEST_F(UtestTensorTransUtils, ShareFromGertTensors) {
  std::initializer_list<int64_t> shape = {99, 2, 3};
  const auto exp_dims = std::vector<int64_t>(shape);
  std::vector<gert::Tensor> tensors;
  {
    gert::Tensor gert_tensor;
    TensorCheckUtils::ConstructGertTensor(gert_tensor, shape);
    tensors.emplace_back(std::move(gert_tensor));
  }
  {
    gert::Tensor gert_tensor;
    TensorCheckUtils::ConstructGertTensor(gert_tensor, shape);
    tensors.emplace_back(std::move(gert_tensor));
  }
  std::vector<gert::Tensor> share_tensors = TensorTransUtils::ShareFromGertTenosrs(tensors);
  tensors.clear();

  for (const auto &gert_tensor : tensors) {
    auto addr = PtrToPtr<void, uint8_t>(gert_tensor.GetAddr());
    auto data = reinterpret_cast<const uint32_t *>(addr);
    for (size_t i = 0; i < gert_tensor.GetSize() / sizeof(uint32_t); ++i) {
      EXPECT_EQ(data[i], i);
    }
  }
}

TEST_F(UtestTensorTransUtils, ContructRtShapeFromGeShape) {
  GeShape ge_shape;
  ge_shape.AppendDim(1);
  auto rt_shape =TensorTransUtils::ContructRtShapeFromGeShape(ge_shape);
  EXPECT_EQ(rt_shape.GetDimNum(), 1);
}

TEST_F(UtestTensorTransUtils, ContructRtShapeFromVector) {
  auto rt_shape =TensorTransUtils::ContructRtShapeFromVector({1});
  EXPECT_EQ(rt_shape.GetDimNum(), 1);
}
}
