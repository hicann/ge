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

#include "macro_utils/dt_public_scope.h"
#include "host_kernels/selection_ops/strided_slice_kernel.h"

#include "common/debug/log.h"
#include "common/debug/memory_dumper.h"
#include "common/ge_inner_error_codes.h"
#include "common/framework_types_internal.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/passes/standard_optimize/constant_folding/dimension_compute_pass.h"
#include "host_kernels/kernel_utils.h"
#include "graph/types.h"
#include "graph/utils/attr_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/tensor_utils.h"
#include "host_kernels/kernel_factory.h"
#include "macro_utils/dt_public_unscope.h"

using namespace testing;
using namespace ge;

class UtestGraphPassesFoldingKernelStridedSliceKernel : public testing::Test {
 protected:
  void SetUp() {}

  void TearDown() {}
};

TEST_F(UtestGraphPassesFoldingKernelStridedSliceKernel, CheckInputSize) {
  OpDescPtr op_desc_ptr = std::make_shared<OpDesc>("StridedSlice", "StridedSlice");
  GeTensorDesc dims_tensor_desc(GeShape({1, 1, 1, 1}), FORMAT_NCHW, DT_FLOAT);
  op_desc_ptr->AddInputDesc(0, dims_tensor_desc);
  op_desc_ptr->AddInputDesc(1, dims_tensor_desc);
  op_desc_ptr->AddInputDesc(2, dims_tensor_desc);

  vector<int64_t> dims_vec_0;
  vector<int32_t> data_vec_0 = {1, 1, 1, 1};
  GeTensorDesc tensor_desc_0(GeShape(dims_vec_0), FORMAT_NCHW, DT_FLOAT);
  ConstGeTensorPtr tensor_0 =
      std::make_shared<GeTensor>(tensor_desc_0, (uint8_t *)data_vec_0.data(), data_vec_0.size() * sizeof(float));

  vector<ConstGeTensorPtr> input = {tensor_0};
  vector<GeTensorPtr> outputs;

  shared_ptr<Kernel> kernel = KernelFactory::Instance().Create(STRIDEDSLICE);
  EXPECT_NE(kernel->Compute(op_desc_ptr, input, outputs), ge::SUCCESS);
}

TEST_F(UtestGraphPassesFoldingKernelStridedSliceKernel, Test2) {
  OpDescPtr op_desc_ptr = std::make_shared<OpDesc>("StridedSlice", "StridedSlice");
  GeTensorDesc dims_tensor_desc(GeShape({1, 1, 1, 1}), FORMAT_NCHW, DT_FLOAT);
  op_desc_ptr->AddInputDesc(0, dims_tensor_desc);
  op_desc_ptr->AddInputDesc(1, dims_tensor_desc);
  op_desc_ptr->AddInputDesc(2, dims_tensor_desc);
  op_desc_ptr->AddInputDesc(3, dims_tensor_desc);

  vector<int64_t> dims_vec_0;
  vector<int32_t> data_vec_0 = {1, 1, 1, 1};
  GeTensorDesc tensor_desc_0(GeShape(dims_vec_0), FORMAT_NCHW, DT_FLOAT);
  ConstGeTensorPtr tensor_0 =
      std::make_shared<GeTensor>(tensor_desc_0, (uint8_t *)data_vec_0.data(), data_vec_0.size() * sizeof(float));
  ConstGeTensorPtr tensor_1 =
      std::make_shared<GeTensor>(tensor_desc_0, (uint8_t *)data_vec_0.data(), data_vec_0.size() * sizeof(float));
  ConstGeTensorPtr tensor_2 =
      std::make_shared<GeTensor>(tensor_desc_0, (uint8_t *)data_vec_0.data(), data_vec_0.size() * sizeof(float));
  ConstGeTensorPtr tensor_3 =
      std::make_shared<GeTensor>(tensor_desc_0, (uint8_t *)data_vec_0.data(), data_vec_0.size() * sizeof(float));

  vector<ConstGeTensorPtr> input = {tensor_0, tensor_1, tensor_2, tensor_3};
  vector<GeTensorPtr> outputs;

  shared_ptr<Kernel> kernel = KernelFactory::Instance().Create(STRIDEDSLICE);
  EXPECT_NE(kernel->Compute(op_desc_ptr, input, outputs), ge::SUCCESS);
}

TEST_F(UtestGraphPassesFoldingKernelStridedSliceKernel, Test3) {
  OpDescPtr op_desc_ptr = std::make_shared<OpDesc>("StridedSlice", "StridedSlice");
  AttrUtils::SetInt(op_desc_ptr, STRIDE_SLICE_ATTR_BEGIN_MASK, 0);

  GeTensorDesc dims_tensor_desc(GeShape({1, 1, 1, 1}), FORMAT_NCHW, DT_FLOAT);
  op_desc_ptr->AddInputDesc(0, dims_tensor_desc);
  op_desc_ptr->AddInputDesc(1, dims_tensor_desc);
  op_desc_ptr->AddInputDesc(2, dims_tensor_desc);
  op_desc_ptr->AddInputDesc(3, dims_tensor_desc);

  vector<int64_t> dims_vec_0;
  vector<int32_t> data_vec_0 = {1, 1, 1, 1};
  GeTensorDesc tensor_desc_0(GeShape(dims_vec_0), FORMAT_NCHW, DT_FLOAT);
  ConstGeTensorPtr tensor_0 =
      std::make_shared<GeTensor>(tensor_desc_0, (uint8_t *)data_vec_0.data(), data_vec_0.size() * sizeof(float));
  ConstGeTensorPtr tensor_1 =
      std::make_shared<GeTensor>(tensor_desc_0, (uint8_t *)data_vec_0.data(), data_vec_0.size() * sizeof(float));
  ConstGeTensorPtr tensor_2 =
      std::make_shared<GeTensor>(tensor_desc_0, (uint8_t *)data_vec_0.data(), data_vec_0.size() * sizeof(float));
  ConstGeTensorPtr tensor_3 =
      std::make_shared<GeTensor>(tensor_desc_0, (uint8_t *)data_vec_0.data(), data_vec_0.size() * sizeof(float));

  vector<ConstGeTensorPtr> input = {tensor_0, tensor_1, tensor_2, tensor_3};
  vector<GeTensorPtr> outputs;

  shared_ptr<Kernel> kernel = KernelFactory::Instance().Create(STRIDEDSLICE);
  EXPECT_NE(kernel->Compute(op_desc_ptr, input, outputs), ge::SUCCESS);
}

TEST_F(UtestGraphPassesFoldingKernelStridedSliceKernel, Test4) {
  OpDescPtr op_desc_ptr = std::make_shared<OpDesc>("StridedSlice", "StridedSlice");
  AttrUtils::SetInt(op_desc_ptr, STRIDE_SLICE_ATTR_BEGIN_MASK, 0);
  AttrUtils::SetInt(op_desc_ptr, STRIDE_SLICE_ATTR_END_MASK, 0);

  GeTensorDesc dims_tensor_desc(GeShape({1, 1, 1, 1}), FORMAT_NCHW, DT_FLOAT);
  op_desc_ptr->AddInputDesc(0, dims_tensor_desc);
  op_desc_ptr->AddInputDesc(1, dims_tensor_desc);
  op_desc_ptr->AddInputDesc(2, dims_tensor_desc);
  op_desc_ptr->AddInputDesc(3, dims_tensor_desc);

  vector<int64_t> dims_vec_0;
  vector<int32_t> data_vec_0 = {1, 1, 1, 1};
  GeTensorDesc tensor_desc_0(GeShape(dims_vec_0), FORMAT_NCHW, DT_FLOAT);
  ConstGeTensorPtr tensor_0 =
      std::make_shared<GeTensor>(tensor_desc_0, (uint8_t *)data_vec_0.data(), data_vec_0.size() * sizeof(float));
  ConstGeTensorPtr tensor_1 =
      std::make_shared<GeTensor>(tensor_desc_0, (uint8_t *)data_vec_0.data(), data_vec_0.size() * sizeof(float));
  ConstGeTensorPtr tensor_2 =
      std::make_shared<GeTensor>(tensor_desc_0, (uint8_t *)data_vec_0.data(), data_vec_0.size() * sizeof(float));
  ConstGeTensorPtr tensor_3 =
      std::make_shared<GeTensor>(tensor_desc_0, (uint8_t *)data_vec_0.data(), data_vec_0.size() * sizeof(float));

  vector<ConstGeTensorPtr> input = {tensor_0, tensor_1, tensor_2, tensor_3};
  vector<GeTensorPtr> outputs;

  shared_ptr<Kernel> kernel = KernelFactory::Instance().Create(STRIDEDSLICE);
  EXPECT_NE(kernel->Compute(op_desc_ptr, input, outputs), ge::SUCCESS);
}

TEST_F(UtestGraphPassesFoldingKernelStridedSliceKernel, Test5) {
  OpDescPtr op_desc_ptr = std::make_shared<OpDesc>("StridedSlice", "StridedSlice");
  AttrUtils::SetInt(op_desc_ptr, STRIDE_SLICE_ATTR_BEGIN_MASK, 0);
  AttrUtils::SetInt(op_desc_ptr, STRIDE_SLICE_ATTR_END_MASK, 0);
  AttrUtils::SetInt(op_desc_ptr, STRIDE_SLICE_ATTR_ELLIPSIS_MASK, 0);

  GeTensorDesc dims_tensor_desc(GeShape({1, 1, 1, 1}), FORMAT_NCHW, DT_FLOAT);
  op_desc_ptr->AddInputDesc(0, dims_tensor_desc);
  op_desc_ptr->AddInputDesc(1, dims_tensor_desc);
  op_desc_ptr->AddInputDesc(2, dims_tensor_desc);
  op_desc_ptr->AddInputDesc(3, dims_tensor_desc);

  vector<int64_t> dims_vec_0;
  vector<int32_t> data_vec_0 = {1, 1, 1, 1};
  GeTensorDesc tensor_desc_0(GeShape(dims_vec_0), FORMAT_NCHW, DT_FLOAT);
  ConstGeTensorPtr tensor_0 =
      std::make_shared<GeTensor>(tensor_desc_0, (uint8_t *)data_vec_0.data(), data_vec_0.size() * sizeof(float));
  ConstGeTensorPtr tensor_1 =
      std::make_shared<GeTensor>(tensor_desc_0, (uint8_t *)data_vec_0.data(), data_vec_0.size() * sizeof(float));
  ConstGeTensorPtr tensor_2 =
      std::make_shared<GeTensor>(tensor_desc_0, (uint8_t *)data_vec_0.data(), data_vec_0.size() * sizeof(float));
  ConstGeTensorPtr tensor_3 =
      std::make_shared<GeTensor>(tensor_desc_0, (uint8_t *)data_vec_0.data(), data_vec_0.size() * sizeof(float));

  vector<ConstGeTensorPtr> input = {tensor_0, tensor_1, tensor_2, tensor_3};
  vector<GeTensorPtr> outputs;

  shared_ptr<Kernel> kernel = KernelFactory::Instance().Create(STRIDEDSLICE);
  EXPECT_NE(kernel->Compute(op_desc_ptr, input, outputs), ge::SUCCESS);
}

TEST_F(UtestGraphPassesFoldingKernelStridedSliceKernel, Test6) {
  OpDescPtr op_desc_ptr = std::make_shared<OpDesc>("StridedSlice", "StridedSlice");
  AttrUtils::SetInt(op_desc_ptr, STRIDE_SLICE_ATTR_BEGIN_MASK, 0);
  AttrUtils::SetInt(op_desc_ptr, STRIDE_SLICE_ATTR_END_MASK, 0);
  AttrUtils::SetInt(op_desc_ptr, STRIDE_SLICE_ATTR_ELLIPSIS_MASK, 0);
  AttrUtils::SetInt(op_desc_ptr, STRIDE_SLICE_ATTR_NEW_AXIS_MASK, 0);

  GeTensorDesc dims_tensor_desc(GeShape({1, 1, 1, 1}), FORMAT_NCHW, DT_FLOAT);
  op_desc_ptr->AddInputDesc(0, dims_tensor_desc);
  op_desc_ptr->AddInputDesc(1, dims_tensor_desc);
  op_desc_ptr->AddInputDesc(2, dims_tensor_desc);
  op_desc_ptr->AddInputDesc(3, dims_tensor_desc);

  vector<int64_t> dims_vec_0;
  vector<int32_t> data_vec_0 = {1, 1, 1, 1};
  GeTensorDesc tensor_desc_0(GeShape(dims_vec_0), FORMAT_NCHW, DT_FLOAT);
  ConstGeTensorPtr tensor_0 =
      std::make_shared<GeTensor>(tensor_desc_0, (uint8_t *)data_vec_0.data(), data_vec_0.size() * sizeof(float));
  ConstGeTensorPtr tensor_1 =
      std::make_shared<GeTensor>(tensor_desc_0, (uint8_t *)data_vec_0.data(), data_vec_0.size() * sizeof(float));
  ConstGeTensorPtr tensor_2 =
      std::make_shared<GeTensor>(tensor_desc_0, (uint8_t *)data_vec_0.data(), data_vec_0.size() * sizeof(float));
  ConstGeTensorPtr tensor_3 =
      std::make_shared<GeTensor>(tensor_desc_0, (uint8_t *)data_vec_0.data(), data_vec_0.size() * sizeof(float));

  vector<ConstGeTensorPtr> input = {tensor_0, tensor_1, tensor_2, tensor_3};
  vector<GeTensorPtr> outputs;

  shared_ptr<Kernel> kernel = KernelFactory::Instance().Create(STRIDEDSLICE);
  EXPECT_NE(kernel->Compute(op_desc_ptr, input, outputs), ge::SUCCESS);
}

TEST_F(UtestGraphPassesFoldingKernelStridedSliceKernel, Test7) {
  OpDescPtr op_desc_ptr = std::make_shared<OpDesc>("StridedSlice", "StridedSlice");
  AttrUtils::SetInt(op_desc_ptr, STRIDE_SLICE_ATTR_BEGIN_MASK, 0);
  AttrUtils::SetInt(op_desc_ptr, STRIDE_SLICE_ATTR_END_MASK, 0);
  AttrUtils::SetInt(op_desc_ptr, STRIDE_SLICE_ATTR_ELLIPSIS_MASK, 0);
  AttrUtils::SetInt(op_desc_ptr, STRIDE_SLICE_ATTR_NEW_AXIS_MASK, 0);
  AttrUtils::SetInt(op_desc_ptr, STRIDE_SLICE_ATTR_SHRINK_AXIS_MASK, 0);

  GeTensorDesc dims_tensor_desc(GeShape({1, 1, 1, 1}), FORMAT_NCHW, DT_FLOAT);
  op_desc_ptr->AddInputDesc(0, dims_tensor_desc);
  op_desc_ptr->AddInputDesc(1, dims_tensor_desc);
  op_desc_ptr->AddInputDesc(2, dims_tensor_desc);
  op_desc_ptr->AddInputDesc(3, dims_tensor_desc);

  vector<int64_t> dims_vec_0;
  vector<int32_t> data_vec_0 = {1, 1, 1, 1};
  GeTensorDesc tensor_desc_0(GeShape(dims_vec_0), FORMAT_NCHW, DT_FLOAT);
  ConstGeTensorPtr tensor_0 =
      std::make_shared<GeTensor>(tensor_desc_0, (uint8_t *)data_vec_0.data(), data_vec_0.size() * sizeof(float));
  ConstGeTensorPtr tensor_1 =
      std::make_shared<GeTensor>(tensor_desc_0, (uint8_t *)data_vec_0.data(), data_vec_0.size() * sizeof(float));
  ConstGeTensorPtr tensor_2 =
      std::make_shared<GeTensor>(tensor_desc_0, (uint8_t *)data_vec_0.data(), data_vec_0.size() * sizeof(float));
  ConstGeTensorPtr tensor_3 =
      std::make_shared<GeTensor>(tensor_desc_0, (uint8_t *)data_vec_0.data(), data_vec_0.size() * sizeof(float));

  vector<ConstGeTensorPtr> input = {tensor_0, tensor_1, tensor_2, tensor_3};
  vector<GeTensorPtr> outputs;

  shared_ptr<Kernel> kernel = KernelFactory::Instance().Create(STRIDEDSLICE);
  EXPECT_NE(kernel->Compute(op_desc_ptr, input, outputs), ge::SUCCESS);
}

TEST_F(UtestGraphPassesFoldingKernelStridedSliceKernel, Test8) {
  OpDescPtr op_desc_ptr = std::make_shared<OpDesc>("StridedSlice", "StridedSlice");
  AttrUtils::SetInt(op_desc_ptr, STRIDE_SLICE_ATTR_BEGIN_MASK, 0);
  AttrUtils::SetInt(op_desc_ptr, STRIDE_SLICE_ATTR_END_MASK, 0);
  AttrUtils::SetInt(op_desc_ptr, STRIDE_SLICE_ATTR_ELLIPSIS_MASK, 0);
  AttrUtils::SetInt(op_desc_ptr, STRIDE_SLICE_ATTR_NEW_AXIS_MASK, 0);
  AttrUtils::SetInt(op_desc_ptr, STRIDE_SLICE_ATTR_SHRINK_AXIS_MASK, 0);
  AttrUtils::SetInt(op_desc_ptr, ATTR_NAME_T, 0);

  GeTensorDesc dims_tensor_desc(GeShape({1, 1, 1, 1}), FORMAT_NCHW, DT_FLOAT);
  op_desc_ptr->AddInputDesc(0, dims_tensor_desc);
  op_desc_ptr->AddInputDesc(1, dims_tensor_desc);
  op_desc_ptr->AddInputDesc(2, dims_tensor_desc);
  op_desc_ptr->AddInputDesc(3, dims_tensor_desc);

  vector<int64_t> dims_vec_0;
  vector<int32_t> data_vec_0 = {1, 1, 1, 1};
  GeTensorDesc tensor_desc_0(GeShape(dims_vec_0), FORMAT_NCHW, DT_FLOAT);
  ConstGeTensorPtr tensor_0 =
      std::make_shared<GeTensor>(tensor_desc_0, (uint8_t *)data_vec_0.data(), data_vec_0.size() * sizeof(float));
  ConstGeTensorPtr tensor_1 =
      std::make_shared<GeTensor>(tensor_desc_0, (uint8_t *)data_vec_0.data(), data_vec_0.size() * sizeof(float));
  ConstGeTensorPtr tensor_2 =
      std::make_shared<GeTensor>(tensor_desc_0, (uint8_t *)data_vec_0.data(), data_vec_0.size() * sizeof(float));
  ConstGeTensorPtr tensor_3 =
      std::make_shared<GeTensor>(tensor_desc_0, (uint8_t *)data_vec_0.data(), data_vec_0.size() * sizeof(float));

  vector<ConstGeTensorPtr> input = {tensor_0, tensor_1, tensor_2, tensor_3};
  vector<GeTensorPtr> outputs;

  shared_ptr<Kernel> kernel = KernelFactory::Instance().Create(STRIDEDSLICE);
  EXPECT_NE(kernel->Compute(op_desc_ptr, input, outputs), ge::SUCCESS);
}

TEST_F(UtestGraphPassesFoldingKernelStridedSliceKernel, Test9) {
  OpDescPtr op_desc_ptr = std::make_shared<OpDesc>("StridedSlice", "StridedSlice");
  AttrUtils::SetInt(op_desc_ptr, STRIDE_SLICE_ATTR_BEGIN_MASK, 0);
  AttrUtils::SetInt(op_desc_ptr, STRIDE_SLICE_ATTR_END_MASK, 0);
  AttrUtils::SetInt(op_desc_ptr, STRIDE_SLICE_ATTR_ELLIPSIS_MASK, 0);
  AttrUtils::SetInt(op_desc_ptr, STRIDE_SLICE_ATTR_NEW_AXIS_MASK, 0);
  AttrUtils::SetInt(op_desc_ptr, STRIDE_SLICE_ATTR_SHRINK_AXIS_MASK, 0);
  AttrUtils::SetInt(op_desc_ptr, ATTR_NAME_T, 0);

  GeTensorDesc dims_tensor_desc(GeShape({1, 1, 1, 1}), FORMAT_NCHW, DT_FLOAT);
  op_desc_ptr->AddInputDesc(0, dims_tensor_desc);
  op_desc_ptr->AddInputDesc(1, dims_tensor_desc);
  op_desc_ptr->AddInputDesc(2, dims_tensor_desc);
  op_desc_ptr->AddInputDesc(3, dims_tensor_desc);

  vector<int64_t> dims_vec_0;
  vector<int32_t> data_vec_0 = {1, 1, 1, 1};
  GeTensorDesc tensor_desc_0(GeShape(dims_vec_0), FORMAT_NCHW, DT_FLOAT);
  ConstGeTensorPtr tensor_0 =
      std::make_shared<GeTensor>(tensor_desc_0, (uint8_t *)data_vec_0.data(), data_vec_0.size() * sizeof(float));
  ConstGeTensorPtr tensor_1 =
      std::make_shared<GeTensor>(tensor_desc_0, (uint8_t *)data_vec_0.data(), data_vec_0.size() * sizeof(float));
  ConstGeTensorPtr tensor_2 =
      std::make_shared<GeTensor>(tensor_desc_0, (uint8_t *)data_vec_0.data(), data_vec_0.size() * sizeof(float));
  ConstGeTensorPtr tensor_3 = nullptr;

  vector<ConstGeTensorPtr> input = {tensor_0, tensor_1, tensor_2, tensor_3};
  vector<GeTensorPtr> outputs;

  shared_ptr<Kernel> kernel = KernelFactory::Instance().Create(STRIDEDSLICE);
  EXPECT_NE(kernel->Compute(op_desc_ptr, input, outputs), ge::SUCCESS);
}

TEST_F(UtestGraphPassesFoldingKernelStridedSliceKernel, Test10) {
  OpDescPtr op_desc_ptr = std::make_shared<OpDesc>("StridedSlice", "StridedSlice");
  AttrUtils::SetInt(op_desc_ptr, STRIDE_SLICE_ATTR_BEGIN_MASK, 0);
  AttrUtils::SetInt(op_desc_ptr, STRIDE_SLICE_ATTR_END_MASK, 0);
  AttrUtils::SetInt(op_desc_ptr, STRIDE_SLICE_ATTR_ELLIPSIS_MASK, 1);
  AttrUtils::SetInt(op_desc_ptr, STRIDE_SLICE_ATTR_NEW_AXIS_MASK, 1);
  AttrUtils::SetInt(op_desc_ptr, STRIDE_SLICE_ATTR_SHRINK_AXIS_MASK, 0);
  AttrUtils::SetInt(op_desc_ptr, ATTR_NAME_T, 0);

  GeTensorDesc dims_tensor_desc(GeShape({1, 1, 1, 1}), FORMAT_NCHW, DT_FLOAT);
  op_desc_ptr->AddInputDesc(0, dims_tensor_desc);
  op_desc_ptr->AddInputDesc(1, dims_tensor_desc);
  op_desc_ptr->AddInputDesc(2, dims_tensor_desc);
  op_desc_ptr->AddInputDesc(3, dims_tensor_desc);

  vector<int64_t> dims_vec_0;
  vector<int32_t> data_vec_0 = {1, 1, 1, 1};
  GeTensorDesc tensor_desc_0(GeShape(dims_vec_0), FORMAT_NCHW, DT_FLOAT);
  ConstGeTensorPtr tensor_0 =
      std::make_shared<GeTensor>(tensor_desc_0, (uint8_t *)data_vec_0.data(), data_vec_0.size() * sizeof(float));
  ConstGeTensorPtr tensor_1 =
      std::make_shared<GeTensor>(tensor_desc_0, (uint8_t *)data_vec_0.data(), data_vec_0.size() * sizeof(float));
  ConstGeTensorPtr tensor_2 =
      std::make_shared<GeTensor>(tensor_desc_0, (uint8_t *)data_vec_0.data(), data_vec_0.size() * sizeof(float));
  ConstGeTensorPtr tensor_3 = nullptr;

  vector<ConstGeTensorPtr> input = {tensor_0, tensor_1, tensor_2, tensor_3};
  vector<GeTensorPtr> outputs;

  shared_ptr<Kernel> kernel = KernelFactory::Instance().Create(STRIDEDSLICE);

  EXPECT_NE(kernel->Compute(op_desc_ptr, input, outputs), ge::SUCCESS);
}

TEST_F(UtestGraphPassesFoldingKernelStridedSliceKernel, Test11) {
  OpDescPtr op_desc_ptr = std::make_shared<OpDesc>("StridedSlice", "StridedSlice");
  AttrUtils::SetInt(op_desc_ptr, STRIDE_SLICE_ATTR_BEGIN_MASK, 0);
  AttrUtils::SetInt(op_desc_ptr, STRIDE_SLICE_ATTR_END_MASK, 0);
  AttrUtils::SetInt(op_desc_ptr, STRIDE_SLICE_ATTR_ELLIPSIS_MASK, 0);
  AttrUtils::SetInt(op_desc_ptr, STRIDE_SLICE_ATTR_NEW_AXIS_MASK, 0);
  AttrUtils::SetInt(op_desc_ptr, STRIDE_SLICE_ATTR_SHRINK_AXIS_MASK, 0);
  AttrUtils::SetInt(op_desc_ptr, ATTR_NAME_T, DT_FLOAT16);

  GeTensorDesc dims_tensor_desc(GeShape({1, 1, 1, 1}), FORMAT_NCHW, DT_FLOAT16);
  op_desc_ptr->AddInputDesc(0, dims_tensor_desc);
  op_desc_ptr->AddInputDesc(1, dims_tensor_desc);
  op_desc_ptr->AddInputDesc(2, dims_tensor_desc);
  op_desc_ptr->AddInputDesc(3, dims_tensor_desc);

  vector<int64_t> dims_vec_0;
  vector<int32_t> data_vec_0 = {1, 1, 1, 1};
  GeTensorDesc tensor_desc_0(GeShape(dims_vec_0), FORMAT_NCHW, DT_FLOAT16);
  ConstGeTensorPtr tensor_0 =
      std::make_shared<GeTensor>(tensor_desc_0, (uint8_t *)data_vec_0.data(), data_vec_0.size() * sizeof(float));
  ConstGeTensorPtr tensor_1 =
      std::make_shared<GeTensor>(tensor_desc_0, (uint8_t *)data_vec_0.data(), data_vec_0.size() * sizeof(float));
  ConstGeTensorPtr tensor_2 =
      std::make_shared<GeTensor>(tensor_desc_0, (uint8_t *)data_vec_0.data(), data_vec_0.size() * sizeof(float));
  ConstGeTensorPtr tensor_3 =
      std::make_shared<GeTensor>(tensor_desc_0, (uint8_t *)data_vec_0.data(), data_vec_0.size() * sizeof(float));

  vector<ConstGeTensorPtr> input = {tensor_0, tensor_1, tensor_2, tensor_3};
  vector<GeTensorPtr> outputs;

  shared_ptr<Kernel> kernel = KernelFactory::Instance().Create(STRIDEDSLICE);
  EXPECT_NE(kernel->Compute(op_desc_ptr, input, outputs), ge::SUCCESS);
}

TEST_F(UtestGraphPassesFoldingKernelStridedSliceKernel, Test12) {
  OpDescPtr op_desc_ptr = std::make_shared<OpDesc>("StridedSlice", "StridedSlice");
  AttrUtils::SetInt(op_desc_ptr, STRIDE_SLICE_ATTR_BEGIN_MASK, 0);
  AttrUtils::SetInt(op_desc_ptr, STRIDE_SLICE_ATTR_END_MASK, 0);
  AttrUtils::SetInt(op_desc_ptr, STRIDE_SLICE_ATTR_ELLIPSIS_MASK, 0);
  AttrUtils::SetInt(op_desc_ptr, STRIDE_SLICE_ATTR_NEW_AXIS_MASK, 0);
  AttrUtils::SetInt(op_desc_ptr, STRIDE_SLICE_ATTR_SHRINK_AXIS_MASK, 1);
  AttrUtils::SetInt(op_desc_ptr, ATTR_NAME_T, 0);

  GeTensorDesc dims_tensor_desc(GeShape({1, 1, 1, 1}), FORMAT_NCHW, DT_INT32);
  op_desc_ptr->AddInputDesc(0, dims_tensor_desc);
  op_desc_ptr->AddInputDesc(1, dims_tensor_desc);
  op_desc_ptr->AddInputDesc(2, dims_tensor_desc);
  op_desc_ptr->AddInputDesc(3, dims_tensor_desc);

  vector<int64_t> dims_vec_0 = {1, 1, 1, 1};
  vector<int32_t> data_vec_0 = {1, 1, 1, 1};
  GeTensorDesc tensor_desc_0(GeShape(dims_vec_0), FORMAT_NCHW, DT_INT32);
  ConstGeTensorPtr tensor_0 =
      std::make_shared<GeTensor>(tensor_desc_0, (uint8_t *)data_vec_0.data(), data_vec_0.size() * sizeof(float));
  ConstGeTensorPtr tensor_1 =
      std::make_shared<GeTensor>(tensor_desc_0, (uint8_t *)data_vec_0.data(), data_vec_0.size() * sizeof(float));
  ConstGeTensorPtr tensor_2 =
      std::make_shared<GeTensor>(tensor_desc_0, (uint8_t *)data_vec_0.data(), data_vec_0.size() * sizeof(float));
  ConstGeTensorPtr tensor_3 =
      std::make_shared<GeTensor>(tensor_desc_0, (uint8_t *)data_vec_0.data(), data_vec_0.size() * sizeof(float));

  vector<ConstGeTensorPtr> input = {tensor_0, tensor_1, tensor_2, tensor_3};
  vector<GeTensorPtr> outputs;

  shared_ptr<Kernel> kernel = KernelFactory::Instance().Create(STRIDEDSLICE);
  EXPECT_NE(kernel->Compute(op_desc_ptr, input, outputs), ge::SUCCESS);
}

TEST_F(UtestGraphPassesFoldingKernelStridedSliceKernel, Test13) {
  OpDescPtr op_desc_ptr = std::make_shared<OpDesc>("StridedSlice", "StridedSlice");
  AttrUtils::SetInt(op_desc_ptr, STRIDE_SLICE_ATTR_BEGIN_MASK, 1);
  AttrUtils::SetInt(op_desc_ptr, STRIDE_SLICE_ATTR_END_MASK, 0);
  AttrUtils::SetInt(op_desc_ptr, STRIDE_SLICE_ATTR_ELLIPSIS_MASK, 0);
  AttrUtils::SetInt(op_desc_ptr, STRIDE_SLICE_ATTR_NEW_AXIS_MASK, 0);
  AttrUtils::SetInt(op_desc_ptr, STRIDE_SLICE_ATTR_SHRINK_AXIS_MASK, 0);
  AttrUtils::SetInt(op_desc_ptr, ATTR_NAME_T, 0);

  GeTensorDesc dims_tensor_desc(GeShape({1, 1, 1, 1}), FORMAT_NCHW, DT_INT32);
  op_desc_ptr->AddInputDesc(0, dims_tensor_desc);
  op_desc_ptr->AddInputDesc(1, dims_tensor_desc);
  op_desc_ptr->AddInputDesc(2, dims_tensor_desc);
  op_desc_ptr->AddInputDesc(3, dims_tensor_desc);

  vector<int64_t> dims_vec_0 = {1, 1, 1, 1};
  vector<int32_t> data_vec_0 = {1, 1, 1, 1};
  GeTensorDesc tensor_desc_0(GeShape(dims_vec_0), FORMAT_NCHW, DT_INT32);
  ConstGeTensorPtr tensor_0 =
      std::make_shared<GeTensor>(tensor_desc_0, (uint8_t *)data_vec_0.data(), data_vec_0.size() * sizeof(float));
  ConstGeTensorPtr tensor_1 =
      std::make_shared<GeTensor>(tensor_desc_0, (uint8_t *)data_vec_0.data(), data_vec_0.size() * sizeof(float));
  ConstGeTensorPtr tensor_2 =
      std::make_shared<GeTensor>(tensor_desc_0, (uint8_t *)data_vec_0.data(), data_vec_0.size() * sizeof(float));
  ConstGeTensorPtr tensor_3 =
      std::make_shared<GeTensor>(tensor_desc_0, (uint8_t *)data_vec_0.data(), data_vec_0.size() * sizeof(float));

  vector<ConstGeTensorPtr> input = {tensor_0, tensor_1, tensor_2, tensor_3};
  vector<GeTensorPtr> outputs;

  shared_ptr<Kernel> kernel = KernelFactory::Instance().Create(STRIDEDSLICE);
  EXPECT_NE(kernel->Compute(op_desc_ptr, input, outputs), ge::SUCCESS);
}

TEST_F(UtestGraphPassesFoldingKernelStridedSliceKernel, Test14) {
  OpDescPtr op_desc_ptr = std::make_shared<OpDesc>("StridedSlice", "StridedSlice");
  AttrUtils::SetInt(op_desc_ptr, STRIDE_SLICE_ATTR_BEGIN_MASK, 0);
  AttrUtils::SetInt(op_desc_ptr, STRIDE_SLICE_ATTR_END_MASK, 1);
  AttrUtils::SetInt(op_desc_ptr, STRIDE_SLICE_ATTR_ELLIPSIS_MASK, 0);
  AttrUtils::SetInt(op_desc_ptr, STRIDE_SLICE_ATTR_NEW_AXIS_MASK, 0);
  AttrUtils::SetInt(op_desc_ptr, STRIDE_SLICE_ATTR_SHRINK_AXIS_MASK, 0);
  AttrUtils::SetInt(op_desc_ptr, ATTR_NAME_T, 0);

  GeTensorDesc dims_tensor_desc(GeShape({1, 1, 1, 1}), FORMAT_NCHW, DT_INT32);
  op_desc_ptr->AddInputDesc(0, dims_tensor_desc);
  op_desc_ptr->AddInputDesc(1, dims_tensor_desc);
  op_desc_ptr->AddInputDesc(2, dims_tensor_desc);
  op_desc_ptr->AddInputDesc(3, dims_tensor_desc);

  vector<int64_t> dims_vec_0 = {1, 1, 1, 1};
  vector<int32_t> data_vec_0 = {1, 1, 1, 1};
  GeTensorDesc tensor_desc_0(GeShape(dims_vec_0), FORMAT_NCHW, DT_INT32);
  ConstGeTensorPtr tensor_0 =
      std::make_shared<GeTensor>(tensor_desc_0, (uint8_t *)data_vec_0.data(), data_vec_0.size() * sizeof(float));
  ConstGeTensorPtr tensor_1 =
      std::make_shared<GeTensor>(tensor_desc_0, (uint8_t *)data_vec_0.data(), data_vec_0.size() * sizeof(float));
  ConstGeTensorPtr tensor_2 =
      std::make_shared<GeTensor>(tensor_desc_0, (uint8_t *)data_vec_0.data(), data_vec_0.size() * sizeof(float));
  ConstGeTensorPtr tensor_3 =
      std::make_shared<GeTensor>(tensor_desc_0, (uint8_t *)data_vec_0.data(), data_vec_0.size() * sizeof(float));

  vector<ConstGeTensorPtr> input = {tensor_0, tensor_1, tensor_2, tensor_3};
  vector<GeTensorPtr> outputs;

  shared_ptr<Kernel> kernel = KernelFactory::Instance().Create(STRIDEDSLICE);
  EXPECT_NE(kernel->Compute(op_desc_ptr, input, outputs), ge::SUCCESS);
}

TEST_F(UtestGraphPassesFoldingKernelStridedSliceKernel, Test15) {
  OpDescPtr op_desc_ptr = std::make_shared<OpDesc>("StridedSlice", "StridedSlice");
  AttrUtils::SetInt(op_desc_ptr, STRIDE_SLICE_ATTR_BEGIN_MASK, 0);
  AttrUtils::SetInt(op_desc_ptr, STRIDE_SLICE_ATTR_END_MASK, 0);
  AttrUtils::SetInt(op_desc_ptr, STRIDE_SLICE_ATTR_ELLIPSIS_MASK, 0);
  AttrUtils::SetInt(op_desc_ptr, STRIDE_SLICE_ATTR_NEW_AXIS_MASK, 0);
  AttrUtils::SetInt(op_desc_ptr, STRIDE_SLICE_ATTR_SHRINK_AXIS_MASK, 0);
  AttrUtils::SetInt(op_desc_ptr, ATTR_NAME_T, 0);

  GeTensorDesc dims_tensor_desc(GeShape({1, 1, 1, 1}), FORMAT_NCHW, DT_INT32);
  op_desc_ptr->AddInputDesc(0, dims_tensor_desc);
  op_desc_ptr->AddInputDesc(1, dims_tensor_desc);
  op_desc_ptr->AddInputDesc(2, dims_tensor_desc);
  op_desc_ptr->AddInputDesc(3, dims_tensor_desc);

  vector<int64_t> dims_vec_0;
  vector<int32_t> data_vec_0 = {1, 1, 1, 1};
  GeTensorDesc tensor_desc_0(GeShape(dims_vec_0), FORMAT_NCHW, DT_INT32);
  ConstGeTensorPtr tensor_0 =
      std::make_shared<GeTensor>(tensor_desc_0, (uint8_t *)data_vec_0.data(), data_vec_0.size() * sizeof(float));
  ConstGeTensorPtr tensor_1 =
      std::make_shared<GeTensor>(tensor_desc_0, (uint8_t *)data_vec_0.data(), data_vec_0.size() * sizeof(float));
  ConstGeTensorPtr tensor_2 =
      std::make_shared<GeTensor>(tensor_desc_0, (uint8_t *)data_vec_0.data(), data_vec_0.size() * sizeof(float));
  ConstGeTensorPtr tensor_3 =
      std::make_shared<GeTensor>(tensor_desc_0, (uint8_t *)data_vec_0.data(), data_vec_0.size() * sizeof(float));

  vector<ConstGeTensorPtr> input = {tensor_0, tensor_1, tensor_2, tensor_3};
  vector<GeTensorPtr> outputs;

  shared_ptr<Kernel> kernel = KernelFactory::Instance().Create(STRIDEDSLICE);
  EXPECT_EQ(kernel->Compute(op_desc_ptr, input, outputs), ge::SUCCESS);
}

TEST_F(UtestGraphPassesFoldingKernelStridedSliceKernel, Test16) {
  OpDescPtr op_desc_ptr = std::make_shared<OpDesc>("StridedSlice", "StridedSlice");
  AttrUtils::SetInt(op_desc_ptr, STRIDE_SLICE_ATTR_BEGIN_MASK, 0);
  AttrUtils::SetInt(op_desc_ptr, STRIDE_SLICE_ATTR_END_MASK, 0);
  AttrUtils::SetInt(op_desc_ptr, STRIDE_SLICE_ATTR_ELLIPSIS_MASK, 0);
  AttrUtils::SetInt(op_desc_ptr, STRIDE_SLICE_ATTR_NEW_AXIS_MASK, 0);
  AttrUtils::SetInt(op_desc_ptr, STRIDE_SLICE_ATTR_SHRINK_AXIS_MASK, 0);
  AttrUtils::SetInt(op_desc_ptr, ATTR_NAME_T, 0);

  GeTensorDesc dims_tensor_desc(GeShape({1, 1, 1, 1}), FORMAT_NCHW, DT_INT32);
  op_desc_ptr->AddInputDesc(0, dims_tensor_desc);
  op_desc_ptr->AddInputDesc(1, dims_tensor_desc);
  op_desc_ptr->AddInputDesc(2, dims_tensor_desc);
  op_desc_ptr->AddInputDesc(3, dims_tensor_desc);

  vector<int64_t> dims_vec_0 = {1, 1, 1, 1};
  vector<int32_t> data_vec_0 = {1, 1, 1, 1};
  GeTensorDesc tensor_desc_0(GeShape(dims_vec_0), FORMAT_NCHW, DT_INT32);
  ConstGeTensorPtr tensor_0 =
      std::make_shared<GeTensor>(tensor_desc_0, (uint8_t *)data_vec_0.data(), data_vec_0.size() * sizeof(float));
  ConstGeTensorPtr tensor_1 =
      std::make_shared<GeTensor>(tensor_desc_0, (uint8_t *)data_vec_0.data(), data_vec_0.size() * sizeof(float));
  ConstGeTensorPtr tensor_2 =
      std::make_shared<GeTensor>(tensor_desc_0, (uint8_t *)data_vec_0.data(), data_vec_0.size() * sizeof(float));
  ConstGeTensorPtr tensor_3 =
      std::make_shared<GeTensor>(tensor_desc_0, (uint8_t *)data_vec_0.data(), data_vec_0.size() * sizeof(float));

  vector<ConstGeTensorPtr> input = {tensor_0, tensor_1, tensor_2, tensor_3};
  vector<GeTensorPtr> outputs;

  shared_ptr<Kernel> kernel = KernelFactory::Instance().Create(STRIDEDSLICE);
  EXPECT_NE(kernel->Compute(op_desc_ptr, input, outputs), ge::SUCCESS);
}

TEST_F(UtestGraphPassesFoldingKernelStridedSliceKernel, Test17) {
  OpDescPtr op_desc_ptr = std::make_shared<OpDesc>("StridedSlice", "StridedSlice");
  AttrUtils::SetInt(op_desc_ptr, STRIDE_SLICE_ATTR_BEGIN_MASK, 0);
  AttrUtils::SetInt(op_desc_ptr, STRIDE_SLICE_ATTR_END_MASK, 0);
  AttrUtils::SetInt(op_desc_ptr, STRIDE_SLICE_ATTR_ELLIPSIS_MASK, 0);
  AttrUtils::SetInt(op_desc_ptr, STRIDE_SLICE_ATTR_NEW_AXIS_MASK, 0);
  AttrUtils::SetInt(op_desc_ptr, STRIDE_SLICE_ATTR_SHRINK_AXIS_MASK, 0);
  AttrUtils::SetInt(op_desc_ptr, ATTR_NAME_T, 0);

  GeTensorDesc dims_tensor_desc(GeShape({1, 1, 1, 1}), FORMAT_NCHW, DT_INT32);
  op_desc_ptr->AddInputDesc(0, dims_tensor_desc);
  op_desc_ptr->AddInputDesc(1, dims_tensor_desc);
  op_desc_ptr->AddInputDesc(2, dims_tensor_desc);
  op_desc_ptr->AddInputDesc(3, dims_tensor_desc);

  vector<int64_t> dims_vec_0 = {10, 10, 10, 10};
  vector<int32_t> data_vec_0 = {3, 3, 3, 3};
  GeTensorDesc tensor_desc_0(GeShape(dims_vec_0), FORMAT_NCHW, DT_INT32);
  ConstGeTensorPtr tensor_0 =
      std::make_shared<GeTensor>(tensor_desc_0, (uint8_t *)data_vec_0.data(), data_vec_0.size() * sizeof(float));
  ConstGeTensorPtr tensor_1 =
      std::make_shared<GeTensor>(tensor_desc_0, (uint8_t *)data_vec_0.data(), data_vec_0.size() * sizeof(float));
  ConstGeTensorPtr tensor_2 =
      std::make_shared<GeTensor>(tensor_desc_0, (uint8_t *)data_vec_0.data(), data_vec_0.size() * sizeof(float));
  ConstGeTensorPtr tensor_3 =
      std::make_shared<GeTensor>(tensor_desc_0, (uint8_t *)data_vec_0.data(), data_vec_0.size() * sizeof(float));

  vector<ConstGeTensorPtr> input = {tensor_0, tensor_1, tensor_2, tensor_3};
  vector<GeTensorPtr> outputs;

  shared_ptr<Kernel> kernel = KernelFactory::Instance().Create(STRIDEDSLICE);
  EXPECT_NE(kernel->Compute(op_desc_ptr, input, outputs), ge::SUCCESS);
}

TEST_F(UtestGraphPassesFoldingKernelStridedSliceKernel, CovSuccessInt64Stride) {
  OpDescPtr op_desc_ptr = std::make_shared<OpDesc>("StridedSlice", "StridedSlice");
  AttrUtils::SetInt(op_desc_ptr, STRIDE_SLICE_ATTR_BEGIN_MASK, 0);
  AttrUtils::SetInt(op_desc_ptr, STRIDE_SLICE_ATTR_END_MASK, 0);
  AttrUtils::SetInt(op_desc_ptr, STRIDE_SLICE_ATTR_ELLIPSIS_MASK, 0);
  AttrUtils::SetInt(op_desc_ptr, STRIDE_SLICE_ATTR_NEW_AXIS_MASK, 0);
  AttrUtils::SetInt(op_desc_ptr, STRIDE_SLICE_ATTR_SHRINK_AXIS_MASK, 0);
  AttrUtils::SetInt(op_desc_ptr, ATTR_NAME_T, 0);

  GeTensorDesc dims_tensor_desc(GeShape({2, 3}), FORMAT_NCHW, DT_INT32);
  op_desc_ptr->AddInputDesc(0, dims_tensor_desc);
  op_desc_ptr->AddInputDesc(1, dims_tensor_desc);
  op_desc_ptr->AddInputDesc(2, dims_tensor_desc);
  op_desc_ptr->AddInputDesc(3, dims_tensor_desc);
  op_desc_ptr->AddOutputDesc(dims_tensor_desc);

  vector<int32_t> x_data = {1, 2, 3, 4, 5, 6};
  GeTensorDesc x_desc(GeShape({2, 3}), FORMAT_NCHW, DT_INT32);
  ConstGeTensorPtr tensor_0 =
      std::make_shared<GeTensor>(x_desc, (uint8_t *)x_data.data(), x_data.size() * sizeof(int32_t));

  vector<int64_t> begin_data = {0, 0};
  vector<int64_t> end_data = {2, 2};
  vector<int64_t> stride_data = {1, 1};
  GeTensorDesc idx_desc(GeShape({2}), FORMAT_NCHW, DT_INT64);
  ConstGeTensorPtr tensor_1 =
      std::make_shared<GeTensor>(idx_desc, (uint8_t *)begin_data.data(), begin_data.size() * sizeof(int64_t));
  ConstGeTensorPtr tensor_2 =
      std::make_shared<GeTensor>(idx_desc, (uint8_t *)end_data.data(), end_data.size() * sizeof(int64_t));
  ConstGeTensorPtr tensor_3 =
      std::make_shared<GeTensor>(idx_desc, (uint8_t *)stride_data.data(), stride_data.size() * sizeof(int64_t));

  vector<ConstGeTensorPtr> input = {tensor_0, tensor_1, tensor_2, tensor_3};
  vector<GeTensorPtr> outputs;

  shared_ptr<Kernel> kernel = KernelFactory::Instance().Create(STRIDEDSLICE);
  EXPECT_EQ(kernel->Compute(op_desc_ptr, input, outputs), ge::SUCCESS);
}

TEST_F(UtestGraphPassesFoldingKernelStridedSliceKernel, CovSuccessInt32Stride) {
  OpDescPtr op_desc_ptr = std::make_shared<OpDesc>("StridedSlice", "StridedSlice");
  AttrUtils::SetInt(op_desc_ptr, STRIDE_SLICE_ATTR_BEGIN_MASK, 0);
  AttrUtils::SetInt(op_desc_ptr, STRIDE_SLICE_ATTR_END_MASK, 0);
  AttrUtils::SetInt(op_desc_ptr, STRIDE_SLICE_ATTR_ELLIPSIS_MASK, 0);
  AttrUtils::SetInt(op_desc_ptr, STRIDE_SLICE_ATTR_NEW_AXIS_MASK, 0);
  AttrUtils::SetInt(op_desc_ptr, STRIDE_SLICE_ATTR_SHRINK_AXIS_MASK, 0);
  AttrUtils::SetInt(op_desc_ptr, ATTR_NAME_T, 0);

  GeTensorDesc dims_tensor_desc(GeShape({2, 3}), FORMAT_NCHW, DT_INT32);
  op_desc_ptr->AddInputDesc(0, dims_tensor_desc);
  op_desc_ptr->AddInputDesc(1, dims_tensor_desc);
  op_desc_ptr->AddInputDesc(2, dims_tensor_desc);
  op_desc_ptr->AddInputDesc(3, dims_tensor_desc);
  op_desc_ptr->AddOutputDesc(dims_tensor_desc);

  vector<int32_t> x_data = {1, 2, 3, 4, 5, 6};
  GeTensorDesc x_desc(GeShape({2, 3}), FORMAT_NCHW, DT_INT32);
  ConstGeTensorPtr tensor_0 =
      std::make_shared<GeTensor>(x_desc, (uint8_t *)x_data.data(), x_data.size() * sizeof(int32_t));

  vector<int32_t> begin_data = {0, 0};
  vector<int32_t> end_data = {2, 2};
  vector<int32_t> stride_data = {1, 1};
  GeTensorDesc idx_desc(GeShape({2}), FORMAT_NCHW, DT_INT32);
  ConstGeTensorPtr tensor_1 =
      std::make_shared<GeTensor>(idx_desc, (uint8_t *)begin_data.data(), begin_data.size() * sizeof(int32_t));
  ConstGeTensorPtr tensor_2 =
      std::make_shared<GeTensor>(idx_desc, (uint8_t *)end_data.data(), end_data.size() * sizeof(int32_t));
  ConstGeTensorPtr tensor_3 =
      std::make_shared<GeTensor>(idx_desc, (uint8_t *)stride_data.data(), stride_data.size() * sizeof(int32_t));

  vector<ConstGeTensorPtr> input = {tensor_0, tensor_1, tensor_2, tensor_3};
  vector<GeTensorPtr> outputs;

  shared_ptr<Kernel> kernel = KernelFactory::Instance().Create(STRIDEDSLICE);
  EXPECT_EQ(kernel->Compute(op_desc_ptr, input, outputs), ge::SUCCESS);
}

TEST_F(UtestGraphPassesFoldingKernelStridedSliceKernel, CovNullOpDesc) {
  OpDescPtr op_desc_ptr = nullptr;

  vector<int32_t> data_vec = {1, 2, 3, 4};
  GeTensorDesc tensor_desc(GeShape({2, 2}), FORMAT_NCHW, DT_INT32);
  ConstGeTensorPtr tensor_0 =
      std::make_shared<GeTensor>(tensor_desc, (uint8_t *)data_vec.data(), data_vec.size() * sizeof(int32_t));
  ConstGeTensorPtr tensor_1 =
      std::make_shared<GeTensor>(tensor_desc, (uint8_t *)data_vec.data(), data_vec.size() * sizeof(int32_t));
  ConstGeTensorPtr tensor_2 =
      std::make_shared<GeTensor>(tensor_desc, (uint8_t *)data_vec.data(), data_vec.size() * sizeof(int32_t));
  ConstGeTensorPtr tensor_3 =
      std::make_shared<GeTensor>(tensor_desc, (uint8_t *)data_vec.data(), data_vec.size() * sizeof(int32_t));

  vector<ConstGeTensorPtr> input = {tensor_0, tensor_1, tensor_2, tensor_3};
  vector<GeTensorPtr> outputs;

  shared_ptr<Kernel> kernel = KernelFactory::Instance().Create(STRIDEDSLICE);
  EXPECT_NE(kernel->Compute(op_desc_ptr, input, outputs), ge::SUCCESS);
}

TEST_F(UtestGraphPassesFoldingKernelStridedSliceKernel, CovMissingAttr) {
  OpDescPtr op_desc_ptr = std::make_shared<OpDesc>("StridedSlice", "StridedSlice");

  GeTensorDesc dims_tensor_desc(GeShape({2, 3}), FORMAT_NCHW, DT_INT32);
  op_desc_ptr->AddInputDesc(0, dims_tensor_desc);
  op_desc_ptr->AddInputDesc(1, dims_tensor_desc);
  op_desc_ptr->AddInputDesc(2, dims_tensor_desc);
  op_desc_ptr->AddInputDesc(3, dims_tensor_desc);
  op_desc_ptr->AddOutputDesc(dims_tensor_desc);

  vector<int32_t> data_vec = {1, 2, 3, 4, 5, 6};
  GeTensorDesc tensor_desc(GeShape({2, 3}), FORMAT_NCHW, DT_INT32);
  ConstGeTensorPtr tensor_0 =
      std::make_shared<GeTensor>(tensor_desc, (uint8_t *)data_vec.data(), data_vec.size() * sizeof(int32_t));
  ConstGeTensorPtr tensor_1 =
      std::make_shared<GeTensor>(tensor_desc, (uint8_t *)data_vec.data(), data_vec.size() * sizeof(int32_t));
  ConstGeTensorPtr tensor_2 =
      std::make_shared<GeTensor>(tensor_desc, (uint8_t *)data_vec.data(), data_vec.size() * sizeof(int32_t));
  ConstGeTensorPtr tensor_3 =
      std::make_shared<GeTensor>(tensor_desc, (uint8_t *)data_vec.data(), data_vec.size() * sizeof(int32_t));

  vector<ConstGeTensorPtr> input = {tensor_0, tensor_1, tensor_2, tensor_3};
  vector<GeTensorPtr> outputs;

  shared_ptr<Kernel> kernel = KernelFactory::Instance().Create(STRIDEDSLICE);
  EXPECT_NE(kernel->Compute(op_desc_ptr, input, outputs), ge::SUCCESS);
}

TEST_F(UtestGraphPassesFoldingKernelStridedSliceKernel, CovInputSizeMismatch) {
  OpDescPtr op_desc_ptr = std::make_shared<OpDesc>("StridedSlice", "StridedSlice");
  AttrUtils::SetInt(op_desc_ptr, STRIDE_SLICE_ATTR_BEGIN_MASK, 0);
  AttrUtils::SetInt(op_desc_ptr, STRIDE_SLICE_ATTR_END_MASK, 0);
  AttrUtils::SetInt(op_desc_ptr, STRIDE_SLICE_ATTR_ELLIPSIS_MASK, 0);
  AttrUtils::SetInt(op_desc_ptr, STRIDE_SLICE_ATTR_NEW_AXIS_MASK, 0);
  AttrUtils::SetInt(op_desc_ptr, STRIDE_SLICE_ATTR_SHRINK_AXIS_MASK, 0);
  AttrUtils::SetInt(op_desc_ptr, ATTR_NAME_T, 0);

  GeTensorDesc dims_tensor_desc(GeShape({2, 3}), FORMAT_NCHW, DT_INT32);
  op_desc_ptr->AddInputDesc(0, dims_tensor_desc);
  op_desc_ptr->AddInputDesc(1, dims_tensor_desc);
  op_desc_ptr->AddInputDesc(2, dims_tensor_desc);
  op_desc_ptr->AddInputDesc(3, dims_tensor_desc);
  op_desc_ptr->AddOutputDesc(dims_tensor_desc);

  vector<int32_t> data_vec = {1, 2, 3, 4, 5, 6};
  GeTensorDesc tensor_desc(GeShape({2, 3}), FORMAT_NCHW, DT_INT32);
  ConstGeTensorPtr tensor_0 =
      std::make_shared<GeTensor>(tensor_desc, (uint8_t *)data_vec.data(), data_vec.size() * sizeof(int32_t));
  ConstGeTensorPtr tensor_1 =
      std::make_shared<GeTensor>(tensor_desc, (uint8_t *)data_vec.data(), data_vec.size() * sizeof(int32_t));
  ConstGeTensorPtr tensor_2 =
      std::make_shared<GeTensor>(tensor_desc, (uint8_t *)data_vec.data(), data_vec.size() * sizeof(int32_t));

  vector<ConstGeTensorPtr> input = {tensor_0, tensor_1, tensor_2};
  vector<GeTensorPtr> outputs;

  shared_ptr<Kernel> kernel = KernelFactory::Instance().Create(STRIDEDSLICE);
  EXPECT_NE(kernel->Compute(op_desc_ptr, input, outputs), ge::SUCCESS);
}

TEST_F(UtestGraphPassesFoldingKernelStridedSliceKernel, CovDataTypeMismatch) {
  OpDescPtr op_desc_ptr = std::make_shared<OpDesc>("StridedSlice", "StridedSlice");
  AttrUtils::SetInt(op_desc_ptr, STRIDE_SLICE_ATTR_BEGIN_MASK, 0);
  AttrUtils::SetInt(op_desc_ptr, STRIDE_SLICE_ATTR_END_MASK, 0);
  AttrUtils::SetInt(op_desc_ptr, STRIDE_SLICE_ATTR_ELLIPSIS_MASK, 0);
  AttrUtils::SetInt(op_desc_ptr, STRIDE_SLICE_ATTR_NEW_AXIS_MASK, 0);
  AttrUtils::SetInt(op_desc_ptr, STRIDE_SLICE_ATTR_SHRINK_AXIS_MASK, 0);
  AttrUtils::SetInt(op_desc_ptr, ATTR_NAME_T, 0);

  GeTensorDesc dims_tensor_desc(GeShape({2, 3}), FORMAT_NCHW, DT_INT32);
  op_desc_ptr->AddInputDesc(0, dims_tensor_desc);
  op_desc_ptr->AddInputDesc(1, dims_tensor_desc);
  op_desc_ptr->AddInputDesc(2, dims_tensor_desc);
  op_desc_ptr->AddInputDesc(3, dims_tensor_desc);
  op_desc_ptr->AddOutputDesc(dims_tensor_desc);

  vector<int32_t> x_data = {1, 2, 3, 4, 5, 6};
  GeTensorDesc x_desc(GeShape({2, 3}), FORMAT_NCHW, DT_INT32);
  ConstGeTensorPtr tensor_0 =
      std::make_shared<GeTensor>(x_desc, (uint8_t *)x_data.data(), x_data.size() * sizeof(int32_t));

  vector<int32_t> begin_data = {0, 0};
  vector<int64_t> end_data = {2, 2};
  vector<int32_t> stride_data = {1, 1};
  GeTensorDesc begin_desc(GeShape({2}), FORMAT_NCHW, DT_INT32);
  GeTensorDesc end_desc(GeShape({2}), FORMAT_NCHW, DT_INT64);
  ConstGeTensorPtr tensor_1 =
      std::make_shared<GeTensor>(begin_desc, (uint8_t *)begin_data.data(), begin_data.size() * sizeof(int32_t));
  ConstGeTensorPtr tensor_2 =
      std::make_shared<GeTensor>(end_desc, (uint8_t *)end_data.data(), end_data.size() * sizeof(int64_t));
  ConstGeTensorPtr tensor_3 =
      std::make_shared<GeTensor>(begin_desc, (uint8_t *)stride_data.data(), stride_data.size() * sizeof(int32_t));

  vector<ConstGeTensorPtr> input = {tensor_0, tensor_1, tensor_2, tensor_3};
  vector<GeTensorPtr> outputs;

  shared_ptr<Kernel> kernel = KernelFactory::Instance().Create(STRIDEDSLICE);
  EXPECT_NE(kernel->Compute(op_desc_ptr, input, outputs), ge::SUCCESS);
}

TEST_F(UtestGraphPassesFoldingKernelStridedSliceKernel, CovUnsupportedXType) {
  OpDescPtr op_desc_ptr = std::make_shared<OpDesc>("StridedSlice", "StridedSlice");
  AttrUtils::SetInt(op_desc_ptr, STRIDE_SLICE_ATTR_BEGIN_MASK, 0);
  AttrUtils::SetInt(op_desc_ptr, STRIDE_SLICE_ATTR_END_MASK, 0);
  AttrUtils::SetInt(op_desc_ptr, STRIDE_SLICE_ATTR_ELLIPSIS_MASK, 0);
  AttrUtils::SetInt(op_desc_ptr, STRIDE_SLICE_ATTR_NEW_AXIS_MASK, 0);
  AttrUtils::SetInt(op_desc_ptr, STRIDE_SLICE_ATTR_SHRINK_AXIS_MASK, 0);
  AttrUtils::SetInt(op_desc_ptr, ATTR_NAME_T, 0);

  GeTensorDesc dims_tensor_desc(GeShape({2, 2}), FORMAT_NCHW, DT_INT32);
  op_desc_ptr->AddInputDesc(0, dims_tensor_desc);
  op_desc_ptr->AddInputDesc(1, dims_tensor_desc);
  op_desc_ptr->AddInputDesc(2, dims_tensor_desc);
  op_desc_ptr->AddInputDesc(3, dims_tensor_desc);
  op_desc_ptr->AddOutputDesc(dims_tensor_desc);

  vector<int32_t> x_data = {1, 2, 3, 4};
  GeTensorDesc x_desc(GeShape({2, 2}), FORMAT_NCHW, DT_STRING);
  ConstGeTensorPtr tensor_0 =
      std::make_shared<GeTensor>(x_desc, (uint8_t *)x_data.data(), x_data.size() * sizeof(int32_t));

  vector<int32_t> begin_data = {0, 0};
  vector<int32_t> end_data = {2, 2};
  vector<int32_t> stride_data = {1, 1};
  GeTensorDesc idx_desc(GeShape({2}), FORMAT_NCHW, DT_INT32);
  ConstGeTensorPtr tensor_1 =
      std::make_shared<GeTensor>(idx_desc, (uint8_t *)begin_data.data(), begin_data.size() * sizeof(int32_t));
  ConstGeTensorPtr tensor_2 =
      std::make_shared<GeTensor>(idx_desc, (uint8_t *)end_data.data(), end_data.size() * sizeof(int32_t));
  ConstGeTensorPtr tensor_3 =
      std::make_shared<GeTensor>(idx_desc, (uint8_t *)stride_data.data(), stride_data.size() * sizeof(int32_t));

  vector<ConstGeTensorPtr> input = {tensor_0, tensor_1, tensor_2, tensor_3};
  vector<GeTensorPtr> outputs;

  shared_ptr<Kernel> kernel = KernelFactory::Instance().Create(STRIDEDSLICE);
  EXPECT_NE(kernel->Compute(op_desc_ptr, input, outputs), ge::SUCCESS);
}

TEST_F(UtestGraphPassesFoldingKernelStridedSliceKernel, CovZeroDataSize) {
  OpDescPtr op_desc_ptr = std::make_shared<OpDesc>("StridedSlice", "StridedSlice");
  AttrUtils::SetInt(op_desc_ptr, STRIDE_SLICE_ATTR_BEGIN_MASK, 0);
  AttrUtils::SetInt(op_desc_ptr, STRIDE_SLICE_ATTR_END_MASK, 0);
  AttrUtils::SetInt(op_desc_ptr, STRIDE_SLICE_ATTR_ELLIPSIS_MASK, 0);
  AttrUtils::SetInt(op_desc_ptr, STRIDE_SLICE_ATTR_NEW_AXIS_MASK, 0);
  AttrUtils::SetInt(op_desc_ptr, STRIDE_SLICE_ATTR_SHRINK_AXIS_MASK, 0);
  AttrUtils::SetInt(op_desc_ptr, ATTR_NAME_T, 0);

  GeTensorDesc dims_tensor_desc(GeShape({2, 3}), FORMAT_NCHW, DT_INT32);
  op_desc_ptr->AddInputDesc(0, dims_tensor_desc);
  op_desc_ptr->AddInputDesc(1, dims_tensor_desc);
  op_desc_ptr->AddInputDesc(2, dims_tensor_desc);
  op_desc_ptr->AddInputDesc(3, dims_tensor_desc);
  op_desc_ptr->AddOutputDesc(dims_tensor_desc);

  GeTensorDesc tensor_desc(GeShape({2, 3}), FORMAT_NCHW, DT_INT32);
  ConstGeTensorPtr tensor_0 = std::make_shared<GeTensor>(tensor_desc);
  ConstGeTensorPtr tensor_1 = std::make_shared<GeTensor>(tensor_desc);
  ConstGeTensorPtr tensor_2 = std::make_shared<GeTensor>(tensor_desc);
  ConstGeTensorPtr tensor_3 = std::make_shared<GeTensor>(tensor_desc);

  vector<ConstGeTensorPtr> input = {tensor_0, tensor_1, tensor_2, tensor_3};
  vector<GeTensorPtr> outputs;

  shared_ptr<Kernel> kernel = KernelFactory::Instance().Create(STRIDEDSLICE);
  EXPECT_NE(kernel->Compute(op_desc_ptr, input, outputs), ge::SUCCESS);
}

TEST_F(UtestGraphPassesFoldingKernelStridedSliceKernel, CovSizeMismatch) {
  OpDescPtr op_desc_ptr = std::make_shared<OpDesc>("StridedSlice", "StridedSlice");
  AttrUtils::SetInt(op_desc_ptr, STRIDE_SLICE_ATTR_BEGIN_MASK, 0);
  AttrUtils::SetInt(op_desc_ptr, STRIDE_SLICE_ATTR_END_MASK, 0);
  AttrUtils::SetInt(op_desc_ptr, STRIDE_SLICE_ATTR_ELLIPSIS_MASK, 0);
  AttrUtils::SetInt(op_desc_ptr, STRIDE_SLICE_ATTR_NEW_AXIS_MASK, 0);
  AttrUtils::SetInt(op_desc_ptr, STRIDE_SLICE_ATTR_SHRINK_AXIS_MASK, 0);
  AttrUtils::SetInt(op_desc_ptr, ATTR_NAME_T, 0);

  GeTensorDesc dims_tensor_desc(GeShape({2, 3}), FORMAT_NCHW, DT_INT32);
  op_desc_ptr->AddInputDesc(0, dims_tensor_desc);
  op_desc_ptr->AddInputDesc(1, dims_tensor_desc);
  op_desc_ptr->AddInputDesc(2, dims_tensor_desc);
  op_desc_ptr->AddInputDesc(3, dims_tensor_desc);
  op_desc_ptr->AddOutputDesc(dims_tensor_desc);

  vector<int32_t> x_data = {1, 2, 3, 4, 5, 6};
  GeTensorDesc x_desc(GeShape({2, 3}), FORMAT_NCHW, DT_INT32);
  ConstGeTensorPtr tensor_0 =
      std::make_shared<GeTensor>(x_desc, (uint8_t *)x_data.data(), x_data.size() * sizeof(int32_t));

  vector<int32_t> begin_data = {0};
  vector<int32_t> end_data = {2, 2};
  vector<int32_t> stride_data = {1};
  GeTensorDesc idx_desc(GeShape({2}), FORMAT_NCHW, DT_INT32);
  ConstGeTensorPtr tensor_1 =
      std::make_shared<GeTensor>(idx_desc, (uint8_t *)begin_data.data(), begin_data.size() * sizeof(int32_t));
  ConstGeTensorPtr tensor_2 =
      std::make_shared<GeTensor>(idx_desc, (uint8_t *)end_data.data(), end_data.size() * sizeof(int32_t));
  ConstGeTensorPtr tensor_3 =
      std::make_shared<GeTensor>(idx_desc, (uint8_t *)stride_data.data(), stride_data.size() * sizeof(int32_t));

  vector<ConstGeTensorPtr> input = {tensor_0, tensor_1, tensor_2, tensor_3};
  vector<GeTensorPtr> outputs;

  shared_ptr<Kernel> kernel = KernelFactory::Instance().Create(STRIDEDSLICE);
  EXPECT_NE(kernel->Compute(op_desc_ptr, input, outputs), ge::SUCCESS);
}

TEST_F(UtestGraphPassesFoldingKernelStridedSliceKernel, CovInvalidEllipsisMask) {
  OpDescPtr op_desc_ptr = std::make_shared<OpDesc>("StridedSlice", "StridedSlice");
  AttrUtils::SetInt(op_desc_ptr, STRIDE_SLICE_ATTR_BEGIN_MASK, 0);
  AttrUtils::SetInt(op_desc_ptr, STRIDE_SLICE_ATTR_END_MASK, 0);
  AttrUtils::SetInt(op_desc_ptr, STRIDE_SLICE_ATTR_ELLIPSIS_MASK, 3);
  AttrUtils::SetInt(op_desc_ptr, STRIDE_SLICE_ATTR_NEW_AXIS_MASK, 0);
  AttrUtils::SetInt(op_desc_ptr, STRIDE_SLICE_ATTR_SHRINK_AXIS_MASK, 0);
  AttrUtils::SetInt(op_desc_ptr, ATTR_NAME_T, 0);

  GeTensorDesc dims_tensor_desc(GeShape({2, 2}), FORMAT_NCHW, DT_INT32);
  op_desc_ptr->AddInputDesc(0, dims_tensor_desc);
  op_desc_ptr->AddInputDesc(1, dims_tensor_desc);
  op_desc_ptr->AddInputDesc(2, dims_tensor_desc);
  op_desc_ptr->AddInputDesc(3, dims_tensor_desc);
  op_desc_ptr->AddOutputDesc(dims_tensor_desc);

  vector<int32_t> data_vec = {1, 2, 3, 4};
  GeTensorDesc tensor_desc(GeShape({2, 2}), FORMAT_NCHW, DT_INT32);
  ConstGeTensorPtr tensor_0 =
      std::make_shared<GeTensor>(tensor_desc, (uint8_t *)data_vec.data(), data_vec.size() * sizeof(int32_t));
  ConstGeTensorPtr tensor_1 =
      std::make_shared<GeTensor>(tensor_desc, (uint8_t *)data_vec.data(), data_vec.size() * sizeof(int32_t));
  ConstGeTensorPtr tensor_2 =
      std::make_shared<GeTensor>(tensor_desc, (uint8_t *)data_vec.data(), data_vec.size() * sizeof(int32_t));
  ConstGeTensorPtr tensor_3 =
      std::make_shared<GeTensor>(tensor_desc, (uint8_t *)data_vec.data(), data_vec.size() * sizeof(int32_t));

  vector<ConstGeTensorPtr> input = {tensor_0, tensor_1, tensor_2, tensor_3};
  vector<GeTensorPtr> outputs;

  shared_ptr<Kernel> kernel = KernelFactory::Instance().Create(STRIDEDSLICE);
  EXPECT_NE(kernel->Compute(op_desc_ptr, input, outputs), ge::SUCCESS);
}

TEST_F(UtestGraphPassesFoldingKernelStridedSliceKernel, CovShrinkAxisMask) {
  OpDescPtr op_desc_ptr = std::make_shared<OpDesc>("StridedSlice", "StridedSlice");
  AttrUtils::SetInt(op_desc_ptr, STRIDE_SLICE_ATTR_BEGIN_MASK, 0);
  AttrUtils::SetInt(op_desc_ptr, STRIDE_SLICE_ATTR_END_MASK, 0);
  AttrUtils::SetInt(op_desc_ptr, STRIDE_SLICE_ATTR_ELLIPSIS_MASK, 0);
  AttrUtils::SetInt(op_desc_ptr, STRIDE_SLICE_ATTR_NEW_AXIS_MASK, 0);
  AttrUtils::SetInt(op_desc_ptr, STRIDE_SLICE_ATTR_SHRINK_AXIS_MASK, 1);
  AttrUtils::SetInt(op_desc_ptr, ATTR_NAME_T, 0);

  GeTensorDesc dims_tensor_desc(GeShape({2, 3}), FORMAT_NCHW, DT_INT32);
  op_desc_ptr->AddInputDesc(0, dims_tensor_desc);
  op_desc_ptr->AddInputDesc(1, dims_tensor_desc);
  op_desc_ptr->AddInputDesc(2, dims_tensor_desc);
  op_desc_ptr->AddInputDesc(3, dims_tensor_desc);
  op_desc_ptr->AddOutputDesc(dims_tensor_desc);

  vector<int32_t> x_data = {1, 2, 3, 4, 5, 6};
  GeTensorDesc x_desc(GeShape({2, 3}), FORMAT_NCHW, DT_INT32);
  ConstGeTensorPtr tensor_0 =
      std::make_shared<GeTensor>(x_desc, (uint8_t *)x_data.data(), x_data.size() * sizeof(int32_t));

  vector<int32_t> begin_data = {0, 0};
  vector<int32_t> end_data = {1, 2};
  vector<int32_t> stride_data = {1, 1};
  GeTensorDesc idx_desc(GeShape({2}), FORMAT_NCHW, DT_INT32);
  ConstGeTensorPtr tensor_1 =
      std::make_shared<GeTensor>(idx_desc, (uint8_t *)begin_data.data(), begin_data.size() * sizeof(int32_t));
  ConstGeTensorPtr tensor_2 =
      std::make_shared<GeTensor>(idx_desc, (uint8_t *)end_data.data(), end_data.size() * sizeof(int32_t));
  ConstGeTensorPtr tensor_3 =
      std::make_shared<GeTensor>(idx_desc, (uint8_t *)stride_data.data(), stride_data.size() * sizeof(int32_t));

  vector<ConstGeTensorPtr> input = {tensor_0, tensor_1, tensor_2, tensor_3};
  vector<GeTensorPtr> outputs;

  shared_ptr<Kernel> kernel = KernelFactory::Instance().Create(STRIDEDSLICE);
  EXPECT_EQ(kernel->Compute(op_desc_ptr, input, outputs), ge::SUCCESS);
}

TEST_F(UtestGraphPassesFoldingKernelStridedSliceKernel, CovNewAxisMask) {
  OpDescPtr op_desc_ptr = std::make_shared<OpDesc>("StridedSlice", "StridedSlice");
  AttrUtils::SetInt(op_desc_ptr, STRIDE_SLICE_ATTR_BEGIN_MASK, 0);
  AttrUtils::SetInt(op_desc_ptr, STRIDE_SLICE_ATTR_END_MASK, 0);
  AttrUtils::SetInt(op_desc_ptr, STRIDE_SLICE_ATTR_ELLIPSIS_MASK, 0);
  AttrUtils::SetInt(op_desc_ptr, STRIDE_SLICE_ATTR_NEW_AXIS_MASK, 1);
  AttrUtils::SetInt(op_desc_ptr, STRIDE_SLICE_ATTR_SHRINK_AXIS_MASK, 0);
  AttrUtils::SetInt(op_desc_ptr, ATTR_NAME_T, 0);

  GeTensorDesc dims_tensor_desc(GeShape({3}), FORMAT_NCHW, DT_INT32);
  op_desc_ptr->AddInputDesc(0, dims_tensor_desc);
  op_desc_ptr->AddInputDesc(1, dims_tensor_desc);
  op_desc_ptr->AddInputDesc(2, dims_tensor_desc);
  op_desc_ptr->AddInputDesc(3, dims_tensor_desc);
  op_desc_ptr->AddOutputDesc(dims_tensor_desc);

  vector<int32_t> x_data = {1, 2, 3};
  GeTensorDesc x_desc(GeShape({3}), FORMAT_NCHW, DT_INT32);
  ConstGeTensorPtr tensor_0 =
      std::make_shared<GeTensor>(x_desc, (uint8_t *)x_data.data(), x_data.size() * sizeof(int32_t));

  vector<int32_t> begin_data = {0};
  vector<int32_t> end_data = {3};
  vector<int32_t> stride_data = {1};
  GeTensorDesc idx_desc(GeShape({1}), FORMAT_NCHW, DT_INT32);
  ConstGeTensorPtr tensor_1 =
      std::make_shared<GeTensor>(idx_desc, (uint8_t *)begin_data.data(), begin_data.size() * sizeof(int32_t));
  ConstGeTensorPtr tensor_2 =
      std::make_shared<GeTensor>(idx_desc, (uint8_t *)end_data.data(), end_data.size() * sizeof(int32_t));
  ConstGeTensorPtr tensor_3 =
      std::make_shared<GeTensor>(idx_desc, (uint8_t *)stride_data.data(), stride_data.size() * sizeof(int32_t));

  vector<ConstGeTensorPtr> input = {tensor_0, tensor_1, tensor_2, tensor_3};
  vector<GeTensorPtr> outputs;

  shared_ptr<Kernel> kernel = KernelFactory::Instance().Create(STRIDEDSLICE);
  EXPECT_EQ(kernel->Compute(op_desc_ptr, input, outputs), ge::SUCCESS);
}

TEST_F(UtestGraphPassesFoldingKernelStridedSliceKernel, CovEllipsisMaskExpand) {
  OpDescPtr op_desc_ptr = std::make_shared<OpDesc>("StridedSlice", "StridedSlice");
  AttrUtils::SetInt(op_desc_ptr, STRIDE_SLICE_ATTR_BEGIN_MASK, 0);
  AttrUtils::SetInt(op_desc_ptr, STRIDE_SLICE_ATTR_END_MASK, 0);
  AttrUtils::SetInt(op_desc_ptr, STRIDE_SLICE_ATTR_ELLIPSIS_MASK, 1);
  AttrUtils::SetInt(op_desc_ptr, STRIDE_SLICE_ATTR_NEW_AXIS_MASK, 0);
  AttrUtils::SetInt(op_desc_ptr, STRIDE_SLICE_ATTR_SHRINK_AXIS_MASK, 0);
  AttrUtils::SetInt(op_desc_ptr, ATTR_NAME_T, 0);

  GeTensorDesc dims_tensor_desc(GeShape({2, 3, 4}), FORMAT_NCHW, DT_INT32);
  op_desc_ptr->AddInputDesc(0, dims_tensor_desc);
  op_desc_ptr->AddInputDesc(1, dims_tensor_desc);
  op_desc_ptr->AddInputDesc(2, dims_tensor_desc);
  op_desc_ptr->AddInputDesc(3, dims_tensor_desc);
  op_desc_ptr->AddOutputDesc(dims_tensor_desc);

  vector<int32_t> x_data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24};
  GeTensorDesc x_desc(GeShape({2, 3, 4}), FORMAT_NCHW, DT_INT32);
  ConstGeTensorPtr tensor_0 =
      std::make_shared<GeTensor>(x_desc, (uint8_t *)x_data.data(), x_data.size() * sizeof(int32_t));

  vector<int32_t> begin_data = {0, 1};
  vector<int32_t> end_data = {2, 2};
  vector<int32_t> stride_data = {1, 1};
  GeTensorDesc idx_desc(GeShape({2}), FORMAT_NCHW, DT_INT32);
  ConstGeTensorPtr tensor_1 =
      std::make_shared<GeTensor>(idx_desc, (uint8_t *)begin_data.data(), begin_data.size() * sizeof(int32_t));
  ConstGeTensorPtr tensor_2 =
      std::make_shared<GeTensor>(idx_desc, (uint8_t *)end_data.data(), end_data.size() * sizeof(int32_t));
  ConstGeTensorPtr tensor_3 =
      std::make_shared<GeTensor>(idx_desc, (uint8_t *)stride_data.data(), stride_data.size() * sizeof(int32_t));

  vector<ConstGeTensorPtr> input = {tensor_0, tensor_1, tensor_2, tensor_3};
  vector<GeTensorPtr> outputs;

  shared_ptr<Kernel> kernel = KernelFactory::Instance().Create(STRIDEDSLICE);
  EXPECT_EQ(kernel->Compute(op_desc_ptr, input, outputs), ge::SUCCESS);
}

TEST_F(UtestGraphPassesFoldingKernelStridedSliceKernel, NullAttr) {
  vector<int64_t> dims_vec_0 = {2, 2};
  vector<float> data_vec_0 = {1.0, 2.0, 3.0, 4.0};
  GeTensorDesc tensor_desc_0(GeShape(dims_vec_0), FORMAT_NCHW, DT_FLOAT);
  ConstGeTensorPtr tensor_0 =
      std::make_shared<GeTensor>(tensor_desc_0, (uint8_t *)data_vec_0.data(), data_vec_0.size() * sizeof(float));

  vector<int32_t> begin_data = {0, 0};
  vector<int32_t> end_data = {2, 2};
  vector<int32_t> stride_data = {1, 1};
  GeTensorDesc idx_desc(GeShape({2}), FORMAT_NCHW, DT_INT32);
  ConstGeTensorPtr tensor_1 =
      std::make_shared<GeTensor>(idx_desc, (uint8_t *)begin_data.data(), begin_data.size() * sizeof(int32_t));
  ConstGeTensorPtr tensor_2 =
      std::make_shared<GeTensor>(idx_desc, (uint8_t *)end_data.data(), end_data.size() * sizeof(int32_t));
  ConstGeTensorPtr tensor_3 =
      std::make_shared<GeTensor>(idx_desc, (uint8_t *)stride_data.data(), stride_data.size() * sizeof(int32_t));

  vector<ConstGeTensorPtr> input = {tensor_0, tensor_1, tensor_2, tensor_3};
  vector<GeTensorPtr> outputs;

  shared_ptr<Kernel> kernel = KernelFactory::Instance().Create(STRIDEDSLICE);
  Status status = kernel->Compute(nullptr, input, outputs);
  EXPECT_EQ(NOT_CHANGED, status);
}

TEST_F(UtestGraphPassesFoldingKernelStridedSliceKernel, MissingMaskAttr) {
  OpDescPtr op_desc_ptr = std::make_shared<OpDesc>("StridedSlice", "StridedSlice");
  GeTensorDesc dims_tensor_desc(GeShape({2, 2}), FORMAT_NCHW, DT_FLOAT);
  op_desc_ptr->AddInputDesc(0, dims_tensor_desc);
  op_desc_ptr->AddInputDesc(1, dims_tensor_desc);
  op_desc_ptr->AddInputDesc(2, dims_tensor_desc);
  op_desc_ptr->AddInputDesc(3, dims_tensor_desc);
  op_desc_ptr->AddOutputDesc(dims_tensor_desc);

  vector<int64_t> dims_vec_0 = {2, 2};
  vector<float> data_vec_0 = {1.0, 2.0, 3.0, 4.0};
  GeTensorDesc tensor_desc_0(GeShape(dims_vec_0), FORMAT_NCHW, DT_FLOAT);
  ConstGeTensorPtr tensor_0 =
      std::make_shared<GeTensor>(tensor_desc_0, (uint8_t *)data_vec_0.data(), data_vec_0.size() * sizeof(float));

  vector<int32_t> begin_data = {0, 0};
  vector<int32_t> end_data = {2, 2};
  vector<int32_t> stride_data = {1, 1};
  GeTensorDesc idx_desc(GeShape({2}), FORMAT_NCHW, DT_INT32);
  ConstGeTensorPtr tensor_1 =
      std::make_shared<GeTensor>(idx_desc, (uint8_t *)begin_data.data(), begin_data.size() * sizeof(int32_t));
  ConstGeTensorPtr tensor_2 =
      std::make_shared<GeTensor>(idx_desc, (uint8_t *)end_data.data(), end_data.size() * sizeof(int32_t));
  ConstGeTensorPtr tensor_3 =
      std::make_shared<GeTensor>(idx_desc, (uint8_t *)stride_data.data(), stride_data.size() * sizeof(int32_t));

  vector<ConstGeTensorPtr> input = {tensor_0, tensor_1, tensor_2, tensor_3};
  vector<GeTensorPtr> outputs;

  shared_ptr<Kernel> kernel = KernelFactory::Instance().Create(STRIDEDSLICE);
  Status status = kernel->Compute(op_desc_ptr, input, outputs);
  EXPECT_EQ(NOT_CHANGED, status);
}
