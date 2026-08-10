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

#include <cstdlib>
#include <cstdio>
#include <iostream>
#include "slice/data_slice_helper.h"
#include "slice/data_slice_toolkit.h"
#include "slice/data_slice_factory.h"
#include "slice/data_slice_elementwise_impl.h"
#include "register/infer_axis_slice_registry.h"
#include "framework/common/debug/ge_log.h"
#include "graph/operator_factory_impl.h"
#include "framework/common/util.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/type_utils.h"
#include "graph/debug/ge_attr_define.h"
#include "slice/data_slice_adapter.h"

using namespace std;
using namespace testing;
namespace ge {
class DataSlice : public testing::Test {
 protected:
  void SetUp() {}
  void TearDown() {}
};
IMPLEMT_COMMON_INFER_AXIS_TYPE_INFO(Temp) {
  AxisTypeInfo info1;
  info1.SetAxisType(ge::AxisType::ELEMENTWISE);
  std::vector<CutInfo> relate_inputs1 = {{0, {0}}};
  std::vector<CutInfo> relate_outputs1 = {{0, {0}}};
  info1.SetRelateInputs(relate_inputs1);
  info1.SetRelateOutputs(relate_outputs1);
  AxisTypeInfo info2;
  info2.SetAxisType(ge::AxisType::ELEMENTWISE);
  std::vector<CutInfo> relate_inputs2 = {{0, {1}}};
  std::vector<CutInfo> relate_outputs2 = {{0, {1}}};
  info2.SetRelateInputs(relate_inputs2);
  info2.SetRelateOutputs(relate_outputs2);
  AxisTypeInfo info3;
  info3.SetAxisType(ge::AxisType::ELEMENTWISE);
  std::vector<CutInfo> relate_inputs3 = {{0, {2}}};
  std::vector<CutInfo> relate_outputs3 = {{0, {2}}};
  info3.SetRelateInputs(relate_inputs3);
  info3.SetRelateOutputs(relate_outputs3);
  AxisTypeInfo info4;
  info4.SetAxisType(ge::AxisType::ELEMENTWISE);
  std::vector<CutInfo> relate_inputs4 = {{0, {3}}, {1, {0}}};
  std::vector<CutInfo> relate_outputs4 = {{0, {3}}};
  info4.SetRelateInputs(relate_inputs4);
  info4.SetRelateOutputs(relate_outputs4);
  AxisTypeInfo info5;
  info5.SetAxisType(ge::AxisType::ELEMENTWISE);
  std::vector<CutInfo> relate_inputs5 = {{0, {3}}};
  std::vector<CutInfo> relate_outputs5 = {{0, {3}}};
  info5.SetRelateInputs(relate_inputs5);
  info5.SetRelateOutputs(relate_outputs5);

  axis_type = {info1, info2, info3, info4, info5};
  return GRAPH_SUCCESS;
}
INFER_AXIS_TYPE_INFO_REG(Add, Temp);
INFER_AXIS_TYPE_INFO_REG(Cast, Temp);
IMPLEMT_COMMON_INFER_AXIS_TYPE_INFO(Func) {
  return GRAPH_FAILED;
}
INFER_AXIS_TYPE_INFO_REG(Softmax, Func);
IMPLEMT_COMMON_INFER_AXIS_SLICE(Temp1) {
  input_param = {{{}, {}, {}, {0, 31}}};
  return GRAPH_SUCCESS;
}
INFER_AXIS_SLICE_FUNC_REG(Add, Temp1);
TEST_F(DataSlice, data_slice_helper_1) {
  OpDescPtr op_desc = std::make_shared<OpDesc>("Add", "Add");
  GeTensorDesc output_desc;
  op_desc->AddOutputDesc("output", output_desc);
  GeTensorDesc input_desc;
  op_desc->AddInputDesc("input", input_desc);
  AxisTypeInfo axis_type_info;
  axis_type_info.SetAxisType(AxisType::ELEMENTWISE);
  std::pair<int64_t, std::vector<int64_t>> input_cut_info(0, {0});
  axis_type_info.AddInputCutInfo(input_cut_info);
  std::pair<int64_t, std::vector<int64_t>> output_cut_info(0, {0});
  axis_type_info.AddOutputCutInfo(output_cut_info);
  Status ret = DataSliceHelper::InferAxisSlice(op_desc, axis_type_info);
  EXPECT_EQ(ret, SUCCESS);
}

TEST_F(DataSlice, data_slice_helper_2) {
  OpDescPtr op_desc = std::make_shared<OpDesc>("Cast", "Cast");
  GeTensorDesc output_desc;
  op_desc->AddOutputDesc("output", output_desc);
  GeTensorDesc input_desc;
  op_desc->AddInputDesc("input", input_desc);
  AxisTypeInfo axis_type_info;
  axis_type_info.SetAxisType(AxisType::ELEMENTWISE);
  std::pair<int64_t, std::vector<int64_t>> input_cut_info(0, {0});
  axis_type_info.AddInputCutInfo(input_cut_info);
  std::pair<int64_t, std::vector<int64_t>> output_cut_info(0, {0});
  axis_type_info.AddOutputCutInfo(output_cut_info);
  Status ret = DataSliceHelper::InferAxisSlice(op_desc, axis_type_info);
  EXPECT_EQ(ret, SUCCESS);
}

TEST_F(DataSlice, data_slice_helper_3) {
  OpDescPtr op_desc = std::make_shared<OpDesc>("Cast", "Cast");
  GeTensorDesc output_desc;
  op_desc->AddOutputDesc("output", output_desc);
  GeTensorDesc input_desc;
  op_desc->AddInputDesc("input", input_desc);
  AxisTypeInfo axis_type_info;
  axis_type_info.SetAxisType(AxisType::UNSPLIT);
  std::pair<int64_t, std::vector<int64_t>> input_cut_info(0, {0});
  axis_type_info.AddInputCutInfo(input_cut_info);
  std::pair<int64_t, std::vector<int64_t>> output_cut_info(0, {0});
  axis_type_info.AddOutputCutInfo(output_cut_info);
  Status ret = DataSliceHelper::InferAxisSlice(op_desc, axis_type_info);
  EXPECT_EQ(ret, SUCCESS);
}

TEST_F(DataSlice, data_slice_helper_4) {
  OpDescPtr op_desc = std::make_shared<OpDesc>("Cast", "Cast");
  GeTensorDesc output_desc;
  op_desc->AddOutputDesc("output", output_desc);
  GeTensorDesc input_desc;
  op_desc->AddInputDesc("input", input_desc);
  AxisTypeInfo axis_type_info;
  axis_type_info.SetAxisType(AxisType::SLIDINGWINDOW);
  std::pair<int64_t, std::vector<int64_t>> input_cut_info(0, {0});
  axis_type_info.AddInputCutInfo(input_cut_info);
  std::pair<int64_t, std::vector<int64_t>> output_cut_info(0, {0});
  axis_type_info.AddOutputCutInfo(output_cut_info);
  Status ret = DataSliceHelper::InferAxisSlice(op_desc, axis_type_info);
  EXPECT_EQ(ret, FAILED);
}

TEST_F(DataSlice, data_slice_helper_5) {
  OpDescPtr op_desc = std::make_shared<OpDesc>("Cast", "Cast");
  GeTensorDesc output_desc;
  op_desc->AddOutputDesc("output", output_desc);
  GeTensorDesc input_desc;
  op_desc->AddInputDesc("input", input_desc);
  std::vector<AxisTypeInfo> axis_type_info;
  Status ret = DataSliceHelper::GetSliceInfo(op_desc, axis_type_info);
  EXPECT_EQ(ret, SUCCESS);
}

TEST_F(DataSlice, data_slice_helper_6) {
  ComputeGraphPtr test_graph = std::make_shared<ComputeGraph>("test_graph");
  OpDescPtr op_desc = std::make_shared<OpDesc>("Cast", "Cast");
  GeTensorDesc output_desc;
  op_desc->AddOutputDesc("output", output_desc);
  GeTensorDesc input_desc;
  op_desc->AddInputDesc("input", input_desc);

  NodePtr node = test_graph->AddNode(op_desc);
  std::vector<AxisTypeInfo> axis_type_info;

  Status ret = DataSliceHelper::GetSliceInfo(node, axis_type_info);
  EXPECT_EQ(ret, SUCCESS);
}

TEST_F(DataSlice, data_slice_helper_7) {
  ComputeGraphPtr test_graph = std::make_shared<ComputeGraph>("test_graph");
  OpDescPtr op_desc = std::make_shared<OpDesc>("test", "test");
  GeTensorDesc output_desc;
  op_desc->AddOutputDesc("output", output_desc);
  GeTensorDesc input_desc;
  op_desc->AddInputDesc("input", input_desc);

  NodePtr node = test_graph->AddNode(op_desc);
  std::vector<AxisTypeInfo> axis_type_info;

  Status ret = DataSliceHelper::GetSliceInfo(node, axis_type_info);
  EXPECT_EQ(ret, FAILED);
}

TEST_F(DataSlice, data_slice_helper_8) {
  ComputeGraphPtr test_graph = std::make_shared<ComputeGraph>("test_graph");
  OpDescPtr op_desc = std::make_shared<OpDesc>("Softmax", "Softmax");
  GeTensorDesc output_desc;
  op_desc->AddOutputDesc("output", output_desc);
  GeTensorDesc input_desc;
  op_desc->AddInputDesc("input", input_desc);

  NodePtr node = test_graph->AddNode(op_desc);
  std::vector<AxisTypeInfo> axis_type_info;

  Status ret = DataSliceHelper::GetSliceInfo(node, axis_type_info);
  EXPECT_EQ(ret, FAILED);
}

TEST_F(DataSlice, data_slice_helper_9) {
  OpDescPtr op_desc = std::make_shared<OpDesc>("Softmax", "Softmax");
  GeTensorDesc output_desc;
  op_desc->AddOutputDesc("output", output_desc);
  GeTensorDesc input_desc;
  op_desc->AddInputDesc("input", input_desc);

  std::vector<AxisTypeInfo> axis_type_info;

  Status ret = DataSliceHelper::GetSliceInfo(op_desc, axis_type_info);
  EXPECT_EQ(ret, FAILED);
}

TEST_F(DataSlice, data_slice_helper_get_avinci_slice_info_Add_NC1HWC0_reshape) {
  ComputeGraphPtr test_graph = std::make_shared<ComputeGraph>("test_graph");
  OpDescPtr op_desc = std::make_shared<OpDesc>("Add", "Add");
  GeTensorDesc output_desc(ge::GeShape({10, 3, 20, 30, 16}), ge::Format::FORMAT_NC1HWC0);
  output_desc.SetOriginShape(ge::GeShape({10, 20}));
  output_desc.SetOriginFormat(ge::Format::FORMAT_NHWC);
  ge::AttrUtils::SetStr(output_desc, ge::ATTR_NAME_RESHAPE_INFER_TYPE, "NH");
  op_desc->AddOutputDesc("output0", output_desc);
  GeTensorDesc input_desc0(ge::GeShape({10, 3, 20, 30, 16}), ge::Format::FORMAT_NC1HWC0);
  input_desc0.SetOriginShape(ge::GeShape({10, 20, 30, 40}));
  input_desc0.SetOriginFormat(ge::Format::FORMAT_NHWC);
  ge::AttrUtils::SetStr(input_desc0, ge::ATTR_NAME_RESHAPE_INFER_TYPE, "NH");
  op_desc->AddInputDesc("input0", input_desc0);
  GeTensorDesc input_desc1(ge::GeShape({40}), ge::Format::FORMAT_NHWC);
  input_desc1.SetOriginShape(ge::GeShape({40}));
  input_desc1.SetOriginFormat(ge::Format::FORMAT_NHWC);
  ge::AttrUtils::SetStr(input_desc1, ge::ATTR_NAME_RESHAPE_INFER_TYPE, "NH");
  op_desc->AddInputDesc("input1", input_desc1);
  NodePtr node = test_graph->AddNode(op_desc);
  std::vector<AxisTypeInfo> axis_type_info;
  Status ret = DataSliceHelper::GetDavinciSliceInfo(node, axis_type_info);
  EXPECT_EQ(ret, SUCCESS);
}

TEST_F(DataSlice, data_slice_helper_get_avinci_slice_info_Add_NC1HWC0) {
  ComputeGraphPtr test_graph = std::make_shared<ComputeGraph>("test_graph");
  OpDescPtr op_desc = std::make_shared<OpDesc>("Add", "Add");
  GeTensorDesc output_desc(ge::GeShape({10, 3, 20, 30, 16}), ge::Format::FORMAT_NC1HWC0);
  output_desc.SetOriginShape(ge::GeShape({10, 20, 30, 40}));
  output_desc.SetOriginFormat(ge::Format::FORMAT_NHWC);
  op_desc->AddOutputDesc("output0", output_desc);
  GeTensorDesc input_desc0(ge::GeShape({10, 3, 20, 30, 16}), ge::Format::FORMAT_NC1HWC0);
  input_desc0.SetOriginShape(ge::GeShape({10, 20, 30, 40}));
  input_desc0.SetOriginFormat(ge::Format::FORMAT_NHWC);
  op_desc->AddInputDesc("input0", input_desc0);
  GeTensorDesc input_desc1(ge::GeShape({40}), ge::Format::FORMAT_NHWC);
  input_desc1.SetOriginShape(ge::GeShape({40}));
  input_desc1.SetOriginFormat(ge::Format::FORMAT_NHWC);
  op_desc->AddInputDesc("input1", input_desc1);
  NodePtr node = test_graph->AddNode(op_desc);
  std::vector<AxisTypeInfo> axis_type_info;
  Status ret = DataSliceHelper::GetDavinciSliceInfo(node, axis_type_info);
  EXPECT_EQ(ret, SUCCESS);
}

TEST_F(DataSlice, data_slice_helper_get_avinci_slice_info_Add_NZ) {
  ComputeGraphPtr test_graph = std::make_shared<ComputeGraph>("test_graph");
  OpDescPtr op_desc = std::make_shared<OpDesc>("Add", "Add");
  GeTensorDesc output_desc(ge::GeShape({10, 3, 20, 30, 16}), ge::Format::FORMAT_FRACTAL_NZ);
  output_desc.SetOriginShape(ge::GeShape({10, 20, 30, 40}));
  output_desc.SetOriginFormat(ge::Format::FORMAT_NHWC);
  op_desc->AddOutputDesc("output0", output_desc);
  GeTensorDesc input_desc0(ge::GeShape({10, 3, 20, 30, 16}), ge::Format::FORMAT_FRACTAL_NZ);
  input_desc0.SetOriginShape(ge::GeShape({10, 20, 30, 40}));
  input_desc0.SetOriginFormat(ge::Format::FORMAT_NHWC);
  op_desc->AddInputDesc("input0", input_desc0);
  GeTensorDesc input_desc1(ge::GeShape({40}), ge::Format::FORMAT_NHWC);
  input_desc1.SetOriginShape(ge::GeShape({40}));
  input_desc1.SetOriginFormat(ge::Format::FORMAT_NHWC);
  op_desc->AddInputDesc("input1", input_desc1);
  NodePtr node = test_graph->AddNode(op_desc);
  std::vector<AxisTypeInfo> axis_type_info;
  Status ret = DataSliceHelper::GetDavinciSliceInfo(node, axis_type_info);
  EXPECT_EQ(ret, SUCCESS);
}

TEST_F(DataSlice, data_slice_helper_get_avinci_slice_info_Add_NoSplit) {
  ComputeGraphPtr test_graph = std::make_shared<ComputeGraph>("test_graph");
  OpDescPtr op_desc = std::make_shared<OpDesc>("Add", "Add");
  GeTensorDesc output_desc(ge::GeShape({10, 3, 40, 40}), ge::Format::FORMAT_NCHW);
  output_desc.SetOriginShape(ge::GeShape({10, 40, 40, 3}));
  output_desc.SetOriginFormat(ge::Format::FORMAT_NHWC);
  op_desc->AddOutputDesc("output0", output_desc);
  GeTensorDesc input_desc0(ge::GeShape({10, 3, 40, 40}), ge::Format::FORMAT_NCHW);
  input_desc0.SetOriginShape(ge::GeShape({10, 40, 40, 3}));
  input_desc0.SetOriginFormat(ge::Format::FORMAT_NHWC);
  op_desc->AddInputDesc("input0", input_desc0);
  GeTensorDesc input_desc1(ge::GeShape({40}), ge::Format::FORMAT_NHWC);
  input_desc1.SetOriginShape(ge::GeShape({40}));
  input_desc1.SetOriginFormat(ge::Format::FORMAT_NHWC);
  op_desc->AddInputDesc("input1", input_desc1);
  NodePtr node = test_graph->AddNode(op_desc);
  std::vector<AxisTypeInfo> axis_type_info;
  Status ret = DataSliceHelper::GetDavinciSliceInfo(node, axis_type_info);
  EXPECT_EQ(ret, SUCCESS);
}

TEST_F(DataSlice, data_slice_helper_get_avinci_slice_info_Add_NoShape) {
  ComputeGraphPtr test_graph = std::make_shared<ComputeGraph>("test_graph");
  OpDescPtr op_desc = std::make_shared<OpDesc>("Add", "Add");
  GeTensorDesc output_desc(ge::GeShape({10, 3, 40, 40}), ge::Format::FORMAT_NCHW);
  output_desc.SetOriginFormat(ge::Format::FORMAT_NHWC);
  op_desc->AddOutputDesc("output0", output_desc);
  GeTensorDesc input_desc0(ge::GeShape({10, 3, 40, 40}), ge::Format::FORMAT_NCHW);
  input_desc0.SetOriginShape(ge::GeShape({10, 40, 40, 3}));
  input_desc0.SetOriginFormat(ge::Format::FORMAT_NHWC);
  op_desc->AddInputDesc("input0", input_desc0);
  GeTensorDesc input_desc1(ge::GeShape({40}), ge::Format::FORMAT_NHWC);
  input_desc1.SetOriginShape(ge::GeShape({40}));
  input_desc1.SetOriginFormat(ge::Format::FORMAT_NHWC);
  op_desc->AddInputDesc("input1", input_desc1);
  NodePtr node = test_graph->AddNode(op_desc);
  std::vector<AxisTypeInfo> axis_type_info;
  Status ret = DataSliceHelper::GetDavinciSliceInfo(node, axis_type_info);
  EXPECT_EQ(ret, SUCCESS);
}

TEST_F(DataSlice, data_slice_helper_infer_avinci_axis_slice_elementwise) {
  ComputeGraphPtr test_graph = std::make_shared<ComputeGraph>("test_graph");
  OpDescPtr op_desc = std::make_shared<OpDesc>("Add", "Add");
  GeTensorDesc output_desc(ge::GeShape({10, 3, 20, 30, 16}), ge::Format::FORMAT_NC1HWC0);
  output_desc.SetOriginShape(ge::GeShape({10, 20, 30, 40}));
  output_desc.SetOriginFormat(ge::Format::FORMAT_NHWC);
  std::vector<std::vector<int64_t>> slice_info = {{}, {0, 1}, {}, {}, {}};
  (void)AttrUtils::SetListListInt(output_desc, ge::ATTR_NAME_DATA_SLICE, slice_info);
  op_desc->AddOutputDesc("output0", output_desc);
  GeTensorDesc input_desc0(ge::GeShape({10, 3, 20, 30, 16}), ge::Format::FORMAT_NC1HWC0);
  input_desc0.SetOriginShape(ge::GeShape({10, 20, 30, 40}));
  input_desc0.SetOriginFormat(ge::Format::FORMAT_NHWC);
  (void)AttrUtils::SetListListInt(input_desc0, ge::ATTR_NAME_DATA_SLICE, slice_info);
  op_desc->AddInputDesc("input0", input_desc0);
  GeTensorDesc input_desc1(ge::GeShape({40}), ge::Format::FORMAT_NHWC);
  input_desc1.SetOriginShape(ge::GeShape({40}));
  input_desc1.SetOriginFormat(ge::Format::FORMAT_NHWC);
  op_desc->AddInputDesc("input1", input_desc1);
  AxisTypeInfo axis_type_info;
  std::vector<CutInfo> relate_inputs = {{0, {1}}};
  std::vector<CutInfo> relate_outputs = {{0, {1}}};
  std::vector<CutInfo> ori_relate_inputs = {{0, {3}}};
  std::vector<CutInfo> ori_relate_outputs = {{0, {3}}};
  axis_type_info.SetAxisType(ge::AxisType::ELEMENTWISE);
  axis_type_info.SetRelateInputs(relate_inputs);
  axis_type_info.SetRelateOutputs(relate_outputs);
  axis_type_info.SetOriRelateInputs(ori_relate_inputs);
  axis_type_info.SetOriRelateOutputs(ori_relate_outputs);
  Status ret = DataSliceHelper::InferDavinciAxisSlice(op_desc, axis_type_info);
  EXPECT_EQ(ret, SUCCESS);
}

TEST_F(DataSlice, data_slice_helper_infer_avinci_axis_slice_elementwise_addn) {
  ComputeGraphPtr test_graph = std::make_shared<ComputeGraph>("test_graph");
  OpDescPtr op_desc = std::make_shared<OpDesc>("AddN", "AddN");
  GeTensorDesc output_desc(ge::GeShape({10, 3, 20, 30, 16}), ge::Format::FORMAT_NC1HWC0);
  output_desc.SetOriginShape(ge::GeShape({10, 20, 30, 40}));
  output_desc.SetOriginFormat(ge::Format::FORMAT_NHWC);
  std::vector<std::vector<int64_t>> slice_info = {{}, {0, 1}, {}, {}, {}};
  (void)AttrUtils::SetListListInt(output_desc, ge::ATTR_NAME_DATA_SLICE, slice_info);
  op_desc->AddOutputDesc("output0", output_desc);
  GeTensorDesc input_desc0(ge::GeShape({10, 3, 20, 30, 16}), ge::Format::FORMAT_NC1HWC0);
  input_desc0.SetOriginShape(ge::GeShape({10, 20, 30, 40}));
  input_desc0.SetOriginFormat(ge::Format::FORMAT_NHWC);
  (void)AttrUtils::SetListListInt(input_desc0, ge::ATTR_NAME_DATA_SLICE, slice_info);
  op_desc->AddInputDesc("input0", input_desc0);
  GeTensorDesc input_desc1(ge::GeShape({40}), ge::Format::FORMAT_NHWC);
  input_desc1.SetOriginShape(ge::GeShape({40}));
  input_desc1.SetOriginFormat(ge::Format::FORMAT_NHWC);
  op_desc->AddInputDesc("input1", input_desc1);
  AxisTypeInfo axis_type_info;
  std::vector<CutInfo> relate_inputs = {{0, {1}}};
  std::vector<CutInfo> relate_outputs = {{0, {1}}};
  std::vector<CutInfo> ori_relate_inputs = {{0, {3}}};
  std::vector<CutInfo> ori_relate_outputs = {{0, {3}}};
  axis_type_info.SetAxisType(ge::AxisType::ELEMENTWISE);
  axis_type_info.SetRelateInputs(relate_inputs);
  axis_type_info.SetRelateOutputs(relate_outputs);
  axis_type_info.SetOriRelateInputs(ori_relate_inputs);
  axis_type_info.SetOriRelateOutputs(ori_relate_outputs);
  Status ret = DataSliceHelper::InferDavinciAxisSlice(op_desc, axis_type_info);
  EXPECT_EQ(ret, SUCCESS);
}

TEST_F(DataSlice, data_slice_helper_infer_avinci_axis_slice_elementwise_addn_noshape) {
  ComputeGraphPtr test_graph = std::make_shared<ComputeGraph>("test_graph");
  OpDescPtr op_desc = std::make_shared<OpDesc>("AddN", "AddN");
  GeTensorDesc output_desc(ge::GeShape({10, 3, 20, 30, 16}), ge::Format::FORMAT_NC1HWC0);
  output_desc.SetOriginShape(ge::GeShape({10, 20, 30, 40}));
  output_desc.SetOriginFormat(ge::Format::FORMAT_NHWC);
  std::vector<std::vector<int64_t>> slice_info = {{}, {0, 1}, {}, {}, {}};
  (void)AttrUtils::SetListListInt(output_desc, ge::ATTR_NAME_DATA_SLICE, slice_info);
  op_desc->AddOutputDesc("output0", output_desc);
  GeTensorDesc input_desc0(ge::GeShape({10, 3, 20, 30, 16}), ge::Format::FORMAT_NC1HWC0);
  input_desc0.SetOriginFormat(ge::Format::FORMAT_NHWC);
  (void)AttrUtils::SetListListInt(input_desc0, ge::ATTR_NAME_DATA_SLICE, slice_info);
  op_desc->AddInputDesc("input0", input_desc0);
  GeTensorDesc input_desc1(ge::GeShape({40}), ge::Format::FORMAT_NHWC);
  input_desc1.SetOriginShape(ge::GeShape({40}));
  input_desc1.SetOriginFormat(ge::Format::FORMAT_NHWC);
  op_desc->AddInputDesc("input1", input_desc1);
  AxisTypeInfo axis_type_info;
  std::vector<CutInfo> relate_inputs = {{0, {1}}};
  std::vector<CutInfo> relate_outputs = {{0, {1}}};
  std::vector<CutInfo> ori_relate_inputs = {{0, {3}}};
  std::vector<CutInfo> ori_relate_outputs = {{0, {3}}};
  axis_type_info.SetAxisType(ge::AxisType::ELEMENTWISE);
  axis_type_info.SetRelateInputs(relate_inputs);
  axis_type_info.SetRelateOutputs(relate_outputs);
  axis_type_info.SetOriRelateInputs(ori_relate_inputs);
  axis_type_info.SetOriRelateOutputs(ori_relate_outputs);
  Status ret = DataSliceHelper::InferDavinciAxisSlice(op_desc, axis_type_info);
  EXPECT_NE(ret, SUCCESS);
}

TEST_F(DataSlice, data_slice_helper_infer_avinci_axis_slice_reducemax) {
  ComputeGraphPtr test_graph = std::make_shared<ComputeGraph>("test_graph");
  OpDescPtr op_desc = std::make_shared<OpDesc>("Add", "Add");
  GeTensorDesc output_desc(ge::GeShape({10, 3, 20, 30, 16}), ge::Format::FORMAT_NC1HWC0);
  output_desc.SetOriginFormat(ge::Format::FORMAT_NHWC);
  output_desc.SetOriginShape(ge::GeShape({10, 20, 30, 40}));
  std::vector<std::vector<int64_t>> slice_info = {{}, {0, 1}, {}, {}, {}};
  (void)AttrUtils::SetListListInt(output_desc, ge::ATTR_NAME_DATA_SLICE, slice_info);
  op_desc->AddOutputDesc("output0", output_desc);
  GeTensorDesc input_desc0(ge::GeShape({10, 3, 20, 30, 16}), ge::Format::FORMAT_NC1HWC0);
  input_desc0.SetOriginFormat(ge::Format::FORMAT_NHWC);
  (void)AttrUtils::SetListListInt(input_desc0, ge::ATTR_NAME_DATA_SLICE, slice_info);
  op_desc->AddInputDesc("input0", input_desc0);
  GeTensorDesc input_desc1(ge::GeShape({40}), ge::Format::FORMAT_NHWC);
  input_desc1.SetOriginShape(ge::GeShape({40}));
  input_desc1.SetOriginFormat(ge::Format::FORMAT_NHWC);
  op_desc->AddInputDesc("input1", input_desc1);
  AxisTypeInfo axis_type_info;
  std::vector<CutInfo> relate_inputs = {{0, {1}}};
  std::vector<CutInfo> relate_outputs = {{0, {1}}};
  std::vector<CutInfo> ori_relate_inputs = {{0, {3}}};
  std::vector<CutInfo> ori_relate_outputs = {{0, {3}}};
  axis_type_info.SetAxisType(ge::AxisType::REDUCEMAX);
  axis_type_info.SetRelateInputs(relate_inputs);
  axis_type_info.SetRelateOutputs(relate_outputs);
  axis_type_info.SetOriRelateInputs(ori_relate_inputs);
  axis_type_info.SetOriRelateOutputs(ori_relate_outputs);
  Status ret = DataSliceHelper::InferDavinciAxisSlice(op_desc, axis_type_info);
  EXPECT_NE(ret, SUCCESS);
}

TEST_F(DataSlice, data_slice_helper_infer_avinci_axis_slice_reducegether) {
  OpDescPtr op_desc = std::make_shared<OpDesc>("Cast", "Cast");
  GeTensorDesc output_desc;
  op_desc->AddOutputDesc("output", output_desc);
  GeTensorDesc input_desc;
  op_desc->AddInputDesc("input", input_desc);
  AxisTypeInfo axis_type_info;
  axis_type_info.SetAxisType(AxisType::REDUCEGATHER);
  std::pair<int64_t, std::vector<int64_t>> input_cut_info(0, {0});
  axis_type_info.AddInputCutInfo(input_cut_info);
  std::pair<int64_t, std::vector<int64_t>> output_cut_info(0, {0});
  axis_type_info.AddOutputCutInfo(output_cut_info);
  Status ret = DataSliceHelper::InferDavinciAxisSlice(op_desc, axis_type_info);
  EXPECT_EQ(ret, FAILED);
}

TEST_F(DataSlice, data_slice_helper_infer_avinci_axis_slice_slidingwindow) {
  OpDescPtr op_desc = std::make_shared<OpDesc>("Add", "Add");
  GeTensorDesc output_desc;
  op_desc->AddOutputDesc("output", output_desc);
  GeTensorDesc input_desc;
  op_desc->AddInputDesc("input", input_desc);
  AxisTypeInfo axis_type_info;
  axis_type_info.SetAxisType(AxisType::SLIDINGWINDOW);
  std::pair<int64_t, std::vector<int64_t>> input_cut_info(0, {0});
  axis_type_info.AddInputCutInfo(input_cut_info);
  std::pair<int64_t, std::vector<int64_t>> output_cut_info(0, {0});
  axis_type_info.AddOutputCutInfo(output_cut_info);
  Status ret = DataSliceHelper::InferDavinciAxisSlice(op_desc, axis_type_info);
  EXPECT_EQ(ret, SUCCESS);
}

TEST_F(DataSlice, data_slice_elementwise_impl_failed) {
  OpDescPtr op_desc = std::make_shared<OpDesc>("Cast", "Cast");
  GeTensorDesc output_desc(ge::GeShape({20, 20, 20, 20}), ge::Format::FORMAT_NHWC);
  op_desc->AddOutputDesc("output", output_desc);
  GeTensorDesc input_desc(ge::GeShape({5, 5, 5, 5}), ge::Format::FORMAT_NHWC);
  op_desc->AddInputDesc("input", input_desc);
  AxisTypeInfo axis_type_info;
  axis_type_info.SetAxisType(AxisType::ELEMENTWISE);
  std::pair<int64_t, std::vector<int64_t>> input_cut_info(0, {0});
  axis_type_info.AddInputCutInfo(input_cut_info);
  std::pair<int64_t, std::vector<int64_t>> output_cut_info(0, {0});
  axis_type_info.AddOutputCutInfo(output_cut_info);
  DataSliceType out_data_slice;
  DataSliceType in_data_slice;
  Operator op_proxy = OpDescUtils::CreateOperatorFromOpDesc(op_desc);
  DataSliceElementwiseImpl dataElementwiseImpl;
  Status ret = dataElementwiseImpl.InferAxisSlice(op_proxy, axis_type_info, out_data_slice, in_data_slice);
  EXPECT_EQ(ret, FAILED);

  DataSliceType in_data_slice_wrong = {{{0}}};
  ret = dataElementwiseImpl.InferAxisSlice(op_proxy, axis_type_info, out_data_slice, in_data_slice_wrong);
  EXPECT_EQ(ret, FAILED);

  AxisTypeInfo axis_type_info_wrong;
  ret = dataElementwiseImpl.InferAxisSlice(op_proxy, axis_type_info_wrong, out_data_slice, in_data_slice);
  EXPECT_EQ(ret, FAILED);

  DataSliceType out_data_slice_wrong = {{{0, 60}}};
  ret = dataElementwiseImpl.InferAxisSlice(op_proxy, axis_type_info, out_data_slice_wrong, in_data_slice);
  EXPECT_EQ(ret, FAILED);

  out_data_slice_wrong = {{{0, 10}}};
  ret = dataElementwiseImpl.InferAxisSlice(op_proxy, axis_type_info, out_data_slice_wrong, in_data_slice);
  EXPECT_EQ(ret, FAILED);
}

TEST_F(DataSlice, ValidateAxisIndex_failed) {
  int64_t from_axis = 1;
  const std::vector<std::vector<int64_t>> slice_info;
  int64_t to_axis = 0;
  const std::vector<std::vector<int64_t>> cur_tensor_range = {{1}};
  EXPECT_EQ(false, DataSliceAdapter::ValidateAxisIndex(from_axis, slice_info, to_axis, cur_tensor_range));
  int64_t from_axis_new = 0;
  const std::vector<std::vector<int64_t>> slice_info_new = {{1}};
  EXPECT_EQ(false, DataSliceAdapter::ValidateAxisIndex(from_axis_new, slice_info_new, to_axis, cur_tensor_range));
}

TEST_F(DataSlice, Cov_PrintOp) {
  OpDescPtr op_desc = std::make_shared<OpDesc>("Add", "Add");
  GeTensorDesc output_desc(ge::GeShape({10, 3, 20, 30, 16}), ge::Format::FORMAT_NC1HWC0);
  output_desc.SetOriginShape(ge::GeShape({10, 20}));
  output_desc.SetOriginFormat(ge::Format::FORMAT_NHWC);
  ge::AttrUtils::SetStr(output_desc, ge::ATTR_NAME_RESHAPE_INFER_TYPE, "NH");
  op_desc->AddOutputDesc("output0", output_desc);
  GeTensorDesc input_desc0(ge::GeShape({10, 3, 20, 30, 16}), ge::Format::FORMAT_NC1HWC0);
  input_desc0.SetOriginShape(ge::GeShape({10, 20, 30, 40}));
  input_desc0.SetOriginFormat(ge::Format::FORMAT_NHWC);
  ge::AttrUtils::SetStr(input_desc0, ge::ATTR_NAME_RESHAPE_INFER_TYPE, "NH");
  op_desc->AddInputDesc("input0", input_desc0);
  EXPECT_NO_THROW(DataSliceAdapter::PrintOp(op_desc));
}

TEST_F(DataSlice, Cov_PrintAxis_WithOri) {
  OpDescPtr op_desc = std::make_shared<OpDesc>("Add", "Add");
  op_desc->AddOutputDesc("output", GeTensorDesc());
  AxisTypeInfo info;
  info.SetAxisType(AxisType::ELEMENTWISE);
  info.SetRelateInputs({{0, {0}}});
  info.SetRelateOutputs({{0, {0}}});
  info.SetOriRelateInputs({{0, {1}}});
  info.SetOriRelateOutputs({{0, {1}}});
  EXPECT_NO_THROW(DataSliceAdapter::PrintAxis(op_desc, {info}, "test_type", true));
}

TEST_F(DataSlice, Cov_PrintAxis_WithoutOri) {
  OpDescPtr op_desc = std::make_shared<OpDesc>("Add", "Add");
  op_desc->AddOutputDesc("output", GeTensorDesc());
  AxisTypeInfo info;
  info.SetAxisType(AxisType::ELEMENTWISE);
  info.SetRelateInputs({{0, {0}}});
  info.SetRelateOutputs({{0, {0}}});
  EXPECT_NO_THROW(DataSliceAdapter::PrintAxis(op_desc, {info}, "test_type", false));
}

TEST_F(DataSlice, Cov_PrintSlice) {
  OpDescPtr op_desc = std::make_shared<OpDesc>("Add", "Add");
  op_desc->AddOutputDesc("output", GeTensorDesc());
  DataSliceAdapter::DataSliceType slice_info = {{{0, 10}, {20, 30}}, {{0, 5}}};
  EXPECT_NO_THROW(DataSliceAdapter::PrintSlice(op_desc, slice_info, "input", "test_tag"));
}

TEST_F(DataSlice, Cov_PrintSlice_Empty) {
  OpDescPtr op_desc = std::make_shared<OpDesc>("Add", "Add");
  op_desc->AddOutputDesc("output", GeTensorDesc());
  DataSliceAdapter::DataSliceType slice_info;
  EXPECT_NO_THROW(DataSliceAdapter::PrintSlice(op_desc, slice_info, "output", "empty_tag"));
}

TEST_F(DataSlice, Cov_CheckOriInfo_True) {
  OpDescPtr op_desc = std::make_shared<OpDesc>("Add", "Add");
  GeTensorDesc output_desc(ge::GeShape({10, 3, 20, 30, 16}), ge::Format::FORMAT_NC1HWC0);
  output_desc.SetOriginShape(ge::GeShape({10, 20, 30, 40}));
  output_desc.SetOriginFormat(ge::Format::FORMAT_NHWC);
  op_desc->AddOutputDesc("output0", output_desc);
  GeTensorDesc input_desc0(ge::GeShape({10, 3, 20, 30, 16}), ge::Format::FORMAT_NC1HWC0);
  input_desc0.SetOriginShape(ge::GeShape({10, 20, 30, 40}));
  input_desc0.SetOriginFormat(ge::Format::FORMAT_NHWC);
  op_desc->AddInputDesc("input0", input_desc0);
  EXPECT_TRUE(DataSliceAdapter::CheckOriInfo(op_desc));
}

TEST_F(DataSlice, Cov_CheckOriInfo_False_NoOrigin) {
  OpDescPtr op_desc = std::make_shared<OpDesc>("Add", "Add");
  GeTensorDesc output_desc(ge::GeShape({10, 3, 20, 30, 16}), ge::Format::FORMAT_NC1HWC0);
  op_desc->AddOutputDesc("output0", output_desc);
  GeTensorDesc input_desc0(ge::GeShape({10, 3, 20, 30, 16}), ge::Format::FORMAT_NC1HWC0);
  op_desc->AddInputDesc("input0", input_desc0);
  EXPECT_FALSE(DataSliceAdapter::CheckOriInfo(op_desc));
}

TEST_F(DataSlice, Cov_TransAxisInfo_Elementwise) {
  OpDescPtr op_desc = std::make_shared<OpDesc>("Add", "Add");
  GeTensorDesc output_desc(ge::GeShape({10, 3, 20, 30, 16}), ge::Format::FORMAT_NC1HWC0);
  output_desc.SetOriginShape(ge::GeShape({10, 20, 30, 40}));
  output_desc.SetOriginFormat(ge::Format::FORMAT_NHWC);
  op_desc->AddOutputDesc("output0", output_desc);
  GeTensorDesc input_desc0(ge::GeShape({10, 3, 20, 30, 16}), ge::Format::FORMAT_NC1HWC0);
  input_desc0.SetOriginShape(ge::GeShape({10, 20, 30, 40}));
  input_desc0.SetOriginFormat(ge::Format::FORMAT_NHWC);
  op_desc->AddInputDesc("input0", input_desc0);
  std::vector<AxisTypeInfo> axis_type_vec;
  AxisTypeInfo info;
  info.SetAxisType(AxisType::ELEMENTWISE);
  info.SetRelateInputs({{0, {1}}});
  info.SetRelateOutputs({{0, {1}}});
  axis_type_vec.push_back(info);
  EXPECT_NO_THROW(DataSliceAdapter::TransAxisInfo(op_desc, axis_type_vec));
}

TEST_F(DataSlice, Cov_TransAxisInfo_ReduceMean) {
  OpDescPtr op_desc = std::make_shared<OpDesc>("Add", "Add");
  GeTensorDesc output_desc(ge::GeShape({10, 3, 20, 30, 16}), ge::Format::FORMAT_NC1HWC0);
  output_desc.SetOriginShape(ge::GeShape({10, 20, 30, 40}));
  output_desc.SetOriginFormat(ge::Format::FORMAT_NHWC);
  op_desc->AddOutputDesc("output0", output_desc);
  GeTensorDesc input_desc0(ge::GeShape({10, 3, 20, 30, 16}), ge::Format::FORMAT_NC1HWC0);
  input_desc0.SetOriginShape(ge::GeShape({10, 20, 30, 40}));
  input_desc0.SetOriginFormat(ge::Format::FORMAT_NHWC);
  op_desc->AddInputDesc("input0", input_desc0);
  std::vector<AxisTypeInfo> axis_type_vec;
  AxisTypeInfo info;
  info.SetAxisType(AxisType::REDUCEMEAN);
  info.SetRelateInputs({{0, {1}}});
  info.SetRelateOutputs({{0, {1}}});
  axis_type_vec.push_back(info);
  EXPECT_NO_THROW(DataSliceAdapter::TransAxisInfo(op_desc, axis_type_vec));
}

TEST_F(DataSlice, Cov_TransAxisInfo_SlidingWindow) {
  OpDescPtr op_desc = std::make_shared<OpDesc>("Add", "Add");
  GeTensorDesc output_desc(ge::GeShape({10, 3, 20, 30, 16}), ge::Format::FORMAT_NC1HWC0);
  output_desc.SetOriginShape(ge::GeShape({10, 20, 30, 40}));
  output_desc.SetOriginFormat(ge::Format::FORMAT_NHWC);
  op_desc->AddOutputDesc("output0", output_desc);
  GeTensorDesc input_desc0(ge::GeShape({10, 3, 20, 30, 16}), ge::Format::FORMAT_NC1HWC0);
  input_desc0.SetOriginShape(ge::GeShape({10, 20, 30, 40}));
  input_desc0.SetOriginFormat(ge::Format::FORMAT_NHWC);
  op_desc->AddInputDesc("input0", input_desc0);
  std::vector<AxisTypeInfo> axis_type_vec;
  AxisTypeInfo info;
  info.SetAxisType(AxisType::SLIDINGWINDOW);
  info.SetRelateInputs({{0, {1}}});
  info.SetRelateOutputs({{0, {1}}});
  axis_type_vec.push_back(info);
  EXPECT_NO_THROW(DataSliceAdapter::TransAxisInfo(op_desc, axis_type_vec));
}

TEST_F(DataSlice, Cov_TransAxisInfo_Unsplit) {
  OpDescPtr op_desc = std::make_shared<OpDesc>("Add", "Add");
  GeTensorDesc output_desc(ge::GeShape({10, 3, 20, 30, 16}), ge::Format::FORMAT_NC1HWC0);
  output_desc.SetOriginShape(ge::GeShape({10, 20, 30, 40}));
  output_desc.SetOriginFormat(ge::Format::FORMAT_NHWC);
  op_desc->AddOutputDesc("output0", output_desc);
  GeTensorDesc input_desc0(ge::GeShape({10, 3, 20, 30, 16}), ge::Format::FORMAT_NC1HWC0);
  input_desc0.SetOriginShape(ge::GeShape({10, 20, 30, 40}));
  input_desc0.SetOriginFormat(ge::Format::FORMAT_NHWC);
  op_desc->AddInputDesc("input0", input_desc0);
  std::vector<AxisTypeInfo> axis_type_vec;
  AxisTypeInfo info;
  info.SetAxisType(AxisType::UNSPLIT);
  info.SetRelateInputs({{0, {1}}});
  info.SetRelateOutputs({{0, {1}}});
  axis_type_vec.push_back(info);
  EXPECT_NO_THROW(DataSliceAdapter::TransAxisInfo(op_desc, axis_type_vec));
}

TEST_F(DataSlice, Cov_TransAxisInfo_UnknownType) {
  OpDescPtr op_desc = std::make_shared<OpDesc>("Add", "Add");
  GeTensorDesc output_desc(ge::GeShape({10, 3, 20, 30, 16}), ge::Format::FORMAT_NC1HWC0);
  output_desc.SetOriginShape(ge::GeShape({10, 20, 30, 40}));
  output_desc.SetOriginFormat(ge::Format::FORMAT_NHWC);
  op_desc->AddOutputDesc("output0", output_desc);
  GeTensorDesc input_desc0(ge::GeShape({10, 3, 20, 30, 16}), ge::Format::FORMAT_NC1HWC0);
  input_desc0.SetOriginShape(ge::GeShape({10, 20, 30, 40}));
  input_desc0.SetOriginFormat(ge::Format::FORMAT_NHWC);
  op_desc->AddInputDesc("input0", input_desc0);
  std::vector<AxisTypeInfo> axis_type_vec;
  AxisTypeInfo info;
  info.SetAxisType(static_cast<AxisType>(999));
  info.SetRelateInputs({{0, {1}}});
  info.SetRelateOutputs({{0, {1}}});
  axis_type_vec.push_back(info);
  EXPECT_NO_THROW(DataSliceAdapter::TransAxisInfo(op_desc, axis_type_vec));
}

TEST_F(DataSlice, Cov_GetOriOutputSlice) {
  OpDescPtr op_desc = std::make_shared<OpDesc>("Add", "Add");
  GeTensorDesc output_desc(ge::GeShape({10, 3, 20, 30, 16}), ge::Format::FORMAT_NC1HWC0);
  output_desc.SetOriginShape(ge::GeShape({10, 20, 30, 40}));
  output_desc.SetOriginFormat(ge::Format::FORMAT_NHWC);
  op_desc->AddOutputDesc("output0", output_desc);
  AxisTypeInfo info;
  info.SetAxisType(AxisType::ELEMENTWISE);
  info.SetOriRelateInputs({{0, {0}}});
  info.SetOriRelateOutputs({{0, {0}}});
  DataSliceAdapter::DataSliceType ori_output_slice;
  auto ret = DataSliceAdapter::GetOriOutputSlice(op_desc, info, ori_output_slice);
  EXPECT_EQ(ret, SUCCESS);
}

TEST_F(DataSlice, Cov_GetCurInputSlice) {
  OpDescPtr op_desc = std::make_shared<OpDesc>("Add", "Add");
  GeTensorDesc input_desc0(ge::GeShape({10, 3, 20, 30, 16}), ge::Format::FORMAT_NC1HWC0);
  input_desc0.SetOriginShape(ge::GeShape({10, 20, 30, 40}));
  input_desc0.SetOriginFormat(ge::Format::FORMAT_NHWC);
  op_desc->AddInputDesc("input0", input_desc0);
  AxisTypeInfo info;
  info.SetAxisType(AxisType::ELEMENTWISE);
  info.SetOriRelateInputs({{0, {0}}});
  info.SetOriRelateOutputs({{0, {0}}});
  DataSliceAdapter::DataSliceType ori_input_slice;
  DataSliceAdapter::DataSliceType cur_input_slice;
  auto ret = DataSliceAdapter::GetCurInputSlice(op_desc, info, ori_input_slice, cur_input_slice);
  EXPECT_EQ(ret, SUCCESS);
}

TEST_F(DataSlice, Cov_SetOriOpInfoAndSetCurOpInfo) {
  OpDescPtr op_desc = std::make_shared<OpDesc>("Add", "Add");
  GeTensorDesc output_desc(ge::GeShape({10, 3, 20, 30, 16}), ge::Format::FORMAT_NC1HWC0);
  output_desc.SetOriginShape(ge::GeShape({10, 20, 30, 40}));
  output_desc.SetOriginFormat(ge::Format::FORMAT_NHWC);
  op_desc->AddOutputDesc("output0", output_desc);
  GeTensorDesc input_desc0(ge::GeShape({10, 3, 20, 30, 16}), ge::Format::FORMAT_NC1HWC0);
  input_desc0.SetOriginShape(ge::GeShape({10, 20, 30, 40}));
  input_desc0.SetOriginFormat(ge::Format::FORMAT_NHWC);
  op_desc->AddInputDesc("input0", input_desc0);
  std::vector<std::pair<Format, GeShape>> cache_input_info;
  std::vector<std::pair<Format, GeShape>> cache_output_info;
  EXPECT_NO_THROW(DataSliceAdapter::SetOriOpInfo(op_desc, cache_input_info, cache_output_info));
  EXPECT_NO_THROW(DataSliceAdapter::SetCurOpInfo(op_desc, cache_input_info, cache_output_info));
}

TEST_F(DataSlice, Cov_GetTmpAxisTypeInfo) {
  AxisTypeInfo info;
  info.SetAxisType(AxisType::ELEMENTWISE);
  info.SetRelateInputs({{0, {0}}});
  info.SetRelateOutputs({{0, {0}}});
  info.SetOriRelateInputs({{0, {1}}});
  info.SetOriRelateOutputs({{0, {1}}});
  auto tmp = DataSliceAdapter::GetTmpAxisTypeInfo(info);
  EXPECT_EQ(tmp.GetAxisType(), AxisType::ELEMENTWISE);
}

TEST_F(DataSlice, Cov_DataSliceHelper_InferAxisSlice_Unsplit) {
  OpDescPtr op_desc = std::make_shared<OpDesc>("Cast", "Cast");
  op_desc->AddOutputDesc("output", GeTensorDesc());
  op_desc->AddInputDesc("input", GeTensorDesc());
  AxisTypeInfo axis_type_info;
  axis_type_info.SetAxisType(AxisType::UNSPLIT);
  CutInfo input_cut_info{0, {0}};
  axis_type_info.AddInputCutInfo(input_cut_info);
  CutInfo output_cut_info{0, {0}};
  axis_type_info.AddOutputCutInfo(output_cut_info);
  Status ret = DataSliceHelper::InferAxisSlice(op_desc, axis_type_info);
  EXPECT_EQ(ret, SUCCESS);
}

TEST_F(DataSlice, Cov_DataSliceElementwiseImpl_EmptyOutput) {
  OpDescPtr op_desc = std::make_shared<OpDesc>("Cast", "Cast");
  op_desc->AddOutputDesc("output", GeTensorDesc());
  op_desc->AddInputDesc("input", GeTensorDesc());
  AxisTypeInfo axis_type_info;
  axis_type_info.SetAxisType(AxisType::ELEMENTWISE);
  CutInfo input_cut_info{0, {0}};
  axis_type_info.AddInputCutInfo(input_cut_info);
  CutInfo output_cut_info{0, {0}};
  axis_type_info.AddOutputCutInfo(output_cut_info);
  DataSliceType out_data_slice;
  DataSliceType in_data_slice;
  Operator op_proxy = OpDescUtils::CreateOperatorFromOpDesc(op_desc);
  DataSliceElementwiseImpl impl;
  auto ret = impl.InferAxisSlice(op_proxy, axis_type_info, out_data_slice, in_data_slice);
  EXPECT_NE(ret, SUCCESS);
}

TEST_F(DataSlice, Cov_DataSliceElementwiseImpl_NullOpDesc) {
  OpDescPtr null_op_desc = nullptr;
  AxisTypeInfo axis_type_info;
  DataSliceType out_data_slice;
  DataSliceType in_data_slice;
  DataSliceElementwiseImpl impl;
  Operator op_proxy;
  auto ret = impl.InferAxisSlice(op_proxy, axis_type_info, out_data_slice, in_data_slice);
  EXPECT_NE(ret, SUCCESS);
}

TEST_F(DataSlice, Cov_TransAxis_NDC1HWC0) {
  auto tensor = std::make_shared<GeTensorDesc>(GeShape({1, 2, 3, 4, 5, 16}), FORMAT_NDC1HWC0);
  tensor->SetOriginFormat(FORMAT_NCHW);
  tensor->SetOriginShape(GeShape({1, 2, 3, 4, 5}));
  auto result = DataSliceAdapter::TransAxis(tensor, 0);
  EXPECT_FALSE(result.empty());
}

TEST_F(DataSlice, Cov_TransAxis_5DSet) {
  auto tensor = std::make_shared<GeTensorDesc>(GeShape({1, 2, 3, 4, 5}), FORMAT_NCDHW);
  tensor->SetOriginFormat(FORMAT_NCHW);
  tensor->SetOriginShape(GeShape({1, 2, 3, 4, 5}));
  auto result = DataSliceAdapter::TransAxis(tensor, 0);
  EXPECT_FALSE(result.empty());
}

TEST_F(DataSlice, Cov_TransAxis_UnsupportedFormat) {
  auto tensor = std::make_shared<GeTensorDesc>(GeShape({1, 2}), FORMAT_ND);
  tensor->SetOriginFormat(FORMAT_NCHW);
  tensor->SetOriginShape(GeShape({1, 2, 3, 4}));
  auto result = DataSliceAdapter::TransAxis(tensor, 0);
  EXPECT_TRUE(result.empty());
}

TEST_F(DataSlice, Cov_TransAxisForSplit_CheckRankFail) {
  auto tensor = std::make_shared<GeTensorDesc>(GeShape({1, 2, 3, 4, 5, 6}), FORMAT_NC1HWC0);
  tensor->SetOriginFormat(FORMAT_NCHW);
  tensor->SetOriginShape(GeShape({1, 2, 3, 4, 5, 6, 7, 8}));
  auto result = DataSliceAdapter::TransAxisForSplit(tensor, 0, 4U);
  EXPECT_TRUE(result.empty());
}

TEST_F(DataSlice, Cov_TransAxisForSplit_FormatNotFound) {
  auto tensor = std::make_shared<GeTensorDesc>(GeShape({1, 2, 3, 4}), FORMAT_NC1HWC0);
  tensor->SetOriginFormat(FORMAT_RESERVED);
  tensor->SetOriginShape(GeShape({1, 2, 3, 4}));
  auto result = DataSliceAdapter::TransAxisForSplit(tensor, 0, 4U);
  EXPECT_TRUE(result.empty());
}

TEST_F(DataSlice, Cov_TransAxisForNoSplit_RankNotEqualDimNum) {
  auto tensor = std::make_shared<GeTensorDesc>(GeShape({1, 2, 3, 4, 5}), FORMAT_NCHW);
  tensor->SetOriginFormat(FORMAT_NCHW);
  tensor->SetOriginShape(GeShape({1, 2, 3, 4, 5}));
  auto result = DataSliceAdapter::TransAxisForNoSplit(tensor, 0, 4U);
  EXPECT_TRUE(result.empty());
}

TEST_F(DataSlice, Cov_TransAxisForNoSplit_OriFormatNotFound) {
  auto tensor = std::make_shared<GeTensorDesc>(GeShape({1, 2, 3, 4}), FORMAT_NCHW);
  tensor->SetOriginFormat(FORMAT_RESERVED);
  tensor->SetOriginShape(GeShape({1, 2, 3, 4}));
  auto result = DataSliceAdapter::TransAxisForNoSplit(tensor, 0, 4U);
  EXPECT_TRUE(result.empty());
}

TEST_F(DataSlice, Cov_TransAxisForNoSplit_AxisOutOfRange) {
  auto tensor = std::make_shared<GeTensorDesc>(GeShape({1, 2, 3, 4}), FORMAT_NCHW);
  tensor->SetOriginFormat(FORMAT_NCHW);
  tensor->SetOriginShape(GeShape({1, 2, 3, 4}));
  auto result = DataSliceAdapter::TransAxisForNoSplit(tensor, 10, 4U);
  EXPECT_TRUE(result.empty());
}

TEST_F(DataSlice, Cov_TransAxisForNoSplit_FormatNotFound) {
  auto tensor = std::make_shared<GeTensorDesc>(GeShape({1, 2, 3, 4}), FORMAT_RESERVED);
  tensor->SetOriginFormat(FORMAT_NCHW);
  tensor->SetOriginShape(GeShape({1, 2, 3, 4}));
  auto result = DataSliceAdapter::TransAxisForNoSplit(tensor, 0, 4U);
  EXPECT_TRUE(result.empty());
}

TEST_F(DataSlice, Cov_FixAxisTypeInfoToOne_DiffOutputSize) {
  AxisTypeInfo axis_type_info;
  axis_type_info.SetAxisType(AxisType::REDUCESUM);
  axis_type_info.SetRelateInputs({{0, {0}}});
  axis_type_info.SetRelateOutputs({{0, {0, 1}}, {1, {0}}});
  EXPECT_EQ(DataSliceAdapter::FixAxisTypeInfoToOne(axis_type_info), FAILED);
}

TEST_F(DataSlice, Cov_FixAxisTypeInfoToOne_ReduceEmpty) {
  AxisTypeInfo axis_type_info;
  axis_type_info.SetAxisType(AxisType::REDUCESUM);
  axis_type_info.SetRelateInputs({{0, {0}}});
  axis_type_info.SetRelateOutputs({{0, {}}, {1, {0}}});
  EXPECT_EQ(DataSliceAdapter::FixAxisTypeInfoToOne(axis_type_info), SUCCESS);
}

TEST_F(DataSlice, Cov_TransAxisForInputTensor_NullTensor) {
  OpDescPtr op_desc = std::make_shared<OpDesc>("test", "Add");
  op_desc->AddInputDesc("input", GeTensorDesc());
  AxisTypeInfo axis_type_info;
  axis_type_info.SetRelateInputs({{5, {0}}});
  EXPECT_EQ(DataSliceAdapter::TransAxisForInputTensor(op_desc, "element_type", axis_type_info), FAILED);
}

TEST_F(DataSlice, Cov_TransAxisForOutputTensor_NullTensor) {
  OpDescPtr op_desc = std::make_shared<OpDesc>("test", "Add");
  op_desc->AddOutputDesc("output", GeTensorDesc());
  AxisTypeInfo axis_type_info;
  axis_type_info.SetRelateOutputs({{5, {0}}});
  EXPECT_EQ(DataSliceAdapter::TransAxisForOutputTensor(op_desc, "element_type", axis_type_info), FAILED);
}

TEST_F(DataSlice, Cov_TransAxisByType_ValidateFail) {
  OpDescPtr op_desc = std::make_shared<OpDesc>("test", "Add");
  AxisTypeInfo axis_type_info;
  EXPECT_EQ(DataSliceAdapter::TransAxisByType(AxisType::ELEMENTWISE, op_desc, axis_type_info), FAILED);
}

TEST_F(DataSlice, Cov_GetAxisTypeForTransAxis_MultiType) {
  AxisTypeInfo axis_type_info1;
  axis_type_info1.SetAxisTypes({AxisType::ELEMENTWISE, AxisType::REDUCESUM, AxisType::REDUCEMAX});
  EXPECT_EQ(DataSliceAdapter::GetAxisTypeForTransAxis(axis_type_info1), AxisType::UNSPLIT);
  AxisTypeInfo axis_type_info2;
  axis_type_info2.SetAxisTypes({AxisType::ELEMENTWISE, AxisType::REDUCESUM});
  EXPECT_EQ(DataSliceAdapter::GetAxisTypeForTransAxis(axis_type_info2), AxisType::SLIDINGWINDOW);
}

TEST_F(DataSlice, Cov_GetAxisTypeForTransSlice_MultiType) {
  AxisTypeInfo axis_type_info;
  axis_type_info.SetAxisTypes({AxisType::ELEMENTWISE, AxisType::REDUCESUM, AxisType::REDUCEMAX});
  EXPECT_EQ(DataSliceAdapter::GetAxisTypeForTransSlice(axis_type_info), AxisType::UNSPLIT);
}

TEST_F(DataSlice, Cov_TransSliceInfo_UnsupportedType) {
  OpDescPtr op_desc = std::make_shared<OpDesc>("test", "Add");
  AxisTypeInfo axis_type_info;
  axis_type_info.SetAxisTypes({AxisType::UNSPLIT});
  DataSliceAdapter::DataSliceType slice_info;
  DataSliceAdapter::DataSliceType out_slice;
  EXPECT_EQ(DataSliceAdapter::TransSliceInfo(op_desc, axis_type_info, TransType::CUR_TO_ORI, slice_info, out_slice),
            FAILED);
}

TEST_F(DataSlice, Cov_TransSliceInfo_SlidingWindow) {
  OpDescPtr op_desc = std::make_shared<OpDesc>("test", "Add");
  AxisTypeInfo axis_type_info;
  axis_type_info.SetAxisTypes({AxisType::SLIDINGWINDOW});
  DataSliceAdapter::DataSliceType slice_info = {{{0, 10}}};
  DataSliceAdapter::DataSliceType out_slice;
  EXPECT_EQ(DataSliceAdapter::TransSliceInfo(op_desc, axis_type_info, TransType::CUR_TO_ORI, slice_info, out_slice),
            SUCCESS);
  EXPECT_EQ(out_slice.size(), 1U);
}

TEST_F(DataSlice, Cov_TransSliceInfoToOri_EmptyOriOutputs) {
  OpDescPtr op_desc = std::make_shared<OpDesc>("test", "Add");
  AxisTypeInfo axis_type_info;
  DataSliceAdapter::DataSliceType slice_info = {{{0, 10}}};
  DataSliceAdapter::DataSliceType out_slice;
  EXPECT_EQ(DataSliceAdapter::TransSliceInfoToOriForElement(op_desc, axis_type_info, slice_info, out_slice), FAILED);
}

TEST_F(DataSlice, Cov_TransSliceInfoToCur_EmptyOriInputs) {
  OpDescPtr op_desc = std::make_shared<OpDesc>("test", "Add");
  AxisTypeInfo axis_type_info;
  DataSliceAdapter::DataSliceType slice_info = {{{0, 10}}};
  DataSliceAdapter::DataSliceType out_slice;
  EXPECT_EQ(DataSliceAdapter::TransSliceInfoToCurForElement(op_desc, axis_type_info, slice_info, out_slice), FAILED);
}

TEST_F(DataSlice, Cov_ValidateRelateInputOutput_False) {
  AxisTypeInfo axis_type_info;
  EXPECT_FALSE(DataSliceAdapter::ValidateRelateInputOutput(axis_type_info));
}

TEST_F(DataSlice, Cov_SetInputSlice_TensorIdxOutOfRange) {
  OpDescPtr op_desc = std::make_shared<OpDesc>("Add", "Add");
  op_desc->AddOutputDesc("output", GeTensorDesc());
  op_desc->AddInputDesc("input", GeTensorDesc());
  AxisTypeInfo axis_type_info;
  axis_type_info.SetAxisType(AxisType::ELEMENTWISE);
  axis_type_info.SetRelateInputs({{0, {0}}, {1, {0}}, {2, {0}}, {5, {0}}});
  axis_type_info.SetRelateOutputs({{0, {0}}});
  Status ret = DataSliceHelper::InferAxisSlice(op_desc, axis_type_info);
  EXPECT_EQ(ret, SUCCESS);
}

TEST_F(DataSlice, Cov_GetSliceInfo_NoAxisSliceFunc) {
  OpDescPtr op_desc = std::make_shared<OpDesc>("UnknownOp", "UnknownOpType");
  op_desc->AddOutputDesc("output", GeTensorDesc());
  op_desc->AddInputDesc("input", GeTensorDesc());
  std::vector<AxisTypeInfo> axis_type_info;
  Status ret = DataSliceHelper::GetSliceInfo(op_desc, axis_type_info);
  EXPECT_EQ(ret, FAILED);
}

TEST_F(DataSlice, Cov_GetDavinciSliceInfo_NoAxisSliceFunc) {
  ComputeGraphPtr test_graph = std::make_shared<ComputeGraph>("test_graph");
  OpDescPtr op_desc = std::make_shared<OpDesc>("UnknownOp2", "UnknownOpType2");
  op_desc->AddOutputDesc("output", GeTensorDesc());
  op_desc->AddInputDesc("input", GeTensorDesc());
  NodePtr node = test_graph->AddNode(op_desc);
  std::vector<AxisTypeInfo> axis_type_info;
  Status ret = DataSliceHelper::GetDavinciSliceInfo(node, axis_type_info);
  EXPECT_EQ(ret, FAILED);
}

TEST_F(DataSlice, Cov_GetDavinciSliceInfo_GetAxisSliceFail) {
  ComputeGraphPtr test_graph = std::make_shared<ComputeGraph>("test_graph");
  OpDescPtr op_desc = std::make_shared<OpDesc>("Softmax", "Softmax");
  GeTensorDesc output_desc(GeShape({10, 20}), FORMAT_NCHW);
  output_desc.SetOriginShape(GeShape({10, 20}));
  output_desc.SetOriginFormat(FORMAT_NCHW);
  op_desc->AddOutputDesc("output", output_desc);
  GeTensorDesc input_desc(GeShape({10, 20}), FORMAT_NCHW);
  input_desc.SetOriginShape(GeShape({10, 20}));
  input_desc.SetOriginFormat(FORMAT_NCHW);
  op_desc->AddInputDesc("input", input_desc);
  NodePtr node = test_graph->AddNode(op_desc);
  std::vector<AxisTypeInfo> axis_type_info;
  Status ret = DataSliceHelper::GetDavinciSliceInfo(node, axis_type_info);
  EXPECT_EQ(ret, FAILED);
}

TEST_F(DataSlice, Cov_InferDavinciSpecialOpSlice_GetOriOutputFail) {
  OpDescPtr op_desc = std::make_shared<OpDesc>("Add", "Add");
  GeTensorDesc output_desc(GeShape({10, 20}), FORMAT_NCHW);
  output_desc.SetOriginShape(GeShape({10, 20}));
  output_desc.SetOriginFormat(FORMAT_NCHW);
  op_desc->AddOutputDesc("output", output_desc);
  GeTensorDesc input_desc(GeShape({10, 20}), FORMAT_NCHW);
  input_desc.SetOriginShape(GeShape({10, 20}));
  input_desc.SetOriginFormat(FORMAT_NCHW);
  op_desc->AddInputDesc("input", input_desc);
  AxisTypeInfo axis_type_info;
  axis_type_info.SetAxisTypes({AxisType::UNSPLIT});
  axis_type_info.SetAxisType(AxisType::UNSPLIT);
  axis_type_info.SetRelateInputs({{0, {0}}});
  axis_type_info.SetRelateOutputs({{0, {0}}});
  axis_type_info.SetOriRelateInputs({{0, {0}}});
  axis_type_info.SetOriRelateOutputs({{0, {0}}});
  Status ret = DataSliceHelper::InferDavinciAxisSlice(op_desc, axis_type_info);
  EXPECT_EQ(ret, FAILED);
}
TEST_F(DataSlice, Cov_TransAxisInfo_NZFormat) {
  OpDescPtr op_desc = std::make_shared<OpDesc>("Add", "Add");
  GeTensorDesc input_desc(GeShape({2, 2, 3, 4}), FORMAT_FRACTAL_NZ);
  input_desc.SetOriginShape(GeShape({2, 2, 3, 4}));
  input_desc.SetOriginFormat(FORMAT_NCHW);
  op_desc->AddInputDesc("input", input_desc);
  GeTensorDesc output_desc(GeShape({2, 2, 3, 4}), FORMAT_FRACTAL_NZ);
  output_desc.SetOriginShape(GeShape({2, 2, 3, 4}));
  output_desc.SetOriginFormat(FORMAT_NCHW);
  op_desc->AddOutputDesc("output", output_desc);
  AxisTypeInfo axis_type_info;
  axis_type_info.SetAxisTypes({AxisType::ELEMENTWISE});
  axis_type_info.SetAxisType(AxisType::ELEMENTWISE);
  axis_type_info.SetRelateInputs({{0, {0}}});
  axis_type_info.SetRelateOutputs({{0, {0}}});
  std::vector<AxisTypeInfo> axis_type_vec = {axis_type_info};
  DataSliceAdapter::TransAxisInfo(op_desc, axis_type_vec);
  EXPECT_EQ(axis_type_vec.size(), 1U);
}

TEST_F(DataSlice, Cov_TransAxisInfo_NC1HWC0Format) {
  OpDescPtr op_desc = std::make_shared<OpDesc>("Add", "Add");
  GeTensorDesc input_desc(GeShape({2, 1, 3, 4, 16}), FORMAT_NC1HWC0);
  input_desc.SetOriginShape(GeShape({2, 16, 3, 4}));
  input_desc.SetOriginFormat(FORMAT_NCHW);
  op_desc->AddInputDesc("input", input_desc);
  GeTensorDesc output_desc(GeShape({2, 1, 3, 4, 16}), FORMAT_NC1HWC0);
  output_desc.SetOriginShape(GeShape({2, 16, 3, 4}));
  output_desc.SetOriginFormat(FORMAT_NCHW);
  op_desc->AddOutputDesc("output", output_desc);
  AxisTypeInfo axis_type_info;
  axis_type_info.SetAxisTypes({AxisType::ELEMENTWISE});
  axis_type_info.SetAxisType(AxisType::ELEMENTWISE);
  axis_type_info.SetRelateInputs({{0, {1}}});
  axis_type_info.SetRelateOutputs({{0, {1}}});
  std::vector<AxisTypeInfo> axis_type_vec = {axis_type_info};
  DataSliceAdapter::TransAxisInfo(op_desc, axis_type_vec);
  EXPECT_EQ(axis_type_vec.size(), 1U);
}

TEST_F(DataSlice, Cov_TransAxisInfo_UnsupportedType) {
  OpDescPtr op_desc = std::make_shared<OpDesc>("Add", "Add");
  GeTensorDesc input_desc(GeShape({2, 2, 3, 4}), FORMAT_NCHW);
  input_desc.SetOriginShape(GeShape({2, 2, 3, 4}));
  input_desc.SetOriginFormat(FORMAT_NCHW);
  op_desc->AddInputDesc("input", input_desc);
  GeTensorDesc output_desc(GeShape({2, 2, 3, 4}), FORMAT_NCHW);
  output_desc.SetOriginShape(GeShape({2, 2, 3, 4}));
  output_desc.SetOriginFormat(FORMAT_NCHW);
  op_desc->AddOutputDesc("output", output_desc);
  AxisTypeInfo axis_type_info;
  axis_type_info.SetAxisTypes({AxisType::UNSPLIT});
  axis_type_info.SetAxisType(AxisType::UNSPLIT);
  axis_type_info.SetRelateInputs({{0, {0}}});
  axis_type_info.SetRelateOutputs({{0, {0}}});
  std::vector<AxisTypeInfo> axis_type_vec = {axis_type_info};
  DataSliceAdapter::TransAxisInfo(op_desc, axis_type_vec);
  EXPECT_TRUE(axis_type_vec.empty());
}

TEST_F(DataSlice, Cov_TransAxisInfo_SlidingWindowCombination) {
  OpDescPtr op_desc = std::make_shared<OpDesc>("Add", "Add");
  GeTensorDesc input_desc(GeShape({2, 2, 3, 4}), FORMAT_NCHW);
  input_desc.SetOriginShape(GeShape({2, 2, 3, 4}));
  input_desc.SetOriginFormat(FORMAT_NCHW);
  op_desc->AddInputDesc("input", input_desc);
  GeTensorDesc output_desc(GeShape({2, 2, 3, 4}), FORMAT_NCHW);
  output_desc.SetOriginShape(GeShape({2, 2, 3, 4}));
  output_desc.SetOriginFormat(FORMAT_NCHW);
  op_desc->AddOutputDesc("output", output_desc);
  AxisTypeInfo axis_type_info;
  axis_type_info.SetAxisTypes({AxisType::ELEMENTWISE, AxisType::REDUCESUM});
  axis_type_info.SetAxisType(AxisType::ELEMENTWISE);
  axis_type_info.SetRelateInputs({{0, {0}}});
  axis_type_info.SetRelateOutputs({{0, {0}}});
  std::vector<AxisTypeInfo> axis_type_vec = {axis_type_info};
  DataSliceAdapter::TransAxisInfo(op_desc, axis_type_vec);
  EXPECT_EQ(axis_type_vec.size(), 1U);
}

TEST_F(DataSlice, Cov_TransAxisInfo_ReduceTypeMultiAxis) {
  OpDescPtr op_desc = std::make_shared<OpDesc>("Add", "Add");
  GeTensorDesc input_desc(GeShape({2, 2, 3, 4}), FORMAT_FRACTAL_NZ);
  input_desc.SetOriginShape(GeShape({2, 2, 3, 4}));
  input_desc.SetOriginFormat(FORMAT_NCHW);
  op_desc->AddInputDesc("input", input_desc);
  GeTensorDesc output_desc(GeShape({2, 2, 3, 4}), FORMAT_FRACTAL_NZ);
  output_desc.SetOriginShape(GeShape({2, 2, 3, 4}));
  output_desc.SetOriginFormat(FORMAT_NCHW);
  op_desc->AddOutputDesc("output", output_desc);
  AxisTypeInfo axis_type_info;
  axis_type_info.SetAxisTypes({AxisType::REDUCEMEAN});
  axis_type_info.SetAxisType(AxisType::REDUCEMEAN);
  axis_type_info.SetRelateInputs({{0, {2}}});
  axis_type_info.SetRelateOutputs({{0, {2}}});
  std::vector<AxisTypeInfo> axis_type_vec = {axis_type_info};
  DataSliceAdapter::TransAxisInfo(op_desc, axis_type_vec);
  EXPECT_TRUE(axis_type_vec.empty());
}

TEST_F(DataSlice, Cov_GetDavinciSliceInfo_InvalidOriInfo) {
  ComputeGraphPtr test_graph = std::make_shared<ComputeGraph>("test_graph");
  OpDescPtr op_desc = std::make_shared<OpDesc>("Add", "Add");
  GeTensorDesc output_desc(GeShape({10, 20}), FORMAT_NCHW);
  output_desc.SetOriginShape(GeShape(std::vector<int64_t>{}));
  output_desc.SetOriginFormat(FORMAT_NCHW);
  op_desc->AddOutputDesc("output", output_desc);
  GeTensorDesc input_desc(GeShape({10, 20}), FORMAT_NCHW);
  input_desc.SetOriginShape(GeShape(std::vector<int64_t>{}));
  input_desc.SetOriginFormat(FORMAT_NCHW);
  op_desc->AddInputDesc("input", input_desc);
  NodePtr node = test_graph->AddNode(op_desc);
  std::vector<AxisTypeInfo> axis_type_info;
  Status ret = DataSliceHelper::GetDavinciSliceInfo(node, axis_type_info);
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_TRUE(axis_type_info.empty());
}

TEST_F(DataSlice, Cov_InferDavinciCommonOpSlice_NullPtr) {
  OpDescPtr op_desc = std::make_shared<OpDesc>("Cast", "Cast");
  GeTensorDesc output_desc(GeShape({10, 20}), FORMAT_NCHW);
  output_desc.SetOriginShape(GeShape({10, 20}));
  output_desc.SetOriginFormat(FORMAT_NCHW);
  op_desc->AddOutputDesc("output", output_desc);
  GeTensorDesc input_desc(GeShape({10, 20}), FORMAT_NCHW);
  input_desc.SetOriginShape(GeShape({10, 20}));
  input_desc.SetOriginFormat(FORMAT_NCHW);
  op_desc->AddInputDesc("input", input_desc);
  AxisTypeInfo axis_type_info;
  axis_type_info.SetAxisTypes({AxisType::SLIDINGWINDOW});
  axis_type_info.SetAxisType(AxisType::SLIDINGWINDOW);
  axis_type_info.SetRelateInputs({{0, {0}}});
  axis_type_info.SetRelateOutputs({{0, {0}}});
  axis_type_info.SetOriRelateInputs({{0, {0}}});
  axis_type_info.SetOriRelateOutputs({{0, {0}}});
  Status ret = DataSliceHelper::InferDavinciAxisSlice(op_desc, axis_type_info);
  EXPECT_EQ(ret, FAILED);
}
}  // namespace ge
