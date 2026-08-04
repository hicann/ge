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

#include "graph/utils/ge_ir_utils.h"
#include "graph/utils/attr_utils.h"
#include "graph/compute_graph.h"
#include "graph/op_desc.h"
#include "graph/utils/graph_utils.h"
#include "graph/model.h"
#include "graph/node.h"
#include "graph/ge_tensor.h"
#include "graph_builder_utils.h"
#include "graph/normal_graph/node_impl.h"
#include "graph/debug/ge_op_types.h"
#include "test_std_structs.h"

namespace ge {
class GeIrUtilsIncCov2 : public testing::Test {
 protected:
  void SetUp() {}

  void TearDown() {}
};

namespace {
NodePtr CreateNodeIncCov2Helper(const ComputeGraphPtr &graph, const string &name, const string &type, int in_num,
                                int out_num) {
  OpDescPtr op_desc = std::make_shared<OpDesc>(name, type);
  GeTensorDesc tensor(GeShape({1}), FORMAT_NCHW, DT_FLOAT);
  for (int i = 0; i < in_num; i++) {
    op_desc->AddInputDesc(tensor);
  }
  for (int i = 0; i < out_num; i++) {
    op_desc->AddOutputDesc(tensor);
  }
  return graph->AddNode(op_desc);
}

ComputeGraphPtr BuildIncCov2Graph() {
  ut::GraphBuilder builder("inc_cov2_graph");
  auto data = builder.AddNode("data1", "Data", 1, 1);
  auto add = builder.AddNode("add1", "Add", 2, 1);
  auto netoutput = builder.AddNode("netoutput1", "NetOutput", 1, 0);
  builder.AddDataEdge(data, 0, add, 0);
  builder.AddDataEdge(data, 0, add, 1);
  builder.AddDataEdge(add, 0, netoutput, 0);
  auto graph = builder.GetGraph();
  graph->AddInputNode(data);
  graph->AddOutputNode(netoutput);
  return graph;
}

ComputeGraphPtr BuildIncCov2GraphWithAttrs() {
  ut::GraphBuilder builder("inc_cov2_graph_attrs");
  auto data = builder.AddNode("data1", "Data", 1, 1);
  auto const1 = builder.AddNode("const1", "Const", 0, 1);
  auto add = builder.AddNode("add1", "Add", 2, 1);
  auto netoutput = builder.AddNode("netoutput1", "NetOutput", 1, 0);

  GeTensorDesc const_td(GeShape({1, 1, 224, 224}), FORMAT_NCHW, DT_FLOAT);
  GeTensor tensor(const_td);
  std::vector<float> tensor_data(224 * 224, 1.0F);
  tensor.SetData(reinterpret_cast<uint8_t *>(tensor_data.data()), sizeof(float) * tensor_data.size());
  AttrUtils::SetTensor(const1->GetOpDesc(), "value", tensor);

  AttrUtils::SetInt(add->GetOpDesc(), "int_attr", 42);
  AttrUtils::SetFloat(add->GetOpDesc(), "float_attr", 3.14F);
  AttrUtils::SetStr(add->GetOpDesc(), "str_attr", "hello");
  AttrUtils::SetBool(add->GetOpDesc(), "bool_attr", true);
  AttrUtils::SetListInt(add->GetOpDesc(), "list_int_attr", {1, 2, 3});
  AttrUtils::SetListFloat(add->GetOpDesc(), "list_float_attr", {1.0F, 2.0F});
  AttrUtils::SetListStr(add->GetOpDesc(), "list_str_attr", {"a", "b"});
  AttrUtils::SetListBool(add->GetOpDesc(), "list_bool_attr", {true, false});

  builder.AddDataEdge(data, 0, add, 0);
  builder.AddDataEdge(const1, 0, add, 1);
  builder.AddDataEdge(add, 0, netoutput, 0);
  auto graph = builder.GetGraph();
  graph->AddInputNode(data);
  graph->AddOutputNode(netoutput);
  return graph;
}

ComputeGraphPtr BuildIncCov2GraphWithSubgraph() {
  auto root_builder = ut::GraphBuilder("root_graph_inc2");
  auto parent = root_builder.AddNode("parent", PARTITIONEDCALL, 0, 1);
  auto netoutput = root_builder.AddNode("netoutput", NETOUTPUT, 1, 0);
  root_builder.AddDataEdge(parent, 0, netoutput, 0);
  auto root_graph = root_builder.GetGraph();

  auto sub_builder = ut::GraphBuilder("sub_graph_inc2");
  auto sub_const = sub_builder.AddNode("sub_const", "Const", 0, 1);
  auto sub_netoutput = sub_builder.AddNode("sub_netoutput", NETOUTPUT, 1, 0);
  sub_builder.AddDataEdge(sub_const, 0, sub_netoutput, 0);
  auto sub_graph = sub_builder.GetGraph();
  sub_graph->SetParentNode(parent);
  sub_graph->SetParentGraph(root_graph);
  parent->GetOpDesc()->AddSubgraphName("f");
  parent->GetOpDesc()->SetSubgraphInstanceName(0, "sub_graph_inc2");
  root_graph->AddSubGraph(sub_graph);

  return root_graph;
}
}  // namespace

TEST_F(GeIrUtilsIncCov2, IncCov2_AddAttrProto_NullNodeProto_RepeatedInt64) {
  ::google::protobuf::RepeatedField<::google::protobuf::int64> data;
  data.Add(1);
  data.Add(2);
  OnnxUtils::AddAttrProto(nullptr, onnx::AttributeProto_AttributeType_INTS, "test_attr", data);
  SUCCEED();
}

TEST_F(GeIrUtilsIncCov2, IncCov2_AddAttrProto_NullNodeProto_RepeatedBool) {
  ::google::protobuf::RepeatedField<bool> data;
  data.Add(true);
  data.Add(false);
  OnnxUtils::AddAttrProto(nullptr, onnx::AttributeProto_AttributeType_INTS, "test_attr", data);
  SUCCEED();
}

TEST_F(GeIrUtilsIncCov2, IncCov2_AddAttrProto_NullNodeProto_RepeatedFloat) {
  ::google::protobuf::RepeatedField<float> data;
  data.Add(1.0F);
  data.Add(2.0F);
  OnnxUtils::AddAttrProto(nullptr, onnx::AttributeProto_AttributeType_FLOATS, "test_attr", data);
  SUCCEED();
}

TEST_F(GeIrUtilsIncCov2, IncCov2_AddAttrProto_NullNodeProto_RepeatedString) {
  ::google::protobuf::RepeatedPtrField<::std::string> data;
  data.Add("hello");
  data.Add("world");
  OnnxUtils::AddAttrProto(nullptr, onnx::AttributeProto_AttributeType_STRINGS, "test_attr", data);
  SUCCEED();
}

TEST_F(GeIrUtilsIncCov2, IncCov2_AddAttrProto_EmptyData_RepeatedInt64) {
  onnx::NodeProto node_proto;
  ::google::protobuf::RepeatedField<::google::protobuf::int64> data;
  OnnxUtils::AddAttrProto(&node_proto, onnx::AttributeProto_AttributeType_INTS, "empty_attr", data);
  SUCCEED();
}

TEST_F(GeIrUtilsIncCov2, IncCov2_AddAttrProto_EmptyData_RepeatedBool) {
  onnx::NodeProto node_proto;
  ::google::protobuf::RepeatedField<bool> data;
  OnnxUtils::AddAttrProto(&node_proto, onnx::AttributeProto_AttributeType_INTS, "empty_attr", data);
  SUCCEED();
}

TEST_F(GeIrUtilsIncCov2, IncCov2_AddAttrProto_EmptyData_RepeatedFloat) {
  onnx::NodeProto node_proto;
  ::google::protobuf::RepeatedField<float> data;
  OnnxUtils::AddAttrProto(&node_proto, onnx::AttributeProto_AttributeType_FLOATS, "empty_attr", data);
  SUCCEED();
}

TEST_F(GeIrUtilsIncCov2, IncCov2_AddAttrProto_EmptyData_RepeatedString) {
  onnx::NodeProto node_proto;
  ::google::protobuf::RepeatedPtrField<::std::string> data;
  OnnxUtils::AddAttrProto(&node_proto, onnx::AttributeProto_AttributeType_STRINGS, "empty_attr", data);
  SUCCEED();
}

TEST_F(GeIrUtilsIncCov2, IncCov2_AddAttrProtoForOpInAndOutDesc_NullParams) {
  OnnxUtils::AddAttrProtoForOpInAndOutDesc(nullptr, nullptr);
  SUCCEED();
}

TEST_F(GeIrUtilsIncCov2, IncCov2_AddAttrProtoForOpInAndOutDesc_NullOpDesc) {
  onnx::NodeProto node_proto;
  OnnxUtils::AddAttrProtoForOpInAndOutDesc(&node_proto, nullptr);
  SUCCEED();
}

TEST_F(GeIrUtilsIncCov2, IncCov2_AddAttrProtoForOpInAndOutDesc_NullNodeProto) {
  OpDescPtr op_desc = std::make_shared<OpDesc>("test_op", "TestOp");
  OnnxUtils::AddAttrProtoForOpInAndOutDesc(nullptr, op_desc);
  SUCCEED();
}

TEST_F(GeIrUtilsIncCov2, IncCov2_EncodeNodeDesc_NullNode) {
  NodePtr null_node;
  onnx::NodeProto node_proto;
  EXPECT_FALSE(OnnxUtils::EncodeNodeDesc(null_node, &node_proto));
}

TEST_F(GeIrUtilsIncCov2, IncCov2_EncodeNodeDesc_NullNodeProto) {
  auto graph = BuildIncCov2Graph();
  auto node = graph->GetDirectNode().at(0);
  EXPECT_FALSE(OnnxUtils::EncodeNodeDesc(node, nullptr));
}

TEST_F(GeIrUtilsIncCov2, IncCov2_EncodeNodeLinkForNetronVisual_NullNode) {
  NodePtr null_node;
  onnx::NodeProto node_proto;
  OnnxUtils::EncodeNodeLinkForNetronVisual(null_node, &node_proto);
  SUCCEED();
}

TEST_F(GeIrUtilsIncCov2, IncCov2_EncodeNodeLinkForNetronVisual_NullNodeProto) {
  auto graph = BuildIncCov2Graph();
  auto node = graph->GetDirectNode().at(0);
  OnnxUtils::EncodeNodeLinkForNetronVisual(node, nullptr);
  SUCCEED();
}

TEST_F(GeIrUtilsIncCov2, IncCov2_EncodeNodeLink_NullNode) {
  NodePtr null_node;
  onnx::NodeProto node_proto;
  EXPECT_FALSE(OnnxUtils::EncodeNodeLink(null_node, &node_proto));
}

TEST_F(GeIrUtilsIncCov2, IncCov2_EncodeNodeLink_NullNodeProto) {
  auto graph = BuildIncCov2Graph();
  auto node = graph->GetDirectNode().at(0);
  EXPECT_FALSE(OnnxUtils::EncodeNodeLink(node, nullptr));
}

TEST_F(GeIrUtilsIncCov2, IncCov2_EncodeTypeProtoTensorType_NullNode) {
  NodePtr null_node;
  onnx::TypeProto_Tensor tensor_type;
  OnnxUtils::EncodeTypeProtoTensorType(null_node, &tensor_type);
  SUCCEED();
}

TEST_F(GeIrUtilsIncCov2, IncCov2_EncodeTypeProtoTensorType_NullTensorType) {
  auto graph = BuildIncCov2Graph();
  auto node = graph->GetDirectNode().at(0);
  OnnxUtils::EncodeTypeProtoTensorType(node, nullptr);
  SUCCEED();
}

TEST_F(GeIrUtilsIncCov2, IncCov2_EncodeValueInfo_NullNode) {
  NodePtr null_node;
  onnx::ValueInfoProto value_info;
  OnnxUtils::EncodeValueInfo(null_node, &value_info);
  SUCCEED();
}

TEST_F(GeIrUtilsIncCov2, IncCov2_EncodeValueInfo_NullValueInfo) {
  auto graph = BuildIncCov2Graph();
  auto node = graph->GetDirectNode().at(0);
  OnnxUtils::EncodeValueInfo(node, nullptr);
  SUCCEED();
}

TEST_F(GeIrUtilsIncCov2, IncCov2_EncodeGraph_NullGraph) {
  onnx::GraphProto graph_proto;
  EXPECT_FALSE(OnnxUtils::EncodeGraph(nullptr, &graph_proto));
}

TEST_F(GeIrUtilsIncCov2, IncCov2_EncodeGraph_NullGraphProto) {
  auto graph = BuildIncCov2Graph();
  EXPECT_FALSE(OnnxUtils::EncodeGraph(graph, nullptr));
}

TEST_F(GeIrUtilsIncCov2, IncCov2_EncodeNode_NullNode) {
  NodePtr null_node;
  onnx::NodeProto node_proto;
  EXPECT_FALSE(OnnxUtils::EncodeNode(null_node, &node_proto));
}

TEST_F(GeIrUtilsIncCov2, IncCov2_EncodeNode_NullNodeProto) {
  auto graph = BuildIncCov2Graph();
  auto node = graph->GetDirectNode().at(0);
  EXPECT_FALSE(OnnxUtils::EncodeNode(node, nullptr));
}

TEST_F(GeIrUtilsIncCov2, IncCov2_ConvertGeModelToModelProto_Success) {
  auto graph = BuildIncCov2GraphWithAttrs();
  ge::Model model("test_model_inc2", "");
  model.SetGraph(graph);
  onnx::ModelProto model_proto;
  EXPECT_TRUE(OnnxUtils::ConvertGeModelToModelProto(model, model_proto));
}

TEST_F(GeIrUtilsIncCov2, IncCov2_ConvertGeModelToModelProto_NullGraph) {
  ge::Model model("test_null_graph_inc2", "");
  onnx::ModelProto model_proto;
  EXPECT_FALSE(OnnxUtils::ConvertGeModelToModelProto(model, model_proto));
}

TEST_F(GeIrUtilsIncCov2, IncCov2_ConvertGeModelToModelProto_WithDumpAll) {
  auto graph = BuildIncCov2GraphWithAttrs();
  ge::Model model("test_dump_all_inc2", "");
  model.SetGraph(graph);
  onnx::ModelProto model_proto;
  EXPECT_TRUE(OnnxUtils::ConvertGeModelToModelProto(model, model_proto, DumpLevel::DUMP_ALL));
}

TEST_F(GeIrUtilsIncCov2, IncCov2_ConvertGeModelToModelProto_WithDumpNoDesc) {
  auto graph = BuildIncCov2GraphWithAttrs();
  ge::Model model("test_dump_no_desc_inc2", "");
  model.SetGraph(graph);
  onnx::ModelProto model_proto;
  EXPECT_TRUE(OnnxUtils::ConvertGeModelToModelProto(model, model_proto, DumpLevel::DUMP_WITH_OUT_DESC));
}

TEST_F(GeIrUtilsIncCov2, IncCov2_ConvertGeModelToModelProto_WithSubgraph) {
  auto graph = BuildIncCov2GraphWithSubgraph();
  ge::Model model("test_subgraph_inc2", "");
  model.SetGraph(graph);
  onnx::ModelProto model_proto;
  EXPECT_TRUE(OnnxUtils::ConvertGeModelToModelProto(model, model_proto));
}

TEST_F(GeIrUtilsIncCov2, IncCov2_EncodeDataType_UnsupportedType) {
  auto result = OnnxUtils::EncodeDataType(DT_UNDEFINED);
  EXPECT_EQ(result, onnx::TensorProto_DataType_UNDEFINED);
}

TEST_F(GeIrUtilsIncCov2, IncCov2_EncodeDataType_Float8E5M2) {
  auto result = OnnxUtils::EncodeDataType(DT_FLOAT8_E5M2);
  EXPECT_EQ(result, onnx::TensorProto_DataType_FLOAT8E5M2);
}

TEST_F(GeIrUtilsIncCov2, IncCov2_EncodeDataType_Float8E4M3FN) {
  auto result = OnnxUtils::EncodeDataType(DT_FLOAT8_E4M3FN);
  EXPECT_EQ(result, onnx::TensorProto_DataType_FLOAT8E4M3FN);
}

TEST_F(GeIrUtilsIncCov2, IncCov2_AddAttrProto_UnsupportedType) {
  onnx::NodeProto node_proto;
  OnnxUtils::AddAttrProto(&node_proto, static_cast<onnx::AttributeProto_AttributeType>(999), "test_attr", nullptr);
  SUCCEED();
}

TEST_F(GeIrUtilsIncCov2, IncCov2_AddAttrProto_NullNodeProto_VoidPtr) {
  int64_t value = 42;
  OnnxUtils::AddAttrProto(nullptr, onnx::AttributeProto_AttributeType_INT, "test_attr", &value);
  SUCCEED();
}

TEST_F(GeIrUtilsIncCov2, IncCov2_EncodeNode_WithInputConst) {
  auto graph = BuildIncCov2GraphWithAttrs();
  auto node = graph->GetDirectNode().at(1);
  if (node != nullptr && node->GetOpDesc() != nullptr) {
    std::vector<bool> is_input_const = {true, false};
    node->GetOpDesc()->SetIsInputConst(is_input_const);
  }
  onnx::NodeProto node_proto;
  EXPECT_TRUE(OnnxUtils::EncodeNode(node, &node_proto));
}

TEST_F(GeIrUtilsIncCov2, IncCov2_DecodeGraph_Success) {
  auto graph = BuildIncCov2GraphWithAttrs();
  ge::Model model("test_decode_inc2", "");
  model.SetGraph(graph);
  onnx::ModelProto model_proto;
  ASSERT_TRUE(OnnxUtils::ConvertGeModelToModelProto(model, model_proto));
  ComputeGraphPtr decoded_graph;
  EXPECT_TRUE(OnnxUtils::DecodeGraph(0, model_proto.graph(), decoded_graph));
}

TEST_F(GeIrUtilsIncCov2, IncCov2_DecodeGraph_MaxDepth) {
  ComputeGraphPtr decoded_graph;
  onnx::GraphProto graph_proto;
  graph_proto.set_name("test_max_depth");
  EXPECT_FALSE(OnnxUtils::DecodeGraph(15, graph_proto, decoded_graph));
}

TEST_F(GeIrUtilsIncCov2, IncCov2_DecodeNodeDesc_NullParams) {
  OpDescPtr op_desc;
  onnx::NodeProto node_proto;
  EXPECT_FALSE(OnnxUtils::DecodeNodeDesc(&node_proto, op_desc));
}

TEST_F(GeIrUtilsIncCov2, IncCov2_DecodeNodeDesc_NullNodeProto) {
  OpDescPtr op_desc = std::make_shared<OpDesc>("test", "TestOp");
  EXPECT_FALSE(OnnxUtils::DecodeNodeDesc(nullptr, op_desc));
}

TEST_F(GeIrUtilsIncCov2, IncCov2_DecodeNodeDesc_NoColonInType) {
  OpDescPtr op_desc = std::make_shared<OpDesc>("test", "TestOp");
  onnx::NodeProto node_proto;
  node_proto.set_name("test_node");
  node_proto.set_op_type("NoPrefixType");
  EXPECT_FALSE(OnnxUtils::DecodeNodeDesc(&node_proto, op_desc));
}

TEST_F(GeIrUtilsIncCov2, IncCov2_DecodeNodeAttributeForOpDesc_NullOpDesc) {
  OpDescPtr null_op_desc;
  onnx::AttributeProto attr_proto;
  OnnxUtils::DecodeNodeAttributeForOpDesc(attr_proto, null_op_desc);
  SUCCEED();
}

TEST_F(GeIrUtilsIncCov2, IncCov2_DecodeNodeAttributeForOpDesc_Id) {
  OpDescPtr op_desc = std::make_shared<OpDesc>("test_id", "TestOp");
  onnx::AttributeProto attr_proto;
  attr_proto.set_name("id");
  attr_proto.set_type(onnx::AttributeProto_AttributeType_INT);
  attr_proto.set_i(42);
  OnnxUtils::DecodeNodeAttributeForOpDesc(attr_proto, op_desc);
  EXPECT_EQ(op_desc->GetId(), 42);
}

TEST_F(GeIrUtilsIncCov2, IncCov2_DecodeNodeAttributeForOpDesc_StreamId) {
  OpDescPtr op_desc = std::make_shared<OpDesc>("test_stream", "TestOp");
  onnx::AttributeProto attr_proto;
  attr_proto.set_name("stream_id");
  attr_proto.set_type(onnx::AttributeProto_AttributeType_INT);
  attr_proto.set_i(7);
  OnnxUtils::DecodeNodeAttributeForOpDesc(attr_proto, op_desc);
  EXPECT_EQ(op_desc->GetStreamId(), 7);
}

TEST_F(GeIrUtilsIncCov2, IncCov2_DecodeNodeAttributeForOpDesc_SrcName) {
  OpDescPtr op_desc = std::make_shared<OpDesc>("test_src_name", "TestOp");
  onnx::AttributeProto attr_proto;
  attr_proto.set_name("src_name");
  attr_proto.set_type(onnx::AttributeProto_AttributeType_STRINGS);
  attr_proto.add_strings("src1");
  attr_proto.add_strings("src2");
  OnnxUtils::DecodeNodeAttributeForOpDesc(attr_proto, op_desc);
  auto src_names = op_desc->GetSrcName();
  EXPECT_EQ(src_names.size(), 2U);
}

TEST_F(GeIrUtilsIncCov2, IncCov2_DecodeNodeAttributeForOpDesc_DstName) {
  OpDescPtr op_desc = std::make_shared<OpDesc>("test_dst_name", "TestOp");
  onnx::AttributeProto attr_proto;
  attr_proto.set_name("dst_name");
  attr_proto.set_type(onnx::AttributeProto_AttributeType_STRINGS);
  attr_proto.add_strings("dst1");
  OnnxUtils::DecodeNodeAttributeForOpDesc(attr_proto, op_desc);
  auto dst_names = op_desc->GetDstName();
  EXPECT_EQ(dst_names.size(), 1U);
}

TEST_F(GeIrUtilsIncCov2, IncCov2_DecodeNodeAttributeForOpDesc_SrcIndex) {
  OpDescPtr op_desc = std::make_shared<OpDesc>("test_src_idx", "TestOp");
  onnx::AttributeProto attr_proto;
  attr_proto.set_name("src_index");
  attr_proto.set_type(onnx::AttributeProto_AttributeType_INTS);
  attr_proto.add_ints(0);
  attr_proto.add_ints(1);
  OnnxUtils::DecodeNodeAttributeForOpDesc(attr_proto, op_desc);
  auto src_idx = op_desc->GetSrcIndex();
  EXPECT_EQ(src_idx.size(), 2U);
}

TEST_F(GeIrUtilsIncCov2, IncCov2_DecodeNodeAttributeForOpDesc_InputI) {
  OpDescPtr op_desc = std::make_shared<OpDesc>("test_input_i", "TestOp");
  onnx::AttributeProto attr_proto;
  attr_proto.set_name("input_i");
  attr_proto.set_type(onnx::AttributeProto_AttributeType_INTS);
  attr_proto.add_ints(1024);
  OnnxUtils::DecodeNodeAttributeForOpDesc(attr_proto, op_desc);
  auto input_offset = op_desc->GetInputOffset();
  EXPECT_EQ(input_offset.size(), 1U);
}

TEST_F(GeIrUtilsIncCov2, IncCov2_DecodeNodeAttributeForOpDesc_OutputI) {
  OpDescPtr op_desc = std::make_shared<OpDesc>("test_output_i", "TestOp");
  onnx::AttributeProto attr_proto;
  attr_proto.set_name("output_i");
  attr_proto.set_type(onnx::AttributeProto_AttributeType_INTS);
  attr_proto.add_ints(2048);
  OnnxUtils::DecodeNodeAttributeForOpDesc(attr_proto, op_desc);
  auto output_offset = op_desc->GetOutputOffset();
  EXPECT_EQ(output_offset.size(), 1U);
}

TEST_F(GeIrUtilsIncCov2, IncCov2_DecodeNodeAttributeForOpDesc_FusionScope) {
  OpDescPtr op_desc = std::make_shared<OpDesc>("test_fusion", "TestOp");
  onnx::AttributeProto attr_proto;
  attr_proto.set_name("fusion_scope");
  attr_proto.set_type(onnx::AttributeProto_AttributeType_INT);
  attr_proto.set_i(99);
  OnnxUtils::DecodeNodeAttributeForOpDesc(attr_proto, op_desc);
  int64_t val = 0;
  AttrUtils::GetInt(op_desc, "fusion_scope", val);
  EXPECT_EQ(val, 99);
}

TEST_F(GeIrUtilsIncCov2, IncCov2_DecodeNodeAttributeForOpDesc_UnknownAttr) {
  OpDescPtr op_desc = std::make_shared<OpDesc>("test_unknown_attr", "TestOp");
  onnx::AttributeProto attr_proto;
  attr_proto.set_name("unknown_attr_name");
  attr_proto.set_type(onnx::AttributeProto_AttributeType_INT);
  attr_proto.set_i(1);
  OnnxUtils::DecodeNodeAttributeForOpDesc(attr_proto, op_desc);
  SUCCEED();
}

TEST_F(GeIrUtilsIncCov2, IncCov2_DecodeNodeAttributeForOpInDesc_NullDesc) {
  OpDescPtr op_desc = std::make_shared<OpDesc>("test_null_in_desc", "TestOp");
  onnx::AttributeProto attr_proto;
  attr_proto.set_name("input_desc_dtype:0");
  attr_proto.set_type(onnx::AttributeProto_AttributeType_STRING);
  attr_proto.set_s("DT_FLOAT");
  OnnxUtils::DecodeNodeAttributeForOpInDesc(attr_proto, "input_desc_dtype", 5, op_desc);
  SUCCEED();
}

TEST_F(GeIrUtilsIncCov2, IncCov2_DecodeNodeAttributeForOpOutDesc_NullDesc) {
  OpDescPtr op_desc = std::make_shared<OpDesc>("test_null_out_desc", "TestOp");
  onnx::AttributeProto attr_proto;
  attr_proto.set_name("output_desc_dtype:0");
  attr_proto.set_type(onnx::AttributeProto_AttributeType_STRING);
  attr_proto.set_s("DT_FLOAT");
  OnnxUtils::DecodeNodeAttributeForOpOutDesc(attr_proto, "output_desc_dtype", 5, op_desc);
  SUCCEED();
}

TEST_F(GeIrUtilsIncCov2, IncCov2_DecodeNodeAttributeForOpInDesc_DeviceType) {
  OpDescPtr op_desc = std::make_shared<OpDesc>("test_device_type", "TestOp");
  op_desc->AddInputDesc(GeTensorDesc(GeShape({1}), FORMAT_NCHW, DT_FLOAT));
  onnx::AttributeProto attr_proto;
  attr_proto.set_name("input_desc_device_type:0");
  attr_proto.set_type(onnx::AttributeProto_AttributeType_STRING);
  attr_proto.set_s("Ascend910");
  OnnxUtils::DecodeNodeAttributeForOpInDesc(attr_proto, "input_desc_device_type", 0, op_desc);
  SUCCEED();
}

TEST_F(GeIrUtilsIncCov2, IncCov2_DecodeNodeAttributeForOpInDesc_OriginDtype) {
  OpDescPtr op_desc = std::make_shared<OpDesc>("test_origin_dtype", "TestOp");
  op_desc->AddInputDesc(GeTensorDesc(GeShape({1}), FORMAT_NCHW, DT_FLOAT));
  onnx::AttributeProto attr_proto;
  attr_proto.set_name("input_desc_origin_dtype:0");
  attr_proto.set_type(onnx::AttributeProto_AttributeType_STRING);
  attr_proto.set_s("DT_FLOAT");
  OnnxUtils::DecodeNodeAttributeForOpInDesc(attr_proto, "input_desc_origin_dtype", 0, op_desc);
  SUCCEED();
}

TEST_F(GeIrUtilsIncCov2, IncCov2_DecodeNodeAttributeForOpInDesc_OriginShape) {
  OpDescPtr op_desc = std::make_shared<OpDesc>("test_origin_shape", "TestOp");
  op_desc->AddInputDesc(GeTensorDesc(GeShape({1}), FORMAT_NCHW, DT_FLOAT));
  onnx::AttributeProto attr_proto;
  attr_proto.set_name("input_desc_origin_shape:0");
  attr_proto.set_type(onnx::AttributeProto_AttributeType_INTS);
  attr_proto.add_ints(1);
  attr_proto.add_ints(3);
  OnnxUtils::DecodeNodeAttributeForOpInDesc(attr_proto, "input_desc_origin_shape", 0, op_desc);
  SUCCEED();
}

TEST_F(GeIrUtilsIncCov2, IncCov2_DecodeNodeAttributeForOpInDesc_OriginLayout) {
  OpDescPtr op_desc = std::make_shared<OpDesc>("test_origin_layout", "TestOp");
  op_desc->AddInputDesc(GeTensorDesc(GeShape({1}), FORMAT_NCHW, DT_FLOAT));
  onnx::AttributeProto attr_proto;
  attr_proto.set_name("input_desc_origin_layout:0");
  attr_proto.set_type(onnx::AttributeProto_AttributeType_STRING);
  attr_proto.set_s("NCHW");
  OnnxUtils::DecodeNodeAttributeForOpInDesc(attr_proto, "input_desc_origin_layout", 0, op_desc);
  SUCCEED();
}

TEST_F(GeIrUtilsIncCov2, IncCov2_DecodeNodeAttributeForOpInDesc_Size) {
  OpDescPtr op_desc = std::make_shared<OpDesc>("test_desc_size", "TestOp");
  op_desc->AddInputDesc(GeTensorDesc(GeShape({1}), FORMAT_NCHW, DT_FLOAT));
  onnx::AttributeProto attr_proto;
  attr_proto.set_name("input_desc_size:0");
  attr_proto.set_type(onnx::AttributeProto_AttributeType_INT);
  attr_proto.set_i(1024);
  OnnxUtils::DecodeNodeAttributeForOpInDesc(attr_proto, "input_desc_size", 0, op_desc);
  SUCCEED();
}

TEST_F(GeIrUtilsIncCov2, IncCov2_DecodeNodeAttributeForOpInDesc_DataOffset) {
  OpDescPtr op_desc = std::make_shared<OpDesc>("test_data_offset", "TestOp");
  op_desc->AddInputDesc(GeTensorDesc(GeShape({1}), FORMAT_NCHW, DT_FLOAT));
  onnx::AttributeProto attr_proto;
  attr_proto.set_name("input_desc_data_offset:0");
  attr_proto.set_type(onnx::AttributeProto_AttributeType_INT);
  attr_proto.set_i(512);
  OnnxUtils::DecodeNodeAttributeForOpInDesc(attr_proto, "input_desc_data_offset", 0, op_desc);
  SUCCEED();
}

TEST_F(GeIrUtilsIncCov2, IncCov2_DecodeNodeAttributeForOpInDesc_UnknownName) {
  OpDescPtr op_desc = std::make_shared<OpDesc>("test_unknown_name", "TestOp");
  op_desc->AddInputDesc(GeTensorDesc(GeShape({1}), FORMAT_NCHW, DT_FLOAT));
  onnx::AttributeProto attr_proto;
  attr_proto.set_name("input_desc_unknown_field:0");
  attr_proto.set_type(onnx::AttributeProto_AttributeType_INT);
  attr_proto.set_i(0);
  OnnxUtils::DecodeNodeAttributeForOpInDesc(attr_proto, "input_desc_unknown_field", 0, op_desc);
  SUCCEED();
}

TEST_F(GeIrUtilsIncCov2, IncCov2_DecodeNodeAttributeForOpOutDesc_Dtype) {
  OpDescPtr op_desc = std::make_shared<OpDesc>("test_out_dtype", "TestOp");
  op_desc->AddOutputDesc(GeTensorDesc(GeShape({1}), FORMAT_NCHW, DT_FLOAT));
  onnx::AttributeProto attr_proto;
  attr_proto.set_name("output_desc_dtype:0");
  attr_proto.set_type(onnx::AttributeProto_AttributeType_STRING);
  attr_proto.set_s("DT_INT32");
  OnnxUtils::DecodeNodeAttributeForOpOutDesc(attr_proto, "output_desc_dtype", 0, op_desc);
  SUCCEED();
}

TEST_F(GeIrUtilsIncCov2, IncCov2_DecodeNodeAttributeForOpOutDesc_Shape) {
  OpDescPtr op_desc = std::make_shared<OpDesc>("test_out_shape", "TestOp");
  op_desc->AddOutputDesc(GeTensorDesc(GeShape({1}), FORMAT_NCHW, DT_FLOAT));
  onnx::AttributeProto attr_proto;
  attr_proto.set_name("output_desc_shape:0");
  attr_proto.set_type(onnx::AttributeProto_AttributeType_INTS);
  attr_proto.add_ints(2);
  attr_proto.add_ints(3);
  OnnxUtils::DecodeNodeAttributeForOpOutDesc(attr_proto, "output_desc_shape", 0, op_desc);
  SUCCEED();
}

TEST_F(GeIrUtilsIncCov2, IncCov2_DecodeNodeAttributeForOpOutDesc_Layout) {
  OpDescPtr op_desc = std::make_shared<OpDesc>("test_out_layout", "TestOp");
  op_desc->AddOutputDesc(GeTensorDesc(GeShape({1}), FORMAT_NCHW, DT_FLOAT));
  onnx::AttributeProto attr_proto;
  attr_proto.set_name("output_desc_layout:0");
  attr_proto.set_type(onnx::AttributeProto_AttributeType_STRING);
  attr_proto.set_s("NCHW");
  OnnxUtils::DecodeNodeAttributeForOpOutDesc(attr_proto, "output_desc_layout", 0, op_desc);
  SUCCEED();
}

TEST_F(GeIrUtilsIncCov2, IncCov2_DecodeNodeAttributeForOpOutDesc_OriginShape) {
  OpDescPtr op_desc = std::make_shared<OpDesc>("test_out_origin_shape", "TestOp");
  op_desc->AddOutputDesc(GeTensorDesc(GeShape({1}), FORMAT_NCHW, DT_FLOAT));
  onnx::AttributeProto attr_proto;
  attr_proto.set_name("output_desc_origin_shape:0");
  attr_proto.set_type(onnx::AttributeProto_AttributeType_INTS);
  attr_proto.add_ints(1);
  OnnxUtils::DecodeNodeAttributeForOpOutDesc(attr_proto, "output_desc_origin_shape", 0, op_desc);
  SUCCEED();
}

TEST_F(GeIrUtilsIncCov2, IncCov2_DecodeNodeAttributeForOpOutDesc_OriginLayout) {
  OpDescPtr op_desc = std::make_shared<OpDesc>("test_out_origin_layout", "TestOp");
  op_desc->AddOutputDesc(GeTensorDesc(GeShape({1}), FORMAT_NCHW, DT_FLOAT));
  onnx::AttributeProto attr_proto;
  attr_proto.set_name("output_desc_origin_layout:0");
  attr_proto.set_type(onnx::AttributeProto_AttributeType_STRING);
  attr_proto.set_s("NCHW");
  OnnxUtils::DecodeNodeAttributeForOpOutDesc(attr_proto, "output_desc_origin_layout", 0, op_desc);
  SUCCEED();
}

TEST_F(GeIrUtilsIncCov2, IncCov2_DecodeNodeAttributeForOpOutDesc_Size) {
  OpDescPtr op_desc = std::make_shared<OpDesc>("test_out_size", "TestOp");
  op_desc->AddOutputDesc(GeTensorDesc(GeShape({1}), FORMAT_NCHW, DT_FLOAT));
  onnx::AttributeProto attr_proto;
  attr_proto.set_name("output_desc_size:0");
  attr_proto.set_type(onnx::AttributeProto_AttributeType_INT);
  attr_proto.set_i(2048);
  OnnxUtils::DecodeNodeAttributeForOpOutDesc(attr_proto, "output_desc_size", 0, op_desc);
  SUCCEED();
}

TEST_F(GeIrUtilsIncCov2, IncCov2_DecodeNodeAttributeForOpOutDesc_DataOffset) {
  OpDescPtr op_desc = std::make_shared<OpDesc>("test_out_data_offset", "TestOp");
  op_desc->AddOutputDesc(GeTensorDesc(GeShape({1}), FORMAT_NCHW, DT_FLOAT));
  onnx::AttributeProto attr_proto;
  attr_proto.set_name("output_desc_data_offset:0");
  attr_proto.set_type(onnx::AttributeProto_AttributeType_INT);
  attr_proto.set_i(100);
  OnnxUtils::DecodeNodeAttributeForOpOutDesc(attr_proto, "output_desc_data_offset", 0, op_desc);
  SUCCEED();
}

TEST_F(GeIrUtilsIncCov2, IncCov2_DecodeNodeAttributeForOpOutDesc_UnknownName) {
  OpDescPtr op_desc = std::make_shared<OpDesc>("test_out_unknown", "TestOp");
  op_desc->AddOutputDesc(GeTensorDesc(GeShape({1}), FORMAT_NCHW, DT_FLOAT));
  onnx::AttributeProto attr_proto;
  attr_proto.set_name("output_desc_unknown:0");
  attr_proto.set_type(onnx::AttributeProto_AttributeType_INT);
  attr_proto.set_i(0);
  OnnxUtils::DecodeNodeAttributeForOpOutDesc(attr_proto, "output_desc_unknown", 0, op_desc);
  SUCCEED();
}

TEST_F(GeIrUtilsIncCov2, IncCov2_ParseNameAndIndex_Valid) {
  std::string name;
  int32_t idx = -1;
  EXPECT_TRUE(OnnxUtils::ParseNameAndIndex("node_name:5", name, idx));
  EXPECT_EQ(name, "node_name");
  EXPECT_EQ(idx, 5);
}

TEST_F(GeIrUtilsIncCov2, IncCov2_ParseNameAndIndex_NoColon) {
  std::string name;
  int32_t idx = -1;
  EXPECT_FALSE(OnnxUtils::ParseNameAndIndex("no_colon_name", name, idx));
}

TEST_F(GeIrUtilsIncCov2, IncCov2_DecodeAttribute_WrongType_Strings) {
  onnx::AttributeProto attr_proto;
  attr_proto.set_name("test_attr");
  attr_proto.set_type(onnx::AttributeProto_AttributeType_INT);
  std::vector<std::string> strings;
  OnnxUtils::DecodeAttribute(attr_proto, strings);
  EXPECT_TRUE(strings.empty());
}

TEST_F(GeIrUtilsIncCov2, IncCov2_DecodeAttribute_WrongType_String) {
  onnx::AttributeProto attr_proto;
  attr_proto.set_name("test_attr");
  attr_proto.set_type(onnx::AttributeProto_AttributeType_INT);
  std::string value;
  OnnxUtils::DecodeAttribute(attr_proto, value);
  EXPECT_TRUE(value.empty());
}

TEST_F(GeIrUtilsIncCov2, IncCov2_DecodeAttribute_WrongType_Ints) {
  onnx::AttributeProto attr_proto;
  attr_proto.set_name("test_attr");
  attr_proto.set_type(onnx::AttributeProto_AttributeType_STRING);
  std::vector<int64_t> ints;
  OnnxUtils::DecodeAttribute(attr_proto, ints);
  EXPECT_TRUE(ints.empty());
}

TEST_F(GeIrUtilsIncCov2, IncCov2_DecodeAttribute_WrongType_Int) {
  onnx::AttributeProto attr_proto;
  attr_proto.set_name("test_attr");
  attr_proto.set_type(onnx::AttributeProto_AttributeType_STRING);
  int64_t value = 42;
  OnnxUtils::DecodeAttribute(attr_proto, value);
  EXPECT_EQ(value, 42);
}

TEST_F(GeIrUtilsIncCov2, IncCov2_DecodeNodeLinkImp_DataAnchorFail) {
  auto graph = std::make_shared<ComputeGraph>("test_anchor_fail");
  auto src_node = CreateNodeIncCov2Helper(graph, "src_anchor_fail", "Data", 1, 1);
  auto dst_node = CreateNodeIncCov2Helper(graph, "dst_anchor_fail", "Relu", 1, 1);
  OnnxUtils::NodeLinkInfo item{"src_anchor_fail", 5, dst_node, 0, "dst_anchor_fail"};
  EXPECT_FALSE(OnnxUtils::DecodeNodeLinkImp(item, src_node));
}

TEST_F(GeIrUtilsIncCov2, IncCov2_AddInputAndOutputNodesForGraph_Success) {
  auto graph = BuildIncCov2Graph();
  ge::Model model("test_add_io_inc2", "");
  model.SetGraph(graph);
  onnx::ModelProto model_proto;
  ASSERT_TRUE(OnnxUtils::ConvertGeModelToModelProto(model, model_proto));

  ComputeGraphPtr new_graph = std::make_shared<ComputeGraph>("new_graph_inc2");
  std::map<std::string, NodePtr> node_map;
  for (const auto &node : graph->GetDirectNode()) {
    node_map[node->GetName()] = node;
  }
  EXPECT_TRUE(OnnxUtils::AddInputAndOutputNodesForGraph(model_proto.graph(), new_graph, node_map));
}

TEST_F(GeIrUtilsIncCov2, IncCov2_AddInputAndOutputNodesForGraph_InputNotFound) {
  auto graph = BuildIncCov2Graph();
  ge::Model model("test_input_notfound_inc2", "");
  model.SetGraph(graph);
  onnx::ModelProto model_proto;
  ASSERT_TRUE(OnnxUtils::ConvertGeModelToModelProto(model, model_proto));

  ComputeGraphPtr new_graph = std::make_shared<ComputeGraph>("new_graph_inc2_nf");
  std::map<std::string, NodePtr> empty_map;
  EXPECT_FALSE(OnnxUtils::AddInputAndOutputNodesForGraph(model_proto.graph(), new_graph, empty_map));
}

TEST_F(GeIrUtilsIncCov2, IncCov2_DecodeNodeLink_DataEdgeSuccess) {
  auto graph = std::make_shared<ComputeGraph>("test_decode_link");
  auto src = CreateNodeIncCov2Helper(graph, "src_link", "Data", 1, 1);
  auto dst = CreateNodeIncCov2Helper(graph, "dst_link", "Relu", 1, 1);

  onnx::NodeProto node_proto;
  node_proto.set_name("dst_link");
  node_proto.add_input("src_link:0");
  node_proto.add_input("");

  std::vector<onnx::NodeProto> node_proto_vector;
  node_proto_vector.push_back(node_proto);

  std::map<std::string, NodePtr> node_map;
  node_map["src_link"] = src;
  node_map["dst_link"] = dst;

  EXPECT_TRUE(OnnxUtils::DecodeNodeLink(node_proto_vector, node_map));
}

TEST_F(GeIrUtilsIncCov2, IncCov2_DecodeNodeLink_ControlEdgeSuccess) {
  auto graph = std::make_shared<ComputeGraph>("test_decode_ctrl");
  auto src = CreateNodeIncCov2Helper(graph, "src_ctrl", "Data", 1, 1);
  auto dst = CreateNodeIncCov2Helper(graph, "dst_ctrl", "Relu", 1, 1);

  onnx::NodeProto node_proto;
  node_proto.set_name("dst_ctrl");
  node_proto.add_input("src_ctrl:-1");

  std::vector<onnx::NodeProto> node_proto_vector;
  node_proto_vector.push_back(node_proto);

  std::map<std::string, NodePtr> node_map;
  node_map["src_ctrl"] = src;
  node_map["dst_ctrl"] = dst;

  EXPECT_TRUE(OnnxUtils::DecodeNodeLink(node_proto_vector, node_map));
}

TEST_F(GeIrUtilsIncCov2, IncCov2_DecodeNodeLink_DstNotFound) {
  auto graph = std::make_shared<ComputeGraph>("test_decode_dst_nf");
  auto src = CreateNodeIncCov2Helper(graph, "src_nf", "Data", 1, 1);

  onnx::NodeProto node_proto;
  node_proto.set_name("nonexist_dst");
  node_proto.add_input("src_nf:0");

  std::vector<onnx::NodeProto> node_proto_vector;
  node_proto_vector.push_back(node_proto);

  std::map<std::string, NodePtr> node_map;
  node_map["src_nf"] = src;

  EXPECT_FALSE(OnnxUtils::DecodeNodeLink(node_proto_vector, node_map));
}

TEST_F(GeIrUtilsIncCov2, IncCov2_DecodeNodeLink_SrcNotFound) {
  auto graph = std::make_shared<ComputeGraph>("test_decode_src_nf");
  auto dst = CreateNodeIncCov2Helper(graph, "dst_src_nf", "Relu", 1, 1);

  onnx::NodeProto node_proto;
  node_proto.set_name("dst_src_nf");
  node_proto.add_input("nonexist_src:0");

  std::vector<onnx::NodeProto> node_proto_vector;
  node_proto_vector.push_back(node_proto);

  std::map<std::string, NodePtr> node_map;
  node_map["dst_src_nf"] = dst;

  EXPECT_FALSE(OnnxUtils::DecodeNodeLink(node_proto_vector, node_map));
}

TEST_F(GeIrUtilsIncCov2, IncCov2_DecodeNodeLink_EmptyInput) {
  auto graph = std::make_shared<ComputeGraph>("test_decode_empty_in");
  auto dst = CreateNodeIncCov2Helper(graph, "dst_empty_in", "Relu", 1, 1);

  onnx::NodeProto node_proto;
  node_proto.set_name("dst_empty_in");
  node_proto.add_input("");

  std::vector<onnx::NodeProto> node_proto_vector;
  node_proto_vector.push_back(node_proto);

  std::map<std::string, NodePtr> node_map;
  node_map["dst_empty_in"] = dst;

  EXPECT_TRUE(OnnxUtils::DecodeNodeLink(node_proto_vector, node_map));
}

TEST_F(GeIrUtilsIncCov2, IncCov2_ConvertGeModelToModelProto_WithEnvDumpLevel) {
  auto graph = BuildIncCov2Graph();
  ge::Model model("test_env_dump_inc2", "");
  model.SetGraph(graph);
  setenv("DUMP_GE_GRAPH", "1", 1);
  onnx::ModelProto model_proto;
  EXPECT_TRUE(OnnxUtils::ConvertGeModelToModelProto(model, model_proto));
  unsetenv("DUMP_GE_GRAPH");
}

TEST_F(GeIrUtilsIncCov2, IncCov2_AddAttrProtoFromAttribute_DefaultCase) {
  onnx::NodeProto node_proto;
  auto op_desc = std::make_shared<OpDesc>("test_op", "Relu");
  AttrUtils::SetBool(op_desc, "bool_attr", true);
  GeAttrValue attr_value;
  op_desc->GetAttr("bool_attr", attr_value);
  std::pair<const std::string, ge::GeAttrValue> pair("bool_attr", attr_value);
  OnnxUtils::AddAttrProtoFromAttribute(pair, &node_proto);
  EXPECT_EQ(node_proto.attribute_size(), 1);
}

TEST_F(GeIrUtilsIncCov2, IncCov2_EncodeNode_JsonPath_ManyInputs) {
  auto graph = std::make_shared<ComputeGraph>("json_test_graph");
  OpDescPtr op_desc = std::make_shared<OpDesc>("json_op", "Relu");
  GeTensorDesc tensor(GeShape({1}), FORMAT_NCHW, DT_FLOAT);
  for (int i = 0; i < 21; i++) {
    op_desc->AddInputDesc(tensor);
  }
  op_desc->AddOutputDesc(tensor);
  auto node = graph->AddNode(op_desc);
  onnx::NodeProto node_proto;
  EXPECT_TRUE(OnnxUtils::EncodeNode(node, &node_proto));
  EXPECT_GT(node_proto.attribute_size(), 0);
}

TEST_F(GeIrUtilsIncCov2, IncCov2_EncodeNode_JsonPath_ManyOutputs) {
  auto graph = std::make_shared<ComputeGraph>("json_out_graph");
  OpDescPtr op_desc = std::make_shared<OpDesc>("json_out_op", "Relu");
  GeTensorDesc tensor(GeShape({1}), FORMAT_NCHW, DT_FLOAT);
  op_desc->AddInputDesc(tensor);
  for (int i = 0; i < 21; i++) {
    op_desc->AddOutputDesc(tensor);
  }
  auto node = graph->AddNode(op_desc);
  onnx::NodeProto node_proto;
  EXPECT_TRUE(OnnxUtils::EncodeNode(node, &node_proto));
  EXPECT_GT(node_proto.attribute_size(), 0);
}

TEST_F(GeIrUtilsIncCov2, IncCov2_EncodeNodeDesc_ListListIntAttr) {
  auto graph = std::make_shared<ComputeGraph>("list_list_int_graph");
  OpDescPtr op_desc = std::make_shared<OpDesc>("list_list_op", "Relu");
  op_desc->AddInputDesc(GeTensorDesc(GeShape({1}), FORMAT_NCHW, DT_FLOAT));
  op_desc->AddOutputDesc(GeTensorDesc(GeShape({1}), FORMAT_NCHW, DT_FLOAT));
  std::vector<std::vector<int64_t>> list_list_int = {{1, 2}, {3, 4, 5}};
  AttrUtils::SetListListInt(op_desc, "list_list_attr", list_list_int);
  auto node = graph->AddNode(op_desc);
  onnx::NodeProto node_proto;
  EXPECT_TRUE(OnnxUtils::EncodeNodeDesc(node, &node_proto));
  EXPECT_GT(node_proto.attribute_size(), 0);
}

TEST_F(GeIrUtilsIncCov2, IncCov2_DecodeNodeAttributeForOpInAndOutDesc_NullOpDesc) {
  OpDescPtr null_op;
  onnx::AttributeProto attr_proto;
  attr_proto.set_name("input_desc_dtype:0");
  attr_proto.set_type(onnx::AttributeProto_AttributeType_STRING);
  attr_proto.set_s("DT_FLOAT");
  OnnxUtils::DecodeNodeAttributeForOpInAndOutDesc(attr_proto, "input_desc_dtype", 0, null_op);
  SUCCEED();
}

TEST_F(GeIrUtilsIncCov2, IncCov2_DecodeNodeAttributeForOpInAndOutDesc_UnknownPrefix) {
  OpDescPtr op_desc = std::make_shared<OpDesc>("test_unknown_prefix", "TestOp");
  onnx::AttributeProto attr_proto;
  attr_proto.set_name("unknown_field:0");
  attr_proto.set_type(onnx::AttributeProto_AttributeType_INT);
  attr_proto.set_i(0);
  OnnxUtils::DecodeNodeAttributeForOpInAndOutDesc(attr_proto, "unknown_field", 0, op_desc);
  SUCCEED();
}

TEST_F(GeIrUtilsIncCov2, IncCov2_DecodeNodeLink_SrcNodeNullptrInMap) {
  auto graph = std::make_shared<ComputeGraph>("test_nullptr_map");
  auto dst = CreateNodeIncCov2Helper(graph, "dst_null_map", "Relu", 1, 1);

  onnx::NodeProto node_proto;
  node_proto.set_name("dst_null_map");
  node_proto.add_input("src_null:0");

  std::vector<onnx::NodeProto> node_proto_vector;
  node_proto_vector.push_back(node_proto);

  std::map<std::string, NodePtr> node_map;
  node_map["src_null"] = nullptr;
  node_map["dst_null_map"] = dst;

  EXPECT_FALSE(OnnxUtils::DecodeNodeLink(node_proto_vector, node_map));
}

TEST_F(GeIrUtilsIncCov2, IncCov2_DecodeNodeLink_LinkImpFail) {
  auto graph = std::make_shared<ComputeGraph>("test_link_imp_fail");
  auto src = CreateNodeIncCov2Helper(graph, "src_imp_fail", "Data", 1, 1);
  auto dst = CreateNodeIncCov2Helper(graph, "dst_imp_fail", "Relu", 1, 1);

  onnx::NodeProto node_proto;
  node_proto.set_name("dst_imp_fail");
  node_proto.add_input("src_imp_fail:5");

  std::vector<onnx::NodeProto> node_proto_vector;
  node_proto_vector.push_back(node_proto);

  std::map<std::string, NodePtr> node_map;
  node_map["src_imp_fail"] = src;
  node_map["dst_imp_fail"] = dst;

  EXPECT_FALSE(OnnxUtils::DecodeNodeLink(node_proto_vector, node_map));
}

TEST_F(GeIrUtilsIncCov2, IncCov2_DecodeGraph_DecodeNodeDescFail) {
  onnx::GraphProto graph_proto;
  graph_proto.set_name("bad_node_graph");
  auto *node_proto = graph_proto.add_node();
  node_proto->set_name("bad_node");
  node_proto->set_op_type("NoPrefixType");

  ComputeGraphPtr graph;
  EXPECT_FALSE(OnnxUtils::DecodeGraph(0, graph_proto, graph));
}

TEST_F(GeIrUtilsIncCov2, IncCov2_DecodeGraph_SubgraphNonGraphAttr) {
  onnx::GraphProto graph_proto;
  graph_proto.set_name("bad_subgraph_graph");
  auto *node_proto = graph_proto.add_node();
  node_proto->set_name("bad_subgraph");
  node_proto->set_op_type("subgraph");
  auto *attr = node_proto->add_attribute();
  attr->set_name("graph");
  attr->set_type(onnx::AttributeProto_AttributeType_INT);
  attr->set_i(42);

  ComputeGraphPtr graph;
  EXPECT_FALSE(OnnxUtils::DecodeGraph(0, graph_proto, graph));
}

TEST_F(GeIrUtilsIncCov2, IncCov2_DecodeNodeAttributeForOpDesc_DstIndex) {
  OpDescPtr op_desc = std::make_shared<OpDesc>("test_dst_idx", "TestOp");
  onnx::AttributeProto attr_proto;
  attr_proto.set_name("dst_index");
  attr_proto.set_type(onnx::AttributeProto_AttributeType_INTS);
  attr_proto.add_ints(0);
  attr_proto.add_ints(1);
  OnnxUtils::DecodeNodeAttributeForOpDesc(attr_proto, op_desc);
  SUCCEED();
}

TEST_F(GeIrUtilsIncCov2, IncCov2_AddAttrProto_RepeatedInt64_WithData) {
  onnx::NodeProto node_proto;
  ::google::protobuf::RepeatedField<::google::protobuf::int64> data;
  data.Add(10);
  data.Add(20);
  OnnxUtils::AddAttrProto(&node_proto, onnx::AttributeProto_AttributeType_INTS, "test_ints", data);
  EXPECT_EQ(node_proto.attribute_size(), 1);
  EXPECT_EQ(node_proto.attribute(0).ints_size(), 2);
}

TEST_F(GeIrUtilsIncCov2, IncCov2_AddAttrProto_RepeatedBool_WithData) {
  onnx::NodeProto node_proto;
  ::google::protobuf::RepeatedField<bool> data;
  data.Add(true);
  data.Add(false);
  OnnxUtils::AddAttrProto(&node_proto, onnx::AttributeProto_AttributeType_INTS, "test_bools", data);
  EXPECT_EQ(node_proto.attribute_size(), 1);
  EXPECT_EQ(node_proto.attribute(0).ints_size(), 2);
}

TEST_F(GeIrUtilsIncCov2, IncCov2_AddAttrProto_RepeatedFloat_WithData) {
  onnx::NodeProto node_proto;
  ::google::protobuf::RepeatedField<float> data;
  data.Add(1.5F);
  data.Add(2.5F);
  OnnxUtils::AddAttrProto(&node_proto, onnx::AttributeProto_AttributeType_FLOATS, "test_floats", data);
  EXPECT_EQ(node_proto.attribute_size(), 1);
  EXPECT_EQ(node_proto.attribute(0).floats_size(), 2);
}

TEST_F(GeIrUtilsIncCov2, IncCov2_AddAttrProto_RepeatedString_WithData) {
  onnx::NodeProto node_proto;
  ::google::protobuf::RepeatedPtrField<::std::string> data;
  data.Add("hello");
  data.Add("world");
  OnnxUtils::AddAttrProto(&node_proto, onnx::AttributeProto_AttributeType_STRINGS, "test_strings", data);
  EXPECT_EQ(node_proto.attribute_size(), 1);
  EXPECT_EQ(node_proto.attribute(0).strings_size(), 2);
}

TEST_F(GeIrUtilsIncCov2, IncCov2_DecodeNodeAttributeForOpInDesc_Dtype) {
  OpDescPtr op_desc = std::make_shared<OpDesc>("test_in_dtype", "TestOp");
  op_desc->AddInputDesc(GeTensorDesc(GeShape({1}), FORMAT_NCHW, DT_FLOAT));
  onnx::AttributeProto attr_proto;
  attr_proto.set_name("input_desc_dtype:0");
  attr_proto.set_type(onnx::AttributeProto_AttributeType_STRING);
  attr_proto.set_s("DT_INT32");
  OnnxUtils::DecodeNodeAttributeForOpInDesc(attr_proto, "input_desc_dtype", 0, op_desc);
  EXPECT_EQ(op_desc->GetInputDesc(0).GetDataType(), DT_INT32);
}

TEST_F(GeIrUtilsIncCov2, IncCov2_DecodeNodeAttributeForOpInDesc_Shape) {
  OpDescPtr op_desc = std::make_shared<OpDesc>("test_in_shape", "TestOp");
  op_desc->AddInputDesc(GeTensorDesc(GeShape({1}), FORMAT_NCHW, DT_FLOAT));
  onnx::AttributeProto attr_proto;
  attr_proto.set_name("input_desc_shape:0");
  attr_proto.set_type(onnx::AttributeProto_AttributeType_INTS);
  attr_proto.add_ints(2);
  attr_proto.add_ints(3);
  OnnxUtils::DecodeNodeAttributeForOpInDesc(attr_proto, "input_desc_shape", 0, op_desc);
  EXPECT_EQ(op_desc->GetInputDesc(0).GetShape().GetDims(), std::vector<int64_t>({2, 3}));
}

TEST_F(GeIrUtilsIncCov2, IncCov2_DecodeNodeAttributeForOpInDesc_Layout) {
  OpDescPtr op_desc = std::make_shared<OpDesc>("test_in_layout", "TestOp");
  op_desc->AddInputDesc(GeTensorDesc(GeShape({1}), FORMAT_NCHW, DT_FLOAT));
  onnx::AttributeProto attr_proto;
  attr_proto.set_name("input_desc_layout:0");
  attr_proto.set_type(onnx::AttributeProto_AttributeType_STRING);
  attr_proto.set_s("NCHW");
  OnnxUtils::DecodeNodeAttributeForOpInDesc(attr_proto, "input_desc_layout", 0, op_desc);
  EXPECT_EQ(op_desc->GetInputDesc(0).GetFormat(), FORMAT_NCHW);
}

TEST_F(GeIrUtilsIncCov2, IncCov2_DecodeNodeAttributeForOpOutDesc_OriginDtype) {
  OpDescPtr op_desc = std::make_shared<OpDesc>("test_out_origin_dtype", "TestOp");
  op_desc->AddOutputDesc(GeTensorDesc(GeShape({1}), FORMAT_NCHW, DT_FLOAT));
  onnx::AttributeProto attr_proto;
  attr_proto.set_name("output_desc_origin_dtype:0");
  attr_proto.set_type(onnx::AttributeProto_AttributeType_STRING);
  attr_proto.set_s("DT_INT32");
  OnnxUtils::DecodeNodeAttributeForOpOutDesc(attr_proto, "output_desc_origin_dtype", 0, op_desc);
  SUCCEED();
}

}  // namespace ge
