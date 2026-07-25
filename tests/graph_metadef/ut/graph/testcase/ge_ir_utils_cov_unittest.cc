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
class GeIrUtilsCov : public testing::Test {
 protected:
  void SetUp() {}

  void TearDown() {}
};

namespace {
ComputeGraphPtr BuildCovGraph() {
  ut::GraphBuilder builder("cov_graph");
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

ComputeGraphPtr BuildCovGraphWithAttrs() {
  ut::GraphBuilder builder("cov_graph_attrs");
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

ComputeGraphPtr BuildCovGraphWithSubgraph() {
  auto root_builder = ut::GraphBuilder("root_graph");
  auto parent = root_builder.AddNode("parent", PARTITIONEDCALL, 0, 1);
  auto netoutput = root_builder.AddNode("netoutput", NETOUTPUT, 1, 0);
  root_builder.AddDataEdge(parent, 0, netoutput, 0);
  auto root_graph = root_builder.GetGraph();

  auto sub_builder = ut::GraphBuilder("sub_graph");
  auto sub_const = sub_builder.AddNode("sub_const", "Const", 0, 1);
  auto sub_netoutput = sub_builder.AddNode("sub_netoutput", NETOUTPUT, 1, 0);
  sub_builder.AddDataEdge(sub_const, 0, sub_netoutput, 0);
  auto sub_graph = sub_builder.GetGraph();
  sub_graph->SetParentNode(parent);
  sub_graph->SetParentGraph(root_graph);
  parent->GetOpDesc()->AddSubgraphName("f");
  parent->GetOpDesc()->SetSubgraphInstanceName(0, "sub_graph");
  root_graph->AddSubGraph(sub_graph);

  return root_graph;
}
}  // namespace

TEST_F(GeIrUtilsCov, ConvertGeModelToModelProtoSuccess) {
  auto compute_graph = BuildCovGraphWithAttrs();
  ge::Model model("test_model", "");
  model.SetGraph(compute_graph);
  onnx::ModelProto model_proto;
  EXPECT_TRUE(OnnxUtils::ConvertGeModelToModelProto(model, model_proto));
  EXPECT_EQ(model_proto.producer_name(), "test_model");
  EXPECT_TRUE(model_proto.has_graph());
  EXPECT_GT(model_proto.graph().node_size(), 0);
}

TEST_F(GeIrUtilsCov, ConvertGeModelToModelProtoNullGraph) {
  ge::Model model("empty_model", "");
  onnx::ModelProto model_proto;
  EXPECT_FALSE(OnnxUtils::ConvertGeModelToModelProto(model, model_proto));
}

TEST_F(GeIrUtilsCov, ConvertGeModelToModelProtoWithDumpLevel) {
  auto compute_graph = BuildCovGraph();
  ge::Model model("dump_model", "");
  model.SetGraph(compute_graph);
  onnx::ModelProto model_proto;
  EXPECT_TRUE(OnnxUtils::ConvertGeModelToModelProto(model, model_proto, DumpLevel::DUMP_ALL));
  EXPECT_GT(model_proto.graph().node_size(), 0);
}

TEST_F(GeIrUtilsCov, ConvertGeModelToModelProtoWithDumpNoDesc) {
  auto compute_graph = BuildCovGraph();
  ge::Model model("dump_nodesc_model", "");
  model.SetGraph(compute_graph);
  onnx::ModelProto model_proto;
  EXPECT_TRUE(OnnxUtils::ConvertGeModelToModelProto(model, model_proto, DumpLevel::DUMP_WITH_OUT_DESC));
  EXPECT_GT(model_proto.graph().node_size(), 0);
}

TEST_F(GeIrUtilsCov, ConvertGeModelToModelProtoWithSubgraph) {
  auto compute_graph = BuildCovGraphWithSubgraph();
  ge::Model model("subgraph_model", "");
  model.SetGraph(compute_graph);
  onnx::ModelProto model_proto;
  EXPECT_TRUE(OnnxUtils::ConvertGeModelToModelProto(model, model_proto));
  EXPECT_GT(model_proto.graph().node_size(), 0);
}

TEST_F(GeIrUtilsCov, EncodeDataTypeAllTypes) {
  EXPECT_EQ(OnnxUtils::EncodeDataType(DT_INT64), onnx::TensorProto_DataType_INT64);
  EXPECT_EQ(OnnxUtils::EncodeDataType(DT_FLOAT), onnx::TensorProto_DataType_FLOAT);
  EXPECT_EQ(OnnxUtils::EncodeDataType(DT_INT32), onnx::TensorProto_DataType_INT32);
  EXPECT_EQ(OnnxUtils::EncodeDataType(DT_FLOAT16), onnx::TensorProto_DataType_FLOAT16);
  EXPECT_EQ(OnnxUtils::EncodeDataType(DT_BOOL), onnx::TensorProto_DataType_BOOL);
  EXPECT_EQ(OnnxUtils::EncodeDataType(DT_DOUBLE), onnx::TensorProto_DataType_DOUBLE);
  EXPECT_EQ(OnnxUtils::EncodeDataType(DT_UINT8), onnx::TensorProto_DataType_UINT8);
  EXPECT_EQ(OnnxUtils::EncodeDataType(DT_INT8), onnx::TensorProto_DataType_INT8);
  EXPECT_EQ(OnnxUtils::EncodeDataType(DT_DUAL), onnx::TensorProto_DataType_UNDEFINED);
}

TEST_F(GeIrUtilsCov, ParseNameAndIndexValid) {
  std::string node_name;
  int32_t idx = -1;
  EXPECT_TRUE(OnnxUtils::ParseNameAndIndex("node1:0", node_name, idx));
  EXPECT_EQ(node_name, "node1");
  EXPECT_EQ(idx, 0);

  EXPECT_TRUE(OnnxUtils::ParseNameAndIndex("node2:5", node_name, idx));
  EXPECT_EQ(node_name, "node2");
  EXPECT_EQ(idx, 5);
}

TEST_F(GeIrUtilsCov, ParseNameAndIndexNoColon) {
  std::string node_name;
  int32_t idx = -1;
  EXPECT_FALSE(OnnxUtils::ParseNameAndIndex("node_no_colon", node_name, idx));
}

TEST_F(GeIrUtilsCov, DecodeAttributeStrings) {
  onnx::AttributeProto attr_proto;
  attr_proto.set_name("test_strings");
  attr_proto.set_type(onnx::AttributeProto_AttributeType_STRINGS);
  attr_proto.add_strings("value1");
  attr_proto.add_strings("value2");
  std::vector<std::string> strings;
  OnnxUtils::DecodeAttribute(attr_proto, strings);
  EXPECT_EQ(strings.size(), 2U);
  EXPECT_EQ(strings[0], "value1");
  EXPECT_EQ(strings[1], "value2");
}

TEST_F(GeIrUtilsCov, DecodeAttributeString) {
  onnx::AttributeProto attr_proto;
  attr_proto.set_name("test_string");
  attr_proto.set_type(onnx::AttributeProto_AttributeType_STRING);
  attr_proto.set_s("hello_world");
  std::string value;
  OnnxUtils::DecodeAttribute(attr_proto, value);
  EXPECT_EQ(value, "hello_world");
}

TEST_F(GeIrUtilsCov, DecodeAttributeInts) {
  onnx::AttributeProto attr_proto;
  attr_proto.set_name("test_ints");
  attr_proto.set_type(onnx::AttributeProto_AttributeType_INTS);
  attr_proto.add_ints(10);
  attr_proto.add_ints(20);
  attr_proto.add_ints(30);
  std::vector<int64_t> ints;
  OnnxUtils::DecodeAttribute(attr_proto, ints);
  EXPECT_EQ(ints.size(), 3U);
  EXPECT_EQ(ints[0], 10);
  EXPECT_EQ(ints[1], 20);
  EXPECT_EQ(ints[2], 30);
}

TEST_F(GeIrUtilsCov, DecodeAttributeInt) {
  onnx::AttributeProto attr_proto;
  attr_proto.set_name("test_int");
  attr_proto.set_type(onnx::AttributeProto_AttributeType_INT);
  attr_proto.set_i(99);
  int64_t value = 0;
  OnnxUtils::DecodeAttribute(attr_proto, value);
  EXPECT_EQ(value, 99);
}

TEST_F(GeIrUtilsCov, DecodeAttributeWrongType) {
  onnx::AttributeProto attr_proto;
  attr_proto.set_name("wrong_type");
  attr_proto.set_type(onnx::AttributeProto_AttributeType_FLOAT);
  std::vector<std::string> strings;
  strings.push_back("existing");
  OnnxUtils::DecodeAttribute(attr_proto, strings);
  EXPECT_EQ(strings.size(), 1U);

  std::vector<int64_t> ints;
  ints.push_back(1);
  OnnxUtils::DecodeAttribute(attr_proto, ints);
  EXPECT_EQ(ints.size(), 1U);

  int64_t val = 5;
  OnnxUtils::DecodeAttribute(attr_proto, val);
  EXPECT_EQ(val, 5);

  std::string str = "orig";
  OnnxUtils::DecodeAttribute(attr_proto, str);
  EXPECT_EQ(str, "orig");
}

TEST_F(GeIrUtilsCov, IsEqualTemplate) {
  EXPECT_TRUE(IsEqual(1, 1, "int_equal"));
  EXPECT_FALSE(IsEqual(1, 2, "int_not_equal"));
  EXPECT_TRUE(IsEqual(std::string("abc"), std::string("abc"), "str_equal"));
  EXPECT_FALSE(IsEqual(std::string("abc"), std::string("xyz"), "str_not_equal"));
}

TEST_F(GeIrUtilsCov, EncodeNodeSuccess) {
  auto graph = BuildCovGraphWithAttrs();
  auto add_node = graph->FindNode("add1");
  ASSERT_NE(add_node, nullptr);
  onnx::NodeProto node_proto;
  EXPECT_TRUE(OnnxUtils::EncodeNode(add_node, &node_proto));
  EXPECT_EQ(node_proto.name(), "add1");
  EXPECT_TRUE(node_proto.op_type().find("Add") != std::string::npos);
}

TEST_F(GeIrUtilsCov, EncodeNodeNullPtr) {
  NodePtr null_node;
  onnx::NodeProto node_proto;
  EXPECT_FALSE(OnnxUtils::EncodeNode(null_node, &node_proto));
}

TEST_F(GeIrUtilsCov, EncodeGraphSuccess) {
  auto graph = BuildCovGraph();
  onnx::GraphProto graph_proto;
  EXPECT_TRUE(OnnxUtils::EncodeGraph(graph, &graph_proto));
  EXPECT_EQ(graph_proto.name(), "cov_graph");
  EXPECT_GT(graph_proto.node_size(), 0);
}

TEST_F(GeIrUtilsCov, EncodeGraphNullPtr) {
  ConstComputeGraphPtr null_graph;
  onnx::GraphProto graph_proto;
  EXPECT_FALSE(OnnxUtils::EncodeGraph(null_graph, &graph_proto));
}

TEST_F(GeIrUtilsCov, EncodeNodeLinkSuccess) {
  auto graph = BuildCovGraph();
  auto netoutput = graph->FindNode("netoutput1");
  ASSERT_NE(netoutput, nullptr);
  onnx::NodeProto node_proto;
  EXPECT_TRUE(OnnxUtils::EncodeNodeLink(netoutput, &node_proto));
  EXPECT_GT(node_proto.input_size(), 0);
}

TEST_F(GeIrUtilsCov, EncodeNodeDescSuccess) {
  auto graph = BuildCovGraphWithAttrs();
  auto add_node = graph->FindNode("add1");
  ASSERT_NE(add_node, nullptr);
  onnx::NodeProto node_proto;
  EXPECT_TRUE(OnnxUtils::EncodeNodeDesc(add_node, &node_proto));
  bool found_id = false;
  for (const auto &attr : node_proto.attribute()) {
    if (attr.name() == "id") {
      found_id = true;
      break;
    }
  }
  EXPECT_TRUE(found_id);
}

TEST_F(GeIrUtilsCov, DecodeNodeDescSuccess) {
  auto graph = BuildCovGraphWithAttrs();
  auto add_node = graph->FindNode("add1");
  ASSERT_NE(add_node, nullptr);
  onnx::NodeProto node_proto;
  node_proto.set_name("decoded_node");
  node_proto.set_op_type("ge:Add");

  onnx::AttributeProto *nums_in = node_proto.add_attribute();
  nums_in->set_name("input_desc_nums");
  nums_in->set_type(onnx::AttributeProto_AttributeType_INT);
  nums_in->set_i(2);

  onnx::AttributeProto *nums_out = node_proto.add_attribute();
  nums_out->set_name("output_desc_nums");
  nums_out->set_type(onnx::AttributeProto_AttributeType_INT);
  nums_out->set_i(1);

  OpDescPtr op_desc = std::make_shared<OpDesc>();
  EXPECT_TRUE(OnnxUtils::DecodeNodeDesc(&node_proto, op_desc));
  EXPECT_EQ(op_desc->GetName(), "decoded_node");
  EXPECT_EQ(op_desc->GetType(), "Add");
}

TEST_F(GeIrUtilsCov, DecodeNodeDescFailNoColon) {
  onnx::NodeProto node_proto;
  node_proto.set_name("bad_node");
  node_proto.set_op_type("NoPrefix");
  OpDescPtr op_desc = std::make_shared<OpDesc>();
  EXPECT_FALSE(OnnxUtils::DecodeNodeDesc(&node_proto, op_desc));
}

TEST_F(GeIrUtilsCov, DecodeNodeDescNullParams) {
  OpDescPtr op_desc;
  onnx::NodeProto node_proto;
  EXPECT_FALSE(OnnxUtils::DecodeNodeDesc(nullptr, op_desc));
  EXPECT_FALSE(OnnxUtils::DecodeNodeDesc(&node_proto, op_desc));
}

TEST_F(GeIrUtilsCov, DecodeNodeLinkImpDataEdgeSuccess) {
  ut::GraphBuilder builder("test_link");
  auto node1 = builder.AddNode("src_node", "Data", 1, 1);
  auto node2 = builder.AddNode("dst_node", "NetOutput", 1, 0);
  OnnxUtils::NodeLinkInfo item("src_node", 0, node2, 0, "dst_node");
  EXPECT_TRUE(OnnxUtils::DecodeNodeLinkImp(item, node1));
}

TEST_F(GeIrUtilsCov, DecodeNodeLinkImpNullNode) {
  NodePtr null_node;
  OnnxUtils::NodeLinkInfo item("src", 0, null_node, 0, "dst");
  EXPECT_FALSE(OnnxUtils::DecodeNodeLinkImp(item, null_node));
}

TEST_F(GeIrUtilsCov, DecodeNodeLinkImpDataAnchorFail) {
  ut::GraphBuilder builder("test_link_fail");
  auto node1 = builder.AddNode("src_node", "Data", 1, 1);
  auto node2 = builder.AddNode("dst_node", "NetOutput", 1, 0);
  OnnxUtils::NodeLinkInfo item("src_node", 5, node2, 0, "dst_node");
  EXPECT_FALSE(OnnxUtils::DecodeNodeLinkImp(item, node1));
}

TEST_F(GeIrUtilsCov, DecodeNodeLinkImpControlEdgeSuccess) {
  ut::GraphBuilder builder("test_ctrl");
  auto node1 = builder.AddNode("src_node", "Data", 1, 1);
  auto node2 = builder.AddNode("dst_node", "NetOutput", 1, 0);
  OnnxUtils::NodeLinkInfo item("src_node", -1, node2, 0, "dst_node");
  EXPECT_TRUE(OnnxUtils::DecodeNodeLinkImp(item, node1));
}

TEST_F(GeIrUtilsCov, DecodeGraphSuccess) {
  auto graph = BuildCovGraph();
  ge::Model model("decode_model", "");
  model.SetGraph(graph);
  onnx::ModelProto model_proto;
  ASSERT_TRUE(OnnxUtils::ConvertGeModelToModelProto(model, model_proto));

  ComputeGraphPtr decoded_graph;
  EXPECT_TRUE(OnnxUtils::DecodeGraph(0, model_proto.graph(), decoded_graph));
  ASSERT_NE(decoded_graph, nullptr);
}

TEST_F(GeIrUtilsCov, DecodeGraphMaxDepth) {
  onnx::GraphProto graph_proto;
  ComputeGraphPtr graph;
  EXPECT_FALSE(OnnxUtils::DecodeGraph(20, graph_proto, graph));
}

TEST_F(GeIrUtilsCov, AddAttrProtoFromAttributeFloat) {
  onnx::NodeProto node_proto;
  std::pair<const std::string, ge::GeAttrValue> attr_pair("float_attr", ge::GeAttrValue());
  auto op_desc = std::make_shared<OpDesc>("test_op", "Relu");
  AttrUtils::SetFloat(op_desc, "float_attr", 1.5F);
  GeAttrValue attr_value;
  op_desc->GetAttr("float_attr", attr_value);
  std::pair<const std::string, ge::GeAttrValue> pair("float_attr", attr_value);
  OnnxUtils::AddAttrProtoFromAttribute(pair, &node_proto);
  EXPECT_EQ(node_proto.attribute_size(), 1);
  EXPECT_EQ(node_proto.attribute(0).name(), "float_attr");
  EXPECT_EQ(node_proto.attribute(0).type(), onnx::AttributeProto_AttributeType_FLOAT);
  EXPECT_FLOAT_EQ(node_proto.attribute(0).f(), 1.5F);
}

TEST_F(GeIrUtilsCov, AddAttrProtoFromAttributeInt) {
  onnx::NodeProto node_proto;
  auto op_desc = std::make_shared<OpDesc>("test_op", "Relu");
  AttrUtils::SetInt(op_desc, "int_attr", 42);
  GeAttrValue attr_value;
  op_desc->GetAttr("int_attr", attr_value);
  std::pair<const std::string, ge::GeAttrValue> pair("int_attr", attr_value);
  OnnxUtils::AddAttrProtoFromAttribute(pair, &node_proto);
  EXPECT_EQ(node_proto.attribute_size(), 1);
  EXPECT_EQ(node_proto.attribute(0).name(), "int_attr");
  EXPECT_EQ(node_proto.attribute(0).type(), onnx::AttributeProto_AttributeType_INT);
  EXPECT_EQ(node_proto.attribute(0).i(), 42);
}

TEST_F(GeIrUtilsCov, AddAttrProtoFromAttributeString) {
  onnx::NodeProto node_proto;
  auto op_desc = std::make_shared<OpDesc>("test_op", "Relu");
  AttrUtils::SetStr(op_desc, "str_attr", "test_value");
  GeAttrValue attr_value;
  op_desc->GetAttr("str_attr", attr_value);
  std::pair<const std::string, ge::GeAttrValue> pair("str_attr", attr_value);
  OnnxUtils::AddAttrProtoFromAttribute(pair, &node_proto);
  EXPECT_EQ(node_proto.attribute_size(), 1);
  EXPECT_EQ(node_proto.attribute(0).name(), "str_attr");
  EXPECT_EQ(node_proto.attribute(0).type(), onnx::AttributeProto_AttributeType_STRING);
  EXPECT_EQ(node_proto.attribute(0).s(), "test_value");
}

TEST_F(GeIrUtilsCov, AddAttrProtoFromAttributeListInt) {
  onnx::NodeProto node_proto;
  auto op_desc = std::make_shared<OpDesc>("test_op", "Relu");
  AttrUtils::SetListInt(op_desc, "list_int_attr", {1, 2, 3});
  GeAttrValue attr_value;
  op_desc->GetAttr("list_int_attr", attr_value);
  std::pair<const std::string, ge::GeAttrValue> pair("list_int_attr", attr_value);
  OnnxUtils::AddAttrProtoFromAttribute(pair, &node_proto);
  EXPECT_EQ(node_proto.attribute_size(), 1);
  EXPECT_EQ(node_proto.attribute(0).name(), "list_int_attr");
  EXPECT_EQ(node_proto.attribute(0).type(), onnx::AttributeProto_AttributeType_INTS);
  EXPECT_EQ(node_proto.attribute(0).ints_size(), 3);
}

TEST_F(GeIrUtilsCov, AddAttrProtoFromAttributeListFloat) {
  onnx::NodeProto node_proto;
  auto op_desc = std::make_shared<OpDesc>("test_op", "Relu");
  AttrUtils::SetListFloat(op_desc, "list_float_attr", {1.0F, 2.0F});
  GeAttrValue attr_value;
  op_desc->GetAttr("list_float_attr", attr_value);
  std::pair<const std::string, ge::GeAttrValue> pair("list_float_attr", attr_value);
  OnnxUtils::AddAttrProtoFromAttribute(pair, &node_proto);
  EXPECT_EQ(node_proto.attribute_size(), 1);
  EXPECT_EQ(node_proto.attribute(0).name(), "list_float_attr");
  EXPECT_EQ(node_proto.attribute(0).type(), onnx::AttributeProto_AttributeType_FLOATS);
  EXPECT_EQ(node_proto.attribute(0).floats_size(), 2);
}

TEST_F(GeIrUtilsCov, AddAttrProtoFromAttributeListString) {
  onnx::NodeProto node_proto;
  auto op_desc = std::make_shared<OpDesc>("test_op", "Relu");
  AttrUtils::SetListStr(op_desc, "list_str_attr", {"a", "b"});
  GeAttrValue attr_value;
  op_desc->GetAttr("list_str_attr", attr_value);
  std::pair<const std::string, ge::GeAttrValue> pair("list_str_attr", attr_value);
  OnnxUtils::AddAttrProtoFromAttribute(pair, &node_proto);
  EXPECT_EQ(node_proto.attribute_size(), 1);
  EXPECT_EQ(node_proto.attribute(0).name(), "list_str_attr");
  EXPECT_EQ(node_proto.attribute(0).type(), onnx::AttributeProto_AttributeType_STRINGS);
  EXPECT_EQ(node_proto.attribute(0).strings_size(), 2);
}

TEST_F(GeIrUtilsCov, AddAttrProtoFromAttributeNullNodeProto) {
  std::pair<const std::string, ge::GeAttrValue> pair("attr", ge::GeAttrValue());
  OnnxUtils::AddAttrProtoFromAttribute(pair, nullptr);
  SUCCEED();
}

TEST_F(GeIrUtilsCov, AddAttrProtoNullNodeProto) {
  float val = 1.0F;
  OnnxUtils::AddAttrProto(nullptr, onnx::AttributeProto_AttributeType_FLOAT, "test", &val);
  SUCCEED();
}

TEST_F(GeIrUtilsCov, AddAttrProtoFloat) {
  onnx::NodeProto node_proto;
  float val = 3.14F;
  OnnxUtils::AddAttrProto(&node_proto, onnx::AttributeProto_AttributeType_FLOAT, "float_attr", &val);
  EXPECT_EQ(node_proto.attribute_size(), 1);
  EXPECT_FLOAT_EQ(node_proto.attribute(0).f(), 3.14F);
}

TEST_F(GeIrUtilsCov, AddAttrProtoInt) {
  onnx::NodeProto node_proto;
  int64_t val = 100;
  OnnxUtils::AddAttrProto(&node_proto, onnx::AttributeProto_AttributeType_INT, "int_attr", &val);
  EXPECT_EQ(node_proto.attribute_size(), 1);
  EXPECT_EQ(node_proto.attribute(0).i(), 100);
}

TEST_F(GeIrUtilsCov, AddAttrProtoString) {
  onnx::NodeProto node_proto;
  std::string val = "test_str";
  OnnxUtils::AddAttrProto(&node_proto, onnx::AttributeProto_AttributeType_STRING, "str_attr", &val);
  EXPECT_EQ(node_proto.attribute_size(), 1);
  EXPECT_EQ(node_proto.attribute(0).s(), "test_str");
}

TEST_F(GeIrUtilsCov, AddAttrProtoFloats) {
  onnx::NodeProto node_proto;
  std::vector<float> val = {1.0F, 2.0F, 3.0F};
  OnnxUtils::AddAttrProto(&node_proto, onnx::AttributeProto_AttributeType_FLOATS, "floats_attr", &val);
  EXPECT_EQ(node_proto.attribute_size(), 1);
  EXPECT_EQ(node_proto.attribute(0).floats_size(), 3);
}

TEST_F(GeIrUtilsCov, AddAttrProtoInts) {
  onnx::NodeProto node_proto;
  std::vector<int64_t> val = {10, 20, 30};
  OnnxUtils::AddAttrProto(&node_proto, onnx::AttributeProto_AttributeType_INTS, "ints_attr", &val);
  EXPECT_EQ(node_proto.attribute_size(), 1);
  EXPECT_EQ(node_proto.attribute(0).ints_size(), 3);
}

TEST_F(GeIrUtilsCov, AddAttrProtoStrings) {
  onnx::NodeProto node_proto;
  std::vector<std::string> val = {"a", "b"};
  OnnxUtils::AddAttrProto(&node_proto, onnx::AttributeProto_AttributeType_STRINGS, "strings_attr", &val);
  EXPECT_EQ(node_proto.attribute_size(), 1);
  EXPECT_EQ(node_proto.attribute(0).strings_size(), 2);
}

TEST_F(GeIrUtilsCov, AddAttrProtoUnsupportedType) {
  onnx::NodeProto node_proto;
  int64_t val = 1;
  OnnxUtils::AddAttrProto(&node_proto, static_cast<onnx::AttributeProto_AttributeType>(999), "unsupported", &val);
  EXPECT_EQ(node_proto.attribute_size(), 1);
}

TEST_F(GeIrUtilsCov, DecodeNodeAttributeForOpInDesc) {
  auto op_desc = std::make_shared<OpDesc>("test_op", "Relu");
  GeTensorDesc tensor_desc(GeShape({4}), FORMAT_NCHW, DT_FLOAT);
  op_desc->AddInputDesc(tensor_desc);

  onnx::AttributeProto attr_proto;
  attr_proto.set_name("input_desc_dtype:0");
  attr_proto.set_type(onnx::AttributeProto_AttributeType_STRING);
  attr_proto.set_s("DT_INT32");
  OnnxUtils::DecodeNodeAttributeForOpInAndOutDesc(attr_proto, "input_desc_dtype", 0, op_desc);
  EXPECT_EQ(op_desc->GetInputDesc(0).GetDataType(), DT_INT32);
}

TEST_F(GeIrUtilsCov, DecodeNodeAttributeForOpOutDesc) {
  auto op_desc = std::make_shared<OpDesc>("test_op", "Relu");
  GeTensorDesc tensor_desc(GeShape({4}), FORMAT_NCHW, DT_FLOAT);
  op_desc->AddOutputDesc(tensor_desc);

  onnx::AttributeProto attr_proto;
  attr_proto.set_name("output_desc_dtype:0");
  attr_proto.set_type(onnx::AttributeProto_AttributeType_STRING);
  attr_proto.set_s("DT_INT32");
  OnnxUtils::DecodeNodeAttributeForOpOutDesc(attr_proto, "output_desc_dtype", 0, op_desc);
  EXPECT_EQ(op_desc->GetOutputDesc(0).GetDataType(), DT_INT32);
}

TEST_F(GeIrUtilsCov, DecodeNodeAttributeForOpDescStreamId) {
  auto op_desc = std::make_shared<OpDesc>("test_op", "Relu");
  onnx::AttributeProto attr_proto;
  attr_proto.set_name("stream_id");
  attr_proto.set_type(onnx::AttributeProto_AttributeType_INT);
  attr_proto.set_i(7);
  OnnxUtils::DecodeNodeAttributeForOpDesc(attr_proto, op_desc);
  EXPECT_EQ(op_desc->GetStreamId(), 7);
}

TEST_F(GeIrUtilsCov, DecodeNodeAttributeForOpDescNullPtr) {
  OpDescPtr null_op;
  onnx::AttributeProto attr_proto;
  OnnxUtils::DecodeNodeAttributeForOpDesc(attr_proto, null_op);
  SUCCEED();
}

TEST_F(GeIrUtilsCov, EncodeValueInfoSuccess) {
  auto graph = BuildCovGraph();
  auto data_node = graph->FindNode("data1");
  ASSERT_NE(data_node, nullptr);
  onnx::ValueInfoProto value_info;
  OnnxUtils::EncodeValueInfo(data_node, &value_info);
  EXPECT_EQ(value_info.name(), "data1");
}

TEST_F(GeIrUtilsCov, EncodeValueInfoNullPtr) {
  NodePtr null_node;
  onnx::ValueInfoProto value_info;
  OnnxUtils::EncodeValueInfo(null_node, &value_info);
  SUCCEED();
}

TEST_F(GeIrUtilsCov, AddInputAndOutputNodesForGraphSuccess) {
  auto graph = BuildCovGraph();
  ge::Model model("test", "");
  model.SetGraph(graph);
  onnx::ModelProto model_proto;
  ASSERT_TRUE(OnnxUtils::ConvertGeModelToModelProto(model, model_proto));

  ComputeGraphPtr new_graph = std::make_shared<ComputeGraph>("new_graph");
  std::map<std::string, NodePtr> node_map;
  for (const auto &node : graph->GetDirectNode()) {
    node_map[node->GetName()] = node;
  }
  EXPECT_TRUE(OnnxUtils::AddInputAndOutputNodesForGraph(model_proto.graph(), new_graph, node_map));
}
}  // namespace ge
