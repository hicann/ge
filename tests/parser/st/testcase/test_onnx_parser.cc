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
#include <cstdlib>
#include <dlfcn.h>
#include <iostream>
#include <string>
#include "parser/common/op_parser_factory.h"
#include "graph/operator_reg.h"
#include "graph/utils/attr_utils.h"
#include "graph/utils/graph_utils_ex.h"
#include "register/op_registry.h"
#include "parser/common/op_registration_tbe.h"
#include "parser/onnx_parser.h"
#include "parser/onnx/python_onnx_plugin_bridge/onnx_plugin_bridge_loader.h"
#include "st/parser_st_utils.h"
#include "ge/ge_api_types.h"
#include "depends/ops_stub/ops_stub.h"
#include "framework/omg/parser/parser_factory.h"
#include "parser/onnx/onnx_util.h"
#include "graph/ge_local_context.h"
#include "common/ge_common/ge_types.h"
#include "parser/onnx/onnx_parser_internal.h"
#include "parser/onnx/onnx_file_constant_parser.h"
#include "common/python_runtime/ge_python_runtime_manager.h"
#include "onnx_plugin_test_helper.h"

namespace ge {
REG_OP(BridgeEluTarget).INPUT(x, TensorType::ALL()).OUTPUT(y, TensorType::ALL()).OP_END_FACTORY_REG(BridgeEluTarget);

using onnx_plugin_test::ScopedInMemoryPlugin;

class STestOnnxParser : public testing::Test {
 protected:
  void SetUp() {
    ParerSTestsUtils::ClearParserInnerCtx();
    RegisterCustomOp();
  }

  void TearDown() {}

 public:
  void RegisterCustomOp();
};

static Status ParseParams(const google::protobuf::Message *op_src, ge::Operator &op_dest) {
  return SUCCESS;
}

static Status ParseParamByOpFunc(const ge::Operator &op_src, ge::Operator &op_dest) {
  return SUCCESS;
}

Status ParseSubgraphPostFnIf(const std::string &subgraph_name, const ge::Graph &graph) {
  domi::AutoMappingSubgraphIOIndexFunc auto_mapping_subgraph_index_func =
      domi::FrameworkRegistry::Instance().GetAutoMappingSubgraphIOIndexFunc(domi::ONNX);
  if (auto_mapping_subgraph_index_func == nullptr) {
    std::cout << "auto mapping if subgraph func is nullptr!" << std::endl;
    return FAILED;
  }
  return auto_mapping_subgraph_index_func(
      graph,
      [&](int data_index, int &parent_index) -> Status {
        parent_index = data_index + 1;
        return SUCCESS;
      },
      [&](int output_index, int &parent_index) -> Status {
        parent_index = output_index;
        return SUCCESS;
      });
}

void STestOnnxParser::RegisterCustomOp() {
  REGISTER_CUSTOM_OP("Conv2D").FrameworkType(domi::ONNX).OriginOpType("ai.onnx::11::Conv").ParseParamsFn(ParseParams);

  // register if op info to GE
  REGISTER_CUSTOM_OP("If")
      .FrameworkType(domi::ONNX)
      .OriginOpType({"ai.onnx::9::If", "ai.onnx::10::If", "ai.onnx::11::If", "ai.onnx::12::If", "ai.onnx::13::If"})
      .ParseParamsFn(ParseParams)
      .ParseParamsByOperatorFn(ParseParamByOpFunc)
      .ParseSubgraphPostFn(ParseSubgraphPostFnIf);

  REGISTER_CUSTOM_OP("Add").FrameworkType(domi::ONNX).OriginOpType("ai.onnx::11::Add").ParseParamsFn(ParseParams);

  REGISTER_CUSTOM_OP("Identity")
      .FrameworkType(domi::ONNX)
      .OriginOpType("ai.onnx::11::Identity")
      .ParseParamsFn(ParseParams);

  std::vector<OpRegistrationData> reg_datas = domi::OpRegistry::Instance()->registrationDatas;
  for (auto reg_data : reg_datas) {
    domi::OpRegTbeParserFactory::Instance()->Finalize(reg_data);
    domi::OpRegistry::Instance()->Register(reg_data);
  }
  domi::OpRegistry::Instance()->registrationDatas.clear();
}

ge::onnx::GraphProto CreateOnnxGraph(const std::string &op_type = "Add") {
  ge::onnx::GraphProto onnx_graph;
  (void)onnx_graph.add_input();
  (void)onnx_graph.add_output();
  ::ge::onnx::NodeProto *node_const1 = onnx_graph.add_node();
  ::ge::onnx::NodeProto *node_const2 = onnx_graph.add_node();
  ::ge::onnx::NodeProto *node_add = onnx_graph.add_node();
  node_const1->set_op_type(kOpTypeConstant);
  node_const2->set_op_type(kOpTypeConstant);
  node_add->set_op_type(op_type);

  ::ge::onnx::AttributeProto *attr = node_const1->add_attribute();
  attr->set_name(ge::kAttrNameValue);
  ::ge::onnx::TensorProto *tensor_proto = attr->mutable_t();
  tensor_proto->set_data_location(ge::onnx::TensorProto_DataLocation_EXTERNAL);
  attr = node_const1->add_attribute();
  tensor_proto->add_external_data();
  ge::onnx::StringStringEntryProto *string_proto = tensor_proto->add_external_data();
  string_proto->set_key("location");
  string_proto->set_value("const.onnx");

  attr = node_const2->add_attribute();
  attr->set_name(ge::kAttrNameValue);
  tensor_proto = attr->mutable_t();
  tensor_proto->set_data_location(ge::onnx::TensorProto_DataLocation_DEFAULT);

  return onnx_graph;
}

TEST_F(STestOnnxParser, onnx_parser_user_output_with_default) {
  std::string case_dir = __FILE__;
  case_dir = case_dir.substr(0, case_dir.find_last_of("/"));
  std::string model_file = case_dir + "/origin_models/onnx_conv2d.onnx";
  std::map<ge::AscendString, ge::AscendString> parser_params;
  ge::Graph graph;
  auto ret = ge::aclgrphParseONNX(model_file.c_str(), parser_params, graph);
  ASSERT_EQ(ret, GRAPH_SUCCESS);
  ge::ComputeGraphPtr compute_graph = ge::GraphUtilsEx::GetComputeGraph(graph);
  auto output_nodes_info = compute_graph->GetGraphOutNodesInfo();
  ASSERT_EQ(output_nodes_info.size(), 1);
  EXPECT_EQ((output_nodes_info.at(0).first->GetName()), "Conv_0");
  EXPECT_EQ((output_nodes_info.at(0).second), 0);
  auto &net_out_name = ge::GetParserContext().net_out_nodes;
  ASSERT_EQ(net_out_name.size(), 1);
  EXPECT_EQ(net_out_name.at(0), "Conv_0:0:y");
}

TEST_F(STestOnnxParser, onnx_parser_precheck) {
  std::string case_dir = __FILE__;
  case_dir = case_dir.substr(0, case_dir.find_last_of("/"));
  std::string model_file = case_dir + "/origin_models/onnx_conv2d.onnx";
  std::map<ge::AscendString, ge::AscendString> parser_params;
  ge::Graph graph;
  ge::GetParserContext().run_mode = ge::ONLY_PRE_CHECK;
  auto ret = ge::aclgrphParseONNX(model_file.c_str(), parser_params, graph);
  ASSERT_EQ(ret, GRAPH_FAILED);
}

TEST_F(STestOnnxParser, onnx_parser_if_node) {
  std::string case_dir = __FILE__;
  case_dir = case_dir.substr(0, case_dir.find_last_of("/"));
  std::string model_file = case_dir + "/origin_models/onnx_if.onnx";
  std::map<ge::AscendString, ge::AscendString> parser_params;
  ge::Graph graph;
  auto ret = ge::aclgrphParseONNX(model_file.c_str(), parser_params, graph);
  // has circle struct, topo sort failed
  EXPECT_EQ(ret, FAILED);
}

TEST_F(STestOnnxParser, onnx_parser_expand_one_to_many) {
  std::string case_dir = __FILE__;
  case_dir = case_dir.substr(0, case_dir.find_last_of("/"));
  std::string model_file = case_dir + "/origin_models/onnx_clip_v9.onnx";
  std::map<ge::AscendString, ge::AscendString> parser_params;
  ge::Graph graph;
  auto ret = ge::aclgrphParseONNX(model_file.c_str(), parser_params, graph);
  EXPECT_EQ(ret, GRAPH_SUCCESS);

  MemBuffer *buffer = ParerSTestsUtils::MemBufferFromFile(model_file.c_str());
  ret = ge::aclgrphParseONNXFromMem(reinterpret_cast<char *>(buffer->data), buffer->size, parser_params, graph);
  EXPECT_EQ(ret, GRAPH_SUCCESS);
}

TEST_F(STestOnnxParser, onnx_parser_expand_one_to_many_with_stable_sort) {
  std::string case_dir = __FILE__;
  case_dir = case_dir.substr(0, case_dir.find_last_of("/"));
  std::string model_file = case_dir + "/origin_models/onnx_clip_v9.onnx";
  std::map<ge::AscendString, ge::AscendString> parser_params;
  ge::Graph graph;
  auto graph_options = GetThreadLocalContext().GetAllGraphOptions();
  graph_options[OPTION_TOPOSORTING_MODE] = "3";
  GetThreadLocalContext().SetGraphOption(graph_options);
  auto ret = ge::aclgrphParseONNX(model_file.c_str(), parser_params, graph);
  EXPECT_EQ(ret, GRAPH_SUCCESS);
  graph_options = GetThreadLocalContext().GetAllGraphOptions();
  graph_options[OPTION_TOPOSORTING_MODE] = "";
  GetThreadLocalContext().SetGraphOption(graph_options);

  MemBuffer *buffer = ParerSTestsUtils::MemBufferFromFile(model_file.c_str());
  ret = ge::aclgrphParseONNXFromMem(reinterpret_cast<char *>(buffer->data), buffer->size, parser_params, graph);
  EXPECT_EQ(ret, GRAPH_SUCCESS);
}

TEST_F(STestOnnxParser, onnx_parser_to_json) {
  std::string case_dir = __FILE__;
  case_dir = case_dir.substr(0, case_dir.find_last_of("/"));
  std::string model_file = case_dir + "/origin_models/onnx_clip_v9.onnx";
  std::map<ge::AscendString, ge::AscendString> parser_params;
  OnnxModelParser onnx_parser;

  const char *json_file = "tmp.json";
  auto ret = onnx_parser.ToJson(model_file.c_str(), json_file);
  EXPECT_EQ(ret, SUCCESS);

  const char *json_null = nullptr;
  ret = onnx_parser.ToJson(model_file.c_str(), json_null);
  EXPECT_EQ(ret, FAILED);
  const char *model_null = nullptr;
  ret = onnx_parser.ToJson(model_null, json_null);
  EXPECT_EQ(ret, FAILED);
}

TEST_F(STestOnnxParser, onnx_parser_const_data_type) {
  std::string case_dir = __FILE__;
  case_dir = case_dir.substr(0, case_dir.find_last_of("/"));
  std::string model_file = case_dir + "/origin_models/onnx_const_type.onnx";
  std::map<ge::AscendString, ge::AscendString> parser_params;
  ge::Graph graph;
  auto ret = ge::aclgrphParseONNX(model_file.c_str(), parser_params, graph);
  EXPECT_EQ(ret, GRAPH_SUCCESS);
}

TEST_F(STestOnnxParser, onnx_parser_if_node_with_const_input) {
  std::string case_dir = __FILE__;
  case_dir = case_dir.substr(0, case_dir.find_last_of("/"));
  std::string model_file = case_dir + "/origin_models/onnx_if_const_intput.onnx";
  std::map<ge::AscendString, ge::AscendString> parser_params;
  ge::Graph graph;
  auto ret = ge::aclgrphParseONNX(model_file.c_str(), parser_params, graph);
  EXPECT_EQ(ret, GRAPH_SUCCESS);
}

TEST_F(STestOnnxParser, onnx_test_ModelParseToGraph) {
  OnnxModelParser modelParser;
  ge::onnx::ModelProto model_proto;
  auto onnx_graph = model_proto.mutable_graph();
  *onnx_graph = CreateOnnxGraph();
  ge::onnx::OperatorSetIdProto *op_st = model_proto.add_opset_import();
  op_st->set_domain("ai.onnx");
  op_st->set_version(11);
  ge::Graph graph;

  Status ret = modelParser.ModelParseToGraph(model_proto, graph);
  EXPECT_EQ(ret, INTERNAL_ERROR);
}

TEST_F(STestOnnxParser, FileConstantParseParam) {
  OnnxFileConstantParser parser;
  ge::onnx::NodeProto input_node;
  ge::OpDescPtr op_desc_src = std::make_shared<ge::OpDesc>("file_constant", "FileConstant");
  ge::Operator op = ge::OpDescUtils::CreateOperatorFromOpDesc(op_desc_src);

  ge::onnx::TensorProto tensor_proto;
  ge::onnx::AttributeProto *attribute = input_node.add_attribute();
  attribute->set_name("value");
  ge::onnx::TensorProto *attribute_tensor = attribute->mutable_t();
  *attribute_tensor = tensor_proto;
  attribute_tensor->set_data_type(OnnxDataType::UINT16);
  attribute_tensor->add_dims(4);

  ge::onnx::StringStringEntryProto *string_proto1 = attribute_tensor->add_external_data();
  string_proto1->set_key("location");
  string_proto1->set_value("/tmp/weight");
  ge::onnx::StringStringEntryProto *string_proto2 = attribute_tensor->add_external_data();
  string_proto2->set_key("offset");
  string_proto2->set_value("4");
  ge::onnx::StringStringEntryProto *string_proto3 = attribute_tensor->add_external_data();
  string_proto3->set_key("length");
  string_proto3->set_value("16");
  Status ret = parser.ParseParams(reinterpret_cast<Message *>(&input_node), op);
  EXPECT_EQ(ret, SUCCESS);
}

TEST_F(STestOnnxParser, onnx_test_PreChecker_not_support) {
  OnnxModelParser modelParser;
  ge::onnx::ModelProto model_proto;
  auto onnx_graph = model_proto.mutable_graph();
  *onnx_graph = CreateOnnxGraph("Test");
  ge::onnx::OperatorSetIdProto *op_st = model_proto.add_opset_import();
  op_st->set_domain("ai.onnx");
  op_st->set_version(11);
  ge::Graph graph;

  Status ret = modelParser.ModelParseToGraph(model_proto, graph);
  EXPECT_EQ(ret, FAILED);

  EXPECT_EQ(PreChecker::Instance().HasError(), true);
}

TEST_F(STestOnnxParser, onnx_test_SetExternalPath) {
  OnnxModelParser modelParser;
  ge::onnx::ModelProto model_proto;
  auto onnx_graph = model_proto.mutable_graph();
  *onnx_graph = CreateOnnxGraph("Test");

  auto ret = modelParser.SetExternalPath("/usr/local", model_proto);
  EXPECT_EQ(ret, SUCCESS);
}

static ge::onnx::ModelProto CreateInt4ModelProto(bool use_raw_data) {
  ge::onnx::ModelProto model_proto;
  auto *onnx_graph = model_proto.mutable_graph();

  auto *input = onnx_graph->add_input();
  input->set_name("A");
  auto *in_type = input->mutable_type()->mutable_tensor_type();
  in_type->set_elem_type(OnnxDataType::FLOAT);
  in_type->mutable_shape()->add_dim()->set_dim_value(8);

  auto *output = onnx_graph->add_output();
  output->set_name("Y");
  auto *out_type = output->mutable_type()->mutable_tensor_type();
  out_type->set_elem_type(OnnxDataType::FLOAT);
  out_type->mutable_shape()->add_dim()->set_dim_value(8);

  auto *const_node = onnx_graph->add_node();
  const_node->set_op_type(kOpTypeConstant);
  const_node->add_output("const_int4_out");
  auto *attr = const_node->add_attribute();
  attr->set_name(ge::kAttrNameValue);
  auto *tensor_proto = attr->mutable_t();
  tensor_proto->set_data_type(OnnxDataType::INT4);
  tensor_proto->add_dims(8);
  if (use_raw_data) {
    tensor_proto->set_raw_data(std::string("\x10\x32\x54\x76", 4));
  } else {
    tensor_proto->add_int32_data(0x76543210);
  }

  auto *identity_node = onnx_graph->add_node();
  identity_node->set_op_type("Identity");
  identity_node->add_input("A");
  identity_node->add_output("Y");

  auto *op_st = model_proto.add_opset_import();
  op_st->set_domain("ai.onnx");
  op_st->set_version(11);
  return model_proto;
}

static void VerifyInt4ConstantNode(const ge::Graph &graph) {
  auto compute_graph = ge::GraphUtilsEx::GetComputeGraph(graph);
  ASSERT_NE(compute_graph, nullptr);
  ge::NodePtr constant_node = nullptr;
  for (const auto &node : compute_graph->GetAllNodes()) {
    if (node->GetType() == "Const") {
      constant_node = node;
      break;
    }
  }
  ASSERT_NE(constant_node, nullptr);
  std::shared_ptr<const ge::GeTensor> tensor = nullptr;
  EXPECT_EQ(ge::AttrUtils::GetTensor(constant_node->GetOpDesc(), ge::kAttrNameValue, tensor), true);
  ASSERT_NE(tensor, nullptr);
  EXPECT_EQ(tensor->GetTensorDesc().GetDataType(), ge::DataType::DT_INT4);
  const ge::TensorData &tensor_data = tensor->GetData();
  EXPECT_EQ(tensor_data.GetSize(), 4U);
  const uint8_t *data = tensor_data.GetData();
  ASSERT_NE(data, nullptr);
  EXPECT_EQ(data[0], 0x10);
  EXPECT_EQ(data[1], 0x32);
  EXPECT_EQ(data[2], 0x54);
  EXPECT_EQ(data[3], 0x76);
}

/**
 * 用例描述：测试ONNX INT4类型Constant节点raw_data路径的解析
 * 预期结果：解析成功，DataType为DT_INT4，数据为packed字节 0x10,0x32,0x54,0x76
 */
TEST_F(STestOnnxParser, onnx_parser_int4_const_raw_data) {
  OnnxModelParser modelParser;
  auto model_proto = CreateInt4ModelProto(true);
  ge::Graph graph;
  EXPECT_EQ(modelParser.ModelParseToGraph(model_proto, graph), SUCCESS);
  VerifyInt4ConstantNode(graph);
}

/**
 * 用例描述：测试ONNX INT4类型Constant节点int32_data路径的解析
 * 预期结果：解析成功，DataType为DT_INT4，数据为packed字节（little-endian）
 */
TEST_F(STestOnnxParser, onnx_parser_int4_const_int32_data) {
  OnnxModelParser modelParser;
  auto model_proto = CreateInt4ModelProto(false);
  ge::Graph graph;
  EXPECT_EQ(modelParser.ModelParseToGraph(model_proto, graph), SUCCESS);
  VerifyInt4ConstantNode(graph);
}

ge::onnx::ModelProto CreateBridgePluginModel() {
  ge::onnx::ModelProto plugin_model;
  auto *plugin_graph = plugin_model.mutable_graph();
  auto *plugin_input = plugin_graph->add_input();
  plugin_input->set_name("X");
  plugin_input->mutable_type()->mutable_tensor_type()->set_elem_type(ge::onnx::TensorProto_DataType_FLOAT);
  auto *plugin_output = plugin_graph->add_output();
  plugin_output->set_name("Y");
  plugin_output->mutable_type()->mutable_tensor_type()->set_elem_type(ge::onnx::TensorProto_DataType_FLOAT);
  auto *plugin_node = plugin_graph->add_node();
  plugin_node->set_name("bridge_elu");
  plugin_node->set_domain("test.domain");
  plugin_node->set_op_type("BridgeElu");
  plugin_node->add_input("X");
  plugin_node->add_output("Y");
  auto *plugin_alpha = plugin_node->add_attribute();
  plugin_alpha->set_name("alpha");
  plugin_alpha->set_type(ge::onnx::AttributeProto_AttributeType_FLOAT);
  plugin_alpha->set_f(0.5F);
  auto *plugin_opset = plugin_model.add_opset_import();
  plugin_opset->set_domain("test.domain");
  plugin_opset->set_version(1);
  return plugin_model;
}
void VerifyBridgePluginNode(const ge::Graph &plugin_result) {
  const auto plugin_compute_graph = ge::GraphUtilsEx::GetComputeGraph(plugin_result);
  ASSERT_NE(plugin_compute_graph, nullptr);
  const auto parsed_plugin_node = plugin_compute_graph->FindNode("bridge_elu");
  ASSERT_NE(parsed_plugin_node, nullptr);
  float parsed_alpha = 0.0F;
  ASSERT_TRUE(ge::AttrUtils::GetFloat(parsed_plugin_node->GetOpDesc(), "alpha", parsed_alpha));
  EXPECT_FLOAT_EQ(parsed_alpha, 0.5F);
}
void VerifyBridgePluginCallbacks() {
  ge::onnx::NodeProto node;
  node.set_op_type("test.domain::1::BridgeElu");
  auto *alpha = node.add_attribute();
  alpha->set_name("alpha");
  alpha->set_type(ge::onnx::AttributeProto_AttributeType_FLOAT);
  alpha->set_f(0.5F);
  Operator op("bridge_node", "BridgeEluTarget");
  const auto parse_elu = domi::OpRegistry::Instance()->GetParseParamFunc("BridgeEluTarget", node.op_type());
  ASSERT_NE(parse_elu, nullptr);
  EXPECT_EQ(parse_elu(&node, op), SUCCESS);
  EXPECT_NE(parse_elu(nullptr, op), SUCCESS);
  ge::onnx::AttributeProto wrong_msg;
  EXPECT_NE(parse_elu(&wrong_msg, op), SUCCESS);
  node.set_op_type("test.domain::1::BridgeError");
  const auto parse_error = domi::OpRegistry::Instance()->GetParseParamFunc("BridgeErrorTarget", node.op_type());
  ASSERT_NE(parse_error, nullptr);
  EXPECT_EQ(parse_error(&node, op), FAILED);
  node.set_op_type("test.domain::1::BridgeReturn");
  const auto parse_return = domi::OpRegistry::Instance()->GetParseParamFunc("BridgeReturnTarget", node.op_type());
  ASSERT_NE(parse_return, nullptr);
  EXPECT_NE(parse_return(&node, op), SUCCESS);

  Operator source_op("operator_source", "BridgeOperator");
  source_op.SetAttr("alpha", 0.5F);
  Operator operator_target("operator_target", "BridgeOperatorTarget");
  const auto parse_operator =
      domi::OpRegistry::Instance()->GetParseParamByOperatorFunc("test.domain::1::BridgeOperator");
  ASSERT_NE(parse_operator, nullptr);
  EXPECT_EQ(parse_operator(source_op, operator_target), SUCCESS);
  float copied_alpha = 0.0F;
  EXPECT_EQ(operator_target.GetAttr("copied_alpha", copied_alpha), GRAPH_SUCCESS);
  EXPECT_FLOAT_EQ(copied_alpha, 0.5F);

  const auto parse_operator_error =
      domi::OpRegistry::Instance()->GetParseParamByOperatorFunc("test.domain::1::BridgeOperatorError");
  ASSERT_NE(parse_operator_error, nullptr);
  EXPECT_EQ(parse_operator_error(source_op, operator_target), FAILED);

  const auto parse_operator_return =
      domi::OpRegistry::Instance()->GetParseParamByOperatorFunc("test.domain::1::BridgeOperatorReturn");
  ASSERT_NE(parse_operator_return, nullptr);
  EXPECT_NE(parse_operator_return(source_op, operator_target), SUCCESS);

  using InitBridgeFunc = Status (*)();
  const auto init_bridge = reinterpret_cast<InitBridgeFunc>(dlsym(RTLD_DEFAULT, "InitOnnxPluginBridge"));
  ASSERT_NE(init_bridge, nullptr);
  EXPECT_EQ(init_bridge(), SUCCESS);
  EXPECT_EQ(LoadOnnxPythonPluginBridge(), SUCCESS);
  using ResetBridgeFunc = void (*)();
  const auto reset_bridge = reinterpret_cast<ResetBridgeFunc>(dlsym(RTLD_DEFAULT, "ResetOnnxPluginBridgeState"));
  ASSERT_NE(reset_bridge, nullptr);
  reset_bridge();
  EXPECT_NE(parse_elu(&node, op), SUCCESS);
}
TEST_F(STestOnnxParser, onnx_python_plugin_bridge_parse) {
  ASSERT_EQ(setenv("PYTHONPATH", ONNX_PLUGIN_PY_INSTALL_DIR, 1), 0);
  ASSERT_EQ(GePythonRuntimeManager::Instance().EnsureReady(), SUCCESS);
  ScopedInMemoryPlugin in_memory_plugin;
  ASSERT_EQ(setenv("ASCEND_CUSTOM_OPP_PATH", "__ge_py_onnx_plugin_in_memory__", 1), 0);
  std::string case_dir = __FILE__;
  case_dir = case_dir.substr(0, case_dir.find_last_of("/"));
  std::map<ge::AscendString, ge::AscendString> parser_params;
  ge::Graph graph;
  ASSERT_EQ(ge::aclgrphParseONNX((case_dir + "/origin_models/onnx_conv2d.onnx").c_str(), parser_params, graph),
            GRAPH_SUCCESS);
  OnnxModelParser plugin_parser;
  ge::Graph plugin_result;
  ASSERT_EQ(plugin_parser.ModelParseToGraph(CreateBridgePluginModel(), plugin_result), SUCCESS);
  VerifyBridgePluginNode(plugin_result);
  VerifyBridgePluginCallbacks();

  unsetenv("ASCEND_CUSTOM_OPP_PATH");
  unsetenv("PYTHONPATH");
}
}  // namespace ge
