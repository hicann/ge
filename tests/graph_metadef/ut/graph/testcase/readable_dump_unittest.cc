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
#include <string>
#include <google/protobuf/text_format.h>

#include "graph/utils/graph_utils.h"
#include "graph/graph_buffer.h"
#include "graph/utils/graph_utils_ex.h"
#include "graph_builder_utils.h"
#include "ge_ir.pb.h"
#include "node_adapter.h"
#include "tensor_adapter.h"
#include "tensor_utils.h"
#include "debug/ge_attr_define.h"
#include "graph/debug/ge_op_types.h"
#include "graph/ge_context.h"
#include "graph/utils/file_utils.h"
#include "graph/utils/readable_dump.h"
#include "graph/operator_reg.h"
#include "es_graph_builder.h"
#include "normal_graph/operator_impl.h"
#include "default_attr_utils.h"
#include "graph_dump_utils.h"
#include "common/util/tiling_utils.h"

using namespace ge;
namespace {
ComputeGraphPtr BuildGraphWithConst(const std::string &graph_name = "graph") {
  auto ge_tensor = std::make_shared<GeTensor>();
  uint8_t data_buf[4096] = {0};
  data_buf[0] = 7;
  data_buf[10] = 8;
  ge_tensor->SetData(data_buf, 4096);

  ut::GraphBuilder builder = ut::GraphBuilder(graph_name);
  auto data_node = builder.AddNode("Data", "Data", 0, 1);
  auto const_node = builder.AddNode("Const", "Const", 0, 1);
  AttrUtils::SetTensor(const_node->GetOpDesc(), ge::ATTR_NAME_WEIGHTS, ge_tensor);
  AttrUtils::SetStr(const_node->GetOpDesc(), "fake_attr_name", "fake_attr_value");
  auto add_node = builder.AddNode("Add", "Add", 2, 1);
  AttrUtils::SetStr(add_node->GetOpDesc(), "fake_attr_name", "fake_attr_value");
  AttrUtils::SetStr(add_node->GetOpDesc(), ge::ATTR_NAME_WEIGHTS, "fake_attr_value");
  auto netoutput = builder.AddNode("Netoutput", "NetOutput", 1, 0);
  builder.AddDataEdge(data_node, 0, add_node, 0);
  builder.AddDataEdge(const_node, 0, add_node, 1);
  builder.AddDataEdge(add_node, 0, netoutput, 0);
  return builder.GetGraph();
}
} // namespace


class UtestReadableDump : public testing::Test {
 protected:
  void SetUp() {
    unsetenv("DUMP_GRAPH_PATH");
    unsetenv("DUMP_GE_GRAPH");
    unsetenv("DUMP_GRAPH_FORMAT");
    unsetenv("NPU_COLLECT_PATH");
  }

  void TearDown() {
    unsetenv("DUMP_GRAPH_PATH");
    unsetenv("DUMP_GE_GRAPH");
    unsetenv("DUMP_GRAPH_FORMAT");
  }

  static std::string GetFilePathWhenDumpPathSet(const ComputeGraphPtr graph, const string &ascend_work_path, const string &suffix = "") {
    std::string real_path_file;
    EXPECT_EQ(ge::SUCCESS, GraphUtils::GenDumpReadableTxtFileName(graph, "test", "", real_path_file));
    auto real_path = getParentDirectory(real_path_file);
    return real_path;
  }
  static std::vector<string> GetSpecificFilePath(const std::string &file_path, const string &suffix) {
    DIR *dir;
    struct dirent *ent;
    dir = opendir(file_path.c_str());
    std::vector<string> file_vec{};
    if (dir == nullptr) {
      return file_vec;
    }
    while ((ent = readdir(dir)) != nullptr) {
      if (strstr(ent->d_name, suffix.c_str()) != nullptr) {
        std::string d_name(ent->d_name);
        file_vec.emplace_back(d_name);
      }
    }
    closedir(dir);
    return file_vec;
  }
  static std::string getParentDirectory(const std::string& filepath) {
    size_t lastSlash = filepath.find_last_of("/\\");
    if (lastSlash != std::string::npos) {
      return filepath.substr(0, lastSlash);
    }
    return "";
  }

};

TEST_F(UtestReadableDump, test_GenReadableDumpTextFile) {
  auto graph = BuildGraphWithConst("GenReadableDumpTextFile");
  std::string ascend_work_path = "./test_ge_readable_dump";
  mmSetEnv("DUMP_GE_GRAPH", "1", 1);
  mmSetEnv("DUMP_GRAPH_PATH", ascend_work_path.c_str(), 1);
  GraphUtils::DumpGEGraphToReadable(graph, "test", true, "");
  std::string dump_file_path = GetFilePathWhenDumpPathSet(graph, ascend_work_path, "test");
  auto dump_graph_files = GetSpecificFilePath(dump_file_path, "test");

  EXPECT_TRUE(!dump_graph_files.empty());
  system(("rm -rf " + dump_file_path).c_str());
}

TEST_F(UtestReadableDump, test_GenDumpReadableTxtFileName) {
  auto graph = BuildGraphWithConst("GenDumpReadableTxtFileName");
  std::string real_path_name;
  EXPECT_EQ(ge::SUCCESS, GraphUtils::GenDumpReadableTxtFileName(graph, "test_dump_filename", "", real_path_name));
  EXPECT_TRUE(real_path_name.find("ge_readable_") != real_path_name.npos);
  EXPECT_TRUE(real_path_name.find("_graph_0_test_dump_filename.txt") != real_path_name.npos);

  graph->SetAttr(ATTR_SINGLE_OP_SCENE, GeAttrValue::CreateFrom<>(true));
  EXPECT_EQ(ge::SUCCESS, GraphUtils::GenDumpReadableTxtFileName(graph, "test_dump_filename", "", real_path_name));
  EXPECT_TRUE(real_path_name.find("_aclop_graph_0_test_dump_filename.txt") != real_path_name.npos);
}

TEST_F(UtestReadableDump, test_WriteReadableDumpToOStream) {
  std::stringstream readable_ss("test write file to os");
  std::ostringstream readable_os;
  GraphUtils::WriteReadableDumpToOStream(readable_ss, readable_os);
  EXPECT_EQ(readable_os.str(), readable_ss.str());
}

TEST_F(UtestReadableDump, test_DumpToFile) {
  auto compute_graph = BuildGraphWithConst("DumpToFile");
  auto graph = GraphUtilsEx::CreateGraphFromComputeGraph(compute_graph);
  std::string ascend_work_path = "./test_ge_readable_dump";
  mmSetEnv("DUMP_GE_GRAPH", "1", 1);
  mmSetEnv("DUMP_GRAPH_PATH", ascend_work_path.c_str(), 1);
  EXPECT_EQ(ge::SUCCESS, graph.DumpToFile(Graph::DumpFormat::kReadable, "test_dump_to_file"));
  std::string dump_file_path = GetFilePathWhenDumpPathSet(compute_graph, ascend_work_path, "test_dump_to_file");
  auto dump_graph_files = GetSpecificFilePath(dump_file_path, "test_dump_to_file");

  EXPECT_EQ(1, dump_graph_files.size());
  system(("rm -rf " + dump_file_path).c_str());
}

TEST_F(UtestReadableDump, test_DumpEnvDefault) {
  auto compute_graph = BuildGraphWithConst("DumpEnvDefault");
  auto graph = GraphUtilsEx::CreateGraphFromComputeGraph(compute_graph);
  std::string ascend_work_path = "./test_ge_readable_dump";
  mmSetEnv("DUMP_GE_GRAPH", "1", 1);
  mmSetEnv("DUMP_GRAPH_PATH", ascend_work_path.c_str(), 1);
  DumpGraph(compute_graph, "PreRunBegin");
  std::string dump_file_path = GetFilePathWhenDumpPathSet(compute_graph, ascend_work_path, "PreRunBegin");
  auto dump_graph_files = GetSpecificFilePath(dump_file_path, "PreRunBegin");

  EXPECT_EQ(2, dump_graph_files.size());
  system(("rm -rf " + dump_file_path).c_str());
}

TEST_F(UtestReadableDump, test_DumpEnvSingle) {
  auto compute_graph = BuildGraphWithConst("DumpEnvSingle");
  auto graph = GraphUtilsEx::CreateGraphFromComputeGraph(compute_graph);
  std::string ascend_work_path = "./test_ge_readable_dump";
  mmSetEnv("DUMP_GE_GRAPH", "1", 1);
  mmSetEnv("DUMP_GRAPH_PATH", ascend_work_path.c_str(), 1);
  mmSetEnv("DUMP_GRAPH_FORMAT", "ReAdAbLe", 1);
  DumpGraph(compute_graph, "PreRunBegin");
  std::string dump_file_path = GetFilePathWhenDumpPathSet(compute_graph, ascend_work_path, "PreRunBegin");
  auto dump_graph_files = GetSpecificFilePath(dump_file_path, "PreRunBegin");

  EXPECT_EQ(1, dump_graph_files.size());
  system(("rm -rf " + dump_file_path).c_str());
}

TEST_F(UtestReadableDump, test_DumpEnvMultiple) {
  auto compute_graph = BuildGraphWithConst("DumpEnvMultiple");
  auto graph = GraphUtilsEx::CreateGraphFromComputeGraph(compute_graph);
  std::string ascend_work_path = "./test_ge_readable_dump";
  mmSetEnv("DUMP_GE_GRAPH", "1", 1);
  mmSetEnv("DUMP_GRAPH_PATH", ascend_work_path.c_str(), 1);
  mmSetEnv("DUMP_GRAPH_FORMAT", "onNx    |gE_PrOtO|ReAdAbLe", 1);
  DumpGraph(compute_graph, "PreRunBegin");
  std::string dump_file_path = GetFilePathWhenDumpPathSet(compute_graph, ascend_work_path, "PreRunBegin");
  auto dump_graph_files = GetSpecificFilePath(dump_file_path, "PreRunBegin");

  EXPECT_EQ(3, dump_graph_files.size());
  system(("rm -rf " + dump_file_path).c_str());
}

TEST_F(UtestReadableDump, test_DumpEnvUnknown) {
  auto compute_graph = BuildGraphWithConst("DumpEnvUnknown");
  auto graph = GraphUtilsEx::CreateGraphFromComputeGraph(compute_graph);
  std::string ascend_work_path = "./test_ge_readable_dump";
  mmSetEnv("DUMP_GE_GRAPH", "1", 1);
  mmSetEnv("DUMP_GRAPH_PATH", ascend_work_path.c_str(), 1);
  mmSetEnv("DUMP_GRAPH_FORMAT", "txt", 1);
  DumpGraph(compute_graph, "PreRunBegin");
  std::string dump_file_path = GetFilePathWhenDumpPathSet(compute_graph, ascend_work_path, "PreRunBegin");
  auto dump_graph_files = GetSpecificFilePath(dump_file_path, "PreRunBegin");

  EXPECT_EQ(0, dump_graph_files.size());
  system(("rm -rf " + dump_file_path).c_str());
}

TEST_F(UtestReadableDump, test_GraphDumpToTile) {
  std::string ascend_work_path = "./test_ge_readable_dump";
  mmSetEnv("DUMP_GE_GRAPH", "1", 1);
  mmSetEnv("DUMP_GRAPH_PATH", ascend_work_path.c_str(), 1);
  auto compute_graph = BuildGraphWithConst("GraphDumpToTile");
  auto graph = GraphUtilsEx::CreateGraphFromComputeGraph(compute_graph);
  EXPECT_EQ(ge::SUCCESS, graph.DumpToFile(ge::Graph::DumpFormat::kReadable, ""));
  EXPECT_EQ(ge::SUCCESS, graph.DumpToFile(ge::Graph::DumpFormat::kOnnx, ""));
  EXPECT_EQ(ge::SUCCESS, graph.DumpToFile(ge::Graph::DumpFormat::kTxt, ""));
  std::string dump_file_path = GetFilePathWhenDumpPathSet(compute_graph, ascend_work_path, "PreRunBegin");
  auto dump_graph_files = GetSpecificFilePath(dump_file_path, "PreRunBegin");
  system(("rm -rf " + dump_file_path).c_str());
}

TEST_F(UtestReadableDump, test_GraphDumpToOStream) {
  auto compute_graph = BuildGraphWithConst("GraphDumpToOStream");
  auto graph = GraphUtilsEx::CreateGraphFromComputeGraph(compute_graph);
  std::ostringstream readable_os;
  EXPECT_EQ(ge::SUCCESS, graph.Dump(ge::Graph::DumpFormat::kReadable, readable_os));
  EXPECT_EQ(ge::SUCCESS, graph.Dump(ge::Graph::DumpFormat::kOnnx, readable_os));
  EXPECT_EQ(ge::SUCCESS, graph.Dump(ge::Graph::DumpFormat::kTxt, readable_os));
}

REG_OP(Data)
    .INPUT(x, TensorType::ALL())
    .OUTPUT(y, TensorType::ALL())
    .ATTR(index, Int, 0)
    .OP_END_FACTORY_REG(Data)
REG_OP(Const)
    .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT8, DT_INT16, DT_UINT16, DT_UINT8, DT_INT32, DT_INT64, DT_UINT32,
                           DT_UINT64, DT_BOOL, DT_DOUBLE}))
    .ATTR(value, Tensor, Tensor())
    .OP_END_FACTORY_REG(Const);
REG_OP(phony_mul_i)
    .INPUT(x, TensorType::ALL())
    .OPTIONAL_INPUT(opt_x, TensorType::ALL())
    .DYNAMIC_INPUT(dx1, TensorType::All())
    .OUTPUT(y, TensorType::NumberType())
    .OP_END_FACTORY_REG(phony_mul_i);
REG_OP(Phony1)
    .INPUT(x, TensorType::ALL())
    .OUTPUT(y, TensorType::NumberType())
    .REQUIRED_ATTR(N, Int)
    .OP_END_FACTORY_REG(Phony1);
REG_OP(phony_multi_attr)
    .OPTIONAL_INPUT(x, TensorType::ALL())
    .ATTR(li, ListInt, {10, 10, 10})
    .ATTR(f, Float, 0.0)
    .ATTR(s, String, "s")
    .ATTR(b, Bool, true)
    .ATTR(lf, ListFloat, {0.1, 0.2})
    .ATTR(lb, ListBool, {false, true})
    .ATTR(opt_data_type, Type, DT_INT64)
    .ATTR(opt_list_data_type, ListType, {DT_FLOAT, DT_DOUBLE})
    .ATTR(opt_list_list_int, ListListInt, {{1,2,3}, {3,2,1}})
    .ATTR(opt_tensor, Tensor, Tensor())
    .ATTR(opt_list_string, ListString, {"test"})
    .OUTPUT(y, TensorType::NumberType())
    .DYNAMIC_OUTPUT(dy, TensorType::All())
    .OP_END_FACTORY_REG(phony_multi_attr);
REG_OP(phony_mix_ios)
    .INPUT(x1, TensorType::All())
    .INPUT(x2, TensorType::All())
    .DYNAMIC_INPUT(dx1, TensorType::All())
    .OUTPUT(y1, TensorType::All())
    .OUTPUT(y2, TensorType::All())
    .DYNAMIC_OUTPUT(dy, TensorType::All())
    .OP_END_FACTORY_REG(phony_mix_ios);

TEST_F(UtestReadableDump, test_GenSimple) {
  auto compute_graph = BuildGraphWithConst("GenSimple");
  auto graph = ge::GraphUtilsEx::CreateGraphFromComputeGraph(compute_graph);
  auto node1 = graph.GetDirectNode().at(0);
  auto node2 = graph.GetDirectNode().at(1);
  GNode node_netOutput;
  for (const auto &node: graph.GetAllNodes()) {
    AscendString node_type;
    node.GetType(node_type);
    if (node_type == "NetOutput") {
      node_netOutput = node;
    }
  }

  graph.AddControlEdge(node1, node2);
  graph.AddControlEdge(node2, node_netOutput);
  std::string readable_dump = R"(graph("GenSimple"):
  %Data : [#users=1] = Node[type=Data] ()
  %Const : [#users=1] = Node[type=Const] ()
  %Add : [#users=1] = Node[type=Add] (inputs = (%Data, %Const))

  return (%Add)
)";
  std::stringstream readable_ss;
  ReadableDump::GenReadableDump(readable_ss, compute_graph);
  EXPECT_EQ(readable_dump, readable_ss.str());
}

TEST_F(UtestReadableDump, test_OptionalInputsAndControlEdges) {
  auto ge_tensor = std::make_shared<GeTensor>();
  uint8_t data_buf[4096] = {0};
  data_buf[0] = 7;
  data_buf[10] = 8;
  ge_tensor->SetData(data_buf, 4096);

  ut::GraphBuilder builder = ut::GraphBuilder("graph_OptionalInputsAndControlEdges");
  const auto &netoutput = builder.AddNode("netoutput", NETOUTPUT, 0, 0);
  auto compute_graph = builder.GetGraph();
  auto graph = ge::GraphUtilsEx::CreateGraphFromComputeGraph(compute_graph);

  auto const_op = op::Const("const").set_attr_value(ge::TensorAdapter::AsTensor(*ge_tensor));
  auto const_node = graph.AddNodeByOp(const_op);

  auto data1_op = op::Data("data1");
  auto data1_node = graph.AddNodeByOp(data1_op);
  graph.AddDataEdge(const_node, 0, data1_node, 0);

  auto data2_op = op::Data("data2");
  auto data2_node = graph.AddNodeByOp(data2_op);
  graph.AddDataEdge(const_node, 0, data2_node, 0);

  auto phony_mul_i_op = op::phony_mul_i("test").create_dynamic_input_byindex_dx1(2, 1);
  auto phony_mul_i_node = graph.AddNodeByOp(phony_mul_i_op);
  graph.AddDataEdge(const_node, 0, phony_mul_i_node, 0);
  graph.AddDataEdge(data1_node, 0, phony_mul_i_node, 2);
  graph.AddDataEdge(data2_node, 0, phony_mul_i_node, 3);

  GNode netoutput_gnode = NodeAdapter::Node2GNode(netoutput);
  graph.AddControlEdge(const_node, netoutput_gnode);

  std::string readable_dump = R"(graph("graph_OptionalInputsAndControlEdges"):
  %const : [#users=1] = Node[type=Const] (attrs = {value: [0.000000]})
  %data1 : [#users=1] = Node[type=Data] (inputs = (%const), attrs = {index: 0})
  %data2 : [#users=1] = Node[type=Data] (inputs = (%const), attrs = {index: 0})
  %test : [#users=1] = Node[type=phony_mul_i] (inputs = (%const, %data1, %data2))
)";

  std::stringstream readable_ss;
  ReadableDump::GenReadableDump(readable_ss, compute_graph);
  EXPECT_EQ(readable_dump, readable_ss.str());
}

TEST_F(UtestReadableDump, test_GenComplex) {
  auto ge_tensor = std::make_shared<GeTensor>();
  uint8_t data_buf[4096] = {0};
  data_buf[0] = 7;
  data_buf[10] = 8;
  ge_tensor->SetData(data_buf, 4096);

  ut::GraphBuilder builder = ut::GraphBuilder("graph_complex");
  const auto &netoutput = builder.AddNode("netoutput", NETOUTPUT, 5, 0);
  auto compute_graph = builder.GetGraph();
  auto graph = ge::GraphUtilsEx::CreateGraphFromComputeGraph(compute_graph);
  auto data_op = op::Data("data");
  auto data_node = graph.AddNodeByOp(data_op);
  auto const_op = op::Const("const").set_attr_value(ge::TensorAdapter::AsTensor(*ge_tensor));
  auto const_node = graph.AddNodeByOp(const_op);

  auto phony1_op = op::Phony1("phony1").set_attr_N(1);
  auto phony1_node = graph.AddNodeByOp(phony1_op);
  graph.AddDataEdge(data_node, 0, phony1_node, 0);

  auto phony_multi_attr_op1 = op::phony_multi_attr("phony_multi_attr_node1");
  auto phony_multi_attr_node1 = graph.AddNodeByOp(phony_multi_attr_op1);
  auto phony_multi_attr_op2 = op::phony_multi_attr("phony_multi_attr_node2");
  phony_multi_attr_op2.create_dynamic_output_dy(2);
  auto phony_multi_attr_node2 = graph.AddNodeByOp(phony_multi_attr_op2);
  graph.AddDataEdge(data_node, 0, phony_multi_attr_node2, 0);

  auto phony_mix_ios_op = op::phony_mix_ios("phony_mix_ios")
                              .create_dynamic_input_byindex_dx1(2, 2);
  phony_mix_ios_op.create_dynamic_output_dy(2);
  auto phony_mix_ios_node = graph.AddNodeByOp(phony_mix_ios_op);
  graph.AddDataEdge(data_node, 0, phony_mix_ios_node, 0);
  graph.AddDataEdge(phony1_node, 0, phony_mix_ios_node, 1);
  graph.AddDataEdge(phony_multi_attr_node2, 0, phony_mix_ios_node, 2);
  graph.AddDataEdge(phony_multi_attr_node2, 1, phony_mix_ios_node, 3);

  GNode netoutput_gnode = NodeAdapter::Node2GNode(netoutput);

  graph.AddDataEdge(phony1_node, 0, netoutput_gnode, 0);
  graph.AddDataEdge(phony_multi_attr_node2, 0, netoutput_gnode, 1);
  graph.AddDataEdge(phony_multi_attr_node2, 1, netoutput_gnode, 2);
  graph.AddDataEdge(phony_mix_ios_node, 0, netoutput_gnode, 3);
  graph.AddDataEdge(phony_mix_ios_node, 1, netoutput_gnode, 4);

  std::vector<Operator> outputs = {phony1_op, phony_multi_attr_op2, phony_mix_ios_op};
  graph.SetOutputs(outputs);

  std::string readable_dump = R"(graph("graph_complex"):
  %data : [#users=1] = Node[type=Data] (attrs = {index: 0})
  %const : [#users=1] = Node[type=Const] (attrs = {value: [0.000000]})
  %phony1 : [#users=1] = Node[type=Phony1] (inputs = (%data), attrs = {N: 1})
  %phony_multi_attr_node1 : [#users=1] = Node[type=phony_multi_attr] (attrs = {li: {10, 10, 10}, f: 0.000000, s: "s", b: true, lf: {0.100000, 0.200000}, lb: {false, true}, opt_data_type: DT_INT64, opt_list_data_type: {DT_FLOAT, DT_DOUBLE}, opt_list_list_int: {{1, 2, 3}, {3, 2, 1}}, opt_tensor: <empty>, opt_list_string: {"test"}})
  %phony_multi_attr_node2 : [#users=3] = Node[type=phony_multi_attr] (inputs = (%data), attrs = {li: {10, 10, 10}, f: 0.000000, s: "s", b: true, lf: {0.100000, 0.200000}, lb: {false, true}, opt_data_type: DT_INT64, opt_list_data_type: {DT_FLOAT, DT_DOUBLE}, opt_list_list_int: {{1, 2, 3}, {3, 2, 1}}, opt_tensor: <empty>, opt_list_string: {"test"}})
  %ret : [#users=2] = get_element[node=%phony_multi_attr_node2](0)
  %ret_1 : [#users=2] = get_element[node=%phony_multi_attr_node2](1)
  %ret_2 : [#users=0] = get_element[node=%phony_multi_attr_node2](2)
  %phony_mix_ios : [#users=4] = Node[type=phony_mix_ios] (inputs = (%data, %phony1, %ret, %ret_1))
  %ret_3 : [#users=1] = get_element[node=%phony_mix_ios](0)
  %ret_4 : [#users=1] = get_element[node=%phony_mix_ios](1)
  %ret_5 : [#users=0] = get_element[node=%phony_mix_ios](2)
  %ret_6 : [#users=0] = get_element[node=%phony_mix_ios](3)

  return (%phony1, %ret, %ret_1, %ret_3, %ret_4)
)";

  std::stringstream readable_ss;
  ReadableDump::GenReadableDump(readable_ss, compute_graph);
  EXPECT_EQ(readable_dump, readable_ss.str());
}

REG_OP(phony_MemSet)
    .REQUIRED_ATTR(sizes, ListInt)
    .ATTR(dtypes, ListType, {})
    .ATTR(values_int, ListInt, {})
    .ATTR(values_float, ListFloat, {})
    .OP_END_FACTORY_REG(phony_MemSet);

TEST_F(UtestReadableDump, test_EmptyListAttr) {
  ut::GraphBuilder builder = ut::GraphBuilder("graph_EmptyListAttr");
  auto compute_graph = builder.GetGraph();
  auto graph = ge::GraphUtilsEx::CreateGraphFromComputeGraph(compute_graph);

  auto data_op = op::Data("data");
  auto data_node = graph.AddNodeByOp(data_op);

  std::vector<ge::DataType> dtype_vec = {DT_FLOAT};
  auto memset_op = op::phony_MemSet("test").set_attr_sizes({2560}).set_attr_dtypes(dtype_vec);
  auto memset_node = graph.AddNodeByOp(memset_op);

  graph.AddControlEdge(data_node, memset_node);

  std::string readable_dump = R"(graph("graph_EmptyListAttr"):
  %data : [#users=1] = Node[type=Data] (attrs = {index: 0})
  %test : [#users=0] = Node[type=phony_MemSet] (attrs = {sizes: {2560}, dtypes: {DT_FLOAT}})
)";
  std::stringstream readable_ss;
  EXPECT_EQ(SUCCESS, ReadableDump::GenReadableDump(readable_ss, compute_graph));
  EXPECT_EQ(readable_dump, readable_ss.str());
}

REG_OP(phony_attr_tensors)
    .ATTR(t1, Tensor, Tensor())
    .ATTR(t2, Tensor, Tensor())
    .ATTR(t3, Tensor, Tensor())
    .ATTR(t4, Tensor, Tensor())
    .ATTR(t5, Tensor, Tensor())
    .ATTR(t6, Tensor, Tensor())
    .ATTR(t7, Tensor, Tensor())
    .ATTR(t8, Tensor, Tensor())
    .ATTR(t9, Tensor, Tensor())
    .ATTR(t10, Tensor, Tensor())
    .ATTR(t11, Tensor, Tensor())
    .OP_END_FACTORY_REG(phony_attr_tensors);

TEST_F(UtestReadableDump, test_AttrTensorValues) {
  ut::GraphBuilder builder = ut::GraphBuilder("graph_EmptyListAttr");
  const auto &netoutput = builder.AddNode("netoutput", NETOUTPUT, 0, 0);
  auto compute_graph = builder.GetGraph();
  auto graph = ge::GraphUtilsEx::CreateGraphFromComputeGraph(compute_graph);

  es::EsGraphBuilder es_builder("test_attr_tensor");
  std::vector<float> data1 = {-5.5, -4.4, -3.3, -2.2, -1.1, 0, 1.1, 2.2, 3.3, 4.4, 5.5};
  std::vector<int64_t> dims1 = {11};
  auto tensor1 = es_builder.CreateTensor<float>(data1, dims1, ge::DT_FLOAT);
  std::vector<int8_t> data2 = {-5, -4, -3, -2, -1, 0};
  std::vector<int64_t> dims2 = {6};
  auto tensor2 = es_builder.CreateTensor<int8_t>(data2, dims2, ge::DT_INT8);
  std::vector<int16_t> data3 = {-5, -4, -3, -2, -1};
  std::vector<int64_t> dims3 = {5};
  auto tensor3 = es_builder.CreateTensor<int16_t>(data3, dims3, ge::DT_INT16);
  std::vector<int32_t> data4 = {-5, -4, -3, -2};
  std::vector<int64_t> dims4 = {4};
  auto tensor4 = es_builder.CreateTensor<int32_t>(data4, dims4, ge::DT_INT32);
  std::vector<int64_t> data5 = {-5, -4, -3};
  std::vector<int64_t> dims5 = {3};
  auto tensor5 = es_builder.CreateTensor<int64_t>(data5, dims5, ge::DT_INT64);
  std::vector<uint8_t> data6 = {5, 4};
  std::vector<int64_t> dims6 = {2};
  auto tensor6 = es_builder.CreateTensor<uint8_t>(data6, dims6, ge::DT_UINT8);
  std::vector<uint16_t> data7 = {5};
  std::vector<int64_t> dims7 = {1};
  auto tensor7 = es_builder.CreateTensor<uint16_t>(data7, dims7, ge::DT_UINT16);
  std::vector<uint32_t> data8 = {0};
  std::vector<int64_t> dims8 = {};
  auto tensor8 = es_builder.CreateTensor<uint32_t>(data8, dims8, ge::DT_UINT32);
  std::vector<uint64_t> data9 = {5, 6};
  std::vector<int64_t> dims9 = {2};
  auto tensor9 = es_builder.CreateTensor<uint64_t>(data9, dims9, ge::DT_UINT64);
  std::vector data10 = {true, false, true, false};
  std::vector<int64_t> dims10 = {4};
  auto tensor10 = es_builder.CreateBoolTensor(data10, dims10);
  // Create FP16 tensor
  std::vector<float> fp16_data_float = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
  std::vector<uint16_t> fp16_data;
  fp16_data.reserve(fp16_data_float.size());
  for (const auto &val : fp16_data_float) {
    fp16_data.push_back(optiling::Float32ToFloat16(val));
  }
  std::vector<int64_t> dims11 = {5};
  auto tensor11 = es_builder.CreateTensor<uint16_t>(fp16_data, dims11, ge::DT_FLOAT16);

  auto op = op::phony_attr_tensors("test_attr_tensor")
                .set_attr_t1(*tensor1)
                .set_attr_t2(*tensor2)
                .set_attr_t3(*tensor3)
                .set_attr_t4(*tensor4)
                .set_attr_t5(*tensor5)
                .set_attr_t6(*tensor6)
                .set_attr_t7(*tensor7)
                .set_attr_t8(*tensor8)
                .set_attr_t9(*tensor9)
                .set_attr_t10(*tensor10)
                .set_attr_t11(*tensor11);
  (void) graph.AddNodeByOp(op);
  std::string readable_dump = R"(graph("graph_EmptyListAttr"):
  %test_attr_tensor : [#users=0] = Node[type=phony_attr_tensors] (attrs = {t1: [-5.500000 -4.400000 -3.300000 ... 3.300000 4.400000 5.500000], t2: [-5 -4 -3 -2 -1 0], t3: [-5 -4 -3 -2 -1], t4: [-5 -4 -3 -2], t5: [-5 -4 -3], t6: [5 4], t7: [5], t8: [0], t9: [5 6], t10: [true false true false], t11: [1.000000 2.000000 3.000000 4.000000 5.000000]})
)";
  std::stringstream readable_ss;
  EXPECT_EQ(SUCCESS, ReadableDump::GenReadableDump(readable_ss, compute_graph));
  EXPECT_EQ(readable_dump, readable_ss.str());
}

REG_OP(phony_VariableV2)
    .INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT8, DT_INT16, DT_UINT16, \
        DT_UINT8, DT_INT32, DT_INT64, DT_UINT32, DT_UINT64, DT_BOOL, DT_DOUBLE}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT8, DT_INT16, DT_UINT16, \
        DT_UINT8, DT_INT32, DT_INT64, DT_UINT32, DT_UINT64, DT_BOOL, DT_DOUBLE}))
    .ATTR(index, Int, 0)
    .ATTR(value, Tensor, Tensor())
    .ATTR(container, String, "")
    .ATTR(shared_name, String, "")
    .OP_END_FACTORY_REG(phony_VariableV2)

TEST_F(UtestReadableDump, test_EmptyString) {
  ut::GraphBuilder builder = ut::GraphBuilder("graph_EmptyListAttr");
  const auto &netoutput = builder.AddNode("netoutput", NETOUTPUT, 0, 0);
  auto compute_graph = builder.GetGraph();
  auto graph = ge::GraphUtilsEx::CreateGraphFromComputeGraph(compute_graph);

  auto op = op::phony_VariableV2("test_empty_string");
  (void) graph.AddNodeByOp(op);

  std::string readable_dump = R"(graph("graph_EmptyListAttr"):
  %test_empty_string : [#users=1] = Node[type=phony_VariableV2] (attrs = {index: 0, value: <empty>})
)";
  std::stringstream readable_ss;
  EXPECT_EQ(SUCCESS, ReadableDump::GenReadableDump(readable_ss, compute_graph));
  EXPECT_EQ(readable_dump, readable_ss.str());
}

REG_OP(phony_Squeeze)
    .INPUT(x, TensorType::ALL())
    .OUTPUT(y, TensorType::ALL())
    .ATTR(axis, ListInt, {})
    .OP_END_FACTORY_REG(phony_Squeeze)

TEST_F(UtestReadableDump, test_phony_Squeeze_WithInputNoAttr) {
  ut::GraphBuilder builder = ut::GraphBuilder("graph_WithInputNoAttr");
  auto compute_graph = builder.GetGraph();
  auto graph = ge::GraphUtilsEx::CreateGraphFromComputeGraph(compute_graph);

  auto data_op = op::Data("data");
  auto data_node = graph.AddNodeByOp(data_op);

  auto squeeze_op = op::phony_Squeeze("phony_Squeeze");
  auto squeeze_node = graph.AddNodeByOp(squeeze_op);

  graph.AddDataEdge(data_node, 0, squeeze_node, 0);

  std::string readable_dump = R"(graph("graph_WithInputNoAttr"):
  %data : [#users=1] = Node[type=Data] (attrs = {index: 0})
  %phony_Squeeze : [#users=1] = Node[type=phony_Squeeze] (inputs = (%data))
)";
  std::stringstream readable_ss;
  EXPECT_EQ(SUCCESS, ReadableDump::GenReadableDump(readable_ss, compute_graph));
  EXPECT_EQ(readable_dump, readable_ss.str());
}

REG_OP(phony_UnsupportedAttr)
.ATTR(unsupported_attr, ListAscendString, {})
.OP_END_FACTORY_REG(phony_UnsupportedAttr)

TEST_F(UtestReadableDump, test_UnsupportedAttr) {
  ut::GraphBuilder builder = ut::GraphBuilder("graph_UnsupportedAttr");
  const auto &netoutput = builder.AddNode("netoutput", NETOUTPUT, 0, 0);
  auto compute_graph = builder.GetGraph();
  auto graph = ge::GraphUtilsEx::CreateGraphFromComputeGraph(compute_graph);

  auto op = op::phony_UnsupportedAttr("test_unsupported_attr");
  (void) graph.AddNodeByOp(op);
  std::string readable_dump = R"(graph("graph_UnsupportedAttr"):
  %test_unsupported_attr : [#users=0] = Node[type=phony_UnsupportedAttr] ()
)";
  std::stringstream readable_ss;
  EXPECT_EQ(SUCCESS, ReadableDump::GenReadableDump(readable_ss, compute_graph));
  EXPECT_EQ(readable_dump, readable_ss.str());
}

REG_OP(phony_attr_empty)
    .OPTIONAL_INPUT(x, TensorType::ALL())
    .ATTR(li, ListInt, {})
    .ATTR(f, Float, 0.0)
    .ATTR(s, String, "")
    .ATTR(b, Bool, true)
    .ATTR(lf, ListFloat, {})
    .ATTR(lb, ListBool, {})
    .ATTR(opt_data_type, Type, DT_INT64)
    .ATTR(opt_list_data_type, ListType, {DT_FLOAT, DT_DOUBLE})
    .ATTR(opt_list_list_int, ListListInt, {{}, {2, 3}, {}})
    .ATTR(opt_tensor, Tensor, Tensor())
    .ATTR(opt_list_string, ListString, {""})
    .OUTPUT(y, TensorType::NumberType())
    .DYNAMIC_OUTPUT(dy, TensorType::All())
    .OP_END_FACTORY_REG(phony_attr_empty);

TEST_F(UtestReadableDump, test_EmtpyDefaultValues) {
  ut::GraphBuilder builder = ut::GraphBuilder("graph_EmtpyDefaultValues");
  const auto &netoutput = builder.AddNode("netoutput", NETOUTPUT, 0, 0);
  auto compute_graph = builder.GetGraph();
  auto graph = ge::GraphUtilsEx::CreateGraphFromComputeGraph(compute_graph);

  auto op = op::phony_attr_empty("test_empty_attr");
  (void) graph.AddNodeByOp(op);
  auto op_desc = OperatorImpl::GetOpDesc(op);
  auto attr_list_int = ge::AttrString::GetDefaultValueString(op_desc, "li", "VT_LIST_INT");
  auto attr_list_list_int = ge::AttrString::GetDefaultValueString(op_desc, "opt_list_list_int", "VT_LIST_LIST_INT");
  EXPECT_EQ("{}", attr_list_int);
  EXPECT_EQ("{{}, {2, 3}, {}}", attr_list_list_int);

  std::string readable_dump = R"(graph("graph_EmtpyDefaultValues"):
  %test_empty_attr : [#users=1] = Node[type=phony_attr_empty] (attrs = {f: 0.000000, b: true, opt_data_type: DT_INT64, opt_list_data_type: {DT_FLOAT, DT_DOUBLE}, opt_list_list_int: {{2, 3}}, opt_tensor: <empty>})
)";
  std::stringstream readable_ss;
  EXPECT_EQ(SUCCESS, ReadableDump::GenReadableDump(readable_ss, compute_graph));
  EXPECT_EQ(readable_dump, readable_ss.str());
}