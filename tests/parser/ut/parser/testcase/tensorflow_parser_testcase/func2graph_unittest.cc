/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include "func_to_graph/func2graph.h"
#include "tensorflow/graph_to_function_def.h"
#include "graph/utils/attr_utils.h"
#include "graph/debug/ge_attr_define.h"
#include <vector>
#include <sstream>

using namespace domi::tensorflow;

namespace ge {
class UtestFuncToGraph : public ::testing::Test {
 protected:
  void SetUp() override {}

  void TearDown() override {}
};

TEST_F(UtestFuncToGraph, GeGraphDefSetName_invalid_value_failed) {
  GeGraphDefHandle geGraphDef = nullptr;
  GeGraphDefSetName(geGraphDef, nullptr);

  geGraphDef = GeGraphDefCreate();
  GeGraphDefSetName(geGraphDef, nullptr);

  GeGraphDefDestroy(&geGraphDef);
}

TEST_F(UtestFuncToGraph, GeGraphDefSetName_success) {
  GeGraphDefHandle geGraphDef = GeGraphDefCreate();
  const std::string graphName = "scale_func";
  GeGraphDefSetName(geGraphDef, graphName.c_str());

  auto *graph = static_cast<GeGraphDef *>(geGraphDef);
  EXPECT_EQ(graph->name(), graphName);

  GeGraphDefDestroy(&geGraphDef);
}

TEST_F(UtestFuncToGraph, GeGraphDefSetGraph_invalid_value_failed) {
  GeGraphDefHandle geGraphDef = nullptr;
  // handle is nullptr
  GeGraphDefSetGraph(geGraphDef, nullptr, 0);

  geGraphDef = GeGraphDefCreate();
  GeGraphDefSetGraph(geGraphDef, nullptr, 0);

  GeGraphDefDestroy(&geGraphDef);
}

TEST_F(UtestFuncToGraph, GeGraphDefSetGraph_success) {
  GeGraphDefHandle geGraphDef = nullptr;
  EXPECT_EQ(GeGraphDefToString(geGraphDef), nullptr);

  geGraphDef = GeGraphDefCreate();

  // create graph
  GraphDef graphDef;
  VersionDef *versionDef = new VersionDef();
  versionDef->set_producer(134);
  graphDef.set_allocated_versions((versionDef));

  int size = static_cast<int>(graphDef.ByteSizeLong());
  std::vector<uint8_t> buffer(size);
  EXPECT_TRUE(graphDef.SerializeToArray(buffer.data(), size));
  GeGraphDefSetGraph(geGraphDef, buffer.data(), size);

  const auto *graph = static_cast<GeGraphDef *>(geGraphDef);
  EXPECT_EQ(graph->graph().versions().producer(), 134);

  auto *debugString = GeGraphDefToString(geGraphDef);

  std::stringstream ss;
  ss << "graph {\n";
  ss << "  versions {\n";
  ss << "    producer: 134\n";
  ss << "  }\n";
  ss << "}\n";

  EXPECT_EQ(debugString, ss.str());

  GeGraphDefDestroy(&geGraphDef);
}

TEST_F(UtestFuncToGraph, GraphDefLibGetGraphDef_invalid_value_failed) {
  GraphDefLibHandle graphDefLib = nullptr;
  GeGraphDefHandle geGraphDef = nullptr;
  GraphDefLibAddGraphDef(graphDefLib, geGraphDef);

  constexpr int index = 0;
  EXPECT_EQ(GraphDefLibGetGraphDef(graphDefLib, index), nullptr);
}

TEST_F(UtestFuncToGraph, GraphDefLibGetGraphDef_success) {
  GraphDefLibHandle graphDefLib = nullptr;
  EXPECT_EQ(GraphDefLibGetPbtxt(graphDefLib), nullptr);

  graphDefLib = GraphDefLibCreate();
  GeGraphDefHandle geGraphDef = GeGraphDefCreate();
  GeGraphDefSetName(geGraphDef, "scale_func");
  GraphDefLibAddGraphDef(graphDefLib, geGraphDef);

  auto *pbtxt = GraphDefLibGetPbtxt(graphDefLib);
  std::stringstream ss;
  ss << "graph_def {\n";
  ss << "  name: \"scale_func\"\n";
  ss << "}\n";
  EXPECT_EQ(pbtxt, ss.str());

  constexpr int index = 0;
  EXPECT_NE(GraphDefLibGetGraphDef(graphDefLib, index), nullptr);

  GraphDefLibDestroy(&graphDefLib);
}

TEST_F(UtestFuncToGraph, GraphToFunctionDef_FindAttrValue_null_node) {
  domi::tensorflow::AttrValue attr_value;
  bool ret = GraphToFunctionDef::FindAttrValue(nullptr, "test_attr", attr_value);
  EXPECT_FALSE(ret);
}

TEST_F(UtestFuncToGraph, GraphToFunctionDef_FindAttrValue_success) {
  domi::tensorflow::NodeDef node_def;
  node_def.set_name("test_node");
  domi::tensorflow::AttrValue attr_value;
  attr_value.set_i(42);
  (*node_def.mutable_attr())["test_attr"] = attr_value;

  domi::tensorflow::AttrValue result;
  bool ret = GraphToFunctionDef::FindAttrValue(&node_def, "test_attr", result);
  EXPECT_TRUE(ret);
  EXPECT_EQ(result.i(), 42);

  ret = GraphToFunctionDef::FindAttrValue(&node_def, "nonexistent", result);
  EXPECT_FALSE(ret);
}

TEST_F(UtestFuncToGraph, GraphToFunctionDef_AddNodeAttr_null_node) {
  domi::tensorflow::AttrValue value;
  value.set_i(10);
  GraphToFunctionDef::AddNodeAttr("test_attr", value, nullptr);
}

TEST_F(UtestFuncToGraph, GraphToFunctionDef_AddNodeAttr_success) {
  domi::tensorflow::NodeDef node_def;
  node_def.set_name("test_node");
  domi::tensorflow::AttrValue value;
  value.set_i(10);
  GraphToFunctionDef::AddNodeAttr("test_attr", value, &node_def);
  EXPECT_TRUE(node_def.attr().find("test_attr") != node_def.attr().end());
}

TEST_F(UtestFuncToGraph, NameMapHelper_GetUniqueName) {
  NameMapHelper helper;
  EXPECT_EQ(helper.GetUniqueName("name1"), "name1");
  EXPECT_EQ(helper.GetUniqueName("name1"), "name1_0");
  EXPECT_EQ(helper.GetUniqueName("name1"), "name1_1");
}

TEST_F(UtestFuncToGraph, NameMapHelper_UniqueInputOrOutputName) {
  NameMapHelper helper;
  EXPECT_EQ(helper.UniqueInputOrOutputName("TestName"), "TestName");
  EXPECT_EQ(helper.UniqueInputOrOutputName(""), "unknown");
  EXPECT_EQ(helper.UniqueInputOrOutputName("test@name#"), "test@name#");
}

TEST_F(UtestFuncToGraph, NameMapHelper_UniqueNodeName) {
  NameMapHelper helper;
  EXPECT_EQ(helper.UniqueNodeName("node1"), "node1");
  std::string result = helper.Renormalize("node1");
  EXPECT_EQ(result, "node1");
  result = helper.Renormalize("nonexistent");
  EXPECT_EQ(result, "");
}
}  // namespace ge
