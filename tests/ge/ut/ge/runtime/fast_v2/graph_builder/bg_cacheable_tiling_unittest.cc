/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph_builder/bg_tiling.h"
#include "graph_builder/bg_platform.h"
#include <gtest/gtest.h>
#include <nlohmann/json.hpp>
#include "ge_graph_dsl/graph_dsl.h"
#include "engine/gelocal/inputs_converter.h"
#include "register/node_converter_registry.h"
#include "exe_graph_comparer.h"
#include "faker/global_data_faker.h"
#include "common/bg_test.h"
#include "common/share_graph.h"
#include "engine/aicore/fe_rt2_common.h"
#include "common/topo_checker.h"
#include "common/summary_checker.h"
#include "macro_utils/dt_public_scope.h"
#include "register/op_impl_registry.h"
#include "macro_utils/dt_public_unscope.h"
#include "register/op_tiling_registry.h"
#include "register/op_tiling/op_tiling_constants.h"
#include "common/const_data_helper.h"
#include "kernel/common_kernel_impl/tiling.h"
#include "graph/utils/graph_dump_utils.h"
#include "graph/ge_local_context.h"
#include "depends/runtime/src/runtime_stub.h"
#include "common/opskernel/ops_kernel_info_types.h"

namespace gert {
namespace bg {
namespace {
using namespace ge;
constexpr char const *kStubJson =
    "{\"vars\": {\"srcFormat\": \"NCHW\", \"dstFormat\": \"NC1HWC0\", \"dType\": \"float16\", \"ub_size\": 126464, "
    "\"block_dim\": 32, \"input_size\": 0, \"hidden_size\": 0, \"group\": 1}}";
constexpr char const *kStubJsonErrorFormat =
    "{\"vars\": {\"srcFormat\": \"NCHW\", \"dstFormat\": \"NC1HWC0\", \"dType\": \"float16\", \"ub_size\": 126464, "
    "\"block_dim\": 32, \"input_size\": 0, \"hidden_size\": 0, \"group\": 1}, }";
constexpr char const *kExpectAtomicJson =
    "{\"vars\": {\"srcFormat\": \"NCHW\", \"dstFormat\": \"NC1HWC0\", \"dType\": \"float16\", \"ub_size\": 126464, "
    "\"block_dim\": 32, \"input_size\": 0, \"hidden_size\": 0, \"group\": 1}, \"_workspace_index_list\":[0,1]}";
void AddCompiledJson(const ge::NodePtr &node, bool atomic, const char *json = nullptr) {
  if (json == nullptr) {
    json = kStubJson;
  }

  if (atomic) {
    AttrUtils::SetStr(node->GetOpDesc(), "_atomic_compile_info_json", json);
    AttrUtils::SetInt(node->GetOpDesc(), "atomic_op_para_size", 2048);
  } else {
    AttrUtils::SetStr(node->GetOpDesc(), "compile_info_json", json);
    AttrUtils::SetInt(node->GetOpDesc(), "op_para_size", 2048);
  }
}
/*
 *
 *     add1
 *    /  \
 * data1 data2
 */
ComputeGraphPtr BuildTwoInputsGraph(const std::string &node_type) {
  DEF_GRAPH(g1) {
    CHAIN(NODE("data1", "Data")->NODE(node_type, node_type)->NODE("NetOutput", "NetOutput"));
    CHAIN(NODE("data2", "Data")->EDGE(0, 1)->NODE(node_type));
  };
  auto graph = ToComputeGraph(g1);

  auto data1 = graph->FindNode("data1");
  data1->GetOpDesc()->MutableOutputDesc(0)->SetShape(GeShape({8, 3, 224, 224}));
  data1->GetOpDesc()->MutableOutputDesc(0)->SetOriginShape(GeShape({8, 3, 224, 224}));
  data1->GetOpDesc()->MutableOutputDesc(0)->SetDataType(DT_FLOAT);
  data1->GetOpDesc()->MutableOutputDesc(0)->SetOriginDataType(DT_FLOAT);
  data1->GetOpDesc()->MutableOutputDesc(0)->SetFormat(ge::FORMAT_NCHW);
  data1->GetOpDesc()->MutableOutputDesc(0)->SetOriginFormat(ge::FORMAT_NCHW);
  AttrUtils::SetInt(data1->GetOpDesc(), "index", 0);

  auto data2 = graph->FindNode("data2");
  data2->GetOpDesc()->MutableOutputDesc(0)->SetShape(GeShape({8, 3, 224, 224}));
  data2->GetOpDesc()->MutableOutputDesc(0)->SetOriginShape(GeShape({8, 3, 224, 224}));
  data2->GetOpDesc()->MutableOutputDesc(0)->SetDataType(DT_FLOAT);
  data2->GetOpDesc()->MutableOutputDesc(0)->SetOriginDataType(DT_FLOAT);
  data2->GetOpDesc()->MutableOutputDesc(0)->SetFormat(ge::FORMAT_NCHW);
  data2->GetOpDesc()->MutableOutputDesc(0)->SetOriginFormat(ge::FORMAT_NCHW);
  AttrUtils::SetInt(data2->GetOpDesc(), "index", 1);

  auto add1 = graph->FindNode(node_type);
  add1->GetOpDesc()->MutableOutputDesc(0)->SetShape(GeShape({8, 3, 224, 224}));
  add1->GetOpDesc()->MutableOutputDesc(0)->SetOriginShape(GeShape({8, 3, 224, 224}));
  add1->GetOpDesc()->MutableOutputDesc(0)->SetDataType(DT_FLOAT);
  add1->GetOpDesc()->MutableOutputDesc(0)->SetOriginDataType(DT_FLOAT);
  add1->GetOpDesc()->MutableOutputDesc(0)->SetFormat(ge::FORMAT_NCHW);
  add1->GetOpDesc()->MutableOutputDesc(0)->SetOriginFormat(ge::FORMAT_NCHW);

  add1->GetOpDesc()->MutableInputDesc(0)->SetShape(GeShape({8, 3, 224, 224}));
  add1->GetOpDesc()->MutableInputDesc(0)->SetOriginShape(GeShape({8, 3, 224, 224}));
  add1->GetOpDesc()->MutableInputDesc(0)->SetDataType(DT_FLOAT);
  add1->GetOpDesc()->MutableInputDesc(0)->SetOriginDataType(DT_FLOAT);
  add1->GetOpDesc()->MutableInputDesc(0)->SetFormat(ge::FORMAT_NCHW);
  add1->GetOpDesc()->MutableInputDesc(0)->SetOriginFormat(ge::FORMAT_NCHW);

  add1->GetOpDesc()->MutableInputDesc(1)->SetShape(GeShape({8, 3, 224, 224}));
  add1->GetOpDesc()->MutableInputDesc(1)->SetOriginShape(GeShape({8, 3, 224, 224}));
  add1->GetOpDesc()->MutableInputDesc(1)->SetDataType(DT_FLOAT);
  add1->GetOpDesc()->MutableInputDesc(1)->SetOriginDataType(DT_FLOAT);
  add1->GetOpDesc()->MutableInputDesc(1)->SetFormat(ge::FORMAT_NCHW);
  add1->GetOpDesc()->MutableInputDesc(1)->SetOriginFormat(ge::FORMAT_NCHW);
  return graph;
}

/*      add2
 *      /    \
 *     add1   \
 *    /     \  \
 * data1    data2
 */
ComputeGraphPtr BuildDifferentOppImplVersionGraph() {
  DEF_GRAPH(g1) {
    CHAIN(NODE("data1", "Data")->NODE("add1", "Add")->NODE("add2", "Add")->NODE("NetOutput", "NetOutput"));
    CHAIN(NODE("data2", "Data")->EDGE(0, 1)->NODE("add1"));
    CHAIN(NODE("data2")->EDGE(0, 1)->NODE("add2"));
  };
  auto graph = ToComputeGraph(g1);

  GeTensorDesc tensor_desc;
  tensor_desc.SetShape(GeShape({8, 3, 224, 224}));
  tensor_desc.SetOriginShape(GeShape({8, 3, 224, 224}));
  tensor_desc.SetDataType(DT_FLOAT);
  tensor_desc.SetOriginDataType(DT_FLOAT);
  tensor_desc.SetFormat(ge::FORMAT_NCHW);
  tensor_desc.SetOriginFormat(ge::FORMAT_NCHW);

  const auto &data1 = graph->FindNode("data1");
  *data1->GetOpDescBarePtr()->MutableOutputDesc(0) = tensor_desc;
  AttrUtils::SetInt(data1->GetOpDesc(), ATTR_NAME_INDEX, 0);

  auto data2 = graph->FindNode("data2");
  *data2->GetOpDescBarePtr()->MutableOutputDesc(0) = tensor_desc;
  AttrUtils::SetInt(data2->GetOpDesc(), ATTR_NAME_INDEX, 1);
  // add1 没有指定 _opp_path
  auto add1 = graph->FindNode("add1");
  *add1->GetOpDesc()->MutableOutputDesc(0) = tensor_desc;
  *add1->GetOpDesc()->MutableInputDesc(0) = tensor_desc;
  *add1->GetOpDesc()->MutableInputDesc(1) = tensor_desc;
  // add2 指定 _opp_path 为 1
  auto add2 = graph->FindNode("add2");
  *add2->GetOpDesc()->MutableOutputDesc(0) = tensor_desc;
  *add2->GetOpDesc()->MutableInputDesc(0) = tensor_desc;
  *add2->GetOpDesc()->MutableInputDesc(1) = tensor_desc;
  AttrUtils::SetInt(add2->GetOpDesc(), ATTR_NAME_BINARY_SOURCE, 1);

  return graph;
}
}  // namespace

class BgCacheableTilingUT : public BgTestAutoCreateFrame {
 public:
  void TilingTopoCorrect(ge::ExecuteGraph *exe_graph, const std::vector<ValueHolderPtr> &tiling_rets,
                         const std::vector<ValueHolderPtr> &io_shapes, const ValueHolderPtr &platform) {
    for (const auto &tiling_ret : tiling_rets) {
      ASSERT_NE(tiling_ret, nullptr);
    }
    std::vector<FastSrcNode> expect_from;
    for (const auto &io_shape : io_shapes) {
      expect_from.emplace_back(io_shape);
    }
    expect_from.emplace_back("TilingParse");
    expect_from.emplace_back(platform);
    // UT中未执行CEM，因此PrepareTilingFwkData还在main图上，需要将校验对象InnerData替换为PrepareTilingFwkData
    expect_from.emplace_back("PrepareCacheableTilingFwkData");
    expect_from.emplace_back("InnerData");
    ASSERT_EQ(FastNodeTopoChecker(tiling_rets[0]).StrictConnectFrom(expect_from), "success");
    auto tiling_parse_node = ge::ExecuteGraphUtils::FindFirstNodeMatchType(exe_graph, "TilingParse");
    ASSERT_NE(tiling_parse_node, nullptr);
    ASSERT_EQ(FastNodeTopoChecker(tiling_parse_node).StrictConnectFrom({{"Const"}, {"Data"}, {"Const"}, {"Const"}}),
              "success");
    auto find_tiling_func_node =
        ge::ExecuteGraphUtils::FindFirstNodeMatchType(init_frame_->GetExecuteGraph().get(), "FindTilingFunc");
    ASSERT_NE(find_tiling_func_node, nullptr);
    ASSERT_EQ(FastNodeTopoChecker(find_tiling_func_node).StrictConnectFrom({{"Const"}, {"GetSpaceRegistry"}}),
              "success");
    auto tiling_fwk_data_node =
        ge::ExecuteGraphUtils::FindFirstNodeMatchType(exe_graph, "PrepareCacheableTilingFwkData");
    // Check "FindTilingFunc" in init graph link to "PrepareCacheableTilingFwkData" in main graph (ut only)
    ConnectFromInitToMain(find_tiling_func_node, 0, tiling_fwk_data_node, 0);
  }
  void CompatibleTopoCorrect(ge::ExecuteGraph *exe_graph, const std::vector<ValueHolderPtr> &tiling_ret,
                             const std::vector<ValueHolderPtr> &io_shapes) {
    for (const auto &tr : tiling_ret) {
      ASSERT_NE(tr, nullptr);
    }

    auto find_tiling_node = ge::ExecuteGraphUtils::FindFirstNodeMatchType(exe_graph, "FindCompatibleTilingFunc");
    ASSERT_NE(find_tiling_node, nullptr);
    EXPECT_EQ(FastNodeTopoChecker(find_tiling_node).StrictConnectFrom({{"Const"}}), "success");

    auto tiling_parse_node = ge::ExecuteGraphUtils::FindFirstNodeMatchType(exe_graph, "CompatibleTilingParse");
    ASSERT_NE(tiling_parse_node, nullptr);
    EXPECT_EQ(
        FastNodeTopoChecker(tiling_parse_node)
            .StrictConnectFrom({{"CreateOpFromBuffer"},
                                {"Const"},  // compile_info_json_holder
                                {"Const"},  // compile_info_key_holder
                                {"FindCompatibleTilingFunc",
                                 static_cast<size_t>(kernel::FindCompatibleTilingFuncOutputIndex::kTilingVersion)},
                                {"FindCompatibleTilingFunc",
                                 static_cast<size_t>(kernel::FindCompatibleTilingFuncOutputIndex::kTilingParseFunc)}}),
        "success");
    ASSERT_NE(tiling_ret[0], nullptr);
    auto tiling_node = tiling_ret[0]->GetFastNode();
    ASSERT_NE(tiling_node, nullptr);

    std::vector<FastSrcNode> expect_tiling_node_from = {
        {"CreateOpFromBuffer"},
        {"CompatibleTilingParse", 0},
        {"FindCompatibleTilingFunc", static_cast<size_t>(kernel::FindCompatibleTilingFuncOutputIndex::kTilingVersion)},
        {"PrepareCacheableTilingFwkData", 0}};
    for (const auto &io_shape : io_shapes) {
      expect_tiling_node_from.emplace_back(io_shape);
    }
    EXPECT_EQ(FastNodeTopoChecker(tiling_node).StrictConnectFrom(expect_tiling_node_from), "success");
    auto tiling_fwk_data_node =
        ge::ExecuteGraphUtils::FindFirstNodeMatchType(exe_graph, "PrepareCacheableTilingFwkData");
    EXPECT_NE(tiling_fwk_data_node, nullptr);
    std::vector<FastSrcNode> fwk_data_expect_nodes = {
        {"FindCompatibleTilingFunc", static_cast<size_t>(kernel::FindCompatibleTilingFuncOutputIndex::kTilingFunc)}};
    EXPECT_EQ(FastNodeTopoChecker(tiling_fwk_data_node).StrictConnectFrom(fwk_data_expect_nodes, true), "success");
  }

 protected:
  void SetUp() override {
    setenv("ENABLE_TILING_CACHE", "1", 1);
    BgTestAutoCreateFrame::SetUp();
    auto init = ValueHolder::CreateVoid<bg::ValueHolder>("Init", {});
    auto main = ValueHolder::CreateVoid<bg::ValueHolder>("Main", {});
    auto de_init = ValueHolder::CreateVoid<bg::ValueHolder>("DeInit", {});

    ValueHolder::PushGraphFrame(init, "Init");
    init_frame_ = ValueHolder::PopGraphFrame({}, {});

    ValueHolder::PushGraphFrame(de_init, "DeInit");
    de_init_frame_ = ValueHolder::PopGraphFrame();

    ValueHolder::PushGraphFrame(main, "Main");
    auto launch_arg_output =
        bg::ValueHolder::CreateDataOutput("AllocLaunchArg", {}, static_cast<size_t>(AllocLaunchArgOutputs::kNum));
    fake_launch_arg_ = launch_arg_output[static_cast<size_t>(AllocLaunchArgOutputs::kRtArg)];
  }

  void TearDown() override {
    BgTest::TearDown();
    init_frame_.reset();
    de_init_frame_.reset();
    fake_launch_arg_ = nullptr;
    unsetenv("ENABLE_TILING_CACHE");
  }

  std::unique_ptr<GraphFrame> init_frame_;
  std::unique_ptr<GraphFrame> de_init_frame_;
  ValueHolderPtr fake_launch_arg_;
};

TEST_F(BgCacheableTilingUT, BgTiling_Ok_TopoCorrectSameNodeWithDifferentOppImplVersion) {
  auto in_shape0 = ValueHolder::CreateFeed(0);
  auto in_shape1 = ValueHolder::CreateFeed(1);
  auto out_shape = ValueHolder::CreateFeed(2);
  auto platform = ValueHolder::CreateFeed(3);
  auto graph = BuildDifferentOppImplVersionGraph();
  ASSERT_NE(graph, nullptr);
  auto add_node1 = graph->FindNode("add1");
  ASSERT_NE(add_node1, nullptr);
  AddCompiledJson(add_node1, false);
  auto add_node2 = graph->FindNode("add2");
  ASSERT_NE(add_node2, nullptr);
  AddCompiledJson(add_node2, false);

  // 构造 global_data 并设置 opp, opp_kernel 的 space_registry
  auto root_model = GeModelBuilder(graph).BuildGeRootModel();
  auto global_data = GlobalDataFaker(root_model).FakeWithHandleAiCore("Add", false).Build();
  auto space_registry_array = SpaceRegistryFaker().BuildRegistryArray();
  global_data.SetSpaceRegistriesV2(*space_registry_array);

  auto launch_arg_output =
    bg::ValueHolder::CreateDataOutput("AllocLaunchArg", {}, static_cast<size_t>(AllocLaunchArgOutputs::kNum));
  auto fake_launch_arg2 = launch_arg_output[static_cast<size_t>(AllocLaunchArgOutputs::kRtArg)];
  bg::LowerConstDataNode(global_data);
  auto tiling_rets1 = Tiling(add_node1, {in_shape0, in_shape1}, {out_shape}, {platform, global_data, fake_launch_arg_});
  auto tiling_rets2 = Tiling(add_node2, {in_shape0, in_shape1}, {out_shape}, {platform, global_data, fake_launch_arg2});
  auto frame = ValueHolder::PopGraphFrame();
  ASSERT_NE(frame, nullptr);
  auto exe_graph = frame->GetExecuteGraph().get();
  ASSERT_NE(exe_graph, nullptr);
  ge::DumpGraph(exe_graph, "CachebaleTilingUT");

  // check main 图中节点数量, 由于没走CEM, main图上有俩InnerData连给PrepareCacheableTilingFwkData
  ASSERT_EQ(ExeGraphSummaryChecker(exe_graph).StrictAllNodeTypes({{"Data", 6},
                                                                  {"CacheableTiling", 2},
                                                                  {"TilingAppendDfxInfo", 2},
                                                                  {"TilingParse", 2},
                                                                  {"InnerData", 3},
                                                                  {"SplitRtStreams", 1},
                                                                  {"Const", 23},
                                                                  {"CalcTensorSizeFromStorage", 2},
                                                                  {"PrepareCacheableTilingFwkData", 2},
                                                                  {"AllocLaunchArg", 2}}),
            "success");
  ge::DumpGraph(init_frame_->GetExecuteGraph().get(), "CachebableTilingUTInit");

  // check init 图中节点数量
  ASSERT_EQ(ExeGraphSummaryChecker(init_frame_->GetExecuteGraph().get())
                .StrictAllNodeTypes({
                    {"InnerNetOutput", 1},
                    {"FindTilingFunc", 2},
                    {"ConstData", 3},
                    {"Data", 1},
                    {"Const", 7},
                    {"SplitRtStreams", 1},
                    {"GetSpaceRegistry", 2},
                }),
            "success");
  // check topo连接关系，包括子图内部以及子图之间
  TilingTopoCorrect(exe_graph, tiling_rets1, {in_shape0, in_shape1, out_shape}, platform);
  ASSERT_EQ(tiling_rets1.size(), static_cast<size_t>(kernel::TilingExOutputIndex::kNum));
  TilingTopoCorrect(exe_graph, tiling_rets2, {in_shape0, in_shape1, out_shape}, platform);
  ASSERT_EQ(tiling_rets2.size(), static_cast<size_t>(kernel::TilingExOutputIndex::kNum));
}

TEST_F(BgCacheableTilingUT, BgTiling_TopoCorrect_KnownWorkspace) {
  auto in_shape0 = ValueHolder::CreateFeed(0);
  auto in_shape1 = ValueHolder::CreateFeed(1);
  auto out_shape = ValueHolder::CreateFeed(2);
  auto platform = ValueHolder::CreateFeed(3);
  auto graph = BuildTwoInputsGraph("Add");
  ASSERT_NE(graph, nullptr);
  auto add_node = graph->FindNode("Add");
  ASSERT_NE(add_node, nullptr);
  AddCompiledJson(add_node, false);
  auto root_model = GeModelBuilder(graph).BuildGeRootModel();
  auto global_data = GlobalDataFaker(root_model).FakeWithHandleAiCore("Add", false).Build();
  bg::LowerConstDataNode(global_data);
  auto op_desc = add_node->GetOpDesc();
  ASSERT_NE(op_desc, nullptr);
  op_desc->SetWorkspaceBytes({4, 8});
  AttrUtils::SetStr(op_desc, ATTR_NAME_ALIAS_ENGINE_NAME, "mix_l2");

  auto tiling_rets = Tiling(add_node, {in_shape0, in_shape1}, {out_shape}, {platform, global_data, fake_launch_arg_});
  auto frame = ValueHolder::PopGraphFrame();
  ASSERT_NE(frame, nullptr);
  auto exe_graph = frame->GetExecuteGraph().get();
  ASSERT_NE(exe_graph, nullptr);
  ge::DumpGraph(exe_graph, "CachebaleTilingUT");

  ASSERT_EQ(ExeGraphSummaryChecker(exe_graph).StrictAllNodeTypes({{"Data", 6},
                                                                  {"CacheableTiling", 1},
                                                                  {"TilingParse", 1},
                                                                  {"InnerData", 2},
                                                                  {"SplitRtStreams", 1},
                                                                  {"Const", 13},
                                                                  {"TilingAppendWorkspace", 1},
                                                                  {"TilingAppendDfxInfo", 1},
                                                                  {"CalcTensorSizeFromStorage", 1},
                                                                  {"PrepareCacheableTilingFwkData", 1},
                                                                  {"AllocLaunchArg", 1}}),
            "success");
  TilingTopoCorrect(exe_graph, tiling_rets, {in_shape0, in_shape1, out_shape}, platform);
  ASSERT_EQ(tiling_rets.size(), static_cast<size_t>(kernel::TilingExOutputIndex::kNum));
}

TEST_F(BgCacheableTilingUT, BgTiling_TopoCorrect_MemCheck) {
  auto in_shape0 = ValueHolder::CreateFeed(0);
  auto in_shape1 = ValueHolder::CreateFeed(1);
  auto out_shape = ValueHolder::CreateFeed(2);
  auto platform = ValueHolder::CreateFeed(3);
  auto graph = BuildTwoInputsGraph("Add");
  ASSERT_NE(graph, nullptr);
  auto add_node = graph->FindNode("Add");
  ASSERT_NE(add_node, nullptr);
  AddCompiledJson(add_node, false);
  auto root_model = GeModelBuilder(graph).BuildGeRootModel();
  auto global_data = GlobalDataFaker(root_model).FakeWithHandleAiCore("Add", false).Build();
  bg::LowerConstDataNode(global_data);
  auto op_desc = add_node->GetOpDesc();
  ASSERT_NE(op_desc, nullptr);
  op_desc->SetWorkspaceBytes({4, 8});
  AttrUtils::SetStr(op_desc, ATTR_NAME_ALIAS_ENGINE_NAME, "mix_l2");
  ge::AttrUtils::SetBool(op_desc, "_memcheck", true);
  auto tiling_rets = Tiling(add_node, {in_shape0, in_shape1}, {out_shape}, {platform, global_data, fake_launch_arg_});
  auto frame = ValueHolder::PopGraphFrame();
  ASSERT_NE(frame, nullptr);
  auto exe_graph = frame->GetExecuteGraph().get();
  ASSERT_NE(exe_graph, nullptr);
  DumpGraph(exe_graph, "CachebaleTilingUT");

  ASSERT_EQ(ExeGraphSummaryChecker(exe_graph).StrictAllNodeTypes({{"Data", 6},
                                                                  {"CacheableTiling", 1},
                                                                  {"TilingParse", 1},
                                                                  {"InnerData", 2},
                                                                  {"SplitRtStreams", 1},
                                                                  {"Const", 13},
                                                                  {"TilingAppendWorkspace", 1},
                                                                  {"TilingAppendDfxInfo", 1},
                                                                  {"CalcTensorSizeFromStorage", 1},
                                                                  {"PrepareCacheableTilingFwkData", 1},
                                                                  {"AllocLaunchArg", 1}}),
            "success");
  TilingTopoCorrect(exe_graph, tiling_rets, {in_shape0, in_shape1, out_shape}, platform);
  ASSERT_EQ(tiling_rets.size(), static_cast<size_t>(kernel::TilingExOutputIndex::kNum));
}

TEST_F(BgCacheableTilingUT, BgTiling_TopoCorrect_MemCheck2) {
  auto in_shape0 = ValueHolder::CreateFeed(0);
  auto in_shape1 = ValueHolder::CreateFeed(1);
  auto out_shape = ValueHolder::CreateFeed(2);
  auto platform = ValueHolder::CreateFeed(3);
  auto graph = BuildTwoInputsGraph("Add");
  ASSERT_NE(graph, nullptr);
  auto add_node = graph->FindNode("Add");
  ASSERT_NE(add_node, nullptr);
  AddCompiledJson(add_node, false);
  auto op_desc = add_node->GetOpDesc();
  ASSERT_NE(op_desc, nullptr);

  op_desc->AppendIrInput("input0", kIrInputDynamic);
  op_desc->AppendIrInput("input1", kIrInputDynamic);
  op_desc->AppendIrOutput("output", kIrOutputDynamic);
  (void)ge::AttrUtils::SetStr(op_desc, "dynamicParamMode", "folded_with_desc");
  (void)ge::AttrUtils::SetListListInt(op_desc, "_dynamic_inputs_indexes", {{1}});
  (void)ge::AttrUtils::SetListListInt(op_desc, "_dynamic_outputs_indexes", {{0}});
  std::shared_ptr<domi::ModelTaskDef> model_task_def = MakeShared<domi::ModelTaskDef>();
  auto &task_def = *model_task_def->add_task();

  domi::FftsPlusTaskDef *ffts_plus_task_def = task_def.mutable_ffts_plus_task();
  domi::FftsPlusCtxDef *ctx_def = ffts_plus_task_def->add_ffts_plus_ctx();
  ctx_def->set_op_index(op_desc->GetId());
  ctx_def->set_context_id(0);
  ctx_def->set_context_type(static_cast<uint32_t>(RT_CTX_TYPE_MIX_AIV));
  domi::FftsPlusMixAicAivCtxDef *mixctx_def = ctx_def->mutable_mix_aic_aiv_ctx();
  mixctx_def->set_args_format("{i0}{i_instance0}{i_desc1}{o_desc0}{o_instance0}{ws0}{overflow_addr}");

  LoweringGlobalData global_data;
  global_data.AddCompiledResult(add_node, {{task_def}});
    auto space_registry_array = OpImplSpaceRegistryV2Array();
  space_registry_array[static_cast<size_t>(gert::OppImplVersionTag::kOpp)] = SpaceRegistryFaker().Build();
  global_data.SetSpaceRegistriesV2(space_registry_array);
  bg::LowerConstDataNode(global_data);
  auto task_defs = global_data.FindCompiledResult(add_node)->GetTaskDefs();
  auto task_def_tmp = task_defs[0];
  auto kernel_def_tmp = task_def_tmp.mutable_kernel();
  GELOGI("arg format: %s", kernel_def_tmp->context().args_format().c_str());

  op_desc->SetWorkspaceBytes({4, 8});
  ge::AttrUtils::SetStr(op_desc, ATTR_NAME_ALIAS_ENGINE_NAME, "mix_l2");
  ge::AttrUtils::SetBool(op_desc, "_memcheck", true);
  auto tiling_rets = Tiling(add_node, {in_shape0, in_shape1}, {out_shape}, {platform, global_data, fake_launch_arg_});
  auto frame = ValueHolder::PopGraphFrame();
  ASSERT_NE(frame, nullptr);
  auto exe_graph = frame->GetExecuteGraph().get();
  ASSERT_NE(exe_graph, nullptr);
  DumpGraph(exe_graph, "CachebaleTilingUT");

  ASSERT_EQ(ExeGraphSummaryChecker(exe_graph).StrictAllNodeTypes({{"Data", 4},
                                                                  {"CacheableTiling", 1},
                                                                  {"TilingParse", 1},
                                                                  {"InnerData", 2},
                                                                  {"Const", 12},
                                                                  {"TilingAppendWorkspace", 1},
                                                                  {"TilingAppendDfxInfo", 1},
                                                                  {"CalcTensorSizeFromStorage", 1},
                                                                  {"PrepareCacheableTilingFwkData", 1},
                                                                  {"AllocLaunchArg", 1}}),
            "success");
  TilingTopoCorrect(exe_graph, tiling_rets, {in_shape0, in_shape1, out_shape}, platform);
  ASSERT_EQ(tiling_rets.size(), static_cast<size_t>(kernel::TilingExOutputIndex::kNum));
}

TEST_F(BgCacheableTilingUT, BgTiling_TopoCorrect_MemCheck3) {
  auto in_shape0 = ValueHolder::CreateFeed(0);
  auto in_shape1 = ValueHolder::CreateFeed(1);
  auto out_shape = ValueHolder::CreateFeed(2);
  auto platform = ValueHolder::CreateFeed(3);
  auto graph = BuildTwoInputsGraph("Add");
  ASSERT_NE(graph, nullptr);
  auto add_node = graph->FindNode("Add");
  ASSERT_NE(add_node, nullptr);
  AddCompiledJson(add_node, false);
  auto op_desc = add_node->GetOpDesc();
  ASSERT_NE(op_desc, nullptr);

  op_desc->AppendIrInput("input0", kIrInputDynamic);
  op_desc->AppendIrInput("input1", kIrInputDynamic);
  op_desc->AppendIrOutput("output", kIrOutputRequired);
  (void)ge::AttrUtils::SetStr(op_desc, "dynamicParamMode", "folded_with_desc");
  (void)ge::AttrUtils::SetListListInt(op_desc, "_dynamic_inputs_indexes", {{1}});
  (void)ge::AttrUtils::SetInt(op_desc, ge::ATTR_NAME_UNKNOWN_SHAPE_TYPE, 3);
  std::shared_ptr<domi::ModelTaskDef> model_task_def = MakeShared<domi::ModelTaskDef>();
  auto &task_def = *model_task_def->add_task();
  task_def.set_type(static_cast<uint32_t>(ge::ModelTaskType::MODEL_TASK_KERNEL));
  domi::KernelDef *kernel_def = task_def.mutable_kernel();
  kernel_def->set_stub_func("stub_func");
  kernel_def->set_args_size(64);
  string args(64, '1');
  kernel_def->set_args(args.data(), 64);
  domi::KernelContext *context = kernel_def->mutable_context();
  context->set_op_index(op_desc->GetId());
  context->set_kernel_type(static_cast<uint32_t>(ccKernelType::TE));
  context->set_args_format("{i0}{i_desc1}{i_instance0}{o0}{o_instance0}{ws0}{overflow_addr}");

  LoweringGlobalData global_data;
  global_data.AddCompiledResult(add_node, {{task_def}});
    auto space_registry_array = OpImplSpaceRegistryV2Array();
  space_registry_array[static_cast<size_t>(gert::OppImplVersionTag::kOpp)] = SpaceRegistryFaker().Build();
  global_data.SetSpaceRegistriesV2(space_registry_array);
  bg::LowerConstDataNode(global_data);
  auto task_defs = global_data.FindCompiledResult(add_node)->GetTaskDefs();
  auto task_def_tmp = task_defs[0];
  auto kernel_def_tmp = task_def_tmp.mutable_kernel();
  GELOGI("arg format: %s", kernel_def_tmp->context().args_format().c_str());

  op_desc->SetWorkspaceBytes({4, 8});
  ge::AttrUtils::SetBool(op_desc, "_memcheck", true);
  auto tiling_rets = Tiling(add_node, {in_shape0, in_shape1}, {out_shape}, {platform, global_data, fake_launch_arg_});
  auto frame = ValueHolder::PopGraphFrame();
  ASSERT_NE(frame, nullptr);
  auto exe_graph = frame->GetExecuteGraph().get();
  ASSERT_NE(exe_graph, nullptr);
  DumpGraph(exe_graph, "CachebaleTilingUT");

  ASSERT_EQ(ExeGraphSummaryChecker(exe_graph).StrictAllNodeTypes({{"Data", 4},
                                                                  {"CacheableTiling", 1},
                                                                  {"TilingParse", 1},
                                                                  {"InnerData", 2},
                                                                  {"Const", 12},
                                                                  {"TilingAppendWorkspace", 1},
                                                                  {"TilingAppendDfxInfo", 1},
                                                                  {"CalcTensorSizeFromStorage", 1},
                                                                  {"PrepareCacheableTilingFwkData", 1},
                                                                  {"AllocLaunchArg", 1}}),
            "success");
  TilingTopoCorrect(exe_graph, tiling_rets, {in_shape0, in_shape1, out_shape}, platform);
  ASSERT_EQ(tiling_rets.size(), static_cast<size_t>(kernel::TilingExOutputIndex::kNum));
}

TEST_F(BgCacheableTilingUT, BgTiling_TopoCorrect_MemCheck4) {
  auto in_shape0 = ValueHolder::CreateFeed(0);
  auto in_shape1 = ValueHolder::CreateFeed(1);
  auto out_shape = ValueHolder::CreateFeed(2);
  auto platform = ValueHolder::CreateFeed(3);
  auto graph = BuildTwoInputsGraph("Add");
  ASSERT_NE(graph, nullptr);
  auto add_node = graph->FindNode("Add");
  ASSERT_NE(add_node, nullptr);
  AddCompiledJson(add_node, false);
  auto op_desc = add_node->GetOpDesc();
  ASSERT_NE(op_desc, nullptr);

  op_desc->AppendIrInput("input0", kIrInputDynamic);
  op_desc->AppendIrInput("input1", kIrInputDynamic);
  op_desc->AppendIrOutput("output", kIrOutputRequired);
  (void)ge::AttrUtils::SetStr(op_desc, "dynamicParamMode", "folded_with_desc");
  (void)ge::AttrUtils::SetListListInt(op_desc, "_dynamic_inputs_indexes", {{1}});
  std::shared_ptr<domi::ModelTaskDef> model_task_def = MakeShared<domi::ModelTaskDef>();
  auto &task_def = *model_task_def->add_task();
  task_def.set_type(static_cast<uint32_t>(ge::ModelTaskType::MODEL_TASK_ALL_KERNEL));
  domi::KernelDefWithHandle *kernel_def = task_def.mutable_kernel_with_handle();
  kernel_def->set_args_size(64);
  string args(64, '1');
  kernel_def->set_args(args.data(), 64);
  domi::KernelContext *context = kernel_def->mutable_context();
  context->set_op_index(op_desc->GetId());
  context->set_kernel_type(static_cast<uint32_t>(ccKernelType::TE));
  context->set_args_format("{i0}{i_instance0}{i_desc1}{o0}{o_instance0}{ws0}{overflow_addr}");

  LoweringGlobalData global_data;
  global_data.AddCompiledResult(add_node, {{task_def}});
    auto space_registry_array = OpImplSpaceRegistryV2Array();
  space_registry_array[static_cast<size_t>(gert::OppImplVersionTag::kOpp)] = SpaceRegistryFaker().Build();
  global_data.SetSpaceRegistriesV2(space_registry_array);
  bg::LowerConstDataNode(global_data);
  auto task_defs = global_data.FindCompiledResult(add_node)->GetTaskDefs();
  auto task_def_tmp = task_defs[0];
  auto kernel_def_tmp = task_def_tmp.mutable_kernel();
  GELOGI("arg format: %s", kernel_def_tmp->context().args_format().c_str());

  op_desc->SetWorkspaceBytes({4, 8});
  ge::AttrUtils::SetBool(op_desc, "_memcheck", true);
  auto tiling_rets = Tiling(add_node, {in_shape0, in_shape1}, {out_shape}, {platform, global_data, fake_launch_arg_});
  auto frame = ValueHolder::PopGraphFrame();
  ASSERT_NE(frame, nullptr);
  auto exe_graph = frame->GetExecuteGraph().get();
  ASSERT_NE(exe_graph, nullptr);
  DumpGraph(exe_graph, "CachebaleTilingUT");

  ASSERT_EQ(ExeGraphSummaryChecker(exe_graph).StrictAllNodeTypes({{"Data", 4},
                                                                  {"CacheableTiling", 1},
                                                                  {"TilingParse", 1},
                                                                  {"InnerData", 2},
                                                                  {"Const", 12},
                                                                  {"TilingAppendWorkspace", 1},
                                                                  {"TilingAppendDfxInfo", 1},
                                                                  {"CalcTensorSizeFromStorage", 1},
                                                                  {"PrepareCacheableTilingFwkData", 1},
                                                                  {"AllocLaunchArg", 1}}),
            "success");
  TilingTopoCorrect(exe_graph, tiling_rets, {in_shape0, in_shape1, out_shape}, platform);
  ASSERT_EQ(tiling_rets.size(), static_cast<size_t>(kernel::TilingExOutputIndex::kNum));
}

TEST_F(BgCacheableTilingUT, BgTiling_TopoCorrect_MemCheck5) {
  auto in_shape0 = ValueHolder::CreateFeed(0);
  auto in_shape1 = ValueHolder::CreateFeed(1);
  auto out_shape = ValueHolder::CreateFeed(2);
  auto platform = ValueHolder::CreateFeed(3);
  auto graph = BuildTwoInputsGraph("Add");
  ASSERT_NE(graph, nullptr);
  auto add_node = graph->FindNode("Add");
  ASSERT_NE(add_node, nullptr);
  AddCompiledJson(add_node, false);
  auto op_desc = add_node->GetOpDesc();
  ASSERT_NE(op_desc, nullptr);

  op_desc->AppendIrInput("input0", kIrInputRequired);
  op_desc->AppendIrInput("input1", kIrInputOptional);
  op_desc->AppendIrOutput("output0", kIrOutputRequired);
  op_desc->AppendIrOutput("output1", kIrOutputRequired);
  (void)ge::AttrUtils::SetListListInt(op_desc, "_dynamic_inputs_indexes", {{1}});
  std::shared_ptr<domi::ModelTaskDef> model_task_def = MakeShared<domi::ModelTaskDef>();
  auto &task_def = *model_task_def->add_task();
  task_def.set_type(static_cast<uint32_t>(ge::ModelTaskType::MODEL_TASK_ALL_KERNEL));
  domi::KernelDefWithHandle *kernel_def = task_def.mutable_kernel_with_handle();
  kernel_def->set_args_size(64);
  string args(64, '1');
  kernel_def->set_args(args.data(), 64);
  domi::KernelContext *context = kernel_def->mutable_context();
  context->set_op_index(op_desc->GetId());
  context->set_kernel_type(static_cast<uint32_t>(ccKernelType::TE));
  context->set_args_format("{i0}{i_instance0}{}{o0}{}{ws0}{overflow_addr}");

  LoweringGlobalData global_data;
  global_data.AddCompiledResult(add_node, {{task_def}});
    auto space_registry_array = OpImplSpaceRegistryV2Array();
  space_registry_array[static_cast<size_t>(gert::OppImplVersionTag::kOpp)] = SpaceRegistryFaker().Build();
  global_data.SetSpaceRegistriesV2(space_registry_array);
  bg::LowerConstDataNode(global_data);
  auto task_defs = global_data.FindCompiledResult(add_node)->GetTaskDefs();
  auto task_def_tmp = task_defs[0];
  auto kernel_def_tmp = task_def_tmp.mutable_kernel();
  GELOGI("arg format: %s", kernel_def_tmp->context().args_format().c_str());

  op_desc->SetWorkspaceBytes({4, 8});
  ge::AttrUtils::SetBool(op_desc, "_memcheck", true);
  auto tiling_rets = Tiling(add_node, {in_shape0, in_shape1}, {out_shape}, {platform, global_data, fake_launch_arg_});
  auto frame = ValueHolder::PopGraphFrame();
  ASSERT_NE(frame, nullptr);
  auto exe_graph = frame->GetExecuteGraph().get();
  ASSERT_NE(exe_graph, nullptr);
  DumpGraph(exe_graph, "CachebaleTilingUT");

  ASSERT_EQ(ExeGraphSummaryChecker(exe_graph).StrictAllNodeTypes({{"Data", 4},
                                                                  {"CacheableTiling", 1},
                                                                  {"TilingParse", 1},
                                                                  {"InnerData", 2},
                                                                  {"Const", 12},
                                                                  {"TilingAppendWorkspace", 1},
                                                                  {"TilingAppendDfxInfo", 1},
                                                                  {"CalcTensorSizeFromStorage", 1},
                                                                  {"PrepareCacheableTilingFwkData", 1},
                                                                  {"AllocLaunchArg", 1}}),
            "success");
  TilingTopoCorrect(exe_graph, tiling_rets, {in_shape0, in_shape1, out_shape}, platform);
  ASSERT_EQ(tiling_rets.size(), static_cast<size_t>(kernel::TilingExOutputIndex::kNum));
}

// Node without new impl and without compatible impl, go new tiling use auto tiling
TEST_F(BgCacheableTilingUT, ConstructAutoTilingOk) {
  std::string node_type = "bg_node_with_auto_tiling";
  auto graph = BuildTwoInputsGraph(node_type);
  auto test_node = graph->FindNode(node_type);
  AttrUtils::SetStr(test_node->GetOpDesc(), optiling::COMPILE_INFO_JSON, "{ key : value}");
  AttrUtils::SetStr(test_node->GetOpDesc(), optiling::COMPILE_INFO_KEY, "{}");

  auto root_model = GeModelBuilder(graph).BuildGeRootModel();
  auto global_data = GlobalDataFaker(root_model).Build();
  bg::LowerConstDataNode(global_data);
  LowerInput data_input = {{}, {}, &global_data};
  auto data1_ret = LoweringDataNode(graph->FindNode("data1"), data_input);
  auto data2_ret = LoweringDataNode(graph->FindNode("data2"), data_input);

  ASSERT_EQ(data1_ret.out_shapes.size(), 1);
  ASSERT_EQ(data2_ret.out_shapes.size(), 1);

  auto add_out_shape = test_node->GetOpDesc()->GetOutputDesc(0).GetShape();
  auto platform = AppendCoreTypeToPlatform(test_node, data_input.global_data);
  // use data2_out_shape as output_shapes of add_node
  auto tiling_ret = Tiling(test_node, {data1_ret.out_shapes[0], data2_ret.out_shapes[0]}, {data2_ret.out_shapes[0]},
                           {platform, global_data, fake_launch_arg_});
  ASSERT_EQ(tiling_ret.size(), static_cast<size_t>(kernel::TilingExOutputIndex::kNum));
  ASSERT_NE(tiling_ret[0], nullptr);
  EXPECT_EQ(tiling_ret[0]->GetFastNode()->GetType(), "CacheableTiling");
}

TEST_F(BgCacheableTilingUT, GetFrameworkOpTypeUsingAutoTilingOk) {
  const string real_node_type = "Add";
  auto graph = ShareGraph::FrameworkOPGraph(real_node_type);
  auto test_node = graph->FindNode("add1");
  AttrUtils::SetStr(test_node->GetOpDesc(), optiling::COMPILE_INFO_JSON, "{ key : value}");
  AttrUtils::SetStr(test_node->GetOpDesc(), optiling::COMPILE_INFO_KEY, "{}");

  auto root_model = GeModelBuilder(graph).BuildGeRootModel();
  auto global_data = GlobalDataFaker(root_model).Build();
  bg::LowerConstDataNode(global_data);
  LowerInput data_input = {{}, {}, &global_data};
  auto data1_ret = LoweringDataNode(graph->FindNode("data1"), data_input);
  auto data2_ret = LoweringDataNode(graph->FindNode("data2"), data_input);

  ASSERT_EQ(data1_ret.out_shapes.size(), 1);
  ASSERT_EQ(data2_ret.out_shapes.size(), 1);

  auto platform = AppendCoreTypeToPlatform(test_node, data_input.global_data);
  // use data2_out_shape as output_shapes of add_node
  auto tiling_ret = Tiling(test_node, {data1_ret.out_shapes[0], data2_ret.out_shapes[0]}, {data2_ret.out_shapes[0]},
                           {platform, *(data_input.global_data), fake_launch_arg_});
  ASSERT_EQ(tiling_ret.size(), static_cast<size_t>(kernel::TilingExOutputIndex::kNum));
  ASSERT_NE(tiling_ret[0], nullptr);
  EXPECT_EQ(tiling_ret[0]->GetFastNode()->GetType(), "CacheableTiling");

  auto find_node =
      ge::ExecuteGraphUtils::FindFirstNodeMatchType(init_frame_->GetExecuteGraph().get(), "FindTilingFunc");
  ASSERT_EQ(find_node->GetInDataNodes().size(), 2);
  auto node_type = *find_node->GetInDataNodes().begin();
  ASSERT_NE(node_type, nullptr);
  ge::Buffer buffer;
  ASSERT_TRUE(ExeGraphComparer::GetAttr(node_type, buffer));
  ASSERT_NE(buffer.GetData(), nullptr);
  EXPECT_STREQ(reinterpret_cast<char *>(buffer.GetData()), "Add");
}

TEST_F(BgCacheableTilingUT, FallibleTiling_Ok_TopoCorrectWithMemCheck) {
  auto in_shape0 = ValueHolder::CreateFeed(0);
  auto in_shape1 = ValueHolder::CreateFeed(1);
  auto out_shape = ValueHolder::CreateFeed(2);
  auto platform = ValueHolder::CreateFeed(3);
  auto graph = BuildTwoInputsGraph("Add");
  ASSERT_NE(graph, nullptr);
  auto add_node = graph->FindNode("Add");
  ASSERT_NE(add_node, nullptr);
  AddCompiledJson(add_node, false);
  auto root_model = GeModelBuilder(graph).BuildGeRootModel();
  auto global_data = GlobalDataFaker(root_model).FakeWithHandleAiCore("Add", false).Build();
  bg::LowerConstDataNode(global_data);
  auto op_desc = add_node->GetOpDesc();
  ASSERT_NE(op_desc, nullptr);
  op_desc->SetWorkspaceBytes({4, 8});
  AttrUtils::SetStr(op_desc, ATTR_NAME_ALIAS_ENGINE_NAME, "mix_l2");
  ge::AttrUtils::SetBool(op_desc, "_memcheck", true);
  auto tiling_rets =
      FallibleTiling(add_node, {in_shape0, in_shape1}, {out_shape}, {platform, global_data, fake_launch_arg_});
  auto frame = ValueHolder::PopGraphFrame();
  ASSERT_NE(frame, nullptr);
  auto exe_graph = frame->GetExecuteGraph().get();
  ASSERT_NE(exe_graph, nullptr);
  DumpGraph(exe_graph, "TilingUT");

  ASSERT_EQ(ExeGraphSummaryChecker(exe_graph).StrictAllNodeTypes({{"Data", 6},
                                                                  {"TilingParse", 1},
                                                                  {"InnerData", 2},
                                                                  {"SplitRtStreams", 1},
                                                                  {"Const", 12},
                                                                  {"CacheableFallibleTiling", 1},
                                                                  {"TilingAppendDfxInfo", 1},
                                                                  {"CalcTensorSizeFromStorage", 1},
                                                                  {"PrepareCacheableTilingFwkData", 1},
                                                                  {"AllocLaunchArg", 1}}),
            "success");
  TilingTopoCorrect(exe_graph, tiling_rets, {in_shape0, in_shape1, out_shape}, platform);
  ASSERT_EQ(tiling_rets.size(), static_cast<size_t>(kernel::FallibleTilingExOutputIndex::kFallibleOutputNum));
}
}  // namespace bg
}  // namespace gert
