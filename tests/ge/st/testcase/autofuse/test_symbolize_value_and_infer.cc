/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <memory>
#include <gtest/gtest.h>
#include <dlfcn.h>
#include "graph/utils/graph_utils_ex.h"
#include "es_ge_test_ops_c.h"
#include "compiler/graph/optimize/symbolic/infer_symbolic_shape/symbolic_shape_inference.h"
#include "compiler/graph/passes/feature/auto_fuse_pass.h"
#include "framework/common/framework_types_internal.h"
#include "faker/space_registry_faker.h"
#include "graph/utils/tensor_adapter.h"
#include "common/env_path.h"
#include "common/topo_checker.h"
#include "utils/autofuse_utils.h"
#include "ge_graph_dsl/graph_dsl.h"
#include "ge_common/ge_common_api_types.h"
#include "graph/compute_graph.h"
#include "common/plugin/ge_make_unique_util.h"
#include "graph/ge_context.h"
#include "graph/ge_local_context.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/op_desc_utils_ex.h"
#include "common/omg_util/omg_util.h"
#include "mmpa/mmpa_api.h"
#include "graph/node.h"
#include "graph/optimize/symbolic/symbolic_kernel_factory.h"
#include "graph/optimize/symbolic/codegen/guard_codegen.h"
#include "attribute_group/attr_group_shape_env.h"
#include "attribute_group/attr_group_symbolic_desc.h"
#include "graph/optimize/symbolic/infer_symbolic_shape/op_impl_infer_symbol_shape.h"
#include "graph/operator_reg.h"

#include <graph/manager/graph_manager.h>
#include <graph/optimize/symbolic/shape_env_guarder.h>
#include "graph/optimize/autofuse/autofuse_optimize.h"
#include "depends/runtime/src/runtime_stub.h"
#include "ge/ge_api.h"
#include "ge_running_env/fake_engine.h"
#include "ge_running_env/ge_running_env_faker.h"
#include "ge_running_env/fake_op.h"
#include "framework/common/taskdown_common.h"
#include "graph/op_kernel_bin.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/operator_factory_impl.h"
#include "faker/fake_value.h"
#include "stub/gert_runtime_stub.h"

namespace ge {

class RuntimeMock : public RuntimeStub {
 public:
  rtError_t rtGetSocSpec(const char *label, const char *key, char *val, const uint32_t maxLen) {
    (void)label;
    (void)key;
    (void)strcpy_s(val, maxLen, "fake");  // fake
    return RT_ERROR_NONE;
  }
};

class SymbolizeValueST : public testing::Test {
 public:
  void SetUp() override {
    RuntimeStub::SetInstance(std::make_shared<RuntimeMock>());
    gert::LoadDefaultSpaceRegistry();
    MM_SYS_GET_ENV(MM_ENV_ASCEND_OPP_PATH, ori_opp_path_env_);
    MM_SYS_GET_ENV(MM_ENV_LD_LIBRARY_PATH, ori_ld_path_env_);
    auto ascend_install_path = EnvPath().GetAscendInstallPath();
    MM_SYS_SET_ENV(MM_ENV_ASCEND_OPP_PATH, (ascend_install_path + "/opp").c_str(), 1, ret_);
    MM_SYS_SET_ENV(MM_ENV_LD_LIBRARY_PATH, (ascend_install_path + "/runtime/lib64").c_str(), 1, ret_);
    mmSetEnv("AUTOFUSE_FLAGS", "--enable_autofuse=true", 1);
    ori_global_options_ = ge::GetThreadLocalContext().GetAllGlobalOptions();
    ori_graph_options_ = ge::GetThreadLocalContext().GetAllGraphOptions();
    ori_session_options_ = ge::GetThreadLocalContext().GetAllSessionOptions();
    ge::GetThreadLocalContext().SetGlobalOption({});
    ge::GetThreadLocalContext().SetGraphOption({});
    ge::GetThreadLocalContext().SetSessionOption({});
    std::map<std::string, std::string> options;
    GetThreadLocalContext().GetOo().Initialize(options, OptionRegistry::GetInstance().GetRegisteredOptTable());
    const auto env_ptr = getenv("LD_PRELOAD");
    if (env_ptr != nullptr) {
      env = env_ptr;
      unsetenv("LD_PRELOAD");
    }
  }
  void TearDown() override {
    RuntimeStub::Reset();
    unsetenv("AUTOFUSE_FLAGS");
    if (ori_ld_path_env_ != nullptr) {
      MM_SYS_SET_ENV(MM_ENV_ASCEND_OPP_PATH, ori_opp_path_env_, 1, ret_);
    } else {
      MM_SYS_UNSET_ENV(MM_ENV_ASCEND_OPP_PATH, ret_);
    }
    if (ori_ld_path_env_ != nullptr) {
      MM_SYS_SET_ENV(MM_ENV_LD_LIBRARY_PATH, ori_ld_path_env_, 1, ret_);
    } else {
      MM_SYS_UNSET_ENV(MM_ENV_LD_LIBRARY_PATH, ret_);
    }
    ge::GetThreadLocalContext().SetGlobalOption(ori_global_options_);
    ge::GetThreadLocalContext().SetGraphOption(ori_graph_options_);
    ge::GetThreadLocalContext().SetSessionOption(ori_session_options_);
    gert::UnLoadDefaultSpaceRegistry();
    if (!env.empty()) {
      setenv("LD_PRELOAD", env.c_str(), 1);
    }
  }

 private:
  int32_t ret_{EN_ERROR};
  const char_t *ori_opp_path_env_{nullptr};
  const char_t *ori_ld_path_env_{nullptr};
  const char_t *enable_auto_fuse_{"1"};
  std::map<std::string, std::string> ori_global_options_;
  std::map<std::string, std::string> ori_graph_options_;
  std::map<std::string, std::string> ori_session_options_;
  std::string env;
};

/*
 *
 *                           data0
 *                             |
 *                            abs
 *                            |
 *             -----------------------------------
 *      data1  |   data2   |   data3  |   data4  |
 *        |   /      |    /      |   /      |   /
 *        | /        |  /        |  /       |  /
 *     repeat1     repeat2      repeat3    repeat4
 *           |      |                 |    |
 *            \     /                  \  /
 *             mul1                    add1
 *               |---------    ----------|
 *                        |    |
 *                       Netoutput
 */
namespace {
REG_OP(Repeat)
    .INPUT(x, TensorType::ALL())
    .INPUT(repeat_times, TensorType::ALL())
    .OUTPUT(y, TensorType::ALL())
    .OP_END_FACTORY_REG(Repeat)

        IMPL_OP(Repeat)
    .InputsDataDependency({1});  // repeat归属自定义二类算子，符号化推导需要获取
graphStatus TestRepeatInferSymbolShapeFunc(gert::InferSymbolShapeContext *context) {
  auto input0 = context->GetInputSymbolShape(0);
  GE_ASSERT_NOTNULL(input0);
  auto input1 = context->GetInputSymbolTensor(1);
  GE_ASSERT_NOTNULL(input1);
  auto symbol_value = input1->GetSymbolicValue();
  if (symbol_value == nullptr) {
    GELOGW("Infer Symbol shape failed, symbol_value is nullptr!");
    return UNSUPPORTED;
  }
  auto output = context->GetOutputSymbolShape(0);
  *output = *input0;
  Expression expr(Symbol(0));
  for (const auto &sym : *symbol_value) {
    expr = expr + sym;
  }
  GE_ASSERT_TRUE(!output->GetDims().empty());
  output->MutableDim(0) = expr;
  return ge::SUCCESS;
}
IMPL_OP_INFER_SYMBOL_SHAPE_INNER(Repeat).InferSymbolShape(TestRepeatInferSymbolShapeFunc);
}  // namespace
TEST_F(SymbolizeValueST, test_symbolize_value_and_repeat_infer) {
  // dlog_setlevel(0, 0, 0);
  auto data0 = OP_CFG("Data")
                   .InCnt(1)
                   .Attr(ATTR_NAME_INDEX, 0)
                   .TensorDesc(FORMAT_ND, DT_FLOAT16, {-1, 2, 3, 4})
                   .OutCnt(1)
                   .OutNames({"y"})
                   .Build("data0");
  auto data1 = OP_CFG("Data")
                   .InCnt(1)
                   .Attr(ATTR_NAME_INDEX, 1)
                   .TensorDesc(FORMAT_ND, DT_INT32, {16})
                   .OutCnt(1)
                   .OutNames({"y"})
                   .Build("data1");
  auto data2 = OP_CFG("Data")
                   .InCnt(1)
                   .Attr(ATTR_NAME_INDEX, 2)
                   .TensorDesc(FORMAT_ND, DT_INT64, {16})
                   .OutCnt(1)
                   .OutNames({"y"})
                   .Build("data2");
  auto data3 = OP_CFG("Data")
                   .InCnt(1)
                   .Attr(ATTR_NAME_INDEX, 3)
                   .TensorDesc(FORMAT_ND, DT_UINT32, {16})
                   .OutCnt(1)
                   .OutNames({"y"})
                   .Build("data3");
  auto data4 = OP_CFG("Data")
                   .InCnt(1)
                   .Attr(ATTR_NAME_INDEX, 4)
                   .TensorDesc(FORMAT_ND, DT_UINT64, {16})
                   .OutCnt(1)
                   .OutNames({"y"})
                   .Build("data4");

  auto repeat1 = OP_CFG("Repeat")
                     .TensorDesc(FORMAT_ND, DT_INT32, {-1, 2, 3, 4})
                     .InCnt(2)
                     .OutCnt(1)
                     .OutNames({"y"})
                     .Build("repeat1");
  auto repeat2 = OP_CFG("Repeat")
                     .TensorDesc(FORMAT_ND, DT_INT64, {-1, 2, 3, 4})
                     .InCnt(2)
                     .OutCnt(1)
                     .OutNames({"y"})
                     .Build("repeat2");
  auto repeat3 = OP_CFG("Repeat")
                     .TensorDesc(FORMAT_ND, DT_UINT32, {-1, 2, 3, 4})
                     .InCnt(2)
                     .OutCnt(1)
                     .OutNames({"y"})
                     .Build("repeat3");
  auto repeat4 = OP_CFG("Repeat")
                     .TensorDesc(FORMAT_ND, DT_UINT64, {-1, 2, 3, 4})
                     .InCnt(2)
                     .OutCnt(1)
                     .OutNames({"y"})
                     .Build("repeat4");
  auto abs =
      OP_CFG("Abs").TensorDesc(FORMAT_ND, DT_FLOAT, {-1, 2, 3, 4}).InCnt(1).OutCnt(1).OutNames({"y"}).Build("abs");
  auto add =
      OP_CFG("Add").TensorDesc(FORMAT_ND, DT_FLOAT, {-1, 2, 3, 4}).InCnt(2).OutCnt(1).OutNames({"y"}).Build("add");
  auto mul =
      OP_CFG("Mul").TensorDesc(FORMAT_ND, DT_FLOAT, {-1, 2, 3, 4}).InCnt(2).OutCnt(1).OutNames({"y"}).Build("mul");

  DEF_GRAPH(g1) {
    CHAIN(NODE(data0)->EDGE(0, 0)->NODE(abs)->EDGE(0, 0)->NODE(repeat1));
    CHAIN(NODE(data0)->EDGE(0, 0)->NODE(abs)->EDGE(0, 0)->NODE(repeat2));
    CHAIN(NODE(data0)->EDGE(0, 0)->NODE(abs)->EDGE(0, 0)->NODE(repeat3));
    CHAIN(NODE(data0)->EDGE(0, 0)->NODE(abs)->EDGE(0, 0)->NODE(repeat4));

    CHAIN(NODE(data1)->EDGE(0, 1)->NODE(repeat1));
    CHAIN(NODE(data2)->EDGE(0, 1)->NODE(repeat2));
    CHAIN(NODE(data3)->EDGE(0, 1)->NODE(repeat3));
    CHAIN(NODE(data4)->EDGE(0, 1)->NODE(repeat4));

    CHAIN(NODE(repeat1)->EDGE(0, 0)->NODE(add));
    CHAIN(NODE(repeat2)->EDGE(0, 1)->NODE(add));
    CHAIN(NODE(repeat3)->EDGE(0, 0)->NODE(mul));
    CHAIN(NODE(repeat4)->EDGE(0, 1)->NODE(mul));

    CHAIN(NODE(add)->EDGE(0, 0)->NODE("NetOutput", NETOUTPUT));
    CHAIN(NODE(mul)->EDGE(0, 1)->NODE("NetOutput", NETOUTPUT));
  };
  auto graph = ToComputeGraph(g1);
  graph->TopologicalSorting();

  GeTensor tensor0(GeTensorDesc(GeShape({16, 2, 3, 4}), FORMAT_ND, DT_FLOAT16));

  GeTensor tensor1(GeTensorDesc(GeShape({16}), FORMAT_ND, DT_INT32));
  vector<int32_t> data_int32(16, 1);
  tensor1.SetData(reinterpret_cast<uint8_t *>(data_int32.data()), data_int32.size() * sizeof(int32_t));

  GeTensor tensor2(GeTensorDesc(GeShape({16}), FORMAT_ND, DT_INT64));
  vector<int64_t> data_int64(16, 1);
  tensor2.SetData(reinterpret_cast<uint8_t *>(data_int64.data()), data_int64.size() * sizeof(int64_t));

  GeTensor tensor3(GeTensorDesc(GeShape({16}), FORMAT_ND, DT_UINT32));
  vector<uint32_t> data_uint32(16, 2);
  tensor3.SetData(reinterpret_cast<uint8_t *>(data_uint32.data()), data_uint32.size() * sizeof(uint32_t));

  GeTensor tensor4(GeTensorDesc(GeShape({16}), FORMAT_ND, DT_UINT64));
  vector<uint64_t> data_uint64(16, 2);
  tensor4.SetData(reinterpret_cast<uint8_t *>(data_uint64.data()), data_uint64.size() * sizeof(uint64_t));

  AutofuseOptimize autofuser;
  ASSERT_EQ(autofuser.Run(graph, {tensor0, tensor1, tensor2, tensor3, tensor4}), ge::GRAPH_SUCCESS);

  auto shape_env = graph->GetAttrsGroup<ShapeEnvAttr>();
  ASSERT_NE(shape_env, nullptr);
  ShapeEnvGuarder guarder(shape_env);  // 此处是为了用例最后校验hint，需要保证有shape_env

  auto repeat1_node = graph->FindNode("repeat1");
  ASSERT_NE(repeat1_node, nullptr);
  auto op_desc1 = repeat1_node->GetOpDesc();
  ASSERT_NE(op_desc1, nullptr);
  auto attr1 = op_desc1->MutableOutputDesc(0)->GetAttrsGroup<SymbolicDescAttr>();
  ASSERT_NE(attr1, nullptr);
  auto symbol_expr1 = attr1->symbolic_tensor.GetOriginSymbolShape().GetDim(0);

  int64_t hint = -1;
  EXPECT_EQ(symbol_expr1.GetHint(hint), true);
  EXPECT_EQ(hint, 16);

  auto netoutput_node = graph->FindNode("NetOutput");
  ASSERT_NE(netoutput_node, nullptr);
  auto netoutput_desc = netoutput_node->GetOpDesc();
  ASSERT_NE(netoutput_desc, nullptr);
  auto attr3 = netoutput_desc->GetInputDesc(1).GetAttrsGroup<SymbolicDescAttr>();
  ASSERT_NE(attr3, nullptr);
  auto symbol_expr3 = attr3->symbolic_tensor.GetOriginSymbolShape().GetDim(0);
  EXPECT_EQ(symbol_expr3.GetHint(hint), true);
  EXPECT_EQ(hint, 16 * 2);
}

// ============ Reshape + hint value helpers ============
ComputeGraphPtr BuildReshapeGraphForTest() {
  auto data0 = OP_CFG("Data")
                   .InCnt(1)
                   .Attr(ATTR_NAME_INDEX, 0)
                   .TensorDesc(FORMAT_ND, DT_FLOAT16, {-1, -1, -1, -1})
                   .OutCnt(1)
                   .OutNames({"y"})
                   .Build("data0");
  auto data1 = OP_CFG("Data")
                   .InCnt(1)
                   .Attr(ATTR_NAME_INDEX, 1)
                   .TensorDesc(FORMAT_ND, DT_INT64, {2})
                   .OutCnt(1)
                   .OutNames({"y"})
                   .Build("data1");
  auto reshape =
      OP_CFG("Reshape").TensorDesc(FORMAT_ND, DT_FLOAT16, {-1, -1}).InCnt(2).OutCnt(1).OutNames({"y"}).Build("reshape");
  DEF_GRAPH(g1) {
    CHAIN(NODE(data0)->EDGE(0, 0)->NODE(reshape)->NODE("NetOutput", "NetOutput"));
    CHAIN(NODE(data1)->EDGE(0, 1)->NODE(reshape));
  };
  auto graph = ToComputeGraph(g1);
  graph->TopologicalSorting();
  for (auto &node : graph->GetAllNodes()) {
    if (node->GetType() == DATA) {
      node->GetOpDesc()->MutableOutputDesc(0)->SetPlacement(kPlacementHost);
    }
  }
  // 设置 Reshape 的 shape 输入为 DT_INT64，与 data1 一致，避免 autofuse 插入 Cast
  auto reshape_node = graph->FindNode("reshape");
  if (reshape_node != nullptr) {
    reshape_node->GetOpDesc()->MutableInputDesc(1)->SetDataType(DT_INT64);
    reshape_node->GetOpDesc()->MutableInputDesc(1)->SetOriginDataType(DT_INT64);
    reshape_node->GetOpDesc()->AppendIrInput("x", ge::kIrInputRequired);
    reshape_node->GetOpDesc()->AppendIrInput("shape", ge::kIrInputRequired);
  }
  return graph;
}

// 空 graph_inputs data + option → Reshape 符号化推导成功
TEST_F(SymbolizeValueST, reshape_symbolize_infer_with_input_hint_value) {
  dlog_setlevel(0, 0, 0);
  auto graph = BuildReshapeGraphForTest();
  ASSERT_NE(graph, nullptr);
  GeTensor tensor0(GeTensorDesc(GeShape({5, 1, 20, 20}), FORMAT_ND, DT_FLOAT16));
  GeTensor tensor1(GeTensorDesc(GeShape({2}), FORMAT_ND, DT_INT64));
  GetThreadLocalContext().SetGraphOption({
      {INPUT_HINT_SHAPE, "0:[5, 1, 20, 20]"},
      {INPUT_HINT_VALUE, "1:[5, 400]"},
  });
  AutofuseOptimize autofuser;
  ASSERT_EQ(autofuser.Run(graph, {tensor0, tensor1}), ge::GRAPH_SUCCESS);

  auto shape_env = graph->GetAttrsGroup<ShapeEnvAttr>();
  ASSERT_NE(shape_env, nullptr);
  ShapeEnvGuarder guarder(shape_env);
  auto reshape_sym = graph->FindNode("reshape")->GetOpDesc()->MutableOutputDesc(0)->GetAttrsGroup<SymbolicDescAttr>();
  ASSERT_NE(reshape_sym, nullptr);
  auto out_shape = reshape_sym->symbolic_tensor.GetOriginSymbolShape();
  ASSERT_EQ(out_shape.GetDimNum(), 2U);
  int64_t hint = -1;
  EXPECT_EQ(out_shape.GetDim(0).GetHint(hint), true);
  EXPECT_EQ(hint, 5);
  hint = -1;
  EXPECT_EQ(out_shape.GetDim(1).GetHint(hint), true);
  EXPECT_EQ(hint, 400);
}

// graph_inputs 有真实 data → 以真实 data 为准
TEST_F(SymbolizeValueST, reshape_symbolize_infer_with_real_data) {
  dlog_setlevel(0, 0, 0);
  auto graph = BuildReshapeGraphForTest();
  ASSERT_NE(graph, nullptr);
  GeTensor tensor0(GeTensorDesc(GeShape({5, 1, 20, 20}), FORMAT_ND, DT_FLOAT16));
  GeTensor tensor1(GeTensorDesc(GeShape({2}), FORMAT_ND, DT_INT64));
  vector<int64_t> shape_val = {100, 20};
  tensor1.SetData(reinterpret_cast<uint8_t *>(shape_val.data()), shape_val.size() * sizeof(int64_t));
  AutofuseOptimize autofuser;
  ASSERT_EQ(autofuser.Run(graph, {tensor0, tensor1}), ge::GRAPH_SUCCESS);

  auto shape_env = graph->GetAttrsGroup<ShapeEnvAttr>();
  ASSERT_NE(shape_env, nullptr);
  ShapeEnvGuarder guarder(shape_env);
  auto reshape_sym = graph->FindNode("reshape")->GetOpDesc()->MutableOutputDesc(0)->GetAttrsGroup<SymbolicDescAttr>();
  ASSERT_NE(reshape_sym, nullptr);
  auto out_shape = reshape_sym->symbolic_tensor.GetOriginSymbolShape();
  ASSERT_EQ(out_shape.GetDimNum(), 2U);
  int64_t hint = -1;
  EXPECT_EQ(out_shape.GetDim(0).GetHint(hint), true);
  EXPECT_EQ(hint, 100);
  hint = -1;
  EXPECT_EQ(out_shape.GetDim(1).GetHint(hint), true);
  EXPECT_EQ(hint, 20);
}

namespace {
class JitFakeAiCoreEngineOptimizer : public FakeGraphOptimizer {
 public:
  Status OptimizeWholeGraph(ComputeGraph &graph) override {
    for (const auto &node : graph.GetAllNodes()) {
      if (node->GetInDataNodes().empty() || node->GetOutDataNodes().empty()) {
        continue;
      }
      std::string input_type = "[";
      for (size_t i = 0U; i < node->GetInDataNodes().size(); i++) {
        input_type += (i == 0U) ? "0" : ", 0";
      }
      input_type += "]";
      // JSON 需包含默认 tiling 注册(IMPL_OP_DEFAULT 的 NormCompileInfo)解析所需的 key
      const std::string compile_info_json =
          std::string("{\"vars\": {\"srcFormat\": \"NCHW\", \"dstFormat\": \"NC1HWC0\", \"dType\": \"float16\", ") +
          "\"ub_size\": 126464, \"block_dim\": 32, \"input_size\": 0, \"hidden_size\": 0, \"group\": 1}, " +
          "\"_input_type\": " + input_type + ", \"_exist_output_after_reduce\": false, " +
          "\"_exist_workspace_after_reduce\": false, \"_available_ub_size\": {\"0\": [126464]}, " +
          "\"_common_info\": [32, 16, 4096], \"_norm_vars\": {\"0\": []}}";
      auto op_desc = node->GetOpDesc();
      AttrUtils::SetStr(op_desc, "compile_info_json", compile_info_json);
      AttrUtils::SetInt(op_desc, "op_para_size", 2048);
      auto bin = std::make_shared<OpKernelBin>("name", std::vector<char>({'F', 'a', 'k', 'e', 'b', 'i', 'n'}));
      op_desc->SetExtAttr(OP_EXTATTR_NAME_TBE_KERNEL, bin);
      AttrUtils::SetStr(op_desc, TVM_ATTR_NAME_MAGIC, "RT_DEV_BINARY_MAGIC_ELF_AIVEC");
      AttrUtils::SetStr(op_desc, TVM_ATTR_NAME_METADATA, "FakeMeta");
      AttrUtils::SetStr(op_desc, node->GetName() + "_kernelname", "FakeKernelName");
      AttrUtils::SetStr(op_desc, ATTR_NAME_KERNEL_BIN_ID, "te_fake_node_123");
      op_desc->SetWorkspaceBytes({20});
    }
    return SUCCESS;
  }
};

class JitFakeAiCoreOpsKernelBuilder : public FakeOpsKernelBuilder {
 public:
  explicit JitFakeAiCoreOpsKernelBuilder(const std::string &engine_name) : FakeOpsKernelBuilder(engine_name) {}
  Status GenerateTask(const Node &node, RunContext &context, std::vector<domi::TaskDef> &tasks) override {
    auto op_desc = node.GetOpDesc();
    op_desc->SetOpKernelLibName("AIcoreEngine");
    EnvPath env_path;
    const std::string autofuse_stub_so_path =
        env_path.GetBinRootPath() + "/tests/depends/op_stub/libslice_autofuse_stub.so";
    (void)ge::AttrUtils::SetStr(op_desc, "bin_file_path", autofuse_stub_so_path);

    size_t arg_size = 100;
    std::vector<uint8_t> args(arg_size, 0);
    domi::TaskDef task_def;
    task_def.set_type(static_cast<uint32_t>(ModelTaskType::MODEL_TASK_KERNEL));
    auto kernel_info = task_def.mutable_kernel();
    kernel_info->set_args(args.data(), args.size());
    kernel_info->set_args_size(arg_size);
    kernel_info->mutable_context()->set_kernel_type(static_cast<uint32_t>(ccKernelType::TE));
    kernel_info->set_block_dim(1);
    uint16_t args_offset[2] = {0};
    kernel_info->mutable_context()->set_args_offset(args_offset, 2 * sizeof(uint16_t));
    kernel_info->mutable_context()->set_op_index(node.GetOpDesc()->GetId());

    auto kernel_with_handle_info = task_def.mutable_kernel_with_handle();
    kernel_with_handle_info->set_args(args.data(), args.size());
    kernel_with_handle_info->set_args_size(arg_size);
    kernel_with_handle_info->mutable_context()->set_kernel_type(static_cast<uint32_t>(ccKernelType::TE));
    kernel_with_handle_info->set_block_dim(1);
    kernel_with_handle_info->mutable_context()->set_args_offset(args_offset, 2 * sizeof(uint16_t));
    kernel_with_handle_info->mutable_context()->set_op_index(node.GetOpDesc()->GetId());
    tasks.emplace_back(task_def);
    return SUCCESS;
  }
};

const auto JitSingleIOForwardInfer = [](Operator &op) {
  auto op_desc = OpDescUtils::GetOpDescFromOperator(op);
  auto in_td = op_desc->GetInputDescPtr(0);
  auto td = op_desc->MutableOutputDesc(0);
  td->SetShape(in_td->GetShape());
  td->SetOriginShape(in_td->GetOriginShape());
  td->SetDataType(in_td->GetDataType());
  td->SetOriginDataType(in_td->GetOriginDataType());
  return GRAPH_SUCCESS;
};

const auto JitUniqueInferFun = [](Operator &op) -> graphStatus {
  auto op_desc = OpDescUtils::GetOpDescFromOperator(op);
  const auto &input_desc = op_desc->GetInputDesc(0);
  const auto input_shape = input_desc.GetShape().GetDims();

  auto output0_desc = op_desc->MutableOutputDesc(0);
  input_desc.GetShape().IsUnknownShape() ? output0_desc->SetShape(GeShape({-1}))
                                         : output0_desc->SetShape(GeShape({16}));
  output0_desc->SetShapeRange({{1, input_shape[0]}});
  return GRAPH_SUCCESS;
};

const auto JitForwardInfer = [](Operator &op) -> graphStatus {
  auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
  *op_desc->MutableOutputDesc(0) = *op_desc->GetInputDescPtr(0);
  return GRAPH_SUCCESS;
};

std::shared_ptr<std::map<std::string, ge::OpCreatorV2>> g_jit_backup_creators_v2;
std::shared_ptr<std::map<std::string, OpCreator>> g_jit_backup_creators;

/*
 *  _arg_0(-1) ──┐
 *               ├─> add ──> unique ──> mul <── const_0
 *  _arg_1(-1) ──┘                       └──────> Node_Output
 * add 的输入0(_arg_0)通过 _op_infer_depends 标记为 value-dependent 输入，
 * unique 输出 shape 不可静态推导，图会被 JIT 切分为 2 个 EP，仅 EP[0] 的 input[0] 为 value-dependent。
 */
ComputeGraphPtr BuildValueDependentSliceGraph() {
  DEF_GRAPH(value_dependent_slice) {
    GeTensorDesc tensor_desc(GeShape({1}), FORMAT_ND, DT_FLOAT);
    GeTensor tensor(tensor_desc);
    int32_t value = 2;
    tensor.SetData((uint8_t *)&value, sizeof(value));

    auto data_0 = OP_CFG(DATA).InCnt(1).OutCnt(1).Attr(ATTR_NAME_INDEX, 0).TensorDesc(FORMAT_ND, DT_FLOAT, {-1});
    auto data_1 = OP_CFG(DATA).InCnt(1).OutCnt(1).Attr(ATTR_NAME_INDEX, 1).TensorDesc(FORMAT_ND, DT_FLOAT, {-1});
    auto add = OP_CFG(ADD).InCnt(2).OutCnt(1).TensorDesc(FORMAT_ND, DT_FLOAT, {-1});
    auto unique_op = OP_CFG("Unique").InCnt(1).OutCnt(1).TensorDesc(FORMAT_ND, DT_FLOAT, {-1});
    auto const_0 = OP_CFG(CONSTANTOP)
                       .OutCnt(1)
                       .Attr(ATTR_NAME_WEIGHTS, tensor)
                       .Attr(ATTR_VARIABLE_PLACEMENT, "host")
                       .TensorDesc(FORMAT_ND, DT_FLOAT, {});
    auto mul = OP_CFG(MUL).InCnt(2).OutCnt(1).TensorDesc(FORMAT_ND, DT_FLOAT, {});
    auto net_output = OP_CFG(NETOUTPUT).InCnt(1).OutCnt(1).TensorDesc(FORMAT_ND, DT_FLOAT, {-1});

    CHAIN(NODE("_arg_0", data_0)->NODE("add", add));
    CHAIN(NODE("_arg_1", data_1)->NODE("add", add));
    CHAIN(NODE("add", add)->NODE("unique", unique_op));
    CHAIN(NODE("unique", unique_op)->NODE("mul", mul));
    CHAIN(NODE("const_0", const_0)->NODE("mul", mul));
    CHAIN(NODE("mul", mul)->NODE("Node_Output", net_output));
  };
  auto compute_graph = ToComputeGraph(value_dependent_slice);
  if (compute_graph == nullptr) {
    return nullptr;
  }
  compute_graph->TopologicalSorting();
  auto add_node = compute_graph->FindNode("add");
  if (add_node != nullptr && add_node->GetOpDesc() != nullptr) {
    // Verify 阶段输入名会对齐 IR 注册名 x1，两个名字都登记以保证 value-dependent 判定命中
    add_node->GetOpDesc()->SetOpInferDepends({"x1", "__input0"});
  }
  return compute_graph;
}

/*
 *  _arg_0(16) ──┐
 *               ├─> add ──> Node_Output
 *  _arg_1(16) ──┘
 * 全静态 shape（整图单 EP，ENABLE_RUNTIME_V2 关闭时走 DavinciModel 静态执行），
 * add 的输入0(_arg_0)同样标记为 value-dependent 输入。
 */
ComputeGraphPtr BuildStaticValueDependentGraph() {
  DEF_GRAPH(static_value_dependent) {
    auto data_0 = OP_CFG(DATA).InCnt(1).OutCnt(1).Attr(ATTR_NAME_INDEX, 0).TensorDesc(FORMAT_ND, DT_FLOAT, {16});
    auto data_1 = OP_CFG(DATA).InCnt(1).OutCnt(1).Attr(ATTR_NAME_INDEX, 1).TensorDesc(FORMAT_ND, DT_FLOAT, {16});
    auto add = OP_CFG(ADD).InCnt(2).OutCnt(1).TensorDesc(FORMAT_ND, DT_FLOAT, {16});
    auto net_output = OP_CFG(NETOUTPUT).InCnt(1).OutCnt(1).TensorDesc(FORMAT_ND, DT_FLOAT, {16});

    CHAIN(NODE("_arg_0", data_0)->NODE("add", add));
    CHAIN(NODE("_arg_1", data_1)->NODE("add", add));
    CHAIN(NODE("add", add)->NODE("Node_Output", net_output));
  };
  auto compute_graph = ToComputeGraph(static_value_dependent);
  if (compute_graph == nullptr) {
    return nullptr;
  }
  compute_graph->TopologicalSorting();
  auto add_node = compute_graph->FindNode("add");
  if (add_node != nullptr && add_node->GetOpDesc() != nullptr) {
    add_node->GetOpDesc()->SetOpInferDepends({"x1", "__input0"});
  }
  return compute_graph;
}
}  // namespace

class JitValueDependentExecuteSTBase : public testing::Test {
 protected:
  void SetUpEnv(bool enable_rt2) {
    g_jit_backup_creators_v2 = ge::OperatorFactoryImpl::operator_creators_v2_;
    g_jit_backup_creators = ge::OperatorFactoryImpl::operator_creators_;
    EXPECT_EQ(GEInitialize(std::map<AscendString, AscendString>{}), SUCCESS);
    gert::LoadDefaultSpaceRegistry();
    gert::SpaceRegistryFaker::UpdateOpImplToDefaultSpaceRegistry();

    auto fe_optimizer = MakeShared<JitFakeAiCoreEngineOptimizer>();
    auto fe_ops_kernel_builder = MakeShared<JitFakeAiCoreOpsKernelBuilder>("AIcoreEngine");
    GeRunningEnvFaker()
        .Reset()
        .Install(FakeEngine("DNN_VM_GE_LOCAL").KernelInfoStore("DNN_VM_GE_LOCAL_OP_STORE"))
        .Install(FakeEngine("AIcoreEngine")
                     .KernelInfoStore("AIcoreEngine")
                     .GraphOptimizer("fe", fe_optimizer)
                     .KernelBuilder(fe_ops_kernel_builder))
        .Install(FakeEngine("DNN_VM_RTS").KernelInfoStore("DNN_VM_RTS_OP_STORE"))
        .Install(FakeOp(IF).InfoStoreAndBuilder("DNN_VM_GE_LOCAL_OP_STORE"))
        .Install(FakeOp(NETOUTPUT).InfoStoreAndBuilder("DNN_VM_GE_LOCAL_OP_STORE"))
        .Install(FakeOp(CONSTANT).InfoStoreAndBuilder("DNN_VM_GE_LOCAL_OP_STORE"))
        .Install(FakeOp(DATA).InfoStoreAndBuilder("DNN_VM_GE_LOCAL_OP_STORE"))
        .Install(FakeOp(PARTITIONEDCALL).InfoStoreAndBuilder("DNN_VM_GE_LOCAL_OP_STORE"))
        .Install(FakeOp(RESHAPE)
                     .Inputs({"x", "shape"})
                     .Outputs({"y"})
                     .AttrsDef("axis", 0)
                     .AttrsDef("num_axes", -1)
                     .InfoStoreAndBuilder("DNN_VM_GE_LOCAL_OP_STORE"))
        .Install(FakeOp("Relu")
                     .Inputs({"x"})
                     .Outputs({"y"})
                     .InfoStoreAndBuilder("AIcoreEngine")
                     .InferShape(JitSingleIOForwardInfer))
        .Install(FakeOp(ADD)
                     .Inputs({"x1", "x2"})
                     .Outputs({"y"})
                     .InfoStoreAndBuilder("AIcoreEngine")
                     .InferShape(JitSingleIOForwardInfer))
        .Install(FakeOp("AscBackend")
                     .Inputs({"x"})
                     .Outputs({"y"})
                     .InfoStoreAndBuilder("AIcoreEngine")
                     .InferShape(JitSingleIOForwardInfer))
        .Install(FakeOp("Add")
                     .Inputs({"x1", "x2"})
                     .Outputs({"y"})
                     .InfoStoreAndBuilder("AIcoreEngine")
                     .InferShape(JitSingleIOForwardInfer))
        .Install(FakeOp("Unique")
                     .Inputs({"x"})
                     .Outputs({"y", "idx"})
                     .InfoStoreAndBuilder("AIcoreEngine")
                     .InferShape(JitUniqueInferFun))
        .Install(FakeOp(MUL).InfoStoreAndBuilder("AIcoreEngine").InferShape(JitForwardInfer))
        .Install(FakeOp(SHAPE).InfoStoreAndBuilder("DNN_VM_GE_LOCAL_OP_STORE"))
        .Install(FakeOp(VARIABLE).InfoStoreAndBuilder("DNN_VM_GE_LOCAL_OP_STORE"))
        .Install(FakeOp(CONSTANTOP).InfoStoreAndBuilder("DNN_VM_GE_LOCAL_OP_STORE"))
        .Install(FakeOp(IDENTITY).InfoStoreAndBuilder("DNN_VM_RTS_OP_STORE"))
        .Install(FakeOp(EXIT).InfoStoreAndBuilder("DNN_VM_RTS_OP_STORE"))
        .Install(FakeOp(RECV).InfoStoreAndBuilder("DNN_VM_RTS_OP_STORE"))
        .Install(FakeOp(SEND).InfoStoreAndBuilder("DNN_VM_RTS_OP_STORE"));

    auto ascend_install_path = EnvPath().GetAscendInstallPath();
    setenv("ASCEND_OPP_PATH", (ascend_install_path + "/opp").c_str(), 1);
    setenv("LD_LIBRARY_PATH", (ascend_install_path + "/runtime/lib64").c_str(), 1);
    auto work_path = EnvPath().GetAirBasePath() + "/output";
    setenv("ASCEND_WORK_PATH", work_path.c_str(), 1);
    mmSetEnv("AUTOFUSE_FLAGS", "--enable_autofuse=true;--experimental_enable_jit_executor_v2=true", 1);
    char runtime2_env[MMPA_MAX_PATH] = {enable_rt2 ? '1' : '0'};
    mmSetEnv("ENABLE_RUNTIME_V2", &(runtime2_env[0U]), static_cast<uint32_t>(MMPA_MAX_PATH));

    runtime_stub_.GetSlogStub().SetLevel(DLOG_DEBUG);
    runtime_stub_.GetSlogStub().Clear();
    runtime_stub_.GetKernelStub().StubTiling();
  }

  void TearDownEnv() {
    char runtime2_env[MMPA_MAX_PATH] = {'0'};
    mmSetEnv("ENABLE_RUNTIME_V2", &(runtime2_env[0U]), static_cast<uint32_t>(MMPA_MAX_PATH));
    unsetenv("ASCEND_OPP_PATH");
    unsetenv("LD_LIBRARY_PATH");
    unsetenv("ASCEND_WORK_PATH");
    unsetenv("AUTOFUSE_FLAGS");

    runtime_stub_.GetSlogStub().Clear();
    GEFinalize();
    GeRunningEnvFaker().InstallDefault();
    gert::UnLoadDefaultSpaceRegistry();
    ge::OperatorFactoryImpl::operator_creators_v2_ = std::move(g_jit_backup_creators_v2);
    ge::OperatorFactoryImpl::operator_creators_ = std::move(g_jit_backup_creators);
  }

  gert::GertRuntimeStub runtime_stub_;
};

class JitValueDependentExecuteST : public JitValueDependentExecuteSTBase {
 protected:
  void SetUp() override {
    SetUpEnv(true);
  }
  void TearDown() override {
    TearDownEnv();
  }
};

/**
 * 用例描述：JIT Execute 路径(gert::Tensor 版 ExecuteGraphWithStreamAsync)下 value-dependent 输入的
 *           编译/执行 placement 一致性守护(RT2 动态图场景)：编译期 BuildCompileInputs 将
 *           value-dependent 输入(input[0])D2H 供 guard/符号化推导(编译假设建立在 host placement 上)，
 *           执行期传入同一份 compile_inputs，由 RT2 执行器按 host placement 消化(alloc+H2D)。
 *           064d09db4 曾误传原始 device inputs，形成"编译假设 host、执行传入 device"的 placement
 *           不一致，真机环境下即 slice_on_core 主进程 coredump 的触发链路。
 * 预置条件：开启自动融合与 JIT 执行器 v2、ENABLE_RUNTIME_V2=1，图含 value-dependent 输入(add 输入0)
 *           且输出 shape 不可静态推导(Unique)被切分为 2 个 EP，仅 EP[0] 的 input[0] 为 value-dependent。
 * 测试步骤：1. 构图并以 kOnDeviceHbm 的 gert::Tensor 输入调用 ExecuteGraphWithStreamAsync；
 *           2. 校验编译期 D2H、GEP 编译、JIT 多 EP 调度链路日志；
 *           3. 校验 value-dependent 输入的 placement 渲染与 device 透传次数。
 * 预期结果：执行返回成功；EP[0] 的 value-dependent input[0] 以 host 进入 RT2
 *           ("placement: kOnHost" 恰 1 次)，仅 EP[1] 的 input[0] 保持 device 透传
 *           ("input[0] address = " 恰 1 次)。若回归为执行传原始 device inputs，
 *           EP[0] 的 value-dependent input[0] 也被 device 透传，两项分别变为 0 次和 2 次，本用例失败。
 *           注：stub 环境 device 指针为合法 host 内存、kernel 为 stub，不会复现 native crash，
 *           以日志断言守护 placement 一致性。
 */
TEST_F(JitValueDependentExecuteST, ExecuteGraphWithStreamAsyncValueDependentInputShouldNotDevicePassthrough) {
  std::map<AscendString, AscendString> options;
  options[OPTION_GRAPH_RUN_MODE] = "1";
  options[VARIABLE_MEMORY_MAX_SIZE] = "12800";
  options[JIT_COMPILE.c_str()] = "1";

  auto compute_graph = BuildValueDependentSliceGraph();
  ASSERT_NE(compute_graph, nullptr);
  Graph graph = GraphUtilsEx::CreateGraphFromComputeGraph(compute_graph);

  Session session(options);
  const uint32_t graph_id = 4321U;
  std::map<AscendString, AscendString> graph_options;
  EXPECT_EQ(session.AddGraph(graph_id, graph, graph_options), SUCCESS);

  std::vector<gert::Tensor> inputs = gert::FakeTensors({16}, 2).Steal();
  std::vector<gert::Tensor> outputs;
  const auto ret = session.ExecuteGraphWithStreamAsync(graph_id, nullptr, inputs, outputs);
  EXPECT_EQ(ret, SUCCESS);

  EXPECT_NE(runtime_stub_.GetSlogStub().FindLog(-1, "BuildCompileInputs:input[0] need copy data to host"), -1);
  EXPECT_NE(runtime_stub_.GetSlogStub().FindLog(-1, "Start to compile GEP"), -1);
  EXPECT_NE(runtime_stub_.GetSlogStub().FindLog(-1, "ExecuteGraphWithStreamAsync GEP[ins_id:"), -1);
  // 正向证据：执行器 DebugString 日志对所有输入打印 placement 枚举名，EP[0] 的 value-dependent input[0]
  // 修复后应渲染为 kOnHost（修复前为 kOnDeviceHbm，出现 0 次）
  EXPECT_EQ(runtime_stub_.GetSlogStub().CountLog(-1, "placement: kOnHost"), 1);
  // 负向证据：device 透传 GELOGD 只在 IsOnDevice 谓词为真的分支打印且内嵌裸枚举值，
  // 修复后仅 EP[1]（无 value-dependent 输入）的 input[0] 保持 device 透传，恰好 1 次
  EXPECT_EQ(runtime_stub_.GetSlogStub().CountLog(-1, "input[0] address = "), 1);

  // outputs 持有 RT2 执行器分配的 MemBlock，依赖 allocator 存活，须在 RemoveGraph 之前释放
  outputs.clear();
  inputs.clear();
  EXPECT_EQ(session.RemoveGraph(graph_id), SUCCESS);
}

class JitValueDependentStaticExecuteST : public JitValueDependentExecuteSTBase {
 protected:
  void SetUp() override {
    SetUpEnv(false);
  }
  void TearDown() override {
    TearDownEnv();
  }
};

/**
 * 用例描述：JIT 静态整图场景(ENABLE_RUNTIME_V2 关闭，走 DavinciModel 静态执行)下，
 *           ge.exec.reuseZeroCopyMemory=1 时，value-dependent 输入经 compile_inputs D2H 后以
 *           host placement 执行；DavinciModel 应识别 Data 节点的 Host Tensor 标记，将该输入纳入
 *           copy_host_input_indexes_，通过 KUpdateHostInput 随路拷贝接纳，而不是进入零拷贝内存不足校验。
 * 预置条件：开启自动融合与 JIT 执行器、ENABLE_RUNTIME_V2=0(静态走 DavinciModel)、
 *           graph options 设 ge.exec.reuseZeroCopyMemory=1，图含 value-dependent 输入且 shape 全静态(整图单 EP)。
 * 测试步骤：1. 构建静态图并以 kOnDeviceHbm 的 gert::Tensor 输入调用 ExecuteGraphWithStreamAsync；
 *           2. 校验执行返回值。
 * 预期结果：执行返回成功；Host value-dependent 输入不触发
 *           "placement is host when ge.exec.reuseZeroCopyMemory=1" 错误。
 */
TEST_F(JitValueDependentStaticExecuteST, StaticGraphReuseZeroCopyHostValueDependentInputShouldExecute) {
  std::map<AscendString, AscendString> options;
  options[OPTION_GRAPH_RUN_MODE] = "1";
  options[VARIABLE_MEMORY_MAX_SIZE] = "12800";
  options[JIT_COMPILE.c_str()] = "1";

  auto compute_graph = BuildStaticValueDependentGraph();
  ASSERT_NE(compute_graph, nullptr);
  Graph graph = GraphUtilsEx::CreateGraphFromComputeGraph(compute_graph);

  Session session(options);
  const uint32_t graph_id = 4322U;
  std::map<AscendString, AscendString> graph_options;
  graph_options.emplace(ge::OPTION_EXEC_REUSE_ZERO_COPY_MEMORY, "1");
  EXPECT_EQ(session.AddGraph(graph_id, graph, graph_options), SUCCESS);

  std::vector<gert::Tensor> inputs = gert::FakeTensors({16}, 2).Steal();
  std::vector<gert::Tensor> outputs;
  const auto ret = session.ExecuteGraphWithStreamAsync(graph_id, nullptr, inputs, outputs);
  EXPECT_EQ(ret, SUCCESS);

  outputs.clear();
  inputs.clear();
  EXPECT_EQ(session.RemoveGraph(graph_id), SUCCESS);
}

}  // namespace ge
