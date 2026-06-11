/* Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * ===================================================================================================================*/

#include <gtest/gtest.h>

#include "macro_utils/dt_public_scope.h"
#include "common/op/ge_op_utils.h"
#include "common/framework_types_internal.h"
#include "graph/anchor.h"
#include "graph/compute_graph.h"
#include "graph/op_desc.h"
#include "graph/passes/base_pass.h"
#include "graph/utils/attr_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/tensor_utils.h"
#include "ge/ut/ge/graph/passes/graph_builder_utils.h"
#include "ge_local_context.h"
#include "graph_utils_ex.h"
#include "graph/passes/standard_optimize/tensor_move_delete_pass.h"
#include "ge_graph_dsl/op_desc/op_desc_cfg_box.h"
#include "easy_graph/builder/graph_dsl.h"
#include "ge_graph_dsl/graph_dsl.h"
#include "graph/operator_reg.h"
#include "graph_metadef/external/ge_common/ge_common_api_types.h"
#include "api/gelib/gelib.h"
#include "ge/ge_api.h"
#include "ge_graph_dsl/assert/graph_assert.h"
#include "exe_graph/runtime/infer_shape_range_context.h"
#include "ge_running_env/ge_running_env_faker.h"
#include "ge_running_env/fake_op.h"
#include "graph/utils/constant_utils.h"
#include "host_kernels/kernel.h"
#include "host_kernels/kernel_factory.h"

using namespace std;
using namespace testing;
using namespace ge;

namespace {
REG_OP(Cast)
.INPUT(x, TensorType({DT_BOOL, DT_FLOAT16, DT_FLOAT, DT_INT8, DT_INT32, DT_UINT32, DT_UINT8, DT_INT64,
                      DT_UINT64, DT_INT16, DT_UINT16, DT_DOUBLE, DT_COMPLEX64, DT_COMPLEX128,
                      DT_QINT8, DT_QUINT8, DT_QINT16, DT_QUINT16, DT_QINT32})) /* input tensor */
.OUTPUT(y, TensorType({DT_BOOL, DT_FLOAT16, DT_FLOAT, DT_INT8, DT_INT32, DT_UINT32, DT_UINT8, DT_INT64,
                       DT_UINT64, DT_INT16, DT_UINT16, DT_DOUBLE, DT_COMPLEX64, DT_COMPLEX128,
                       DT_QINT8, DT_QUINT8, DT_QINT16, DT_QUINT16, DT_QINT32})) /* output tensor */
.ATTR(dst_type, Int, 0)
.ATTR(truncate, Bool, false)
.OP_END_FACTORY_REG(Cast)

REG_OP(TensorMove)
    .INPUT(x, TensorType({DT_DOUBLE, DT_FLOAT16, DT_FLOAT, DT_INT32, DT_UINT32, DT_INT16, DT_UINT16, DT_INT8, DT_UINT8,
                          DT_UINT64, DT_INT64, DT_BOOL, DT_BF16, DT_HIFLOAT8, DT_FLOAT8_E5M2, DT_FLOAT8_E4M3FN, DT_COMPLEX32, DT_COMPLEX64}))
    .OUTPUT(y, TensorType({DT_DOUBLE, DT_FLOAT16, DT_FLOAT, DT_INT32, DT_UINT32, DT_INT16, DT_UINT16, DT_INT8, DT_UINT8,
                           DT_UINT64, DT_INT64, DT_BOOL, DT_BF16, DT_HIFLOAT8, DT_FLOAT8_E5M2, DT_FLOAT8_E4M3FN, DT_COMPLEX32, DT_COMPLEX64}))
    .OP_END_FACTORY_REG(TensorMove)

bool SetTransDataTensorDesc(const ComputeGraphPtr &root_graph, const std::vector<std::string> &node_names, Format format = FORMAT_NCL) {
  GeTensorDesc tensor_desc{GeShape{{2022, 2023}}, format, DT_FLOAT16};
  std::map<std::string, NodePtr> all_transdata_map;
  for (auto &node : root_graph->GetAllNodes()) {
    if (node->GetType() == TRANSDATA) {
      all_transdata_map[node->GetName()] = node;
    }
  }
  for (const auto &node_name : node_names) {
    const auto iter = all_transdata_map.find(node_name);
    if (iter != all_transdata_map.end()) {
      iter->second->GetOpDesc()->UpdateOutputDesc(0, tensor_desc);
    } else {
      std::cout << "========================================" << std::endl;
      std::cout << "cannot find " << node_name << std::endl;
      std::cout << "========================================" << std::endl;
      return false;
    }
  }
  return true;
}

using NetoutputParentIndexes = std::vector<std::pair<std::string, std::vector<uint32_t>>>;
bool AddParentIndexForNetoutput(ComputeGraphPtr &root_graph, NetoutputParentIndexes &indexes) {
  std::map<std::string, NodePtr> netoutput_map;
  for (auto &node : root_graph->GetAllNodes()) {
    netoutput_map[node->GetName()] = node;
  }
  for (auto &name_indexes_pair : indexes) {
    const auto iter = netoutput_map.find(name_indexes_pair.first);
    if (iter == netoutput_map.end()) {
      std::cout << "========================================" << std::endl;
      std::cout << "cannot find " << name_indexes_pair.first << std::endl;
      std::cout << "========================================" << std::endl;
      return false;
    }
    auto op_desc = iter->second->GetOpDesc();
    size_t input_index = 0U;
    if (name_indexes_pair.second.size() != op_desc->GetInputsSize()) {
      std::cout << "========================================" << std::endl;
      std::cout << name_indexes_pair.first << " real inputs size: " << op_desc->GetInputsSize()
                << ", but name_indexes_pair.second.size(): " << name_indexes_pair.second.size() << std::endl;
      std::cout << "========================================" << std::endl;
      return false;
    }
    for (auto parent_index : name_indexes_pair.second) {
      auto tensor_desc = op_desc->MutableInputDesc(input_index++);
      AttrUtils::SetInt(tensor_desc, ATTR_NAME_PARENT_NODE_INDEX, parent_index);
    }
  }
  return true;
}

void SetInplaceOutput(const NodePtr &node, const uint32_t output_idx = 0U, const int32_t input_idx = 0) {
  auto out_desc = node->GetOpDescBarePtr()->MutableOutputDesc(output_idx);
  AttrUtils::SetInt(out_desc, INPLACE_SUPPORT_INPUT_INDEX, input_idx);
}

size_t CountNodesByType(const ComputeGraphPtr &graph, const std::string &type) {
  size_t count = 0U;
  for (const auto &node : graph->GetAllNodes()) {
    if (node->GetType() == type) {
      ++count;
    }
  }
  return count;
}

void SetMlaDumpReuseOptions() {
  std::map<std::string, std::string> options;
  options[OPTION_OUTPUT_REUSE_INPUT_MEM_INDEXES] = "0,0|1,1";
  options["ge.oo.level"] = "O3";
  ge::GetThreadLocalContext().SetGraphOption(options);
}

Status RunTensorMoveDeletePass(const ComputeGraphPtr &graph) {
  graph->TopologicalSorting();
  ge::GEPass pass(graph);
  TensorMoveDeletePass tensor_move_delete_pass;
  ge::NamesToPass names_to_pass;
  names_to_pass.emplace_back("TensorMoveDeletePass", &tensor_move_delete_pass);
  return pass.Run(names_to_pass);
}

NodePtr AddTestNode(const ComputeGraphPtr &graph, const std::string &name, const std::string &type, int in_cnt,
                    int out_cnt, Format format = FORMAT_NCHW, DataType data_type = DT_FLOAT,
                    const std::vector<int64_t> &shape = {1, 1, 224, 224}) {
  auto tensor_desc = std::make_shared<GeTensorDesc>();
  tensor_desc->SetShape(GeShape(shape));
  tensor_desc->SetFormat(format);
  tensor_desc->SetDataType(data_type);

  auto op_desc = std::make_shared<OpDesc>(name, type);
  for (int i = 0; i < in_cnt; ++i) {
    op_desc->AddInputDesc(tensor_desc->Clone());
  }
  for (int i = 0; i < out_cnt; ++i) {
    op_desc->AddOutputDesc(tensor_desc->Clone());
  }
  op_desc->AddInferFunc([](Operator &op) { return GRAPH_SUCCESS; });
  return graph->AddNode(op_desc);
}

void SetWeightForConstNode(NodePtr &const_node) {
  // new a tensor
  ge::GeTensorPtr tensor = std::make_shared<GeTensor>();
  std::vector<uint8_t> value{1, 2, 3, 4, 5, 6, 7, 8, 9};
  std::vector<int64_t> shape{9};
  tensor->MutableTensorDesc().SetShape(GeShape(shape));
  tensor->SetData(value);
  tensor->MutableTensorDesc().SetDataType(DT_UINT8);
  ConstantUtils::SetWeight(const_node->GetOpDesc(), 0, tensor);
}

const char *AddNYes = "AddNYes";
const char *ShapeNo = "ShapeNo";
class TestAddNKernel : public Kernel {
public:
  Status Compute(const ge::OpDescPtr op_desc_ptr, const std::vector<ge::ConstGeTensorPtr> &input,
                 std::vector<ge::GeTensorPtr> &v_output) override {
    auto output = std::make_shared<GeTensor>();
    std::vector<uint8_t> data{1, 2, 3};
    std::vector<int64_t> shape{3};
    output->MutableTensorDesc().SetShape(GeShape(shape));
    output->SetData(data);
    output->MutableTensorDesc().SetDataType(DT_UINT8);
    v_output.push_back(output);
    return SUCCESS;
  }
};

REGISTER_COMPUTE_NODE_KERNEL(AddNYes, TestAddNKernel);
}

class TensorMoveTest : public Test {
  protected:
  void SetUp() {
    dlog_setlevel(0, 0, 0);
    std::map<std::string, std::string> options = {{"ge.oo.level", "O3"}};
    GetThreadLocalContext().SetGraphOption(options);
    GetThreadLocalContext().GetOo().Initialize({}, OptionRegistry::GetInstance().GetRegisteredOptTable());

    GeRunningEnvFaker().Reset().InstallDefault()
        .Install(FakeOp(AddNYes).InfoStoreAndBuilder("DNN_VM_GE_LOCAL_OP_STORE"))
        .Install(FakeOp(ShapeNo).InfoStoreAndBuilder("DNN_VM_GE_LOCAL_OP_STORE"))
        .Install(FakeEngine("DNN_VM_AICPU_ASCEND").KernelInfoStore("aicpu_ascend_kernel"));
  }

  void TearDown() {
    dlog_setlevel(0, 3, 0);
    unsetenv("DUMP_GRAPH_LEVEL");
    unsetenv("DUMP_GE_GRAPH");
    GetThreadLocalContext().SetGraphOption({});
  }
};

/**
 * 父图:                子图 sub_1:
 * Data                 sub_Data (ParentIndex: 0)
 * |                      |
 * PartitionedCall ------> TensorMove
 * |                      |
 * NetOutput            sub_NetOutput
 * (复用输入地址)
 *
 * 场景说明：
 * - 子图内部 TensorMove 的前驱是 sub_Data，其在父图的实际源头是 Data。
 * - 设置根图 NetOutput 复用输入内存，触发 TensorMove 优化逻辑。
 *
 * 预期行为：
 * - Trace 能够跨越子图边界识别到 Data 是源头。
 * - TensorMove 被成功识别并删除。
 */
TEST_F(TensorMoveTest, TensorMoveInSubgraph_FromParentData_Deleted) {
  dlog_setlevel(0, 0, 0);

  // 1. 设置内存复用选项：设置根图的第 0 个输出复用第 0 个输入
  std::map<std::string, std::string> options;
  options[OPTION_OUTPUT_REUSE_INPUT_MEM_INDEXES] = "0,0";
  options["ge.oo.level"] = "O3";
  ge::GetThreadLocalContext().SetGraphOption(options);

  // 2. 构造子图 sub_1
  // sub_Data 的 ParentNodeIndex(0) 代表它对应父图中 PartitionedCall 的第 0 个 Input
  const auto sub_data_cfg = OP_CFG(DATA).ParentNodeIndex(0);
  DEF_GRAPH(sub_1) {
    CHAIN(NODE("sub_data", sub_data_cfg)
          ->NODE("sub_tensormove", TENSORMOVE)
          ->NODE("sub_netoutput", NETOUTPUT));
  };

  // 3. 构造父图 g1
  const auto data_cfg = OP_CFG(DATA).Attr(ATTR_NAME_INDEX, 0);
  DEF_GRAPH(g1) {
    CHAIN(NODE("data", data_cfg)
          ->EDGE(0, 0)->NODE("partitioned_call", PARTITIONEDCALL, sub_1)
          ->EDGE(0, 0)->NODE("netoutput", NETOUTPUT));
  };

  // 4. 将子图挂载到父图
  auto compute_graph = ToComputeGraph(g1);
  const auto sub_graph_1 = compute_graph->GetSubgraph("sub_1");
  ASSERT_NE(sub_graph_1, nullptr);

  auto p_call_node = compute_graph->FindNode("partitioned_call");
  ASSERT_NE(p_call_node, nullptr);

  // 设置父子图关联属性
  sub_graph_1->SetParentGraph(compute_graph);
  sub_graph_1->SetParentNode(p_call_node);
  NetoutputParentIndexes indexes{{"netoutput", {0}},
                                 {"sub_netoutput", {0}}};
  ASSERT_TRUE(AddParentIndexForNetoutput(compute_graph, indexes));

  // 6. 执行 Pass
  ge::GEPass pass(compute_graph);
  TensorMoveDeletePass tensor_move_delete_pass;
  ge::NamesToPass names_to_pass;
  names_to_pass.emplace_back("TensorMoveDeletePass", &tensor_move_delete_pass);
  EXPECT_EQ(pass.Run(names_to_pass), SUCCESS);

  // 7. 验证结果：子图内部的 tensormove 应该被删除
  // 注意：FindNode 在子图中查找
  EXPECT_EQ(sub_graph_1->FindNode("sub_tensormove"), nullptr);

  // 清理环境
  ge::GetThreadLocalContext().SetGraphOption({});
}

/**
 *         Relu
 *        /    \
 *      Add   TensorMove
 *       |       |
 *       +---- NetOutput
 *
 * 说明：
 * - 基本单输出多引用场景
 * - Add 只读取 Relu 输出，不透传也不 inplace
 * - TensorMove 后继 NetOutput 也是纯读
 *
 * 预期：
 * - 删除 TensorMove
 * - 旁路与后继均为纯读，二者间无读写冒险，不新增 Add 到 NetOutput 的控制边
 */
TEST_F(TensorMoveTest, TensorMove_BasicMultiRefBranch_DeletedWithoutControlEdge) {
  auto graph = std::make_shared<ComputeGraph>("g1");
  auto relu_node = AddTestNode(graph, "Relu", RELU, 1, 1);
  auto add_node = AddTestNode(graph, "Add", ADD, 1, 1);
  auto tensor_move_node = AddTestNode(graph, "TensorMove", TENSORMOVE, 1, 1);
  auto netoutput_node = AddTestNode(graph, "NetOutput", NETOUTPUT, 2, 1);

  GraphUtils::AddEdge(relu_node->GetOutDataAnchor(0), add_node->GetInDataAnchor(0));
  GraphUtils::AddEdge(add_node->GetOutDataAnchor(0), netoutput_node->GetInDataAnchor(1));
  GraphUtils::AddEdge(relu_node->GetOutDataAnchor(0), tensor_move_node->GetInDataAnchor(0));
  GraphUtils::AddEdge(tensor_move_node->GetOutDataAnchor(0), netoutput_node->GetInDataAnchor(0));

  graph->TopologicalSorting();
  ge::GEPass pass(graph);
  TensorMoveDeletePass tensor_move_delete_pass;
  ge::NamesToPass names_to_pass;
  names_to_pass.emplace_back("TensorMoveDeletePass", &tensor_move_delete_pass);
  EXPECT_EQ(pass.Run(names_to_pass), SUCCESS);

  EXPECT_EQ(graph->FindNode("TensorMove"), nullptr);
  EXPECT_TRUE(relu_node->GetOutDataAnchor(0)->IsLinkedWith(netoutput_node->GetInDataAnchor(0)));
  EXPECT_FALSE(add_node->GetOutControlAnchor()->IsLinkedWith(netoutput_node->GetInControlAnchor()));
}

/**
 *           Relu
 *      /      |      \
 *   Add0     Add1   TensorMove
 *     |        |         |
 *     +--------+----- NetOutput
 *
 * 说明：
 * - Relu 单输出被多个普通读分支和 TensorMove 同时引用
 * - TensorMove 后继 NetOutput 也是纯读
 *
 * 预期：
 * - 删除 TensorMove
 * - 旁路与后继均为纯读，不向 NetOutput 补控制边
 */
TEST_F(TensorMoveTest, TensorMove_MultipleBasicMultiRefBranches_DeletedWithoutControlEdges) {
  auto graph = std::make_shared<ComputeGraph>("g1");
  auto relu_node = AddTestNode(graph, "Relu", RELU, 1, 1);
  auto add0_node = AddTestNode(graph, "Add0", ADD, 1, 1);
  auto add1_node = AddTestNode(graph, "Add1", ADD, 1, 1);
  auto tensor_move_node = AddTestNode(graph, "TensorMove", TENSORMOVE, 1, 1);
  auto netoutput_node = AddTestNode(graph, "NetOutput", NETOUTPUT, 3, 1);

  GraphUtils::AddEdge(relu_node->GetOutDataAnchor(0), add0_node->GetInDataAnchor(0));
  GraphUtils::AddEdge(add0_node->GetOutDataAnchor(0), netoutput_node->GetInDataAnchor(1));
  GraphUtils::AddEdge(relu_node->GetOutDataAnchor(0), add1_node->GetInDataAnchor(0));
  GraphUtils::AddEdge(add1_node->GetOutDataAnchor(0), netoutput_node->GetInDataAnchor(2));
  GraphUtils::AddEdge(relu_node->GetOutDataAnchor(0), tensor_move_node->GetInDataAnchor(0));
  GraphUtils::AddEdge(tensor_move_node->GetOutDataAnchor(0), netoutput_node->GetInDataAnchor(0));

  graph->TopologicalSorting();
  ge::GEPass pass(graph);
  TensorMoveDeletePass tensor_move_delete_pass;
  ge::NamesToPass names_to_pass;
  names_to_pass.emplace_back("TensorMoveDeletePass", &tensor_move_delete_pass);
  EXPECT_EQ(pass.Run(names_to_pass), SUCCESS);

  EXPECT_EQ(graph->FindNode("TensorMove"), nullptr);
  EXPECT_TRUE(relu_node->GetOutDataAnchor(0)->IsLinkedWith(netoutput_node->GetInDataAnchor(0)));
  EXPECT_FALSE(add0_node->GetOutControlAnchor()->IsLinkedWith(netoutput_node->GetInControlAnchor()));
  EXPECT_FALSE(add1_node->GetOutControlAnchor()->IsLinkedWith(netoutput_node->GetInControlAnchor()));
}

/**
 *        Data
 *         |
 *  TensorMove0(保留)
 *         |
 *  ScatterNDUpdate0
 *      /         \
 * MlpLightningIndexer0 TensorMove1
 *         |             |
 * SparseFlashAttention0 ScatterNDUpdate1
 *         |             |
 *         +------ NetOutput ---- MlpLightningIndexer1
 *
 * 说明：
 * - 待删节点是 TensorMove1
 * - 分叉点位于 TensorMove1 的直接源节点 ScatterNDUpdate0，不是根 Data
 * - 上游 TensorMove0 通过保留属性固定不删除
 * - TensorMove1 后继 ScatterNDUpdate1 通过 INPLACE_SUPPORT_INPUT_INDEX 覆写源内存
 *
 * 预期：
 * - 删除 TensorMove1
 * - ScatterNDUpdate0 直连 ScatterNDUpdate1
 * - 旁路 MlpLightningIndexer0 纯读、后继 ScatterNDUpdate1 覆写，新增 MlpLightningIndexer0 到 ScatterNDUpdate1 控制边
 * - TensorMove0 保留
 */
TEST_F(TensorMoveTest, TensorMove_FromIntermediateSource_WithScatterBranch_DeletedAndAddControlEdge) {
  auto graph = std::make_shared<ComputeGraph>("g1");
  auto data_node = AddTestNode(graph, "Data", DATA, 1, 1);
  auto tensor_move0_node = AddTestNode(graph, "TensorMove0", TENSORMOVE, 1, 1);
  auto scatter0_node = AddTestNode(graph, "ScatterNDUpdate0", RELU, 1, 1);
  auto mlp_indexer0_node = AddTestNode(graph, "MlpLightningIndexer0", RELU, 1, 1);
  auto sfa0_node = AddTestNode(graph, "SparseFlashAttention0", RELU, 1, 1);
  auto tensor_move1_node = AddTestNode(graph, "TensorMove1", TENSORMOVE, 1, 1);
  auto scatter1_node = AddTestNode(graph, "ScatterNDUpdate1", RELU, 1, 1);
  auto mlp_indexer1_node = AddTestNode(graph, "MlpLightningIndexer1", RELU, 1, 1);
  auto netoutput_node = AddTestNode(graph, "NetOutput", NETOUTPUT, 3, 1);

  AttrUtils::SetBool(tensor_move0_node->GetOpDesc(), ATTR_NAME_CANNOT_BE_DELETED, true);
  SetInplaceOutput(scatter1_node, 0, 0);

  GraphUtils::AddEdge(data_node->GetOutDataAnchor(0), tensor_move0_node->GetInDataAnchor(0));
  GraphUtils::AddEdge(tensor_move0_node->GetOutDataAnchor(0), scatter0_node->GetInDataAnchor(0));
  GraphUtils::AddEdge(scatter0_node->GetOutDataAnchor(0), mlp_indexer0_node->GetInDataAnchor(0));
  GraphUtils::AddEdge(mlp_indexer0_node->GetOutDataAnchor(0), sfa0_node->GetInDataAnchor(0));
  GraphUtils::AddEdge(scatter0_node->GetOutDataAnchor(0), tensor_move1_node->GetInDataAnchor(0));
  GraphUtils::AddEdge(tensor_move1_node->GetOutDataAnchor(0), scatter1_node->GetInDataAnchor(0));
  GraphUtils::AddEdge(scatter1_node->GetOutDataAnchor(0), netoutput_node->GetInDataAnchor(0));
  GraphUtils::AddEdge(sfa0_node->GetOutDataAnchor(0), netoutput_node->GetInDataAnchor(1));
  GraphUtils::AddEdge(scatter1_node->GetOutDataAnchor(0), mlp_indexer1_node->GetInDataAnchor(0));
  GraphUtils::AddEdge(mlp_indexer1_node->GetOutDataAnchor(0), netoutput_node->GetInDataAnchor(2));

  graph->TopologicalSorting();
  ge::GEPass pass(graph);
  TensorMoveDeletePass tensor_move_delete_pass;
  ge::NamesToPass names_to_pass;
  names_to_pass.emplace_back("TensorMoveDeletePass", &tensor_move_delete_pass);
  EXPECT_EQ(pass.Run(names_to_pass), SUCCESS);

  EXPECT_NE(graph->FindNode("TensorMove0"), nullptr);
  EXPECT_EQ(graph->FindNode("TensorMove1"), nullptr);
  EXPECT_TRUE(scatter0_node->GetOutDataAnchor(0)->IsLinkedWith(scatter1_node->GetInDataAnchor(0)));
  EXPECT_TRUE(mlp_indexer0_node->GetOutControlAnchor()->IsLinkedWith(scatter1_node->GetInControlAnchor()));
}

/**
 *        Relu
 *       /    \
 *  Sibling  TensorMove
 *     ^         |
 *     |       Succ(inplace)
 *     +---------+
 *
 * 说明：
 * - Sibling 纯读 Relu 输出
 * - Succ 通过 INPLACE_SUPPORT_INPUT_INDEX 覆写源内存
 * - Succ 已经通过数据边到达 Sibling，若补 Sibling -> Succ 控制边会成环
 *
 * 预期：
 * - 保留 TensorMove
 * - 不添加 Sibling 到 Succ 的控制边
 */
TEST_F(TensorMoveTest, TensorMove_SuccessorReachesSiblingViaDataPath_Kept) {
  auto graph = std::make_shared<ComputeGraph>("g1");
  auto relu_node = AddTestNode(graph, "Relu", RELU, 1, 1);
  auto sibling_node = AddTestNode(graph, "Sibling", ADD, 2, 1);
  auto tensor_move_node = AddTestNode(graph, "TensorMove", TENSORMOVE, 1, 1);
  auto succ_node = AddTestNode(graph, "Succ", ADD, 1, 1);
  auto netoutput_node = AddTestNode(graph, "NetOutput", NETOUTPUT, 1, 1);

  SetInplaceOutput(succ_node, 0, 0);

  GraphUtils::AddEdge(relu_node->GetOutDataAnchor(0), sibling_node->GetInDataAnchor(0));
  GraphUtils::AddEdge(relu_node->GetOutDataAnchor(0), tensor_move_node->GetInDataAnchor(0));
  GraphUtils::AddEdge(tensor_move_node->GetOutDataAnchor(0), succ_node->GetInDataAnchor(0));
  GraphUtils::AddEdge(succ_node->GetOutDataAnchor(0), sibling_node->GetInDataAnchor(1));
  GraphUtils::AddEdge(sibling_node->GetOutDataAnchor(0), netoutput_node->GetInDataAnchor(0));

  EXPECT_EQ(RunTensorMoveDeletePass(graph), SUCCESS);

  EXPECT_NE(graph->FindNode("TensorMove"), nullptr);
  EXPECT_TRUE(tensor_move_node->GetOutDataAnchor(0)->IsLinkedWith(succ_node->GetInDataAnchor(0)));
  EXPECT_FALSE(sibling_node->GetOutControlAnchor()->IsLinkedWith(succ_node->GetInControlAnchor()));
}

/**
 *         Relu
 *        /    \
 *   Add(inplace) TensorMove
 *        |          |
 *        +------ NetOutput
 *
 * 说明：
 * - 旁路分支会原地写回源 buffer
 *
 * 预期：
 * - 保留 TensorMove
 */
TEST_F(TensorMoveTest, TensorMove_BasicMultiRefWithInplaceBranch_Kept) {
  auto graph = std::make_shared<ComputeGraph>("g1");
  auto relu_node = AddTestNode(graph, "Relu", RELU, 1, 1);
  auto add_node = AddTestNode(graph, "Add", ADD, 1, 1);
  auto tensor_move_node = AddTestNode(graph, "TensorMove", TENSORMOVE, 1, 1);
  auto netoutput_node = AddTestNode(graph, "NetOutput", NETOUTPUT, 2, 1);

  SetInplaceOutput(add_node, 0, 0);

  GraphUtils::AddEdge(relu_node->GetOutDataAnchor(0), add_node->GetInDataAnchor(0));
  GraphUtils::AddEdge(add_node->GetOutDataAnchor(0), netoutput_node->GetInDataAnchor(1));
  GraphUtils::AddEdge(relu_node->GetOutDataAnchor(0), tensor_move_node->GetInDataAnchor(0));
  GraphUtils::AddEdge(tensor_move_node->GetOutDataAnchor(0), netoutput_node->GetInDataAnchor(0));

  graph->TopologicalSorting();
  ge::GEPass pass(graph);
  TensorMoveDeletePass tensor_move_delete_pass;
  ge::NamesToPass names_to_pass;
  names_to_pass.emplace_back("TensorMoveDeletePass", &tensor_move_delete_pass);
  EXPECT_EQ(pass.Run(names_to_pass), SUCCESS);

  EXPECT_NE(graph->FindNode("TensorMove"), nullptr);
  EXPECT_FALSE(add_node->GetOutControlAnchor()->IsLinkedWith(netoutput_node->GetInControlAnchor()));
}

/**
 * 对应 bbit Test4/5/6 的 46/45/331 dump（MLA KV Cache 双分支主模式）。
 *
 * 备注：本用例验证通用 TensorMoveDeletePass 对 MLA 拓扑的兜底删除行为；
 *       MLA 专用 pattern 识别由 UT 侧覆盖。
 *
 * 构图（pass 前）：
 *
 *   arg11_1 ──► TensorMove    ──┐
 *                               ├──► MlaPrologV3[9]/[10]
 *   arg12_1 ──► TensorMove_1  ──┘          │
 *                                          ├─:2─► Reshape_60 ─► Reshape_61 ─► Reshape_62 ──┬─► TensorMove_2    ─► ScatterNdUpdate[0]
 *                                          │                                               └─► IndexByTensor_2 ─► ScatterNdUpdate[2]
 *                                          │
 *                                          └─:3─► Squeeze_19 ─► Reshape_63 ─┬─► TensorMove_3    ─► ScatterNdUpdate_1[0]
 *                                                                           └─► IndexByTensor_3 ─► ScatterNdUpdate_1[2]
 *
 *   arg19_1 ──► IndexByTensor_2[1]、IndexByTensor_3[1]
 *   arg25_1 ──► ScatterNdUpdate[1]、ScatterNdUpdate_1[1]
 *
 *   ScatterNdUpdate   ──► NetOutput[0]
 *   ScatterNdUpdate_1 ──► NetOutput[1]
 *
 * Pass 后预期：
 *   - 4 个 TensorMove 全部删除
 *   - arg11_1/arg12_1 直连 MlaPrologV3[9]/[10]
 *   - Reshape_62/Reshape_63 直连 ScatterNdUpdate/_1[0]
 *   - 多引用分支序列化：IndexByTensor_2 ─ctrl─► ScatterNdUpdate
 *                       IndexByTensor_3 ─ctrl─► ScatterNdUpdate_1
 */
TEST_F(TensorMoveTest, TensorMove_MlaDump46ThreeReshapeAndSqueezeBranches_Deleted) {
  SetMlaDumpReuseOptions();
  auto graph = std::make_shared<ComputeGraph>("g1");

  auto kv_cache = AddTestNode(graph, "arg11_1", DATA, 0, 1);
  auto kr_cache = AddTestNode(graph, "arg12_1", DATA, 0, 1);
  auto indices = AddTestNode(graph, "arg19_1", DATA, 0, 1);
  auto update_indices = AddTestNode(graph, "arg25_1", DATA, 0, 1);
  auto tensor_move_kv = AddTestNode(graph, "TensorMove", TENSORMOVE, 1, 1);
  auto tensor_move_kr = AddTestNode(graph, "TensorMove_1", TENSORMOVE, 1, 1);
  auto mla = AddTestNode(graph, "MlaPrologV3", "MlaPrologV3", 21, 7);
  auto reshape0 = AddTestNode(graph, "Reshape_60", RESHAPE, 2, 1);
  auto reshape1 = AddTestNode(graph, "Reshape_61", RESHAPE, 2, 1);
  auto reshape2 = AddTestNode(graph, "Reshape_62", RESHAPE, 2, 1);
  auto index_by_tensor0 = AddTestNode(graph, "IndexByTensor_2", "IndexByTensor", 2, 1);
  auto tensor_move0 = AddTestNode(graph, "TensorMove_2", TENSORMOVE, 1, 1);
  auto scatter0 = AddTestNode(graph, "ScatterNdUpdate", "ScatterNdUpdate", 3, 1);
  auto squeeze = AddTestNode(graph, "Squeeze_19", SQUEEZE, 1, 1);
  auto reshape3 = AddTestNode(graph, "Reshape_63", RESHAPE, 2, 1);
  auto index_by_tensor1 = AddTestNode(graph, "IndexByTensor_3", "IndexByTensor", 2, 1);
  auto tensor_move1 = AddTestNode(graph, "TensorMove_3", TENSORMOVE, 1, 1);
  auto scatter1 = AddTestNode(graph, "ScatterNdUpdate_1", "ScatterNdUpdate", 3, 1);
  auto netoutput = AddTestNode(graph, "NetOutput", NETOUTPUT, 2, 1);

  AttrUtils::SetInt(kv_cache->GetOpDesc(), ATTR_NAME_INDEX, 0);
  AttrUtils::SetInt(kr_cache->GetOpDesc(), ATTR_NAME_INDEX, 1);

  GraphUtils::AddEdge(kv_cache->GetOutDataAnchor(0), tensor_move_kv->GetInDataAnchor(0));
  GraphUtils::AddEdge(tensor_move_kv->GetOutDataAnchor(0), mla->GetInDataAnchor(9));
  GraphUtils::AddEdge(kr_cache->GetOutDataAnchor(0), tensor_move_kr->GetInDataAnchor(0));
  GraphUtils::AddEdge(tensor_move_kr->GetOutDataAnchor(0), mla->GetInDataAnchor(10));

  GraphUtils::AddEdge(mla->GetOutDataAnchor(2), reshape0->GetInDataAnchor(0));
  GraphUtils::AddEdge(reshape0->GetOutDataAnchor(0), reshape1->GetInDataAnchor(0));
  GraphUtils::AddEdge(reshape1->GetOutDataAnchor(0), reshape2->GetInDataAnchor(0));
  // ScatterNdUpdate 原地写回 input0 对应的源内存，旁路 IndexByTensor 纯读，需补 reader-before-writer 控制边
  SetInplaceOutput(scatter0, 0, 0);
  SetInplaceOutput(scatter1, 0, 0);

  GraphUtils::AddEdge(reshape2->GetOutDataAnchor(0), index_by_tensor0->GetInDataAnchor(0));
  GraphUtils::AddEdge(indices->GetOutDataAnchor(0), index_by_tensor0->GetInDataAnchor(1));
  GraphUtils::AddEdge(index_by_tensor0->GetOutDataAnchor(0), scatter0->GetInDataAnchor(2));
  GraphUtils::AddEdge(reshape2->GetOutDataAnchor(0), tensor_move0->GetInDataAnchor(0));
  GraphUtils::AddEdge(tensor_move0->GetOutDataAnchor(0), scatter0->GetInDataAnchor(0));
  GraphUtils::AddEdge(update_indices->GetOutDataAnchor(0), scatter0->GetInDataAnchor(1));

  GraphUtils::AddEdge(mla->GetOutDataAnchor(3), squeeze->GetInDataAnchor(0));
  GraphUtils::AddEdge(squeeze->GetOutDataAnchor(0), reshape3->GetInDataAnchor(0));
  GraphUtils::AddEdge(reshape3->GetOutDataAnchor(0), index_by_tensor1->GetInDataAnchor(0));
  GraphUtils::AddEdge(indices->GetOutDataAnchor(0), index_by_tensor1->GetInDataAnchor(1));
  GraphUtils::AddEdge(index_by_tensor1->GetOutDataAnchor(0), scatter1->GetInDataAnchor(2));
  GraphUtils::AddEdge(reshape3->GetOutDataAnchor(0), tensor_move1->GetInDataAnchor(0));
  GraphUtils::AddEdge(tensor_move1->GetOutDataAnchor(0), scatter1->GetInDataAnchor(0));
  GraphUtils::AddEdge(update_indices->GetOutDataAnchor(0), scatter1->GetInDataAnchor(1));

  GraphUtils::AddEdge(scatter0->GetOutDataAnchor(0), netoutput->GetInDataAnchor(0));
  GraphUtils::AddEdge(scatter1->GetOutDataAnchor(0), netoutput->GetInDataAnchor(1));

  EXPECT_EQ(CountNodesByType(graph, TENSORMOVE), 4U);
  EXPECT_EQ(RunTensorMoveDeletePass(graph), SUCCESS);

  EXPECT_EQ(CountNodesByType(graph, TENSORMOVE), 0U);
  EXPECT_TRUE(kv_cache->GetOutDataAnchor(0)->IsLinkedWith(mla->GetInDataAnchor(9)));
  EXPECT_TRUE(kr_cache->GetOutDataAnchor(0)->IsLinkedWith(mla->GetInDataAnchor(10)));
  EXPECT_TRUE(reshape2->GetOutDataAnchor(0)->IsLinkedWith(scatter0->GetInDataAnchor(0)));
  EXPECT_TRUE(reshape3->GetOutDataAnchor(0)->IsLinkedWith(scatter1->GetInDataAnchor(0)));
  EXPECT_TRUE(index_by_tensor0->GetOutControlAnchor()->IsLinkedWith(scatter0->GetInControlAnchor()));
  EXPECT_TRUE(index_by_tensor1->GetOutControlAnchor()->IsLinkedWith(scatter1->GetInControlAnchor()));
  ge::GetThreadLocalContext().SetGraphOption({});
}

/**
 * 对应 bbit Test7 的 70 leftover dump（单 TM1 + 单层 Reshape 分支 + 直连分支）。
 *
 * 备注：本用例验证通用 TensorMoveDeletePass 对 MLA 拓扑的兜底删除行为；
 *       MLA P3/P4 共用 TM1 的协调逻辑由 UT 侧覆盖。
 *
 * 构图（pass 前）：
 *
 *   arg11_1 ──► TensorMove ──► MlaPrologV3[9]
 *                                    │
 *                                    ├─:2─► Reshape_23 ──┬─► TensorMove_2    ─► ScatterNdUpdate[0]
 *                                    │                   └─► IndexByTensor_2 ─► ScatterNdUpdate[2]
 *                                    │
 *                                    └─:3─► Squeeze_19 ─► Reshape_24 ──┬─► ScatterNdUpdate_1[0]   （无 TM，直连）
 *                                                                      └─► IndexByTensor_3 ─► ScatterNdUpdate_1[2]
 *
 *   arg19_1 ──► IndexByTensor_2[1]、IndexByTensor_3[1]
 *   arg25_1 ──► ScatterNdUpdate[1]、ScatterNdUpdate_1[1]
 *
 *   ScatterNdUpdate   ──► NetOutput[0]
 *   ScatterNdUpdate_1 ──► NetOutput[1]
 *
 * Pass 后预期：
 *   - 2 个 TensorMove（TM1 + TM2）全部删除
 *   - arg11_1 直连 MlaPrologV3[9]
 *   - Reshape_23 直连 ScatterNdUpdate[0]；Reshape_24 保持直连 ScatterNdUpdate_1[0]
 *   - 多引用分支序列化：IndexByTensor_2 ─ctrl─► ScatterNdUpdate
 */
TEST_F(TensorMoveTest, TensorMove_MlaDump70LeftTmP3P4_Deleted) {
  SetMlaDumpReuseOptions();
  auto graph = std::make_shared<ComputeGraph>("g1");

  auto kv_cache = AddTestNode(graph, "arg11_1", DATA, 0, 1);
  auto indices = AddTestNode(graph, "arg19_1", DATA, 0, 1);
  auto update_indices = AddTestNode(graph, "arg25_1", DATA, 0, 1);
  auto tensor_move_kv = AddTestNode(graph, "TensorMove", TENSORMOVE, 1, 1);
  auto mla = AddTestNode(graph, "MlaPrologV3", "MlaPrologV3", 21, 7);
  auto reshape_p4 = AddTestNode(graph, "Reshape_23", RESHAPE, 2, 1);
  auto index_by_tensor0 = AddTestNode(graph, "IndexByTensor_2", "IndexByTensor", 2, 1);
  auto tensor_move0 = AddTestNode(graph, "TensorMove_2", TENSORMOVE, 1, 1);
  auto scatter0 = AddTestNode(graph, "ScatterNdUpdate", "ScatterNdUpdate", 3, 1);
  auto squeeze = AddTestNode(graph, "Squeeze_19", SQUEEZE, 1, 1);
  auto reshape_p3 = AddTestNode(graph, "Reshape_24", RESHAPE, 2, 1);
  auto index_by_tensor1 = AddTestNode(graph, "IndexByTensor_3", "IndexByTensor", 2, 1);
  auto scatter1 = AddTestNode(graph, "ScatterNdUpdate_1", "ScatterNdUpdate", 3, 1);
  auto netoutput = AddTestNode(graph, "NetOutput", NETOUTPUT, 2, 1);

  AttrUtils::SetInt(kv_cache->GetOpDesc(), ATTR_NAME_INDEX, 0);

  GraphUtils::AddEdge(kv_cache->GetOutDataAnchor(0), tensor_move_kv->GetInDataAnchor(0));
  GraphUtils::AddEdge(tensor_move_kv->GetOutDataAnchor(0), mla->GetInDataAnchor(9));

  GraphUtils::AddEdge(mla->GetOutDataAnchor(2), reshape_p4->GetInDataAnchor(0));
  // ScatterNdUpdate 原地写回 input0 对应的源内存，旁路 IndexByTensor_2 纯读，需补 reader-before-writer 控制边
  SetInplaceOutput(scatter0, 0, 0);

  GraphUtils::AddEdge(reshape_p4->GetOutDataAnchor(0), index_by_tensor0->GetInDataAnchor(0));
  GraphUtils::AddEdge(indices->GetOutDataAnchor(0), index_by_tensor0->GetInDataAnchor(1));
  GraphUtils::AddEdge(index_by_tensor0->GetOutDataAnchor(0), scatter0->GetInDataAnchor(2));
  GraphUtils::AddEdge(reshape_p4->GetOutDataAnchor(0), tensor_move0->GetInDataAnchor(0));
  GraphUtils::AddEdge(tensor_move0->GetOutDataAnchor(0), scatter0->GetInDataAnchor(0));
  GraphUtils::AddEdge(update_indices->GetOutDataAnchor(0), scatter0->GetInDataAnchor(1));

  GraphUtils::AddEdge(mla->GetOutDataAnchor(3), squeeze->GetInDataAnchor(0));
  GraphUtils::AddEdge(squeeze->GetOutDataAnchor(0), reshape_p3->GetInDataAnchor(0));
  GraphUtils::AddEdge(reshape_p3->GetOutDataAnchor(0), index_by_tensor1->GetInDataAnchor(0));
  GraphUtils::AddEdge(indices->GetOutDataAnchor(0), index_by_tensor1->GetInDataAnchor(1));
  GraphUtils::AddEdge(index_by_tensor1->GetOutDataAnchor(0), scatter1->GetInDataAnchor(2));
  GraphUtils::AddEdge(reshape_p3->GetOutDataAnchor(0), scatter1->GetInDataAnchor(0));
  GraphUtils::AddEdge(update_indices->GetOutDataAnchor(0), scatter1->GetInDataAnchor(1));

  GraphUtils::AddEdge(scatter0->GetOutDataAnchor(0), netoutput->GetInDataAnchor(0));
  GraphUtils::AddEdge(scatter1->GetOutDataAnchor(0), netoutput->GetInDataAnchor(1));

  EXPECT_EQ(CountNodesByType(graph, TENSORMOVE), 2U);
  EXPECT_EQ(RunTensorMoveDeletePass(graph), SUCCESS);

  EXPECT_EQ(CountNodesByType(graph, TENSORMOVE), 0U);
  EXPECT_TRUE(kv_cache->GetOutDataAnchor(0)->IsLinkedWith(mla->GetInDataAnchor(9)));
  EXPECT_TRUE(reshape_p4->GetOutDataAnchor(0)->IsLinkedWith(scatter0->GetInDataAnchor(0)));
  EXPECT_TRUE(reshape_p3->GetOutDataAnchor(0)->IsLinkedWith(scatter1->GetInDataAnchor(0)));
  EXPECT_TRUE(index_by_tensor0->GetOutControlAnchor()->IsLinkedWith(scatter0->GetInControlAnchor()));
  ge::GetThreadLocalContext().SetGraphOption({});
}

/**
 * 对应 bbit Test8 的 46 ctrl dump（Identity 承载 ctrl 边，插入 Reshape 与 TM 之间）。
 *
 * 备注：本用例验证通用 TensorMoveDeletePass 对 MLA 拓扑的兜底删除行为；
 *       依赖 mock 节点未配置 ref 输出属性 —— 源头回溯在 Identity 处停止，
 *       真实 RefOp 场景（Reshape/Identity 均透传）请在 UT 侧覆盖。
 *
 * 构图（pass 前）：
 *
 *   arg11_1 ──► TensorMove    ──┐
 *                               ├──► MlaPrologV3[9]/[10]
 *   arg12_1 ──► TensorMove_1  ──┘          │
 *                                          ├─:2─► Reshape_60 ─► Reshape_61 ─► Reshape_62 ──┬─► Identity_28     ─► TensorMove_2 ─► ScatterNdUpdate[0]
 *                                          │                                               └─► IndexByTensor_2                  ─► ScatterNdUpdate[2]
 *                                          │
 *                                          └─:3─► Squeeze_19 ─► Reshape_63 ─┬─► Identity_29     ─► TensorMove_3 ─► ScatterNdUpdate_1[0]
 *                                                                           └─► IndexByTensor_3                  ─► ScatterNdUpdate_1[2]
 *
 *   arg19_1 ──► IndexByTensor_2[1]、IndexByTensor_3[1]
 *   arg25_1 ──► ScatterNdUpdate[1]、ScatterNdUpdate_1[1]
 *
 *   FusedInferAttentionScore ─ctrl─► Identity_28
 *                            ─ctrl─► Identity_29
 *
 *   ScatterNdUpdate   ──► NetOutput[0]
 *   ScatterNdUpdate_1 ──► NetOutput[1]
 *
 * Pass 后预期：
 *   - 4 个 TensorMove 全部删除
 *   - Identity_28/_29 保留，分别直连 ScatterNdUpdate/_1[0]
 *   - FusedInferAttentionScore ─ctrl─► Identity_28/_29 保持
 */
TEST_F(TensorMoveTest, TensorMove_MlaDump46CtrlIdentityBranches_Deleted) {
  SetMlaDumpReuseOptions();
  auto graph = std::make_shared<ComputeGraph>("g1");

  auto kv_cache = AddTestNode(graph, "arg11_1", DATA, 0, 1);
  auto kr_cache = AddTestNode(graph, "arg12_1", DATA, 0, 1);
  auto indices = AddTestNode(graph, "arg19_1", DATA, 0, 1);
  auto update_indices = AddTestNode(graph, "arg25_1", DATA, 0, 1);
  auto tensor_move_kv = AddTestNode(graph, "TensorMove", TENSORMOVE, 1, 1);
  auto tensor_move_kr = AddTestNode(graph, "TensorMove_1", TENSORMOVE, 1, 1);
  auto mla = AddTestNode(graph, "MlaPrologV3", "MlaPrologV3", 21, 7);
  auto reshape0 = AddTestNode(graph, "Reshape_60", RESHAPE, 2, 1);
  auto reshape1 = AddTestNode(graph, "Reshape_61", RESHAPE, 2, 1);
  auto reshape2 = AddTestNode(graph, "Reshape_62", RESHAPE, 2, 1);
  auto index_by_tensor0 = AddTestNode(graph, "IndexByTensor_2", "IndexByTensor", 2, 1);
  auto identity0 = AddTestNode(graph, "Identity_28", IDENTITY, 1, 1);
  auto tensor_move0 = AddTestNode(graph, "TensorMove_2", TENSORMOVE, 1, 1);
  auto scatter0 = AddTestNode(graph, "ScatterNdUpdate", "ScatterNdUpdate", 3, 1);
  auto squeeze = AddTestNode(graph, "Squeeze_19", SQUEEZE, 1, 1);
  auto reshape3 = AddTestNode(graph, "Reshape_63", RESHAPE, 2, 1);
  auto index_by_tensor1 = AddTestNode(graph, "IndexByTensor_3", "IndexByTensor", 2, 1);
  auto identity1 = AddTestNode(graph, "Identity_29", IDENTITY, 1, 1);
  auto tensor_move1 = AddTestNode(graph, "TensorMove_3", TENSORMOVE, 1, 1);
  auto scatter1 = AddTestNode(graph, "ScatterNdUpdate_1", "ScatterNdUpdate", 3, 1);
  auto ctrl_src = AddTestNode(graph, "FusedInferAttentionScore", "FusedInferAttentionScore", 0, 1);
  auto netoutput = AddTestNode(graph, "NetOutput", NETOUTPUT, 2, 1);

  AttrUtils::SetInt(kv_cache->GetOpDesc(), ATTR_NAME_INDEX, 0);
  AttrUtils::SetInt(kr_cache->GetOpDesc(), ATTR_NAME_INDEX, 1);

  GraphUtils::AddEdge(kv_cache->GetOutDataAnchor(0), tensor_move_kv->GetInDataAnchor(0));
  GraphUtils::AddEdge(tensor_move_kv->GetOutDataAnchor(0), mla->GetInDataAnchor(9));
  GraphUtils::AddEdge(kr_cache->GetOutDataAnchor(0), tensor_move_kr->GetInDataAnchor(0));
  GraphUtils::AddEdge(tensor_move_kr->GetOutDataAnchor(0), mla->GetInDataAnchor(10));

  GraphUtils::AddEdge(mla->GetOutDataAnchor(2), reshape0->GetInDataAnchor(0));
  GraphUtils::AddEdge(reshape0->GetOutDataAnchor(0), reshape1->GetInDataAnchor(0));
  GraphUtils::AddEdge(reshape1->GetOutDataAnchor(0), reshape2->GetInDataAnchor(0));
  GraphUtils::AddEdge(reshape2->GetOutDataAnchor(0), index_by_tensor0->GetInDataAnchor(0));
  GraphUtils::AddEdge(indices->GetOutDataAnchor(0), index_by_tensor0->GetInDataAnchor(1));
  GraphUtils::AddEdge(index_by_tensor0->GetOutDataAnchor(0), scatter0->GetInDataAnchor(2));
  GraphUtils::AddEdge(reshape2->GetOutDataAnchor(0), identity0->GetInDataAnchor(0));
  GraphUtils::AddEdge(ctrl_src->GetOutControlAnchor(), identity0->GetInControlAnchor());
  GraphUtils::AddEdge(identity0->GetOutDataAnchor(0), tensor_move0->GetInDataAnchor(0));
  GraphUtils::AddEdge(tensor_move0->GetOutDataAnchor(0), scatter0->GetInDataAnchor(0));
  GraphUtils::AddEdge(update_indices->GetOutDataAnchor(0), scatter0->GetInDataAnchor(1));

  GraphUtils::AddEdge(mla->GetOutDataAnchor(3), squeeze->GetInDataAnchor(0));
  GraphUtils::AddEdge(squeeze->GetOutDataAnchor(0), reshape3->GetInDataAnchor(0));
  GraphUtils::AddEdge(reshape3->GetOutDataAnchor(0), index_by_tensor1->GetInDataAnchor(0));
  GraphUtils::AddEdge(indices->GetOutDataAnchor(0), index_by_tensor1->GetInDataAnchor(1));
  GraphUtils::AddEdge(index_by_tensor1->GetOutDataAnchor(0), scatter1->GetInDataAnchor(2));
  GraphUtils::AddEdge(reshape3->GetOutDataAnchor(0), identity1->GetInDataAnchor(0));
  GraphUtils::AddEdge(ctrl_src->GetOutControlAnchor(), identity1->GetInControlAnchor());
  GraphUtils::AddEdge(identity1->GetOutDataAnchor(0), tensor_move1->GetInDataAnchor(0));
  GraphUtils::AddEdge(tensor_move1->GetOutDataAnchor(0), scatter1->GetInDataAnchor(0));
  GraphUtils::AddEdge(update_indices->GetOutDataAnchor(0), scatter1->GetInDataAnchor(1));

  GraphUtils::AddEdge(scatter0->GetOutDataAnchor(0), netoutput->GetInDataAnchor(0));
  GraphUtils::AddEdge(scatter1->GetOutDataAnchor(0), netoutput->GetInDataAnchor(1));

  EXPECT_EQ(CountNodesByType(graph, TENSORMOVE), 4U);
  EXPECT_EQ(RunTensorMoveDeletePass(graph), SUCCESS);

  EXPECT_EQ(CountNodesByType(graph, TENSORMOVE), 0U);
  EXPECT_TRUE(identity0->GetOutDataAnchor(0)->IsLinkedWith(scatter0->GetInDataAnchor(0)));
  EXPECT_TRUE(identity1->GetOutDataAnchor(0)->IsLinkedWith(scatter1->GetInDataAnchor(0)));
  EXPECT_TRUE(ctrl_src->GetOutControlAnchor()->IsLinkedWith(identity0->GetInControlAnchor()));
  EXPECT_TRUE(ctrl_src->GetOutControlAnchor()->IsLinkedWith(identity1->GetInControlAnchor()));
  ge::GetThreadLocalContext().SetGraphOption({});
}

/**
 * 对应 bbit Test9 的 58 ctrl dump（Identity ctrl 边 + 单层 Reshape + Squeeze 分支）。
 *
 * 备注：本用例验证通用 TensorMoveDeletePass 对 MLA 拓扑的兜底删除行为；
 *       依赖 mock 节点未配置 ref 输出属性 —— 源头回溯在 Identity 处停止，
 *       真实 RefOp 场景请在 UT 侧覆盖。
 *
 * 构图（pass 前）：
 *
 *   arg11_1 ──► TensorMove    ──┐
 *                               ├──► MlaPrologV3[9]/[10]
 *   arg12_1 ──► TensorMove_1  ──┘          │
 *                                          ├─:2─► Reshape_23 ──┬─► Identity_28     ─► TensorMove_2 ─► ScatterNdUpdate[0]
 *                                          │                   └─► IndexByTensor_2                  ─► ScatterNdUpdate[2]
 *                                          │
 *                                          └─:3─► Squeeze_19 ─► Reshape_24 ─┬─► Identity_29     ─► TensorMove_3 ─► ScatterNdUpdate_1[0]
 *                                                                           └─► IndexByTensor_3                  ─► ScatterNdUpdate_1[2]
 *
 *   arg19_1 ──► IndexByTensor_2[1]、IndexByTensor_3[1]
 *   arg25_1 ──► ScatterNdUpdate[1]、ScatterNdUpdate_1[1]
 *
 *   FusedInferAttentionScore ─ctrl─► Identity_28
 *                            ─ctrl─► Identity_29
 *
 *   ScatterNdUpdate   ──► NetOutput[0]
 *   ScatterNdUpdate_1 ──► NetOutput[1]
 *
 * Pass 后预期：
 *   - 4 个 TensorMove 全部删除
 *   - arg11_1/arg12_1 直连 MlaPrologV3[9]/[10]
 *   - Identity_28/_29 保留，分别直连 ScatterNdUpdate/_1[0]
 *   - FusedInferAttentionScore ─ctrl─► Identity_28/_29 保持
 */
TEST_F(TensorMoveTest, TensorMove_MlaDump58CtrlSingleReshapeAndSqueezeBranches_Deleted) {
  SetMlaDumpReuseOptions();
  auto graph = std::make_shared<ComputeGraph>("g1");

  auto kv_cache = AddTestNode(graph, "arg11_1", DATA, 0, 1);
  auto kr_cache = AddTestNode(graph, "arg12_1", DATA, 0, 1);
  auto indices = AddTestNode(graph, "arg19_1", DATA, 0, 1);
  auto update_indices = AddTestNode(graph, "arg25_1", DATA, 0, 1);
  auto tensor_move_kv = AddTestNode(graph, "TensorMove", TENSORMOVE, 1, 1);
  auto tensor_move_kr = AddTestNode(graph, "TensorMove_1", TENSORMOVE, 1, 1);
  auto mla = AddTestNode(graph, "MlaPrologV3", "MlaPrologV3", 21, 7);
  auto reshape_p4 = AddTestNode(graph, "Reshape_23", RESHAPE, 2, 1);
  auto index_by_tensor0 = AddTestNode(graph, "IndexByTensor_2", "IndexByTensor", 2, 1);
  auto identity0 = AddTestNode(graph, "Identity_28", IDENTITY, 1, 1);
  auto tensor_move0 = AddTestNode(graph, "TensorMove_2", TENSORMOVE, 1, 1);
  auto scatter0 = AddTestNode(graph, "ScatterNdUpdate", "ScatterNdUpdate", 3, 1);
  auto squeeze = AddTestNode(graph, "Squeeze_19", SQUEEZE, 1, 1);
  auto reshape_p2 = AddTestNode(graph, "Reshape_24", RESHAPE, 2, 1);
  auto index_by_tensor1 = AddTestNode(graph, "IndexByTensor_3", "IndexByTensor", 2, 1);
  auto identity1 = AddTestNode(graph, "Identity_29", IDENTITY, 1, 1);
  auto tensor_move1 = AddTestNode(graph, "TensorMove_3", TENSORMOVE, 1, 1);
  auto scatter1 = AddTestNode(graph, "ScatterNdUpdate_1", "ScatterNdUpdate", 3, 1);
  auto ctrl_src = AddTestNode(graph, "FusedInferAttentionScore", "FusedInferAttentionScore", 0, 1);
  auto netoutput = AddTestNode(graph, "NetOutput", NETOUTPUT, 2, 1);

  AttrUtils::SetInt(kv_cache->GetOpDesc(), ATTR_NAME_INDEX, 0);
  AttrUtils::SetInt(kr_cache->GetOpDesc(), ATTR_NAME_INDEX, 1);

  GraphUtils::AddEdge(kv_cache->GetOutDataAnchor(0), tensor_move_kv->GetInDataAnchor(0));
  GraphUtils::AddEdge(tensor_move_kv->GetOutDataAnchor(0), mla->GetInDataAnchor(9));
  GraphUtils::AddEdge(kr_cache->GetOutDataAnchor(0), tensor_move_kr->GetInDataAnchor(0));
  GraphUtils::AddEdge(tensor_move_kr->GetOutDataAnchor(0), mla->GetInDataAnchor(10));

  GraphUtils::AddEdge(mla->GetOutDataAnchor(2), reshape_p4->GetInDataAnchor(0));
  GraphUtils::AddEdge(reshape_p4->GetOutDataAnchor(0), index_by_tensor0->GetInDataAnchor(0));
  GraphUtils::AddEdge(indices->GetOutDataAnchor(0), index_by_tensor0->GetInDataAnchor(1));
  GraphUtils::AddEdge(index_by_tensor0->GetOutDataAnchor(0), scatter0->GetInDataAnchor(2));
  GraphUtils::AddEdge(reshape_p4->GetOutDataAnchor(0), identity0->GetInDataAnchor(0));
  GraphUtils::AddEdge(ctrl_src->GetOutControlAnchor(), identity0->GetInControlAnchor());
  GraphUtils::AddEdge(identity0->GetOutDataAnchor(0), tensor_move0->GetInDataAnchor(0));
  GraphUtils::AddEdge(tensor_move0->GetOutDataAnchor(0), scatter0->GetInDataAnchor(0));
  GraphUtils::AddEdge(update_indices->GetOutDataAnchor(0), scatter0->GetInDataAnchor(1));

  GraphUtils::AddEdge(mla->GetOutDataAnchor(3), squeeze->GetInDataAnchor(0));
  GraphUtils::AddEdge(squeeze->GetOutDataAnchor(0), reshape_p2->GetInDataAnchor(0));
  GraphUtils::AddEdge(reshape_p2->GetOutDataAnchor(0), index_by_tensor1->GetInDataAnchor(0));
  GraphUtils::AddEdge(indices->GetOutDataAnchor(0), index_by_tensor1->GetInDataAnchor(1));
  GraphUtils::AddEdge(index_by_tensor1->GetOutDataAnchor(0), scatter1->GetInDataAnchor(2));
  GraphUtils::AddEdge(reshape_p2->GetOutDataAnchor(0), identity1->GetInDataAnchor(0));
  GraphUtils::AddEdge(ctrl_src->GetOutControlAnchor(), identity1->GetInControlAnchor());
  GraphUtils::AddEdge(identity1->GetOutDataAnchor(0), tensor_move1->GetInDataAnchor(0));
  GraphUtils::AddEdge(tensor_move1->GetOutDataAnchor(0), scatter1->GetInDataAnchor(0));
  GraphUtils::AddEdge(update_indices->GetOutDataAnchor(0), scatter1->GetInDataAnchor(1));

  GraphUtils::AddEdge(scatter0->GetOutDataAnchor(0), netoutput->GetInDataAnchor(0));
  GraphUtils::AddEdge(scatter1->GetOutDataAnchor(0), netoutput->GetInDataAnchor(1));

  EXPECT_EQ(CountNodesByType(graph, TENSORMOVE), 4U);
  EXPECT_EQ(RunTensorMoveDeletePass(graph), SUCCESS);

  EXPECT_EQ(CountNodesByType(graph, TENSORMOVE), 0U);
  EXPECT_TRUE(kv_cache->GetOutDataAnchor(0)->IsLinkedWith(mla->GetInDataAnchor(9)));
  EXPECT_TRUE(kr_cache->GetOutDataAnchor(0)->IsLinkedWith(mla->GetInDataAnchor(10)));
  EXPECT_TRUE(identity0->GetOutDataAnchor(0)->IsLinkedWith(scatter0->GetInDataAnchor(0)));
  EXPECT_TRUE(identity1->GetOutDataAnchor(0)->IsLinkedWith(scatter1->GetInDataAnchor(0)));
  EXPECT_TRUE(ctrl_src->GetOutControlAnchor()->IsLinkedWith(identity0->GetInControlAnchor()));
  EXPECT_TRUE(ctrl_src->GetOutControlAnchor()->IsLinkedWith(identity1->GetInControlAnchor()));
  ge::GetThreadLocalContext().SetGraphOption({});
}

/**
 * 单引用多输出 + 外部已有 reader-before-writer ctrl 边，inplace 分支也可放行删 TM。
 *
 * 构图（pass 前）：
 *
 *                        ┌─► TensorMove ──► Reader ──► NetOutput
 *                Relu ───┤                     │
 *                        │                     └─ctrl─► InplaceWriter
 *                        └─────────────────────► InplaceWriter
 *                                             （INPLACE_SUPPORT_INPUT_INDEX=0）
 *
 * Pass 后预期：
 *   - TensorMove 被删除
 *   - Relu ─► Reader、Relu ─► InplaceWriter 数据边保留
 *   - 外部已有的 Reader ─ctrl─► InplaceWriter 保留，保证 reader-before-writer 语义
 */
TEST_F(TensorMoveTest, TensorMove_InplaceBranchWithExistingReaderCtrl_Deleted) {
  auto graph = std::make_shared<ComputeGraph>("g1");
  auto relu = AddTestNode(graph, "Relu", RELU, 1, 1);
  auto tm = AddTestNode(graph, "TensorMove", TENSORMOVE, 1, 1);
  auto reader = AddTestNode(graph, "Reader", ADD, 1, 1);
  auto writer = AddTestNode(graph, "InplaceWriter", ADD, 1, 1);
  auto netoutput = AddTestNode(graph, "NetOutput", NETOUTPUT, 1, 1);

  SetInplaceOutput(writer, 0U, 0);

  GraphUtils::AddEdge(relu->GetOutDataAnchor(0), tm->GetInDataAnchor(0));
  GraphUtils::AddEdge(tm->GetOutDataAnchor(0), reader->GetInDataAnchor(0));
  GraphUtils::AddEdge(relu->GetOutDataAnchor(0), writer->GetInDataAnchor(0));
  GraphUtils::AddEdge(reader->GetOutDataAnchor(0), netoutput->GetInDataAnchor(0));
  GraphUtils::AddEdge(reader->GetOutControlAnchor(), writer->GetInControlAnchor());

  EXPECT_EQ(CountNodesByType(graph, TENSORMOVE), 1U);
  EXPECT_EQ(RunTensorMoveDeletePass(graph), SUCCESS);
  EXPECT_EQ(CountNodesByType(graph, TENSORMOVE), 0U);
  EXPECT_TRUE(relu->GetOutDataAnchor(0)->IsLinkedWith(reader->GetInDataAnchor(0)));
  EXPECT_TRUE(relu->GetOutDataAnchor(0)->IsLinkedWith(writer->GetInDataAnchor(0)));
  EXPECT_TRUE(reader->GetOutControlAnchor()->IsLinkedWith(writer->GetInControlAnchor()));
}

/**
 * 同上拓扑，但无预置 ctrl 边。回归：inplace 分支无外部保序证据，TM 必须保留。
 */
TEST_F(TensorMoveTest, TensorMove_InplaceBranchWithoutReaderCtrl_Kept) {
  auto graph = std::make_shared<ComputeGraph>("g1");
  auto relu = AddTestNode(graph, "Relu", RELU, 1, 1);
  auto tm = AddTestNode(graph, "TensorMove", TENSORMOVE, 1, 1);
  auto reader = AddTestNode(graph, "Reader", ADD, 1, 1);
  auto writer = AddTestNode(graph, "InplaceWriter", ADD, 1, 1);
  auto netoutput = AddTestNode(graph, "NetOutput", NETOUTPUT, 1, 1);

  SetInplaceOutput(writer, 0U, 0);

  GraphUtils::AddEdge(relu->GetOutDataAnchor(0), tm->GetInDataAnchor(0));
  GraphUtils::AddEdge(tm->GetOutDataAnchor(0), reader->GetInDataAnchor(0));
  GraphUtils::AddEdge(relu->GetOutDataAnchor(0), writer->GetInDataAnchor(0));
  GraphUtils::AddEdge(reader->GetOutDataAnchor(0), netoutput->GetInDataAnchor(0));

  EXPECT_EQ(CountNodesByType(graph, TENSORMOVE), 1U);
  EXPECT_EQ(RunTensorMoveDeletePass(graph), SUCCESS);
  EXPECT_EQ(CountNodesByType(graph, TENSORMOVE), 1U);
}

/**
 * atomic 输出分支（ATOMIC_ATTR_OUTPUT_INDEX 存在）但输出不复用输入时，
 * 不代表旁路会覆写 source 内存，可按普通旁路分支删除 TM。
 * 旁路 AtomicBranch 与 TM 后继 Reader 均为纯读，二者间无读写冒险，不补控制边。
 */
TEST_F(TensorMoveTest, TensorMove_AtomicIndependentOutputBranch_Deleted) {
  auto graph = std::make_shared<ComputeGraph>("g1");
  auto relu = AddTestNode(graph, "Relu", RELU, 1, 1);
  auto tm = AddTestNode(graph, "TensorMove", TENSORMOVE, 1, 1);
  auto reader = AddTestNode(graph, "Reader", ADD, 1, 1);
  auto atomic_branch = AddTestNode(graph, "AtomicBranch", ADD, 1, 1);
  auto netoutput = AddTestNode(graph, "NetOutput", NETOUTPUT, 1, 1);

  AttrUtils::SetListInt(atomic_branch->GetOpDesc(), ATOMIC_ATTR_OUTPUT_INDEX, std::vector<int64_t>{0});

  GraphUtils::AddEdge(relu->GetOutDataAnchor(0), tm->GetInDataAnchor(0));
  GraphUtils::AddEdge(tm->GetOutDataAnchor(0), reader->GetInDataAnchor(0));
  GraphUtils::AddEdge(relu->GetOutDataAnchor(0), atomic_branch->GetInDataAnchor(0));
  GraphUtils::AddEdge(reader->GetOutDataAnchor(0), netoutput->GetInDataAnchor(0));

  EXPECT_EQ(CountNodesByType(graph, TENSORMOVE), 1U);
  EXPECT_EQ(RunTensorMoveDeletePass(graph), SUCCESS);
  EXPECT_EQ(CountNodesByType(graph, TENSORMOVE), 0U);
  EXPECT_TRUE(relu->GetOutDataAnchor(0)->IsLinkedWith(reader->GetInDataAnchor(0)));
  EXPECT_TRUE(relu->GetOutDataAnchor(0)->IsLinkedWith(atomic_branch->GetInDataAnchor(0)));
  EXPECT_FALSE(atomic_branch->GetOutControlAnchor()->IsLinkedWith(reader->GetInControlAnchor()));
}

/**
 * 旁路和 TM 后继都会 inplace 覆写源内存。即使外部已经有 "TM 后继 → 旁路" 的 ctrl 边，
 * 删 TM 后两者会抢同一块 source 内存，语义不等价，必须保留 TM。
 *
 *                          ┌─► TensorMove ──► InplaceWriterA ──► NetOutput
 *                  Relu ───┤                         │
 *                          │                         └─ctrl─► InplaceWriterB
 *                          └─────────────────────► InplaceWriterB
 *                                            （两者都 INPLACE_SUPPORT_INPUT_INDEX=0）
 */
TEST_F(TensorMoveTest, TensorMove_BothTmSuccAndSiblingOverwriteSource_Kept) {
  auto graph = std::make_shared<ComputeGraph>("g1");
  auto relu = AddTestNode(graph, "Relu", RELU, 1, 1);
  auto tm = AddTestNode(graph, "TensorMove", TENSORMOVE, 1, 1);
  auto writer_a = AddTestNode(graph, "InplaceWriterA", ADD, 1, 1);
  auto writer_b = AddTestNode(graph, "InplaceWriterB", ADD, 1, 1);
  auto netoutput = AddTestNode(graph, "NetOutput", NETOUTPUT, 1, 1);

  SetInplaceOutput(writer_a, 0U, 0);
  SetInplaceOutput(writer_b, 0U, 0);

  GraphUtils::AddEdge(relu->GetOutDataAnchor(0), tm->GetInDataAnchor(0));
  GraphUtils::AddEdge(tm->GetOutDataAnchor(0), writer_a->GetInDataAnchor(0));
  GraphUtils::AddEdge(relu->GetOutDataAnchor(0), writer_b->GetInDataAnchor(0));
  GraphUtils::AddEdge(writer_a->GetOutDataAnchor(0), netoutput->GetInDataAnchor(0));
  GraphUtils::AddEdge(writer_a->GetOutControlAnchor(), writer_b->GetInControlAnchor());

  EXPECT_EQ(CountNodesByType(graph, TENSORMOVE), 1U);
  EXPECT_EQ(RunTensorMoveDeletePass(graph), SUCCESS);
  EXPECT_EQ(CountNodesByType(graph, TENSORMOVE), 1U);
}

/**
 * 主图：
 *         Data
 *        /    \
 *     Cast    PartitionedCall
 *      |        |
 *   TransData  NetOutput
 *
 * 子图 sub_1：
 *      sub_Data
 *          |
 *    sub_partitioned_call
 *          |
 *     TensorMove
 *          |
 *     sub_NetOutput
 *
 * 子子图 sub_sub_1：
 *       sub_sub_data
 *        /       \
 *     Cast        \
 *      |           \
 *   TransData      Add
 *      \            /
 *      Add        /
 *        \      /
 *      sub_sub_NetOutput
 *
 *
 * 预期行为：
 * - 删除
 * TensorMove,sub_sub_NetOutput两个输出，一个空悬，一个给到TensorMove，但是任意一个的输入都是计算节点(TransData或Add)
 */
TEST_F(TensorMoveTest, TensorMove_NestedPCall_FromAdd_Deleted) {
  dlog_setlevel(0, 0, 0);

  std::map<std::string, std::string> options;
  options[OPTION_OUTPUT_REUSE_INPUT_MEM_INDEXES] = "1,1|0,0";
  options["ge.oo.level"] = "O3";
  ge::GetThreadLocalContext().SetGraphOption(options);
  const auto sub_sub_data = OP_CFG(DATA).ParentNodeIndex(0);
  DEF_GRAPH(sub_sub_1) {
                         CHAIN(NODE("sub_sub_data", sub_sub_data)->NODE("sub_sub_cast", CAST)
                                   ->NODE("sub_sub_transdata", TRANSDATA)
                                   ->NODE("sub_sub_add0", ADD)->NODE("sub_sub_netoutput", NETOUTPUT));
                         CHAIN(NODE("sub_sub_data", sub_sub_data)->EDGE(0, 0)->NODE("sub_sub_add1", ADD)
                                   ->NODE("sub_sub_netoutput", NETOUTPUT));
                   };
  const auto sub_data = OP_CFG(DATA).ParentNodeIndex(0);
  DEF_GRAPH(sub_1) {
                     CHAIN(NODE("sub_data", sub_data)->NODE("sub_partitioned_call", PARTITIONEDCALL, sub_sub_1)
                               ->NODE("sub_tensor_move", TENSORMOVE)
                               ->NODE("sub_netoutput", NETOUTPUT));
                   };
  DEF_GRAPH(g1) {
                  CHAIN(NODE("data", DATA)->EDGE(0, 0)->NODE("cast", CAST)->NODE("transdata", TRANSDATA)
                            ->NODE("netoutput", NETOUTPUT));
                  CHAIN(NODE("data", DATA)
                    ->EDGE(0, 0)->NODE("partitioned_call", PARTITIONEDCALL, sub_1)
                    ->EDGE(0, 0)->NODE("netoutput", NETOUTPUT));
                };
  auto sub_sub_1_graph = ToComputeGraph(sub_sub_1);
  sub_1.Layout();
  auto compute_graph = ToComputeGraph(g1);
  const auto sub_graph_1 = compute_graph->GetSubgraph("sub_1");
  ASSERT_NE(sub_graph_1, nullptr);

  auto sub_partitioned_call_node = sub_graph_1->FindNode("sub_partitioned_call");
  ASSERT_NE(sub_partitioned_call_node, nullptr);
  sub_sub_1_graph->SetParentGraph(compute_graph);
  sub_sub_1_graph->SetParentNode(sub_partitioned_call_node);
  compute_graph->AddSubGraph(sub_sub_1_graph);  // 嵌套子图

  const auto sub_sub_graph_1 = compute_graph->GetSubgraph("sub_sub_1");
  ASSERT_NE(sub_sub_graph_1, nullptr);

  ASSERT_TRUE(SetTransDataTensorDesc(compute_graph, {"transdata", "sub_sub_transdata"}));

  NetoutputParentIndexes indexes{{"sub_netoutput", {0}}, {"sub_sub_netoutput", {0, 1}}};
  ASSERT_TRUE(AddParentIndexForNetoutput(compute_graph, indexes));

  ge::GEPass pass(compute_graph);
  TensorMoveDeletePass tensor_move_delete_pass;
  ge::NamesToPass names_to_pass;
  names_to_pass.emplace_back("TensorMoveDeletePass", &tensor_move_delete_pass);
  EXPECT_NE(sub_graph_1->FindNode("sub_tensor_move"), nullptr);
  EXPECT_EQ(pass.Run(names_to_pass), SUCCESS);
  EXPECT_EQ(sub_graph_1->FindNode("sub_tensor_move"), nullptr);

  ge::GetThreadLocalContext().SetGraphOption({});
}

/**
 *        data
 *          |
 *  PartitionedCall
 *          |
 *      netoutput
 *
 * 子图 sub_1：
 *
 *      sub_data
 *          |
 *  sub_partitioned_call
 *          |
 *       tensormove
 *          |
 *     sub_netoutput
 *
 * 子子图 sub_sub_1：
 *
 *       sub_sub_data
 *        /       \
 *   sub_sub_cast   sub_sub_add1
 *        |           |
 * sub_sub_transdata  |
 *        |           |
 *   sub_sub_add0 -----
 *        |
 *  sub_sub_netoutput
 *
 * 预期结果：
 * tensormove的输入是sub_sub_add0，sub_sub_add0只有一条路径，删除
 */
TEST_F(TensorMoveTest, TensorMoveInSub_FromSubSubAdd_Deleted) {
  dlog_setlevel(0, 0, 0);

  std::map<std::string, std::string> options;
  options[OPTION_OUTPUT_REUSE_INPUT_MEM_INDEXES] = "1,1|0,0";
  options["ge.oo.level"] = "O3";
  ge::GetThreadLocalContext().SetGraphOption(options);
  const auto sub_sub_data = OP_CFG(DATA).ParentNodeIndex(0);
  DEF_GRAPH(sub_sub_1) {
                         CHAIN(NODE("sub_sub_data", sub_sub_data)->NODE("sub_sub_cast", CAST)
                                   ->NODE("sub_sub_transdata", TRANSDATA)
                                   ->NODE("sub_sub_add0", ADD)->NODE("sub_sub_netoutput", NETOUTPUT));
                         CHAIN(NODE("sub_sub_data", sub_sub_data)->EDGE(0, 0)->NODE("sub_sub_add1", ADD)
                                   ->NODE("sub_sub_netoutput", NETOUTPUT));
                   };
  const auto sub_data = OP_CFG(DATA).ParentNodeIndex(0);
  DEF_GRAPH(sub_1) {
                     CHAIN(NODE("sub_data", sub_data)->NODE("sub_partitioned_call", PARTITIONEDCALL, sub_sub_1)
                               ->NODE("sub_tensor_move", TENSORMOVE)
                               ->NODE("sub_netoutput", NETOUTPUT));
                   };
  DEF_GRAPH(g1) {
                  CHAIN(NODE("data", DATA)
                    ->EDGE(0, 0)->NODE("partitioned_call", PARTITIONEDCALL, sub_1)
                    ->EDGE(0, 0)->NODE("netoutput", NETOUTPUT));
                };
  auto sub_sub_1_graph = ToComputeGraph(sub_sub_1);
  sub_1.Layout();
  auto compute_graph = ToComputeGraph(g1);
  const auto sub_graph_1 = compute_graph->GetSubgraph("sub_1");
  ASSERT_NE(sub_graph_1, nullptr);

  auto sub_partitioned_call_node = sub_graph_1->FindNode("sub_partitioned_call");
  ASSERT_NE(sub_partitioned_call_node, nullptr);
  sub_sub_1_graph->SetParentGraph(compute_graph);
  sub_sub_1_graph->SetParentNode(sub_partitioned_call_node);
  compute_graph->AddSubGraph(sub_sub_1_graph);  // 嵌套子图

  const auto sub_sub_graph_1 = compute_graph->GetSubgraph("sub_sub_1");
  ASSERT_NE(sub_sub_graph_1, nullptr);

  ASSERT_TRUE(SetTransDataTensorDesc(compute_graph, {"sub_sub_transdata"}));

  NetoutputParentIndexes indexes{{"sub_netoutput", {0}},
                                 {"sub_sub_netoutput", {0, 1}}};
  ASSERT_TRUE(AddParentIndexForNetoutput(compute_graph, indexes));

  ge::GEPass pass(compute_graph);
  TensorMoveDeletePass tensor_move_delete_pass;
  ge::NamesToPass names_to_pass;
  names_to_pass.emplace_back("TensorMoveDeletePass", &tensor_move_delete_pass);
  EXPECT_NE(sub_graph_1->FindNode("sub_tensor_move"), nullptr);
  EXPECT_EQ(pass.Run(names_to_pass), SUCCESS);
  EXPECT_EQ(sub_graph_1->FindNode("sub_tensor_move"), nullptr);

  ge::GetThreadLocalContext().SetGraphOption({});
}

/**
 * 主图 g1：
 *
 *        data
 *          |
 *   PartitionedCall
 *          |
 *       tensormove
 *          |
 *       netoutput
 *
 * 子图 sub_1：
 *
 *      sub_data
 *          |
 *  sub_partitioned_call
 *          |
 *     sub_tensormove
 *          |
 *     sub_netoutput
 *
 * 子子图 sub_sub_1：
 *
 *       sub_sub_data
 *        /       \
 *   sub_sub_cast   sub_sub_add1
 *        |           |
 * sub_sub_transdata  |
 *        |           |
 *   sub_sub_add0     |
 *        |
 *     sub_sub_netoutput

 *
 * 预期结果：
 * - 主图 tensormove 的真实输入应追溯至子子图的 sub_sub_add0，tensormove 被成功删除；
 * - sub_tensormove 的真实输入应追溯至子子图中的 sub_sub_add0，sub_tensormove 也被成功删除；
 */

TEST_F(TensorMoveTest, TensorMoveInRootAndSub_FromSubSubAdd_Deleted) {
  dlog_setlevel(0, 0, 0);

  std::map<std::string, std::string> options;
  options[OPTION_OUTPUT_REUSE_INPUT_MEM_INDEXES] = "1,1|0,0";
  options["ge.oo.level"] = "O3";
  ge::GetThreadLocalContext().SetGraphOption(options);
  const auto sub_sub_data = OP_CFG(DATA).ParentNodeIndex(0);
  DEF_GRAPH(sub_sub_1) {
                         CHAIN(NODE("sub_sub_data", sub_sub_data)->NODE("sub_sub_cast", CAST)
                                   ->NODE("sub_sub_transdata", TRANSDATA)
                                   ->NODE("sub_sub_add0", ADD)->NODE("sub_sub_netoutput", NETOUTPUT));
                         CHAIN(NODE("sub_sub_data", sub_sub_data)->EDGE(0, 0)->NODE("sub_sub_add1", ADD)
                                   ->NODE("sub_sub_netoutput", NETOUTPUT));
                   };
  const auto sub_data = OP_CFG(DATA).ParentNodeIndex(0);
  DEF_GRAPH(sub_1) {
                     CHAIN(NODE("sub_data", sub_data)->NODE("sub_partitioned_call", PARTITIONEDCALL, sub_sub_1)
                               ->NODE("sub_tensormove", TENSORMOVE)
                               ->NODE("sub_netoutput", NETOUTPUT));
                   };
  DEF_GRAPH(g1) {
                  CHAIN(NODE("data", DATA)
                    ->EDGE(0, 0)->NODE("partitioned_call", PARTITIONEDCALL, sub_1)
                    ->EDGE(0, 0)->NODE("tensormove", TENSORMOVE)
                    ->EDGE(0, 0)->NODE("netoutput", NETOUTPUT));
                };
  auto sub_sub_1_graph = ToComputeGraph(sub_sub_1);
  sub_1.Layout();
  auto compute_graph = ToComputeGraph(g1);
  const auto sub_graph_1 = compute_graph->GetSubgraph("sub_1");
  ASSERT_NE(sub_graph_1, nullptr);

  auto sub_partitioned_call_node = sub_graph_1->FindNode("sub_partitioned_call");
  ASSERT_NE(sub_partitioned_call_node, nullptr);
  sub_sub_1_graph->SetParentGraph(compute_graph);
  sub_sub_1_graph->SetParentNode(sub_partitioned_call_node);
  sub_sub_1_graph->SetOutputSize(2);
  compute_graph->AddSubGraph(sub_sub_1_graph);  // 嵌套子图

  const auto sub_sub_graph_1 = compute_graph->GetSubgraph("sub_sub_1");
  ASSERT_NE(sub_sub_graph_1, nullptr);

  ASSERT_TRUE(SetTransDataTensorDesc(compute_graph, {"sub_sub_transdata"}));

  NetoutputParentIndexes indexes{{"sub_netoutput", {0}},
                                 {"sub_sub_netoutput", {0, 1}}};
  ASSERT_TRUE(AddParentIndexForNetoutput(compute_graph, indexes));

  ge::GEPass pass(compute_graph);
  TensorMoveDeletePass tensor_move_delete_pass;
  ge::NamesToPass names_to_pass;
  names_to_pass.emplace_back("TensorMoveDeletePass", &tensor_move_delete_pass);
  EXPECT_EQ(pass.Run(names_to_pass), SUCCESS);

  EXPECT_EQ(compute_graph->FindNode("tensormove"), nullptr);
  EXPECT_EQ(compute_graph->FindNode("sub_tensormove"), nullptr);

  ge::GetThreadLocalContext().SetGraphOption({});
}

/**
 * 主图 g1：
 *
 *        data
 *          |
 *        relu
 *          |
 *          IF
 *          |
 *      transdata
 *          |
 *      tensormove
 *          |
 *      netoutput
 *
 * if 分支子图 if_sub：
 *
 *        if_sub_data
 *           |\
 *           | if_transdata
 *           |     |
 *           |  if_tensormove
 *           |     |
 *           |   if_relu
 *           |     |
 *           ----if_sub_netoutput
 *
 * then 分支子图 then_sub：
 *
 *      then_sub_data
 *           |
 *       then_relu
 *           |
 *     then_sub_netoutput
 *
 * 预期行为：
 * - tensormove的输入是transdata，只有一条路径，被删除
 * - if_tensormove的输入是if_transdata，只有一条路径，被删除
 */
TEST_F(TensorMoveTest, TensorMoveInRootAndIfSub_ViaTransData_Deleted) {
  dlog_setlevel(0, 0, 0);
  const auto if_sub_data = OP_CFG(DATA).ParentNodeIndex(0);
  DEF_GRAPH(if_sub) {
    CHAIN(NODE("if_sub_data", if_sub_data)->EDGE(0, 0)
              ->NODE("if_sub_netoutput", NETOUTPUT));
    CHAIN(NODE("if_sub_data", if_sub_data)
              ->EDGE(0, 0)->NODE("if_transdata", TRANSDATA)
              ->NODE("if_tensormove", TENSORMOVE)
              ->NODE("if_relu", RELU)
              ->Ctrl()->NODE("if_sub_netoutput", NETOUTPUT));
  };
  const auto then_sub_data = OP_CFG(DATA).ParentNodeIndex(0);
  DEF_GRAPH(then_sub) {
    CHAIN(NODE("then_sub_data", then_sub_data)->NODE("then_relu", RELU)->NODE("then_sub_netoutput", NETOUTPUT));
  };
  DEF_GRAPH(g1) {
    CHAIN(NODE("data", DATA)->NODE("relu", RELU)->NODE("if", IF, if_sub, then_sub)->NODE("transdata", TRANSDATA)
              ->NODE("tensormove", TENSORMOVE)->NODE("netoutput", NETOUTPUT));
  };

  auto compute_graph = ToComputeGraph(g1);
  const auto then_sub_graph = compute_graph->GetSubgraph("then_sub");
  ASSERT_NE(then_sub_graph, nullptr);
  const auto if_sub_graph = compute_graph->GetSubgraph("if_sub");
  ASSERT_NE(if_sub_graph, nullptr);

  compute_graph->TopologicalSorting();

  NetoutputParentIndexes indexes{{"if_sub_netoutput", {0}}, {"then_sub_netoutput", {0}}};
  ASSERT_TRUE(AddParentIndexForNetoutput(compute_graph, indexes));

  ge::GEPass pass(compute_graph);
  TensorMoveDeletePass tensor_move_delete_pass;
  ge::NamesToPass names_to_pass;
  names_to_pass.emplace_back("TensorMoveDeletePass", &tensor_move_delete_pass);
  EXPECT_NE(compute_graph->FindNode("tensormove"), nullptr);
  EXPECT_NE(if_sub_graph->FindNode("if_tensormove"), nullptr);
  EXPECT_EQ(pass.Run(names_to_pass), SUCCESS);
  EXPECT_EQ(compute_graph->FindNode("tensormove"), nullptr);
  EXPECT_EQ(if_sub_graph->FindNode("if_tensormove"), nullptr);
}

/**
 * 主图 g1：
 *
 *        data
 *          |
 *        relu
 *          |
 *          IF
 *          |
 *     transdata
 *          |
 *     tensormove
 *          |
 *     netoutput
 *
 *
 * if 分支子图 if_sub：
 *                if_sub_data
 *                 /       \
 *                /         \
 *   if_sub_netoutput     if_tensormove
 *                             |
 *                           if_relu
 *                             |
 *                     if_sub_netoutput
 *
 * then 分支子图 then_sub：
 *
 *        then_sub_data
 *              |
 *          then_relu
 *              |
 *      then_sub_netoutput
 *
 * 预期行为：
 * - if_sub_graph 中的 if_tensormove 保留，上游源节点为 if_sub_data，但 if_sub_data 存在多条输出路径
 * - 主图中的 tensormove 被成功删除，上游源节点为 transdata，transdata → tensormove → netoutput
 */
TEST_F(TensorMoveTest, TensorMove_RootDeleted_SubKept_DueToSourceBranching) {
  dlog_setlevel(0, 0, 0);
  const auto if_sub_data = OP_CFG(DATA).ParentNodeIndex(0);
  DEF_GRAPH(if_sub) {
    CHAIN(NODE("if_sub_data", if_sub_data)->EDGE(0, 0)
              ->NODE("if_sub_netoutput", NETOUTPUT));
    CHAIN(NODE("if_sub_data", if_sub_data)
              ->EDGE(0, 0)->NODE("if_tensormove", TENSORMOVE)
              ->EDGE(0, 0)->NODE("if_relu", RELU)
              ->EDGE(0, 0)->NODE("if_sub_netoutput", NETOUTPUT));
  };
  const auto then_sub_data = OP_CFG(DATA).ParentNodeIndex(0);
  DEF_GRAPH(then_sub) {
    CHAIN(NODE("then_sub_data", then_sub_data)->NODE("then_relu", RELU)->NODE("then_sub_netoutput", NETOUTPUT));
  };
  DEF_GRAPH(g1) {
    CHAIN(NODE("data", DATA)->NODE("relu", RELU)->NODE("if", IF, if_sub, then_sub)->NODE("transdata", TRANSDATA)
              ->NODE("tensormove", TENSORMOVE)->NODE("netoutput", NETOUTPUT));
  };

  auto compute_graph = ToComputeGraph(g1);
  const auto then_sub_graph = compute_graph->GetSubgraph("then_sub");
  ASSERT_NE(then_sub_graph, nullptr);
  const auto if_sub_graph = compute_graph->GetSubgraph("if_sub");
  ASSERT_NE(if_sub_graph, nullptr);

  compute_graph->TopologicalSorting();

  NetoutputParentIndexes indexes{{"if_sub_netoutput", {0}}, {"then_sub_netoutput", {0}}};
  ASSERT_TRUE(AddParentIndexForNetoutput(compute_graph, indexes));

  EXPECT_NE(if_sub_graph->FindNode("if_tensormove"), nullptr);
  ge::GEPass pass(compute_graph);
  TensorMoveDeletePass tensor_move_delete_pass;
  ge::NamesToPass names_to_pass;
  names_to_pass.emplace_back("TensorMoveDeletePass", &tensor_move_delete_pass);
  EXPECT_NE(if_sub_graph->FindNode("if_tensormove"), nullptr);
  EXPECT_NE(compute_graph->FindNode("tensormove"), nullptr);

  EXPECT_EQ(pass.Run(names_to_pass), SUCCESS);

  EXPECT_NE(if_sub_graph->FindNode("if_tensormove"), nullptr);
  EXPECT_EQ(compute_graph->FindNode("tensormove"), nullptr);
}

/**
 * 主图 g1：
 *
 *        data
 *          |
 *        relu
 *          |
 *          IF
 *          |
 *     transdata
 *          |
 *     tensormove
 *          |
 *     netoutput
 *
 *
 * if 分支子图 if_sub：
 *        if_sub_data
 *             |
 *        if_tensormove
 *             |
 *     if_sub_netoutput
 *
 * then 分支子图 then_sub：
 *        then_sub_data
 *              |
 *          then_relu
 *              |
 *      then_sub_netoutput
 *
 * 预期行为：
 * - if_sub_graph 中的 if_tensormove 保留，其源输入为主图中的 relu，但 relu 的下游节点是 IF 控制流算子
 * - 主图中的 tensormove 被成功删除，其源输入为 transdata，输出是netoutput
 */
TEST_F(TensorMoveTest, TensorMove_RootDeleted_SubInIfKept_DueToIfOp) {
  dlog_setlevel(0, 0, 0);
  const auto if_sub_data = OP_CFG(DATA).ParentNodeIndex(0);
  DEF_GRAPH(if_sub) {
    CHAIN(NODE("if_sub_data", if_sub_data)->EDGE(0, 0)->NODE("if_tensormove", TENSORMOVE)
              ->EDGE(0, 0)->NODE("if_sub_netoutput", NETOUTPUT));
  };
  const auto then_sub_data = OP_CFG(DATA).ParentNodeIndex(0);
  DEF_GRAPH(then_sub) {
    CHAIN(NODE("then_sub_data", then_sub_data)->NODE("then_relu", RELU)->NODE("then_sub_netoutput", NETOUTPUT));
  };
  DEF_GRAPH(g1) {
    CHAIN(NODE("data", DATA)->NODE("relu", RELU)->NODE("if", IF, if_sub, then_sub)->NODE("transdata", TRANSDATA)
              ->NODE("tensormove", TENSORMOVE)->NODE("netoutput", NETOUTPUT));
  };

  auto compute_graph = ToComputeGraph(g1);
  const auto then_sub_graph = compute_graph->GetSubgraph("then_sub");
  ASSERT_NE(then_sub_graph, nullptr);
  const auto if_sub_graph = compute_graph->GetSubgraph("if_sub");
  ASSERT_NE(if_sub_graph, nullptr);

  compute_graph->TopologicalSorting();

  NetoutputParentIndexes indexes{{"if_sub_netoutput", {0}}, {"then_sub_netoutput", {0}}};
  ASSERT_TRUE(AddParentIndexForNetoutput(compute_graph, indexes));

  ge::GEPass pass(compute_graph);
  TensorMoveDeletePass tensor_move_delete_pass;
  ge::NamesToPass names_to_pass;
  names_to_pass.emplace_back("TensorMoveDeletePass", &tensor_move_delete_pass);
  EXPECT_EQ(pass.Run(names_to_pass), SUCCESS);

  EXPECT_NE(if_sub_graph->FindNode("if_tensormove"), nullptr);
  EXPECT_EQ(compute_graph->FindNode("tensormove"), nullptr);
}

/**
 * 主图 g1：
 *          data
 *            |
 *            IF
 *            |
 *        tensormove
 *            |
 *        netoutput
 *
 * if_sub：
 *        if_sub_data
 *          |      \
 *          |       \
 *          |        if_tensormove
 *          |           |
 *          |          |
 *          |         |
 *     if_sub_netoutput
 *
 * then 分支子图 then_sub：
 *
 *        then_sub_data
 *              |
 *          then_relu
 *              |
 *      then_sub_netoutput
 *
 * 预期行为：
 * - if 分支子图中的 if_tensormove 不删除，其输入为根图Data,路径上有IF算子
 * - 主图中的 tensormove 不删除，其输入为根图Data,路径上有IF算子
 */
TEST_F(TensorMoveTest, TensorMove_InRootAndSub_ConnectedToIf_Kept) {
  dlog_setlevel(0, 0, 0);
  std::map<std::string, std::string> options;
  options[OPTION_OUTPUT_REUSE_INPUT_MEM_INDEXES] = "1,1|0,0";
  options["ge.oo.level"] = "O3";
  ge::GetThreadLocalContext().SetGraphOption(options);
  const auto if_sub_data = OP_CFG(DATA).ParentNodeIndex(0);
  DEF_GRAPH(if_sub) {
    CHAIN(NODE("if_sub_data", if_sub_data)->EDGE(0, 1)->NODE("if_sub_netoutput", NETOUTPUT));
    CHAIN(NODE("if_sub_data", if_sub_data)
              ->EDGE(0, 1)->NODE("if_sub_netoutput", NETOUTPUT));
    CHAIN(NODE("if_sub_data", if_sub_data)
              ->EDGE(0, 0)->NODE("if_tensormove", TENSORMOVE)
              ->EDGE(0, 0)->NODE("if_sub_netoutput", NETOUTPUT));
  };
  const auto then_sub_data = OP_CFG(DATA).ParentNodeIndex(0);
  DEF_GRAPH(then_sub) {
    CHAIN(NODE("then_sub_data", then_sub_data)->NODE("then_relu", RELU)->NODE("then_sub_netoutput", NETOUTPUT));
  };
  DEF_GRAPH(g1) {
    CHAIN(NODE("data", DATA)->NODE("if", IF, if_sub, then_sub)
              ->NODE("tensormove", TENSORMOVE)->NODE("netoutput", NETOUTPUT));
  };

  auto compute_graph = ToComputeGraph(g1);
  const auto then_sub_graph = compute_graph->GetSubgraph("then_sub");
  ASSERT_NE(then_sub_graph, nullptr);
  const auto if_sub_graph = compute_graph->GetSubgraph("if_sub");
  ASSERT_NE(if_sub_graph, nullptr);

  compute_graph->TopologicalSorting();

  NetoutputParentIndexes indexes{{"if_sub_netoutput", {0, 1}}, {"then_sub_netoutput", {0}}};
  ASSERT_TRUE(AddParentIndexForNetoutput(compute_graph, indexes));

  ge::GEPass pass(compute_graph);
  TensorMoveDeletePass tensor_move_delete_pass;
  ge::NamesToPass names_to_pass;
  names_to_pass.emplace_back("TensorMoveDeletePass", &tensor_move_delete_pass);
  EXPECT_NE(if_sub_graph->FindNode("if_tensormove"), nullptr);
  EXPECT_NE(compute_graph->FindNode("tensormove"), nullptr);

  EXPECT_EQ(pass.Run(names_to_pass), SUCCESS);

  EXPECT_NE(if_sub_graph->FindNode("if_tensormove"), nullptr);
  EXPECT_NE(compute_graph->FindNode("tensormove"), nullptr);

  ge::GetThreadLocalContext().SetGraphOption({});
}

/**
 * 父图:                子图 sub_1:
 * Data                 sub_Data (ParentIndex: 0)
 * |                      |
 * PartitionedCall ------> TensorMove
 * |                      |
 * NetOutput            sub_NetOutput
 * (复用输入地址)
 *
 * 场景说明：
 * - 子图内部 TensorMove 的前驱是 sub_Data，其在父图的实际源头是 Data。
 * - 设置根图 NetOutput 复用输入内存，触发 TensorMove 优化逻辑。
 *
 * 预期行为：
 * - Trace 能够跨越子图边界识别到 Data 是源头。
 * - TensorMove 被成功识别并删除。
 */
TEST_F(TensorMoveTest, TensorMoveInSubgraph_FromParentData_Deleted2) {
  // 1. 设置内存复用选项：设置根图的第 0 个输出复用第 0 个输入
  std::map<std::string, std::string> options;
  options[OPTION_INPUT_REUSE_MEM_INDEXES] = "0";
  options["ge.oo.level"] = "O3";
  ge::GetThreadLocalContext().SetGraphOption(options);

  // 2. 构造子图 sub_1
  // sub_Data 的 ParentNodeIndex(0) 代表它对应父图中 PartitionedCall 的第 0 个 Input
  const auto sub_data_cfg = OP_CFG(DATA).ParentNodeIndex(0);
  DEF_GRAPH(sub_1) {
    CHAIN(NODE("sub_data", sub_data_cfg)
          ->NODE("sub_tensormove", TENSORMOVE)
          ->NODE("sub_netoutput", NETOUTPUT));
  };

  // 3. 构造父图 g1
  const auto data_cfg = OP_CFG(DATA).Attr(ATTR_NAME_INDEX, 0);
  DEF_GRAPH(g1) {
    CHAIN(NODE("data", data_cfg)
          ->EDGE(0, 0)->NODE("partitioned_call", PARTITIONEDCALL, sub_1)
          ->EDGE(0, 0)->NODE("netoutput", NETOUTPUT));
  };

  // 4. 将子图挂载到父图
  auto compute_graph = ToComputeGraph(g1);
  const auto sub_graph_1 = compute_graph->GetSubgraph("sub_1");
  ASSERT_NE(sub_graph_1, nullptr);

  auto p_call_node = compute_graph->FindNode("partitioned_call");
  ASSERT_NE(p_call_node, nullptr);

  // 设置父子图关联属性
  sub_graph_1->SetParentGraph(compute_graph);
  sub_graph_1->SetParentNode(p_call_node);
  NetoutputParentIndexes indexes{{"netoutput", {0}},
                                 {"sub_netoutput", {0}}};
  ASSERT_TRUE(AddParentIndexForNetoutput(compute_graph, indexes));

  // 6. 执行 Pass
  ge::GEPass pass(compute_graph);
  TensorMoveDeletePass tensor_move_delete_pass;
  ge::NamesToPass names_to_pass;
  names_to_pass.emplace_back("TensorMoveDeletePass", &tensor_move_delete_pass);
  EXPECT_EQ(pass.Run(names_to_pass), SUCCESS);

  // 7. 验证结果：子图内部的 tensormove 应该被删除
  // 注意：FindNode 在子图中查找
  EXPECT_EQ(sub_graph_1->FindNode("sub_tensormove"), nullptr);

  // 清理环境
  ge::GetThreadLocalContext().SetGraphOption({});
}

TEST_F(TensorMoveTest, TensorMoveInSubgraph_FromParentData_NotDeleted) {
  // 1. 设置内存复用选项：设置根图的第 0 个输出复用第 0 个输入
  std::map<std::string, std::string> options;
  options["ge.oo.level"] = "O3";
  ge::GetThreadLocalContext().SetGraphOption(options);

  // 2. 构造子图 sub_1
  // sub_Data 的 ParentNodeIndex(0) 代表它对应父图中 PartitionedCall 的第 0 个 Input
  const auto sub_data_cfg = OP_CFG(DATA).ParentNodeIndex(0);
  DEF_GRAPH(sub_1) {
    CHAIN(NODE("sub_data", sub_data_cfg)
          ->NODE("sub_tensormove", TENSORMOVE)
          ->NODE("sub_netoutput", NETOUTPUT));
  };

  // 3. 构造父图 g1
  const auto data_cfg = OP_CFG(DATA).Attr(ATTR_NAME_INDEX, 0);
  DEF_GRAPH(g1) {
    CHAIN(NODE("data", data_cfg)
          ->EDGE(0, 0)->NODE("partitioned_call", PARTITIONEDCALL, sub_1)
          ->EDGE(0, 0)->NODE("netoutput", NETOUTPUT));
  };

  // 4. 将子图挂载到父图
  auto compute_graph = ToComputeGraph(g1);
  const auto sub_graph_1 = compute_graph->GetSubgraph("sub_1");
  ASSERT_NE(sub_graph_1, nullptr);

  auto p_call_node = compute_graph->FindNode("partitioned_call");
  ASSERT_NE(p_call_node, nullptr);

  // 设置父子图关联属性
  sub_graph_1->SetParentGraph(compute_graph);
  sub_graph_1->SetParentNode(p_call_node);
  NetoutputParentIndexes indexes{{"netoutput", {0}},
                                 {"sub_netoutput", {0}}};
  ASSERT_TRUE(AddParentIndexForNetoutput(compute_graph, indexes));

  // 6. 执行 Pass
  ge::GEPass pass(compute_graph);
  TensorMoveDeletePass tensor_move_delete_pass;
  ge::NamesToPass names_to_pass;
  names_to_pass.emplace_back("TensorMoveDeletePass", &tensor_move_delete_pass);
  EXPECT_EQ(pass.Run(names_to_pass), SUCCESS);

  // 7. 验证结果：子图内部的 tensormove 应该被删除
  // 注意：FindNode 在子图中查找
  EXPECT_NE(sub_graph_1->FindNode("sub_tensormove"), nullptr);

  // 清理环境
  ge::GetThreadLocalContext().SetGraphOption({});
}

// 公共子表达式消除场景，添加内置Identity
TEST_F(TensorMoveTest, Add_InnerIdentity1) {
  DEF_GRAPH(g1) {
    auto assign = OP_CFG(ASSIGN)
                      .TensorDesc(FORMAT_ND, DT_FLOAT, {2, 2})
                      .InCnt(2)
                      .OutCnt(1)
                      .InNames({"ref", "value"})
                      .OutNames({"ref"})
                      .Build("assign");
    CHAIN(NODE("data1", DATA)->EDGE(0, 0)->NODE("add1", ADD)->EDGE(0, 0)->NODE(assign)->CTRL_EDGE()->NODE("add3", ADD));
    CHAIN(NODE("data1")->EDGE(0, 1)->NODE("add1"));
    CHAIN(NODE("add1")->EDGE(0, 0)->NODE("add3"));
    CHAIN(NODE("data2", DATA)->EDGE(0, 1)->NODE(assign));
    CHAIN(NODE("data1")->EDGE(0, 0)->NODE("add2", ADD)->EDGE(0, 1)->NODE("add3"));
    CHAIN(NODE("data1")->EDGE(0, 1)->NODE("add2"));
  };

  auto graph = ToGeGraph(g1);
  map<string, string> options;

  Status ret = ge::GELib::Initialize(options);
  EXPECT_EQ(ret, SUCCESS);

  Session session(options);
  ret = session.AddGraph(0, graph, options);
  EXPECT_EQ(ret, SUCCESS);
  // build input tensor
  std::vector<InputTensorInfo> inputs;
  // build_graph through session
  ret = session.BuildGraph(0, inputs);
  EXPECT_EQ(ret, SUCCESS);

  CHECK_GRAPH(PreRunAfterBuild) {
    size_t add_count = 0U;
    for (const auto &node : graph->GetAllNodes()) {
      if (node->GetType() == ADD) {
        add_count++;
      }
    }
    // 公共子表达式消除，add1和add2合并
    EXPECT_EQ(add_count, 2U);

    auto identity = graph->FindFirstNodeMatchType(IDENTITY);
    ASSERT_NE(identity, nullptr);
    auto assign = graph->FindFirstNodeMatchType(ASSIGN);
    ASSERT_NE(assign, nullptr);
    EXPECT_EQ(assign->GetInDataNodes().at(0), identity);
  };
}

// 常量折叠场景，添加内置Identity
TEST_F(TensorMoveTest, Add_InnerIdentity2) {
  DEF_GRAPH(g1) {
    auto assign = OP_CFG(ASSIGN)
                      .TensorDesc(FORMAT_ND, DT_FLOAT, {2, 2})
                      .InCnt(2)
                      .OutCnt(1)
                      .InNames({"ref", "value"})
                      .OutNames({"ref"})
                      .Build("assign");
    CHAIN(NODE("const1", CONSTANT)->NODE("addn", AddNYes)->NODE(assign)->CTRL_EDGE()->NODE("shape1", ShapeNo));
    CHAIN(NODE("const2", CONSTANT)->EDGE(0, 1)->NODE("addn"));
    CHAIN(NODE("data", DATA)->EDGE(0, 1)->NODE(assign));
    CHAIN(NODE("addn")->EDGE(0, 0)->NODE("shape1")->NODE("net_output", NETOUTPUT));
  };
  auto graph = ToGeGraph(g1);
  auto compute_graph = GraphUtilsEx::GetComputeGraph(graph);
  auto const1 = compute_graph->FindNode("const1");
  auto const2 = compute_graph->FindNode("const2");
  SetWeightForConstNode(const1);
  SetWeightForConstNode(const2);
  map<string, string> options;

  Status ret = ge::GELib::Initialize(options);
  EXPECT_EQ(ret, SUCCESS);

  Session session(options);
  ret = session.AddGraph(0, graph, options);
  EXPECT_EQ(ret, SUCCESS);
  // build input tensor
  std::vector<InputTensorInfo> inputs;
  // build_graph through session
  ret = session.BuildGraph(0, inputs);
  EXPECT_EQ(ret, SUCCESS);

  CHECK_GRAPH(PreRunAfterBuild) {
    auto identity = graph->FindFirstNodeMatchType(IDENTITY);
    ASSERT_NE(identity, nullptr);
    auto assign = graph->FindFirstNodeMatchType(ASSIGN);
    ASSERT_NE(assign, nullptr);
    EXPECT_EQ(assign->GetInDataNodes().at(0), identity);
  };
}

// relu多引用，连给两个ref op，且ref之间没有连边关系，需要插入内置inner Identity
TEST_F(TensorMoveTest, Add_InnerIdentity3) {
  DEF_GRAPH(g1) {
    auto assign1 = OP_CFG(ASSIGN)
                      .TensorDesc(FORMAT_ND, DT_FLOAT, {2, 2})
                      .InCnt(2)
                      .OutCnt(1)
                      .InNames({"ref", "value"})
                      .OutNames({"ref"})
                      .Build("assign1");
    auto assign2 = OP_CFG(ASSIGN)
                      .TensorDesc(FORMAT_ND, DT_FLOAT, {2, 2})
                      .InCnt(2)
                      .OutCnt(1)
                      .InNames({"ref", "value"})
                      .OutNames({"ref"})
                      .Build("assign2");
    CHAIN(NODE("data",DATA)->NODE("relu", RELU)->NODE(assign1));
    CHAIN(NODE("data")->EDGE(0, 1)->NODE(assign1));
    CHAIN(NODE("relu")->EDGE(0, 0)->NODE(assign2));
    CHAIN(NODE("data")->EDGE(0, 1)->NODE(assign2));
  };
  auto graph = ToGeGraph(g1);
  map<string, string> options;

  Status ret = ge::GELib::Initialize(options);
  EXPECT_EQ(ret, SUCCESS);

  Session session(options);
  ret = session.AddGraph(0, graph, options);
  EXPECT_EQ(ret, SUCCESS);
  // build input tensor
  std::vector<InputTensorInfo> inputs;
  // build_graph through session
  ret = session.BuildGraph(0, inputs);
  EXPECT_EQ(ret, SUCCESS);

  CHECK_GRAPH(PreRunAfterBuild) {
    size_t identity_count = 0U;
    for (const auto &node : graph->GetAllNodes()) {
      if (node->GetType() == IDENTITY) {
        identity_count++;
      }
    }
    EXPECT_EQ(identity_count, 2U);
  };
}

// relu多引用，且relu的另一个输出节点依赖ref算子，不需要插入内置inner Identity
TEST_F(TensorMoveTest, InnerIdentity_Delete1) {
  DEF_GRAPH(g1) {
    auto assign = OP_CFG(ASSIGN)
                      .TensorDesc(FORMAT_ND, DT_FLOAT, {2, 2})
                      .InCnt(2)
                      .OutCnt(1)
                      .InNames({"ref", "value"})
                      .OutNames({"ref"})
                      .Build("assign");
    CHAIN(NODE("data1", DATA)->NODE("relu", RELU)->NODE(assign));
    CHAIN(NODE("relu")->EDGE(0, 0)->NODE("add", ADD)->NODE("net_output", NETOUTPUT));
    CHAIN(NODE("data2", DATA))->EDGE(0, 1)->NODE(assign)->EDGE(0, 1)->NODE("add");
  };

  auto graph = ToGeGraph(g1);
  map<string, string> options;

  Status ret = ge::GELib::Initialize(options);
  EXPECT_EQ(ret, SUCCESS);

  Session session(options);
  ret = session.AddGraph(0, graph, options);
  EXPECT_EQ(ret, SUCCESS);
  // build input tensor
  std::vector<InputTensorInfo> inputs;
  // build_graph through session
  ret = session.BuildGraph(0, inputs);
  EXPECT_EQ(ret, SUCCESS);

  CHECK_GRAPH(PreRunAfterBuild) {
    auto identity = graph->FindFirstNodeMatchType(IDENTITY);
    ASSERT_EQ(identity, nullptr);
  };
}
