/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 ("the License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>

#include <memory>
#include <limits>
#include <vector>

#define private public
#define protected public
#include "common/om2/codegen/task_args_manager/om2_model_args_manager.h"
#include "common/om2/codegen/task_code_builder/task_code_builder.h"
#undef private
#undef protected

#include "common/share_graph.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/utils/graph_utils.h"

namespace ge {
namespace om2 {
namespace {

class ModelArgsManagerUT : public testing::Test {};

ModelArgs MakeModelArgs(std::initializer_list<ModelArgPartition> partitions) {
  ModelArgs model_args;
  model_args.placement = ArgsPlacement::kArgsPlacementHbm;
  model_args.model_args_host_addr = std::make_unique<uint8_t[]>(256U);
  model_args.model_args_device_addr = 0x1000U;
  model_args.model_args_partitions.assign(partitions.begin(), partitions.end());
  return model_args;
}

class FakeTaskCodeBuilder : public TaskCodeBuilder {
 public:
  FakeTaskCodeBuilder(AstBuildContext &ast, int64_t op_index, bool fill_args)
      : TaskCodeBuilder(ast), op_index_(op_index), fill_args_(fill_args) {}

  Status RenderDistHelper(std::vector<DeclNode *> &items) override {
    (void)items;
    return SUCCESS;
  }

  std::string GetFuncName() const override {
    return "fake";
  }

  int64_t ParseOpIndex(const domi::TaskDef &task_def) override {
    (void)task_def;
    return op_index_;
  }

  Status ParseTaskRunParam(const domi::TaskDef &task_def, const RuntimeParam &rts_param, OpDescPtr op_desc,
                           TaskRunParam &task_run_param) override {
    (void)task_def;
    (void)rts_param;
    (void)op_desc;
    if (!fill_args_) {
      return SUCCESS;
    }

    task_run_param.args_descs = {{16, ArgsPlacement::kArgsPlacementHbm}, {8, ArgsPlacement::kArgsPlacementTs}};
    task_run_param.parsed_input_addrs.resize(1U);
    task_run_param.parsed_output_addrs.resize(2U);
    task_run_param.parsed_workspace_addrs.resize(3U);
    task_run_param.persistent_workspace_descs = {{4, ArgsPlacement::kArgsPlacementHostSvm}};
    return SUCCESS;
  }

 private:
  int64_t op_index_;
  bool fill_args_;
};

GeModelPtr MakeArgsManagerDebugModel() {
  auto graph = std::make_shared<ComputeGraph>("om2_model_args_manager_ut");

  GeTensorDesc data_desc(GeShape({1, 2, 3, 4}), FORMAT_ND, DT_FLOAT);
  TensorUtils::SetSize(data_desc, 16U);
  auto data_op = std::make_shared<OpDesc>("data0", DATA);
  data_op->SetId(0);
  (void)data_op->AddInputDesc(data_desc);
  (void)data_op->AddOutputDesc(data_desc);
  data_op->SetWorkspace({0});
  data_op->SetWorkspaceBytes({64});
  auto data = graph->AddNode(data_op);

  auto netoutput_op = std::make_shared<OpDesc>("netoutput0", NETOUTPUT);
  netoutput_op->SetId(1);
  (void)netoutput_op->AddInputDesc(data_desc);
  netoutput_op->SetSrcName({"data0"});
  netoutput_op->SetSrcIndex({0});
  auto netoutput = graph->AddNode(netoutput_op);

  if ((data == nullptr) || (netoutput == nullptr)) {
    return nullptr;
  }

  GraphUtils::AddEdge(data->GetOutDataAnchor(0), netoutput->GetInDataAnchor(0));
  graph->TopologicalSorting();
  graph->SetGraphUnknownFlag(false);

  auto ge_model = MakeShared<GeModel>();
  if (ge_model == nullptr) {
    return nullptr;
  }
  ge_model->SetGraph(graph);
  ge_model->SetName("om2_model_args_manager_ut");
  ge_model->SetVersion(1U);
  (void)AttrUtils::SetInt(ge_model, ATTR_MODEL_MEMORY_SIZE, 0x1000);
  (void)AttrUtils::SetInt(ge_model, ATTR_MODEL_WEIGHT_SIZE, 0x200);
  return ge_model;
}

void FillGenerateArgsDataForProgramGenerator(ModelArgsManager &mam) {
  ModelArgs model_args;
  model_args.placement = ArgsPlacement::kArgsPlacementHbm;
  model_args.model_args_host_addr = std::make_unique<uint8_t[]>(64U);
  model_args.model_args_device_addr = 0x1000U;
  model_args.model_args_partitions = {{UpdateTriggerType::kTriggerByFm, 0, 8},
                                      {UpdateTriggerType::KTriggerByHostInput, 8, 16}};
  mam.model_args_.push_back(std::move(model_args));
  mam.model_args_len_.push_back(24U);
  mam.task_indexes_to_args_.resize(1U);
  mam.task_indexes_to_args_[0][static_cast<size_t>(ArgsPlacement::kArgsPlacementHbm)] = {0x20U, nullptr, 12, 3U};
  mam.task_indexes_to_args_[0][static_cast<size_t>(ArgsPlacement::kArgsPlacementTs)] = {0x40U, nullptr, 5, 7U};

  ModelArgsRefreshInfo refresh_info{};
  refresh_info.id = 0U;
  refresh_info.offset = 4U;
  refresh_info.base_args_offset = 9U;
  refresh_info.placement = ArgsPlacement::kArgsPlacementTs;
  mam.allocation_ids_to_model_args_refresh_infos_addr_all.resize(1U);
  mam.allocation_ids_to_model_args_refresh_infos_addr_all[0].push_back(refresh_info);

  mam.model_adapter_.GetInputIndexToAllocationIds() = {2U, 3U};
  mam.model_adapter_.GetOutputIndexToAllocationIds() = {5U};
}

NodePtr CreateNodeV2(ComputeGraph &graph, const std::string &name, const std::string &type, int in_num, int out_num) {
  OpDescPtr op_desc = MakeShared<OpDesc>(name, type);
  op_desc->SetStreamId(0);
  static int32_t index = 0;
  op_desc->SetId(index++);

  GeTensorDesc tensor(GeShape({1, 2, 3, 4}), FORMAT_ND, DT_FLOAT);
  TensorUtils::SetSize(tensor, 64);
  std::vector<int64_t> input_offset;
  for (int i = 0; i < in_num; i++) {
    input_offset.emplace_back(index * 64 + i * 64);
  }
  op_desc->SetInputOffset(input_offset);

  std::vector<int64_t> output_offset;
  for (int i = 0; i < out_num; i++) {
    op_desc->AddOutputDesc(tensor);
    output_offset.emplace_back(index * 64 + in_num * 64 + i * 64);
  }
  op_desc->SetOutputOffset(output_offset);
  op_desc->SetWorkspace({});
  op_desc->SetWorkspaceBytes({});
  op_desc->SetOpKernelLibName("DNN_VM_RTS_OP_STORE");

  return graph.AddNode(op_desc);
}

GeModelPtr MakeGeModel(const ComputeGraphPtr &graph) {
  auto ge_model = MakeShared<GeModel>();
  ge_model->SetGraph(graph);
  ge_model->SetName("om2_model_adapter_ut");
  ge_model->SetVersion(1U);
  (void)AttrUtils::SetInt(ge_model, ATTR_MODEL_MEMORY_SIZE, 0x1000);
  (void)AttrUtils::SetInt(ge_model, ATTR_MODEL_WEIGHT_SIZE, 0x200);
  (void)AttrUtils::SetInt(ge_model, ATTR_MODEL_ZERO_COPY_MEMORY_SIZE, 0);
  (void)AttrUtils::SetInt(ge_model, MODEL_ATTR_TASK_GEN_BASE_ADDR, 0x1000);
  (void)AttrUtils::SetInt(ge_model, MODEL_ATTR_TASK_GEN_WEIGHT_ADDR, 0x2000);
  (void)AttrUtils::SetInt(ge_model, ATTR_MODEL_TASK_GEN_VAR_ADDR, 0x3000);
  (void)AttrUtils::SetInt(ge_model, ATTR_MODEL_VAR_SIZE, 0x1000);
  (void)AttrUtils::SetInt(ge_model, MODEL_ATTR_HOST_MEMORY_SIZE, 0x1000);
  (void)AttrUtils::SetInt(ge_model, MODEL_ATTR_TASK_GEN_HOST_BASE_ADDR, 0x4000);
  (void)AttrUtils::SetInt(ge_model, MODEL_ATTR_HOST_SVM_SIZE, 0x1000);
  (void)AttrUtils::SetInt(ge_model, MODEL_ATTR_TASK_GEN_HOST_SVM_BASE_ADDR, 0x5000);
  (void)AttrUtils::SetInt(ge_model, ATTR_MODEL_P2P_MEMORY_SIZE, 0x1000);
  (void)AttrUtils::SetInt(ge_model, ATTR_MODEL_SESSION_SCOPE_MEMORY_SIZE, 0x1000);
  return ge_model;
}

TEST_F(ModelArgsManagerUT, ValidateTaskRunParam_DetectsDuplicatePlacement) {
  ModelArgsManager mam;
  EXPECT_NE(mam.ValidateTaskRunParam({{16, ArgsPlacement::kArgsPlacementHbm}, {8, ArgsPlacement::kArgsPlacementHbm}}),
            SUCCESS);
  EXPECT_EQ(mam.ValidateTaskRunParam({{16, ArgsPlacement::kArgsPlacementHbm}, {8, ArgsPlacement::kArgsPlacementTs}}),
            SUCCESS);
}

TEST_F(ModelArgsManagerUT, ConstructH2DCopyParams_CoversAllPolicies) {
  ModelArgsManager mam;

  ModelArgs host_input = MakeModelArgs({{UpdateTriggerType::KTriggerByHostInput, 8, 16}});
  ModelArgsManager::H2DCopyArg cp_arg{};
  EXPECT_EQ(ModelArgsManager::ConstructH2DCopyParams(host_input, ModelArgsManager::KUpdateHostInput, cp_arg), SUCCESS);
  EXPECT_EQ(cp_arg.len, 16U);
  EXPECT_EQ(cp_arg.device_addr, 0x1000U + 8U);
  EXPECT_EQ(cp_arg.host_addr, host_input.model_args_host_addr.get() + 8);

  ModelArgs model_io = MakeModelArgs({{UpdateTriggerType::kTriggerByFmAndIo, 4, 12},
                                      {UpdateTriggerType::kTriggerByFmAndIo, 20, 8},
                                      {UpdateTriggerType::KTriggerByHostInput, 40, 4}});
  cp_arg = {};
  EXPECT_EQ(ModelArgsManager::ConstructH2DCopyParams(model_io, ModelArgsManager::kUpdateModelIo, cp_arg), SUCCESS);
  EXPECT_EQ(cp_arg.len, 24U);
  EXPECT_EQ(cp_arg.device_addr, 0x1000U + 4U);
  EXPECT_EQ(cp_arg.host_addr, model_io.model_args_host_addr.get() + 4);

  ModelArgs fm_and_io = MakeModelArgs({{UpdateTriggerType::kTriggerByFm, 0, 10},
                                       {UpdateTriggerType::kTriggerByFmAndIo, 16, 6},
                                       {UpdateTriggerType::KTriggerByHostInput, 32, 2}});
  cp_arg = {};
  EXPECT_EQ(ModelArgsManager::ConstructH2DCopyParams(fm_and_io, ModelArgsManager::kUpdateFmAndModelIo, cp_arg),
            SUCCESS);
  EXPECT_EQ(cp_arg.len, 18U);
  EXPECT_EQ(cp_arg.device_addr, 0x1000U);
  EXPECT_EQ(cp_arg.host_addr, fm_and_io.model_args_host_addr.get());

  ModelArgs one_time = MakeModelArgs({{UpdateTriggerType::kNoNeedUpdate, 12, 20}});
  cp_arg = {};
  EXPECT_EQ(ModelArgsManager::ConstructH2DCopyParams(one_time, ModelArgsManager::kInitOneTime, cp_arg), SUCCESS);
  EXPECT_EQ(cp_arg.len, 20U);
  EXPECT_EQ(cp_arg.device_addr, 0x1000U);
  EXPECT_EQ(cp_arg.host_addr, one_time.model_args_host_addr.get());

  cp_arg = {};
  EXPECT_EQ(
      ModelArgsManager::ConstructH2DCopyParams(one_time, static_cast<ModelArgsManager::UpdatePolicy>(999), cp_arg),
      FAILED);
}

TEST_F(ModelArgsManagerUT, ConstructH2DCopyParams_ReturnsGraphNotExistWhenNoPartitionMatches) {
  ModelArgs model_args = MakeModelArgs({{UpdateTriggerType::kNoNeedUpdate, 0, 16}});
  ModelArgsManager::H2DCopyArg cp_arg{};
  EXPECT_EQ(ModelArgsManager::ConstructH2DCopyParams(model_args, ModelArgsManager::KUpdateHostInput, cp_arg),
            GE_GRAPH_GRAPH_NOT_EXIST);
  EXPECT_EQ(ModelArgsManager::ConstructH2DCopyParams(model_args, ModelArgsManager::kUpdateModelIo, cp_arg),
            GE_GRAPH_GRAPH_NOT_EXIST);
  EXPECT_EQ(ModelArgsManager::ConstructH2DCopyParams(model_args, ModelArgsManager::kUpdateFmAndModelIo, cp_arg),
            GE_GRAPH_GRAPH_NOT_EXIST);
}

TEST_F(ModelArgsManagerUT, GenerateTriggerTypesToCorrespondingUpdatePolicies_ReflectsRefreshability) {
  ModelArgsManager mam;

  mam.model_adapter_.feature_base_refreshable_ = true;
  auto policies = mam.GenerateTriggerTypesToCorrespondingUpdatePolicies();
  ASSERT_EQ(policies.size(), static_cast<size_t>(UpdateTriggerType::kEnd));
  EXPECT_EQ(policies[0], (ModelArgsManager::TriggerPolicies{ModelArgsManager::kInitOneTime}));
  EXPECT_EQ(policies[1],
            (ModelArgsManager::TriggerPolicies{ModelArgsManager::kUpdateFmAndModelIo, ModelArgsManager::kInitOneTime}));
  EXPECT_EQ(policies[2],
            (ModelArgsManager::TriggerPolicies{ModelArgsManager::kUpdateModelIo, ModelArgsManager::kUpdateFmAndModelIo,
                                               ModelArgsManager::kInitOneTime}));
  EXPECT_EQ(policies[3],
            (ModelArgsManager::TriggerPolicies{ModelArgsManager::KUpdateHostInput, ModelArgsManager::kUpdateModelIo,
                                               ModelArgsManager::kUpdateFmAndModelIo, ModelArgsManager::kInitOneTime}));

  mam.model_adapter_.feature_base_refreshable_ = false;
  policies = mam.GenerateTriggerTypesToCorrespondingUpdatePolicies();
  EXPECT_EQ(policies[1], (ModelArgsManager::TriggerPolicies{ModelArgsManager::kUpdatePolicyEnd}));
}

TEST_F(ModelArgsManagerUT, InitForUpdate_SetsPolicyAndLengthTables) {
  ModelArgsManager mam;
  mam.model_adapter_.GetLogicalMemAllocation() = {{0U, 0U, 11U, MemAllocation::FEATURE_MAP, 0U, 0U, 0U, 0U},
                                                  {1U, 0x100U, 22U, MemAllocation::FEATURE_MAP, 0U, 0U, 0U, 0U},
                                                  {2U, 0x200U, 33U, MemAllocation::ABSOLUTE, 0U, 0U, 0U, 0U}};
  mam.model_adapter_.fm_mem_allocations_start_id_ = 1U;
  mam.model_adapter_.logical_fm_mem_allocations_size_ = 1U;

  mam.InitForUpdate();

  EXPECT_EQ(mam.id_to_len_, (std::vector<uint64_t>{11U, 22U, 33U}));
  EXPECT_EQ(mam.id_to_plicy_, (std::vector<uint32_t>{static_cast<uint32_t>(ModelArgsManager::kUpdateModelIo),
                                                     static_cast<uint32_t>(ModelArgsManager::kUpdateFmAndModelIo),
                                                     static_cast<uint32_t>(ModelArgsManager::kInitOneTime)}));
  ASSERT_EQ(mam.last_bases_.size(), 3U);
  EXPECT_EQ(mam.last_bases_[0], std::numeric_limits<uint64_t>::max());
  EXPECT_EQ(mam.last_bases_[1], std::numeric_limits<uint64_t>::max());
  EXPECT_EQ(mam.last_bases_[2], std::numeric_limits<uint64_t>::max());
}

TEST_F(ModelArgsManagerUT, GenerateArgsDataForProgramGenerator_CopiesSemanticTables) {
  ModelArgsManager mam;
  FillGenerateArgsDataForProgramGenerator(mam);

  Om2CodegenModel codegen_model;
  ASSERT_EQ(mam.GenerateArgsDataForProgramGenerator(codegen_model), SUCCESS);

  ASSERT_EQ(codegen_model.args_table.model_args_semantic.size(), 1U);
  EXPECT_EQ(codegen_model.args_table.model_args_semantic[0].placement, ArgsPlacement::kArgsPlacementHbm);
  EXPECT_EQ(codegen_model.args_table.model_args_semantic[0].len, 24U);
  ASSERT_EQ(codegen_model.args_table.model_args_semantic[0].partitions.size(), 2U);
  EXPECT_EQ(codegen_model.args_table.model_args_semantic[0].partitions[1].offset, 8);
  EXPECT_EQ(codegen_model.args_table.task_indexes_to_args_semantic.size(), 1U);
  EXPECT_EQ(
      codegen_model.args_table.task_indexes_to_args_semantic[0][static_cast<size_t>(ArgsPlacement::kArgsPlacementHbm)]
          .offset,
      3U);
  EXPECT_EQ(
      codegen_model.args_table.task_indexes_to_args_semantic[0][static_cast<size_t>(ArgsPlacement::kArgsPlacementHbm)]
          .len,
      12);
  EXPECT_EQ(
      codegen_model.args_table.task_indexes_to_args_semantic[0][static_cast<size_t>(ArgsPlacement::kArgsPlacementTs)]
          .offset,
      7U);
  EXPECT_EQ(codegen_model.args_table.allocation_ids_to_model_args_refresh_infos_addr_all_semantic.size(), 1U);
  ASSERT_EQ(codegen_model.args_table.allocation_ids_to_model_args_refresh_infos_addr_all_semantic[0].size(), 1U);
  EXPECT_EQ(
      codegen_model.args_table.allocation_ids_to_model_args_refresh_infos_addr_all_semantic[0][0].base_args_offset, 9U);
  EXPECT_EQ(codegen_model.args_table.allocation_ids_to_model_args_refresh_infos_addr_all_semantic[0][0].offset, 4U);
  EXPECT_EQ(codegen_model.args_table.allocation_ids_to_model_args_refresh_infos_addr_all_semantic[0][0].placement,
            ArgsPlacement::kArgsPlacementTs);
  EXPECT_EQ(codegen_model.args_table.input_index_to_allocation_ids, (std::vector<uint32_t>{2U, 3U}));
  EXPECT_EQ(codegen_model.args_table.output_index_to_allocation_ids, (std::vector<uint32_t>{5U}));
}

TEST_F(ModelArgsManagerUT, ModelAdapter_Init_CoversMainPaths) {
  auto graph = std::make_shared<ComputeGraph>("graph");
  auto data1 = CreateNodeV2(*graph, "data1", DATA, 0, 1);
  auto netoutput = CreateNodeV2(*graph, "NetOutput", NETOUTPUT, 3, 0);
  GeTensorDesc data_in_desc(GeShape({1, 2, 3, 4}), FORMAT_ND, DT_FLOAT);
  TensorUtils::SetSize(data_in_desc, 64);
  data1->GetOpDesc()->AddInputDesc(data_in_desc);
  GeTensorDesc net_in_desc(GeShape({1, 2, 3, 4}), FORMAT_ND, DT_FLOAT);
  TensorUtils::SetSize(net_in_desc, 64);
  netoutput->GetOpDesc()->AddInputDesc(net_in_desc);
  netoutput->GetOpDesc()->AddInputDesc(net_in_desc);
  netoutput->GetOpDesc()->AddInputDesc(net_in_desc);
  netoutput->AddLinkFrom(data1);
  netoutput->AddLinkFrom(data1);
  netoutput->AddLinkFrom(data1);
  netoutput->GetOpDesc()->SetSrcName({"data1", "data1", "data1"});
  netoutput->GetOpDesc()->SetSrcIndex({0, 0, 0});
  ASSERT_NE(data1, nullptr);
  ASSERT_NE(netoutput, nullptr);

  (void)AttrUtils::SetBool(data1->GetOpDesc()->MutableOutputDesc(0), ATTR_IS_ZERO_COPY_BLOCK, true);
  (void)AttrUtils::SetBool(netoutput->GetOpDesc()->MutableInputDesc(0), ATTR_NAME_TENSOR_NO_TILING_MEM_TYPE, true);
  (void)AttrUtils::SetInt(netoutput->GetOpDesc()->MutableInputDesc(0), ATTR_NAME_SPECIAL_INPUT_SIZE, 64);
  auto ge_model = MakeGeModel(graph);
  (void)AttrUtils::SetListStr(ge_model, ATTR_MODEL_OUT_NODES_NAME, std::vector<std::string>{"out0", "out1", "out2"});
  (void)AttrUtils::SetBool(netoutput->GetOpDesc(), ATTR_GETNEXT_SINK_DYNMAIC, true);
  (void)AttrUtils::SetListStr(netoutput->GetOpDesc(), ATTR_NAME_DYNAMIC_OUTPUT_DIMS,
                              std::vector<std::string>{"1,2,3,4"});

  ModelAdapter adapter;
  EXPECT_NE(adapter.Init(ge_model), SUCCESS);
  EXPECT_EQ(adapter.GetOmName(), "om2_model_adapter_ut");
  EXPECT_TRUE(adapter.IsFeatureBaseRefreshable());
  EXPECT_FALSE(adapter.GetPhysicalMemoryRefreshable());
}

TEST_F(ModelArgsManagerUT, ModelAdapter_InitHelpers_CoverBranches) {
  auto graph = gert::ShareGraph::BuildWithKnownSubgraph();
  std::vector<ComputeGraphPtr> subgraphs;
  ASSERT_EQ(GraphUtils::GetSubgraphsRecursively(graph, subgraphs), SUCCESS);
  ASSERT_FALSE(subgraphs.empty());
  auto subgraph = subgraphs.front();
  auto sub_data = subgraph->FindNode("data1");
  auto sub_netoutput = subgraph->FindNode("netoutput_sub");
  ASSERT_NE(sub_data, nullptr);
  ASSERT_NE(sub_netoutput, nullptr);

  ModelAdapter adapter;
  std::map<uint32_t, OpDescPtr> data_by_index;
  std::set<uint64_t> addrs;
  uint32_t data_index = 0U;
  EXPECT_EQ(adapter.InitDataOp(graph, sub_data, data_index, data_by_index, addrs), SUCCESS);
  EXPECT_TRUE(data_by_index.empty());

  std::vector<OpDescPtr> output_ops;
  std::set<uint64_t> output_addrs;
  EXPECT_EQ(adapter.InitNetOutput(graph, sub_netoutput, output_ops, output_addrs), SUCCESS);
  EXPECT_TRUE(output_ops.empty());

  auto root_data = graph->FindNode("data_a");
  ASSERT_NE(root_data, nullptr);
  EXPECT_EQ(adapter.InitDataOp(graph, root_data, data_index, data_by_index, addrs), SUCCESS);
  EXPECT_FALSE(data_by_index.empty());
}

TEST_F(ModelArgsManagerUT, ModelAdapter_PrivateHelpers_MemoryAndInput_CoverBranches) {
  ModelAdapter adapter;

  int64_t total_useful_size = 0;
  adapter.runtime_param_.mem_size = 0x1000;
  adapter.runtime_param_.zero_copy_size = 0x2000;
  EXPECT_EQ(adapter.GetTotalMemSizeExcludeZeroCopy(total_useful_size), FAILED);
  adapter.runtime_param_.zero_copy_size = 0x200;
  EXPECT_EQ(adapter.GetTotalMemSizeExcludeZeroCopy(total_useful_size), SUCCESS);
  EXPECT_EQ(total_useful_size, 0xE00);

  auto data = std::make_shared<OpDesc>("data", DATA);
  GeTensorDesc input_desc(GeShape({1, 2, 3, 4}), FORMAT_NCHW, DT_FLOAT);
  TensorUtils::SetSize(input_desc, 128);
  data->AddInputDesc(input_desc);
  GeTensorDesc output_desc(GeShape({1, 2, 3, 4}), FORMAT_FRACTAL_Z, DT_FLOAT);
  TensorUtils::SetSize(output_desc, 128);
  data->AddOutputDesc(output_desc);
  (void)AttrUtils::SetInt(data->MutableOutputDesc(0), ATTR_NAME_SPECIAL_INPUT_SIZE, 256);

  EXPECT_EQ(adapter.InitInputDescInfo(data), SUCCESS);
  ASSERT_EQ(adapter.origin_input_descs_.size(), 1U);
  EXPECT_EQ(adapter.origin_input_descs_[0].size, 256U);
}

TEST_F(ModelArgsManagerUT, ModelAdapter_PrivateHelpers_Output_CoverBranches) {
  ModelAdapter adapter;

  auto netoutput = std::make_shared<OpDesc>("netoutput", NETOUTPUT);
  GeTensorDesc net_in0(GeShape({1, 2, 3, 4}), FORMAT_FRACTAL_Z, DT_FLOAT);
  GeTensorDesc net_in1(GeShape({1, 2, 3, 4}), FORMAT_ND, DT_FLOAT);
  TensorUtils::SetSize(net_in0, 128);
  TensorUtils::SetSize(net_in1, 128);
  (void)AttrUtils::SetInt(net_in0, ATTR_NAME_SPECIAL_OUTPUT_SIZE, 512);
  netoutput->AddInputDesc(net_in0);
  netoutput->AddInputDesc(net_in1);
  netoutput->SetSrcName({"data", "data"});
  netoutput->SetSrcIndex({0, 0});

  InputOutputDescInfo output_info;
  uint32_t format_result = 0U;
  EXPECT_NO_THROW(adapter.CreateOutput(0U, netoutput, output_info, format_result));
  EXPECT_EQ(format_result, static_cast<uint32_t>(FORMAT_HWCN));
  EXPECT_EQ(output_info.size, 512U);

  std::vector<std::string> out_node_name = {"out0", "out1"};
  EXPECT_EQ(adapter.InitOutputDescInfo(netoutput, out_node_name), SUCCESS);
  EXPECT_EQ(adapter.output_descs_.size(), 2U);
  EXPECT_EQ(adapter.output_descs_[0].name, "out0:0");

  ModelAdapter default_name_adapter;
  EXPECT_EQ(default_name_adapter.InitOutputDescInfo(netoutput, {"out0"}), SUCCESS);
  EXPECT_EQ(default_name_adapter.output_descs_[0].name, "output_0_data_0");

  EXPECT_EQ(adapter.InitOutputTensorInfo(netoutput), SUCCESS);
  ASSERT_EQ(adapter.output_buffer_size_.size(), 2U);
  EXPECT_EQ(adapter.output_buffer_size_[0], 96);
  EXPECT_EQ(adapter.output_no_tiling_flag_[0], false);
}

TEST_F(ModelArgsManagerUT, ModelArgsManager_DebugAndUpdateData_CoverBranches) {
  int32_t event_level = 0;
  const int32_t old_level = dlog_getlevel(GE_MODULE_NAME, &event_level);
  dlog_setlevel(GE_MODULE_NAME, DLOG_DEBUG, event_level);

  auto ge_model = MakeArgsManagerDebugModel();
  ASSERT_NE(ge_model, nullptr);

  ModelArgsManager mam;
  mam.model_adapter_.ge_model_ = ge_model;
  mam.model_adapter_.op_list_[0] = ge_model->GetGraph()->FindNode("data0")->GetOpDesc();

  AstContext ast_ctx;
  AstBuildContext ast(ast_ctx);
  std::vector<TaskCodeBuilderPtr> task_list;
  task_list.emplace_back(std::make_shared<FakeTaskCodeBuilder>(ast, -1, false));
  task_list.emplace_back(std::make_shared<FakeTaskCodeBuilder>(ast, 0, true));
  mam.task_list_ptr_ = &task_list;

  domi::ModelTaskDef model_task_def;
  (void)model_task_def.add_task();
  (void)model_task_def.add_task();

  std::vector<TaskRunParam> task_indexes_to_run_param(2U);
  TaskNodeMap task_node_map;
  ASSERT_EQ(task_node_map.Init(mam.model_adapter_.GetCompiledComputeGraph(), 2U), SUCCESS);
  EXPECT_EQ(mam.ParseModelTaskDef(model_task_def, task_indexes_to_run_param, task_node_map), SUCCESS);

  ASSERT_EQ(task_node_map.AddRelation(1U, 0), SUCCESS);

  ModelArgsLayoutPlannedResult layout;
  layout.task_indexes_to_arg_results.resize(2U);
  layout.task_indexes_to_arg_results[0].push_back(
      {ArgsPlacement::kArgsPlacementHbm, UpdateTriggerType::kTriggerByFm, 0});
  layout.task_indexes_to_arg_results[1].push_back(
      {ArgsPlacement::kArgsPlacementHbm, UpdateTriggerType::kTriggerByFmAndIo, 0});

  task_indexes_to_run_param[0].args_descs = {{16, ArgsPlacement::kArgsPlacementHbm}};
  task_indexes_to_run_param[1].args_descs = {{8, ArgsPlacement::kArgsPlacementTs}};

  mam.model_args_.clear();
  mam.model_args_len_.clear();
  mam.task_indexes_to_args_.clear();
  EXPECT_EQ(mam.ConstructUpdateData(task_node_map, layout, task_indexes_to_run_param, mam.task_indexes_to_args_),
            SUCCESS);

  dlog_setlevel(GE_MODULE_NAME, old_level, event_level);
}

TEST_F(ModelArgsManagerUT, ModelArgsManager_FixedAddrs_CoverBranches) {
  auto ge_model = MakeArgsManagerDebugModel();
  ASSERT_NE(ge_model, nullptr);

  ModelArgsManager mam;
  mam.model_adapter_.ge_model_ = ge_model;
  mam.model_adapter_.op_list_[0] = ge_model->GetGraph()->FindNode("data0")->GetOpDesc();

  AstContext ast_ctx;
  AstBuildContext ast(ast_ctx);
  std::vector<TaskCodeBuilderPtr> task_list;
  task_list.emplace_back(std::make_shared<FakeTaskCodeBuilder>(ast, -1, false));
  task_list.emplace_back(std::make_shared<FakeTaskCodeBuilder>(ast, 0, true));
  mam.task_list_ptr_ = &task_list;

  domi::ModelTaskDef model_task_def;
  (void)model_task_def.add_task();
  (void)model_task_def.add_task();

  std::vector<TaskRunParam> task_indexes_to_run_param(2U);
  TaskNodeMap task_node_map;
  ASSERT_EQ(task_node_map.Init(mam.model_adapter_.GetCompiledComputeGraph(), 2U), SUCCESS);
  EXPECT_EQ(mam.ParseModelTaskDef(model_task_def, task_indexes_to_run_param, task_node_map), SUCCESS);
  ASSERT_EQ(task_node_map.AddRelation(1U, 0), SUCCESS);

  TaskArgsRefreshTypeClassifier::FixedAddrs fixed_addrs(3);
  fixed_addrs[0].push_back({1U, 0U, TaskArgsRefreshTypeClassifier::kInput});
  fixed_addrs[1].push_back({1U, 0U, TaskArgsRefreshTypeClassifier::kOutput});
  fixed_addrs[2].push_back({1U, 0U, TaskArgsRefreshTypeClassifier::kWorkspace});

  EXPECT_EQ(mam.AllocFixedAddrs(task_node_map, fixed_addrs), SUCCESS);

  TaskArgsRefreshTypeClassifier::FixedAddrs bad_addrs(1);
  bad_addrs[0].push_back({1U, 0U, static_cast<TaskArgsRefreshTypeClassifier::IndexType>(999)});
  EXPECT_NE(mam.AllocFixedAddrs(task_node_map, bad_addrs), SUCCESS);
}

}  // namespace
}  // namespace om2
}  // namespace ge
