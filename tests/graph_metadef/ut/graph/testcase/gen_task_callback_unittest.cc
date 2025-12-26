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
#include "graph/kernel_launch_info.h"
#include "graph/arg_desc_info.h"
#include "runtime/rt_model.h"
#include "proto/task.pb.h"
#include "exe_graph/runtime/exe_res_generation_context.h"
#include "exe_graph/lowering/exe_res_generation_ctx_builder.h"
#include "ge/framework/common/taskdown_common.h"
#include "graph/utils/args_format_desc_utils.h"
#include "graph/compute_graph.h"
#include "graph/utils/graph_utils.h"
#include "common/checker.h"
#include "graph/debug/ge_attr_define.h"
#include "common/opskernel/ops_kernel_info_types.h"

namespace ge {
namespace {
ComputeGraphPtr CreateMc2NodeGraph() {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  OpDescPtr x1_desc = std::make_shared<OpDesc>("x1", "Data");
  OpDescPtr x2_desc = std::make_shared<OpDesc>("x2", "Data");
  OpDescPtr bias_desc = std::make_shared<OpDesc>("bias", "Data");
  OpDescPtr all_gather_matmul_desc = std::make_shared<OpDesc>("mc2", "AllGatherMatmul");
  OpDescPtr net_output_desc = std::make_shared<OpDesc>("output", "NetOutput");

  // add descriptor
  ge::GeShape shape1({2,4});
  GeTensorDesc tensor_desc1(shape1, ge::FORMAT_ND, ge::DT_FLOAT16);
  tensor_desc1.SetOriginFormat(ge::FORMAT_ND);
  tensor_desc1.SetOriginDataType(ge::DT_FLOAT16);
  tensor_desc1.SetOriginShape(shape1);

  ge::GeShape shape2({4,3});
  GeTensorDesc tensor_desc2(shape2, ge::FORMAT_ND, ge::DT_FLOAT16);
  tensor_desc2.SetOriginFormat(ge::FORMAT_ND);
  tensor_desc2.SetOriginDataType(ge::DT_FLOAT16);
  tensor_desc2.SetOriginShape(shape2);

  ge::GeShape shape3({3});
  GeTensorDesc tensor_desc3(shape3, ge::FORMAT_ND, ge::DT_FLOAT16);
  tensor_desc3.SetOriginFormat(ge::FORMAT_ND);
  tensor_desc3.SetOriginDataType(ge::DT_FLOAT16);
  tensor_desc3.SetOriginShape(shape3);

  ge::GeShape shape4({2, 3});
  GeTensorDesc tensor_desc4(shape4, ge::FORMAT_ND, ge::DT_FLOAT16);
  tensor_desc4.SetOriginFormat(ge::FORMAT_ND);
  tensor_desc4.SetOriginDataType(ge::DT_FLOAT16);
  tensor_desc4.SetOriginShape(shape4);

  x1_desc->AddOutputDesc(tensor_desc1);
  x2_desc->AddOutputDesc(tensor_desc2);
  bias_desc->AddOutputDesc(tensor_desc3);

  all_gather_matmul_desc->AddInputDesc(tensor_desc1);
  all_gather_matmul_desc->AddInputDesc(tensor_desc2);
  all_gather_matmul_desc->AddInputDesc(tensor_desc3);
  all_gather_matmul_desc->AddOutputDesc(tensor_desc4);
  all_gather_matmul_desc->AddOutputDesc(tensor_desc4);
  all_gather_matmul_desc->AppendIrInput("x1", ge::kIrInputRequired);
  all_gather_matmul_desc->AppendIrInput("x2", ge::kIrInputRequired);
  all_gather_matmul_desc->AppendIrInput("bias", ge::kIrInputOptional);
  all_gather_matmul_desc->AppendIrOutput("y", ge::kIrOutputRequired);
  all_gather_matmul_desc->AppendIrOutput("gather_out", ge::kIrOutputRequired);

  net_output_desc->AddInputDesc(tensor_desc4);
  net_output_desc->AddInputDesc(tensor_desc4);
  // create nodes
  NodePtr x1_node = graph->AddNode(x1_desc);
  NodePtr x2_node = graph->AddNode(x2_desc);
  NodePtr bias_node = graph->AddNode(bias_desc);
  NodePtr mc2_node = graph->AddNode(all_gather_matmul_desc);
  NodePtr output_node = graph->AddNode(net_output_desc);

  ge::GraphUtils::AddEdge(x1_node->GetOutDataAnchor(0), mc2_node->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(x2_node->GetOutDataAnchor(0), mc2_node->GetInDataAnchor(1));
  ge::GraphUtils::AddEdge(bias_node->GetOutDataAnchor(0), mc2_node->GetInDataAnchor(2));
  ge::GraphUtils::AddEdge(mc2_node->GetOutDataAnchor(0), output_node->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(mc2_node->GetOutDataAnchor(1), output_node->GetInDataAnchor(1));

  all_gather_matmul_desc->SetStreamId(2);
  all_gather_matmul_desc->SetId(4);
  std::vector<int64_t> ori_work_sizes{22,33,44};
  all_gather_matmul_desc->SetWorkspaceBytes(ori_work_sizes);
  return graph;
}

gert::ExeResGenerationCtxHolderPtr CreateNodeExeResContext(const NodePtr &node) {
  gert::ExeResGenerationCtxBuilder exe_ctx_builder;
  auto res_ptr_holder = exe_ctx_builder.CreateOpExeContext(*node);
  auto op_exe_res_ctx = reinterpret_cast<gert::ExeResGenerationContext *>(res_ptr_holder->GetKernelContext());
  std::vector<gert::StreamInfo> stream_info_vec;
  gert::StreamInfo si_1;
  si_1.name = "aicpu kfc server";
  si_1.reuse_key = "kfc_stream";
  si_1.depend_value_input_indices = {};
  si_1.required = true;
  stream_info_vec.emplace_back(si_1);
  op_exe_res_ctx->SetAttachedStreamInfos(stream_info_vec);
  std::vector<ge::GeAttrValue::NAMED_ATTRS> stream_info_attrs;
  (void)ge::AttrUtils::GetListNamedAttrs(node->GetOpDesc(), ge::ATTR_NAME_ATTACHED_STREAM_INFO_LIST,
      stream_info_attrs);
  (void)ge::AttrUtils::SetInt(stream_info_attrs.front(), ge::ATTR_NAME_ATTACHED_RESOURCE_ID, 4);
  (void)ge::AttrUtils::SetListNamedAttrs(node->GetOpDesc(), ge::ATTR_NAME_ATTACHED_STREAM_INFO_LIST,
      stream_info_attrs);
  return res_ptr_holder;
}

struct HcclCommParamDesc {
  uint64_t version : 4;
  uint64_t group_num : 4;
  uint64_t has_ffts : 1;
  uint64_t tiling_off : 7;
  uint64_t is_dyn : 48;
};

graphStatus Mc2GenTaskCallback(const gert::ExeResGenerationContext *context,
    std::vector<std::vector<uint8_t>> &tasks) {
  GE_ASSERT_NOTNULL(context);
  GE_ASSERT_TRUE(tasks.size() == 1UL);
  auto aicore_index = 0;
  // 获取attach流id
  auto stream_infos = context->GetAttachedStreamInfos();
  GE_ASSERT_TRUE(!stream_infos.empty());
  const int64_t attach_stream_id = stream_infos[0].stream_id;
  const int64_t stream_id = context->GetStreamId();
  // 创建WaitTask
  auto wait_task = KernelLaunchInfo::CreateHcomWaitTask(context);
  wait_task.SetStreamId(attach_stream_id);
  tasks.insert(tasks.begin() + aicore_index, wait_task.Serialize());
  aicore_index++;
  // 设置aicpu任务
  auto aicpu_task = KernelLaunchInfo::CreateAicpuKfcTask(context,
      "libccl_kernel.so", "RunAicpuKfcSrvLaunch");
  size_t input_size = context->GetComputeNodeInfo()->GetIrInputsNum();
  size_t output_size = context->GetComputeNodeInfo()->GetIrOutputsNum();
  const size_t offset = 3UL;
  union {
    HcclCommParamDesc hccl_desc;
    uint64_t custom_value;
  } desc;
  desc.hccl_desc.version = 1;
  desc.hccl_desc.group_num = 1;
  desc.hccl_desc.has_ffts = 0;
  desc.hccl_desc.tiling_off = offset + input_size + output_size;
  desc.hccl_desc.is_dyn = 0;
  std::vector<ArgDescInfo> aicpu_args_format;
  aicpu_args_format.emplace_back(ArgDescInfo::CreateCustomValue(desc.custom_value));
  aicpu_args_format.emplace_back(ArgDescInfo::CreateHiddenInput(HiddenInputSubType::kHcom));
  aicpu_args_format.emplace_back(ArgDescInfo(ArgDescType::kIrInput, 0));
  for (size_t i = 1; i < input_size; i++) {
    aicpu_args_format.emplace_back(ArgDescInfo::CreateCustomValue(0));
  }
  for (size_t i = 0; i < output_size; i++) {
    aicpu_args_format.emplace_back(ArgDescInfo(ArgDescType::kIrOutput, i));
  }
  aicpu_args_format.emplace_back(ArgDescInfo(ArgDescType::kWorkspace));
  aicpu_args_format.emplace_back(ArgDescInfo(ArgDescType::kTiling));
  aicpu_task.SetArgsFormat(ArgsFormatSerializer::Serialize(aicpu_args_format).GetString());
  aicpu_task.SetStreamId(attach_stream_id);
  tasks.insert(tasks.begin() + aicore_index, aicpu_task.Serialize());
  aicore_index++;
  // 创建RecordTask
  auto record_task = KernelLaunchInfo::CreateHcomRecordTask(context);
  record_task.SetStreamId(stream_id);
  tasks.insert(tasks.begin() + aicore_index, record_task.Serialize());
  aicore_index++;
  // 更改原AICORE任务的argsformat
  auto aicore_task = KernelLaunchInfo::LoadFromData(context, tasks.back());
  auto aicore_args_format_str = aicore_task.GetArgsFormat();
  auto aicore_args_format = ArgsFormatSerializer::Deserialize(aicore_args_format_str);
  size_t i = 0UL;
  for (; i < aicore_args_format.size(); i++) {
    if (aicore_args_format[i].GetType() == ArgDescType::kIrInput ||
        aicore_args_format[i].GetType() == ArgDescType::kInputInstance) {
      break;
    }
  }
  aicore_args_format.insert(aicore_args_format.begin() + i, ArgDescInfo::CreateHiddenInput(HiddenInputSubType::kHcom));
  aicore_task.SetArgsFormat(ArgsFormatSerializer::Serialize(aicore_args_format).GetString());
  tasks.back() = aicore_task.Serialize();
  return SUCCESS;
}
}
class TestGenTaskCallback : public testing::Test {
 protected:
  void SetUp() {}
  void TearDown() {}
};
// 验证使用kernel_def的mc2算子在GenTaskCallback函数中构造taskDef的功能
TEST_F(TestGenTaskCallback, TestNormalMc2NodeGenTaskCallback) {
  auto graph = CreateMc2NodeGraph();
  auto mc2_node = graph->FindNode("mc2");
  auto res_context_holder = CreateNodeExeResContext(mc2_node);
  auto op_exe_res_ctx = reinterpret_cast<gert::ExeResGenerationContext *>(res_context_holder->GetKernelContext());
  domi::TaskDef aicore_task_def;
  aicore_task_def.set_type(static_cast<uint32_t>(ModelTaskType::MODEL_TASK_KERNEL));
  aicore_task_def.set_id(op_exe_res_ctx->GetOpId());
  aicore_task_def.set_stream_id(op_exe_res_ctx->GetStreamId());
  auto kernel_def = aicore_task_def.mutable_kernel();
  kernel_def->set_block_dim(32);
  kernel_def->set_schedule_mode(0);
  auto kernel_context = kernel_def->mutable_context();
  kernel_context->set_kernel_type(static_cast<uint32_t>(ccKernelType::TE_AI_CORE));
  kernel_context->set_op_index(op_exe_res_ctx->GetOpId());
  std::vector<ArgDesc> args;
  size_t input_size = op_exe_res_ctx->GetComputeNodeInfo()->GetIrInputsNum();
  size_t output_size = op_exe_res_ctx->GetComputeNodeInfo()->GetIrOutputsNum();
  for (size_t i = 0UL; i < input_size; i++) {
    ArgsFormatDescUtils::Append(args, AddrType::INPUT, i);
  }
  for (size_t i = 0UL; i < output_size; i++) {
    ArgsFormatDescUtils::Append(args, AddrType::OUTPUT, i);
  }
  ArgsFormatDescUtils::Append(args, AddrType::WORKSPACE);
  ArgsFormatDescUtils::Append(args, AddrType::TILING);
  kernel_context->set_args_format(ArgsFormatDescUtils::Serialize(args));
  // 序列化
  std::vector<std::vector<uint8_t>> tasks;
  auto buffer_size = aicore_task_def.ByteSizeLong();
  std::vector<uint8_t> buffer(buffer_size, 0);
  aicore_task_def.SerializeToArray(buffer.data(), buffer_size);
  tasks.emplace_back(buffer);
  // 执行mc2的gentaskcallback
  EXPECT_EQ(Mc2GenTaskCallback(op_exe_res_ctx, tasks), SUCCESS);
  EXPECT_EQ(tasks.size(), 4UL);
  // 校验wait算子的结果
  domi::TaskDef wait_task;
  wait_task.ParseFromArray(tasks[0].data(), tasks[0].size());
  EXPECT_EQ(wait_task.id(), 4);
  EXPECT_EQ(wait_task.notify_id(), UINT32_MAX);
  EXPECT_EQ(wait_task.type(), static_cast<uint32_t>(ModelTaskType::MODEL_TASK_NOTIFY_WAIT));
  EXPECT_EQ(wait_task.private_def(), "group");
  EXPECT_EQ(wait_task.stream_id(), 4);
  // 校验aicpu算子结果
  domi::TaskDef aicpu_task;
  aicpu_task.ParseFromArray(tasks[1].data(), tasks[1].size());
  EXPECT_EQ(aicpu_task.type(), static_cast<uint32_t>(ModelTaskType::MODEL_TASK_KERNEL));
  EXPECT_EQ(aicpu_task.stream_id(), 4);
  auto aicpu_kernel_def = aicpu_task.mutable_kernel();
  EXPECT_EQ(aicpu_kernel_def->so_name(), "libccl_kernel.so");
  EXPECT_EQ(aicpu_kernel_def->kernel_name(), "RunAicpuKfcSrvLaunch");
  auto aicpu_kernel_context = aicpu_kernel_def->mutable_context();
  EXPECT_EQ(aicpu_kernel_context->kernel_type(), static_cast<uint32_t>(ccKernelType::AI_CPU_KFC));
  EXPECT_EQ(aicpu_kernel_context->op_index(), 4);
  auto aicpu_args_format = aicpu_kernel_context->args_format();
  EXPECT_EQ(aicpu_args_format, "{#4113}{hi.hcom0*}{i0*}{#0}{#0}{o0*}{o1*}{ws*}{t}");

  // 校验record算子结果
  domi::TaskDef record_task;
  record_task.ParseFromArray(tasks[2].data(), tasks[2].size());
  EXPECT_EQ(record_task.id(), 4);
  EXPECT_EQ(record_task.notify_id(), UINT32_MAX);
  EXPECT_EQ(record_task.type(), static_cast<uint32_t>(ModelTaskType::MODEL_TASK_NOTIFY_RECORD));
  EXPECT_EQ(record_task.private_def(), "group");
  EXPECT_EQ(record_task.stream_id(), 2);
  // 校验aicore算子结果
  domi::TaskDef aicore_task;
  aicore_task.ParseFromArray(tasks[3].data(), tasks[3].size());
  EXPECT_EQ(aicore_task.type(), static_cast<uint32_t>(ModelTaskType::MODEL_TASK_KERNEL));
  EXPECT_EQ(aicore_task.stream_id(), 2);
  auto aicore_kernel_def = aicore_task.mutable_kernel();
  EXPECT_EQ(aicore_kernel_def->block_dim(), 32);
  EXPECT_EQ(aicore_kernel_def->schedule_mode(), 0);
  auto aicore_kernel_context = aicore_kernel_def->mutable_context();
  EXPECT_EQ(aicore_kernel_context->kernel_type(), static_cast<uint32_t>(ccKernelType::TE_AI_CORE));
  EXPECT_EQ(aicore_kernel_context->op_index(), 4);
  auto aicore_args_format = aicore_kernel_context->args_format();
  EXPECT_EQ(aicore_args_format, "{hi.hcom0*}{i0*}{i1*}{i2*}{o0*}{o1*}{ws*}{t}");
}

// 验证使用kernel_def_with_handle的mc2算子在GenTaskCallback函数中构造taskDef的功能
TEST_F(TestGenTaskCallback, TestMc2NodeWithHandleGenTaskCallback) {
  auto graph = CreateMc2NodeGraph();
  auto mc2_node = graph->FindNode("mc2");
  auto res_context_holder = CreateNodeExeResContext(mc2_node);
  auto op_exe_res_ctx = reinterpret_cast<gert::ExeResGenerationContext *>(res_context_holder->GetKernelContext());
  domi::TaskDef aicore_task_def;
  aicore_task_def.set_type(static_cast<uint32_t>(ModelTaskType::MODEL_TASK_ALL_KERNEL));
  aicore_task_def.set_id(op_exe_res_ctx->GetOpId());
  aicore_task_def.set_stream_id(op_exe_res_ctx->GetStreamId());
  auto kernel_def_with_handle = aicore_task_def.mutable_kernel_with_handle();
  kernel_def_with_handle->set_block_dim(32);
  kernel_def_with_handle->set_schedule_mode(0);
  auto kernel_context = kernel_def_with_handle->mutable_context();
  kernel_context->set_kernel_type(static_cast<uint32_t>(ccKernelType::TE_AI_CORE));
  kernel_context->set_op_index(op_exe_res_ctx->GetOpId());
  std::vector<ArgDesc> args;
  size_t input_size = op_exe_res_ctx->GetComputeNodeInfo()->GetIrInputsNum();
  size_t output_size = op_exe_res_ctx->GetComputeNodeInfo()->GetIrOutputsNum();
  for (size_t i = 0UL; i < input_size; i++) {
    ArgsFormatDescUtils::Append(args, AddrType::INPUT, i);
  }
  for (size_t i = 0UL; i < output_size; i++) {
    ArgsFormatDescUtils::Append(args, AddrType::OUTPUT, i);
  }
  ArgsFormatDescUtils::Append(args, AddrType::WORKSPACE);
  ArgsFormatDescUtils::Append(args, AddrType::TILING);
  kernel_context->set_args_format(ArgsFormatDescUtils::Serialize(args));
  // 序列化
  std::vector<std::vector<uint8_t>> tasks;
  auto buffer_size = aicore_task_def.ByteSizeLong();
  std::vector<uint8_t> buffer(buffer_size, 0);
  aicore_task_def.SerializeToArray(buffer.data(), buffer_size);
  tasks.emplace_back(buffer);
  // 执行mc2的gentaskcallback
  EXPECT_EQ(Mc2GenTaskCallback(op_exe_res_ctx, tasks), SUCCESS);
  EXPECT_EQ(tasks.size(), 4UL);
  // 校验wait算子的结果
  domi::TaskDef wait_task;
  wait_task.ParseFromArray(tasks[0].data(), tasks[0].size());
  EXPECT_EQ(wait_task.id(), 4);
  EXPECT_EQ(wait_task.notify_id(), UINT32_MAX);
  EXPECT_EQ(wait_task.type(), static_cast<uint32_t>(ModelTaskType::MODEL_TASK_NOTIFY_WAIT));
  EXPECT_EQ(wait_task.private_def(), "group");
  EXPECT_EQ(wait_task.stream_id(), 4);
  // 校验aicpu算子结果
  domi::TaskDef aicpu_task;
  aicpu_task.ParseFromArray(tasks[1].data(), tasks[1].size());
  EXPECT_EQ(aicpu_task.type(), static_cast<uint32_t>(ModelTaskType::MODEL_TASK_KERNEL));
  EXPECT_EQ(aicpu_task.stream_id(), 4);
  auto aicpu_kernel_def = aicpu_task.mutable_kernel();
  EXPECT_EQ(aicpu_kernel_def->so_name(), "libccl_kernel.so");
  EXPECT_EQ(aicpu_kernel_def->kernel_name(), "RunAicpuKfcSrvLaunch");
  auto aicpu_kernel_context = aicpu_kernel_def->mutable_context();
  EXPECT_EQ(aicpu_kernel_context->kernel_type(), static_cast<uint32_t>(ccKernelType::AI_CPU_KFC));
  EXPECT_EQ(aicpu_kernel_context->op_index(), 4);
  auto aicpu_args_format = aicpu_kernel_context->args_format();
  EXPECT_EQ(aicpu_args_format, "{#4113}{hi.hcom0*}{i0*}{#0}{#0}{o0*}{o1*}{ws*}{t}");

  // 校验record算子结果
  domi::TaskDef record_task;
  record_task.ParseFromArray(tasks[2].data(), tasks[2].size());
  EXPECT_EQ(record_task.id(), 4);
  EXPECT_EQ(record_task.notify_id(), UINT32_MAX);
  EXPECT_EQ(record_task.type(), static_cast<uint32_t>(ModelTaskType::MODEL_TASK_NOTIFY_RECORD));
  EXPECT_EQ(record_task.private_def(), "group");
  EXPECT_EQ(record_task.stream_id(), 2);
  // 校验aicore算子结果
  domi::TaskDef aicore_task;
  aicore_task.ParseFromArray(tasks[3].data(), tasks[3].size());
  EXPECT_EQ(aicore_task.type(), static_cast<uint32_t>(ModelTaskType::MODEL_TASK_ALL_KERNEL));
  EXPECT_EQ(aicore_task.stream_id(), 2);
  auto aicore_kernel_def = aicore_task.mutable_kernel_with_handle();
  EXPECT_EQ(aicore_kernel_def->block_dim(), 32);
  EXPECT_EQ(aicore_kernel_def->schedule_mode(), 0);
  auto aicore_kernel_context = aicore_kernel_def->mutable_context();
  EXPECT_EQ(aicore_kernel_context->kernel_type(), static_cast<uint32_t>(ccKernelType::TE_AI_CORE));
  EXPECT_EQ(aicore_kernel_context->op_index(), 4);
  auto aicore_args_format = aicore_kernel_context->args_format();
  EXPECT_EQ(aicore_args_format, "{hi.hcom0*}{i0*}{i1*}{i2*}{o0*}{o1*}{ws*}{t}");
}

// 验证使用mixL2的mc2算子在GenTaskCallback函数中构造taskDef的功能
TEST_F(TestGenTaskCallback, TestMixL2Mc2NodeGenTaskCallback) {
  auto graph = CreateMc2NodeGraph();
  auto mc2_node = graph->FindNode("mc2");
  auto res_context_holder = CreateNodeExeResContext(mc2_node);
  auto op_exe_res_ctx = reinterpret_cast<gert::ExeResGenerationContext *>(res_context_holder->GetKernelContext());
  domi::TaskDef aicore_task_def;
  aicore_task_def.set_type(static_cast<uint32_t>(ModelTaskType::MODEL_TASK_ALL_KERNEL));
  aicore_task_def.set_id(op_exe_res_ctx->GetOpId());
  aicore_task_def.set_stream_id(op_exe_res_ctx->GetStreamId());
  auto kernel_def_with_handle = aicore_task_def.mutable_kernel_with_handle();
  kernel_def_with_handle->set_block_dim(32);
  kernel_def_with_handle->set_schedule_mode(0);
  auto kernel_context = kernel_def_with_handle->mutable_context();
  kernel_context->set_kernel_type(static_cast<uint32_t>(ccKernelType::TE));
  kernel_context->set_op_index(op_exe_res_ctx->GetOpId());
  std::vector<ArgDesc> args;
  ArgsFormatDescUtils::Append(args, AddrType::FFTS_ADDR);
  size_t input_size = op_exe_res_ctx->GetComputeNodeInfo()->GetIrInputsNum();
  size_t output_size = op_exe_res_ctx->GetComputeNodeInfo()->GetIrOutputsNum();
  for (size_t i = 0UL; i < input_size; i++) {
    ArgsFormatDescUtils::Append(args, AddrType::INPUT, i);
  }
  for (size_t i = 0UL; i < output_size; i++) {
    ArgsFormatDescUtils::Append(args, AddrType::OUTPUT, i);
  }
  ArgsFormatDescUtils::Append(args, AddrType::WORKSPACE);
  ArgsFormatDescUtils::Append(args, AddrType::TILING);
  kernel_context->set_args_format(ArgsFormatDescUtils::Serialize(args));
  // 序列化
  std::vector<std::vector<uint8_t>> tasks;
  auto buffer_size = aicore_task_def.ByteSizeLong();
  std::vector<uint8_t> buffer(buffer_size, 0);
  aicore_task_def.SerializeToArray(buffer.data(), buffer_size);
  tasks.emplace_back(buffer);
  // 执行mc2的gentaskcallback
  EXPECT_EQ(Mc2GenTaskCallback(op_exe_res_ctx, tasks), SUCCESS);
  EXPECT_EQ(tasks.size(), 4UL);
  // 校验wait算子的结果
  domi::TaskDef wait_task;
  wait_task.ParseFromArray(tasks[0].data(), tasks[0].size());
  EXPECT_EQ(wait_task.id(), 4);
  EXPECT_EQ(wait_task.notify_id(), UINT32_MAX);
  EXPECT_EQ(wait_task.type(), static_cast<uint32_t>(ModelTaskType::MODEL_TASK_NOTIFY_WAIT));
  EXPECT_EQ(wait_task.private_def(), "group");
  EXPECT_EQ(wait_task.stream_id(), 4);
  // 校验aicpu算子结果
  domi::TaskDef aicpu_task;
  aicpu_task.ParseFromArray(tasks[1].data(), tasks[1].size());
  EXPECT_EQ(aicpu_task.type(), static_cast<uint32_t>(ModelTaskType::MODEL_TASK_KERNEL));
  EXPECT_EQ(aicpu_task.stream_id(), 4);
  auto aicpu_kernel_def = aicpu_task.mutable_kernel();
  EXPECT_EQ(aicpu_kernel_def->so_name(), "libccl_kernel.so");
  EXPECT_EQ(aicpu_kernel_def->kernel_name(), "RunAicpuKfcSrvLaunch");
  auto aicpu_kernel_context = aicpu_kernel_def->mutable_context();
  EXPECT_EQ(aicpu_kernel_context->kernel_type(), static_cast<uint32_t>(ccKernelType::AI_CPU_KFC));
  EXPECT_EQ(aicpu_kernel_context->op_index(), 4);
  auto aicpu_args_format = aicpu_kernel_context->args_format();
  EXPECT_EQ(aicpu_args_format, "{#4113}{hi.hcom0*}{i0*}{#0}{#0}{o0*}{o1*}{ws*}{t}");

  // 校验record算子结果
  domi::TaskDef record_task;
  record_task.ParseFromArray(tasks[2].data(), tasks[2].size());
  EXPECT_EQ(record_task.id(), 4);
  EXPECT_EQ(record_task.notify_id(), UINT32_MAX);
  EXPECT_EQ(record_task.type(), static_cast<uint32_t>(ModelTaskType::MODEL_TASK_NOTIFY_RECORD));
  EXPECT_EQ(record_task.private_def(), "group");
  EXPECT_EQ(record_task.stream_id(), 2);
  // 校验aicore算子结果
  domi::TaskDef aicore_task;
  aicore_task.ParseFromArray(tasks[3].data(), tasks[3].size());
  EXPECT_EQ(aicore_task.type(), static_cast<uint32_t>(ModelTaskType::MODEL_TASK_ALL_KERNEL));
  EXPECT_EQ(aicore_task.stream_id(), 2);
  auto aicore_kernel_def = aicore_task.mutable_kernel_with_handle();
  EXPECT_EQ(aicore_kernel_def->block_dim(), 32);
  EXPECT_EQ(aicore_kernel_def->schedule_mode(), 0);
  auto aicore_kernel_context = aicore_kernel_def->mutable_context();
  EXPECT_EQ(aicore_kernel_context->kernel_type(), static_cast<uint32_t>(ccKernelType::TE));
  EXPECT_EQ(aicore_kernel_context->op_index(), 4);
  auto aicore_args_format = aicore_kernel_context->args_format();
  EXPECT_EQ(aicore_args_format, "{ffts_addr}{hi.hcom0*}{i0*}{i1*}{i2*}{o0*}{o1*}{ws*}{t}");
}


// 验证使用带有input_instance的mc2算子在GenTaskCallback函数中构造taskDef的功能
TEST_F(TestGenTaskCallback, TestMc2WithInputInstanceNodeGenTaskCallback) {
  auto graph = CreateMc2NodeGraph();
  auto mc2_node = graph->FindNode("mc2");
  auto res_context_holder = CreateNodeExeResContext(mc2_node);
  auto op_exe_res_ctx = reinterpret_cast<gert::ExeResGenerationContext *>(res_context_holder->GetKernelContext());
  domi::TaskDef aicore_task_def;
  aicore_task_def.set_type(static_cast<uint32_t>(ModelTaskType::MODEL_TASK_ALL_KERNEL));
  aicore_task_def.set_id(op_exe_res_ctx->GetOpId());
  aicore_task_def.set_stream_id(op_exe_res_ctx->GetStreamId());
  auto kernel_def_with_handle = aicore_task_def.mutable_kernel_with_handle();
  kernel_def_with_handle->set_block_dim(32);
  kernel_def_with_handle->set_schedule_mode(0);
  auto kernel_context = kernel_def_with_handle->mutable_context();
  kernel_context->set_kernel_type(static_cast<uint32_t>(ccKernelType::TE));
  kernel_context->set_op_index(op_exe_res_ctx->GetOpId());
  std::vector<ArgDesc> args;
  ArgsFormatDescUtils::Append(args, AddrType::FFTS_ADDR);
  size_t input_size = op_exe_res_ctx->GetComputeNodeInfo()->GetIrInputsNum();
  size_t output_size = op_exe_res_ctx->GetComputeNodeInfo()->GetIrOutputsNum();
  for (size_t i = 0UL; i < input_size; i++) {
    ArgsFormatDescUtils::Append(args, AddrType::INPUT_INSTANCE, i);
  }
  for (size_t i = 0UL; i < output_size; i++) {
    ArgsFormatDescUtils::Append(args, AddrType::OUTPUT_INSTANCE, i);
  }
  ArgsFormatDescUtils::Append(args, AddrType::WORKSPACE);
  ArgsFormatDescUtils::Append(args, AddrType::TILING);
  kernel_context->set_args_format(ArgsFormatDescUtils::Serialize(args));
  // 序列化
  std::vector<std::vector<uint8_t>> tasks;
  auto buffer_size = aicore_task_def.ByteSizeLong();
  std::vector<uint8_t> buffer(buffer_size, 0);
  aicore_task_def.SerializeToArray(buffer.data(), buffer_size);
  tasks.emplace_back(buffer);
  // 执行mc2的gentaskcallback
  EXPECT_EQ(Mc2GenTaskCallback(op_exe_res_ctx, tasks), SUCCESS);
  EXPECT_EQ(tasks.size(), 4UL);
  // 校验wait算子的结果
  domi::TaskDef wait_task;
  wait_task.ParseFromArray(tasks[0].data(), tasks[0].size());
  EXPECT_EQ(wait_task.id(), 4);
  EXPECT_EQ(wait_task.notify_id(), UINT32_MAX);
  EXPECT_EQ(wait_task.type(), static_cast<uint32_t>(ModelTaskType::MODEL_TASK_NOTIFY_WAIT));
  EXPECT_EQ(wait_task.private_def(), "group");
  EXPECT_EQ(wait_task.stream_id(), 4);
  // 校验aicpu算子结果
  domi::TaskDef aicpu_task;
  aicpu_task.ParseFromArray(tasks[1].data(), tasks[1].size());
  EXPECT_EQ(aicpu_task.type(), static_cast<uint32_t>(ModelTaskType::MODEL_TASK_KERNEL));
  EXPECT_EQ(aicpu_task.stream_id(), 4);
  auto aicpu_kernel_def = aicpu_task.mutable_kernel();
  EXPECT_EQ(aicpu_kernel_def->so_name(), "libccl_kernel.so");
  EXPECT_EQ(aicpu_kernel_def->kernel_name(), "RunAicpuKfcSrvLaunch");
  auto aicpu_kernel_context = aicpu_kernel_def->mutable_context();
  EXPECT_EQ(aicpu_kernel_context->kernel_type(), static_cast<uint32_t>(ccKernelType::AI_CPU_KFC));
  EXPECT_EQ(aicpu_kernel_context->op_index(), 4);
  auto aicpu_args_format = aicpu_kernel_context->args_format();
  EXPECT_EQ(aicpu_args_format, "{#4113}{hi.hcom0*}{i0*}{#0}{#0}{o0*}{o1*}{ws*}{t}");

  // 校验record算子结果
  domi::TaskDef record_task;
  record_task.ParseFromArray(tasks[2].data(), tasks[2].size());
  EXPECT_EQ(record_task.id(), 4);
  EXPECT_EQ(record_task.notify_id(), UINT32_MAX);
  EXPECT_EQ(record_task.type(), static_cast<uint32_t>(ModelTaskType::MODEL_TASK_NOTIFY_RECORD));
  EXPECT_EQ(record_task.private_def(), "group");
  EXPECT_EQ(record_task.stream_id(), 2);
  // 校验aicore算子结果
  domi::TaskDef aicore_task;
  aicore_task.ParseFromArray(tasks[3].data(), tasks[3].size());
  EXPECT_EQ(aicore_task.type(), static_cast<uint32_t>(ModelTaskType::MODEL_TASK_ALL_KERNEL));
  EXPECT_EQ(aicore_task.stream_id(), 2);
  auto aicore_kernel_def = aicore_task.mutable_kernel_with_handle();
  EXPECT_EQ(aicore_kernel_def->block_dim(), 32);
  EXPECT_EQ(aicore_kernel_def->schedule_mode(), 0);
  auto aicore_kernel_context = aicore_kernel_def->mutable_context();
  EXPECT_EQ(aicore_kernel_context->kernel_type(), static_cast<uint32_t>(ccKernelType::TE));
  EXPECT_EQ(aicore_kernel_context->op_index(), 4);
  auto aicore_args_format = aicore_kernel_context->args_format();
  EXPECT_EQ(aicore_args_format,
      "{ffts_addr}{hi.hcom0*}{i_instance0*}{i_instance1*}{i_instance2*}{o_instance0*}{o_instance1*}{ws*}{t}");
}

// 验证KernelLaunchInfo的移动构造函数和移动赋值函数
TEST_F(TestGenTaskCallback, TestKernelLaunchInfoMoveConstruct) {
  auto graph = CreateMc2NodeGraph();
  auto mc2_node = graph->FindNode("mc2");
  auto res_context_holder = CreateNodeExeResContext(mc2_node);
  auto op_exe_res_ctx = reinterpret_cast<gert::ExeResGenerationContext *>(res_context_holder->GetKernelContext());
  auto aicpu_task = KernelLaunchInfo::CreateAicpuKfcTask(op_exe_res_ctx,
      "libccl_kernel.so", "RunAicpuKfcSrvLaunch");
  aicpu_task.SetStreamId(2);
  aicpu_task.SetBlockDim(32);
  std::string args_format = "{#4113}{hi.hcom0*}{i0*}{#0}{#0}{o0*}{o1*}{ws*}{t}";
  aicpu_task.SetArgsFormat(args_format.c_str());
  // 验证移动赋值函数
  KernelLaunchInfo aicpu_task_1 = KernelLaunchInfo::CreateHcomRecordTask(op_exe_res_ctx);
  aicpu_task_1 = std::move(aicpu_task);
  EXPECT_EQ(std::string(aicpu_task_1.GetSoName()), "libccl_kernel.so");
  EXPECT_EQ(std::string(aicpu_task_1.GetKernelName()), "RunAicpuKfcSrvLaunch");
  EXPECT_EQ(aicpu_task_1.GetStreamId(), 2);
  EXPECT_EQ(aicpu_task_1.GetBlockDim(), 32);
  EXPECT_EQ(std::string(aicpu_task_1.GetArgsFormat()), "{#4113}{hi.hcom0*}{i0*}{#0}{#0}{o0*}{o1*}{ws*}{t}");
  // 验证移动构造函数
  KernelLaunchInfo aicpu_task_2(std::move(aicpu_task_1));
  EXPECT_EQ(std::string(aicpu_task_2.GetSoName()), "libccl_kernel.so");
  EXPECT_EQ(std::string(aicpu_task_2.GetKernelName()), "RunAicpuKfcSrvLaunch");
  EXPECT_EQ(aicpu_task_2.GetStreamId(), 2);
  EXPECT_EQ(std::string(aicpu_task_2.GetArgsFormat()), "{#4113}{hi.hcom0*}{i0*}{#0}{#0}{o0*}{o1*}{ws*}{t}");
}

// 验证KernelLaunchInfo的拷贝构造函数和拷贝赋值函数
TEST_F(TestGenTaskCallback, TestKernelLaunchInfoCopyConstruct) {
  auto graph = CreateMc2NodeGraph();
  auto mc2_node = graph->FindNode("mc2");
  auto res_context_holder = CreateNodeExeResContext(mc2_node);
  auto op_exe_res_ctx = reinterpret_cast<gert::ExeResGenerationContext *>(res_context_holder->GetKernelContext());
  domi::TaskDef aicore_task_def;
  aicore_task_def.set_type(static_cast<uint32_t>(ModelTaskType::MODEL_TASK_ALL_KERNEL));
  aicore_task_def.set_id(op_exe_res_ctx->GetOpId());
  aicore_task_def.set_stream_id(op_exe_res_ctx->GetStreamId());
  auto kernel_def_with_handle = aicore_task_def.mutable_kernel_with_handle();
  kernel_def_with_handle->set_block_dim(32);
  kernel_def_with_handle->set_schedule_mode(0);
  auto kernel_context = kernel_def_with_handle->mutable_context();
  kernel_context->set_kernel_type(static_cast<uint32_t>(ccKernelType::TE));
  kernel_context->set_op_index(op_exe_res_ctx->GetOpId());
  std::vector<ArgDesc> args;
  ArgsFormatDescUtils::Append(args, AddrType::FFTS_ADDR);
  size_t input_size = op_exe_res_ctx->GetComputeNodeInfo()->GetIrInputsNum();
  size_t output_size = op_exe_res_ctx->GetComputeNodeInfo()->GetIrOutputsNum();
  for (size_t i = 0UL; i < input_size; i++) {
    ArgsFormatDescUtils::Append(args, AddrType::INPUT_INSTANCE, i);
  }
  for (size_t i = 0UL; i < output_size; i++) {
    ArgsFormatDescUtils::Append(args, AddrType::OUTPUT_INSTANCE, i);
  }
  ArgsFormatDescUtils::Append(args, AddrType::WORKSPACE);
  ArgsFormatDescUtils::Append(args, AddrType::TILING);
  kernel_context->set_args_format(ArgsFormatDescUtils::Serialize(args));
  // 序列化
  std::vector<std::vector<uint8_t>> tasks;
  auto buffer_size = aicore_task_def.ByteSizeLong();
  std::vector<uint8_t> buffer(buffer_size, 0);
  aicore_task_def.SerializeToArray(buffer.data(), buffer_size);
  auto aicore_task = KernelLaunchInfo::LoadFromData(op_exe_res_ctx, buffer);
  EXPECT_EQ(aicore_task.SetBlockDim(48), SUCCESS);
  // 验证拷贝赋值函数
  KernelLaunchInfo copy_task = KernelLaunchInfo::CreateHcomRecordTask(op_exe_res_ctx);
  copy_task = aicore_task;
  EXPECT_EQ(copy_task.GetStreamId(), 2);
  EXPECT_EQ(copy_task.GetBlockDim(), 48);
  EXPECT_EQ(std::string(copy_task.GetArgsFormat()), "{ffts_addr}{i_instance0*}{i_instance1*}{i_instance2*}{o_instance0*}{o_instance1*}{ws*}{t}");

  // 验证拷贝构造函数
  KernelLaunchInfo copy_task_2(copy_task);
  EXPECT_EQ(copy_task_2.GetStreamId(), 2);
  EXPECT_EQ(copy_task_2.GetBlockDim(), 48);
  EXPECT_EQ(std::string(copy_task_2.GetArgsFormat()), "{ffts_addr}{i_instance0*}{i_instance1*}{i_instance2*}{o_instance0*}{o_instance1*}{ws*}{t}");
}

// 验证非aicore和aicpu算子设置blockdim场景
TEST_F(TestGenTaskCallback, TestNonAicoreNodeSetBlockDimFailed) {
  auto graph = CreateMc2NodeGraph();
  auto mc2_node = graph->FindNode("mc2");
  auto res_context_holder = CreateNodeExeResContext(mc2_node);
  auto op_exe_res_ctx = reinterpret_cast<gert::ExeResGenerationContext *>(res_context_holder->GetKernelContext());

  auto notify_task = KernelLaunchInfo::CreateHcomRecordTask(op_exe_res_ctx);
  EXPECT_EQ(notify_task.SetBlockDim(48), PARAM_INVALID);
  EXPECT_EQ(notify_task.GetBlockDim(), 0);
}
// 验证非aicore和aicpu算子设置argsformat场景
TEST_F(TestGenTaskCallback, TestNonAicoreNodeSetArgsFormatFailed) {
  auto graph = CreateMc2NodeGraph();
  auto mc2_node = graph->FindNode("mc2");
  auto res_context_holder = CreateNodeExeResContext(mc2_node);
  auto op_exe_res_ctx = reinterpret_cast<gert::ExeResGenerationContext *>(res_context_holder->GetKernelContext());

  auto notify_task = KernelLaunchInfo::CreateHcomRecordTask(op_exe_res_ctx);
  EXPECT_EQ(notify_task.SetArgsFormat("aaaaa"), PARAM_INVALID);
  EXPECT_EQ(notify_task.GetArgsFormat(), nullptr);
}
}