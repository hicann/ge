
/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aicpu_node_converter.h"
#include "aicpu_callback.h"
#include "engine/node_converter_utils.h"
#include "graph_builder/bg_infer_shape.h"
#include "graph_builder/bg_memory.h"
#include "graph_builder/bg_identity.h"
#include "engine/aicpu/graph_builder/bg_launch.h"
#include "framework/common/ge_types.h"
#include "framework/common/framework_types_internal.h"
#include "common/hyper_status.h"
#include "graph/debug/ge_attr_define.h"
#include "aicpu_engine_struct.h"
#include "engine/aicpu/graph_builder/bg_aicpu_arg.h"
#include "engine/aicpu/graph_builder/bg_ext_info.h"
#include "graph/utils/node_utils.h"
#include "graph_builder/converter_checker.h"
#include "common/omg_util/omg_util.h"
#include "register/kernel_registry.h"
#include "graph_builder/bg_rt_session.h"
#include "engine/aicpu/kernel/aicpu_resource_manager.h"
#include "engine/aicpu/kernel/fused_host_cpu_compute.h"
#include "graph/utils/graph_utils.h"
#include "rt_external_mem.h"
#include "exe_graph/lowering/frame_selector.h"
#include "framework/common/host_cpu_fusion_attr.h"

namespace gert {
namespace {
const std::set<std::string> kResourceOp = {"TensorListPushBack", "TensorListPopBack"};

void SetSingleOpScene(const ge::NodePtr &node) {
  const auto root_graph = ge::GraphUtils::FindRootGraph(node->GetOwnerComputeGraph());
  if (root_graph != nullptr) {
    bool is_single_op = false;
    (void)ge::AttrUtils::GetBool(root_graph, ge::ATTR_SINGLE_OP_SCENE, is_single_op);
    AicpuResourceManager::GetInstance().SetSingleOp(is_single_op);
  }
}

bg::ValueHolderPtr UpdateWorkSpaceSizeAndAddr(const ge::NodePtr &node, const LowerInput &lower_input,
                                              const bg::ValueHolderPtr &ext_info_handler,
                                              bg::ValueHolderPtr &update_workspace_holder) {
  const auto &op_desc = node->GetOpDesc();
  std::vector<bg::ValueHolderPtr> workspace_info;
  int64_t workspace_size = 0;
  std::vector<int64_t> workspace_bytes = op_desc->GetWorkspaceBytes();
  std::vector<uint32_t> aicpu_workspace_type;
  bool has_aicpu_workspace_type_attr =
      ge::AttrUtils::GetListInt(op_desc, ge::ATTR_NAME_AICPU_WORKSPACE_TYPE, aicpu_workspace_type);
  if (has_aicpu_workspace_type_attr) {
    if (aicpu_workspace_type.size() != workspace_bytes.size()) {
      GELOGE(ge::PARAM_INVALID,
             "Op[%s] aicpu_workspace_type size and workspace_bytes size should be equal, but now aicpu_workspace_type "
             "size "
             "is [%zu], workspace_bytes is [%zu].",
             node->GetName().c_str(), aicpu_workspace_type.size(), workspace_bytes.size());
      return nullptr;
    }

    for (size_t temp_index = 0; temp_index < aicpu_workspace_type.size(); temp_index++) {
      if (aicpu_workspace_type[temp_index] == static_cast<uint32_t>(ge::AicpuWorkSpaceType::CUST_LOG)) {
        // workspace type 与 workspace size应该是一一对应关系
        workspace_size = workspace_bytes[temp_index];
        GELOGD("Op[%s] workspace size for CUST_LOG is [%ld].", node->GetName().c_str(), workspace_size);
        break;
      }
    }
  }

  if (workspace_size > 0) {
    auto workspace_size_holder = bg::ValueHolder::CreateConst(&workspace_size, sizeof(workspace_size));
    workspace_info.emplace_back(workspace_size_holder);
    auto workspace_addr_holder =
        bg::AllocMem(kOnDeviceHbm, workspace_size_holder, *(lower_input.global_data), op_desc->GetStreamId());
    workspace_info.emplace_back(workspace_addr_holder);
    workspace_info.emplace_back(ext_info_handler);
    update_workspace_holder = bg::ValueHolder::CreateSingleDataOutput("UpdateExtWorkSpaceInfo", workspace_info);
    bg::ValueHolder::AddDependency(workspace_addr_holder, update_workspace_holder);
    return workspace_addr_holder;
  } else {
    GELOGD("Op[%s] workspace size is zero for CUST_LOG.", node->GetName().c_str());
    return nullptr;
  }
}

NodeOutput UpdateOutputShapeAndAddr(const ge::NodePtr &node, const LowerInput &lower_input,
                                    const bg::AicpuArgs &aicpu_args, bg::ValueHolderPtr &update_holder,
                                    bg::ValueHolderPtr &workspace_addr_holder) {
  auto node_output =
      GetOutputShapeAndAddr(node, lower_input.input_shapes, lower_input.input_addrs, *(lower_input.global_data));

  const auto &op_desc = node->GetOpDesc();

  // get workspace size and addr, update them in ext_info
  bg::ValueHolderPtr update_workspace_holder = nullptr;
  workspace_addr_holder =
      UpdateWorkSpaceSizeAndAddr(node, lower_input, aicpu_args.ext_info_handler, update_workspace_holder);

  auto update_input_shapes = lower_input.input_shapes;
  std::vector<bg::DevMemValueHolderPtr> update_input_addrs = lower_input.input_addrs;
  bool optional_input_placeholder = false;
  (void)ge::AttrUtils::GetBool(op_desc, bg::kOptionalInputPlaceholder, optional_input_placeholder);
  bg::ValueHolderPtr expanded_input_shapes_holder = nullptr;
  bg::ValueHolderPtr expanded_input_addrs_holder = nullptr;
  if (optional_input_placeholder && node->GetOpDescBarePtr()->GetOpKernelLibName() == ge::kEngineNameAiCpu) {
    update_input_shapes = bg::ExpandAicpuOptionalInputShapes(node, lower_input.input_shapes);
    if (update_input_shapes.size() > lower_input.input_shapes.size()) {
      expanded_input_shapes_holder = update_input_shapes.front();
    }
    update_input_addrs = bg::ExpandAicpuOptionalInputAddrs(node, lower_input.input_addrs, kOnDeviceHbm);
    if (update_input_addrs.size() > lower_input.input_addrs.size()) {
      expanded_input_addrs_holder = update_input_addrs.front();
    }
  }
  auto update_ext = bg::UpdateExtInfo(op_desc, {update_input_shapes, node_output.shapes}, aicpu_args.ext_info_handler,
                                      lower_input.global_data->GetStream());
  update_holder = bg::UpdateAicpuIoAddr(aicpu_args.args_handler, update_input_addrs, node_output.addrs);
  if (optional_input_placeholder && expanded_input_addrs_holder != nullptr) {
    bg::ValueHolder::AddDependency(expanded_input_addrs_holder, update_holder);
  }

  if (update_ext != nullptr) {
    if (optional_input_placeholder && expanded_input_shapes_holder != nullptr) {
      bg::ValueHolder::AddDependency(expanded_input_shapes_holder, update_ext);
    }
    if (update_workspace_holder != nullptr) {
      bg::ValueHolder::AddDependency(update_workspace_holder, update_ext);
    }
    bg::ValueHolder::AddDependency(update_ext, update_holder);
  }
  return node_output;
}
}  // namespace

LowerResult LoweringAiCpuTfNode(const ge::NodePtr &node, const LowerInput &lower_input) {
  auto compile_result = lower_input.global_data->FindCompiledResult(node);
  const domi::TaskDef *task_def = GetTaskDef(node, compile_result, TaskDefType::kAICpu);
  if (task_def == nullptr) {
    return {HyperStatus::ErrorStatus(static_cast<const char *>("Cannot find AI cpu Tf taskdef.")), {}, {}, {}};
  }
  auto &kernel_ex_def = task_def->kernel_ex();
  auto session_id = bg::GetSessionId(*lower_input.global_data);

  // gen function handle
  auto rts_args = bg::BuildTfArgsBinHandle(node);

  // alloc args
  auto step_id = GetStepId(*lower_input.global_data);
  auto io_num = node->GetInDataNodesAndAnchors().size() + node->GetAllOutDataAnchorsSize();
  auto aicpu_args = bg::BuildTfAicpuArg(node, {kernel_ex_def, io_num, session_id, step_id}, false);

  // get output shape & addr, update ext_info & io_addr
  bg::ValueHolderPtr update_holder = nullptr;
  bg::ValueHolderPtr workspace_addr_holder = nullptr;
  auto node_output = UpdateOutputShapeAndAddr(node, lower_input, aicpu_args, update_holder, workspace_addr_holder);

  // launch
  auto launch_holder =
      bg::AicpuTfLaunchKernel(aicpu_args.args_handler, lower_input.global_data->GetStream(), rts_args.bin_handle, node);
  bg::ValueHolder::AddDependency(update_holder, launch_holder);

  SetReleaseAfter(lower_input.input_addrs, launch_holder);
  SetReleaseAfter(node_output.addrs, launch_holder);

  std::vector<bg::ValueHolderPtr> ordered_holders;
  ordered_holders.emplace_back(launch_holder);
  AicpuCallback(node, aicpu_args.ext_info_handler, launch_holder, *(lower_input.global_data), node_output);
  ordered_holders.emplace_back(launch_holder);
  if (kResourceOp.count(node->GetType()) > 0U) {
    std::vector<bg::ValueHolderPtr> inputs;
    inputs.emplace_back(lower_input.global_data->GetStream());
    inputs.insert(inputs.cend(), lower_input.input_addrs.cbegin(), lower_input.input_addrs.cend());
    inputs.insert(inputs.cend(), node_output.addrs.cbegin(), node_output.addrs.cend());
    auto resource_op = bg::ValueHolder::CreateSingleDataOutput("TensorListOp", inputs);
    bg::ValueHolder::AddDependency(launch_holder, resource_op);
    ordered_holders.emplace_back(resource_op);
  }
  std::vector<bg::DevMemValueHolderPtr> out_addrs;
  for (const auto &addrs : node_output.addrs) {
    out_addrs.emplace_back(std::dynamic_pointer_cast<bg::DevMemValueHolder>(addrs));
  }

  return {HyperStatus::Success(), ordered_holders, node_output.shapes, out_addrs};
}

LowerResult LoweringAiCpuCCNode(const ge::NodePtr &node, const LowerInput &lower_input) {
  auto compile_result = lower_input.global_data->FindCompiledResult(node);
  const domi::TaskDef *task_def = GetTaskDef(node, compile_result, TaskDefType::kAICpu);
  if (task_def == nullptr) {
    return {HyperStatus::ErrorStatus(static_cast<const char *>("Cannot find AI cpu CC taskdef.")), {}, {}, {}};
  }
  auto &kernel_def = task_def->kernel();
  const auto &op_desc = node->GetOpDesc();
  const auto &stream = lower_input.global_data->GetStream();
  auto session_id = bg::GetSessionId(*lower_input.global_data);

  // gen function handle
  auto rts_args = bg::BuildCCArgsBinHandle(node);

  // alloc args
  auto in_num = node->GetInDataNodesAndAnchors().size();
  bool optional_input_placeholder = false;
  (void)ge::AttrUtils::GetBool(node->GetOpDescBarePtr(), bg::kOptionalInputPlaceholder, optional_input_placeholder);
  if (optional_input_placeholder) {
    in_num = node->GetOpDescBarePtr()->GetAllInputsSize();
    GELOGI("Op %s type %s: all input size is %zu, all input data anchors size is %zu.", node->GetName().c_str(),
           ge::NodeUtils::GetNodeType(node).c_str(), in_num, node->GetInDataNodesAndAnchors().size());
  }
  auto io_num = in_num + node->GetAllOutDataAnchorsSize();
  auto aicpu_args = bg::BuildCCAicpuArg(node, kernel_def, io_num, session_id, false);

  // get output shape & addr, update ext_info & io_addr
  bg::ValueHolderPtr update_holder = nullptr;
  bg::ValueHolderPtr workspace_addr_holder = nullptr;
  auto node_output = UpdateOutputShapeAndAddr(node, lower_input, aicpu_args, update_holder, workspace_addr_holder);

  // launch
  auto block_dim = bg::CalcBlockDim(op_desc, lower_input.input_shapes);
  auto launch_holder = bg::AicpuCCLaunchKernel(aicpu_args.args_handler, stream, block_dim, kernel_def, op_desc,
                                               aicpu_args.ext_info_handler, rts_args.bin_handle, node);

  bg::ValueHolder::AddDependency(update_holder, launch_holder);
  SetReleaseAfter(lower_input.input_addrs, launch_holder);
  SetReleaseAfter(node_output.addrs, launch_holder);
  if (workspace_addr_holder != nullptr) {
    SetReleaseAfter({workspace_addr_holder}, launch_holder);
  }

  auto cc_launch_holder = launch_holder;
  AicpuCallback(node, aicpu_args.ext_info_handler, launch_holder, *(lower_input.global_data), node_output);
  std::vector<bg::DevMemValueHolderPtr> out_addrs;
  for (const auto &addrs : node_output.addrs) {
    out_addrs.emplace_back(std::dynamic_pointer_cast<bg::DevMemValueHolder>(addrs));
  }
  return {HyperStatus::Success(), {cc_launch_holder, launch_holder}, node_output.shapes, out_addrs};
}

bool GetFusedHostCpuSoData(const ge::NodePtr &node, std::string &fused_register_name, ge::Buffer &so_data,
                           const char *&error_message, std::string &so_key, ge::ComputeGraphPtr &root_graph) {
  error_message = "Load fused HostCPU kernel failed";
  if (!ge::AttrUtils::GetStr(node->GetOpDescBarePtr(), ge::kFusedHostCpuRegisterName, fused_register_name)) {
    error_message = "Load fused HostCPU kernel failed";
    GELOGE(ge::INTERNAL_ERROR, "Load fused HostCPU kernel failed for node %s: register name is missing.",
           node->GetNamePtr());
    return false;
  }
  if (!ge::AttrUtils::GetStr(node->GetOpDescBarePtr(), ge::kFusedHostCpuSoKey, so_key)) {
    error_message = "Load fused HostCPU kernel failed";
    GELOGE(ge::INTERNAL_ERROR, "Load fused HostCPU kernel failed for node %s: so key is missing.", node->GetNamePtr());
    return false;
  }
  const auto owner_graph = node->GetOwnerComputeGraph();
  root_graph = ge::GraphUtils::FindRootGraph(owner_graph);
  if (root_graph == nullptr) {
    error_message = "Load fused HostCPU kernel failed";
    GELOGE(ge::INTERNAL_ERROR, "Load fused HostCPU kernel failed for node %s: root graph was not found.",
           node->GetNamePtr());
    return false;
  }
  if (!ge::AttrUtils::GetBytes(root_graph, so_key, so_data)) {
    error_message = "Load fused HostCPU kernel failed";
    GELOGE(ge::INTERNAL_ERROR,
           "Load fused HostCPU kernel failed for node %s: so key[%s] was not found in root graph[%s], "
           "owner graph[%s].",
           node->GetNamePtr(), so_key.c_str(), root_graph->GetName().c_str(),
           owner_graph == nullptr ? "null" : owner_graph->GetName().c_str());
    return false;
  }
  return true;
}

bool LoadFusedHostCpuKernel(const ge::NodePtr &node, std::string &fused_register_name,
                            FusedHostCpuKernelFunctions &kernel_funcs, void *&fused_kernel_state,
                            const char *&error_message) {
  std::string so_key;
  ge::Buffer so_data;
  ge::ComputeGraphPtr root_graph;
  if (!GetFusedHostCpuSoData(node, fused_register_name, so_data, error_message, so_key, root_graph)) {
    return false;
  }
  GELOGD("Load fused HostCPU kernel for node[%s]: register_name[%s], so_key[%s], so_graph[%s], so_size=%zu.",
         node->GetNamePtr(), fused_register_name.c_str(), so_key.c_str(), root_graph->GetName().c_str(),
         so_data.GetSize());
  auto &resource_manager = AicpuResourceManager::GetInstance();
  if (resource_manager.LoadFusedHostCpuSo(fused_register_name, so_data.GetData(), so_data.GetSize()) !=
      ge::GRAPH_SUCCESS) {
    GELOGE(ge::INTERNAL_ERROR, "Load fused HostCPU kernel failed for node %s.", node->GetNamePtr());
    return false;
  }
  kernel_funcs = resource_manager.GetFusedHostCpuKernelFunctions(fused_register_name);
  if ((kernel_funcs.create_func == nullptr) || (kernel_funcs.destroy_func == nullptr) ||
      (kernel_funcs.run_func == nullptr)) {
    error_message = "Resolve fused HostCPU entry failed";
    GELOGE(ge::INTERNAL_ERROR, "Resolve fused HostCPU private entry failed for node %s.", node->GetNamePtr());
    return false;
  }
  fused_kernel_state = kernel_funcs.create_func();
  if (fused_kernel_state == nullptr) {
    error_message = "Prepare fused HostCPU state failed";
    (void)resource_manager.ReleaseFusedHostCpuSo(fused_register_name);
    GELOGE(ge::INTERNAL_ERROR, "Prepare fused HostCPU execution state failed for node %s.", node->GetNamePtr());
    return false;
  }
  return true;
}

void *CreateFusedHostCpuComputeState(const ge::NodePtr &node, const size_t in_num, const size_t io_num,
                                     const std::string &fused_register_name,
                                     const FusedHostCpuKernelFunctions &kernel_funcs, void *fused_kernel_state) {
  std::vector<FusedHostCpuTensorMeta> tensor_metas;
  tensor_metas.reserve(io_num);
  for (size_t i = 0U; i < in_num; ++i) {
    const ge::GeTensorDesc desc = node->GetOpDescBarePtr()->GetInputDesc(i);
    tensor_metas.emplace_back(FusedHostCpuTensorMeta{desc.GetShape().GetDimNum()});
  }
  for (size_t i = 0U; i < node->GetAllOutDataAnchorsSize(); ++i) {
    const ge::GeTensorDesc desc = node->GetOpDescBarePtr()->GetOutputDesc(i);
    tensor_metas.emplace_back(FusedHostCpuTensorMeta{desc.GetShape().GetDimNum()});
  }
  void *fused_compute_state =
      kernel::CreateFusedHostCpuComputeState(fused_register_name.c_str(), fused_kernel_state, kernel_funcs.destroy_func,
                                             kernel_funcs.run_func, tensor_metas.data(), tensor_metas.size());
  if (fused_compute_state == nullptr) {
    kernel_funcs.destroy_func(fused_kernel_state);
    (void)AicpuResourceManager::GetInstance().ReleaseFusedHostCpuSo(fused_register_name);
    GELOGE(ge::INTERNAL_ERROR, "Prepare fused HostCPU compute state failed for node %s.", node->GetNamePtr());
    return nullptr;
  }
  const FusedHostCpuDestroyMeta destroy_meta = {fused_compute_state};
  bg::FrameSelector::OnDeInitRoot([destroy_meta]() -> std::vector<bg::ValueHolderPtr> {
    auto meta_holder = bg::ValueHolder::CreateConst(&destroy_meta, sizeof(destroy_meta));
    return {bg::ValueHolder::CreateVoidGuarder("ReleaseFusedHostCpuKernelState", meta_holder, {})};
  });
  bg::FrameSelector::OnDeInitRoot([fused_register_name]() -> std::vector<bg::ValueHolderPtr> {
    auto name_holder = bg::ValueHolder::CreateConst(fused_register_name.c_str(), fused_register_name.size() + 1U, true);
    return {bg::ValueHolder::CreateVoidGuarder("ReleaseFusedHostCpuSo", name_holder, {})};
  });
  GELOGD("Fused HostCPU kernel[%s] is ready for node[%s].", fused_register_name.c_str(), node->GetNamePtr());
  return fused_compute_state;
}

void *PrepareFusedHostCpuComputeState(const ge::NodePtr &node, const size_t in_num, const size_t io_num,
                                      std::string &fused_register_name, const char *&error_message) {
  FusedHostCpuKernelFunctions kernel_funcs;
  void *fused_kernel_state = nullptr;
  if (!LoadFusedHostCpuKernel(node, fused_register_name, kernel_funcs, fused_kernel_state, error_message)) {
    return nullptr;
  }
  void *fused_compute_state =
      CreateFusedHostCpuComputeState(node, in_num, io_num, fused_register_name, kernel_funcs, fused_kernel_state);
  if (fused_compute_state == nullptr) {
    error_message = "Prepare fused HostCPU compute state failed";
  }
  return fused_compute_state;
}

struct HostAiCpuLoweringData {
  const domi::KernelDef *kernel_def = nullptr;
  bg::ValueHolderPtr session_id;
  bg::AicpuArgs aicpu_args;
  size_t in_num = 0U;
  bool is_fused_host_cpu = false;
  std::string fused_register_name;
  void *fused_compute_state = nullptr;
};

const char *PrepareHostAiCpuLowering(const ge::NodePtr &node, const LowerInput &lower_input,
                                     HostAiCpuLoweringData &lowering_data) {
  auto compile_result = lower_input.global_data->FindCompiledResult(node);
  const domi::TaskDef *task_def = GetTaskDef(node, compile_result, TaskDefType::kAICpu);
  if (task_def == nullptr) {
    return "Cannot find host AI cpu taskdef.";
  }
  lowering_data.kernel_def = &task_def->kernel();
  lowering_data.session_id = bg::GetSessionId(*lower_input.global_data);
  lowering_data.is_fused_host_cpu = node->GetType() == ge::kFusedHostCpuOpType;
  if (lowering_data.is_fused_host_cpu) {
    GELOGD("Lower fused HostCPU node[%s]: inputs=%zu, outputs=%zu.", node->GetNamePtr(),
           node->GetAllInDataAnchorsSize(), node->GetAllOutDataAnchorsSize());
  }

  // 融合 so 只注册编排 kernel，原始 HostCPU kernel 仍由基础库提供，必须先加载基础库。
  if (lowering_data.is_fused_host_cpu &&
      (AicpuResourceManager::GetInstance().LoadConstantFoldingLib() != ge::GRAPH_SUCCESS)) {
    GELOGE(ge::INTERNAL_ERROR, "Load HostCPU base library failed for fused node %s.", node->GetNamePtr());
    return "Load HostCPU base library failed";
  }

  // alloc args
  lowering_data.in_num = node->GetInDataNodesAndAnchors().size();
  bool optional_input_placeholder = false;
  (void)ge::AttrUtils::GetBool(node->GetOpDescBarePtr(), bg::kOptionalInputPlaceholder, optional_input_placeholder);
  if (optional_input_placeholder) {
    lowering_data.in_num = node->GetOpDescBarePtr()->GetAllInputsSize();
    GELOGI("Op %s type %s in all input size is %zu, all input data anchors size is %zu.", node->GetName().c_str(),
           ge::NodeUtils::GetNodeType(node).c_str(), lowering_data.in_num, node->GetInDataNodesAndAnchors().size());
  }
  const auto io_num = lowering_data.in_num + node->GetAllOutDataAnchorsSize();
  lowering_data.aicpu_args = bg::BuildHostCCAicpuArg(node, *lowering_data.kernel_def, io_num, lowering_data.session_id);

  if (lowering_data.is_fused_host_cpu) {
    const char *fused_state_error = nullptr;
    lowering_data.fused_compute_state = PrepareFusedHostCpuComputeState(
        node, lowering_data.in_num, io_num, lowering_data.fused_register_name, fused_state_error);
    if (lowering_data.fused_compute_state == nullptr) {
      return fused_state_error;
    }
  }
  return nullptr;
}

LowerResult BuildHostAiCpuLoweringResult(const ge::NodePtr &node, const LowerInput &lower_input,
                                         const HostAiCpuLoweringData &lowering_data) {
  // get output shape and addr
  auto output_shapes = bg::GetMemAllocShape(node, lower_input.input_shapes, *(lower_input.global_data));
  auto output_sizes = bg::CalcTensorSize(node, output_shapes);

  std::vector<bg::DevMemValueHolderPtr> output_addrs;
  const bg::IoInfo io_info{lower_input.input_addrs, lower_input.input_shapes, output_sizes, output_shapes};
  if (lowering_data.is_fused_host_cpu) {
    auto compute_holder = bg::BuildFusedHostCpuComputeNode(node, lowering_data.fused_compute_state, io_info,
                                                           *lower_input.global_data, output_addrs);
    SetReleaseAfter(lower_input.input_addrs, compute_holder);
    return {HyperStatus::Success(), {}, output_shapes, output_addrs};
  }
  auto compute_holder =
      bg::AicpuHostCompute(node, lowering_data.aicpu_args, io_info, *lower_input.global_data, output_addrs);

  auto after_compute_addrs = IdentityAddr(output_addrs, node->GetOpDescBarePtr()->GetStreamId());
  for (auto addr : after_compute_addrs) {
    bg::ValueHolder::AddDependency(compute_holder, addr);
  }
  SetReleaseAfter(lower_input.input_addrs, compute_holder);
  return {HyperStatus::Success(), {}, output_shapes, after_compute_addrs};
}

LowerResult LoweringHostAiCpuNode(const ge::NodePtr &node, const LowerInput &lower_input) {
  HostAiCpuLoweringData lowering_data;
  const char *error_message = PrepareHostAiCpuLowering(node, lower_input, lowering_data);
  if (error_message != nullptr) {
    return {HyperStatus::ErrorStatus(error_message), {}, {}, {}};
  }
  return BuildHostAiCpuLoweringResult(node, lower_input, lowering_data);
}

LowerResult LoweringAiCpuNode(const ge::NodePtr &node, const LowerInput &lower_input) {
  if ((node == nullptr) || (node->GetOpDescBarePtr() == nullptr)) {
    GELOGE(ge::PARAM_INVALID, "[Check][Op]Can not find op.");
    REPORT_INNER_ERR_MSG("E19999", "Can not find op.");
    return {HyperStatus::ErrorStatus(static_cast<const char *>("Can not find op")), {}, {}, {}};
  }
  auto ret = CheckLowerInput(lower_input);
  if (!ret.IsSuccess()) {
    GELOGE(ge::PARAM_INVALID, "[Check][LowerInput]Op %s type %s lower_input is invalid.", node->GetName().c_str(),
           ge::NodeUtils::GetNodeType(node).c_str());
    REPORT_INNER_ERR_MSG("E19999", "Op %s type %s lower_input is invalid.", node->GetName().c_str(),
                         ge::NodeUtils::GetNodeType(node).c_str());
    return {ret, {}, {}, {}};
  }
  auto compile_result = lower_input.global_data->FindCompiledResult(node);
  if (compile_result == nullptr) {
    GELOGE(ge::PARAM_INVALID, "[Check][CompileResult]Can not find compile result for node %s type %s",
           node->GetName().c_str(), ge::NodeUtils::GetNodeType(node).c_str());
    REPORT_INNER_ERR_MSG("E19999", "[Check][CompileResult]Can not find compile result for node %s type %s",
                         node->GetName().c_str(), ge::NodeUtils::GetNodeType(node).c_str());
    return {HyperStatus::ErrorStatus(static_cast<const char *>("Can not find compile result")), {}, {}, {}};
  }
  if (compile_result->task_defs.empty()) {
    GELOGE(ge::PARAM_INVALID, "[Check][TaskDef]Unexpected task defs count %zu", compile_result->task_defs.size());
    REPORT_INNER_ERR_MSG("E19999", "Unexpected task defs count %zu", compile_result->task_defs.size());
    return {HyperStatus::ErrorStatus(static_cast<const char *>("Unexpected task defs count")), {}, {}, {}};
  }
  int32_t unknown_shape_type_val = 0;
  (void)ge::AttrUtils::GetInt(node->GetOpDescBarePtr(), ge::ATTR_NAME_UNKNOWN_SHAPE_TYPE, unknown_shape_type_val);
  if ((bg::IsAicpuUnknownShape(node)) && (unknown_shape_type_val == static_cast<int32_t>(ge::DEPEND_COMPUTE))) {
    // when the operator is the fourth type, and corresponding node is unknown, then 2 tasks are required.
    if (compile_result->task_defs.size() != 2U) {
      GELOGE(ge::PARAM_INVALID, "[Check][TaskDef]Op %s type %s is 4th op, unexpected task defs count %zu",
             node->GetName().c_str(), ge::NodeUtils::GetNodeType(node).c_str(), compile_result->task_defs.size());
      REPORT_INNER_ERR_MSG("E19999", "Op %s type %s is 4th op, unexpected task defs count %zu", node->GetName().c_str(),
                           ge::NodeUtils::GetNodeType(node).c_str(), compile_result->task_defs.size());
      return {HyperStatus::ErrorStatus(static_cast<const char *>("Unexpected task defs count")), {}, {}, {}};
    }
  }

  SetSingleOpScene(node);
  if (node->GetOpDescBarePtr()->GetOpKernelLibName() == ge::kEngineNameAiCpuTf) {
    GELOGI("Op %s type %s in tf_aicpu lowering.", node->GetName().c_str(), ge::NodeUtils::GetNodeType(node).c_str());
    return LoweringAiCpuTfNode(node, lower_input);
  } else if (node->GetOpDescBarePtr()->GetOpKernelLibName() == ge::kEngineNameAiCpu) {
    GELOGI("Op %s type %s in cc_aicpu lowering.", node->GetName().c_str(), ge::NodeUtils::GetNodeType(node).c_str());
    return LoweringAiCpuCCNode(node, lower_input);
  } else {
    GELOGI("Op %s type %s in host_cpu lowering.", node->GetName().c_str(), ge::NodeUtils::GetNodeType(node).c_str());
    return LoweringHostAiCpuNode(node, lower_input);
  }
}

REGISTER_NODE_CONVERTER_PLACEMENT(ge::kEngineNameAiCpuTf.c_str(), kOnDeviceHbm, LoweringAiCpuNode);
REGISTER_NODE_CONVERTER_PLACEMENT(ge::kEngineNameAiCpu.c_str(), kOnDeviceHbm, LoweringAiCpuNode);
REGISTER_NODE_CONVERTER_PLACEMENT(ge::kEngineNameHostCpu.c_str(), kOnHost, LoweringAiCpuNode);
}  // namespace gert
