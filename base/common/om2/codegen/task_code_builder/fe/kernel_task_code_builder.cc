/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "kernel_task_code_builder.h"
#include <cinttypes>
#include <numeric>
#include <sstream>
#include "common/om2/codegen/task_code_builder/task_code_builder_util.h"
#include "common/om2/codegen/task_code_builder_factory.h"
#include "common/om2/codegen/om2_model_utils.h"
#include "common/checker.h"
#include "common/ge_common/debug/ge_log.h"
#include "common/opskernel/ops_kernel_info_types.h"
#include "graph/args_format_desc.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/utils/attr_utils.h"
#include "graph/utils/tensor_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/def_types.h"
#include "register/op_tiling/op_tiling_constants.h"
#include "common/op_tiling/op_tiling_rt2.h"
#include "graph/utils/math_util.h"
#include "common/om2/codegen/task_args_manager/om2_model_args_utils.h"
#include "aicpu_task_struct.h"

namespace ge {
namespace {
constexpr uint32_t k2BitsMask = 0x00000003U;
const std::string kAllShapeInAicpu = "_AllShape";
constexpr int64_t kDefaultDimInfo = static_cast<int64_t>(0x100000001ULL);
constexpr uint64_t kDefaultShapeNum = 0x100000000U;
const std::string kWspUnfoldedMode = "unfolded";
const std::string kWspFoldedMode = "folded";
const std::string kAttrNameAtomicWspMode = "wspMode";
constexpr char_t const *kMaxTilingSize = "op_para_size";
constexpr char_t const *kMaxAtomicCleanTilingSize = "atomic_op_para_size";
constexpr char_t const *kLocalMemorySize = "local_memory_size";
constexpr uint32_t kUBAlignedLen = 32U;
constexpr int32_t kSessionInfoOffset = 8;
constexpr uint32_t kAicpuArgsExtInfoAddrOffset = 12U;
constexpr uint32_t kAicpuArgsioAddrOffset = 20U;
const std::string kOptionalInputPlaceholder = "optional_input_placeholder";

void AppendShapeDesc(const ge::GeTensorDesc &tensor_desc, std::vector<int64_t> &shape_infos) {
  const auto &shape = tensor_desc.GetShape();
  if (shape.IsScalar()) {
    shape_infos.push_back(kDefaultDimInfo);
    shape_infos.push_back(0x1);  // shape value [1]
  } else {
    uint64_t dim_info{kDefaultShapeNum};
    dim_info |= (static_cast<uint64_t>(shape.GetDimNum()));
    shape_infos.push_back(static_cast<int64_t>(dim_info));
    for (const int64_t dim : shape.GetDims()) {
      shape_infos.push_back(dim);
    }
  }
}

bool IsWspAddrFolded(const OpDescPtr &op_desc) {
  const string *wsp_mode = ge::AttrUtils::GetStr(op_desc, kAttrNameAtomicWspMode);
  return (wsp_mode != nullptr) && (*wsp_mode == kWspFoldedMode);
}

bool IsMaterializedOutput(const AddrSemantic &semantic) {
  return semantic.kind != AddrValueKind::kOptionalEmpty;
}

bool IsAllKernelTask(const KernelTaskSemantic &semantic) {
  return Om2CodegenUtils::IsAllKernel(semantic.task_type);
}

uint32_t ConvertEngineType(const std::string &engine_type_str) {
  if (engine_type_str == "ACL_RT_ENGINE_TYPE_AIV") {
    return 1U;
  }
  return 0U;  // ACL_RT_ENGINE_TYPE_AIC
}

bool IsAicpuTask(const KernelTaskSemantic &semantic) {
  return semantic.kernel_type == ge::ccKernelType::AI_CPU;
}

bool IsCustAicpuTask(const KernelTaskSemantic &semantic) {
  return semantic.kernel_type == ge::ccKernelType::CUST_AI_CPU;
}

uint64_t GetOrderedArgByteSize(const AddrSemantic &semantic) {
  if ((semantic.kind == AddrValueKind::kShapeInfoBuffer) && semantic.shape_info.has_value()) {
    return semantic.shape_info->size() * kAddressLen;
  }
  return kAddressLen;
}

uint64_t GetOrderedArgsByteSize(const std::vector<AddrSemantic> &ordered_args) {
  uint64_t size = 0U;
  for (const auto &ordered_arg : ordered_args) {
    size += GetOrderedArgByteSize(ordered_arg);
  }
  return size;
}
}  // namespace

void KernelTaskCodeBuilder::AppendOrderedArgValue(const AddrSemantic &semantic) {
  if (args_table_entry_ == nullptr) {
    REPORT_INNER_ERR_MSG("E19999", "Args table entry is required before append ordered arg.");
    GELOGE(FAILED, "[OM2] Args table entry is required before append ordered arg.");
    return;
  }
  if (semantic.memory_app == om2::MemoryAppType::kMemoryTypeModelIo) {
    uint64_t current_offset = args_table_entry_->host_offset;
    for (const auto &ordered_arg : build_data_.semantic.ordered_arg_values) {
      current_offset += (ordered_arg.kind == AddrValueKind::kShapeInfoBuffer && ordered_arg.shape_info.has_value())
                            ? ordered_arg.shape_info->size() * kAddressLen
                            : kAddressLen;
    }
    io_addr_refresh_records_.push_back(
        IoAddrRefreshRecord{static_cast<uint64_t>(semantic.compile_state_io_addr_offset), current_offset});
  }
  build_data_.semantic.ordered_arg_values.push_back(semantic);
}

Status KernelTaskCodeBuilder::AssembleBuildData() {
  const bool is_aicpu = IsAicpuTask(build_data_.semantic) || IsCustAicpuTask(build_data_.semantic);
  uint64_t current_args_offset = 0U;
  for (const auto &addr : build_data_.semantic.ordered_arg_values) {
    if (addr.kind == AddrValueKind::kShapeInfoBuffer) {
      HandleShapeInfoBufferArg(addr, current_args_offset, build_data_.ordered_args);
      continue;
    }

    OpArgDesc arg = TaskCodeBuilderUtil::ConvertAddrDesc(addr);
    // kernel-specific post-processing
    if (addr.kind == AddrValueKind::kLevel1DescPtr && build_data_.semantic.args_table_entry.has_value() &&
        addr.level1_target_offset.has_value()) {
      arg.custom_value = build_data_.semantic.args_table_entry->host_offset + *addr.level1_target_offset;
    }
    if (addr.kind == AddrValueKind::kTiling) {
      arg.raw_data.assign(tiling_data_.begin(), tiling_data_.end());
    }
    arg.args_offset = current_args_offset;
    current_args_offset += kAddressLen;
    build_data_.ordered_args.push_back(std::move(arg));
  }

  if (is_aicpu) {
    build_data_.dispatch_info = BuildAicpuTaskData();
  } else {
    AicoreTaskData aicore = BuildAicoreTaskData();
    build_data_.dispatch_info = std::move(aicore);
  }
  return SUCCESS;
}

std::string KernelTaskCodeBuilder::GetFuncName() const {
  return kDispatchFuncName;
}

void KernelTaskCodeBuilder::HandleShapeInfoBufferArg(const AddrSemantic &addr, uint64_t &current_args_offset,
                                                     std::vector<OpArgDesc> &ordered_args) const {
  if (!addr.shape_info.has_value()) {
    return;
  }
  for (int64_t dim : *addr.shape_info) {
    OpArgDesc shape_arg;
    shape_arg.type = OP_ARG_SHAPE_INFO;
    shape_arg.custom_value = static_cast<uint64_t>(dim);
    if (addr.memory_type == (kSessionScopeMemoryMask | RT_MEMORY_HBM)) {
      shape_arg.mem_src = MEM_SRC_SESSION;
    }
    shape_arg.args_offset = current_args_offset;
    current_args_offset += kAddressLen;
    ordered_args.push_back(std::move(shape_arg));
  }
}

AicoreTaskData KernelTaskCodeBuilder::BuildAicoreTaskData() const {
  AicoreTaskData aicore;
  aicore.engine_type = ConvertEngineType(build_data_.semantic.launch.config.engine_type);
  aicore.need_assert_or_printf = op_need_assert_or_printf_ ? 1U : 0U;
  GELOGI("[OM2] GetOpDefBuildData: op=%s, func_idx=%u", header_.op_name.c_str(),
         build_data_.semantic.launch.func_handle_index);
  return aicore;
}

AicpuTaskData KernelTaskCodeBuilder::BuildAicpuTaskData() const {
  AicpuTaskData aicpu;
  aicpu.engine_type = ConvertEngineType(build_data_.semantic.launch.config.engine_type);
  GELOGI("[OM2] GetOpDefBuildData: op=%s (AICPU), func_idx=%u", header_.op_name.c_str(),
         build_data_.semantic.launch.func_handle_index);
  return aicpu;
}

Status KernelTaskCodeBuilder::AppendOrderedArgValueForCommon(const AddrSemantic &semantic, const uint64_t addr_offset) {
  if (semantic.memory_app == om2::MemoryAppType::kMemoryTypeModelIo) {
    io_addr_refresh_records_.push_back(
        IoAddrRefreshRecord{static_cast<uint64_t>(semantic.compile_state_io_addr_offset), addr_offset});
    GELOGI("[OM2]append input addr offset map: compile offset[%lu], args info offset[%lu]",
           semantic.compile_state_io_addr_offset, addr_offset);
  }
  build_data_.semantic.ordered_arg_values.push_back(semantic);
  return SUCCESS;
}

void KernelTaskCodeBuilder::AppendOrderedArg(const AddrSemantic &semantic) {
  AppendOrderedArgValue(semantic);
  current_args_offset_ += GetOrderedArgByteSize(semantic);
}

Status KernelTaskCodeBuilder::ValidateLevel1DescTargetOffsets() const {
  for (size_t i = 0UL; i < build_data_.semantic.ordered_arg_values.size(); ++i) {
    const auto &ordered_arg = build_data_.semantic.ordered_arg_values[i];
    if (ordered_arg.kind == AddrValueKind::kLevel1DescPtr) {
      GE_ASSERT_TRUE(ordered_arg.level1_target_offset.has_value(),
                     "[OM2] Level1 desc target offset is missing, index[%zu], symbol[%s].", i,
                     ordered_arg.symbol_hint.c_str());
    }
  }
  return SUCCESS;
}

Status KernelTaskCodeBuilder::Contribute(TaskSemanticContributeContext &context) {
  GE_ASSERT_SUCCESS(TaskCodeBuilder::Contribute(context));
  GE_ASSERT_NOTNULL(context.next_args_table_index);
  GE_ASSERT_NOTNULL(context.next_host_args_offset);
  GE_ASSERT_NOTNULL(context.aicpu_task_count);
  build_data_.semantic.task_type = context.task_type;
  build_data_.semantic.kernel_type = static_cast<ccKernelType>(
      Om2CodegenUtils::IsAllKernel(context.task_type) ? context.task_def.kernel_with_handle().context().kernel_type()
                                                      : context.task_def.kernel().context().kernel_type());
  GE_ASSERT_SUCCESS(ResolveKernelName(build_data_.semantic, context.op_desc, context.task_def, kernel_name_));
  GE_ASSERT_NOTNULL(context.op_desc);
  op_need_print_ = Om2CodegenUtils::OpNeedPrint(context.op_desc);
  op_need_assert_or_printf_ = Om2CodegenUtils::OpNeedAssertOrPrintf(context.op_desc);
  is_soft_sync_op_ = IsAllKernelTask(build_data_.semantic) && Om2CodegenUtils::IsSoftSyncOp(context.op_desc);
  is_separately_clean_task_ =
      (!IsAllKernelTask(build_data_.semantic)) &&
      Om2CodegenUtils::IsSeparatelyCleanTask(context.op_desc, context.task_def.kernel().kernel_name());
  is_blocking_aicpu_op_ = IsAicpuTask(build_data_.semantic) && Om2CodegenUtils::IsBlockingAicpuOp(context.op_desc);
  GE_ASSERT_SUCCESS(CheckTaskSupport());
  GE_ASSERT_SUCCESS(ResolveTaskAddrs(context));
  AssignTaskLocalIoNames();

  GE_ASSERT_SUCCESS(BuildLaunchSemantic(context));

  if (IsAicpuTask(build_data_.semantic) || IsCustAicpuTask(build_data_.semantic)) {
    GE_ASSERT_SUCCESS(BuildOrderedArgValuesForAicpu(context));
    GE_ASSERT_SUCCESS(BuildAicpuArgsSemantic(context));
    GE_ASSERT_SUCCESS(BuildAicpuExtInfoSemantic(context));
  } else {
    ArgsFormatInfo args_format_holder;
    GE_ASSERT_SUCCESS(BuildOrderedArgValuesForAicore(context, args_format_holder));
  }
  if (build_data_.semantic.args_table_entry.has_value()) {
    ++(*context.next_args_table_index);
    *context.next_host_args_offset +=
        Om2ModelUtils::ArgsSizeAlign8(static_cast<size_t>(build_data_.semantic.args_table_entry->args_size));
  }
  dispatch_type_ = (IsAicpuTask(build_data_.semantic) || IsCustAicpuTask(build_data_.semantic))
                       ? OpDispatchType::DISPATCH_AICPU
                       : OpDispatchType::DISPATCH_AICORE;

  GE_CHK_STATUS_RET(ReadFusionOpInfo(context.op_desc));

  GE_ASSERT_SUCCESS(AssembleBuildData());
  return SUCCESS;
}

Status KernelTaskCodeBuilder::ReadFusionOpInfo(const OpDescPtr &op_desc) {
  if (!AttrUtils::GetListStr(op_desc, ATTR_NAME_DATA_DUMP_ORIGIN_OP_NAMES, build_data_.semantic.original_op_names)) {
    return SUCCESS;
  }

  const auto &is_const = op_desc->GetIsInputConst();
  uint64_t input_mem = 0U;
  uint64_t weight_mem = 0U;
  for (size_t i = 0U; i < op_desc->GetAllInputsSize(); ++i) {
    int64_t tensor_size = 0;
    const auto tensor_desc = op_desc->MutableInputDesc(static_cast<uint32_t>(i));
    if (tensor_desc != nullptr && TensorUtils::GetSize(*tensor_desc, tensor_size) == GRAPH_SUCCESS) {
      auto tensor_mem = static_cast<uint64_t>(tensor_size);
      input_mem += tensor_mem;
      if (i < is_const.size() && is_const[i]) {
        weight_mem += tensor_mem;
      }
    }
  }
  build_data_.semantic.input_mem_size = input_mem;
  build_data_.semantic.weight_mem_size = weight_mem;

  uint64_t output_mem = 0U;
  for (uint32_t i = 0U; i < op_desc->GetAllOutputsDescSize(); ++i) {
    int64_t tensor_size = 0;
    const auto tensor_desc = op_desc->MutableOutputDesc(i);
    if (tensor_desc != nullptr && TensorUtils::GetSize(*tensor_desc, tensor_size) == GRAPH_SUCCESS) {
      output_mem += static_cast<uint64_t>(tensor_size);
    }
  }
  build_data_.semantic.output_mem_size = output_mem;

  const auto &ws_bytes = op_desc->GetWorkspaceBytes();
  build_data_.semantic.workspace_mem_size =
      static_cast<uint64_t>(std::accumulate(ws_bytes.begin(), ws_bytes.end(), int64_t{0}));
  return SUCCESS;
}

Status KernelTaskCodeBuilder::ResolveKernelName(const KernelTaskSemantic &semantic, const OpDescPtr &op_desc,
                                                const domi::TaskDef &task_def, std::string &kernel_name) {
  if (IsAllKernelTask(semantic)) {
    const auto kernel_name_ptr = AttrUtils::GetStr(op_desc, "_kernelname");
    GE_ASSERT_NOTNULL(kernel_name_ptr, "[OM2] Failed to get kernel_name from op_desc, op=%s",
                      op_desc->GetName().c_str());
    kernel_name = *kernel_name_ptr;
  } else {
    kernel_name = task_def.kernel().kernel_name();
  }
  return SUCCESS;
}

std::string KernelTaskCodeBuilder::ResolveFuncHandleKey(const TaskSemanticContributeContext &context,
                                                        const std::string &kernel_name) const {
  std::string func_handle_key;
  if (is_separately_clean_task_) {
    const auto *kernel_name_ptr = AttrUtils::GetStr(context.op_desc, ATOMIC_ATTR_TBE_KERNEL_NAME);
    if (kernel_name_ptr != nullptr) {
      func_handle_key = *kernel_name_ptr;
    }
    func_handle_key += "_atomic";
  } else if (IsAicpuTask(build_data_.semantic) || IsCustAicpuTask(build_data_.semantic)) {
    func_handle_key = context.op_desc->GetType() + kernel_name;
  } else if (IsAllKernelTask(build_data_.semantic)) {
    func_handle_key = kernel_name + "#" + std::to_string(build_data_.semantic.tiling_key);
  } else {
    func_handle_key = kernel_name;
  }
  return func_handle_key;
}

Status KernelTaskCodeBuilder::ResolveTaskAddrs(TaskSemanticContributeContext &context) {
  const bool is_aicpu = IsAicpuTask(build_data_.semantic) || IsCustAicpuTask(build_data_.semantic);
  if (is_aicpu) {
    build_data_.semantic.aicpu_task_index = *context.aicpu_task_count;
    ++(*context.aicpu_task_count);
    GE_ASSERT_SUCCESS(Om2ModelUtils::ResolveInputAddrs(context, build_data_.semantic.input_addrs));
    GE_ASSERT_SUCCESS(Om2ModelUtils::ResolveOutputAddrs(context, true, build_data_.semantic.output_addrs));
  } else {
    GE_ASSERT_SUCCESS(Om2ModelUtils::ResolveWorkspaceAddrs(context, build_data_.semantic.workspace_addrs));
    GE_ASSERT_SUCCESS(Om2ModelUtils::ResolveInputAddrs(context, build_data_.semantic.input_addrs));
    GE_ASSERT_SUCCESS(Om2ModelUtils::ResolveOutputAddrs(context, true, build_data_.semantic.output_addrs));
  }
  if (IsAllKernelTask(build_data_.semantic)) {
    const auto tiling_info =
        context.op_desc->GetExtAttr<std::shared_ptr<optiling::utils::OpRunInfo>>(ge::ATTR_NAME_OP_RUN_INFO);
    if ((tiling_info != nullptr) && (*tiling_info != nullptr)) {
      build_data_.semantic.tiling_key = (*tiling_info)->GetTilingKey();
    }
  }
  return SUCCESS;
}

Status KernelTaskCodeBuilder::BuildLaunchSemantic(const TaskSemanticContributeContext &context) {
  GE_ASSERT_NOTNULL(context.func_handle_indices);
  GE_ASSERT_SUCCESS(BuildLaunchConfigSemantic(context));
  GE_ASSERT_SUCCESS(BuildLaunchFuncHandleSemantic(context));
  return SUCCESS;
}

Status KernelTaskCodeBuilder::BuildLaunchConfigSemantic(const TaskSemanticContributeContext &context) {
  auto &launch_semantic = build_data_.semantic.launch;
  launch_semantic.stream_id = context.task_def.stream_id();
  if ((build_data_.semantic.task_type == ModelTaskType::MODEL_TASK_VECTOR_ALL_KERNEL) ||
      (build_data_.semantic.task_type == ModelTaskType::MODEL_TASK_VECTOR_KERNEL)) {
    launch_semantic.config.engine_type = "ACL_RT_ENGINE_TYPE_AIV";
  }
  bool op_exec_never_timeout = false;
  if (AttrUtils::GetBool(context.op_desc, public_attr::OP_EXEC_NEVER_TIMEOUT, op_exec_never_timeout) &&
      op_exec_never_timeout) {
    launch_semantic.config.time_out = op_exec_never_timeout;
  }
  (void)AttrUtils::GetInt(context.op_desc, kLocalMemorySize, launch_semantic.config.local_memory_size);
  if (IsAllKernelTask(build_data_.semantic)) {
    const auto &kernel_def = context.task_def.kernel_with_handle();
    launch_semantic.config.block_dim_offset = kernel_def.block_dim_offset();
    launch_semantic.config.is_block_task_prefetch = kernel_def.is_block_task_prefetch();
    launch_semantic.block_dim = kernel_def.block_dim() == 0U ? 1U : kernel_def.block_dim();
    launch_semantic.config.schedule_mode = static_cast<uint8_t>(kernel_def.schedule_mode() & k2BitsMask);
  } else {
    const auto &kernel_def = context.task_def.kernel();
    launch_semantic.config.block_dim_offset = kernel_def.block_dim_offset();
    launch_semantic.config.is_block_task_prefetch = kernel_def.is_block_task_prefetch();
    launch_semantic.block_dim = kernel_def.block_dim() == 0U ? 1U : kernel_def.block_dim();
    const auto kernel_type = static_cast<ccKernelType>(kernel_def.context().kernel_type());
    if (Om2CodegenUtils::IsAICoreKernel(kernel_type)) {
      launch_semantic.config.schedule_mode = static_cast<uint8_t>(kernel_def.schedule_mode() & k2BitsMask);
    }
  }
  return SUCCESS;
}

Status KernelTaskCodeBuilder::BuildLaunchFuncHandleSemantic(const TaskSemanticContributeContext &context) {
  auto &launch_semantic = build_data_.semantic.launch;
  std::string kernel_name;
  GE_ASSERT_SUCCESS(ResolveKernelName(build_data_.semantic, context.op_desc, context.task_def, kernel_name));
  const std::string func_handle_key = ResolveFuncHandleKey(context, kernel_name);
  const auto func_handle_it = context.func_handle_indices->find(func_handle_key);
  GE_ASSERT_TRUE(func_handle_it != context.func_handle_indices->end(), "[OM2] Func handle key %s not found.",
                 func_handle_key.c_str());
  launch_semantic.func_handle_index = func_handle_it->second;
  GELOGI("[OM2] BuildLaunchSemantic: op=%s, func_handle_key=%s, func_idx=%u", context.op_desc->GetNamePtr(),
         func_handle_key.c_str(), launch_semantic.func_handle_index);
  return SUCCESS;
}

Status KernelTaskCodeBuilder::CopyTilingDataIfNeeded(const TaskSemanticContributeContext &context,
                                                     const ArgsFormatInfo &args_format_holder) {
  std::shared_ptr<optiling::utils::OpRunInfo> default_tiling = nullptr;
  std::shared_ptr<optiling::utils::OpRunInfo> run_info = nullptr;
  if (is_separately_clean_task_) {
    // 预埋，走不进去
    run_info = context.op_desc->TryGetExtAttr(ge::ATTR_NAME_ATOMIC_OP_RUN_INFO, default_tiling);
  } else {
    run_info = context.op_desc->TryGetExtAttr(ge::ATTR_NAME_OP_RUN_INFO, default_tiling);
  }

  if (is_soft_sync_op_ && Om2CodegenUtils::IsAllKernel(context.task_type)) {
    REPORT_INNER_ERR_MSG("E19999", "Unsupported scenario for soft sync.");
    GELOGE(FAILED, "[OM2] Unsupported scenario, is_soft_sync_op_[%d], task_type[%d].", is_soft_sync_op_,
           static_cast<int32_t>(context.task_type));
    return FAILED;
  }

  if (run_info != nullptr) {
    if (run_info->GetAllTilingData().str().empty()) {
      GELOGD("Tiling data of %s is empty.", context.op_desc->GetNamePtr());
      return SUCCESS;
    }
    has_tiling_ = true;
    tiling_data_ = run_info->GetAllTilingData().str();
    if (!is_separately_clean_task_) {
      std::string dfx_info;
      GE_CHK_STATUS_RET(ConstructDfxInfo(context.op_desc, *run_info, args_format_holder.arg_descs, dfx_info),
                        "Append memcheck data for node: %s failed.", context.op_desc->GetNamePtr());
      tiling_data_ += dfx_info;
    }
    GELOGI("Success to update tiling data to io_addr of %s, tiling data %s, size: %zu.", context.op_desc->GetNamePtr(),
           tiling_data_.c_str(), tiling_data_.size());
  }
  return SUCCESS;
}

Status KernelTaskCodeBuilder::ConstructDfxInfo(const ge::OpDescPtr &op_desc, const optiling::OpRunInfoV2 &run_info,
                                               const std::vector<ge::ArgDesc> &arg_descs, std::string &dfx_info) const {
  bool is_mem_check_enable = false;
  (void)ge::AttrUtils::GetBool(op_desc, optiling::kMemoryCheck, is_mem_check_enable);
  if (!is_mem_check_enable) {
    return ge::SUCCESS;
  }

  GE_ASSERT_NOTNULL(op_desc);
  auto input_descs = op_desc->GetAllInputsDescPtr();

  // 获取size
  std::vector<int64_t> args_size_vec;
  std::vector<optiling::ArgsIndexToIoIndex> args_idx_to_io_idx_vec;
  if (!arg_descs.empty()) {
    GE_ASSERT_SUCCESS(
        optiling::TilingDfx::GetArgsSizeWithArgsFormat(op_desc, arg_descs, args_size_vec, args_idx_to_io_idx_vec));
  } else {
    GELOGI("OP [%s] not has formatted args_format. input desc size [%zu], out desc size [%zu]", op_desc->GetNamePtr(),
           input_descs.size(), op_desc->GetOutputsSize());
    GE_ASSERT_SUCCESS(optiling::TilingDfx::GetArgsSizeWithoutArgsFormat(input_descs.size(), op_desc->GetOutputsSize(),
                                                                        args_size_vec, args_idx_to_io_idx_vec));
  }

  std::vector<int64_t> shape_size_vec;
  GE_ASSERT_SUCCESS(UpdateDfxArgsAndShapeSize(op_desc, args_idx_to_io_idx_vec, args_size_vec, shape_size_vec));
  (void)args_size_vec.insert(args_size_vec.cend(), run_info.GetAllWorkspaces().cbegin(),
                             run_info.GetAllWorkspaces().cend());

  // tiling data为0的场景 或者args Size 为0的场景，直接返回
  const int64_t tiling_data_size = static_cast<int64_t>(run_info.GetAllTilingData().str().size());
  if ((tiling_data_size == 0) || (args_size_vec.size() == 0U)) {
    return ge::SUCCESS;
  }

  int64_t max_size = -1;
  if (!ge::AttrUtils::GetInt(op_desc, kMaxTilingSize, max_size) || max_size < 0) {
    GELOGI("No max tiling size in opdesc.");
    max_size = static_cast<int64_t>(kMaxTilingDataSize);
  }

  const auto memcheck_info_capacity = ge::RoundUp(static_cast<uint64_t>(max_size), sizeof(uintptr_t));
  GELOGI("Get memcheck info capacity: %zu, op_name: %s", memcheck_info_capacity, op_desc->GetNamePtr());
  const auto memcheck_data_holder = gert::TilingData::CreateCap(memcheck_info_capacity);
  auto memcheck_data = reinterpret_cast<gert::TilingData *>(memcheck_data_holder.get());
  int64_t memcheck_start_size = 0L;
  GE_ASSERT_SUCCESS(GetMemCheckStartSize(op_desc, tiling_data_size, memcheck_start_size));
  memcheck_data->SetDataSize(static_cast<size_t>(memcheck_start_size));

  // append size
  for (size_t i = 0U; i < args_size_vec.size(); i++) {
    GELOGI("[TilingAppendDfxInfo] size idx[%zu], val[%lld]", i, args_size_vec[i]);
  }
  GE_ASSERT_SUCCESS(memcheck_data->Append(args_size_vec.data(), args_size_vec.size()));
  GELOGI("Op name[%s] memcheck info size: %lld, start size: %lld", op_desc->GetNamePtr(), memcheck_data->GetDataSize(),
         memcheck_start_size);
  dfx_info = std::string(reinterpret_cast<ge::char_t *>(memcheck_data->GetData()), memcheck_data->GetDataSize());
  return ge::SUCCESS;
}

Status KernelTaskCodeBuilder::UpdateDfxArgsAndShapeSize(
    const OpDescPtr &op_desc, const std::vector<optiling::ArgsIndexToIoIndex> &args_idx_to_io_idx_vec,
    std::vector<int64_t> &args_size_vec, std::vector<int64_t> &shape_size_vec) const {
  auto input_descs = op_desc->GetAllInputsDescPtr();

  // 更新size以及shape
  for (size_t i = 0U; i < args_idx_to_io_idx_vec.size(); i++) {
    const size_t io_index = args_idx_to_io_idx_vec[i].io_index;
    const size_t args_index = args_idx_to_io_idx_vec[i].args_index;
    GE_ASSERT(args_index < args_size_vec.size(), "args index [%zu] not less than args list size [%zu]", args_index,
              args_size_vec.size());

    if (args_idx_to_io_idx_vec[i].args_role == optiling::ArgsRole::kInput) {
      const auto tensor = input_descs.at(io_index);
      GE_ASSERT_NOTNULL(tensor);
      int64_t tensor_size = 0;
      GE_ASSERT_SUCCESS(ge::TensorUtils::GetSize(*tensor, tensor_size));
      GELOGI("Update input tensor size, node[%s], index:%zu, args index: %zu, io index: %zu, tensor size: %lld",
             op_desc->GetNamePtr(), i, args_index, io_index, tensor_size);
      args_size_vec[args_index] = tensor_size;
      // shape
      AppendShapeInfo(tensor->GetShape(), shape_size_vec);
    } else if (args_idx_to_io_idx_vec[i].args_role == optiling::ArgsRole::kOutput) {
      const auto tensor = op_desc->GetOutputDesc(static_cast<uint32_t>(io_index));
      int64_t tensor_size = 0L;
      GE_ASSERT_SUCCESS(ge::TensorUtils::GetSize(tensor, tensor_size));
      GELOGI("Update output tensor size, node[%s], index:%zu, args index: %zu, io index: %zu, tensor size: %lld",
             op_desc->GetNamePtr(), i, args_index, io_index, tensor_size);
      args_size_vec[args_index] = tensor_size;
      // shape
      shape_size_vec.push_back(0);
    }
  }
  return ge::SUCCESS;
}

void KernelTaskCodeBuilder::AppendShapeInfo(const ge::GeShape &shape, std::vector<int64_t> &shape_info_vec) const {
  const auto dim_num = shape.GetDimNum();
  shape_info_vec.push_back(static_cast<int64_t>(dim_num));
  GELOGD("[AppendShapeInfo] Append shape num: %zu", dim_num);
  if (dim_num > 0) {
    const auto dims = shape.GetDims();
    for (size_t i = 0; i < dim_num; i++) {
      shape_info_vec.push_back(dims[i]);
    }
  }
}

ge::Status KernelTaskCodeBuilder::GetMemCheckStartSize(const ge::OpDescPtr &op_desc,
                                                       const int64_t origin_tiling_data_size,
                                                       int64_t &memcheck_start_size) const {
  int64_t ori_param_size = 0LL;
  (void)ge::AttrUtils::GetInt(op_desc, optiling::kOriOpParaSize, ori_param_size);
  if (ori_param_size > 0LL) {
    // tik场景下TilingAppendMem添加的数据需要从偏移为ori_param_size的地址开始添加，此处需要将DataSize设置成ori_param_size
    GE_ASSERT_TRUE(origin_tiling_data_size <= ori_param_size);
    GELOGI("Current tiling data size: %zu, set ori_param_size to %lld by attr, op_name: %s", origin_tiling_data_size,
           ori_param_size, op_desc->GetNamePtr());
  } else {
    ori_param_size = static_cast<int64_t>(
        ((static_cast<uint64_t>(origin_tiling_data_size) + sizeof(int64_t) - 1UL) / sizeof(int64_t)) * sizeof(int64_t));
    GELOGI("Current tiling data size: %zu, set ori_param_size to %lld by aligned by %zu, op_name: %s",
           origin_tiling_data_size, ori_param_size, sizeof(int64_t), op_desc->GetNamePtr());
  }
  memcheck_start_size = ori_param_size - origin_tiling_data_size;
  return ge::SUCCESS;
}

Status KernelTaskCodeBuilder::CheckTaskSupport() const {
  if (op_need_print_) {
    REPORT_INNER_ERR_MSG("E19999", "Unsupported scenario for dfx.");
    GELOGE(FAILED, "Unsupported scenario for dfx.");
    return FAILED;
  }
  if (is_soft_sync_op_) {
    REPORT_INNER_ERR_MSG("E19999", "Unsupported scenario for dfx.");
    GELOGE(FAILED, "Unsupported scenario for static_to_dynamic_softsync_op.");
    return FAILED;
  }
  if (is_blocking_aicpu_op_) {
    REPORT_INNER_ERR_MSG("E19999", "Unsupported scenario for dfx.");
    GELOGE(FAILED, "Unsupported scenario for blocking_op.");
    return FAILED;
  }
  return SUCCESS;
}

Status KernelTaskCodeBuilder::RenderDistHelper(std::vector<DeclNode *> &items) {
  (void)items.push_back(ast_.Field("constexpr int64_t", "kDImEndFlag = std::numeric_limits<int64_t>::min()"));
  (void)items.push_back(RenderKernelTaskDistribute());
  (void)items.push_back(
      ast_.Field("constexpr uint32_t", "kAicpuArgsExtInfoAddrOffset", ast_.UInt(kAicpuArgsExtInfoAddrOffset)));
  (void)items.push_back(
      ast_.Field("constexpr uint32_t", "kAicpuArgsio_addr_offset", ast_.UInt(kAicpuArgsioAddrOffset)));
  (void)items.push_back(RenderUpdateExtInfoSession());
  (void)items.push_back(RenderAssembleAicpuExtInfo());
  (void)items.push_back(RenderAssembleAicpuArgs());
  (void)items.push_back(RenderAicpuKernelTaskDistribute());
  (void)items.push_back(RenderGetEventIdAddr());

  // dispatch function — 拆分为 AICORE/AICPU 独立函数 + 薄壳 wrapper
  auto op = ast_.Var("const TaskDispatchInfo *", "op");
  auto ctx = ast_.Var("const DispatchOpContext &", "ctx");

  GE_ASSERT_SUCCESS(RenderDispatchAicore(op, ctx, items));
  GE_ASSERT_SUCCESS(RenderDispatchAicpu(op, ctx, items));

  // DispatchKernel 薄壳 wrapper: 根据 dispatch_type 分发到对应子函数
  std::vector<BodyItem> wrapper_body;
  wrapper_body.push_back(ast_.If(ast_.Var("", "(uint32_t)op->dispatch_type") == ast_.Var("", "DISPATCH_AICPU"),
                                 {ast_.Return(ast_.Call("DispatchKernelAicpu", {Arg(op), Arg(ctx)}))},
                                 {ast_.Return(ast_.Call("DispatchKernelAicore", {Arg(op), Arg(ctx)}))}));
  GE_ASSERT_SUCCESS(TaskCodeBuilderUtil::RenderDispatchFunc(ast_, kDispatchFuncName, wrapper_body, items));
  return SUCCESS;
}

FunctionDef *KernelTaskCodeBuilder::RenderKernelTaskDistribute() const {
  auto io_addrs = ast_.Var("const std::vector<uint64_t> &", "io_addrs");
  auto args_info = ast_.Var("ArgsInfo *", "args_info");
  auto func_handle = ast_.Var("aclrtFuncHandle", "func_handle");
  auto block_dim = ast_.Var("uint32_t", "block_dim");
  auto stream = ast_.Var("aclrtStream", "stream");
  auto config = ast_.Var("aclrtLaunchKernelCfg *", "config");
  return ast_.DefineFunction("KernelTaskDistribute", {io_addrs, args_info, func_handle, block_dim, stream, config},
                             "aclError",
                             {
                                 ChkNotNull(args_info),
                                 ChkStatus(MemcpyS(args_info.Arrow("host_addr"), args_info.Arrow("size"),
                                                   io_addrs.Data(), io_addrs.Size() * ast_.Sizeof("uint64_t"))),
                                 ChkStatus(AclrtLaunchKernelV2(func_handle, block_dim, args_info.Arrow("dev_addr"),
                                                               args_info.Arrow("size"), config, stream)),
                                 ast_.Return("ACL_SUCCESS"),
                             });
}

FunctionDef *KernelTaskCodeBuilder::RenderUpdateExtInfoSession() const {
  auto ext_info = ast_.Var("uint8_t *", "extInfo");
  auto session_info_offset = ast_.Var("size_t", "session_info_offset");
  auto session_id = ast_.Var("uint64_t *", "session_id");
  auto kernel_id = ast_.Var("uint64_t *", "kernel_id");
  auto session_info = ast_.Var("AicpuSessionInfo *", "session_info");
  return ast_.DefineFunction(
      "UpdateExtInfoSession", {ext_info, session_info_offset, session_id, kernel_id}, "aclError",
      {
          ast_.VarDecl(session_info, ast_.ReinterpretCast("AicpuSessionInfo *", ext_info[session_info_offset].Addr())),
          ast_.Assign(session_info.Arrow("sessionId"), ast_.Deref(session_id)),
          ast_.Assign(session_info.Arrow("kernelId"), ast_.Deref(kernel_id)),
          ast_.Assign(session_info.Arrow("sessFlag"), true),
          ast_.PreInc(ast_.Deref(kernel_id)),
          ast_.Return("ACL_SUCCESS"),
      });
}

FunctionDef *KernelTaskCodeBuilder::RenderAssembleAicpuExtInfo() const {
  auto ext_info = ast_.Var("const uint8_t *", "ext_info");
  auto ext_info_len = ast_.Var("size_t", "ext_info_len");
  auto session_info_offset = ast_.Var("int32_t", "session_info_offset");
  auto session_id = ast_.Var("uint64_t *", "session_id");
  auto kernel_id = ast_.Var("uint64_t *", "kernel_id");
  auto dev_ext_info_mem_ptrs = ast_.Var("std::vector<void *> &", "dev_ext_info_mem_ptrs");
  auto index = ast_.Var("size_t", "index");
  auto tmp_ext_info = ast_.Var("std::unique_ptr<uint8_t[]>", "tmp_ext_info");
  auto dev_ptr = ast_.Var("void *", "dev_ptr");
  return ast_.DefineFunction(
      "AssembleAicpuExtInfo",
      {ext_info, ext_info_len, session_info_offset, session_id, kernel_id, dev_ext_info_mem_ptrs, index}, "aclError",
      {
          ast_.VarDecl(tmp_ext_info, ast_.MakeUniqueArray(BuiltinType::kUInt8, ext_info_len)),
          MemcpyS(tmp_ext_info.GetPtr(), ext_info_len, ext_info, ext_info_len),
          ast_.If(session_info_offset != -1,
                  {
                      ChkStatus(ast_.Call("UpdateExtInfoSession",
                                          {tmp_ext_info.GetPtr(), session_info_offset, session_id, kernel_id})),
                  }),
          ast_.VarDecl(dev_ptr, nullptr),
          ChkStatus(AclrtMallocAlign32(dev_ptr.Addr(), ext_info_len, "ACL_MEM_MALLOC_HUGE_FIRST")),
          ChkStatus(
              AclrtMemcpy(dev_ptr, ext_info_len, tmp_ext_info.GetPtr(), ext_info_len, "ACL_MEMCPY_HOST_TO_DEVICE")),
          ast_.Assign(dev_ext_info_mem_ptrs[index], dev_ptr),
          ast_.Return("ACL_SUCCESS"),
      });
}

FunctionDef *KernelTaskCodeBuilder::RenderAssembleAicpuArgs() const {
  auto args = ast_.Var("const uint8_t *", "args");
  auto args_len = ast_.Var("size_t", "args_len");
  auto ext_info_addr = ast_.Var("void *", "ext_info_addr");
  auto ext_info_len = ast_.Var("size_t", "ext_info_len");
  auto io_addr = ast_.Var("std::vector<uint64_t> &", "io_addr");
  auto target_args_ptr = ast_.Var("void *", "target_args_ptr");
  auto tmp_args = ast_.Var("std::unique_ptr<uint8_t[]>", "tmp_args");
  auto aicpu_param_head = ast_.Var("const auto", "aicpu_param_head");
  auto ext_info_addr_value = ast_.Var("uint64_t", "ext_info_addr_value");
  auto addrs_size = ast_.Var("size_t", "addrs_size");
  return ast_.DefineFunction(
      "AssembleAicpuArgs", {args, args_len, ext_info_addr, ext_info_len, io_addr, target_args_ptr}, "aclError",
      {
          ast_.VarDecl(tmp_args, ast_.MakeUniqueArray(BuiltinType::kUInt8, args_len)),
          MemcpyS(tmp_args.GetPtr(), args_len, args, args_len),
          ast_.VarDecl(aicpu_param_head, ast_.ReinterpretCast("AicpuParamHead *", tmp_args.GetPtr())),
          ast_.Assign(aicpu_param_head.Arrow("extInfoLength"), ast_.StaticCast("uint32_t", ext_info_len)),
          ast_.VarDecl(ext_info_addr_value, ast_.ReinterpretCast("uint64_t", ext_info_addr)),
          MemcpyS(tmp_args.GetPtr() + "kAicpuArgsExtInfoAddrOffset", ast_.Sizeof("uint64_t"),
                  ext_info_addr_value.Addr(), ast_.Sizeof("uint64_t")),
          ast_.VarDecl(addrs_size, ast_.Sizeof("uint64_t") * io_addr.Size()),
          MemcpyS(tmp_args.GetPtr() + "kAicpuArgsio_addr_offset", addrs_size, io_addr.Data(), addrs_size),
          MemcpyS(target_args_ptr, args_len, tmp_args.GetPtr(), args_len),
          ast_.Return("ACL_SUCCESS"),
      });
}

FunctionDef *KernelTaskCodeBuilder::RenderAicpuKernelTaskDistribute() const {
  auto args = ast_.Var("const std::vector<uint8_t> &", "args");
  auto args_info = ast_.Var("ArgsInfo *", "args_info");
  auto func_handle = ast_.Var("aclrtFuncHandle", "func_handle");
  auto block_dim = ast_.Var("uint32_t", "block_dim");
  auto stream = ast_.Var("aclrtStream", "stream");
  auto config = ast_.Var("aclrtLaunchKernelCfg *", "config");
  return ast_.DefineFunction(
      "AicpuKernelTaskDistribute", {args, args_info, func_handle, block_dim, stream, config}, "aclError",
      {
          ChkNotNull(args_info),
          ChkStatus(MemcpyS(args_info.Arrow("host_addr"), args_info.Arrow("size"), args.Data(), args.Size())),
          ChkStatus(AclrtLaunchKernelV2(func_handle, block_dim, args_info.Arrow("dev_addr"), args_info.Arrow("size"),
                                        config, stream)),
          ast_.Return("ACL_SUCCESS"),
      });
}

FunctionDef *KernelTaskCodeBuilder::RenderGetEventIdAddr() const {
  auto event_addr = ast_.Var("void *&", "event_addr");
  auto event_id_mem_map = ast_.Var("std::map<uint32_t, void *> &", "event_id_mem_map");
  auto event_id = ast_.Var("uint32_t", "event_id");
  auto mem_ptrs = ast_.Var("std::vector<void *> &", "mem_ptrs");
  auto it = ast_.Var("auto", "it");

  constexpr size_t mem_event_size = 8;

  return ast_.DefineFunction("GetEventIdAddr", {event_addr, event_id_mem_map, event_id, mem_ptrs}, "aclError",
                             {
                                 ast_.VarDecl(it, event_id_mem_map.Attr("find")(event_id)),
                                 ast_.If(event_id_mem_map.Attr("end")() != "it",
                                         {
                                             ast_.Assign(event_addr, it.Arrow("second")),
                                             ast_.Return("ACL_SUCCESS"),
                                         }),
                                 ChkStatus(ast_.Call("MallocDeviceMemory", {event_addr, mem_event_size, 2, mem_ptrs})),
                                 ChkStatus(ast_.Call("aclrtMemset", {event_addr, mem_event_size, 0, mem_event_size})),
                                 ast_.Assign(event_id_mem_map[event_id], event_addr),
                                 ast_.Return("ACL_SUCCESS"),
                             });
}

void KernelTaskCodeBuilder::AssignTaskLocalIoNames() {
  const std::string task_prefix = "op" + std::to_string(header_.op_index);
  for (size_t i = 0U; i < build_data_.semantic.input_addrs.size(); ++i) {
    if (build_data_.semantic.input_addrs[i].tensor_info.has_value()) {
      build_data_.semantic.input_addrs[i].symbol_hint = task_prefix + "_input" + std::to_string(i);
    }
  }
  for (size_t i = 0U; i < build_data_.semantic.output_addrs.size(); ++i) {
    if (build_data_.semantic.output_addrs[i].tensor_info.has_value()) {
      build_data_.semantic.output_addrs[i].symbol_hint = task_prefix + "_output" + std::to_string(i);
    }
  }
}

Status KernelTaskCodeBuilder::GetKernelTaskMeta(const domi::TaskDef &task_def, domi::KernelContext &kernel_context,
                                                uint32_t &args_size, uint32_t &kernel_type) const {
  if (Om2CodegenUtils::IsAllKernel(static_cast<ModelTaskType>(task_def.type()))) {
    const domi::KernelDefWithHandle &kernel_def = task_def.kernel_with_handle();
    args_size = static_cast<uint32_t>(kernel_def.args().size());
    kernel_context = kernel_def.context();
  } else {
    const domi::KernelDef &kernel_def = task_def.kernel();
    args_size = static_cast<uint32_t>(kernel_def.args().size());
    kernel_context = kernel_def.context();
  }
  kernel_type = kernel_context.kernel_type();
  return SUCCESS;
}

std::string KernelTaskCodeBuilder::SerializeBytesToOctalString(const std::vector<uint8_t> &buffer) const {
  std::ostringstream code_stream;
  for (size_t i = 0; i < buffer.size(); ++i) {
    code_stream << "\\";
    code_stream << std::oct << std::setw(kWidthPerChar) << std::setfill('0') << static_cast<int32_t>(buffer[i]);
  }
  return code_stream.str();
}

int64_t KernelTaskCodeBuilder::ParseOpIndex(const domi::TaskDef &task_def) {
  const auto task_type = static_cast<ModelTaskType>(task_def.type());
  domi::KernelContext context;
  if (!Om2CodegenUtils::IsAllKernel(task_type)) {
    const domi::KernelDef &kernel_def = task_def.kernel();
    context = kernel_def.context();
  } else {
    const domi::KernelDefWithHandle &kernel_def = task_def.kernel_with_handle();
    context = kernel_def.context();
  }
  return static_cast<int64_t>(context.op_index());
}

Status KernelTaskCodeBuilder::UpdateShapeAndType(const GeShape &shape, AicpuShapeAndType *const shape_and_type) const {
  const auto dim_num = shape.GetDimNum();
  if (dim_num > aicpu::FWKAdapter::kMaxShapeDims) {
    GELOGE(ACL_ERROR_GE_PARAM_INVALID,
           "[OM2][Check][DimNum]Update shape and type failed because dim_num %zu exceeds the maximum shape dims %u.",
           dim_num, aicpu::FWKAdapter::kMaxShapeDims);
    REPORT_INNER_ERR_MSG("E19999",
                         "Update shape and type failed because dim_num %zu exceeds the maximum shape dims %u.", dim_num,
                         aicpu::FWKAdapter::kMaxShapeDims);
    return ACL_ERROR_GE_PARAM_INVALID;
  }
  size_t index = 0U;
  for (; index < dim_num; ++index) {
    shape_and_type->dims[index] = shape.GetDim(index);
  }
  if (index < aicpu::FWKAdapter::kMaxShapeDims) {
    shape_and_type->dims[index] = kDimEndFlag;
  }

  // now only support update shape, type is not support
  return SUCCESS;
}

Status KernelTaskCodeBuilder::ParseExtShape(AicpuExtInfo &aicpu_ext_info, const uint32_t num_tensor,
                                            const std::string &node_name, const bool all_shape,
                                            const OpDescPtr &op_desc) const {
  std::vector<AicpuShapeAndType *> shape_and_type;
  shape_and_type.clear();
  GE_IF_BOOL_EXEC(
      aicpu_ext_info.infoLen != (num_tensor * sizeof(AicpuShapeAndType)),
      REPORT_INNER_ERR_MSG("E19999",
                           "Node[%s] parse ext shape failed as infoLen must be "
                           "tensor_num[%u]*sizeof(ShapeAndType)[%zu] but %u.",
                           node_name.c_str(), num_tensor, sizeof(AicpuShapeAndType), aicpu_ext_info.infoLen);
      GELOGE(ACL_ERROR_GE_PARAM_INVALID,
             "[OM2][Check][DataLen]Node[%s] parse ext shape failed as infoLen must be "
             "tensor_num[%u]*sizeof(ShapeAndType)[%zu] but %u.",
             node_name.c_str(), num_tensor, sizeof(AicpuShapeAndType), aicpu_ext_info.infoLen);
      return ACL_ERROR_GE_PARAM_INVALID;);
  const auto tensor_info = PtrToPtr<char, AicpuShapeAndType>(aicpu_ext_info.infoMsg);
  if (all_shape) {
    for (uint32_t i = 0U; i < num_tensor; ++i) {
      (void)shape_and_type.emplace_back(
          PtrAdd<AicpuShapeAndType>(tensor_info, static_cast<size_t>(num_tensor), static_cast<size_t>(i)));
      const auto tensor_desc = op_desc->MutableInputDesc(i);
      GE_CHECK_NOTNULL(tensor_desc);
      const auto &shape = tensor_desc->GetShape();
      GE_CHK_STATUS_RET(UpdateShapeAndType(shape, shape_and_type[static_cast<size_t>(i)]),
                        "[OM2][Update][ShapeAndType] failed, Node[%s] tensor_info[%u] .", node_name.c_str(), i);
    }
  }
  GELOGI("[OM2]Node[%s] parse ext shape success infoLen=%u.", node_name.c_str(), aicpu_ext_info.infoLen);
  return SUCCESS;
}

Status KernelTaskCodeBuilder::ParseExtBitmap(AicpuExtInfo &aicpu_ext_info, const std::string &node_name) const {
  GE_IF_BOOL_EXEC(
      aicpu_ext_info.infoLen != sizeof(uint64_t),
      REPORT_INNER_ERR_MSG("E19999", "Node[%s] parse bit_map info failed as infoLen must be %zu but %u.",
                           node_name.c_str(), sizeof(uint64_t), aicpu_ext_info.infoLen);
      GELOGE(PARAM_INVALID, "[OM2][Check][DataLen]Node[%s] parse bit_map info failed as infoLen must be %zu but %u.",
             node_name.c_str(), sizeof(uint64_t), aicpu_ext_info.infoLen);
      return PARAM_INVALID;);

  uint64_t *bit_map = PtrToPtr<char, uint64_t>(aicpu_ext_info.infoMsg);
  *(bit_map) |= 1UL;
  GELOGI("[OM2] Node[%s] bit_map info success infoLen=%u, value = %" PRIu64 ".", node_name.c_str(),
         aicpu_ext_info.infoLen, *(bit_map));
  return SUCCESS;
}

Status KernelTaskCodeBuilder::ParseExtTopicType(AicpuExtInfo &aicpu_ext_info, const std::string &node_name) const {
  if (aicpu_ext_info.infoLen != sizeof(int32_t)) {
    REPORT_INNER_ERR_MSG("E19999", "Node[%s] parse topic_type info failed as infoLen must be %zu but %u.",
                         node_name.c_str(), sizeof(int32_t), aicpu_ext_info.infoLen);
    GELOGE(ACL_ERROR_GE_PARAM_INVALID,
           "[Check][DataLen]Node[%s] parse topic_type info failed as infoLen must be %zu but %u.", node_name.c_str(),
           sizeof(int32_t), aicpu_ext_info.infoLen);
    return ACL_ERROR_GE_PARAM_INVALID;
  }
  GE_CHECK_NOTNULL(aicpu_ext_info.infoMsg);
  const int32_t type = *(PtrToPtr<char, int32_t>(aicpu_ext_info.infoMsg));
  const int32_t deploy_type_flag = Om2CodegenUtils::TopicTypeToRtsFlag(type);
  if (deploy_type_flag == -1) {
    REPORT_INNER_ERR_MSG("E19999", "Node[%s] parse ext topic type failed because it requires %d %d %d %d but got %d.",
                         node_name.c_str(), aicpu::FWKAdapter::FWK_ADPT_TOPIC_DEVICE_ONLY,
                         aicpu::FWKAdapter::FWK_ADPT_TOPIC_DEVICE_FIRST, aicpu::FWKAdapter::FWK_ADPT_TOPIC_HOST_ONLY,
                         aicpu::FWKAdapter::FWK_ADPT_TOPIC_HOST_FIRST, type);
    GELOGE(ACL_ERROR_GE_PARAM_INVALID,
           "[Check][Type]Node[%s] parse ext topic type failed because it requires %d %d %d %d but got %d.",
           node_name.c_str(), aicpu::FWKAdapter::FWK_ADPT_TOPIC_DEVICE_ONLY,
           aicpu::FWKAdapter::FWK_ADPT_TOPIC_DEVICE_FIRST, aicpu::FWKAdapter::FWK_ADPT_TOPIC_HOST_ONLY,
           aicpu::FWKAdapter::FWK_ADPT_TOPIC_HOST_FIRST, type);
    return ACL_ERROR_GE_PARAM_INVALID;
  } else if (deploy_type_flag == static_cast<int32_t>(RT_KERNEL_HOST_ONLY)) {
    REPORT_INNER_ERR_MSG("E19999", "Unsupported scenario. Node[%s], infoType=%d, infoLen=%u.", node_name.c_str(),
                         aicpu_ext_info.infoType, aicpu_ext_info.infoLen);
    GELOGE(FAILED, "[OM2] Unsupported scenario. Node[%s], infoType=%d, infoLen=%u.", node_name.c_str(),
           aicpu_ext_info.infoType, aicpu_ext_info.infoLen);
    return FAILED;
  }
  return SUCCESS;
}

Status KernelTaskCodeBuilder::ParseExtAsyncWait(AicpuExtInfo &aicpu_ext_info, const std::string &node_name) const {
  if (aicpu_ext_info.infoLen != sizeof(aicpu::FWKAdapter::AsyncWait)) {
    REPORT_INNER_ERR_MSG("E19999", "Node[%s] parse ext async wait info failed as infoLen must be %zu but %u.",
                         node_name.c_str(), sizeof(aicpu::FWKAdapter::AsyncWait), aicpu_ext_info.infoLen);
    GELOGE(ACL_ERROR_GE_PARAM_INVALID,
           "[Check][DataLen]Node[%s] parse ext async wait info failed as infoLen must be %zu but %u.",
           node_name.c_str(), sizeof(aicpu::FWKAdapter::AsyncWait), aicpu_ext_info.infoLen);
    return ACL_ERROR_GE_PARAM_INVALID;
  }
  return SUCCESS;
}

Status KernelTaskCodeBuilder::ParseExtInfo(uint8_t *ext_info, const size_t ext_info_len, const OpDescPtr &op_desc,
                                           int32_t &session_info_offset, const uint32_t num_inputs,
                                           const uint32_t num_outputs, const std::string &node_name,
                                           const bool all_shape) const {
  size_t offset = 0UL;
  while ((offset + sizeof(AicpuExtInfo)) <= ext_info_len) {
    auto tmp_ext_info_data = PtrAdd(ext_info, ext_info_len, offset);
    GE_CHECK_NOTNULL(tmp_ext_info_data);
    auto &aicpu_ext_info = *(PtrToPtr<uint8_t, AicpuExtInfo>(tmp_ext_info_data));
    GELOGD("[OM2] Ext infoType=%d, infoLen=%u.", aicpu_ext_info.infoType, aicpu_ext_info.infoLen);
    switch (aicpu_ext_info.infoType) {
      case aicpu::FWKAdapter::FWK_ADPT_EXT_SHAPE_TYPE:
        GELOGI("[OM2] Reserve infoType[%d] for Node[%s].", aicpu_ext_info.infoType, node_name.c_str());
        break;
      case aicpu::FWKAdapter::FWK_ADPT_EXT_INPUT_SHAPE:
        GE_CHK_STATUS_RET(ParseExtShape(aicpu_ext_info, num_inputs, node_name, all_shape, op_desc),
                          "[OM2] Parse ext input shape failed, Node[%s].", node_name.c_str());
        break;
      case aicpu::FWKAdapter::FWK_ADPT_EXT_OUTPUT_SHAPE:
        GE_CHK_STATUS_RET(ParseExtShape(aicpu_ext_info, num_outputs, node_name, all_shape, op_desc),
                          "[OM2] Parse ext output shape failed, Node[%s].", node_name.c_str());
        break;
      case aicpu::FWKAdapter::FWK_ADPT_EXT_SESSION_INFO:
        session_info_offset = static_cast<int32_t>(offset) + kSessionInfoOffset;
        break;
      case aicpu::FWKAdapter::FWK_ADPT_EXT_BITMAP:
        GE_CHK_STATUS_RET(ParseExtBitmap(aicpu_ext_info, node_name.c_str()), "[OM2] Parse ext bitmap failed, Node[%s].",
                          node_name.c_str());
        break;
      case aicpu::FWKAdapter::FWK_ADPT_EXT_TOPIC_TYPE:
        GE_CHK_STATUS_RET(ParseExtTopicType(aicpu_ext_info, node_name.c_str()),
                          "[OM2] Parse ext topic type failed, Node[%s].", node_name.c_str());
        break;
      case aicpu::FWKAdapter::FWK_ADPT_EXT_ASYNCWAIT: {
        GE_CHK_STATUS_RET(ParseExtAsyncWait(aicpu_ext_info, node_name.c_str()),
                          "[OM2] Parse ext async wait failed, Node[%s].", node_name.c_str());
        break;
      }
      default:
        GELOGD("[OM2] Node[%s] ignore infoType=%d, infoLen=%u.", node_name.c_str(), aicpu_ext_info.infoType,
               aicpu_ext_info.infoLen);
        break;
    }
    offset += sizeof(AicpuExtInfo);
    offset += aicpu_ext_info.infoLen;
  }

  GE_IF_BOOL_EXEC(offset != ext_info_len, REPORT_INNER_ERR_MSG("E19999",
                                                               "Node[%s] ext_info format error, parse not reach end,"
                                                               "offset=%zu, ext_info_len=%zu.",
                                                               node_name.c_str(), offset, ext_info_len);
                  GELOGE(ACL_ERROR_GE_PARAM_INVALID,
                         "[OM2][Check][Size]Node[%s] ext_info format error,"
                         "parse not reach end, offset=%zu, ext_info_len=%zu.",
                         node_name.c_str(), offset, ext_info_len);
                  return ACL_ERROR_GE_PARAM_INVALID;);
  return SUCCESS;
}

Status KernelTaskCodeBuilder::InitAicpuTaskExtInfo(uint8_t *ext_info, size_t ext_info_len, const OpDescPtr op_desc,
                                                   int32_t &session_info_offset) const {
  GELOGD("[OM2] start to init aicpu task ext info.");
  std::string node_name = op_desc->GetName();
  const uint32_t num_inputs = static_cast<uint32_t>(op_desc->GetInputsSize());
  const uint32_t num_outputs = static_cast<uint32_t>(op_desc->GetOutputsSize());

  std::vector<AicpuShapeAndType *> output_shape_and_type;
  output_shape_and_type.clear();

  bool all_shape = false;
  (void)AttrUtils::GetBool(op_desc, kAllShapeInAicpu, all_shape);
  GE_ASSERT_SUCCESS(ParseExtInfo(ext_info, ext_info_len, op_desc, session_info_offset, num_inputs, num_outputs,
                                 node_name, all_shape));
  GELOGI("[OM2] Node[%s] parse ext info end.", node_name.c_str());
  return SUCCESS;
}

Status KernelTaskCodeBuilder::ParseArgsFormat(const OpDescPtr &op_desc, ArgsFormatInfo &args_format_holder) const {
  GE_ASSERT_NOTNULL(op_desc);
  (void)OpDescUtils::GetIrInputInstanceDescRange(op_desc, args_format_holder.ir_input_2_range);
  (void)OpDescUtils::GetIrOutputDescRange(op_desc, args_format_holder.ir_output_2_range);
  auto &arg_descs = args_format_holder.arg_descs;
  auto input_descs = op_desc->GetAllInputsDescPtr();
  for (const auto &arg_format : arg_descs) {
    if (arg_format.addr_type == AddrType::INPUT_DESC) {
      GE_ASSERT(arg_format.ir_idx >= 0 &&
                static_cast<size_t>(arg_format.ir_idx) < args_format_holder.ir_input_2_range.size());
      const auto &ir_range = args_format_holder.ir_input_2_range[static_cast<size_t>(arg_format.ir_idx)];
      std::vector<int64_t> shape_info{0};  // placeholder for offset
      for (size_t idx = 0UL; idx < ir_range.second; ++idx) {
        const size_t instance_idx = static_cast<size_t>(ir_range.first + idx);
        GE_ASSERT_TRUE(instance_idx < input_descs.size(), "Instance index [%zu] is out of range, max_size:[%zu].",
                       instance_idx, input_descs.size());
        AppendShapeDesc(*input_descs.at(instance_idx), shape_info);
      }
      shape_info[0UL] = static_cast<int64_t>(shape_info.size()) * static_cast<int64_t>(sizeof(uintptr_t));
      args_format_holder.level1_addr_cnt += ir_range.second + shape_info.size();
      (void)args_format_holder.shape_infos.push_back(shape_info);
    } else if (arg_format.addr_type == AddrType::OUTPUT_DESC) {
      GE_ASSERT(arg_format.ir_idx >= 0 &&
                static_cast<size_t>(arg_format.ir_idx) < args_format_holder.ir_output_2_range.size());
      const auto &ir_range = args_format_holder.ir_output_2_range[static_cast<size_t>(arg_format.ir_idx)];
      std::vector<int64_t> shape_info{0};  // placeholder for offset
      args_format_holder.level1_addr_cnt += ir_range.second;
      for (size_t idx = 0UL; idx < ir_range.second; ++idx) {
        auto output_desc = op_desc->MutableOutputDesc(static_cast<uint32_t>(ir_range.first + idx));
        GE_ASSERT_NOTNULL(output_desc);
        AppendShapeDesc(*output_desc, shape_info);
      }
      shape_info[0UL] = static_cast<int64_t>(shape_info.size()) * static_cast<int64_t>(sizeof(uintptr_t));
      args_format_holder.level1_addr_cnt += ir_range.second + shape_info.size();
      (void)args_format_holder.shape_infos.push_back(shape_info);
    } else if (arg_format.addr_type == AddrType::TILING_CONTEXT &&
               (arg_format.ir_idx == static_cast<int32_t>(TilingContextSubType::TILING_CONTEXT) ||
                arg_format.ir_idx == static_cast<int32_t>(TilingContextSubType::TILING_DATA))) {
      REPORT_INNER_ERR_MSG("E19999", "Unsupported scenario. addr_type[%d], ir_idx[%d].",
                           static_cast<int32_t>(AddrType::TILING_CONTEXT), arg_format.ir_idx);
      GELOGE(FAILED, "[OM2] Unsupported scenario. addr_type[%d], ir_idx[%d].",
             static_cast<int32_t>(AddrType::TILING_CONTEXT), arg_format.ir_idx);
      return FAILED;
    }
  }
  return SUCCESS;
}

size_t KernelTaskCodeBuilder::GetArgsSizeByFormat(const OpDescPtr op_desc,
                                                  const ArgsFormatInfo &args_format_holder) const {
  const auto &arg_descs = args_format_holder.arg_descs;
  size_t tmp_size = 0U;
  for (const auto &arg_desc : arg_descs) {
    (void)ArgsFormatDesc::GetArgSize(op_desc, arg_desc, tmp_size);
  }
  return tmp_size;
}

size_t KernelTaskCodeBuilder::GetExtraArgsSize(const OpDescPtr &op_desc, const ccKernelType kernel_type,
                                               const ArgsFormatInfo &args_format_holder) const {
  size_t extra_size = 0UL;
  int32_t max_tiling_len{-1};
  (void)AttrUtils::GetInt(op_desc, kMaxTilingSize, max_tiling_len);
  int32_t max_atomic_tiling_len{-1};
  (void)AttrUtils::GetInt(op_desc, kMaxAtomicCleanTilingSize, max_atomic_tiling_len);
  if ((max_tiling_len > 0) || (max_atomic_tiling_len > 0)) {
    extra_size += kAddressLen;
  }

  if (kernel_type == ccKernelType::TE) {
    const auto is_wsp_addr_folded = IsWspAddrFolded(op_desc);
    if (is_wsp_addr_folded) {
      // kAddressLen: if folded mode, need add a memory for point to wsl addr list
      // kUBAlignedLen:
      // reserved 32B for aligned start with wsl addr list
      // -----------------------------------------------------------
      // | point to wsl addr list | over flow addr | wsl addr list |
      // -----------------------------------------------------------
      extra_size += kAddressLen + kUBAlignedLen;
    }
  }

  // level2 addr
  const size_t shape_info_size = args_format_holder.level1_addr_cnt * sizeof(int64_t);
  extra_size += shape_info_size;

  // reserved tiling sink tensor size
  return extra_size;
}

void KernelTaskCodeBuilder::InitArgsTableEntry(const TaskSemanticContributeContext &context, const uint32_t args_size) {
  (void)build_data_.semantic.args_table_entry.emplace();
  build_data_.semantic.args_table_entry->table_index = *context.next_args_table_index;
  build_data_.semantic.args_table_entry->args_size = args_size;
  build_data_.semantic.args_table_entry->host_offset = *context.next_host_args_offset;
  args_table_entry_ = &(*build_data_.semantic.args_table_entry);
}

std::vector<size_t> KernelTaskCodeBuilder::BuildMaterializedOutputIndices(
    const KernelTaskSemantic &kernel_semantic) const {
  std::vector<size_t> materialized_output_indices;
  for (size_t i = 0U; i < kernel_semantic.output_addrs.size(); ++i) {
    if (IsMaterializedOutput(kernel_semantic.output_addrs[i])) {
      materialized_output_indices.push_back(i);
    }
  }
  return materialized_output_indices;
}

void KernelTaskCodeBuilder::AppendOrderedPlaceholder(const TaskSemanticContributeContext &context) {
  AddrSemantic placeholder;
  placeholder.kind = AddrValueKind::kPlaceholder;
  placeholder.symbol_hint =
      "op" + std::to_string(context.op_index) + "_place_holder" + std::to_string(place_holder_var_index_++);
  AppendOrderedArg(placeholder);
}

void KernelTaskCodeBuilder::AppendOrderedCustomValue(const TaskSemanticContributeContext &context,
                                                     const uint64_t custom_value) {
  AddrSemantic custom_value_semantic;
  custom_value_semantic.kind = AddrValueKind::kCustomValue;
  custom_value_semantic.symbol_hint =
      "op" + std::to_string(context.op_index) + "_custom_value" + std::to_string(cust_value_var_index_++);
  custom_value_semantic.custom_value = custom_value;
  AppendOrderedArg(custom_value_semantic);
}

Status KernelTaskCodeBuilder::AppendOrderedInputArg(size_t input_idx) {
  GE_ASSERT_TRUE(input_idx < build_data_.semantic.input_addrs.size(),
                 "[OM2] Input instance idx [%zu] is invalid, size:[%zu].", input_idx,
                 build_data_.semantic.input_addrs.size());
  auto &input_addr = build_data_.semantic.input_addrs[input_idx];
  if (input_addr.tensor_info.has_value()) {
    input_addr.tensor_info->args_offset = current_args_offset_;
  }
  AppendOrderedArg(input_addr);
  return SUCCESS;
}

Status KernelTaskCodeBuilder::AppendOrderedOutputArg(size_t output_idx) {
  GE_ASSERT_TRUE(output_idx < materialized_output_indices_.size(),
                 "[OM2] Output instance idx [%zu] is invalid, size:[%zu].", output_idx,
                 materialized_output_indices_.size());
  auto &output_addr = build_data_.semantic.output_addrs[materialized_output_indices_[output_idx]];
  if (output_addr.tensor_info.has_value()) {
    output_addr.tensor_info->args_offset = current_args_offset_;
  }
  AppendOrderedArg(output_addr);
  return SUCCESS;
}

Status KernelTaskCodeBuilder::AppendOrderedInputOutputByInstanceIndex(const ArgDesc &arg_format) {
  if (arg_format.addr_type == AddrType::INPUT_INSTANCE) {
    return AppendOrderedInputArg(static_cast<size_t>(arg_format.ir_idx));
  }
  GE_ASSERT_SUCCESS(AppendOrderedOutputArg(static_cast<size_t>(arg_format.ir_idx)));
  return SUCCESS;
}

Status KernelTaskCodeBuilder::AppendOrderedInputOutputRange(const ArgDesc &arg_format,
                                                            const ArgsFormatInfo &args_format_holder,
                                                            const TaskSemanticContributeContext &context) {
  const bool is_input = (arg_format.addr_type == AddrType::INPUT);
  const auto &ir_2_range = is_input ? args_format_holder.ir_input_2_range : args_format_holder.ir_output_2_range;
  const auto iter = ir_2_range.find(static_cast<size_t>(arg_format.ir_idx));
  GE_ASSERT(iter != ir_2_range.end());
  const auto &range_pair = iter->second;
  if (is_input && range_pair.second == 0UL) {
    AppendOrderedPlaceholder(context);
    return SUCCESS;
  }
  size_t begin_idx = range_pair.first;
  while (begin_idx < range_pair.first + range_pair.second) {
    if (is_input) {
      GE_ASSERT_SUCCESS(AppendOrderedInputArg(begin_idx));
    } else {
      GE_ASSERT_SUCCESS(AppendOrderedOutputArg(begin_idx));
    }
    ++begin_idx;
  }
  return SUCCESS;
}

Status KernelTaskCodeBuilder::AppendOrderedWorkspace(const ArgDesc &arg_format) {
  if (arg_format.ir_idx < 0) {
    for (const auto &workspace_addr : build_data_.semantic.workspace_addrs) {
      AppendOrderedArg(workspace_addr);
    }
    return SUCCESS;
  }
  const size_t workspace_idx = static_cast<size_t>(arg_format.ir_idx);
  GE_ASSERT_TRUE(workspace_idx < build_data_.semantic.workspace_addrs.size(),
                 "[OM2] Workspace idx [%zu] is invalid, size:[%zu].", workspace_idx,
                 build_data_.semantic.workspace_addrs.size());
  AppendOrderedArg(build_data_.semantic.workspace_addrs[workspace_idx]);
  return SUCCESS;
}

Status KernelTaskCodeBuilder::AppendOrderedArgsByFormat(const TaskSemanticContributeContext &context,
                                                        const ArgsFormatInfo &args_format_holder,
                                                        std::vector<ArgDesc> &dynamic_args_desc,
                                                        std::vector<size_t> &level1_desc_indices) {
  place_holder_var_index_ = 0;
  cust_value_var_index_ = 0;
  uint32_t event_addr_index = 0;
  for (const auto &arg_format : args_format_holder.arg_descs) {
    switch (arg_format.addr_type) {
      case AddrType::INPUT_INSTANCE:
      case AddrType::OUTPUT_INSTANCE:
        GE_ASSERT_SUCCESS(AppendOrderedInputOutputByInstanceIndex(arg_format));
        break;
      case AddrType::INPUT:
      case AddrType::OUTPUT:
        GE_ASSERT_SUCCESS(AppendOrderedInputOutputRange(arg_format, args_format_holder, context));
        break;
      case AddrType::WORKSPACE:
        GE_ASSERT_SUCCESS(AppendOrderedWorkspace(arg_format));
        break;
      case AddrType::PLACEHOLDER:
        AppendOrderedPlaceholder(context);
        break;
      case AddrType::CUSTOM_VALUE:
        AppendOrderedCustomValue(context, *(PtrToPtr<uint8_t, uint64_t>(arg_format.reserved)));
        break;
      case AddrType::INPUT_DESC:
      case AddrType::OUTPUT_DESC:
        AppendOrderedDescArg(context, arg_format, dynamic_args_desc, level1_desc_indices);
        break;
      case AddrType::FFTS_ADDR:
        AppendOrderedFftsAddrArg(context);
        break;
      case AddrType::EVENT_ADDR:
        AppendOrderedEventAddrArg(context, arg_format, event_addr_index);
        break;
      case AddrType::OVERFLOW_ADDR:
        AppendOrderedOverflowAddrArg(context);
        break;
      case AddrType::TILING:
        AppendOrderedTilingArg(context);
        break;
      default:
        REPORT_INNER_ERR_MSG("E19999", "Args Format type %d is currently not supported.",
                             static_cast<int32_t>(arg_format.addr_type));
        GELOGE(FAILED, "[OM2] Args Format type %d is currently not supported.",
               static_cast<int32_t>(arg_format.addr_type));
        return FAILED;
    }
  }
  return SUCCESS;
}

void KernelTaskCodeBuilder::AppendOrderedDescArg(const TaskSemanticContributeContext &context,
                                                 const ArgDesc &arg_format, std::vector<ArgDesc> &dynamic_args_desc,
                                                 std::vector<size_t> &level1_desc_indices) {
  const size_t dynamic_idx = dynamic_args_desc.size();
  dynamic_args_desc.push_back(arg_format);
  level1_desc_indices.push_back(build_data_.semantic.ordered_arg_values.size());
  AddrSemantic level1_desc_ptr;
  level1_desc_ptr.kind = AddrValueKind::kLevel1DescPtr;
  level1_desc_ptr.symbol_hint = "op" + std::to_string(context.op_index) + "_io_desc" + std::to_string(dynamic_idx);
  AppendOrderedArg(level1_desc_ptr);
}

void KernelTaskCodeBuilder::AppendOrderedFftsAddrArg(const TaskSemanticContributeContext &context) {
  AddrSemantic ffts_addr_semantic;
  ffts_addr_semantic.kind = AddrValueKind::kFftsAddr;
  ffts_addr_semantic.symbol_hint = "op" + std::to_string(context.op_index) + "_hardware_sync_addr_";
  AppendOrderedArg(ffts_addr_semantic);
}

void KernelTaskCodeBuilder::AppendOrderedEventAddrArg(const TaskSemanticContributeContext &context,
                                                      const ArgDesc &arg_format, uint32_t &event_addr_index) {
  AddrSemantic event_addr_semantic;
  event_addr_semantic.kind = AddrValueKind::kEventAddr;
  event_addr_semantic.event_id = static_cast<uint32_t>(arg_format.ir_idx);
  event_addr_semantic.symbol_hint =
      "op" + std::to_string(context.op_index) + "_event_desc" + std::to_string(event_addr_index);
  event_addr_index++;
  AppendOrderedArg(event_addr_semantic);
}

void KernelTaskCodeBuilder::AppendOrderedOverflowAddrArg(const TaskSemanticContributeContext &context) {
  if (!AttrUtils::HasAttr(context.op_desc, GLOBALWORKSPACE_TYPE)) {
    return;
  }
  AddrSemantic overflow_addr_semantic;
  overflow_addr_semantic.kind = AddrValueKind::kOverflowAddr;
  overflow_addr_semantic.symbol_hint = "overflow_addr_";
  AppendOrderedArg(overflow_addr_semantic);
}

void KernelTaskCodeBuilder::AppendOrderedTilingArg(const TaskSemanticContributeContext &context) {
  AddrSemantic tiling_semantic;
  tiling_semantic.kind = AddrValueKind::kTiling;
  tiling_semantic.symbol_hint = "op" + std::to_string(context.op_index) + "_tiling";
  tiling_semantic.byte_size = tiling_data_.size();
  AppendOrderedArg(tiling_semantic);
}

Status KernelTaskCodeBuilder::AppendShapeInfoOrderedArgs(const TaskSemanticContributeContext &context,
                                                         const ArgsFormatInfo &args_format_holder,
                                                         const std::vector<ArgDesc> &dynamic_args_desc,
                                                         const std::vector<size_t> &level1_desc_indices) {
  GE_ASSERT(dynamic_args_desc.size() == args_format_holder.shape_infos.size());
  GE_ASSERT(dynamic_args_desc.size() == level1_desc_indices.size());
  for (size_t i = 0UL; i < dynamic_args_desc.size(); ++i) {
    const size_t level1_desc_index = level1_desc_indices[i];
    GE_ASSERT(level1_desc_index < build_data_.semantic.ordered_arg_values.size());
    auto &level1_desc = build_data_.semantic.ordered_arg_values[level1_desc_index];
    GE_ASSERT(level1_desc.kind == AddrValueKind::kLevel1DescPtr);
    level1_desc.level1_target_offset = GetOrderedArgsByteSize(build_data_.semantic.ordered_arg_values);

    AddrSemantic shape_info_buffer;
    shape_info_buffer.kind = AddrValueKind::kShapeInfoBuffer;
    shape_info_buffer.symbol_hint = "op" + std::to_string(context.op_index) + "_shape_info" + std::to_string(i);
    shape_info_buffer.shape_info = args_format_holder.shape_infos[i];
    AppendOrderedArg(shape_info_buffer);

    const auto &dynamic_arg = dynamic_args_desc[i];
    const bool is_input = (dynamic_arg.addr_type == AddrType::INPUT_DESC);
    const auto &ir_2_range = is_input ? args_format_holder.ir_input_2_range : args_format_holder.ir_output_2_range;
    const auto iter = ir_2_range.find(static_cast<size_t>(dynamic_arg.ir_idx));
    GE_ASSERT(iter != ir_2_range.end());
    const auto &range_pair = iter->second;
    size_t begin_idx = range_pair.first;
    while (begin_idx < range_pair.first + range_pair.second) {
      if (is_input) {
        GE_ASSERT_SUCCESS(AppendOrderedInputArg(begin_idx));
      } else {
        GE_ASSERT_SUCCESS(AppendOrderedOutputArg(begin_idx));
      }
      ++begin_idx;
    }
  }
  return SUCCESS;
}

Status KernelTaskCodeBuilder::BuildOrderedArgValuesForAicore(const TaskSemanticContributeContext &context,
                                                             ArgsFormatInfo &args_format_holder) {
  GE_ASSERT_NOTNULL(context.op_desc);
  domi::KernelContext kernel_context;
  uint32_t args_size = 0U;
  uint32_t kernel_type = 0U;
  GE_ASSERT_SUCCESS(GetKernelTaskMeta(context.task_def, kernel_context, args_size, kernel_type));
  if (kernel_context.args_format().empty()) {
    GELOGI("Op %s has empty args format.", context.op_desc->GetNamePtr());
    GE_ASSERT_SUCCESS(CopyTilingDataIfNeeded(context, args_format_holder));
    GE_ASSERT_SUCCESS(BuildOrderedArgValuesWithoutArgsFormat(context));
    return SUCCESS;
  }
  GE_ASSERT_SUCCESS(ArgsFormatDesc::Parse(context.op_desc, kernel_context.args_format(), args_format_holder.arg_descs),
                    "[OM2] Formatted args [%s] parsed failed.", kernel_context.args_format().c_str());

  GE_ASSERT_SUCCESS(ParseArgsFormat(context.op_desc, args_format_holder), "[OM2] ParseArgsFormat failed, op:[%s].",
                    context.op_desc->GetNamePtr());

  const size_t format_args_size = GetArgsSizeByFormat(context.op_desc, args_format_holder);
  args_size = std::max(args_size, static_cast<uint32_t>(format_args_size));
  const size_t extra_args_size =
      GetExtraArgsSize(context.op_desc, static_cast<ccKernelType>(kernel_type), args_format_holder);
  GE_ASSERT_TRUE(!AddOverflow(args_size, static_cast<uint32_t>(extra_args_size), args_size));

  InitArgsTableEntry(context, args_size);
  materialized_output_indices_ = BuildMaterializedOutputIndices(build_data_.semantic);

  GE_ASSERT_SUCCESS(CopyTilingDataIfNeeded(context, args_format_holder));
  current_args_offset_ = 0U;
  std::vector<ArgDesc> dynamic_args_desc;
  std::vector<size_t> level1_desc_indices;
  GE_ASSERT_SUCCESS(AppendOrderedArgsByFormat(context, args_format_holder, dynamic_args_desc, level1_desc_indices));
  GE_ASSERT_SUCCESS(AppendShapeInfoOrderedArgs(context, args_format_holder, dynamic_args_desc, level1_desc_indices));
  GE_ASSERT_SUCCESS(ValidateLevel1DescTargetOffsets());
  return SUCCESS;
}

Status KernelTaskCodeBuilder::BuildOrderedArgValuesWithoutArgsFormat(const TaskSemanticContributeContext &context) {
  uint32_t args_addr_num = 0U;
  uint64_t addr_offset = *context.next_host_args_offset;
  for (const auto &input_addr : build_data_.semantic.input_addrs) {
    GE_ASSERT_SUCCESS(AppendOrderedArgValueForCommon(input_addr, addr_offset));
    addr_offset += kAddressLen;
    args_addr_num++;
  }
  for (const auto &output_addr : build_data_.semantic.output_addrs) {
    if (!IsMaterializedOutput(output_addr)) {
      continue;
    }
    GE_ASSERT_SUCCESS(AppendOrderedArgValueForCommon(output_addr, addr_offset));
    addr_offset += kAddressLen;
    args_addr_num++;
  }
  for (const auto &workspace_addr : build_data_.semantic.workspace_addrs) {
    GE_ASSERT_SUCCESS(AppendOrderedArgValueForCommon(workspace_addr, addr_offset));
    addr_offset += kAddressLen;
    args_addr_num++;
  }
  if (has_tiling_) {
    AddrSemantic tiling_semantic;
    tiling_semantic.kind = AddrValueKind::kTiling;
    tiling_semantic.symbol_hint = "op" + std::to_string(context.op_index) + "_tiling";
    tiling_semantic.byte_size = tiling_data_.size();
    GE_ASSERT_SUCCESS(AppendOrderedArgValueForCommon(tiling_semantic, addr_offset));
    addr_offset += kAddressLen;
    args_addr_num++;
  }
  if (AttrUtils::HasAttr(context.op_desc, GLOBALWORKSPACE_TYPE)) {
    AddrSemantic overflow_addr_semantic;
    overflow_addr_semantic.kind = AddrValueKind::kOverflowAddr;
    overflow_addr_semantic.symbol_hint = "overflow_addr_";
    GE_ASSERT_SUCCESS(AppendOrderedArgValueForCommon(overflow_addr_semantic, addr_offset));
    args_addr_num++;
  }
  (void)build_data_.semantic.args_table_entry.emplace();
  build_data_.semantic.args_table_entry->table_index = *context.next_args_table_index;
  build_data_.semantic.args_table_entry->args_size = static_cast<uint32_t>(args_addr_num * kAddressLen);
  build_data_.semantic.args_table_entry->host_offset = *context.next_host_args_offset;
  args_table_entry_ = &(*build_data_.semantic.args_table_entry);
  return SUCCESS;
}

Status KernelTaskCodeBuilder::BuildOrderedArgValuesForAicpu(const TaskSemanticContributeContext &context) {
  domi::KernelContext kernel_context;
  uint32_t args_size = 0U;
  uint32_t kernel_type = 0U;
  GE_ASSERT_SUCCESS(GetKernelTaskMeta(context.task_def, kernel_context, args_size, kernel_type));
  GE_ASSERT_NOTNULL(context.next_args_table_index);
  GE_ASSERT_NOTNULL(context.next_host_args_offset);
  (void)build_data_.semantic.args_table_entry.emplace();
  build_data_.semantic.args_table_entry->table_index = *context.next_args_table_index;
  build_data_.semantic.args_table_entry->args_size = args_size;
  build_data_.semantic.args_table_entry->host_offset = *context.next_host_args_offset;
  args_table_entry_ = &(*build_data_.semantic.args_table_entry);
  uint64_t addr_offset = *context.next_host_args_offset + kAicpuArgsioAddrOffset;
  for (const auto &input_addr : build_data_.semantic.input_addrs) {
    GE_ASSERT_SUCCESS(AppendOrderedArgValueForCommon(input_addr, addr_offset));
    addr_offset += kAddressLen;
  }
  for (const auto &output_addr : build_data_.semantic.output_addrs) {
    if (!IsMaterializedOutput(output_addr)) {
      continue;
    }
    GE_ASSERT_SUCCESS(AppendOrderedArgValueForCommon(output_addr, addr_offset));
    addr_offset += kAddressLen;
  }
  return SUCCESS;
}

Status KernelTaskCodeBuilder::BuildAicpuArgsSemantic(const TaskSemanticContributeContext &context) {
  domi::KernelContext kernel_context;
  uint32_t args_size = 0U;
  uint32_t kernel_type = 0U;
  GE_ASSERT_SUCCESS(GetKernelTaskMeta(context.task_def, kernel_context, args_size, kernel_type));
  const auto &args = context.task_def.kernel().args();
  (void)build_data_.semantic.aicpu_args.emplace();
  build_data_.semantic.aicpu_args->args_size = args_size;
  build_data_.semantic.aicpu_args->args_buffer.assign(args.begin(), args.end());
  return SUCCESS;
}

Status KernelTaskCodeBuilder::BuildAicpuExtInfoSemantic(const TaskSemanticContributeContext &context) {
  const auto &ext_info = context.task_def.kernel().kernel_ext_info();
  std::vector<uint8_t> ext_info_buffer(ext_info.begin(), ext_info.end());
  int32_t session_info_offset = -1;
  GE_ASSERT_SUCCESS(
      InitAicpuTaskExtInfo(ext_info_buffer.data(), ext_info_buffer.size(), context.op_desc, session_info_offset));
  (void)build_data_.semantic.aicpu_ext_info.emplace();
  build_data_.semantic.aicpu_ext_info->total_len = ext_info_buffer.size();
  build_data_.semantic.aicpu_ext_info->session_info_offset = session_info_offset;
  build_data_.semantic.aicpu_ext_info->serialized_bytes = std::move(ext_info_buffer);
  return SUCCESS;
}

Status KernelTaskCodeBuilder::RenderDispatchAicore(const VarRef &op, const VarRef &ctx,
                                                   std::vector<DeclNode *> &items) {
  std::vector<BodyItem> body;
  auto setup = RenderDispatchSetup(op, ctx);
  body.insert(body.end(), setup.begin(), setup.end());
  body.push_back(RenderDispatchLoop(op, ctx));
  auto distribution = RenderDistribution(op, ctx);
  body.insert(body.end(), distribution.begin(), distribution.end());
  return TaskCodeBuilderUtil::RenderDispatchFunc(ast_, "DispatchKernelAicore", body, items);
}

Status KernelTaskCodeBuilder::RenderDispatchAicpu(const VarRef &op, const VarRef &ctx, std::vector<DeclNode *> &items) {
  std::vector<BodyItem> body;
  auto setup = RenderAicpuDispatchSetup(op, ctx);
  body.insert(body.end(), setup.begin(), setup.end());
  auto launch = RenderAicpuLaunchAndAssemble(op, ctx);
  body.insert(body.end(), launch.begin(), launch.end());
  auto report = RenderAicpuLaunchAndReport(op, ctx);
  body.insert(body.end(), report.begin(), report.end());
  return TaskCodeBuilderUtil::RenderDispatchFunc(ast_, "DispatchKernelAicpu", body, items);
}

std::vector<BodyItem> KernelTaskCodeBuilder::RenderAicpuDispatchSetup(const VarRef &op, const VarRef &ctx) {
  auto aicpu = op.Arrow("dispatch_info").Attr("aicpu");
  auto v_a = ast_.Var("", "a");
  auto v_addr = ast_.Var("", "_addr");
  auto data_t = v_a.Attr("data").Attr("tensor");

  auto resolve_addr = ast_.ReinterpretCast(
      "uint64_t",
      ast_.Call("ResolveOpAddr", {v_a.Attr("addr").Attr("mem_src"), v_a.Attr("addr").Attr("index"),
                                  v_a.Attr("addr").Attr("offset"), ctx.Attr("total_dev_mem_ptr"),
                                  ctx.Attr("session_scope_mem_ptr"), ctx.Attr("constants"), ctx.Attr("var_addrs")}));

  auto build_tensor = ast_.Call("BuildOm2Tensor",
                                {ast_.ReinterpretCast("void *", v_addr), data_t.Attr("size"), data_t.Attr("data_type"),
                                 data_t.Attr("format"), data_t.Attr("shape"), data_t.Attr("shape_dims")});

  std::initializer_list<BodyItem> loop_body = {
      ast_.VarDecl(ast_.Var("const auto &", "a"), aicpu.Attr("args_info")[ast_.Var("", "i")]),
      ast_.VarDecl(ast_.Var("uint64_t", "_addr"), ast_.UInt(0)),
      ast_.If(v_a.Attr("type") == ast_.Var("", "OP_ARG_OPTIONAL_EMPTY"), {ast_.Assign(v_addr, ast_.UInt(0))},
              {ast_.Assign(v_addr, resolve_addr), ast_.Var("", "aicpu_io_tensors").PushBack(build_tensor),
               ast_.VarDecl(
                   ast_.Var("Om2TaskIoEntry", "_entry"),
                   ast_.InitList({ast_.Var("", "aicpu_io_tensors").Attr("back")().Addr(), data_t.Attr("args_offset")})),
               ast_.If(v_a.Attr("type") != ast_.Var("", "OP_ARG_OUTPUT"),
                       {ast_.Var("", "aicpu_report_inputs").PushBack(ast_.Var("", "_entry"))},
                       {ast_.Var("", "aicpu_report_outputs").PushBack(ast_.Var("", "_entry"))})}),
      ast_.Var("", "iow_addr").PushBack(v_addr),
  };

  return {
      ast_.VarDecl("uint32_t", "num_io", aicpu.Attr("args_info_num")),
      ast_.VarDecl("uint32_t", "aicpu_args_idx", aicpu.Attr("args_idx")),
      ast_.VarDecl("const uint8_t *", "args_blob", aicpu.Attr("args_blob")),
      ast_.VarDecl("uint32_t", "args_blob_len", aicpu.Attr("args_blob_len")),
      ast_.VarDecl("const uint8_t *", "ext_info_blob", aicpu.Attr("ext_info_blob")),
      ast_.VarDecl("uint32_t", "ext_info_blob_len", aicpu.Attr("ext_info_blob_len")),
      ast_.VarDecl(ast_.Var("std::vector<uint64_t>", "iow_addr")),
      ast_.VarDecl(ast_.Var("std::vector<Om2Tensor>", "aicpu_io_tensors")),
      ast_.Call("", {ast_.Var("", "aicpu_io_tensors").Attr("reserve")(ast_.Var("", "num_io"))}),
      ast_.VarDecl(ast_.Var("std::vector<Om2TaskIoEntry>", "aicpu_report_inputs")),
      ast_.VarDecl(ast_.Var("std::vector<Om2TaskIoEntry>", "aicpu_report_outputs")),
      ast_.For(ast_.VarDecl("uint32_t", "i", ast_.UInt(0U)), ast_.Var("", "i") < ast_.Var("", "num_io"),
               ast_.PostInc(ast_.Var("", "i")), loop_body),
  };
}

std::vector<BodyItem> KernelTaskCodeBuilder::RenderAicpuLaunchAndAssemble(const VarRef &op, const VarRef &ctx) {
  return {
      ast_.VarDecl(ast_.Var("LaunchKernelCfgHolder", "aicpu_cfg_holder")),
      ast_.VarDecl(
          ast_.Var("LaunchKernelConfig", "aicpu_launch_config"),
          ast_.InitList({
              op.Arrow("dispatch_info").Attr("aicpu").Attr("launch").Attr("schedule_mode"),
              ast_.StaticCast("aclrtEngineType",
                              op.Arrow("dispatch_info").Attr("aicpu").Attr("launch").Attr("engine_type")),
              op.Arrow("dispatch_info").Attr("aicpu").Attr("launch").Attr("block_dim_offset"),
              op.Arrow("dispatch_info").Attr("aicpu").Attr("launch").Attr("is_block_task_prefetch"),
              ast_.Call("GetIsDataDump", {op.Arrow("op_name"), ctx.Attr("model_id"), ctx.Attr("instance_handle")}),
              op.Arrow("dispatch_info").Attr("aicpu").Attr("launch").Attr("time_out"),
              op.Arrow("dispatch_info").Attr("aicpu").Attr("launch").Attr("local_memory_size"),
          })),
      ChkStatus(
          ast_.Call("AssembleLaunchConfig", {ast_.Var("", "aicpu_cfg_holder"), ast_.Var("", "aicpu_launch_config")})),
      ast_.VarDecl(ast_.Var("uint64_t", "local_session_id"), ast_.Deref(ctx.Attr("session_id"))),
      ChkStatus(ast_.Call("AssembleAicpuExtInfo", {ast_.Var("", "ext_info_blob"), ast_.Var("", "ext_info_blob_len"),
                                                   op.Arrow("dispatch_info").Attr("aicpu").Attr("session_info_offset"),
                                                   ast_.Var("", "local_session_id").Addr(), ctx.Attr("kernel_id"),
                                                   ctx.Attr("dev_ext_info_mem_ptrs"),
                                                   op.Arrow("dispatch_info").Attr("aicpu").Attr("aicpu_task_index")})),
      ast_.VarDecl(ast_.Var("std::vector<uint8_t>", "aicpu_args_var")),
      ast_.Var("", "aicpu_args_var").Resize(ast_.Var("", "args_blob_len")),
      ChkStatus(ast_.Call(
          "AssembleAicpuArgs",
          {ast_.Var("", "args_blob"), ast_.Var("", "args_blob_len"),
           ctx.Attr("dev_ext_info_mem_ptrs")[op.Arrow("dispatch_info").Attr("aicpu").Attr("aicpu_task_index")],
           ast_.Var("", "ext_info_blob_len"), ast_.Var("", "iow_addr"), ast_.Var("", "aicpu_args_var").Data()})),
  };
}

std::vector<BodyItem> KernelTaskCodeBuilder::RenderAicpuLaunchAndReport(const VarRef &op, const VarRef &ctx) {
  return {
      ast_.VarDecl(ast_.Var("ArgsInfo *", "aicpu_args_info"),
                   ctx.Attr("args_table").Attr("GetArgsInfo")(ast_.Var("", "aicpu_args_idx"))),
      ast_.VarDecl(ast_.Var("uint64_t", "_launch_begin"), ast_.Call("MsprofSysCycleTime", {})),
      ChkStatus(ast_.Call("AicpuKernelTaskDistribute",
                          {ast_.Var("", "aicpu_args_var"), ast_.Var("", "aicpu_args_info"),
                           ctx.Attr("func_handles")[op.Arrow("dispatch_info").Attr("aicpu").Attr("func_idx")],
                           op.Arrow("dispatch_info").Attr("aicpu").Attr("block_dim"),
                           ctx.Attr("stream_list")[op.Arrow("dispatch_info").Attr("aicpu").Attr("stream_id")],
                           ast_.Var("", "aicpu_cfg_holder").Attr("cfg").Addr()})),
      ChkStatus(ast_.Call(
          "ReportLaunchedOm2Task",
          {op.Arrow("op_name"), op.Arrow("dispatch_info").Attr("aicpu").Attr("op_type"), ast_.UInt(0),
           ast_.ReinterpretCast("uintptr_t", ast_.Var("", "aicpu_args_info").Arrow("dev_addr")),
           ast_.Var("", "aicpu_args_info").Arrow("size"), ast_.Var("", "aicpu_report_inputs").Data(),
           ast_.StaticCast("uint64_t", ast_.Var("", "aicpu_report_inputs").Size()),
           ast_.Var("", "aicpu_report_outputs").Data(),
           ast_.StaticCast("uint32_t", ast_.Var("", "aicpu_report_outputs").Size()), Arg(nullptr), Arg(nullptr),
           ast_.UInt(0U), op.Arrow("dispatch_info").Attr("aicpu").Attr("task_type"),
           op.Arrow("dispatch_info").Attr("aicpu").Attr("block_dim"),
           ctx.Attr("stream_list")[op.Arrow("dispatch_info").Attr("aicpu").Attr("stream_id")], ctx.Attr("model_id"),
           ctx.Attr("instance_handle"), ast_.UInt(0U), ast_.Var("uint64_t", "_launch_begin")})),
  };
}

std::vector<BodyItem> KernelTaskCodeBuilder::RenderDispatchSetup(const VarRef &op, const VarRef &ctx) {
  return {
      ast_.VarDecl(ast_.Var("LaunchKernelCfgHolder", "cfg_holder")),
      ast_.VarDecl(
          ast_.Var("LaunchKernelConfig", "launch_config"),
          ast_.InitList({
              op.Arrow("dispatch_info").Attr("aicore").Attr("launch").Attr("schedule_mode"),
              ast_.StaticCast("aclrtEngineType",
                              op.Arrow("dispatch_info").Attr("aicore").Attr("launch").Attr("engine_type")),
              op.Arrow("dispatch_info").Attr("aicore").Attr("launch").Attr("block_dim_offset"),
              op.Arrow("dispatch_info").Attr("aicore").Attr("launch").Attr("is_block_task_prefetch"),
              ast_.Call("GetIsDataDump", {op.Arrow("op_name"), ctx.Attr("model_id"), ctx.Attr("instance_handle")}),
              op.Arrow("dispatch_info").Attr("aicore").Attr("launch").Attr("time_out"),
              op.Arrow("dispatch_info").Attr("aicore").Attr("launch").Attr("local_memory_size"),
          })),
      ChkStatus(ast_.Call("AssembleLaunchConfig", {ast_.Var("", "cfg_holder"), ast_.Var("", "launch_config")})),
      ast_.VarDecl(
          ast_.Var("ArgsInfo *", "args_info"),
          ctx.Attr("args_table").Attr("GetArgsInfo")(op.Arrow("dispatch_info").Attr("aicore").Attr("args_idx"))),
      ChkNotNull(ast_.Var("", "args_info")),
      // -- 声明 ordered_io_addrs 和 Report IO 向量 --
      ast_.VarDecl(ast_.Var("std::vector<uint64_t>", "ordered_io_addrs")),
      ast_.VarDecl(ast_.Var("std::vector<Om2Tensor>", "io_tensors")),
      ast_.Call(
          "",
          {ast_.Var("", "io_tensors").Attr("reserve")(op.Arrow("dispatch_info").Attr("aicore").Attr("args_info_num"))}),
      ast_.VarDecl(ast_.Var("std::vector<Om2TaskIoEntry>", "report_inputs")),
      ast_.VarDecl(ast_.Var("std::vector<Om2TaskIoEntry>", "report_outputs")),
      ast_.VarDecl(ast_.Var("std::vector<uint64_t>", "report_workspace_addrs")),
      ast_.VarDecl(ast_.Var("std::vector<uint64_t>", "report_workspace_sizes")),
  };
}

BodyItem KernelTaskCodeBuilder::RenderDispatchLoop(const VarRef &op, const VarRef &ctx) {
  auto a = ast_.Var("const auto &", "a");
  return ast_.For(ast_.VarDecl("uint32_t", "j", ast_.UInt(0)),
                  ast_.Var("", "j") < op.Arrow("dispatch_info").Attr("aicore").Attr("args_info_num"),
                  ast_.PostInc(ast_.Var("", "j")),
                  std::initializer_list<BodyItem>{
                      ast_.VarDecl(a, op.Arrow("dispatch_info").Attr("aicore").Attr("args_info")[ast_.Var("", "j")]),
                      ast_.VarDecl(ast_.Var("uint64_t", "_addr"), ast_.UInt(0)),
                      ast_.Switch(ast_.Var("", "a").Attr("type"),
                                  std::vector<BodyItem>{
                                      // INPUT / OUTPUT / CONST_TENSOR → 共享 handler（内部根据 a.type 区分）
                                      ast_.Case(ast_.Var("", "OP_ARG_INPUT")),
                                      ast_.Case(ast_.Var("", "OP_ARG_OUTPUT")),
                                      ast_.Case(ast_.Var("", "OP_ARG_CONST_TENSOR")),
                                      ast_.Case(ast_.Var("", "OP_ARG_VAR_TENSOR")),
                                      ast_.Block(HandleInputOutputArg(a, ctx)),
                                      // WORKSPACE
                                      ast_.Case(ast_.Var("", "OP_ARG_WORKSPACE")),
                                      ast_.Block(HandleWorkspaceArg(a, ctx)),
                                      // LEVEL1_DESC
                                      ast_.Case(ast_.Var("", "OP_ARG_LEVEL1_DESC")),
                                      ast_.Block(HandleLevel1DescArg(a, ctx)),
                                      // SHAPE_INFO / CUSTOM_VALUE → 共享 handler
                                      ast_.Case(ast_.Var("", "OP_ARG_SHAPE_INFO")),
                                      ast_.Case(ast_.Var("", "OP_ARG_CUSTOM_VALUE")),
                                      ast_.Block(HandleShapeInfoOrCustomValueArg(a)),
                                      // PLACEHOLDER / OPTIONAL_EMPTY → 共享 handler
                                      ast_.Case(ast_.Var("", "OP_ARG_PLACEHOLDER")),
                                      ast_.Case(ast_.Var("", "OP_ARG_OPTIONAL_EMPTY")),
                                      ast_.Block(HandlePlaceholderOrOptionalEmptyArg()),
                                      // FFTS_ADDR
                                      ast_.Case(ast_.Var("", "OP_ARG_FFTS_ADDR")),
                                      ast_.Block(HandleFftsAddrArg()),
                                      // EVENT_ADDR
                                      ast_.Case(ast_.Var("", "OP_ARG_EVENT_ADDR")),
                                      ast_.Block(HandleEventAddrArg(a, ctx)),
                                      // OVERFLOW_ADDR
                                      ast_.Case(ast_.Var("", "OP_ARG_OVERFLOW_ADDR")),
                                      ast_.Block(HandleOverflowAddrArg(ctx)),
                                      // TILING
                                      ast_.Case(ast_.Var("", "OP_ARG_TILING")),
                                      ast_.Block(HandleTilingArg(a, ctx)),
                                      // default
                                      ast_.Case(Arg(nullptr)),
                                      ast_.Block(HandleDefaultArg()),
                                  }),
                      ast_.Var("", "ordered_io_addrs").PushBack(ast_.Var("", "_addr")),
                  });
}

std::vector<BodyItem> KernelTaskCodeBuilder::RenderDistribution(const VarRef &op, const VarRef &ctx) {
  auto aicore = op.Arrow("dispatch_info").Attr("aicore");
  auto slot_args = aicore.Attr("slot_args");
  auto task_type = aicore.Attr("task_type");
  auto stream = ctx.Attr("stream_list")[aicore.Attr("stream_id")];

  return {
      ast_.VarDecl(
          ast_.Var("Om2L0TaskRawInfo", "l0_info"),
          ast_.InitList({ast_.UInt(1U), slot_args.Attr("need_assert_or_printf"),
                         ast_.StaticCast("uint64_t", slot_args.Attr("slots_num")), slot_args.Attr("slot_info")})),
      ChkStatus(ast_.Call("ReportOm2TaskPreprocess",
                          {op.Arrow("op_name"), aicore.Attr("op_type"),
                           ast_.UInt(0),  // op_desc_id
                           ast_.ReinterpretCast("uintptr_t", ast_.Var("", "args_info").Arrow("dev_addr")),
                           ast_.Var("", "args_info").Arrow("size"), ast_.Var("", "report_inputs"),
                           ast_.Var("", "report_outputs"), ast_.Var("", "report_workspace_addrs"),
                           ast_.Var("", "report_workspace_sizes"), task_type, aicore.Attr("block_dim"), stream,
                           ast_.Var("", "l0_info").Addr(), ctx.Attr("model_id"), ctx.Attr("instance_handle")})),
      ast_.VarDecl(ast_.Var("uint64_t", "_launch_begin"), ast_.Call("MsprofSysCycleTime", {})),
      ChkStatus(ast_.Call("KernelTaskDistribute",
                          {ast_.Var("", "ordered_io_addrs"), ast_.Var("", "args_info"),
                           ctx.Attr("func_handles")[aicore.Attr("func_idx")], aicore.Attr("block_dim"), stream,
                           ast_.Var("", "cfg_holder").Attr("cfg").Addr()})),
      ChkStatus(ast_.Call("ReportLaunchedOm2Task",
                          {op.Arrow("op_name"),
                           aicore.Attr("op_type"),
                           ast_.UInt(0),  // op_desc_id
                           ast_.ReinterpretCast("uintptr_t", ast_.Var("", "args_info").Arrow("dev_addr")),
                           ast_.Var("", "args_info").Arrow("size"),
                           ast_.Var("", "report_inputs").Data(),
                           ast_.StaticCast("uint64_t", ast_.Var("", "report_inputs").Size()),
                           ast_.Var("", "report_outputs").Data(),
                           ast_.StaticCast("uint32_t", ast_.Var("", "report_outputs").Size()),
                           ast_.Var("", "report_workspace_addrs").Data(),
                           ast_.Var("", "report_workspace_sizes").Data(),
                           ast_.StaticCast("uint32_t", ast_.Var("", "report_workspace_sizes").Size()),
                           task_type,
                           aicore.Attr("block_dim"),
                           stream,
                           ctx.Attr("model_id"),
                           ctx.Attr("instance_handle"),
                           ast_.UInt(0U),
                           ast_.Var("uint64_t", "_launch_begin"),
                           aicore.Attr("fusion_op").Attr("original_op_names"),
                           aicore.Attr("fusion_op").Attr("input_mem_size"),
                           aicore.Attr("fusion_op").Attr("output_mem_size"),
                           aicore.Attr("fusion_op").Attr("workspace_mem_size"),
                           aicore.Attr("fusion_op").Attr("weight_mem_size")})),
  };
}

std::vector<BodyItem> KernelTaskCodeBuilder::HandleInputOutputArg(const VarRef &a, const VarRef &ctx) {
  return {
      ast_.Assign(
          ast_.Var("", "_addr"),
          ast_.ReinterpretCast("uint64_t",
                               ast_.Call("ResolveOpAddr", {a.Attr("addr").Attr("mem_src"), a.Attr("addr").Attr("index"),
                                                           a.Attr("addr").Attr("offset"), ctx.Attr("total_dev_mem_ptr"),
                                                           ctx.Attr("session_scope_mem_ptr"), ctx.Attr("constants"),
                                                           ctx.Attr("var_addrs")}))),
      ast_.Var("", "io_tensors")
          .PushBack(ast_.Call(
              "BuildOm2Tensor",
              {ast_.ReinterpretCast("void *", ast_.Var("", "_addr")), a.Attr("data").Attr("tensor").Attr("size"),
               a.Attr("data").Attr("tensor").Attr("data_type"), a.Attr("data").Attr("tensor").Attr("format"),
               a.Attr("data").Attr("tensor").Attr("shape"), a.Attr("data").Attr("tensor").Attr("shape_dims")})),
      ast_.VarDecl(ast_.Var("Om2TaskIoEntry", "_entry"),
                   ast_.InitList({ast_.Var("", "io_tensors").Attr("back")().Addr(),
                                  a.Attr("data").Attr("tensor").Attr("args_offset")})),
      ast_.If(a.Attr("type") == ast_.Var("", "OP_ARG_INPUT") || a.Attr("type") == ast_.Var("", "OP_ARG_CONST_TENSOR"),
              {ast_.Var("", "report_inputs").PushBack(ast_.Var("", "_entry"))},
              {ast_.Var("", "report_outputs").PushBack(ast_.Var("", "_entry"))}),
      ast_.Break(),
  };
}

std::vector<BodyItem> KernelTaskCodeBuilder::HandleWorkspaceArg(const VarRef &a, const VarRef &ctx) {
  return {
      ast_.Assign(
          ast_.Var("", "_addr"),
          ast_.ReinterpretCast("uint64_t",
                               ast_.Call("ResolveOpAddr", {a.Attr("addr").Attr("mem_src"), a.Attr("addr").Attr("index"),
                                                           a.Attr("addr").Attr("offset"), ctx.Attr("total_dev_mem_ptr"),
                                                           ctx.Attr("session_scope_mem_ptr"), ctx.Attr("constants"),
                                                           ctx.Attr("var_addrs")}))),
      ast_.Var("", "report_workspace_addrs").PushBack(ast_.Var("", "_addr")),
      ast_.Var("", "report_workspace_sizes").PushBack(a.Attr("data").Attr("tensor").Attr("size")),
      ast_.Break(),
  };
}
std::vector<BodyItem> KernelTaskCodeBuilder::HandleLevel1DescArg(const VarRef &a, const VarRef &ctx) {
  return {
      ast_.VarDecl(ast_.Var("void *", "_desc"),
                   ctx.Attr("args_table").Attr("GetDevArgAddr")(a.Attr("data").Attr("custom_value"), 0)),
      ChkNotNull(ast_.Var("", "_desc")),
      ast_.Assign(ast_.Var("", "_addr"), ast_.ReinterpretCast("uint64_t", ast_.Var("", "_desc"))),
      ast_.Break(),
  };
}

std::vector<BodyItem> KernelTaskCodeBuilder::HandleShapeInfoOrCustomValueArg(const VarRef &a) {
  return {
      ast_.Assign(ast_.Var("", "_addr"), a.Attr("data").Attr("custom_value")),
      ast_.Break(),
  };
}

std::vector<BodyItem> KernelTaskCodeBuilder::HandlePlaceholderOrOptionalEmptyArg() {
  return {
      ast_.Assign(ast_.Var("", "_addr"), ast_.UInt(0)),
      ast_.Break(),
  };
}

std::vector<BodyItem> KernelTaskCodeBuilder::HandleFftsAddrArg() {
  return {
      ast_.VarDecl(ast_.Var("void *", "_ffts"), Arg(nullptr)),
      ChkStatus(ast_.Call("aclrtGetHardwareSyncAddr", {ast_.Var("", "_ffts").Addr()})),
      ast_.Assign(ast_.Var("", "_addr"), ast_.ReinterpretCast("uint64_t", ast_.Var("", "_ffts"))),
      ast_.Break(),
  };
}

std::vector<BodyItem> KernelTaskCodeBuilder::HandleEventAddrArg(const VarRef &a, const VarRef &ctx) {
  return {
      ast_.VarDecl(ast_.Var("void *", "_event"), Arg(nullptr)),
      ChkStatus(ast_.Call("GetEventIdAddr", {ast_.Var("", "_event"), ctx.Attr("event_id_mem_map"),
                                             ast_.StaticCast("uint32_t", a.Attr("data").Attr("custom_value")),
                                             ctx.Attr("dev_dynamic_mem_ptrs")})),
      ast_.Assign(ast_.Var("", "_addr"), ast_.ReinterpretCast("uint64_t", ast_.Var("", "_event"))),
      ast_.Break(),
  };
}

std::vector<BodyItem> KernelTaskCodeBuilder::HandleOverflowAddrArg(const VarRef &ctx) {
  return {
      ast_.Assign(ast_.Var("", "_addr"), ast_.ReinterpretCast("uint64_t", ctx.Attr("overflow_addr"))),
      ast_.Break(),
  };
}

std::vector<BodyItem> KernelTaskCodeBuilder::HandleTilingArg(const VarRef &a, const VarRef &ctx) {
  return {
      ast_.VarDecl(ast_.Var("void *", "_tiling"), Arg(nullptr)),
      ChkStatus(
          ast_.Call("MallocDeviceMemory", {ast_.Var("", "_tiling"), a.Attr("data").Attr("tiling").Attr("raw_data_len"),
                                           ast_.UInt(2), ctx.Attr("dev_dynamic_mem_ptrs")})),
      ChkStatus(ast_.Call("aclrtMemcpy", {ast_.Var("", "_tiling"), a.Attr("data").Attr("tiling").Attr("raw_data_len"),
                                          a.Attr("data").Attr("tiling").Attr("raw_data"),
                                          a.Attr("data").Attr("tiling").Attr("raw_data_len"),
                                          ast_.Var("", "ACL_MEMCPY_HOST_TO_DEVICE")})),
      ast_.Assign(ast_.Var("", "_addr"), ast_.ReinterpretCast("uint64_t", ast_.Var("", "_tiling"))),
      ast_.Break(),
  };
}

std::vector<BodyItem> KernelTaskCodeBuilder::HandleDefaultArg() {
  return {
      ast_.Assign(ast_.Var("", "_addr"), ast_.UInt(0)),
      ast_.Break(),
  };
}

Status KernelTaskCodeBuilder::RenderOpDefTableFields(std::vector<std::pair<std::string, Arg>> &fields) {
  fields.push_back({"dispatch_type", ast_.StaticCast("OpDispatchType", static_cast<int64_t>(dispatch_type_))});
  fields.push_back({"op_name", Arg::StringLiteral(header_.op_name)});
  if (std::holds_alternative<AicoreTaskData>(build_data_.dispatch_info)) {
    const auto &data = std::get<AicoreTaskData>(build_data_.dispatch_info);
    GELOGI("[OM2] BuildOpDefTable: op=%s, func_idx=%u", header_.op_name.c_str(),
           build_data_.semantic.launch.func_handle_index);
    fields.emplace_back("dispatch_info", RenderAicoreOpDefFields(data));
  } else {
    const auto &data = std::get<AicpuTaskData>(build_data_.dispatch_info);
    GELOGI("[OM2] BuildOpDefTable: op=%s (AICPU), func_idx=%u", header_.op_name.c_str(),
           build_data_.semantic.launch.func_handle_index);
    fields.emplace_back("dispatch_info", RenderAicpuOpDefFields(data));
  }
  return SUCCESS;
}

Arg KernelTaskCodeBuilder::RenderAicoreOpDefFields(const AicoreTaskData &data) {
  auto launch_values = std::vector<Arg>{
      build_data_.semantic.launch.config.schedule_mode,    static_cast<int64_t>(data.engine_type),
      build_data_.semantic.launch.config.block_dim_offset, build_data_.semantic.launch.config.is_block_task_prefetch,
      build_data_.semantic.launch.config.time_out,         build_data_.semantic.launch.config.local_memory_size,
  };
  auto l0_values = std::vector<Arg>{
      static_cast<int64_t>(data.need_assert_or_printf),
      static_cast<int64_t>(build_data_.semantic.ordered_arg_values.size()),
      ast_.CCast("const Om2L0ArgSlotInfo[]",
                 TaskCodeBuilderUtil::BuildL0ArgSlotEntries(ast_, build_data_.semantic.ordered_arg_values)),
  };
  // 融合算子字段：从 semantic 直接读取，编译期嵌入 kOpDefs[]
  const auto &orig_names = build_data_.semantic.original_op_names;
  Arg orig_names_arg = Arg(nullptr);
  if (!orig_names.empty()) {
    std::string joined;
    for (size_t i = 0U; i < orig_names.size(); ++i) {
      if (i != 0U) joined += ';';
      joined += orig_names[i];
    }
    orig_names_arg = Arg::StringLiteral(joined);
  }
  auto aicore_fields = std::vector<std::pair<std::string, Arg>>{
      {"args_info", TaskCodeBuilderUtil::RenderOpArgDesc(ast_, build_data_.ordered_args)},
      {"args_info_num", static_cast<int64_t>(build_data_.ordered_args.size())},
      {"op_type", Arg::StringLiteral(header_.op_type)},
      {"args_idx", static_cast<int64_t>(build_data_.semantic.args_table_entry->table_index)},
      {"block_dim", build_data_.semantic.launch.block_dim},
      {"func_idx", static_cast<int64_t>(build_data_.semantic.launch.func_handle_index)},
      {"stream_id", static_cast<uint32_t>(header_.stream_id)},
      {"task_type", static_cast<int64_t>(build_data_.semantic.task_type)},
      {"launch", ast_.InitList(launch_values)},
      {"slot_args", ast_.InitList(l0_values)},
      {"fusion_op", ast_.InitList({
                        orig_names_arg,
                        ast_.ULong(build_data_.semantic.input_mem_size),
                        ast_.ULong(build_data_.semantic.output_mem_size),
                        ast_.ULong(build_data_.semantic.workspace_mem_size),
                        ast_.ULong(build_data_.semantic.weight_mem_size),
                    })},
  };
  return ast_.DesignatedInit({{"aicore", ast_.DesignatedInit(aicore_fields)}});
}

Arg KernelTaskCodeBuilder::RenderAicpuOpDefFields(const AicpuTaskData &data) {
  auto launch_values = std::vector<Arg>{
      build_data_.semantic.launch.config.schedule_mode,    static_cast<int64_t>(data.engine_type),
      build_data_.semantic.launch.config.block_dim_offset, build_data_.semantic.launch.config.is_block_task_prefetch,
      build_data_.semantic.launch.config.time_out,         build_data_.semantic.launch.config.local_memory_size,
  };
  auto aicpu_fields = std::vector<std::pair<std::string, Arg>>{
      {"args_info", TaskCodeBuilderUtil::RenderOpArgDesc(ast_, build_data_.ordered_args)},
      {"args_info_num", static_cast<int64_t>(build_data_.ordered_args.size())},
      {"op_type", Arg::StringLiteral(header_.op_type)},
      {"args_idx", static_cast<int64_t>(build_data_.semantic.args_table_entry->table_index)},
      {"func_idx", static_cast<int64_t>(build_data_.semantic.launch.func_handle_index)},
      {"block_dim", build_data_.semantic.launch.block_dim},
      {"stream_id", static_cast<uint32_t>(header_.stream_id)},
      {"args_blob", !build_data_.semantic.aicpu_args.has_value() || build_data_.semantic.aicpu_args->args_buffer.empty()
                        ? Arg(nullptr)
                        : ast_.ReinterpretCast("const uint8_t *", Arg::StringLiteral(SerializeBytesToOctalString(
                                                                      build_data_.semantic.aicpu_args->args_buffer)))},
      {"args_blob_len", static_cast<int64_t>(build_data_.semantic.aicpu_args.has_value()
                                                 ? build_data_.semantic.aicpu_args->args_buffer.size()
                                                 : 0U)},
      {"ext_info_blob",
       !build_data_.semantic.aicpu_ext_info.has_value() || build_data_.semantic.aicpu_ext_info->serialized_bytes.empty()
           ? Arg(nullptr)
           : ast_.ReinterpretCast("const uint8_t *", Arg::StringLiteral(SerializeBytesToOctalString(
                                                         build_data_.semantic.aicpu_ext_info->serialized_bytes)))},
      {"ext_info_blob_len", static_cast<int64_t>(build_data_.semantic.aicpu_ext_info.has_value()
                                                     ? build_data_.semantic.aicpu_ext_info->serialized_bytes.size()
                                                     : 0U)},
      {"launch", ast_.InitList(launch_values)},
      {"session_info_offset", static_cast<int64_t>(build_data_.semantic.aicpu_ext_info.has_value()
                                                       ? build_data_.semantic.aicpu_ext_info->session_info_offset
                                                       : -1)},
      {"aicpu_task_index", static_cast<uint32_t>(build_data_.semantic.aicpu_task_index)},
      {"task_type", static_cast<int64_t>(build_data_.semantic.task_type)},
  };
  return ast_.DesignatedInit({{"aicpu", ast_.DesignatedInit(aicpu_fields)}});
}

Status KernelTaskCodeBuilder::ParseAicpuExtInfoHandler(const OpDescPtr &op_desc, const string &ext_info,
                                                       std::unique_ptr<om2::Om2AicpuExtInfoHandler> &ex_handle) const {
  if (ext_info.empty()) {
    return SUCCESS;
  }
  int32_t unknown_shape_type_val = 0;
  (void)AttrUtils::GetInt(op_desc, ATTR_NAME_UNKNOWN_SHAPE_TYPE, unknown_shape_type_val);
  const auto unknown_type = static_cast<UnknowShapeOpType>(unknown_shape_type_val);
  const uint32_t num_inputs =
      static_cast<uint32_t>(is_optional_input_placeholder_ ? op_desc->GetAllInputsSize() : op_desc->GetInputsSize());
  const uint32_t num_outputs = static_cast<uint32_t>(op_desc->GetOutputsSize());

  ex_handle = MakeUnique<om2::Om2AicpuExtInfoHandler>(op_desc->GetName(), num_inputs, num_outputs, unknown_type);
  GE_CHECK_NOTNULL(ex_handle);
  GE_CHK_STATUS_RET(ex_handle->Parse(ext_info), "[OM2][Parse][KernelExtInfo] failed, kernel_ext_info_size=%zu, op:%s.",
                    ext_info.size(), op_desc_->GetName().c_str());
  return SUCCESS;
};

Status KernelTaskCodeBuilder::UpdateArgsSizeWithCustomized(const OpDescPtr &op_desc) {
  GE_ASSERT_NOTNULL(op_desc);
  args_size_ = static_cast<uint32_t>(MemSizeAlign(static_cast<size_t>(args_size_)));
  customized_args_info_.customized_aligned = true;

  GE_ASSERT_TRUE(!ge::MulOverflow(om2::ModelUtils::GetInputDescs(op_desc).size(), kAddressLen,
                                  customized_args_info_.input_addr_size));
  customized_args_info_.input_addr_offset = args_size_;
  GE_ASSERT_TRUE(!AddOverflow(args_size_, customized_args_info_.input_addr_size, args_size_));

  GE_ASSERT_TRUE(!ge::MulOverflow(om2::ModelUtils::GetOutputDescs(op_desc).size(), kAddressLen,
                                  customized_args_info_.output_addr_size));
  customized_args_info_.output_addr_offset = args_size_;
  GE_ASSERT_TRUE(!AddOverflow(args_size_, customized_args_info_.input_addr_size, args_size_));

  std::stringstream ss;
  ss << "customized_args_info: args/after_align:" << customized_args_info_.kernel_def_args_size << "/ " << args_size_
     << ", is aligned: " << customized_args_info_.customized_aligned << ", input_addr_size/offset is "
     << customized_args_info_.input_addr_size << " / " << customized_args_info_.input_addr_offset
     << ", output_addr_size/offset is " << customized_args_info_.output_addr_size << " / "
     << customized_args_info_.output_addr_offset;
  GELOGD("[OM2]%s ", ss.str().c_str());
  return SUCCESS;
}

Status KernelTaskCodeBuilder::ParseTaskRunParam(const domi::TaskDef &task_def, const om2::RuntimeParam &rts_param,
                                                OpDescPtr op_desc, om2::TaskRunParam &task_run_param) {
  task_type_ = static_cast<ModelTaskType>(task_def.type());
  GE_CHECK_NOTNULL(&rts_param);
  domi::KernelContext context;
  size_t extra_name_size = 0U;
  if (Om2CodegenUtils::IsAllKernel(task_type_)) {
    const domi::KernelDefWithHandle &kernel_def = task_def.kernel_with_handle();
    args_size_ = static_cast<uint32_t>(kernel_def.args().size());
    context = kernel_def.context();
    kernel_type_ = static_cast<ccKernelType>(context.kernel_type());
  } else {
    const domi::KernelDef &kernel_def = task_def.kernel();
    args_size_ = static_cast<uint32_t>(kernel_def.args().size());
    context = kernel_def.context();
    kernel_type_ = static_cast<ccKernelType>(context.kernel_type());
    if (kernel_type_ == ccKernelType::AI_CPU_KFC) {
      GELOGE(FAILED, "[OM2] Unsupported ai cpu kfc");
      return FAILED;
    }
  }
  GE_CHECK_NOTNULL(op_desc);
  op_desc_ = op_desc;
  super_kernel_op_desc_ = (op_desc_->GetType() == "SuperKernel") ? op_desc_ : nullptr;
  if (super_kernel_op_desc_ != nullptr) {
    GE_ASSERT_TRUE(!context.args_format().empty());
    GELOGE(FAILED, "[OM2] Unsupported SuperKernel");
    return FAILED;
  }
  (void)AttrUtils::GetBool(op_desc_, kOptionalInputPlaceholder, is_optional_input_placeholder_);
  if (is_optional_input_placeholder_) {
    GELOGE(FAILED, "[OM2] Unsupported optional_input_placeholder");
    return FAILED;
  }
  if (!context.args_format().empty()) {
    GE_ASSERT_SUCCESS(ArgsFormatDesc::Parse(op_desc_, context.args_format(), args_format_holder_.arg_descs),
                      "[OM2]Formatted args [%s] parsed failed.", context.args_format().c_str());
    GE_ASSERT_SUCCESS(ParseArgsFormat(op_desc_, args_format_holder_), "[OM2]ParseArgsFormat failed, op:[%s].",
                      op_desc_->GetNamePtr());
    const size_t format_args_size = GetArgsSizeByFormat(op_desc_, args_format_holder_) + extra_name_size;
    args_size_ = std::max(args_size_, static_cast<uint32_t>(format_args_size));
    if (task_type_ == ModelTaskType::MODEL_TASK_PREPROCESS_KERNEL && kernel_type_ == ccKernelType::CUST_AI_CPU) {
      GELOGE(FAILED, "[OM2] Unsupported preprocess kernel");
      return FAILED;
    }
    GELOGI("[OM2]OP [%s] has formatted args_format:[%s], args size by format is [%" PRIu64 "], final size is [%u]",
           op_desc_->GetNamePtr(), context.args_format().c_str(), format_args_size, args_size_);
  }

  const size_t extra_args_size = GetExtraArgsSize(op_desc_, kernel_type_, args_format_holder_);
  GELOGD("[OM2]Op:[%s] args size from_task:[%u], extra_size:[%zu]", op_desc_->GetNamePtr(), args_size_,
         extra_args_size);
  GE_ASSERT_TRUE(!AddOverflow(args_size_, static_cast<uint32_t>(extra_args_size), args_size_));

  input_data_addrs_ =
      om2::ModelUtils::GetInputAddrsValue(rts_param, op_desc_, input_mem_types_, is_optional_input_placeholder_);
  if (!context.args_format().empty()) {
    output_data_addrs_ = om2::ModelUtils::GetOutputAddrsValue(rts_param, op_desc_, output_mem_types_, true);
  } else {
    output_data_addrs_ = om2::ModelUtils::GetOutputAddrsValue(rts_param, op_desc_, output_mem_types_);
  }
  workspace_addrs_ = om2::ModelUtils::GetWorkspaceDataAddrsValue(rts_param, op_desc_, workspace_mem_types_);
  for (size_t i = 0UL; i < input_data_addrs_.size(); i++) {
    task_run_param.parsed_input_addrs.push_back({input_data_addrs_[i], input_mem_types_[i], true, {0}});
  }
  for (size_t i = 0UL; i < output_data_addrs_.size(); i++) {
    task_run_param.parsed_output_addrs.push_back({output_data_addrs_[i], output_mem_types_[i], true, {0}});
  }
  for (size_t i = 0UL; i < workspace_addrs_.size(); i++) {
    task_run_param.parsed_workspace_addrs.push_back({workspace_addrs_[i], workspace_mem_types_[i], true, {0}});
  }

  size_t append_size = 0U;
  if ((kernel_type_ == ccKernelType::AI_CPU) || (kernel_type_ == ccKernelType::CUST_AI_CPU)) {
    std::unique_ptr<om2::Om2AicpuExtInfoHandler> ex_handle = nullptr;
    const auto &kernel_def = task_def.kernel();
    const auto &ext_info = kernel_def.kernel_ext_info();
    GE_ASSERT_SUCCESS(ParseAicpuExtInfoHandler(op_desc_, ext_info, ex_handle));
    if ((ex_handle != nullptr) && (ex_handle->GetDeployTypeFlag() == static_cast<int32_t>(RT_KERNEL_HOST_ONLY))) {
      args_placement_ = om2::ArgsPlacement::kArgsPlacementHostSvm;
    }
    append_size = sizeof(uintptr_t);  // 多申请8字节，用来做aicpuhead结构体的对齐
  } else if (kernel_type_ == ccKernelType::CUSTOMIZED) {
    customized_args_info_.kernel_def_args_size = args_size_;
    GE_ASSERT_SUCCESS(UpdateArgsSizeWithCustomized(op_desc_));
  }
  task_run_param.args_descs.push_back(
      {static_cast<int64_t>(MemSizeAlign(static_cast<size_t>(args_size_), static_cast<uint32_t>(sizeof(uintptr_t))) +
                            append_size),
       args_placement_});
  return SUCCESS;
}

void KernelTaskCodeBuilder::UpdateIoAndWorkspaceAddrs(const om2::IowAddrs &iow_addrs) {
  for (size_t i = 0UL; i < input_data_addrs_.size(); i++) {
    input_data_addrs_[i] =
        (iow_addrs.input_logic_addrs.empty()) ? input_data_addrs_[i] : iow_addrs.input_logic_addrs[i].logic_addr;
    input_mem_types_[i] =
        (iow_addrs.input_logic_addrs.empty()) ? input_mem_types_[i] : iow_addrs.input_logic_addrs[i].memory_type;
  }

  for (size_t i = 0UL; i < output_data_addrs_.size(); i++) {
    output_data_addrs_[i] =
        (iow_addrs.output_logic_addrs.empty()) ? output_data_addrs_[i] : iow_addrs.output_logic_addrs[i].logic_addr;
    output_mem_types_[i] =
        (iow_addrs.output_logic_addrs.empty()) ? output_mem_types_[i] : iow_addrs.output_logic_addrs[i].memory_type;
  }

  for (size_t i = 0UL; i < workspace_addrs_.size(); i++) {
    workspace_addrs_[i] =
        (iow_addrs.workspace_logic_addrs.empty()) ? workspace_addrs_[i] : iow_addrs.workspace_logic_addrs[i].logic_addr;
    workspace_mem_types_[i] = (iow_addrs.workspace_logic_addrs.empty())
                                  ? workspace_mem_types_[i]
                                  : iow_addrs.workspace_logic_addrs[i].memory_type;
  }
}

void KernelTaskCodeBuilder::AppendIoAddr(const uint64_t addr, const uint64_t addr_type) {
  io_addrs_.push_back(addr);
  io_addr_mem_types_.push_back(addr_type);
}

Status KernelTaskCodeBuilder::AppendInputOutputAddrByInstanceIndex(size_t ins_idx, bool is_input) {
  if (is_input) {
    GE_ASSERT_TRUE(ins_idx < input_data_addrs_.size(), "[OM2]Instance idx [%zu] is invalid, input_size:[%zu]", ins_idx,
                   input_data_addrs_.size());
    cust_to_relevant_offset_[ins_idx] = io_addrs_.size();
    AppendIoAddr(input_data_addrs_[ins_idx], input_mem_types_[ins_idx]);
  } else {
    GE_ASSERT_TRUE(ins_idx < output_data_addrs_.size(), "[OM2]Instance idx [%zu] is invalid, output_size:[%zu]",
                   ins_idx, output_data_addrs_.size());
    cust_to_relevant_offset_[input_data_addrs_.size() + ins_idx] = io_addrs_.size();
    AppendIoAddr(output_data_addrs_[ins_idx], output_mem_types_[ins_idx]);
  }
  return SUCCESS;
}

Status KernelTaskCodeBuilder::AppendInputOutputAddr(size_t ir_idx, bool is_input) {
  const std::map<size_t, std::pair<size_t, size_t>> &ir_2_range =
      is_input ? args_format_holder_.ir_input_2_range : args_format_holder_.ir_output_2_range;
  const auto iter = ir_2_range.find(ir_idx);
  GE_ASSERT(iter != ir_2_range.end(), "[OM2]Ir idx [%zu] is not found, input flag %u.", ir_idx, is_input);
  const auto &range_pair = iter->second;
  if (is_input && range_pair.second == 0UL) {
    AppendIoAddr(0UL, om2::IowMemoryType::kAbsoluteMemType);
    return SUCCESS;
  }
  size_t begin_idx = range_pair.first;
  std::vector<uint64_t> &addrs = is_input ? input_data_addrs_ : output_data_addrs_;
  std::vector<uint64_t> &types = is_input ? input_mem_types_ : output_mem_types_;
  const size_t cust_offset = is_input ? 0U : input_data_addrs_.size();
  for (size_t i = 0UL; i < range_pair.second; ++i, ++begin_idx) {
    GE_ASSERT(begin_idx < addrs.size(), "[OM2]ir_idx:[%zu], begin_index [%zu] is out of range, max_size:[%zu].", ir_idx,
              begin_idx, addrs.size());
    cust_to_relevant_offset_[begin_idx + cust_offset] = io_addrs_.size();
    AppendIoAddr(addrs[begin_idx], types[begin_idx]);
  }
  return SUCCESS;
}

Status KernelTaskCodeBuilder::AppendWorkspaceAddr(int32_t ir_idx) {
  if (ir_idx < 0) {
    (void)io_addrs_.insert(io_addrs_.cend(), workspace_addrs_.cbegin(), workspace_addrs_.cend());
    (void)io_addr_mem_types_.insert(io_addr_mem_types_.cend(), workspace_mem_types_.cbegin(),
                                    workspace_mem_types_.cend());
  } else {
    const size_t idx = static_cast<size_t>(ir_idx);
    GE_ASSERT(idx < workspace_addrs_.size(), "[OM2]workspace index[%zu] is output of workspace addrs range[%zu]", idx,
              workspace_addrs_.size());
    AppendIoAddr(workspace_addrs_[idx], workspace_mem_types_[idx]);
    GELOGI("[OM2]op[%s], workspace_addrs_[%zu] = 0x%" PRIx64 ", workspace_mem_types_[%zu] = %" PRIu64,
           op_desc_->GetName().c_str(), idx, workspace_addrs_[idx], idx, workspace_mem_types_[idx]);
    if (task_type_ == ModelTaskType::MODEL_TASK_PREPROCESS_KERNEL && kernel_type_ == ccKernelType::CUST_AI_CPU) {
      const std::vector<int64_t> v_workspace_bytes = op_desc_->GetWorkspaceBytes();
      GE_ASSERT(idx < v_workspace_bytes.size(), "[OM2]workspace index[%zu] is output of workspace bytes range[%zu]",
                idx, v_workspace_bytes.size());
      AppendIoAddr(v_workspace_bytes[idx], om2::IowMemoryType::kAbsoluteMemType);
      GELOGI("[OM2]preprocess custom op[%s], v_workspace_bytes[%zu] = %" PRId64, op_desc_->GetName().c_str(), idx,
             v_workspace_bytes[idx]);
    }
  }
  return SUCCESS;
}

Status KernelTaskCodeBuilder::AssembleShapeInfoAddrs(const std::vector<ArgDesc> &dynamic_args_desc,
                                                     const std::vector<size_t> &level2_addr_idx) {
  std::map<size_t, std::pair<size_t, size_t>> &ir_input_2_range = args_format_holder_.ir_input_2_range;
  std::map<size_t, std::pair<size_t, size_t>> &ir_output_2_range = args_format_holder_.ir_output_2_range;
  // append additional level1 addr
  GE_ASSERT(dynamic_args_desc.size() == args_format_holder_.shape_infos.size());
  for (size_t i = 0UL; i < dynamic_args_desc.size(); ++i) {
    auto &shape_info = args_format_holder_.shape_infos[i];
    const size_t ptr_offset_idx = io_addrs_.size();
    GE_ASSERT(level2_addr_idx[i] < io_addrs_.size());
    io_addrs_[level2_addr_idx[i]] = PtrToValue(args_) + static_cast<uint64_t>(ptr_offset_idx * sizeof(uint64_t));
    GELOGD("[OM2]Set ptr_offset idx:[%zu], addr:[%" PRIx64 "] io index:[%zu]", ptr_offset_idx,
           io_addrs_[level2_addr_idx[i]], level2_addr_idx[i]);
    (void)io_addrs_.insert(io_addrs_.cend(), shape_info.cbegin(), shape_info.cend());
    (void)io_addr_mem_types_.insert(io_addr_mem_types_.cend(), shape_info.size(), om2::IowMemoryType::kAbsoluteMemType);

    if (dynamic_args_desc[i].addr_type == AddrType::INPUT_DESC) {
      const size_t ir_idx = static_cast<size_t>(dynamic_args_desc[i].ir_idx);
      const auto &range_pair = ir_input_2_range[ir_idx];
      size_t begin_idx = range_pair.first;
      for (size_t idx = 0UL; idx < range_pair.second; ++idx) {
        GE_ASSERT(begin_idx < input_data_addrs_.size(),
                  "[OM2]ir_idx:[%zu], begin_index [%zu] is out of range, max_size:[%zu].", ir_idx, begin_idx,
                  input_data_addrs_.size());
        cust_to_relevant_offset_[begin_idx] = io_addrs_.size();
        AppendIoAddr(input_data_addrs_[begin_idx], input_mem_types_[begin_idx]);
        ++begin_idx;
      }
    } else if (dynamic_args_desc[i].addr_type == AddrType::OUTPUT_DESC) {
      const size_t ir_idx = static_cast<size_t>(dynamic_args_desc[i].ir_idx);
      const auto &range_pair = ir_output_2_range[ir_idx];
      size_t begin_idx = range_pair.first;
      for (size_t idx = 0UL; idx < range_pair.second; ++idx) {
        GE_ASSERT(begin_idx < output_data_addrs_.size(),
                  "[OM2]ir_idx:[%zu], begin_index [%zu] is out of range, max_size:[%zu].", ir_idx, begin_idx,
                  output_data_addrs_.size());
        cust_to_relevant_offset_[begin_idx + input_data_addrs_.size()] = io_addrs_.size();
        AppendIoAddr(output_data_addrs_[begin_idx], output_mem_types_[begin_idx]);
        ++begin_idx;
      }
    } else {
    }
  }
  return SUCCESS;
}

Status KernelTaskCodeBuilder::AssembleIoByArgsFormat() {
  const auto &arg_descs = args_format_holder_.arg_descs;
  io_addrs_.reserve(arg_descs.size());
  io_addr_mem_types_.reserve(arg_descs.size());
  std::vector<ArgDesc> dynamic_args_desc;
  std::vector<size_t> level_addr_idx;
  std::vector<void *> context_addrs;
  for (const auto &arg_format : arg_descs) {
    switch (arg_format.addr_type) {
      case AddrType::INPUT_INSTANCE: {
        GE_ASSERT_SUCCESS(AppendInputOutputAddrByInstanceIndex(static_cast<size_t>(arg_format.ir_idx), true));
        break;
      }
      case AddrType::OUTPUT_INSTANCE: {
        GE_ASSERT_SUCCESS(AppendInputOutputAddrByInstanceIndex(static_cast<size_t>(arg_format.ir_idx), false));
        break;
      }
      case AddrType::INPUT_DESC:
      case AddrType::OUTPUT_DESC: {
        level_addr_idx.push_back(io_addrs_.size());
        dynamic_args_desc.push_back(arg_format);
        AppendIoAddr(0UL, om2::IowMemoryType::kAbsoluteMemType);
        break;
      }
      case AddrType::INPUT: {
        GE_ASSERT_SUCCESS(AppendInputOutputAddr(static_cast<size_t>(arg_format.ir_idx), true));
        break;
      }
      case AddrType::OUTPUT: {
        GE_ASSERT_SUCCESS(AppendInputOutputAddr(static_cast<size_t>(arg_format.ir_idx), false));
        break;
      }
      case AddrType::WORKSPACE: {
        GE_ASSERT_SUCCESS(AppendWorkspaceAddr(arg_format.ir_idx));
        break;
      }
      case AddrType::PLACEHOLDER: {
        AppendIoAddr(0UL, om2::IowMemoryType::kAbsoluteMemType);
        break;
      }
      case AddrType::CUSTOM_VALUE: {
        AppendIoAddr(*reinterpret_cast<const uint64_t *>(arg_format.reserved), om2::IowMemoryType::kAbsoluteMemType);
        break;
      }
      case AddrType::FFTS_ADDR: {
        AppendIoAddr(0UL, om2::IowMemoryType::kAbsoluteMemType);
        break;
      }
      default:
        break;
    }
  }
  GE_ASSERT_SUCCESS(AssembleShapeInfoAddrs(dynamic_args_desc, level_addr_idx));
  return SUCCESS;
}

Status KernelTaskCodeBuilder::SetIoAddrsForCustomized() {
  if (kernel_type_ != ccKernelType::CUSTOMIZED) {
    return SUCCESS;
  }
  std::vector<uint64_t> mem_types;
  std::vector<uint64_t> tensor_device_addrs;
  const size_t kernel_def_args_size_align =
      MemSizeAlign(static_cast<size_t>(customized_args_info_.kernel_def_args_size), kAddressLen);
  const size_t args_num = kernel_def_args_size_align / kAddressLen;
  (void)tensor_device_addrs.insert(tensor_device_addrs.cend(), args_num, 0);
  (void)mem_types.insert(mem_types.cend(), args_num, static_cast<uint64_t>(om2::MemoryAppType::kMemoryTypeFix));
  GELOGD("[OM2]customized has kernel_def_args_size:%u, after align:%u, args num:%zu",
         customized_args_info_.kernel_def_args_size, kernel_def_args_size_align, args_num);

  (void)tensor_device_addrs.insert(tensor_device_addrs.cend(), input_data_addrs_.cbegin(), input_data_addrs_.cend());
  (void)tensor_device_addrs.insert(tensor_device_addrs.cend(), output_data_addrs_.cbegin(), output_data_addrs_.cend());
  (void)mem_types.insert(mem_types.cend(), input_mem_types_.cbegin(), input_mem_types_.cend());
  (void)mem_types.insert(mem_types.cend(), output_mem_types_.cbegin(), output_mem_types_.cend());

  io_addrs_.resize(tensor_device_addrs.size());
  (void)io_addr_mem_types_.insert(io_addr_mem_types_.cend(), mem_types.cbegin(), mem_types.cend());
  size_t args_size = 0UL;
  GE_ASSERT_TRUE(!ge::MulOverflow(io_addrs_.size(), kAddressLen, args_size));
  return SUCCESS;
}

Status KernelTaskCodeBuilder::SetIoAddrs() {
  if (kernel_type_ == ccKernelType::CUSTOMIZED) {
    return SetIoAddrsForCustomized();
  }
  std::vector<uint64_t> mem_types;
  std::vector<uint64_t> tensor_device_addrs;
  if (!is_separately_clean_task_) {
    (void)tensor_device_addrs.insert(tensor_device_addrs.cend(), input_data_addrs_.cbegin(), input_data_addrs_.cend());
    (void)tensor_device_addrs.insert(tensor_device_addrs.cend(), output_data_addrs_.cbegin(),
                                     output_data_addrs_.cend());
    (void)mem_types.insert(mem_types.cend(), input_mem_types_.cbegin(), input_mem_types_.cend());
    (void)mem_types.insert(mem_types.cend(), output_mem_types_.cbegin(), output_mem_types_.cend());
  }

  if (Om2CodegenUtils::IsAICoreKernel(kernel_type_)) {
    if (!is_separately_clean_task_) {
      (void)tensor_device_addrs.insert(tensor_device_addrs.cend(), workspace_addrs_.cbegin(), workspace_addrs_.cend());
      (void)mem_types.insert(mem_types.cend(), workspace_mem_types_.cbegin(), workspace_mem_types_.cend());
    }
  }

  size_t io_addrs_element_num = tensor_device_addrs.size();
  if (is_addrs_folded_) {
    io_addrs_element_num += 1UL;
  }
  io_addrs_.resize(io_addrs_element_num);
  (void)io_addr_mem_types_.insert(io_addr_mem_types_.cend(), mem_types.cbegin(), mem_types.cend());
  size_t args_size = 0UL;
  GE_ASSERT_TRUE(!ge::MulOverflow(io_addrs_.size(), kAddressLen, args_size));
  return SUCCESS;
}

Status KernelTaskCodeBuilder::InitKernelByContext(const domi::TaskDef &task_def, const domi::KernelContext &context,
                                                  const om2::PisToArgs &args) {
  (void)task_def;
  kernel_type_ = static_cast<ccKernelType>(context.kernel_type());
  if ((kernel_type_ == ccKernelType::AI_CPU) || (kernel_type_ == ccKernelType::CUST_AI_CPU)) {
    args_offset_from_pls_ =
        ge::MemSizeAlign(sizeof(aicpu::AicpuParamHead), sizeof(uintptr_t)) - sizeof(aicpu::AicpuParamHead);
  }
  GE_ASSERT_TRUE((args[static_cast<size_t>(args_placement_)].dev_addr != 0U),
                 "[OM2][Check][Param] Op:%s, dev addr is nullptr.", op_desc_->GetName().c_str());
  args_ = ValueToPtr(args[static_cast<size_t>(args_placement_)].dev_addr + args_offset_from_pls_);
  const bool assemble_by_args_manager =
      (!args_format_holder_.arg_descs.empty()) && (kernel_type_ != ccKernelType::CUSTOMIZED) && (!is_addrs_folded_);
  if (assemble_by_args_manager) {
    GE_ASSERT_SUCCESS(AssembleIoByArgsFormat(), "[OM2][Assemble][Addresses] failed, op = %s.", op_desc_->GetNamePtr());
  } else {
    GE_ASSERT_SUCCESS(SetIoAddrs(), "[OM2][Set][Addresses] failed, op = %s.", op_desc_->GetName().c_str());
  }
  return SUCCESS;
}

Status KernelTaskCodeBuilder::InitTVMContext(const domi::KernelContext &context) {
  if ((context.args_offset().size() / sizeof(uint16_t)) < 1U) {
    REPORT_INNER_ERR_MSG("E19999",
                         "[OM2]args_offset().size():%zu / sizeof(uint16_t) less than 1, op:%s(%s), check invalid",
                         context.args_offset().size(), op_desc_->GetName().c_str(), op_desc_->GetType().c_str());
    GELOGE(FAILED, "[OM2][Check][Param]invalid, args_offset().size():%zu / sizeof(uint16_t) less than 1, op:%s(%s)",
           context.args_offset().size(), op_desc_->GetName().c_str(), op_desc_->GetType().c_str());
    return FAILED;
  }

  uint16_t args_offset = 0U;
  GE_ASSERT_EOK(memcpy_s(&args_offset, sizeof(uint16_t), context.args_offset().data(), sizeof(uint16_t)));
  GE_CHECK_LE(args_offset, args_size_);
  io_addr_offset_ = static_cast<size_t>(args_offset);
  GELOGD("[OM2]Get args_offset[%u] of op[%s]", static_cast<uint32_t>(args_offset), op_desc_->GetName().c_str());
  return SUCCESS;
}

Status KernelTaskCodeBuilder::InitTVMTask(const domi::KernelDef &kernel_def) {
  GELOGD("[OM2]Do InitTVMTask of %s.", op_desc_->GetName().c_str());
  GE_CHK_STATUS_RET_NOLOG(InitTVMContext(kernel_def.context()));
  GELOGI("[OM2]io_addrs_size:%zu, args_size:%zu", io_addrs_.size(), kernel_def.args().size() / kAddressLen);
  if ((io_addrs_.size() * kAddressLen) < kernel_def.args().size()) {
    const size_t offset = io_addrs_.size() * kAddressLen;
    const size_t len = kernel_def.args().size() - offset;
    io_addrs_.resize(MemSizeAlign(static_cast<size_t>(kernel_def.args().size()), kAddressLen) / kAddressLen);
    uint8_t *dst_addr = const_cast<uint8_t *>(reinterpret_cast<const uint8_t *>(io_addrs_.data())) + offset;
    uint8_t *src_addr = const_cast<uint8_t *>(reinterpret_cast<const uint8_t *>(kernel_def.args().data())) + offset;
    const errno_t sec_ret = memcpy_s(dst_addr, len, src_addr, len);
    GE_ASSERT_TRUE(sec_ret == EOK);
    io_addr_mem_types_.resize(io_addrs_.size(), static_cast<uint64_t>(om2::MemoryAppType::kMemoryTypeFix));
  }
  return SUCCESS;
}

Status KernelTaskCodeBuilder::InitAicpuTask(const OpDescPtr &op_desc, const domi::KernelDef &kernel_def) {
  (void)op_desc;
  (void)kernel_def;
  GE_CHECK_GE(args_size_, sizeof(aicpu::AicpuParamHead));
  io_addr_offset_ = sizeof(aicpu::AicpuParamHead);
  return SUCCESS;
}

Status KernelTaskCodeBuilder::InitKernel(const domi::TaskDef &task_def, const om2::PisToArgs &args) {
  const domi::KernelDef &kernel_def = task_def.kernel();
  const domi::KernelContext &context = kernel_def.context();
  is_addrs_folded_ = IsWspAddrFolded(op_desc_);
  GE_CHK_STATUS_RET_NOLOG(InitKernelByContext(task_def, context, args));
  Status ret = FAILED;
  if (Om2CodegenUtils::IsAICoreKernel(kernel_type_)) {
    ret = InitTVMTask(kernel_def);
  } else if (kernel_type_ == ccKernelType::CUSTOMIZED) {
    ret = SUCCESS;
  } else if (kernel_type_ == ccKernelType::AI_CPU_KFC) {
    GELOGE(FAILED, "[OM2] Unsupported ai cpu kfc");
    ret = FAILED;
  } else if ((kernel_type_ == ccKernelType::AI_CPU) || (kernel_type_ == ccKernelType::CUST_AI_CPU)) {
    ret = InitAicpuTask(op_desc_, kernel_def);
  } else {
    REPORT_INNER_ERR_MSG("E19999", "[OM2]Node op:%s(%s) kernel type invalid", op_desc_->GetName().c_str(),
                         op_desc_->GetType().c_str());
    GELOGE(FAILED, "[OM2][Check][Param] Node op:%s(%s) kernel type invalid", op_desc_->GetName().c_str(),
           op_desc_->GetType().c_str());
    return ret;
  }
  GELOGD("[OM2]KernelTaskInfo %s init finish, result=%u.", op_desc_->GetNamePtr(), ret);
  return ret;
}

Status KernelTaskCodeBuilder::InitTVMTask(const domi::KernelDefWithHandle &kernel_def) {
  GELOGD("[OM2]Do InitTVMTask with handle of %s.", op_desc_->GetName().c_str());
  GE_CHK_STATUS_RET_NOLOG(InitTVMContext(kernel_def.context()));
  return SUCCESS;
}

Status KernelTaskCodeBuilder::InitKernelWithHandle(const domi::TaskDef &task_def, const om2::PisToArgs &args) {
  const domi::KernelDefWithHandle &kernel_def = task_def.kernel_with_handle();
  const domi::KernelContext &context = kernel_def.context();
  GE_CHK_STATUS_RET_NOLOG(InitKernelByContext(task_def, context, args));

  if (!Om2CodegenUtils::IsAICoreKernel(kernel_type_)) {
    GELOGE(FAILED, "[OM2]Op[%s] kernel type[%d] invalid.", op_desc_->GetName().c_str(),
           static_cast<int32_t>(kernel_type_));
    return FAILED;
  }
  GE_CHK_STATUS_RET_NOLOG(InitTVMTask(kernel_def));
  GELOGD("[OM2]KernelTaskInfo %s init with handle finish.", op_desc_->GetNamePtr());
  return SUCCESS;
}

Status KernelTaskCodeBuilder::Init(const domi::TaskDef &task_def,
                                   std::vector<om2::MemAllocation> &logical_mem_allocations, const om2::PisToArgs &args,
                                   const om2::IowAddrs &iow_addrs) {
  (void)task_def;
  UpdateIoAndWorkspaceAddrs(iow_addrs);
  if (Om2CodegenUtils::IsAllKernel(task_type_)) {
    GE_CHK_STATUS_RET_NOLOG(InitKernelWithHandle(task_def, args));
  } else {
    GE_CHK_STATUS_RET_NOLOG(InitKernel(task_def, args));
  }
  io_addr_mem_types_.resize(io_addrs_.size(), static_cast<uint64_t>(om2::MemoryAppType::kMemoryTypeFix));
  GE_ASSERT_SUCCESS(args_io_addrs_updater_.Init(logical_mem_allocations, io_addrs_, io_addr_mem_types_,
                                                {op_desc_->GetName(), op_desc_->GetType()}));

  if ((kernel_type_ == ccKernelType::AI_CPU) || (kernel_type_ == ccKernelType::CUST_AI_CPU) ||
      (kernel_type_ == ccKernelType::AI_CPU_KFC)) {
    uint32_t pls = static_cast<uint32_t>(args_placement_);
    GE_ASSERT_TRUE(args[pls].len >= args_offset_from_pls_);
  }
  return SUCCESS;
}

Status KernelTaskCodeBuilder::GetTaskArgsRefreshInfos(std::vector<om2::TaskArgsRefreshInfo> &infos) {
  GELOGI("[OM2]KernelTaskCodeBuilder::GetTaskArgsRefreshInfos in.");
  if (Om2CodegenUtils::IsAICoreKernel(kernel_type_) || kernel_type_ == ccKernelType::CUSTOMIZED) {
    args_io_addrs_updater_.GenArgsRefreshInfos(infos, 0UL, args_placement_);
    return SUCCESS;
  }

  if ((kernel_type_ == ccKernelType::AI_CPU) || (kernel_type_ == ccKernelType::CUST_AI_CPU) ||
      (kernel_type_ == ccKernelType::AI_CPU_KFC)) {
    args_io_addrs_updater_.GenArgsRefreshInfos(infos, io_addr_offset_ + args_offset_from_pls_, args_placement_);
    return SUCCESS;
  }
  return SUCCESS;
}

REGISTER_TASK_CODE_BUILDER(MODEL_TASK_KERNEL, KernelTaskCodeBuilder);
REGISTER_TASK_CODE_BUILDER(MODEL_TASK_ALL_KERNEL, KernelTaskCodeBuilder);
REGISTER_TASK_CODE_BUILDER(MODEL_TASK_VECTOR_KERNEL, KernelTaskCodeBuilder);
REGISTER_TASK_CODE_BUILDER(MODEL_TASK_VECTOR_ALL_KERNEL, KernelTaskCodeBuilder);
REGISTER_TASK_CODE_BUILDER(MODEL_TASK_PREPROCESS_KERNEL, KernelTaskCodeBuilder);
}  // namespace ge
