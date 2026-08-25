/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph/load/model_manager/task_info/ge/custom_task_info.h"

#include <cinttypes>
#include <limits>
#include <set>

#include "acl/acl_rt.h"
#include "common/checker.h"
#include "common/ge_common/ge_types.h"
#include "common/kernel_handles_manager/kernel_handle_utils.h"
#include "common/tbe_handle_store/tbe_handle_store.h"
#include "exe_graph/runtime/eager_op_execution_context.h"
#include "exe_graph/runtime/update_args_context.h"
#include "framework/runtime/args_handler.h"
#include "framework/runtime/subscriber/global_dumper.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/debug/ge_util.h"
#include "graph/load/model_manager/model_manager.h"
#include "graph/load/model_manager/model_utils.h"
#include "graph/manager/graph_var_manager.h"
#include "graph/utils/args_format_desc_utils.h"
#include "graph/utils/math_util.h"
#include "graph/args_format_desc.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/node_utils.h"
#include "graph/custom_op.h"
#include "graph/custom_op/cast.h"
#include "graph/custom_op_registry.h"
#include "graph/load/model_manager/sink_only_allocator.h"
#include "graph/load/model_manager/task_info/ge/sink_op_args_handler.h"
#include "graph/manager/graph_var_manager.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/tensor_utils.h"
#include "exe_graph/lowering/data_dependent_interpreter.h"

namespace ge {

namespace {
const ge::char_t *const kDumpOutput = "output";
const ge::char_t *const kDumpInput = "input";
constexpr uint32_t kAddressLen = static_cast<uint32_t>(sizeof(uint64_t));
constexpr size_t kCustomOpArgsReserved = 16UL;
constexpr size_t kCustomOpArgsFieldSize = sizeof(void *);
constexpr const char_t *kDefaultCustomKernelMagic = "RT_DEV_BINARY_MAGIC_ELF_AIVEC";

bool IsInputDescValid(const ge::GeTensorDesc &input_desc, size_t &invalid_index_num) {
  if (input_desc.IsValid() != ge::GRAPH_SUCCESS) {
    if (invalid_index_num < std::numeric_limits<size_t>::max()) {
      invalid_index_num++;
    }
    return false;
  }
  return true;
}

void GetStorageShape(const ge::GeTensorDesc &tensor_desc, gert::StorageShape &storage_shape) {
  const auto &storage_dims = tensor_desc.GetShape().GetDims();
  for (const auto &dim : storage_dims) {
    (void)storage_shape.MutableStorageShape().AppendDim(dim);
  }
  const auto &origin_dims = tensor_desc.GetOriginShape().GetDims();
  for (const auto &dim : origin_dims) {
    (void)storage_shape.MutableOriginShape().AppendDim(dim);
  }
}

Status TryCopyNonTensorInputToHost(const ge::OpDescPtr &op_desc, const ge::GeTensorDescPtr &input_desc,
                                   size_t instance_index, const std::vector<int64_t> &input_kinds,
                                   const gert::DataDependentInterpreter *ddi, gert::TensorAddress &address,
                                   gert::TensorPlacement &placement,
                                   std::vector<std::unique_ptr<uint8_t[]>> &host_input_mem) {
  if ((ddi == nullptr) || input_kinds.empty()) {
    return SUCCESS;
  }
  size_t ir_index = 0U;
  if (ge::OpDescUtils::GetInputIrIndexByInstanceIndex(op_desc, instance_index, ir_index) != ge::SUCCESS) {
    return SUCCESS;
  }
  int64_t non_tensor_base = 3L;
  (void)ge::AttrUtils::GetInt(op_desc, "_custom_op_non_tensor_kind_base", non_tensor_base);
  if (ir_index >= input_kinds.size() || input_kinds[ir_index] < non_tensor_base) {
    return SUCCESS;
  }
  bool is_data_dependent = false;
  if (ddi->IsDataDependent(static_cast<int32_t>(instance_index), is_data_dependent) != ge::GRAPH_SUCCESS ||
      !is_data_dependent) {
    return SUCCESS;
  }
  int64_t tensor_size = 0;
  GE_ASSERT_SUCCESS(ge::TensorUtils::GetTensorSizeInBytes(*input_desc, tensor_size));
  if (tensor_size > 0) {
    GELOGI(
        "NeedHostInput: op=%s, input instance_index=%zu, ir_index=%zu, input_kind=%ld, non_tensor_base=%ld, "
        "D2H copy %ld bytes to host",
        op_desc->GetNamePtr(), instance_index, ir_index, input_kinds[ir_index], non_tensor_base, tensor_size);
    auto host_mem = ge::ComGraphMakeUnique<uint8_t[]>(static_cast<size_t>(tensor_size));
    GE_ASSERT_NOTNULL(host_mem);
    GE_ASSERT_RT_OK(aclrtMemcpy(host_mem.get(), static_cast<size_t>(tensor_size), address,
                                static_cast<size_t>(tensor_size), ACL_MEMCPY_DEVICE_TO_HOST));
    address = host_mem.get();
    placement = gert::kOnHost;
    host_input_mem.push_back(std::move(host_mem));
  }
  return SUCCESS;
}

Status ConstructOutputTensorHolders(const ge::OpDescPtr &op_desc, const std::vector<uint64_t> &output_data_addrs,
                                    std::vector<std::unique_ptr<uint8_t[]>> &outputs) {
  for (size_t i = 0UL; i < op_desc->GetOutputsSize(); i++) {
    gert::StorageShape storage_shape;
    auto output_desc = op_desc->MutableOutputDesc(i);
    GE_ASSERT_NOTNULL(output_desc);
    GetStorageShape(*output_desc, storage_shape);
    GE_ASSERT_TRUE((output_data_addrs.size() > i), "output index %zu is invalid, total output size %zu", i,
                   output_data_addrs.size());
    gert::TensorAddress address = ValueToPtr(output_data_addrs[i]);
    std::unique_ptr<uint8_t[]> tensor_holder = ge::ComGraphMakeUnique<uint8_t[]>(sizeof(gert::Tensor));
    GE_ASSERT_NOTNULL(tensor_holder, "Create context holder outputs failed.");
    new (tensor_holder.get())
        gert::Tensor(storage_shape, {output_desc->GetOriginFormat(), output_desc->GetFormat(), {}}, gert::kOnDeviceHbm,
                     output_desc->GetDataType(), address);
    (void)outputs.emplace_back(std::move(tensor_holder));
  }
  return SUCCESS;
}

// inputs layout is input tensors
std::vector<void *> GetHoldersRawPtr(const std::vector<std::unique_ptr<uint8_t[]>> &holders) {
  std::vector<void *> holderRawPtr;
  holderRawPtr.reserve(holders.size());
  for (const auto &holder : holders) {
    (void)holderRawPtr.emplace_back(holder.get());
  }
  return holderRawPtr;
}

Status GetCustomKernelBinaryMagic(const OpDescPtr &op_desc, int32_t &binary_magic) {
  GE_ASSERT_NOTNULL(op_desc);
  std::string magic_value;
  (void)AttrUtils::GetStr(op_desc, TVM_ATTR_NAME_MAGIC, magic_value);
  if (magic_value.empty()) {
    magic_value = kDefaultCustomKernelMagic;
    GELOGW("Custom op %s(%s) has no %s attr, use default magic %s.", op_desc->GetNamePtr(), op_desc->GetTypePtr(),
           TVM_ATTR_NAME_MAGIC.c_str(), magic_value.c_str());
  }

  if (magic_value == "RT_DEV_BINARY_MAGIC_ELF") {
    binary_magic = RT_DEV_BINARY_MAGIC_ELF;
    return SUCCESS;
  }
  if (magic_value == "RT_DEV_BINARY_MAGIC_ELF_AIVEC") {
    binary_magic = RT_DEV_BINARY_MAGIC_ELF_AIVEC;
    return SUCCESS;
  }
  if (magic_value == "RT_DEV_BINARY_MAGIC_ELF_AICUBE") {
    binary_magic = RT_DEV_BINARY_MAGIC_ELF_AICUBE;
    return SUCCESS;
  }

  GELOGE(PARAM_INVALID, "[CUSTOM OP] invalid %s attr %s for op %s(%s)", TVM_ATTR_NAME_MAGIC.c_str(),
         magic_value.c_str(), op_desc->GetNamePtr(), op_desc->GetTypePtr());
  return PARAM_INVALID;
}

std::string MakeCustomKernelBinName(const uint32_t model_id, const OpDescPtr &op_desc, const std::string &kernel_name) {
  return std::to_string(model_id) + "_" + op_desc->GetName() + "_" + kernel_name;
}

ArgsRefreshStrategy GetLegacyCustomTaskArgsRefreshStrategy(const AscendString &op_type,
                                                           const domi::KernelContext &context,
                                                           const CustomOpRegistryPtr &custom_op_registry) {
  const auto registry_strategy = custom_op_registry->GetArgsRefreshStrategy(op_type);
  if ((registry_strategy == ArgsRefreshStrategy::kNone) && (!context.args_format().empty())) {
    return ArgsRefreshStrategy::kAnnotatedArgs;
  }
  return registry_strategy;
}

Status GetCustomTaskArgsRefreshStrategy(const OpDescPtr &op_desc, const domi::KernelContext &context,
                                        const CustomOpRegistryPtr &custom_op_registry,
                                        ArgsRefreshStrategy &args_refresh_strategy) {
  GE_ASSERT_NOTNULL(op_desc);
  GE_ASSERT_NOTNULL(custom_op_registry);
  const AscendString op_type(op_desc->GetTypePtr());
  if (!op_desc->HasAttr(ATTR_NAME_CUSTOM_TASK_ARGS_MODE)) {
    args_refresh_strategy = GetLegacyCustomTaskArgsRefreshStrategy(op_type, context, custom_op_registry);
    return SUCCESS;
  }

  int64_t args_mode = 0;
  GE_ASSERT_TRUE(AttrUtils::GetInt(op_desc, ATTR_NAME_CUSTOM_TASK_ARGS_MODE, args_mode),
                 "[CUSTOM OP] get %s failed for op %s(%s).", ATTR_NAME_CUSTOM_TASK_ARGS_MODE.c_str(),
                 op_desc->GetNamePtr(), op_desc->GetTypePtr());
  switch (static_cast<CustomTaskArgsMode>(args_mode)) {
    case CustomTaskArgsMode::kUnspecified:
      args_refresh_strategy = GetLegacyCustomTaskArgsRefreshStrategy(op_type, context, custom_op_registry);
      return SUCCESS;
    case CustomTaskArgsMode::kNone:
      args_refresh_strategy = ArgsRefreshStrategy::kNone;
      return SUCCESS;
    case CustomTaskArgsMode::kAnnotatedArgs:
      args_refresh_strategy = ArgsRefreshStrategy::kAnnotatedArgs;
      return SUCCESS;
    case CustomTaskArgsMode::kUpdateCallback:
      GE_ASSERT_TRUE(custom_op_registry->GetArgsRefreshStrategy(op_type) == ArgsRefreshStrategy::kUpdateCallback,
                     "[CUSTOM OP] update callback is not registered for op %s(%s).", op_desc->GetNamePtr(),
                     op_desc->GetTypePtr());
      args_refresh_strategy = ArgsRefreshStrategy::kUpdateCallback;
      return SUCCESS;
    default:
      GELOGE(PARAM_INVALID, "[CUSTOM OP] invalid %s value %" PRId64 " for op %s(%s).",
             ATTR_NAME_CUSTOM_TASK_ARGS_MODE.c_str(), args_mode, op_desc->GetNamePtr(), op_desc->GetTypePtr());
      return PARAM_INVALID;
  }
}
}  // namespace

void CustomTaskInfo::SetCustomDumpInfo(const DumpProperties &dump_properties, DumpOp &dump_op) const {
  dump_op.SetDumpInfo(dump_properties, op_desc_, dump_input_addrs_, dump_output_addrs_, stream_);
  if (davinci_model_->IsKnownNode()) {
    dump_op.SetLoopAddr(davinci_model_->GetGlobalStep(), 0U, 0U);
  } else {
    dump_op.SetLoopAddr(davinci_model_->GetGlobalStep(), davinci_model_->GetLoopPerIter(),
                        davinci_model_->GetLoopCond());
  }
  dump_op.SetDynamicModelInfo(davinci_model_->GetDumpModelName(), davinci_model_->GetOmName(),
                              davinci_model_->GetDumpModelId());
  dump_op.SetRootGraphName(davinci_model_->GetRootGraphName());
}

Status CustomTaskInfo::UpdateCustomDumpAddrs(const std::vector<uint64_t> &input_addrs_value,
                                             const std::vector<uint64_t> &output_addrs_value) {
  GE_CHECK_NOTNULL(davinci_model_);
  GE_CHECK_NOTNULL(op_desc_);
  if (!davinci_model_->OpNeedDump(op_desc_->GetName())) {
    return SUCCESS;
  }

  GELOGI("UpdateCustomDumpAddrs: op_name=%s, inputs=%zu, outputs=%zu", op_desc_->GetName().c_str(),
         input_addrs_value.size(), output_addrs_value.size());
  GE_CHK_STATUS_RET(UpdateDumpInputAddrs(input_addrs_value), "[Update][CustomDumpInputAddrs] fail! op:%s",
                    op_desc_->GetName().c_str());
  GE_ASSERT_TRUE(dump_output_addrs_.size() == output_addrs_value.size(),
                 "Output dump address buffer size[%zu] does not match output address size[%zu], op %s",
                 dump_output_addrs_.size(), output_addrs_value.size(), op_desc_->GetNamePtr());
  for (size_t i = 0U; i < output_addrs_value.size(); ++i) {
    dump_output_addrs_[i] = static_cast<uintptr_t>(output_addrs_value[i]);
  }

  GE_CHK_STATUS_RET(input_custom_dump_.UpdateAddrs(dump_input_addrs_, dump_empty_addrs_),
                    "[Update][CustomDumpAddrs] fail! op:%s", op_desc_->GetName().c_str());
  GE_CHK_STATUS_RET(output_custom_dump_.UpdateAddrs(dump_empty_addrs_, dump_output_addrs_),
                    "[Update][CustomDumpAddrs] fail! op:%s", op_desc_->GetName().c_str());
  return SUCCESS;
}

Status CustomTaskInfo::UpdateDumpInputAddrs(const std::vector<uint64_t> &input_addrs_value) {
  GE_CHECK_NOTNULL(op_desc_);
  // Runtime input addresses omit absent optional inputs, while DumpInput indexes all input descriptors.
  const size_t input_desc_count = op_desc_->GetAllInputsSize();
  if (dump_input_addrs_.size() != input_desc_count) {
    dump_input_addrs_.resize(input_desc_count);
  }
  for (auto &addr : dump_input_addrs_) {
    addr = 0U;
  }

  size_t input_addr_index = 0U;
  for (size_t input_desc_index = 0U; input_desc_index < input_desc_count; ++input_desc_index) {
    if (op_desc_->MutableInputDesc(static_cast<uint32_t>(input_desc_index)) == nullptr) {
      continue;
    }
    GE_ASSERT_TRUE(input_addr_index < input_addrs_value.size(),
                   "Input address index[%zu] exceeds input address size[%zu], op %s", input_addr_index,
                   input_addrs_value.size(), op_desc_->GetNamePtr());
    dump_input_addrs_[input_desc_index] = static_cast<uintptr_t>(input_addrs_value[input_addr_index++]);
  }
  GE_ASSERT_TRUE(input_addr_index == input_addrs_value.size(),
                 "Input address size[%zu] does not match valid input desc size[%zu], op %s", input_addrs_value.size(),
                 input_addr_index, op_desc_->GetNamePtr());
  return SUCCESS;
}

Status CustomTaskInfo::InsertDumpOp(const std::string &dump_mode) {
  GE_CHECK_NOTNULL(davinci_model_);
  GE_CHECK_NOTNULL(op_desc_);
  if (!davinci_model_->OpNeedDump(op_desc_->GetName())) {
    return SUCCESS;
  }

  GELOGI("Data Dump is on, dump custom op for node: %s, type: %s.", op_desc_->GetName().c_str(),
         op_desc_->GetType().c_str());
  auto custom_dump_properties = davinci_model_->GetDumpProperties();
  DumpOp *dump_op = nullptr;
  if (dump_mode == kDumpInput) {
    if (custom_dump_properties.GetDumpMode() == kDumpOutput) {
      return SUCCESS;
    }
    GELOGI("Insert input dump op for custom node: %s, type: %s.", op_desc_->GetName().c_str(),
           op_desc_->GetType().c_str());
    custom_dump_properties.ClearOpDebugFlag();
    custom_dump_properties.SetDumpMode(kDumpInput);
    dump_op = &input_custom_dump_;
  } else if (dump_mode == kDumpOutput) {
    if (custom_dump_properties.GetDumpMode() == kDumpInput) {
      return SUCCESS;
    }
    GELOGI("Insert output dump op for custom node: %s, type: %s.", op_desc_->GetName().c_str(),
           op_desc_->GetType().c_str());
    custom_dump_properties.ClearOpDebugFlag();
    custom_dump_properties.SetDumpMode(kDumpOutput);
    dump_op = &output_custom_dump_;
  } else {
    return SUCCESS;
  }

  // Address buffers are allocated during model loading and reused in the execute path.
  GE_CHK_STATUS_RET(UpdateDumpInputAddrs(input_data_addrs_), "[Update][CustomDumpInputAddrs] fail! op:%s",
                    op_desc_->GetName().c_str());
  dump_output_addrs_.resize(output_data_addrs_.size());
  for (size_t i = 0U; i < output_data_addrs_.size(); ++i) {
    dump_output_addrs_[i] = static_cast<uintptr_t>(output_data_addrs_[i]);
  }
  SetCustomDumpInfo(custom_dump_properties, *dump_op);
  return dump_op->LaunchDumpOp(false, false);
}

Status CustomTaskInfo::ParseTaskRunParam(const domi::TaskDef &task_def, DavinciModel *const davinci_model,
                                         TaskRunParam &task_run_param) {
  GELOGI("CustomTaskInfo  ParseTaskRunParam start");
  const domi::KernelDef &kernel_def = task_def.kernel();
  domi::KernelContext context = kernel_def.context();

  GE_CHECK_NOTNULL(davinci_model);
  op_desc_ = davinci_model->GetOpByIndex(context.op_index());
  GE_CHECK_NOTNULL(op_desc_);

  const RuntimeParam &rts_param = davinci_model->GetRuntimeParam();
  input_data_addrs_ = ModelUtils::GetInputAddrsValue(rts_param, op_desc_, input_mem_types_);
  output_data_addrs_ = ModelUtils::GetOutputAddrsValue(rts_param, op_desc_, output_mem_types_, true);
  exception_dump_io_addrs_.reserve(input_data_addrs_.size() + output_data_addrs_.size());
  workspace_addrs_ = ModelUtils::GetWorkspaceDataAddrsValue(rts_param, op_desc_, workspace_mem_types_);
  GE_ASSERT_SUCCESS(ValidateIoWorkspaceAddrAndMemTypeSizes());

  const auto &custom_op_registry = davinci_model->GetCustomOpRegistry();
  GE_ASSERT_NOTNULL(custom_op_registry, "[CUSTOM OP] custom op registry is nullptr for op %s.",
                    op_desc_->GetName().c_str());
  GE_ASSERT_SUCCESS(GetCustomTaskArgsRefreshStrategy(op_desc_, context, custom_op_registry, args_refresh_strategy_));
  is_args_refreshable_ = args_refresh_strategy_ != ArgsRefreshStrategy::kNone;

  if (args_refresh_strategy_ == ArgsRefreshStrategy::kAnnotatedArgs) {
    return ParseAnnotatedArgsTaskRunParam(kernel_def, context, task_run_param);
  }

  for (size_t i = 0UL; i < input_data_addrs_.size(); i++) {
    task_run_param.parsed_input_addrs.push_back({input_data_addrs_[i], input_mem_types_[i], is_args_refreshable_, {0}});
  }
  for (size_t i = 0UL; i < output_data_addrs_.size(); i++) {
    task_run_param.parsed_output_addrs.push_back(
        {output_data_addrs_[i], output_mem_types_[i], is_args_refreshable_, {0}});
  }
  for (size_t i = 0UL; i < workspace_addrs_.size(); i++) {
    task_run_param.parsed_workspace_addrs.push_back(
        {workspace_addrs_[i], workspace_mem_types_[i], is_args_refreshable_, {0}});
  }
  int64_t io_count = 0;
  GE_ASSERT_TRUE(!ge::AddOverflow(input_data_addrs_.size(), output_data_addrs_.size(), io_count),
                 "[CUSTOM OP] input/output count overflow for op %s", op_desc_->GetNamePtr());
  int64_t args_field_count = 0;
  GE_ASSERT_TRUE(!ge::AddOverflow(io_count, kCustomOpArgsReserved, args_field_count),
                 "[CUSTOM OP] args field count overflow for op %s", op_desc_->GetNamePtr());
  int64_t args_size = 0;
  GE_ASSERT_TRUE(!ge::MulOverflow(args_field_count, kCustomOpArgsFieldSize, args_size),
                 "[CUSTOM OP] args size overflow for op %s", op_desc_->GetNamePtr());
  task_run_param.args_descs.push_back({args_size, args_placement_});
  GELOGI("Get args size[%" PRId64 "] of op[%s], is known node[%d], task_type: %d, placement: %d.", args_size,
         op_desc_->GetName().c_str(), static_cast<int32_t>(davinci_model->IsFeatureBaseRefreshable()),
         static_cast<int32_t>(static_cast<ModelTaskType>(task_def.type())), args_placement_);
  return SUCCESS;
}

Status CustomTaskInfo::ParseAnnotatedArgsTaskRunParam(const domi::KernelDef &kernel_def,
                                                      const domi::KernelContext &context,
                                                      TaskRunParam &task_run_param) {
  const auto &args_format_str = context.args_format();
  GE_ASSERT_TRUE(!args_format_str.empty(), "[CUSTOM OP] kAnnotatedArgs requires non-empty args_format for op %s",
                 op_desc_->GetNamePtr());
  GE_ASSERT_SUCCESS(ArgsFormatDesc::Parse(op_desc_, args_format_str, args_format_holder_.arg_descs));
  GE_ASSERT_SUCCESS(ValidateIoWorkspaceAddrAndMemTypeSizes());

  GE_ASSERT_TRUE(!kernel_def.kernel_name().empty(), "[CUSTOM OP] kAnnotatedArgs kernel_name is empty for op %s",
                 op_desc_->GetNamePtr());
  GE_ASSERT_TRUE(kernel_def.block_dim() > 0U, "[CUSTOM OP] kAnnotatedArgs block_dim is 0 for op %s",
                 op_desc_->GetNamePtr());
  kernel_name_ = kernel_def.kernel_name();
  block_dim_ = kernel_def.block_dim();

  for (size_t i = 0UL; i < input_data_addrs_.size(); i++) {
    task_run_param.parsed_input_addrs.push_back({input_data_addrs_[i], input_mem_types_[i], true, {0}});
  }
  for (size_t i = 0UL; i < output_data_addrs_.size(); i++) {
    task_run_param.parsed_output_addrs.push_back({output_data_addrs_[i], output_mem_types_[i], true, {0}});
  }
  for (size_t i = 0UL; i < workspace_addrs_.size(); i++) {
    task_run_param.parsed_workspace_addrs.push_back({workspace_addrs_[i], workspace_mem_types_[i], true, {0}});
  }

  const auto args_size = GetArgsSizeByFormat();
  GE_ASSERT_TRUE(args_size <= static_cast<size_t>(std::numeric_limits<uint32_t>::max()),
                 "[CUSTOM OP] args_size %zu exceeds uint32 max for op %s", args_size, op_desc_->GetNamePtr());
  task_run_param.args_descs.push_back(
      {static_cast<int64_t>(MemSizeAlign(args_size, sizeof(uintptr_t))), args_placement_});
  GELOGI("kAnnotatedArgs parsed args[%zu] of op[%s], args format[%s], placement: %d.", args_size,
         op_desc_->GetNamePtr(), args_format_str.c_str(), args_placement_);
  return SUCCESS;
}

Status CustomTaskInfo::Init(const domi::TaskDef &task_def, DavinciModel *const davinci_model, const PisToArgs &args,
                            const PisToPersistentWorkspace &persistent_workspace, const IowAddrs &iow_addrs) {
  GE_CHECK_NOTNULL(davinci_model);
  GE_CHECK_NOTNULL(op_desc_);
  GELOGI("CustomTaskInfo Init Start, op: %s", op_desc_->GetNamePtr());

  (void)persistent_workspace;
  davinci_model_ = davinci_model;
  GE_CHK_STATUS_RET_NOLOG(SetStream(task_def.stream_id(), davinci_model_->GetStreamList()));

  GE_ASSERT_TRUE(
      iow_addrs.input_logic_addrs.empty() || (iow_addrs.input_logic_addrs.size() == input_data_addrs_.size()),
      "Input IOW address size[%zu] does not match parsed input size[%zu].", iow_addrs.input_logic_addrs.size(),
      input_data_addrs_.size());
  GE_ASSERT_TRUE(
      iow_addrs.output_logic_addrs.empty() || (iow_addrs.output_logic_addrs.size() == output_data_addrs_.size()),
      "Output IOW address size[%zu] does not match parsed output size[%zu].", iow_addrs.output_logic_addrs.size(),
      output_data_addrs_.size());
  GE_ASSERT_TRUE(
      iow_addrs.workspace_logic_addrs.empty() || (iow_addrs.workspace_logic_addrs.size() == workspace_addrs_.size()),
      "Workspace IOW address size[%zu] does not match parsed workspace size[%zu].",
      iow_addrs.workspace_logic_addrs.size(), workspace_addrs_.size());
  UpdateIoAndWorkspaceAddrs(iow_addrs);
  GE_ASSERT_SUCCESS(ValidateIoWorkspaceAddrAndMemTypeSizes());
  stream_id_ = task_def.stream_id();
  GE_ASSERT_TRUE((args[static_cast<size_t>(args_placement_)].dev_addr != 0U),
                 "[Check][Param] Op:%s, dev addr is nullptr.", op_desc_->GetName().c_str());
  auto mem_block_manager_allocator = davinci_model_->GetAllocator();
  sink_only_allocator_ = ComGraphMakeShared<gert::memory::SinkOnlyAllocator>();
  sink_only_allocator_->SetAllocator(mem_block_manager_allocator);

  if (args_refresh_strategy_ == ArgsRefreshStrategy::kAnnotatedArgs) {
    args_ = ValueToPtr(args[static_cast<size_t>(args_placement_)].dev_addr);
    GE_ASSERT_SUCCESS(AssembleIoByArgsFormat());
    ArgsIoAddrsUpdater::OpInfo op_info{op_desc_->GetName(), op_desc_->GetType()};
    GE_ASSERT_SUCCESS(
        args_io_addrs_updater_.Init(davinci_model_->GetLogicalMemAllocation(), io_addrs_, io_addr_mem_types_, op_info));
  }

  GELOGI("CustomTaskInfo Init Success, node: %s, logic stream id: %u, stream: %p, args_refresh_strategy: %d.",
         op_desc_->GetName().c_str(), task_def.stream_id(), stream_, static_cast<int32_t>(args_refresh_strategy_));
  return SUCCESS;
}

void CustomTaskInfo::UpdateIoAndWorkspaceAddrs(const IowAddrs &iow_addrs) {
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
Status CustomTaskInfo::ConstructCustomKernelContextInputsOutputs(const ge::OpDescPtr &op_desc,
                                                                 std::vector<std::unique_ptr<uint8_t[]>> &inputs,
                                                                 std::vector<std::unique_ptr<uint8_t[]>> &outputs) {
  std::vector<int64_t> input_kinds;
  (void)ge::AttrUtils::GetListInt(op_desc, "input_kinds", input_kinds);
  auto space_registries = davinci_model_->GetSpaceRegistries();
  std::unique_ptr<gert::DataDependentInterpreter> ddi = nullptr;
  if (space_registries != nullptr) {
    ddi = ge::ComGraphMakeUnique<gert::DataDependentInterpreter>(op_desc, *space_registries);
  }

  size_t invalid_index_num = 0UL;
  for (size_t i = 0UL; i < op_desc->GetAllInputsSize(); i++) {
    if (!IsInputDescValid(op_desc->GetInputDesc(static_cast<uint32_t>(i)), invalid_index_num)) {
      GELOGD("input desc is not valid, skip add input[%zu] into context inputs.", i);
      continue;
    }
    gert::StorageShape storage_shape;
    auto input_desc = op_desc->MutableInputDesc(i);
    GE_ASSERT_NOTNULL(input_desc);
    GetStorageShape(*input_desc, storage_shape);
    const size_t instance_index = i - invalid_index_num;
    GE_ASSERT_TRUE((input_data_addrs_.size() > instance_index),
                   "instance_index %zu is invalid, %zu - %zu, total input size %zu", instance_index, i,
                   invalid_index_num, input_data_addrs_.size());
    gert::TensorAddress address = ValueToPtr(input_data_addrs_[instance_index]);
    gert::TensorPlacement placement = gert::kOnDeviceHbm;

    GE_ASSERT_SUCCESS(TryCopyNonTensorInputToHost(op_desc, input_desc, instance_index, input_kinds, ddi.get(), address,
                                                  placement, host_input_mem_));

    std::unique_ptr<uint8_t[]> tensor_holder = ge::ComGraphMakeUnique<uint8_t[]>(sizeof(gert::Tensor));
    GE_ASSERT_NOTNULL(tensor_holder, "Create context holder inputs failed.");
    new (tensor_holder.get()) gert::Tensor(storage_shape, {input_desc->GetOriginFormat(), input_desc->GetFormat(), {}},
                                           placement, input_desc->GetDataType(), address);
    (void)inputs.emplace_back(std::move(tensor_holder));
  }
  return ConstructOutputTensorHolders(op_desc, output_data_addrs_, outputs);
}

Status CustomTaskInfo::Distribute() {
  GE_ASSERT_NOTNULL(op_desc_);
  GELOGI("CustomTaskInfo Distribute Start, op: %s", op_desc_->GetName().c_str());
  const TaskProfGuarder prof_guarder(this);

  AscendString op_type(op_desc_->GetType().c_str());
  GE_ASSERT_NOTNULL(davinci_model_);

  if (args_refresh_strategy_ == ArgsRefreshStrategy::kAnnotatedArgs) {
    GE_CHK_STATUS_RET(InsertDumpOp(kDumpInput), "Insert custom input dump op failed, node: %s", op_desc_->GetNamePtr());
    GE_ASSERT_SUCCESS(DistributeAnnotatedArgsFromTaskDef());
    GE_CHK_STATUS_RET(InsertDumpOp(kDumpOutput), "Insert custom output dump op failed, node: %s",
                      op_desc_->GetNamePtr());
    return SUCCESS;
  }

  const auto &custom_op_registry = davinci_model_->GetCustomOpRegistry();
  GE_ASSERT_NOTNULL(custom_op_registry, "[CUSTOM OP] custom op registry is nullptr for op %s.",
                    op_desc_->GetName().c_str());
  BaseCustomOp *custom_op_ptr = custom_op_registry->CreateOrGetCustomOp(op_type);
  GE_ASSERT_NOTNULL(custom_op_ptr, "[CUSTOM OP] custom op %s is not found in registry.", op_desc_->GetType().c_str());

  args_update_op_ = CustomOpCast<ArgsUpdater>(custom_op_ptr);
  if (args_update_op_ != nullptr) {
    GELOGI("ArgsUpdater operator detected: %s", op_desc_->GetName().c_str());
  }

  GE_ASSERT_SUCCESS(ConstructCustomKernelContextInputsOutputs(op_desc_, inputs_holder_, outputs_holder_));

  args_handler_ = ge::ComGraphMakeUnique<SinkOpArgsHandler>(this);
  GE_ASSERT_NOTNULL(args_handler_);
  std::vector<void *> additional_inputs = {sink_only_allocator_.get(), stream_};
  std::vector<void *> additional_outputs = {&ws_vec_, args_handler_.get()};

  eager_context_holder_ = gert::KernelRunContextBuilder()
                              .Inputs(GetHoldersRawPtr(inputs_holder_))
                              .Inputs(additional_inputs)
                              .Outputs(GetHoldersRawPtr(outputs_holder_))
                              .Outputs(additional_outputs)
                              .Build(op_desc_);
  auto eager_context = reinterpret_cast<gert::EagerOpExecutionContext *>(eager_context_holder_.context_);
  auto *eager_execute_op_ptr = CustomOpCast<ge::EagerExecuteOp>(custom_op_ptr);
  if (eager_execute_op_ptr == nullptr) {
    GELOGW("%s is custom op but did not implement EagerExecuteOp", eager_context->GetNodeType());
    return ge::GRAPH_FAILED;
  }
  GE_CHK_STATUS_RET(InsertDumpOp(kDumpInput), "Insert custom input dump op failed, node: %s", op_desc_->GetNamePtr());
  GE_ASSERT_SUCCESS(eager_execute_op_ptr->Execute(eager_context));
  GE_CHK_STATUS_RET(InsertDumpOp(kDumpOutput), "Insert custom output dump op failed, node: %s", op_desc_->GetNamePtr());

  GE_ASSERT_SUCCESS(InitArgsIoAddrsUpdater());

  input_count_ = input_data_addrs_.size();
  output_count_ = output_data_addrs_.size();

  GELOGI("CustomTaskInfo Distribute Success, node: %s, stream_id: %u, stream: %p, task_id: %u",
         op_desc_->GetName().c_str(), stream_id_, stream_, task_id_);
  return SUCCESS;
}

size_t CustomTaskInfo::GetArgsSizeByFormat() const {
  const auto &arg_descs = args_format_holder_.arg_descs;
  size_t tmp_size = 0UL;
  for (const auto &arg_desc : arg_descs) {
    (void)ArgsFormatDesc::GetArgSize(op_desc_, arg_desc, tmp_size);
  }
  return tmp_size;
}

void CustomTaskInfo::AppendIoAddr(const uint64_t addr, const uint64_t addr_type) {
  io_addrs_.push_back(addr);
  io_addr_mem_types_.push_back(addr_type);
}

Status CustomTaskInfo::ValidateIoWorkspaceAddrAndMemTypeSizes() const {
  GE_ASSERT_TRUE(input_data_addrs_.size() == input_mem_types_.size(),
                 "Input address size[%zu] does not match memory type size[%zu] for op %s.", input_data_addrs_.size(),
                 input_mem_types_.size(), op_desc_->GetNamePtr());
  GE_ASSERT_TRUE(output_data_addrs_.size() == output_mem_types_.size(),
                 "Output address size[%zu] does not match memory type size[%zu] for op %s.", output_data_addrs_.size(),
                 output_mem_types_.size(), op_desc_->GetNamePtr());
  GE_ASSERT_TRUE(workspace_addrs_.size() == workspace_mem_types_.size(),
                 "Workspace address size[%zu] does not match memory type size[%zu] for op %s.", workspace_addrs_.size(),
                 workspace_mem_types_.size(), op_desc_->GetNamePtr());
  return SUCCESS;
}

Status CustomTaskInfo::AppendInputOutputAddrByInstanceIndex(const int32_t instance_index, const bool is_input) {
  GE_ASSERT_TRUE(instance_index >= 0, "Instance index[%d] is negative, input flag[%u].", instance_index, is_input);
  const size_t index = static_cast<size_t>(instance_index);
  const auto &addrs = is_input ? input_data_addrs_ : output_data_addrs_;
  const auto &mem_types = is_input ? input_mem_types_ : output_mem_types_;
  GE_ASSERT_TRUE((index < addrs.size()) && (index < mem_types.size()),
                 "Instance index[%zu] is out of range, input flag[%u], addr size[%zu], type size[%zu].", index,
                 is_input, addrs.size(), mem_types.size());
  AppendIoAddr(addrs[index], mem_types[index]);
  return SUCCESS;
}

Status CustomTaskInfo::AppendWorkspaceAddr(int32_t ir_idx) {
  if (ir_idx < 0) {
    (void)io_addrs_.insert(io_addrs_.cend(), workspace_addrs_.cbegin(), workspace_addrs_.cend());
    (void)io_addr_mem_types_.insert(io_addr_mem_types_.cend(), workspace_mem_types_.cbegin(),
                                    workspace_mem_types_.cend());
  } else {
    const size_t idx = static_cast<size_t>(ir_idx);
    GE_ASSERT(idx < workspace_addrs_.size(), "workspace index[%zu] is out of workspace addrs range[%zu]", idx,
              workspace_addrs_.size());
    AppendIoAddr(workspace_addrs_[idx], workspace_mem_types_[idx]);
  }
  return SUCCESS;
}

Status CustomTaskInfo::AssembleIoByArgsFormat() {
  GE_ASSERT_SUCCESS(ValidateIoWorkspaceAddrAndMemTypeSizes());
  const auto &arg_descs = args_format_holder_.arg_descs;
  io_addrs_.reserve(arg_descs.size());
  io_addr_mem_types_.reserve(arg_descs.size());
  for (const auto &arg_format : arg_descs) {
    switch (arg_format.addr_type) {
      case AddrType::INPUT_INSTANCE: {
        GE_ASSERT_SUCCESS(AppendInputOutputAddrByInstanceIndex(arg_format.ir_idx, true));
        break;
      }
      case AddrType::OUTPUT_INSTANCE: {
        GE_ASSERT_SUCCESS(AppendInputOutputAddrByInstanceIndex(arg_format.ir_idx, false));
        break;
      }
      case AddrType::INPUT:
      case AddrType::OUTPUT: {
        GELOGE(FAILED, "[CUSTOM OP] legacy IR-index addr_type %d is unsupported for AnnotatedArgs op %s",
               static_cast<int32_t>(arg_format.addr_type), op_desc_->GetNamePtr());
        return FAILED;
      }
      case AddrType::WORKSPACE: {
        GE_ASSERT_SUCCESS(AppendWorkspaceAddr(arg_format.ir_idx));
        break;
      }
      case AddrType::CUSTOM_VALUE: {
        AppendIoAddr(*reinterpret_cast<const uint64_t *>(arg_format.reserved), kAbsoluteMemType);
        break;
      }
      case AddrType::PLACEHOLDER: {
        AppendIoAddr(0UL, kAbsoluteMemType);
        break;
      }
      default: {
        GELOGE(FAILED, "[CUSTOM OP] unsupported addr_type %d for op %s", static_cast<int32_t>(arg_format.addr_type),
               op_desc_->GetNamePtr());
        return FAILED;
      }
    }
  }
  return SUCCESS;
}

Status CustomTaskInfo::DistributeAnnotatedArgsFromTaskDef() {
  GE_ASSERT_TRUE(!kernel_name_.empty(), "Annotated args kernel_name is empty, op=%s", op_desc_->GetNamePtr());

  const auto kernel_bin = davinci_model_->FindTbeKernelBin(kernel_name_);
  GE_ASSERT_NOTNULL(kernel_bin, "[CUSTOM OP] cannot find kernel bin %s for op %s(%s)", kernel_name_.c_str(),
                    op_desc_->GetNamePtr(), op_desc_->GetTypePtr());

  int32_t binary_magic = 0;
  GE_ASSERT_SUCCESS(GetCustomKernelBinaryMagic(op_desc_, binary_magic));
  AicoreRegisterInfo aicore_register_info;
  aicore_register_info.magic = binary_magic;
  aicore_register_info.kernel_bin = kernel_bin;
  aicore_register_info.kernel_bin_name = MakeCustomKernelBinName(davinci_model_->GetModelId(), op_desc_, kernel_name_);
  KernelRegisterInfo register_info = aicore_register_info;

  auto kernel_handles_manager = davinci_model_->GetKernelHandlesManager(KernelHandleType::kAicore);
  GE_ASSERT_NOTNULL(kernel_handles_manager);
  const auto bin_name = kernel_handles_manager->GenerateKey(register_info);
  auto bin_handle = kernel_handles_manager->GetOrRegisterKernel(register_info, bin_name);
  GE_ASSERT_NOTNULL(bin_handle);
  auto func_handle = KernelHandleUtils::GetFuncHandle(bin_handle, kernel_name_);
  GE_ASSERT_NOTNULL(func_handle);

  SetTaskTag(op_desc_->GetNamePtr());
  LaunchKernelParam launch_kernel_param;
  launch_kernel_param.args = args_;
  launch_kernel_param.args_size = static_cast<uint32_t>(GetArgsSizeByFormat());
  launch_kernel_param.block_dim = block_dim_;
  launch_kernel_param.stream = stream_;
  GE_ASSERT_SUCCESS(KernelHandleUtils::LaunchKernel(func_handle, launch_kernel_param));
  GE_ASSERT_RT_OK(aclrtGetThreadLastTaskId(&task_id_));
  int32_t rt_stream_id = 0;
  GE_ASSERT_RT_OK(aclrtStreamGetId(stream_, &rt_stream_id));
  stream_id_ = static_cast<uint32_t>(rt_stream_id);
  CacheLastTaskExtendInfoIfCollective(op_desc_->GetName(), op_desc_->GetType());
  input_count_ = input_data_addrs_.size();
  output_count_ = output_data_addrs_.size();
  GELOGI(
      "CustomTaskInfo distribute annotated args from TaskDef success, node: %s, kernel: %s, stream_id: %u, task_id: %u",
      op_desc_->GetName().c_str(), kernel_name_.c_str(), stream_id_, task_id_);
  return SUCCESS;
}

Status CustomTaskInfo::Release() {
  aclrtContext ctx = nullptr;
  GE_CHK_RT(aclrtGetCurrentContext(&ctx));
  args_update_op_ = nullptr;
  sink_only_allocator_.reset();
  return SUCCESS;
}

int64_t CustomTaskInfo::ParseOpIndex(const domi::TaskDef &task_def) const {
  const domi::KernelDef &kernel_def = task_def.kernel();
  domi::KernelContext context = kernel_def.context();
  return static_cast<int64_t>(context.op_index());
}

void CustomTaskInfo::PostProcess(const domi::TaskDef &task_def) {
  const domi::KernelDef &kernel_def = task_def.kernel();
  const domi::KernelContext &context = kernel_def.context();
  davinci_model_->SaveDfxInfo(context.op_index(), task_def, *this);
}

const gert::KernelArgs *CustomTaskInfo::MallocReadOnlyDevArgsImpl(void *host_args, size_t args_size) {
  GE_ASSERT_TRUE(host_args != nullptr && args_size != 0U && davinci_model_ != nullptr);

  if (is_args_refreshable_) {
    // 使用预留段分配，支持地址刷新
    ArgsAllocationResult result;
    GE_ASSERT_SUCCESS(davinci_model_->AllocateArgsBuffer(args_size, args_placement_, result));

    GE_ASSERT_EOK(memcpy_s(result.host_addr, args_size, host_args, args_size));

    gert::KernelArgs host_args_entry;
    host_args_entry.args_data = result.host_addr;
    host_args_entry.args_size = args_size;
    host_args_entry.placement = gert::Placement::kPlacementHost;
    kernel_args_host_deque_.push_back(host_args_entry);

    gert::KernelArgs device_args;
    device_args.args_data = reinterpret_cast<void *>(result.device_addr);
    device_args.args_size = args_size;
    device_args.placement = gert::Placement::kPlacementDevice;
    kernel_args_device_deque_.push_back(device_args);

    args_allocation_results_.push_back(result);

    GELOGI(
        "MallocReadOnlyDevArgsImpl: reserved path, task_id=%u, args_size=%zu, "
        "host_addr=%p, device_addr=0x%" PRIx64 ", is_from_reserved=%d, pool_index=%u",
        task_id_, args_size, result.host_addr, result.device_addr, result.is_from_reserved, result.extra_pool_index);

    return &kernel_args_device_deque_.back();
  }

  // 直接分配动态内存 + H2D 拷贝，当前 args_placement_ 仅支持 HBM
  void *device_ptr = davinci_model_->MallocDynamicMemory(args_size, RT_MEMORY_HBM);
  GE_ASSERT_NOTNULL(device_ptr);

  GE_ASSERT_RT_OK(aclrtMemcpy(device_ptr, args_size, host_args, args_size, ACL_MEMCPY_HOST_TO_DEVICE));

  gert::KernelArgs device_args;
  device_args.args_data = device_ptr;
  device_args.args_size = args_size;
  device_args.placement = gert::Placement::kPlacementDevice;
  kernel_args_device_deque_.push_back(device_args);

  GELOGI("MallocReadOnlyDevArgsImpl: dynamic path, task_id=%u, args_size=%zu, device_addr=%p", task_id_, args_size,
         device_ptr);

  return &kernel_args_device_deque_.back();
}

Status CustomTaskInfo::GetTaskArgsRefreshInfos(std::vector<TaskArgsRefreshInfo> &infos) {
  if (args_refresh_strategy_ != ArgsRefreshStrategy::kAnnotatedArgs) {
    return SUCCESS;
  }
  args_io_addrs_updater_.GenArgsRefreshInfos(infos, io_addr_offset_, args_placement_);
  return SUCCESS;
}

const std::deque<gert::KernelArgs> &CustomTaskInfo::GetKernelArgsDeque(gert::Placement placement) const {
  if (placement == gert::Placement::kPlacementHost) {
    return kernel_args_host_deque_;
  } else {
    return kernel_args_device_deque_;
  }
}

Status CustomTaskInfo::UpdateHostArgs(void *base_addr, size_t mem_size) {
  if (args_refresh_strategy_ == ArgsRefreshStrategy::kAnnotatedArgs) {
    return SUCCESS;
  }
  if (args_refresh_strategy_ != ArgsRefreshStrategy::kUpdateCallback) {
    return SUCCESS;
  }
  GE_ASSERT_NOTNULL(args_update_op_);

  auto *active_mem_base_addr = reinterpret_cast<uint64_t *>(base_addr);
  if (active_mem_base_addr == nullptr || mem_size == 0) {
    GELOGE(FAILED, "active_mem_base_addr is null or mem_size is zero, task_id=%u", task_id_);
    return FAILED;
  }

  std::vector<MemAllocationAndOffset> mem_allocs;
  args_io_addrs_updater_.GetArgsMemAllocationAndOffset(mem_allocs);

  if (mem_allocs.empty()) {
    GELOGE(FAILED, "mem_allocs is empty, no I/O addresses to update, task_id=%u", task_id_);
    return FAILED;
  }

  size_t io_index = 0;
  for (const auto &mem_alloc : mem_allocs) {
    uint64_t allocation_id = mem_alloc.id;
    uint64_t offset = mem_alloc.offset;

    if (allocation_id >= mem_size) {
      GELOGE(FAILED, "allocation_id %" PRIu64 " out-of-bounds (max %" PRIu64 "), io_index=%zu, task_id=%u",
             allocation_id, mem_size - 1, io_index, task_id_);
      return FAILED;
    }

    uint64_t new_addr = active_mem_base_addr[allocation_id] + offset;

    gert::Tensor *tensor = nullptr;
    if (io_index < input_count_) {
      input_data_addrs_[io_index] = new_addr;
      auto *chain = eager_context_holder_.context_->MutableInput(io_index);
      if (chain != nullptr) {
        tensor = chain->GetPointer<gert::Tensor>();
      }
    } else if (io_index < input_count_ + output_count_) {
      size_t output_index = io_index - input_count_;
      output_data_addrs_[output_index] = new_addr;
      auto *chain = eager_context_holder_.context_->GetOutput(output_index);
      if (chain != nullptr) {
        tensor = chain->GetPointer<gert::Tensor>();
      }
    }

    if (tensor != nullptr) {
      tensor->MutableTensorData().SetAddr(reinterpret_cast<void *>(new_addr), nullptr);
      const bool is_input = io_index < input_count_;
      GELOGI(
          "UpdateHostArgs: op_name=%s, io_index=%zu, %s[%zu], "
          "allocation_id=%" PRIu64 ", offset=%" PRIu64 ", new_addr=0x%" PRIx64,
          op_desc_->GetName().c_str(), io_index, is_input ? "input" : "output",
          is_input ? io_index : io_index - input_count_, allocation_id, offset, new_addr);
    }

    io_index++;
  }

  auto *ctx = reinterpret_cast<gert::UpdateArgsContext *>(eager_context_holder_.context_);
  graphStatus ret = args_update_op_->UpdateHostArgs(ctx);
  if (ret != GRAPH_SUCCESS) {
    GELOGE(FAILED, "Operator UpdateHostArgs failed, task_id=%u", task_id_);
    return FAILED;
  }

  GE_ASSERT_SUCCESS(UpdateCustomDumpAddrs(input_data_addrs_, output_data_addrs_));

  GELOGD("UpdateHostArgs succeeded, task_id=%u, updated %zu I/O addresses", task_id_, io_index);
  return SUCCESS;
}

Status CustomTaskInfo::UpdateDumpInfos(void *const host_args, const size_t host_args_max_len) {
  if (args_refresh_strategy_ != ArgsRefreshStrategy::kAnnotatedArgs) {
    return SUCCESS;
  }
  GE_CHECK_NOTNULL(davinci_model_);
  GE_CHECK_NOTNULL(op_desc_);
  const bool need_data_dump = davinci_model_->OpNeedDump(op_desc_->GetName());
  const bool need_exception_dump = gert::GlobalDumper::GetInstance()->IsEnable(gert::DumpType::kExceptionDump);
  if (!need_data_dump && !need_exception_dump) {
    return SUCCESS;
  }
  GE_ASSERT_NOTNULL(host_args);

  size_t args_offset = 0U;
  // Validate the complete args buffer before changing the cached addresses.
  for (const auto &arg_desc : args_format_holder_.arg_descs) {
    size_t arg_size = 0U;
    GE_ASSERT_SUCCESS(ArgsFormatDesc::GetArgSize(op_desc_, arg_desc, arg_size));
    GE_ASSERT_TRUE(args_offset <= host_args_max_len && arg_size <= (host_args_max_len - args_offset),
                   "[CUSTOM OP] args buffer is too small for dump info, op %s, offset %zu, arg size %zu, total %zu",
                   op_desc_->GetNamePtr(), args_offset, arg_size, host_args_max_len);

    if ((arg_desc.addr_type == AddrType::INPUT_INSTANCE) || (arg_desc.addr_type == AddrType::OUTPUT_INSTANCE)) {
      GE_ASSERT_TRUE(arg_size >= sizeof(uint64_t), "[CUSTOM OP] address arg size %zu is too small for dump info, op %s",
                     arg_size, op_desc_->GetNamePtr());
      GE_ASSERT_TRUE(arg_desc.ir_idx >= 0, "[CUSTOM OP] address arg index is negative for dump info, op %s",
                     op_desc_->GetNamePtr());
      const size_t index = static_cast<size_t>(arg_desc.ir_idx);
      if (arg_desc.addr_type == AddrType::INPUT_INSTANCE) {
        GE_ASSERT_TRUE(index < input_data_addrs_.size(),
                       "[CUSTOM OP] input instance index %zu is out of range for dump info, op %s", index,
                       op_desc_->GetNamePtr());
      } else {
        GE_ASSERT_TRUE(index < output_data_addrs_.size(),
                       "[CUSTOM OP] output instance index %zu is out of range for dump info, op %s", index,
                       op_desc_->GetNamePtr());
      }
    }
    args_offset += arg_size;
  }

  const auto *args_data = static_cast<const uint8_t *>(host_args);
  args_offset = 0U;
  for (const auto &arg_desc : args_format_holder_.arg_descs) {
    size_t arg_size = 0U;
    GE_ASSERT_SUCCESS(ArgsFormatDesc::GetArgSize(op_desc_, arg_desc, arg_size));
    if ((arg_desc.addr_type == AddrType::INPUT_INSTANCE) || (arg_desc.addr_type == AddrType::OUTPUT_INSTANCE)) {
      uint64_t addr = 0U;
      GE_ASSERT_EOK(memcpy_s(&addr, sizeof(addr), args_data + args_offset, sizeof(addr)));
      const size_t index = static_cast<size_t>(arg_desc.ir_idx);
      if (arg_desc.addr_type == AddrType::INPUT_INSTANCE) {
        input_data_addrs_[index] = addr;
      } else {
        output_data_addrs_[index] = addr;
      }
    }
    args_offset += arg_size;
  }

  if (need_data_dump) {
    GE_ASSERT_SUCCESS(UpdateCustomDumpAddrs(input_data_addrs_, output_data_addrs_));
  }

  if (need_exception_dump) {
    // Exception dump records are maintained independently of normal data dump ops.
    // Keep their IO addresses in sync when annotated args are refreshed.
    exception_dump_io_addrs_.clear();
    exception_dump_io_addrs_.insert(exception_dump_io_addrs_.end(), input_data_addrs_.begin(), input_data_addrs_.end());
    exception_dump_io_addrs_.insert(exception_dump_io_addrs_.end(), output_data_addrs_.begin(),
                                    output_data_addrs_.end());
    davinci_model_->UpdateOpIOAddrs(task_id_, stream_id_, exception_dump_io_addrs_);
  }
  return SUCCESS;
}

Status CustomTaskInfo::InitArgsIoAddrsUpdater() {
  ArgsIoAddrsUpdater::OpInfo op_info{op_desc_->GetName(), op_desc_->GetType()};

  if (args_update_op_ != nullptr) {
    std::vector<uint64_t> logical_addrs;
    logical_addrs.insert(logical_addrs.end(), input_data_addrs_.begin(), input_data_addrs_.end());
    logical_addrs.insert(logical_addrs.end(), output_data_addrs_.begin(), output_data_addrs_.end());

    std::vector<uint64_t> mem_types;
    mem_types.insert(mem_types.end(), input_mem_types_.begin(), input_mem_types_.end());
    mem_types.insert(mem_types.end(), output_mem_types_.begin(), output_mem_types_.end());

    GE_ASSERT_SUCCESS(
        args_io_addrs_updater_.Init(davinci_model_->GetLogicalMemAllocation(), logical_addrs, mem_types, op_info));
    GELOGI("ArgsUpdater operator stored: %s", op_desc_->GetName().c_str());
  }

  return SUCCESS;
}

REGISTER_TASK_INFO(MODEL_TASK_CUSTOM_KERNEL, CustomTaskInfo);
}  // namespace ge
