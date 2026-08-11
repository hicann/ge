/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "engines/custom_engine/custom_ops_kernel_builder.h"
#include <cinttypes>
#include <limits>
#include <map>
#include <memory>
#include <new>
#include <string>
#include <utility>
#include <vector>
#include "framework/common/ge_inner_error_codes.h"
#include "framework/common/debug/ge_log.h"
#include "framework/omg/omg_inner_types.h"
#include "register/ops_kernel_builder_registry.h"
#include "common/ge_common/ge_types.h"
#include "common/checker.h"
#include "graph/utils/tensor_utils.h"
#include "graph/utils/tensor_utils_ex.h"
#include "graph/utils/type_utils.h"
#include "graph/utils/node_utils.h"
#include "common/opskernel/ops_kernel_info_types.h"
#include "exe_graph/lowering/kernel_run_context_builder.h"
#include "exe_graph/runtime/eager_op_execution_context.h"
#include "exe_graph/runtime/gert_mem_allocator.h"
#include "exe_graph/runtime/gert_mem_block.h"
#include "exe_graph/runtime/gert_tensor_data.h"
#include "exe_graph/runtime/annotated_args_context.h"
#include "graph_metadef/exe_graph/runtime/annotated_args_handler.h"
#include "exe_graph/runtime/storage_shape.h"
#include "graph/buffer.h"
#include "graph/custom_op.h"
#include "graph/custom_op/args_refresh.h"
#include "graph/custom_op/cast.h"
#include "graph/custom_op_factory.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/ge_context.h"
#include "graph/op_kernel_bin.h"
#include "graph/utils/args_format_desc_utils.h"
#include "graph/utils/attr_utils.h"
#include "graph/utils/math_util.h"
#include "graph_metadef/graph/debug/ge_util.h"

namespace ge {
namespace {
// A5上没有sqe的限制，此处后续要打开限制
constexpr uint32_t kMaxCustomOpSqeNum = 5;
constexpr uint32_t kKernelArgSlotSize = sizeof(uint64_t);
constexpr const char_t *kCustomOmcAppendWs = "_custom_omc_append_ws";
constexpr const char_t *kAppendWs = "_append_ws";
constexpr const char_t *kAnnotatedArgsTaskPlan = "_custom_annotated_args_task_plan";
constexpr const char_t *kCustomLaunchPrefix = "_custom_launch_";
constexpr const char_t *kDefaultCustomKernelMagic = "RT_DEV_BINARY_MAGIC_ELF_AIVEC";
constexpr size_t kWorkspaceAlignment = 512U;

struct AnnotatedArgsTaskPlan {
  std::vector<domi::TaskDef> task_templates;
};
using AnnotatedArgsTaskPlanPtr = std::shared_ptr<const AnnotatedArgsTaskPlan>;

bool IsMobileSocVersion(const std::string &soc_version) {
  return (soc_version == "KirinX90") || (soc_version == "Kirin9030");
}

void GetStorageShape(const GeTensorDesc &tensor_desc, gert::StorageShape &storage_shape) {
  const auto &storage_dims = tensor_desc.GetShape().GetDims();
  for (const auto &dim : storage_dims) {
    (void)storage_shape.MutableStorageShape().AppendDim(dim);
  }
  const auto &origin_dims = tensor_desc.GetOriginShape().GetDims();
  for (const auto &dim : origin_dims) {
    (void)storage_shape.MutableOriginShape().AppendDim(dim);
  }
}

std::vector<void *> GetHoldersRawPtr(const std::vector<std::unique_ptr<gert::Tensor>> &holders) {
  std::vector<void *> holder_raw_ptr;
  holder_raw_ptr.reserve(holders.size());
  for (const auto &holder : holders) {
    (void)holder_raw_ptr.emplace_back(holder.get());
  }
  return holder_raw_ptr;
}

Status CheckStaticGraph(const Node &node) {
  const auto *const owner_graph = node.GetOwnerComputeGraphBarePtr();
  GE_ASSERT_NOTNULL(owner_graph, "Owner graph of custom op %s(%s) is null.", node.GetNamePtr(), node.GetTypePtr());
  GE_ASSERT_TRUE(!owner_graph->GetGraphUnknownFlag(), "Custom op %s(%s) does not support unknown graph.",
                 node.GetNamePtr(), node.GetTypePtr());
  return SUCCESS;
}

Status BuildTensorHolder(const GeTensorDesc &tensor_desc, void *const address,
                         std::unique_ptr<gert::Tensor> &tensor_holder) {
  gert::StorageShape storage_shape;
  GetStorageShape(tensor_desc, storage_shape);
  tensor_holder = ComGraphMakeUnique<gert::Tensor>(
      storage_shape, gert::StorageFormat{tensor_desc.GetOriginFormat(), tensor_desc.GetFormat(), {}},
      gert::kOnDeviceHbm, tensor_desc.GetDataType(), address);
  GE_ASSERT_NOTNULL(tensor_holder, "Create eager context tensor holder failed.");
  return SUCCESS;
}

void *GetLogicAddr(const uint8_t *const base, const int64_t offset) {
  return ValueToPtr(PtrToValue(base) + static_cast<uint64_t>(offset));
}

Status SizeToInt64(const size_t size, int64_t &value) {
  GE_ASSERT_TRUE(size <= static_cast<size_t>(std::numeric_limits<int64_t>::max()), "Size %zu overflow int64.", size);
  value = static_cast<int64_t>(size);
  return SUCCESS;
}

Status AlignWorkspaceSize(const size_t size, int64_t &aligned_size) {
  size_t aligned = size;
  GE_ASSERT_TRUE(!RoundUpOverflow(size, kWorkspaceAlignment, aligned), "Workspace size %zu align overflow.", size);
  return SizeToInt64(aligned, aligned_size);
}

Status GetInputLogicAddr(const OpDescPtr &op_desc, const RunContext &context, const uint32_t index,
                         const bool is_const_input, const int64_t input_offset, void *&address) {
  if (is_const_input) {
    GE_ASSERT_NOTNULL(context.weightMemBase, "Custom op %s(%s) weightMemBase is null.", op_desc->GetNamePtr(),
                      op_desc->GetTypePtr());
    int64_t weight_offset = 0;
    const auto &input_desc = op_desc->GetInputDesc(index);
    GE_ASSERT_SUCCESS(TensorUtils::GetDataOffset(input_desc, weight_offset),
                      "Get weight offset failed for custom op %s(%s) input[%u].", op_desc->GetNamePtr(),
                      op_desc->GetTypePtr(), index);
    GE_ASSERT_TRUE(weight_offset >= 0, "Tensor offset %ld is invalid.", weight_offset);
    address = GetLogicAddr(context.weightMemBase, weight_offset);
    return SUCCESS;
  }
  GE_ASSERT_NOTNULL(context.dataMemBase, "Custom op %s(%s) dataMemBase is null.", op_desc->GetNamePtr(),
                    op_desc->GetTypePtr());
  GE_ASSERT_TRUE(input_offset >= 0, "Tensor offset %ld is invalid.", input_offset);
  address = GetLogicAddr(context.dataMemBase, input_offset);
  return SUCCESS;
}

Status ConstructInputTensors(const OpDescPtr &op_desc, const RunContext &context,
                             std::vector<std::unique_ptr<gert::Tensor>> &inputs) {
  const auto input_offsets = op_desc->GetInputOffset();
  const auto is_input_const = op_desc->GetIsInputConst();
  size_t valid_input_index = 0U;
  for (uint32_t raw_input_index = 0U; raw_input_index < op_desc->GetAllInputsSize(); ++raw_input_index) {
    const auto &input_desc = op_desc->GetInputDesc(raw_input_index);
    if (input_desc.IsValid() != GRAPH_SUCCESS) {
      GE_ASSERT_TRUE(op_desc->IsOptionalInput(raw_input_index),
                     "Custom op %s(%s) input[%u] desc is invalid but the input is not optional.", op_desc->GetNamePtr(),
                     op_desc->GetTypePtr(), raw_input_index);
      continue;
    }
    const bool is_const_input = (is_input_const.size() > raw_input_index) && is_input_const[raw_input_index];
    GE_ASSERT_TRUE(is_const_input || (input_offsets.size() > valid_input_index),
                   "Custom op %s(%s) input offset size %zu is less than valid input index %zu (raw index %u).",
                   op_desc->GetNamePtr(), op_desc->GetTypePtr(), input_offsets.size(), valid_input_index,
                   raw_input_index);
    const int64_t offset = is_const_input ? 0 : input_offsets[valid_input_index];
    std::unique_ptr<gert::Tensor> tensor_holder;
    void *address = nullptr;
    GE_ASSERT_SUCCESS(GetInputLogicAddr(op_desc, context, raw_input_index, is_const_input, offset, address));
    GE_ASSERT_SUCCESS(BuildTensorHolder(input_desc, address, tensor_holder));
    (void)inputs.emplace_back(std::move(tensor_holder));
    ++valid_input_index;
  }
  GE_ASSERT_TRUE(valid_input_index == op_desc->GetInputsSize(),
                 "Custom op %s(%s) constructed input size %zu does not match valid input desc size %zu.",
                 op_desc->GetNamePtr(), op_desc->GetTypePtr(), valid_input_index, op_desc->GetInputsSize());
  return SUCCESS;
}

Status ConstructOutputTensors(const OpDescPtr &op_desc, const RunContext &context,
                              std::vector<std::unique_ptr<gert::Tensor>> &outputs) {
  GE_ASSERT_NOTNULL(context.dataMemBase, "Custom op %s(%s) dataMemBase is null.", op_desc->GetNamePtr(),
                    op_desc->GetTypePtr());
  const auto output_offsets = op_desc->GetOutputOffset();
  GE_ASSERT_TRUE(output_offsets.size() >= op_desc->GetAllOutputsDescSize(),
                 "Custom op %s(%s) output offset size %zu is less than output size %u.", op_desc->GetNamePtr(),
                 op_desc->GetTypePtr(), output_offsets.size(), op_desc->GetAllOutputsDescSize());
  for (uint32_t i = 0U; i < op_desc->GetAllOutputsDescSize(); ++i) {
    const int64_t offset = output_offsets[i];
    GE_ASSERT_TRUE(offset >= 0, "Tensor offset %ld is invalid.", offset);
    std::unique_ptr<gert::Tensor> tensor_holder;
    GE_ASSERT_SUCCESS(
        BuildTensorHolder(op_desc->GetOutputDesc(i), GetLogicAddr(context.dataMemBase, offset), tensor_holder));
    (void)outputs.emplace_back(std::move(tensor_holder));
  }
  return SUCCESS;
}

class OmcWorkspaceMemBlock : public gert::GertMemBlock {
 public:
  explicit OmcWorkspaceMemBlock(void *addr) : addr_(addr) {}
  ~OmcWorkspaceMemBlock() override = default;

  void Free(int64_t stream_id) override {
    (void)stream_id;
  }

  void *GetAddr() override {
    return addr_;
  }

 private:
  void *addr_;
};

class WorkspaceAllocator : public gert::GertAllocator {
 public:
  WorkspaceAllocator(const OpDescPtr &op_desc, const RunContext &context)
      : GertAllocator(static_cast<int64_t>(op_desc->GetStreamId()), gert::kOnDeviceHbm),
        op_desc_(op_desc),
        context_(context) {}

  ~WorkspaceAllocator() noexcept override {
    for (auto *block : blocks_) {
      delete block;
    }
    blocks_.clear();
  }

  gert::GertMemBlock *Malloc(size_t size) override;
  gert::GertTensorData MallocTensorData(size_t size) override;
  gert::TensorData MallocTensorDataFromL1(size_t size) override;
  void Free(gert::GertMemBlock *block) override;
  ge::graphStatus FreeAt(int64_t stream_id, gert::GertMemBlock *block) override;
  ge::graphStatus ShareFromTensorData(const gert::TensorData &td, gert::GertTensorData &gtd) override;
  int64_t GetStreamNum() override;
  ge::graphStatus SetL1Allocator(ge::Allocator *allocator) override;

  const std::vector<int64_t> &GetProbeWorkspaceSizes() const {
    return probe_workspace_sizes_;
  }

 private:
  gert::GertMemBlock *CreateBlock(void *addr);

  OpDescPtr op_desc_;
  const RunContext &context_;
  int64_t probe_offset_ = 0;
  std::vector<int64_t> probe_workspace_sizes_;
  std::vector<gert::GertMemBlock *> blocks_;
};

void WorkspaceAllocator::Free(gert::GertMemBlock *block) {
  (void)block;
}

gert::GertTensorData WorkspaceAllocator::MallocTensorData(size_t size) {
  (void)size;
  GELOGE(FAILED, "WorkspaceAllocator::MallocTensorData is not supported.");
  return {};
}

gert::TensorData WorkspaceAllocator::MallocTensorDataFromL1(size_t size) {
  (void)size;
  GELOGE(FAILED, "WorkspaceAllocator::MallocTensorDataFromL1 is not supported.");
  return gert::TensorData();
}

ge::graphStatus WorkspaceAllocator::ShareFromTensorData(const gert::TensorData &td, gert::GertTensorData &gtd) {
  (void)td;
  (void)gtd;
  GELOGE(GRAPH_FAILED, "WorkspaceAllocator::ShareFromTensorData is not supported.");
  return GRAPH_FAILED;
}

ge::graphStatus WorkspaceAllocator::SetL1Allocator(ge::Allocator *allocator) {
  (void)allocator;
  GELOGE(GRAPH_FAILED, "WorkspaceAllocator::SetL1Allocator is not supported.");
  return GRAPH_FAILED;
}

ge::graphStatus WorkspaceAllocator::FreeAt(int64_t stream_id, gert::GertMemBlock *block) {
  (void)stream_id;
  (void)block;
  return GRAPH_SUCCESS;
}

int64_t WorkspaceAllocator::GetStreamNum() {
  return 1;
}

gert::GertMemBlock *WorkspaceAllocator::CreateBlock(void *addr) {
  auto *block = new (std::nothrow) OmcWorkspaceMemBlock(addr);
  GE_ASSERT_NOTNULL(block, "Create OMC workspace memory block failed.");
  blocks_.push_back(block);
  return block;
}

gert::GertMemBlock *WorkspaceAllocator::Malloc(size_t size) {
  GE_ASSERT_TRUE(size > 0U, "Custom op %s(%s) MallocWorkSpace size is zero.", op_desc_->GetNamePtr(),
                 op_desc_->GetTypePtr());
  int64_t aligned_size = 0;
  GE_ASSERT_SUCCESS(AlignWorkspaceSize(size, aligned_size));
  probe_workspace_sizes_.emplace_back(aligned_size);
  GE_ASSERT_NOTNULL(context_.dataMemBase, "Custom op %s(%s) dataMemBase is null.", op_desc_->GetNamePtr(),
                    op_desc_->GetTypePtr());
  GE_ASSERT_TRUE(context_.dataMemSize <= static_cast<uint64_t>(std::numeric_limits<int64_t>::max()),
                 "Custom op %s(%s) dataMemSize %" PRIu64 " overflow int64.", op_desc_->GetNamePtr(),
                 op_desc_->GetTypePtr(), context_.dataMemSize);
  GE_ASSERT_TRUE(probe_offset_ <= std::numeric_limits<int64_t>::max() - static_cast<int64_t>(context_.dataMemSize),
                 "Custom op %s(%s) probe workspace logic offset overflow.", op_desc_->GetNamePtr(),
                 op_desc_->GetTypePtr());
  GE_ASSERT_TRUE(aligned_size <= std::numeric_limits<int64_t>::max() - probe_offset_,
                 "Custom op %s(%s) probe workspace offset overflow.", op_desc_->GetNamePtr(), op_desc_->GetTypePtr());
  void *addr = GetLogicAddr(context_.dataMemBase, static_cast<int64_t>(context_.dataMemSize) + probe_offset_);
  probe_offset_ += aligned_size;
  return CreateBlock(addr);
}

Status ResolveTensorPlanAddr(const int32_t instance_index,
                             const std::vector<std::unique_ptr<gert::Tensor>> &tensor_holders, uint64_t &addr) {
  GE_ASSERT_TRUE(instance_index >= 0, "Annotated args tensor instance index %d is invalid.", instance_index);
  const auto index = static_cast<size_t>(instance_index);
  GE_ASSERT_TRUE(index < tensor_holders.size(),
                 "Annotated args tensor instance index %zu exceeds tensor holder count %zu.", index,
                 tensor_holders.size());
  const auto *const tensor = tensor_holders[index].get();
  GE_ASSERT_NOTNULL(tensor);
  addr = PtrToValue(tensor->GetAddr());
  return SUCCESS;
}

Status ResolveWorkspacePlanAddr(const OpDescPtr &op_desc, const RunContext &context, const int32_t workspace_index,
                                uint64_t &addr) {
  GE_ASSERT_TRUE(workspace_index >= 0, "Annotated args workspace index %d is invalid.", workspace_index);
  std::vector<int64_t> append_ws;
  GE_ASSERT_TRUE(AttrUtils::GetListInt(op_desc, kAppendWs, append_ws),
                 "Custom op %s(%s) has no %s when materializing task plan.", op_desc->GetNamePtr(),
                 op_desc->GetTypePtr(), kAppendWs);
  const auto index = static_cast<size_t>(workspace_index);
  GE_ASSERT_TRUE(index < append_ws.size(), "Custom op %s(%s) workspace index %zu exceeds planned count %zu.",
                 op_desc->GetNamePtr(), op_desc->GetTypePtr(), index, append_ws.size());

  const auto workspace_offsets = op_desc->GetWorkspace();
  const auto workspace_bytes = op_desc->GetWorkspaceBytes();
  GE_ASSERT_TRUE(workspace_offsets.size() >= append_ws.size(),
                 "Custom op %s(%s) workspace offset size %zu is less than planned workspace size %zu.",
                 op_desc->GetNamePtr(), op_desc->GetTypePtr(), workspace_offsets.size(), append_ws.size());
  GE_ASSERT_TRUE(workspace_bytes.size() >= append_ws.size(),
                 "Custom op %s(%s) workspace byte size %zu is less than planned workspace size %zu.",
                 op_desc->GetNamePtr(), op_desc->GetTypePtr(), workspace_bytes.size(), append_ws.size());
  const size_t append_start = workspace_offsets.size() - append_ws.size();
  const size_t append_bytes_start = workspace_bytes.size() - append_ws.size();
  GE_ASSERT_TRUE(workspace_bytes[append_bytes_start + index] == append_ws[index],
                 "Custom op %s(%s) workspace[%zu] byte size mismatch, current %ld, planned %ld.", op_desc->GetNamePtr(),
                 op_desc->GetTypePtr(), index, workspace_bytes[append_bytes_start + index], append_ws[index]);
  const int64_t offset = workspace_offsets[append_start + index];
  GE_ASSERT_TRUE(offset >= 0, "Custom op %s(%s) workspace offset %ld is invalid.", op_desc->GetNamePtr(),
                 op_desc->GetTypePtr(), offset);
  GE_ASSERT_NOTNULL(context.dataMemBase, "Custom op %s(%s) dataMemBase is null.", op_desc->GetNamePtr(),
                    op_desc->GetTypePtr());
  GE_ASSERT_TRUE(static_cast<uint64_t>(offset) <= context.dataMemSize,
                 "Custom op %s(%s) workspace offset %ld exceeds dataMemSize %" PRIu64 ".", op_desc->GetNamePtr(),
                 op_desc->GetTypePtr(), offset, context.dataMemSize);
  GE_ASSERT_TRUE(static_cast<uint64_t>(append_ws[index]) <= context.dataMemSize - static_cast<uint64_t>(offset),
                 "Custom op %s(%s) workspace[%zu] range [%ld, %ld) exceeds dataMemSize %" PRIu64 ".",
                 op_desc->GetNamePtr(), op_desc->GetTypePtr(), index, offset, offset + append_ws[index],
                 context.dataMemSize);
  addr = PtrToValue(context.dataMemBase) + static_cast<uint64_t>(offset);
  return SUCCESS;
}

Status PrepareAnnotatedArgsTaskForMaterialization(const OpDescPtr &op_desc, const uint32_t stream_id,
                                                  const domi::TaskDef &task_template, domi::TaskDef &task,
                                                  std::vector<ArgDesc> &arg_descs) {
  GE_ASSERT_TRUE(task_template.type() == static_cast<uint32_t>(ModelTaskType::MODEL_TASK_CUSTOM_KERNEL),
                 "Custom op %s(%s) task plan type %u is invalid.", op_desc->GetNamePtr(), op_desc->GetTypePtr(),
                 task_template.type());
  task = task_template;
  task.set_stream_id(stream_id);
  auto *const kernel = task.mutable_kernel();
  GE_ASSERT_NOTNULL(kernel);
  kernel->mutable_context()->set_op_index(op_desc->GetId());
  GE_ASSERT_GRAPH_SUCCESS(ArgsFormatDescUtils::Parse(kernel->context().args_format(), arg_descs));
  GE_ASSERT_TRUE(arg_descs.size() <= (std::numeric_limits<size_t>::max() / kKernelArgSlotSize));
  GE_ASSERT_TRUE(kernel->args().size() == (arg_descs.size() * kKernelArgSlotSize),
                 "Custom op %s(%s) planned args size %zu does not match arg desc count %zu.", op_desc->GetNamePtr(),
                 op_desc->GetTypePtr(), kernel->args().size(), arg_descs.size());
  return SUCCESS;
}

Status MaterializeAnnotatedArgsTaskPlan(const OpDescPtr &op_desc, const RunContext &context,
                                        const AnnotatedArgsTaskPlan &plan, std::vector<domi::TaskDef> &tasks) {
  std::vector<std::unique_ptr<gert::Tensor>> input_tensor_holders;
  std::vector<std::unique_ptr<gert::Tensor>> output_tensor_holders;
  GE_ASSERT_SUCCESS(ConstructInputTensors(op_desc, context, input_tensor_holders));
  GE_ASSERT_SUCCESS(ConstructOutputTensors(op_desc, context, output_tensor_holders));

  GE_ASSERT_TRUE((op_desc->GetStreamId() >= 0) &&
                     (op_desc->GetStreamId() <= static_cast<int64_t>(std::numeric_limits<uint32_t>::max())),
                 "Custom op stream id %ld is invalid, op_name:%s, op_type:%s", op_desc->GetStreamId(),
                 op_desc->GetNamePtr(), op_desc->GetTypePtr());
  const auto stream_id = static_cast<uint32_t>(op_desc->GetStreamId());
  for (const auto &task_template : plan.task_templates) {
    domi::TaskDef task;
    std::vector<ArgDesc> arg_descs;
    GE_CHK_STATUS_RET_NOLOG(
        PrepareAnnotatedArgsTaskForMaterialization(op_desc, stream_id, task_template, task, arg_descs));
    auto *const kernel = task.mutable_kernel();
    std::string args = kernel->args();
    for (size_t i = 0U; i < arg_descs.size(); ++i) {
      uint64_t addr = 0U;
      switch (arg_descs[i].addr_type) {
        case AddrType::INPUT_INSTANCE:
          GE_ASSERT_SUCCESS(ResolveTensorPlanAddr(arg_descs[i].ir_idx, input_tensor_holders, addr));
          break;
        case AddrType::OUTPUT_INSTANCE:
          GE_ASSERT_SUCCESS(ResolveTensorPlanAddr(arg_descs[i].ir_idx, output_tensor_holders, addr));
          break;
        case AddrType::INPUT:
        case AddrType::OUTPUT:
          GELOGE(INTERNAL_ERROR, "Custom op %s(%s) planned arg[%zu] uses unsupported legacy addr_type %d.",
                 op_desc->GetNamePtr(), op_desc->GetTypePtr(), i, static_cast<int32_t>(arg_descs[i].addr_type));
          return INTERNAL_ERROR;
        case AddrType::WORKSPACE:
          GE_ASSERT_SUCCESS(ResolveWorkspacePlanAddr(op_desc, context, arg_descs[i].ir_idx, addr));
          break;
        case AddrType::CUSTOM_VALUE:
          continue;
        default:
          GELOGE(INTERNAL_ERROR, "Custom op %s(%s) planned arg[%zu] type %d is not materializable.",
                 op_desc->GetNamePtr(), op_desc->GetTypePtr(), i, static_cast<int32_t>(arg_descs[i].addr_type));
          return INTERNAL_ERROR;
      }
      GE_ASSERT_EOK(memcpy_s(&args[i * kKernelArgSlotSize], kKernelArgSlotSize, &addr, sizeof(addr)));
    }
    kernel->set_args(args);
    tasks.emplace_back(std::move(task));
  }
  return SUCCESS;
}

enum class OfflineLaunchGenMode {
  kMobileLegacySingleTask,
  kStandardOmMultiTask,
};

Status CheckAnnotatedArgsTaskPlan(const OpDescPtr &op_desc, const gert::AnnotatedArgsHandler &args_handler,
                                  const uint32_t stream_id) {
  GE_ASSERT_NOTNULL(op_desc);
  const size_t launch_count = args_handler.GetLaunchCount();
  if (launch_count == 0U) {
    GELOGE(INTERNAL_ERROR, "Custom op did not call AddLaunch, op_name:%s, op_type:%s", op_desc->GetNamePtr(),
           op_desc->GetTypePtr());
    return INTERNAL_ERROR;
  }
  for (size_t i = 0U; i < launch_count; ++i) {
    const auto *launch = args_handler.GetLaunch(i);
    GE_ASSERT_NOTNULL(launch);
    if (launch->GetStreamId() != stream_id) {
      GELOGE(INTERNAL_ERROR,
             "Custom op launch[%zu] stream id %u does not match node stream id %u, op_name:%s, op_type:%s", i,
             launch->GetStreamId(), stream_id, op_desc->GetNamePtr(), op_desc->GetTypePtr());
      return INTERNAL_ERROR;
    }
  }
  return SUCCESS;
}

Status CheckAnnotatedArgsTaskPlanForMode(const OpDescPtr &op_desc, const gert::AnnotatedArgsHandler &args_handler,
                                         const uint32_t stream_id, const OfflineLaunchGenMode mode) {
  const auto check_ret = CheckAnnotatedArgsTaskPlan(op_desc, args_handler, stream_id);
  if (check_ret != SUCCESS) {
    return check_ret;
  }
  if ((mode == OfflineLaunchGenMode::kMobileLegacySingleTask) && (args_handler.GetLaunchCount() != 1U)) {
    GELOGE(INTERNAL_ERROR,
           "Custom op AddLaunch count %zu is not supported in mobile OMC scenario, op_name:%s, op_type:%s",
           args_handler.GetLaunchCount(), op_desc->GetNamePtr(), op_desc->GetTypePtr());
    return INTERNAL_ERROR;
  }
  return SUCCESS;
}

struct AnnotatedArgsCapture {
  std::vector<std::unique_ptr<gert::Tensor>> input_tensor_holders;
  std::vector<std::unique_ptr<gert::Tensor>> output_tensor_holders;
  gert::AnnotatedArgsHandler args_handler;
};

Status GetAnnotatedArgsOp(const OpDescPtr &op_desc, AnnotatedArgsOp *&annotated_args_op) {
  auto *const base_custom_op = CustomOpFactory::CreateOrGetCustomOp(AscendString(op_desc->GetTypePtr()));
  GE_ASSERT_NOTNULL(base_custom_op, "Create custom op failed, op_name:%s, op_type:%s", op_desc->GetNamePtr(),
                    op_desc->GetTypePtr());
  annotated_args_op = CustomOpCast<AnnotatedArgsOp>(base_custom_op);
  GE_ASSERT_NOTNULL(annotated_args_op, "Custom op does not implement AnnotatedArgsOp, op_name:%s, op_type:%s",
                    op_desc->GetNamePtr(), op_desc->GetTypePtr());
  return SUCCESS;
}

Status GetCustomOpStreamId(const OpDescPtr &op_desc, uint32_t &stream_id) {
  GE_ASSERT_TRUE((op_desc->GetStreamId() >= 0) &&
                     (op_desc->GetStreamId() <= static_cast<int64_t>(std::numeric_limits<uint32_t>::max())),
                 "Custom op stream id %ld is invalid, op_name:%s, op_type:%s", op_desc->GetStreamId(),
                 op_desc->GetNamePtr(), op_desc->GetTypePtr());
  stream_id = static_cast<uint32_t>(op_desc->GetStreamId());
  return SUCCESS;
}

Status SetArgsOffset(const size_t slot_count, domi::KernelContext *const kernel_context) {
  GE_ASSERT_NOTNULL(kernel_context);
  GE_ASSERT_TRUE(slot_count <= ((static_cast<size_t>(std::numeric_limits<uint16_t>::max()) / kKernelArgSlotSize) + 1U),
                 "Custom kernel args count %zu exceeds max args offset.", slot_count);
  std::vector<uint16_t> args_offset;
  args_offset.reserve(slot_count);
  for (size_t i = 0U; i < slot_count; ++i) {
    const size_t offset = i * kKernelArgSlotSize;
    GE_ASSERT_TRUE(offset <= std::numeric_limits<uint16_t>::max(), "Custom kernel args offset %zu overflow.", offset);
    args_offset.emplace_back(static_cast<uint16_t>(offset));
  }
  kernel_context->set_args_offset(args_offset.data(), args_offset.size() * sizeof(uint16_t));
  return SUCCESS;
}

std::string MakeCustomLaunchPrefix(const size_t launch_index) {
  return std::string(kCustomLaunchPrefix) + std::to_string(launch_index) + "_";
}

Status SetKernelBinAttrs(const char *const kernel_name, const uint8_t *const kernel_bin_data,
                         const size_t kernel_bin_size, const OpDescPtr &op_desc, const std::string &prefix) {
  std::vector<char> kernel_bin(kernel_bin_data, kernel_bin_data + kernel_bin_size);
  auto tbe_kernel = ge::ComGraphMakeShared<OpKernelBin>(kernel_name, std::move(kernel_bin));
  GE_ASSERT_NOTNULL(tbe_kernel, "Create OpKernelBin failed.");
  GE_ASSERT_TRUE(op_desc->SetExtAttr(prefix + OP_EXTATTR_NAME_TBE_KERNEL, tbe_kernel),
                 "Set ext attr %s failed for custom op %s(%s).", (prefix + OP_EXTATTR_NAME_TBE_KERNEL).c_str(),
                 op_desc->GetNamePtr(), op_desc->GetTypePtr());
  GE_ASSERT_TRUE(AttrUtils::SetStr(op_desc, prefix + ATTR_NAME_TBE_KERNEL_NAME, kernel_name),
                 "Set %s failed for custom op %s(%s).", (prefix + ATTR_NAME_TBE_KERNEL_NAME).c_str(),
                 op_desc->GetNamePtr(), op_desc->GetTypePtr());
  GE_ASSERT_TRUE(AttrUtils::SetBytes(op_desc, prefix + ATTR_NAME_TBE_KERNEL_BUFFER,
                                     Buffer::CopyFrom(kernel_bin_data, kernel_bin_size)),
                 "Set %s failed for custom op %s(%s).", (prefix + ATTR_NAME_TBE_KERNEL_BUFFER).c_str(),
                 op_desc->GetNamePtr(), op_desc->GetTypePtr());
  std::string binary_magic;
  if (!AttrUtils::GetStr(op_desc, TVM_ATTR_NAME_MAGIC, binary_magic) || binary_magic.empty()) {
    binary_magic = kDefaultCustomKernelMagic;
    GE_ASSERT_TRUE(AttrUtils::SetStr(op_desc, TVM_ATTR_NAME_MAGIC, binary_magic), "Set %s failed for custom op %s(%s).",
                   TVM_ATTR_NAME_MAGIC.c_str(), op_desc->GetNamePtr(), op_desc->GetTypePtr());
  }
  GE_ASSERT_TRUE(AttrUtils::SetStr(op_desc, prefix + TVM_ATTR_NAME_MAGIC, binary_magic),
                 "Set %s failed for custom op %s(%s).", (prefix + TVM_ATTR_NAME_MAGIC).c_str(), op_desc->GetNamePtr(),
                 op_desc->GetTypePtr());
  return SUCCESS;
}

Status FillKernelTask(const Node &node, const gert::AnnotatedArgsHandler::LaunchRecord &launch_task,
                      const std::string &kernel_attr_prefix, domi::TaskDef &task_def) {
  const auto op_desc = node.GetOpDesc();
  GE_ASSERT_NOTNULL(op_desc);
  const auto *args_data = launch_task.GetArgsData();
  const auto args_size = launch_task.GetArgsSize();
  const auto *arg_descs = launch_task.GetArgDescs();
  const size_t slot_count = launch_task.GetArgDescCount();
  const auto *kernel_name = launch_task.GetKernelName();
  const auto *kernel_bin_data = launch_task.GetKernelBinData();
  const auto kernel_bin_size = launch_task.GetKernelBinSize();
  GE_ASSERT_TRUE(slot_count <= (std::numeric_limits<size_t>::max() / kKernelArgSlotSize),
                 "Custom op %s(%s) args count %zu overflow.", op_desc->GetNamePtr(), op_desc->GetTypePtr(), slot_count);
  GE_ASSERT_TRUE((kernel_name != nullptr) && (kernel_name[0] != '\0'), "Custom op %s(%s) kernel name is empty.",
                 op_desc->GetNamePtr(), op_desc->GetTypePtr());
  GE_ASSERT_TRUE((kernel_bin_data != nullptr) && (kernel_bin_size > 0U), "Custom op %s(%s) kernel bin is empty.",
                 op_desc->GetNamePtr(), op_desc->GetTypePtr());
  GE_ASSERT_TRUE(launch_task.GetBlockDim() > 0U, "Custom op %s(%s) block dim is zero.", op_desc->GetNamePtr(),
                 op_desc->GetTypePtr());
  GE_ASSERT_TRUE((args_data != nullptr) && (arg_descs != nullptr) && (slot_count > 0U) &&
                     (args_size == (slot_count * kKernelArgSlotSize)),
                 "Custom op %s(%s) args size %zu does not match arg desc size %zu.", op_desc->GetNamePtr(),
                 op_desc->GetTypePtr(), args_size, slot_count);
  GE_ASSERT_TRUE(args_size <= std::numeric_limits<uint32_t>::max(), "Custom op %s(%s) args size %zu overflow.",
                 op_desc->GetNamePtr(), op_desc->GetTypePtr(), args_size);
  GE_ASSERT_TRUE(slot_count <= std::numeric_limits<uint32_t>::max(), "Custom op %s(%s) args count %zu overflow.",
                 op_desc->GetNamePtr(), op_desc->GetTypePtr(), slot_count);

  task_def.set_stream_id(launch_task.GetStreamId());
  task_def.set_type(static_cast<uint32_t>(ModelTaskType::MODEL_TASK_CUSTOM_KERNEL));
  task_def.set_sqe_num(kMaxCustomOpSqeNum);
  domi::KernelDef *kernel_def = task_def.mutable_kernel();
  kernel_def->set_stub_func(kernel_name);
  kernel_def->set_kernel_name(kernel_name);
  kernel_def->set_block_dim(launch_task.GetBlockDim());
  kernel_def->set_args(args_data, args_size);
  kernel_def->set_args_size(static_cast<uint32_t>(args_size));

  auto *kernel_context = kernel_def->mutable_context();
  kernel_context->set_op_index(op_desc->GetId());
  const std::vector<ArgDesc> arg_desc_vec(arg_descs, arg_descs + slot_count);
  kernel_context->set_args_format(ArgsFormatDescUtils::Serialize(arg_desc_vec));
  kernel_context->set_args_count(static_cast<uint32_t>(slot_count));
  GE_ASSERT_SUCCESS(SetArgsOffset(slot_count, kernel_context));
  GE_ASSERT_SUCCESS(SetKernelBinAttrs(kernel_name, kernel_bin_data, kernel_bin_size, op_desc, kernel_attr_prefix));
  return SUCCESS;
}

Status ValidateKernelBinNames(const OpDescPtr &op_desc, const gert::AnnotatedArgsHandler &args_handler) {
  const size_t launch_count = args_handler.GetLaunchCount();
  for (size_t i = 0U; i < launch_count; ++i) {
    const auto *const launch = args_handler.GetLaunch(i);
    GE_ASSERT_NOTNULL(launch);
    const auto *const name = launch->GetKernelName();
    const auto *const data = launch->GetKernelBinData();
    const auto size = launch->GetKernelBinSize();
    GE_ASSERT_TRUE((name != nullptr) && (name[0] != '\0'),
                   "Custom op launch[%zu] kernel name is empty, op_name:%s, op_type:%s", i, op_desc->GetNamePtr(),
                   op_desc->GetTypePtr());
    GE_ASSERT_TRUE((data != nullptr) && (size > 0U),
                   "Custom op launch[%zu] kernel bin is empty, op_name:%s, op_type:%s", i, op_desc->GetNamePtr(),
                   op_desc->GetTypePtr());
  }
  return SUCCESS;
}

Status DeclareLaunchArgsTaskPlan(const OpDescPtr &op_desc, AnnotatedArgsOp &annotated_args_op,
                                 WorkspaceAllocator &workspace_allocator, const uint32_t stream_id,
                                 const OfflineLaunchGenMode mode, AnnotatedArgsCapture &capture) {
  std::vector<void *> inputs = GetHoldersRawPtr(capture.input_tensor_holders);
  inputs.push_back(&workspace_allocator);

  auto workspace_mems = ge::ComGraphMakeShared<std::vector<gert::GertMemBlock *>>();
  GE_ASSERT_NOTNULL(workspace_mems, "Create workspace memory list failed.");
  std::vector<void *> outputs = GetHoldersRawPtr(capture.output_tensor_holders);
  outputs.push_back(workspace_mems.get());
  outputs.push_back(static_cast<gert::ArgsHandler *>(&capture.args_handler));

  auto annotated_args_context_holder =
      gert::KernelRunContextBuilder().Inputs(std::move(inputs)).Outputs(std::move(outputs)).Build(op_desc);
  auto *const kernel_context = annotated_args_context_holder.GetKernelContext();
  GE_ASSERT_NOTNULL(kernel_context, "Create annotated args launch kernel context failed, op_name:%s, op_type:%s",
                    op_desc->GetNamePtr(), op_desc->GetTypePtr());
  auto *const annotated_args_context = reinterpret_cast<gert::AnnotatedArgsContext *>(kernel_context);
  GE_ASSERT_NOTNULL(annotated_args_context, "Create annotated args launch context failed, op_name:%s, op_type:%s",
                    op_desc->GetNamePtr(), op_desc->GetTypePtr());
  const size_t compute_input_num = annotated_args_context->GetComputeNodeInputNum();
  GE_ASSERT_TRUE(
      kernel_context->GetInputNum() == (compute_input_num + 1U),
      "Custom op %s(%s) annotated args context input size %zu does not equal compute input size %zu plus one "
      "workspace allocator.",
      op_desc->GetNamePtr(), op_desc->GetTypePtr(), kernel_context->GetInputNum(), compute_input_num);
  auto *const context_allocator = kernel_context->GetInputValue<gert::GertAllocator *>(compute_input_num);
  GE_ASSERT_TRUE(context_allocator == &workspace_allocator,
                 "Custom op %s(%s) annotated args context workspace allocator does not match the expected allocator.",
                 op_desc->GetNamePtr(), op_desc->GetTypePtr());
  const auto declare_ret = annotated_args_op.DeclareLaunchArgs(*annotated_args_context);
  if (declare_ret != GRAPH_SUCCESS) {
    GELOGE(GRAPH_FAILED, "Declare annotated args launch failed, op_name:%s, op_type:%s", op_desc->GetNamePtr(),
           op_desc->GetTypePtr());
    return GRAPH_FAILED;
  }
  return CheckAnnotatedArgsTaskPlanForMode(op_desc, capture.args_handler, stream_id, mode);
}

Status UpdateAnnotatedArgsWorkspaceAttrs(const OpDescPtr &op_desc, const WorkspaceAllocator &workspace_allocator) {
  const auto &append_ws = workspace_allocator.GetProbeWorkspaceSizes();
  if (append_ws.empty()) {
    return SUCCESS;
  }
  GE_ASSERT_TRUE(AttrUtils::SetListInt(op_desc, kAppendWs, append_ws), "Set %s failed for custom op %s(%s).", kAppendWs,
                 op_desc->GetNamePtr(), op_desc->GetTypePtr());
  GE_ASSERT_TRUE(AttrUtils::SetBool(op_desc, kCustomOmcAppendWs, true), "Set %s failed for custom op %s(%s).",
                 kCustomOmcAppendWs, op_desc->GetNamePtr(), op_desc->GetTypePtr());
  return SUCCESS;
}

Status CacheAnnotatedArgsTaskPlan(const OpDescPtr &op_desc, const std::vector<domi::TaskDef> &tasks,
                                  const size_t task_start) {
  GE_ASSERT_TRUE(task_start < tasks.size(), "Custom op %s(%s) generated no task for annotated args plan.",
                 op_desc->GetNamePtr(), op_desc->GetTypePtr());
  auto mutable_plan = ge::ComGraphMakeShared<AnnotatedArgsTaskPlan>();
  GE_ASSERT_NOTNULL(mutable_plan);
  mutable_plan->task_templates.assign(tasks.cbegin() + static_cast<std::ptrdiff_t>(task_start), tasks.cend());
  AnnotatedArgsTaskPlanPtr plan = std::move(mutable_plan);
  GE_ASSERT_TRUE(op_desc->SetExtAttr(kAnnotatedArgsTaskPlan, plan),
                 "Cache annotated args task plan failed for custom op %s(%s).", op_desc->GetNamePtr(),
                 op_desc->GetTypePtr());
  return SUCCESS;
}

Status GenerateMobileAnnotatedArgsTask(const Node &node, const OpDescPtr &op_desc,
                                       const gert::AnnotatedArgsHandler &args_handler, const size_t task_start,
                                       std::vector<domi::TaskDef> &tasks) {
  domi::TaskDef task_def = {};
  const auto *launch = args_handler.GetLaunch(0U);
  GE_ASSERT_NOTNULL(launch);
  GE_ASSERT_SUCCESS(FillKernelTask(node, *launch, "", task_def));
  tasks.emplace_back(std::move(task_def));
  GE_ASSERT_SUCCESS(CacheAnnotatedArgsTaskPlan(op_desc, tasks, task_start));
  return SUCCESS;
}

Status GenerateAnnotatedArgsTask(const Node &node, const OpDescPtr &op_desc, RunContext &context,
                                 const OfflineLaunchGenMode mode, std::vector<domi::TaskDef> &tasks) {
  GE_ASSERT_SUCCESS(CheckStaticGraph(node));
  const auto plan = op_desc->TryGetExtAttr(kAnnotatedArgsTaskPlan, AnnotatedArgsTaskPlanPtr());
  if (plan != nullptr) {
    if ((mode == OfflineLaunchGenMode::kMobileLegacySingleTask) && (plan->task_templates.size() != 1U)) {
      GELOGE(INTERNAL_ERROR,
             "Custom op planned task count %zu is not supported in mobile OMC scenario, op_name:%s, op_type:%s",
             plan->task_templates.size(), op_desc->GetNamePtr(), op_desc->GetTypePtr());
      return INTERNAL_ERROR;
    }
    return MaterializeAnnotatedArgsTaskPlan(op_desc, context, *plan, tasks);
  }

  AnnotatedArgsOp *annotated_args_op = nullptr;
  GE_ASSERT_SUCCESS(GetAnnotatedArgsOp(op_desc, annotated_args_op));

  const size_t task_start = tasks.size();
  AnnotatedArgsCapture capture;
  GE_ASSERT_SUCCESS(ConstructInputTensors(op_desc, context, capture.input_tensor_holders));
  GE_ASSERT_SUCCESS(ConstructOutputTensors(op_desc, context, capture.output_tensor_holders));
  WorkspaceAllocator workspace_allocator(op_desc, context);
  uint32_t stream_id = 0U;
  GE_ASSERT_SUCCESS(GetCustomOpStreamId(op_desc, stream_id));
  const auto launch_task_plan_ret =
      DeclareLaunchArgsTaskPlan(op_desc, *annotated_args_op, workspace_allocator, stream_id, mode, capture);
  if (launch_task_plan_ret != SUCCESS) {
    return launch_task_plan_ret;
  }
  GE_ASSERT_SUCCESS(UpdateAnnotatedArgsWorkspaceAttrs(op_desc, workspace_allocator));

  GE_ASSERT_SUCCESS(ValidateKernelBinNames(op_desc, capture.args_handler));
  if (mode == OfflineLaunchGenMode::kMobileLegacySingleTask) {
    return GenerateMobileAnnotatedArgsTask(node, op_desc, capture.args_handler, task_start, tasks);
  }

  std::vector<std::string> prefixes;
  prefixes.reserve(capture.args_handler.GetLaunchCount());
  for (size_t i = 0U; i < capture.args_handler.GetLaunchCount(); ++i) {
    const std::string prefix = MakeCustomLaunchPrefix(i);
    prefixes.emplace_back(prefix);
    domi::TaskDef task_def = {};
    const auto *launch = capture.args_handler.GetLaunch(i);
    GE_ASSERT_NOTNULL(launch);
    GE_ASSERT_SUCCESS(FillKernelTask(node, *launch, prefix, task_def));
    tasks.emplace_back(std::move(task_def));
  }
  GE_ASSERT_TRUE(AttrUtils::SetListStr(op_desc, ATTR_NAME_KERNEL_NAMES_PREFIX, prefixes),
                 "Set %s failed for custom op %s(%s).", ATTR_NAME_KERNEL_NAMES_PREFIX.c_str(), op_desc->GetNamePtr(),
                 op_desc->GetTypePtr());
  GE_ASSERT_SUCCESS(CacheAnnotatedArgsTaskPlan(op_desc, tasks, task_start));
  return SUCCESS;
}

Status FillBasicCustomKernelTask(const Node &node, domi::TaskDef &task_def) {
  const auto op_desc = node.GetOpDesc();
  GE_ASSERT_NOTNULL(op_desc);
  task_def.set_stream_id(op_desc->GetStreamId());
  task_def.set_type(static_cast<uint32_t>(ModelTaskType::MODEL_TASK_CUSTOM_KERNEL));
  task_def.set_sqe_num(kMaxCustomOpSqeNum);
  domi::KernelDef *kernel_def = task_def.mutable_kernel();
  kernel_def->mutable_context()->set_op_index(op_desc->GetId());
  return SUCCESS;
}

Status GenerateBasicCustomKernelTask(const Node &node, const OpDescPtr &op_desc, const std::string &soc_version,
                                     const CustomTaskArgsMode args_mode, std::vector<domi::TaskDef> &tasks) {
  GELOGI("Custom op %s(%s) generate basic custom kernel task, soc_version: %s", op_desc->GetNamePtr(),
         op_desc->GetTypePtr(), soc_version.c_str());
  domi::TaskDef task_def = {};
  GE_ASSERT_SUCCESS(FillBasicCustomKernelTask(node, task_def));
  GE_ASSERT_TRUE(AttrUtils::SetInt(op_desc, ATTR_NAME_CUSTOM_TASK_ARGS_MODE, static_cast<int64_t>(args_mode)),
                 "Set %s failed for custom op %s(%s).", ATTR_NAME_CUSTOM_TASK_ARGS_MODE.c_str(), op_desc->GetNamePtr(),
                 op_desc->GetTypePtr());
  tasks.push_back(task_def);
  return SUCCESS;
}
}  // namespace
namespace custom {
REGISTER_OPS_KERNEL_BUILDER(kCustomOpKernelLibName, CustomOpsKernelBuilder);

CustomOpsKernelBuilder::~CustomOpsKernelBuilder() {
  GELOGI("CustomOpsKernelBuilder destroyed");
}

Status CustomOpsKernelBuilder::Initialize(const std::map<std::string, std::string> &options) {
  (void)options;
  return SUCCESS;
}

Status CustomOpsKernelBuilder::Finalize() {
  return SUCCESS;
}

Status CustomOpsKernelBuilder::CalcOpRunningParam(Node &node) {
  auto op_desc = node.GetOpDesc();
  GE_ASSERT_NOTNULL(op_desc);
  const auto args_refresh_strategy = CustomOpFactory::GetArgsRefreshStrategy(AscendString(op_desc->GetTypePtr()));
  if (args_refresh_strategy == ArgsRefreshStrategy::kAnnotatedArgs) {
    (void)op_desc->DelExtAttr(kAnnotatedArgsTaskPlan);
    (void)op_desc->DelAttr(kAppendWs);
    (void)op_desc->DelAttr(kCustomOmcAppendWs);
  }
  for (size_t i = 0; i < op_desc->GetOutputsSize(); i++) {
    if (op_desc->GetOutputDesc(i).GetShape().IsUnknownShape()) {
      GELOGI("Node[%s] output[%zu] is unknown shape.", op_desc->GetName().c_str(), i);
      continue;
    }
    int64_t tensor_size = 0;
    GE_ASSERT_SUCCESS(
        TensorUtilsEx::GetTensorMemorySizeInBytesWithAutoPadding(*op_desc->MutableOutputDesc(i), tensor_size));
    TensorUtils::SetSize(*op_desc->MutableOutputDesc(i), tensor_size);
  }
  (void)node;
  return SUCCESS;
}

Status CustomOpsKernelBuilder::GenerateTask(const Node &node, RunContext &context, std::vector<domi::TaskDef> &tasks) {
  const auto op_desc = node.GetOpDesc();
  GE_ASSERT_NOTNULL(op_desc);

  std::string soc_version;
  (void)ge::GetContext().GetOption(ge::SOC_VERSION, soc_version);

  const auto *const owner_graph = node.GetOwnerComputeGraphBarePtr();
  GE_ASSERT_NOTNULL(owner_graph, "Owner graph of custom op %s(%s) is null.", node.GetNamePtr(), node.GetTypePtr());
  ArgsRefreshStrategy args_refresh_strategy = ArgsRefreshStrategy::kNone;
  if (!owner_graph->GetGraphUnknownFlag()) {
    const bool is_mobile_omc = IsMobileSocVersion(soc_version);
    args_refresh_strategy = CustomOpFactory::GetArgsRefreshStrategy(AscendString(op_desc->GetTypePtr()));
    if (args_refresh_strategy == ArgsRefreshStrategy::kAnnotatedArgs) {
      // 端侧场景下的自定义算子只支持单任务的 AnnotatedArgsOp 生成方式，非端侧场景下的自定义算子支持多任务的
      // AnnotatedArgsOp 生成方式
      const auto mode =
          is_mobile_omc ? OfflineLaunchGenMode::kMobileLegacySingleTask : OfflineLaunchGenMode::kStandardOmMultiTask;
      const auto status = GenerateAnnotatedArgsTask(node, op_desc, context, mode, tasks);
      if (status != SUCCESS) {
        return status;
      }
      GE_ASSERT_TRUE(AttrUtils::SetInt(op_desc, ATTR_NAME_CUSTOM_TASK_ARGS_MODE,
                                       static_cast<int64_t>(CustomTaskArgsMode::kAnnotatedArgs)),
                     "Set %s failed for custom op %s(%s).", ATTR_NAME_CUSTOM_TASK_ARGS_MODE.c_str(),
                     op_desc->GetNamePtr(), op_desc->GetTypePtr());
      return SUCCESS;
    }
    // 端侧场景下的自定义算子必须实现 AnnotatedArgsOp 接口，其他生成方式不支持
    if (is_mobile_omc) {
      GELOGE(INTERNAL_ERROR, "Custom op %s(%s) does not implement AnnotatedArgsOp, soc_version: %s",
             op_desc->GetNamePtr(), op_desc->GetTypePtr(), soc_version.c_str());
      return INTERNAL_ERROR;
    }
  }
  // 全部动态图场景的自定义算子或者在非端侧场景中没有实现 AnnotatedArgsOp 的自定义算子，使用最基础的 CustomKernelTask
  // 生成方式
  const auto basic_args_mode = (args_refresh_strategy == ArgsRefreshStrategy::kUpdateCallback)
                                   ? CustomTaskArgsMode::kUpdateCallback
                                   : CustomTaskArgsMode::kNone;
  return GenerateBasicCustomKernelTask(node, op_desc, soc_version, basic_args_mode, tasks);
}
}  // namespace custom
}  // namespace ge
