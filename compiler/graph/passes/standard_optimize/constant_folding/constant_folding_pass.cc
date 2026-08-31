/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph/passes/standard_optimize/constant_folding/constant_folding_pass.h"

#include <memory>
#include <new>
#include <utility>
#include <vector>
#include "graph_metadef/common/ge_common/util.h"
#include "rt_external_mem.h"
#include "common/memory/tensor_trans_utils.h"
#include "exe_graph/runtime/host_cpu_op_execution_context.h"
#include "exe_graph/runtime/gert_mem_allocator.h"
#include "exe_graph/runtime/runtime_tensor.h"
#include "exe_graph/lowering/kernel_run_context_builder.h"
#include "graph/custom_op.h"
#include "graph/custom_op/cast.h"
#include "graph/custom_op_factory.h"
#include "graph/ge_tensor.h"
#include "graph/op_desc.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/constant_utils.h"
#include "host_cpu_engine/host_cpu_engine.h"
#include "api/gelib/gelib.h"
#include "register/op_kernel_registry.h"
#include "graph/ge_context.h"

namespace ge {
namespace {
const int64_t kStartCallNum = 1;
const int64_t kShapeCalNum = 8;
const char *const kKernelLibName = "aicpu_ascend_kernel";
const char *const kOpsFlagClose = "0";
const char *const kPassName = "ConstantFoldingPass";

class HostCpuConstFoldingMemAllocator final : public ge::Allocator {
 public:
  ge::MemBlock *Malloc(size_t size) override {
    auto *buffer = new (std::nothrow) uint8_t[size];
    if (buffer == nullptr) {
      return nullptr;
    }
    auto *block = new (std::nothrow) ge::MemBlock(*this, buffer, size);
    if (block == nullptr) {
      delete[] buffer;
      return nullptr;
    }
    return block;
  }

  void Free(ge::MemBlock *block) override {
    if (block != nullptr) {
      delete[] static_cast<uint8_t *>(block->GetAddr());
      delete block;
    }
  }
};

HostCpuConstFoldingMemAllocator &GetHostCpuConstFoldingMemAllocator() {
  static HostCpuConstFoldingMemAllocator allocator;
  return allocator;
}

ge::graphStatus HostCpuMemBlockManager(void *block, gert::TensorOperateType operate_type, void **out) {
  GE_ASSERT_NOTNULL(block);
  auto *mem_block = static_cast<ge::MemBlock *>(block);
  GE_ASSERT((operate_type == gert::kGetTensorAddress || operate_type == gert::kFreeTensor ||
             operate_type == gert::kPlusShareCount),
            "Unexpected operate type %d", static_cast<int32_t>(operate_type));
  if (operate_type == gert::kGetTensorAddress) {
    GE_ASSERT_NOTNULL(out);
    *out = mem_block->GetAddr();
  }
  if (operate_type == gert::kPlusShareCount) {
    mem_block->AddCount();
  }
  if (operate_type == gert::kFreeTensor) {
    mem_block->Free();
  }
  return ge::GRAPH_SUCCESS;
}

gert::TensorData MakeHostCpuConstFoldingTensorData(size_t size, gert::TensorPlacement placement) {
  auto *mem_block = GetHostCpuConstFoldingMemAllocator().Malloc(size);
  if ((mem_block == nullptr) || (mem_block->GetAddr() == nullptr)) {
    if (mem_block != nullptr) {
      GetHostCpuConstFoldingMemAllocator().Free(mem_block);
    }
    return gert::TensorData();
  }
  return gert::TensorData(mem_block, HostCpuMemBlockManager, size, placement);
}

class HostCpuConstFoldingMemGertAllocator final : public gert::GertAllocator {
 public:
  HostCpuConstFoldingMemGertAllocator() : gert::GertAllocator(0, gert::kOnHost) {}
  ~HostCpuConstFoldingMemGertAllocator() override = default;

  gert::GertMemBlock *Malloc(size_t size) override {
    (void)size;
    return nullptr;
  }

  gert::GertTensorData MallocTensorData(size_t size) override {
    const auto tensor_data = MakeHostCpuConstFoldingTensorData(size, GetPlacement());
    if (tensor_data.GetAddr() == nullptr) {
      return {};
    }
    gert::GertTensorData gtd;
    if (gtd.MutableTensorData().ShareFrom(tensor_data) != ge::GRAPH_SUCCESS) {
      return {};
    }
    return gtd;
  }

  gert::TensorData MallocTensorDataFromL1(size_t size) override {
    return MakeHostCpuConstFoldingTensorData(size, GetPlacement());
  }

  void Free(gert::GertMemBlock *block) override {
    (void)block;
  }

  ge::graphStatus FreeAt(int64_t stream_id, gert::GertMemBlock *block) override {
    (void)stream_id;
    (void)block;
    return ge::GRAPH_SUCCESS;
  }

  ge::graphStatus ShareFromTensorData(const gert::TensorData &td, gert::GertTensorData &gtd) override {
    return gtd.MutableTensorData().ShareFrom(td);
  }

  int64_t GetStreamNum() override {
    return 1;
  }

  ge::graphStatus SetL1Allocator(ge::Allocator *allocator) override {
    (void)allocator;
    return ge::GRAPH_SUCCESS;
  }
};

Status BuildHostCpuOpContext(const NodePtr &node, const std::vector<ConstGeTensorPtr> &inputs,
                             std::vector<gert::Tensor> &input_tensors, std::vector<gert::Tensor> &output_tensors,
                             HostCpuConstFoldingMemGertAllocator &allocator,
                             gert::KernelContextHolder &context_holder) {
  const auto op_desc = node->GetOpDesc();
  GE_ASSERT_NOTNULL(op_desc);
  const size_t input_num = inputs.size();
  const size_t output_num = op_desc->GetOutputsSize();
  input_tensors.resize(input_num);
  output_tensors.resize(output_num);

  for (size_t i = 0U; i < input_num; ++i) {
    GE_ASSERT_SUCCESS(TensorTransUtils::GeTensor2GertTensor(*inputs[i], input_tensors[i]));
  }

  for (size_t i = 0U; i < output_num; ++i) {
    GE_ASSERT_SUCCESS(TensorTransUtils::GeTensor2GertTensor(GeTensor(op_desc->GetOutputDesc(i)), output_tensors[i]));
  }

  std::vector<void *> input_ptrs;
  input_ptrs.reserve(input_num + 1U);
  for (auto &input_tensor : input_tensors) {
    input_ptrs.emplace_back(&input_tensor);
  }
  input_ptrs.emplace_back(&allocator);

  std::vector<void *> output_ptrs;
  output_ptrs.reserve(output_num);
  for (auto &output_tensor : output_tensors) {
    output_ptrs.emplace_back(&output_tensor);
  }

  context_holder =
      gert::KernelRunContextBuilder().Inputs(std::move(input_ptrs)).Outputs(std::move(output_ptrs)).Build(op_desc);
  if (context_holder.GetKernelContext() == nullptr) {
    GELOGE(FAILED, "Build HostCpu op context failed for node %s.", node->GetName().c_str());
    return FAILED;
  }
  return SUCCESS;
}
}  // namespace

bool ConstantFoldingPass::NeedIgnorePass(const NodePtr &node) {
  if (folding_pass::IsNoNeedConstantFolding(node)) {
    return true;
  }
  if (AreAllOutputsEmptyShape(node->GetOpDesc())) {
    GELOGI("Current node %s is potential empty const, ignore pass.", node->GetName().c_str());
    return true;
  }
  return false;
}

bool ConstantFoldingPass::NeedFold() const {
  return need_fold_;
}

Status ConstantFoldingPass::ComputePotentialWeight(NodePtr &node, std::vector<GeTensorPtr> &outputs) {
  need_fold_ = true;
  GELOGD("Begin to perform constant folding computation on node %s.", node->GetName().c_str());
  const OpDescPtr &node_desc = node->GetOpDesc();
  auto input_nodes_2_out_anchors = OpDescUtils::GetConstInputNodeAndAnchor(*node);
  if (input_nodes_2_out_anchors.empty() || input_nodes_2_out_anchors.size() != node_desc->GetInputsSize()) {
    GELOGD("Node:%s, const input nodes size is %zu, and nodeDesc inputsSize is %zu.", node->GetName().c_str(),
           input_nodes_2_out_anchors.size(), node_desc->GetInputsSize());
    if (ConstantUtils::IsPotentialConst(node_desc)) {
      need_fold_ = false;
    }
    return NOT_CHANGED;
  }
  auto inputs = OpDescUtils::GetWeightsFromNodes(input_nodes_2_out_anchors);
  if (inputs.size() != input_nodes_2_out_anchors.size()) {
    GELOGW("Get weights from const_inputs size %zu, not match with inputs size %zu. Ignore pass.", inputs.size(),
           input_nodes_2_out_anchors.size());
    return NOT_CHANGED;
  }
  std::string memory_optimization_policy;
  (void)ge::GetContext().GetOption(MEMORY_OPTIMIZATION_POLICY, memory_optimization_policy);
  // check input nodes has potential const
  for (const auto &node_2_anchor : input_nodes_2_out_anchors) {
    if (ConstantUtils::IsPotentialConst(node_2_anchor.first->GetOpDesc())) {
      need_fold_ = false;
      break;
    }
    if (memory_optimization_policy == kMemoryPriority) {
      // in case input const node has multiple connect edge, do not fold when use memory priority policy.
      const int64_t shape_size = node_2_anchor.first->GetOpDesc()->GetOutputDesc(0).GetShape().GetShapeSize();
      if ((shape_size > kShapeCalNum) && (node_2_anchor.second->GetPeerInDataNodesSize() > 1U)) {
        GELOGI("In MemoryPriority mode, ignore constant folding for node:%s when const node has multiple out edges.",
               node->GetName().c_str());
        return NOT_CHANGED;
      }
    }
  }
  // Try to run kernel on host cpu
  uint64_t start_time = GetCurrentTimestamp();
  Status compute_ret = ComputeWithHostCpuCustomOp(node, inputs, outputs);
  if (compute_ret == SUCCESS) {
    CollectCostTimeOfOpConstantFolding(node, start_time);
  } else {
    // If host custom op computation is not possible, try running the HostCpu kernel.
    GELOGD("Try to compute weight of %s with HostCpu kernel.", node->GetName().c_str());
    compute_ret = ComputeWithHostCpuKernel(node, inputs, outputs);
    if (compute_ret == SUCCESS) {
      CollectCostTimeOfOpConstantFolding(node, start_time);
    } else {
      // If computation on AICPU is not possible, try running the host kernel within GE.
      GELOGD("Try to compute weight of %s with built-in kernel.", node->GetName().c_str());
      compute_ret = ComputeWithBuiltInKernel(node, inputs, outputs);
    }
  }
  GELOGD("Constant folding computation for node %s (type: %s) finished, return code: %u.", node->GetName().c_str(),
         node->GetType().c_str(), compute_ret);
  return compute_ret;
}

Status ConstantFoldingPass::ComputeWithHostCpuCustomOp(const NodePtr &node, const vector<ConstGeTensorPtr> &inputs,
                                                       std::vector<GeTensorPtr> &outputs) {
  const std::string op_type = NodeUtils::GetNodeType(node);
  const AscendString op_type_str(op_type.c_str());
  if (!CustomOpFactory::IsExistOp(op_type_str, OpBackend::kHostCPU)) {
    GELOGD("Op of type %s is not supported by host cpu custom op.", op_type.c_str());
    return UNSUPPORTED;
  }
  auto base_custom_op = CustomOpFactory::CreateOrGetCustomOp(op_type_str, OpBackend::kHostCPU);
  GE_ASSERT_NOTNULL(base_custom_op, "Op %s is registered as host cpu custom op but create instance failed.",
                    op_type.c_str());
  auto *host_custom_op = CustomOpCast<HostCpuExecuteOp>(base_custom_op);
  GE_ASSERT_NOTNULL(host_custom_op,
                    "Op %s is registered as host cpu custom op but does not implement HostCpuExecuteOp.",
                    op_type.c_str());

  std::vector<gert::Tensor> input_tensors;
  std::vector<gert::Tensor> output_tensors;
  HostCpuConstFoldingMemGertAllocator allocator;
  gert::KernelContextHolder context_holder;
  GE_ASSERT_SUCCESS(BuildHostCpuOpContext(node, inputs, input_tensors, output_tensors, allocator, context_holder));
  auto *host_context = reinterpret_cast<gert::HostCpuOpExecutionContext *>(context_holder.GetKernelContext());
  GE_ASSERT_NOTNULL(host_context);
  GE_ASSERT_SUCCESS(host_custom_op->Execute(host_context));

  outputs.clear();
  outputs.reserve(output_tensors.size());
  for (const auto &output_tensor : output_tensors) {
    GeTensorPtr output = MakeShared<GeTensor>();
    GE_ASSERT_NOTNULL(output);
    GE_ASSERT_SUCCESS(TensorTransUtils::GertTensor2GeTensor(output_tensor, *output));
    outputs.emplace_back(std::move(output));
  }
  return SUCCESS;
}

Status ConstantFoldingPass::ComputeWithBuiltInKernel(NodePtr &node, const vector<ConstGeTensorPtr> &inputs,
                                                     std::vector<GeTensorPtr> &outputs) {
  auto op_kernel = folding_pass::GetKernelByType(node);
  if (op_kernel == nullptr) {
    GELOGD("No op kernel for node %s type %s, skip the constant folding", node->GetName().c_str(),
           node->GetType().c_str());
    return NOT_CHANGED;
  }

  // Statistic of ge constant folding kernel
  uint64_t start_time = GetCurrentTimestamp();
  auto ret = op_kernel->Compute(node->GetOpDesc(), inputs, outputs);
  CollectCostTimeOfGeConstantFolding(node, start_time);
  return ret;
}

Status ConstantFoldingPass::ComputeWithHostCpuKernel(const NodePtr &node, const vector<ConstGeTensorPtr> &inputs,
                                                     std::vector<GeTensorPtr> &outputs) {
  std::shared_ptr<GELib> instance_ptr = ge::GELib::GetInstance();
  if ((instance_ptr == nullptr) || (!instance_ptr->InitFlag())) {
    GELOGE(GE_CLI_GE_NOT_INITIALIZED, "[Check][Param] GE is not initialized or is finalized.");
    return UNSUPPORTED;
  }
  OpsKernelInfoStorePtr kernel_info = instance_ptr->OpsKernelManagerObj().GetOpsKernelInfoStore(kKernelLibName);
  if (kernel_info == nullptr) {
    GELOGE(FAILED, "[Get][OpsKernelInfoStore] %s failed", kKernelLibName);
    return UNSUPPORTED;
  }

  std::string ops_flag;
  kernel_info->opsFlagCheck(*node, ops_flag);
  if (ops_flag == kOpsFlagClose) {
    return UNSUPPORTED;
  }
  return RunOpKernel(node, inputs, outputs);
}

Status ConstantFoldingPass::RunOpKernel(const NodePtr &node, const std::vector<ConstGeTensorPtr> &inputs,
                                        std::vector<GeTensorPtr> &outputs) {
  const std::string op_type = NodeUtils::GetNodeType(node);
  auto kernel = OpKernelRegistry::GetInstance().CreateHostCpuOp(op_type);
  if (kernel == nullptr) {
    GELOGD("Op of type %s is not supported by host cpu engine", op_type.c_str());
    return UNSUPPORTED;
  }

  GELOGD("Successfully created op kernel. op type = %s", op_type.c_str());
  return HostCpuEngine::GetInstance().Run(node, *kernel, inputs, outputs);
}

const std::map<std::string, std::pair<uint64_t, uint64_t>> &ConstantFoldingPass::GetGeConstantFoldingPerfStatistic()
    const {
  return statistic_of_ge_constant_folding_;
}

const std::map<std::string, std::pair<uint64_t, uint64_t>> &ConstantFoldingPass::GetOpConstantFoldingPerfStatistic()
    const {
  return statistic_of_op_constant_folding_;
}

void ConstantFoldingPass::CollectCostTimeOfGeConstantFolding(const NodePtr &node, uint64_t start_time) {
  uint64_t cost_time = GetCurrentTimestamp() - start_time;
  if (statistic_of_ge_constant_folding_.find(node->GetType()) != statistic_of_ge_constant_folding_.end()) {
    uint64_t &cnt = statistic_of_ge_constant_folding_[node->GetType()].first;
    uint64_t &cur_cost_time = statistic_of_ge_constant_folding_[node->GetType()].second;
    cnt++;
    cur_cost_time += cost_time;
  } else {
    statistic_of_ge_constant_folding_[node->GetType()] = std::pair<uint64_t, uint64_t>(kStartCallNum, cost_time);
  }
}

void ConstantFoldingPass::CollectCostTimeOfOpConstantFolding(const NodePtr &node, uint64_t start_time) {
  if (statistic_of_op_constant_folding_.find(node->GetType()) != statistic_of_op_constant_folding_.end()) {
    uint64_t &cnt = statistic_of_op_constant_folding_[node->GetType()].first;
    uint64_t &cost_time = statistic_of_op_constant_folding_[node->GetType()].second;
    cnt++;
    cost_time += GetCurrentTimestamp() - start_time;
  } else {
    statistic_of_op_constant_folding_[node->GetType()] =
        std::pair<uint64_t, uint64_t>(kStartCallNum, GetCurrentTimestamp() - start_time);
  }
}

string ConstantFoldingPass::GetPassName() const {
  return kPassName;
}

REG_PASS_OPTION("ConstantFoldingPass").SWITCH_OPT(ge::OO_CONSTANT_FOLDING);
}  // namespace ge
