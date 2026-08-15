/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_CXX_BASE_COMMON_OM2_CODEGEN_OM2_MODEL_ADAPTER_H_
#define AIR_CXX_BASE_COMMON_OM2_CODEGEN_OM2_MODEL_ADAPTER_H_

#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>
#include "common/om2/codegen/om2_codegen_types.h"
#include "om2_memory_segment_planner.h"
#include "common/model/ge_model.h"
#include "graph/model.h"
#include "graph/node.h"
#include "graph/op_desc.h"
#include "graph/utils/tensor_utils.h"
#include "ge/ge_api_types.h"
#include "base/registry/op_impl_space_registry_v2.h"

namespace ge {
namespace om2 {
class ModelAdapter {
 public:
  ModelAdapter();

  ~ModelAdapter() noexcept;

  Status Init(const GeModelPtr &ge_model);

  string GetOmName() {
    return name_;
  }

  const RuntimeParam &GetRuntimeParam() const {
    return runtime_param_;
  }

  ComputeGraphPtr GetCompiledComputeGraph() const {
    return ge_model_->GetGraph();
  }

  OpDescPtr GetOpByIndex(const uint32_t op_index) const {
    const auto it = op_list_.find(static_cast<int64_t>(op_index));
    if (it == op_list_.end()) {
      return nullptr;
    }
    return it->second;
  }

  size_t GetFmMemAllocationsSize() const {
    return logical_fm_mem_allocations_size_;
  }

  size_t GetFmMemAllocationsStartId() const {
    return fm_mem_allocations_start_id_;
  }

  bool IsFeatureBaseRefreshable() const;

  bool GetPhysicalMemoryRefreshable() const;

  uintptr_t MallocDynamicMemory(SegmentType type, const size_t size);

  std::vector<MemAllocation> &GetLogicalMemAllocation() {
    return logical_mem_allocations_;
  }

  std::vector<uint32_t> &GetInputIndexToAllocationIds() {
    return input_index_to_allocation_ids_;
  }

  std::vector<uint32_t> &GetOutputIndexToAllocationIds() {
    return output_index_to_allocation_ids_;
  }

 private:
  void PrintNoFrozenInputIndexes();

  Status GetTotalMemSizeExcludeZeroCopy(int64_t &total_useful_size);

  void CreateInputDimsInfo(const OpDescPtr &op_desc, const Format format, ShapeDescription &shape_info,
                           ShapeDescription &dims_info) const;

  void SetInputDimsInfo(const std::vector<int64_t> &input_dims, const Format format,
                        ShapeDescription &shape_info) const;

  Status InitIoNodes(const ComputeGraphPtr &compute_graph);

  Status InitNodes(const ComputeGraphPtr &compute_graph);

  Status InitDataOp(const ComputeGraphPtr &graph, const NodePtr &node, uint32_t &data_op_index,
                    std::map<uint32_t, OpDescPtr> &index_to_data, std::set<uint64_t> &input_outside_addrs);

  Status GenInputOutputInfo(const std::map<uint32_t, OpDescPtr> &index_to_data,
                            const std::vector<OpDescPtr> &output_op_list);

  Status InitNetOutput(const ComputeGraphPtr &graph, const NodePtr &node, std::vector<OpDescPtr> &output_op_list,
                       std::set<uint64_t> &output_outside_addrs);

  void CreateOutput(const size_t index, const OpDescPtr &op_desc, InputOutputDescInfo &output,
                    uint32_t &format_result) const;

  Status InitOutputTensorInfo(const OpDescPtr &op_desc);

  Status InitInputDescInfo(const OpDescPtr &op_desc);

  Status InitOutputDescInfo(const OpDescPtr &op_desc, const std::vector<std::string> &out_node_name);

  Status GenFmMemAllocations();

  Status GenFixedFmMemAllocations();

  Status GenInputMemAllocations(const std::map<uint32_t, OpDescPtr> &index_to_data);

  Status GenOutputMemAllocations(const std::vector<OpDescPtr> &output_op_list);

  Status GenSliceOutputMemAllocations(const std::vector<OpDescPtr> &output_op_list);

  Status GenMemAllocations(const std::map<uint32_t, OpDescPtr> &index_to_data,
                           const std::vector<OpDescPtr> &output_op_list);

  Status GetMemAllocationByLogicAddr(const uint64_t addr, MemAllocationAndOffset &allocation_info) const;

  Status ParseHostInputIndexOption(const size_t input_num);

 private:
  std::string name_;
  uint32_t version_{0U};
  GeModelPtr ge_model_;
  std::map<int64_t, OpDescPtr> op_list_;
  std::map<int64_t, std::shared_ptr<Operator>> operator_list_;
  std::map<uint32_t, bool> output_data_to_slice_flag_;
  RuntimeParam runtime_param_;
  bool feature_base_refreshable_ = true;
  bool is_getnext_sink_dynamic_ = false;
  std::vector<std::string> dynamic_output_shape_info_;
  std::vector<std::vector<uint64_t>> input_addrs_list_;
  std::vector<std::vector<uint64_t>> output_addrs_list_;
  std::vector<int64_t> output_buffer_size_;
  std::vector<GeShape> output_shape_info_;
  std::vector<bool> output_no_tiling_flag_;
  bool has_no_tiling_output_ = false;
  std::vector<InputOutputDescInfo> origin_input_descs_;
  std::vector<InputOutputDescInfo> input_descs_;
  std::vector<InputOutputDescInfo> input_descs_dims_;
  std::vector<uint32_t> input_formats_;
  std::vector<InputOutputDescInfo> output_descs_;
  std::vector<uint32_t> output_formats_;
  size_t logical_fm_mem_allocations_size_{0U};
  std::vector<MemAllocation> logical_mem_allocations_;
  std::vector<std::pair<uint32_t, uint32_t>> refreshable_input_index_and_allocation_ids_;
  std::vector<std::pair<uint32_t, uint32_t>> refreshable_output_index_and_allocation_ids_;
  std::vector<std::pair<uint32_t, uint32_t>> refreshable_input_index_no_frozen_and_allocation_ids_;
  std::vector<std::pair<uint32_t, uint32_t>> refreshable_fm_index_and_allocation_ids_;
  std::vector<std::pair<uint32_t, uint32_t>> fixed_fm_index_and_allocation_ids_;
  std::map<uint32_t, MemAllocationSlice> input_indexes_to_copy_info_;
  std::map<uint32_t, MemAllocationSlice> output_indexes_to_copy_info_;
  std::vector<uint32_t> input_index_to_allocation_ids_;         // 保存零拷贝的input index和allocation id的关系
  std::vector<uint32_t> output_index_to_allocation_ids_;        // 保存零拷贝的output index和allocation id的关系
  std::vector<uint64_t> input_index_to_active_mem_base_addrs_;  // 保存零拷贝的input index和对应的active mem base的关系
  std::vector<uint64_t>
      output_index_to_active_mem_base_addrs_;                // 保存零拷贝的output index和对应的active mem base的关系
  std::vector<uint32_t> zero_copy_input_indexes_;            // 保存零拷贝的input indexes
  std::vector<uint32_t> zero_copy_output_indexes_;           // 保存零拷贝的output indexes
  std::vector<uint32_t> zero_copy_input_indexes_no_frozen_;  // 保存执行时需要零拷贝的output indexes
  std::map<uint32_t, uint64_t> output_indexes_to_tensor_size_;
  size_t fm_mem_allocations_start_id_{0U};
  uint8_t logLevel_ = DLOG_DEBUG;
  size_t fixed_fm_mem_allocations_start_id_{0U};
  size_t logical_fixed_fm_mem_allocations_size_{0U};
  std::unordered_set<uint32_t> frozen_input_indexes_;
  uint64_t no_frozen_input_allocation_base_id_{0U};
  bool support_extend_memory_full_{false};
  std::unordered_set<uint32_t> copy_host_input_indexes_;
  std::vector<CopyHostInputInfo> copy_host_input_infos_;
  uint64_t host_input_size_{0UL};
  MemorySegmentPlanner memory_segment_planner_;
};
}  // namespace om2
}  // namespace ge
#endif  // AIR_CXX_BASE_COMMON_OM2_CODEGEN_OM2_MODEL_ADAPTER_H_
