/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GE_GRAPH_LOAD_NEW_MODEL_MANAGER_DAVINCI_MODEL_INPUT_H2D_OVERLAP_PLAN_H_
#define GE_GRAPH_LOAD_NEW_MODEL_MANAGER_DAVINCI_MODEL_INPUT_H2D_OVERLAP_PLAN_H_

#include <cstdint>
#include <map>
#include <memory>
#include <set>
#include <vector>

#include "acl/acl_rt.h"
#include "ge/ge_api_types.h"

namespace domi {
class ModelTaskDef;
}

namespace gert {
class Tensor;
}

namespace ge {
class GeModel;
class InputH2DOverlapSyncCopyWorker;
class GeTensor;
struct InputData;
class NamedAttrs;
struct MemAllocation;
struct MemAllocationSlice;
struct RuntimeParam;

struct InputH2DOverlapCopyItem {
  uint32_t input_index = 0U;
  uint32_t allocation_id = 0U;
  uint64_t allocation_offset = 0U;
  uint64_t size = 0U;
};

struct InputH2DOverlapWaitPoint {
  uint32_t stream_id = 0U;
  uint32_t event_id = 0U;
  uint32_t wait_task_id = 0U;
};

struct InputH2DOverlapCopyGroup {
  std::vector<InputH2DOverlapCopyItem> inputs;
  std::vector<InputH2DOverlapWaitPoint> wait_points;
};

class InputH2DOverlapRuntimePlan {
 public:
  InputH2DOverlapRuntimePlan();
  ~InputH2DOverlapRuntimePlan();
  InputH2DOverlapRuntimePlan(InputH2DOverlapRuntimePlan &&other) noexcept;
  InputH2DOverlapRuntimePlan &operator=(InputH2DOverlapRuntimePlan &&other) noexcept;
  InputH2DOverlapRuntimePlan(const InputH2DOverlapRuntimePlan &) = delete;
  InputH2DOverlapRuntimePlan &operator=(const InputH2DOverlapRuntimePlan &) = delete;

  Status LoadInputIndexes(const GeModel *ge_model, uint32_t model_id);

  Status Init(const GeModel *ge_model, const domi::ModelTaskDef &model_task_def, bool is_online_infer_dynamic,
              bool has_no_tiling_input, uint32_t model_id, const RuntimeParam &runtime_param,
              const std::map<uint32_t, uint32_t> &stream_to_first_task_id,
              const std::vector<MemAllocation> &logical_mem_allocations,
              const std::map<uint32_t, MemAllocationSlice> &input_indexes_to_copy_info,
              const std::vector<uint32_t> &input_index_to_allocation_ids);

  Status Launch(uint32_t model_id, const std::vector<aclrtStream> &stream_list,
                const std::vector<aclrtEvent> &event_list, const std::vector<MemAllocation> &logical_mem_allocations,
                const uint64_t *allocation_ids_to_active_base_addr, const std::vector<gert::Tensor> &tensors,
                const std::set<uint32_t> &prepared_input_indexes, bool is_dynamic, bool is_dynamic_aipp,
                int32_t log_level) const;

  Status Launch(uint32_t model_id, const std::vector<aclrtStream> &stream_list,
                const std::vector<aclrtEvent> &event_list, const std::vector<MemAllocation> &logical_mem_allocations,
                const uint64_t *allocation_ids_to_active_base_addr, const InputData &input_data,
                const std::vector<GeTensor> &tensors, const std::set<uint32_t> &prepared_input_indexes,
                bool use_tensor_desc_placement, bool is_dynamic, bool is_dynamic_aipp, int32_t log_level) const;

  bool Enabled() const;
  bool IsPlannedInput(uint32_t input_index) const;
  uint32_t GetCopyStreamId() const;

 private:
  Status ParseAttr(const NamedAttrs &plan_attr);
  Status ValidateParsedPlan(uint32_t model_id, bool is_online_infer_dynamic, bool has_no_tiling_input,
                            const RuntimeParam &runtime_param,
                            const std::map<uint32_t, uint32_t> &stream_to_first_task_id) const;
  Status BuildRuntimePlan(uint32_t model_id, const domi::ModelTaskDef &model_task_def,
                          const RuntimeParam &runtime_param, const std::vector<MemAllocation> &logical_mem_allocations,
                          const std::map<uint32_t, MemAllocationSlice> &input_indexes_to_copy_info,
                          const std::vector<uint32_t> &input_index_to_allocation_ids,
                          InputH2DOverlapRuntimePlan &runtime_plan, std::set<uint32_t> &event_ids) const;
  Status AppendCopyGroup(uint32_t model_id, const InputH2DOverlapCopyGroup &group_def,
                         const domi::ModelTaskDef &model_task_def, const RuntimeParam &runtime_param,
                         const std::vector<MemAllocation> &logical_mem_allocations,
                         const std::map<uint32_t, MemAllocationSlice> &input_indexes_to_copy_info,
                         const std::vector<uint32_t> &input_index_to_allocation_ids, std::set<uint32_t> &event_ids);
  Status ResolveInputCopyItem(uint32_t model_id, const InputH2DOverlapCopyItem &input_def,
                              const std::vector<MemAllocation> &logical_mem_allocations,
                              const std::map<uint32_t, MemAllocationSlice> &input_indexes_to_copy_info,
                              const std::vector<uint32_t> &input_index_to_allocation_ids,
                              InputH2DOverlapCopyItem &runtime_input);
  Status AppendWaitPoint(uint32_t model_id, const InputH2DOverlapWaitPoint &wait_def,
                         const domi::ModelTaskDef &model_task_def, const RuntimeParam &runtime_param,
                         std::set<uint32_t> &event_ids, InputH2DOverlapCopyGroup &group) const;

  bool enabled_ = false;
  uint32_t copy_stream_id_ = 0U;
  std::vector<InputH2DOverlapCopyGroup> groups_;
  std::set<uint32_t> input_indexes_;
  mutable std::unique_ptr<InputH2DOverlapSyncCopyWorker> sync_copy_worker_;
};

}  // namespace ge

#endif  // GE_GRAPH_LOAD_NEW_MODEL_MANAGER_DAVINCI_MODEL_INPUT_H2D_OVERLAP_PLAN_H_
