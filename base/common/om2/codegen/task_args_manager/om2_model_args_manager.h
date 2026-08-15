/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_CXX_BASE_COMMON_OM2_CODEGEN_OM2_MODEL_ARGS_MANAGER_H_
#define AIR_CXX_BASE_COMMON_OM2_CODEGEN_OM2_MODEL_ARGS_MANAGER_H_

#include "graph/def_types.h"
#include "common/om2/codegen/task_code_builder_factory.h"
#include "graph/utils/math_util.h"
#include "proto/task.pb.h"
#include "om2_model_adapter.h"
#include <map>
#include <memory>
#include <cmath>
#include <string>
#include <thread>
#include <vector>
#include "common/opskernel/ge_task_info.h"
#include "framework/common/helper/model_helper.h"
#include "graph/node.h"
#include "om2_memory_app_type_classifier.h"
#include "om2_model_args_layout_planner.h"
#include "om2_task_args_refresh_type_classifier.h"

namespace ge {
namespace om2 {

constexpr size_t kArgsReserved = 16UL;
constexpr size_t kArgsFieldSize = sizeof(void *);

class ModelArgsManager {
 public:
  enum UpdatePolicy : int32_t {
    kNoNeedUpdate = 0,
    KUpdateHostInput = 1,
    kUpdateModelIo = 2,
    kUpdateFmAndModelIo = 3,
    kInitOneTime = 4,
    kUpdatePolicyEnd = 5,
  };
  struct UpdateHostArgsArg {
    size_t task_index;
    TaskCodeBuilder *task_info;
    std::vector<HostArg> host_args;
  };
  struct H2DCopyArg {
    uint64_t len;
    uint64_t device_addr;
    void *host_addr;
    ArgsPlacement placement;
  };
  struct SqeUpdateArg {
    uint32_t stream_id;
    uint32_t task_id;
    uint64_t dev_addr;
    uint64_t len;
  };
  struct ArgsUpdateData {
    std::vector<UpdateHostArgsArg> update_datas;
    SmallVector<H2DCopyArg, static_cast<uint32_t>(ArgsPlacement::kEnd)> h2d_copy_datas;
    std::vector<SqeUpdateArg> seq_update_datas;
  };

  struct FixedAddrPiece {
    TaskArgsRefreshTypeClassifier::TaskFixedAddr desc;
    uint64_t device_addr;
  };

  struct FixedAddrBulk {
    uint64_t device_addr;
    std::vector<FixedAddrPiece> pieces;
  };

 public:
  ModelArgsManager();

  ~ModelArgsManager() noexcept;

  Status Init(const GeModelPtr &model, const std::vector<TaskCodeBuilderPtr> *task_code_builder_list_ptr);

  Status GenerateArgsDataForProgramGenerator(Om2CodegenModel &codegen_model);

 private:
  struct OneTaskUpdateData {
    UpdateHostArgsArg update_data;
    bool has_sqe_placement;
    SqeUpdateArg sqe_update_arg;
    std::vector<PisToArgs> *task_indexes_to_args;
  };

 private:
  Status InitTaskInfoV2(domi::ModelTaskDef &model_task_def);

  Status AllocModelArgs(const ModelArgsLayoutPlannedResult &layout, std::vector<ModelArgs> &model_args,
                        std::vector<uint64_t> &model_args_len, ArgsPlacement &pls);
  Status ConstructUpdateData(const TaskNodeMap &task_node_map, const ModelArgsLayoutPlannedResult &layout,
                             const std::vector<TaskRunParam> &task_indexes_to_param,
                             std::vector<PisToArgs> &task_indexes_to_args);
  Status ConstructOneTaskUpdateData(
      const size_t task_index, const OneTaskArgsLayoutResult &task_arg_results,
      const std::vector<TaskRunParam> &task_indexes_to_param,
      const std::array<const ModelArgs *, static_cast<size_t>(ArgsPlacement::kEnd)> &pis_to_model_args,
      OneTaskUpdateData &task_update_data, const AddrUseFor addr_use_for) const;
  Status AddToTaskUpdateDataToPolicies(
      const size_t task_index,
      const SmallVector<ModelArgsManager::UpdatePolicy, ModelArgsManager::kUpdatePolicyEnd> &upis,
      const OneTaskUpdateData &one_task_update_data);
  Status AllocFixedAddrs(const TaskNodeMap &task_node_map,
                         const TaskArgsRefreshTypeClassifier::FixedAddrs &fixed_addrs);

  Status ParseModelTaskDef(domi::ModelTaskDef &model_task_def, std::vector<TaskRunParam> &task_indexes_to_run_param,
                           TaskNodeMap &task_node_map);
  Status ConstructTaskInitParams(
      const std::vector<TaskArgsRefreshTypeClassifier::TaskRefreshType> &task_indexes_to_refresh_type,
      const std::map<std::pair<uint64_t, uint64_t>, MemoryAppType> &logical_addrs_to_mem_app_type,
      std::vector<TaskRunParam> &&task_indexes_to_param, std::vector<IowAddrs> &task_indexes_to_init_param) const;

  static Status ConstructH2DCopyParams(const ModelArgs &model_arg, const UpdatePolicy up, H2DCopyArg &cp_arg);
  using TriggerPolicies = SmallVector<ModelArgsManager::UpdatePolicy, ModelArgsManager::kUpdatePolicyEnd>;
  using TriggerTypesToPolicies = std::array<TriggerPolicies, static_cast<uint32_t>(UpdateTriggerType::kEnd)>;
  TriggerTypesToPolicies GenerateTriggerTypesToCorrespondingUpdatePolicies() const;

  void DebugLogTaskUpdatePolicies(const TaskNodeMap &task_node_map, const TriggerPolicies &upis,
                                  size_t task_index) const;
  Status ValidateTaskRunParam(const std::vector<TaskArgsDesc> &args_descs) const;

  Status GenModelArgsRefreshInfosForTask(std::vector<TaskArgsRefreshInfo> &infos, PisToArgs &pls_to_args,
                                         const NodePtr &node);

  void InitForUpdate();

 private:
  uint32_t update_version_{2};
  const std::vector<TaskCodeBuilderPtr> *task_list_ptr_{nullptr};
  ModelAdapter model_adapter_;
  std::vector<ModelArgs> model_args_;
  std::array<std::unique_ptr<ArgsUpdateData>, kUpdatePolicyEnd> update_policies_to_model_data_;
  std::unordered_map<size_t, SmallVector<std::function<void(const TaskCodeBuilder *)>, kUpdatePolicyEnd>>
      task_indexes_to_update_data_appenders_on_distributed_;
  FixedAddrBulk fixed_addr_bulk_{};
  bool has_args_{false};
  std::vector<uint64_t> last_bases_;
  std::vector<uint32_t> id_to_plicy_;
  std::vector<uint64_t> id_to_len_;
  std::vector<uint64_t> model_args_len_;
  uint64_t host_input_size_{0U};
  uint64_t host_input_partition_len_{0U};
  int8_t logLevel_{DLOG_DEBUG};
  ArgsPlacement op_refresh_placement_{ArgsPlacement::kEnd};

  std::vector<std::vector<ModelArgsRefreshInfo>> allocation_ids_to_model_args_refresh_infos_addr_all;
  std::vector<std::vector<ModelArgsRefreshInfo>> allocation_ids_to_model_args_refresh_infos_addr_low_32bit;
  std::vector<std::vector<ModelArgsRefreshInfo>> allocation_ids_to_model_args_refresh_infos_addr_high_32bit;

  std::vector<PisToArgs> task_indexes_to_args_;
};
}  // namespace om2
}  // namespace ge
#endif  // AIR_CXX_BASE_COMMON_OM2_CODEGEN_OM2_MODEL_ARGS_MANAGER_H_
