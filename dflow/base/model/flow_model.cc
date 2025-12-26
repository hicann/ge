/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "dflow/inc/data_flow/model/flow_model.h"
#include "dflow/inc/data_flow/model/graph_model.h"

namespace ge {
FlowModel::FlowModel(const ComputeGraphPtr &root_graph) : PneModel(root_graph) {};

Status FlowModel::SerializeModel(ModelBufferData &model_buff) {
  (void) model_buff;
  return SUCCESS;
}

Status FlowModel::UnSerializeModel(const ModelBufferData &model_buff) {
  (void) model_buff;
  return SUCCESS;
}

const std::map<std::string, std::vector<uint32_t>> &FlowModel::GetGroupNameToRankIds() const {
  const std::lock_guard<std::mutex> lk(flow_model_mutex_);
  static const std::map<std::string, std::vector<uint32_t>> kEmpty;
  const auto it = hcom_cluster_descs_.find(GetModelName());
  return it == hcom_cluster_descs_.cend() ? kEmpty : it->second.group_name_to_rank_ids;
}

void FlowModel::SetGroupNameToRankIds(const std::map<std::string, std::vector<uint32_t>> &group_name_to_rank_ids) {
  const std::lock_guard<std::mutex> lk(flow_model_mutex_);
  GetOrCreateHcomClusterDesc(GetModelName()).group_name_to_rank_ids = group_name_to_rank_ids;
}

void FlowModel::SetModelNameToRankId(const std::map<std::string, uint32_t> &model_name_to_rank_id) {
  const std::lock_guard<std::mutex> lk(flow_model_mutex_);
  for (const auto &model_name_and_rank_id : model_name_to_rank_id) {
    const auto &model_name = model_name_and_rank_id.first;
    const uint32_t rank_id = model_name_and_rank_id.second;
    model_name_to_cluster_and_rank_id_[model_name] = std::make_pair(GetModelName(), rank_id);
  }
}

const std::map<std::string, std::vector<uint32_t>> &FlowModel::GetDeviceToRankIds() const {
  const std::lock_guard<std::mutex> lk(flow_model_mutex_);
  static const std::map<std::string, std::vector<uint32_t>> kEmpty;
  const auto it = hcom_cluster_descs_.find(GetModelName());
  return it == hcom_cluster_descs_.cend() ? kEmpty : it->second.device_to_rank_ids;
}

void FlowModel::SetDeviceToRankIds(const std::map<std::string, std::vector<uint32_t>> &device_to_rank_ids) {
  const std::lock_guard<std::mutex> lk(flow_model_mutex_);
  GetOrCreateHcomClusterDesc(GetModelName()).device_to_rank_ids = device_to_rank_ids;
}

void FlowModel::SetHcomClusterDescs(const std::map<std::string, HcomClusterDesc> &hcom_cluster_descs) {
  const std::lock_guard<std::mutex> lk(flow_model_mutex_);
  hcom_cluster_descs_ = hcom_cluster_descs;
}

const std::map<std::string, HcomClusterDesc> &FlowModel::GetHcomClusterDescs() const {
  const std::lock_guard<std::mutex> lk(flow_model_mutex_);
  return hcom_cluster_descs_;
}

void FlowModel::SetModelNameToClusterAndRankId(
    const std::map<std::string, std::pair<std::string, uint32_t>> &model_name_to_cluster_and_rank_id) {
  const std::lock_guard<std::mutex> lk(flow_model_mutex_);
  model_name_to_cluster_and_rank_id_ = model_name_to_cluster_and_rank_id;
}

const std::map<std::string, std::pair<std::string, uint32_t>> &FlowModel::GetModelNameToClusterAndRankId() const {
  const std::lock_guard<std::mutex> lk(flow_model_mutex_);
  return model_name_to_cluster_and_rank_id_;
}

Status FlowModel::MergeHcomClusterInfo(FlowModel &sub_flow_model) {
  const std::lock_guard<std::mutex> lk(flow_model_mutex_);
  if (sub_flow_model.hcom_cluster_descs_.empty()) {
    return SUCCESS;
  }
  for (auto &name_and_cluster_desc : sub_flow_model.hcom_cluster_descs_) {
    const auto &cluster_name = name_and_cluster_desc.first;
    auto &cluster_desc = name_and_cluster_desc.second;
    auto it = this->hcom_cluster_descs_.find(cluster_name);
    if (it != hcom_cluster_descs_.cend()) {
      GE_CHK_BOOL_RET_STATUS(it->second == cluster_desc, PARAM_INVALID,
                             "Cannot merge hcom cluster desc from sub flow model = [%s], cluster_name = [%s]",
                             sub_flow_model.GetModelName().c_str(),
                             cluster_name.c_str());
    } else {
      hcom_cluster_descs_[cluster_name] = std::move(cluster_desc);
      GELOGI("HcomClusterDesc merged, sub flow model = [%s], cluster_name = [%s]",
             sub_flow_model.GetModelName().c_str(),
             cluster_name.c_str());
    }
  }
  for (const auto &model_name_and_cluster_and_rank_id : sub_flow_model.model_name_to_cluster_and_rank_id_) {
    const auto &model_name = model_name_and_cluster_and_rank_id.first;
    auto &cluster_and_rank_id = model_name_and_cluster_and_rank_id.second;
    GE_CHK_BOOL_RET_STATUS(model_name_to_cluster_and_rank_id_.emplace(model_name, cluster_and_rank_id).second,
                           PARAM_INVALID,
                           "model already exists, sub flow model name = %s, name = %s",
                           sub_flow_model.GetModelName().c_str(),
                           model_name.c_str());
    GELOGI("model to cluster rank merged, sub flow model name = %s, model = %s, cluster_name = %s, rank_id = %u",
           sub_flow_model.GetModelName().c_str(),
           model_name.c_str(),
           cluster_and_rank_id.first.c_str(),
           cluster_and_rank_id.second);
  }
  sub_flow_model.hcom_cluster_descs_.clear();
  sub_flow_model.model_name_to_cluster_and_rank_id_.clear();
  return SUCCESS;
}

HcomClusterDesc &FlowModel::GetOrCreateHcomClusterDesc(const std::string &name) {
  auto &hcom_cluster_desc = hcom_cluster_descs_[name];
  if (hcom_cluster_desc.name.empty()) {
    hcom_cluster_desc.name = name;
  }
  return hcom_cluster_desc;
}

bool HcomClusterDesc::operator==(const HcomClusterDesc &rhs) const {
  return (name == rhs.name) &&
      (rank_table == rhs.rank_table) &&
      (device_to_rank_ids == rhs.device_to_rank_ids) &&
      (group_name_to_rank_ids == rhs.group_name_to_rank_ids);
}

Status FlowModel::AddSubModel(const GeRootModelPtr &ge_root_model, const std::string &type) {
  GE_ASSERT_NOTNULL(ge_root_model);
  auto graph_model = MakeShared<ge::GraphModel>(ge_root_model);
  GE_ASSERT_NOTNULL(graph_model);
  graph_model->SetModelType(type);
  return PneModel::AddSubModel(graph_model, type);
}
}  // namespace ge