/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "dflow/inc/data_flow/model/flow_model_helper.h"
#include <fstream>
#include "graph/debug/ge_attr_define.h"
#include "common/helper/model_parser_base.h"
#include "common/helper/model_helper.h"
#include "dflow/base/model/flow_model_om_loader.h"
#include "dflow/base/model/flow_model_om_saver.h"
#include "graph/build/memory/var_mem_assign_util.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/op_type_utils.h"
#include "dflow/inc/data_flow/model/graph_model.h"

namespace ge {
Status FlowModelHelper::LoadToFlowModel(const std::string &model_path, FlowModelPtr &flow_model,
                                        const std::string &split_om_data_base_dir) {
  flow_model = nullptr;
  ModelData model;
  // Load model from file, default 0 priority.
  GE_CHK_STATUS_RET_NOLOG(ModelParserBase::LoadFromFile(model_path.c_str(), 0, model));
  GE_MAKE_GUARD(model_guard, [&model]() {
    if (model.model_data != nullptr) {
      delete[] static_cast<char *>(model.model_data);
      model.model_data = nullptr;
    }
  });

  const ModelFileHeader *mdl_file_header = nullptr;
  auto ret = ModelHelper::GetModelFileHead(model, mdl_file_header);
  GE_CHK_STATUS_RET(ret, "Failed to get model file head, model_path:%s", model_path.c_str());

  if (mdl_file_header->modeltype != MODEL_TYPE_FLOW_MODEL) {
    ret = LoadGeRootModelToFlowModel(model, flow_model);
    GE_CHK_STATUS_RET(ret, "Failed to load root model to flow model, model_path:%s", model_path.c_str());
  } else {
    ret = FlowModelOmLoader::LoadToFlowModel(model, flow_model, split_om_data_base_dir);
    GE_CHK_STATUS_RET(ret, "Failed to load flow model, model_path:%s", model_path.c_str());
  }
  return SUCCESS;
}

Status FlowModelHelper::LoadGeRootModelToFlowModel(const ModelData &model, FlowModelPtr &flow_model) {
  ModelHelper model_helper;
  GE_CHK_STATUS_RET(model_helper.LoadRootModel(model), "[Load][RootModel] failed.");
  auto ge_root_model = model_helper.GetGeRootModel();
  GE_CHECK_NOTNULL(ge_root_model);
  GE_CHECK_NOTNULL(ge_root_model->GetRootGraph());
  GELOGD("load ge root model success, model_name=%s, graph=%s.", ge_root_model->GetModelName().c_str(),
         ge_root_model->GetRootGraph()->GetName().c_str());
  // ge root model dynamic model name is empty, need set by graph name.
  if (ge_root_model->GetModelName().empty()) {
    ge_root_model->SetModelName(ge_root_model->GetRootGraph()->GetName());
  }
  flow_model = MakeShared<FlowModel>(ge_root_model->GetRootGraph());
  GE_CHECK_NOTNULL(flow_model);
  auto graph_model = MakeShared<ge::GraphModel>(ge_root_model);
  GE_CHECK_NOTNULL(graph_model);
  Status ret = flow_model->AddSubModel(graph_model, PNE_ID_NPU);
  GE_CHK_STATUS_RET(ret, "AddSubModel failed, model_name=%s.", ge_root_model->GetModelName().c_str());
  return ret;
}

Status FlowModelHelper::UpdateGeModelSessionId(const FlowModelPtr &flow_model, const uint64_t session_id) {
  for (const auto &submodel_pair : flow_model->GetSubmodels()) {
    const auto &submodel = submodel_pair.second;
    const auto graph_model = std::dynamic_pointer_cast<GraphModel>(submodel);
    if (graph_model == nullptr) {
      continue;
    }
    const auto &ge_root_model = graph_model->GetGeRootModel();
    if (ge_root_model == nullptr) {
      continue;
    }
    const auto &ge_models = ge_root_model->GetSubgraphInstanceNameToModel();
    for (const auto &pair : ge_models) {
      const auto &ge_model = pair.second;
      uint64_t old_session_id = std::numeric_limits<uint64_t>::max();
      (void)AttrUtils::GetInt(ge_model, MODEL_ATTR_SESSION_ID, old_session_id);
      if (old_session_id != session_id) {
        GE_ASSERT_TRUE(AttrUtils::SetInt(ge_model, MODEL_ATTR_SESSION_ID, session_id));
        GELOGI("update ge model[%s] session id from %llu to %llu",
               ge_model->GetName().c_str(), old_session_id, session_id);
      }
    }
  }
  return SUCCESS;
}

Status FlowModelHelper::UpdateSessionGraphId(const FlowModelPtr &flow_model, const std::string &session_graph_id) {
  std::set<PneModelPtr> refreshed_models;
  return UpdateSessionGraphId(flow_model, session_graph_id, refreshed_models);
}

Status FlowModelHelper::UpdateSessionGraphId(const FlowModelPtr &flow_model,
                                             const std::string &session_graph_id,
                                             std::set<PneModelPtr> &refreshed_models) {
  const auto &root_graph = flow_model->GetRootGraph();
  GE_CHECK_NOTNULL(root_graph, ", flow model root graph is null");
  std::string old_session_graph_id;
  if (AttrUtils::GetStr(*root_graph, ATTR_NAME_SESSION_GRAPH_ID, old_session_graph_id) &&
      (old_session_graph_id == session_graph_id)) {
    GELOGD("session graph id is same, no need update.");
    return SUCCESS;
  }
  // -1 is offline session, no need to update
  if (old_session_graph_id.find("-1") != std::string::npos) {
    GELOGI("old session graph id[%s] is offline, no need update.", old_session_graph_id.c_str());
    return SUCCESS;
  }
  GELOGI("need update graph[%s] session graph id from %s to %s.", root_graph->GetName().c_str(),
         old_session_graph_id.c_str(), session_graph_id.c_str());
  bool root_refreshed = false;
  GE_CHK_STATUS_RET(ModelHelper::UpdateSessionGraphId(root_graph, session_graph_id, root_refreshed),
                    "update graph[%s] session graph id failed",
                    root_graph->GetName().c_str());
  for (const auto &submodel_pair : flow_model->GetSubmodels()) {
    const auto &submodel = submodel_pair.second;
    bool refreshed = false;
    GE_CHK_STATUS_RET(ModelHelper::UpdateSessionGraphId(submodel->GetRootGraph(), session_graph_id, refreshed),
                      "update submodel[%s] session graph id failed", submodel->GetModelName().c_str());
    if ((submodel->GetModelType() == PNE_ID_NPU) || (submodel->GetModelType() == PNE_ID_CPU)) {
      const auto graph_model = std::dynamic_pointer_cast<GraphModel>(submodel);
      GE_CHECK_NOTNULL(graph_model, ", submodel is not GraphModel, model name=%s, model type=%s.",
                       submodel->GetModelName().c_str(), submodel->GetModelType().c_str());
      const auto &ge_root_model = graph_model->GetGeRootModel();
      GE_CHECK_NOTNULL(ge_root_model, ", submodel GetGeRootModel is null, model name=%s, model type=%s.",
                       submodel->GetModelName().c_str(), submodel->GetModelType().c_str());
      const auto &ge_models = ge_root_model->GetSubgraphInstanceNameToModel();
      for (const auto &pair : ge_models) {
        const auto &ge_model = pair.second;
        GE_CHK_STATUS_RET(ModelHelper::UpdateSessionGraphId(ge_model->GetGraph(), session_graph_id, refreshed),
                          "update ge model[%s] session graph id failed", ge_model->GetName().c_str());
      }
    }
    if (refreshed) {
      refreshed_models.emplace(submodel);
    }
  }
  return SUCCESS;
}

Status FlowModelHelper::SaveToOmModel(const FlowModelPtr &flow_model, const std::string &output_file) {
  FlowModelOmSaver om_saver(flow_model);
  return om_saver.SaveToOm(output_file);
}

Status FlowModelHelper::LoadFlowModelFromBuffData(const ModelBufferData &model_buffer_data,
                                                  ge::FlowModelPtr &flow_model) {
  ge::ModelData model_data;
  model_data.model_data = model_buffer_data.data.get();
  model_data.model_len = model_buffer_data.length;

  return TransModelDataToFlowModel(model_data, flow_model);
}

Status FlowModelHelper::LoadFlowModelFromOmFile(const char_t *const model_path, ge::FlowModelPtr &flow_model) {
  flow_model = nullptr;
  ModelData model;
  // Load model from file, default 0 priority.
  GE_CHK_STATUS_RET_NOLOG(ModelParserBase::LoadFromFile(model_path, 0, model));
  GE_MAKE_GUARD(model_guard, [&model]() {
    if (model.model_data != nullptr) {
      delete[] static_cast<char *>(model.model_data);
      model.model_data = nullptr;
    }
  });
  return TransModelDataToFlowModel(model, flow_model);
}

Status FlowModelHelper::TransModelDataToFlowModel(const ge::ModelData &model_data, ge::FlowModelPtr &flow_model) {
  const ModelFileHeader *mdl_file_header = nullptr;
  auto ret = ModelHelper::GetModelFileHead(model_data, mdl_file_header);
  GE_CHK_STATUS_RET(ret, "Failed to get model file header, model_name:%s", model_data.om_name.c_str());
  GELOGI("model_name:%s, mdl_file_header name:%s, model_type:%d", model_data.om_name.c_str(), mdl_file_header->name,
         mdl_file_header->modeltype);

  if (mdl_file_header->modeltype != MODEL_TYPE_FLOW_MODEL) {
    GELOGI("mdl_file_header->modeltype = %d.", mdl_file_header->modeltype);
    ret = LoadGeRootModelToFlowModel(model_data, flow_model);
    GE_CHK_STATUS_RET(ret, "Failed to load root model to flow model, om_name:%s.", model_data.om_name.c_str());
    if (flow_model->GetModelRelation() == nullptr) {
      GELOGI("flow_model->GetModelRelation() == nullptr");
      auto sub_model_relation = std::make_shared<ge::ModelRelation>();
      GE_CHECK_NOTNULL(sub_model_relation);
      const auto &graph = flow_model->GetRootGraph();
      GE_DUMP(graph, "flow_model_" + graph->GetName());
      GE_CHK_STATUS_RET(ge::ModelRelationBuilder().BuildForSingleModel(*graph, *sub_model_relation),
                        "Failed to build model relation for graph[%s].", graph->GetName().c_str());
      flow_model->SetModelRelation(sub_model_relation);
    }
  } else {
    GELOGI("mdl_file_header->modeltype == MODEL_TYPE_FLOW_MODEL.");
    ret = FlowModelOmLoader::LoadToFlowModel(model_data, flow_model);
    GE_CHK_STATUS_RET(ret, "Failed to load flow model, om_name:%s", model_data.om_path.c_str());
  }
  return ge::SUCCESS;
}

PneModelPtr FlowModelHelper::ToPneModel(const GeRootModelPtr &root_model, const std::string &model_type) {
  GE_ASSERT_NOTNULL(root_model);
  auto graph_model = MakeShared<ge::GraphModel>(root_model);
  GE_ASSERT_NOTNULL(graph_model);
  graph_model->SetModelType(model_type);
  return graph_model;
}

Status FlowModelHelper::EnsureWithModelRelation(const FlowModelPtr &flow_model) {
  if (flow_model->GetModelRelation() != nullptr) {
    GELOGD("Model[%s] has model relation.", flow_model->GetModelName().c_str());
    return SUCCESS;
  }
  auto graph = flow_model->GetRootGraph();
  GE_CHECK_NOTNULL(graph);
  auto model_relation = MakeUnique<ModelRelation>();
  GE_CHECK_NOTNULL(model_relation);
  GE_CHK_STATUS_RET(ModelRelationBuilder().BuildForSingleModel(*graph, *model_relation),
                    "Failed to build ModelRelation from root graph: %s.", graph->GetName().c_str());
  flow_model->SetModelRelation(std::shared_ptr<ModelRelation>(model_relation.release()));
  GELOGD("Make model relation for graph[%s] success.", graph->GetName().c_str());
  return SUCCESS;
}
}  // namespace ge
