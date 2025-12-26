/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "dflow/inc/data_flow/model/graph_model.h"
#include "framework/common/helper/model_helper.h"

namespace ge {

GraphModel::GraphModel(const GeRootModelPtr &ge_root_model)
    : PneModel(ge_root_model->GetRootGraph()), ge_root_model_(ge_root_model) {
  PneModel::SetModelName(ge_root_model->GetModelName());
  PneModel::SetModelId(ge_root_model->GetModelId());
}

Status GraphModel::SerializeModel(ModelBufferData &model_buff) {
  bool is_unknown_shape = false;
  (void)ge_root_model_->CheckIsUnknownShape(is_unknown_shape);
  ModelHelper model_helper;
  model_helper.SetSaveMode(false);
  GE_CHK_STATUS_RET(model_helper.SaveToOmRootModel(ge_root_model_, "no-output.om", model_buff, is_unknown_shape),
                    "[Serialize][Submodel] failed, model_name = [%s]", GetModelName().c_str());
  GELOGD("[Serialize][Submodel] succeeded, model_name = [%s], size = %lu", GetModelName().c_str(), model_buff.length);
  return SUCCESS;
}

Status GraphModel::UnSerializeModel(const ModelBufferData &model_buff) {
  (void)model_buff;
  return SUCCESS;
}
}  // namespace ge