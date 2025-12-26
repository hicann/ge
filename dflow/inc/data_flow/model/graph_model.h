/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef DFLOW_INC_MODEL_GRAPH_MODEL_H_
#define DFLOW_INC_MODEL_GRAPH_MODEL_H_
#include "dflow/inc/data_flow/model/pne_model.h"
#include "common/model/ge_root_model.h"
namespace ge {
class GraphModel : public PneModel {
 public:
  explicit GraphModel(const GeRootModelPtr &ge_root_model);
  ~GraphModel() override = default;

  GeRootModelPtr GetGeRootModel() const {
    return ge_root_model_;
  }

  Status SerializeModel(ModelBufferData &model_buff) override;

  Status UnSerializeModel(const ModelBufferData &model_buff) override;

  void SetModelId(uint32_t model_id) override {
    PneModel::SetModelId(model_id);
    ge_root_model_->SetModelId(model_id);
  }

 private:
  GeRootModelPtr ge_root_model_;
};
}  // namespace ge
#endif  // DFLOW_INC_MODEL_GRAPH_MODEL_H_
