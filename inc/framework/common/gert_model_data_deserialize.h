/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
/*
 * Public deserialization API header for libgert_model_data.so.
 * Declares the entry point that deserializes OM2 ZIP model data
 * into GertModelData structures, plus per-category deserialize entry points.
 */

#ifndef INC_FRAMEWORK_COMMON_GERT_MODEL_DATA_DESERIALIZE_H_
#define INC_FRAMEWORK_COMMON_GERT_MODEL_DATA_DESERIALIZE_H_

#include <cstdint>

#include "framework/om2/model_data/gert_model_data.h"

namespace gert {

GERT_MODEL_DATA_API uint32_t DeserializeGertModelData(const uint8_t *data, uint64_t data_size,
                                                      GertModelData *model_data, uint32_t model_index = 0U);

GERT_MODEL_DATA_API uint32_t DeserializeGertModelMeta(const uint8_t *data, uint64_t data_size,
                                                      GertModelData *model_data, uint32_t model_index = 0U);

GERT_MODEL_DATA_API uint32_t DeserializeGertConstantsConfig(const uint8_t *data, uint64_t data_size,
                                                            GertModelData *model_data, uint32_t model_index = 0U);

GERT_MODEL_DATA_API uint32_t DeserializeGertVisualJson(const uint8_t *data, uint64_t data_size,
                                                       GertModelData *model_data, uint32_t model_index = 0U);

// 仅反序列化 data/model_<index>/variables_config.json 的配置部分（graph_id/var_metas/global_shared_var_size），
// 不解析 entries 与权重数据
GERT_MODEL_DATA_API uint32_t DeserializeGertVariablesConfig(const uint8_t *data, uint64_t data_size,
                                                            GertModelData *model_data, uint32_t model_index = 0U);

}  // namespace gert

#endif  // INC_FRAMEWORK_COMMON_GERT_MODEL_DATA_DESERIALIZE_H_
