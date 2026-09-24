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
 * Public serialization API header.
 * Declares the entry point that serializes GertModelData structures
 * into OM2 ZIP model data.
 */

#ifndef INC_FRAMEWORK_COMMON_GERT_MODEL_DATA_SERIALIZE_H_
#define INC_FRAMEWORK_COMMON_GERT_MODEL_DATA_SERIALIZE_H_

#include <string>
#include "common/ge_common/ge_types.h"
#include "ge/ge_ir_build.h"
#include "framework/om2/model_data/gert_model_data.h"

namespace gert {

// 序列化 GertModelData 为 OM2 ZIP 模型数据（ModelBufferData 由调用方持有）
GERT_MODEL_DATA_API ge::Status SerializeGertModelData(const GertModelData &model_data, ge::ModelBufferData &model,
                                                      const bool is_offline, const std::string &writer_path = "");

}  // namespace gert

#endif  // INC_FRAMEWORK_COMMON_GERT_MODEL_DATA_SERIALIZE_H_
