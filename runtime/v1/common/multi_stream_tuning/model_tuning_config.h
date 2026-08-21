/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GE_BASE_COMMON_MULTI_STREAM_TUNING_MODEL_TUNING_CONFIG_H_
#define GE_BASE_COMMON_MULTI_STREAM_TUNING_MODEL_TUNING_CONFIG_H_

#include <memory>
#include <string>

#include "common/ge_visibility.h"

namespace ge {
class GeModel;
class GeRootModel;

namespace multistream_tune {
VISIBILITY_EXPORT bool GetTuningMode(const std::shared_ptr<GeModel> &model, std::string &mode);
VISIBILITY_EXPORT bool GetTuningMode(const std::shared_ptr<GeRootModel> &root_model, std::string &mode);
}  // namespace multistream_tune
}  // namespace ge

#endif  // GE_BASE_COMMON_MULTI_STREAM_TUNING_MODEL_TUNING_CONFIG_H_
