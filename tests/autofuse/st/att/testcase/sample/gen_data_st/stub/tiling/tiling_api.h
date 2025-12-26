/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2023. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef SAMPLEPROJECT_ATT_SAMPLE_ST_PROJECT_STUB_TILING_TILING_API_H_
#define SAMPLEPROJECT_ATT_SAMPLE_ST_PROJECT_STUB_TILING_TILING_API_H_
#include "register/tilingdata_base.h"
#include "exe_graph/runtime/tiling_context.h"
#include "platform_ascendc.h"
namespace optiling {
BEGIN_TILING_DATA_DEF(TCubeTiling)
END_TILING_DATA_DEF;

BEGIN_TILING_DATA_DEF(SoftMaxTiling)
END_TILING_DATA_DEF;
}  // namespace optiling
#endif