/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GE_COMMON_HELPER_OM2_RT_VAR_RESOURCE_BUILDER_H_
#define GE_COMMON_HELPER_OM2_RT_VAR_RESOURCE_BUILDER_H_

#include <memory>
#include <vector>

#include "common/om2/rt_var_resource.h"

namespace ge {
class VarManager;
class ComputeGraph;
using ComputeGraphPtr = std::shared_ptr<ComputeGraph>;
struct Om2VarMeta;
}  // namespace ge

namespace gert {

ge::Status BuildRTVarResource(ge::VarManager &var_manager, const ge::ComputeGraphPtr &compute_graph,
                              const std::vector<ge::Om2VarMeta> &var_metas, std::unique_ptr<RTVarResource> &resource);

}  // namespace gert

#endif  // GE_COMMON_HELPER_OM2_RT_VAR_RESOURCE_BUILDER_H_
