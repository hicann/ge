/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GE_COMPILER_GRAPH_FUSION_PASS_PYTHON_PASS_BRIDGE_LOADER_HELPER_H_
#define GE_COMPILER_GRAPH_FUSION_PASS_PYTHON_PASS_BRIDGE_LOADER_HELPER_H_

#include <cstdint>

#include "python_pass_bridge_c_api.h"

namespace ge {
namespace fusion {
namespace python_pass_bridge_loader {

inline bool IsBridgeApiValid(const PythonFusionPassBridgeApi *api, const uint32_t expected_abi) {
  return (api != nullptr) && (api->abi_version == expected_abi) && (api->set_artifact_config != nullptr) &&
         (api->register_passes != nullptr) && (api->reset_bridge_state != nullptr) && (api->shutdown_bridge != nullptr);
}

}  // namespace python_pass_bridge_loader
}  // namespace fusion
}  // namespace ge

#endif  // GE_COMPILER_GRAPH_FUSION_PASS_PYTHON_PASS_BRIDGE_LOADER_HELPER_H_
