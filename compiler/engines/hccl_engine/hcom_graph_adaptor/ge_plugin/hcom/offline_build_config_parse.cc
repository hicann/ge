/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <nlohmann/json.hpp>
#include "offline_build_config_parse.h"
#include <securec.h>
#include <functional>
#include <vector>
#include <memory>
#include <mutex>

#include "graph/utils/tensor_utils.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "framework/memory/memory_api.h"
#include "graph/ge_local_context.h"
#include "framework/common/ge_types.h"  // ge对外options
#include "hccl/hcom.h"
#include "hcom_op_utils.h"
#include "ops_kernel_builder_base.h"
#include "op_hcom_comm.h"
#include "device_capability.h"
#include "mmpa/mmpa_api.h"

using namespace std;

namespace hccl {
static std::mutex g_taskNumCalModeMutex;

bool IsOfflineCompilation() {
  std::string offlineString;
  if (ge::GetThreadLocalContext().GetOption("ge.offline_hccl_compile", offlineString) == ge::GRAPH_SUCCESS) {
    return true;
  }
  return false;
}

HcclResult GetDeterministic(u8 &deterministic) {
  deterministic = DETERMINISTIC_DISABLE;  // 默认为不支持

  char *mmSysGetEnvValue = nullptr;
  MM_SYS_GET_ENV(MM_ENV_HCCL_DETERMINISTIC, mmSysGetEnvValue);
  std::string hcclDeterministicEnv = (mmSysGetEnvValue != nullptr) ? mmSysGetEnvValue : "EmptyString";
  if (hcclDeterministicEnv != "EmptyString") {
    // 环境变量优先
    std::transform(hcclDeterministicEnv.begin(), hcclDeterministicEnv.end(), hcclDeterministicEnv.begin(), ::toupper);
    if (hcclDeterministicEnv == "FALSE") {
      deterministic = DETERMINISTIC_DISABLE;
    } else if (hcclDeterministicEnv == "TRUE") {
      deterministic = DETERMINISTIC_ENABLE;
    } else if (hcclDeterministicEnv == "STRICT") {
      CHK_PRT_RET(!DeviceCapability::Instance().SupportsStrictDeterministic(),
                  HCCL_ERROR("ParserHcclDeterministic: reduce order preservation is not supported"),
                  HCCL_E_NOT_SUPPORT);
      deterministic = DETERMINISTIC_STRICT;
    } else {
      HCCL_ERROR("[GetDeterministic] HCCL_DETERMINISTIC is set to [%s], which is incorrect. Please check",
                 hcclDeterministicEnv.c_str());
      return HCCL_E_PARA;
    }
  } else {
    // 未配环境变量，检查ge option
    std::string geOption;
    if (ge::GetThreadLocalContext().GetOption(ge::DETERMINISTIC, geOption) == ge::GRAPH_SUCCESS) {
      if (geOption == "1") {
        deterministic = DETERMINISTIC_ENABLE;
      } else if (geOption == "2") {
        CHK_PRT_RET(!DeviceCapability::Instance().SupportsStrictDeterministic(),
                    HCCL_ERROR("ParserHcclDeterministic: reduce order preservation is not supported"),
                    HCCL_E_NOT_SUPPORT);
        deterministic = DETERMINISTIC_STRICT;
      }
    }
  }

  return HCCL_SUCCESS;
}
}  // namespace hccl
