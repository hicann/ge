/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "system_utils.h"
#include <csignal>
#include "framework/common/debug/log.h"
#include "common/scope_guard.h"

namespace ge {
Status SystemUtils::System(const std::string &cmd, bool print_err) {
  sighandler_t old_handler = signal(SIGCHLD, SIG_DFL);
  GE_MAKE_GUARD(old_handler, [&old_handler]() { signal(SIGCHLD, old_handler); });
  auto status = system(cmd.c_str());
  if (status == -1) {
    if (print_err) {
      GELOGE(FAILED, "Failed to execute cmd.");
    }
    return FAILED;
  }

  if (WIFEXITED(status) && (WEXITSTATUS(status) == 0)) {
    return SUCCESS;
  }

  if (print_err) {
    GELOGE(FAILED, "Execute cmd result failed, ret = %d.", WEXITSTATUS(status));
  }
  return FAILED;
}
}  // namespace ge
