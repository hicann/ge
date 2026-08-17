/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "pass_plugin_loader.h"

#include <cstdlib>
#include <mutex>

#include "framework/common/debug/ge_log.h"
#include "register/custom_pass_helper.h"
#include "python_pass_pybind_bridge.h"

namespace ge {
namespace fusion {
namespace {
constexpr const char *kEnvPythonPassPath = "ASCEND_GE_PY_PASS_PATH";

bool NeedLoadPythonPasses() {
  const char *env_value = std::getenv(kEnvPythonPassPath);
  return (env_value != nullptr) && (env_value[0] != '\0');
}

class PassPluginLoader {
 public:
  static PassPluginLoader &GetInstance() {
    static PassPluginLoader instance;
    return instance;
  }

  Status Load() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (active_users_ == 0U) {
      if (!cpp_pass_loaded_) {
        const auto ret = CustomPassHelper::Instance().Load();
        if (ret != SUCCESS) {
          GELOGE(ret, "Load C++ custom pass plugins failed.");
          return ret;
        }
        cpp_pass_loaded_ = true;
      }
      if ((!python_pass_loaded_) && NeedLoadPythonPasses()) {
        const auto ret = RegisterPythonPassesFromPlugin();
        if (ret != SUCCESS) {
          GELOGE(ret, "Load Python fusion pass plugins failed.");
          (void)CustomPassHelper::Instance().Unload();
          cpp_pass_loaded_ = false;
          return ret;
        }
        python_pass_loaded_ = true;
      }
    }
    active_users_++;
    GELOGD("LoadPassPlugins active_users_=%zu", active_users_);
    return SUCCESS;
  }

  Status Unload() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (active_users_ == 0U) {
      GELOGW("UnloadPassPlugins called with no active users, possible reference leak.");
      return SUCCESS;
    }
    active_users_--;
    GELOGD("UnloadPassPlugins active_users_=%zu", active_users_);
    if (active_users_ == 0U) {
      if (python_pass_loaded_) {
        UnloadPythonPasses();
        python_pass_loaded_ = false;
      }
      if (cpp_pass_loaded_) {
        cpp_pass_loaded_ = false;
        (void)CustomPassHelper::Instance().Unload();
      }
      if (!shutdown_done_) {
        shutdown_done_ = true;
        ShutdownPythonPassesForProcess();
        GELOGI("[PythonPass] ShutdownPythonPassesForProcess done.");
      }
    }
    return SUCCESS;
  }

 private:
  std::mutex mutex_;
  size_t active_users_{0U};
  bool cpp_pass_loaded_{false};
  bool python_pass_loaded_{false};
  bool shutdown_done_{false};
};
}  // namespace

Status LoadPassPlugins() {
  return PassPluginLoader::GetInstance().Load();
}

Status UnloadPassPlugins() {
  return PassPluginLoader::GetInstance().Unload();
}
}  // namespace fusion
}  // namespace ge
