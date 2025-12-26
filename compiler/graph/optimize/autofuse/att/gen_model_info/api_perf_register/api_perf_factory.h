/**
* Copyright (c) Huawei Technologies Co., Ltd. 2025 All rights reserved.
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
#ifndef __API_PERF_REGISTER_API_PERF_FACTORY_H__
#define __API_PERF_REGISTER_API_PERF_FACTORY_H__

#include <string>
#include <memory>
#include <functional>
#include <map>
#include <mutex>
#include <tuple>
#include <utility>
#include "api_perf.h"
#include "common_utils.h"
namespace att {
using ApiPerfCreatorFun = std::function<std::unique_ptr<ApiPerf>(const std::string &)>;
class __attribute__((visibility("default"))) ApiPerfFactory {
 public:
  static ApiPerfFactory &Instance();

  std::unique_ptr<ApiPerf> Create(const std::string &class_name);

  ApiPerfFactory(const ApiPerfFactory &) = delete;
  ApiPerfFactory &operator=(const ApiPerfFactory &) = delete;

  class Registerar {
   public:
    Registerar(const std::string &api_name, ApiPerfCreatorFun creator) noexcept {
      ApiPerfFactory::Instance().RegisterCreator(api_name, std::move(creator));
    }

    ~Registerar() = default;
  };

 private:
  ApiPerfFactory() = default;

  ~ApiPerfFactory() = default;
  void RegisterCreator(const std::string &api_name, ApiPerfCreatorFun creator) {
    std::lock_guard<std::mutex> lock(mutex_);
    const auto iter = creator_map_.find(api_name);
    if (iter != creator_map_.end()) {
      GELOGD("ApiCallFactory::RegisterCreator: %s creator already exist", api_name.c_str());
      return;
    }
    creator_map_[api_name] = std::move(creator);
    GELOGD("RegisterCreator success, api_name: %s", api_name.c_str());
  }

  std::map<std::string, ApiPerfCreatorFun> creator_map_;
  std::mutex mutex_;
};

template <typename T>
class ApiPerfRegister {
 public:
  // 构造函数：自动注册类
  explicit ApiPerfRegister(const std::string &api_name, Perf perf_func, MicroPerfFunc micro_perf_func,
                           const PerfParamTable *perf_param,
                           const TilingScheduleConfigTable *tiling_schedule_config_table) {
    // 正确用法：使用模板参数T创建对象
    ApiPerfCreatorFun creator = [perf_func, micro_perf_func, perf_param,
                                 tiling_schedule_config_table](const std::string &name) {
      return std::make_unique<T>(name, perf_func, micro_perf_func, perf_param, tiling_schedule_config_table);
    };
    ApiPerfFactory::Registerar(api_name, std::move(creator));
  }
};

}  // namespace att
#endif  // __API_PERF_REGISTER_API_PERF_FACTORY_H__
