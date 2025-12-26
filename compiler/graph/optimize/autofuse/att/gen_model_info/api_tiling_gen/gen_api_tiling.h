/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2024 All rights reserved.
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

#ifndef GEN_API_TILING_H_
#define GEN_API_TILING_H_

#include <string>
#include <map>
#include "ascendc_ir/ascendc_ir_core/ascendc_ir.h"
#include "base/base_types.h"
#include "gen_autofuse_api_tiling.h"

namespace att {
class ApiTilingMgr {
 public:
  static ApiTilingMgr &Instance();

  void SetApiTilingDataType(const uint32_t tiling_key, const std::string &api_name,
                            const std::pair<std::string, std::string> &tiling_name_to_type) {
    api_tiling_data_type_[tiling_key].emplace(api_name, tiling_name_to_type);
  }

  void SetApiTilingFunc(const uint32_t tiling_key, const std::string &api_name, const std::string &func_impl) {
    api_tiling_func_[tiling_key].emplace(api_name, func_impl);
  }

  std::map<std::string, std::string> GetApiTilingFunc(const uint32_t api_key) {
    if (api_tiling_func_.find(api_key) != api_tiling_func_.end()) {
      return api_tiling_func_[api_key];
    }
    std::map<std::string, std::string> res;
    return res;
  }

  std::map<std::string, std::pair<std::string, std::string>> GetApiTilingDataType(const uint32_t api_key) {
    if (api_tiling_data_type_.find(api_key) != api_tiling_data_type_.end()) {
      return api_tiling_data_type_[api_key];
    }
    std::map<std::string, std::pair<std::string, std::string>> res;
    return res;
  }

  void Reset() {
    api_tiling_func_.clear();
    api_tiling_data_type_.clear();
  }

 private:
  ApiTilingMgr() = default;
  ApiTilingMgr(const ApiTilingMgr &) = delete;
  ApiTilingMgr &operator=(const ApiTilingMgr &) = delete;
  ~ApiTilingMgr() = default;

 private:
  std::map<uint32_t, std::map<std::string, std::string>> api_tiling_func_;
  std::map<uint32_t, std::map<std::string, std::pair<std::string, std::string>>> api_tiling_data_type_;
};

struct ApiTilingParams {
  ge::AscGraph graph;
  std::string tiling_data_type;
  TilingScenarioType type;
};

struct NodeApiTilingParams {
  ApiTilingParams api_tiling_params;
  NodeApiTilingCode api_tiling_code;
  ge::AscNodePtr node;
};

ge::Status GetApiTilingInfo(const uint32_t tiling_case_id, const ApiTilingParams &params,
                            std::map<std::string, NodeApiTilingCode> &node_name_to_api_code);
}  // namespace att
#endif  // GEN_API_TILING_H_
