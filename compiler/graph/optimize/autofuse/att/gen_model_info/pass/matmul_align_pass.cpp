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

#include "pass/pass_mgr.h"
#include "nlohmann/json.hpp"

namespace att {
const std::string kLoadToL1 = "loadTcsm";
const uint32_t kNd2NzAlign = 128U;

struct MatmulConfigItem {
  std::map<std::string, uint32_t> align_true;
  std::map<std::string, uint32_t> align_false;
};

using MatmulConfig = std::map<std::string, MatmulConfigItem>;

void ToJson(nlohmann::json &j, const MatmulConfigItem &p) {
  j = nlohmann::json{
    {"0", p.align_false},
    {"1", p.align_true}
  };
}

bool GetMatmulAlignConfig(const TuningSpacePtr &tuning_space, std::map<std::string, std::string> &matmul_config) {
  for (const auto &node : tuning_space->node_infos) {
    if ((node.node_type != kLoadToL1) && node.trans_config.empty()) {
      continue;
    }
    GE_ASSERT_TRUE(!node.outputs.empty(), "Node [%s] is LoadL1 but has no output.", node.name.c_str());
    MatmulConfigItem config;
    const auto &dims = node.outputs[0]->dim_info;
    if (dims.empty()) {
      GELOGW("Node [%s] has no dims.", node.name.c_str());
      nlohmann::json j;
      ToJson(j, config);
      matmul_config.emplace(node.trans_config, j.dump());
      continue;
    }
    size_t dim_size = node.outputs[0]->dim_info.size();
    if (dim_size > 0U) {
      config.align_false.emplace(node.outputs[0]->dim_info[dim_size - 1U]->name, kNd2NzAlign);
    }
    if (dim_size > 1U) {
      config.align_true.emplace(node.outputs[0]->dim_info[dim_size - 2U]->name, kNd2NzAlign);
    }
    nlohmann::json j;
    ToJson(j, config);
    matmul_config.emplace(node.trans_config, nlohmann::to_string(j));
  }

  return true;
}

static std::string kmatmul_align_pass = "matmul_align_pass";
REGISTER_GTC_PASS(kmatmul_align_pass, GetMatmulAlignConfig);
}  // namespace att