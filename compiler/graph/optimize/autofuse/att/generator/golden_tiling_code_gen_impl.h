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

#ifndef ATT_GOLDEN_TILING_CODE_GEN_IMPL_H_
#define ATT_GOLDEN_TILING_CODE_GEN_IMPL_H_

#include <string>
#include <set>
#include <map>
#include "tiling_code_gen_impl.h"
#include "high_perf_tiling_code_gen_impl.h"

namespace att {
class GoldenTilingCodeGenImpl : public HighPerfTilingCodeGenImpl {
 public:
  explicit GoldenTilingCodeGenImpl(const std::string &op_name, const TilingCodeGenConfig &config,
                                   const TilingModelInfo &model_infos, const ScoreFuncs &score_funcs,
                                   const bool is_uniq_group)
      : HighPerfTilingCodeGenImpl(op_name, config, model_infos, score_funcs, is_uniq_group) {}

 protected:
  ge::Status GenDoTiling(const ModelInfo &model_info) override;
};
}  // namespace att
#endif