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

#ifndef SAMPLEPROJECT_ATT_SAMPLE_TESTS_TILING_ST_UTILS_H_
#define SAMPLEPROJECT_ATT_SAMPLE_TESTS_TILING_ST_UTILS_H_
#include <cstdio>
#include <vector>
#include <string>
#include "graph/ascendc_ir/ascendc_ir_core/ascendc_ir.h"
#include "common/checker.h"
namespace att {
constexpr char kTilingDataEntrance[] = "tiling_data";
constexpr char kTilingContextEntrance[] = "tiling_context";
constexpr char kAttToolScence[] = "tool";
constexpr char kAutofuseScence[] = "autofuse";
struct InputArgs {
  std::string output_path{"./"};
  std::string tiling_entrance{"tiling_data"};
  std::string op_name{"OpTest"};
  std::string scence{"tool"};
};
class GenExecTilingUtils {
 public:
  static ge::Status GenExecFunc(std::vector<ge::AscGraph> &graphs, const InputArgs &input_args);

 private:
  static ge::Status GenInputJson(std::vector<ge::AscGraph> &graphs, const std::string &output_path);
  static ge::Status GenExecFunc(std::vector<ge::AscGraph> &graphs, const std::string &op_name,
                                const std::string &output_path);
  static ge::Status GenContextEntranceExecFunc(std::vector<ge::AscGraph> &graphs, const std::string &op_name,
                                               const std::string &output_path);
};
}  // namespace att

#endif  // SAMPLEPROJECT_ATT_SAMPLE_TESTS_TILING_ST_UTILS_H_
