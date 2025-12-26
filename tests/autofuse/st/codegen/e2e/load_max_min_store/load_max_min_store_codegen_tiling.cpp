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
#include "ascendc_ir.h"

#include <sstream>
#include "e2e_common.h"

extern "C" bool CodegenTiling(const std::string &op_name, const ascir::FusedScheduledResult &fused_schedule_result,
                              std::map<std::string, std::string> &options,
                              std::map<std::string, std::string> &tiling_func) {
  std::stringstream ss;

  ss << OptilingStub(fused_schedule_result) << std::endl;
  ss << "extern \"C\" void GetTiling(AutofuseTilingData& tiling_data) {" << std::endl;
  ss << "  tiling_data.set_block_dim(1);" << std::endl;
  ss << "  tiling_data.set_z0Tb_size(tiling_data.get_s0() / 8);" << std::endl;
  ss << "  tiling_data.set_z0t_size(8);" << std::endl;
  ss << "  tiling_data.set_z1t_size(8);" << std::endl;
  ss << "}" << std::endl;

  tiling_func["TilingHead"] += ss.str();
  return true;
}

