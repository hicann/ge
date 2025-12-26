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

#ifndef ATT_EXTRA_INFO_CONFIG_H_
#define ATT_EXTRA_INFO_CONFIG_H_
#include <string>
namespace att {
struct ExtraInfoConfig {
  std::string tiling_data_type_name{"TilingData"};
  bool do_api_tiling{false};       // 控制高阶api tiling是否需要生成
  bool do_input_args_proc{false};  // 控制入参校验逻辑是否需要生成
  bool do_axes_calc{false};        // 控制外轴、尾轴等逻辑是否需要生成
  bool with_tiling_ctx{false};
};
}  // namespace att
#endif // ATT_EXTRA_INFO_CONFIG_H_