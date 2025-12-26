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

#ifndef ATT_GRAPH_CONSTRUCT_UTILS_H
#define ATT_GRAPH_CONSTRUCT_UTILS_H
#include <vector>
#include "ascir_ops.h"
namespace att {
class GraphConstructUtils {
 public:
  static void UpdateVectorizedStride(const std::vector<int64_t> &axis, const std::vector<ge::Expression> &strides,
                                     const std::vector<int64_t> &vectorized_axis,
                                     std::vector<ge::Expression> &vectorized_strides);
  static void UpdateGraphVectorizedStride(ge::AscGraph &graph);
  static void UpdateGraphsVectorizedStride(std::vector<ge::AscGraph> &impl_graphs);
  static ge::Status UpdateTensorAxes(const std::vector<ge::Axis> &axes, ge::AscOpOutput &output, int32_t loop_id = -1);
  static ge::Status UpdateOutputTensorAxes(const std::vector<ge::Axis> &axes, std::vector<ge::AscOpOutput> &&outputs,
                                           int32_t loop_id = -1);
};
}  // namespace att
#endif  // ATT_GRAPH_CONSTRUCT_UTILS_H
