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
#include "graph_construct_utils.h"
#include <string>
#include <vector>
#include "gen_model_info.h"
namespace att {
void GraphConstructUtils::UpdateVectorizedStride(const std::vector<int64_t> &axis,
                                                 const std::vector<ge::Expression> &strides,
                                                 const std::vector<int64_t> &vectorized_axis,
                                                 std::vector<ge::Expression> &vectorized_strides) {
  for (auto axis_id : vectorized_axis) {
    int idx = 0;
    for (auto a : axis) {
      if (a == axis_id) {
        vectorized_strides.emplace_back(strides[idx]);
        break;
      }
      idx += 1;
    }
  }
}

void GraphConstructUtils::UpdateGraphVectorizedStride(ge::AscGraph &graph) {
  for (auto x : graph.GetAllNodes()) {
    for (size_t i = 0; i < x->GetAllOutDataAnchorsSize(); i++) {
      UpdateVectorizedStride(x->outputs[i].attr.axis, x->outputs[i].attr.strides, x->outputs[i].attr.vectorized_axis,
                             x->outputs[i].attr.vectorized_strides);
    }
  }
}

void GraphConstructUtils::UpdateGraphsVectorizedStride(std::vector<ge::AscGraph> &impl_graphs) {
  for (auto &graph : impl_graphs) {
    for (auto x : graph.GetAllNodes()) {
      for (size_t i = 0; i < x->GetAllOutDataAnchorsSize(); i++) {
        UpdateVectorizedStride(x->outputs[i].attr.axis, x->outputs[i].attr.strides, x->outputs[i].attr.vectorized_axis,
                               x->outputs[i].attr.vectorized_strides);
      }
    }
  }
}

ge::Status GraphConstructUtils::UpdateTensorAxes(const std::vector<ge::Axis> &axes, ge::AscOpOutput &output,
                                                 const int32_t loop_id) {
  GE_ASSERT_TRUE(loop_id < static_cast<int32_t>(axes.size()));
  ge::Expression stride = att::CreateExpr(1);
  // {z0TB->id, z0Tb->id, z0T->id, z0t->id, z2.id, z3.id};
  // axes size = 6, loop axis id = 2, vectorized_axis size = 3
  const auto vectorized_axis_size = static_cast<int32_t>((loop_id >= 0) ? (axes.size() - loop_id - 1) : axes.size());
  GE_ASSERT_TRUE(vectorized_axis_size >= 0);
  output.axis->resize(axes.size());
  output.vectorized_axis->resize(vectorized_axis_size);
  output.repeats->resize(axes.size());
  output.strides->resize(axes.size());
  // id = 5, 4, 3, 2, 1, 0
  for (auto id = static_cast<int32_t>(axes.size() - 1); id >= 0; id--) {
    if (id - loop_id - 1 >= 0) {
      // vectorized_axis id = (5 - 2 - 1), (4 - 2 - 1), (3 - 2 - 1)
      (*output.vectorized_axis)[id - loop_id - 1] = (axes[id].id);
    }
    // axis id = 5, 4, 3, 2, 1, 0
    (*output.axis)[id] = (axes[id].id);
    (*output.repeats)[id] = (axes[id].size);
    (*output.strides)[id] = (stride);
    stride = stride * axes[id].size;
  }
  return ge::SUCCESS;
}

ge::Status GraphConstructUtils::UpdateOutputTensorAxes(const std::vector<ge::Axis> &axes,
                                                       std::vector<ge::AscOpOutput> &&outputs, const int32_t loop_id) {
  for (auto &output : outputs) {
    GE_ASSERT_SUCCESS(UpdateTensorAxes(axes, output, loop_id));
  }
  return ge::SUCCESS;
}
}  // namespace att