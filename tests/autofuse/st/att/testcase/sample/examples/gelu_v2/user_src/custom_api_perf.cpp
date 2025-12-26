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
#include <iostream>
#include <numeric>
#include "custom_ascend_graph.h"
#include "ascendc_api_perf.h"
namespace att {
constexpr char kVfCall1[] = "VfCall1";
constexpr char kVfCall2[] = "VfCall2";
ge::Status VfCall1Compute(const std::vector<TensorShapeInfo> &input_shapes,
              const std::vector<TensorShapeInfo> &output_shapes,
              const ge::AscNodePtr &node,
              PerfOutputInfo &perf_res) {
 (void)output_shapes;
 (void)node;
 Expr t = CreateExpr(4232.45f);
 Expr a = CreateExpr(0.0f);
 Expr b = CreateExpr(0.0f);
 Expr c = CreateExpr(146.72f);
 Expr h = CreateExpr(1.04f);
 auto dims = input_shapes[0].dims;
 Expr dim_product =
     std::accumulate(dims.begin(), dims.end(), CreateExpr(1), [](Expr &a, Expr &b) { return ge::sym::Mul(a, b); });
 auto cycles = ge::sym::Div(dim_product, t);
 auto weight = ge::sym::Add(ge::sym::Div(a, ge::sym::Add(dims.back(), b)), c);
 cycles = ge::sym::Mul(cycles, weight);
 perf_res.pipe_res[PipeType::AICORE_VEC] = ge::sym::Add(cycles, h);
 return ge::SUCCESS;
}

ge::Status VfCall2Compute(const std::vector<TensorShapeInfo> &input_shapes,
              const std::vector<TensorShapeInfo> &output_shapes,
              const ge::AscNodePtr &node,
              PerfOutputInfo &perf_res) {
 (void)output_shapes;
 (void)node;
 Expr t = CreateExpr(4232.45f);
 Expr h = CreateExpr(1.04f);
 auto dims = input_shapes[0].dims;
 Expr dim_product = std::accumulate(dims.begin(), dims.end(), CreateExpr(1), [](Expr &a, Expr &b) { return ge::sym::Mul(a, b);});
 auto cycles = ge::sym::Mul(dim_product, t);
 perf_res.pipe_res[PipeType::AICORE_VEC] = ge::sym::Add(cycles, h);
 return ge::SUCCESS;
}

REGISTER_EVAL_FUNC(kVfCall1, VfCall1Compute);
REGISTER_EVAL_FUNC(kVfCall2, VfCall2Compute);
}
