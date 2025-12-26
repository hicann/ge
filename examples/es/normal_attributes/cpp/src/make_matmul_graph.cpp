/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "es_showcase.h"
#include "es_MatMul.h"
#include <memory>
using namespace ge;
using namespace ge::es;
namespace {
es::EsTensorHolder MakeMatMulGraph(es::EsTensorHolder input, EsGraphBuilder &graph_builder) {
  auto weight = graph_builder.CreateVector(std::vector<int64_t>(6, 1));
  weight.SetShape({2, 3});
  /*
  MatMul原型注释：
  REG_OP(MatMul)
    .INPUT(x1, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32, DT_BF16, DT_HIFLOAT8}))
    .INPUT(x2, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32, DT_BF16, DT_HIFLOAT8}))
    .OPTIONAL_INPUT(bias, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32, DT_BF16}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32, DT_BF16, DT_HIFLOAT8}))
    .ATTR(transpose_x1, Bool, false)
    .ATTR(transpose_x2, Bool, false)
    .OP_END_FACTORY_REG(MatMul)
  */
  // 矩阵乘法: [2, 3]^T × [2, 3] = [3, 2] × [2, 3] = [3, 3]
  // transpose_x1 和 transpose_x2 为 MatMul 的属性
  auto matmul = MatMul(weight, input, nullptr, true);
  return matmul;
}
}
namespace es_showcase {
std::unique_ptr<ge::Graph> MakeMatMulGraphByEs() {
  // 1、创建图构建器
  auto graph_builder = std::make_unique<EsGraphBuilder>("MakeMatMulGraph");
  // 2、创建输入节点
  auto input = graph_builder->CreateInput(0, "input", ge::DT_INT64, ge::FORMAT_ND, {2, 3});  
  auto result = MakeMatMulGraph(input,*graph_builder);
  // 3、设置输出
  (void) graph_builder->SetOutput(result, 0);
  // 4、构建图
  return graph_builder->BuildAndReset();
}
void MakeMatMulGraphByEsAndDump() {
  std::unique_ptr<ge::Graph> graph = MakeMatMulGraphByEs();
  graph->DumpToFile(ge::Graph::DumpFormat::kOnnx, ge::AscendString("make_matmul_graph"));
}
}