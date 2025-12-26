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
#include "es_BatchNorm.h"
#include "utils.h"
#include <memory>
#include "ge/ge_api.h"
using namespace ge;
using namespace ge::es;
namespace {
es::EsTensorHolder MakeBatchNormGraph(es::EsTensorHolder input, es::EsTensorHolder variance, EsGraphBuilder &graph_builder) {
  auto scale = graph_builder.CreateVector(std::vector<int64_t>(3, 1));
  auto offset = graph_builder.CreateVector(std::vector<int64_t>(3, 0));
  auto batchnorm = BatchNorm(input, scale, offset, nullptr, variance);
  return batchnorm.y;  
}
}
namespace es_showcase {
void MakeBatchNormGraphByEsAndDump() {
  std::unique_ptr<ge::Graph> graph = MakeBatchNormGraphByEs();
  graph->DumpToFile(ge::Graph::DumpFormat::kOnnx, ge::AscendString("make_batchnorm_garph"));
}

std::unique_ptr<ge::Graph> MakeBatchNormGraphByEs() {
  // 1、创建图构建器
  auto graph_builder = std::make_unique<EsGraphBuilder>("MakeBatchNormGarph");
  // 2、创建输入节点
  auto input = graph_builder->CreateInput(0, "input", ge::DT_INT64, ge::FORMAT_ND, {2, 3});
  // 可选输入
  auto variance = graph_builder->CreateInput(1, "variance", ge::DT_INT64, ge::FORMAT_ND, {2, 3});
  auto result = MakeBatchNormGraph(input, variance, *graph_builder);
  // 3、设置输出
  (void) graph_builder->SetOutput(result, 0);
  // 4、构建图
  auto graph = graph_builder->BuildAndReset();
  return graph;
}
}
