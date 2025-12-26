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
#include "es_Add.h"
#include "es_Sigmoid.h"
#include <memory>
using namespace ge;
using namespace ge::es;
namespace {
  es::EsTensorHolder MakeSigmoidAddGraph(es::EsTensorHolder input) {
    /*
    REG_OP(Sigmoid)
      .INPUT(x, TensorType::UnaryDataType())
      .OUTPUT(y, TensorType::UnaryDataType())
      .OP_END_FACTORY_REG(Sigmoid)
    */
    //
    auto sigmoid0 = Sigmoid(input);
    sigmoid0.SetAttrForNode("_stream_label", "label_0");
    auto sigmoid1 = Sigmoid(input);
    // 添加私有属性
    sigmoid1.SetAttrForNode("node_rank", static_cast<int64_t>(1));
    sigmoid1.SetAttrForNode("execution_priority", static_cast<int64_t>(2));
    sigmoid1.SetAttrForNode("memory_optimization", true);
    sigmoid1.SetAttrForNode("_stream_label", "label_1");
    auto add = Add(sigmoid0, sigmoid1);
    return add;
  }
}
namespace es_showcase {
  std::unique_ptr<ge::Graph> MakeSigmoidAddGraphByEs() {
    // 1、创建图构建器
    auto graph_builder = std::make_unique<EsGraphBuilder>("MakeSigmoidAddGraph");
    // 2、创建输入节点
    auto input = graph_builder->CreateInput(0, "input", ge::DT_FLOAT, ge::FORMAT_ND, {2, 3});
    auto result = MakeSigmoidAddGraph(input);
    // 3、设置输出
    (void) graph_builder->SetOutput(result, 0);
    // 4、构建图
    return graph_builder->BuildAndReset();
  }

  void MakeSigmoidAddGraphByEsAndDump() {
    std::unique_ptr<ge::Graph> graph = MakeSigmoidAddGraphByEs();
    graph->DumpToFile(ge::Graph::DumpFormat::kOnnx, ge::AscendString("make_sigmoid_add_graph"));
    graph->DumpToFile(ge::Graph::DumpFormat::kTxt, ge::AscendString("make_sigmoid_add_graph"));
  }
}