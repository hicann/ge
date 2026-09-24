/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include <string>
#include "compiler/graph/passes/feature/auto_fuse_pass.h"
#include "common/compliant_share_graph.h"
#include "compiler/graph/optimize/symbolic/infer_symbolic_shape/symbolic_shape_inference.h"
#include "tests/framework/ge_runtime_stub/include/common/summary_checker.h"
#include "attribute_group/attr_group_symbolic_desc.h"
#include "faker/space_registry_faker.h"
#include "common/env_path.h"

namespace ge {
class AutoFuseTestPass : public testing::Test {
 protected:
  void SetUp() {
    gert::LoadDefaultSpaceRegistry();
    auto ascend_install_path = EnvPath().GetAscendInstallPath();
    setenv("ASCEND_OPP_PATH", (ascend_install_path + "/opp").c_str(), 1);
    setenv("LD_LIBRARY_PATH", (ascend_install_path + "/runtime/lib64").c_str(), 1);
  }

  void TearDown() {
    gert::UnLoadDefaultSpaceRegistry();
    unsetenv("ASCEND_OPP_PATH");
    unsetenv("LD_LIBRARY_PATH");
  }
};

TEST_F(AutoFuseTestPass, test_auto_fuse_base) {
  const auto graph = cg::BuildAddReluReluGraph({4, 5, 6}, {4, 5, 6});
  EXPECT_NE(graph, nullptr);
  std::vector<GeTensor> inputs;
  GeTensorDesc td;
  td.SetShape((GeShape({4, 5, 6})));
  td.SetOriginShape((GeShape({4, 5, 6})));
  inputs.emplace_back(td);
  inputs.emplace_back(td);

  EXPECT_EQ(SymbolicShapeSymbolizer::Symbolize(graph, inputs), SUCCESS);
  SymbolicShapeInference symbolic;
  symbolic.Infer(graph);
  AutoFusePass pass;
  EXPECT_EQ(pass.Run(graph), ge::SUCCESS);

  // 动态shape，输入数据shape与声明rank不一致：降级跳过该输入，不打挂整个符号化
  const auto graph2 = cg::BuildAddReluReluGraph({4, -1, 6}, {-1, 5, 6});
  std::vector<GeTensor> inputs2;
  GeTensorDesc td2;
  td2.SetShape((GeShape()));
  td2.SetOriginShape((GeShape()));
  inputs2.emplace_back(td2);
  inputs2.emplace_back(td2);
  EXPECT_EQ(SymbolicShapeSymbolizer::Symbolize(graph2, inputs2), SUCCESS);
  // 被跳过的输入不得残留空的 SymbolicDescAttr（否则会被 IsUnInferredNode 误判为推导成功）
  for (const auto &name : {"input_0", "input_1"}) {
    const auto data_node = graph2->FindNode(name);
    ASSERT_NE(data_node, nullptr);
    EXPECT_EQ(data_node->GetOpDesc()->GetOutputDesc(0).GetAttrsGroup<SymbolicDescAttr>(), nullptr);
  }
}
}  // namespace ge
