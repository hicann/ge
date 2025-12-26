/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_CXX_EXPECT_NODE_INFO_CHECK_TEST_H
#define AIR_CXX_EXPECT_NODE_INFO_CHECK_TEST_H
#include "depends/symbol/symbolic_shape_frame_test.h"

namespace ge {
class ExpectNodeInfo : public ExpectNodeInfoCheckBase {
 public:
  ExpectNodeInfo(std::string node_name,
                 std::vector<Expression> expect_symbol_output_shape,
                 std::set<std::string> expect_guard_infos,
                 std::set<std::string> expect_assert_infos,
                 std::vector<Expression> expect_symbolic_value)
      : ExpectNodeInfoCheckBase(
            std::move(node_name),
            std::move(expect_symbol_output_shape),
            std::move(expect_guard_infos),
            std::move(expect_assert_infos),
            std::move(expect_symbolic_value)){}
  ~ExpectNodeInfo() override = default;
  bool ExpectShapeCheck(const gert::SymbolShape &real_shape) const override;
  bool ExpectGuardInfoCheck(std::vector<SymbolCheckInfo> real_guard) const override;
  bool ExpectAssertInfoCheck(std::vector<SymbolCheckInfo> real_assert) const override;
  bool ExpectSymbolValCheck(const std::vector<ge::Expression> * real_val) const override;
};

Status RunSymbolInferenceTest(const ComputeGraphPtr &cg, const std::vector<ExpectNodeInfo> &node_info_vec,
                              const std::vector<ge::GeTensor> &input_vec);
Status SetNoStorage(ComputeGraphPtr &cg, const std::string &DataName, const DataInfo &di, int64_t idx);
}
#endif  // AIR_CXX_EXPECT_NODE_INFO_CHECK_TEST_H
