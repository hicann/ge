/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include <vector>
#include "common/om2/codegen/task_code_builder/rts/stream_switch_task_code_builder.h"
#include "common/om2/codegen/ast/ast_nodes.h"
#include "common/om2/codegen/ast/ast_build_context.h"
#include "common/om2/codegen/ast/ast_context.h"
#include "framework/common/taskdown_common.h"

namespace ge {

class UtestStreamSwitchTaskCodeBuilder : public ::testing::Test {
 protected:
  void SetUp() override {
    builder_ = std::make_unique<StreamSwitchTaskCodeBuilder>(ast_);
  }

  void TearDown() override {}

  AstContext ctx_;
  AstBuildContext ast_{ctx_};
  std::unique_ptr<StreamSwitchTaskCodeBuilder> builder_;
};

TEST_F(UtestStreamSwitchTaskCodeBuilder, RenderDistHelperSuccess) {
  std::vector<DeclNode *> items;
  Status ret = builder_->RenderDistHelper(items);
  EXPECT_EQ(ret, SUCCESS);
  ASSERT_GE(items.size(), 1U);
  ASSERT_NE(items[0], nullptr);
  items.clear();
}

TEST_F(UtestStreamSwitchTaskCodeBuilder, ParseOpIndex_Success) {
  domi::TaskDef task_def;
  task_def.mutable_stream_switch()->set_op_index(42);
  EXPECT_EQ(builder_->ParseOpIndex(task_def), 42);
}

TEST_F(UtestStreamSwitchTaskCodeBuilder, ParseOpIndex_DefaultZero) {
  domi::TaskDef task_def;
  EXPECT_EQ(builder_->ParseOpIndex(task_def), 0);
}

TEST_F(UtestStreamSwitchTaskCodeBuilder, GetFuncName_NotEmpty) {
  EXPECT_FALSE(builder_->GetFuncName().empty());
}

}  // namespace ge
