/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <vector>
#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <functional>
#include <memory>
#include <hccl/hccl_types.h>
#define private public
#include "graph/load/model_manager/task_info/hccl/hccl_util.h"
#undef private


namespace ge {

namespace {
HcclResult InitializeHeterogeneousRuntime(const std::string &group, void *tilingData, void *ccuTaskGroup) {
  return HCCL_SUCCESS;
}
}  // namespace

class UtestHccl: public testing::Test {
 protected:
  void SetUp() {
  }
  void TearDown() {
  }
};

TEST_F(UtestHccl, HcomGetCcuTaskInfo) {
  HcclDllHcomMgr mgr = HcclDllHcomMgr::GetInstance();
  HcclDllHcomMgr::GetInstance().hccl_HcomGetCcuTaskInfo_func = &InitializeHeterogeneousRuntime;
  std::string group = "comm1";
  mgr.HcomGetCcuTaskInfoFunc(group, nullptr, nullptr);
}

TEST_F(UtestHccl, HcomGetCcuTaskInfo1) {
  HcclDllHcomMgr mgr = HcclDllHcomMgr::GetInstance();
  std::string group = "comm1";
  mgr.HcomGetCcuTaskInfoFunc(group, nullptr, nullptr);
}

}