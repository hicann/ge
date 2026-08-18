/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "faker/custom_taskdef_faker.h"
#include "faker/task_def_faker.h"
#include "framework/common/debug/ge_log.h"
namespace gert {

CustomTaskDefFaker::CustomTaskDefFaker(std::string stub_name) : inited_(false), stub_name_(stub_name) {}

vector<domi::TaskDef> CustomTaskDefFaker::CreateTaskDef(uint64_t op_index) {
  Init();
  auto task_def = TaskDefFaker::CreateTaskDef(op_index);
  task_def[0].mutable_kernel()->set_stub_func(stub_name_);
  if (!args_format_.empty()) {
    task_def[0].mutable_kernel()->mutable_context()->set_args_format(args_format_);
  }
  GELOGD("CreateTaskDef size:%zu.", task_def.size());
  return task_def;
}

std::unique_ptr<TaskDefFaker> CustomTaskDefFaker::Clone() const {
  return std::unique_ptr<CustomTaskDefFaker>(new CustomTaskDefFaker(*this));
}

CustomTaskDefFaker &CustomTaskDefFaker::BinData(uint64_t data) {
  bin_data = data;
  return *this;
}

CustomTaskDefFaker &CustomTaskDefFaker::ArgsFormat(const std::string &args_format) {
  args_format_ = args_format;
  return *this;
}

void CustomTaskDefFaker::Init() {
  if (inited_) {
    return;
  }
  AddTask({kCustom, kTE_AiCore, bin_data});
  inited_ = true;
}

};  // namespace gert
