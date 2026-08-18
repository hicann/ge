/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef INC_CUSTOM_TASKDEF_FAKER_H
#define INC_CUSTOM_TASKDEF_FAKER_H

#include "task_def_faker.h"
namespace gert {

struct CustomTaskDefFaker : public TaskDefFaker {
  CustomTaskDefFaker(std::string stub_name = "");
  CustomTaskDefFaker &BinData(uint64_t data);
  CustomTaskDefFaker &ArgsFormat(const std::string &args_format);

 private:
  vector<domi::TaskDef> CreateTaskDef(uint64_t op_index = 0) override;
  std::unique_ptr<TaskDefFaker> Clone() const override;

 private:
  void Init();
  bool inited_;
  uint64_t bin_data = 0;
  std::string stub_name_;
  std::string args_format_;
};

}  // namespace gert

#endif
