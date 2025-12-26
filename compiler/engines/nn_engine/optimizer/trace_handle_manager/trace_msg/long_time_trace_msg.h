/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
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

#ifndef AIR_COMPILER_GRAPHCOMPILER_ENGINES_NNENG_TRACE_HANDLE_MANAGER_TRACE_MSG_LONG_TIME_TRACE_MSG_H_
#define AIR_COMPILER_GRAPHCOMPILER_ENGINES_NNENG_TRACE_HANDLE_MANAGER_TRACE_MSG_LONG_TIME_TRACE_MSG_H_

#include "trace_handle_manager/trace_msg/trace_msg_base.h"

namespace fe {
class LongTimeTraceMsg : public TraceMsgBase {
public:
  LongTimeTraceMsg(const bool is_fusion_op, const int64_t op_id, const std::string &op_type,
                   const size_t task_wait_second);
  ~LongTimeTraceMsg();
  std::string GenerateTraceMsg() override;

private:
  bool is_fusion_op_;
  int64_t op_id_;
  std::string op_type_;
  size_t task_wait_second_;
};
}
#endif  // AIR_COMPILER_GRAPHCOMPILER_ENGINES_NNENG_TRACE_HANDLE_MANAGER_TRACE_MSG_LONG_TIME_TRACE_MSG_H_
