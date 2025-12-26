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

#ifndef AIR_COMPILER_GRAPHCOMPILER_ENGINES_NNENG_TRACE_HANDLE_MANAGER_TRACE_MSG_COMPILE_PROCESS_TRACE_MSG_H_
#define AIR_COMPILER_GRAPHCOMPILER_ENGINES_NNENG_TRACE_HANDLE_MANAGER_TRACE_MSG_COMPILE_PROCESS_TRACE_MSG_H_

#include "trace_handle_manager/trace_msg/trace_msg_base.h"

namespace fe {
class CompileProcessTraceMsg : public TraceMsgBase {
public:
  CompileProcessTraceMsg(const size_t total_task_count, const size_t wait_task_count);
  explicit CompileProcessTraceMsg(const size_t total_task_count);
  ~CompileProcessTraceMsg();
  std::string GenerateTraceMsg() override;

private:
  bool is_compile_end_;
  size_t total_task_count_;
  size_t wait_task_count_;
};
}
#endif  // AIR_COMPILER_GRAPHCOMPILER_ENGINES_NNENG_TRACE_HANDLE_MANAGER_TRACE_MSG_COMPILE_PROCESS_TRACE_MSG_H_
