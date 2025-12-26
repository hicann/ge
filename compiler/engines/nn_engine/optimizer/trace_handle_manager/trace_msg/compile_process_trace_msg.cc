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

#include "trace_handle_manager/trace_msg/compile_process_trace_msg.h"
#include <sstream>
#include "common/fe_utils.h"

namespace fe {
CompileProcessTraceMsg::CompileProcessTraceMsg(const size_t total_task_count, const size_t wait_task_count)
  : is_compile_end_(false), total_task_count_(total_task_count), wait_task_count_(wait_task_count) {}

CompileProcessTraceMsg::CompileProcessTraceMsg(const size_t total_task_count)
  : is_compile_end_(true), total_task_count_(total_task_count), wait_task_count_(0) {}

CompileProcessTraceMsg::~CompileProcessTraceMsg() {}

std::string CompileProcessTraceMsg::GenerateTraceMsg() {
  std::stringstream ss;
  if (is_compile_end_) {
    ss << "Finish subgraph compile. ThreadId:" << GetCurThreadIdStr() << "|";
    ss << "Total task cout:" << std::to_string(total_task_count_);
  } else {
    ss << "Compile process status. ThreadId:" << GetCurThreadIdStr() << "|";
    ss << "Total task cout:" << std::to_string(total_task_count_) << "|";
    ss << "Finished task cout:" << std::to_string(total_task_count_ - wait_task_count_) << "|";
    ss << "Waiting task cout:" << std::to_string(wait_task_count_);
  }

  return ss.str();
}
}
