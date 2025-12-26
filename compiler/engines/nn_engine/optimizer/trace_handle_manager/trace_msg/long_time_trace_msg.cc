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

#include "trace_handle_manager/trace_msg/long_time_trace_msg.h"
#include <sstream>
#include "common/fe_utils.h"

namespace fe {
LongTimeTraceMsg::LongTimeTraceMsg(const bool is_fusion_op, const int64_t op_id, const std::string &op_type,
                                   const size_t task_wait_second)
  : is_fusion_op_(is_fusion_op), op_id_(op_id), op_type_(op_type), task_wait_second_(task_wait_second) {}

LongTimeTraceMsg::~LongTimeTraceMsg() {}

std::string LongTimeTraceMsg::GenerateTraceMsg() {
  std::stringstream ss;
  ss << "ThreadId:" << std::to_string(GetCurThreadId()) << "|";
  if (is_fusion_op_) {
    ss << "FusionOp:";
  } else {
    ss << "SingleOp:";
  }
  ss << op_type_ << "_" << std::to_string(op_id_) << "|";
  ss << "Task wait second:" << std::to_string(task_wait_second_);
  return ss.str();
}
}
