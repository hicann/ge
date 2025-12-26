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

#ifndef AIR_COMPILER_GRAPHCOMPILER_ENGINES_NNENG_TRACE_HANDLE_MANAGER_TRACE_HANDLE_MANAGER_H_
#define AIR_COMPILER_GRAPHCOMPILER_ENGINES_NNENG_TRACE_HANDLE_MANAGER_TRACE_HANDLE_MANAGER_H_

#include <string>
#include <map>
#include <mutex>
#include "atrace_types.h"
#include "trace_handle_manager/trace_msg/trace_msg_base.h"
#include "register/graph_optimizer/graph_optimize_register_error_codes.h"

namespace fe {
class TraceHandleManager {
public:
  TraceHandleManager(const TraceHandleManager &) = delete;
  TraceHandleManager &operator=(const TraceHandleManager &) = delete;
  static TraceHandleManager& Instance();
  Status Initialize();
  void Finalize();
  void AddSubGraphTraceHandle();
  void SubmitGlobalTrace(const std::string &trace_msg) const;
  void SubmitGlobalTrace(const TraceMsgBasePtr &trace_msg) const;
  void SubmitStatisticsTrace(const std::string &trace_msg) const;

private:
  TraceHandleManager();
  ~TraceHandleManager();
  static bool SubmitTrace(const TraHandle &trace_handle, const std::string &trace_msg);
  void SaveAndDestroyTraceHandle();
  bool is_init_;
  TraHandle global_handle_;
  TraHandle statistics_handle_;
  TraEventHandle finalize_event_handle_;
  std::map<uint64_t, TraHandle> subgraph_handle_map_;
  std::map<uint64_t, TraEventHandle> subgraph_event_map_;
  mutable std::mutex subgraph_mutex_;
};
}
#endif  // AIR_COMPILER_GRAPHCOMPILER_ENGINES_NNENG_TRACE_HANDLE_MANAGER_TRACE_HANDLE_MANAGER_H_
