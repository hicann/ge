/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GE_SESSION_V2_INNER_GE_SESSION_H_
#define GE_SESSION_V2_INNER_GE_SESSION_H_

#include <map>
#include <string>
#include <vector>
#include "common/dump/dump_properties.h"
#include "framework/common/ge_types.h"
#include "ge/ge_api_types.h"
#include "ge/ge_data_flow_api.h"
#include "graph/manager/graph_manager.h"
#include "graph/execute/model_executor.h"
#include "ge/ge_allocator.h"

namespace ge {
class InnerGeSession {
 public:
  InnerGeSession(uint64_t session_id, const std::map<std::string, std::string> &options);

  ~InnerGeSession() = default;

  Status Initialize();

  Status Finalize();

  Status AddGraph(uint32_t graph_id, const Graph &graph, const std::map<std::string, std::string> &options);

  Status AddGraphWithCopy(uint32_t graph_id, const Graph &graph, const std::map<std::string, std::string> &options);

  Status RemoveGraph(uint32_t graph_id);

  Status CompileGraph(uint32_t graph_id, const std::vector<ge::Tensor> &inputs);

  Status LoadGraph(const uint32_t graph_id, const std::map<AscendString, AscendString> &options, void *stream);

  Status RunGraph(uint32_t graph_id, const std::vector<gert::Tensor> &inputs, std::vector<gert::Tensor> &outputs);

  Status RunGraphAsync(uint32_t graph_id, const std::vector<gert::Tensor> &inputs,
      std::function<void(Status status, std::vector<gert::Tensor> &outputs)> callback);

  Status RunGraphWithStreamAsync(uint32_t graph_id, const rtStream_t stream,
                                 const std::vector<gert::Tensor> &inputs, std::vector<gert::Tensor> &outputs);

  Status RegisterCallBackFunc(
    const std::string &key,
    const std::function<Status(uint32_t, const std::map<AscendString, gert::Tensor> &)> &callback);

  const GraphManager &getGraphManagerObj() const;

  bool IsGraphNeedRebuild(uint32_t graph_id);

  Status AddDumpProperties(const DumpProperties &dump_properties);

  Status RemoveDumpProperties();

  static void SetRtSocVersion();

  Status GetCompiledGraphSummary(uint32_t graph_id, CompiledGraphSummaryPtr &summary);

  Status SetGraphConstMemoryBase(uint32_t graph_id, const void *const memory, size_t size);

  Status UpdateGraphFeatureMemoryBase(uint32_t graph_id, const void *const memory, size_t size);

  Status SetGraphFixedFeatureMemoryBase(uint32_t graph_id, MemoryType type, const void *const memory, size_t size);

  Status UpdateGraphRefreshableFeatureMemoryBase(uint32_t graph_id, const void *const memory, size_t size);

  Status RegisterExternalAllocator(const void *const stream, AllocatorPtr allocator) const;

  Status UnregisterExternalAllocator(const void * const stream) const;

  Status GetCompiledModel(uint32_t graph_id, ModelBufferData &model_buffer);

  /*
   * @brief 将origin_graph_id图的fork一份，fork出的图与原始图共享编译model，fork出的图可以独立加载出新实例并执行
   * 原始图应该是已编译的状态
   * 当原始图被卸载的时候，fork图也会被卸载
   */
  Status ForkGraph(uint32_t origin_graph_id, uint32_t forked_graph_id);
  uint64_t GetSessionId() const {
    return session_id_;
  }
  void UpdateThreadContext(const std::map<std::string, std::string> &options) const;
  void UpdateThreadContext(uint32_t graph_id);
  void UpdateGlobalSessionContext() const;

 private:
  Status InnerInitialize();
  Status InnerFinalize();
  static void SetTrainFlagOption();
  static Status InitializeExecutionRuntime(const std::map<std::string, std::string> &options);

  // 仅用于防重复初始化，若初始化失败，inner session对象不应被获取到，其成员方法也不会被调用, 因此初始化成功后成员方法内不必再判断
  bool is_initialized_{false};
  uint64_t session_id_;
  uint8_t logLevel_ = DLOG_DEBUG;
  std::map<std::string, std::string> options_;
  GraphManager graph_manager_;
  ModelExecutor model_executor_;
  std::mutex resource_mutex_;  // AddGraph, RemoveGraph and Finalize use
  std::mutex build_run_mutex_;  // BuildGraph and RunGraph use
  Status InitializeVarManager();
  static bool is_dump_server_inited_;
  std::mutex set_mutex_;
};

void CopyGeOutputsMemToUserOutputs(const rtStream_t stream, const std::vector<GeTensor> &ge_outputs,
                                   std::vector<Tensor> &outputs);
}  // namespace ge

#endif  // GE_SESSION_V2_INNER_GE_SESSION_H_
