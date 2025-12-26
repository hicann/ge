/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_RUNTIME_HETEROGENEOUS_DEPLOY_EXECFWK_BUILTIN_THREAD_CLIENT_H_
#define AIR_RUNTIME_HETEROGENEOUS_DEPLOY_EXECFWK_BUILTIN_THREAD_CLIENT_H_

#include "deploy/execfwk/pne_executor_client.h"
#include "executor/engine_thread.h"


namespace ge {
class BuiltinThreadClient : public PneExecutorClient {
 public:
  explicit BuiltinThreadClient(int32_t device_id, bool is_host = false);
  ~BuiltinThreadClient() override = default;
  Status Initialize() override;
  Status Finalize() override;
  Status SyncVarManager(deployer::ExecutorRequest_SyncVarManageRequest sync_var_manage_desc) override;
  Status LoadModel(deployer::ExecutorRequest_BatchLoadModelMessage load_model_desc) override;
  Status UnloadModel(uint32_t model_id) override;
  ProcStatus GetSubProcStat() override;
  void GetAbnormalModelInsName(std::map<uint32_t, std::vector<std::string>> &abnormal_model_instances_name) override {
    (void) abnormal_model_instances_name;
  }
  Status ClearModelRunningData(uint32_t model_id, int32_t type, const std::set<int32_t> &device_ids) override;
  Status DataFlowExceptionNotify(const deployer::DataFlowExceptionNotifyRequest &req_body) override;
  Status UpdateProfilingFromExecutor(deployer::ExecutorRequest_UpdateProfRequest &prof_message) override {
    (void) prof_message;
    return SUCCESS;
  }

 private:
  EngineThread engine_thread_;
  bool is_host_ = false;
  std::atomic_bool heartbeat_listening_{false};
};

class HostCpuThreadClient : public BuiltinThreadClient {
 public:
  explicit HostCpuThreadClient(int32_t device_id);
  ~HostCpuThreadClient() override = default;
};
}  // namespace ge
#endif  // AIR_RUNTIME_HETEROGENEOUS_DEPLOY_EXECFWK_BUILTIN_THREAD_CLIENT_H_
