/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ge/ge_api.h"
#include <atomic>
#include <iostream>
#include <malloc.h>
#include "ge/ge_api_v2.h"
#include "ge_is_initialize.h"
#include "common/compile_profiling/ge_call_wrapper.h"
#include "common/plugin/datatype_util.h"
#include "common/plugin/plugin_manager.h"
#include "common/plugin/tbe_plugin_manager.h"
#include "common/profiling/profiling_init.h"
#include "common/profiling/profiling_properties.h"
#include "base/err_msg.h"
#include "rt_error_codes.h"
#include "ge/ge_api_types.h"
#include "register/custom_pass_helper.h"
#include "framework/common/debug/ge_log.h"
#include "framework/common/debug/log.h"
#include "framework/common/ge_types.h"
#include "framework/executor/ge_executor.h"
#include "framework/memory/memory_api.h"
#include "framework/common/helper/model_helper.h"
#include "graph/detail/model_serialize_imp.h"
#include "graph/ge_context.h"
#include "graph/manager/util/rt_context_util.h"
#include "graph/manager/graph_external_weight_manager.h"
#include "graph/model_serialize.h"
#include "graph/opsproto_manager.h"
#include "register/op_lib_register_impl.h"
#include "graph/utils/tensor_adapter.h"
#include "graph/utils/type_utils.h"
#include "api/gelib/gelib.h"
#include "api/aclgrph/option_utils.h"
#include "proto/ge_api.pb.h"
#include "register/op_registry.h"
#include "runtime/v2/core/debug/kernel_tracing.h"
#include "session/session_manager.h"
#include "session/session_utils.h"
#include "plog.h"
#include "common/checker.h"
#include "framework/runtime/subscriber/global_profiler.h"
#include "common/option_supportion_checker.h"
#include "base/err_msg.h"
#include "base/err_mgr.h"

namespace {
constexpr int32_t kMaxStrLen = 128;
constexpr uint32_t kExternalErrorCodeMaxValue = 9999999U; // user define error code max value
constexpr size_t kGesizefloat = sizeof(float);
constexpr size_t kGesizehalffloat = sizeof(float) / 2U;
constexpr size_t kGesizefloat8 = sizeof(int8_t);
constexpr size_t kGesizeint8 = sizeof(int8_t);
constexpr size_t kGesizeint16 = sizeof(int16_t);
constexpr size_t kGesizeint32 = sizeof(int32_t);
constexpr size_t kGesizeint64 = sizeof(int64_t);
constexpr size_t kGesizeuint8 = sizeof(uint8_t);
constexpr size_t kGesizebool = sizeof(bool);
constexpr size_t kGesizedouble = sizeof(double);
constexpr size_t kGesizeuint64 = sizeof(uint64_t);
constexpr size_t kGesizeuint16 = sizeof(uint16_t);
constexpr size_t kGesizeuint32 = sizeof(uint32_t);

std::map<ge::DataType, size_t> CONST_OPDATA_TYPE_SIZE_MAP = {
    {ge::DT_FLOAT, kGesizefloat},   {ge::DT_FLOAT16, kGesizehalffloat}, {ge::DT_INT8, kGesizeint8},
    {ge::DT_INT16, kGesizeint16},   {ge::DT_INT32, kGesizeint32},       {ge::DT_INT64, kGesizeint64},
    {ge::DT_UINT8, kGesizeuint8},   {ge::DT_UINT16, kGesizeuint16},     {ge::DT_UINT32, kGesizeuint32},
    {ge::DT_UINT64, kGesizeuint64}, {ge::DT_DOUBLE, kGesizedouble},     {ge::DT_BOOL, kGesizebool},
    {ge::DT_HIFLOAT8, kGesizefloat8}, {ge::DT_FLOAT8_E5M2, kGesizefloat8}, {ge::DT_FLOAT8_E4M3FN, kGesizefloat8},
    {ge::DT_FLOAT8_E8M0, kGesizefloat8}, {ge::DT_FLOAT6_E3M2, kGesizefloat8}, {ge::DT_FLOAT6_E2M3, kGesizefloat8},
    {ge::DT_FLOAT4_E2M1, kGesizefloat8}, {ge::DT_FLOAT4_E1M2, kGesizefloat8},
};

// dfx for RunGraphAsync, log error on error return
void RunGraphAsyncCallback(ge::Status ret, uint64_t session_id, uint32_t graph_id, std::vector<ge::Tensor> &outputs,
                           ge::RunAsyncCallback callback) {
  if ((ret != ge::SUCCESS) && (ret != ge::END_OF_SEQUENCE)) {
    GELOGE(ret, "Run graph async failed, error code:%u, session_id:%lu, graph_id:%u", ret, session_id, graph_id);
    REPORT_INNER_ERR_MSG("E19999", "Run graph async failed, error code:%u, session_id:%lu, graph_id:%u", ret, session_id,
                       graph_id);
  }
  if (callback != nullptr) {
    callback(ret, outputs);
  }
  GELOGI("run graph async finished, session_id: %lu, graph_id: %u, result=%u", session_id, graph_id, ret);
}
}  // namespace

static std::mutex g_ge_release_mutex;  // GEFinalize and ~Session use
static std::shared_ptr<ge::SessionManager> g_session_manager;

ge::SessionManager *GetSessionManager() {
  return g_session_manager.get();
}

namespace ge {
namespace {
  void ConstructSession(const std::map<std::string, std::string> &options, SessionId &session_id) {
    GELOGT(TRACE_INIT, "Session Constructor start");
    // check init status
    session_id = 0U;
    if (!IsGEInitialize()) {
      GELOGE(GE_CLI_GE_NOT_INITIALIZED, "Construct session failed because lack GEInitialize call before.");
      REPORT_INNER_ERR_MSG("E19999", "Construct session failed because lack GEInitialize call before.");
      return;
    }
    // call Initialize
    if (GEAPICheckSupportedSessionOptions(options) != SUCCESS) {
      GELOGW("[Check][Param] Check supported options failed.");
    }
    if (CheckAllowParallelCompile(options) != SUCCESS) {
      return;
    }
    uint64_t tmp_session_id = 0UL;
    const Status ret = g_session_manager->CreateSession(options, tmp_session_id);
    // failed guarder, should call GE_DISMISS_GUARD if success
    GE_DISMISSABLE_GUARD(create_failed,
                         ([tmp_session_id]() {g_session_manager->DestroySession(tmp_session_id);}));
    if (ret != SUCCESS) {
      GELOGE(ret, "Construct session failed, error code:%u.", ret);
      REPORT_INNER_ERR_MSG("E19999", "Construct session failed, error code:%u.", ret);
      return;
    }

    session_id = tmp_session_id;
    GE_DISMISS_GUARD(create_failed);
    GELOGT(TRACE_STOP, "Session construct finished, session id is %lu", session_id);
  }
} // namespace
size_t SessionUtils::NumSessions() {
  std::lock_guard<std::mutex> lock(g_ge_release_mutex);
  return g_session_manager != nullptr ? g_session_manager->NumSessions() : 0U;
}

// Initialize GE, prepare for execution, call GELib::Initialize
Status GEInitialize(const std::map<std::string, std::string> &options) {
  if (IsGEInitialize()) {
    return SUCCESS;
  }
  std::map<AscendString, AscendString> str_options;
  for (const auto &option_item : options) {
    if (option_item.first.length() == 0) {
      GELOGE(FAILED, "[Check][Param] GEInitialize failed, option key is empty.");
      REPORT_INNER_ERR_MSG("E19999", "Check parameter's options invalid, option key is empty.");
      return FAILED;
    }
    const AscendString &key =
        AscendString(option_item.first.c_str(), option_item.first.length());
    const AscendString &val =
        AscendString(option_item.second.c_str(), option_item.second.length());
    str_options[key] = val;
  }
  return GEInitialize(str_options);
}

Status GEInitialize(const std::map<AscendString, AscendString> &options) {
  if (IsGEInitialize()) {
    return SUCCESS;
  }
  auto ret = GEInitializeV2(options);
  if (ret != SUCCESS) {
    GELOGE(ret, "[Init][GEInitializeV2] initial failed.");
    REPORT_INNER_ERR_MSG("E19999", "GEInitializeV2 initialize failed.");
    return ret;
  }
  ret = DFlowInitializeInner(options);
  if (ret != SUCCESS) {
    GELOGE(ret, "[Init][DFlowInitializeInner] initial failed.");
    REPORT_INNER_ERR_MSG("E19999", "DFlowInitializeInner initialize failed.");
    return ret;
  }
  GELOGI("sessionManager initial.");
  GE_TIMESTAMP_START(SessionManagerInitialize);
  g_session_manager = MakeShared<ge::SessionManager>();
  if (g_session_manager == nullptr) {
    GELOGE(GE_CLI_INIT_FAILED, "[Init][Create]GeSessionManager failed");
    return FAILED;
  }
  ret = g_session_manager->Initialize();
  GE_TIMESTAMP_EVENT_END(SessionManagerInitialize, "InnerInitialize::SessionManagerInitialize");
  return ret;
}

// GE finalize, releasing all resources
Status GEFinalize() {
  if (!IsGEInitialize()) {
    return SUCCESS;
  }
  GELOGI("SessionManager finalization.");
  if (g_session_manager != nullptr) {
    (void)g_session_manager->Finalize();  // always success.
  }
  DFlowFinalizeInner();
  return GEFinalizeV2();
}

std::string GEGetErrorMsg() {
  return std::string(error_message::GetErrMgrErrorMessage().get());
}

ge::AscendString GEGetErrorMsgV2() {
  return ge::AscendString(error_message::GetErrMgrErrorMessage().get());
}

std::string GEGetWarningMsg() {
  return std::string(error_message::GetErrMgrWarningMessage().get());
}

ge::AscendString GEGetWarningMsgV2() {
  return ge::AscendString(error_message::GetErrMgrWarningMessage().get());
}

// Initialize session，which calls innerSession
Session::Session(const std::map<std::string, std::string> &options) {
  ConstructSession(options, sessionId_);
}

Session::Session(const std::map<AscendString, AscendString> &options) {
  std::map<std::string, std::string> str_options;
  for (auto &option_item : options) {
    if (option_item.first.GetLength() == 0) {
      GELOGE(FAILED, "Construct session failed, option key is empty.");
      REPORT_INNER_ERR_MSG("E19999", "Construct session failed, option key is empty.");
      return;
    }
    const std::string &key = option_item.first.GetString();
    const std::string &val = option_item.second.GetString();
    str_options[key] = val;
  }
  ConstructSession(str_options, sessionId_);
}

// session destructor
Session::~Session() {
  GELOGT(TRACE_INIT, "Start to destroy session.");
  // 0.check init status
  if (!IsGEInitialize()) {
    GELOGW("GE is not yet initialized or is finalized.");
    return;
  }
  ExternalWeightManagerPool::Instance().RemoveManager(sessionId_);
  Status ret = FAILED;
  std::lock_guard<std::mutex> lock(g_ge_release_mutex);
  try {
    const uint64_t session_id = sessionId_;
    // call DestroySession
    GELOGT(TRACE_RUNNING, "Session id is %lu", session_id);
    ret = g_session_manager->DestroySession(session_id);
  } catch (std::exception &e) {
    (void)e;
    GELOGE(GE_CLI_SESS_DESTROY_FAILED, "[Destructor][Session]Failed: an exception occurred");
    REPORT_INNER_ERR_MSG("E19999", "Failed to destroy session: an exception occurred");
  }

  // check return status, return, update session id if success
  if (ret != SUCCESS) {
    GELOGE(ret, "[Destructor][Session]Failed, error code:%u.", ret);
    REPORT_INNER_ERR_MSG("E19999", "Destroy session failed, error code:%u.", ret);
  }

  GELOGT(TRACE_STOP, "Session has been successfully destroyed");
}

// Add Graph
Status Session::AddGraph(uint32_t graph_id, const Graph &graph) {
  std::map<AscendString, AscendString> options;
  return AddGraph(graph_id, graph, options);
}

// Add Graph
Status Session::AddGraph(uint32_t graph_id, const Graph &graph, const std::map<std::string, std::string> &options) {
  if (!IsGEInitialize()) {
    GELOGE(GE_CLI_GE_NOT_INITIALIZED, "[Construct][Session]Failed because lack GEInitialize call before.");
    REPORT_INNER_ERR_MSG("E19999", "Creating session failed because lack GEInitialize call before.");
    return FAILED;
  }

  if (GEAPICheckSupportedGraphOptions(options) != SUCCESS) {
    GELOGW("[Check][Param] Check supported options failed.");
  }
  GRAPH_PROFILING_REG(gert::GeProfInfoType::kAddGraph);
  AscendString graph_name;
  GE_ASSERT_SUCCESS(graph.GetName(graph_name), "Add graph failed, get graph name failed.");
  GELOGT(TRACE_INIT, "Start to add graph in Session. graph_id: %u, graph_name: %s, session_id: %lu.", graph_id,
         graph_name.GetString(), sessionId_);

  GELOGD("Adding graph to session, graph_id: %u", graph_id);

  Status ret = FAILED;
  const UserHybridGraphManagerPtr user_hybrid_graph_manager = g_session_manager->GetUserHybridGraphManager(sessionId_);
  GE_CHK_BOOL_RET_STATUS(user_hybrid_graph_manager != nullptr, FAILED, "Add graph failed, session_id:%lu.", sessionId_);
  const bool is_enable_slice_schedule = EnableSliceSchedule();
  ret = user_hybrid_graph_manager->AddGraph(graph_id, graph, options);
  GE_CHK_BOOL_RET_STATUS(ret == SUCCESS, FAILED, "Add graph failed, error code:%u, session_id:%lu, graph_id:%u.", ret,
                         sessionId_, graph_id);

  GELOGD("AddGraph finished in Session, graph_id: %u, is_enable_slice_schedule:%u", graph_id, is_enable_slice_schedule);
  return ret;
}

// Add Graph
Status Session::AddGraph(uint32_t graph_id, const Graph &graph, const std::map<AscendString, AscendString> &options) {
  if (!IsGEInitialize()) {
    GELOGE(GE_CLI_GE_NOT_INITIALIZED, "[Construct][Session]Failed because lack GEInitialize call before.");
    REPORT_INNER_ERR_MSG("E19999", "Creating session failed because lack GEInitialize call before.");
    return FAILED;
  }
  AscendString graph_name;
  GE_ASSERT_SUCCESS(graph.GetName(graph_name), "Add graph failed, get graph name failed.");
  GELOGT(TRACE_INIT, "Start to add graph in Session. graph_id: %u, graph_name: %s, session_id: %lu.", graph_id,
         graph_name.GetString(), sessionId_);

  std::map<std::string, std::string> str_options;
  for (auto &option_item : options) {
    if (option_item.first.GetLength() == 0) {
      GELOGE(FAILED, "Add graph failed, option key is empty.");
      REPORT_INNER_ERR_MSG("E19999", "Add graph failed, option key is empty.");
      return FAILED;
    }

    const std::string &key = option_item.first.GetString();
    const std::string &val = option_item.second.GetString();
    str_options[key] = val;
  }

  if (GEAPICheckSupportedGraphOptions(str_options) != SUCCESS) {
    GELOGW("[Check][Param] Check supported options failed.");
  }
  GELOGD("Adding graph to session");
  Status ret = FAILED;
  auto isEnableSliceSchedule = EnableSliceSchedule();
  const UserHybridGraphManagerPtr user_hybrid_graph_manager = g_session_manager->GetUserHybridGraphManager(sessionId_);
  GE_CHK_BOOL_RET_STATUS(user_hybrid_graph_manager != nullptr, FAILED, "Add graph failed, session_id:%lu.", sessionId_);
  ret = user_hybrid_graph_manager->AddGraph(graph_id, graph, str_options);
  GE_CHK_BOOL_RET_STATUS(ret == SUCCESS, FAILED, "Add graph failed, error code:%u, session_id:%lu, graph_id:%u.", ret,
                         sessionId_, graph_id);

  GELOGD("AddGraph finished in Session, graph_id: %u, isEnableSliceSchedule:%u", graph_id, isEnableSliceSchedule);
  return SUCCESS;
}

Status Session::AddGraphWithCopy(uint32_t graph_id, const Graph &graph) {
  const std::map<AscendString, AscendString> options;
  return AddGraphWithCopy(graph_id, graph, options);
}

// Add Graph With Copy
Status Session::AddGraphWithCopy(uint32_t graph_id, const Graph &graph,
                                 const std::map<AscendString, AscendString> &options) {
  if (!IsGEInitialize()) {
    GELOGE(GE_CLI_GE_NOT_INITIALIZED, "[Construct][Session]Failed because lack GEInitialize call before.");
    REPORT_INNER_ERR_MSG("E19999", "Creating session failed because lack GEInitialize call before.");
    return FAILED;
  }

  const SessionPtr inner_session = g_session_manager->GetSession(sessionId_);
  GE_CHK_BOOL_RET_STATUS(inner_session != nullptr, FAILED, "Add graph failed, session_id:%lu.", sessionId_);

  AscendString graph_name;
  GE_ASSERT_SUCCESS(graph.GetName(graph_name), "Add graph failed, get graph name failed.");
  GELOGT(TRACE_INIT, "Start to add graph in Session. graph_id: %u, graph_name: %s, session_id: %lu.", graph_id,
         graph_name.GetString(), sessionId_);

  std::map<std::string, std::string> str_options;
  for (auto it = options.begin(); it != options.end(); ++it) {
    (void)str_options.emplace(it->first.GetString(), it->second.GetString());
  }

  if (GEAPICheckSupportedGraphOptions(str_options) != SUCCESS) {
    GELOGW("[Check][Param] Check supported options failed.");
  }

  GELOGD("Adding graph to session");
  const Status ret = inner_session->AddGraphWithCopy(graph_id, graph, str_options);
  GE_CHK_BOOL_RET_STATUS(ret == SUCCESS, FAILED, "Add graph failed, error code:%u, session_id:%lu, graph_id:%u.", ret,
                         sessionId_, graph_id);

  GELOGD("AddGraph finished in Session.");
  return ret;
}

// Remove Graph
Status Session::RemoveGraph(uint32_t graph_id) {
  if (!IsGEInitialize()) {
    GELOGE(GE_CLI_GE_NOT_INITIALIZED, "[Construct][Session]Failed because lack GEInitialize call before.");
    REPORT_INNER_ERR_MSG("E19999", "Creating session failed because lack GEInitialize call before.");
    return FAILED;
  }
  
  GRAPH_PROFILING_REG(gert::GeProfInfoType::kRemoveGraph);
  GELOGT(TRACE_INIT, "Session RemoveGraph start, graph_id: %u", graph_id);

  // call RemoveGraph
  Status ret = FAILED;
  auto isEnableSliceSchedule = EnableSliceSchedule();
  const UserHybridGraphManagerPtr user_hybrid_graph_manager = g_session_manager->GetUserHybridGraphManager(sessionId_);
  GE_CHK_BOOL_RET_STATUS(user_hybrid_graph_manager != nullptr, FAILED, "Remove graph failed, session_id:%lu.", sessionId_);
  ret = user_hybrid_graph_manager->RemoveGraph(graph_id);
  GE_CHK_BOOL_RET_STATUS(ret == SUCCESS, FAILED, "Remove graph failed, error code:%u, session_id:%lu, graph_id:%u.",
                         ret, sessionId_, graph_id);

  GELOGT(TRACE_STOP, "Session RemoveGraph finished, graph_id: %u, isEnableSliceSchedule:%u", graph_id, isEnableSliceSchedule);
  return ret;
}

// Print Output Result
static void PrintOutputResult(std::vector<Tensor> &outputs) {
  if (outputs.empty() || (outputs[0].GetData() == nullptr)) {
    GELOGW("outputs is empty or data is nullptr.");
    return;
  }

  const DataType data_type = outputs[0].GetTensorDesc().GetDataType();
  if (CONST_OPDATA_TYPE_SIZE_MAP.find(data_type) == CONST_OPDATA_TYPE_SIZE_MAP.end()) {
    GELOGI("DataType %s has not defined size", TypeUtils::DataTypeToSerialString(data_type).c_str());
    return;
  }
  // take first 10 at most
  for (size_t i = 0UL; (i < 10UL) && (i < (outputs[0].GetSize() / CONST_OPDATA_TYPE_SIZE_MAP[data_type])); ++i) {
    switch (data_type) {
      case DT_BOOL:
      case DT_INT8:
      case DT_UINT8:
      case DT_HIFLOAT8: case DT_FLOAT8_E5M2: case DT_FLOAT8_E4M3FN:
      case DT_FLOAT8_E8M0: case DT_FLOAT6_E3M2: case DT_FLOAT6_E2M3:
      case DT_FLOAT4_E2M1: case DT_FLOAT4_E1M2:
        GELOGI("output data[%zu]=%d", i, *(reinterpret_cast<int8_t *>(outputs[0].GetData()) + i));
        break;
      case DT_INT16:
      case DT_UINT16:
        GELOGI("output data[%zu]=%d", i, *(reinterpret_cast<int16_t *>(outputs[0].GetData()) + i));
        break;
      case DT_INT32:
      case DT_UINT32:
        GELOGI("output data[%zu]=%d", i, *(reinterpret_cast<int32_t *>(outputs[0].GetData()) + i));
        break;
      case DT_INT64:
      case DT_UINT64:
        GELOGI("output data[%zu]=%ld", i, *(reinterpret_cast<int64_t *>(outputs[0].GetData()) + i));
        break;
      case DT_FLOAT:
        GELOGI("output data[%zu]=%f", i, *(reinterpret_cast<float *>(outputs[0].GetData()) + i));
        break;
      case DT_DOUBLE:
        GELOGI("output data[%zu]=%lf", i, *(reinterpret_cast<double *>(outputs[0].GetData()) + i));
        break;
      default:
        GELOGI("Output datatype %s is not supported.", TypeUtils::DataTypeToSerialString(data_type).c_str());
        return;
    }
  }
}

// Run Graph
Status Session::RunGraph(uint32_t graph_id, const std::vector<Tensor> &inputs, std::vector<Tensor> &outputs) {
  if (!IsGEInitialize()) {
    GELOGE(GE_CLI_GE_NOT_INITIALIZED, "[Construct][Session]Failed because lack GEInitialize call before.");
    REPORT_INNER_ERR_MSG("E19999", "Creating session failed because lack GEInitialize call before.");
    return FAILED;
  }

  const SessionPtr inner_session = g_session_manager->GetSession(sessionId_);
  GE_CHK_BOOL_RET_STATUS(inner_session != nullptr, FAILED, "Run graph failed, session_id:%lu.", sessionId_);

  GELOGI("Session RunGraph start, session_id: %lu, graph_id: %u, input size %zu, output size %zu",
         sessionId_, graph_id, inputs.size(), outputs.size());


  // call RunGraph
  Status ret = inner_session->RunGraph(graph_id, inputs, outputs);
  // check return status
  const bool need_convert_error_code = (ret == RT_ERROR_TO_GE_STATUS(ACL_ERROR_RT_QUEUE_EMPTY));
  ret = need_convert_error_code ? ACL_ERROR_GE_MODEL_EXECUTE_TIMEOUT : ret;
  const auto status = ret > kExternalErrorCodeMaxValue ? FAILED : ret;
  GE_CHK_BOOL_RET_STATUS(ret == SUCCESS, status,
                         "Run graph failed, error code:%u, session_id:%lu, graph_id:%u.", ret, sessionId_, graph_id);

  // print output
  if (outputs.size() > 0) {
    PrintOutputResult(outputs);
  }

  // return
  GELOGI("Session RunGraph finished");
  return ret;
}

// Run Graph with stream Asynchronously
Status Session::RunGraphWithStreamAsync(uint32_t graph_id, void *stream, const std::vector<Tensor> &inputs,
                                        std::vector<Tensor> &outputs) {
  if (!IsGEInitialize()) {
    GELOGE(GE_CLI_GE_NOT_INITIALIZED, "[Construct][Session]Failed because lack GEInitialize call before.");
    REPORT_INNER_ERR_MSG("E19999", "Creating session failed because lack GEInitialize call before.");
    return FAILED;
  }

  const SessionPtr inner_session = g_session_manager->GetSession(sessionId_);
  GE_CHK_BOOL_RET_STATUS(inner_session != nullptr, FAILED, "Run graph with stream async failed, session_id:%lu.",
                         sessionId_);


  const Status ret = inner_session->RunGraphWithStreamAsync(graph_id, stream, inputs, outputs);
  GE_CHK_BOOL_RET_STATUS(ret == SUCCESS, FAILED,
                         "Run graph with stream async failed, error code:%u, session_id:%lu, graph_id:%u, stream:%p.",
                         ret, sessionId_, graph_id, stream);

  GELOGI("Session run graph with stream async finished.");
  return SUCCESS;
}

Status Session::ExecuteGraphWithStreamAsync(uint32_t graph_id, void *stream, const std::vector<gert::Tensor> &inputs,
                                        std::vector<gert::Tensor> &outputs) {
  RT2_PROFILING_SCOPE(gert::profiling::kUnknownName, gert::profiling::kStaticGraphExecute);
  if (!IsGEInitialize()) {
    GELOGE(GE_CLI_GE_NOT_INITIALIZED, "[Construct][Session]Failed because lack GEInitialize call before.");
    REPORT_INNER_ERR_MSG("E19999", "Creating session failed because lack GEInitialize call before.");
    return FAILED;
  }

  Status ret = FAILED;
  auto is_enable_slice_schedule = EnableSliceSchedule();
  const UserGraphsManagerPtr user_graphs_manager = g_session_manager->GetUserGraphsManager(sessionId_);
  GE_CHK_BOOL_RET_STATUS(user_graphs_manager != nullptr, FAILED, "Execute graph with stream async failed, session_id:%lu.",
                         sessionId_);
  ret = user_graphs_manager->ExecuteGraphWithStreamAsync(graph_id, stream, inputs, outputs);
  GE_CHK_BOOL_RET_STATUS(ret == SUCCESS, FAILED,
                         "Execute graph with stream async failed, error code:%u, session_id:%lu, graph_id:%u, "
                         "stream:%p, is_enable_slice_schedule:%d",
                         ret, sessionId_, graph_id, stream, is_enable_slice_schedule);
  return SUCCESS;
}

// Register Call Back
Status Session::RegisterCallBackFunc(const std::string &key, const pCallBackFunc &callback) {
  if (!IsGEInitialize()) {
    GELOGE(GE_CLI_GE_NOT_INITIALIZED, "[Construct][Session]Failed because lack GEInitialize call before.");
    REPORT_INNER_ERR_MSG("E19999", "Creating session failed because lack GEInitialize call before.");
    return FAILED;
  }

  const SessionPtr inner_session = g_session_manager->GetSession(sessionId_);
  GE_CHK_BOOL_RET_STATUS(inner_session != nullptr, FAILED, "[Get][Session] failed, session_id:%lu.", sessionId_);

  return inner_session->RegisterCallBackFunc(key, callback);
}

Status Session::RegisterCallBackFunc(const char *key, const session::pCallBackFunc &callback) {
  if (!IsGEInitialize()) {
    GELOGE(GE_CLI_GE_NOT_INITIALIZED, "[Construct][Session]Failed because lack GEInitialize call before.");
    REPORT_INNER_ERR_MSG("E19999", "Creating session failed because lack GEInitialize call before.");
    return FAILED;
  }

  std::string str_key;
  if (key != nullptr) {
    str_key = key;
  }

  const SessionPtr inner_session = g_session_manager->GetSession(sessionId_);
  GE_CHK_BOOL_RET_STATUS(inner_session != nullptr, FAILED, "[Get][Session] failed, session_id:%lu.", sessionId_);

  return inner_session->RegisterCallBackFunc(str_key, callback);
}

// Build Graph
Status Session::BuildGraph(uint32_t graph_id, const std::vector<InputTensorInfo> &inputs) {
  if (!IsGEInitialize()) {
    GELOGE(GE_CLI_GE_NOT_INITIALIZED, "[Construct][Session]Failed because lack GEInitialize call before.");
    REPORT_INNER_ERR_MSG("E19999", "Creating session failed because lack GEInitialize call before.");
    return FAILED;
  }

  const SessionPtr inner_session = g_session_manager->GetSession(sessionId_);
  GE_CHK_BOOL_RET_STATUS(inner_session != nullptr, FAILED, "Build graph failed, session_id:%lu.", sessionId_);

  GELOGT(TRACE_INIT, "start to build graph, session_id: %lu, graph_id: %u, input size %zu",
         sessionId_, graph_id, inputs.size());

  const Status ret = inner_session->BuildGraph(graph_id, inputs);
  GE_CHK_BOOL_RET_STATUS(ret == SUCCESS, FAILED, "Build graph failed, error code:%u, session_id:%lu, graph_id:%u.", ret,
                         sessionId_, graph_id);
  return SUCCESS;
}

Status Session::LoadGraph(const uint32_t graph_id, const std::map<AscendString, AscendString> &options,
                          void *stream) const {
  GELOGD("Loading graph to session, graph_id: %u, session_id: %u, stream:%p .",
         graph_id, sessionId_, stream);
  if (!IsGEInitialize()) {
    GELOGE(GE_CLI_GE_NOT_INITIALIZED, "[Construct][Session]Failed because lack GEInitialize call before.");
    REPORT_INNER_ERR_MSG("E19999", "Creating session failed because lack GEInitialize call before.");
    return FAILED;
  }

  Status ret = FAILED;
  const UserGraphsManagerPtr user_graphs_manager = g_session_manager->GetUserGraphsManager(sessionId_);
  GE_CHK_BOOL_RET_STATUS(user_graphs_manager != nullptr, FAILED, "Load graph failed, session_id:%lu.", sessionId_);
  auto is_enable_slice_schedule = EnableSliceSchedule();
  ret = user_graphs_manager->LoadGraph(graph_id, options, stream);

  GE_CHK_BOOL_RET_STATUS(ret == SUCCESS, FAILED,
                         "Load graph failed, error code:%u, session_id:%lu, graph_id:%u, is_enable_slice_schedule:%d",
                         ret, sessionId_, graph_id, is_enable_slice_schedule);

  return ret;
}

// Build Graph
Status Session::BuildGraph(uint32_t graph_id, const std::vector<ge::Tensor> &inputs) {
  if (!IsGEInitialize()) {
    GELOGE(GE_CLI_GE_NOT_INITIALIZED, "[Construct][Session]Failed because lack GEInitialize call before.");
    REPORT_INNER_ERR_MSG("E19999", "Creating session failed because lack GEInitialize call before.");
    return FAILED;
  }

  GRAPH_PROFILING_REG(gert::GeProfInfoType::kBuildGraph);
  GELOGT(TRACE_INIT, "start to build graph, session_id: %lu, graph_id: %u, input size %zu",
         sessionId_, graph_id, inputs.size());

  Status ret = FAILED;
  auto isEnableSliceSchedule = EnableSliceSchedule();
  const UserHybridGraphManagerPtr user_hybrid_graph_manager = g_session_manager->GetUserHybridGraphManager(sessionId_);
  GE_CHK_BOOL_RET_STATUS(user_hybrid_graph_manager != nullptr, FAILED, "Build graph failed, session_id:%lu.", sessionId_);
  ret = user_hybrid_graph_manager->BuildGraph(graph_id, inputs);
  GE_CHK_BOOL_RET_STATUS(ret == SUCCESS, FAILED, "Build graph failed, error code:%u, session_id:%lu, graph_id:%u.", ret,
                         sessionId_, graph_id);
  GELOGD("BuildGraph finished in Session, graph_id: %u, isEnableSliceSchedule:%u", graph_id, isEnableSliceSchedule);
  return SUCCESS;
}

// Run Graph Asynchronously
Status Session::RunGraphAsync(uint32_t graph_id, const std::vector<ge::Tensor> &inputs,
                              RunAsyncCallback callback) {
  if (!IsGEInitialize()) {
    GELOGE(GE_CLI_GE_NOT_INITIALIZED, "[Construct][Session]Failed because lack GEInitialize call before.");
    REPORT_INNER_ERR_MSG("E19999", "Creating session failed because lack GEInitialize call before.");
    return FAILED;
  }
  GRAPH_PROFILING_REG(gert::GeProfInfoType::kRunGraphAsync);
  GELOGI("start to run graph async, session_id: %lu, graph_id: %u, input size %zu",
         sessionId_, graph_id, inputs.size());

  GELOGI("The callback function will not be checked. Please ensure that the implementation of the function is trusted,"
      " graph_id: %u", graph_id);

  const uint64_t session_id = sessionId_;
  auto callback_wrapper = [session_id, graph_id, callback](Status ret, std::vector<ge::Tensor> &outputs) -> void {
    RunGraphAsyncCallback(ret, session_id, graph_id, outputs, callback);
  };

  Status ret = FAILED;
  auto isEnableSliceSchedule = EnableSliceSchedule();
  const UserHybridGraphManagerPtr user_hybrid_graph_manager = g_session_manager->GetUserHybridGraphManager(sessionId_);
  GE_CHK_BOOL_RET_STATUS(user_hybrid_graph_manager != nullptr, FAILED, "Run graph async failed, session_id:%lu.", sessionId_);
  ret = user_hybrid_graph_manager->RunGraphAsync(graph_id, inputs, callback_wrapper);
  GE_CHK_BOOL_RET_STATUS(ret == SUCCESS, FAILED, "Run graph async failed, error code:%u, session_id:%lu, graph_id:%u.",
                         ret, sessionId_, graph_id);
  GELOGD("RunGraphAsync finished in Session, graph_id: %u, isEnableSliceSchedule:%u", graph_id, isEnableSliceSchedule);
  return SUCCESS;
}

// Get Variables
Status Session::GetVariables(const std::vector<std::string> &var_names, std::vector<Tensor> &var_values) {
  if (!IsGEInitialize()) {
    GELOGE(GE_CLI_GE_NOT_INITIALIZED, "[Construct][Session]Failed because lack GEInitialize call before.");
    REPORT_INNER_ERR_MSG("E19999", "Creating session failed because lack GEInitialize call before.");
    return FAILED;
  }

  GELOGI("Get Variables");
  const Status ret = g_session_manager->GetVariables(sessionId_, var_names, var_values);
  if (ret != SUCCESS) {
    GELOGE(ret, "[Get][Variables]Failed, error code:%u, session_id:%lu.", ret, sessionId_);
    return FAILED;
  }
  return SUCCESS;
}

// Get Variables
Status Session::GetVariables(const std::vector<AscendString> &var_names, std::vector<Tensor> &var_values) {
  if (!IsGEInitialize()) {
    GELOGE(GE_CLI_GE_NOT_INITIALIZED, "[Construct][Session]Failed because lack GEInitialize call before.");
    REPORT_INNER_ERR_MSG("E19999", "Creating session failed because lack GEInitialize call before.");
    return FAILED;
  }

  GELOGI("Get Variables");
  std::vector<std::string> str_var_names;
  for (auto &var_name : var_names) {
    if (var_name.GetString() == nullptr) {
      GELOGE(FAILED, "[Get][Variable]Failed, variables' names are nullptr.");
      REPORT_INNER_ERR_MSG("E19999", "GetVariables failed, variables' names are nullptr.");
      return FAILED;
    }
    str_var_names.emplace_back(var_name.GetString());
  }
  const Status ret = g_session_manager->GetVariables(sessionId_, str_var_names, var_values);
  if (ret != SUCCESS) {
    GELOGE(ret, "[Get][Variables]Failed, error code:%u, session_id:%lu.", ret, sessionId_);
    REPORT_INNER_ERR_MSG("E19999", "Get variables failed, error code:%u, session_id:%lu.", ret, sessionId_);
    return FAILED;
  }
  return SUCCESS;
}

bool Session::IsGraphNeedRebuild(uint32_t graph_id) {
  GRAPH_PROFILING_REG(gert::GeProfInfoType::kIsGraphNeedRebuild);
  if (!IsGEInitialize()) {
    GELOGE(GE_CLI_GE_NOT_INITIALIZED, "[Construct][Session]Failed because lack GEInitialize call before.");
    REPORT_INNER_ERR_MSG("E19999", "Creating session failed because lack GEInitialize call before.");
    return false;
  }

  const UserHybridGraphManagerPtr user_hybrid_graph_manager = g_session_manager->GetUserHybridGraphManager(sessionId_);
  GE_CHK_BOOL_RET_STATUS(user_hybrid_graph_manager != nullptr, FAILED, "Add graph failed, session_id:%lu.", sessionId_);
  return user_hybrid_graph_manager->IsGraphNeedRebuild(graph_id);
}

uint64_t Session::GetSessionId() const {
  return sessionId_;
}

Status Session::FeedDataFlowGraph(uint32_t graph_id, const std::vector<Tensor> &inputs, const DataFlowInfo &info,
                                  int32_t timeout) {
  return FeedDataFlowGraph(graph_id, {}, inputs, info, timeout);
}

Status Session::FeedDataFlowGraph(uint32_t graph_id, const std::vector<uint32_t> &indexes,
                                  const std::vector<Tensor> &inputs, const DataFlowInfo &info, int32_t timeout) {
  if (!IsGEInitialize()) {
    GELOGE(GE_CLI_GE_NOT_INITIALIZED, "[Feed][Data]Failed because lack GEInitialize call before.");
    REPORT_INNER_ERR_MSG("E19999", "Feed data failed because lack GEInitialize call before.");
    return FAILED;
  }

  const SessionPtr inner_session = g_session_manager->GetSession(sessionId_);
  GE_CHK_BOOL_RET_STATUS(inner_session != nullptr, FAILED, "[Get][Session] failed, session_id:%lu.", sessionId_);


  GELOGI("Feed data flow graph, graph_id: %u, timeout: %d ms", graph_id, timeout);
  const Status ret = inner_session->FeedDataFlowGraph(graph_id, indexes, inputs, info, timeout);
  if (ret != SUCCESS && ret != ACL_ERROR_GE_REDEPLOYING && ret != ACL_ERROR_GE_SUBHEALTHY) {
    GELOGE(ret, "[Feed][Data]Failed, error code:%u, session_id:%lu, graph_id:%u.", ret, sessionId_, graph_id);
    REPORT_INNER_ERR_MSG("E19999", "Feed data flow graph failed , error code:%u, session_id:%lu, graph_id:%u", ret,
                      sessionId_, graph_id);
    return (ret > kExternalErrorCodeMaxValue) ? FAILED : ret;
  }
  return ret;
}

Status Session::FeedDataFlowGraph(uint32_t graph_id, const std::vector<FlowMsgPtr> &inputs, int32_t timeout) {
  return FeedDataFlowGraph(graph_id, {}, inputs, timeout);
}

Status Session::FeedDataFlowGraph(uint32_t graph_id, const std::vector<uint32_t> &indexes,
                                  const std::vector<FlowMsgPtr> &inputs, int32_t timeout) {
  GE_CHK_BOOL_RET_STATUS(IsGEInitialize(), FAILED,
                         "[Feed][FlowMsg]Failed because lack GEInitialize call before.");

  const SessionPtr inner_session = g_session_manager->GetSession(sessionId_);
  GE_CHK_BOOL_RET_STATUS(inner_session != nullptr, FAILED, "[Get][Session] failed, session_id:%lu.", sessionId_);


  GELOGI("Feed flow msg, graph_id: %u, timeout: %d ms", graph_id, timeout);
  const Status ret = inner_session->FeedDataFlowGraph(graph_id, indexes, inputs, timeout);
  const auto status = ret > kExternalErrorCodeMaxValue ? FAILED : ret;
  GE_CHK_BOOL_RET_STATUS((ret == SUCCESS || ret == ACL_ERROR_GE_REDEPLOYING || ret == ACL_ERROR_GE_SUBHEALTHY),
                         status, "[Feed][FlowMsg]Failed, error code:%u, session_id:%lu, graph_id:%u.",
                         ret, sessionId_, graph_id);
  return ret;
}

Status Session::FeedRawData(uint32_t graph_id, const std::vector<RawData> &raw_data_list, const uint32_t index,
                            const DataFlowInfo &info, int32_t timeout) {
  if (!IsGEInitialize()) {
    GELOGE(GE_CLI_GE_NOT_INITIALIZED, "[Feed][RawData]Failed because lack GEInitialize call before.");
    REPORT_INNER_ERR_MSG("E19999", "Feed raw data failed because lack GEInitialize call before.");
    return FAILED;
  }

  const SessionPtr inner_session = g_session_manager->GetSession(sessionId_);
  GE_CHK_BOOL_RET_STATUS(inner_session != nullptr, FAILED, "[Get][Session] failed, session_id:%lu.", sessionId_);


  GELOGI("Feed raw data to data flow graph, graph_id: %u, timeout: %d ms", graph_id, timeout);
  const Status ret = inner_session->FeedRawData(graph_id, raw_data_list, index, info, timeout);
  if (ret != SUCCESS && ret != ACL_ERROR_GE_REDEPLOYING && ret != ACL_ERROR_GE_SUBHEALTHY) {
    GELOGE(ret, "[Feed][Data]Failed, error code:%u, session_id:%lu, graph_id:%u.", ret, sessionId_, graph_id);
    REPORT_INNER_ERR_MSG("E19999", "Feed data flow graph failed , error code:%u, session_id:%lu, graph_id:%u", ret,
                      sessionId_, graph_id);
    return (ret > kExternalErrorCodeMaxValue) ? FAILED : ret;
  }
  return ret;
}

Status Session::FetchDataFlowGraph(uint32_t graph_id, std::vector<Tensor> &outputs, DataFlowInfo &info,
                                   int32_t timeout) {
  return FetchDataFlowGraph(graph_id, {}, outputs, info, timeout);
}

Status Session::FetchDataFlowGraph(uint32_t graph_id, const std::vector<uint32_t> &indexes,
                                   std::vector<Tensor> &outputs, DataFlowInfo &info, int32_t timeout) {
  if (!IsGEInitialize()) {
    GELOGE(GE_CLI_GE_NOT_INITIALIZED, "[Fetch][Data]Failed because lack GEInitialize call before.");
    REPORT_INNER_ERR_MSG("E19999", "Fetch data failed because lack GEInitialize call before.");
    return FAILED;
  }

  const SessionPtr inner_session = g_session_manager->GetSession(sessionId_);
  GE_CHK_BOOL_RET_STATUS(inner_session != nullptr, FAILED, "[Get][Session] failed, session_id:%lu.", sessionId_);


  GELOGI("Fetch data flow graph, graph_id: %u, timeout: %d ms", graph_id, timeout);
  Status ret = inner_session->FetchDataFlowGraph(graph_id, indexes, outputs, info, timeout);
  const bool need_convert_error_code = ((ret == RT_ERROR_TO_GE_STATUS(ACL_ERROR_RT_QUEUE_EMPTY)) && timeout != 0);
  ret = need_convert_error_code ? ACL_ERROR_GE_MODEL_EXECUTE_TIMEOUT : ret;
  if (ret != SUCCESS && ret != ACL_ERROR_GE_REDEPLOYING && ret != ACL_ERROR_GE_SUBHEALTHY) {
    GELOGE(ret, "[Fetch][Data]Failed, error code:%u, session_id:%lu, graph_id:%u.", ret, sessionId_, graph_id);
    REPORT_INNER_ERR_MSG("E19999", "Fetch data flow graph failed , error code:%u, session_id:%lu, graph_id:%u", ret,
                      sessionId_, graph_id);
    return (ret > kExternalErrorCodeMaxValue) ? FAILED : ret;
  }
  return ret;
}

Status Session::FetchDataFlowGraph(uint32_t graph_id, std::vector<FlowMsgPtr> &outputs, int32_t timeout) {
  return FetchDataFlowGraph(graph_id, {}, outputs, timeout);
}

Status Session::FetchDataFlowGraph(uint32_t graph_id, const std::vector<uint32_t> &indexes,
                                   std::vector<FlowMsgPtr> &outputs, int32_t timeout) {
  GE_CHK_BOOL_RET_STATUS(IsGEInitialize(), FAILED,
                         "[Fetch][FlowMsg]Failed because lack GEInitialize call before.");

  const SessionPtr inner_session = g_session_manager->GetSession(sessionId_);
  GE_CHK_BOOL_RET_STATUS(inner_session != nullptr, FAILED, "[Get][Session] failed, session_id:%lu.", sessionId_);


  GELOGI("Fetch flow msg, graph_id: %u, timeout: %d ms", graph_id, timeout);
  Status ret = inner_session->FetchDataFlowGraph(graph_id, indexes, outputs, timeout);
  const bool need_convert_error_code = ((ret == RT_ERROR_TO_GE_STATUS(ACL_ERROR_RT_QUEUE_EMPTY)) && timeout != 0);
  ret = need_convert_error_code ? ACL_ERROR_GE_MODEL_EXECUTE_TIMEOUT : ret;
  const auto status = ret > kExternalErrorCodeMaxValue ? FAILED : ret;
  GE_CHK_BOOL_RET_STATUS((ret == SUCCESS || ret == ACL_ERROR_GE_REDEPLOYING || ret == ACL_ERROR_GE_SUBHEALTHY),
                         status, "[Fetch][FlowMsg]Failed, error code:%u, session_id:%lu, graph_id:%u.",
                         ret, sessionId_, graph_id);
  return ret;
}

Status Session::CompileGraph(uint32_t graph_id) {
  GE_ASSERT(IsGEInitialize(), "[Construct][Session]Failed because lack GEInitialize call before.");
  GELOGT(TRACE_INIT, "Start to compile graph, graph_id: %u", graph_id);

  Status ret = FAILED;
  auto is_enable_slice_schedule = EnableSliceSchedule();
  const UserGraphsManagerPtr user_graphs_manager = g_session_manager->GetUserGraphsManager(sessionId_);
  GE_ASSERT_NOTNULL(user_graphs_manager, "[Get][User Graph]Failed, session_id:%lu.", sessionId_);
  ret = user_graphs_manager->CompileGraph(graph_id);

  GE_ASSERT_SUCCESS(
      ret,
      "[Compile][Graph]Compile graph failed, error code:%u, session_id:%lu, graph_id:%u, is_enable_slice_schedule:%d",
      ret, sessionId_, graph_id, is_enable_slice_schedule);
  GELOGT(TRACE_STOP, "Compile graph success, graph_id: %u.", graph_id);
  return SUCCESS;
}

CompiledGraphSummaryPtr Session::GetCompiledGraphSummary(uint32_t graph_id) {
  GE_ASSERT(IsGEInitialize(), "[Construct][Session]Failed because lack GEInitialize call before.");
  CompiledGraphSummaryPtr summary = nullptr;
  Status ret = FAILED;
  
  const UserGraphsManagerPtr user_graphs_manager = g_session_manager->GetUserGraphsManager(sessionId_);
  GE_ASSERT_NOTNULL(user_graphs_manager, "[Get][User Graph]Failed, session_id:%lu.", sessionId_);
  auto is_enable_slice_schedule = EnableSliceSchedule();
  ret = user_graphs_manager->GetCompiledGraphSummary(graph_id, summary);
  GE_ASSERT_SUCCESS(ret,
                    "[Get][Summary]Failed, error code:%u, session_id:%lu, graph_id:%u, is_enable_slice_schedule:%d",
                    ret, sessionId_, graph_id, is_enable_slice_schedule);
  return summary;
}

Status Session::SetGraphConstMemoryBase(uint32_t graph_id, const void *const memory, size_t size) {
  GE_ASSERT(IsGEInitialize(), "[Construct][Session]Failed because lack GEInitialize call before.");
  if (EnableSliceSchedule()) {
    GELOGE(UNSUPPORTED, "[Construct][Session]SetGraphConstMemoryBase unsupport slice scheduler currently, session_id:%lu, graph_id:%u, memory:%p, size:%zu",
           sessionId_, graph_id, memory, size);
    REPORT_INNER_ERR_MSG("E19999", "SetGraphConstMemoryBase unsupport slice scheduler currently, session_id:%lu, graph_id:%u, memory:%p, size:%zu",
                       sessionId_, graph_id, memory, size);
    return UNSUPPORTED;
  }

  const auto inner_session = g_session_manager->GetSession(sessionId_);
  GE_ASSERT_NOTNULL(inner_session, "[Get][Session]Failed, session_id:%lu.", sessionId_);


  const auto ret = inner_session->SetGraphConstMemoryBase(graph_id, memory, size);
  GE_ASSERT_SUCCESS(ret, "[Set][Memory]Failed, error code:%u, session_id:%lu, graph_id:%u, memory:%p, size:%zu",
                    ret, sessionId_, graph_id, memory, size);
  return SUCCESS;
}

Status Session::UpdateGraphFeatureMemoryBase(uint32_t graph_id, const void *const memory, size_t size) {
  GE_ASSERT(IsGEInitialize(), "[Construct][Session]Failed because lack GEInitialize call before.");
  if (EnableSliceSchedule()) {
    GELOGE(UNSUPPORTED, "[Construct][Session]UpdateGraphFeatureMemoryBase unsupport slice scheduler currently, session_id:%lu, graph_id:%u, memory:%p, size:%zu",
           sessionId_, graph_id, memory, size);
    REPORT_INNER_ERR_MSG("E19999", "UpdateGraphFeatureMemoryBase unsupport slice scheduler currently, session_id:%lu, graph_id:%u, memory:%p, size:%zu",
                       sessionId_, graph_id, memory, size);
    return UNSUPPORTED;
  }
  const auto inner_session = g_session_manager->GetSession(sessionId_);
  GE_ASSERT_NOTNULL(inner_session, "[Get][Session]Failed, session_id:%lu.", sessionId_);


  const auto ret = inner_session->UpdateGraphFeatureMemoryBase(graph_id, memory, size);
  GE_ASSERT_SUCCESS(ret, "[Update][Memory]Failed, error code:%u, session_id:%lu, graph_id:%u, memory:%p, size:%zu",
                    ret, sessionId_, graph_id, memory, size);
  return SUCCESS;
}

Status Session::SetGraphFixedFeatureMemoryBase(uint32_t graph_id, const void *const memory, size_t size) {
  return SetGraphFixedFeatureMemoryBaseWithType(graph_id, MemoryType::MEMORY_TYPE_DEFAULT, memory, size);
}

Status Session::SetGraphFixedFeatureMemoryBaseWithType(uint32_t graph_id, MemoryType type, const void *const memory,
                                                       size_t size) {
  GE_ASSERT(IsGEInitialize(), "[Construct][Session]Failed because lack GEInitialize call before.");
  if (EnableSliceSchedule()) {
    GELOGE(UNSUPPORTED, "[Construct][Session]SetGraphFixedFeatureMemoryBaseWithType unsupport slice scheduler currently, session_id:%lu, graph_id:%u, type:%d, memory:%p, size:%zu",
           sessionId_, graph_id, type, memory, size);
    REPORT_INNER_ERR_MSG("E19999", "SetGraphFixedFeatureMemoryBaseWithType unsupport slice scheduler currently, session_id:%lu, graph_id:%u, memory:%p, size:%zu",
                       sessionId_, graph_id, memory, size);
    return UNSUPPORTED;
  }
  const auto inner_session = g_session_manager->GetSession(sessionId_);
  GE_ASSERT_NOTNULL(inner_session, "[Get][Session]Failed, session_id:%lu.", sessionId_);


  const auto ret = inner_session->SetGraphFixedFeatureMemoryBase(graph_id, type, memory, size);
  GE_ASSERT_SUCCESS(ret, "[Set][Memory]Failed, error code:%u, session_id:%lu, graph_id:%u, type:%d,"
                    " memory:%p, size:%zu", ret, sessionId_, graph_id, type, memory, size);
  return SUCCESS;
}

Status Session::UpdateGraphRefreshableFeatureMemoryBase(uint32_t graph_id, const void *const memory, size_t size) {
  GE_ASSERT(IsGEInitialize(), "[Construct][Session]Failed because lack GEInitialize call before.");
  if (EnableSliceSchedule()) {
    GELOGE(UNSUPPORTED, "[Construct][Session]UpdateGraphRefreshableFeatureMemoryBase unsupport slice scheduler currently, session_id:%lu, graph_id:%u, memory:%p, size:%zu",
           sessionId_, graph_id, memory, size);
    REPORT_INNER_ERR_MSG("E19999", "UpdateGraphRefreshableFeatureMemoryBase unsupport slice scheduler currently, session_id:%lu, graph_id:%u, memory:%p, size:%zu",
                       sessionId_, graph_id, memory, size);
    return UNSUPPORTED;
  }
  const auto inner_session = g_session_manager->GetSession(sessionId_);
  GE_ASSERT_NOTNULL(inner_session, "[Get][Session]Failed, session_id:%lu.", sessionId_);


  const auto ret = inner_session->UpdateGraphRefreshableFeatureMemoryBase(graph_id, memory, size);
  GE_ASSERT_SUCCESS(ret, "[Update][Memory]Failed, error code:%u, session_id:%lu, graph_id:%u, memory:%p, size:%zu",
                    ret, sessionId_, graph_id, memory, size);
  return SUCCESS;
}

Status Session::RegisterExternalAllocator(const void *const stream, AllocatorPtr allocator) const {
  GE_ASSERT(IsGEInitialize(), "[Construct][Session]Failed because lack GEInitialize call before.");
  const SessionPtr inner_session = g_session_manager->GetSession(sessionId_);
  GE_CHK_BOOL_RET_STATUS(inner_session != nullptr, FAILED, "[Get][Session] failed, session_id:%lu.", sessionId_);

  GE_CHK_STATUS_RET(inner_session->RegisterExternalAllocator(stream, allocator), "register external allocator failed");
  return SUCCESS;
}

Status Session::UnregisterExternalAllocator(const void *const stream) const {
  GE_ASSERT(IsGEInitialize(), "[Construct][Session]Failed because lack GEInitialize call before.");
  const SessionPtr inner_session = g_session_manager->GetSession(sessionId_);
  GE_CHK_BOOL_RET_STATUS(inner_session != nullptr, FAILED, "[Get][Session] failed, session_id:%lu.", sessionId_);

  GE_CHK_STATUS_RET(inner_session->UnregisterExternalAllocator(stream), "unregister external allocator failed");
  return SUCCESS;
}

Status Session::ShardGraphsToFile(const char_t *file_path) const {
  (void)file_path;
  GELOGE(FAILED, "This interface in the current version has been taken offline.");
  return FAILED;
}

Status Session::ShardGraphs() const {
  GELOGE(FAILED, "This interface in the current version has been taken offline.");
  return FAILED;
}

Status Session::SaveGraphsToPb(const char_t *file_path) const {
  (void)file_path;
  GELOGE(FAILED, "This interface in the current version has been taken offline.");
  return FAILED;
}

Status Session::PaRemapped(const uint64_t va, const uint64_t new_pa, const uint64_t len) const {
  GE_ASSERT(IsGEInitialize(), "[Construct][Session]Failed because lack GEInitialize call before.");
  const SessionPtr inner_session = g_session_manager->GetSession(sessionId_);
  GE_CHK_BOOL_RET_STATUS(inner_session != nullptr, INTERNAL_ERROR, "[Get][Session] failed, session_id:%lu.",
                         sessionId_);
  return inner_session->PaRemapped(va, new_pa, len);
}
}  // namespace ge

extern "C" {
ge::Status GeSessionLoadGraph(ge::Session &session, uint32_t graph_id,
                              const std::map<ge::AscendString, ge::AscendString> &options,
                              void *stream) {
  return session.LoadGraph(graph_id, options, stream);
}

ge::Status GeSessionExecuteGraphWithStreamAsync(ge::Session &session, uint32_t graph_id, void *stream,
                                                const std::vector<gert::Tensor> &inputs,
                                                std::vector<gert::Tensor> &outputs) {
  return session.ExecuteGraphWithStreamAsync(graph_id, stream, inputs, outputs);
}

ge::Status GetRegisteredIrDef(const char *op_type, std::vector<std::pair<ge::AscendString, ge::AscendString>> &inputs,
                              std::vector<std::pair<ge::AscendString, ge::AscendString>> &outputs,
                              std::vector<std::pair<ge::AscendString, ge::AscendString>> &attrs) {
  GE_ASSERT_NOTNULL(op_type);
  const auto op = ge::OperatorFactory::CreateOperator("_", op_type);
  GE_WARN_ASSERT(!op.IsEmpty(), "No operator found for type: %s", op_type);
  const auto desc = ge::OpDescUtils::GetOpDescFromOperator(op);

  static const auto kInputTypeString = []() {
    std::map<ge::IrInputType, ge::AscendString> typeStr;
    typeStr[ge::IrInputType::kIrInputRequired] = "required";
    typeStr[ge::IrInputType::kIrInputOptional] = "optional";
    typeStr[ge::IrInputType::kIrInputDynamic] = "dynamic";
    return typeStr;
  }();

  static const auto kOutputTypeString = []() {
    std::map<ge::IrOutputType, ge::AscendString> typeStr;
    typeStr[ge::IrOutputType::kIrOutputRequired] = "required";
    typeStr[ge::IrOutputType::kIrOutputDynamic] = "dynamic";
    return typeStr;
  }();

  GE_ASSERT_NOTNULL(desc, "Failed to get OpDesc from operator: %s", op_type);
  for (const auto &name2type : desc->GetIrInputs()) {
    auto iter = kInputTypeString.find(name2type.second);
    GE_ASSERT(iter != kInputTypeString.end(), "Unknown input type: %d for operator: %s", name2type.second, op_type);
    inputs.emplace_back(ConvertToAscendString(name2type.first), iter->second);
  }
  for (const auto &name2type : desc->GetIrOutputs()) {
    auto iter = kOutputTypeString.find(name2type.second);
    GE_ASSERT(iter != kOutputTypeString.end(), "Unknown output type: %d for operator: %s", name2type.second, op_type);
    outputs.emplace_back(ConvertToAscendString(name2type.first), iter->second);
  }

  std::map<ge::AscendString, ge::AscendString> attrs_and_types;
  GE_ASSERT_GRAPH_SUCCESS(op.GetAllIrAttrNamesAndTypes(attrs_and_types),
                          "Failed to get attr names and types for operator: %s", op_type);
  for (const auto &attr : desc->GetIrAttrNames()) {
    attrs.emplace_back(ConvertToAscendString(attr), attrs_and_types[ConvertToAscendString(attr)]);
  }
  return ge::SUCCESS;
}
}
