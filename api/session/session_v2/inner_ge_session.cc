/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "session_v2/inner_ge_session.h"

#include <map>
#include <memory>
#include <vector>

#include "analyzer/analyzer.h"
#include "adx_datadump_server.h"
#include "common/checker.h"
#include "common/dump/dump_properties.h"
#include "common/dump/dump_manager.h"
#include "framework/common/util.h"
#include "framework/common/debug/ge_log.h"
#include "framework/common/helper/model_helper.h"
#include "graph/ge_context.h"
#include "graph/ge_global_options.h"
#include "graph/ge_local_context.h"
#include "common/context/local_context.h"
#include "graph/manager/graph_var_manager.h"
#include "graph/utils/tensor_adapter.h"
#include "graph/utils/graph_utils_ex.h"
#include "runtime/mem.h"
#include "api/aclgrph/option_utils.h"
#include "common/profiling/profiling_manager.h"
#include "common/profiling/profiling_init.h"
#include "common/model/external_allocator_manager.h"
#include "graph/manager/active_memory_allocator.h"
#include "graph/load/graph_loader.h"
#include "common/platform_info_util.h"
#include "common/model/ge_root_model.h"
#include "common/model/ge_model.h"
#include "common/memory/tensor_trans_utils.h"
#include "generator/ge_generator.h"

#include <api/gelib/gelib.h>

namespace ge {
void CopyGeOutputsMemToUserOutputs(const rtStream_t stream, const std::vector<GeTensor> &ge_outputs,
                                   std::vector<Tensor> &outputs) {
  // if alloc output memory by external allocator, should copy to user.
  AllocatorPtr external_allocator = ExternalAllocatorManager::GetExternalAllocator(stream);
  if (external_allocator == nullptr) {
    return;
  }

  if (outputs.size() == 0U) {
    outputs.reserve(ge_outputs.size());
    for (size_t i = 0UL; i < ge_outputs.size(); i++) {
      outputs.emplace_back(TensorAdapter::AsTensor(ge_outputs[i]));
      GELOGI("Return outputs memory malloc by external allocator success, mem:%p, size:%u", outputs[i].GetData(),
             outputs[i].GetSize());
    }
  }
}
namespace {
constexpr int32_t kDumpStatus = 0;
constexpr int32_t kDecimalSystem = 10;
constexpr int32_t kSocVersionLen = 50;

Status CheckReuseMemoryOption(const std::map<std::string, std::string> &options) {
  auto iter = options.find(OPTION_EXEC_DISABLE_REUSED_MEMORY);
  if (iter != options.end()) {
    if (iter->second == "0") {
      GELOGD("%s=0, reuse memory is open", OPTION_EXEC_DISABLE_REUSED_MEMORY);
    } else if (iter->second == "1") {
      GELOGD("%s=1, reuse memory is close", OPTION_EXEC_DISABLE_REUSED_MEMORY);
    } else {
      GELOGE(PARAM_INVALID, "[CheckReuse][MemoryOption]option %s=%s is invalid",
             OPTION_EXEC_DISABLE_REUSED_MEMORY, iter->second.c_str());
      REPORT_INNER_ERR_MSG("E19999", "CheckReuseMemoryOption failed because option %s=%s is invalid.",
                         OPTION_EXEC_DISABLE_REUSED_MEMORY, iter->second.c_str());
      return FAILED;
    }
  }
  return SUCCESS;
}

Status CheckAutoTuneMode(const std::map<std::string, std::string> &options) {
  auto option_key = options.find("ge.autoTuneMode");
  if (option_key != options.end() && !option_key->second.empty()) {
    REPORT_INNER_ERR_MSG(
        "E19999",
        "Check parameter's options[%s] unsupport, The Auto Tune function has been discarded. Please use the "
        "AOE tool for tuning.",
        option_key->first.c_str());
    GELOGE(
        FAILED,
        "[Check][Param]Options[%s] unsupport, The Auto Tune function has been discarded. Please use the AOE tool for "
        "tuning.",
        option_key->first.c_str());
    return FAILED;
  }
  return SUCCESS;
}

Status CheckOpPrecisionMode(const std::map<std::string, std::string> &options) {
  auto iter = options.find(ge::OP_PRECISION_MODE);
  if (iter != options.end() && !iter->second.empty() && !ge::CheckInputPathValid(iter->second)) {
    REPORT_PREDEFINED_ERR_MSG(
        "E10001", std::vector<const char *>({"parameter", "value", "reason"}),
        std::vector<const char *>({ge::OP_PRECISION_MODE.c_str(), iter->second.c_str(), "path is not found"}));
    GELOGE(PARAM_INVALID, "[Check][OP_PRECISION_MODE] %s not found", iter->second.c_str());
    return FAILED;
  }
  if (iter != options.end()) {
    GELOGI("Option set successfully, option = %s, value=%s",
           ge::OP_PRECISION_MODE.c_str(), iter->second.c_str());
  }
  return CheckPrecisionModeParamValid(options);
}

void SetSessionDeviceId() {
  std::string str_session_device_id;
  if (GetContext().GetOption("ge.session_device_id", str_session_device_id) == SUCCESS) {
    GELOGI("Option session device id has set, value is %s.", str_session_device_id.c_str());
    try {
      const uint32_t session_device_id = static_cast<uint32_t>(std::stoi(str_session_device_id.c_str()));
      GetContext().SetCtxDeviceId(session_device_id);
    } catch (...) {
      GELOGW("Option session device id is invalid, value is %s.", str_session_device_id.c_str());
    }
  }
}

Status CheckRunGraphMode(GraphManager &graph_manager, uint32_t graph_id, const RunGraphMode &expect_mode) {
  GraphNodePtr graph_node = nullptr;
  GE_ASSERT_SUCCESS(graph_manager.GetGraphNode(graph_id, graph_node));
  GE_ASSERT_NOTNULL(graph_node);
  const auto cur_mode = graph_node->GetRunGraphMode();
  if ((cur_mode != RunGraphMode::kRunGraphModeEnd) && (cur_mode != expect_mode)) {
    GELOGE(UNSUPPORTED, "Failed to execute %s for graph[%u] because %s was already called."
        " These execution methods are mutually exclusive and cannot be mixed.",
        GetRunGraphModeStr(expect_mode), graph_id, GetRunGraphModeStr(cur_mode));
    REPORT_INNER_ERR_MSG("E19999", "Failed to execute %s for graph[%u] because %s was already called."
        " These execution methods are mutually exclusive and cannot be mixed.",
        GetRunGraphModeStr(expect_mode), graph_id, GetRunGraphModeStr(cur_mode));
    return UNSUPPORTED;
  }
  return SUCCESS;
}
Status SetRunGraphMode(GraphManager &graph_manager, uint32_t graph_id, const RunGraphMode &mode) {
  GraphNodePtr graph_node = nullptr;
  GE_ASSERT_SUCCESS(graph_manager.GetGraphNode(graph_id, graph_node));
  GE_ASSERT_NOTNULL(graph_node);
  graph_node->SetRunGraphMode(mode);
  return SUCCESS;
}

Status SaveRootModel(const GeRootModelPtr &ge_root_model, ModelBufferData &model_buff) {
  GeGenerator::SetModelNameForDump(ge_root_model);
  bool is_unknown_shape = false;
  GE_ASSERT_SUCCESS(ge_root_model->CheckIsUnknownShape(is_unknown_shape),
                    "root model(id:%u) CheckIsUnknownShape failed", ge_root_model->GetModelId());
  GELOGD("begin save root model, cur model is %s", (is_unknown_shape ? "unknown shape model" : "known shape model"));
  GE_CHK_BOOL_EXEC(!ge_root_model->GetSubgraphInstanceNameToModel().empty(),
                   REPORT_INNER_ERR_MSG("E19999", "root model(id:%u) has no sub model.", ge_root_model->GetModelId());
                   return FAILED, "[Get][SubModel] ge root model has no sub model");
  GeModelPtr model_root = nullptr;
  if (is_unknown_shape) {
    auto name_to_ge_model = ge_root_model->GetSubgraphInstanceNameToModel();
    model_root = name_to_ge_model[ge_root_model->GetRootGraph()->GetName()];
  } else {
    model_root = ge_root_model->GetSubgraphInstanceNameToModel().begin()->second;
  }
  GE_CHECK_NOTNULL(model_root);

  const auto model_save_helper =
    ModelSaveHelperFactory::Instance().Create(OfflineModelFormat::OM_FORMAT_DEFAULT);
  GE_CHECK_NOTNULL(model_save_helper);
  model_save_helper->SetSaveMode(false);
  GE_ASSERT_SUCCESS(model_save_helper->SaveToOmRootModel(ge_root_model, ge_root_model->GetModelName(),
    model_buff, is_unknown_shape),
                    "SaveToOmRootModel failed, model id:%u", ge_root_model->GetModelId());
  return SUCCESS;
}
}

static std::mutex mutex_;  // BuildGraph and RunGraph use
bool InnerGeSession::is_dump_server_inited_ = false;
InnerGeSession::InnerGeSession(uint64_t session_id, const std::map<std::string, std::string> &options)
    : is_initialized_(false), session_id_(session_id), options_(options) {}

Status InnerGeSession::InitializeVarManager() {
  constexpr uint32_t version = static_cast<uint32_t>(SessionVersion::ClOUD_VERSION);
  constexpr uint32_t DEFAULT_JOB_ID = 0;
  GE_CHECK_NOTNULL(VarManager::Instance(session_id_));
  const Status ret =
      VarManager::Instance(session_id_)->Init(version, session_id_, GetContext().DeviceId(), DEFAULT_JOB_ID);
  if (ret != SUCCESS) {
    GELOGE(ret, "[Init][VarManager] failed.");
    REPORT_INNER_ERR_MSG("E19999", "VarManager init failed, InnerGeSession:%lu.", session_id_);
    GE_CHK_STATUS(RemoveDumpProperties(), "[Remove][DumpProperties] failed.");
  }
  return ret;
}

Status InnerGeSession::Initialize() {
  if (is_initialized_) {
    GELOGW("[InnerGeSession:%lu] session already initialize.", session_id_);
    return SUCCESS;
  }
  GE_CHK_STATUS_RET(PlatformInfoUtil::parseAicoreNumOption(options_));

  const std::map<std::string, std::string>::const_iterator it = options_.find(ge::SOC_VERSION);
  if (it == options_.cend()) {
    char version[kSocVersionLen] = {0};
    rtError_t rt_ret = rtGetSocVersion(version, kSocVersionLen);
    GE_IF_BOOL_EXEC(rt_ret != RT_ERROR_NONE,
        REPORT_INNER_ERR_MSG("E19999", "rtGetSocVersion failed.");
        GELOGE(rt_ret, "[Get][SocVersion]rtGetSocVersion failed");
        return FAILED;)
    GELOGI("Succeeded in getting SOC_VERSION[%s] from runtime in InnerGeSession::Initialize.", version);
    options_.insert(std::make_pair(ge::SOC_VERSION, version));
  }

  logLevel_ = static_cast<uint8_t>(dlog_getlevel(GE_MODULE_NAME, nullptr));
  // If the global options and the session options are duplicated, the session options is preferred.
  auto all_options = options_;
  {
    auto &global_options_mutex = GetGlobalOptionsMutex();
    const std::lock_guard<std::mutex> lock(global_options_mutex);
    all_options.insert(GetMutableGlobalOptions().cbegin(), GetMutableGlobalOptions().cend());
  }

  GE_ASSERT_SUCCESS(CheckAutoTuneMode(all_options));

  Status ret = CheckReuseMemoryOption(all_options);
  if (ret != SUCCESS) {
    GELOGE(ret, "[CheckReuse][MemoryOption] failed, [InnerGeSession:%lu].", session_id_);
    REPORT_INNER_ERR_MSG("E19999", "CheckReuseMemoryOption failed, InnerGeSession=%lu.", session_id_);
    return ret;
  }

  GE_ASSERT_SUCCESS(CheckOpPrecisionMode(all_options));

  // Check option modify_mixlist
  if (ge::CheckModifyMixlistParamValid(all_options) != ge::SUCCESS) {
    return FAILED;
  }
  GE_ASSERT_SUCCESS(CheckOptionValidValues(all_options, OPTION_FEATURE_BASE_REFRESHABLE, kFeatureMapRefreshOptions));
  GE_ASSERT_SUCCESS(CheckOptionValidValues(all_options, OPTION_CONST_LIFECYCLE, kConstLifecycleOptions));
  GE_ASSERT_SUCCESS(CheckOptionValidThreshold(all_options, OPTION_HOST_SCHEDULING_MAX_THRESHOLD));
  GE_ASSERT_SUCCESS(CheckOptionValidValues(all_options, TILING_SCHEDULE_OPTIMIZE, kStateOptions));
  GE_ASSERT_GRAPH_SUCCESS(CheckOptimizationOptionValid(all_options));

  UpdateThreadContext(std::map<std::string, std::string>{});

  SetSessionDeviceId();
  GE_CHK_STATUS_RET(rtSetDevice(static_cast<int32_t>(GetContext().DeviceId())), "Set device failed.");

  ModelHelper model_helper;
  GE_CHK_STATUS_RET(model_helper.GetHardwareInfo(options_), "[Get][Hardware]InnerGeSession Initialize:"
                    " Get hardware info failed.");

  DumpProperties dump_properties;
  GE_CHK_STATUS_RET(dump_properties.InitByOptions(), "Init dump properties failed.");
  GE_CHK_STATUS_RET(AddDumpProperties(dump_properties), "[Add][DumpProperties] failed.");

  ret = InnerInitialize();
  if (ret != SUCCESS) {
    GELOGE(ret, "[Init][GraphManager] failed, InnerGeSession:%lu.", session_id_);
    REPORT_INNER_ERR_MSG("E19999", "GraphManager initialize failed, InnerGeSession:%lu.", session_id_);
    GE_CHK_STATUS(RemoveDumpProperties(), "[Remove][DumpProperties] failed.");
    return ret;
  }

  GE_ASSERT_SUCCESS(InitializeVarManager());

  is_initialized_ = true;
  return SUCCESS;
}

Status InnerGeSession::Finalize() {
  std::lock_guard<std::mutex> lock(resource_mutex_);
  if (!is_initialized_) {
    GELOGW("[InnerGeSession:%lu] session does not initialize.", session_id_);
    return SUCCESS;
  }
  UpdateThreadContext(std::map<std::string, std::string>{});
  Status ret = InnerFinalize();
  if (ret != SUCCESS) {
    // Subsequent code execution is required, so no return is required
    GELOGE(ret, "[Finalize][GraphManager] failed, InnerGeSession:%lu.", session_id_);
    REPORT_INNER_ERR_MSG("E19999", "GraphManager Finalize failed, InnerGeSession:%lu.", session_id_);
  }

  is_initialized_ = false;

  // release analyzer saved info(Session Level)
  Analyzer::GetInstance()->DestroySessionJsonObject(session_id_);

  GE_CHK_RT(rtDeviceReset(static_cast<int32_t>(GetContext().DeviceId())));
  GE_CHK_STATUS_RET(RemoveDumpProperties(), "[Remove][DumpProperties] failed.");
  VarManagerPool::Instance().RemoveVarManager(session_id_);
  SessionMemAllocator<ExpandableActiveMemoryAllocator>::Instance().RemoveAllocator(session_id_);
  SessionMemAllocator<FixedBaseExpandableAllocator>::Instance().RemoveAllocator(session_id_);
  SessionMemAllocator<ActiveMemoryAllocator>::Instance().RemoveAllocator(session_id_);
  return ret;
}

Status InnerGeSession::InnerInitialize() {
  Status ret = model_executor_.Initialize(options_, session_id_);
  if (ret != SUCCESS) {
    GELOGE(ret, "[Init][GraphExecutor] failed, InnerGeSession:%lu.", session_id_);
    REPORT_INNER_ERR_MSG("E19999", "GraphExecutor initialize failed, InnerGeSession:%lu.", session_id_);
    GE_CHK_STATUS(RemoveDumpProperties(), "[Remove][DumpProperties] failed.");
    return ret;
  }

  ret = graph_manager_.Initialize(options_, &model_executor_);
  if (ret != SUCCESS) {
    GELOGE(ret, "[Init][GraphManager] failed, InnerGeSession:%lu.", session_id_);
    REPORT_INNER_ERR_MSG("E19999", "GraphManager initialize failed, InnerGeSession:%lu.", session_id_);
    GE_CHK_STATUS(RemoveDumpProperties(), "[Remove][DumpProperties] failed.");
    return ret;
  }
  // model executor thread should run later, in case graph_manager init failed.
  model_executor_.StartRunThread();
  return SUCCESS;
}

Status InnerGeSession::InnerFinalize() {
  Status ret = graph_manager_.Finalize();
  if (ret != SUCCESS) {
    // Subsequent code execution is required, so no return is required
    GELOGE(ret, "[Finalize][GraphManager] failed, InnerGeSession:%lu.", session_id_);
    REPORT_INNER_ERR_MSG("E19999", "GraphManager Finalize failed, InnerGeSession:%lu.", session_id_);
  }

  ret = model_executor_.Finalize();
  if (ret != SUCCESS) {
    // Subsequent code execution is required, so no return is required
    GELOGE(ret, "[Finalize][GraphExecutor] failed, InnerGeSession:%lu.", session_id_);
    REPORT_INNER_ERR_MSG("E19999", "GraphExecutor Finalize failed, InnerGeSession:%lu.", session_id_);
  }
  return SUCCESS;
}

Status InnerGeSession::AddGraph(uint32_t graph_id, const Graph &graph,
                              const std::map<std::string, std::string> &options) {
  std::lock_guard<std::mutex> lock(resource_mutex_);

  for (const auto &item : options) {
    GELOGI("GE option: %s, value: %s, innerSession:%lu, graphid: %u.", item.first.c_str(), item.second.c_str(),
           session_id_, graph_id);
  }

  auto iter = options.find("ge.autoTuneMode");
  if ((iter != options.end()) && (!iter->second.empty())) {
    REPORT_INNER_ERR_MSG(
        "E19999",
        "Check parameter's options[%s] unsupport, The Auto Tune function has been discarded. Please use the "
        "AOE tool for tuning.",
        iter->first.c_str());
    GELOGE(
        FAILED,
        "[Check][Param]Options[%s] unsupport, The Auto Tune function has been discarded. Please use the AOE tool for "
        "tuning.",
        iter->first.c_str());
    return FAILED;
  }
  auto compute_graph = GraphUtilsEx::GetComputeGraph(graph);
  GE_CHECK_NOTNULL(compute_graph);
  std::string session_graph_id = std::to_string(session_id_) + "_" + std::to_string(graph_id);
  if (!AttrUtils::SetStr(*compute_graph, ATTR_NAME_SESSION_GRAPH_ID, session_graph_id)) {
    GELOGW("Set graph session_graph_id attr failed.");
  } else {
    GELOGD("Set graph session_graph_id attr to [%s]", session_graph_id.c_str());
  }
  for (auto sub_graph : compute_graph->GetAllSubgraphs()) {
    (void)AttrUtils::SetStr(*sub_graph, ATTR_NAME_SESSION_GRAPH_ID, session_graph_id);
  }
  UpdateThreadContext(options);
  Status ret = graph_manager_.AddGraph(graph_id, graph, options, domi::GetContext());
  if (ret != SUCCESS) {
    GELOGE(ret, "[Add][Graph] failed, InnerGeSession:%lu graphid: %u.", session_id_, graph_id);
    REPORT_INNER_ERR_MSG("E19999", "GraphManager AddGraph failed, InnerGeSession:%lu graphid: %u.",
                         session_id_, graph_id);
    return ret;
  }
  const uint32_t device_id = GetContext().DeviceId();
  GELOGD("The device id is %u", device_id);
  (void)ProfilingInit::Instance().SetDeviceIdByModelId(graph_id, device_id);
  ProfilingManager::Instance().SetGraphIdToDeviceMap(graph_id, device_id);
  GELOGI("[InnerGeSession:%lu] Add graph success, graph_id=%u.", session_id_, graph_id);
  return SUCCESS;
}

Status InnerGeSession::LoadGraph(const uint32_t graph_id,
                               const std::map<AscendString, AscendString> &options, void *stream) {
  GELOGI("[InnerGeSession] Load graph by graph_id=%u, stream = %p", graph_id, stream);
  GraphNodePtr graph_node = nullptr;
  GE_ASSERT_SUCCESS(graph_manager_.GetGraphNode(graph_id, graph_node), "get graph failed, graph_id:%u.", graph_id);
  GE_ASSERT_NOTNULL(graph_node, "graph_node is nullptr, graph_id:%u.", graph_id);

  if (graph_node->GetRunFlag()) {
    REPORT_INNER_ERR_MSG("E19999", "Graph is already running, can't be loaded again, graph_id:%u, check invalid",
                         graph_id);
    GELOGE(GE_GRAPH_ALREADY_RUNNING, "[Get][RunFlag] graph already running, graph id = %u", graph_id);
    return GE_GRAPH_ALREADY_RUNNING;
  }

  if (!graph_node->GetBuildFlag()) {
    GELOGI("Graph is not compiled, start to compile graph, session_id:%lu, graph_id:%u", session_id_, graph_id);
    GE_ASSERT_SUCCESS(CompileGraph(graph_id, {}));
    GELOGI("Graph compiled successfully, continue to load graph, session_id:%lu, graph_id:%u", session_id_, graph_id);
  }

  if (graph_node->GetLoadFlag()) {
    REPORT_INNER_ERR_MSG("E19999", "Graph has been loaded, graph_id = %u", graph_id);
    GELOGE(GE_GRAPH_REPEAT_OPERATION, "[Check][LoadFlag] Graph has been loaded, graph_id = %u", graph_id);
    return GE_GRAPH_REPEAT_OPERATION;
  }

  auto &graph_options = const_cast<std::map<std::string, std::string>&>(graph_node->GetOptions());
  for (const auto &iter : options) {
    GELOGI("Get option key[%s] value[%s].", iter.first.GetString(), iter.second.GetString());
    if (graph_options.find(iter.first.GetString()) == graph_options.end()) {
      (void)graph_options.emplace(iter.first.GetString(), iter.second.GetString());
    }
  }

  auto ge_root_model = graph_node->GetGeRootModel();
  GE_CHECK_NOTNULL(ge_root_model);
  UpdateThreadContext(graph_options);
  const auto ret = graph_manager_.LoadGraph(graph_id, ge_root_model, graph_node, stream);
  if (ret != SUCCESS) {
    GELOGE(ret, "[Load][Graph] Failed, graph_id:%u.", graph_node->GetGraphId());
    for (const auto &iter : options) {
      (void)graph_options.erase(iter.first.GetString());
    }
    return ret;
  }
  graph_node->SetLoadFlag(true);
  return SUCCESS;
}

Status InnerGeSession::AddGraphWithCopy(uint32_t graph_id, const Graph &graph,
                                      const std::map<std::string, std::string> &options) {
  std::lock_guard<std::mutex> lock(resource_mutex_);
  auto compute_graph = GraphUtilsEx::GetComputeGraph(graph);
  GE_CHECK_NOTNULL(compute_graph);
  std::string session_graph_id = std::to_string(session_id_) + "_" + std::to_string(graph_id);
  if (!AttrUtils::SetStr(*compute_graph, ATTR_NAME_SESSION_GRAPH_ID, session_graph_id)) {
    GELOGW("Set graph session_graph_id attr failed.");
  } else {
    GELOGD("Set graph session_graph_id attr to [%s]", session_graph_id.c_str());
  }
  for (auto sub_graph : compute_graph->GetAllSubgraphs()) {
    (void)AttrUtils::SetStr(*sub_graph, ATTR_NAME_SESSION_GRAPH_ID, session_graph_id);
  }
  UpdateThreadContext(options);
  Status ret = graph_manager_.AddGraphWithCopy(graph_id, graph, options, domi::GetContext());
  if (ret != SUCCESS) {
    GELOGE(ret, "[Add][Graph] failed, InnerGeSession:%lu graphid: %u.", session_id_, graph_id);
    REPORT_INNER_ERR_MSG("E19999",
                      "GraphManager AddGraphWithCopy failed, InnerGeSession:%lu graphid: %u.", session_id_, graph_id);
    return ret;
  }

  GELOGI("[InnerGeSession:%lu] add graph success, graph_id=%u.", session_id_, graph_id);
  return SUCCESS;
}

Status InnerGeSession::RunGraph(uint32_t graph_id, const std::vector<gert::Tensor> &inputs,
                                std::vector<gert::Tensor> &outputs) {
  GELOGI("[InnerGeSession:%lu] Run graph on session, graph_id=%u.", session_id_, graph_id);
  // find graph
  GraphNodePtr graph_node = nullptr;
  Status ret = graph_manager_.GetGraphNode(graph_id, graph_node);
  if (ret != SUCCESS) {
    REPORT_INNER_ERR_MSG("E19999", "Graph:%u does not exist in graph_map, check invalid", graph_id);
    GELOGE(ret, "[Get][GraphNode] failed, graph does not exist, graph_id = %u.", graph_id);
    return ret;
  }

  if (graph_node == nullptr) {
    REPORT_INNER_ERR_MSG("E19999", "Graph node is nullptr in graph_map, graph_id:%u, check invalid", graph_id);
    GELOGE(GE_GRAPH_GRAPH_NODE_NULL, "[Check][Param] graph node is NULL, graph_id = %u.", graph_id);
    return GE_GRAPH_GRAPH_NODE_NULL;
  }

  if (graph_node->GetRunFlag()) {
    REPORT_INNER_ERR_MSG("E19999", "Graph is already running, can't be run again, graph_id:%u, check invalid", graph_id);
    GELOGE(GE_GRAPH_ALREADY_RUNNING, "[Get][RunFlag] graph already running, graph id = %u", graph_id);
    return GE_GRAPH_ALREADY_RUNNING;
  }

  if (!graph_node->GetLoadFlag()) {
    GELOGI("Graph is not loaded, start to load graph, session_id:%lu, graph_id:%u", session_id_, graph_id);
    GE_ASSERT_SUCCESS(LoadGraph(graph_id, {}, nullptr));
    GELOGI("Graph loaded successfully, continue to run graph, session_id:%lu, graph_id:%u", session_id_, graph_id);
  }
  const auto check_ret = CheckRunGraphMode(graph_manager_, graph_id, RunGraphMode::kRunGraph);
  if (check_ret != SUCCESS) {
    REPORT_INNER_ERR_MSG("E19999", "check run graph mode failed, graph_id:%u", graph_id);
    GELOGE(check_ret, "check run graph mode failed, graph_id:%u", graph_id);
    return check_ret;
  }
  if (mutex_.try_lock()) {
    std::lock_guard<std::mutex> lock(mutex_, std::adopt_lock);
    UpdateThreadContext(graph_id);
    Status ret = graph_manager_.RunGraph(graph_id, inputs, outputs);
    domi::GetContext().out_nodes_map.clear();
    domi::GetContext().user_out_nodes.clear();
    GE_ASSERT_SUCCESS(SetRunGraphMode(graph_manager_, graph_id, RunGraphMode::kRunGraph));
    if (ret != SUCCESS) {
      GELOGE(ret, "[Run][Graph]failed, InnerGeSession:%lu graph_id=%u.", session_id_, graph_id);
      REPORT_INNER_ERR_MSG("E19999",
                        "GraphManager RunGraph failed, InnerGeSession:%lu graph_id=%u.", session_id_, graph_id);
      return ret;
    }
    GELOGI("[InnerGeSession:%lu] run graph success, graph_id=%u.", session_id_, graph_id);
    return SUCCESS;
  } else {
    GELOGE(GE_SESS_ALREADY_RUNNING, "[Run][Graph]failed, InnerGeSession:%lu, graph_id=%u.", session_id_, graph_id);
    REPORT_INNER_ERR_MSG("E19999",
                       "RunGraph failed because mutex try_lock false, InnerGeSession:%lu, graph_id=%u.",
                       session_id_, graph_id);
    return GE_SESS_ALREADY_RUNNING;
  }
}

Status InnerGeSession::RunGraphWithStreamAsync(uint32_t graph_id, const rtStream_t stream,
                                               const std::vector<gert::Tensor> &inputs,
                                               std::vector<gert::Tensor> &outputs) {
  if (logLevel_ <= DLOG_INFO) {
    GELOGI("Execute graph with stream begin, session id = %lu, graph id = %u,"
          "stream = %p, input size = %zu, output size = %zu",
          session_id_, graph_id, stream, inputs.size(), outputs.size());
  }
  // find graph
  GraphNodePtr graph_node = nullptr;
  Status ret = graph_manager_.GetGraphNode(graph_id, graph_node);
  if (ret != SUCCESS) {
    REPORT_INNER_ERR_MSG("E19999", "Graph:%u does not exist in graph_map, check invalid", graph_id);
    GELOGE(ret, "[Get][GraphNode] failed, graph does not exist, graph_id = %u.", graph_id);
    return ret;
  }
  GE_CHK_BOOL_RET_STATUS(graph_node != nullptr, GE_GRAPH_GRAPH_NODE_NULL,
      "Graph node is nullptr in graph_map, graph_id:%u, check invalid", graph_id);

  if (!graph_node->GetLoadFlag()) {
    GELOGI("Graph is not loaded, start to load graph, session_id:%lu, graph_id:%u", session_id_, graph_id);
    GE_ASSERT_SUCCESS(LoadGraph(graph_id, {}, nullptr));
    GELOGI("Graph loaded successfully, continue to run graph, session_id:%lu, graph_id:%u", session_id_, graph_id);
  }
  const auto check_ret = CheckRunGraphMode(graph_manager_, graph_id, RunGraphMode::kRunGraphWithStreamAsync);
  if (check_ret != SUCCESS) {
    REPORT_INNER_ERR_MSG("E19999", "check run graph mode failed, graph_id:%u", graph_id);
    GELOGE(check_ret, "check run graph mode failed, graph_id:%u", graph_id);
    return check_ret;
  }
  std::lock_guard<std::mutex> lock(build_run_mutex_);
  const Status res = graph_manager_.ExecuteGraphWithStreamAsync(graph_id, stream, inputs, outputs);
  GE_ASSERT_SUCCESS(SetRunGraphMode(graph_manager_, graph_id, RunGraphMode::kRunGraphWithStreamAsync));
  if (res != SUCCESS) {
    GELOGE(res, "[Execute][GraphWithStreamAsync]failed,"
            "session id = %lu, graph id = %u, stream = %p.", session_id_, graph_id, stream);
    REPORT_INNER_ERR_MSG("E19999", "GraphManager ExecuteGrapWithStreamhAsync failed,"
                      "session id = %lu, graph id = %u, stream = %p.", session_id_, graph_id, stream);
    return res;
  }

  if (logLevel_ <= DLOG_INFO) {
    GELOGI("Execute graph with stream async success, session id = %lu, graph id = %u, stream = %p.",
          session_id_, graph_id, stream);
  }

  return SUCCESS;
}

Status InnerGeSession::RemoveGraph(uint32_t graph_id) {
  std::lock_guard<std::mutex> lock(resource_mutex_);
  const auto device_id = GetContext().DeviceId();
  GELOGD("Remove device id %u", device_id);
  (void)ProfilingInit::Instance().UnsetDeviceIdByModelId(graph_id, device_id);
  UpdateThreadContext(graph_id);
  const Status ret = graph_manager_.RemoveGraph(graph_id);
  if (ret != SUCCESS) {
    GELOGE(ret, "[Remove][Graph] failed, InnerGeSession:%lu, graph_id=%u.", session_id_, graph_id);
    REPORT_INNER_ERR_MSG("E19999",
                      "GraphManager RemoveGraph failed, InnerGeSession:%lu, graph_id=%u.", session_id_, graph_id);
    return ret;
  }
  GELOGI("[InnerGeSession:%lu] Remove graph success, graph_id=%u.", session_id_, graph_id);
  return SUCCESS;
}

Status InnerGeSession::RegisterCallBackFunc(
    const std::string &key,
    const std::function<Status(uint32_t, const std::map<AscendString, gert::Tensor> &)> &callback) {
  std::lock_guard<std::mutex> lock(resource_mutex_);
  UpdateThreadContext(std::map<std::string, std::string>{});
  const Status ret = graph_manager_.RegisterCallBackFunc(key, callback);
  if (ret != SUCCESS) {
    GELOGE(ret, "[Register][CallBackFunc] failed, InnerGeSession:%lu register %s.", session_id_, key.c_str());
    REPORT_INNER_ERR_MSG("E19999",
                      "GraphManager RegisterCallBackFunc failed, InnerGeSession:%lu register %s.",
                      session_id_, key.c_str());
    return ret;
  }

  GELOGI("[InnerGeSession:%lu] register %s callback function success.", session_id_, key.c_str());
  return SUCCESS;
}

Status InnerGeSession::RunGraphAsync(uint32_t graph_id, const std::vector<gert::Tensor> &inputs,
      std::function<void(Status status, std::vector<gert::Tensor> &outputs)> callback) {
  GELOGI("[InnerGeSession:%lu] run graph on session, graph_id=%u.", session_id_, graph_id);
  UpdateThreadContext(graph_id);
  const auto check_ret = CheckRunGraphMode(graph_manager_, graph_id, RunGraphMode::kRunGraphAsync);
  if (check_ret != SUCCESS) {
    if (callback != nullptr) {
      std::vector<gert::Tensor> outputs;
      callback(check_ret, outputs);
    }
    REPORT_INNER_ERR_MSG("E19999", "check run graph mode failed, graph_id:%u", graph_id);
    GELOGE(check_ret, "check run graph mode failed, graph_id:%u", graph_id);
    return check_ret;
  }
  auto inputs_share = TensorTransUtils::ShareFromGertTenosrs(inputs);
  Status ret = graph_manager_.RunGraphAsync(graph_id, std::move(inputs_share), session_id_, callback);
  if (ret != SUCCESS) {
    GELOGE(ret, "[Run][GraphAsync]failed, InnerGeSession:%lu graph_id=%u.", session_id_, graph_id);
    REPORT_INNER_ERR_MSG("E19999",
                      "GraphManager RunGraphAsync failed, InnerGeSession:%lu graph_id=%u.", session_id_, graph_id);
    return ret;
  }
  GELOGI("[InnerGeSession:%lu] run graph async submit success, graph_id=%u.", session_id_, graph_id);
  return ret;
}

const GraphManager &InnerGeSession::getGraphManagerObj() const { return graph_manager_; }

void InnerGeSession::UpdateThreadContext(const std::map<std::string, std::string> &options) const {
  UpdateGlobalSessionContext();
  GetThreadLocalContext().SetGraphOption(options);
}

void InnerGeSession::UpdateGlobalSessionContext() const {
  {
    auto &global_options_mutex = GetGlobalOptionsMutex();
    const std::lock_guard<std::mutex> lock(global_options_mutex);
    GetThreadLocalContext().SetGlobalOption(GetMutableGlobalOptions());
  }
  GetThreadLocalContext().SetSessionOption(options_);
  GetContext().SetSessionId(session_id_);
  SetTrainFlagOption();
  SetRtSocVersion();
}

void InnerGeSession::UpdateThreadContext(uint32_t graph_id) {
  auto options = graph_manager_.GetGraphOptions(graph_id);
  if (options == nullptr) {
    GELOGW("graph level options is null.");
    UpdateThreadContext(std::map<std::string, std::string>{});
  } else {
    UpdateThreadContext(*options);
  }
}

bool InnerGeSession::IsGraphNeedRebuild(uint32_t graph_id) {
  UpdateThreadContext(graph_id);
  return graph_manager_.IsGraphNeedRebuild(graph_id);
}

Status InnerGeSession::AddDumpProperties(const DumpProperties &dump_properties) {
  if (!is_dump_server_inited_) {
    if ((dump_properties.IsDumpOpen() || dump_properties.IsOpDebugOpen())) {
      GE_IF_BOOL_EXEC(AdxDataDumpServerInit() != kDumpStatus,
                      GELOGE(PARAM_INVALID, "[Init][AdxDataDumpServer] failed, session_id:%lu.", session_id_);
                      return PARAM_INVALID)
      GELOGI("Init adx data dump server success");
      is_dump_server_inited_ = true;
    }
  }
  if ((!dump_properties.GetEnableDump().empty()) || (!dump_properties.GetEnableDumpDebug().empty())) {
    // if dump option set, add dump property
    GE_IF_BOOL_EXEC(DumpManager::GetInstance().AddDumpProperties(session_id_, dump_properties) != SUCCESS,
                    GELOGE(PARAM_INVALID, "[Add][DumpProperties] failed, session_id:%lu.", session_id_);
                    return PARAM_INVALID);
    if (DumpManager::GetInstance().CheckIfAclDumpSet()) {
      GELOGW("Set dump by options and acl simultaneously, will use the option setting.");
    }
    DumpManager::GetInstance().ClearAclDumpSet();
  }
  return SUCCESS;
}

Status InnerGeSession::RemoveDumpProperties() {
  DumpManager::GetInstance().RemoveDumpProperties(session_id_);
  if (is_dump_server_inited_ && DumpManager::GetInstance().GetDumpPropertiesMap().empty()) {
    GE_IF_BOOL_EXEC(AdxDataDumpServerUnInit() != kDumpStatus,
                    GELOGE(PARAM_INVALID, "[UnInit][AdxDataDumpServer] failed, session_id:%lu.", session_id_);
                    REPORT_INNER_ERR_MSG("E19999", "RemoveDumpProperties failed because AdxDataDumpServerUnInit failed,"
                                       "session_id:%lu", session_id_);
                    return PARAM_INVALID)
    GELOGI("UnInit adx data dump server success");
    is_dump_server_inited_ = false;
  }
  return SUCCESS;
}

void InnerGeSession::SetRtSocVersion() {
  auto &global_options_mutex = GetGlobalOptionsMutex();
  const std::lock_guard<std::mutex> lock(global_options_mutex);
  const auto &global_options = GetMutableGlobalOptions();
  auto it = global_options.find(ge::SOC_VERSION);
  if (it != global_options.end()) {
    rtError_t rt_ret = rtSetSocVersion(it->second.c_str());
    if (rt_ret != RT_ERROR_NONE) {
      GELOGW("Set soc version %s failed. ret:0x%X", it->second.c_str(), rt_ret);
    }
    GELOGI("Set soc version %s success.", it->second.c_str());
  }
}

void InnerGeSession::SetTrainFlagOption() {
  auto train_flag = false;
  std::string run_mode;
  if ((GetContext().GetOption(ge::OPTION_GRAPH_RUN_MODE, run_mode) == SUCCESS) && (!run_mode.empty())) {
    if (GraphRunMode(std::strtol(run_mode.c_str(), nullptr, kDecimalSystem)) >= TRAIN) {
      train_flag = true;
    }
  }
  domi::GetContext().train_flag = train_flag;
  GELOGI("train flag is %d in session", train_flag);
}

Status InnerGeSession::CompileGraph(uint32_t graph_id, const vector<ge::Tensor> &inputs) {
  UpdateThreadContext(graph_id);
  const auto ret = graph_manager_.CompileGraph(graph_id, session_id_, inputs);
  GE_CHK_STATUS_RET(ret, "[Compile][Graph]Failed, InnerGeSession:%lu, graph_id:%u, inputs size:%zu.",
                    session_id_, graph_id, inputs.size());
  GELOGI("[InnerGeSession:%lu]Compile graph success, session_id:%lu, graph_id:%u, inputs size:%zu.",
         session_id_, graph_id, inputs.size());
  return SUCCESS;
}

Status InnerGeSession::GetCompiledModel(uint32_t graph_id, ModelBufferData &model_buffer) {
  GELOGI("Start to get the compiled model. graph_id: %u.", graph_id);
  UpdateThreadContext(graph_id);
  GraphNodePtr graph_node = nullptr;
  Status ret = graph_manager_.GetGraphNode(graph_id, graph_node);
  if (ret != SUCCESS) {
    REPORT_INNER_ERR_MSG("E19999", "Graph:%u does not exist in graph_map, check invalid", graph_id);
    GELOGE(ret, "[Get][GraphNode] failed, graph does not exist, graph_id = %u.", graph_id);
    return ret;
  }
  if (!graph_node->GetBuildFlag()) {
    REPORT_INNER_ERR_MSG("E19999", "Graph is not compiled, session_id:%lu, graph_id:%u", session_id_, graph_id);
    GELOGE(PARAM_INVALID, "[Check][CompileFlag] Graph is not compiled session_id:%lu, graph_id:%u",
      session_id_, graph_id);
    return PARAM_INVALID;
  }
  std::string options;
  (void)GetContext().GetOption("ge.exec.variable_acc", options);
  if (options == "True") {
    REPORT_PREDEFINED_ERR_MSG("E10001", std::vector<const char *>({"parameter", "value", "reason"}),
        std::vector<const char *>({ge::OPTION_EXEC_VARIABLE_ACC, "True",
        "This interface is incompatible with the configuration where 'option ge.exec.variable_acc' is set to \"True\","
        " Please set the option to \"False\" and try again."}));
    GELOGE(UNSUPPORTED, "This interface is incompatible with the configuration where"
        " 'option ge.exec.variable_acc' is set to \"True\". Please set the option to \"False\" and try again.");
    return UNSUPPORTED;
  }
  const auto ge_root_model = graph_node->GetGeRootModel();
  GE_CHECK_NOTNULL(ge_root_model);
  return SaveRootModel(ge_root_model, model_buffer);
}

Status InnerGeSession::GetCompiledGraphSummary(uint32_t graph_id, CompiledGraphSummaryPtr &summary) {
  UpdateThreadContext(graph_id);
  return graph_manager_.GetCompiledGraphSummary(graph_id, summary);
}

Status InnerGeSession::SetGraphConstMemoryBase(uint32_t graph_id, const void *const memory, size_t size) {
  UpdateThreadContext(graph_id);
  const auto ret = graph_manager_.SetConstMemoryBase(graph_id, memory, size);
  GE_CHK_STATUS_RET(ret, "[Set][Memory]Failed, InnerGeSession:%lu, graph_id:%u, memory:%p, size:%zu.",
                    session_id_, graph_id, memory, size);
  GELOGI("[InnerGeSession:%lu]Set graph const memory base success, graph_id:%u, memory:%p, size:%zu.",
         session_id_, graph_id, memory, size);
  return SUCCESS;
}

Status InnerGeSession::UpdateGraphFeatureMemoryBase(uint32_t graph_id, const void *const memory, size_t size) {
  UpdateThreadContext(graph_id);
  const auto ret = graph_manager_.UpdateFeatureMemoryBase(graph_id, memory, size);
  GE_CHK_STATUS_RET(ret, "[Update][Memory]Failed, InnerGeSession:%lu, graph_id:%u, memory:%p, size:%zu.",
                    session_id_, graph_id, memory, size);
  GELOGI("[InnerGeSession:%lu]Update graph feature memory base success, graph_id:%u, memory:%p, size:%zu.",
         session_id_, graph_id, memory, size);
  return SUCCESS;
}

Status InnerGeSession::SetGraphFixedFeatureMemoryBase(uint32_t graph_id, MemoryType type, const void *const memory,
                                                    size_t size) {
  UpdateThreadContext(graph_id);
  const auto ret = graph_manager_.SetFixedFeatureMemoryBase(graph_id, type, memory, size);
  GE_CHK_STATUS_RET(ret, "[Set][Memory]Failed, InnerGeSession:%lu, graph_id:%u, type:%d, memory:%p, size:%zu.",
                    session_id_, graph_id, type, memory, size);
  return SUCCESS;
}

Status InnerGeSession::UpdateGraphRefreshableFeatureMemoryBase(uint32_t graph_id, const void *const memory, size_t size) {
  UpdateThreadContext(graph_id);
  const auto ret = graph_manager_.UpdateRefreshableFeatureMemoryBase(graph_id, memory, size);
  GE_CHK_STATUS_RET(ret, "[Update][Memory]Failed, InnerGeSession:%lu, graph_id:%u, memory:%p, size:%zu.",
                    session_id_, graph_id, memory, size);
  GELOGI("[InnerGeSession:%lu]Update graph refreshable feature memory base success, graph_id:%u, memory:%p, size:%zu.",
         session_id_, graph_id, memory, size);
  return SUCCESS;
}

Status InnerGeSession::RegisterExternalAllocator(const void *const stream, AllocatorPtr allocator) const {
  GE_ASSERT_NOTNULL(stream, "stream is nullptr, session_id:%u.", session_id_);
  GE_ASSERT_NOTNULL(allocator, "allocator is nullptr, session_id:%u.", session_id_);

  GELOGI("[InnerGeSession:%lu]Register external allocator success, stream:%p, allocator:%p.",
         session_id_, stream, allocator.get());
  ExternalAllocatorManager::SetExternalAllocator(stream, allocator);
  return SUCCESS;
}

Status InnerGeSession::UnregisterExternalAllocator(const void * const stream) const {
  GE_ASSERT_NOTNULL(stream, "stream is nullptr, session_id:%u.", session_id_);

  GELOGI("[InnerGeSession:%lu]Unregister external allocator success, stream:%p.", session_id_, stream);
  ExternalAllocatorManager::DeleteExternalAllocator(stream);
  return SUCCESS;
}

Status InnerGeSession::ForkGraph(uint32_t origin_graph_id, uint32_t forked_graph_id) {
  return graph_manager_.ForkGraph(origin_graph_id, forked_graph_id);
}
}  // namespace ge
