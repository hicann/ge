/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "acl/acl_rt_impl.h"
#include "executor/ge_executor.h"
#include "acl_resource_manager.h"
#include "acl_model_error_codes_inner.h"
#include "acl_model_json_parser.h"

#ifdef __cplusplus
extern "C" {
#endif
typedef aclError (*aclDumpSetCallbackFunc)(const char *configStr);
typedef aclError (*aclDumpUnsetCallbackFunc)();
extern ACL_FUNC_VISIBILITY aclError aclDumpSetCallbackRegister(aclDumpSetCallbackFunc cbFunc);
extern ACL_FUNC_VISIBILITY aclError aclDumpSetCallbackUnRegister();
extern ACL_FUNC_VISIBILITY aclError aclDumpUnsetCallbackRegister(aclDumpUnsetCallbackFunc cbFunc);
extern ACL_FUNC_VISIBILITY aclError aclDumpUnsetCallbackUnRegister();
#ifdef __cplusplus
}
#endif

namespace {
struct DumpBlacklist {
    std::string name;
    std::vector<std::string> pos;
};

struct OpNameRange {
    std::string begin;
    std::string end;
};

struct ModelDumpConfig {
    std::string modelName;
    std::vector<std::string> layer;
    std::vector<std::string> watcherNodes;
    bool isLayer = false; // Whether the label of "layer" exists
    bool isModelName = false; // Whether the label of "model_name" exists
    std::vector<DumpBlacklist> optypeBlacklist;
    std::vector<DumpBlacklist> opnameBlacklist;
    std::vector<OpNameRange> dumpOpNameRanges;
};

struct aclDumpConfig {
    std::string dumpPath;
    std::string dumpMode;
    std::vector<ModelDumpConfig> dumpList;
    std::string dumpOpSwitch;
    std::string dumpDebug;
    std::string dumpScene;
    std::string dumpStep;
    std::string dumpData;
    std::string dumpLevel;
    std::vector<std::string> dumpStats;
};
const std::string ACL_DUMP = "dump";
const std::string ACL_DUMP_MODEL_NAME = "model_name";
const std::string ACL_DUMP_LAYER = "layer";
const std::string ACL_DUMP_WATCHER_NODES = "watcher_nodes";
const std::string ACL_DUMP_OPNAME_RANGE = "opname_range";
const std::string ACL_DUMP_OPNAME_RANGE_BEGIN = "begin";
const std::string ACL_DUMP_OPNAME_RANGE_END = "end";
const std::string ACL_DUMP_PATH = "dump_path";
const std::string ACL_DUMP_LIST = "dump_list";
const std::string ACL_DUMP_MODE = "dump_mode";
const std::string ACL_DUMP_STEP = "dump_step";
const std::string ACL_DUMP_OP_SWITCH = "dump_op_switch";
const std::string ACL_DUMP_MODE_OUTPUT = "output";
const std::string ACL_DUMP_DEBUG = "dump_debug";
const std::string ACL_DUMP_DATA = "dump_data";
const std::string ACL_DUMP_LEVEL = "dump_level";
const std::string ACL_DUMP_SCENE = "dump_scene";
const std::string ACL_DUMP_STATS = "dump_stats";
const std::string ACL_DUMP_LEVEL_ALL = "all";
const std::string ACL_DUMP_STATUS_SWITCH_ON = "on";
const std::string ACL_DUMP_STATUS_SWITCH_OFF = "off";
const std::string ACL_DUMP_LEVEL_OP = "op";
const std::string ACL_DUMP_OPTYPE_BLACKLIST = "optype_blacklist";
const std::string ACL_DUMP_OPNAME_BLACKLIST = "opname_blacklist";
const std::string ACL_DUMP_BLACKLIST_NAME = "name";
const std::string ACL_DUMP_BLACKLIST_POS = "pos";

const std::string ACL_DUMP_LITE_EXCEPTION = "lite_exception";
const std::string ACL_DUMP_EXCEPTION_AIC_ERR_BRIEF = "aic_err_brief_dump";    // l0 exception dump
const std::string ACL_DUMP_EXCEPTION_AIC_ERR_NORM = "aic_err_norm_dump";      // l1 exception dump
const std::string ACL_DUMP_EXCEPTION_AIC_ERR_DETAIL = "aic_err_detail_dump";  // npu coredump
const std::set<std::string> aclDumpSceneExceptions = {
    ACL_DUMP_LITE_EXCEPTION, ACL_DUMP_EXCEPTION_AIC_ERR_BRIEF,
    ACL_DUMP_EXCEPTION_AIC_ERR_NORM, ACL_DUMP_EXCEPTION_AIC_ERR_DETAIL,
};

void HandleReleaseSourceByDevice(uint32_t devId, bool isReset)
{
    acl::AclResourceManager::GetInstance().HandleReleaseSourceByDevice(devId, isReset);
}

void HandleReleaseSourceByStream(aclrtStream stream, bool isCreate)
{
    acl::AclResourceManager::GetInstance().HandleReleaseSourceByStream(stream, isCreate);
}

std::string GetCfgStrByKey(const nlohmann::json &js , const std::string &key) {
    return js.at(key).get<std::string>();
}

bool ContainKey(const nlohmann::json &js, const std::string &key) {
    return (js.find(key) != js.end());
}

static void from_json(const nlohmann::json &js, OpNameRange &range) {
    if (ContainKey(js, ACL_DUMP_OPNAME_RANGE_BEGIN)) {
        range.begin = js.at(ACL_DUMP_OPNAME_RANGE_BEGIN).get<std::string>();
    }
    if (ContainKey(js, ACL_DUMP_OPNAME_RANGE_END)) {
        range.end = js.at(ACL_DUMP_OPNAME_RANGE_END).get<std::string>();
    }
}

static void from_json(const nlohmann::json &js, DumpBlacklist &blacklist)
{
    if (ContainKey(js, ACL_DUMP_BLACKLIST_NAME)) {
        blacklist.name = GetCfgStrByKey(js, ACL_DUMP_BLACKLIST_NAME);
    }
    if (ContainKey(js, ACL_DUMP_BLACKLIST_POS)) {
        blacklist.pos = js.at(ACL_DUMP_BLACKLIST_POS).get<std::vector<std::string>>();
    }
}

static void from_json(const nlohmann::json &js, ModelDumpConfig &info)
{
    info.isLayer = false;
    if (ContainKey(js, ACL_DUMP_MODEL_NAME)) {
        info.modelName = GetCfgStrByKey(js, ACL_DUMP_MODEL_NAME);
        info.isModelName = true;
    }
    if (ContainKey(js, ACL_DUMP_LAYER)) {
        info.layer = js.at(ACL_DUMP_LAYER).get<std::vector<std::string>>();
        info.isLayer = true;
    }
    if (ContainKey(js, ACL_DUMP_WATCHER_NODES)) {
        info.watcherNodes = js.at(ACL_DUMP_WATCHER_NODES).get<std::vector<std::string>>();
    }
    if (ContainKey(js, ACL_DUMP_OPTYPE_BLACKLIST)) {
        info.optypeBlacklist = js.at(ACL_DUMP_OPTYPE_BLACKLIST).get<std::vector<DumpBlacklist>>();
    }
    if (ContainKey(js, ACL_DUMP_OPNAME_BLACKLIST)) {
        info.opnameBlacklist = js.at(ACL_DUMP_OPNAME_BLACKLIST).get<std::vector<DumpBlacklist>>();
    }
    if (ContainKey(js, ACL_DUMP_OPNAME_RANGE)) {
        info.dumpOpNameRanges = js.at(ACL_DUMP_OPNAME_RANGE).get<std::vector<OpNameRange>>();
    }
}

static void from_json(const nlohmann::json &js, aclDumpConfig &config)
{
    if (ContainKey(js, ACL_DUMP_PATH)) {
        config.dumpPath = GetCfgStrByKey(js, ACL_DUMP_PATH);
    }
    if (ContainKey(js, ACL_DUMP_LIST)) {
        config.dumpList = js.at(ACL_DUMP_LIST).get<std::vector<ModelDumpConfig>>();
    }
    if (ContainKey(js, ACL_DUMP_MODE)) {
        config.dumpMode = GetCfgStrByKey(js, ACL_DUMP_MODE);
        ACL_LOG_DEBUG("dump_mode field parse successfully, value = %s.", config.dumpMode.c_str());
    } else {
        // dump_mode is an optional field, valid values include input/output/all
        // default value is output
        config.dumpMode = ACL_DUMP_MODE_OUTPUT;
    }
    if (ContainKey(js, ACL_DUMP_OP_SWITCH)) {
        config.dumpOpSwitch = GetCfgStrByKey(js, ACL_DUMP_OP_SWITCH);
        ACL_LOG_DEBUG("dump_op_switch field parse successfully, value = %s.", config.dumpOpSwitch.c_str());
    } else {
        // dump_op_switch is an optional field, valid values include on/off
        // default value is off
        config.dumpOpSwitch = ACL_DUMP_STATUS_SWITCH_OFF;
    }
    // dump_debug is an optional field, valid values include on/off
    // default value is off
    config.dumpDebug = ACL_DUMP_STATUS_SWITCH_OFF;
    if (ContainKey(js, ACL_DUMP_DEBUG)) {
        config.dumpDebug = GetCfgStrByKey(js, ACL_DUMP_DEBUG);
        ACL_LOG_DEBUG("dump_debug field parse successfully, value = %s", config.dumpDebug.c_str());
    }

    // dump_scene is an optional field, valid values include lite_exception
    // default value is empty
    config.dumpScene.clear();
    if (ContainKey(js, ACL_DUMP_SCENE)) {
        config.dumpScene = GetCfgStrByKey(js, ACL_DUMP_SCENE);
        ACL_LOG_DEBUG("dump_scene field parse successfully, value = %s", config.dumpScene.c_str());
    }
    if (ContainKey(js, ACL_DUMP_STEP)) {
        config.dumpStep = GetCfgStrByKey(js, ACL_DUMP_STEP);
    }
    if (ContainKey(js, ACL_DUMP_DATA)) {
        config.dumpData = GetCfgStrByKey(js, ACL_DUMP_DATA);
    }
    if (ContainKey(js, ACL_DUMP_LEVEL)) {
        config.dumpLevel = GetCfgStrByKey(js, ACL_DUMP_LEVEL);
        ACL_LOG_DEBUG("dump_level field parse successfully, value = %s.", config.dumpLevel.c_str());
    } else {
        // dump_level is an optional field, valid values include op/kernel/all
        // default value is all
        config.dumpLevel = ACL_DUMP_LEVEL_ALL;
    }
    if (ContainKey(js, ACL_DUMP_STATS)) {
        config.dumpStats = js.at(ACL_DUMP_STATS).get<std::vector<std::string>>();
    }
}

static aclError HandleDumpExceptionConfig(ge::DumpConfig &dumpCfg, const aclDumpConfig &config)
{
    if (aclDumpSceneExceptions.find(config.dumpScene) == aclDumpSceneExceptions.end()) {
        return ACL_ERROR_INVALID_PARAM;
    }

    dumpCfg.dump_exception = config.dumpScene;
    dumpCfg.dump_path = config.dumpPath;
    dumpCfg.dump_status = ACL_DUMP_STATUS_SWITCH_ON;

    const char_t *ascendWorkPath = nullptr;
    MM_SYS_GET_ENV(MM_ENV_ASCEND_WORK_PATH, ascendWorkPath);
    if (ascendWorkPath != nullptr) {
        dumpCfg.dump_path = ascendWorkPath;
        ACL_LOG_INFO("get env ASCEND_WORK_PATH %s", ascendWorkPath);
    }

    ACL_LOG_INFO("convert to ge dump config successfully, enable %s dump, path=%s",
                 config.dumpScene.c_str(), dumpCfg.dump_path.c_str());
    return ACL_SUCCESS;
}

static aclError HandleDumpDebugConfig(ge::DumpConfig &dumpCfg, const aclDumpConfig &config)
{
    if (dumpCfg.dump_debug != ACL_DUMP_STATUS_SWITCH_ON) {
        return ACL_ERROR_INVALID_PARAM;
    }

    dumpCfg.dump_path = config.dumpPath;
    dumpCfg.dump_status = ACL_DUMP_STATUS_SWITCH_OFF;
    dumpCfg.dump_step = config.dumpStep;

    ACL_LOG_INFO("convert to ge dump config successfully, dump_path = %s, dump_debug = %s, dump_step = %s",
                 dumpCfg.dump_path.c_str(), dumpCfg.dump_debug.c_str(), dumpCfg.dump_step.c_str());
    return ACL_SUCCESS;
}

static bool ProcessModelDumpConfig(ge::ModelDumpConfig &modelConfig,
                                   const ModelDumpConfig &dumpModelConfig)
{
    if (dumpModelConfig.isModelName && dumpModelConfig.modelName.empty()) {
        ACL_LOG_WARN("[Check][modelName]the modelName field is null");
        return false;
    }

    if (dumpModelConfig.isLayer && dumpModelConfig.layer.empty()) {
        ACL_LOG_WARN("[Check][Layer]layer field is null in model %s",
                     dumpModelConfig.modelName.c_str());
        return false;
    }

    modelConfig.model_name = dumpModelConfig.modelName;

    // Process opname blacklist
    for (const auto &item : dumpModelConfig.opnameBlacklist) {
        modelConfig.opname_blacklist.emplace_back(ge::DumpBlacklist{item.name, item.pos});
    }

    // Process optype blacklist
    for (const auto &item : dumpModelConfig.optypeBlacklist) {
        modelConfig.optype_blacklist.emplace_back(ge::DumpBlacklist{item.name, item.pos});
    }

    // Process layers
    modelConfig.layers.assign(dumpModelConfig.layer.begin(), dumpModelConfig.layer.end());

    // Process watcher nodes
    modelConfig.watcher_nodes.assign(dumpModelConfig.watcherNodes.begin(),
                                     dumpModelConfig.watcherNodes.end());

    // Process dump op ranges
    for (const auto &range : dumpModelConfig.dumpOpNameRanges) {
        modelConfig.dump_op_ranges.emplace_back(std::make_pair(range.begin, range.end));
    }
    return true;
}

static aclError SetUpDumpConfig(ge::DumpConfig &dumpCfg, const aclDumpConfig &config)
{
    // Handle exception dump config
    if (aclDumpSceneExceptions.find(config.dumpScene) != aclDumpSceneExceptions.end()) {
        return HandleDumpExceptionConfig(dumpCfg, config);
    }

    // Handle debug dump config
    dumpCfg.dump_debug = config.dumpDebug;
    if (dumpCfg.dump_debug == ACL_DUMP_STATUS_SWITCH_ON) {
        return HandleDumpDebugConfig(dumpCfg, config);
    }

    // Set basic dump config
    dumpCfg.dump_path = config.dumpPath;
    dumpCfg.dump_mode = config.dumpMode;
    dumpCfg.dump_step = config.dumpStep;
    dumpCfg.dump_op_switch = config.dumpOpSwitch;
    dumpCfg.dump_data = config.dumpData;
    dumpCfg.dump_level = config.dumpLevel;
    dumpCfg.dump_status = ((dumpCfg.dump_level == ACL_DUMP_LEVEL_OP) ||
                           (dumpCfg.dump_level == ACL_DUMP_LEVEL_ALL))
        ? ACL_DUMP_STATUS_SWITCH_ON
        : ACL_DUMP_STATUS_SWITCH_OFF;

    // Process dump list
    for (const auto &item : config.dumpList) {
        ge::ModelDumpConfig modelConfig;
        if (ProcessModelDumpConfig(modelConfig, item)) {
            dumpCfg.dump_list.emplace_back(std::move(modelConfig));
        }
    }

    // Process dump stats
    dumpCfg.dump_stats.assign(config.dumpStats.begin(), config.dumpStats.end());

    ACL_LOG_INFO("convert to ge dump config successfully, dump_mode = %s, dump_path = %s, "
                 "dump_op_switch = %s, dump_step = %s, dump_data = %s, dumplist size is %zu",
                 dumpCfg.dump_mode.c_str(), dumpCfg.dump_path.c_str(),
                 dumpCfg.dump_op_switch.c_str(), dumpCfg.dump_step.c_str(),
                 dumpCfg.dump_data.c_str(), dumpCfg.dump_list.size());

    return ACL_SUCCESS;
}
}

namespace acl {
// --------------------------------initialize----------------------------------------------------------------------
aclError AclMdlInitCallbackFunc(const char *configStr, size_t len, void *userData)
{
    (void)configStr;
    (void)len;
    (void)userData;
    ACL_LOG_INFO("start to enter AclMdlInitCallbackFunc");
    // init GeExecutor
    ge::GeExecutor executor;
    ACL_LOG_INFO("call ge interface executor.Initialize");
    auto geRet = executor.Initialize();
    ACL_REQUIRES_CALL_GE_OK(geRet, "[Init][Geexecutor]init ge executor failed, ge errorCode = %u", geRet);
    return ACL_SUCCESS;
}
__attribute__((constructor)) aclError RegAclMdlInitCallback()
{
    return aclInitCallbackRegisterImpl(ACL_REG_TYPE_ACL_MODEL, AclMdlInitCallbackFunc, nullptr);
}
__attribute__((destructor)) aclError UnRegAclMdlInitCallback()
{
    return aclInitCallbackUnRegisterImpl(ACL_REG_TYPE_ACL_MODEL, AclMdlInitCallbackFunc);
}

aclError ResourceInitCallbackFunc(const char *configStr, size_t len, void *userData)
{
    (void)configStr;
    (void)len;
    (void)userData;
    ACL_LOG_INFO("start to enter ResourceInitCallbackFunc");
    // register ge release function by stream
    auto rtErr = rtRegStreamStateCallback("ACL_MODULE_STREAM_MODEL", &HandleReleaseSourceByStream);
    if (rtErr != RT_ERROR_NONE) {
        ACL_LOG_ERROR("register release function by stream to runtime failed, ret:%d", rtErr);
        return ACL_GET_ERRCODE_RTS(rtErr);
    }

    // register ge release function by device
    rtErr= rtRegDeviceStateCallbackEx("ACL_MODULE_DEVICE", &HandleReleaseSourceByDevice, DEV_CB_POS_FRONT);
    if (rtErr != RT_ERROR_NONE) {
        ACL_LOG_ERROR("register release function by device to runtime failed, ret:%d", rtErr);
        return ACL_GET_ERRCODE_RTS(rtErr);
    }
    return ACL_SUCCESS;
}
__attribute__((constructor)) aclError RegResourceInitCallback()
{
    return aclInitCallbackRegisterImpl(ACL_REG_TYPE_OTHER, ResourceInitCallbackFunc, nullptr);
}
__attribute__((destructor)) aclError UnRegResourceInitCallback()
{
    return aclInitCallbackUnRegisterImpl(ACL_REG_TYPE_OTHER, ResourceInitCallbackFunc);
}

// --------------------------------finalize----------------------------------------------------------------------
aclError AclMdlFinalizeCallbackFunc(void *userData)
{
    (void)userData;
    ACL_LOG_INFO("start to enter AclMdlFinalizeCallbackFunc");
    // Finalize GeExecutor
    ge::GeExecutor executor;
    const ge::Status geRet = executor.Finalize();
    if (geRet != ge::SUCCESS) {
        ACL_LOG_ERROR("[Finalize][Ge]finalize ge executor failed, ge errorCode = %u", geRet);
        return ACL_GET_ERRCODE_GE(geRet);
    }
    return ACL_SUCCESS;
}
__attribute__((constructor)) aclError RegAclMdlFinalizeCallback()
{
    return aclFinalizeCallbackRegisterImpl(ACL_REG_TYPE_ACL_MODEL, AclMdlFinalizeCallbackFunc, nullptr);
}
__attribute__((destructor)) aclError UnRegAclMdlFinalizeCallback()
{
    return aclFinalizeCallbackUnRegisterImpl(ACL_REG_TYPE_ACL_MODEL, AclMdlFinalizeCallbackFunc);
}

aclError ResourceFinalizeCallbackFunc(void *userData)
{
    (void)userData;
    ACL_LOG_INFO("start to enter ResourceFinalizeCallbackFunc");
    // unregister ge release function by stream
    auto rtErr = rtRegStreamStateCallback("ACL_MODULE_STREAM_MODEL", nullptr);
    if (rtErr != RT_ERROR_NONE) {
        ACL_LOG_ERROR("unregister release function by stream to runtime failed, ret:%d", rtErr);
        return ACL_GET_ERRCODE_RTS(rtErr);
    }

    // unregister ge release function by device
    rtErr = rtRegDeviceStateCallbackEx("ACL_MODULE_DEVICE", nullptr, DEV_CB_POS_FRONT);
    if (rtErr != RT_ERROR_NONE) {
        ACL_LOG_ERROR("unregister release function by device to runtime failed, ret:%d", rtErr);
        return ACL_GET_ERRCODE_RTS(rtErr);
    }
    return ACL_SUCCESS;
}
__attribute__((constructor)) aclError RegResourceFinalizeCallback()
{
    return aclFinalizeCallbackRegisterImpl(ACL_REG_TYPE_OTHER, ResourceFinalizeCallbackFunc, nullptr);
}
__attribute__((destructor)) aclError UnRegResourceFinalizeCallback()
{
    return aclFinalizeCallbackUnRegisterImpl(ACL_REG_TYPE_OTHER, ResourceFinalizeCallbackFunc);
}

// ----------------------------------dump callback -------------------------------------------------
aclError DumpSetCallbackFunc(const char *configStr)
{
    ACL_LOG_INFO("start to enter DumpCallbackFunc");
    nlohmann::json js;
    aclError ret = acl::JsonParser::ParseJson(configStr, js);
    if (ret != ACL_SUCCESS) {
        ACL_LOG_ERROR("Parse dump config from buffer failed, errorCode = %d", ret);
        return ret;
    }
    if (!ContainKey(js, ACL_DUMP)) {
        return ACL_SUCCESS;
    }
    aclDumpConfig aclCfg = js.at(ACL_DUMP);
    ge::DumpConfig dumpCfg;
    SetUpDumpConfig(dumpCfg, aclCfg);
    ge::GeExecutor geExecutor;
    const ge::Status geRet = geExecutor.SetDump(dumpCfg);
    if (geRet != ge::SUCCESS) {
        ACL_LOG_ERROR("[Set][Dump]set dump config for model failed, ge errorCode = %d", geRet);
        return ACL_GET_ERRCODE_GE(static_cast<int32_t>(geRet));
    }
    return ACL_SUCCESS;
}

__attribute__((constructor)) aclError RegAclMdlSetDumpCallback()
{
    return aclDumpSetCallbackRegister(DumpSetCallbackFunc);
}
__attribute__((destructor)) aclError UnRegAclMdlSetDumpCallback()
{
    return aclDumpSetCallbackUnRegister();
}

aclError DumpSetCallbackFunc()
{
    ge::DumpConfig dumpCfg;
    ge::GeExecutor geExecutor;
    // clear dump config
    dumpCfg.dump_status = ACL_DUMP_STATUS_SWITCH_OFF;
    dumpCfg.dump_debug = ACL_DUMP_STATUS_SWITCH_OFF;
    const ge::Status geRet = geExecutor.SetDump(dumpCfg);
    if (geRet != ge::SUCCESS) {
        ACL_LOG_CALL_ERROR("[Clear][DumpConfig]Clear dump config failed, ge errorCode = %d", geRet);
        return ACL_GET_ERRCODE_GE(static_cast<int32_t>(geRet));
    }
    return ACL_SUCCESS;
}
__attribute__((constructor)) aclError RegAclMdlUnsetDumpCallback()
{
    return aclDumpUnsetCallbackRegister(DumpSetCallbackFunc);
}
__attribute__((destructor)) aclError UnRegAclMdlUnsetDumpCallback()
{
    return aclDumpUnsetCallbackUnRegister();
}
}