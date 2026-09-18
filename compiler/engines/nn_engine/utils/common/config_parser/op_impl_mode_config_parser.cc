/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "common/config_parser/op_impl_mode_config_parser.h"
#include <dirent.h>
#include <fstream>
#include <set>
#include <sstream>
#include "common/fe_log.h"
#include "common/string_utils.h"
#include "common/comm_error_codes.h"
#include "common/fe_type_utils.h"
#include "common/platform_utils.h"
#include "common/fe_report_error.h"
#include "ge/ge_api_types.h"
#include "graph/ge_context.h"

namespace fe {
namespace {
constexpr char const *kByNodeName = "[ByNodeName]";
constexpr char const *kByOpType = "[ByOpType]";
constexpr char const *kConfigFileSuffix = ".ini";
constexpr char const *kOpImplModeRelativePath = "built-in/op_impl/ai_core/tbe/impl_mode/";
constexpr char const *kAllowHF32NewRelativePath = "built-in/op_impl/ai_core/config/allow_hf32/";
constexpr char const *kAllowHF32NewFilePrefix = "ops_allow_hf32_";
constexpr char const *kOpSelectImplModeDefaultValue = "high_performance";
constexpr char const *kAllowHF32AllTrueForAtc = "true";
constexpr char const *kAllowHF32AllFalseForAtc = "false";
constexpr char const *kAllowHF32AllTrue = "1";
constexpr char const *kAllowHF32AllFalse = "0";
constexpr char const *kAllowHF32MatmulFConvF = "00";  // matmul(bit0):false,conv(bit1):false
constexpr char const *kAllowHF32MatmulTConvF = "01";  // matmul(bit0):true,conv(bit1):false
constexpr char const *kAllowHF32DefaultMode = "10";   // matmul(bit0):false,conv(bit1):true
constexpr char const *kAllowHF32MatmulTConvT = "11";  // matmul(bit0):true,conv(bit1):true
constexpr char const *kAllowHF32ModeValue_EnableHF32 = "enable_hi_float_32_execution";
constexpr char const *kAllowHF32ModeValue_EnableFP32 = "enable_float_32_execution";
const std::vector<std::string> kOpSelectImplModeVec = {"high_precision", "high_performance"};
const std::vector<std::string> kOpSelectImplModeAllVec = {"high_precision_for_all", "high_performance_for_all"};
const std::unordered_map<std::string, std::string> kAllowHF32LegacyModeMap = {
    {kAllowHF32MatmulFConvF, "allow_hf32_matmul_f_conv_f"},  {kAllowHF32MatmulTConvF, "allow_hf32_matmul_t_conv_f"},
    {kAllowHF32DefaultMode, "allow_hf32_matmul_f_conv_t"},   {kAllowHF32MatmulTConvT, "allow_hf32_matmul_t_conv_t"},
    {kAllowHF32AllTrue, "allow_hf32_matmul_t_conv_t"},       {kAllowHF32AllFalse, "allow_hf32_matmul_f_conv_f"},
    {kAllowHF32AllTrueForAtc, "allow_hf32_matmul_t_conv_t"}, {kAllowHF32AllFalseForAtc, "allow_hf32_matmul_f_conv_f"}};
const std::map<std::string, std::string> kAllowHF32ForAclnnFallbackMap = {{kAllowHF32AllTrue, "11"},
                                                                          {kAllowHF32AllFalse, "00"},
                                                                          {kAllowHF32AllTrueForAtc, "11"},
                                                                          {kAllowHF32AllFalseForAtc, "00"}};
const std::set<std::string> kSupportImpyType = {"high_performance",
                                                "enable_float_32_execution",
                                                "enable_hi_float_32_execution",
                                                "high_precision",
                                                "support_out_of_bound_index",
                                                "super_performance",
                                                "norm_class",
                                                "keep_fp16"};
const std::set<std::string> kAllowHF32ValidValues = {kAllowHF32ModeValue_EnableHF32, kAllowHF32ModeValue_EnableFP32};
// 历史默认 allow_hf32=10: matmul(bit0):false=FP32, conv(bit1):true=HF32
const std::string kAllowHF32DefaultForTrue = kAllowHF32ModeValue_EnableHF32;
const std::string kAllowHF32DefaultForFalse = kAllowHF32ModeValue_EnableFP32;
}  // namespace
OpImplModeConfigParser::OpImplModeConfigParser(const std::string &ascend_opp_path)
    : BaseConfigParser(), ascend_opp_path_(ascend_opp_path) {}

OpImplModeConfigParser::~OpImplModeConfigParser() {}

Status OpImplModeConfigParser::InitializeFromOptions(const std::map<std::string, std::string> &options) {
  std::string op_precision_mode;
  std::map<std::string, std::string>::const_iterator iter = options.find(ge::OP_PRECISION_MODE);
  if (iter != options.cend() && !iter->second.empty()) {
    op_precision_mode = iter->second;
  }
  std::string op_select_impl_mode;
  iter = options.find(ge::OP_SELECT_IMPL_MODE);
  if (iter != options.cend() && !iter->second.empty()) {
    op_select_impl_mode = iter->second;
  }
  std::string op_type_list_str;
  iter = options.find(ge::OPTYPELIST_FOR_IMPLMODE);
  if (iter != options.cend() && !iter->second.empty()) {
    op_type_list_str = iter->second;
    (void)StringUtils::Trim(op_type_list_str);
  }
  std::string allow_hf32;
  iter = options.find(ge::ALLOW_HF32);
  if (iter != options.end() && !iter->second.empty()) {
    allow_hf32 = iter->second;
    FE_LOGI("[Init][AllowHF32] Parameter ge.exec.allow_hf32 is set to %s.", allow_hf32.c_str());
  }
  UpDateDefaultValue(op_precision_mode, op_select_impl_mode, allow_hf32);
  return Initialize(op_precision_mode, op_select_impl_mode, op_type_list_str, allow_hf32);
}

Status OpImplModeConfigParser::InitializeFromContext() {
  std::string op_precision_mode;
  (void)ge::GetContext().GetOption(ge::OP_PRECISION_MODE, op_precision_mode);
  std::string op_select_impl_mode;
  (void)ge::GetContext().GetOption(ge::OP_SELECT_IMPL_MODE, op_select_impl_mode);
  std::string op_type_list_str;
  (void)ge::GetContext().GetOption(ge::OPTYPELIST_FOR_IMPLMODE, op_type_list_str);
  std::string allow_hf32;
  (void)ge::GetContext().GetOption(ge::ALLOW_HF32, allow_hf32);
  FE_LOGD("[Refresh][AllowHF32] Parameter [ge.exec.allow_hf32] is set to [%s].", allow_hf32.c_str());

  UpDateDefaultValue(op_precision_mode, op_select_impl_mode, allow_hf32);
  return Initialize(op_precision_mode, op_select_impl_mode, op_type_list_str, allow_hf32);
}

/*
 * priority: allow_hf32 > op_precision_mode > op_select_impl_mode
 *
 * A5 新算子包不再提供 high_* preset ini, 不自动注入 op_select_impl_mode 默认值
 * 仅当用户显式传入 op_select_impl_mode / optypelist_for_implmode 时处理，
 * 否则留空，默认场景不触发废弃 WARNING
 */
void OpImplModeConfigParser::UpDateDefaultValue(const std::string &op_precision_mode, std::string &op_select_impl_mode,
                                                std::string &allow_hf32) {
  if (op_select_impl_mode.empty()) {
    FE_LOGD("The value of param[%s] is empty, skip preset ini loading.", ge::OP_SELECT_IMPL_MODE.c_str());
    if (op_precision_mode.empty() && allow_hf32.empty() && PlatformUtils::Instance().IsEnableAllowHF32()) {
      FE_LOGD("[Update][AllowHF32] Parameter ge.exec.allow_hf32 uses the default value %s.", kAllowHF32DefaultMode);
      allow_hf32 = kAllowHF32DefaultMode;
    }
  }
}

// PRECISIONMODE
// 优先级: allow_hf32 > op_precision_mode > op_select_impl_mode
// 先加载 allow_hf32 （默认或显式），再加载 op_precision_mode （低优先级不覆盖）
// 最后加载 op_select_impl_mode (最低优先级)
Status OpImplModeConfigParser::Initialize(const std::string &op_precision_mode, const std::string &op_select_impl_mode,
                                          const std::string &op_type_list_for_impl_mode,
                                          const std::string &allow_hf32) {
  bool enable_allow_hf32 = PlatformUtils::Instance().IsEnableAllowHF32();
  std::string tmp_allow_hf32;
  if (enable_allow_hf32) {
    tmp_allow_hf32 = allow_hf32.empty() ? kAllowHF32DefaultMode : allow_hf32;
  }
  std::lock_guard<std::mutex> lock_guard(op_impl_mode_mutex_);
  bool not_change = op_precision_mode == op_precision_mode_ && op_select_impl_mode == op_select_impl_mode_ &&
                    op_type_list_for_impl_mode == op_type_list_for_impl_mode_ && tmp_allow_hf32 == allow_hf32_;
  if (not_change) {
    FE_LOGD("The parameters of op_impl_mode are the same as last time.");
    return SUCCESS;
  }
  op_precision_mode_ = op_precision_mode;
  op_select_impl_mode_ = op_select_impl_mode;
  op_type_list_for_impl_mode_ = op_type_list_for_impl_mode;
  allow_hf32_ = tmp_allow_hf32;
  op_name_select_impl_mode_map_.clear();
  op_type_select_impl_mode_map_.clear();

  // Step 1: 加载 allow_hf32 (最高优先级)
  if (enable_allow_hf32 && !tmp_allow_hf32.empty()) {
    if (InitExplicitAllowHF32(tmp_allow_hf32, allow_hf32) != SUCCESS) {
      return FAILED;
    }
  }

  // Step 2: 加载 op_precision_mode (中优先级)
  // 使用 insert 而非 emplace， 避免覆盖 Step 1 已加载的 allow_hf32 配置
  Status status = InitOpPrecisionMode(op_precision_mode, op_select_impl_mode, op_type_list_for_impl_mode);
  if (status != SUCCESS) {
    return status;
  }

  // Step 3: 加载默认 allow_hf32 （显式 allow_hf32 为空且 enable 时）
  if (enable_allow_hf32 && allow_hf32.empty()) {
    if (InitDefaultAllowHF32() != SUCCESS) {
      return FAILED;
    }
  }
  return SUCCESS;
}

Status OpImplModeConfigParser::InitExplicitAllowHF32(const std::string &hf32_val, const std::string &raw_allow_hf32) {
  if (HasNewAllowHF32Files()) {
    FE_LOGI("[Init][AllowHF32] Found new %s*.ini files, using new allow_hf32 config.", kAllowHF32NewFilePrefix);
    if (InitAllowHF32NewFiles(hf32_val) != SUCCESS) {
      return FAILED;
    }
  } else {
    FE_LOGI("[Init][AllowHF32] No new allow_hf32 files found, fallback to legacy files.");
    if (InitAllowHF32LegacyFiles(hf32_val) != SUCCESS) {
      ErrorMessageDetail err_msg(EM_INPUT_OPTION_INVALID,
                                 {raw_allow_hf32, ge::ALLOW_HF32, "The current value is not within the valid range"});
      ReportErrorMessage(err_msg);
      REPORT_FE_ERROR(
          "[GraphOpt][Init][InitAllowHF32] ge.exec.allow_hf32[%s] is invalid, only support [0,00,01,10,11,1].",
          raw_allow_hf32.c_str());
      return FAILED;
    }
  }
  return SUCCESS;
}

Status OpImplModeConfigParser::InitDefaultAllowHF32() {
  if (HasNewAllowHF32Files()) {
    if (InitAllowHF32NewFiles(kAllowHF32DefaultMode) != SUCCESS) {
      REPORT_FE_ERROR("[Init][allow_hf32] Init default allow_hf32 new files failed.");
      return FAILED;
    }
  } else {
    if (InitAllowHF32LegacyFiles(kAllowHF32DefaultMode) != SUCCESS) {
      REPORT_FE_ERROR("[Init][allow_hf32] Init default allow_hf32 legacy files failed.");
      return FAILED;
    }
  }
  return SUCCESS;
}

Status OpImplModeConfigParser::InitOpPrecisionMode(const std::string &op_precision_mode,
                                                   const std::string &op_select_impl_mode,
                                                   const std::string &op_type_list_str) {
  if (!op_precision_mode.empty()) {
    FE_LOGD("The value of parameter [op_precision_mode] is [%s].", op_precision_mode.c_str());
    return InitOpPrecisionModeByPrecisionMode(op_precision_mode);
  }
  FE_LOGD("The parameter [op_precision_mode] is not found or its value is empty.");

  if (op_select_impl_mode.empty()) {
    FE_LOGD("The parameter [op_select_impl_mode] is not found or its value is empty.");
    return SUCCESS;
  }
  FE_LOGD(
      "[Init][OpSelectImplMode] Parameter op_select_impl_mode is set to [%s]."
      "Note: A5 no longer provides built-in %s preset ini files."
      "Consider migrating to --op_precision_mode instead.",
      op_select_impl_mode.c_str(), op_select_impl_mode.c_str());
  if (std::find(kOpSelectImplModeAllVec.begin(), kOpSelectImplModeAllVec.end(), op_select_impl_mode) !=
      kOpSelectImplModeAllVec.end()) {
    return InitOpPrecisionModeByImplModeAll(op_select_impl_mode);
  }
  if (std::find(kOpSelectImplModeVec.begin(), kOpSelectImplModeVec.end(), op_select_impl_mode) !=
      kOpSelectImplModeVec.end()) {
    if (!op_type_list_str.empty()) {
      FE_LOGD("The value of parameter [optypelist_for_implmode] is [%s].", op_type_list_str.c_str());
    }
    return InitOpPrecisionModeByImplMode(op_select_impl_mode, op_type_list_str);
  }

  FE_LOGE("[GraphOpt][Init][InitOpPrecisionMode] Para:op_select_impl_mode[%s] is invalid.",
          op_select_impl_mode.c_str());
  ErrorMessageDetail err_msg(EM_INPUT_OPTION_INVALID, {op_select_impl_mode, ge::OP_SELECT_IMPL_MODE,
                                                       "The current value is not within the valid range"});
  ReportErrorMessage(err_msg);
  return FAILED;
}

Status OpImplModeConfigParser::InitOpPrecisionModeByPrecisionMode(const std::string &op_precision_mode) {
  if (op_precision_mode.empty()) {
    return SUCCESS;
  }
  // check whether file is existed
  std::string file_path = GetRealPath(op_precision_mode);
  if (file_path.empty()) {
    ErrorMessageDetail err_msg(EM_INPUT_OPTION_INVALID, {op_precision_mode, ge::OP_PRECISION_MODE,
                                                         "The file does not exist or its access permission is denied"});
    ReportErrorMessage(err_msg);
    REPORT_FE_ERROR(
        "[GraphOpt][Init][InitOpPrecisionMode] The op precision mode configuration file [%s] does not exist.",
        op_precision_mode.c_str());
    return FAILED;
  }
  Status ret = GetOpPrecisonModeStrFromConfigFile(file_path);
  if (ret != SUCCESS) {
    REPORT_FE_ERROR("[GraphOpt][Init][InitOpPrecisionMode] Failed to retrieve op_precision_mode string from file [%s].",
                    op_precision_mode.c_str());
    return ret;
  }
  return SUCCESS;
}

Status OpImplModeConfigParser::InitOpPrecisionModeByImplModeAll(const std::string &op_select_impl_mode_all) {
  std::string file_path = ascend_opp_path_ + kOpImplModeRelativePath + op_select_impl_mode_all + kConfigFileSuffix;
  std::string real_file_path = GetRealPath(file_path);
  if (real_file_path.empty()) {
    FE_LOGW(
        "[%s] Preset file [%s] does not exist. A5 no longer provides built-in %s preset ini. "
        "Consider migrating to --op_precision_mode.",
        op_select_impl_mode_all.c_str(), file_path.c_str(), op_select_impl_mode_all.c_str());
    return SUCCESS;
  }
  Status ret = GetOpPrecisonModeStrFromConfigFile(real_file_path);
  if (ret != SUCCESS) {
    REPORT_FE_ERROR("[GraphOpt][Init][InitOpPrecisionMode] Failed to retrieve op_precision_mode string from file [%s].",
                    op_select_impl_mode_all.c_str());
    return ret;
  }
  return SUCCESS;
}

Status OpImplModeConfigParser::InitOpPrecisionModeByImplMode(const std::string &op_select_impl_mode,
                                                             const std::string &op_type_list_str) {
  if (!op_type_list_str.empty()) {
    FE_LOGW(
        "Parameter --optypelist_for_implmode is set with --op_select_implmode. "
        "A5 no longer provides built-in preset ini files. "
        "Consider migrating to --op_precision_mode with custom ini file.");
    std::vector<std::string> op_type_list = StringUtils::Split(op_type_list_str, ',');
    for (std::string &op_type : op_type_list) {
      op_type = StringUtils::Trim(op_type);
      op_type_select_impl_mode_map_.insert(std::make_pair(op_type, op_select_impl_mode));
    }
  } else {
    string file_path = ascend_opp_path_ + kOpImplModeRelativePath + op_select_impl_mode + kConfigFileSuffix;
    std::string real_file_path = GetRealPath(file_path);
    if (real_file_path.empty()) {
      FE_LOGW(
          "[%s] Preset file [%s] does not exist. A5 no longer provides built-in %s preset ini. "
          "Consider migrating to --op_precision_mode.",
          op_select_impl_mode.c_str(), file_path.c_str(), op_select_impl_mode.c_str());
      return SUCCESS;
    }
    Status status = GetOpPrecisonModeStrFromConfigFile(real_file_path);
    if (status != SUCCESS) {
      REPORT_FE_ERROR(
          "[GraphOpt][Init][InitOpPrecisionMode] Failed to retrieve op_precision_mode string from file [%s].",
          op_select_impl_mode.c_str());
      return status;
    }
  }
  return SUCCESS;
}

// ----- allow_hf32 新方案：扫描新目录 ops_allow_hf32_*.ini -----

bool OpImplModeConfigParser::HasNewAllowHF32Files() const {
  std::string dir_path = ascend_opp_path_ + kAllowHF32NewRelativePath;
  DIR *dir = opendir(dir_path.c_str());
  if (dir == nullptr) {
    FE_LOGD("[AllowHF32] New config directory [%s] does not exist.", dir_path.c_str());
    return false;
  }
  bool found = false;
  struct dirent *entry = nullptr;
  while ((entry = readdir(dir)) != nullptr) {
    std::string name(entry->d_name);
    if (name.size() > 0 && name.find(kAllowHF32NewFilePrefix) == 0 && name.size() > strlen(kConfigFileSuffix) &&
        name.compare(name.size() - strlen(kConfigFileSuffix), strlen(kConfigFileSuffix), kConfigFileSuffix) == 0) {
      found = true;
      break;
    }
  }
  closedir(dir);
  FE_LOGD("[AllowHF32] New %s*.ini files %s.", kAllowHF32NewFilePrefix, found ? "found" : "not found");
  return found;
}

Status OpImplModeConfigParser::InitAllowHF32Mode(const std::string &allow_hf32) {
  if (HasNewAllowHF32Files()) {
    FE_LOGI("[Init][AllowHF32] Found new %s*.ini files, ignoring legacy allow_hf32_matmul_*.ini.",
            kAllowHF32NewFilePrefix);
    return InitAllowHF32NewFiles(allow_hf32);
  }
  FE_LOGI("[Init][AllowHF32] No new %s*.ini files, fallback to legacy allow_hf32_matmul_*.ini.",
          kAllowHF32NewFilePrefix);
  return InitAllowHF32LegacyFiles(allow_hf32);
}

Status OpImplModeConfigParser::InitAllowHF32NewFiles(const std::string &allow_hf32) {
  std::string dir_path = ascend_opp_path_ + kAllowHF32NewRelativePath;
  DIR *dir = opendir(dir_path.c_str());
  if (dir == nullptr) {
    FE_LOGI("[AllowHF32] New config directory [%s] does not exist, no allow_hf32 ops configured.", dir_path.c_str());
    return SUCCESS;
  }

  bool explicit_true = (allow_hf32 == kAllowHF32AllTrue || allow_hf32 == kAllowHF32AllTrueForAtc);
  bool explicit_false = (allow_hf32 == kAllowHF32AllFalse || allow_hf32 == kAllowHF32AllFalseForAtc);
  bool explicit_override = (explicit_true || explicit_false);
  std::string override_value = explicit_true ? kAllowHF32DefaultForTrue : kAllowHF32DefaultForFalse;

  struct dirent *entry = nullptr;
  bool any_file_found = false;
  while ((entry = readdir(dir)) != nullptr) {
    std::string name(entry->d_name);
    if (name.size() > 0 && name.find(kAllowHF32NewFilePrefix) == 0 && name.size() > strlen(kConfigFileSuffix) &&
        name.compare(name.size() - strlen(kConfigFileSuffix), strlen(kConfigFileSuffix), kConfigFileSuffix) == 0) {
      any_file_found = true;
      std::string file_path = dir_path + name;
      if (ParseAllowHF32IniFile(file_path) != SUCCESS) {
        closedir(dir);
        return FAILED;
      }
    }
  }
  closedir(dir);

  if (!any_file_found) {
    FE_LOGI("[AllowHF32] No %s*.ini files found in [%s].", kAllowHF32NewFilePrefix, dir_path.c_str());
    return SUCCESS;
  }

  if (explicit_override) {
    for (auto &pair : op_type_select_impl_mode_map_) {
      pair.second = override_value;
      FE_LOGD("[AllowHF32] Explicit %s: override OpType[%s] to %s.", allow_hf32.c_str(), pair.first.c_str(),
              override_value.c_str());
    }
  }
  return SUCCESS;
}

Status OpImplModeConfigParser::InitAllowHF32LegacyFiles(const std::string &allow_hf32) {
  auto iter = kAllowHF32LegacyModeMap.find(allow_hf32);
  if (iter == kAllowHF32LegacyModeMap.end()) {
    FE_LOGW("[Init][AllowHF32] ge.exec.allow_hf32[%s] is invalid, only support [0,00,01,10,11,1].", allow_hf32.c_str());
    return FAILED;
  }
  std::string allow_hf32_mode = iter->second;
  std::string file_path = ascend_opp_path_ + kOpImplModeRelativePath + allow_hf32_mode + kConfigFileSuffix;
  std::string real_file_path = GetRealPath(file_path);
  if (real_file_path.empty()) {
    FE_LOGW("Allow hf32 legacy file [%s] does not exist.", file_path.c_str());
    return SUCCESS;
  }
  Status status = GetOpPrecisonModeStrFromConfigFile(real_file_path);
  if (status != SUCCESS) {
    REPORT_FE_ERROR("[GraphOpt][Init][InitAllowHF32] Failed to retrieve allow_hf32 string from [%s] file.",
                    allow_hf32_mode.c_str());
    return status;
  }
  return SUCCESS;
}

Status OpImplModeConfigParser::ParseAllowHF32IniFile(const std::string &file_path) {
  FE_LOGD("Begin to parse allow_hf32 new file [%s].", file_path.c_str());
  std::ifstream ifs(file_path);
  if (!ifs.is_open()) {
    ErrorMessageDetail err_msg(EM_OPEN_FILE_FAILED, {file_path});
    ReportErrorMessage(err_msg);
    REPORT_FE_ERROR("[GraphOpt][ParseAllowHF32Ini] Failed to open new allow_hf32 file [%s].", file_path.c_str());
    return INVALID_FILE_PATH;
  }
  Status status = ParseAllowHF32IniFileContent(file_path, ifs);
  ifs.close();
  if (status == SUCCESS) {
    FE_LOGD("Finish parsing allow_hf32 new file [%s].", file_path.c_str());
  }
  return status;
}

Status OpImplModeConfigParser::ParseAllowHF32IniFileContent(const std::string &file_path, std::ifstream &ifs) {
  std::string line;
  bool in_by_op_type = false;
  while (std::getline(ifs, line)) {
    if (line.empty() || line.find('#') == 0) {
      continue;
    }
    size_t pos_of_equal = line.find('=');
    if (pos_of_equal == std::string::npos) {
      std::string line_tmp = StringUtils::Trim(line);
      if (line_tmp == kByOpType) {
        in_by_op_type = true;
      } else if (line_tmp == kByNodeName) {
        REPORT_FE_ERROR(
            "[GraphOpt][ParseAllowHF32Ini] File [%s] contains [ByNodeName], which is not supported in allow_hf32 ini. "
            "Only [ByOpType] is allowed.",
            file_path.c_str());
        return FAILED;
      }
      continue;
    }
    if (!in_by_op_type) {
      continue;
    }
    std::string op_type = line.substr(0, pos_of_equal);
    std::string value = line.substr(pos_of_equal + 1);
    StringUtils::Trim(op_type);
    StringUtils::Trim(value);
    if (op_type.empty() || value.empty()) {
      FE_LOGW("Allow hf32 file [%s] line [%s]: op_type or value is empty, skip.", file_path.c_str(), line.c_str());
      continue;
    }
    if (kAllowHF32ValidValues.find(value) == kAllowHF32ValidValues.end()) {
      REPORT_FE_ERROR(
          "[GraphOpt][ParseAllowHF32Ini] File [%s] line [%s]: value [%s] is invalid. "
          "Only [%s] and [%s] are allowed.",
          file_path.c_str(), line.c_str(), value.c_str(), kAllowHF32ModeValue_EnableHF32,
          kAllowHF32ModeValue_EnableFP32);
      return FAILED;
    }
    auto result = op_type_select_impl_mode_map_.insert(std::make_pair(op_type, value));
    if (!result.second && result.first->second != value) {
      REPORT_FE_ERROR(
          "[GraphOpt][ParseAllowHF32Ini] Duplicate OpType [%s] in allow_hf32 ini with different values. "
          "File: [%s], existing value: [%s], new value: [%s].",
          op_type.c_str(), file_path.c_str(), result.first->second.c_str(), value.c_str());
      return FAILED;
    }
  }
  return SUCCESS;
}
bool OpImplModeConfigParser::CheckConfigImplType(const std::string &impl_mode) const {
  std::stringstream ss;
  for (auto impl : kSupportImpyType) {
    ss << impl << "|";
  }
  ss.flush();
  auto iter = kSupportImpyType.find(impl_mode);
  if (iter == kSupportImpyType.end()) {
    FE_LOGW("impl mode %s is not in current support list: %s.", impl_mode.c_str(), ss.str().c_str());
    return false;
  }
  return true;
}

void OpImplModeConfigParser::ParseLineContentWithMode(const std::string &line_content, bool parse_by_op_type,
                                                      const size_t &pos_of_equal) {
  std::string op_type_or_name = line_content.substr(0, pos_of_equal);
  std::string impl_mode = line_content.substr(pos_of_equal + 1);
  op_type_or_name = StringUtils::Trim(op_type_or_name);
  impl_mode = StringUtils::Trim(impl_mode);
  if (op_type_or_name.empty() || impl_mode.empty()) {
    FE_LOGW("in op_precision_mode config file current line %s optype or opname or op_impl_mode is empty",
            line_content.c_str());
    return;
  }
  if (!CheckConfigImplType(impl_mode)) {
    FE_LOGW("Op_type_or_name %s with op_impl_mode %s is invalid according to the configuration.",
            op_type_or_name.c_str(), impl_mode.c_str());
    return;
  }
  if (parse_by_op_type) {
    op_type_select_impl_mode_map_.insert(std::make_pair(op_type_or_name, impl_mode));
  } else {
    op_name_select_impl_mode_map_.insert(std::make_pair(op_type_or_name, impl_mode));
  }
  return;
}

Status OpImplModeConfigParser::GetOpPrecisonModeStrFromConfigFile(const std::string &file_path) {
  FE_LOGD("Begin to load op select implementation mode file [%s].", file_path.c_str());
  std::ifstream ifs(file_path);
  if (!ifs.is_open()) {
    ErrorMessageDetail err_msg(EM_OPEN_FILE_FAILED, {file_path});
    ReportErrorMessage(err_msg);
    REPORT_FE_ERROR("[GraphOpt][InitOpPrecisionMode] Failed to open config file [%s].", file_path.c_str());
    return INVALID_FILE_PATH;
  }

  std::string line;
  bool parse_by_op_type = true;
  while (std::getline(ifs, line)) {
    if (line.empty() || line.find('#') == 0) {
      continue;
    }
    size_t pos_of_equal = line.find('=');
    if (pos_of_equal == std::string::npos) {
      std::string line_tmp = StringUtils::Trim(line);
      if (line_tmp == kByNodeName) {
        parse_by_op_type = false;
      } else if (line_tmp == kByOpType) {
        parse_by_op_type = true;
      }
      continue;
    }
    ParseLineContentWithMode(line, parse_by_op_type, pos_of_equal);
  }
  ifs.close();
  FE_LOGD("Finish parsing select implementation mode file [%s].", file_path.c_str());
  return SUCCESS;
}

bool OpImplModeConfigParser::GetOpImplModeByOpType(const std::string &op_type, std::string &op_impl_mode) const {
  std::lock_guard<std::mutex> lock_guard(op_impl_mode_mutex_);
  auto iter = op_type_select_impl_mode_map_.find(op_type);
  if (iter == op_type_select_impl_mode_map_.cend()) {
    return false;
  }
  op_impl_mode = iter->second;
  return true;
}

bool OpImplModeConfigParser::GetOpImplModeByOpName(const std::string &op_name, std::string &op_impl_mode) const {
  std::lock_guard<std::mutex> lock_guard(op_impl_mode_mutex_);
  auto iter = op_name_select_impl_mode_map_.find(op_name);
  if (iter == op_name_select_impl_mode_map_.cend()) {
    return false;
  }
  op_impl_mode = iter->second;
  return true;
}

bool OpImplModeConfigParser::GetOpImplMode(const std::string &op_name, const std::string &op_type,
                                           std::string &op_impl_mode) const {
  if (GetOpImplModeByOpName(op_name, op_impl_mode)) {
    return true;
  }
  return GetOpImplModeByOpType(op_type, op_impl_mode);
}

bool OpImplModeConfigParser::IsEnableCustomImplMode() const {
  return !op_precision_mode_.empty();
}

std::string OpImplModeConfigParser::EmplaceHf32ModeForAclnn(const std::string &hf32_mode) const {
  const auto it_hf32 = kAllowHF32ForAclnnFallbackMap.find(hf32_mode);
  if (it_hf32 != kAllowHF32ForAclnnFallbackMap.cend()) {
    FE_LOGD("Origin hf32 mode %s, new hf32 mode %s", hf32_mode.c_str(), it_hf32->second.c_str());
    return it_hf32->second;
  }
  return hf32_mode;
}
}  // namespace fe
