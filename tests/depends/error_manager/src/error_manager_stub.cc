/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <fstream>
#include <iostream>
#include <mutex>
#include <nlohmann/json.hpp>
#include <sstream>
#include <securec.h>
#include "mmpa/mmpa_api.h"
#include "dlog_pub.h"
#include "graph/def_types.h"
#include "base/err_msg.h"
#include "common/util/error_manager/error_manager.h"

#define GE_MODULE_NAME static_cast<int32_t>(GE)
namespace {
const std::string kParamCheckErrorSuffix = "8888";
class GeLog {
 public:
  static uint64_t GetTid() {
#ifdef __GNUC__
    thread_local static const uint64_t tid = static_cast<uint64_t>(syscall(__NR_gettid));
#else
    thread_local static const uint64_t tid = static_cast<uint64_t>(GetCurrentThreadId());
#endif
    return tid;
  }
};

inline bool IsLogEnable(const int32_t module_name, const int32_t log_level) {
  const int32_t enable = CheckLogLevel(module_name, log_level);
  // 1:enable, 0:disable
  return (enable == 1);
}

std::string CurrentTimeFormatStr() {
  std::string time_str;
  auto now = std::chrono::system_clock::now();
  auto milli_seconds = std::chrono::time_point_cast<std::chrono::milliseconds>(now);
  auto micro_seconds = std::chrono::time_point_cast<std::chrono::microseconds>(now);
  const auto now_t = std::chrono::system_clock::to_time_t(now);
  const std::tm *tm_now = std::localtime(&now_t);
  if (tm_now == nullptr) {
    return time_str;
  }

  constexpr int32_t year_base = 1900;
  constexpr size_t kMaxTimeLen = 128U;
  constexpr int64_t kOneThousandMs = 1000L;
  error_message::char_t format_time[kMaxTimeLen] = {};
  (void) snprintf_s(format_time, kMaxTimeLen, kMaxTimeLen - 1U, "%04d-%02d-%02d-%02d:%02d:%02d.%03ld.%03ld",
                    tm_now->tm_year + year_base, tm_now->tm_mon + 1, tm_now->tm_mday, tm_now->tm_hour, tm_now->tm_min,
                    tm_now->tm_sec, milli_seconds.time_since_epoch().count() % kOneThousandMs,
                    micro_seconds.time_since_epoch().count() % kOneThousandMs);
  time_str = format_time;
  return time_str;
}
}

#define GELOGE(fmt, ...) \
  do {                                                                                            \
    dlog_error(GE_MODULE_NAME, "%" PRIu64 " %s: %s" fmt, GeLog::GetTid(), &__FUNCTION__[0],       \
               ErrorManager::GetInstance().GetLogHeader().c_str(), ##__VA_ARGS__);                \
  } while (false)

#define GELOGW(fmt, ...)                                                                          \
  do {                                                                                            \
    if (IsLogEnable(GE_MODULE_NAME, DLOG_WARN)) {                                                 \
      dlog_warn(GE_MODULE_NAME, "%" PRIu64 " %s:" fmt, GeLog::GetTid(), &__FUNCTION__[0],         \
                ##__VA_ARGS__);                                                                   \
    }                                                                                             \
  } while (false)

#define GELOGI(fmt, ...)                                                                          \
  do {                                                                                            \
    if (IsLogEnable(GE_MODULE_NAME, DLOG_INFO)) {                                                 \
      dlog_info(GE_MODULE_NAME, "%" PRIu64 " %s:" fmt, GeLog::GetTid(), &__FUNCTION__[0],         \
                ##__VA_ARGS__);                                                                   \
    }                                                                                             \
  } while (false)

#define GELOGD(fmt, ...)                                                                           \
  do {                                                                                             \
    if (IsLogEnable(GE_MODULE_NAME, DLOG_DEBUG)) {                                                 \
      dlog_debug(GE_MODULE_NAME, "%" PRIu64 " %s:" fmt, GeLog::GetTid(), &__FUNCTION__[0],         \
                 ##__VA_ARGS__);                                                                   \
    }                                                                                              \
  } while (false)

namespace {
int32_t ReportInnerErrorMessage(const char *file_name, const char *func, uint32_t line, const char *error_code,
                                const char *format, va_list arg_list) {
  std::vector<char> buf(LIMIT_PER_MESSAGE, '\0');
  auto ret = vsprintf_s(buf.data(), LIMIT_PER_MESSAGE, format, arg_list);
  if (ret < 0) {
    GELOGE("[Check][Param] FormatErrorMessage failed, ret:%d, file:%s, line:%u", ret, file_name, line);
    return -1;
  }
  ret = sprintf_s(buf.data() + ret, LIMIT_PER_MESSAGE - static_cast<size_t>(ret), "[FUNC:%s][FILE:%s][LINE:%u]",
                  func, error_message::TrimPath(std::string(file_name)).c_str(), line);
  if (ret < 0) {
    GELOGE("[Check][Param] FormatErrorMessage failed, ret:%d, file:%s, line:%u", ret, file_name, line);
    return -1;
  }

  return ErrorManager::GetInstance().ReportInterErrMessage(error_code, std::string(buf.data()));
}

std::unique_ptr<error_message::char_t[]> CreateUniquePtrFromString(const std::string &str) {
  const size_t buf_size = str.empty() ? 1 : (str.size() + 1);
  auto uni_ptr = std::unique_ptr<error_message::char_t[]>(new (std::nothrow) error_message::char_t[buf_size]);
  if (uni_ptr == nullptr) {
    return nullptr;
  }

  if (str.empty()) {
    uni_ptr[0U] = '\0';
  } else {
    // 当src size < dst size时，strncpy_s会在末尾str.size()位置添加'\0'
    if (strncpy_s(uni_ptr.get(), str.size() + 1, str.c_str(), str.size()) != EOK) {
      return nullptr;
    }
  }
  return uni_ptr;
}

void ClearMessageContainerByWorkId(std::map<uint64_t, std::vector<ErrorManager::ErrorItem>> &message_container,
                                   const uint64_t work_stream_id) {
  const std::map<uint64_t, std::vector<ErrorManager::ErrorItem>>::const_iterator err_iter =
      message_container.find(work_stream_id);
  if (err_iter != message_container.cend()) {
    (void) message_container.erase(err_iter);
  }
}

std::vector<ErrorManager::ErrorItem> &GetOrCreateMessageContainerByWorkId(
    std::map<uint64_t, std::vector<ErrorManager::ErrorItem>> &message_container, uint64_t work_id) {
  auto iter = message_container.find(work_id);
  if (iter == message_container.end()) {
    (void) message_container.emplace(work_id, std::vector<ErrorManager::ErrorItem>());
    iter = message_container.find(work_id);
  }
  return iter->second;
}
}  // namespace


namespace error_message {
// first stage
const std::string kInitialize   = "INIT";
const std::string kModelCompile = "COMP";
const std::string kModelLoad    = "LOAD";
const std::string kModelExecute = "EXEC";
const std::string kFinalize     = "FINAL";

// SecondStage
// INITIALIZE
const std::string kParser               = "PARSER";
const std::string kOpsProtoInit         = "OPS_PRO";
const std::string kSystemInit           = "SYS";
const std::string kEngineInit           = "ENGINE";
const std::string kOpsKernelInit        = "OPS_KER";
const std::string kOpsKernelBuilderInit = "OPS_KER_BLD";
// MODEL_COMPILE
const std::string kPrepareOptimize    = "PRE_OPT";
const std::string kOriginOptimize     = "ORI_OPT";
const std::string kSubGraphOptimize   = "SUB_OPT";
const std::string kMergeGraphOptimize = "MERGE_OPT";
const std::string kPreBuild           = "PRE_BLD";
const std::string kStreamAlloc        = "STM_ALLOC";
const std::string kMemoryAlloc        = "MEM_ALLOC";
const std::string kTaskGenerate       = "TASK_GEN";
// COMMON
const std::string kOther = "DEFAULT";

#ifdef __GNUC__
std::string TrimPath(const std::string &str) {
  if (str.find_last_of('/') != std::string::npos) {
    return str.substr(str.find_last_of('/') + 1U);
  }
  return str;
}
#else
std::string TrimPath(const std::string &str) {
  if (str.find_last_of('\\') != std::string::npos) {
    return str.substr(str.find_last_of('\\') + 1U);
  }
  return str;
}
#endif

int32_t FormatErrorMessage(char_t *str_dst, size_t dst_max, const char_t *format, ...) {
  int32_t ret;
  va_list arg_list;

  va_start(arg_list, format);
  ret = vsprintf_s(str_dst, dst_max, format, arg_list);
  (void)arg_list;
  va_end(arg_list);
  if (ret < 0) {
    GELOGE("[Check][Param] FormatErrorMessage failed, ret:%d, pattern:%s", ret, format);
  }
  return ret;
}

void ReportInnerError(const char_t *file_name, const char_t *func, uint32_t line, const std::string error_code,
                      const char_t *format, ...) {
  va_list arg_list;
  va_start(arg_list, format);
  (void)ReportInnerErrorMessage(file_name, func, line, error_code.c_str(), format, arg_list);
  va_end(arg_list);
  return;
}
}

namespace {
#ifdef __GNUC__
constexpr const error_message::char_t *const kErrorCodePath = "../conf/error_manager/error_code.json";
constexpr const error_message::char_t *const kSeparator = "/";
#else
const error_message::char_t *const kErrorCodePath = "..\\conf\\error_manager\\error_code.json";
const error_message::char_t *const kSeparator = "\\";
#endif

constexpr uint64_t kLength = 2UL;

void Ltrim(std::string &s) {
  (void) s.erase(s.begin(),
                 std::find_if(s.begin(),
                              s.end(),
                              [](const error_message::char_t c) -> bool {
                                return static_cast<bool>(std::isspace(static_cast<uint8_t>(c)) == 0);
                              }));
}

void Rtrim(std::string &s) {
  (void) s.erase(std::find_if(s.rbegin(),
                              s.rend(),
                              [](const error_message::char_t c) -> bool {
                                return static_cast<bool>(std::isspace(static_cast<uint8_t>(c)) == 0);
                              }).base(),
                 s.end());
}

/// @ingroup domi_common
/// @brief trim space
void Trim(std::string &s) {
  Rtrim(s);
  Ltrim(s);
}

// split string
std::vector<std::string> SplitByDelim(const std::string &str, const error_message::char_t delim) {
  std::vector<std::string> elems;

  if (str.empty()) {
    elems.emplace_back("");
    return elems;
  }

  std::stringstream ss(str);
  std::string item;

  while (getline(ss, item, delim)) {
    Trim(item);
    elems.push_back(item);
  }
  const auto str_size = str.size();
  if ((str_size > 0U) && (str[str_size - 1U] == delim)) {
    elems.emplace_back("");
  }

  return elems;
}
}  // namespace


thread_local error_message::Context ErrorManager::error_context_ = {0UL, "", "", ""};

struct StubErrorItem {
  std::string error_id;
  std::string error_title;
  std::string error_message;
  std::string possible_cause;
  std::string solution;
  std::map<std::string, std::string> args_map;
  std::string report_time;

  friend bool operator==(const StubErrorItem &lhs, const StubErrorItem &rhs) noexcept {
    return (lhs.error_id == rhs.error_id) && (lhs.error_message == rhs.error_message) &&
        (lhs.possible_cause == rhs.possible_cause) && (lhs.solution == rhs.solution);
  }
};

std::mutex stub_mutex_;
static std::vector<StubErrorItem> stub_error_message_process_;

///
/// @brief Obtain ErrorManager instance
/// @return ErrorManager instance
///
ErrorManager &ErrorManager::GetInstance() {
  static ErrorManager instance;
  return instance;
}

///
/// @brief init
/// @param [in] path: current so path
/// @return int 0(success) -1(fail)
///
int32_t ErrorManager::Init(const std::string path) {
  std::lock_guard<std::mutex> lock(mutex_);
  error_message_process_.clear();
  return 0;
}

///
/// @brief init
/// @return int 0(success) -1(fail)
///
int32_t ErrorManager::Init() {
  std::lock_guard<std::mutex> lock(mutex_);
  error_message_process_.clear();
  return 0;
}

int32_t ErrorManager::Init(error_message::ErrorMsgMode error_mode) {
  std::lock_guard<std::mutex> lock(stub_mutex_);
  stub_error_message_process_.clear();
  return 0;
}

int32_t ErrorManager::ReportInterErrMessage(const std::string error_code, const std::string &error_msg) {
  std::lock_guard<std::mutex> lock(mutex_);
  ErrorItem item = {error_code, "", error_msg};
  error_message_process_.emplace_back(item);
  return 0;
}

///
/// @brief report error message
/// @param [in] error_code: error code
/// @param [in] args_map: parameter map
/// @return int 0(success) -1(fail)
///
int32_t ErrorManager::ReportErrMessage(const std::string error_code,
                                       const std::map<std::string, std::string> &args_map) {
  return 0;
}

int32_t ErrorManager::ReportErrMsgWithoutTpl(const std::string &error_code, const std::string &errmsg) {
  std::string report_time = CurrentTimeFormatStr();
  if (!is_init_) {
    const auto ret = Init();
    if (ret == -1) {
      GELOGI("ErrorManager has not been initialized, can't report error_message.");
      return -1;
    }
  }

  if (error_context_.work_stream_id == 0UL) {
    GenWorkStreamIdDefault();
  }

  auto final_error_code = error_code;
  if (!IsUserDefinedErrorCode(final_error_code)) {
    GELOGW("[Report] Current error code is [%s], suggest using the recommended U segment. "
           "The error code EU0000 is reported!", final_error_code.c_str());
    final_error_code = "EU0000";
  }

  GELOGI("report error_message, error_code:%s, work_stream_id:%lu, error_mode:%u.",
         error_code.c_str(), error_context_.work_stream_id, error_mode_);

  const std::unique_lock<std::mutex> lock(mutex_);
  auto &error_messages = GetErrorMsgContainer(error_context_.work_stream_id);

  ErrorItem error_item{final_error_code, "", errmsg, "", "", {}, report_time};
  const auto it = find(error_messages.begin(), error_messages.end(), error_item);
  if (it == error_messages.end()) {
    error_messages.emplace_back(error_item);
  }
  return 0;
}

void ErrorManager::AssembleInnerErrorMessage(const std::vector<ErrorItem> &error_messages,
                                             const std::string &first_code,
                                             std::stringstream &err_stream) const {
  std::string current_code_print = first_code;
  const bool IsErrorId = IsParamCheckErrorId(first_code);
  for (auto &item : error_messages) {
    if (!IsParamCheckErrorId(item.error_id)) {
      current_code_print = item.error_id;
      break;
    }
  }
  err_stream << current_code_print << ": Inner Error!" << std::endl;
  bool print_traceback_once = false;
  for (auto &item : error_messages) { // Display the first non 8888 error code
    if (IsParamCheckErrorId(item.error_id) && IsErrorId) {
      err_stream << "        " << item.error_message << std::endl;
      continue;
    }
    current_code_print == "      "
    ? (err_stream << current_code_print << " " << item.error_message << std::endl)
    : (err_stream << current_code_print << "[PID: " << std::to_string(mmGetPid()) << "] " << item.report_time
                  << " " << item.error_title << "(" << item.error_id << "): "
                  << " " << item.error_message << std::endl);

    current_code_print = "      ";
    if (!print_traceback_once) {
      err_stream << "        TraceBack (most recent call last):" << std::endl;
      print_traceback_once = true;
    }
  }
}

std::string ErrorManager::GetErrorMessage() {
  std::lock_guard<std::mutex> lock(mutex_);
  if (!error_message_process_.empty()) {
    return error_message_process_.front().error_message;
  } else {
    return std::string();
  }
}

std::string ErrorManager::GetWarningMessage() {
  GELOGI("current work_stream_id:%lu, error_mode:%u", error_context_.work_stream_id, error_mode_);
  const std::unique_lock<std::mutex> lck(mutex_);
  auto &warning_messages = GetWarningMsgContainer(error_context_.work_stream_id);

  std::stringstream warning_stream;
  for (auto &item : warning_messages) {
    warning_stream << "[PID: " << std::to_string(mmGetPid()) << "] " << item.report_time << " " << item.error_title
                   << "(" << item.error_id << "): " << item.error_message << std::endl;
  }
  ClearWarningMsgContainer(error_context_.work_stream_id);
  return warning_stream.str();
}

///
/// @brief output error message
/// @param [in] handle: print handle
/// @return int 0(success) -1(fail)
///
int32_t ErrorManager::OutputErrMessage(int32_t handle) {
  std::string err_msg = GetErrorMessage();

  if (err_msg.empty()) {
    std::stringstream err_stream;
    err_stream << "E19999: Inner Error!" << std::endl;
    err_stream << "        " << "Unknown error occurred. Please check the log." << std::endl;
    err_msg = err_stream.str();
  }

  if (handle <= fileno(stderr)) {
    std::cout << err_msg << std::endl;
  } else {
    const mmSsize_t ret =
        mmWrite(handle, const_cast<error_message::char_t *>(err_msg.c_str()), static_cast<uint32_t>(err_msg.length()));
    if (ret == -1) {
      GELOGE("[Write][File]fail, reason:%s",  strerror(errno));
      return -1;
    }
  }
  return 0;
}

///
/// @brief output message
/// @param [in] handle: print handle
/// @return int 0(success) -1(fail)
///
int32_t ErrorManager::OutputMessage(int32_t handle) {
  const std::string warning_msg = GetWarningMessage();
  std::cout << warning_msg << std::endl;
  handle = 0;
  return handle;
}

int32_t ErrorManager::ParseJsonFile(const std::string path) {
  return 0;
}
///
/// @brief parse json file
/// @param [in] handle: json handle
/// @return int 0(success) -1(fail)
///
int32_t ErrorManager::ParseJsonFormatString(const void *const handle) {
  return 0;
}

///
/// @brief read json file
/// @param [in] file_path: json path
/// @param [in] handle:  print handle
/// @return int 0(success) -1(fail)
///
int32_t ErrorManager::ReadJsonFile(const std::string &file_path, void *const handle) {
  return 0;
}

///
/// @brief report error message
/// @param [in] error_code: error code
/// @param [in] vector parameter key, vector parameter value
/// @return int 0(success) -1(fail)
///
void ErrorManager::ATCReportErrMessage(const std::string error_code, const std::vector<std::string> &key,
                                       const std::vector<std::string> &value) {
  if (!is_init_) {
    const int32_t kRetInit = Init();
    if (kRetInit == -1) {
      GELOGI("ErrorManager has not been initialized, can't report error_message.");
      return;
    }
  }
  std::map<std::string, std::string> args_map;
  if (key.empty()) {
    (void)ErrorManager::GetInstance().ReportErrMessage(error_code, args_map);
  } else if (key.size() == value.size()) {
    for (size_t i = 0UL; i < key.size(); ++i) {
      (void)args_map.insert(std::make_pair(key[i], value[i]));
    }
    (void)ErrorManager::GetInstance().ReportErrMessage(error_code, args_map);
  } else {
    GELOGW("ATCReportErrMessage wrong, vector key and value size is not equal");
  }
}

///
/// @brief report graph compile failed message such as error code and op_name in mustune case
/// @param [in] msg: failed message map, key is error code, value is op_name
/// @param [out] classified_msg: classified_msg message map, key is error code, value is op_name vector
///
void ErrorManager::ClassifyCompileFailedMsg(const std::map<std::string, std::string> &msg,
                                            std::map<std::string,
                                            std::vector<std::string>> &classified_msg) {
  for (const auto &itr : msg) {
    GELOGD("msg is error_code:%s, op_name:%s", itr.first.c_str(), itr.second.c_str());
    const auto err_code_itr = classified_msg.find(itr.first);
    if (err_code_itr == classified_msg.end()) {
      (void)classified_msg.emplace(itr.first, std::vector<std::string>{itr.second});
    } else {
      std::vector<std::string> &op_name_list = err_code_itr->second;
      op_name_list.emplace_back(itr.second);
    }
  }
}

///
/// @brief report graph compile failed message such as error code and op_name in mustune case
/// @param [in] root_graph_name: root graph name
/// @param [in] msg: failed message map, key is error code, value is op_name
/// @return int 0(success) -1(fail)
///
int32_t ErrorManager::ReportMstuneCompileFailedMsg(const std::string &root_graph_name,
                                                   const std::map<std::string, std::string> &msg) {
  if (!is_init_) {
    const int32_t kRetInit = Init();
    if (kRetInit == -1) {
      GELOGI("ErrorManager has not been initialized, can't report error_message.");
      return 0;
    }
  }
  if (msg.empty() || root_graph_name.empty()) {
    GELOGW("Msg or root graph name is empty, msg size is %zu, root graph name is %s",
           msg.size(), root_graph_name.c_str());
    return -1;
  }
  GELOGD("Report graph:%s compile failed msg", root_graph_name.c_str());
  const std::unique_lock<std::mutex> lock(mutex_);
  const auto itr = compile_failed_msg_map_.find(root_graph_name);
  if (itr != compile_failed_msg_map_.end()) {
    std::map<std::string, std::vector<std::string>> &classified_msg = itr->second;
    ClassifyCompileFailedMsg(msg, classified_msg);
  } else {
    std::map<std::string, std::vector<std::string>> classified_msg;
    ClassifyCompileFailedMsg(msg, classified_msg);
    (void)compile_failed_msg_map_.emplace(root_graph_name, classified_msg);
  }
  return 0;
}

///
/// @brief get graph compile failed message in mustune case
/// @param [in] graph_name: graph name
/// @param [out] msg_map: failed message map, key is error code, value is op_name list
/// @return int 0(success) -1(fail)
///
int32_t ErrorManager::GetMstuneCompileFailedMsg(const std::string &graph_name, std::map<std::string,
std::vector<std::string>> &msg_map) {
  if (!is_init_) {
    const int32_t kRetInit = Init();
    if (kRetInit == -1) {
      GELOGI("ErrorManager has not been initialized, can't report error_message.");
      return 0;
    }
  }
  if (!msg_map.empty()) {
    GELOGW("msg_map is not empty, exist msg");
    return -1;
  }
  const std::unique_lock<std::mutex> lock(mutex_);
  const auto iter = compile_failed_msg_map_.find(graph_name);
  if (iter == compile_failed_msg_map_.end()) {
    GELOGW("can not find graph, name is:%s", graph_name.c_str());
    return -1;
  } else {
    auto &compile_failed_msg = iter->second;
    msg_map.swap(compile_failed_msg);
    (void)compile_failed_msg_map_.erase(graph_name);
  }
  GELOGI("get graph:%s compile result msg success", graph_name.c_str());

  return 0;
}

std::vector<ErrorManager::ErrorItem> &ErrorManager::GetErrorMsgContainerByWorkId(uint64_t work_id) {
  return GetOrCreateMessageContainerByWorkId(error_message_per_work_id_, work_id);
}

std::vector<ErrorManager::ErrorItem> &ErrorManager::GetWarningMsgContainerByWorkId(uint64_t work_id) {
  return GetOrCreateMessageContainerByWorkId(warning_messages_per_work_id_, work_id);
}

std::vector<ErrorManager::ErrorItem> &ErrorManager::GetErrorMsgContainer(uint64_t work_stream_id) {
  return (error_mode_ == error_message::ErrorMsgMode::INTERNAL_MODE) ?
         GetErrorMsgContainerByWorkId(work_stream_id) : error_message_process_;
}

std::vector<ErrorManager::ErrorItem> &ErrorManager::GetWarningMsgContainer(uint64_t work_stream_id) {
  return (error_mode_ == error_message::ErrorMsgMode::INTERNAL_MODE) ?
         GetWarningMsgContainerByWorkId(work_stream_id) : warning_messages_process_;
}

void ErrorManager::GenWorkStreamIdDefault() {
  // system getpid and gettid is always successful
  const int32_t pid = mmGetPid();
  const int32_t tid = mmGetTid();

  constexpr uint64_t kPidOffset = 100000UL;
  const uint64_t work_stream_id = static_cast<uint64_t>(static_cast<uint32_t>(pid) * kPidOffset) +
      static_cast<uint64_t>(tid);
  error_context_.work_stream_id = work_stream_id;
}

void ErrorManager::GenWorkStreamIdBySessionGraph(const uint64_t session_id, const uint64_t graph_id) {
  constexpr uint64_t kSessionIdOffset = 100000UL;
  const uint64_t work_stream_id = (session_id * kSessionIdOffset) + graph_id;
  error_context_.work_stream_id = work_stream_id;

  const std::unique_lock<std::mutex> lck(mutex_);
  ClearErrorMsgContainerByWorkId(work_stream_id);
  ClearWarningMsgContainerByWorkId(work_stream_id);
}

void ErrorManager::GenWorkStreamIdWithSessionIdGraphId(const uint64_t session_id, const uint64_t graph_id) {
  constexpr uint64_t kSessionIdOffset = 100000UL;
  const uint64_t work_stream_id = (session_id * kSessionIdOffset) + graph_id;
  error_context_.work_stream_id = work_stream_id;
}

void ErrorManager::ClearErrorMsgContainerByWorkId(const uint64_t work_stream_id) {
  return ClearMessageContainerByWorkId(error_message_per_work_id_, work_stream_id);
}

void ErrorManager::ClearWarningMsgContainerByWorkId(const uint64_t work_stream_id) {
  return ClearMessageContainerByWorkId(warning_messages_per_work_id_, work_stream_id);
}

void ErrorManager::ClearErrorMsgContainer(const uint64_t work_stream_id) {
  if (error_mode_ == error_message::ErrorMsgMode::PROCESS_MODE) {
    error_message_process_.clear();
  } else {
    ClearErrorMsgContainerByWorkId(work_stream_id);
  }
}

void ErrorManager::ClearWarningMsgContainer(const uint64_t work_stream_id) {
  if (error_mode_ == error_message::ErrorMsgMode::PROCESS_MODE) {
    warning_messages_process_.clear();
  } else {
    ClearWarningMsgContainerByWorkId(work_stream_id);
  }
}

const std::string &ErrorManager::GetLogHeader() {
  if ((error_context_.first_stage == "") && (error_context_.second_stage == "")) {
    error_context_.log_header = "";
  } else {
    error_context_.log_header = "[" + error_context_.first_stage + "][" + error_context_.second_stage + "]";
  }
  return error_context_.log_header;
}

error_message::Context &ErrorManager::GetErrorManagerContext() {
  // son thread need set father thread work_stream_id, but work_stream_id cannot be zero
  // so GenWorkStreamIdDefault here directly
  if (error_context_.work_stream_id == 0UL) {
    GenWorkStreamIdDefault();
  }
  return error_context_;
}

void ErrorManager::SetErrorContext(error_message::Context error_context) {
  error_context_.work_stream_id = error_context.work_stream_id;
  error_context_.first_stage = std::move(error_context.first_stage);
  error_context_.second_stage = std::move(error_context.second_stage);
  error_context_.log_header = std::move(error_context.log_header);
}

void ErrorManager::SetStage(const std::string &first_stage, const std::string &second_stage) {
  error_context_.first_stage = first_stage;
  error_context_.second_stage = second_stage;
}

bool ErrorManager::IsInnerErrorCode(const std::string &error_code) const {
  const std::string kInterErrorCodePrefix = "9999";
  if (!IsValidErrorCode(error_code)) {
    return false;
  } else {
    return (error_code.substr(2U, 4U) == kInterErrorCodePrefix) || IsParamCheckErrorId(error_code);
  }
}

// 这里只做简单校验, 校验是非内部错误码、非预定义错误码的6位字符串即可
bool ErrorManager::IsUserDefinedErrorCode(const std::string &error_code) {
  if (!IsValidErrorCode(error_code) || IsInnerErrorCode(error_code)) {
    return false;
  }

  if (!is_init_) {
    const auto ret = Init();
    if (ret == -1) {
      GELOGI("ErrorManager has not been initialized, can't verify error code.");
      return false;
    }
  }

  if (error_map_.find(error_code) != error_map_.end()) {
    GELOGW("Report error_code:[%s] is predefined error code, suggested use U error code", error_code.c_str());
    return false;
  }
  return true;
}

bool ErrorManager::IsParamCheckErrorId(const std::string &error_code) const {
  return (error_code.substr(2U, 4U) == kParamCheckErrorSuffix);
}

std::vector<error_message::ErrorItem> ErrorManager::GetRawErrorMessages() {
  GELOGI("current work_stream_id:%lu", error_context_.work_stream_id);
  const std::unique_lock<std::mutex> lck(mutex_);
  auto error_items = GetErrorMsgContainer(error_context_.work_stream_id);
  ClearErrorMsgContainer(error_context_.work_stream_id);
  return error_items;
}

namespace error_message {
int32_t RegisterFormatErrorMessage(const char_t *error_msg, size_t error_msg_len) {
  (void)error_msg_len;
  (void)error_msg;
  return 0;
}

int32_t ReportInnerErrMsg(const char *file_name, const char *func, uint32_t line, const char *error_code,
                          const char *format, ...) {
  (void)file_name;
  (void)func;
  (void)line;
  std::lock_guard<std::mutex> lock(stub_mutex_);
  StubErrorItem item = {error_code, "", format};
  stub_error_message_process_.emplace_back(item);
  return 0;
}

int32_t ReportUserDefinedErrMsg(const char *error_code, const char *format, ...) {
  va_list arg_list;
  std::vector<char> buf(LIMIT_PER_MESSAGE, '\0');
  va_start(arg_list, format);
  const auto ret = vsprintf_s(buf.data(), LIMIT_PER_MESSAGE, format, arg_list);
  if (ret < 0) {
    GELOGE("[Check][Param] Format error message failed, ret:%d", ret);
    return -1;
  }

  return ErrorManager::GetInstance().ReportErrMsgWithoutTpl(error_code, std::string(buf.data()));
}

int32_t ReportPredefinedErrMsg(const char *error_code, const std::vector<const char *> &key,
                               const std::vector<const char *> &value) {
  if (key.size() != value.size()) {
    GELOGE("[Check][Param] ReportPredefinedErrMsg failed, vector key size:[%zu] and value size:[%zu] is not equal",
           key.size(), value.size());
    return -1;
  }
  std::map<std::string, std::string> args_map;
  for (size_t i = 0UL; i < key.size(); ++i) {
    (void)args_map.insert(std::make_pair(key[i], value[i]));
  }
  return ErrorManager::GetInstance().ReportErrMessage(error_code, args_map);
}

int32_t ReportPredefinedErrMsg(const char *error_code) {
  return ReportPredefinedErrMsg(error_code, {}, {});
}

int32_t ErrMgrInit(ErrorMessageMode error_mode) {
  return ErrorManager::GetInstance().Init(static_cast<error_message::ErrorMsgMode>(error_mode));
}

ErrorManagerContext GetErrMgrContext() {
  auto ctx = ErrorManager::GetInstance().GetErrorManagerContext();
  ErrorManagerContext error_context{};
  error_context.work_stream_id = ctx.work_stream_id;
  return error_context;
}

void SetErrMgrContext(ErrorManagerContext error_context) {
  Context ctx;
  ctx.work_stream_id = error_context.work_stream_id;
  return ErrorManager::GetInstance().SetErrorContext(ctx);
}

unique_const_char_array GetErrMgrErrorMessage() {
  std::lock_guard<std::mutex> lock(stub_mutex_);
  if (!stub_error_message_process_.empty()) {
    return CreateUniquePtrFromString(stub_error_message_process_.front().error_message);
  } else {
    return CreateUniquePtrFromString("");
  }
}

unique_const_char_array GetErrMgrWarningMessage() {
  return CreateUniquePtrFromString(ErrorManager::GetInstance().GetWarningMessage());
}
}  // namespace error_message
