/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2024 All rights reserved.
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
#include "duration.h"
#include "graph/debug/ge_util.h"

namespace att {
uint32_t kg_duration_level = 0U;
namespace {
DurationDef kg_duration_def[static_cast<uint32_t>(DurationType::DURATION_MAX)] = {
  {"GEN_MODEL_INFO", 0U}
};
}

DurationManager &DurationManager::GetInstance() {
  static DurationManager ins;
  return ins;
}

DurationManager::DurationManager() {
  for (uint32_t index = 0U; index < static_cast<uint32_t>(
    DurationType::DURATION_MAX); index++) {
    AddDuration(index, kg_duration_def[index].name, kg_duration_def[index].level);
  }
}

void DurationManager::AddDuration(const uint32_t type, const std::string &name, const uint32_t level) {
  duration_infos_[type].level = level;
  duration_infos_[type].stat = std::unique_ptr<Duration>(new(std::nothrow) Duration(name));
  if (duration_infos_[type].stat == nullptr) {
    GELOGW("Create Duration failed.");
  }
}

void DurationManager::Begin(const DurationType type) {
  const auto &stat = duration_infos_[static_cast<int32_t>(type)].stat;
  if (stat == nullptr) {
    return;
  }
  stat->Begin();
}

void DurationManager::End(const DurationType type) {
  const auto &stat = duration_infos_[static_cast<int32_t>(type)].stat;
  if (stat == nullptr) {
    return;
  }
  stat->End();
}

DurationInitGuard::DurationInitGuard(const uint32_t level) {
  DurationInit(level);
}

DurationInitGuard::~DurationInitGuard() {
  DurationFinalize();
}

void DurationInit(const uint32_t level) {
  kg_duration_level = IsProfilingEnabled() ?
    static_cast<uint32_t>(TilingFuncDurationType::TILING_FUNC_DURATION_MAX) : level;
}

void DurationFinalize() {
  if (kg_duration_level > 0U) {
    DurationManager::GetInstance().Print();
    DurationManager::GetInstance().Clear();
  }
  kg_duration_level = 0U;
}

void DurationBegin(const DurationType type) {
  if (kg_duration_level > kg_duration_def[static_cast<int32_t>(type)].level) {
    DurationManager::GetInstance().Begin(type);
  }
}

void DurationEnd(const DurationType type) {
  if (kg_duration_level > kg_duration_def[static_cast<int32_t>(type)].level) {
    DurationManager::GetInstance().End(type);
  }
}

DurationDef kg_tiling_func_duration_def[static_cast<uint32_t>(
  TilingFuncDurationType::TILING_FUNC_DURATION_MAX)] = {
  {"TILING_FUNC_DURATION_TOTAL", 0U},
  {"TILING_FUNC_DURATION_DOTILING", 1U}
};

std::string DurationGenCommonCode() {
  if (kg_duration_level == 0U) {
    return "";
  }
  std::string code =
    "namespace {\n" \
    "enum DurationType {\n";
  int32_t duration_num = 0;
  for (uint32_t index = 0U; index < static_cast<uint32_t>(
    TilingFuncDurationType::TILING_FUNC_DURATION_MAX); index++) {
    if (kg_duration_level > kg_tiling_func_duration_def[index].level) {
      if (duration_num == 0) {
        code += ("  " + kg_tiling_func_duration_def[index].name + " = 0,\n");
      } else {
        code += ("  " + kg_tiling_func_duration_def[index].name + ",\n");
      }
      duration_num++;
    }
  }
  code +=
    "  TILING_FUNC_DURATION_MAX,\n" \
    "};\n" \
    "\n" \
    "struct DurationDef {\n" \
    "  std::string name;\n" \
    "};\n" \
    "\n";
  code +=
    "DurationDef g_duration_def[TILING_FUNC_DURATION_MAX] = {\n";
  for (uint32_t index = 0U; index < static_cast<uint32_t>(
    TilingFuncDurationType::TILING_FUNC_DURATION_MAX); index++) {
    if (kg_duration_level > kg_tiling_func_duration_def[index].level) {
        code += ("  {\"" + kg_tiling_func_duration_def[index].name + "\"},\n");
    }
  }
  code +=
    "};\n" \
    "\n" \
    "class Duration {\n" \
    " public:\n" \
    "  Duration(const std::string &name): name_(name) {}\n" \
    "\n" \
    "  void Begin() {\n" \
    "    call_start_ = Now();\n" \
    "  }\n" \
    "\n" \
    "  void End() {\n" \
    "    auto now = Now();\n" \
    "    uint64_t duration = now - call_start_;\n" \
    "    total_count_++;\n" \
    "    total_time_ += duration;\n" \
    "    if (duration > max_time_) max_time_ = duration;\n" \
    "    if (duration < min_time_) min_time_ = duration;\n" \
    "  }\n" \
    "\n" \
    "  void Print() {\n" \
    "    if (total_count_ == 0ULL) return;\n" \
    "    OP_EVENT(OP_NAME, \"Duration record: name[%s], total_count[%lu], total_time[%lu], " \
    "max_time[%lu], min_time[%lu], average_time[%lu].\",\n" \
    "      name_.c_str(), total_count_, total_time_, max_time_, min_time_,\n" \
    "      static_cast<uint64_t>(total_time_ / total_count_));\n" \
    "  } \n" \
    "\n" \
    "  void Clear() {\n" \
    "    total_count_ = 0ULL;\n" \
    "    total_time_ = 0ULL;\n" \
    "    max_time_ = 0ULL;\n" \
    "    min_time_ = UINT64_MAX;\n" \
    "    call_start_ = 0ULL;\n" \
    "  }\n" \
    "\n" \
    "private:\n" \
    "  uint64_t Now() {\n" \
    "    auto now = std::chrono::high_resolution_clock::now();\n" \
    "    auto nanoseconds = std::chrono::duration_cast<std::chrono::nanoseconds>(now.time_since_epoch());\n" \
    "    return static_cast<uint64_t>(nanoseconds.count());\n" \
    "  }\n" \
    "\n" \
    "  std::string name_;\n" \
    "  uint64_t total_count_ = 0ULL;\n" \
    "  uint64_t total_time_ = 0ULL;\n" \
    "  uint64_t max_time_ = 0ULL;\n" \
    "  uint64_t min_time_ = UINT64_MAX;\n" \
    "  uint64_t call_start_ = 0ULL;\n" \
    "};\n" \
    "\n" \
    "struct DurationInfo {\n" \
    "  std::unique_ptr<Duration> stat;\n" \
    "};\n" \
    "\n" \
    "constexpr size_t CASE_ID_LENGTH = 20;\n" \
    "struct IterInfo {\n" \
    "  std::array<char, CASE_ID_LENGTH> case_id;\n" \
    "  int iter_count;\n" \
    "};\n" \
    "\n" \
    "class DurationManager {\n" \
    "public:\n" \
    "  static DurationManager &GetInstance() {\n" \
    "    static DurationManager ins;\n" \
    "    return ins;\n" \
    "  }\n" \
    "\n" \
    "  DurationManager() {\n" \
    "    for (uint32_t index = 0U; index < static_cast<uint32_t>(TILING_FUNC_DURATION_MAX); index++) {\n" \
    "      AddDuration(index, g_duration_def[index].name);\n" \
    "    }\n" \
    "  }\n" \
    "  \n" \
    "  void AddDuration(const uint32_t type, const std::string &name) {\n" \
    "    if (!duration_open_now_) {\n" \
    "      return;\n" \
    "    }\n" \
    "    duration_infos_[type].stat = std::unique_ptr<Duration>(new(std::nothrow) Duration(name));\n" \
    "    if (duration_infos_[type].stat == nullptr) {\n" \
    "      OP_LOGW(OP_NAME, \"Create Duration failed.\");\n" \
    "    }\n" \
    "  }\n" \
    "\n" \
    "  void AddIterInfo(const char* case_id, uint32_t iter_count) {\n" \
    "    if (!duration_open_now_) {\n" \
    "      return;\n" \
    "    }\n" \
    "    IterInfo info;\n" \
    "    size_t len = Min(strlen(case_id), CASE_ID_LENGTH);\n" \
    "    std::copy(case_id, case_id + len, info.case_id.begin());\n" \
    "    if (len < CASE_ID_LENGTH) {\n" \
    "      info.case_id[len] = '\\0';\n" \
    "    }\n" \
    "    info.iter_count = iter_count;\n" \
    "    iter_infos_.push_back(info);\n" \
    "  }\n" \
    "\n" \
    "  void AddCaseNumInfo(uint32_t num) {\n" \
    "    if (!duration_open_now_) {\n" \
    "      return;\n" \
    "    }\n" \
    "    case_num_ += num;\n" \
    "  }\n" \
    "\n" \
    "  void Begin(const DurationType type) {\n" \
    "    if (!duration_open_now_) {\n" \
    "      return;\n" \
    "    }\n" \
    "    const auto &stat = duration_infos_[type].stat;\n" \
    "    if (stat == nullptr) {\n" \
    "      return;\n" \
    "    }\n" \
    "    stat->Begin();\n" \
    "  }\n" \
    "\n" \
    "  void End(const DurationType type) {\n" \
    "    if (!duration_open_now_) {\n" \
    "      return;\n" \
    "    }\n" \
    "    const auto &stat = duration_infos_[type].stat;\n" \
    "    if (stat == nullptr) {\n" \
    "      return;\n" \
    "    }\n" \
    "    stat->End();\n" \
    "  }\n" \
    "  void Print() {\n" \
    "    if (!duration_open_now_) {\n" \
    "      return;\n" \
    "    }\n" \
    "    for (int32_t index = 0; index < static_cast<int32_t>(DurationType::TILING_FUNC_DURATION_MAX); index++) {\n" \
    "      const auto &stat = duration_infos_[index].stat;\n" \
    "      if (stat != nullptr) {\n" \
    "        stat->Print();\n" \
    "      }\n" \
    "    }\n" \
    "    OP_EVENT(OP_NAME, \"Case num is %u.\", case_num_);\n" \
    "    for (const auto& info : iter_infos_) {\n" \
    "      OP_EVENT(OP_NAME, \"%s\'s iter is %u.\", info.case_id.data(), info.iter_count);\n" \
    "    }\n" \
    "  }\n" \
    "  void Clear() {\n" \
    "    if (!duration_open_now_) {\n" \
    "      return;\n" \
    "    }\n" \
    "    for (int32_t index = 0; index < static_cast<int32_t>(DurationType::TILING_FUNC_DURATION_MAX); index++) {\n" \
    "      const auto &stat = duration_infos_[index].stat;\n" \
    "      if (stat != nullptr) {\n" \
    "        stat->Clear();\n" \
    "      }\n" \
    "    }\n" \
    "    iter_infos_.clear();\n" \
    "    case_num_ = 0;\n" \
    "  }\n" \
    "private:\n";

    code += "  bool duration_open_now_ = true;\n";
    code +=
    "  DurationInfo duration_infos_[TILING_FUNC_DURATION_MAX];\n" \
    "  std::vector<IterInfo> iter_infos_;\n" \
    "  uint32_t case_num_{0};\n" \
    "};\n" \
    "\n" \
    "static inline void DurationBegin(const DurationType type) {\n" \
    "  DurationManager::GetInstance().Begin(type);\n" \
    "}\n" \
    "\n" \
    "static inline void DurationEnd(const DurationType type) {\n" \
    "  DurationManager::GetInstance().End(type);\n" \
    "}\n" \
    "\n" \
    "static inline void SaveIterInfo(const char* case_id, uint32_t iter_count) {\n" \
    "  DurationManager::GetInstance().AddIterInfo(case_id, iter_count);\n" \
    "}\n" \
    "static inline void SaveCaseNumInfo(uint32_t num) {\n" \
    "  DurationManager::GetInstance().AddCaseNumInfo(num);\n" \
    "}\n" \
    "\n" \
    "class DurationGuard {\n" \
    "public:\n" \
    "  DurationGuard(const DurationType type) : type_(type)\n" \
    "  {\n" \
    "    DurationBegin(type);\n" \
    "  }\n" \
    "\n" \
    "  ~DurationGuard() {\n" \
    "    DurationEnd(type_);\n" \
    "  }\n" \
    "private:\n" \
    "  DurationType type_;\n" \
    "};\n" \
    "\n" \
    "#define DURATION_GUARD(type) DurationGuard g_duration##__COUNTER__(type);\n" \
    "} // namespace\n";
  return code;
}

std::string DurationPrintGenCode() {
  if (kg_duration_level == 0U) {
    return "";
  }
  std::string code = "DurationManager::GetInstance().Print();";
  return code;
}

std::string DurationClearGenCode() {
  if (kg_duration_level == 0U) {
    return "";
  }
  std::string code = "DurationManager::GetInstance().Clear();";
  return code;
}

std::string DurationBeginGenCode(const TilingFuncDurationType type) {
  if (kg_duration_level <= kg_tiling_func_duration_def[static_cast<int32_t>(type)].level) {
    return "";
  }
  return std::string("DurationBegin(") + kg_tiling_func_duration_def[static_cast<int32_t>(
    type)].name + ");";
}

std::string DurationEndGenCode(const TilingFuncDurationType type) {
  if (kg_duration_level <= kg_tiling_func_duration_def[static_cast<int32_t>(type)].level) {
    return "";
  }
  return std::string("DurationEnd(") + kg_tiling_func_duration_def[static_cast<int32_t>(
    type)].name + ");";
}

std::string DurationGuardGenCode(const TilingFuncDurationType type) {
  if (kg_duration_level <= kg_tiling_func_duration_def[static_cast<int32_t>(type)].level) {
    return "";
  }
  return std::string("DURATION_GUARD(") +
    kg_tiling_func_duration_def[static_cast<int32_t>(type)].name + ")";
}
} //namespace att
