/**
* Copyright (c) Huawei Technologies Co., Ltd. 2026 All rights reserved.
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

#include "result_checker_utils.h"
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <regex>
#include <filesystem>

namespace att {
namespace fs = std::filesystem;
std::string ResultCheckerUtils::DefineCheckerFunction() {
  return R"(
#include <sstream>
#include <iostream>
#include <stdio.h>

#define MY_ASSERT_EQ(x, y)                                                                                    \
 do {                                                                                                        \
   const auto &xv = (x);                                                                                     \
   const auto &yv = (y);                                                                                     \
   if (xv != yv) {                                                                                           \
     std::stringstream ss;                                                                                   \
     ss << "Assert (" << #x << " == " << #y << ") failed, expect " << yv << " actual " << xv;                \
     printf("%s\n", ss.str().c_str());                                                             \
     std::exit(1);                                                                                           \
   }                                                                                                         \
 } while (false))";
}

bool ResultCheckerUtils::IsFileContainsString(const std::string &filename, const std::string &search_sub_string) {
  std::ifstream file(filename);
  if (!file.is_open()) {
    std::cerr << "无法打开文件: " << filename << std::endl;
    return false;
  }
  std::string line;
  while (std::getline(file, line)) {
    if (line.find(search_sub_string) != std::string::npos) {
      file.close();
      return true;
    }
  }
  file.close();
  return false;
}
bool ResultCheckerUtils::ReplaceLogMacros(const std::string& filename) {
  // 检查文件是否存在
  if (!fs::exists(filename)) {
    std::cerr << "Error: filename " << filename << " is not exist!" << std::endl;
    return false;
  }

  // 读取文件内容
  std::ifstream in_file(filename);
  if (!in_file.is_open()) {
    std::cerr << "Error: can not open file " << filename << " can not read!" << std::endl;
    return false;
  }

  std::vector<std::string> lines;
  std::string line;
  while (std::getline(in_file, line)) {
    lines.push_back(line);
  }
  in_file.close();

  // 正则表达式模式
  std::regex pattern_LOGD(
      R"(#define\s+OP_LOGD\(name,\s*fmt,\s*\.\.\.\)\s+GELOGD\("\[%s\]"\s*fmt,\s*name,\s*##__VA_ARGS__\))");
  std::regex pattern_LOGI(
      R"(#define\s+OP_LOGI\(name,\s*fmt,\s*\.\.\.\)\s+GELOGI\("\[%s\]"\s*fmt,\s*name,\s*##__VA_ARGS__\))");
  std::regex pattern_LOGW(
      R"(#define\s+OP_LOGW\(name,\s*fmt,\s*\.\.\.\)\s+GELOGW\("\[%s\]"\s*fmt,\s*name,\s*##__VA_ARGS__\))");
  std::regex pattern_LOGE(
      R"(#define\s+OP_LOGE\(name,\s*fmt,\s*\.\.\.\)\s+GELOGE\(-1,\s*"\[%s\]"\s*fmt,\s*name,\s*##__VA_ARGS__\))");

  // 替换字符串
  std::string replacement_LOGD = R"(#define OP_LOGD(name, fmt, ...) printf("\n[DEBUG][%s]" fmt, name, ##__VA_ARGS__))";
  std::string replacement_LOGI = R"(#define OP_LOGI(name, fmt, ...) printf("\n[INFO][%s]" fmt, name, ##__VA_ARGS__))";
  std::string replacement_LOGW = R"(#define OP_LOGW(name, fmt, ...) printf("\n[WARNING][%s]" fmt, name, ##__VA_ARGS__))";
  std::string replacement_LOGE = R"(#define OP_LOGE(name, fmt, ...) printf("\n[ERROR][%s]" fmt, name, ##__VA_ARGS__))";

  bool modified = false;

  // 处理每一行
  for (auto& current_line : lines) {
    std::string original_line = current_line;
    if (std::regex_search(current_line, pattern_LOGD)) {
      current_line = std::regex_replace(current_line, pattern_LOGD, replacement_LOGD);
      std::cout << "Replace OP_LOGD: " << original_line << " -> " << current_line << std::endl;
      modified = true;
    }
    else if (std::regex_search(current_line, pattern_LOGI)) {
      current_line = std::regex_replace(current_line, pattern_LOGI, replacement_LOGI);
      std::cout << "Replace OP_LOGI: " << original_line << " -> " << current_line << std::endl;
      modified = true;
    }
    else if (std::regex_search(current_line, pattern_LOGW)) {
      current_line = std::regex_replace(current_line, pattern_LOGW, replacement_LOGW);
      std::cout << "Replace OP_LOGW: " << original_line << " -> " << current_line << std::endl;
      modified = true;
    }
    else if (std::regex_search(current_line, pattern_LOGE)) {
      current_line = std::regex_replace(current_line, pattern_LOGE, replacement_LOGE);
      std::cout << "Replace OP_LOGE: " << original_line << " -> " << current_line << std::endl;
      modified = true;
    }
  }

  // 如果没有修改，直接返回
  if (!modified) {
    std::cout << "Can not find log macros in " << filename << std::endl;
    return true;
  }

  // 创建备份文件
  std::string backup_filename = filename + ".bak";
  if (fs::exists(backup_filename)) {
    fs::remove(backup_filename);
  }
  fs::copy(filename, backup_filename);
  std::cout << "Has backup filename: " << backup_filename << std::endl;

  // 写回修改后的内容
  std::ofstream outFile(filename);
  if (!outFile.is_open()) {
    std::cerr << "Error: can not open " << filename << " , can not write!" << std::endl;
    return false;
  }

  for (const auto& current_line : lines) {
    outFile << current_line << std::endl;
  }
  outFile.close();

  std::cout << "Replace log micros in " << filename << " successfully!" << std::endl;
  return true;
}

std::string ResultCheckerUtils::GetDependAscendIncPath() {
  std::string depend_ascend_inc_path;
#ifdef ASCEND_INSTALL_PATH
  depend_ascend_inc_path.append(" -I ")
      .append(ASCEND_INSTALL_PATH)
      .append("/pkg_inc/base/")
      .append(" -I ")
      .append(ASCEND_INSTALL_PATH)
      .append("/x86_64-linux/include")
      .append(" -I ")
      .append(ASCEND_INSTALL_PATH)
      .append("/aarch64-linux/include");
#endif
  return depend_ascend_inc_path;
}
}