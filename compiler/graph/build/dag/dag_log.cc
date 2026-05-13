/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph/build/dag/dag_log.h"

#include <cstdlib>
#include <cstring>

namespace minidag {
DagLogger& DagLogger::GetInstance() {
  static DagLogger instance;
  return instance;
}

DagLogger::DagLogger() {
  ReadLogLevelFromEnv();
  ReadLogFileFromEnv();
}

DagLogger::~DagLogger() {
  if (is_file_ && output_ != nullptr) {
    fclose(output_);
    output_ = nullptr;
  }
}

void DagLogger::ReadLogLevelFromEnv() {
  const char* level_str = getenv("MINIDAG_LOG_LEVEL");
  if (level_str != nullptr) {
    int level = atoi(level_str);
    if (level >= 0 && level <= static_cast<int>(DagLogLevel::ERROR)) {
      level_ = static_cast<DagLogLevel>(level);
    }
  }
}

void DagLogger::ReadLogFileFromEnv() {
  const char* file_path = getenv("MINIDAG_LOG_FILE");
  if (file_path != nullptr && strlen(file_path) > 0) {
    output_ = fopen(file_path, "a");
    if (output_ != nullptr) {
      is_file_ = true;
    } else {
      // 文件打开失败，fallback 到 stdout
      output_ = stdout;
      is_file_ = false;
    }
  } else {
    // 默认 stdout
    output_ = stdout;
    is_file_ = false;
  }
}

bool DagLogger::IsEnabled(DagLogLevel level) const {
  return static_cast<int>(level) >= static_cast<int>(level_);
}

void DagLogger::Reset() {
  // 关闭之前的文件输出
  if (is_file_ && output_ != nullptr) {
    fclose(output_);
    output_ = nullptr;
    is_file_ = false;
  }
  // 重新读取环境变量配置
  ReadLogLevelFromEnv();
  ReadLogFileFromEnv();
}

void DagLogger::Log(DagLogLevel level, const char* file, int line,
                    const char* func, const char* fmt, ...) {
  if (output_ == nullptr) {
    return;
  }

  const char* level_str[] = {"DEBUG", "INFO", "WARN", "ERROR"};

  va_list args;
  va_start(args, fmt);

  fprintf(output_, "[MINIDAG][%s][%s:%d][%s] ",
          level_str[static_cast<int>(level)], file, line, func);
  vfprintf(output_, fmt, args);
  fprintf(output_, "\n");

  va_end(args);

  if (is_file_) {
    fflush(output_);
  }
}
}  // namespace minidag