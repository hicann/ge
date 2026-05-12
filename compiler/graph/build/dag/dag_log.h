/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GE_GRAPH_BUILD_DAG_MINIDAG_DAG_LOG_H_
#define GE_GRAPH_BUILD_DAG_MINIDAG_DAG_LOG_H_

#include <cstdarg>
#include <cstdio>
#include <string>

namespace minidag {
enum class DagLogLevel {
  DEBUG = 0,
  INFO = 1,
  WARN = 2,
  ERROR = 3
};

class DagLogger {
 public:
  static DagLogger& GetInstance();
  void Log(DagLogLevel level, const char* file, int line,
           const char* func, const char* fmt, ...);
  bool IsEnabled(DagLogLevel level) const;
  void Reset();

 private:
  DagLogger();
  ~DagLogger();
  DagLogger(const DagLogger&) = delete;
  DagLogger& operator=(const DagLogger&) = delete;
  DagLogLevel level_ = DagLogLevel::INFO;
  FILE* output_ = nullptr;
  bool is_file_ = false;
  void ReadLogLevelFromEnv();
  void ReadLogFileFromEnv();
};
}  // namespace minidag

#define MINIDAG_LOG_DEBUG(fmt, ...)                                           \
  do {                                                                        \
    auto& logger = minidag::DagLogger::GetInstance();                         \
    if (logger.IsEnabled(minidag::DagLogLevel::DEBUG)) {                      \
      logger.Log(minidag::DagLogLevel::DEBUG, __FILE__, __LINE__,             \
                 __FUNCTION__, fmt, ##__VA_ARGS__);                           \
    }                                                                         \
  } while (0)

#define MINIDAG_LOG_INFO(fmt, ...)                                            \
  do {                                                                        \
    auto& logger = minidag::DagLogger::GetInstance();                         \
    if (logger.IsEnabled(minidag::DagLogLevel::INFO)) {                       \
      logger.Log(minidag::DagLogLevel::INFO, __FILE__, __LINE__,              \
                 __FUNCTION__, fmt, ##__VA_ARGS__);                           \
    }                                                                         \
  } while (0)

#define MINIDAG_LOG_WARN(fmt, ...)                                            \
  do {                                                                        \
    auto& logger = minidag::DagLogger::GetInstance();                         \
    if (logger.IsEnabled(minidag::DagLogLevel::WARN)) {                      \
      logger.Log(minidag::DagLogLevel::WARN, __FILE__, __LINE__,              \
                 __FUNCTION__, fmt, ##__VA_ARGS__);                           \
    }                                                                         \
  } while (0)

#define MINIDAG_LOG_ERROR(fmt, ...)                                           \
  do {                                                                        \
    auto& logger = minidag::DagLogger::GetInstance();                         \
    logger.Log(minidag::DagLogLevel::ERROR, __FILE__, __LINE__,               \
               __FUNCTION__, fmt, ##__VA_ARGS__);                             \
  } while (0)
#endif  // GE_GRAPH_BUILD_DAG_MINIDAG_DAG_LOG_H_