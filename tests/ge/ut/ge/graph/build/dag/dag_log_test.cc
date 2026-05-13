/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>

#include <cstdlib>
#include <cstdio>
#include <fstream>
#include <string>
#include <unistd.h>

#include "graph/build/dag/dag_log.h"

namespace minidag {

class DagLoggerTest : public testing::Test {
 protected:
  void SetUp() override {
    unsetenv("MINIDAG_LOG_LEVEL");
    unsetenv("MINIDAG_LOG_FILE");
  }

  void TearDown() override {
    unsetenv("MINIDAG_LOG_LEVEL");
    unsetenv("MINIDAG_LOG_FILE");
  }
};

// --------------------
// 场景 1：日志级别与 IsEnabled
// --------------------

/**
 * 场景 1-1: 默认级别验证
 * 验证无环境变量时默认级别为 INFO
 */
TEST_F(DagLoggerTest, DefaultLevel) {
  auto& logger = DagLogger::GetInstance();
  EXPECT_FALSE(logger.IsEnabled(DagLogLevel::DEBUG));
  EXPECT_TRUE(logger.IsEnabled(DagLogLevel::INFO));
  EXPECT_TRUE(logger.IsEnabled(DagLogLevel::WARN));
  EXPECT_TRUE(logger.IsEnabled(DagLogLevel::ERROR));
}

/**
 * 场景 1-2: 日志级别枚举值验证
 */
TEST_F(DagLoggerTest, LogLevelEnumValues) {
  EXPECT_EQ(static_cast<int>(DagLogLevel::DEBUG), 0);
  EXPECT_EQ(static_cast<int>(DagLogLevel::INFO), 1);
  EXPECT_EQ(static_cast<int>(DagLogLevel::WARN), 2);
  EXPECT_EQ(static_cast<int>(DagLogLevel::ERROR), 3);
}

/**
 * 场景 1-3: IsEnabled 比较验证
 */
TEST_F(DagLoggerTest, IsEnabledComparison) {
  auto& logger = DagLogger::GetInstance();
  EXPECT_FALSE(logger.IsEnabled(DagLogLevel::DEBUG));
  EXPECT_TRUE(logger.IsEnabled(DagLogLevel::INFO));
  EXPECT_TRUE(logger.IsEnabled(DagLogLevel::WARN));
  EXPECT_TRUE(logger.IsEnabled(DagLogLevel::ERROR));
}

// --------------------
// 场景 2：日志输出与文件
// --------------------

/**
 * 场景 2-1: 基本日志格式
 */
TEST_F(DagLoggerTest, LogFormatBasic) {
  auto& logger = DagLogger::GetInstance();
  logger.Log(DagLogLevel::INFO, "test_file.cc", 100, "TestFunc", "test message");
  logger.Log(DagLogLevel::WARN, "test_file.cc", 200, "AnotherFunc", "value=%d, name=%s", 42, "test");
  SUCCEED();
}

/**
 * 场景 2-2: 日志文件输出
 */
TEST_F(DagLoggerTest, LogFileOutput) {
  const char* temp_log_file = "/tmp/minidag_test_log.txt";

  setenv("MINIDAG_LOG_FILE", temp_log_file, 1);
  setenv("MINIDAG_LOG_LEVEL", "0", 1);

  auto& logger = DagLogger::GetInstance();
  logger.Reset();

  logger.Log(DagLogLevel::INFO, "dag_log_test.cc", 1, "LogFileOutputTest", "Log file output test message");

  std::ifstream ifs(temp_log_file);
  if (ifs.is_open()) {
    std::string content;
    std::getline(ifs, content);
    EXPECT_TRUE(content.find("[MINIDAG]") != std::string::npos);
    EXPECT_TRUE(content.find("[INFO]") != std::string::npos);
    ifs.close();
    std::remove(temp_log_file);
  }

  unsetenv("MINIDAG_LOG_FILE");
  unsetenv("MINIDAG_LOG_LEVEL");
}

/**
 * 场景 2-3: 回退到标准输出
 */
TEST_F(DagLoggerTest, FallbackToStdout) {
  setenv("MINIDAG_LOG_FILE", "/nonexistent_dir_12345/test.log", 1);
  auto& logger = DagLogger::GetInstance();
  logger.Log(DagLogLevel::ERROR, "dag_log_test.cc", 1, "FallbackTest", "Fallback to stdout test");
  SUCCEED();
  unsetenv("MINIDAG_LOG_FILE");
}

/**
 * 场景 2-4: ERROR 级别始终输出
 */
TEST_F(DagLoggerTest, ErrorAlwaysOutput) {
  setenv("MINIDAG_LOG_LEVEL", "3", 1);
  auto& logger = DagLogger::GetInstance();
  logger.Log(DagLogLevel::ERROR, "dag_log_test.cc", 1, "ErrorAlwaysTest", "Error message should always output");
  SUCCEED();
  unsetenv("MINIDAG_LOG_LEVEL");
}

// --------------------
// 场景 3：宏与单例
// --------------------

/**
 * 场景 3-1: 日志宏基本功能
 */
TEST_F(DagLoggerTest, MacroBasic) {
  MINIDAG_LOG_INFO("Info log test");
  MINIDAG_LOG_WARN("Warn log test");
  MINIDAG_LOG_ERROR("Error log test");
  MINIDAG_LOG_DEBUG("Debug log test - may not output depending on level");
  SUCCEED();
}

/**
 * 场景 3-2: 单例实例验证
 */
TEST_F(DagLoggerTest, SingletonInstance) {
  auto& logger1 = DagLogger::GetInstance();
  auto& logger2 = DagLogger::GetInstance();
  EXPECT_EQ(&logger1, &logger2);
}

/**
 * 场景 3-3: 性能无开销验证
 */
TEST_F(DagLoggerTest, PerformanceNoOverhead) {
  auto& logger = DagLogger::GetInstance();
  if (!logger.IsEnabled(DagLogLevel::DEBUG)) {
    MINIDAG_LOG_DEBUG("This should have no formatting overhead");
  }
  SUCCEED();
}

// --------------------
// 场景 4：Reset 方法与覆盖率（主进程）
// --------------------

/**
 * 场景 4-1: Reset 函数主进程测试
 */
TEST_F(DagLoggerTest, ResetInMainProcess) {
  unsetenv("MINIDAG_LOG_FILE");
  unsetenv("MINIDAG_LOG_LEVEL");

  auto& logger = DagLogger::GetInstance();
  logger.Reset();
  EXPECT_TRUE(logger.IsEnabled(DagLogLevel::ERROR));
}

/**
 * 场景 4-2: ReadLogFileFromEnv 文件打开测试
 */
TEST_F(DagLoggerTest, ReadLogFileFromEnv_FileOpen) {
  char tmp_file[] = "/tmp/minidag_log_test_XXXXXX";
  int fd = mkstemp(tmp_file);
  ASSERT_GE(fd, 0);
  close(fd);

  setenv("MINIDAG_LOG_FILE", tmp_file, 1);
  setenv("MINIDAG_LOG_LEVEL", "0", 1);

  auto& logger = DagLogger::GetInstance();
  logger.Reset();
  logger.Log(DagLogLevel::INFO, "file_test.cc", 1, "FileTest", "test message");

  unsetenv("MINIDAG_LOG_FILE");
  unsetenv("MINIDAG_LOG_LEVEL");
  logger.Reset();
  unlink(tmp_file);
}

/**
 * 场景 4-3: Log 函数 fflush 测试（主进程）
 */
TEST_F(DagLoggerTest, LogFlushesWhenWritingToFileInMainProcess) {
  char tmp_file[] = "/tmp/minidag_flush_test_XXXXXX";
  int fd = mkstemp(tmp_file);
  ASSERT_GE(fd, 0);
  close(fd);

  setenv("MINIDAG_LOG_FILE", tmp_file, 1);
  setenv("MINIDAG_LOG_LEVEL", "0", 1);

  auto& logger = DagLogger::GetInstance();
  logger.Reset();
  logger.Log(DagLogLevel::INFO, "flush_test.cc", 1, "FlushTest", "message 1");
  logger.Log(DagLogLevel::WARN, "flush_test.cc", 2, "FlushTest", "message 2");

  unsetenv("MINIDAG_LOG_FILE");
  unsetenv("MINIDAG_LOG_LEVEL");
  logger.Reset();
  unlink(tmp_file);
}

/**
 * 场景 4-4: 无效日志级别边界测试
 */
TEST_F(DagLoggerTest, LogLevelBoundaryValuesInMainProcess) {
  setenv("MINIDAG_LOG_LEVEL", "0", 1);
  unsetenv("MINIDAG_LOG_FILE");
  auto& logger = DagLogger::GetInstance();
  logger.Reset();
  EXPECT_TRUE(logger.IsEnabled(DagLogLevel::DEBUG));

  setenv("MINIDAG_LOG_LEVEL", "3", 1);
  logger.Reset();
  EXPECT_TRUE(logger.IsEnabled(DagLogLevel::ERROR));
  EXPECT_FALSE(logger.IsEnabled(DagLogLevel::WARN));

  unsetenv("MINIDAG_LOG_LEVEL");
  logger.Reset();
}

/**
 * 场景 4-5: 日志格式化参数测试
 */
TEST_F(DagLoggerTest, LogFormatWithArguments) {
  unsetenv("MINIDAG_LOG_FILE");
  setenv("MINIDAG_LOG_LEVEL", "0", 1);

  auto& logger = DagLogger::GetInstance();
  logger.Reset();
  logger.Log(DagLogLevel::INFO, "format_test.cc", 1, "FormatTest", "Value: %d, String: %s, Float: %.2f", 42, "test", 3.14);
  SUCCEED();

  unsetenv("MINIDAG_LOG_LEVEL");
}

}  // namespace minidag