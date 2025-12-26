/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cinttypes>
#include <sys/syscall.h>
#include <unistd.h>
#include <cstdint>
#include <stdio.h>
#include <stdlib.h>
#include <map>
#include "dlog_pub.h"

#define GE_MODULE_NAME static_cast<int32_t>(45)
inline bool IsLogPrintStdout() {
 static int32_t stdout_flag = -1;
 if (stdout_flag == -1) {
   const char *env_ret = getenv("ASCEND_SLOG_PRINT_TO_STDOUT");
   const bool print_stdout = ((env_ret != nullptr) && (strcmp(env_ret, "1") == 0));
   stdout_flag = print_stdout ? 1 : 0;
 }
 return (stdout_flag == 1) ? true : false;
}

inline uint64_t GetTid() {
   return static_cast<uint64_t>(syscall(__NR_gettid));
}

#define GELOGE(ERROR_CODE, fmt, ...)                                                                               \
  do {                                                                                                             \
    dlog_error(GE_MODULE_NAME, "%" PRIu64 " %s: ErrorNo: %" PRIuLEAST8 "(%s) %s" fmt, GetTid(), &__FUNCTION__[0U], \
               (ERROR_CODE), "", "", ##__VA_ARGS__);                                                               \
  } while (false)

#define GELOGW(fmt, ...)                                                                          \
  do {                                                                                            \
    dlog_warn(GE_MODULE_NAME, "%" PRIu64 " %s:" fmt, GetTid(), &__FUNCTION__[0U], ##__VA_ARGS__); \
  } while (false)

#define GELOGI(fmt, ...)                                                                          \
  do {                                                                                            \
    dlog_info(GE_MODULE_NAME, "%" PRIu64 " %s:" fmt, GetTid(), &__FUNCTION__[0U], ##__VA_ARGS__); \
  } while (false)

#define GELOGD(fmt, ...)                                                                           \
  do {                                                                                             \
    dlog_debug(GE_MODULE_NAME, "%" PRIu64 " %s:" fmt, GetTid(), &__FUNCTION__[0U], ##__VA_ARGS__); \
  } while (false)

#define GEEVENT(fmt, ...)                                                                                        \
  do {                                                                                                           \
    dlog_info(static_cast<int32_t>(static_cast<uint32_t>(RUN_LOG_MASK) | static_cast<uint32_t>(GE_MODULE_NAME)), \
              "%" PRIu64 " %s:" fmt, GetTid(), &__FUNCTION__[0U], ##__VA_ARGS__);                                \
    if (!IsLogPrintStdout()) {                                                                                   \
      dlog_info(GE_MODULE_NAME, "%" PRIu64 " %s:" fmt, GetTid(), &__FUNCTION__[0U], ##__VA_ARGS__);              \
    }                                                                                                            \
  } while (false)
