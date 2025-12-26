/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ACL_COMMON_LOG_INNER_H_
#define ACL_COMMON_LOG_INNER_H_

#include <string>
#include <vector>
#include "dlog_pub.h"
#include "mmpa/mmpa_api.h"
#include "acl/acl_base.h"

#ifndef char_t
using char_t = char;
#endif

#ifndef float32_t
using float32_t = float;
#endif

namespace acl {
constexpr int32_t ACL_MODE_ID = static_cast<int32_t>(ASCENDCL);
constexpr const char_t *const INVALID_PARAM_MSG = "EH0001";
constexpr const char_t *const INVALID_NULL_POINTER_MSG = "EH0002";
constexpr const char_t *const INVALID_PATH_MSG = "EH0003";
constexpr const char_t *const INVALID_FILE_MSG = "EH0004";
constexpr const char_t *const INVALID_AIPP_MSG = "EH0005";
constexpr const char_t *const UNSUPPORTED_FEATURE_MSG = "EH0006";

// first stage
constexpr const char_t *const ACL_STAGE_CREATE = "CREATE";
constexpr const char_t *const ACL_STAGE_DESTROY = "DESTROY";
constexpr const char_t *const ACL_STAGE_BLAS = "BLAS";

// second stage
constexpr const char_t *const ACL_STAGE_DEFAULT = "DEFAULT";

constexpr size_t MAX_LOG_STRING = 1024U;

constexpr size_t MAX_ERROR_STRING = 128U;

inline const char_t* format_cast(const char_t *const src)
{
    return static_cast<const char_t *>(src);
}

class AclLog {
public:
    static bool IsLogOutputEnable(const aclLogLevel logLevel);
    static mmPid_t GetTid();
    static bool IsEventLogOutputEnable();
private:
    static aclLogLevel GetCurLogLevel();
    static bool isEnableEvent_;
};

class AclErrorLogManager {
public:
    static std::string FormatStr(const char_t *const fmt, ...);
#if !defined(ENABLE_DVPP_INTERFACE) || defined(RUN_TEST)
    static void ReportInputError(const char *errorCode, const std::vector<const char *> &key = {},
        const std::vector<const char *> &val = {});
#else
#endif
    static void ReportInnerError(const char_t *const fmt, ...);
    static void ReportCallError(const char_t *const fmt, ...);
};
} // namespace acl

#if defined(__GNUC__) || defined(__clang__)
#define LIKELY(expr) (static_cast<bool>(__builtin_expect(static_cast<bool>(expr), 1)))
#define UNLIKELY(expr) (static_cast<bool>(__builtin_expect(static_cast<bool>(expr), 0)))
#else
#define LIKELY(expr) (expr)
#define UNLIKELY(expr) (expr)
#endif

#ifdef RUN_TEST
#define ACL_LOG_INFO(fmt, ...)                                                                      \
    do {                                                                                            \
            if (acl::AclLog::IsLogOutputEnable(ACL_INFO)) {                                         \
                constexpr const char_t *const funcName = __FUNCTION__;                                        \
                printf("INFO %d %s:%s:%d: "#fmt "\n", acl::AclLog::GetTid(), acl::format_cast(funcName), \
                    __FILE__, __LINE__, ##__VA_ARGS__);                                             \
            }                                                                                       \
    } while (false)
#define ACL_LOG_DEBUG(fmt, ...)                                                                     \
    do {                                                                                            \
            if (acl::AclLog::IsLogOutputEnable(ACL_DEBUG)) {                                        \
                constexpr const char_t *const funcName = __FUNCTION__;                                        \
                printf("DEBUG %d %s:%s:%d: "#fmt "\n", acl::AclLog::GetTid(), acl::format_cast(funcName), \
                    __FILE__, __LINE__, ##__VA_ARGS__);                                             \
            }                                                                                       \
    } while (false)
#define ACL_LOG_WARN(fmt, ...)                                                                      \
    do {                                                                                            \
            if (acl::AclLog::IsLogOutputEnable(ACL_WARNING)) {                                      \
                constexpr const char_t *const funcName = __FUNCTION__;                                        \
                printf("WARN %d %s:%s:%d: "#fmt "\n", acl::AclLog::GetTid(), acl::format_cast(funcName), \
                    __FILE__, __LINE__, ##__VA_ARGS__);                                             \
            }                                                                                       \
    } while (false)
#define ACL_LOG_ERROR(fmt, ...)                                                                     \
    do {                                                                                            \
            constexpr const char_t *const funcName = __FUNCTION__;                                            \
            printf("ERROR %d %s:%s:%d:" fmt "\n", acl::AclLog::GetTid(), acl::format_cast(funcName), \
                __FILE__, __LINE__, ##__VA_ARGS__);  \
    } while (false)
#define ACL_LOG_INNER_ERROR(fmt, ...)                                                               \
    do {                                                                                            \
            constexpr const char_t *const funcName = __FUNCTION__;                                            \
            printf("ERROR %d %s:%s:%d:" fmt "\n", acl::AclLog::GetTid(), acl::format_cast(funcName), \
                __FILE__, __LINE__, ##__VA_ARGS__); \
    } while (false)
#define ACL_LOG_CALL_ERROR(fmt, ...)                                                                \
    do {                                                                                            \
            constexpr const char_t *const funcName = __FUNCTION__;                                            \
            printf("ERROR %d %s:%s:%d:" fmt "\n", acl::AclLog::GetTid(), acl::format_cast(funcName), \
                __FILE__, __LINE__, ##__VA_ARGS__); \
    } while (false)
#define ACL_LOG_EVENT(fmt, ...)                                                                     \
    do {                                                                                            \
            if (acl::AclLog::IsEventLogOutputEnable()) {                                            \
                constexpr const char_t *const funcName = __FUNCTION__;                                        \
                printf("EVENT %d %s:%s:%d: "#fmt "\n", acl::AclLog::GetTid(), acl::format_cast(funcName), \
                    __FILE__, __LINE__, ##__VA_ARGS__);                                             \
            }                                                                                       \
    } while (false)
#else
#define ACL_LOG_INFO(fmt, ...)                                                                                        \
    do {                                                                                                              \
        constexpr const char_t *const funcName = __FUNCTION__;                                                        \
        dlog_info(acl::ACL_MODE_ID, "%d %s: " fmt, acl::AclLog::GetTid(), acl::format_cast(funcName), ##__VA_ARGS__); \
    } while (false)
#define ACL_LOG_DEBUG(fmt, ...)                                                                                        \
    do {                                                                                                               \
        constexpr const char_t *const funcName = __FUNCTION__;                                                         \
        dlog_debug(acl::ACL_MODE_ID, "%d %s: " fmt, acl::AclLog::GetTid(), acl::format_cast(funcName), ##__VA_ARGS__); \
    } while (false)
#define ACL_LOG_WARN(fmt, ...)                                                                                        \
    do {                                                                                                              \
        constexpr const char_t *const funcName = __FUNCTION__;                                                        \
        dlog_warn(acl::ACL_MODE_ID, "%d %s: " fmt, acl::AclLog::GetTid(), acl::format_cast(funcName), ##__VA_ARGS__); \
    } while (false)
#define ACL_LOG_ERROR(fmt, ...)                                                                          \
    do {                                                                                                 \
        constexpr const char_t *const funcName = __FUNCTION__;                                           \
        dlog_error(acl::ACL_MODE_ID, "%d %s:" fmt, acl::AclLog::GetTid(), acl::format_cast(funcName), \
            ##__VA_ARGS__);                          \
    } while (false)
#define ACL_LOG_INNER_ERROR(fmt, ...)                                                                    \
    do {                                                                                                 \
        constexpr const char_t *const funcName = __FUNCTION__;                                           \
        dlog_error(acl::ACL_MODE_ID, "%d %s:" fmt, acl::AclLog::GetTid(), acl::format_cast(funcName), \
            ##__VA_ARGS__);                          \
        acl::AclErrorLogManager::ReportInnerError((fmt), ##__VA_ARGS__);                                 \
    } while (false)
#define ACL_LOG_CALL_ERROR(fmt, ...)                                                                     \
    do {                                                                                                 \
        constexpr const char_t *const funcName = __FUNCTION__;                                           \
        dlog_error(acl::ACL_MODE_ID, "%d %s:" fmt, acl::AclLog::GetTid(), acl::format_cast(funcName), \
            ##__VA_ARGS__);                          \
        acl::AclErrorLogManager::ReportCallError((fmt), ##__VA_ARGS__);                                  \
    } while (false)
#define ACL_LOG_EVENT(fmt, ...)                                                                  \
    do {                                                                                         \
        if (acl::AclLog::IsEventLogOutputEnable()) {                                             \
            constexpr const char_t *const funcName = __FUNCTION__;                               \
            dlog_info((acl::ACL_MODE_ID | (RUN_LOG_MASK)), "%d %s: " fmt, acl::AclLog::GetTid(), \
                acl::format_cast(funcName), ##__VA_ARGS__);                                      \
        }                                                                                        \
    } while (false)
#endif

#define ACL_REQUIRES_NOT_NULL_WITH_INPUT_REPORT(val) \
    do { \
    if (UNLIKELY((val) == nullptr)) { \
        ACL_LOG_ERROR("[Check][%s]param must not be null.", #val); \
        acl::AclErrorLogManager::ReportInputError("EH0002", {"param"}, {#val}); \
        return ACL_ERROR_INVALID_PARAM; } \
    } \
    while (false)

#define ACL_REQUIRES_NOT_NULL(val) \
    do { \
        if (UNLIKELY((val) == nullptr)) { \
            ACL_LOG_ERROR("[Check][%s]param must not be null.", #val); \
            return ACL_ERROR_INVALID_PARAM; } \
        } \
    while (false)

#endif // ACL_COMMON_LOG_H_
