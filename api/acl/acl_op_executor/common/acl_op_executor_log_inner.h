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
constexpr const char_t *const ACL_MODULE_NAME = "ASCENDCL";
constexpr const char_t *const INVALID_PARAM_MSG = "EH0001";
constexpr const char_t *const INVALID_NULL_POINTER_MSG = "EH0002";
constexpr const char_t *const INVALID_PATH_MSG = "EH0003";
constexpr const char_t *const INVALID_FILE_MSG = "EH0004";
constexpr const char_t *const INVALID_AIPP_MSG = "EH0005";
constexpr const char_t *const UNSUPPORTED_FEATURE_MSG = "EH0006";

// first stage
constexpr const char_t *const ACL_STAGE_SET = "SET";
constexpr const char_t *const ACL_STAGE_GET = "GET";
constexpr const char_t *const ACL_STAGE_CREATE = "CREATE";
constexpr const char_t *const ACL_STAGE_DESTROY = "DESTROY";
constexpr const char_t *const ACL_STAGE_INFER = "INFER";
constexpr const char_t *const ACL_STAGE_COMP = "COMP";
constexpr const char_t *const ACL_STAGE_LOAD = "LOAD";
constexpr const char_t *const ACL_STAGE_UNLOAD = "UNLOAD";
constexpr const char_t *const ACL_STAGE_EXEC = "EXEC";

// second stage
constexpr const char_t *const ACL_STAGE_DEFAULT = "DEFAULT";

constexpr size_t MAX_LOG_STRING = 1024U;

constexpr size_t MAX_ERROR_STRING = 128U;

inline const char_t* format_cast(const char_t *const src)
{
    return static_cast<const char_t *>(src);
}

std::string AclGetErrorFormatMessage(const mmErrorMsg errnum);

class AclLog {
public:
    static bool IsLogOutputEnable(const aclLogLevel logLevel);
    static mmPid_t GetTid();
    static void ACLSaveLog(const aclLogLevel logLevel, const char_t *const strLog);
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

#define ACL_REQUIRES_OK(expr) \
    do { \
        const aclError __ret = (expr); \
        if (__ret != ACL_SUCCESS) { \
            return __ret; \
        } \
    } \
    while (false)

#define ACL_REQUIRES_OK_WITH_INNER_MESSAGE(expr, ...) \
    do { \
        const aclError __ret = (expr); \
        if (__ret != ACL_SUCCESS) { \
            ACL_LOG_INNER_ERROR(__VA_ARGS__); \
            return __ret; \
        } \
    } \
    while (false)

// Validate whether the expr value is true
#define ACL_REQUIRES_TRUE(expr, errCode, errDesc) \
    do { \
        const bool __ret = (expr); \
        if (!__ret) { \
            ACL_LOG_ERROR(errDesc); \
            return (errCode); \
        } \
    } \
    while (false)

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

#define ACL_REQUIRES_NOT_NULL_RET_NULL_INPUT_REPORT(val) \
    do { \
        if (UNLIKELY((val) == nullptr)) { \
            ACL_LOG_ERROR("[Check][%s]param must not be null.", #val); \
            acl::AclErrorLogManager::ReportInputError("EH0002", {"param"}, {#val}); \
            return nullptr; } \
        } \
    while (false)

#define ACL_REQUIRES_NOT_NULL_RET_VOID(val) \
    do { \
        if (UNLIKELY((val) == nullptr)) { \
            ACL_LOG_ERROR("[Check][%s]param must not be null.", #val); \
            return; } \
        } \
    while (false)

#define ACL_CHECK_MALLOC_RESULT(val) \
    do { \
        if ((val) == nullptr) { \
            ACL_LOG_INNER_ERROR("[Check][Malloc]Allocate memory for [%s] failed.", #val); \
            return ACL_ERROR_BAD_ALLOC; } \
        } \
    while (false)

#define ACL_REQUIRES_NON_NEGATIVE(val) \
    do { \
        if ((val) < 0) { \
            ACL_LOG_ERROR("[Check][%s]param must be non-negative.", #val); \
            return ACL_ERROR_INVALID_PARAM; } \
        } \
    while (false)

#define ACL_REQUIRES_NON_NEGATIVE_WITH_INPUT_REPORT(val) \
    do { \
        if ((val) < 0) { \
            ACL_LOG_ERROR("[Check][%s]param must be non-negative.", #val); \
            acl::AclErrorLogManager::ReportInputError("EH0001", \
                std::vector<const char *>({"param", "value", "reason"}), \
                std::vector<const char *>({#val, std::to_string(val).c_str(), "must be non-negative"})); \
            return ACL_ERROR_INVALID_PARAM; } \
        } \
    while (false)

#define ACL_REQUIRES_POSITIVE(val) \
    do { \
        if ((val) <= 0) { \
            ACL_LOG_ERROR("[Check][%s]param must be positive.", #val); \
            return ACL_ERROR_INVALID_PARAM; } \
        } \
    while (false)

#define ACL_CHECK_WITH_MESSAGE_AND_RETURN(exp, ret, ...) \
    do { \
        if (!(exp)) { \
            ACL_LOG_ERROR(__VA_ARGS__); \
            return (ret); \
        } \
    } \
    while (false)

#define ACL_CHECK_WITH_INNER_MESSAGE_AND_RETURN(exp, ret, ...) \
    do { \
        if (!(exp)) { \
            ACL_LOG_INNER_ERROR(__VA_ARGS__); \
            return (ret); \
        } \
    } \
    while (false)

#define ACL_DELETE_AND_SET_NULL(var) \
    do { \
        if ((var) != nullptr) { \
            delete (var); \
            (var) = nullptr; \
        } \
    } \
    while (false)

#define ACL_DELETE_ARRAY_AND_SET_NULL(var) \
    do { \
        if ((var) != nullptr) { \
            delete[] (var); \
            (var) = nullptr; \
        } \
    } \
    while (false)

// If make_shared is abnormal, print the log and execute the statement
#define ACL_MAKE_SHARED(expr0, expr1) \
    try { \
        (expr0); \
    } catch (const std::bad_alloc &) { \
        ACL_LOG_INNER_ERROR("[Make][Shared]Make shared failed"); \
        expr1; \
    }

#define ACL_CHECK_INT32_EQUAL(leftValue, rightValue) \
    do { \
        if ((leftValue) != (rightValue)) { \
            ACL_LOG_INFO("[%d] is not equal to [%d].", (leftValue), (rightValue)); \
            return false; \
        } \
    } \
    while (false)

#endif // ACL_COMMON_LOG_H_
