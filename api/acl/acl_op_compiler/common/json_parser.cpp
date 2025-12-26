/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "json_parser.h"

#include <fstream>
#include <sstream>
#include <regex>
#include <sys/stat.h>
#include "mmpa/mmpa_api.h"

namespace {
const std::string ACL_JSON_DEFAULT_DEVICE = "defaultDevice";
const std::string ACL_JSON_DEFAULT_DEVICE_ID = "default_device";
constexpr int32_t DECIMAL = 10;

void CountDepth(const char_t ch, size_t &objDepth, size_t &maxObjDepth, size_t &arrayDepth, size_t &maxArrayDepth)
{
    switch (ch) {
        case '{': {
            ++objDepth;
            if (objDepth > maxObjDepth) {
                maxObjDepth = objDepth;
            }
            break;
        }
        case '}': {
            if (objDepth > 0) {
                --objDepth;
            }
            break;
        }
        case '[': {
            ++arrayDepth;
            if (arrayDepth > maxArrayDepth) {
                maxArrayDepth = arrayDepth;
            }
            break;
        }
        case ']': {
            if (arrayDepth > 0) {
                --arrayDepth;
            }
            break;
        }
        default: {
            return;
        }
    }
}
} // namespace
namespace acl {
    // 配置文件最大字节数目10MBytes
    constexpr int64_t MAX_CONFIG_FILE_BYTE = 10 * 1024 * 1024;
    // 配置文件最大递归深度
    constexpr size_t MAX_CONFIG_OBJ_DEPTH = 10U;
    // 配置文件最大数组个数
    constexpr size_t MAX_CONFIG_ARRAY_DEPTH = 10U;

    void JsonParser::GetMaxNestedLayers(const char_t *const fileName, const size_t length,
        size_t &maxObjDepth, size_t &maxArrayDepth)
    {
        if (length <= 0) {
            ACL_LOG_INNER_ERROR("[Check][Length]the length of file %s must be larger than 0.", fileName);
            return;
        }

        char_t *pBuffer = new(std::nothrow) char_t[length];
        ACL_REQUIRES_NOT_NULL_RET_VOID(pBuffer);
        const std::shared_ptr<char_t> buffer(pBuffer, [](char_t *const deletePtr) { delete[] deletePtr; });

        std::ifstream fin(fileName);
        if (!fin.is_open()) {
            ACL_LOG_INNER_ERROR("[Open][File]read file %s failed.", fileName);
            return;
        }
        (void)fin.seekg(0, fin.beg);
        (void)fin.read(buffer.get(), static_cast<int64_t>(length));

        size_t arrayDepth = 0U;
        size_t objDepth = 0U;
        for (size_t i = 0U; i < length; ++i) {
            const char_t v = buffer.get()[i];
            if (v == '\0') {
                fin.close();
                return;
            }
            CountDepth(v, objDepth, maxObjDepth, arrayDepth, maxArrayDepth);
        }
        fin.close();
    }

    bool JsonParser::IsValidFileName(const char_t *const fileName)
    {
        char_t trustedPath[MMPA_MAX_PATH] = {};
        int32_t ret = mmRealPath(fileName, trustedPath, MMPA_MAX_PATH);
        if (ret != EN_OK) {
            const auto formatErrMsg = acl::AclGetErrorFormatMessage(mmGetErrorCode());
            ACL_LOG_INNER_ERROR("[Trans][RealPath]the file path %s is not like a real path, mmRealPath return %d, "
                "errMessage is %s", fileName, ret, formatErrMsg.c_str());
            return false;
        }

        mmStat_t pathStat;
        ret = mmStatGet(trustedPath, &pathStat);
        if (ret != EN_OK) {
            ACL_LOG_INNER_ERROR("[Get][FileStatus]cannot get config file status, which path is %s, "
                "maybe does not exist, return %d, errcode %d", trustedPath, ret, mmGetErrorCode());
            return false;
        }
        if ((pathStat.st_mode & S_IFMT) != S_IFREG) {
            ACL_LOG_INNER_ERROR("[Config][ConfigFile]config file is not a common file, which path is %s, "
                "mode is %u", trustedPath, pathStat.st_mode);
            return false;
        }
        if (pathStat.st_size > MAX_CONFIG_FILE_BYTE) {
            ACL_LOG_INNER_ERROR("[Check][FileSize]config file %s size[%ld] is larger than "
                "max config file Bytes[%ld]", trustedPath, pathStat.st_size, MAX_CONFIG_FILE_BYTE);
            return false;
        }
        return true;
    }

    bool JsonParser::ParseJson(const char_t *const fileName, nlohmann::json &js, const size_t fileLength)
    {
        std::ifstream fin(fileName);
        if (!fin.is_open()) {
            ACL_LOG_INNER_ERROR("[Read][File]read file %s failed.", fileName);
            return false;
        }

        // checking the depth of file
        size_t maxObjDepth = 0U;
        size_t maxArrayDepth = 0U;
        GetMaxNestedLayers(fileName, fileLength, maxObjDepth, maxArrayDepth);
        if ((maxObjDepth > MAX_CONFIG_OBJ_DEPTH) || (maxArrayDepth > MAX_CONFIG_ARRAY_DEPTH)) {
            ACL_LOG_INNER_ERROR("[Check][MaxArrayDepth]invalid json file, the object's depth[%zu] is larger than %zu, "
                "or the array's depth[%zu] is larger than %zu.",
                maxObjDepth, MAX_CONFIG_OBJ_DEPTH, maxArrayDepth, MAX_CONFIG_ARRAY_DEPTH);
            fin.close();
            return false;
        }
        ACL_LOG_DEBUG("json file's obj's depth is %zu, array's depth is %zu", maxObjDepth, maxArrayDepth);

        try {
            fin >> js;
        } catch (const nlohmann::json::exception &e) {
            ACL_LOG_INNER_ERROR("[Check][JsonFile]invalid json file, exception:%s.", e.what());
            fin.close();
            return false;
        }

        fin.close();
        return true;
    }

    aclError JsonParser::ParseJsonFromFile(const char_t *const fileName, nlohmann::json &js)
    {
        if (fileName == nullptr) {
            ACL_LOG_DEBUG("filename is nullptr, no need to parse json");
            return ACL_SUCCESS;
        }
        ACL_LOG_DEBUG("before ParseJsonFromFile in ParseJsonFromFile");
        if (!IsValidFileName(fileName)) {
            ACL_LOG_INNER_ERROR("[Check][File]invalid config file[%s]", fileName);
            return ACL_ERROR_INVALID_FILE;
        }
        std::ifstream fin(fileName);
        if (!fin.is_open()) {
            ACL_LOG_INNER_ERROR("[Read][File]read file %s failed cause it cannnot be read.", fileName);
            return ACL_ERROR_INVALID_FILE;
        }
        (void)fin.seekg(0, std::ios::end);
        const std::streampos fp = fin.tellg();
        if (static_cast<int32_t>(fp) == 0) {
            ACL_LOG_DEBUG("parse file is null");
            fin.close();
            return ACL_SUCCESS;
        }
        fin.close();
        if (!ParseJson(fileName, js, static_cast<size_t>(fp))) {
            ACL_LOG_INNER_ERROR("[Parse][File]parse config file[%s] to json failed.", fileName);
            return ACL_ERROR_PARSE_FILE;
        }

        ACL_LOG_DEBUG("parse json from file[%s] successfully.", fileName);
        return ACL_SUCCESS;
    }

    aclError JsonParser::GetJsonCtxByKey(const char_t *const fileName,
        std::string &strJsonCtx, const std::string &subStrKey, bool &found) {
        found = false;
        nlohmann::json js;
        aclError ret = acl::JsonParser::ParseJsonFromFile(fileName, js);
        if (ret != ACL_SUCCESS) {
            ACL_LOG_INNER_ERROR("parse json from file falied, errorCode = %d", ret);
            return ret;
        }
        const auto configIter = js.find(subStrKey);
        if (configIter != js.end()) {
            strJsonCtx = configIter->dump();
            found = true;
        }
        return ACL_SUCCESS;
    }
} // namespace acl
