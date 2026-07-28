/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef RUNTIME_OM2_OM2_AIPP_UTILS_H_
#define RUNTIME_OM2_OM2_AIPP_UTILS_H_

#include <string>
#include <vector>
#include "common/dynamic_aipp.h"
#include "common/ge_common/ge_types.h"
#include "common/helper/om2/json_file.h"
#include "common/om2/om2_model_data.h"
#include "graph/types.h"

namespace gert {
namespace om2 {

// AIPP dim info 解析常量
constexpr size_t kAippDimPartsNum = 6U;
constexpr size_t kAippDimNameIdx = 2U;
constexpr size_t kAippDimSizeIdx = 3U;
constexpr size_t kAippDimDimNumIdx = 4U;
constexpr size_t kAippDimShapeIdx = 5U;
constexpr int32_t kAippDecimalRadix = 10;

// 将 "NCHW:DT_FLOAT:data:0:4:1,3,224,224" 格式的字符串解析为 InputOutputDims
ge::Status ParseAippDimInfo(const std::string &info_str, ge::InputOutputDims &dims_info);

// 从 JSON 解析 AippConfigInfo（打包侧保证所有字段存在）
ge::AippConfigInfo ParseAippConfigFromJson(const ge::JsonFile &entry);

// 从 JSON 解析 OriginInputInfo
ge::OriginInputInfo ParseOriginInputFromJson(const ge::JsonFile &entry);

// 从 JSON 字符串数组解析 InputOutputDims 列表
std::vector<ge::InputOutputDims> ParseAippDimsFromJson(const ge::JsonFile &entry, const char *key);

// 从 model_meta.json 的 aipp 字段 JSON 对象中解析 AIPP 信息
ge::Status ParseAippJson(const ge::JsonFile &aipp_json, std::vector<Om2AippMeta> &aipp_infos, bool &has_aipp);

}  // namespace om2
}  // namespace gert

#endif  // RUNTIME_OM2_OM2_AIPP_UTILS_H_
