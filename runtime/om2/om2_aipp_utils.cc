/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "om2_aipp_utils.h"
#include <cstdlib>
#include "common/ge_common/string_util.h"
#include "framework/common/debug/log.h"
#include "nlohmann/json.hpp"

namespace gert {
namespace om2 {

// 将 "NCHW:DT_FLOAT:data:0:4:1,3,224,224" 格式的字符串解析为 InputOutputDims
ge::Status ParseAippDimInfo(const std::string &info_str, ge::InputOutputDims &dims_info) {
  const auto parts = ge::StringUtils::Split(info_str, ':');
  if (parts.size() != kAippDimPartsNum) {
    GELOGW("[OM2][AIPP] Invalid aipp dim info: %s, parts=%zu", info_str.c_str(), parts.size());
    return ge::FAILED;
  }
  dims_info.name = parts[kAippDimNameIdx];
  dims_info.size = static_cast<uint32_t>(std::strtol(parts[kAippDimSizeIdx].c_str(), nullptr, kAippDecimalRadix));
  dims_info.dim_num = static_cast<size_t>(std::strtol(parts[kAippDimDimNumIdx].c_str(), nullptr, kAippDecimalRadix));

  const auto dim_strs = ge::StringUtils::Split(parts[kAippDimShapeIdx], ',');
  for (const auto &dim_str : dim_strs) {
    if (dim_str.empty()) {
      continue;
    }
    dims_info.dims.emplace_back(std::strtol(dim_str.c_str(), nullptr, kAippDecimalRadix));
  }
  return ge::SUCCESS;
}

// 从 JsonFile 解析 AippConfigInfo（打包侧保证所有字段存在）
ge::AippConfigInfo ParseAippConfigFromJson(const ge::JsonFile &entry) {
  ge::AippConfigInfo info = {};

  entry.Get("aipp_mode", info.aipp_mode);
  entry.Get("input_format", info.input_format);
  entry.Get("src_image_size_w", info.src_image_size_w);
  entry.Get("src_image_size_h", info.src_image_size_h);
  entry.Get("crop", info.crop);
  entry.Get("load_start_pos_w", info.load_start_pos_w);
  entry.Get("load_start_pos_h", info.load_start_pos_h);
  entry.Get("crop_size_w", info.crop_size_w);
  entry.Get("crop_size_h", info.crop_size_h);
  entry.Get("resize", info.resize);
  entry.Get("resize_output_w", info.resize_output_w);
  entry.Get("resize_output_h", info.resize_output_h);
  entry.Get("padding", info.padding);
  entry.Get("left_padding_size", info.left_padding_size);
  entry.Get("right_padding_size", info.right_padding_size);
  entry.Get("top_padding_size", info.top_padding_size);
  entry.Get("bottom_padding_size", info.bottom_padding_size);
  entry.Get("csc_switch", info.csc_switch);
  entry.Get("rbuv_swap_switch", info.rbuv_swap_switch);
  entry.Get("ax_swap_switch", info.ax_swap_switch);
  entry.Get("single_line_mode", info.single_line_mode);
  entry.Get("matrix_r0c0", info.matrix_r0c0);
  entry.Get("matrix_r0c1", info.matrix_r0c1);
  entry.Get("matrix_r0c2", info.matrix_r0c2);
  entry.Get("matrix_r1c0", info.matrix_r1c0);
  entry.Get("matrix_r1c1", info.matrix_r1c1);
  entry.Get("matrix_r1c2", info.matrix_r1c2);
  entry.Get("matrix_r2c0", info.matrix_r2c0);
  entry.Get("matrix_r2c1", info.matrix_r2c1);
  entry.Get("matrix_r2c2", info.matrix_r2c2);
  entry.Get("output_bias_0", info.output_bias_0);
  entry.Get("output_bias_1", info.output_bias_1);
  entry.Get("output_bias_2", info.output_bias_2);
  entry.Get("input_bias_0", info.input_bias_0);
  entry.Get("input_bias_1", info.input_bias_1);
  entry.Get("input_bias_2", info.input_bias_2);
  entry.Get("mean_chn_0", info.mean_chn_0);
  entry.Get("mean_chn_1", info.mean_chn_1);
  entry.Get("mean_chn_2", info.mean_chn_2);
  entry.Get("mean_chn_3", info.mean_chn_3);
  entry.Get("min_chn_0", info.min_chn_0);
  entry.Get("min_chn_1", info.min_chn_1);
  entry.Get("min_chn_2", info.min_chn_2);
  entry.Get("min_chn_3", info.min_chn_3);
  entry.Get("var_reci_chn_0", info.var_reci_chn_0);
  entry.Get("var_reci_chn_1", info.var_reci_chn_1);
  entry.Get("var_reci_chn_2", info.var_reci_chn_2);
  entry.Get("var_reci_chn_3", info.var_reci_chn_3);
  entry.Get("support_rotation", info.support_rotation);
  entry.Get("related_input_rank", info.related_input_rank);
  entry.Get("max_src_image_size", info.max_src_image_size);

  return info;
}

// 从 JsonFile 解析 OriginInputInfo
ge::OriginInputInfo ParseOriginInputFromJson(const ge::JsonFile &entry) {
  ge::OriginInputInfo info = {};
  entry.Get("orig_input_format", info.format);
  entry.Get("orig_input_data_type", info.data_type);
  entry.Get("orig_input_dim_num", info.dim_num);
  return info;
}

// 从 JsonFile 的字符串数组解析 InputOutputDims 列表
std::vector<ge::InputOutputDims> ParseAippDimsFromJson(const ge::JsonFile &entry, const char *key) {
  std::vector<ge::InputOutputDims> result;
  for (const auto &str : entry[key]) {
    ge::InputOutputDims dims;
    if (ParseAippDimInfo(str.get<std::string>(), dims) == ge::SUCCESS) {
      result.emplace_back(std::move(dims));
    }
  }
  return result;
}

// 从 model_meta.json 的 aipp 字段解析 AIPP 信息
ge::Status ParseAippJson(const ge::JsonFile &aipp_json, std::vector<Om2AippMeta> &aipp_infos, bool &has_aipp) {
  GELOGI("[OM2][AIPP] Parsing aipp section from model_meta.json");
  try {
    if (!aipp_json["aipp_infos"].is_array()) {
      return ge::SUCCESS;
    }
    has_aipp = true;
    for (const auto &item : aipp_json["aipp_infos"]) {
      if (!item.is_object()) {
        continue;
      }
      const ge::JsonFile aipp_item(item);
      uint32_t input_index = 0U;
      aipp_item.Get("index", input_index);
      if (input_index >= aipp_infos.size()) {
        aipp_infos.resize(input_index + 1U);
      }
      int32_t aipp_type = 0;
      aipp_item.Get("aipp_type", aipp_type);
      size_t aipp_data_index = 0U;
      aipp_item.Get("aipp_data_index", aipp_data_index);
      aipp_infos[input_index] = {
          static_cast<ge::InputAippType>(aipp_type),
          aipp_data_index,
          ParseAippConfigFromJson(aipp_item),
          ParseAippDimsFromJson(aipp_item, "aipp_inputs"),
          ParseAippDimsFromJson(aipp_item, "aipp_outputs"),
          ParseOriginInputFromJson(aipp_item),
      };
    }
  } catch (const std::exception &e) {
    GELOGW("[OM2][AIPP] Failed to parse aipp json: %s, falling back to no-AIPP", e.what());
    aipp_infos.clear();
    has_aipp = false;
    return ge::FAILED;
  }
  GELOGI("[OM2][AIPP] Successfully parsed aipp section");
  return ge::SUCCESS;
}

}  // namespace om2
}  // namespace gert
