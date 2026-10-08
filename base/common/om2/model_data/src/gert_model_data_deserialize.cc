/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "framework/om2/model_data/gert_model_data.h"
#include "common/dynamic_aipp.h"

#include <algorithm>
#include <cinttypes>
#include <cstring>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "framework/common/json_file.h"
#include "framework/om2/model_data/om2_package_contants.h"
#include "framework/common/zip_archive_reader.h"
#include "common/checker.h"
#include "framework/common/gert_model_data_deserialize.h"
#include "framework/common/gert_model_data_utils.h"
#include "ge/ge_error_codes.h"

namespace gert {
namespace {

constexpr size_t kAippDimPartsNum = 6U;
constexpr size_t kAippDimNameIdx = 2U;
constexpr size_t kAippDimSizeIdx = 3U;
constexpr size_t kAippDimDimNumIdx = 4U;
constexpr size_t kAippDimShapeIdx = 5U;
constexpr int32_t kAippDecimalRadix = 10;

// ===== 1. 通用工具 =====
bool EndsWith(const std::string &str, const std::string &suffix) {
  return (str.size() >= suffix.size()) && (str.compare(str.size() - suffix.size(), suffix.size(), suffix) == 0);
}

std::pair<std::string, std::string> ExtractParentDirAndFileName(const std::string &abs_path) {
  if (abs_path.empty()) {
    return {"", ""};
  }
  size_t last_slash = abs_path.find_last_of('/');
  if (last_slash == std::string::npos) {
    return {"", abs_path};
  }
  if (last_slash == 0) {
    return {"/", abs_path.substr(1)};
  }
  return {abs_path.substr(0, last_slash + 1), abs_path.substr(last_slash + 1)};
}

// ===== 2. 通用 JSON 解析 =====
ge::Status ParseTensorDescFromJson(const ge::JsonFile &json_file, gert::GertTensorDesc &desc) {
  ge::JsonFile::TryGetAndApply<std::string>(json_file, "name",
                                            [&](const std::string &v) { desc.name = gert::GertMakeStr(v); });
  ge::JsonFile::TryGetAndApply<std::vector<int64_t>>(json_file, "shape",
                                                     [&](const std::vector<int64_t> &v) { desc.shape = v; });
  ge::JsonFile::TryGetAndApply<int32_t>(json_file, "data_type",
                                        [&](const int32_t &v) { desc.data_type = static_cast<ge::DataType>(v); });
  ge::JsonFile::TryGetAndApply<int32_t>(json_file, "format",
                                        [&](const int32_t &v) { desc.format = static_cast<ge::Format>(v); });
  ge::JsonFile::TryGetAndApply<size_t>(json_file, "size", [&](const size_t &v) { desc.size = v; });
  ge::JsonFile::TryGetAndApply<std::vector<std::pair<int64_t, int64_t>>>(
      json_file, "shape_range", [&](const std::vector<std::pair<int64_t, int64_t>> &v) { desc.shape_range = v; });
  return ge::SUCCESS;
}

// ===== 3. AIPP 解析 =====

std::vector<std::string> SplitString(const std::string &str, const char delim) {
  std::vector<std::string> elems;
  size_t start = 0U;
  while (start <= str.size()) {
    const auto pos = str.find(delim, start);
    elems.emplace_back(str.substr(start, pos == std::string::npos ? str.size() - start : pos - start));
    if (pos == std::string::npos) {
      break;
    }
    start = pos + 1U;
  }
  return elems;
}

// 将 "NCHW:DT_FLOAT:data:0:4:1,3,224,224" 格式的字符串解析为 InputOutputDims
ge::Status ParseAippDimInfo(const std::string &info_str, ge::InputOutputDims &dims_info) {
  const auto parts = SplitString(info_str, ':');
  if (parts.size() != kAippDimPartsNum) {
    GELOGW("[OM2][AIPP] Invalid aipp dim info: %s, parts=%zu", info_str.c_str(), parts.size());
    return ge::FAILED;
  }
  dims_info.name = parts[kAippDimNameIdx];
  dims_info.size = static_cast<uint32_t>(std::strtol(parts[kAippDimSizeIdx].c_str(), nullptr, kAippDecimalRadix));
  dims_info.dim_num = static_cast<size_t>(std::strtol(parts[kAippDimDimNumIdx].c_str(), nullptr, kAippDecimalRadix));

  const auto dim_strs = SplitString(parts[kAippDimShapeIdx], ',');
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

  (void)entry.Get("aipp_mode", info.aipp_mode);
  (void)entry.Get("input_format", info.input_format);
  (void)entry.Get("src_image_size_w", info.src_image_size_w);
  (void)entry.Get("src_image_size_h", info.src_image_size_h);
  (void)entry.Get("crop", info.crop);
  (void)entry.Get("load_start_pos_w", info.load_start_pos_w);
  (void)entry.Get("load_start_pos_h", info.load_start_pos_h);
  (void)entry.Get("crop_size_w", info.crop_size_w);
  (void)entry.Get("crop_size_h", info.crop_size_h);
  (void)entry.Get("resize", info.resize);
  (void)entry.Get("resize_output_w", info.resize_output_w);
  (void)entry.Get("resize_output_h", info.resize_output_h);
  (void)entry.Get("padding", info.padding);
  (void)entry.Get("left_padding_size", info.left_padding_size);
  (void)entry.Get("right_padding_size", info.right_padding_size);
  (void)entry.Get("top_padding_size", info.top_padding_size);
  (void)entry.Get("bottom_padding_size", info.bottom_padding_size);
  (void)entry.Get("csc_switch", info.csc_switch);
  (void)entry.Get("rbuv_swap_switch", info.rbuv_swap_switch);
  (void)entry.Get("ax_swap_switch", info.ax_swap_switch);
  (void)entry.Get("single_line_mode", info.single_line_mode);
  (void)entry.Get("matrix_r0c0", info.matrix_r0c0);
  (void)entry.Get("matrix_r0c1", info.matrix_r0c1);
  (void)entry.Get("matrix_r0c2", info.matrix_r0c2);
  (void)entry.Get("matrix_r1c0", info.matrix_r1c0);
  (void)entry.Get("matrix_r1c1", info.matrix_r1c1);
  (void)entry.Get("matrix_r1c2", info.matrix_r1c2);
  (void)entry.Get("matrix_r2c0", info.matrix_r2c0);
  (void)entry.Get("matrix_r2c1", info.matrix_r2c1);
  (void)entry.Get("matrix_r2c2", info.matrix_r2c2);
  (void)entry.Get("output_bias_0", info.output_bias_0);
  (void)entry.Get("output_bias_1", info.output_bias_1);
  (void)entry.Get("output_bias_2", info.output_bias_2);
  (void)entry.Get("input_bias_0", info.input_bias_0);
  (void)entry.Get("input_bias_1", info.input_bias_1);
  (void)entry.Get("input_bias_2", info.input_bias_2);
  (void)entry.Get("mean_chn_0", info.mean_chn_0);
  (void)entry.Get("mean_chn_1", info.mean_chn_1);
  (void)entry.Get("mean_chn_2", info.mean_chn_2);
  (void)entry.Get("mean_chn_3", info.mean_chn_3);
  (void)entry.Get("min_chn_0", info.min_chn_0);
  (void)entry.Get("min_chn_1", info.min_chn_1);
  (void)entry.Get("min_chn_2", info.min_chn_2);
  (void)entry.Get("min_chn_3", info.min_chn_3);
  (void)entry.Get("var_reci_chn_0", info.var_reci_chn_0);
  (void)entry.Get("var_reci_chn_1", info.var_reci_chn_1);
  (void)entry.Get("var_reci_chn_2", info.var_reci_chn_2);
  (void)entry.Get("var_reci_chn_3", info.var_reci_chn_3);
  (void)entry.Get("support_rotation", info.support_rotation);
  (void)entry.Get("related_input_rank", info.related_input_rank);
  (void)entry.Get("max_src_image_size", info.max_src_image_size);

  return info;
}

// 从 JsonFile 解析 OriginInputInfo
ge::OriginInputInfo ParseOriginInputFromJson(const ge::JsonFile &entry) {
  ge::OriginInputInfo info = {};
  int32_t format_val = 0;
  int32_t data_type_val = 0;
  (void)entry.Get("orig_input_format", format_val);
  (void)entry.Get("orig_input_data_type", data_type_val);
  (void)entry.Get("orig_input_dim_num", info.dim_num);
  info.format = static_cast<ge::Format>(format_val);
  info.data_type = static_cast<ge::DataType>(data_type_val);
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
ge::Status ParseAippJson(const ge::JsonFile &aipp_json,
                         std::vector<std::unique_ptr<GertModelDataAippMeta>> &aipp_infos) {
  GELOGI("[OM2][AIPP] Parsing aipp section from model_meta.json");
  try {
    if (!aipp_json["aipp_infos"].is_array()) {
      return ge::SUCCESS;
    }
    for (const auto &item : aipp_json["aipp_infos"]) {
      if (!item.is_object()) {
        continue;
      }
      const ge::JsonFile aipp_item(item);
      uint32_t input_index = 0U;
      (void)aipp_item.Get("index", input_index);
      if (input_index >= aipp_infos.size()) {
        aipp_infos.resize(input_index + 1U);
      }
      if (!aipp_infos[input_index]) {
        aipp_infos[input_index] = std::make_unique<GertModelDataAippMeta>();
      }
      int32_t aipp_type = 0;
      (void)aipp_item.Get("aipp_type", aipp_type);
      size_t aipp_data_index = 0U;
      (void)aipp_item.Get("aipp_data_index", aipp_data_index);
      aipp_infos[input_index]->aipp_type = static_cast<ge::InputAippType>(aipp_type);
      aipp_infos[input_index]->aipp_data_index = aipp_data_index;
      aipp_infos[input_index]->aipp_config_info =
          std::make_unique<ge::AippConfigInfo>(ParseAippConfigFromJson(aipp_item));
      auto aipp_input_dims_vec = ParseAippDimsFromJson(aipp_item, "aipp_inputs");
      for (auto &d : aipp_input_dims_vec) {
        aipp_infos[input_index]->aipp_input_dims.push_back(std::make_unique<ge::InputOutputDims>(std::move(d)));
      }
      auto aipp_output_dims_vec = ParseAippDimsFromJson(aipp_item, "aipp_outputs");
      for (auto &d : aipp_output_dims_vec) {
        aipp_infos[input_index]->aipp_output_dims.push_back(std::make_unique<ge::InputOutputDims>(std::move(d)));
      }
      if (!aipp_input_dims_vec.empty()) {
        aipp_infos[input_index]->orig_input_info =
            std::make_unique<ge::OriginInputInfo>(ParseOriginInputFromJson(aipp_item));
      }
    }
  } catch (const std::exception &e) {
    GELOGW("[OM2][AIPP] Failed to parse aipp json: %s, falling back to no-AIPP", e.what());
    aipp_infos.clear();
    return ge::FAILED;
  }
  GELOGI("[OM2][AIPP] Successfully parsed aipp section");
  return ge::SUCCESS;
}

// ===== 4. RTVar 解析 =====

ge::Status ParseTransNodeFromJson(const ge::JsonFile &json_file, RTTransNodeInfo &node_info) {
  ge::JsonFile::TryGetAndApply<std::string>(json_file, "node_type",
                                            [&](const std::string &v) { node_info.node_type = gert::GertMakeStr(v); });
  ge::JsonFile input_json;
  if (json_file.Get("input", input_json)) {
    GE_ASSERT_SUCCESS(ParseTensorDescFromJson(input_json, node_info.input));
  }
  ge::JsonFile output_json;
  if (json_file.Get("output", output_json)) {
    GE_ASSERT_SUCCESS(ParseTensorDescFromJson(output_json, node_info.output));
  }
  return ge::SUCCESS;
}

ge::Status ParseCopyInfoFromJson(const ge::JsonFile &json_file, RTCopyNodeInfo &copy_info) {
  ge::JsonFile::TryGetAndApply<std::string>(
      json_file, "src_var_name", [&](const std::string &v) { copy_info.src_var_name = gert::GertMakeStr(v); });
  ge::JsonFile src_tensor_desc_json;
  if (json_file.Get("src_tensor_desc", src_tensor_desc_json)) {
    GE_ASSERT_SUCCESS(ParseTensorDescFromJson(src_tensor_desc_json, copy_info.src_tensor_desc));
  }
  return ge::SUCCESS;
}

// init_data_offset/init_data_size 指向 var_weight_data 文件内的区间，校验后切片写入 entry.init_data
ge::Status ParseVarEntryFromJson(const ge::JsonFile &json_file, const uint8_t *weight_data, size_t weight_data_size,
                                 RTVarEntry &entry) {
  ge::JsonFile::TryGetAndApply<std::string>(json_file, "var_name",
                                            [&](const std::string &v) { entry.var_name = gert::GertMakeStr(v); });
  ge::JsonFile::TryGetAndApply<std::string>(json_file, "file_name",
                                            [&](const std::string &v) { entry.file_name = gert::GertMakeStr(v); });
  ge::JsonFile::TryGetAndApply<std::string>(json_file, "var_key",
                                            [&](const std::string &v) { entry.var_key = gert::GertMakeStr(v); });
  ge::JsonFile::TryGetAndApply<std::string>(json_file, "op_type",
                                            [&](const std::string &v) { entry.op_type = gert::GertMakeStr(v); });
  ge::JsonFile::TryGetAndApply<uint64_t>(json_file, "logic_addr", [&](const uint64_t &v) { entry.logic_addr = v; });
  ge::JsonFile::TryGetAndApply<uint64_t>(json_file, "size", [&](const uint64_t &v) { entry.size = v; });
  ge::JsonFile::TryGetAndApply<uint64_t>(json_file, "memory_type", [&](const uint64_t &v) { entry.memory_type = v; });
  ge::JsonFile::TryGetAndApply<uint64_t>(json_file, "changed_graph_id",
                                         [&](const uint64_t &v) { entry.changed_graph_id = v; });
  ge::JsonFile::TryGetAndApply<uint64_t>(json_file, "allocated_graph_id",
                                         [&](const uint64_t &v) { entry.allocated_graph_id = v; });

  ge::JsonFile tensor_desc_json;
  if (json_file.Get("tensor_desc", tensor_desc_json)) {
    GE_ASSERT_SUCCESS(ParseTensorDescFromJson(tensor_desc_json, entry.tensor_desc));
  }

  ge::JsonFile::json trans_road_json;
  if (json_file.Get("trans_road", trans_road_json) && trans_road_json.is_array()) {
    for (const auto &node_json : trans_road_json) {
      RTTransNodeInfo node_info;
      GE_ASSERT_SUCCESS(ParseTransNodeFromJson(ge::JsonFile(node_json), node_info));
      entry.trans_road.emplace_back(std::move(node_info));
    }
  }

  ge::JsonFile copy_info_json;
  if (json_file.Get("copy_info", copy_info_json)) {
    GE_ASSERT_SUCCESS(ParseCopyInfoFromJson(copy_info_json, entry.copy_info));
  }

  size_t init_data_offset = 0U;
  size_t init_data_size = 0U;
  (void)json_file.Get("init_data_offset", init_data_offset);
  (void)json_file.Get("init_data_size", init_data_size);
  if (init_data_size > 0U) {
    const bool is_init_data_valid =
        (init_data_offset <= weight_data_size) && (init_data_size <= (weight_data_size - init_data_offset));
    if (!is_init_data_valid) {
      GELOGW("[OM2][Var] Invalid or missing init data for var=%s, skip init_data.", gert::GertGetStr(entry.var_name));
    } else {
      entry.init_data.assign(weight_data + init_data_offset, weight_data + init_data_offset + init_data_size);
    }
  }
  return ge::SUCCESS;
}

ge::Status ParseVarMetaFromJson(const ge::JsonFile &json_file, gert::GertModelDataVarMeta &meta) {
  ge::JsonFile::TryGetAndApply<size_t>(json_file, "index", [&](const size_t &v) { meta.index = v; });
  ge::JsonFile::TryGetAndApply<std::string>(json_file, "var_name",
                                            [&](const std::string &v) { meta.var_name = gert::GertMakeStr(v); });
  ge::JsonFile::TryGetAndApply<std::string>(json_file, "op_type",
                                            [&](const std::string &v) { meta.op_type = gert::GertMakeStr(v); });
  ge::JsonFile::TryGetAndApply<std::string>(json_file, "op_name",
                                            [&](const std::string &v) { meta.op_name = gert::GertMakeStr(v); });

  ge::JsonFile tensor_desc_json;
  if (json_file.Get("tensor_desc", tensor_desc_json)) {
    GE_ASSERT_SUCCESS(ParseTensorDescFromJson(tensor_desc_json, meta.tensor_desc));
  }
  return ge::SUCCESS;
}

// ===== 5. 版本兼容性校验 =====

uint32_t ParseVersion(const std::string &version) {
  uint32_t major = 0U;
  uint32_t minor = 0U;
  (void)sscanf_s(version.c_str(), "%u.%u", &major, &minor);
  return major * 10000U + minor;
}

uint32_t GetMajorVersion(uint32_t version) {
  return version / 10000U;
}

std::string BuildUsedFeaturesStr(
    const std::map<std::unique_ptr<char[]>, std::unique_ptr<char[]>, UniquePtrCharCompare> &used_features) {
  if (used_features.empty()) {
    return "{}";
  }
  ge::JsonFile json;
  for (const auto &[name, version] : used_features) {
    (void)json.Set(gert::GertGetStr(name), gert::GertGetStr(version));
  }
  return json.Dump(false);
}

ge::Status ValidateVersionCompatibility(const GertModelDataCompatibility &compat) {
  const uint32_t compiler_ver = ParseVersion(gert::GertGetStr(compat.compiler_version));
  const uint32_t executor_ver = ParseVersion(GERT_EXECUTOR_VERSION);
  const std::string features_str = BuildUsedFeaturesStr(compat.used_features);

  if (GetMajorVersion(compiler_ver) > GetMajorVersion(executor_ver)) {
    REPORT_INNER_ERR_MSG("E19999",
                         "[OM2] Version incompatible: compiler_version=%s (major=%u) > executor_version=%s (major=%u), "
                         "used_features=%s",
                         gert::GertGetStr(compat.compiler_version), GetMajorVersion(compiler_ver),
                         GERT_EXECUTOR_VERSION, GetMajorVersion(executor_ver), features_str.c_str());
    GELOGE(ACL_ERROR_GE_PARAM_INVALID,
           "[OM2] Version incompatible: compiler_version=%s (major=%u) > executor_version=%s (major=%u)",
           gert::GertGetStr(compat.compiler_version), GetMajorVersion(compiler_ver), GERT_EXECUTOR_VERSION,
           GetMajorVersion(executor_ver));
    return ACL_ERROR_GE_PARAM_INVALID;
  }

  if (compat.required_executor_version != nullptr && gert::GertGetStr(compat.required_executor_version)[0] != '\0') {
    const uint32_t required_ver = ParseVersion(gert::GertGetStr(compat.required_executor_version));
    if (required_ver > executor_ver) {
      REPORT_INNER_ERR_MSG("E19999",
                           "[OM2] Version incompatible: required_executor_version=%s > executor_version=%s, "
                           "used_features=%s",
                           gert::GertGetStr(compat.required_executor_version), GERT_EXECUTOR_VERSION,
                           features_str.c_str());
      GELOGE(ACL_ERROR_GE_PARAM_INVALID,
             "[OM2] Version incompatible: required_executor_version=%s > executor_version=%s",
             gert::GertGetStr(compat.required_executor_version), GERT_EXECUTOR_VERSION);
      return ACL_ERROR_GE_PARAM_INVALID;
    }
  }

  return ge::SUCCESS;
}

// ===== 6. 文件反序列化（按主流程调用序）=====

// manifest 是机制依赖（model_num 驱动模型定位），不占用掩码位，两种模式下均必须存在
ge::Status DeserializeManifest(const gert::ZipArchiveReader &archive, const std::string &relative_path,
                               GertModelData &model_data) {
  const auto full_path = archive.FindEntry(relative_path);
  if (full_path.empty()) {
    REPORT_PREDEFINED_ERR_MSG(
        "E10059", std::vector<const char *>({"stage", "reason"}),
        std::vector<const char *>({"DeserializeGertModelData", "manifest.json not found in ZIP archive."}));
    GELOGE(ACL_ERROR_GE_PARAM_INVALID, "[OM2] manifest.json not found in ZIP archive.");
    return ACL_ERROR_GE_PARAM_INVALID;
  }
  size_t buff_size = 0U;
  auto buff_data = archive.ExtractToMem(full_path, buff_size);
  GE_ASSERT_NOTNULL(buff_data, "[OM2] Failed to extract %s", full_path.c_str());
  GE_ASSERT_TRUE(buff_size > 0U);

  const ge::JsonFile json_file(buff_data.get(), buff_size);
  GE_ASSERT_TRUE(json_file.IsValid(), "[OM2] Invalid manifest.json");

  auto &manifest = *model_data.manifest;
  ge::JsonFile compat_json;
  GE_ASSERT_TRUE(json_file.Get(gert::OM2_MANIFEST_KEY_COMPATIBILITY, compat_json),
                 "[OM2] manifest.json missing 'compatibility' field");
  ge::JsonFile::TryGetAndApply<std::string>(
      compat_json, gert::OM2_MANIFEST_KEY_COMPILER_VERSION,
      [&](const std::string &v) { manifest.compatibility.compiler_version = gert::GertMakeStr(v); });
  ge::JsonFile::TryGetAndApply<std::string>(
      compat_json, gert::OM2_MANIFEST_KEY_REQUIRED_EXECUTOR_VERSION,
      [&](const std::string &v) { manifest.compatibility.required_executor_version = gert::GertMakeStr(v); });
  ge::JsonFile::TryGetAndApply<std::map<std::string, std::string>>(
      compat_json, gert::OM2_MANIFEST_KEY_USED_FEATURES, [&](const std::map<std::string, std::string> &v) {
        for (const auto &[feature_name, feature_version] : v) {
          manifest.compatibility.used_features[gert::GertMakeStr(feature_name)] = gert::GertMakeStr(feature_version);
        }
      });

  GE_ASSERT_TRUE(json_file.Get(gert::OM2_MODEL_NUM, manifest.model_num),
                 "[OM2] manifest.json missing 'model_num' field");
  ge::JsonFile::TryGetAndApply<std::string>(json_file, gert::OM2_ATC_COMMAND,
                                            [&](const std::string &v) { manifest.atc_command = gert::GertMakeStr(v); });

  GELOGI("[OM2] Manifest deserialized: compiler_version=%s, required_executor_version=%s, executor_version=%s",
         gert::GertGetStr(manifest.compatibility.compiler_version),
         gert::GertGetStr(manifest.compatibility.required_executor_version), GERT_EXECUTOR_VERSION);
  return ValidateVersionCompatibility(manifest.compatibility);
}

ge::Status ParseModelMetaInputs(const ge::JsonFile &json_file, gert::GertModelDataModelMeta &model_meta) {
  ge::JsonFile::json inputs_json;
  if (json_file.Get("inputs", inputs_json) && inputs_json.is_array()) {
    for (size_t i = 0UL; i < inputs_json.size(); ++i) {
      const ge::JsonFile input_file(inputs_json[i]);
      gert::GertTensorDesc desc;
      GE_ASSERT_SUCCESS(ParseTensorDescFromJson(input_file, desc));
      GE_ASSERT_TRUE(!desc.shape.empty(), "[OM2] Input tensor at index %zu is missing 'shape' field", i);
      std::vector<int64_t> max_gear_shape;
      if (input_file.Get("max_gear_shape", max_gear_shape)) {
        model_meta.origin_input_dims.emplace_back(desc.shape);
        desc.shape = max_gear_shape;
      }
      gert::GertTensorDesc desc_v2 = gert::MakeGertTensorDesc(desc);
      std::vector<int64_t> shape_v2;
      if (input_file.Get("shape_aclmdlGetInputDimsV2", shape_v2)) {
        desc_v2.shape = shape_v2;
      }
      model_meta.input_desc.emplace_back(std::move(desc));
      model_meta.input_desc_v2.emplace_back(std::move(desc_v2));
    }
  }
  return ge::SUCCESS;
}

ge::Status ParseModelMetaOutputs(const ge::JsonFile &json_file, gert::GertModelDataModelMeta &model_meta) {
  ge::JsonFile::json outputs_json;
  if (json_file.Get("outputs", outputs_json) && outputs_json.is_array()) {
    for (const auto &output_json : outputs_json) {
      const ge::JsonFile output_file(output_json);
      gert::GertTensorDesc desc;
      GE_ASSERT_SUCCESS(ParseTensorDescFromJson(output_file, desc));
      gert::GertTensorDesc desc_v2 = gert::MakeGertTensorDesc(desc);
      model_meta.output_desc.emplace_back(std::move(desc));
      model_meta.output_desc_v2.emplace_back(std::move(desc_v2));
    }
  }
  return ge::SUCCESS;
}

void ParseGearFromJson(const ge::JsonFile &gear_file, const size_t gear_idx, gert::GertModelDataModelMeta &model_meta) {
  std::vector<int64_t> gear_inputs;
  if (gear_file.Get("inputs", gear_inputs)) {
    model_meta.dynamic_batch_info.push_back(std::move(gear_inputs));
  }

  ge::JsonFile::json gear_outputs_json;
  if (gear_file.Get("outputs", gear_outputs_json) && gear_outputs_json.is_array()) {
    for (size_t out_idx = 0UL; out_idx < gear_outputs_json.size(); ++out_idx) {
      const auto &dims = gear_outputs_json[out_idx];
      if (!dims.is_array()) {
        continue;
      }
      std::string shape_str = std::to_string(gear_idx) + "," + std::to_string(out_idx);
      for (size_t i = 0UL; i < dims.size(); ++i) {
        shape_str += ",";
        shape_str += std::to_string(dims[i].get<int64_t>());
      }
      model_meta.dynamic_output_shape.emplace_back(gert::GertMakeStr(shape_str));
    }
  }
}

ge::Status ParseModelMetaDynamicDims(const ge::JsonFile &json_file, gert::GertModelDataModelMeta &model_meta) {
  ge::JsonFile dynamic_dims_file;
  if (!json_file.Get("dynamic_dims", dynamic_dims_file) || !dynamic_dims_file.IsValid()) {
    return ge::SUCCESS;
  }
  (void)dynamic_dims_file.Get("dynamic_type", model_meta.dynamic_type);
  std::vector<std::string> user_designate_shape_order;
  (void)dynamic_dims_file.Get("user_designate_shape_order", user_designate_shape_order);
  for (auto &s : user_designate_shape_order) {
    model_meta.user_designate_shape_order.emplace_back(gert::GertMakeStr(s));
  }

  ge::JsonFile::json gears_json;
  if (dynamic_dims_file.Get("gears", gears_json) && gears_json.is_array()) {
    for (size_t gear_idx = 0UL; gear_idx < gears_json.size(); ++gear_idx) {
      ParseGearFromJson(ge::JsonFile(gears_json[gear_idx]), gear_idx, model_meta);
    }
  }
  return ge::SUCCESS;
}

ge::Status DeserializeModelMeta(const gert::ZipArchiveReader &archive, const std::string &relative_path,
                                gert::GertModelDataModel &unit) {
  const auto full_path = archive.FindEntry(relative_path);
  if (full_path.empty()) {
    REPORT_PREDEFINED_ERR_MSG(
        "E10059", std::vector<const char *>({"stage", "reason"}),
        std::vector<const char *>({"DeserializeGertModelData", "model_meta.json not found in ZIP archive."}));
    GELOGE(ACL_ERROR_GE_PARAM_INVALID, "[OM2] model_meta.json not found in ZIP archive.");
    return ACL_ERROR_GE_PARAM_INVALID;
  }
  size_t buff_size = 0U;
  auto buff_data = archive.ExtractToMem(full_path, buff_size);
  GE_ASSERT_NOTNULL(buff_data, "[OM2] Failed to extract %s", full_path.c_str());
  GE_ASSERT_TRUE(buff_size > 0U);
  const ge::JsonFile json_file(buff_data.get(), buff_size);
  GE_ASSERT_TRUE(json_file.IsValid(), "[OM2] Invalid model_meta.json");
  auto &model_meta = *unit.model_meta;

  GE_CHK_STATUS_RET_NOLOG(ParseModelMetaInputs(json_file, model_meta));
  GE_CHK_STATUS_RET_NOLOG(ParseModelMetaOutputs(json_file, model_meta));
  GE_CHK_STATUS_RET_NOLOG(ParseModelMetaDynamicDims(json_file, model_meta));

  (void)json_file.Get("work_size", model_meta.work_size);
  (void)json_file.Get("zero_copy_size", model_meta.zero_copy_size);
  std::string model_name_str;
  (void)json_file.Get("name", model_name_str);
  model_meta.model_name = gert::GertMakeStr(model_name_str);
  GE_ASSERT_TRUE(gert::GertGetStr(model_meta.model_name)[0] != '\0', "[OM2] model_meta.json missing 'name' field");

  // 读取 aipp 字段
  ge::JsonFile aipp_json;
  if (json_file.Get("aipp", aipp_json) && aipp_json.IsValid()) {
    const ge::Status aipp_ret = ParseAippJson(aipp_json, model_meta.aipp_infos);
    if (aipp_ret != ge::SUCCESS) {
      GELOGW("[OM2][AIPP] ParseAippJson failed, ret=%u", aipp_ret);
    }
  }
  return ge::SUCCESS;
}

ge::Status DeserializeCodegen(const gert::ZipArchiveReader &archive, const std::string &runtime_dir_prefix,
                              gert::GertModelDataModel &unit) {
  for (const auto &entry : archive.ListFilesByRelativePrefix(runtime_dir_prefix)) {
    if (!EndsWith(entry, ".so")) {
      continue;
    }
    gert::GertModelDataFile artifact;
    artifact.file_name = gert::GertMakeStr(ExtractParentDirAndFileName(entry).second);
    // ExtractToMem：STORED entry 零拷贝视图，压缩 entry 解压拥有，避免中转拷贝
    size_t so_size = 0U;
    artifact.data = archive.ExtractToMem(entry, so_size);
    GE_ASSERT_NOTNULL(artifact.data, "[OM2] Failed to extract entry %s", entry.c_str());
    GE_ASSERT_TRUE(so_size > 0U, "[OM2] Empty archive entry %s", entry.c_str());
    artifact.data_size = static_cast<uint64_t>(so_size);
    unit.runtime->so_artifact = std::move(artifact);
    return ge::SUCCESS;
  }
  REPORT_PREDEFINED_ERR_MSG(
      "E10059", std::vector<const char *>({"stage", "reason"}),
      std::vector<const char *>({"DeserializeGertModelData", "Compiled .so not found in ZIP archive."}));
  GELOGE(ACL_ERROR_GE_PARAM_INVALID, "[OM2] Compiled .so not found in ZIP archive.");
  return ACL_ERROR_GE_PARAM_INVALID;
}

ge::Status DeserializeConstantsConfig(const gert::ZipArchiveReader &archive, const std::string &relative_path,
                                      gert::GertModelDataModel &unit) {
  const auto full_path = archive.FindEntry(relative_path);
  if (full_path.empty()) {
    REPORT_PREDEFINED_ERR_MSG(
        "E10059", std::vector<const char *>({"stage", "reason"}),
        std::vector<const char *>({"DeserializeGertModelData", "constants config not found in ZIP archive."}));
    GELOGE(ACL_ERROR_GE_PARAM_INVALID, "[OM2] constants config not found in ZIP archive.");
    return ACL_ERROR_GE_PARAM_INVALID;
  }
  size_t buff_size = 0U;
  auto buff_data = archive.ExtractToMem(full_path, buff_size);
  GE_ASSERT_NOTNULL(buff_data, "[OM2] Failed to extract %s", full_path.c_str());
  GE_ASSERT_TRUE(buff_size > 0U);
  const ge::JsonFile json_file(buff_data.get(), buff_size);
  GE_ASSERT_TRUE(json_file.IsValid(), "[OM2] Invalid constants config JSON from entry %s", full_path.c_str());

  (void)json_file.Get("internal_weight_size", unit.constants_config->internal_weight_size);

  ge::JsonFile::json consts_json;
  if (json_file.Get("consts", consts_json) && consts_json.is_object()) {
    for (auto &[key, val] : consts_json.items()) {
      (void)key;
      const ge::JsonFile val_file(val);
      gert::GertModelDataConstMeta meta;
      (void)val_file.Get("index", meta.index);
      std::string type_str;
      (void)val_file.Get("type", type_str);
      meta.type = gert::GertMakeStr(type_str);
      std::string file_name_str;
      (void)val_file.Get("file_name", file_name_str);
      meta.file_name = gert::GertMakeStr(file_name_str);
      std::string file_path_str;
      (void)val_file.Get("file_path", file_path_str);
      meta.file_path = gert::GertMakeStr(file_path_str);
      (void)val_file.Get("offset", meta.offset);
      (void)val_file.Get("size", meta.size);
      std::string op_name_str;
      (void)val_file.Get("op_name", op_name_str);
      meta.op_name = gert::GertMakeStr(op_name_str);
      (void)unit.constants_config->consts.emplace_back(std::make_unique<gert::GertModelDataConstMeta>(std::move(meta)));
    }
  }
  return ge::SUCCESS;
}

ge::Status DeserializeWeight(const gert::ZipArchiveReader &archive, const std::string &relative_path,
                             gert::GertModelDataModel &unit, std::unique_ptr<gert::GertModelDataFile> &weight_slot) {
  const auto full_path = archive.FindEntry(relative_path);
  if (full_path.empty()) {
    GELOGW("[OM2] Optional file [%s] not found in ZIP archive, skipped.", relative_path.c_str());
    return ge::SUCCESS;
  }
  size_t buffer_size{0U};
  auto buffer = archive.ExtractToMem(full_path, buffer_size);
  GE_ASSERT_NOTNULL(buffer, "[OM2] Failed to extract entry %s", full_path.c_str());
  GE_ASSERT_TRUE(buffer_size > 0U, "[OM2] Empty archive entry %s", full_path.c_str());
  weight_slot = std::make_unique<gert::GertModelDataFile>();
  weight_slot->data = std::move(buffer);
  weight_slot->data_size = static_cast<uint64_t>(buffer_size);
  // file_name 取 entry 基名（如 "constant_0"），与 INTERNAL 常量 meta 的 file_name 对应，供按名查找数据源
  weight_slot->file_name = gert::GertMakeStr(ExtractParentDirAndFileName(full_path).second);
  if (weight_slot->data_size != unit.constants_config->internal_weight_size) {
    REPORT_INNER_ERR_MSG("E19999", "[OM2] constant_0 size mismatch with constants config, file size %llu, json %llu.",
                         static_cast<unsigned long long>(weight_slot->data_size),
                         static_cast<unsigned long long>(unit.constants_config->internal_weight_size));
    GELOGE(ACL_ERROR_GE_PARAM_INVALID,
           "[OM2] constant_0 size mismatch with constants config, file size %llu, json %llu.",
           static_cast<unsigned long long>(weight_slot->data_size),
           static_cast<unsigned long long>(unit.constants_config->internal_weight_size));
    return ACL_ERROR_GE_PARAM_INVALID;
  }
  return ge::SUCCESS;
}

ge::Status DeserializeOpAttr(const gert::ZipArchiveReader &archive, const std::string &relative_path,
                             gert::GertModelDataModel &unit) {
  const auto full_path = archive.FindEntry(relative_path);
  if (full_path.empty()) {
    REPORT_PREDEFINED_ERR_MSG(
        "E10059", std::vector<const char *>({"stage", "reason"}),
        std::vector<const char *>({"DeserializeGertModelData", "op_attr.json not found in ZIP archive."}));
    GELOGE(ACL_ERROR_GE_PARAM_INVALID, "[OM2] op_attr.json not found in ZIP archive.");
    return ACL_ERROR_GE_PARAM_INVALID;
  }
  size_t buff_size = 0U;
  auto buff_data = archive.ExtractToMem(full_path, buff_size);
  GE_ASSERT_NOTNULL(buff_data, "[OM2] Failed to extract %s", full_path.c_str());
  GE_ASSERT_TRUE(buff_size > 0U);
  // 原文直存，由 executor 侧解析（与主线 develop 行为一致）
  unit.op_attr_json = gert::GertMakeStr(std::string(reinterpret_cast<const char *>(buff_data.get()), buff_size));
  return ge::SUCCESS;
}

// data/model_%s/variables_config.json（graph_id/var_metas/entries）+ data/variables/var_weight_data_%s：
// 解析 graph_id/var_metas 与 entries，按 entry 的 file_name 在 data/variables/ 下寻址权重文件，
// 并按 init_data_offset/size 切片写入 entry.init_data
ge::Status DeserializeVariablesData(const gert::ZipArchiveReader &archive, const std::string &relative_path,
                                    gert::GertModelDataModel &unit) {
  const auto full_path = archive.FindEntry(relative_path);
  if (full_path.empty()) {
    GELOGW("[OM2] Optional file [%s] not found in ZIP archive, skipped.", relative_path.c_str());
    return ge::SUCCESS;
  }
  size_t buff_size = 0U;
  auto buff_data = archive.ExtractToMem(full_path, buff_size);
  GE_ASSERT_NOTNULL(buff_data, "[OM2] Failed to extract %s", full_path.c_str());
  GE_ASSERT_TRUE(buff_size > 0U);
  const ge::JsonFile json_file(buff_data.get(), buff_size);
  GE_ASSERT_TRUE(json_file.IsValid(), "[OM2] Invalid variables config JSON from entry %s", full_path.c_str());

  unit.variables_config = std::make_unique<gert::GertModelDataVariablesConfig>();
  (void)json_file.Get("graph_id", unit.variables_config->graph_id);
  ge::JsonFile::json var_metas_json;
  if (json_file.Get("var_metas", var_metas_json) && var_metas_json.is_array()) {
    for (const auto &meta_json : var_metas_json) {
      gert::GertModelDataVarMeta meta;
      GE_ASSERT_SUCCESS(ParseVarMetaFromJson(ge::JsonFile(meta_json), meta));
      (void)unit.variables_config->var_metas.emplace_back(
          std::make_unique<gert::GertModelDataVarMeta>(std::move(meta)));
    }
  }

  ge::JsonFile::json entries_json;
  if (!json_file.Get("entries", entries_json) || !entries_json.is_object() || entries_json.empty()) {
    return ge::SUCCESS;
  }
  // 权重文件按第一条 entry 的 file_name 在 data/variables/ 下寻址（每模型一个文件，全部 entry 共享）
  std::string first_file_name;
  (void)ge::JsonFile(entries_json.begin().value()).Get("file_name", first_file_name);
  size_t weight_data_size = 0U;
  ge::ReadonlyByteBuffer weight_data(nullptr, ge::ConditionalDeleter{false});
  if (!first_file_name.empty()) {
    const auto var_weight_path = archive.FindEntry(std::string(gert::OM2_VARIABLES_DIR) + first_file_name);
    if (!var_weight_path.empty()) {
      weight_data = archive.ExtractToMem(var_weight_path, weight_data_size);
      GE_ASSERT_NOTNULL(weight_data, "[OM2] Failed to extract %s", var_weight_path.c_str());
    } else {
      GELOGW("[OM2] Optional file [%s] not found in ZIP archive.", first_file_name.c_str());
    }
  }

  for (const auto &[key, val] : entries_json.items()) {
    (void)key;
    RTVarEntry var_entry;
    GE_ASSERT_SUCCESS(ParseVarEntryFromJson(ge::JsonFile(val), weight_data.get(), weight_data_size, var_entry));
    GE_ASSERT_SUCCESS(RTVarAddEntry(unit.variables_config->entries, std::move(var_entry)));
  }
  return ge::SUCCESS;
}

ge::Status DeserializeKernels(const gert::ZipArchiveReader &archive, const std::string &dir_prefix,
                              GertModelData &model_data) {
  for (const auto &entry : archive.ListFilesByRelativePrefix(dir_prefix)) {
    if (!EndsWith(entry, ".o")) {
      continue;
    }
    GertModelDataFile kernel_binary;
    kernel_binary.file_name = gert::GertMakeStr(ExtractParentDirAndFileName(entry).second);
    size_t buffer_size{0U};
    auto buffer = archive.ExtractToMem(entry, buffer_size);
    GE_ASSERT_NOTNULL(buffer, "[OM2] Failed to extract entry %s", entry.c_str());
    GE_ASSERT_TRUE(buffer_size > 0U, "[OM2] Empty archive entry %s", entry.c_str());
    kernel_binary.data = std::move(buffer);
    kernel_binary.data_size = buffer_size;
    (void)model_data.kernels->binaries.emplace_back(std::make_unique<GertModelDataFile>(std::move(kernel_binary)));
  }
  return ge::SUCCESS;
}

ge::Status DeserializeCustomKernels(const gert::ZipArchiveReader &archive, const std::string &dir_prefix,
                                    GertModelData &model_data) {
  for (const auto &entry : archive.ListFilesByRelativePrefix(dir_prefix)) {
    if (!EndsWith(entry, ".bin")) {
      continue;
    }
    GertModelDataFile kernel_binary;
    kernel_binary.file_name = gert::GertMakeStr(ExtractParentDirAndFileName(entry).second);
    size_t buffer_size{0U};
    auto buffer = archive.ExtractToMem(entry, buffer_size);
    GE_ASSERT_NOTNULL(buffer, "[OM2] Failed to extract entry %s", entry.c_str());
    GE_ASSERT_TRUE(buffer_size > 0U, "[OM2] Empty archive entry %s", entry.c_str());
    kernel_binary.data = std::move(buffer);
    kernel_binary.data_size = buffer_size;
    (void)model_data.custom_ops->binaries.emplace_back(std::make_unique<GertModelDataFile>(std::move(kernel_binary)));
  }
  return ge::SUCCESS;
}

ge::Status DeserializeCustomSharedLibs(const gert::ZipArchiveReader &archive, const std::string &dir_prefix,
                                       GertModelData &model_data) {
  for (const auto &entry : archive.ListFilesByRelativePrefix(dir_prefix)) {
    if (!EndsWith(entry, ".so")) {
      continue;
    }
    GertModelDataFile kernel_binary;
    kernel_binary.file_name = gert::GertMakeStr(ExtractParentDirAndFileName(entry).second);
    size_t buffer_size{0U};
    auto buffer = archive.ExtractToMem(entry, buffer_size);
    GE_ASSERT_NOTNULL(buffer, "[OM2] Failed to extract entry %s", entry.c_str());
    GE_ASSERT_TRUE(buffer_size > 0U, "[OM2] Empty archive entry %s", entry.c_str());
    kernel_binary.data = std::move(buffer);
    kernel_binary.data_size = buffer_size;
    (void)model_data.custom_ops->libraries.emplace_back(std::make_unique<GertModelDataFile>(std::move(kernel_binary)));
  }
  return ge::SUCCESS;
}

ge::Status DeserializeVisualJson(const gert::ZipArchiveReader &archive, const std::string &relative_path,
                                 gert::GertModelDataModel &unit) {
  const auto visual_path = archive.FindEntry(relative_path);
  if (visual_path.empty()) {
    return ge::SUCCESS;
  }
  size_t buffer_size{0U};
  const auto buffer = archive.ExtractToMem(visual_path, buffer_size);
  GE_ASSERT_NOTNULL(buffer, "[OM2] Failed to extract entry %s", visual_path.c_str());
  GE_ASSERT_TRUE(buffer_size > 0U, "[OM2] Empty archive entry %s", visual_path.c_str());
  unit.debug->visual_json = gert::GertMakeStr(std::string(reinterpret_cast<const char *>(buffer.get()), buffer_size));
  return ge::SUCCESS;
}

// ===== 7. 主流程与公共入口 =====

// 公共前置：解析 manifest 校验 model_index，并在 models[model_index] 放置模型单元及各子结构
// （下标与 data/model_N 对应；入口统一分配目录聚合结构，多次反序列化仅首次分配）
ge::Status PrepareModelUnit(gert::ZipArchiveReader &archive, GertModelData &model_data, const uint32_t model_index) {
  model_data.manifest = std::make_unique<GertModelDataManifest>();
  gert::InitGertModelData(model_data);

  GE_ASSERT_SUCCESS(DeserializeManifest(archive, gert::OM2_MANIFEST_PATH, model_data));
  const uint32_t model_num = static_cast<uint32_t>(model_data.manifest->model_num);
  GE_ASSERT_TRUE(model_num >= 1U, "[OM2] manifest model_num must be at least 1, got %llu.",
                 static_cast<unsigned long long>(model_data.manifest->model_num));
  GE_ASSERT_TRUE(model_index < model_num, "[OM2] model_index %u is out of range, model_num %u.", model_index,
                 model_num);
  if (model_data.models.size() <= model_index) {
    model_data.models.resize(model_index + 1U);
  }
  model_data.models[model_index] = std::make_unique<gert::GertModelDataModel>();
  auto &unit = *model_data.models[model_index];
  unit.model_meta = std::make_unique<gert::GertModelDataModelMeta>();
  unit.runtime = std::make_unique<gert::GertModelDataRuntime>();
  unit.constants_config = std::make_unique<gert::GertModelDataConstantsConfig>();
  unit.debug = std::make_unique<gert::GertModelDataDebug>();
  return ge::SUCCESS;
}

// 全量反序列化（除 visual json）：manifest + data/model_<index>/ 全部类别 + kernels/custom_ops
ge::Status DeserializeGertModelDataFromArchive(gert::ZipArchiveReader &archive, GertModelData &model_data,
                                               const uint32_t model_index) {
  GE_ASSERT_SUCCESS(PrepareModelUnit(archive, model_data, model_index));
  auto &unit = *model_data.models[model_index];
  const auto idx = std::to_string(model_index);
  if (model_data.constants->constants_data.size() <= model_index) {
    model_data.constants->constants_data.resize(model_index + 1U);
  }
  auto &weight_slot = model_data.constants->constants_data[model_index];
  GE_ASSERT_SUCCESS(
      DeserializeModelMeta(archive, gert::FormatOm2Path(gert::OM2_MODEL_META_PATH_FORMAT, idx.c_str()), unit));
  GE_ASSERT_SUCCESS(DeserializeCodegen(archive, gert::FormatOm2Path(gert::OM2_RUNTIME_DIR_FORMAT, idx.c_str()), unit));
  GE_ASSERT_SUCCESS(DeserializeConstantsConfig(
      archive, gert::FormatOm2Path(gert::OM2_CONSTANTS_CONFIG_PATH_FORMAT, idx.c_str()), unit));
  GE_ASSERT_SUCCESS(DeserializeWeight(
      archive, std::string(gert::OM2_CONSTANTS_DIR) + gert::OM2_CONSTANTS_FILE_PREFIX + idx, unit, weight_slot));
  GE_ASSERT_SUCCESS(DeserializeOpAttr(archive, gert::FormatOm2Path(gert::OM2_OP_ATTR_PATH_FORMAT, idx.c_str()), unit));
  GE_ASSERT_SUCCESS(DeserializeVariablesData(
      archive, gert::FormatOm2Path(gert::OM2_VARIABLES_CONFIG_PATH_FORMAT, idx.c_str()), unit));
  GE_ASSERT_SUCCESS(DeserializeKernels(archive, gert::OM2_KERNELS_DIR, model_data));
  GE_ASSERT_SUCCESS(DeserializeCustomKernels(archive, "data/custom_ops/binaries_", model_data));
  GE_ASSERT_SUCCESS(DeserializeCustomSharedLibs(archive, "data/custom_ops/shared_libs", model_data));
  return ge::SUCCESS;
}

// 仅反序列化 data/model_<index>/model_meta.json
ge::Status DeserializeGertModelMetaFromArchive(gert::ZipArchiveReader &archive, GertModelData &model_data,
                                               const uint32_t model_index) {
  GE_ASSERT_SUCCESS(PrepareModelUnit(archive, model_data, model_index));
  auto &unit = *model_data.models[model_index];
  GE_ASSERT_SUCCESS(DeserializeModelMeta(
      archive, gert::FormatOm2Path(gert::OM2_MODEL_META_PATH_FORMAT, std::to_string(model_index).c_str()), unit));
  return ge::SUCCESS;
}

// 仅反序列化 data/model_<index>/constants_config.json
ge::Status DeserializeGertConstantsConfigFromArchive(gert::ZipArchiveReader &archive, GertModelData &model_data,
                                                     const uint32_t model_index) {
  GE_ASSERT_SUCCESS(PrepareModelUnit(archive, model_data, model_index));
  auto &unit = *model_data.models[model_index];
  const auto idx = std::to_string(model_index);
  GE_ASSERT_SUCCESS(DeserializeConstantsConfig(
      archive, gert::FormatOm2Path(gert::OM2_CONSTANTS_CONFIG_PATH_FORMAT, idx.c_str()), unit));
  return ge::SUCCESS;
}

// 仅反序列化 data/model_<index>/debug/ge_visual_*.json 原文
ge::Status DeserializeGertVisualJsonFromArchive(gert::ZipArchiveReader &archive, GertModelData &model_data,
                                                const uint32_t model_index) {
  GE_ASSERT_SUCCESS(PrepareModelUnit(archive, model_data, model_index));
  auto &unit = *model_data.models[model_index];
  GE_ASSERT_SUCCESS(DeserializeVisualJson(
      archive, gert::FormatOm2Path(gert::OM2_VISUAL_JSON_PATH_FORMAT, std::to_string(model_index).c_str()), unit));
  return ge::SUCCESS;
}

// 公共入口前置校验 + 打开归档，执行指定归档级流程
uint32_t RunDeserializeEntry(const uint8_t *data, const uint64_t data_size, GertModelData *model_data,
                             const uint32_t model_index,
                             ge::Status (*from_archive)(gert::ZipArchiveReader &, GertModelData &, uint32_t)) {
  if (data == nullptr || model_data == nullptr) {
    return static_cast<uint32_t>(ge::FAILED);
  }
  if (model_data->struct_size != sizeof(GertModelData)) {
    GELOGE(ge::FAILED, "[OM2] GertModelData struct_size mismatch: caller=%zu, so=%zu", model_data->struct_size,
           sizeof(GertModelData));
    return static_cast<uint32_t>(ge::FAILED);
  }
  gert::ZipArchiveReader archive(data, data_size);
  if (!archive.IsGood()) {
    GELOGE(ACL_ERROR_GE_PARAM_INVALID, "[OM2] Failed to open OM2 ZIP archive for deserialization");
    return static_cast<uint32_t>(ACL_ERROR_GE_PARAM_INVALID);
  }
  return static_cast<uint32_t>(from_archive(archive, *model_data, model_index));
}

}  // namespace

uint32_t DeserializeGertModelData(const uint8_t *data, uint64_t data_size, GertModelData *model_data,
                                  const uint32_t model_index) {
  return RunDeserializeEntry(data, data_size, model_data, model_index, DeserializeGertModelDataFromArchive);
}

uint32_t DeserializeGertModelMeta(const uint8_t *data, uint64_t data_size, GertModelData *model_data,
                                  const uint32_t model_index) {
  return RunDeserializeEntry(data, data_size, model_data, model_index, DeserializeGertModelMetaFromArchive);
}

uint32_t DeserializeGertConstantsConfig(const uint8_t *data, uint64_t data_size, GertModelData *model_data,
                                        const uint32_t model_index) {
  return RunDeserializeEntry(data, data_size, model_data, model_index, DeserializeGertConstantsConfigFromArchive);
}

uint32_t DeserializeGertVisualJson(const uint8_t *data, uint64_t data_size, GertModelData *model_data,
                                   const uint32_t model_index) {
  return RunDeserializeEntry(data, data_size, model_data, model_index, DeserializeGertVisualJsonFromArchive);
}

}  // namespace gert
