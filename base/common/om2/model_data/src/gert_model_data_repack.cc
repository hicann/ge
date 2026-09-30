/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "framework/common/gert_model_data_repack.h"

#include <map>
#include <string>
#include <vector>

#include "common/checker.h"
#include "common/ge_common/ge_types.h"
#include "common/ge_common/string_util.h"
#include "common/util/mem_utils.h"
#include "framework/common/json_file.h"
#include "framework/common/zip_archive_reader.h"
#include "framework/common/zip_archive_writer.h"
#include "framework/om2/model_data/gert_model_data.h"
#include "mmpa/mmpa_api.h"

namespace gert {
namespace {
const std::string kOm2ConstantsConfigSuffix = "constants_config.json";
const std::string kOm2ExternalWeightDirName = "weight";

bool EndsWith(const std::string &str, const std::string &suffix) {
  return (str.size() >= suffix.size()) && (str.compare(str.size() - suffix.size(), suffix.size(), suffix) == 0);
}

std::string StripOm2ArchiveRoot(const std::string &entry_name) {
  const auto pos = entry_name.find('/');
  return (pos == std::string::npos) ? entry_name : entry_name.substr(pos + 1U);
}

bool IsOm2ConstantsConfigEntry(const std::string &entry_name) {
  return (entry_name.find("data/model_") == 0U) && EndsWith(entry_name, kOm2ConstantsConfigSuffix);
}

bool ShouldCompressRepackedOm2Entry(const std::string &entry_name) {
  const std::string model_dir_prefix = "data/model_";
  const std::string runtime_dir = "/runtime/";
  if (entry_name.find(model_dir_prefix) != 0U) {
    return false;
  }
  const auto model_index_start = model_dir_prefix.length();
  const auto runtime_pos = entry_name.find(runtime_dir, model_index_start);
  return (runtime_pos != std::string::npos) && (runtime_pos > model_index_start);
}

std::string MakeOm2ExternalWeightPath(const std::string &output_file_name, const std::string &file_name) {
  std::string path = output_file_name;
  const char *const om_dir = mmDirName(&path[0]);
  if (om_dir == nullptr) {
    return "";
  }
  return std::string(om_dir) + "/" + kOm2ExternalWeightDirName + "/" + file_name;
}

ge::Status RewriteOm2ConstantsConfig(const std::string &output_file_name, ge::JsonFile &constants_json,
                                     std::map<std::string, std::string> &old_file_to_new_file, bool &changed) {
  ge::JsonFile::json consts_json;
  if (!constants_json.Get("consts", consts_json) || !consts_json.is_object()) {
    return ge::SUCCESS;
  }
  for (auto &const_item : consts_json.items()) {
    auto &const_info = const_item.value();
    if (!const_info.is_object()) {
      continue;
    }
    ge::JsonFile const_info_json(const_info);
    std::string type;
    if (const_info_json.Get("type", type) && (type == "INTERNAL")) {
      continue;
    }
    std::string old_file_path;
    if (!const_info_json.Get("file_path", old_file_path) || old_file_path.empty()) {
      continue;
    }
    std::string file_name;
    if (!const_info_json.Get("file_name", file_name) || file_name.empty()) {
      file_name = ge::StringUtils::GetFileName(old_file_path);
    }
    GE_ASSERT_TRUE(!file_name.empty(), "[OM2] External weight file name is empty, file_path=%s", old_file_path.c_str());
    const std::string new_file_path = MakeOm2ExternalWeightPath(output_file_name, file_name);
    GE_ASSERT_TRUE(!new_file_path.empty(), "[OM2] Failed to make external weight path, output=%s",
                   output_file_name.c_str());
    const_info["file_name"] = file_name;
    (void)const_info.erase("file_path");
    old_file_to_new_file[old_file_path] = new_file_path;
    changed = true;
  }
  if (changed) {
    (void)constants_json.Set("consts", consts_json);
  }
  return ge::SUCCESS;
}

// 收集需要外置权重搬迁的 constants config 改写结果
ge::Status CollectExternalWeightRelocation(const std::string &output_file_name, const ZipArchiveReader &archive,
                                           const std::vector<std::string> &archive_entries,
                                           std::map<std::string, std::string> &rewritten_configs,
                                           std::map<std::string, std::string> &old_file_to_new_file) {
  for (const auto &entry_name : archive_entries) {
    const std::string relative_entry_name = StripOm2ArchiveRoot(entry_name);
    if (!IsOm2ConstantsConfigEntry(relative_entry_name)) {
      continue;
    }
    size_t buffer_size = 0U;
    const auto buffer = archive.ExtractToMem(entry_name, buffer_size);
    GE_ASSERT_NOTNULL(buffer, "[OM2] Failed to extract constants config entry %s", entry_name.c_str());
    const ge::JsonFile const_json_readonly(reinterpret_cast<const uint8_t *>(buffer.get()), buffer_size);
    GE_ASSERT_TRUE(const_json_readonly.IsValid(), "[OM2] Invalid constants config entry %s", entry_name.c_str());
    ge::JsonFile const_json(const_json_readonly.Raw());
    bool changed = false;
    GE_ASSERT_SUCCESS(RewriteOm2ConstantsConfig(output_file_name, const_json, old_file_to_new_file, changed));
    if (changed) {
      rewritten_configs[entry_name] = const_json.Dump();
    }
  }
  return ge::SUCCESS;
}

// 按改写结果重打包归档条目
ge::Status RepackArchiveEntries(const std::string &output_file_name, const ZipArchiveReader &archive,
                                const std::vector<std::string> &archive_entries,
                                const std::map<std::string, std::string> &rewritten_configs,
                                GertBuffer &relocated_model) {
  auto zip_writer = ge::MakeShared<ZipArchiveWriter>(output_file_name);
  GE_ASSERT_NOTNULL(zip_writer);
  GE_ASSERT_TRUE(zip_writer->IsMemFileOpened());
  for (const auto &entry_name : archive_entries) {
    const std::string relative_entry_name = StripOm2ArchiveRoot(entry_name);
    const auto rewritten_config = rewritten_configs.find(entry_name);
    if (rewritten_config != rewritten_configs.end()) {
      GE_ASSERT_TRUE(zip_writer->WriteBytes(relative_entry_name, rewritten_config->second.data(),
                                            rewritten_config->second.size(),
                                            ShouldCompressRepackedOm2Entry(relative_entry_name)));
      continue;
    }
    size_t buffer_size = 0U;
    const auto buffer = archive.ExtractToMem(entry_name, buffer_size);
    GE_ASSERT_NOTNULL(buffer, "[OM2] Failed to extract archive entry %s", entry_name.c_str());
    GE_ASSERT_TRUE(buffer_size > 0U, "[OM2] Empty archive entry %s is invalid", entry_name.c_str());
    GE_ASSERT_TRUE(zip_writer->WriteBytes(relative_entry_name, buffer.get(), buffer_size,
                                          ShouldCompressRepackedOm2Entry(relative_entry_name)));
  }
  GE_ASSERT_TRUE(zip_writer->SaveModelData(relocated_model, false));
  return ge::SUCCESS;
}
}  // namespace

ge::Status RepackOm2ModelData(const std::string &output_file_name, const uint8_t *model_data, uint64_t model_len,
                              GertBuffer &relocated_model, std::map<std::string, std::string> &old_file_to_new_file) {
  ZipArchiveReader archive(model_data, model_len);
  if (!archive.IsGood()) {
    GELOGW("[OM2] Model buffer has zip magic but is not a valid zip archive, skip repack.");
    return ge::SUCCESS;
  }
  const auto archive_entries = archive.ListFiles();

  std::map<std::string, std::string> rewritten_configs;
  GE_ASSERT_SUCCESS(CollectExternalWeightRelocation(output_file_name, archive, archive_entries, rewritten_configs,
                                                    old_file_to_new_file));
  if (old_file_to_new_file.empty()) {
    return ge::SUCCESS;
  }
  GE_ASSERT_SUCCESS(
      RepackArchiveEntries(output_file_name, archive, archive_entries, rewritten_configs, relocated_model));
  return ge::SUCCESS;
}

}  // namespace gert
