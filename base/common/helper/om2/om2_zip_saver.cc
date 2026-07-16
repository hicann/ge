/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "common/helper/om2/om2_zip_saver.h"

#include <map>

#include "common/ge_common/ge_types.h"
#include "common/helper/om2/json_file.h"
#include "common/helper/om2/om2_package_contants.h"
#include "common/helper/om2/zip_archive_writer.h"
#include "common/om2/om2_model_data.h"
#include "graph/utils/type_utils.h"
#include "nlohmann/json.hpp"

namespace ge {

namespace {

bool EndsWith(const std::string &str, const std::string &suffix) {
  return (str.size() >= suffix.size()) && (str.compare(str.size() - suffix.size(), suffix.size(), suffix) == 0);
}

std::string SerializeOpAttrMapToJson(const std::map<std::string, std::map<std::string, std::string>> &op_attr_map) {
  nlohmann::json json_obj = nlohmann::json::object();

  for (const auto &[op_name, attrs] : op_attr_map) {
    nlohmann::json op_attrs = nlohmann::json::object();
    for (const auto &[attr_name, value] : attrs) {
      op_attrs[attr_name] = value;
    }
    json_obj[op_name] = op_attrs;
  }

  return json_obj.dump();
}

Status SerializeCodegenArtifacts(const gert::Om2ModelData &model_data,
                                 const std::shared_ptr<ZipArchiveWriter> &zip_writer) {
  const size_t model_index = 0UL;
  const std::string artifacts_base_dir = FormatOm2Path(OM2_RUNTIME_DIR_FORMAT, std::to_string(model_index).c_str());
  for (const auto &artifact : model_data.program_body.source_artifacts) {
    if (EndsWith(artifact.file_name, ".so")) {
      continue;
    }
    const std::string entry_name = artifacts_base_dir + artifact.file_name;
    GE_ASSERT_TRUE(zip_writer->WriteBytes(entry_name, artifact.data.data(), artifact.data.size(), true),
                   "Failed to write artifact [%s]", artifact.file_name.c_str());
  }
  if (!model_data.program_body.so_artifact.data.empty() && !model_data.program_body.so_artifact.file_name.empty()) {
    const std::string so_entry = artifacts_base_dir + model_data.program_body.so_artifact.file_name;
    GE_ASSERT_TRUE(zip_writer->WriteBytes(so_entry, model_data.program_body.so_artifact.data.data(),
                                          model_data.program_body.so_artifact.data.size(), false),
                   "Failed to write so artifact [%s]", model_data.program_body.so_artifact.file_name.c_str());
  }
  return SUCCESS;
}

Status SerializeWeightData(const gert::Om2ModelData &model_data, const std::shared_ptr<ZipArchiveWriter> &zip_writer) {
  const bool has_internal_const = (model_data.constants_data.internal_weight_size > 0U);
  if (!has_internal_const || model_data.constants_data.weight_data.empty()) {
    return SUCCESS;
  }
  const size_t model_index = 0UL;
  const auto constant_file_name = FormatOm2Path("%s%s%zu", OM2_CONSTANTS_DIR, OM2_CONSTANTS_FILE_PREFIX, model_index);
  GE_ASSERT_TRUE(zip_writer->WriteBytes(constant_file_name, model_data.constants_data.weight_data.data(),
                                        model_data.constants_data.weight_data.size(), false));
  return SUCCESS;
}

Status SerializeConstantsConfig(const gert::Om2ModelData &model_data,
                                const std::shared_ptr<ZipArchiveWriter> &zip_writer, const bool is_offline) {
  const size_t model_index = 0UL;
  JsonFile json_file;
  (void)json_file.Set("internal_weight_size", model_data.constants_data.internal_weight_size);
  auto const_json_object = JsonFile::json::object();
  for (const auto &const_meta : model_data.constants_data.consts) {
    std::string const_key = const_meta.op_name.empty() ? const_meta.file_name : const_meta.op_name;
    if (const_key.empty()) {
      const_key = "constant_" + std::to_string(const_meta.index);
    }
    JsonFile const_info;
    (void)const_info.Set("index", const_meta.index);
    (void)const_info.Set("type", const_meta.type);
    (void)const_info.Set("file_name", const_meta.file_name);
    if (!is_offline && const_meta.type != "INTERNAL" && !const_meta.file_path.empty()) {
      (void)const_info.Set("file_path", const_meta.file_path);
    }
    (void)const_info.Set("offset", const_meta.offset);
    (void)const_info.Set("size", const_meta.size);
    (void)const_info.Set("op_name", const_meta.op_name);
    const_json_object[const_key] = const_info.Raw();
  }
  (void)json_file.Set("consts", const_json_object);
  const std::string constants_json_str = json_file.Dump();
  const auto constants_config_path =
      FormatOm2Path(OM2_CONSTANTS_CONFIG_PATH_FORMAT, std::to_string(model_index).c_str());
  GE_ASSERT_TRUE(
      zip_writer->WriteBytes(constants_config_path, constants_json_str.data(), constants_json_str.size(), false));
  return SUCCESS;
}

Status SerializeKernelBinaries(const gert::Om2ModelData &model_data,
                               const std::shared_ptr<ZipArchiveWriter> &zip_writer) {
  const auto kernel_bin_dir = FormatOm2Path(OM2_KERNELS_DIR_FORMAT, "npu_arch");
  for (const auto &kb : model_data.kernel_binaries) {
    const auto entry_path = kernel_bin_dir + kb.name;
    GE_ASSERT_TRUE(zip_writer->WriteBytes(entry_path, kb.data.data(), kb.data.size(), false));
  }
  return SUCCESS;
}

Status SerializeModelMeta(const gert::Om2ModelData &model_data, const std::shared_ptr<ZipArchiveWriter> &zip_writer) {
  const size_t model_index = 0UL;
  JsonFile model_meta_info;
  auto input_json_array = JsonFile::json::array();
  for (size_t i = 0UL; i < model_data.model_meta.input_desc.size(); ++i) {
    const auto &desc = model_data.model_meta.input_desc[i];
    const auto &desc_v2 =
        (i < model_data.model_meta.input_desc_v2.size()) ? model_data.model_meta.input_desc_v2[i] : desc;
    JsonFile input_info;
    (void)input_info.Set("name", desc.GetName());
    (void)input_info.Set("index", i);
    (void)input_info.Set("shape", desc.GetShape());
    (void)input_info.Set("shape_v2", desc_v2.GetShape());
    (void)input_info.Set("data_type", TypeUtils::DataTypeToSerialString(desc.GetDataType()));
    (void)input_info.Set("format", TypeUtils::FormatToSerialString(desc.GetFormat()));
    (void)input_info.Set("size", desc.GetSize());
    (void)input_info.Set("shape_range", desc.GetShapeRange());
    const auto origin_dims = (i < model_data.model_meta.origin_input_dims.size())
                                 ? model_data.model_meta.origin_input_dims[i]
                                 : desc.GetShape();
    (void)input_info.Set("origin_input_dims", origin_dims);
    input_json_array.push_back(input_info.Raw());
  }

  auto output_json_array = JsonFile::json::array();
  for (size_t i = 0UL; i < model_data.model_meta.output_desc.size(); ++i) {
    const auto &desc = model_data.model_meta.output_desc[i];
    JsonFile output_info;
    (void)output_info.Set("name", desc.GetName());
    (void)output_info.Set("index", i);
    (void)output_info.Set("shape", desc.GetShape());
    (void)output_info.Set("data_type", TypeUtils::DataTypeToSerialString(desc.GetDataType()));
    (void)output_info.Set("format", TypeUtils::FormatToSerialString(desc.GetFormat()));
    (void)output_info.Set("size", desc.GetSize());
    (void)output_info.Set("shape_range", desc.GetShapeRange());
    output_json_array.push_back(output_info.Raw());
  }

  (void)model_meta_info.Set("inputs", input_json_array);
  (void)model_meta_info.Set("outputs", output_json_array);
  (void)model_meta_info.Set("dynamic_output_shape", model_data.model_meta.dynamic_output_shape);
  (void)model_meta_info.Set("dynamic_batch_info", model_data.model_meta.dynamic_batch_info);
  (void)model_meta_info.Set("user_designate_shape_order", model_data.model_meta.user_designate_shape_order);
  (void)model_meta_info.Set("dynamic_type", model_data.model_meta.dynamic_type);
  (void)model_meta_info.Set("work_size", model_data.model_meta.work_size);
  (void)model_meta_info.Set("zero_copy_size", model_data.model_meta.zero_copy_size);
  (void)model_meta_info.Set("name", model_data.model_meta.model_name);
  (void)model_meta_info.Set("root_graph_name", model_data.model_meta.root_graph_name);

  const auto model_meta_info_str = model_meta_info.Dump();
  const auto model_meta_entry_path = FormatOm2Path(OM2_MODEL_META_PATH_FORMAT, std::to_string(model_index).c_str());
  GE_ASSERT_TRUE(
      zip_writer->WriteBytes(model_meta_entry_path, model_meta_info_str.data(), model_meta_info_str.size(), false));
  return SUCCESS;
}

Status SerializeDebugInfo(const gert::Om2ModelData &model_data, const std::shared_ptr<ZipArchiveWriter> &zip_writer) {
  const size_t model_index = 0UL;
  // op_attr.json
  const auto op_attr_json_str = model_data.debug_info.op_attr_map.empty()
                                    ? std::string("{}")
                                    : SerializeOpAttrMapToJson(model_data.debug_info.op_attr_map);
  const auto op_attr_entry_path = FormatOm2Path(OM2_OP_ATTR_PATH_FORMAT, std::to_string(model_index).c_str());
  GE_ASSERT_TRUE(zip_writer->WriteBytes(op_attr_entry_path, op_attr_json_str.data(), op_attr_json_str.size(), false));

  // visual json
  const auto visual_entry_path = FormatOm2Path(OM2_VISUAL_JSON_PATH_FORMAT, std::to_string(model_index).c_str());
  GE_ASSERT_TRUE(zip_writer->WriteBytes(visual_entry_path, model_data.debug_info.visual_json.data(),
                                        model_data.debug_info.visual_json.size(), true));
  return SUCCESS;
}

Status SerializeManifest(const gert::Om2ModelData &model_data, const std::shared_ptr<ZipArchiveWriter> &zip_writer) {
  nlohmann::json manifest_json = nlohmann::json::object();
  for (const auto &[key, value] : model_data.manifest) {
    manifest_json[key] = value;
  }
  const auto manifest_str = manifest_json.dump();
  GE_ASSERT_TRUE(zip_writer->WriteBytes(OM2_MANIFEST_PATH, manifest_str.data(), manifest_str.size(), false));
  return SUCCESS;
}

}  // namespace

Status Om2ZipSaver::Save(const gert::Om2ModelData &model_data, ModelBufferData &model, const bool is_offline,
                         const std::string &writer_path) {
  GELOGI(
      "[OM2] Begin to serialize Om2ModelData to ZIP, model_name:%s, root_graph:%s, "
      "inputs:%zu, outputs:%zu, kernels:%zu, weight_size:%zu",
      model_data.model_meta.model_name.c_str(), model_data.model_meta.root_graph_name.c_str(),
      model_data.model_meta.input_desc.size(), model_data.model_meta.output_desc.size(),
      model_data.kernel_binaries.size(), model_data.constants_data.weight_data.size());
  const std::string path = writer_path.empty() ? "om2_model" : writer_path;
  auto zip_writer = std::make_shared<ZipArchiveWriter>(path);
  GE_ASSERT_NOTNULL(zip_writer);
  GE_ASSERT_TRUE(zip_writer->IsMemFileOpened());

  GE_ASSERT_SUCCESS(SerializeCodegenArtifacts(model_data, zip_writer));
  GE_ASSERT_SUCCESS(SerializeWeightData(model_data, zip_writer));
  GE_ASSERT_SUCCESS(SerializeConstantsConfig(model_data, zip_writer, is_offline));
  GE_ASSERT_SUCCESS(SerializeKernelBinaries(model_data, zip_writer));
  GE_ASSERT_SUCCESS(SerializeModelMeta(model_data, zip_writer));
  GE_ASSERT_SUCCESS(SerializeDebugInfo(model_data, zip_writer));
  GE_ASSERT_SUCCESS(SerializeManifest(model_data, zip_writer));

  GE_ASSERT_TRUE(zip_writer->SaveModelData(model, is_offline));
  GELOGI(
      "[OM2] Successfully serialized Om2ModelData to ZIP, model_name:%s, "
      "buffer_size:%zu, is_offline:%d",
      model_data.model_meta.model_name.c_str(), model.length, is_offline);
  return SUCCESS;
}

}  // namespace ge
