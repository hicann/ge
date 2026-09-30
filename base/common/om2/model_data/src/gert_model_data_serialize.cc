/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "framework/common/gert_model_data_serialize.h"

#include <map>
#include <sstream>

#include "common/ge_common/ge_types.h"
#include "common/checker.h"
#include "framework/common/json_file.h"
#include "framework/om2/model_data/om2_package_contants.h"
#include "framework/common/zip_archive_writer.h"
#include "framework/om2/model_data/gert_model_data.h"
#include "framework/common/gert_model_data_utils.h"
#include "graph/utils/type_utils.h"
#include "nlohmann/json.hpp"

namespace gert {

namespace {

bool EndsWith(const std::string &str, const std::string &suffix) {
  return (str.size() >= suffix.size()) && (str.compare(str.size() - suffix.size(), suffix.size(), suffix) == 0);
}

ge::JsonFile SerializeTensorDesc(const gert::GertTensorDesc &desc) {
  ge::JsonFile json;
  (void)json.Set("name", gert::GertGetStr(desc.name));
  (void)json.Set("shape", desc.shape);
  (void)json.Set("data_type", ge::TypeUtils::DataTypeToSerialString(desc.data_type));
  (void)json.Set("format", ge::TypeUtils::FormatToSerialString(desc.format));
  (void)json.Set("size", desc.size);
  (void)json.Set("shape_range", desc.shape_range);
  return json;
}

ge::Status SerializeCodegenArtifacts(const gert::GertModelDataModel &unit,
                                     const std::shared_ptr<ZipArchiveWriter> &zip_writer, const size_t model_index) {
  if (unit.runtime == nullptr) {
    return ge::SUCCESS;
  }
  const std::string runtime_dir = FormatOm2Path(OM2_RUNTIME_DIR_FORMAT, std::to_string(model_index).c_str());
  const std::string csrc_dir = FormatOm2Path(OM2_RUNTIME_CSRC_DIR_FORMAT, std::to_string(model_index).c_str());
  for (const auto &artifact : unit.runtime->source_artifacts) {
    const std::string file_name(gert::GertGetStr(artifact.file_name));
    if (EndsWith(file_name, ".so")) {
      continue;
    }
    const std::string entry_name = csrc_dir + file_name;
    GE_ASSERT_TRUE(zip_writer->WriteBytes(entry_name, gert::GertGetStr(artifact.data), artifact.data_len, true),
                   "Failed to write artifact [%s]", file_name.c_str());
  }
  const auto &so_artifact = unit.runtime->so_artifact;
  if ((so_artifact.data != nullptr) && (gert::GertGetStr(so_artifact.file_name)[0] != '\0')) {
    const std::string so_entry = runtime_dir + gert::GertGetStr(so_artifact.file_name);
    GE_ASSERT_TRUE(zip_writer->WriteBytes(so_entry, gert::GertGetStr(so_artifact.data), so_artifact.data_len, false),
                   "Failed to write so artifact [%s]", gert::GertGetStr(so_artifact.file_name));
  }
  return ge::SUCCESS;
}

ge::Status SerializeWeightData(const gert::GertModelDataModel &unit, const gert::GertModelDataConstantsData &weight,
                               const std::shared_ptr<ZipArchiveWriter> &zip_writer, const size_t model_index) {
  const bool has_internal_const =
      (unit.constants_config != nullptr) && (unit.constants_config->internal_weight_size > 0U);
  if (!has_internal_const || weight.data == nullptr) {
    return ge::SUCCESS;
  }
  const auto constant_file_name = FormatOm2Path("%s%s%zu", OM2_CONSTANTS_DIR, OM2_CONSTANTS_FILE_PREFIX, model_index);
  GE_ASSERT_TRUE(zip_writer->WriteBytes(constant_file_name, weight.data.get(), weight.size, false));
  return ge::SUCCESS;
}

ge::Status SerializeConstantsConfig(const gert::GertModelDataModel &unit,
                                    const std::shared_ptr<ZipArchiveWriter> &zip_writer, const bool is_offline,
                                    const size_t model_index) {
  if (unit.constants_config == nullptr) {
    return ge::SUCCESS;
  }
  ge::JsonFile json_file;
  (void)json_file.Set("internal_weight_size", unit.constants_config->internal_weight_size);
  auto const_json_object = ge::JsonFile::json::object();
  for (const auto &const_meta_ptr : unit.constants_config->consts) {
    const auto &const_meta = *const_meta_ptr;
    const std::string type(gert::GertGetStr(const_meta.type));
    const std::string op_name(gert::GertGetStr(const_meta.op_name));
    const std::string file_name(gert::GertGetStr(const_meta.file_name));
    const std::string file_path(gert::GertGetStr(const_meta.file_path));
    const std::string const_key = (type == "INTERNAL") ? "constant_" + std::to_string(const_meta.index) : op_name;
    ge::JsonFile const_info;
    (void)const_info.Set("index", const_meta.index);
    (void)const_info.Set("type", type);
    (void)const_info.Set("file_name", file_name);
    if (!is_offline && type != "INTERNAL" && !file_path.empty()) {
      (void)const_info.Set("file_path", file_path);
    }
    (void)const_info.Set("offset", const_meta.offset);
    (void)const_info.Set("size", const_meta.size);
    const_json_object[const_key] = const_info.Raw();
  }
  (void)json_file.Set("consts", const_json_object);
  const std::string constants_json_str = json_file.Dump();
  const std::string model_index_str = std::to_string(model_index);
  const auto constants_config_path =
      FormatOm2Path(OM2_CONSTANTS_CONFIG_PATH_FORMAT, model_index_str.c_str(), model_index_str.c_str());
  GE_ASSERT_TRUE(
      zip_writer->WriteBytes(constants_config_path, constants_json_str.data(), constants_json_str.size(), true));
  return ge::SUCCESS;
}

ge::JsonFile::json SerializeVarTransRoad(const gert::RTVarTransRoad &trans_road) {
  auto trans_road_json = ge::JsonFile::json::array();
  for (const auto &node : trans_road) {
    ge::JsonFile node_json;
    (void)node_json.Set("node_type", gert::GertGetStr(node.node_type));
    (void)node_json.Set("input", SerializeTensorDesc(node.input).Raw());
    (void)node_json.Set("output", SerializeTensorDesc(node.output).Raw());
    trans_road_json.push_back(node_json.Raw());
  }
  return trans_road_json;
}

ge::JsonFile::json SerializeVarEntry(const gert::RTVarEntry &entry, std::vector<uint8_t> &weight_buffer) {
  ge::JsonFile entry_json;
  (void)entry_json.Set("var_name", gert::GertGetStr(entry.var_name));
  (void)entry_json.Set("var_key", gert::GertGetStr(entry.var_key));
  (void)entry_json.Set("op_type", gert::GertGetStr(entry.op_type));
  (void)entry_json.Set("logic_addr", entry.logic_addr);
  (void)entry_json.Set("size", entry.size);
  (void)entry_json.Set("memory_type", entry.memory_type);
  (void)entry_json.Set("changed_graph_id", entry.changed_graph_id);
  (void)entry_json.Set("allocated_graph_id", entry.allocated_graph_id);
  (void)entry_json.Set("tensor_desc", SerializeTensorDesc(entry.tensor_desc).Raw());
  (void)entry_json.Set("trans_road", SerializeVarTransRoad(entry.trans_road));
  ge::JsonFile copy_info_json;
  (void)copy_info_json.Set("src_var_name", gert::GertGetStr(entry.copy_info.src_var_name));
  (void)copy_info_json.Set("src_tensor_desc", SerializeTensorDesc(entry.copy_info.src_tensor_desc).Raw());
  (void)entry_json.Set("copy_info", copy_info_json.Raw());
  size_t init_data_offset = 0U;
  size_t init_data_size = 0U;
  if (!entry.init_data.empty()) {
    init_data_offset = weight_buffer.size();
    init_data_size = entry.init_data.size();
    (void)weight_buffer.insert(weight_buffer.end(), entry.init_data.begin(), entry.init_data.end());
  }
  (void)entry_json.Set("init_data_offset", init_data_offset);
  (void)entry_json.Set("init_data_size", init_data_size);
  return entry_json.Raw();
}

// data/model_%s/variables_config.json（graph_id + var_metas + entries）+ data/model_%s/var_weight_data
ge::Status SerializeVariablesData(const gert::GertModelDataModel &unit,
                                  const std::shared_ptr<ZipArchiveWriter> &zip_writer, const size_t model_index) {
  if ((unit.variables_config == nullptr) ||
      (unit.variables_config->var_metas.empty() && unit.variables_config->entries.empty())) {
    return ge::SUCCESS;
  }
  const auto &variables_config = *unit.variables_config;
  ge::JsonFile json_file;
  (void)json_file.Set("graph_id", variables_config.graph_id);
  auto var_metas_json = ge::JsonFile::json::array();
  for (const auto &meta_ptr : variables_config.var_metas) {
    const auto &meta = *meta_ptr;
    ge::JsonFile meta_json;
    (void)meta_json.Set("index", meta.index);
    (void)meta_json.Set("var_name", gert::GertGetStr(meta.var_name));
    (void)meta_json.Set("op_type", gert::GertGetStr(meta.op_type));
    (void)meta_json.Set("op_name", gert::GertGetStr(meta.op_name));
    (void)meta_json.Set("tensor_desc", SerializeTensorDesc(meta.tensor_desc).Raw());
    var_metas_json.push_back(meta_json.Raw());
  }
  (void)json_file.Set("var_metas", var_metas_json);

  if (!variables_config.entries.empty()) {
    auto entries_json = ge::JsonFile::json::object();
    std::vector<uint8_t> weight_buffer;
    for (const auto &entry : variables_config.entries) {
      entries_json[gert::GertGetStr(entry.var_key)] = SerializeVarEntry(entry, weight_buffer);
    }
    (void)json_file.Set("entries", entries_json);
    if (!weight_buffer.empty()) {
      const auto weight_file = FormatOm2Path(OM2_VAR_WEIGHT_FILE_FORMAT, std::to_string(model_index).c_str());
      GE_ASSERT_TRUE(zip_writer->WriteBytes(weight_file, weight_buffer.data(), weight_buffer.size(), false));
    }
  }

  const std::string json_str = json_file.Dump();
  const auto config_path = FormatOm2Path(OM2_VARIABLES_CONFIG_PATH_FORMAT, std::to_string(model_index).c_str());
  GE_ASSERT_TRUE(zip_writer->WriteBytes(config_path, json_str.data(), json_str.size(), false));
  return ge::SUCCESS;
}

ge::Status SerializeKernelBinaries(const gert::GertModelData &model_data,
                                   const std::shared_ptr<ZipArchiveWriter> &zip_writer) {
  const auto kernel_bin_dir = OM2_KERNELS_DIR;
  for (const auto &kb_ptr : model_data.kernels->binaries) {
    const auto entry_path = kernel_bin_dir + std::string(gert::GertGetStr(kb_ptr->name));
    GE_ASSERT_TRUE(zip_writer->WriteBytes(entry_path, kb_ptr->data.get(), kb_ptr->data_size, false));
  }
  return ge::SUCCESS;
}

ge::Status SerializeCustomKernelBinaries(const gert::GertModelData &model_data,
                                         const std::shared_ptr<ZipArchiveWriter> &zip_writer) {
  const auto kernel_bin_dir = FormatOm2Path(OM2_CUSTOM_KERNELS_DIR_FORMAT, "binaries_npu_arch");
  for (const auto &kb_ptr : model_data.custom_ops->binaries) {
    const auto entry_path = kernel_bin_dir + std::string(gert::GertGetStr(kb_ptr->name));
    GE_ASSERT_TRUE(zip_writer->WriteBytes(entry_path, kb_ptr->data.get(), kb_ptr->data_size, false));
  }
  return ge::SUCCESS;
}

ge::Status SerializeCustomKernelSharedLibs(const gert::GertModelData &model_data,
                                           const std::shared_ptr<ZipArchiveWriter> &zip_writer) {
  const auto kernel_bin_dir = FormatOm2Path(OM2_CUSTOM_KERNELS_DIR_FORMAT, "shared_libs");
  for (const auto &kb_ptr : model_data.custom_ops->libraries) {
    const auto entry_path = kernel_bin_dir + std::string(gert::GertGetStr(kb_ptr->name));
    GE_ASSERT_TRUE(zip_writer->WriteBytes(entry_path, kb_ptr->data.get(), kb_ptr->data_size, false));
  }
  return ge::SUCCESS;
}

ge::JsonFile SerializeAippDimsToJson(const std::vector<std::unique_ptr<ge::InputOutputDims>> &dims_list,
                                     const std::string &fmt_str, const std::string &dt_str) {
  ge::JsonFile::json arr = ge::JsonFile::json::array();
  for (const auto &dims_ptr : dims_list) {
    if (dims_ptr == nullptr) {
      continue;
    }
    const auto &dims = *dims_ptr;
    std::string dim_csv;
    for (size_t d = 0U; d < dims.dims.size(); ++d) {
      if (d > 0U) {
        dim_csv += ",";
      }
      dim_csv += std::to_string(dims.dims[d]);
    }
    arr.push_back(fmt_str + ":" + dt_str + ":" + dims.name + ":" + std::to_string(dims.size) + ":" +
                  std::to_string(dims.dim_num) + ":" + dim_csv);
  }
  return ge::JsonFile(arr);
}

void SerializeAippMeta(const gert::GertModelDataModelMeta &model_meta, ge::JsonFile &model_meta_info) {
  if (model_meta.aipp_infos.empty()) {
    return;
  }
  GELOGI("[OM2] Serializing %zu AIPP entries to model_meta.json", model_meta.aipp_infos.size());
  ge::JsonFile::json aipp_infos_arr = ge::JsonFile::json::array();
  for (size_t i = 0U; i < model_meta.aipp_infos.size(); ++i) {
    const auto &meta_ptr = model_meta.aipp_infos[i];
    if (meta_ptr == nullptr || meta_ptr->aipp_type == ge::DATA_WITHOUT_AIPP) {
      continue;
    }
    const auto &meta = *meta_ptr;
    const ge::OriginInputInfo orig_input_info =
        (meta.orig_input_info != nullptr) ? *meta.orig_input_info : ge::OriginInputInfo{};
    const ge::AippConfigInfo config_info =
        (meta.aipp_config_info != nullptr) ? *meta.aipp_config_info : ge::AippConfigInfo{};
    const std::string fmt_str = ge::TypeUtils::FormatToSerialString(orig_input_info.format);
    const std::string dt_str = ge::TypeUtils::DataTypeToSerialString(orig_input_info.data_type);
    ge::JsonFile entry;
    (void)entry.Set("index", i)
        .Set("aipp_type", static_cast<int32_t>(meta.aipp_type))
        .Set("aipp_data_index", meta.aipp_data_index)
        .Set("aipp_mode", static_cast<int32_t>(config_info.aipp_mode))
        .Set("input_format", static_cast<int32_t>(config_info.input_format))
        .Set("src_image_size_w", config_info.src_image_size_w)
        .Set("src_image_size_h", config_info.src_image_size_h)
        .Set("crop", static_cast<int32_t>(config_info.crop))
        .Set("load_start_pos_w", config_info.load_start_pos_w)
        .Set("load_start_pos_h", config_info.load_start_pos_h)
        .Set("crop_size_w", config_info.crop_size_w)
        .Set("crop_size_h", config_info.crop_size_h)
        .Set("resize", static_cast<int32_t>(config_info.resize))
        .Set("resize_output_w", config_info.resize_output_w)
        .Set("resize_output_h", config_info.resize_output_h)
        .Set("padding", static_cast<int32_t>(config_info.padding))
        .Set("left_padding_size", config_info.left_padding_size)
        .Set("right_padding_size", config_info.right_padding_size)
        .Set("top_padding_size", config_info.top_padding_size)
        .Set("bottom_padding_size", config_info.bottom_padding_size)
        .Set("csc_switch", static_cast<int32_t>(config_info.csc_switch))
        .Set("rbuv_swap_switch", static_cast<int32_t>(config_info.rbuv_swap_switch))
        .Set("ax_swap_switch", static_cast<int32_t>(config_info.ax_swap_switch))
        .Set("single_line_mode", static_cast<int32_t>(config_info.single_line_mode))
        .Set("matrix_r0c0", config_info.matrix_r0c0)
        .Set("matrix_r0c1", config_info.matrix_r0c1)
        .Set("matrix_r0c2", config_info.matrix_r0c2)
        .Set("matrix_r1c0", config_info.matrix_r1c0)
        .Set("matrix_r1c1", config_info.matrix_r1c1)
        .Set("matrix_r1c2", config_info.matrix_r1c2)
        .Set("matrix_r2c0", config_info.matrix_r2c0)
        .Set("matrix_r2c1", config_info.matrix_r2c1)
        .Set("matrix_r2c2", config_info.matrix_r2c2)
        .Set("output_bias_0", config_info.output_bias_0)
        .Set("output_bias_1", config_info.output_bias_1)
        .Set("output_bias_2", config_info.output_bias_2)
        .Set("input_bias_0", config_info.input_bias_0)
        .Set("input_bias_1", config_info.input_bias_1)
        .Set("input_bias_2", config_info.input_bias_2)
        .Set("mean_chn_0", config_info.mean_chn_0)
        .Set("mean_chn_1", config_info.mean_chn_1)
        .Set("mean_chn_2", config_info.mean_chn_2)
        .Set("mean_chn_3", config_info.mean_chn_3)
        .Set("min_chn_0", config_info.min_chn_0)
        .Set("min_chn_1", config_info.min_chn_1)
        .Set("min_chn_2", config_info.min_chn_2)
        .Set("min_chn_3", config_info.min_chn_3)
        .Set("var_reci_chn_0", config_info.var_reci_chn_0)
        .Set("var_reci_chn_1", config_info.var_reci_chn_1)
        .Set("var_reci_chn_2", config_info.var_reci_chn_2)
        .Set("var_reci_chn_3", config_info.var_reci_chn_3)
        .Set("support_rotation", static_cast<int32_t>(config_info.support_rotation))
        .Set("related_input_rank", config_info.related_input_rank)
        .Set("max_src_image_size", config_info.max_src_image_size)
        .Set("aipp_inputs", SerializeAippDimsToJson(meta.aipp_input_dims, fmt_str, dt_str))
        .Set("aipp_outputs", SerializeAippDimsToJson(meta.aipp_output_dims, fmt_str, dt_str))
        .Set("orig_input_format", static_cast<int32_t>(orig_input_info.format))
        .Set("orig_input_data_type", static_cast<int32_t>(orig_input_info.data_type))
        .Set("orig_input_dim_num", orig_input_info.dim_num);
    aipp_infos_arr.push_back(entry.Raw());
  }
  ge::JsonFile aipp_json;
  (void)aipp_json.Set("aipp_infos", aipp_infos_arr);
  (void)model_meta_info.Set("aipp", aipp_json);
}

void SerializeModelInputDescs(const gert::GertModelDataModelMeta &model_meta, ge::JsonFile &model_meta_info) {
  auto input_json_array = ge::JsonFile::json::array();
  for (size_t i = 0UL; i < model_meta.input_desc.size(); ++i) {
    const auto &desc = model_meta.input_desc[i];
    ge::JsonFile input_info;
    (void)input_info.Set("name", gert::GertGetStr(desc.name));
    (void)input_info.Set("index", i);
    if (!model_meta.dynamic_batch_info.empty()) {
      (void)input_info.Set("shape", model_meta.origin_input_dims[i]);
      (void)input_info.Set("max_gear_shape", desc.shape);
    } else {
      (void)input_info.Set("shape", desc.shape);
    }
    if (model_meta.has_aipp != 0U) {
      const auto &desc_v2 = (i < model_meta.input_desc_v2.size()) ? model_meta.input_desc_v2[i] : desc;
      (void)input_info.Set("shape_aclmdlGetInputDimsV2", desc_v2.shape);
    }
    (void)input_info.Set("data_type", ge::TypeUtils::DataTypeToSerialString(desc.data_type));
    (void)input_info.Set("format", ge::TypeUtils::FormatToSerialString(desc.format));
    (void)input_info.Set("size", desc.size);
    (void)input_info.Set("shape_range", desc.shape_range);
    input_json_array.push_back(input_info.Raw());
  }
  (void)model_meta_info.Set("inputs", input_json_array);
}

void SerializeModelOutputDescs(const gert::GertModelDataModelMeta &model_meta, ge::JsonFile &model_meta_info) {
  auto output_json_array = ge::JsonFile::json::array();
  for (size_t i = 0UL; i < model_meta.output_desc.size(); ++i) {
    const auto &desc = model_meta.output_desc[i];
    ge::JsonFile output_info;
    (void)output_info.Set("name", gert::GertGetStr(desc.name));
    (void)output_info.Set("index", i);
    (void)output_info.Set("shape", desc.shape);
    (void)output_info.Set("data_type", ge::TypeUtils::DataTypeToSerialString(desc.data_type));
    (void)output_info.Set("format", ge::TypeUtils::FormatToSerialString(desc.format));
    (void)output_info.Set("size", desc.size);
    (void)output_info.Set("shape_range", desc.shape_range);
    output_json_array.push_back(output_info.Raw());
  }
  (void)model_meta_info.Set("outputs", output_json_array);
}

void SerializeModelMetaDynamicInfo(const gert::GertModelDataModelMeta &model_meta, ge::JsonFile &model_meta_info) {
  if (model_meta.dynamic_batch_info.empty()) {
    return;
  }
  ge::JsonFile dynamic_dims_json;
  (void)dynamic_dims_json.Set("dynamic_type", model_meta.dynamic_type);
  std::vector<std::string> user_designate_shape_order;
  user_designate_shape_order.reserve(model_meta.user_designate_shape_order.size());
  for (const auto &s : model_meta.user_designate_shape_order) {
    user_designate_shape_order.emplace_back(gert::GertGetStr(s));
  }
  (void)dynamic_dims_json.Set("user_designate_shape_order", user_designate_shape_order);

  std::map<size_t, std::vector<ge::JsonFile::json>> gear_outputs;
  for (const auto &shape_str_ptr : model_meta.dynamic_output_shape) {
    const std::string shape_str(gert::GertGetStr(shape_str_ptr));
    std::vector<int64_t> values;
    std::istringstream iss(shape_str);
    std::string token;
    while (std::getline(iss, token, ',')) {
      values.push_back(std::stoll(token));
    }
    if (values.size() >= 2UL && values[0] >= 0) {
      auto dims = ge::JsonFile::json::array();
      for (size_t i = 2UL; i < values.size(); ++i) {
        dims.push_back(values[i]);
      }
      gear_outputs[static_cast<size_t>(values[0])].push_back(std::move(dims));
    }
  }

  auto gears_array = ge::JsonFile::json::array();
  for (size_t gear_idx = 0UL; gear_idx < model_meta.dynamic_batch_info.size(); ++gear_idx) {
    ge::JsonFile gear_json;
    (void)gear_json.Set("inputs", model_meta.dynamic_batch_info[gear_idx]);

    auto outputs_array = ge::JsonFile::json::array();
    auto it = gear_outputs.find(gear_idx);
    if (it != gear_outputs.end()) {
      for (const auto &dims : it->second) {
        outputs_array.push_back(dims);
      }
    }
    (void)gear_json.Set("outputs", outputs_array);
    gears_array.push_back(gear_json.Raw());
  }
  (void)dynamic_dims_json.Set("gears", gears_array);
  (void)model_meta_info.Set("dynamic_dims", dynamic_dims_json);
}

ge::Status SerializeModelMeta(const gert::GertModelDataModel &unit, const std::shared_ptr<ZipArchiveWriter> &zip_writer,
                              const size_t model_index) {
  if (unit.model_meta == nullptr) {
    return ge::SUCCESS;
  }
  const auto &model_meta = *unit.model_meta;
  ge::JsonFile model_meta_info;
  SerializeModelInputDescs(model_meta, model_meta_info);
  SerializeModelOutputDescs(model_meta, model_meta_info);
  SerializeModelMetaDynamicInfo(model_meta, model_meta_info);
  (void)model_meta_info.Set("work_size", model_meta.work_size);
  (void)model_meta_info.Set("zero_copy_size", model_meta.zero_copy_size);
  (void)model_meta_info.Set("name", gert::GertGetStr(model_meta.model_name));

  // 序列化 AIPP 元数据
  SerializeAippMeta(model_meta, model_meta_info);

  const auto model_meta_info_str = model_meta_info.Dump();
  const auto model_meta_entry_path = FormatOm2Path(OM2_MODEL_META_PATH_FORMAT, std::to_string(model_index).c_str());
  GE_ASSERT_TRUE(
      zip_writer->WriteBytes(model_meta_entry_path, model_meta_info_str.data(), model_meta_info_str.size(), false));
  return ge::SUCCESS;
}

ge::Status SerializeDebugInfo(const gert::GertModelDataModel &unit, const std::shared_ptr<ZipArchiveWriter> &zip_writer,
                              const size_t model_index) {
  // op_attr.json：Build 侧生成的 JSON 字符串直通写入
  const char *op_attr_json_str = (unit.op_attr_json != nullptr) ? gert::GertGetStr(unit.op_attr_json) : "{}";
  const auto op_attr_entry_path = FormatOm2Path(OM2_OP_ATTR_PATH_FORMAT, std::to_string(model_index).c_str());
  GE_ASSERT_TRUE(zip_writer->WriteBytes(op_attr_entry_path, op_attr_json_str, std::strlen(op_attr_json_str), false));

  // visual json
  if (unit.debug == nullptr) {
    return ge::SUCCESS;
  }
  const auto visual_entry_path = FormatOm2Path(OM2_VISUAL_JSON_PATH_FORMAT, std::to_string(model_index).c_str());
  const auto visual_json_str = gert::GertGetStr(unit.debug->visual_json);
  GE_ASSERT_TRUE(zip_writer->WriteBytes(visual_entry_path, visual_json_str, std::strlen(visual_json_str), true));
  return ge::SUCCESS;
}

ge::Status SerializeManifest(const gert::GertModelData &model_data,
                             const std::shared_ptr<ZipArchiveWriter> &zip_writer) {
  if (model_data.manifest == nullptr) {
    return ge::SUCCESS;
  }
  ge::JsonFile manifest_json;
  const auto &manifest = *model_data.manifest;

  ge::JsonFile compatibility_json;
  (void)compatibility_json.Set(OM2_MANIFEST_KEY_COMPILER_VERSION,
                               gert::GertGetStr(manifest.compatibility.compiler_version));
  (void)compatibility_json.Set(OM2_MANIFEST_KEY_REQUIRED_EXECUTOR_VERSION,
                               gert::GertGetStr(manifest.compatibility.required_executor_version));

  auto used_features_json = ge::JsonFile::json::object();
  for (const auto &[feature_name, feature_version] : manifest.compatibility.used_features) {
    used_features_json[gert::GertGetStr(feature_name)] = gert::GertGetStr(feature_version);
  }
  (void)compatibility_json.Set(OM2_MANIFEST_KEY_USED_FEATURES, used_features_json);

  (void)manifest_json.Set(OM2_MODEL_NUM, manifest.model_num);
  (void)manifest_json.Set(OM2_ATC_COMMAND, gert::GertGetStr(manifest.atc_command));
  (void)manifest_json.Set(OM2_MANIFEST_KEY_COMPATIBILITY, compatibility_json);

  const std::string manifest_str = manifest_json.Dump();
  GE_ASSERT_TRUE(zip_writer->WriteBytes(OM2_MANIFEST_PATH, manifest_str.data(), manifest_str.size(), false));
  return ge::SUCCESS;
}

}  // namespace

ge::Status SerializeGertModelData(const GertModelData &model_data, ge::ModelBufferData &model, const bool is_offline,
                                  const std::string &writer_path) {
  GE_ASSERT_TRUE(!model_data.models.empty(), "[OM2] models is empty, nothing to serialize.");
  GELOGI(
      "[OM2] Begin to serialize GertModelData to ZIP, model_name:%s, "
      "inputs:%zu, outputs:%zu, kernels:%zu, custom kernels: %zu, weight_size:%zu",
      gert::GertGetStr(model_data.models[0]->model_meta->model_name),
      model_data.models[0]->model_meta->input_desc.size(), model_data.models[0]->model_meta->output_desc.size(),
      model_data.kernels->binaries.size(), model_data.custom_ops->binaries.size(),
      model_data.models[0]->constants_config->internal_weight_size);
  const std::string path = writer_path.empty() ? "om2_model" : writer_path;
  auto zip_writer = std::make_shared<gert::ZipArchiveWriter>(path);
  GE_ASSERT_NOTNULL(zip_writer);
  GE_ASSERT_TRUE(zip_writer->IsMemFileOpened());

  GE_ASSERT_SUCCESS(SerializeManifest(model_data, zip_writer));
  for (size_t model_index = 0UL; model_index < model_data.models.size(); ++model_index) {
    const auto &unit = *model_data.models[model_index];
    GE_ASSERT_SUCCESS(SerializeCodegenArtifacts(unit, zip_writer, model_index));
    if ((model_index < model_data.constants->constants_data.size()) &&
        (model_data.constants->constants_data[model_index] != nullptr)) {
      GE_ASSERT_SUCCESS(
          SerializeWeightData(unit, *model_data.constants->constants_data[model_index], zip_writer, model_index));
    }
    GE_ASSERT_SUCCESS(SerializeConstantsConfig(unit, zip_writer, is_offline, model_index));
    GE_ASSERT_SUCCESS(SerializeModelMeta(unit, zip_writer, model_index));
    GE_ASSERT_SUCCESS(SerializeDebugInfo(unit, zip_writer, model_index));
    GE_ASSERT_SUCCESS(SerializeVariablesData(unit, zip_writer, model_index));
  }
  GE_ASSERT_SUCCESS(SerializeKernelBinaries(model_data, zip_writer));
  GE_ASSERT_SUCCESS(SerializeCustomKernelBinaries(model_data, zip_writer));
  GE_ASSERT_SUCCESS(SerializeCustomKernelSharedLibs(model_data, zip_writer));

  GertBuffer om2_buf;
  GE_ASSERT_TRUE(zip_writer->SaveModelData(om2_buf, is_offline));
  model.data = om2_buf.data;
  model.length = om2_buf.length;
  GELOGI(
      "[OM2] Successfully serialized GertModelData to ZIP, model_name:%s, "
      "buffer_size:%zu, is_offline:%d",
      gert::GertGetStr(model_data.models[0]->model_meta->model_name), model.length, is_offline);
  return ge::SUCCESS;
}

}  // namespace gert
