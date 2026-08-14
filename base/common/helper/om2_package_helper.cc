/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "common/ge_common/string_util.h"
#include "framework/common/helper/om2_package_helper.h"
#include "framework/common/helper/model_save_helper_factory.h"
#include "common/file_constant_utils/file_constant_utils.h"
#include "common/ge_common/ge_types.h"
#include "common/helper/om2/zip_archive_writer.h"
#include "common/helper/om2/om2_zip_saver.h"
#include "common/helper/om2/om2_package_contants.h"
#include "common/helper/om2/json_file.h"
#include "common/om2/om2_model_data.h"
#include "common/om2/codegen/om2_codegen.h"
#include "common/om2/codegen/om2_codegen_utils.h"
#include "framework/omg/omg_inner_types.h"
#include "graph/debug/ge_attr_define.h"
#include "graph_metadef/common/plugin/plugin_manager.h"
#include "graph/utils/type_utils.h"
#include "graph/utils/tensor_utils.h"
#include "graph_metadef/graph/utils/file_utils.h"
#include "graph/model.h"
#include "common/helper/visual_json_converter.h"
#include "graph/ge_context.h"
#include "graph/manager/graph_var_manager.h"
#include "common/helper/om2/rt_var_resource_builder.h"

namespace ge {
namespace {
constexpr auto kAttrKernelName = "_kernelname";
const std::string kOm2ConstantsConfigSuffix = "_constants_config.json";
const std::string kOm2ExternalWeightDirName = "weight";

constexpr size_t kAippDimPartsNum = 6U;
constexpr size_t kAippDimNameIdx = 2U;
constexpr size_t kAippDimSizeIdx = 3U;
constexpr size_t kAippDimDimNumIdx = 4U;
constexpr size_t kAippDimShapeIdx = 5U;
constexpr int32_t kAippDecimalRadix = 10;

struct ModelIoNodes {
  std::map<uint32_t, OpDescPtr> input_ops;
  std::vector<OpDescPtr> output_ops;
  std::vector<OpDescPtr> case_ops;
};

struct ModelMetaExtraInfo {
  JsonFile::json dynamic_output_shape = JsonFile::json::array();
  JsonFile::json dynamic_batch_info = JsonFile::json::array();
  JsonFile::json user_designate_shape_order = JsonFile::json::array();
  int32_t dynamic_type = 0;
};

Status GetDynamicBatchInfo(const OpDescPtr &op_desc, JsonFile::json &batch_info,
                           JsonFile::json &user_designate_shape_order, int32_t &dynamic_type) {
  uint32_t batch_num = 0U;
  if (!AttrUtils::GetInt(op_desc, ATTR_NAME_BATCH_NUM, batch_num)) {
    GELOGI("Not multi-batch Node: %s", op_desc->GetName().c_str());
    return SUCCESS;
  }
  batch_info.clear();

  (void)AttrUtils::GetInt(op_desc, ATTR_DYNAMIC_TYPE, dynamic_type);
  std::vector<std::string> user_designate_shape_order_vec;
  (void)AttrUtils::GetListStr(op_desc, ATTR_USER_DESIGNEATE_SHAPE_ORDER, user_designate_shape_order_vec);
  for (const auto &s : user_designate_shape_order_vec) {
    user_designate_shape_order.push_back(s);
  }
  for (uint32_t i = 0U; i < batch_num; ++i) {
    std::vector<int64_t> batch_shape;
    const std::string attr_name = ATTR_NAME_PRED_VALUE + "_" + std::to_string(i);
    if (!AttrUtils::GetListInt(op_desc, attr_name, batch_shape)) {
      REPORT_INNER_ERR_MSG("E19999", "Get Attr:%s from op:%s(%s) fail", attr_name.c_str(), op_desc->GetName().c_str(),
                           op_desc->GetType().c_str());
      GELOGE(FAILED, "[Get][Attr] %s from op:%s(%s) fail", attr_name.c_str(), op_desc->GetName().c_str(),
             op_desc->GetType().c_str());
      batch_info.clear();
      return FAILED;
    }
    batch_info.push_back(batch_shape);
  }
  return SUCCESS;
}

bool EndsWith(const std::string &str, const std::string &suffix) {
  return (str.size() >= suffix.size()) && (str.compare(str.size() - suffix.size(), suffix.size(), suffix) == 0);
}

std::string StripOm2ArchiveRoot(const std::string &entry_name) {
  const auto pos = entry_name.find('/');
  return (pos == std::string::npos) ? entry_name : entry_name.substr(pos + 1U);
}

bool IsOm2ConstantsConfigEntry(const std::string &entry_name) {
  return (entry_name.find(OM2_CONSTANTS_DIR) == 0U) && EndsWith(entry_name, kOm2ConstantsConfigSuffix);
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
  const char_t *const om_dir = mmDirName(&path[0]);
  if (om_dir == nullptr) {
    return "";
  }
  return std::string(om_dir) + "/" + kOm2ExternalWeightDirName + "/" + file_name;
}

Status RewriteOm2ConstantsConfig(const std::string &output_file_name, JsonFile &constants_json,
                                 std::map<std::string, std::string> &old_file_to_new_file, bool &changed) {
  JsonFile::json consts_json;
  if (!constants_json.Get("consts", consts_json) || !consts_json.is_object()) {
    return SUCCESS;
  }
  for (auto &const_item : consts_json.items()) {
    auto &const_info = const_item.value();
    if (!const_info.is_object()) {
      continue;
    }
    JsonFile const_info_json(const_info);
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
      file_name = StringUtils::GetFileName(old_file_path);
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
  return SUCCESS;
}

Status CollectOm2ExternalWeightRelocation(const std::string &output_file_name, const SimpleZipArchiveReader &archive,
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
    const JsonFile const_json_readonly(reinterpret_cast<const uint8_t *>(buffer.get()), buffer_size);
    GE_ASSERT_TRUE(const_json_readonly.IsValid(), "[OM2] Invalid constants config entry %s", entry_name.c_str());
    JsonFile const_json(const_json_readonly.Raw());
    bool changed = false;
    GE_ASSERT_SUCCESS(RewriteOm2ConstantsConfig(output_file_name, const_json, old_file_to_new_file, changed));
    if (changed) {
      rewritten_configs[entry_name] = const_json.Dump();
    }
  }
  return SUCCESS;
}

Status RepackOm2Model(const std::string &output_file_name, const SimpleZipArchiveReader &archive,
                      const std::vector<std::string> &archive_entries,
                      const std::map<std::string, std::string> &rewritten_configs, ModelBufferData &relocated_model) {
  auto zip_writer = MakeShared<ZipArchiveWriter>(output_file_name);
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
  return SUCCESS;
}

Status CollectModelIoNodes(const ComputeGraphPtr &graph, ModelIoNodes &io_nodes) {
  uint32_t data_index = 0U;
  const std::set<std::string> kDataOpTypes{DATA, REFDATA, AIPPDATA, ANN_DATA};
  for (const auto &node : graph->GetDirectNode()) {
    const auto &op_desc = node->GetOpDesc();
    GE_ASSERT_NOTNULL(op_desc);
    if (kDataOpTypes.count(op_desc->GetType()) > 0U) {
      uint32_t tmp_index = data_index++;
      if (AttrUtils::GetInt(op_desc, ATTR_NAME_INDEX, tmp_index)) {
        GELOGD("Get new data index %u, old index is %u", tmp_index, data_index - 1U);
      }
      io_nodes.input_ops[tmp_index] = op_desc;
      GELOGD("Find input node [%s], index [%u]", node->GetNamePtr(), tmp_index);
      continue;
    }
    if (op_desc->GetType() == NETOUTPUT) {
      io_nodes.output_ops.push_back(op_desc);
      GELOGD("Find output node [%s]", node->GetNamePtr());
    }
    if (op_desc->GetType() == CASE) {
      io_nodes.case_ops.push_back(op_desc);
      GELOGD("Find case node [%s]", node->GetNamePtr());
    }
  }
  return SUCCESS;
}

Status CollectDynamicBatchInfo(const std::vector<OpDescPtr> &case_ops, ModelMetaExtraInfo &extra_info) {
  for (const auto &op_desc : case_ops) {
    GE_ASSERT_SUCCESS(GetDynamicBatchInfo(op_desc, extra_info.dynamic_batch_info, extra_info.user_designate_shape_order,
                                          extra_info.dynamic_type));
  }
  return SUCCESS;
}

std::string GetRootGraphName(const GeModelPtr &ge_model) {
  if (ge_model == nullptr) {
    return "";
  }
  auto graph = ge_model->GetGraph();
  while ((graph != nullptr) && (graph->GetParentGraph() != nullptr)) {
    graph = graph->GetParentGraph();
  }
  return (graph == nullptr) ? "" : graph->GetName();
}

Status SetOm2CompatibleOmInfoList(const GeModelPtr &ge_model) {
  std::vector<int64_t> om_info;
  om_info.push_back(static_cast<int64_t>(ge_model->GetWeightSize()));
  om_info.push_back(static_cast<int64_t>(ge_model->GetTBEKernelStore().DataSize()));
  om_info.push_back(static_cast<int64_t>(ge_model->GetCustAICPUKernelStore().DataSize()));
  const auto &task_def = ge_model->GetModelTaskDefPtr();
  om_info.push_back(task_def != nullptr ? static_cast<int64_t>(task_def->ByteSizeLong()) : 0);
  // 保持 OM om_info_list 的字段结构，便于 JSON 对齐。OM2 资源以 ZIP entry 存储，
  // 不再构造旧 SO_STORE 分区，因此 so_store_size 记为 0。
  om_info.push_back(0);
  GE_CHK_BOOL_EXEC(ge::AttrUtils::SetListInt(*(ge_model.get()), "om_info_list", om_info),
                   GELOGE(FAILED, "[OM2] SetListInt of om_info_list failed.");
                   return FAILED);
  return SUCCESS;
}

static void ConvertAippAttrToConfigInfo(const GeAttrValue::NamedAttrs &aipp_attr, ge::AippConfigInfo &info) {
  GELOGD("[OM2] Converting NamedAttrs to AippConfigInfo");
  int64_t i64_val = 0;
  float32_t f32_val = 0.0F;
  bool b_val = false;
  std::vector<int64_t> i64_vec;
  std::vector<float32_t> f32_vec;

  auto getInt = [&aipp_attr, &i64_val](const char *k) -> int64_t {
    (void)aipp_attr.GetItem(k).GetValue<GeAttrValue::INT>(i64_val);
    return i64_val;
  };
  auto getF32 = [&aipp_attr, &f32_val](const char *k) -> float32_t {
    (void)aipp_attr.GetItem(k).GetValue<GeAttrValue::FLOAT>(f32_val);
    return f32_val;
  };
  auto getBool = [&aipp_attr, &b_val](const char *k) -> bool {
    (void)aipp_attr.GetItem(k).GetValue<GeAttrValue::BOOL>(b_val);
    return b_val;
  };
  auto getListIntFirst = [&aipp_attr, &i64_vec](const char *k) -> int32_t {
    if (aipp_attr.GetItem(k).GetValue<GeAttrValue::LIST_INT>(i64_vec) == SUCCESS && !i64_vec.empty()) {
      return static_cast<int32_t>(i64_vec[0]);
    }
    return 0;
  };
  auto getListF32First = [&aipp_attr, &f32_vec](const char *k) -> float32_t {
    if (aipp_attr.GetItem(k).GetValue<GeAttrValue::LIST_FLOAT>(f32_vec) == SUCCESS && !f32_vec.empty()) {
      return f32_vec[0];
    }
    return 0.0F;
  };

  info.aipp_mode = static_cast<int8_t>(getInt("aipp_mode"));
  info.input_format = static_cast<int8_t>(getInt("input_format"));
  info.src_image_size_w = static_cast<int32_t>(getInt("src_image_size_w"));
  info.src_image_size_h = static_cast<int32_t>(getInt("src_image_size_h"));
  info.crop = static_cast<int8_t>(getBool("crop"));
  info.load_start_pos_w = static_cast<int32_t>(getInt("load_start_pos_w"));
  info.load_start_pos_h = static_cast<int32_t>(getInt("load_start_pos_h"));
  info.crop_size_w = static_cast<int32_t>(getInt("crop_size_w"));
  info.crop_size_h = static_cast<int32_t>(getInt("crop_size_h"));
  info.resize = static_cast<int8_t>(getBool("resize"));
  info.resize_output_w = static_cast<int32_t>(getInt("resize_output_w"));
  info.resize_output_h = static_cast<int32_t>(getInt("resize_output_h"));
  info.padding = static_cast<int8_t>(getBool("padding"));
  info.left_padding_size = static_cast<int32_t>(getInt("left_padding_size"));
  info.right_padding_size = static_cast<int32_t>(getInt("right_padding_size"));
  info.top_padding_size = static_cast<int32_t>(getInt("top_padding_size"));
  info.bottom_padding_size = static_cast<int32_t>(getInt("bottom_padding_size"));
  info.csc_switch = static_cast<int8_t>(getBool("csc_switch"));
  info.rbuv_swap_switch = static_cast<int8_t>(getBool("rbuv_swap_switch"));
  info.ax_swap_switch = static_cast<int8_t>(getBool("ax_swap_switch"));
  info.single_line_mode = static_cast<int8_t>(getBool("single_line_mode"));
  info.matrix_r0c0 = getListIntFirst("matrix_r0c0");
  info.matrix_r0c1 = getListIntFirst("matrix_r0c1");
  info.matrix_r0c2 = getListIntFirst("matrix_r0c2");
  info.matrix_r1c0 = getListIntFirst("matrix_r1c0");
  info.matrix_r1c1 = getListIntFirst("matrix_r1c1");
  info.matrix_r1c2 = getListIntFirst("matrix_r1c2");
  info.matrix_r2c0 = getListIntFirst("matrix_r2c0");
  info.matrix_r2c1 = getListIntFirst("matrix_r2c1");
  info.matrix_r2c2 = getListIntFirst("matrix_r2c2");
  info.output_bias_0 = getListIntFirst("output_bias_0");
  info.output_bias_1 = getListIntFirst("output_bias_1");
  info.output_bias_2 = getListIntFirst("output_bias_2");
  info.input_bias_0 = getListIntFirst("input_bias_0");
  info.input_bias_1 = getListIntFirst("input_bias_1");
  info.input_bias_2 = getListIntFirst("input_bias_2");
  info.mean_chn_0 = static_cast<int32_t>(getInt("mean_chn_0"));
  info.mean_chn_1 = static_cast<int32_t>(getInt("mean_chn_1"));
  info.mean_chn_2 = static_cast<int32_t>(getInt("mean_chn_2"));
  info.mean_chn_3 = static_cast<int32_t>(getInt("mean_chn_3"));
  info.min_chn_0 = getF32("min_chn_0");
  info.min_chn_1 = getF32("min_chn_1");
  info.min_chn_2 = getF32("min_chn_2");
  info.min_chn_3 = getF32("min_chn_3");
  info.var_reci_chn_0 = getListF32First("var_reci_chn_0");
  info.var_reci_chn_1 = getListF32First("var_reci_chn_1");
  info.var_reci_chn_2 = getListF32First("var_reci_chn_2");
  info.var_reci_chn_3 = getListF32First("var_reci_chn_3");
  info.support_rotation = static_cast<int8_t>(getBool("support_rotation"));
  info.related_input_rank = static_cast<uint32_t>(getInt("related_input_rank"));
  info.max_src_image_size = static_cast<uint32_t>(getInt("max_src_image_size"));
}

static Status ParseAippModeStr(const std::string &mode, ge::InputAippType &aipp_type) {
  if (mode == "static_aipp") {
    aipp_type = ge::DATA_WITH_STATIC_AIPP;
  } else if (mode == "dynamic_aipp") {
    aipp_type = ge::DATA_WITH_DYNAMIC_AIPP;
  } else if (mode == "dynamic_aipp_conf") {
    aipp_type = ge::DYNAMIC_AIPP_NODE;
  } else {
    GELOGE(PARAM_INVALID, "[OM2] Unknown AIPP mode: %s", mode.c_str());
    return PARAM_INVALID;
  }
  return SUCCESS;
}

static size_t ResolveAippDataIndex(const std::map<std::string, uint32_t> &data_index_map,
                                   const std::string &target_name) {
  const auto iter = data_index_map.find(target_name);
  return (iter != data_index_map.end()) ? static_cast<size_t>(iter->second) : 0U;
}

static void ParseOrigInputInfoFromStr(const std::string &input_str, ge::OriginInputInfo &orig_info) {
  const auto parts = StringUtils::Split(input_str, ':');
  if (parts.size() >= 5U) {
    orig_info.format = static_cast<ge::Format>(ge::TypeUtils::SerialStringToFormat(parts[0]));
    orig_info.data_type = static_cast<ge::DataType>(ge::TypeUtils::SerialStringToDataType(parts[1]));
    orig_info.dim_num =
        static_cast<uint32_t>(std::strtol(parts[kAippDimDimNumIdx].c_str(), nullptr, kAippDecimalRadix));
  }
}

// 将 "NCHW:DT_FLOAT:data:0:4:1,3,224,224" 格式的字符串解析为 InputOutputDims
static Status ParseAippDimInfo(const std::string &info_str, ge::InputOutputDims &dims_info) {
  const auto parts = StringUtils::Split(info_str, ':');
  if (parts.size() != kAippDimPartsNum) {
    GELOGW("[OM2][AIPP] Invalid aipp dim info: %s, parts=%zu", info_str.c_str(), parts.size());
    return FAILED;
  }
  dims_info.name = parts[kAippDimNameIdx];
  dims_info.size = static_cast<uint32_t>(std::strtol(parts[kAippDimSizeIdx].c_str(), nullptr, kAippDecimalRadix));
  dims_info.dim_num = static_cast<size_t>(std::strtol(parts[kAippDimDimNumIdx].c_str(), nullptr, kAippDecimalRadix));

  const auto dim_strs = StringUtils::Split(parts[kAippDimShapeIdx], ',');
  for (const auto &dim_str : dim_strs) {
    if (dim_str.empty()) {
      continue;
    }
    dims_info.dims.emplace_back(std::strtol(dim_str.c_str(), nullptr, kAippDecimalRadix));
  }
  return SUCCESS;
}

static Status ParseAippDims(const std::vector<std::string> &dim_strs, std::vector<ge::InputOutputDims> &dims) {
  for (const auto &s : dim_strs) {
    ge::InputOutputDims dim_info;
    GE_CHK_STATUS_RET(ParseAippDimInfo(s, dim_info), "[Parse][AippDimInfo] failed for: %s", s.c_str());
    dims.push_back(std::move(dim_info));
  }
  return SUCCESS;
}

static Status ExtractAippMetaFromOpDesc(const OpDescPtr &op_desc, const std::map<std::string, uint32_t> &data_index_map,
                                        gert::Om2AippMeta &meta) {
  GELOGD("[OM2] Extract AIPP meta from node: %s", op_desc->GetName().c_str());
  GeAttrValue::NamedAttrs aipp_attr;
  if (ge::AttrUtils::GetNamedAttrs(op_desc, ATTR_NAME_AIPP, aipp_attr)) {
    ConvertAippAttrToConfigInfo(aipp_attr, meta.aipp_config_info);
  }
  const std::string *related_name = ge::AttrUtils::GetStr(op_desc, ATTR_DATA_AIPP_DATA_NAME_MAP);
  if (related_name != nullptr) {
    meta.aipp_data_index = ResolveAippDataIndex(data_index_map, *related_name);
  }
  std::vector<std::string> aipp_inputs;
  std::vector<std::string> aipp_outputs;
  (void)ge::AttrUtils::GetListStr(op_desc, ATTR_NAME_AIPP_INPUTS, aipp_inputs);
  (void)ge::AttrUtils::GetListStr(op_desc, ATTR_NAME_AIPP_OUTPUTS, aipp_outputs);
  GE_CHK_STATUS_RET(ParseAippDims(aipp_inputs, meta.aipp_input_dims));
  GE_CHK_STATUS_RET(ParseAippDims(aipp_outputs, meta.aipp_output_dims));
  if (!aipp_inputs.empty()) {
    ParseOrigInputInfoFromStr(aipp_inputs[0], meta.orig_input_info);
  }
  return SUCCESS;
}

static Status CollectAippMetas(const ComputeGraphPtr &graph, gert::Om2ModelMeta &model_meta) {
  std::map<std::string, uint32_t> data_index_map;
  for (const auto &node : graph->GetDirectNode()) {
    const auto op_desc = node->GetOpDesc();
    if (op_desc != nullptr) {
      uint32_t index = 0U;
      if (ge::AttrUtils::GetInt(op_desc, ATTR_NAME_INDEX, index)) {
        data_index_map[op_desc->GetName()] = index;
      }
    }
  }

  for (const auto &node : graph->GetDirectNode()) {
    const auto op_desc = node->GetOpDesc();
    if (op_desc == nullptr) {
      continue;
    }
    const std::string *mode = ge::AttrUtils::GetStr(op_desc, ATTR_DATA_RELATED_AIPP_MODE);
    if (mode == nullptr) {
      continue;
    }
    GELOGI("[OM2] Found AIPP node: %s, mode=%s", op_desc->GetName().c_str(), mode->c_str());
    ge::InputAippType aipp_type;
    GE_CHK_STATUS_RET(ParseAippModeStr(*mode, aipp_type), "[Parse][AippMode] Unknown AIPP mode for node: %s",
                      op_desc->GetName().c_str());
    uint32_t input_index = 0U;
    (void)ge::AttrUtils::GetInt(op_desc, ATTR_NAME_INDEX, input_index);
    if (input_index >= model_meta.aipp_infos.size()) {
      model_meta.aipp_infos.resize(input_index + 1U);
    }
    gert::Om2AippMeta &meta = model_meta.aipp_infos[input_index];
    meta.aipp_type = aipp_type;
    const Status ret = ExtractAippMetaFromOpDesc(op_desc, data_index_map, meta);
    if (ret != SUCCESS) {
      GELOGE(ret, "[OM2] ExtractAippMetaFromOpDesc failed for node: %s", op_desc->GetName().c_str());
      return ret;
    }
    model_meta.has_aipp = true;
  }
  GELOGI("[OM2] Collected %zu AIPP metas", model_meta.aipp_infos.size());
  return SUCCESS;
}

}  // namespace

Status Om2PackageHelper::SaveToOmRootModel(const GeRootModelPtr &ge_root_model, const std::string &output_file,
                                           ModelBufferData &model, const bool is_unknown_shape) {
  GE_ASSERT_NOTNULL(ge_root_model, "[OM2] ge_root_model is nullptr");
  GE_ASSERT_TRUE(!output_file.empty(), "[OM2] Empty path of output file is invalid");
  const auto &name_to_ge_model = ge_root_model->GetSubgraphInstanceNameToModel();
  GE_ASSERT_TRUE(!name_to_ge_model.empty(), "[OM2] No subgraphs found in ge_root_model");

  if (!is_unknown_shape) {
    auto &model_root = name_to_ge_model.begin()->second;
    return SaveToOmModel(model_root, output_file, model, ge_root_model);
  }

  // todo 动态 shape 场景暂时不支持
  GELOGE(FAILED, "[OM2] Unknown shape models are not supported for .om2 format conversion");
  (void)REPORT_PREDEFINED_ERR_MSG(
      "E10055", std::vector<const char *>({"reason"}),
      std::vector<const char *>({"Unknown shape models are not supported for .om2 format conversion"}));
  return FAILED;
}

Status Om2PackageHelper::SaveToOmModel(const GeModelPtr &ge_model, const std::string &output_file,
                                       ModelBufferData &model, const GeRootModelPtr &ge_root_model) {
  GE_ASSERT_NOTNULL(ge_model, "ge_model is nullptr");
  GE_ASSERT_TRUE(!output_file.empty(), "[OM2] Empty path of the output file is invalid");

  gert::Om2ModelData model_data;
  GE_ASSERT_SUCCESS(BuildOm2ModelData(ge_model, model_data, ge_root_model));

  // Serialize to ZIP via Om2ZipSaver
  const std::string writer_path = (!is_offline_ && !ge_model->GetName().empty()) ? ge_model->GetName() : output_file;
  GE_ASSERT_SUCCESS(Om2ZipSaver::Save(model_data, model, is_offline_, writer_path));

  GELOGI("[OM2] Successfully created OM2 model");
  return SUCCESS;
}

void Om2PackageHelper::SetSaveMode(const bool val) {
  is_offline_ = val;
}

Status Om2PackageHelper::RelocateExternalWeights(const std::string &output_file_name, const ModelBufferData &model,
                                                 ModelBufferData &relocated_model, bool &relocated) {
  relocated = false;
  SimpleZipArchiveReader archive(model.data.get(), model.length);
  if (!archive.IsGood()) {
    GELOGW("[OM2] Model buffer has zip magic but is not a valid zip archive, save original buffer.");
    return SUCCESS;
  }
  const auto archive_entries = archive.ListFiles();
  std::map<std::string, std::string> rewritten_configs;
  std::map<std::string, std::string> old_file_to_new_file;
  GE_ASSERT_SUCCESS(CollectOm2ExternalWeightRelocation(output_file_name, archive, archive_entries, rewritten_configs,
                                                       old_file_to_new_file));
  if (old_file_to_new_file.empty()) {
    return SUCCESS;
  }
  GE_ASSERT_SUCCESS(RepackOm2Model(output_file_name, archive, archive_entries, rewritten_configs, relocated_model));
  GE_ASSERT_SUCCESS(FileConstantUtils::MoveExternalWeightFiles(old_file_to_new_file));
  relocated = true;
  return SUCCESS;
}

Status Om2PackageHelper::ExtractVisualJson(const void *model_data, size_t model_len, std::string &json_out) {
  GE_ASSERT_NOTNULL(model_data, "[OM2] model_data is nullptr");
  GE_ASSERT_TRUE(model_len > 0U, "[OM2] model_len is 0");

  const auto *data = static_cast<const uint8_t *>(model_data);
  SimpleZipArchiveReader reader(data, model_len);
  GE_ASSERT_TRUE(reader.IsGood(), "[OM2] Failed to open OM2 ZIP archive");

  const auto file_list = reader.ListFiles();
  std::string entry_path;
  for (const auto &f : file_list) {
    if (((f.find("/debug/ge_visual_") != std::string::npos) && EndsWith(f, ".json")) ||
        EndsWith(f, "/debug/visual.json") || EndsWith(f, "debug/visual.json")) {
      entry_path = f;
      break;
    }
  }
  GE_ASSERT_TRUE(!entry_path.empty(), "[OM2] visual JSON not found in OM2 archive");

  size_t json_size = 0U;
  auto json_buf = reader.ExtractToMem(entry_path, json_size);
  GE_ASSERT_NOTNULL(json_buf, "[OM2] Failed to extract %s from OM2 archive", entry_path.c_str());
  GE_ASSERT_TRUE(json_size > 0U, "[OM2] Extracted visual JSON is empty");

  json_out.assign(reinterpret_cast<const char *>(json_buf.get()), json_size);
  GELOGI("[OM2] Extracted visual JSON, entry:%s, size:%zu", entry_path.c_str(), json_out.size());
  return SUCCESS;
}

Status Om2PackageHelper::BuildProgramBody(const GeModelPtr &ge_model, gert::Om2ProgramBody &body,
                                          std::vector<Om2ConstMeta> &const_metas, std::vector<Om2VarMeta> &var_metas) {
  GELOGI("[OM2] Begin to build program body");
  Om2Codegen codegen;
  GE_ASSERT_SUCCESS(codegen.Om2CodegenAndCompile(ge_model, body.source_artifacts, const_metas, var_metas));
  GE_ASSERT_TRUE(!body.source_artifacts.empty());

  for (const auto &artifact : body.source_artifacts) {
    if (artifact.file_name.find(".so") != std::string::npos) {
      body.so_artifact = artifact;
      break;
    }
  }

  GELOGI("[OM2] Successfully built program body, artifacts count=%zu, const_metas count=%zu, var_metas count=%zu",
         body.source_artifacts.size(), const_metas.size(), var_metas.size());
  return SUCCESS;
}

Status Om2PackageHelper::BuildKernelBinaries(const GeModelPtr &ge_model,
                                             std::vector<gert::Om2KernelBinary> &kernel_binaries) {
  GELOGI("[OM2] Begin to build kernel binaries");
  const auto &graph = ge_model->GetGraph();
  GE_ASSERT_NOTNULL(graph);

  // Collect TBE kernels
  const auto &tbe_kernel_store = ge_model->GetTBEKernelStore();
  std::unordered_set<std::string> added_kernels;
  for (const auto &node : graph->GetNodes(graph->GetGraphUnknownFlag())) {
    std::string kernel_name;
    const auto kernel_name_ptr = AttrUtils::GetStr(node->GetOpDesc(), kAttrKernelName);
    if (kernel_name_ptr != nullptr) {
      kernel_name = *kernel_name_ptr;
    }
    auto kernel_bin = tbe_kernel_store.FindKernel(kernel_name);
    if ((kernel_bin != nullptr) && (added_kernels.count(kernel_name) == 0)) {
      gert::Om2KernelBinary kb;
      kb.name = Om2CodegenUtils::GetKernelNameWithExtension(kernel_name);
      kb.data = ge::ReadonlyByteBuffer(kernel_bin->GetBinData(), ge::ConditionalDeleter{false});
      kb.data_size = kernel_bin->GetBinDataSize();
      kernel_binaries.push_back(std::move(kb));
      (void)added_kernels.insert(kernel_name);
    }

    std::string atomic_kernel_name;
    const auto atomic_kernel_name_ptr = AttrUtils::GetStr(node->GetOpDesc(), ATOMIC_ATTR_TBE_KERNEL_NAME);
    if (atomic_kernel_name_ptr != nullptr) {
      atomic_kernel_name = *atomic_kernel_name_ptr;
    }
    if (!atomic_kernel_name.empty()) {
      const auto atomic_kernel_bin = tbe_kernel_store.FindKernel(atomic_kernel_name);
      if ((atomic_kernel_bin != nullptr) && (added_kernels.count(atomic_kernel_name) == 0)) {
        gert::Om2KernelBinary kb;
        kb.name = Om2CodegenUtils::GetKernelNameWithExtension(atomic_kernel_name);
        kb.data = ge::ReadonlyByteBuffer(atomic_kernel_bin->GetBinData(), ge::ConditionalDeleter{false});
        kb.data_size = atomic_kernel_bin->GetBinDataSize();
        kernel_binaries.push_back(std::move(kb));
        (void)added_kernels.insert(atomic_kernel_name);
      }
    }
  }

  // Collect CustAICPU kernels
  const auto &cust_aicpu_kernel_store = ge_model->GetCustAICPUKernelStore();
  if (cust_aicpu_kernel_store.DataSize() > 0U) {
    for (const auto &node : graph->GetNodes(graph->GetGraphUnknownFlag())) {
      const auto op_desc = node->GetOpDesc();
      GE_IF_BOOL_EXEC(op_desc == nullptr, continue);
      const auto cust_aicpu_kernel = op_desc->TryGetExtAttr(OP_EXTATTR_CUSTAICPU_KERNEL, CustAICPUKernelPtr());
      GE_IF_BOOL_EXEC(cust_aicpu_kernel == nullptr, continue);
      std::string kernel_name = cust_aicpu_kernel->GetName();
      auto kernel_bin = cust_aicpu_kernel_store.FindKernel(kernel_name);
      if ((kernel_bin != nullptr) && (added_kernels.count(kernel_name) == 0)) {
        const size_t hash_id = std::hash<std::string>{}(
            std::string(reinterpret_cast<const char *>(kernel_bin->GetBinData()), kernel_bin->GetBinDataSize()));
        gert::Om2KernelBinary kb;
        kb.name = std::to_string(hash_id) + "_CustAicpuKernel.o";
        kb.data = ge::ReadonlyByteBuffer(kernel_bin->GetBinData(), ge::ConditionalDeleter{false});
        kb.data_size = kernel_bin->GetBinDataSize();
        kernel_binaries.push_back(std::move(kb));
        (void)added_kernels.insert(cust_aicpu_kernel->GetName());
      }
    }
  }

  GELOGI("[OM2] Successfully built kernel binaries, count=%zu", kernel_binaries.size());
  return SUCCESS;
}

Status Om2PackageHelper::BuildModelMeta(const GeModelPtr &ge_model, gert::Om2ModelMeta &model_meta) {
  GELOGI("[OM2] Begin to build model meta");
  const auto &graph = ge_model->GetGraph();
  GE_ASSERT_NOTNULL(graph);

  ModelIoNodes io_nodes;
  GE_ASSERT_SUCCESS(CollectModelIoNodes(graph, io_nodes));

  // Build input descriptors
  for (const auto &[index, op_desc] : io_nodes.input_ops) {
    (void)index;
    const auto &tensor_desc = op_desc->GetInputDescPtr(0);
    GE_ASSERT_NOTNULL(tensor_desc);

    ge::Om2TensorDesc desc;
    desc.SetName(op_desc->GetName());
    desc.SetDataType(tensor_desc->GetDataType());
    desc.SetFormat(tensor_desc->GetFormat());
    desc.SetShape(tensor_desc->GetShape().GetDims());

    int64_t input_size = 0;
    const auto output_desc = op_desc->GetOutputDescPtr(0U);
    if ((output_desc != nullptr) && AttrUtils::GetInt(*output_desc, ATTR_NAME_SPECIAL_INPUT_SIZE, input_size) &&
        (input_size > 0)) {
      desc.SetSize(static_cast<size_t>(input_size));
    } else {
      GE_CHK_STATUS_RET(TensorUtils::GetSize(*tensor_desc, input_size), "[Get][InputSize] failed for op: %s.",
                        op_desc->GetName().c_str());
      desc.SetSize(static_cast<size_t>(input_size));
    }

    std::vector<std::pair<int64_t, int64_t>> range;
    if (tensor_desc->GetShapeRange(range) == SUCCESS) {
      desc.SetShapeRange(range);
    }

    ge::Om2TensorDesc desc_v2 = desc;
    std::vector<int64_t> model_input_dims;
    if (op_desc->HasAttr(ATTR_NAME_INPUT_DIMS)) {
      (void)AttrUtils::GetListInt(op_desc, ATTR_NAME_INPUT_DIMS, model_input_dims);
    } else {
      model_input_dims = tensor_desc->GetShape().GetDims();
    }
    desc_v2.SetShape(model_input_dims);

    std::vector<int64_t> origin_input_dims;
    if (op_desc->HasAttr(ATTR_MBATCH_ORIGIN_INPUT_DIMS) &&
        AttrUtils::GetListInt(op_desc, ATTR_MBATCH_ORIGIN_INPUT_DIMS, origin_input_dims)) {
      model_meta.origin_input_dims.push_back(origin_input_dims);
    } else {
      model_meta.origin_input_dims.push_back(tensor_desc->GetShape().GetDims());
    }

    model_meta.input_desc.push_back(desc);
    model_meta.input_desc_v2.push_back(desc_v2);
  }

  // Build output descriptors
  std::vector<std::string> out_node_name;
  (void)AttrUtils::GetListStr(ge_model, ATTR_MODEL_OUT_NODES_NAME, out_node_name);
  ModelMetaExtraInfo extra_info;

  for (const auto &op_desc : io_nodes.output_ops) {
    const auto out_size = op_desc->GetInputsSize();
    const auto src_name = op_desc->GetSrcName();
    const auto src_index = op_desc->GetSrcIndex();
    GE_ASSERT_TRUE(src_name.size() >= out_size && src_index.size() >= out_size);

    for (size_t i = 0UL; i < out_size; ++i) {
      std::string output_name;
      if (out_size == out_node_name.size()) {
        const bool contains_colon = out_node_name[i].find(':') != std::string::npos;
        output_name = contains_colon ? out_node_name[i] : (out_node_name[i] + ":" + std::to_string(src_index[i]));
      } else {
        output_name =
            std::string("output_") + std::to_string(i) + "_" + src_name[i] + "_" + std::to_string(src_index[i]);
      }

      const auto &tensor_desc = op_desc->GetInputDescPtr(static_cast<uint32_t>(i));
      GE_ASSERT_NOTNULL(tensor_desc);

      ge::Om2TensorDesc desc;
      desc.SetName(output_name);
      desc.SetDataType(tensor_desc->GetDataType());
      desc.SetFormat(tensor_desc->GetFormat());
      desc.SetShape(tensor_desc->GetShape().GetDims());

      int64_t tensor_size = 0;
      if (AttrUtils::GetInt(tensor_desc, ATTR_NAME_SPECIAL_OUTPUT_SIZE, tensor_size) && (tensor_size > 0)) {
        desc.SetSize(static_cast<size_t>(tensor_size));
      } else {
        (void)TensorUtils::GetTensorSizeInBytes(*tensor_desc, tensor_size);
        desc.SetSize(static_cast<size_t>(tensor_size));
      }

      std::vector<std::pair<int64_t, int64_t>> range;
      if (tensor_desc->GetShapeRange(range) == SUCCESS) {
        desc.SetShapeRange(range);
      }

      model_meta.output_desc.push_back(desc);
      model_meta.output_desc_v2.push_back(desc);
    }

    std::vector<std::string> shape_info;
    if (AttrUtils::GetListStr(op_desc, ATTR_NAME_DYNAMIC_OUTPUT_DIMS, shape_info)) {
      for (const auto &s : shape_info) {
        extra_info.dynamic_output_shape.push_back(s);
      }
    }
  }

  GE_ASSERT_SUCCESS(CollectDynamicBatchInfo(io_nodes.case_ops, extra_info));

  model_meta.model_name = ge_model->GetName();
  model_meta.root_graph_name = GetRootGraphName(ge_model);
  int64_t work_size = 0;
  (void)AttrUtils::GetInt(ge_model, ATTR_MODEL_MEMORY_SIZE, work_size);
  model_meta.work_size = static_cast<size_t>(work_size);
  int64_t zero_copy_size = 0;
  (void)AttrUtils::GetInt(ge_model, ATTR_MODEL_ZERO_COPY_MEMORY_SIZE, zero_copy_size);
  model_meta.zero_copy_size = zero_copy_size;
  model_meta.dynamic_batch_info = extra_info.dynamic_batch_info;
  model_meta.dynamic_type = extra_info.dynamic_type;
  model_meta.dynamic_output_shape = extra_info.dynamic_output_shape;
  model_meta.user_designate_shape_order = extra_info.user_designate_shape_order;

  GE_CHK_STATUS_RET(CollectAippMetas(graph, model_meta));

  GELOGI("[OM2] Successfully built model meta");
  return SUCCESS;
}

Status Om2PackageHelper::BuildConstantsData(const GeModelPtr &ge_model, const std::vector<Om2ConstMeta> &const_metas,
                                            gert::Om2ConstantsData &data) {
  GELOGI("[OM2] Begin to build constants data");
  bool has_internal_const = false;
  for (const auto &const_meta : const_metas) {
    if (const_meta.type == "INTERNAL") {
      has_internal_const = true;
      break;
    }
  }

  data.internal_weight_size = has_internal_const ? ge_model->GetWeightSize() : 0U;

  for (const auto &const_meta : const_metas) {
    data.consts.push_back(const_meta);
  }

  if (has_internal_const) {
    const uint8_t *weight_ptr = ge_model->GetWeightData();
    GE_ASSERT_NOTNULL(weight_ptr, "[OM2] Weight data pointer is null");
    data.weight_data = ge::ReadonlyByteBuffer(weight_ptr, ge::ConditionalDeleter{false});
  }

  GELOGI("[OM2] Successfully built constants data, internal_weight_size=%zu, consts count=%zu",
         data.internal_weight_size, data.consts.size());
  return SUCCESS;
}

Status Om2PackageHelper::BuildDebugInfo(const GeModelPtr &ge_model, gert::Om2DebugInfo &debug_info) {
  GELOGI("[OM2] Begin to build debug info");
  const auto &graph = ge_model->GetGraph();
  GE_ASSERT_NOTNULL(graph);

  // Build op_attr_map
  for (const auto &node : graph->GetNodes(graph->GetGraphUnknownFlag())) {
    const auto &op_desc = node->GetOpDesc();
    GE_ASSERT_NOTNULL(op_desc);
    std::vector<std::string> original_op_names;
    if (AttrUtils::GetListStr(op_desc, ATTR_NAME_DATA_DUMP_ORIGIN_OP_NAMES, original_op_names)) {
      std::map<std::string, std::string> op_attrs;
      // Serialize the LIST_STRING value as "[N]value[N]value..." format
      std::string serialized_value;
      for (const auto &op_name : original_op_names) {
        serialized_value += "[" + std::to_string(op_name.size()) + "]" + op_name;
      }
      if (!serialized_value.empty()) {
        op_attrs[ATTR_NAME_DATA_DUMP_ORIGIN_OP_NAMES] = serialized_value;
        debug_info.op_attr_map[op_desc->GetName()] = op_attrs;
      }
    }
  }

  // Build visual json
  GE_ASSERT_SUCCESS(SetOm2CompatibleOmInfoList(ge_model));
  GE_ASSERT_SUCCESS(VisualJsonConverter::SerializeFromGeModel(ge_model, debug_info.visual_json));
  GELOGI("[OM2] Successfully built debug info");
  return SUCCESS;
}

Status Om2PackageHelper::BuildManifest(const GeRootModelPtr &ge_root_model,
                                       std::map<std::string, std::string> &manifest) {
  GELOGI("[OM2] Begin to build manifest");
  manifest[OM2_ARCHIVE_VERSION] = OM2_ARCHIVE_VERSION_VALUE;
  if (ge_root_model != nullptr) {
    manifest[OM2_MODEL_NUM] = std::to_string(ge_root_model->GetSubgraphInstanceNameToModel().size());
  } else {
    manifest[OM2_MODEL_NUM] = "1";
  }
  manifest[OM2_ATC_COMMAND] = domi::GetContext().atc_cmdline;
  GELOGI("[OM2] Successfully built manifest");
  return SUCCESS;
}

Status Om2PackageHelper::BuildOm2ModelData(const GeModelPtr &ge_model, gert::Om2ModelData &model_data,
                                           const GeRootModelPtr &ge_root_model) {
  GE_ASSERT_NOTNULL(ge_model, "[OM2] ge_model is nullptr");

  // Set model-level attrs for OM2 JSON compatibility
  const bool set_atc_cmdline =
      ge::AttrUtils::SetStr(*(ge_model.get()), ATTR_MODEL_ATC_CMDLINE, domi::GetContext().atc_cmdline);
  GE_CHK_BOOL_EXEC(set_atc_cmdline, GELOGE(FAILED, "[OM2] SetStr for atc_cmdline failed."); return FAILED);
  std::string opp_version;
  std::string opp_path;
  (void)PluginManager::GetOppPath(opp_path);
  const std::string version_path = opp_path + "/version.info";
  if ((!PluginManager::GetVersionFromPath(version_path, opp_version)) ||
      (!ge::AttrUtils::SetStr(*(ge_model.get()), ATTR_MODEL_OPP_VERSION, opp_version))) {
    GELOGW("[OM2] Ge model set opp version unsuccessful!");
  }

  std::vector<Om2ConstMeta> const_metas;
  std::vector<Om2VarMeta> var_metas;
  GE_ASSERT_SUCCESS(BuildProgramBody(ge_model, model_data.program_body, const_metas, var_metas));
  GE_ASSERT_SUCCESS(BuildKernelBinaries(ge_model, model_data.kernel_binaries));
  GE_ASSERT_SUCCESS(BuildModelMeta(ge_model, model_data.model_meta));
  GE_ASSERT_SUCCESS(BuildConstantsData(ge_model, const_metas, model_data.constants_data));
  model_data.var_metas = std::move(var_metas);
  const auto compute_graph = ge_model->GetGraph();
  model_data.graph_id = (compute_graph != nullptr) ? compute_graph->GetGraphID() : 0U;
  const auto session_id = GetContext().SessionId();
  auto var_manager = ge::VarManager::Instance(session_id);
  if (var_manager != nullptr) {
    GE_ASSERT_SUCCESS(
        gert::BuildRTVarResource(*var_manager, ge_model->GetGraph(), model_data.var_metas, model_data.rt_var_resource));
  }
  GE_ASSERT_SUCCESS(BuildDebugInfo(ge_model, model_data.debug_info));
  GE_ASSERT_SUCCESS(BuildManifest(ge_root_model, model_data.manifest));

  GELOGI("[OM2] Successfully built Om2ModelData");
  return SUCCESS;
}

REGISTER_MODEL_SAVE_HELPER(OM_FORMAT_OM2, Om2PackageHelper);
}  // namespace ge
