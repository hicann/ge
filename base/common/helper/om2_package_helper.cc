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
#include "base/err_msg.h"
#include "framework/common/helper/om2_package_helper.h"
#include "framework/common/helper/model_save_helper_factory.h"
#include "common/file_constant_utils/file_constant_utils.h"
#include "common/ge_common/ge_types.h"
#include "framework/common/gert_model_data_serialize.h"
#include "framework/common/gert_model_data_deserialize.h"
#include "framework/om2/model_data/om2_package_contants.h"
#include "framework/common/json_file.h"
#include "framework/om2/model_data/gert_model_data.h"
#include "framework/common/gert_model_data_utils.h"
#include "common/om2/codegen/om2_codegen.h"
#include "common/om2/codegen/om2_codegen_utils.h"
#include "framework/omg/omg_inner_types.h"
#include "graph/debug/ge_attr_define.h"
#include "graph_metadef/common/plugin/plugin_manager.h"
#include "graph/utils/type_utils.h"
#include "graph/utils/tensor_utils.h"
#include "graph_metadef/graph/utils/file_utils.h"
#include "graph/custom_op_factory.h"
#include "common/helper/visual_json_converter.h"
#include "graph/ge_context.h"
#include "graph/manager/graph_var_manager.h"
#include "common/helper/om2/rt_var_resource_builder.h"
#include "common/op_so_store/op_so_store_utils.h"
#include "framework/common/gert_model_data_repack.h"

namespace ge {
namespace {
constexpr auto kAttrKernelName = "_kernelname";

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
    (void)dims_info.dims.emplace_back(std::strtol(dim_str.c_str(), nullptr, kAippDecimalRadix));
  }
  return SUCCESS;
}

static Status ParseAippDims(const std::vector<std::string> &dim_strs,
                            std::vector<std::unique_ptr<ge::InputOutputDims>> &dims) {
  for (const auto &s : dim_strs) {
    auto dim_info = std::make_unique<ge::InputOutputDims>();
    GE_CHK_STATUS_RET(ParseAippDimInfo(s, *dim_info), "[Parse][AippDimInfo] failed for: %s", s.c_str());
    dims.push_back(std::move(dim_info));
  }
  return SUCCESS;
}

static Status ExtractAippMetaFromOpDesc(const OpDescPtr &op_desc, const std::map<std::string, uint32_t> &data_index_map,
                                        gert::GertModelDataAippMeta &meta) {
  GELOGD("[OM2] Extract AIPP meta from node: %s", op_desc->GetName().c_str());
  GeAttrValue::NamedAttrs aipp_attr;
  if (ge::AttrUtils::GetNamedAttrs(op_desc, ATTR_NAME_AIPP, aipp_attr)) {
    meta.aipp_config_info = std::make_unique<ge::AippConfigInfo>();
    ConvertAippAttrToConfigInfo(aipp_attr, *meta.aipp_config_info);
  }
  if (meta.aipp_type == ge::DATA_WITH_DYNAMIC_AIPP) {
    const std::string *related_name = ge::AttrUtils::GetStr(op_desc, ATTR_DATA_AIPP_DATA_NAME_MAP);
    if (related_name != nullptr) {
      meta.aipp_data_index = ResolveAippDataIndex(data_index_map, *related_name);
    }
  } else {
    meta.aipp_data_index = gert::kOm2InvalidAippDataIndex;
  }
  std::vector<std::string> aipp_inputs;
  std::vector<std::string> aipp_outputs;
  (void)ge::AttrUtils::GetListStr(op_desc, ATTR_NAME_AIPP_INPUTS, aipp_inputs);
  (void)ge::AttrUtils::GetListStr(op_desc, ATTR_NAME_AIPP_OUTPUTS, aipp_outputs);
  GE_CHK_STATUS_RET(ParseAippDims(aipp_inputs, meta.aipp_input_dims));
  GE_CHK_STATUS_RET(ParseAippDims(aipp_outputs, meta.aipp_output_dims));
  if (!aipp_inputs.empty()) {
    meta.orig_input_info = std::make_unique<ge::OriginInputInfo>();
    ParseOrigInputInfoFromStr(aipp_inputs[0], *meta.orig_input_info);
  }
  return SUCCESS;
}

static Status CollectAippMetas(const ComputeGraphPtr &graph, gert::GertModelDataModelMeta &model_meta) {
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
    if (!model_meta.aipp_infos[input_index]) {
      model_meta.aipp_infos[input_index] = std::make_unique<gert::GertModelDataAippMeta>();
    }
    gert::GertModelDataAippMeta &meta = *model_meta.aipp_infos[input_index];
    meta.aipp_type = aipp_type;
    const Status ret = ExtractAippMetaFromOpDesc(op_desc, data_index_map, meta);
    if (ret != SUCCESS) {
      GELOGE(ret, "[OM2] ExtractAippMetaFromOpDesc failed for node: %s", op_desc->GetName().c_str());
      return ret;
    }
  }
  GELOGI("[OM2] Collected %zu AIPP metas", model_meta.aipp_infos.size());
  return SUCCESS;
}

Status CollectSerializableCustomOps(const std::set<std::string> &used_custom_op_types,
                                    std::vector<std::pair<std::string, PortableOp *>> &serializable_ops) {
  // PortableOp 与非 PortableOp 自定义算子可混合使用：仅 PortableOp 需要携带序列化数据，
  // 非 PortableOp 算子的 kernel 由算子so自行管理，无需序列化数据。
  serializable_ops.reserve(used_custom_op_types.size());
  for (const auto &op_type_str : used_custom_op_types) {
    auto serializable_op = CustomOpFactory::GetCustomOpCommonCapability<PortableOp>(AscendString(op_type_str.c_str()));
    if (serializable_op != nullptr) {
      (void)serializable_ops.emplace_back(op_type_str, serializable_op);
    }
  }
  return SUCCESS;
}

Status SerializeCustomOpToBinary(const std::string &op_type, PortableOp *serializable_op,
                                 std::vector<std::unique_ptr<gert::GertModelDataFile>> &kernel_binaries) {
  if (serializable_op == nullptr) {
    GELOGE(FAILED, "[OM2] serializable custom op is null, op_type:%s", op_type.c_str());
    return FAILED;
  }
  std::vector<uint8_t> buffer;
  const auto ret = serializable_op->Serialize(buffer);
  if (ret != GRAPH_SUCCESS) {
    GELOGE(ret, "[OM2] serialize failed, op_type:%s", op_type.c_str());
    return ret;
  }
  if (buffer.empty()) {
    GELOGW("[OM2] serialized buffer is empty, skip, op_type:%s", op_type.c_str());
    return SUCCESS;
  }
  auto bin_data = new (std::nothrow) uint8_t[buffer.size()];
  if (bin_data == nullptr) {
    GELOGE(FAILED, "[Allocate][Mem]Allocate mem failed");
    return FAILED;
  }
  auto kb = std::make_unique<gert::GertModelDataFile>();
  kb->data = ge::ReadonlyByteBuffer(bin_data, ge::ConditionalDeleter{true});
  kb->data_size = buffer.size();
  GE_ASSERT_EOK(memcpy_s(bin_data, buffer.size(), buffer.data(), buffer.size()));
  const size_t hash_id = std::hash<std::string>{}(std::string(kb->data.get(), kb->data.get() + kb->data_size));
  const auto entry_path = op_type + "_" + std::to_string(hash_id) + "_CustomKernel.bin";
  kb->file_name = gert::GertMakeStr(entry_path);
  kernel_binaries.push_back(std::move(kb));
  return SUCCESS;
}

void CollectTbeKernels(const GeModelPtr &ge_model,
                       std::vector<std::unique_ptr<gert::GertModelDataFile>> &kernel_binaries,
                       std::unordered_set<std::string> &added_kernels) {
  const auto &graph = ge_model->GetGraph();
  const auto &tbe_kernel_store = ge_model->GetTBEKernelStore();
  for (const auto &node : graph->GetNodes(graph->GetGraphUnknownFlag())) {
    std::string kernel_name;
    const auto kernel_name_ptr = AttrUtils::GetStr(node->GetOpDesc(), kAttrKernelName);
    if (kernel_name_ptr != nullptr) {
      kernel_name = *kernel_name_ptr;
    }
    auto kernel_bin = tbe_kernel_store.FindKernel(kernel_name);
    if ((kernel_bin != nullptr) && (added_kernels.count(kernel_name) == 0)) {
      gert::GertModelDataFile kb;
      kb.file_name = gert::GertMakeStr(Om2CodegenUtils::GetKernelNameWithExtension(kernel_name));
      kb.data = ge::ReadonlyByteBuffer(kernel_bin->GetBinData(), ge::ConditionalDeleter{false});
      kb.data_size = kernel_bin->GetBinDataSize();
      kernel_binaries.push_back(std::make_unique<gert::GertModelDataFile>(std::move(kb)));
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
        gert::GertModelDataFile kb;
        kb.file_name = gert::GertMakeStr(Om2CodegenUtils::GetKernelNameWithExtension(atomic_kernel_name));
        kb.data = ge::ReadonlyByteBuffer(atomic_kernel_bin->GetBinData(), ge::ConditionalDeleter{false});
        kb.data_size = atomic_kernel_bin->GetBinDataSize();
        kernel_binaries.push_back(std::make_unique<gert::GertModelDataFile>(std::move(kb)));
        (void)added_kernels.insert(atomic_kernel_name);
      }
    }
  }
}

void CollectCustAicpuKernels(const GeModelPtr &ge_model,
                             std::vector<std::unique_ptr<gert::GertModelDataFile>> &kernel_binaries,
                             std::unordered_set<std::string> &added_kernels) {
  const auto &graph = ge_model->GetGraph();
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
        gert::GertModelDataFile kb;
        kb.file_name = gert::GertMakeStr(std::to_string(hash_id) + "_CustAicpuKernel.o");
        kb.data = ge::ReadonlyByteBuffer(kernel_bin->GetBinData(), ge::ConditionalDeleter{false});
        kb.data_size = kernel_bin->GetBinDataSize();
        kernel_binaries.push_back(std::make_unique<gert::GertModelDataFile>(std::move(kb)));
        (void)added_kernels.insert(cust_aicpu_kernel->GetName());
      }
    }
  }
}

Status BuildModelInputDescs(const ModelIoNodes &io_nodes, gert::GertModelDataModelMeta &model_meta) {
  for (const auto &[index, op_desc] : io_nodes.input_ops) {
    (void)index;
    const auto &tensor_desc = op_desc->GetInputDescPtr(0);
    GE_ASSERT_NOTNULL(tensor_desc);

    gert::GertTensorDesc desc = gert::MakeGertTensorDesc(op_desc->GetName(), tensor_desc->GetDataType(),
                                                         tensor_desc->GetFormat(), tensor_desc->GetShape().GetDims());

    int64_t input_size = 0;
    const auto output_desc = op_desc->GetOutputDescPtr(0U);
    if ((output_desc != nullptr) && AttrUtils::GetInt(*output_desc, ATTR_NAME_SPECIAL_INPUT_SIZE, input_size) &&
        (input_size > 0)) {
      desc.size = static_cast<size_t>(input_size);
    } else {
      GE_CHK_STATUS_RET(TensorUtils::GetSize(*tensor_desc, input_size), "[Get][InputSize] failed for op: %s.",
                        op_desc->GetName().c_str());
      desc.size = static_cast<size_t>(input_size);
    }

    std::vector<std::pair<int64_t, int64_t>> range;
    if (tensor_desc->GetShapeRange(range) == SUCCESS) {
      desc.shape_range = range;
    }

    gert::GertTensorDesc desc_v2 = gert::MakeGertTensorDesc(desc);
    std::vector<int64_t> model_input_dims;
    if (op_desc->HasAttr(ATTR_NAME_INPUT_DIMS)) {
      (void)AttrUtils::GetListInt(op_desc, ATTR_NAME_INPUT_DIMS, model_input_dims);
    } else {
      model_input_dims = tensor_desc->GetShape().GetDims();
    }
    desc_v2.shape = model_input_dims;

    std::vector<int64_t> origin_input_dims;
    if (op_desc->HasAttr(ATTR_MBATCH_ORIGIN_INPUT_DIMS) &&
        AttrUtils::GetListInt(op_desc, ATTR_MBATCH_ORIGIN_INPUT_DIMS, origin_input_dims)) {
      model_meta.origin_input_dims.push_back(origin_input_dims);
    } else {
      model_meta.origin_input_dims.push_back(tensor_desc->GetShape().GetDims());
    }

    model_meta.input_desc.push_back(std::move(desc));
    model_meta.input_desc_v2.push_back(std::move(desc_v2));
  }
  return SUCCESS;
}

Status BuildSingleOutputDesc(const OpDescPtr &op_desc, const std::vector<std::string> &out_node_name, const size_t i,
                             gert::GertModelDataModelMeta &model_meta) {
  const auto out_size = op_desc->GetInputsSize();
  const auto src_name = op_desc->GetSrcName();
  const auto src_index = op_desc->GetSrcIndex();
  std::string output_name;
  if (out_size == out_node_name.size()) {
    const bool contains_colon = out_node_name[i].find(':') != std::string::npos;
    output_name = contains_colon ? out_node_name[i] : (out_node_name[i] + ":" + std::to_string(src_index[i]));
  } else {
    output_name = std::string("output_") + std::to_string(i) + "_" + src_name[i] + "_" + std::to_string(src_index[i]);
  }

  const auto &tensor_desc = op_desc->GetInputDescPtr(static_cast<uint32_t>(i));
  GE_ASSERT_NOTNULL(tensor_desc);

  gert::GertTensorDesc desc = gert::MakeGertTensorDesc(output_name, tensor_desc->GetDataType(),
                                                       tensor_desc->GetFormat(), tensor_desc->GetShape().GetDims());

  int64_t tensor_size = 0;
  if (AttrUtils::GetInt(tensor_desc, ATTR_NAME_SPECIAL_OUTPUT_SIZE, tensor_size) && (tensor_size > 0)) {
    desc.size = static_cast<size_t>(tensor_size);
  } else {
    (void)TensorUtils::GetTensorSizeInBytes(*tensor_desc, tensor_size);
    desc.size = static_cast<size_t>(tensor_size);
  }

  std::vector<std::pair<int64_t, int64_t>> range;
  if (tensor_desc->GetShapeRange(range) == SUCCESS) {
    desc.shape_range = range;
  }

  gert::GertTensorDesc desc_v2 = gert::MakeGertTensorDesc(desc);
  model_meta.output_desc.push_back(std::move(desc));
  model_meta.output_desc_v2.push_back(std::move(desc_v2));
  return SUCCESS;
}

Status BuildModelOutputDescs(const GeModelPtr &ge_model, const ModelIoNodes &io_nodes, ModelMetaExtraInfo &extra_info,
                             gert::GertModelDataModelMeta &model_meta) {
  std::vector<std::string> out_node_name;
  (void)AttrUtils::GetListStr(ge_model, ATTR_MODEL_OUT_NODES_NAME, out_node_name);

  for (const auto &op_desc : io_nodes.output_ops) {
    const auto out_size = op_desc->GetInputsSize();
    const auto src_name = op_desc->GetSrcName();
    const auto src_index = op_desc->GetSrcIndex();
    GE_ASSERT_TRUE(src_name.size() >= out_size && src_index.size() >= out_size);

    for (size_t i = 0UL; i < out_size; ++i) {
      GE_CHK_STATUS_RET_NOLOG(BuildSingleOutputDesc(op_desc, out_node_name, i, model_meta));
    }

    std::vector<std::string> shape_info;
    if (AttrUtils::GetListStr(op_desc, ATTR_NAME_DYNAMIC_OUTPUT_DIMS, shape_info)) {
      for (const auto &s : shape_info) {
        extra_info.dynamic_output_shape.push_back(s);
      }
    }
  }
  return SUCCESS;
}

Status FillModelMetaScalars(const GeModelPtr &ge_model, const ModelIoNodes &io_nodes, ModelMetaExtraInfo &extra_info,
                            gert::GertModelDataModelMeta &model_meta) {
  GE_ASSERT_SUCCESS(CollectDynamicBatchInfo(io_nodes.case_ops, extra_info));

  model_meta.model_name = gert::GertMakeStr(ge_model->GetName());
  int64_t work_size = 0;
  (void)AttrUtils::GetInt(ge_model, ATTR_MODEL_MEMORY_SIZE, work_size);
  model_meta.work_size = static_cast<uint64_t>(work_size);
  int64_t zero_copy_size = 0;
  (void)AttrUtils::GetInt(ge_model, ATTR_MODEL_ZERO_COPY_MEMORY_SIZE, zero_copy_size);
  model_meta.zero_copy_size = zero_copy_size;
  model_meta.dynamic_batch_info = extra_info.dynamic_batch_info;
  model_meta.dynamic_type = extra_info.dynamic_type;
  const std::vector<std::string> dynamic_output_shape = extra_info.dynamic_output_shape;
  for (const auto &s : dynamic_output_shape) {
    model_meta.dynamic_output_shape.emplace_back(gert::GertMakeStr(s));
  }
  const std::vector<std::string> user_designate_shape_order = extra_info.user_designate_shape_order;
  for (const auto &s : user_designate_shape_order) {
    model_meta.user_designate_shape_order.emplace_back(gert::GertMakeStr(s));
  }
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

  gert::GertModelData model_data;
  GE_ASSERT_SUCCESS(BuildOm2ModelData(ge_model, model_data, ge_root_model));

  const std::string writer_path = (!is_offline_ && !ge_model->GetName().empty()) ? ge_model->GetName() : output_file;
  GE_ASSERT_SUCCESS(gert::SerializeGertModelData(model_data, model, is_offline_, writer_path));

  GELOGI("[OM2] Successfully created OM2 model");
  return SUCCESS;
}

void Om2PackageHelper::SetSaveMode(const bool val) {
  is_offline_ = val;
}

Status Om2PackageHelper::RelocateExternalWeights(const std::string &output_file_name, const ModelBufferData &model,
                                                 ModelBufferData &relocated_model, bool &relocated) {
  relocated = false;
  std::map<std::string, std::string> old_file_to_new_file;
  gert::GertBuffer relocated_buffer;
  GE_ASSERT_SUCCESS(gert::RepackOm2ModelData(output_file_name, model.data.get(), model.length, relocated_buffer,
                                             old_file_to_new_file));
  if (old_file_to_new_file.empty()) {
    return SUCCESS;
  }
  relocated_model.data = relocated_buffer.data;
  relocated_model.length = relocated_buffer.length;
  GE_ASSERT_SUCCESS(FileConstantUtils::MoveExternalWeightFiles(old_file_to_new_file));
  relocated = true;
  return SUCCESS;
}

Status Om2PackageHelper::ExtractVisualJson(const void *model_data, size_t model_len, std::string &json_out) {
  const auto report_extract_failed = [](const char *reason) {
    (void)REPORT_PREDEFINED_ERR_MSG("E10059", std::vector<const char *>({"stage", "reason"}),
                                    std::vector<const char *>({"ExtractVisualJson", reason}));
    GELOGE(FAILED, "[OM2] ExtractVisualJson failed. Reason: %s", reason);
  };

  GE_ASSERT_NOTNULL(model_data, "[OM2] model_data is nullptr");
  GE_ASSERT_TRUE(model_len > 0U, "[OM2] model_len is 0");

  gert::GertModelData om2_data;
  const uint32_t deserialize_ret =
      gert::DeserializeGertVisualJson(static_cast<const uint8_t *>(model_data), model_len, &om2_data);
  const bool visual_json_valid = (!om2_data.models.empty() && (om2_data.models[0]->debug != nullptr) &&
                                  (om2_data.models[0]->debug->visual_json != nullptr));
  if ((deserialize_ret != 0U) || !visual_json_valid) {
    report_extract_failed("Failed to extract visual JSON from OM2 archive.");
    return FAILED;
  }

  json_out = gert::GertGetStr(om2_data.models[0]->debug->visual_json);
  GELOGI("[OM2] Extracted visual JSON, size:%zu", json_out.size());
  return SUCCESS;
}

Status Om2PackageHelper::BuildProgramBody(const GeModelPtr &ge_model, gert::GertModelData &model_data,
                                          gert::GertModelDataModel &unit) {
  GELOGI("[OM2] Begin to build program body");
  auto &body = *unit.runtime;
  Om2Codegen codegen;
  GE_ASSERT_SUCCESS(codegen.Om2CodegenAndCompile(ge_model, model_data, unit));
  GE_ASSERT_TRUE(!body.source_artifacts.empty());

  for (const auto &artifact : body.source_artifacts) {
    if (std::string(gert::GertGetStr(artifact.file_name)).find(".so") != std::string::npos) {
      body.so_artifact.file_name = gert::GertMakeStr(artifact.file_name);
      body.so_artifact.data = gert::GertMakeFileData(artifact.data.get(), artifact.data_size);
      body.so_artifact.data_size = artifact.data_size;
      break;
    }
  }

  auto &const_metas = unit.constants_config->consts;
  const auto var_metas_size = (unit.variables_config != nullptr) ? unit.variables_config->var_metas.size() : 0UL;
  GELOGI("[OM2] Successfully built program body, artifacts count=%zu, const_metas count=%zu, var_metas count=%zu",
         body.source_artifacts.size(), const_metas.size(), var_metas_size);
  return SUCCESS;
}

Status Om2PackageHelper::BuildCustomSharedLibs(const GeRootModelPtr &ge_root_model, gert::GertModelData &model_data) {
  GELOGI("[OM2] Begin to build custom kernel shared libraries");
  GE_ASSERT_NOTNULL(ge_root_model);
  if (!OpSoStoreUtils::IsSoBinType(ge_root_model->GetSoInOmFlag(), SoBinType::kCustomOp)) {
    return SUCCESS;
  }
  GE_ASSERT_SUCCESS(ReadCustomOpSoToBuffer(ge_root_model->GetCustomOpSoSet(), model_data.custom_ops->libraries));
  GELOGI("[OM2] Save %zu custom op so to OpSoStore success.", ge_root_model->GetCustomOpSoSet().size());
  return SUCCESS;
}

Status Om2PackageHelper::ReadCustomOpSoToBuffer(
    const std::unordered_set<std::string> &ops_so_set,
    std::vector<std::unique_ptr<gert::GertModelDataFile>> &shared_lib_binaries) {
  for (const auto &op_so : ops_so_set) {
    uint32_t bin_len = 0U;
    auto op_so_bin = GetBinDataFromFile(op_so, bin_len);
    GE_ASSERT_NOTNULL(op_so_bin, "open so fail, path=%s", op_so.c_str());
    const auto &pos = op_so.find_last_of("/");
    GE_ASSERT_TRUE(pos != std::string::npos);
    const auto &so_name = op_so.substr(pos + 1UL);
    const size_t hash_id = std::hash<std::string>{}(std::string(op_so_bin.get(), op_so_bin.get() + bin_len));
    auto kb = std::make_unique<gert::GertModelDataFile>();
    kb->file_name = gert::GertMakeStr(std::to_string(bin_len) + "_" + std::to_string(hash_id) + "_" + so_name);
    kb->data = ge::ReadonlyByteBuffer(reinterpret_cast<uint8_t *>(op_so_bin.release()), ge::ConditionalDeleter{true});
    kb->data_size = bin_len;
    (void)shared_lib_binaries.emplace_back(std::move(kb));

    GELOGD("[OM2] Serialized custom op so '%s', bin size:%zu", so_name.c_str(), bin_len);
  }
  return SUCCESS;
}

Status Om2PackageHelper::CollectUsedCustomOpTypes(const GeRootModelPtr &ge_root_model,
                                                  std::set<std::string> &used_custom_op_types) {
  if (ge_root_model->GetRootGraph() != nullptr) {
    const auto &root_graph = ge_root_model->GetRootGraph();
    for (const auto &node : root_graph->GetAllNodes()) {
      const std::string op_type = node->GetType();
      if (CustomOpFactory::IsExistOp(AscendString(op_type.c_str()))) {
        (void)used_custom_op_types.insert(op_type);
      }
    }
  }

  // subgraph_instance_name_to_model_ 中的 GeModel 可能持有独立的 ComputeGraph 对象，
  // 这些子图未必通过 AddSubgraph 挂入 root_graph 的子图树，因此无法被上方 GetAllNodes() 遍历到。
  // 典型场景：编译分区后各分区独立持有自己的 ComputeGraph，或反序列化时每个 GeModel 单独还原图对象。
  // 此处需额外遍历，以确保这类游离子图中的自定义算子也被收集到。
  const auto &subgraph_map = ge_root_model->GetSubgraphInstanceNameToModel();
  for (const auto &subgraph_pair : subgraph_map) {
    const auto &ge_model = subgraph_pair.second;
    if (ge_model == nullptr || ge_model->GetGraph() == nullptr) {
      continue;
    }
    const auto &graph = ge_model->GetGraph();
    if (graph == ge_root_model->GetRootGraph()) {
      continue;
    }
    for (const auto &node : graph->GetAllNodes()) {
      const std::string op_type = node->GetType();
      if (CustomOpFactory::IsExistOp(AscendString(op_type.c_str()))) {
        (void)used_custom_op_types.insert(op_type);
      }
    }
  }
  return SUCCESS;
}

Status Om2PackageHelper::BuildCustomKernelBinaries(const GeRootModelPtr &ge_root_model,
                                                   gert::GertModelData &model_data) {
  GELOGI("[OM2] Begin to build custom kernel binaries");
  auto &kernel_binaries = model_data.custom_ops->binaries;

  std::set<std::string> used_custom_op_types;
  GE_ASSERT_SUCCESS(CollectUsedCustomOpTypes(ge_root_model, used_custom_op_types));
  if (used_custom_op_types.empty()) {
    GELOGI("[OM2] No custom ops used in graph, skip building custom kernels.");
    return SUCCESS;
  }

  std::vector<std::pair<std::string, PortableOp *>> serializable_ops;
  GE_CHK_STATUS_RET_NOLOG(CollectSerializableCustomOps(used_custom_op_types, serializable_ops));
  for (const auto &[op_type, serializable_op] : serializable_ops) {
    GE_CHK_STATUS_RET_NOLOG(SerializeCustomOpToBinary(op_type, serializable_op, kernel_binaries));
  }
  GELOGI("[OM2] Successfully built custom kernel binaries, count=%zu", kernel_binaries.size());
  return SUCCESS;
}

Status Om2PackageHelper::BuildKernelBinaries(const GeModelPtr &ge_model, gert::GertModelData &model_data) {
  GELOGI("[OM2] Begin to build kernel binaries");
  const auto &graph = ge_model->GetGraph();
  GE_ASSERT_NOTNULL(graph);
  auto &kernel_binaries = model_data.kernels->binaries;
  std::unordered_set<std::string> added_kernels;

  CollectTbeKernels(ge_model, kernel_binaries, added_kernels);
  CollectCustAicpuKernels(ge_model, kernel_binaries, added_kernels);
  GELOGI("[OM2] Successfully built kernel binaries, count=%zu", kernel_binaries.size());
  return SUCCESS;
}

Status Om2PackageHelper::BuildModelMeta(const GeModelPtr &ge_model, gert::GertModelDataModel &unit) {
  GELOGI("[OM2] Begin to build model meta");
  const auto &graph = ge_model->GetGraph();
  GE_ASSERT_NOTNULL(graph);
  gert::GertModelDataModelMeta &model_meta = *unit.model_meta;

  ModelIoNodes io_nodes;
  GE_ASSERT_SUCCESS(CollectModelIoNodes(graph, io_nodes));
  ModelMetaExtraInfo extra_info;

  GE_CHK_STATUS_RET_NOLOG(BuildModelInputDescs(io_nodes, model_meta));
  GE_CHK_STATUS_RET_NOLOG(BuildModelOutputDescs(ge_model, io_nodes, extra_info, model_meta));
  GE_CHK_STATUS_RET_NOLOG(FillModelMetaScalars(ge_model, io_nodes, extra_info, model_meta));
  GE_CHK_STATUS_RET(CollectAippMetas(graph, model_meta));

  GELOGI("[OM2] Successfully built model meta");
  return SUCCESS;
}

Status Om2PackageHelper::BuildConstantsData(const GeModelPtr &ge_model, gert::GertModelDataModel &unit,
                                            std::unique_ptr<gert::GertModelDataFile> &weight_slot,
                                            const size_t model_index) {
  GELOGI("[OM2] Begin to build constants data");
  gert::GertModelDataConstantsConfig &config = *unit.constants_config;
  auto &const_metas = config.consts;

  bool has_internal_const = false;
  for (const auto &const_meta : const_metas) {
    if (std::string(gert::GertGetStr(const_meta->type)) == "INTERNAL") {
      has_internal_const = true;
      break;
    }
  }

  config.internal_weight_size = has_internal_const ? ge_model->GetWeightSize() : 0U;
  if (has_internal_const) {
    const uint8_t *weight_ptr = ge_model->GetWeightData();
    GE_ASSERT_NOTNULL(weight_ptr, "[OM2] Weight data pointer is null");
    weight_slot = std::make_unique<gert::GertModelDataFile>();
    weight_slot->data_size = static_cast<uint64_t>(ge_model->GetWeightSize());
    weight_slot->file_name =
        gert::GertMakeStr(gert::FormatOm2Path("%s%zu", gert::OM2_CONSTANTS_FILE_PREFIX, model_index));
    weight_slot->data = ge::ReadonlyByteBuffer(weight_ptr, ge::ConditionalDeleter{false});
  }

  GELOGI("[OM2] Successfully built constants data, internal_weight_size=%zu, consts count=%zu",
         config.internal_weight_size, config.consts.size());
  return SUCCESS;
}

Status Om2PackageHelper::BuildDebugInfo(const GeModelPtr &ge_model, gert::GertModelDataModel &unit) {
  GELOGI("[OM2] Begin to build debug info");
  const auto &graph = ge_model->GetGraph();
  GE_ASSERT_NOTNULL(graph);
  gert::GertModelDataDebug &debug_info = *unit.debug;

  auto op_attr_object = JsonFile::json::object();
  for (const auto &node : graph->GetNodes(graph->GetGraphUnknownFlag())) {
    const auto &op_desc = node->GetOpDesc();
    GE_ASSERT_NOTNULL(op_desc);
    std::vector<std::string> original_op_names;
    if (AttrUtils::GetListStr(op_desc, ATTR_NAME_DATA_DUMP_ORIGIN_OP_NAMES, original_op_names) &&
        !original_op_names.empty()) {
      auto attr_value_object = JsonFile::json::object();
      attr_value_object["type"] = "LIST_STRING";
      attr_value_object["value"] = original_op_names;

      auto op_attr_entry = JsonFile::json::object();
      op_attr_entry[ATTR_NAME_DATA_DUMP_ORIGIN_OP_NAMES] = attr_value_object;

      op_attr_object[op_desc->GetName()] = op_attr_entry;
    }
  }
  const JsonFile op_attr_json(op_attr_object);
  unit.op_attr_json = gert::GertMakeStr(op_attr_json.Dump());

  GE_ASSERT_SUCCESS(SetOm2CompatibleOmInfoList(ge_model));
  std::string visual_json;
  GE_ASSERT_SUCCESS(VisualJsonConverter::SerializeFromGeModel(ge_model, visual_json));
  debug_info.visual_json = gert::GertMakeStr(visual_json);
  GELOGI("[OM2] Successfully built debug info");
  return SUCCESS;
}

Status Om2PackageHelper::BuildManifest(gert::GertModelData &model_data) {
  GELOGI("[OM2] Begin to build manifest");
  gert::GertModelDataManifest &manifest = *model_data.manifest;
  manifest.compatibility.compiler_version = gert::GertMakeStr(gert::GERT_EXECUTOR_VERSION);
  manifest.model_num = static_cast<uint64_t>(model_data.models.size());
  manifest.atc_command = gert::GertMakeStr(domi::GetContext().atc_cmdline);
  GELOGI("[OM2] Successfully built manifest");
  return SUCCESS;
}

Status Om2PackageHelper::BuildOm2ModelData(const GeModelPtr &ge_model, gert::GertModelData &model_data,
                                           const GeRootModelPtr &ge_root_model) {
  GE_ASSERT_NOTNULL(ge_model, "[OM2] ge_model is nullptr");

  auto &unit = *model_data.models.emplace_back(std::make_unique<gert::GertModelDataModel>());
  unit.runtime = std::make_unique<gert::GertModelDataRuntime>();
  unit.model_meta = std::make_unique<gert::GertModelDataModelMeta>();
  unit.constants_config = std::make_unique<gert::GertModelDataConstantsConfig>();
  unit.debug = std::make_unique<gert::GertModelDataDebug>();
  // 入口统一分配目录聚合结构（多次构建追加模型时仅首次分配）
  gert::InitGertModelData(model_data);
  model_data.constants->constants_data.emplace_back();
  model_data.manifest = std::make_unique<gert::GertModelDataManifest>();

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

  GE_ASSERT_SUCCESS(BuildCustomKernelBinaries(ge_root_model, model_data));
  GE_ASSERT_SUCCESS(BuildCustomSharedLibs(ge_root_model, model_data));
  GE_ASSERT_SUCCESS(BuildProgramBody(ge_model, model_data, unit));
  GE_ASSERT_SUCCESS(BuildKernelBinaries(ge_model, model_data));
  GE_ASSERT_SUCCESS(BuildModelMeta(ge_model, unit));
  GE_ASSERT_SUCCESS(BuildConstantsData(ge_model, unit, model_data.constants->constants_data[0], 0UL));

  const auto compute_graph = ge_model->GetGraph();
  if (unit.variables_config == nullptr) {
    unit.variables_config = std::make_unique<gert::GertModelDataVariablesConfig>();
  }
  unit.variables_config->graph_id = (compute_graph != nullptr) ? compute_graph->GetGraphID() : 0U;
  const auto session_id = GetContext().SessionId();
  auto var_manager = ge::VarManager::Instance(session_id);
  if (var_manager != nullptr) {
    GE_ASSERT_SUCCESS(gert::BuildRTVarResource(*var_manager, ge_model->GetGraph(), unit.variables_config->var_metas,
                                               unit.variables_config->entries));
  }
  GE_ASSERT_SUCCESS(BuildDebugInfo(ge_model, unit));
  GE_ASSERT_SUCCESS(BuildManifest(model_data));

  GELOGI("[OM2] Successfully built GertModelData");
  return SUCCESS;
}

REGISTER_MODEL_SAVE_HELPER(OM_FORMAT_OM2, Om2PackageHelper);
}  // namespace ge
