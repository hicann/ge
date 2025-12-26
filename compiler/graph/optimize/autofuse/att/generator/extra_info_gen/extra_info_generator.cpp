/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2024 All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "extra_info_gen/extra_info_generator.h"
#include <set>
#include <iostream>
#include <regex>
#include "common/checker.h"
#include "code_printer.h"
#include "generator_utils/tilingdata_gen_utils.h"
#include "tiling_data_gen/tiling_data_generator.h"
#include "gen_model_info/api_tiling_gen/gen_api_tiling.h"

namespace att {
// ------------------------------以下为tilingfunc实现------------------------------
namespace {
std::set<uint32_t> GetUsedInputIndexes(const std::map<std::string, std::map<uint32_t, std::vector<int64_t>>> &axis_map,
                                       bool retain_first_shape = false) {
  std::set<uint32_t> ret;
  for (const auto &pair : axis_map) {
    for (const auto &shape : pair.second) {
      for (const auto item : shape.second) {
        if (item < 0) {
          ret.insert(shape.first);
          break;
        }
      }
      if (retain_first_shape) {
        break;
      }
    }
  }
  return ret;
}
    
std::string GetIdx(int32_t input_idx, int32_t axis_idx) {
  if (axis_idx >= 0) {
    return std::to_string(axis_idx);
  } else {
    return "input" + std::to_string(input_idx) + "_size - " + std::to_string(-axis_idx);
  }
}

std::string ConstructAttLog(std::string var_name) {
  return "    OP_LOGD(OP_NAME, \"Set " + var_name + " to %u.\", tiling_data.get_" + var_name + "());";
}

std::string GetInputSize(const std::set<uint32_t> &used_input_indexes) {
  std::string ret;
  for (const auto &idx : used_input_indexes) {
    ret += "    uint64_t input" + std::to_string(idx) + "_size = context->GetInputShape(" + std::to_string(idx) +
           ")->GetStorageShape().GetDimNum();\n";
  }
  return ret;
}

std::string GetCurSize(uint64_t input_idx, std::vector<int64_t> idx_interval) {
  std::string ret;
  if (idx_interval.size() == 1u) {
    ret += "    cur_size = context->GetInputShape(" + std::to_string(input_idx) + ")->GetStorageShape().GetDim(" +
           GetIdx(input_idx, idx_interval[0]) + ");\n";
  } else {
    ret += "    cur_size = 1;\n";
    ret += "    for (size_t i = " + GetIdx(input_idx, idx_interval[0]) + "; i <= " + GetIdx(input_idx, idx_interval[1]) +
           "; i++) {\n";
    ret += "      cur_size *= context->GetInputShape(" + std::to_string(input_idx) + ")->GetStorageShape().GetDim(i);\n";
    ret += "    }\n";
  }
  return ret;
}

bool SameSymbol(int32_t left_value, int32_t right_value) {
  int32_t l_sign = left_value >= 0 ? 1 : -1;
  int32_t r_sign = right_value >= 0 ? 1 : -1;
  return (l_sign * r_sign) > 0;
}

std::string GetInputIndex(int64_t axis_idx) {
  if (axis_idx >= 0) {
    return std::to_string(axis_idx);
  } else {
    return "input_size - " + std::to_string(-axis_idx);
  }
}
}

bool CheckFullLeftCover(int32_t value, std::vector<std::vector<int64_t>> intervals) {
  if (value == 0) {
    return true;
  }
  std::vector<bool> blank = std::vector<bool>(value);
  for (int32_t i = 0; i < value; ++i) {
    blank[static_cast<size_t>(i)] = false;
  }
  for (const auto interval : intervals) {
    if (interval.size() == 2u && interval[0] >= 0 && interval[1] >= 0) {
      for (int32_t i = interval[0]; i <= interval[1]; ++i) {
        if (i < value) {
          blank[static_cast<size_t>(i)] = true;
        }
      }
    } else if (interval.size() == 1u && interval[0] >= 0) {
      if (interval[0] >= 0) {
        if (interval[0] < value) {
          blank[static_cast<size_t>(interval[0])] = true;
        }
      }
    }
  }
  for (int32_t i = 0; i < value; ++i) {
    if (!blank[static_cast<size_t>(i)]) {
      return false;
    }
  }
  return true;
}

bool CheckFullRightCover(int32_t value, std::vector<std::vector<int64_t>> intervals) {
  if (value == -1) {
    return true;
  }
  std::vector<std::vector<int64_t>> reverse_intervals;
  for (const auto interval : intervals) {
    if (interval.size() == 2u && interval[0] < 0 && interval[1] < 0) {
      reverse_intervals.push_back({-interval[1] - 1, -interval[0] - 1});
    } else if (interval.size() == 1u && interval[0] < 0) {
      reverse_intervals.push_back({-interval[0] - 1});
    }
  }
  return CheckFullLeftCover(-value - 1, reverse_intervals);
}

bool RequireCoverCheck(std::vector<std::vector<int64_t>> intervals) {
  int32_t left_value = -1;
  int32_t right_value = 1;
  for (const auto interval : intervals) {
    if (interval.size() == 2u && interval[0] >= 0 && interval[1] < 0) {
      if (left_value == -1) {
        left_value = interval[0];
      } else {
        left_value = left_value < interval[0] ? left_value : interval[0];
      }
      if (right_value == 1) {
        right_value = interval[1];
      } else {
        right_value = right_value > interval[-1] ? right_value : interval[1];
      }
    }
  }
  if (left_value >= 0 && right_value < 0) {
    if (CheckFullLeftCover(left_value, intervals) && CheckFullRightCover(right_value, intervals)) {
      return false;
    }
  }
  return true;
}

std::string GenCheckInputCoverFunc(uint32_t input_idx, std::vector<std::vector<int64_t>> intervals) {
  std::string codes;
  if (RequireCoverCheck(intervals)) {
    codes += "  bool CheckInput" + std::to_string(input_idx) + "Cover(gert::TilingContext *context) {\n";
    codes += "    return true;\n";
    codes += "    uint64_t input_size = context->GetInputShape(" + std::to_string(input_idx) +
           ")->GetStorageShape().GetDimNum();\n";
    codes += "    uint32_t cover_cnt = 0;\n";
    codes += "    bool* axis_map = bool_space_;\n";
    codes += "    for (uint32_t i = 0; i < input_size; ++i) {\n";
    codes += "      axis_map[i] = false;\n";
    codes += "    }\n";
    for (const auto interval : intervals) {
      if (interval.size() == 1u) {
        codes += "    axis_map[" + GetInputIndex(interval[0]) + "] = true;\n";
      } else {
        codes +=
            "    for (uint32_t i = " + GetInputIndex(interval[0]) + "; i <= " + GetInputIndex(interval[1]) + "; ++i) {\n";
        codes += "      axis_map[i] = true;\n";
        codes += "    }\n";
      }
    }
    codes += "    for (uint32_t i = 0; i < input_size; ++i) {\n";
    codes += "      cover_cnt += axis_map[i] ? 1 : 0;\n";
    codes += "    }\n";
    codes += "    return cover_cnt == input_size;\n";
    codes += "  }\n";
  }
  return codes;
}

inline std::vector<std::string> GetVarsValidCond(
    const std::vector<std::pair<Expr, std::pair<int64_t, int64_t>>> &vars_range) {
  std::vector<std::string> var_cond;
  for (const auto &var : vars_range) {
    if ((var.second.first != -1) && (var.second.first <= var.second.second)) {
      auto var_name = "tiling_data.get_" + Str(var.first) + "()";
      auto valid_cond = "(" + var_name + " < " + std::to_string(var.second.first) + ") || (" + var_name + " > " +
                        std::to_string(var.second.second) + ")";
      var_cond.emplace_back(valid_cond);
    }
  }
  return var_cond;
}

bool IsValidVariableName(const std::string& name) {
  // 正则表达式模式：变量名必须以字母或下划线开始，后面可以跟字母、数字或下划线
  std::regex pattern("^[a-zA-Z_][a-zA-Z0-9_]*$");
  return std::regex_match(name, pattern);
}

ge::Status ExtraInfoGenerator::WriteInputNumParam(const ArgsManager &args_manager, std::vector<std::string> &def_func) const {
  uint32_t var_num = args_manager.GetInputAtts().size();
  std::string input_num = "   input_num_ = " + std::to_string(var_num) + ";\n";
  def_func.emplace_back(input_num);
  return ge::SUCCESS;
}

ge::Status ExtraInfoGenerator::WriteInputDtypeParam(const ArgsManager &args_manager, std::vector<std::string> &def_func) const {
  const auto &input_atts = args_manager.GetInputAtts();
  std::string input_dtype;
  for (const auto &input_att : input_atts) {
    input_dtype += "   input_dtype_[" + std::to_string(input_att.first) +
                   "] = " + std::to_string(input_att.second.data_type) + ";\n";
  }
  def_func.emplace_back(input_dtype);
  return ge::SUCCESS;
}

ge::Status ExtraInfoGenerator::WriteInputFormatParam(const ArgsManager &args_manager, std::vector<std::string> &def_func) const {
  const auto &input_atts = args_manager.GetInputAtts();
  std::string input_format;
  for (const auto &input_att : input_atts) {
    input_format += "   input_format_[" + std::to_string(input_att.first) +
                    "] = " + std::to_string(input_att.second.format) + ";\n";
  }
  def_func.emplace_back(input_format);
  return ge::SUCCESS;
}

void UpdateMinDim(std::map<uint32_t, uint32_t> &min_dim, uint32_t idx, int32_t value) {
  uint32_t real_idx = value >= 0 ? (value + 1) : (-value);
  auto iter = min_dim.find(idx);
  if (min_dim.end() != iter) {
    min_dim[idx] = real_idx > min_dim[idx] ? real_idx : min_dim[idx];
  } else {
    min_dim[idx] = real_idx;
  }
}

void UpdateDim(std::map<uint32_t, uint32_t> &max_dim, std::map<uint32_t, uint32_t> &min_dim, uint32_t idx, int32_t value, bool update_max) {
  uint32_t cur_val = value;
  if (update_max) {
    if (max_dim.find(idx) != max_dim.end()) {
      cur_val = max_dim[idx] < cur_val ? max_dim[idx] : cur_val;
    }
    max_dim[idx] = cur_val;
  } else {
    if (min_dim.find(idx) != min_dim.end()) {
      cur_val = min_dim[idx] > cur_val ? min_dim[idx] : cur_val;
    }
    min_dim[idx] = cur_val;
  }
}

ge::Status GenDimMap(const std::map<std::string, std::map<uint32_t, std::vector<int64_t>>> &axis_map,
                     std::map<uint32_t, uint32_t> &max_dim, std::map<uint32_t, uint32_t> &min_dim) {
  uint32_t cur_val;
  std::map<uint32_t, std::vector<std::pair<int32_t, int32_t>>> idx_map;
  for (const auto pair : axis_map) {
    for (const auto shape : pair.second) {
      for (const auto item : shape.second) {
        UpdateMinDim(min_dim, shape.first, item);
      }
      if (shape.second.size() > 1) {
        idx_map[shape.first].emplace_back(std::make_pair(shape.second[0], shape.second[1]));
      }
    }
  }
  for (const auto &map_info : idx_map) {
    for (const auto elem : map_info.second) {
      if (SameSymbol(elem.first, elem.second)) {
        GE_ASSERT_TRUE(elem.first <= elem.second, "Irrational shape dim(%d, %d) for continous axis map!", elem.first,
                       elem.second);
      } else {
        bool update_max = (elem.first < 0);
        cur_val = update_max ? (elem.second - elem.first) : (elem.first - elem.second);
        UpdateDim(max_dim, min_dim, map_info.first, cur_val, update_max);
      }
    }
  }
  return ge::SUCCESS;
}

ge::Status ExtraInfoGenerator::WriteInputShapeDimParam(const ArgsManager &args_manager, std::vector<std::string> &def_func) const {
  std::string var_str;
  std::string input_dim;
  std::map<uint32_t, uint32_t> max_dim;
  std::map<uint32_t, uint32_t> min_dim;
  GE_ASSERT_SUCCESS(GenDimMap(args_manager.GetAxisMap(), max_dim, min_dim), "Gen assign attr failed.");
  for (size_t i = 0; i < args_manager.GetInputAtts().size(); ++i) {
    var_str = max_dim.find(i) != max_dim.end() ? std::to_string(max_dim[i]) : "0";
    input_dim += "   max_dim_[" + std::to_string(i) + "] = " + var_str + ";\n";
    var_str = min_dim.find(i) != min_dim.end() ? std::to_string(min_dim[i]) : "0";
    input_dim += "   min_dim_[" + std::to_string(i) + "] = " + var_str + ";\n";
  }
  def_func.emplace_back(input_dim);
  return ge::SUCCESS;
}

ge::Status ExtraInfoGenerator::WriteInputAttrParam(const std::map<uint32_t, std::string> &dtype_info, const ArgsManager &args_manager, std::vector<std::string> &def_func) const {
  uint32_t idx = 0u;
  std::string str_idx;
  std::string attr_info;
  auto optional_atts = args_manager.GetOptionalAtts();
  for (const auto &pair : dtype_info) {
    str_idx = std::to_string(idx++);
    if (optional_atts.find(pair.first) != optional_atts.end()) {
      attr_info += "   max_att_[" + str_idx + "] = " + optional_atts[pair.first].max_value + ";\n";
      attr_info += "   min_att_[" + str_idx + "] = " + optional_atts[pair.first].min_value + ";\n";
      attr_info += "   check_att_[" + str_idx + "] = true;\n";
    } else {
      attr_info += "   max_att_[" + str_idx + "] = 0;\n";
      attr_info += "   min_att_[" + str_idx + "] = 0;\n";
      attr_info += "   check_att_[" + str_idx + "] = false;\n";
    }
  }
  def_func.emplace_back(attr_info);
  return ge::SUCCESS;
}

ge::Status ExtraInfoGenerator::WriteCheckShapeFunc(const ArgsManager &args_manager, std::vector<std::string> &def_func) const {
  auto axis_map = args_manager.GetAxisMap();
  ge::CodePrinter printer;
  std::string var_name;
  std::string used_codes;
  std::set<uint32_t> used_input_indexes = GetUsedInputIndexes(axis_map);
  bool is_first;
  printer.AddLine("  bool TilingVarsShapeCheck(gert::TilingContext *context) override {");
  for (const auto &pair : axis_map) {
    if (pair.second.size() > 1) {
      var_name = pair.first + "_size";
      GE_ASSERT_TRUE(IsValidVariableName(var_name));
      used_codes += "    int64_t " + var_name + " = 1;\n";
      is_first = true;
      for (const auto &shape : pair.second) {
        used_codes += GetCurSize(shape.first, shape.second);
        if (is_first) {
          used_codes += "    " + var_name + " = cur_size;\n";
          is_first = false;
        } else {
          used_codes += "    if (" + var_name + " != cur_size) {\n";
          used_codes += "      OP_LOGW(OP_NAME, \"Inconsistent shape for " + var_name + " from input0 and input" +
                        std::to_string(shape.first) + ".\");\n";
          used_codes += "      return false;\n";
          used_codes += "    }\n";
        }
      }
    }
  }
  if (!used_codes.empty()) {
    printer.AddLine("    int64_t cur_size;");
    printer.AddLine(GetInputSize(used_input_indexes));
    printer.AddLine(used_codes);
  }
  printer.AddLine("    OP_LOGD(OP_NAME, \"TilingVarsShapeCheck success.\");");
  printer.AddLine("    return true;");
  printer.AddLine("  }");
  def_func.emplace_back(printer.GetOutputStr());
  return ge::SUCCESS;
}

ge::Status ExtraInfoGenerator::WriteCheckCoverFunc(const ArgsManager &args_manager, std::vector<std::string> &def_func) const {
  auto axis_map = args_manager.GetAxisContinousMap();
  ge::CodePrinter printer;
  std::string impl_func;
  std::string invoke_code;
  std::string func_code;
  for (const auto pair : axis_map) {
    impl_func = GenCheckInputCoverFunc(pair.first, pair.second);
    if (impl_func.size() > 0) {
      printer.AddLine(impl_func);
      invoke_code +=
        "    if (!CheckInput" + std::to_string(pair.first) + "Cover(context)) {\n";
      invoke_code += "      OP_LOGW(OP_NAME, \"Map for input " + std::to_string(pair.first) + " cannot cover all axis.\");\n";
      invoke_code += "      delete[] bool_space_;\n";
      invoke_code += "      return false;\n";
      invoke_code += "    }\n";
    }
  }
  if (!invoke_code.empty()) {
    func_code += "  bool TilingVarsCoverCheck(gert::TilingContext *context) override {\n";
    func_code += "    uint64_t input_size = 0;\n";
    for (const auto &pair : axis_map) {
      func_code += "    input_size = Max(context->GetInputShape(" + std::to_string(pair.first) +
            ")->GetStorageShape().GetDimNum(), input_size);\n";
    }
    func_code += "    bool_space_ = new bool[input_size];\n";
    func_code += invoke_code;
    func_code += "    OP_LOGD(OP_NAME, \"TilingVarsCoverCheck success.\");\n";
    func_code += "    delete[] bool_space_;\n";
    func_code += "    return true;\n";
    func_code += "  }\n";
    printer.AddLine(func_code);
    def_func.emplace_back(printer.GetOutputStr());
  }
  return ge::SUCCESS;
}

ge::Status ExtraInfoGenerator::GenCheckFunc(const ArgsManager &args_manager, std::string &impl_code) {
  ge::CodePrinter printer;
  printer.AddLine("  bool CheckIsCapable(" + config_.tiling_data_type_name +
                       " &tiling_data) {");
  if (config_.do_input_args_proc) {
    auto input_vars_cond = GetVarsValidCond(args_manager.GetInputVarsRange());
    for (const auto &var_cond : input_vars_cond) {
      printer.AddLine("    if (" + var_cond + ") {");
      printer.AddLine("      OP_LOGW(OP_NAME, \"" + var_cond + ", invalid input var.\");");
      printer.AddLine("      return false;");
      printer.AddLine("    }");
    }
  }
  printer.AddLine("    OP_LOGD(OP_NAME, \"CheckIsCapable success.\");");
  printer.AddLine("    return true;");
  printer.AddLine("  }");
  impl_code = printer.GetOutputStr();
  return ge::SUCCESS;
}

ge::Status ExtraInfoGenerator::WriteAssignAttAndOutputSize(const ModelInfo &model_info, std::string &impl_code) {
  std::string add_log = "\n";
  ge::CodePrinter printer;
  // 获取输入数据有效值
  const uint32_t tiling_id = model_info.tiling_case_id;
  auto optional_atts = model_info.graph_input_infos.optional_atts;
  if (config_.with_tiling_ctx) {
    printer.AddLine("  void AssignAttAndOutputSize(" + config_.tiling_data_type_name +
                    " &tiling_data, gert::TilingContext *context) {");
    printer.AddLine("    OP_LOGD(OP_NAME, \"Start assigning attr and output size for tiling case " + std::to_string(tiling_id) +
                    ".\");");
    printer.AddLine("    auto attrs = context->GetAttrs();");
    for (const auto &optional_att : optional_atts) {
      auto att_info = optional_att.second;
      auto var_name = att_info.optional_name + "_ptr";
      printer.AddLine("    auto " + var_name + " = attrs->GetAttrPointer<" + att_info.data_type + ">(" +
                      std::to_string(optional_att.first) + "U);");
      printer.AddLine("    " + att_info.data_type + " " + att_info.optional_name + " = *" + var_name + ";");
      printer.AddLine("    tiling_data.set_" + att_info.optional_name + "(" + att_info.optional_name + ");");
      add_log += ConstructAttLog(att_info.optional_name) + "\n";
    }
    for (uint32_t output_index = 0U; output_index < model_info.output_size; output_index++) {
      printer.AddLine("    tiling_data.set_" + std::string("output") + std::to_string(output_index) +
                      "_total_size(context->GetOutputShape(" + std::to_string(output_index) +
                      ")->GetStorageShape().GetShapeSize());");
      printer.AddLine("    tiling_data.set_" + std::string("output") + std::to_string(output_index) +
                      "_single_core_size(context->GetOutputShape(" + std::to_string(output_index) +
                      ")->GetStorageShape().GetShapeSize() / corenum_);");
      add_log += ConstructAttLog("output" + std::to_string(output_index) + "_single_core_size") + "\n";
      add_log += ConstructAttLog("output" + std::to_string(output_index) + "_total_size") + "\n";
    }
    printer.AddLine(add_log);
    printer.AddLine("    OP_LOGD(OP_NAME, \"Assigned attr and output size for tiling case " + std::to_string(tiling_id) +
                    " successfully.\");");
    printer.AddLine("  }");
  }
  impl_code = printer.GetOutputStr();
  return ge::SUCCESS;
}

//  --------------------------------以下为tilingdata定义---------------------------
//  返回值std::string tilingdata定义
std::string ExtraInfoGenerator::WriteCoreParamData(const ModelInfo &model_info,
                                                   const TilingDataGenType tiling_data_gen_type,
                                                   std::set<std::string> &tiling_data_vars) {
  ge::CodePrinter printer;
  const auto &tiling_datas =
      tiling_data_generator_.GetTilingDataWithAnnotation(model_info.tiling_case_id, tiling_data_gen_type);
  for (const auto &tiling_data_name : tiling_datas) {
    std::vector<std::string> tiling_data_name_vec{tiling_data_name.first};
    if (TilingDataGenUtils::NeedWrittenTilingData(tiling_data_name_vec, tiling_data_vars)) {
      printer.AddLine(tiling_data_name.second);
      TilingDataGenUtils::WriteTilingDataElement(printer, tiling_data_vars, tiling_data_name_vec);
      GELOGD("Write tiling data: name[%s]", tiling_data_name.first.c_str());
    }
  }
  return printer.GetOutputStr();
}

std::string GenApiTilingParam(const uint32_t tiling_case_id, std::set<std::string> &tiling_data_vars) {
  auto api_tiling_data = ApiTilingMgr::Instance().GetApiTilingDataType(tiling_case_id);
  return TilingDataGenUtils::WriteTilingDataElement(tiling_data_vars, api_tiling_data);
}

ge::Status ExtraInfoGenerator::GetExtraTilingDataDef(std::map<std::string, std::string> &type_name_to_definition) {
  std::set<std::string> tiling_data_vars;
  for (const auto &model_info : model_info_list_) {
    if (!model_info.sub_case_tag.empty()) {
      continue;
    }
    // 轴对应的TilingData相关参数
    type_name_to_definition["CoreParams"] +=
        WriteCoreParamData(model_info, TilingDataGenType::AXES_TILING_DATA_GEN, tiling_data_vars);
    if (config_.do_api_tiling) {
      type_name_to_definition["ApiParams"] += GenApiTilingParam(model_info.tiling_case_id, tiling_data_vars);
    }
  }
  return ge::SUCCESS;
}

ge::Status ExtraInfoGenerator::GetExtraTilingVars(const uint32_t tiling_key, std::set<std::string> &tiling_vars) {
  const auto model_info = GetModelInfo(tiling_key);
  GE_ASSERT_NOTNULL(model_info);
  // 轴对应的TilingData相关参数
  WriteCoreParamData(*model_info, TilingDataGenType::AXES_TILING_DATA_GEN, tiling_vars);
  if (config_.do_api_tiling) {
    GenApiTilingParam(model_info->tiling_case_id, tiling_vars);
  }
  return ge::SUCCESS;
}

ge::Status ExtraInfoGenerator::GenExtraTilingData(const ArgsManager &args_manager, std::string &impl_code) {
  const auto model_info = GetModelInfo(args_manager.GetTilingCaseId());
  GE_ASSERT_NOTNULL(model_info);
  GE_ASSERT_SUCCESS(WriteAssignAttAndOutputSize(*model_info, impl_code), "Gen assign attr failed.");
  return ge::SUCCESS;
}

const ModelInfo *ExtraInfoGenerator::GetModelInfo(const uint32_t tiling_key) const {
  for (const auto &model_info : model_info_list_) {
    if (model_info.tiling_case_id == tiling_key) {
      return &model_info;
    }
  }
  return nullptr;
}
ge::Status ExtraInfoGenerator::GenCtxCheck(const ArgsManager &args_manager, std::vector<std::string> &impl_codes) {
  if (config_.with_tiling_ctx) {
    ge::CodePrinter printer;
    GE_ASSERT_SUCCESS(WriteCheckShapeFunc(args_manager, impl_codes), "Gen check shape failed.");
    GE_ASSERT_SUCCESS(WriteCheckCoverFunc(args_manager, impl_codes), "Gen check cover failed.");
  }
  return ge::SUCCESS;
}

ge::Status ExtraInfoGenerator::GenGetShapeAttr(const ArgsManager &args_manager, std::vector<std::string> &impl_codes) {
  auto axis_map = args_manager.GetAxisMap();
  ge::CodePrinter printer;
  std::string used_code;
  std::vector<std::string> cover_codes;
  std::set<uint32_t> used_input_indexes = GetUsedInputIndexes(axis_map, true);
  const uint32_t tiling_id = args_manager.GetTilingCaseId();
  GE_ASSERT_SUCCESS(WriteCheckCoverFunc(args_manager, cover_codes), "Gen check cover failed.");
  printer.AddLine("  bool GetShapeAttrsInfo(" + config_.tiling_data_type_name +
                    " &tiling_data, gert::TilingContext *context) {");
  printer.AddLine("    if (!TilingVarsShapeCheck(context)) {");
  printer.AddLine("      OP_LOGW(OP_NAME, \"TilingVarsShapeCheck failed.\");");
  printer.AddLine("      return false;");
  printer.AddLine("    }");
  if (!cover_codes.empty()) {
    printer.AddLine("    if (!TilingVarsCoverCheck(context)) {");
    printer.AddLine("      OP_LOGW(OP_NAME, \"TilingVarsCoverCheck failed.\");");
    printer.AddLine("      return false;");
    printer.AddLine("    }");
  }
  printer.AddLine("    OP_LOGD(OP_NAME, \"Start setting axis size for " + std::to_string(tiling_id) + ".\");");
  for (const auto &pair : axis_map) {
    for (const auto &shape : pair.second) {
      if (shape.second.size() == 1u) {
        used_code += "    uint32_t " + pair.first + "_size = context->GetInputShape(" + std::to_string(shape.first) +
                       ")->GetStorageShape().GetDim(" + GetIdx(shape.first, shape.second[0]) + ");\n";
      } else {
        used_code += "    uint32_t " + pair.first + "_size = 1;\n";
        used_code += "    for (size_t i = " + GetIdx(shape.first, shape.second[0]) +
                       "; i <= " + GetIdx(shape.first, shape.second[1]) + "; i++) {\n";
        used_code += "      " + pair.first + "_size *= context->GetInputShape(" + std::to_string(shape.first) +
                       ")->GetStorageShape().GetDim(i);\n";
        used_code += "    }\n";
      }
      used_code += "    tiling_data.set_" + pair.first + "(" + pair.first + "_size);\n";
      used_code += "    OP_LOGD(OP_NAME, \"Initiate " + pair.first + " to %d.\", tiling_data.get_" + pair.first + "());\n";
    }
  }
  if (!used_code.empty()) {
    printer.AddLine(GetInputSize(used_input_indexes));
    printer.AddLine(used_code);
  }
  printer.AddLine("    OP_LOGD(OP_NAME, \"End setting axis size for " + std::to_string(tiling_id) + ".\");");
  printer.AddLine("    return true;");
  printer.AddLine("  }");
  impl_codes.emplace_back(printer.GetOutputStr());
  return ge::SUCCESS;
}

ge::Status ExtraInfoGenerator::GenWorkSpacePass(const ModelInfo &model_info, std::string &impl_code) {
  Expr workspace_size = model_info.workspace_size;
  std::set<std::string> arg_names;
  std::string tiling_case_id = std::to_string(model_info.tiling_case_id);
  if (!IsValid(workspace_size)) {
    workspace_size = ge::sym::kSymbolZero;
  }
  impl_code += "  void GetWorkSpaceSize(" + config_.tiling_data_type_name + "& tiling_data) {\n";
  impl_code += "    OP_LOGD(OP_NAME, \"Start setting workspace for case " + tiling_case_id + ".\");\n";
  for (const auto &arg : workspace_size.FreeSymbols()) {
    if (arg.GetExprType() == ge::ExprType::kExprVariable) {
      arg_names.insert(Str(arg));
    }
  }
  for (const auto &arg : arg_names) {
    impl_code += "    double " + arg + " = static_cast<double>(tiling_data.get_" + arg + "());\n";
  }
  impl_code += "    tiling_data.set_workspaceSize(static_cast<uint32_t>(" + Str(workspace_size) + "));\n";
  impl_code += "    OP_LOGD(OP_NAME, \"Setting workspace to %u for case " + tiling_case_id + ".\", tiling_data.get_workspaceSize());\n";
  impl_code += "  }\n";
  return ge::SUCCESS;
}
}  // namespace att
