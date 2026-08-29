/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
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

/**
 * @file graph_node_setted_format_pass.cpp
 * @brief 从 custom_formats.cfg 读取算子格式配置，修改图中对应算子输入/输出 format 和 shape，
 *        若输出直连 NetOutput 则同步修改其输入 format 和 shape，通过 CheckOpSupported 校验修改是否合法，
 *        校验通过后检查并删除前后因 format 变更而变冗余的 Transpose 节点。
 *
 * custom_formats.cfg 配置格式（简单文本）：
 *
 *   [NodeName1]
 *   input.1=FORMAT_NHWC
 *   output.0=FORMAT_NCHW
 *
 *   [NodeName2]
 *   input.0=FORMAT_NCHW
 *   output.0=FORMAT_NHWC
 *
 * - 以 [NodeName] 作为 section 分隔，NodeName 为图中节点的名称（node_name）
 * - input.<index>=<format> 或 output.<index>=<format>
 * - format 值参考 ge::Format 枚举（如 FORMAT_NCHW, FORMAT_NHWC 等）
 * - 空行和 # 开头的注释行会被忽略
 *
 * CheckOpSupported 校验逻辑参考：
 *   实现：调用对外接口 GeUtils::CheckNodeSupportOnAicore
 */

#include <algorithm>
#include <cstdlib>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "ge/fusion/pass/pattern_fusion_pass.h"
#include "ge/ge_utils.h"

using namespace ge;
using namespace fusion;

namespace {

// ---------- Format 字符串映射表 ----------

static const std::unordered_map<std::string, Format> kFormatMap = {
    {"FORMAT_NCHW", FORMAT_NCHW},
    {"FORMAT_NHWC", FORMAT_NHWC},
};

static const std::unordered_set<Format> kSupportedFormats = {FORMAT_NCHW, FORMAT_NHWC};

/**
 * @brief 判断 format 是否在 kFormatMap 支持范围内。
 */
bool IsFormatSupported(Format fmt) {
  return kSupportedFormats.find(fmt) != kSupportedFormats.end();
}

/**
 * @brief 将格式字符串（如 "FORMAT_NCHW"）转换为 ge::Format 枚举值
 */
bool ParseFormatString(const std::string &format_str, Format &out_format) {
  auto it = kFormatMap.find(format_str);
  if (it != kFormatMap.end()) {
    out_format = it->second;
    return true;
  }
  return false;
}

// ---------- 配置数据结构 ----------

struct FormatConfig {
  std::unordered_map<uint32_t, Format> input_formats;
  std::unordered_map<uint32_t, Format> output_formats;
};

// ---------- 配置文件解析 ----------

std::string Trim(const std::string &str) {
  size_t start = str.find_first_not_of(" \t\r\n");
  if (start == std::string::npos) {
    return "";
  }
  size_t end = str.find_last_not_of(" \t\r\n");
  return str.substr(start, end - start + 1);
}

/**
 * @brief 解析 section 行: [NodeName]，返回 true 表示已消费该行。
 */
bool ParseSectionLine(const std::string &trimmed, size_t line_num, std::string &current_node,
                      std::unordered_map<std::string, FormatConfig> &op_configs) {
  if (trimmed[0] != '[') {
    return false;
  }
  size_t close = trimmed.find(']');
  if (close == std::string::npos) {
    std::cout << "[GraphNodeSettedFormatPass] Parse error at line " << line_num << ": missing ']' in section header"
              << std::endl;
    return true;
  }
  current_node = trimmed.substr(1, close - 1);
  if (current_node.empty()) {
    std::cout << "[GraphNodeSettedFormatPass] Parse error at line " << line_num << ": empty node name" << std::endl;
    return true;
  }
  op_configs.try_emplace(current_node);
  return true;
}

/**
 * @brief 解析 key=value 行: input.<idx>=<format> 或 output.<idx>=<format>
 */
void ParseFormatKeyValue(const std::string &trimmed, size_t line_num, const std::string &current_node,
                         std::unordered_map<std::string, FormatConfig> &op_configs) {
  size_t eq_pos = trimmed.find('=');
  if (eq_pos == std::string::npos) {
    std::cout << "[GraphNodeSettedFormatPass] Parse error at line " << line_num << ": missing '='" << std::endl;
    return;
  }
  if (current_node.empty()) {
    std::cout << "[GraphNodeSettedFormatPass] Parse error at line " << line_num << ": key=value before any [section]"
              << std::endl;
    return;
  }

  std::string key = Trim(trimmed.substr(0, eq_pos));
  std::string value = Trim(trimmed.substr(eq_pos + 1));

  bool is_input = (key.compare(0, 6, "input.") == 0);
  bool is_output = (key.compare(0, 7, "output.") == 0);
  if (!is_input && !is_output) {
    std::cout << "[GraphNodeSettedFormatPass] Parse error at line " << line_num
              << ": key must start with 'input.' or 'output.'" << std::endl;
    return;
  }

  std::string idx_str = is_input ? key.substr(6) : key.substr(7);
  uint32_t idx;
  try {
    idx = static_cast<uint32_t>(std::stoul(idx_str));
  } catch (const std::exception &) {
    std::cout << "[GraphNodeSettedFormatPass] Parse error at line " << line_num << ": invalid index '" << idx_str << "'"
              << std::endl;
    return;
  }

  Format fmt;
  if (!ParseFormatString(value, fmt)) {
    std::cout << "[GraphNodeSettedFormatPass] Parse error at line " << line_num << ": unknown format string '" << value
              << "'" << std::endl;
    return;
  }

  if (is_input) {
    op_configs[current_node].input_formats[idx] = fmt;
  } else {
    op_configs[current_node].output_formats[idx] = fmt;
  }
}

/**
 * @brief 从环境变量 ASCEND_CUSTOM_FORMATS_CFG 获取配置文件路径并解析节点格式配置
 * @return node_name -> FormatConfig 的映射表
 * 运行前设置环境变量 export ASCEND_CUSTOM_FORMATS_CFG=/path/to/custom_formats.cfg
 */
std::unordered_map<std::string, FormatConfig> ParseConfigFile() {
  std::unordered_map<std::string, FormatConfig> op_configs;
  const char *env_path = std::getenv("ASCEND_CUSTOM_FORMATS_CFG");
  if (env_path == nullptr || env_path[0] == '\0') {
    std::cout << "[GraphNodeSettedFormatPass] Environment variable ASCEND_CUSTOM_FORMATS_CFG is not set, skip parsing"
              << std::endl;
    return op_configs;
  }

  std::string filename(env_path);
  std::ifstream file(filename);
  if (!file.is_open()) {
    std::cout << "[GraphNodeSettedFormatPass] Cannot open config file: " << filename << std::endl;
    return op_configs;
  }

  std::string current_node;
  std::string line;
  size_t line_num = 0;

  while (std::getline(file, line)) {
    ++line_num;
    std::string trimmed = Trim(line);
    if (trimmed.empty() || trimmed[0] == '#') {
      continue;
    }

    if (ParseSectionLine(trimmed, line_num, current_node, op_configs)) {
      continue;
    }

    ParseFormatKeyValue(trimmed, line_num, current_node, op_configs);
  }

  std::cout << "[GraphNodeSettedFormatPass] Parsed " << op_configs.size() << " op config(s) from " << filename
            << std::endl;
  return op_configs;
}

// ---------- CheckOpSupported ----------

/**
 * @brief 校验算子当前的输入/输出格式组合是否被 AI Core 支持。
 *
 * 参考 user_semantic_inference.cc 中的使用方式：
 *   1. 修改 tensor 的 format
 *   2. 调用 CheckOpSupported(node) 校验
 *   3. 若返回 false，回退修改并报错
 *
 * 实现：调用对外接口 GeUtils::CheckNodeSupportOnAicore 校验节点是否支持在 AI Core 上执行。
 *
 * @param gnode 待校验的图节点（GNode 引用）
 * @param out_reason 输出：不支持的原因描述
 * @return true 支持, false 不支持
 */
bool CheckOpSupported(const GNode &gnode, std::string &out_reason) {
  bool is_supported = false;
  AscendString reason;
  Status ret = GeUtils::CheckNodeSupportOnAicore(gnode, is_supported, reason);
  if (ret != SUCCESS) {
    out_reason = "CheckNodeSupportOnAicore failed";
    return false;
  }
  if (!is_supported) {
    out_reason = reason.GetString();
  }
  return is_supported;
}

// ---------- Format 与 Shape 联动转换 ----------

/// 4D NCHW -> NHWC: [N,C,H,W] -> [N,H,W,C]
std::vector<int64_t> NchwToNhwc(const std::vector<int64_t> &dims) {
  return {dims[0], dims[2], dims[3], dims[1]};
}

/// 4D NHWC -> NCHW: [N,H,W,C] -> [N,C,H,W]
std::vector<int64_t> NhwcToNchw(const std::vector<int64_t> &dims) {
  return {dims[0], dims[3], dims[1], dims[2]};
}

/// 5D NCDHW -> NDHWC: [N,C,D,H,W] -> [N,D,H,W,C]
std::vector<int64_t> NcdhwToNdhwc(const std::vector<int64_t> &dims) {
  return {dims[0], dims[2], dims[3], dims[4], dims[1]};
}

/// 5D NDHWC -> NCDHW: [N,D,H,W,C] -> [N,C,D,H,W]
std::vector<int64_t> NdhwcToNcdhw(const std::vector<int64_t> &dims) {
  return {dims[0], dims[4], dims[1], dims[2], dims[3]};
}

/**
 * @brief 根据目标 format 转换 TensorDesc 的 shape。
 *
 * 修改 format 时同步重排 shape 维度，避免 format 与 shape 不一致导致报错。
 *
 * 仅处理维度排布变化的 format 转换（如 NCHW<->NHWC、NCDHW<->NDHWC）。
 * 对于 FORMAT_ND 等不改变维度语义的 format，保持 shape 不变。
 *
 * @param desc 待修改的 TensorDesc（同时修改 format 和 shape）
 * @param old_format 原始 format
 * @param new_format 目标 format
 * @return true 成功转换, false 不支持的 format 组合或维度不匹配
 */
bool UpdateShapeByFormat(TensorDesc &desc, Format old_format, Format new_format) {
  // format 未变化，无需转换 shape
  if (old_format == new_format) {
    desc.SetFormat(new_format);
    return true;
  }

  // FORMAT_ND 与任意 format 互转时 shape 不变（ND 不约束维度语义）
  if (new_format == FORMAT_ND || old_format == FORMAT_ND) {
    desc.SetFormat(new_format);
    return true;
  }

  auto shape = desc.GetShape();
  auto dims = shape.GetDims();
  auto dim_num = dims.size();
  // 4D: NCHW <-> NHWC
  if (dim_num == 4U) {
    if (old_format == FORMAT_NCHW && new_format == FORMAT_NHWC) {
      desc.SetShape(Shape(NchwToNhwc(dims)));
    } else if (old_format == FORMAT_NHWC && new_format == FORMAT_NCHW) {
      desc.SetShape(Shape(NhwcToNchw(dims)));
    } else {
      // 其他 4D format 组合暂不处理，仅改 format
      std::cout << "[GraphNodeSettedFormatPass] Unsupported 4D format conversion: " << static_cast<int>(old_format)
                << " -> " << static_cast<int>(new_format) << ", shape unchanged" << std::endl;
    }
    // 5D: NCDHW <-> NDHWC
  } else if (dim_num == 5U) {
    if (old_format == FORMAT_NCDHW && new_format == FORMAT_NDHWC) {
      desc.SetShape(Shape(NcdhwToNdhwc(dims)));
    } else if (old_format == FORMAT_NDHWC && new_format == FORMAT_NCDHW) {
      desc.SetShape(Shape(NdhwcToNcdhw(dims)));
    } else {
      std::cout << "[GraphNodeSettedFormatPass] Unsupported 5D format conversion: " << static_cast<int>(old_format)
                << " -> " << static_cast<int>(new_format) << ", shape unchanged" << std::endl;
    }
  } else {
    std::cout << "[GraphNodeSettedFormatPass] Dim num " << dim_num
              << " not handled for format conversion, shape unchanged" << std::endl;
  }

  desc.SetFormat(new_format);
  return true;
}

/// NetOutput 算子类型名
constexpr const char *kNetOutputType = "NetOutput";

/// 备份 TensorDesc 的 format 和 shape，用于回退
struct TensorDescBackup {
  Format format;
  Shape shape;
};

/**
 * @brief 回退节点的 input/output format 和 shape 到备份值。
 */
void RollbackFormats(GNode &node, const std::unordered_map<uint32_t, TensorDescBackup> &input_backups,
                     const std::unordered_map<uint32_t, TensorDescBackup> &output_backups) {
  for (const auto &[idx, backup] : input_backups) {
    TensorDesc desc;
    if (node.GetInputDesc(static_cast<int64_t>(idx), desc) == GRAPH_SUCCESS) {
      desc.SetFormat(backup.format);
      desc.SetShape(backup.shape);
      node.UpdateInputDesc(static_cast<int64_t>(idx), desc);
    }
  }
  for (const auto &[idx, backup] : output_backups) {
    TensorDesc desc;
    if (node.GetOutputDesc(static_cast<int64_t>(idx), desc) == GRAPH_SUCCESS) {
      desc.SetFormat(backup.format);
      desc.SetShape(backup.shape);
      node.UpdateOutputDesc(static_cast<int64_t>(idx), desc);
    }
  }
}

/**
 * @brief 记录对 NetOutput 节点输入格式和 shape 的修改，用于回退。
 */
struct NetOutputBackup {
  GNodePtr node;           // 被修改的 NetOutput 节点
  int32_t input_index;     // 被修改的输入端口
  Format original_format;  // 原始格式
  Shape original_shape;    // 原始 shape
};

/**
 * @brief 当节点输出格式被修改后，检查该输出是否直连 NetOutput 节点，
 *        若是则同步修改 NetOutput 对应输入端口的格式。
 *
 * @param node 刚修改完输出格式的节点
 * @param config 该节点的格式配置（仅用 output_formats）
 * @param node_name 节点名（用于日志）
 * @param backups 输出：记录所有被修改的 NetOutput 节点信息，用于后续回退
 */
void PropagateFormatToNetOutput(GNode &node, const FormatConfig &config, const std::string &node_name,
                                std::vector<NetOutputBackup> &backups) {
  for (const auto &[out_idx, fmt] : config.output_formats) {
    // 获取该输出端口连接的所有后继数据节点及其输入端口索引
    auto successors = node.GetOutDataNodesAndPortIndexs(static_cast<int32_t>(out_idx));
    for (const auto &[succ_node, succ_in_idx] : successors) {
      if (succ_node == nullptr) {
        continue;
      }
      AscendString succ_type;
      if (succ_node->GetType(succ_type) != GRAPH_SUCCESS) {
        continue;
      }
      std::cout << "[GraphNodeSettedFormatPass] Node[" << node_name << "]: succ_type is " << succ_type.GetString()
                << std::endl;
      if (std::string(succ_type.GetString()) != "NetOutput") {
        continue;
      }

      // 备份 NetOutput 输入端口的原始格式和 shape
      TensorDesc desc;
      if (succ_node->GetInputDesc(succ_in_idx, desc) != GRAPH_SUCCESS) {
        std::cout << "[GraphNodeSettedFormatPass] Node[" << node_name << "]: cannot get NetOutput input desc at port "
                  << succ_in_idx << std::endl;
        continue;
      }
      backups.push_back({succ_node, succ_in_idx, desc.GetFormat(), desc.GetShape()});

      // 修改 NetOutput 输入端口的格式和 shape
      Format old_format = desc.GetFormat();
      UpdateShapeByFormat(desc, old_format, fmt);
      if (succ_node->UpdateInputDesc(succ_in_idx, desc) != GRAPH_SUCCESS) {
        std::cout << "[GraphNodeSettedFormatPass] Node[" << node_name
                  << "]: failed to update NetOutput input desc at port " << succ_in_idx << std::endl;
        continue;
      }

      AscendString succ_name;
      std::string name_str = (succ_node->GetName(succ_name) == GRAPH_SUCCESS) ? succ_name.GetString() : "unknown";
      std::cout << "[GraphNodeSettedFormatPass] Node[" << node_name << "]: propagated output." << out_idx
                << " format to NetOutput[" << name_str << "] input." << succ_in_idx << std::endl;
    }
  }
}

/**
 * @brief 回退所有通过 PropagateFormatToNetOutput 修改的 NetOutput 节点。
 */
void RollbackNetOutput(const std::vector<NetOutputBackup> &backups) {
  for (const auto &b : backups) {
    if (b.node == nullptr) {
      continue;
    }
    TensorDesc desc;
    if (b.node->GetInputDesc(b.input_index, desc) == GRAPH_SUCCESS) {
      desc.SetFormat(b.original_format);
      desc.SetShape(b.original_shape);
      b.node->UpdateInputDesc(b.input_index, desc);
    }
  }
}

// ---------- 格式修改与校验 ----------

/**
 * @brief 备份节点输入/输出端口当前 format 和 shape 到 backup 映射表中。
 * @return true 备份成功, false 无法获取某个 tensor desc
 */
bool BackupFormatDescs(GNode &node, const FormatConfig &config, const std::string &node_name,
                       std::unordered_map<uint32_t, TensorDescBackup> &input_backups,
                       std::unordered_map<uint32_t, TensorDescBackup> &output_backups) {
  for (const auto &[idx, fmt] : config.input_formats) {
    TensorDesc desc;
    if (node.GetInputDesc(static_cast<int64_t>(idx), desc) != GRAPH_SUCCESS) {
      std::cout << "[GraphNodeSettedFormatPass] Node[" << node_name << "]: cannot get input desc at index " << idx
                << std::endl;
      return false;
    }
    input_backups[idx] = {desc.GetFormat(), desc.GetShape()};
  }
  for (const auto &[idx, fmt] : config.output_formats) {
    TensorDesc desc;
    if (node.GetOutputDesc(static_cast<int64_t>(idx), desc) != GRAPH_SUCCESS) {
      std::cout << "[GraphNodeSettedFormatPass] Node[" << node_name << "]: cannot get output desc at index " << idx
                << std::endl;
      return false;
    }
    output_backups[idx] = {desc.GetFormat(), desc.GetShape()};
  }
  return true;
}

/**
 * @brief 将配置中的 input/output format 和 shape 应用到节点上。
 *        若 output 修改失败，会自动回退已修改的 input format 和 shape。
 * @param input_backups 已备份的 input format/shape（用于 output 失败时的回退）
 * @return true 全部应用成功, false 应用失败（input 已回退）
 */
bool ApplyFormatChanges(GNode &node, const FormatConfig &config, const std::string &node_name,
                        const std::unordered_map<uint32_t, TensorDescBackup> &input_backups) {
  // 修改 input format 和 shape
  for (const auto &[idx, fmt] : config.input_formats) {
    TensorDesc desc;
    if (node.GetInputDesc(static_cast<int64_t>(idx), desc) != GRAPH_SUCCESS) {
      return false;
    }
    Format old_format = desc.GetFormat();
    UpdateShapeByFormat(desc, old_format, fmt);
    if (node.UpdateInputDesc(static_cast<int64_t>(idx), desc) != GRAPH_SUCCESS) {
      std::cout << "[GraphNodeSettedFormatPass] Node[" << node_name << "]: failed to update input desc " << idx
                << std::endl;
      return false;
    }
  }

  // 修改 output format 和 shape
  for (const auto &[idx, fmt] : config.output_formats) {
    TensorDesc desc;
    if (node.GetOutputDesc(static_cast<int64_t>(idx), desc) != GRAPH_SUCCESS) {
      RollbackFormats(node, input_backups, {});
      return false;
    }
    Format old_format = desc.GetFormat();
    UpdateShapeByFormat(desc, old_format, fmt);
    if (node.UpdateOutputDesc(static_cast<int64_t>(idx), desc) != GRAPH_SUCCESS) {
      RollbackFormats(node, input_backups, {});
      return false;
    }
  }
  return true;
}

// ---------- 冗余 Transpose 消除 ----------

/// Transpose 算子类型名
constexpr const char *kTransposeType = "Transpose";
constexpr int32_t kTransposeDataInput = 0;
constexpr int32_t kTransposePermInput = 1;
constexpr int32_t kTransposeOutput = 0;

/**
 * @brief 判断节点是否为 Transpose 类型。
 */
bool IsTransposeNode(const GNode &node) {
  AscendString type;
  return node.GetType(type) == GRAPH_SUCCESS && std::string(type.GetString()) == "Transpose";
}

/**
 * @brief 判断 Transpose 节点的 perm 输入（input.1）是否为无输入的 Const 节点。
 *
 * 只有 perm 为无输入的 Const 时才能安全删除 Transpose 并清理 perm 节点，
 * 非 Const perm（如运行时计算得到的 perm）或 Const 有输入时删除后会导致语义错误。
 *
 * @param transpose_node 待检查的 Transpose 节点
 * @return true perm 输入为无输入的 Const, false perm 输入不存在、非 Const 或 Const 有输入
 */
bool IsTransposePermConst(const GNode &transpose_node) {
  auto [perm_node, perm_port] = transpose_node.GetInDataNodesAndPortIndexs(kTransposePermInput);
  if (perm_node == nullptr) {
    AscendString tp_name;
    std::string name_str = (transpose_node.GetName(tp_name) == GRAPH_SUCCESS) ? tp_name.GetString() : "unknown";
    std::cout << "[GraphNodeSettedFormatPass] Transpose[" << name_str << "] has no perm input, skip removal"
              << std::endl;
    return false;
  }
  AscendString perm_type;
  if (perm_node->GetType(perm_type) != GRAPH_SUCCESS || std::string(perm_type.GetString()) != "Const") {
    AscendString tp_name;
    std::string tp_name_str = (transpose_node.GetName(tp_name) == GRAPH_SUCCESS) ? tp_name.GetString() : "unknown";
    AscendString perm_name;
    std::string perm_name_str = (perm_node->GetName(perm_name) == GRAPH_SUCCESS) ? perm_name.GetString() : "unknown";
    std::cout << "[GraphNodeSettedFormatPass] Transpose[" << tp_name_str << "] perm input node[" << perm_name_str
              << "] is not Const (type=" << (perm_type.GetString() == nullptr ? "unknown" : perm_type.GetString())
              << "), skip removal" << std::endl;
    return false;
  }
  bool has_data_input = false;
  for (size_t i = 0; i < perm_node->GetInputsSize(); ++i) {
    auto [in_node, in_port] = perm_node->GetInDataNodesAndPortIndexs(static_cast<int32_t>(i));
    if (in_node != nullptr) {
      has_data_input = true;
      break;
    }
  }
  if (!perm_node->GetInControlNodes().empty() || has_data_input) {
    AscendString tp_name;
    std::string tp_name_str = (transpose_node.GetName(tp_name) == GRAPH_SUCCESS) ? tp_name.GetString() : "unknown";
    AscendString perm_name;
    std::string perm_name_str = (perm_node->GetName(perm_name) == GRAPH_SUCCESS) ? perm_name.GetString() : "unknown";
    std::cout << "[GraphNodeSettedFormatPass] Transpose[" << tp_name_str << "] perm Const node[" << perm_name_str
              << "] has input edges, skip removal" << std::endl;
    return false;
  }
  return true;
}

/**
 * @brief 判断 Transpose 节点是否存在控制边输入或输出。
 *
 * 若 Transpose 有控制边输入或输出，删除后会导致控制依赖丢失。
 *
 * @param transpose_node 待检查的 Transpose 节点
 * @return true 无控制边输入和输出（可安全删除）, false 存在控制边输入或输出
 */
bool HasNoControlEdge(const GNode &transpose_node) {
  if (!transpose_node.GetInControlNodes().empty()) {
    AscendString tp_name;
    std::string name_str = (transpose_node.GetName(tp_name) == GRAPH_SUCCESS) ? tp_name.GetString() : "unknown";
    std::cout << "[GraphNodeSettedFormatPass] Transpose[" << name_str << "] has control input edges, skip removal"
              << std::endl;
    return false;
  }
  if (!transpose_node.GetOutControlNodes().empty()) {
    AscendString tp_name;
    std::string name_str = (transpose_node.GetName(tp_name) == GRAPH_SUCCESS) ? tp_name.GetString() : "unknown";
    std::cout << "[GraphNodeSettedFormatPass] Transpose[" << name_str << "] has control output edges, skip removal"
              << std::endl;
    return false;
  }
  return true;
}

/**
 * @brief 判断 Transpose 节点是否冗余：输入侧 format 与输出侧 format 是否一致。
 *
 * 修改节点 format 后，与该节点直连的 Transpose 两侧的 format 可能变为一致，
 * 此时 Transpose 不再做有意义的格式转换，可以安全删除。
 *
 * 例如：
 *   修改前: Data(NCHW) -> Transpose -> relu(NHWC)  # 两侧 format 不同，需要
 *   修改后: Data(NHWC) -> Transpose -> relu(NHWC)  # 两侧 format 相同，冗余
 *
 * 判断方式：
 *   - 获取 Transpose data input 的源头节点输出 format（输入侧 format）
 *   - 获取 Transpose output 的消费者节点输入 format（输出侧 format）
 *   - 若两侧 format 一致则 Transpose 冗余
 *
 * @param transpose_node 待检查的 Transpose 节点
 * @return true 冗余可删除, false 不冗余或不满足删除条件
 */
bool IsTransposeRedundant(const GNode &transpose_node) {
  // 遍历 Transpose 的所有数据输入，找到第一个非 Const 的源节点获取输入侧 format
  Format input_side_format;
  bool format_found = false;
  for (size_t i = 0; i < transpose_node.GetInputsSize(); ++i) {
    auto [src_node, src_port] = transpose_node.GetInDataNodesAndPortIndexs(static_cast<int32_t>(i));
    if (src_node == nullptr) {
      continue;
    }
    // 跳过 Const 节点，其 format 通常为 FORMAT_ND，不参与 format 连续性判断
    AscendString src_type;
    if (src_node->GetType(src_type) == GRAPH_SUCCESS && std::string(src_type.GetString()) == "Const") {
      std::cout << "[GraphNodeSettedFormatPass] src_type is " << src_type.GetString() << std::endl;
      continue;
    }
    TensorDesc src_desc;
    if (src_node->GetOutputDesc(src_port, src_desc) != GRAPH_SUCCESS) {
      std::cout << "[GraphNodeSettedFormatPass] src_node get output desc unsuccess" << std::endl;
      continue;
    }
    input_side_format = src_desc.GetFormat();
    format_found = true;
    break;
  }
  if (!format_found) {
    return false;
  }

  // 获取 Transpose output 侧的 format（消费者节点的输入 format）
  auto successors = transpose_node.GetOutDataNodesAndPortIndexs(kTransposeOutput);
  if (successors.empty()) {
    std::cout << "[GraphNodeSettedFormatPass] get successors unsuccess" << std::endl;
    return false;
  }
  for (const auto &[succ_node, succ_port] : successors) {
    if (succ_node == nullptr) {
      return false;
    }
    TensorDesc succ_desc;
    if (succ_node->GetInputDesc(succ_port, succ_desc) != GRAPH_SUCCESS) {
      return false;
    }
    // 任一消费者 format 不一致则不能删除
    if (succ_desc.GetFormat() != input_side_format) {
      std::cout << "[GraphNodeSettedFormatPass] succ_desc format is " << succ_desc.GetFormat()
                << " input_side_format is " << input_side_format << std::endl;
      return false;
    }
  }

  return true;
}

/**
 * @brief 删除 Transpose 节点并将输入直连到输出端消费者。
 *
 *   1. 获取 Transpose 的 data 输入节点和 perm 输入节点
 *   2. 移除 Transpose 的两条输入边
 *   3. 将 Transpose 的输出消费者重连到 Transpose 的 data 输入
 *   4. 移除 Transpose 节点
 *   5. 若 perm 输入节点（通常是 Const）无其他消费者则一并移除
 *
 * @param graph 图指针
 * @param transpose_node 待删除的 Transpose 节点
 * @return true 成功删除, false 删除失败
 */
bool RemoveTransposeAndRelink(const GraphPtr &graph, const GNodePtr &transpose_node) {
  if (transpose_node == nullptr) {
    return false;
  }

  auto [data_node, data_output_index] = transpose_node->GetInDataNodesAndPortIndexs(kTransposeDataInput);
  auto [perm_node, perm_output_index] = transpose_node->GetInDataNodesAndPortIndexs(kTransposePermInput);

  // 移除 Transpose 的两条输入边
  if (data_node != nullptr) {
    if (graph->RemoveEdge(*data_node, data_output_index, *transpose_node, kTransposeDataInput) != GRAPH_SUCCESS) {
      std::cout << "[GraphNodeSettedFormatPass] Remove transpose data input edge failed" << std::endl;
      return false;
    }
  }
  if (perm_node != nullptr) {
    if (graph->RemoveEdge(*perm_node, perm_output_index, *transpose_node, kTransposePermInput) != GRAPH_SUCCESS) {
      std::cout << "[GraphNodeSettedFormatPass] Remove transpose perm input edge failed" << std::endl;
      return false;
    }
  }

  // 将 Transpose 的输出消费者重连到 data 输入
  auto consumers = transpose_node->GetOutDataNodesAndPortIndexs(kTransposeOutput);
  for (const auto &[out_node, out_input_index] : consumers) {
    if (out_node == nullptr) {
      continue;
    }
    if (graph->RemoveEdge(*transpose_node, kTransposeOutput, *out_node, out_input_index) != GRAPH_SUCCESS) {
      std::cout << "[GraphNodeSettedFormatPass] Remove transpose output edge failed" << std::endl;
      return false;
    }
    if (data_node != nullptr) {
      if (graph->AddDataEdge(*data_node, data_output_index, *out_node, out_input_index) != GRAPH_SUCCESS) {
        std::cout << "[GraphNodeSettedFormatPass] Relink edge to transpose consumer failed" << std::endl;
        return false;
      }
    }
  }

  AscendString name;
  std::string name_str = (transpose_node->GetName(name) == GRAPH_SUCCESS) ? name.GetString() : "unknown";
  // 若 perm 输入节点（通常是 Const）无其他消费者则一并移除
  // 先删除 Transpose perm 输入节点，RemoveNode 内部会递归删除 Transpose 所有输入
  if (perm_node != nullptr) {
    if (perm_node->GetOutDataNodesAndPortIndexs(0).empty()) {
      if (graph->RemoveNode(*perm_node) != GRAPH_SUCCESS) {
        std::cout << "[GraphNodeSettedFormatPass] Remove perm const node failed" << std::endl;
      }
    }
  }
  // 移除 Transpose 节点
  if (graph->RemoveNode(*transpose_node) != GRAPH_SUCCESS) {
    std::cout << "[GraphNodeSettedFormatPass] Remove transpose node failed" << std::endl;
    return false;
  }

  std::cout << "[GraphNodeSettedFormatPass] Removed redundant Transpose[" << name_str << "]" << std::endl;

  return true;
}

/**
 * @brief 检查并删除节点输入侧和输出侧的冗余 Transpose。
 *
 * 修改节点 format 后，与其直连的 Transpose 两侧 format 可能变为一致，
 * 此时 Transpose 不再做有意义的格式转换，可以安全删除并将前后节点直连。
 *
 * 检查范围：
 *   - 输入侧：节点的每个 input 的数据源如果是 Transpose，检查其输入侧 format 与输出侧 format 是否一致
 *   - 输出侧：节点的每个 output 的消费者如果是 Transpose，检查其输入侧 format 与输出侧 format 是否一致
 *
 * @param graph 图指针
 * @param node 刚修改完 format 的节点
 * @param node_name 节点名（用于日志）
 */
bool RemoveRedundantTranspose(const GraphPtr &graph, GNode &node, const std::string &node_name) {
  bool all_success = true;
  std::unordered_set<std::string> transpose_names;
  // ----- 按名字收集输入侧的 Transpose -----
  for (size_t i = 0; i < node.GetInputsSize(); ++i) {
    auto [src_node, src_port] = node.GetInDataNodesAndPortIndexs(static_cast<int32_t>(i));
    if (src_node == nullptr) {
      continue;
    }
    if (IsTransposeNode(*src_node) && IsTransposePermConst(*src_node) && HasNoControlEdge(*src_node) &&
        IsTransposeRedundant(*src_node)) {
      AscendString trans_name;
      if (src_node->GetName(trans_name) == GRAPH_SUCCESS) {
        transpose_names.insert(trans_name.GetString());
      }
    }
  }

  // ----- 按名字收集输出侧的 Transpose -----
  for (size_t i = 0; i < node.GetOutputsSize(); ++i) {
    auto successors = node.GetOutDataNodesAndPortIndexs(static_cast<int32_t>(i));
    for (const auto &[succ_node, succ_port] : successors) {
      if (succ_node == nullptr) {
        continue;
      }
      if (IsTransposeNode(*succ_node) && IsTransposePermConst(*succ_node) && HasNoControlEdge(*succ_node) &&
          IsTransposeRedundant(*succ_node)) {
        AscendString trans_name;
        if (succ_node->GetName(trans_name) == GRAPH_SUCCESS) {
          transpose_names.insert(trans_name.GetString());
        }
      }
    }
  }

  // ----- 按名字实时取节点并删除 -----
  for (const auto &trans_name : transpose_names) {
    GNodePtr transpose_node = graph->FindNodeByName(AscendString(trans_name.c_str()));
    if (transpose_node == nullptr) {
      // 节点已被其它 Transpose 的删除级联移除，跳过即可
      std::cout << "[GraphNodeSettedFormatPass] Redundant Transpose[" << trans_name
                << "] not found in graph (maybe removed), skip" << std::endl;
      continue;
    }
    if (!RemoveTransposeAndRelink(graph, transpose_node)) {
      std::cout << "[GraphNodeSettedFormatPass] Remove redundant Transpose[" << trans_name << "] failed" << std::endl;
      all_success = false;
    }
  }
  return all_success;
}

/**
 * @brief 校验待修改端口的原始 format 是否在 kFormatMap 支持范围内。
 *
 * @param node 目标节点
 * @param config 该算子对应的格式配置
 * @param node_name 节点名（用于日志）
 * @return true 全部支持, false 存在不支持的原始 format
 */
bool ValidateOriginalFormats(const GNode &node, const FormatConfig &config, const std::string &node_name) {
  for (const auto &[idx, fmt] : config.input_formats) {
    TensorDesc desc;
    if (node.GetInputDesc(static_cast<int64_t>(idx), desc) != GRAPH_SUCCESS) {
      std::cout << "[GraphNodeSettedFormatPass] Node[" << node_name << "]: cannot get input desc at index " << idx
                << std::endl;
      return false;
    }
    Format old_format = desc.GetFormat();
    if (!IsFormatSupported(old_format)) {
      std::cout << "[GraphNodeSettedFormatPass] Node[" << node_name << "] input." << idx
                << " original format=" << static_cast<int>(old_format)
                << " is not supported (must be FORMAT_NCHW or FORMAT_NHWC), skip" << std::endl;
      return false;
    }
  }
  for (const auto &[idx, fmt] : config.output_formats) {
    TensorDesc desc;
    if (node.GetOutputDesc(static_cast<int64_t>(idx), desc) != GRAPH_SUCCESS) {
      std::cout << "[GraphNodeSettedFormatPass] Node[" << node_name << "]: cannot get output desc at index " << idx
                << std::endl;
      return false;
    }
    Format old_format = desc.GetFormat();
    if (!IsFormatSupported(old_format)) {
      std::cout << "[GraphNodeSettedFormatPass] Node[" << node_name << "] output." << idx
                << " original format=" << static_cast<int>(old_format)
                << " is not supported (must be FORMAT_NCHW or FORMAT_NHWC), skip" << std::endl;
      return false;
    }
  }
  return true;
}

/**
 * @brief 校验算子是否支持修改后的格式组合（Data 节点跳过校验）。
 *
 * @param node 目标节点
 * @param node_name 节点名（用于日志）
 * @param backup_input_formats input format 备份（校验失败时回退）
 * @param backup_output_formats output format 备份（校验失败时回退）
 * @param netoutput_backups NetOutput 备份（校验失败时回退）
 * @return true 校验通过（或 Data 节点跳过）, false 校验失败（已回退）
 */
bool ValidateOpSupported(GNode &node, const std::string &node_name,
                         const std::unordered_map<uint32_t, TensorDescBackup> &backup_input_formats,
                         const std::unordered_map<uint32_t, TensorDescBackup> &backup_output_formats,
                         const std::vector<NetOutputBackup> &netoutput_backups) {
  AscendString node_type;
  bool is_data_node = (node.GetType(node_type) == GRAPH_SUCCESS && std::string(node_type.GetString()) == "Data");
  std::cout << "[GraphNodeSettedFormatPass] Node[" << node_name << "] node_type is " << node_type.GetString()
            << std::endl;

  if (is_data_node) {
    std::cout << "[GraphNodeSettedFormatPass] Node[" << node_name << "]: Data node, skip CheckOpSupported" << std::endl;
    return true;
  }

  std::string reason;
  if (!CheckOpSupported(node, reason)) {
    std::cout << "[GraphNodeSettedFormatPass] Node[" << node_name << "]: CheckOpSupported FAILED! Reason: " << reason
              << std::endl;
    std::cout << "[GraphNodeSettedFormatPass] Rollback format changes for node[" << node_name << "]" << std::endl;
    RollbackFormats(node, backup_input_formats, backup_output_formats);
    RollbackNetOutput(netoutput_backups);
    return false;
  }
  std::cout << "[GraphNodeSettedFormatPass] Node[" << node_name << "]: format applied and CheckOpSupported passed"
            << std::endl;
  return true;
}

/**
 * @brief 对单个节点根据 FormatConfig 修改输入/输出 format，并通过 CheckOpSupported 校验。
 *
 * 流程（参考 user_semantic_inference.cc）：
 *   1. 备份当前 format 和 shape
 *   2. 修改 input/output format 和 shape
 *   3. 将 output format 和 shape 联动传播到直连的 NetOutput 节点
 *   4. CheckOpSupported 校验 → 不支持则回退（含 NetOutput）
 *      注：Data 节点跳过校验，因为 Data 不是计算算子，无对应 kernel
 *   5. 校验通过后，检查并删除前后变冗余的 Transpose 节点
 *
 * @param graph 图指针（用于删除冗余 Transpose 节点）
 * @param node 目标节点
 * @param config 该算子对应的格式配置
 * @param node_name 节点名（用于日志）
 * @return true 修改成功并通过校验, false 修改不可行（已回退）
 */
bool ApplyFormatAndCheck(const GraphPtr &graph, GNode &node, const FormatConfig &config, const std::string &node_name) {
  // ----- Step 0: 校验待修改端口的原始 format 是否在 kFormatMap 支持范围内 -----
  if (!ValidateOriginalFormats(node, config, node_name)) {
    return false;
  }

  // ----- Step 1-2: 备份并应用 format + shape -----
  std::unordered_map<uint32_t, TensorDescBackup> backup_input_formats;
  std::unordered_map<uint32_t, TensorDescBackup> backup_output_formats;

  if (!BackupFormatDescs(node, config, node_name, backup_input_formats, backup_output_formats)) {
    return false;
  }
  if (!ApplyFormatChanges(node, config, node_name, backup_input_formats)) {
    return false;
  }

  // ----- Step 3: 将 output format 联动传播到直连的 NetOutput 节点 -----
  std::vector<NetOutputBackup> netoutput_backups;
  PropagateFormatToNetOutput(node, config, node_name, netoutput_backups);

  // ----- Step 4: CheckOpSupported 校验（Data 节点跳过）-----
  if (!ValidateOpSupported(node, node_name, backup_input_formats, backup_output_formats, netoutput_backups)) {
    return false;
  }

  // ----- Step 5: 检查并删除前后变冗余的 Transpose 节点 -----
  if (!RemoveRedundantTranspose(graph, node, node_name)) {
    std::cout << "[GraphNodeSettedFormatPass] Node[" << node_name
              << "]: some redundant Transpose removal failed, will rollback entire graph" << std::endl;
    return false;
  }

  return true;
}

// ---------- Format 连续性检查 ----------

/**
 * @brief 检查节点配置过的 input 端口与数据源节点之间的 format 连续性。
 *
 * @param node 目标节点
 * @param config 该算子对应的格式配置
 * @param node_name 节点名（用于日志）
 * @return true 连续, false 不连续
 */
bool CheckInputFormatContinuity(const GNodePtr &node, const FormatConfig &config, const std::string &node_name) {
  for (const auto &[idx, expected_fmt] : config.input_formats) {
    auto [src_node, src_port] = node->GetInDataNodesAndPortIndexs(static_cast<int32_t>(idx));
    if (src_node == nullptr) {
      continue;
    }
    TensorDesc src_desc;
    if (src_node->GetOutputDesc(src_port, src_desc) != GRAPH_SUCCESS) {
      continue;
    }
    Format src_format = src_desc.GetFormat();
    if (src_format != expected_fmt) {
      AscendString src_name_asc;
      std::string src_name = (src_node->GetName(src_name_asc) == GRAPH_SUCCESS) ? src_name_asc.GetString() : "unknown";
      std::cout << "[GraphNodeSettedFormatPass] Format discontinuity at input: Node[" << node_name << "] input." << idx
                << " format=" << static_cast<int>(expected_fmt) << " != src Node[" << src_name << "] output."
                << src_port << " format=" << static_cast<int>(src_format) << std::endl;
      return false;
    }
  }
  return true;
}

/**
 * @brief 检查节点配置过的 output 端口与消费者节点之间的 format 连续性。
 *
 * @param node 目标节点
 * @param config 该算子对应的格式配置
 * @param node_name 节点名（用于日志）
 * @return true 连续, false 不连续
 */
bool CheckOutputFormatContinuity(const GNodePtr &node, const FormatConfig &config, const std::string &node_name) {
  for (const auto &[idx, expected_fmt] : config.output_formats) {
    auto successors = node->GetOutDataNodesAndPortIndexs(static_cast<int32_t>(idx));
    for (const auto &[succ_node, succ_in_idx] : successors) {
      if (succ_node == nullptr) {
        continue;
      }
      TensorDesc succ_desc;
      if (succ_node->GetInputDesc(succ_in_idx, succ_desc) != GRAPH_SUCCESS) {
        continue;
      }
      Format succ_format = succ_desc.GetFormat();
      if (succ_format != expected_fmt) {
        AscendString succ_name_asc;
        std::string succ_name =
            (succ_node->GetName(succ_name_asc) == GRAPH_SUCCESS) ? succ_name_asc.GetString() : "unknown";
        std::cout << "[GraphNodeSettedFormatPass] Format discontinuity at output: Node[" << node_name << "] output."
                  << idx << " format=" << static_cast<int>(expected_fmt) << " != dst Node[" << succ_name << "] input."
                  << succ_in_idx << " format=" << static_cast<int>(succ_format) << std::endl;
        return false;
      }
    }
  }
  return true;
}

/**
 * @brief 检查配置文件中修改了 format 的节点，其输入/输出端口与直连节点之间的 format 是否连续。
 *
 * 仅检查已配置成功的节点：
 *   - 对于配置了 input.<idx> 的端口：获取该输入的数据源节点对应输出的 format，比较是否一致
 *   - 对于配置了 output.<idx> 的端口：获取该输出的所有消费者节点对应输入的 format，比较是否一致
 * 若不一致则说明 format 修改后出现了断裂，记录日志并返回 false。
 *
 * 入口处会打印未参与检查的节点：
 *   - 未在图中找到的配置节点
 *   - 在图中找到但未配置成功的节点
 *
 * @param graph 图指针
 * @param op_configs 配置文件中解析的 node_name → FormatConfig 映射
 * @param configured_nodes 已配置成功的节点名集合
 * @return true 所有被修改端口的 format 与直连节点连续, false 存在 format 不连续
 */
bool CheckFormatContinuity(const GraphPtr &graph, const std::unordered_map<std::string, FormatConfig> &op_configs,
                           const std::unordered_set<std::string> &configured_nodes) {
  for (const auto &[node_name, config] : op_configs) {
    if (configured_nodes.find(node_name) == configured_nodes.end()) {
      std::cout << "[GraphNodeSettedFormatPass] Config node[" << node_name
                << "] not successfully configured, skip continuity check" << std::endl;
      continue;
    }

    GNodePtr node = graph->FindNodeByName(AscendString(node_name.c_str()));
    if (node == nullptr) {
      std::cout << "[GraphNodeSettedFormatPass] Config node[" << node_name
                << "] not found in graph, skip continuity check" << std::endl;
      continue;
    }

    if (!CheckInputFormatContinuity(node, config, node_name)) {
      return false;
    }
    if (!CheckOutputFormatContinuity(node, config, node_name)) {
      return false;
    }
  }
  return true;
}

}  // namespace

/**
 * @brief 打印节点的配置信息（用于校验失败时定位问题）。
 */
void PrintNodeConfig(const std::string &node_name, const FormatConfig &config) {
  std::cout << "[GraphNodeSettedFormatPass] Node[" << node_name << "] check failed. Config was:" << std::endl;
  for (const auto &[idx, fmt] : config.input_formats) {
    std::cout << "  input." << idx << " = " << static_cast<int>(fmt) << std::endl;
  }
  for (const auto &[idx, fmt] : config.output_formats) {
    std::cout << "  output." << idx << " = " << static_cast<int>(fmt) << std::endl;
  }
}

/**
 * @brief 遍历目标节点列表，逐个执行 format 修改与校验。
 *
 * @param graph 图指针
 * @param op_configs 配置文件中解析的 node_name → FormatConfig 映射
 * @param target_nodes 待处理的目标节点名列表
 * @param configured_nodes 输出：配置成功的节点名集合
 * @return true 有节点失败, false 全部成功
 */
bool ProcessTargetNodes(const GraphPtr &graph, const std::unordered_map<std::string, FormatConfig> &op_configs,
                        const std::vector<std::string> &target_nodes,
                        std::unordered_set<std::string> &configured_nodes) {
  bool any_failed = false;
  for (const auto &node_name : target_nodes) {
    GNodePtr node = graph->FindNodeByName(AscendString(node_name.c_str()));
    if (node == nullptr) {
      std::cout << "[GraphNodeSettedFormatPass] Config node[" << node_name
                << "] not found (may have been removed by another node), skip" << std::endl;
      continue;
    }

    AscendString node_type_asc;
    std::string node_type = (node->GetType(node_type_asc) == GRAPH_SUCCESS) ? node_type_asc.GetString() : "unknown";
    std::cout << "[GraphNodeSettedFormatPass] Processing node[" << node_name << "] (type=" << node_type << ")"
              << std::endl;

    const auto &config = op_configs.at(node_name);
    if (!ApplyFormatAndCheck(graph, *node, config, node_name)) {
      PrintNodeConfig(node_name, config);
      any_failed = true;
    } else {
      configured_nodes.insert(node_name);
    }
  }
  return any_failed;
}

// =============================================================================
// GraphNodeSettedFormatPass 主类
// =============================================================================

/**
 * GraphNodeSettedFormatPass
 *
 * 功能：
 *   1. 读取当前工作目录下的 custom_formats.cfg，解析节点格式配置。
 *   2. 遍历图中所有节点，以节点名（node_name）匹配配置，修改对应节点的输入/输出 format 和 shape。
 *   3. 若配置了节点输出格式，且该输出直连 NetOutput 节点，则同步修改 NetOutput 对应输入端口的 format 和 shape。
 *   4. 每修改完一个节点，调用 CheckOpSupported 校验算子是否支持该格式组合（Data 节点跳过校验）；
 *      若不支持，回退该节点的修改（含 NetOutput 联动修改），打印节点名和对应配置，返回 FAILED 并回滚整个图。
 *   5. 校验通过后，检查节点前后直连的 Transpose 两侧 format 是否一致，
 *      若一致说明 Transpose 不再做有意义的格式转换，删除该 Transpose 并将前后节点直连。
 *
 * 参考：
 *   - 格式修改 API：参考同目录 DataTransposeFusionPass / ExtendConvTransposeFusionPass
 *   - CheckOpSupported：调用对外接口 GeUtils::CheckNodeSupportOnAicore，
 *     模式参考 compiler/engines/nn_engine/.../user_semantic_inference.cc
 */
class GraphNodeSettedFormatPass : public FusionBasePass {
 public:
  Status Run(GraphPtr &graph, CustomPassContext &pass_context) override {
    std::cout << "GraphNodeSettedFormatPass is starting" << std::endl;

    // ----- 1. 解析配置文件 -----
    auto op_configs = ParseConfigFile();
    if (op_configs.empty()) {
      std::cout << "[GraphNodeSettedFormatPass] No op configs parsed, pass does nothing" << std::endl;
      return SUCCESS;
    }

    // ----- 2. 备份原图（任一个节点失败时回滚整个图）-----
    Graph origin_graph = *graph;

    // ----- 3. 收集与配置匹配的目标节点 -----
    std::vector<std::string> target_nodes;
    for (auto &node : graph->GetDirectNode()) {
      AscendString node_name_asc;
      if (node.GetName(node_name_asc) != GRAPH_SUCCESS) {
        continue;
      }
      const std::string node_name = node_name_asc.GetString();
      if (op_configs.find(node_name) != op_configs.end()) {
        target_nodes.push_back(node_name);
      }
    }

    // ----- 4. 逐个节点执行 format 修改与校验 -----
    std::unordered_set<std::string> configured_nodes;
    bool any_failed = ProcessTargetNodes(graph, op_configs, target_nodes, configured_nodes);
    // ----- 4. 如果任一个节点失败，回滚整个图 -----
    if (any_failed) {
      std::cout << "[GraphNodeSettedFormatPass] Some nodes failed check, rolling back entire graph" << std::endl;
      *graph = origin_graph;
      return SUCCESS;
    }

    // ----- 5. 配置节点 format 连续性检查 -----
    if (!CheckFormatContinuity(graph, op_configs, configured_nodes)) {
      std::cout << "[GraphNodeSettedFormatPass] Format continuity check failed, rolling back entire graph" << std::endl;
      *graph = origin_graph;
      return SUCCESS;
    }

    std::cout << "GraphNodeSettedFormatPass completed successfully" << std::endl;
    return SUCCESS;
  }
};

REG_FUSION_PASS(GraphNodeSettedFormatPass).Stage(CustomPassStage::kAfterOriginGraphOptimize);
