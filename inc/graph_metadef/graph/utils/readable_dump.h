/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef INC_GRAPH_UTILS_READABLE_DUMP_H_
#define INC_GRAPH_UTILS_READABLE_DUMP_H_

#include <sstream>
#include <string>
#include <vector>
#include <google/protobuf/text_format.h>

#include "graph/debug/ge_util.h"
#include "common/checker.h"
#include "graph/node.h"
#include "graph/serialization/attr_serializer_registry.h"

namespace ge {
static constexpr const char *const kIndentZero = "";   // 0个空格
static constexpr const char *const kIndentTwo = "  ";  // 2个空格
static constexpr const char *const kNetOutput = "NetOutput";

class ReadableDump {
 private:
  class OutputHandler;

 public:
  ReadableDump() = delete;
  ReadableDump(const ReadableDump&) = delete;
  ReadableDump& operator=(const ReadableDump&) = delete;
  ~ReadableDump() = delete;

  /**
   * @brief 生成Readable Dump主函数
   * @param readable_ss 字符串流
   * @param graph Dump图
   */
  static Status GenReadableDump(std::stringstream &readable_ss, const ComputeGraphPtr &graph);

 private:
  class OutputHandler {
   public:
    OutputHandler() = default;
    ~OutputHandler() = default;

    std::unordered_map<std::string, std::shared_ptr<std::vector<std::string>>> &GetNodeToOutputsMap() {
      return node_to_outputs_;
    }
    std::string GetOutputRet() {
      std::stringstream index;
      if (index_ == 0) {
        index << "ret";
        index_++;
      } else {
        index << "ret_" << index_++;
      }
      return index.str();
    }
    void GenNodeToOutputsMap(const ge::ComputeGraphPtr &graph) {
      for (const auto &node : graph->GetDirectNode()) {
        if (node->GetOpDesc()->GetType() != kNetOutput) {
          std::shared_ptr<std::vector<std::string>> output_rets = ComGraphMakeShared<std::vector<std::string>>();
          if (output_rets == nullptr) {
            REPORT_INNER_ERR_MSG("E18888", "Initial output vector failed");
            GELOGE(GRAPH_FAILED, "[OutputHandler][GenNodeToOutputsMap] failed to initial output vector");
            return;
          }
          if (node->GetAllOutDataAnchorsPtr().size() <= 1) {
            output_rets->emplace_back(node->GetName());
          } else {
            for (const auto output_idx : node->GetAllOutDataAnchorsPtr()) {
              if (output_idx != nullptr) {
                output_rets->emplace_back(GetOutputRet());
              }
            }
          }

          node_to_outputs_.emplace(node->GetName(), output_rets);
        }
      }
    }

   private:
    int32_t index_ = 0;
    std::unordered_map<std::string, std::shared_ptr<std::vector<std::string>>> node_to_outputs_{};
  };

  /**
   * @brief 生成节点Readable Dump
   * @param readable_ss 字符串流
   * @param node Dump节点
   * @param output_handler 节点输出处理器
   */
  static void GenNodeDump(std::stringstream &readable_ss, OutputHandler &output_handler, const Node *node);

  /**
   * @brief 获取实例名称
   * @param name 实例名
   * @param indent 前空行
   * @return Readable Dump节点名或输出名
   */
  static std::string GetInstanceName(const std::string &name, const std::string &indent = kIndentTwo);

  /**
   * @brief 获取节点出度
   * @param node 节点
   * @return 节点出度字符串
   */
  static std::string GetNodeOutDegree(const Node *node);

  /**
   * @brief 获取节点IR类型
   * @param node 节点
   * @return 节点IR类型字符串
   */
  static std::string GetNodeType(const Node *node);

  /**
   * @brief 获取节点入参实例
   * @param node 节点
   * @param output_handler 节点输出处理器
   * @return 节点实例入参字符串
   */
  static std::string GetNodeInputInstance(const Node *node, OutputHandler &output_handler);

  /**
   * @brief 生成节点的可读入参信息
   * @param readable_ss 字符串流
   * @param node 节点
   * @param output_handler 节点输出处理器
   */
  static void GenNodeInputs(std::stringstream &readable_ss, const Node *node, OutputHandler &output_handler);

  static std::string GetAttrValueStr(const OpDescPtr &op_desc, const std::string &attr_name, const AnyValue &attr_value,
                                     const std::string &av_type);

  /**
   * @brief 生成节点属性信息
   * @param readable_ss 字符串流
   * @param node 节点
   */
  static void GenNodeAttrs(std::stringstream &readable_ss, const Node *node);

  /**
   * @brief 获取节点输出实例的出度信息
   * @param node 节点
   * @param index 节点输出下标
   * @return 节点输出实例的出度字符串
   */
  static std::string GetOutputOutDegree(const Node *node, int32_t index);

  /**
   * @brief 当节点包含多数出时，生成多数出信息，输出实例名称按照'ret', 'ret_1', 'ret_2'递增
   * @param readable_ss 字符串流
   * @param node Dump节点
   * @param output_handler 输出处理器
   */
  static void GenMultipleOutputsIfNeeded(std::stringstream &readable_ss, const Node *node,
                                         OutputHandler &output_handler);

  /**
   * @brief 获取图的输出实例信息
   * @param graph_output_ss 字符串流
   * @param net_output 图的NetOutput节点
   * @param output_handler 节点输出处理器
   */
  static void GenGraphOutput(std::stringstream &graph_output_ss, const Node *net_output, OutputHandler &output_handler);
};
}  // namespace ge

#endif  // INC_GRAPH_UTILS_READABLE_DUMP_H_
