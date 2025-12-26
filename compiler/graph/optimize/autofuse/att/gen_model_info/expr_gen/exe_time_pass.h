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

#ifndef EXPR_GEN_EXE_TIME_PASS_MANAGER_H_
#define EXPR_GEN_EXE_TIME_PASS_MANAGER_H_

#include <set>
#include "util/tenary_op.h"
#include "parser/tuning_space.h"

namespace att {
class ExeTimePassManager {
public:
  // 解析B轴，a轴，r轴
  explicit ExeTimePassManager(const TuningSpacePtr &tuning_space) {
    broadcast_axis_.clear();
    reduce_axis_.clear();
    brc_buf_node_.clear();
    tuning_space_ = tuning_space;
    std::vector<NodeInfo> nodes = tuning_space->node_infos;
    for (const auto &node : nodes) {
      CheckBroadcast(node);
      CheckReduce(node);
    }
    GenLog("B", broadcast_axis_);
    GenLog("R", reduce_axis_);
    GenLog("A", non_reduce_axis_);
    UpdateBufNode(nodes);
  }
  ~ExeTimePassManager() = default;
  
  // 获取需要处理的节点
  TenaryOp UpdateNodeExeTime(const NodeInfo &node, const Expr &exe_time) const;
private:
  void CheckReduce(const NodeInfo &node);
  void CheckBroadcast(const NodeInfo &node);
  void AddRAxis(const std::string &dim_name, const Expr &repeat, const Expr &stride, const NodeInfo &node_info);
  void AddBAxis(const std::string &dim_name, const Expr &repeat, const Expr &stride, const NodeInfo &node_info);
  void UpdateBufNode(const std::vector<NodeInfo> &nodes);
  void GenLog(const std::string &type_name, const std::map<std::string, std::set<std::string>> &axis_list) const;
  bool GetRLoop(const NodeInfo &node, Expr &r_loop) const;
  TuningSpacePtr tuning_space_;
  std::map<std::string, std::set<std::string>> broadcast_axis_;
  std::map<std::string, std::set<std::string>> reduce_axis_;
  std::map<std::string, std::set<std::string>> non_reduce_axis_;
  std::set<std::string> brc_buf_node_;
};
}

#endif