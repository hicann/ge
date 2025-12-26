/**
* Copyright (c) Huawei Technologies Co., Ltd. 2025 All rights reserved.
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

#ifndef AUTOFUSE_VF_PERF_UTILS_H
#define AUTOFUSE_VF_PERF_UTILS_H

#include "base/att_const_values.h"
#include "api_perf_register/perf_param.h"
namespace att {
class VfPerfUtils {
 public:
  // 根据MicroApiType和DataType获取MicroApi级别的性能
  // 场景1，没有直接可以映射的指令，需要基于原有指令进行拼接，可以使用该接口拼接
  // 场景2，有直接可以映射的指令，可以直接调用该接口获取
  static ge::Status GetVfInstructPerf(const std::string &micro_api_type, const std::string &data_type,
                                      Expr &latency, Expr &throughput);
  static ge::Status AddVfInstructPerf(const std::string &vf_instruct_type, const std::string &data_type, Expr &latency,
                                      Expr &throughput, Expr repeat_time);
  // 获取vf头开销
  static Expr GetVFHeadCost();
  // 根据vf function子图解析的结果获取vf function的性能
  static ge::Status GetVectorFunctionPerf(const std::vector<NodePerfInfo> &node_perf_infos, Expr &res);
};
}  // namespace att

#endif  // AUTOFUSE_VF_PERF_UTILS_H
