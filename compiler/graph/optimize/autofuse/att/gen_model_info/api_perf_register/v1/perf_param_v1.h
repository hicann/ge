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

#ifndef ATT_UTIL_PARAMS_V1_H_
#define ATT_UTIL_PARAMS_V1_H_

#include <string>
#include <map>
#include "api_perf_register/perf_param.h"

namespace att {
class PerfParamTableV1 : public PerfParamTable {
 public:
  PerfParamTableV1();
  ~PerfParamTableV1() = default;
  [[nodiscard]] const std::string *GetAscendCApiPerfTable() const override;
  [[nodiscard]] PipeHeadPerfFunc GetPipeHeadPerfFunc(PipeType pipe_type) const override;
  [[nodiscard]] Expr GetOpHeadCost() const override;
  [[nodiscard]] static Expr GetMTE2PipeHead(const std::vector<NodeInfo> &node_infos, std::map<Expr, TenaryOp, ExprCmp> &tenary_ops);

 private:
  std::map<PipeType, PipeHeadPerfFunc> pipes_head_perf;
};

class TilingScheduleConfigTableV1 : public TilingScheduleConfigTable {
  [[nodiscard]] bool IsEnableBlockLoopAutoTune() const override {
    return true;
  }
  [[nodiscard]] TradeOffConfig GetTradeOffConfig() const override {
    return TradeOffConfig{false};
  }
  [[nodiscard]] double GetUbThresholdPerfValEffect() const override {
    constexpr double kDefaultUbThresholdPerfValEffect = 0.19;
    return kDefaultUbThresholdPerfValEffect;
  }
};

class TilingScheduleConfigTableV1HeavyOp : public TilingScheduleConfigTableV1 {
  [[nodiscard]] bool IsEnableBlockLoopAutoTune() const override{
    return false;
  }
  [[nodiscard]] TradeOffConfig GetTradeOffConfig() const override {
    return TradeOffConfig{true, 0.1, 0.4};
  }
  [[nodiscard]] TilingScheduleConfigPriority GetConfigPriority() const override {
    return TilingScheduleConfigPriority::kHeavyOpPriority;
  }
};
// 临时对外，后面AscAttImpl注册后删除
extern const std::string kParamV1Info;}  // namespace att
#endif  // ATT_UTIL_PARAMS_V1_H_