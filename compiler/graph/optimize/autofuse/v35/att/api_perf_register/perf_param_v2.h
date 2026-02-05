/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ATT_UTIL_PARAMS_V2_H_
#define ATT_UTIL_PARAMS_V2_H_

#include <string>
#include "base/base_types.h"
#include "api_perf_register/perf_param.h"

namespace att {
class PerfParamTableV2 : public PerfParamTable {
 public:
  PerfParamTableV2();
  ~PerfParamTableV2() = default;
  const std::map<std::string, std::vector<VfInstructPerf>> &GetVfInstructPerfTable() const;
  [[nodiscard]] const std::vector<VfInstructPerf> &GetVfInstructPerfTable(
      [[maybe_unused]] const std::string &vf_instruct_type) const override;
  [[nodiscard]] Expr GetVectorFunctionHeadCost() const override;
  [[nodiscard]] std::string GetApiRegisterVerName() const override;
  [[nodiscard]] Expr GetOpHeadCost() const override;
  [[nodiscard]] const std::string *GetAscendCApiPerfTable() const override;
  [[nodiscard]] PipeHeadPerfFunc GetPipeHeadPerfFunc(PipeType pipe_type) const override;
  [[nodiscard]] static Expr GetMTE2PipeHead(const std::vector<NodeInfo> &node_infos, std::map<Expr, TenaryOp, ExprCmp> &tenary_ops);

 private:
  std::map<std::string, std::vector<VfInstructPerf>> vf_instruct_type_2_api_perf_;
  std::map<PipeType, PipeHeadPerfFunc> pipes_head_perf_;
};

class TilingScheduleConfigTableV2 : public TilingScheduleConfigTable {
  [[nodiscard]] bool IsEnableBlockLoopAutoTune() const override{
    return false;
  }
  [[nodiscard]] bool IsEnableCacheLineCheck() const override {
    return true;
  }
  [[nodiscard]] TradeOffConfig GetTradeOffConfig() const override {
    return TradeOffConfig{true, 0.1, 1.0};
  }
  [[nodiscard]] double GetUbThresholdPerfValEffect() const override {
    constexpr double kDefaultUbThresholdPerfValEffect = 0.05;
    return kDefaultUbThresholdPerfValEffect;
  }
  [[nodiscard]] double GetPerfEffectVal() const override {
    constexpr double kDefaultPerfEffectVal = 2000.0;
    return kDefaultPerfEffectVal;
  }
};

class TilingScheduleConfigTableV2HeavyOp : public TilingScheduleConfigTableV2 {
  [[nodiscard]] TradeOffConfig GetTradeOffConfig() const override {
    return TradeOffConfig{true, 0.05, 1.0};
  }
  [[nodiscard]] TilingScheduleConfigPriority GetConfigPriority() const override {
    return TilingScheduleConfigPriority::kHeavyOpPriority;
  }
};
extern const std::string kParamV2Info;
inline const std::string kNddma = "Nddma";
}       // namespace att
#endif  // ATT_UTIL_PARAMS_V2_H_
