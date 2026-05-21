/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph/fusion/pass/pattern_fusion_run.h"

#include "common/checker.h"
#include "common/plugin/ge_make_unique_util.h"
#include "framework/common/debug/ge_log.h"
#include "ge/fusion/graph_rewriter.h"
#include "ge/fusion/pattern_matcher.h"
#include "ge/fusion/subgraph_boundary.h"
#include "graph/fusion/fusion_utils.h"
#include "graph/utils/graph_utils_ex.h"

namespace ge {
namespace fusion {
Status RunPatternFusion(GraphPtr &graph,
                        CustomPassContext &pass_context,
                        const PatternMatcherConfig &match_config_template,
                        std::vector<PatternUniqPtr> patterns,
                        const MeetRequirementsFn &meet_requirements,
                        const ReplacementFn &replacement) {
  bool is_changed = false;
  std::string pass_name = pass_context.GetPassName().GetString();
  for (auto &pattern : patterns) {
    int32_t effect_times = 0;
    int32_t match_times = 0;
    auto pattern_graph = pattern->GetGraph();
    auto match_config = std::make_unique<PatternMatcherConfig>(match_config_template);
    GE_ASSERT_NOTNULL(match_config);
    PatternMatcher matcher(std::move(pattern), graph, std::move(match_config));
    std::unique_ptr<MatchResult> match_result;
    while (match_result = matcher.MatchNext(), match_result != nullptr) {
      match_times++;
      if (!meet_requirements(match_result)) {
        GELOGD("Match result[%s] is not meet requirements, skip replace.", match_result->ToAscendString().GetString());
        continue;
      }
      if (FusionUtils::WillCauseCycleIfFuse(match_result)) {
        GELOGI("[Replace]Skip to replace match result [%s] in case of causing cycle after fusion.",
               match_result->ToAscendString().GetString());
        continue;
      }
      auto boundary = match_result->ToSubgraphBoundary();
      GE_ASSERT_NOTNULL(boundary);
      const auto replacement_graph = replacement(match_result);
      GE_ASSERT_NOTNULL(replacement_graph);
      (void)FusionUtils::MarkPassNameOnReplacementNodes(replacement_graph, boundary, pass_name);
      if (SubgraphRewriter::Replace(*boundary, *replacement_graph) != SUCCESS) {
        AscendString replacement_name;
        GE_ASSERT_GRAPH_SUCCESS(replacement_graph->GetName(replacement_name));
        GELOGE(FAILED, "Failed to replace %s to %s", match_result->ToAscendString().GetString(),
               replacement_name.GetString());
        return FAILED;
      }
      GE_ASSERT_SUCCESS(
          FusionUtils::UpdateToCycleDetector(GraphUtilsEx::GetComputeGraph(*graph), match_result, replacement_graph));
      if (!is_changed) {
        is_changed = true;
      }
      GELOGI("Replace [%s] to [%s] success", match_result->ToAscendString().GetString(),
             FusionUtils::ToString(replacement_graph).c_str());
      effect_times++;
    }
    AscendString pattern_name;
    GE_ASSERT_GRAPH_SUCCESS(pattern_graph.GetName(pattern_name));
    auto compute_graph = GraphUtilsEx::GetComputeGraph(*graph);
    FusionUtils::RecordFusionStatistic(compute_graph->GetSessionID(), to_string(compute_graph->GetGraphID()),
                                       pass_name, match_times, effect_times);
    GELOGD("GraphId[%d], GraphFusionPass[%s]: pattern=%s, matched_times=%d, effected_times=%d",
           compute_graph->GetGraphID(), pass_name.c_str(), pattern_name.GetString(), match_times, effect_times);
  }
  return is_changed ? SUCCESS : NOT_CHANGED;
}
} // namespace fusion
} // namespace ge
