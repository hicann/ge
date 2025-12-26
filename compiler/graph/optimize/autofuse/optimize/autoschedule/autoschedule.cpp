/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "autoschedule.h"
#include <algorithm>
#include <cstddef>
#include <list>
#include <sstream>
#include <string>
#include <queue>
#include "ascir.h"
#include "ascir_utils.h"
#include "schedule_utils.h"
#include "common_utils.h"
#include "node_utils.h"
#include "ascendc_ir/core/ascendc_ir_impl.h"
#include "graph/symbolizer/symbolic_utils.h"
#include "autoschedule/template_generator_handler.h"

namespace {
void FindNotLoopAxis(const ascir::NodeView &node, ascir::ImplGraph &impl_graph, std::vector<int64_t> &not_loop_axis,
                     bool has_reduce, bool is_reduce_first_stage) {
  for (auto output : node->outputs()) {
    not_loop_axis.insert(not_loop_axis.end(), output->attr.vectorized_axis.begin(), output->attr.vectorized_axis.end());
  }
  if (!has_reduce) {
    return;
  }
  ge::AxisPtr block_axis = nullptr;
  for (const auto &axis : impl_graph.GetAllAxis()) {
    if (axis != nullptr && axis->type == ascir::Axis::Type::kAxisTypeBlockOuter) {
      block_axis = axis;
      break;
    }
  }
  // find reduce axis of input
  for (auto input : node->inputs()) {
    for (size_t i = 0; i < input->attr.repeats.size(); i++) {
      if ((ge::SymbolicUtils::StaticCheckEq(input->attr.repeats[i], ge::sym::kSymbolOne) != ge::TriBool::kTrue) ||
          (ge::SymbolicUtils::StaticCheckEq(input->attr.strides[i], ge::sym::kSymbolZero) != ge::TriBool::kTrue)) {
        continue;
      }
      auto r = impl_graph.FindAxis(input->attr.axis[i]);
      if (r == nullptr) {
        continue;
      }
      if (is_reduce_first_stage) {
        if (block_axis != nullptr &&
            std::find(block_axis->from.begin(), block_axis->from.end(), r->id) != block_axis->from.end()) {
          continue;
        }
      } else {
        if (r->type == ascir::Axis::Type::kAxisTypeBlockOuter || r->type == ascir::Axis::Type::kAxisTypeBlockInner) {
          continue;
        }
      }
      not_loop_axis.push_back(input->attr.axis[i]);
    }
  }
}

bool IsNotLoopAxis(ascir::ImplGraph &impl_graph, int64_t axis, std::vector<int64_t> &not_loop_axis) {
  if (std::find(not_loop_axis.begin(), not_loop_axis.end(), axis) != not_loop_axis.end()) {
    return true;
  }
  auto r = impl_graph.FindAxis(axis);
  if (r == nullptr) {
    return false;
  }
  if (r->from.empty()) {
    return false;
  }
  for (auto c : r->from) {
    if (!IsNotLoopAxis(impl_graph, c, not_loop_axis)) {
      return false;
    }
  }
  return true;
}
}  // namespace

namespace optimize::autoschedule {
Status AutoSchedule::SelectLoopAxis(ascir::ImplGraph &impl_graph) const {
  bool has_reduce = false;
  for (auto node : impl_graph.GetAllNodes()) {
    GE_ASSERT_NOTNULL(node);
    node->attr.sched.loop_axis = ge::kIdNone;
    if (node->attr.api.type != ge::ApiType::kAPITypeCompute) {
      continue;
    }
    if (ScheduleUtils::IsReduce(node)) {
      has_reduce = true;
    }
    auto axis = node->attr.sched.axis;
    std::vector<int64_t> not_loop_axis;
    FindNotLoopAxis(node, impl_graph, not_loop_axis, has_reduce, is_reduce_first_stage_);
    for (auto &s : axis) {
      if (IsNotLoopAxis(impl_graph, s, not_loop_axis)) {
        s = ge::kIdNone;
        continue;
      }
    }

    auto it = std::find_if(axis.rbegin(), axis.rend(), [](const auto &val) { return val != ge::kIdNone; });
    if (it != axis.rend()) {
      node->attr.sched.loop_axis = *it;
    }
    GE_ASSERT_TRUE((node->attr.sched.loop_axis != ge::kIdNone), "Can not find loop axis for node: [%s].",
                   node->GetNamePtr());
  }
  return ge::SUCCESS;
}

void AutoSchedule::GenTilingCase(std::vector<TilingCase> &tiling_cases) {
  auto set_tiling_id = [](auto &field, auto tid) {
    if (tid != kDefaultAxisId) {
      field = tid;
    }
  };

  if (cube_template_ != ascir::CubeTemplateType::kDefault) {
    for (const auto &y_id : axes_group_.y_group) {
      TilingCase tiling_case;
      set_tiling_id(tiling_case.ub_tiling_id_y, y_id);
      tiling_cases.push_back(tiling_case);
    }
    return;
  }
  auto append_reduce_case = [&tiling_cases](const TilingCase &base_case, auto reduce_id) {
    auto reduce_case = base_case;
    reduce_case.block_tiling_id = 1;
    reduce_case.reduce_is_block = true;
    tiling_cases.push_back(reduce_case);
  };

  int64_t attr_axis = -1L;
  int64_t params_size = -1L;
  bool has_gather = ScheduleUtils::GetGatherParams(graph_, attr_axis, params_size);
  if (has_gather && !(attr_axis == params_size - 1 && attr_axis == 0)) {
    int64_t cnt = 0;
    for (const auto &y_id : axes_group_.y_group) {
      if (++cnt == 1) {
        continue;
      }
      TilingCase tiling_case;
      set_tiling_id(tiling_case.ub_tiling_id_y, y_id);
      tiling_case.block_tiling_id = 0;
      tiling_cases.push_back(tiling_case);
    }
    return;
  }

  // 生成通用pattern
  // 遍历所有的group，分别从每个group中取出1个值，组成所有的tiling case
  for (const auto &x_id : axes_group_.x_group) {
    for (const auto &y_id : axes_group_.y_group) {
      for (const auto &r_id : axes_group_.r_group) {
        TilingCase tiling_case;
        set_tiling_id(tiling_case.ub_tiling_id_x, x_id);
        set_tiling_id(tiling_case.ub_tiling_id_y, y_id);
        set_tiling_id(tiling_case.ub_tiling_id_r, r_id);
        tiling_case.block_tiling_id = 0;
        if (is_reduce_first_stage_ && r_id != kDefaultAxisId) {
          append_reduce_case(tiling_case, r_id);
        } else {
          tiling_cases.push_back(tiling_case);
        }
      }
    }
  }
}

// 尽量不生成切size=1轴的tilingcase
Status AutoSchedule::PruneTilingCase(std::vector<TilingCase> &tiling_cases) const {
  auto all_axis = graph_.GetAllAxis();
  for (auto it = tiling_cases.begin(); it != tiling_cases.end();) {
    // 仅处理单切分场景
    if ((it->ub_tiling_id_r != kDefaultAxisId) || (it->ub_tiling_id_x != kDefaultAxisId)) {
      ++it;
      continue;
    }

    auto axis = graph_.FindAxis(it->ub_tiling_id_y);
    GE_ASSERT_NOTNULL(axis, "Tiling case is invalid axis_id:[%ld].", it->ub_tiling_id_y);
    if (ge::SymbolicUtils::StaticCheckEq(axis->size, ge::sym::kSymbolOne) == ge::TriBool::kTrue &&
        tiling_cases.size() > 1UL) {
      it = tiling_cases.erase(it);
      GELOGD("Axis [%s]'s size is 1, will skip to cut.", axis->name.c_str());
    } else {
      ++it;
    }
  }
  return ge::SUCCESS;
}

static std::string GetTilingCaseStr(const std::string &graph_name, const TilingCase &tiling_case) {
  std::stringstream ss;
  ss << graph_name << "_general";
  auto id_str = ascir::utils::IdentifierToStr;
  ss << "_" << id_str(tiling_case.block_tiling_id);
  ss << "_" << id_str(tiling_case.ub_tiling_id_x) << "_" << id_str(tiling_case.ub_tiling_id_y) << "_"
     << id_str(tiling_case.ub_tiling_id_r);
  return ss.str();
}

Status AutoSchedule::DoAutoSchedule() {
  graph_.SetGraphType(ge::AscGraphType::kImplGraph);
  const bool is_reduce_full_load = (reduce_template_ == optimize::ReduceTemplateType::kAllLoad);
  GE_CHK_STATUS_RET(TilingGroup::GenTilingGroup(graph_, axes_group_, is_reduce_full_load),
                    "Gen tiling group failed for graph: [%s]", graph_.GetName().c_str());
  TilingGroup::NormGroup(axes_group_);
  std::vector<TilingCase> tiling_cases;
  GenTilingCase(tiling_cases);
  GE_CHK_STATUS_RET(PruneTilingCase(tiling_cases), "Failed to prune tiling cases for graph: [%s]",
                    graph_.GetName().c_str());
  GE_ASSERT_TRUE(!tiling_cases.empty(), "No valid tiling cases for graph: [%s]. Please check graph legality.",
                 graph_.GetName().c_str());

  const bool is_last_axis_reduce = ScheduleUtils::IsLastAxisReduce(graph_);
  for (size_t index = 0UL; index < tiling_cases.size(); ++index) {
    TilingCase &tiling_case = tiling_cases[index];
    const std::string graph_name = GetTilingCaseStr(ascgen_utils::GenValidName(graph_.GetName()), tiling_case);
    AutoScheduleOutput output(graph_name.c_str());
    GE_ASSERT_TRUE(output.scheduled_graph.CopyFrom(graph_), "Failed to copy graph for tiling case %zu in graph: [%s]",
                   index, graph_.GetName().c_str());

    Scheduler scheduler(output.scheduled_graph, axes_group_, tiling_case, is_last_axis_reduce, reduce_template_,
                        cube_template_);
    GE_CHK_STATUS_RET(scheduler.DoScheduler(), "Scheduler failed for tiling case %zu in graph: [%s]", index,
                      graph_name.c_str());
    if (tiling_case.reduce_is_block) {
      GE_ASSERT_TRUE(
          output.scheduled_graph.BindBlock(tiling_case.block_tiling_id, tiling_case.reduce_block_tiling.second->id));
      output.var_relations_["Rm_org_size"] = tiling_case.rm_org_size;
      output.var_relations_["A_org_size"] = tiling_case.a_org_size;
    }

    GE_CHK_STATUS_RET(SelectLoopAxis(output.scheduled_graph),
                      "Failed to select loop axis for tiling case %zu in graph: [%s]", index, graph_name.c_str());

    schd_outputs_.emplace_back(output);
  }
  // 多模板
  GE_CHK_STATUS_RET(TemplateGeneratorHandler::GenerateTemplates(graph_, schd_outputs_),
                    "Failed to generate templates for graph: [%s]", graph_.GetName().c_str());
  return ge::SUCCESS;
}
}  // namespace optimize::autoschedule
