/**
 * Copyright (C) Huawei Technologies Co., Ltd. 2024 All rights reserved.
 *
 * Licensed unde the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the license is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef ATT_MODEL_INFO_H
#define ATT_MODEL_INFO_H
#include <memory>
#include <map>
#include "base/base_types.h"
#include "schedule_result.h"
#include "util/tenary_op.h"

namespace att {
// NO_TAIL用于modelifno等式约束表达父轴大小要整除子轴，no_tail对应的表达式应为div
const std::string kFatherToChildNoTail = "NO_TAIL";
// NORMAL 用于modelinfo不等式约束中表达父轴大于子轴， normal对应的表达式应为sub
const std::string kFatherToChildLarger = "NORMAL";
enum class kModelInfoLevel : int32_t {
  K_SCHEDULE_RESULT_LEVEL = 0,
  K_SCHEDULE_GROUP_LEVEL,
  K_INVALID_SCHEDULE_LEVEL,
};

struct SymInfo {
  SymInfo() = default;
  explicit SymInfo(const Expr &e) : symbol_expr(e) {}
  virtual ~SymInfo() = default;
  Expr symbol_expr;
  uint32_t prompt_align{1u};
  uint32_t data_type_size{4U};
  std::pair<int64_t, int64_t> value_range = {-1, -1};
};
using SymInfoPtr = std::shared_ptr<SymInfo>;

struct SymVarInfo : public SymInfo {
  explicit SymVarInfo(const Expr &e) : SymInfo(e) {}
  ~SymVarInfo() override = default;
  uint32_t align{1u};
  std::vector<HardwareDef> related_scope;
  Expr max_value;
};
using SymVarInfoPtr = std::shared_ptr<SymVarInfo>;

struct SymConstInfo : public SymInfo {
  explicit SymConstInfo(const Expr &e) : SymInfo(e) {}
  ~SymConstInfo() override = default;
  uint32_t const_value{0u};
};
using SymConstInfoPtr = std::shared_ptr<SymConstInfo>;

struct AttAxis {
  std::string name;  // 轴的名称
  AxisPosition axis_pos;  // 切分轴的位置。origin原始轴，对应求解问题的输入，inner轴，对应待求解变量，outter轴可由origin和inner轴推导
  bool bind_multicore;  // 是否绑多核
  bool is_last;  // 原始轴的最内切分轴，用于设定轴的默认初始值
  bool is_node_innerest_dim;  // 是否是某个node的最内轴，用于决定轴的搜索优先级
  bool is_concat_outer_dim; // 是否是concat node的concat dim外轴
  bool is_concat_inner_dim; // 是否是concat node的concat dim尾轴
  SymInfoPtr size;  // 用于表达轴的size
  std::vector<AttAxis *> orig_axis;  // 原始轴的信息
  std::vector<AttAxis *> from_axis;  // 父轴的信息
  std::map<uint32_t, std::vector<int64_t>> axis_continuous_map; // key:对应输入的索引，value：dim索引范围
};

using AttAxisPtr = std::shared_ptr<AttAxis>;

struct ATTConfig {
  std::vector<std::string> config_names;
  std::map<std::string, std::string> config_value;
};

struct Optional {
  std::string optional_name;
  std::string data_type;
  std::string min_value;
  std::string max_value;
};

struct InputTensor {
  int32_t data_type;
  int32_t format;
};

struct GraphInputInfo {
  std::map<uint32_t, Optional> optional_atts; // graph的可选属性信息 key:index
  std::map<uint32_t, InputTensor> input_atts; // graph的输入tensor信息 key:index
};

struct ScheduleGroupIdent {
  size_t asc_graph_id{0L}; // AscGraph的ID
  size_t impl_graph_id{0L}; // ImplGraph的ID
  size_t group_id{0L}; // ScheduleGroup的ID
  bool operator < (const ScheduleGroupIdent &other) const {
    if (impl_graph_id < other.impl_graph_id) {
      return true;
    } else if (impl_graph_id > other.impl_graph_id) {
      return false;
    }
    // 如果 impl_graph_id 相等，则比较 group_id
    return group_id < other.group_id;
  }
  bool operator == (const ScheduleGroupIdent &other) const {
    return (impl_graph_id == other.impl_graph_id) && (group_id == other.group_id);
  }
  bool operator != (const ScheduleGroupIdent &other) const {
    return (impl_graph_id != other.impl_graph_id) || (group_id != other.group_id);
  }
  [[nodiscard]] std::string GetGroupPrefix() const {
    return "AscGraph" + std::to_string(asc_graph_id) + "ScheduleResult" + std::to_string(impl_graph_id) + "G" +
           std::to_string(group_id);
  }
  // 小写加短下划线风格：snake_case
  [[nodiscard]] std::string GetGroupPrefixSnakeCase() const {
    return "asc_graph" + std::to_string(asc_graph_id) + "_schedule_result" + std::to_string(impl_graph_id) + "_g" +
           std::to_string(group_id);
  }
  [[nodiscard]] std::string GetItemPrefix() const {
    return "graph" + std::to_string(asc_graph_id) + "_result" + std::to_string(impl_graph_id) + "_g" +
           std::to_string(group_id);
  }
};

struct ReuseScheduleGroupInfo {
  std::vector<std::string> reuse_input_axes;  // 复用的schedule group内所有输入轴名称
  std::vector<std::string> reuse_search_axes;  // 复用的schedule group内所有求解轴名称
  std::vector<uint32_t> tiling_keys; // 复用的schedule group内对应的tiling key
};
struct ReuseScheduleGroup {
  ScheduleGroupIdent reuse_group_ident; // 复用的schedule group信息
  ReuseScheduleGroupInfo info;
  std::map<ScheduleGroupIdent, ReuseScheduleGroupInfo>
      schedule_group_to_info;  // 所有schedule group对应的轴名称，映射的轴与reuse_axes对应
  bool IsReuseGroup(const ScheduleGroupIdent &schedule_group_ident) const {
    if (reuse_group_ident == schedule_group_ident) {
      return false;
    }
    const auto &iter = schedule_group_to_info.find(schedule_group_ident);
    if (iter == schedule_group_to_info.end()) {
      return false;
    }
    return (info.reuse_search_axes.size() == iter->second.reuse_search_axes.size()) &&
           (info.reuse_input_axes.size() == iter->second.reuse_input_axes.size()) &&
           (info.tiling_keys.size() == iter->second.tiling_keys.size());
  }
};
using ReuseScheduleGroupPtr = std::shared_ptr<ReuseScheduleGroup>;

enum class TilingScheduleConfigPriority : int32_t {
  kDefaultPriority = 0,
  kHeavyOpPriority = 1,
};

struct TradeOffConfig {
  bool default_enable = false;
  double ub_ratio = 0.1;
  double core_num_ratio = 0.8;
};

class TilingScheduleConfigTable {
 public:
  [[nodiscard]] virtual bool IsEnableBlockLoopAutoTune() const = 0;
  [[nodiscard]] virtual TradeOffConfig GetTradeOffConfig() const = 0;
  [[nodiscard]] virtual TilingScheduleConfigPriority GetConfigPriority() const {
    return TilingScheduleConfigPriority::kDefaultPriority;
  }
  // ub利用率大于该值时，性能公式在模板选择是才会生效
  [[nodiscard]] virtual double GetUbThresholdPerfValEffect() const = 0;
  // 模板比较时，相差超过该值时，才会使用性能公式进行比较，否则直接比较ub利用率
  [[nodiscard]] virtual double GetPerfEffectVal() const {
    constexpr double kDefaultPerfEffectVal = 5000.0;
    return kDefaultPerfEffectVal;
  }
};

struct ModelInfo {
  uint32_t tiling_case_id;
  std::string graph_name;
  std::string score_func;
  std::string sub_case_tag; // Reduce切R，优先切R轴的模板
  Expr workspace_size;  // 用于描述workspace占用
  std::map<HardwareDef, Expr> hardware_cons;  // 用于描述硬件约束
  Expr reserved_ub_size{CreateExpr(0)};
  std::map<std::string, std::vector<std::pair<Expr, Expr>>> eq_exprs;  // 用于描述等式约束,切分轴之间的整除约束key值为NO_TAIL
  std::map<std::string, std::vector<Expr>> leq_exprs;  // 用于描述不等式约束
  std::map<std::string, NodeApiTilingCode> node_name_to_api_code;  // 用于定义高阶API的代码
  std::map<std::string, std::pair<std::string, std::string>>
      tiling_api_name_to_vars;       // 工具场景高阶API使用, API名,高阶API变量名,高阶API变量类型
  std::map<PipeType, Expr> objects;  // 用于描述目标表达式
  std::vector<AttAxisPtr> arg_list;  // 用于描述轴以及轴size的信息, owner att axis ptr
  std::map<std::string, Expr> container_exprs;
  std::map<std::string, Expr> tensor_exprs;
  Expr head_cost{CreateExpr(0)}; // 用于描述多核头开销
  GraphInputInfo graph_input_infos; // graph输入信息
  ScheduleGroupIdent schedule_group_ident; // 标记graph的schedule group信息
  ReuseScheduleGroupPtr reuse_schedule_group; // 标记reuse group信息
  ExprExprMap variable_expr_map; //用于记录tensor的表达式
  std::map<Expr, std::string, ExprCmp> variable_name_map; //用于记录tensor的名称
  std::map<Expr, TenaryOp, ExprCmp> tenary_op_map; //用于记录三目运算符的名称
  uint32_t output_size;
  bool enable_ub_mc_tradeoff{false}; // 使能多核ub权衡，存在非连续搬运的时候使能
  std::vector<ge::AscNodePtr> input_nodes; // 获取输入schedule_results[0].input_nodes
  std::vector<ge::AscNodePtr> output_nodes; // 获取输入出schedule_results[0].output_nodes
  bool contains_heavy_op{false}; // 包含重型算子，比如where
  bool enable_group_parallel{false}; // 使能group并行
  std::vector<Expr> sizes{}; // 图原始Sizes信息
  const TilingScheduleConfigTable *tiling_schedule_config_table{nullptr};
};

using TilingModelInfo = std::vector<ModelInfo>;
using GroupsTilingModelInfo = std::map<size_t, TilingModelInfo>;
// score_funcs: {level, {asc_graph_id, {impl_graph_id, score_func}}
using ScoreFuncs = std::map<kModelInfoLevel, std::map<size_t, std::map<size_t, std::string>>>;
using EnableGroupParallels = std::map<size_t, std::map<size_t, bool>>;
using VarRelations = std::map<size_t, std::map<size_t, std::map<size_t, std::map<size_t, std::map<std::string, ge::Expression>>>>>;
// schedule result id->{score_func, group_model_infos}
struct ParsedScheduleResult {
  size_t asc_graph_id{0UL};
  size_t impl_graph_id{0UL};
  std::string score_func;
  std::map<size_t, std::map<size_t, std::map<std::string, ge::Expression>>> var_relations;
  GroupsTilingModelInfo groups_tiling_model_info;
  bool enable_group_parallel{false};
};
using FusedParsedScheduleResult = std::map<size_t, std::map<size_t, ParsedScheduleResult>>;
}
#endif

