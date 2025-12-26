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

#ifndef ATT_TILING_CODE_GEN_IMPL_H_
#define ATT_TILING_CODE_GEN_IMPL_H_

#include <string>
#include <set>
#include <memory>
#include "code_printer.h"
#include "base/model_info.h"
#include "generator_config.h"
#include "tiling_data_gen/tiling_data_generator.h"
#include "extra_info_gen/extra_info_generator.h"
#include "util/duration.h"
#include "gen_model_info/api_tiling_gen/gen_api_tiling.h"

namespace att {
class TilingCodeGenImpl {
  // asc_graph_id->impl_graph_id->schedule_result_id->(score_func_name, score_func_impl)
  using AscGraphNamepspaceMap = std::map<size_t, std::map<size_t, std::pair<std::string, std::string>>>;
  using FusedGraphNamespaceMap = std::map<size_t, AscGraphNamepspaceMap>;

 public:
  TilingCodeGenImpl(const std::string &op_name, const TilingCodeGenConfig &config,
                    const TilingModelInfo &tiling_model_info, const ScoreFuncs &score_funcs, const bool is_uniq_group);
  virtual ~TilingCodeGenImpl() = default;
  
  ge::Status GenTilingHead(std::map<std::string, std::string> &tiling_res,
                          const EnableGroupParallels &enable_group_parrallels = {});
  ge::Status GenTilingTail(std::map<std::string, std::string> &tiling_res,
                           const std::unordered_map<std::string, std::string>& cache_reuse_info = {},
                           VarRelations var_relations = {},
                           const EnableGroupParallels &enable_group_parallels = {});
  ge::Status GenTiling(std::map<std::string, std::string> &tiling_res,
                       std::unordered_map<std::string, std::string> cache_reuse_info = {},
                       uint32_t cache_capacity = 0,
                       const EnableGroupParallels &enable_group_parrallels = {});

 protected:
  // 用于判断求解器是否有效
  ge::Status CheckImplPtr(const std::string &indent);
  ge::Status GetReuseVarNames(std::map<std::string, std::string> &var_names_to_reuse_var_name);
  // 用于构造一个用于复制的结构体
  ge::Status GenStructCopyDef();
  // 用于构造一个用于缓存复用的哈希表
  ge::Status GenCacheHashMapDef();

  // 用于生成duration相关的代码段
  ge::Status GenDurationBeginCode(const TilingFuncDurationType type, const std::string &indent);
  ge::Status GenDurationEndCode(const TilingFuncDurationType type, const std::string &indent);

  // schedule group相关
  ge::Status ObtainInnerParams(std::map<std::string, std::set<std::string>> &hardware_map,
                               FusedGraphNamespaceMap &namespace_map);
  // 生成sche group的tiling函数初始化部分
  ge::Status GenGetTilingForAllInitLines(bool pgo = false);
  ge::Status GenGetResultSummary(
      const size_t asc_graph_id,
      const std::map<size_t, std::map<size_t, std::pair<std::string, std::string>>> &namespace_map);
  // 生成sche group的tiling函数，不支持工具场景
  ge::Status GenGetTilingForScheduleResult();
  ge::Status GenFusedScheduleResultsGetTilingDefine(const FusedGraphNamespaceMap &namespace_map);
  ge::Status GenEnableGroupParallelFunctions(const FusedGraphNamespaceMap &namespace_map);
  ge::Status GenEnableGroupParallelInvoke(size_t asc_graph_id, const AscGraphNamepspaceMap &asc_graph_namespace_map);
  ge::Status GenEnableGroupParallelPgoInvoke(const std::string &tiling_name, const std::string &access,
                                             const std::string &indent, std::string &invoke_code);
  ge::Status GenPGOFusedScheduleResultsGetTilingDefine(const FusedGraphNamespaceMap &namespace_map);
  ge::Status GenPGOByCoreNumFusedScheduleResultsGetTilingDefine(const FusedGraphNamespaceMap &namespace_map);
  ge::Status GenPGOByCoreNumSearchTilingKeyCollectTilingData(FusedGraphNamespaceMap namespace_map);
  void GenGetScoreFuncs(const size_t asc_graph_id,
                        const std::map<size_t, std::map<size_t, std::pair<std::string, std::string>>> &namespace_map);
  ge::Status GenPGOGetTilingForAll();
  void GenGetScoreFuncsCalling(
      const size_t asc_graph_id,
      const std::map<size_t, std::map<size_t, std::pair<std::string, std::string>>> &namespace_map);
  // 生成sche group的cache初始化部分
  void GenCacheInit();
  void GenSetHardwareCodes(const std::string& group_prefix, const std::set<std::string>& hardware_names);
  void GenGetScheduleResultTail(const std::map<size_t, std::pair<std::string, std::string>> &graph_info);
  void GenGetScheduleResult(const size_t asc_graph_id, const size_t impl_graph_id,
                            const std::map<size_t, std::pair<std::string, std::string>> &graph_info,
                            const std::map<std::string, std::set<std::string>> &hardware_map,
                            const std::map<size_t, std::map<size_t, std::map<std::string, ge::Expression>>> &var_relation);
  void GenGetMaxScoreIndex(const AscGraphNamepspaceMap &namespace_map);

  void ProcessGroupInfo(const std::map<size_t, std::pair<std::string, std::string>> &graph_info,
                            const std::map<std::string, std::set<std::string>> &hardware_map,
                            std::string &check_cond, std::string &cal_perf, std::vector<std::string> &block_num);
  void GenPGOGetScheduleResult(const size_t impl_graph_id,
                            const std::map<size_t, std::pair<std::string, std::string>> &graph_info,
                            const std::map<std::string, std::set<std::string>> &hardware_map);
  void GenScheduleResultGetTilingCalling(const std::string &index, const std::string &ident = "");
  void GenGetAllSchedulesResults(const AscGraphNamepspaceMap &namespace_map);
  void GenPGOGetScheduleResult(const size_t asc_graph_id, const size_t impl_graph_id,
                            const std::map<size_t, std::pair<std::string, std::string>> &graph_info,
                            const std::map<std::string, std::set<std::string>> &hardware_map);
  void GenPGOGetAllSchedulesResults(const size_t asc_graph_id, const AscGraphNamepspaceMap &namespace_map);
  void GenPGOByCoreNumGetScheduleResult(const size_t asc_graph_id, const size_t impl_graph_id,
                                        const std::map<size_t, std::pair<std::string, std::string>> &graph_info,
                                        const std::map<std::string, std::set<std::string>> &hardware_map,
                                        const std::map<size_t, std::map<size_t, std::map<std::string, ge::Expression>>> &var_relation);
  std::string GenLaunchLikeInputOutputDef(bool is_define = true);
  ge::Status GenCastReuseTilingDataCode(const ReuseScheduleGroupInfo &reuse_info, const ReuseScheduleGroupInfo &info);
  // -----------------------小shape优化相关---------------------------
  bool HitSmallShapePattern(ArgsManager &args_manager) const;
  // 生成TilingCaseNum获取接口函数
  ge::Status GenGetTilingOptionRange();
  // 生成GetTiling的PGO接口函数
  ge::Status GenGetTilingWithOption();
  ge::Status GenGetTilingWithCaseId(bool no_cache = false);

  // 构造公用的check函数
  ge::Status GenPublicCheck();
  // 对tiling context作校验
  ge::Status GenCheckInputVars();
  // 由tiling context生成tiling data信息
  ge::Status GenGetShapeAttrsInfo(const ModelInfo &model_info);
  // 生成输入判断合法性的逻辑
  ge::Status GenCheckIsCapable(const ModelInfo &model_info);
  // 生成硬件信息的日志语句
  ge::Status GenHardwareSummary(const ModelInfo &model_info);
  // 生成硬件信息的判断语句
  ge::Status GenHardwareJudge(const ModelInfo &model_info);
  // 生成输入信息的日志语句
  ge::Status GenInputSummary(const ModelInfo &model_info);
  // 生成目标函数
  ge::Status GenGetObj(const ModelInfo &model_info);
  // 生成score函数
  ge::Status GenCalcScore(const ModelInfo &model_info);
  // 生成score计算相关变量
  ge::Status GenCalcScoreVars();
  // 生成ub/corenum相关的tiling的上限值
  void InitTilingUpperBound(const std::vector<Expr> &hardware_args, const ArgsManager &args_manager, 
    const HardwareDef &hardware_def, std::map<std::string, bool> &visited);
  ge::Status GenSmallShapeTiling(const ModelInfo &model_info);
  // 生成求解器的基类
  virtual ge::Status GenSolverBaseClass() = 0;
  // 生成由solver_pass_gen构造的求解器子类
  virtual ge::Status GenSolverTiling(const ModelInfo &model_info) = 0;
  // 调用求解器的函数
  virtual ge::Status GenDoTiling(const ModelInfo &model_info) = 0;
  // 获取tiling data拷贝
  virtual ge::Status GenGetTilingDataFromCopy();
  // 缓存复用
  virtual ge::Status GenFindCacheAndSaveCache();
  // 更新最优模板
  virtual ge::Status GenUpdateBetterTiling();
  // 根据目标表达式和ub占用率选择更好的模板
  virtual ge::Status GenSelectBetterTilingBasedOnObjAndUbRatio();
  // 寻找最优模板
  virtual ge::Status GenFindPerfBetterTilingbyCaseId();
  virtual ge::Status GenSearchAllTilingbyCaseId();
  // 多模板情况下算法的模板选择逻辑
  virtual ge::Status GenGetTilingKey();
  virtual ge::Status GenPGOSearchTilingKey();
  virtual ge::Status ValidateSingleResultAndGroup();
  // 保存模板数
  ge::Status GenSaveCaseNumInfo(uint32_t case_num);
  // 生成初始化缓存并查询缓存逻辑
  virtual ge::Status GenInitAndQueryCacheCode();
  // 根据caseid生成选择逻辑
  virtual ge::Status GenGetTilingbyCaseId();
  virtual ge::Status GenPGODefaultTiling();
  virtual ge::Status GenPGOTilingCase(const ModelInfo& model_info);
  virtual ge::Status GenPGOEvaluatePerf(const std::string& tiling_id_str);
  virtual ge::Status GenPGOFinalize();
  virtual ge::Status GenPGOGetTilingbyCaseId();
  virtual ge::Status GenerateInputParamsAndTiling();
  virtual ge::Status GenPGOByCoreNumSearchTilingKeySingleGroup();
  virtual ge::Status GenPGOByCoreNumSearchTilingKey();
  virtual ge::Status GenPGOByCoreNumTilingForAll();
  void GenPGOByCoreNumDoTiling(const std::pair<size_t, std::pair<std::string, std::string>> &group_info,
                               const uint32_t group_index, const size_t asc_graph_id, const size_t impl_graph_id);
  void GenPGOByCoreNumGetAllSchedulesResults(const size_t asc_graph_id, const AscGraphNamepspaceMap &namespace_map);
  //该函数用于构造tiling data的set与get内容
  virtual ge::Status GenExtraParamCode(const ModelInfo &model_info, std::string &pass_code);
  virtual ge::Status GenGetSetTilingImpl(const ModelInfo &model_info);
  // 在tiling data头文件中生成外部函数的定义
  virtual ge::Status GenExternFuncDef();
  // 生成宏函数与include信息
  virtual ge::Status GenMacroInclude();
  // 生成工具函数
  virtual ge::Status GenToolFuncs();
  // 生成tilingimpl的基类public函数
  virtual ge::Status GenTilingImplPublicFunc();
  // 生成求解器子类
  virtual ge::Status GenTilingCaseImpl(const ModelInfo &model_info);
  // 生成预处理函数
  virtual ge::Status GenPreTiling(const ModelInfo &model_info);
  // 生成高阶api的tiling
  virtual ge::Status GenDoApiTiling(const ModelInfo &model_info);
  // 提供tiling data的额外性能评估函数
  virtual ge::Status GenExtraEvalFunc(const ModelInfo &model_info);
  // 生成基于基本tiling参数计算其他参数的逻辑，如外轴大小等
  virtual ge::Status GenExtraTilingData(const ModelInfo &model_info);
  // 生成tiling评估打印
  virtual ge::Status GenExtraSummaryInfo(const ArgsManager &args_manager, std::string &case_info_str);
  // 生成不同pipe的obj
  virtual ge::Status GenPipeTypeObj(const ModelInfo &model_info);
  //该函数用于生成memory tiling的相关参数
  virtual ge::Status GenMemoryParamCode(const ModelInfo &model_info);
  virtual ge::Status GenExtraTilingFuncImpl(const ModelInfo &model_info);
  virtual ge::Status GenExtraTilingFuncInvoke(const ModelInfo &model_info);

  ge::CodePrinter tiling_data_;
  ge::CodePrinter tiling_func_;
  ge::CodePrinter tiling_head_;
  std::string op_name_;
  TilingCodeGenConfig config_;
  ExtraInfoConfig extra_info_config_;
  TilingDataGenerator tiling_data_manager_;
  ExtraInfoGenerator extra_info_generator_;
  const TilingModelInfo &tiling_model_info_;
  bool is_uniq_group_{true};  // 表示是否是唯一的ScheduleGroup，大部分场景不会切分成多个ScheduleGroup，所以默认为true
  bool hardware_has_ub_{false};  // 表示model_info的hardware_cons中是否包含UB，如果包含的话在选择模板时同时考虑目标表达式和UB占用率
  // schedule result打分函数, 当前支持ScheduleResult的选择，考虑未来支持其他级别的打分选择
  ScoreFuncs score_funcs_;
  std::unordered_map<std::string, std::string> cache_reuse_info_{};
  VarRelations var_relations_{};
  EnableGroupParallels enable_group_parallels_{};
  uint32_t cache_capacity_{0};
  bool with_reuse_info_{false};
  std::string arrange_code_;

 private:
  ge::Status GenExpressionMacro();
  // 用于获取不同硬件信息的获取代码
  ge::Status GetRelatedHardware(std::map<std::string, std::string> &hardware_info);

  // 用于生成duration相关的代码段
  ge::Status GenDurationCommonCode();
  ge::Status GenDurationPrintCode(const std::string &indent);
  ge::Status GenDurationClearCode(const std::string &indent);
  
  // -----------------------以下函数生成tilingdata------------------------
  ge::Status GenProtectedVars();
  ge::Status GenBaseTilingData(std::map<std::string, std::string> &type_name_to_definition);
  ge::Status GenHeaderCodesHead();
  ge::Status GenHeaderCodesTail();
  ge::Status GenHeaderCodesBody();
  ge::Status GenHeaderCodesSummaryBody();
  ge::Status GenHeaderInclude();
  ge::Status GenHeaderVarsDef();
  ge::Status GenScheduleGroupTilingHead();
  ge::Status GenScheduleGroupTilingTail();

  // -----------------------以下函数生成tilingimpl的基类------------------------
  ge::Status GenGetTiling();
  ge::Status GenTilingImplBaseClass();

  //  -----------------------以下函数生成固定的check函数------------------------
  ge::Status UpdateAttInfo(std::map<uint32_t, std::string> &dtype_map);
  ge::Status GenCheckNumFunc();
  ge::Status GenCheckDtypeFunc();
  ge::Status GenCheckFormatFunc();
  ge::Status GenCheckShapeDimFunc();
  ge::Status GenCheckAttr();
  ge::Status GenCheckParams(const ModelInfo &model_info);
  
  // 生成公共框架代码，不同类型的求解器可以自行构建基类及求解器信息
  ge::Status GenCommonFrameWork();
  // 生成公共框架结构体定义，不同类型求解均需要使用
  ge::Status GenCommonStruct();
  ge::Status GenUsedTilingOption();
  // 由tilingcontext获取硬件参数
  ge::Status GenGetPlatformInfo();

  // 生成硬件只占用的评估函数
  ge::Status GenHardwareCons(const ModelInfo &model_info);
  // 生成workspace的大小回填
  ge::Status GenGetWorkspaceSize(const ModelInfo &model_info);
  // 生成性能评估函数
  ge::Status GenEvalFunc(const ModelInfo &model_info);
  // 生成TilingSummary函数
  ge::Status GenTilingSummary(const ModelInfo &model_info);
  // 生成后处理函数
  ge::Status GenPostTiling(const ModelInfo &model_info);

  // 由tilingkey获取对应的tilingimpl的指针
  ge::Status GenImplPtr();
  // 由tilingkey获取对应的性能公式
  ge::Status GenGetPerf();
  // 由tilingkey获取对应的日志信息
  ge::Status GenGetSummary();
  ge::Status GenReuseGroupTilingWrapperGetTiling(
      const std::string &cur_prefix, const std::string &reuse_prefix, const ReuseScheduleGroupInfo &reuse_info,
      std::map<ScheduleGroupIdent, ReuseScheduleGroupInfo>::const_iterator iter);
  ge::Status GenReuseGroupTilingWrapperGetPerf(
      const std::string &cur_prefix, const std::string &reuse_prefix, const ReuseScheduleGroupInfo &reuse_info,
      std::map<ScheduleGroupIdent, ReuseScheduleGroupInfo>::const_iterator iter);
  ge::Status GenReuseGroupTilingWrapperGetSummary(
      const std::string &cur_prefix, const std::string &reuse_prefix, const ReuseScheduleGroupInfo &reuse_info,
      std::map<ScheduleGroupIdent, ReuseScheduleGroupInfo>::const_iterator iter);
  ge::Status GenReuseGroupTilingWrapper(std::map<std::string, std::string> &tiling_res);
  ge::Status GenPGOReuseGroupTilingWrapper();
  ge::Status GenTilingKeyFunc();
  void GenTilingHeadMultiGroup();

  // -----------------------生成固定的入口函数---------------------------
  ge::Status GenGetTilingHeadImpl();
  ge::Status GenGetTilingImpl();
  ge::Status GenIsStaticShape();
  ge::Status GenCtxGetTilingImpl();
  ge::Status GenPostTilingImpl();
  ge::Status GenParseInfo();
  ge::Status GenTilingFuncCallEntrance();
  ge::Status GenGeneralTiling(const ModelInfo &model_info);
  
  ge::Status GenVariableAnnotation(const ArgsManager &args_manager);

  ge::Status GenOpLog(const std::string &indent, const std::string &log);
  ge::Status GenOpLog(const std::string &indent, const std::string &uniq_log, const std::string &sched_log);
};

using TilingCodeGenImplPtr = std::shared_ptr<TilingCodeGenImpl>;
}  // namespace att
#endif
