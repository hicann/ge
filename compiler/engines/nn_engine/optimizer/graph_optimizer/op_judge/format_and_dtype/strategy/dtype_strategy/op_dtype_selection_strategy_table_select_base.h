/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
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

 #ifndef AIR_COMPILER_ENGINES_NNENG_OPTIMIZER_GRAPH_OPTIMIZER_OP_JUDGE_FORMAT_AND_DTYPE_STRATEGY_DTYPE_STRATEGY_OP_DTYPE_SELECTION_STRATEGY_TABLE_SELECT_BASE_H_
 #define AIR_COMPILER_ENGINES_NNENG_OPTIMIZER_GRAPH_OPTIMIZER_OP_JUDGE_FORMAT_AND_DTYPE_STRATEGY_DTYPE_STRATEGY_OP_DTYPE_SELECTION_STRATEGY_TABLE_SELECT_BASE_H_
 
 #include "graph_optimizer/op_judge/format_and_dtype/strategy/matcher/dtype/op_dtype_precise_matcher.h"
 #include "op_dtype_selection_strategy_base.h"
 #include "op_dtype_selection_strategy_table_select_map.h"
 
 namespace fe {
 using OpDtypePreciseMatcherPtr = std::shared_ptr<OpDtypePreciseMatcher>;
 
 class OpDtypeSelectionStrategyTableSelectBase : public OpDtypeSeletionStrategyBase {
  public:
   OpDtypeSelectionStrategyTableSelectBase(
       FormatDtypeQuerierPtr format_dtype_querier_ptr,
       OpDtypePreciseMatcherPtr op_dtype_precise_matcher_ptr);
 
   ~OpDtypeSelectionStrategyTableSelectBase() override = default;
 
   Status GetOpPrecisionPolicy(const OpKernelInfoPtr &op_kernel_info_ptr,
       PrecisionPolicy &precision_policy) const;
 
   void GetDtypeMatchListFromMap(const std::map<ge::DataType, std::vector<ge::DataType>> &dtype_match_map,
       const ge::DataType &key_dtype, vector<ge::DataType> &match_dtype_vec) const;
 
   Status MatchDtypeFromList(const vector<ge::DataType> &input_or_output_dtype_vec,
       const vector<ge::DataType> &match_dtype_vec, const ForbiddenDtype &forbidden_dtype,
       vector<uint32_t> &matched_index_vec);
 
   void DumpDtypeMatchMapAndList(const std::map<ge::DataType, std::vector<ge::DataType>> &dtype_match_map,
       const vector<ge::DataType> &match_dtype_vec) const;
 
   virtual Status RunWithDtypeMap(const std::map<ge::DataType, std::vector<ge::DataType>> &dtype_match_map,
       const ForbiddenDtype &forbidden_dtype, FormatDtypeSelectionBasicInfo &basic_info);
 
  private:
   OpDtypePreciseMatcherPtr op_dtype_precise_matcher_ptr_;
 };
 }  // namespace fe
 #endif  // AIR_COMPILER_ENGINES_NNENG_OPTIMIZER_GRAPH_OPTIMIZER_OP_JUDGE_FORMAT_AND_DTYPE_STRATEGY_DTYPE_STRATEGY_OP_DTYPE_SELECTION_STRATEGY_TABLE_SELECT_BASE_H_