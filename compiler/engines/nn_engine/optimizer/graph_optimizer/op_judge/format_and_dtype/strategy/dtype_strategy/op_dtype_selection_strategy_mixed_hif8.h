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

 #ifndef AIR_COMPILER_ENGINES_NNENG_OPTIMIZER_GRAPH_OPTIMIZER_OP_JUDGE_FORMAT_AND_DTYPE_STRATEGY_DTYPE_STRATEGY_OP_DTYPE_SELECTION_STRATEGY_MIXED_HIF8_H_
 #define AIR_COMPILER_ENGINES_NNENG_OPTIMIZER_GRAPH_OPTIMIZER_OP_JUDGE_FORMAT_AND_DTYPE_STRATEGY_DTYPE_STRATEGY_OP_DTYPE_SELECTION_STRATEGY_MIXED_HIF8_H_
 
 #include "op_dtype_selection_strategy_table_select_base.h"
 
 namespace fe {
 class OpDtypeSelectionStrategyMixedHif8 : public OpDtypeSelectionStrategyTableSelectBase {
  public:
   OpDtypeSelectionStrategyMixedHif8(
       FormatDtypeQuerierPtr format_dtype_querier_ptr,
       OpDtypePreciseMatcherPtr op_dtype_precise_matcher_ptr);
 
   ~OpDtypeSelectionStrategyMixedHif8() override = default;
 
   Status Run(FormatDtypeSelectionBasicInfo& basic_info, ForbiddenDtype forbidden_dtype) override;
 
   Status RunWithDtypeMapForGrayList(const std::map<ge::DataType, std::vector<ge::DataType>> &dtype_match_map,
     const ForbiddenDtype &forbidden_dtype, const ge::InDataAnchorPtr &in_data_anchor,
     FormatDtypeSelectionBasicInfo &basic_info);
 
  private:
   OpDtypePreciseMatcherPtr op_dtype_precise_matcher_ptr_;
 
   Status GetFatherOutputDtype(const ge::InDataAnchorPtr &in_data_anchor, ge::DataType &father_output_dtype) const;
 };
 }  // namespace fe
 #endif  // AIR_COMPILER_ENGINES_NNENG_OPTIMIZER_GRAPH_OPTIMIZER_OP_JUDGE_FORMAT_AND_DTYPE_STRATEGY_DTYPE_STRATEGY_OP_DTYPE_SELECTION_STRATEGY_MIXED_HIF8_H_