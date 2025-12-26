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

 #include "op_dtype_selection_strategy_cube_hif8.h"
 #include "common/configuration.h"
 #include "ops_store/ops_kernel_manager.h"
 
 namespace fe {
 OpDtypeSelectionStrategyCubeHif8::OpDtypeSelectionStrategyCubeHif8(
     FormatDtypeQuerierPtr format_dtype_querier_ptr,
     OpDtypePreciseMatcherPtr op_dtype_precise_matcher_ptr) :
     OpDtypeSelectionStrategyTableSelectBase(format_dtype_querier_ptr, op_dtype_precise_matcher_ptr) {}
 
 Status OpDtypeSelectionStrategyCubeHif8::Run(FormatDtypeSelectionBasicInfo& basic_info,
     ForbiddenDtype forbidden_dtype) {
   FE_CHECK_NOTNULL(basic_info.node);
   auto cur_op_desc_ptr = basic_info.node->GetOpDesc();
   FE_CHECK_NOTNULL(cur_op_desc_ptr);
   FE_LOGD("[GraphOpt][DtypeJdg][CubeHif8] Op[name=%s,type=%s]: Start match dtype for tensor[%u]",
           cur_op_desc_ptr->GetNamePtr(), cur_op_desc_ptr->GetTypePtr(), basic_info.index);
 
   const std::unordered_set<string> &fp16_op_type_list = Configuration::Instance(AI_CORE_NAME).GetFp16OpTypeList();
   bool is_cube_op = fp16_op_type_list.count(cur_op_desc_ptr->GetTypePtr()) > 0;
   if (is_cube_op) {
     RunWithDtypeMap(dtype_match_map_white_hif8, forbidden_dtype, basic_info);
   } else {
     RunWithDtypeMap(dtype_match_map_black_hif8, forbidden_dtype, basic_info);
   }
 
   FE_LOGD("[GraphOpt][DtypeJdg][CubeHif8] Op[name=%s,type=%s]: End match dtype for tensor[%u]",
           cur_op_desc_ptr->GetNamePtr(), cur_op_desc_ptr->GetTypePtr(), basic_info.index);
   return SUCCESS;
 }
 }  // namespace fe