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

 #ifndef AIR_COMPILER_ENGINES_NNENG_OPTIMIZER_GRAPH_OPTIMIZER_OP_JUDGE_FORMAT_AND_DTYPE_STRATEGY_DTYPE_STRATEGY_OP_DTYPE_SELECTION_STRATEGY_TABLE_SELECT_MAP_H_
 #define AIR_COMPILER_ENGINES_NNENG_OPTIMIZER_GRAPH_OPTIMIZER_OP_JUDGE_FORMAT_AND_DTYPE_STRATEGY_DTYPE_STRATEGY_OP_DTYPE_SELECTION_STRATEGY_TABLE_SELECT_MAP_H_

 namespace fe {
 /*
 origin_dtype       match_dtype_vec
 hif8               hif8,fp16,fp32
 bf16               hif8,bf16,fp32
 fp16               hif8,fp16,fp32
 fp32               hif8,fp16,fp32
 other              keep origin_dtype
 */
 const std::map<ge::DataType, std::vector<ge::DataType>> dtype_match_map_white_hif8 = {
   {ge::DT_HIFLOAT8, {ge::DT_HIFLOAT8, ge::DT_FLOAT16, ge::DT_FLOAT}},
   {ge::DT_BF16, {ge::DT_HIFLOAT8, ge::DT_BF16, ge::DT_FLOAT}},
   {ge::DT_FLOAT16, {ge::DT_HIFLOAT8, ge::DT_FLOAT16, ge::DT_FLOAT}},
   {ge::DT_FLOAT, {ge::DT_HIFLOAT8, ge::DT_FLOAT16, ge::DT_FLOAT}}
 };
 
 /*
 origin_dtype       match_dtype_vec
 bf16               fp16,fp32
 fp16               fp16,fp32
 fp32               fp32
 other              keep origin_dtype
 */
 const std::map<ge::DataType, std::vector<ge::DataType>> dtype_match_map_black_hif8 = {
   {ge::DT_BF16, {ge::DT_BF16, ge::DT_FLOAT}},
   {ge::DT_FLOAT16, {ge::DT_FLOAT16, ge::DT_FLOAT}},
   {ge::DT_FLOAT, {ge::DT_FLOAT}}
 };
 
 /*
 father_dtype        origin_dtype        match_dtype_vec
 hif8                other               hif8,fp16,fp32
                     bf16                hif8,bf16,fp32(need to replace fp16 with bf16, after get list)
 bf16                any                 bf16,fp32
 fp16                any                 fp16,fp32
 fp32                any                 fp32
 other               any                 keep father_dtype
 */
 const std::map<ge::DataType, std::vector<ge::DataType>> dtype_match_map_gray_hif8 = {
   {ge::DT_HIFLOAT8, {ge::DT_HIFLOAT8, ge::DT_FLOAT16, ge::DT_FLOAT}},
   {ge::DT_BF16, {ge::DT_BF16, ge::DT_FLOAT}},
   {ge::DT_FLOAT16, {ge::DT_FLOAT16, ge::DT_FLOAT}},
   {ge::DT_FLOAT, {ge::DT_FLOAT}}
 };
 }  // namespace fe
 #endif  // AIR_COMPILER_ENGINES_NNENG_OPTIMIZER_GRAPH_OPTIMIZER_OP_JUDGE_FORMAT_AND_DTYPE_STRATEGY_DTYPE_STRATEGY_OP_DTYPE_SELECTION_STRATEGY_TABLE_SELECT_MAP_H_
 