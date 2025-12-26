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
#include <fstream>

#include "codegen.h"
#include "e2e_constant_load_gt_store.h"
#include "e2e_common.h"

int main() {
  ge::AscGraph test_graph("constant_load_gt_store");
  std::string tilig_stub = R"(
#define REGISTER_TILING_DEFAULT(tiling)
#define GET_TILING_DATA(t, tiling)  AutofuseTilingData t = *(AutofuseTilingData*)tiling;
)";
  std::vector<ge::AscGraph> test_impl_graphs;
  ConstantLoadGtStore_AfterAutofuse(test_graph, test_impl_graphs);

  auto codegen = codegen::Codegen(codegen::CodegenOptions{
      .tiling_lib_path = "./libtest_constant_load_gt_store_codegen_tiling_gen.so", .tiling_lib_codegen_symbol = "CodegenTiling", .using_att_calc_qbt_size = false});

  std::fstream kernel_file("constant_load_gt_store_kernel.cpp", std::ios::out);
  std::fstream tiling_file("constant_load_gt_store_tiling.cpp", std::ios::out);
  std::fstream tiling_data_file("autofuse_tiling_data.h", std::ios::out);

  ascir::ScheduledResult schedule_result;
  std::vector<ascir::ScheduledResult> schedule_results{schedule_result};
  ascir::FusedScheduledResult fused_schedule_result;
  fused_schedule_result.fused_graph_name = ge::AscendString("constant_load_gt_store");
  fused_schedule_result.node_idx_to_scheduled_results.push_back(schedule_results);
  InitScheduleResultsByImplGraphs(test_impl_graphs, fused_schedule_result);
  codegen::CodegenResult result;
  codegen.Generate(fused_schedule_result, result);
  kernel_file << tilig_stub << result.kernel;
  tiling_file << result.tiling;
  tiling_data_file << result.tiling_data;
}
