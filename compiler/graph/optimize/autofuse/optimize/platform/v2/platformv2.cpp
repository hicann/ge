/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "platformv2.h"
#include "un_alignment_strategy.h"
#include "pass_runner_v2.h"
#include "template_generator_v2.h"
#include "partition/vector_func_partitioner.h"

namespace optimize {
constexpr size_t kMaxVecQueNum = 14UL;

PlatformV2::PlatformV2() {
  config_.max_que_num = kMaxVecQueNum;
  config_.is_support_compat_mode = true;
}

ge::Status PlatformV2::PartitionSubFunctions(ge::AscGraph &impl_graph) {
  // 待codegen上库之后使能
  if (getenv("AUTOFUSE_DISABLE_VF") == nullptr) {
    VectorFuncPartitioner partitioner(impl_graph);
    GE_ASSERT_SUCCESS(partitioner.Partition(), "Failed to partition sub funcs for graph [%s].",
                      impl_graph.GetName().c_str());
  }
  return ge::SUCCESS;
}

std::unique_ptr<BaseAlignmentStrategy> PlatformV2::GetAlignmentStrategy() {
  return ge::ComGraphMakeUnique<UnAlignmentStrategy>();
}

unique_ptr<BasePassRunner> PlatformV2::GetPassRunner() {
  return std::make_unique<PassRunnerV2>();
}

std::unique_ptr<BaseTemplateGenerator> PlatformV2::GetTemplateGenerator() {
  return ge::ComGraphMakeUnique<TemplateGeneratorV2>();
}

std::unique_ptr<BackendSpec> PlatformV2::GetBackendSpec() const {
  constexpr uint32_t kConcatMaxInputNum = 512;
  constexpr int32_t kConcatAlgGather = 1;
  constexpr uint32_t kMaxLoadNum = 15;
  constexpr uint32_t kMaxInputNum = 14U;
  auto ret = ge::ComGraphMakeUnique<BackendSpec>();
  ret->concat_max_input_num = kConcatMaxInputNum;
  ret->concat_alg = kConcatAlgGather;
  ret->gather_spec = {true, true, true};
  ret->slice_split_spec.split_lowered_to_split = true;
  ret->slice_split_spec.slice_fuse_with_end_dim_1 = true;
  ret->slice_split_spec.enable_split_flatten = true;
  ret->max_load_num = kMaxLoadNum;
  ret->max_input_nums_after_fuse = kMaxInputNum;
  ret->max_group_num_per_compile_unit = 10;
  ret->enable_matmul_lowering_to_matmul = true;
  GELOGD(
      "platform_v2, enable_non_tail_gather = %d, enable_reduce_gather_fusion = %d, "
      "enable_gather_concat_fusion = %d, max load_num = %u, max input num = %u",
      ret->gather_spec.enable_non_tail_gather, ret->gather_spec.enable_reduce_gather_fusion,
      ret->gather_spec.enable_gather_concat_fusion, ret->max_load_num, ret->max_input_nums_after_fuse);
  ret->transpose_mode = static_cast<uint32_t>(TransposeMode::TRANSPOSE_MODE_UNNORMAL);
  ret->set_local_memory_size = 256 * 1024 - 8 * 1024 - 32 * 1024;
  ret->pgo_spec = {false};
  return ret;
}

const PlatformConfig &PlatformV2::GetPlatformConfig() const {
  return config_;
}

#define REGISTER_PLATFORM_V2(platform_name, suffix) \
  static PlatformRegistrar<PlatformV2> registrar_##suffix(platform_name)

REGISTER_PLATFORM_V2("Ascend910_9591", 9591);
REGISTER_PLATFORM_V2("Ascend910_950z", 950z);
REGISTER_PLATFORM_V2("Ascend910_957b", 957b);
REGISTER_PLATFORM_V2("Ascend910_957d", 957d);
REGISTER_PLATFORM_V2("Ascend910_9589", 9589);
REGISTER_PLATFORM_V2("Ascend910_958a", 958a);
REGISTER_PLATFORM_V2("Ascend910_958b", 958b);
REGISTER_PLATFORM_V2("Ascend910_9599", 9599);
REGISTER_PLATFORM_V2("Ascend910_9581", 9581);
REGISTER_PLATFORM_V2("Ascend910_9579", 9579);
REGISTER_PLATFORM_V2("Ascend910_957c", 957c);
}  // namespace optimize