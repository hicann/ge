/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include "faker/kernel_run_context_facker.h"
#include "register/kernel_registry.h"
#include "register/ffts_node_calculater_registry.h"
#include "common/plugin/ge_make_unique_util.h"
#include <kernel/memory/mem_block.h>
#include "engine/aicore/kernel/aicore_update_kernel.h"
#include "kernel/common_kernel_impl/sink_node_bin.h"
#include "stub/gert_runtime_stub.h"
#include "exe_graph/runtime/runtime_tensor.h"
#include "exe_graph/runtime/continuous_vector.h"
#include "engine/ffts_plus/kernel/ffts_update_kernel.h"
#include "engine/aicore/kernel/rt_ffts_plus_launch_args.h"
#include "engine/ffts_plus/converter/ffts_plus_proto_transfer.h"
#include "engine/ffts_plus/converter/ffts_plus_common.h"
namespace gert {
namespace kernel {}
using namespace kernel;

class FFTSUpdateKernelTestUT : public testing::Test {
 public:
  KernelRegistryImpl &registry = KernelRegistryImpl::GetInstance();
};

TEST_F(FFTSUpdateKernelTestUT, test_ExecuteOpFunc) {
  auto run_context = BuildKernelRunContext(0, 1);
  ASSERT_EQ(registry.FindKernelFuncs("ExecuteOpFunc")->outputs_creator(nullptr, run_context), ge::GRAPH_SUCCESS);
}

TEST_F(FFTSUpdateKernelTestUT, test_FFTSTaskAndArgsCopy_mem_guard_mismatch) {
  int64_t guard_actual_val = 100;
  int64_t guard_expected_val = 200;
  auto mem_guard = ContinuousVector::Create<MemGuard>(1);
  auto mem_guard_vec = reinterpret_cast<ContinuousVector *>(mem_guard.get());
  mem_guard_vec->SetSize(1);
  auto guard_data = reinterpret_cast<MemGuard *>(mem_guard_vec->MutableData());
  guard_data[0].guard_ptr = &guard_actual_val;
  guard_data[0].guard_val = guard_expected_val;

  auto run_context = BuildKernelRunContext(static_cast<size_t>(H2DInKey::RESERVED), 0);
  run_context.value_holder[static_cast<size_t>(H2DInKey::MEM_GUARD)].Set(mem_guard_vec, nullptr);

  ASSERT_NE(registry.FindKernelFuncs("FFTSTaskAndArgsCopy"), nullptr);
  auto msgs = registry.FindKernelFuncs("FFTSTaskAndArgsCopy")->trace_printer(run_context);
  ASSERT_EQ(msgs.size(), 1U);
  EXPECT_NE(msgs[0U].find("overwritten"), std::string::npos);
}
}  // namespace gert
