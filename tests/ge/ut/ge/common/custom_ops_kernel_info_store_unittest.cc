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
#include <atomic>
#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "engines/custom_engine/custom_graph_optimizer.h"
#include "engines/custom_engine/custom_ops_kernel_builder.h"
#include "engines/custom_engine/custom_ops_kernel_info_store.h"
#include "common/checker.h"
#include "exe_graph/runtime/annotated_args_context.h"
#include "graph/compute_graph.h"
#include "graph/custom_op_factory.h"
#include "graph/custom_op/args_refresh.h"
#include "graph/ascend_string.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/ge_tensor.h"
#include "graph/op_kernel_bin.h"
#include "graph/op_desc.h"
#include "graph/operator_reg.h"
#include "graph/custom_op.h"
#include "graph/ge_context.h"
#include "graph/ge_local_context.h"
#include "debug/ge_util.h"
#include "graph/utils/args_format_desc_utils.h"
#include "graph/args_format_desc.h"
#include "graph/utils/attr_utils.h"
#include "graph/utils/tensor_utils.h"
#include "runtime/custom_op/python_custom_op_adapter.h"
#include "securec.h"

namespace ge {
REG_OP(TestPythonAnnotatedArgsCustomOp_BuilderTest)
    .INPUT(x, TensorType::ALL())
    .INPUT(w, TensorType::ALL())
    .OUTPUT(y, TensorType::ALL())
    .OP_END_FACTORY_REG(TestPythonAnnotatedArgsCustomOp_BuilderTest);
}  // namespace ge

namespace ge {
namespace custom {

class MockCustomOp : public EagerExecuteOp {
 public:
  graphStatus Execute(gert::EagerOpExecutionContext *ctx) override {
    (void)ctx;
    return GRAPH_SUCCESS;
  }
};

class MockCompilableCustomOp : public EagerExecuteOp, public CompilableOp {
 public:
  graphStatus Execute(gert::EagerOpExecutionContext *ctx) override {
    (void)ctx;
    return GRAPH_SUCCESS;
  }

  graphStatus Compile(gert::OpCompileContext *ctx) override {
    (void)ctx;
    ++compile_count_;
    return GRAPH_SUCCESS;
  }

  static void ResetCompileCount() {
    compile_count_ = 0;
  }

  static int32_t GetCompileCount() {
    return compile_count_;
  }

 private:
  static int32_t compile_count_;
};

int32_t MockCompilableCustomOp::compile_count_ = 0;
class MockBaseOnlyCustomOp : public BaseCustomOp {};

class MockPortableCustomOp : public PortableOp {
 public:
  graphStatus Serialize(std::vector<uint8_t> &buffer) override {
    static const uint8_t kBin[] = {0x51U, 0x52U};
    buffer.assign(kBin, kBin + sizeof(kBin));
    return GRAPH_SUCCESS;
  }

  graphStatus Deserialize(const std::vector<uint8_t> &buffer) override {
    (void)buffer;
    return GRAPH_SUCCESS;
  }
};

std::atomic_uint32_t g_python_annotated_args_declare_count{0U};

struct MockPythonAnnotatedArgsHolder {};

void *CreateMockPythonAnnotatedArgsHolder(const custom_op::PythonCustomOpDescriptor *desc) {
  return (desc == nullptr) ? nullptr : new (std::nothrow) MockPythonAnnotatedArgsHolder();
}

void DestroyMockPythonAnnotatedArgsHolder(void *holder) {
  delete static_cast<MockPythonAnnotatedArgsHolder *>(holder);
}

graphStatus DeclareMockPythonAnnotatedArgs(const void *holder, gert::AnnotatedArgsContext *ctx) {
  if ((holder == nullptr) || (ctx == nullptr)) {
    return GRAPH_FAILED;
  }
  const auto *x = ctx->GetInputTensor(0U);
  const auto *w = ctx->GetInputTensor(1U);
  const auto *y = ctx->GetOutputTensor(0U);
  if ((x == nullptr) || (w == nullptr) || (y == nullptr)) {
    return GRAPH_FAILED;
  }
  ++g_python_annotated_args_declare_count;
  const auto workspace = ctx->MallocWorkSpace(64U);
  if (workspace.addr == nullptr) {
    return GRAPH_FAILED;
  }
  gert::AnnotatedKernelArgs args(gert::InputAddr{0U, x->GetAddr()}, gert::InputAddr{1U, w->GetAddr()},
                                 gert::OutputAddr{0U, y->GetAddr()}, workspace, uint64_t{7U});
  static const uint8_t kBin[] = {0x91U, 0x92U};
  return ctx->AddLaunch(
      gert::AnnotatedKernelLaunchInfo{"python_annotated_args_ut", kBin, sizeof(kBin), 1U, ctx->GetStreamId()},
      std::move(args));
}

std::atomic_bool g_compile_context_output_called{false};
constexpr uintptr_t kLogicDataMemBase = 0x80000000UL;
constexpr uintptr_t kLogicWeightMemBase = 0x90000000UL;

uint64_t ReadUint64Slot(const std::string &args, const size_t index) {
  uint64_t value = 0U;
  EXPECT_EQ(memcpy_s(&value, sizeof(value), args.data() + (index * sizeof(value)), sizeof(value)), EOK);
  return value;
}

uint16_t ReadUint16Slot(const std::string &args, const size_t index) {
  uint16_t value = 0U;
  EXPECT_EQ(memcpy_s(&value, sizeof(value), args.data() + (index * sizeof(value)), sizeof(value)), EOK);
  return value;
}

void ExpectPythonAnnotatedArgsKernel(const domi::KernelDef &kernel) {
  EXPECT_EQ(kernel.kernel_name(), "python_annotated_args_ut");
  EXPECT_EQ(kernel.args_size(), 40U);
  EXPECT_EQ(ReadUint64Slot(kernel.args(), 0U), static_cast<uint64_t>(kLogicDataMemBase + 1024U));
  EXPECT_EQ(ReadUint64Slot(kernel.args(), 1U), static_cast<uint64_t>(kLogicWeightMemBase + 4096U));
  EXPECT_EQ(ReadUint64Slot(kernel.args(), 2U), static_cast<uint64_t>(kLogicDataMemBase + 2048U));
  EXPECT_EQ(ReadUint64Slot(kernel.args(), 3U), static_cast<uint64_t>(kLogicDataMemBase + 4096U));
  EXPECT_EQ(ReadUint64Slot(kernel.args(), 4U), 7U);

  std::vector<ArgDesc> arg_descs;
  ASSERT_EQ(ArgsFormatDescUtils::Parse(kernel.context().args_format(), arg_descs), GRAPH_SUCCESS);
  ASSERT_EQ(arg_descs.size(), 5U);
  EXPECT_EQ(arg_descs[0].addr_type, AddrType::INPUT_INSTANCE);
  EXPECT_EQ(arg_descs[0].ir_idx, 0);
  EXPECT_EQ(arg_descs[1].addr_type, AddrType::INPUT_INSTANCE);
  EXPECT_EQ(arg_descs[1].ir_idx, 1);
  EXPECT_EQ(arg_descs[2].addr_type, AddrType::OUTPUT_INSTANCE);
  EXPECT_EQ(arg_descs[2].ir_idx, 0);
  EXPECT_EQ(arg_descs[3].addr_type, AddrType::WORKSPACE);
  EXPECT_EQ(arg_descs[3].ir_idx, 0);
  EXPECT_EQ(arg_descs[4].addr_type, AddrType::CUSTOM_VALUE);
}

class MockCompileContextOutputOp : public EagerExecuteOp, public CompilableOp {
 public:
  graphStatus Execute(gert::EagerOpExecutionContext *ctx) override {
    (void)ctx;
    return GRAPH_SUCCESS;
  }

  graphStatus Compile(gert::OpCompileContext *ctx) override {
    g_compile_context_output_called.store(true);
    if (ctx == nullptr) {
      return GRAPH_FAILED;
    }
    if (ctx->GetInputTensor(0U) != nullptr) {
      return GRAPH_FAILED;
    }

    const auto output0 = ctx->GetOutputTensor(0U);
    const auto output1 = ctx->GetOutputTensor(1U);
    const auto output2 = ctx->GetOutputTensor(2U);
    if ((output0 == nullptr) || (output1 == nullptr) || (output2 == nullptr)) {
      return GRAPH_FAILED;
    }
    if ((output0 != ctx->GetRequiredOutputTensor(0U)) || (output1 != ctx->GetDynamicOutputTensor(1U, 0U)) ||
        (output2 != ctx->GetDynamicOutputTensor(1U, 1U))) {
      return GRAPH_FAILED;
    }
    if ((ctx->GetOutputTensor(3U) != nullptr) || (ctx->GetRequiredOutputTensor(2U) != nullptr) ||
        (ctx->GetDynamicOutputTensor(1U, 2U) != nullptr)) {
      return GRAPH_FAILED;
    }
    if ((output0->GetStorageShape() != gert::Shape({8, 16})) || (output0->GetDataType() != DT_FLOAT16) ||
        (output0->GetStorageFormat() != FORMAT_ND)) {
      return GRAPH_FAILED;
    }
    if ((output1->GetStorageShape() != gert::Shape({16, 16})) || (output1->GetDataType() != DT_FLOAT)) {
      return GRAPH_FAILED;
    }
    if ((output2->GetStorageShape() != gert::Shape({32, 16})) || (output2->GetDataType() != DT_INT32)) {
      return GRAPH_FAILED;
    }
    return GRAPH_SUCCESS;
  }
};

class MockAnnotatedArgsCustomOp : public AnnotatedArgsOp, public MockPortableCustomOp {
 public:
  graphStatus DeclareLaunchArgs(gert::AnnotatedArgsContext &ctx) override {
    static const uint8_t kBin[] = {0x11U, 0x22U, 0x33U, 0x44U};
    auto input = ctx.GetInputTensor(0U);
    auto output = ctx.GetOutputTensor(0U);
    GE_ASSERT_NOTNULL(input);
    GE_ASSERT_NOTNULL(output);
    gert::AnnotatedKernelArgs args(gert::InputAddr{0U, input->GetAddr()}, gert::OutputAddr{0U, output->GetAddr()},
                                   uint64_t{7U});
    return ctx.AddLaunch(
        gert::AnnotatedKernelLaunchInfo{"custom_add_kernel", kBin, sizeof(kBin), 8U, ctx.GetStreamId()},
        std::move(args));
  }
};

class MockAnnotatedArgsWithConstInputCustomOp : public AnnotatedArgsOp, public MockPortableCustomOp {
 public:
  graphStatus DeclareLaunchArgs(gert::AnnotatedArgsContext &ctx) override {
    static const uint8_t kBin[] = {0x11U, 0x22U, 0x33U, 0x44U};
    auto input0 = ctx.GetInputTensor(0U);
    auto input1 = ctx.GetInputTensor(1U);
    auto output = ctx.GetOutputTensor(0U);
    GE_ASSERT_NOTNULL(input0);
    GE_ASSERT_NOTNULL(input1);
    GE_ASSERT_NOTNULL(output);
    gert::AnnotatedKernelArgs args(gert::InputAddr{0U, input0->GetAddr()}, gert::InputAddr{1U, input1->GetAddr()},
                                   gert::OutputAddr{0U, output->GetAddr()}, uint64_t{7U});
    return ctx.AddLaunch(
        gert::AnnotatedKernelLaunchInfo{"custom_add_kernel", kBin, sizeof(kBin), 8U, ctx.GetStreamId()},
        std::move(args));
  }
};

class MockAnnotatedArgsWithInvalidOptionalAndRequiredInputCustomOp : public AnnotatedArgsOp,
                                                                     public MockPortableCustomOp {
 public:
  graphStatus DeclareLaunchArgs(gert::AnnotatedArgsContext &ctx) override {
    static const uint8_t kBin[] = {0x15U, 0x16U, 0x17U, 0x18U};
    const auto input = ctx.GetRequiredInputTensor(0U);
    const auto optional_input = ctx.GetOptionalInputTensor(1U);
    const auto input_after_optional = ctx.GetRequiredInputTensor(2U);
    const auto output = ctx.GetOutputTensor(0U);
    if ((input == nullptr) || (optional_input != nullptr) || (input_after_optional == nullptr) || (output == nullptr)) {
      return GRAPH_FAILED;
    }
    if (input_after_optional->GetAddr() != reinterpret_cast<void *>(kLogicDataMemBase + 3072U)) {
      return GRAPH_FAILED;
    }
    gert::AnnotatedKernelArgs args(gert::InputAddr{0U, input->GetAddr()},
                                   gert::InputAddr{1U, input_after_optional->GetAddr()},
                                   gert::OutputAddr{0U, output->GetAddr()});
    return ctx.AddLaunch(gert::AnnotatedKernelLaunchInfo{"custom_invalid_optional_required_kernel", kBin, sizeof(kBin),
                                                         8U, ctx.GetStreamId()},
                         std::move(args));
  }
};

class MockAnnotatedArgsWithInvalidOptionalAndConstInputCustomOp : public AnnotatedArgsOp, public MockPortableCustomOp {
 public:
  graphStatus DeclareLaunchArgs(gert::AnnotatedArgsContext &ctx) override {
    static const uint8_t kBin[] = {0x19U, 0x1AU, 0x1BU, 0x1CU};
    const auto input = ctx.GetRequiredInputTensor(0U);
    const auto optional_input = ctx.GetOptionalInputTensor(1U);
    const auto const_input_after_optional = ctx.GetRequiredInputTensor(2U);
    const auto output = ctx.GetOutputTensor(0U);
    if ((input == nullptr) || (optional_input != nullptr) || (const_input_after_optional == nullptr) ||
        (output == nullptr)) {
      return GRAPH_FAILED;
    }
    if (const_input_after_optional->GetAddr() != reinterpret_cast<void *>(kLogicWeightMemBase + 4096U)) {
      return GRAPH_FAILED;
    }
    const uint32_t stream_id = ctx.GetStreamId();
    gert::AnnotatedKernelArgs args(gert::InputAddr{0U, input->GetAddr()},
                                   gert::InputAddr{1U, const_input_after_optional->GetAddr()},
                                   gert::OutputAddr{0U, output->GetAddr()});
    return ctx.AddLaunch(
        gert::AnnotatedKernelLaunchInfo{"custom_invalid_optional_const_kernel", kBin, sizeof(kBin), 8U, stream_id},
        std::move(args));
  }
};

class MockAnnotatedArgsWithPresentOptionalInputCustomOp : public AnnotatedArgsOp, public MockPortableCustomOp {
 public:
  graphStatus DeclareLaunchArgs(gert::AnnotatedArgsContext &ctx) override {
    static const uint8_t kBin[] = {0x31U};
    const auto *input0 = ctx.GetInputTensor(0U);
    const auto *input1 = ctx.GetInputTensor(1U);
    const auto *input2 = ctx.GetInputTensor(2U);
    const auto *output0 = ctx.GetOutputTensor(0U);
    if ((input0 == nullptr) || (input1 == nullptr) || (input2 == nullptr) || (output0 == nullptr) ||
        (ctx.GetOptionalInputTensor(1U) != input1) || (ctx.GetRequiredInputTensor(2U) != input2)) {
      return GRAPH_FAILED;
    }
    gert::AnnotatedKernelArgs args(gert::InputAddr{0U, input0->GetAddr()}, gert::InputAddr{1U, input1->GetAddr()},
                                   gert::InputAddr{2U, input2->GetAddr()}, gert::OutputAddr{0U, output0->GetAddr()});
    return ctx.AddLaunch(
        gert::AnnotatedKernelLaunchInfo{"present_optional_kernel", kBin, sizeof(kBin), 1U, ctx.GetStreamId()},
        std::move(args));
  }
};

class MockAnnotatedArgsWithExplicitOptionalZeroCustomOp : public AnnotatedArgsOp, public MockPortableCustomOp {
 public:
  graphStatus DeclareLaunchArgs(gert::AnnotatedArgsContext &ctx) override {
    static const uint8_t kBin[] = {0x32U};
    const auto *input = ctx.GetRequiredInputTensor(0U);
    const auto *optional_input = ctx.GetOptionalInputTensor(1U);
    const auto *input_after_optional = ctx.GetRequiredInputTensor(2U);
    const auto *output = ctx.GetOutputTensor(0U);
    if ((input == nullptr) || (optional_input != nullptr) || (input_after_optional == nullptr) || (output == nullptr)) {
      return GRAPH_FAILED;
    }
    gert::AnnotatedKernelArgs args(gert::InputAddr{0U, input->GetAddr()}, uint64_t{0U},
                                   gert::InputAddr{1U, input_after_optional->GetAddr()},
                                   gert::OutputAddr{0U, output->GetAddr()});
    return ctx.AddLaunch(
        gert::AnnotatedKernelLaunchInfo{"explicit_optional_zero_kernel", kBin, sizeof(kBin), 1U, ctx.GetStreamId()},
        std::move(args));
  }
};

class MockAnnotatedArgsWithDynamicIoCustomOp : public AnnotatedArgsOp, public MockPortableCustomOp {
 public:
  graphStatus DeclareLaunchArgs(gert::AnnotatedArgsContext &ctx) override {
    static const uint8_t kBin[] = {0x41U};
    const auto *input3 = ctx.GetInputTensor(3U);    // z: IR2, flat3
    const auto *input1 = ctx.GetInputTensor(1U);    // dx0: IR1/relative0, flat1
    const auto *output2 = ctx.GetOutputTensor(2U);  // dy1: IR1/relative1, flat2
    const auto *output0 = ctx.GetOutputTensor(0U);  // y: IR0, flat0
    if ((input3 == nullptr) || (input1 == nullptr) || (output2 == nullptr) || (output0 == nullptr) ||
        (ctx.GetRequiredInputTensor(2U) != input3) || (ctx.GetDynamicInputTensor(1U, 0U) != input1) ||
        (ctx.GetDynamicOutputTensor(1U, 1U) != output2)) {
      return GRAPH_FAILED;
    }
    gert::AnnotatedKernelArgs args(gert::InputAddr{3U, input3->GetAddr()}, gert::InputAddr{1U, input1->GetAddr()},
                                   gert::InputAddr{1U, input1->GetAddr()}, gert::OutputAddr{2U, output2->GetAddr()},
                                   gert::OutputAddr{0U, output0->GetAddr()});
    return ctx.AddLaunch(
        gert::AnnotatedKernelLaunchInfo{"dynamic_instance_kernel", kBin, sizeof(kBin), 1U, ctx.GetStreamId()},
        std::move(args));
  }
};

class MockAnnotatedArgsWithSlotCountCustomOp : public AnnotatedArgsOp, public MockPortableCustomOp {
 public:
  explicit MockAnnotatedArgsWithSlotCountCustomOp(const size_t slot_count) : slot_count_(slot_count) {}

  graphStatus DeclareLaunchArgs(gert::AnnotatedArgsContext &ctx) override {
    static const uint8_t kBin[] = {0x1U};
    const auto *input = ctx.GetInputTensor(0U);
    const auto *output = ctx.GetOutputTensor(0U);
    if ((slot_count_ < 2U) || (input == nullptr) || (output == nullptr)) {
      return GRAPH_FAILED;
    }
    gert::AnnotatedKernelArgs args;
    GE_ASSERT_SUCCESS(args.AppendArg(gert::InputAddr{0U, input->GetAddr()}));
    GE_ASSERT_SUCCESS(args.AppendArg(gert::OutputAddr{0U, output->GetAddr()}));
    for (size_t i = 2U; i < slot_count_; ++i) {
      GE_ASSERT_SUCCESS(args.AppendArg(static_cast<uint64_t>(i)));
    }
    return ctx.AddLaunch(gert::AnnotatedKernelLaunchInfo{"many_args_kernel", kBin, sizeof(kBin), 1U, ctx.GetStreamId()},
                         std::move(args));
  }

 private:
  size_t slot_count_;
};

class MockAnnotatedArgsWithWorkspaceCustomOp : public AnnotatedArgsOp, public MockPortableCustomOp {
 public:
  graphStatus DeclareLaunchArgs(gert::AnnotatedArgsContext &ctx) override {
    static const uint8_t kBin[] = {0x21U, 0x22U, 0x23U, 0x24U};
    auto input = ctx.GetInputTensor(0U);
    auto output = ctx.GetOutputTensor(0U);
    GE_ASSERT_NOTNULL(input);
    GE_ASSERT_NOTNULL(output);
    auto workspace = ctx.MallocWorkSpace(100U);
    GE_ASSERT_NOTNULL(workspace.addr);
    gert::AnnotatedKernelArgs args(gert::InputAddr{0U, input->GetAddr()}, gert::OutputAddr{0U, output->GetAddr()},
                                   workspace);
    return ctx.AddLaunch(
        gert::AnnotatedKernelLaunchInfo{"custom_workspace_kernel", kBin, sizeof(kBin), 4U, ctx.GetStreamId()},
        std::move(args));
  }
};

std::atomic_uint32_t g_single_declare_no_workspace_count{0U};

class MockSingleDeclareNoWorkspaceCustomOp : public AnnotatedArgsOp, public MockPortableCustomOp {
 public:
  graphStatus DeclareLaunchArgs(gert::AnnotatedArgsContext &ctx) override {
    ++g_single_declare_no_workspace_count;
    static const uint8_t kBin[] = {0x21U, 0x22U};
    const auto *input = ctx.GetInputTensor(0U);
    const auto *output = ctx.GetOutputTensor(0U);
    GE_ASSERT_NOTNULL(input);
    GE_ASSERT_NOTNULL(output);
    gert::AnnotatedKernelArgs args(gert::InputAddr{0U, input->GetAddr()}, gert::InputAddr{0U, input->GetAddr()},
                                   gert::OutputAddr{0U, output->GetAddr()});
    return ctx.AddLaunch(
        gert::AnnotatedKernelLaunchInfo{"single_declare_no_workspace", kBin, sizeof(kBin), 4U, ctx.GetStreamId()},
        std::move(args));
  }
};

std::atomic_uint32_t g_single_declare_workspace_count{0U};

class MockSingleDeclareWorkspaceCustomOp : public AnnotatedArgsOp, public MockPortableCustomOp {
 public:
  graphStatus DeclareLaunchArgs(gert::AnnotatedArgsContext &ctx) override {
    ++g_single_declare_workspace_count;
    static const uint8_t kBin[] = {0x31U, 0x32U};
    const auto *input0 = ctx.GetInputTensor(0U);
    const auto *input1 = ctx.GetInputTensor(1U);
    const auto *output = ctx.GetOutputTensor(0U);
    GE_ASSERT_NOTNULL(input0);
    GE_ASSERT_NOTNULL(input1);
    GE_ASSERT_NOTNULL(output);
    const auto workspace = ctx.MallocWorkSpace(100U);
    GE_ASSERT_NOTNULL(workspace.addr);
    gert::AnnotatedKernelArgs args(gert::InputAddr{0U, input0->GetAddr()}, gert::InputAddr{1U, input1->GetAddr()},
                                   gert::OutputAddr{0U, output->GetAddr()}, workspace);
    return ctx.AddLaunch(
        gert::AnnotatedKernelLaunchInfo{"single_declare_workspace", kBin, sizeof(kBin), 4U, ctx.GetStreamId()},
        std::move(args));
  }
};

class MockAnnotatedArgsWithMultipleWorkspacesCustomOp : public AnnotatedArgsOp, public MockPortableCustomOp {
 public:
  graphStatus DeclareLaunchArgs(gert::AnnotatedArgsContext &ctx) override {
    static const uint8_t kBin[] = {0x25U, 0x26U, 0x27U, 0x28U};
    auto input = ctx.GetInputTensor(0U);
    auto output = ctx.GetOutputTensor(0U);
    GE_ASSERT_NOTNULL(input);
    GE_ASSERT_NOTNULL(output);
    auto workspace0 = ctx.MallocWorkSpace(100U);
    auto workspace1 = ctx.MallocWorkSpace(600U);
    GE_ASSERT_NOTNULL(workspace0.addr);
    GE_ASSERT_NOTNULL(workspace1.addr);
    gert::AnnotatedKernelArgs args(gert::InputAddr{0U, input->GetAddr()}, gert::OutputAddr{0U, output->GetAddr()},
                                   workspace0, workspace1);
    return ctx.AddLaunch(
        gert::AnnotatedKernelLaunchInfo{"custom_multi_workspace_kernel", kBin, sizeof(kBin), 4U, ctx.GetStreamId()},
        std::move(args));
  }
};

class MockAnnotatedArgsWithMultipleLaunchesCustomOp : public AnnotatedArgsOp, public MockPortableCustomOp {
 public:
  graphStatus DeclareLaunchArgs(gert::AnnotatedArgsContext &ctx) override {
    static const uint8_t kBin0[] = {0x61U, 0x62U};
    static const uint8_t kBin1[] = {0x71U, 0x72U};
    auto input = ctx.GetInputTensor(0U);
    auto output = ctx.GetOutputTensor(0U);
    GE_ASSERT_NOTNULL(input);
    GE_ASSERT_NOTNULL(output);
    gert::AnnotatedKernelArgs args0(gert::InputAddr{0U, input->GetAddr()});
    GE_ASSERT_SUCCESS(ctx.AddLaunch(
        gert::AnnotatedKernelLaunchInfo{"custom_multi_launch_kernel0", kBin0, sizeof(kBin0), 4U, ctx.GetStreamId()},
        std::move(args0)));
    gert::AnnotatedKernelArgs args1(gert::OutputAddr{0U, output->GetAddr()});
    return ctx.AddLaunch(
        gert::AnnotatedKernelLaunchInfo{"custom_multi_launch_kernel1", kBin1, sizeof(kBin1), 4U, ctx.GetStreamId()},
        std::move(args1));
  }
};

class MockAnnotatedArgsWithSameNameDifferentBinsCustomOp : public AnnotatedArgsOp, public MockPortableCustomOp {
 public:
  graphStatus DeclareLaunchArgs(gert::AnnotatedArgsContext &ctx) override {
    static const uint8_t kBin0[] = {0x61U, 0x62U};
    static const uint8_t kBin1[] = {0x71U, 0x72U};
    auto input = ctx.GetInputTensor(0U);
    auto output = ctx.GetOutputTensor(0U);
    GE_ASSERT_NOTNULL(input);
    GE_ASSERT_NOTNULL(output);
    gert::AnnotatedKernelArgs args0(gert::InputAddr{0U, input->GetAddr()});
    GE_ASSERT_SUCCESS(ctx.AddLaunch(
        gert::AnnotatedKernelLaunchInfo{"custom_shared_kernel", kBin0, sizeof(kBin0), 4U, ctx.GetStreamId()},
        std::move(args0)));
    gert::AnnotatedKernelArgs args1(gert::OutputAddr{0U, output->GetAddr()});
    return ctx.AddLaunch(
        gert::AnnotatedKernelLaunchInfo{"custom_shared_kernel", kBin1, sizeof(kBin1), 4U, ctx.GetStreamId()},
        std::move(args1));
  }
};

std::atomic_uint32_t g_both_refresh_interfaces_declare_count{0U};

class MockBothRefreshInterfacesCustomOp : public AnnotatedArgsOp, public ArgsUpdater, public EagerExecuteOp {
 public:
  graphStatus DeclareLaunchArgs(gert::AnnotatedArgsContext &ctx) override {
    (void)ctx;
    ++g_both_refresh_interfaces_declare_count;
    return GRAPH_FAILED;
  }

  graphStatus UpdateHostArgs(gert::UpdateArgsContext *ctx) override {
    (void)ctx;
    return GRAPH_SUCCESS;
  }

  graphStatus Execute(gert::EagerOpExecutionContext *ctx) override {
    (void)ctx;
    return GRAPH_SUCCESS;
  }
};

std::atomic_uint32_t g_annotated_eager_declare_count{0U};

class MockAnnotatedArgsAndEagerCustomOp : public AnnotatedArgsOp, public EagerExecuteOp {
 public:
  graphStatus DeclareLaunchArgs(gert::AnnotatedArgsContext &ctx) override {
    (void)ctx;
    ++g_annotated_eager_declare_count;
    return GRAPH_FAILED;
  }

  graphStatus Execute(gert::EagerOpExecutionContext *ctx) override {
    (void)ctx;
    return GRAPH_SUCCESS;
  }
};

class MockAnnotatedArgsWithMismatchStreamCustomOp : public AnnotatedArgsOp, public MockPortableCustomOp {
 public:
  graphStatus DeclareLaunchArgs(gert::AnnotatedArgsContext &ctx) override {
    static const uint8_t kBin[] = {0x91U, 0x92U};
    auto input = ctx.GetInputTensor(0U);
    GE_ASSERT_NOTNULL(input);
    gert::AnnotatedKernelArgs args(gert::InputAddr{0U, input->GetAddr()});
    return ctx.AddLaunch(gert::AnnotatedKernelLaunchInfo{"custom_mismatch_stream_kernel", kBin, sizeof(kBin), 4U,
                                                         ctx.GetStreamId() + 1U},
                         std::move(args));
  }
};

class MockAnnotatedArgsWithConstInputAndWorkspaceCustomOp : public AnnotatedArgsOp, public MockPortableCustomOp {
 public:
  graphStatus DeclareLaunchArgs(gert::AnnotatedArgsContext &ctx) override {
    static const uint8_t kBin[] = {0x41U, 0x42U, 0x43U, 0x44U};
    auto input0 = ctx.GetInputTensor(0U);
    auto input1 = ctx.GetInputTensor(1U);
    auto output = ctx.GetOutputTensor(0U);
    GE_ASSERT_NOTNULL(input0);
    GE_ASSERT_NOTNULL(input1);
    GE_ASSERT_NOTNULL(output);
    auto workspace = ctx.MallocWorkSpace(100U);
    GE_ASSERT_NOTNULL(workspace.addr);
    gert::AnnotatedKernelArgs args(gert::InputAddr{0U, input0->GetAddr()}, gert::InputAddr{1U, input1->GetAddr()},
                                   gert::OutputAddr{0U, output->GetAddr()}, workspace);
    return ctx.AddLaunch(
        gert::AnnotatedKernelLaunchInfo{"custom_const_workspace_kernel", kBin, sizeof(kBin), 4U, ctx.GetStreamId()},
        std::move(args));
  }
};

enum class InvalidAnnotatedArgsCase {
  kEmptyName,
  kEmptyBin,
  kZeroBlockDim,
  kEmptyArgs,
};

class MockAnnotatedArgsWithoutPortableCustomOp : public AnnotatedArgsOp {
 public:
  graphStatus DeclareLaunchArgs(gert::AnnotatedArgsContext &ctx) override {
    static const uint8_t kBin[] = {0xA1U, 0xA2U, 0xA3U, 0xA4U};
    auto input = ctx.GetInputTensor(0U);
    auto output = ctx.GetOutputTensor(0U);
    GE_ASSERT_NOTNULL(input);
    GE_ASSERT_NOTNULL(output);
    gert::AnnotatedKernelArgs args(gert::InputAddr{0U, input->GetAddr()}, gert::OutputAddr{0U, output->GetAddr()},
                                   uint64_t{9U});
    return ctx.AddLaunch(
        gert::AnnotatedKernelLaunchInfo{"custom_without_portable_kernel", kBin, sizeof(kBin), 2U, ctx.GetStreamId()},
        std::move(args));
  }
};

class MockAnnotatedArgsWithoutAddLaunchCustomOp : public AnnotatedArgsOp, public MockPortableCustomOp {
 public:
  graphStatus DeclareLaunchArgs(gert::AnnotatedArgsContext &ctx) override {
    (void)ctx;
    return GRAPH_SUCCESS;
  }
};

class MockAnnotatedArgsReturnsInternalErrorCustomOp : public AnnotatedArgsOp, public MockPortableCustomOp {
 public:
  graphStatus DeclareLaunchArgs(gert::AnnotatedArgsContext &ctx) override {
    (void)ctx;
    return INTERNAL_ERROR;
  }
};

class MockInvalidAnnotatedArgsCustomOp : public AnnotatedArgsOp, public MockPortableCustomOp {
 public:
  explicit MockInvalidAnnotatedArgsCustomOp(const InvalidAnnotatedArgsCase invalid_case)
      : invalid_case_(invalid_case) {}

  graphStatus DeclareLaunchArgs(gert::AnnotatedArgsContext &ctx) override {
    static const uint8_t kBin[] = {0x11U, 0x22U, 0x33U, 0x44U};
    auto input = ctx.GetInputTensor(0U);
    GE_ASSERT_NOTNULL(input);
    gert::AnnotatedKernelArgs valid_args(gert::InputAddr{0U, input->GetAddr()});
    gert::AnnotatedKernelArgs empty_args;
    switch (invalid_case_) {
      case InvalidAnnotatedArgsCase::kEmptyName:
        return ctx.AddLaunch(gert::AnnotatedKernelLaunchInfo{"", kBin, sizeof(kBin), 8U, ctx.GetStreamId()},
                             std::move(valid_args));
      case InvalidAnnotatedArgsCase::kEmptyBin:
        return ctx.AddLaunch(gert::AnnotatedKernelLaunchInfo{"custom_add_kernel", kBin, 0U, 8U, ctx.GetStreamId()},
                             std::move(valid_args));
      case InvalidAnnotatedArgsCase::kZeroBlockDim:
        return ctx.AddLaunch(
            gert::AnnotatedKernelLaunchInfo{"custom_add_kernel", kBin, sizeof(kBin), 0U, ctx.GetStreamId()},
            std::move(valid_args));
      case InvalidAnnotatedArgsCase::kEmptyArgs:
        return ctx.AddLaunch(
            gert::AnnotatedKernelLaunchInfo{"custom_add_kernel", kBin, sizeof(kBin), 8U, ctx.GetStreamId()},
            std::move(empty_args));
      default:
        return GRAPH_FAILED;
    }
  }

 private:
  InvalidAnnotatedArgsCase invalid_case_;
};

class UtestCustomOpsKernelInfoStore : public testing::Test {
 protected:
  void SetUp() override {
    graph_options_ = GetThreadLocalContext().GetAllGraphOptions();
    session_options_ = GetThreadLocalContext().GetAllSessionOptions();
    global_options_ = GetThreadLocalContext().GetAllGlobalOptions();
    GetThreadLocalContext().SetGraphOption({{ge::SOC_VERSION, "KirinX90"}});
  }

  void TearDown() override {
    GetThreadLocalContext().SetGraphOption(graph_options_);
    GetThreadLocalContext().SetSessionOption(session_options_);
    GetThreadLocalContext().SetGlobalOption(global_options_);
  }

 private:
  std::map<std::string, std::string> graph_options_;
  std::map<std::string, std::string> session_options_;
  std::map<std::string, std::string> global_options_;
};

NodePtr BuildStaticCustomNode(const std::string &op_type, ComputeGraphPtr &graph) {
  graph = std::make_shared<ComputeGraph>("custom_builder_graph");
  auto op_desc = std::make_shared<OpDesc>("custom_builder_node", op_type);
  op_desc->SetId(7);
  op_desc->SetStreamId(3);
  op_desc->AppendIrInput("x", kIrInputRequired);
  op_desc->AppendIrOutput("y", kIrOutputRequired);

  GeTensorDesc input_desc(GeShape({1, 16}), FORMAT_ND, DT_FLOAT16);
  input_desc.SetOriginShape(GeShape({1, 16}));
  GeTensorDesc output_desc(GeShape({1, 16}), FORMAT_ND, DT_FLOAT16);
  output_desc.SetOriginShape(GeShape({1, 16}));
  (void)op_desc->AddInputDesc("x", input_desc);
  (void)op_desc->AddOutputDesc("y", output_desc);
  op_desc->SetInputOffset({1024});
  op_desc->SetOutputOffset({2048});
  auto node = graph->AddNode(op_desc);
  if (node != nullptr) {
    node->GetOpDesc()->SetId(7);
  }
  return node;
}

NodePtr BuildStaticCustomNodeWithConstInput(const std::string &op_type, ComputeGraphPtr &graph) {
  auto node = BuildStaticCustomNode(op_type, graph);
  if (node == nullptr) {
    return nullptr;
  }
  auto op_desc = node->GetOpDesc();
  if (op_desc == nullptr) {
    return nullptr;
  }

  op_desc->AppendIrInput("w", kIrInputRequired);
  GeTensorDesc weight_desc(GeShape({1, 16}), FORMAT_ND, DT_FLOAT16);
  weight_desc.SetOriginShape(GeShape({1, 16}));
  TensorUtils::SetDataOffset(weight_desc, 4096);
  (void)op_desc->AddInputDesc("w", weight_desc);
  op_desc->SetInputOffset({1024, 8192});
  op_desc->SetIsInputConst({false, true});
  return node;
}

NodePtr BuildStaticCustomNodeWithInvalidOptionalInput(const std::string &op_type, const bool last_input_is_const,
                                                      ComputeGraphPtr &graph) {
  graph = std::make_shared<ComputeGraph>("custom_builder_invalid_optional_graph");
  auto op_desc = std::make_shared<OpDesc>("custom_builder_invalid_optional_node", op_type);
  op_desc->SetId(7);
  op_desc->SetStreamId(3);
  op_desc->AppendIrInput("x", kIrInputRequired);
  op_desc->AppendIrInput("optional", kIrInputOptional);
  const std::string last_input_name = last_input_is_const ? "w" : "z";
  op_desc->AppendIrInput(last_input_name, kIrInputRequired);
  op_desc->AppendIrOutput("y", kIrOutputRequired);

  GeTensorDesc input_desc(GeShape({1, 16}), FORMAT_ND, DT_FLOAT16);
  input_desc.SetOriginShape(GeShape({1, 16}));
  GeTensorDesc invalid_optional_desc(GeShape(), FORMAT_RESERVED, DT_UNDEFINED);
  GeTensorDesc last_input_desc(GeShape({1, 16}), FORMAT_ND, DT_FLOAT16);
  last_input_desc.SetOriginShape(GeShape({1, 16}));
  if (last_input_is_const) {
    TensorUtils::SetDataOffset(last_input_desc, 4096);
  }
  GeTensorDesc output_desc(GeShape({1, 16}), FORMAT_ND, DT_FLOAT16);
  output_desc.SetOriginShape(GeShape({1, 16}));

  (void)op_desc->AddInputDesc("x", input_desc);
  (void)op_desc->AddOptionalInputDesc("optional", invalid_optional_desc);
  (void)op_desc->AddInputDesc(last_input_name, last_input_desc);
  (void)op_desc->AddOutputDesc("y", output_desc);
  op_desc->SetInputOffset(last_input_is_const ? std::vector<int64_t>{1024, 8192} : std::vector<int64_t>{1024, 3072});
  op_desc->SetOutputOffset({2048});
  op_desc->SetIsInputConst({false, false, last_input_is_const});

  auto node = graph->AddNode(op_desc);
  if (node != nullptr) {
    node->GetOpDesc()->SetId(7);
  }
  return node;
}

NodePtr BuildStaticCustomNodeWithPresentOptionalInput(const std::string &op_type, ComputeGraphPtr &graph) {
  graph = std::make_shared<ComputeGraph>("custom_builder_present_optional_graph");
  auto op_desc = std::make_shared<OpDesc>("custom_builder_present_optional_node", op_type);
  op_desc->SetId(7);
  op_desc->SetStreamId(3);
  op_desc->AppendIrInput("x", kIrInputRequired);
  op_desc->AppendIrInput("optional", kIrInputOptional);
  op_desc->AppendIrInput("z", kIrInputRequired);
  op_desc->AppendIrOutput("y", kIrOutputRequired);

  GeTensorDesc input_desc(GeShape({1, 16}), FORMAT_ND, DT_FLOAT16);
  input_desc.SetOriginShape(GeShape({1, 16}));
  GeTensorDesc output_desc(GeShape({1, 16}), FORMAT_ND, DT_FLOAT16);
  output_desc.SetOriginShape(GeShape({1, 16}));
  if ((op_desc->AddInputDesc("x", input_desc) != GRAPH_SUCCESS) ||
      (op_desc->AddOptionalInputDesc("optional", input_desc) != GRAPH_SUCCESS) ||
      (op_desc->AddInputDesc("z", input_desc) != GRAPH_SUCCESS) ||
      (op_desc->AddOutputDesc("y", output_desc) != GRAPH_SUCCESS)) {
    return nullptr;
  }
  op_desc->SetInputOffset({1024, 2048, 3072});
  op_desc->SetOutputOffset({1280});

  auto node = graph->AddNode(op_desc);
  if (node != nullptr) {
    node->GetOpDesc()->SetId(7);
  }
  return node;
}

NodePtr BuildStaticCustomNodeWithDynamicIo(const std::string &op_type, ComputeGraphPtr &graph) {
  graph = std::make_shared<ComputeGraph>("custom_builder_dynamic_io_graph");
  auto op_desc = std::make_shared<OpDesc>("custom_builder_dynamic_io_node", op_type);
  op_desc->SetId(7);
  op_desc->SetStreamId(3);

  GeTensorDesc tensor_desc(GeShape({1, 16}), FORMAT_ND, DT_FLOAT16);
  tensor_desc.SetOriginShape(GeShape({1, 16}));
  if ((op_desc->AddInputDesc("x", tensor_desc) != GRAPH_SUCCESS) ||
      (op_desc->AddDynamicInputDesc("dx", 2U, true) != GRAPH_SUCCESS) ||
      (op_desc->AddInputDesc("z", tensor_desc) != GRAPH_SUCCESS) ||
      (op_desc->AddOutputDesc("y", tensor_desc) != GRAPH_SUCCESS) ||
      (op_desc->AddDynamicOutputDesc("dy", 2U, true) != GRAPH_SUCCESS)) {
    return nullptr;
  }
  op_desc->AppendIrInput("x", kIrInputRequired);
  op_desc->AppendIrInput("dx", kIrInputDynamic);
  op_desc->AppendIrInput("z", kIrInputRequired);
  op_desc->AppendIrOutput("y", kIrOutputRequired);
  op_desc->AppendIrOutput("dy", kIrOutputDynamic);
  for (uint32_t i = 0U; i < 4U; ++i) {
    if (op_desc->UpdateInputDesc(i, tensor_desc) != GRAPH_SUCCESS) {
      return nullptr;
    }
  }
  for (uint32_t i = 0U; i < 3U; ++i) {
    if (op_desc->UpdateOutputDesc(i, tensor_desc) != GRAPH_SUCCESS) {
      return nullptr;
    }
  }
  op_desc->SetInputOffset({256, 512, 768, 1024});
  op_desc->SetOutputOffset({1280, 1536, 1792});

  auto node = graph->AddNode(op_desc);
  if (node != nullptr) {
    node->GetOpDesc()->SetId(7);
  }
  return node;
}

Status GenerateTaskForNode(const NodePtr &node, std::vector<domi::TaskDef> &tasks) {
  CustomOpsKernelBuilder builder;
  RunContext context = {};
  context.dataMemBase = reinterpret_cast<uint8_t *>(kLogicDataMemBase);
  context.dataMemSize = 4096U;
  context.weightMemBase = reinterpret_cast<uint8_t *>(kLogicWeightMemBase);
  context.weightMemSize = 4096U;
  return builder.GenerateTask(*node, context, tasks);
}

void FinalizeCustomWorkspaceForDirectBuilderTest(const NodePtr &node, const int64_t workspace_offset) {
  ASSERT_NE(node, nullptr);
  auto op_desc = node->GetOpDesc();
  ASSERT_NE(op_desc, nullptr);
  std::vector<int64_t> append_ws;
  ASSERT_TRUE(AttrUtils::GetListInt(op_desc, "_append_ws", append_ws));
  ASSERT_EQ(append_ws.size(), 1U);
  op_desc->SetWorkspace({workspace_offset});
  op_desc->SetWorkspaceBytes(append_ws);
}

void ExpectGenerateTaskFailed(const std::string &op_type) {
  ComputeGraphPtr graph;
  auto node = BuildStaticCustomNode(op_type, graph);
  ASSERT_NE(node, nullptr);

  std::vector<domi::TaskDef> tasks;
  EXPECT_NE(GenerateTaskForNode(node, tasks), SUCCESS);
  EXPECT_TRUE(tasks.empty());
}

void ExpectGenerateTaskFailedWithStatus(const std::string &op_type, const Status expected_status) {
  ComputeGraphPtr graph;
  auto node = BuildStaticCustomNode(op_type, graph);
  ASSERT_NE(node, nullptr);

  std::vector<domi::TaskDef> tasks;
  EXPECT_EQ(GenerateTaskForNode(node, tasks), expected_status);
  EXPECT_TRUE(tasks.empty());
}

TEST_F(UtestCustomOpsKernelInfoStore, InitializeSuccess) {
  CustomOpsKernelInfoStore store;
  std::map<std::string, std::string> options;
  EXPECT_EQ(store.Initialize(options), SUCCESS);
}

TEST_F(UtestCustomOpsKernelInfoStore, FinalizeSuccess) {
  CustomOpsKernelInfoStore store;
  std::map<std::string, std::string> options;
  EXPECT_EQ(store.Initialize(options), SUCCESS);
  EXPECT_EQ(store.Finalize(), SUCCESS);
}

TEST_F(UtestCustomOpsKernelInfoStore, RefreshSuccess) {
  CustomOpsKernelInfoStore store;
  std::map<std::string, std::string> options;
  EXPECT_EQ(store.Initialize(options), SUCCESS);
  EXPECT_EQ(store.Refresh(), SUCCESS);
}

TEST_F(UtestCustomOpsKernelInfoStore, GetAllOpsKernelInfo) {
  CustomOpsKernelInfoStore store;
  std::map<std::string, std::string> options;
  EXPECT_EQ(store.Initialize(options), SUCCESS);

  std::map<std::string, OpInfo> infos;
  store.GetAllOpsKernelInfo(infos);

  std::vector<AscendString> registered_ops;
  CustomOpFactory::GetAllRegisteredOps(registered_ops);
  EXPECT_EQ(infos.size(), registered_ops.size());
}

TEST_F(UtestCustomOpsKernelInfoStore, CheckSupported) {
  CustomOpsKernelInfoStore store;
  std::map<std::string, std::string> options;
  EXPECT_EQ(store.Initialize(options), SUCCESS);

  std::vector<AscendString> registered_ops;
  CustomOpFactory::GetAllRegisteredOps(registered_ops);

  if (!registered_ops.empty()) {
    auto op_desc = std::make_shared<OpDesc>(std::string(registered_ops[0].GetString()),
                                            std::string(registered_ops[0].GetString()));
    std::string reason;
    EXPECT_TRUE(store.CheckSupported(op_desc, reason));
  }
}

TEST_F(UtestCustomOpsKernelInfoStore, RefreshCapturesNewRegisteredOp) {
  const std::string kTestOpType = "TestDynamicRegisteredOp_RefreshTest";

  CustomOpsKernelInfoStore store;
  std::map<std::string, std::string> options;
  EXPECT_EQ(store.Initialize(options), SUCCESS);

  std::map<std::string, OpInfo> infos_before;
  store.GetAllOpsKernelInfo(infos_before);
  size_t count_before = infos_before.size();
  EXPECT_EQ(infos_before.count(kTestOpType), 0U);

  auto creator = []() -> std::unique_ptr<BaseCustomOp> { return std::make_unique<MockCustomOp>(); };
  EXPECT_EQ(CustomOpFactory::RegisterCustomOpCreator(AscendString(kTestOpType.c_str()), creator), GRAPH_SUCCESS);

  std::map<std::string, OpInfo> infos_no_refresh;
  store.GetAllOpsKernelInfo(infos_no_refresh);
  EXPECT_EQ(infos_no_refresh.count(kTestOpType), 0U);

  EXPECT_EQ(store.Refresh(), SUCCESS);

  std::map<std::string, OpInfo> infos_after;
  store.GetAllOpsKernelInfo(infos_after);
  EXPECT_GT(infos_after.size(), count_before);
  EXPECT_NE(infos_after.count(kTestOpType), 0U);

  auto op_desc = std::make_shared<OpDesc>(kTestOpType, kTestOpType);
  std::string reason;
  EXPECT_TRUE(store.CheckSupported(op_desc, reason));
}

TEST_F(UtestCustomOpsKernelInfoStore, ThreadSafety) {
  CustomOpsKernelInfoStore store;
  std::map<std::string, std::string> options;
  EXPECT_EQ(store.Initialize(options), SUCCESS);

  std::map<std::string, OpInfo> infos1;
  std::map<std::string, OpInfo> infos2;

  store.GetAllOpsKernelInfo(infos1);
  store.GetAllOpsKernelInfo(infos2);

  EXPECT_EQ(infos1.size(), infos2.size());
}

TEST_F(UtestCustomOpsKernelInfoStore, CustomGraphOptimizerCompileRunsDuringWholeGraphOnly) {
  const std::string kTestOpType = "TestCompilableOp_CompileDuringWholeGraphOnly";
  auto creator = []() -> std::unique_ptr<BaseCustomOp> { return std::make_unique<MockCompilableCustomOp>(); };
  const auto register_ret = CustomOpFactory::RegisterCustomOpCreator(AscendString(kTestOpType.c_str()), creator);
  EXPECT_TRUE((register_ret == GRAPH_SUCCESS) || (register_ret == GRAPH_FAILED));

  auto graph = ComGraphMakeShared<ComputeGraph>("custom_compile_hook_graph");
  ASSERT_NE(graph, nullptr);
  auto op_desc = ComGraphMakeShared<OpDesc>("custom_compile_hook_node", kTestOpType);
  ASSERT_NE(op_desc, nullptr);
  ASSERT_NE(graph->AddNode(op_desc), nullptr);

  MockCompilableCustomOp::ResetCompileCount();
  CustomGraphOptimizer optimizer;
  EXPECT_EQ(optimizer.OptimizeWholeGraph(*graph), SUCCESS);
  EXPECT_EQ(MockCompilableCustomOp::GetCompileCount(), 1);

  EXPECT_EQ(optimizer.OptimizeGraphBeforeBuild(*graph), SUCCESS);
  EXPECT_EQ(MockCompilableCustomOp::GetCompileCount(), 1);
}

TEST_F(UtestCustomOpsKernelInfoStore, OOptimizeWholeGraphConstructsCompileContextOutputs) {
  const std::string kTestOpType = "TestCompileContextOutputOp_OptimizerTest";
  auto graph = std::make_shared<ComputeGraph>("compile_context_output_graph");
  auto op_desc = std::make_shared<OpDesc>("compile_context_output_op", kTestOpType);
  op_desc->AppendIrOutput("y", kIrOutputRequired);
  op_desc->AppendIrOutput("dy", kIrOutputDynamic);

  GeTensorDesc required_output_desc(GeShape({8, 16}), FORMAT_ND, DT_FLOAT16);
  op_desc->AddOutputDesc("y", required_output_desc);
  GeTensorDesc dynamic_output_desc0(GeShape({16, 16}), FORMAT_ND, DT_FLOAT);
  op_desc->AddOutputDesc("dy0", dynamic_output_desc0);
  GeTensorDesc dynamic_output_desc1(GeShape({32, 16}), FORMAT_ND, DT_INT32);
  op_desc->AddOutputDesc("dy1", dynamic_output_desc1);
  ASSERT_NE(graph->AddNode(op_desc), nullptr);

  auto creator = []() -> std::unique_ptr<BaseCustomOp> { return std::make_unique<MockCompileContextOutputOp>(); };
  ASSERT_EQ(CustomOpFactory::RegisterCustomOpCreator(AscendString(kTestOpType.c_str()), creator), GRAPH_SUCCESS);

  g_compile_context_output_called.store(false);
  CustomGraphOptimizer optimizer;
  EXPECT_EQ(optimizer.OptimizeWholeGraph(*graph), SUCCESS);
  EXPECT_TRUE(g_compile_context_output_called.load());
}

TEST_F(UtestCustomOpsKernelInfoStore, GenerateTaskDeclaresAnnotatedArgsAndFillsKernelDef) {
  const std::string kTestOpType = "TestAnnotatedArgsCustomOp_BuilderTest";
  auto creator = []() -> std::unique_ptr<BaseCustomOp> {
    return std::make_unique<MockAnnotatedArgsWithConstInputCustomOp>();
  };
  ASSERT_EQ(CustomOpFactory::RegisterCustomOpCreator(AscendString(kTestOpType.c_str()), creator), GRAPH_SUCCESS);

  ComputeGraphPtr graph;
  auto node = BuildStaticCustomNodeWithConstInput(kTestOpType, graph);
  ASSERT_NE(node, nullptr);

  std::vector<domi::TaskDef> tasks;
  EXPECT_EQ(GenerateTaskForNode(node, tasks), SUCCESS);

  ASSERT_EQ(tasks.size(), 1U);
  const auto &task = tasks[0];
  EXPECT_EQ(task.stream_id(), 3U);
  EXPECT_EQ(task.type(), static_cast<uint32_t>(ModelTaskType::MODEL_TASK_CUSTOM_KERNEL));
  const auto &kernel = task.kernel();
  EXPECT_EQ(kernel.stub_func(), "custom_add_kernel");
  EXPECT_EQ(kernel.kernel_name(), "custom_add_kernel");
  EXPECT_EQ(kernel.block_dim(), 8U);
  EXPECT_EQ(kernel.args_size(), 32U);
  ASSERT_EQ(kernel.args().size(), 32);

  EXPECT_EQ(ReadUint64Slot(kernel.args(), 0U), static_cast<uint64_t>(kLogicDataMemBase + 1024U));
  EXPECT_EQ(ReadUint64Slot(kernel.args(), 1U), static_cast<uint64_t>(kLogicWeightMemBase + 4096U));
  EXPECT_EQ(ReadUint64Slot(kernel.args(), 2U), static_cast<uint64_t>(kLogicDataMemBase + 2048U));
  EXPECT_EQ(ReadUint64Slot(kernel.args(), 3U), 7U);

  const auto &kernel_context = kernel.context();
  EXPECT_EQ(kernel_context.op_index(), 7);
  EXPECT_EQ(kernel_context.args_count(), 4U);
  ASSERT_EQ(kernel_context.args_offset().size(), static_cast<int>(4U * sizeof(uint16_t)));
  EXPECT_EQ(ReadUint16Slot(kernel_context.args_offset(), 0U), 0U);
  EXPECT_EQ(ReadUint16Slot(kernel_context.args_offset(), 1U), 8U);
  EXPECT_EQ(ReadUint16Slot(kernel_context.args_offset(), 2U), 16U);
  EXPECT_EQ(ReadUint16Slot(kernel_context.args_offset(), 3U), 24U);

  EXPECT_EQ(kernel_context.args_format(), "{i_instance0*}{i_instance1*}{o_instance0*}{#7}");

  std::vector<ArgDesc> parsed_arg_descs;
  ASSERT_EQ(ArgsFormatDesc::Parse(node->GetOpDesc(), kernel_context.args_format(), parsed_arg_descs), GRAPH_SUCCESS);
  ASSERT_EQ(parsed_arg_descs.size(), 4U);
  EXPECT_EQ(parsed_arg_descs[0].addr_type, AddrType::INPUT_INSTANCE);
  EXPECT_EQ(parsed_arg_descs[0].ir_idx, 0);
  EXPECT_EQ(parsed_arg_descs[1].addr_type, AddrType::INPUT_INSTANCE);
  EXPECT_EQ(parsed_arg_descs[1].ir_idx, 1);
  EXPECT_EQ(parsed_arg_descs[2].addr_type, AddrType::OUTPUT_INSTANCE);
  EXPECT_EQ(parsed_arg_descs[2].ir_idx, 0);
  EXPECT_EQ(parsed_arg_descs[3].addr_type, AddrType::CUSTOM_VALUE);

  auto op_desc = node->GetOpDesc();
  int64_t task_args_mode = -1;
  ASSERT_TRUE(AttrUtils::GetInt(op_desc, ATTR_NAME_CUSTOM_TASK_ARGS_MODE, task_args_mode));
  EXPECT_EQ(task_args_mode, static_cast<int64_t>(CustomTaskArgsMode::kAnnotatedArgs));
  std::vector<std::string> prefixes;
  EXPECT_FALSE(AttrUtils::GetListStr(op_desc, ATTR_NAME_KERNEL_NAMES_PREFIX, prefixes));
  const std::string prefixed_attr = "_custom_launch_0_";
  std::string prefixed_kernel_name;
  EXPECT_FALSE(AttrUtils::GetStr(op_desc, prefixed_attr + ATTR_NAME_TBE_KERNEL_NAME, prefixed_kernel_name));
  Buffer prefixed_kernel_buffer;
  EXPECT_FALSE(AttrUtils::GetBytes(op_desc, prefixed_attr + ATTR_NAME_TBE_KERNEL_BUFFER, prefixed_kernel_buffer));
  EXPECT_EQ(op_desc->TryGetExtAttr(prefixed_attr + OP_EXTATTR_NAME_TBE_KERNEL, OpKernelBinPtr()), nullptr);

  std::string kernel_name;
  ASSERT_TRUE(AttrUtils::GetStr(op_desc, ATTR_NAME_TBE_KERNEL_NAME, kernel_name));
  EXPECT_EQ(kernel_name, "custom_add_kernel");
  Buffer kernel_buffer;
  ASSERT_TRUE(AttrUtils::GetBytes(op_desc, ATTR_NAME_TBE_KERNEL_BUFFER, kernel_buffer));
  auto tbe_kernel = op_desc->TryGetExtAttr(OP_EXTATTR_NAME_TBE_KERNEL, OpKernelBinPtr());
  ASSERT_NE(tbe_kernel, nullptr);
  EXPECT_EQ(tbe_kernel->GetName(), "custom_add_kernel");
  ASSERT_EQ(tbe_kernel->GetBinDataSize(), 4U);
  const uint8_t expected_bin[] = {0x11U, 0x22U, 0x33U, 0x44U};
  ASSERT_EQ(kernel_buffer.GetSize(), sizeof(expected_bin));
  ASSERT_NE(kernel_buffer.GetData(), nullptr);
  EXPECT_EQ(std::memcmp(kernel_buffer.GetData(), expected_bin, sizeof(expected_bin)), 0);
  EXPECT_EQ(std::memcmp(tbe_kernel->GetBinData(), expected_bin, sizeof(expected_bin)), 0);
}

TEST_F(UtestCustomOpsKernelInfoStore, GenerateTaskUsesPythonAnnotatedArgsAdapter) {
  GetThreadLocalContext().SetGraphOption({{ge::SOC_VERSION, "Ascend910B"}});
  const std::string kTestOpType = "TestPythonAnnotatedArgsCustomOp_BuilderTest";
  custom_op::PythonCustomOpDescriptor desc;
  desc.descriptor_key = "python_annotated_args_custom_engine";
  desc.op_type = kTestOpType;
  AddCustomOpCapability(desc.capabilities, CustomOpCapability::kAnnotatedArgs);

  custom_op::PythonCustomOpCallbacks callbacks;
  callbacks.create = CreateMockPythonAnnotatedArgsHolder;
  callbacks.destroy = DestroyMockPythonAnnotatedArgsHolder;
  callbacks.declare_launch_args = DeclareMockPythonAnnotatedArgs;
  ASSERT_TRUE(custom_op::PythonCustomOpRuntimeRegistry::Register(desc, callbacks));
  const auto creator = [desc]() -> std::unique_ptr<BaseCustomOp> {
    auto adapter = std::make_unique<custom_op::PythonCustomOpAdapter>(desc);
    if (!adapter->IsValid()) {
      return nullptr;
    }
    return adapter;
  };
  ASSERT_EQ(CustomOpFactory::RegisterCustomOpCreator(AscendString(kTestOpType.c_str()), creator), GRAPH_SUCCESS);

  ComputeGraphPtr graph;
  auto node = BuildStaticCustomNodeWithConstInput(kTestOpType, graph);
  ASSERT_NE(node, nullptr);
  EXPECT_EQ(CustomOpFactory::GetArgsRefreshStrategy(AscendString(kTestOpType.c_str())),
            ArgsRefreshStrategy::kAnnotatedArgs);

  g_python_annotated_args_declare_count.store(0U);
  std::vector<domi::TaskDef> tasks;
  ASSERT_EQ(GenerateTaskForNode(node, tasks), SUCCESS);
  ASSERT_EQ(tasks.size(), 1U);
  EXPECT_EQ(g_python_annotated_args_declare_count.load(), 1U);

  ExpectPythonAnnotatedArgsKernel(tasks[0].kernel());

  CustomOpFactory::RemoveCustomOps({AscendString(kTestOpType.c_str())});
  EXPECT_TRUE(custom_op::PythonCustomOpRuntimeRegistry::Unregister(desc.descriptor_key));
}

TEST_F(UtestCustomOpsKernelInfoStore, GenerateTaskOnNonMobileSocDeclaresAnnotatedArgsOp) {
  GetThreadLocalContext().SetGraphOption({{ge::SOC_VERSION, "Ascend910B"}});

  const std::string kTestOpType = "TestAnnotatedArgsNonMobile_BuilderTest";
  auto creator = []() -> std::unique_ptr<BaseCustomOp> {
    return std::make_unique<MockAnnotatedArgsWithConstInputCustomOp>();
  };
  ASSERT_EQ(CustomOpFactory::RegisterCustomOpCreator(AscendString(kTestOpType.c_str()), creator), GRAPH_SUCCESS);

  ComputeGraphPtr graph;
  auto node = BuildStaticCustomNodeWithConstInput(kTestOpType, graph);
  ASSERT_NE(node, nullptr);

  std::vector<domi::TaskDef> tasks;
  EXPECT_EQ(GenerateTaskForNode(node, tasks), SUCCESS);

  ASSERT_EQ(tasks.size(), 1U);
  EXPECT_EQ(tasks[0].kernel().kernel_name(), "custom_add_kernel");
  EXPECT_EQ(tasks[0].kernel().block_dim(), 8U);
  EXPECT_FALSE(tasks[0].kernel().context().args_format().empty());
}

TEST_F(UtestCustomOpsKernelInfoStore, GenerateTaskReusesImmutablePlanWithoutWorkspace) {
  const std::string kTestOpType = "TestSingleDeclareNoWorkspace_BuilderTest";
  auto creator = []() -> std::unique_ptr<BaseCustomOp> {
    return std::make_unique<MockSingleDeclareNoWorkspaceCustomOp>();
  };
  ASSERT_EQ(CustomOpFactory::RegisterCustomOpCreator(AscendString(kTestOpType.c_str()), creator), GRAPH_SUCCESS);

  ComputeGraphPtr graph;
  auto node = BuildStaticCustomNode(kTestOpType, graph);
  ASSERT_NE(node, nullptr);
  g_single_declare_no_workspace_count.store(0U);

  std::vector<domi::TaskDef> first_tasks;
  ASSERT_EQ(GenerateTaskForNode(node, first_tasks), SUCCESS);
  ASSERT_EQ(first_tasks.size(), 1U);

  auto op_desc = node->GetOpDesc();
  ASSERT_NE(op_desc, nullptr);
  op_desc->SetInputOffset({512U});
  op_desc->SetOutputOffset({3072U});
  op_desc->SetStreamId(9);
  op_desc->SetId(11);

  std::vector<domi::TaskDef> second_tasks;
  ASSERT_EQ(GenerateTaskForNode(node, second_tasks), SUCCESS);
  ASSERT_EQ(second_tasks.size(), 1U);
  EXPECT_EQ(second_tasks[0].stream_id(), 9U);
  EXPECT_EQ(second_tasks[0].kernel().context().op_index(), 11);
  EXPECT_EQ(ReadUint64Slot(second_tasks[0].kernel().args(), 0U), static_cast<uint64_t>(kLogicDataMemBase + 512U));
  EXPECT_EQ(ReadUint64Slot(second_tasks[0].kernel().args(), 1U), static_cast<uint64_t>(kLogicDataMemBase + 512U));
  EXPECT_EQ(ReadUint64Slot(second_tasks[0].kernel().args(), 2U), static_cast<uint64_t>(kLogicDataMemBase + 3072U));

  std::vector<domi::TaskDef> third_tasks;
  ASSERT_EQ(GenerateTaskForNode(node, third_tasks), SUCCESS);
  EXPECT_EQ(g_single_declare_no_workspace_count.load(), 1U);
}

TEST_F(UtestCustomOpsKernelInfoStore, CalcOpRunningParamStartsNewAnnotatedArgsPlanLifecycle) {
  const std::string kTestOpType = "TestSingleDeclareNewLifecycle_BuilderTest";
  auto creator = []() -> std::unique_ptr<BaseCustomOp> {
    return std::make_unique<MockSingleDeclareNoWorkspaceCustomOp>();
  };
  ASSERT_EQ(CustomOpFactory::RegisterCustomOpCreator(AscendString(kTestOpType.c_str()), creator), GRAPH_SUCCESS);

  ComputeGraphPtr graph;
  auto node = BuildStaticCustomNode(kTestOpType, graph);
  ASSERT_NE(node, nullptr);
  g_single_declare_no_workspace_count.store(0U);
  CustomOpsKernelBuilder builder;

  ASSERT_EQ(builder.CalcOpRunningParam(*node), SUCCESS);
  std::vector<domi::TaskDef> first_tasks;
  ASSERT_EQ(GenerateTaskForNode(node, first_tasks), SUCCESS);
  std::vector<domi::TaskDef> repeated_tasks;
  ASSERT_EQ(GenerateTaskForNode(node, repeated_tasks), SUCCESS);
  EXPECT_EQ(g_single_declare_no_workspace_count.load(), 1U);

  ASSERT_EQ(builder.CalcOpRunningParam(*node), SUCCESS);
  std::vector<domi::TaskDef> next_compile_tasks;
  ASSERT_EQ(GenerateTaskForNode(node, next_compile_tasks), SUCCESS);
  EXPECT_EQ(g_single_declare_no_workspace_count.load(), 2U);
}

TEST_F(UtestCustomOpsKernelInfoStore, GenerateTaskOnNonMobileSocWritesPrefixedKernelAttrs) {
  GetThreadLocalContext().SetGraphOption({{ge::SOC_VERSION, "Ascend910B"}});

  const std::string kTestOpType = "TestAnnotatedArgsNonMobilePrefixedAttrs_BuilderTest";
  auto creator = []() -> std::unique_ptr<BaseCustomOp> {
    return std::make_unique<MockAnnotatedArgsWithConstInputCustomOp>();
  };
  ASSERT_EQ(CustomOpFactory::RegisterCustomOpCreator(AscendString(kTestOpType.c_str()), creator), GRAPH_SUCCESS);

  ComputeGraphPtr graph;
  auto node = BuildStaticCustomNodeWithConstInput(kTestOpType, graph);
  ASSERT_NE(node, nullptr);

  std::vector<domi::TaskDef> tasks;
  ASSERT_EQ(GenerateTaskForNode(node, tasks), SUCCESS);
  ASSERT_EQ(tasks.size(), 1U);

  std::vector<std::string> prefixes;
  ASSERT_TRUE(AttrUtils::GetListStr(node->GetOpDesc(), ATTR_NAME_KERNEL_NAMES_PREFIX, prefixes));
  ASSERT_EQ(prefixes.size(), 1U);
  EXPECT_EQ(prefixes[0], "_custom_launch_0_");

  std::string kernel_name;
  ASSERT_TRUE(AttrUtils::GetStr(node->GetOpDesc(), prefixes[0] + ATTR_NAME_TBE_KERNEL_NAME, kernel_name));
  EXPECT_EQ(kernel_name, tasks[0].kernel().kernel_name());
  auto tbe_kernel = node->GetOpDesc()->TryGetExtAttr(prefixes[0] + OP_EXTATTR_NAME_TBE_KERNEL, OpKernelBinPtr());
  ASSERT_NE(tbe_kernel, nullptr);
  EXPECT_EQ(tbe_kernel->GetName(), tasks[0].kernel().kernel_name());
}

TEST_F(UtestCustomOpsKernelInfoStore, GenerateTaskSupportsInvalidOptionalBeforeRequiredInput) {
  const std::string kTestOpType = "TestInvalidOptionalBeforeRequiredInput_BuilderTest";
  auto creator = []() -> std::unique_ptr<BaseCustomOp> {
    return std::make_unique<MockAnnotatedArgsWithInvalidOptionalAndRequiredInputCustomOp>();
  };
  ASSERT_EQ(CustomOpFactory::RegisterCustomOpCreator(AscendString(kTestOpType.c_str()), creator), GRAPH_SUCCESS);

  ComputeGraphPtr graph;
  auto node = BuildStaticCustomNodeWithInvalidOptionalInput(kTestOpType, false, graph);
  ASSERT_NE(node, nullptr);

  std::vector<domi::TaskDef> tasks;
  ASSERT_EQ(GenerateTaskForNode(node, tasks), SUCCESS);

  ASSERT_EQ(tasks.size(), 1U);
  const auto &kernel = tasks[0].kernel();
  ASSERT_EQ(kernel.args().size(), 24);
  EXPECT_EQ(ReadUint64Slot(kernel.args(), 0U), static_cast<uint64_t>(kLogicDataMemBase + 1024U));
  EXPECT_EQ(ReadUint64Slot(kernel.args(), 1U), static_cast<uint64_t>(kLogicDataMemBase + 3072U));
  EXPECT_EQ(kernel.context().args_format(), "{i_instance0*}{i_instance1*}{o_instance0*}");

  auto op_desc = node->GetOpDesc();
  ASSERT_NE(op_desc, nullptr);
  op_desc->SetInputOffset({512U, 3584U});
  op_desc->SetOutputOffset({3840U});

  std::vector<domi::TaskDef> materialized_tasks;
  ASSERT_EQ(GenerateTaskForNode(node, materialized_tasks), SUCCESS);
  ASSERT_EQ(materialized_tasks.size(), 1U);
  const auto &materialized_kernel = materialized_tasks[0].kernel();
  EXPECT_EQ(ReadUint64Slot(materialized_kernel.args(), 0U), static_cast<uint64_t>(kLogicDataMemBase + 512U));
  EXPECT_EQ(ReadUint64Slot(materialized_kernel.args(), 1U), static_cast<uint64_t>(kLogicDataMemBase + 3584U));
  EXPECT_EQ(ReadUint64Slot(materialized_kernel.args(), 2U), static_cast<uint64_t>(kLogicDataMemBase + 3840U));
}

TEST_F(UtestCustomOpsKernelInfoStore, GenerateTaskSupportsInvalidOptionalBeforeConstInput) {
  const std::string kTestOpType = "TestInvalidOptionalBeforeConstInput_BuilderTest";
  auto creator = []() -> std::unique_ptr<BaseCustomOp> {
    return std::make_unique<MockAnnotatedArgsWithInvalidOptionalAndConstInputCustomOp>();
  };
  ASSERT_EQ(CustomOpFactory::RegisterCustomOpCreator(AscendString(kTestOpType.c_str()), creator), GRAPH_SUCCESS);

  ComputeGraphPtr graph;
  auto node = BuildStaticCustomNodeWithInvalidOptionalInput(kTestOpType, true, graph);
  ASSERT_NE(node, nullptr);

  std::vector<domi::TaskDef> tasks;
  ASSERT_EQ(GenerateTaskForNode(node, tasks), SUCCESS);

  ASSERT_EQ(tasks.size(), 1U);
  const auto &kernel = tasks[0].kernel();
  ASSERT_EQ(kernel.args().size(), 24);
  EXPECT_EQ(ReadUint64Slot(kernel.args(), 0U), static_cast<uint64_t>(kLogicDataMemBase + 1024U));
  EXPECT_EQ(ReadUint64Slot(kernel.args(), 1U), static_cast<uint64_t>(kLogicWeightMemBase + 4096U));
  EXPECT_EQ(kernel.context().args_format(), "{i_instance0*}{i_instance1*}{o_instance0*}");
}

TEST_F(UtestCustomOpsKernelInfoStore, GenerateTaskSupportsPresentOptionalInput) {
  const std::string kTestOpType = "TestPresentOptionalInput_BuilderTest";
  auto creator = []() -> std::unique_ptr<BaseCustomOp> {
    return std::make_unique<MockAnnotatedArgsWithPresentOptionalInputCustomOp>();
  };
  ASSERT_EQ(CustomOpFactory::RegisterCustomOpCreator(AscendString(kTestOpType.c_str()), creator), GRAPH_SUCCESS);

  ComputeGraphPtr graph;
  auto node = BuildStaticCustomNodeWithPresentOptionalInput(kTestOpType, graph);
  ASSERT_NE(node, nullptr);

  std::vector<domi::TaskDef> tasks;
  ASSERT_EQ(GenerateTaskForNode(node, tasks), SUCCESS);

  ASSERT_EQ(tasks.size(), 1U);
  const auto &kernel = tasks[0].kernel();
  ASSERT_EQ(kernel.args().size(), 32);
  EXPECT_EQ(ReadUint64Slot(kernel.args(), 0U), static_cast<uint64_t>(kLogicDataMemBase + 1024U));
  EXPECT_EQ(ReadUint64Slot(kernel.args(), 1U), static_cast<uint64_t>(kLogicDataMemBase + 2048U));
  EXPECT_EQ(ReadUint64Slot(kernel.args(), 2U), static_cast<uint64_t>(kLogicDataMemBase + 3072U));
  EXPECT_EQ(ReadUint64Slot(kernel.args(), 3U), static_cast<uint64_t>(kLogicDataMemBase + 1280U));
  EXPECT_EQ(kernel.context().args_format(), "{i_instance0*}{i_instance1*}{i_instance2*}{o_instance0*}");
  EXPECT_EQ(kernel.context().args_count(), 4U);
  EXPECT_EQ(kernel.args_size(), 4U * sizeof(uint64_t));
}

TEST_F(UtestCustomOpsKernelInfoStore, GenerateTaskSupportsExplicitZeroSlotForMissingOptionalInput) {
  const std::string kTestOpType = "TestExplicitOptionalZeroSlot_BuilderTest";
  auto creator = []() -> std::unique_ptr<BaseCustomOp> {
    return std::make_unique<MockAnnotatedArgsWithExplicitOptionalZeroCustomOp>();
  };
  ASSERT_EQ(CustomOpFactory::RegisterCustomOpCreator(AscendString(kTestOpType.c_str()), creator), GRAPH_SUCCESS);

  ComputeGraphPtr graph;
  auto node = BuildStaticCustomNodeWithInvalidOptionalInput(kTestOpType, false, graph);
  ASSERT_NE(node, nullptr);

  std::vector<domi::TaskDef> tasks;
  ASSERT_EQ(GenerateTaskForNode(node, tasks), SUCCESS);

  ASSERT_EQ(tasks.size(), 1U);
  const auto &kernel = tasks[0].kernel();
  EXPECT_EQ(kernel.context().args_format(), "{i_instance0*}{#0}{i_instance1*}{o_instance0*}");
  EXPECT_EQ(ReadUint64Slot(kernel.args(), 1U), 0U);
  EXPECT_EQ(kernel.context().args_count(), 4U);
  EXPECT_EQ(kernel.args_size(), 4U * sizeof(uint64_t));
}

TEST_F(UtestCustomOpsKernelInfoStore, GenerateTaskSupportsDynamicIoWithReorderedRepeatedSubsetIndexes) {
  const std::string kTestOpType = "TestDynamicIoCustomOp_BuilderTest";
  auto creator = []() -> std::unique_ptr<BaseCustomOp> {
    return std::make_unique<MockAnnotatedArgsWithDynamicIoCustomOp>();
  };
  ASSERT_EQ(CustomOpFactory::RegisterCustomOpCreator(AscendString(kTestOpType.c_str()), creator), GRAPH_SUCCESS);

  ComputeGraphPtr graph;
  auto node = BuildStaticCustomNodeWithDynamicIo(kTestOpType, graph);
  ASSERT_NE(node, nullptr);

  std::vector<domi::TaskDef> tasks;
  ASSERT_EQ(GenerateTaskForNode(node, tasks), SUCCESS);

  ASSERT_EQ(tasks.size(), 1U);
  const auto &kernel = tasks[0].kernel();
  EXPECT_EQ(kernel.context().args_format(), "{i_instance3*}{i_instance1*}{i_instance1*}{o_instance2*}{o_instance0*}");
  EXPECT_EQ(kernel.context().args_count(), 5U);
  EXPECT_EQ(kernel.args_size(), 5U * sizeof(uint64_t));
  EXPECT_EQ(ReadUint64Slot(kernel.args(), 0U), static_cast<uint64_t>(kLogicDataMemBase + 1024U));
  EXPECT_EQ(ReadUint64Slot(kernel.args(), 1U), static_cast<uint64_t>(kLogicDataMemBase + 512U));
  EXPECT_EQ(ReadUint64Slot(kernel.args(), 2U), static_cast<uint64_t>(kLogicDataMemBase + 512U));
  EXPECT_EQ(ReadUint64Slot(kernel.args(), 3U), static_cast<uint64_t>(kLogicDataMemBase + 1792U));
  EXPECT_EQ(ReadUint64Slot(kernel.args(), 4U), static_cast<uint64_t>(kLogicDataMemBase + 1280U));

  std::vector<ArgDesc> parsed_arg_descs;
  ASSERT_EQ(ArgsFormatDesc::Parse(node->GetOpDesc(), kernel.context().args_format(), parsed_arg_descs), GRAPH_SUCCESS);
  ASSERT_EQ(parsed_arg_descs.size(), 5U);
  EXPECT_EQ(parsed_arg_descs[0].addr_type, AddrType::INPUT_INSTANCE);
  EXPECT_EQ(parsed_arg_descs[0].ir_idx, 3);
  EXPECT_EQ(parsed_arg_descs[1].addr_type, AddrType::INPUT_INSTANCE);
  EXPECT_EQ(parsed_arg_descs[1].ir_idx, 1);
  EXPECT_EQ(parsed_arg_descs[2].addr_type, AddrType::INPUT_INSTANCE);
  EXPECT_EQ(parsed_arg_descs[2].ir_idx, 1);
  EXPECT_EQ(parsed_arg_descs[3].addr_type, AddrType::OUTPUT_INSTANCE);
  EXPECT_EQ(parsed_arg_descs[3].ir_idx, 2);
  EXPECT_EQ(parsed_arg_descs[4].addr_type, AddrType::OUTPUT_INSTANCE);
  EXPECT_EQ(parsed_arg_descs[4].ir_idx, 0);

  auto op_desc = node->GetOpDesc();
  ASSERT_NE(op_desc, nullptr);
  op_desc->SetInputOffset({128U, 640U, 896U, 1152U});
  op_desc->SetOutputOffset({1408U, 1664U, 1920U});

  std::vector<domi::TaskDef> materialized_tasks;
  ASSERT_EQ(GenerateTaskForNode(node, materialized_tasks), SUCCESS);
  ASSERT_EQ(materialized_tasks.size(), 1U);
  const auto &materialized_kernel = materialized_tasks[0].kernel();
  EXPECT_EQ(ReadUint64Slot(materialized_kernel.args(), 0U), static_cast<uint64_t>(kLogicDataMemBase + 1152U));
  EXPECT_EQ(ReadUint64Slot(materialized_kernel.args(), 1U), static_cast<uint64_t>(kLogicDataMemBase + 640U));
  EXPECT_EQ(ReadUint64Slot(materialized_kernel.args(), 2U), static_cast<uint64_t>(kLogicDataMemBase + 640U));
  EXPECT_EQ(ReadUint64Slot(materialized_kernel.args(), 3U), static_cast<uint64_t>(kLogicDataMemBase + 1920U));
  EXPECT_EQ(ReadUint64Slot(materialized_kernel.args(), 4U), static_cast<uint64_t>(kLogicDataMemBase + 1408U));
}

TEST_F(UtestCustomOpsKernelInfoStore, GenerateTaskArgsOffsetLimit8192Success) {
  const std::string kTestOpType = "TestAnnotatedArgsOffset8192";
  auto creator = []() -> std::unique_ptr<BaseCustomOp> {
    return std::make_unique<MockAnnotatedArgsWithSlotCountCustomOp>(8192U);
  };
  ASSERT_EQ(CustomOpFactory::RegisterCustomOpCreator(AscendString(kTestOpType.c_str()), creator), GRAPH_SUCCESS);

  ComputeGraphPtr graph;
  auto node = BuildStaticCustomNode(kTestOpType, graph);
  ASSERT_NE(node, nullptr);

  std::vector<domi::TaskDef> tasks;
  ASSERT_EQ(GenerateTaskForNode(node, tasks), SUCCESS);
  ASSERT_EQ(tasks.size(), 1U);

  const auto &kernel = tasks[0].kernel();
  EXPECT_EQ(kernel.context().args_count(), 8192U);
  EXPECT_EQ(kernel.args_size(), 65536U);
  EXPECT_EQ(kernel.context().args_offset().size(), static_cast<int>(8192U * sizeof(uint16_t)));
  EXPECT_EQ(ReadUint16Slot(kernel.context().args_offset(), 8191U), 65528U);
  EXPECT_EQ(ReadUint64Slot(kernel.args(), 8191U), 8191U);
}

TEST_F(UtestCustomOpsKernelInfoStore, GenerateTaskArgsOffsetLimit8193Failed) {
  const std::string kTestOpType = "TestAnnotatedArgsOffset8193";
  auto creator = []() -> std::unique_ptr<BaseCustomOp> {
    return std::make_unique<MockAnnotatedArgsWithSlotCountCustomOp>(8193U);
  };
  ASSERT_EQ(CustomOpFactory::RegisterCustomOpCreator(AscendString(kTestOpType.c_str()), creator), GRAPH_SUCCESS);

  ComputeGraphPtr graph;
  auto node = BuildStaticCustomNode(kTestOpType, graph);
  ASSERT_NE(node, nullptr);

  std::vector<domi::TaskDef> tasks;
  EXPECT_NE(GenerateTaskForNode(node, tasks), SUCCESS);
  EXPECT_TRUE(tasks.empty());
}

TEST_F(UtestCustomOpsKernelInfoStore, GenerateTaskOnNonMobileSocFillsBasicCustomKernelTask) {
  GetThreadLocalContext().SetGraphOption({{ge::SOC_VERSION, "Ascend910B"}});

  ComputeGraphPtr graph;
  auto node = BuildStaticCustomNode("TestUnregisteredCustomOp_NonMobileSoc", graph);
  ASSERT_NE(node, nullptr);

  std::vector<domi::TaskDef> tasks;
  EXPECT_EQ(GenerateTaskForNode(node, tasks), SUCCESS);

  ASSERT_EQ(tasks.size(), 1U);
  const auto &task = tasks[0];
  EXPECT_EQ(task.stream_id(), 3U);
  EXPECT_EQ(task.type(), static_cast<uint32_t>(ModelTaskType::MODEL_TASK_CUSTOM_KERNEL));
  EXPECT_EQ(task.sqe_num(), 5U);
  EXPECT_EQ(task.kernel().context().op_index(), 7);
  int64_t task_args_mode = -1;
  ASSERT_TRUE(AttrUtils::GetInt(node->GetOpDesc(), ATTR_NAME_CUSTOM_TASK_ARGS_MODE, task_args_mode));
  EXPECT_EQ(task_args_mode, static_cast<int64_t>(CustomTaskArgsMode::kNone));
}

TEST_F(UtestCustomOpsKernelInfoStore, GenerateTaskOnNonMobileSocUsesUpdateCallbackWhenBothRefreshInterfacesExist) {
  GetThreadLocalContext().SetGraphOption({{ge::SOC_VERSION, "Ascend910B"}});
  g_both_refresh_interfaces_declare_count.store(0U);
  const std::string kTestOpType = "TestBothRefreshInterfacesCustomOp_NonMobile_BuilderTest";
  auto creator = []() -> std::unique_ptr<BaseCustomOp> {
    return std::make_unique<MockBothRefreshInterfacesCustomOp>();
  };
  ASSERT_EQ(CustomOpFactory::RegisterCustomOpCreator(AscendString(kTestOpType.c_str()), creator), GRAPH_SUCCESS);
  EXPECT_EQ(CustomOpFactory::GetArgsRefreshStrategy(AscendString(kTestOpType.c_str())),
            ArgsRefreshStrategy::kUpdateCallback);

  ComputeGraphPtr graph;
  auto node = BuildStaticCustomNode(kTestOpType, graph);
  ASSERT_NE(node, nullptr);

  std::vector<domi::TaskDef> tasks;
  EXPECT_EQ(GenerateTaskForNode(node, tasks), SUCCESS);

  ASSERT_EQ(tasks.size(), 1U);
  const auto &task = tasks[0];
  EXPECT_EQ(task.stream_id(), 3U);
  EXPECT_EQ(task.type(), static_cast<uint32_t>(ModelTaskType::MODEL_TASK_CUSTOM_KERNEL));
  EXPECT_EQ(task.sqe_num(), 5U);
  EXPECT_EQ(task.kernel().context().op_index(), 7);
  EXPECT_TRUE(task.kernel().kernel_name().empty());
  EXPECT_TRUE(task.kernel().context().args_format().empty());
  EXPECT_EQ(g_both_refresh_interfaces_declare_count.load(), 0U);
  int64_t task_args_mode = -1;
  ASSERT_TRUE(AttrUtils::GetInt(node->GetOpDesc(), ATTR_NAME_CUSTOM_TASK_ARGS_MODE, task_args_mode));
  EXPECT_EQ(task_args_mode, static_cast<int64_t>(CustomTaskArgsMode::kUpdateCallback));
}

TEST_F(UtestCustomOpsKernelInfoStore, GenerateTaskOnMobileSocRejectsUpdateCallbackWhenBothRefreshInterfacesExist) {
  g_both_refresh_interfaces_declare_count.store(0U);
  const std::string kTestOpType = "TestBothRefreshInterfacesCustomOp_Mobile_BuilderTest";
  auto creator = []() -> std::unique_ptr<BaseCustomOp> {
    return std::make_unique<MockBothRefreshInterfacesCustomOp>();
  };
  ASSERT_EQ(CustomOpFactory::RegisterCustomOpCreator(AscendString(kTestOpType.c_str()), creator), GRAPH_SUCCESS);
  EXPECT_EQ(CustomOpFactory::GetArgsRefreshStrategy(AscendString(kTestOpType.c_str())),
            ArgsRefreshStrategy::kUpdateCallback);

  ComputeGraphPtr graph;
  auto node = BuildStaticCustomNode(kTestOpType, graph);
  ASSERT_NE(node, nullptr);

  std::vector<domi::TaskDef> tasks;
  EXPECT_EQ(GenerateTaskForNode(node, tasks), INTERNAL_ERROR);
  EXPECT_TRUE(tasks.empty());
  EXPECT_EQ(g_both_refresh_interfaces_declare_count.load(), 0U);
}

TEST_F(UtestCustomOpsKernelInfoStore, GenerateTaskOnMobileSocRejectsDefaultStrategy) {
  const std::string kTestOpType = "TestUnregisteredCustomOp_MobileDefaultStrategy_BuilderTest";
  EXPECT_EQ(CustomOpFactory::GetArgsRefreshStrategy(AscendString(kTestOpType.c_str())), ArgsRefreshStrategy::kNone);

  ComputeGraphPtr graph;
  auto node = BuildStaticCustomNode(kTestOpType, graph);
  ASSERT_NE(node, nullptr);

  std::vector<domi::TaskDef> tasks;
  EXPECT_EQ(GenerateTaskForNode(node, tasks), INTERNAL_ERROR);
  EXPECT_TRUE(tasks.empty());
}

TEST_F(UtestCustomOpsKernelInfoStore, GenerateTaskProbeRecordsDynamicWorkspace) {
  const std::string kTestOpType = "TestAnnotatedArgsWithWorkspaceProbe_BuilderTest";
  auto creator = []() -> std::unique_ptr<BaseCustomOp> {
    return std::make_unique<MockAnnotatedArgsWithWorkspaceCustomOp>();
  };
  ASSERT_EQ(CustomOpFactory::RegisterCustomOpCreator(AscendString(kTestOpType.c_str()), creator), GRAPH_SUCCESS);

  ComputeGraphPtr graph;
  auto node = BuildStaticCustomNode(kTestOpType, graph);
  ASSERT_NE(node, nullptr);

  std::vector<domi::TaskDef> tasks;
  EXPECT_EQ(GenerateTaskForNode(node, tasks), SUCCESS);

  std::vector<int64_t> append_ws;
  ASSERT_TRUE(AttrUtils::GetListInt(node->GetOpDesc(), "_append_ws", append_ws));
  ASSERT_EQ(append_ws.size(), 1U);
  EXPECT_EQ(append_ws[0], 512);

  bool custom_append_ws = false;
  EXPECT_TRUE(AttrUtils::GetBool(node->GetOpDesc(), "_custom_omc_append_ws", custom_append_ws));
  EXPECT_TRUE(custom_append_ws);
}

TEST_F(UtestCustomOpsKernelInfoStore, GenerateTaskSupportsMultipleWorkspaceArgs) {
  const std::string kTestOpType = "TestAnnotatedArgsWithMultipleWorkspaces_BuilderTest";
  auto creator = []() -> std::unique_ptr<BaseCustomOp> {
    return std::make_unique<MockAnnotatedArgsWithMultipleWorkspacesCustomOp>();
  };
  ASSERT_EQ(CustomOpFactory::RegisterCustomOpCreator(AscendString(kTestOpType.c_str()), creator), GRAPH_SUCCESS);

  ComputeGraphPtr graph;
  auto node = BuildStaticCustomNode(kTestOpType, graph);
  ASSERT_NE(node, nullptr);

  std::vector<domi::TaskDef> tasks;
  EXPECT_EQ(GenerateTaskForNode(node, tasks), SUCCESS);

  ASSERT_EQ(tasks.size(), 1U);
  const auto &kernel = tasks[0].kernel();
  ASSERT_EQ(kernel.args().size(), 32);
  EXPECT_EQ(ReadUint64Slot(kernel.args(), 0U), static_cast<uint64_t>(kLogicDataMemBase + 1024U));
  EXPECT_EQ(ReadUint64Slot(kernel.args(), 1U), static_cast<uint64_t>(kLogicDataMemBase + 2048U));
  EXPECT_EQ(ReadUint64Slot(kernel.args(), 2U), static_cast<uint64_t>(kLogicDataMemBase + 4096U));
  EXPECT_EQ(ReadUint64Slot(kernel.args(), 3U), static_cast<uint64_t>(kLogicDataMemBase + 4096U + 512U));
  EXPECT_EQ(kernel.context().args_format(), "{i_instance0*}{o_instance0*}{ws0*}{ws1*}");

  std::vector<ArgDesc> parsed_arg_descs;
  ASSERT_EQ(ArgsFormatDescUtils::Parse(kernel.context().args_format(), parsed_arg_descs), GRAPH_SUCCESS);
  ASSERT_EQ(parsed_arg_descs.size(), 4U);
  EXPECT_EQ(parsed_arg_descs[2].addr_type, AddrType::WORKSPACE);
  EXPECT_EQ(parsed_arg_descs[2].ir_idx, 0);
  EXPECT_EQ(parsed_arg_descs[3].addr_type, AddrType::WORKSPACE);
  EXPECT_EQ(parsed_arg_descs[3].ir_idx, 1);
}

TEST_F(UtestCustomOpsKernelInfoStore, GenerateTaskOnMobileSocRejectsMultipleLaunches) {
  const std::string kTestOpType = "TestMultipleLaunchesCustomOp_MobileReject_BuilderTest";
  auto creator = []() -> std::unique_ptr<BaseCustomOp> {
    return std::make_unique<MockAnnotatedArgsWithMultipleLaunchesCustomOp>();
  };
  ASSERT_EQ(CustomOpFactory::RegisterCustomOpCreator(AscendString(kTestOpType.c_str()), creator), GRAPH_SUCCESS);

  ComputeGraphPtr graph;
  auto node = BuildStaticCustomNode(kTestOpType, graph);
  ASSERT_NE(node, nullptr);

  std::vector<domi::TaskDef> tasks;
  EXPECT_EQ(GenerateTaskForNode(node, tasks), INTERNAL_ERROR);
  EXPECT_TRUE(tasks.empty());
}

TEST_F(UtestCustomOpsKernelInfoStore, GenerateTaskSupportsMultipleLaunches) {
  GetThreadLocalContext().SetGraphOption({{ge::SOC_VERSION, "Ascend910B"}});

  const std::string kTestOpType = "TestMultipleLaunchesCustomOp_BuilderTest";
  auto creator = []() -> std::unique_ptr<BaseCustomOp> {
    return std::make_unique<MockAnnotatedArgsWithMultipleLaunchesCustomOp>();
  };
  ASSERT_EQ(CustomOpFactory::RegisterCustomOpCreator(AscendString(kTestOpType.c_str()), creator), GRAPH_SUCCESS);

  ComputeGraphPtr graph;
  auto node = BuildStaticCustomNode(kTestOpType, graph);
  ASSERT_NE(node, nullptr);

  std::vector<domi::TaskDef> tasks;
  EXPECT_EQ(GenerateTaskForNode(node, tasks), SUCCESS);

  ASSERT_EQ(tasks.size(), 2U);
  EXPECT_EQ(tasks[0].stream_id(), 3U);
  EXPECT_EQ(tasks[1].stream_id(), 3U);
  EXPECT_EQ(tasks[0].kernel().context().op_index(), 7);
  EXPECT_EQ(tasks[1].kernel().context().op_index(), 7);
  EXPECT_NE(tasks[0].kernel().kernel_name(), tasks[1].kernel().kernel_name());
  EXPECT_FALSE(tasks[0].kernel().context().args_format().empty());
  EXPECT_FALSE(tasks[1].kernel().context().args_format().empty());

  auto op_desc = node->GetOpDesc();
  ASSERT_NE(op_desc, nullptr);
  op_desc->SetInputOffset({512U});
  op_desc->SetOutputOffset({3072U});
  op_desc->SetStreamId(9);
  op_desc->SetId(11);

  std::vector<domi::TaskDef> materialized_tasks;
  ASSERT_EQ(GenerateTaskForNode(node, materialized_tasks), SUCCESS);
  ASSERT_EQ(materialized_tasks.size(), 2U);
  EXPECT_EQ(materialized_tasks[0].stream_id(), 9U);
  EXPECT_EQ(materialized_tasks[1].stream_id(), 9U);
  EXPECT_EQ(materialized_tasks[0].kernel().context().op_index(), 11);
  EXPECT_EQ(materialized_tasks[1].kernel().context().op_index(), 11);
  EXPECT_EQ(ReadUint64Slot(materialized_tasks[0].kernel().args(), 0U), static_cast<uint64_t>(kLogicDataMemBase + 512U));
  EXPECT_EQ(ReadUint64Slot(materialized_tasks[1].kernel().args(), 0U),
            static_cast<uint64_t>(kLogicDataMemBase + 3072U));
}

TEST_F(UtestCustomOpsKernelInfoStore, GenerateTaskAllowsSameKernelNameWithDifferentBins) {
  GetThreadLocalContext().SetGraphOption({{ge::SOC_VERSION, "Ascend910B"}});

  const std::string kTestOpType = "TestSameNameDifferentBinsCustomOp_BuilderTest";
  auto creator = []() -> std::unique_ptr<BaseCustomOp> {
    return std::make_unique<MockAnnotatedArgsWithSameNameDifferentBinsCustomOp>();
  };
  ASSERT_EQ(CustomOpFactory::RegisterCustomOpCreator(AscendString(kTestOpType.c_str()), creator), GRAPH_SUCCESS);

  ComputeGraphPtr graph;
  auto node = BuildStaticCustomNode(kTestOpType, graph);
  ASSERT_NE(node, nullptr);

  std::vector<domi::TaskDef> tasks;
  ASSERT_EQ(GenerateTaskForNode(node, tasks), SUCCESS);
  ASSERT_EQ(tasks.size(), 2U);
  EXPECT_EQ(tasks[0].kernel().kernel_name(), "custom_shared_kernel");
  EXPECT_EQ(tasks[1].kernel().kernel_name(), "custom_shared_kernel");
}

TEST_F(UtestCustomOpsKernelInfoStore, GenerateTaskMultipleLaunchesWritesPrefixedKernelAttrs) {
  GetThreadLocalContext().SetGraphOption({{ge::SOC_VERSION, "Ascend910B"}});

  const std::string kTestOpType = "TestMultipleLaunchesKernelAttrs_BuilderTest";
  auto creator = []() -> std::unique_ptr<BaseCustomOp> {
    return std::make_unique<MockAnnotatedArgsWithMultipleLaunchesCustomOp>();
  };
  ASSERT_EQ(CustomOpFactory::RegisterCustomOpCreator(AscendString(kTestOpType.c_str()), creator), GRAPH_SUCCESS);

  ComputeGraphPtr graph;
  auto node = BuildStaticCustomNode(kTestOpType, graph);
  ASSERT_NE(node, nullptr);

  std::vector<domi::TaskDef> tasks;
  ASSERT_EQ(GenerateTaskForNode(node, tasks), SUCCESS);
  ASSERT_EQ(tasks.size(), 2U);

  std::vector<std::string> prefixes;
  ASSERT_TRUE(AttrUtils::GetListStr(node->GetOpDesc(), ATTR_NAME_KERNEL_NAMES_PREFIX, prefixes));
  ASSERT_EQ(prefixes.size(), 2U);
  EXPECT_EQ(prefixes[0], "_custom_launch_0_");
  EXPECT_EQ(prefixes[1], "_custom_launch_1_");

  for (size_t i = 0U; i < prefixes.size(); ++i) {
    const auto &prefix = prefixes[i];
    std::string kernel_name;
    ASSERT_TRUE(AttrUtils::GetStr(node->GetOpDesc(), prefix + ATTR_NAME_TBE_KERNEL_NAME, kernel_name));
    EXPECT_EQ(kernel_name, tasks[i].kernel().kernel_name());
    std::string binary_magic;
    ASSERT_TRUE(AttrUtils::GetStr(node->GetOpDesc(), prefix + TVM_ATTR_NAME_MAGIC, binary_magic));
    EXPECT_EQ(binary_magic, "RT_DEV_BINARY_MAGIC_ELF_AIVEC");
    auto tbe_kernel = node->GetOpDesc()->TryGetExtAttr(prefix + OP_EXTATTR_NAME_TBE_KERNEL, OpKernelBinPtr());
    ASSERT_NE(tbe_kernel, nullptr);
    EXPECT_EQ(tbe_kernel->GetName(), tasks[i].kernel().kernel_name());
  }
}

TEST_F(UtestCustomOpsKernelInfoStore, GenerateTaskMultipleLaunchesProducesModelBuilderReadableKernelAttrs) {
  GetThreadLocalContext().SetGraphOption({{ge::SOC_VERSION, "Ascend910B"}});

  const std::string kTestOpType = "TestMultiLaunchModelBuilderReadableAttrs_BuilderTest";
  auto creator = []() -> std::unique_ptr<BaseCustomOp> {
    return std::make_unique<MockAnnotatedArgsWithMultipleLaunchesCustomOp>();
  };
  ASSERT_EQ(CustomOpFactory::RegisterCustomOpCreator(AscendString(kTestOpType.c_str()), creator), GRAPH_SUCCESS);

  ComputeGraphPtr graph;
  auto node = BuildStaticCustomNode(kTestOpType, graph);
  ASSERT_NE(node, nullptr);

  std::vector<domi::TaskDef> tasks;
  ASSERT_EQ(GenerateTaskForNode(node, tasks), SUCCESS);
  ASSERT_EQ(tasks.size(), 2U);

  std::vector<std::string> prefixes;
  ASSERT_TRUE(AttrUtils::GetListStr(node->GetOpDesc(), ATTR_NAME_KERNEL_NAMES_PREFIX, prefixes));
  for (size_t i = 0U; i < prefixes.size(); ++i) {
    Buffer kernel_buffer;
    std::string kernel_name;
    ASSERT_TRUE(AttrUtils::GetStr(node->GetOpDesc(), prefixes[i] + ATTR_NAME_TBE_KERNEL_NAME, kernel_name));
    ASSERT_TRUE(AttrUtils::GetBytes(node->GetOpDesc(), prefixes[i] + ATTR_NAME_TBE_KERNEL_BUFFER, kernel_buffer));
    EXPECT_EQ(kernel_name, tasks[i].kernel().kernel_name());
    EXPECT_GT(kernel_buffer.GetSize(), 0U);
  }
}

TEST_F(UtestCustomOpsKernelInfoStore, GenerateTaskFailsWhenDeclareSetsMismatchStreamId) {
  const std::string kTestOpType = "TestMismatchStreamCustomOp_BuilderTest";
  auto creator = []() -> std::unique_ptr<BaseCustomOp> {
    return std::make_unique<MockAnnotatedArgsWithMismatchStreamCustomOp>();
  };
  ASSERT_EQ(CustomOpFactory::RegisterCustomOpCreator(AscendString(kTestOpType.c_str()), creator), GRAPH_SUCCESS);

  ExpectGenerateTaskFailedWithStatus(kTestOpType, INTERNAL_ERROR);
}

TEST_F(UtestCustomOpsKernelInfoStore, GenerateTaskFinalDeclareUsesPlannedWorkspaceOffset) {
  const std::string kTestOpType = "TestAnnotatedArgsWithWorkspaceFinal_BuilderTest";
  auto creator = []() -> std::unique_ptr<BaseCustomOp> {
    return std::make_unique<MockAnnotatedArgsWithWorkspaceCustomOp>();
  };
  ASSERT_EQ(CustomOpFactory::RegisterCustomOpCreator(AscendString(kTestOpType.c_str()), creator), GRAPH_SUCCESS);

  ComputeGraphPtr graph;
  auto node = BuildStaticCustomNode(kTestOpType, graph);
  ASSERT_NE(node, nullptr);

  std::vector<domi::TaskDef> probe_tasks;
  ASSERT_EQ(GenerateTaskForNode(node, probe_tasks), SUCCESS);
  FinalizeCustomWorkspaceForDirectBuilderTest(node, 4096);

  CustomOpsKernelBuilder builder;
  RunContext final_context = {};
  final_context.dataMemBase = reinterpret_cast<uint8_t *>(kLogicDataMemBase);
  final_context.dataMemSize = 4608U;
  final_context.weightMemBase = reinterpret_cast<uint8_t *>(kLogicWeightMemBase);
  final_context.weightMemSize = 4096U;

  std::vector<domi::TaskDef> final_tasks;
  EXPECT_EQ(builder.GenerateTask(*node, final_context, final_tasks), SUCCESS);

  ASSERT_EQ(final_tasks.size(), 1U);
  const auto &kernel = final_tasks[0].kernel();
  ASSERT_EQ(kernel.args().size(), 24);
  EXPECT_EQ(ReadUint64Slot(kernel.args(), 0U), static_cast<uint64_t>(kLogicDataMemBase + 1024U));
  EXPECT_EQ(ReadUint64Slot(kernel.args(), 1U), static_cast<uint64_t>(kLogicDataMemBase + 2048U));
  EXPECT_EQ(ReadUint64Slot(kernel.args(), 2U), static_cast<uint64_t>(kLogicDataMemBase + 4096U));

  std::vector<ArgDesc> parsed_arg_descs;
  ASSERT_EQ(ArgsFormatDescUtils::Parse(kernel.context().args_format(), parsed_arg_descs), GRAPH_SUCCESS);
  ASSERT_EQ(parsed_arg_descs.size(), 3U);
  EXPECT_EQ(parsed_arg_descs[2].addr_type, AddrType::WORKSPACE);
  EXPECT_EQ(parsed_arg_descs[2].ir_idx, 0);
}

TEST_F(UtestCustomOpsKernelInfoStore, GenerateTaskReusesImmutablePlanForFinalWorkspaceAndCurrentWeightBase) {
  const std::string kTestOpType = "TestSingleDeclareWorkspaceFinal_BuilderTest";
  auto creator = []() -> std::unique_ptr<BaseCustomOp> {
    return std::make_unique<MockSingleDeclareWorkspaceCustomOp>();
  };
  ASSERT_EQ(CustomOpFactory::RegisterCustomOpCreator(AscendString(kTestOpType.c_str()), creator), GRAPH_SUCCESS);

  ComputeGraphPtr graph;
  auto node = BuildStaticCustomNodeWithConstInput(kTestOpType, graph);
  ASSERT_NE(node, nullptr);
  g_single_declare_workspace_count.store(0U);

  std::vector<domi::TaskDef> probe_tasks;
  ASSERT_EQ(GenerateTaskForNode(node, probe_tasks), SUCCESS);
  FinalizeCustomWorkspaceForDirectBuilderTest(node, 4096);

  CustomOpsKernelBuilder builder;
  RunContext final_context = {};
  final_context.dataMemBase = reinterpret_cast<uint8_t *>(kLogicDataMemBase);
  final_context.dataMemSize = 4608U;
  final_context.weightMemBase = reinterpret_cast<uint8_t *>(kLogicWeightMemBase + 512U);
  final_context.weightMemSize = 4096U;

  std::vector<domi::TaskDef> final_tasks;
  ASSERT_EQ(builder.GenerateTask(*node, final_context, final_tasks), SUCCESS);
  ASSERT_EQ(final_tasks.size(), 1U);
  EXPECT_EQ(ReadUint64Slot(final_tasks[0].kernel().args(), 0U), static_cast<uint64_t>(kLogicDataMemBase + 1024U));
  EXPECT_EQ(ReadUint64Slot(final_tasks[0].kernel().args(), 1U),
            static_cast<uint64_t>(kLogicWeightMemBase + 512U + 4096U));
  EXPECT_EQ(ReadUint64Slot(final_tasks[0].kernel().args(), 2U), static_cast<uint64_t>(kLogicDataMemBase + 2048U));
  EXPECT_EQ(ReadUint64Slot(final_tasks[0].kernel().args(), 3U), static_cast<uint64_t>(kLogicDataMemBase + 4096U));
  EXPECT_EQ(g_single_declare_workspace_count.load(), 1U);
}

TEST_F(UtestCustomOpsKernelInfoStore, GenerateTaskFailsWhenFinalWorkspaceLayoutDiffersFromPlan) {
  const std::string kTestOpType = "TestAnnotatedArgsWithWorkspaceMismatch_BuilderTest";
  auto creator = []() -> std::unique_ptr<BaseCustomOp> {
    return std::make_unique<MockAnnotatedArgsWithWorkspaceCustomOp>();
  };
  ASSERT_EQ(CustomOpFactory::RegisterCustomOpCreator(AscendString(kTestOpType.c_str()), creator), GRAPH_SUCCESS);

  ComputeGraphPtr graph;
  auto node = BuildStaticCustomNode(kTestOpType, graph);
  ASSERT_NE(node, nullptr);

  std::vector<domi::TaskDef> probe_tasks;
  ASSERT_EQ(GenerateTaskForNode(node, probe_tasks), SUCCESS);
  FinalizeCustomWorkspaceForDirectBuilderTest(node, 4096);
  node->GetOpDesc()->SetWorkspaceBytes({1024});

  CustomOpsKernelBuilder builder;
  RunContext final_context = {};
  final_context.dataMemBase = reinterpret_cast<uint8_t *>(kLogicDataMemBase);
  final_context.dataMemSize = 5120U;
  final_context.weightMemBase = reinterpret_cast<uint8_t *>(kLogicWeightMemBase);
  final_context.weightMemSize = 4096U;

  std::vector<domi::TaskDef> final_tasks;
  EXPECT_NE(builder.GenerateTask(*node, final_context, final_tasks), SUCCESS);
  EXPECT_TRUE(final_tasks.empty());
}

TEST_F(UtestCustomOpsKernelInfoStore, GenerateTaskFinalDeclareRefreshesConstInputWeightBase) {
  const std::string kTestOpType = "TestConstInputWorkspaceFinal_BuilderTest";
  auto creator = []() -> std::unique_ptr<BaseCustomOp> {
    return std::make_unique<MockAnnotatedArgsWithConstInputAndWorkspaceCustomOp>();
  };
  ASSERT_EQ(CustomOpFactory::RegisterCustomOpCreator(AscendString(kTestOpType.c_str()), creator), GRAPH_SUCCESS);

  ComputeGraphPtr graph;
  auto node = BuildStaticCustomNodeWithConstInput(kTestOpType, graph);
  ASSERT_NE(node, nullptr);

  std::vector<domi::TaskDef> probe_tasks;
  ASSERT_EQ(GenerateTaskForNode(node, probe_tasks), SUCCESS);
  FinalizeCustomWorkspaceForDirectBuilderTest(node, 4096);

  CustomOpsKernelBuilder builder;
  RunContext final_context = {};
  final_context.dataMemBase = reinterpret_cast<uint8_t *>(kLogicDataMemBase);
  final_context.dataMemSize = 4608U;
  final_context.weightMemBase = reinterpret_cast<uint8_t *>(kLogicWeightMemBase + 512U);
  final_context.weightMemSize = 4096U;

  std::vector<domi::TaskDef> final_tasks;
  EXPECT_EQ(builder.GenerateTask(*node, final_context, final_tasks), SUCCESS);

  ASSERT_EQ(final_tasks.size(), 1U);
  const auto &kernel = final_tasks[0].kernel();
  ASSERT_EQ(kernel.args().size(), 32);
  EXPECT_EQ(ReadUint64Slot(kernel.args(), 0U), static_cast<uint64_t>(kLogicDataMemBase + 1024U));
  EXPECT_EQ(ReadUint64Slot(kernel.args(), 1U), static_cast<uint64_t>(kLogicWeightMemBase + 512U + 4096U));
  EXPECT_EQ(ReadUint64Slot(kernel.args(), 2U), static_cast<uint64_t>(kLogicDataMemBase + 2048U));
  EXPECT_EQ(ReadUint64Slot(kernel.args(), 3U), static_cast<uint64_t>(kLogicDataMemBase + 4096U));
}

TEST_F(UtestCustomOpsKernelInfoStore, GenerateTaskFailsWhenCustomOpNotRegistered) {
  ExpectGenerateTaskFailed("TestUnregisteredCustomOp_BuilderTest");
}

TEST_F(UtestCustomOpsKernelInfoStore, GenerateTaskFailsWhenRegisteredOpIsNotAnnotatedArgsOp) {
  const std::string kTestOpType = "TestBaseOnlyCustomOp_BuilderTest";
  auto creator = []() -> std::unique_ptr<BaseCustomOp> { return std::make_unique<MockBaseOnlyCustomOp>(); };
  ASSERT_EQ(CustomOpFactory::RegisterCustomOpCreator(AscendString(kTestOpType.c_str()), creator), GRAPH_SUCCESS);

  ExpectGenerateTaskFailed(kTestOpType);
}

TEST_F(UtestCustomOpsKernelInfoStore, GenerateTaskSucceedsWhenRegisteredOpIsNotPortableOp) {
  const std::string kTestOpType = "TestAnnotatedArgsWithoutPortableOp_BuilderTest";
  auto creator = []() -> std::unique_ptr<BaseCustomOp> {
    return std::make_unique<MockAnnotatedArgsWithoutPortableCustomOp>();
  };
  ASSERT_EQ(CustomOpFactory::RegisterCustomOpCreator(AscendString(kTestOpType.c_str()), creator), GRAPH_SUCCESS);

  ComputeGraphPtr graph;
  auto node = BuildStaticCustomNode(kTestOpType, graph);
  ASSERT_NE(node, nullptr);

  std::vector<domi::TaskDef> tasks;
  EXPECT_EQ(GenerateTaskForNode(node, tasks), SUCCESS);

  ASSERT_EQ(tasks.size(), 1U);
  const auto &kernel = tasks[0].kernel();
  EXPECT_EQ(kernel.kernel_name(), "custom_without_portable_kernel");
  EXPECT_EQ(kernel.block_dim(), 2U);
  ASSERT_EQ(kernel.args().size(), 24);
  EXPECT_EQ(ReadUint64Slot(kernel.args(), 0U), static_cast<uint64_t>(kLogicDataMemBase + 1024U));
  EXPECT_EQ(ReadUint64Slot(kernel.args(), 1U), static_cast<uint64_t>(kLogicDataMemBase + 2048U));
  EXPECT_EQ(ReadUint64Slot(kernel.args(), 2U), 9U);
}

TEST_F(UtestCustomOpsKernelInfoStore, GenerateTaskFailsWhenDeclareDoesNotAddLaunch) {
  const std::string kTestOpType = "TestNoAnnotatedArgsCustomOp_BuilderTest";
  auto creator = []() -> std::unique_ptr<BaseCustomOp> {
    return std::make_unique<MockAnnotatedArgsWithoutAddLaunchCustomOp>();
  };
  ASSERT_EQ(CustomOpFactory::RegisterCustomOpCreator(AscendString(kTestOpType.c_str()), creator), GRAPH_SUCCESS);

  ExpectGenerateTaskFailedWithStatus(kTestOpType, INTERNAL_ERROR);
}

TEST_F(UtestCustomOpsKernelInfoStore, GenerateTaskFailsWithGraphFailedWhenDeclareLaunchArgsFails) {
  const std::string kTestOpType = "TestDeclareLaunchArgsFailsCustomOp_BuilderTest";
  auto creator = []() -> std::unique_ptr<BaseCustomOp> {
    return std::make_unique<MockAnnotatedArgsReturnsInternalErrorCustomOp>();
  };
  ASSERT_EQ(CustomOpFactory::RegisterCustomOpCreator(AscendString(kTestOpType.c_str()), creator), GRAPH_SUCCESS);

  ExpectGenerateTaskFailedWithStatus(kTestOpType, GRAPH_FAILED);
}

TEST_F(UtestCustomOpsKernelInfoStore, GenerateTaskSupportsUnknownTensorShapesInKnownGraph) {
  const std::string kTestOpType = "TestUnknownTensorShapesCustomOp_BuilderTest";
  auto creator = []() -> std::unique_ptr<BaseCustomOp> { return std::make_unique<MockAnnotatedArgsCustomOp>(); };
  ASSERT_EQ(CustomOpFactory::RegisterCustomOpCreator(AscendString(kTestOpType.c_str()), creator), GRAPH_SUCCESS);

  ComputeGraphPtr graph;
  auto node = BuildStaticCustomNode(kTestOpType, graph);
  ASSERT_NE(node, nullptr);
  auto op_desc = node->GetOpDesc();
  ASSERT_NE(op_desc, nullptr);
  op_desc->MutableInputDesc(0U)->SetShape(GeShape({-1, 16}));
  op_desc->MutableOutputDesc(0U)->SetShape(GeShape({-1, 16}));

  std::vector<domi::TaskDef> tasks;
  EXPECT_EQ(GenerateTaskForNode(node, tasks), SUCCESS);
  EXPECT_EQ(tasks.size(), 1U);
}

TEST_F(UtestCustomOpsKernelInfoStore, GenerateTaskOnUnknownGraphDoesNotValidateEagerCapability) {
  const std::string kTestOpType = "TestUnknownGraphCustomOp_BuilderTest";
  auto creator = []() -> std::unique_ptr<BaseCustomOp> { return std::make_unique<MockAnnotatedArgsCustomOp>(); };
  ASSERT_EQ(CustomOpFactory::RegisterCustomOpCreator(AscendString(kTestOpType.c_str()), creator), GRAPH_SUCCESS);

  ComputeGraphPtr graph;
  auto node = BuildStaticCustomNode(kTestOpType, graph);
  ASSERT_NE(node, nullptr);
  auto *const owner_graph = node->GetOwnerComputeGraphBarePtr();
  ASSERT_NE(owner_graph, nullptr);
  owner_graph->SetGraphUnknownFlag(true);

  std::vector<domi::TaskDef> tasks;
  EXPECT_EQ(GenerateTaskForNode(node, tasks), SUCCESS);

  ASSERT_EQ(tasks.size(), 1U);
  const auto &task = tasks[0];
  EXPECT_EQ(task.type(), static_cast<uint32_t>(ModelTaskType::MODEL_TASK_CUSTOM_KERNEL));
  EXPECT_EQ(task.kernel().context().op_index(), 7);
  EXPECT_TRUE(task.kernel().kernel_name().empty());
  EXPECT_TRUE(task.kernel().context().args_format().empty());
  int64_t task_args_mode = -1;
  ASSERT_TRUE(AttrUtils::GetInt(node->GetOpDesc(), ATTR_NAME_CUSTOM_TASK_ARGS_MODE, task_args_mode));
  EXPECT_EQ(task_args_mode, static_cast<int64_t>(CustomTaskArgsMode::kNone));
}

TEST_F(UtestCustomOpsKernelInfoStore, GenerateTaskOnUnknownGraphUsesBasicTaskForAnnotatedEagerOp) {
  const std::vector<std::pair<std::string, std::string>> cases = {
      {"KirinX90", "TestUnknownGraphAnnotatedEagerOp_Mobile"},
      {"Ascend910B", "TestUnknownGraphAnnotatedEagerOp_NonMobile"},
  };

  for (const auto &test_case : cases) {
    const std::string &soc_version = test_case.first;
    const std::string &op_type = test_case.second;
    GetThreadLocalContext().SetGraphOption({{ge::SOC_VERSION, soc_version}});
    g_annotated_eager_declare_count.store(0U);
    auto creator = []() -> std::unique_ptr<BaseCustomOp> {
      return std::make_unique<MockAnnotatedArgsAndEagerCustomOp>();
    };
    ASSERT_EQ(CustomOpFactory::RegisterCustomOpCreator(AscendString(op_type.c_str()), creator), GRAPH_SUCCESS);

    ComputeGraphPtr graph;
    auto node = BuildStaticCustomNode(op_type, graph);
    ASSERT_NE(node, nullptr);
    auto *const owner_graph = node->GetOwnerComputeGraphBarePtr();
    ASSERT_NE(owner_graph, nullptr);
    owner_graph->SetGraphUnknownFlag(true);

    std::vector<domi::TaskDef> tasks;
    EXPECT_EQ(GenerateTaskForNode(node, tasks), SUCCESS) << "soc_version=" << soc_version;
    EXPECT_EQ(g_annotated_eager_declare_count.load(), 0U) << "soc_version=" << soc_version;
    EXPECT_EQ(tasks.size(), 1U) << "soc_version=" << soc_version;
    if (tasks.size() != 1U) {
      continue;
    }
    const auto &task = tasks[0];
    EXPECT_EQ(task.stream_id(), 3U);
    EXPECT_EQ(task.type(), static_cast<uint32_t>(ModelTaskType::MODEL_TASK_CUSTOM_KERNEL));
    EXPECT_EQ(task.sqe_num(), 5U);
    EXPECT_EQ(task.kernel().context().op_index(), 7);
    EXPECT_TRUE(task.kernel().kernel_name().empty());
    EXPECT_TRUE(task.kernel().context().args_format().empty());
    int64_t task_args_mode = -1;
    ASSERT_TRUE(AttrUtils::GetInt(node->GetOpDesc(), ATTR_NAME_CUSTOM_TASK_ARGS_MODE, task_args_mode));
    EXPECT_EQ(task_args_mode, static_cast<int64_t>(CustomTaskArgsMode::kNone));
  }
}

TEST_F(UtestCustomOpsKernelInfoStore, GenerateTaskFailsWhenAnnotatedArgsRejectsParameters) {
  const std::vector<std::pair<std::string, InvalidAnnotatedArgsCase>> cases = {
      {"TestInvalidEmptyKernelNameCustomOp_BuilderTest", InvalidAnnotatedArgsCase::kEmptyName},
      {"TestInvalidEmptyKernelBinCustomOp_BuilderTest", InvalidAnnotatedArgsCase::kEmptyBin},
      {"TestInvalidZeroBlockDimCustomOp_BuilderTest", InvalidAnnotatedArgsCase::kZeroBlockDim},
      {"TestInvalidEmptyArgsCustomOp_BuilderTest", InvalidAnnotatedArgsCase::kEmptyArgs},
  };

  for (const auto &test_case : cases) {
    const std::string &op_type = test_case.first;
    const InvalidAnnotatedArgsCase invalid_case = test_case.second;
    auto creator = [invalid_case]() -> std::unique_ptr<BaseCustomOp> {
      return std::make_unique<MockInvalidAnnotatedArgsCustomOp>(invalid_case);
    };
    ASSERT_EQ(CustomOpFactory::RegisterCustomOpCreator(AscendString(op_type.c_str()), creator), GRAPH_SUCCESS);
    ExpectGenerateTaskFailedWithStatus(op_type, GRAPH_FAILED);
  }
}

TEST_F(UtestCustomOpsKernelInfoStore, CustomGraphOptimizerInitializeAndFinalize) {
  CustomGraphOptimizer optimizer;
  std::map<std::string, std::string> options;
  EXPECT_EQ(optimizer.Initialize(options, nullptr), SUCCESS);
  EXPECT_EQ(optimizer.Finalize(), SUCCESS);
}

TEST_F(UtestCustomOpsKernelInfoStore, CustomGraphOptimizerOptimizeOriginalGraph) {
  CustomGraphOptimizer optimizer;
  auto graph = std::make_shared<ComputeGraph>("test_graph");
  EXPECT_EQ(optimizer.OptimizeOriginalGraph(*graph), SUCCESS);
}

TEST_F(UtestCustomOpsKernelInfoStore, CustomGraphOptimizerOptimizeFusedGraph) {
  CustomGraphOptimizer optimizer;
  auto graph = std::make_shared<ComputeGraph>("test_graph");
  EXPECT_EQ(optimizer.OptimizeFusedGraph(*graph), SUCCESS);
}

TEST_F(UtestCustomOpsKernelInfoStore, CustomGraphOptimizerOptimizeWholeGraphEmptyGraph) {
  CustomGraphOptimizer optimizer;
  auto graph = std::make_shared<ComputeGraph>("empty_graph");
  EXPECT_EQ(optimizer.OptimizeWholeGraph(*graph), SUCCESS);
}

TEST_F(UtestCustomOpsKernelInfoStore, CustomGraphOptimizerOptimizeWholeGraphNonCustomOp) {
  CustomGraphOptimizer optimizer;
  auto graph = std::make_shared<ComputeGraph>("non_custom_graph");
  auto op_desc = std::make_shared<OpDesc>("non_custom_node", "NonCustomType");
  GeTensorDesc input_desc(GeShape({1, 16}), FORMAT_ND, DT_FLOAT16);
  GeTensorDesc output_desc(GeShape({1, 16}), FORMAT_ND, DT_FLOAT16);
  (void)op_desc->AddInputDesc("x", input_desc);
  (void)op_desc->AddOutputDesc("y", output_desc);
  ASSERT_NE(graph->AddNode(op_desc), nullptr);
  EXPECT_EQ(optimizer.OptimizeWholeGraph(*graph), SUCCESS);
}

TEST_F(UtestCustomOpsKernelInfoStore, CustomGraphOptimizerOptimizeWholeGraphBaseOnlyOp) {
  const std::string kTestOpType = "TestBaseOnlyOp_OptimizerTest";
  auto creator = []() -> std::unique_ptr<BaseCustomOp> { return std::make_unique<MockBaseOnlyCustomOp>(); };
  ASSERT_EQ(CustomOpFactory::RegisterCustomOpCreator(AscendString(kTestOpType.c_str()), creator), GRAPH_SUCCESS);

  auto graph = std::make_shared<ComputeGraph>("base_only_graph");
  auto op_desc = std::make_shared<OpDesc>("base_only_node", kTestOpType);
  GeTensorDesc input_desc(GeShape({1, 16}), FORMAT_ND, DT_FLOAT16);
  GeTensorDesc output_desc(GeShape({1, 16}), FORMAT_ND, DT_FLOAT16);
  (void)op_desc->AddInputDesc("x", input_desc);
  (void)op_desc->AddOutputDesc("y", output_desc);
  ASSERT_NE(graph->AddNode(op_desc), nullptr);

  CustomGraphOptimizer optimizer;
  EXPECT_EQ(optimizer.OptimizeWholeGraph(*graph), SUCCESS);
}

TEST_F(UtestCustomOpsKernelInfoStore, CustomGraphOptimizerGetAttributes) {
  CustomGraphOptimizer optimizer;
  GraphOptimizerAttribute attrs;
  EXPECT_EQ(optimizer.GetAttributes(attrs), SUCCESS);
  EXPECT_EQ(attrs.engineName, "DNN_VM_CUSTOM");
}

TEST_F(UtestCustomOpsKernelInfoStore, CustomOpsKernelBuilderCalcOpRunningParamKnownShape) {
  const std::string kTestOpType = "TestCalcOpRunningParam_KnownShape";
  auto creator = []() -> std::unique_ptr<BaseCustomOp> { return std::make_unique<MockCompilableCustomOp>(); };
  ASSERT_EQ(CustomOpFactory::RegisterCustomOpCreator(AscendString(kTestOpType.c_str()), creator), GRAPH_SUCCESS);

  auto graph = std::make_shared<ComputeGraph>("calc_param_graph");
  auto op_desc = std::make_shared<OpDesc>("calc_param_node", kTestOpType);
  op_desc->AppendIrInput("x", kIrInputRequired);
  op_desc->AppendIrOutput("y", kIrOutputRequired);
  GeTensorDesc input_desc(GeShape({1, 16}), FORMAT_ND, DT_FLOAT16);
  GeTensorDesc output_desc(GeShape({1, 16}), FORMAT_ND, DT_FLOAT16);
  (void)op_desc->AddInputDesc("x", input_desc);
  (void)op_desc->AddOutputDesc("y", output_desc);
  auto node = graph->AddNode(op_desc);
  ASSERT_NE(node, nullptr);

  CustomOpsKernelBuilder builder;
  EXPECT_EQ(builder.CalcOpRunningParam(*node), SUCCESS);
}

TEST_F(UtestCustomOpsKernelInfoStore, CustomOpsKernelBuilderCalcOpRunningParamUnknownShape) {
  const std::string kTestOpType = "TestCalcOpRunningParam_UnknownShape";
  auto creator = []() -> std::unique_ptr<BaseCustomOp> { return std::make_unique<MockCompilableCustomOp>(); };
  ASSERT_EQ(CustomOpFactory::RegisterCustomOpCreator(AscendString(kTestOpType.c_str()), creator), GRAPH_SUCCESS);

  auto graph = std::make_shared<ComputeGraph>("calc_param_graph_unknown");
  auto op_desc = std::make_shared<OpDesc>("calc_param_node_unknown", kTestOpType);
  op_desc->AppendIrInput("x", kIrInputRequired);
  op_desc->AppendIrOutput("y", kIrOutputRequired);
  GeTensorDesc input_desc(GeShape({-1, 16}), FORMAT_ND, DT_FLOAT16);
  GeTensorDesc output_desc(GeShape({-1, 16}), FORMAT_ND, DT_FLOAT16);
  (void)op_desc->AddInputDesc("x", input_desc);
  (void)op_desc->AddOutputDesc("y", output_desc);
  auto node = graph->AddNode(op_desc);
  ASSERT_NE(node, nullptr);

  CustomOpsKernelBuilder builder;
  EXPECT_EQ(builder.CalcOpRunningParam(*node), SUCCESS);
}

TEST_F(UtestCustomOpsKernelInfoStore, CustomOpsKernelBuilderInitializeAndFinalize) {
  CustomOpsKernelBuilder builder;
  std::map<std::string, std::string> options;
  EXPECT_EQ(builder.Initialize(options), SUCCESS);
  EXPECT_EQ(builder.Finalize(), SUCCESS);
}
}  // namespace custom
}  // namespace ge
