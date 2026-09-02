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

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "common/plugin/ge_make_unique_util.h"
#include "exe_graph/lowering/kernel_run_context_builder.h"
#include "exe_graph/runtime/infer_symbol_shape_context.h"
#include "graph/compute_graph.h"
#include "graph/optimize/symbolic/infer_symbolic_shape/symbolic_infer_util.h"
#include "register/op_impl_registry.h"
#include "faker/space_registry_faker.h"

namespace ge {
namespace {

template <typename T>
std::vector<void *> GetVoidPtr(const std::vector<std::unique_ptr<T>> &holders) {
  std::vector<void *> result;
  result.reserve(holders.size());
  for (const auto &holder : holders) {
    result.push_back(holder.get());
  }
  return result;
}

class InferContextBuilder {
 public:
  InferContextBuilder(std::string op_type, std::string op_name)
      : op_type_(std::move(op_type)), op_name_(std::move(op_name)) {}

  ~InferContextBuilder() {
    Destroy();
  }

  InferContextBuilder &Input(const gert::SymbolShape &shape) {
    auto holder = ge::ComGraphMakeUnique<gert::SymbolTensor>();
    for (size_t i = 0U; i < shape.GetDimNum(); ++i) {
      holder->MutableOriginSymbolShape().AppendDim(shape.GetDim(i));
    }
    inputs_.emplace_back(std::move(holder));
    return *this;
  }

  InferContextBuilder &OptionalInput(const gert::SymbolShape *shape) {
    if (shape == nullptr) {
      inputs_.emplace_back(nullptr);
      return *this;
    }
    return Input(*shape);
  }

  InferContextBuilder &Outputs(size_t count) {
    outputs_.reserve(count);
    for (size_t i = 0U; i < count; ++i) {
      outputs_.emplace_back(ge::ComGraphMakeUnique<gert::SymbolShape>());
    }
    return *this;
  }

  OpDescPtr GetOrCreateOpDesc() {
    if (op_desc_ == nullptr) {
      op_desc_ = ComGraphMakeShared<OpDesc>(op_name_, op_type_);
    }
    return op_desc_;
  }

  gert::InferSymbolShapeContext *Build() {
    auto holder = gert::KernelRunContextBuilder()
                      .Inputs(GetVoidPtr<gert::SymbolTensor>(inputs_))
                      .Outputs(GetVoidPtr<gert::SymbolShape>(outputs_))
                      .Build(op_desc_);
    context_holder_ = std::move(holder);
    return reinterpret_cast<gert::InferSymbolShapeContext *>(context_holder_.context_);
  }

  void Destroy() {
    inputs_.clear();
    outputs_.clear();
    op_desc_.reset();
    context_holder_.value_holder_.clear();
    context_holder_.context_holder_.reset();
    context_holder_.compute_node_extend_holder_.reset();
    context_holder_.buffer_pool_ = gert::bg::BufferPool();
    context_holder_.context_ = nullptr;
  }

 private:
  std::string op_type_;
  std::string op_name_;
  OpDescPtr op_desc_;
  std::vector<std::unique_ptr<gert::SymbolTensor>> inputs_;
  std::vector<std::unique_ptr<gert::SymbolShape>> outputs_;
  gert::KernelContextHolder context_holder_;
};

gert::OpImplKernelRegistry::InferSymbolShapeKernelFunc GetInferFunc(const char *op_type) {
  const auto &impl = gert::OpImplInferSymbolShapeRegistry::GetInstance().GetOpImpl(op_type);
  return impl == nullptr ? nullptr : impl->infer_symbol_shape;
}

class SymbolicShapeInferCoverageST : public testing::Test {
 protected:
  static void SetUpTestSuite() {
    gert::SpaceRegistryFaker::CreateDefaultSpaceRegistry();
  }
};

void SetIndexByTensorDesc(const OpDescPtr &desc, size_t index_count) {
  desc->AppendIrAttrName("indices_mask");
  desc->AppendIrInput("x", kIrInputRequired);
  desc->AppendIrInput("indices", kIrInputDynamic);
  desc->AddInputDesc(GeTensorDesc());
  desc->AddDynamicInputDesc("indices", index_count, true);
}

TEST_F(SymbolicShapeInferCoverageST, IndexByTensorPaths) {
  const auto infer = GetInferFunc("IndexByTensor");
  ASSERT_NE(infer, nullptr);
  const auto x0 = Symbol("x0");
  const auto x1 = Symbol("x1");
  const auto x2 = Symbol("x2");
  const auto x3 = Symbol("x3");

  InferContextBuilder contiguous("IndexByTensor", "index_contiguous");
  auto desc = contiguous.GetOrCreateOpDesc();
  SetIndexByTensorDesc(desc, 2U);
  AttrUtils::SetListInt(desc, "indices_mask", {0, 1, 1});
  contiguous.Input(gert::SymbolShape({x0, x1, x2, x3}))
      .Input(gert::SymbolShape({Symbol("i0"), Symbol("i1")}))
      .Input(gert::SymbolShape({Symbol("i1")}))
      .Outputs(1U);
  auto *context = contiguous.Build();
  ASSERT_EQ(infer(context), GRAPH_SUCCESS);
  EXPECT_EQ(context->GetOutputSymbolShape(0)->GetDims(), (std::vector<Expression>{x0, Symbol("i0"), Symbol("i1"), x3}));

  InferContextBuilder non_contiguous("IndexByTensor", "index_non_contiguous");
  desc = non_contiguous.GetOrCreateOpDesc();
  SetIndexByTensorDesc(desc, 2U);
  AttrUtils::SetListInt(desc, "indices_mask", {1, 0, 1});
  non_contiguous.Input(gert::SymbolShape({x0, x1, x2}))
      .Input(gert::SymbolShape({Symbol("j0")}))
      .Input(gert::SymbolShape({Symbol(1)}))
      .Outputs(1U);
  context = non_contiguous.Build();
  ASSERT_EQ(infer(context), GRAPH_SUCCESS);
  EXPECT_EQ(context->GetOutputSymbolShape(0)->GetDims(), (std::vector<Expression>{Symbol("j0"), x1}));

  InferContextBuilder no_index("IndexByTensor", "index_identity");
  desc = no_index.GetOrCreateOpDesc();
  SetIndexByTensorDesc(desc, 0U);
  AttrUtils::SetListInt(desc, "indices_mask", {0, 0, 0});
  no_index.Input(gert::SymbolShape({x0, x1, x2})).Outputs(1U);
  context = no_index.Build();
  ASSERT_EQ(infer(context), GRAPH_SUCCESS);
  EXPECT_EQ(context->GetOutputSymbolShape(0)->GetDims(), (std::vector<Expression>{x0, x1, x2}));

  InferContextBuilder empty_mask("IndexByTensor", "index_empty");
  desc = empty_mask.GetOrCreateOpDesc();
  SetIndexByTensorDesc(desc, 0U);
  AttrUtils::SetListInt(desc, "indices_mask", {});
  empty_mask.Input(gert::SymbolShape({x0, x1})).Outputs(1U);
  context = empty_mask.Build();
  ASSERT_EQ(infer(context), GRAPH_SUCCESS);

  InferContextBuilder invalid("IndexByTensor", "index_invalid");
  desc = invalid.GetOrCreateOpDesc();
  SetIndexByTensorDesc(desc, 1U);
  AttrUtils::SetListInt(desc, "indices_mask", {1, 0, 0});
  invalid.Input(gert::SymbolShape({x0, x1})).Input(gert::SymbolShape({Symbol("k0")})).Outputs(1U);
  context = invalid.Build();
  EXPECT_EQ(infer(context), UNSUPPORTED);

  InferContextBuilder mismatched_indices("IndexByTensor", "index_mismatched_indices");
  desc = mismatched_indices.GetOrCreateOpDesc();
  SetIndexByTensorDesc(desc, 2U);
  AttrUtils::SetListInt(desc, "indices_mask", {1, 0});
  mismatched_indices.Input(gert::SymbolShape({x0, x1}))
      .Input(gert::SymbolShape({Symbol("m0")}))
      .Input(gert::SymbolShape({Symbol("m1")}))
      .Outputs(1U);
  context = mismatched_indices.Build();
  EXPECT_EQ(infer(context), UNSUPPORTED);

  InferContextBuilder broadcast_failure("IndexByTensor", "index_broadcast_failure");
  desc = broadcast_failure.GetOrCreateOpDesc();
  SetIndexByTensorDesc(desc, 2U);
  AttrUtils::SetListInt(desc, "indices_mask", {1, 1});
  broadcast_failure.Input(gert::SymbolShape({x0, x1}))
      .Input(gert::SymbolShape({Symbol("b0")}))
      .Input(gert::SymbolShape({Symbol("b1")}))
      .Outputs(1U);
  context = broadcast_failure.Build();
  EXPECT_EQ(infer(context), UNSUPPORTED);
}

void SetMoeV2Attrs(const OpDescPtr &desc, int64_t drop_pad_mode, int64_t count_flag, bool before_capacity) {
  for (const auto &name :
       {"active_num", "expert_capacity", "expert_num", "drop_pad_mode", "count_flag", "before_capacity_flag"}) {
    desc->AppendIrAttrName(name);
  }
  AttrUtils::SetInt(desc, "active_num", 8);
  AttrUtils::SetInt(desc, "expert_capacity", 4);
  AttrUtils::SetInt(desc, "expert_num", 2);
  AttrUtils::SetInt(desc, "drop_pad_mode", drop_pad_mode);
  AttrUtils::SetInt(desc, "count_flag", count_flag);
  AttrUtils::SetBool(desc, "before_capacity_flag", before_capacity);
}

void SetMoeV3Attrs(const OpDescPtr &desc, int64_t drop_pad_mode, int64_t token_type, bool count_flag,
                   int64_t quant_mode) {
  for (const auto &name : {"active_num", "expert_capacity", "expert_num", "drop_pad_mode", "expert_token_num_type",
                           "count_flag", "quant_mode", "expert_range"}) {
    desc->AppendIrAttrName(name);
  }
  AttrUtils::SetInt(desc, "active_num", 8);
  AttrUtils::SetInt(desc, "expert_capacity", 4);
  AttrUtils::SetInt(desc, "expert_num", 2);
  AttrUtils::SetInt(desc, "drop_pad_mode", drop_pad_mode);
  AttrUtils::SetInt(desc, "expert_token_num_type", token_type);
  AttrUtils::SetBool(desc, "count_flag", count_flag);
  AttrUtils::SetInt(desc, "quant_mode", quant_mode);
  AttrUtils::SetListInt(desc, "expert_range", {0, 2});
}

TEST_F(SymbolicShapeInferCoverageST, MoeRoutingPaths) {
  const auto n = Symbol("n");
  const auto h = Symbol("h");
  const auto k = Symbol("k");

  const auto infer_v2 = GetInferFunc("MoeInitRoutingV2");
  ASSERT_NE(infer_v2, nullptr);
  InferContextBuilder v2("MoeInitRoutingV2", "moe_v2");
  auto desc = v2.GetOrCreateOpDesc();
  SetMoeV2Attrs(desc, 0, 1, false);
  v2.Input(gert::SymbolShape({n, h})).Input(gert::SymbolShape({n, k})).Outputs(3U);
  auto *context = v2.Build();
  ASSERT_EQ(infer_v2(context), GRAPH_SUCCESS);

  InferContextBuilder v2_drop("MoeInitRoutingV2", "moe_v2_drop");
  desc = v2_drop.GetOrCreateOpDesc();
  SetMoeV2Attrs(desc, 1, 0, true);
  v2_drop.Input(gert::SymbolShape({n, h})).Input(gert::SymbolShape({n})).Outputs(4U);
  context = v2_drop.Build();
  ASSERT_EQ(infer_v2(context), GRAPH_SUCCESS);

  const auto infer_v3 = GetInferFunc("MoeInitRoutingV3");
  ASSERT_NE(infer_v3, nullptr);
  for (const int64_t quant_mode : {2, 4, 9, 11, 8, 1, 0}) {
    InferContextBuilder v3("MoeInitRoutingV3", "moe_v3");
    desc = v3.GetOrCreateOpDesc();
    SetMoeV3Attrs(desc, quant_mode == 1 ? 1 : 0, quant_mode == 11 ? 2 : 0, quant_mode == 11, quant_mode);
    v3.Input(gert::SymbolShape({n, h})).Input(gert::SymbolShape({n, k})).Outputs(4U);
    context = v3.Build();
    EXPECT_EQ(infer_v3(context), GRAPH_SUCCESS);
  }

  InferContextBuilder v3_invalid("MoeInitRoutingV3", "moe_v3_invalid");
  desc = v3_invalid.GetOrCreateOpDesc();
  SetMoeV3Attrs(desc, 0, 0, false, 99);
  v3_invalid.Input(gert::SymbolShape({n, h})).Input(gert::SymbolShape({n, k})).Outputs(4U);
  context = v3_invalid.Build();
  EXPECT_EQ(infer_v3(context), UNSUPPORTED);
}

TEST_F(SymbolicShapeInferCoverageST, MoeFinalizeRoutingBasicPaths) {
  const auto infer = GetInferFunc("MoeFinalizeRoutingV2");
  ASSERT_NE(infer, nullptr);
  const auto rows = Symbol("rows");
  const auto h = Symbol("h");
  const auto k = Symbol("k");

  InferContextBuilder dropless("MoeFinalizeRoutingV2", "moe_finalize_dropless");
  auto desc = dropless.GetOrCreateOpDesc();
  desc->AppendIrAttrName("drop_pad_mode");
  AttrUtils::SetInt(desc, "drop_pad_mode", 0);
  dropless.Input(gert::SymbolShape({rows, h})).Input(gert::SymbolShape({rows})).Outputs(1U);
  auto *context = dropless.Build();
  ASSERT_EQ(infer(context), GRAPH_SUCCESS);

  InferContextBuilder drop_pad("MoeFinalizeRoutingV2", "moe_finalize_drop");
  desc = drop_pad.GetOrCreateOpDesc();
  desc->AppendIrAttrName("drop_pad_mode");
  AttrUtils::SetInt(desc, "drop_pad_mode", 1);
  drop_pad.Input(gert::SymbolShape({Symbol("experts"), Symbol("capacity"), h}))
      .Input(gert::SymbolShape({rows}))
      .Outputs(1U);
  context = drop_pad.Build();
  ASSERT_EQ(infer(context), GRAPH_SUCCESS);

  // Exercise optional residual/scales/index inputs used by the dropless finalize path.
  InferContextBuilder optional("MoeFinalizeRoutingV2", "moe_finalize_optional");
  desc = optional.GetOrCreateOpDesc();
  desc->AppendIrAttrName("drop_pad_mode");
  desc->AppendIrAttrName("unused1");
  desc->AppendIrAttrName("unused2");
  desc->AppendIrAttrName("unused3");
  desc->AppendIrAttrName("k");
  AttrUtils::SetInt(desc, "drop_pad_mode", 0);
  AttrUtils::SetInt(desc, "k", 1);
  const gert::SymbolShape residual_shape({rows, h});
  const gert::SymbolShape scales_shape({rows, k});
  optional.Input(gert::SymbolShape({rows, h}))
      .Input(gert::SymbolShape({rows}))
      .OptionalInput(&residual_shape)
      .OptionalInput(nullptr)
      .OptionalInput(nullptr)
      .OptionalInput(&scales_shape)
      .OptionalInput(&scales_shape)
      .Outputs(1U);
  context = optional.Build();
  ASSERT_EQ(infer(context), GRAPH_SUCCESS);
}

}  // namespace
}  // namespace ge
