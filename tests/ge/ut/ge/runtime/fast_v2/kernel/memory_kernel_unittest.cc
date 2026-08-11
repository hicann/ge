/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "kernel/memory/memory_kernel.h"
#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <limits>
#include "register/kernel_registry.h"
#include "faker/kernel_run_context_facker.h"
#include "faker/node_faker.h"
#include "faker/multi_stream_allocator_faker.h"
#include "kernel/memory/caching_mem_allocator.h"
#include "core/debug/kernel_tracing.h"
#include "exe_graph/runtime/continuous_vector.h"
#include "exe_graph/runtime/runtime_tensor.h"
#include "kernel/memory/single_stream_l2_allocator.h"
#include "kernel/memory/multi_stream_mem_block.h"
#include "kernel/memory/host_mem_allocator.h"
#include "kernel/memory/ffts_mem_allocator.h"
#include "exe_graph/runtime/gert_mem_allocator.h"
#include "exe_graph/runtime/gert_tensor_data.h"
#include "runtime/mem_allocator.h"
#include "exe_graph/runtime/tensor_data.h"
#include "exe_graph/runtime/allocator.h"
#include "runtime/gert_api.h"
#include "graph/types.h"
#include "kernel/memory/rts_caching_mem_allocator.h"
#include "kernel/memory/multi_stream_l2_allocator.h"
#include "rt_external.h"
#include "checker/memory_profiling_log_matcher.h"
#include "stub/gert_runtime_stub.h"
#include "core/utils/rt2_tensor_utils.h"
#include "depends/runtime/src/runtime_stub.h"

namespace gert {
namespace kernel {
ge::graphStatus AllocBatchHbm(KernelContext *context);
ge::graphStatus AllocHbmMem(KernelContext *context);
}  // namespace kernel
namespace {
class MockRtStreamSync : public ge::RuntimeStub {
 public:
  MOCK_METHOD2(rtStreamSynchronizeWithTimeout, int32_t(rtStream_t stm, int32_t timeout));
};

class CountingMemBlock : public memory::MultiStreamMemBlock {
 public:
  CountingMemBlock(size_t &free_count, size_t &duplicate_free_count, bool has_address)
      : free_count_(free_count), duplicate_free_count_(duplicate_free_count), has_address_(has_address) {}

  void Free(int64_t stream_id) override {
    (void)stream_id;
    if (freed_) {
      ++duplicate_free_count_;
      return;
    }
    ++free_count_;
    freed_ = true;
  }

  void *GetAddr() override {
    return has_address_ ? this : nullptr;
  }

 private:
  size_t &free_count_;
  size_t &duplicate_free_count_;
  bool has_address_;
  bool freed_ = false;
};

class CountingGertAllocator : public GertAllocator {
 public:
  CountingGertAllocator(size_t fail_index = std::numeric_limits<size_t>::max(),
                        size_t null_address_index = std::numeric_limits<size_t>::max())
      : GertAllocator(0, kOnDeviceHbm), fail_index_(fail_index), null_address_index_(null_address_index) {}

  GertMemBlock *Malloc(size_t size) override {
    (void)size;
    const auto index = alloc_count_++;
    if (index == fail_index_) {
      return nullptr;
    }
    blocks_.emplace_back(
        std::make_unique<CountingMemBlock>(free_count_, duplicate_free_count_, index != null_address_index_));
    return blocks_.back().get();
  }

  GertTensorData MallocTensorData(size_t size) override {
    auto block = reinterpret_cast<memory::MultiStreamMemBlock *>(Malloc(size));
    return TensorUtils::ToGertTensorData(block, GetPlacement(), GetStreamId());
  }

  TensorData MallocTensorDataFromL1(size_t size) override {
    (void)size;
    return {};
  }

  void Free(GertMemBlock *block) override {
    if (block != nullptr) {
      block->Free(GetStreamId());
    }
  }

  ge::graphStatus FreeAt(int64_t stream_id, GertMemBlock *block) override {
    if (block != nullptr) {
      block->Free(stream_id);
    }
    return ge::GRAPH_SUCCESS;
  }

  ge::graphStatus ShareFromTensorData(const TensorData &td, GertTensorData &gtd) override {
    (void)td;
    (void)gtd;
    return ge::GRAPH_SUCCESS;
  }

  int64_t GetStreamNum() override {
    return 1;
  }

  ge::graphStatus SetL1Allocator(ge::Allocator *allocator) override {
    (void)allocator;
    return ge::GRAPH_SUCCESS;
  }

  size_t GetAllocCount() const {
    return alloc_count_;
  }

  size_t GetFreeCount() const {
    return free_count_;
  }

  size_t GetDuplicateFreeCount() const {
    return duplicate_free_count_;
  }

 private:
  size_t fail_index_;
  size_t null_address_index_;
  size_t alloc_count_ = 0U;
  size_t free_count_ = 0U;
  size_t duplicate_free_count_ = 0U;
  std::vector<std::unique_ptr<CountingMemBlock>> blocks_;
};

KernelRegistry &registry = KernelRegistry::GetInstance();
std::unique_ptr<gert::Allocators> CreateDefaultAllocators() {
  std::shared_ptr<ge::Allocator> device_allocator(AllocatorFactory::Create(kOnDeviceHbm).release());
  std::shared_ptr<ge::Allocator> host_allocator(AllocatorFactory::Create(kOnHost).release());
  if ((device_allocator == nullptr) || (host_allocator == nullptr)) {
    GELOGE(ge::PARAM_INVALID, "device_allocator is nullptr or host_allocator is nullptr");
  }
  std::unique_ptr<Allocators> allocators = std::make_unique<Allocators>();
  for (size_t i = 0U; i < static_cast<size_t>(kTensorPlacementEnd); ++i) {
    for (size_t j = 0U; j < static_cast<size_t>(AllocatorUsage::kEnd); ++j) {
      if (i == static_cast<size_t>(kOnDeviceHbm)) {
        allocators->SetAllocator(static_cast<TensorPlacement>(i), j, device_allocator);
      } else if (i == static_cast<size_t>(kOnHost) || i == static_cast<size_t>(kFollowing)) {
        allocators->SetAllocator(static_cast<TensorPlacement>(i), j, host_allocator);
      } else {
        GELOGE(ge::PARAM_INVALID, "Unsupported placement %zu to set allocator", i);
      }
    }
  }
  return allocators;
}
}  // namespace
class MemoryKernelUT : public testing::Test {
 public:
  KernelRunContextHolder Alloc(memory::SingleStreamL2Allocator &gert_mem_allocator) {
    auto alloc_context_holder =
        KernelRunContextFaker().KernelIONum(2, 1).Inputs({&gert_mem_allocator, reinterpret_cast<void *>(0x64)}).Build();
    auto alloc_context = alloc_context_holder.GetContext<KernelContext>();

    registry.FindKernelFuncs("AllocMemHbm")->outputs_creator(nullptr, alloc_context);
    GertRuntimeStub runtime_stub;
    runtime_stub.GetSlogStub().NoConsoleOut().SetLevelInfo();
    kernel::AllocHbmMem(alloc_context);
    return alloc_context_holder;
  }
};
TEST_F(MemoryKernelUT, AllocCreateOutputOk) {
  auto run_context = BuildKernelRunContext(1, 1);
  ASSERT_NE(registry.FindKernelFuncs("AllocMemHbm"), nullptr);
  ASSERT_EQ(registry.FindKernelFuncs("AllocMemHbm")->outputs_creator(nullptr, run_context), ge::GRAPH_SUCCESS);
  ASSERT_NE(run_context.value_holder[1].GetValue<void *>(), nullptr);
  run_context.value_holder[1].Set(nullptr, nullptr);
}
TEST_F(MemoryKernelUT, AllocFreeSuccess) {
  memory::CachingMemAllocator allocator(0, RT_MEMORY_HBM);
  memory::SingleStreamL2Allocator single_stream_l2_allocator(&allocator);

  ASSERT_NE(registry.FindKernelFuncs("AllocMemHbm"), nullptr);
  ASSERT_NE(registry.FindKernelFuncs("AllocMemHbm")->outputs_creator, nullptr);
  ASSERT_NE(registry.FindKernelFuncs("AllocMemHbm")->run_func, nullptr);
  ASSERT_NE(registry.FindKernelFuncs("FreeMemHbm"), nullptr);
  ASSERT_NE(registry.FindKernelFuncs("FreeMemHbm")->run_func, nullptr);

  // alloc
  auto alloc_context_holder = KernelRunContextFaker()
                                  .KernelIONum(2, 1)
                                  .Inputs({&single_stream_l2_allocator, reinterpret_cast<void *>(0x64)})
                                  .Build();
  auto alloc_context = alloc_context_holder.GetContext<KernelContext>();

  ASSERT_EQ(registry.FindKernelFuncs("AllocMemHbm")->outputs_creator(nullptr, alloc_context), ge::GRAPH_SUCCESS);
  GertRuntimeStub runtime_stub;
  runtime_stub.GetSlogStub().NoConsoleOut().SetLevelInfo();
  ASSERT_EQ(kernel::AllocHbmMem(alloc_context), ge::GRAPH_SUCCESS);
  ASSERT_TRUE(runtime_stub.GetSlogStub().FindInfoLogRegex(kAllocRe, {{2, "0"}}) >= 0);

  auto tensor_data = alloc_context->GetOutputPointer<GertTensorData>(0);
  ASSERT_NE(tensor_data, nullptr);
  ASSERT_NE(tensor_data->GetAddr(), nullptr);

  // free
  auto free_context_holder = KernelRunContextFaker().KernelIONum(1, 0).Inputs({tensor_data}).Build();
  auto free_context = free_context_holder.GetContext<KernelContext>();

  runtime_stub.GetSlogStub().Clear();
  ASSERT_EQ(registry.FindKernelFuncs("FreeMemHbm")->run_func(free_context), ge::GRAPH_SUCCESS);
  ASSERT_TRUE(runtime_stub.GetSlogStub().FindInfoLogRegex(kFreeRe) >= 0);

  allocator.Finalize();
  alloc_context_holder.FreeAll();
  free_context_holder.FreeAll();
}

TEST_F(MemoryKernelUT, AllocSizeZeroSuccess) {
  memory::CachingMemAllocator allocator(0, RT_MEMORY_HBM);
  memory::SingleStreamL2Allocator single_stream_l2_allocator(&allocator);

  auto alloc_context_holder = KernelRunContextFaker()
                                  .KernelIONum(2, 1)
                                  .Inputs({&single_stream_l2_allocator, reinterpret_cast<void *>(0UL)})
                                  .Build();
  auto alloc_context = alloc_context_holder.GetContext<KernelContext>();

  ASSERT_NE(registry.FindKernelFuncs("AllocMemHbm"), nullptr);
  ASSERT_NE(registry.FindKernelFuncs("AllocMemHbm")->outputs_creator, nullptr);
  ASSERT_NE(registry.FindKernelFuncs("AllocMemHbm")->run_func, nullptr);
  ASSERT_EQ(registry.FindKernelFuncs("AllocMemHbm")->outputs_creator(nullptr, alloc_context), ge::GRAPH_SUCCESS);
  ASSERT_EQ(kernel::AllocHbmMem(alloc_context), ge::GRAPH_SUCCESS);

  auto tensor_data = alloc_context->GetOutputPointer<GertTensorData>(0);
  ASSERT_NE(tensor_data, nullptr);
  EXPECT_EQ(tensor_data->GetAddr(), nullptr);
  EXPECT_EQ(tensor_data->GetSize(), 0U);
  EXPECT_EQ(tensor_data->GetPlacement(), kOnDeviceHbm);

  allocator.Finalize();
  alloc_context_holder.FreeAll();
}

TEST_F(MemoryKernelUT, FreeHoldAddrSuccess) {
  memory::CachingMemAllocator allocator(0, RT_MEMORY_HBM);
  memory::SingleStreamL2Allocator single_stream_l2_allocator(&allocator);
  ASSERT_NE(registry.FindKernelFuncs("FreeMemHbmHoldAddr"), nullptr);
  ASSERT_NE(registry.FindKernelFuncs("FreeMemHbmHoldAddr")->run_func, nullptr);

  auto tensor_data = single_stream_l2_allocator.MallocTensorData(1);
  auto addr = tensor_data.GetAddr();

  auto free_context_holder = KernelRunContextFaker().KernelIONum(1, 0).Inputs({&tensor_data}).Build();
  auto free_context = free_context_holder.GetContext<KernelContext>();

  ASSERT_EQ(registry.FindKernelFuncs("FreeMemHbmHoldAddr")->run_func(free_context), ge::GRAPH_SUCCESS);
  ASSERT_EQ(tensor_data.GetAddr(), addr);

  allocator.Finalize();
  free_context_holder.FreeAll();
}

TEST_F(MemoryKernelUT, BatchAllocCreateOutputOk) {
  auto run_context = BuildKernelRunContext(2, 1);
  ASSERT_EQ(registry.FindKernelFuncs("AllocBatchHbm")->outputs_creator(nullptr, run_context), ge::GRAPH_SUCCESS);
  run_context.FreeValue(2);
}

TEST_F(MemoryKernelUT, AllocCopyFlowHbmCreateOutputUsesCopyFlowCount) {
  auto funcs = registry.FindKernelFuncs("AllocCopyFlowHbm");
  ASSERT_NE(funcs, nullptr);
  ASSERT_NE(funcs->outputs_creator, nullptr);
  ASSERT_NE(funcs->run_func, nullptr);
  memory::CachingMemAllocator allocator(0, RT_MEMORY_HBM);
  memory::SingleStreamL2Allocator single_stream_l2_allocator(&allocator);
  for (const int64_t copy_flow_count : {1, 16, 17, 32}) {
    auto sizes_holder = ContinuousVector::Create<size_t>(copy_flow_count);
    auto sizes = reinterpret_cast<TypedContinuousVector<size_t> *>(sizes_holder.get());
    sizes->SetSize(copy_flow_count);
    for (size_t i = 0U; i < static_cast<size_t>(copy_flow_count); ++i) {
      sizes->MutableData()[i] = 0U;
    }
    auto context_holder =
        KernelRunContextFaker().KernelIONum(2, 1).Inputs({&single_stream_l2_allocator, sizes_holder.get()}).Build();
    auto run_context = context_holder.GetContext<KernelContext>();
    auto graph = std::make_shared<ge::ExecuteGraph>("copy_flow_alloc_test");
    auto node = FastNodeFaker(graph)
                    .NameAndType("alloc_copy_flow_hbm", "AllocCopyFlowHbm")
                    .Attr("copy_flow_count", copy_flow_count)
                    .Build();
    ASSERT_NE(node, nullptr);

    ASSERT_EQ(funcs->outputs_creator(node, run_context), ge::GRAPH_SUCCESS);
    ASSERT_EQ(funcs->run_func(run_context), ge::GRAPH_SUCCESS);
    auto addresses = run_context->GetOutputPointer<TypedContinuousVector<GertTensorData *>>(0U);
    ASSERT_NE(addresses, nullptr);
    EXPECT_EQ(addresses->GetCapacity(), static_cast<size_t>(copy_flow_count));
    EXPECT_EQ(addresses->GetSize(), static_cast<size_t>(copy_flow_count));
    context_holder.FreeAll();
  }
  allocator.Finalize();
}

TEST_F(MemoryKernelUT, AllocCopyFlowHbmHandlesZeroSizeAndRepeatedExecution) {
  constexpr int64_t kCopyFlowCount = 2;
  CountingGertAllocator allocator;
  auto sizes_holder = ContinuousVector::Create<size_t>(kCopyFlowCount);
  auto sizes = reinterpret_cast<TypedContinuousVector<size_t> *>(sizes_holder.get());
  sizes->SetSize(kCopyFlowCount);
  sizes->MutableData()[0] = 128U;
  sizes->MutableData()[1] = 128U;

  auto context_holder = KernelRunContextFaker().KernelIONum(2, 1).Inputs({&allocator, sizes_holder.get()}).Build();
  auto run_context = context_holder.GetContext<KernelContext>();
  auto graph = std::make_shared<ge::ExecuteGraph>("copy_flow_alloc_repeat_test");
  auto node = FastNodeFaker(graph).Attr("copy_flow_count", kCopyFlowCount).Build();
  auto funcs = registry.FindKernelFuncs("AllocCopyFlowHbm");
  ASSERT_NE(funcs, nullptr);
  ASSERT_EQ(funcs->outputs_creator(node, run_context), ge::GRAPH_SUCCESS);
  ASSERT_EQ(funcs->run_func(run_context), ge::GRAPH_SUCCESS);
  EXPECT_EQ(allocator.GetAllocCount(), 2U);
  EXPECT_EQ(allocator.GetFreeCount(), 0U);
  sizes->MutableData()[0] = 256U;
  sizes->MutableData()[1] = 0U;
  ASSERT_EQ(funcs->run_func(run_context), ge::GRAPH_SUCCESS);
  EXPECT_EQ(allocator.GetAllocCount(), 3U);
  EXPECT_EQ(allocator.GetFreeCount(), 2U);

  auto addresses = run_context->GetOutputPointer<TypedContinuousVector<GertTensorData *>>(0U);
  ASSERT_NE(addresses, nullptr);
  ASSERT_EQ(addresses->GetSize(), static_cast<size_t>(kCopyFlowCount));
  ASSERT_NE(addresses->GetData()[0], nullptr);
  EXPECT_NE(addresses->GetData()[0]->GetAddr(), nullptr);
  ASSERT_NE(addresses->GetData()[1], nullptr);
  EXPECT_EQ(addresses->GetData()[1]->GetAddr(), nullptr);
  context_holder.FreeAll();
  EXPECT_EQ(allocator.GetFreeCount(), 3U);
  EXPECT_EQ(allocator.GetDuplicateFreeCount(), 0U);
}

TEST_F(MemoryKernelUT, AllocCopyFlowHbmKeepsSuccessfulRangeWhenLaterAllocationFails) {
  constexpr int64_t kCopyFlowCount = 2;
  CountingGertAllocator allocator(1U);
  auto sizes_holder = ContinuousVector::Create<size_t>(kCopyFlowCount);
  auto sizes = reinterpret_cast<TypedContinuousVector<size_t> *>(sizes_holder.get());
  sizes->SetSize(kCopyFlowCount);
  sizes->MutableData()[0] = 128U;
  sizes->MutableData()[1] = 128U;

  auto context_holder = KernelRunContextFaker().KernelIONum(2, 1).Inputs({&allocator, sizes_holder.get()}).Build();
  auto run_context = context_holder.GetContext<KernelContext>();
  auto graph = std::make_shared<ge::ExecuteGraph>("copy_flow_alloc_failure_test");
  auto node = FastNodeFaker(graph).Attr("copy_flow_count", kCopyFlowCount).Build();
  auto funcs = registry.FindKernelFuncs("AllocCopyFlowHbm");
  ASSERT_NE(funcs, nullptr);
  ASSERT_EQ(funcs->outputs_creator(node, run_context), ge::GRAPH_SUCCESS);
  ASSERT_EQ(funcs->run_func(run_context), ge::GRAPH_FAILED);
  EXPECT_EQ(allocator.GetAllocCount(), 2U);
  EXPECT_EQ(allocator.GetFreeCount(), 0U);

  auto addresses = run_context->GetOutputPointer<TypedContinuousVector<GertTensorData *>>(0U);
  ASSERT_NE(addresses, nullptr);
  ASSERT_EQ(addresses->GetSize(), 1U);
  ASSERT_NE(addresses->GetData()[0], nullptr);
  EXPECT_NE(addresses->GetData()[0]->GetAddr(), nullptr);
  context_holder.FreeAll();
  EXPECT_EQ(allocator.GetFreeCount(), 1U);
  EXPECT_EQ(allocator.GetDuplicateFreeCount(), 0U);
}

TEST_F(MemoryKernelUT, AllocCopyFlowHbmRejectsCountMismatchBeforeAllocation) {
  CountingGertAllocator allocator;
  auto sizes_holder = ContinuousVector::Create<size_t>(1U);
  auto sizes = reinterpret_cast<TypedContinuousVector<size_t> *>(sizes_holder.get());
  sizes->SetSize(1U);
  sizes->MutableData()[0] = 128U;
  auto context_holder = KernelRunContextFaker().KernelIONum(2, 1).Inputs({&allocator, sizes_holder.get()}).Build();
  auto run_context = context_holder.GetContext<KernelContext>();
  auto graph = std::make_shared<ge::ExecuteGraph>("copy_flow_alloc_count_mismatch_test");
  auto node = FastNodeFaker(graph).Attr("copy_flow_count", static_cast<int64_t>(2)).Build();
  auto funcs = registry.FindKernelFuncs("AllocCopyFlowHbm");
  ASSERT_NE(funcs, nullptr);
  ASSERT_EQ(funcs->outputs_creator(node, run_context), ge::GRAPH_SUCCESS);
  EXPECT_NE(funcs->run_func(run_context), ge::GRAPH_SUCCESS);
  EXPECT_EQ(allocator.GetAllocCount(), 0U);
  context_holder.FreeAll();
}

TEST_F(MemoryKernelUT, AllocCopyFlowHbmFreesBlockWithNullAddress) {
  CountingGertAllocator allocator(std::numeric_limits<size_t>::max(), 0U);
  auto sizes_holder = ContinuousVector::Create<size_t>(1U);
  auto sizes = reinterpret_cast<TypedContinuousVector<size_t> *>(sizes_holder.get());
  sizes->SetSize(1U);
  sizes->MutableData()[0] = 128U;
  auto context_holder = KernelRunContextFaker().KernelIONum(2, 1).Inputs({&allocator, sizes_holder.get()}).Build();
  auto run_context = context_holder.GetContext<KernelContext>();
  auto graph = std::make_shared<ge::ExecuteGraph>("copy_flow_alloc_null_address_test");
  auto node = FastNodeFaker(graph).Attr("copy_flow_count", static_cast<int64_t>(1)).Build();
  auto funcs = registry.FindKernelFuncs("AllocCopyFlowHbm");
  ASSERT_NE(funcs, nullptr);
  ASSERT_EQ(funcs->outputs_creator(node, run_context), ge::GRAPH_SUCCESS);
  EXPECT_NE(funcs->run_func(run_context), ge::GRAPH_SUCCESS);
  EXPECT_EQ(allocator.GetAllocCount(), 1U);
  context_holder.FreeAll();
  EXPECT_EQ(allocator.GetFreeCount(), 1U);
  EXPECT_EQ(allocator.GetDuplicateFreeCount(), 0U);
}
TEST_F(MemoryKernelUT, BatchAlloc_Success) {
  // fake allocator -- adapt multistream
  memory::CachingMemAllocator allocator(0, RT_MEMORY_HBM);
  memory::SingleStreamL2Allocator single_stream_l2_allocator(&allocator);
  // fake input sizes
  auto sizes_holder = ContinuousVector::Create<size_t>(8);
  auto sizes = reinterpret_cast<TypedContinuousVector<size_t> *>(sizes_holder.get());
  sizes->SetSize(2);
  sizes->MutableData()[0] = 128;
  sizes->MutableData()[1] = 128;

  // fake alloc context
  auto alloc_context_holder =
      KernelRunContextFaker().KernelIONum(2, 1).Inputs({&single_stream_l2_allocator, sizes_holder.get()}).Build();
  auto alloc_context = alloc_context_holder.GetContext<KernelContext>();

  ASSERT_NE(registry.FindKernelFuncs("AllocBatchHbm"), nullptr);
  ASSERT_NE(registry.FindKernelFuncs("AllocBatchHbm")->outputs_creator, nullptr);
  ASSERT_NE(registry.FindKernelFuncs("AllocBatchHbm")->run_func, nullptr);

  ASSERT_EQ(registry.FindKernelFuncs("AllocBatchHbm")->outputs_creator(nullptr, alloc_context), ge::GRAPH_SUCCESS);
  GertRuntimeStub runtime_stub;
  runtime_stub.GetSlogStub().NoConsoleOut().SetLevelInfo();
  ASSERT_EQ(registry.FindKernelFuncs("AllocBatchHbm")->run_func(alloc_context), ge::GRAPH_SUCCESS);
  ASSERT_TRUE(runtime_stub.GetSlogStub().FindInfoLogRegex(kAllocRe, {{2, "0"}}) >= 0);

  auto addresses = alloc_context->GetOutputPointer<TypedContinuousVector<GertTensorData *>>(0);
  ASSERT_NE(addresses, nullptr);
  ASSERT_EQ(addresses->GetSize(), 2);
  ASSERT_GT(addresses->GetCapacity(), 2);
  ASSERT_NE(addresses->MutableData()[0], nullptr);
  ASSERT_NE(addresses->MutableData()[1], nullptr);
}
TEST_F(MemoryKernelUT, BatchAlloc_NoLeaks_TwoTimesWithoutFree) {
  // fake allocator -- adapt multistream
  memory::CachingMemAllocator allocator(0, RT_MEMORY_HBM);
  memory::SingleStreamL2Allocator single_stream_l2_allocator(&allocator);
  // fake input sizes
  auto sizes_holder = ContinuousVector::Create<size_t>(8);
  auto sizes = reinterpret_cast<TypedContinuousVector<size_t> *>(sizes_holder.get());
  sizes->SetSize(2);
  sizes->MutableData()[0] = 128;
  sizes->MutableData()[1] = 128;

  // fake alloc context
  auto alloc_context_holder =
      KernelRunContextFaker().KernelIONum(2, 1).Inputs({&single_stream_l2_allocator, sizes_holder.get()}).Build();
  auto alloc_context = alloc_context_holder.GetContext<KernelContext>();

  ASSERT_NE(registry.FindKernelFuncs("AllocBatchHbm"), nullptr);
  ASSERT_NE(registry.FindKernelFuncs("AllocBatchHbm")->outputs_creator, nullptr);
  ASSERT_NE(registry.FindKernelFuncs("AllocBatchHbm")->run_func, nullptr);

  ASSERT_EQ(registry.FindKernelFuncs("AllocBatchHbm")->outputs_creator(nullptr, alloc_context), ge::GRAPH_SUCCESS);
  ASSERT_EQ(registry.FindKernelFuncs("AllocBatchHbm")->run_func(alloc_context), ge::GRAPH_SUCCESS);

  auto addresses = alloc_context->GetOutputPointer<TypedContinuousVector<GertTensorData *>>(0);
  ASSERT_NE(addresses, nullptr);
  ASSERT_EQ(addresses->GetSize(), 2);
  ASSERT_GT(addresses->GetCapacity(), 2);
  ASSERT_NE(addresses->MutableData()[0], nullptr);
  ASSERT_NE(addresses->MutableData()[1], nullptr);

  ASSERT_EQ(registry.FindKernelFuncs("AllocBatchHbm")->run_func(alloc_context), ge::GRAPH_SUCCESS);
}
TEST_F(MemoryKernelUT, BatchAllocFreeSuccess) {
  // fake allocator -- adapt multistream
  auto holder = memory::MultiStreamAllocatorFaker().StreamNum(2).Build();

  auto sizes_holder = ContinuousVector::Create<size_t>(8);
  auto sizes = reinterpret_cast<TypedContinuousVector<size_t> *>(sizes_holder.get());
  sizes->SetSize(2);
  sizes->MutableData()[0] = 128;
  sizes->MutableData()[1] = 128;

  // fake alloc context
  auto alloc_context_holder =
      KernelRunContextFaker().KernelIONum(2, 1).Inputs({holder.Allocator(1), sizes_holder.get()}).Build();
  auto alloc_context = alloc_context_holder.GetContext<KernelContext>();

  ASSERT_NE(registry.FindKernelFuncs("AllocBatchHbm"), nullptr);
  ASSERT_NE(registry.FindKernelFuncs("AllocBatchHbm")->outputs_creator, nullptr);
  ASSERT_NE(registry.FindKernelFuncs("AllocBatchHbm")->run_func, nullptr);
  ASSERT_NE(registry.FindKernelFuncs("FreeBatchHbm")->outputs_creator, nullptr);
  ASSERT_NE(registry.FindKernelFuncs("FreeBatchHbm")->run_func, nullptr);

  ASSERT_EQ(registry.FindKernelFuncs("AllocBatchHbm")->outputs_creator(nullptr, alloc_context), ge::GRAPH_SUCCESS);
  ASSERT_EQ(registry.FindKernelFuncs("AllocBatchHbm")->run_func(alloc_context), ge::GRAPH_SUCCESS);

  auto addresses = alloc_context->GetOutputPointer<TypedContinuousVector<GertTensorData *>>(0);
  ASSERT_NE(addresses, nullptr);
  ASSERT_EQ(addresses->GetSize(), 2);
  ASSERT_GT(addresses->GetCapacity(), 2);
  ASSERT_NE(addresses->MutableData()[0], nullptr);
  ASSERT_NE(addresses->MutableData()[1], nullptr);
  auto addr1 = ToHex(addresses->GetData()[0]->GetAddr());
  auto addr2 = ToHex(addresses->GetData()[1]->GetAddr());

  auto free_context_holder = KernelRunContextFaker().KernelIONum(1, 1).Inputs({addresses}).Build();
  auto free_context = free_context_holder.GetContext<KernelContext>();
  EXPECT_NE(registry.FindKernelFuncs("FreeBatchHbm")->outputs_creator, nullptr);
  GertRuntimeStub runtime_stub;
  runtime_stub.GetSlogStub().SetLevelInfo();
  ASSERT_EQ(registry.FindKernelFuncs("FreeBatchHbm")->run_func(free_context), ge::GRAPH_SUCCESS);
  ASSERT_TRUE(runtime_stub.GetSlogStub().FindInfoLogRegex(kFreeRe, {{2, "1"}, {3, addr1}}) >= 0);
  ASSERT_TRUE(runtime_stub.GetSlogStub().FindInfoLogRegex(kFreeRe, {{2, "1"}, {3, addr2}}) >= 0);

  ASSERT_EQ(addresses->GetSize(), 2U);
}

TEST_F(MemoryKernelUT, BatchAllocSizeZeroFreeSuccess) {
  // fake allocator -- adapt multistream
  auto holder = memory::MultiStreamAllocatorFaker().StreamNum(2).Build();

  auto sizes_holder = ContinuousVector::Create<size_t>(8);
  auto sizes = reinterpret_cast<TypedContinuousVector<size_t> *>(sizes_holder.get());
  sizes->SetSize(2);
  sizes->MutableData()[0] = 0;
  sizes->MutableData()[1] = 128;

  // fake alloc context
  auto alloc_context_holder =
      KernelRunContextFaker().KernelIONum(2, 1).Inputs({holder.Allocator(1), sizes_holder.get()}).Build();
  auto alloc_context = alloc_context_holder.GetContext<KernelContext>();

  ASSERT_NE(registry.FindKernelFuncs("AllocBatchHbm"), nullptr);
  ASSERT_NE(registry.FindKernelFuncs("AllocBatchHbm")->outputs_creator, nullptr);
  ASSERT_NE(registry.FindKernelFuncs("AllocBatchHbm")->run_func, nullptr);
  ASSERT_NE(registry.FindKernelFuncs("FreeBatchHbm")->outputs_creator, nullptr);
  ASSERT_NE(registry.FindKernelFuncs("FreeBatchHbm")->run_func, nullptr);

  ASSERT_EQ(registry.FindKernelFuncs("AllocBatchHbm")->outputs_creator(nullptr, alloc_context), ge::GRAPH_SUCCESS);
  ASSERT_EQ(registry.FindKernelFuncs("AllocBatchHbm")->run_func(alloc_context), ge::GRAPH_SUCCESS);

  auto addresses = alloc_context->GetOutputPointer<TypedContinuousVector<GertTensorData *>>(0);
  ASSERT_NE(addresses, nullptr);
  ASSERT_EQ(addresses->GetSize(), 2);
  ASSERT_GT(addresses->GetCapacity(), 2);
  ASSERT_NE(addresses->MutableData()[0], nullptr);
  ASSERT_NE(addresses->MutableData()[1], nullptr);
  auto addr1 = ToHex(addresses->GetData()[0]->GetAddr());
  auto addr2 = ToHex(addresses->GetData()[1]->GetAddr());

  auto free_context_holder = KernelRunContextFaker().KernelIONum(1, 1).Inputs({addresses}).Build();
  auto free_context = free_context_holder.GetContext<KernelContext>();
  EXPECT_NE(registry.FindKernelFuncs("FreeBatchHbm")->outputs_creator, nullptr);
  GertRuntimeStub runtime_stub;
  runtime_stub.GetSlogStub().SetLevelInfo();
  ASSERT_EQ(registry.FindKernelFuncs("FreeBatchHbm")->run_func(free_context), ge::GRAPH_SUCCESS);
  ASSERT_TRUE(runtime_stub.GetSlogStub().FindInfoLogRegex(kFreeRe, {{2, "1"}, {3, addr2}}) >= 0);

  ASSERT_EQ(addresses->GetSize(), 2U);
}

TEST_F(MemoryKernelUT, FreeMemory) {
  // fake allocator -- adapt multistream
  memory::CachingMemAllocator allocator(0, RT_MEMORY_HBM);
  memory::SingleStreamL2Allocator single_stream_l2_allocator(&allocator);

  ASSERT_NE(registry.FindKernelFuncs("FreeMemory"), nullptr);
  ASSERT_NE(registry.FindKernelFuncs("FreeMemory")->run_func, nullptr);

  // alloc
  auto context_holder1 = Alloc(single_stream_l2_allocator);

  // free
  auto free_context_holder =
      KernelRunContextFaker()
          .Inputs({context_holder1.GetContext<KernelContext>()->GetOutputPointer<GertTensorData>(0)})
          .Build();
  auto free_context = free_context_holder.GetContext<KernelContext>();

  auto addr = ToHex(context_holder1.GetContext<KernelContext>()->GetOutputPointer<GertTensorData>(0)->GetAddr());
  GertRuntimeStub runtime_stub;
  runtime_stub.GetSlogStub().NoConsoleOut().SetLevelInfo();
  ASSERT_EQ(registry.FindKernelFuncs("FreeMemory")->run_func(free_context), ge::GRAPH_SUCCESS);
  ASSERT_TRUE(runtime_stub.GetSlogStub().FindInfoLogRegex(kFreeRe, {{2, "0"}, {3, addr}}) >= 0);

  allocator.Finalize();
  free_context_holder.FreeAll();
}

TEST_F(MemoryKernelUT, FreeTensorMemory) {
  // fake allocator -- adapt multistream
  memory::CachingMemAllocator allocator(0, RT_MEMORY_HBM);
  memory::SingleStreamL2Allocator single_stream_l2_allocator(&allocator);

  ASSERT_NE(registry.FindKernelFuncs("AllocMemHbm"), nullptr);
  ASSERT_NE(registry.FindKernelFuncs("AllocMemHbm")->outputs_creator, nullptr);
  ASSERT_NE(registry.FindKernelFuncs("AllocMemHbm")->run_func, nullptr);
  ASSERT_NE(registry.FindKernelFuncs("FreeTensorMemory"), nullptr);
  ASSERT_NE(registry.FindKernelFuncs("FreeTensorMemory")->run_func, nullptr);

  // alloc
  auto alloc_context_holder = KernelRunContextFaker()
                                  .KernelIONum(2, 1)
                                  .Inputs({&single_stream_l2_allocator, reinterpret_cast<void *>(0x64)})
                                  .Build();
  auto alloc_context = alloc_context_holder.GetContext<KernelContext>();

  ASSERT_EQ(registry.FindKernelFuncs("AllocMemHbm")->outputs_creator(nullptr, alloc_context), ge::GRAPH_SUCCESS);
  ASSERT_EQ(kernel::AllocHbmMem(alloc_context), ge::GRAPH_SUCCESS);

  auto tensor_data = alloc_context->GetOutputPointer<GertTensorData>(0);
  ASSERT_NE(tensor_data, nullptr);
  ASSERT_NE(tensor_data->GetAddr(), nullptr);
  auto addr = ToHex(tensor_data->GetAddr());

  Tensor tensor;
  TensorUtils::ShareGtdToTd(*tensor_data, tensor.MutableTensorData());
  tensor.SetPlacement(kOnDeviceHbm);

  // free
  auto free_context_holder = KernelRunContextFaker().KernelIONum(1, 0).Inputs({&tensor}).Build();
  auto free_context = free_context_holder.GetContext<KernelContext>();

  GertRuntimeStub runtime_stub;
  runtime_stub.GetSlogStub().NoConsoleOut().SetLevelInfo();
  ASSERT_EQ(registry.FindKernelFuncs("FreeTensorMemory")->run_func(free_context), ge::GRAPH_SUCCESS);
  ASSERT_TRUE(runtime_stub.GetSlogStub().FindInfoLogRegex(kFreeRe, {{2, "-1"}, {3, addr}}) >= 0);

  allocator.Finalize();
  alloc_context_holder.FreeAll();
  free_context_holder.FreeAll();
}

TEST_F(MemoryKernelUT, CreateAllocator_Success_DeviceHbmOutput) {
  ASSERT_NE(registry.FindKernelFuncs("CreateL1Allocator"), nullptr);
  ASSERT_NE(registry.FindKernelFuncs("CreateL1Allocator")->outputs_creator, nullptr);
  ASSERT_NE(registry.FindKernelFuncs("CreateL1Allocator")->run_func, nullptr);

  auto hbm_context_holder =
      KernelRunContextFaker()
          .KernelIONum(2, 1)
          .Inputs({reinterpret_cast<void *>(kOnDeviceHbm), reinterpret_cast<void *>(kernel::MemoryType::kNodeOutput)})
          .Build();
  auto create_hbm_context = hbm_context_holder.GetContext<KernelContext>();
  ASSERT_EQ(registry.FindKernelFuncs("CreateL1Allocator")->run_func(create_hbm_context), ge::GRAPH_SUCCESS);

  // 校验创建的allocator类型不是RT_MEMORY_DEFAULT
  int32_t device_id = 0;
  aclrtGetDevice(&device_id);
  const auto iter = memory::RtsCachingMemAllocator::device_id_to_allocators_[device_id].find(RT_MEMORY_DEFAULT);
  EXPECT_EQ(iter, memory::RtsCachingMemAllocator::device_id_to_allocators_[device_id].end());
}

TEST_F(MemoryKernelUT, GetUserAllocatorOrFixedBaseAllocator_Success_DeviceHbmOutput) {
  ASSERT_NE(registry.FindKernelFuncs("GetUserAllocatorOrFixedBaseAllocator"), nullptr);
  ASSERT_NE(registry.FindKernelFuncs("GetUserAllocatorOrFixedBaseAllocator")->outputs_creator, nullptr);
  ASSERT_NE(registry.FindKernelFuncs("GetUserAllocatorOrFixedBaseAllocator")->run_func, nullptr);

  uint32_t session_id = 12051502;
  int stream = 0x12;
  auto hbm_context_holder = KernelRunContextFaker()
                                .KernelIONum(2, 1)
                                .Inputs({reinterpret_cast<void *>(kOnDeviceHbm), reinterpret_cast<void *>(session_id),
                                         reinterpret_cast<void *>(stream)})
                                .Build();
  auto create_hbm_context = hbm_context_holder.GetContext<KernelContext>();
  ASSERT_EQ(registry.FindKernelFuncs("GetUserAllocatorOrFixedBaseAllocator")->run_func(create_hbm_context),
            ge::GRAPH_SUCCESS);
  auto chain = create_hbm_context->GetOutput(0);
  auto allocator_in_chain = chain->GetValue<ge::Allocator *>();
  int32_t device_id = 0;
  aclrtGetDevice(&device_id);
  ge::Allocator *allocator = ge::SessionMemAllocator<ge::FixedBaseExpandableAllocator>::Instance()
                                 .GetMemAllocator(session_id, device_id)
                                 .get();

  EXPECT_EQ(ge::PtrToValue(allocator_in_chain), ge::PtrToValue(allocator));
  ge::SessionMemAllocator<ge::FixedBaseExpandableAllocator>::Instance().RemoveAllocator(session_id, device_id);
}

TEST_F(MemoryKernelUT, GerUserAllocatorOrFixedBaseAllocator_Success_DeviceP2pOutput) {
  ASSERT_NE(registry.FindKernelFuncs("GetUserAllocatorOrFixedBaseAllocator"), nullptr);
  ASSERT_NE(registry.FindKernelFuncs("GetUserAllocatorOrFixedBaseAllocator")->outputs_creator, nullptr);
  ASSERT_NE(registry.FindKernelFuncs("GetUserAllocatorOrFixedBaseAllocator")->run_func, nullptr);

  uint32_t session_id = 12051502;
  int stream = 0x12;
  auto context_holder = KernelRunContextFaker()
                            .KernelIONum(2, 1)
                            .Inputs({reinterpret_cast<void *>(kOnDeviceP2p), reinterpret_cast<void *>(session_id),
                                     reinterpret_cast<void *>(stream)})
                            .Build();
  auto context = context_holder.GetContext<KernelContext>();
  ASSERT_EQ(registry.FindKernelFuncs("GetUserAllocatorOrFixedBaseAllocator")->run_func(context), ge::GRAPH_SUCCESS);
  auto chain = context->GetOutput(0);
  auto allocator_in_chain = chain->GetValue<ge::Allocator *>();
  int32_t device_id = 0;
  aclrtGetDevice(&device_id);
  ge::Allocator *hbm_allocator = ge::SessionMemAllocator<ge::FixedBaseExpandableAllocator>::Instance()
                                     .GetMemAllocator(session_id, device_id)
                                     .get();
  EXPECT_NE(allocator_in_chain, hbm_allocator);

  ge::Allocator *p2p_allocator = ge::SessionMemAllocator<ge::FixedBaseExpandableAllocator>::Instance()
                                     .GetMemAllocator(session_id, device_id, RT_MEMORY_P2P_DDR)
                                     .get();
  EXPECT_EQ(ge::PtrToValue(allocator_in_chain), ge::PtrToValue(p2p_allocator));
  ge::SessionMemAllocator<ge::FixedBaseExpandableAllocator>::Instance().RemoveAllocator(session_id, device_id);
}

TEST_F(MemoryKernelUT, GerUserAllocatorOrFixedBaseAllocator_Failed_InvalidParam) {
  ASSERT_NE(registry.FindKernelFuncs("GetUserAllocatorOrFixedBaseAllocator"), nullptr);
  ASSERT_NE(registry.FindKernelFuncs("GetUserAllocatorOrFixedBaseAllocator")->outputs_creator, nullptr);
  ASSERT_NE(registry.FindKernelFuncs("GetUserAllocatorOrFixedBaseAllocator")->run_func, nullptr);

  uint32_t session_id = 12051502;
  int stream = 0x12;
  auto context_holder = KernelRunContextFaker()
                            .KernelIONum(2, 1)
                            .Inputs({reinterpret_cast<void *>(kOnHost), reinterpret_cast<void *>(session_id),
                                     reinterpret_cast<void *>(stream)})
                            .Build();
  auto context = context_holder.GetContext<KernelContext>();
  ASSERT_NE(registry.FindKernelFuncs("GetUserAllocatorOrFixedBaseAllocator")->run_func(context), ge::GRAPH_SUCCESS);
}

TEST_F(MemoryKernelUT, AllocFixedFeatureMemory_Success_CheckAddress) {
  ASSERT_NE(registry.FindKernelFuncs("AllocFixedFeatureMemory"), nullptr);
  ASSERT_NE(registry.FindKernelFuncs("AllocFixedFeatureMemory")->outputs_creator, nullptr);
  ASSERT_NE(registry.FindKernelFuncs("AllocFixedFeatureMemory")->run_func, nullptr);

  uint32_t session_id = 120515021;
  int32_t device_id = 0;
  uint32_t mem_size = 4U * 1024U * 1024U;
  aclrtGetDevice(&device_id);
  auto allocator =
      ge::SessionMemAllocator<ge::FixedBaseExpandableAllocator>::Instance().GetMemAllocator(session_id, device_id);
  memory::CachingMemAllocator caching_mem_allocator(0, RT_MEMORY_HBM);
  memory::SingleStreamL2Allocator single_stream_l2_allocator(&caching_mem_allocator);
  ge::Allocator *interface_allocator = allocator.get();
  auto hbm_context_holder =
      KernelRunContextFaker()
          .KernelIONum(3, 1)
          .Inputs({reinterpret_cast<void *>(interface_allocator), reinterpret_cast<void *>(mem_size),
                   reinterpret_cast<void *>(&single_stream_l2_allocator)})
          .Build();
  auto context = hbm_context_holder.GetContext<KernelContext>();
  ASSERT_EQ(registry.FindKernelFuncs("AllocFixedFeatureMemory")->outputs_creator(nullptr, context), ge::GRAPH_SUCCESS);
  ASSERT_EQ(registry.FindKernelFuncs("AllocFixedFeatureMemory")->run_func(context), ge::GRAPH_SUCCESS);
  auto tensor_data = context->GetOutputPointer<GertTensorData>(0);
  ASSERT_NE(tensor_data, nullptr);
  ASSERT_NE(tensor_data->GetAddr(), nullptr);

  auto mem_block = allocator->Malloc(mem_size);
  ASSERT_EQ(tensor_data->GetAddr(), mem_block->GetAddr());
  ASSERT_EQ(tensor_data->GetSize(), mem_size);

  auto msb = reinterpret_cast<memory::MultiStreamMemBlock *>(tensor_data->GetGertMemBlock());
  ASSERT_NE(msb, nullptr);
  auto ti_block = reinterpret_cast<memory::TiBlock *>(msb->GetMemBlock());
  ASSERT_NE(ti_block, nullptr);
  TensorAddrManager manager;
  void *block = ti_block->GetTd().Release(manager);
  ASSERT_NE(block, nullptr);
  auto mb = reinterpret_cast<ge::MemBlock *>(block);
  ASSERT_EQ(mb->GetCount(), 1);

  mem_block->Free();
  tensor_data->Free();
  mb->Free();
  ge::SessionMemAllocator<ge::FixedBaseExpandableAllocator>::Instance().RemoveAllocator(session_id, device_id);
}

TEST_F(MemoryKernelUT, FreeFixedFeatureMemory_Success) {
  ASSERT_NE(registry.FindKernelFuncs("FreeFixedFeatureMemory"), nullptr);
  ASSERT_NE(registry.FindKernelFuncs("FreeFixedFeatureMemory")->outputs_creator, nullptr);
  ASSERT_NE(registry.FindKernelFuncs("FreeFixedFeatureMemory")->run_func, nullptr);

  ASSERT_NE(registry.FindKernelFuncs("AllocFixedFeatureMemory"), nullptr);
  ASSERT_NE(registry.FindKernelFuncs("AllocFixedFeatureMemory")->outputs_creator, nullptr);
  ASSERT_NE(registry.FindKernelFuncs("AllocFixedFeatureMemory")->run_func, nullptr);

  uint32_t session_id = 120515021;
  int32_t device_id = 0;
  uint32_t mem_size = 4U * 1024U * 1024U;
  aclrtGetDevice(&device_id);
  auto allocator =
      ge::SessionMemAllocator<ge::FixedBaseExpandableAllocator>::Instance().GetMemAllocator(session_id, device_id);
  ge::Allocator *interface_allocator = allocator.get();
  memory::CachingMemAllocator caching_mem_allocator(0, RT_MEMORY_HBM);
  memory::SingleStreamL2Allocator single_stream_l2_allocator(&caching_mem_allocator);
  auto hbm_context_holder =
      KernelRunContextFaker()
          .KernelIONum(3, 1)
          .Inputs({reinterpret_cast<void *>(interface_allocator), reinterpret_cast<void *>(mem_size),
                   reinterpret_cast<void *>(&single_stream_l2_allocator)})
          .Build();
  auto context = hbm_context_holder.GetContext<KernelContext>();
  ASSERT_EQ(registry.FindKernelFuncs("AllocFixedFeatureMemory")->outputs_creator(nullptr, context), ge::GRAPH_SUCCESS);
  ASSERT_EQ(registry.FindKernelFuncs("AllocFixedFeatureMemory")->run_func(context), ge::GRAPH_SUCCESS);
  auto tensor_data = context->GetOutputPointer<GertTensorData>(0);
  ASSERT_NE(tensor_data, nullptr);
  ASSERT_NE(tensor_data->GetAddr(), nullptr);

  auto free_context_holder =
      KernelRunContextFaker()
          .KernelIONum(2, 1)
          .Inputs({reinterpret_cast<void *>(tensor_data), reinterpret_cast<void *>(allocator.get())})
          .Build();
  auto free_context = free_context_holder.GetContext<KernelContext>();
  ASSERT_EQ(registry.FindKernelFuncs("FreeFixedFeatureMemory")->run_func(free_context), ge::GRAPH_SUCCESS);

  ge::SessionMemAllocator<ge::FixedBaseExpandableAllocator>::Instance().RemoveAllocator(session_id, device_id);
}

TEST_F(MemoryKernelUT, CreateAllocator_Success_DeviceP2pOutput) {
  ASSERT_NE(registry.FindKernelFuncs("CreateL1Allocator"), nullptr);
  ASSERT_NE(registry.FindKernelFuncs("CreateL1Allocator")->outputs_creator, nullptr);
  ASSERT_NE(registry.FindKernelFuncs("CreateL1Allocator")->run_func, nullptr);

  auto hbm_context_holder =
      KernelRunContextFaker()
          .KernelIONum(2, 1)
          .Inputs({reinterpret_cast<void *>(kOnDeviceP2p), reinterpret_cast<void *>(kernel::MemoryType::kNodeOutput)})
          .Build();
  auto create_hbm_context = hbm_context_holder.GetContext<KernelContext>();
  ASSERT_EQ(registry.FindKernelFuncs("CreateL1Allocator")->run_func(create_hbm_context), ge::GRAPH_SUCCESS);

  int32_t device_id = 0;
  aclrtGetDevice(&device_id);
  const auto iter = memory::RtsCachingMemAllocator::device_id_to_allocators_[device_id].find(RT_MEMORY_P2P_DDR);
  EXPECT_NE(iter, memory::RtsCachingMemAllocator::device_id_to_allocators_[device_id].end());
}

TEST_F(MemoryKernelUT, SelectL1Allocator_SelectExternal_HostExternalNotNull) {
  std::unique_ptr<Allocators> allocators(CreateDefaultAllocators().release());
  std::unique_ptr<memory::HostMemAllocator> created_allocator = std::make_unique<memory::HostMemAllocator>();
  auto host_context_holder =
      KernelRunContextFaker()
          .KernelIONum(4, 1)
          .Inputs({reinterpret_cast<void *>(kOnHost), reinterpret_cast<void *>(allocators.get()),
                   reinterpret_cast<void *>(created_allocator.get()), reinterpret_cast<void *>(1)})
          .Build();
  auto host_context = host_context_holder.GetContext<KernelContext>();
  ASSERT_NE(registry.FindKernelFuncs("SelectL1Allocator")->outputs_creator, nullptr);
  ASSERT_EQ(registry.FindKernelFuncs("SelectL1Allocator")->run_func(host_context), ge::GRAPH_SUCCESS);
  auto ret = host_context->GetOutput(0)->GetValue<ge::Allocator *>();
  ASSERT_EQ(ret, allocators->GetAllocator(kOnHost, 0));
}

TEST_F(MemoryKernelUT, CreateInitL2Allocator_Success_DeviceHbmOutput) {
  ASSERT_NE(registry.FindKernelFuncs("CreateInitL2Allocator"), nullptr);
  ASSERT_NE(registry.FindKernelFuncs("CreateInitL2Allocator")->run_func, nullptr);

  auto caching_mem_allocator = std::make_shared<memory::CachingMemAllocator>(0, RT_MEMORY_HBM);
  std::unique_ptr<memory::SingleStreamL2Allocator> l1_allocator =
      std::make_unique<memory::SingleStreamL2Allocator>(caching_mem_allocator.get());

  const size_t stream_num = 2UL;
  auto l2_allocators_holder = ContinuousVector::Create<memory::MultiStreamL2Allocator *>(stream_num);
  ASSERT_NE(l2_allocators_holder, nullptr);
  auto l2_allocators =
      reinterpret_cast<TypedContinuousVector<memory::MultiStreamL2Allocator *> *>(l2_allocators_holder.get());
  l2_allocators->SetSize(stream_num);

  auto multi_stream_allocator = std::make_shared<memory::MultiStreamL2Allocator>(nullptr, kOnDeviceHbm);
  l2_allocators->MutableData()[0] = multi_stream_allocator.get();
  auto hbm_context_holder = KernelRunContextFaker()
                                .KernelIONum(3, 2)
                                .Inputs({reinterpret_cast<void *>(l1_allocator.get()),
                                         reinterpret_cast<void *>(l2_allocators), reinterpret_cast<void *>(0)})
                                .Build();
  auto create_hbm_context = hbm_context_holder.GetContext<KernelContext>();
  ASSERT_EQ(registry.FindKernelFuncs("CreateInitL2Allocator")->outputs_creator(nullptr, create_hbm_context),
            ge::GRAPH_SUCCESS);
  ASSERT_EQ(registry.FindKernelFuncs("CreateInitL2Allocator")->run_func(create_hbm_context), ge::GRAPH_SUCCESS);

  auto output_allocator = create_hbm_context->GetOutputPointer<memory::MultiStreamL2Allocator>(0);
  auto output_l2_allocators =
      create_hbm_context->GetOutputPointer<TypedContinuousVector<memory::MultiStreamL2Allocator *>>(1);
  ASSERT_NE(output_allocator, nullptr);
  ASSERT_EQ(output_allocator->GetPlacement(), kOnDeviceHbm);
  ASSERT_NE(output_allocator, l2_allocators->MutableData()[0]);  // 需要复制一份，不能修改输入
  ASSERT_NE(output_l2_allocators, nullptr);
  ASSERT_EQ(output_l2_allocators->GetSize(), stream_num);
  ASSERT_EQ(output_l2_allocators->MutableData()[0], output_allocator);
  ASSERT_EQ(output_l2_allocators->MutableData()[1], l2_allocators->MutableData()[1]);
}

TEST_F(MemoryKernelUT, CreateInitL2Allocator_Success_DeviceP2pOutput) {
  ASSERT_NE(registry.FindKernelFuncs("CreateInitL2Allocator"), nullptr);
  ASSERT_NE(registry.FindKernelFuncs("CreateInitL2Allocator")->run_func, nullptr);

  auto caching_mem_allocator = std::make_shared<memory::CachingMemAllocator>(0, RT_MEMORY_P2P_DDR);
  std::unique_ptr<memory::SingleStreamL2Allocator> l1_allocator =
      std::make_unique<memory::SingleStreamL2Allocator>(kOnDeviceP2p, caching_mem_allocator.get());

  const size_t stream_num = 2UL;
  auto l2_allocators_holder = ContinuousVector::Create<memory::MultiStreamL2Allocator *>(stream_num);
  ASSERT_NE(l2_allocators_holder, nullptr);
  auto l2_allocators =
      reinterpret_cast<TypedContinuousVector<memory::MultiStreamL2Allocator *> *>(l2_allocators_holder.get());
  l2_allocators->SetSize(stream_num);

  auto multi_stream_allocator = std::make_shared<memory::MultiStreamL2Allocator>(nullptr, kOnDeviceP2p);
  l2_allocators->MutableData()[0] = multi_stream_allocator.get();
  auto hbm_context_holder = KernelRunContextFaker()
                                .KernelIONum(3, 2)
                                .Inputs({reinterpret_cast<void *>(l1_allocator.get()),
                                         reinterpret_cast<void *>(l2_allocators), reinterpret_cast<void *>(0)})
                                .Build();
  auto create_hbm_context = hbm_context_holder.GetContext<KernelContext>();
  ASSERT_EQ(registry.FindKernelFuncs("CreateInitL2Allocator")->outputs_creator(nullptr, create_hbm_context),
            ge::GRAPH_SUCCESS);
  ASSERT_EQ(registry.FindKernelFuncs("CreateInitL2Allocator")->run_func(create_hbm_context), ge::GRAPH_SUCCESS);

  auto output_allocator = create_hbm_context->GetOutputPointer<memory::MultiStreamL2Allocator>(0);
  auto output_l2_allocators =
      create_hbm_context->GetOutputPointer<TypedContinuousVector<memory::MultiStreamL2Allocator *>>(1);
  ASSERT_NE(output_allocator, nullptr);
  ASSERT_EQ(output_allocator->GetPlacement(), kOnDeviceP2p);
  ASSERT_NE(output_allocator, l2_allocators->MutableData()[0]);  // 需要复制一份，不能修改输入
  ASSERT_NE(output_l2_allocators, nullptr);
  ASSERT_EQ(output_l2_allocators->GetSize(), stream_num);
  ASSERT_EQ(output_l2_allocators->MutableData()[0], output_allocator);
  ASSERT_EQ(output_l2_allocators->MutableData()[1], l2_allocators->MutableData()[1]);
}

TEST_F(MemoryKernelUT, CreateL2Allocators_Success_DeviceHbmOutput) {
  ASSERT_NE(registry.FindKernelFuncs("CreateL2Allocators"), nullptr);
  ASSERT_NE(registry.FindKernelFuncs("CreateL2Allocators")->run_func, nullptr);

  auto stream_num = 2UL;
  auto hbm_context_holder =
      KernelRunContextFaker().KernelIONum(1, 2).Inputs({reinterpret_cast<void *>(stream_num)}).Build();
  auto create_hbm_context = hbm_context_holder.GetContext<KernelContext>();
  ASSERT_EQ(registry.FindKernelFuncs("CreateL2Allocators")->outputs_creator(nullptr, create_hbm_context),
            ge::GRAPH_SUCCESS);
  ASSERT_EQ(registry.FindKernelFuncs("CreateL2Allocators")->run_func(create_hbm_context), ge::GRAPH_SUCCESS);

  auto output_allocator =
      create_hbm_context->GetOutputPointer<TypedContinuousVector<memory::MultiStreamL2Allocator *>>(0);
  ASSERT_EQ(output_allocator->GetSize(), stream_num);
  auto all_l2_mem_pool = create_hbm_context->GetOutputPointer<TypedContinuousVector<memory::L2MemPool *>>(1U);
  ASSERT_EQ(all_l2_mem_pool->GetSize(), stream_num);
  for (size_t i = 0U; i < stream_num; ++i) {
    ASSERT_NE(output_allocator->MutableData()[i], nullptr);
    ASSERT_NE(all_l2_mem_pool->MutableData()[i], nullptr);
    ASSERT_EQ(output_allocator->MutableData()[i]->GetPlacement(), kOnDeviceHbm);
  }
}

TEST_F(MemoryKernelUT, CreateL2Allocators_Success_DeviceP2pOutput) {
  ASSERT_NE(registry.FindKernelFuncs("CreateL2Allocators"), nullptr);
  ASSERT_NE(registry.FindKernelFuncs("CreateL2Allocators")->run_func, nullptr);

  size_t stream_num = 2UL;
  auto hbm_context_holder = KernelRunContextFaker()
                                .KernelIONum(2, 2)
                                .Inputs({reinterpret_cast<void *>(stream_num), reinterpret_cast<void *>(kOnDeviceP2p)})
                                .Build();
  auto create_hbm_context = hbm_context_holder.GetContext<KernelContext>();
  ASSERT_EQ(registry.FindKernelFuncs("CreateL2Allocators")->outputs_creator(nullptr, create_hbm_context),
            ge::GRAPH_SUCCESS);
  ASSERT_EQ(registry.FindKernelFuncs("CreateL2Allocators")->run_func(create_hbm_context), ge::GRAPH_SUCCESS);

  auto output_allocator =
      create_hbm_context->GetOutputPointer<TypedContinuousVector<memory::MultiStreamL2Allocator *>>(0);
  ASSERT_EQ(output_allocator->GetSize(), stream_num);
  auto all_l2_mem_pool = create_hbm_context->GetOutputPointer<TypedContinuousVector<memory::L2MemPool *>>(1U);
  ASSERT_EQ(all_l2_mem_pool->GetSize(), stream_num);
  for (size_t i = 0; i < stream_num; ++i) {
    ASSERT_NE(output_allocator->MutableData()[i], nullptr);
    ASSERT_NE(all_l2_mem_pool->MutableData()[i], nullptr);
    ASSERT_EQ(output_allocator->MutableData()[i]->GetPlacement(), kOnDeviceP2p);
  }
}

TEST_F(MemoryKernelUT, SelectL2Allocator_Device_Success_SingleStream) {
  auto caching_mem_allocator = std::make_shared<memory::CachingMemAllocator>(0, RT_MEMORY_HBM);
  const size_t stream_num = 1UL;
  const int64_t stream_id = 0;
  auto l2_allocators_holder = ContinuousVector::Create<memory::SingleStreamL2Allocator *>(stream_num);
  ASSERT_NE(l2_allocators_holder, nullptr);
  auto l2_allocators =
      reinterpret_cast<TypedContinuousVector<memory::SingleStreamL2Allocator *> *>(l2_allocators_holder.get());
  l2_allocators->SetSize(stream_num);
  std::unique_ptr<memory::SingleStreamL2Allocator> l2_allocator =
      std::make_unique<memory::SingleStreamL2Allocator>(caching_mem_allocator.get());
  l2_allocators->MutableData()[0] = l2_allocator.get();
  auto host_context_holder =
      KernelRunContextFaker()
          .KernelIONum(4, 1)
          .Inputs({reinterpret_cast<void *>(stream_id), reinterpret_cast<void *>(1),
                   reinterpret_cast<void *>(caching_mem_allocator.get()), l2_allocators_holder.get()})
          .Build();
  auto host_context = host_context_holder.GetContext<KernelContext>();
  ASSERT_EQ(registry.FindKernelFuncs("SelectL2Allocator")->run_func(host_context), ge::GRAPH_SUCCESS);
  auto ret = host_context->GetOutput(0)->GetValue<GertAllocator *>();
  ASSERT_EQ(ret, l2_allocator.get());
  ASSERT_EQ(ret->GetPlacement(), l2_allocator->GetPlacement());
  ASSERT_EQ(ret->GetStreamId(), 0);
}

TEST_F(MemoryKernelUT, SelectL2Allocator_DeviceP2p_Success_SingleStream) {
  auto caching_mem_allocator = std::make_shared<memory::CachingMemAllocator>(0, RT_MEMORY_P2P_DDR);
  const size_t stream_num = 1UL;
  const int64_t stream_id = 0;
  auto l2_allocators_holder = ContinuousVector::Create<memory::SingleStreamL2Allocator *>(stream_num);
  ASSERT_NE(l2_allocators_holder, nullptr);
  auto l2_allocators =
      reinterpret_cast<TypedContinuousVector<memory::SingleStreamL2Allocator *> *>(l2_allocators_holder.get());
  l2_allocators->SetSize(stream_num);
  std::unique_ptr<memory::SingleStreamL2Allocator> l2_allocator =
      std::make_unique<memory::SingleStreamL2Allocator>(kOnDeviceP2p, caching_mem_allocator.get());
  l2_allocators->MutableData()[0] = l2_allocator.get();
  auto host_context_holder =
      KernelRunContextFaker()
          .KernelIONum(4, 1)
          .Inputs({reinterpret_cast<void *>(stream_id), reinterpret_cast<void *>(1),
                   reinterpret_cast<void *>(caching_mem_allocator.get()), l2_allocators_holder.get()})
          .Build();
  auto host_context = host_context_holder.GetContext<KernelContext>();
  ASSERT_EQ(registry.FindKernelFuncs("SelectL2Allocator")->run_func(host_context), ge::GRAPH_SUCCESS);
  auto ret = host_context->GetOutput(0)->GetValue<GertAllocator *>();
  ASSERT_EQ(ret, l2_allocator.get());
  ASSERT_EQ(ret->GetPlacement(), l2_allocator->GetPlacement());
  ASSERT_EQ(ret->GetPlacement(), kOnDeviceP2p);
  ASSERT_EQ(ret->GetStreamId(), 0);
}

TEST_F(MemoryKernelUT, SelectL2Allocator_Device_Success_MultiStream) {
  auto caching_mem_allocator = std::make_shared<memory::CachingMemAllocator>(0, RT_MEMORY_P2P_DDR);
  const size_t stream_num = 2UL;
  auto holder = memory::MultiStreamAllocatorFaker().StreamNum(stream_num).L1Allocator(caching_mem_allocator).Build();
  auto host_context_holder =
      KernelRunContextFaker()
          .KernelIONum(4, 1)
          .Inputs({reinterpret_cast<void *>(1), reinterpret_cast<void *>(1),
                   reinterpret_cast<void *>(caching_mem_allocator.get()), holder.l2_allocators_holder.get()})
          .Build();
  auto host_context = host_context_holder.GetContext<KernelContext>();
  ASSERT_EQ(registry.FindKernelFuncs("SelectL2Allocator")->run_func(host_context), ge::GRAPH_SUCCESS);
  auto ret = host_context->GetOutput(0)->GetValue<GertAllocator *>();
  ASSERT_EQ(ret, holder.at(1));
  ASSERT_EQ(ret->GetPlacement(), kOnDeviceHbm);
  ASSERT_EQ(ret->GetStreamId(), 1);
}

TEST_F(MemoryKernelUT, SelectL2Allocator_DeviceP2p_Success_MultiStream) {
  auto caching_mem_allocator = std::make_shared<memory::CachingMemAllocator>(0, RT_MEMORY_P2P_DDR);
  const size_t stream_num = 2UL;
  auto holder = memory::MultiStreamAllocatorFaker()
                    .StreamNum(stream_num)
                    .Placement(kOnDeviceP2p)
                    .L1Allocator(caching_mem_allocator)
                    .Build();
  auto host_context_holder =
      KernelRunContextFaker()
          .KernelIONum(4, 1)
          .Inputs({reinterpret_cast<void *>(1), reinterpret_cast<void *>(1),
                   reinterpret_cast<void *>(caching_mem_allocator.get()), holder.l2_allocators_holder.get()})
          .Build();
  auto host_context = host_context_holder.GetContext<KernelContext>();
  ASSERT_EQ(registry.FindKernelFuncs("SelectL2Allocator")->run_func(host_context), ge::GRAPH_SUCCESS);
  auto ret = host_context->GetOutput(0)->GetValue<GertAllocator *>();
  ASSERT_EQ(ret, holder.at(1));
  ASSERT_EQ(ret->GetPlacement(), kOnDeviceP2p);
  ASSERT_EQ(ret->GetStreamId(), 1);
}

TEST_F(MemoryKernelUT, SelectL2Allocator_PlacementInvalid) {
  auto host_mem_allocator = std::make_shared<memory::HostMemAllocator>();
  std::unique_ptr<memory::HostGertMemAllocator> l1_allocator =
      std::make_unique<memory::HostGertMemAllocator>(host_mem_allocator.get());
  l1_allocator->SetPlacement(kTensorPlacementEnd);
  auto l2_allocators = ContinuousVector::Create<GertAllocator *>(1);
  auto host_context_holder = KernelRunContextFaker()
                                 .KernelIONum(4, 1)
                                 .Inputs({reinterpret_cast<void *>(0), reinterpret_cast<void *>(0),
                                          reinterpret_cast<void *>(l1_allocator.get()), l2_allocators.get()})
                                 .Build();
  auto host_context = host_context_holder.GetContext<KernelContext>();
  ASSERT_EQ(registry.FindKernelFuncs("SelectL2Allocator")->run_func(host_context), ge::PARAM_INVALID);
}

TEST_F(MemoryKernelUT, SelectL1Allocator_SelectExternal_DeviceExternalNotNull) {
  std::unique_ptr<Allocators> allocators(CreateDefaultAllocators().release());
  std::unique_ptr<memory::HostMemAllocator> created_allocator = std::make_unique<memory::HostMemAllocator>();
  auto host_context_holder =
      KernelRunContextFaker()
          .KernelIONum(4, 1)
          .Inputs({reinterpret_cast<void *>(kOnDeviceHbm), reinterpret_cast<void *>(allocators.get()),
                   reinterpret_cast<void *>(created_allocator.get())})
          .Build();
  auto device_context = host_context_holder.GetContext<KernelContext>();
  ASSERT_NE(registry.FindKernelFuncs("SelectL1Allocator")->outputs_creator, nullptr);
  ASSERT_EQ(registry.FindKernelFuncs("SelectL1Allocator")->run_func(device_context), ge::GRAPH_SUCCESS);
  auto ret = device_context->GetOutput(0)->GetValue<ge::Allocator *>();
  ASSERT_EQ(ret, allocators->GetAllocator(kOnDeviceHbm, 0));
}
TEST_F(MemoryKernelUT, SelectL1Allocator_WorkspaceAndOutputUseSameALlocator) {
  std::unique_ptr<Allocators> allocators(CreateDefaultAllocators().release());
  auto context_node_output =
      KernelRunContextFaker()
          .KernelIONum(4, 1)
          .Inputs({reinterpret_cast<void *>(kOnDeviceHbm), reinterpret_cast<void *>(allocators.get()), nullptr})
          .Build();
  auto context1 = context_node_output.GetContext<KernelContext>();
  ASSERT_NE(registry.FindKernelFuncs("SelectL1Allocator")->outputs_creator, nullptr);
  ASSERT_EQ(registry.FindKernelFuncs("SelectL1Allocator")->run_func(context1), ge::GRAPH_SUCCESS);
  auto allocator_node_output = context1->GetOutput(0)->GetValue<memory::SingleStreamL2Allocator *>();
  auto context_workspace =
      KernelRunContextFaker()
          .KernelIONum(4, 1)
          .Inputs({reinterpret_cast<void *>(kOnDeviceHbm), reinterpret_cast<void *>(allocators.get()), nullptr})
          .Build();
  auto context2 = context_workspace.GetContext<KernelContext>();
  EXPECT_NE(registry.FindKernelFuncs("SelectL1Allocator")->outputs_creator, nullptr);
  ASSERT_EQ(registry.FindKernelFuncs("SelectL1Allocator")->run_func(context2), ge::GRAPH_SUCCESS);
  auto allocator_workspace = context2->GetOutput(0)->GetValue<memory::SingleStreamL2Allocator *>();

  ASSERT_EQ(allocator_node_output->GetL1Allocator(), allocator_workspace->GetL1Allocator());
}

// 如果新增placement，需要支持新的allocator创建，或者复用已有allocator，此UT校验新增placement场景
TEST_F(MemoryKernelUT, CreateAllocator_With_New_Placement) {
  (void)AllocatorFactory::Create(kTensorPlacementEnd).release();
  for (size_t i = 0U; i < static_cast<size_t>(kTensorPlacementEnd); ++i) {
    ASSERT_NE(AllocatorFactory::Create(static_cast<TensorPlacement>(i)), nullptr);
  }
}

TEST_F(MemoryKernelUT, CreateAllocator_Success_CheckDeviceId) {
  auto allocator = AllocatorFactory::Create(kOnDeviceHbm);
  auto caching_mem_allocator = dynamic_cast<gert::memory::CachingMemAllocator *>(allocator.get());
  ASSERT_NE(caching_mem_allocator, nullptr);
  ASSERT_NE(caching_mem_allocator->GetDeviceId(), 2);
}

// 校验GetAllocator之前未调用SetAllocator场景
TEST_F(MemoryKernelUT, CreateAllocator_GetALlocatorIsNullWhenNotSet) {
  std::unique_ptr<Allocators> allocators = std::make_unique<Allocators>();
  ASSERT_EQ(allocators->GetAllocator(kOnDeviceHbm, 0), nullptr);
}
TEST_F(MemoryKernelUT, SelectL1Allocator_SelectInner_HostInnerNotNull) {
  std::unique_ptr<Allocators> allocators = std::make_unique<Allocators>();
  if (allocators == nullptr) {
    GELOGE(ge::PARAM_INVALID, "allocators is nullptr");
  }
  std::unique_ptr<memory::HostMemAllocator> created_allocator = std::make_unique<memory::HostMemAllocator>();
  auto host_context_holder = KernelRunContextFaker()
                                 .KernelIONum(4, 1)
                                 .Inputs({reinterpret_cast<void *>(kOnHost), reinterpret_cast<void *>(allocators.get()),
                                          reinterpret_cast<void *>(created_allocator.get())})
                                 .Build();
  auto host_context = host_context_holder.GetContext<KernelContext>();
  ASSERT_NE(registry.FindKernelFuncs("SelectL1Allocator")->outputs_creator, nullptr);
  ASSERT_EQ(registry.FindKernelFuncs("SelectL1Allocator")->run_func(host_context), ge::GRAPH_SUCCESS);
  ASSERT_EQ(host_context->GetOutput(0)->GetValue<ge::Allocator *>(), created_allocator.get());
}
TEST_F(MemoryKernelUT, SelectL1Allocator_SelectInner_DeviceInnerNotNull) {
  std::unique_ptr<Allocators> allocators = std::make_unique<Allocators>();
  if (allocators == nullptr) {
    GELOGE(ge::PARAM_INVALID, "allocators is nullptr");
  }
  std::unique_ptr<memory::CachingMemAllocator> created_allocator =
      std::make_unique<memory::CachingMemAllocator>(0, RT_MEMORY_HBM);
  auto host_context_holder =
      KernelRunContextFaker()
          .KernelIONum(4, 1)
          .Inputs({reinterpret_cast<void *>(kOnDeviceHbm), reinterpret_cast<void *>(allocators.get()),
                   reinterpret_cast<void *>(created_allocator.get())})
          .Build();
  auto host_context = host_context_holder.GetContext<KernelContext>();
  ASSERT_NE(registry.FindKernelFuncs("SelectL1Allocator")->outputs_creator, nullptr);
  ASSERT_EQ(registry.FindKernelFuncs("SelectL1Allocator")->run_func(host_context), ge::GRAPH_SUCCESS);
  ASSERT_EQ(host_context->GetOutput(0)->GetValue<ge::Allocator *>(), created_allocator.get());
}

TEST_F(MemoryKernelUT, SelectL1Allocator_ExternalCachingMemAllocator_SetStreamSuccess) {
  std::unique_ptr<Allocators> allocators = std::make_unique<Allocators>();
  if (allocators == nullptr) {
    GELOGE(ge::PARAM_INVALID, "allocators is nullptr");
  }
  std::shared_ptr<ge::Allocator> created_allocator = std::make_shared<memory::CachingMemAllocator>(0, RT_MEMORY_HBM);
  allocators->SetAllocator(kOnDeviceHbm, 0, created_allocator);
  rtStream stream = (void *)0x202401022154;
  auto host_context_holder =
      KernelRunContextFaker()
          .KernelIONum(4, 1)
          .Inputs({reinterpret_cast<void *>(kOnDeviceHbm), reinterpret_cast<void *>(allocators.get()),
                   reinterpret_cast<void *>(created_allocator.get()), stream})
          .Build();
  auto context = host_context_holder.GetContext<KernelContext>();
  ASSERT_NE(registry.FindKernelFuncs("SelectL1Allocator")->outputs_creator, nullptr);
  ASSERT_EQ(registry.FindKernelFuncs("SelectL1Allocator")->run_func(context), ge::GRAPH_SUCCESS);
  ASSERT_EQ(context->GetOutput(0)->GetValue<ge::Allocator *>(), created_allocator.get());

  auto mock_memcpy = [](rtStream_t stm, int32_t timeout) -> int {
    if (stm == (void *)0x202401022154) {
      return 0;
    }
    std::cerr << " stream not correct! " << std::endl;
    return -1;
  };
  auto runtime_stub = std::make_shared<MockRtStreamSync>();
  ge::RuntimeStub::SetInstance(runtime_stub);
  EXPECT_CALL(*runtime_stub, rtStreamSynchronizeWithTimeout).WillRepeatedly(testing::Invoke(mock_memcpy));
  EXPECT_EQ(reinterpret_cast<memory::CachingMemAllocator *>(created_allocator.get())->Synchronize(), ge::SUCCESS);
  ge::RuntimeStub::Reset();
}

TEST_F(MemoryKernelUT, SelectL1Allocator_SelectInner_ExternalNull) {
  std::unique_ptr<Allocators> allocators = nullptr;
  if (allocators == nullptr) {
    GELOGE(ge::PARAM_INVALID, "allocators is nullptr");
  }
  std::unique_ptr<memory::CachingMemAllocator> created_allocator =
      std::make_unique<memory::CachingMemAllocator>(0, RT_MEMORY_HBM);
  auto host_context_holder =
      KernelRunContextFaker()
          .KernelIONum(4, 1)
          .Inputs({reinterpret_cast<void *>(kOnDeviceHbm), reinterpret_cast<void *>(allocators.get()),
                   reinterpret_cast<void *>(created_allocator.get())})
          .Build();
  auto host_context = host_context_holder.GetContext<KernelContext>();
  ASSERT_NE(registry.FindKernelFuncs("SelectL1Allocator")->outputs_creator, nullptr);
  ASSERT_EQ(registry.FindKernelFuncs("SelectL1Allocator")->run_func(host_context), ge::GRAPH_SUCCESS);
  ASSERT_EQ(host_context->GetOutput(0)->GetValue<ge::Allocator *>(), created_allocator.get());
}
TEST_F(MemoryKernelUT, SelectL1Allocator_Failed_PlacementInvalid) {
  std::unique_ptr<Allocators> allocators(CreateDefaultAllocators().release());
  std::unique_ptr<memory::CachingMemAllocator> created_allocator =
      std::make_unique<memory::CachingMemAllocator>(0, RT_MEMORY_HBM);
  auto host_context_holder =
      KernelRunContextFaker()
          .KernelIONum(4, 1)
          .Inputs({reinterpret_cast<void *>(TensorPlacement::kTensorPlacementEnd),
                   reinterpret_cast<void *>(allocators.get()), reinterpret_cast<void *>(created_allocator.get())})
          .Build();
  auto host_context = host_context_holder.GetContext<KernelContext>();
  ASSERT_EQ(registry.FindKernelFuncs("SelectL1Allocator")->run_func(host_context), ge::PARAM_INVALID);
}
TEST_F(MemoryKernelUT, CreateAllocator_Success_HostOutput) {
  auto host_context_holder =
      KernelRunContextFaker()
          .KernelIONum(2, 1)
          .Inputs({reinterpret_cast<void *>(kOnHost), reinterpret_cast<void *>(kernel::MemoryType::kNodeOutput)})
          .Build();
  auto host_context = host_context_holder.GetContext<KernelContext>();
  ASSERT_EQ(registry.FindKernelFuncs("CreateL1Allocator")->run_func(host_context), ge::GRAPH_SUCCESS);
}
TEST_F(MemoryKernelUT, CreateAllocator_Failed_PlacementInvalid) {
  auto unknown_context_holder = KernelRunContextFaker()
                                    .KernelIONum(2, 1)
                                    .Inputs({reinterpret_cast<void *>(kTensorPlacementEnd),
                                             reinterpret_cast<void *>(kernel::MemoryType::kNodeOutput)})
                                    .Build();
  auto unknown_context = unknown_context_holder.GetContext<KernelContext>();
  ASSERT_EQ(registry.FindKernelFuncs("CreateL1Allocator")->run_func(unknown_context), ge::PARAM_INVALID);
}

TEST_F(MemoryKernelUT, CreateAndRecycleFftsMem) {
  // fake allocator
  memory::CachingMemAllocator allocator(0, RT_MEMORY_HBM);

  ASSERT_NE(registry.FindKernelFuncs("CreateFftsMemAllocator"), nullptr);
  ASSERT_NE(registry.FindKernelFuncs("CreateFftsMemAllocator")->run_func, nullptr);

  ASSERT_NE(registry.FindKernelFuncs("RecycleFftsMems"), nullptr);
  ASSERT_NE(registry.FindKernelFuncs("RecycleFftsMems")->run_func, nullptr);

  auto create_context_holder =
      KernelRunContextFaker().KernelIONum(2, 1).Inputs({&allocator, reinterpret_cast<void *>(0x2)}).Build();
  auto create_context = create_context_holder.GetContext<KernelContext>();

  ASSERT_EQ(registry.FindKernelFuncs("CreateFftsMemAllocator")->run_func(create_context), ge::GRAPH_SUCCESS);

  auto ffts_holder = create_context->GetOutputPointer<memory::FftsMemAllocator *>(0U);
  ASSERT_NE(ffts_holder, nullptr);
  ASSERT_NE(*ffts_holder, nullptr);

  ASSERT_EQ(registry.FindKernelFuncs("CreateFftsMemAllocator")->run_func(create_context), ge::GRAPH_SUCCESS);
  auto ffts_holder2 = create_context->GetOutputPointer<memory::FftsMemAllocator *>(0U);
  ASSERT_NE(ffts_holder2, nullptr);
  ASSERT_EQ(*ffts_holder, *ffts_holder2);

  // recycle
  auto recycle_context_holder = KernelRunContextFaker().KernelIONum(1, 1).Inputs({*ffts_holder}).Build();
  auto recycle_context = recycle_context_holder.GetContext<KernelContext>();
  ASSERT_EQ(registry.FindKernelFuncs("RecycleFftsMems")->run_func(recycle_context), ge::GRAPH_SUCCESS);
}

TEST_F(MemoryKernelUT, AllocAndFreeFftsMem) {
  // fake allocator
  auto level_1_allocator = memory::CachingMemAllocator::GetAllocator();
  ASSERT_NE(level_1_allocator, nullptr);
  auto level_2_allocator = memory::FftsMemAllocator::GetAllocator(*level_1_allocator, 2U);
  ASSERT_NE(level_2_allocator, nullptr);

  // allocate & free
  ASSERT_NE(registry.FindKernelFuncs("AllocateFftsMem"), nullptr);
  ASSERT_NE(registry.FindKernelFuncs("AllocateFftsMem")->run_func, nullptr);

  ASSERT_NE(registry.FindKernelFuncs("FreeFftsMem"), nullptr);
  ASSERT_NE(registry.FindKernelFuncs("FreeFftsMem")->run_func, nullptr);

  auto allocate_context_holder = KernelRunContextFaker()
                                     .KernelIONum(2, 1)
                                     .Inputs({level_2_allocator.get(), reinterpret_cast<void *>(0x10)})
                                     .Build();
  auto allocate_context = allocate_context_holder.GetContext<KernelContext>();
  ASSERT_EQ(registry.FindKernelFuncs("AllocateFftsMem")->run_func(allocate_context), ge::GRAPH_SUCCESS);

  auto block_holder = allocate_context->GetOutputPointer<memory::FftsMemBlock *>(0UL);
  ASSERT_NE(block_holder, nullptr);
  ASSERT_NE(*block_holder, nullptr);

  auto free_context_holder = KernelRunContextFaker().KernelIONum(1, 0).Inputs({*block_holder}).Build();
  auto free_context = free_context_holder.GetContext<KernelContext>();
  ASSERT_EQ(registry.FindKernelFuncs("FreeFftsMem")->run_func(free_context), ge::GRAPH_SUCCESS);

  // allocate & free batch
  ASSERT_NE(registry.FindKernelFuncs("AllocateBatchFftsMems"), nullptr);
  ASSERT_NE(registry.FindKernelFuncs("AllocateBatchFftsMems")->run_func, nullptr);
  ASSERT_NE(registry.FindKernelFuncs("AllocateBatchFftsMems")->outputs_creator, nullptr);

  ASSERT_NE(registry.FindKernelFuncs("FreeBatchFftsMems"), nullptr);
  ASSERT_NE(registry.FindKernelFuncs("FreeBatchFftsMems")->run_func, nullptr);

  auto sizes_holder = ContinuousVector::Create<size_t>(8UL);
  auto sizes = reinterpret_cast<TypedContinuousVector<size_t> *>(sizes_holder.get());
  sizes->SetSize(2UL);
  sizes->MutableData()[0UL] = 128UL;
  sizes->MutableData()[1UL] = 128UL;

  auto batch_alloc_context_holder =
      KernelRunContextFaker().KernelIONum(2, 1).Inputs({level_2_allocator.get(), sizes_holder.get()}).Build();
  auto batch_alloc_context = batch_alloc_context_holder.GetContext<KernelContext>();
  ASSERT_EQ(registry.FindKernelFuncs("AllocateBatchFftsMems")->outputs_creator(nullptr, batch_alloc_context),
            ge::GRAPH_SUCCESS);
  ASSERT_EQ(registry.FindKernelFuncs("AllocateBatchFftsMems")->run_func(batch_alloc_context), ge::GRAPH_SUCCESS);

  auto blocks = batch_alloc_context->GetOutputPointer<TypedContinuousVector<memory::FftsMemBlock *>>(0UL);
  ASSERT_NE(blocks, nullptr);
  ASSERT_GT(blocks->GetCapacity(), 2);
  ASSERT_NE(blocks->MutableData()[0], nullptr);
  ASSERT_NE(blocks->MutableData()[1], nullptr);

  auto batch_free_context_holder = KernelRunContextFaker().KernelIONum(1, 0).Inputs({blocks}).Build();
  auto batch_free_context = batch_free_context_holder.GetContext<KernelContext>();
  ASSERT_EQ(registry.FindKernelFuncs("FreeBatchFftsMems")->run_func(batch_free_context), ge::GRAPH_SUCCESS);

  level_2_allocator->Recycle();
  level_1_allocator->Finalize();
}

TEST_F(MemoryKernelUT, GetAllocator_GetAllocatorSuccess) {
  std::unique_ptr<Allocators> allocators(CreateDefaultAllocators().release());
  auto context_node_output =
      KernelRunContextFaker()
          .KernelIONum(2, 1)
          .Inputs({reinterpret_cast<void *>(kOnDeviceHbm), reinterpret_cast<void *>(allocators.get())})
          .Build();
  auto context1 = context_node_output.GetContext<KernelContext>();
  ASSERT_NE(registry.FindKernelFuncs("GetExternalL1Allocator")->outputs_creator, nullptr);
  ASSERT_EQ(registry.FindKernelFuncs("GetExternalL1Allocator")->run_func(context1), ge::GRAPH_SUCCESS);
  auto out_allocator = context1->GetOutput(0)->GetValue<ge::Allocator *>();

  ASSERT_NE(out_allocator, nullptr);
}

TEST_F(MemoryKernelUT, GetAllocator_GetAllocatorFailed) {
  std::unique_ptr<Allocators> allocators1(nullptr);
  auto context_node_output =
      KernelRunContextFaker()
          .KernelIONum(2, 1)
          .Inputs({reinterpret_cast<void *>(kOnDeviceHbm), reinterpret_cast<void *>(allocators1.get())})
          .Build();
  auto context = context_node_output.GetContext<KernelContext>();
  ASSERT_NE(registry.FindKernelFuncs("GetExternalL1Allocator")->outputs_creator, nullptr);
  ASSERT_NE(registry.FindKernelFuncs("GetExternalL1Allocator")->run_func(context), ge::GRAPH_SUCCESS);
  auto chain = context->GetOutput(0);
  chain->Set(nullptr, nullptr);
}

TEST_F(MemoryKernelUT, GetAllocator_GetExternalAllocatorFailed) {
  std::unique_ptr<Allocators> allocators1(nullptr);
  auto context_node_output =
      KernelRunContextFaker()
          .KernelIONum(2, 1)
          .Inputs({reinterpret_cast<void *>(kOnDeviceHbm), reinterpret_cast<void *>(allocators1.get())})
          .Build();
  auto context = context_node_output.GetContext<KernelContext>();
  ASSERT_NE(registry.FindKernelFuncs("GetExternalL1Allocator")->outputs_creator, nullptr);
  ASSERT_NE(registry.FindKernelFuncs("GetExternalL1Allocator")->run_func(context), ge::GRAPH_SUCCESS);
  auto chain = context->GetOutput(0);
  chain->Set(nullptr, nullptr);
}

TEST_F(MemoryKernelUT, AccessMemCrossStream_Success) {
  auto allocators = memory::MultiStreamAllocatorFaker().StreamNum(3).Build();
  auto td = allocators.Alloc(0, 1024);
  auto context = KernelRunContextFaker().KernelIONum(2, 1).Inputs({&td, (void *)1}).Build();

  ASSERT_EQ(registry.FindKernelFuncs("AccessMemCrossStream")->outputs_creator(nullptr, context), ge::GRAPH_SUCCESS);
  GertRuntimeStub runtime_stub;
  runtime_stub.GetSlogStub().NoConsoleOut().SetLevelInfo();
  ASSERT_EQ(registry.FindKernelFuncs("AccessMemCrossStream")->run_func(context), ge::GRAPH_SUCCESS);
  EXPECT_TRUE(runtime_stub.GetSlogStub().FindInfoLogRegex(kWander, {{2, "0"}, {3, "1"}, {4, ToHex(td.GetAddr())}}) >=
              0);

  auto dst_td = context.GetContext<KernelContext>()->GetOutputPointer<GertTensorData>(0);
  ASSERT_NE(dst_td, nullptr);
  EXPECT_EQ(dst_td->GetStreamId(), 1);
  EXPECT_EQ(dst_td->GetAddr(), td.GetAddr());
  EXPECT_EQ(dst_td->GetPlacement(), td.GetPlacement());
  EXPECT_EQ(dst_td->GetSize(), td.GetSize());
  dst_td->Free();
  reinterpret_cast<memory::MultiStreamMemBlock *>(td.GetGertMemBlock())->SyncLocalRecycleStatus(1, 0);
}
}  // namespace gert
