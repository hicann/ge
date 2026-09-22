/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software: you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * @file custom_op.cpp
 * @brief Eager 自定义算子辅流申请样例
 *
 * 演示 EagerOpExecutionContext::RequestAttachedStream 的使用方法：
 * 在 Execute 回调（模型加载/下沉阶段执行一次）中按 key 申请框架托管的物理辅流，
 * 用 event 与主流做执行顺序同步，并把 add kernel 下发到辅流（而非主流）上，
 * 模型重放时辅流上的 task 重复执行。
 */

#include <algorithm>
#include "acl/acl_rt.h"
#include "graph/custom_op.h"
#include "exe_graph/runtime/eager_op_execution_context.h"
#include "add_custom.h"
#include "add_custom_kernel.h"
#include "utils/rtc_kernel_loader.h"
#include "utils/log.h"

namespace {
constexpr size_t kInputX = 0U;
constexpr size_t kInputY = 1U;
constexpr size_t kOutputZ = 0U;
constexpr uint32_t kMaxBlocks = 65535;
constexpr const char *kKernelSourceFile = "add_custom.asc";

// 辅流复用 key：同一模型内相同 key 共享同一条物理辅流
constexpr const char *kAuxStreamKey = "eager_aux";

struct __attribute__((packed)) AddArgs {
  const void *x_ptr __attribute__((aligned(8)));
  const void *y_ptr __attribute__((aligned(8)));
  void *z_ptr __attribute__((aligned(8)));
};

RtcKernelLoader g_kernel_loader;

ge::graphStatus LoadKernel() {
  return g_kernel_loader.Load(kAddCustomKernelName, kKernelSourceFile);
}

uint32_t CalcNumBlocks(uint32_t n_elements) {
  return std::min((n_elements + kAddCustomBlockSize - 1U) / kAddCustomBlockSize, kMaxBlocks);
}
}  // namespace

namespace ge {

/**
 * @brief Eager 辅流 Add 算子
 *
 * 继承 EagerExecuteOp + ShapeInferOp。
 * Execute 在 V1 下沉阶段（模型加载）调用一次，之后模型重放不再进入 Execute。
 *
 * 辅流使用链路：
 *   ctx->RequestAttachedStream(key) → 返回框架托管的物理辅流
 *   → aclrtLaunchKernelV2(..., 辅流) → kernel 任务下沉到辅流
 */
class EagerAttachedStreamAddOp : public EagerExecuteOp, public ShapeInferOp {
 public:
  graphStatus Execute(gert::EagerOpExecutionContext *ctx) override {
    if (LoadKernel() != GRAPH_SUCCESS) {
      LOG_ERROR("LoadKernel failed");
      return GRAPH_FAILED;
    }

    const gert::Tensor *input_x = ctx->GetInputTensor(kInputX);
    const gert::Tensor *input_y = ctx->GetInputTensor(kInputY);
    if (input_x == nullptr || input_y == nullptr) {
      LOG_ERROR("GetInputTensor failed, x=", input_x, " y=", input_y);
      return GRAPH_FAILED;
    }

    gert::Tensor *output_z =
        ctx->MallocOutputTensor(kOutputZ, input_x->GetShape(), input_x->GetFormat(), input_x->GetDataType());
    if (output_z == nullptr) {
      LOG_ERROR("MallocOutputTensor failed");
      return GRAPH_FAILED;
    }

    // 按 key 申请框架托管的辅流，后续 kernel 下发到该辅流而非主流
    const gert::rtStream aux_stream = ctx->RequestAttachedStream(kAuxStreamKey);
    if (aux_stream == nullptr) {
      LOG_ERROR("RequestAttachedStream(\"", kAuxStreamKey, "\") returned nullptr");
      return GRAPH_FAILED;
    }

    // 主辅流 event 同步：辅流与主流并发启动（HEAD 绑定），执行顺序必须由算子用 event 保证
    //   主流： [输入搬运(Identity)] → record(ev_in) → wait(ev_out) → [输出搬运(Identity)]
    //   辅流： wait(ev_in) → kernel → record(ev_out)
    const auto main_stream = static_cast<aclrtStream>(ctx->GetStream());
    if (EnsureSyncEvents() != GRAPH_SUCCESS || SyncMainToAux(main_stream, aux_stream) != GRAPH_SUCCESS) {
      return GRAPH_FAILED;
    }

    // kernel 下发到辅流（而非主流 ctx->GetStream()），完成后主流等待 ev_out
    const uint32_t num_blocks = CalcNumBlocks(static_cast<uint32_t>(input_x->GetShapeSize()));
    if (LaunchAddKernel(*input_x, *input_y, *output_z, aux_stream, num_blocks) != GRAPH_SUCCESS) {
      return GRAPH_FAILED;
    }
    if (SyncAuxToMain(main_stream, aux_stream) != GRAPH_SUCCESS) {
      return GRAPH_FAILED;
    }

    LOG_INFO("[AttachedStream] kernel launched on attached stream ", aux_stream, ", blocks=", num_blocks);
    return GRAPH_SUCCESS;
  }

  graphStatus InferShape(gert::InferShapeContext *ctx) override {
    const auto *input_shape = ctx->GetInputShape(kInputX);
    auto *output_shape = ctx->GetOutputShape(kOutputZ);
    if (input_shape == nullptr || output_shape == nullptr) {
      LOG_ERROR("InferShape failed");
      return GRAPH_FAILED;
    }
    output_shape->SetDimNum(input_shape->GetDimNum());
    for (size_t i = 0U; i < input_shape->GetDimNum(); ++i) {
      output_shape->SetDim(i, input_shape->GetDim(i));
    }
    return GRAPH_SUCCESS;
  }

  graphStatus InferDataType(gert::InferDataTypeContext *ctx) override {
    return ctx->SetOutputDataType(kOutputZ, ctx->GetInputDataType(kInputX));
  }

 private:
  /**
   * @brief 首次 Execute 时创建主辅流同步 event，之后跨模型重放复用
   *
   * event 任务在模型构建期录制，每次模型重放按此顺序执行。
   * 注：event 缓存于算子实例（CustomOpRegistry 按 op_type 缓存单例），本样例假设单模型使用同一算子类型。
   */
  graphStatus EnsureSyncEvents() {
    if (ev_main_to_aux_ != nullptr) {
      return GRAPH_SUCCESS;
    }
    // 与 GE 模型 event 相同的 flag（davinci_model.cc）：
    // RT_EVENT_DEFAULT 类型 event 不允许在绑定模型的流上 record（RTS 返回 207000），
    // 必须用 ACL_EVENT_SYNC | CAPTURE_STREAM_PROGRESS | TIME_LINE 组合。
    constexpr uint32_t kEventFlags = ACL_EVENT_SYNC | ACL_EVENT_CAPTURE_STREAM_PROGRESS | ACL_EVENT_TIME_LINE;
    aclError ev_ret = aclrtCreateEventWithFlag(&ev_main_to_aux_, kEventFlags);
    if (ev_ret != ACL_ERROR_NONE) {
      LOG_ERROR("aclrtCreateEventWithFlag(ev_in) failed, ret=", ev_ret);
      return GRAPH_FAILED;
    }
    ev_ret = aclrtCreateEventWithFlag(&ev_aux_to_main_, kEventFlags);
    if (ev_ret != ACL_ERROR_NONE) {
      LOG_ERROR("aclrtCreateEventWithFlag(ev_out) failed, ret=", ev_ret);
      return GRAPH_FAILED;
    }
    return GRAPH_SUCCESS;
  }

  // (a) 主流录制 ev_in：位于输入搬运任务之后；(b) 辅流等待 ev_in：保证 kernel 在输入搬运完成后执行
  graphStatus SyncMainToAux(aclrtStream main_stream, gert::rtStream aux_stream) {
    aclError ret = aclrtRecordEvent(ev_main_to_aux_, main_stream);
    if (ret != ACL_ERROR_NONE) {
      LOG_ERROR("aclrtRecordEvent(ev_in, main) failed, ret=", ret);
      return GRAPH_FAILED;
    }
    ret = aclrtStreamWaitEvent(static_cast<aclrtStream>(aux_stream), ev_main_to_aux_);
    if (ret != ACL_ERROR_NONE) {
      LOG_ERROR("aclrtStreamWaitEvent(aux, ev_in) failed, ret=", ret);
      return GRAPH_FAILED;
    }
    return GRAPH_SUCCESS;
  }

  // (c) 辅流录制 ev_out：kernel 完成后；(d) 主流等待 ev_out：保证输出搬运在 kernel 完成后执行
  graphStatus SyncAuxToMain(aclrtStream main_stream, gert::rtStream aux_stream) {
    aclError ret = aclrtRecordEvent(ev_aux_to_main_, static_cast<aclrtStream>(aux_stream));
    if (ret != ACL_ERROR_NONE) {
      LOG_ERROR("aclrtRecordEvent(ev_out, aux) failed, ret=", ret);
      return GRAPH_FAILED;
    }
    ret = aclrtStreamWaitEvent(main_stream, ev_aux_to_main_);
    if (ret != ACL_ERROR_NONE) {
      LOG_ERROR("aclrtStreamWaitEvent(main, ev_out) failed, ret=", ret);
      return GRAPH_FAILED;
    }
    return GRAPH_SUCCESS;
  }

  // 分配 device args 并把 add kernel 下发到辅流——注意使用辅流而非 ctx->GetStream()
  graphStatus LaunchAddKernel(const gert::Tensor &input_x, const gert::Tensor &input_y, const gert::Tensor &output_z,
                              gert::rtStream aux_stream, uint32_t num_blocks) {
    AddArgs args = {input_x.GetAddr(), input_y.GetAddr(), const_cast<void *>(output_z.GetAddr())};

    void *dev_args = nullptr;
    aclError ret = aclrtMalloc(&dev_args, sizeof(args), ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != ACL_ERROR_NONE) {
      LOG_ERROR("aclrtMalloc failed, error: ", ret);
      return GRAPH_FAILED;
    }
    ret = aclrtMemcpy(dev_args, sizeof(args), &args, sizeof(args), ACL_MEMCPY_HOST_TO_DEVICE);
    if (ret != ACL_ERROR_NONE) {
      LOG_ERROR("aclrtMemcpy failed, error: ", ret);
      aclrtFree(dev_args);
      return GRAPH_FAILED;
    }

    ret = aclrtLaunchKernelV2(g_kernel_loader.GetFuncHandle(), num_blocks, dev_args, sizeof(args), nullptr,
                              static_cast<aclrtStream>(aux_stream));
    if (ret != ACL_ERROR_NONE) {
      LOG_ERROR("aclrtLaunchKernelV2 on attached stream failed, error: ", ret);
      aclrtFree(dev_args);
      return GRAPH_FAILED;
    }
    return GRAPH_SUCCESS;
  }

  // 主辅流同步 event（首次 Execute 创建，跨模型重放复用；单模型假设见 EnsureSyncEvents 注释）
  aclrtEvent ev_main_to_aux_{nullptr};
  aclrtEvent ev_aux_to_main_{nullptr};
};

REG_AUTO_MAPPING_OP(EagerAttachedStreamAddOp);

}  // namespace ge
