/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph/load/model_manager/task_info/rts/cmo_addr_task_info.h"

#include "runtime/mem.h"
#include "graph/load/model_manager/davinci_model.h"
#include "graph/load/model_manager/model_utils.h"

namespace ge {
constexpr uint32_t kMaxPrefetchLen = 120U * 1024U * 1024U;
constexpr uint64_t kAlignedBytes = 64U;
constexpr char_t const *kAttrMaxSize = "max_size";
constexpr char_t const *kAttrAddrOffset = "offset";

Status CmoAddrTaskInfo::ParseTaskRunParam(const domi::TaskDef &task_def, DavinciModel *const davinci_model,
                                          TaskRunParam &task_run_param) {
  const auto &cmo_addr_task = task_def.cmo_addr_task();
  op_desc_ = davinci_model->GetOpByIndex(cmo_addr_task.op_index());
  GE_CHECK_NOTNULL(op_desc_);

  const RuntimeParam &rts_param = davinci_model->GetRuntimeParam();
  std::vector<uint64_t> mem_type;
  std::vector<uint64_t> input_addrs = ModelUtils::GetInputAddrsValue(rts_param, op_desc_, mem_type);
  GE_ASSERT_TRUE(input_addrs.size() == 1UL, "Input_addr size [%zu] is invalid, op: %s", input_addrs.size(),
                 op_desc_->GetNamePtr());
  GE_ASSERT_TRUE(mem_type.size() == 1UL, "Input_addr size [%zu] is invalid, op: %s", mem_type.size(),
                 op_desc_->GetNamePtr());
  args_size_ = sizeof(rtCmoAddrInfo) + kAlignedBytes;
  task_run_param.parsed_input_addrs.push_back({input_addrs[0UL], mem_type[0UL], true, {0}});
  const uint32_t args_mem_type = rtGetTsMemType(MEM_REQUEST_FEATURE_DEFAULT, static_cast<uint32_t>(args_size_));
  args_placement_ =
      ((args_mem_type & RT_MEMORY_TS) == 0U) ? ArgsPlacement::kArgsPlacementHbm : ArgsPlacement::kArgsPlacementTs;
  GELOGI("args mem type:%u, args size:%" PRIu64 ", args placement:%d", args_mem_type, args_size_, args_placement_);
  task_run_param.args_descs.push_back({static_cast<int64_t>(args_size_), args_placement_});

  return SUCCESS;
}

Status CmoAddrTaskInfo::Init(const domi::TaskDef &task_def, DavinciModel *const davinci_model, const PisToArgs &args,
                             const PisToPersistentWorkspace &persistent_workspace, const IowAddrs &iow_addrs) {
  (void)persistent_workspace;
  GE_CHECK_NOTNULL(davinci_model);
  davinci_model_ = davinci_model;
  GE_CHK_STATUS_RET_NOLOG(SetStream(task_def.stream_id(), davinci_model_->GetStreamList()));
  GE_CHECK_NOTNULL(op_desc_);

  const uint64_t args_base = args[static_cast<size_t>(args_placement_)].dev_addr;
  GE_ASSERT_TRUE((args_base != 0UL), "[Check][Param] Op:%s, args_placement:%d, dev addr is nullptr.",
                 op_desc_->GetNamePtr(), args_placement_);
  const uint64_t align_addr = ((args_base / kAlignedBytes) + 1U) * kAlignedBytes;
  const size_t aligned_size = static_cast<size_t>(align_addr - args_base);

  args_ = ValueToPtr(align_addr);
  GE_ASSERT(static_cast<size_t>(args[static_cast<size_t>(args_placement_)].len) >= args_size_);
  const uint64_t host_base = PtrToValue(args[static_cast<size_t>(args_placement_)].host_addr);
  addr_info_ = reinterpret_cast<rtCmoAddrInfo *>(host_base + aligned_size);
  GE_ASSERT_NOTNULL(addr_info_);
  const auto &cmo_addr_task = task_def.cmo_addr_task();
  addr_info_->resv0 = cmo_addr_task.resv0();
  addr_info_->resv1 = cmo_addr_task.resv1();
  addr_info_->num_inner = static_cast<uint16_t>(cmo_addr_task.num_inner());
  addr_info_->num_outer = static_cast<uint16_t>(cmo_addr_task.num_outer());
  addr_info_->stride_inner = cmo_addr_task.stride_inner();
  addr_info_->stride_outer = cmo_addr_task.stride_outer();

  const GeTensorDesc &tensor_desc = op_desc_->GetInputDesc(0U);
  int64_t num_cnt = tensor_desc.GetShape().IsScalar() ? 1 : tensor_desc.GetShape().GetShapeSize();
  int64_t shape_len = GetSizeInBytes(num_cnt, tensor_desc.GetDataType());
  GE_ASSERT_TRUE(shape_len > 0);
  int64_t offset{0};
  (void)AttrUtils::GetInt(op_desc_, kAttrAddrOffset, offset);
  GELOGD("[%s] got offset [%" PRId64 "], logic_addr:[%" PRIu64 "] size:[%" PRId64 "]", op_desc_->GetNamePtr(), offset,
         iow_addrs.input_logic_addrs[0U].logic_addr, shape_len);
  if ((offset < 0) || (offset >= shape_len)) {
    REPORT_INNER_ERR_MSG("E19999", "The offset %" PRId64 " should be within the range of [0, %" PRId64 ").", offset,
                       shape_len);
    GELOGE(ge::PARAM_INVALID, "The offset [%" PRId64 "] should be within the range of [0, %" PRId64 ").",
      offset, shape_len);
    return ge::PARAM_INVALID;
  }
  shape_len -= offset;

  uint32_t max_size{0U};
  (void)AttrUtils::GetInt(op_desc_, kAttrMaxSize, max_size);
  if (max_size == 0) {
    max_size = kMaxPrefetchLen;
  }
  addr_info_->len_inner = std::min(static_cast<uint32_t>(shape_len), max_size);
  io_align_offset_ = static_cast<size_t>(PtrToValue(&addr_info_->src) - host_base);
  cmo_op_code_ = static_cast<rtCmoOpCode_t>(cmo_addr_task.cmo_op_code());

  GE_ASSERT_TRUE(iow_addrs.input_logic_addrs.size() == 1UL, "Input_addr size [%zu] is invalid, op: %s",
                 iow_addrs.input_logic_addrs.size(), op_desc_->GetNamePtr());

  io_addrs_.emplace_back(iow_addrs.input_logic_addrs[0U].logic_addr + offset);
  io_addr_mem_types_.emplace_back(iow_addrs.input_logic_addrs[0U].memory_type);

  GELOGI("CmoAddrTaskInfo init successfully, op [%s], args:[%p], inner_len:[%u] max prefetch len:[%u].",
         op_desc_->GetNamePtr(), args_, addr_info_->len_inner, max_size);

  davinci_model_->SetZeroCopyAddr(op_desc_, io_addrs_, io_addrs_.data(), static_cast<uintptr_t>(args_base),
                                  sizeof(uint64_t), io_align_offset_, {});

  GE_ASSERT_SUCCESS(args_io_addrs_updater_.Init(davinci_model_->GetLogicalMemAllocation(), io_addrs_,
                    io_addr_mem_types_, {op_desc_->GetName(), op_desc_->GetType()}),
                    "args io addrs updater init failed.");

  GELOGI("CmoAddrTaskInfo Init Success, logic stream id: %u, stream: %p", task_def.stream_id(), stream_);
  return SUCCESS;
}

Status CmoAddrTaskInfo::UpdateHostArgs(const std::vector<uint64_t> &active_mem_base_addr, void *const host_args,
                                       const size_t host_args_max_len) {
  GE_ASSERT_SUCCESS(args_io_addrs_updater_.SetArgIoAddrs(active_mem_base_addr,
                                                         ValueToPtr(PtrToValue(host_args) + io_align_offset_),
                                                         static_cast<size_t>(host_args_max_len - io_align_offset_)));
  GELOGI("CmoAddrTaskInfo::UpdateArgs success.");
  return SUCCESS;
}

Status CmoAddrTaskInfo::GetTaskArgsRefreshInfos(std::vector<TaskArgsRefreshInfo> &infos) {
  args_io_addrs_updater_.GenArgsRefreshInfos(infos, static_cast<uint64_t>(io_align_offset_), args_placement_);
  return SUCCESS;
}

Status CmoAddrTaskInfo::Distribute() {
  GE_ASSERT_NOTNULL(op_desc_);
  GELOGI("CmoAddrTaskInfo Distribute Start, op:[%s].", op_desc_->GetNamePtr());
  SetTaskTag(op_desc_->GetNamePtr());
  GE_CHK_RT_RET(rtCmoAddrTaskLaunch(args_, sizeof(rtCmoAddrInfo), cmo_op_code_, stream_, 0U));
  // refresh sqe from rts
  rtCmoAddrInfo rts_sqe;
  GE_CHK_RT_RET(rtMemcpy(&rts_sqe, sizeof(rtCmoAddrInfo), args_, sizeof(rtCmoAddrInfo), RT_MEMCPY_DEVICE_TO_HOST));
  GE_ASSERT_NOTNULL(addr_info_);
  addr_info_->resv0 = rts_sqe.resv0;
  addr_info_->resv1 = rts_sqe.resv1;

  is_support_redistribute_ = true;
  GELOGI("CmoAddrTaskInfo Distribute Success, stream: %p.", stream_);
  return SUCCESS;
}

int64_t CmoAddrTaskInfo::ParseOpIndex(const domi::TaskDef &task_def) const {
  const domi::CmoAddrTaskDef &cmo_addr_task = task_def.cmo_addr_task();
  return static_cast<int64_t>(cmo_addr_task.op_index());
}

REGISTER_TASK_INFO(MODEL_TASK_CMO_ADDR, CmoAddrTaskInfo);
}  // namespace ge
