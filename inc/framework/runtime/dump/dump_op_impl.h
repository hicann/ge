/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GE_FRAMEWORK_RUNTIME_DUMP_DUMP_OP_IMPL_H_
#define GE_FRAMEWORK_RUNTIME_DUMP_DUMP_OP_IMPL_H_

#include <map>
#include <string>
#include "framework/common/ge_types.h"
#include "framework/runtime/dump/model_dump_c_api.h"
#include "proto/op_mapping.pb.h"

namespace ge {
namespace dump {

// ========================================================================
// DumpOp 类，自定义算子的 dump 使用
// ========================================================================
class DumpOp {
 public:
  DumpOp() = default;
  ~DumpOp();
  Status ExecutorDumpOp(const std::string &op_name, aclrtStream stream);
  Status BuildTaskInputs(const GertModelTaskDesc &task_desc);
  Status BuildTaskOutputs(const GertModelTaskDesc &task_desc);
  static Status SetTaskBasicInfo(const GertModelTaskDesc &task_desc, toolkit::aicpu::dump::Task *task);
  toolkit::aicpu::dump::OpMappingInfo &GetOpMappingInfo() {
    return op_mapping_info_;
  }

 private:
  Status ProtoMallocAndMemcpy(const size_t proto_size, const std::string &proto_msg);

  void *proto_dev_mem_ = nullptr;
  size_t proto_dev_mem_capacity_ = 0U;
  void *proto_size_dev_mem_ = nullptr;
  toolkit::aicpu::dump::OpMappingInfo op_mapping_info_;
};

using DumpOpPtr = std::unique_ptr<DumpOp>;

class DumpOpImpl {
 public:
  DumpOpPtr &GetInputDumpOp(const std::string &op_name) {
    return GetDumpOp(op_name, true);
  }

  DumpOpPtr &GetOutputDumpOp(const std::string &op_name) {
    return GetDumpOp(op_name, false);
  }

 private:
  DumpOpPtr &GetDumpOp(const std::string &op_name, bool is_input) {
    auto &item = custom_dump_ops_[op_name];
    auto &dump_op = is_input ? item.first : item.second;
    if (dump_op == nullptr) {
      dump_op.reset(new DumpOp());
    }
    dump_op->GetOpMappingInfo().clear_task();
    return dump_op;
  }
  std::map<std::string, std::pair<DumpOpPtr, DumpOpPtr>> custom_dump_ops_;
};

}  // namespace dump
}  // namespace ge

#endif  // GE_FRAMEWORK_RUNTIME_DUMP_DUMP_OP_IMPL_H_
