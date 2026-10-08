/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_CXX_BASE_COMMON_OM2_CODEGEN_TASK_CODE_BUILDER_TASK_CODE_BUILDER_UTIL_H_
#define AIR_CXX_BASE_COMMON_OM2_CODEGEN_TASK_CODE_BUILDER_TASK_CODE_BUILDER_UTIL_H_

#include "aprof_pub.h"
#include "common/om2/codegen/ast/ast_build_context.h"
#include "common/om2/codegen/om2_codegen_types.h"

namespace ge {

class TaskCodeBuilderUtil {
 public:
  static Expr *BuildTaskIoEntries(AstBuildContext &ast, const std::vector<AddrSemantic> &addrs);
  static Expr *BuildWorkspaceAddrs(AstBuildContext &ast, const std::vector<AddrSemantic> &addrs);
  static Expr *BuildWorkspaceSizes(AstBuildContext &ast, const std::vector<AddrSemantic> &addrs);
  static Expr *BuildL0ArgSlotEntries(AstBuildContext &ast, const std::vector<AddrSemantic> &ordered_args);
  static Status AppendReportLaunchedTaskCall(AstBuildContext &ast, std::vector<BodyItem> &items,
                                             const std::string &var_prefix, const TaskSemanticHeader &header,
                                             const ArgsTableEntrySemantic *args_table_entry,
                                             const std::vector<AddrSemantic> &input_addrs,
                                             const std::vector<AddrSemantic> &output_addrs,
                                             const std::vector<AddrSemantic> &workspace_addrs, ModelTaskType task_type,
                                             uint32_t block_dim, Arg stream, const VarRef &model_id,
                                             const VarRef &instance_handle, const VarRef &args_table,
                                             bool use_args_info_size, bool is_raw_address = false);
  // 将 case body 包装为独立的 dispatch 函数并添加到 items
  static Status RenderDispatchFunc(AstBuildContext &ast, const std::string &func_name,
                                   const std::vector<BodyItem> &body, std::vector<DeclNode *> &items);
  static ExprRef BuildReportTaskPreprocessCall(
      AstBuildContext &ast, const TaskSemanticHeader &header, const ArgsTableEntrySemantic *args_table_entry,
      const std::vector<AddrSemantic> &input_addrs, const std::vector<AddrSemantic> &output_addrs,
      const std::vector<AddrSemantic> &workspace_addrs, ModelTaskType task_type, uint32_t block_dim, Arg stream,
      const VarRef &model_id, const VarRef &instance_handle, const VarRef &args_table, Arg l0_info,
      bool use_args_info_size, bool is_raw_address = false);
  // 将 OpArgDesc 列表转换为 OpArgInfo 数组的 AST 表达式（表驱动优化）
  static Arg RenderOpArgDesc(AstBuildContext &ast, const std::vector<OpArgDesc> &args);
  static Arg BuildTensorDataField(AstBuildContext &ast, const OpArgDesc &arg_desc);
  static Arg BuildWorkspaceDataField(AstBuildContext &ast, const OpArgDesc &arg_desc);
  static Arg BuildCustomValueDataField(AstBuildContext &ast, const OpArgDesc &arg_desc);
  static Arg BuildTilingDataField(AstBuildContext &ast, const OpArgDesc &arg_desc);
  // OpArgInfo 的 addr 字段填充：需要地址的参数取 OpArgDesc 实际值，其余填充默认值（全 0）
  static Arg RenderArgAddrField(AstBuildContext &ast, const OpArgDesc &arg_desc);
  // OpArgInfo 的 data 字段填充：有对应数据的参数按类型查表构建实际值，其余填充默认值（custom_value = 0）
  static Arg RenderArgDataField(AstBuildContext &ast, const OpArgDesc &arg_desc);
  // 将 AddrSemantic 转换为 OpArgDesc（RAW_ADDR 类型）
  static OpArgDesc ConvertAddrDesc(const AddrSemantic &addr);

  // 完整镜像 davinci_model.cc:GetProfilingTaskType() 的判别逻辑，
  // 将 ModelTaskType + op 属性 + kernel_type 转换为 profiling task type 对应的 uint32_t 值
  // @param op_desc   算子描述，用于读取 ATTR_NAME_CUBE_VECTOR_CORE_TYPE / kAttrIsFFTSTask / kAttrIsAiv
  // @param task_def   protobuf task 定义，用于获取 ModelTaskType 和 kernel context
  // @return          MsprofGeTaskType 对应的 uint32_t 值 (0~11)
  static uint32_t ConvertToProfilingTaskType(const OpDescPtr &op_desc, const domi::TaskDef &task_def);

  // 镜像 global_profiler.cc:BuildNodeBasicInfo 的 HF32 判别输入：读取 op 的 _op_impl_mode_enum 属性原值，
  // 未设置时返回 0(默认模式)，由运行时上报侧与 kEnableHf32(0x40) 比较后填 MsprofNodeBasicInfo.opFlag
  static uint32_t GetOpImplMode(const OpDescPtr &op_desc);

  // 完整镜像 davinci_model.cc:GetBlockDim() 的 profiling 加工逻辑(与 launch 用的归一 block_dim 分离)：
  // 1) 从 task_def 取未归一的原始 block_dim(launch 侧的 0→1 归一不适用于上报口径)；
  // 2) FFTS+ mix 算子编码：低16位为主加速器 blockdim，高16位为从加速器 ratio 值，由 msprof 工具解析；
  // 3) tiling sink 依赖算子返回 0xFFFFFFFF 占位值(表示运行时由 Tiling 结果决定)
  static uint32_t GetProfilingBlockDim(const OpDescPtr &op_desc, const domi::TaskDef &task_def);
};
}  // namespace ge

#endif  // AIR_CXX_BASE_COMMON_OM2_CODEGEN_TASK_CODE_BUILDER_TASK_CODE_BUILDER_UTIL_H_
