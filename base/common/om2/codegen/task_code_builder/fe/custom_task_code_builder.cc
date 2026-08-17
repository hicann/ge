/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "custom_task_code_builder.h"
#include "common/om2/codegen/om2_model_utils.h"
#include "common/om2/codegen/task_code_builder_factory.h"
#include "common/om2/codegen/task_code_builder/task_code_builder_util.h"
#include "opskernel/ops_kernel_info_types.h"
#include "graph/utils/args_format_desc_utils.h"
#include "graph/args_format_desc.h"
#include "common/checker.h"

namespace ge {
namespace {
constexpr uint32_t kAddressLen = static_cast<uint32_t>(sizeof(uint64_t));
}  // namespace

int64_t CustomTaskCodeBuilder::ParseOpIndex(const domi::TaskDef &task_def) {
  const domi::KernelDef &kernel_def = task_def.kernel();
  domi::KernelContext context = kernel_def.context();
  return static_cast<int64_t>(context.op_index());
}

Status CustomTaskCodeBuilder::Contribute(TaskSemanticContributeContext &context) {
  GE_ASSERT_SUCCESS(TaskCodeBuilder::Contribute(context));
  GE_ASSERT_NOTNULL(context.next_args_table_index);
  GE_ASSERT_NOTNULL(context.next_host_args_offset);
  GE_ASSERT_NOTNULL(context.op_desc);

  build_data_.semantic.task_type = context.task_type;
  build_data_.semantic.kernel_type = static_cast<ccKernelType>(context.task_def.kernel().context().kernel_type());

  GE_ASSERT_SUCCESS(Om2ModelUtils::ResolveWorkspaceAddrs(context, build_data_.semantic.workspace_addrs));
  GE_ASSERT_SUCCESS(Om2ModelUtils::ResolveInputAddrs(context, build_data_.semantic.input_addrs));
  GE_ASSERT_SUCCESS(Om2ModelUtils::ResolveOutputAddrs(context, true, build_data_.semantic.output_addrs));
  AssignTaskLocalIoNames();

  dispatch_type_ = OpDispatchType::DISPATCH_CUSTOM_KERNEL;

  // parse args format
  std::vector<ArgDesc> arg_descs;
  domi::KernelContext kernel_context = context.task_def.kernel().context();
  GE_ASSERT_SUCCESS(ArgsFormatDesc::Parse(context.op_desc, kernel_context.args_format(), arg_descs),
                    "[OM2] Formatted args [%s] parsed failed.", kernel_context.args_format().c_str());

  // calc args size
  size_t args_size = 0U;
  for (const auto &arg_desc : arg_descs) {
    (void)ArgsFormatDesc::GetArgSize(context.op_desc, arg_desc, args_size);
  }
  InitArgsTableEntry(context, args_size);

  // update values of context.next_args_table_index and context.next_host_args_offset
  if (build_data_.semantic.args_table_entry.has_value()) {
    ++(*context.next_args_table_index);
    *context.next_host_args_offset +=
        Om2ModelUtils::ArgsSizeAlign8(static_cast<size_t>(build_data_.semantic.args_table_entry->args_size));
  }

  // construct build_data_.ordered_args
  uint64_t current_args_offset = 0U;
  auto append_args = [this, &current_args_offset](const std::vector<AddrSemantic> &addrs) {
    for (const auto &addr : addrs) {
      OpArgDesc arg = TaskCodeBuilderUtil::ConvertAddrDesc(addr);
      arg.args_offset = current_args_offset;
      current_args_offset += kAddressLen;
      build_data_.ordered_args.push_back(std::move(arg));
    }
  };
  append_args(build_data_.semantic.input_addrs);
  append_args(build_data_.semantic.output_addrs);
  append_args(build_data_.semantic.workspace_addrs);

  return SUCCESS;
}

Status CustomTaskCodeBuilder::RenderDistHelper(std::vector<DeclNode *> &items) {
  auto op = ast_.Var("const TaskDispatchInfo *", "op");
  auto ctx = ast_.Var("const DispatchOpContext &", "ctx");
  GE_ASSERT_SUCCESS(RenderDispatchCustomKernel(op, ctx, items));
  return SUCCESS;
}

Status CustomTaskCodeBuilder::RenderOpDefTableFields(std::vector<std::pair<std::string, Arg>> &fields) {
  GELOGI("[OM2] BuildOpDefTable: op=%s, func_idx=%u", header_.op_name.c_str(),
         build_data_.semantic.launch.func_handle_index);
  fields.push_back({"dispatch_type", ast_.StaticCast("OpDispatchType", static_cast<int64_t>(dispatch_type_))});
  fields.push_back({"op_name", Arg::StringLiteral(header_.op_name)});

  auto custom_fields = std::vector<std::pair<std::string, Arg>>{
      {"args_info", TaskCodeBuilderUtil::RenderOpArgDesc(ast_, build_data_.ordered_args)},
      {"args_info_num", static_cast<int64_t>(build_data_.ordered_args.size())},
      {"op_type", Arg::StringLiteral(header_.op_type)},
      {"args_idx", static_cast<int64_t>(build_data_.semantic.args_table_entry->table_index)},
      {"stream_id", static_cast<uint32_t>(header_.stream_id)},
      {"task_type", static_cast<int64_t>(build_data_.semantic.task_type)},
  };
  auto custom_dispatch = ast_.DesignatedInit({{"custom", ast_.DesignatedInit(custom_fields)}});
  fields.emplace_back("dispatch_info", custom_dispatch);

  return SUCCESS;
}

std::string CustomTaskCodeBuilder::GetFuncName() const {
  return kDispatchFuncName;
}

void CustomTaskCodeBuilder::AssignTaskLocalIoNames() {
  const std::string task_prefix = "op" + std::to_string(header_.op_index);
  for (size_t i = 0U; i < build_data_.semantic.input_addrs.size(); ++i) {
    if (build_data_.semantic.input_addrs[i].tensor_info.has_value()) {
      build_data_.semantic.input_addrs[i].symbol_hint = task_prefix + "_input" + std::to_string(i);
    }
  }
  for (size_t i = 0U; i < build_data_.semantic.output_addrs.size(); ++i) {
    if (build_data_.semantic.output_addrs[i].tensor_info.has_value()) {
      build_data_.semantic.output_addrs[i].symbol_hint = task_prefix + "_output" + std::to_string(i);
    }
  }
}

void CustomTaskCodeBuilder::InitArgsTableEntry(const TaskSemanticContributeContext &context, const uint32_t args_size) {
  (void)build_data_.semantic.args_table_entry.emplace();
  build_data_.semantic.args_table_entry->table_index = *context.next_args_table_index;
  build_data_.semantic.args_table_entry->args_size = args_size;
  build_data_.semantic.args_table_entry->host_offset = *context.next_host_args_offset;
  args_table_entry_ = &(*build_data_.semantic.args_table_entry);
}

Status CustomTaskCodeBuilder::RenderDispatchCustomKernel(const VarRef &op, const VarRef &ctx,
                                                         std::vector<DeclNode *> &items) {
  std::vector<BodyItem> body;
  auto setup = RenderDispatchSetup(op, ctx);
  body.insert(body.end(), setup.begin(), setup.end());
  body.push_back(RenderDispatchLoop(op, ctx));
  auto distribution = RenderDistribution(op, ctx);
  body.insert(body.end(), distribution.begin(), distribution.end());
  return TaskCodeBuilderUtil::RenderDispatchFunc(ast_, "DispatchCustomKernel", body, items);
}

std::vector<BodyItem> CustomTaskCodeBuilder::RenderDispatchSetup(const VarRef &op, const VarRef &ctx) {
  return {
      ast_.VarDecl(
          ast_.Var("ArgsInfo *", "args_info"),
          ctx.Attr("args_table").Attr("GetArgsInfo")(op.Arrow("dispatch_info").Attr("custom").Attr("args_idx"))),
      ChkNotNull(ast_.Var("", "args_info")),
      // -- 声明 ordered_io_addrs 和 Report IO 向量 --
      ast_.VarDecl(ast_.Var("std::vector<Om2Tensor>", "io_tensors")),
      ast_.VarDecl(ast_.Var("std::vector<Om2Tensor>", "input_tensors")),
      ast_.VarDecl(ast_.Var("std::vector<Om2Tensor>", "output_tensors")),
      ast_.Call(
          "",
          {ast_.Var("", "io_tensors").Attr("reserve")(op.Arrow("dispatch_info").Attr("custom").Attr("args_info_num"))}),
      ast_.VarDecl(ast_.Var("std::vector<Om2TaskIoEntry>", "report_inputs")),
      ast_.VarDecl(ast_.Var("std::vector<Om2TaskIoEntry>", "report_outputs")),
      ast_.VarDecl(ast_.Var("std::vector<uint64_t>", "report_workspace_addrs")),
      ast_.VarDecl(ast_.Var("std::vector<uint64_t>", "report_workspace_sizes")),
  };
}

BodyItem CustomTaskCodeBuilder::RenderDispatchLoop(const VarRef &op, const VarRef &ctx) {
  auto a = ast_.Var("const auto &", "a");
  return ast_.For(ast_.VarDecl("uint32_t", "j", ast_.UInt(0)),
                  ast_.Var("", "j") < op.Arrow("dispatch_info").Attr("custom").Attr("args_info_num"),
                  ast_.PostInc(ast_.Var("", "j")),
                  std::initializer_list<BodyItem>{
                      ast_.VarDecl(a, op.Arrow("dispatch_info").Attr("custom").Attr("args_info")[ast_.Var("", "j")]),
                      ast_.VarDecl(ast_.Var("uint64_t", "_addr"), ast_.UInt(0)),
                      ast_.Switch(ast_.Var("", "a").Attr("type"),
                                  std::vector<BodyItem>{
                                      // INPUT / OUTPUT / CONST_TENSOR → 共享 handler（内部根据 a.type 区分）
                                      ast_.Case(ast_.Var("", "OP_ARG_INPUT")),
                                      ast_.Case(ast_.Var("", "OP_ARG_OUTPUT")),
                                      ast_.Case(ast_.Var("", "OP_ARG_CONST_TENSOR")),
                                      ast_.Block(HandleInputOutputArg(a, ctx)),
                                      ast_.Case(Arg(nullptr)),
                                      ast_.Block({
                                          ast_.Break(),
                                      }),
                                  }),
                  });
}

std::vector<BodyItem> CustomTaskCodeBuilder::RenderDistribution(const VarRef &op, const VarRef &ctx) {
  auto custom = op.Arrow("dispatch_info").Attr("custom");
  auto dispatch_type = ast_.StaticCast("uint32_t", op.Arrow("dispatch_type"));
  auto stream = ctx.Attr("stream_list")[custom.Attr("stream_id")];

  return {
      ChkStatus(ast_.Call(
          "ReportOm2TaskPreprocess",
          {op.Arrow("op_name"), custom.Attr("op_type"),
           ast_.UInt(0),  // op_desc_id
           ast_.ReinterpretCast("uintptr_t", ast_.Var("", "args_info").Arrow("dev_addr")),
           ast_.Var("", "args_info").Arrow("size"), ast_.Var("", "report_inputs"), ast_.Var("", "report_outputs"),
           ast_.Var("", "report_workspace_addrs"), ast_.Var("", "report_workspace_sizes"), dispatch_type,
           ast_.Var("", "0"), stream, ast_.Var("", "nullptr"), ctx.Attr("model_id"), ctx.Attr("instance_handle")})),
      ast_.VarDecl(ast_.Var("uint64_t", "_launch_begin"), ast_.Call("MsprofSysCycleTime", {})),
      ChkStatus(ast_.Call("KernelCustTaskDistribute",
                          {ast_.Var("", "op->op_name"), ast_.Var("", "op->dispatch_info.custom.op_type"),
                           ast_.Var("", "input_tensors"), ast_.Var("", "output_tensors"), stream})),
      ChkStatus(ast_.Call(
          "ReportLaunchedOm2Task",
          {op.Arrow("op_name"), custom.Attr("op_type"),
           ast_.UInt(0),  // op_desc_id
           ast_.ReinterpretCast("uintptr_t", ast_.Var("", "args_info").Arrow("dev_addr")),
           ast_.Var("", "args_info").Arrow("size"), ast_.Var("", "report_inputs").Data(),
           ast_.StaticCast("uint64_t", ast_.Var("", "report_inputs").Size()), ast_.Var("", "report_outputs").Data(),
           ast_.StaticCast("uint32_t", ast_.Var("", "report_outputs").Size()),
           ast_.Var("", "report_workspace_addrs").Data(), ast_.Var("", "report_workspace_sizes").Data(),
           ast_.StaticCast("uint32_t", ast_.Var("", "report_workspace_sizes").Size()), dispatch_type, ast_.Var("", "0"),
           stream, ctx.Attr("model_id"), ctx.Attr("instance_handle"), ast_.UInt(0U),
           ast_.Var("uint64_t", "_launch_begin")})),
  };
}

std::vector<BodyItem> CustomTaskCodeBuilder::HandleInputOutputArg(const VarRef &a, const VarRef &ctx) {
  return {
      ast_.Assign(
          ast_.Var("", "_addr"),
          ast_.ReinterpretCast("uint64_t",
                               ast_.Call("ResolveOpAddr", {a.Attr("addr").Attr("mem_src"), a.Attr("addr").Attr("index"),
                                                           a.Attr("addr").Attr("offset"), ctx.Attr("total_dev_mem_ptr"),
                                                           ctx.Attr("session_scope_mem_ptr"), ctx.Attr("constants"),
                                                           ctx.Attr("var_addrs")}))),
      ast_.Var("", "io_tensors")
          .PushBack(ast_.Call(
              "BuildOm2Tensor",
              {ast_.ReinterpretCast("void *", ast_.Var("", "_addr")), a.Attr("data").Attr("tensor").Attr("size"),
               a.Attr("data").Attr("tensor").Attr("data_type"), a.Attr("data").Attr("tensor").Attr("format"),
               a.Attr("data").Attr("tensor").Attr("shape"), a.Attr("data").Attr("tensor").Attr("shape_dims")})),
      ast_.VarDecl(ast_.Var("Om2TaskIoEntry", "_entry"),
                   ast_.InitList({ast_.Var("", "io_tensors").Attr("back")().Addr(),
                                  a.Attr("data").Attr("tensor").Attr("args_offset")})),
      ast_.If(a.Attr("type") == ast_.Var("", "OP_ARG_INPUT") || a.Attr("type") == ast_.Var("", "OP_ARG_CONST_TENSOR"),
              {ast_.Var("", "report_inputs").PushBack(ast_.Var("", "_entry")),
               ast_.Var("", "input_tensors").PushBack(ast_.Var("", "io_tensors.back()"))},
              {ast_.Var("", "report_outputs").PushBack(ast_.Var("", "_entry")),
               ast_.Var("", "output_tensors").PushBack(ast_.Var("", "io_tensors.back()"))}),
      ast_.Break(),
  };
}

REGISTER_TASK_CODE_BUILDER(MODEL_TASK_CUSTOM_KERNEL, CustomTaskCodeBuilder);
}  // namespace ge
