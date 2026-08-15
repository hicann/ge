/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "memcpy_async_task_code_builder.h"
#include "common/om2/codegen/task_code_builder/task_code_builder_util.h"

#include <cinttypes>

#include "common/om2/codegen/task_code_builder_factory.h"
#include "common/om2/codegen/om2_model_utils.h"
#include "common/om2/codegen/task_args_manager/om2_model_args_utils.h"
#include "graph/debug/ge_attr_define.h"

namespace ge {
constexpr uint64_t ioAddrArgSize = 16;

Status MemcpyAsyncTaskCodeBuilder::Contribute(TaskSemanticContributeContext &context) {
  FillTaskSemanticHeader(context, header_);
  GE_ASSERT_NOTNULL(context.runtime);
  GE_ASSERT_NOTNULL(context.op_desc);
  GE_ASSERT_NOTNULL(context.op_index_to_count_map);
  const domi::MemcpyAsyncDef &memcpy_async = context.task_def.memcpy_async();

  ResolveInternalIndex(context);

  GE_ASSERT_SUCCESS(Om2ModelUtils::GetRtAddress(context, static_cast<uintptr_t>(memcpy_async.src()), input_addr_node_,
                                                true, internal_index_));
  GE_ASSERT_SUCCESS(Om2ModelUtils::GetRtAddress(context, static_cast<uintptr_t>(memcpy_async.dst()), output_addr_node_,
                                                false, internal_index_));

  CheckIoRefresh(context);

  build_data_.dst_max = memcpy_async.dst_max();
  build_data_.count = memcpy_async.count();
  build_data_.kind = memcpy_async.kind();

  if (build_data_.io_refresh) {
    SetupIoAddrRefresh(context);
    build_data_.kind = RT_MEMCPY_ADDR_DEVICE_TO_DEVICE;
  }
  GE_ASSERT_TRUE(header_.stream_id < context.runtime->stream_num, "[OM2][Check][Param] stream list size:%u, cur:%u!",
                 context.runtime->stream_num, header_.stream_id);
  GELOGI("Memcpy Async Task Codegen: op[%s], dst max[%" PRIu64 "], count[%" PRIu64 "], kind[%u], stream id[%u].",
         context.op_desc->GetName().c_str(), build_data_.dst_max, build_data_.count, build_data_.kind,
         header_.stream_id);
  GELOGI("op_index %u, op_id %" PRId64, memcpy_async.op_index(), context.op_desc->GetId());
  build_data_.stream_id = header_.stream_id;
  build_data_.ordered_args.push_back(TaskCodeBuilderUtil::ConvertAddrDesc(input_addr_node_));
  build_data_.ordered_args.push_back(TaskCodeBuilderUtil::ConvertAddrDesc(output_addr_node_));
  build_data_.args_table_idx = static_cast<uint32_t>(entry_.has_value() ? entry_->table_index : 0U);
  return SUCCESS;
}

void MemcpyAsyncTaskCodeBuilder::ResolveInternalIndex(TaskSemanticContributeContext &context) {
  auto it = context.op_index_to_count_map->find(static_cast<uint32_t>(header_.op_index));
  if (it == context.op_index_to_count_map->end()) {
    internal_index_ = 0U;
    (*context.op_index_to_count_map)[static_cast<uint32_t>(header_.op_index)] = 1U;
  } else {
    internal_index_ = it->second;
    ++(*context.op_index_to_count_map)[static_cast<uint32_t>(header_.op_index)];
  }
}

void MemcpyAsyncTaskCodeBuilder::CheckIoRefresh(TaskSemanticContributeContext &context) {
  (void)context;
  build_data_.io_refresh = (input_addr_node_.memory_app == om2::MemoryAppType::kMemoryTypeModelIo) ||
                           (output_addr_node_.memory_app == om2::MemoryAppType::kMemoryTypeModelIo);
}

void MemcpyAsyncTaskCodeBuilder::SetupIoAddrRefresh(TaskSemanticContributeContext &context) {
  const uint64_t addr_offset = *context.next_host_args_offset;
  if (input_addr_node_.memory_app == om2::MemoryAppType::kMemoryTypeModelIo) {
    io_addr_refresh_records_.push_back(
        IoAddrRefreshRecord{static_cast<uint64_t>(input_addr_node_.compile_state_io_addr_offset), addr_offset});
    GELOGI("[OM2]append input addr offset map: compile offset[%lu], args info offset[%lu]",
           static_cast<uint64_t>(input_addr_node_.compile_state_io_addr_offset), addr_offset);
  }
  if (output_addr_node_.memory_app == om2::MemoryAppType::kMemoryTypeModelIo) {
    io_addr_refresh_records_.push_back(IoAddrRefreshRecord{
        static_cast<uint64_t>(output_addr_node_.compile_state_io_addr_offset), addr_offset + sizeof(uint64_t)});
    GELOGI("[OM2]append output addr offset map: compile offset[%lu], args info offset[%lu]",
           static_cast<uint64_t>(output_addr_node_.compile_state_io_addr_offset), addr_offset + sizeof(uint64_t));
  }

  (void)entry_.emplace();
  entry_->table_index = *context.next_args_table_index;
  entry_->args_size = ioAddrArgSize;
  entry_->host_offset = *context.next_host_args_offset;
  args_table_entry_ = &(*entry_);
  ++(*context.next_args_table_index);
  *context.next_host_args_offset += Om2ModelUtils::ArgsSizeAlign8(static_cast<uint64_t>(entry_->args_size));
}

DeclNode *MemcpyAsyncTaskCodeBuilder::RenderMemcpyAsyncDistribute() {
  auto op_name = ast_.Var("const char_t *const", "op_name");
  auto dst = ast_.Var("void *", "dst");
  auto dest_max = ast_.Var("uint64_t", "destMax");
  auto src = ast_.Var("void *", "src");
  auto count = ast_.Var("uint64_t", "count");
  auto kind = ast_.Var("const rtMemcpyKind_t", "kind");
  auto stream = ast_.Var("aclrtStream &", "stream");
  auto qos_cfg = ast_.Var("const uint32_t", "qosCfg");
  auto inputs = ast_.Var("std::array<uintptr_t, 7U>", "inputs");
  return ast_.DefineFunction("KernelMemcpyAsyncDistribute", {op_name, dst, dest_max, src, count, kind, stream, qos_cfg},
                             "aclError",
                             {
                                 ChkRt(RtSetTaskTag(op_name)),
                                 ast_.VarDecl(inputs, ast_.InitList({
                                                          ast_.ReinterpretCast("uintptr_t", dst),
                                                          ast_.StaticCast("uintptr_t", dest_max),
                                                          ast_.ReinterpretCast("uintptr_t", src),
                                                          ast_.StaticCast("uintptr_t", count),
                                                          ast_.StaticCast("uintptr_t", kind),
                                                          ast_.ReinterpretCast("uintptr_t", stream),
                                                          ast_.StaticCast("uintptr_t", qos_cfg),
                                                      })),
                                 ChkRt(RtGeneralCtrl(inputs[0].Addr(), ast_.StaticCast("uint32_t", 7), 0)),
                                 ast_.Return("ACL_SUCCESS"),
                             });
}

BodyItem MemcpyAsyncTaskCodeBuilder::RenderIoRefreshDispatch(const VarRef &op, const VarRef &ctx) {
  const auto td = op.Arrow("dispatch_info").Attr("memcpy_async");

  // ResolveOpAddr for src / dst
  auto resolve_src =
      ast_.Call("ResolveOpAddr", {td.Attr("args_info")[0U].Attr("addr").Attr("mem_src"),
                                  td.Attr("args_info")[0U].Attr("addr").Attr("index"),
                                  td.Attr("args_info")[0U].Attr("addr").Attr("offset"), ctx.Attr("total_dev_mem_ptr"),
                                  ctx.Attr("session_scope_mem_ptr"), ctx.Attr("constants"), ctx.Attr("var_addrs")});
  auto resolve_dst =
      ast_.Call("ResolveOpAddr", {td.Attr("args_info")[1U].Attr("addr").Attr("mem_src"),
                                  td.Attr("args_info")[1U].Attr("addr").Attr("index"),
                                  td.Attr("args_info")[1U].Attr("addr").Attr("offset"), ctx.Attr("total_dev_mem_ptr"),
                                  ctx.Attr("session_scope_mem_ptr"), ctx.Attr("constants"), ctx.Attr("var_addrs")});

  auto args_table_key = ast_.Var("auto", "args_table_key");
  auto iow_addr = ast_.Var("std::vector<uint64_t>", "iow_addr");

  return ast_.Block({
      ast_.VarDecl(args_table_key, ctx.Attr("args_table").Attr("GetArgsInfo")(td.Attr("args_table_idx"))),
      ast_.VarDecl(iow_addr, ast_.InitList({
                                 ast_.ReinterpretCast("uint64_t", resolve_src),
                                 ast_.ReinterpretCast("uint64_t", resolve_dst),
                             })),
      ChkStatus(MemcpyS(ast_.Var("", "args_table_key").Arrow("host_addr"), ast_.Var("", "args_table_key").Arrow("size"),
                        ast_.Var("std::vector<uint64_t>", "iow_addr").Data(),
                        ast_.Var("std::vector<uint64_t>", "iow_addr").Size() * ast_.Sizeof("uint64_t"))),
      ChkStatus(ast_.Call(
          "KernelMemcpyAsyncDistribute",
          {
              op.Arrow("op_name"),
              ast_.Call("ValueToPtr", {ast_.Call("PtrToValue", {ast_.Var("", "args_table_key").Arrow("dev_addr")}) +
                                       ast_.Sizeof("uint64_t")}),
              td.Attr("dst_max"),
              ast_.Var("", "args_table_key").Arrow("dev_addr"),
              td.Attr("count"),
              ast_.StaticCast("rtMemcpyKind_t", td.Attr("kind")),
              ctx.Attr("stream_list")[td.Attr("stream_id")],
              ast_.UInt(0),
          })),
  });
}

BodyItem MemcpyAsyncTaskCodeBuilder::RenderDirectDispatch(const VarRef &op, const VarRef &ctx) {
  const auto td = op.Arrow("dispatch_info").Attr("memcpy_async");

  auto resolve_src =
      ast_.Call("ResolveOpAddr", {td.Attr("args_info")[0].Attr("addr").Attr("mem_src"),
                                  td.Attr("args_info")[0].Attr("addr").Attr("index"),
                                  td.Attr("args_info")[0].Attr("addr").Attr("offset"), ctx.Attr("total_dev_mem_ptr"),
                                  ctx.Attr("session_scope_mem_ptr"), ctx.Attr("constants"), ctx.Attr("var_addrs")});
  auto resolve_dst =
      ast_.Call("ResolveOpAddr", {td.Attr("args_info")[1].Attr("addr").Attr("mem_src"),
                                  td.Attr("args_info")[1].Attr("addr").Attr("index"),
                                  td.Attr("args_info")[1].Attr("addr").Attr("offset"), ctx.Attr("total_dev_mem_ptr"),
                                  ctx.Attr("session_scope_mem_ptr"), ctx.Attr("constants"), ctx.Attr("var_addrs")});

  return ChkStatus(ast_.Call("KernelMemcpyAsyncDistribute", {
                                                                op.Arrow("op_name"),
                                                                resolve_dst,
                                                                td.Attr("dst_max"),
                                                                resolve_src,
                                                                td.Attr("count"),
                                                                ast_.StaticCast("rtMemcpyKind_t", td.Attr("kind")),
                                                                ctx.Attr("stream_list")[td.Attr("stream_id")],
                                                                ast_.UInt(0),
                                                            }));
}

Status MemcpyAsyncTaskCodeBuilder::RenderDistHelper(std::vector<DeclNode *> &items) {
  items.push_back(RenderMemcpyAsyncDistribute());

  std::vector<BodyItem> body;
  auto op = ast_.Var("const TaskDispatchInfo *", "op");
  auto ctx = ast_.Var("const DispatchOpContext &", "ctx");
  const auto td = op.Arrow("dispatch_info").Attr("memcpy_async");
  body.push_back(ast_.If(td.Attr("io_refresh"), {RenderIoRefreshDispatch(op, ctx)}, {RenderDirectDispatch(op, ctx)}));

  GE_ASSERT_SUCCESS(TaskCodeBuilderUtil::RenderDispatchFunc(ast_, kDispatchFuncName, body, items));
  return SUCCESS;
}

int64_t MemcpyAsyncTaskCodeBuilder::ParseOpIndex(const domi::TaskDef &task_def) {
  const domi::MemcpyAsyncDef &memcpy_async = task_def.memcpy_async();
  return static_cast<int64_t>(memcpy_async.op_index());
}

std::string MemcpyAsyncTaskCodeBuilder::GetFuncName() const {
  return kDispatchFuncName;
}

Status MemcpyAsyncTaskCodeBuilder::RenderOpDefTableFields(std::vector<std::pair<std::string, Arg>> &fields) {
  fields.push_back({"dispatch_type", ast_.StaticCast("OpDispatchType", static_cast<int64_t>(kDispatchType))});
  fields.push_back({"op_name", Arg::StringLiteral(header_.op_name)});
  fields.push_back(
      {"dispatch_info",
       ast_.DesignatedInit(
           {{"memcpy_async",
             ast_.InitList({TaskCodeBuilderUtil::RenderOpArgDesc(ast_, build_data_.ordered_args),
                            static_cast<int64_t>(build_data_.dst_max), static_cast<int64_t>(build_data_.count),
                            build_data_.kind, build_data_.stream_id, build_data_.args_table_idx,
                            build_data_.io_refresh})}})});
  return SUCCESS;
}

Status MemcpyAsyncTaskCodeBuilder::ParseTaskRunParam(const domi::TaskDef &task_def, const om2::RuntimeParam &rts_param,
                                                     OpDescPtr op_desc, om2::TaskRunParam &task_run_param) {
  const domi::MemcpyAsyncDef &memcpy_async = task_def.memcpy_async();
  GE_ASSERT_NOTNULL(op_desc);
  op_desc_ = op_desc;
  uint8_t *src = nullptr;
  uint8_t *dst = nullptr;
  const Status ret1 =
      om2::ModelUtils::GetRtAddress(rts_param, static_cast<uintptr_t>(memcpy_async.src()), src, logical_src_mem_type_);
  const Status ret2 =
      om2::ModelUtils::GetRtAddress(rts_param, static_cast<uintptr_t>(memcpy_async.dst()), dst, logical_dst_mem_type_);
  if ((ret1 != SUCCESS) || (ret2 != SUCCESS)) {
    return PARAM_INVALID;
  }
  task_run_param.parsed_input_addrs.push_back({PtrToValue(src), logical_src_mem_type_, true, {0}});

  // dst_ needs different address for different chips
  std::vector<int64_t> memory_type_list;
  (void)AttrUtils::GetListInt(op_desc_, ATTR_NAME_OUTPUT_MEM_TYPE_LIST, memory_type_list);
  if ((!memory_type_list.empty()) && (memory_type_list[0U] == static_cast<int64_t>(RT_MEMORY_TS))) {
    GELOGE(FAILED, "[OM2] Unsupported output_memory_type attr");
    return FAILED;
  }
  task_run_param.parsed_output_addrs.push_back({PtrToValue(dst), logical_dst_mem_type_, true, {0}});

  constexpr uint32_t args_size = static_cast<uint32_t>(sizeof(void *) * 2U);
  task_run_param.args_descs.push_back({static_cast<int64_t>(args_size), pls_});
  return SUCCESS;
}

Status MemcpyAsyncTaskCodeBuilder::SetIoAddrs(const om2::IowAddrs &iow_addrs) {
  io_addrs_.emplace_back(ValueToPtr(iow_addrs.input_logic_addrs[0U].logic_addr));
  io_addrs_.emplace_back(ValueToPtr(iow_addrs.output_logic_addrs[0U].logic_addr));
  io_addr_mem_types_.emplace_back(iow_addrs.input_logic_addrs[0U].memory_type);
  io_addr_mem_types_.emplace_back(iow_addrs.output_logic_addrs[0U].memory_type);
  return SUCCESS;
}

Status MemcpyAsyncTaskCodeBuilder::Init(const domi::TaskDef &task_def,
                                        std::vector<om2::MemAllocation> &logical_mem_allocations,
                                        const om2::PisToArgs &args, const om2::IowAddrs &iow_addrs) {
  (void)args;
  (void)task_def;
  GE_ASSERT_TRUE((!(iow_addrs.input_logic_addrs.empty())), "[OM2][Check][Param] Op:%s, input logic addr list is empty.",
                 op_desc_->GetName().c_str());

  GE_ASSERT_TRUE((!(iow_addrs.output_logic_addrs.empty())),
                 "[OM2][Check][Param] Op:%s, output logic addr list is empty.", op_desc_->GetName().c_str());

  GE_CHK_STATUS_RET(SetIoAddrs(iow_addrs), "[OM2][Set][Addrs] failed, op:%s", op_desc_->GetName().c_str());
  const auto addrs = VPtrToValue(io_addrs_);
  GE_ASSERT_SUCCESS(args_io_addrs_updater_.Init(logical_mem_allocations, addrs, io_addr_mem_types_,
                                                {op_desc_->GetName(), op_desc_->GetType()}));
  return SUCCESS;
}

Status MemcpyAsyncTaskCodeBuilder::GetTaskArgsRefreshInfos(std::vector<om2::TaskArgsRefreshInfo> &infos) {
  args_io_addrs_updater_.GenArgsRefreshInfos(infos, 0UL, pls_);
  return SUCCESS;
}

REGISTER_TASK_CODE_BUILDER(MODEL_TASK_MEMCPY_ASYNC, MemcpyAsyncTaskCodeBuilder);
}  // namespace ge
