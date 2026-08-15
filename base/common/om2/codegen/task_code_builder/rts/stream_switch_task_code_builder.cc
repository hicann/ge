/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "stream_switch_task_code_builder.h"
#include "common/om2/codegen/task_code_builder/task_code_builder_util.h"
#include "common/om2/codegen/task_args_manager/om2_model_args_utils.h"
#include "common/om2/codegen/task_code_builder_factory.h"
#include "common/om2/codegen/om2_model_utils.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/utils/attr_utils.h"

namespace ge {
Status StreamSwitchTaskCodeBuilder::Contribute(TaskSemanticContributeContext &context) {
  FillTaskSemanticHeader(context, header_);
  GE_ASSERT_NOTNULL(context.runtime);
  GE_ASSERT_NOTNULL(context.op_desc);
  // input_ptr, value_ptr
  GE_ASSERT_SUCCESS(Om2ModelUtils::ResolveInputAddrs(context, input_addr_nodes_));
  GE_ASSERT_TRUE(input_addr_nodes_.size() >= 2U, "[OM2][Check][Param] %s(%s) input addr size:%zu is invalid.",
                 context.op_desc->GetName().c_str(), context.op_desc->GetType().c_str(), input_addr_nodes_.size());

  // cond
  GE_ASSERT_TRUE(AttrUtils::GetInt(context.op_desc, ATTR_NAME_STREAM_SWITCH_COND, build_data_.cond),
                 "[Get][Attr] %s in op:%s(%s) fail", ATTR_NAME_STREAM_SWITCH_COND.c_str(),
                 context.op_desc->GetName().c_str(), context.op_desc->GetType().c_str());

  // true_stream
  std::vector<uint32_t> active_stream_list;
  GE_ASSERT_TRUE(AttrUtils::GetListInt(context.op_desc, ATTR_NAME_ACTIVE_STREAM_LIST, active_stream_list) &&
                     active_stream_list.size() == kTrueBranchStreamNum_1,
                 "[Get][Attr] %s in op:%s fail, active_stream_list.size():%zu", ATTR_NAME_ACTIVE_STREAM_LIST.c_str(),
                 context.op_desc->GetName().c_str(), active_stream_list.size());
  build_data_.true_stream_id = active_stream_list.front();
  GE_ASSERT_TRUE(build_data_.true_stream_id < context.runtime->stream_num,
                 "[OM2][Check][Param] active_stream_index:%zu in op:%s(%s) >= stream list size:%u in model",
                 static_cast<size_t>(build_data_.true_stream_id), context.op_desc->GetName().c_str(),
                 context.op_desc->GetType().c_str(), context.runtime->stream_num);
  // stream
  GE_ASSERT_TRUE(header_.stream_id < context.runtime->stream_num, "[OM2][Check][Param] stream list size:%u, cur:%u!",
                 context.runtime->stream_num, header_.stream_id);

  // data_type
  if (context.op_desc->HasAttr(ATTR_NAME_SWITCH_DATA_TYPE) &&
      !AttrUtils::GetInt(context.op_desc, ATTR_NAME_SWITCH_DATA_TYPE, build_data_.data_type)) {
    REPORT_INNER_ERR_MSG("E19999", "Get Attr:%s in op:%s(%s) fail, attribute value not int",
                         ATTR_NAME_SWITCH_DATA_TYPE.c_str(), context.op_desc->GetName().c_str(),
                         context.op_desc->GetType().c_str());
    GELOGE(FAILED, "[Get][Attr] %s in op:%s(%s) fail, attribute value not int", ATTR_NAME_SWITCH_DATA_TYPE.c_str(),
           context.op_desc->GetName().c_str(), context.op_desc->GetType().c_str());
    return FAILED;
  }
  GELOGI("Stream Switch Task Codegen: op[%s], cond_[%u], true stream id[%lu], stream id[%u], data type[%ld].",
         context.op_desc->GetName().c_str(), build_data_.cond, static_cast<unsigned long>(build_data_.true_stream_id),
         header_.stream_id, build_data_.data_type);
  build_data_.stream_id = header_.stream_id;
  for (const auto &addr : input_addr_nodes_) {
    auto arg = TaskCodeBuilderUtil::ConvertAddrDesc(addr);
    arg.has_tensor_info = false;
    build_data_.ordered_args.push_back(std::move(arg));
  }
  return SUCCESS;
}

Status StreamSwitchTaskCodeBuilder::RenderDistHelper(std::vector<DeclNode *> &items) {
  std::vector<BodyItem> dispatch_body;
  auto op = ast_.Var("const TaskDispatchInfo *", "op");
  auto ctx = ast_.Var("const DispatchOpContext &", "ctx");
  dispatch_body.push_back(ChkRt(RtSetTaskTag(op.Arrow("op_name"))));
  dispatch_body.push_back(ChkRt(AclrtSwitchStream(
      ast_.Call("ResolveOpAddr",
                {op.Arrow("dispatch_info").Attr("stream_switch").Attr("args_info")[0].Attr("addr").Attr("mem_src"),
                 op.Arrow("dispatch_info").Attr("stream_switch").Attr("args_info")[0].Attr("addr").Attr("index"),
                 op.Arrow("dispatch_info").Attr("stream_switch").Attr("args_info")[0].Attr("addr").Attr("offset"),
                 ctx.Attr("total_dev_mem_ptr"), ctx.Attr("session_scope_mem_ptr"), ctx.Attr("constants"),
                 ctx.Attr("var_addrs")}),
      ast_.StaticCast("aclrtCondition", op.Arrow("dispatch_info").Attr("stream_switch").Attr("cond")),
      ast_.Call("ResolveOpAddr",
                {op.Arrow("dispatch_info").Attr("stream_switch").Attr("args_info")[1].Attr("addr").Attr("mem_src"),
                 op.Arrow("dispatch_info").Attr("stream_switch").Attr("args_info")[1].Attr("addr").Attr("index"),
                 op.Arrow("dispatch_info").Attr("stream_switch").Attr("args_info")[1].Attr("addr").Attr("offset"),
                 ctx.Attr("total_dev_mem_ptr"), ctx.Attr("session_scope_mem_ptr"), ctx.Attr("constants"),
                 ctx.Attr("var_addrs")}),
      ast_.StaticCast("aclrtCompareDataType", op.Arrow("dispatch_info").Attr("stream_switch").Attr("data_type")),
      ctx.Attr("stream_list")[op.Arrow("dispatch_info").Attr("stream_switch").Attr("true_stream_id")],
      ctx.Attr("stream_list")[op.Arrow("dispatch_info").Attr("stream_switch").Attr("stream_id")])));
  GE_ASSERT_SUCCESS(TaskCodeBuilderUtil::RenderDispatchFunc(ast_, kDispatchFuncName, dispatch_body, items));
  return SUCCESS;
}

int64_t StreamSwitchTaskCodeBuilder::ParseOpIndex(const domi::TaskDef &task_def) {
  const auto &stream_switch_def = task_def.stream_switch();
  return static_cast<int64_t>(stream_switch_def.op_index());
}

std::string StreamSwitchTaskCodeBuilder::GetFuncName() const {
  return kDispatchFuncName;
}

Status StreamSwitchTaskCodeBuilder::RenderOpDefTableFields(std::vector<std::pair<std::string, Arg>> &fields) {
  fields.push_back({"dispatch_type", ast_.StaticCast("OpDispatchType", static_cast<int64_t>(kDispatchType))});
  fields.push_back({"op_name", Arg::StringLiteral(header_.op_name)});
  fields.push_back(
      {"dispatch_info",
       ast_.DesignatedInit(
           {{"stream_switch", ast_.InitList({TaskCodeBuilderUtil::RenderOpArgDesc(ast_, build_data_.ordered_args),
                                             build_data_.true_stream_id, build_data_.stream_id, build_data_.cond,
                                             build_data_.data_type})}})});
  return SUCCESS;
}

Status StreamSwitchTaskCodeBuilder::ParseTaskRunParam(const domi::TaskDef &task_def, const om2::RuntimeParam &rts_param,
                                                      OpDescPtr op_desc, om2::TaskRunParam &task_run_param) {
  GE_CHECK_NOTNULL(&rts_param);
  const auto &stream_switch_def = task_def.stream_switch();
  const uint32_t op_index = stream_switch_def.op_index();
  GELOGI("[OM2] Begin to calculate args, op_index is: %u", op_index);
  GE_CHECK_NOTNULL(op_desc);
  op_desc_ = op_desc;
  GELOGI("[OM2] Calc opType[%s] args size. Node name is [%s]", op_desc->GetType().c_str(), op_desc->GetName().c_str());
  const size_t input_size = op_desc->GetInputsSize();
  std::vector<uint64_t> mem_types;
  const auto input_data_addrs = om2::ModelUtils::GetInputAddrsValue(rts_param, op_desc, mem_types);
  if ((input_data_addrs.size() != STREAM_SWITCH_INPUT_NUM) || (input_size != STREAM_SWITCH_INPUT_NUM)) {
    REPORT_INNER_ERR_MSG("E19999", "[OM2] Op:%s, input_data_addrs.size():%zu or input size:%zu != %u, check invalid",
                         op_desc->GetName().c_str(), input_data_addrs.size(), input_size, STREAM_SWITCH_INPUT_NUM);
    GELOGE(FAILED, "[OM2][Check][Param] Op:%s, input_data_addrs.size():%zu, input size:%zu != %u.",
           op_desc->GetName().c_str(), input_data_addrs.size(), input_size, STREAM_SWITCH_INPUT_NUM);
    return FAILED;
  }

  task_run_param.parsed_input_addrs.push_back({input_data_addrs[0U], mem_types[0U], true, {0}});
  task_run_param.parsed_input_addrs.push_back({input_data_addrs[1U], mem_types[1U], true, {0}});
  GELOGD("[OM2]parse task param, input_addrs[0] %llu, mem_types[0] %llu, input_addrs[1] %llu, mem_types[1] %llu",
         input_data_addrs[0U], mem_types[0U], input_data_addrs[1U], mem_types[1U]);
  return SUCCESS;
}

Status StreamSwitchTaskCodeBuilder::Init(const domi::TaskDef &task_def,
                                         std::vector<om2::MemAllocation> &logical_mem_allocations,
                                         const om2::PisToArgs &args, const om2::IowAddrs &iow_addrs) {
  (void)task_def;
  (void)logical_mem_allocations;
  (void)args;
  (void)iow_addrs;
  return SUCCESS;
}

REGISTER_TASK_CODE_BUILDER(MODEL_TASK_STREAM_SWITCH, StreamSwitchTaskCodeBuilder);
}  // namespace ge
