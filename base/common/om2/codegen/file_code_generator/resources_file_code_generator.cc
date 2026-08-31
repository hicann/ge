/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "common/om2/codegen/file_code_generator/resources_file_code_generator.h"

#include "common/om2/codegen/task_code_builder/task_code_builder.h"

namespace ge {

MethodDef *ResourcesFileCodeGenerator::BuildOm2ModelConstructor(const Om2CodegenModel &codegen_model) {
  auto bin_files = ast_.Var("const char **", "bin_files");
  auto bin_data = ast_.Var("const void **", "bin_data");
  auto bin_size = ast_.Var("uint64_t *", "bin_size");
  auto bin_num = ast_.Var("size_t", "bin_num");
  auto constants = ast_.Var("void **", "constants");
  auto var_addrs = ast_.Var("void **", "var_addrs");
  auto work_ptr = ast_.Var("void *", "work_ptr");
  auto session_id = ast_.Var("uint64_t *", "session_id");
  auto model_id = ast_.Var("uint32_t", "model_id");
  auto instance_handle = ast_.Var("void *", "instance_handle");
  auto priority = ast_.Var("int32_t", "priority");
  auto i = ast_.Var("size_t", "i");
  std::vector<BodyItem> body = {
      ast_.For(ast_.VarDecl(i, 0), i < bin_num, ast_.PreInc(i),
               {
                   ast_.Assign(bin_info_map_[ast_.ToStr(bin_files[i])], {bin_data[i], bin_size[i]}),
               }),
  };
  const auto &runtime = codegen_model.runtime;
  if (runtime.kernel_bin_num > 0U) {
    (void)body.emplace_back(bin_handles_.Resize(runtime.kernel_bin_num));
    (void)body.emplace_back(func_handles_.Resize(runtime.kernel_bin_num));
  }
  if (runtime.stream_num > 0U) {
    (void)body.emplace_back(stream_list_.Resize(runtime.stream_num));
  }
  if (runtime.notify_num > 0U) {
    (void)body.emplace_back(notify_list_.Resize(runtime.notify_num));
  }
  if (runtime.event_num > 0U) {
    (void)body.emplace_back(event_list_.Resize(runtime.event_num));
  }
  if (runtime.label_num > 0U) {
    (void)body.emplace_back(label_list_.Resize(runtime.label_num));
  }
  (void)body.emplace_back(ast_.Call("OM2_LOGD", {ast_.Str("Om2Model created")}));
  return ast_.DefineMethod(
      "Om2Model", "Om2Model",
      {bin_files, bin_data, bin_size, bin_num, constants, var_addrs, work_ptr, session_id, model_id, instance_handle,
       priority},
      "",
      {ast_.MemberInit("constants_", constants), ast_.MemberInit("var_addrs_", var_addrs),
       ast_.MemberInit("total_dev_mem_ptr_", work_ptr), ast_.MemberInit("owns_total_dev_mem_", false),
       ast_.MemberInit("session_id_", session_id), ast_.MemberInit("model_id_", model_id),
       ast_.MemberInit("instance_handle_", instance_handle), ast_.MemberInit("kernel_id_", 0),
       ast_.MemberInit("session_scope_mem_ptr_", nullptr), ast_.MemberInit("priority_", priority),
       ast_.MemberInit("is_external_rt_model_", false), ast_.MemberInit("is_external_streams_", false),
       ast_.MemberInit("is_external_notifies_", false), ast_.MemberInit("is_external_events_", false),
       ast_.MemberInit("is_external_labels_", false)},
      body);
}

MethodDef *ResourcesFileCodeGenerator::BuildOm2ModelDestructor() const {
  return ast_.DefineMethod("Om2Model", "~Om2Model", {}, "",
                           {
                               ast_.Call("OM2_LOGD", {ast_.Str("~Om2Model")}),
                               ast_.IgnoreOutput(ast_.Call("ReleaseResources", {})),
                           });
}

MethodDef *ResourcesFileCodeGenerator::BuildInitResourcesMethod(
    const Om2CodegenModel &codegen_model, const std::vector<TaskCodeBuilderPtr> &task_code_builders) {
  const auto reuse_zero_copy = ast_.Var("uint64_t", "reuse_zero_copy");
  const auto required_work_size = ast_.Var("size_t", "required_work_size");
  const auto model_work_size = ast_.Var("size_t", "kModelWorkSize");
  const auto model_zero_copy_size = ast_.Var("size_t", "kModelZeroCopySize");
  auto external_resources = ast_.Var("const GertModelExternalResources &", "external_resources");

  std::vector<BodyItem> body = {
      ast_.Call("OM2_LOGI", {ast_.Str("InitResources begin")}),
      ast_.Call("OM2_LOGI", {ast_.Str("model_id=%u, InitResources: work_ptr=%p, work_size=%zu, zero_copy_size=%zu"),
                             model_id_, total_dev_mem_ptr_, model_work_size, model_zero_copy_size}),
      ast_.VarDecl(required_work_size, model_work_size),
      ast_.If(reuse_zero_copy != 0U, {ast_.Assign(required_work_size, model_work_size - model_zero_copy_size)}),
      ast_.If(total_dev_mem_ptr_ != nullptr,
              {ast_.Call("OM2_LOGI", {ast_.Str("model_id=%u, InitResources: use external work_ptr=%p"), model_id_,
                                      total_dev_mem_ptr_})}),
      ast_.If((total_dev_mem_ptr_ == nullptr) && (required_work_size != 0U),
              {ast_.Call("OM2_LOGI", {ast_.Str("model_id=%u, InitResources: prepare work_ptr allocation, "
                                               "work_size=%zu, zero_copy_size=%zu, malloc_size=%zu"),
                                      model_id_, model_work_size, model_zero_copy_size, required_work_size}),
               ChkStatus(AclrtMallocHelper(total_dev_mem_ptr_.Addr(), required_work_size, "RT_MEMORY_HBM", 0U)),
               ast_.Assign(owns_total_dev_mem_, true)}),
      ast_.BlankLine(),
      ast_.Comment("1. 创建 model"),
      ast_.If(external_resources.Attr("external_rt_model") != nullptr,
              {ast_.Assign(model_handle_, external_resources.Attr("external_rt_model")),
               ast_.Assign(is_external_rt_model_, true)},
              {ChkStatus(AclmdlRIBuildBegin(model_handle_.Addr(), 0))}),
      ast_.BlankLine(),
      ast_.Comment("2. 获取overflow地址"),
      ChkStatus(AclrtCtxGetFloatOverflowAddr(overflow_addr_.Addr())),
      ast_.BlankLine(),
      ast_.Comment("3. 创建其他资源"),
  };
  const auto &runtime = codegen_model.runtime;
  BuildInitStreamResources(body, runtime, external_resources);
  BuildInitNotifyResources(body, runtime, external_resources);
  BuildInitEventResources(body, runtime, external_resources);
  BuildInitLabelResources(body, runtime, external_resources);
  for (const auto &task_code_builder : task_code_builders) {
    GE_ASSERT_NOTNULL(task_code_builder);
    GE_ASSERT_SUCCESS(task_code_builder->RenderInitResource(body));
  }
  BuildInitSessionScopeMemory(body, runtime);
  (void)body.emplace_back(args_table_.Attr("Init")());
  (void)body.emplace_back(ast_.Call("OM2_LOGI", {ast_.Str("InitResources done")}));
  (void)body.emplace_back(ast_.Return("ACL_SUCCESS"));
  return ast_.DefineMethod("Om2Model", "InitResources", {reuse_zero_copy, external_resources}, "aclError", body);
}

void ResourcesFileCodeGenerator::BuildInitStreamResources(std::vector<BodyItem> &body,
                                                          const RuntimeResourceSemantic &runtime,
                                                          const VarRef &external_resources) {
  if (runtime.stream_num == 0U) {
    return;
  }
  (void)body.emplace_back(ast_.Comment("创建下沉Stream并绑定模型"));
  auto ext_i = ast_.Var("size_t", "ext_i");
  auto ext_streams = external_resources.Attr("external_streams");
  auto ext_stream_num = external_resources.Attr("external_stream_num");
  std::vector<BodyItem> ext_items = {
      ast_.If(ext_stream_num != stream_list_.Size(),
              {ast_.Call("OM2_LOGE", {ast_.Str("external_stream_num mismatch, expected %zu, got %lu"),
                                      stream_list_.Size(), ext_stream_num}),
               ast_.Return("ACL_ERROR_FAILURE")}),
      ast_.For(ast_.VarDecl(ext_i, 0), ext_i < ext_stream_num, ast_.PreInc(ext_i),
               {ast_.Assign(stream_list_[ext_i], ext_streams[ext_i])}),
      ast_.Assign(is_external_streams_, true),
  };
  std::vector<BodyItem> create_items;
  for (uint32_t i = 0U; i < runtime.stream_num; ++i) {
    const auto stream_flag = ast_.Var("uint32_t", "stream" + std::to_string(i) + "_flag");
    create_items.emplace_back(ast_.VarDecl(stream_flag, runtime.stream_flag_values[i]));
    create_items.emplace_back(ChkRt(RtStreamCreateWithFlags(stream_list_[i].Addr(), priority_, stream_flag)));
  }
  (void)body.emplace_back(ast_.If(ext_stream_num != ast_.UInt(0U), ext_items, create_items));
  for (uint32_t i = 0U; i < runtime.stream_num; ++i) {
    const auto bind_flag = ast_.Var("auto", "bind" + std::to_string(i) + "_flag");
    (void)body.emplace_back(ast_.VarDecl(bind_flag, runtime.bind_flag_values[i]));
    (void)body.emplace_back(ChkStatus(AclmdlRIBindStream(model_handle_, stream_list_[i], bind_flag)));
  }
  (void)body.emplace_back(ast_.Assign(is_stream_list_bind_, true));
}

void ResourcesFileCodeGenerator::BuildInitNotifyResources(std::vector<BodyItem> &body,
                                                          const RuntimeResourceSemantic &runtime,
                                                          const VarRef &external_resources) {
  if (runtime.notify_num == 0U) {
    return;
  }
  (void)body.emplace_back(ast_.Comment("创建Notify"));
  auto ext_i = ast_.Var("size_t", "ext_i");
  auto ext_notifies = external_resources.Attr("external_notifies");
  auto ext_notify_num = external_resources.Attr("external_notify_num");
  std::vector<BodyItem> ext_items = {
      ast_.If(ext_notify_num != notify_list_.Size(),
              {ast_.Call("OM2_LOGE", {ast_.Str("external_notify_num mismatch, expected %zu, got %lu"),
                                      notify_list_.Size(), ext_notify_num}),
               ast_.Return("ACL_ERROR_FAILURE")}),
      ast_.For(ast_.VarDecl(ext_i, 0), ext_i < ext_notify_num, ast_.PreInc(ext_i),
               {ast_.Assign(notify_list_[ext_i], ext_notifies[ext_i])}),
      ast_.Assign(is_external_notifies_, true),
  };
  auto i = ast_.Var("size_t", "i");
  std::vector<BodyItem> create_items = {
      ast_.For(ast_.VarDecl(i, 0), i < runtime.notify_num, ast_.PreInc(i),
               {ChkStatus(AclrtCreateNotify(notify_list_[i].Addr(), "ACL_NOTIFY_DEVICE_USE_ONLY"))}),
  };
  (void)body.emplace_back(ast_.If(ext_notify_num != ast_.UInt(0U), ext_items, create_items));
}

void ResourcesFileCodeGenerator::BuildInitEventResources(std::vector<BodyItem> &body,
                                                         const RuntimeResourceSemantic &runtime,
                                                         const VarRef &external_resources) {
  if (runtime.event_num == 0U) {
    return;
  }
  (void)body.emplace_back(ast_.Comment("创建Event"));
  auto ext_i = ast_.Var("size_t", "ext_i");
  auto ext_events = external_resources.Attr("external_events");
  auto ext_event_num = external_resources.Attr("external_event_num");
  std::vector<BodyItem> ext_items = {
      ast_.If(ext_event_num != event_list_.Size(),
              {ast_.Call("OM2_LOGE", {ast_.Str("external_event_num mismatch, expected %zu, got %lu"),
                                      event_list_.Size(), ext_event_num}),
               ast_.Return("ACL_ERROR_FAILURE")}),
      ast_.For(ast_.VarDecl(ext_i, 0), ext_i < ext_event_num, ast_.PreInc(ext_i),
               {ast_.Assign(event_list_[ext_i], ext_events[ext_i])}),
      ast_.Assign(is_external_events_, true),
  };
  auto i = ast_.Var("size_t", "i");
  std::vector<BodyItem> create_items = {
      ast_.For(
          ast_.VarDecl(i, 0), i < runtime.event_num, ast_.PreInc(i),
          {ChkStatus(AclrtCreateEventWithFlag(
              event_list_[i].Addr(), "ACL_EVENT_SYNC | ACL_EVENT_CAPTURE_STREAM_PROGRESS | ACL_EVENT_TIME_LINE"))}),
  };
  (void)body.emplace_back(ast_.If(ext_event_num != ast_.UInt(0U), ext_items, create_items));
}

void ResourcesFileCodeGenerator::BuildInitLabelResources(std::vector<BodyItem> &body,
                                                         const RuntimeResourceSemantic &runtime,
                                                         const VarRef &external_resources) {
  if (runtime.label_num == 0U) {
    return;
  }
  (void)body.emplace_back(ast_.Comment("创建Label"));
  auto ext_i = ast_.Var("size_t", "ext_i");
  auto ext_labels = external_resources.Attr("external_labels");
  auto ext_label_num = external_resources.Attr("external_label_num");
  std::vector<BodyItem> ext_items = {
      ast_.If(ext_label_num != label_list_.Size(),
              {ast_.Call("OM2_LOGE", {ast_.Str("external_label_num mismatch, expected %zu, got %lu"),
                                      label_list_.Size(), ext_label_num}),
               ast_.Return("ACL_ERROR_FAILURE")}),
      ast_.For(ast_.VarDecl(ext_i, 0), ext_i < ext_label_num, ast_.PreInc(ext_i),
               {ast_.Assign(label_list_[ext_i], ext_labels[ext_i])}),
      ast_.Assign(is_external_labels_, true),
  };
  auto i = ast_.Var("size_t", "i");
  std::vector<BodyItem> create_items = {
      ast_.For(ast_.VarDecl(i, 0), i < runtime.label_num, ast_.PreInc(i),
               {ChkStatus(AclrtCreateLabel(label_list_[i].Addr()))}),
  };
  (void)body.emplace_back(ast_.If(ext_label_num != ast_.UInt(0U), ext_items, create_items));
}

void ResourcesFileCodeGenerator::BuildInitSessionScopeMemory(std::vector<BodyItem> &body,
                                                             const RuntimeResourceSemantic &runtime) {
  const uint64_t ss_key = kSessionScopeMemoryMask | RT_MEMORY_HBM;
  const auto ss_it = runtime.memory_infos.find(ss_key);
  if (ss_it == runtime.memory_infos.end() || ss_it->second.memory_size <= 0) {
    return;
  }
  (void)body.emplace_back(ast_.Comment("Allocate session scope memory"));
  (void)body.emplace_back(ChkStatus(AclrtMalloc(
      session_scope_mem_ptr_.Addr(), static_cast<int64_t>(ss_it->second.memory_size), "ACL_MEM_MALLOC_HUGE_FIRST")));
}

MethodDef *ResourcesFileCodeGenerator::BuildReleaseResourcesMethod(const Om2CodegenModel &codegen_model) {
  std::vector<BodyItem> body;
  (void)body.emplace_back(ast_.Call("OM2_LOGI", {ast_.Str("ReleaseResources begin")}));
  const auto &runtime = codegen_model.runtime;
  if (runtime.label_num > 0U) {
    auto label = ast_.Var("auto", "label");
    (void)body.emplace_back(ast_.If(!is_external_labels_,
                                    {ast_.RangeFor(label, label_list_,
                                                   {
                                                       ast_.If(label != nullptr, {ChkStatus(AclrtDestroyLabel(label))}),
                                                   })},
                                    {}, false));
  }
  if (runtime.event_num > 0U) {
    auto event = ast_.Var("auto", "event");
    (void)body.emplace_back(ast_.If(!is_external_events_,
                                    {ast_.RangeFor(event, event_list_,
                                                   {
                                                       ChkStatus(AclrtDestroyEvent(event)),
                                                   })},
                                    {}, false));
  }
  if (runtime.notify_num > 0U) {
    auto notify = ast_.Var("auto", "notify");
    (void)body.emplace_back(ast_.If(!is_external_notifies_,
                                    {ast_.RangeFor(notify, notify_list_,
                                                   {
                                                       ChkStatus(AclrtDestroyNotify(notify)),
                                                   })},
                                    {}, false));
  }
  if (runtime.stream_num > 0U) {
    auto stream = ast_.Var("auto", "stream");
    (void)body.emplace_back(
        ast_.If(is_stream_list_bind_, {ast_.RangeFor(stream, stream_list_,
                                                     {
                                                         ChkStatus(AclmdlRIUnbindStream(model_handle_, stream)),
                                                     })}));
    (void)body.emplace_back(ast_.If(!is_external_streams_,
                                    {ast_.RangeFor(stream, stream_list_,
                                                   {
                                                       ChkStatus(AclrtDestroyStream(stream)),
                                                   })},
                                    {}, false));
  }
  if (runtime.kernel_bin_num > 0U) {
    auto bin_handle = ast_.Var("auto", "bin_handle");
    (void)body.emplace_back(ast_.RangeFor(bin_handle, bin_handles_,
                                          {
                                              ChkStatus(AclrtBinaryUnLoad(bin_handle)),
                                          }));
  }
  BuildReleaseResourcesMethodForControlTask(body, runtime);
  auto i = ast_.Var("int", "i");
  (void)body.emplace_back(ast_.If(!is_external_rt_model_, {ChkStatus(AclmdlRIDestroy(model_handle_))}, {}, false));
  (void)body.emplace_back(ast_.If(session_scope_mem_ptr_ != nullptr, {
                                                                         ChkStatus(AclrtFree(session_scope_mem_ptr_)),
                                                                     }));
  (void)body.emplace_back(
      ast_.For(ast_.VarDecl(i, 0), i < dev_ext_info_mem_ptrs_.Size(), ast_.PostInc(i),
               {
                   ast_.If(dev_ext_info_mem_ptrs_[i] != nullptr, {ChkStatus(AclrtFree(dev_ext_info_mem_ptrs_[i]))}),
               }));
  (void)body.emplace_back(
      ast_.For(ast_.VarDecl(i, 0), i < dev_dynamic_mem_ptrs_.Size(), ast_.PostInc(i),
               {
                   ast_.If(dev_dynamic_mem_ptrs_[i] != nullptr, {ChkStatus(AclrtFree(dev_dynamic_mem_ptrs_[i]))}),
               }));
  (void)body.emplace_back(ast_.If((owns_total_dev_mem_ == true) && (total_dev_mem_ptr_ != nullptr),
                                  {ChkStatus(AclrtFree(total_dev_mem_ptr_))}));
  (void)body.emplace_back(ast_.Assign(total_dev_mem_ptr_, nullptr));
  (void)body.emplace_back(ast_.Assign(owns_total_dev_mem_, false));
  (void)body.emplace_back(ast_.Call("OM2_LOGI", {ast_.Str("ReleaseResources done")}));
  (void)body.emplace_back(ast_.Return("ACL_SUCCESS"));
  return ast_.DefineMethod("Om2Model", "ReleaseResources", {}, "aclError", body);
}

void ResourcesFileCodeGenerator::BuildReleaseResourcesMethodForControlTask(std::vector<BodyItem> &body,
                                                                           const RuntimeResourceSemantic &runtime) {
  if (runtime.has_label_switch) {
    auto label = ast_.Var("auto &", "label");
    (void)body.emplace_back(ast_.RangeFor(
        label, label_switch_label_list_,
        {
            ast_.If(label.Attr("second") != nullptr, {ChkStatus(AclrtDestroyLabelList(label.Attr("second")))}),
        }));
  }
  if (runtime.has_label_goto) {
    auto label_goto_ex_index_value = ast_.Var("auto &", "label_goto_ex_index_value");
    (void)body.emplace_back(ast_.RangeFor(label_goto_ex_index_value, label_goto_ex_index_values_,
                                          {
                                              ChkStatus(AclrtFree(label_goto_ex_index_value)),
                                          }));
    auto label_goto_arg = ast_.Var("auto &", "label_goto_arg");
    auto arg_addr = ast_.Var("void *", "arg_addr");
    (void)body.emplace_back(
        ast_.RangeFor(label_goto_arg, label_goto_args_,
                      {ast_.VarDecl(arg_addr, label_goto_arg.Attr("second").Attr("first")),
                       ast_.If(arg_addr != nullptr, {ChkStatus(AclrtDestroyLabelList(arg_addr))})}));
    (void)body.emplace_back(label_goto_args_.Clear());
  }
}
}  // namespace ge
