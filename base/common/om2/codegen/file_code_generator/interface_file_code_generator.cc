/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "common/om2/codegen/file_code_generator/interface_file_code_generator.h"

namespace ge {
StructDecl *InterfaceFileCodeGenerator::BuildBinDataInfoStruct() {
  return ast_.Struct("BinDataInfo", {
                                        ast_.Field("const void *", "data"),
                                        ast_.Field("size_t", "size"),
                                    });
}

StructDecl *InterfaceFileCodeGenerator::BuildAicpuParamHeadStruct() {
  return ast_.Struct("AicpuParamHead", {
                                           ast_.Field("uint32_t", "length"),
                                           ast_.Field("uint32_t", "ioAddrNum"),
                                           ast_.Field("uint32_t", "extInfoLength"),
                                           ast_.Field("uint64_t", "extInfoAddr"),
                                       });
}

StructDecl *InterfaceFileCodeGenerator::BuildAicpuSessionInfoStruct() {
  return ast_.Struct("AicpuSessionInfo", {
                                             ast_.Field("uint64_t", "sessionId"),
                                             ast_.Field("uint64_t", "kernelId"),
                                             ast_.Field("bool", "sessFlag"),
                                         });
}

StructDecl *InterfaceFileCodeGenerator::BuildTfAiCpuExInfoStruct() {
  return ast_.Struct("TfAiCpuExInfo", {
                                          ast_.Field("uint32_t", "fwkKernelType"),
                                          ast_.Field("uint32_t", "fwkOperateType"),
                                          ast_.Field("uint64_t", "sessionID"),
                                          ast_.Field("uint64_t", "stepIDAddr"),
                                          ast_.Field("uint64_t", "kernelID"),
                                          ast_.Field("uint64_t", "nodeDefLen"),
                                          ast_.Field("uint64_t", "nodeDefBuf"),
                                          ast_.Field("uint64_t", "funDefLibLen"),
                                          ast_.Field("uint64_t", "funDefLibBuf"),
                                          ast_.Field("uint64_t", "inputOutputLen"),
                                          ast_.Field("uint64_t", "inputOutputBuf"),
                                          ast_.Field("uint64_t", "workspaceBaseAddr"),
                                          ast_.Field("uint64_t", "inputOutputAddr"),
                                          ast_.Field("uint64_t", "extInfoLen"),
                                          ast_.Field("uint64_t", "extInfoAddr"),
                                      });
}

StructDecl *InterfaceFileCodeGenerator::BuildArgsInfoStruct() {
  return ast_.Struct("ArgsInfo", {
                                     ast_.Field("void *", "host_addr"),
                                     ast_.Field("void *", "dev_addr"),
                                     ast_.Field("size_t", "size"),
                                 });
}

StructDecl *InterfaceFileCodeGenerator::BuildArgsRefreshInfoStruct() {
  return ast_.Struct("ArgsRefreshInfo", {
                                            ast_.Field("uint64_t", "args_offset"),
                                            ast_.Field("uint64_t", "offset"),
                                            ast_.Field("int32_t", "args_type"),
                                        });
}

FunctionDef *InterfaceFileCodeGenerator::BuildAclrtMallocFunction() const {
  auto ptr = ast_.Var("void **", "ptr");
  auto size = ast_.Var("size_t", "size");
  auto mem_type = ast_.Var("uint32_t", "mem_type");
  auto module_id = ast_.Var("uint16_t", "module_id");
  auto attr = ast_.Var("aclrtMallocAttribute", "attr");
  auto cfg = ast_.Var("aclrtMallocConfig", "cfg");

  std::vector<BodyItem> body;
  body.push_back(ast_.Assign(ast_.Deref(ptr), nullptr));
  body.push_back(ast_.If(size == ast_.UInt(0), {ast_.Return("ACL_SUCCESS")}));
  body.push_back(ast_.VarDecl(attr));
  body.push_back(ast_.Assign(attr.Attr("attr"), "ACL_RT_MEM_ATTR_MODULE_ID"));
  body.push_back(ast_.Assign(attr.Attr("value").Attr("moduleId"), module_id));
  body.push_back(ast_.VarDecl(cfg));
  body.push_back(ast_.Assign(cfg.Attr("attrs"), attr.Addr()));
  body.push_back(ast_.Assign(cfg.Attr("numAttrs"), ast_.UInt(1)));

  std::vector<BodyItem> switch_body;
  switch_body.push_back(ast_.Case("RT_MEMORY_TS"));
  switch_body.push_back(
      ast_.Return(ast_.Call("aclrtMallocForTaskScheduler", {ptr, size, "ACL_MEM_MALLOC_HUGE_FIRST", cfg.Addr()})));
  switch_body.push_back(ast_.Case("RT_MEMORY_HOST"));
  switch_body.push_back(ast_.Return(ast_.Call("aclrtMallocHostWithCfg", {ptr, size, cfg.Addr()})));
  switch_body.push_back(ast_.Case("RT_MEMORY_P2P_HBM"));
  switch_body.push_back(ast_.Case("RT_MEMORY_P2P_DDR"));
  switch_body.push_back(
      ast_.Return(ast_.Call("aclrtMallocWithCfg", {ptr, size, "ACL_MEM_MALLOC_HUGE_FIRST_P2P", cfg.Addr()})));
  switch_body.push_back(ast_.Case("RT_MEMORY_DDR"));
  switch_body.push_back(ast_.Case("RT_MEMORY_DDR_NC"));
  switch_body.push_back(
      ast_.Return(ast_.Call("aclrtMallocWithCfg", {ptr, size, "ACL_MEM_TYPE_LOW_BAND_WIDTH", cfg.Addr()})));
  switch_body.push_back(ast_.Case("RT_MEMORY_HBM"));
  switch_body.push_back(ast_.Case(nullptr));
  switch_body.push_back(
      ast_.Return(ast_.Call("aclrtMallocWithCfg", {ptr, size, "ACL_MEM_TYPE_HIGH_BAND_WIDTH", cfg.Addr()})));

  body.push_back(ast_.Switch(mem_type, switch_body));
  return ast_.DefineFunction("AclrtMalloc", {ptr, size, mem_type, module_id}, "inline aclError", ast_.Body(body));
}

ClassDecl *InterfaceFileCodeGenerator::BuildOm2ArgsTableClass() {
  std::vector<DeclNode *> items = {
      ast_.Public(),
      ast_.DeclareMethod("Om2ArgsTable", {}, "", " = default"),
      ast_.DeclareMethod("~Om2ArgsTable", {}, ""),
      ast_.DeclareMethod("Init", {}, "aclError"),
      ast_.DeclareMethod("GetArgsInfo", {ast_.Var("size_t", "index")}, "ArgsInfo *"),
      ast_.DeclareMethod("GetDevArgAddr", {ast_.Var("size_t", "offset"), ast_.Var("int32_t", "args_type")}, "void *"),
      ast_.DeclareMethod("GetHostArgAddr", {ast_.Var("size_t", "offset"), ast_.Var("int32_t", "args_type")}, "void *"),

      ast_.DeclareMethod(
          "UpdateHostArgs",
          {ast_.Var("int32_t", "type"), ast_.Var("size_t", "index"), ast_.Var("const uintptr_t", "addr")}, "aclError"),
      ast_.DeclareMethod("CopyArgsToDevice", {ast_.Var("void *", "stream"), ast_.Var("bool", "is_async")}, "aclError"),
      ast_.Private(),
      ast_.Field("std::array<int64_t,  static_cast<size_t>(4)>", "args_sizes_{}"),
      ast_.Field("std::array<std::vector<uint8_t>, static_cast<size_t>(4)>", "host_args_{}"),
      ast_.Field("std::array<void *, static_cast<size_t>(4)>", "dev_args_{}"),
      ast_.Field("std::vector<ArgsInfo>", "args_info_"),
      ast_.Field("std::vector<uint32_t>", "input_index_to_allocation_ids_"),
      ast_.Field("std::vector<uint32_t>", "output_index_to_allocation_ids_"),
      ast_.Field("std::vector<std::vector<ArgsRefreshInfo>>", "allocation_ids_to_model_args_refresh_infos_addr_all_"),
      //
  };
  return ast_.Class("Om2ArgsTable", items);
}

ClassDecl *InterfaceFileCodeGenerator::BuildOm2ModelClass(const Om2CodegenModel &codegen_model) {
  const auto &runtime = codegen_model.runtime;
  std::vector<DeclNode *> items = {
      ast_.Public(),
      ast_.DeclareMethod(
          "Om2Model",
          {ast_.Var("const char **", "bin_files"), ast_.Var("const void **", "bin_data"),
           ast_.Var("uint64_t *", "bin_size"), ast_.Var("size_t", "bin_num"), ast_.Var("void **", "constants"),
           ast_.Var("void **", "var_addrs"), ast_.Var("void *", "work_ptr"), ast_.Var("uint64_t *", "session_id"),
           ast_.Var("uint32_t", "model_id"), ast_.Var("void *", "instance_handle"),
           ast_.Var("const GertModelLoadCallbacks *", "callbacks"), ast_.Var("int32_t", "priority")},
          ""),
      ast_.DeclareMethod("~Om2Model", {}, ""),
      ast_.DeclareMethod("InitResources",
                         {ast_.Var("uint64_t", "reuse_zero_copy"),
                          ast_.Var("const GertModelExternalResources &", "external_resources")},
                         "aclError"),
      ast_.DeclareMethod("RegisterKernels", {}, "aclError"),
      ast_.DeclareMethod("Load", {ast_.Var("const GertModelLoadCallbacks *", "callbacks")}, "aclError"),
      ast_.DeclareMethod("GetRtModelHandle", {}, "aclmdlRI"),
      ast_.DeclareMethod(
          "Run",
          {ast_.Var("size_t", "input_count"), ast_.Var("gert::Tensor **", "input_data"),
           ast_.Var("size_t", "output_count"), ast_.Var("gert::Tensor **", "output_data"),
           ast_.Var("int32_t", "stream_sync_timeout"), ast_.Var("const GertModelRunCallbacks *", "run_callbacks")},
          "aclError"),
      ast_.DeclareMethod(
          "RunAsync",
          {ast_.Var("aclrtStream &", "exe_stream"), ast_.Var("size_t", "input_count"),
           ast_.Var("gert::Tensor **", "input_data"), ast_.Var("size_t", "output_count"),
           ast_.Var("gert::Tensor **", "output_data"), ast_.Var("const GertModelRunCallbacks *", "run_callbacks")},
          "aclError"),
      ast_.DeclareMethod("ReleaseResources", {}, "aclError"),
      ast_.Private(),
      ast_.Field("void **", "constants_"),
      ast_.Field("void **", "var_addrs_"),
      ast_.Field("aclmdlRI", "model_handle_"),
      ast_.Field("bool", "is_external_rt_model_"),
  };
  DealParamForOm2ModelClass(items, runtime);
  items.push_back(ast_.Field("void *", "total_dev_mem_ptr_"));
  items.push_back(ast_.Field("bool", "owns_total_dev_mem_", false));
  items.push_back(ast_.Field("bool", "is_stream_list_bind_"));
  items.push_back(ast_.Field("std::unordered_map<std::string, BinDataInfo>", "bin_info_map_"));
  items.push_back(ast_.Field("Om2ArgsTable", "args_table_"));
  items.push_back(ast_.Field("uint64_t *", "session_id_"));
  items.push_back(ast_.Field("uint32_t", "model_id_"));
  items.push_back(ast_.Field("void *", "instance_handle_"));
  items.push_back(ast_.Field("GertModelLoadCallbacks", "callbacks_"));
  items.push_back(ast_.Field("uint64_t", "kernel_id_"));
  items.push_back(ast_.Field("std::vector<void *>", "dev_ext_info_mem_ptrs_"));
  items.push_back(ast_.Field("std::map<uint32_t, void *>", "mem_event_id_mem_map_"));
  items.push_back(ast_.Field("void *", "overflow_addr_"));
  items.push_back(ast_.Field("std::vector<void *>", "dev_dynamic_mem_ptrs_"));
  items.push_back(ast_.Field("void *", "session_scope_mem_ptr_"));
  items.push_back(ast_.Field("int32_t", "priority_"));
  return ast_.Class("Om2Model", items);
}

void InterfaceFileCodeGenerator::DealParamForOm2ModelClass(std::vector<DeclNode *> &items,
                                                           const RuntimeResourceSemantic &runtime) {
  if (runtime.kernel_bin_num > 0U) {
    items.push_back(ast_.Field("std::vector<std::string>", "bin_ids_"));
    items.push_back(ast_.Field("std::vector<aclrtBinHandle>", "bin_handles_"));
  }
  items.push_back(ast_.Field("std::vector<aclrtFuncHandle>", "func_handles_"));
  items.push_back(ast_.Field("std::vector<aclrtStream>", "stream_list_"));
  items.push_back(ast_.Field("bool", "is_external_streams_"));
  items.push_back(ast_.Field("std::vector<aclrtNotify>", "notify_list_"));
  items.push_back(ast_.Field("bool", "is_external_notifies_"));
  items.push_back(ast_.Field("std::vector<aclrtEvent>", "event_list_"));
  items.push_back(ast_.Field("bool", "is_external_events_"));
  items.push_back(ast_.Field("std::vector<aclrtLabel>", "label_list_"));
  items.push_back(ast_.Field("bool", "is_external_labels_"));
  if (runtime.label_num > 0U) {
    items.push_back(ast_.Field("aclrtLabelList", "aclrt_label_list_"));
  }
  items.push_back(ast_.Field("std::vector<aclrtLabel>", "label_used_"));
  items.push_back(ast_.Field("std::map<uint32_t, aclrtLabelList>", "label_switch_label_list_"));
  items.push_back(ast_.Field("std::map<uint32_t, std::pair<void *, uint32_t>>", "label_goto_args_"));
  items.push_back(ast_.Field("std::map<uint32_t, aclrtLabelList>", "label_goto_ex_label_list_"));
  if (runtime.has_label_switch) {
    items.push_back(ast_.DeclareMethod(
        "CreateLabelListForLabelSwitch",
        {ast_.Var("uint32_t", "op_index"), ast_.Var("std::vector<uint32_t>", "label_list_indexs")}, "aclError"));
  }
  if (runtime.has_label_goto) {
    items.push_back(ast_.DeclareMethod("CreateLabelListForLabelGotoEx",
                                       {ast_.Var("uint32_t", "op_index"), ast_.Var("uint32_t", "label_index")},
                                       "aclError"));
    items.push_back(ast_.Field("std::vector<void *>", "label_goto_ex_index_values_"));
  }
}

std::vector<DeclNode *> InterfaceFileCodeGenerator::BuildRtForwardDecls() {
  // rtLabelDevInfo / rtCmoAddrTaskLaunch 不在 rt_external*.h 中，前向声明以供 sizeof/调用使用
  return {
      ast_.Struct("rtLabelDevInfo",
                  {
                      ast_.Field("uint16_t", "modelId"),
                      ast_.Field("uint16_t", "streamId"),
                      ast_.Field("uint16_t", "labelId"),
                      ast_.Field("uint16_t", "reserved[7]"),
                  }),
      ast_.ExternBlock("C",
                       {
                           ast_.DeclareFunction("rtCmoAddrTaskLaunch",
                                                {ast_.Var("void *", "cmoAddrInfo"), ast_.Var("uint64_t", "destMax"),
                                                 ast_.Var("rtCmoOpCode_t", "cmoOpCode"), ast_.Var("rtStream_t", "stm"),
                                                 ast_.Var("uint32_t", "flag")},
                                                "rtError_t"),
                       }),
  };
}

std::vector<DeclNode *> InterfaceFileCodeGenerator::BuildExternalApiDecls() {
  return {
      ast_.TypeAlias("void *", "GertModelHandle"),
      ast_.DeclareFunction(
          "GertModelLoad",
          {ast_.Var("const struct GertModelLoadConfig *", "config"), ast_.Var("GertModelHandle *", "model_handle"),
           ast_.Var("struct GertModelLoadOutput *", "output")},
          "int32_t"),
      ast_.DeclareFunction(
          "GertModelRunAsync",
          {ast_.Var("GertModelHandle", "model_handle"), ast_.Var("aclrtStream", "stream"),
           ast_.Var("const struct GertModelRunConfig *", "config"), ast_.Var("struct GertModelRunOutput *", "output")},
          "int32_t"),
      ast_.DeclareFunction(
          "GertModelRun",
          {ast_.Var("GertModelHandle", "model_handle"), ast_.Var("const struct GertModelRunConfig *", "config"),
           ast_.Var("struct GertModelRunOutput *", "output")},
          "int32_t"),
      ast_.DeclareFunction(
          "GertModelUnload",
          {ast_.Var("GertModelHandle", "model_handle"), ast_.Var("const struct GertModelUnloadConfig *", "config"),
           ast_.Var("struct GertModelUnloadOutput *", "output")},
          "int32_t"),
      ast_.DeclareFunction("GertModelGetStreamNum", {}, "uint64_t"),
      ast_.DeclareFunction("GertModelGetStreamDesc",
                           {ast_.Var("uint32_t *", "stream_flags"), ast_.Var("uint64_t", "stream_num"),
                            ast_.Var("void *", "extended_attrs")},
                           "int32_t"),
      ast_.DeclareFunction("GertModelGetEventNum", {}, "uint64_t"),
      ast_.DeclareFunction("GertModelGetEventDesc",
                           {ast_.Var("uint32_t *", "event_flags"), ast_.Var("uint64_t", "event_num"),
                            ast_.Var("void *", "extended_attrs")},
                           "int32_t"),
      ast_.DeclareFunction("GertModelGetLabelNum", {}, "uint64_t"),
      ast_.DeclareFunction("GertModelGetNotifyNum", {}, "uint64_t"),
      ast_.DeclareFunction("GertModelGetNotifyDesc",
                           {ast_.Var("uint64_t *", "notify_flags"), ast_.Var("uint64_t", "notify_num"),
                            ast_.Var("void *", "extended_attrs")},
                           "int32_t"),
  };
}
}  // namespace ge
