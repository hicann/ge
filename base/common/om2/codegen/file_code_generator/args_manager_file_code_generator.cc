/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "common/om2/codegen/file_code_generator/args_manager_file_code_generator.h"

namespace ge {
ArgsManagerFileCodeGenerator::ArgsManagerFileCodeGenerator(AstBuildContext &ast)
    : CodeGeneratorBase(ast),
      args_sizes_(ast_.Var("std::array<int64_t,  static_cast<size_t>(4)>", "args_sizes_")),
      args_info_(ast_.Var("std::vector<ArgsInfo>", "args_info_")),
      host_args_(ast_.Var("std::vector<uint8_t>", "host_args_")),
      dev_args_(ast_.Var("void *", "dev_args_")),
      input_index_to_allocation_ids_(ast_.Var("std::vector<uint32_t>", "input_index_to_allocation_ids_")),
      output_index_to_allocation_ids_(ast_.Var("std::vector<uint32_t>", "output_index_to_allocation_ids_")),
      allocation_ids_to_model_args_refresh_infos_addr_all_(ast_.Var(
          "std::vector<std::vector<ArgsRefreshInfo>>", "allocation_ids_to_model_args_refresh_infos_addr_all_")) {}

MethodDef *ArgsManagerFileCodeGenerator::BuildInitMethod(const Om2CodegenModel &codegen_model) {
  std::vector<Arg> args_size_items;
  args_size_items.reserve(codegen_model.args_table.model_args_semantic.size());
  std::vector<Arg> args_size_temp_items;
  for (const auto &model_arg : codegen_model.args_table.model_args_semantic) {
    args_size_temp_items.push_back(model_arg.len);
  }
  args_size_items.push_back(args_size_temp_items);

  std::vector<Arg> args_info_items;
  args_info_items.reserve(codegen_model.args_table.entries.size());
  for (const auto &entry : codegen_model.args_table.entries) {
    args_info_items.push_back(
        {GetHostArgAddr(entry.host_offset, 0), GetDevArgAddr(entry.host_offset, 0), entry.args_size});
  }

  std::vector<Arg> input_index_to_allocation_ids_items;
  for (const auto &entry : codegen_model.args_table.input_index_to_allocation_ids) {
    input_index_to_allocation_ids_items.push_back(entry);
  }

  std::vector<Arg> output_index_to_allocation_ids_items;
  for (const auto &entry : codegen_model.args_table.output_index_to_allocation_ids) {
    output_index_to_allocation_ids_items.push_back(entry);
  }

  std::vector<Arg> allocation_ids_to_model_args_refresh_infos_items;
  for (const auto &entry : codegen_model.args_table.allocation_ids_to_model_args_refresh_infos_addr_all_semantic) {
    std::vector<Arg> temp_items;
    temp_items.reserve(entry.size());
    for (const auto &entry2 : entry) {
      temp_items.push_back({entry2.base_args_offset, entry2.offset, static_cast<int32_t>(entry2.placement)});
    }
    allocation_ids_to_model_args_refresh_infos_items.push_back(temp_items);
  }

  auto i = ast_.Var("size_t", "i");

  return ast_.DefineMethod(
      "Om2ArgsTable", "Init", {}, "aclError",
      {
          ast_.Assign(args_sizes_, args_size_items),
          ast_.For(
              ast_.VarDecl(i, 0), i < args_sizes_.Size(), ast_.PostInc(i),
              {
                  ast_.If(args_sizes_[i] > 0,
                          {
                              host_args_[i].Clear(),
                              host_args_[i].Resize(args_sizes_[i]),
                              ChkStatus(AclrtMalloc(dev_args_[i].Addr(), args_sizes_[i], "ACL_MEM_MALLOC_HUGE_FIRST")),
                          }),
              }),
          ast_.Assign(args_info_, args_info_items),

          input_index_to_allocation_ids_.Clear(),
          input_index_to_allocation_ids_.Resize(codegen_model.args_table.input_index_to_allocation_ids.size()),
          ast_.Assign(input_index_to_allocation_ids_, input_index_to_allocation_ids_items),

          output_index_to_allocation_ids_.Clear(),
          output_index_to_allocation_ids_.Resize(codegen_model.args_table.output_index_to_allocation_ids.size()),
          ast_.Assign(output_index_to_allocation_ids_, output_index_to_allocation_ids_items),

          allocation_ids_to_model_args_refresh_infos_addr_all_.Clear(),
          allocation_ids_to_model_args_refresh_infos_addr_all_.Resize(
              codegen_model.args_table.allocation_ids_to_model_args_refresh_infos_addr_all_semantic.size()),
          ast_.Assign(allocation_ids_to_model_args_refresh_infos_addr_all_,
                      allocation_ids_to_model_args_refresh_infos_items),
          //
          ast_.Return("ACL_SUCCESS"),
      });
}

MethodDef *ArgsManagerFileCodeGenerator::BuildDestructor() {
  auto i = ast_.Var("size_t", "i");
  return ast_.DefineMethod("Om2ArgsTable", "~Om2ArgsTable", {}, "",
                           {
                               ast_.For(ast_.VarDecl(i, 0), i < args_sizes_.Size(), ast_.PostInc(i),
                                        {
                                            ast_.If(dev_args_[i] != nullptr, {AclrtFree(dev_args_[i])}),
                                        }),

                           });
}

MethodDef *ArgsManagerFileCodeGenerator::BuildGetArgsInfoMethod() {
  auto index = ast_.Var("size_t", "index");
  return ast_.DefineMethod("Om2ArgsTable", "GetArgsInfo", {index}, "ArgsInfo *",
                           {
                               ast_.If(index >= args_info_.Size(), {ast_.Return(nullptr)}),
                               ast_.Return(args_info_[index].Addr()),
                           });
}

MethodDef *ArgsManagerFileCodeGenerator::BuildGetDevArgAddrMethod() {
  auto offset = ast_.Var("size_t", "offset");
  auto args_type = ast_.Var("int32_t", "args_type");
  return ast_.DefineMethod("Om2ArgsTable", "GetDevArgAddr", {offset, args_type}, "void *",
                           {
                               ast_.If(offset >= args_sizes_[args_type], {ast_.Return(nullptr)}),
                               ast_.Return(GetAddr(dev_args_[args_type], offset)),
                           });
}

MethodDef *ArgsManagerFileCodeGenerator::BuildGetHostArgAddrMethod() {
  auto offset = ast_.Var("size_t", "offset");
  auto args_type = ast_.Var("int32_t", "args_type");
  return ast_.DefineMethod("Om2ArgsTable", "GetHostArgAddr", {offset, args_type}, "void *",
                           {
                               ast_.If(offset >= args_sizes_[args_type], {ast_.Return(nullptr)}),
                               ast_.Return(GetAddr(host_args_[args_type].Data(), offset)),
                           });
}

MethodDef *ArgsManagerFileCodeGenerator::BuildUpdateHostArgsMethod() {
  auto type = ast_.Var("int32_t", "type");
  auto index = ast_.Var("size_t", "index");
  auto addr = ast_.Var("const uintptr_t", "addr");
  auto host_addr = ast_.Var("void *", "host_addr");

  auto allocation_id = ast_.Var("int32_t", "allocation_id");
  auto infos = ast_.Var("const auto&", "infos");
  auto info = ast_.Var("const auto&", "info");
  auto base_ptr = ast_.Var("const uint8_t*", "base_ptr");
  auto target_addr = ast_.Var("const uint8_t*", "target_addr");

  return ast_.DefineMethod(
      "Om2ArgsTable", "UpdateHostArgs", {type, index, addr}, "aclError",
      {
          ast_.If(
              type == 0,
              {
                  ast_.VarDecl(allocation_id, input_index_to_allocation_ids_.At(index)),
                  ast_.VarDecl(infos, allocation_ids_to_model_args_refresh_infos_addr_all_.At(allocation_id)),
                  ast_.VarDecl(base_ptr, ast_.ReinterpretCast("const uint8_t*", addr)),
                  ast_.RangeFor(info, infos,
                                {
                                    BodyItem(ast_.VarDecl(host_addr, GetAddr(host_args_[info.Attr("args_type")].Data(),
                                                                             info.Attr("args_offset")))),
                                    BodyItem(ast_.VarDecl(target_addr, base_ptr + info.Attr("offset"))),
                                    BodyItem(MemcpyS(host_addr, ast_.Sizeof(target_addr), target_addr.Addr(),
                                                     ast_.Sizeof(target_addr))),
                                }),
              }),
          ast_.If(
              type == 1,
              {
                  ast_.VarDecl(allocation_id, output_index_to_allocation_ids_.At(index)),
                  ast_.VarDecl(infos, allocation_ids_to_model_args_refresh_infos_addr_all_.At(allocation_id)),
                  ast_.VarDecl(base_ptr, ast_.ReinterpretCast("const uint8_t*", addr)),
                  ast_.RangeFor(info, infos,
                                {
                                    BodyItem(ast_.VarDecl(host_addr, GetAddr(host_args_[info.Attr("args_type")].Data(),
                                                                             info.Attr("args_offset")))),
                                    BodyItem(ast_.VarDecl(target_addr, base_ptr + info.Attr("offset"))),
                                    BodyItem(MemcpyS(host_addr, ast_.Sizeof(target_addr), target_addr.Addr(),
                                                     ast_.Sizeof(target_addr))),
                                }),
              }),
          ast_.Return("ACL_SUCCESS"),
      });
}

MethodDef *ArgsManagerFileCodeGenerator::BuildCopyArgsToDeviceMethod() {
  return ast_.DefineMethod("Om2ArgsTable", "CopyArgsToDevice", {}, "aclError",
                           {
                               ChkStatus(AclrtMemcpy(dev_args_[0], args_sizes_[0], host_args_[0].Data(), args_sizes_[0],
                                                     "ACL_MEMCPY_HOST_TO_DEVICE")),
                               ast_.Return("ACL_SUCCESS"),
                           });
}

ExprRef ArgsManagerFileCodeGenerator::GetHostArgAddr(Arg offset, Arg args_type) {
  return ast_.Call("GetHostArgAddr", {offset, args_type});
}

ExprRef ArgsManagerFileCodeGenerator::GetDevArgAddr(Arg offset, Arg args_type) {
  return ast_.Call("GetDevArgAddr", {offset, args_type});
}
}  // namespace ge
