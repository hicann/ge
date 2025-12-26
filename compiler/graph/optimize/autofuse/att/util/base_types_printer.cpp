/**
 * Copyright (C) Huawei Technologies Co., Ltd. 2024 All rights reserved.
 *
 * Licensed unde the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the license is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and limitations under the License.
 */
#include "base_types_printer.h"

namespace ge {
void to_json(nlohmann::json &j, const Expression &arg) {
  auto expr = arg.IsValid() ? "" : std::string(arg.Str().get());
  j = nlohmann::json{
      {expr},
  };
}
}
namespace att {
const std::string AddAnotationBlock(std::string strs, std::string indent) {
  std::string str = indent + "/*\n" + strs + indent + "*/\n";
  return str;
}
const std::string AddAnotationLine(std::string strs, std::string indent) {
  std::string str = indent + "// " + strs;
  return str;
}

void ScanContainer(const Expr &container, const Expr &container_expr, std::set<std::string> &arg_names, ExprExprMap &param_map) {
  for (const auto &arg : container_expr.FreeSymbols()) {
    arg_names.insert(Str(arg));
  }
  param_map[container] = container_expr;
}

void AnalysisArg(const Expr &arg, const ExprExprMap &container_expr, std::set<std::string> &arg_names, ExprExprMap &param_map) {
  if (arg.GetExprType() == ge::ExprType::kExprVariable) {
    auto iter1 = container_expr.find(arg);
    if (iter1 != container_expr.end()) {
      ScanContainer(arg, iter1->second, arg_names, param_map);
    } else {
      arg_names.insert(Str(arg));
    }
  }
}

const std::string GenRelatedVars(const std::vector<Expr> &funcs, const ExprExprMap &container_expr, const std::map<Expr, std::vector<Expr>, ExprCmp> &args) {
  std::string ret;
  std::set<std::string> arg_names;
  ExprExprMap params_map;
  for (const auto &func : funcs) {
    GELOGD("The func is [%s].", Str(func).c_str());
    for (const auto &arg : func.FreeSymbols()) {
      GELOGD("Analysis arg [%s].", Str(arg).c_str());
      auto iter = args.find(arg);
      if (iter != args.end()) {
        for (const auto &item : iter->second) {
          GELOGD("Insert arg [%s].", Str(item).c_str());
          arg_names.insert(Str(item));
        }
      } else {
        AnalysisArg(arg, container_expr, arg_names, params_map);
      }
    }
  }
  for (const auto &arg_name : arg_names) {
    ret += "    double " + arg_name + " = tiling_data.get_" + arg_name + "();\n";
  }
  for (const auto &pair : params_map) {
    ret += "    double " + Str(pair.first) + " = " + Str(pair.second) + ";\n";
  }
  return ret;
}

const std::string GenBufRelatedVars(const Expr &func, const ExprExprMap &container_expr) {
  std::string ret;
  std::set<std::string> arg_names;
  ExprExprMap params_map;
  for (const auto &arg : func.FreeSymbols()) {
    AnalysisArg(arg, container_expr, arg_names, params_map);
  }
  for (const auto &arg_name : arg_names) {
    ret += "    double " + arg_name + " = tiling_data.get_" + arg_name + "();\n";
  }
  
  std::map<std::string, ASTNode> ast_expr_map;
  Optimizer ast_optimizer;
  for (const auto &pair : params_map) {
    Parser parser(Str(pair.second)); 
    ASTPtr ast = parser.Parse();
    ast_optimizer.Optimize(ast);
    ast_expr_map.emplace(Str(pair.first), *ast.get());
  }
  std::string tmp_vars = ast_optimizer.GenerateCode();
  ret += tmp_vars;
  for (const auto &pair : ast_expr_map) {
    auto &ast = pair.second;
    std::string return_expr = ast_optimizer.RebuildExpr(ast, 1);
    ret += "    double " + pair.first + " = " + return_expr + ";\n";
  }
  Parser parser(Str(func));
  ASTPtr ast = parser.Parse();
  ast_optimizer.Optimize(ast);
  std::string func_tmp_vars = ast_optimizer.GenerateCode();
  ret += func_tmp_vars;
  std::string func_return_expr = ast_optimizer.RebuildExpr(*ast.get(), 1);
  ret += "    return " + func_return_expr + ";\n";
  return ret;
}

std::string BaseTypeUtils::DumpHardware(const HardwareDef hardware) {
  const auto &hardware_name_iter = kHardwareNameMap.find(hardware);
  if (hardware_name_iter == kHardwareNameMap.cend()) {
    return kHardwareNameMap.at(HardwareDef::HARDWAREERR);
  }
  return hardware_name_iter->second;
}

std::string BaseTypeUtils::DtypeToStr(ge::DataType dtype) {
  const std::map<ge::DataType, const ge::char_t *> kTypeName = {
      {ge::DT_FLOAT, "float32"}, {ge::DT_FLOAT16, "float16"}, {ge::DT_BF16, "bfloat16"}, {ge::DT_INT8, "int8"},
      {ge::DT_UINT8, "uint8"},   {ge::DT_INT16, "int16"},     {ge::DT_UINT16, "uint16"}, {ge::DT_INT32, "int32"},
      {ge::DT_UINT32, "uint32"}, {ge::DT_INT64, "int64"},     {ge::DT_UINT64, "uint64"}, {ge::DT_DOUBLE, "double"}};
  const auto &type_name_iter = kTypeName.find(dtype);
  if (type_name_iter == kTypeName.end()) {
    return "unknown";
  }
  return type_name_iter->second;
}

}  // namespace att
