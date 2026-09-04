/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "common/om2/codegen/emitter/cpp_emitter.h"
#include "common/om2/codegen/ast/ast_context.h"
#include "common/om2/codegen/ast/ast_build_context.h"
#include "common/om2/codegen/ast/ast_nodes.h"
#include "common/om2/codegen/file_code_generator/args_manager_file_code_generator.h"
#define private public
#include "common/om2/codegen/file_code_generator/load_and_run_file_code_generator.h"
#undef private
#include "common/om2/codegen/emitter/stable_parts/stable_part_provider.h"
#include "common/om2/codegen/task_code_builder/task_code_builder_util.h"
#include "common/om2/codegen/om2_code_printer.h"
#include "common/helper/om2/om2_utils.h"
#include "common/om2/codegen/om2_codegen_types.h"
#include "common/ge_common/ge_types.h"
#include "common/util/error_manager/error_manager.h"
#include "graph/ge_local_context.h"
#include "common/om2/rt_var_resource.h"
#include "framework/runtime/gert_model/gert_model_executor_types.h"
#include "framework/common/taskdown_common.h"

#include <gtest/gtest.h>

#include <cerrno>
#include <cstddef>
#include <cstdlib>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <sys/stat.h>
#include <sys/utsname.h>
#include <map>
#include <string>
#include <vector>
#include <unistd.h>
#include <type_traits>

namespace ge {
namespace {

template <typename T, typename = void>
struct HasLegacyLaunchCallback : std::false_type {};

template <typename T>
struct HasLegacyLaunchCallback<T, std::void_t<decltype(&T::report_task_preprocess)>> : std::true_type {};

template <typename T, typename = void>
struct HasLegacyPostCallback : std::false_type {};

template <typename T>
struct HasLegacyPostCallback<T, std::void_t<decltype(&T::report_task_postprocess)>> : std::true_type {};

template <typename T, typename = void>
struct HasLegacyDataDumpCallback : std::false_type {};

template <typename T>
struct HasLegacyDataDumpCallback<T, std::void_t<decltype(&T::get_data_dump_enabled)>> : std::true_type {};
class ScopedEnvVar {
 public:
  ScopedEnvVar(const char *name, const char *value) : name_(name) {
    const char *old_value = getenv(name);
    if (old_value != nullptr) {
      old_value_ = old_value;
      has_old_value_ = true;
    }
    (void)setenv(name, value, 1);
  }

  ~ScopedEnvVar() {
    if (has_old_value_) {
      (void)setenv(name_.c_str(), old_value_.c_str(), 1);
      return;
    }
    (void)unsetenv(name_.c_str());
  }

 private:
  std::string name_;
  std::string old_value_;
  bool has_old_value_ = false;
};

class ScopedUnsetEnvVar {
 public:
  explicit ScopedUnsetEnvVar(const char *name) : name_(name) {
    const char *old_value = getenv(name);
    if (old_value != nullptr) {
      old_value_ = old_value;
      has_old_value_ = true;
    }
    (void)unsetenv(name);
  }

  ~ScopedUnsetEnvVar() {
    if (has_old_value_) {
      (void)setenv(name_.c_str(), old_value_.c_str(), 1);
    }
  }

 private:
  std::string name_;
  std::string old_value_;
  bool has_old_value_ = false;
};

class ScopedGraphOptions {
 public:
  ScopedGraphOptions() : old_options_(GetThreadLocalContext().GetAllGraphOptions()) {}
  ~ScopedGraphOptions() {
    GetThreadLocalContext().SetGraphOption(old_options_);
  }

 private:
  std::map<std::string, std::string> old_options_;
};

class ScopedTempDir {
 public:
  ScopedTempDir() {
    char dir_template[] = "/tmp/ge_om2_codegen_ut_XXXXXX";
    const char *created_dir = mkdtemp(dir_template);
    root_ = (created_dir == nullptr) ? std::string() : std::string(created_dir);
  }

  ~ScopedTempDir() {
    for (auto file_iter = files_.rbegin(); file_iter != files_.rend(); ++file_iter) {
      (void)remove(file_iter->c_str());
    }
    for (auto dir_iter = dirs_.rbegin(); dir_iter != dirs_.rend(); ++dir_iter) {
      (void)rmdir(dir_iter->c_str());
    }
    if (!root_.empty()) {
      (void)rmdir(root_.c_str());
    }
  }

  const std::string &Root() const {
    return root_;
  }

  std::string Path(const std::string &relative_path) const {
    return root_ + "/" + relative_path;
  }

  bool CreateDir(const std::string &relative_path) {
    if (root_.empty()) {
      return false;
    }
    size_t begin = 0U;
    while (begin < relative_path.size()) {
      size_t end = relative_path.find('/', begin);
      if (end == std::string::npos) {
        end = relative_path.size();
      }
      const std::string dir = Path(relative_path.substr(0U, end));
      const int32_t ret = mkdir(dir.c_str(), 0755);
      if ((ret != 0) && (errno != EEXIST)) {
        return false;
      }
      if (ret == 0) {
        dirs_.push_back(dir);
      }
      begin = end + 1U;
    }
    return true;
  }

  bool WriteFile(const std::string &relative_path, const std::string &content, const mode_t mode = 0644) {
    const std::string path = Path(relative_path);
    std::ofstream output(path, std::ios::out | std::ios::binary);
    if (!output.is_open()) {
      return false;
    }
    output << content;
    output.close();
    if (chmod(path.c_str(), mode) != 0) {
      return false;
    }
    files_.push_back(path);
    return true;
  }

  bool Symlink(const std::string &relative_path, const std::string &target_path) {
    const std::string path = Path(relative_path);
    if (symlink(target_path.c_str(), path.c_str()) != 0) {
      return false;
    }
    files_.push_back(path);
    return true;
  }

 private:
  std::string root_;
  std::vector<std::string> files_;
  std::vector<std::string> dirs_;
};

class ScopedStdoutCapture {
 public:
  ScopedStdoutCapture() {
    capture_file_ = tmpfile();
    if (capture_file_ == nullptr) {
      return;
    }
    saved_stdout_fd_ = dup(STDOUT_FILENO);
    if (saved_stdout_fd_ < 0) {
      CloseCaptureFile();
      return;
    }
    (void)fflush(stdout);
    if (dup2(fileno(capture_file_), STDOUT_FILENO) < 0) {
      CloseSavedStdout();
      CloseCaptureFile();
    }
  }

  ~ScopedStdoutCapture() {
    (void)Stop();
  }

  std::string Stop() {
    if (stopped_) {
      return output_;
    }
    stopped_ = true;
    if (saved_stdout_fd_ >= 0) {
      (void)fflush(stdout);
      (void)dup2(saved_stdout_fd_, STDOUT_FILENO);
      CloseSavedStdout();
    }
    if (capture_file_ != nullptr) {
      (void)fflush(capture_file_);
      (void)fseek(capture_file_, 0, SEEK_SET);
      char buffer[4096];
      while (!feof(capture_file_)) {
        const size_t read_size = fread(buffer, 1, sizeof(buffer), capture_file_);
        output_.append(buffer, read_size);
      }
      CloseCaptureFile();
    }
    return output_;
  }

 private:
  void CloseSavedStdout() {
    if (saved_stdout_fd_ >= 0) {
      (void)close(saved_stdout_fd_);
      saved_stdout_fd_ = -1;
    }
  }

  void CloseCaptureFile() {
    if (capture_file_ != nullptr) {
      (void)fclose(capture_file_);
      capture_file_ = nullptr;
    }
  }

  FILE *capture_file_ = nullptr;
  int32_t saved_stdout_fd_ = -1;
  bool stopped_ = false;
  std::string output_;
};

std::string EmitNode(const AstNode &node) {
  CppEmitter emitter;
  std::string output;
  EXPECT_EQ(node.Accept(emitter, output), SUCCESS);
  return output;
}

std::string EmitBodyItems(AstBuildContext &ast, const std::vector<BodyItem> &items) {
  CppEmitter emitter;
  std::stringstream output;
  const auto nodes = ast.Body(items);
  for (const auto *node : nodes) {
    if (node == nullptr) {
      ADD_FAILURE() << "unexpected null stmt";
      continue;
    }
    std::string code;
    EXPECT_EQ(node->Accept(emitter, code), SUCCESS);
    output << code << '\n';
  }
  return output.str();
}

void ExpectContainsAll(const std::string &output, const std::vector<std::string> &snippets) {
  for (const auto &snippet : snippets) {
    EXPECT_NE(output.find(snippet), std::string::npos) << snippet << "\n=== output ===\n" << output;
  }
}
}  // namespace

class Om2CodegenUt : public testing::Test {
 public:
  void SetUp() override {}
  void TearDown() override {}
};

TEST_F(Om2CodegenUt, AstNodes_AllPublicInterfaces_Ok) {
  AstContext ctx;

  auto *param_x = ParamDecl::Create(ctx, "int", "x");
  auto *param_y = ParamDecl::Create(ctx, "int", "y");
  ASSERT_NE(param_x, nullptr);
  ASSERT_NE(param_y, nullptr);
  EXPECT_EQ(std::string(param_x->GetTypeSpec().Data(), param_x->GetTypeSpec().Length()), "int");
  EXPECT_EQ(std::string(param_x->GetName().Data(), param_x->GetName().Length()), "x");

  auto *ident_x = IdentifierExpr::Create(ctx, "x");
  auto *ident_y = IdentifierExpr::Create(ctx, "y");
  auto *ident_obj = IdentifierExpr::Create(ctx, "obj");
  auto *ident_ptr = IdentifierExpr::Create(ctx, "ptr");
  auto *ident_vec = IdentifierExpr::Create(ctx, "vec");
  auto *ident_consume = IdentifierExpr::Create(ctx, "Consume");
  auto *ident_runner = IdentifierExpr::Create(ctx, "runner");
  auto *ident_guard = IdentifierExpr::Create(ctx, "guard");
  auto *lit_int = LiteralExpr::CreateInt(ctx, 7, LiteralExpr::IntSuffix::kU);
  auto *lit_bool = LiteralExpr::CreateBool(ctx, true);
  auto *lit_str = LiteralExpr::CreateString(ctx, "txt");
  auto *lit_null = LiteralExpr::CreateNullptr(ctx);
  ASSERT_NE(ident_x, nullptr);
  ASSERT_NE(lit_int, nullptr);
  EXPECT_EQ(lit_int->GetKind(), LiteralExpr::Kind::kInt);
  EXPECT_EQ(lit_int->GetIntValue(), 7);
  EXPECT_EQ(lit_int->GetIntSuffix(), LiteralExpr::IntSuffix::kU);
  EXPECT_EQ(lit_bool->GetKind(), LiteralExpr::Kind::kBool);
  EXPECT_TRUE(lit_bool->GetBoolValue());
  EXPECT_EQ(lit_str->GetKind(), LiteralExpr::Kind::kString);
  EXPECT_EQ(std::string(lit_str->GetStringValue().Data(), lit_str->GetStringValue().Length()), "txt");
  EXPECT_EQ(lit_null->GetKind(), LiteralExpr::Kind::kNullptr);

  auto *assign_expr = AssignExpr::Create(ctx, ident_x, lit_int);
  auto *binary_add = BinaryExpr::Create(ctx, BinaryExpr::Op::kAdd, ident_x, ident_y);
  auto *binary_eq = BinaryExpr::Create(ctx, BinaryExpr::Op::kEq, ident_x, lit_int);
  auto *unary_not = UnaryExpr::Create(ctx, UnaryExpr::Op::kLogicalNot, binary_eq);
  auto *call_expr = CallExpr::Create(ctx, ident_consume, {ident_x, lit_int});
  auto *addr_expr = AddrOfExpr::Create(ctx, ident_x);
  auto *subscript_expr = SubscriptExpr::Create(ctx, ident_vec, lit_int);
  auto *member_expr = MemberExpr::Create(ctx, ident_obj, "field");
  auto *arrow_expr = CppArrowMemberExpr::Create(ctx, ident_ptr, "field");
  auto *reinterpret_cast_expr = CppCastExpr::Create(ctx, CppCastExpr::Kind::kReinterpret, "void *", ident_ptr);
  auto *static_cast_expr = CppCastExpr::Create(ctx, CppCastExpr::Kind::kStatic, "uint32_t", ident_x);
  auto *init_list_expr = InitListExpr::Create(ctx, {lit_int, lit_bool, lit_null});
  auto *lambda_body = BlockStmt::Create(ctx, {ReturnStmt::Create(ctx, ident_x)});
  auto *lambda_expr = LambdaExpr::Create(ctx, {"x", "&y"}, lambda_body);
  ASSERT_NE(assign_expr, nullptr);
  ASSERT_EQ(assign_expr->GetLhs(), ident_x);
  ASSERT_EQ(assign_expr->GetRhs(), lit_int);
  ASSERT_EQ(binary_add->GetOp(), BinaryExpr::Op::kAdd);
  ASSERT_EQ(binary_add->GetLhs(), ident_x);
  ASSERT_EQ(binary_add->GetRhs(), ident_y);
  ASSERT_EQ(unary_not->GetOp(), UnaryExpr::Op::kLogicalNot);
  ASSERT_EQ(call_expr->GetCallee(), ident_consume);
  ASSERT_EQ(call_expr->GetArgs().Size(), 2U);
  ASSERT_EQ(addr_expr->GetExpr(), ident_x);
  ASSERT_EQ(subscript_expr->GetBase(), ident_vec);
  ASSERT_EQ(subscript_expr->GetIndex(), lit_int);
  ASSERT_EQ(member_expr->GetObject(), ident_obj);
  ASSERT_EQ(arrow_expr->GetObject(), ident_ptr);
  ASSERT_EQ(reinterpret_cast_expr->GetKind(), CppCastExpr::Kind::kReinterpret);
  ASSERT_EQ(static_cast_expr->GetKind(), CppCastExpr::Kind::kStatic);
  ASSERT_EQ(init_list_expr->GetElements().Size(), 3U);
  ASSERT_EQ(lambda_expr->GetCaptures().Size(), 2U);
  ASSERT_EQ(lambda_expr->GetBody(), lambda_body);

  auto *comment_stmt = CommentStmt::Create(ctx, "comment");
  auto *blank_stmt = BlankLineStmt::Create(ctx);
  auto *var_decl_stmt = VarDeclStmt::Create(ctx, "int", "sum", binary_add);
  auto *expr_stmt = ExprStmt::Create(ctx, assign_expr);
  auto *return_stmt = ReturnStmt::Create(ctx, init_list_expr);
  auto *return_void_stmt = ReturnStmt::Create(ctx);
  auto *then_block = BlockStmt::Create(ctx, {comment_stmt, blank_stmt, var_decl_stmt, expr_stmt});
  auto *else_block = BlockStmt::Create(ctx, {ExprStmt::Create(ctx, call_expr), return_void_stmt});
  auto *if_stmt = IfStmt::Create(ctx, unary_not, then_block, else_block);
  auto *method_body = BlockStmt::Create(ctx, {ExprStmt::Create(ctx, member_expr), ExprStmt::Create(ctx, arrow_expr)});
  auto *func_body =
      BlockStmt::Create(ctx, {if_stmt, ExprStmt::Create(ctx, addr_expr), ExprStmt::Create(ctx, subscript_expr),
                              ExprStmt::Create(ctx, reinterpret_cast_expr), ExprStmt::Create(ctx, static_cast_expr),
                              ExprStmt::Create(ctx, lambda_expr), return_stmt});
  ASSERT_NE(comment_stmt, nullptr);
  ASSERT_EQ(std::string(comment_stmt->GetText().Data(), comment_stmt->GetText().Length()), "comment");
  ASSERT_EQ(var_decl_stmt->GetInit(), binary_add);
  ASSERT_EQ(expr_stmt->GetExpr(), assign_expr);
  ASSERT_EQ(return_stmt->GetValue(), init_list_expr);
  ASSERT_EQ(return_void_stmt->GetValue(), nullptr);
  ASSERT_EQ(then_block->GetStatements().Size(), 4U);
  ASSERT_EQ(if_stmt->GetCond(), unary_not);
  ASSERT_EQ(if_stmt->GetThenBlock(), then_block);
  ASSERT_EQ(if_stmt->GetElseBlock(), else_block);

  auto *field_decl = FieldDecl::Create(ctx, "int", "value", lit_int);
  auto *type_alias = TypeAliasDecl::Create(ctx, "void *", "Handle");
  auto *method_decl = MethodDecl::Create(ctx, "Run", {param_x}, "void");
  auto *function_decl = FunctionDecl::Create(ctx, "Add", {param_x, param_y}, "int");
  auto *function_def = FunctionDef::Create(ctx, "Build", {param_x, param_y}, "int", func_body);
  auto *method_def = MethodDef::Create(ctx, "Worker", "Exec", {param_x}, "void", {}, {}, method_body);
  auto *access_decl = AccessSectionDecl::Create(ctx, AccessSectionDecl::Kind::kPublic);
  auto *private_decl = AccessSectionDecl::Create(ctx, AccessSectionDecl::Kind::kPrivate);
  auto *class_decl = ClassDecl::Create(
      ctx, "Worker", {access_decl, field_decl, method_decl, private_decl, FieldDecl::Create(ctx, "int", "hidden")});
  auto *struct_decl = StructDecl::Create(ctx, "Pod", {FieldDecl::Create(ctx, "bool", "ready", lit_bool)});
  auto *extern_decl = ExternBlockDecl::Create(ctx, "C", {function_decl});
  auto *namespace_decl =
      NamespaceDecl::Create(ctx, "om2", {class_decl, struct_decl, function_def, method_def, extern_decl});
  auto *stable_part = StablePartDecl::Create(ctx, StablePartId::kChkStatusMacro, StablePartRole::kMacroGroup,
                                             StablePartPlacement::kTranslationUnit);
  auto *include_decl = IncludeDecl::Create(ctx, "vector", IncludeDecl::Kind::kAngle);
  auto *space_decl = SpaceDecl::Create(ctx);
  auto *tu = TranslationUnit::Create(ctx, {include_decl, space_decl, stable_part, type_alias, namespace_decl});
  ASSERT_NE(field_decl, nullptr);
  ASSERT_NE(type_alias, nullptr);
  ASSERT_EQ(field_decl->GetInit(), lit_int);
  EXPECT_EQ(std::string(type_alias->GetTypeSpec().Data(), type_alias->GetTypeSpec().Length()), "void *");
  EXPECT_EQ(std::string(type_alias->GetName().Data(), type_alias->GetName().Length()), "Handle");
  ASSERT_EQ(method_decl->GetParams().Size(), 1U);
  ASSERT_EQ(function_decl->GetParams().Size(), 2U);
  ASSERT_EQ(function_def->GetBody(), func_body);
  ASSERT_EQ(method_def->GetBody(), method_body);
  ASSERT_EQ(class_decl->GetItems().Size(), 5U);
  ASSERT_EQ(struct_decl->GetItems().Size(), 1U);
  ASSERT_EQ(extern_decl->GetItems().Size(), 1U);
  ASSERT_EQ(namespace_decl->GetItems().Size(), 5U);
  ASSERT_EQ(stable_part->GetId(), StablePartId::kChkStatusMacro);
  ASSERT_EQ(stable_part->GetRole(), StablePartRole::kMacroGroup);
  ASSERT_EQ(stable_part->GetPlacement(), StablePartPlacement::kTranslationUnit);
  ASSERT_EQ(include_decl->GetKind(), IncludeDecl::Kind::kAngle);
  ASSERT_NE(space_decl, nullptr);
  ASSERT_EQ(tu->GetItems().Size(), 5U);

  const auto output = EmitNode(*tu);
  ExpectContainsAll(output, {
                                "#include <vector>\n\n",
                                "#define OM2_CHK_STATUS",
                                "typedef void *Handle;\n",
                                "namespace om2 {\n",
                                "class Worker {\n",
                                "  public:\n",
                                "    int value = 7U;\n",
                                "    void Run(int x);\n",
                                "  private:\n",
                                "    int hidden;\n",
                                "struct Pod {\n",
                                "bool ready = true;\n",
                                "int Build(int x, int y) {\n",
                                "if (!((x == 7U)))",
                                "// comment\n",
                                "int sum = (x + y);\n",
                                "x = 7U;\n",
                                "Consume(x, 7U);\n",
                                "return {7U, true, nullptr};\n",
                                "void Worker::Exec(int x) {\n",
                                "obj.field;\n",
                                "ptr->field;\n",
                                "extern \"C\" {\n",
                                "int Add(int x, int y);\n",
                            });
}

TEST_F(Om2CodegenUt, AstDsl_AllPublicInterfaces_Ok) {
  AstContext ctx;
  AstBuildContext ast(ctx);

  Arg empty_arg;
  EXPECT_TRUE(empty_arg.Empty());
  EXPECT_EQ(empty_arg.Resolve(ctx), nullptr);
  Arg ident_from_cstr("symbol_from_cstr");
  Arg ident_from_string(std::string("symbol_from_string"));
  Arg int_arg(3);
  Arg bool_arg(false);
  Arg null_arg(nullptr);
  Arg string_lit = Arg::StringLiteral("string_literal");
  EXPECT_FALSE(ident_from_cstr.Empty());

  auto lhs = ast.Var("uint32_t", "lhs");
  auto rhs = ast.Var("uint32_t", "rhs");
  auto ptr = ast.Var("Node *", "ptr");
  auto obj = ast.Var("Node", "obj");
  auto arr = ast.Var("std::vector<uint32_t>", "arr");
  auto runner = ast.Var("Runner", "runner");

  EXPECT_EQ(lhs.TypeName(), "uint32_t");
  EXPECT_EQ(lhs.SymbolName(), "lhs");
  ASSERT_NE(lhs.Get(), nullptr);
  ASSERT_NE(lhs.Ref().Get(), nullptr);
  ASSERT_NE(lhs.Addr().Get(), nullptr);
  ASSERT_NE(lhs.Attr("size").Get(), nullptr);
  ASSERT_NE(ptr.Arrow("field").Get(), nullptr);
  ASSERT_NE(arr[1].Get(), nullptr);
  ASSERT_NE(runner().Get(), nullptr);
  ASSERT_NE(runner({lhs}).Get(), nullptr);
  ASSERT_NE(runner(lhs, rhs).Get(), nullptr);

  static_assert(std::is_same<decltype(obj.Ref().Attr("field")), ExprRef>::value,
                "Attr should directly materialize a member expression");
  static_assert(std::is_same<decltype(ptr.Ref().Arrow("field")), ExprRef>::value,
                "Arrow should directly materialize an arrow member expression");
  auto expr_attr = obj.Ref().Attr("field");
  auto expr_arrow = ptr.Ref().Arrow("field");
  ASSERT_NE(expr_attr.Get(), nullptr);
  ASSERT_NE(expr_arrow.Get(), nullptr);
  ASSERT_NE(expr_attr.Addr().Get(), nullptr);
  ASSERT_NE(expr_attr().Get(), nullptr);
  ASSERT_NE(expr_attr({lhs}).Get(), nullptr);
  ASSERT_NE(expr_attr(lhs, rhs).Get(), nullptr);
  ASSERT_NE(obj.Ref().Addr().Get(), nullptr);
  ASSERT_NE(obj.Ref()().Get(), nullptr);
  ASSERT_NE(obj.Ref()({lhs, rhs}).Get(), nullptr);
  ASSERT_NE(obj.Ref()(lhs, rhs).Get(), nullptr);
  ASSERT_NE(obj.Ref()[2].Get(), nullptr);

  ASSERT_NE(Arg(lhs).Resolve(ctx), nullptr);
  ASSERT_NE(Arg(lhs.Ref()).Resolve(ctx), nullptr);
  ASSERT_NE(Arg(expr_attr).Resolve(ctx), nullptr);
  ASSERT_NE(Arg(lhs.Get()).Resolve(ctx), nullptr);
  ASSERT_NE(ident_from_cstr.Resolve(ctx), nullptr);
  ASSERT_NE(ident_from_string.Resolve(ctx), nullptr);
  ASSERT_NE(int_arg.Resolve(ctx), nullptr);
  ASSERT_NE(bool_arg.Resolve(ctx), nullptr);
  ASSERT_NE(null_arg.Resolve(ctx), nullptr);
  ASSERT_NE(string_lit.Resolve(ctx), nullptr);
  ASSERT_NE(Arg({lhs, rhs, nullptr}).Resolve(ctx), nullptr);
  ASSERT_NE(Arg(std::vector<Arg>{lhs, rhs, nullptr}).Resolve(ctx), nullptr);

  auto lambda_capture = ast.CaptureRef(lhs);
  EXPECT_EQ(lambda_capture.name, "lhs");
  EXPECT_EQ(lambda_capture.kind, LambdaCaptureSpec::Kind::kByRef);

  auto unary_not = !lhs.Ref();
  auto unary_neg = -lhs.Ref();
  auto unary_bit_not = ~lhs.Ref();
  auto eq_expr = lhs.Ref() == rhs;
  auto ne_expr = lhs.Ref() != 1;
  auto lt_expr = lhs.Ref() < rhs;
  auto le_expr = lhs.Ref() <= rhs;
  auto gt_expr = lhs.Ref() > rhs;
  auto ge_expr = lhs.Ref() >= rhs;
  auto land_expr = lhs.Ref() && rhs;
  auto lor_expr = lhs.Ref() || rhs;
  auto add_expr = lhs.Ref() + rhs;
  auto sub_expr = lhs.Ref() - rhs;
  auto mul_expr = lhs.Ref() * rhs;
  auto div_expr = lhs.Ref() / rhs;
  auto mod_expr = lhs.Ref() % rhs;
  auto band_expr = lhs.Ref() & rhs;
  auto bor_expr = lhs.Ref() | rhs;
  auto bxor_expr = lhs.Ref() ^ rhs;
  auto shl_expr = lhs.Ref() << 2;
  auto shr_expr = lhs.Ref() >> 1;
  auto assign_expr = ast.Assign(lhs, rhs);
  auto lambda_expr =
      ast.Lambda({LambdaCaptureSpec{"rhs", LambdaCaptureSpec::Kind::kByValue}, lambda_capture}, {ast.Return(lhs)});
  auto call_expr = ast.Call("Compute", {lhs, rhs, string_lit});
  auto reinterpret_expr = ast.ReinterpretCast("void *", ptr);

  std::vector<Stmt *> body_vec = {
      ast.VarDecl("auto", "name", string_lit),
      ast.VarDecl(lhs, 1),
      ast.Return(),
  };
  auto resolved_body = ast.Body(std::vector<BodyItem>{BodyItem(ast.Call("Touch", {lhs})), BodyItem(ast.Return(lhs))});
  ASSERT_EQ(resolved_body.size(), 2U);

  auto *decl_fn = ast.DeclareFunction("DeclVec", std::vector<VarRef>{lhs, rhs}, "uint32_t");
  auto *decl_fn_init = ast.DeclareFunction("DeclInit", {lhs}, "void");
  auto *decl_method = ast.DeclareMethod("MethodVec", std::vector<VarRef>{lhs}, "void");
  auto *decl_method_init = ast.DeclareMethod("MethodInit", {lhs, rhs}, "void");
  auto *def_fn_vec = ast.DefineFunction("DefVec", std::vector<VarRef>{lhs, rhs}, "uint32_t", body_vec);
  auto *def_fn_init = ast.DefineFunction("DefInit", {lhs}, "uint32_t",
                                         {
                                             ast.VarDecl("uint32_t", "tmp", add_expr),
                                             ast.Return(lhs),
                                         });
  auto *def_method_vec = ast.DefineMethod("Worker", "ExecVec", std::vector<VarRef>{lhs}, "void",
                                          std::vector<Stmt *>{ExprStmt::Create(ctx, call_expr.Get())});
  auto *def_method_init = ast.DefineMethod("Worker", "ExecInit", {lhs}, "void",
                                           {
                                               ast.Return(),
                                           });
  auto *all_ops_def =
      ast.DefineFunction("AllOps", {lhs, rhs, ptr, obj, arr, runner}, "uint32_t",
                         {
                             CommentStmt::Create(ctx, "dsl-all-ops"),
                             BlankLineStmt::Create(ctx),
                             ast.VarDecl("auto", "from_cstr", ident_from_cstr),
                             ast.VarDecl("auto", "from_string", ident_from_string),
                             ast.VarDecl("auto", "str_node", ast.Str("dsl")),
                             ast.VarDecl("auto", "uint_node", ast.UInt(9)),
                             ast.VarDecl("auto", "ulong_node", ast.ULong(9)),
                             ast.VarDecl("auto", "eq_v", eq_expr),
                             ast.VarDecl("auto", "ne_v", ne_expr),
                             ast.VarDecl("auto", "lt_v", lt_expr),
                             ast.VarDecl("auto", "le_v", le_expr),
                             ast.VarDecl("auto", "gt_v", gt_expr),
                             ast.VarDecl("auto", "ge_v", ge_expr),
                             ast.VarDecl("auto", "land_v", land_expr),
                             ast.VarDecl("auto", "lor_v", lor_expr),
                             ast.VarDecl("auto", "add_v", add_expr),
                             ast.VarDecl("auto", "sub_v", sub_expr),
                             ast.VarDecl("auto", "mul_v", mul_expr),
                             ast.VarDecl("auto", "div_v", div_expr),
                             ast.VarDecl("auto", "mod_v", mod_expr),
                             ast.VarDecl("auto", "band_v", band_expr),
                             ast.VarDecl("auto", "bor_v", bor_expr),
                             ast.VarDecl("auto", "bxor_v", bxor_expr),
                             ast.VarDecl("auto", "shl_v", shl_expr),
                             ast.VarDecl("auto", "shr_v", shr_expr),
                             ast.VarDecl("auto", "not_v", unary_not),
                             ast.VarDecl("auto", "neg_v", unary_neg),
                             ast.VarDecl("auto", "bit_not_v", unary_bit_not),
                             ast.VarDecl("auto", "index_v", arr[2]),
                             ast.VarDecl("auto", "addr_v", lhs.Addr()),
                             ast.VarDecl("auto", "attr_v", obj.Attr("field")),
                             ast.VarDecl("auto", "arrow_v", ptr.Arrow("field")),
                             ast.VarDecl("auto", "call0_v", runner()),
                             ast.VarDecl("auto", "call1_v", runner(lhs)),
                             ast.VarDecl("auto", "member_call_v", obj.Attr("Exec")(lhs, rhs)),
                             ast.VarDecl("auto", "lambda_v", lambda_expr),
                             ast.VarDecl("auto", "reinterpret_v", reinterpret_expr),
                             ast.VarDecl("auto", "init_v", ast.InitList(std::vector<Arg>{lhs, rhs, null_arg})),
                             ast.Assign(obj.Attr("field"), call_expr),
                             IfStmt::Create(ctx, gt_expr.Get(), BlockStmt::Create(ctx, ast.Body({ast.Return(lhs)})),
                                            BlockStmt::Create(ctx, ast.Body({ast.Return(rhs)}))),
                             ast.Return(lhs),
                         });
  auto *dsl_class = ast.Class(
      "DslClass", std::vector<DeclNode *>{ast.Public(), ast.Field("uint32_t", "field", ast.UInt(4)), decl_method,
                                          decl_method_init, ast.Private(), ast.Field("uint32_t", "hidden")});
  auto *dsl_struct = ast.Struct("DslStruct", {ast.Field("const char *", "name", ast.Str("abc"))});
  std::vector<DeclNode *> namespace_items{ast.TypeAlias("void *", "DslHandle"),
                                          dsl_class,
                                          dsl_struct,
                                          decl_fn_init,
                                          def_fn_vec,
                                          def_fn_init,
                                          def_method_vec,
                                          def_method_init,
                                          all_ops_def};
  auto *dsl_namespace = ast.Namespace("dsl_ns", namespace_items);
  std::vector<DeclNode *> tu_items{
      ast.Include("vector", IncludeDecl::Kind::kAngle), ast.Include("local.h"),          ast.Space(),
      ast.StablePart(StablePartId::kScopeGuard),        ast.ExternBlock("C", {decl_fn}), dsl_namespace};
  auto *tu = ast.File(tu_items);
  ASSERT_NE(tu, nullptr);

  EXPECT_EQ(decl_fn->GetParams().Size(), 2U);
  EXPECT_EQ(decl_fn_init->GetParams().Size(), 1U);
  EXPECT_EQ(decl_method->GetParams().Size(), 1U);
  EXPECT_EQ(def_fn_vec->GetBody()->GetStatements().Size(), 3U);
  EXPECT_EQ(def_fn_init->GetBody()->GetStatements().Size(), 2U);
  EXPECT_EQ(def_method_vec->GetBody()->GetStatements().Size(), 1U);

  const auto output = EmitNode(*tu);
  ExpectContainsAll(output, {
                                "#include <vector>\n#include \"local.h\"\n\n",
                                "class ScopeGuard {\n",
                                "extern \"C\" {\n",
                                "uint32_t DeclVec(uint32_t lhs, uint32_t rhs);\n",
                                "namespace dsl_ns {\n",
                                "typedef void *DslHandle;\n",
                                "class DslClass {\n",
                                "  public:\n",
                                "    uint32_t field = 4U;\n",
                                "    void MethodVec(uint32_t lhs);\n",
                                "    void MethodInit(uint32_t lhs, uint32_t rhs);\n",
                                "  private:\n",
                                "    uint32_t hidden;\n",
                                "struct DslStruct {\n",
                                "const char *name = \"abc\";\n",
                                "void DeclInit(uint32_t lhs);\n",
                                "uint32_t DefVec(uint32_t lhs, uint32_t rhs) {\n",
                                "return;\n",
                                "uint32_t DefInit(uint32_t lhs) {\n",
                                "void Worker::ExecVec(uint32_t lhs) {\n",
                                "void Worker::ExecInit(uint32_t lhs) {\n",
                                "// dsl-all-ops\n",
                                "auto from_cstr = symbol_from_cstr;\n",
                                "auto from_string = symbol_from_string;\n",
                                "auto str_node = \"dsl\";\n",
                                "auto uint_node = 9U;\n",
                                "auto ulong_node = 9UL;\n",
                                "auto eq_v = (lhs == rhs);\n",
                                "auto ne_v = (lhs != 1);\n",
                                "auto lt_v = (lhs < rhs);\n",
                                "auto le_v = (lhs <= rhs);\n",
                                "auto gt_v = (lhs > rhs);\n",
                                "auto ge_v = (lhs >= rhs);\n",
                                "auto land_v = (lhs && rhs);\n",
                                "auto lor_v = (lhs || rhs);\n",
                                "auto add_v = (lhs + rhs);\n",
                                "auto sub_v = (lhs - rhs);\n",
                                "auto mul_v = (lhs * rhs);\n",
                                "auto div_v = (lhs / rhs);\n",
                                "auto mod_v = (lhs % rhs);\n",
                                "auto band_v = (lhs & rhs);\n",
                                "auto bor_v = (lhs | rhs);\n",
                                "auto bxor_v = (lhs ^ rhs);\n",
                                "auto shl_v = (lhs << 2);\n",
                                "auto shr_v = (lhs >> 1);\n",
                                "auto not_v = !lhs;\n",
                                "auto neg_v = -lhs;\n",
                                "auto bit_not_v = ~lhs;\n",
                                "auto index_v = arr[2];\n",
                                "auto addr_v = &lhs;\n",
                                "auto attr_v = obj.field;\n",
                                "auto arrow_v = ptr->field;\n",
                                "auto call0_v = runner();\n",
                                "auto call1_v = runner(lhs);\n",
                                "auto member_call_v = obj.Exec(lhs, rhs);\n",
                                "auto lambda_v = [rhs, &lhs]() {\n",
                                "auto reinterpret_v = reinterpret_cast<void *>(ptr);\n",
                                "auto init_v = {lhs, rhs, nullptr};\n",
                                "obj.field = Compute(lhs, rhs, \"string_literal\");\n",
                                "if ((lhs > rhs)) {\n",
                                "return lhs;\n",
                                "return rhs;\n",
                            });
}

TEST_F(Om2CodegenUt, AstDsl_ArgSupportsAllIntegralTypes_Ok) {
  AstContext ctx;
  AstBuildContext ast(ctx);

  const uint32_t u32_value = 7U;
  const size_t size_value = 9U;
  const uint8_t u8_value = 3U;

  ASSERT_NE(Arg(u32_value).Resolve(ctx), nullptr);
  ASSERT_NE(Arg(size_value).Resolve(ctx), nullptr);
  ASSERT_NE(Arg(u8_value).Resolve(ctx), nullptr);

  auto *fn = ast.DefineFunction("IntegralArgs", {}, "void",
                                {
                                    ast.VarDecl("auto", "u32_v", u32_value),
                                    ast.VarDecl("auto", "size_v", size_value),
                                    ast.VarDecl("auto", "u8_v", u8_value),
                                    ast.Return(),
                                });
  ASSERT_NE(fn, nullptr);

  const auto output = EmitNode(*fn);
  ExpectContainsAll(output, {
                                "auto u32_v = 7;\n",
                                "auto size_v = 9;\n",
                                "auto u8_v = 3;\n",
                            });
}

TEST_F(Om2CodegenUt, AstDsl_MakeUniqueArray_Ok) {
  AstContext ctx;
  AstBuildContext ast(ctx);

  auto count = ast.Var("size_t", "count");
  auto *fn =
      ast.DefineFunction("BuildArrays", {count}, "void",
                         {
                             ast.VarDecl("auto", "builtin_buffer", ast.MakeUniqueArray(BuiltinType::kUInt8, count)),
                             ast.VarDecl("auto", "custom_buffer", ast.MakeUniqueArray("CustomType", 4)),
                             ast.Return(),
                         });
  ASSERT_NE(fn, nullptr);

  const auto output = EmitNode(*fn);
  ExpectContainsAll(output, {
                                "auto builtin_buffer = std::make_unique<uint8_t[]>(count);\n",
                                "auto custom_buffer = std::make_unique<CustomType[]>(4);\n",
                            });
}

TEST_F(Om2CodegenUt, AstDsl_ControlFlowAndCtorInit_Ok) {
  AstContext ctx;
  AstBuildContext ast(ctx);

  auto value = ast.Var("int", "value");
  auto values = ast.Var("std::vector<int>", "values");
  auto i = ast.Var("size_t", "i");
  auto lhs = ast.Var("int", "lhs");
  auto item = ast.Var("auto", "item");

  auto *for_stmt = ast.For(ast.VarDecl(i, 0), i < 4, ast.PreInc(i),
                           {
                               ast.Assign(value, value + 1),
                           });
  auto *range_for_stmt = ast.RangeFor(item, values,
                                      {
                                          ast.Assign(value, value + item),
                                      });
  auto *ctor_def = ast.DefineMethod("Worker", "Worker", {lhs}, "", {ast.MemberInit("value_", lhs)},
                                    {
                                        ast.Assign(value, lhs),
                                    });
  auto *tu = ast.File({
      ast.Namespace("om2",
                    {
                        ctor_def,
                        ast.DefineFunction("Touch", {value, values}, "void",
                                           {
                                               for_stmt,
                                               range_for_stmt,
                                               ast.Return(),
                                           }),
                    }),
  });

  ASSERT_NE(tu, nullptr);
  const auto output = EmitNode(*tu);
  ExpectContainsAll(output, {
                                "Worker::Worker(int lhs)\n",
                                "  : value_(lhs) {\n",
                                "for (size_t i = 0; (i < 4); ++i) {\n",
                                "for (auto item : values) {\n",
                            });
}

TEST_F(Om2CodegenUt, AstDsl_MemcpyAndSizeof_Ok) {
  AstContext ctx;
  AstBuildContext ast(ctx);

  auto dst = ast.Var("void *", "dst");
  auto src = ast.Var("const void *", "src");
  auto addr = ast.Var("uintptr_t", "addr");
  auto *tu = ast.File({
      ast.Namespace("om2",
                    {
                        ast.DefineFunction("TouchMemcpy", {dst, src, addr}, "void",
                                           {
                                               BodyItem(ast.Memcpy(dst, src, ast.Sizeof(addr))),
                                               ast.Return(),
                                           }),
                    }),
  });

  ASSERT_NE(tu, nullptr);
  const auto output = EmitNode(*tu);
  ExpectContainsAll(output, {
                                "void TouchMemcpy(void *dst, const void *src, uintptr_t addr) {\n",
                                "std::memcpy(dst, src, sizeof(addr));\n",
                            });
}

TEST_F(Om2CodegenUt, AstDsl_IgnoreOutputRemoveFile_Ok) {
  AstContext ctx;
  AstBuildContext ast(ctx);

  auto json_path = ast.Var("std::string", "json_path");
  auto *fn = ast.DefineFunction("CleanupJsonFile", {json_path}, "void",
                                {
                                    ast.IgnoreOutput(ast.RemoveFile(json_path.CStr())),
                                    ast.Return(),
                                });
  ASSERT_NE(fn, nullptr);

  const auto output = EmitNode(*fn);
  ExpectContainsAll(output, {
                                "void CleanupJsonFile(std::string json_path) {\n",
                                "(void)std::remove(json_path.c_str());\n",
                            });
}

TEST_F(Om2CodegenUt, InterfaceDumpApis_EmitInCLinkageAndPtrToU64Outside_Ok) {
  AstContext ctx;
  AstBuildContext ast(ctx);

  auto *tu = ast.File({
      ast.StablePart(StablePartId::kInterfacePointerHelpers),
      ast.ExternBlock("C", {ast.StablePart(StablePartId::kInterfaceDumpApis)}),
  });
  ASSERT_NE(tu, nullptr);

  const auto output = EmitNode(*tu);
  ExpectContainsAll(output, {
                                "inline void *ValueToPtr(const uint64_t value) {\n",
                                "inline uint64_t PtrToU64(const void *ptr) {\n",
                                "extern \"C\" {\n",
                                "enum GertModelArgKind : uint64_t {\n",
                                "struct GertModelArgSlotInfo {\n",
                                "struct GertModelTaskRawInfo {\n",
                                "const struct GertModelTaskRawInfo *task_raw_info = nullptr;\n",
                                "  uint64_t kernel_type = 10000U;\n",
                                "enum GertModelTaskLaunchType : uint64_t",
                                "struct GertModelLaunchKernelV2Params {\n",
                                "  uint64_t struct_size = sizeof(GertModelLaunchKernelV2Params);\n",
                                "  aclrtFuncHandle func_handle = nullptr;\n",
                                "  uint32_t block_dim = 0;\n",
                                "  uint32_t reserved_1 = 0;\n",
                                "  const void *args_data = nullptr;\n",
                                "  size_t args_size = 0;\n",
                                "  aclrtLaunchKernelCfg *config = nullptr;\n",
                                "  aclrtStream stream = nullptr;\n",
                                "struct GertModelLaunchStarsTaskWithFlagParams {\n",
                                "  uint64_t struct_size = sizeof(GertModelLaunchStarsTaskWithFlagParams);\n",
                                "  const void *task_sqe = nullptr;\n",
                                "  uint32_t sqe_len = 0;\n",
                                "  uint32_t reserved_1 = 0;\n",
                                "  aclrtStream stream = nullptr;\n",
                                "  uint32_t flag = 0;\n",
                                "  uint32_t reserved_2 = 0;\n",
                                "struct GertModelTaskLaunchInfo {\n",
                                "  uint64_t struct_size = sizeof(GertModelTaskLaunchInfo);\n",
                                "  GertModelTaskLaunchType launch_type = ACL_RT_LAUNCH_KERNEL_V2;\n",
                                "GertModelLaunchFunc launch_func = nullptr;",
                            });
  EXPECT_LT(output.find("  uint64_t task_type = 0;\n"), output.find("  uint64_t kernel_type = 10000U;\n"));
  EXPECT_EQ(output.find("report_task_preprocess"), std::string::npos);
  EXPECT_EQ(output.find("report_task_postprocess"), std::string::npos);
  EXPECT_EQ(output.find("get_data_dump_enabled"), std::string::npos);
  EXPECT_LT(output.find("inline uint64_t PtrToU64"), output.find("extern \"C\" {"));
  EXPECT_GT(output.find("struct Om2Tensor"), output.find("extern \"C\" {"));
}

TEST_F(Om2CodegenUt, LoadAndRunDumpHelpers_EmitInAnonymousNamespace_Ok) {
  AstContext ctx;
  AstBuildContext ast(ctx);

  auto *tu = ast.File({
      ast.Namespace("om2",
                    {
                        ast.Namespace("", {ast.StablePart(StablePartId::kLoadAndRunDumpHelpers)}),
                    }),
  });
  ASSERT_NE(tu, nullptr);

  const auto output = EmitNode(*tu);
  ExpectContainsAll(
      output, {
                  "namespace om2 {\n",
                  "namespace {\n",
                  "gert::Tensor BuildTensor(void *device_address, uint64_t size, int32_t data_type, int32_t format,\n",
                  "const int64_t *shape_dims, uint64_t shape_dims_num) {\n",
              });
  EXPECT_EQ(output.find("GetIsDataDump("), std::string::npos);
}

TEST_F(Om2CodegenUt, BuildL0ArgSlotEntries_EmitsTensorWorkspaceAndIgnoredKinds) {
  AstContext ctx;
  AstBuildContext ast(ctx);

  AddrSemantic input;
  input.kind = AddrValueKind::kInputInstance;
  input.byte_size = 32U;
  input.tensor_info = Om2TensorInfo{};
  input.tensor_info->args_offset = 0U;
  input.tensor_info->size = 32U;

  AddrSemantic workspace;
  workspace.kind = AddrValueKind::kWorkspace;
  workspace.byte_size = 128U;

  AddrSemantic placeholder;
  placeholder.kind = AddrValueKind::kPlaceholder;

  auto *entries = TaskCodeBuilderUtil::BuildL0ArgSlotEntries(ast, {input, workspace, placeholder});
  ASSERT_NE(entries, nullptr);
  auto output = EmitNode(*entries);
  ExpectContainsAll(output, {
                                "{sizeof(GertModelArgSlotInfo), GERT_MODEL_ARG_INPUT, 0U, 0U, 0UL, 0U, 0U, 0U}",
                                "{sizeof(GertModelArgSlotInfo), GERT_MODEL_ARG_WORKSPACE, 0U, 8U, 0UL, 0U, 0U, 0U}",
                                "{sizeof(GertModelArgSlotInfo), GERT_MODEL_ARG_PLACEHOLDER, 0U, 16U, 0UL, 0U, 0U, 0U}",
                            });
}

TEST_F(Om2CodegenUt, AstDsl_ContainerMethods_Ok) {
  AstContext ctx;
  AstBuildContext ast(ctx);

  auto vec = ast.Var("std::vector<uint8_t>", "vec");
  auto ptr = ast.Var("std::unique_ptr<uint8_t[]>", "ptr");
  auto index = ast.Var("size_t", "index");
  auto value = ast.Var("uint8_t", "value");
  auto *fn = ast.DefineFunction("TouchContainer", {vec, ptr, index, value}, "void",
                                {
                                    vec.Clear(),
                                    vec.Resize(8),
                                    vec.PushBack(value),
                                    ast.Call("Use", {vec.Size(), vec.Data(), vec.Empty(), vec.At(index), ptr.GetPtr()}),
                                    ast.Return(),
                                });
  ASSERT_NE(fn, nullptr);

  const auto output = EmitNode(*fn);
  ExpectContainsAll(output, {
                                "void TouchContainer(std::vector<uint8_t> vec, std::unique_ptr<uint8_t[]> ptr, size_t "
                                "index, uint8_t value) {\n",
                                "vec.clear();\n",
                                "vec.resize(8);\n",
                                "vec.push_back(value);\n",
                                "Use(vec.size(), vec.data(), vec.empty(), vec.at(index), ptr.get());\n",
                            });
}

TEST_F(Om2CodegenUt, Arg_AutoPromoteInitList_Ok) {
  AstContext ctx;
  AstBuildContext ast(ctx);

  auto value = ast.Var("std::vector<ArgsInfo>", "value");
  auto *fn = ast.DefineFunction("Build", {}, "void",
                                {
                                    ast.Assign(value, {{1, 2, 3}, {4, 5, 6}}),
                                });

  ASSERT_NE(fn, nullptr);
  const auto output = EmitNode(*fn);
  ExpectContainsAll(output, {
                                "void Build() {\n",
                                "value = {{1, 2, 3}, {4, 5, 6}};\n",
                            });
}

TEST_F(Om2CodegenUt, CompileGeneratedCppToSo_MakefileVariableContinuation_Ok) {
  ScopedEnvVar asan_guard("ASAN_OPTIONS", "detect_leaks=0:halt_on_error=0");
  ScopedEnvVar lsan_guard("LSAN_OPTIONS", "exitcode=0");
  const std::string model_name = "continuation_test";
  const std::string interface_name = model_name + "_internal.h";
  const std::string include_line = "#include \"" + interface_name + "\"\n";
  Om2CodegenArtifacts artifacts = {
      {"om2_model_api.h", "#pragma once\n"},
      {interface_name, "#pragma once\n#include \"om2_model_api.h\"\n#define CONTINUATION_TEST_VALUE 7\n"},
      {model_name + "_resources.cpp",
       include_line + "extern \"C\" int ContinuationTestResources() { return CONTINUATION_TEST_VALUE; }\n"},
      {model_name + "_kernel_reg.cpp",
       include_line + "extern \"C\" int ContinuationTestKernelReg() { return CONTINUATION_TEST_VALUE; }\n"},
      {model_name + "_load_and_run.cpp",
       include_line + "extern \"C\" int ContinuationTestLoadAndRun() { return CONTINUATION_TEST_VALUE; }\n"},
      {model_name + "_args_manager.cpp",
       include_line + "extern \"C\" int ContinuationTestArgsManager() { return CONTINUATION_TEST_VALUE; }\n"},
      {"Makefile", R"(CXX := g++
TARGET := ../libcontinuation_test_om2.so
SRC_FILES := continuation_test_resources.cpp \
  \
  continuation_test_kernel_reg.cpp \
  continuation_test_load_and_run.cpp \
  continuation_test_args_manager.cpp

CXXFLAGS := -std=c++17 -fPIC
LDFLAGS := -shared

all: $(TARGET)

$(TARGET): $(SRC_FILES)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)
)"},
  };

  Om2CodegenArtifact so_artifact;
  ScopedStdoutCapture stdout_capture;
  ASSERT_EQ(Om2Utils::CompileGeneratedCppToSo(artifacts, model_name, so_artifact, false), SUCCESS);
  const std::string compile_stdout = stdout_capture.Stop();
  EXPECT_EQ(so_artifact.file_name, "lib" + model_name + "_om2.so");
  EXPECT_FALSE(so_artifact.data.empty());
  EXPECT_EQ(compile_stdout.find("g++ -std=c++17 -fPIC"), std::string::npos);
}

// build_config 校验 UT：通过 SetGraphOption 注入 ge.buildConfig，走 CompileGeneratedCppToSo 触发校验
static Om2CodegenArtifacts MakeBuildConfigTestArtifacts(const std::string &model_name) {
  const std::string interface_name = model_name + "_internal.h";
  const std::string include_line = "#include \"" + interface_name + "\"\n";
  return {
      {"om2_model_api.h", "#pragma once\n"},
      {interface_name, "#pragma once\n#include \"om2_model_api.h\"\n#define BC_TEST_VALUE 1\n"},
      {model_name + "_load_and_run.cpp", include_line + "extern \"C\" int BcTest() { return BC_TEST_VALUE; }\n"},
      {"Makefile", R"(CXX := c++
TARGET := ../libbc_test_om2.so
SRC_FILES := bc_test_load_and_run.cpp

CXXFLAGS := -std=c++17 -fPIC
LDFLAGS := -shared

all: $(TARGET)

$(TARGET): $(SRC_FILES)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)
)"},
  };
}

TEST_F(Om2CodegenUt, CompileGeneratedCppToSo_MissingModelApiArtifact_Failed) {
  const std::string model_name = "missing_model_api";
  auto artifacts = MakeBuildConfigTestArtifacts(model_name);
  artifacts.erase(artifacts.begin());
  Om2CodegenArtifact so_artifact;
  EXPECT_NE(Om2Utils::CompileGeneratedCppToSo(artifacts, model_name, so_artifact, false), SUCCESS);
}

std::string GetNativeMachine() {
  struct utsname uts;
  if (uname(&uts) != 0) {
    return "";
  }
  return uts.machine;
}

Status CompileBuildConfigArtifacts(const std::string &model_name) {
  Om2CodegenArtifact so_artifact;
  const Status ret =
      Om2Utils::CompileGeneratedCppToSo(MakeBuildConfigTestArtifacts(model_name), model_name, so_artifact, false);
  if (ret == SUCCESS) {
    EXPECT_FALSE(so_artifact.data.empty());
  }
  return ret;
}

std::string FindExecutable(const std::string &name) {
  std::ostringstream command;
  command << "command -v " << name;
  FILE *pipe = popen(command.str().c_str(), "r");
  if (pipe == nullptr) {
    return "";
  }
  std::string path;
  char buffer[1024];
  if (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
    path = buffer;
    while (!path.empty() && ((path.back() == '\n') || (path.back() == '\r'))) {
      path.pop_back();
    }
  }
  (void)pclose(pipe);
  return path;
}

std::string ShellQuote(const std::string &value) {
  std::string quoted = "'";
  for (const auto c : value) {
    if (c == '\'') {
      quoted += "'\\''";
      continue;
    }
    quoted.push_back(c);
  }
  quoted.push_back('\'');
  return quoted;
}

bool PrepareCommandSymlink(ScopedTempDir &temp_dir, const std::string &relative_path, const std::string &command) {
  const std::string path = FindExecutable(command);
  return (!path.empty()) && temp_dir.Symlink(relative_path, path);
}

bool PrepareMakeRuntime(ScopedTempDir &temp_dir, const std::string &bin_dir) {
  return PrepareCommandSymlink(temp_dir, bin_dir + "/env", "env") &&
         PrepareCommandSymlink(temp_dir, bin_dir + "/make", "make");
}

bool PrepareFakeCompiler(ScopedTempDir &temp_dir, const std::string &relative_path) {
  std::string cxx = FindExecutable("c++");
  if (cxx.empty()) {
    cxx = FindExecutable("g++");
  }
  const char *path_env = getenv("PATH");
  if ((cxx.empty()) || (path_env == nullptr)) {
    return false;
  }
  const std::string script =
      "#!/bin/sh\nexport PATH=" + ShellQuote(path_env) + "\nexec " + ShellQuote(cxx) + " \"$@\"\n";
  return temp_dir.WriteFile(relative_path, script, 0755);
}

TEST_F(Om2CodegenUt, CompileGeneratedCppToSo_BuildConfigInvalidChar_Rejected) {
  ScopedEnvVar asan_guard("ASAN_OPTIONS", "detect_leaks=0:halt_on_error=0");
  ScopedEnvVar lsan_guard("LSAN_OPTIONS", "exitcode=0");
  ScopedGraphOptions graph_guard;
  GetThreadLocalContext().SetGraphOption({{"ge.buildConfig", "make -s; rm -rf /"}});
  (void)ErrorManager::GetInstance().GetErrorMessage();

  const std::string model_name = "bc_invalid_char";
  Om2CodegenArtifact so_artifact;
  EXPECT_NE(Om2Utils::CompileGeneratedCppToSo(MakeBuildConfigTestArtifacts(model_name), model_name, so_artifact, false),
            SUCCESS);
  const std::string error_message = ErrorManager::GetInstance().GetErrorMessage();
  EXPECT_NE(error_message.find("E10001"), std::string::npos);
  EXPECT_NE(error_message.find("build_config contains an unsupported character."), std::string::npos);
}

TEST_F(Om2CodegenUt, CompileGeneratedCppToSo_BuildConfigNonWhitelisted_Rejected) {
  ScopedEnvVar asan_guard("ASAN_OPTIONS", "detect_leaks=0:halt_on_error=0");
  ScopedEnvVar lsan_guard("LSAN_OPTIONS", "exitcode=0");
  ScopedGraphOptions graph_guard;
  GetThreadLocalContext().SetGraphOption({{"ge.buildConfig", "make -s SHELL=/bin/bash"}});

  const std::string model_name = "bc_invalid_var";
  Om2CodegenArtifact so_artifact;
  EXPECT_NE(Om2Utils::CompileGeneratedCppToSo(MakeBuildConfigTestArtifacts(model_name), model_name, so_artifact, false),
            SUCCESS);
}

TEST_F(Om2CodegenUt, CompileGeneratedCppToSo_BuildConfigUnbalancedQuote_Rejected) {
  ScopedEnvVar asan_guard("ASAN_OPTIONS", "detect_leaks=0:halt_on_error=0");
  ScopedEnvVar lsan_guard("LSAN_OPTIONS", "exitcode=0");
  ScopedGraphOptions graph_guard;
  GetThreadLocalContext().SetGraphOption({{"ge.buildConfig", "make -s CXXFLAGS='-O2"}});

  const std::string model_name = "bc_invalid_quote";
  Om2CodegenArtifact so_artifact;
  EXPECT_NE(Om2Utils::CompileGeneratedCppToSo(MakeBuildConfigTestArtifacts(model_name), model_name, so_artifact, false),
            SUCCESS);
}

TEST_F(Om2CodegenUt, CompileGeneratedCppToSo_BuildConfigNotMakeCommand_Rejected) {
  ScopedEnvVar asan_guard("ASAN_OPTIONS", "detect_leaks=0:halt_on_error=0");
  ScopedEnvVar lsan_guard("LSAN_OPTIONS", "exitcode=0");
  ScopedGraphOptions graph_guard;
  GetThreadLocalContext().SetGraphOption({{"ge.buildConfig", "gcc -o out main.c"}});

  const std::string model_name = "bc_not_make";
  Om2CodegenArtifact so_artifact;
  EXPECT_NE(Om2Utils::CompileGeneratedCppToSo(MakeBuildConfigTestArtifacts(model_name), model_name, so_artifact, false),
            SUCCESS);
}

TEST_F(Om2CodegenUt, CompileGeneratedCppToSo_BuildConfigUseStubLib_Rejected) {
  ScopedEnvVar asan_guard("ASAN_OPTIONS", "detect_leaks=0:halt_on_error=0");
  ScopedEnvVar lsan_guard("LSAN_OPTIONS", "exitcode=0");
  ScopedGraphOptions graph_guard;
  GetThreadLocalContext().SetGraphOption({{"ge.buildConfig", "make -s USE_STUB_LIB=0"}});

  const std::string model_name = "bc_stub_lib";
  Om2CodegenArtifact so_artifact;
  EXPECT_NE(Om2Utils::CompileGeneratedCppToSo(MakeBuildConfigTestArtifacts(model_name), model_name, so_artifact, false),
            SUCCESS);
}

TEST_F(Om2CodegenUt, CompileGeneratedCppToSo_BuildConfigMakeOnly_Ok) {
  ScopedEnvVar asan_guard("ASAN_OPTIONS", "detect_leaks=0:halt_on_error=0");
  ScopedEnvVar lsan_guard("LSAN_OPTIONS", "exitcode=0");
  ScopedGraphOptions graph_guard;
  GetThreadLocalContext().SetGraphOption({{"ge.buildConfig", "make"}});

  EXPECT_EQ(CompileBuildConfigArtifacts("bc_make_only"), SUCCESS);
}

TEST_F(Om2CodegenUt, CompileGeneratedCppToSo_BuildConfigAbsoluteMakeOnly_Ok) {
  ScopedEnvVar asan_guard("ASAN_OPTIONS", "detect_leaks=0:halt_on_error=0");
  ScopedEnvVar lsan_guard("LSAN_OPTIONS", "exitcode=0");
  ScopedGraphOptions graph_guard;
  GetThreadLocalContext().SetGraphOption({{"ge.buildConfig", "/usr/bin/make"}});

  EXPECT_EQ(CompileBuildConfigArtifacts("bc_absolute_make_only"), SUCCESS);
}

TEST_F(Om2CodegenUt, CompileGeneratedCppToSo_BuildConfigQuotedMakePath_Ok) {
  ScopedEnvVar asan_guard("ASAN_OPTIONS", "detect_leaks=0:halt_on_error=0");
  ScopedEnvVar lsan_guard("LSAN_OPTIONS", "exitcode=0");
  ScopedGraphOptions graph_guard;
  GetThreadLocalContext().SetGraphOption(
      {{"ge.buildConfig", "  /usr/bin/make -s CXX=c++ CXXFLAGS='-std=c++17 -fPIC'"}});

  const std::string model_name = "bc_quoted_make_path";
  Om2CodegenArtifact so_artifact;
  ASSERT_EQ(Om2Utils::CompileGeneratedCppToSo(MakeBuildConfigTestArtifacts(model_name), model_name, so_artifact, false),
            SUCCESS);
  EXPECT_FALSE(so_artifact.data.empty());
}

TEST_F(Om2CodegenUt, CompileGeneratedCppToSo_BuildConfigMakefileOptionRejected) {
  ScopedEnvVar asan_guard("ASAN_OPTIONS", "detect_leaks=0:halt_on_error=0");
  ScopedEnvVar lsan_guard("LSAN_OPTIONS", "exitcode=0");
  ScopedGraphOptions graph_guard;
  const std::vector<std::string> invalid_build_configs = {
      "make -f user.mk",     "make -fuser.mk",          "make --file user.mk",
      "make --file=user.mk", "make --makefile user.mk", "make --makefile=user.mk",
  };

  for (size_t i = 0U; i < invalid_build_configs.size(); ++i) {
    GetThreadLocalContext().SetGraphOption({{"ge.buildConfig", invalid_build_configs[i]}});
    const std::string model_name = "bc_makefile_option_" + std::to_string(i);
    Om2CodegenArtifact so_artifact;
    EXPECT_NE(
        Om2Utils::CompileGeneratedCppToSo(MakeBuildConfigTestArtifacts(model_name), model_name, so_artifact, false),
        SUCCESS)
        << invalid_build_configs[i];
  }
}

TEST_F(Om2CodegenUt, CompileGeneratedCppToSo_BuildConfigCrossCompilerMissing_ReportsInternalError) {
  ScopedEnvVar asan_guard("ASAN_OPTIONS", "detect_leaks=0:halt_on_error=0");
  ScopedEnvVar lsan_guard("LSAN_OPTIONS", "exitcode=0");
  ScopedGraphOptions graph_guard;
  const std::string build_config = "make -s CXX=/nonexistent/om2/aarch64-linux-gnu-g++";
  GetThreadLocalContext().SetGraphOption({{"ge.buildConfig", build_config}});
  (void)ErrorManager::GetInstance().GetErrorMessage();

  EXPECT_NE(CompileBuildConfigArtifacts("bc_cross_compiler_missing"), SUCCESS);
  const std::string error_message = ErrorManager::GetInstance().GetErrorMessage();
  EXPECT_EQ(error_message.find("E10001"), std::string::npos);
  EXPECT_NE(error_message.find("E19999"), std::string::npos);
  EXPECT_NE(error_message.find(build_config), std::string::npos);
}

TEST_F(Om2CodegenUt, CompileGeneratedCppToSo_BuildConfigTargetDevlibMissing_ReportsInternalError) {
  ScopedEnvVar asan_guard("ASAN_OPTIONS", "detect_leaks=0:halt_on_error=0");
  ScopedEnvVar lsan_guard("LSAN_OPTIONS", "exitcode=0");
  ScopedGraphOptions graph_guard;
  const std::string build_config = "make -s LDFLAGS='-shared -L/nonexistent/om2/devlib -lom2_missing_devlib'";
  GetThreadLocalContext().SetGraphOption({{"ge.buildConfig", build_config}});
  (void)ErrorManager::GetInstance().GetErrorMessage();

  EXPECT_NE(CompileBuildConfigArtifacts("bc_target_devlib_missing"), SUCCESS);
  const std::string error_message = ErrorManager::GetInstance().GetErrorMessage();
  EXPECT_EQ(error_message.find("E10001"), std::string::npos);
  EXPECT_NE(error_message.find("E19999"), std::string::npos);
  EXPECT_NE(error_message.find(build_config), std::string::npos);
}

TEST_F(Om2CodegenUt, CompileGeneratedCppToSo_BuildConfigMakeFailure_ReportsInternalError) {
  ScopedEnvVar asan_guard("ASAN_OPTIONS", "detect_leaks=0:halt_on_error=0");
  ScopedEnvVar lsan_guard("LSAN_OPTIONS", "exitcode=0");
  ScopedGraphOptions graph_guard;
  ScopedTempDir temp_dir;
  ASSERT_TRUE(temp_dir.WriteFile("make", "#!/bin/sh\nexit 1\n", 0755));
  const std::string build_config = temp_dir.Path("make");
  GetThreadLocalContext().SetGraphOption({{"ge.buildConfig", build_config}});
  (void)ErrorManager::GetInstance().GetErrorMessage();

  EXPECT_NE(CompileBuildConfigArtifacts("bc_make_failure"), SUCCESS);
  const std::string error_message = ErrorManager::GetInstance().GetErrorMessage();
  EXPECT_EQ(error_message.find("E10001"), std::string::npos);
  EXPECT_NE(error_message.find("E19999"), std::string::npos);
  EXPECT_NE(error_message.find(build_config), std::string::npos);
}

TEST_F(Om2CodegenUt, CompileGeneratedCppToSo_DefaultMakeFailure_DoesNotReportInvalidArgument) {
  ScopedEnvVar asan_guard("ASAN_OPTIONS", "detect_leaks=0:halt_on_error=0");
  ScopedEnvVar lsan_guard("LSAN_OPTIONS", "exitcode=0");
  ScopedGraphOptions graph_guard;
  ScopedTempDir temp_dir;
  ASSERT_TRUE(temp_dir.CreateDir("bin"));
  ASSERT_TRUE(PrepareCommandSymlink(temp_dir, "bin/env", "env"));
  ASSERT_TRUE(temp_dir.WriteFile("bin/make", "#!/bin/sh\nexit 1\n", 0755));
  ScopedEnvVar path_guard("PATH", temp_dir.Path("bin").c_str());
  GetThreadLocalContext().SetGraphOption({{"ge.buildConfig", ""}});
  (void)ErrorManager::GetInstance().GetErrorMessage();

  EXPECT_NE(CompileBuildConfigArtifacts("default_make_failure"), SUCCESS);
  const std::string error_message = ErrorManager::GetInstance().GetErrorMessage();
  EXPECT_EQ(error_message.find("E10001"), std::string::npos);
  EXPECT_NE(error_message.find("E19999"), std::string::npos);
  EXPECT_EQ(error_message.find("specified by build_config"), std::string::npos);
}

TEST_F(Om2CodegenUt, CompileGeneratedCppToSo_HostEnvNativeArmAlias_Ok) {
  struct utsname uts;
  if ((uname(&uts) != 0) || (std::string(uts.machine) != "aarch64")) {
    GTEST_SKIP() << "native arm64 alias branch is only stable on aarch64 host";
  }

  ScopedEnvVar asan_guard("ASAN_OPTIONS", "detect_leaks=0:halt_on_error=0");
  ScopedEnvVar lsan_guard("LSAN_OPTIONS", "exitcode=0");
  ScopedGraphOptions graph_guard;
  GetThreadLocalContext().SetGraphOption({{"ge.host_env_os", "linux"}, {"ge.host_env_cpu", "arm64"}});

  const std::string model_name = "host_env_arm64_alias";
  Om2CodegenArtifact so_artifact;
  ASSERT_EQ(Om2Utils::CompileGeneratedCppToSo(MakeBuildConfigTestArtifacts(model_name), model_name, so_artifact, false),
            SUCCESS);
  EXPECT_FALSE(so_artifact.data.empty());
}

TEST_F(Om2CodegenUt, CompileGeneratedCppToSo_HostEnvNativeX86_Ok) {
  if (GetNativeMachine() != "x86_64") {
    GTEST_SKIP() << "native x86_64 branch runs on x86_64 host";
  }
  ScopedEnvVar asan_guard("ASAN_OPTIONS", "detect_leaks=0:halt_on_error=0");
  ScopedEnvVar lsan_guard("LSAN_OPTIONS", "exitcode=0");
  ScopedGraphOptions graph_guard;
  GetThreadLocalContext().SetGraphOption({{"ge.host_env_os", "linux"}, {"ge.host_env_cpu", "x86_64"}});

  EXPECT_EQ(CompileBuildConfigArtifacts("host_env_x86_native"), SUCCESS);
}

TEST_F(Om2CodegenUt, CompileGeneratedCppToSo_HostEnvNonArmTarget_Ok) {
  ScopedEnvVar asan_guard("ASAN_OPTIONS", "detect_leaks=0:halt_on_error=0");
  ScopedEnvVar lsan_guard("LSAN_OPTIONS", "exitcode=0");
  ScopedGraphOptions graph_guard;
  GetThreadLocalContext().SetGraphOption({{"ge.host_env_os", "linux"}, {"ge.host_env_cpu", "riscv64"}});

  const std::string model_name = "host_env_non_arm";
  Om2CodegenArtifact so_artifact;
  ASSERT_EQ(Om2Utils::CompileGeneratedCppToSo(MakeBuildConfigTestArtifacts(model_name), model_name, so_artifact, false),
            SUCCESS);
  EXPECT_FALSE(so_artifact.data.empty());
}

TEST_F(Om2CodegenUt, CompileGeneratedCppToSo_CrossCompileSystemCompiler_Ok) {
  if (GetNativeMachine() != "x86_64") {
    GTEST_SKIP() << "cross-compiler injection coverage runs on x86_64 host";
  }
  ScopedEnvVar asan_guard("ASAN_OPTIONS", "detect_leaks=0:halt_on_error=0");
  ScopedEnvVar lsan_guard("LSAN_OPTIONS", "exitcode=0");
  ScopedGraphOptions graph_guard;
  ScopedTempDir temp_dir;
  ASSERT_TRUE(temp_dir.CreateDir("bin"));
  ASSERT_TRUE(temp_dir.CreateDir("ascend/devlib/linux/aarch64"));
  ASSERT_TRUE(PrepareMakeRuntime(temp_dir, "bin"));
  ASSERT_TRUE(PrepareFakeCompiler(temp_dir, "bin/aarch64-linux-gnu-g++"));

  ScopedEnvVar path_guard("PATH", temp_dir.Path("bin").c_str());
  ScopedEnvVar ascend_home_guard("ASCEND_HOME_PATH", temp_dir.Path("ascend").c_str());
  GetThreadLocalContext().SetGraphOption({{"ge.host_env_os", "linux"}, {"ge.host_env_cpu", "aarch64"}});

  EXPECT_EQ(CompileBuildConfigArtifacts("cross_system_compiler"), SUCCESS);
}

TEST_F(Om2CodegenUt, CompileGeneratedCppToSo_CrossCompileCannCompiler_Ok) {
  if (GetNativeMachine() != "x86_64") {
    GTEST_SKIP() << "cross-compiler injection coverage runs on x86_64 host";
  }
  ScopedEnvVar asan_guard("ASAN_OPTIONS", "detect_leaks=0:halt_on_error=0");
  ScopedEnvVar lsan_guard("LSAN_OPTIONS", "exitcode=0");
  ScopedGraphOptions graph_guard;
  ScopedTempDir temp_dir;
  ASSERT_TRUE(temp_dir.CreateDir("empty_bin"));
  ASSERT_TRUE(temp_dir.CreateDir("ascend/tools/hcc/bin"));
  ASSERT_TRUE(temp_dir.CreateDir("ascend/devlib/linux/aarch64"));
  ASSERT_TRUE(PrepareMakeRuntime(temp_dir, "empty_bin"));
  ASSERT_TRUE(PrepareFakeCompiler(temp_dir, "ascend/tools/hcc/bin/aarch64-target-linux-gnu-g++"));

  ScopedEnvVar path_guard("PATH", temp_dir.Path("empty_bin").c_str());
  ScopedEnvVar ascend_home_guard("ASCEND_HOME_PATH", temp_dir.Path("ascend").c_str());
  GetThreadLocalContext().SetGraphOption({{"ge.host_env_os", "linux"}, {"ge.host_env_cpu", "arm64"}});

  EXPECT_EQ(CompileBuildConfigArtifacts("cross_cann_compiler"), SUCCESS);
}

TEST_F(Om2CodegenUt, CompileGeneratedCppToSo_CrossCompileCompilerMissing_Rejected) {
  if (GetNativeMachine() != "x86_64") {
    GTEST_SKIP() << "cross-compiler injection coverage runs on x86_64 host";
  }
  ScopedEnvVar asan_guard("ASAN_OPTIONS", "detect_leaks=0:halt_on_error=0");
  ScopedEnvVar lsan_guard("LSAN_OPTIONS", "exitcode=0");
  ScopedGraphOptions graph_guard;
  ScopedTempDir temp_dir;
  ASSERT_TRUE(temp_dir.CreateDir("empty_bin"));
  ASSERT_TRUE(temp_dir.CreateDir("ascend/devlib/linux/aarch64"));
  ASSERT_TRUE(PrepareMakeRuntime(temp_dir, "empty_bin"));

  ScopedEnvVar path_guard("PATH", temp_dir.Path("empty_bin").c_str());
  ScopedEnvVar ascend_home_guard("ASCEND_HOME_PATH", temp_dir.Path("ascend").c_str());
  GetThreadLocalContext().SetGraphOption({{"ge.host_env_os", "linux"}, {"ge.host_env_cpu", "aarch64"}});
  (void)ErrorManager::GetInstance().GetErrorMessage();

  EXPECT_NE(CompileBuildConfigArtifacts("cross_compiler_missing"), SUCCESS);
  const std::string error_message = ErrorManager::GetInstance().GetErrorMessage();
  EXPECT_NE(error_message.find("E10001"), std::string::npos);
  EXPECT_EQ(error_message.find("E19999"), std::string::npos);
  EXPECT_NE(error_message.find("cross-compiler not found"), std::string::npos);
}

TEST_F(Om2CodegenUt, CompileGeneratedCppToSo_CrossCompileDevlibMissing_Rejected) {
  if (GetNativeMachine() != "x86_64") {
    GTEST_SKIP() << "cross-compiler injection coverage runs on x86_64 host";
  }
  ScopedEnvVar asan_guard("ASAN_OPTIONS", "detect_leaks=0:halt_on_error=0");
  ScopedEnvVar lsan_guard("LSAN_OPTIONS", "exitcode=0");
  ScopedGraphOptions graph_guard;
  ScopedTempDir temp_dir;
  ASSERT_TRUE(temp_dir.CreateDir("bin"));
  ASSERT_TRUE(temp_dir.CreateDir("ascend"));
  ASSERT_TRUE(PrepareMakeRuntime(temp_dir, "bin"));
  ASSERT_TRUE(PrepareFakeCompiler(temp_dir, "bin/aarch64-linux-gnu-g++"));

  ScopedEnvVar path_guard("PATH", temp_dir.Path("bin").c_str());
  ScopedEnvVar ascend_home_guard("ASCEND_HOME_PATH", temp_dir.Path("ascend").c_str());
  GetThreadLocalContext().SetGraphOption({{"ge.host_env_os", "linux"}, {"ge.host_env_cpu", "aarch64"}});
  (void)ErrorManager::GetInstance().GetErrorMessage();

  EXPECT_NE(CompileBuildConfigArtifacts("cross_devlib_missing"), SUCCESS);
  const std::string error_message = ErrorManager::GetInstance().GetErrorMessage();
  EXPECT_NE(error_message.find("E10001"), std::string::npos);
  EXPECT_EQ(error_message.find("E19999"), std::string::npos);
  EXPECT_NE(error_message.find("devlib not found"), std::string::npos);
}

TEST_F(Om2CodegenUt, CompileGeneratedCppToSo_CrossCompileAscendHomeMissing_Rejected) {
  if (GetNativeMachine() != "x86_64") {
    GTEST_SKIP() << "cross-compiler injection coverage runs on x86_64 host";
  }
  ScopedEnvVar asan_guard("ASAN_OPTIONS", "detect_leaks=0:halt_on_error=0");
  ScopedEnvVar lsan_guard("LSAN_OPTIONS", "exitcode=0");
  ScopedGraphOptions graph_guard;
  ScopedTempDir temp_dir;
  ASSERT_TRUE(temp_dir.CreateDir("bin"));
  ASSERT_TRUE(PrepareMakeRuntime(temp_dir, "bin"));
  ASSERT_TRUE(PrepareFakeCompiler(temp_dir, "bin/aarch64-linux-gnu-g++"));

  ScopedEnvVar path_guard("PATH", temp_dir.Path("bin").c_str());
  ScopedUnsetEnvVar ascend_home_guard("ASCEND_HOME_PATH");
  GetThreadLocalContext().SetGraphOption({{"ge.host_env_os", "linux"}, {"ge.host_env_cpu", "aarch64"}});

  EXPECT_NE(CompileBuildConfigArtifacts("cross_ascend_home_missing"), SUCCESS);
}

TEST_F(Om2CodegenUt, StablePartProvider_AllIds_Ok) {
  const std::vector<std::pair<StablePartId, std::string>> cases = {
      {StablePartId::kChkStatusMacro, "#define OM2_CHK_STATUS"},
      {StablePartId::kChkNotNullMacro, "#define OM2_CHK_NOTNULL"},
      {StablePartId::kChkTrueMacro, "#define OM2_CHK_TRUE"},
      {StablePartId::kGetAddrMacro, "#define GET_ADDR"},
      {StablePartId::kMakeGuardMacro, "#define OM2_MAKE_GUARD"},
      {StablePartId::kInterfaceMacros, "#define OM2_CHK_STATUS"},
      {StablePartId::kPointerHelpers, "inline uint64_t PtrToValue"},
      {StablePartId::kFlattenHostArgs, "inline std::vector<uint64_t> FlattenHostArgs"},
      {StablePartId::kInterfacePointerHelpers, "inline std::vector<uint64_t> FlattenHostArgs"},
      {StablePartId::kScopeGuard, "class ScopeGuard"},
      {StablePartId::kReadBinaryFileToBuffer, "BinaryBuffer ReadBinaryFileToBuffer"},
      {StablePartId::kGenerateJsonFile, "aclError GenerateJsonFile"},
      {StablePartId::kInterfaceDumpApis, "struct GertModelTaskDesc"},
      {StablePartId::kOm2LogMacros, "#define OM2_LOGD"},
  };

  for (const auto &test_case : cases) {
    std::string output;
    ASSERT_EQ(ResolveStablePart(test_case.first, output), SUCCESS);
    EXPECT_NE(output.find(test_case.second), std::string::npos) << output;
  }

  std::string output;
  ASSERT_EQ(ResolveStablePart(static_cast<StablePartId>(0xff), output), FAILED);
  EXPECT_TRUE(output.empty());
}

TEST_F(Om2CodegenUt, StablePartProvider_Om2LogMacros_Ok) {
  std::string output;
  ASSERT_EQ(ResolveStablePart(StablePartId::kOm2LogMacros, output), SUCCESS);
  ExpectContainsAll(output, {
                                "#define OM2_LOGD",
                                "#define OM2_LOGI",
                                "#define OM2_LOGW",
                                "#define OM2_LOGE",
                                "Om2GetTid",
                                "Om2IsLogEnable",
                                "OM2_MODULE_NAME",
                                "OM2_LOG_HEADER",
                            });
}

TEST_F(Om2CodegenUt, Om2CodePrinter_GetFileName_DefaultNames) {
  const std::string model_name = "test_model";
  Om2CodePrinter printer(model_name);

  EXPECT_EQ(printer.GetFileName(GeneratedFileIndex::kModelApiHeaderFile), "om2_model_api.h");
  EXPECT_EQ(printer.GetFileName(GeneratedFileIndex::kInterfaceHeaderFile), model_name + "_internal.h");
  EXPECT_EQ(printer.GetFileName(GeneratedFileIndex::kResourcesFile), model_name + "_resources.cpp");
  EXPECT_EQ(printer.GetFileName(GeneratedFileIndex::kArgsManagerFile), model_name + "_args_manager.cpp");
  EXPECT_EQ(printer.GetFileName(GeneratedFileIndex::kKernelRegistryFile), model_name + "_kernel_reg.cpp");
  EXPECT_EQ(printer.GetFileName(GeneratedFileIndex::kLoadingAndRunningFile), model_name + "_load_and_run.cpp");
  EXPECT_EQ(printer.GetFileName(GeneratedFileIndex::kCMakeListsFile), "Makefile");
}

// ============ TaskCodeBuilderUtil coverage tests ============

TEST_F(Om2CodegenUt, TaskCodeBuilderUtil_BuildTaskIoEntries_WithTensorInfo) {
  AstContext ctx;
  AstBuildContext ast(ctx);

  AddrSemantic addr;
  addr.kind = AddrValueKind::kInputInstance;
  addr.symbol_hint = "input_var";
  addr.tensor_info = Om2TensorInfo{};
  addr.tensor_info->args_offset = 16U;

  auto *entries = TaskCodeBuilderUtil::BuildTaskIoEntries(ast, {addr});
  ASSERT_NE(entries, nullptr);
  auto output = EmitNode(*entries);
  ExpectContainsAll(output, {"input_var", "16U"});
}

TEST_F(Om2CodegenUt, TaskCodeBuilderUtil_BuildTaskIoEntries_SkipNoTensorInfo) {
  AstContext ctx;
  AstBuildContext ast(ctx);

  AddrSemantic addr;
  addr.kind = AddrValueKind::kWorkspace;
  addr.symbol_hint = "ws_var";

  auto *entries = TaskCodeBuilderUtil::BuildTaskIoEntries(ast, {addr});
  ASSERT_NE(entries, nullptr);
  auto output = EmitNode(*entries);
  EXPECT_TRUE(output.find("ws_var") == std::string::npos);
}

TEST_F(Om2CodegenUt, TaskCodeBuilderUtil_BuildTaskIoEntries_MixedAddrs) {
  AstContext ctx;
  AstBuildContext ast(ctx);

  AddrSemantic addr_with_tensor;
  addr_with_tensor.kind = AddrValueKind::kInputInstance;
  addr_with_tensor.symbol_hint = "with_tensor";
  addr_with_tensor.tensor_info = Om2TensorInfo{};
  addr_with_tensor.tensor_info->args_offset = 0U;

  AddrSemantic addr_without_tensor;
  addr_without_tensor.kind = AddrValueKind::kWorkspace;
  addr_without_tensor.symbol_hint = "without_tensor";

  auto *entries = TaskCodeBuilderUtil::BuildTaskIoEntries(ast, {addr_with_tensor, addr_without_tensor});
  ASSERT_NE(entries, nullptr);
  auto output = EmitNode(*entries);
  ExpectContainsAll(output, {"with_tensor"});
  EXPECT_TRUE(output.find("without_tensor") == std::string::npos);
}

TEST_F(Om2CodegenUt, TaskCodeBuilderUtil_BuildTaskIoEntries_EmptyAddrs) {
  AstContext ctx;
  AstBuildContext ast(ctx);

  auto *entries = TaskCodeBuilderUtil::BuildTaskIoEntries(ast, {});
  ASSERT_NE(entries, nullptr);
  auto output = EmitNode(*entries);
  EXPECT_FALSE(output.empty());
}

TEST_F(Om2CodegenUt, TaskCodeBuilderUtil_BuildWorkspaceAddrs_Normal) {
  AstContext ctx;
  AstBuildContext ast(ctx);

  AddrSemantic addr1;
  addr1.kind = AddrValueKind::kWorkspace;
  addr1.symbol_hint = "ws1";

  AddrSemantic addr2;
  addr2.kind = AddrValueKind::kWorkspace;
  addr2.symbol_hint = "ws2";

  auto *entries = TaskCodeBuilderUtil::BuildWorkspaceAddrs(ast, {addr1, addr2});
  ASSERT_NE(entries, nullptr);
  auto output = EmitNode(*entries);
  ExpectContainsAll(output, {"PtrToU64", "ws1", "ws2"});
}

TEST_F(Om2CodegenUt, TaskCodeBuilderUtil_BuildWorkspaceAddrs_Empty) {
  AstContext ctx;
  AstBuildContext ast(ctx);

  auto *entries = TaskCodeBuilderUtil::BuildWorkspaceAddrs(ast, {});
  ASSERT_NE(entries, nullptr);
  auto output = EmitNode(*entries);
  EXPECT_FALSE(output.empty());
}

TEST_F(Om2CodegenUt, TaskCodeBuilderUtil_BuildWorkspaceSizes_Normal) {
  AstContext ctx;
  AstBuildContext ast(ctx);

  AddrSemantic addr1;
  addr1.kind = AddrValueKind::kWorkspace;
  addr1.byte_size = 256U;

  AddrSemantic addr2;
  addr2.kind = AddrValueKind::kWorkspace;
  addr2.byte_size = 512U;

  auto *entries = TaskCodeBuilderUtil::BuildWorkspaceSizes(ast, {addr1, addr2});
  ASSERT_NE(entries, nullptr);
  auto output = EmitNode(*entries);
  ExpectContainsAll(output, {"256U", "512U"});
}

TEST_F(Om2CodegenUt, TaskCodeBuilderUtil_BuildWorkspaceSizes_Empty) {
  AstContext ctx;
  AstBuildContext ast(ctx);

  auto *entries = TaskCodeBuilderUtil::BuildWorkspaceSizes(ast, {});
  ASSERT_NE(entries, nullptr);
  auto output = EmitNode(*entries);
  EXPECT_FALSE(output.empty());
}

TEST_F(Om2CodegenUt, TaskCodeBuilderUtil_BuildL0ArgSlotEntries_AllKinds) {
  AstContext ctx;
  AstBuildContext ast(ctx);

  AddrSemantic input_addr;
  input_addr.kind = AddrValueKind::kInputInstance;
  input_addr.tensor_info = Om2TensorInfo{};
  input_addr.tensor_info->args_offset = 0U;

  AddrSemantic output_addr;
  output_addr.kind = AddrValueKind::kOutputInstance;
  output_addr.tensor_info = Om2TensorInfo{};
  output_addr.tensor_info->args_offset = 8U;

  AddrSemantic workspace_addr;
  workspace_addr.kind = AddrValueKind::kWorkspace;

  AddrSemantic tiling_addr;
  tiling_addr.kind = AddrValueKind::kTiling;
  tiling_addr.byte_size = 64U;

  AddrSemantic custom_value_addr;
  custom_value_addr.kind = AddrValueKind::kCustomValue;
  custom_value_addr.custom_value = 42U;

  AddrSemantic placeholder_addr;
  placeholder_addr.kind = AddrValueKind::kPlaceholder;

  AddrSemantic level1_desc_addr;
  level1_desc_addr.kind = AddrValueKind::kLevel1DescPtr;

  AddrSemantic shape_info_addr;
  shape_info_addr.kind = AddrValueKind::kShapeInfoBuffer;
  shape_info_addr.shape_info = std::vector<int64_t>{1, 2, 3};

  AddrSemantic ffts_addr;
  ffts_addr.kind = AddrValueKind::kFftsAddr;

  AddrSemantic event_addr;
  event_addr.kind = AddrValueKind::kEventAddr;
  event_addr.event_id = 7U;

  AddrSemantic overflow_addr;
  overflow_addr.kind = AddrValueKind::kOverflowAddr;

  AddrSemantic optional_empty_addr;
  optional_empty_addr.kind = AddrValueKind::kOptionalEmpty;

  AddrSemantic empty_addr;
  empty_addr.kind = AddrValueKind::kEmptyAddr;

  AddrSemantic const_tensor_addr;
  const_tensor_addr.kind = AddrValueKind::kConstTensor;
  const_tensor_addr.tensor_info = Om2TensorInfo{};
  const_tensor_addr.tensor_info->args_offset = 16U;

  auto *entries = TaskCodeBuilderUtil::BuildL0ArgSlotEntries(
      ast, {input_addr, output_addr, workspace_addr, tiling_addr, custom_value_addr, placeholder_addr, level1_desc_addr,
            shape_info_addr, ffts_addr, event_addr, overflow_addr, optional_empty_addr, empty_addr, const_tensor_addr});
  ASSERT_NE(entries, nullptr);
  auto output = EmitNode(*entries);
  ExpectContainsAll(output, {
                                "GERT_MODEL_ARG_INPUT",
                                "GERT_MODEL_ARG_OUTPUT",
                                "GERT_MODEL_ARG_WORKSPACE",
                                "GERT_MODEL_ARG_TILING",
                                "GERT_MODEL_ARG_CUSTOM_VALUE",
                                "GERT_MODEL_ARG_PLACEHOLDER",
                                "GERT_MODEL_ARG_LEVEL1_DESC",
                                "GERT_MODEL_ARG_SHAPE_INFO",
                                "GERT_MODEL_ARG_FFTS_ADDR",
                                "GERT_MODEL_ARG_EVENT_ADDR",
                                "GERT_MODEL_ARG_OVERFLOW_ADDR",
                                "GERT_MODEL_ARG_EMPTY_ADDR",
                            });
}

TEST_F(Om2CodegenUt, TaskCodeBuilderUtil_RenderDispatchFunc_Normal) {
  AstContext ctx;
  AstBuildContext ast(ctx);

  std::vector<BodyItem> body;
  body.push_back(ast.Call("DoSomething", {}));

  std::vector<DeclNode *> items;
  auto ret = TaskCodeBuilderUtil::RenderDispatchFunc(ast, "TestDispatch", body, items);
  EXPECT_EQ(ret, SUCCESS);
  ASSERT_EQ(items.size(), 1U);
  auto output = EmitNode(*items[0U]);
  ExpectContainsAll(output, {"aclError", "TestDispatch", "DoSomething", "ACL_SUCCESS"});
}

TEST_F(Om2CodegenUt, TaskCodeBuilderUtil_RenderDispatchFunc_EmptyBody) {
  AstContext ctx;
  AstBuildContext ast(ctx);

  std::vector<BodyItem> body;
  std::vector<DeclNode *> items;
  auto ret = TaskCodeBuilderUtil::RenderDispatchFunc(ast, "EmptyDispatch", body, items);
  EXPECT_EQ(ret, SUCCESS);
  ASSERT_EQ(items.size(), 1U);
  auto output = EmitNode(*items[0U]);
  ExpectContainsAll(output, {"aclError", "EmptyDispatch", "ACL_SUCCESS"});
}

TEST_F(Om2CodegenUt, TaskCodeBuilderUtil_AppendReportLaunchedTaskCall_NoArgsTable) {
  AstContext ctx;
  AstBuildContext ast(ctx);

  TaskSemanticHeader header;
  header.op_name = "test_op";
  header.op_type = "TestOp";
  header.op_desc_id = 1;

  AddrSemantic input_addr;
  input_addr.kind = AddrValueKind::kInputInstance;
  input_addr.symbol_hint = "input_var";
  input_addr.tensor_info = Om2TensorInfo{};
  input_addr.tensor_info->args_offset = 0U;

  AddrSemantic output_addr;
  output_addr.kind = AddrValueKind::kOutputInstance;
  output_addr.symbol_hint = "output_var";
  output_addr.tensor_info = Om2TensorInfo{};
  output_addr.tensor_info->args_offset = 8U;

  AddrSemantic ws_addr;
  ws_addr.kind = AddrValueKind::kWorkspace;
  ws_addr.symbol_hint = "ws_var";
  ws_addr.byte_size = 128U;

  std::vector<BodyItem> items;
  auto model_id = ast.Var("uint32_t", "model_id");
  auto instance_handle = ast.Var("void *", "instance_handle");
  auto args_table = ast.Var("const ArgsTable *", "args_table");
  auto stream = ast.Var("void *", "stream");

  auto ret = TaskCodeBuilderUtil::AppendReportLaunchedTaskCall(
      ast, items, "prefix", header, nullptr, {input_addr}, {output_addr}, {ws_addr}, ModelTaskType::MODEL_TASK_KERNEL,
      1U, stream, model_id, instance_handle, args_table, false);
  EXPECT_EQ(ret, SUCCESS);
  ASSERT_FALSE(items.empty());
  std::string output;
  for (auto &item : items) {
    auto *stmt = item.Resolve(ctx);
    if (stmt != nullptr) {
      output += EmitNode(*stmt);
    }
  }
  ExpectContainsAll(output,
                    {"ReportLaunchedOm2Task", "test_op", "TestOp", "prefix_report_inputs", "prefix_report_outputs",
                     "prefix_report_workspace_addrs", "prefix_report_workspace_sizes"});
}

TEST_F(Om2CodegenUt, TaskCodeBuilderUtil_AppendReportLaunchedTaskCall_NoInputsOutputsWorkspaces) {
  AstContext ctx;
  AstBuildContext ast(ctx);

  TaskSemanticHeader header;
  header.op_name = "empty_op";
  header.op_type = "EmptyOp";
  header.op_desc_id = 0;

  std::vector<BodyItem> items;
  auto model_id = ast.Var("uint32_t", "model_id");
  auto instance_handle = ast.Var("void *", "instance_handle");
  auto args_table = ast.Var("const ArgsTable *", "args_table");
  auto stream = ast.Var("void *", "stream");

  auto ret = TaskCodeBuilderUtil::AppendReportLaunchedTaskCall(ast, items, "p", header, nullptr, {}, {}, {},
                                                               ModelTaskType::MODEL_TASK_KERNEL, 1U, stream, model_id,
                                                               instance_handle, args_table, false);
  EXPECT_EQ(ret, SUCCESS);
  ASSERT_FALSE(items.empty());
  auto *stmt = items[0U].Resolve(ctx);
  ASSERT_NE(stmt, nullptr);
  auto output = EmitNode(*stmt);
  ExpectContainsAll(output, {"ReportLaunchedOm2Task", "nullptr"});
}

TEST_F(Om2CodegenUt, TaskCodeBuilderUtil_AppendReportLaunchedTaskCall_WithRawAddress) {
  AstContext ctx;
  AstBuildContext ast(ctx);

  TaskSemanticHeader header;
  header.op_name = "raw_op";
  header.op_type = "RawOp";
  header.op_desc_id = 2;

  std::vector<BodyItem> items;
  auto model_id = ast.Var("uint32_t", "model_id");
  auto instance_handle = ast.Var("void *", "instance_handle");
  auto args_table = ast.Var("const ArgsTable *", "args_table");
  auto stream = ast.Var("void *", "stream");

  auto ret = TaskCodeBuilderUtil::AppendReportLaunchedTaskCall(ast, items, "raw", header, nullptr, {}, {}, {},
                                                               ModelTaskType::MODEL_TASK_KERNEL, 1U, stream, model_id,
                                                               instance_handle, args_table, false, true);
  EXPECT_EQ(ret, SUCCESS);
  ASSERT_FALSE(items.empty());
  auto *stmt = items[0U].Resolve(ctx);
  ASSERT_NE(stmt, nullptr);
  auto output = EmitNode(*stmt);
  ExpectContainsAll(output, {"ReportLaunchedOm2Task"});
}

TEST_F(Om2CodegenUt, TaskCodeBuilderUtil_BuildReportTaskPreprocessCall_NoArgsTable) {
  AstContext ctx;
  AstBuildContext ast(ctx);

  TaskSemanticHeader header;
  header.op_name = "preprocess_op";
  header.op_type = "PreprocessOp";
  header.op_desc_id = 3;

  AddrSemantic input_addr;
  input_addr.kind = AddrValueKind::kInputInstance;
  input_addr.symbol_hint = "in_var";
  input_addr.tensor_info = Om2TensorInfo{};
  input_addr.tensor_info->args_offset = 0U;

  AddrSemantic ws_addr;
  ws_addr.kind = AddrValueKind::kWorkspace;
  ws_addr.symbol_hint = "ws_var";
  ws_addr.byte_size = 64U;

  auto model_id = ast.Var("uint32_t", "model_id");
  auto instance_handle = ast.Var("void *", "instance_handle");
  auto args_table = ast.Var("const ArgsTable *", "args_table");
  auto stream = ast.Var("void *", "stream");
  auto l0_info = ast.Var("const void *", "l0_info");

  auto result = TaskCodeBuilderUtil::BuildReportTaskPreprocessCall(
      ast, header, nullptr, {input_addr}, {}, {ws_addr}, ModelTaskType::MODEL_TASK_KERNEL, 2U, stream, model_id,
      instance_handle, args_table, l0_info, false);
  auto *result_expr = result.Get();
  ASSERT_NE(result_expr, nullptr);
  auto output = EmitNode(*result_expr);
  ExpectContainsAll(output, {"ReportOm2TaskPreprocess", "preprocess_op", "PreprocessOp"});
}

TEST_F(Om2CodegenUt, TaskCodeBuilderUtil_BuildReportTaskPreprocessCall_WithRawAddress) {
  AstContext ctx;
  AstBuildContext ast(ctx);

  TaskSemanticHeader header;
  header.op_name = "raw_pre_op";
  header.op_type = "RawPreOp";
  header.op_desc_id = 4;

  auto model_id = ast.Var("uint32_t", "model_id");
  auto instance_handle = ast.Var("void *", "instance_handle");
  auto args_table = ast.Var("const ArgsTable *", "args_table");
  auto stream = ast.Var("void *", "stream");
  auto l0_info = ast.Var("const void *", "l0_info");

  auto result = TaskCodeBuilderUtil::BuildReportTaskPreprocessCall(
      ast, header, nullptr, {}, {}, {}, ModelTaskType::MODEL_TASK_KERNEL, 1U, stream, model_id, instance_handle,
      args_table, l0_info, false, true);
  auto *result_expr = result.Get();
  ASSERT_NE(result_expr, nullptr);
  auto output = EmitNode(*result_expr);
  ExpectContainsAll(output, {"ReportOm2TaskPreprocess", "raw_pre_op"});
}

TEST_F(Om2CodegenUt, TaskCodeBuilderUtil_BuildAddrField_Normal) {
  AstContext ctx;
  AstBuildContext ast(ctx);

  OpArgDesc desc;
  desc.mem_src = 1U;
  desc.offset = 128U;

  auto arg = TaskCodeBuilderUtil::BuildAddrField(ast, desc);
  auto *arg_expr = arg.Resolve(ctx);
  ASSERT_NE(arg_expr, nullptr);
  auto output = EmitNode(*arg_expr);
  ExpectContainsAll(output, {"mem_src", "offset", "128"});
}

TEST_F(Om2CodegenUt, TaskCodeBuilderUtil_BuildTensorDataField_WithTensorInfo) {
  AstContext ctx;
  AstBuildContext ast(ctx);

  OpArgDesc desc;
  desc.has_tensor_info = true;
  desc.size = 256U;
  desc.data_type = 0;
  desc.format = 1;
  desc.shape_dims = {1, 3, 224, 224};
  desc.args_offset = 16U;

  auto arg = TaskCodeBuilderUtil::BuildTensorDataField(ast, desc);
  auto *arg_expr = arg.Resolve(ctx);
  ASSERT_NE(arg_expr, nullptr);
  auto output = EmitNode(*arg_expr);
  ExpectContainsAll(output, {"tensor", "size", "256", "data_type", "format", "shape", "shape_dims", "args_offset"});
}

TEST_F(Om2CodegenUt, TaskCodeBuilderUtil_BuildTensorDataField_NoTensorInfo) {
  AstContext ctx;
  AstBuildContext ast(ctx);

  OpArgDesc desc;
  desc.has_tensor_info = false;

  auto arg = TaskCodeBuilderUtil::BuildTensorDataField(ast, desc);
  EXPECT_TRUE(arg.Empty());
}

TEST(Om2CodegenTypesUt, PublicTypesHaveStableDefaultsAndCallbacks) {
  GertModelLaunchKernelV2Params kernel{};
  GertModelLaunchStarsTaskWithFlagParams dsa{};
  GertModelTaskLaunchInfo launch{};
  GertModelLoadCallbacks callbacks{};
  GertModelTaskDesc task{};

  EXPECT_EQ(kernel.struct_size, sizeof(GertModelLaunchKernelV2Params));
  EXPECT_EQ(dsa.struct_size, sizeof(GertModelLaunchStarsTaskWithFlagParams));
  EXPECT_EQ(launch.struct_size, sizeof(GertModelTaskLaunchInfo));
  EXPECT_EQ(callbacks.struct_size, sizeof(GertModelLoadCallbacks));
  EXPECT_EQ(launch.launch_type, ACL_RT_LAUNCH_KERNEL_V2);
  EXPECT_EQ(task.kernel_type, static_cast<uint64_t>(ccKernelType::INVALID));
  static_assert(std::is_same<decltype(task.kernel_type), uint64_t>::value);
  static_assert(offsetof(GertModelTaskDesc, task_type) < offsetof(GertModelTaskDesc, kernel_type));
  static_assert(offsetof(GertModelTaskDesc, kernel_type) < offsetof(GertModelTaskDesc, stream));
  EXPECT_EQ(kernel.reserved_1, 0U);
  EXPECT_EQ(dsa.reserved_1, 0U);
  EXPECT_EQ(dsa.reserved_2, 0U);
  EXPECT_EQ(callbacks.report_model_base_info, nullptr);
  EXPECT_EQ(callbacks.launch_func, nullptr);
  EXPECT_FALSE(HasLegacyLaunchCallback<GertModelLoadCallbacks>::value);
  EXPECT_FALSE(HasLegacyPostCallback<GertModelLoadCallbacks>::value);
  EXPECT_FALSE(HasLegacyDataDumpCallback<GertModelLoadCallbacks>::value);
}

TEST(Om2CodegenTypesUt, LaunchTypeValuesAndUnionLayoutAreStable) {
  EXPECT_EQ(static_cast<uint64_t>(ACL_RT_LAUNCH_KERNEL_V2), 0U);
  EXPECT_EQ(static_cast<uint64_t>(RT_STARS_TASK_LAUNCH_WITH_FLAG), 1U);
  static_assert(offsetof(GertModelTaskLaunchParams, launch_kernel_v2_params) == 0U);
  static_assert(offsetof(GertModelTaskLaunchParams, launch_stars_task_params) == 0U);
  static_assert(sizeof(GertModelTaskLaunchParams) >= sizeof(GertModelLaunchKernelV2Params));
}

TEST_F(Om2CodegenUt, TaskCodeBuilderUtil_BuildWorkspaceDataField_Normal) {
  AstContext ctx;
  AstBuildContext ast(ctx);

  OpArgDesc desc;
  desc.size = 512U;

  auto arg = TaskCodeBuilderUtil::BuildWorkspaceDataField(ast, desc);
  auto *arg_expr = arg.Resolve(ctx);
  ASSERT_NE(arg_expr, nullptr);
  auto output = EmitNode(*arg_expr);
  ExpectContainsAll(output, {"tensor", "size", "512"});
}

TEST_F(Om2CodegenUt, TaskCodeBuilderUtil_BuildCustomValueDataField_Normal) {
  AstContext ctx;
  AstBuildContext ast(ctx);

  OpArgDesc desc;
  desc.custom_value = 99U;

  auto arg = TaskCodeBuilderUtil::BuildCustomValueDataField(ast, desc);
  auto *arg_expr = arg.Resolve(ctx);
  ASSERT_NE(arg_expr, nullptr);
  auto output = EmitNode(*arg_expr);
  ExpectContainsAll(output, {"custom_value", "99"});
}

TEST_F(Om2CodegenUt, TaskCodeBuilderUtil_BuildTilingDataField_WithRawData) {
  AstContext ctx;
  AstBuildContext ast(ctx);

  OpArgDesc desc;
  desc.raw_data = {0x01, 0x02, 0x03, 0x04};

  auto arg = TaskCodeBuilderUtil::BuildTilingDataField(ast, desc);
  auto *arg_expr = arg.Resolve(ctx);
  ASSERT_NE(arg_expr, nullptr);
  auto output = EmitNode(*arg_expr);
  ExpectContainsAll(output, {"tiling", "raw_data", "raw_data_len", "4"});
}

TEST_F(Om2CodegenUt, TaskCodeBuilderUtil_BuildTilingDataField_EmptyRawData) {
  AstContext ctx;
  AstBuildContext ast(ctx);

  OpArgDesc desc;
  desc.raw_data = {};

  auto arg = TaskCodeBuilderUtil::BuildTilingDataField(ast, desc);
  auto *arg_expr = arg.Resolve(ctx);
  ASSERT_NE(arg_expr, nullptr);
  auto output = EmitNode(*arg_expr);
  ExpectContainsAll(output, {"tiling", "raw_data", "raw_data_len", "0"});
}

TEST_F(Om2CodegenUt, TaskCodeBuilderUtil_RenderOpArgDesc_EmptyArgs) {
  AstContext ctx;
  AstBuildContext ast(ctx);

  auto arg = TaskCodeBuilderUtil::RenderOpArgDesc(ast, {});
  auto *expr = arg.Resolve(ctx);
  ASSERT_NE(expr, nullptr);
  auto output = EmitNode(*expr);
  EXPECT_NE(output.find("nullptr"), std::string::npos);
}

TEST_F(Om2CodegenUt, TaskCodeBuilderUtil_RenderOpArgDesc_AllTypes) {
  AstContext ctx;
  AstBuildContext ast(ctx);

  OpArgDesc input_desc;
  input_desc.type = OP_ARG_INPUT;
  input_desc.has_tensor_info = true;
  input_desc.size = 128U;
  input_desc.shape_dims = {1, 2};
  input_desc.args_offset = 0U;
  input_desc.mem_src = 0U;
  input_desc.offset = 0U;

  OpArgDesc output_desc;
  output_desc.type = OP_ARG_OUTPUT;
  output_desc.has_tensor_info = true;
  output_desc.size = 256U;
  output_desc.shape_dims = {1, 2};
  output_desc.args_offset = 8U;

  OpArgDesc workspace_desc;
  workspace_desc.type = OP_ARG_WORKSPACE;
  workspace_desc.size = 512U;

  OpArgDesc const_tensor_desc;
  const_tensor_desc.type = OP_ARG_CONST_TENSOR;
  const_tensor_desc.has_tensor_info = true;
  const_tensor_desc.size = 64U;
  const_tensor_desc.shape_dims = {1};
  const_tensor_desc.args_offset = 16U;

  OpArgDesc level1_desc;
  level1_desc.type = OP_ARG_LEVEL1_DESC;
  level1_desc.custom_value = 1U;

  OpArgDesc shape_info_desc;
  shape_info_desc.type = OP_ARG_SHAPE_INFO;
  shape_info_desc.custom_value = 2U;

  OpArgDesc custom_value_desc;
  custom_value_desc.type = OP_ARG_CUSTOM_VALUE;
  custom_value_desc.custom_value = 3U;

  OpArgDesc event_addr_desc;
  event_addr_desc.type = OP_ARG_EVENT_ADDR;
  event_addr_desc.custom_value = 4U;

  OpArgDesc tiling_desc;
  tiling_desc.type = OP_ARG_TILING;
  tiling_desc.raw_data = {0xAB, 0xCD};

  OpArgDesc placeholder_desc;
  placeholder_desc.type = OP_ARG_PLACEHOLDER;

  OpArgDesc optional_empty_desc;
  optional_empty_desc.type = OP_ARG_OPTIONAL_EMPTY;

  OpArgDesc ffts_addr_desc;
  ffts_addr_desc.type = OP_ARG_FFTS_ADDR;

  OpArgDesc overflow_addr_desc;
  overflow_addr_desc.type = OP_ARG_OVERFLOW_ADDR;

  OpArgDesc raw_addr_desc;
  raw_addr_desc.type = OP_ARG_RAW_ADDR;
  raw_addr_desc.mem_src = 0U;
  raw_addr_desc.offset = 32U;

  OpArgDesc unknown_type_desc;
  unknown_type_desc.type = 999;

  auto arg = TaskCodeBuilderUtil::RenderOpArgDesc(
      ast, {input_desc, output_desc, workspace_desc, const_tensor_desc, level1_desc, shape_info_desc, custom_value_desc,
            event_addr_desc, tiling_desc, placeholder_desc, optional_empty_desc, ffts_addr_desc, overflow_addr_desc,
            raw_addr_desc, unknown_type_desc});
  auto *arg_expr = arg.Resolve(ctx);
  ASSERT_NE(arg_expr, nullptr);
  auto output = EmitNode(*arg_expr);
  ExpectContainsAll(output, {
                                "OP_ARG_INPUT",
                                "OP_ARG_OUTPUT",
                                "OP_ARG_WORKSPACE",
                                "OP_ARG_CONST_TENSOR",
                                "OP_ARG_LEVEL1_DESC",
                                "OP_ARG_SHAPE_INFO",
                                "OP_ARG_CUSTOM_VALUE",
                                "OP_ARG_EVENT_ADDR",
                                "OP_ARG_TILING",
                                "OP_ARG_PLACEHOLDER",
                                "OP_ARG_OPTIONAL_EMPTY",
                                "OP_ARG_FFTS_ADDR",
                                "OP_ARG_OVERFLOW_ADDR",
                                "OP_ARG_RAW_ADDR",
                            });
}

TEST_F(Om2CodegenUt, TaskCodeBuilderUtil_ConvertAddrDesc_InputInstance) {
  AddrSemantic addr;
  addr.kind = AddrValueKind::kInputInstance;
  addr.tensor_info = Om2TensorInfo{};
  addr.tensor_info->size = 128U;
  addr.tensor_info->data_type = 0;
  addr.tensor_info->format = 1;
  addr.tensor_info->shape_dims = {1, 2};
  addr.const_index = 0;
  addr.mem_offset = 64;

  auto desc = TaskCodeBuilderUtil::ConvertAddrDesc(addr);
  EXPECT_EQ(desc.type, OP_ARG_INPUT);
  EXPECT_TRUE(desc.has_tensor_info);
  EXPECT_EQ(desc.size, 128U);
  EXPECT_EQ(desc.mem_src, MEM_SRC_CONST);
  EXPECT_EQ(desc.index, 0U);
  EXPECT_EQ(desc.offset, 64U);
}

TEST_F(Om2CodegenUt, TaskCodeBuilderUtil_ConvertAddrDesc_OutputInstance) {
  AddrSemantic addr;
  addr.kind = AddrValueKind::kOutputInstance;
  addr.tensor_info = Om2TensorInfo{};
  addr.mem_offset = 32;

  auto desc = TaskCodeBuilderUtil::ConvertAddrDesc(addr);
  EXPECT_EQ(desc.type, OP_ARG_OUTPUT);
  EXPECT_TRUE(desc.has_tensor_info);
}

TEST_F(Om2CodegenUt, TaskCodeBuilderUtil_ConvertAddrDesc_Workspace) {
  AddrSemantic addr;
  addr.kind = AddrValueKind::kWorkspace;
  addr.byte_size = 256U;
  addr.mem_offset = 128;

  auto desc = TaskCodeBuilderUtil::ConvertAddrDesc(addr);
  EXPECT_EQ(desc.type, OP_ARG_WORKSPACE);
  EXPECT_FALSE(desc.has_tensor_info);
  EXPECT_EQ(desc.size, 256U);
}

TEST_F(Om2CodegenUt, TaskCodeBuilderUtil_ConvertAddrDesc_ConstTensor) {
  AddrSemantic addr;
  addr.kind = AddrValueKind::kConstTensor;
  addr.tensor_info = Om2TensorInfo{};
  addr.const_index = 2;
  addr.mem_offset = 16;

  auto desc = TaskCodeBuilderUtil::ConvertAddrDesc(addr);
  EXPECT_EQ(desc.type, OP_ARG_CONST_TENSOR);
  EXPECT_TRUE(desc.has_tensor_info);
  EXPECT_EQ(desc.mem_src, MEM_SRC_CONST);
  EXPECT_EQ(desc.index, 2U);
}

TEST_F(Om2CodegenUt, TaskCodeBuilderUtil_ConvertAddrDesc_Variable) {
  AddrSemantic addr;
  addr.kind = AddrValueKind::kVariable;
  addr.tensor_info = Om2TensorInfo{};
  addr.tensor_info->size = 128U;
  addr.var_index = 2U;
  addr.mem_offset = 512;

  auto desc = TaskCodeBuilderUtil::ConvertAddrDesc(addr);
  EXPECT_EQ(desc.type, OP_ARG_VAR_TENSOR);
  EXPECT_EQ(desc.mem_src, MEM_SRC_VAR);
  EXPECT_EQ(desc.index, 2U);
  EXPECT_EQ(desc.offset, 512U);
  EXPECT_TRUE(desc.has_tensor_info);
  EXPECT_EQ(desc.size, 128U);
}

TEST_F(Om2CodegenUt, TaskCodeBuilderUtil_ConvertAddrDesc_Level1DescPtr) {
  AddrSemantic addr;
  addr.kind = AddrValueKind::kLevel1DescPtr;
  addr.level1_target_offset = 42U;
  addr.mem_offset = 8;

  auto desc = TaskCodeBuilderUtil::ConvertAddrDesc(addr);
  EXPECT_EQ(desc.type, OP_ARG_LEVEL1_DESC);
  EXPECT_EQ(desc.custom_value, 42U);
}

TEST_F(Om2CodegenUt, TaskCodeBuilderUtil_ConvertAddrDesc_CustomValue) {
  AddrSemantic addr;
  addr.kind = AddrValueKind::kCustomValue;
  addr.custom_value = 77U;

  auto desc = TaskCodeBuilderUtil::ConvertAddrDesc(addr);
  EXPECT_EQ(desc.type, OP_ARG_CUSTOM_VALUE);
  EXPECT_EQ(desc.custom_value, 77U);
}

TEST_F(Om2CodegenUt, TaskCodeBuilderUtil_ConvertAddrDesc_EventAddr) {
  AddrSemantic addr;
  addr.kind = AddrValueKind::kEventAddr;
  addr.event_id = 5U;

  auto desc = TaskCodeBuilderUtil::ConvertAddrDesc(addr);
  EXPECT_EQ(desc.type, OP_ARG_EVENT_ADDR);
  EXPECT_EQ(desc.custom_value, 5U);
}

TEST_F(Om2CodegenUt, TaskCodeBuilderUtil_ConvertAddrDesc_Placeholder) {
  AddrSemantic addr;
  addr.kind = AddrValueKind::kPlaceholder;

  auto desc = TaskCodeBuilderUtil::ConvertAddrDesc(addr);
  EXPECT_EQ(desc.type, OP_ARG_PLACEHOLDER);
}

TEST_F(Om2CodegenUt, TaskCodeBuilderUtil_ConvertAddrDesc_OptionalEmpty) {
  AddrSemantic addr;
  addr.kind = AddrValueKind::kOptionalEmpty;

  auto desc = TaskCodeBuilderUtil::ConvertAddrDesc(addr);
  EXPECT_EQ(desc.type, OP_ARG_OPTIONAL_EMPTY);
}

TEST_F(Om2CodegenUt, TaskCodeBuilderUtil_ConvertAddrDesc_EmptyAddr) {
  AddrSemantic addr;
  addr.kind = AddrValueKind::kEmptyAddr;

  auto desc = TaskCodeBuilderUtil::ConvertAddrDesc(addr);
  EXPECT_EQ(desc.type, OP_ARG_OPTIONAL_EMPTY);
}

TEST_F(Om2CodegenUt, TaskCodeBuilderUtil_ConvertAddrDesc_FftsAddr) {
  AddrSemantic addr;
  addr.kind = AddrValueKind::kFftsAddr;

  auto desc = TaskCodeBuilderUtil::ConvertAddrDesc(addr);
  EXPECT_EQ(desc.type, OP_ARG_FFTS_ADDR);
}

TEST_F(Om2CodegenUt, TaskCodeBuilderUtil_ConvertAddrDesc_OverflowAddr) {
  AddrSemantic addr;
  addr.kind = AddrValueKind::kOverflowAddr;

  auto desc = TaskCodeBuilderUtil::ConvertAddrDesc(addr);
  EXPECT_EQ(desc.type, OP_ARG_OVERFLOW_ADDR);
}

TEST_F(Om2CodegenUt, TaskCodeBuilderUtil_ConvertAddrDesc_Tiling) {
  AddrSemantic addr;
  addr.kind = AddrValueKind::kTiling;
  addr.byte_size = 32U;

  auto desc = TaskCodeBuilderUtil::ConvertAddrDesc(addr);
  EXPECT_EQ(desc.type, OP_ARG_TILING);
}

TEST_F(Om2CodegenUt, TaskCodeBuilderUtil_ConvertAddrDesc_SessionScopeMemory) {
  AddrSemantic addr;
  addr.kind = AddrValueKind::kInputInstance;
  addr.memory_type = kSessionScopeMemoryMask | RT_MEMORY_HBM;
  addr.mem_offset = 256;

  auto desc = TaskCodeBuilderUtil::ConvertAddrDesc(addr);
  EXPECT_EQ(desc.mem_src, MEM_SRC_SESSION);
  EXPECT_EQ(desc.offset, 256U);
}

TEST_F(Om2CodegenUt, TaskCodeBuilderUtil_ConvertAddrDesc_UnknownKind_DefaultsToRawAddr) {
  AddrSemantic addr;
  addr.kind = static_cast<AddrValueKind>(999);

  auto desc = TaskCodeBuilderUtil::ConvertAddrDesc(addr);
  EXPECT_EQ(desc.type, OP_ARG_RAW_ADDR);
}

TEST_F(Om2CodegenUt, TaskCodeBuilderUtil_ConvertAddrDesc_Level1DescPtrNoOffset) {
  AddrSemantic addr;
  addr.kind = AddrValueKind::kLevel1DescPtr;

  auto desc = TaskCodeBuilderUtil::ConvertAddrDesc(addr);
  EXPECT_EQ(desc.type, OP_ARG_LEVEL1_DESC);
  EXPECT_EQ(desc.custom_value, 0U);
}

TEST_F(Om2CodegenUt, TaskCodeBuilderUtil_ConvertAddrDesc_NoConstIndex) {
  AddrSemantic addr;
  addr.kind = AddrValueKind::kInputInstance;
  addr.tensor_info = Om2TensorInfo{};
  addr.mem_offset = 0;

  auto desc = TaskCodeBuilderUtil::ConvertAddrDesc(addr);
  EXPECT_EQ(desc.type, OP_ARG_INPUT);
  EXPECT_EQ(desc.mem_src, 0U);
}

TEST_F(Om2CodegenUt, CppEmitter_ProtectedAccessAndConstCast) {
  AstContext ctx;
  AstBuildContext ast(ctx);

  auto *protected_decl = AccessSectionDecl::Create(ctx, AccessSectionDecl::Kind::kProtected);
  ASSERT_NE(protected_decl, nullptr);
  auto *class_decl =
      ClassDecl::Create(ctx, "ProtectedClass", {protected_decl, FieldDecl::Create(ctx, "int", "hidden")});
  ASSERT_NE(class_decl, nullptr);

  auto *ident_x = IdentifierExpr::Create(ctx, "x");
  auto *const_cast_expr = CppCastExpr::Create(ctx, CppCastExpr::Kind::kConst, "int &", ident_x);
  ASSERT_NE(const_cast_expr, nullptr);

  auto *tu = TranslationUnit::Create(ctx, {class_decl});
  ASSERT_NE(tu, nullptr);
  const auto class_output = EmitNode(*tu);
  const auto cast_output = EmitNode(*const_cast_expr);
  const std::string output = class_output + "\n" + cast_output;
  ExpectContainsAll(output, {"protected:", "const_cast<int &>(x)"});
}

TEST_F(Om2CodegenUt, CppEmitter_AllBuiltinTypes) {
  AstContext ctx;
  AstBuildContext ast(ctx);

  auto count = ast.Var("size_t", "count");
  const std::vector<std::pair<BuiltinType, std::string>> type_pairs = {
      {BuiltinType::kVoid, "void"},       {BuiltinType::kBool, "bool"},       {BuiltinType::kChar, "char"},
      {BuiltinType::kInt8, "int8_t"},     {BuiltinType::kUInt8, "uint8_t"},   {BuiltinType::kInt16, "int16_t"},
      {BuiltinType::kUInt16, "uint16_t"}, {BuiltinType::kInt32, "int32_t"},   {BuiltinType::kUInt32, "uint32_t"},
      {BuiltinType::kInt64, "int64_t"},   {BuiltinType::kUInt64, "uint64_t"}, {BuiltinType::kFloat, "float"},
      {BuiltinType::kDouble, "double"},
  };

  std::vector<Stmt *> body;
  for (const auto &pair : type_pairs) {
    body.push_back(ast.VarDecl("auto", "buf_" + pair.second, ast.MakeUniqueArray(pair.first, count)));
  }
  body.push_back(ast.Return());

  auto *fn = ast.DefineFunction("TestAllBuiltinTypes", {count}, "void", body);
  ASSERT_NE(fn, nullptr);

  const auto output = EmitNode(*fn);
  for (const auto &pair : type_pairs) {
    EXPECT_NE(output.find("std::make_unique<" + pair.second + "[]>(count)"), std::string::npos)
        << "Missing builtin type: " << pair.second;
  }
}

TEST_F(Om2CodegenUt, CppEmitter_IntSuffixL) {
  AstContext ctx;

  auto *lit_l = LiteralExpr::CreateInt(ctx, 42, LiteralExpr::IntSuffix::kL);
  ASSERT_NE(lit_l, nullptr);
  EXPECT_EQ(lit_l->GetIntSuffix(), LiteralExpr::IntSuffix::kL);

  auto *var_decl = VarDeclStmt::Create(ctx, "long", "val", lit_l);
  ASSERT_NE(var_decl, nullptr);

  const auto output = EmitNode(*var_decl);
  EXPECT_NE(output.find("42L"), std::string::npos);
}

TEST_F(Om2CodegenUt, CppEmitter_ForLoopWithExprStmtInit) {
  AstContext ctx;
  AstBuildContext ast(ctx);

  auto i = ast.Var("size_t", "i");
  auto assign_init = ast.Assign(i, 0);
  auto *for_stmt = ast.For(ExprStmt::Create(ctx, assign_init.Get()), i < 10, ast.PreInc(i), {ast.Assign(i, i + 1)});

  auto *fn = ast.DefineFunction("TestForWithExprInit", std::vector<VarRef>{}, "void",
                                std::vector<Stmt *>{for_stmt, ast.Return()});
  ASSERT_NE(fn, nullptr);

  const auto output = EmitNode(*fn);
  EXPECT_NE(output.find("for (i = 0;"), std::string::npos);
}

TEST_F(Om2CodegenUt, CppEmitter_EmptyTypeNameSeparator) {
  AstContext ctx;
  AstBuildContext ast(ctx);

  auto *type_alias = TypeAliasDecl::Create(ctx, "", "EmptyTypeAlias");
  ASSERT_NE(type_alias, nullptr);
  auto *tu = TranslationUnit::Create(ctx, {type_alias});
  ASSERT_NE(tu, nullptr);
  const auto output = EmitNode(*tu);
  EXPECT_NE(output.find("EmptyTypeAlias"), std::string::npos);
}

TEST_F(Om2CodegenUt, CppEmitter_SwitchCaseBodyIndentation) {
  AstContext ctx;
  AstBuildContext ast(ctx);

  const auto *switch_stmt = ast.Switch("value", {
                                                    ast.Case("CASE_A"),
                                                    ast.Return("result_a"),
                                                    ast.Case("CASE_B"),
                                                    ast.Return("result_b"),
                                                    ast.Case(nullptr),
                                                    ast.Return("default_result"),
                                                });

  EXPECT_EQ(EmitNode(*switch_stmt),
            "switch (value) {\n"
            "  case CASE_A:\n"
            "    return result_a;\n"
            "  case CASE_B:\n"
            "    return result_b;\n"
            "  default:\n"
            "    return default_result;\n"
            "}\n");
}

TEST_F(Om2CodegenUt, CppEmitter_NestedSwitchCaseBodyIndentation) {
  AstContext ctx;
  AstBuildContext ast(ctx);

  const auto *switch_stmt = ast.Switch("outer", {
                                                    ast.Case("OUTER_CASE"),
                                                    ast.Switch("inner",
                                                               {
                                                                   ast.Case("INNER_CASE"),
                                                                   ast.Return("inner_result"),
                                                               }),
                                                    ast.Case(nullptr),
                                                    ast.Return("outer_default_result"),
                                                });

  EXPECT_EQ(EmitNode(*switch_stmt),
            "switch (outer) {\n"
            "  case OUTER_CASE:\n"
            "    switch (inner) {\n"
            "      case INNER_CASE:\n"
            "        return inner_result;\n"
            "    }\n"
            "  default:\n"
            "    return outer_default_result;\n"
            "}\n");
}

class RTVarResourceCoverageTest : public testing::Test {
 protected:
  gert::RTVarEntry MakeEntry(const std::string &var_name, int format, int dtype) {
    gert::RTVarEntry entry;
    entry.var_name = var_name;
    ge::Om2TensorDesc desc;
    desc.SetFormat(static_cast<ge::Format>(format));
    desc.SetDataType(static_cast<ge::DataType>(dtype));
    entry.var_key = gert::RTVarResource::BuildVarKey(var_name, desc);
    entry.tensor_desc = desc;
    return entry;
  }
};

TEST_F(RTVarResourceCoverageTest, GetEntryFound) {
  gert::RTVarResource resource;
  auto entry = MakeEntry("weight1", 1, 0);
  ASSERT_EQ(resource.AddEntry(std::move(entry)), ge::SUCCESS);
  const auto *result = resource.GetEntry("weight11_0");
  ASSERT_NE(result, nullptr);
  EXPECT_EQ(result->var_name, "weight1");
}

TEST_F(RTVarResourceCoverageTest, GetEntryNotFound) {
  gert::RTVarResource resource;
  EXPECT_EQ(resource.GetEntry("nonexistent"), nullptr);
}

TEST_F(RTVarResourceCoverageTest, GetEntryByNameFound) {
  gert::RTVarResource resource;
  auto entry = MakeEntry("weight1", 1, 0);
  ASSERT_EQ(resource.AddEntry(std::move(entry)), ge::SUCCESS);
  const auto *result = resource.GetEntryByName("weight1");
  ASSERT_NE(result, nullptr);
  EXPECT_EQ(result->var_key, "weight11_0");
}

TEST_F(RTVarResourceCoverageTest, GetEntryByNameNotFound) {
  gert::RTVarResource resource;
  EXPECT_EQ(resource.GetEntryByName("nonexistent"), nullptr);
}

TEST_F(RTVarResourceCoverageTest, GetAllVarKeys) {
  gert::RTVarResource resource;
  ASSERT_EQ(resource.AddEntry(MakeEntry("a", 1, 0)), ge::SUCCESS);
  ASSERT_EQ(resource.AddEntry(MakeEntry("b", 1, 0)), ge::SUCCESS);
  auto keys = resource.GetAllVarKeys();
  EXPECT_EQ(keys.size(), 2U);
}

TEST_F(RTVarResourceCoverageTest, AddEntryEmptyKeyFails) {
  gert::RTVarResource resource;
  gert::RTVarEntry entry;
  entry.var_key = "";
  EXPECT_NE(resource.AddEntry(std::move(entry)), ge::SUCCESS);
}

TEST_F(Om2CodegenUt, CppEmitter_InvalidEnumDefaults) {
  AstContext ctx;
  auto *ident_x = IdentifierExpr::Create(ctx, "x");
  auto *ident_y = IdentifierExpr::Create(ctx, "y");
  auto *ident_vec = IdentifierExpr::Create(ctx, "vec");
  auto *count = IdentifierExpr::Create(ctx, "count");

  CppEmitter emitter;
  std::string output;

  AccessSectionDecl invalid_access(static_cast<AccessSectionDecl::Kind>(99));
  output.clear();
  EXPECT_EQ(invalid_access.Accept(emitter, output), SUCCESS);

  CppCastExpr invalid_cast_kind(static_cast<CppCastExpr::Kind>(99), StringRef("int"), ident_x);
  output.clear();
  EXPECT_EQ(invalid_cast_kind.Accept(emitter, output), SUCCESS);

  BinaryExpr invalid_binary_op(static_cast<BinaryExpr::Op>(99), ident_x, ident_y);
  output.clear();
  EXPECT_EQ(invalid_binary_op.Accept(emitter, output), SUCCESS);

  UnaryExpr invalid_unary_op(static_cast<UnaryExpr::Op>(99), ident_x);
  output.clear();
  EXPECT_EQ(invalid_unary_op.Accept(emitter, output), SUCCESS);

  ContainerMethodExpr invalid_container_method(static_cast<ContainerMethodExpr::Method>(99), ident_vec,
                                               ArrayRef<Expr *>());
  output.clear();
  EXPECT_EQ(invalid_container_method.Accept(emitter, output), SUCCESS);

  MakeUniqueArrayExpr invalid_builtin_type(static_cast<BuiltinType>(99), count);
  output.clear();
  EXPECT_EQ(invalid_builtin_type.Accept(emitter, output), SUCCESS);

  LiteralExpr invalid_literal_kind(static_cast<LiteralExpr::Kind>(99), 0, LiteralExpr::IntSuffix::kNone, false,
                                   StringRef());
  output.clear();
  EXPECT_EQ(invalid_literal_kind.Accept(emitter, output), FAILED);

  LiteralExpr invalid_int_suffix(LiteralExpr::Kind::kInt, 42, static_cast<LiteralExpr::IntSuffix>(99), false,
                                 StringRef());
  output.clear();
  EXPECT_EQ(invalid_int_suffix.Accept(emitter, output), SUCCESS);
}

TEST_F(Om2CodegenUt, CppEmitter_ForInitEdgeCases) {
  AstContext ctx;
  AstBuildContext ast(ctx);

  auto i = ast.Var("size_t", "i");

  auto *for_no_init = ast.For(VarDeclStmt::Create(ctx, "size_t", "j"), i < 4, ast.PreInc(i), {ast.Assign(i, 0)});
  ASSERT_NE(for_no_init, nullptr);
  const auto output1 = EmitNode(*for_no_init);
  EXPECT_NE(output1.find("for (size_t j;"), std::string::npos);

  auto *for_comment_init = ast.For(CommentStmt::Create(ctx, "init"), i < 4, ast.PreInc(i), {ast.Assign(i, 0)});
  ASSERT_NE(for_comment_init, nullptr);
  CppEmitter emitter;
  std::string output2;
  EXPECT_EQ(for_comment_init->Accept(emitter, output2), FAILED);
}

TEST_F(Om2CodegenUt, CppEmitter_ErrorPropagationInExpressions) {
  AstContext ctx;
  auto *ident_x = IdentifierExpr::Create(ctx, "x");
  auto *ident_y = IdentifierExpr::Create(ctx, "y");
  auto *ident_vec = IdentifierExpr::Create(ctx, "vec");
  auto *ident_fn = IdentifierExpr::Create(ctx, "fn");
  auto *count = IdentifierExpr::Create(ctx, "count");
  auto *path = IdentifierExpr::Create(ctx, "path");

  LiteralExpr failing(static_cast<LiteralExpr::Kind>(99), 0, LiteralExpr::IntSuffix::kNone, false, StringRef());

  CppEmitter emitter;
  std::string output;

  AssignExpr assign_failing_lhs(&failing, ident_y);
  output.clear();
  EXPECT_EQ(assign_failing_lhs.Accept(emitter, output), FAILED);

  BinaryExpr binary_failing_lhs(BinaryExpr::Op::kAdd, &failing, ident_y);
  output.clear();
  EXPECT_EQ(binary_failing_lhs.Accept(emitter, output), FAILED);

  BinaryExpr binary_failing_rhs(BinaryExpr::Op::kAdd, ident_x, &failing);
  output.clear();
  EXPECT_EQ(binary_failing_rhs.Accept(emitter, output), FAILED);

  UnaryExpr unary_failing(UnaryExpr::Op::kNegate, &failing);
  output.clear();
  EXPECT_EQ(unary_failing.Accept(emitter, output), FAILED);

  auto *call_failing_callee = CallExpr::Create(ctx, &failing, {ident_x});
  output.clear();
  EXPECT_EQ(call_failing_callee->Accept(emitter, output), FAILED);

  auto *call_failing_arg = CallExpr::Create(ctx, ident_fn, {ident_x, &failing});
  output.clear();
  EXPECT_EQ(call_failing_arg->Accept(emitter, output), FAILED);

  MakeUniqueArrayExpr make_unique_failing(BuiltinType::kUInt8, &failing);
  output.clear();
  EXPECT_EQ(make_unique_failing.Accept(emitter, output), FAILED);

  ToStrExpr to_str_failing(&failing);
  output.clear();
  EXPECT_EQ(to_str_failing.Accept(emitter, output), FAILED);

  MemcpyExpr memcpy_failing_dst(&failing, ident_x, ident_y);
  output.clear();
  EXPECT_EQ(memcpy_failing_dst.Accept(emitter, output), FAILED);

  MemcpyExpr memcpy_failing_src(ident_x, &failing, ident_y);
  output.clear();
  EXPECT_EQ(memcpy_failing_src.Accept(emitter, output), FAILED);

  MemcpyExpr memcpy_failing_size(ident_x, ident_y, &failing);
  output.clear();
  EXPECT_EQ(memcpy_failing_size.Accept(emitter, output), FAILED);

  SizeofExpr sizeof_failing(&failing);
  output.clear();
  EXPECT_EQ(sizeof_failing.Accept(emitter, output), FAILED);

  RemoveFileExpr remove_file_failing(&failing);
  output.clear();
  EXPECT_EQ(remove_file_failing.Accept(emitter, output), FAILED);

  IgnoreOutputExpr ignore_output_failing(&failing);
  output.clear();
  EXPECT_EQ(ignore_output_failing.Accept(emitter, output), FAILED);

  std::vector<Expr *> container_args = {ident_x, &failing};
  ContainerMethodExpr container_failing_arg(ContainerMethodExpr::Method::kAt, ident_vec,
                                            ArrayRef<Expr *>(container_args.data(), container_args.size()));
  output.clear();
  EXPECT_EQ(container_failing_arg.Accept(emitter, output), FAILED);

  SubscriptExpr subscript_failing_base(&failing, ident_x);
  output.clear();
  EXPECT_EQ(subscript_failing_base.Accept(emitter, output), FAILED);

  SubscriptExpr subscript_failing_index(ident_vec, &failing);
  output.clear();
  EXPECT_EQ(subscript_failing_index.Accept(emitter, output), FAILED);

  MemberExpr member_failing(&failing, StringRef("field"));
  output.clear();
  EXPECT_EQ(member_failing.Accept(emitter, output), FAILED);

  CppArrowMemberExpr arrow_failing(&failing, StringRef("field"));
  output.clear();
  EXPECT_EQ(arrow_failing.Accept(emitter, output), FAILED);

  CppCastExpr cast_failing(CppCastExpr::Kind::kStatic, StringRef("int"), &failing);
  output.clear();
  EXPECT_EQ(cast_failing.Accept(emitter, output), FAILED);

  std::vector<Expr *> init_list_elems = {ident_x, &failing};
  InitListExpr compact_init_failing(ArrayRef<Expr *>(init_list_elems.data(), init_list_elems.size()), true);
  output.clear();
  EXPECT_EQ(compact_init_failing.Accept(emitter, output), FAILED);

  InitListExpr noncompact_init_failing(ArrayRef<Expr *>(init_list_elems.data(), init_list_elems.size()), false);
  output.clear();
  EXPECT_EQ(noncompact_init_failing.Accept(emitter, output), FAILED);

  std::vector<StringRef> desig_names = {StringRef("a"), StringRef("b")};
  std::vector<Expr *> desig_values = {ident_x, &failing};
  DesignatedInitListExpr compact_desig_failing(ArrayRef<StringRef>(desig_names.data(), desig_names.size()),
                                               ArrayRef<Expr *>(desig_values.data(), desig_values.size()), true);
  output.clear();
  EXPECT_EQ(compact_desig_failing.Accept(emitter, output), FAILED);

  DesignatedInitListExpr noncompact_desig_failing(ArrayRef<StringRef>(desig_names.data(), desig_names.size()),
                                                  ArrayRef<Expr *>(desig_values.data(), desig_values.size()), false);
  output.clear();
  EXPECT_EQ(noncompact_desig_failing.Accept(emitter, output), FAILED);
}

TEST_F(Om2CodegenUt, CppEmitter_ErrorPropagationInStatements) {
  AstContext ctx;
  AstBuildContext ast(ctx);

  auto *ident_x = IdentifierExpr::Create(ctx, "x");
  auto *ident_y = IdentifierExpr::Create(ctx, "y");

  LiteralExpr failing(static_cast<LiteralExpr::Kind>(99), 0, LiteralExpr::IntSuffix::kNone, false, StringRef());

  CppEmitter emitter;
  std::string output;

  VarDeclStmt var_decl_failing(StringRef("int"), StringRef("v"), &failing);
  output.clear();
  EXPECT_EQ(var_decl_failing.Accept(emitter, output), FAILED);

  ExprStmt expr_stmt_failing(&failing);
  output.clear();
  EXPECT_EQ(expr_stmt_failing.Accept(emitter, output), FAILED);

  ReturnStmt return_failing(&failing);
  output.clear();
  EXPECT_EQ(return_failing.Accept(emitter, output), FAILED);

  std::vector<Stmt *> block_stmts = {ExprStmt::Create(ctx, ident_x), ExprStmt::Create(ctx, &failing)};
  BlockStmt block_failing(ArrayRef<Stmt *>(block_stmts.data(), block_stmts.size()));
  output.clear();
  EXPECT_EQ(block_failing.Accept(emitter, output), FAILED);

  auto *then_block = BlockStmt::Create(ctx, {ExprStmt::Create(ctx, ident_x)});
  auto *else_block = BlockStmt::Create(ctx, {ExprStmt::Create(ctx, ident_y)});
  IfStmt if_failing_cond(&failing, then_block, else_block);
  output.clear();
  EXPECT_EQ(if_failing_cond.Accept(emitter, output), FAILED);

  std::vector<Stmt *> failing_then_stmts = {ExprStmt::Create(ctx, &failing)};
  BlockStmt failing_then_block(ArrayRef<Stmt *>(failing_then_stmts.data(), failing_then_stmts.size()));
  IfStmt if_failing_then(ident_x, &failing_then_block, else_block);
  output.clear();
  EXPECT_EQ(if_failing_then.Accept(emitter, output), FAILED);

  IfStmt if_failing_else(ident_x, then_block, &failing_then_block);
  output.clear();
  EXPECT_EQ(if_failing_else.Accept(emitter, output), FAILED);

  IfStmt pp_if_failing_cond(&failing, then_block, else_block, true);
  output.clear();
  EXPECT_EQ(pp_if_failing_cond.Accept(emitter, output), FAILED);

  IfStmt pp_if_failing_then(ident_x, &failing_then_block, else_block, true);
  output.clear();
  EXPECT_EQ(pp_if_failing_then.Accept(emitter, output), FAILED);

  IfStmt pp_if_failing_else(ident_x, then_block, &failing_then_block, true);
  output.clear();
  EXPECT_EQ(pp_if_failing_else.Accept(emitter, output), FAILED);

  auto i = ast.Var("size_t", "i");
  auto *for_failing_init =
      ast.For(VarDeclStmt::Create(ctx, "int", "v", &failing), i < 4, ast.PreInc(i), {ast.Assign(i, 0)});
  output.clear();
  EXPECT_EQ(for_failing_init->Accept(emitter, output), FAILED);

  auto *for_failing_cond = ast.For(ast.VarDecl(i, 0), &failing, ast.PreInc(i), {ast.Assign(i, 0)});
  output.clear();
  EXPECT_EQ(for_failing_cond->Accept(emitter, output), FAILED);

  auto *for_failing_step = ast.For(ast.VarDecl(i, 0), i < 4, &failing, {ast.Assign(i, 0)});
  output.clear();
  EXPECT_EQ(for_failing_step->Accept(emitter, output), FAILED);

  auto *for_failing_body =
      ast.For(ast.VarDecl(i, 0), i < 4, ast.PreInc(i), {BodyItem(ExprStmt::Create(ctx, &failing))});
  output.clear();
  EXPECT_EQ(for_failing_body->Accept(emitter, output), FAILED);

  auto values = ast.Var("std::vector<int>", "values");
  auto range_body_stmts = ast.Body({BodyItem(ast.Assign(i, 0))});
  RangeForStmt range_for_failing_range(StringRef("auto"), StringRef("item"), &failing,
                                       BlockStmt::Create(ctx, range_body_stmts));
  output.clear();
  EXPECT_EQ(range_for_failing_range.Accept(emitter, output), FAILED);

  RangeForStmt range_for_failing_body(StringRef("auto"), StringRef("item"), values.Get(), &failing_then_block);
  output.clear();
  EXPECT_EQ(range_for_failing_body.Accept(emitter, output), FAILED);
}

TEST_F(Om2CodegenUt, CppEmitter_ErrorPropagationInDeclarations) {
  AstContext ctx;
  AstBuildContext ast(ctx);

  auto *ident_x = IdentifierExpr::Create(ctx, "x");

  LiteralExpr failing(static_cast<LiteralExpr::Kind>(99), 0, LiteralExpr::IntSuffix::kNone, false, StringRef());

  CppEmitter emitter;
  std::string output;

  FieldDecl field_failing(StringRef("int"), StringRef("v"), &failing);
  output.clear();
  EXPECT_EQ(field_failing.Accept(emitter, output), FAILED);

  std::vector<DeclNode *> class_items = {AccessSectionDecl::Create(ctx, AccessSectionDecl::Kind::kPublic),
                                         &field_failing};
  ClassDecl class_failing(StringRef("Cls"), ArrayRef<DeclNode *>(class_items.data(), class_items.size()));
  output.clear();
  EXPECT_EQ(class_failing.Accept(emitter, output), FAILED);

  StructDecl struct_failing(StringRef("S"), ArrayRef<DeclNode *>(class_items.data(), class_items.size()));
  output.clear();
  EXPECT_EQ(struct_failing.Accept(emitter, output), FAILED);

  std::vector<DeclNode *> ns_items = {&field_failing};
  NamespaceDecl ns_failing(StringRef("ns"), ArrayRef<DeclNode *>(ns_items.data(), ns_items.size()));
  output.clear();
  EXPECT_EQ(ns_failing.Accept(emitter, output), FAILED);

  ExternBlockDecl extern_failing(StringRef("C"), ArrayRef<DeclNode *>(ns_items.data(), ns_items.size()));
  output.clear();
  EXPECT_EQ(extern_failing.Accept(emitter, output), FAILED);

  auto *param_x = ParamDecl::Create(ctx, "int", "x");
  auto *body = BlockStmt::Create(ctx, {ReturnStmt::Create(ctx, ident_x)});
  std::vector<ParamDecl *> method_params = {param_x};
  std::vector<StringRef> method_init_names = {StringRef("val_")};
  std::vector<Expr *> method_init_exprs = {&failing};
  MethodDef method_def_failing_init(StringRef("Worker"), StringRef("Worker"),
                                    ArrayRef<ParamDecl *>(method_params.data(), method_params.size()), StringRef(""),
                                    ArrayRef<StringRef>(method_init_names.data(), method_init_names.size()),
                                    ArrayRef<Expr *>(method_init_exprs.data(), method_init_exprs.size()), body);
  output.clear();
  EXPECT_EQ(method_def_failing_init.Accept(emitter, output), FAILED);
}

TEST_F(Om2CodegenUt, CppEmitter_ContainerMethodMultiArgs) {
  AstContext ctx;

  auto *ident_vec = IdentifierExpr::Create(ctx, "vec");
  auto *ident_x = IdentifierExpr::Create(ctx, "x");
  auto *ident_y = IdentifierExpr::Create(ctx, "y");

  std::vector<Expr *> multi_args = {ident_x, ident_y};
  ContainerMethodExpr multi_arg_method(ContainerMethodExpr::Method::kAt, ident_vec,
                                       ArrayRef<Expr *>(multi_args.data(), multi_args.size()));

  CppEmitter emitter;
  std::string output;
  EXPECT_EQ(multi_arg_method.Accept(emitter, output), SUCCESS);
  EXPECT_NE(output.find(", "), std::string::npos);
}

TEST_F(Om2CodegenUt, ArgsManagerFileCodeGenerator_CopyArgsToDevice_CoverBranches) {
  AstContext ctx;
  AstBuildContext ast(ctx);
  ArgsManagerFileCodeGenerator generator(ast);

  Om2CodegenModel with_va2pa_model;
  with_va2pa_model.is_need_va2pa = true;
  auto *with_va2pa_method = generator.BuildCopyArgsToDeviceMethod(with_va2pa_model);
  ASSERT_NE(with_va2pa_method, nullptr);
  const auto with_va2pa_output = EmitNode(*with_va2pa_method);
  ExpectContainsAll(with_va2pa_output,
                    {"aclError Om2ArgsTable::CopyArgsToDevice(void *stream, bool is_async) {\n",
                     "OM2_CHK_STATUS(rtDevVA2PA((uint64_t)dev_args_[0], args_sizes_[0], stream, is_async));\n"});

  Om2CodegenModel without_va2pa_model;
  without_va2pa_model.is_need_va2pa = false;
  auto *without_va2pa_method = generator.BuildCopyArgsToDeviceMethod(without_va2pa_model);
  ASSERT_NE(without_va2pa_method, nullptr);
  const auto without_va2pa_output = EmitNode(*without_va2pa_method);
  ExpectContainsAll(without_va2pa_output, {"aclError Om2ArgsTable::CopyArgsToDevice(void *stream, bool is_async) {\n",
                                           "(void)stream;\n", "(void)is_async;\n",
                                           "OM2_CHK_STATUS(aclrtMemcpy(dev_args_[0], args_sizes_[0], "
                                           "host_args_[0].data(), args_sizes_[0], ACL_MEMCPY_HOST_TO_DEVICE));\n"});
}

TEST_F(Om2CodegenUt, LoadAndRunFileCodeGenerator_PhaseModelExecute_CoverBranches) {
  AstContext ctx;
  AstBuildContext ast(ctx);
  LoadAndRunFileCodeGenerator generator(ast);

  auto exe_stream = ast.Var("aclrtStream &", "exe_stream");
  auto run_callbacks = ast.Var("const GertModelRunCallbacks *", "run_callbacks");

  std::vector<BodyItem> async_body;
  generator.BuildRunBodyPhaseModelExecute(async_body, exe_stream, true, run_callbacks, true);
  EXPECT_FALSE(async_body.empty());
  const auto async_output = EmitBodyItems(ast, async_body);
  ExpectContainsAll(async_output, {"OM2_CHK_STATUS(args_table_.CopyArgsToDevice(exe_stream, false));\n",
                                   "OM2_CHK_STATUS(aclmdlRIExecuteAsync(model_handle_, exe_stream));\n",
                                   "report_run_info_preprocess", "report_run_info_postprocess"});

  std::vector<BodyItem> sync_body;
  generator.BuildRunBodyPhaseModelExecute(sync_body, exe_stream, false, run_callbacks, false);
  EXPECT_FALSE(sync_body.empty());
  const auto sync_output = EmitBodyItems(ast, sync_body);
  ExpectContainsAll(sync_output, {"OM2_CHK_STATUS(args_table_.CopyArgsToDevice(nullptr, false));\n",
                                  "OM2_CHK_STATUS(aclmdlRIExecute(model_handle_, stream_sync_timeout));\n"});
}
}  // namespace ge
