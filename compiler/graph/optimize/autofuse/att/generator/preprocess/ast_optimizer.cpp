/**
 * Copyright (C) Huawei Technologies Co., Ltd. 2025 All rights reserved.
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
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "ast_optimizer.h"

namespace att {
// 表达式涉及的函数
const vector<string> functions_set = {"Ceiling", "Min", "Max", "Rational", "Floor", "Log", "Pow", "Mod"};

// 判断字符是否是数字或相关符号
bool IsNumberChar(char c) {
  return isdigit(c) || c == '.' || c == '/';
}

// 处理负数
void HandleNegativeNumber(const string &s, size_t &i, vector<string> &tokens) {
  string num;
  num += s[i++];
  while (i < s.size() && IsNumberChar(s[i])) {
    num += s[i++];
  }
  tokens.push_back(num);
}

// 处理非负数字
void HandleNumber(const string &s, size_t &i, vector<string> &tokens) {
  string num;
  while (i < s.size() && IsNumberChar(s[i])) {
    num += s[i++];
  }
  tokens.push_back(num);
}

// 处理变量
void HandleIdentifier(const string &s, size_t &i, vector<string> &tokens) {
  string token;
  while (i < s.size() && (isalnum(s[i]) || s[i] == '_')) {
    token += s[i++];
  }
  tokens.push_back(token);
}

// 词法分析器
vector<string> Parser::Tokenize(const string &s) {
  vector<string> tokens;
  for (size_t i = 0; i < s.size();) {
    if (isspace(s[i])) {
      ++i;
      continue;
    }
    // 处理负数
    if ((s[i] == '-') && ((i == 0u) || tokens.empty() || (tokens.back() == "(") || (tokens.back() == ",")
              || (string("+-*/(").find(tokens.back()[0]) != string::npos))) {
      HandleNegativeNumber(s, i, tokens);
    }
    // 处理非负数字
    else if (IsNumberChar(s[i])) {
      HandleNumber(s, i, tokens);
    }
    // 处理变量
    else if (isalpha(s[i]) || s[i] == '_') {
      HandleIdentifier(s, i, tokens);
    } else {
      tokens.push_back(string(1, s[i++]));
    }
  }
  return tokens;
}

ASTPtr Parser::ParseFunction(const string &func) {
  // consume两次，第一次是函数名，第二次是(
  Consume();
  Consume();
  vector<ASTPtr> args;
  while (Peek() != ")") {
    args.push_back(ParseExpr());
    if (Peek() == ",") {
      Consume();
    }
  }
  Consume();
  return make_shared<ASTNode>(func, NodeType::FUNCTION, func, move(args));
}

ASTPtr Parser::ParsePrimary() {
  string token = Peek();
  if (token == "(") {
    Consume();
    auto node = ParseExpr();
    if (Peek() != ")") {
      GELOGD("error: expected ')', got '%s'", Peek().c_str());
      return nullptr;
    }
    Consume();
    return node;
  }
  if (std::find(functions_set.begin(), functions_set.end(), token) != functions_set.end()) {
    return ParseFunction(token);
  }
  if ((token[0] == '-' && token.size() > 1u && (isdigit(token[1]) || token[1] == '.')) || isdigit(token[0]) ||
      token.find('.') != string::npos || token.find('/') != string::npos) {
    Consume();
    return make_shared<ASTNode>(token, NodeType::NUMBER);
  }
  if (isdigit(token[0]) || token.find('.') != string::npos || token.find('/') != string::npos) {
    Consume();
    return make_shared<ASTNode>(token, NodeType::NUMBER);
  }
  if (isalpha(token[0])) {
    Consume();
    return make_shared<ASTNode>(token, NodeType::VARIABLE);
  }
  GELOGD("error: invalid expression: '%s'", token.c_str());
  return nullptr;
}

ASTPtr CreateBinaryOpNode(ASTPtr &&lhs, const string &op, ASTPtr &&rhs) {
  vector<ASTPtr> children;
  children.push_back(move(lhs));
  children.push_back(move(rhs));
  return make_shared<ASTNode>("", NodeType::OPERATOR, op, move(children));
}

ASTPtr Parser::ParseExpr() {
  ASTPtr lhs = ParseTerm();
  if (!lhs) {
    return nullptr;
  }
  // 处理不带括号的连加连减
  while ((Peek() == "+") || (Peek() == "-")) {
    string op = Peek();
    Consume();
    ASTPtr rhs = ParseTerm();
    if (!rhs) {
      return nullptr;
    }
    lhs = CreateBinaryOpNode(move(lhs), op, move(rhs));
  }
  return lhs;
}

ASTPtr Parser::ParseTerm() {
  ASTPtr lhs = ParsePrimary();
  if (!lhs) {
    return nullptr;
  }
  // 处理不带括号的连乘连除
  while ((Peek() == "*") || (Peek() == "/")) {
    string op = Peek();
    Consume();
    ASTPtr rhs = ParsePrimary();
    if (!rhs) {
      return nullptr;
    }
    lhs = CreateBinaryOpNode(move(lhs), op, move(rhs));
  }
  return lhs;
}

ASTPtr Parser::Parse() {
  tokens_ = Tokenize(expr_);
  GELOGD("tokenize success, tokens are: ");
  for (auto &token : tokens_) {
    GELOGD("%s", token.c_str());
  }
  return ParseExpr();
}

// 处理操作符或函数节点
void ProcessOperatorOrFunction(ASTNode *node, unordered_map<string, string> &expr_map_, vector<ASTNode> &temp_order_,
                               int32_t &temp_count_) {
  auto it = expr_map_.find(node->hash);
  if (it != expr_map_.end()) {
    node->temp_var = it->second;  // 复用已有变量名
  } else {
    // 分配新变量名并记录
    node->temp_var = "temp" + to_string(temp_count_++);
    expr_map_[node->hash] = node->temp_var;
    temp_order_.push_back(*node);
  }
}

void Optimizer::Traverse(ASTNode *node) {
  if (!node) {
    return;
  }

  // 先递归处理所有子节点
  for (auto &c : node->children) {
    Traverse(c.get());
  }

  // 处理操作符或函数节点
  if (node->type == NodeType::OPERATOR || node->type == NodeType::FUNCTION) {
    ProcessOperatorOrFunction(node, expr_map_, temp_order_, temp_count_);
  }
}

string RebuildFunctionCall(const ASTNode &node, int iter, function<string(const ASTNode &, int)> rebuild_expr) {
  stringstream ss;
  ss << node.op << "(";
  for (size_t i = 0; i < node.children.size(); ++i) {
    if (i > 0u) {
      ss << ",";
    }
    ss << rebuild_expr(*node.children[i].get(), iter + 1);
  }
  ss << ")";
  return ss.str();
}

string RebuildBinaryOperation(const ASTNode &node, int iter, function<string(const ASTNode &, int)> rebuild_expr) {
  if (node.children.size() != 2u) {
    return node.expr;
  }
  return "(" + rebuild_expr(*node.children[0].get(), iter + 1) + " " + node.op + " " +
              rebuild_expr(*node.children[1].get(), iter + 1) + ")";
}

string Optimizer::RebuildExpr(const ASTNode &node, int iter) {
  // 复用已有变量名
  if (!node.temp_var.empty() && (iter != 0)) {
    return node.temp_var;
  }
  auto rebuild_expr = [this](const ASTNode &n, int i) { 
    return this->RebuildExpr(n, i); 
  };
  switch (node.type) {
    case NodeType::FUNCTION:
      return RebuildFunctionCall(node, iter, rebuild_expr);
    case NodeType::OPERATOR:
      return RebuildBinaryOperation(node, iter, rebuild_expr);
    default:
      return node.expr;
  }
}
string Optimizer::GenerateCode() {
  stringstream ss;
  if (temp_order_.empty()) {
    return "";
  }
  for (const auto &node : temp_order_) {
    // 跳过已生成的节点
    if (visited_.find(node.hash) != visited_.end()) {
      continue;
    }
    ss << "    auto " << node.temp_var << " = " << RebuildExpr(node, 0) << ";\n";
    visited_.insert(node.hash);
  }
  return ss.str();
}

void Optimizer::Optimize(ASTPtr &root) {
  if (!root) {
    return;
  }
  Traverse(root.get());
}
}  // namespace att
