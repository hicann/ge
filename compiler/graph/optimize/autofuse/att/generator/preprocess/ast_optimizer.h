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

#ifndef ATT_CODE_GEN_PREPROCESS_AST_OPTIMIZER_H_
#define ATT_CODE_GEN_PREPROCESS_AST_OPTIMIZER_H_
#include <iostream>
#include <vector>
#include <unordered_map>
#include <memory>
#include <sstream>
#include <cmath>
#include <cctype>
#include <fstream>
#include <set>
#include <functional>
#include "graph/debug/ge_log.h"

using namespace std;
namespace att {
// AST节点类型
enum class NodeType : uint8_t{ 
  OPERATOR,
  FUNCTION,
  VARIABLE, 
  NUMBER 
};

// AST节点数据结构
struct ASTNode {
  string expr;
  string op;
  NodeType type;
  vector<shared_ptr<ASTNode>> children;
  string hash;
  string temp_var;

  ASTNode(string e, NodeType t, string o = "", vector<shared_ptr<ASTNode>> &&c = {}) : expr(e), type(t), op(o), children(move(c)) {
    GenerateHash();
  }

 private:
  // 生成变量或数字节点的hash
  void GenerateLeafHash() {
    if (type == NodeType::VARIABLE) {
      hash = "VAR" + expr;
    } else {
      hash = "NUMBER" + expr;
    }
  }

  // 生成操作符或函数节点的hash
  void GenerateOperatorHash() {
    stringstream ss;
    size_t children_size = children.size();
    ss << op << "(";
    for (size_t i = 0u; i < children_size; ++i) {
      ss << children[i]->hash;
      if (static_cast<size_t>(i + 1) != children_size) {
        ss << ",";
      }
    }
    ss << ")";
    hash = ss.str();
  }

  void GenerateHash() {
    if (type == NodeType::VARIABLE || type == NodeType::NUMBER) {
      GenerateLeafHash();
    } else {
      if (children.empty()) {
        hash = op + "()";  // 处理无子节点的情况
      } else {
        GenerateOperatorHash();
      }
    }
  }
};

using ASTPtr = shared_ptr<ASTNode>;

// AST解析模块，功能包含词法分析和语法分析，最终生成AST
class Parser {
 public:
  explicit Parser(const string &e) : expr_(e) {}
  ~Parser() = default;
  ASTPtr Parse();

 private:
  string Peek(size_t offset = 0) {
    return (pos_ + offset) < tokens_.size() ? tokens_[pos_ + offset] : "";
  }
  void Consume() {
    ++pos_;
  }
  vector<string> Tokenize(const string &s);
  ASTPtr ParseFunction(const string &func);
  ASTPtr ParsePrimary();
  ASTPtr ParseTerm();
  ASTPtr ParseExpr();

  string expr_;
  vector<string> tokens_;
  size_t pos_ = 0;
};

// AST优化器，功能为遍历语法树，提取公共子表达式，表达为临时变量，返回优化后的表达式
class Optimizer {
 public:
  Optimizer() = default;
  ~Optimizer() = default;
  string GenerateCode();
  void Optimize(ASTPtr &root);
  string RebuildExpr(const ASTNode &node, int iter);

 private:
  void Traverse(ASTNode *node);
  unordered_map<string, string> expr_map_;
  vector<ASTNode> temp_order_;
  set<string> visited_;
  int32_t temp_count_ = 0;
};

// AST可视化模块
class ASTVisualizer {
 public:
  void InitDotFile(const string &filename) {
    dot_file_.open(filename + ".dot");
    dot_file_ << "digraph AST {\n";
    dot_file_ << "node [shape=box, fontname=\"Courier\"];\n";
  }
  void GenerateDotImage(const string &filename) {
    dot_file_ << "}\n";
    dot_file_.close();
    system(("dot -Tpng " + filename + ".dot -o " + filename + ".png").c_str());
  }

  void Visualize(ASTPtr &root, const string &filename = "ast") {
    if (!root) {
      return;
    }
    InitDotFile(filename);
    Traverse(root.get());
    GenerateDotImage(filename);
  }

 private:
  ofstream dot_file_;
  unordered_map<ASTNode *, string> node_ids_;
  uint32_t node_counter_ = 0u;

  string GenerateNodeId() {
    return "node_" + to_string(node_counter_++);
  }

  string GetNodeId(ASTNode *node) {
    if (!node) {
      return "null_node";
    }
    if (node_ids_.find(node) == node_ids_.end()) {
      node_ids_[node] = GenerateNodeId();
    }
    return node_ids_[node];
  }

  string GetNodeLabel(ASTNode *node) {
    string label;
    switch (node->type) {
      case NodeType::OPERATOR:
        label = node->op;
        break;
      case NodeType::FUNCTION:
        label = node->op + "()";
        break;
      case NodeType::VARIABLE:
      case NodeType::NUMBER:
        label = node->expr;
        break;
      default:
        label = "unknown";
    }
    if (!node->temp_var.empty()) {
      label = node->temp_var + " = " + label;
    }
    return label;
  }

  string GetNodeColor(ASTNode *node) {
    switch (node->type) {
      case NodeType::OPERATOR:
        return "lightblue";
      case NodeType::FUNCTION:
        return "orange";
      case NodeType::VARIABLE:
        return "green";
      case NodeType::NUMBER:
        return "yellow";
      default:
        return "gray";
    }
  }

  void AddNode(ASTNode *node) {
    if (!node) {
      return;
    }
    string node_id = GetNodeId(node);
    string label = GetNodeLabel(node);
    string color = GetNodeColor(node);
    dot_file_ << node_id << " [label=\"" << label << "\", style=filled, color=" << color << "];\n";
  }

  void AddNodeAndEdges(ASTNode *node) {
    if (!node) {
      return;
    }
    AddNode(node);
    for (auto &child : node->children) {
      if (child) {
        dot_file_ << GetNodeId(node) << " -> " << GetNodeId(child.get()) << ";\n";
      }
    }
  }

  void Traverse(ASTNode *node) {
    if (!node) {
      return;
    }
    AddNodeAndEdges(node);
    for (auto &child : node->children) {
      Traverse(child.get());
    }
  }
};
}  // namespace att
#endif
