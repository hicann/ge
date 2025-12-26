/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2024 All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef ATT_CODE_PRINTER_H_
#define ATT_CODE_PRINTER_H_
#include <string>
#include <sstream>
#include "common/checker.h"
namespace att {
class CodePrinter {
 public:
  // 默认无函数构造
  CodePrinter() = default;
  ~CodePrinter() = default;
  /**
   * 将拼接好的字符串输出
   */
  std::string GetOutputStr() const;
  /**
   * 清空之前的内容
   */
  void Reset();
  /**
   * 将拼接好的字符串写入文件
   */
  ge::Status SaveToFile(const std::string &output_file_path);
  /**
   * 添加一行，自动换行
   */
  void AddLine(const std::string &input_string);

 public:
  void AddInclude(const std::string &include_name);
  void AddNamespaceBegin(const std::string &namespace_name);
  void AddNamespaceEnd(const std::string &namespace_name);
  void DefineClassBegin(const std::string &class_name);
  void DefineClassEnd();
  void AddStructBegin(const std::string &struct_name);
  void AddStructEnd();
  void DefineFuncBegin(const std::string &return_type, const std::string &func_name, const std::string &param_name);
  void DefineFuncEnd();
 private:
  std::stringstream output_;
  std::string output_file_path_;
};
}  // namespace att
#endif  // ATT_CODE_PRINTER_H_