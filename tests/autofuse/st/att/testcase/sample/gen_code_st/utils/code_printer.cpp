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
#include "code_printer.h"
#include <fstream>
#include "common/checker.h"

namespace att {
void CodePrinter::AddInclude(const std::string &include_name) {
  output_ << "#include \"" << include_name << "\"" << std::endl;
}

void CodePrinter::AddNamespaceBegin(const std::string &namespace_name) {
  output_ << "namespace " << namespace_name << " {" << std::endl;
}

void CodePrinter::AddNamespaceEnd(const std::string &namespace_name) {
  output_ << "} // namespace " << namespace_name << std::endl;
}

void CodePrinter::DefineClassBegin(const std::string &class_name) {
  output_ << "class " << class_name << " {" << std::endl;
}

void CodePrinter::DefineClassEnd() {
  output_ << "};" << std::endl;
}

void CodePrinter::AddStructBegin(const std::string &struct_name) {
  output_ << "struct " << struct_name << " {" << std::endl;
}

void CodePrinter::AddStructEnd() {
  output_ << "};" << std::endl;
}

void CodePrinter::DefineFuncBegin(const std::string &return_type, const std::string &func_name,
                                  const std::string &param) {
  output_ << return_type << " " << func_name << "(" << param << ")" << std::endl << "{" << std::endl;
}

void CodePrinter::DefineFuncEnd() {
  output_ << "}" << std::endl;
}

void CodePrinter::AddLine(const std::string &input_string) {
  output_ << input_string << std::endl;
}

ge::Status CodePrinter::SaveToFile(const std::string &output_file_path) {
  std::ofstream out_file(output_file_path);
  if (out_file.is_open()) {
    out_file << output_.str();
    out_file.close();
    return ge::SUCCESS;
  }
  return ge::SUCCESS;
}

std::string CodePrinter::GetOutputStr() const {
  return output_.str();
}

void CodePrinter::Reset() {
  std::stringstream new_output;
  output_.swap(new_output);
}
}  // namespace att