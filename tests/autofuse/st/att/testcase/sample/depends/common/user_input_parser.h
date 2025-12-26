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
#ifndef SAMPLEPROJECT_ATT_SAMPLE_TESTS_UTILS_USER_INPUT_PARSER_H_
#define SAMPLEPROJECT_ATT_SAMPLE_TESTS_UTILS_USER_INPUT_PARSER_H_
#include <vector>
#include <string>
#include "common/checker.h"
namespace att {
struct InputShape {
  uint32_t tiling_key;
  std::vector<std::pair<std::string, uint32_t>> axes;
};
struct UserInput {
  std::vector<InputShape> input_shapes;
};
class UserInputParser {
 public:
  static std::string ToJson(const InputShape &input_shape, const int32_t indent = 4);
  static ge::Status FromJson(const std::string &json_str, InputShape &input_shape);
  static std::string ToJson(const UserInput &user_input, const int32_t indent = 4);
  static ge::Status FromJson(const std::string &json_str, UserInput &user_input);
  static bool CheckInputValid(const std::string &input_file);
  static std::string ReadFileToString(const std::string& file_path);
};
}  // namespace att

#endif  // SAMPLEPROJECT_ATT_SAMPLE_TESTS_UTILS_USER_INPUT_PARSER_H_
