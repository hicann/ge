/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include <string>
#include <cstdlib>
#include "es_codegen_default_value.h"
#include "cmd_flag_info.h"
namespace ge {
namespace es {

// 定义命令行参数
DEFINE_string(output_dir, ge::es::kEsCodeGenDefaultOutputDir, "Output directory for generated code files");
DEFINE_string(module_name, ge::es::kEsCodeGenDefaultModelName, "Module name for aggregate header file naming");
DEFINE_string(h_guard_prefix, ge::es::kEsCodeGenDefaultPrefixGuard, "Header guard prefix for generated headers");
DEFINE_string(exclude_ops, ge::es::kEsCodeGenDefaultExcludeOps, "Exclude ops for code generation");
DEFINE_bool(help, false, "show this help message");

void GenEsImpl(const std::string &output_dir, const std::string &module_name, const std::string &h_guard_prefix, const std::string &exclude_ops);
/**
 * ES (Eager Style) Graph Builder Code Generator
 *
 * 功能说明：
 * 本程序用于生成ES图构建器的C和C++,Python代码，包括：
 * - 所有支持的算子(ops)的C接口
 * - 所有支持的算子的C++接口
 * - 所有支持的算子的Python接口
 * - 聚合头文件，方便用户一次性包含所有算子
 * - 聚合Python文件，方便用户一次性导入所有算子
 *
 * 使用方法：
 * ./gen_esb [--output_dir=DIR] [--module_name=NAME] [--h_guard_prefix=PREFIX] [--exclude_ops=OP_TYPE1,OP_TYPE2]
 * 参数说明：
 * --output_dir：可选参数，指定代码生成的目标目录
 *   如果不指定，默认输出到当前目录
 * --module_name：可选参数，控制聚合头文件的命名
 *   - "math" -> es_math_ops_c.h, es_math_ops.h, es_math_ops.py
 *   - "all" -> es_all_ops_c.h, es_all_ops.h, es_all_ops.py
 *   - 不传递 -> 默认为"all"
 * --h_guard_prefix：可选参数，控制生成的头文件保护宏前缀，用于可能的内外部算子同名情况的区分
 *   - 如果不指定，使用默认前缀
 *   - 指定时，拼接默认前缀
 *   - python文件不感知此参数，同名场景通过不同的路径避免冲突
 * --exclude_ops: 可选参数, 控制排除生成的算子
 *   - 根据','分隔算子名
 * 环境变量要求：
 * - ASCEND_OPP_PATH：必须设置，指向CANN安装目录下的ops路径
 *   例如：export ASCEND_OPP_PATH=<cann-install-path>/ops
 * - LD_LIBRARY_PATH:
 *   例如：export LD_LIBRARY_PATH=<cann-install-path>/x86_64-linux/lib64
 *
 * 输出文件说明：
 * - es_<module>_ops_c.h：C接口聚合头文件
 * - es_<module>_ops.h：C++接口聚合头文件
 * - es_<module>_ops.py：Python接口聚合文件
 * - es_<op_type>_c.h：单个算子的C接口头文件
 * - es_<op_type>.cpp：单个算子的C接口实现文件
 * - es_<op_type>.h：单个算子的C++接口头文件
 * - es_<op_type>.py：单个算子的Python接口文件
 *
 * 使用示例：
 * # 生成到当前目录，使用默认模块名"all"，默认保护宏前缀
 * ./gen_esb
 *
 * # 生成到指定目录，使用默认模块名"all"，默认保护宏前缀
 * ./gen_esb --output_dir=./output
 *
 * # 生成到指定目录，使用"math"模块名，默认保护宏前缀
 * ./gen_esb --output_dir=./output --module_name=math
 *
 * # 生成到指定目录，使用"all"模块名，默认保护宏前缀
 * ./gen_esb --output_dir=./output --module_name=all
 *
 * # 生成到指定目录，使用"math"模块名，自定义保护宏前缀"MY_CUSTOM"
 * ./gen_esb --output_dir=./output --module_name=math --h_guard_prefix=MY_CUSTOM
 *
 * # 生成到指定目录，使用"math"模块名，自定义保护宏前缀"MY_CUSTOM", 并排除Add算子生成
 * ./gen_esb --output_dir=./output --module_name=math --h_guard_prefix=MY_CUSTOM --exclude_ops=Add
 *
 * # 检查环境变量
 * echo $ASCEND_OPP_PATH
 *
 * 注意事项：
 * 1. 确保ASCEND_OPP_PATH环境变量已正确设置
 * 2. 确保有足够的磁盘空间存储生成的代码文件
 * 3. 生成的代码文件数量取决于系统中注册的算子数量
 * 4. 保护宏前缀应该以大写字母和下划线组成，避免与C++关键字冲突
 *
 * 错误处理：
 * - 如果环境变量未设置，程序会提示错误并退出
 * - 如果输出目录创建失败，会回退到当前目录
 * - 不支持的算子会被记录在生成的代码注释中
 *
 * @return 程序退出码，0表示成功
 */
/**
 * 显示程序标题和版本信息
 */
void DisplayProgramHeader() {
  std::cout << "==========================================" << std::endl;
  std::cout << "ES Graph Builder Code Generator v1.0" << std::endl;
  std::cout << "Copyright (c) 2025 Huawei Technologies Co., Ltd." << std::endl;
  std::cout << "==========================================" << std::endl;
}

/**
 * 解析命令行参数
 * @param argc 参数个数
 * @param argv 参数数组
 * @param output_dir 输出参数，输出目录
 * @param module_name 输出参数，模块名
 * @param h_guard_prefix 输出参数，保护宏前缀
 * @return 是否解析成功
 */
bool ParseCommandLineArgs(int argc, char *argv[], std::string &output_dir, std::string &module_name,
                          std::string &h_guard_prefix, std::string &exclude_ops) {
  // 使用cmd_flag_info库解析参数
  ge::flgs::SetUsageMessage(R"(
ES Graph Builder Code Generator v1.0
Usage: ./gen_esb [--output_dir=DIR] [--module_name=NAME] [--h_guard_prefix=PREFIX] [--exclude_ops=OP_TYPE]

Examples:
  ./gen_esb                                    # Use all defaults
  ./gen_esb --output_dir=./output             # Output to ./output directory
  ./gen_esb --module_name=math                # Use 'math' module name
  ./gen_esb --h_guard_prefix=MY_CUSTOM       # Custom header guard prefix
  ./gen_esb --output_dir=./output --module_name=math --h_guard_prefix=MY_CUSTOM # Combine options
  ./gen_esb --exclude_ops=Add,Conv2D
  
Environment variables required:
  ASCEND_OPP_PATH        # Must be set, pointing to CANN ops path
  LD_LIBRARY_PATH        # Must be set, pointing to CANN lib path
)");

  // 解析命令行参数
  ge::flgs::GfStatus status = ge::flgs::ParseCommandLine(argc, argv);
  if (status == ge::flgs::GF_HELP) {
    // 显示帮助信息
    return false;
  } else if (status != ge::flgs::GF_SUCCESS) {
    // 参数解析失败
    std::cerr << "Error: Failed to parse command line arguments! gen_esb --help may be helpful" << std::endl;
    return false;
  }

  // 从解析后的参数中获取值
  output_dir = FLAGS_output_dir;
  module_name = FLAGS_module_name;
  h_guard_prefix = FLAGS_h_guard_prefix;
  exclude_ops = FLAGS_exclude_ops;
  return true;
}

/**
 * 检查必需的环境变量
 * @return 环境变量检查是否通过
 */
bool CheckEnvironmentVariables() {
  const char *opp_path = std::getenv("ASCEND_OPP_PATH");
  if (!opp_path) {
    std::cerr << "\nError: Environment variable ASCEND_OPP_PATH is not set!" << std::endl;
    std::cerr << "Please set the environment variable:" << std::endl;
    std::cerr << "  export ASCEND_OPP_PATH=<cann-install-path>/ops" << std::endl;
    return false;
  }

  const char *lib_path = std::getenv("LD_LIBRARY_PATH");
  if (!lib_path) {
    std::cout << "\nWarning: Environment variable LD_LIBRARY_PATH is not set!" << std::endl;
    std::cout << "It is recommended to set the environment variable:" << std::endl;
    std::cout << "  export LD_LIBRARY_PATH=<cann-install-path>/x86_64-linux/lib64" << std::endl;
  }

  std::cout << "Environment variable check: ASCEND_OPP_PATH = " << opp_path << std::endl;
  if (lib_path) {
    std::cout << "Environment variable check: LD_LIBRARY_PATH = " << lib_path << std::endl;
  }

  return true;
}

/**
 * 执行代码生成
 * @param output_dir 输出目录
 * @param module_name 模块名
 * @param h_guard_prefix 保护宏前缀
 * @return 是否成功
 */
bool ExecuteCodeGeneration(const std::string &output_dir, const std::string &module_name,
                           const std::string &h_guard_prefix, const std::string &exclude_ops_str) {
  try {
    GenEsImpl(output_dir, module_name, h_guard_prefix, exclude_ops_str);
    return true;
  } catch (const std::exception &e) {
    std::cerr << "\nError: Exception occurred during code generation:" << std::endl;
    std::cerr << "  " << e.what() << std::endl;
    std::cerr << "\nPlease check environment configuration and permissions." << std::endl;
    return false;
  } catch (...) {
    std::cerr << "\nError: Unknown exception occurred during code generation!" << std::endl;
    std::cerr << "Please check system environment and CANN installation status." << std::endl;
    return false;
  }
}
}  // namespace es
}  // namespace ge


int main(int argc, char *argv[]) {
  try {
    // 1. 显示程序标题
    ge::es::DisplayProgramHeader();

    // 2. 初始化默认参数
    std::string output_dir = ge::es::kEsCodeGenDefaultOutputDir;
    std::string module_name = ge::es::kEsCodeGenDefaultModelName;
    std::string h_guard_prefix = ge::es::kEsCodeGenDefaultPrefixGuard;
    std::string exclude_ops_str = ge::es::kEsCodeGenDefaultExcludeOps;
    // 3. 解析命令行参数
    if (!ge::es::ParseCommandLineArgs(argc, argv, output_dir, module_name, h_guard_prefix, exclude_ops_str)) {
      // 显示帮助信息或参数解析失败
      return 0;  // 帮助信息正常退出
    }

    // 4. 开始执行
    std::cout << "Starting code generation..." << std::endl;

    // 5. 检查环境变量
    if (!ge::es::CheckEnvironmentVariables()) {
      return 1;
    }

    // 6. 执行代码生成
    if (ge::es::ExecuteCodeGeneration(output_dir, module_name, h_guard_prefix, exclude_ops_str)) {
      std::cout << "\n==========================================" << std::endl;
      std::cout << "Code generation completed!" << std::endl;
      std::cout << "Module name: " << module_name << std::endl;
      std::cout << "==========================================" << std::endl;
      return 0;
    } else {
      return 1;
    }
  } catch (const std::exception &e) {
    std::cerr << "\nError: " << e.what() << std::endl;
    return 1;
  } catch (...) {
    std::cerr << "\nError: Unknown exception occurred!" << std::endl;
    return 1;
  }
}