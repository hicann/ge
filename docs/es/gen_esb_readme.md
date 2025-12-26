# ES (Eager Style) Graph Builder Code Generator
## 前置要求:
1. 通过[安装指导](../../README.md)正确安装`toolkit`包, 并正确配置环境变量
2. 通过[安装指导](https://gitcode.com/cann/ops-math/blob/master/docs/zh/context/quick_install.md)正确安装算子`ops`包 (ES依赖算子原型进行API生成), 并正确配置环境变量
## 环境变量要求
注: 环境变量在前置要求中已经被配置, 仅列出gen_esb所需环境变量列表
- ASCEND_OPP_PATH: 指向安装目录下的opp路径
- LD_LIBRARY_PATH: 指定动态链接库搜索路径的环境变量

## 功能说明：
### 本程序用于生成ES图构建器的C和C++,Python代码，包括：
- 所有支持的算子(ops)的C接口
- 所有支持的算子的C++接口
- 所有支持的算子的Python接口
- 聚合头文件，方便用户一次性包含所有算子
- 聚合Python文件，方便用户一次性导入所有算子
 
## 使用方法：
### `./gen_esb [--output_dir=DIR] [--module_name=NAME] [--h_guard_prefix=PREFIX] [--exclude_ops=OP_TYPE1,OP_TYPE2]`

### 参数说明：
- --output_dir：可选参数，指定代码生成的目标目录
  如果不指定，默认输出到当前目录
- --module_name：可选参数，控制聚合头文件的命名
  - "math" -> es_math_ops_c.h, es_math_ops.h, es_math_ops.py
  - "all" -> es_all_ops_c.h, es_all_ops.h, es_all_ops.py
  - 不传递 -> 默认为"all"
- --h_guard_prefix：可选参数，控制生成的头文件保护宏前缀，用于可能的内外部算子同名情况的区分
  - 如果不指定，使用默认前缀
  - 指定时，拼接默认前缀
  - python文件不感知此参数，同名场景通过不同的路径避免冲突
- --exclude_ops: 可选参数, 控制排除生成的算子
  - 根据','分隔算子名
 
### 输出文件说明：
- es_\<module>_ops_c.h：C接口聚合头文件
- es_\<module>_ops.h：C++接口聚合头文件
- es_\<module>_ops.py：Python接口聚合文件
- es_<op_type>_c.h：单个算子的C接口头文件
- es_<op_type>.cpp：单个算子的C接口实现文件
- es_<op_type>.h：单个算子的C++接口头文件
- es_<op_type>.py：单个算子的Python接口文件
 
## 使用示例：
### 生成到当前目录，使用默认模块名"all"，默认保护宏前缀
`./gen_esb`
 
### 生成到指定目录，使用默认模块名"all"，默认保护宏前缀
`./gen_esb --output_dir=./output`
 
### 生成到指定目录，使用"math"模块名，默认保护宏前缀
`./gen_esb --output_dir=./output --module_name=math`
 
### 生成到指定目录，使用"all"模块名，默认保护宏前缀
`./gen_esb --output_dir=./output --module_name=all`
 
### 生成到指定目录，使用"math"模块名，自定义保护宏前缀"MY_CUSTOM"
`./gen_esb --output_dir=./output --module_name=math --h_guard_prefix=MY_CUSTOM`

### 生成到指定目录，使用"math"模块名，自定义保护宏前缀"MY_CUSTOM", 并排除Add算子生成
`./gen_esb --output_dir=./output --module_name=math --h_guard_prefix=MY_CUSTOM --exclude_ops=Add`

## 检查环境变量
`echo $ASCEND_OPP_PATH`
 
## 注意事项：
1. 确保ASCEND_OPP_PATH环境变量已正确设置
2. 确保有足够的磁盘空间存储生成的代码文件
3. 生成的代码文件数量取决于系统中注册的算子数量
4. 保护宏前缀应该以大写字母和下划线组成，避免与C++关键字冲突
 
## 错误处理：
- 如果环境变量未设置，程序会提示错误并退出
- 如果输出目录创建失败，会回退到当前目录
- 不支持的算子会被记录在生成的代码注释中