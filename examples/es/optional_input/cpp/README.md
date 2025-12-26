# 样例使用指导

## 1、功能描述
本样例使用BatchNorm算子可选输入进行构图，旨在帮助构图开发者快速理解可选输入定义和使用该类型算子进行构图
## 2、目录结构
```angular2html
cpp/
├── src/
|   └── CMakeLists.txt            // 编译脚本
|   └── es_showcase.h             // 头文件
|   └── make_batchnorm_garph.cpp  // sample文件
├── CMakeLists.txt                // 编译脚本
├── main.cpp                      // 程序主入口
├── README.md                     // README文件
├── run_sample.sh                 // 执行脚本
├── utils.h                       // 工具文件
```

## 3、使用方法
### 3.1、准备cann包
- 通过安装指导 [环境准备](../../../../README.md)正确安装`toolkit`包
- 设置环境变量 (假设包安装在/usr/local/Ascend/)
```
source /usr/local/Ascend/cann/set_env.sh 
```
- 通过[安装指导](https://gitcode.com/cann/ops-math/blob/master/docs/zh/context/quick_install.md)正确安装算子`ops`包 (ES依赖算子原型进行API生成), 并正确配置环境变量
- 安装算子`ops`包 (ES依赖算子原型进行API生成)
### 3.2、编译和执行

只需运行下述命令即可完成清理、生成接口、构图和DUMP图：
```bash
bash run_sample.sh
```
当前 run_sample.sh 的行为是：先自动清理旧的 build，构建 sample并默认执行sample dump 。当看到如下信息，代表执行成功：
```
[Success] sample 执行成功，pbtxt dump 已生成在当前目录。该文件以 ge_onnx_ 开头，可以在 netron 中打开显示
```
执行成功后会在当前目录生成以下文件：
- `ge_onnx_*.pbtxt` - 图结构的protobuf文本格式，可用netron查看

### 3.3、日志打印
可执行程序执行过程中如果需要日志打印来辅助定位，可以在bash run_sample.sh之前设置如下环境变量来让日志打印到屏幕
```bash
export ASCEND_SLOG_PRINT_TO_STDOUT=1 #日志打印到屏幕
export ASCEND_GLOBAL_LOG_LEVEL=0 #日志级别为debug级别
```

## 4、核心概念介绍

### 4.1、构图步骤如下：
- 创建图构建器(用于提供构图所需的上下文、工作空间及构建相关方法)
- 添加起始节点(起始节点指无输入依赖的节点，通常包括图的输入(如 Data 节点)和权重常量(如 Const 节点))
- 添加中间节点(中间节点为具有输入依赖的计算节点，通常由用户构图逻辑生成，并通过已有节点作为输入连接)
- 设置图输出(明确图的输出节点，作为计算结果的终点)

### 4.2、概念说明：
可选输入是指算子的某些输入是非必选输入。

例如 BatchNorm 算子原型如下所示，ES 构图生成的API是`BatchNorm()`，支持在 Python 层使用
```bash
  REG_OP(BatchNorm)
    .INPUT(x, TensorType({DT_FLOAT16,DT_FLOAT}))
    .INPUT(scale, TensorType({DT_FLOAT}))
    .INPUT(offset, TensorType({DT_FLOAT}))
    .OPTIONAL_INPUT(mean, TensorType({DT_FLOAT}))
    .OPTIONAL_INPUT(variance, TensorType({DT_FLOAT}))
    .OUTPUT(y, TensorType({DT_FLOAT16,DT_FLOAT}))
    .OUTPUT(batch_mean, TensorType({DT_FLOAT}))
    .OUTPUT(batch_variance, TensorType({DT_FLOAT}))
    .OUTPUT(reserve_space_1, TensorType({DT_FLOAT}))
    .OUTPUT(reserve_space_2, TensorType({DT_FLOAT}))
    .OUTPUT(reserve_space_3, TensorType({DT_FLOAT}))
    .ATTR(epsilon, Float, 0.0001f)
    .ATTR(data_format, String, "NHWC")
    .ATTR(is_training, Bool, true)
    .ATTR(exponential_avg_factor, Float, 1.0)
    .OP_END_FACTORY_REG(BatchNorm)
```

其对应的函数原型为：
- 函数名：BatchNorm
- 参数：共 9 个，依次为 x， scale， offset， mean(可选输入)， variance(可选输入)， epsilon， data_format， is_training， exponential_avg_factor
- 返回值：输出 y， batch_mean， batch_variance， reserve_space_1， reserve_space_2， reserve_space_3

**C API中：**
```
EsBatchNormOutput EsBatchNorm(EsCTensorHolder *x, EsCTensorHolder *scale, EsCTensorHolder *offset, EsCTensorHolder *mean, EsCTensorHolder *variance, float epsilon, const char *data_format, bool is_training, float exponential_avg_factor);
typedef struct {
    EsCTensorHolder *y;
    EsCTensorHolder *batch_mean;
    EsCTensorHolder *batch_variance;
    EsCTensorHolder *reserve_space_1;
    EsCTensorHolder *reserve_space_2;
    EsCTensorHolder *reserve_space_3;
} EsBatchNormOutput;
```
**C++ API中：**
```
BatchNormOutput BatchNorm(const EsTensorLike &x, const EsTensorLike &scale, const EsTensorLike &offset, const EsTensorLike &mean=nullptr, const EsTensorLike &variance=nullptr, float epsilon=0.000100, const char *data_format="NHWC", bool is_training=true, float exponential_avg_factor=1.000000);
struct BatchNormOutput {
    EsTensorHolder y;
    EsTensorHolder batch_mean;
    EsTensorHolder batch_variance;
    EsTensorHolder reserve_space_1;
    EsTensorHolder reserve_space_2;
    EsTensorHolder reserve_space_3;
};
```
注： 使用TensorLike类型表达输入，以支持实参可以直接传递数值的情况