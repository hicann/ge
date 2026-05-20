# AddOutput<a name="ZH-CN_TOPIC_0000002520225201"></a>

## 产品支持情况<a name="section789110355115"></a>

| 产品 | 是否支持 |
| :--- | :---: |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √ |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √ |

## 头文件/库文件<a name="section710017525393"></a>

-   头文件：#include <graph/named_io_node_builder.h>
-   库文件：libgraph.so、libgraph_static.a

## 功能说明<a name="section44282631"></a>

添加输出实例端口。用户按IR定义顺序依次添加输出实例，普通输出使用IR名，动态输出使用ir_name0、ir_name1、……形式的连续实例名。

## 函数原型<a name="section1831611148523"></a>

```
NamedIoNodeBuilder &AddOutput(const char_t *name)
NamedIoNodeBuilder &AddOutput(const char_t *name, const TensorDesc &desc)
```

-   第一个原型：添加输出实例端口，使用默认TensorDesc（DT_FLOAT + FORMAT_ND）。
-   第二个原型：添加带描述的输出实例端口，可指定数据类型、格式和形状。

## 参数说明<a name="section62999334"></a>

| 参数名 | 输入/输出 | 描述 |
| :--- | :---: | :--- |
| name | 输入 | 输出实例名称。普通输出使用IR名，动态输出使用ir_name0、ir_name1、……形式的连续实例名。 |
| desc | 输入 | 输出张量描述，包含数据类型、格式和形状信息。仅在第二个原型中使用。 |

## 返回值说明<a name="section30123067"></a>

返回构建器引用，支持链式调用。

## 约束说明<a name="section24049043"></a>

-   输出实例需按IR定义顺序添加。
-   若传递动态输出实例，实例名需从ir_name0开始连续编号，不能跳过中间序号。
-   必选输出必须添加，否则Build返回nullptr。
-   若传入nullptr为name，该输出不会被添加。
