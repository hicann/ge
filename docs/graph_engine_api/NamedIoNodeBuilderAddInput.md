# AddInput<a name="ZH-CN_TOPIC_0000002520225200"></a>

## 产品支持情况<a name="section789110355114"></a>

| 产品 | 是否支持 |
| :--- | :---: |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √ |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √ |

## 头文件/库文件<a name="section710017525392"></a>

-   头文件：#include <graph/named_io_node_builder.h>
-   库文件：libgraph.so、libgraph_static.a

## 功能说明<a name="section44282630"></a>

添加输入实例端口。用户按IR定义顺序依次添加输入实例，普通/可选输入使用IR名，动态输入使用ir_name0、ir_name1、……形式的连续实例名。

## 函数原型<a name="section1831611148522"></a>

```
NamedIoNodeBuilder &AddInput(const char_t *name)
NamedIoNodeBuilder &AddInput(const char_t *name, const TensorDesc &desc)
```

-   第一个原型：添加输入实例端口，使用默认TensorDesc（DT_FLOAT + FORMAT_ND）。
-   第二个原型：添加带描述的输入实例端口，可指定数据类型、格式和形状。

## 参数说明<a name="section62999333"></a>

| 参数名 | 输入/输出 | 描述 |
| :--- | :---: | :--- |
| name | 输入 | 输入实例名称。普通/可选输入使用IR名，动态输入使用ir_name0、ir_name1、……形式的连续实例名。 |
| desc | 输入 | 输入张量描述，包含数据类型、格式和形状信息。仅在第二个原型中使用。 |

## 返回值说明<a name="section30123066"></a>

返回构建器引用，支持链式调用。

## 约束说明<a name="section24049042"></a>

-   输入实例需按IR定义顺序添加。
-   可选输入（OPTIONAL_INPUT）的使用方式：
    -   不传递可选输入：仅添加必选输入，跳过该可选输入即可。
    -   传递有效可选输入：按IR顺序在对应位置添加，使用默认TensorDesc。
    -   传递占位可选输入：使用带TensorDesc的重载，传入DT_UNDEFINED和FORMAT_RESERVED。
    -   可选输入最多只能提供一个实例，重复添加会导致校验失败。
-   若传递动态输入实例，实例名需从ir_name0开始连续编号，不能跳过中间序号。
-   若传入nullptr为name，该输入不会被添加。
