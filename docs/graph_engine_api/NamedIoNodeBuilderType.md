# Type<a name="ZH-CN_TOPIC_0000002520225198"></a>

## 产品支持情况<a name="section789110355112"></a>

| 产品 | 是否支持 |
| :--- | :---: |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √ |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √ |

## 头文件/库文件<a name="section710017525390"></a>

-   头文件：#include <graph/named_io_node_builder.h>
-   库文件：libgraph.so、libgraph_static.a

## 功能说明<a name="section44282628"></a>

设置节点类型（必填）。节点类型对应的算子IR需已通过REG_OP完成注册，否则Build时返回nullptr。

## 函数原型<a name="section1831611148520"></a>

```
NamedIoNodeBuilder &Type(const char_t *type)
```

## 参数说明<a name="section62999331"></a>

| 参数名 | 输入/输出 | 描述 |
| :--- | :---: | :--- |
| type | 输入 | 算子类型，如"Add"、"MatMul"等，对应算子IR需已通过REG_OP完成注册。 |

## 返回值说明<a name="section30123064"></a>

返回构建器引用，支持链式调用。

## 约束说明<a name="section24049040"></a>

-   该方法为必填项，未调用Type时Build将返回nullptr。
-   若传入nullptr，Type不会被设置，Build时将报错"Type must be set before Build()"。
