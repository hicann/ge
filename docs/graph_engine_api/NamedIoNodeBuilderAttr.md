# Attr<a name="ZH-CN_TOPIC_0000002520225202"></a>

## 产品支持情况<a name="section789110355116"></a>

| 产品 | 是否支持 |
| :--- | :---: |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √ |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √ |

## 头文件/库文件<a name="section710017525394"></a>

-   头文件：#include <graph/named_io_node_builder.h>
-   库文件：libgraph.so、libgraph_static.a

## 功能说明<a name="section44282632"></a>

设置节点属性。Build时用户设置的属性优先，已注册IR定义中的默认属性仅用于补全，不覆盖用户设置值。

## 函数原型<a name="section1831611148524"></a>

```
NamedIoNodeBuilder &Attr(const char_t *name, const AttrValue &value)
```

## 参数说明<a name="section62999335"></a>

| 参数名 | 输入/输出 | 描述 |
| :--- | :---: | :--- |
| name | 输入 | 属性名称。 |
| value | 输入 | 属性值，通过AttrValue::SetAttrValue构造。 |

## 返回值说明<a name="section30123068"></a>

返回构建器引用，支持链式调用。

## 约束说明<a name="section24049044"></a>

-   若传入nullptr为name，该属性不会被设置。
-   若AttrValue未设置值（默认构造），Build时可能返回nullptr。
-   用户设置的属性值不会被已注册IR定义中的默认属性覆盖。
