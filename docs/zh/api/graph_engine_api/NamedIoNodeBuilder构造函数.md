# NamedIoNodeBuilder构造函数<a name="ZH-CN_TOPIC_0000002488071607"></a>

## 产品支持情况<a name="section789110355111"></a>

| 产品 | 是否支持 |
| :--- | :---: |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √ |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √ |

## 头文件/库文件<a name="section710017525389"></a>

-   头文件：#include <graph/named_io_node_builder.h>
-   库文件：libgraph.so、libgraph_static.a

## 功能说明<a name="zh-cn_topic_0204328165_zh-cn_topic_0182636384_section36893359"></a>

NamedIoNodeBuilder构造函数，创建基于输入、输出和属性名称构造图节点的Builder对象。

NamedIoNodeBuilder面向直接按照节点输入/输出名称构图的场景。调用方可指定节点实例的输入名、输出名、属性名及相关描述信息；构建时根据节点类型匹配已通过REG_OP注册的算子IR定义，校验输入/输出实例名称及顺序是否与该定义兼容，最终创建对应的图节点。

## 函数原型<a name="zh-cn_topic_0204328165_zh-cn_topic_0182636384_section136951948195410"></a>

```
explicit NamedIoNodeBuilder(Graph &graph)
~NamedIoNodeBuilder()
```

## 参数说明<a name="section144401754174019"></a>

| 参数名 | 输入/输出 | 描述 |
| :--- | :---: | :--- |
| graph | 输入 | 所属的图对象，构建完成的节点将被添加到此图中。 |

## 返回值说明<a name="zh-cn_topic_0204328165_zh-cn_topic_0182636384_section35572113"></a>

构造函数返回NamedIoNodeBuilder类型的对象。

## 约束说明<a name="zh-cn_topic_0204328165_zh-cn_topic_0182636384_section62768826"></a>

-   NamedIoNodeBuilder对象不可拷贝构造和赋值。
-   构建器对象Build成功后不应再次使用。
