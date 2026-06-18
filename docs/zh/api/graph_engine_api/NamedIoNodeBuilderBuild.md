# Build<a name="ZH-CN_TOPIC_0000002520225203"></a>

## 产品支持情况<a name="section789110355117"></a>

| 产品 | 是否支持 |
| :--- | :---: |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √ |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √ |

## 头文件/库文件<a name="section710017525395"></a>

-   头文件：#include <graph/named_io_node_builder.h>
-   库文件：libgraph.so、libgraph_static.a

## 功能说明<a name="section44282633"></a>

构建图节点并添加到Graph。Build会基于已设置的输入、输出和属性信息创建GNode，并校验输入/输出实例名称及顺序是否与已注册的算子IR定义兼容。成功时返回的GNode已被添加到Graph中。

Build执行以下对外可见校验和操作：
1. 校验Type已设置，且节点类型对应的算子IR已通过REG_OP完成注册。
2. 校验输入/输出实例名称及顺序是否与已注册的算子IR定义兼容。
3. 创建GNode并添加到Graph。
4. 返回unique_ptr<GNode>，成功时清空error_message。

## 函数原型<a name="section1831611148525"></a>

```
std::unique_ptr<GNode> Build(AscendString &error_message)
```

## 参数说明<a name="section62999336"></a>

| 参数名 | 输入/输出 | 描述 |
| :--- | :---: | :--- |
| error_message | 输出 | 构建失败时的错误信息。成功时error_message被清空。 |

## 返回值说明<a name="section30123069"></a>

| 类型 | 描述 |
| :--- | :--- |
| std::unique_ptr<GNode> | 成功时返回GNode的unique_ptr，已被添加到Graph中。失败时返回nullptr。 |

## 约束说明<a name="section24049045"></a>

-   Builder对象Build成功后不应再次使用，Build成功后再次调用返回nullptr并输出错误信息"Build() has already been called on this builder"。
-   节点类型对应的算子IR需已通过REG_OP完成注册，否则Build返回nullptr。
-   输入/输出实例需按IR定义顺序添加，否则Build返回nullptr。
-   Data和NetOutput节点不进行输入/输出实例兼容性校验。

## 调用示例<a name="section16305113853314"></a>

```
ge::Graph graph("test_graph");
ge::AscendString error_msg;

// 构建一个Add算子节点
auto node = ge::NamedIoNodeBuilder(graph)
    .Type("Add")
    .Name("add_node")
    .AddInput("x1")
    .AddInput("x2")
    .AddOutput("y")
    .Build(error_msg);

if (node != nullptr) {
    // 构建成功，节点已被添加到graph中
} else {
    // 构建失败，查看error_msg获取错误信息
}
```
