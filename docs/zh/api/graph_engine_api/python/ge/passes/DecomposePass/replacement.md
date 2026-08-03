# replacement

## 产品支持情况

全量芯片支持。

## 功能说明

生成分解子图。

## 函数原型

```python
replacement(self, node: Node) -> Graph
replacement(self, node: Node, context: PassContext) -> Graph
```

## 参数说明

| 参数名 | 输入/输出 | 说明 |
| --- | --- | --- |
| node | 输入 | 待分解的节点，类型为ge.graph.Node。 |
| context | 输入 | 当前编译期PassContext，仅在当前回调栈内有效，可读取编译选项或设置错误信息。 |

## 返回值说明

| 类型 | 说明 |
| --- | --- |
| Graph | 返回分解后的子图，类型为ge.graph.Graph。 |

## 约束说明

可选声明`context`参数；未声明时保持原有单参数调用方式。
