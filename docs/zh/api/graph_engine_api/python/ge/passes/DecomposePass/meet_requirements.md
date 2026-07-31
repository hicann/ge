# meet\_requirements

## 产品支持情况

全量芯片支持。

## 功能说明

判断节点是否需要分解。为可选实现，默认返回True。

## 函数原型

```python
meet_requirements(self, node: Node) -> bool
meet_requirements(self, node: Node, context: PassContext) -> bool
```

## 参数说明

| 参数名 | 输入/输出 | 说明 |
| --- | --- | --- |
| node | 输入 | 待判断的节点，类型为ge.graph.Node。 |
| context | 输入 | 当前编译期PassContext，仅在当前回调栈内有效，可读取编译选项或设置错误信息。 |

## 返回值说明

返回True表示需要分解，将执行替换；返回False表示不需要分解，跳过。默认返回True。

## 约束说明

可选声明`context`参数；未声明时保持原有单参数调用方式。
