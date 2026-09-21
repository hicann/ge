# SetInputs

## 产品支持情况

<!-- npu="A3" id1 -->
- Atlas A3系列产品：支持
<!-- end id1 -->
<!-- npu="910b" id2 -->
- Atlas A2系列产品：支持
<!-- end id2 -->

## 函数功能

设置FlowGraph的输入节点，会自动根据节点的输出连接关系构建出一张FlowGraph图，并返回该图。

## 函数原型

```cpp
FlowGraph &SetInputs(const std::vector<FlowOperator> &inputs)
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| inputs | 输入 | 输入的节点集。 |

## 返回值

返回构建好输入节点的FlowGraph图。

## 异常处理

无。

## 约束说明

无。
