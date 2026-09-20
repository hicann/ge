# SetOutputs（index）

## 产品支持情况

<!-- npu="A3" id1 -->
- Atlas A3系列产品：支持
<!-- end id1 -->
<!-- npu="910b" id2 -->
- Atlas A2系列产品：支持
<!-- end id2 -->

## 函数功能

设置FlowGraph中的FlowNode和FlowNode输出index的关联关系，并返回该图。常用于设置FlowNode部分输出场景，比如FlowNode1有2个输出，但是作为FlowNode2输入的时候只需要FlowNode1的一个输出，这种情况下可以设置FlowNode1的一个输出index。

## 函数原型

```cpp
FlowGraph &SetOutputs(const std::vector<std::pair<FlowOperator, std::vector<size_t>>> &output_indexes)
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| output_indexes | 输入 | 输出节点和节点输出index关联关系集合。 |

## 返回值

返回设置了输出节点和节点输出index关联关系的FlowGraph图。

## 异常处理

无。

## 约束说明

无。
