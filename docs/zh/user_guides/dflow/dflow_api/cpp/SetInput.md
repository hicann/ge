# SetInput

## 产品支持情况

<!-- npu="A3" id1 -->
- Atlas A3系列产品：支持
<!-- end id1 -->
<!-- npu="910b" id2 -->
- Atlas A2系列产品：支持
<!-- end id2 -->

## 函数功能

给FlowNode设置输入，表示将src\_op的第src\_index个输出作为FlowNode的第dst\_index个输入，返回设置好输入的FlowNode节点。

## 函数原型

```cpp
FlowNode &SetInput(uint32_t dst_index, const FlowOperator &src_op, uint32_t src_index = 0)
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| dst_index | 输入 | FlowNode输入index。 |
| src_op | 输入 | FlowNode输入的节点，只能是FlowNode或者FlowData。 |
| src_index | 输入 | src_op的输出index。 |

## 返回值

返回设置好输入的FlowNode节点。

## 异常处理

无。

## 约束说明

无。
