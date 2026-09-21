# dataflow.FlowOutput

## 产品支持情况

<!-- npu="A3" id1 -->
- Atlas A3系列产品：支持
<!-- end id1 -->
<!-- npu="910b" id2 -->
- Atlas A2系列产品：支持
<!-- end id2 -->

## 函数功能

描述FlowNode的输出。

用户无需自己定义，调用FlowNode的\_\_call\_\_方法会返回FlowOutput或者Tuple\[FlowOutput\]。

## 函数原型

```python
FlowOutput(node, index)
```

## 参数说明

| 参数名称 | 数据类型 | 取值说明 |
| --- | --- | --- |
| node | FlowNode | FlowNode节点，具体请参见[dataflow.FlowNode](dataflow-FlowNode.md)。 |
| index | int | FlowNode节点的输出index |

## 返回值

正常场景下返回None。

## 调用示例

无

## 约束说明

无
