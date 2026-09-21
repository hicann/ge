# fnode

## 产品支持情况

<!-- npu="A3" id1 -->
- Atlas A3系列产品：支持
<!-- end id1 -->
<!-- npu="910b" id2 -->
- Atlas A2系列产品：支持
<!-- end id2 -->

## 函数功能

根据当前的GraphProcessPoint生成一个FlowNode，返回一个FlowNode对象。

## 函数原型

```python
fnode(input_num, output_num, name=None)
```

## 参数说明

| 参数名称 | 数据类型 | 取值说明 |
| --- | --- | --- |
| input_num | int | 节点的输入个数。 |
| output_num | int | 节点的输出个数。 |
| name | str | 节点名称，框架会自动保证名称唯一，不设置时会自动生成FlowNode, FlowNode_1, FlowNode_2,...的名称。 |

## 返回值

返回FlowNode对象。

## 调用示例

```python
import dataflow as df
pp1 = df.GraphProcessPoint(...)
flow_node = pp1.fnode(input_num=2, output_num=1)
```

## 约束说明

无
