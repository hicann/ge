# \_\_call\_\_

## 产品支持情况

<!-- npu="A3" id1 -->
- Atlas A3系列产品：支持
<!-- end id1 -->
<!-- npu="910b" id2 -->
- Atlas A2系列产品：支持
<!-- end id2 -->

## 函数功能

调用FlowNode进行计算。

## 函数原型

```python
__call__(*inputs)
```

## 参数说明

| 参数名称 | 数据类型 | 取值说明 |
| --- | --- | --- |
| *inputs | Union[FlowData, FlowOutput] | 动态参数，类型为[FlowData](dataflow-FlowData.md)或者[FlowOutput](dataflow-FlowOutput.md)。 |

## 返回值

一个输出时返回FlowOutput对象，多个输出时返回FlowOutput元组。

异常情况如下。

| 异常信息 | 含义 |
| --- | --- |
| TypeError | 参数类型不正确 |
| ValueError | 参数值不正确 |

## 调用示例

```python
import dataflow as df
data = df.FlowData()
flow_node = df.FlowNode(...)
flow_output = flow_node(data)
```

## 约束说明

无
