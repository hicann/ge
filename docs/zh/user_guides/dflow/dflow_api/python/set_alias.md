# set\_alias

## 产品支持情况

<!-- npu="A3" id1 -->
- Atlas A3系列产品：支持
<!-- end id1 -->
<!-- npu="910b" id2 -->
- Atlas A2系列产品：支持
<!-- end id2 -->

## 函数功能

设置节点别名，使用option:ge.experiment.data\_flow\_deploy\_info\_path指定节点部署位置时，flow\_node\_list字段可使用别名进行指定。

## 函数原型

```python
def set_alias(self, name) -> FlowNode
```

## 参数说明

| 参数名称 | 数据类型 | 取值说明 |
| --- | --- | --- |
| name | str | FlowNode的别名。 |

## 返回值

返回当前FlowNode。

## 调用示例

```python
import dataflow as df
flow_node = df.FlowNode(...)
flow_node.set_alias(name="hello")
```

## 约束说明

无
