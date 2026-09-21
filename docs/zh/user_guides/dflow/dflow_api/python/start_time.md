# start\_time

## 产品支持情况

<!-- npu="A3" id1 -->
- Atlas A3系列产品：支持
<!-- end id1 -->
<!-- npu="910b" id2 -->
- Atlas A2系列产品：支持
<!-- end id2 -->

## 函数功能

以属性方式读取和设置FlowInfo的开始时间。

## 函数原型

```python
@property
def start_time(self)
@start_time.setter
def start_time(self, new_value)
```

## 参数说明

| 参数名称 | 数据类型 | 取值说明 |
| --- | --- | --- |
| new_value | int | 要设置的FlowInfo开始时间的值。 |

## 返回值

start\_time属性。

## 调用示例

```python
import dataflow as df
graph = df.FlowGraph(...)
flowinfo = FlowInfo(...)
flowinfo.start_time = 100
print(flowinfo.start_time)
```

## 约束说明

无
