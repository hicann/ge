# end\_time

## 产品支持情况

<!-- npu="A3" id1 -->
- Atlas A3系列产品：支持
<!-- end id1 -->
<!-- npu="910b" id2 -->
- Atlas A2系列产品：支持
<!-- end id2 -->

## 函数功能

以属性方法读取和设置FlowInfo的结束时间。

## 函数原型

```python
@property
def end_time(self)
@end_time.setter
def end_time(self, new_value)
```

## 参数说明

| 参数名称 | 数据类型 | 取值说明 |
| --- | --- | --- |
| new_value | int | 要设置的FlowInfo结束时间的值。 |

## 返回值

end\_time属性。

## 调用示例

```python
import dataflow as df
graph = df.FlowGraph(...)
flowinfo = FlowInfo(...)
flowinfo.end_time = 200
print(flowinfo.end_time)
```

## 约束说明

无
