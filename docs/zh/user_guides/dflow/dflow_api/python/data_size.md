# data\_size

## 产品支持情况

<!-- npu="A3" id1 -->
- Atlas A3系列产品：支持
<!-- end id1 -->
<!-- npu="910b" id2 -->
- Atlas A2系列产品：支持
<!-- end id2 -->

## 函数功能

获取user\_data的长度。

## 函数原型

```python
@property
def data_size(self)
```

## 参数说明

无

## 返回值

返回user\_data的长度。

## 调用示例

```python
import dataflow as df
graph = df.FlowGraph(...)
user_data_str = "UserData123"
result = graph.fetch_data() # 异步取结果
flowinfo = result[1]
user_data_size = flowinfo.data_size
print(user_data_size)
```

## 约束说明

无
