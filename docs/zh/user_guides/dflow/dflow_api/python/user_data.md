# user\_data

## 产品支持情况

<!-- npu="A3" id1 -->
- Atlas A3系列产品：支持
<!-- end id1 -->
<!-- npu="910b" id2 -->
- Atlas A2系列产品：支持
<!-- end id2 -->

## 函数功能

获取用户信息。

## 函数原型

```python
@property
def user_data(self)
```

## 参数说明

无

## 返回值

以属性方式返回user\_data对象。

## 调用示例

```python
import dataflow as df
graph = df.FlowGraph(...)
user_data_str = "UserData123"
result = graph.fetch_data() # 异步取结果
flowinfo = result[1]
fetch_user_data = flowinfo.user_data[0:len(user_data_str)]
name = fetch_user_data.decode('utf-8')
print(name)
```

## 约束说明

无
