# to\_flow\_msg

## 产品支持情况

<!-- npu="A3" id1 -->
- Atlas A3系列产品：支持
<!-- end id1 -->
<!-- npu="910b" id2 -->
- Atlas A2系列产品：支持
<!-- end id2 -->

## 函数功能

将dataflow Tensor转换成FlowMsg。

## 函数原型

```python
to_flow_msg(self, tensor)
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| tensor | 输入 | 待转换的dataflow Tensor。 |

## 返回值

正常返回FlowMsg的实例。失败返回None。

## 异常处理

无

## 约束说明

无
