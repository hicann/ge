# get\_msg\_type（UDF）

## 产品支持情况

<!-- npu="A3" id1 -->
- Atlas A3系列产品：支持
<!-- end id1 -->
<!-- npu="910b" id2 -->
- Atlas A2系列产品：支持
<!-- end id2 -->

## 函数功能

获取FlowMsg的消息类型。

## 函数原型

```python
get_msg_type(self)
```

## 参数说明

无

## 返回值

返回FlowMsg的消息类型。

```python
import dataflow.flow_func.flow_func as ff
# 消息返回如下两种value：
ff.MSG_TYPE_TENSOR_DATA
ff.MSG_TYPE_RAW_MSG
# 大于等于1024为用户自定义类型
```

## 异常处理

无

## 约束说明

无
