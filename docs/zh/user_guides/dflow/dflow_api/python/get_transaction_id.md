# get\_transaction\_id

## 产品支持情况

<!-- npu="A3" id1 -->
- Atlas A3系列产品：支持
<!-- end id1 -->
<!-- npu="910b" id2 -->
- Atlas A2系列产品：支持
<!-- end id2 -->

## 函数功能

获取FlowMsg消息中的事务ID，事务ID从1开始计数，每feed一批数据，事务ID会加一，可用于识别哪一批数据。

## 函数原型

```python
get_transaction_id(self) -> int
```

## 参数说明

无

## 返回值

事务ID。

## 异常处理

无

## 约束说明

无
