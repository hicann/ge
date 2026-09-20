# SetTransactionId（FlowMsg类）

## 产品支持情况

<!-- npu="A3" id1 -->
- Atlas A3系列产品：支持
<!-- end id1 -->
<!-- npu="910b" id2 -->
- Atlas A2系列产品：支持
<!-- end id2 -->

## 函数功能

设置FlowMsg消息中的事务ID，事务ID从1开始计数，可用于识别哪一批数据，设置为0时表示不使用自定义的transaction\_id，内部会采用自增的方式自动生成transaction\_id。

## 函数原型

```cpp
virtual void SetTransactionId(uint64_t transactionId)
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| transactionId | 输入 | 消息的事务ID |

## 返回值

无。

## 异常处理

无。

## 约束说明

无。
