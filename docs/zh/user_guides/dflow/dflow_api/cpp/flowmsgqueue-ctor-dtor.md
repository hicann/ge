# FlowMsgQueue构造函数和析构函数

## 产品支持情况

<!-- npu="A3" id1 -->
- Atlas A3系列产品：支持
<!-- end id1 -->
<!-- npu="910b" id2 -->
- Atlas A2系列产品：支持
<!-- end id2 -->

## 函数功能

FlowMsgQueue的构造和析构函数。

## 函数原型

```cpp
FlowMsgQueue() = default
~FlowMsgQueue() = default
```

## 参数说明

无。

## 返回值

无。

## 异常处理

无。

## 约束说明

流式输入（即flow func函数入参为队列）场景下，框架不支持数据对齐和UDF主动上报异常。
