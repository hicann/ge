# SetBalanceScatter

## 产品支持情况

<!-- npu="A3" id1 -->
- Atlas A3系列产品：支持
<!-- end id1 -->
<!-- npu="910b" id2 -->
- Atlas A2系列产品：支持
<!-- end id2 -->

## 函数功能

设置节点balance scatter属性，具有balance scatter属性的UDF可以使用balance options设置负载均衡输出。

## 函数原型

```cpp
FlowNode &SetBalanceScatter()
```

## 参数说明

无。

## 返回值

返回设置完属性的FlowNode节点。

## 异常处理

无。

## 约束说明

与[SetBalanceGather](SetBalanceGather.md)接口互斥。
