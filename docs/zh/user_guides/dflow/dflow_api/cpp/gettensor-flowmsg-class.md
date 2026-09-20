# GetTensor（FlowMsg类）

## 产品支持情况

<!-- npu="A3" id1 -->
- Atlas A3系列产品：支持
<!-- end id1 -->
<!-- npu="910b" id2 -->
- Atlas A2系列产品：支持
<!-- end id2 -->

## 函数功能

获取FlowMsg中的Tensor指针。

## 函数原型

```cpp
Tensor *GetTensor() const
```

## 参数说明

无

## 返回值

返回Tensor类型指针。

## 异常处理

无。

## 约束说明

只有消息类型为TENSOR\_DATA\_TYPE时，才能获取Tensor类型指针，否则返回NULL。
