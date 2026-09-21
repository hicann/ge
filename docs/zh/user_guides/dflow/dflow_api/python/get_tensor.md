# get\_tensor

## 产品支持情况

<!-- npu="A3" id1 -->
- Atlas A3系列产品：支持
<!-- end id1 -->
<!-- npu="910b" id2 -->
- Atlas A2系列产品：支持
<!-- end id2 -->

## 函数功能

获取FlowMsg中的tensor对象。

## 函数原型

```python
get_tensor() -> dataflow.Tensor
```

## 参数说明

无

## 返回值

返回dataflow.Tensor类型对象。

## 异常处理

无

## 约束说明

如果FlowMsg中是空，则tensor返回None。
