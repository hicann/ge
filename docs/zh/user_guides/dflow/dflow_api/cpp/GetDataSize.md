# GetDataSize

## 产品支持情况

<!-- npu="A3" id1 -->
- Atlas A3系列产品：支持
<!-- end id1 -->
<!-- npu="910b" id2 -->
- Atlas A2系列产品：支持
<!-- end id2 -->

## 函数功能

获取Tensor中的数据大小。

## 函数原型

```cpp
uint64_t *GetDataSize() const
```

## 参数说明

无。

## 返回值

返回Tensor的数据大小。

## 异常处理

无。

## 约束说明

修改返回的TensorDesc信息，不影响Tensor对象中已有的TensorDesc信息。
