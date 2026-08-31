# origin\_shape

## 产品支持情况

全量芯片支持。

## 功能说明

获取Tensor的原始逻辑Shape。

## 函数原型

```python
tensor.origin_shape -> Shape
```

## 参数说明

无

## 返回值说明

返回表示原始逻辑形状的`Shape`对象。该对象由运行时提供，仅在当前执行回调期间有效。

## 约束说明

- 返回对象只能在当前执行回调期间使用。
- 需要运行时实际内存布局时，应使用[`storage_shape`](storage_shape.md)。

## 调用示例

```python
origin_shape = x.origin_shape
```
