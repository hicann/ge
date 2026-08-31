# storage\_shape

## 产品支持情况

全量芯片支持。

## 功能说明

获取Tensor的运行时存储Shape。

## 函数原型

```python
tensor.storage_shape -> Shape
```

## 参数说明

无

## 返回值说明

返回表示实际存储形状的`Shape`对象。该对象由运行时提供，仅在当前执行回调期间有效。

## 约束说明

- 返回对象只能在当前执行回调期间使用。
- `storage_shape`可能因格式转换或对齐而与[`origin_shape`](origin_shape.md)不同。

## 调用示例

```python
storage_shape = x.storage_shape
```
