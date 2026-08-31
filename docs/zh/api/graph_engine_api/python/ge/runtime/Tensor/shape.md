# shape

## 产品支持情况

全量芯片支持。

## 功能说明

获取Tensor的Shape描述，其中同时保存OriginShape和StorageShape。

## 函数原型

```python
tensor.shape -> StorageShape
```

## 参数说明

无

## 返回值说明

返回当前Tensor的`StorageShape`对象。该对象由运行时提供，仅在当前执行回调期间有效。

## 约束说明

- 返回对象只能在当前执行回调期间使用。
- 返回对象的`origin_shape`表示原始逻辑形状，`storage_shape`表示运行时存储形状。

## 调用示例

```python
storage_shape = x.shape
```
