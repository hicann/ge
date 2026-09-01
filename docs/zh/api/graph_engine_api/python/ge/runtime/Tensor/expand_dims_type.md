# expand\_dims\_type

## 产品支持情况

全量芯片支持。

## 功能说明

获取OriginShape转换为StorageShape时使用的维度扩展规则。

## 函数原型

```python
tensor.expand_dims_type -> ExpandDimsType
```

## 参数说明

无

## 返回值说明

返回当前Tensor Format中的`ExpandDimsType`对象。该对象由运行时提供，仅在当前执行回调期间有效。

## 约束说明

- 返回对象只能在当前执行回调期间使用。
- 该规则描述格式引入的扩展维度，不表示数据拷贝或内存分配操作。

## 调用示例

```python
expand_dims_type = x.expand_dims_type
```
