# format

## 产品支持情况

全量芯片支持。

## 功能说明

获取Tensor的Format描述，其中同时保存OriginFormat、StorageFormat和维度扩展规则。

## 函数原型

```python
tensor.format -> StorageFormat
```

## 参数说明

无

## 返回值说明

返回当前Tensor的`StorageFormat`对象。该对象由运行时提供，仅在当前执行回调期间有效。

## 约束说明

- 返回对象只能在当前执行回调期间使用。
- 申请输出Tensor时，可将该对象传入`EagerOpExecutionContext.malloc_output_tensor`。

## 调用示例

```python
storage_format = x.format
```
