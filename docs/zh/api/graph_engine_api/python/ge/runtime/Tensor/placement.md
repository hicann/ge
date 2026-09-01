# placement

## 产品支持情况

全量芯片支持。

## 功能说明

获取Tensor数据的存储位置。

## 函数原型

```python
tensor.placement -> TensorPlacement
```

## 参数说明

无

## 返回值说明

返回`TensorPlacement`，用于表示数据位于Device HBM、Host、Following或Device P2P等位置。

## 约束说明

返回值为TensorPlacement枚举值的拷贝。

## 调用示例

```python
placement = x.placement
```
