# shape\_size

## 产品支持情况

全量芯片支持。

## 功能说明

获取Tensor的StorageShape所表示的元素个数。

## 函数原型

```python
tensor.shape_size -> int
```

## 参数说明

无

## 返回值说明

返回StorageShape的元素个数。

## 约束说明

该值根据StorageShape计算，不能用来替代`size`取得字节数。

## 调用示例

```python
element_count = x.shape_size
```
