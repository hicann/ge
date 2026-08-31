# size

## 产品支持情况

全量芯片支持。

## 功能说明

获取Tensor数据占用的内存大小。

## 函数原型

```python
tensor.size -> int
```

## 参数说明

无

## 返回值说明

返回Tensor数据占用的字节数。

## 约束说明

返回值表示运行时存储空间大小，不一定等于OriginShape的元素个数乘以单个元素字节数。

## 调用示例

```python
byte_size = x.size
```
