# storage\_format

## 产品支持情况

全量芯片支持。

## 功能说明

获取Tensor的运行时存储Format。

## 函数原型

```python
tensor.storage_format -> Format
```

## 参数说明

无

## 返回值说明

返回表示实际内存布局的`Format`。

## 约束说明

返回值是Format枚举值的拷贝，可以在当前回调外保存；但不应据此延长Tensor的使用生命周期。

## 调用示例

```python
storage_format = x.storage_format
```
