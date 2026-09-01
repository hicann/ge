# shape

## 产品支持情况

全量芯片支持。

## 功能说明

获取或设置TensorDesc的StorageShape。

## 函数原型

```python
desc.shape -> StorageShape
desc.shape = shape
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| shape | 输入 | Tensor的Shape，类型为`StorageShape`或`List[int]`。 |

## 返回值说明

获取时返回StorageShape；设置时无返回值。

## 约束说明

- `shape`不是StorageShape、整数列表或None时，抛出TypeError。
- 返回的StorageShape隶属于当前TensorDesc对象，TensorDesc对象销毁后不可继续使用。

## 调用示例

```python
shape = desc.shape
desc.shape = [2, 3]
```
