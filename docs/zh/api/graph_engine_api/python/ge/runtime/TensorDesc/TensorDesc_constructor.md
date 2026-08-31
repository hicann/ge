# TensorDesc构造函数

## 产品支持情况

全量芯片支持。

## 功能说明

创建一个用于Python自定义算子元信息推导的`TensorDesc`。

## 函数原型

```python
TensorDesc(shape: Optional[Union[StorageShape, List[int]]], data_type: DataType)
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| shape | 输入 | Tensor的Shape，类型为`StorageShape`或`List[int]`。`None`表示标量。 |
| data_type | 输入 | Tensor的数据类型，类型为`DataType`。 |

## 返回值说明

TensorDesc对象。

## 约束说明

- `shape`不是`StorageShape`、整数列表或`None`时抛出`TypeError`。
- `data_type`不是`DataType`或等于`DataType.DT_MAX`时抛出`TypeError`或`ValueError`。

## 调用示例

```python
from ge.graph import DataType
from ge.runtime import TensorDesc

desc = TensorDesc([2, 3], DataType.DT_FLOAT)
```
