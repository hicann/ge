# data\_type

## 产品支持情况

全量芯片支持。

## 功能说明

获取或设置TensorDesc的DataType。

## 函数原型

```python
desc.data_type -> DataType
desc.data_type = data_type
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| data\_type | 输入 | Tensor的数据类型，类型为DataType。 |

## 返回值说明

获取时返回DataType；设置时无返回值。

## 约束说明

- `data_type`不是DataType类型时，抛出TypeError。
- `data_type`等于DataType.DT_MAX时，抛出ValueError。

## 调用示例

```python
data_type = desc.data_type
desc.data_type = DataType.DT_FLOAT16
```
