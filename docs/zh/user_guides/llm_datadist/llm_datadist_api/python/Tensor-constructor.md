# Tensor-constructor

## 产品支持情况

<!-- npu="A3" id1 -->
- Atlas A3系列产品：支持
<!-- end id1 -->
<!-- npu="910b" id2 -->
- Atlas A2推理系列产品：支持
<!-- end id2 -->
<!-- npu="910b" id3 -->
- Atlas A2训练系列产品：不支持
<!-- end id3 -->

## 函数功能

构造Tensor。

## 函数原型

```python
__init__(data, tensor_desc: TensorDesc = None)
```

## 参数说明

| 参数名 | 数据类型 | 取值说明 |
| --- | --- | --- |
| data | Union[np.ndarray, Tensor] | 表示Tensor的数据。 |
| tensor_desc | [TensorDesc](TensorDesc-constructor.md) | 表示Tensor的描述信息。 |

## 调用示例

```python
from llm_datadist import Tensor
tensor = Tensor(numpy.array([1]))
```

## 返回值

正确情况下返回Tensor的实例。

传入data信息和tensor\_desc信息不匹配时，会抛出RuntimeError。

## 约束说明

无
